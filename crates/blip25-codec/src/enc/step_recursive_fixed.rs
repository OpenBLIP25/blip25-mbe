//! The DOMINANT `STEP` (`array[0xc]`, `band_decompress`'s `step` argument)
//! write mechanism: a per-frame RECURSIVE update law (`new_step = f(old_step)`),
//! **not** a `COUNT`-indexed table lookup.
//!
//! ## The mechanism
//!
//! The per-frame caller dispatches on a mode/context check to exactly ONE of two
//! writer functions per frame: a rarely-taken literal writer or the recursive
//! writer. Across 300 captured frames on 3 files the dispatch picks the
//! recursive writer on EVERY frame; the literal writer is never observed to fire
//! on this corpus (its own STEP write is a narrow conditional literal,
//! `STEP<-0x563b`, gated on 3 separate flags).
//!
//! The recursive writer branches on the same accept/reject gate primitive the
//! b0-gate work already closed:
//! - **ACCEPT** (~3-10% of frames): hardcodes `STEP<-4217` (`0x1079`),
//!   `COUNT<-56` (`0x38`) as LITERAL constants -- a "reset to steady state"
//!   case, not a computed value.
//! - **REJECT** (~90-97% of frames, THE dominant path): calls the step update
//!   helper `(step_ptr, count_ptr, mode, K=[array+0xa], arg4)`, which dispatches
//!   on `mode`:
//!   - `mode==0`: no-op (returns without writing either output).
//!   - `mode==2`: hardcodes `STEP<-0x4027` (16423), `COUNT<-15` -- another
//!     literal reset case.
//!   - `mode==1` (the ONLY mode observed in all 284 REJECT captures, with `K` a
//!     FIXED `7` in every one): the recursive formula this module ports.
//!
//! ## The `mode==1` formula
//!
//! A LOG-DOMAIN recursive step-size update, structurally an ADPCM-style adaptive
//! step-size law rather than a lookup. It takes the OLD `STEP`, converts it to a
//! Q12 log2 representation ([`band_decompress::log2_fn`], the same primitive the
//! `gamma_ref_exact` gain chain uses), evaluates a degree-3 Horner polynomial
//! (its own coefficients, distinct from the log2 table), re-quantizes via 2
//! calls to [`trunc_div16`] with 2 fixed scale constants (`0x54a4`=21668,
//! `TWO_SCALE1`=240 for `K=7`), clamps to `[4217, 27594]`, and finally derives a
//! matching `COUNT` from the new `STEP` via an approximately-inverse
//! `243100/STEP` relation clamped to `[9, 56]`.
//!
//! **`COUNT` is DERIVED FROM `STEP`, not the other way around.** Treating
//! `COUNT` as the independent variable and `STEP` as a `COUNT`-keyed lookup is
//! backwards.
//!
//! Pinned bit-exact against captured `(step_before, step_after, count_after)`
//! triples on 3 files -- every REJECT-FORMULA frame, not a subset.
//!
//! ## Scope -- what this does NOT close
//!
//! This implements ONLY the `mode==1` recursive law plus the ACCEPT-LITERAL
//! constants (`4217`/`56`). It does **not** implement:
//! - The accept/reject GATE decision itself: re-deriving its exact args for the
//!   STEP call site is future work. The experimental path below approximates the
//!   gate as "always REJECT-FORMULA" -- the ~90-97% majority behavior, not a
//!   verified per-frame decision.
//! - `mode==0` (no-op) or `mode==2` (literal `16423`/`15`), never observed in
//!   the 300-frame capture. `K` (fixed at `7` in every observation) is likewise
//!   not verified to be a true per-instance constant rather than merely constant
//!   across this corpus.
//! - the literal writer's own write, never observed to fire.
//!
//! The reference's recursion depends on the REAL previous-frame `STEP`, not on an
//! independently-estimated `L`/`COUNT`, so the experimental path threads a
//! SELF-REFERENTIAL `STEP` state frame to frame, seeded at `8192` (the observed
//! frame-0 `step_before` on all 3 test files). Same approximation pattern as
//! `next_bias_raw`'s IIR bias recursion: structurally the right recursion,
//! bit-exact GIVEN the real previous `STEP`, but not verified to stay locked to
//! the reference's trajectory over many frames without observing the real previous
//! value.

use super::band_decompress;

// `trunc_div16` / `sar32` / `shl32` and `count_from_step` moved to
// `crate::shared::step_count` so the decoder's site-A pitch path can reach
// `count_from_step` without `enc/`. Re-imported here so this module's other
// stages (which use the divide/shift helpers) are unchanged.
use crate::shared::step_count::{count_from_step, sar32, shl32, trunc_div16};

/// Real x86 `sar r16,cl` semantics: the shift count is masked to 5 bits
/// REGARDLESS of the 16-bit operand size, so counts >=16 (up to the 31 the
/// mask allows) collapse to pure sign-extension (`0` or `-1`), not a
/// no-op or a wrap. [`step_horner_poly`]'s own default (unadjusted) shift count is
/// `-4` (`0xFFFFFFFC`), whose low byte `0xFC` masks to `28`. This matters for
/// non-`mode==1` completeness: every capture on record takes the "adjusted"
/// branch (see [`step_horner_poly`]) with `cl` in a small, sane range.
fn sar16_fixed(val: i16, count: i32) -> i16 {
    let masked = (count & 0x1F) as u32;
    if masked == 0 {
        val
    } else if masked >= 16 {
        if val < 0 {
            -1
        } else {
            0
        }
    } else {
        val >> masked
    }
}

// `count_from_step` now lives in `crate::shared::step_count` (re-imported above).

/// Given the log-domain candidate `raw_value`
/// (the final `poly_in` value [`step_update_mode1`] computes),
/// evaluates a degree-3 Horner polynomial (Q12 fixed point, coefficients
/// `0x13B/0x71B`, `0x1EC0`, `0x58B9`, `-0x8000` -- distinct from the log2
/// polynomial's own 6-term table), clamps to `[4217, 27594]`, and derives
/// the paired `COUNT` via [`count_from_step`]. Returns `(new_step,
/// new_count)`.
fn step_horner_poly(raw_value: i32) -> (i16, i16) {
    let raw16 = raw_value as i16;
    let mut work = raw_value;
    let mut sh: i32 = -4; // default shift count (never exercised by any capture
                          // on record -- see sar16_fixed's doc)
    if raw16 < -4096 {
        let neg4097: i16 = 0xEFFFu16 as i16; // -4097
        let sub_result = neg4097.wrapping_sub(raw16) as u16; // "sub ax,dx" (16-bit wrap)
        let shr_val = sub_result >> 12; // logical shift (shr, not sar)
        let adj = shr_val.wrapping_add(1);
        work = work.wrapping_add((adj as i32) << 12);
        sh = adj as i32 - 4;
    }
    let x = (work as i16) as i32; // movsx edx,dx

    let cwde16 = |v: i32| -> i32 { (v as i16) as i32 };

    let mut acc = ((x * 0x13B) >> 12) + 0x71B;
    acc = cwde16(acc);
    acc = ((acc * x) >> 12) + 0x1EC0;
    acc = cwde16(acc);
    acc = ((acc * x) >> 12) + 0x58B9;
    acc = cwde16(acc);
    acc = ((acc * x) >> 12) - 0x8000;

    let poly16 = acc as i16;
    let shifted = sar16_fixed(poly16, sh);
    let mut poly_result = shifted;

    // `arg4==0` in EVERY capture on record, so the conditional upper clamp
    // (`cmove`, gated on `arg4==0` in the asm) is unconditional for all verified
    // data. Kept as a named condition rather than hardcoded, in case a capture
    // ever finds arg4!=0.
    const ARG4_OBSERVED_ALWAYS_ZERO: bool = true;
    if poly_result < 0x1079 {
        poly_result = 0x1079; // 4217, lower clamp
    } else if poly_result > 0x6bca && ARG4_OBSERVED_ALWAYS_ZERO {
        poly_result = 0x6bca; // 27594, upper clamp
    }

    let new_step = poly_result;
    let new_count = count_from_step(new_step);
    (new_step, new_count)
}

/// The `mode==1` path -- the only mode observed in any REJECT-FORMULA capture:
/// the recursive `STEP` update law. `old_step` is the reference's PREVIOUS `STEP`
/// (`array[0xc]` before this frame's write); `k` is `[array+0xa]`, a FIXED `7`
/// in every observation and not verified to vary. Returns
/// `(new_step, new_count)`, both bit-exact against captured ground truth.
pub(crate) fn step_update_mode1(old_step: i16, k: i32) -> (i16, i16) {
    let scale1 = 60u16.wrapping_shl(((k - 6) & 0x1f) as u32);
    let two_scale1 = scale1.wrapping_mul(2);

    let log2_old = band_decompress::log2_fn(old_step, -4);
    let neg = if log2_old == 0x8000_0000 {
        0x7FFF_FFFFu32
    } else {
        (-(log2_old as i32)) as u32
    };
    let shl12 = shl32(neg, 0xc);
    let sar16v = sar32(shl12, 0x10);
    let cwde_val = (sar16v as u16 as i16) as i32;
    let biased = cwde_val - 0x44FD;
    let log_step_16 = (biased as u16 as i16) as i32; // movsx ecx,ax

    let scale_signed = (two_scale1 as i16) as i32; // movsx eax,bx
    let product = log_step_16.wrapping_mul(scale_signed);

    let cand1 = trunc_div16(product, 0x54a4);

    let mut bounded: u16 = cand1 as u16;
    if cand1 < 0 {
        bounded = 0;
    } else if (bounded as i16) >= (scale1 as i16) {
        bounded = scale1.wrapping_sub(1);
    }

    let val_full = (bounded as u32).wrapping_mul(2).wrapping_add(1);
    let val16 = (val_full as u16 as i16) as i32; // cwde
    let product2 = val16.wrapping_mul(0x2a52);

    let cand2 = trunc_div16(product2, two_scale1 as i32);
    let result = cand2;

    let result_hi = shl32(result as u16 as u32, 16);
    let base_const: u32 = 0xbb03_0000;
    let diff = base_const.wrapping_sub(result_hi);
    let poly_in = (diff as i32) >> 16; // sar edx,0x10 (arithmetic)

    step_horner_poly(poly_in)
}

/// The recursive writer's accept-gate branch: `STEP<-4217`, `COUNT<-56`
/// unconditionally, whenever the (not-independently-ported) accept/reject gate
/// accepts.
pub(crate) const ACCEPT_LITERAL_STEP: i16 = 4217;
pub(crate) const ACCEPT_LITERAL_COUNT: i16 = 56;

/// Observed frame-0 `step_before` seed (`0x2000`=8192), identical across all 3
/// test files -- see this module's top-level doc, "Scope".
pub(crate) const INITIAL_STEP_SEED: i16 = 8192;

/// Combines [`step_update_mode1`] (REJECT path) with the ACCEPT-LITERAL
/// constants under an explicit `accept` decision -- [`step_gate_accept`], or a
/// caller-supplied reference/fallback for content with no gate capture. Returns
/// `(new_step, new_count)`.
pub(crate) fn step_update_with_gate(old_step: i16, k: i32, accept: bool) -> (i16, i16) {
    if accept {
        (ACCEPT_LITERAL_STEP, ACCEPT_LITERAL_COUNT)
    } else {
        step_update_mode1(old_step, k)
    }
}

/// Captured per-frame ACCEPT/REJECT gate decisions for 3 test files, in
/// per-file frame order: the `real_accept` boolean, one entry per frame
/// (`mark`/`cpvbad` = 200 frames each, `dtone_10` = 125).
///
/// **This is an REFERENCE, not a computable mechanism.** `Q`/`P`'s source -- what
/// feeds the accept/reject decision from audio -- is not traced (see
/// [`step_gate_accept`]). It exists to answer one question: GIVEN the gate's
/// real decision sequence, how much of `STEPFORMULA`'s remaining `b2` gap does
/// the gate explain versus some other open cause? See
/// [`super::Encoder::b2_x86_stepformula_gateref_log`] for where it is
/// consumed.
pub mod gate_ref {
    include!("step_gate_ref_seqs.rs.inc");
}
