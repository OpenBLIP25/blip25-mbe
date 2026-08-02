//! The encoder's "main loop": the 3-tap boundary-clamped smoother +
//! conditional `0x4000` silence default + divide that composes the
//! block-consolidated P/Q arrays ([`super::pq_builder`] +
//! [`super::block_consolidate`]) into `oldgen_pre`/`exps_pre`, the raw input to
//! the FINAL [`super::block_consolidate::block_consolidate`] call that produces
//! `oldgen`/`s_new`. `s_new` is this loop's return value, i.e. that final
//! call's shared exponent.
//!
//! ## The exponent bookkeeping
//!
//! It reads like a running accumulator and is not one:
//! - `exp_P` (the P-array's block-consolidated shared exponent, the tag=0
//!   consolidate call's return value) is stored ONCE, before the loop, at a
//!   FIXED stack slot.
//! - `exp_Q` (tag=1's return value) is stored ONCE at its own fixed slot.
//! - At the END of every iteration, the working exponent is reloaded FRESH from
//!   the `exp_P` slot. Nothing in the loop body ever writes back to that slot,
//!   so it reloads the same constant each time.
//! - WITHIN one iteration, that freshly-reloaded `exp_P` is locally reduced by
//!   the iteration's own P-sum normalize shift to become `exp_seed`; `exp_Q` is
//!   locally reduced by the iteration's own Q-sum normalize shift to become
//!   `exp_adjust`. Both feed [`super::atan2_bfp_divide::bfp_divide`]. This is
//!   the "true exponent = block-consolidated exponent minus this call's local
//!   normalize shift" idiom used throughout this codec, not a novel mechanism.
//!
//! ## The formula
//!
//! For each `i` in `0..8`:
//! 1. **3-tap boundary-clamped sum**: `p_sum = P[i-1] + P[i] + P[i+1]`,
//!    `q_sum = Q[i-1] + Q[i] + Q[i+1]`, skipping the `i-1` term when `i==0` and
//!    the `i+1` term when `i==7`.
//! 2. **Normalize both sums** -- a plain 32-bit `bsr`-based leading-bit count,
//!    distinct from the 64-bit [`super::band_decompress::normalize64`] the P/Q
//!    builders' internals use, since these sums fit in 32 bits:
//!    `shift = 30 - bsr(v if v>=0 else !v)` (or `0` if `v==0`);
//!    `v_normalized = v << shift`.
//! 3. **Silence default**: if the normalized `q_sum` is exactly `0`,
//!    `oldgen_pre[i] = 0x4000`, `exps_pre[i] = 1`.
//! 4. **Otherwise, divide**: `bfp_divide(x = p_sum_normalized (full 32-bit),
//!    exp_seed = exp_P - p_shift, y = q_sum_normalized >> 16 (top 16 bits,
//!    arithmetic), exp_adjust = exp_Q - q_shift)` -> `(mantissa, exponent)`,
//!    written directly to `oldgen_pre[i]`/`exps_pre[i]`.
//!
//! ## Coverage caveat
//!
//! Real captures never produce a negative 3-tap `p_sum`/`q_sum` -- both are
//! energy/correlation-like quantities, non-negative in every captured row. The
//! negative-sum paths (the `bsr(!v)` branch, the `sign_xor`-driven negation
//! inside [`super::atan2_bfp_divide::bfp_divide`]) are covered only by synthetic
//! CPU-emulation cross-checks, not by real audio.
//!
//! ## What this closes and what remains open
//!
//! **Closes**: the main loop itself, completing the
//! `(raw P/Q window content) -> P/Q builders -> block-consolidate -> main loop
//! -> final block-consolidate -> (oldgen[8], s_new)` chain bit-exact, GIVEN real
//! P/Q pre-pass builder inputs. See [`oldgen_and_s_new`] for the composed chain.
//!
//! **Does NOT close**: what feeds the main loop's own incoming args -- the raw
//! correlation/window array source upstream of the P/Q builders -- or array B's
//! separate formula. `Encoder::attempt_b1_fixed` therefore still cannot use real
//! computed arrays end-to-end from raw audio and remains fed
//! `B1_X86_PLACEHOLDER_ABC`.

use super::atan2_bfp_divide::bfp_divide;
use super::block_consolidate::block_consolidate;

/// The main loop's 32-bit normalize, distinct from the 64-bit `normalize64` the
/// P/Q builders' internals use (see module doc point 2).
#[inline]
fn normalize32(v: i32) -> i32 {
    if v == 0 {
        return 0;
    }
    let src: u32 = if v >= 0 { v as u32 } else { !(v as u32) };
    let highbit = 31 - src.leading_zeros() as i32;
    (30 - highbit) & 0xFFFF
}

#[inline]
fn shl32(v: i32, shift: i32) -> i32 {
    let sh = (shift & 0xFF) & 0x1F;
    ((v as i64) << sh) as i32
}

#[inline]
fn sar32(v: i32, shift: i32) -> i32 {
    v >> (shift & 0x1F)
}

/// One iteration (`i` in `0..8`) of the main loop.
/// `p`/`q` are the 8-word block-consolidated arrays (already at
/// `exp_p`/`exp_q` shared scale, i.e. [`super::block_consolidate::block_consolidate`]'s
/// own `dest` output for the P-array/Q-array call sites). Returns
/// `(oldgen_pre_i, exps_pre_i)`.
pub(crate) fn main_loop_word(
    p: &[i16; 8],
    exp_p: i16,
    q: &[i16; 8],
    exp_q: i16,
    i: usize,
) -> (i16, i16) {
    let mut p_sum = p[i] as i32;
    let mut q_sum = q[i] as i32;
    if i > 0 {
        p_sum += p[i - 1] as i32;
        q_sum += q[i - 1] as i32;
    }
    if i < 7 {
        p_sum += p[i + 1] as i32;
        q_sum += q[i + 1] as i32;
    }

    let p_shift = normalize32(p_sum);
    let exp_seed = (exp_p as i32 - p_shift) as i16; // truncate to low 16 bits
    let p_norm = shl32(p_sum, p_shift);

    let q_shift = normalize32(q_sum);
    let q_norm = shl32(q_sum, q_shift);

    if q_norm == 0 {
        return (0x4000, 1);
    }

    let y = sar32(q_norm, 16);
    let exp_adjust = (exp_q as i32 - q_shift) as i16;
    match bfp_divide(p_norm, exp_seed as i32, y, exp_adjust as i32) {
        Some((mantissa, exponent)) => (mantissa, exponent),
        None => (0, 0), // real DLL would divide-by-zero; never observed for real audio
    }
}

/// All 8 words of the main loop, live-verified 1600/1600
/// exact. Returns `(oldgen_pre[8],
/// exps_pre[8])` -- the raw per-word mantissa/exponent pairs immediately
/// before the FINAL [`block_consolidate`] call.
pub(crate) fn main_loop(
    p: &[i16; 8],
    exp_p: i16,
    q: &[i16; 8],
    exp_q: i16,
) -> ([i16; 8], [i16; 8]) {
    let mut oldgen_pre = [0i16; 8];
    let mut exps_pre = [0i16; 8];
    for i in 0..8 {
        let (m, e) = main_loop_word(p, exp_p, q, exp_q, i);
        oldgen_pre[i] = m;
        exps_pre[i] = e;
    }
    (oldgen_pre, exps_pre)
}

/// The FULL composed chain: block-consolidated `(P, exp_P)`/`(Q, exp_Q)`
/// -> [`main_loop`] -> final [`block_consolidate`] -> `(oldgen[8],
/// s_new)`. `s_new` is the main loop's return value: the caller's body does
/// nothing to it between the main-loop call and its own return, so `s_new` IS
/// this final shared exponent, not an independently computed quantity --
/// `s_new` and `oldgen` are a matched mantissa-array / shared-exponent pair.
pub(crate) fn oldgen_and_s_new(
    p: &[i16; 8],
    exp_p: i16,
    q: &[i16; 8],
    exp_q: i16,
) -> ([i16; 8], i16) {
    let (oldgen_pre, exps_pre) = main_loop(p, exp_p, q, exp_q);
    let mut oldgen = [0i16; 8];
    let s_new = block_consolidate(&oldgen_pre, &exps_pre, &mut oldgen);
    (oldgen, s_new)
}
