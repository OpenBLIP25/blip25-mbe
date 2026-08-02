//! `T`'s conditional 8-iteration "fresh-insert" loop -- the inner loop of the
//! same update function that closes `S`.
//!
//! ## The gate
//!
//! `T` (`ebp_callee+0x28..+0x36`, 8 words) is conditionally re-derived on every
//! call, behind the same 4-condition confidence check that guards `S`'s
//! unconditional write: `ebp+0x2>0`, `ebp+0x0<0`, `ZX(ebp+0x12)>=ZX(ebp+0x14)`,
//! and -- only when the 3rd comparands are EQUAL -- an extra `ebp+0xc>=ebp+0xe`
//! recheck. See `t_ring_gate_passed`.
//!
//! ## The loop
//!
//! When the gate passes, an 8-iteration loop runs (the counter is hardcoded to
//! `8`). Per iteration `i` (`0..8`), reading `oldgen[i]` (`ebp+0x18+2*i`, the
//! "older" neighboring generation slot) and `T[i]` itself (`ebp+0x28+2*i`, the
//! CURRENT, not-yet-overwritten value):
//!
//! - `k0 = max(s_new, old_4a) + 1` (16-bit wrapping; `s_new` is the S producer's
//!   return value, still live in `dx` at this point -- nothing between the
//!   `movzx edx,ax` and here touches `edx`; `old_4a` is the `ebp+0x4a`
//!   self-referential slot).
//! - `term1 = rescale(oldgen[i] << 16, shift = s_new - k0)`
//! - `term2 = rescale(T[i] << 16, shift = old_4a - k0)`
//! - if `term1 - term2 < 0` (32-bit signed): `T[i] :=
//!   trunc16(rescale(oldgen[i] << 16, shift = s_new - old_4a) >> 16)` -- a
//!   re-blend of `oldgen[i]` at a freshly-computed relative scale factor.
//! - else: `T[i]` is left **completely unchanged** for this iteration. The loop
//!   writes AT MOST one fresh word per iteration, not exactly one.
//!
//! `rescale(value, shift)` is the "shl by a masked, non-saturating count when
//! `shift>=0`; arithmetic-shift-right by `min(-shift, 31)` when `shift<0`"
//! idiom, also present in [`super::atan2_chain`]'s `shl_sar_sat`. It is
//! reimplemented locally as [`t_shift`] rather than shared, to keep this module
//! free of a cross-module dependency for one helper.
//!
//! ## Entry-time generation shift
//!
//! `T` also changes on gate-SKIPPED calls, because the update function's ENTRY
//! -- well before the loop above -- runs an unconditional 2-iteration
//! "generation shift": the block-move primitive `(dest=edi, src=edi-0x10,
//! count=8)` over the 8-word blocks, in parallel with the 1-word `S` shift at
//! `ebp+0x48`/`+0x4a`/`+0x4c`. It writes into `T`'s slot (`ebp+0x28`) before the
//! conditional loop ever runs.
//!
//! It is a plain, unconditional 3-generation ring shift entirely WITHIN the same
//! call -- **not** "one call back", a reading that scores 0/52 against capture.
//! Per call:
//! - `T_post == oldgen_pre` (i.e. [`t_ring_entry_shift`]'s `t_out`)
//! - `T2_post (ebp+0x38) == T_pre (ebp+0x28, this call's pre-shift value)` -- a
//!   genuine 3rd generation slot
//! - `s4c_post (ebp+0x4c) == s4a_pre (ebp+0x4a)`
//! - `s4a_post (ebp+0x4a) == s48_pre (ebp+0x48)`
//! - `s48_post (ebp+0x48) == s48_pre` (untouched by the shift)
//!
//! `ebp_callee` is a PERSISTENT caller-owned pointer into the caller's long-lived
//! state, not this call's fresh stack. That is why `oldgen` carries real content
//! INTO the call without this function ever writing it: it is a ring buffer that
//! persists call-to-call, and this loop is its age/decay step.
//!
//! ## `oldgen`'s writer
//!
//! `oldgen` (`ebp_callee+0x18`) is not written anywhere inside the update
//! function's body. Its writer sits at the end of the S producer's tail: the
//! update function calls the S producer (the same `S`/`s_new` producer that
//! supplies `s_new`), whose tail calls the rotation transform and then the
//! oldgen writer, in that order.
//!
//! The oldgen writer's structure: a pre-pass loop (calling the two P/Q array
//! builders) builds two internal 8-word sum arrays ("P"/"Q"); a main loop (8
//! iterations) applies a 3-tap boundary-clamped smoother to P and Q at each
//! index, block-floating-point normalizes both sums (the `bsr`-based
//! leading-zero-count idiom used throughout this codec), and either writes a
//! constant `0x4000` mantissa (when the normalized Q sum is zero) or calls the
//! divide primitive -- the same one ported as
//! [`super::atan2_bfp_divide::bfp_divide`] -- to compute a data-dependent
//! mantissa, written directly into `oldgen[i]`. After all 8 words are written, a
//! call to the block-consolidate routine rescales the 8 raw mantissas to ONE
//! shared block-float exponent, writing the final `oldgen[0..8]` in place and a
//! companion exponent word to `ebp_callee+0x12`, and returns that exponent in
//! `ax`.
//!
//! The S producer's `ret` follows its `call` to the oldgen writer with only
//! `add esp,...`/`pop esi`/`pop ebp`/`add esp,...`/`ret` -- none of which touch
//! `eax`/`ax`. So the S producer's return value (`S`/`s_new`, used throughout
//! this module's gate/blend formulas) IS the oldgen writer's return value:
//! **`s_new` and `oldgen` are a matched mantissa-array / shared-exponent pair,
//! produced by the same final computation.**
//!
//! ## What is still open
//!
//! `oldgen`'s (and therefore `s_new`'s) numeric formula is not closed end-to-end
//! from raw audio. The pre-pass P/Q array builders are deep (loops, min/max
//! clamping, accumulation -- comparable in scope to the whole atan2-core
//! closure) and untraced. The rotation transform feeding one of the oldgen
//! writer's arguments is identified but not formula-closed. The
//! block-consolidate routine is structurally understood -- analogous to, but a
//! DIFFERENT function from, [`super::block_exponent`]'s routine -- but not
//! verified or ported. Array B's formula is separately open.

/// "Shift left by a signed count, non-saturating; arithmetic right-shift,
/// clamped to a max magnitude of 31, for negative counts" -- the `test cx,cx;
/// js ...; shl edx,cl` / `cmp cx,0xffe1; jg ...; sar edx,cl` / `sar edx,0x1f`
/// idiom from the update function. Semantically identical to
/// [`super::atan2_chain`]'s private `shl_sar_sat`.
fn t_shift(value: i32, count: i16) -> i32 {
    let count = count as i32;
    if count >= 0 {
        let sh = (count as u32) & 31;
        value.wrapping_shl(sh)
    } else {
        let mag = (-count) as u32;
        if mag < 31 {
            value >> mag
        } else {
            value >> 31
        }
    }
}

/// `T`'s conditional per-iteration re-blend, given the gate has already passed
/// (see `t_ring_gate_passed`). `oldgen`/`t_pre` are `T`'s two upstream 8-word
/// neighbors at the moment this loop is about to run (`ebp+0x18..+0x26` /
/// `ebp+0x28..+0x36`), `s_new` is the S producer's return value for this call,
/// and `old_4a` is the `ebp+0x4a` self-referential slot. Returns `T`'s new
/// 8-word content.
pub(crate) fn t_ring_update(
    oldgen: [i16; 8],
    t_pre: [i16; 8],
    s_new: i16,
    old_4a: i16,
) -> [i16; 8] {
    let k0 = s_new.max(old_4a).wrapping_add(1);
    let shift1 = s_new.wrapping_sub(k0);
    let shift2 = old_4a.wrapping_sub(k0);
    let shift_blend = s_new.wrapping_sub(old_4a);

    let mut out = [0i16; 8];
    for i in 0..8 {
        let term1 = t_shift((oldgen[i] as i32) << 16, shift1);
        let term2 = t_shift((t_pre[i] as i32) << 16, shift2);
        let diff = term1.wrapping_sub(term2);
        out[i] = if diff < 0 {
            let blend = t_shift((oldgen[i] as i32) << 16, shift_blend);
            (blend >> 16) as i16
        } else {
            t_pre[i]
        };
    }
    out
}

/// The repair gate's `f0` reference (`ctx[0]`), PCM-derivable end-to-end.
///
/// The gate field `f0 = ctx[0]`, written inside the S producer (the `S`/`s_new`
/// producer the update function calls). It is the `atan2` output of the same
/// `windowed_complex_correlation` result the gate tail consumes, scaled by the fixed constants the
/// disassembly spells out:
///
/// ```text
/// call windowed_complex_correlation        ; (out0,out1,R) = windowed_complex_correlation(...)
/// movzx eax,ax        ; R
/// call atan2          ; (mant,exp) = atan2(di=out1, corr_exp=R, cx=out0)
/// movsx edx,ax        ; edx = mant
/// mov eax,[arg6]      ; eax = -20860 (0xffffae84)
/// imul edx,eax        ; edx = mant * -20860
/// add ax,[arg7=6]     ; exp + 6
/// add edx,edx         ; edx *= 2
/// sub cx,7            ; cx = exp + 6 - 7 = exp - 1
/// shl/sar edx,cl      ; edx = shl_sar_sat(edx, exp-1)
/// sar edx,0x10        ; edx >>= 16
/// mov [ebp+0x0],dx    ; ctx[0] = f0
/// ```
///
/// `atan2_dispatch` reads its `corr_exp` argument sign-extended, so `R` must be
/// passed as a signed `i16` to match
/// [`super::atan2_chain::atan2_dispatch`].
///
/// Given captured `windowed_complex_correlation` outputs this reproduces the captured gate firing
/// set exactly. Given PCM-reconstructed `windowed_complex_correlation` outputs it fires on 46/47
/// voiced frames with 4 extras; that residual is the upstream `windowed_complex_correlation` `R±1`
/// error, not this arithmetic -- do not "fix" it here.
pub(crate) fn repair_gate_f0(out0: i16, out1: i16, r: i16) -> i16 {
    let (mant, exp) = crate::enc::atan2_chain::atan2_dispatch(out1 as i32, r as i32, out0 as i32);
    let mut acc: i32 = (mant as i32).wrapping_mul(-20860);
    acc = acc.wrapping_add(acc); // *2 (double)
    let count = exp.wrapping_add(6).wrapping_sub(7); // exp + 6 - 7 = exp - 1
    acc = t_shift(acc, count);
    acc >>= 16; // arithmetic shift right by 16
    acc as i16
}
