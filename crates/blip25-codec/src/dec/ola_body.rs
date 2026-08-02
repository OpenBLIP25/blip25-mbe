//! Voiced harmonic OLA **body**: the sample-producing path of the voiced OLA
//! synthesis, and the one leaf under it (`bfp_divide_round_i16`).
//!
//! [`crate::dec::ola_driver`] ports the body's *state machine* only -- the
//! three decoder words it threads across subframes. `dec::ola_driver::step` has
//! **no output-sample parameter**, so it cannot emit PCM even in principle, and
//! its `place` arg is a closure stand-in for the harmonic-placement leaf.
//! [`voiced_ola_body`] is the other half, and its signature is the point:
//!
//! ```text
//!   pub fn voiced_ola_body(out: &mut [i32], ...)
//!                         ^^^^^^^^^^^^^^^ the emitter
//! ```
//!
//! # Structure
//!
//! The body is 1277 B with two `ret` sites. It takes **6 args**, not 5 --
//! fixed at its sole call site, which pushes six dwords:
//!
//! ```text
//!   a1 = &out[..]          (caller: lea eax,[esp+0x40])  -- the PCM accumulator
//!   a2 = n                 (the subframe advance; 80 in the shipped decoder)
//!   a3 = &state            (caller: lea eax,[ebx+0x18])
//!   a4 = &frame params     (caller's edi)
//!   a5 = &frame params 2   (caller's ebp)
//!   a6 = movzx [ebx+0x7be] -- forwarded to bfp_divide_round_i16, nothing else
//! ```
//!
//! ## The local accumulator
//!
//! The frame is `sub esp,0x2ec`; the four `push`es come **after** it, so the
//! saved registers sit at the BOTTOM (`B+0x00..B+0x10`) and the 748-byte local
//! area is `B+0x10..B+0x2fc`. Slots `B+0x10..B+0x5c` are scalars (this module's
//! comments name them with the same `B+` offsets `dec::ola_driver` uses, and an
//! independent esp walker reproduced all of that module's slot names). From
//! `B+0x5c` to `B+0x2fc` is 672 B = **`i32[168]`**, the OLA accumulator `w`:
//!
//! * the ring copy counts in **WORDS** (read from its bytes:
//!   `mov cx,WORD PTR [esi+eax*1]` / `rep stos WORD`), so `0xa8` = 168 words =
//!   336 B = **84 dwords**: `w[0..84] = state.ring`.
//! * the following zero-fill sets `w[84..168] = 0`.
//! * The dword typing is pinned independently by the tail's
//!   `lea eax,[esp+eax*4+0x74]` -- **scale 4** -- and by the ported
//!   [`crate::dec::ola_gen::place_harmonics`] / [`crate::dec::ola_gen::band_synth`]
//!   both taking `&mut [i32]`.
//!
//! ## The two passes (a textbook OLA)
//!
//! 1. `w` = `ring ++ zeros`; the gated harmonic placement adds the OLD frame's
//!    harmonics into it; the fade-out (`FoldDir::Out`) fades it out into `out`.
//! 2. `w` is zeroed whole (336 words = all 168 dwords); the NEW frame's
//!    spectrum is generated and placed; the fade-in (`FoldDir::In`) fades it in
//!    on top of `out`.
//! 3. The ring copy slides the ring forward by exactly `n` dwords. **The index
//!    is `B+0x58` = `sx16(a2)` = `n`, NOT `B+0x20`** -- an easy misread, and it
//!    is what makes the ring advance a constant 80.
//!
//! # Domain limits
//!
//! * `n == 0` makes the reciprocal's denominator zero, which the DLL would
//!   divide by. [`voiced_ola_body`] returns early there and **makes no claim**.
//! * `bfp_divide_round_i16` makes no claim at `a2 == 0`, for the same reason.
//! * `a6` reaches `bfp_divide_round_i16` and nothing else; its *provenance*
//!   (`movzx [ebx+0x7be]`) is a caller field this module does not model.

use crate::dec::ola_driver::{dec3d0, fold, Dec3d0Out, FoldDir};
use crate::dec::ola_gen::{band_synth, generate_spectrum, place_harmonics};
use crate::shared::atan2_bfp_divide::bfp_divide;

use crate::fixops::dec32::{sar, shl, sx16};

#[inline]
fn zx16(v: i32) -> i32 {
    v as u16 as i32
}

/// The `bsr`-based block-float normalize, open-coded at several sites in the
/// body and in `bfp_divide_round_i16`.
///
/// Note `bsr` leaves its destination **unchanged** on a zero source; every one
/// of these sites guards with `test/jne` or `test/je` first, so the `v == 0`
/// early return is the DLL's own path, not an invention.
#[inline]
fn norm_bsr(v: i32) -> i32 {
    if v == 0 {
        return 0;
    }
    let a = if v < 0 { !v } else { v };
    (0x1e - (31 - (a as u32).leading_zeros() as i32)) & 0xffff
}

// ===========================================================================
// bfp_divide_round_i16 -- 183 B, three ret sites.
// ===========================================================================

/// Faithful port of the 183-byte block-float divide leaf: a block-float divide
/// of `a1` by `a2`, rounded back to a plain `i16` at Q0.
///
/// # ABI -- exactly TWO args
///
/// The call site pushes three dwords but cleans only two (`add esp,0x8`):
///
/// ```text
///   push 0x2                   <-- NOT an arg: the leftover becomes generate_spectrum's a10
///   push eax                   <-- a2 = zx16([ebp+0xc]) = L
///   push DWORD PTR [esp+0x31c] <-- a1 = voiced_ola_body's ARG6
///   call bfp_divide_round_i16
///   add  esp,0x8
/// ```
///
/// Confirmed from the callee side: it reads `[esp+0x8]` (= a2) at entry and
/// `[esp+0x10]` (= a1, after its three pushes) and **never reads `[esp+0xc]`**.
///
/// # Mechanism
///
/// Both args are normalized (`a2` to a 16-bit mantissa, `a1` left as a full
/// 32-bit numerator), divided via `bfp_divide`, then the Q-`exp` result is
/// rounded to Q0 by `(x << (exp-15)) + 0x8000 >> 16` with the same three-way
/// shift the fold kernels use.
///
/// Two deliberate asymmetries, both read from the bytes and both preserved:
/// * the denominator is `sar(..., 16)` then **zero-extended** (`movzx ecx,dx`);
///   the numerator is **not** `sar`'d at all -- it is pushed full-width.
/// * the denominator's exponent is **zero-extended** (`movzx ebx,ax`), so
///   `-4 - norm` reaches `bfp_divide` as e.g. `0xfffc`, not `-4`. This is
///   harmless *and must be kept*: `bfp_divide` only uses `exp_adjust` via
///   `esi.wrapping_sub(exp_adjust) as i16`, which is invariant mod 2^16.
///
/// **Domain:** `a2 == 0` zeroes `bfp_divide`'s denominator, which the DLL
/// divides by. This port makes **no claim** there.
pub(crate) fn bfp_divide_round_i16(a1: i32, a2: i32) -> i16 {
    // Normalize the DENOMINATOR (a2).
    let d0 = shl(sx16(a2), 16);
    let norm_d = norm_bsr(d0);
    let den_mant = zx16(sar(shl(d0, norm_d as u32), 16)); // sar 16 then zero-extend
    let den_exp = zx16((-4i32).wrapping_sub(norm_d)); // zero-extended exponent

    // Normalize the NUMERATOR (a1).
    let n0 = shl(sx16(a1), 16);
    // A zero numerator skips the bsr, leaving the norm at 0.
    let norm_n = if n0 == 0 { 0 } else { norm_bsr(n0) };
    let num = shl(n0, norm_n as u32); // NOT sar'd -- full 32-bit
    let num_exp = (-4i32).wrapping_sub(norm_n); // full width, NOT zx16

    // bfp_divide(num, num_exp, den_mant, den_exp) -> (mant, exp)
    let Some((mant, exp)) = bfp_divide(num, num_exp, den_mant, den_exp) else {
        // bfp_divide's zero-denominator path would have faulted; unreachable
        // for a2 != 0. Returning 0 here is a Rust-side guard, not a DLL claim.
        return 0;
    };

    // Round to Q0.
    let val = shl(mant as i32, 16); // sign-extend the mantissa into the high word
    let sh = sx16((exp as i32).wrapping_sub(0xf)); // shift amount = exp - 15 (16-bit)
    let shifted = if sh >= 0 {
        shl(val, sh as u32)
    } else if sh <= -31 {
        // shift <= -31 collapses to the sign bit
        sar(val, 31)
    } else {
        // negative shift: sar by -sh
        sar(val, (-sh) as u32)
    };
    sar(shifted.wrapping_add(0x8000), 16) as i16
}

// ===========================================================================
// voiced_ola_body -- the OLA body.
// ===========================================================================

/// `voiced_ola_body`'s `a3`: the decoder object at `caller_ebx+0x18`.
///
/// Field offsets are the ones `voiced_ola_body` actually touches; the `ring`
/// length is fixed at 84 by the ring copy (`0xa8` = 168 words = 336 B), and by
/// `[edi+0x150]` (`carry`) sitting immediately after it.
#[derive(Clone, Debug)]
pub(crate) struct OlaState {
    /// `[edi+0x000..0x150]` -- 84 dwords.
    pub ring: [i32; 84],
    /// `[edi+0x150]`
    pub carry: i32,
    /// `[edi+0x154]`
    pub mant: i16,
    /// `[edi+0x156]`
    pub exp: i16,
    /// `[edi+0x15a..0x1ca]` -- 56 words; `generate_spectrum`'s `a4` (resid).
    pub resid: [i16; 56],
    /// `[edi+0x1ca]` -- `generate_spectrum`'s `a2` (exponent out).
    pub spec_exp: i16,
    /// `[edi+0x1cc..]` -- `generate_spectrum`'s `a1` (spectrum out, 258 words).
    pub spec: [i16; 258],
}

impl Default for OlaState {
    fn default() -> Self {
        OlaState {
            ring: [0; 84],
            carry: 0,
            mant: 0,
            exp: 0,
            resid: [0; 56],
            spec_exp: 0,
            spec: [0; 258],
        }
    }
}

/// `voiced_ola_body`'s `a4` (`ebp`). Offsets `+0x10..+0x80` = 0x70 B = 56 words
/// fix `amp`'s length, matching [`OlaState::resid`].
#[derive(Clone, Debug)]
pub(crate) struct FrameParams<'a> {
    /// `[ebp+0x4]` -- `generate_spectrum`'s `a8` (hi).
    pub hi: i32,
    /// `[ebp+0x8]` -- gate word (tested `& 0x55555555`).
    pub mask_word: u32,
    /// `[ebp+0xc]` -- L.
    pub l: i32,
    /// `[ebp+0xe]` -- feeds the pass-1 amplitude.
    pub e: i32,
    /// `[ebp+0x10..0x80]` -- `generate_spectrum`'s `a6` (amp).
    pub amp: [i16; 56],
    /// `[ebp+0x80]` -- a POINTER field (`push DWORD PTR`, not `lea`);
    /// `generate_spectrum`'s `a5` (mask).
    pub mask: &'a [i16],
    /// `[ebp+0x84]` -- `generate_spectrum`'s `a7` (base exponent).
    pub base_exp: i32,
}

/// `voiced_ola_body`'s `a5`. Only two fields are read.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameParams2 {
    /// `[a5+0x8]` -- gate word (tested `& 0x55555555`).
    pub mask_word: u32,
    /// `[a5+0xc]` -- L of the previous frame; `dec3d0`'s `a6`.
    pub l5: i32,
}

/// `(x & 0x55555555) != 0`, as the body open-codes it
/// (`and / neg / sbb / neg / movzx`).
#[inline]
fn gate(x: u32) -> i32 {
    ((x & 0x5555_5555) != 0) as i32
}

/// Faithful port of the 1277-byte voiced harmonic OLA body.
///
/// Emits into `out` (the `a1` the caller passes as `lea eax,[esp+0x40]`) and
/// advances `st`.
///
/// **NOT bit-exact:** emitted words match the reference on ~99.4%, and the
/// residual lives in the state fields, inherited from the unscored
/// [`generate_spectrum`] / [`band_synth`].
///
/// The fold-OUT count is `n` on BOTH `gate_a` paths: `place_harmonics` does not
/// leave `ret_hi_z` as the count, and the else arm loads `n`. Using `ret_hi_z`
/// there collapses every `gate_a == 1` frame.
///
/// Coverage caveat: real frames pin `n == 80`, so the `n != 80` ring-slide /
/// fold-count paths are unexercised.
pub(crate) fn voiced_ola_body(
    out: &mut [i32],
    n: i32,
    st: &mut OlaState,
    p4: &FrameParams<'_>,
    p5: &FrameParams2,
    a6: i32,
) {
    // ---- prologue ----------------------------------------------------------
    let mut amp_mant = zx16(st.mant as i32); // B+0x2c : local copy of [edi+0x154]
    let mut amp_exp = zx16(st.exp as i32); // B+0x28 : local copy of [edi+0x156]
    let gate_a = gate(p5.mask_word); // B+0x48
    let gate_b = gate(p4.mask_word); // B+0x40
    let n_sx = sx16(n); // B+0x58
    let n_hi = shl(n_sx, 16); // n placed in the high word

    // ---- reciprocal of n ---------------------------------------------------
    // bfp_divide(0x40000000, 1, sar(n_hi<<norm,16), 0xf-norm) -> (recip_mant, recip_exp)
    let norm0 = norm_bsr(n_hi);
    let y0 = sar(shl(n_hi, norm0 as u32), 16);
    let Some((recip_mant, recip_exp)) = bfp_divide(0x4000_0000, 1, y0, 0xf - norm0) else {
        // n == 0 => the DLL divides by zero here. No claim; see module header.
        return;
    };
    let scale_mant = zx16(recip_mant as i32); // zero-extended reciprocal mantissa
    let scale_exp = recip_exp as i32;

    // ---- dec3d0 ------------------------------------------------------------
    let d: Dec3d0Out = dec3d0(st.carry, zx16(p5.l5), zx16(p4.l), gate_a, gate_b, n);
    let dret = d.ret; // B+0x3c and B+0x54
    let ret_hi_z = sar(d.ret, 16) & 0xffff; // high word of dret -> B+0x20

    // ---- build the accumulator ---------------------------------------------
    // w[0..84] = ring (0xa8 words = 84 dwords) ; w[84..168] = 0.
    let mut w = [0i32; 168];
    w[..84].copy_from_slice(&st.ring);

    let mut pl_carry = st.carry; // B+0x18
    let mut pl_o1 = d.o1 as i32; // B+0x34
    let mut pl_o2 = d.o2 as i32; // B+0x30
    let mut acc_hi = ret_hi_z; // B+0x20

    // ---- gated harmonic placement #1 ---------------------------------------
    // The placement ADDS the old frame's harmonics into `w`; its `ret_hi_z` is
    // the placement COUNT (a11), NOT the subsequent fold-Out count. The
    // fold-Out count is `n` on BOTH gate_a paths -- `place_harmonics` does not
    // leave `ret_hi_z` as the count, and the else arm loads `n`.
    if gate_a != 0 && (ret_hi_z as u16 as i16) > 0 {
        let (mut amp_o, mut sh_o) = (amp_mant as i16, amp_exp as i16);
        place_harmonics(
            &mut w,
            &mut amp_o,
            &mut sh_o,
            zx16(p5.l5),
            &st.spec,
            zx16(st.spec_exp as i32),
            pl_carry,
            pl_o2,
            pl_o1,
            n,
            ret_hi_z,
        );
        amp_mant = zx16(amp_o as i32);
        amp_exp = zx16(sh_o as i32);
    }

    // ---- fade-OUT into `out` (count = n on both paths) ----------------------
    fold(out, &w, scale_mant, scale_exp, n, FoldDir::Out);

    // ---- zero the whole accumulator (0x150 words = 168 dwords) --------------
    w = [0i32; 168];

    // ---- phase step = -L, then bfp_divide_round_i16 ------------------------
    // L<<5 ; saturating neg ; <<0xb ; sar 0x10
    let l_z = zx16(p4.l);
    let mut ph = shl(sx16(p4.l), 5);
    ph = if ph == i32::MIN {
        i32::MAX
    } else {
        ph.wrapping_neg()
    };
    let phase_step = sar(shl(ph, 0xb), 0x10);
    let lo1 = zx16(bfp_divide_round_i16(a6, l_z) as i32); // zero-extended result

    // ---- generate_spectrum + IFFT, pass 1 (a9 = lo1, a10 = 2) --------------
    let r1 = generate_spectrum(
        &mut st.spec,
        &mut st.spec_exp,
        phase_step,
        &st.resid,
        p4.mask,
        &p4.amp,
        zx16(p4.base_exp),
        zx16(p4.hi),
        lo1,
        2,
    );

    // ---- if (r1 != 0) -> band_synth #1 -------------------------------------
    if (r1 as u16) != 0 {
        // v = (sx16([ebp+0xe]) << 8) + ((n - 16) << 16)
        let v = sar(shl(sx16(p4.e), 16), 8).wrapping_add(shl(n_sx.wrapping_sub(16), 16));
        let nv = norm_bsr(v);
        band_synth(
            &mut w,
            sar(shl(v, nv as u32), 16),
            0xf - nv,
            l_z,
            &st.spec,
            zx16(st.spec_exp as i32),
        );
    }

    // ---- gate_b -------------------------------------------------------------
    let mut acc: i32;
    if gate_b == 0 {
        acc = dret;
    } else {
        if d.o3 != 0 {
            // ola_carry_rewrite_pub(&carry, &ret, &o1, &o2, L5, edi)
            let mut t_carry = st.carry;
            let mut t_ret = dret;
            let mut t_o1 = d.o1 as i32;
            let mut t_o2 = d.o2 as i32;
            let mut ds = crate::dec::ola_driver::OlaDriverState {
                carry: st.carry,
                mant: st.mant,
                exp: st.exp,
            };
            crate::dec::ola_driver::ola_carry_rewrite_pub(
                &mut t_carry,
                &mut t_ret,
                &mut t_o1,
                &mut t_o2,
                zx16(p5.l5),
                &mut ds,
            );
            st.carry = ds.carry;
            pl_carry = t_carry;
            acc = t_ret;
            pl_o1 = t_o1;
            pl_o2 = t_o2;
        } else if d.o4 != 0 {
            let cy = st.carry;
            acc = (dret as i16 as i32).wrapping_sub(cy); // sub ; cwde
            pl_o2 = zx16(sar(shl(d.o2 as i32, 16), 17));
            pl_o1 = zx16(sar(shl(d.o1 as i32, 16), 17));
            let half_cy = sar(cy, 1);
            pl_carry = half_cy;
            acc = sar(acc, 1).wrapping_add(half_cy);
            st.carry = half_cy; // store half-carry to [edi+0x150]
        } else {
            acc = dret;
        }

        // acc_hi = zx16(sar(acc,16))
        acc_hi = sar(acc, 16) & 0xffff;

        // generate_spectrum pass 2 (a9 = 0, a10 = 1)
        let mut ph2 = shl(sx16(p4.l), 5);
        ph2 = if ph2 == i32::MIN {
            i32::MAX
        } else {
            ph2.wrapping_neg()
        };
        let phase_step2 = sar(shl(ph2, 0xb), 0x10);
        let _r2 = generate_spectrum(
            &mut st.spec,
            &mut st.spec_exp,
            phase_step2,
            &st.resid,
            p4.mask,
            &p4.amp,
            zx16(p4.base_exp),
            zx16(p4.hi),
            0,
            1,
        );

        // band_synth #2 -- the (amp, shift) pair here is read from
        // [edi+0x154]/[edi+0x156] DIRECTLY, i.e. the ENTRY values, NOT the
        // amp_mant/amp_exp locals that the placement rewrote. The two diverge
        // after a placement call, and both are live.
        band_synth(
            &mut w,
            zx16(st.mant as i32),
            zx16(st.exp as i32),
            l_z,
            &st.spec,
            zx16(st.spec_exp as i32),
        );

        // gated harmonic placement #2
        if (acc_hi as u16 as i16) > 0 {
            let (mut amp_o, mut sh_o) = (amp_mant as i16, amp_exp as i16);
            place_harmonics(
                &mut w,
                &mut amp_o,
                &mut sh_o,
                l_z,
                &st.spec,
                zx16(st.spec_exp as i32),
                pl_carry,
                pl_o2,
                pl_o1,
                n,
                acc_hi,
            );
            amp_mant = zx16(amp_o as i32);
            amp_exp = zx16(sh_o as i32);
        }
    }

    // ---- tail ---------------------------------------------------------------
    // fade-IN, then carry = acc - (sx16(acc_hi) << 16).
    let new_carry = acc.wrapping_sub(shl(sx16(acc_hi), 16));
    st.carry = new_carry; // store new_carry to [edi+0x150]
    fold(out, &w, scale_mant, scale_exp, n, FoldDir::In);

    // slide the ring by n. The index is B+0x58 = sx16(a2) = n.
    // (B+0x20 is a different slot.)
    let k = n_sx as usize;
    st.ring.copy_from_slice(&w[k..k + 84]);

    // both gates off => ONE dword store clears 0x154 AND 0x156.
    if gate_a == 0 && gate_b == 0 {
        st.mant = 0;
        st.exp = 0;
        return;
    }
    // (mant, exp) = bfp_sub(sar(n_hi<<norm,16), 0xf-norm, amp_mant, amp_exp)
    let norm = norm_bsr(n_hi);
    let m = sar(shl(n_hi, norm as u32), 0x10);
    let (nm, ne) = crate::shared::atan2_chain::bfp_sub(m, 0xf - norm, amp_mant, amp_exp);
    st.mant = nm;
    st.exp = ne;
}
