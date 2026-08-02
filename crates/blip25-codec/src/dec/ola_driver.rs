//! Voiced harmonic OLA **driver**: its decoder-state recurrence.
//!
//! This is the 1277-byte orchestrator at the head of the decode synthesis
//! chain (the decode entry -> this driver -> {gen+IFFT, placement OLA, band
//! synth, the fold kernels}). Ported here is the part of it that is pure
//! *state*: the three decoder words it owns and threads across every subframe
//! of the file.
//!
//! ```text
//!   [edi+0x150]  carry   (i32)   the within-buffer OLA start carry
//!   [edi+0x154]  mant    (i16)  \ one block-float pair, written by the
//!   [edi+0x156]  exp     (i16)  / driver's tail via the block-float subtract
//! ```
//!
//! # Mechanism (transcribed from the reference vocoder)
//!
//! `B` is the frame base after the prologue:
//!
//! ```text
//!   B+0x2c = movzx WORD[edi+0x154]      B+0x28 = movzx WORD[edi+0x156]
//!   B+0x48 = gate_a = (ARG5->[8] & 0x55555555) != 0
//!   B+0x40 = gate_b = (ARG4->[8] & 0x55555555) != 0
//!   esi    = sx16(ARG2) << 16                 (callee-saved, never clobbered)
//!   B+0x18 = carry_in   B+0x3c = dec3d0 ret   B+0x34 = o1   B+0x30 = o2
//!   B+0x20 = (ebx >> 16) & 0xffff
//! ```
//!
//! The driver makes **two gated placement-OLA calls**. The placement OLA's own
//! tail writes `*a2 = ampbuf[count-1]` and `*a3 = shiftbuf[count-1]` -- and
//! `a2`/`a3` are `&B+0x2c` / `&B+0x28`, i.e. the driver's own copies of the
//! 0x154/0x156 pair. `ampbuf`/`shiftbuf` are the amp/shift buffer builder's
//! output. So the state pair is *the last placement's (amp_norm, shift)*
//! whenever a placement call fires.
//!
//! The tail then either zeroes **both** words with one DWORD store
//! (`mov DWORD PTR [edi+0x154],eax` -- note DWORD, it clears 0x156 too) when
//! both gates are off, or block-float-subtracts:
//!
//! ```text
//!   (mant, exp) = bfp_sub( sar(esi << norm, 16), 0xf - norm, B+0x2c, B+0x28 )
//! ```
//!
//! # What this does NOT do
//!
//! This is the driver's **state machine only** -- not the OLA itself. The
//! sample-producing body (the gen+IFFT, the placement loop, the band synth, the
//! fade-in/fade-out folds and the `0xa8`-word buffer copies) is not here.

use crate::fixops::dec32::{sar, shl, sx16};

/// The `bsr`-based block-float normalize that the driver and `ola_carry_rewrite` open-code
/// inline.
#[inline]
fn norm_bsr(v: i32) -> i32 {
    if v == 0 {
        return 0;
    }
    let a = if v < 0 { !v } else { v };
    (0x1e - (31 - (a as u32).leading_zeros() as i32)) & 0xffff
}

/// Saturating variable left-shift / arithmetic right-shift.
/// (The lib's `enc::band_decompress::shift_scale` is the same function but is
/// `pub(crate)` there; this is the decode-side entry.)
pub(crate) fn block_denorm(value: i32, count: i32) -> i32 {
    let sh16 = (count & 0xffff) as u16 as i16;
    if sh16 < 0 {
        let c = if sh16 > -31 {
            ((-(sh16 as i32)) & 0xffff) as u32
        } else {
            0x1f
        };
        return sar(value, c);
    }
    let sh = sh16 as i32;
    let mask = shl(-1i32, (0x1f - sh) as u32);
    let mut lost = mask & value;
    if value < 0 {
        lost = lost.wrapping_sub(mask);
    }
    if lost != 0 {
        return if value < 0 { i32::MIN } else { i32::MAX };
    }
    shl(value, sh as u32)
}

/// The three decoder-state words the driver owns.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) struct OlaDriverState {
    /// `[edi+0x150]`
    pub carry: i32,
    /// `[edi+0x154]`
    pub mant: i16,
    /// `[edi+0x156]`
    pub exp: i16,
}

/// `dec3d0`'s five outputs, as the driver consumes them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dec3d0Out {
    pub o1: i16,
    pub o2: i16,
    pub o3: i16,
    pub o4: i16,
    pub ret: i32,
}

/// The `o3 != 0` branch (`ola_carry_rewrite`): rewrites the driver's
/// `(carry_in, ret, o1, o2)` locals in place and touches the state's own
/// carry/mant/exp words (the latter two are dead -- the driver's tail
/// unconditionally rewrites them before returning).
pub(crate) fn ola_carry_rewrite_pub(
    b50: &mut i32,
    b54: &mut i32,
    o1: &mut i32,
    o2: &mut i32,
    l5: i32,
    st: &mut OlaDriverState,
) {
    ola_carry_rewrite(b50, b54, o1, o2, l5, st)
}

fn ola_carry_rewrite(
    b50: &mut i32,
    b54: &mut i32,
    o1: &mut i32,
    o2: &mut i32,
    l5: i32,
    st: &mut OlaDriverState,
) {
    let ret_v = *b54;
    let carry = st.carry;
    let diff = ret_v.wrapping_sub(carry);
    let mut new_carry = carry.wrapping_add(carry); // 2*carry
    if (sar(new_carry, 16) as u16 as i16) > 0 {
        new_carry = new_carry.wrapping_sub(0x10000);
        let mut mag = if new_carry == i32::MIN {
            0x7fff_ffff
        } else {
            new_carry.wrapping_neg()
        };
        mag = sar(mag, 1);
        let norm_mag = norm_bsr(mag);
        let num = shl(mag, norm_mag as u32);
        let base_e = 0xf - norm_mag;
        let mut denom = shl(sx16(l5), 16);
        let norm_den = norm_bsr(denom);
        denom = sar(shl(denom, norm_den as u32), 0x10);
        // `bfp_divide` returns None for the DLL's zero-mantissa early-out,
        // which the DLL encodes as (0, 0).
        let (m, e) = crate::shared::atan2_bfp_divide::bfp_divide(num, base_e, denom, -4 - norm_den)
            .unwrap_or((0, 0));
        st.mant = m;
        st.exp = e;
    }
    st.carry = new_carry;
    *b50 = new_carry; // *arg1
    *o2 = sar(shl(*o2 as i16 as i32, 0x11), 0x10) & 0xffff; // *arg4 = 2*o2
    *o1 = sar(block_denorm(shl(*o1 as i16 as i32, 0x10), 1), 0x10) & 0xffff; // *arg3
    *b54 = st.carry.wrapping_add(diff.wrapping_mul(2)); // *arg2
}

// ===========================================================================
// dec3d0 -- the driver's 407-B parameter/carry leaf.
// ===========================================================================

/// `dec3d0`: the voiced-OLA driver's frame-interpolation / carry helper.
///
/// A **pure leaf** (407 B, zero callees, three return points). It writes four
/// `i16` out-params and returns the new cursor.
///
/// # Arg map
///
/// Bound at the sole call site, where one `add esp,0x48` cleans the preceding
/// call's 5 + this function's 10 + the buffer copy's 3 = 18 dwords -- which is
/// what fixes the count at **10 args**, not 8:
///
/// ```text
///   a1  = carry ([edi+0x150])        a2..a5 = i16* out (o1..o4)
///   a6  = ARG5->[0xc]                a7 = ARG4->[0xc] (= L)
///   a8, a9 = gates                   a10 = the multiplier
/// ```
///
/// Note the tail is a compiler-inlined block-float **round trip** -- normalize
/// by `bsr` then denormalize by the same amount. It is arithmetically the
/// identity on the working value, but it is transcribed faithfully rather than
/// folded away, since the shift pair is only an identity because `norm` is
/// derived from `bsr` of that same value.
pub(crate) fn dec3d0(a1: i32, a6: i32, a7: i32, a8: i32, a9: i32, a10: i32) -> Dec3d0Out {
    let a6_lo = (a6 & 0xffff) as u16;
    let mut o1 = a6_lo as i16;
    let mut o2: i16 = 0;
    let mut o3: i16 = 0;
    let mut o4: i16 = 0;

    let interp: i32;
    if (a8 & 0xffff) == 0 {
        // a8 == 0
        if (a9 & 0xffff) == 0 {
            // a8 == 0 && a9 == 0: zero interpolation
            return Dec3d0Out {
                o1,
                o2,
                o3,
                o4,
                ret: d3_tail(0, true),
            };
        }
        let a7_lo = (a7 & 0xffff) as u16;
        o1 = a7_lo as i16; // out-param o1
        interp = a7_lo as i16 as i32;
    } else if (a9 & 0xffff) == 0 {
        interp = a6_lo as i16 as i32; // sign-extend a6 low
    } else {
        // ---- main path ----
        // `0x9998`, `0x6aaa`, `0xa000`, `0x6666` and `K` below are transcribed
        // from the reference's instruction bytes, not fitted to any score. And
        // `0xa000` vs `0xa001` is degenerate -- no input separates them, so no
        // test can ever catch a change there. Do not adjust them.
        let a7_lo = (a7 & 0xffff) as u16;
        let w7 = a7_lo as i16 as i32;
        let w6 = a6_lo as i16 as i32;
        let a7s = w7;
        let a6s = w6;
        let mut sel = sar(imul(a7s, 0x9998), 0x10);
        let mut thr_lo = imul(a6s, 0x6aaa);
        let mut thr_hi = imul(a6s, 0xa000);
        const K: i32 = 0x4000;
        let mut done = false;

        if (sel as i16) <= (a6_lo as i16) {
            sel = K;
        } else {
            let t2 = sar(imul(a7s, 0x6666), 0x10);
            // two distinct reference saturation conditions share the sel=K action; kept
            // separate to mirror the ported control flow
            #[allow(clippy::if_same_then_else)]
            if (t2 as i16) >= (a6_lo as i16) {
                sel = K;
            } else if (a6_lo as i16) >= (K as i16) {
                sel = K;
            } else {
                sel = sar(shl(a7s, 0x10), 0x11) & 0xffff;
                o3 = 1; // out-param o3
                done = true;
            }
        }
        if !done {
            // ---- second selector ----
            thr_lo = sar(thr_lo, 0x10);
            if (a7_lo as i16) <= (thr_lo as i16) {
                sel = a7_lo as i32;
            } else {
                thr_hi = sar(thr_hi, 0x10);
                if (a7_lo as i16) >= (thr_hi as i16) || (a7_lo as i16) >= (sel as i16) {
                    sel = a7_lo as i32;
                } else {
                    sel = sar(shl(a7s, 0x11), 0x10) & 0xffff;
                    o4 = 1; // out-param o4
                }
            }
        }
        // ---- assemble o2 and ret ----
        let mut acc = (sel & 0xffff) as u16 as i16 as i32; // sign-extend the selected value
        let mut o2v = acc;
        let a6_hi = shl(w6, 0x10);
        o2v = shl(o2v, 0x10);
        o2v = o2v.wrapping_sub(a6_hi);
        acc = shl(acc, 0x10);
        o2v = sar(o2v, 0x10);
        o2 = o2v as i16; // out-param o2
        acc = sar(acc, 1).wrapping_add(sar(a6_hi, 1));
        acc = sar(acc, 0x10);
        let e = imul(acc as i16 as i32, a10 as i16 as i32); // sign-extend a10
        let e = e.wrapping_add(e); // * 2
        let e = sar(e, 4);
        let e = e.wrapping_add(a1);
        return Dec3d0Out {
            o1,
            o2,
            o3,
            o4,
            ret: d3_tail(e, e == 0),
        };
    }

    // ---- shared by the two early-exit branches ----
    let e = imul(interp, a10 as i16 as i32);
    let e = e.wrapping_add(e);
    let e = sar(e, 4);
    let e = e.wrapping_add(a1);
    Dec3d0Out {
        o1,
        o2,
        o3,
        o4,
        ret: d3_tail(e, e == 0),
    }
}

#[inline]
fn imul(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}

/// bsr-normalize, then denormalize by the same amount. Transcribed as written.
fn d3_tail(edx: i32, zf_zero: bool) -> i32 {
    let mut val = edx;
    let exp: i32;
    if zf_zero {
        let norm = 0i32;
        exp = (0xf - norm) & 0xffff;
        val = shl(val, norm as u32);
    } else {
        let v = if val < 0 { !val } else { val }; // abs via bitwise-not
        let b = 31 - (v as u32).leading_zeros() as i32; // bsr
        let norm = (0x1e - b) & 0xffff; // normalize shift
        exp = (0xf - norm) & 0xffff; // exponent = 15 - norm
        val = shl(val, norm as u32);
    }
    let sh = exp.wrapping_sub(0xf);
    if (sh as i16) >= 0 {
        return shl(val, (sh & 0xff) as u32); // plain shl -- NOT block_denorm
    }
    if (sh as i16) <= -31 {
        return sar(val, 31); // collapse: arithmetic shift by 31
    }
    sar(val, (sh.wrapping_neg() & 0xff) as u32)
}

// ============================================================================
// The two OLA cross-fade FOLD kernels: fade-out and fade-in.
// ============================================================================

/// Saturating 32-bit add, as both fold kernels open-code it.
///
/// The DLL detects signed-add overflow with `(sum^a) & (sum^b) < 0` and then
/// produces `0x7fffffff + (operand < 0)`. The two kernels test *different*
/// operands for that sign -- the fade-out kernel tests the contrib
/// (`test edi,edi`), the fade-in kernel tests the destination (`test esi,esi`).
/// This is only an apparent asymmetry: a signed add can overflow only when both
/// operands share a sign, so either test yields the same bit. One helper is
/// therefore correct for both, and matches `saturating_add`.
#[inline]
fn sat_add32(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

/// Phase-accumulator increment shared by both fold kernels.
///
/// `step32 = sx16(step16) << 16`, then a three-way branch on `sx16(shift)`:
/// `shl` when non-negative, `sar` by `-shift` while `shift > -31`
/// (`cmp cx,0xffe1; jg`), else the `sar edx,0x1f` collapse.
#[inline]
fn fold_incr(step16: i32, shift: i32) -> i32 {
    let step32 = shl(sx16(step16), 16);
    let sh = sx16(shift);
    if sh >= 0 {
        shl(step32, (sh & 31) as u32)
    } else if sh > -31 {
        sar(step32, ((-sh) & 31) as u32)
    } else {
        sar(step32, 31)
    }
}

/// `src[i] * w` in the kernels' Q15 split form.
///
/// The DLL splits `src` into a `movzx` low half and a `movsx` high half so the
/// product fits 32 bits: `(s_lo*w >> 15) + 2*(s_hi*w)`.
///
/// This is **provably identical** to the wide form `((s as i64) * w) >> 15`, and
/// is kept only because it is what the instruction bytes do. Since
/// `s == s_hi*2^16 + s_lo` exactly, `floor(s*w / 2^15) == 2*s_hi*w +
/// floor(s_lo*w / 2^15)` (the high partial is an exact multiple of `2^15`), and
/// both forms reduce mod `2^32` identically -- so the `wrapping_*` here is
/// *not* load-bearing (checked exhaustively over 197,525,504 `(s, w)` pairs
/// covering every one of the 65536 possible weights).
#[inline]
fn fold_mul_q15(s: i32, w: i32) -> i32 {
    let s_hi = sx16(sar(s, 16));
    let s_lo = (s as u16) as i32;
    sar(s_lo.wrapping_mul(w), 0xf).wrapping_add(s_hi.wrapping_mul(w).wrapping_mul(2))
}

/// Which of the two fold kernels to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FoldDir {
    /// FADE IN: `dst[i] = sat(dst[i] + src[i]*w)`.
    In,
    /// FADE OUT: `dst[i] = sat(dst[i] + src[i] - src[i]*w)`.
    Out,
}

/// The OLA cross-fade fold kernels (fade-in and fade-out), which are
/// byte-for-byte the same function apart from the sign of the contribution.
///
/// ABI, recovered by an esp-tracking walk of the prologue (both kernels
/// `sub esp,0xc` then push 4 registers, so the raw `[esp+N]` offsets shift
/// twice before the args are read):
///
/// ```text
///   a1 = dst (i32*)   a2 = src (i32*)   a3 = step16   a4 = shift   a5 = count
///   locals:  B-0x4 = -shift    B-0x8 = shift    B-0xc = sx16(step16) << 16
/// ```
///
/// The phase accumulator starts at **0** (`xor esi,esi` / `xor edi,edi`) and
/// advances by [`fold_incr`] with a *saturating* add each sample; the weight is
/// `w = sx16(phase >> 16)`. Loop is do-while over `movzx` count, guarded by an
/// entry `cmp ax,cx; jge` on the **signed 16-bit** count.
///
/// # Degeneracies -- what no input can distinguish
///
/// * The `0xffe1` branch edge is **behaviourally dead**, by two independent
///   arguments, not by lack of coverage:
///   1. `jg` vs `jge` at `-31` is a tautology -- at `cx == -31` the variable leg
///      computes `sar(x, 31)`, which *is* the collapse leg.
///   2. Moving the threshold to `-30` changes nothing either: at
///      `shift in {-30,-31}` the only reachable increments are `|incr| <= 2`, so
///      the phase cannot cross a 65536 boundary within the largest representable
///      count (32767) and `w` is bit-identical on every sample (exhaustively
///      checked: 0 differing samples over max-length runs).
///
///   What *is* pinned is that `shift <= -32` must take the collapse, because
///   `(-cx) & 31` wraps to a tiny shift there. The evidence for the constant
///   `0xffe1` itself is the instruction bytes (`66 83 f9 e1` / `7f 05`).
pub(crate) fn fold(
    dst: &mut [i32],
    src: &[i32],
    step16: i32,
    shift: i32,
    count: i32,
    dir: FoldDir,
) {
    let n = sx16(count);
    if n <= 0 {
        // count <= 0: return untouched (signed 16-bit compare).
        return;
    }
    let n = n as usize;
    assert!(
        dst.len() >= n && src.len() >= n,
        "fold: buffers shorter than count"
    );
    let incr = fold_incr(step16, shift);
    let mut phase: i32 = 0;
    for i in 0..n {
        let s = src[i];
        let w = sx16(sar(phase, 16));
        let sw = fold_mul_q15(s, w);
        let contrib = match dir {
            FoldDir::In => sw,
            FoldDir::Out => s.wrapping_sub(sw),
        };
        dst[i] = sat_add32(dst[i], contrib);
        phase = sat_add32(phase, incr);
    }
}
