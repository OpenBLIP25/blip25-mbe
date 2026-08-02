//! The three pure LEAVES of the voiced-OLA generator/placement closure
//! (generation + IFFT -> placement -> output): `round_residual` (120 B),
//! `trunc_toward_zero` (82 B) and `block_exp_masked` (108 B). All three have
//! zero callees.
//!
//! # Degeneracies -- what no input can distinguish
//!
//! * **`cx > -31` vs `cx > -32`**, and **`round_residual`'s `movsx dx`**. Both
//!   are provably degenerate, not coverage gaps: at `cx == -31` both arms
//!   compute `sar(v, 31)`; and `edx >> 16` always lies in `[-32768, 32767]`, so
//!   the `movsx` is a mathematical no-op.
//! * **`block_exp_masked`'s 16-bit vs 32-bit bounds compare** is a genuine gap.
//!   Separating them needs `lo`/`hi` >= 65536, which would index far out of
//!   bounds, so it cannot be exercised safely. Its only evidence is the
//!   instruction bytes: `66 3b eb` = `cmp bp,bx` -- the `66` operand-size
//!   prefix is explicit.
//!
//! # Two branches are UNREACHABLE, not merely unexercised
//!
//! * `block_exp_masked`'s `not eax` is **dead code**. `eax` is a running max
//!   seeded at 0 and only replaced via `cmovg`, so it is provably never
//!   negative; the `eax == 0` case returns earlier.
//! * `trunc_toward_zero`'s SECOND `cmp eax,0x80000000` is likewise **dead**: at
//!   that point `eax = (-x) & 0xffff0000` for `x < 0`, which lies in
//!   `[0, 0x7fff0000]` and can never equal `0x80000000`.
//!
//! Both are ported faithfully anyway because the DLL encodes them; their only
//! evidence is the instruction bytes.

/// The DLL's ubiquitous signed-shift idiom, appearing at every shift site in
/// `round_residual` / `trunc_toward_zero`:
///
/// ```text
///   test cx,cx        ; 16-bit sign test of the LOW WORD
///   js   neg
///   shl  edx,cl       ; cl = low byte, x86 masks the count to &31
///   jmp  done
/// neg:
///   cmp  cx,0xffe1    ; signed compare against -31
///   jg   shr
///   sar  edx,0x1f     ; saturate to the sign
///   jmp  done
/// shr:
///   neg  ecx
///   sar  edx,cl
/// ```
///
/// Note the asymmetry that makes this worth a named helper: the branch is
/// decided by the **16-bit** low word, while the shift amount is `cl & 31`.
#[inline]
fn shiftop(v: i32, ecx: i32) -> i32 {
    let lo16 = ecx as i16;
    if lo16 >= 0 {
        ((v as u32) << (ecx & 31)) as i32
    } else if lo16 > -31 {
        v >> (ecx.wrapping_neg() & 31)
    } else {
        v >> 31
    }
}

/// Port of the OLA generator's rounding-RESIDUAL helper (120 B, leaf).
///
/// Extracts the part of `a1` BELOW the Q-point implied by `a2`: it rounds `a1`
/// to 16-bit granularity at scale `a2` (truncating toward -inf, then nudging
/// negatives up by one ulp -- the `add edx,0x10000`), subtracts that from
/// `a1`, and rescales the remainder by `a2`.
#[inline]
pub(crate) fn round_residual(a1: i32, a2: i32) -> i32 {
    // shiftop(a1, a2 - 15)
    let mut quant = shiftop(a1, a2.wrapping_sub(15));
    // sar 16 / movsx dx / shl 16
    quant = (((quant >> 16) as i16) as i32) << 16;
    // negatives round UP one ulp
    if quant < 0 {
        quant = quant.wrapping_add(0x10000);
    }
    // shiftop(quant, 15 - a2)
    quant = shiftop(quant, 15i32.wrapping_sub(a2));
    // the residual
    let residual = a1.wrapping_sub(quant);
    // shiftop(residual, a2)
    shiftop(residual, a2)
}

/// Port of the truncate-toward-ZERO helper (82 B, leaf): truncates to 16-bit
/// granularity at scale `a2`, with `0x80000000` saturation.
///
/// Unlike [`round_residual`]'s toward-`-inf` masking, this one takes the
/// magnitude first (`neg`), masks, then restores the sign -- i.e. symmetric
/// truncation toward zero.
#[inline]
pub(crate) fn trunc_toward_zero(a1: i32, a2: i32) -> i32 {
    // shiftop(a1, a2 - 15)
    let shifted = shiftop(a1, a2.wrapping_sub(15));
    // the positive leg is a plain mask
    if shifted >= 0 {
        return shifted & 0xffff_0000u32 as i32;
    }
    // INT_MIN saturates rather than overflowing neg
    let mag = if shifted == i32::MIN {
        i32::MAX
    } else {
        shifted.wrapping_neg()
    };
    let masked = mag & 0xffff_0000u32 as i32;
    // DEAD -- see module docs; the masked magnitude is in [0, 0x7fff0000] here.
    if masked == i32::MIN {
        return i32::MAX;
    }
    masked.wrapping_neg()
}

/// Port of the masked block-exponent finder (108 B, leaf).
///
/// Over `i` in `[lo, hi)` where `mask[i] == mv` (16-bit equality), takes the
/// maximum of `(arr[i] as i32) << 16` seeded at 0, then returns the NEGATED
/// normalization shift `-(30 - bsr(max))`, or 0 when no element matched or the
/// max is 0.
///
/// The bounds test (`cmp bp,bx`) is a **16-bit signed** compare and the trip
/// count is `(hi - lo) & 0xffff` taken as a do-while -- both reproduced here.
pub(crate) fn block_exp_masked(arr: &[i16], mask: &[i16], lo: i32, hi: i32, mv: i32) -> i32 {
    let mut max: i32 = 0;
    // 16-bit SIGNED bounds compare
    if (lo as i16) < (hi as i16) {
        // trip count is the low 16 bits, consumed do-while
        let cnt = ((hi.wrapping_sub(lo)) & 0xffff) as usize;
        let start = (lo as i16) as i32;
        for k in 0..cnt {
            let i = (start as usize).wrapping_add(k);
            // 16-bit equality against arg5
            if (mask[i] as u16) == (mv as u16) {
                // movsx / shl 16 / cmovg
                let c = (arr[i] as i32) << 16;
                if c > max {
                    max = c;
                }
            }
        }
    }
    // running max is 0 -> return 0
    if max == 0 {
        return 0;
    }
    // DEAD -- the running max is seeded at 0, never negative.
    let v = if max < 0 { !max } else { max };
    // bsr / 30 - bsr / movzx / neg
    let bsr = 31 - v.leading_zeros() as i32;
    -((30 - bsr) & 0xffff)
}
