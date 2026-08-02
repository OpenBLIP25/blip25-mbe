//! IMBE math primitives (`Pow2`, `cos_fxp`, `sin_fxp`, `L_mpy_ls`,
//! `sqrt_l_exp`) from the TIA-102.BABA fixed-point reference, which fixes both
//! the naming and the table-lookup-plus-interpolation method. Cross-checked for
//! bit-exactness against OP25's `imbe_vocoder`, another implementation of the
//! same standard.
#![allow(clippy::all)]

use super::fixp::*;
use super::tables::{COS_TABLE, POW2_TABLE, SQRT_TABLE};

pub(crate) const X05_Q15: i16 = 16384;
pub(crate) const ONE_Q15: i16 = 32767;

/// `pow(2.0, x)`: input signed Q10.22, output signed Q14.2.
pub(crate) fn pow2(x: i32) -> i16 {
    let mut exponent = extract_h(l_shr(x, 6));
    if exponent < 0 {
        exponent = add(exponent, 1);
    }
    let mut fraction = extract_l(l_shr(l_sub(x, l_shl(l_deposit_l(exponent), 6 + 16)), 7));
    if x < 0 {
        fraction = negate(fraction);
    }
    let mut l_x = l_mult(fraction, 32); // fraction << 6
    let i = extract_h(l_x) as usize; // b10..b16 of fraction
    l_x = l_shr(l_x, 1);
    let mut a = extract_l(l_x); // b0..b9
    a &= 0x7fff;

    l_x = l_deposit_h(POW2_TABLE[i]);
    let tmp = sub(POW2_TABLE[i], POW2_TABLE[i + 1]);
    l_x = l_msu(l_x, tmp, a);

    if x < 0 {
        l_x = l_deposit_h(div_s(0x4000, extract_h(l_x)));
        exponent = sub(exponent, 1);
    }
    let exp = sub(12, exponent);
    l_x = l_shr_r(l_x, exp);
    extract_h(l_x)
}

/// Multiply a 32-bit number by a 16-bit number, returning 32 bits. `L_var2` is
/// truncated to 31 bits before the multiply (matches the reference exactly).
pub(crate) fn l_mpy_ls(l_var2: i32, var1: i16) -> i32 {
    let mut swtemp = shr(extract_l(l_var2), 1);
    swtemp &= 0x7fff;
    let mut out = l_mult(var1, swtemp);
    out = l_shr(out, 15);
    l_mac(out, var1, extract_h(l_var2))
}

/// Cosine of `x` (argument in radians/π, Q1.15), result Q1.15.
pub(crate) fn cos_fxp(x: i16) -> i16 {
    let mut sign = false;
    let mut tx = if x < 0 { negate(x) } else { x };
    if tx > X05_Q15 {
        tx = sub(ONE_Q15, tx);
        sign = true;
    }
    let index1 = shr(tx, 7);
    let index2 = add(index1, 1);
    if index1 == 128 {
        return 0;
    }
    let mut m = sub(tx, shl(index1, 7));
    m = shl(m, 8);
    let temp = sub(COS_TABLE[index2 as usize], COS_TABLE[index1 as usize]);
    let temp = mult(m, temp);
    let ty = add(COS_TABLE[index1 as usize], temp);
    if sign {
        negate(ty)
    } else {
        ty
    }
}

/// `sqrt(L_x)` for positive `L_x` (Q1.31). Returns the mantissa in Q1.31 and the
/// right-shift `exp` to apply afterwards (`*exp` in the reference).
pub(crate) fn sqrt_l_exp(l_x: i32, exp: &mut i16) -> i32 {
    if l_x <= 0 {
        *exp = 0;
        return 0;
    }
    let e = norm_l(l_x) & 0xfffeu16 as i16; // next lower EVEN norm
    let mut lx = l_shl(l_x, e);
    *exp = e >> 1;
    lx = l_shr(lx, 9);
    let mut i = extract_h(lx);
    lx = l_shr(lx, 1);
    let mut a = extract_l(lx);
    a &= 0x7fff;
    i = sub(i, 16);
    let mut l_y = l_deposit_h(SQRT_TABLE[i as usize]);
    let tmp = sub(SQRT_TABLE[i as usize], SQRT_TABLE[(i + 1) as usize]);
    l_y = l_msu(l_y, tmp, a);
    l_y
}

#[cfg(test)]
mod tests {
    //! Pins for the four table-plus-interpolation primitives.
    //!
    //! The exact-value assertions are anchors whose answer follows from the
    //! Q-format contract rather than from running the code: `pow2(0)` must be
    //! 1.0, `cos_fxp(0)` must be +1.0, `cos_fxp(0.5)` must be 0, and
    //! `sqrt_l_exp(0.5)` must be 1/sqrt(2). Interpolated points in between are
    //! covered by the sweep hashes, which is the honest split — an
    //! interpolated value is only derivable by redoing the arithmetic.
    //!
    //! Index-range assertions are here too, because the guards that keep
    //! `POW2_TABLE[i + 1]` and `SQRT_TABLE[i + 1]` in bounds are implicit in
    //! the normalization, not checked at the access.

    use super::*;

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    fn hash_i64s(vals: &[i64]) -> u64 {
        let mut bytes = Vec::with_capacity(vals.len() * 8);
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        fnv1a(&bytes)
    }

    /// One unit in the Q10.22 input format of [`pow2`] — i.e. `2^1`.
    const Q22_ONE: i32 = 1 << 22;

    #[test]
    fn q_format_constants() {
        assert_eq!(X05_Q15, 16384);
        assert_eq!(ONE_Q15, 32767);
        assert_eq!(POW2_TABLE.len(), 33);
        assert_eq!(COS_TABLE.len(), 129);
        assert_eq!(SQRT_TABLE.len(), 49);
        // The endpoints the exact-value pins below rely on.
        assert_eq!(POW2_TABLE[0], 16384); // 2^0 in Q1.15
        assert_eq!(POW2_TABLE[32], 32767); // 2^1 in Q1.15, minus 1 ulp
        assert_eq!(COS_TABLE[0], 32767); // cos(0)
        assert_eq!(COS_TABLE[128], 0); // cos(pi/2)
        assert_eq!(SQRT_TABLE[16], 23170); // sqrt(0.5) in Q1.15
        assert_eq!(SQRT_TABLE[48], 32767); // sqrt(1.0), minus 1 ulp
    }

    #[test]
    fn pow2_hits_the_exact_powers() {
        // Output is Q14.2, so 1.0 is 4.
        assert_eq!(pow2(0), 4);
        assert_eq!(pow2(Q22_ONE), 8); // 2^1
        assert_eq!(pow2(-Q22_ONE), 2); // 2^-1
        assert_eq!(pow2(2 * Q22_ONE), 16); // 2^2
        assert_eq!(pow2(-2 * Q22_ONE), 1); // 2^-2
                                           // 2^-3 = 0.125 is below the Q14.2 resolution and truncates to 0.
        assert_eq!(pow2(-3 * Q22_ONE), 0);
        // Q14.2 tops out at 8191.75; a large exponent saturates there.
        assert_eq!(pow2(11 * Q22_ONE), 8192);
        assert_eq!(pow2(i32::MAX), 32767);
        assert_eq!(pow2(i32::MIN), 0);
    }

    #[test]
    fn pow2_table_index_stays_in_range_over_the_whole_i32_domain() {
        // `POW2_TABLE[i + 1]` is reachable only because the fraction extracted
        // above is bounded to +-32767 and then shifted down by 10. There is no
        // bounds guard at the access, so exercise the domain rather than trust
        // it: any escape panics here instead of in the decoder.
        for k in -600i64..=600 {
            let x = (k * (Q22_ONE as i64) / 7) as i32;
            let _ = pow2(x);
        }
        for &x in &[i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX] {
            let _ = pow2(x);
        }
    }

    #[test]
    fn cos_fxp_hits_the_exact_quadrant_points() {
        // Argument is radians/pi in Q1.15: 0 -> cos(0), 16384 -> cos(pi/2),
        // 32767 -> cos(pi).
        assert_eq!(cos_fxp(0), 32767);
        assert_eq!(cos_fxp(X05_Q15), 0);
        assert_eq!(cos_fxp(-X05_Q15), 0);
        assert_eq!(cos_fxp(ONE_Q15), -32767);
        // cos(pi/4) = 0.7071 -> 23170, a table entry hit exactly.
        assert_eq!(cos_fxp(8192), 23170);
        assert_eq!(cos_fxp(-8192), 23170);
        // i16::MIN folds through `negate`, which saturates to 32767 first, so
        // it lands on cos(pi) rather than on cos(-pi) by a separate path.
        assert_eq!(cos_fxp(i16::MIN), -32767);
        // Even function.
        for x in (-32768i32..=32767).step_by(97) {
            let x = x as i16;
            if x == i16::MIN {
                // negate(i16::MIN) saturates, so the symmetry is broken at
                // exactly one input; -32767 is its true mirror.
                assert_eq!(cos_fxp(x), cos_fxp(32767));
                continue;
            }
            assert_eq!(cos_fxp(x), cos_fxp(-x), "cos_fxp is not even at {x}");
        }
    }

    #[test]
    fn sqrt_l_exp_returns_mantissa_and_halved_shift() {
        let mut e: i16 = 99;
        // 0.5 in Q1.31 -> sqrt = 0.7071 -> 0x5a82_0000, no post-shift.
        assert_eq!(sqrt_l_exp(0x4000_0000, &mut e), 0x5a82_0000);
        assert_eq!(e, 0);

        // Non-positive inputs short-circuit and clear the exponent.
        e = 99;
        assert_eq!(sqrt_l_exp(0, &mut e), 0);
        assert_eq!(e, 0);
        e = 99;
        assert_eq!(sqrt_l_exp(-1, &mut e), 0);
        assert_eq!(e, 0);
        e = 99;
        assert_eq!(sqrt_l_exp(i32::MIN, &mut e), 0);
        assert_eq!(e, 0);

        // The smallest positive input normalizes by 30, so the caller shifts
        // the same 0.7071 mantissa down by 15.
        e = 99;
        assert_eq!(sqrt_l_exp(1, &mut e), 0x5a82_0000);
        assert_eq!(e, 15);

        // Just under 1.0: mantissa just under 1.0, no post-shift.
        e = 99;
        assert_eq!(sqrt_l_exp(i32::MAX, &mut e), 2_147_417_600);
        assert_eq!(e, 0);

        // The normalization is forced to an EVEN shift so that halving it is
        // exact; 2 and 3 therefore share an exponent with 1 rather than
        // getting 29 >> 1.
        e = 99;
        let _ = sqrt_l_exp(2, &mut e);
        assert_eq!(e, 14);
        e = 99;
        let _ = sqrt_l_exp(3, &mut e);
        assert_eq!(e, 14);
    }

    #[test]
    fn sqrt_table_index_stays_in_range_over_every_positive_input() {
        // `SQRT_TABLE[i + 1]` has no bounds guard; it is in range only because
        // the even normalization pins `extract_h(lx >> 9)` to 16..=63.
        let mut e: i16 = 0;
        let mut x: i64 = 1;
        while x <= i32::MAX as i64 {
            let _ = sqrt_l_exp(x as i32, &mut e);
            let _ = sqrt_l_exp((x + x / 3 + 1).min(i32::MAX as i64) as i32, &mut e);
            x *= 2;
        }
        for &v in &[1i32, 2, 3, 0x2000_0000, 0x3fff_ffff, 0x4000_0000, i32::MAX] {
            let _ = sqrt_l_exp(v, &mut e);
        }
    }

    #[test]
    fn l_mpy_ls_splits_the_operand_at_the_halfword() {
        // Only the high half survives when the low half is zero.
        assert_eq!(l_mpy_ls(0x1_0000, 16384), 32768);
        assert_eq!(l_mpy_ls(-0x1_0000, 16384), -32768);
        assert_eq!(l_mpy_ls(0, 16384), 0);
        assert_eq!(l_mpy_ls(0x1234_5678, 0), 0);

        // The low half is shifted right one and masked to 15 bits, which
        // deliberately discards its low bit and its sign — that truncation to
        // 31 bits is the documented behaviour, not an accident.
        assert_eq!(l_mpy_ls(i32::MAX, 32767), 2_147_418_110);
        assert_eq!(l_mpy_ls(i32::MIN, 32767), -2_147_418_112);
        // 0xffff and 0xfffe in the low half give the same answer: the low bit
        // is shifted away before the mask.
        assert_eq!(l_mpy_ls(0xffff, 32767), l_mpy_ls(0xfffe, 32767));

        // The one saturating path is `l_mult`'s MIN_16*MIN_16 special case,
        // reached through the high-half term. The final `l_add` cannot
        // saturate: the accumulator and the product always have opposite
        // signs whenever the product is at full scale.
        assert_eq!(l_mpy_ls(i32::MIN, i16::MIN), i32::MAX);
    }

    // ── sweeps ────────────────────────────────────────────────────────────

    /// Re-bless by running with `--nocapture` and pasting the printed block.
    const SWEEPS: [(&str, u64); 4] = [
        ("pow2", 0x1ae03461582b9b17),
        ("cos_fxp", 0xff89fadb37ffdee0),
        ("sqrt_l_exp", 0xaca40fe39c2e2a46),
        ("l_mpy_ls", 0x29eb2355aa33548e),
    ];

    #[test]
    fn math_primitive_sweeps_are_pinned() {
        fn push(name: &'static str, v: &mut Vec<i64>, acc: &mut Vec<(&'static str, u64)>) {
            acc.push((name, hash_i64s(v)));
            v.clear();
        }

        let mut out: Vec<i64> = Vec::new();
        let mut actual: Vec<(&'static str, u64)> = Vec::new();

        // 1/8-of-an-octave steps across +-5 octaves, plus the extremes.
        for k in -40i32..=40 {
            out.push(pow2(k * (Q22_ONE / 8)) as i64);
        }
        for &x in &[i32::MIN, -1, 0, 1, 12_345_678, -12_345_678, i32::MAX] {
            out.push(pow2(x) as i64);
        }
        push("pow2", &mut out, &mut actual);

        // Every i16 argument: cheap, and it covers both the folded upper half
        // and the `index1 == 128` early return.
        for x in -32768i32..=32767 {
            out.push(cos_fxp(x as i16) as i64);
        }
        push("cos_fxp", &mut out, &mut actual);

        let mut e: i16 = 0;
        for k in 0..64 {
            let x = ((1i64 << (k / 2)) * (1 + (k as i64 % 5)) / 2).clamp(0, i32::MAX as i64) as i32;
            out.push(sqrt_l_exp(x, &mut e) as i64);
            out.push(e as i64);
        }
        for &x in &[i32::MIN, -1, 0, 1, 2, 3, 0x4000_0000, i32::MAX] {
            out.push(sqrt_l_exp(x, &mut e) as i64);
            out.push(e as i64);
        }
        push("sqrt_l_exp", &mut out, &mut actual);

        for &l in &[
            0i32,
            1,
            -1,
            0x1_0000,
            -0x1_0000,
            0x7fff_ffff,
            -0x8000_0000,
            0x1234_5678,
            -0x1234_5678,
            0x0000_8000,
            0x0000_ffff,
        ] {
            for &v in &[0i16, 1, -1, 16384, -16384, 32767, -32768] {
                out.push(l_mpy_ls(l, v) as i64);
            }
        }
        push("l_mpy_ls", &mut out, &mut actual);

        println!("\n// Re-bless by copying this block into SWEEPS:");
        println!("    const SWEEPS: [(&str, u64); {}] = [", actual.len());
        for (name, h) in &actual {
            println!("        (\"{name}\", {h:#018x}),");
        }
        println!("    ];\n");

        assert!(
            SWEEPS.iter().all(|(_, h)| *h != 0),
            "SWEEPS still holds placeholder zeros — an unpinned gate reads as \
             passing, which is worse than no gate."
        );
        assert_eq!(SWEEPS.len(), actual.len(), "SWEEPS entry count drifted");

        let mut bad = Vec::new();
        for ((name, got), (want_name, want)) in actual.iter().zip(SWEEPS.iter()) {
            assert_eq!(name, want_name, "SWEEPS is out of order");
            if got != want {
                bad.push(format!("{name}: expected {want:#018x}, got {got:#018x}"));
            }
        }
        assert!(
            bad.is_empty(),
            "an IMBE math primitive changed:\n  {}\n\nThese feed spectral \
             amplitude reconstruction directly; a one-ulp move here is audible \
             across the whole frame.",
            bad.join("\n  ")
        );
    }
}
