//! ETSI / ITU-T G.191 fixed-point basic operators — the subset the IMBE decode
//! path uses. These are a **published ITU-T standard** operator set (Software
//! Tools Library), not specific to any vocoder: every fixed-point speech codec
//! implements the same saturating semantics, OP25's `imbe_vocoder` included.
//! Implemented here from the G.191 definitions so IMBE decode is **bit-exact**
//! to the standard fixed-point vocoder.
//!
//! `Word16` = [`i16`], `Word32` = [`i32`]. All ops saturate on overflow exactly
//! as the ETSI library does. The `Overflow`/`Carry` global flags are not read by
//! the decode path, so they are omitted.
#![allow(clippy::all)]

pub(crate) const MAX_16: i16 = 0x7fff;
pub(crate) const MIN_16: i16 = -0x8000;
pub(crate) const MAX_32: i32 = 0x7fff_ffff;
pub(crate) const MIN_32: i32 = -0x8000_0000;

#[inline]
pub(crate) fn sat_l(x: i64) -> i32 {
    if x > MAX_32 as i64 {
        MAX_32
    } else if x < MIN_32 as i64 {
        MIN_32
    } else {
        x as i32
    }
}
#[inline]
fn sat_s(x: i32) -> i16 {
    if x > MAX_16 as i32 {
        MAX_16
    } else if x < MIN_16 as i32 {
        MIN_16
    } else {
        x as i16
    }
}

#[inline]
pub(crate) fn add(a: i16, b: i16) -> i16 {
    sat_s(a as i32 + b as i32)
}
#[inline]
pub(crate) fn sub(a: i16, b: i16) -> i16 {
    sat_s(a as i32 - b as i32)
}
#[inline]
pub(crate) fn negate(a: i16) -> i16 {
    if a == MIN_16 {
        MAX_16
    } else {
        -a
    }
}
#[inline]
pub(crate) fn extract_h(l: i32) -> i16 {
    (l >> 16) as i16
}
#[inline]
pub(crate) fn extract_l(l: i32) -> i16 {
    (l & 0xffff) as i16
}
#[inline]
pub(crate) fn l_deposit_h(a: i16) -> i32 {
    (a as i32) << 16
}
#[inline]
pub(crate) fn l_deposit_l(a: i16) -> i32 {
    a as i32
}

#[inline]
pub(crate) fn shr(a: i16, b: i16) -> i16 {
    if b < 0 {
        return shl(a, -b);
    }
    if b >= 15 {
        return if a < 0 { -1 } else { 0 };
    }
    a >> b
}
#[inline]
pub(crate) fn shl(a: i16, b: i16) -> i16 {
    if b < 0 {
        return shr(a, -b);
    }
    if b >= 15 {
        return if a == 0 {
            0
        } else if a > 0 {
            MAX_16
        } else {
            MIN_16
        };
    }
    sat_s((a as i32) << b)
}
#[inline]
pub(crate) fn mult(a: i16, b: i16) -> i16 {
    if a == MIN_16 && b == MIN_16 {
        return MAX_16;
    }
    (((a as i32) * (b as i32)) >> 15) as i16
}
#[inline]
pub(crate) fn l_mult(a: i16, b: i16) -> i32 {
    if a == MIN_16 && b == MIN_16 {
        return MAX_32;
    }
    ((a as i32) * (b as i32)) << 1
}
#[inline]
pub(crate) fn mult_r(a: i16, b: i16) -> i16 {
    if a == MIN_16 && b == MIN_16 {
        return MAX_16;
    }
    let l = ((a as i32) * (b as i32) + 0x4000) >> 15;
    sat_s(l)
}
#[inline]
pub(crate) fn l_add(a: i32, b: i32) -> i32 {
    sat_l(a as i64 + b as i64)
}
#[inline]
pub(crate) fn l_sub(a: i32, b: i32) -> i32 {
    sat_l(a as i64 - b as i64)
}
#[inline]
pub(crate) fn l_mac(acc: i32, a: i16, b: i16) -> i32 {
    l_add(acc, l_mult(a, b))
}
#[inline]
pub(crate) fn l_msu(acc: i32, a: i16, b: i16) -> i32 {
    l_sub(acc, l_mult(a, b))
}
#[inline]
pub(crate) fn l_shr(l: i32, b: i16) -> i32 {
    if b < 0 {
        return l_shl(l, -b);
    }
    if b >= 31 {
        return if l < 0 { -1 } else { 0 };
    }
    l >> b
}
#[inline]
pub(crate) fn l_shl(l: i32, b: i16) -> i32 {
    if b < 0 {
        return l_shr(l, -b);
    }
    let mut out = l;
    for _ in 0..b {
        if out > 0x3fff_ffff {
            return MAX_32;
        } else if out < -0x4000_0000 {
            return MIN_32;
        }
        out <<= 1;
    }
    out
}
#[inline]
pub(crate) fn l_shr_r(l: i32, b: i16) -> i32 {
    if b > 31 {
        return 0;
    }
    let out = l_shr(l, b);
    if b > 0 && (l & (1i32 << (b - 1))) != 0 {
        l_add(out, 1)
    } else {
        out
    }
}
#[inline]
pub(crate) fn norm_s(a: i16) -> i16 {
    if a == 0 {
        return 0;
    }
    if a == -1 {
        return 15;
    }
    let mut v = if a < 0 { !a } else { a };
    let mut n = 0i16;
    while v < 0x4000 {
        v <<= 1;
        n += 1;
    }
    n
}
#[inline]
pub(crate) fn norm_l(l: i32) -> i16 {
    if l == 0 {
        return 0;
    }
    if l == -1 {
        return 31;
    }
    let mut v = if l < 0 { !l } else { l };
    let mut n = 0i16;
    while v < 0x4000_0000 {
        v <<= 1;
        n += 1;
    }
    n
}
/// Q15 division `var1/var2`, both non-negative and `var1 <= var2`. Result in
/// `[0, 0x7fff]`. ETSI iterative restoring division.
#[inline]
pub(crate) fn div_s(var1: i16, var2: i16) -> i16 {
    if var1 == 0 {
        return 0;
    }
    if var1 == var2 {
        return MAX_16;
    }
    // (the reference relies on 0 <= var1 <= var2; mirror its behaviour)
    let mut l_num = l_deposit_l(var1);
    let l_denom = l_deposit_l(var2);
    let mut out: i16 = 0;
    for _ in 0..15 {
        out <<= 1;
        l_num <<= 1;
        if l_num >= l_denom {
            l_num = l_sub(l_num, l_denom);
            out = add(out, 1);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    //! Saturation boundaries and rounding directions for the G.191 subset.
    //!
    //! Where the ITU-T Software Tools Library semantics are unambiguous the
    //! assertion states the standard's answer, so this doubles as a
    //! conformance check and not merely a change detector. Two places where
    //! this implementation reaches the same answer by a different route are
    //! called out inline (`mult`, `l_mult`): the reference gets there by
    //! masking and saturating, this code by an explicit `MIN_16 * MIN_16`
    //! special case. One genuine deviation is recorded in
    //! `shr_negative_count_is_not_clamped`.
    //!
    //! Sweep hashes back the hand-picked points up; FNV-1a is hand-inlined so
    //! no dependency can move one.

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

    /// Word16 probes: both extremes, both sides of the Q15 half-scale point
    /// (16384), and the small magnitudes where rounding direction shows.
    const W16: [i16; 18] = [
        0,
        1,
        -1,
        2,
        -2,
        3,
        -3,
        100,
        -100,
        16383,
        16384,
        16385,
        -16383,
        -16384,
        -16385,
        0x7ffe,
        i16::MAX,
        i16::MIN,
    ];

    /// Word32 probes, straddling the `l_shl` saturation thresholds
    /// (`0x3fff_ffff` / `-0x4000_0000`) as well as both extremes.
    const W32: [i32; 18] = [
        0,
        1,
        -1,
        2,
        -2,
        3,
        -3,
        0x7fff,
        -0x8000,
        0x1_0000,
        0x1234_5678,
        -0x1234_5678,
        0x3fff_ffff,
        0x4000_0000,
        -0x4000_0000,
        -0x4000_0001,
        i32::MAX,
        i32::MIN,
    ];

    /// Word16 shift counts. `i16::MIN` is deliberately absent: `shr`/`shl`
    /// negate the count to delegate to each other, and negating `i16::MIN`
    /// does not fit in an `i16` — see `shr_negative_count_is_not_clamped`.
    const SB: [i16; 12] = [-32, -16, -15, -8, -1, 0, 1, 8, 14, 15, 16, 31];

    /// Word32 shift counts, including 31/32/33 where `l_shr` and `l_shr_r`
    /// change branch.
    const SB32: [i16; 12] = [-32, -31, -16, -1, 0, 1, 15, 16, 30, 31, 32, 33];

    /// `div_s` operands, respecting its stated domain `0 <= var1 <= var2`.
    const DIV_PAIRS: [(i16, i16); 12] = [
        (0, 1),
        (0, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (3, 4),
        (1, i16::MAX),
        (i16::MAX, i16::MAX),
        (16384, 32767),
        (12345, 23456),
        (1, 32767),
    ];

    // ── saturation primitives ─────────────────────────────────────────────

    #[test]
    fn saturation_primitives_clamp_at_the_word_boundaries() {
        assert_eq!(sat_l(0x1_0000_0000), MAX_32);
        assert_eq!(sat_l(-0x1_0000_0000), MIN_32);
        assert_eq!(sat_l(MAX_32 as i64), MAX_32);
        assert_eq!(sat_l(MIN_32 as i64), MIN_32);
        assert_eq!(sat_l(MAX_32 as i64 + 1), MAX_32);
        assert_eq!(sat_l(MIN_32 as i64 - 1), MIN_32);
        assert_eq!(sat_l(0), 0);

        assert_eq!(sat_s(32768), MAX_16);
        assert_eq!(sat_s(-32769), MIN_16);
        assert_eq!(sat_s(32767), MAX_16);
        assert_eq!(sat_s(-32768), MIN_16);
        assert_eq!(sat_s(i32::MAX), MAX_16);
        assert_eq!(sat_s(i32::MIN), MIN_16);
    }

    #[test]
    fn add_sub_negate_saturate() {
        assert_eq!(add(MAX_16, 1), MAX_16);
        assert_eq!(add(MAX_16, MAX_16), MAX_16);
        assert_eq!(add(MIN_16, -1), MIN_16);
        assert_eq!(add(MIN_16, MIN_16), MIN_16);
        assert_eq!(add(MAX_16, MIN_16), -1);

        assert_eq!(sub(MIN_16, 1), MIN_16);
        assert_eq!(sub(MAX_16, -1), MAX_16);
        // 0 - (-32768) = 32768, one past the positive limit.
        assert_eq!(sub(0, MIN_16), MAX_16);
        assert_eq!(sub(MAX_16, MAX_16), 0);

        // G.191 negate(-32768) is 32767, NOT -32768: the asymmetry of two's
        // complement is resolved toward the positive limit.
        assert_eq!(negate(MIN_16), MAX_16);
        assert_eq!(negate(MAX_16), -32767);
        assert_eq!(negate(0), 0);
        assert_eq!(negate(1), -1);
    }

    #[test]
    fn extract_and_deposit_are_exact_halves() {
        assert_eq!(extract_h(0x1234_5678), 0x1234);
        assert_eq!(extract_h(-1), -1);
        assert_eq!(extract_h(MIN_32), MIN_16);
        assert_eq!(extract_h(MAX_32), MAX_16);
        // extract_h truncates toward -inf (arithmetic shift), it does not
        // round: everything below 0x1_0000 is 0, and -1 stays -1.
        assert_eq!(extract_h(0xffff), 0);
        assert_eq!(extract_h(-0x1_0000), -1);

        assert_eq!(extract_l(0x1234_5678), 0x5678);
        // The low half is reinterpreted as signed, so 0x8000 comes back
        // negative.
        assert_eq!(extract_l(0x8000), MIN_16);
        assert_eq!(extract_l(-1), -1);
        assert_eq!(extract_l(MAX_32), -1);
        assert_eq!(extract_l(MIN_32), 0);

        assert_eq!(l_deposit_h(1), 0x1_0000);
        assert_eq!(l_deposit_h(MIN_16), MIN_32);
        assert_eq!(l_deposit_h(-1), -0x1_0000);
        assert_eq!(l_deposit_h(MAX_16), 0x7fff_0000);

        // l_deposit_l SIGN-extends; a zero-extending version would give 32768.
        assert_eq!(l_deposit_l(MIN_16), -32768);
        assert_eq!(l_deposit_l(MAX_16), 32767);
        assert_eq!(l_deposit_l(-1), -1);
    }

    #[test]
    fn shr_and_shl_saturate_and_floor() {
        // shr is an arithmetic (floor) shift, so it never rounds toward zero.
        assert_eq!(shr(-1, 1), -1);
        assert_eq!(shr(-3, 1), -2);
        assert_eq!(shr(3, 1), 1);
        // Counts of 15 or more collapse to the sign.
        assert_eq!(shr(1, 15), 0);
        assert_eq!(shr(-1, 15), -1);
        assert_eq!(shr(MIN_16, 15), -1);
        assert_eq!(shr(MAX_16, 15), 0);
        assert_eq!(shr(MIN_16, 31), -1);
        // A negative count delegates to shl.
        assert_eq!(shr(4, -2), 16);
        assert_eq!(shr(MAX_16, -1), MAX_16);

        // shl saturates rather than wrapping.
        assert_eq!(shl(1, 14), 16384);
        assert_eq!(shl(2, 14), MAX_16);
        assert_eq!(shl(-2, 14), MIN_16);
        assert_eq!(shl(-1, 14), -16384);
        // At 15 and beyond the value is replaced by the signed limit, except
        // that zero stays zero.
        assert_eq!(shl(1, 15), MAX_16);
        assert_eq!(shl(-1, 15), MIN_16);
        assert_eq!(shl(0, 15), 0);
        assert_eq!(shl(0, 31), 0);
        assert_eq!(shl(MIN_16, 31), MIN_16);
        // A negative count delegates to shr.
        assert_eq!(shl(4, -2), 1);
        assert_eq!(shl(-1, -1), -1);
    }

    /// Documented deviation from ITU-T G.191.
    ///
    /// The reference `shr`/`shl` clamp a negative count to -16 before
    /// delegating; this pair does not. The clamp is unobservable for every
    /// count reachable here — both routes end in the same
    /// "count >= 15 => saturate/collapse" arm — which is why the omission is
    /// pinned rather than fixed.
    ///
    /// It is observable at exactly one input: `i16::MIN`, where negating the
    /// count cannot be represented and the two functions recurse into each
    /// other. That input is unreachable from the decode path (every call site
    /// passes a `norm_s`/`norm_l` result or a small constant) and is left
    /// unprobed here rather than silently "fixed" in shipping code.
    #[test]
    fn shr_negative_count_is_not_clamped() {
        for &a in &W16 {
            for &b in &[-17i16, -20, -32] {
                assert_eq!(shr(a, b), shl(a, -b), "shr({a}, {b})");
                assert_eq!(shl(a, b), shr(a, -b), "shl({a}, {b})");
            }
        }
        // Same answer as a G.191 build that clamps the count to -16.
        assert_eq!(shr(1, -32), MAX_16);
        assert_eq!(shr(1, -16), MAX_16);
        assert_eq!(shr(-1, -32), MIN_16);
        assert_eq!(shr(0, -32), 0);
    }

    #[test]
    fn mult_floors_and_l_mult_doubles() {
        // The only product that overflows Q15 is MIN_16 * MIN_16 (+1.0
        // exactly). G.191 reaches MAX_16 there by saturating; this code takes
        // an explicit branch to the same value. Without it the `as i16` cast
        // would wrap 32768 to MIN_16 — a full-scale sign flip.
        assert_eq!(mult(MIN_16, MIN_16), MAX_16);
        assert_eq!(l_mult(MIN_16, MIN_16), MAX_32);

        // mult truncates toward -inf, so a negative product never rounds up
        // to zero.
        assert_eq!(mult(1, 1), 0);
        assert_eq!(mult(-1, 1), -1);
        assert_eq!(mult(1, -1), -1);
        assert_eq!(mult(MIN_16, 1), -1);
        assert_eq!(mult(16384, 16384), 8192);
        assert_eq!(mult(MAX_16, MAX_16), 32766);
        assert_eq!(mult(MIN_16, MAX_16), -32767);
        assert_eq!(mult(0, MIN_16), 0);

        assert_eq!(l_mult(1, 1), 2);
        assert_eq!(l_mult(-1, 1), -2);
        assert_eq!(l_mult(MAX_16, MAX_16), 2_147_352_578);
        assert_eq!(l_mult(MIN_16, MAX_16), -2_147_418_112);
        assert_eq!(l_mult(0, MIN_16), 0);
    }

    #[test]
    fn mult_r_rounds_half_up() {
        // The bias is +0x4000 before the >>15, so the tie breaks toward +inf
        // on both signs — NOT away from zero.
        assert_eq!(mult_r(1, 16384), 1); // exactly +0.5 -> +1
        assert_eq!(mult_r(1, 16383), 0); // just under -> 0
        assert_eq!(mult_r(-1, 16384), 0); // exactly -0.5 -> 0
        assert_eq!(mult_r(-1, 16385), -1); // just past -> -1
        assert_eq!(mult_r(MIN_16, MIN_16), MAX_16);
        assert_eq!(mult_r(MAX_16, MAX_16), 32766);
        assert_eq!(mult_r(MIN_16, MAX_16), -32767);
        assert_eq!(mult_r(0, 0), 0);
        assert_eq!(mult_r(16384, 16384), 8192);
    }

    #[test]
    fn long_add_sub_mac_saturate() {
        assert_eq!(l_add(MAX_32, 1), MAX_32);
        assert_eq!(l_add(MAX_32, MAX_32), MAX_32);
        assert_eq!(l_add(MIN_32, -1), MIN_32);
        assert_eq!(l_add(MAX_32, MIN_32), -1);
        assert_eq!(l_sub(MIN_32, 1), MIN_32);
        assert_eq!(l_sub(MAX_32, MIN_32), MAX_32);
        assert_eq!(l_sub(0, MIN_32), MAX_32);

        assert_eq!(l_mac(0, MIN_16, MIN_16), MAX_32);
        assert_eq!(l_mac(MAX_32, 1, 1), MAX_32);
        assert_eq!(l_mac(0, 3, 5), 30);
        // l_msu(0, MIN_16, MIN_16) subtracts the saturated MAX_32, landing one
        // short of MIN_32 rather than on it.
        assert_eq!(l_msu(0, MIN_16, MIN_16), -2_147_483_647);
        assert_eq!(l_msu(MIN_32, 1, 1), MIN_32);
        assert_eq!(l_msu(0, 3, 5), -30);
    }

    #[test]
    fn long_shifts_saturate_and_floor() {
        assert_eq!(l_shr(-1, 1), -1);
        assert_eq!(l_shr(-3, 1), -2);
        assert_eq!(l_shr(MIN_32, 1), -0x4000_0000);
        // 31 or more collapses to the sign.
        assert_eq!(l_shr(1, 31), 0);
        assert_eq!(l_shr(-1, 31), -1);
        assert_eq!(l_shr(MAX_32, 31), 0);
        assert_eq!(l_shr(MIN_32, 32), -1);
        assert_eq!(l_shr(8, -1), 16);

        // l_shl saturates one bit before the shift would overflow.
        assert_eq!(l_shl(0x3fff_ffff, 1), 0x7fff_fffe);
        assert_eq!(l_shl(0x4000_0000, 1), MAX_32);
        assert_eq!(l_shl(1, 30), 0x4000_0000);
        assert_eq!(l_shl(1, 31), MAX_32);
        // The negative threshold is asymmetric: -0x4000_0000 shifts exactly
        // onto MIN_32 without tripping the guard, and only then saturates.
        assert_eq!(l_shl(-0x4000_0000, 1), MIN_32);
        assert_eq!(l_shl(MIN_32, 1), MIN_32);
        assert_eq!(l_shl(-1, 31), MIN_32);
        assert_eq!(l_shl(0, 31), 0);
        assert_eq!(l_shl(-8, -1), -4);
        assert_eq!(l_shl(12345, 0), 12345);
    }

    #[test]
    fn l_shr_r_rounds_half_up() {
        assert_eq!(l_shr_r(3, 1), 2);
        assert_eq!(l_shr_r(2, 1), 1);
        assert_eq!(l_shr_r(1, 1), 1);
        // -0.5 rounds to 0 and -1.5 rounds to -1: the bias is toward +inf,
        // matching mult_r rather than "away from zero".
        assert_eq!(l_shr_r(-1, 1), 0);
        assert_eq!(l_shr_r(-3, 1), -1);
        assert_eq!(l_shr_r(-2, 1), -1);
        assert_eq!(l_shr_r(12345, 0), 12345);
        // Past 31 the result is 0 even for a negative input, where l_shr
        // alone would answer -1.
        assert_eq!(l_shr_r(MIN_32, 32), 0);
        assert_eq!(l_shr(MIN_32, 32), -1);
        assert_eq!(l_shr_r(MIN_32, 31), -1);
        assert_eq!(l_shr_r(-1, 31), 0);
        assert_eq!(l_shr_r(MAX_32, 31), 1);
    }

    #[test]
    fn norm_s_and_norm_l_count_redundant_sign_bits() {
        assert_eq!(norm_s(0), 0);
        assert_eq!(norm_s(-1), 15);
        assert_eq!(norm_s(1), 14);
        assert_eq!(norm_s(-2), 14);
        assert_eq!(norm_s(0x4000), 0);
        assert_eq!(norm_s(0x3fff), 1);
        assert_eq!(norm_s(-16384), 1);
        assert_eq!(norm_s(MIN_16), 0);
        assert_eq!(norm_s(MAX_16), 0);

        assert_eq!(norm_l(0), 0);
        assert_eq!(norm_l(-1), 31);
        assert_eq!(norm_l(1), 30);
        assert_eq!(norm_l(-2), 30);
        assert_eq!(norm_l(0x4000_0000), 0);
        assert_eq!(norm_l(0x3fff_ffff), 1);
        assert_eq!(norm_l(MIN_32), 0);
        assert_eq!(norm_l(MAX_32), 0);

        // The defining property: shifting left by norm_l(x) is the largest
        // shift that l_shl performs without saturating.
        for &x in &W32 {
            let n = norm_l(x);
            assert_eq!(
                l_shr(l_shl(x, n), n),
                x,
                "norm_l({x:#x}) = {n} is not loss-free"
            );
        }
    }

    #[test]
    fn div_s_is_q15_with_truncation() {
        assert_eq!(div_s(0, 1), 0);
        // The zero test precedes the equality test, so 0/0 is 0 and not
        // MAX_16.
        assert_eq!(div_s(0, 0), 0);
        assert_eq!(div_s(5, 5), MAX_16);
        assert_eq!(div_s(1, 2), 16384);
        assert_eq!(div_s(1, 4), 8192);
        assert_eq!(div_s(3, 4), 24576);
        // 15 restoring steps truncate: 32768/3 = 10922.67 -> 10922.
        assert_eq!(div_s(1, 3), 10922);
        assert_eq!(div_s(1, MAX_16), 1);
        assert_eq!(div_s(16384, 32767), 16384);
    }

    // ── sweeps ────────────────────────────────────────────────────────────

    /// Re-bless by running with `--nocapture` and pasting the printed block.
    const SWEEPS: [(&str, u64); 17] = [
        ("add/sub", 0x5c0d5ff71f0a1580),
        ("negate", 0x83ff6071ebaa60aa),
        ("extract/deposit", 0x510d5d829478f044),
        ("shr", 0x9d59c6e79bddea5b),
        ("shl", 0x3bab1f9847252277),
        ("mult", 0x6145406cddac8b0d),
        ("l_mult", 0x4985521c1412bffd),
        ("mult_r", 0x1bd3b553f921fc19),
        ("l_add", 0x3c3918e26e5f8fdf),
        ("l_sub", 0xa5a5d09577a413ff),
        ("l_mac", 0x24fd46b681ce4ce4),
        ("l_msu", 0x69d94ab46ed6bf24),
        ("l_shr", 0xcd239363b44a912c),
        ("l_shl", 0x676430319bfba4a3),
        ("l_shr_r", 0x109e2c006a6abc68),
        ("norm_s/norm_l", 0x9ca5ecfc608f51aa),
        ("div_s", 0x995e3a3e1c6a98a5),
    ];

    #[test]
    fn basic_operator_sweeps_are_pinned() {
        fn push(name: &'static str, v: &mut Vec<i64>, acc: &mut Vec<(&'static str, u64)>) {
            acc.push((name, hash_i64s(v)));
            v.clear();
        }

        let mut out: Vec<i64> = Vec::new();
        let mut actual: Vec<(&'static str, u64)> = Vec::new();

        for &a in &W16 {
            for &b in &W16 {
                out.push(add(a, b) as i64);
                out.push(sub(a, b) as i64);
            }
        }
        push("add/sub", &mut out, &mut actual);
        for &a in &W16 {
            out.push(negate(a) as i64);
        }
        push("negate", &mut out, &mut actual);
        for &l in &W32 {
            out.push(extract_h(l) as i64);
            out.push(extract_l(l) as i64);
        }
        for &a in &W16 {
            out.push(l_deposit_h(a) as i64);
            out.push(l_deposit_l(a) as i64);
        }
        push("extract/deposit", &mut out, &mut actual);
        for &a in &W16 {
            for &b in &SB {
                out.push(shr(a, b) as i64);
            }
        }
        push("shr", &mut out, &mut actual);
        for &a in &W16 {
            for &b in &SB {
                out.push(shl(a, b) as i64);
            }
        }
        push("shl", &mut out, &mut actual);
        for &a in &W16 {
            for &b in &W16 {
                out.push(mult(a, b) as i64);
            }
        }
        push("mult", &mut out, &mut actual);
        for &a in &W16 {
            for &b in &W16 {
                out.push(l_mult(a, b) as i64);
            }
        }
        push("l_mult", &mut out, &mut actual);
        for &a in &W16 {
            for &b in &W16 {
                out.push(mult_r(a, b) as i64);
            }
        }
        push("mult_r", &mut out, &mut actual);
        for &a in &W32 {
            for &b in &W32 {
                out.push(l_add(a, b) as i64);
            }
        }
        push("l_add", &mut out, &mut actual);
        for &a in &W32 {
            for &b in &W32 {
                out.push(l_sub(a, b) as i64);
            }
        }
        push("l_sub", &mut out, &mut actual);
        for &acc32 in &W32 {
            for &a in &W16 {
                out.push(l_mac(acc32, a, 0x4321) as i64);
            }
        }
        push("l_mac", &mut out, &mut actual);
        for &acc32 in &W32 {
            for &a in &W16 {
                out.push(l_msu(acc32, a, 0x4321) as i64);
            }
        }
        push("l_msu", &mut out, &mut actual);
        for &l in &W32 {
            for &b in &SB32 {
                out.push(l_shr(l, b) as i64);
            }
        }
        push("l_shr", &mut out, &mut actual);
        for &l in &W32 {
            for &b in &SB32 {
                out.push(l_shl(l, b) as i64);
            }
        }
        push("l_shl", &mut out, &mut actual);
        for &l in &W32 {
            for &b in &SB32 {
                out.push(l_shr_r(l, b) as i64);
            }
        }
        push("l_shr_r", &mut out, &mut actual);
        for &a in &W16 {
            out.push(norm_s(a) as i64);
        }
        for &l in &W32 {
            out.push(norm_l(l) as i64);
        }
        push("norm_s/norm_l", &mut out, &mut actual);
        for &(a, b) in &DIV_PAIRS {
            out.push(div_s(a, b) as i64);
        }
        push("div_s", &mut out, &mut actual);

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
            "a G.191 basic operator changed:\n  {}\n\nIMBE decode is bit-exact \
             only while these are; a saturation bound or rounding direction \
             moving here changes decoded audio on every frame.",
            bad.join("\n  ")
        );
    }
}
