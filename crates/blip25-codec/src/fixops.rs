//! Exact 32-bit two's-complement arithmetic primitives shared by the
//! fixed-point encoder/decoder ports: masked shift counts, 16-bit
//! sign-extension round-trips, block-float normalize helpers.
//!
//! Several helper names (`s32`, `s16`, `sar`, `shl`, `bsr`, `sar32`) exist in
//! more than one signature/semantics cluster across the ports, so the shared
//! copies are grouped into submodules by dialect; each consumer file imports
//! exactly the cluster it was written against. Bodies are character-exact
//! transcriptions of the arithmetic they model — do NOT "simplify" or reorder
//! anything here.

/// The i64-accumulator dialect: 32-bit x86 ops modeled on `i64` intermediates
/// (mask-and-truncate), plus the block-float normalize/denormalize helpers
/// built on them.
pub(crate) mod acc64 {
    #[inline]
    pub(crate) fn m32(x: i64) -> i64 {
        x & 0xffff_ffff
    }
    #[inline]
    pub(crate) fn s32(x: i64) -> i32 {
        (x & 0xffff_ffff) as u32 as i32
    }
    #[inline]
    pub(crate) fn i16s(x: i64) -> i32 {
        (x & 0xffff) as u16 as i16 as i32
    }
    #[inline]
    pub(crate) fn sar(x: i32, n: i32) -> i32 {
        x >> (n & 31)
    }
    #[inline]
    pub(crate) fn shl(x: i32, n: i32) -> i32 {
        ((x as u32).wrapping_shl((n & 31) as u32)) as i32
    }
    #[inline]
    pub(crate) fn bsr(x: i32) -> i32 {
        if x == 0 {
            0
        } else {
            31 - (x as u32).leading_zeros() as i32
        }
    }
    #[inline]
    pub(crate) fn bshift(v: i32, sh: i32) -> i32 {
        let sh = i16s(sh as i64);
        if sh >= 0 {
            shl(v, sh)
        } else if sh > -31 {
            sar(v, -sh)
        } else {
            sar(v, 31)
        }
    }
    /// Signed saturating add (a + b), clamped on overflow.
    pub(crate) fn sat_add(a: i32, b: i32) -> i32 {
        a.saturating_add(b)
    }
    /// Normalizing shift: index of the top set bit of |v|, then 0x1e minus it.
    pub(crate) fn norm_shift(v: i32) -> i32 {
        if v == 0 {
            return 0;
        }
        let t = if v >= 0 { v } else { !v };
        (0x1e - bsr(t)) & 0xffff
    }
    /// Block-float apply: shift `v` by i16 `sh` (left if >=0).
    pub(crate) fn bf_shift(v: i32, sh: i32) -> i32 {
        let sh = i16s(sh as i64);
        if sh >= 0 {
            shl(v, sh & 0xff)
        } else if sh > -31 {
            sar(v, (-sh) & 0xff)
        } else {
            sar(v, 31)
        }
    }
    pub(crate) fn norm_edx(edx: i32) -> (i32, i32) {
        let sh = if edx == 0 {
            0
        } else {
            let t = if edx >= 0 { edx } else { s32(!(edx as i64)) };
            (0x1e - bsr(t)) & 0xffff
        };
        let expc = i16s(((0xf - sh) & 0xffff) as i64);
        let mant = i16s((sar(shl(edx, sh), 16) & 0xffff) as i64);
        (mant, expc)
    }
    pub(crate) fn denorm(mant: i32, expc: i32) -> i32 {
        let sh = i16s(((expc - 6) & 0xffff) as i64);
        let v = shl(i16s(mant as i64), 16);
        let v = sar(bshift(v, sh), 16);
        i16s((v & 0xffff) as i64)
    }
    pub(crate) fn norm_sh(edx: i32) -> i32 {
        if edx == 0 {
            0
        } else {
            let t = if edx >= 0 { edx } else { s32(!(edx as i64)) };
            (0x1e - bsr(t)) & 0xffff
        }
    }
    #[inline]
    pub(crate) fn s16m(x: i64) -> i64 {
        let x = x & 0xffff;
        if x >= 0x8000 {
            x - 0x10000
        } else {
            x
        }
    }
    #[inline]
    pub(crate) fn sat16(x: i64) -> i16 {
        x.clamp(-32768, 32767) as i16
    }
    #[inline]
    pub(crate) fn s16(x: i32) -> i16 {
        (x & 0xffff) as u16 as i16
    }
}

/// The u32-register dialect: values carried as `u32` bit patterns with
/// explicit signed/unsigned view-change helpers.
pub(crate) mod u32r {
    #[inline]
    pub(crate) fn s32(x: u32) -> i32 {
        x as i32
    }
    #[inline]
    pub(crate) fn u32v(x: i32) -> u32 {
        x as u32
    }
    #[inline]
    pub(crate) fn sar32(x: u32, n: u32) -> u32 {
        u32v(s32(x) >> (n & 0x1f))
    }
    #[inline]
    pub(crate) fn shl32(x: u32, n: u32) -> u32 {
        x.wrapping_shl(n & 0x1f)
    }
    #[inline]
    pub(crate) fn imul32(a: i32, b: i32) -> u32 {
        u32v(a.wrapping_mul(b))
    }
    #[inline]
    pub(crate) fn s16(x: u16) -> i16 {
        x as i16
    }
    #[inline]
    pub(crate) fn u16v(x: u32) -> u16 {
        x as u16
    }
    #[inline]
    pub(crate) fn bsr32(x: u32) -> u32 {
        // x must be nonzero -- bit-scan-reverse is undefined on zero; callers
        // here only invoke it on values already proven nonzero.
        debug_assert!(x != 0);
        31 - x.leading_zeros()
    }
}

/// The i32-register dialect: plain-cast truncation/widening markers and the
/// unguarded `bsr` (returns -1 for 0; callers guard).
pub(crate) mod i32r {
    #[inline]
    pub(crate) fn s32(x: i32) -> i32 {
        x
    }
    #[inline]
    pub(crate) fn s16(x: i32) -> i16 {
        x as i16
    }
    #[inline]
    pub(crate) fn sar32(x: i32, n: u32) -> i32 {
        x >> (n & 0x1f)
    }
    #[inline]
    pub(crate) fn bsr(x: i32) -> i32 {
        31 - (x as u32).leading_zeros() as i32
    }
    #[inline]
    pub(crate) fn i16w(x: i32) -> i32 {
        x as i16 as i32
    }
}

/// The i16-source dialect: `i16` inputs widened/truncated through `i32`.
pub(crate) mod i16r {
    #[inline]
    pub(crate) fn i16t(x: i32) -> i16 {
        x as i16
    }
    #[inline]
    pub(crate) fn s32(x: i16) -> i32 {
        x as i32
    }
}

/// The decoder-side dialect: `u32` shift counts masked to 5 bits, and the
/// `movsx`-style low-16 sign extension.
pub(crate) mod dec32 {
    /// Sign-extend the low 16 bits.
    #[inline]
    pub(crate) fn sx16(v: i32) -> i32 {
        v as u16 as i16 as i32
    }
    #[inline]
    pub(crate) fn shl(v: i32, s: u32) -> i32 {
        ((v as u32).wrapping_shl(s & 31)) as i32
    }
    #[inline]
    pub(crate) fn sar(v: i32, s: u32) -> i32 {
        v >> (s & 31)
    }
}

#[cfg(test)]
mod tests {
    //! Per-dialect pins for the five same-named-helper clusters above.
    //!
    //! Two kinds of assertion live here, and they do different jobs:
    //!
    //! * **Reasoned literals** — hand-derived from the bodies above, covering
    //!   `i32::MIN`, `i32::MAX`, `0`, `-1` and every shift count that lands on
    //!   a masking or branch edge. These are the bisection targets: a failure
    //!   names one operator.
    //! * **Sweep hashes** — FNV-1a over each operator's output across a fixed
    //!   probe grid, so a change outside the hand-picked points still fails.
    //!   FNV is hand-inlined for the same reason it is in
    //!   `blip25-mbe/tests/golden_output.rs`: no dependency may move a hash.
    //!
    //! The module header forbids "simplifying" or merging anything in this
    //! file. `bsr_zero_case_differs_between_dialects` and the `*_dialects`
    //! tests make that enforceable instead of advisory: they pin the points at
    //! which two identically-named helpers actually disagree.

    use super::*;

    // ── hashing ───────────────────────────────────────────────────────────

    /// FNV-1a 64, defined here rather than pulled in so the sweep gates have
    /// no dependency that could itself change a hash.
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

    // ── probe grids (fixed, integer-only, platform-independent) ────────────

    /// 32-bit probes: identities, small magnitudes, the 16-bit boundary, the
    /// sign-bit boundary and both extremes.
    const P32: [i32; 24] = [
        0,
        1,
        -1,
        2,
        -2,
        3,
        -3,
        7,
        -7,
        0x7f,
        -0x80,
        0x7fff,
        -0x8000,
        0x1_0000,
        0x1_8000,
        0x1234_5678,
        -0x1234_5678,
        0x4000_0000,
        -0x4000_0000,
        0x5555_5555,
        -0x5555_5556,
        i32::MAX,
        i32::MIN,
        -0x7fff_ffff,
    ];

    /// 64-bit probes for the accumulator-dialect narrowing helpers: values
    /// that straddle every mask this file applies (0xffff, 0xffff_ffff).
    const P64: [i64; 16] = [
        0,
        1,
        -1,
        0x7fff,
        -0x8000,
        0x8000,
        0xffff,
        0x1_0000,
        0x1_8000,
        0x7fff_ffff,
        -0x8000_0000,
        0x8000_0000,
        0xffff_ffff,
        0x1_0000_0000,
        0x1_2345_6789,
        -0x1_2345_6789,
    ];

    /// Signed shift counts: negatives (which the `& 31` mask folds back into
    /// range), the 15/16 half-word edges, and 31/32 where the mask wraps.
    const SH_I: [i32; 14] = [-64, -33, -32, -31, -30, -16, -1, 0, 1, 15, 16, 30, 31, 32];

    /// Unsigned shift counts, including several past 31 so the `& 0x1f` /
    /// `& 31` masks are exercised rather than assumed.
    const SH_U: [u32; 12] = [0, 1, 2, 15, 16, 30, 31, 32, 33, 47, 63, 64];

    /// `denorm` mantissas: i16-range endpoints plus the block-float midpoint.
    const MANTISSAS: [i32; 9] = [0, 1, -1, 0x4000, -0x4000, 0x7fff, -0x8000, 1234, -1234];

    /// `denorm` exponents, straddling the unity point (6) in both directions
    /// and reaching the 16-bit fold at 0x8000.
    const EXPONENTS: [i32; 11] = [-31, -15, -1, 0, 5, 6, 7, 15, 22, 37, 0x8000];

    // ── acc64: reasoned literals ──────────────────────────────────────────

    #[test]
    fn acc64_narrowing_helpers() {
        // m32 keeps the low 32 bits as a *non-negative* i64 — it is a mask,
        // not a sign-extending truncation.
        assert_eq!(acc64::m32(-1), 0xffff_ffff);
        assert_eq!(acc64::m32(0x1_2345_6789), 0x2345_6789);
        assert_eq!(acc64::m32(0x1_0000_0000), 0);

        // s32 truncates to 32 bits and *reinterprets* as signed.
        assert_eq!(acc64::s32(0xffff_ffff), -1);
        assert_eq!(acc64::s32(0x8000_0000), i32::MIN);
        assert_eq!(acc64::s32(0x1_0000_0000), 0);
        assert_eq!(acc64::s32(0x1_2345_6789), 0x2345_6789);
        assert_eq!(acc64::s32(-1), -1);

        // i16s takes the low 16 bits and sign-extends them to i32.
        assert_eq!(acc64::i16s(0x8000), -32768);
        assert_eq!(acc64::i16s(0xffff), -1);
        assert_eq!(acc64::i16s(0x1_7fff), 32767);
        assert_eq!(acc64::i16s(0x1_8000), -32768);
        assert_eq!(acc64::i16s(-1), -1);

        // s16m is i16s's i64-valued twin: same fold, different return type.
        assert_eq!(acc64::s16m(0x7fff), 32767);
        assert_eq!(acc64::s16m(0x8000), -32768);
        assert_eq!(acc64::s16m(0xffff), -1);
        assert_eq!(acc64::s16m(0x1_8000), -32768);
        assert_eq!(acc64::s16m(-1), -1);

        // sat16 CLAMPS where s16m/i16s WRAP — the pair is the reason both
        // exist, so pin them on the same input.
        assert_eq!(acc64::sat16(32768), 32767);
        assert_eq!(acc64::sat16(-32769), -32768);
        assert_eq!(acc64::sat16(i64::MAX), 32767);
        assert_eq!(acc64::sat16(i64::MIN), -32768);
        assert_eq!(acc64::sat16(0x8000), 32767);
        assert_ne!(acc64::sat16(0x8000) as i64, acc64::s16m(0x8000));

        // s16 wraps the low 16 bits of an i32.
        assert_eq!(acc64::s16(0x1_8000), -32768);
        assert_eq!(acc64::s16(-1), -1);
        assert_eq!(acc64::s16(i32::MIN), 0);
        assert_eq!(acc64::s16(i32::MAX), -1);
    }

    #[test]
    fn acc64_shifts_mask_the_count_to_five_bits() {
        // sar is arithmetic (floors toward -inf), shl is a wrapping logical
        // left shift — neither saturates.
        assert_eq!(acc64::sar(-1, 1), -1);
        assert_eq!(acc64::sar(-3, 1), -2);
        assert_eq!(acc64::sar(i32::MIN, 31), -1);
        assert_eq!(acc64::sar(i32::MAX, 31), 0);
        // count 32 masks to 0 => identity, NOT "shifted away".
        assert_eq!(acc64::sar(i32::MIN, 32), i32::MIN);
        // negative counts fold: -1 & 31 == 31, -32 & 31 == 0.
        assert_eq!(acc64::sar(i32::MIN, -1), -1);
        assert_eq!(acc64::sar(0x1234_5678, -32), 0x1234_5678);

        assert_eq!(acc64::shl(1, 31), i32::MIN);
        assert_eq!(acc64::shl(1, 32), 1);
        assert_eq!(acc64::shl(i32::MIN, 1), 0);
        assert_eq!(acc64::shl(i32::MAX, 1), -2);
        assert_eq!(acc64::shl(1, -1), i32::MIN);
        assert_eq!(acc64::shl(-1, 31), i32::MIN);
    }

    #[test]
    fn acc64_bshift_and_bf_shift() {
        // The shift amount is first folded through a 16-bit sign extension,
        // so 0x1_0000 means "shift by 0" and 0x8000 means "shift by -32768".
        assert_eq!(acc64::bshift(0x1234, 0x1_0000), 0x1234);
        assert_eq!(acc64::bshift(-8, -1), -4);
        assert_eq!(acc64::bshift(-8, -30), -1);
        // -31 is NOT "> -31", so it takes the >>31 arm.
        assert_eq!(acc64::bshift(-8, -31), -1);
        assert_eq!(acc64::bshift(8, -31), 0);
        assert_eq!(acc64::bshift(-8, 0x8000), -1);
        // sh >= 0 goes through shl, whose count is masked to 5 bits.
        assert_eq!(acc64::bshift(1, 32), 1);
        assert_eq!(acc64::bshift(1, 31), i32::MIN);

        // bf_shift differs from bshift only by an extra `& 0xff` on the count
        // before shl/sar mask it again to 5 bits — which is a no-op, so the
        // two agree on every probe. Pinned so that stops being an assumption.
        for &v in &P32 {
            for &sh in &SH_I {
                assert_eq!(
                    acc64::bshift(v, sh),
                    acc64::bf_shift(v, sh),
                    "bshift/bf_shift diverged at v={v:#x} sh={sh}"
                );
            }
        }
        assert_eq!(acc64::bf_shift(1, 32), 1);
        assert_eq!(acc64::bf_shift(-8, -31), -1);
    }

    #[test]
    fn acc64_sat_add_clamps() {
        assert_eq!(acc64::sat_add(i32::MAX, 1), i32::MAX);
        assert_eq!(acc64::sat_add(i32::MAX, i32::MAX), i32::MAX);
        assert_eq!(acc64::sat_add(i32::MIN, -1), i32::MIN);
        assert_eq!(acc64::sat_add(i32::MIN, i32::MIN), i32::MIN);
        assert_eq!(acc64::sat_add(i32::MAX, i32::MIN), -1);
        assert_eq!(acc64::sat_add(0, 0), 0);
    }

    #[test]
    fn acc64_normalize_helpers() {
        // norm_shift(v) = 0x1e - bsr(|v| via ones-complement for negatives),
        // masked to 16 bits. It rests on bsr(0) == 0, so the -1 and -2 cases
        // below double as pins on that zero convention.
        assert_eq!(acc64::norm_shift(0), 0);
        assert_eq!(acc64::norm_shift(1), 30);
        assert_eq!(acc64::norm_shift(2), 29);
        // !(-1) == 0, so this reaches bsr(0); acc64's bsr answers 0 => 30.
        assert_eq!(acc64::norm_shift(-1), 30);
        assert_eq!(acc64::norm_shift(-2), 30);
        assert_eq!(acc64::norm_shift(i32::MAX), 0);
        assert_eq!(acc64::norm_shift(i32::MIN), 0);
        assert_eq!(acc64::norm_shift(0x4000_0000), 0);

        // norm_sh is norm_shift written with an explicit i64 ones-complement;
        // the two are value-identical on every probe.
        for &v in &P32 {
            assert_eq!(acc64::norm_shift(v), acc64::norm_sh(v), "at v={v:#x}");
        }

        // norm_edx: (mantissa, exponent) block-float split.
        assert_eq!(acc64::norm_edx(0), (0, 15));
        assert_eq!(acc64::norm_edx(1), (16384, -15));
        assert_eq!(acc64::norm_edx(-1), (-16384, -15));
        assert_eq!(acc64::norm_edx(i32::MAX), (32767, 15));
        assert_eq!(acc64::norm_edx(i32::MIN), (-32768, 15));

        // denorm: exponent 6 is the unity point (shift 0), so it is the
        // identity on any i16-valued mantissa.
        assert_eq!(acc64::denorm(1234, 6), 1234);
        assert_eq!(acc64::denorm(-1234, 6), -1234);
        assert_eq!(acc64::denorm(-32768, 6), -32768);
        assert_eq!(acc64::denorm(1, 7), 2);
        // Right shifts drop the low bits entirely rather than rounding.
        assert_eq!(acc64::denorm(1, 5), 0);
        assert_eq!(acc64::denorm(0x100, 5), 128);
        // A large left shift walks the mantissa off the top of the register.
        assert_eq!(acc64::denorm(32767, 22), 0);
        assert_eq!(acc64::denorm(1, 37), 0);
    }

    // ── u32r: reasoned literals ───────────────────────────────────────────

    #[test]
    fn u32r_view_changes_and_shifts() {
        assert_eq!(u32r::s32(0xffff_ffff), -1);
        assert_eq!(u32r::s32(0x8000_0000), i32::MIN);
        assert_eq!(u32r::s32(0x7fff_ffff), i32::MAX);
        assert_eq!(u32r::u32v(-1), 0xffff_ffff);
        assert_eq!(u32r::u32v(i32::MIN), 0x8000_0000);

        // sar32 is ARITHMETIC despite the u32 carrier: the sign bit
        // replicates. A logical shift would give 0x4000_0000 here.
        assert_eq!(u32r::sar32(0x8000_0000, 1), 0xc000_0000);
        assert_eq!(u32r::sar32(0x8000_0000, 31), 0xffff_ffff);
        assert_eq!(u32r::sar32(0xffff_ffff, 31), 0xffff_ffff);
        assert_eq!(u32r::sar32(0x7fff_ffff, 31), 0);
        // count 32 masks to 0.
        assert_eq!(u32r::sar32(0x8000_0000, 32), 0x8000_0000);
        assert_eq!(u32r::sar32(0x8000_0000, 33), 0xc000_0000);

        assert_eq!(u32r::shl32(1, 31), 0x8000_0000);
        assert_eq!(u32r::shl32(1, 32), 1);
        assert_eq!(u32r::shl32(0xffff_ffff, 4), 0xffff_fff0);
        assert_eq!(u32r::shl32(0x8000_0000, 1), 0);

        // imul32 wraps; it does not saturate and it does not widen.
        assert_eq!(u32r::imul32(-1, -1), 1);
        assert_eq!(u32r::imul32(i32::MIN, -1), 0x8000_0000);
        assert_eq!(u32r::imul32(0x1_0000, 0x1_0000), 0);
        assert_eq!(u32r::imul32(i32::MAX, 2), 0xffff_fffe);

        assert_eq!(u32r::s16(0xffff), -1);
        assert_eq!(u32r::s16(0x8000), -32768);
        assert_eq!(u32r::s16(0x7fff), 32767);
        assert_eq!(u32r::u16v(0x1234_5678), 0x5678);
        assert_eq!(u32r::u16v(0xffff_ffff), 0xffff);

        // bsr32 is the *unsigned* bit-scan-reverse. Its zero input is a
        // documented precondition (debug_assert), so it is not probed here;
        // see `bsr_zero_case_differs_between_dialects` for the two `bsr`s
        // that DO define zero, differently.
        assert_eq!(u32r::bsr32(1), 0);
        assert_eq!(u32r::bsr32(2), 1);
        assert_eq!(u32r::bsr32(3), 1);
        assert_eq!(u32r::bsr32(0x8000_0000), 31);
        assert_eq!(u32r::bsr32(0xffff_ffff), 31);
        assert_eq!(u32r::bsr32(0x7fff_ffff), 30);
    }

    // ── i32r: reasoned literals ───────────────────────────────────────────

    #[test]
    fn i32r_markers_and_shifts() {
        for &v in &P32 {
            assert_eq!(i32r::s32(v), v, "i32r::s32 must be the identity");
        }

        assert_eq!(i32r::s16(0x1_2345), 0x2345);
        assert_eq!(i32r::s16(-1), -1);
        assert_eq!(i32r::s16(0x8000), -32768);
        assert_eq!(i32r::s16(i32::MIN), 0);
        assert_eq!(i32r::s16(i32::MAX), -1);

        assert_eq!(i32r::sar32(-1, 1), -1);
        assert_eq!(i32r::sar32(-3, 1), -2);
        assert_eq!(i32r::sar32(i32::MIN, 31), -1);
        assert_eq!(i32r::sar32(i32::MAX, 31), 0);
        assert_eq!(i32r::sar32(i32::MIN, 32), i32::MIN);
        assert_eq!(i32r::sar32(i32::MIN, 33), -0x4000_0000);

        assert_eq!(i32r::bsr(1), 0);
        assert_eq!(i32r::bsr(2), 1);
        assert_eq!(i32r::bsr(i32::MAX), 30);
        // Negative inputs are scanned as unsigned, so the sign bit wins.
        assert_eq!(i32r::bsr(-1), 31);
        assert_eq!(i32r::bsr(i32::MIN), 31);

        assert_eq!(i32r::i16w(0x8000), -32768);
        assert_eq!(i32r::i16w(0x1_8000), -32768);
        assert_eq!(i32r::i16w(0x7fff), 32767);
        assert_eq!(i32r::i16w(-1), -1);
        assert_eq!(i32r::i16w(i32::MIN), 0);
    }

    // ── i16r: reasoned literals ───────────────────────────────────────────

    #[test]
    fn i16r_widen_and_truncate() {
        assert_eq!(i16r::i16t(0x1_2345), 0x2345);
        // -32769 is 0xffff_7fff: the low half is 0x7fff, so truncation flips
        // the sign. This is a wrap, not a clamp.
        assert_eq!(i16r::i16t(-32769), 32767);
        assert_eq!(i16r::i16t(32768), -32768);
        assert_eq!(i16r::i16t(i32::MIN), 0);
        assert_eq!(i16r::i16t(i32::MAX), -1);

        // s32 here SIGN-extends. A zero-extending version would give 65535.
        assert_eq!(i16r::s32(-1), -1);
        assert_eq!(i16r::s32(i16::MIN), -32768);
        assert_eq!(i16r::s32(i16::MAX), 32767);
    }

    // ── dec32: reasoned literals ──────────────────────────────────────────

    #[test]
    fn dec32_sign_extend_and_shifts() {
        assert_eq!(dec32::sx16(0x8000), -32768);
        assert_eq!(dec32::sx16(0x1_8000), -32768);
        assert_eq!(dec32::sx16(0x7fff), 32767);
        assert_eq!(dec32::sx16(-1), -1);
        assert_eq!(dec32::sx16(i32::MIN), 0);

        assert_eq!(dec32::shl(1, 31), i32::MIN);
        assert_eq!(dec32::shl(1, 32), 1);
        assert_eq!(dec32::shl(-1, 1), -2);
        // Wrapping, not saturating.
        assert_eq!(dec32::shl(i32::MIN, 1), 0);
        assert_eq!(dec32::shl(i32::MAX, 1), -2);

        assert_eq!(dec32::sar(i32::MIN, 31), -1);
        assert_eq!(dec32::sar(-1, 31), -1);
        assert_eq!(dec32::sar(-3, 1), -2);
        assert_eq!(dec32::sar(i32::MIN, 32), i32::MIN);
        assert_eq!(dec32::sar(i32::MAX, 31), 0);
    }

    // ── the points where identically-named helpers disagree ───────────────

    #[test]
    fn bsr_zero_case_differs_between_dialects() {
        // This is the divergence the module header exists to protect.
        // `acc64::bsr` is guarded and answers 0 for 0; `i32r::bsr` is the raw
        // instruction and answers -1 (31 - 32 leading zeros). Merging them in
        // either direction silently shifts every block-float exponent that
        // flows through `norm_shift`/`norm_sh`/`norm_edx`.
        assert_eq!(acc64::bsr(0), 0, "acc64::bsr must answer 0 for 0");
        assert_eq!(i32r::bsr(0), -1, "i32r::bsr must answer -1 for 0");
        assert_ne!(
            acc64::bsr(0),
            i32r::bsr(0),
            "the two bsr dialects have been unified — see the module header"
        );

        // Everywhere else the two agree, which is exactly why the zero case
        // is easy to lose. Pin the agreement so the divergence is known to be
        // a single point rather than assumed to be.
        for &v in &P32 {
            if v == 0 {
                continue;
            }
            assert_eq!(acc64::bsr(v), i32r::bsr(v), "at v={v:#x}");
        }
    }

    #[test]
    fn s32_has_four_distinct_dialects() {
        // Four `s32`s with four signatures. The compiler already rejects a
        // naive merge, but these pin the *behaviour* each domain relies on.
        assert_eq!(acc64::s32(0x1_0000_0000_i64), 0); // truncate 64 -> 32
        assert_eq!(u32r::s32(0x8000_0000_u32), i32::MIN); // reinterpret
        assert_eq!(i32r::s32(i32::MIN), i32::MIN); // identity marker
        assert_eq!(i16r::s32(i16::MIN), -32768); // sign-extend 16 -> 32

        // Same 32-bit pattern, three carriers, one answer — so the truncating
        // and reinterpreting versions cannot be told apart on a 32-bit input.
        // Only the 33rd bit separates them, which is what the first line does.
        assert_eq!(acc64::s32(0xffff_ffff_i64), -1);
        assert_eq!(u32r::s32(0xffff_ffff_u32), -1);
        assert_eq!(i32r::s32(-1), -1);
    }

    #[test]
    fn s16_dialects() {
        // `acc64::s16(i32)` and `i32r::s16(i32)` are value-identical for every
        // input — `(x & 0xffff) as u16 as i16` and `x as i16` are the same
        // function. They are kept apart by provenance, not by behaviour, so
        // no assertion here can catch a merge of those two; that is recorded
        // rather than papered over.
        for &v in &P32 {
            assert_eq!(acc64::s16(v), i32r::s16(v), "at v={v:#x}");
        }
        // `u32r::s16(u16)` is the one with a different domain.
        assert_eq!(u32r::s16(0x8000_u16), -32768);
        assert_eq!(acc64::s16(0x8000_i32), -32768);
        assert_eq!(i32r::s16(0x8000_i32), -32768);
    }

    #[test]
    fn sar32_and_shl_dialects() {
        // `u32r::sar32` and `i32r::sar32` differ only in carrier type; pin
        // that they agree bit-for-bit so a future change to one is visible.
        for &v in &P32 {
            for &n in &SH_U {
                assert_eq!(
                    u32r::sar32(v as u32, n),
                    i32r::sar32(v, n) as u32,
                    "sar32 dialects diverged at v={v:#x} n={n}"
                );
            }
        }
        // `acc64::shl` takes a SIGNED count and `dec32::shl` an unsigned one;
        // both mask to 5 bits, so they agree wherever both are defined.
        for &v in &P32 {
            for &n in &SH_U {
                assert_eq!(
                    acc64::shl(v, n as i32),
                    dec32::shl(v, n),
                    "shl dialects diverged at v={v:#x} n={n}"
                );
                assert_eq!(
                    acc64::sar(v, n as i32),
                    dec32::sar(v, n),
                    "sar dialects diverged at v={v:#x} n={n}"
                );
            }
        }
        // The signed count is where `acc64::shl` goes somewhere `dec32::shl`
        // cannot follow: -1 folds to 31.
        assert_eq!(acc64::shl(1, -1), i32::MIN);
        assert_eq!(acc64::sar(i32::MIN, -1), -1);
    }

    // ── sweeps ────────────────────────────────────────────────────────────

    /// FNV-1a over each operator's outputs across the fixed probe grids.
    ///
    /// Re-bless by running with `--nocapture` and pasting the printed block.
    /// One entry per helper, so a failure names a single function.
    ///
    /// Several hashes are expected to coincide, and the coincidences are
    /// themselves pinned facts rather than copy-paste slips:
    ///
    /// * `acc64::i16s` / `acc64::s16m` — the same 16-bit fold, different
    ///   return type.
    /// * `acc64::bshift` / `acc64::bf_shift` — agree on every probe (see
    ///   `acc64_bshift_and_bf_shift`).
    /// * `acc64::norm_shift` / `acc64::norm_sh` — the same function written
    ///   two ways.
    /// * `u32r::s32` / `i32r::s32` — a 32-bit reinterpret and the identity
    ///   agree on every 32-bit input; only a 33rd bit separates them, which
    ///   is `s32_has_four_distinct_dialects`'s job.
    /// * `acc64::s16`, `u32r::s16`, `i32r::s16`, `i32r::i16w`, `i16r::i16t`,
    ///   `i16r::s32`, `dec32::sx16` — every one of these reduces its input to
    ///   the sign-extended low 16 bits, so no VALUE can tell them apart. What
    ///   keeps them separate is their signatures, which the compiler enforces,
    ///   and their call sites. A sweep cannot protect these against being
    ///   merged; that limit is recorded here rather than papered over.
    /// * `i32r::sar32` / `dec32::sar` — identical arithmetic, differing only
    ///   in carrier type.
    const SWEEPS: [(&str, u64); 34] = [
        ("acc64::m32", 0x517aeadff46fa24b),
        ("acc64::s32", 0x6fb3878c443c14a3),
        ("acc64::i16s", 0x8e8e2e58dc60ce3d),
        ("acc64::s16m", 0x8e8e2e58dc60ce3d),
        ("acc64::sat16", 0x73d0e5266866d20e),
        ("acc64::s16", 0x79c9a25653a37ad3),
        ("acc64::sar", 0xc02ee1d158d76997),
        ("acc64::shl", 0xa6534346fb68502c),
        ("acc64::bsr", 0x9fcb284460c82092),
        ("acc64::bshift", 0x72ed2f05b1bb0fa1),
        ("acc64::bf_shift", 0x72ed2f05b1bb0fa1),
        ("acc64::sat_add", 0x66904baadf20174f),
        ("acc64::norm_shift", 0x47f83629748d98a7),
        ("acc64::norm_sh", 0x47f83629748d98a7),
        ("acc64::norm_edx", 0x4d7a35654823c8c4),
        ("acc64::denorm", 0x5ecee77e40159e1c),
        ("u32r::s32", 0x38c452df806bc945),
        ("u32r::u32v", 0x27cd48b05f245661),
        ("u32r::sar32", 0x1279d03831481a12),
        ("u32r::shl32", 0x4bfa733c49e5362a),
        ("u32r::imul32", 0x5870f7b73dad6fd1),
        ("u32r::s16", 0x79c9a25653a37ad3),
        ("u32r::u16v", 0x9097737b003cca1f),
        ("u32r::bsr32", 0x9557b442a3d2bdf2),
        ("i32r::s32", 0x38c452df806bc945),
        ("i32r::s16", 0x79c9a25653a37ad3),
        ("i32r::sar32", 0x52f0ebb750239f0a),
        ("i32r::bsr", 0x8a42d91f938eafea),
        ("i32r::i16w", 0x79c9a25653a37ad3),
        ("i16r::i16t", 0x79c9a25653a37ad3),
        ("i16r::s32", 0x79c9a25653a37ad3),
        ("dec32::sx16", 0x79c9a25653a37ad3),
        ("dec32::shl", 0xa68c72cda82ae17e),
        ("dec32::sar", 0x52f0ebb750239f0a),
    ];

    #[test]
    fn operator_sweeps_are_pinned() {
        fn push(name: &'static str, v: &mut Vec<i64>, acc: &mut Vec<(&'static str, u64)>) {
            acc.push((name, hash_i64s(v)));
            v.clear();
        }

        let mut out: Vec<i64> = Vec::new();
        let mut actual: Vec<(&'static str, u64)> = Vec::new();

        // ---- acc64 ----
        for &x in &P64 {
            out.push(acc64::m32(x));
        }
        push("acc64::m32", &mut out, &mut actual);
        for &x in &P64 {
            out.push(acc64::s32(x) as i64);
        }
        push("acc64::s32", &mut out, &mut actual);
        for &x in &P64 {
            out.push(acc64::i16s(x) as i64);
        }
        push("acc64::i16s", &mut out, &mut actual);
        for &x in &P64 {
            out.push(acc64::s16m(x));
        }
        push("acc64::s16m", &mut out, &mut actual);
        for &x in &P64 {
            out.push(acc64::sat16(x) as i64);
        }
        push("acc64::sat16", &mut out, &mut actual);
        for &x in &P32 {
            out.push(acc64::s16(x) as i64);
        }
        push("acc64::s16", &mut out, &mut actual);
        for &x in &P32 {
            for &n in &SH_I {
                out.push(acc64::sar(x, n) as i64);
            }
        }
        push("acc64::sar", &mut out, &mut actual);
        for &x in &P32 {
            for &n in &SH_I {
                out.push(acc64::shl(x, n) as i64);
            }
        }
        push("acc64::shl", &mut out, &mut actual);
        for &x in &P32 {
            out.push(acc64::bsr(x) as i64);
        }
        push("acc64::bsr", &mut out, &mut actual);
        for &x in &P32 {
            for &n in &SH_I {
                out.push(acc64::bshift(x, n) as i64);
            }
        }
        push("acc64::bshift", &mut out, &mut actual);
        for &x in &P32 {
            for &n in &SH_I {
                out.push(acc64::bf_shift(x, n) as i64);
            }
        }
        push("acc64::bf_shift", &mut out, &mut actual);
        for &a in &P32 {
            for &b in &P32 {
                out.push(acc64::sat_add(a, b) as i64);
            }
        }
        push("acc64::sat_add", &mut out, &mut actual);
        for &x in &P32 {
            out.push(acc64::norm_shift(x) as i64);
        }
        push("acc64::norm_shift", &mut out, &mut actual);
        for &x in &P32 {
            out.push(acc64::norm_sh(x) as i64);
        }
        push("acc64::norm_sh", &mut out, &mut actual);
        for &x in &P32 {
            let (m, e) = acc64::norm_edx(x);
            out.push(m as i64);
            out.push(e as i64);
        }
        push("acc64::norm_edx", &mut out, &mut actual);
        for &m in &MANTISSAS {
            for &e in &EXPONENTS {
                out.push(acc64::denorm(m, e) as i64);
            }
        }
        push("acc64::denorm", &mut out, &mut actual);

        // ---- u32r ----
        for &x in &P32 {
            out.push(u32r::s32(x as u32) as i64);
        }
        push("u32r::s32", &mut out, &mut actual);
        for &x in &P32 {
            out.push(u32r::u32v(x) as i64);
        }
        push("u32r::u32v", &mut out, &mut actual);
        for &x in &P32 {
            for &n in &SH_U {
                out.push(u32r::sar32(x as u32, n) as i64);
            }
        }
        push("u32r::sar32", &mut out, &mut actual);
        for &x in &P32 {
            for &n in &SH_U {
                out.push(u32r::shl32(x as u32, n) as i64);
            }
        }
        push("u32r::shl32", &mut out, &mut actual);
        for &a in &P32 {
            for &b in &P32 {
                out.push(u32r::imul32(a, b) as i64);
            }
        }
        push("u32r::imul32", &mut out, &mut actual);
        for &x in &P32 {
            out.push(u32r::s16(x as u16) as i64);
        }
        push("u32r::s16", &mut out, &mut actual);
        for &x in &P32 {
            out.push(u32r::u16v(x as u32) as i64);
        }
        push("u32r::u16v", &mut out, &mut actual);
        for &x in &P32 {
            // Zero is a documented precondition violation for bsr32.
            if x != 0 {
                out.push(u32r::bsr32(x as u32) as i64);
            }
        }
        push("u32r::bsr32", &mut out, &mut actual);

        // ---- i32r ----
        for &x in &P32 {
            out.push(i32r::s32(x) as i64);
        }
        push("i32r::s32", &mut out, &mut actual);
        for &x in &P32 {
            out.push(i32r::s16(x) as i64);
        }
        push("i32r::s16", &mut out, &mut actual);
        for &x in &P32 {
            for &n in &SH_U {
                out.push(i32r::sar32(x, n) as i64);
            }
        }
        push("i32r::sar32", &mut out, &mut actual);
        for &x in &P32 {
            out.push(i32r::bsr(x) as i64);
        }
        push("i32r::bsr", &mut out, &mut actual);
        for &x in &P32 {
            out.push(i32r::i16w(x) as i64);
        }
        push("i32r::i16w", &mut out, &mut actual);

        // ---- i16r ----
        for &x in &P32 {
            out.push(i16r::i16t(x) as i64);
        }
        push("i16r::i16t", &mut out, &mut actual);
        for &x in &P32 {
            out.push(i16r::s32(x as i16) as i64);
        }
        push("i16r::s32", &mut out, &mut actual);

        // ---- dec32 ----
        for &x in &P32 {
            out.push(dec32::sx16(x) as i64);
        }
        push("dec32::sx16", &mut out, &mut actual);
        for &x in &P32 {
            for &n in &SH_U {
                out.push(dec32::shl(x, n) as i64);
            }
        }
        push("dec32::shl", &mut out, &mut actual);
        for &x in &P32 {
            for &n in &SH_U {
                out.push(dec32::sar(x, n) as i64);
            }
        }
        push("dec32::sar", &mut out, &mut actual);

        println!("\n// Re-bless by copying this block into SWEEPS:");
        println!("    const SWEEPS: [(&str, u64); {}] = [", actual.len());
        for (name, h) in &actual {
            println!("        (\"{name}\", {h:#018x}),");
        }
        println!("    ];\n");

        assert!(
            SWEEPS.iter().all(|(_, h)| *h != 0),
            "SWEEPS still holds placeholder zeros — an unpinned gate reads as \
             passing, which is worse than no gate. Run with --nocapture and \
             paste the printed block."
        );
        assert_eq!(
            SWEEPS.len(),
            actual.len(),
            "SWEEPS has a different number of entries than the sweep produces"
        );

        let mut bad = Vec::new();
        for ((name, got), (want_name, want)) in actual.iter().zip(SWEEPS.iter()) {
            assert_eq!(name, want_name, "SWEEPS is out of order");
            if got != want {
                bad.push(format!("{name}: expected {want:#018x}, got {got:#018x}"));
            }
        }
        assert!(
            bad.is_empty(),
            "fixops operator behaviour changed:\n  {}\n\nThis file's header \
             forbids simplifying or reordering its arithmetic. If you did that \
             deliberately, the reference bit-exactness claim is what you are \
             trading away.",
            bad.join("\n  ")
        );
    }
}
