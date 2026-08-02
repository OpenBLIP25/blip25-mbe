//! The reference vocoder's 32-real-sample -> 16-complex-bin transform.
//!
//! This is a **direct port**, not a reimplementation: the algorithm was
//! recovered by instruction-level disassembly and was validated bit-exact
//! against captured the reference SDK output. That capture is gone, so the validation
//! cannot be repeated — the only remaining guard is the end-to-end output
//! hashing in `blip25-mbe`. Every shift and truncation below is load-bearing;
//! "simplifying" the arithmetic changes the output bits.
//!
//! ## What this computes
//!
//! Given 32 real `i16` input samples (in-place, overwritten by the
//! result), this performs a real-input FFT-style transform (sum/difference
//! butterfly decomposition, an 8-tap-equivalent DC/bin-0 accumulation, and
//! a 15-term-plus-tail coefficient-table-driven complex accumulation using
//! a dedicated 32-point twiddle table distinct from every other twiddle
//! table already closed in this project) and produces 16 packed
//! `(real, imag)` `i16` pairs (32 `i16` values total, same footprint as
//! the input, matching the real function's true in-place behaviour).
//!
//! ## What is NOT yet wired up
//!
//! This transform's 32-sample input (produced by a windowed
//! resampling/tapering step in the SDK, itself the writer of the 22-byte
//! per-band array) is not assembled end-to-end with
//! [`super::band_decompress::band_decompress`]'s `bins`/`step`/`outer` inputs.

const MASK32: u32 = 0xFFFF_FFFF;

#[inline]
fn sar32(x: u32, n: u32) -> u32 {
    (((x as i32) >> (n & 0x1f)) as u32) & MASK32
}

/// The twiddle table: `(round(32767*cos(k*11.25deg)),
/// round(-32767*sin(k*11.25deg)))` for `k = 0..32` (a standard Q15
/// 32-point real-FFT twiddle table), read from the reference binary's
/// `.rdata` section and confirmed byte-exact.
const TWIDDLE32: [(i16, i16); 32] = [
    (32767, 0),
    (32138, -6393),
    (30274, -12540),
    (27246, -18205),
    (23170, -23170),
    (18205, -27246),
    (12540, -30274),
    (6393, -32138),
    (0, -32767),
    (-6393, -32138),
    (-12540, -30274),
    (-18205, -27246),
    (-23170, -23170),
    (-27246, -18205),
    (-30274, -12540),
    (-32138, -6393),
    (-32767, 0),
    (-32138, 6393),
    (-30274, 12540),
    (-27246, 18205),
    (-23170, 23170),
    (-18205, 27246),
    (-12540, 30274),
    (-6393, 32138),
    (0, 32767),
    (6393, 32138),
    (12540, 30274),
    (18205, 27246),
    (23170, 23170),
    (27246, 18205),
    (30274, 12540),
    (32138, 6393),
];

/// Transform 32 real `i16` samples in place into 16 packed `(real, imag)`
/// `i16` pairs.
///
/// `buf` must have exactly 32 elements and is overwritten, matching the
/// reference function's in-place behaviour.
pub(crate) fn real_fft32(buf: &mut [i16; 32]) {
    // Phase 1: sum/difference butterfly decomposition. dst[0] is halved alone;
    // dst[16] (the Nyquist-like middle sample) is also halved alone and is only
    // ever consumed via that halved form (scratch[16]), for both the bin-0 sum
    // and the per-bin tail term. dst[1..=15] and dst[17..=31] are consumed via
    // symmetric sum/diff pairs: for i in 0..15,
    // scratch[1+i] = (dst[31-i] + dst[1+i]) >> 1 (sum),
    // scratch[17+i] = (dst[1+i] - dst[31-i]) >> 1 (diff).
    let dst: [i32; 32] = std::array::from_fn(|i| buf[i] as i32);
    let mut scratch = [0i32; 32];
    scratch[0] = dst[0] >> 1;
    scratch[16] = dst[16] >> 1;
    for i in 0..15 {
        let a = dst[31 - i];
        let b = dst[1 + i];
        scratch[1 + i] = (a + b) >> 1;
        scratch[17 + i] = (b - a) >> 1;
    }

    // Phase 2: bin 0 (DC). Sum scratch[0..=16] -- 17 terms, not 8, despite the
    // shape resembling an 8-tap moving average -- then >>3. Imag part of bin 0
    // is always exactly 0.
    let bin0_sum: i32 = scratch[0..17].iter().sum();
    buf[0] = (bin0_sum >> 3) as i16;
    buf[1] = 0;

    // Phase 3: bins 1..=15. Each bin's real accumulator is seeded with
    // `scratch[0] << 16` -- a carry-over from bin 0's setup code that is NOT
    // reset between bins. Without this seed every bin's real part is off by a
    // constant amount. The imag accumulator starts at 0. For 15 inner
    // steps (k = 0..15), the running twiddle-table index starts at the
    // bin index and increments by the bin index each step (mod 32); each
    // step accumulates `2 * twiddle[idx].re * scratch[1+k]` into the real
    // accumulator and `2 * twiddle[idx].im * scratch[17+k]` into the imag
    // accumulator. After the 15 steps, one more "tail" term
    // `2 * twiddle[idx].re * scratch[16]` (using the twiddle index the
    // running index has wrapped to after the 15th step) is added to the
    // real accumulator only. Each final accumulator is combined as a true
    // 64-bit signed value, shifted right 3 then right 16 (matching the
    // real `shrd 3` + `sar 16` instruction pair), then truncated to i16.
    let seed_re: i64 = (scratch[0] as i64) << 16;
    for bin in 1..=15i32 {
        let mut acc_re: i64 = seed_re;
        let mut acc_im: i64 = 0;
        let mut idx: i32 = bin;
        for k in 0..15usize {
            let (tre, tim) = TWIDDLE32[(idx & 0x1f) as usize];
            acc_re += 2 * (tre as i64) * (scratch[1 + k] as i64);
            acc_im += 2 * (tim as i64) * (scratch[17 + k] as i64);
            idx = (idx + bin) & 0x1f;
        }
        let (tre_tail, _tim_tail) = TWIDDLE32[(idx & 0x1f) as usize];
        acc_re += 2 * (tre_tail as i64) * (scratch[16] as i64);

        let re = shrd3_sar16(acc_re);
        let im = shrd3_sar16(acc_im);
        buf[(2 * bin) as usize] = re as i16;
        buf[(2 * bin + 1) as usize] = im as i16;
    }
}

/// Replicates the reference code's `shrd eax,edx,3` (logical shift of the full
/// 64-bit accumulator by 3, keeping only the low 32 bits) followed by
/// `sar eax,0x10` (arithmetic shift of that 32-bit result by 16).
#[inline]
fn shrd3_sar16(acc: i64) -> i32 {
    let u64v = acc as u64;
    let shifted = u64v >> 3;
    let low32 = (shifted & 0xFFFF_FFFF) as u32;
    sar32(low32, 16) as i32
}

#[cfg(test)]
mod tests {
    //! Pinned transforms on four fixed integer inputs, plus the two pieces of
    //! arithmetic that are easy to "simplify" into something that looks
    //! equivalent and is not: `shrd3_sar16` (a *logical* 64-bit >>3 followed
    //! by an *arithmetic* 32-bit >>16, which is not >>19 of anything) and the
    //! final `as i16`, which wraps rather than saturating.
    //!
    //! The impulse and DC cases below are derived from the algorithm in the
    //! comments above rather than read off a run; the alternating and
    //! pseudo-random cases are captured, which is the honest split for a
    //! 16-bin transform.

    use super::*;

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    fn hash_i16(vals: &[i16]) -> u64 {
        let mut bytes = Vec::with_capacity(vals.len() * 2);
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        fnv1a(&bytes)
    }

    fn run(input: [i16; 32]) -> [i16; 32] {
        let mut buf = input;
        real_fft32(&mut buf);
        buf
    }

    /// xorshift64 — fully specified integer arithmetic, so the probe vector is
    /// bit-identical on every target and only the transform is under test.
    fn pseudo_random_input() -> [i16; 32] {
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        std::array::from_fn(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 48) as u16 as i16
        })
    }

    #[test]
    fn twiddle_table_is_the_q15_32_point_table() {
        assert_eq!(TWIDDLE32.len(), 32);
        // The four axis points fix the table's phase convention: it runs
        // (cos, -sin), i.e. clockwise, starting at +1.
        assert_eq!(TWIDDLE32[0], (32767, 0));
        assert_eq!(TWIDDLE32[8], (0, -32767));
        assert_eq!(TWIDDLE32[16], (-32767, 0));
        assert_eq!(TWIDDLE32[24], (0, 32767));
        // 45 degrees is the same magnitude on both axes.
        assert_eq!(TWIDDLE32[4], (23170, -23170));
        assert_eq!(TWIDDLE32[20], (-23170, 23170));
        // Half-turn antisymmetry: entry k+16 is the negation of entry k.
        for k in 0..16 {
            let (re, im) = TWIDDLE32[k];
            let (re16, im16) = TWIDDLE32[k + 16];
            assert_eq!((re16, im16), (-re, -im), "TWIDDLE32[{}] vs [{}]", k, k + 16);
        }

        let flat: Vec<i16> = TWIDDLE32.iter().flat_map(|&(r, i)| [r, i]).collect();
        assert_eq!(
            hash_i16(&flat),
            0xf4dcb065fd1e1f39,
            "the twiddle table was read byte-exact from the reference \
             binary's .rdata; it is data, not a derivation"
        );
    }

    #[test]
    fn shrd3_sar16_is_not_an_arithmetic_shift_by_19() {
        assert_eq!(shrd3_sar16(0), 0);
        assert_eq!(shrd3_sar16(1 << 19), 1);
        assert_eq!(shrd3_sar16((1 << 19) - 1), 0);
        assert_eq!(shrd3_sar16(-8), -1);

        // Bits at or above 2^35 are shifted into the discarded upper half by
        // the LOGICAL >>3, so they vanish instead of scaling the result. An
        // `(acc >> 19) as i32` would answer 65537 and -65536 here.
        assert_eq!(shrd3_sar16((1i64 << 35) + (1 << 19)), 1);
        assert_eq!(shrd3_sar16(-(1i64 << 35)), 0);
        assert_ne!(shrd3_sar16(-(1i64 << 35)), (-(1i64 << 35) >> 19) as i32);

        // The second shift IS arithmetic, so a low half with its top bit set
        // comes back negative.
        assert_eq!(shrd3_sar16(-(1i64 << 19)), -1);
        assert_eq!(shrd3_sar16(0x7fff_ffffi64 << 3), 0x7fff);
        assert_eq!(shrd3_sar16(0x8000_0000i64 << 3), -0x8000);
    }

    #[test]
    fn impulse_puts_the_same_dc_seed_in_every_bin() {
        // dst = [1000, 0 x 31]. Only scratch[0] = 500 is non-zero, so bin 0 is
        // 500 >> 3 = 62 and every other bin is just the un-reset seed
        // (500 << 16) pushed through shrd3_sar16: 32_768_000 >> 3 = 4_096_000,
        // >> 16 = 62. The imaginary parts are all zero because every
        // difference term is zero. That constant 62 in bins 1..15 IS the
        // carry-over seed the module comment describes; without it they would
        // all be 0.
        let mut input = [0i16; 32];
        input[0] = 1000;
        let out = run(input);
        let mut want = [0i16; 32];
        for bin in 0..16 {
            want[2 * bin] = 62;
        }
        assert_eq!(out, want);
    }

    #[test]
    fn dc_input_lands_entirely_in_bin_zero() {
        // dst = [1000 x 32]. Every sum term is 1000 and every difference term
        // is 0, so bin0_sum = 500 + 15*1000 + 500 = 16000 and bin 0 is
        // 16000 >> 3 = 2000. In bins 1..15 the twiddle real parts sum to
        // exactly -32767 against the tail term's +32767, cancelling the seed
        // down to 1000, which shrd3_sar16 floors to 0.
        let out = run([1000i16; 32]);
        let mut want = [0i16; 32];
        want[0] = 2000;
        assert_eq!(out, want);
    }

    #[test]
    fn bin_zero_wraps_instead_of_saturating() {
        // Full-scale DC: bin0_sum = 16383 + 15*32767 + 16383 = 524_271, and
        // 524_271 >> 3 = 65_533, which does not fit an i16. The `as i16` cast
        // wraps it to -3. A saturating store would give 32767 — this is the
        // reference's behaviour and it is load-bearing.
        let out = run([32767i16; 32]);
        assert_eq!(out[0], -3);
        assert_eq!(out[1], 0);
    }

    #[test]
    fn transform_outputs_are_pinned() {
        let mut input_alt = [0i16; 32];
        for (i, s) in input_alt.iter_mut().enumerate() {
            *s = if i % 2 == 0 { 1000 } else { -1000 };
        }

        let mut impulse = [0i16; 32];
        impulse[0] = 1000;
        let mut impulse_mid = [0i16; 32];
        impulse_mid[16] = 1000;

        let cases: [(&str, [i16; 32]); 6] = [
            ("impulse@0", impulse),
            ("impulse@16", impulse_mid),
            ("dc_1000", [1000i16; 32]),
            ("dc_full_scale", [32767i16; 32]),
            ("alternating", input_alt),
            ("pseudo_random", pseudo_random_input()),
        ];

        /// Re-bless by running with `--nocapture` and pasting the printed
        /// block.
        const EXPECTED: [(&str, [i16; 32]); 6] = [
            (
                "impulse@0",
                [
                    62, 0, 62, 0, 62, 0, 62, 0, 62, 0, 62, 0, 62, 0, 62, 0, 62, 0, 62, 0, 62, 0,
                    62, 0, 62, 0, 62, 0, 62, 0, 62, 0,
                ],
            ),
            (
                "impulse@16",
                [
                    62, 0, -63, 0, 62, 0, -63, 0, 62, 0, -63, 0, 62, 0, -63, 0, 62, 0, -63, 0, 62,
                    0, -63, 0, 62, 0, -63, 0, 62, 0, -63, 0,
                ],
            ),
            (
                "dc_1000",
                [
                    2000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0,
                ],
            ),
            (
                "dc_full_scale",
                [
                    -3, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0,
                    -1, 0, 0, 0, -1, 0, 0, 0,
                ],
            ),
            (
                "alternating",
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                ],
            ),
            (
                "pseudo_random",
                [
                    8607, 0, -359, -2705, -5470, -984, 4444, 12983, -5067, 10007, 4739, -1392,
                    -4830, 2669, -5234, 4171, 1858, -3774, -5939, -458, -3724, 1452, -4914, 4164,
                    -2475, -1961, 346, 5542, 4175, -1529, -1032, 4578,
                ],
            ),
        ];

        let actual: Vec<(&str, [i16; 32])> = cases
            .iter()
            .map(|&(name, input)| (name, run(input)))
            .collect();

        println!("\n// Re-bless by copying this block into EXPECTED:");
        println!("        const EXPECTED: [(&str, [i16; 32]); 6] = [");
        for (name, out) in &actual {
            println!("            (");
            println!("                \"{name}\",");
            println!("                {out:?},");
            println!("            ),");
        }
        println!("        ];\n");

        assert!(
            EXPECTED.iter().any(|(_, v)| v.iter().any(|&x| x != 0)),
            "EXPECTED still holds placeholder zeros — an unpinned gate reads \
             as passing, which is worse than no gate."
        );
        // The pseudo-random case is the one that touches every twiddle entry
        // and both accumulators; it must not be all-zero, or the gate covers
        // nothing.
        assert!(
            EXPECTED
                .iter()
                .find(|(k, _)| *k == "pseudo_random")
                .map(|(_, v)| v.iter().any(|&x| x != 0))
                .expect("pseudo_random case present"),
            "the pseudo_random expectation is all zeros — it was never blessed"
        );

        let mut bad = Vec::new();
        for ((name, got), (want_name, want)) in actual.iter().zip(EXPECTED.iter()) {
            assert_eq!(name, want_name, "EXPECTED is out of order");
            if got != want {
                bad.push(format!(
                    "{name}:\n    expected {want:?}\n    got      {got:?}"
                ));
            }
        }
        assert!(
            bad.is_empty(),
            "real_fft32 output changed:\n  {}\n\nThis is a direct port whose \
             validating capture no longer exists; every shift and truncation \
             in it is load-bearing and cannot be re-derived.",
            bad.join("\n  ")
        );

        // Input generator stability, so a change to the probe can never be
        // mistaken for a change to the transform.
        assert_eq!(hash_i16(&pseudo_random_input()), 0xf8dca57ed4d5da0f);
    }
}
