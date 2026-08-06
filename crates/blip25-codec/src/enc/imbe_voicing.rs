//! The IMBE packer's own voicing search (`FUN_10312af0`).
//!
//! The full-rate packer does not transmit the AMBE+2 five-bit voicing index. It
//! runs a second search over the SAME three analysis measurement arrays
//! (`a4`/`a5`/`a6`, `enc::voicing_vq`) with a different candidate set, and the
//! index it returns decides `b̂₀` as well as the voicing field:
//!
//! | winning index `v` | `b̂₀`                        | `b̂₁` |
//! |---|---|---|
//! | `v > 56`  | the quantized prequantiser pitch word | `v - 56` |
//! | `v == 56` | `25`                                  | `0` |
//! | `v < 56`  | `v + 128`                             | `0` |
//!
//! so a frame whose voicing search lands below 57 transmits no pitch at all.
//!
//! The candidates are `0..=56` followed by `56 + 2^(k-1) ..= 56 + 2^k - 1`,
//! where `k = min(num_bands(L), 8)` and `L` is the harmonic count the pitch
//! index implies. Indices below 56 read [`IMBE_VOICING_CB`]; the rest build
//! their codeword from the index's own bits, spread over `k` bands by
//! [`BAND_SPREAD`], combined with a 16-bit carry of the previous frame's
//! winning codeword.
//!
//! The scoring is `enc::voicing_vq`'s, unchanged — the same `build_arrays` and
//! the same normalize/compare ladder, so there is one implementation of the
//! metric for both codecs.

use crate::enc::voicing_vq::{build_arrays, score_one};

/// The below-57 codebook (`DAT_10381980`): 56 packed dwords, two bits per band.
/// Distinct from the AMBE+2 codebook `shared::voicing_map::VOICING_CB` — a
/// different table at a different address, entered at stride 1.
const IMBE_VOICING_CB: [u32; 56] = [
    131072, 524288, 655360, 2097152, 2752512, 11141120, 33554432, 41943040, 32768, 640, 512, 2728,
    2048, 2560, 2688, 672, 2, 42, 8, 32, 10, 170, 128, 160, 2147483648, 176160768, 2818572288,
    44040192, 2684354560, 2852126720, 44695552, 167772160, 2863267840, 8388608, 178913280,
    2860515328, 134217728, 2862612480, 715784192, 2863136768, 43690, 43648, 43008, 10922, 40960,
    682, 2730, 43680, 2863311530, 2863270570, 2818615296, 44696234, 178916010, 2863310848,
    2684395520, 134219776,
];

/// How many of the codeword's eight sub-bands each of the `k` voicing bands
/// covers (`DAT_10381920`), row `k - 3`. Every row sums to eight.
const BAND_SPREAD: [[u8; 8]; 6] = [
    [3, 2, 3, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 0, 0, 0, 0],
    [2, 1, 2, 1, 2, 0, 0, 0],
    [1, 1, 2, 1, 1, 2, 0, 0],
    [1, 1, 1, 1, 1, 1, 2, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
];

/// The index at which the candidate ladder switches from the codebook to the
/// per-band words; also the index that means "all unvoiced".
pub const IMBE_DIRECT_BASE: u16 = 56;

/// The `b̂₀` the packer transmits when the search returns [`IMBE_DIRECT_BASE`].
pub const IMBE_UV_B0: u8 = 25;

/// The offset added to a sub-56 index to form the transmitted `b̂₀`.
pub const IMBE_CB_B0_BASE: u8 = 128;

/// The band count the packer's search runs at: `min(num_bands(L), 8)`, where
/// `num_bands` is the truncating `(L + 2) / 3` clamped to twelve.
pub fn search_bands(num_harms: i16) -> u16 {
    let raw = (i32::from(num_harms) + 2) / 3;
    raw.clamp(0, 8) as u16
}

/// One candidate's codeword (`FUN_103124b0`).
///
/// `prev` is the 16-bit carry described on [`advance`]; `k` is
/// [`search_bands`]'s value.
fn codeword(idx: u16, prev: u16, k: u16) -> u32 {
    if idx < IMBE_DIRECT_BASE {
        return IMBE_VOICING_CB[idx as usize];
    }
    let k = k.min(8) as usize;
    let mut m: u16 = 0;
    if k >= 3 {
        let row = &BAND_SPREAD[k - 3];
        for (band, &reps) in row.iter().enumerate().take(k) {
            let bit = u32::from(idx - IMBE_DIRECT_BASE) >> band & 1;
            for _ in 0..reps {
                m = (((m as i16) >> 2) as u16) | ((bit as u16) << 14);
            }
        }
    }
    let p = u32::from(prev) & 0x5555;
    let mm = u32::from(m);
    let u = p ^ mm;
    (u.wrapping_mul(4) & u) | (p & mm) | (mm << 16)
}

/// The 16-bit voicing carry the next frame's search reads: the high half of the
/// winning codeword.
///
/// The reference stores the winner expanded to two 32-bit words (its low and
/// high halves, `FUN_10312570`) and re-compresses the one it keeps
/// (`FUN_10317060`) on the next frame; the round trip is the identity on the
/// half it keeps, so the carry is just this.
pub fn advance(idx: u16, prev: u16, k: u16) -> u16 {
    (codeword(idx, prev, k) >> 16) as u16
}

/// Search the packer's candidate ladder and return the winning index
/// (`FUN_10312700` at `param_6 == 0`).
pub fn search(a4: &[i16; 16], a5: &[i16; 16], a6: &[i16; 16], prev: u16, k: u16) -> u16 {
    let k = k.min(8);
    let (aa1, aa2, aa3) = build_arrays(a4, a5);
    let mut best_exp: i64 = 0x7fff;
    let mut best_mant: i64 = 0x7fff;
    let mut best = 0u16;
    let hi = (1u32 << k) + u32::from(IMBE_DIRECT_BASE);
    let resume = ((hi - u32::from(IMBE_DIRECT_BASE)) >> 1) + u32::from(IMBE_DIRECT_BASE);
    let mut i = 0u32;
    while i < hi {
        if i == u32::from(IMBE_DIRECT_BASE) + 1 {
            i = resume;
            if i >= hi {
                break;
            }
        }
        let (neg_exp, mant) = score_one(&aa1, &aa2, &aa3, a6, codeword(i as u16, prev, k));
        if mant == 0 {
            return i as u16;
        }
        let e = neg_exp as i16;
        let m = mant as i16;
        if e < best_exp as i16 || (e == best_exp as i16 && m < best_mant as i16) {
            best_exp = neg_exp;
            best_mant = mant;
            best = i as u16;
        }
        i += 1;
    }
    best
}

/// Resample an eight-band voicing word onto `k > 8` bands (`FUN_10312b90`).
///
/// The search never runs at more than eight bands, so a frame whose harmonic
/// count wants more of them gets its winner spread over the wider field: output
/// band `i` copies the input bit the harmonic frequency `32 + 48i` falls on,
/// scaled by the fundamental. The output is MSB-first in `i`.
pub fn spread_bands(word8: u16, k: u16, num_harms: i16) -> u16 {
    let l32 = i32::from(num_harms) << 16;
    let shift: i32 = if l32 == 0 {
        0
    } else {
        let m = (if l32 < 0 { !l32 } else { l32 }) as u32;
        0x1e - (31 - m.leading_zeros() as i32)
    };
    let scale = crate::shared::fractional_divide::a600(0x3b39_c0ec, (l32 << shift) >> 16) as i16;
    let sh = shift - 0xf;
    let mut out: u16 = 0;
    let mut freq: i32 = 0x20;
    for i in 0..i32::from(k) {
        let rem = i32::from(k) - 1 - i;
        let mut t = freq * i32::from(scale) * 2;
        t = if sh < 0 {
            if sh < -0x1e {
                t >> 31
            } else {
                t >> (-sh)
            }
        } else {
            t << sh
        };
        let raw = 7 - ((t + 0x4000) >> 16);
        let mut band = (raw & 0xffff) as u16;
        if (raw as i16) < 0 {
            band = 0;
        }
        if rem < i32::from(band as i16) {
            band = rem as u16;
        }
        out = (out << 1) | ((word8 >> (band & 0x1f)) & 1);
        freq += 0x30;
    }
    out
}

/// What the packer transmits for a winning index: `(b̂₀, b̂₁)`.
///
/// `pitch_b0` is the quantized prequantiser pitch word, used only on the
/// `v > 56` arm.
pub fn packer_fields(idx: u16, pitch_b0: u8) -> (u8, u16) {
    match idx.cmp(&IMBE_DIRECT_BASE) {
        std::cmp::Ordering::Greater => (pitch_b0, idx - IMBE_DIRECT_BASE),
        std::cmp::Ordering::Equal => (IMBE_UV_B0, 0),
        std::cmp::Ordering::Less => (IMBE_CB_B0_BASE.wrapping_add(idx as u8), 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `BAND_SPREAD` row covers exactly the eight sub-bands the codeword
    /// carries — the property that makes the shift loop terminate on a full word.
    #[test]
    fn band_spread_rows_sum_to_eight() {
        for (i, row) in BAND_SPREAD.iter().enumerate() {
            let k = i + 3;
            assert_eq!(row.iter().map(|&r| u32::from(r)).sum::<u32>(), 8);
            assert!(row[k..].iter().all(|&r| r == 0), "row {k} has tail entries");
        }
    }

    /// The direct arm only ever asks for codes 0 and 1: every codeword bit it
    /// can set lies on an even position. Code 2 is reachable only through the
    /// codebook.
    #[test]
    fn direct_codewords_are_even_bits_only() {
        for k in 3..=8u16 {
            for idx in IMBE_DIRECT_BASE..IMBE_DIRECT_BASE + (1 << k) {
                for prev in [0u16, 0x5555, 0xffff, 0x1234] {
                    assert_eq!(codeword(idx, prev, k) & 0xaaaa_aaaa, 0);
                }
            }
        }
    }

    /// `search_bands` is the reference's truncating divide, not a rounding one:
    /// `L = 18` gives six bands, `L = 19` seven.
    #[test]
    fn search_bands_truncates() {
        assert_eq!(search_bands(18), 6);
        assert_eq!(search_bands(19), 7);
        assert_eq!(search_bands(21), 7);
        assert_eq!(search_bands(22), 8);
        assert_eq!(search_bands(56), 8);
    }

    /// The wide-band resample only ever copies input bits, so a uniform input
    /// stays uniform: all-unvoiced spreads to all-unvoiced and an all-voiced
    /// eight-band word to `k` set bits. An off-by-one in the band index would
    /// break the second half by pulling in a bit outside the word.
    #[test]
    fn spread_preserves_uniform_words() {
        for l in 9..=56i16 {
            let k = crate::imbe::num_bands(l) as u16;
            if k <= 8 {
                continue;
            }
            assert_eq!(spread_bands(0, k, l), 0, "L={l}");
            assert_eq!(
                spread_bands(0xff, k, l),
                (1u16 << k) - 1,
                "L={l} k={k} all-voiced"
            );
        }
    }

    /// The three arms partition the index range and never collide on `b̂₀`:
    /// the codebook arm's indices land in `128..=183`, which the pitch arm can
    /// also reach, but they carry `b̂₁ = 0` where the pitch arm carries `v - 56`.
    #[test]
    fn packer_fields_arms() {
        assert_eq!(packer_fields(57, 99), (99, 1));
        assert_eq!(packer_fields(56, 99), (IMBE_UV_B0, 0));
        assert_eq!(packer_fields(0, 99), (128, 0));
        assert_eq!(packer_fields(55, 99), (183, 0));
    }
}
