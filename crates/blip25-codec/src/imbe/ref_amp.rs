// index loops are deliberate: the index is the harmonic/block/coefficient number
#![allow(clippy::needless_range_loop)]

//! The reference encoder's IMBE **amplitude back end**, in its own arithmetic.
//!
//! [`super::quantize::sa_encode`] is the TIA-102.BABA fixed-point reference's
//! amplitude encoder, and it is the exact left-inverse of [`super::dequantize`].
//! The DVSI encoder computes the same transform in a different fixed-point
//! pipeline, and the two disagree by a level here and there:
//!
//! * its DCT accumulates full 32-bit products in a 64-bit accumulator and
//!   block-normalises once at the end, where the standard's truncates every
//!   product to `i16` before summing;
//! * its prediction resamples the previous frame's reconstruction into `i16`
//!   before the `ρ` multiply, and takes the mean through a 16-bit divide;
//! * its coefficient quantiser is an exact restoring division, not a
//!   reciprocal multiply.
//!
//! Those are not sign-preserving rewrites of one another: a `±1` disagreement in
//! one coefficient is a wrong bit in the frame. This module is the DVSI pipeline
//! itself, so the frame it emits is the frame the reference emits.
//!
//! The predictor memory is the reference's own too — 56 `i16` words in Q11, the
//! reconstruction the back end forms — rather than the decoder mirror
//! [`super::quantize::ImbeEncState`] keeps.

use super::quantize::{
    apply_unvoiced_offset, encode_frame_vector, get_bit_allocation, num_bands, ref_gain_dc_quant,
    unvoiced_log_offset,
};
use super::tables::{GAIN_QNT_TBL, HI_ORD_STD_TBL};
use crate::enc::cos_quadrature_interp::COS_TABLE;
use crate::shared::fractional_divide::a600;

const NUM_HARMS_MAX: usize = 56;
const NUM_PRED_RES_BLKS: usize = 6;
const B_NUM: usize = NUM_HARMS_MAX - 1;

/// `min(32767, 32768/n)` for `n = 0..=56`: the DCT's per-length output scale.
const DCT_SCALE: [i16; 57] = [
    0, 32767, 16384, 10922, 8192, 6553, 5461, 4681, 4096, 3640, 3276, 2978, 2730, 2520, 2340, 2184,
    2048, 1927, 1820, 1724, 1638, 1560, 1489, 1424, 1365, 1310, 1260, 1213, 1170, 1129, 1092, 1057,
    1024, 992, 963, 936, 910, 885, 862, 840, 819, 799, 780, 762, 744, 728, 712, 697, 682, 668, 655,
    642, 630, 618, 606, 595, 585,
];

/// Quantiser step scale by bit count, index `0..=12`. Saturates from 7 bits up,
/// which is what makes the cell width halve with every extra bit beyond that.
const STEP_SIZE_BY_BITS: [i16; 13] = [
    0, 4915, 6963, 10650, 13107, 18350, 19661, 20972, 20972, 20972, 20972, 20972, 20972,
];

/// Per-coefficient standard deviations for the five quantised gain coefficients.
const GAIN_STD_TBL: [i16; 5] = [20316, 13173, 11010, 9503, 8651];

/// Initial predictor memory. The reference opens with a harmonic count of 15 and
/// a pitch word of 16423 against an all-zero amplitude vector, so the first
/// frame's prediction is not zero.
const INIT_NUM_HARMS_PREV: i16 = 15;
const INIT_PITCH_WORD_PREV: u16 = 16423;

/// The reference encoder's amplitude predictor memory (its `state` block).
#[derive(Clone)]
pub(crate) struct RefAmpState {
    num_harms_prev: i16,
    pitch_word_prev: u16,
    voicing_prev: i16,
    /// The previous frame's reconstruction with the unvoiced-band offset added
    /// back, tail-replicated to 56 words — the block the reference stores.
    amps_prev: [i16; NUM_HARMS_MAX],
}

impl RefAmpState {
    pub(crate) fn new() -> Self {
        Self {
            num_harms_prev: INIT_NUM_HARMS_PREV,
            pitch_word_prev: INIT_PITCH_WORD_PREV,
            voicing_prev: 0,
            amps_prev: [0; NUM_HARMS_MAX],
        }
    }
}

impl Default for RefAmpState {
    fn default() -> Self {
        Self::new()
    }
}

/// `L_shl` with saturation, the reference's shared shifter.
fn l_shl_sat(x: i32, n: i16) -> i32 {
    if n < 0 {
        return if n < -30 {
            x >> 31
        } else {
            x >> ((-n) as u32 & 0x1f)
        };
    }
    let mask: i32 = -1i32 << ((31 - n) as u32 & 0x1f);
    let mut over = mask & x;
    if x < 0 {
        over = over.wrapping_sub(mask);
    }
    if over != 0 {
        return if x < 0 { i32::MIN } else { i32::MAX };
    }
    ((x as u32) << (n as u32 & 0x1f)) as i32
}

/// `L_mult` negated, with the reference's single overflow guard.
fn neg_l_mult(a: i32, b: i32) -> i32 {
    let t = a.wrapping_mul(b).wrapping_mul(2);
    if t == i32::MIN {
        i32::MAX
    } else {
        t.wrapping_neg()
    }
}

/// Leading-bit normalisation of a 64-bit accumulator over its low 40 bits,
/// returning the left shift that puts the most significant bit at position 30.
fn norm_acc(v: i64) -> i16 {
    if v == 0 {
        return 0;
    }
    let w = if (v >> 32) < 0 { !v } else { v };
    let mut j = 0i32;
    while j < 0x28 {
        if (w >> (39 - j)) & 1 != 0 {
            break;
        }
        j += 1;
    }
    (j - 9) as i16
}

/// Forward DCT-II of `input[0..n]` into `out[0..min(n, m)]`, block-floating-point.
///
/// Every product enters a 64-bit accumulator at full width; the sum is
/// normalised once, scaled by the length's own reciprocal and rounded. The
/// standard's DCT truncates each product to `i16` first, which costs a level in
/// the coefficients a short block feeds.
fn dct(input: &[i16], n: i16, m: i16, out: &mut [i16]) {
    let scale = DCT_SCALE[n as usize] as i32;
    let m = if n <= m { n } else { m };
    let mut phase: i32 = 0;
    for k in 0..m as usize {
        let mut angle: i32 = (phase >> 1).wrapping_add(0x40);
        let step = phase as i16 as i32;
        let mut acc: i64 = 0;
        for i in 0..n as usize {
            let c = COS_TABLE[((angle >> 7) & 0x1ff) as usize] as i32;
            acc += (input[i] as i32).wrapping_mul(c).wrapping_mul(2) as i64;
            angle = angle.wrapping_add(step);
        }
        let sh = norm_acc(acc);
        let norm = if sh >= 0 {
            ((acc as u64) << (sh as u32 & 63)) as i64
        } else {
            acc >> ((-sh) as u32 & 63)
        };
        let mant = ((norm as u32 as i32) >> 16) as i16 as i32;
        let scaled = l_shl_sat(mant.wrapping_mul(scale).wrapping_mul(2), -sh);
        out[k] = (scaled.wrapping_add(0x8000) >> 16) as i16;
        phase = phase.wrapping_add(scale);
    }
}

/// Inverse of [`dct`], in place over `x[0..n]` from its first `min(n, m)`
/// coefficients. The DC keeps unit weight; every AC coefficient is doubled.
fn idct(x: &mut [i16], n: i16, m: i16) {
    let m = if n <= m { n } else { m };
    let mut buf = [0i16; NUM_HARMS_MAX];
    buf[0] = x[0];
    for i in 1..m as usize {
        buf[i] = ((x[i] as i32).wrapping_mul(2)) as i16;
    }
    let scale = DCT_SCALE[n as usize];
    let mut phase = scale >> 1;
    for k in 0..n as usize {
        let mut acc: i32 = 0;
        let mut angle: i32 = 0x40;
        for i in 0..m as usize {
            let c = COS_TABLE[((angle >> 7) & 0x1ff) as usize] as i32;
            angle = angle.wrapping_add(phase as i32);
            acc = acc.wrapping_add(c.wrapping_mul(buf[i] as i32).wrapping_mul(2));
        }
        phase = phase.wrapping_add(scale);
        x[k] = (acc.wrapping_add(0x8000) >> 16) as i16;
    }
}

/// Resample the previous frame's `nhp` amplitudes onto `nh` harmonics, returning
/// the resampled vector and the accumulated sum the mean is taken from.
///
/// The resampled words land in `i16` before `ρ` is applied — the precision the
/// standard's version keeps in Q10.22 and this one does not.
fn resample(nh: i16, prev: &[i16; NUM_HARMS_MAX], nhp: i16) -> ([i16; NUM_HARMS_MAX], i32) {
    let recip = a600(0x0008_0000, i32::from(nh)) as i16 as i32;
    let scaled_nhp = ((((nhp as i32) << 25) >> 16) as i16) as i32;
    let step = recip.wrapping_mul(scaled_nhp).wrapping_mul(2) >> 12;
    let mut acc = step;
    let mut out = [0i16; NUM_HARMS_MAX];
    let mut sum = 0i32;
    for i in 0..nh as usize {
        let raw = ((acc >> 16) as u16) as i16;
        // Index 0 has no lower neighbour to interpolate against, so the first
        // harmonic is taken whole.
        let v: i32 = if raw == 0 {
            (prev[0] as i32) << 16
        } else {
            let idx = if raw < 0x37 { raw } else { 0x37 } as usize;
            let frac = (((acc >> 1) & 0x7fff) as i16) as i32;
            let lo = prev[idx - 1] as i32;
            let hi = prev[idx] as i32;
            neg_l_mult(frac, lo).wrapping_add(
                hi.wrapping_mul(frac)
                    .wrapping_add(lo.wrapping_mul(0x8000))
                    .wrapping_mul(2),
            )
        };
        out[i] = (((v as u32) >> 16) as u16) as i16;
        sum = sum.wrapping_add(v >> 6);
        acc = acc.wrapping_add(step);
    }
    (out, sum)
}

/// The prediction coefficient `ρ` for a harmonic count.
fn rho(nh: i16) -> i16 {
    if nh < 16 {
        0x3333
    } else if nh < 25 {
        ((nh as i32)
            .wrapping_mul(0x03d7_0800)
            .wrapping_sub(0x0666_0000)
            >> 16) as i16
    } else {
        0x599a
    }
}

/// The predicted log-amplitude vector, Q11, mean removed.
fn prediction(nh: i16, prev: &[i16; NUM_HARMS_MAX], nhp: i16) -> [i16; NUM_HARMS_MAX] {
    let (resampled, sum) = resample(nh, prev, nhp);
    let r = rho(nh);
    let scaled_nh = ((((nh as i32) << 25) >> 16) as i16) as i32;
    let mean = a600(sum, scaled_nh) as i16 as i32;
    let mean_term = neg_l_mult(mean, i32::from(r)) >> 1;
    let mut out = [0i16; NUM_HARMS_MAX];
    for i in 0..nh as usize {
        let scaled = (resampled[i] as i32)
            .wrapping_mul(i32::from(r))
            .wrapping_mul(2)
            >> 1;
        out[i] = (l_shl_sat(scaled.wrapping_add(mean_term), 1) >> 16) as i16;
    }
    out
}

/// The cell width for `num_bits` bits at standard deviation `std`.
fn cell(std: i16, num_bits: i16) -> i16 {
    (((STEP_SIZE_BY_BITS[num_bits as usize] as i32) * (std as i32) * 2) >> 17) as i16
}

/// Quantise one coefficient by exact restoring division, returning its index.
///
/// The value is offset by half a cell and divided by the cell width one
/// quotient bit at a time. A reciprocal multiply agrees with this on most
/// inputs and lands a level low on the rest.
fn quantise(val: i16, std: i16, num_bits: i16) -> i16 {
    if num_bits <= 0 {
        return 0;
    }
    let width = cell(std, num_bits);
    let mut acc = ((i32::from(width) << 16) >> 1).wrapping_add(i32::from(val) << 16);
    if acc < 0 {
        return 0;
    }
    if ((acc >> 16) as i16) >= width {
        return ((1i32 << num_bits) - 1) as i16;
    }
    let denom = i32::from(width).wrapping_mul(0x8000);
    for _ in 0..num_bits {
        let t = acc.wrapping_sub(denom);
        acc = if t >= 0 {
            t.wrapping_mul(2).wrapping_add(1)
        } else {
            acc.wrapping_mul(2)
        };
    }
    (acc & 0xffff) as i16
}

/// The value the decoder reconstructs for quantiser index `q`.
fn reconstruct(q: i16, std: i16, num_bits: i16) -> i16 {
    let width = cell(std, num_bits);
    let max = (1i32 << num_bits) as i16;
    let level = ((i32::from(q) << 17)
        .wrapping_sub(i32::from(max) << 16)
        .wrapping_add(0x1_0000)
        >> 16) as i16;
    let prod = i32::from(level)
        .wrapping_mul(i32::from(width))
        .wrapping_mul(2);
    (l_shl_sat(prod, 14 - num_bits) >> 16) as i16
}

/// The frame's six prediction-residual block lengths.
fn block_lengths(nh: i16) -> [usize; NUM_PRED_RES_BLKS] {
    let base = (nh / 6) as usize;
    let short_blocks = (6 * base + 6).saturating_sub(nh as usize);
    let mut out = [base; NUM_PRED_RES_BLKS];
    for (b, len) in out.iter_mut().enumerate() {
        if b >= short_blocks {
            *len = base + 1;
        }
    }
    out
}

/// Encode one frame's amplitudes and advance the predictor memory.
///
/// `amps` is the reference's own log-domain band vector (`params[+0x10]`, Q11,
/// one word per harmonic, before the unvoiced-band offset); `voic` is `b̂₁` and
/// `pitch_word` is `params[+0xc]`. Returns the eight info words.
pub(crate) fn encode_frame(
    st: &mut RefAmpState,
    b0: u8,
    nh: i16,
    voic: i16,
    amps: &[i16],
    pitch_word: u16,
    sync: u8,
) -> [u16; 8] {
    let nb = num_bands(nh) as i16;
    let mut bit_alloc = [0i16; B_NUM + 4];
    get_bit_allocation(nh, &mut bit_alloc);

    // The frame's amplitude vector, unvoiced bands lowered by the pitch-keyed
    // log offset.
    let mut a = [0i16; NUM_HARMS_MAX];
    a[..nh as usize].copy_from_slice(&amps[..nh as usize]);
    apply_unvoiced_offset(&mut a, nh, voic, unvoiced_log_offset(pitch_word));

    // The previous frame's reconstruction, offset removed with the words that
    // frame carried, then resampled and scaled by rho.
    let mut prev = st.amps_prev;
    apply_unvoiced_offset(
        &mut prev,
        st.num_harms_prev,
        st.voicing_prev,
        unvoiced_log_offset(st.pitch_word_prev),
    );
    let pred = prediction(nh, &prev, st.num_harms_prev);

    // Residual.
    for i in 0..nh as usize {
        a[i] = (i32::from(a[i]) - i32::from(pred[i])).clamp(-0x7fff, 0x7fff) as i16;
    }

    // Per-block forward DCT, in place: each block's DC goes to the gain vector.
    let lengths = block_lengths(nh);
    let mut gain_vec = [0i16; NUM_PRED_RES_BLKS];
    let mut off = 0usize;
    for (b, &len) in lengths.iter().enumerate() {
        let mut c = [0i16; 10];
        dct(&a[off..off + len], len as i16, len as i16, &mut c);
        a[off..off + len].copy_from_slice(&c[..len]);
        gain_vec[b] = c[0];
        off += len;
    }

    let mut b_vec = [0i16; NUM_HARMS_MAX + 3];
    b_vec[0] = i16::from(b0);
    b_vec[1] = voic;

    // Gain DCT: the DC becomes b̂₂, the other five are quantised in place.
    let mut gain_r = [0i16; NUM_PRED_RES_BLKS];
    dct(
        &gain_vec,
        NUM_PRED_RES_BLKS as i16,
        NUM_PRED_RES_BLKS as i16,
        &mut gain_r,
    );
    let mut gain_dc = gain_r[0];
    gain_r[0] = 0;
    for i in 0..5usize {
        let bits = bit_alloc[i];
        let q = quantise(gain_r[i + 1], GAIN_STD_TBL[i], bits);
        b_vec[3 + i] = q;
        gain_r[i + 1] = reconstruct(q, GAIN_STD_TBL[i], bits);
    }

    // Block coefficients, quantised and reconstructed in place.
    let mut ba = 5usize;
    let mut bv = 8usize;
    let mut off = 0usize;
    for &len in lengths.iter() {
        for j in 1..len {
            let bits = bit_alloc[ba];
            ba += 1;
            if bits < 1 {
                a[off + j] = 0;
                b_vec[bv] = 0;
            } else {
                let std = HI_ORD_STD_TBL[j - 1] as i16;
                let q = quantise(a[off + j], std, bits);
                b_vec[bv] = q;
                a[off + j] = reconstruct(q, std, bits);
            }
            bv += 1;
        }
        off += len;
    }

    // Rebuild the residual from what was transmitted.
    idct(
        &mut gain_r,
        NUM_PRED_RES_BLKS as i16,
        NUM_PRED_RES_BLKS as i16,
    );
    let mut off = 0usize;
    for (b, &len) in lengths.iter().enumerate() {
        a[off] = gain_r[b];
        idct(&mut a[off..off + len], len as i16, len as i16);
        off += len;
    }

    let b2 = ref_gain_dc_quant(gain_dc);
    b_vec[2] = b2;
    gain_dc = GAIN_QNT_TBL[b2 as usize];

    // Reconstruction: residual + prediction + gain, at the reference's clamp.
    for i in 0..nh as usize {
        let sum = ((i32::from(a[i]) << 16) >> 2)
            .wrapping_add((i32::from(pred[i]) << 16) >> 2)
            .wrapping_add((i32::from(gain_dc) << 16) >> 2);
        a[i] = ((l_shl_sat(sum, 2) >> 16) as i16).clamp(-0x7800, 0x77ff);
    }

    b_vec[nh as usize + 2] = i16::from(sync & 1);
    let u = encode_frame_vector(&b_vec, nh, nb, &bit_alloc);

    // Advance the predictor memory: the offset goes back on, then the tail is
    // replicated from the last harmonic.
    apply_unvoiced_offset_back(&mut a, nh, voic, unvoiced_log_offset(pitch_word));
    let last = a[nh as usize - 1];
    for i in nh as usize..NUM_HARMS_MAX {
        a[i] = last;
    }
    st.amps_prev = a;
    st.num_harms_prev = nh;
    st.pitch_word_prev = pitch_word;
    st.voicing_prev = voic;
    u
}

/// [`apply_unvoiced_offset`] with the sign flag set: the offset is added back.
fn apply_unvoiced_offset_back(words: &mut [i16], nh: i16, voic: i16, delta: i16) {
    apply_unvoiced_offset(words, nh, voic, delta.wrapping_neg());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One frame of the reference encoder's own amplitude back end, captured
    /// under a live hook on `FUN_10311fa0`: the log-domain band vector it was
    /// handed (`params[+0x10]`, before the unvoiced-band offset), the pitch
    /// word, `b0`, `b1` and the synchronisation bit it carried, and the eight
    /// info words it emitted.
    struct RefFrame {
        l: i16,
        pitch_word: u16,
        voicing: i16,
        b0: u8,
        sync: u8,
        amps: &'static [i16],
        u: [u16; 8],
    }

    const REFERENCE_FRAMES: [RefFrame; 6] = [
        RefFrame {
            l: 14,
            pitch_word: 16256,
            voicing: 0,
            b0: 25,
            sync: 0,
            amps: &[
                -30550, -30550, -30550, -30550, -30550, -30550, -30550, -30550, -30550, -30550,
                -30550, -30550, -30550, -30550,
            ],
            u: [387, 3730, 801, 3648, 1, 258, 1536, 82],
        },
        RefFrame {
            l: 48,
            pitch_word: 5053,
            voicing: 0,
            b0: 168,
            sync: 1,
            amps: &[
                -2033, -2054, -2582, -2087, -2288, -3180, -3923, -5756, -4383, -4726, -4900, -3930,
                -2367, -4034, -704, -1691, -2808, -3636, -1783, -1441, -1288, -3238, -4083, -3105,
                -3098, -4478, -1345, 338, 733, -1659, -6056, -4214, -3653, -5227, -5625, -5554,
                -3486, -2955, -2519, -3621, -1003, -456, 791, -1343, -1026, -3740, -3262, -4023,
            ],
            u: [2698, 2757, 3379, 3219, 0, 69, 1231, 1],
        },
        RefFrame {
            l: 44,
            pitch_word: 5418,
            voicing: 0,
            b0: 154,
            sync: 0,
            amps: &[
                10647, 14287, 15830, 15054, 13158, 11412, 10699, 9348, 6616, 6090, 6648, 5896,
                6237, 5785, 4189, 5858, 7170, 8058, 7383, 4300, 2777, 4411, 4241, 5007, 4924, 4721,
                6756, 7008, 4232, 2011, 1721, 1850, 2813, 1018, 1056, 906, 1015, 2502, 5209, 6506,
                4345, 4148, 3977, 3036,
            ],
            u: [2471, 4065, 3883, 2097, 0, 1, 1500, 52],
        },
        RefFrame {
            l: 47,
            pitch_word: 5102,
            voicing: 4042,
            b0: 166,
            sync: 1,
            amps: &[
                16007, 18687, 18493, 17719, 15138, 13277, 11844, 12169, 11672, 10114, 9338, 7191,
                6363, 7269, 8789, 8987, 10688, 12587, 13449, 11181, 9360, 7184, 9198, 9096, 9765,
                10440, 12486, 13294, 13249, 11418, 9318, 8347, 8179, 7250, 7158, 7080, 8208, 8713,
                9555, 8482, 8692, 8649, 8798, 8372, 5586, 5025, 3217,
            ],
            u: [2669, 3569, 1063, 1831, 2021, 814, 1465, 53],
        },
        RefFrame {
            l: 47,
            pitch_word: 5102,
            voicing: 4093,
            b0: 166,
            sync: 0,
            amps: &[
                15524, 19201, 18038, 18396, 17040, 14694, 13094, 9495, 10417, 10369, 11127, 8727,
                7919, 10611, 10927, 11229, 11918, 14132, 14391, 12543, 10540, 8381, 9654, 11048,
                11024, 11602, 13908, 15411, 15208, 9757, 9149, 9202, 8569, 9836, 8912, 9043, 9619,
                10336, 10858, 10013, 7882, 10831, 11142, 9388, 5229, 5865, 4724,
            ],
            u: [2679, 116, 1312, 2106, 2046, 1195, 423, 28],
        },
        RefFrame {
            l: 47,
            pitch_word: 5127,
            voicing: 4080,
            b0: 165,
            sync: 1,
            amps: &[
                16402, 19473, 17639, 17622, 16592, 14809, 13054, 11152, 9323, 9989, 10425, 10762,
                10906, 11362, 11365, 11880, 12853, 14850, 14980, 14664, 12729, 11982, 10826, 10903,
                11340, 13197, 14468, 15233, 14972, 12474, 12074, 8880, 9312, 9700, 10261, 10594,
                11183, 11160, 11396, 11099, 8571, 9879, 11165, 10508, 8163, 8267, 7523,
            ],
            u: [2677, 2415, 2074, 2093, 2040, 469, 1657, 123],
        },
    ];

    /// The back end reproduces the reference's own frames bit for bit, running
    /// its predictor memory forward from the reference's initial state -- so the
    /// cross-frame reconstruction is pinned too, not only the arithmetic. The
    /// full capture is 9477 frames over mark/dam/clean/noisy, all exact.
    #[test]
    fn reproduces_the_reference_frames() {
        let mut st = RefAmpState::new();
        for (i, f) in REFERENCE_FRAMES.iter().enumerate() {
            let u = encode_frame(&mut st, f.b0, f.l, f.voicing, f.amps, f.pitch_word, f.sync);
            assert_eq!(u, f.u, "frame {i}");
        }
    }

    /// Length 1 is the degenerate DCT: one coefficient, scaled by 32767/32768.
    #[test]
    fn dct_of_a_single_sample_is_almost_the_sample() {
        let mut out = [0i16; 4];
        dct(&[1000], 1, 1, &mut out);
        assert_eq!(out[0], 1000);
    }

    /// The DCT and its inverse round-trip a constant block: a flat input has
    /// only a DC coefficient, and the inverse puts the constant back.
    #[test]
    fn dct_round_trips_a_flat_block() {
        let input = [700i16; 8];
        let mut c = [0i16; 8];
        dct(&input, 8, 8, &mut c);
        assert_eq!(&c[1..], &[0i16; 7], "a flat block has no AC energy");
        let mut x = c;
        idct(&mut x, 8, 8);
        for v in x {
            assert!((v - 700).abs() <= 1, "round trip gave {v}");
        }
    }

    /// The quantiser is the exact left-inverse of the reconstruction: every
    /// index's own reconstruction quantises back to that index.
    #[test]
    fn quantise_inverts_reconstruct_on_every_cell() {
        for num_bits in 1..=6i16 {
            for std in [20316i16, 13173, 11338, 10813] {
                for q in 0..(1i16 << num_bits) {
                    let v = reconstruct(q, std, num_bits);
                    assert_eq!(
                        quantise(v, std, num_bits),
                        q,
                        "bits {num_bits} std {std} q {q}"
                    );
                }
            }
        }
    }

    /// Out-of-range values saturate to the end cells rather than wrapping.
    #[test]
    fn quantise_saturates_at_both_ends() {
        let (std, bits) = (20316i16, 4i16);
        let width = cell(std, bits);
        assert_eq!(quantise(-width, std, bits), 0, "below the first cell");
        assert_eq!(quantise(width, std, bits), 15, "above the last cell");
        assert_eq!(quantise(i16::MIN, std, bits), 0);
    }

    /// The six block lengths always partition the harmonics and never differ by
    /// more than one.
    #[test]
    fn block_lengths_partition_the_harmonics() {
        for nh in 9..=56i16 {
            let l = block_lengths(nh);
            assert_eq!(l.iter().sum::<usize>(), nh as usize, "nh {nh}");
            assert!(l[5] - l[0] <= 1, "nh {nh} lengths {l:?}");
            assert!(l.windows(2).all(|w| w[0] <= w[1]), "nh {nh} lengths {l:?}");
        }
    }

    /// The predictor memory starts where the reference's does, so the first
    /// frame's prediction is the reference's.
    #[test]
    fn initial_state_matches_the_reference() {
        let st = RefAmpState::new();
        assert_eq!(st.num_harms_prev, 15);
        assert_eq!(st.pitch_word_prev, 16423);
        assert_eq!(st.voicing_prev, 0);
        assert_eq!(st.amps_prev, [0i16; NUM_HARMS_MAX]);
    }

    /// `L_shl` saturates rather than dropping the sign bit.
    #[test]
    fn shifter_saturates() {
        assert_eq!(l_shl_sat(0x4000_0000, 1), i32::MAX);
        assert_eq!(l_shl_sat(-0x4000_0001, 1), i32::MIN);
        assert_eq!(l_shl_sat(0x1000, 1), 0x2000);
        assert_eq!(l_shl_sat(0x1000, -1), 0x800);
    }
}
