// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! IMBE full-rate **quantizer** (encoder back half): MBE [`MbeParams`] → b-vector
//! indices → 8 info words → 144-bit frame. The exact inverse of
//! [`super::dequantize`], using the **firmware MONO65** gain ladder
//! ([`super::tables::MONO65_ENCODE`]). Correctness is proven by idempotence
//! (decode → re-quantize reproduces the bits), the same standard used for the
//! AMBE+2 quantizer. Fixed-point throughout, mirroring the C55x semantics.

use super::fixp::*;
use super::frame::{enframe, ImbeFrame, U_WIDTHS};
use super::math::{cos_fxp, l_mpy_ls};
use super::tables::{
    BIT_ALLOCATION_OFFSET_TBL, BIT_ALLOCATION_TBL, GAIN_STEP_SIZE_TBL, HI_ORD_STD_TBL,
    HI_ORD_STEP_SIZE_TBL, LMPRBL_TBL, MONO65_ENCODE,
};

const NUM_HARMS_MAX: usize = 56;
const NUM_HARMS_MIN: i16 = 9;
const NUM_BANDS_MAX: i16 = 12;
const NUM_PRED_RES_BLKS: usize = 6;
const BIT_STREAM_LEN: usize = 75;
const B_NUM: usize = NUM_HARMS_MAX - 1;

const CNST_0_33_Q0_16: u32 = 0x5556;
const CNST_ONE_Q8_24: i32 = 0x0100_0000;
const CNST_0_7_Q1_15: i16 = 0x599A;
const CNST_0_4_Q1_15: i16 = 0x3333;
const CNST_0_03_Q1_15: i32 = 0x03D7;
const CNST_0_05_Q1_15: i32 = 0x0666;
const CNST_0_5_Q1_15: u16 = 0x4000;
const CNST_1_0_Q1_15: u16 = 0x7FFF;
const CNST_0_5_Q5_11: i16 = 0x0400;

// ---- forward DCT (dsp_sub.cc::dct), the partner of idct ---------------------

fn dct(input: &[i16], m_lim: i16, i_lim: i16, out: &mut [i16]) {
    let (angl_intl, angl_intl_2): (u16, u16) = if m_lim == 1 {
        (CNST_0_5_Q1_15, CNST_1_0_Q1_15)
    } else {
        let a = div_s(CNST_0_5_Q5_11, m_lim << 11) as u16;
        (a, a.wrapping_shl(1))
    };
    // coeff 0
    let mut sum: i32 = 0;
    for m in 0..m_lim as usize {
        sum = l_add(sum, l_deposit_l(input[m]));
    }
    out[0] = extract_l(l_mpy_ls(sum, angl_intl_2 as i16));
    let mut angl_begin = angl_intl;
    let mut angl_step = angl_intl_2;
    for i in 1..i_lim as usize {
        let mut s: i32 = 0;
        let mut angl_acc = angl_begin;
        for m in 0..m_lim as usize {
            s = l_add(s, l_deposit_l(mult(input[m], cos_fxp(angl_acc as i16))));
            angl_acc = angl_acc.wrapping_add(angl_step);
        }
        out[i] = extract_l(l_mpy_ls(s, angl_intl_2 as i16));
        angl_step = angl_step.wrapping_add(angl_intl_2);
        angl_begin = angl_begin.wrapping_add(angl_intl);
    }
}

// ---- forward scalar / table quantizers (qnt_sub.cc) ------------------------

fn qnt_by_step(val: i16, step_size: u16, bit_num: i16) -> i16 {
    if bit_num == 0 {
        return 0;
    }
    let shift = norm_s(step_size as i16);
    let tmp = div_s(0x4000, shl(step_size as i16, shift));
    // Mid-rise inverse: deqnt reconstructs at cell-centre (val = step*m + step/2),
    // so the encoder must FLOOR (truncate toward -inf), not round-half-up, or every
    // cell-centre value tips to m+1. Using `shr` (arithmetic) instead of `shr_r`
    // makes qnt the exact left-inverse of deqnt_by_step.
    let q_index = shr(mult(val, tmp), sub(9, shift));
    let max_val = 1i16 << (bit_num - 1);
    let min_val = negate(max_val);
    if q_index < min_val {
        0
    } else if q_index >= max_val {
        (1i16 << bit_num) - 1
    } else {
        max_val + q_index
    }
}

fn tbl_quant(val: i16, tbl: &[i16]) -> i16 {
    let mut lo = 0usize;
    let mut hi = tbl.len() - 1;
    if val >= tbl[hi] {
        return hi as i16;
    }
    if val <= tbl[lo] {
        return lo as i16;
    }
    while hi - lo != 1 {
        let mid = lo + ((hi - lo) >> 1);
        if tbl[mid] > val {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    if tbl[hi] - val <= val - tbl[lo] {
        hi as i16
    } else {
        lo as i16
    }
}

// ---- bit allocation (shared with decode) -----------------------------------

fn get_bit_allocation(num_harms: i16, ptr: &mut [i16]) {
    let mut bat = if num_harms == NUM_HARMS_MIN {
        0
    } else {
        let index = (num_harms - NUM_HARMS_MIN - 1) as usize;
        BIT_ALLOCATION_OFFSET_TBL[index >> 2] as usize + (3 + (index >> 2)) * (index & 0x3)
    };
    let mut o = 0usize;
    let mut i = 0i16;
    while i < num_harms - 1 {
        let tmp = BIT_ALLOCATION_TBL[bat];
        bat += 1;
        ptr[o + 3] = (tmp & 0xF) as i16;
        let tmp = tmp >> 4;
        ptr[o + 2] = (tmp & 0xF) as i16;
        let tmp = tmp >> 4;
        ptr[o + 1] = (tmp & 0xF) as i16;
        let tmp = tmp >> 4;
        ptr[o] = (tmp & 0xF) as i16;
        o += 4;
        i += 4;
    }
}

// ---- Log2 (math_sub.cc), used by the forward amplitude transform -----------

use super::tables::LOG_TABLE;
fn log2(x: i16) -> i32 {
    if x <= 0 {
        return 0;
    }
    let exp = norm_s(x);
    let x = shl(x, exp);
    let i = shr(x, 9) as usize;
    let a = shl(x & 0x1FF, 6);
    let i = i - 32;
    let mut l_y = l_deposit_h(LOG_TABLE[i]);
    let tmp = sub(LOG_TABLE[i], LOG_TABLE[i + 1]);
    l_y = l_msu(l_y, tmp, a);
    l_y = l_shr(l_y, 9);
    let exp = sub(12, exp);
    l_add(l_y, l_deposit_h(shl(exp, 6)))
}

// ---- forward spectral-amplitude encode (inverse of sa_decode) --------------

/// Per-frame source RMS below which an IMBE frame is treated as inter-word
/// silence for the low-energy gain floor in [`sa_encode`]. Mirrors
/// `blip25_mbe::vocoder::IMBE_SILENCE_RMS` (the same threshold that pins the
/// silence pitch `b0`); the caller passes the RMS it already computed.
pub(crate) const IMBE_SILENCE_RMS: f32 = 400.0;

/// Produce gain index `b_vec[2]` and the L−1 amplitude-coefficient indices
/// (`b_vec[3..]`) from the target log2-amplitudes `la` (Q10.22, full precision),
/// using the prediction state `sa_prev`/`num_harms_prev` (which must be the SAME
/// pre-frame state the decoder used — for idempotence, the decoder's snapshot).
#[allow(clippy::too_many_arguments)]
fn sa_encode(
    la: &[i32],
    num_harms: i16,
    sa_prev: &[i32; NUM_HARMS_MAX + 2],
    num_harms_prev: i16,
    bit_alloc: &[i16],
    b_vec: &mut [i16],
    mech: bool,
    src_rms: f32,
) {
    let nh = num_harms;
    let nhp = num_harms_prev;
    let index = (nh - NUM_HARMS_MIN) as usize;

    // prediction (mirror decode exactly), accumulating pred[i] and sum.
    let k_coef: u32 = if nh == nhp {
        CNST_ONE_Q8_24 as u32
    } else if nh > nhp {
        (div_s(nhp << 9, nh << 9) as u32) << 9
    } else {
        let mut kc: u32 = 0;
        let mut tmp = nhp;
        while tmp > nh {
            tmp -= nh;
            kc = kc.wrapping_add(CNST_ONE_Q8_24 as u32);
        }
        kc.wrapping_add((div_s(tmp << 9, nh << 9) as u32) << 9)
    };
    let ro_coef: i16 = if nh <= 15 {
        CNST_0_4_Q1_15
    } else if nh <= 24 {
        (nh as i32 * CNST_0_03_Q1_15 - CNST_0_05_Q1_15) as i16
    } else {
        CNST_0_7_Q1_15
    };

    let mut prev = *sa_prev;
    for i in (nhp as usize + 1)..(NUM_HARMS_MAX + 2) {
        prev[i] = prev[nhp as usize];
    }

    let mut k_acc: u32 = k_coef;
    let mut sum: i32 = 0;
    let mut pred = [0i32; NUM_HARMS_MAX];
    for i in 0..nh as usize {
        let idx = (k_acc >> 24) as usize;
        let si_coef = ((k_acc - ((idx as u32) << 24)) >> 9) as i16;
        if si_coef == 0 {
            pred[i] = l_mpy_ls(prev[idx], ro_coef);
            sum = l_add(sum, prev[idx]);
        } else {
            let tw = l_mpy_ls(prev[idx], sub(0x7FFF, si_coef));
            sum = l_add(sum, tw);
            let tw2 = l_mpy_ls(prev[idx + 1], si_coef);
            sum = l_add(sum, tw2);
            pred[i] = l_add(l_mpy_ls(tw, ro_coef), l_mpy_ls(tw2, ro_coef));
        }
        k_acc = k_acc.wrapping_add(k_coef);
    }
    let sh = norm_s(nh);
    let one_by_l = div_s(0x4000, shl(nh, sh));
    sum = l_shr(l_mpy_ls(l_mpy_ls(sum, ro_coef), one_by_l), 14 - sh);

    // residual t_vec (Q5.11) per harmonic. `la` is the target log2-amplitude
    // (Q10.22) — full precision, so the inverse is exact (no Q14.2 zero loss).
    let mut t_vec = [0i16; NUM_HARMS_MAX];
    for i in 0..nh as usize {
        let v = l_sub(la[i], pred[i]); // vec32_tmp
        t_vec[i] = extract_h(l_shl(l_add(v, sum), 5));
    }

    // Per-block forward DCT: block DCs -> gain_vec, higher coeffs quantized.
    let mut lmprbl_item = LMPRBL_TBL[index];
    let mut gain_vec = [0i16; 6];
    let mut ba_blk = 5usize; // block coeffs use bit_alloc[5..]
    let mut bv_blk = 8usize; // and b_vec[8..]
    let mut t_off = 0usize;
    for blk in 0..NUM_PRED_RES_BLKS {
        let bl_len = ((lmprbl_item >> 28) & 0xF) as usize;
        lmprbl_item <<= 4;
        let mut c_vec = [0i16; 10];
        dct(
            &t_vec[t_off..t_off + bl_len],
            bl_len as i16,
            bl_len as i16,
            &mut c_vec,
        );
        gain_vec[blk] = c_vec[0];
        for j in 1..bl_len {
            let num_bits = bit_alloc[ba_blk];
            ba_blk += 1;
            if num_bits != 0 {
                let step_size = extract_h(
                    (HI_ORD_STD_TBL[j - 1] as i32
                        * HI_ORD_STEP_SIZE_TBL[(num_bits - 1) as usize] as i32)
                        << 1,
                );
                b_vec[bv_blk] = qnt_by_step(c_vec[j], step_size as u16, num_bits);
            } else {
                b_vec[bv_blk] = 0;
            }
            bv_blk += 1;
        }
        t_off += bl_len;
    }

    // Gain: DCT the block DCs, table-quantize the DC, step-quantize the rest.
    let mut gain_r = [0i16; 6];
    dct(
        &gain_vec,
        NUM_PRED_RES_BLKS as i16,
        NUM_PRED_RES_BLKS as i16,
        &mut gain_r,
    );
    // Gain-DC bias, regime-dependent on the amplitude source (`mech`):
    //
    //  - `mech` (whole-buffer, reference fixed-point spectrum mechanism): the
    //    voiced-frame gain DC is already centred on reference's MONO65 cells — the
    //    per-frame ideal offset (`gain_r[0] - MONO65[reference_b2]`) has a median
    //    within one cell of zero on all four reference vectors — so the voiced DC
    //    correction is 0, and the low-L/unvoiced floor is ~+2100 Q5.11 (ideal
    //    median +1988..+2073, tightly consistent across the four vectors).
    //  - non-`mech` (streaming, synthesized-DFT proxy): the proxy envelope runs
    //    hot, needing the empirical +300 voiced DC and +2350 low-L floor these
    //    values had before the mechanism landed.
    //
    // The two amplitude sources sit at different absolute levels, so a single
    // bias set cannot serve both; `mech` selects the matching pair.
    let (dc_bias, lowl_bias): (i16, i16) = if mech { (0, 2100) } else { (300, 2350) };
    let mut gdc = gain_r[0].saturating_sub(dc_bias);
    // Energy-gated floor correction. On the low-energy regime — the quiet /
    // aperiodic inter-word frames the b0 silence gate pins to the default pitch
    // (b0=25 => L=14) — the envelope FLOOR sits further above reference, so these
    // frames need an additional constant subtraction. Two conditions gate it: low
    // SOURCE energy (`src_rms` below the silence threshold) and fully unvoiced
    // (b_vec[1]==0, the packed per-band V/UV mask, set above before this call).
    // The energy gate — not a harmonic-count proxy — is what keeps this off the
    // high-energy fricatives (their SH/T noise reads unvoiced but sits well above
    // the silence floor), while still catching the low-energy word-edge frames
    // (L>=15) that a bare nh<15 proxy missed and that seed the decoder's
    // differential-amplitude predictor into the gap.
    if src_rms < IMBE_SILENCE_RMS && b_vec[1] == 0 {
        gdc = gdc.saturating_sub(lowl_bias);
    }
    b_vec[2] = tbl_quant(gdc, &MONO65_ENCODE);
    let gss = &GAIN_STEP_SIZE_TBL[index * 5..];
    for j in 1..6 {
        b_vec[2 + j] = qnt_by_step(gain_r[j], gss[j - 1], bit_alloc[j - 1]);
    }
}

// ---- forward channel encode (inverse of decode_frame_vector) ---------------

/// Pack the b-vector indices into the 8 info words (u0..u7), the inverse of
/// `decode_frame_vector`'s priority rescan + word assembly.
fn encode_frame_vector(
    b_vec: &[i16],
    num_harms: i16,
    num_bands: i16,
    bit_alloc: &[i16],
) -> [u16; 8] {
    let k = num_bands as usize;
    let mut bs = [0i16; BIT_STREAM_LEN];

    // Reverse the priority rescan: emit each coefficient's bits MSB-first by
    // descending threshold into the (compacted) amplitude region [0..73-K].
    let mut idx = 0usize;
    let mut bit_thr = if num_harms == 0xb { 9 } else { bit_alloc[0] };
    let limit = BIT_STREAM_LEN - k - 2;
    while idx < limit {
        for i in 0..(num_harms as usize - 1) {
            if bit_thr != 0 && bit_thr <= bit_alloc[i] {
                bs[idx] = (b_vec[3 + i] >> (bit_thr - 1)) & 1;
                idx += 1;
            }
        }
        bit_thr -= 1;
    }

    // Un-compact: shift the amplitude tail back up to make room for b1/b2, which
    // sit at stream positions [39..39+K] and [39+K..39+K+2].
    let shiftn = k + 2;
    let b1_pos = 3 + 3 * 12; // 39
                             // move [b1_pos .. limit] up to [b1_pos+shiftn ..] from the top down
    let mut j = limit;
    while j > b1_pos {
        j -= 1;
        bs[j + shiftn] = bs[j];
        bs[j] = 0;
    }
    // place b1 (K bits) then b2's two stream bits. Decode reads them as
    // `t = bs[idx]<<1 | bs[idx+1]; b2 |= t<<1`, i.e. bs[idx]->b2 bit2,
    // bs[idx+1]->b2 bit1 (bit0 and bits3..5 are the scattered bits below).
    let mut p = b1_pos;
    for bit in 0..k {
        bs[p] = (b_vec[1] >> (k - 1 - bit)) & 1;
        p += 1;
    }
    bs[p] = (b_vec[2] >> 2) & 1;
    p += 1;
    bs[p] = (b_vec[2] >> 1) & 1;

    // Now distribute the bit stream + scattered bits into the 8 info words.
    let mut u = [0u16; 8];

    // b0 (pitch) -> fv[0] high bits + fv[7] low bits.
    let b0 = b_vec[0];
    u[0] |= ((b0 & 0xFC) as u16) << 4;
    u[7] |= ((b0 & 0x3) as u16) << 1;

    // b2 scattered bits -> fv[0]&0x38 and fv[7] bit3.
    u[0] |= (b_vec[2] as u16) & 0x38;
    u[7] |= ((b_vec[2] as u16) & 0x01) << 3;

    // bit_stream[0..3] -> fv[0] bits 2,1,0.
    if bs[0] != 0 {
        u[0] |= 0x4;
    }
    if bs[1] != 0 {
        u[0] |= 0x2;
    }
    if bs[2] != 0 {
        u[0] |= 0x1;
    }
    // bit_stream tail [72..75] -> fv[7] bits 6,5,4.
    if bs[BIT_STREAM_LEN - 3] != 0 {
        u[7] |= 0x40;
    }
    if bs[BIT_STREAM_LEN - 2] != 0 {
        u[7] |= 0x20;
    }
    if bs[BIT_STREAM_LEN - 1] != 0 {
        u[7] |= 0x10;
    }
    // sync bit.
    u[7] |= (b_vec[num_harms as usize + 2] as u16) & 1;

    // bit_stream[3..39] -> fv[3],fv[2],fv[1] (LSB-first fill, mirroring decode).
    let mut index0: i32 = 3 + 3 * 12 - 1;
    for vec_num in (1..=3).rev() {
        let mut w = 0u16;
        // decode filled bit_stream[index0] = fv&1 then fv>>=1, index0--; so the
        // LAST bit placed (index0 smallest) is the MSB. Reconstruct by reading
        // bits from index0 down.
        for b in 0..12 {
            let bit = bs[(index0 - b) as usize] as u16;
            w |= bit << b;
        }
        u[vec_num] = w;
        index0 -= 12;
    }
    index0 = 3 + 3 * 12 + 3 * 11 - 1;
    for vec_num in (4..=6).rev() {
        let mut w = 0u16;
        for b in 0..11 {
            let bit = bs[(index0 - b) as usize] as u16;
            w |= bit << b;
        }
        u[vec_num] = w;
        index0 -= 11;
    }

    u
}

// ---- public API ------------------------------------------------------------

/// Cross-frame encoder prediction state (mirror of the decoder's, kept in sync
/// by re-running the decode on the produced bits — guarantees the encoder's
/// prediction equals what the decoder will compute).
#[derive(Clone)]
pub struct ImbeEncState {
    st: super::dequantize::ImbeState,
}

impl ImbeEncState {
    pub fn new() -> Self {
        Self {
            st: super::dequantize::ImbeState::new(),
        }
    }
}

impl Default for ImbeEncState {
    fn default() -> Self {
        Self::new()
    }
}

/// Quantize one frame to the 8 info words (u0..u7) from the **log2-amplitudes**
/// `la` (Q10.22, length `num_harms`) — the exact, lossless input. Pure (no state
/// mutation); `sa_prev`/`num_harms_prev` is the pre-frame prediction.
pub fn quantize_to_u(
    b0: u8,
    num_harms: i16,
    voiced: &[bool],
    la: &[i32],
    sa_prev: &[i32; NUM_HARMS_MAX + 2],
    num_harms_prev: i16,
) -> [u16; 8] {
    // Generic non-mech entry point (no live source samples): the energy-gated
    // low-L floor is off (never fires) — it is applied only on the whole-buffer /
    // streaming encode paths that pass the real per-frame source RMS.
    quantize_to_u_mech(
        b0,
        num_harms,
        voiced,
        la,
        sa_prev,
        num_harms_prev,
        false,
        f32::MAX,
        0,
    )
}

/// As [`quantize_to_u`], with `mech` selecting the gain-DC bias regime in
/// [`sa_encode`]: `true` for the mechanism-sourced (pitch-aligned fixed-point
/// spectrum) amplitude level, `false` for the synthesized-DFT proxy level.
///
/// `sync` is `b̂_{L+2}`, the frame's synchronisation bit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantize_to_u_mech(
    b0: u8,
    num_harms: i16,
    voiced: &[bool],
    la: &[i32],
    sa_prev: &[i32; NUM_HARMS_MAX + 2],
    num_harms_prev: i16,
    mech: bool,
    src_rms: f32,
    sync: u8,
) -> [u16; 8] {
    let nh = num_harms;
    let num_bands = if nh <= 36 {
        extract_h(((nh as u32 + 2) * CNST_0_33_Q0_16) as i32)
    } else {
        NUM_BANDS_MAX
    };
    let mut bit_alloc = [0i16; B_NUM + 4];
    get_bit_allocation(nh, &mut bit_alloc);

    let mut b_vec = [0i16; NUM_HARMS_MAX + 3];
    b_vec[0] = b0 as i16;
    // b1: per-band V/UV bit = band's (first) harmonic voicing.
    let mut b1 = 0i16;
    for band in 0..num_bands as usize {
        let h = (band * 3).min(nh as usize - 1);
        let v = if voiced[h] { 1 } else { 0 };
        b1 = (b1 << 1) | v;
    }
    b_vec[1] = b1;

    sa_encode(
        la,
        nh,
        sa_prev,
        num_harms_prev,
        &bit_alloc,
        &mut b_vec,
        mech,
        src_rms,
    );
    b_vec[nh as usize + 2] = i16::from(sync & 1);
    encode_frame_vector(&b_vec, nh, num_bands, &bit_alloc)
}

/// Convert linear amplitudes `M_l` (Q14.2) to log2-amplitudes (Q10.22) for the
/// encoder. Real analysis amplitudes carry full precision; amplitudes that round
/// to 0 in Q14.2 (linear < 0.25) are floored to a very small log value.
pub(crate) fn amps_to_la(sa_q142: &[i16]) -> Vec<i32> {
    sa_q142
        .iter()
        .map(|&s| if s > 0 { log2(s) } else { -(20i32 << 22) })
        .collect()
}

// ---- IMBE pitch quantizer (inverse of the decoder's b0 -> fund_freq/L) ------

const CNST_0_9254_Q0_16: u32 = 60647;

/// The decoder's fundamental frequency + harmonic count for pitch index `b0`.
///
/// Exact fixed-point replica of the `b0 -> fund_freq`/`b0 -> num_harms`
/// computation in [`super::dequantize`]'s `decode_frame_vector` (the only
/// authority on the IMBE pitch mapping). Kept here so the encoder can invert it
/// bit-consistently rather than re-deriving from a float formula.
pub fn pitch_cell(b0: i16) -> (i32, i16) {
    let mut tmp = ((b0 & 0xFF) << 1) + 0x4F; // 2*(b0+39.5), Q15.1
    let shift = norm_s(tmp);
    let tmp1 = shl(tmp, shift);

    let mut tmp2 = div_s(0x4000, tmp1);
    let mut fund_freq = l_shr(l_deposit_h(tmp2), 11 - shift);
    let l_tmp = l_sub(0x4000_0000, l_mult(tmp1, tmp2));
    tmp2 = div_s(extract_l(l_shr(l_tmp, 2)), tmp1);
    let l_tmp = l_shr(l_deposit_l(tmp2), 11 - shift - 2);
    fund_freq = l_add(fund_freq, l_tmp);

    tmp = (tmp + 0x2) >> 3; // (b0 + 40.5)/4
    let num_harms = ((CNST_0_9254_Q0_16 * tmp as u32) >> 16) as i16;
    (fund_freq, num_harms)
}

/// Number of IMBE voicing bands `K` for a harmonic count `num_harms`, matching
/// the value [`quantize_to_u_mech`] and the decoder compute. `K` is the width of
/// the per-band V/UV word `b1`; band `b` covers harmonics `3b .. 3b+2`.
pub fn num_bands(num_harms: i16) -> usize {
    (if num_harms <= 36 {
        extract_h(((num_harms as u32 + 2) * CNST_0_33_Q0_16) as i32)
    } else {
        NUM_BANDS_MAX
    }) as usize
}

/// The decoder's fundamental frequency ω₀ (rad/sample) for pitch index `b0`.
pub fn pitch_omega0(b0: i16) -> f64 {
    let (fund_freq, _) = pitch_cell(b0);
    std::f64::consts::PI * (fund_freq as f64) / 2_147_483_648.0
}

/// The reference's IMBE pitch quantiser: prequantiser pitch word → `b̂₀`
/// (`FUN_10311170`).
///
/// `word` is the encoder's `block[+0xc]`, the same value the AMBE+2 quantiser
/// [`crate::enc::pitch::reject_recurrence::reject_b0`] consumes. The two grids
/// differ: AMBE+2 takes `log2` of the word onto a 120-cell ladder, IMBE divides
/// into it, which is linear in the pitch period and gives 208 cells. Going
/// through the AMBE+2 index and back out through a float ω₀ therefore cannot
/// address the IMBE grid — it can only reach the 113 cells that are images of
/// AMBE+2 cells.
///
/// The divide is [`crate::shared::fractional_divide::a600`] with a `2^27`
/// numerator; the affine tail is the reference's, exactly.
pub fn imbe_b0_from_pitch_word(word: u16) -> u8 {
    let q = crate::shared::fractional_divide::a600(0x0800_0000, i32::from(word)) as i16;
    let v = (((i32::from(q) << 16) >> 5) - 0x004E_0000) >> 17;
    (v as i16).clamp(0, 0xCF) as u8
}

/// Quantize an analysis fundamental frequency `omega_0` (rad/sample) to the
/// nearest IMBE pitch index `b0`, returning `(b0, num_harms)` where `num_harms`
/// is exactly the decoder's harmonic count for that `b0`. Only voice cells whose
/// harmonic count lands in the codec's valid `[9, 56]` range are considered.
pub fn quantize_pitch(omega_0: f32) -> (u8, i16) {
    let target = omega_0 as f64;
    let mut best_b0 = 0i16;
    let mut best_nh = NUM_HARMS_MIN;
    let mut best_err = f64::INFINITY;
    // b0 in [0, 207] is the valid voice range; the decoder rejects >207.
    for b0 in 0..=207i16 {
        let (fund, nh) = pitch_cell(b0);
        if nh < NUM_HARMS_MIN || nh > NUM_HARMS_MAX as i16 {
            continue;
        }
        let w0 = std::f64::consts::PI * (fund as f64) / 2_147_483_648.0;
        let err = (w0 - target).abs();
        if err < best_err {
            best_err = err;
            best_b0 = b0;
            best_nh = nh;
        }
    }
    (best_b0 as u8, best_nh)
}

/// Encode one frame's MBE parameters to a complete 18-byte IMBE frame, advancing
/// the encoder's prediction state. `sa_q142` are the target linear amplitudes in
/// Q14.2 (= round(MbeParams.amplitudes * 4)).
pub fn quantize_frame(
    enc: &mut ImbeEncState,
    b0: u8,
    num_harms: i16,
    voiced: &[bool],
    sa_q142: &[i16],
) -> [u8; super::frame::FRAME_BYTES] {
    // Generic non-mech entry point: no source RMS, energy-gated floor off.
    quantize_frame_mech(enc, b0, num_harms, voiced, sa_q142, false, f32::MAX, 0)
}

/// As [`quantize_frame`], with `mech` selecting the gain-DC bias regime (see
/// [`quantize_to_u_mech`] / `sa_encode`): `true` for the whole-buffer reference
/// fixed-point spectrum amplitude source, `false` for the streaming
/// synthesized-DFT proxy. `sync` is the frame's synchronisation bit `b̂_{L+2}`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantize_frame_mech(
    enc: &mut ImbeEncState,
    b0: u8,
    num_harms: i16,
    voiced: &[bool],
    sa_q142: &[i16],
    mech: bool,
    src_rms: f32,
    sync: u8,
) -> [u8; super::frame::FRAME_BYTES] {
    let snap = enc.st.snapshot();
    let la = amps_to_la(sa_q142);
    let u = quantize_to_u_mech(
        b0, num_harms, voiced, &la, &snap.0, snap.1, mech, src_rms, sync,
    );
    // Advance the encoder state to mirror the decoder on the produced bits.
    let fr = ImbeFrame { u, errors: [0; 8] };
    let _ = super::dequantize::decode_params(&fr, &mut enc.st);
    debug_assert!(u
        .iter()
        .zip(U_WIDTHS.iter())
        .all(|(&v, &w)| (v as u32) < (1 << w)));
    enframe(&u)
}

#[cfg(test)]
mod pitch_word_tests {
    use super::imbe_b0_from_pitch_word;

    /// `(prequantiser pitch word, b0)` pairs read off the reference encoder's
    /// own `FUN_10311170` under a live hook, spanning the observed word range
    /// (3276..25438 across mark/dam/clean/noisy).
    const REFERENCE: [(u16, u8); 6] = [
        (3276, 207),
        (4908, 174),
        (6022, 135),
        (8494, 84),
        (12426, 45),
        (25438, 2),
    ];

    #[test]
    fn matches_the_reference_pitch_quantizer() {
        for (w, b0) in REFERENCE {
            assert_eq!(imbe_b0_from_pitch_word(w), b0, "word {w}");
        }
    }

    #[test]
    fn stays_inside_the_valid_pitch_range() {
        for w in 1..=u16::MAX {
            assert!(imbe_b0_from_pitch_word(w) <= 207);
        }
        assert_eq!(imbe_b0_from_pitch_word(0), 0);
    }

    /// Every cell the grid can address is reachable -- the property the AMBE+2
    /// index round trip does not have (it reaches 113 of 208).
    #[test]
    fn addresses_the_whole_grid() {
        let mut seen = [false; 208];
        for w in 1..=u16::MAX {
            seen[imbe_b0_from_pitch_word(w) as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }
}
