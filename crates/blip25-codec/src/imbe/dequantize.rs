// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! IMBE full-rate dequantization: deframed 88 info bits → [`MbeParams`] for the
//! shared MBE synthesis back-end.
//!
//! Implements the TIA-102.BABA / the reference 7200x4400 fixed-point IMBE decode. The
//! standard's normative reference fixes the stage decomposition (channel
//! decode, spectral-amplitude decode, enhancement) and the bit-exact
//! arithmetic, so any conforming implementation converges on this shape.
//! Because the IMBE bitstream is frozen by P25 for interoperability, this
//! dequant is identical to the reference firmware's by construction (the firmware
//! MUST decode IMBE to the standard).
//!
//! OP25's `imbe_vocoder` — another implementation of the same standard — was
//! consulted as a bit-exactness cross-check. Where the two disagree, the
//! standard plus the firmware RE wins: see `tables::GAIN_QNT_TBL`, which is
//! firmware-corrected against a value OP25 gets wrong.
//!
//! Cross-frame state ([`ImbeState`]) holds the previous frame's log-amplitudes
//! and harmonic count for the §7 prediction. The C55x fixed-point semantics are
//! reproduced exactly via [`super::fixp`] / [`super::math`] / [`super::dsp`].

use core::f64::consts::PI;

use crate::dequantize::MbeParams;

use super::dsp::idct;
use super::fixp::*;
use super::frame::ImbeFrame;
use super::math::{cos_fxp, l_mpy_ls, pow2, sqrt_l_exp};
use super::tables::{
    BIT_ALLOCATION_OFFSET_TBL, BIT_ALLOCATION_TBL, GAIN_QNT_TBL, GAIN_STEP_SIZE_TBL,
    HI_ORD_STD_TBL, HI_ORD_STEP_SIZE_TBL, LMPRBL_TBL,
};

const NUM_HARMS_MAX: usize = 56;
const NUM_HARMS_MIN: i16 = 9;
const NUM_BANDS_MAX: i16 = 12;
const NUM_PRED_RES_BLKS: usize = 6;
const BIT_STREAM_LEN: usize = 3 + 3 * 12 + 3 * 11 + 3; // 75
const B_NUM: usize = NUM_HARMS_MAX - 1; // 55

const CNST_0_9254_Q0_16: u32 = 60647;
const CNST_0_33_Q0_16: u32 = 0x5556;
const CNST_ONE_Q8_24: i32 = 0x0100_0000;
const CNST_0_7_Q1_15: i16 = 0x599A;
const CNST_0_4_Q1_15: i16 = 0x3333;
const CNST_0_03_Q1_15: i32 = 0x03D7;
const CNST_0_05_Q1_15: i32 = 0x0666;
const CNST_0_5_Q0_16: i32 = 0x8000;
const CNST_0_9898_Q1_15: i16 = 0x7EB3u16 as i16;
const CNST_0_5_Q2_14: i16 = 0x2000;
const CNST_1_2_Q2_14: i16 = 0x4CCC;

/// The IMBE-decoded model parameters the shared DLL-exact OLA synthesis needs,
/// in IMBE's OWN fixed-point domains (before the f32 [`MbeParams`] conversion):
/// pre-enhancement per-harmonic log-amplitudes (`log_amps`, Q10.22 log2,
/// 1-indexed — harmonic `i` is `log_amps[i+1]`), the fundamental (`fund_freq` =
/// ω₀/π in Q1.31), per-harmonic voicing (`v_uv_dsn`, 0/1), and `num_harms` (L).
/// Populated by [`decode_params`], read back via [`ImbeState::ola_params`].
#[derive(Clone, Debug)]
pub struct ImbeOlaParams {
    pub num_harms: i16,
    pub fund_freq: i32,
    pub v_uv_dsn: [i16; NUM_HARMS_MAX],
    pub log_amps: [i32; NUM_HARMS_MAX + 2],
}

/// Cross-frame IMBE spectral-amplitude decoder state (the §7 predictor memory).
#[derive(Clone, Debug)]
pub struct ImbeState {
    num_harms_prev1: i16,
    /// Previous log-amplitudes, Q10.22, 1-indexed (`[0]` unused).
    sa_prev1: [i32; NUM_HARMS_MAX + 2],
    last_params: Option<MbeParams>,
    /// Smoothed error rate, TIA-102.BABA §7.6: e_R(0) = 0.95*e_R(-1) + 0.000365*e_T.
    epsilon_r: f64,
    /// Count of consecutive invalid (repeated) frames.
    consecutive_invalid: u32,
    /// Raw Golay error counts (e_0, e_1) from the two highest-priority words of
    /// the most recent frame, exactly as the gate consumed them (an
    /// uncorrectable word carries the [`u8::MAX`] sentinel). Reported to the
    /// consumer alongside the concealment disposition.
    last_errors: (u8, u8),
    /// The current (or, on repeat, most recent) frame's OLA-domain params — the
    /// pre-enhancement log-amps + fundamental + voicing the shared OLA synthesis
    /// consumes. Set by [`decode_params`]; retained across a repeat frame.
    cur: Option<ImbeOlaParams>,
}

/// Snapshot of the predictor memory (`sa_prev1`, `num_harms_prev1`), used by the
/// encoder quantizer to predict from the exact pre-frame state.
pub(crate) type ImbeSnapshot = ([i32; NUM_HARMS_MAX + 2], i16);

impl ImbeState {
    pub fn new() -> Self {
        // sa_decode_init(): num_harms_prev1 = 30, sa_prev1[] = 0.
        Self {
            num_harms_prev1: 30,
            sa_prev1: [0; NUM_HARMS_MAX + 2],
            last_params: None,
            epsilon_r: 0.0,
            consecutive_invalid: 0,
            last_errors: (0, 0),
            cur: None,
        }
    }

    /// Capture the pre-frame predictor memory.
    pub fn snapshot(&self) -> ImbeSnapshot {
        (self.sa_prev1, self.num_harms_prev1)
    }

    /// The current frame's OLA-domain params (see [`ImbeOlaParams`]); set by
    /// [`decode_params`]. `None` before the first valid frame.
    pub fn ola_params(&self) -> Option<&ImbeOlaParams> {
        self.cur.as_ref()
    }

    /// Smoothed error rate e_R(n) (TIA-102.BABA §7.6).
    pub fn epsilon_r(&self) -> f64 {
        self.epsilon_r
    }

    /// Consecutive invalid (repeated) frame count (§7.7 mute gate input).
    pub fn consecutive_invalid(&self) -> u32 {
        self.consecutive_invalid
    }

    /// (e_0, e_t) for the most recent frame: errors corrected in the highest-
    /// priority Golay word, and their sum across the top two words. An
    /// uncorrectable c0 reports as 4, matching the half-rate `epsilon_0`
    /// convention so a consumer can treat both codecs' counts alike.
    pub fn last_errors(&self) -> (u8, u8) {
        let (e0_raw, e1) = self.last_errors;
        let e0 = if e0_raw == u8::MAX { 4 } else { e0_raw };
        (e0, e0.saturating_add(e1))
    }

    /// True when the most recent frame was concealed rather than decoded from
    /// its own bits — the error-rate gate fired, or the pitch index was outside
    /// the valid range. Reset by the next frame that decodes cleanly.
    pub fn last_was_repeat(&self) -> bool {
        self.consecutive_invalid > 0
    }
}

impl Default for ImbeState {
    fn default() -> Self {
        Self::new()
    }
}

/// Working parameters for one frame (mirror of the reference `IMBE_PARAM`).
struct ImbeParam {
    b_vec: [i16; NUM_HARMS_MAX + 3],
    num_harms: i16,
    num_bands: i16,
    fund_freq: i32,
    v_uv_dsn: [i16; NUM_HARMS_MAX],
    bit_alloc: [i16; B_NUM + 4],
    sa: [i16; NUM_HARMS_MAX],
}

impl ImbeParam {
    fn zeroed() -> Self {
        Self {
            b_vec: [0; NUM_HARMS_MAX + 3],
            num_harms: 0,
            num_bands: 0,
            fund_freq: 0,
            v_uv_dsn: [0; NUM_HARMS_MAX],
            bit_alloc: [0; B_NUM + 4],
            sa: [0; NUM_HARMS_MAX],
        }
    }
}

// ---- bit allocation (aux_sub.cc) -------------------------------------------

fn get_bit_allocation_arr(num_harms: i16) -> usize {
    if num_harms == NUM_HARMS_MIN {
        0
    } else {
        let index = (num_harms - NUM_HARMS_MIN - 1) as usize;
        BIT_ALLOCATION_OFFSET_TBL[index >> 2] as usize + (3 + (index >> 2)) * (index & 0x3)
    }
}

fn get_bit_allocation(num_harms: i16, ptr: &mut [i16]) {
    let mut bat = get_bit_allocation_arr(num_harms);
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

// ---- quantizer (qnt_sub.cc) ------------------------------------------------

fn deqnt_by_step(qval: i16, step_size: u16, bit_num: i16) -> i32 {
    if bit_num == 0 {
        return 0;
    }
    let mut res = (step_size as i32) * (qval as i32 - (1 << (bit_num - 1)));
    res = l_add(res, ((step_size as i32) * CNST_0_5_Q0_16) >> 16);
    res
}

// ---- channel decode (ch_decode.cc) -----------------------------------------

/// Returns false if b0 is out of range (frame repeat).
fn decode_frame_vector(fv: &[i16; 8], p: &mut ImbeParam) -> bool {
    p.b_vec[0] = (shr(fv[0], 4) & 0xFC) | (shr(fv[7], 1) & 0x3);
    if p.b_vec[0] < 0 || p.b_vec[0] > 207 {
        return false;
    }

    let mut tmp = ((p.b_vec[0] & 0xFF) << 1) + 0x4F; // 2*(b0+39.5), Q15.1
    let shift = norm_s(tmp);
    let tmp1 = shl(tmp, shift);

    let mut tmp2 = div_s(0x4000, tmp1);
    p.fund_freq = l_shr(l_deposit_h(tmp2), 11 - shift);

    let l_tmp = l_sub(0x4000_0000, l_mult(tmp1, tmp2));
    tmp2 = div_s(extract_l(l_shr(l_tmp, 2)), tmp1);
    let l_tmp = l_shr(l_deposit_l(tmp2), 11 - shift - 2);
    p.fund_freq = l_add(p.fund_freq, l_tmp);

    tmp = (tmp + 0x2) >> 3; // (b0 + 40.5)/4
    p.num_harms = ((CNST_0_9254_Q0_16 * tmp as u32) >> 16) as i16;

    p.num_bands = if p.num_harms <= 36 {
        extract_h(((p.num_harms as u32 + 2) * CNST_0_33_Q0_16) as i32)
    } else {
        NUM_BANDS_MAX
    };

    // Build the priority bit stream.
    let mut bs = [0i16; BIT_STREAM_LEN];
    bs[0] = if fv[0] & 0x4 != 0 { 1 } else { 0 };
    bs[1] = if fv[0] & 0x2 != 0 { 1 } else { 0 };
    bs[2] = if fv[0] & 0x1 != 0 { 1 } else { 0 };
    bs[BIT_STREAM_LEN - 3] = if fv[7] & 0x40 != 0 { 1 } else { 0 };
    bs[BIT_STREAM_LEN - 2] = if fv[7] & 0x20 != 0 { 1 } else { 0 };
    bs[BIT_STREAM_LEN - 1] = if fv[7] & 0x10 != 0 { 1 } else { 0 };

    let mut index0: i32 = 3 + 3 * 12 - 1;
    for vec_num in (1..=3).rev() {
        let mut t = fv[vec_num];
        for _ in 0..12 {
            bs[index0 as usize] = t & 0x1;
            t >>= 1;
            index0 -= 1;
        }
    }
    index0 = 3 + 3 * 12 + 3 * 11 - 1;
    for vec_num in (4..=6).rev() {
        let mut t = fv[vec_num];
        for _ in 0..11 {
            bs[index0 as usize] = t & 0x1;
            t >>= 1;
            index0 -= 1;
        }
    }

    // Rebuild b1 (V/UV) — num_bands bits.
    let mut idx = 3 + 3 * 12; // 39
    let mut t = 0i16;
    for _ in 0..p.num_bands {
        t = (t << 1) | bs[idx];
        idx += 1;
    }
    p.b_vec[1] = t;

    // Rebuild b2 (gain) — 2 stream bits + scattered bits.
    let mut t = 0i16;
    t |= bs[idx] << 1;
    idx += 1;
    t |= bs[idx];
    idx += 1;
    p.b_vec[2] = (fv[0] & 0x38) | (t << 1) | (shr(fv[7], 3) & 0x01);

    // Compact away the consumed b1/b2 bits.
    let shiftn = (p.num_bands + 2) as usize;
    let mut j = idx;
    while j < BIT_STREAM_LEN {
        bs[j - shiftn] = bs[j];
        j += 1;
    }

    // Priority rescanning of the spectral-amplitude residual coefficients.
    for i in 0..B_NUM {
        p.bit_alloc[i] = 0;
        p.b_vec[3 + i] = 0;
    }
    get_bit_allocation(p.num_harms, &mut p.bit_alloc);

    let mut k = 0usize;
    let mut bit_thr = if p.num_harms == 0xb {
        9
    } else {
        p.bit_alloc[0]
    };
    let limit = BIT_STREAM_LEN - p.num_bands as usize - 2;
    while k < limit {
        for i in 0..(p.num_harms as usize - 1) {
            if bit_thr != 0 && bit_thr <= p.bit_alloc[i] {
                p.b_vec[3 + i] = (p.b_vec[3 + i] << 1) | bs[k];
                k += 1;
            }
        }
        bit_thr -= 1;
    }

    // Synchronization bit.
    p.b_vec[p.num_harms as usize + 2] = fv[7] & 1;
    true
}

fn v_uv_decode(p: &mut ImbeParam) {
    let mut num_harms = p.num_harms;
    let mut num_bands = p.num_bands;
    let vu_vec = p.b_vec[1];
    let mut mask = 1i16 << (num_bands - 1);
    let mut i = 0i16;
    let mut h = 0usize;
    while num_harms > 0 {
        if vu_vec & mask != 0 {
            p.v_uv_dsn[h] = 1;
        } else {
            p.v_uv_dsn[h] = 0;
        }
        h += 1;
        num_harms -= 1;
        i += 1;
        if i == 3 {
            if num_bands > 1 {
                num_bands -= 1;
                mask >>= 1;
            }
            i = 0;
        }
    }
}

fn sa_decode(p: &mut ImbeParam, st: &mut ImbeState) {
    let num_harms = p.num_harms as usize;
    let index = (p.num_harms - NUM_HARMS_MIN) as usize;

    let mut gain_vec = [0i16; 6];
    let mut gain_r = [0i16; 6];
    let mut ba_i = 0usize; // bit_alloc cursor
    let mut bv_i = 2usize; // b_vec cursor (starts at &b_vec[2])

    // Gain vector (signed Q5.11).
    let gss = &GAIN_STEP_SIZE_TBL[index * 5..];
    gain_vec[0] = GAIN_QNT_TBL[p.b_vec[bv_i] as usize];
    bv_i += 1;
    for i in 1..6 {
        gain_vec[i] = extract_l(l_shr(
            deqnt_by_step(p.b_vec[bv_i], gss[i - 1], p.bit_alloc[ba_i]),
            5,
        ));
        bv_i += 1;
        ba_i += 1;
    }
    idct(
        &gain_vec,
        NUM_PRED_RES_BLKS as i16,
        NUM_PRED_RES_BLKS as i16,
        &mut gain_r,
    );

    let mut lmprbl_item = LMPRBL_TBL[index];
    let mut t_vec = [0i16; NUM_HARMS_MAX];

    // Higher-order DCT coefficients, per prediction-residual block.
    let mut t_off = 0usize;
    for i in 0..NUM_PRED_RES_BLKS {
        let bl_len = ((lmprbl_item >> 28) & 0xF) as usize;
        lmprbl_item <<= 4;
        let mut c_vec = [0i16; 10];
        c_vec[0] = gain_r[i];
        for j in 1..bl_len {
            let num_bits = p.bit_alloc[ba_i];
            ba_i += 1;
            if num_bits != 0 {
                let step_size = extract_h(
                    (HI_ORD_STD_TBL[j - 1] as i32
                        * HI_ORD_STEP_SIZE_TBL[(num_bits - 1) as usize] as i32)
                        << 1,
                );
                c_vec[j] = extract_l(l_shr(
                    deqnt_by_step(p.b_vec[bv_i], step_size as u16, num_bits),
                    5,
                ));
            } else {
                c_vec[j] = 0;
            }
            bv_i += 1;
        }
        idct(&c_vec, bl_len as i16, bl_len as i16, &mut t_vec[t_off..]);
        t_off += bl_len;
    }

    // k_coef = num_harms_prev / num_harms, unsigned Q8.24.
    let nh = p.num_harms;
    let nhp = st.num_harms_prev1;
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

    // Replicate previous-frame tail.
    for i in (nhp as usize + 1)..(NUM_HARMS_MAX + 2) {
        st.sa_prev1[i] = st.sa_prev1[nhp as usize];
    }

    let mut k_acc: u32 = k_coef;
    let mut sum: i32 = 0;
    let mut sa_tmp = [0i32; NUM_HARMS_MAX];
    for i in 0..num_harms {
        let index = (k_acc >> 24) as usize; // integer part
        let si_coef = ((k_acc - ((index as u32) << 24)) >> 9) as i16; // fractional, Q15
        if si_coef == 0 {
            let tw = l_mpy_ls(st.sa_prev1[index], ro_coef);
            sa_tmp[i] = l_add(l_shr(l_deposit_h(t_vec[i]), 5), tw);
            sum = l_add(sum, st.sa_prev1[index]);
        } else {
            let tw = l_mpy_ls(st.sa_prev1[index], sub(0x7FFF, si_coef));
            sum = l_add(sum, tw);
            sa_tmp[i] = l_add(l_shr(l_deposit_h(t_vec[i]), 5), l_mpy_ls(tw, ro_coef));
            let tw2 = l_mpy_ls(st.sa_prev1[index + 1], si_coef);
            sum = l_add(sum, tw2);
            sa_tmp[i] = l_add(sa_tmp[i], l_mpy_ls(tw2, ro_coef));
        }
        k_acc = k_acc.wrapping_add(k_coef);
    }

    let sh = norm_s(p.num_harms);
    let tmp1 = div_s(0x4000, shl(p.num_harms, sh));
    sum = l_shr(l_mpy_ls(l_mpy_ls(sum, ro_coef), tmp1), 14 - sh);

    for i in 1..=num_harms {
        st.sa_prev1[i] = l_sub(sa_tmp[i - 1], sum);
        p.sa[i - 1] = pow2(st.sa_prev1[i]);
    }
    st.num_harms_prev1 = p.num_harms;
}

fn l_v_magsq(v: &[i16]) -> i32 {
    let mut acc = 0i32;
    for &x in v {
        acc = l_mac(acc, x, x);
    }
    acc
}

fn sa_enh(p: &mut ImbeParam) {
    let num_harm = p.num_harms as usize;
    let mut sa_tmp = [0i16; NUM_HARMS_MAX];
    sa_tmp[..num_harm].copy_from_slice(&p.sa[..num_harm]);

    let mut rm0 = l_v_magsq(&p.sa[..num_harm]);
    if rm0 == 0 {
        return;
    }
    let mut nm = norm_l(rm0);
    if rm0 == MAX_32 {
        nm = 1;
        for i in 0..num_harm {
            sa_tmp[i] = shr(p.sa[i], nm);
        }
        rm0 = l_v_magsq(&sa_tmp[..num_harm]);
    } else if nm > 2 {
        nm = -(nm >> 1);
        for i in 0..num_harm {
            sa_tmp[i] = shr(p.sa[i], nm);
        }
        rm0 = l_v_magsq(&sa_tmp[..num_harm]);
    }

    let w0 = p.fund_freq;
    let mut cos_w = [0i16; NUM_HARMS_MAX];
    let mut cos_acc = 0i32;
    let mut rm1 = 0i32;
    for i in 0..num_harm {
        cos_acc = l_add(cos_acc, w0);
        cos_w[i] = cos_fxp(extract_h(cos_acc));
        rm1 = l_add(rm1, l_mpy_ls(l_mult(sa_tmp[i], sa_tmp[i]), cos_w[i]));
    }

    let rm0_s = extract_h(rm0);
    let rm1_s = extract_h(rm1);
    let l_rm0_2 = l_mult(rm0_s, rm0_s);
    let l_rm1_2 = l_mult(rm1_s, rm1_s);
    let mut l_den = l_sub(l_rm0_2, l_rm1_2);
    l_den = l_mult(extract_h(l_den), rm0_s);
    let mut nm1 = norm_l(l_den);
    l_den = l_shl(l_den, nm1);
    let nm2 = norm_l(w0);
    l_den = l_mpy_ls(l_den, extract_h(l_shl(w0, nm2)));
    nm1 += nm2;
    if l_den < 1 {
        return;
    }

    let l_sum_rm02_rm12 = l_add(l_shr(l_rm0_2, 2), l_shr(l_rm1_2, 2));
    let rm0rm1 = shr(mult_r(rm0_s, rm1_s), 1);

    for i in 0..num_harm {
        if (((i + 1) << 3) > num_harm) && sa_tmp[i] != 0 {
            let mut l_num = l_sub(l_sum_rm02_rm12, l_mult(rm0rm1, cos_w[i]));
            let mut tot_nm = norm_l(l_num);
            l_num = l_shl(l_num, tot_nm);
            while l_num >= l_den {
                l_num = l_shr(l_num, 1);
                tot_nm -= 1;
            }
            let mut tmp = div_s(extract_h(l_num), extract_h(l_den));
            tot_nm -= nm1;

            let mut l_tmp = l_mult(sa_tmp[i], sa_tmp[i]);
            let nm2b = norm_l(l_tmp);
            l_tmp = l_shl(l_tmp, nm2b);
            l_tmp = l_mult(extract_h(l_tmp), tmp);
            tot_nm += nm2b;
            tot_nm -= 2;

            if tot_nm <= 0 {
                l_tmp = l_shr(l_tmp, add(8, tot_nm));
                l_tmp = sqrt_l_exp(l_tmp, &mut tot_nm);
                l_tmp = l_shr(l_tmp, tot_nm);
                l_tmp = sqrt_l_exp(l_tmp, &mut tot_nm);
                l_tmp = l_shr(l_tmp, tot_nm);
                l_tmp = l_mult(extract_h(l_tmp), CNST_0_9898_Q1_15);
                tmp = extract_h(l_shl(l_tmp, 1));
            } else if tot_nm <= 8 {
                l_tmp = l_shr(l_tmp, tot_nm);
                l_tmp = sqrt_l_exp(l_tmp, &mut tot_nm);
                l_tmp = l_shr(l_tmp, tot_nm);
                l_tmp = sqrt_l_exp(l_tmp, &mut tot_nm);
                l_tmp = l_shr(l_tmp, tot_nm + 1);
                tmp = mult(extract_h(l_tmp), CNST_0_9898_Q1_15);
            } else {
                let mut nm1b = tot_nm & 0xfffeu16 as i16;
                l_tmp = l_shr(l_tmp, tot_nm - nm1b);
                l_tmp = sqrt_l_exp(l_tmp, &mut tot_nm);
                l_tmp = l_shr(l_tmp, tot_nm);
                tot_nm = nm1b >> 1;
                nm1b = tot_nm & 0xfffeu16 as i16;
                l_tmp = l_shr(l_tmp, tot_nm - nm1b);
                l_tmp = sqrt_l_exp(l_tmp, &mut tot_nm);
                l_tmp = l_shr(l_tmp, tot_nm);
                l_tmp = l_mult(extract_h(l_tmp), CNST_0_9898_Q1_15);
                tot_nm = nm1b >> 1;
                tmp = extract_h(l_shr(l_tmp, tot_nm + 1));
            }

            if tmp > CNST_1_2_Q2_14 {
                p.sa[i] = extract_h(l_shl(l_mult(p.sa[i], CNST_1_2_Q2_14), 1));
            } else if tmp < CNST_0_5_Q2_14 {
                p.sa[i] = shr(p.sa[i], 1);
            } else {
                p.sa[i] = extract_h(l_shl(l_mult(p.sa[i], tmp), 1));
            }
        }
    }

    // Re-normalise the modified amplitude energy back toward Rm0.
    for i in 0..num_harm {
        sa_tmp[i] = shr(p.sa[i], nm);
    }
    let l_sum_mod = l_v_magsq(&sa_tmp[..num_harm]);
    if l_sum_mod > rm0 {
        let tmp = div_s(extract_h(rm0), extract_h(l_sum_mod));
        let mut tot_nm = 0i16;
        let l_tmp = sqrt_l_exp(l_deposit_h(tmp), &mut tot_nm);
        let tmp = shr(extract_h(l_tmp), tot_nm);
        for i in 0..num_harm {
            p.sa[i] = mult_r(p.sa[i], tmp);
        }
    }
}

/// Full IMBE dequant: deframed info word → [`MbeParams`] for synthesis.
/// On an out-of-range pitch index, OR when the TIA-102.BABA §7.6-7.7
/// error-rate gate fires (e_0>2 and e_1>=10+40*e_R on the two highest-
/// priority Golay words), the previous frame's parameters are repeated
/// (matching the reference's frame-repeat behaviour); `None` only before
/// the first valid frame.
pub fn decode_params(fr: &ImbeFrame, st: &mut ImbeState) -> Option<MbeParams> {
    // e_0/e_1 error-smoothing (companion of the half-rate BABA-1 addendum;
    // c0/c1 are the two highest-priority [23,12] Golay words).
    let e0 = fr.errors[0];
    let e1 = fr.errors[1];
    let e_t = e0.saturating_add(e1);
    st.last_errors = (e0, e1);
    st.epsilon_r = 0.95 * st.epsilon_r + 0.000365 * f64::from(e_t);
    let do_repeat = e0 > 2 && f64::from(e1) >= 10.0 + 40.0 * st.epsilon_r;

    let fv: [i16; 8] = core::array::from_fn(|i| fr.u[i] as i16);
    let mut p = ImbeParam::zeroed();

    let valid_vector = decode_frame_vector(&fv, &mut p);

    if do_repeat || !valid_vector {
        st.consecutive_invalid = st.consecutive_invalid.saturating_add(1);
        return st.last_params.clone();
    }
    st.consecutive_invalid = 0;

    v_uv_decode(&mut p);
    sa_decode(&mut p, st);
    // Capture the OLA-domain params BEFORE sa_enh: st.sa_prev1 now holds this
    // frame's PRE-enhancement log-amps (sa_enh only touches p.sa, the linear
    // amplitudes for the f32 path). The shared OLA amp chain re-does the
    // enhancement itself, so it wants these pre-enhancement log-amps.
    st.cur = Some(ImbeOlaParams {
        num_harms: p.num_harms,
        fund_freq: p.fund_freq,
        v_uv_dsn: p.v_uv_dsn,
        log_amps: st.sa_prev1,
    });
    sa_enh(&mut p);

    let l = p.num_harms as usize;
    // ω₀ (rad/sample) = π · fund_freq / 2^31 (fund_freq is ω₀/π in Q1.31).
    let omega_0 = (PI * (p.fund_freq as f64) / 2_147_483_648.0) as f32;
    let voiced: Vec<bool> = (0..l).map(|i| p.v_uv_dsn[i] != 0).collect();
    // sa is linear amplitude in Q14.2 → divide by 4.
    let amplitudes: Vec<f32> = (0..l).map(|i| p.sa[i] as f32 / 4.0).collect();

    let params = MbeParams {
        b0: p.b_vec[0] as u8,
        omega_0,
        l: p.num_harms as u8,
        voiced,
        amplitudes,
    };
    st.last_params = Some(params.clone());
    Some(params)
}
