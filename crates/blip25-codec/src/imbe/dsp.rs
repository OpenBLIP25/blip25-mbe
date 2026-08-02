//! IMBE inverse DCT, per the TIA-102.BABA fixed-point reference. The unsigned
//! wrapping angle accumulators are what the standard's fixed-point arithmetic
//! requires; cross-checked for bit-exactness against OP25's `imbe_vocoder`,
//! which implements the same standard.
#![allow(clippy::all)]

use super::fixp::*;
use super::math::cos_fxp;

const CNST_0_5_Q1_15: u16 = 0x4000;
const CNST_1_0_Q1_15: u16 = 0x7FFF;
const CNST_0_5_Q5_11: i16 = 0x0400;

/// Inverse DCT: `in[0..m_lim]` -> `out[0..i_lim]`. Angle accumulators are
/// unsigned 16-bit and intentionally wrap (matching the reference's `UWord16`).
pub(crate) fn idct(input: &[i16], m_lim: i16, i_lim: i16, out: &mut [i16]) {
    let (angl_intl, angl_intl_2): (u16, u16) = if m_lim == 1 {
        (CNST_0_5_Q1_15, CNST_1_0_Q1_15)
    } else {
        let a = div_s(CNST_0_5_Q5_11, (m_lim << 11) as i16) as u16;
        (a, a.wrapping_shl(1))
    };

    let mut angl_step = angl_intl;
    for i in 0..i_lim as usize {
        let mut sum: i32 = 0;
        let mut angl_acc = angl_step;
        for m in 1..m_lim as usize {
            sum = l_add(sum, l_shr(l_mult(input[m], cos_fxp(angl_acc as i16)), 7));
            angl_acc = angl_acc.wrapping_add(angl_step);
        }
        sum = l_add(sum, l_shr(l_deposit_h(input[0]), 8));
        out[i] = extract_l(l_shr_r(sum, 8));
        angl_step = angl_step.wrapping_add(angl_intl_2);
    }
}
