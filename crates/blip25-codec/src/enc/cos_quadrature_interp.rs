//! The inner correction routine's one callee (site 2 of the P/Q window
//! composition; see [`super::pq_builder`] part 3's correction).
//!
//! ## What it is
//!
//! A **cosine-table phase-to-quadrature interpolator/resampler**. For each of
//! `count` input samples (read from `src` at a 2-word / one-complex stride, so
//! it takes the REAL part of an interleaved complex array), it normalises the
//! sample by a block-floating shift `shift`, treats the normalised value as a
//! phase angle, and emits TWO output words per input: a linearly-interpolated
//! cosine-table lookup at that phase, and a second lookup at phase + a fixed
//! 384/512-turn offset (the quadrature companion). Output is written to `dst`
//! at 2 words per input (`dst[2k]`, `dst[2k+1]`). In the real call site
//! `src == dst + 1 word` (in-place), `shift` is the caller's block-float
//! shift, and `count == 128`.
//!
//! The lookup table is a 512-entry cosine table
//! (`round(32767*cos(2*pi*i/512))`), embedded here as [`COS_TABLE`]. The
//! function's first (in-phase) lookup indexes the table UNMASKED and can reach
//! index 512 (only when the normalised sample equals `i16::MIN`), so a 513th
//! entry (`49`, the real byte that follows the table in `.rdata`) is included
//! to reproduce that edge exactly. The quadrature lookup and both `+1`
//! neighbours are masked `& 0x1ff`.
//!
//! ## Provenance
//!
//! Disassembled as a leaf routine (contains no `call`, so it has no callees of
//! its own) and pinned against capture: at the sole call site the captured
//! "out" arg slot `sp[2] = 0x07ffffff` is exactly this port's mask
//! `(1<<(31-shift))-1` for `shift=4`, and `sp[3]` decrements to 0 (the loop
//! counter).

/// 512-entry cosine table `round(32767*cos(2*pi*i/512))`, plus one trailing
/// `.rdata` word (`49`) at index 512 that the unmasked in-phase lookup can
/// touch. The 513th entry is not padding -- removing it changes the output when
/// the normalised sample equals `i16::MIN`.
pub(crate) static COS_TABLE: [i16; 513] = [
    32767, 32766, 32758, 32746, 32729, 32706, 32679, 32647, 32610, 32568, 32522, 32470, 32413,
    32352, 32286, 32214, 32138, 32058, 31972, 31881, 31786, 31686, 31581, 31471, 31357, 31238,
    31114, 30986, 30853, 30715, 30572, 30425, 30274, 30118, 29957, 29792, 29622, 29448, 29269,
    29086, 28899, 28707, 28511, 28311, 28106, 27897, 27684, 27467, 27246, 27020, 26791, 26557,
    26320, 26078, 25833, 25583, 25330, 25073, 24812, 24548, 24279, 24008, 23732, 23453, 23170,
    22884, 22595, 22302, 22006, 21706, 21403, 21097, 20788, 20475, 20160, 19841, 19520, 19195,
    18868, 18538, 18205, 17869, 17531, 17190, 16846, 16500, 16151, 15800, 15447, 15091, 14733,
    14373, 14010, 13646, 13279, 12910, 12540, 12167, 11793, 11417, 11039, 10660, 10279, 9896, 9512,
    9127, 8740, 8351, 7962, 7571, 7180, 6787, 6393, 5998, 5602, 5205, 4808, 4410, 4011, 3612, 3212,
    2811, 2411, 2009, 1608, 1206, 804, 402, 0, -402, -804, -1206, -1608, -2009, -2411, -2811,
    -3212, -3612, -4011, -4410, -4808, -5205, -5602, -5998, -6393, -6787, -7180, -7571, -7962,
    -8351, -8740, -9127, -9512, -9896, -10279, -10660, -11039, -11417, -11793, -12167, -12540,
    -12910, -13279, -13646, -14010, -14373, -14733, -15091, -15447, -15800, -16151, -16500, -16846,
    -17190, -17531, -17869, -18205, -18538, -18868, -19195, -19520, -19841, -20160, -20475, -20788,
    -21097, -21403, -21706, -22006, -22302, -22595, -22884, -23170, -23453, -23732, -24008, -24279,
    -24548, -24812, -25073, -25330, -25583, -25833, -26078, -26320, -26557, -26791, -27020, -27246,
    -27467, -27684, -27897, -28106, -28311, -28511, -28707, -28899, -29086, -29269, -29448, -29622,
    -29792, -29957, -30118, -30274, -30425, -30572, -30715, -30853, -30986, -31114, -31238, -31357,
    -31471, -31581, -31686, -31786, -31881, -31972, -32058, -32138, -32214, -32286, -32352, -32413,
    -32470, -32522, -32568, -32610, -32647, -32679, -32706, -32729, -32746, -32758, -32766, -32767,
    -32766, -32758, -32746, -32729, -32706, -32679, -32647, -32610, -32568, -32522, -32470, -32413,
    -32352, -32286, -32214, -32138, -32058, -31972, -31881, -31786, -31686, -31581, -31471, -31357,
    -31238, -31114, -30986, -30853, -30715, -30572, -30425, -30274, -30118, -29957, -29792, -29622,
    -29448, -29269, -29086, -28899, -28707, -28511, -28311, -28106, -27897, -27684, -27467, -27246,
    -27020, -26791, -26557, -26320, -26078, -25833, -25583, -25330, -25073, -24812, -24548, -24279,
    -24008, -23732, -23453, -23170, -22884, -22595, -22302, -22006, -21706, -21403, -21097, -20788,
    -20475, -20160, -19841, -19520, -19195, -18868, -18538, -18205, -17869, -17531, -17190, -16846,
    -16500, -16151, -15800, -15447, -15091, -14733, -14373, -14010, -13646, -13279, -12910, -12540,
    -12167, -11793, -11417, -11039, -10660, -10279, -9896, -9512, -9127, -8740, -8351, -7962,
    -7571, -7180, -6787, -6393, -5998, -5602, -5205, -4808, -4410, -4011, -3612, -3212, -2811,
    -2411, -2009, -1608, -1206, -804, -402, 0, 402, 804, 1206, 1608, 2009, 2411, 2811, 3212, 3612,
    4011, 4410, 4808, 5205, 5602, 5998, 6393, 6787, 7180, 7571, 7962, 8351, 8740, 9127, 9512, 9896,
    10279, 10660, 11039, 11417, 11793, 12167, 12540, 12910, 13279, 13646, 14010, 14373, 14733,
    15091, 15447, 15800, 16151, 16500, 16846, 17190, 17531, 17869, 18205, 18538, 18868, 19195,
    19520, 19841, 20160, 20475, 20788, 21097, 21403, 21706, 22006, 22302, 22595, 22884, 23170,
    23453, 23732, 24008, 24279, 24548, 24812, 25073, 25330, 25583, 25833, 26078, 26320, 26557,
    26791, 27020, 27246, 27467, 27684, 27897, 28106, 28311, 28511, 28707, 28899, 29086, 29269,
    29448, 29622, 29792, 29957, 30118, 30274, 30425, 30572, 30715, 30853, 30986, 31114, 31238,
    31357, 31471, 31581, 31686, 31786, 31881, 31972, 32058, 32138, 32214, 32286, 32352, 32413,
    32470, 32522, 32568, 32610, 32647, 32679, 32706, 32729, 32746, 32758, 32766, 49,
];

use crate::fixops::i32r::sar32;

/// Saturating absolute value: returns `|x|`, but maps `i32::MIN` to `i32::MAX`
/// rather than overflowing.
#[inline]
fn sat_abs(x: i32) -> i32 {
    if x >= 0 {
        x
    } else if x == i32::MIN {
        i32::MAX
    } else {
        x.wrapping_neg()
    }
}

/// One cosine-table lookup with linear interpolation on a saturated-abs phase.
/// `mask_first` selects whether the *direct* (integer part) index is masked
/// `& 0x1ff` (quadrature lookup) or used raw up to 512 (in-phase lookup). The
/// `+1` neighbour is always masked.
#[inline]
fn cos_interp(phase: i32, mask_first: bool) -> i32 {
    let ap = sat_abs(phase);
    let frac = ap & 0xffff; // 0..0xffff
    let idx_full = sar32(ap, 16); // integer part
    let idx = if mask_first {
        idx_full & 0x1ff
    } else {
        idx_full & 0xffff
    } as usize;
    let idx1 = ((idx_full + 1) & 0x1ff) as usize;
    let a = COS_TABLE[idx] as i32;
    let b = COS_TABLE[idx1] as i32;
    let t = a
        .wrapping_mul(0x10000i32.wrapping_sub(frac))
        .wrapping_add(b.wrapping_mul(frac));
    sar32(t, 16)
}

/// The `(dst, src, shift, count)` routine.
///
/// Reads `count` samples from `src` at a 2-word stride (`src[2k]`, taking the
/// real part of an interleaved complex array), block-float-normalises each by
/// `shift`, and writes two interpolated cosine-table words per sample to
/// `dst` (`dst[2k]` = in-phase, `dst[2k+1]` = quadrature). `dst` must hold at
/// least `2*count` words; `src` at least `2*count - 1`. Matches the reference
/// function's in-place use (`src == dst + 1 word`): the read at index `2k+1`
/// (relative to `dst`) always precedes its own overwrite, so a straight
/// forward pass reproduces it bit-exactly.
pub(crate) fn quadrature_interp(dst: &mut [i16], src: &[i16], shift: i16, count: i16) {
    let n = count as i32;
    if n <= 0 {
        return;
    }
    let shift = shift as i32;

    // Setup: A = max(shift,0), B = min(shift,0), M = mask.
    let a_shift = shift.max(0);
    let b_shift = shift.min(0);
    let mask: i32 = if shift > 0 {
        // (1 << (31 - shift)) - 1
        (1i32.wrapping_shl((31 - shift) as u32)).wrapping_sub(1)
    } else {
        0x7fff_ffff
    };

    for k in 0..(n as usize) {
        let v = src[2 * k] as i32; // real part, stride 4 bytes
        let mut e = v << 16;

        // shift by B (min(shift,0)): >=0 no-op; -31<B<0 sar |B|; B<=-31 sign-fill
        e = if b_shift >= 0 {
            e // no shift
        } else if b_shift <= -31 {
            sar32(e, 0x1f)
        } else {
            sar32(e, (-b_shift) as u32)
        };

        e &= mask;

        // shift by A (max(shift,0))
        e = (e as u32).wrapping_shl((a_shift & 0x1f) as u32) as i32;

        e = sar32(e, 16);

        // t = sign_extend16(low16(e)); phase = t << 10 (via <<16 then sar 6)
        let t = (e as i16) as i32;
        let phase = sar32(t << 16, 6); // == t << 10

        let out0 = cos_interp(phase, false) as i16; // in-phase (direct index unmasked)
        let out1 = cos_interp(phase.wrapping_add(0x0180_0000), true) as i16; // quadrature

        dst[2 * k] = out0;
        dst[2 * k + 1] = out1;
    }
}
