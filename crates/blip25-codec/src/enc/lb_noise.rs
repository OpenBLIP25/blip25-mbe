//! The low-band noise rewrite of the band-voicing row: `FUN_1030eaa0`'s tail,
//! `FUN_1030dc80` and `FUN_1030f630`.
//!
//! After `FUN_1031c960` writes the fresh band-voicing ratio codes into p0 slot 0
//! (and `FUN_1031c820` refines them), the reference rewrites the three LOWEST
//! bands' codes from the `ctx+0x938` noise tracker before the a4 measurement
//! reads the ring. The row it rewrites is `S+0x636`, i.e. p0 slot 0 words 0..2.
//! The mask stored at `ctx+0x91a` is taken BEFORE the rewrite, so nothing here
//! moves the band-voicing mask — only `a4`, and through it `b1`.
//!
//! Two arms, selected by the pitch fit-error ring and a two-frame voiced-band
//! count:
//!
//! ```text
//!   cnt = popcount8(mask[f-2]) + popcount8(mask[f])       (FUN_1031ecc0)
//!   if (nt[0x44] & 0x1c) == 0 { cnt = 0 }                 (hdr30 bits 2,3,4)
//!   if (Q[0] < 0x2001 || Q[2] < 0x2001) && cnt >= 4 {
//!       quiet arm: one gated rescale of P[0] by P[1]
//!   } else {
//!       FUN_1030dc80(P, nt+0x122, nt+0x52, nt+0x72, cnt)
//!       plus a P[0] = 0x1c29 floor under four further conditions
//!   }
//! ```
//!
//! `FUN_1030dc80` is three `FUN_1030f630` calls, one per band, differing only in
//! their three constants and in which noise-tracker words they read.
//! `FUN_1030f630` denormalises the band's noise floor `(mantissa, exponent)`,
//! declines unless it exceeds the band's threshold, forms a headroom
//! `C1 - hist2 - (cnt << 8)` clamped at `0x200`, and applies `2^(headroom * C3)`
//! as a Q12 gain to the band's ratio code.
//!
//! Validated against live capture (`0x1030eba8` / `0x1030ebee` / `0x1030ed22`
//! hooks on the Wave7k DLL), fed the reference's own pre-state: the arm choice
//! is right on 1800/1800 frames and the rewritten row on 1036/1036 `dc80` frames
//! and 6/6 quiet-arm stores across mark / noisy / clean / dam.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use super::dp_score::{a4f0, norm_shift, s16, shift_sat, shr_dsp, ST_WORDS};
use super::noise_track::a890;

/// Opt-in switch (`BLIP25_LB_NOISE=1`, OFF by default). Read once.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BLIP25_LB_NOISE").as_deref() == Ok("1"))
}

/// Word index into the noise-tracker window for a block BYTE offset.
#[inline]
fn w(st: &[i16; ST_WORDS], off: usize) -> i32 {
    st[(off - 0x40) / 2] as i32
}

/// `FUN_1030f630`: one band's rewrite. Returns the band's new code.
fn f630(
    p: i16,
    hist2: i32,
    nf_mant: i32,
    nf_exp: i32,
    c1: i32,
    c2: i32,
    c3: i32,
    arg8: i32,
) -> i16 {
    let lvl = s16(shr_dsp(s16(nf_mant).wrapping_mul(0x10000), s16(nf_exp - 5)) >> 16);
    if s16(c2) >= lvl {
        return p;
    }
    let head = s16(c1)
        .wrapping_mul(0x10000)
        .wrapping_sub(s16(hist2).wrapping_mul(0x10000))
        .wrapping_sub(s16(arg8).wrapping_mul(0x10000));
    if head <= 0 {
        return p;
    }
    let d = (head >> 16).min(0x200);
    let prod = d.wrapping_mul(s16(c3)).wrapping_mul(2);
    let n = norm_shift(prod);
    let mant = (((prod as u32) << (n & 0x1f)) as i32) >> 16;
    let g = s16(a890(mant, 8 - n, 3));
    s16(shift_sat(g.wrapping_mul(p as i32).wrapping_mul(2), 3) >> 16) as i16
}

/// `FUN_1030dc80`: the three low bands, in place.
fn dc80(p: &mut [i16], st: &[i16; ST_WORDS], cnt: i32) {
    let arg8 = s16(((s16(cnt) as u32).wrapping_shl(24) as i32) >> 16);
    const K: [(i32, i32, i32); 3] = [
        (0x480, -0x4800, 0x42b8),
        (0x400, -0x5000, 0x32b8),
        (0x380, -0x5400, 0x22b8),
    ];
    for (b, &(c1, c2, c3)) in K.iter().enumerate() {
        p[b] = f630(
            p[b],
            w(st, 0x122 + 2 * b),
            w(st, 0x52 + 4 * b),
            w(st, 0x72 + 4 * b),
            c1,
            c2,
            c3,
            arg8,
        );
    }
}

/// `FUN_1031ecc0(x, 8)`: the population count of the low eight bits.
#[inline]
fn popcount8(x: u16) -> i32 {
    (x as u8).count_ones() as i32
}

/// The `FUN_1030eaa0` tail. `p` is p0 slot 0 (at least three words), `st` the
/// noise-tracker window, `q0`/`q2` the pitch fit-error ring words `S+0x62c` and
/// `S+0x630`, and the two masks this frame's and the frame-before-last's
/// `ctx+0x91a`.
pub(crate) fn apply(
    p: &mut [i16],
    st: &[i16; ST_WORDS],
    q0: i32,
    q2: i32,
    mask_f: u16,
    mask_f2: u16,
) {
    let cnt = if w(st, 0x44) & 0x1c == 0 {
        0
    } else {
        s16(popcount8(mask_f2) + popcount8(mask_f))
    };
    if (s16(q0) < 0x2001 || s16(q2) < 0x2001) && cnt >= 4 {
        let lvl = ((s16(w(st, 0x48)).wrapping_mul(0x550a)).wrapping_sub(0x0b00_0000)) >> 16;
        if !a4f0(lvl, 7, w(st, 0x52), w(st, 0x72)) {
            return;
        }
        if !a4f0(w(st, 0x56), w(st, 0x76), w(st, 0x52), w(st, 0x72)) {
            return;
        }
        let w1 = p[1] as i32;
        if w1 >= 0x199a {
            return;
        }
        let t = s16((p[0] as i32).wrapping_mul(w1).wrapping_mul(2) >> 16);
        p[0] = s16(shift_sat(t.wrapping_mul(0xa000), 3) >> 16) as i16;
        return;
    }
    dc80(p, st, cnt);
    let lvl = ((s16(w(st, 0x48)).wrapping_mul(0x550a)).wrapping_sub(0x0d00_0000)) >> 16;
    if !a4f0(lvl, 7, w(st, 0x52), w(st, 0x72)) {
        return;
    }
    if w(st, 0xfa) >= 0x1c00 {
        return;
    }
    if !a4f0(w(st, 0x56), w(st, 0x76), w(st, 0x52), w(st, 0x72)) {
        return;
    }
    if mask_f & 0x7f != 0 || mask_f2 & 0x7f != 0 {
        return;
    }
    if w(st, 0x48) - w(st, 0x13a) <= 0x1600 {
        return;
    }
    if p[0] >= 0x1c29 {
        return;
    }
    p[0] = 0x1c29;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live-capture rows: `(st words 0x44/0x48/0x52/0x56/0x5a/0x72/0x76/0x7a/
    /// 0xfa/0x122/0x124/0x126/0x13a, q0, q2, mask_f, mask_f2, P_in, P_out)`,
    /// read out of the Wave7k DLL at `0x1030eba8` / `0x1030ebee` / `0x1030ed22`.
    /// The first four rows take the `dc80` arm (`noisy`), the last two the quiet
    /// arm (`mark`).
    const ROWS: &[([i32; 13], i32, i32, u16, u16, [i16; 3], [i16; 3])] = &[
        (
            [0, -7683, -23646, -19227, -18306, 5, 5, 5, 0, 0, 0, 0, -8912],
            19418,
            32767,
            0,
            0,
            [21410, 19033, 19397],
            [21410, 32767, 32767],
        ),
        (
            [
                0, -7687, -23256, -18401, -17897, 5, 5, 5, -11121, 461, 428, 369, -9458,
            ],
            17999,
            19418,
            0,
            0,
            [14017, 20119, 14715],
            [14017, 32767, 31215],
        ),
        (
            [
                0, -7691, -23174, -17985, -17741, 5, 5, 5, -12565, 177, 259, 392, -9749,
            ],
            14525,
            17999,
            0,
            0,
            [18353, 14187, 14156],
            [18353, 32767, 29677],
        ),
        (
            [
                0, -7695, -23147, -17965, -17798, 5, 5, 5, -11899, 281, 356, 444, -10001,
            ],
            12357,
            14525,
            0,
            0,
            [19102, 9319, 13432],
            [19102, 27957, 26086],
        ),
        (
            [
                63, -7134, -31725, -17448, -22498, 4, 5, 5, 3456, 2714, 2317, 2145, -8836,
            ],
            701,
            7130,
            255,
            0,
            [709, 478, 423],
            [50, 478, 423],
        ),
        (
            [
                255, -6692, -31725, -17448, -22498, 4, 5, 5, 720, 2232, 1862, 1773, -7913,
            ],
            545,
            701,
            255,
            64,
            [503, 293, 1181],
            [20, 293, 1181],
        ),
    ];

    #[test]
    fn dc80_matches_live_capture() {
        for (i, &(cols, q0, q2, mf, mf2, pin, pout)) in ROWS.iter().enumerate() {
            let mut st = [0i16; ST_WORDS];
            for (k, off) in [
                0x44, 0x48, 0x52, 0x56, 0x5a, 0x72, 0x76, 0x7a, 0xfa, 0x122, 0x124, 0x126, 0x13a,
            ]
            .iter()
            .enumerate()
            {
                st[(off - 0x40) / 2] = cols[k] as i16;
            }
            let mut p = [pin[0], pin[1], pin[2], 0, 0, 0, 0, 0];
            apply(&mut p, &st, q0, q2, mf, mf2);
            assert_eq!(&p[..3], &pout[..], "row {i}");
        }
    }
}
