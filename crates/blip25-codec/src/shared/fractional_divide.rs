//! `FUN_1030a600` — the reference's saturating fractional divide.
//!
//! `|p1| / |s16(p2)|`, halved, sign-restored, with `|num| < |den|` collapsing to
//! zero rather than rounding. Shared between the encoder's noise tracker
//! (`enc::noise_track`) and the IMBE pitch quantiser
//! ([`crate::imbe::quantize::imbe_b0_from_pitch_word`]), which is compiled in
//! decode-only builds and so cannot reach into `enc/`.

/// The divide. `p2` is taken as a 16-bit signed word; `p1` keeps full width.
pub fn a600(p1: i32, p2: i32) -> i32 {
    let den0 = i32::from(p2 as i16);
    let sgn = den0 ^ p1;
    let num = if p1 < 0 {
        if p1 == i32::MIN {
            i32::MAX
        } else {
            -p1
        }
    } else {
        p1
    };
    let den = if den0 < 0 {
        if den0 == -0x8000 {
            0x7fff
        } else {
            -den0
        }
    } else {
        den0
    };
    if den == 0 {
        return 0;
    }
    let mut r = if num < den { 0 } else { (num / den) >> 1 };
    if sgn < 0 {
        r = -r;
    }
    r
}
