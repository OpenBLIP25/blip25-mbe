//! The packer's tone/silence classifier — the branch that replaces the pitch
//! quantiser's output with a frequency index taken straight from the f0 ring.
//!
//! The packer calls the classifier before the pitch quantiser. When it returns
//! non-zero the packer writes the constants `step = 0x1079` / `L = 56` into both
//! parameter blocks, skips the pitch quantiser entirely, and packs the
//! classifier's own index as `b̂₀`. The voicing byte `b̂₁` is packed unchanged.
//!
//! ## The gate
//!
//! The classifier's two voicing words are the halves of the 32-bit voicing word
//! `v = VOICING_CB[b1 * 4]`, each 2-bit code duplicated into two adjacent slots.
//! Because the duplication preserves the code bits, the classifier's
//! "no band carries code 1, some band carries code 2" test over the two halves
//! is exactly [`l56_gate_from_b1`] over `v` — the same predicate the decoder's
//! `(L, step)` derivation already uses. It is reused here verbatim, plus the
//! packer's redundant `mode == 1` guard.
//!
//! ## The index
//!
//! The two quantiser inputs are the f0 ring's two previous entries
//! (`ring[1]`, `ring[2]`): the [`repair_gate_f0`](super::t_ring::repair_gate_f0)
//! value of the current frame's first analysis call and of the previous frame's
//! second call. Whichever half of the voicing word carries the code-2 bands
//! decides which of them is quantised:
//!
//! * only the `A` half → one 7-bit index over `ring[1]`
//! * only the `B` half → one 7-bit index over `ring[2]`
//! * both halves → two base-`R` digits (`R = 11` for 7 bits), combined as
//!   `dB + dA * R` and clamped to `maxidx`
//!
//! The quantiser is a symmetric uniform scalar quantiser over `[-0x4100,
//! +0x4100]` in block-floating-point, shared with the rest of the codec's
//! ladders. Its closed form holds for `range = 128` but NOT for `range = 11`
//! (5 index and 31257 reconstruction mismatches over the i16 domain), so the
//! block-float division is reproduced rather than approximated.

use crate::shared::voicing_map::l56_gate_from_b1;

/// Quantiser full scale: the ladder spans `[-0x4100, +0x4100]`.
const FULL_SCALE: i32 = 0x4100_0000;

/// `norm_l`: `0x1e - <index of the highest set bit of |v| as spelled by the
/// binary>`, masked to 16 bits, `0` for `v == 0`. `v == -1` leaves the search
/// index at its `0x1f` seed and yields `0xffff`; that is the binary's behaviour
/// and shift counts downstream are masked to 5 bits, so it is reproduced rather
/// than clamped.
fn norm_shift(v: i32) -> u16 {
    if v == 0 {
        return 0;
    }
    let a = if v < 0 { !v as u32 } else { v as u32 };
    let mut i: i32 = 31;
    if a != 0 {
        while (a >> i) == 0 {
            i -= 1;
        }
    }
    (0x1e_i32.wrapping_sub(i) as u32 & 0xffff) as u16
}

/// The block-float divide primitive: `num / den` with an exponent handoff.
/// Returns `(mantissa, exponent)`; a zero mantissa forces a zero exponent.
fn bfp_div(mut num: u32, mut e1: i16, mut den: i16, esub: i16) -> (i16, i16) {
    let sign = (den as i32) ^ (num as i32);
    if (num as i32) < 0 {
        num = if num == 0x8000_0000 {
            0x7fff_ffff
        } else {
            (num as i32).wrapping_neg() as u32
        };
    }
    if den < 0 {
        den = if den == i16::MIN { i16::MAX } else { -den };
    }
    if den == 0 {
        // Unreachable from the packer (the denominators are the ladder's own
        // range and a normalized mantissa); guarded so the primitive is total.
        return (0, 0);
    }
    if (den as i32).wrapping_mul(0x1_0000) <= num as i32 {
        num = ((num as i32) >> 1) as u32;
        e1 = e1.wrapping_add(1);
    }
    let mut m = (((num as i32) / (den as i32)) >> 1) as i16;
    if sign < 0 {
        m = m.wrapping_neg();
    }
    let u = (m as i32) << 16;
    let sh = norm_shift(u);
    let m = (u.wrapping_shl(u32::from(sh) & 0x1f) >> 16) as i16;
    if m == 0 {
        return (0, 0);
    }
    (m, e1.wrapping_sub(sh as i16).wrapping_sub(esub))
}

/// `shl` for a non-negative count, arithmetic `shr` clamped at 31 for a
/// negative one — the codec's rescale idiom, spelled as the packer spells it.
fn rescale(v: i32, count: i16) -> i32 {
    if count < 0 {
        if count < -30 {
            v >> 31
        } else {
            v >> ((-count) as u32 & 0x1f)
        }
    } else {
        v.wrapping_shl(count as u32 & 0x1f)
    }
}

/// The symmetric uniform scalar quantiser over `[-0x4100, +0x4100]`:
/// quantise `value` into `range` levels, clamp the index to `maxidx`, and
/// return `(index, reconstruction)`.
pub(crate) fn quantize(value: i16, range: i32, maxidx: i16) -> (i16, i16) {
    let r16 = range as i16;
    let u = (r16 as i32) << 16;
    let sh = norm_shift(u);
    let den = (u.wrapping_shl(u32::from(sh) & 0x1f) >> 16) as i16;
    let esub = (0x0f_u16.wrapping_sub(sh)) as i16;
    let (scale, scale_e) = bfp_div(FULL_SCALE as u32, 8, den, esub);

    let shifted = (((value as i32) << 16) >> 1).wrapping_add(0x2080_0000);
    let sh2 = norm_shift(shifted);
    let (m2, e2) = bfp_div(
        (shifted as u32).wrapping_shl(u32::from(sh2) & 0x1f),
        (8i32 - i32::from(sh2)) as i16,
        scale,
        scale_e,
    );
    let acc = rescale((m2 as i32) << 16, e2.wrapping_sub(15));
    let mut idx = (acc >> 16) as i16;
    if idx > r16.wrapping_sub(1) {
        idx = r16.wrapping_sub(1);
    } else if acc < 0 {
        idx = 0;
    }
    let mut index = maxidx;
    if idx <= maxidx {
        index = idx;
        if idx < 0 {
            index = 0;
        }
    }

    let mut recon = rescale(-FULL_SCALE, (-7i16).wrapping_sub(scale_e));
    let level =
        (((index as i32).wrapping_mul(0x2_0000).wrapping_add(0x1_0000) as u32) >> 16) as i16;
    recon = recon.wrapping_add((level as i32).wrapping_mul(scale as i32).wrapping_mul(2));
    let out = rescale(recon, scale_e.wrapping_add(7)).wrapping_add(0x8000) >> 16;
    (index, out as i16)
}

/// The branch's gate: the shared `L = 56` predicate over the frame's voicing
/// word. Exposed because the branch body's `(L, step)` override is decided by
/// the gate alone, independently of which index the classifier then picks.
pub(crate) fn l56_gate(b1: u16) -> bool {
    l56_gate_from_b1(b1)
}

/// The classifier's verdict for one frame: `Some(index)` when the branch fires
/// (the packer then emits this as `b̂₀`), `None` when the packer falls through
/// to the pitch quantiser.
///
/// `b1` is the frame's voicing index, `(f0_a, f0_b)` are the f0 ring's
/// `[1]` / `[2]` entries, and `bits` is the pitch field width (7).
pub(crate) fn tone_index(b1: u16, f0_a: i16, f0_b: i16, bits: u32) -> Option<i16> {
    if !l56_gate_from_b1(b1) {
        return None;
    }
    let v = crate::shared::voicing_map::VOICING_CB[(b1 as usize) * 4];
    // The classifier reads the two expanded halves separately: the low 8 codes
    // of `v` land in the `B` word, the high 8 in the `A` word. Only the code-2
    // presence per half is consulted (the code-1 test is the shared gate).
    let has_b = (v & 0x0000_aaaa) != 0;
    let has_a = (v & 0xaaaa_0000) != 0;
    let maxidx = ((0x3c_i32 << (bits - 6)) - 1) as i16;

    if !has_b {
        // Only the A half carries code-2 bands.
        Some(quantize(f0_a, 1 << bits, maxidx).0)
    } else if !has_a {
        Some(quantize(f0_b, 1 << bits, maxidx).0)
    } else {
        let mut r = 1i32 << (bits >> 1);
        if bits & 1 != 0 {
            r = ((r as i16 as i32).wrapping_mul(0x1_6a08) >> 16) & 0xffff;
        }
        let d_b = quantize(f0_b, r, maxidx).0;
        let d_a = quantize(f0_a, r, maxidx).0;
        let s = d_b.wrapping_add(d_a.wrapping_mul(r as i16));
        Some(if s > maxidx {
            maxidx
        } else if s < 0 {
            0
        } else {
            s
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `range = 128` admits a closed form; the quantiser must match it over the
    /// whole i16 input domain (index AND reconstruction).
    #[test]
    fn range128_matches_the_closed_form() {
        for v in i16::MIN..=i16::MAX {
            let (idx, recon) = quantize(v, 128, 119);
            let want = (((v as i64 + 16640) * 128) / 33280).clamp(0, 127) as i16;
            let want = want.min(119);
            assert_eq!(idx, want, "index at v={v}");
            let wr = (16640 * (2 * want as i64 + 1 - 128)) / 128;
            assert_eq!(recon as i64, wr, "recon at v={v}");
        }
    }

    /// `range = 11` does NOT admit that closed form. Pin the decision
    /// boundaries and the reconstruction table read out of the binary.
    #[test]
    fn range11_boundaries_and_reconstruction() {
        const BOUND: [i16; 10] = [
            -13614, -10589, -7563, -4538, -1513, 1513, 4538, 7563, 10589, 13614,
        ];
        const RECON: [i16; 11] = [
            -15127, -12102, -9077, -6051, -3026, 0, 3025, 6050, 9076, 12101, 15126,
        ];
        for (k, &b) in BOUND.iter().enumerate() {
            assert_eq!(quantize(b - 1, 11, 119).0, k as i16, "below boundary {k}");
            assert_eq!(quantize(b, 11, 119).0, k as i16 + 1, "at boundary {k}");
        }
        for (k, &r) in RECON.iter().enumerate() {
            let v = if k == 0 { i16::MIN } else { BOUND[k - 1] };
            assert_eq!(quantize(v, 11, 119).1, r, "recon at level {k}");
        }
    }

    /// The gate is the shared `L = 56` predicate: exactly `b1 in 18..=31`.
    #[test]
    fn gate_is_the_shared_predicate() {
        let fired: Vec<u16> = (0u16..32)
            .filter(|&b| tone_index(b, 1000, 2000, 7).is_some())
            .collect();
        assert_eq!(fired, (18u16..=31).collect::<Vec<_>>());
    }
}
