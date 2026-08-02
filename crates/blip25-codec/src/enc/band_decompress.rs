//! The encoder's per-band loudness/energy integration step, plus its
//! surrounding normalize / denormalize / log2 / bias-clamp helpers.
//!
//! ## What this computes
//!
//! Given a 129-entry table of per-bin values (`bins`, the magnitude-squared
//! spectrum `2*(Re^2+Im^2)` anchored at `encoder_buf+0x1c`), a phase-accumulator
//! step (`step`), an output count, and a small per-frame "outer" exponent field,
//! this produces `count` output words: a phase-accumulator resampling of the
//! table into per-output-band loudness values (interpolating between two
//! adjacent table entries, or summing multiple full entries when the step
//! advances more than one table entry per output), normalized to a floating
//! representation, log2-compressed, and bias/clamped.
//!
//! ## What is NOT yet wired up
//!
//! The caller-side derivation of `step` -- a per-frame phase-step "seed",
//! believed but not confirmed to be a direct function of the estimated
//! pitch/fundamental -- is unresolved. Its `COUNT`->`STEP` lookup table is only
//! ~6-7 entries known out of an estimated 30-60.
//!
//! The `outer` field (a small per-frame signed exponent, observed range roughly
//! -2..-42, feeding both the silence-floor test and the log2 exponent) is closer:
//! [`super::loudness_transform::fft_bfp_transform`] composed with
//! [`super::outer_transform::combine_outer`] reproduces it directly from the same
//! windowed spectral array that is this module's `bins` input's source. See
//! [`super::loudness_transform`] for that chain's accuracy.
//!
//! **`Encoder::analyze_frame` does not use this path for shipped bits.** It gets
//! its `b0..b8` amplitudes via `loudness_fixed::amps_from_bins_omega`, which calls
//! `band_decompress` with its own `outer` constant, not the full
//! `compute_outer`/`fft_bfp_transform` chain. Closing `step` and switching
//! `analyze_frame` over to the mechanism-matched path are both open, substantial
//! tasks. Improving the sub-mechanism does not by itself change the shipped
//! encoder's measured accuracy.

const MASK32: u32 = 0xFFFF_FFFF;

use crate::fixops::u32r::{imul32, s16, s32, shl32, u16v, u32v};

/// Arithmetic shift right on the 32-bit two's-complement value `x`, shift
/// count taken mod 32 (matches x86 `sar`).
#[inline]
fn sar32(x: u32, n: u32) -> u32 {
    u32v((s32(x)) >> (n & 0x1f))
}

// `normalize64` / `shift64` (the 64-bit block-float normalize + shift helpers)
// moved to `crate::shared::q_energy` (alongside `q_builder`, which rides on
// them) so the decoder can reach them without `enc/`. Re-exported so this
// module and `enc::b1_audio` / `enc::pq_builder` call sites are unchanged.
pub(crate) use crate::shared::q_energy::{normalize64, shift64};

/// The 6 signed int16 coefficients of the reference vocoder's log2 polynomial
/// table. There are 6, not 5: the last coefficient IS used, in the final stage.
/// Fixed data, not derived at runtime.
const LOG2_COEF: [i16; 6] = [
    0x5a82u16 as i16,
    0xa3acu16 as i16,
    0x570du16 as i16,
    0xa3acu16 as i16,
    0x414au16 as i16,
    0xc000u16 as i16,
];

/// A `bsr`-based leading-bit scan combined with a 6-term Horner-style
/// fixed-point log2 polynomial.
pub(crate) fn log2_fn(mantissa16: i16, exp_arg: i16) -> u32 {
    let [c0, c1, c2, c3, c4, c5] = LOG2_COEF;
    let mant_sh = shl32((mantissa16 as i32 as u32) & MASK32, 16);
    let norm_sh: i32 = if mant_sh == 0 {
        0
    } else {
        let bsr_in = if s32(mant_sh) >= 0 { mant_sh } else { !mant_sh };
        let bitpos = if bsr_in == 0 {
            0
        } else {
            31 - bsr_in.leading_zeros() as i32
        };
        0x1e - bitpos
    };
    let sh = (norm_sh as u32) & 0xff;
    let c0_term = shl32(c0 as u16 as u32, 16);
    let scaled = u32v(s32(shl32(mant_sh, sh & 0x1f)).wrapping_sub(s32(c0_term)));
    let scaled = sar32(scaled, 16);
    let x0 = s16(u16v(scaled));
    let poly_x = x0 as i32;

    // Stage 1 (C1 -> X1)
    let prod = imul32(poly_x, c1 as i32);
    let mut acc = (prod.wrapping_mul(2)).wrapping_add(0x8000) & MASK32;
    let cterm = shl32(c2 as u16 as u32, 16);
    acc &= 0xFFFF_0000;
    acc = acc.wrapping_add(cterm);
    acc = sar32(acc, 16);
    let x1 = s16(u16v(acc));

    // Stage 2 (X1,C3 -> X2)
    let prod = imul32(poly_x, x1 as i32);
    let mut acc = (prod.wrapping_mul(2)).wrapping_add(0x8000) & MASK32;
    let cterm = shl32(c3 as u16 as u32, 16);
    acc &= 0xFFFF_0000;
    acc = acc.wrapping_add(cterm);
    acc = sar32(acc, 16);
    let x2 = s16(u16v(acc));

    // Stage 3 (X2,C4 -> X3) -- has an extra `sar 1` not present in the other stages.
    let prod = imul32(poly_x, x2 as i32);
    let mut acc = (prod.wrapping_mul(2)).wrapping_add(0x8000) & MASK32;
    let cterm = shl32(c4 as u16 as u32, 16);
    acc = sar32(acc, 1);
    acc &= 0xFFFF_8000;
    acc = acc.wrapping_add(cterm);
    acc = sar32(acc, 16);
    let x3 = s16(u16v(acc));

    // Final stage (X3,C5 -> fractional part)
    let prod = imul32(poly_x, x3 as i32);
    let cterm = shl32(c5 as u16 as u32, 16);
    let mut acc = (prod.wrapping_mul(8)).wrapping_add(0x2_0000) & MASK32;
    acc &= 0xFFFC_0000;
    acc = acc.wrapping_add(cterm);
    let frac_part = sar32(acc, 15);

    let exp_local = (exp_arg as i32) - norm_sh;
    let exp_part = shl32((exp_local as i16 as u16) as u32, 16);
    u32v(s32(frac_part).wrapping_add(s32(exp_part)))
}

/// A saturating variable left-shift-or-arithmetic-right-shift utility.
pub(crate) fn shift_scale(value: u32, count: i16) -> u32 {
    if count >= 0 {
        let sh = count as u32;
        let mask_sh = 31u32.wrapping_sub(sh);
        let mask = shl32(0xFFFF_FFFF, mask_sh & 0x1f);
        let mut overflow = mask & value;
        if s32(value) < 0 {
            overflow = overflow.wrapping_sub(mask);
        }
        if overflow != 0 {
            if s32(value) < 0 {
                0x8000_0000
            } else {
                0x7FFF_FFFF
            }
        } else {
            shl32(value, sh & 0x1f)
        }
    } else {
        let mag = (-(count as i32)) as u32;
        if mag < 31 {
            sar32(value, mag & 0x1f)
        } else {
            sar32(value, 31)
        }
    }
}

/// The per-element bias/clamp body: `+0xAA` bias with overflow-saturate,
/// then a `0x77FF` ceiling clamp.
pub(crate) fn bias_clamp_one(v16: i16) -> i16 {
    let val_sh = shl32((v16 as i32 as u32) & MASK32, 16);
    let biased = val_sh.wrapping_add(0xAA_0000);
    let overflow = (biased ^ val_sh) & biased;
    let result16 = if s32(overflow) >= 0 {
        u16v(sar32(biased, 16))
    } else if s32(val_sh) < 0 {
        u16v(sar32(0x8000_0000, 16))
    } else {
        u16v(sar32(0x7FFF_FFFF, 16))
    };
    let mut r16 = s16(result16);
    if r16 > 0x77FFu16 as i16 {
        r16 = 0x77FFu16 as i16;
    }
    r16
}

/// Composes [`super::loudness_transform::fft_bfp_transform`] and
/// [`super::outer_transform::combine_outer`] into the reference's complete
/// `outer(frame) = 2 * ((i16)(fft_bfp_transform(a1ptr, a2ptr, arg3-1)) + 1)`
/// formula. `win` is mutated in place exactly like the reference's scratch
/// buffer.
///
/// Not called from the live encoder -- see this module's "What is NOT yet wired
/// up" section for why.
pub(crate) fn compute_outer(win: &mut [i16], base: usize, arg2: i16, arg3: u32) -> i16 {
    // This is the LOUDNESS call site, so it uses `loudness_array_transform` (single-stage
    // array mutation), NOT `fft_bfp_transform` (which selects the VOICING call
    // site's genuinely different multi-stage behavior). See
    // `loudness_transform::array_transform_with_mode`.
    let loudness = super::loudness_transform::loudness_array_transform(win, base, arg2, arg3);

    // The per-frame sequencer is:
    //   1. a windowing step writes into `win` (a1ptr)
    //   2. an inner routine, on the SAME `win` array and back to back, calls
    //      BOTH the fft_bfp_transform AND a second array transform
    //   3. the routine that is the source of `bins` reads `win` AFTER step 2 --
    //      i.e. after BOTH transforms, not just fft_bfp_transform.
    // So `compute_outer` must run the second transform too, even though
    // `outer_transform::combine_outer` correctly states that the second
    // transform's array work never feeds the SCALAR `outer` return: that is a
    // different claim from "the array content doesn't matter downstream", and
    // the `bins`-source routine's input does depend on the second mutation.
    //
    // `array_a_stage2::inverse_fft_butterfly_stage` is the `arg4=0, arg5=1`
    // branch this call site uses: `arg4` is a hardcoded literal 0 inside the
    // inner sequencer, and `arg5` is the inner sequencer's 4th argument,
    // threaded from the windowing step's literal `1`. The second transform's
    // `arg3` is fft_bfp_transform's `arg3 + 1`, since fft_bfp_transform receives
    // `arg3 - 1` from the same caller:
    // `fft_bfp_transform(a1ptr, a2ptr, arg3-1)` then
    // `second_transform(a1ptr, loudness, arg3, 0, arg5)`.
    //
    // `inverse_fft_butterfly_stage` needs 2 scratch words past the 256-word window (see
    // its own doc) that this call site never reads back (bins only reads
    // win[0..256]), so a local 258-word buffer is used and only the
    // in-window part is copied back.
    let win_len = win.len().min(256);
    let mut scratch = [0i16; 258];
    scratch[..win_len].copy_from_slice(&win[..win_len]);
    let _ =
        super::array_a_stage2::inverse_fft_butterfly_stage(&mut scratch, loudness, arg3 + 1, 0, 1);
    win[..win_len].copy_from_slice(&scratch[..win_len]);

    super::outer_transform::combine_outer(loudness)
}

/// The phase-accumulator band-loudness integration loop.
///
/// * `step` — the raw 16-bit phase-step seed (`array[0xc]` in the real
///   object; caller-side derivation not yet confirmed, see module docs).
/// * `count` — number of output words to produce (`array[0x4]`).
/// * `bins` — the 129-entry per-bin table (the confirmed magnitude-squared
///   spectrum, `2*(Re^2+Im^2)`, as signed 32-bit values — out-of-range
///   reads past `bins[128]` clamp to `bins[128]`, matching the original's
///   own lack of bounds checking on a fixed 129-entry table).
/// * `outer` — the small per-frame signed exponent field (`ctx+0x40`, the
///   integration loop's own arg4; derivation not yet confirmed).
///
/// Returns `count` **raw** (pre-bias) output words; apply
/// [`bias_clamp_one`] to each to match the real reference's final per-band
/// output array.
pub(crate) fn band_decompress(step: i16, count: usize, bins: &[i32], outer: i16) -> Vec<i16> {
    debug_assert_eq!(bins.len(), 129);
    let bin_at = |i: i32| -> i32 {
        let i = i.clamp(0, 128) as usize;
        bins[i]
    };

    let phase_step = sar32(shl32((step as i32 as u32) & MASK32, 9), 4);
    let half_step = sar32(phase_step, 1);
    let phase0 = half_step.wrapping_add(0x8000);
    let mut idx_state: u16 = u16v(phase0 >> 16);
    let mut dx_state: u16 = u16v(phase0);
    let mut ebp_state: u16 = u16v(phase0);

    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let old_idx = idx_state as i32;
        let cur_phase = shl32((s16(old_idx as u16) as i32 as u32) & MASK32, 16);
        let cur_phase = cur_phase | (dx_state as u32);
        let new_phase = cur_phase.wrapping_add(phase_step);
        let new_idx_raw = u16v(sar32(new_phase, 16));
        let new_idx_clamped: u16 = if s16(new_idx_raw) < 0x7f {
            new_idx_raw
        } else {
            0x7f
        };
        let idx_delta = u16v(u32v((new_idx_clamped as i32) - old_idx));

        let frac1 = ((ebp_state as u32) >> 1) & 0x7fff;
        let frac2 = ((u16v(new_phase) as u32) >> 1) & 0x7fff;

        // term1 = bin[old_idx] - round(bin[old_idx] * frac1)
        let bin1 = bin_at(old_idx);
        let bin1_hi = ((bin1 >> 16) & 0xFFFF) as i16 as i32;
        let bin1_lo = (bin1 as u32 & 0xFFFF) as i32;
        let p_lo = sar32(imul32(bin1_lo, frac1 as i32), 15);
        let p_hi = imul32(bin1_hi, frac1 as i32);
        let prod = p_lo.wrapping_add(p_hi.wrapping_mul(2));
        let prod_sat = if prod == 0x8000_0000 {
            0x7FFF_FFFFu32
        } else {
            u32v(-s32(prod))
        };
        let term1 = bin1.wrapping_add(prod_sat as i32);

        // term2 = round(bin[old_idx + idx_delta] * frac2) -- NOT a fixed
        // old_idx+1.
        let bin2 = bin_at(old_idx + idx_delta as i32);
        let bin2_hi = ((bin2 >> 16) & 0xFFFF) as i16 as i32;
        let bin2_lo = (bin2 as u32 & 0xFFFF) as i32;
        let q_lo = sar32(imul32(bin2_lo, frac2 as i32), 15);
        let q_hi = imul32(bin2_hi, frac2 as i32);
        let term2 = s32(q_lo.wrapping_add(q_hi.wrapping_mul(2)));

        let mut big: i64 = term1 as i64 + term2 as i64;

        // Extra full-resolution bins when idx_delta > 1: sums
        // bin[old_idx+1 .. old_idx+idx_delta-1] (idx_delta-1 raw terms;
        // bin[old_idx+idx_delta] is already covered, interpolated, by
        // term2 above -- so every table entry from old_idx+1 through
        // old_idx+idx_delta is counted exactly once).
        if (idx_delta as i16) > 1 {
            let extra_n = idx_delta as i32 - 1;
            for k in 0..extra_n {
                big += bin_at(old_idx + 1 + k) as i64;
            }
        }

        let acc_lo = (big as u64 & 0xFFFF_FFFF) as u32;
        let acc_hi = ((big as u64) >> 32) as u32;

        let shift = normalize64(acc_lo, acc_hi);
        let (norm_lo, _norm_hi) = shift64(acc_lo, acc_hi, shift);
        let mant16 = s16(u16v(norm_lo >> 16));

        let shift_u16 = u16v(shift as u32);
        let diffval = u16v(u32v(outer as i32 - shift_u16 as i32));
        let take_floor = s16(diffval) < (0xffc5u16 as i16) || mant16 == 0;

        let raw16 = if take_floor {
            0x8800u16 as i16
        } else {
            let exp_arg = u16v(u32v(outer as i32 - shift_u16 as i32 + 0x1e)) as i16;
            let logres = log2_fn(mant16, exp_arg);
            let shifted = shift_scale(logres, 0xf - 5);
            let rounded = shifted.wrapping_add(0x8000);
            let ov = (rounded ^ shifted) & rounded;
            let fin = if s32(ov) >= 0 {
                sar32(rounded, 16)
            } else if s32(shifted) < 0 {
                sar32(0x8000_0000, 16)
            } else {
                sar32(0x7FFF_FFFF, 16)
            };
            s16(u16v(fin))
        };

        out.push(raw16);

        idx_state = new_idx_clamped;
        dx_state = u16v(new_phase);
        ebp_state = u16v(new_phase);
    }
    out
}

// ===================== LOW-BAND FLOOR =====================
// reference-target only. Verbatim port, objdump-verified
// instruction-by-instruction. Applied AFTER band_decompress + bias_clamp,
// matching the real call order (decompress -> bias -> gated low-band floor).
//
// GATING (this is the whole subtlety): the call site tests bit 0 of the
// `a4` byte and, when it is clear, skips the ENTIRE call: the floor loop
// AND the "always-run" L0 tail. `l0` is a RECURSIVE leaky integrator, so
// advancing it on a gated-off frame desyncs the whole remaining
// trajectory. An earlier attempt ran the tail unconditionally and thereby
// helped `mark` (M_l 141->182/199) while REGRESSING `voiced` (195->180):
// voiced has 72 gated-off frames vs mark's 43.
//
// Live-captured (`[lb750]`/`[lb7b0]` hooks, both PCMs, 199 frames):
//   * the outer call fires exactly ONCE per frame, 199/199 both files --
//     there is no sub-frame update rate.
//   * the floor call fires 127/199 (voiced) and 156/199 (mark) -- the gate
//     is real and content-dependent.
//   * `l0` init = 13107 (0x3333); gated, this tail reproduces the captured
//     `l0` trajectory 199/199 on BOTH files.
// With the gate correct, self-driven M_l is exact on every scoreable frame of
// both files (voiced 196/196, mark 197/197).
