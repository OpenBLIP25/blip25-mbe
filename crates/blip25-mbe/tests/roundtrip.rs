//! Property tests — invariants that must hold across the pipeline.
//!
//! These run on synthetic inputs only; no DVSI material required, so
//! they stay CI-safe. End-to-end validation against DVSI test vectors
//! lives in the `conformance-vectors` harness (`rate-convert`,
//! `rate-convert-roundtrip` subcommands).
//!
//! ## Rate-conversion regression net
//!
//! Builds a stream of voice-like MbeParams frames across a pitch
//! sweep that stays inside both the full-rate (BABA-A §1.3.1 Eqs.
//! 46–47, `b̂₀ ∈ [0, 207]`) and half-rate (Annex L, 120 entries)
//! grids with comfortable margin, then measures parameter distortion
//! through the converters. Thresholds are picked from empirical
//! observation on DVSI `alert` — loose enough for the quantizer's
//! lossy gain/DCT stages, tight enough to catch structural regressions
//! (e.g. a broken pitch table, a priority-map off-by-one, a sign flip
//! in `disassemble_hoc_matrix`).
//!
//! **Boundary behaviour** — inputs at the extreme edges of either
//! grid can fall outside the opposite grid after requantization
//! (e.g. half-rate ω₀=0.051051 maps to full-rate b̂₀=207 → 0.050979,
//! which is below the half-rate minimum). The test set deliberately
//! stays inside the intersection of the two grids; the
//! `conformance-vectors rate-convert-roundtrip` subcommand surfaces
//! the boundary behaviour on real speech.

use blip25_mbe::mbe_params::MbeParams;
use blip25_mbe::p25_fullrate::dequantize::{
    DecoderState, dequantize as dequantize_full, quantize as quantize_full,
};
use blip25_mbe::p25_fullrate::frame::{
    decode_frame as decode_full_frame, encode_frame as encode_full_frame,
};
use blip25_mbe::p25_fullrate::priority::prioritize as prioritize_full;
use blip25_mbe::p25_halfrate::dequantize::{
    DecoderState as HalfDecoderState, dequantize as dequantize_half,
    quantize as quantize_half,
};
use blip25_mbe::p25_halfrate::frame::{
    decode_frame as decode_half_frame, encode_frame as encode_half_frame,
};
use blip25_mbe::rate_conversion::{FullToHalfConverter, HalfToFullConverter};

/// Pitch sweep staying well inside both grids' interior (excludes the
/// top/bottom boundary rows where cross-grid rounding can fall off).
/// Full-rate ω₀ range is [4π/246.5, 4π/39.5] ≈ [0.05098, 0.31819];
/// half-rate ω₀ range is [0.05105, 0.31397]. These all sit comfortably
/// inside both.
const PITCH_SWEEP: &[f32] = &[
    0.060, 0.085, 0.110, 0.140, 0.175, 0.210, 0.240, 0.275, 0.300,
];

/// Build a voice-like MbeParams at the given ω₀ with a plausible
/// amplitude envelope and a mid-band V/UV transition (voiced low,
/// unvoiced high — the common case for speech).
fn voice_params(omega_0: f32) -> MbeParams {
    let l = MbeParams::harmonic_count_for(omega_0);
    let voiced: Vec<bool> = (1..=l).map(|h| h <= l / 2).collect();
    // Smooth 1/f-ish envelope scaled to typical speech range.
    let amps: Vec<f32> = (1..=l)
        .map(|h| 100.0 * (-(h as f32) * 0.05).exp())
        .collect();
    MbeParams::new(omega_0, l, &voiced, &amps).unwrap()
}

fn stream_full_to_half(params: &MbeParams, n_frames: usize) -> (MbeParams, MbeParams) {
    let mut enc_state = DecoderState::new();
    let mut src_state = DecoderState::new();
    let mut dst_state = HalfDecoderState::new();
    let mut conv = FullToHalfConverter::new();
    let mut last_src = params.clone();
    let mut last_dst = params.clone();
    for _ in 0..n_frames {
        let b = quantize_full(params, &mut enc_state).expect("quantize_full");
        let u = prioritize_full(&b, params.harmonic_count());
        let dibits = encode_full_frame(&u);

        let src_frame = decode_full_frame(&dibits);
        let src_params =
            dequantize_full(&src_frame.info, &mut src_state).expect("source dequant");

        let dst_dibits = conv.convert(&dibits).expect("convert full→half");
        let dst_frame = decode_half_frame(&dst_dibits);
        let dst_params =
            dequantize_half(&dst_frame.info, &mut dst_state).expect("dst dequant");

        last_src = src_params;
        last_dst = dst_params;
    }
    (last_src, last_dst)
}

fn stream_half_to_full(params: &MbeParams, n_frames: usize) -> (MbeParams, MbeParams) {
    let mut enc_state = HalfDecoderState::new();
    let mut src_state = HalfDecoderState::new();
    let mut dst_state = DecoderState::new();
    let mut conv = HalfToFullConverter::new();
    let mut last_src = params.clone();
    let mut last_dst = params.clone();
    for _ in 0..n_frames {
        let u = quantize_half(params, &mut enc_state).expect("quantize_half");
        let dibits = encode_half_frame(&u);

        let src_frame = decode_half_frame(&dibits);
        let src_params =
            dequantize_half(&src_frame.info, &mut src_state).expect("source dequant");

        let dst_dibits = conv.convert(&dibits).expect("convert half→full");
        let dst_frame = decode_full_frame(&dst_dibits);
        let dst_params =
            dequantize_full(&dst_frame.info, &mut dst_state).expect("dst dequant");

        last_src = src_params;
        last_dst = dst_params;
    }
    (last_src, last_dst)
}

fn pitch_cents(src: &MbeParams, dst: &MbeParams) -> f64 {
    1200.0 * (f64::from(dst.omega_0()) / f64::from(src.omega_0())).log2()
}

fn voicing_agreement(src: &MbeParams, dst: &MbeParams) -> f64 {
    let l = (src.harmonic_count().min(dst.harmonic_count())) as usize;
    if l == 0 {
        return 1.0;
    }
    let matches = (0..l)
        .filter(|&i| src.voiced_slice()[i] == dst.voiced_slice()[i])
        .count();
    matches as f64 / l as f64
}

#[test]
fn full_to_half_preserves_pitch_within_one_grid_step() {
    // The worst-case re-quantization error for a pitch that sits
    // between two half-rate grid points is half the half-rate step.
    // 40 cents is a comfortable margin (half-rate Annex L step in the
    // middle of the table is ~1% ≈ 17 cents).
    for &omega_0 in PITCH_SWEEP {
        let params = voice_params(omega_0);
        let (src, dst) = stream_full_to_half(&params, 6);
        let cents = pitch_cents(&src, &dst).abs();
        assert!(
            cents < 40.0,
            "ω₀={omega_0}: full→half pitch drifted {cents:.2} cents"
        );
        let l_delta =
            (src.harmonic_count() as i32 - dst.harmonic_count() as i32).abs();
        assert!(l_delta <= 1, "ω₀={omega_0}: ΔL={l_delta} exceeds 1");
    }
}

#[test]
fn half_to_full_preserves_pitch_within_one_grid_step() {
    for &omega_0 in PITCH_SWEEP {
        let params = voice_params(omega_0);
        let (src, dst) = stream_half_to_full(&params, 6);
        let cents = pitch_cents(&src, &dst).abs();
        assert!(
            cents < 40.0,
            "ω₀={omega_0}: half→full pitch drifted {cents:.2} cents"
        );
        let l_delta =
            (src.harmonic_count() as i32 - dst.harmonic_count() as i32).abs();
        assert!(l_delta <= 1, "ω₀={omega_0}: ΔL={l_delta} exceeds 1");
    }
}

#[test]
fn full_to_half_preserves_voicing_majority() {
    // Half-rate uses 8 codebook slots with variable harmonic grouping;
    // full-rate uses K bands of 3 harmonics each. Exact per-harmonic
    // agreement isn't guaranteed, but ≥ 75% catches a broken Annex M
    // codebook lookup.
    for &omega_0 in PITCH_SWEEP {
        let params = voice_params(omega_0);
        let (src, dst) = stream_full_to_half(&params, 6);
        let agreement = voicing_agreement(&src, &dst);
        assert!(
            agreement >= 0.75,
            "ω₀={omega_0}: voicing agreement {:.1}% < 75%",
            agreement * 100.0
        );
    }
}

#[test]
fn half_to_full_preserves_voicing_mostly() {
    // Half-rate's 5-bit Annex M codebook has 32 patterns that quantize
    // per-harmonic voicing through 8 grouped slots (§2.3.6). A
    // synthetic mid-band transition lands between codebook entries at
    // some pitches, costing several harmonics — at ω₀=0.21 only 80%
    // of the 28 harmonics match. 75% is tight enough to catch a
    // broken Annex M lookup and loose enough for the worst synthetic
    // pattern. (DVSI speech runs ≥ 95% in practice; see
    // `rate-convert --direction half-to-full`.)
    for &omega_0 in PITCH_SWEEP {
        let params = voice_params(omega_0);
        let (src, dst) = stream_half_to_full(&params, 6);
        let agreement = voicing_agreement(&src, &dst);
        assert!(
            agreement >= 0.75,
            "ω₀={omega_0}: voicing agreement {:.1}% below 75%",
            agreement * 100.0
        );
    }
}

#[test]
fn full_round_trip_converges_within_grid_step() {
    // full → half → full. After the two quantizer passes the second
    // hop sees an already-full-rate-gridded ω₀, so the round-trip
    // pitch drift collapses to the half-rate step size.
    for &omega_0 in PITCH_SWEEP {
        let params = voice_params(omega_0);
        let mut a_to_b = FullToHalfConverter::new();
        let mut b_to_a = HalfToFullConverter::new();
        let mut enc_state = DecoderState::new();
        let mut src_state = DecoderState::new();
        let mut final_state = DecoderState::new();
        let mut last_src = params.clone();
        let mut last_final = params.clone();

        for _ in 0..6 {
            let b = quantize_full(&params, &mut enc_state).unwrap();
            let u = prioritize_full(&b, params.harmonic_count());
            let dibits = encode_full_frame(&u);

            let src_frame = decode_full_frame(&dibits);
            let src_params =
                dequantize_full(&src_frame.info, &mut src_state).unwrap();

            let mid = a_to_b.convert(&dibits).unwrap();
            let back = b_to_a.convert(&mid).unwrap();

            let final_frame = decode_full_frame(&back);
            let final_params =
                dequantize_full(&final_frame.info, &mut final_state).unwrap();
            last_src = src_params;
            last_final = final_params;
        }

        let cents = pitch_cents(&last_src, &last_final).abs();
        assert!(
            cents < 30.0,
            "ω₀={omega_0}: round-trip pitch drift {cents:.2} cents"
        );
    }
}

#[test]
fn silence_does_not_panic_either_converter() {
    // Silence: all unvoiced, zero amplitudes. MbeParams::silence()
    // sits at the top of the full-rate grid (ω₀ = 4π/39.5), which is
    // above the half-rate grid top. The converter may legitimately
    // return HalfPitchOutOfRange — the invariant is just that nothing
    // panics (the boundary case is covered by the DVSI harness).
    let params = MbeParams::silence();
    let mut enc_state = DecoderState::new();
    let mut conv = FullToHalfConverter::new();
    for _ in 0..3 {
        let b = quantize_full(&params, &mut enc_state).unwrap();
        let u = prioritize_full(&b, params.harmonic_count());
        let dibits = encode_full_frame(&u);
        let _ = conv.convert(&dibits);
    }
}
