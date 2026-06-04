//! Offline synth-chain amplitude audit for Annex T tone frames.
//!
//! Sweeps `(I_D, A_D)` through `tone_to_mbe_params` →
//! `codecs::ambe_plus2::synthesize_frame` and measures steady-state PCM
//! peak/RMS. Compares to Eq. 209's expected slope: at A_D=127 we expect
//! the harmonic magnitude to be M̃_l = 16384, and the synth's peak
//! output should track 10^{0.03555·(A_D-127)} as A_D moves.
//!
//! Run:
//!   `cargo run --release --example tone_amp_audit`
//!
//! Reads no inputs; emits a CSV table to stdout.

use blip25_mbe::rate33::dequantize::{
    tone_to_mbe_params, TONE_AMPLITUDE_EXPONENT_STEP, TONE_AMPLITUDE_PEAK,
};
use blip25_mbe::codecs::ambe_plus2::{synthesize_frame, SynthState, SAMPLES_PER_FRAME};
use blip25_mbe::codecs::mbe_baseline::analysis::{detect_tone, ToneDetection};
use blip25_mbe::codecs::mbe_baseline::analysis::tone_detect::TONE_DETECT_FRAME;

const STEADY_FRAMES: usize = 30; // discard first half, measure second half

fn synth_steady(id: u8, amplitude: u8) -> Vec<i16> {
    let params = tone_to_mbe_params(id, amplitude).expect("valid Annex T id");
    let mut state = SynthState::new();
    let mut all = Vec::with_capacity(STEADY_FRAMES * SAMPLES_PER_FRAME);
    for _ in 0..STEADY_FRAMES {
        let frame = synthesize_frame(&params, &mut state);
        all.extend_from_slice(&frame);
    }
    // Drop the first half (transient); measure the latter half only.
    all[(STEADY_FRAMES / 2) * SAMPLES_PER_FRAME..].to_vec()
}

fn peak(pcm: &[i16]) -> f64 {
    pcm.iter().map(|&s| (s as f64).abs()).fold(0.0, f64::max)
}

fn rms(pcm: &[i16]) -> f64 {
    let s2: f64 = pcm.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (s2 / pcm.len() as f64).sqrt()
}

fn db20(x: f64) -> f64 {
    if x <= 0.0 { f64::NEG_INFINITY } else { 20.0 * x.log10() }
}

fn expected_m_tilde(amplitude: u8) -> f64 {
    TONE_AMPLITUDE_PEAK
        * 10f64.powf(TONE_AMPLITUDE_EXPONENT_STEP * (f64::from(amplitude) - 127.0))
}

fn sweep_amp(label: &str, id: u8, ad_values: &[u8]) {
    println!("# tone={label} id={id}");
    println!("ad,m_tilde,peak,rms,peak/m_tilde,peak_db_per_step,rms/m_tilde,note");
    let mut prev_peak: Option<f64> = None;
    let mut prev_ad: Option<u8> = None;
    for &ad in ad_values {
        let pcm = synth_steady(id, ad);
        let p = peak(&pcm);
        let r = rms(&pcm);
        let m = expected_m_tilde(ad);
        let p_per_m = if m > 0.0 { p / m } else { 0.0 };
        let r_per_m = if m > 0.0 { r / m } else { 0.0 };
        let db_per_step = match (prev_peak, prev_ad) {
            (Some(pp), Some(pad)) if pp > 0.0 && p > 0.0 && ad != pad => {
                db20(p / pp) / (f64::from(ad) - f64::from(pad))
            }
            _ => f64::NAN,
        };
        let note = if ad == 127 {
            "ref"
        } else if ad < 32 {
            "very low"
        } else {
            ""
        };
        println!(
            "{ad},{m:.1},{p:.1},{r:.1},{p_per_m:.4},{db_per_step:+.4},{r_per_m:.4},{note}"
        );
        prev_peak = Some(p);
        prev_ad = Some(ad);
    }
    println!();
}

fn main() {
    // Sweep A_D values: dense near full-scale + knox_1's measured 94 +
    // sparse low end. Eq. 209 step is 0.7115 dB/step.
    let ad_sweep: Vec<u8> = vec![
        0, 16, 32, 48, 64, 80, 90, 94, 96, 100, 104, 108, 112, 116, 120, 124, 127,
    ];

    // Single-tone references: id=2 has l1=l2=1 at f0=263.7 Hz (low),
    // id=10 around 600 Hz, id=20 around 1700 Hz. Pick a few to bracket
    // pitch range.
    // Annex T 0..4 are None. ids 5..12 are single-tone l1=l2=1. ids
    // 13..25 are single-tone l1=l2=2. id=128+ are DTMF/Knox.
    sweep_amp("single-tone-id=5 (l=1, 156 Hz)", 5, &ad_sweep);
    sweep_amp("single-tone-id=10 (l=1, 312 Hz)", 10, &ad_sweep);
    sweep_amp("single-tone-id=20 (l=2, 359 Hz)", 20, &ad_sweep);

    // Knox / DTMF — two-voiced-harmonic. id=145 is the knox_1 match.
    sweep_amp("knox-id=145 (l1=4, l2=7)", 145, &ad_sweep);
    sweep_amp("dtmf-id=128", 128, &ad_sweep);

    // Slope diagnostic: do the peak measurements honor 0.7115 dB/step?
    // The "peak_db_per_step" column should be ~0.7115 for a perfectly
    // calibrated synth.
    println!("# Eq. 209 reference: 10·log10(10^0.03555) = {:.4} dB/step",
        20.0 * TONE_AMPLITUDE_EXPONENT_STEP);
    println!("# (the per-step delta in 'peak_db_per_step' should match this)");

    // Round-trip probe: known input → encode → decode → output.
    println!();
    println!("=== Round-trip: PCM in → detect_tone → tone_to_mbe_params → synth → PCM out ===");
    println!("freq_hz,input_peak,detected_id,detected_ad,output_peak,output_rms,roundtrip_db,note");
    for &freq in &[156.25_f64, 218.75, 312.5, 375.0] {
        for &input_peak in &[1024_i16, 2048, 3184, 4096, 8192, 16384, 24576, 32760] {
            let pcm = pure_sine_frame(freq, input_peak);
            let det = detect_tone(&pcm);
            match det {
                Some(ToneDetection { id, amplitude }) => {
                    let out = synth_steady(id, amplitude);
                    let p = peak(&out);
                    let r = rms(&out);
                    let rt_db = if input_peak > 0 {
                        20.0 * (p / input_peak as f64).log10()
                    } else {
                        f64::NAN
                    };
                    println!(
                        "{freq:.2},{input_peak},{id},{amplitude},{p:.1},{r:.1},{rt_db:+.3},single-tone"
                    );
                }
                None => {
                    println!("{freq:.2},{input_peak},,,,,,,no-detection");
                }
            }
        }
    }

    // Two-tone round-trip (Knox/DTMF shape).
    println!();
    println!("=== Round-trip two-tone: synth knox-like input at varying per-tone peak ===");
    println!("f1,f2,per_tone_peak,total_peak,detected_id,detected_ad,output_peak,output_rms,roundtrip_db");
    // id=145 = (4·150.89, 7·150.89) = (603.56, 1056.23)
    let f1 = 4.0 * 150.89;
    let f2 = 7.0 * 150.89;
    for &per_tone in &[512_i16, 1024, 1592, 2048, 3184, 4096, 8192, 12000] {
        let pcm = sum_two_sines_frame(f1, f2, per_tone);
        let total_peak = peak(&pcm) as i16;
        match detect_tone(&pcm) {
            Some(ToneDetection { id, amplitude }) => {
                let out = synth_steady(id, amplitude);
                let p = peak(&out);
                let r = rms(&out);
                let rt_db = if total_peak > 0 {
                    20.0 * (p / total_peak as f64).log10()
                } else {
                    f64::NAN
                };
                println!(
                    "{f1:.1},{f2:.1},{per_tone},{total_peak},{id},{amplitude},{p:.1},{r:.1},{rt_db:+.3}"
                );
            }
            None => {
                println!("{f1:.1},{f2:.1},{per_tone},{total_peak},,,,,no-detection");
            }
        }
    }
}

fn pure_sine_frame(freq_hz: f64, amplitude: i16) -> [i16; TONE_DETECT_FRAME] {
    let mut out = [0i16; TONE_DETECT_FRAME];
    let two_pi = 2.0 * core::f64::consts::PI;
    for n in 0..TONE_DETECT_FRAME {
        let t = n as f64 / 8000.0;
        out[n] = ((amplitude as f64) * (two_pi * freq_hz * t).sin()).round() as i16;
    }
    out
}

fn sum_two_sines_frame(f1: f64, f2: f64, per_tone_peak: i16) -> [i16; TONE_DETECT_FRAME] {
    let mut out = [0i16; TONE_DETECT_FRAME];
    let two_pi = 2.0 * core::f64::consts::PI;
    for n in 0..TONE_DETECT_FRAME {
        let t = n as f64 / 8000.0;
        let s = (per_tone_peak as f64)
            * ((two_pi * f1 * t).sin() + (two_pi * f2 * t).sin());
        out[n] = s.round().clamp(-32768.0, 32767.0) as i16;
    }
    out
}
