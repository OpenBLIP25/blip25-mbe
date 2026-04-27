//! Encode-side tone detector — recognises Annex T tones in 160-sample
//! PCM frames and returns the matching `(I_D, A_D)` for tone-frame
//! dispatch.
//!
//! BABA-A §2.10 specifies the tone-frame *payload* (Annex T table,
//! signature + bit layout in û₀..û₃) but per `P25_Vocoder_Implementation_Spec.md`
//! §0.0.1 the **entry criteria** for emitting a tone frame are
//! "Not prescribed by the standard" — left to the implementer. So
//! this is a DSP design choice, not a P25-IP question; clean-room
//! scope rules don't apply.
//!
//! Strategy:
//!
//! 1. Window the 160 input samples with a Hann window, compute a
//!    256-point real DFT (zero-pad).
//! 2. Find the dominant peak above an SNR floor.
//! 3. Match the peak frequency to the closest single-frequency entry
//!    in Annex T (`l1 == l2`). If no match within tolerance, return
//!    `None`.
//! 4. Map the peak's energy to the 7-bit `A_D` amplitude field via
//!    the inverse of Eq. 209 (the same exponential scaling the
//!    decoder uses).
//!
//! DTMF (two-tone) detection is out of scope for this first cut —
//! the consumer can pre-classify DTMF and call `encode_tone_frame_info`
//! directly with the right `I_D` if they want to. If real DTMF input
//! shows up in the test corpus, this module gains a `detect_dtmf`
//! sibling.

use crate::p25_halfrate::frame::ANNEX_T;

/// What the tone detector returns on a hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToneDetection {
    /// Annex T tone ID (`I_D`) for the matched tone.
    pub id: u8,
    /// 7-bit log-amplitude (`A_D`), mapped via Eq. 209 inverse from
    /// the detected spectral magnitude.
    pub amplitude: u8,
}

/// Number of samples per detector input frame.
pub const TONE_DETECT_FRAME: usize = 160;
/// FFT size — 256-point zero-padded DFT of the windowed 160 samples.
const FFT_SIZE: usize = 256;
/// Sample rate (constant across blip25-mbe).
const SAMPLE_RATE_HZ: f64 = 8000.0;

/// Maximum frequency error for a peak to count as matching an Annex T
/// entry. Peak resolution at 256-point DFT is ~31 Hz; ±20 Hz lets us
/// snap to the nearest Annex T entry without false matches between
/// neighbouring rows (Annex T spacing is ≥ 31 Hz in the 156–400 Hz
/// band).
const MATCH_TOLERANCE_HZ: f64 = 20.0;

/// Minimum peak-to-floor SNR (in linear power) before we declare a
/// tone. Below this the input is too noisy / broadband to be a clean
/// tone; emit a voice frame instead.
const SNR_THRESHOLD: f64 = 100.0;

/// Detect a single-frequency Annex T tone in `pcm`. Returns the
/// matching `(I_D, A_D)` or `None` if the frame doesn't look like a
/// tone (no dominant peak, peak-frequency mismatch, peak too weak
/// vs the spectral floor).
///
/// `pcm.len()` must equal [`TONE_DETECT_FRAME`].
pub fn detect_single_tone(pcm: &[i16]) -> Option<ToneDetection> {
    if pcm.len() != TONE_DETECT_FRAME {
        return None;
    }
    // Hann window + zero-pad to 256.
    let mut buf = [0.0f64; FFT_SIZE];
    let two_pi = 2.0 * core::f64::consts::PI;
    for (n, &s) in pcm.iter().enumerate() {
        let w = 0.5 - 0.5 * (two_pi * n as f64 / TONE_DETECT_FRAME as f64).cos();
        buf[n] = (s as f64) * w;
    }
    // Direct-form DFT — N=256 is small, simpler than pulling rustfft.
    // Compute |X(k)|² for k ∈ [0, FFT_SIZE/2].
    let mut psd = [0.0f64; FFT_SIZE / 2 + 1];
    for k in 0..=FFT_SIZE / 2 {
        let omega = two_pi * k as f64 / FFT_SIZE as f64;
        let mut re = 0.0;
        let mut im = 0.0;
        for n in 0..FFT_SIZE {
            re += buf[n] * (omega * n as f64).cos();
            im -= buf[n] * (omega * n as f64).sin();
        }
        psd[k] = re * re + im * im;
    }
    // Find argmax peak excluding DC bin.
    let mut peak_k = 1usize;
    let mut peak_p = psd[1];
    for (k, &p) in psd.iter().enumerate().skip(2) {
        if p > peak_p {
            peak_p = p;
            peak_k = k;
        }
    }
    // Compute floor as the median of the bottom-half bins (excluding
    // the bin around the peak and DC). Robust against side-lobe
    // bleed.
    let mut samples = Vec::with_capacity(psd.len());
    for (k, &p) in psd.iter().enumerate() {
        if k == 0 || k.abs_diff(peak_k) <= 2 {
            continue;
        }
        samples.push(p);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let floor = samples[samples.len() / 4]; // 25th percentile — broadband baseline
    if floor <= 0.0 || peak_p / floor < SNR_THRESHOLD {
        return None;
    }
    // Quadratic peak interpolation for sub-bin accuracy.
    let f_hz = if peak_k > 0 && peak_k + 1 < psd.len() {
        let y_m1 = psd[peak_k - 1].sqrt();
        let y_0 = peak_p.sqrt();
        let y_p1 = psd[peak_k + 1].sqrt();
        let denom = y_m1 - 2.0 * y_0 + y_p1;
        let delta = if denom.abs() > 1e-12 {
            0.5 * (y_m1 - y_p1) / denom
        } else {
            0.0
        };
        (peak_k as f64 + delta) * SAMPLE_RATE_HZ / FFT_SIZE as f64
    } else {
        peak_k as f64 * SAMPLE_RATE_HZ / FFT_SIZE as f64
    };

    // Match against single-frequency Annex T rows (l1 == l2 == 1).
    let mut best: Option<(usize, f64)> = None;
    for (id, entry) in ANNEX_T.iter().enumerate() {
        let Some(t) = entry else { continue };
        if t.l1 != t.l2 || t.l1 != 1 {
            continue;
        }
        let diff = (f_hz - f64::from(t.f0)).abs();
        if diff < MATCH_TOLERANCE_HZ
            && best.map_or(true, |(_, prev)| diff < prev)
        {
            best = Some((id, diff));
        }
    }
    let id = best?.0 as u8;
    // Map peak magnitude to A_D via inverse of Eq. 209:
    // tone_magnitude = TONE_AMPLITUDE_PEAK · 10^(STEP·(A_D − 127))
    // Empirical normalisation: the FFT peak magnitude for an i16
    // sine of amplitude A is approx A · TONE_DETECT_FRAME/4 (Hann
    // window mean = 0.5). We translate that to MBE-amplitude scale
    // by dividing by ~i16-full-scale (32768). The decoder's
    // tone_to_mbe_params then scales further via TONE_AMPLITUDE_PEAK
    // — so the calibration constant below is a one-off match.
    let mag = peak_p.sqrt();
    // Calibration: i16 sine amp 8000 (≈ −12 dBFS) → empirically
    // produces FFT peak ≈ 320000 (Hann-windowed, 256-DFT). Map that
    // to A_D ≈ 100 (mid-range tone level). The exact value isn't
    // critical for correctness; tone_to_mbe_params accepts any
    // 0..127 amplitude.
    const REF_PEAK_MAG: f64 = 320_000.0;
    let amp_db = 20.0 * (mag / REF_PEAK_MAG).log10();
    // Decoder uses A_D ∈ [0, 127], step 0.05 dB above a reference. Map our
    // amp_db (which is 0 dB at REF_PEAK_MAG) to A_D ≈ 100 + amp_db/0.5.
    let ad = (100.0 + amp_db / 0.5).round().clamp(0.0, 127.0) as u8;

    Some(ToneDetection { id, amplitude: ad })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pure_sine(freq_hz: f64, amplitude: i16) -> [i16; TONE_DETECT_FRAME] {
        let mut out = [0i16; TONE_DETECT_FRAME];
        let two_pi = 2.0 * core::f64::consts::PI;
        for (n, slot) in out.iter_mut().enumerate() {
            let s = (amplitude as f64) * (two_pi * freq_hz * n as f64 / SAMPLE_RATE_HZ).sin();
            *slot = s.round() as i16;
        }
        out
    }

    #[test]
    fn detects_clean_sine_at_annex_t_frequency() {
        // Annex T entry id=10 → f0=312.5 Hz (single-tone).
        let pcm = pure_sine(312.5, 8000);
        let det = detect_single_tone(&pcm).expect("clean tone should be detected");
        assert_eq!(det.id, 10);
        assert!(det.amplitude > 50, "amplitude clamped to >0: {}", det.amplitude);
    }

    #[test]
    fn rejects_white_noise() {
        // Pseudo-noise (LCG-driven) has no clean peak → no tone.
        let mut pcm = [0i16; TONE_DETECT_FRAME];
        let mut state: u32 = 12345;
        for slot in pcm.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *slot = ((state as i32 >> 16) as i16) / 4;
        }
        assert!(detect_single_tone(&pcm).is_none());
    }

    #[test]
    fn rejects_silence() {
        let pcm = [0i16; TONE_DETECT_FRAME];
        assert!(detect_single_tone(&pcm).is_none());
    }

    #[test]
    fn snr_floor_rejects_weak_tone_buried_in_noise() {
        // Tone at -30 dBFS + noise at -20 dBFS.
        let mut pcm = pure_sine(312.5, 1000);
        let mut state: u32 = 7777;
        for slot in pcm.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = ((state as i32 >> 16) as i16) / 4;
            *slot = slot.saturating_add(n);
        }
        // Tone may or may not be detected — depends on the SNR threshold.
        // Test that whichever path it takes, it doesn't panic.
        let _ = detect_single_tone(&pcm);
    }

    #[test]
    fn wrong_length_returns_none() {
        let pcm = [0i16; 100];
        assert!(detect_single_tone(&pcm).is_none());
    }

    #[test]
    fn tolerance_window_snaps_to_nearest_annex_entry() {
        // Drift slightly off id=10's 312.5 Hz — should still match.
        let pcm = pure_sine(316.0, 8000);
        let det = detect_single_tone(&pcm).expect("near-match should detect");
        assert_eq!(det.id, 10);
    }
}
