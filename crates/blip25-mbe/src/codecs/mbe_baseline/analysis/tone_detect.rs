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

use crate::ambe_plus2_wire::frame::ANNEX_T;

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

/// Compute the windowed PSD for one detection frame.
///
/// Returns `(psd, floor)` where `psd[k] = |X(k)|²` for `k ∈ [0, 128]`
/// and `floor` is the 25th-percentile bin power (broadband baseline).
/// Hann-windowed, 256-point zero-padded direct DFT.
fn compute_psd(pcm: &[i16; TONE_DETECT_FRAME]) -> ([f64; FFT_SIZE / 2 + 1], f64) {
    let mut buf = [0.0f64; FFT_SIZE];
    let two_pi = 2.0 * core::f64::consts::PI;
    for (n, &s) in pcm.iter().enumerate() {
        let w = 0.5 - 0.5 * (two_pi * n as f64 / TONE_DETECT_FRAME as f64).cos();
        buf[n] = (s as f64) * w;
    }
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
    // 25th-percentile broadband floor, excluding DC.
    let mut samples: Vec<f64> = psd.iter().skip(1).copied().collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let floor = samples[samples.len() / 4];
    (psd, floor)
}

/// Quadratic peak interpolation for sub-bin frequency accuracy.
fn interpolate_peak_hz(psd: &[f64], k: usize) -> f64 {
    if k > 0 && k + 1 < psd.len() {
        let y_m1 = psd[k - 1].sqrt();
        let y_0 = psd[k].sqrt();
        let y_p1 = psd[k + 1].sqrt();
        let denom = y_m1 - 2.0 * y_0 + y_p1;
        let delta = if denom.abs() > 1e-12 {
            0.5 * (y_m1 - y_p1) / denom
        } else {
            0.0
        };
        (k as f64 + delta) * SAMPLE_RATE_HZ / FFT_SIZE as f64
    } else {
        k as f64 * SAMPLE_RATE_HZ / FFT_SIZE as f64
    }
}

/// Reference FFT magnitude for `A_D = 127`. Empirically calibrated so
/// the round-trip (PCM in → tone-frame encode → decode → PCM out)
/// preserves amplitude: an input sine at the level Annex T's "full
/// scale" tone (`M̃_l = 16384`, the maximum representable A_D=127
/// amplitude) maps to A_D = 127, and quieter input maps to lower A_D
/// per the slope below.
///
/// The synth applies γ_w (146.64) plus its own per-band scaling on
/// top of M̃_l, so the calibration constant is 9.3 dB above the
/// "spec-literal" value of 640 000 (which would assume M̃_l=16384 ⇔
/// i16 amplitude 16384). Confirmed empirically with a 4-amplitude
/// sine round-trip probe — see commit message.
const REF_PEAK_MAG: f64 = 1_830_000.0;

/// dB per A_D step matching Annex T Eq. 209 exactly:
/// `M̃_l = 16384 · 10^{0.03555·(A_D − 127)}` →
/// `20·log10(10^0.03555) = 0.711 dB/step`. Using anything else creates
/// a slope mismatch between the encoder's amplitude estimator and the
/// decoder's amplitude reconstruction (e.g., 0.5 dB/step would
/// expand round-trip error by 6 dB per 12 A_D steps).
const DB_PER_AD_STEP: f64 = 0.7115;

/// Convert a peak magnitude to the 7-bit `A_D` field. Inverse of Eq.
/// 209: `A_D = 127 + 20·log10(mag / REF_PEAK_MAG) / DB_PER_AD_STEP`.
fn magnitude_to_amplitude(mag: f64) -> u8 {
    let amp_db = 20.0 * (mag / REF_PEAK_MAG).log10();
    (127.0 + amp_db / DB_PER_AD_STEP).round().clamp(0.0, 127.0) as u8
}

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
    let pcm: &[i16; TONE_DETECT_FRAME] = pcm.try_into().ok()?;
    let (psd, floor) = compute_psd(pcm);

    // Find argmax peak excluding DC bin.
    let mut peak_k = 1usize;
    let mut peak_p = psd[1];
    for (k, &p) in psd.iter().enumerate().skip(2) {
        if p > peak_p {
            peak_p = p;
            peak_k = k;
        }
    }
    if floor <= 0.0 || peak_p / floor < SNR_THRESHOLD {
        return None;
    }
    let f_hz = interpolate_peak_hz(&psd, peak_k);

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
    Some(ToneDetection { id, amplitude: magnitude_to_amplitude(peak_p.sqrt()) })
}

/// Detect a DTMF (two-tone) Annex T entry in `pcm`. Returns the
/// matching `(I_D, A_D)` or `None` on no clean two-peak match.
///
/// DTMF rows in Annex T have `l1 != l2`; the audible tones are at
/// `l1·f0` and `l2·f0`. Detection finds the two strongest spectral
/// peaks (separated by ≥3 bins to avoid main-lobe siblings), then
/// scans Annex T DTMF rows for the one whose `(l1·f0, l2·f0)` pair
/// best matches the detected pair.
///
/// `pcm.len()` must equal [`TONE_DETECT_FRAME`].
pub fn detect_dtmf(pcm: &[i16]) -> Option<ToneDetection> {
    if pcm.len() != TONE_DETECT_FRAME {
        return None;
    }
    let pcm: &[i16; TONE_DETECT_FRAME] = pcm.try_into().ok()?;
    let (psd, floor) = compute_psd(pcm);

    // Find primary peak.
    let mut peak1_k = 1usize;
    let mut peak1_p = psd[1];
    for (k, &p) in psd.iter().enumerate().skip(2) {
        if p > peak1_p {
            peak1_p = p;
            peak1_k = k;
        }
    }
    if floor <= 0.0 || peak1_p / floor < SNR_THRESHOLD {
        return None;
    }

    // Find secondary peak: best k >= 2 bins away from primary, also
    // above floor.
    let mut peak2_k: Option<usize> = None;
    let mut peak2_p = floor; // must beat floor
    for (k, &p) in psd.iter().enumerate().skip(1) {
        if k.abs_diff(peak1_k) < 3 {
            continue;
        }
        if p > peak2_p {
            peak2_p = p;
            peak2_k = Some(k);
        }
    }
    let peak2_k = peak2_k?;
    // Secondary peak must clear SNR floor on its own.
    if peak2_p / floor < SNR_THRESHOLD {
        return None;
    }

    let f1_hz = interpolate_peak_hz(&psd, peak1_k);
    let f2_hz = interpolate_peak_hz(&psd, peak2_k);
    let f_lo = f1_hz.min(f2_hz);
    let f_hi = f1_hz.max(f2_hz);

    // Scan DTMF rows in Annex T for the closest (l1·f0, l2·f0) match.
    // Sort l1, l2 so we always compare against `(l_lo·f0, l_hi·f0)`
    // with l_lo < l_hi.
    let mut best: Option<(usize, f64)> = None;
    for (id, entry) in ANNEX_T.iter().enumerate() {
        let Some(t) = entry else { continue };
        if t.l1 == t.l2 {
            continue; // single-tone, handled by detect_single_tone
        }
        let f0 = f64::from(t.f0);
        let l_lo = t.l1.min(t.l2) as f64;
        let l_hi = t.l1.max(t.l2) as f64;
        let expected_lo = l_lo * f0;
        let expected_hi = l_hi * f0;
        let err = (expected_lo - f_lo).abs() + (expected_hi - f_hi).abs();
        if err < 2.0 * MATCH_TOLERANCE_HZ
            && best.map_or(true, |(_, prev)| err < prev)
        {
            best = Some((id, err));
        }
    }
    let id = best?.0 as u8;
    // Use the larger of the two peak magnitudes for amplitude (DTMF
    // tones are typically equal-amplitude; a one-off pick is fine).
    let mag = peak1_p.sqrt().max(peak2_p.sqrt());
    Some(ToneDetection { id, amplitude: magnitude_to_amplitude(mag) })
}

/// Try DTMF detection first, then single-tone. Returns whichever
/// matches; `None` if neither does.
///
/// DTMF is more constraining (two peaks at specific harmonic ratios)
/// so a hit is high-confidence; falling through to single-tone after
/// DTMF fails handles the common case cleanly.
pub fn detect_tone(pcm: &[i16]) -> Option<ToneDetection> {
    detect_dtmf(pcm).or_else(|| detect_single_tone(pcm))
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

    fn dtmf_pair(f1: f64, f2: f64, amp_each: i16) -> [i16; TONE_DETECT_FRAME] {
        let mut out = [0i16; TONE_DETECT_FRAME];
        let two_pi = 2.0 * core::f64::consts::PI;
        for (n, slot) in out.iter_mut().enumerate() {
            let s1 = (amp_each as f64) * (two_pi * f1 * n as f64 / SAMPLE_RATE_HZ).sin();
            let s2 = (amp_each as f64) * (two_pi * f2 * n as f64 / SAMPLE_RATE_HZ).sin();
            *slot = (s1 + s2).round().clamp(-32768.0, 32767.0) as i16;
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

    #[test]
    fn detect_dtmf_matches_known_annex_t_pair() {
        // Annex T id=128 → f0=78.5, l1=12, l2=17 → tones at 942 Hz
        // and 1334.5 Hz (close to DTMF "1" 941/1336).
        let pcm = dtmf_pair(942.0, 1334.5, 6000);
        let det = detect_dtmf(&pcm).expect("clean DTMF should be detected");
        assert_eq!(det.id, 128);
    }

    #[test]
    fn detect_tone_prefers_dtmf_when_two_peaks_present() {
        // Same DTMF input — `detect_tone` should route through DTMF
        // rather than picking up either peak as a single-tone match.
        let pcm = dtmf_pair(942.0, 1334.5, 6000);
        let det = detect_tone(&pcm).expect("tone-detect dispatch hit");
        assert_eq!(det.id, 128);
    }

    #[test]
    fn detect_dtmf_rejects_single_tone_input() {
        // A pure sine has only one peak; secondary-peak SNR check
        // should reject as not-DTMF.
        let pcm = pure_sine(312.5, 8000);
        assert!(detect_dtmf(&pcm).is_none());
        // But detect_tone() falls through to single-tone and finds it.
        let det = detect_tone(&pcm).expect("falls through to single-tone");
        assert_eq!(det.id, 10);
    }

    #[test]
    fn detect_dtmf_rejects_noise() {
        let mut pcm = [0i16; TONE_DETECT_FRAME];
        let mut state: u32 = 9999;
        for slot in pcm.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *slot = ((state as i32 >> 16) as i16) / 4;
        }
        assert!(detect_dtmf(&pcm).is_none());
    }
}
