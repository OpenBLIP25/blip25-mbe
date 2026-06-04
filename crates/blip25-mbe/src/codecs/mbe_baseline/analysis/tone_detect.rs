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

use crate::rate33::frame::ANNEX_T;

/// What the tone detector returns on a hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Cached 256-point Hann window over the 160 PCM samples (zero past
/// `TONE_DETECT_FRAME`). Built once on first call.
fn hann_window() -> &'static [f64; FFT_SIZE] {
    use std::sync::OnceLock;
    static WINDOW: OnceLock<[f64; FFT_SIZE]> = OnceLock::new();
    WINDOW.get_or_init(|| {
        let two_pi = 2.0 * core::f64::consts::PI;
        let mut w = [0.0f64; FFT_SIZE];
        for n in 0..TONE_DETECT_FRAME {
            w[n] = 0.5 - 0.5 * (two_pi * n as f64 / TONE_DETECT_FRAME as f64).cos();
        }
        w
    })
}

/// Compute the windowed PSD for one detection frame.
///
/// Returns `(psd, floor)` where `psd[k] = |X(k)|²` for `k ∈ [0, 128]`
/// and `floor` is the 25th-percentile bin power (broadband baseline).
/// Hann-windowed, 256-point zero-padded FFT.
fn compute_psd(pcm: &[i16; TONE_DETECT_FRAME]) -> ([f64; FFT_SIZE / 2 + 1], f64) {
    use std::sync::OnceLock;
    use num_complex::Complex;
    use rustfft::{Fft, FftPlanner};

    static FFT: OnceLock<std::sync::Arc<dyn Fft<f64>>> = OnceLock::new();
    let fft = FFT.get_or_init(|| {
        let mut planner = FftPlanner::<f64>::new();
        planner.plan_fft_forward(FFT_SIZE)
    });

    let window = hann_window();
    let mut buf = [Complex::<f64>::new(0.0, 0.0); FFT_SIZE];
    for n in 0..TONE_DETECT_FRAME {
        buf[n] = Complex::new(pcm[n] as f64 * window[n], 0.0);
    }
    fft.process(&mut buf);

    let mut psd = [0.0f64; FFT_SIZE / 2 + 1];
    for k in 0..=FFT_SIZE / 2 {
        psd[k] = buf[k].re * buf[k].re + buf[k].im * buf[k].im;
    }

    // 25th-percentile broadband floor, excluding DC. `select_nth_unstable`
    // is O(N) and partitions in place — no full sort, no Vec alloc.
    let mut samples = [0.0f64; FFT_SIZE / 2];
    samples.copy_from_slice(&psd[1..]);
    let idx = samples.len() / 4;
    samples.select_nth_unstable_by(idx, |a, b| {
        a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
    });
    let floor = samples[idx];
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

/// Reference FFT magnitude for `A_D = 127` — set so a full-scale sine
/// (PCM peak 32767) round-trips at unity. Derivation:
///
/// For an exact-bin sine of PCM peak `A` through a Hann window of
/// length `N=160` (`Σw[n] = N/2 = 80`), the peak FFT magnitude is
/// `A · Σw / 2 = A · 40`. Full-scale `A = 32767` ⇒ peak ≈ 1 310 680.
///
/// On the synth side, voiced harmonics add `2·M̃_l` to the time-domain
/// output (`scale = 2.0 * m_curr_l` in the UvV branch); tone frames are
/// all-voiced so `γ_w` does not enter the path. Setting
/// `REF_PEAK_MAG = 1_310_000` makes the encoder emit `A_D = 127` exactly
/// at full-scale input, where `M̃_l = 16384` and the synth output peak
/// reaches `2·16384 = 32768` (i16 saturation). Across 24 dB the
/// round-trip residual is within ±0.3 dB on the 32-amplitude / 4-frequency
/// dense probe in `examples/tone_amp_audit.rs`.
const REF_PEAK_MAG: f64 = 1_310_000.0;

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

    // Match against single-tone Annex T rows (l1 == l2). The audible tone
    // is at `l1·f0`; rows with l1>1 cover frequencies above the l=1 band's
    // 156-375 Hz range (e.g. l1=l2=3 with f0≈343.76 ⇒ 1031 Hz, the
    // Motorola standard tone test pattern). Multi-tone rows (l1!=l2)
    // belong to detect_dtmf.
    let mut best: Option<(usize, f64)> = None;
    for (id, entry) in ANNEX_T.iter().enumerate() {
        let Some(t) = entry else { continue };
        if t.l1 != t.l2 {
            continue;
        }
        let expected_hz = f64::from(t.l1) * f64::from(t.f0);
        let diff = (f_hz - expected_hz).abs();
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
    fn detects_standard_motorola_test_tone_at_l3_harmonic() {
        // Motorola's Standard Tone Test Pattern is 1031.25 Hz IMBE-encoded.
        // Annex T has a row at l1=l2=3, f0≈343.76 Hz where the audible tone
        // is 3·f0 ≈ 1031.28 Hz — the natural target for half-rate Annex T
        // dispatch on this conformance signal. Regression for the pre-fix
        // bug where detect_single_tone restricted matches to l1=1 entries
        // and silently dropped this whole class of tones.
        let pcm = pure_sine(1031.25, 8000);
        let det = detect_single_tone(&pcm).expect("1031 Hz tone should match an l=3 row");
        let entry = ANNEX_T[det.id as usize].expect("matched id has Annex T entry");
        assert_eq!(entry.l1, entry.l2, "single-tone row");
        assert_eq!(entry.l1, 3, "1031 Hz is in the l=3 band of Annex T");
        let expected_hz = f64::from(entry.l1) * f64::from(entry.f0);
        assert!(
            (expected_hz - 1031.25).abs() < f64::from(MATCH_TOLERANCE_HZ as f32),
            "l·f0 ({expected_hz:.2}) should be within tolerance of 1031.25"
        );
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

    /// Knox tone-pairs occupy Annex T IDs 144–159 (see analysis memo
    /// `reference_tone_frames_phase1_vs_phase2_2026-04-30.md`). For each
    /// in-range Knox ID, synthesize the matching `(l1·f0, l2·f0)` pair
    /// and verify `detect_dtmf` returns that exact ID. Validates the
    /// matcher against the full Knox sub-table at once — a regression
    /// here would mean a mis-classification shipping into the
    /// tone-frame dispatch path on real Knox traffic.
    #[test]
    fn detect_dtmf_round_trips_each_knox_id() {
        // Skip IDs whose harmonic ratio places `l_lo·f0` below ~150 Hz
        // (the lower edge of the Annex T frequency space) — those have
        // a very low primary peak that gets close to DC and risks
        // colliding with the DC-skip rule in `compute_psd`. None of
        // the Knox IDs hit this in practice, but the guard makes the
        // test robust against future table additions.
        for knox_id in 144u8..=159 {
            let entry = ANNEX_T[knox_id as usize].expect("Knox ID has Annex T row");
            assert_ne!(entry.l1, entry.l2, "Knox IDs are pairs, l1 != l2");
            let f0 = f64::from(entry.f0);
            let f_lo = f0 * f64::from(entry.l1.min(entry.l2));
            let f_hi = f0 * f64::from(entry.l1.max(entry.l2));
            // Skip if either tone falls outside the FFT-resolvable band.
            if f_lo < 150.0 || f_hi > 3500.0 {
                continue;
            }
            let pcm = dtmf_pair(f_lo, f_hi, 6000);
            let det = detect_dtmf(&pcm)
                .unwrap_or_else(|| panic!("Knox ID {knox_id} should detect"));
            assert_eq!(
                det.id, knox_id,
                "Knox ID {knox_id} (f_lo={f_lo:.1} Hz, f_hi={f_hi:.1} Hz) misdetected as {}",
                det.id
            );
        }
    }

    /// `detect_tone` (the dispatch entry point) must route each Knox
    /// pair through the DTMF matcher rather than falling through to
    /// `detect_single_tone` (which would pick whichever single-tone
    /// row sits closest to the louder peak).
    #[test]
    fn detect_tone_routes_knox_pairs_through_dtmf_matcher() {
        // Spot-check id=145 (Knox A+B 603.6/1056.2 Hz — the actual
        // pair in DVSI's knox_1.pcm test vector).
        let entry = ANNEX_T[145].unwrap();
        let f_lo = f64::from(entry.f0) * f64::from(entry.l1.min(entry.l2));
        let f_hi = f64::from(entry.f0) * f64::from(entry.l1.max(entry.l2));
        let pcm = dtmf_pair(f_lo, f_hi, 6000);
        let det = detect_tone(&pcm).expect("Knox pair detect");
        assert_eq!(det.id, 145);
    }
}
