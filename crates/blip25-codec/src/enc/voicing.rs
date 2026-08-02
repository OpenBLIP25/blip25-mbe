//! Half-rate AMBE+2 VOICING decision (V/UV per band -> `b1`).
//!
//! Derived from the published MBE/IMBE algorithm (TIA-102.BABA-A §2.7 and the
//! original Multi-Band Excitation V/UV decision, Griffin & Lim 1988). This
//! is the analysis-side inverse of [`crate::dequantize::expand_vuv`]: the
//! decoder takes a 5-bit codebook index `b1`, expands it through the 32-row
//! Annex-M V/UV codebook into a per-harmonic voiced mask. The encoder must
//! make a per-harmonic V/UV decision and then collapse it onto the single best
//! of the 32 codebook rows.
//!
//! ## Algorithm overview
//!
//! 1. **Per-harmonic voiced-comb fit error.** For each harmonic `l` we model
//!    the band `[(l-½)ω₀, (l+½)ω₀]` as a single voiced spectral tooth: a Hann
//!    main-lobe `t(ω)` centred on `lω₀` spanning exactly one harmonic spacing.
//!    The least-squares amplitude `â = Σ|S|t / Σt²` is fit and the *normalized
//!    residual* `D_l = Σ(|S| - â·t)² / Σ|S|²` is taken as the voicing measure.
//!    `D_l → 0` for a clean harmonic (voiced); `D_l → 1` for spread/noisy
//!    energy (unvoiced). This is the magnitude-domain MBE V/UV functional.
//!
//! 2. **Loudness-graded threshold.** A harmonic is declared voiced when
//!    `D_l < θ(l)`. The threshold is *graded by frame loudness* (louder frames
//!    tolerate larger residuals, matching the reference's loudness-graded Θ), tilted
//!    down with frequency (high bands need stronger evidence), and nudged by
//!    the time-domain **crest factor** (the reference's V/UV is phase-sensitive — an
//!    impulsive / high-crest frame is biased toward voiced).
//!
//! 3. **Onset voicing-extent ramp.** Quiet / onset frames only allow the lower
//!    harmonics to go voiced; the maximum voiced harmonic index ramps up with
//!    loudness (and crest). Harmonics above the extent are parked unvoiced.
//!
//! 4. **Silence rule.** Below an energy floor the whole frame is forced
//!    unvoiced; the extent ramp additionally parks the *high* bands unvoiced as
//!    energy falls (the reference parks high bands near zero in near-silence).
//!
//! 5. **Collapse to `b1`.** The per-harmonic mask is matched against all 32
//!    Annex-M codebook rows expanded through the *same* band->row mapping that
//!    [`crate::dequantize::expand_vuv`] uses (`j = ⌊l·16·ω₀/2π⌋`, clamped 0..7),
//!    and the row with the smallest Hamming distance is chosen (ties -> lowest
//!    index, which keeps the canonical all-voiced row 0 / all-unvoiced row 16).
//!
//! ## Inputs / conventions
//! * `spectrum`: magnitude spectrum of the (centred, windowed) analysis frame,
//!   length `n`, covering `[0, 2π)` so that bin `k` ↔ `ω = 2π·k/n`. This matches
//!   the 256-pt DFT convention used by [`crate::synth`].
//! * `omega_0`: fundamental in rad/sample.
//! * `frame_energy`: mean-square level of the (unwindowed) time-domain frame.
//! * `crest`: time-domain crest factor `peak / rms` of the frame.
//!
//! ## Assumptions / caveats
//! * The §0.4 E_R pitch refinement is out of scope here; the supplied
//!   `omega_0` is taken as given.
//! * Thresholds in [`VoicingConfig`] are derived from the published algorithm's
//!   shape but are tuning constants; fixed-point + reference-anchor calibration are
//!   left to integration.

use core::f64::consts::PI;

use crate::tables::AMBE_VUV_CODEBOOK;

/// Number of frequency bands in the Annex-M V/UV codebook.
pub(crate) const VUV_BANDS: usize = 8;
/// Number of rows in the Annex-M V/UV codebook.
pub(crate) const VUV_ROWS: usize = 32;

/// Tunable constants for the voicing decision. [`VoicingConfig::default`]
/// follows the published MBE shape; integration may re-calibrate against the
/// reference anchors.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VoicingConfig {
    /// Base normalized-residual voicing threshold (mid loudness, low band).
    pub base_thr: f64,
    /// Threshold multiplier at/below `quiet_db` (stricter -> less voiced).
    pub thr_quiet: f64,
    /// Threshold multiplier at/above `loud_db` (looser -> more voiced).
    pub thr_loud: f64,
    /// Frame level (dB, 10·log10 mean-square) mapped to `thr_quiet`.
    pub quiet_db: f64,
    /// Frame level (dB) mapped to `thr_loud`.
    pub loud_db: f64,
    /// Down-tilt of the threshold across frequency (0 = flat, 1 = high band
    /// gets zero threshold). High bands require stronger voiced evidence.
    pub freq_tilt: f64,
    /// Crest factor that maps to a neutral (×1) threshold.
    pub crest_ref: f64,
    /// Lower clamp on the crest threshold multiplier.
    pub crest_min: f64,
    /// Upper clamp on the crest threshold multiplier.
    pub crest_max: f64,
    /// Mean-square level below which the whole frame is forced unvoiced.
    pub silence_floor: f64,
    /// Minimum voiced extent (fraction of L) even for the quietest non-silent
    /// frame — guarantees the onset ramp starts above zero.
    pub min_extent: f64,
    /// Extra voiced extent contributed by a high crest factor (fraction of L).
    pub crest_extent_gain: f64,
    /// Fricative protection (IMBE-only). When set, the bimodal snap-to-all-
    /// voiced is withheld on frames whose TOP-third harmonics are mostly
    /// unvoiced — the fricative signature (voiced low band + unvoiced high SH/T
    /// noise). Without it the snap voices the high fricative energy, which the
    /// ear hears as buzz. The AMBE+2 path leaves this `false` (default).
    pub fricative_protect: bool,
    /// Voiced fraction the top-third harmonics must reach for the upward snap to
    /// be allowed when `fricative_protect` is set. Below it the frame keeps its
    /// partial (low-V / high-UV) per-band pattern.
    pub fricative_hi_frac: f64,
    /// Per-3-harmonic-band V/UV decision (IMBE-only). When set, the per-harmonic
    /// voiced mask is REPLACED by the standard MBE per-band voicing error
    /// [`band_voicing_errors`] evaluated over each 3-harmonic band (the true
    /// window-transform synthetic): every harmonic of a band shares one V/UV
    /// decision. This matches the resolution at which IMBE actually decides
    /// voicing (one bit per K=⌊(L+2)/3⌋ band) and separates sustained fricative
    /// (SH/T) bands — flat noise across a 3-harmonic span — from voiced bands
    /// (three sharp comb teeth). The AMBE+2 path leaves this `false` (default).
    pub band_voicing: bool,
    /// Band voicing error above which a 3-harmonic band is declared unvoiced,
    /// when `band_voicing` is set.
    pub band_thr: f64,
}

impl Default for VoicingConfig {
    fn default() -> Self {
        Self {
            // `base_thr` is the normalized-residual ceiling for calling a
            // harmonic voiced: RAISING it voices more harmonics, lowering it
            // voices fewer. `freq_tilt` runs the OPPOSITE way: it scales the
            // threshold *down* as frequency rises, so raising it voices FEWER
            // high harmonics (0 = flat, one threshold across the whole band).
            // Both are fitted against the reference bits, NOT by ear -- a change
            // that sounds better here can still lose bit-match. The audible
            // failure modes sit on both sides: too low over-suppresses high
            // bands on loud voiced onsets and adds breathiness to /w/; too high
            // buzzes fricatives. Paired with the bimodal snap below.
            base_thr: 0.48,
            thr_quiet: 0.45,
            thr_loud: 1.6,
            quiet_db: 20.0,
            loud_db: 70.0,
            freq_tilt: 0.25,
            crest_ref: 4.0,
            crest_min: 0.7,
            crest_max: 1.6,
            silence_floor: 4.0,
            min_extent: 0.15,
            crest_extent_gain: 0.25,
            fricative_protect: false,
            fricative_hi_frac: 0.5,
            band_voicing: false,
            band_thr: 0.5,
        }
    }
}

/// Per-harmonic voicing analysis result (before collapse to `b1`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VoicingDecision {
    /// Per-harmonic voiced mask, length `L`.
    pub per_band: Vec<bool>,
    /// Raw per-harmonic normalized fit error `D_l`, length `L` (diagnostics).
    pub fit_error: Vec<f64>,
    /// Chosen 5-bit Annex-M codebook index.
    pub b1: u8,
}

/// Band index `j` for harmonic `l_h` (1-based) — the exact mapping that
/// [`crate::dequantize::expand_vuv`] inverts.
#[inline]
fn band_index(l_h: usize, omega_0: f64) -> usize {
    let j = (l_h as f64 * 16.0 * omega_0 / (2.0 * PI)).floor() as i32;
    j.clamp(0, (VUV_BANDS - 1) as i32) as usize
}

/// Expand a codebook row index to a per-harmonic mask (encoder-side mirror of
/// [`crate::dequantize::expand_vuv`], without the `[bool; L_MAX]` padding).
fn expand_row(b1: usize, omega_0: f64, l: usize) -> Vec<bool> {
    let row = &AMBE_VUV_CODEBOOK[b1];
    (1..=l).map(|l_h| row[band_index(l_h, omega_0)]).collect()
}

/// Collapse a per-harmonic voiced mask onto the best of the 32 Annex-M rows,
/// inverting [`crate::dequantize::expand_vuv`]. Ties break to the lowest index.
pub(crate) fn collapse_to_b1(per_band: &[bool], omega_0: f64) -> u8 {
    let l = per_band.len();
    let mut best_idx = 0usize;
    let mut best_dist = usize::MAX;
    for b1 in 0..VUV_ROWS {
        let cand = expand_row(b1, omega_0, l);
        let dist = cand
            .iter()
            .zip(per_band.iter())
            .filter(|(a, b)| a != b)
            .count();
        if dist < best_dist {
            best_dist = dist;
            best_idx = b1;
            if dist == 0 {
                break;
            }
        }
    }
    best_idx as u8
}

/// Compute the normalized voiced-comb fit error `D_l` for one harmonic.
///
/// Returns `None` when the band carries no resolvable energy (degenerate band
/// or harmonic beyond Nyquist) — such bands are treated as unvoiced.
fn band_fit_error(spectrum: &[f64], omega_0: f64, l_h: usize) -> Option<f64> {
    let n = spectrum.len();
    if n < 2 {
        return None;
    }
    let omega_l = l_h as f64 * omega_0;
    if omega_l >= PI {
        return None; // harmonic at/above Nyquist
    }
    let half = omega_0 / 2.0;
    let w_lo = (omega_l - half).max(0.0);
    let w_hi = (omega_l + half).min(PI);

    // bin k <-> omega = 2*pi*k/n
    let to_bin = |w: f64| -> f64 { w * n as f64 / (2.0 * PI) };
    let k_lo = to_bin(w_lo).ceil() as i64;
    let k_hi = to_bin(w_hi).floor() as i64;
    if k_hi < k_lo {
        return None;
    }

    let mut s_t = 0.0; // Σ |S| t
    let mut t_t = 0.0; // Σ t²
    let mut s_s = 0.0; // Σ |S|²
    for k in k_lo..=k_hi {
        if k < 0 || k as usize >= n {
            continue;
        }
        let w = 2.0 * PI * k as f64 / n as f64;
        // Hann main-lobe centred on omega_l, zero at the band edges.
        let phase = PI * (w - omega_l) / half; // ±π at edges
        let t = 0.5 * (1.0 + phase.cos());
        let s = spectrum[k as usize].abs();
        s_t += s * t;
        t_t += t * t;
        s_s += s * s;
    }
    if s_s <= 0.0 || t_t <= 0.0 {
        return None;
    }
    let a_hat = s_t / t_t; // least-squares tooth amplitude
                           // residual = Σ(S - a t)² = Σ S² - a²·Σ t²  (a is the LS optimum)
    let residual = (s_s - a_hat * a_hat * t_t).max(0.0);
    Some((residual / s_s).clamp(0.0, 1.0))
}

/// Magnitude of the analysis-window frequency response `|W(θ)|`, where
/// `W(θ) = Σ_i wA(i)·e^{-jθ(i-WIN_HALF)}` over the firmware 250-tap reference window.
///
/// This is the shape a single windowed harmonic tooth casts onto the centred
/// 256-pt DFT (`enc::analysis::centered_rdft`): a real cosine at frequency `Ω`
/// appears as `≈½·|W(ω_k − Ω)|`. It is the *true* MBE voiced synthetic — a
/// narrow main lobe (~2 bins) plus the window's sidelobes — unlike the wide
/// one-harmonic-spacing Hann lobe used by [`band_fit_error`]. The narrowness is
/// what lets a 3-harmonic band tell three sharp voiced teeth from flat fricative
/// noise. `|W|` is even in `θ`; a dense table over `[0, π]` is interpolated.
fn window_kernel(theta: f64) -> f64 {
    use std::sync::OnceLock;
    use super::analysis::{window_coeff, WIN_HALF, WIN_LEN};
    const TAB_N: usize = 1 << 15; // samples over [0, π]
    static TAB: OnceLock<Box<[f64]>> = OnceLock::new();
    let tab = TAB.get_or_init(|| {
        let mut t = vec![0.0f64; TAB_N + 1];
        for (j, slot) in t.iter_mut().enumerate() {
            let th = PI * j as f64 / TAB_N as f64;
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for i in 0..WIN_LEN {
                let n = i as f64 - WIN_HALF as f64;
                let a = -th * n;
                let w = window_coeff(i);
                re += w * a.cos();
                im += w * a.sin();
            }
            *slot = (re * re + im * im).sqrt();
        }
        t.into_boxed_slice()
    });
    let t = theta.abs();
    if t >= PI {
        return tab[TAB_N];
    }
    let x = t / PI * TAB_N as f64;
    let j = x.floor() as usize;
    let frac = x - j as f64;
    tab[j] * (1.0 - frac) + tab[j + 1] * frac
}

/// Standard IMBE/MBE per-3-harmonic-band voicing error over the magnitude
/// spectrum, using the analysis-window transform ([`window_kernel`]) as the
/// voiced synthetic.
///
/// Bands group harmonics in threes exactly as the IMBE V/UV word does:
/// band `b` (0-based) spans harmonics `3b+1 ..= 3b+3` (1-based), `K = ⌈L/3⌉`
/// bands. For each harmonic `l` in the band the least-squares tooth amplitude
/// `â_l = Σ|S|·E / ΣE²` is fit over its `[(l−½)ω₀,(l+½)ω₀]` sub-band (`E(k) =
/// |W(ω_k − lω₀)|`), and the band error is the energy-aggregated residual
///
/// ```text
///   ε_b = Σ_l Σ_k |S − â_l·E|²  /  Σ_l Σ_k |S|²
/// ```
///
/// summed over the band's harmonics. A voiced band (sharp teeth) drives `ε_b →
/// 0`; a fricative band (flat noise the narrow teeth cannot explain) drives it
/// toward the noise floor. Returns one error per band, length `K`.
pub(crate) fn band_voicing_errors(spectrum: &[f64], omega_0: f64, l: usize) -> Vec<f64> {
    let n = spectrum.len();
    let k_bands = l.div_ceil(3).max(1);
    let mut errs = vec![1.0f64; k_bands];
    if n < 2 || omega_0 <= 0.0 || l == 0 {
        return errs;
    }
    let to_bin = |w: f64| -> f64 { w * n as f64 / (2.0 * PI) };
    for (b, err) in errs.iter_mut().enumerate() {
        let l_first = 3 * b + 1;
        let l_last = (3 * b + 3).min(l);
        let mut num = 0.0f64; // Σ residual over the band
        let mut den = 0.0f64; // Σ |S|² over the band
        for l_h in l_first..=l_last {
            let omega_l = l_h as f64 * omega_0;
            if omega_l >= PI {
                continue;
            }
            let half = omega_0 / 2.0;
            let w_lo = (omega_l - half).max(0.0);
            let w_hi = (omega_l + half).min(PI);
            let k_lo = to_bin(w_lo).ceil() as i64;
            let k_hi = to_bin(w_hi).floor() as i64;
            if k_hi < k_lo {
                continue;
            }
            let mut s_e = 0.0f64; // Σ |S|·E
            let mut e_e = 0.0f64; // Σ E²
            let mut s_s = 0.0f64; // Σ |S|²
            for k in k_lo..=k_hi {
                if k < 0 || k as usize >= n {
                    continue;
                }
                let w = 2.0 * PI * k as f64 / n as f64;
                let ek = window_kernel(w - omega_l);
                let s = spectrum[k as usize].abs();
                s_e += s * ek;
                e_e += ek * ek;
                s_s += s * s;
            }
            if e_e <= 0.0 {
                num += s_s;
                den += s_s;
                continue;
            }
            let a = s_e / e_e; // least-squares tooth amplitude
            let resid = (s_s - a * a * e_e).max(0.0);
            num += resid;
            den += s_s;
        }
        if den > 0.0 {
            *err = (num / den).clamp(0.0, 1.0);
        }
    }
    errs
}

/// Decide per-harmonic V/UV and the collapsed `b1` index using the default
/// [`VoicingConfig`].
///
/// * `spectrum` — magnitude spectrum over `[0, 2π)`, bin `k` ↔ `ω = 2π·k/len`.
/// * `omega_0` — fundamental (rad/sample).
/// * `l` — number of harmonics `L`.
/// * `frame_energy` — mean-square time-domain level of the frame.
/// * `crest` — time-domain crest factor (`peak / rms`).
///
/// Returns `(per_band, b1)` where `per_band` has length `l`.
pub(crate) fn decide_voicing(
    spectrum: &[f64],
    omega_0: f64,
    l: usize,
    frame_energy: f64,
    crest: f64,
) -> (Vec<bool>, u8) {
    let d = decide_voicing_cfg(
        spectrum,
        omega_0,
        l,
        frame_energy,
        crest,
        &VoicingConfig::default(),
    );
    (d.per_band, d.b1)
}

/// Full voicing decision with explicit configuration and diagnostics.
pub(crate) fn decide_voicing_cfg(
    spectrum: &[f64],
    omega_0: f64,
    l: usize,
    frame_energy: f64,
    crest: f64,
    cfg: &VoicingConfig,
) -> VoicingDecision {
    let mut per_band = vec![false; l];
    let mut fit_error = vec![1.0; l];

    if l == 0 {
        let b1 = collapse_to_b1(&per_band, omega_0);
        return VoicingDecision {
            per_band,
            fit_error,
            b1,
        };
    }

    // --- loudness grading (linear interp of threshold multiplier in dB) ---
    let e = frame_energy.max(1e-12);
    let e_db = 10.0 * e.log10();
    let loud = ((e_db - cfg.quiet_db) / (cfg.loud_db - cfg.quiet_db)).clamp(0.0, 1.0);
    let energy_factor = cfg.thr_quiet + loud * (cfg.thr_loud - cfg.thr_quiet);

    // --- crest grading (phase-sensitive bias) ---
    let crest_factor = (crest / cfg.crest_ref).clamp(cfg.crest_min, cfg.crest_max);

    // --- silence rule: whole frame unvoiced below the floor ---
    let silent = frame_energy < cfg.silence_floor;

    // --- onset voicing-extent ramp ---
    // Quiet frames only let the low harmonics go voiced; the cutoff ramps up
    // with loudness and (a little) crest. High bands above the extent park UV.
    let extent_frac = (cfg.min_extent
        + loud * (1.0 - cfg.min_extent)
        + cfg.crest_extent_gain * (crest_factor - 1.0))
        .clamp(0.0, 1.0);
    let max_voiced = ((extent_frac * l as f64).ceil() as usize).min(l);

    // --- frame periodicity gate ---
    // Fricatives (/s/, /f/, /ks/) are aperiodic noise: HIGH fit error across
    // ALL harmonics, including the low ones. Genuinely voiced sounds have a
    // clean low-harmonic comb (LOW fit error). Measure the low-harmonic fit
    // and only apply the loose (aggressive) per-band threshold when the frame
    // is actually periodic -- otherwise tighten it so unvoiced fricatives are
    // not buzzed into partial voicing. This lets a loud voiced onset (/w/)
    // keep its high bands voiced without also voicing the sibilants.
    let lo_n = l.clamp(1, 6);
    let lo_fit: f64 = (1..=lo_n)
        .map(|h| band_fit_error(spectrum, omega_0, h).unwrap_or(1.0))
        .sum::<f64>()
        / lo_n as f64;
    // periodicity: 1.0 when lo_fit <= 0.15 (clean voiced), 0.0 when >= 0.45 (noise)
    let periodicity = ((0.45 - lo_fit) / 0.30).clamp(0.0, 1.0);
    // fricative frames scale the threshold down toward a tighter value; voiced
    // frames keep the full loose threshold.
    let pgate_floor = 0.75;
    let frame_voiced_factor = pgate_floor + (1.0 - pgate_floor) * periodicity;

    for l_h in 1..=l {
        let idx = l_h - 1;
        let d_l = band_fit_error(spectrum, omega_0, l_h);
        let d_val = d_l.unwrap_or(1.0);
        fit_error[idx] = d_val;

        if silent || l_h > max_voiced || d_l.is_none() {
            per_band[idx] = false;
            continue;
        }
        // frequency-tilted, loudness/crest-graded threshold
        let freq_pos = if l > 1 {
            idx as f64 / (l - 1) as f64
        } else {
            0.0
        };
        let freq_factor = 1.0 - cfg.freq_tilt * freq_pos;
        let thr = cfg.base_thr * energy_factor * crest_factor * freq_factor * frame_voiced_factor;
        per_band[idx] = d_val < thr;
    }

    // --- per-3-harmonic-band unvoicing veto (IMBE) ------------------------
    // A single harmonic band (~6 bins) cannot separate a sustained fricative
    // (SH/T) from a voiced tooth, but a 3-harmonic band (~18 bins) can: three
    // sharp comb teeth vs flat noise. We compute the standard MBE per-band
    // voicing error over each 3-harmonic band (the resolution IMBE transmits
    // at) and use it ONLY to VETO — unvoice a whole band whose error clears
    // `band_thr`, i.e. flat noise the narrow window teeth cannot explain. It
    // never ADDS voicing, so genuinely voiced bands (very low error) are
    // untouched and no new breathiness is introduced; it only removes the
    // fricative buzz the per-harmonic staircase leaves behind.
    if cfg.band_voicing && !silent {
        let errs = band_voicing_errors(spectrum, omega_0, l);
        for (b, &e) in errs.iter().enumerate() {
            if e >= cfg.band_thr {
                let first = 3 * b;
                let last = (3 * b + 2).min(l - 1);
                for slot in per_band.iter_mut().take(last + 1).skip(first) {
                    *slot = false;
                }
            }
        }
    }

    // --- bimodal snap -----------------------------------------------------
    // The reference's V/UV field is strongly bimodal: the great majority
    // of frames land on the all-voiced codebook row (index 0) or an all-
    // unvoiced row (16). Partial staircase patterns are the minority. Our
    // per-harmonic decision naturally produces staircase patterns, which map
    // to partial rows and cost low-order b1 bits against the reference's
    // predominantly-zero low bits. Snapping a mostly-voiced frame to fully
    // voiced (row 0) and a mostly-unvoiced frame to fully unvoiced (row 16)
    // raises the b1 bit-match. Setting hi<=lo disables the snap.
    if !silent && l > 0 {
        // A single binary threshold at ~0.32 of the harmonics voiced (hi just
        // above lo). The reference's V/UV field is near-binary, so a hard
        // frame-level all-voiced / all-unvoiced decision beats the staircase.
        let hi = 0.325f64;
        let lo = 0.315f64;
        if hi > lo {
            let vc = per_band.iter().filter(|&&v| v).count();
            let frac = vc as f64 / l as f64;
            // Fricative protection (IMBE): withhold the upward snap when the
            // TOP third of harmonics are mostly unvoiced. A frame with a voiced
            // low band but an unvoiced high band is a fricative/affricate (SH,
            // T, soft-G); snapping it all-voiced buzzes the high SH noise. Reference
            // keeps these frames partial, so we keep the per-band pattern too.
            let snap_up_ok = if cfg.fricative_protect {
                let hi_start = l - l.div_ceil(3); // top third (ceil) of harmonics
                let hi_n = l - hi_start;
                let hi_vc = per_band[hi_start..].iter().filter(|&&v| v).count();
                hi_n == 0 || (hi_vc as f64 / hi_n as f64) >= cfg.fricative_hi_frac
            } else {
                true
            };
            if frac >= hi && snap_up_ok {
                for v in per_band.iter_mut() {
                    *v = true;
                }
            } else if frac <= lo {
                for v in per_band.iter_mut() {
                    *v = false;
                }
            }
        }
    }

    let b1 = collapse_to_b1(&per_band, omega_0);
    VoicingDecision {
        per_band,
        fit_error,
        b1,
    }
}
