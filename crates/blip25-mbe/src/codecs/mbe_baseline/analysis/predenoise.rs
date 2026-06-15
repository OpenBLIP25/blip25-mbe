//! Pre-analysis speech denoiser front-end (§3.4 "exceed-the-chip" lever).
//!
//! A **separable, general-DSP** noise suppressor that runs on the raw i16
//! PCM frame *before* the codec sees it (HPF / §0.2 / §0.3 / §0.5). The
//! AMBE-3000R applies no noise suppression on the bench (noise passes at
//! ~0.94×, `QUALITY_FINDINGS.md` §3.4), so a modern denoiser front-end can
//! *exceed* the chip on noisy field audio. The codec core is untouched;
//! this stage is opt-in (default OFF) and lives outside the spec
//! `AnalysisState`.
//!
//! Pipeline (all general DSP — outside the TIA-102 clean-room scope):
//! streaming STFT (256-pt, 128 hop, √-Hann WOLA) → per-bin noise PSD via
//! minimum-statistics / MCRA recursive averaging (Cohen/Martin) → log-MMSE
//! (Ephraim-Malah) suppression gain with decision-directed a-priori SNR →
//! inverse STFT / overlap-add. √-Hann at 50 % overlap is COLA-exact, so at
//! unity gain the output equals the input delayed by one hop.
//!
//! Self-priming guard (the defect that sank the legacy in-spectrum Boll
//! stage, `denoise.rs` / §3.4 defect (h)): the noise model is **never**
//! updated on a spectral peak (a sustained tone / voiced harmonic sits far
//! above its local-minimum floor and above its bin neighbourhood), so
//! tonal/voiced content is not learned as noise. A global high-SNR bypass
//! keeps clean speech transparent.

use std::sync::Arc;

use num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::mbe_params::SAMPLES_PER_FRAME;

const FRAME: usize = SAMPLES_PER_FRAME as usize; // 160
const FFT_SIZE: usize = 256;
const HOP: usize = 128; // 50 % overlap
const N_BINS: usize = FFT_SIZE / 2 + 1; // 129 one-sided bins

// --- Noise tracking (IMCRA two-iteration min-controlled recursive avg,
//     Cohen 2003 Table-I @ 8 kHz / 256-FFT / 62.5 hop·s⁻¹) -------------------
const ALPHA_S: f64 = 0.9; // 1st-pass periodogram time-smoothing
const ALPHA_S2: f64 = 0.9; // 2nd-pass (speech-excluded) smoothing
const ALPHA_D: f64 = 0.85; // base noise-PSD update smoothing
const ALPHA_P: f64 = 0.2; // speech-presence-probability smoothing (Cohen α_p)
const GAMMA0: f64 = 4.6; // rough speech indicator threshold on γ_min (1st min)
const GAMMA1: f64 = 3.0; // soft p(k) ramp ceiling on γ_min (2nd min)
const ZETA0: f64 = 1.67; // smoothed (ζ) indicator threshold
/// Minimum-tracking bias `Bmin` (Martin 2001 / Cohen 2003): the running
/// minimum of a smoothed periodogram under-estimates the mean; for the
/// `α_s=0.9` smoother over `D = U_SUB·V_LEN = 120` hops, Cohen's Table-I
/// gives ≈1.66. Applied ONLY where the minimum is a noise-floor *reference*
/// in a ratio test (γ_min, ζ) — never to `n_psd` itself (it is already
/// recursively unbiased; double-correcting would over-suppress).
const B_MIN: f64 = 1.66;
const U_SUB: usize = 8; // sub-windows
const V_LEN: usize = 15; // hops per sub-window ⇒ D = 120 hops ≈ 1.9 s
const PEAK_GUARD_DB: f64 = 6.0; // bin > this above its neighbourhood ⇒ tone/harmonic, freeze
const BOOTSTRAP_FRAMES: u32 = 6; // seed the noise floor, pass-through until primed

// --- Suppression gain (log-MMSE / decision-directed) ------------------------
const ALPHA_DD: f64 = 0.98; // decision-directed a-priori SNR smoothing (E&M 1984)
const XI_MIN: f64 = 0.003_162_277_66; // −25 dB a-priori SNR floor
const G_MIN: f64 = 0.1; // −20 dB gain floor (anti musical-noise null)
const BETA_SUBSUB: f64 = 2.0; // spectral-subtraction over-subtraction (A/B)

// --- Clean-speech transparency ---------------------------------------------
const SNR_BYPASS_DB: f64 = 25.0; // above this frame SNR: full bypass (gains = 1)
const SNR_SOFT_LO: f64 = 15.0; // below this: full suppression strength
/// GLOBAL engage gate: if the long-term input SNR (speech peak vs noise
/// floor) exceeds this, the *input* is clean and the whole stage bypasses
/// (gains = 1). The per-frame gate alone over-suppresses quiet phonemes on
/// clean speech; this keeps clean input transparent while still engaging on
/// genuinely noisy field audio (≤ ~20 dB mixes track well below it).
const GLOBAL_BYPASS_DB: f64 = 33.0;
/// Spectral-flatness bypass: a near-pure tone has very low flatness (its
/// energy is in a few bins) — bypass it so the denoiser never destroys a
/// sustained tone (alarm / whistle / Knox, which the codec routes via the
/// separate Annex-T tone path). Broadband / babble noise has high flatness
/// and still engages. Voiced speech sits well above this floor.
const SFM_BYPASS: f64 = 0.02;

/// Suppression-gain rule. `LogMmse` is the default (least musical noise);
/// `Wiener` and `SpecSub` are kept for A/B against the §3.4 baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenoiseKind {
    /// Ephraim-Malah log-MMSE STSA (default).
    LogMmse,
    /// Decision-directed Wiener `ξ/(1+ξ)`.
    Wiener,
    /// Power spectral subtraction (Boll, the §3.4 crude baseline).
    SpecSub,
}

/// √-Hann window, COLA-exact at 50 % overlap (analysis = synthesis).
fn sqrt_hann() -> [f64; FFT_SIZE] {
    let mut w = [0.0f64; FFT_SIZE];
    for (i, slot) in w.iter_mut().enumerate() {
        let h = 0.5 - 0.5 * (2.0 * core::f64::consts::PI * i as f64 / FFT_SIZE as f64).cos();
        *slot = h.sqrt();
    }
    w
}

/// Abramowitz-Stegun rational/series approximation of the exponential
/// integral `E1(x)` for `x > 0` (deterministic, no table).
#[inline]
fn expint_e1(x: f64) -> f64 {
    if x < 1.0 {
        // 5.1.53 series.
        -x.ln() - 0.577_215_664_9 + x - x * x / 4.0 + x * x * x / 18.0 - x.powi(4) / 96.0
    } else {
        // 5.1.56 rational form for x·eˣ·E1(x).
        let (a1, a2, b1, b2) = (2.334_733, 0.250_621, 3.330_657, 1.681_534);
        let num = x * x + a1 * x + a2;
        let den = x * x + b1 * x + b2;
        (num / den) / (x * x.exp())
    }
}

/// Streaming pre-analysis denoiser. Construct once per stream; feed
/// 160-sample frames, receive 160 cleaned samples (with a fixed one-hop
/// = 128-sample group delay, which PESQ's xcorr alignment absorbs).
pub struct PreDenoise {
    kind: DenoiseKind,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
    win: [f64; FFT_SIZE],

    // STFT / OLA buffers.
    in_buf: Vec<f64>,  // input samples awaiting a full window
    out_buf: Vec<f64>, // cleaned samples ready to emit
    ola: [f64; FFT_SIZE],

    // Noise tracking (per bin). IMCRA runs TWO smoothing/minimum chains that
    // share the sub-window phase (`sub_idx`/`frame_in_sub`): the first feeds
    // the rough speech indicator, the second (speech-excluded) tracks the
    // true noise floor through continuous/babble speech.
    s_smooth: [f64; N_BINS], // 1st-pass time-smoothed periodogram
    run_min: [f64; N_BINS],  // running min of s_smooth in current sub-window
    sub_min: [[f64; N_BINS]; U_SUB],
    sf_tilde: [f64; N_BINS], // 2nd-pass smoothed power (speech bins held out)
    run_min_t: [f64; N_BINS], // running min of sf_tilde
    sub_min_t: [[f64; N_BINS]; U_SUB],
    sub_idx: usize,
    frame_in_sub: usize,
    n_psd: [f64; N_BINS],    // noise power estimate
    p_speech: [f64; N_BINS], // IMCRA speech-presence probability p(k)

    // Decision-directed SNR memory.
    prev_g: [f64; N_BINS],
    prev_gamma: [f64; N_BINS],

    primed: bool,
    boot_count: u32,
    sig_peak: f64,      // fast-attack / slow-decay speech-level tracker
    input_snr_ema: f64, // slow EMA of long-term input SNR for the engage gate
}

impl std::fmt::Debug for PreDenoise {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreDenoise")
            .field("kind", &self.kind)
            .field("primed", &self.primed)
            .finish()
    }
}

impl Clone for PreDenoise {
    fn clone(&self) -> Self {
        // FFT plans are re-planned (cheap, cached by rustfft); numeric
        // state is copied so a cloned encoder reproduces the stream.
        let mut c = Self::new(self.kind);
        c.in_buf = self.in_buf.clone();
        c.out_buf = self.out_buf.clone();
        c.ola = self.ola;
        c.s_smooth = self.s_smooth;
        c.run_min = self.run_min;
        c.sub_min = self.sub_min;
        c.sf_tilde = self.sf_tilde;
        c.run_min_t = self.run_min_t;
        c.sub_min_t = self.sub_min_t;
        c.sub_idx = self.sub_idx;
        c.frame_in_sub = self.frame_in_sub;
        c.n_psd = self.n_psd;
        c.p_speech = self.p_speech;
        c.prev_g = self.prev_g;
        c.prev_gamma = self.prev_gamma;
        c.primed = self.primed;
        c.boot_count = self.boot_count;
        c.sig_peak = self.sig_peak;
        c.input_snr_ema = self.input_snr_ema;
        c
    }
}

impl PreDenoise {
    /// New denoiser with the given gain rule. Cold-start: passes input
    /// through unchanged until the noise floor is bootstrapped
    /// (`BOOTSTRAP_FRAMES` hops).
    pub fn new(kind: DenoiseKind) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let ifft = planner.plan_fft_inverse(FFT_SIZE);
        Self {
            kind,
            fft,
            ifft,
            win: sqrt_hann(),
            in_buf: Vec::with_capacity(FFT_SIZE + FRAME),
            out_buf: Vec::with_capacity(FFT_SIZE + FRAME),
            ola: [0.0; FFT_SIZE],
            s_smooth: [0.0; N_BINS],
            run_min: [f64::INFINITY; N_BINS],
            sub_min: [[f64::INFINITY; N_BINS]; U_SUB],
            sf_tilde: [0.0; N_BINS],
            run_min_t: [f64::INFINITY; N_BINS],
            sub_min_t: [[f64::INFINITY; N_BINS]; U_SUB],
            sub_idx: 0,
            frame_in_sub: 0,
            n_psd: [0.0; N_BINS],
            p_speech: [0.0; N_BINS],
            prev_g: [1.0; N_BINS],
            prev_gamma: [1.0; N_BINS],
            primed: false,
            boot_count: 0,
            sig_peak: 0.0,
            input_snr_ema: f64::NAN, // set on the first primed frame
        }
    }

    /// Reset all streaming + noise state (matches `Vocoder::reset`).
    pub fn reset(&mut self) {
        let kind = self.kind;
        let fft = self.fft.clone();
        let ifft = self.ifft.clone();
        *self = Self::new(kind);
        self.fft = fft;
        self.ifft = ifft;
    }

    /// The configured suppression rule.
    #[inline]
    pub fn kind(&self) -> DenoiseKind {
        self.kind
    }

    /// Denoise one 160-sample frame. Returns exactly 160 cleaned samples
    /// (one-hop group delay; zero-padded only during the initial fill).
    pub fn process_frame(&mut self, frame: &[i16]) -> Vec<i16> {
        for &s in frame {
            self.in_buf.push(f64::from(s));
        }
        while self.in_buf.len() >= FFT_SIZE {
            self.process_hop();
            self.in_buf.drain(0..HOP);
        }
        // Emit one frame; underflow (startup) pads with zeros.
        let mut out = Vec::with_capacity(FRAME);
        let avail = self.out_buf.len().min(FRAME);
        for s in self.out_buf.drain(0..avail) {
            out.push(s.round().clamp(-32768.0, 32767.0) as i16);
        }
        while out.len() < FRAME {
            out.push(0);
        }
        out
    }

    fn process_hop(&mut self) {
        // Windowed analysis frame.
        let mut buf = vec![Complex::<f64>::new(0.0, 0.0); FFT_SIZE];
        for i in 0..FFT_SIZE {
            buf[i] = Complex::new(self.in_buf[i] * self.win[i], 0.0);
        }
        self.fft.process(&mut buf);

        // Per-bin power on the one-sided spectrum.
        let mut power = [0.0f64; N_BINS];
        for (m, p) in power.iter_mut().enumerate() {
            *p = buf[m].norm_sqr();
        }

        let gains = self.compute_gains(&power);

        // Apply symmetric gains and inverse-transform.
        for m in 0..N_BINS {
            buf[m] *= gains[m];
        }
        for m in 1..(FFT_SIZE - N_BINS + 1) {
            // Mirror conjugate-symmetric upper half (m = 1..127 → 255..129).
            buf[FFT_SIZE - m] = buf[m].conj();
        }
        self.ifft.process(&mut buf);

        // Overlap-add with the synthesis window; rustfft is unnormalized,
        // so divide by FFT_SIZE. √-Hann²-COLA = 1 at 50 % overlap.
        let norm = FFT_SIZE as f64;
        for i in 0..FFT_SIZE {
            self.ola[i] += buf[i].re / norm * self.win[i];
        }
        for i in 0..HOP {
            self.out_buf.push(self.ola[i]);
        }
        self.ola.copy_within(HOP.., 0);
        for slot in self.ola[(FFT_SIZE - HOP)..].iter_mut() {
            *slot = 0.0;
        }
    }

    fn compute_gains(&mut self, power: &[f64; N_BINS]) -> [f64; N_BINS] {
        // Bootstrap the noise floor; pass-through (gain 1) until primed.
        // Seed from the per-bin MINIMUM (not mean) so transient energy in
        // the lead-in doesn't inflate the floor, then PEAK-GUARD the seed:
        // a spectral peak (tone / voiced harmonic) is signal, not noise, so
        // its seed is pulled down to its neighbourhood. This keeps a clean
        // tonal/voiced lead-in from being learned as noise (the bootstrap
        // analogue of the running peak guard).
        if !self.primed {
            for m in 0..N_BINS {
                if self.boot_count == 0 || power[m] < self.n_psd[m] {
                    self.n_psd[m] = power[m];
                }
                self.s_smooth[m] = power[m];
            }
            self.boot_count += 1;
            if self.boot_count >= BOOTSTRAP_FRAMES {
                let seed = self.n_psd;
                for m in 0..N_BINS {
                    let nb = local_floor(&seed, m);
                    if seed[m] > exp2(PEAK_GUARD_DB) * nb {
                        self.n_psd[m] = nb;
                    }
                    // Seed BOTH IMCRA minimum chains from the smoothed power.
                    self.sf_tilde[m] = self.s_smooth[m];
                    self.run_min[m] = self.s_smooth[m];
                    self.run_min_t[m] = self.s_smooth[m];
                    for u in 0..U_SUB {
                        self.sub_min[u][m] = self.s_smooth[m];
                        self.sub_min_t[u][m] = self.s_smooth[m];
                    }
                }
                self.primed = true;
            }
            return [1.0; N_BINS];
        }

        // Long-term INPUT SNR (speech peak vs noise floor) decides whether
        // the input is clean (bypass) or noisy (engage); the per-frame SNR
        // drives the soft crossfade within a noisy stream.
        let frame_pow: f64 = power.iter().sum();
        let noise_pow = self.n_psd.iter().sum::<f64>().max(1e-9);
        if frame_pow > self.sig_peak {
            self.sig_peak = frame_pow; // fast attack
        } else {
            self.sig_peak = 0.999 * self.sig_peak + 0.001 * frame_pow; // slow decay
        }
        let input_snr_db = 10.0 * (self.sig_peak / noise_pow).log10();
        // Slow EMA of the input SNR drives the engage gate: a clean stream
        // stays well above the bypass threshold even through quiet dips, so a
        // few low-energy frames can't trip suppression on otherwise-clean
        // input (IMCRA's sharper floor makes the instantaneous value dip).
        self.input_snr_ema = if self.input_snr_ema.is_nan() {
            input_snr_db
        } else {
            0.95 * self.input_snr_ema + 0.05 * input_snr_db
        };
        let snr_db = 10.0 * (frame_pow / noise_pow).log10();

        // Spectral flatness (geo/arith mean of power): ~0 for a pure tone,
        // ~1 for white noise. Protects sustained tones from being suppressed.
        let sfm = {
            let mut log_sum = 0.0;
            let mut ar_sum = 0.0;
            for &p in power.iter() {
                log_sum += (p + 1e-9).ln();
                ar_sum += p;
            }
            (log_sum / N_BINS as f64).exp() / (ar_sum / N_BINS as f64 + 1e-9)
        };

        self.update_noise(power);

        // GLOBAL ENGAGE GATE — clean input or a near-pure tone is fully
        // transparent; only genuinely noisy broadband input is suppressed.
        if self.input_snr_ema >= GLOBAL_BYPASS_DB || sfm < SFM_BYPASS {
            for m in 0..N_BINS {
                self.prev_g[m] = 1.0;
                self.prev_gamma[m] = (power[m] / self.n_psd[m].max(1e-12)).max(1e-6);
            }
            return [1.0; N_BINS];
        }

        let mut g = [1.0f64; N_BINS];
        for m in 0..N_BINS {
            let y2 = power[m];
            let nz = self.n_psd[m].max(1e-12);
            let gamma = (y2 / nz).max(1e-6); // a-posteriori SNR
            let xi = (ALPHA_DD * self.prev_g[m].powi(2) * self.prev_gamma[m]
                + (1.0 - ALPHA_DD) * (gamma - 1.0).max(0.0))
            .max(XI_MIN); // decision-directed a-priori SNR
            let gain = match self.kind {
                DenoiseKind::LogMmse => {
                    let v = (xi / (1.0 + xi) * gamma).max(1e-8);
                    (xi / (1.0 + xi)) * (0.5 * expint_e1(v)).exp()
                }
                DenoiseKind::Wiener => xi / (1.0 + xi),
                DenoiseKind::SpecSub => (1.0 - BETA_SUBSUB * nz / y2.max(1e-12)).max(0.0).sqrt(),
            }
            .clamp(G_MIN, 1.0);
            g[m] = gain;
            self.prev_g[m] = gain;
            self.prev_gamma[m] = gamma;
        }

        // Soft crossfade toward unity between SNR_SOFT_LO and SNR_BYPASS_DB.
        let blend = ((snr_db - SNR_SOFT_LO) / (SNR_BYPASS_DB - SNR_SOFT_LO)).clamp(0.0, 1.0);
        if blend > 0.0 {
            for gm in g.iter_mut() {
                *gm = (1.0 - blend) * *gm + blend;
            }
        }
        g
    }

    /// IMCRA (Cohen 2003) two-iteration minimum-controlled recursive average.
    /// The first chain feeds a rough speech indicator; the second smooths the
    /// power with strong-speech bins **excluded**, so its minimum tracks the
    /// true noise floor *through* continuous/babble speech (where a single
    /// minimum sits above the floor and under-estimates the noise).
    fn update_noise(&mut self, power: &[f64; N_BINS]) {
        // 1) frequency-smooth (1-2-1) then time-smooth (FIRST pass).
        let mut sf = [0.0f64; N_BINS];
        for m in 0..N_BINS {
            let lo = if m == 0 { 0 } else { m - 1 };
            let hi = if m == N_BINS - 1 { N_BINS - 1 } else { m + 1 };
            sf[m] = 0.25 * power[lo] + 0.5 * power[m] + 0.25 * power[hi];
        }
        for m in 0..N_BINS {
            self.s_smooth[m] = ALPHA_S * self.s_smooth[m] + (1.0 - ALPHA_S) * sf[m];
            if self.s_smooth[m] < self.run_min[m] {
                self.run_min[m] = self.s_smooth[m];
            }
        }

        // 2b) ROUGH speech indicator I(k) from the FIRST minimum (+ Bmin).
        //     A bin is strong-speech if Y²/(Bmin·Smin) or Sf/(Bmin·Smin) is high.
        let mut indicator = [false; N_BINS];
        for m in 0..N_BINS {
            let mut smin1 = self.run_min[m];
            for u in 0..U_SUB {
                if self.sub_min[u][m] < smin1 {
                    smin1 = self.sub_min[u][m];
                }
            }
            let smin1 = (B_MIN * smin1).max(1e-12);
            indicator[m] = power[m] / smin1 > GAMMA0 || self.s_smooth[m] / smin1 > ZETA0;
        }

        // 2c) SECOND smoothing EXCLUDING strong-speech bins (Cohen Eq.14-15):
        //     speech bins FREEZE sf_tilde (speech can't leak into the floor),
        //     noise bins track — so the second minimum follows the noise.
        for m in 0..N_BINS {
            if !indicator[m] {
                self.sf_tilde[m] = ALPHA_S2 * self.sf_tilde[m] + (1.0 - ALPHA_S2) * sf[m];
            }
            if self.sf_tilde[m] < self.run_min_t[m] {
                self.run_min_t[m] = self.sf_tilde[m];
            }
        }

        // 2d) sub-window swap drives BOTH minima together (shared phase).
        self.frame_in_sub += 1;
        if self.frame_in_sub >= V_LEN {
            for m in 0..N_BINS {
                self.sub_min[self.sub_idx][m] = self.run_min[m];
                self.sub_min_t[self.sub_idx][m] = self.run_min_t[m];
                self.run_min[m] = self.s_smooth[m];
                self.run_min_t[m] = self.sf_tilde[m];
            }
            self.sub_idx = (self.sub_idx + 1) % U_SUB;
            self.frame_in_sub = 0;
        }

        // 3) SOFT speech-presence probability p(k) from the SECOND minimum,
        //    + median peak guard, + the conditional alpha_tilde noise update.
        for m in 0..N_BINS {
            let mut smin2 = self.run_min_t[m];
            for u in 0..U_SUB {
                if self.sub_min_t[u][m] < smin2 {
                    smin2 = self.sub_min_t[u][m];
                }
            }
            let smin2 = (B_MIN * smin2).max(1e-12);

            let gamma_min_t = power[m] / smin2;
            let zeta_t = self.s_smooth[m] / smin2;
            let mut p_target = if gamma_min_t <= 1.0 && zeta_t < ZETA0 {
                0.0
            } else {
                ((gamma_min_t - 1.0) / (GAMMA1 - 1.0)).clamp(0.0, 1.0)
            };

            // PEAK GUARD (anti-self-priming): a tone / voiced harmonic bin is
            // ALWAYS speech — never learn it as noise.
            let nb = local_floor(power, m);
            if power[m] > exp2(PEAK_GUARD_DB) * nb {
                p_target = 1.0;
            }

            self.p_speech[m] = ALPHA_P * self.p_speech[m] + (1.0 - ALPHA_P) * p_target;

            // Conditional noise smoothing, frozen as p → 1. Bmin is NOT applied
            // here — n_psd is already recursively unbiased.
            let alpha_tilde = ALPHA_D + (1.0 - ALPHA_D) * self.p_speech[m];
            self.n_psd[m] = alpha_tilde * self.n_psd[m] + (1.0 - alpha_tilde) * power[m];
        }
    }
}

/// Local spectral FLOOR for the peak/harmonic guard: the MEDIAN power over
/// a ±[`FLOOR_HALF`]-bin window. The median is robust to (a) a windowed
/// tone's main-lobe leakage (~4 bins) and (b) a few harmonic peaks inside
/// the window — both leave the median at the noise floor, so a tone /
/// voiced-harmonic bin sits far above it and is recognised as signal.
/// Broadband noise has a flat spectrum, so its bins sit near the median
/// and are NOT guarded (the noise model still learns them).
#[inline]
fn local_floor(power: &[f64; N_BINS], m: usize) -> f64 {
    const FLOOR_HALF: usize = 8;
    let lo = m.saturating_sub(FLOOR_HALF);
    let hi = (m + FLOOR_HALF).min(N_BINS - 1);
    let mut win: [f64; 2 * FLOOR_HALF + 1] = [0.0; 2 * FLOOR_HALF + 1];
    let n = hi - lo + 1;
    win[..n].copy_from_slice(&power[lo..=hi]);
    let w = &mut win[..n];
    w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    w[n / 2]
}

/// `2^x` for the dB peak-guard threshold (power ratio = 10^(dB/10), but
/// the guard is expressed in "dB above neighbourhood" on power, so we use
/// `10^(PEAK_GUARD_DB/10)`).
#[inline]
fn exp2(db: f64) -> f64 {
    10.0_f64.powf(db / 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(x: &[i16]) -> f64 {
        if x.is_empty() {
            return 0.0;
        }
        let s: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        (s / x.len() as f64).sqrt()
    }

    fn sine(freq: f64, n: usize, amp: f64) -> Vec<i16> {
        (0..n)
            .map(|i| (amp * (2.0 * core::f64::consts::PI * freq * i as f64 / 8000.0).sin()) as i16)
            .collect()
    }

    /// Unity-gain reconstruction: with no noise to suppress (a clean loud
    /// tone bypasses), the output equals the input apart from the one-hop
    /// group delay and quantisation.
    #[test]
    fn clean_tone_is_passed_through() {
        let mut d = PreDenoise::new(DenoiseKind::LogMmse);
        let tone = sine(1000.0, FRAME * 40, 6000.0);
        let mut out = Vec::new();
        for f in tone.chunks(FRAME) {
            out.extend(d.process_frame(f));
        }
        // Compare steady-state RMS (skip the warm-up + group delay).
        let in_rms = rms(&tone[FRAME * 10..FRAME * 35]);
        let out_rms = rms(&out[FRAME * 10..FRAME * 35]);
        let ratio_db = 20.0 * (out_rms / in_rms).log10();
        assert!(
            ratio_db.abs() < 1.0,
            "clean tone should pass ~unchanged, got {ratio_db:.2} dB"
        );
    }

    /// A sustained tone must NOT be learned as noise (the legacy
    /// self-priming defect): its level may not be suppressed over time.
    #[test]
    fn sustained_tone_is_not_self_primed_away() {
        let mut d = PreDenoise::new(DenoiseKind::LogMmse);
        let tone = sine(1000.0, FRAME * 80, 4000.0);
        let mut out = Vec::new();
        for f in tone.chunks(FRAME) {
            out.extend(d.process_frame(f));
        }
        let early = rms(&out[FRAME * 12..FRAME * 20]);
        let late = rms(&out[FRAME * 70..FRAME * 78]);
        let drift_db = 20.0 * (late / early).log10();
        assert!(
            drift_db.abs() < 1.0,
            "sustained tone drifted {drift_db:.2} dB — noise model primed on it"
        );
    }

    /// White noise should be attenuated once the noise floor is learned.
    #[test]
    fn white_noise_is_attenuated() {
        let mut d = PreDenoise::new(DenoiseKind::LogMmse);
        // Deterministic pseudo-noise (LCG), no RNG dependency.
        let mut x: u64 = 0x1234_5678;
        let noise: Vec<i16> = (0..FRAME * 60)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((x >> 40) as i32 & 0x7FF) - 1024) as i16
            })
            .collect();
        let mut out = Vec::new();
        for f in noise.chunks(FRAME) {
            out.extend(d.process_frame(f));
        }
        let in_rms = rms(&noise[FRAME * 20..FRAME * 55]);
        let out_rms = rms(&out[FRAME * 20..FRAME * 55]);
        assert!(
            out_rms < in_rms * 0.9,
            "white noise not attenuated: in {in_rms:.0} out {out_rms:.0}"
        );
    }

    /// Frame I/O contract: exactly 160 samples in → 160 out, every call.
    #[test]
    fn emits_one_frame_per_call() {
        let mut d = PreDenoise::new(DenoiseKind::Wiener);
        for _ in 0..50 {
            let out = d.process_frame(&vec![0i16; FRAME]);
            assert_eq!(out.len(), FRAME);
        }
    }

    #[test]
    fn reset_restores_cold_start() {
        let mut d = PreDenoise::new(DenoiseKind::LogMmse);
        for _ in 0..30 {
            d.process_frame(&sine(700.0, FRAME, 3000.0));
        }
        assert!(d.primed);
        d.reset();
        assert!(!d.primed);
        assert_eq!(d.boot_count, 0);
    }
}
