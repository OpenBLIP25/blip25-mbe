// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! Baseline MBE synthesis: MbeParams -> 8 kHz i16 PCM.
//!
//! TIA-102.BABA-A §1.10 (amplitude enhancement) + §1.12 (voiced
//! sum-of-sinusoids, unvoiced DFT-noise). The voiced phase uses the
//! AMBE+2 magnitude-driven phase regeneration (US5701390 §5); the
//! reference's exact per-harmonic phase back-end (the P-8 frontier) is NOT
//! reproduced — magnitude matches the reference (~1.0x RMS) but per-harmonic
//! phase does not, so the raw time-domain SNR vs reference is negative by
//! design. See README.
//!
//! The reference's phase back-end is a structurally different algorithm from the one
//! below — a discrete per-frame `k*omega_0 + lag-cached-residual`
//! recomputation, not a continuously-accumulated running phase plus
//! magnitude-driven regen. It is not reproduced here.

use core::f64::consts::PI as PI64;

use crate::dequantize::{MbeParams, L_MAX};
use crate::phase_regen::ambe_phase_regen;

/// Samples per 20 ms frame at 8 kHz.
pub const FRAME_SAMPLES: usize = 160;

/// Unvoiced spectral scale gamma_w (Eq. 121), pre-evaluated from Annex I/C.
pub(crate) const GAMMA_W: f64 = 146.643269;

/// Local-energy state init (§10 Annex A).
const INIT_S_E: f64 = 75000.0;
const S_E_FLOOR: f64 = 10000.0;
/// Reference-validated effective phase-regen scale.
const REGEN_SCALE: f64 = 0.4;

/// Voiced/regen split fraction: harmonics with l <= round(frac·L) use the pure
/// linear phase track (no regen). frac = 0.25 is the §1.12 lower-quarter rule.
const REGEN_THRESH_FRAC: f64 = 0.25;

/// Reference half-rate unvoiced LCG (173, 13849, 65536). Same generator as the IMBE
/// PN descramble in `imbe::frame`, but consumed differently: here the FULL
/// 16-bit word is a noise sample (unvoiced spectrum), whereas the PN descramble
/// takes only the MSB per step. The two uses are deliberately split.
const LCG_A: u32 = 173;
const LCG_C: u32 = 13849;
const LCG_M: u32 = 65536;
/// Cold-start seed for the continuous unvoiced-noise generator.
///
/// This value is a mid-stream sample of the DLL's noise LCG (`LCG^32824(0)`),
/// not a cold-start state, and it must stay that way. The DLL's true cold start
/// is state 0 followed by 84 draws that fill an 84-word noise ring — but that
/// ring is consumed only by the DLL's FIXED-POINT unvoiced path. This module is
/// the Annex-I float path, which uses a 209-sample window
/// ([`SynthState::advance_window`]) instead, so there is no structural
/// correspondence between the two generators.
///
/// Substituting the DLL's cold-start pair here is a regression, not a fix:
/// whole-file sample-exact agreement with the reference decode falls from
/// ~31.6% to ~23.2% (correlation 0.054 -> 0.033). Do NOT "correct" this
/// constant to the cold-start pair (seed 0, 84 prefill draws); that pair
/// belongs to the fixed-point chain.
const LCG_SEED: u32 = 60584;

/// Annex I synthesis window wS(n): trapezoid, flat |n|<=55, linear taper
/// (105-|n|)/50 for 55<|n|<=105, zero beyond.
#[inline]
fn synth_window(n: i32) -> f64 {
    let a = n.unsigned_abs() as i32;
    if a <= 55 {
        1.0
    } else if a <= 105 {
        (105 - a) as f64 / 50.0
    } else {
        0.0
    }
}

#[inline]
fn wrap_phase(d: f64) -> f64 {
    d - 2.0 * PI64 * ((d + PI64) / (2.0 * PI64)).floor()
}

/// §1.10 spectral amplitude enhancement: M_tilde -> M_bar, updating S_E.
fn enhance(m_tilde: &[f32], omega_0: f32, s_e_prev: f64) -> (Vec<f64>, f64) {
    let l = m_tilde.len();
    let omega_0 = f64::from(omega_0);

    let mut r_m0 = 0.0f64;
    let mut r_m1 = 0.0f64;
    for (i, &m) in m_tilde.iter().enumerate() {
        let l_one = (i + 1) as f64;
        let m2 = f64::from(m) * f64::from(m);
        r_m0 += m2;
        r_m1 += m2 * (omega_0 * l_one).cos();
    }

    let mut m_bar = vec![0f64; l];
    let denom = omega_0 * r_m0 * (r_m0 - r_m1);

    if r_m0 <= 0.0 || denom.abs() < 1e-30 {
        for (i, &m) in m_tilde.iter().enumerate() {
            m_bar[i] = f64::from(m);
        }
    } else {
        for (i, &m) in m_tilde.iter().enumerate() {
            let l_one = (i + 1) as f64;
            let bar = if 8 * (i + 1) <= l {
                f64::from(m)
            } else {
                let num = r_m0 * r_m0 + r_m1 * r_m1 - 2.0 * r_m0 * r_m1 * (omega_0 * l_one).cos();
                if num <= 0.0 {
                    f64::from(m)
                } else {
                    let w_l = f64::from(m).sqrt() * (0.96 * num / denom).powf(0.25);
                    let w_l = w_l.clamp(0.5, 1.2);
                    w_l * f64::from(m)
                }
            };
            m_bar[i] = bar;
        }
        let sum_sq: f64 = m_bar.iter().map(|v| v * v).sum();
        if sum_sq > 1e-30 {
            let gamma = (r_m0 / sum_sq).sqrt();
            for v in m_bar.iter_mut() {
                *v *= gamma;
            }
        }
    }

    let s_e = (0.95 * s_e_prev + 0.05 * r_m0).max(S_E_FLOOR);
    (m_bar, s_e)
}

// ---- 256-pt complex DFT/IDFT (self-contained, no external crates) ---------

/// Precomputed `cos`/`sin` of the DFT kernel angle, indexed `k * 256 + n`.
///
/// # Why this table exists
///
/// Both transforms below are O(N²) by construction: 256 bins x 256 samples =
/// 65 536 inner iterations each. They used to evaluate `ang.cos()` and
/// `ang.sin()` from scratch inside that loop, i.e. 131 072 transcendental
/// pairs per transform and 262 144 per synthesized frame. Measured on a Xeon
/// E5-2687W v4 that put `synthesize_frame` at **3.13 ms per 20 ms of audio**
/// — 15.6% of one core for every active voice stream. On the live scanner,
/// where voice synthesis runs inline on the single-threaded thread that also
/// drains IQ from the radio, a handful of concurrent talkers exhausted that
/// thread's budget, the DSP fell behind the USB ring, and IQ was dropped on
/// every channel at once (2026-07-26 investigation).
///
/// The angle depends only on `(k, n)`, so it is loop-invariant across frames
/// and can simply be cached. The multiply-accumulate that remains is what the
/// transform actually needs.
///
/// # Bit-exactness
///
/// The table stores `cos`/`sin` of the **unreduced** angle
/// `-2*pi*k*n/256`, evaluated with the identical expression the inner loops
/// used, so every product is the same `f64` as before and output is
/// bit-identical. Reducing the index to `(k*n) mod 256` — which would shrink
/// the table to 256 entries — is *not* bit-identical: 61 025 of the 65 536
/// cosines and 61 047 sines differ in the last bits (max 2.6e-13) because
/// `cos()` range-reduces a large argument differently. That is numerically
/// irrelevant but this codec is pinned bit-for-bit against the reference
/// reference, so the full table is the right trade: 1 MiB of L3-resident
/// f64, built once.
///
/// The inverse transform's kernel is the conjugate (`+2*pi*k*n/256`), and
/// `cos`/`sin` are exactly even/odd here — verified over all 256 angles, zero
/// mismatches — so it reuses this table with the sine negated rather than
/// carrying a second copy.
struct Twiddle {
    cos: Vec<f64>,
    sin: Vec<f64>,
}

fn twiddle() -> &'static Twiddle {
    static TW: std::sync::OnceLock<Twiddle> = std::sync::OnceLock::new();
    TW.get_or_init(|| {
        let mut cos = vec![0f64; 256 * 256];
        let mut sin = vec![0f64; 256 * 256];
        for k in 0..256usize {
            for n in 0..256usize {
                // Same expression as the original inner loops, so the stored
                // values are bit-for-bit what they used to compute.
                let ang = -2.0 * PI64 * (k as f64) * (n as f64) / 256.0;
                cos[k * 256 + n] = ang.cos();
                sin[k * 256 + n] = ang.sin();
            }
        }
        Twiddle { cos, sin }
    })
}

/// 256-pt DFT of a real, centered window over n=-104..104.
/// Returns (re, im) indexed [m+128] for m in -128..127.
fn dft_256_windowed(windowed: &[f64; 209]) -> ([f64; 256], [f64; 256]) {
    // place samples at natural index n mod 256
    let mut x = [0f64; 256];
    for (i, &w) in windowed.iter().enumerate() {
        let n = i as i32 - 104;
        let n_nat = n.rem_euclid(256) as usize;
        x[n_nat] = w;
    }
    let tw = twiddle();
    let mut re = [0f64; 256];
    let mut im = [0f64; 256];
    for m_idx in 0..256 {
        let m = m_idx as i32 - 128;
        let k = m.rem_euclid(256) as usize;
        let (cos_k, sin_k) = (
            &tw.cos[k * 256..k * 256 + 256],
            &tw.sin[k * 256..k * 256 + 256],
        );
        let mut ar = 0f64;
        let mut ai = 0f64;
        for n in 0..256 {
            ar += x[n] * cos_k[n];
            ai += x[n] * sin_k[n];
        }
        re[m_idx] = ar;
        im[m_idx] = ai;
    }
    (re, im)
}

/// 256-pt inverse DFT of the centered (re, im) packing.
/// Output indexed [n+128] for n in -128..127, normalized by 1/256.
fn idft_256(re: &[f64; 256], im: &[f64; 256]) -> [f64; 256] {
    // undo centered rotation: freq[k] = packed[(k+128) mod 256]
    let mut fr = [0f64; 256];
    let mut fi = [0f64; 256];
    for k in 0..256 {
        let m_idx = (k + 128) & 255;
        fr[k] = re[m_idx];
        fi[k] = im[m_idx];
    }
    let tw = twiddle();
    let mut out = [0f64; 256];
    for n_idx in 0..256 {
        let n = (n_idx + 128) & 255;
        // Kernel here is +2*pi*k*n/256, the conjugate of the stored angle:
        // cos matches, sin flips (exact even/odd — see `Twiddle`). The table
        // is symmetric in (k, n), so row `n` indexed by `k` is the same as
        // row `k` indexed by `n`, which keeps this walk contiguous too.
        let (cos_n, sin_n) = (
            &tw.cos[n * 256..n * 256 + 256],
            &tw.sin[n * 256..n * 256 + 256],
        );
        let mut acc = 0f64;
        for k in 0..256 {
            // `- fi*sin(+x)` with `sin(+x) == -sin_n[k]` is exactly
            // `+ fi*sin_n[k]`: IEEE negation and multiplication commute on
            // the sign bit, so this stays bit-identical to the original.
            acc += fr[k] * cos_n[k] + fi[k] * sin_n[k];
        }
        out[n_idx] = acc / 256.0;
    }
    out
}

/// Per-band spectral shaping (Eq. 119-124): zero voiced/out-of-band,
/// scale unvoiced bands so per-bin RMS = gamma_w * M_bar.
fn shape_spectrum(
    re: &mut [f64; 256],
    im: &mut [f64; 256],
    omega_0: f64,
    m_bar: &[f64],
    v_bar: &[bool],
    gamma_w: f64,
) {
    let l_count = m_bar.len();
    let scale = 256.0 / (2.0 * PI64);
    // reference-parity: round-to-nearest band edges.
    let edge = |x: f64| -> i32 { x.round() as i32 };

    let a1 = edge(scale * 0.5 * omega_0);
    let b_last = edge(scale * (l_count as f64 + 0.5) * omega_0);
    for m_idx in 0..256 {
        let m = m_idx as i32 - 128;
        if m.unsigned_abs() < a1 as u32 || (m.unsigned_abs() as i32) >= b_last {
            re[m_idx] = 0.0;
            im[m_idx] = 0.0;
        }
    }

    let mut band_norm = Vec::with_capacity(l_count);
    let mut band_edges = Vec::with_capacity(l_count);
    for l in 1..=l_count as i32 {
        let l_f = f64::from(l);
        let a_l = edge(scale * (l_f - 0.5) * omega_0);
        let b_l = edge(scale * (l_f + 0.5) * omega_0);
        band_edges.push((a_l, b_l));
        let mut norm_sum = 0f64;
        let count = (b_l - a_l).max(0) as usize;
        for eta in a_l..b_l {
            for &sign in &[1i32, -1] {
                let m_idx = (sign * eta + 128) as usize;
                if m_idx < 256 {
                    norm_sum += re[m_idx] * re[m_idx] + im[m_idx] * im[m_idx];
                }
            }
        }
        let norm = if count > 0 && norm_sum > 0.0 {
            (norm_sum / (2.0 * count as f64)).sqrt()
        } else {
            1.0
        };
        band_norm.push(norm);
    }

    for (l_idx, &(a_l, b_l)) in band_edges.iter().enumerate() {
        if v_bar[l_idx] {
            for m_abs in a_l..b_l {
                for &sign in &[1i32, -1] {
                    let m_idx = (sign * m_abs + 128) as usize;
                    if m_idx < 256 {
                        re[m_idx] = 0.0;
                        im[m_idx] = 0.0;
                    }
                }
            }
        } else {
            let norm = band_norm[l_idx];
            let factor = if norm > 0.0 {
                gamma_w * m_bar[l_idx] / norm
            } else {
                0.0
            };
            for m_abs in a_l..b_l {
                for &sign in &[1i32, -1] {
                    let m_idx = (sign * m_abs + 128) as usize;
                    if m_idx < 256 {
                        re[m_idx] *= factor;
                        im[m_idx] *= factor;
                    }
                }
            }
        }
    }
}

// ---- cross-frame synthesizer state ----------------------------------------

/// Cross-frame synthesizer state (§1.13 / Annex A cold start).
#[derive(Clone)]
pub struct SynthState {
    s_e: f64,
    // voiced
    phi: [f64; L_MAX + 1],
    psi: [f64; L_MAX + 1],
    prev_m_bar: [f64; L_MAX + 1],
    prev_v_bar: [bool; L_MAX + 1],
    prev_l: u8,
    prev_omega_0: f64,
    // unvoiced
    lcg: u32,
    noise_window: [f64; 209],
    prev_idft: [f64; 256],
    uv_initialized: bool,
    /// One-shot per-harmonic **absolute onset-phase override** for a tone
    /// burst-start frame (see [`SynthState::seed_tone_onset`]). `Some(arr)` with
    /// `arr[l]` finite pins harmonic `l`'s phase at this frame's first sample to
    /// exactly `arr[l]` (radians), reproducing the reference's deterministic per-tone-ID
    /// onset seed; `NaN` entries mean "no override, use the normal accumulation
    /// path". Consumed (taken) by the next [`synthesize_voiced`] call, so it only
    /// affects the single frame it was set for. Only ever set by the AMBE+2
    /// decoder's tone branch; the voice path never touches it.
    onset_override: Option<[f64; L_MAX + 1]>,
    /// When true, skip the AMBE+2 upper-harmonic phase regeneration
    /// (`rscale·phi_regen`) so `φ_l = ψ_l` for every harmonic. Phase regen is a
    /// voice-naturalness feature; a tone is a pure sinusoid whose harmonic
    /// phases must track the deterministic onset seed exactly, so its constant
    /// per-frame regen offset would otherwise pull every harmonic above
    /// `l_quarter` off the seeded phase. Set for every tone frame by the AMBE+2
    /// decoder's tone branch; false (regen on) for all voice synthesis.
    suppress_regen: bool,
}

impl SynthState {
    pub fn new() -> Self {
        // The noise LCG starts mid-stream and is consumed with no prefill;
        // see LCG_SEED.
        Self {
            s_e: INIT_S_E,
            phi: [0.0; L_MAX + 1],
            psi: [0.0; L_MAX + 1],
            prev_m_bar: [0.0; L_MAX + 1],
            prev_v_bar: [false; L_MAX + 1],
            prev_l: 30,
            prev_omega_0: 0.02985 * PI64,
            lcg: LCG_SEED,
            noise_window: [0.0; 209],
            prev_idft: [0.0; 256],
            uv_initialized: false,
            onset_override: None,
            suppress_regen: false,
        }
    }

    /// Mark the next synthesized frame as a **tone** (pure sinusoid): suppress
    /// the AMBE+2 upper-harmonic phase regeneration so every harmonic's phase is
    /// exactly its accumulated / seeded value. Called on every tone frame by the
    /// AMBE+2 decoder; the voice path leaves it false.
    pub fn set_tone_frame(&mut self, on: bool) {
        self.suppress_regen = on;
    }

    /// Seed a **tone burst start**: pin the given harmonics' absolute onset
    /// phase for the next synthesized frame, and clear the stale
    /// previous-frame voiced state so every tone harmonic is treated as a fresh
    /// onset (a `(prev=false, curr=true)` transition) rather than continuing /
    /// ringing out the harmonics of a *previous* tone burst that the intervening
    /// silence frames never advanced. `override_rad[l]` finite ⇒ harmonic `l`
    /// starts at exactly that phase (radians); `NaN` ⇒ untouched. Isolated to the
    /// tone path: the voice decoder uses [`crate::FixedState`], not this state.
    pub fn seed_tone_onset(&mut self, override_rad: [f64; L_MAX + 1]) {
        self.onset_override = Some(override_rad);
        // Fresh burst: no previous harmonics carry over from the last tone.
        self.prev_v_bar = [false; L_MAX + 1];
        self.prev_m_bar = [0.0; L_MAX + 1];
        self.prev_l = 0;
    }

    /// IMBE-driven synthesis state. Used by [`crate::ImbeDecoder`] only;
    /// currently identical to [`SynthState::new`].
    pub fn new_imbe() -> Self {
        Self::new()
    }

    #[inline]
    fn next_noise(&mut self) -> f64 {
        self.lcg = LCG_A.wrapping_mul(self.lcg).wrapping_add(LCG_C) % LCG_M;
        f64::from(self.lcg)
    }

    fn advance_window(&mut self) {
        if !self.uv_initialized {
            for i in 0..209 {
                self.noise_window[i] = self.next_noise();
            }
            self.uv_initialized = true;
        } else {
            self.noise_window.copy_within(160..209, 0);
            for i in 49..209 {
                self.noise_window[i] = self.next_noise();
            }
        }
    }
}

impl Default for SynthState {
    fn default() -> Self {
        Self::new()
    }
}

/// Unvoiced synthesis for one frame (§1.12.1), 160 f64 samples.
fn synthesize_unvoiced(
    omega_0: f32,
    m_bar: &[f64],
    v_bar: &[bool],
    state: &mut SynthState,
) -> [f64; FRAME_SAMPLES] {
    state.advance_window();

    let mut windowed = [0f64; 209];
    for i in 0..209 {
        let n = i as i32 - 104;
        windowed[i] = state.noise_window[i] * synth_window(n);
    }

    let (mut re, mut im) = dft_256_windowed(&windowed);
    shape_spectrum(&mut re, &mut im, f64::from(omega_0), m_bar, v_bar, GAMMA_W);
    let u_w = idft_256(&re, &im);

    let mut out = [0f64; FRAME_SAMPLES];
    for n in 0..FRAME_SAMPLES as i32 {
        let ws_n = synth_window(n);
        let ws_nm = synth_window(n - FRAME_SAMPLES as i32);
        let prev_idft = if (0..=127).contains(&n) {
            state.prev_idft[(n + 128) as usize]
        } else {
            0.0
        };
        let curr_idft = if (32..=159).contains(&n) {
            u_w[(n - 32) as usize]
        } else {
            0.0
        };
        let denom = ws_n * ws_n + ws_nm * ws_nm;
        out[n as usize] = if denom > 1e-30 {
            (ws_n * prev_idft + ws_nm * curr_idft) / denom
        } else {
            0.0
        };
    }
    state.prev_idft = u_w;
    out
}

/// Per-frame voiced-phase trace: the per-harmonic ψ_l/φ_l/voiced arrays a
/// single frame used.
#[derive(Clone, Default, Debug)]
pub(crate) struct PhaseFrame {
    /// Harmonic count L this frame.
    pub l: usize,
    /// Fundamental ω₀ (rad/sample).
    pub omega_0: f64,
    /// ψ_l(0), l=1..=L — linear/predicted phase track (Eq. 139). [0] unused.
    pub psi: Vec<f64>,
    /// φ_l(0), l=1..=L — synthesis phase actually used (Eq. 140); the direct
    /// analog of the firmware θ_l array. [0] unused.
    pub phi: Vec<f64>,
    /// Per-harmonic voiced flag, l=1..=L. [0] unused.
    pub voiced: Vec<bool>,
}

/// Voiced synthesis for one frame (§1.12.2) with AMBE+2 phase regen.
/// If `phase_out` is Some, the per-harmonic ψ_l/φ_l/voiced arrays used this
/// frame are recorded (for firmware phase-parity validation).
fn synthesize_voiced(
    omega_0: f32,
    m_bar: &[f64],
    v_bar: &[bool],
    state: &mut SynthState,
    phase_out: Option<&mut PhaseFrame>,
) -> [f64; FRAME_SAMPLES] {
    let l_curr = m_bar.len() as u8;
    let l_prev = state.prev_l;
    let max_l = l_curr.max(l_prev) as usize;

    let omega_curr = f64::from(omega_0);
    let omega_prev = state.prev_omega_0;
    let n_f = FRAME_SAMPLES as f64;
    let delta_omega = omega_curr - omega_prev;

    // ψ_l(0) = ψ_l(-1) + (ω_prev + ω_curr)·l·N/2  (Eq. 139)
    let mut psi_curr = [0f64; L_MAX + 1];
    for l in 1..=L_MAX {
        let l_f = l as f64;
        psi_curr[l] = state.psi[l] + (omega_prev + omega_curr) * l_f * n_f / 2.0;
    }

    // φ_l(0) (Eq. 140) with AMBE+2 phase regen on the upper harmonics.
    let l_quarter = (REGEN_THRESH_FRAC * l_curr as f64).round() as usize;
    let rscale = REGEN_SCALE;
    let m_bar_f32: Vec<f32> = m_bar.iter().map(|&v| v as f32).collect();
    let mut phi_regen = [0f64; L_MAX + 1];
    ambe_phase_regen(&m_bar_f32, &mut phi_regen);
    let mut phi_curr = [0f64; L_MAX + 1];
    for l in 1..=L_MAX {
        if l <= l_quarter || state.suppress_regen {
            phi_curr[l] = psi_curr[l];
        } else {
            phi_curr[l] = psi_curr[l] + rscale * phi_regen[l];
        }
    }

    // Tone burst-start onset seed (one-shot): for each overridden harmonic pin
    // the phase at this frame's first sample (n=0) to `R_l`. We set the *stored*
    // phasor `phi_curr[l]`/`psi_curr[l]` to `R_l + ω·N·l` so that (a) the
    // `(false, true)` onset branch below, which reads `phi0 = phi_curr[l] −
    // ω·N·l`, starts the oscillator at exactly `R_l`, and (b) the end-of-frame
    // state advance (`state.phi = phi_curr`, `state.psi = psi_curr`) stores the
    // phase at n=N, so Eq. 139 accumulation continues seamlessly on the rest of
    // the burst. Overwriting here (after the regen term) also skips the
    // `rscale·phi_regen` offset for these harmonics, which would otherwise
    // reintroduce a per-tone phase error. NaN entries are left untouched.
    if let Some(ov) = state.onset_override.take() {
        for l in 1..=L_MAX {
            if ov[l].is_finite() {
                let seed = ov[l] + omega_curr * n_f * l as f64;
                psi_curr[l] = seed;
                phi_curr[l] = seed;
            }
        }
    }

    // Record the per-harmonic phase trace (firmware-parity schema).
    if let Some(pf) = phase_out {
        let lc = l_curr as usize;
        pf.l = lc;
        pf.omega_0 = omega_curr;
        pf.psi = vec![0.0; lc + 1];
        pf.phi = vec![0.0; lc + 1];
        pf.voiced = vec![false; lc + 1];
        for l in 1..=lc {
            pf.psi[l] = wrap_phase(psi_curr[l]);
            pf.phi[l] = wrap_phase(phi_curr[l]);
            pf.voiced[l] = v_bar[l - 1];
        }
    }

    let mut ws_n = [0f64; FRAME_SAMPLES];
    let mut ws_nm = [0f64; FRAME_SAMPLES];
    for n in 0..FRAME_SAMPLES {
        ws_n[n] = synth_window(n as i32);
        ws_nm[n] = synth_window(n as i32 - FRAME_SAMPLES as i32);
    }

    let mut s_v = [0f64; FRAME_SAMPLES];
    for l in 1..=max_l {
        let l_f = l as f64;
        let m_curr_l = if l <= l_curr as usize {
            m_bar[l - 1]
        } else {
            0.0
        };
        let m_prev_l = if l <= l_prev as usize {
            state.prev_m_bar[l]
        } else {
            0.0
        };
        let v_curr = l <= l_curr as usize && v_bar[l - 1];
        let v_prev = l <= l_prev as usize && state.prev_v_bar[l];

        match (v_prev, v_curr) {
            (false, false) => {}
            (true, false) => {
                // prev-frame phasor only
                let phi0 = state.phi[l];
                let (mut pre, mut pim) = (phi0.cos(), phi0.sin());
                let step = omega_prev * l_f;
                let (dc, ds) = (step.cos(), step.sin());
                let scale = 2.0 * m_prev_l;
                for n in 0..FRAME_SAMPLES {
                    s_v[n] += scale * ws_n[n] * pre;
                    let (npre, npim) = (pre * dc - pim * ds, pre * ds + pim * dc);
                    pre = npre;
                    pim = npim;
                }
            }
            (false, true) => {
                let phi0 = phi_curr[l] - omega_curr * n_f * l_f;
                let (mut cre, mut cim) = (phi0.cos(), phi0.sin());
                let step = omega_curr * l_f;
                let (dc, ds) = (step.cos(), step.sin());
                let scale = 2.0 * m_curr_l;
                for n in 0..FRAME_SAMPLES {
                    s_v[n] += scale * ws_nm[n] * cre;
                    let (ncre, ncim) = (cre * dc - cim * ds, cre * ds + cim * dc);
                    cre = ncre;
                    cim = ncim;
                }
            }
            (true, true) => {
                let pitch_change_ratio = if omega_curr.abs() > 1e-30 {
                    (delta_omega * l_f / omega_curr).abs()
                } else {
                    0.0
                };
                if l >= 8 || pitch_change_ratio >= 0.1 {
                    // VVSum: two phasors
                    let (mut pre, mut pim) = (state.phi[l].cos(), state.phi[l].sin());
                    let pstep = omega_prev * l_f;
                    let (pdc, pds) = (pstep.cos(), pstep.sin());
                    let phi_c0 = phi_curr[l] - omega_curr * n_f * l_f;
                    let (mut cre, mut cim) = (phi_c0.cos(), phi_c0.sin());
                    let cstep = omega_curr * l_f;
                    let (cdc, cds) = (cstep.cos(), cstep.sin());
                    let scale_p = 2.0 * m_prev_l;
                    let scale_c = 2.0 * m_curr_l;
                    for n in 0..FRAME_SAMPLES {
                        s_v[n] += scale_p * ws_n[n] * pre + scale_c * ws_nm[n] * cre;
                        let (npre, npim) = (pre * pdc - pim * pds, pre * pds + pim * pdc);
                        pre = npre;
                        pim = npim;
                        let (ncre, ncim) = (cre * cdc - cim * cds, cre * cds + cim * cdc);
                        cre = ncre;
                        cim = ncim;
                    }
                } else {
                    // VVRamp: quadratic phase
                    let delta_phi =
                        phi_curr[l] - state.phi[l] - (omega_prev + omega_curr) * l_f * n_f / 2.0;
                    let wrapped = wrap_phase(delta_phi);
                    let delta_omega_l = wrapped / n_f;
                    let theta0 = state.phi[l];
                    let coef_n2 = (omega_curr * l_f - omega_prev * l_f) / (2.0 * n_f);
                    for n in 0..FRAME_SAMPLES {
                        let nf = n as f64;
                        let a_l = m_prev_l + (nf / n_f) * (m_curr_l - m_prev_l);
                        let theta_l =
                            theta0 + (omega_prev * l_f + delta_omega_l) * nf + coef_n2 * nf * nf;
                        s_v[n] += 2.0 * a_l * theta_l.cos();
                    }
                }
            }
        }
    }

    // advance voiced state
    state.psi[1..(L_MAX + 1)].copy_from_slice(&psi_curr[1..(L_MAX + 1)]);
    state.phi[1..(L_MAX + 1)].copy_from_slice(&phi_curr[1..(L_MAX + 1)]);
    state.prev_m_bar[1..(l_curr as usize + 1)].copy_from_slice(&m_bar[..((l_curr as usize - 1) + 1)]);
    state.prev_v_bar[1..(l_curr as usize + 1)].copy_from_slice(&v_bar[..((l_curr as usize - 1) + 1)]);
    for l in (l_curr as usize + 1)..=L_MAX {
        state.prev_m_bar[l] = 0.0;
        state.prev_v_bar[l] = false;
    }
    state.prev_l = l_curr;
    state.prev_omega_0 = omega_curr;

    s_v
}

/// Synthesize one frame of PCM from MbeParams (§1.10 + §1.12 + Eq. 142).
pub fn synthesize_frame(params: &MbeParams, state: &mut SynthState) -> [i16; FRAME_SAMPLES] {
    synthesize_frame_inner(params, state, None)
}

/// Output soft-limiter knee/ceiling. The reference decode keeps ~1.1 dB of
/// headroom (IMBE peaks ~28.8k of 32767 on dam/mark, ZERO clipped samples)
/// because its phase back-end disperses the harmonic peaks. This
/// phase-approximate synthesis aligns the harmonics instead, so loud voiced
/// peaks stack onto the i16 rail and HARD-CLIP a handful of samples per loud
/// vowel; the broadband clip harmonics are audible as distortion on a raised
/// voice. This is a general-DSP peak limiter (not a reproduction of the un-RE'd
/// phase back-end): linear below KNEE, smooth exponential approach to CEIL
/// above, so |out| < CEIL always and the rail is never reached. KNEE = 27000
/// sits ABOVE the maximum observed AMBE+2 peak (mark, 26909), so AMBE+2 output
/// is byte-identical; only IMBE's clipping peaks are shaped.
const SOFT_LIMIT_KNEE: f64 = 27000.0;
const SOFT_LIMIT_CEIL: f64 = 31500.0;

/// Odd-symmetric soft peak limiter: identity for |s| <= KNEE, then an
/// exponential ramp that asymptotes to CEIL, so the output magnitude is always
/// strictly below CEIL (< 32767 ⇒ no hard clip). Leaves the RMS-dominant body
/// of the signal untouched, so loudness and crest-factor-safe frames are
/// unchanged.
fn soft_limit(s: f64) -> f64 {
    let a = s.abs();
    if a <= SOFT_LIMIT_KNEE {
        return s;
    }
    let range = SOFT_LIMIT_CEIL - SOFT_LIMIT_KNEE;
    let y = SOFT_LIMIT_KNEE + range * (1.0 - (-(a - SOFT_LIMIT_KNEE) / range).exp());
    y.copysign(s)
}

fn synthesize_frame_inner(
    params: &MbeParams,
    state: &mut SynthState,
    phase_out: Option<&mut PhaseFrame>,
) -> [i16; FRAME_SAMPLES] {
    let (m_bar, s_e) = enhance(&params.amplitudes, params.omega_0, state.s_e);
    state.s_e = s_e;

    let v = &params.voiced;
    let unv = synthesize_unvoiced(params.omega_0, &m_bar, v, state);
    let vo = synthesize_voiced(params.omega_0, &m_bar, v, state, phase_out);

    let mut out = [0i16; FRAME_SAMPLES];
    for n in 0..FRAME_SAMPLES {
        let s = vo[n] + unv[n];
        let s = soft_limit(s);
        out[n] = s.round().clamp(-32768.0, 32767.0) as i16;
    }
    out
}
