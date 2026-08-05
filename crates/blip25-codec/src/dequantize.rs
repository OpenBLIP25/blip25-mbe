// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! Half-rate AMBE+2 dequantization: b0..b8 -> MbeParams
//! (TIA-102.BABA-A §2.11-§2.13).
//!
//! The VQ codebooks (PRBA24/PRBA58/HOC b5..b8, gain Annex-O) are loaded
//! bit-exact from the firmware via the `blip25_codebooks` crate.

use core::f64::consts::{LN_2, PI as PI64, SQRT_2};
use std::sync::OnceLock;

use blip25_codebooks as cb;

use crate::tables::{deprioritize, AMBE_BLOCK_LENGTHS, AMBE_PITCH_TABLE, AMBE_VUV_CODEBOOK};

/// Number of harmonics ceiling (BABA-A §1.1).
pub(crate) const L_MAX: usize = 56;
/// Min harmonics.
pub(crate) const L_MIN: u8 = 9;
/// Highest valid voice pitch index; [120,255] are tone/erasure.
pub(crate) const PITCH_INDEX_MAX: u8 = 119;
/// Initial L(-1) for the half-rate predictor (§2.13 Annex A).
pub(crate) const INIT_PREV_L: u8 = 15;
/// Max per-block length in Annex N.
pub(crate) const MAX_BLOCK_SIZE: usize = 17;

/// MBE model parameters for one 20 ms frame — the synthesis boundary.
#[derive(Clone, Debug)]
pub struct MbeParams {
    /// Raw 7-bit pitch index b0 (0..=119 for voice).
    pub b0: u8,
    pub omega_0: f32,
    pub l: u8,
    pub voiced: Vec<bool>,
    pub amplitudes: Vec<f32>,
}

/// Error from the dequant pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// b0 in [120, 255] — tone/erasure, not a voice frame.
    BadPitch,
}

// ---- lazily-converted firmware codebooks (raw int16 -> f64 rows) ----------

struct Books {
    prba24: Vec<[f64; 3]>,
    prba58: Vec<[f64; 4]>,
    hoc_b5: Vec<[f64; 4]>,
    hoc_b6: Vec<[f64; 4]>,
    hoc_b7: Vec<[f64; 4]>,
    hoc_b8: Vec<[f64; 4]>,
    gain_o: Vec<f64>,
}

fn books() -> &'static Books {
    static B: OnceLock<Books> = OnceLock::new();
    B.get_or_init(|| {
        let p24 = cb::prba24();
        let p58 = cb::prba58();
        let h5 = cb::hoc_b5();
        let h6 = cb::hoc_b6();
        let h7 = cb::hoc_b7();
        let h8 = cb::hoc_b8();
        let g = cb::gain_o();

        let row3 = |c: &cb::Codebook, i: usize| -> [f64; 3] {
            let r = c.raw_row(i);
            [
                r[0] as f64 / c.scale as f64,
                r[1] as f64 / c.scale as f64,
                r[2] as f64 / c.scale as f64,
            ]
        };
        let row4 = |c: &cb::Codebook, i: usize| -> [f64; 4] {
            let r = c.raw_row(i);
            [
                r[0] as f64 / c.scale as f64,
                r[1] as f64 / c.scale as f64,
                r[2] as f64 / c.scale as f64,
                r[3] as f64 / c.scale as f64,
            ]
        };

        Books {
            prba24: (0..p24.rows).map(|i| row3(&p24, i)).collect(),
            prba58: (0..p58.rows).map(|i| row4(&p58, i)).collect(),
            hoc_b5: (0..h5.rows).map(|i| row4(&h5, i)).collect(),
            hoc_b6: (0..h6.rows).map(|i| row4(&h6, i)).collect(),
            hoc_b7: (0..h7.rows).map(|i| row4(&h7, i)).collect(),
            hoc_b8: (0..h8.rows).map(|i| row4(&h8, i)).collect(),
            // gain_o is Q11 (scale 2048).
            gain_o: g.iter().map(|&v| v as f64 / 2048.0).collect(),
        }
    })
}

// ---- DCT cosine LUT --------------------------------------------------------

fn dct_cos(j_i: usize) -> &'static [f64] {
    static TABLES: OnceLock<[Vec<f64>; MAX_BLOCK_SIZE + 1]> = OnceLock::new();
    let tables = TABLES.get_or_init(|| {
        core::array::from_fn(|j| {
            if j == 0 {
                Vec::new()
            } else {
                let mut t = vec![0f64; j * j];
                for k_0 in 0..j {
                    for j_0 in 0..j {
                        t[k_0 * j + j_0] =
                            (PI64 * (k_0 as f64) * (j_0 as f64 + 0.5) / j as f64).cos();
                    }
                }
                t
            }
        })
    });
    &tables[j_i]
}

// ---- cross-frame decoder state --------------------------------------------

/// Cross-frame state for the half-rate decoder (§2.13 Annex A).
#[derive(Clone, Debug)]
pub struct DecoderState {
    prev_lambda: [f64; L_MAX + 2],
    prev_l: u8,
    prev_gamma: f64,
}

impl DecoderState {
    pub fn new() -> Self {
        Self {
            prev_lambda: [1.0; L_MAX + 2],
            prev_l: INIT_PREV_L,
            prev_gamma: 0.0,
        }
    }
    fn prev_lambda_at(&self, l: u8) -> f64 {
        if l == 0 {
            return self.prev_lambda[1];
        }
        let idx = (l as usize).min(self.prev_l as usize);
        self.prev_lambda[idx]
    }
    pub fn previous_l(&self) -> u8 {
        self.prev_l
    }

    /// Reconstructed gain of the previous frame, gamma(-1) (Eq. 168).
    /// Needed by the encoder to form the differential gain target.
    pub fn prev_gamma(&self) -> f64 {
        self.prev_gamma
    }

    /// Log-domain prediction term used by [`apply_log_prediction`], exposed
    /// for the encoder's forward transform. Returns, per current harmonic
    /// `l_h = 1..=l`, the value `0.65 * ((1-delta)*prev_lo + delta*prev_hi)`
    /// (i.e. `P_l`, the pc-weighted interpolation of the previous frame's
    /// log magnitudes). Index `l_h-1` holds the term for harmonic `l_h`.
    pub fn log_prediction(&self, l: u8) -> [f64; L_MAX] {
        let pc = 0.65;
        let l_curr = f64::from(l);
        let l_prev = f64::from(self.prev_l);
        let mut p = [0f64; L_MAX];
        for l_h in 1..=l {
            let k_l = l_prev * f64::from(l_h) / l_curr;
            let k_floor = k_l.floor();
            let delta = k_l - k_floor;
            let log_lo = self.prev_lambda_at(k_floor as u8);
            let log_hi = self.prev_lambda_at(k_floor as u8 + 1);
            p[(l_h - 1) as usize] = pc * ((1.0 - delta) * log_lo + delta * log_hi);
        }
        p
    }
}

impl Default for DecoderState {
    fn default() -> Self {
        Self::new()
    }
}

// ---- pitch / V-UV ----------------------------------------------------------

/// Decoded pitch info.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PitchInfo {
    pub omega_0: f32,
    pub l: u8,
}

/// Decode the 7-bit pitch index b0 via Annex L. None for tone/erasure.
pub fn decode_pitch(b0: u8) -> Option<PitchInfo> {
    if b0 > PITCH_INDEX_MAX {
        return None;
    }
    let e = &AMBE_PITCH_TABLE[b0 as usize];
    Some(PitchInfo {
        omega_0: e.omega_0,
        l: e.l,
    })
}

/// Expand the 5-bit V/UV index b1 into per-harmonic voicing
/// (j_l = floor(l*16*omega0/2pi), clamped [0,7]).
pub fn expand_vuv(b1: u8, omega_0: f32, l: u8) -> [bool; L_MAX] {
    let codebook = &AMBE_VUV_CODEBOOK[b1 as usize];
    let omega_0 = f64::from(omega_0);
    let mut out = [false; L_MAX];
    for l_h in 1..=l {
        let j = (f64::from(l_h) * 16.0 * omega_0 / (2.0 * PI64)).floor() as i32;
        let j = j.clamp(0, 7) as usize;
        out[(l_h - 1) as usize] = codebook[j];
    }
    out
}

// ---- gain / PRBA / HOC -----------------------------------------------------

/// gamma(0) = delta_gamma + 0.5*gamma(-1) (Eq. 168).
pub(crate) fn decode_gain(b2: u8, prev_gamma: f64) -> f64 {
    books().gain_o[b2 as usize] + 0.5 * prev_gamma
}

/// Combine PRBA codebooks into the 8-element G vector. G1 = 0.
pub(crate) fn decode_prba_vector(b3: u16, b4: u8) -> [f64; 8] {
    let p = books().prba24[b3 as usize];
    let q = books().prba58[b4 as usize];
    [0.0, p[0], p[1], p[2], q[0], q[1], q[2], q[3]]
}

/// Fixed 8-point inverse DCT (Eq. 169-170): G -> R.
pub(crate) fn prba_to_residuals(g: &[f64; 8]) -> [f64; 8] {
    let mut r = [0f64; 8];
    for i_0 in 0..8 {
        let i_half = i_0 as f64 + 0.5;
        let mut acc = 0f64;
        for m_0 in 0..8 {
            let alpha = if m_0 == 0 { 1.0 } else { 2.0 };
            let arg = PI64 * (m_0 as f64) * i_half / 8.0;
            acc += alpha * g[m_0] * arg.cos();
        }
        r[i_0] = acc;
    }
    r
}

/// Pair-wise split (Eq. 171-178) of R into per-block (mean, k2).
pub(crate) fn pair_split(r: &[f64; 8]) -> [(f64, f64); 4] {
    let w = SQRT_2 / 4.0;
    [
        ((r[0] + r[1]) / 2.0, w * (r[0] - r[1])),
        ((r[2] + r[3]) / 2.0, w * (r[2] - r[3])),
        ((r[4] + r[5]) / 2.0, w * (r[4] - r[5])),
        ((r[6] + r[7]) / 2.0, w * (r[6] - r[7])),
    ]
}

/// Populate the 4-block DCT coefficient matrix (Reading #1 of Eq. 179).
pub(crate) fn assemble_hoc_matrix(
    pair: &[(f64, f64); 4],
    b5: u8,
    b6: u8,
    b7: u8,
    b8: u8,
    blocks: &[u8; 4],
) -> [[f64; MAX_BLOCK_SIZE]; 4] {
    let bk = books();
    let hoc: [[f64; 4]; 4] = [
        bk.hoc_b5[b5 as usize],
        bk.hoc_b6[b6 as usize],
        bk.hoc_b7[b7 as usize],
        bk.hoc_b8[b8 as usize],
    ];
    let mut c = [[0f64; MAX_BLOCK_SIZE]; 4];
    for i in 0..4 {
        c[i][0] = pair[i].0;
        c[i][1] = pair[i].1;
        let j_i = blocks[i] as usize;
        let k_max = j_i.min(6);
        for k in 3..=k_max {
            c[i][k - 1] = hoc[i][k - 3];
        }
    }
    c
}

/// Per-block inverse DCT (Eq. 180-181), concatenated into T_l.
pub(crate) fn inverse_block_dct(c: &[[f64; MAX_BLOCK_SIZE]; 4], blocks: &[u8; 4]) -> [f64; L_MAX] {
    let mut t = [0f64; L_MAX];
    let mut l_offset = 0usize;
    for i in 0..4 {
        let j_i = blocks[i] as usize;
        if j_i == 0 {
            continue;
        }
        let cos_tab = dct_cos(j_i);
        for j_0 in 0..j_i {
            let mut acc = 0f64;
            for k_0 in 0..j_i {
                let alpha = if k_0 == 0 { 1.0 } else { 2.0 };
                acc += alpha * c[i][k_0] * cos_tab[k_0 * j_i + j_0];
            }
            t[l_offset + j_0] = acc;
        }
        l_offset += j_i;
    }
    t
}

/// Log-magnitude prediction (Eq. 182-187), rho = 0.65.
pub(crate) fn apply_log_prediction(
    t: &[f64; L_MAX],
    l: u8,
    gamma: f64,
    state: &DecoderState,
) -> [f64; L_MAX + 2] {
    let mut lambda = [0f64; L_MAX + 2];
    let l_curr = f64::from(l);
    let l_prev = f64::from(state.prev_l);

    let t_sum: f64 = t[..l as usize].iter().sum();
    let gamma_intercept = gamma - 0.5 * l_curr.log2() - t_sum / l_curr;

    let mut mean_sum = 0f64;
    for lambda_idx in 1..=l {
        let k_l = l_prev * f64::from(lambda_idx) / l_curr;
        let k_floor = k_l.floor();
        let delta = k_l - k_floor;
        let log_lo = state.prev_lambda_at(k_floor as u8);
        let log_hi = state.prev_lambda_at(k_floor as u8 + 1);
        mean_sum += (1.0 - delta) * log_lo + delta * log_hi;
    }
    let mean = mean_sum / l_curr;

    let pc = 0.65;
    for l_h in 1..=l {
        let k_l = l_prev * f64::from(l_h) / l_curr;
        let k_floor = k_l.floor();
        let delta = k_l - k_floor;
        let log_lo = state.prev_lambda_at(k_floor as u8);
        let log_hi = state.prev_lambda_at(k_floor as u8 + 1);
        lambda[l_h as usize] =
            t[(l_h - 1) as usize] + pc * (1.0 - delta) * log_lo + pc * delta * log_hi - pc * mean
                + gamma_intercept;
    }
    lambda
}

/// Convert Lambda to linear M (Eq. 188); unvoiced rescale 0.2046/sqrt(omega0).
pub(crate) fn compute_m_tilde(
    lambda: &[f64; L_MAX + 2],
    voiced: &[bool],
    omega_0: f32,
) -> [f32; L_MAX] {
    let l = voiced.len();
    let mut m = [0f32; L_MAX];
    let uv_scale = 0.2046 / f64::from(omega_0).sqrt();
    for l_h in 1..=l {
        let linear = (LN_2 * lambda[l_h]).exp();
        m[l_h - 1] = if voiced[l_h - 1] {
            linear as f32
        } else {
            (uv_scale * linear) as f32
        };
    }
    m
}

/// Run the half-rate dequantization end-to-end: (u0..u3, state) -> MbeParams.
pub fn dequantize(u: &[u16; 4], state: &mut DecoderState) -> Result<MbeParams, DecodeError> {
    dequantize_on_grid(u, state, None)
}

/// [`dequantize`] with the harmonic grid supplied instead of decoded from `b̂₀`.
///
/// `grid = Some((L, omega_0))` is the packer tone branch's `(56, 0x1079)`
/// override: on those frames `b̂₀` is a tone index, so the pitch table is not
/// what the frame was analysed — or is synthesised — on. `None` is the ordinary
/// path and decodes the grid from `b̂₀`.
pub fn dequantize_on_grid(
    u: &[u16; 4],
    state: &mut DecoderState,
    grid: Option<(u8, f32)>,
) -> Result<MbeParams, DecodeError> {
    let b = deprioritize(u);
    let b0 = b[0] as u8;
    let (l, omega_0) = match grid {
        Some((l, w)) => (l, w),
        None => {
            let pitch = decode_pitch(b0).ok_or(DecodeError::BadPitch)?;
            (pitch.l, pitch.omega_0)
        }
    };
    let pitch = PitchInfo { omega_0, l };

    let voiced = expand_vuv(b[1] as u8, pitch.omega_0, l);
    let gamma = decode_gain(b[2] as u8, state.prev_gamma);

    let g = decode_prba_vector(b[3], b[4] as u8);
    let r = prba_to_residuals(&g);
    let pair = pair_split(&r);

    let blocks = AMBE_BLOCK_LENGTHS[(l - 9) as usize];
    let c = assemble_hoc_matrix(
        &pair, b[5] as u8, b[6] as u8, b[7] as u8, b[8] as u8, &blocks,
    );
    let t = inverse_block_dct(&c, &blocks);
    let lambda = apply_log_prediction(&t, l, gamma, state);
    let m_tilde = compute_m_tilde(&lambda, &voiced[..l as usize], pitch.omega_0);

    let params = MbeParams {
        b0,
        omega_0: pitch.omega_0,
        l,
        voiced: voiced[..l as usize].to_vec(),
        amplitudes: m_tilde[..l as usize].to_vec(),
    };

    // Advance cross-frame state (voice frame).
    state.prev_lambda[1..(l as usize + 1)].copy_from_slice(&lambda[1..(l as usize + 1)]);
    for l_h in (l as usize + 1)..=L_MAX + 1 {
        state.prev_lambda[l_h] = lambda[l as usize];
    }
    state.prev_l = l;
    state.prev_gamma = gamma;

    Ok(params)
}
