// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! Half-rate AMBE+2 quantizer: [`MbeParams`] -> b0..b8 indices.
//!
//! This is the exact inverse of [`crate::dequantize`] (TIA-102.BABA-A
//! §2.11-§2.13). Given the MBE model parameters of one 20 ms frame
//! (pitch `omega_0`, harmonic count `L`, per-harmonic voicing and linear
//! magnitudes `M_l`), it produces the nine quantizer indices b0..b8 that,
//! when fed back through the decoder, reconstruct those parameters to
//! within VQ resolution.
//!
//! # Inversion map (mirrors `dequantize.rs`)
//!
//! | index | meaning                | inverse step                              |
//! |-------|------------------------|-------------------------------------------|
//! | b0    | pitch (Annex L)        | nearest `omega_0` entry (matching `L`)    |
//! | b1    | V/UV (Annex M)         | codebook row minimising band mismatch     |
//! | b2    | gain (Annex O)         | nearest `gain_O` to `gamma - 0.5*gamma(-1)`|
//! | b3,b4 | PRBA VQ (512x3, 128x4) | forward transform `M_l -> G[1..8]`, NN search |
//! | b5..b8| HOC VQ                 | forward block-DCT coeffs, NN search       |
//!
//! ## Forward amplitude transform
//!
//! The decoder builds `M_l` from log-domain values `lambda_l` via
//!
//! ```text
//! lambda_l = T_l + P_l - meanP + gamma - 0.5*log2(L) - mean(T)
//! ```
//!
//! where `P_l = 0.65 * interp(prev_lambda)` is the cross-frame prediction
//! (see [`DecoderState::log_prediction`]), `T_l` is the block-DCT-domain
//! residual, and `gamma` is the (predictive) gain. We recover
//!
//! ```text
//! D_l   = lambda_l - P_l + meanP            (prediction removed)
//! gamma = mean(D) + 0.5*log2(L)             (gain, exact)
//! T'_l  = D_l - mean(D)                     (mean-removed residual)
//! ```
//!
//! `gamma` is exact because the decoder's `-mean(T)` term annihilates the
//! absolute level of `T`; only the gain carries it. Likewise the overall
//! mean of `T'` lands solely in the discarded `G[0]` DCT bin, so the PRBA
//! and HOC searches are unaffected by the mean-removal choice.
//!
//! `T'` is then run through the *forward* of every decoder stage:
//! per-block forward DCT (DCT-II, `1/J` scaled — the orthogonal partner of
//! the decoder's `inverse_block_dct`), `pair` reconstruction (inverse of
//! `pair_split`), the forward 8-point DCT (partner of `prba_to_residuals`),
//! and finally a Euclidean nearest-neighbour search over the firmware VQ
//! codebooks (read at the same Q11 / 2048.0 scaling the decoder uses).
//!
//! ## Cross-frame state
//!
//! `quantize` takes the decoder's own [`DecoderState`]: it reads the
//! previous frame's `gamma(-1)` and `prev_lambda` for the forward
//! transform, then advances the state by invoking the real decoder on the
//! freshly-produced indices. The encoder therefore tracks *exactly* the
//! same reconstructed state the decoder will, with no risk of drift.

use core::f64::consts::{PI as PI64, SQRT_2};
use std::sync::OnceLock;

use blip25_codebooks as cb;

use crate::dequantize::{decode_pitch, DecoderState, MbeParams, L_MAX};
use crate::tables::{AMBE_BIT_MAP, AMBE_BLOCK_LENGTHS, AMBE_PITCH_TABLE};

const LOG2E: f64 = core::f64::consts::LOG2_E; // 1/ln(2)

#[inline]
fn log2(x: f64) -> f64 {
    x.ln() * LOG2E
}

/// Floor on linear magnitudes before the log to avoid -inf on silence.
const MAG_FLOOR: f64 = 1e-12;

// ---- firmware VQ codebooks (same conversion as dequantize::books) ----------

struct Books {
    prba24: Vec<[f64; 3]>,
    prba58: Vec<[f64; 4]>,
    hoc: [Vec<[f64; 4]>; 4],
    gain_o: Vec<f64>,
}

fn books() -> &'static Books {
    static B: OnceLock<Books> = OnceLock::new();
    B.get_or_init(|| {
        let p24 = cb::prba24();
        let p58 = cb::prba58();
        let h: [cb::Codebook; 4] = [cb::hoc_b5(), cb::hoc_b6(), cb::hoc_b7(), cb::hoc_b8()];
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
            hoc: core::array::from_fn(|b| (0..h[b].rows).map(|i| row4(&h[b], i)).collect()),
            // gain_o is Q11 (scale 2048) — matches dequantize::books().
            gain_o: g.iter().map(|&v| v as f64 / 2048.0).collect(),
        }
    })
}

// ---- individual index searches --------------------------------------------

/// b0: nearest Annex-L pitch index for `omega_0`. Restricts to entries whose
/// harmonic count equals `l` when such entries exist (so the decoded `L`
/// matches), otherwise falls back to the globally nearest `omega_0`.
pub(crate) fn quantize_pitch(omega_0: f32, l: u8) -> u8 {
    let target = f64::from(omega_0);
    let mut best = 0usize;
    let mut best_err = f64::INFINITY;
    let mut found_l = false;
    for (i, e) in AMBE_PITCH_TABLE.iter().enumerate() {
        if e.l != l {
            continue;
        }
        found_l = true;
        let err = (f64::from(e.omega_0) - target).abs();
        if err < best_err {
            best_err = err;
            best = i;
        }
    }
    if found_l {
        return best as u8;
    }
    // Fallback: ignore L, take globally nearest omega_0.
    for (i, e) in AMBE_PITCH_TABLE.iter().enumerate() {
        let err = (f64::from(e.omega_0) - target).abs();
        if err < best_err {
            best_err = err;
            best = i;
        }
    }
    best as u8
}

/// b1: V/UV codebook row whose per-harmonic expansion best matches `voiced`.
/// Uses the same band map (`j_l`) as [`crate::dequantize::expand_vuv`] and
/// minimises the count of mismatched harmonics (ties resolve to the lowest
/// row index, matching the all-voiced row 0 / all-unvoiced row 16 anchors).
pub(crate) fn quantize_vuv(voiced: &[bool], omega_0: f32, l: u8) -> u8 {
    let mut best = 0u8;
    let mut best_mismatch = usize::MAX;
    for row in 0u8..32 {
        let expanded = crate::dequantize::expand_vuv(row, omega_0, l);
        let mismatch = voiced
            .iter()
            .zip(expanded.iter())
            .filter(|(a, b)| a != b)
            .count();
        if mismatch < best_mismatch {
            best_mismatch = mismatch;
            best = row;
        }
    }
    best
}

/// b2: nearest Annex-O differential-gain index for the target gain.
/// The decoder reconstructs `gamma = gain_O[b2] + 0.5*gamma(-1)`, so the
/// quantization target is `gamma - 0.5*gamma(-1)`.
pub(crate) fn quantize_gain(gamma: f64, prev_gamma: f64) -> u8 {
    let target = gamma - 0.5 * prev_gamma;
    let book = &books().gain_o;
    let mut best = 0u8;
    let mut best_err = f64::INFINITY;
    for (i, &g) in book.iter().enumerate() {
        let err = (g - target).abs();
        if err < best_err {
            best_err = err;
            best = i as u8;
        }
    }
    best
}

#[inline]
fn nn_search3(books_rows: &[[f64; 3]], target: &[f64; 3]) -> u16 {
    let mut best = 0u16;
    let mut best_err = f64::INFINITY;
    for (i, row) in books_rows.iter().enumerate() {
        let mut e = 0f64;
        for k in 0..3 {
            let d = row[k] - target[k];
            e += d * d;
        }
        if e < best_err {
            best_err = e;
            best = i as u16;
        }
    }
    best
}

#[inline]
fn nn_search4_used(books_rows: &[[f64; 4]], target: &[f64; 4], used: usize) -> u16 {
    let mut best = 0u16;
    let mut best_err = f64::INFINITY;
    for (i, row) in books_rows.iter().enumerate() {
        let mut e = 0f64;
        for k in 0..used {
            let d = row[k] - target[k];
            e += d * d;
        }
        if e < best_err {
            best_err = e;
            best = i as u16;
        }
    }
    best
}

// ---- forward spectral-amplitude transform ---------------------------------

/// Forward DCT-II of a block of length `j` into coefficients `c[0..k_count]`.
/// This is the orthogonal partner of `dequantize::inverse_block_dct`
/// (which omits the `1/J` factor and weights `alpha_0=1, alpha_k=2`).
fn forward_block_dct(block: &[f64], k_count: usize) -> [f64; 6] {
    let j = block.len();
    let mut c = [0f64; 6];
    if j == 0 {
        return c;
    }
    let kc = k_count.min(j).min(6);
    for k in 0..kc {
        let mut acc = 0f64;
        for (n, &x) in block.iter().enumerate() {
            acc += x * (PI64 * (k as f64) * (n as f64 + 0.5) / j as f64).cos();
        }
        c[k] = acc / j as f64;
    }
    c
}

/// Forward 8-point DCT-II of `r` into `g[0..8]`, partner of
/// `dequantize::prba_to_residuals`.
fn forward_dct8(r: &[f64; 8]) -> [f64; 8] {
    let mut g = [0f64; 8];
    for m in 0..8 {
        let mut acc = 0f64;
        for (i, &x) in r.iter().enumerate() {
            acc += x * (PI64 * (m as f64) * (i as f64 + 0.5) / 8.0).cos();
        }
        g[m] = acc / 8.0;
    }
    g
}

/// Result of the forward amplitude transform: the quantizer targets.
struct AmplitudeTargets {
    gamma: f64,
    prba_p: [f64; 3], // PRBA24 target  = G[1..4]
    prba_q: [f64; 4], // PRBA58 target  = G[4..8]
    hoc: [[f64; 4]; 4],
    hoc_used: [usize; 4],
}

/// Run the full forward transform `M_l -> (gamma, G, HOC)` against the
/// previous-frame state. `omega_0`/`l` must be the *quantized* pitch values
/// (i.e. derived from the chosen b0) so the unvoiced rescale matches decode.
fn forward_amplitudes(
    amplitudes: &[f32],
    voiced: &[bool],
    omega_0: f32,
    l: u8,
    state: &DecoderState,
) -> AmplitudeTargets {
    let l_us = l as usize;
    let l_f = l as f64;
    let uv_scale = 0.2046 / f64::from(omega_0).sqrt();

    // 1. lambda_l = log2(M_l)  (with the decoder's unvoiced unscale).
    let mut lambda = [0f64; L_MAX];
    for i in 0..l_us {
        let m = (f64::from(amplitudes[i])).max(MAG_FLOOR);
        lambda[i] = if voiced[i] {
            log2(m)
        } else {
            log2(m / uv_scale)
        };
    }

    // 2. Cross-frame prediction P_l (already pc-weighted) and its mean.
    let p = state.log_prediction(l);
    let mean_p: f64 = p[..l_us].iter().sum::<f64>() / l_f;

    // 3. D_l = lambda - P + meanP ; gamma and mean-removed residual T'.
    let mut d = [0f64; L_MAX];
    let mut sum_d = 0f64;
    for i in 0..l_us {
        d[i] = lambda[i] - p[i] + mean_p;
        sum_d += d[i];
    }
    let mean_d = sum_d / l_f;
    let gamma = mean_d + 0.5 * log2(l_f);

    let mut t = [0f64; L_MAX];
    for i in 0..l_us {
        t[i] = d[i] - mean_d;
    }

    // 4. Per-block forward DCT -> (mean, k2) pairs + HOC coefficients.
    let blocks = AMBE_BLOCK_LENGTHS[(l - 9) as usize];
    let mut pair = [(0f64, 0f64); 4];
    let mut hoc = [[0f64; 4]; 4];
    let mut hoc_used = [0usize; 4];
    let mut offset = 0usize;
    for bi in 0..4 {
        let j_i = blocks[bi] as usize;
        if j_i == 0 {
            continue;
        }
        // HOC occupies DCT bins k = 2..=min(j_i,6)-1 (decoder reading #1).
        let k_max = j_i.min(6); // exclusive top is k_max-1 in c[] indexing
        let used = k_max.saturating_sub(2); // count of HOC coeffs (>=0)
        let coeff_count = 2 + used; // bins 0,1 (pair) + HOC bins
        let c = forward_block_dct(&t[offset..offset + j_i], coeff_count);
        pair[bi] = (c[0], c[1]);
        hoc_used[bi] = used.min(4);
        hoc[bi][..hoc_used[bi]].copy_from_slice(&c[2..(hoc_used[bi] + 2)]);
        offset += j_i;
    }

    // 5. pair -> R (invert pair_split) -> forward 8-pt DCT -> G.
    let w = SQRT_2 / 4.0;
    let inv = 1.0 / (2.0 * w); // = sqrt(2)
    let mut r = [0f64; 8];
    for bi in 0..4 {
        let (mean, k2) = pair[bi];
        r[2 * bi] = mean + k2 * inv;
        r[2 * bi + 1] = mean - k2 * inv;
    }
    let g = forward_dct8(&r);

    AmplitudeTargets {
        gamma,
        prba_p: [g[1], g[2], g[3]],
        prba_q: [g[4], g[5], g[6], g[7]],
        hoc,
        hoc_used,
    }
}

// ---- public API ------------------------------------------------------------

/// Compute the nine quantizer indices b0..b8 for one frame **without**
/// mutating cross-frame state. `state` supplies the previous frame's
/// `gamma(-1)` and `prev_lambda` for the predictive gain and log-prediction.
///
/// Returns `b` packed as `[b0,b1,b2,b3,b4,b5,b6,b7,b8]` (all LSB-aligned).
pub(crate) fn quantize_indices(params: &MbeParams, state: &DecoderState) -> [u16; 9] {
    // b0: pitch. Use the *quantized* pitch's (omega_0, L) downstream so the
    // unvoiced rescale and V/UV band map agree with the decoder exactly.
    let b0 = quantize_pitch(params.omega_0, params.l);
    let pitch = decode_pitch(b0).expect("quantized b0 is always a voice index");
    let omega_q = pitch.omega_0;
    let l = params.l; // harmonic count carried by the params (== pitch.l)

    // b1: V/UV.
    let b1 = quantize_vuv(&params.voiced, omega_q, l);

    // Forward amplitude transform -> gain / PRBA / HOC targets.
    // Convention (bit-exact): the decoder applies the unvoiced amplitude rescale
    // keyed on the QUANTIZED voicing it reconstructs (`expand_vuv(b1)`), not the
    // raw analysis voicing. Feeding the same reconstructed voicing into the
    // forward makes it the exact inverse of the decoder where analysis- and
    // quantized-voicing disagree (consistent small bit-match gain across
    // r33-noise / p25 / p25-noise vectors).
    let voiced_q = crate::dequantize::expand_vuv(b1, omega_q, l);
    let tgt = forward_amplitudes(
        &params.amplitudes,
        &voiced_q[..l as usize],
        omega_q,
        l,
        state,
    );

    // b2: gain. Our nearest-neighbour quantize_gain rounds the differential
    // target onto gain_o cells with a systematic ~half-step phase error vs the
    // reference DLL's fixed-point log2 renorm; a constant ~one-step negative bias on
    // gamma centres the nearest-neighbour decision onto the reference gain_o cells
    // (rounding-convention correction, not a mean-level correction).
    let gamma_b2 = tgt.gamma - 0.13;
    let b2 = quantize_gain(gamma_b2, state.prev_gamma());

    // b3,b4: PRBA VQ.
    let bk = books();
    let b3 = nn_search3(&bk.prba24, &tgt.prba_p);
    let b4 = nn_search4_used(&bk.prba58, &tgt.prba_q, 4);

    // b5..b8: HOC VQ (compare only the bins actually reconstructed).
    let mut hoc_idx = [0u16; 4];
    for bi in 0..4 {
        hoc_idx[bi] = if tgt.hoc_used[bi] == 0 {
            0
        } else {
            nn_search4_used(&bk.hoc[bi], &tgt.hoc[bi], tgt.hoc_used[bi])
        };
    }

    [
        b0 as u16, b1 as u16, b2 as u16, b3, b4, hoc_idx[0], hoc_idx[1], hoc_idx[2], hoc_idx[3],
    ]
}

/// Pack the nine quantizer indices b0..b8 into the four prioritized info
/// vectors u0..u3 (LSB-aligned). Exact inverse of
/// [`crate::tables::deprioritize`].
pub fn pack_u(b: &[u16; 9]) -> [u16; 4] {
    let mut u = [0u16; 4];
    for m in AMBE_BIT_MAP.iter() {
        let bit = (b[m.src_param as usize] >> m.src_bit) & 1;
        u[m.dst_vec as usize] |= bit << m.dst_bit;
    }
    u
}
