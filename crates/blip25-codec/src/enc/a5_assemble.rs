//! a5 assembly: PCM front-half outputs -> the P/Q driver -> per-call
//! `(oldgen[8], s_new)`, and the cross-frame T/S ring threading that feeds
//! [`super::voicing_fixed::a_writer_band`] to produce `a5` (array A).
//!
//! ## Two structural facts that are easy to get wrong
//!
//! 1. **The Q builder is a SELF-energy of the y (exp_y) arm**, not a cross
//!    term: `q_builder(y_slice, y_slice, 2*count)`. Its consolidation exponent
//!    is `qe = q_builder_exp + 2*exp_y` -- the arm's own block-float exponent,
//!    squared.
//! 2. **The P builder's first (exp_x) arm is NOT the raw x-arm.** It is the
//!    block-normalized PHASOR-ROTATED x produced by the exp_x stage:
//!    `X = normalize( rotate(x_arm, phasor) )`, where the normalize shift is
//!    `-block_exponent(rotated)` and `exp_x` is exactly that stage's returned
//!    `sv + block_exponent(rotated)`.
//!
//! With both, [`pq_driver`] reproduces the captured per-call
//! `(oldgen[8], s_new)` on both conformance clips.
//!
//! ## The cross-frame ring (see [`RingState`])
//!
//! Per call to the ring routine (2 calls per frame, `c=0,1`):
//! 1. entry-shift (unconditional, `t_ring_entry_shift` semantics): `gen1 <-
//!    gen0`, `gen2 <- gen1`; `s4a <- s48`, `s4c <- s4a`. `gen0`/`s48` are
//!    untouched by the shift, then overwritten by the body.
//! 2. body: `gen0 = oldgen` (this call, from [`pq_driver`]), `s48 = s_new`.
//! 3. repair: iff the gate passes, `gen1 = t_ring_update(gen0, gen1, s_new,
//!    s4a)` (see [`super::t_ring`]).
//!
//! `a_writer_band` for output frame `f` reads AFTER frame `f`'s `c=1` call:
//! `si=1` reads `gen1` (the repair slot) with `S = s4a`; `si=2` reads `gen2`
//! with `S = s4c`. `a5 = [ a_writer_band(gen1, s4a) || a_writer_band(gen2,
//! s4c) ]` (two 8-word halves).
//!
//! ## Residuals (still needed for a fully-standalone wire)
//! * `exp_x` carries the front-half's ±1 block-exp LSB (15/18 exact on the
//!   voiced lever calls; the 3 misses cost voiced levers 195/197). mark is
//!   18/18.
//! * the repair gate `f0=ctx[0]` (the atan2 output) is not PCM-derived.
//! * `a5` must be voicing-GATED to `[0;16]` off the voiced frames. The raw
//!   `a_writer` output is nonzero on ~85 non-voiced frames -- the zeros-trap.

use super::block_consolidate::block_consolidate;
use super::block_exponent::block_exponent;
use super::main_loop::oldgen_and_s_new;
use super::pq_builder::{p_builder, q_builder};

/// Fixed per-band complex-bin partition summing to 127 (=254 words).
pub(crate) const A5_PARTITION: [usize; 8] = [13, 16, 16, 16, 16, 16, 16, 18];

/// The p-builder's exp_x arm: rotate the actual-spectrum x-arm by
/// `(ph0=cos, ph1=sin)`, then block-normalize the whole 254-word buffer.
/// Returns the 254-word normalized rotated buffer.
///
/// `x_arm` must be at least 254 words ([`super::pq_builder::assemble_x_arm`]).
pub(crate) fn rotate_normalize_x(x_arm: &[i16], ph0: i16, ph1: i16) -> Vec<i16> {
    let (cos, sin) = (ph0 as i32, ph1 as i32);
    let mut out = [0i16; 254];
    for k in 0..127 {
        let xre = x_arm[2 * k] as i32;
        let xim = x_arm[2 * k + 1] as i32;
        out[2 * k] = ((cos * xre - sin * xim) >> 16) as i16;
        out[2 * k + 1] = ((xre * sin + cos * xim) >> 16) as i16;
    }
    let sh = -(block_exponent(&out) as i32);
    let mut xn = vec![0i16; 254];
    for i in 0..254 {
        let v = out[i] as i32;
        xn[i] = if sh >= 0 {
            (v << sh) as i16
        } else {
            (v >> (-sh)) as i16
        };
    }
    xn
}

/// The P/Q driver: over the 8-band [`A5_PARTITION`], run
/// `p_builder` on `(X = rotate_normalize_x(x_arm), y_arm)` and `q_builder` as
/// a self-energy of `y_arm`, block-consolidate each into `P`/`Q` arrays, and
/// run [`oldgen_and_s_new`]. Returns `(oldgen[8], s_new)`.
///
/// Live-verified **398/398** (both files, all lever calls) vs the DLL's
/// captured per-call `oldgen`/`s_new`.
///
/// * `y_arm` = [`super::pq_builder::assemble_y_arm`] output (>=254 words).
/// * `x_arm` = [`super::pq_builder::assemble_x_arm`] output (>=254 words).
/// * `(ph0, ph1)` = the gate phasor.
/// * `exp_x` = the exp_x stage's returned exponent; `exp_y` =
///   [`super::pq_builder::assemble_y_arm_with_exp`]'s exponent.
pub(crate) fn pq_driver(
    y_arm: &[i16],
    x_arm: &[i16],
    ph0: i16,
    ph1: i16,
    exp_x: i16,
    exp_y: i16,
) -> ([i16; 8], i16) {
    let xn = rotate_normalize_x(x_arm, ph0, ph1);
    let mut pm = [0i16; 8];
    let mut pe = [0i16; 8];
    let mut qm = [0i16; 8];
    let mut qe = [0i16; 8];
    let mut woff = 0usize;
    for b in 0..8 {
        let s = A5_PARTITION[b];
        let w = 2 * s;
        let a2s = &xn[woff..woff + w];
        let a4s = &y_arm[woff..woff + w];
        let (pmb, peb) = p_builder(exp_x, exp_y, a2s, a4s, s as i16);
        pm[b] = pmb;
        pe[b] = peb;
        let (qraw, qeb) = q_builder(a4s, a4s, (2 * s) as i16);
        qm[b] = ((qraw as i32) >> 16) as i16;
        qe[b] = (qeb as i32 + 2 * (exp_y as i32)) as i16;
        woff += w;
    }
    let mut pc = [0i16; 8];
    let mut qc = [0i16; 8];
    let exp_p = block_consolidate(&pm, &pe, &mut pc);
    let exp_q = block_consolidate(&qm, &qe, &mut qc);
    oldgen_and_s_new(&pc, exp_p, &qc, exp_q)
}

/// The 3-generation T ring + 3-word S ring maintained by the ring routine.
/// `gen0/gen1/gen2` are the three T-ring generations; `s48/s4a/s4c` are the
/// three S-ring words. Feed one call per `push_call`; read `a_writer`
/// inputs after each frame's `c=1` call.
#[derive(Clone, Debug)]
pub(crate) struct RingState {
    pub gen0: [i16; 8],
    pub gen1: [i16; 8],
    pub gen2: [i16; 8],
    pub s48: i16,
    pub s4a: i16,
    pub s4c: i16,
}

impl Default for RingState {
    /// The DLL initializes the T ring to saturated `0x7fff`.
    fn default() -> Self {
        RingState {
            gen0: [32767; 8],
            gen1: [32767; 8],
            gen2: [32767; 8],
            s48: 0,
            s4a: 0,
            s4c: 0,
        }
    }
}

impl RingState {
    /// One ring-routine call: entry-shift, then body (`oldgen`/`s_new`), then
    /// optional repair (`gate_passed` — currently the caller must supply this;
    /// the `ctx[0]` derivation is the open residual). Uses
    /// [`super::t_ring::t_ring_update`] for the repair blend.
    pub fn push_call(&mut self, oldgen: [i16; 8], s_new: i16, gate_passed: bool) {
        // entry-shift (gen0/s48 untouched)
        self.gen2 = self.gen1;
        self.gen1 = self.gen0;
        self.s4c = self.s4a;
        self.s4a = self.s48;
        // body
        self.gen0 = oldgen;
        self.s48 = s_new;
        // repair
        if gate_passed {
            self.gen1 = super::t_ring::t_ring_update(self.gen0, self.gen1, s_new, self.s4a);
        }
    }

    /// `a_writer` inputs for the current frame (call after the frame's `c=1`
    /// `push_call`): `((T_si1, S_si1), (T_si2, S_si2))`.
    pub fn a_writer_inputs(&self) -> (([i16; 8], i16), ([i16; 8], i16)) {
        ((self.gen1, self.s4a), (self.gen2, self.s4c))
    }
}
