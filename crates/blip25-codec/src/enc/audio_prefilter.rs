//! The encoder's raw-audio pre-filter — a persistent-state, 3-stage cascaded
//! recursive (IIR) fixed-point filter applied to raw int16 PCM *before* the
//! FFT/spectrum front-end sees it.
//!
//! ## What this computes
//!
//! Given a persistent 6-`i32` filter state (`ctx` offsets `0x0`, `0x4`,
//! `0x8`, `0xc`, `0x10`, `0x14` in the real DLL) and `count` raw int16
//! input samples, produces `count` int16 output samples plus the updated
//! state, via:
//!
//! 1. **Stage 0**: each input sample is upscaled to a wider fixed-point
//!    representation (`sample << 14`, exact — the shift never overflows
//!    since `sample` is a 16-bit quantity).
//! 2. **Stage 1**: a one-tap recursive predictor,
//!    `acc = K(state1) - state0 + x`, saturated to `i32` — a genuine IIR
//!    section (`state1`'s own "previous" role is fed by the *previous
//!    iteration's own stage-1 output*, not a fixed value; `state0` is a
//!    1-sample delay line of the raw upscaled input).
//! 3. **Stage 2**: continues accumulating onto the *same, unsaturated*
//!    64-bit-range accumulator from stage 1 (a real two-stage cascade
//!    sharing one accumulator, not two independent filters) with a
//!    2-tap shift-register term plus two more `K`-style scaled terms,
//!    saturated to `i32` again.
//! 4. **Stage 3**: a separate pass over stage 1+2's output buffer, one
//!    more `K`-style one-pole term, a saturating `<<2`
//!    (`shift_sat`), then a classic
//!    round-and-saturate to int16 (`(x + 0x8000) >> 16` with overflow
//!    detection).
//!
//! All the scale factors are the *same family* of hi/lo-split fixed-point
//! multiply (`K(v, hi_coef, lo_coef) = v * lo_coef / 32768`, computed as
//! `(v>>16)*hi_coef + ((v&0xffff)*lo_coef) >> 15` with `hi_coef == 2*lo_coef`
//! in every case), but there are **four** independently-quantized coefficient
//! pairs across the three stages, not three: 0.9427185 (stage 1), then
//! **0.9697266** and 0.9428406 (stage 2, which applies two), then 0.9426880
//! (stage 3). The stage-2 `ka` coefficient is not in the 0.94 family at all.
//! These are separately-tuned filter sections, not a copy/paste of one
//! constant. The hi/lo split is not equivalent to a 32x32 multiply — do not
//! "simplify" it.
//!
//! ## Caller-side state cadence
//!
//! **The state just persists, indefinitely, with no reset logic anywhere** --
//! within a frame and across frames alike. [`super::Encoder`] does exactly this
//! (see `prefilter_state` there): zero-initialized once at construction, to
//! match `Reset()`, and carried forward every frame thereafter.
//!
//! Instrumentation that writes to `encoder_buf+4` will appear to show the state
//! resetting to zero or jumping to unrelated values between frames. That address
//! IS this filter's persistent state, and the raw-audio input path is
//! `state_buf`, not `encoder_buf+4` -- a harness that confuses the two overwrites
//! the state with the frame's own PCM bytes. The "jump" values are literally the
//! frame's first 12 `i16` samples reinterpreted as 6 `i32`s. Do not model a reset.

const MASK32: u32 = 0xFFFF_FFFF;

use crate::fixops::u32r::{imul32, s32, sar32};

#[inline]
fn u32v(x: i32) -> u32 {
    (x as u32) & MASK32
}

/// `K(v) = (v>>16 as i16)*hi_coef + ((v&0xffff)*lo_coef)>>15`, all 32-bit
/// truncating fixed-point ops — algebraically `v * lo_coef / 32768` given
/// `hi_coef == 2*lo_coef`, but implemented exactly as the reference does it
/// (hi/lo 16x16 split, no 32x32 multiply available in this build).
fn k_scale(v: u32, hi_coef: i32, lo_coef: i32) -> u32 {
    let hi = s32(sar32(v, 16));
    let term1 = imul32(hi, hi_coef);
    let lo = (v & 0xFFFF) as i32;
    let term2 = imul32(lo, lo_coef);
    let term2s = sar32(term2, 15);
    u32v(s32(term1).wrapping_add(s32(term2s)))
}

/// Saturate an exact (unbounded-range, but always well within i64) signed
/// accumulator to i32, matching the reference's 64-bit-pair saturation branch
/// (`js`/`jg`/`cmp $0x7fffffff`/`cmp $0x80000000`).
fn sat_to_i32(acc: i64) -> i32 {
    if acc > i32::MAX as i64 {
        i32::MAX
    } else if acc < i32::MIN as i64 {
        i32::MIN
    } else {
        acc as i32
    }
}

/// Persistent filter state (`ctx` offsets `0x0..0x18` in the real DLL).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefilterState {
    pub s: [i32; 6],
}

/// Saturating left shift, called with a fixed shift amount of `2`: shift
/// `value` left by `shift` with saturation (overflow detected by checking
/// the top `shift+1` bits are all equal to the sign bit).
fn shift_sat_left(value: u32, shift: u32) -> u32 {
    let mask = u32v((-1i32) << (31 - shift as i32));
    let top = value & mask;
    let overflow = if s32(value) >= 0 {
        top != 0
    } else {
        top != mask
    };
    if overflow {
        if s32(value) < 0 {
            0x8000_0000
        } else {
            0x7FFF_FFFF
        }
    } else {
        value.wrapping_shl(shift)
    }
}

/// Run the pre-filter over `count` raw int16 samples starting at `state`,
/// returning the produced int16 samples and the updated state. Direct,
/// bit-exact port of the DLL routine's Blocks A (upscale), B (2-stage IIR),
/// C (final one-pole + round/saturate to int16).
pub fn prefilter(state: &PrefilterState, src: &[i16]) -> (Vec<i16>, PrefilterState) {
    let [c0, c1, c2, c3, c4, c5] = state.s;
    let mut x_prev = c0 as u32;
    let mut sr1 = c2 as u32; // ctx[8] delay slot
    let mut sr2 = c4 as u32; // ctx[0x10] delay slot
    let mut s2_delay = c5 as u32; // ctx[0x14] delay slot
    let mut s1_state = c1 as u32; // ctx[4] role, evolves to prior stage-1 output
    let mut s2_state = c3 as u32; // ctx[0xc] role, evolves to prior stage-2 output

    let mut s2buf: Vec<i32> = Vec::with_capacity(src.len());

    for &samp in src {
        let x = u32v((samp as i32) << 14); // Block A: exact <<14 upscale, no overflow

        // ---- Stage 1 ----
        let k1 = k_scale(s1_state, 0xf156, 0x78ab);
        let mut acc: i64 = i64::from(s32(k1));
        acc -= i64::from(s32(x_prev));
        s1_state = x; // becomes this sample's raw X
        acc += i64::from(s32(s1_state));
        x_prev = x; // ctx[0] delay-line update
        let s1_sat = sat_to_i32(acc);
        s1_state = u32v(s1_sat); // now holds stage-1's own saturated output

        // ---- Stage 2 (continues the SAME unsaturated accumulator) ----
        acc += i64::from(s32(sr2)) - 2 * i64::from(s32(sr1));
        sr2 = sr1;
        sr1 = s1_state;

        let ka = k_scale(s2_state, 0xf840, 0x7c20);
        acc += 2 * i64::from(s32(ka));

        let k2 = k_scale(s2_delay, 0xf15e, 0x78af);
        acc -= i64::from(s32(k2));
        s2_delay = s2_state;

        let s2_sat = sat_to_i32(acc);
        s2_state = u32v(s2_sat);

        s2buf.push(s2_sat);
    }

    // ---- Stage 3: separate pass, K-scale + saturating <<2 + round/sat to i16 ----
    let mut out = Vec::with_capacity(src.len());
    for &v in &s2buf {
        let v = v as u32;
        let k3 = k_scale(v, 0xf154, 0x78aa);
        let shifted = shift_sat_left(k3, 2);
        let val = shifted;
        let biased = val.wrapping_add(0x8000);
        let ovf = (biased ^ val) & biased;
        let result16 = if s32(ovf) >= 0 {
            sar32(biased, 16)
        } else {
            let clamp = if s32(val) < 0 {
                0x8000_0000u32
            } else {
                0x7FFF_FFFFu32
            };
            sar32(clamp, 16)
        };
        out.push((result16 & 0xFFFF) as u16 as i16);
    }

    let post = PrefilterState {
        s: [
            s32(x_prev),
            s32(s1_state),
            s32(s1_state),
            s32(s2_state),
            s32(sr2),
            s32(s2_delay),
        ],
    };
    (out, post)
}
