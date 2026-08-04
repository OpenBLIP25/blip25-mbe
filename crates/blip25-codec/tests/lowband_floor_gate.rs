//! Regression guard for `Encoder::set_lowband_floor` (the off-by-default gated
//! low-band spectral floor on the Route A amplitude path):
//!
//! (a) the flag off must not change a single shipped output bit,
//! (b) the flag on with an all-gate-off `b1` sequence must also not change a
//!     bit — the gate must suppress the floor loop AND its recursive `l0` tail,
//!     and the un-biased delta carry-back must be exactly zero,
//! (c) the flag on with an all-gate-on `b1` sequence must change bits, so (b)
//!     cannot pass by the floor being unreachable.

use blip25_codec::enc::{encode_frame, Encoder};

const FRAME: usize = 160;

/// `b1` values either side of the floor's gate (`VOICING_CB[b1*4] >> 30 & 1`,
/// which is exactly `b1 < 16`).
const B1_GATE_OFF: u16 = 16;
const B1_GATE_ON: u16 = 0;

/// Deterministic pseudo-speech PCM: two summed sines with a slow amplitude
/// ramp plus fixed-seed LCG noise. No I/O, no external fixtures.
fn test_pcm(nframes: usize) -> Vec<i16> {
    let n = nframes * FRAME;
    let mut lcg: u32 = 0x1234_5678;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = i32::from((lcg >> 16) as i16) / 8;
        let t = i as f64 / 8000.0;
        let s = 6000.0 * (2.0 * std::f64::consts::PI * 155.0 * t).sin()
            + 2500.0 * (2.0 * std::f64::consts::PI * 470.0 * t).sin();
        let ramp = 0.25 + 0.75 * (i as f64 / n as f64);
        let v = (s * ramp) as i32 + noise;
        out.push(v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16);
    }
    out
}

/// Stream the clip through the r33 encoder on the Route A path (the only path
/// the floor is reachable from), draining the final look-ahead frames.
fn encode_all(
    pcm: &[i16],
    floor: Option<u16>,
    nframes: usize,
) -> Vec<[u8; encode_frame::R33_BYTES]> {
    let mut enc = Encoder::new();
    enc.set_live_gap2_amps(true);
    if let Some(b1) = floor {
        enc.set_lowband_floor(true);
        enc.set_lowband_gate_b1(vec![b1; nframes + 4]);
    }
    let mut frames = Vec::new();
    for ch in pcm.chunks_exact(FRAME) {
        let mut fr = [0i16; FRAME];
        fr.copy_from_slice(ch);
        if let Some(b) = enc.encode_frame_r33(&fr) {
            frames.push(b);
        }
    }
    frames.extend(enc.flush_r33());
    frames
}

#[test]
fn lowband_floor_gate_is_bit_transparent_when_gated_off() {
    const NFRAMES: usize = 300;
    let pcm = test_pcm(NFRAMES);

    let off = encode_all(&pcm, None, NFRAMES);
    let gated_off = encode_all(&pcm, Some(B1_GATE_OFF), NFRAMES);
    let gated_on = encode_all(&pcm, Some(B1_GATE_ON), NFRAMES);

    assert!(!off.is_empty(), "no frames emitted");
    assert_eq!(off.len(), gated_off.len());
    assert_eq!(off.len(), gated_on.len());
    assert_eq!(
        off, gated_off,
        "the floor changed output bits on frames where its gate is off"
    );
    assert_ne!(
        off, gated_on,
        "the floor changed nothing with its gate forced on — the test cannot \
         distinguish a working gate from an unreachable floor"
    );
}
