//! Regression guard for `Encoder::set_imbe_lowband_floor` (the off-by-default
//! gated low-band spectral floor on the IMBE amplitude vector):
//!
//! (a) the flag off must not change a single shipped output bit,
//! (b) the flag on with an all-gate-off voicing sequence must also not change a
//!     bit — the gate must suppress the floor loop AND its recursive `l0` tail,
//! (c) the flag on with an all-gate-on voicing sequence must change bits, so (b)
//!     cannot pass by the floor being unreachable.

use blip25_codec::enc::Encoder;
use blip25_codec::imbe::quantize::pitch_word_for_b0;

const FRAME: usize = 160;

/// Voicing indices either side of the floor's gate, which is the first
/// harmonic's V/UV decision. Index 0 is all-voiced; index 16 is the codebook
/// entry whose band 0 is unvoiced (`VOICING_CB[b1*4] >> 30 & 1`).
const B1_GATE_ON: u16 = 0;
const B1_GATE_OFF: u16 = 16;

/// A `b̂₀` whose fundamental matches [`test_pcm`]'s harmonic spacing, so the
/// band the floor acts on is the one the stimulus leaves empty.
const B0: u8 = 46;

/// Deterministic missing-fundamental harmonic stack plus fixed-seed LCG noise.
/// The floor raises a low harmonic toward the one above it, so it is only
/// observable on a spectrum whose bottom band is well below its neighbour —
/// a strong fundamental makes the whole mechanism a no-op. No I/O, no external
/// fixtures.
fn test_pcm(nframes: usize) -> Vec<i16> {
    let n = nframes * FRAME;
    let mut lcg: u32 = 0x1234_5678;
    let mut out = Vec::with_capacity(n);
    let f0 = 200.0;
    for i in 0..n {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = i32::from((lcg >> 16) as i16) / 8;
        let t = i as f64 / 8000.0;
        let mut s = 0.0;
        for h in [2.0, 3.0, 4.0, 5.0, 7.0, 9.0] {
            s += 3500.0 * (2.0 * std::f64::consts::PI * f0 * h * t).sin();
        }
        let ramp = 0.25 + 0.75 * (i as f64 / n as f64);
        let v = (s * ramp) as i32 + noise;
        out.push(v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16);
    }
    out
}

/// Stream the clip through the IMBE encoder on the reference-amplitude path (the
/// only path the floor is reachable from), draining the final look-ahead frames.
fn encode_all(
    pcm: &[i16],
    b1: u16,
    floor: bool,
    nframes: usize,
) -> Vec<[u8; blip25_codec::imbe::FRAME_BYTES]> {
    let n = nframes + 4;
    let mut enc = Encoder::new();
    enc.set_live_gap2_amps(true);
    enc.set_forced_imbe_b0(vec![B0; n]);
    enc.set_forced_imbe_word(vec![pitch_word_for_b0(B0); n]);
    enc.set_forced_b1(vec![b1; n]);
    enc.set_imbe_ref_amps(true);
    enc.set_imbe_lowband_floor(floor);
    let mut frames = Vec::new();
    for ch in pcm.chunks_exact(FRAME) {
        let mut fr = [0i16; FRAME];
        fr.copy_from_slice(ch);
        if let Some(b) = enc.encode_imbe_frame(&fr) {
            frames.push(b);
        }
    }
    frames.extend(enc.flush_imbe());
    frames
}

#[test]
fn imbe_lowband_floor_gate_is_bit_transparent_when_gated_off() {
    const NFRAMES: usize = 300;
    let pcm = test_pcm(NFRAMES);

    let off_on_gate = encode_all(&pcm, B1_GATE_ON, false, NFRAMES);
    let on_on_gate = encode_all(&pcm, B1_GATE_ON, true, NFRAMES);
    let off_off_gate = encode_all(&pcm, B1_GATE_OFF, false, NFRAMES);
    let on_off_gate = encode_all(&pcm, B1_GATE_OFF, true, NFRAMES);

    assert!(!off_on_gate.is_empty(), "no frames emitted");
    assert_eq!(
        off_off_gate, on_off_gate,
        "the floor changed output bits on frames where its gate is off"
    );
    assert_ne!(
        off_on_gate, on_on_gate,
        "the floor changed nothing with its gate forced on — the test cannot \
         distinguish a working gate from an unreachable floor"
    );
}
