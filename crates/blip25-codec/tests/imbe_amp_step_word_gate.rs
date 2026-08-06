//! Regression guard for `Encoder::set_imbe_amp_step_word` (the off-by-default
//! params-block pitch word behind the IMBE amplitude back end's unvoiced-band
//! log offset):
//!
//! (a) the flag off must not change a single shipped output bit,
//! (b) the flag on must make the emitted bits independent of the prequantiser
//!     word the caller supplies — that is what "read `params[+0xc]` instead" means,
//! (c) the flag off must leave them dependent on it, so (b) cannot pass by the
//!     offset being unreachable on this stimulus.

use blip25_codec::enc::Encoder;
use blip25_codec::imbe::quantize::pitch_word_for_b0;

const FRAME: usize = 160;

/// A voicing index whose band 0 is unvoiced, so the offset the flag re-keys is
/// actually subtracted from at least one band.
const B1_UNVOICED_BAND0: u16 = 16;

/// The forced pitch index. `set_forced_imbe_b0` fixes `b̂₀` independently of the
/// supplied word, so the word can be varied without moving `L` or the step.
const B0: u8 = 46;

/// Deterministic harmonic stack plus fixed-seed LCG noise. No I/O, no fixtures.
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
        for h in [1.0, 2.0, 3.0, 5.0, 7.0, 9.0] {
            s += 3500.0 * (2.0 * std::f64::consts::PI * f0 * h * t).sin();
        }
        let ramp = 0.25 + 0.75 * (i as f64 / n as f64);
        let v = (s * ramp) as i32 + noise;
        out.push(v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16);
    }
    out
}

fn encode_all(
    pcm: &[i16],
    word: u16,
    step_word: bool,
    nframes: usize,
) -> Vec<[u8; blip25_codec::imbe::FRAME_BYTES]> {
    let n = nframes + 4;
    let mut enc = Encoder::new();
    enc.set_live_gap2_amps(true);
    enc.set_forced_imbe_b0(vec![B0; n]);
    enc.set_forced_imbe_word(vec![word; n]);
    enc.set_forced_b1(vec![B1_UNVOICED_BAND0; n]);
    enc.set_imbe_ref_amps(true);
    enc.set_imbe_amp_step_word(step_word);
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
fn imbe_amp_step_word_replaces_the_supplied_prequantiser_word() {
    const NFRAMES: usize = 200;
    let pcm = test_pcm(NFRAMES);
    let w = pitch_word_for_b0(B0);
    let w_other = w / 2;
    assert_ne!(w, w_other);

    let off_w = encode_all(&pcm, w, false, NFRAMES);
    let off_other = encode_all(&pcm, w_other, false, NFRAMES);
    let on_w = encode_all(&pcm, w, true, NFRAMES);
    let on_other = encode_all(&pcm, w_other, true, NFRAMES);

    assert!(!off_w.is_empty(), "no frames emitted");
    assert_ne!(
        off_w, off_other,
        "the supplied word changed nothing with the flag off — the test cannot \
         distinguish a working re-key from an unreachable offset"
    );
    assert_eq!(
        on_w, on_other,
        "with the flag on the output still depends on the supplied prequantiser \
         word, so the back end is not reading the params block's own word"
    );
}
