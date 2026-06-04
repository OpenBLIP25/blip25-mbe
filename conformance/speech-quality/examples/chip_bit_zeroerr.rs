//! Bisection probe: decode chip.bit through our pipeline but OVERRIDE
//! the per-frame error counts to zero before synth. If the resulting
//! audio is loud and correct, the bug is in our FEC error-count
//! reporting triggering spurious MUTE/REPEAT disposition; if it's
//! still silent, the recovered params themselves are wrong.

use blip25_mbe::codecs::mbe_baseline::{FrameErrorContext, GAMMA_W, SynthState, synthesize_frame};
use blip25_mbe::imbe7200::dequantize::{DecoderState, dequantize};
use blip25_mbe::imbe7200::frame::decode_frame;
use std::fs;

const FRAME_BYTES: usize = 18;
const SAMPLES_PER_FRAME: usize = 160;
const SAMPLE_RATE: u32 = 8000;

fn unpack_dibits_msb(bytes: &[u8]) -> [u8; 72] {
    let mut out = [0u8; 72];
    let mut bit = 0usize;
    for slot in &mut out {
        let mut d = 0u8;
        for _ in 0..2 {
            let b = (bytes[bit / 8] >> (7 - (bit % 8))) & 1;
            d = (d << 1) | b;
            bit += 1;
        }
        *slot = d;
    }
    out
}

fn write_wav(path: &str, samples: &[i16]) {
    use std::io::Write;
    let sr = SAMPLE_RATE;
    let n = samples.len() as u32;
    let byte_rate = sr * 2;
    let data_size = n * 2;
    let chunk_size = 36 + data_size;
    let mut f = fs::File::create(path).unwrap();
    f.write_all(b"RIFF").unwrap();
    f.write_all(&chunk_size.to_le_bytes()).unwrap();
    f.write_all(b"WAVEfmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&sr.to_le_bytes()).unwrap();
    f.write_all(&byte_rate.to_le_bytes()).unwrap();
    f.write_all(&2u16.to_le_bytes()).unwrap();
    f.write_all(&16u16.to_le_bytes()).unwrap();
    f.write_all(b"data").unwrap();
    f.write_all(&data_size.to_le_bytes()).unwrap();
    for s in samples {
        f.write_all(&s.to_le_bytes()).unwrap();
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("bit file arg");
    let out_path = std::env::args()
        .nth(2)
        .expect("out wav arg");
    let bytes = fs::read(&path).unwrap();
    let n = bytes.len() / FRAME_BYTES;
    let mut dec_state = DecoderState::new();
    let mut synth = SynthState::new();
    let mut pcm: Vec<i16> = Vec::with_capacity(n * SAMPLES_PER_FRAME);
    let mut ok_frames = 0;
    let mut bad_pitch = 0;
    for f in 0..n {
        let chunk = &bytes[f * FRAME_BYTES..(f + 1) * FRAME_BYTES];
        let dibits = unpack_dibits_msb(chunk);
        let imbe = decode_frame(&dibits);
        let params = match dequantize(&imbe.info, &mut dec_state) {
            Ok(p) => {
                ok_frames += 1;
                p
            }
            Err(_) => {
                bad_pitch += 1;
                pcm.extend(std::iter::repeat(0i16).take(SAMPLES_PER_FRAME));
                continue;
            }
        };
        // Force ZERO errors into the synthesizer — bypass MUTE / REPEAT
        // dispositions driven by our (possibly spurious) per-vector
        // error counts.
        let err = FrameErrorContext {
            epsilon_0: 0,
            epsilon_4: 0,
            epsilon_t: 0,
            bad_pitch: false,
        };
        let frame = synthesize_frame(&params, &err, GAMMA_W, &mut synth);
        pcm.extend_from_slice(&frame);
    }
    write_wav(&out_path, &pcm);
    let rms = (pcm.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / pcm.len() as f64).sqrt();
    let peak = pcm.iter().map(|&s| s.unsigned_abs()).max().unwrap_or(0);
    eprintln!(
        "frames={} ok={} bad_pitch={} rms={:.1} peak={} → {}",
        n, ok_frames, bad_pitch, rms, peak, out_path
    );
}
