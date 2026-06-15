use blip25_mbe::codecs::mbe_baseline::{synthesize_frame, FrameErrorContext, SynthState, GAMMA_W};
use blip25_mbe::imbe7200::dequantize::{dequantize, DecoderState};
use blip25_mbe::imbe7200::frame::decode_frame;
use std::fs;

fn unpack_msb(bytes: &[u8]) -> [u8; 72] {
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
    let sr = 8000u32;
    let n = samples.len() as u32;
    let mut f = fs::File::create(path).unwrap();
    let data_size = n * 2;
    let chunk_size = 36 + data_size;
    f.write_all(b"RIFF").unwrap();
    f.write_all(&chunk_size.to_le_bytes()).unwrap();
    f.write_all(b"WAVEfmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&sr.to_le_bytes()).unwrap();
    f.write_all(&(sr * 2).to_le_bytes()).unwrap();
    f.write_all(&2u16.to_le_bytes()).unwrap();
    f.write_all(&16u16.to_le_bytes()).unwrap();
    f.write_all(b"data").unwrap();
    f.write_all(&data_size.to_le_bytes()).unwrap();
    for s in samples {
        f.write_all(&s.to_le_bytes()).unwrap();
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("bit file");
    let out = std::env::args().nth(2).expect("out wav");
    let reset_after: u32 = std::env::args()
        .nth(3)
        .map(|s| s.parse().expect("reset_after"))
        .unwrap_or(0);
    let bytes = fs::read(&path).unwrap();
    let n = bytes.len() / 18;
    let mut dec = DecoderState::new();
    let mut synth = SynthState::new();
    if reset_after > 0 {
        synth.set_repeat_reset_after(Some(reset_after));
    }
    let mut pcm: Vec<i16> = Vec::with_capacity(n * 160);
    for f in 0..n {
        let dibits = unpack_msb(&bytes[f * 18..(f + 1) * 18]);
        let imbe = decode_frame(&dibits);
        let params = match dequantize(&imbe.info, &mut dec) {
            Ok(p) => p,
            Err(_) => {
                pcm.extend(std::iter::repeat(0i16).take(160));
                continue;
            }
        };
        let err = FrameErrorContext {
            epsilon_0: imbe.errors[0],
            epsilon_4: imbe.errors[4],
            epsilon_t: imbe.error_total().min(255) as u8,
            bad_pitch: false,
        };
        pcm.extend_from_slice(&synthesize_frame(&params, &err, GAMMA_W, &mut synth));
    }
    write_wav(&out, &pcm);
    let rms = (pcm.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / pcm.len() as f64).sqrt();
    let peak = pcm.iter().map(|&s| s.unsigned_abs()).max().unwrap_or(0);
    eprintln!("frames={} rms={:.1} peak={} -> {}", n, rms, peak, out);
}
