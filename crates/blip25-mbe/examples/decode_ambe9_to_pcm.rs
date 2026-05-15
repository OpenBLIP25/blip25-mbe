use std::fs;
use std::env;
use blip25_mbe::vocoder::{Rate, Vocoder};

fn main() {
    let args: Vec<String> = env::args().collect();
    let in_path = &args[1];
    let out_path = &args[2];
    let ambe = fs::read(in_path).unwrap();
    let mut voc = Vocoder::new(Rate::AmbePlus2_3600x2450);
    let fb = voc.fec_frame_bytes();
    let n = ambe.len() / fb;
    let mut out: Vec<i16> = Vec::with_capacity(n * 160);
    for i in 0..n {
        let frame = &ambe[i * fb..(i + 1) * fb];
        out.extend_from_slice(&voc.decode_bits(frame).unwrap());
    }
    let mut blob = Vec::with_capacity(out.len() * 2);
    for &s in &out {
        blob.extend_from_slice(&s.to_le_bytes());
    }
    fs::write(out_path, &blob).unwrap();
    eprintln!("decoded {} frames to {}", n, out_path);
}
