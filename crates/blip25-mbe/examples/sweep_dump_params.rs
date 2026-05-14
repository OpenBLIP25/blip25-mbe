//! Decode an .ambe9 stream and dump per-frame params (L, ω₀, M̄_l after
//! enhancement) for the spectral-clamp characterisation sweep. Used
//! alongside sweep_b0_jump probes to compute cosine similarity between
//! consecutive M̄ vectors.
//!
//! Usage: sweep_dump_params <in.ambe9> <out.csv>
//! Output CSV columns: frame, L, omega_0, M_bar_0, M_bar_1, ..., M_bar_(L-1)
//! (variable column count per row; readers should split per row)

use blip25_mbe::ambe_plus2_wire::dequantize::{
    DecoderState, Decoded, decode_to_params,
};
use blip25_mbe::ambe_plus2_wire::frame::decode_frame;
use blip25_mbe::codecs::mbe_baseline::{enhance_spectral_amplitudes, INIT_S_E};
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let in_path = &args[1];
    let out_path = &args[2];
    let bytes = std::fs::read(in_path).unwrap();
    let nf = bytes.len() / 9;
    let mut decoder = DecoderState::new();
    let mut s_e = INIT_S_E;
    let mut out = std::fs::File::create(out_path).unwrap();
    writeln!(out, "frame,L,omega_0,M_bar...").unwrap();
    for i in 0..nf {
        let frame_bytes = &bytes[i * 9..i * 9 + 9];
        let mut dibits = [0u8; 36];
        let mut bit = 0;
        for slot in dibits.iter_mut() {
            let mut d = 0u8;
            for _ in 0..2 {
                let b = (frame_bytes[bit / 8] >> (7 - (bit % 8))) & 1;
                d = (d << 1) | b;
                bit += 1;
            }
            *slot = d;
        }
        let f = decode_frame(&dibits);
        let p = match decode_to_params(&f.info, &mut decoder) {
            Ok(Decoded::Voice(p)) => p,
            _ => {
                writeln!(out, "{},0,0,", i).unwrap();
                continue;
            }
        };
        let l = p.harmonic_count();
        let amps = p.amplitudes_slice();
        let (m_bar, new_s_e) = enhance_spectral_amplitudes(&amps[..l as usize], p.omega_0(), s_e);
        s_e = new_s_e;
        write!(out, "{},{},{:.6}", i, l, p.omega_0()).unwrap();
        for v in &m_bar[..l as usize] {
            write!(out, ",{:.4}", v).unwrap();
        }
        writeln!(out).unwrap();
    }
}
