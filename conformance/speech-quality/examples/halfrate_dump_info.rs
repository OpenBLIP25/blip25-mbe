use blip25_mbe::p25_halfrate::frame::{DIBITS_PER_FRAME, decode_frame};
use std::fs;

fn unpack(bytes: &[u8]) -> [u8; DIBITS_PER_FRAME] {
    let mut out = [0u8; DIBITS_PER_FRAME];
    let mut bit = 0usize;
    for slot in &mut out {
        let mut d = 0u8;
        for _ in 0..2 {
            let b = (bytes[bit/8] >> (7-(bit%8))) & 1;
            d = (d<<1) | b;
            bit += 1;
        }
        *slot = d;
    }
    out
}

fn main() {
    let path = std::env::args().nth(1).expect("bit file");
    let bytes = fs::read(&path).unwrap();
    let n = bytes.len() / 9;
    for f in 0..n {
        let dibits = unpack(&bytes[f*9..(f+1)*9]);
        let frame = decode_frame(&dibits);
        println!("{:04x} {:04x} {:04x} {:04x}",
            frame.info[0], frame.info[1], frame.info[2], frame.info[3]);
    }
}
