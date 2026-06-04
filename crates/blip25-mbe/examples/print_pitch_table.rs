use blip25_mbe::rate33::dequantize::decode_pitch;

fn main() {
    println!("b0\tL\tomega_0");
    for b0 in 0u8..=119 {
        if let Some(p) = decode_pitch(b0) {
            println!("{}\t{}\t{:.6}", b0, p.l, p.omega_0);
        }
    }
}
