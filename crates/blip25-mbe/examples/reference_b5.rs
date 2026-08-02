//! Dump reference b5 (deprioritized field 5) per frame from a reference .bit file.
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let wb = std::fs::read(&a[0]).expect("read bit file");
    for f7 in wb.chunks_exact(7) {
        let bit = |p: usize| ((f7[p / 8] >> (7 - (p % 8))) & 1) as u16;
        let mut u = [0u16; 4];
        let widths = [12usize, 12, 11, 14];
        let mut p = 0;
        for (i, &w) in widths.iter().enumerate() {
            let mut v = 0u16;
            for _ in 0..w { v = (v << 1) | bit(p); p += 1; }
            u[i] = v;
        }
        let b = blip25_codec::tables::deprioritize(&u);
        println!("{}", b[5]);
    }
}
