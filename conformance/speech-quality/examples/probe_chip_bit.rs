use blip25_mbe::imbe7200::dequantize::{DecoderState, dequantize, DecodeError};
use blip25_mbe::imbe7200::frame::decode_frame;
use std::fs;
use std::collections::BTreeMap;

fn unpack_dibits_msb(bytes: &[u8]) -> [u8; 72] {
    let mut out = [0u8; 72];
    let mut bit = 0usize;
    for slot in &mut out {
        let mut d = 0u8;
        for _ in 0..2 {
            let b = (bytes[bit/8] >> (7 - (bit%8))) & 1;
            d = (d << 1) | b; bit += 1;
        }
        *slot = d;
    }
    out
}

fn unpack_dibits_lsb(bytes: &[u8]) -> [u8; 72] {
    let mut out = [0u8; 72];
    let mut bit = 0usize;
    for slot in &mut out {
        let mut d = 0u8;
        for _ in 0..2 {
            let b = (bytes[bit/8] >> (bit%8)) & 1;
            d = (d << 1) | b; bit += 1;
        }
        *slot = d;
    }
    out
}

fn unpack_dibits_msb_swap_dibit(bytes: &[u8]) -> [u8; 72] {
    let mut out = [0u8; 72];
    let mut bit = 0usize;
    for slot in &mut out {
        let b0 = (bytes[bit/8] >> (7 - (bit%8))) & 1; bit += 1;
        let b1 = (bytes[bit/8] >> (7 - (bit%8))) & 1; bit += 1;
        *slot = (b1 << 1) | b0;
    }
    out
}

fn unpack_dibits_byterev(bytes: &[u8]) -> [u8; 72] {
    // MSB-first unpacking but with each byte bit-reversed first.
    let rev: Vec<u8> = bytes.iter().map(|b| b.reverse_bits()).collect();
    unpack_dibits_msb(&rev)
}

fn run(path: &str, name: &str, unpacker: fn(&[u8]) -> [u8; 72]) {
    let bytes = fs::read(path).unwrap();
    let mut st = DecoderState::new();
    let mut ok = 0; let mut bad_pitch = 0; let mut other_err = 0;
    let mut err0_hist: BTreeMap<u8, u32> = BTreeMap::new();
    let mut info0_hist = BTreeMap::<u16, u32>::new();
    let n = bytes.len() / 18;
    for f in 0..n {
        let chunk = &bytes[f*18..(f+1)*18];
        let dibits = unpacker(chunk);
        let imbe = decode_frame(&dibits);
        *err0_hist.entry(imbe.errors[0]).or_insert(0) += 1;
        *info0_hist.entry(imbe.info[0]).or_insert(0) += 1;
        match dequantize(&imbe.info, &mut st) {
            Ok(_) => ok += 1,
            Err(DecodeError::BadPitch) => bad_pitch += 1,
            Err(_) => other_err += 1,
        }
    }
    println!("== {} via {} ==", path, name);
    println!("  ok={} bad_pitch={} other_err={}", ok, bad_pitch, other_err);
    let mut v: Vec<_> = err0_hist.into_iter().collect();
    v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    print!("  err_hist:");
    for (k, c) in v.iter().take(4) { print!(" {}->{}", k, c); }
    println!();
    let mut v: Vec<_> = info0_hist.into_iter().collect();
    v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    print!("  info[0] top:");
    for (k, c) in v.iter().take(4) { print!(" 0x{:04x}->{}", k, c); }
    println!();
}

// No-PN variant: unpack MSB-first, then run a custom decode_frame
// path that skips PN XOR onto c[1..6]. The chip may emit
// FEC-encoded-but-pre-PN bits (the modem is expected to apply PN on
// the air interface).
fn run_nopn(path: &str) {
    use blip25_mbe::fec::{golay_23_12_decode, hamming_15_11_decode};
    use blip25_mbe::imbe7200::fec::deinterleave;
    let bytes = fs::read(path).unwrap();
    let mut st = DecoderState::new();
    let mut ok = 0; let mut bad_pitch = 0; let mut other_err = 0;
    let mut err0_hist: BTreeMap<u8, u32> = BTreeMap::new();
    let mut info0_hist = BTreeMap::<u16, u32>::new();
    let n = bytes.len() / 18;
    for f in 0..n {
        let chunk = &bytes[f*18..(f+1)*18];
        let dibits = unpack_dibits_msb(chunk);
        let c = deinterleave(&dibits);
        let d0 = golay_23_12_decode(c[0]);
        *err0_hist.entry(d0.errors).or_insert(0) += 1;
        // Skip masks[i] XOR — feed raw c[1..6] to the FEC decoders.
        let d1 = golay_23_12_decode(c[1]);
        let d2 = golay_23_12_decode(c[2]);
        let d3 = golay_23_12_decode(c[3]);
        let d4 = hamming_15_11_decode(c[4] as u16);
        let d5 = hamming_15_11_decode(c[5] as u16);
        let d6 = hamming_15_11_decode(c[6] as u16);
        let info = [d0.info, d1.info, d2.info, d3.info, d4.info, d5.info, d6.info, c[7] as u16];
        *info0_hist.entry(info[0]).or_insert(0) += 1;
        match dequantize(&info, &mut st) {
            Ok(_) => ok += 1,
            Err(DecodeError::BadPitch) => bad_pitch += 1,
            Err(_) => other_err += 1,
        }
    }
    println!("== {} via MSB + NO-PN ==", path);
    println!("  ok={} bad_pitch={} other_err={}", ok, bad_pitch, other_err);
    let mut v: Vec<_> = err0_hist.into_iter().collect();
    v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    print!("  err_hist(d0):");
    for (k, c) in v.iter().take(4) { print!(" {}->{}", k, c); }
    println!();
    let mut v: Vec<_> = info0_hist.into_iter().collect();
    v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    print!("  info[0] top:");
    for (k, c) in v.iter().take(4) { print!(" 0x{:04x}->{}", k, c); }
    println!();
}

// Sequential-codeword variant: the chip may emit c[0..7] concatenated
// directly without P25 Annex-H interleaving. Widths sum to 144 bits.
fn run_sequential(path: &str) {
    use blip25_mbe::fec::{golay_23_12_decode, hamming_15_11_decode};
    use blip25_mbe::imbe7200::fec::modulation_masks;
    let widths = [23u8, 23, 23, 23, 15, 15, 15, 7];
    let bytes = fs::read(path).unwrap();
    let mut st = DecoderState::new();
    let mut ok = 0; let mut bad_pitch = 0; let mut other_err = 0;
    let mut err0_hist: BTreeMap<u8, u32> = BTreeMap::new();
    let mut info0_hist = BTreeMap::<u16, u32>::new();
    let n = bytes.len() / 18;
    for f in 0..n {
        let chunk = &bytes[f*18..(f+1)*18];
        // MSB-first bit stream into 8 variable-width codewords.
        let mut c = [0u32; 8];
        let mut bitpos = 0usize;
        for (idx, &w) in widths.iter().enumerate() {
            let mut v = 0u32;
            for _ in 0..w {
                let b = (chunk[bitpos/8] >> (7 - (bitpos%8))) & 1;
                v = (v << 1) | u32::from(b);
                bitpos += 1;
            }
            c[idx] = v;
        }
        let d0 = golay_23_12_decode(c[0]);
        *err0_hist.entry(d0.errors).or_insert(0) += 1;
        let masks = modulation_masks(d0.info);
        let d1 = golay_23_12_decode(c[1] ^ masks[1]);
        let d2 = golay_23_12_decode(c[2] ^ masks[2]);
        let d3 = golay_23_12_decode(c[3] ^ masks[3]);
        let d4 = hamming_15_11_decode((c[4] ^ masks[4]) as u16);
        let d5 = hamming_15_11_decode((c[5] ^ masks[5]) as u16);
        let d6 = hamming_15_11_decode((c[6] ^ masks[6]) as u16);
        let info = [d0.info, d1.info, d2.info, d3.info, d4.info, d5.info, d6.info, c[7] as u16];
        *info0_hist.entry(info[0]).or_insert(0) += 1;
        match dequantize(&info, &mut st) {
            Ok(_) => ok += 1,
            Err(DecodeError::BadPitch) => bad_pitch += 1,
            Err(_) => other_err += 1,
        }
    }
    println!("== {} via sequential-codeword ==", path);
    println!("  ok={} bad_pitch={} other_err={}", ok, bad_pitch, other_err);
    let mut v: Vec<_> = err0_hist.into_iter().collect();
    v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    print!("  err_hist(d0):");
    for (k, c) in v.iter().take(4) { print!(" {}->{}", k, c); }
    println!();
    let mut v: Vec<_> = info0_hist.into_iter().collect();
    v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    print!("  info[0] top:");
    for (k, c) in v.iter().take(4) { print!(" 0x{:04x}->{}", k, c); }
    println!();
}

fn main() {
    let path = std::env::args().nth(1).expect("bit file arg");
    run(&path, "MSB",            unpack_dibits_msb);
    run(&path, "LSB",            unpack_dibits_lsb);
    run(&path, "MSB-swap-dibit", unpack_dibits_msb_swap_dibit);
    run(&path, "byte-reversed",  unpack_dibits_byterev);
    run_nopn(&path);
    run_sequential(&path);
}
