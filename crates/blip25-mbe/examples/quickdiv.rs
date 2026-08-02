//! What triggers the whole-buffer vs streaming divergence?
//!
//! `AmbeStream` (behind `LiveEncoder`) is byte-exact to `Vocoder::encode` on
//! real audio, including recordings with long genuine pauses and literal
//! digital-zero frames. A highly periodic synthetic signal breaks that at some
//! clip lengths and not others. Neither sweep below isolates a cause: the
//! attenuated-only, near-zero and digital-zero cases all diverge while the
//! untouched signal does not, and identical content diverges at 24 and 400
//! frames but not at 48, 96 or 200. Silence is not the trigger, and the
//! mechanism is unidentified.
//!
//! Usage: `cargo run --release --example quickdiv`

use blip25_mbe::vocoder::{LiveEncoder, Rate, Vocoder};

const RATE: Rate = Rate::AmbePlus2_3600x2450;

fn base(frames: usize) -> Vec<i16> {
    let mut st: u64 = 0x2545_F491_4F6C_DD1D;
    let mut o = Vec::new();
    for i in 0..frames * 160 {
        st ^= st >> 12;
        st ^= st << 25;
        st ^= st >> 27;
        let n = ((st.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 48) as i32) - 16384;
        let tri = |p: i32| {
            let q = (i as i32) % p;
            let h = p / 2;
            if q < h {
                q * 2 - h
            } else {
                (p - q) * 2 - h
            }
        };
        let env = 1 + ((i / 160) % 5) as i32;
        o.push(
            ((tri(51) * 90 + tri(23) * 40 + tri(11) * 18) * env / 5 + n / 24).clamp(-32768, 32767)
                as i16,
        );
    }
    o
}

fn live(pcm: &[i16]) -> Vec<Vec<u8>> {
    let mut e = LiveEncoder::new(RATE);
    let mut a = Vec::new();
    for c in pcm.chunks(377) {
        for f in e.push(c) {
            a.push(f.expect("live encode"));
        }
    }
    a.extend(e.flush().expect("flush"));
    a
}

fn diff(pcm: &[i16]) -> usize {
    let w = Vocoder::new(RATE).encode(pcm);
    let l = live(pcm);
    w.iter().zip(l.iter()).filter(|(a, b)| a != b).count() + w.len().abs_diff(l.len())
}

fn main() {
    // Length sweep: does the divergence survive once the clip is long relative
    // to the streamer's look-ahead reserve (B1_RESERVE = 16 frames)?
    for n in [24usize, 48, 96, 200, 400] {
        let mut v = base(n);
        for c in v.chunks_mut(160).skip(6).take(4) {
            c.iter_mut().for_each(|s| *s /= 64);
        }
        for c in v.chunks_mut(160).skip(17).take(2) {
            c.fill(0);
        }
        println!("len={n:<4} frames  differing={}", diff(&v));
    }
    println!();

    let mut cases: Vec<(&str, Vec<i16>)> = vec![("untouched", base(24))];

    let mut a = base(24);
    for c in a.chunks_mut(160).skip(6).take(4) {
        c.iter_mut().for_each(|s| *s /= 64);
    }
    cases.push(("attenuated /64 only", a));

    let mut b = base(24);
    for c in b.chunks_mut(160).skip(17).take(2) {
        c.fill(0);
    }
    cases.push(("DIGITAL ZERO only", b));

    let mut c2 = base(24);
    for c in c2.chunks_mut(160).skip(17).take(2) {
        c.iter_mut().for_each(|s| *s = (*s / 512).clamp(-3, 3));
    }
    cases.push(("near-zero (rms ~2)", c2));

    let rms = |v: &[i16]| {
        (v.iter().map(|&s| f64::from(s) * f64::from(s)).sum::<f64>() / v.len() as f64).sqrt()
    };
    for (name, pcm) in cases {
        let quietest = pcm.chunks_exact(160).map(rms).fold(f64::INFINITY, f64::min);
        println!(
            "{name:<22} differing_frames={:<4} quietest_frame_rms={quietest:.2}",
            diff(&pcm)
        );
    }
}
