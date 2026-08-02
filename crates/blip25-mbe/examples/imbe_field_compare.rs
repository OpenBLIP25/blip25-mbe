//! Compare OUR IMBE encoder to the reference x86 reference (11-byte / 88-bit
//! prioritized info frames, u0(12)|u1(12)|u2(12)|u3(12)|u4(11)|u5(11)|u6(11)|
//! u7(7), MSB-first, info-only, no FEC). Both x86 -> bit-exact is the reachable
//! target. This is the IMBE analogue of `wave_field_compare.rs`: encode a source
//! .pcm with Rate::Imbe4400x4400 -> 88-bit frames -> deprioritize to the
//! b-vector (b0 pitch, b1 V/UV, b2 gain, b3..b_{L+1} spectral amplitudes); load
//! the reference .bits (12-byte units, strip the trailing 0x00 pad -> 11-byte
//! payload) -> deprioritize; search the best frame alignment offset; print
//! per-field % agreement + an all-fields-match %.
//!
//! Usage:
//!   cargo run --release --example imbe_field_compare -- <source.pcm> <reference.bits>

use blip25_mbe::imbe7200::priority::{deprioritize, L_MAX, L_MIN};
use blip25_mbe::vocoder::{Rate, Vocoder};

/// Parameter-array length b̂₀..b̂_{L+2} for L=56 => 59 slots (= priority::IMBE_B_MAX).
const IMBE_B_MAX: usize = 59;

/// Unpack an 11-byte / 88-bit info payload into u0..u7 (MSB-first).
fn unpack_u(payload: &[u8]) -> [u16; 8] {
    let bit = |p: usize| ((payload[p / 8] >> (7 - (p % 8))) & 1) as u16;
    let widths = [12usize, 12, 12, 12, 11, 11, 11, 7];
    let mut u = [0u16; 8];
    let mut p = 0;
    for (i, &w) in widths.iter().enumerate() {
        let mut v = 0u16;
        for _ in 0..w {
            v = (v << 1) | bit(p);
            p += 1;
        }
        u[i] = v;
    }
    u
}

/// b0 (pitch index) read straight from the info vectors — independent of L, so
/// it can be recovered before deprioritizing (which needs L). Mirrors
/// `dequantize::decode_frame_vector`: b0 = (u0>>4 & 0xFC) | (u7>>1 & 0x3).
fn b0_of(u: &[u16; 8]) -> i16 {
    (((u[0] >> 4) & 0xFC) | ((u[7] >> 1) & 0x3)) as i16
}

/// A decoded frame: the pitch index, harmonic count L, and full b-vector.
/// `valid=false` when b0/L is out of the codec's voice range (frame repeat).
struct Dec {
    b0: i16,
    l: u8,
    b: [u16; IMBE_B_MAX],
    valid: bool,
}

fn decode(u: &[u16; 8]) -> Dec {
    let b0 = b0_of(u);
    // L = num_harms from the decoder's pitch cell (fixed-point exact).
    let (_fund, nh) = blip25_codec::imbe::pitch_cell(b0);
    if !(0..=207).contains(&b0) || !(L_MIN as i16..=L_MAX as i16).contains(&nh) {
        return Dec {
            b0,
            l: nh.clamp(L_MIN as i16, L_MAX as i16) as u8,
            b: [0u16; IMBE_B_MAX],
            valid: false,
        };
    }
    let l = nh as u8;
    let b = deprioritize(u, l);
    Dec {
        b0,
        l,
        b,
        valid: true,
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        eprintln!("usage: imbe_field_compare <source.pcm> <reference.bits>");
        std::process::exit(2);
    }
    let raw = std::fs::read(&a[0]).expect("src pcm");
    let pcm: Vec<i16> = raw
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let gb = std::fs::read(&a[1]).expect("reference bits");

    // Reference: 12-byte units, strip the trailing 0x00 pad -> 11-byte / 88-bit payload.
    let reference: Vec<Dec> = gb
        .chunks_exact(12)
        .map(|unit| decode(&unpack_u(&unit[..11])))
        .collect();

    // Ours: Rate::Imbe4400x4400 emits 11-byte info-only frames.
    let ours: Vec<Dec> = Vocoder::new(Rate::Imbe4400x4400)
        .encode(&pcm)
        .iter()
        .map(|f| decode(&unpack_u(f)))
        .collect();

    // Alignment search on b0 (the pitch field), like wave_field_compare.
    let score = |off: i64| {
        let (mut e, mut n) = (0usize, 0usize);
        for (k, o) in ours.iter().enumerate() {
            let j = k as i64 + off;
            if j < 0 || j as usize >= reference.len() {
                continue;
            }
            n += 1;
            if o.b0 == reference[j as usize].b0 {
                e += 1;
            }
        }
        (e, n)
    };
    let mut best = (0i64, 0usize, 1usize);
    for off in -6..=6 {
        let (e, n) = score(off);
        if n > 0 && e * best.2 > best.1 * n {
            best = (off, e, n);
        }
    }
    let off = best.0;
    println!(
        "frames ours={} reference={}  best offset={off} (b0 {:.1}%)\n",
        ours.len(),
        reference.len(),
        100.0 * best.1 as f64 / best.2.max(1) as f64
    );

    // b0 range/mean — a convention/units mismatch (e.g. AMBE 7-bit b0 stuffed
    // into IMBE's 8-bit cell) shows up as a shifted range here.
    let stat = |d: &[Dec]| {
        let (mut lo, mut hi, mut s, mut n) = (i32::MAX, i32::MIN, 0i64, 0i64);
        for x in d {
            let v = x.b0 as i32;
            lo = lo.min(v);
            hi = hi.max(v);
            s += v as i64;
            n += 1;
        }
        (lo, hi, s as f64 / n.max(1) as f64)
    };
    let (ol, oh, om) = stat(&ours);
    let (gl, gh, gm) = stat(&reference);
    println!("b0 range  ours[{ol}..{oh}] mean {om:.1}   reference[{gl}..{gh}] mean {gm:.1}\n");

    // Per-field tallies.
    let (mut tot, mut both_valid) = (0usize, 0usize);
    let (mut b0_ag, mut b1_ag, mut b2_ag, mut l_ag) = (0usize, 0usize, 0usize, 0usize);
    // conditional-on-L-match: isolates IMBE-wire deficit from the pitch/front-end root.
    let (mut lm_n, mut lm_b1, mut lm_b2, mut lm_amp_n, mut lm_amp_d) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    // amplitude aggregate (coefficients b3.. up to min(Lo,Lg)+1), among L-matching frames
    let (mut amp_num, mut amp_den) = (0usize, 0usize);
    let mut all_ag = 0usize; // b0..b_{L+2} all equal (requires L equal)
    // per-index amplitude agreement, indices 3..=18, among frames where both have that index
    let mut idx_ag = [0usize; IMBE_B_MAX];
    let mut idx_n = [0usize; IMBE_B_MAX];
    // b0 closeness histogram (voiced-ish), like wave_field_compare
    let mut bd = [0usize; 5];
    let mut vo = 0usize;

    for (k, o) in ours.iter().enumerate() {
        let j = k as i64 + off;
        if j < 0 || j as usize >= reference.len() {
            continue;
        }
        let g = &reference[j as usize];
        tot += 1;

        if o.b0 == g.b0 {
            b0_ag += 1;
        }
        let dl = (o.b0 as i32 - g.b0 as i32).unsigned_abs();
        vo += 1;
        if dl == 0 {
            bd[0] += 1;
        }
        if dl <= 1 {
            bd[1] += 1;
        }
        if dl <= 2 {
            bd[2] += 1;
        }
        if dl <= 4 {
            bd[3] += 1;
        }
        if dl >= 8 {
            bd[4] += 1;
        }

        if !(o.valid && g.valid) {
            continue;
        }
        both_valid += 1;
        if o.l == g.l {
            l_ag += 1;
        }
        if o.b[1] == g.b[1] {
            b1_ag += 1;
        }
        if o.b[2] == g.b[2] {
            b2_ag += 1;
        }

        // amplitude coefficients: b[3 .. 3+(L-1)]; compare the overlapping range.
        let amps_o = 3 + o.l as usize - 1;
        let amps_g = 3 + g.l as usize - 1;
        let hi = amps_o.min(amps_g);
        for p in 3..hi {
            amp_den += 1;
            if o.b[p] == g.b[p] {
                amp_num += 1;
            }
            if p < IMBE_B_MAX {
                idx_n[p] += 1;
                if o.b[p] == g.b[p] {
                    idx_ag[p] += 1;
                }
            }
        }

        // Conditional on L matching: the wire fields become directly comparable.
        if o.l == g.l {
            lm_n += 1;
            if o.b[1] == g.b[1] {
                lm_b1 += 1;
            }
            if o.b[2] == g.b[2] {
                lm_b2 += 1;
            }
            for p in 3..(3 + o.l as usize - 1) {
                lm_amp_d += 1;
                if o.b[p] == g.b[p] {
                    lm_amp_n += 1;
                }
            }
            let top = (o.l as usize + 2).min(IMBE_B_MAX - 1);
            if (0..=top).all(|p| o.b[p] == g.b[p]) {
                all_ag += 1;
            }
        }
    }

    let pc = |x: usize, n: usize| 100.0 * x as f64 / n.max(1) as f64;
    println!("per-field agreement vs reference x86 reference ({tot} frames, {both_valid} both-valid):");
    println!("  b0 pitch   {:>6.1}%   ({tot} frames)", pc(b0_ag, tot));
    println!("  L harms    {:>6.1}%   (both-valid)", pc(l_ag, both_valid));
    println!("  b1 vuv     {:>6.1}%   (both-valid)", pc(b1_ag, both_valid));
    println!("  b2 gain    {:>6.1}%   (both-valid)", pc(b2_ag, both_valid));
    println!(
        "  amps agg   {:>6.1}%   ({amp_den} coeffs over L-overlap)",
        pc(amp_num, amp_den)
    );
    println!("  ALL match  {:>6.2}%   (b0..b_L+2, both-valid)", pc(all_ag, both_valid));

    println!("\nconditional on L matching ({lm_n} frames) -- isolates IMBE-wire from pitch root:");
    println!("  b1 vuv | L  {:>6.1}%", pc(lm_b1, lm_n));
    println!("  b2 gain| L  {:>6.1}%", pc(lm_b2, lm_n));
    println!(
        "  amps   | L  {:>6.1}%   ({lm_amp_d} coeffs)",
        pc(lm_amp_n, lm_amp_d)
    );

    println!("\nper-amplitude-index agreement (b3.. among frames sharing that index):");
    for p in 3..20 {
        if idx_n[p] > 0 {
            println!("  b{:<3} {:>6.1}%  (n={})", p, pc(idx_ag[p], idx_n[p]), idx_n[p]);
        }
    }

    println!(
        "\nb0 closeness ({vo} frames): exact {:.1}  +-1 {:.1}  +-2 {:.1}  +-4 {:.1}  gross>=8 {:.1}",
        pc(bd[0], vo),
        pc(bd[1], vo),
        pc(bd[2], vo),
        pc(bd[3], vo),
        pc(bd[4], vo)
    );
}
