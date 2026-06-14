//! Diagnostic probe: compare OUR encoder's per-frame analysis params on
//! clean vs hum-laden PCM, to localize the low-band/HUM weakness
//! (QUALITY_FINDINGS §3.4). Reports how 60/120 Hz hum perturbs:
//!   - b0 / pitch (omega_0, fundamental Hz) — does hum pull pitch onto the
//!     60/120 Hz sub-harmonic?
//!   - b1 / voicing — how many bands flip voiced/unvoiced?
//!   - low-band amplitude (M_1, M_2) — does the hum dominate the low bins?
//!
//! Usage: hum_diag_probe <clean.pcm> <hum.pcm>

use blip25_mbe::codecs::mbe_baseline::analysis::{encode_ambe_plus2, AnalysisOutput, AnalysisState};

const FRAME: usize = 160;

fn read_pcm(path: &str) -> Vec<i16> {
    let bytes = std::fs::read(path).expect("read pcm");
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

struct FrameParams {
    voice: bool,
    f0_hz: f32,
    l: u8,
    n_voiced: usize,
    m1: f32,
    m2: f32,
    m_low_sum: f32, // sum of |M_l| for harmonics whose center < 250 Hz
}

fn encode_all(pcm: &[i16]) -> Vec<FrameParams> {
    let mut st = AnalysisState::new();
    let mut out = Vec::new();
    for frame in pcm.chunks_exact(FRAME) {
        let r = encode_ambe_plus2(frame, &mut st).expect("encode");
        match r {
            AnalysisOutput::Voice(p) => {
                let f0 = p.fundamental_hz();
                let l = p.harmonic_count();
                let amps = p.amplitudes_slice();
                let vs = p.voiced_slice();
                let n_voiced = vs.iter().filter(|&&v| v).count();
                let m1 = amps.first().copied().unwrap_or(0.0);
                let m2 = amps.get(1).copied().unwrap_or(0.0);
                // harmonics with center freq < 250 Hz
                let mut m_low = 0.0f32;
                for (i, &a) in amps.iter().enumerate() {
                    let hz = f0 * (i as f32 + 1.0);
                    if hz < 250.0 {
                        m_low += a.abs();
                    }
                }
                out.push(FrameParams {
                    voice: true,
                    f0_hz: f0,
                    l,
                    n_voiced,
                    m1,
                    m2,
                    m_low_sum: m_low,
                });
            }
            AnalysisOutput::Silence => out.push(FrameParams {
                voice: false,
                f0_hz: 0.0,
                l: 0,
                n_voiced: 0,
                m1: 0.0,
                m2: 0.0,
                m_low_sum: 0.0,
            }),
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: hum_diag_probe <clean.pcm> <hum.pcm>");
        std::process::exit(1);
    }
    let clean = read_pcm(&args[1]);
    let hum = read_pcm(&args[2]);
    let n = clean.len().min(hum.len());
    let clean = &clean[..n];
    let hum = &hum[..n];

    let pc = encode_all(clean);
    let ph = encode_all(hum);
    let nf = pc.len().min(ph.len());

    // Counters
    let mut n_voiced_frames = 0usize; // frames where clean is Voice
    let mut f0_shift_frames = 0usize; // |Δf0| > 10 Hz on voiced-voiced
    let mut f0_halved = 0usize; // hum f0 ~= clean/2 (sub-harmonic lock)
    let mut f0_to_low = 0usize; // hum f0 lands in 50-130 Hz band when clean was > 150
    let mut voicing_flip_total = 0usize; // total band voicing flips
    let mut voicing_flip_frames = 0usize; // frames with >=1 flip
    let mut silence_flip = 0usize; // clean Voice -> hum Silence or vice versa
    let mut m1_ratio_sum = 0.0f64;
    let mut m2_ratio_sum = 0.0f64;
    let mut mlow_ratio_sum = 0.0f64;
    let mut ratio_count = 0usize;
    let mut sum_abs_df0 = 0.0f64;

    for i in 0..nf {
        let c = &pc[i];
        let h = &ph[i];
        if c.voice {
            n_voiced_frames += 1;
        }
        if c.voice != h.voice {
            silence_flip += 1;
        }
        if c.voice && h.voice {
            let df0 = (h.f0_hz - c.f0_hz).abs();
            sum_abs_df0 += df0 as f64;
            if df0 > 10.0 {
                f0_shift_frames += 1;
            }
            // sub-harmonic: hum f0 about half clean f0
            if c.f0_hz > 120.0 && (h.f0_hz - c.f0_hz / 2.0).abs() < 8.0 {
                f0_halved += 1;
            }
            // pulled into the 50-130 Hz hum band
            if c.f0_hz > 150.0 && h.f0_hz < 130.0 {
                f0_to_low += 1;
            }
            // band voicing flips over common harmonics
            let cv = c.n_voiced as i64;
            let hv = h.n_voiced as i64;
            let flips = (cv - hv).unsigned_abs() as usize;
            voicing_flip_total += flips;
            if flips > 0 {
                voicing_flip_frames += 1;
            }
            // amplitude ratios (hum/clean) on the low bins
            if c.m1 > 1.0 {
                m1_ratio_sum += (h.m1 / c.m1) as f64;
                ratio_count += 1;
            }
            if c.m2 > 1.0 {
                m2_ratio_sum += (h.m2 / c.m2) as f64;
            }
            if c.m_low_sum > 1.0 {
                mlow_ratio_sum += (h.m_low_sum / c.m_low_sum) as f64;
            }
        }
    }

    println!("=== HUM DIAGNOSTIC: clean={} hum={} ===", args[1], args[2]);
    println!("frames analyzed: {nf}  (clean Voice frames: {n_voiced_frames})");
    println!();
    println!("--- (a) PITCH / b0 ---");
    println!(
        "  mean |Δf0| over voiced-voiced frames: {:.2} Hz",
        sum_abs_df0 / (n_voiced_frames.max(1) as f64)
    );
    println!(
        "  f0 shifted >10 Hz: {f0_shift_frames} frames ({:.1}% of voiced)",
        100.0 * f0_shift_frames as f64 / n_voiced_frames.max(1) as f64
    );
    println!(
        "  f0 HALVED (sub-harmonic lock, hum≈clean/2): {f0_halved} frames ({:.1}%)",
        100.0 * f0_halved as f64 / n_voiced_frames.max(1) as f64
    );
    println!(
        "  f0 pulled into <130 Hz hum band (clean was >150): {f0_to_low} frames ({:.1}%)",
        100.0 * f0_to_low as f64 / n_voiced_frames.max(1) as f64
    );
    println!();
    println!("--- (b) VOICING / b1 ---");
    println!(
        "  frames with >=1 voiced-band-count change: {voicing_flip_frames} ({:.1}%)",
        100.0 * voicing_flip_frames as f64 / n_voiced_frames.max(1) as f64
    );
    println!(
        "  total |Δ(n_voiced bands)| across frames: {voicing_flip_total}"
    );
    println!("  Voice<->Silence flips: {silence_flip}");
    println!();
    println!("--- (c) LOW-BAND AMPLITUDE / §0.5 ---");
    println!(
        "  mean M_1(hum)/M_1(clean) [harmonic 1, ~f0]: {:.3}x",
        m1_ratio_sum / ratio_count.max(1) as f64
    );
    println!(
        "  mean M_2(hum)/M_2(clean) [harmonic 2]:      {:.3}x",
        m2_ratio_sum / ratio_count.max(1) as f64
    );
    println!(
        "  mean Σ|M_l<250Hz|(hum)/clean [low-band sum]: {:.3}x",
        mlow_ratio_sum / ratio_count.max(1) as f64
    );

    // Print a few representative voiced frames where f0 diverged
    println!();
    println!("--- sample diverging voiced frames (idx: clean -> hum) ---");
    let mut shown = 0;
    for i in 0..nf {
        let c = &pc[i];
        let h = &ph[i];
        if c.voice && h.voice && (h.f0_hz - c.f0_hz).abs() > 10.0 {
            println!(
                "  f{i}: f0 {:.1}->{:.1} Hz  L {}->{}  nV {}->{}  M1 {:.0}->{:.0}  Mlow {:.0}->{:.0}",
                c.f0_hz, h.f0_hz, c.l, h.l, c.n_voiced, h.n_voiced, c.m1, h.m1, c.m_low_sum, h.m_low_sum
            );
            shown += 1;
            if shown >= 15 {
                break;
            }
        }
    }
}
