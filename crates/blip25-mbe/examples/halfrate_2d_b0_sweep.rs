//! 2-D b̂₀ sweep harness for characterising the AMBE-3000R chip's
//! spectral-discontinuity attenuation (gap 0026, follow-up probe #2).
//!
//! Each cell is a canonical 100-frame stream matching `spectral_jump_probe`:
//!   frames 0..49: low_b0
//!   frame 50:     high_b0  ← trigger
//!   frames 51..99: low_b0
//! Each cell runs as a standalone ssh invocation to the chip so chip state
//! is reset between cells (init_codec + set_p25_halfrate per call).
//!
//! Subcommands:
//!   gen <out_dir> [--b0-values csv] [--b2-values csv]
//!     Writes per-cell .ambe9 files cell_{idx}.ambe9 plus manifest.csv.
//!   score <out_dir>
//!     Reads per-cell chip_{idx}.pcm + manifest, decodes ours frame-by-
//!     frame per cell, emits results.csv.

use std::env;
use std::fs;
use std::path::PathBuf;

use blip25_mbe::ambe_plus2_wire::dequantize::{
    DecoderState, Decoded, decode_to_params, dequantize,
};
use blip25_mbe::ambe_plus2_wire::frame::{decode_frame, encode_frame, DIBITS_PER_FRAME};
use blip25_mbe::ambe_plus2_wire::priority::{prioritize, AMBE_B_COUNT};
use blip25_mbe::mbe_params::MbeParams;
use blip25_mbe::vocoder::{Rate, Vocoder};

const FRAMES_PER_CELL: usize = 100;
const TRIGGER_IDX: usize = 50;
const SAMPLES_PER_FRAME: usize = 160;

fn pack_dibits(dibits: &[u8; DIBITS_PER_FRAME]) -> [u8; 9] {
    let mut out = [0u8; 9];
    let mut bit = 0;
    for &d in dibits {
        for pos in (0..2).rev() {
            let b = (d >> pos) & 1;
            out[bit / 8] |= b << (7 - (bit % 8));
            bit += 1;
        }
    }
    out
}

fn build_frame(b0: u16, b2: u16) -> ([u8; 9], MbeParams) {
    let mut b = [0u16; AMBE_B_COUNT];
    b[0] = b0;
    b[2] = b2;
    let info = prioritize(&b);
    let mut dec = DecoderState::new();
    let params = match decode_to_params(&info, &mut dec) {
        Ok(Decoded::Voice(p)) => p,
        _ => panic!("frame b0={} b2={} not Voice", b0, b2),
    };
    (pack_dibits(&encode_frame(&info)), params)
}

fn cosine_similarity(a: &MbeParams, b: &MbeParams) -> f64 {
    let aa = a.amplitudes_slice();
    let bb = b.amplitudes_slice();
    let n = aa.len().max(bb.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let va = *aa.get(i).unwrap_or(&0.0) as f64;
        let vb = *bb.get(i).unwrap_or(&0.0) as f64;
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn default_b0_values() -> Vec<u16> {
    // Step b̂₀ by 4 across the valid voice range 0..119 → 30 representative
    // values that sample both L jumps and within-L ω̃₀ variation.
    (0u16..=119).step_by(4).collect()
}

fn default_b2_values() -> Vec<u16> {
    vec![8, 20]
}

fn parse_csv_u16(s: &str) -> Vec<u16> {
    s.split(',').filter_map(|t| t.trim().parse().ok()).collect()
}

fn cmd_gen(args: &[String]) -> std::io::Result<()> {
    let mut it = args.iter();
    let out_dir = PathBuf::from(it.next().expect("out_dir"));
    let mut b0_vals = default_b0_values();
    let mut b2_vals = default_b2_values();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--b0-values" => b0_vals = parse_csv_u16(it.next().unwrap()),
            "--b2-values" => b2_vals = parse_csv_u16(it.next().unwrap()),
            _ => panic!("unknown flag {}", flag),
        }
    }
    fs::create_dir_all(&out_dir)?;
    let mut manifest = String::new();
    manifest.push_str(
        "cell_idx,low_b0,high_b0,b2,low_l,high_l,low_omega,high_omega,delta_l,delta_omega,cos_sim\n",
    );

    let mut idx = 0usize;
    for &b2 in &b2_vals {
        for (i, &low_b0) in b0_vals.iter().enumerate() {
            for &high_b0 in &b0_vals[i..] {
                let (low_bytes, low_params) = build_frame(low_b0, b2);
                let (high_bytes, high_params) = build_frame(high_b0, b2);
                let cos_sim = cosine_similarity(&low_params, &high_params);
                let dl = (high_params.harmonic_count() as i32 - low_params.harmonic_count() as i32).abs();
                let domega = (high_params.omega_0() as f64 - low_params.omega_0() as f64).abs();
                let mut stream = Vec::with_capacity(FRAMES_PER_CELL * 9);
                for f in 0..FRAMES_PER_CELL {
                    let frame = if f == TRIGGER_IDX { high_bytes } else { low_bytes };
                    stream.extend_from_slice(&frame);
                }
                fs::write(out_dir.join(format!("cell_{:05}.ambe9", idx)), &stream)?;
                manifest.push_str(&format!(
                    "{},{},{},{},{},{},{:.6},{:.6},{},{:.6},{:.6}\n",
                    idx,
                    low_b0, high_b0, b2,
                    low_params.harmonic_count(), high_params.harmonic_count(),
                    low_params.omega_0(), high_params.omega_0(),
                    dl, domega, cos_sim,
                ));
                idx += 1;
            }
        }
    }
    fs::write(out_dir.join("manifest.csv"), &manifest)?;
    eprintln!("wrote {} cells to {}", idx, out_dir.display());
    Ok(())
}

fn peak_rms(pcm: &[i16]) -> (f64, f64) {
    let mut peak = 0i32;
    let mut sq = 0f64;
    for &s in pcm {
        let a = (s as i32).abs();
        if a > peak { peak = a; }
        sq += (s as f64) * (s as f64);
    }
    let rms = if pcm.is_empty() { 0.0 } else { (sq / pcm.len() as f64).sqrt() };
    (peak as f64, rms)
}

fn read_pcm(path: &PathBuf) -> std::io::Result<Vec<i16>> {
    let raw = fs::read(path)?;
    Ok(raw.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect())
}

fn cmd_score(args: &[String]) -> std::io::Result<()> {
    let out_dir = PathBuf::from(&args[0]);
    let manifest = fs::read_to_string(out_dir.join("manifest.csv"))?;
    let mut out = String::new();
    out.push_str(
        "cell_idx,low_b0,high_b0,b2,low_l,high_l,delta_l,delta_omega,cos_sim,\
         chip_peak,chip_rms,ours_peak,ours_rms,rms_ratio,peak_ratio,\
         chip_pre_rms,ours_pre_rms\n",
    );
    let mut missing = 0usize;
    let mut counted = 0usize;
    for (li, line) in manifest.lines().enumerate() {
        if li == 0 { continue; }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 11 { continue; }
        let cell_idx: usize = f[0].parse().unwrap();
        let chip_path = out_dir.join(format!("chip_{:05}.pcm", cell_idx));
        if !chip_path.exists() {
            missing += 1;
            continue;
        }
        let ambe_path = out_dir.join(format!("cell_{:05}.ambe9", cell_idx));
        let ambe = fs::read(&ambe_path)?;
        let chip = read_pcm(&chip_path)?;

        // Decode through our half-rate path
        let mut voc = Vocoder::new(Rate::AmbePlus2_3600x2450);
        let fb = voc.fec_frame_bytes();
        let n_frames = ambe.len() / fb;
        let mut ours: Vec<i16> = Vec::with_capacity(n_frames * SAMPLES_PER_FRAME);
        for i in 0..n_frames {
            let frame = &ambe[i * fb..(i + 1) * fb];
            ours.extend_from_slice(&voc.decode_bits(frame).expect("decode"));
        }
        let s_trig = TRIGGER_IDX * SAMPLES_PER_FRAME;
        let e_trig = s_trig + SAMPLES_PER_FRAME;
        if chip.len() < e_trig || ours.len() < e_trig {
            missing += 1;
            continue;
        }
        let (chip_peak, chip_rms) = peak_rms(&chip[s_trig..e_trig]);
        let (ours_peak, ours_rms) = peak_rms(&ours[s_trig..e_trig]);
        // Pre-trigger steady state: 5 frames ending at frame 49.
        let s_pre = 45 * SAMPLES_PER_FRAME;
        let e_pre = 50 * SAMPLES_PER_FRAME;
        let (_, chip_pre_rms) = peak_rms(&chip[s_pre..e_pre]);
        let (_, ours_pre_rms) = peak_rms(&ours[s_pre..e_pre]);
        let rms_ratio = if ours_rms > 0.0 { chip_rms / ours_rms } else { 0.0 };
        let peak_ratio = if ours_peak > 0.0 { chip_peak / ours_peak } else { 0.0 };

        out.push_str(&format!(
            "{},{},{},{},{},{},{},{:.6},{:.6},{:.1},{:.1},{:.1},{:.1},{:.4},{:.4},{:.1},{:.1}\n",
            cell_idx, f[1], f[2], f[3], f[4], f[5], f[8], f[9].parse::<f64>().unwrap_or(0.0),
            f[10].parse::<f64>().unwrap_or(0.0),
            chip_peak, chip_rms, ours_peak, ours_rms, rms_ratio, peak_ratio,
            chip_pre_rms, ours_pre_rms,
        ));
        counted += 1;
    }
    fs::write(out_dir.join("results.csv"), &out)?;
    eprintln!("scored {} cells ({} missing); results.csv written", counted, missing);
    Ok(())
}

fn unpack_dibits_half(bytes: &[u8]) -> [u8; DIBITS_PER_FRAME] {
    let mut out = [0u8; DIBITS_PER_FRAME];
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

fn cmd_dump_params(args: &[String]) -> std::io::Result<()> {
    let out_dir = PathBuf::from(&args[0]);
    let manifest = fs::read_to_string(out_dir.join("manifest.csv"))?;
    // Header: per-cell metadata then per-frame (omega_0, L, M̄_1..M̄_56)
    let params_dir = out_dir.join("params");
    fs::create_dir_all(&params_dir)?;
    for (li, line) in manifest.lines().enumerate() {
        if li == 0 { continue; }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 11 { continue; }
        let cell_idx: usize = f[0].parse().unwrap();
        let ambe_path = out_dir.join(format!("cell_{:05}.ambe9", cell_idx));
        let ambe = fs::read(&ambe_path)?;
        let mut state = DecoderState::new();
        let mut buf = String::new();
        buf.push_str("frame,omega_0,l");
        for k in 1..=56 { buf.push_str(&format!(",M{}", k)); }
        buf.push('\n');
        let fb = 9; // half-rate FEC frame is 9 bytes
        let n = ambe.len() / fb;
        for i in 0..n {
            let frame_bytes = &ambe[i * fb..(i + 1) * fb];
            let dibits = unpack_dibits_half(frame_bytes);
            let frame = decode_frame(<&[u8; DIBITS_PER_FRAME]>::try_from(&dibits[..]).unwrap());
            // dequantize takes the 4 info codewords directly
            let params = match dequantize(&frame.info, &mut state) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("cell {} frame {} dequantize failed: {:?}", cell_idx, i, e);
                    continue;
                }
            };
            buf.push_str(&format!("{},{:.6},{}", i, params.omega_0(), params.harmonic_count()));
            let amps = params.amplitudes_slice();
            for k in 0..56 {
                let v = if k < amps.len() { amps[k] } else { 0.0 };
                buf.push_str(&format!(",{:.4}", v));
            }
            buf.push('\n');
        }
        fs::write(params_dir.join(format!("params_{:05}.csv", cell_idx)), buf)?;
    }
    eprintln!("wrote per-cell params to {}/params/", out_dir.display());
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: halfrate_2d_b0_sweep gen <out_dir> [--b0-values csv] [--b2-values csv]\n\
                    halfrate_2d_b0_sweep score <out_dir>\n\
                    halfrate_2d_b0_sweep dump-params <out_dir>"
        );
        std::process::exit(2);
    }
    let sub = &args[1];
    let rest: Vec<String> = args[2..].to_vec();
    match sub.as_str() {
        "gen" => cmd_gen(&rest),
        "score" => cmd_score(&rest),
        "dump-params" => cmd_dump_params(&rest),
        _ => { eprintln!("unknown subcommand"); std::process::exit(2); }
    }
}
