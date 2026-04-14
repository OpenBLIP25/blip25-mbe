//! Build script for blip25-mbe.
//!
//! Parses normative CSV tables from `spec_tables/` into Rust const
//! arrays and writes them to `$OUT_DIR`. The CSVs are vendored copies
//! of the tables in `~/blip25-specs/standards/TIA-102.BABA-A/annex_tables/`;
//! when a spec table is revised the vendored copy must be re-sync'd.
//!
//! Keeping the generation at build time (rather than hand-transcribing
//! into a `const` in source) makes the CSV the single source of truth
//! and removes an entire class of transcription errors.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    println!("cargo:rerun-if-changed=build.rs");
    gen_annex_h(&out_dir);
    gen_annex_e(&out_dir);
    gen_imbe_bit_prioritization(&out_dir);
    gen_ambe_bit_prioritization(&out_dir);
}

/// Parse `spec_tables/annex_h_interleave.csv` into
/// `$OUT_DIR/annex_h.rs`, emitting the 72-entry `ANNEX_H` table.
fn gen_annex_h(out_dir: &PathBuf) {
    let csv_path = "spec_tables/annex_h_interleave.csv";
    println!("cargo:rerun-if-changed={csv_path}");

    let content = fs::read_to_string(csv_path)
        .unwrap_or_else(|e| panic!("failed to read {csv_path}: {e}"));

    let mut entries: Vec<(u8, u8, u8, u8)> = Vec::with_capacity(72);
    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with("symbol") {
            continue; // header row
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        assert_eq!(
            cols.len(),
            5,
            "Annex H line {}: expected 5 columns, got {}: {raw:?}",
            lineno + 1,
            cols.len()
        );
        let symbol: usize = cols[0].parse().expect("symbol index");
        let bit1_vec: u8 = cols[1].parse().expect("bit1_vector");
        let bit1_idx: u8 = cols[2].parse().expect("bit1_index");
        let bit0_vec: u8 = cols[3].parse().expect("bit0_vector");
        let bit0_idx: u8 = cols[4].parse().expect("bit0_index");
        assert_eq!(
            symbol,
            entries.len(),
            "Annex H symbols must be sequential 0..72"
        );
        assert!(bit1_vec < 8 && bit0_vec < 8, "vector index out of range");
        entries.push((bit1_vec, bit1_idx, bit0_vec, bit0_idx));
    }
    assert_eq!(
        entries.len(),
        72,
        "Annex H must have 72 symbols, got {}",
        entries.len()
    );

    // Validate coverage: every (vector, bit_index) pair from the frame
    // appears exactly once across the 144 dibit slots. Widths are the
    // IMBE vector lengths from BABA-A §1.2.
    const VECTOR_LENGTHS: [u8; 8] = [23, 23, 23, 23, 15, 15, 15, 7];
    let mut seen = [[false; 23]; 8];
    for (bit1_vec, bit1_idx, bit0_vec, bit0_idx) in &entries {
        for &(v, i) in &[(*bit1_vec, *bit1_idx), (*bit0_vec, *bit0_idx)] {
            assert!(
                i < VECTOR_LENGTHS[v as usize],
                "Annex H: vec {v} index {i} exceeds vector length"
            );
            assert!(
                !seen[v as usize][i as usize],
                "Annex H: (vec {v}, idx {i}) appears more than once"
            );
            seen[v as usize][i as usize] = true;
        }
    }
    for (v, row) in seen.iter().enumerate() {
        for (i, &b) in row.iter().enumerate().take(VECTOR_LENGTHS[v] as usize) {
            assert!(b, "Annex H: (vec {v}, idx {i}) never appears");
        }
    }

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_h_interleave.csv\n");
    out.push_str("// Do not edit — regenerated each build.\n");
    out.push_str("pub(crate) const ANNEX_H: [AnnexHEntry; 72] = [\n");
    for (bit1_vec, bit1_idx, bit0_vec, bit0_idx) in &entries {
        out.push_str(&format!(
            "    AnnexHEntry {{ bit1_vec: {bit1_vec}, bit1_idx: {bit1_idx}, \
             bit0_vec: {bit0_vec}, bit0_idx: {bit0_idx} }},\n"
        ));
    }
    out.push_str("];\n");

    fs::write(out_dir.join("annex_h.rs"), out).expect("write annex_h.rs");
}

/// Parse `spec_tables/annex_e_gain_quantizer.csv` into
/// `$OUT_DIR/annex_e_gain.rs`, emitting the 64-entry `IMBE_GAIN_LEVELS`
/// `[f32; 64]` table.
fn gen_annex_e(out_dir: &PathBuf) {
    let csv_path = "spec_tables/annex_e_gain_quantizer.csv";
    println!("cargo:rerun-if-changed={csv_path}");

    let content = fs::read_to_string(csv_path)
        .unwrap_or_else(|e| panic!("failed to read {csv_path}: {e}"));

    let mut levels: Vec<f32> = Vec::with_capacity(64);
    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("b2_index") {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        assert_eq!(
            cols.len(),
            2,
            "annex E line {}: expected 2 columns, got {}: {raw:?}",
            lineno + 1,
            cols.len()
        );
        let idx: usize = cols[0].parse().expect("b2_index");
        let level: f32 = cols[1].parse().expect("level");
        assert_eq!(idx, levels.len(), "Annex E rows must be sequential 0..64");
        levels.push(level);
    }
    assert_eq!(levels.len(), 64, "Annex E must have exactly 64 entries");

    // Monotonicity invariant — Annex E is non-uniform but strictly increasing.
    for w in levels.windows(2) {
        assert!(w[0] < w[1], "Annex E levels must be strictly increasing");
    }

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_e_gain_quantizer.csv\n");
    out.push_str("// Do not edit — regenerated each build.\n");
    out.push_str("pub const IMBE_GAIN_LEVELS: [f32; 64] = [\n");
    for (i, level) in levels.iter().enumerate() {
        out.push_str(&format!("    {level:.6}, // b̂₂ = {i}\n"));
    }
    out.push_str("];\n");

    fs::write(out_dir.join("annex_e_gain.rs"), out).expect("write annex_e_gain.rs");
}

/// Parse `spec_tables/imbe_bit_prioritization.csv` into
/// `$OUT_DIR/imbe_bit_priority.rs`, emitting a `[[BitMap; 88]; 48]`
/// table indexed by `L - 9`.
fn gen_imbe_bit_prioritization(out_dir: &PathBuf) {
    let csv_path = "spec_tables/imbe_bit_prioritization.csv";
    println!("cargo:rerun-if-changed={csv_path}");

    let content = fs::read_to_string(csv_path)
        .unwrap_or_else(|e| panic!("failed to read {csv_path}: {e}"));

    // Group rows by L. Each L in [9, 56] must contain exactly 88 rows.
    let mut per_l: Vec<Vec<(u8, u8, u8, u8)>> = (0..48).map(|_| Vec::with_capacity(88)).collect();

    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('L') {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        assert_eq!(
            cols.len(),
            5,
            "imbe prioritization line {}: expected 5 columns, got {}: {raw:?}",
            lineno + 1,
            cols.len()
        );
        let l: u8 = cols[0].parse().expect("L");
        let src_param: u8 = cols[1].parse().expect("src_param");
        let src_bit: u8 = cols[2].parse().expect("src_bit");
        let dst_vec: u8 = cols[3].parse().expect("dst_vec");
        let dst_bit: u8 = cols[4].parse().expect("dst_bit");
        assert!((9..=56).contains(&l), "L={l} out of range");
        assert!(dst_vec < 8, "dst_vec out of range");
        let dst_width = [12u8, 12, 12, 12, 11, 11, 11, 7][dst_vec as usize];
        assert!(dst_bit < dst_width, "dst_bit {dst_bit} exceeds vec {dst_vec} width");
        per_l[(l - 9) as usize].push((src_param, src_bit, dst_vec, dst_bit));
    }

    for (i, rows) in per_l.iter().enumerate() {
        assert_eq!(rows.len(), 88, "L={}: expected 88 rows, got {}", i + 9, rows.len());
        // Destination coverage: every (dst_vec, dst_bit) appears exactly once.
        let mut seen = [[false; 12]; 8];
        for (_, _, v, b) in rows {
            assert!(!seen[*v as usize][*b as usize],
                "L={}: (dst_vec={v}, dst_bit={b}) appears twice", i + 9);
            seen[*v as usize][*b as usize] = true;
        }
        let widths = [12u8, 12, 12, 12, 11, 11, 11, 7];
        for (v, w) in widths.iter().enumerate() {
            for b in 0..*w {
                assert!(seen[v][b as usize],
                    "L={}: (dst_vec={v}, dst_bit={b}) never appears", i + 9);
            }
        }
    }

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/imbe_bit_prioritization.csv\n");
    out.push_str("// Do not edit — regenerated each build.\n");
    out.push_str("pub(crate) const IMBE_BIT_MAP: [[BitMap; 88]; 48] = [\n");
    for (l_idx, rows) in per_l.iter().enumerate() {
        out.push_str(&format!("    // L = {}\n    [\n", l_idx + 9));
        for (sp, sb, dv, db) in rows {
            out.push_str(&format!(
                "        BitMap {{ src_param: {sp}, src_bit: {sb}, dst_vec: {dv}, dst_bit: {db} }},\n"
            ));
        }
        out.push_str("    ],\n");
    }
    out.push_str("];\n");

    fs::write(out_dir.join("imbe_bit_priority.rs"), out).expect("write imbe_bit_priority.rs");
}

/// Parse `spec_tables/ambe_bit_prioritization.csv` into
/// `$OUT_DIR/ambe_bit_priority.rs`, emitting a flat `[BitMap; 49]`
/// (half-rate prioritization is L-independent).
fn gen_ambe_bit_prioritization(out_dir: &PathBuf) {
    let csv_path = "spec_tables/ambe_bit_prioritization.csv";
    println!("cargo:rerun-if-changed={csv_path}");

    let content = fs::read_to_string(csv_path)
        .unwrap_or_else(|e| panic!("failed to read {csv_path}: {e}"));

    let mut rows: Vec<(u8, u8, u8, u8)> = Vec::with_capacity(49);
    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("src_param") {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        assert_eq!(
            cols.len(),
            4,
            "ambe prioritization line {}: expected 4 columns, got {}: {raw:?}",
            lineno + 1,
            cols.len()
        );
        let src_param: u8 = cols[0].parse().expect("src_param");
        let src_bit: u8 = cols[1].parse().expect("src_bit");
        let dst_vec: u8 = cols[2].parse().expect("dst_vec");
        let dst_bit: u8 = cols[3].parse().expect("dst_bit");
        assert!(dst_vec < 4, "half-rate dst_vec out of range");
        // Half-rate vector widths per the CSV's header: û₀=12, û₁=12, û₂=11, û₃=14.
        let dst_width = [12u8, 12, 11, 14][dst_vec as usize];
        assert!(dst_bit < dst_width, "dst_bit {dst_bit} exceeds vec {dst_vec} width");
        rows.push((src_param, src_bit, dst_vec, dst_bit));
    }
    assert_eq!(rows.len(), 49, "ambe prioritization must have 49 entries");

    // Destination coverage.
    let widths = [12u8, 12, 11, 14];
    let mut seen = [[false; 14]; 4];
    for (_, _, v, b) in &rows {
        assert!(!seen[*v as usize][*b as usize],
            "ambe: (dst_vec={v}, dst_bit={b}) appears twice");
        seen[*v as usize][*b as usize] = true;
    }
    for (v, w) in widths.iter().enumerate() {
        for b in 0..*w {
            assert!(seen[v][b as usize],
                "ambe: (dst_vec={v}, dst_bit={b}) never appears");
        }
    }

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/ambe_bit_prioritization.csv\n");
    out.push_str("// Do not edit — regenerated each build.\n");
    out.push_str("pub(crate) const AMBE_BIT_MAP: [BitMap; 49] = [\n");
    for (sp, sb, dv, db) in &rows {
        out.push_str(&format!(
            "    BitMap {{ src_param: {sp}, src_bit: {sb}, dst_vec: {dv}, dst_bit: {db} }},\n"
        ));
    }
    out.push_str("];\n");

    fs::write(out_dir.join("ambe_bit_priority.rs"), out).expect("write ambe_bit_priority.rs");
}
