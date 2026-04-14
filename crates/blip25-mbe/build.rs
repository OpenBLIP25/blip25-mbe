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
    gen_annex_h(&out_dir);
}

/// Parse `spec_tables/annex_h_interleave.csv` into
/// `$OUT_DIR/annex_h.rs`, emitting the 72-entry `ANNEX_H` table.
fn gen_annex_h(out_dir: &PathBuf) {
    let csv_path = "spec_tables/annex_h_interleave.csv";
    println!("cargo:rerun-if-changed={csv_path}");
    println!("cargo:rerun-if-changed=build.rs");

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
