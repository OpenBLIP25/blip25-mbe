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
    gen_annex_f(&out_dir);
    gen_annex_g(&out_dir);
    gen_annex_i(&out_dir);
    gen_annex_j(&out_dir);
    gen_annex_s(&out_dir);
    gen_annex_l(&out_dir);
    gen_annex_m(&out_dir);
    gen_annex_n(&out_dir);
    gen_annex_o(&out_dir);
    gen_annex_p(&out_dir);
    gen_annex_q(&out_dir);
    gen_annex_r(&out_dir);
    gen_annex_t(&out_dir);
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

/// Parse `spec_tables/annex_f_gain_allocation.csv` into
/// `$OUT_DIR/annex_f_gain_alloc.rs`, emitting `IMBE_GAIN_ALLOC: [[GainAlloc; 5]; 48]`.
///
/// Indexed by `[L - 9][m - 3]` for `m ∈ {3, 4, 5, 6, 7}`. Each entry
/// carries `B_m` (bit count, 1–10) and `delta_m` (uniform-quantizer
/// step size, float).
fn gen_annex_f(out_dir: &PathBuf) {
    let csv_path = "spec_tables/annex_f_gain_allocation.csv";
    println!("cargo:rerun-if-changed={csv_path}");

    let content = fs::read_to_string(csv_path)
        .unwrap_or_else(|e| panic!("failed to read {csv_path}: {e}"));

    // [L_idx][m_idx] = (B_m, Delta_m)
    let mut table: Vec<Vec<(u8, f32)>> = (0..48).map(|_| Vec::with_capacity(5)).collect();

    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('L') {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        assert_eq!(
            cols.len(),
            4,
            "annex F line {}: expected 4 columns, got {}: {raw:?}",
            lineno + 1,
            cols.len()
        );
        let l: u8 = cols[0].parse().expect("L");
        let m: u8 = cols[1].parse().expect("m");
        let b_m: u8 = cols[2].parse().expect("B_m");
        let delta_m: f32 = cols[3].parse().expect("Delta_m");
        assert!((9..=56).contains(&l), "annex F: L={l} out of range");
        assert!((3..=7).contains(&m), "annex F: m={m} out of range");
        assert!(b_m >= 1 && b_m <= 10, "annex F: B_m={b_m} out of range");
        assert!(delta_m > 0.0, "annex F: delta_m must be positive");
        let l_idx = (l - 9) as usize;
        let m_idx = (m - 3) as usize;
        assert_eq!(
            table[l_idx].len(),
            m_idx,
            "annex F rows must be sorted by (L, m)"
        );
        table[l_idx].push((b_m, delta_m));
    }
    for (i, row) in table.iter().enumerate() {
        assert_eq!(row.len(), 5, "annex F: L={} expected 5 entries", i + 9);
    }

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_f_gain_allocation.csv\n");
    out.push_str("// Do not edit — regenerated each build.\n");
    out.push_str("pub(crate) const IMBE_GAIN_ALLOC: [[GainAlloc; 5]; 48] = [\n");
    for (l_idx, row) in table.iter().enumerate() {
        out.push_str(&format!("    // L = {}\n    [\n", l_idx + 9));
        for (b_m, delta_m) in row {
            out.push_str(&format!(
                "        GainAlloc {{ b_m: {b_m}, delta_m: {delta_m:.6} }},\n"
            ));
        }
        out.push_str("    ],\n");
    }
    out.push_str("];\n");

    fs::write(out_dir.join("annex_f_gain_alloc.rs"), out).expect("write annex_f_gain_alloc.rs");
}

/// Parse `spec_tables/annex_g_hoc_allocation.csv` into
/// `$OUT_DIR/annex_g_hoc_alloc.rs`. Variable-length per L̂; emits both
/// a flat entries array and a per-L̂ offset/length index.
///
/// Each entry holds `(C_i, C_k, b_m, B_m)`. Total rows: 1272 (sum of
/// `L̂ − 6` across L̂ ∈ [9, 56]).
fn gen_annex_g(out_dir: &PathBuf) {
    let csv_path = "spec_tables/annex_g_hoc_allocation.csv";
    println!("cargo:rerun-if-changed={csv_path}");

    let content = fs::read_to_string(csv_path)
        .unwrap_or_else(|e| panic!("failed to read {csv_path}: {e}"));

    let mut per_l: Vec<Vec<(u8, u8, u8, u8)>> = (0..48).map(|_| Vec::new()).collect();

    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('L') {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        assert_eq!(
            cols.len(),
            5,
            "annex G line {}: expected 5 columns, got {}: {raw:?}",
            lineno + 1,
            cols.len()
        );
        let l: u8 = cols[0].parse().expect("L");
        let c_i: u8 = cols[1].parse().expect("C_i");
        let c_k: u8 = cols[2].parse().expect("C_k");
        let b_m: u8 = cols[3].parse().expect("b_m");
        let b_m_bits: u8 = cols[4].parse().expect("B_m");
        assert!((9..=56).contains(&l), "annex G: L={l} out of range");
        assert!((1..=6).contains(&c_i), "annex G: C_i={c_i} out of range");
        assert!(c_k >= 2, "annex G: C_k={c_k} below 2");
        assert!(b_m_bits <= 10, "annex G: B_m={b_m_bits} exceeds 10");
        per_l[(l - 9) as usize].push((c_i, c_k, b_m, b_m_bits));
    }

    // Validate entry counts: L̂ − 6 per L̂.
    let mut total = 0usize;
    for (i, rows) in per_l.iter().enumerate() {
        let l = (i + 9) as u8;
        let expected = (l - 6) as usize;
        assert_eq!(
            rows.len(),
            expected,
            "annex G: L={l} expected {expected} rows, got {}",
            rows.len()
        );
        // b_m must increment by 1 starting at 8.
        for (j, row) in rows.iter().enumerate() {
            let expected_bm = 8 + j as u8;
            assert_eq!(row.2, expected_bm, "annex G: L={l} row {j} b_m mismatch");
        }
        total += rows.len();
    }
    assert_eq!(total, 1272, "annex G: total row count mismatch");

    // Flatten + emit offset table.
    let mut flat: Vec<(u8, u8, u8, u8)> = Vec::with_capacity(total);
    let mut offsets: [(u32, u32); 48] = [(0, 0); 48];
    for (i, rows) in per_l.iter().enumerate() {
        let off = flat.len() as u32;
        let len = rows.len() as u32;
        offsets[i] = (off, len);
        flat.extend_from_slice(rows);
    }

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_g_hoc_allocation.csv\n");
    out.push_str("// Do not edit — regenerated each build.\n");
    out.push_str(&format!(
        "pub(crate) const IMBE_HOC_ENTRIES: [HocAlloc; {}] = [\n",
        flat.len()
    ));
    for (c_i, c_k, b_m, b_m_bits) in &flat {
        out.push_str(&format!(
            "    HocAlloc {{ c_i: {c_i}, c_k: {c_k}, b_m: {b_m}, b_m_bits: {b_m_bits} }},\n"
        ));
    }
    out.push_str("];\n\n");
    out.push_str("/// `(offset, len)` pairs into IMBE_HOC_ENTRIES, indexed by `L − 9`.\n");
    out.push_str("pub(crate) const IMBE_HOC_OFFSETS: [(u32, u32); 48] = [\n");
    for (off, len) in &offsets {
        out.push_str(&format!("    ({off}, {len}),\n"));
    }
    out.push_str("];\n");

    fs::write(out_dir.join("annex_g_hoc_alloc.rs"), out).expect("write annex_g_hoc_alloc.rs");
}

/// Parse `spec_tables/annex_i_synthesis_window.csv` into
/// `$OUT_DIR/annex_i_synth_window.rs`, emitting `IMBE_SYNTH_WINDOW`
/// (`[f32; 211]`) and the `SYNTH_WINDOW_LEN` constant.
///
/// The window covers `n = −105..=105`. Index 0 of the array corresponds
/// to `n = −105`. Validates length 211 and even symmetry around `n = 0`.
fn gen_annex_i(out_dir: &PathBuf) {
    let csv_path = "spec_tables/annex_i_synthesis_window.csv";
    println!("cargo:rerun-if-changed={csv_path}");

    let content = fs::read_to_string(csv_path)
        .unwrap_or_else(|e| panic!("failed to read {csv_path}: {e}"));

    let mut samples: Vec<(i32, f32)> = Vec::with_capacity(211);
    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('n') {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        assert_eq!(
            cols.len(),
            2,
            "annex I line {}: expected 2 columns, got {}: {raw:?}",
            lineno + 1,
            cols.len()
        );
        let n: i32 = cols[0].parse().expect("n");
        let w: f32 = cols[1].parse().expect("wS_n");
        assert!((-105..=105).contains(&n), "annex I: n={n} out of range");
        let expected_idx = (n + 105) as usize;
        assert_eq!(
            samples.len(),
            expected_idx,
            "annex I rows must be sequential n = −105..=105"
        );
        assert!(w >= 0.0, "annex I: wS({n}) = {w} negative");
        samples.push((n, w));
    }
    assert_eq!(samples.len(), 211, "Annex I must have 211 entries");

    // Check even symmetry: wS(-n) == wS(n).
    for k in 1..=105 {
        let a = samples[(105 - k) as usize].1;
        let b = samples[(105 + k) as usize].1;
        assert!(
            (a - b).abs() < 1e-6,
            "annex I: wS({}) = {a} != wS({}) = {b}",
            -k,
            k
        );
    }

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_i_synthesis_window.csv\n");
    out.push_str("// Do not edit — regenerated each build.\n");
    out.push_str("/// Length of the speech synthesis window (n = −105..=105).\n");
    out.push_str("pub const SYNTH_WINDOW_LEN: usize = 211;\n\n");
    out.push_str("/// Speech synthesis window wS(n) per BABA-A Annex I.\n");
    out.push_str("/// Indexed `[n + 105]` so `IMBE_SYNTH_WINDOW[0] = wS(−105)`.\n");
    out.push_str("pub const IMBE_SYNTH_WINDOW: [f32; SYNTH_WINDOW_LEN] = [\n");
    for (n, w) in &samples {
        out.push_str(&format!("    {w:.6}, // n = {n}\n"));
    }
    out.push_str("];\n");

    fs::write(out_dir.join("annex_i_synth_window.rs"), out)
        .expect("write annex_i_synth_window.rs");
}

/// Parse `spec_tables/annex_j_block_lengths.csv` into
/// `$OUT_DIR/annex_j_blocks.rs`, emitting `IMBE_BLOCK_LENGTHS: [[u8; 6]; 48]`.
///
/// Validates Eq. 65 (`Σ J̃_i = L̃`) and Eq. 66 (`⌊L/6⌋ ≤ J̃_i ≤ J̃_{i+1} ≤ ⌈L/6⌉`).
fn gen_annex_j(out_dir: &PathBuf) {
    let csv_path = "spec_tables/annex_j_block_lengths.csv";
    println!("cargo:rerun-if-changed={csv_path}");

    let content = fs::read_to_string(csv_path)
        .unwrap_or_else(|e| panic!("failed to read {csv_path}: {e}"));

    let mut blocks: Vec<[u8; 6]> = Vec::with_capacity(48);

    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('L') {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        assert_eq!(
            cols.len(),
            7,
            "annex J line {}: expected 7 columns, got {}: {raw:?}",
            lineno + 1,
            cols.len()
        );
        let l: u8 = cols[0].parse().expect("L");
        assert!((9..=56).contains(&l));
        assert_eq!((l - 9) as usize, blocks.len(), "annex J rows must be sequential");
        let mut row = [0u8; 6];
        for i in 0..6 {
            row[i] = cols[i + 1].parse().expect("J_i");
        }
        // Eq. 65: sum equals L.
        let sum: u32 = row.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, l as u32, "annex J: L={l}, sum(J)={sum}");
        // Eq. 66: ⌊L/6⌋ ≤ J̃_i ≤ J̃_{i+1} ≤ ⌈L/6⌉.
        let lo = l / 6;
        let hi = (l + 5) / 6;
        for &j in &row {
            assert!(j >= lo && j <= hi, "annex J: L={l}, J̃ out of range");
        }
        for i in 0..5 {
            assert!(row[i] <= row[i + 1], "annex J: L={l} blocks not non-decreasing");
        }
        blocks.push(row);
    }
    assert_eq!(blocks.len(), 48, "annex J: expected 48 rows");

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_j_block_lengths.csv\n");
    out.push_str("// Do not edit — regenerated each build.\n");
    out.push_str("pub(crate) const IMBE_BLOCK_LENGTHS: [[u8; 6]; 48] = [\n");
    for (l_idx, row) in blocks.iter().enumerate() {
        out.push_str(&format!(
            "    [{}, {}, {}, {}, {}, {}], // L = {}\n",
            row[0], row[1], row[2], row[3], row[4], row[5], l_idx + 9
        ));
    }
    out.push_str("];\n");

    fs::write(out_dir.join("annex_j_blocks.rs"), out).expect("write annex_j_blocks.rs");
}

/// Parse `spec_tables/annex_s_interleave.csv` into
/// `$OUT_DIR/annex_s.rs`, emitting the 36-entry `ANNEX_S` table for
/// half-rate interleaving.
///
/// Half-rate code vectors have widths c₀=24, c₁=23, c₂=11, c₃=14
/// (total 72 bits = 36 dibits). Validates that every
/// `(vector, bit_index)` across all four vectors appears exactly
/// once.
fn gen_annex_s(out_dir: &PathBuf) {
    let csv_path = "spec_tables/annex_s_interleave.csv";
    println!("cargo:rerun-if-changed={csv_path}");

    let content = fs::read_to_string(csv_path)
        .unwrap_or_else(|e| panic!("failed to read {csv_path}: {e}"));

    let mut entries: Vec<(u8, u8, u8, u8)> = Vec::with_capacity(36);
    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with("symbol") {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        assert_eq!(
            cols.len(),
            5,
            "Annex S line {}: expected 5 columns, got {}: {raw:?}",
            lineno + 1,
            cols.len()
        );
        let symbol: usize = cols[0].parse().expect("symbol");
        let bit1_vec: u8 = cols[1].parse().expect("bit1_vector");
        let bit1_idx: u8 = cols[2].parse().expect("bit1_index");
        let bit0_vec: u8 = cols[3].parse().expect("bit0_vector");
        let bit0_idx: u8 = cols[4].parse().expect("bit0_index");
        assert_eq!(symbol, entries.len(), "Annex S symbols must be sequential 0..36");
        assert!(bit1_vec < 4 && bit0_vec < 4, "vector index out of range");
        entries.push((bit1_vec, bit1_idx, bit0_vec, bit0_idx));
    }
    assert_eq!(entries.len(), 36, "Annex S must have 36 symbols");

    // Half-rate code vector widths. c₀ and c₁ are FEC-protected;
    // c₂ and c₃ are uncoded bit streams (not FEC codewords despite the
    // `c_i` naming).
    const AMBE_CODE_WIDTHS: [u8; 4] = [24, 23, 11, 14];
    let mut seen = [[false; 24]; 4];
    for (bit1_vec, bit1_idx, bit0_vec, bit0_idx) in &entries {
        for &(v, i) in &[(*bit1_vec, *bit1_idx), (*bit0_vec, *bit0_idx)] {
            assert!(
                i < AMBE_CODE_WIDTHS[v as usize],
                "Annex S: vec {v} idx {i} exceeds width"
            );
            assert!(!seen[v as usize][i as usize], "Annex S: ({v}, {i}) twice");
            seen[v as usize][i as usize] = true;
        }
    }
    for (v, row) in seen.iter().enumerate() {
        for (i, &b) in row.iter().enumerate().take(AMBE_CODE_WIDTHS[v] as usize) {
            assert!(b, "Annex S: ({v}, {i}) never appears");
        }
    }

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_s_interleave.csv\n");
    out.push_str("// Do not edit — regenerated each build.\n");
    out.push_str("pub(crate) const ANNEX_S: [AnnexSEntry; 36] = [\n");
    for (bit1_vec, bit1_idx, bit0_vec, bit0_idx) in &entries {
        out.push_str(&format!(
            "    AnnexSEntry {{ bit1_vec: {bit1_vec}, bit1_idx: {bit1_idx}, \
             bit0_vec: {bit0_vec}, bit0_idx: {bit0_idx} }},\n"
        ));
    }
    out.push_str("];\n");

    fs::write(out_dir.join("annex_s.rs"), out).expect("write annex_s.rs");
}

// ---------------------------------------------------------------------------
// Half-rate codebooks (Annex L / M / N / O / P / Q / R)
// ---------------------------------------------------------------------------

/// Generic helper: read a CSV, skip comment and header lines, parse
/// each remaining line into `cols` fields, and pass each row's parsed
/// values to `row_fn`.
fn parse_csv_rows(
    csv_path: &str,
    expected_cols: usize,
    mut row_fn: impl FnMut(usize, &[&str]),
) -> usize {
    println!("cargo:rerun-if-changed={csv_path}");
    let content = fs::read_to_string(csv_path)
        .unwrap_or_else(|e| panic!("failed to read {csv_path}: {e}"));
    let mut count = 0;
    let mut saw_header = false;
    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !saw_header && line.chars().next().map_or(false, |c| c.is_ascii_alphabetic()) {
            saw_header = true;
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        assert_eq!(
            cols.len(),
            expected_cols,
            "{csv_path} line {}: expected {expected_cols} cols, got {}",
            lineno + 1,
            cols.len()
        );
        row_fn(count, &cols);
        count += 1;
    }
    count
}

/// Annex L — half-rate pitch quantization table (120 entries).
/// Emits `AMBE_PITCH_TABLE: [PitchEntry; 120]` indexed by `b̂₀`.
///
/// CSV stores `omega_0` in cycles/sample (= ω₀ / 2π). BABA-A §13.1
/// states the half-rate voice range as `2π/123.125 ≤ ω̂₀ ≤ 2π/19.875`
/// rad/sample, which numerically matches `2π × CSV` to six places.
/// We convert at load time so every consumer reads ω₀ in rad/sample
/// — the unit MbeParams uses everywhere else.
/// See `analysis/vocoder_decode_disambiguations.md` §13.
fn gen_annex_l(out_dir: &PathBuf) {
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut entries: Vec<(u8, u8, f32)> = Vec::with_capacity(120);
    parse_csv_rows(
        "spec_tables/annex_l_pitch_table.csv",
        3,
        |row_idx, cols| {
            let b0: u8 = cols[0].parse().expect("b0");
            let l: u8 = cols[1].parse().expect("L");
            let w_cycles: f32 = cols[2].parse().expect("omega_0");
            assert_eq!(b0 as usize, row_idx, "Annex L must be sequential");
            assert!((9..=56).contains(&l), "Annex L L={l} out of range");
            assert!(w_cycles > 0.0, "Annex L omega_0 must be positive");
            entries.push((b0, l, w_cycles * two_pi));
        },
    );
    assert_eq!(entries.len(), 120, "Annex L must have 120 entries");
    // ω₀ strictly decreases with b̂₀.
    for w in entries.windows(2) {
        assert!(w[0].2 > w[1].2, "Annex L ω₀ not monotone decreasing");
    }

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_l_pitch_table.csv\n");
    out.push_str("// CSV stores cycles/sample; values converted to rad/sample at load.\n");
    out.push_str("pub const AMBE_PITCH_TABLE: [PitchEntry; 120] = [\n");
    for (_, l, w) in &entries {
        out.push_str(&format!("    PitchEntry {{ l: {l}, omega_0: {w:.6} }},\n"));
    }
    out.push_str("];\n");
    fs::write(out_dir.join("annex_l_pitch.rs"), out).expect("write annex_l_pitch.rs");
}

/// Annex M — half-rate V/UV codebook (32 entries, 8 bits each).
fn gen_annex_m(out_dir: &PathBuf) {
    let mut rows: Vec<[u8; 8]> = Vec::with_capacity(32);
    parse_csv_rows(
        "spec_tables/annex_m_vuv_codebook.csv",
        9,
        |row_idx, cols| {
            let b1: u8 = cols[0].parse().expect("b1");
            assert_eq!(b1 as usize, row_idx);
            let mut v = [0u8; 8];
            for i in 0..8 {
                let bit: u8 = cols[i + 1].parse().expect("v_k");
                assert!(bit <= 1, "V/UV bit must be 0 or 1");
                v[i] = bit;
            }
            rows.push(v);
        },
    );
    assert_eq!(rows.len(), 32);

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_m_vuv_codebook.csv\n");
    out.push_str("pub const AMBE_VUV_CODEBOOK: [[bool; 8]; 32] = [\n");
    for row in &rows {
        out.push_str(&format!(
            "    [{}, {}, {}, {}, {}, {}, {}, {}],\n",
            row[0] == 1, row[1] == 1, row[2] == 1, row[3] == 1,
            row[4] == 1, row[5] == 1, row[6] == 1, row[7] == 1,
        ));
    }
    out.push_str("];\n");
    fs::write(out_dir.join("annex_m_vuv.rs"), out).expect("write annex_m_vuv.rs");
}

/// Annex N — half-rate block lengths (4 blocks, indexed by L − 9).
fn gen_annex_n(out_dir: &PathBuf) {
    let mut rows: Vec<[u8; 4]> = Vec::with_capacity(48);
    parse_csv_rows(
        "spec_tables/annex_n_block_lengths.csv",
        5,
        |row_idx, cols| {
            let l: u8 = cols[0].parse().expect("L");
            assert_eq!((l - 9) as usize, row_idx);
            let mut r = [0u8; 4];
            for i in 0..4 {
                r[i] = cols[i + 1].parse().expect("J_i");
            }
            let sum: u32 = r.iter().map(|&x| x as u32).sum();
            assert_eq!(sum, l as u32, "Annex N L={l}: Σ J̃_i = {sum}");
            rows.push(r);
        },
    );
    assert_eq!(rows.len(), 48);

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_n_block_lengths.csv\n");
    out.push_str("pub const AMBE_BLOCK_LENGTHS: [[u8; 4]; 48] = [\n");
    for (l_idx, r) in rows.iter().enumerate() {
        out.push_str(&format!(
            "    [{}, {}, {}, {}], // L = {}\n",
            r[0], r[1], r[2], r[3], l_idx + 9
        ));
    }
    out.push_str("];\n");
    fs::write(out_dir.join("annex_n_blocks.rs"), out).expect("write annex_n_blocks.rs");
}

/// Annex O — half-rate gain quantizer (32 levels).
fn gen_annex_o(out_dir: &PathBuf) {
    let mut levels: Vec<f32> = Vec::with_capacity(32);
    parse_csv_rows(
        "spec_tables/annex_o_gain_quantizer.csv",
        2,
        |row_idx, cols| {
            let b2: u8 = cols[0].parse().expect("b2");
            assert_eq!(b2 as usize, row_idx);
            let lvl: f32 = cols[1].parse().expect("gain_level");
            levels.push(lvl);
        },
    );
    assert_eq!(levels.len(), 32);
    for w in levels.windows(2) {
        assert!(w[0] < w[1], "Annex O not monotone increasing");
    }

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_o_gain_quantizer.csv\n");
    out.push_str("pub const AMBE_GAIN_LEVELS: [f32; 32] = [\n");
    for (i, lvl) in levels.iter().enumerate() {
        out.push_str(&format!("    {lvl:.6}, // b̂₂ = {i}\n"));
    }
    out.push_str("];\n");
    fs::write(out_dir.join("annex_o_gain.rs"), out).expect("write annex_o_gain.rs");
}

/// Annex P — PRBA24 VQ (512 entries × 3 values).
fn gen_annex_p(out_dir: &PathBuf) {
    let mut rows: Vec<[f32; 3]> = Vec::with_capacity(512);
    parse_csv_rows(
        "spec_tables/annex_p_prba24_codebook.csv",
        4,
        |row_idx, cols| {
            let b3: u16 = cols[0].parse().expect("b3");
            assert_eq!(b3 as usize, row_idx);
            rows.push([
                cols[1].parse().expect("G2"),
                cols[2].parse().expect("G3"),
                cols[3].parse().expect("G4"),
            ]);
        },
    );
    assert_eq!(rows.len(), 512);

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_p_prba24_codebook.csv\n");
    out.push_str("pub const AMBE_PRBA24: [[f32; 3]; 512] = [\n");
    for r in &rows {
        out.push_str(&format!("    [{:.6}, {:.6}, {:.6}],\n", r[0], r[1], r[2]));
    }
    out.push_str("];\n");
    fs::write(out_dir.join("annex_p_prba24.rs"), out).expect("write annex_p_prba24.rs");
}

/// Annex Q — PRBA58 VQ (128 entries × 4 values).
fn gen_annex_q(out_dir: &PathBuf) {
    let mut rows: Vec<[f32; 4]> = Vec::with_capacity(128);
    parse_csv_rows(
        "spec_tables/annex_q_prba58_codebook.csv",
        5,
        |row_idx, cols| {
            let b4: u8 = cols[0].parse().expect("b4");
            assert_eq!(b4 as usize, row_idx);
            rows.push([
                cols[1].parse().expect("G5"),
                cols[2].parse().expect("G6"),
                cols[3].parse().expect("G7"),
                cols[4].parse().expect("G8"),
            ]);
        },
    );
    assert_eq!(rows.len(), 128);

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_q_prba58_codebook.csv\n");
    out.push_str("pub const AMBE_PRBA58: [[f32; 4]; 128] = [\n");
    for r in &rows {
        out.push_str(&format!(
            "    [{:.6}, {:.6}, {:.6}, {:.6}],\n",
            r[0], r[1], r[2], r[3]
        ));
    }
    out.push_str("];\n");
    fs::write(out_dir.join("annex_q_prba58.rs"), out).expect("write annex_q_prba58.rs");
}

/// Annex R — four HOC VQ tables (32/16/16/8 entries × 4 values).
/// Emits four const arrays: `AMBE_HOC_B5` … `AMBE_HOC_B8`.
fn gen_annex_r(out_dir: &PathBuf) {
    let specs = [
        ("spec_tables/annex_r_hoc_b5.csv", 32usize, "AMBE_HOC_B5", "b5"),
        ("spec_tables/annex_r_hoc_b6.csv", 16, "AMBE_HOC_B6", "b6"),
        ("spec_tables/annex_r_hoc_b7.csv", 16, "AMBE_HOC_B7", "b7"),
        ("spec_tables/annex_r_hoc_b8.csv", 8,  "AMBE_HOC_B8", "b8"),
    ];

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_r_hoc_b{5..8}.csv\n");

    for (path, expected_count, const_name, col_label) in specs {
        let mut rows: Vec<[f32; 4]> = Vec::with_capacity(expected_count);
        parse_csv_rows(path, 5, |row_idx, cols| {
            let b: u16 = cols[0].parse().expect(col_label);
            assert_eq!(b as usize, row_idx);
            rows.push([
                cols[1].parse().expect("H_i,1"),
                cols[2].parse().expect("H_i,2"),
                cols[3].parse().expect("H_i,3"),
                cols[4].parse().expect("H_i,4"),
            ]);
        });
        assert_eq!(rows.len(), expected_count, "{path}: row count");

        out.push_str(&format!(
            "pub const {const_name}: [[f32; 4]; {expected_count}] = [\n"
        ));
        for r in &rows {
            out.push_str(&format!(
                "    [{:.6}, {:.6}, {:.6}, {:.6}],\n",
                r[0], r[1], r[2], r[3]
            ));
        }
        out.push_str("];\n\n");
    }

    fs::write(out_dir.join("annex_r_hoc.rs"), out).expect("write annex_r_hoc.rs");
}

/// Annex T — tone frame parameters, sparse table keyed by `I_D`.
///
/// The CSV skips reserved ID ranges (0–4, 123–127, 164–254). We emit
/// a dense 256-entry `Option<ToneParams>` table so the decoder can
/// index directly by `I_D`.
fn gen_annex_t(out_dir: &PathBuf) {
    let mut rows: Vec<(u8, f32, u8, u8)> = Vec::new();
    parse_csv_rows(
        "spec_tables/annex_t_tone_params.csv",
        4,
        |_row_idx, cols| {
            let id: u8 = cols[0].parse().expect("tone_id");
            let f0: f32 = cols[1].parse().expect("f0");
            let l1: u8 = cols[2].parse().expect("l1");
            let l2: u8 = cols[3].parse().expect("l2");
            rows.push((id, f0, l1, l2));
        },
    );
    assert!(!rows.is_empty(), "Annex T must have at least one row");

    // Sanity: IDs should be strictly increasing.
    for w in rows.windows(2) {
        assert!(w[0].0 < w[1].0, "Annex T IDs not strictly increasing");
    }

    // Emit a dense 256-entry lookup — None for reserved slots, Some(...) for live rows.
    let mut table: [Option<(f32, u8, u8)>; 256] = [None; 256];
    for (id, f0, l1, l2) in &rows {
        table[*id as usize] = Some((*f0, *l1, *l2));
    }

    let mut out = String::new();
    out.push_str("// Auto-generated from spec_tables/annex_t_tone_params.csv\n");
    out.push_str("pub const ANNEX_T: [Option<ToneParams>; 256] = [\n");
    for entry in &table {
        match entry {
            None => out.push_str("    None,\n"),
            Some((f0, l1, l2)) => out.push_str(&format!(
                "    Some(ToneParams {{ f0: {f0:.4}, l1: {l1}, l2: {l2} }}),\n"
            )),
        }
    }
    out.push_str("];\n");

    fs::write(out_dir.join("annex_t_tones.rs"), out).expect("write annex_t_tones.rs");
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
