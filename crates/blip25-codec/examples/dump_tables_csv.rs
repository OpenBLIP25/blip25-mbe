//! Emit the frozen Annex tables back out as CSV.
//!
//! The tables are committed Rust source, and neither population may be
//! hand-edited: Annex L/M/N/S and the bit-prioritization map are normative
//! TIA-102 data; the PRBA/HOC/gain VQ codebooks read from `blip25_codebooks`
//! are firmware-recovered, not published spec tables (see
//! `crates/blip25-codebooks/ATTRIBUTION.md`). This example dumps them the other
//! way, one file per table, so a table can be diffed against its source of
//! record or handed to a tool that wants CSV. The dumps are outputs only:
//! nothing in the build reads them back.
//!
//! Usage: `cargo run --release --example dump_tables_csv [out-dir]`
//! (default `./table_csv`).

use std::fmt::Write as _;
use std::fs;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "table_csv".into());
    fs::create_dir_all(&dir).expect("create output dir");

    let mut wrote = 0usize;
    let mut emit = |name: &str, body: String| {
        let path = format!("{dir}/{name}.csv");
        fs::write(&path, body).unwrap_or_else(|e| panic!("write {path}: {e}"));
        println!("wrote {path}");
        wrote += 1;
    };

    // --- Annex L: pitch index -> (L, omega_0) ---
    {
        let mut s = String::from("index,l,omega_0\n");
        for (i, e) in blip25_codec::tables::AMBE_PITCH_TABLE.iter().enumerate() {
            let _ = writeln!(s, "{i},{},{}", e.l, e.omega_0);
        }
        emit("annex_l_pitch_table", s);
    }

    // --- Annex M: 32 x 8 V/UV codebook ---
    {
        let mut s = String::from("row,b0,b1,b2,b3,b4,b5,b6,b7\n");
        for (i, row) in blip25_codec::tables::AMBE_VUV_CODEBOOK.iter().enumerate() {
            let _ = write!(s, "{i}");
            for &v in row.iter() {
                let _ = write!(s, ",{}", u8::from(v));
            }
            s.push('\n');
        }
        emit("annex_m_vuv_codebook", s);
    }

    // --- Annex N: per-L block lengths ---
    {
        let mut s = String::from("l,j0,j1,j2,j3\n");
        for (i, row) in blip25_codec::tables::AMBE_BLOCK_LENGTHS.iter().enumerate() {
            let _ = writeln!(s, "{},{},{},{},{}", i + 9, row[0], row[1], row[2], row[3]);
        }
        emit("annex_n_block_lengths", s);
    }

    // --- Annex S: dibit interleave ---
    {
        let mut s = String::from("symbol,bit1_vec,bit1_idx,bit0_vec,bit0_idx\n");
        for (i, e) in blip25_codec::tables::ANNEX_S.iter().enumerate() {
            let _ = writeln!(
                s,
                "{i},{},{},{},{}",
                e.bit1_vec, e.bit1_idx, e.bit0_vec, e.bit0_idx
            );
        }
        emit("annex_s_interleave", s);
    }

    // --- Bit prioritization: b -> u scatter map ---
    {
        let mut s = String::from("src_param,src_bit,dst_vec,dst_bit\n");
        for m in blip25_codec::tables::AMBE_BIT_MAP.iter() {
            let _ = writeln!(
                s,
                "{},{},{},{}",
                m.src_param, m.src_bit, m.dst_vec, m.dst_bit
            );
        }
        emit("ambe_bit_prioritization", s);
    }

    // --- Firmware VQ codebooks (raw int16, the bit-exact form) ---
    for (name, cb) in [
        ("prba24", blip25_codebooks::prba24()),
        ("prba58", blip25_codebooks::prba58()),
        ("hoc_b5", blip25_codebooks::hoc_b5()),
        ("hoc_b6", blip25_codebooks::hoc_b6()),
        ("hoc_b7", blip25_codebooks::hoc_b7()),
        ("hoc_b8", blip25_codebooks::hoc_b8()),
    ] {
        let mut s = String::from("entry");
        for k in 0..cb.cols {
            let _ = write!(s, ",v{k}");
        }
        s.push('\n');
        for i in 0..cb.rows {
            let _ = write!(s, "{i}");
            for &v in cb.raw_row(i) {
                let _ = write!(s, ",{v}");
            }
            s.push('\n');
        }
        emit(name, s);
    }
    {
        let mut s = String::from("entry,value\n");
        for (i, &v) in blip25_codebooks::gain_o().iter().enumerate() {
            let _ = writeln!(s, "{i},{v}");
        }
        emit("gain_o", s);
    }

    println!("\n{wrote} tables written to {dir}/");
    println!(
        "Note: these are dumps FROM the committed source, which is authoritative. \
         Nothing in the build reads them back."
    );
}
