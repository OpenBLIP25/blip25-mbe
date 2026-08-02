// Annex S — AMBE+2 half-rate dibit interleave table.
//
// Normative TIA-102.BABA-A Annex data, fixed and verified against the
// published standard. Read verbatim: do not reformat, "clean up", or
// reconstruct these values from a formula.
//
// The dump_tables_csv example emits this table as CSV, for diffing against the
// standard. It dumps blip25-codec's copy of the table, not this file:
//     cargo run --release -p blip25-codec --example dump_tables_csv

pub(crate) const ANNEX_S: [AnnexSEntry; 36] = [
    AnnexSEntry { bit1_vec: 0, bit1_idx: 23, bit0_vec: 0, bit0_idx: 5 },
    AnnexSEntry { bit1_vec: 1, bit1_idx: 10, bit0_vec: 2, bit0_idx: 3 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 22, bit0_vec: 0, bit0_idx: 4 },
    AnnexSEntry { bit1_vec: 1, bit1_idx: 9, bit0_vec: 2, bit0_idx: 2 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 21, bit0_vec: 0, bit0_idx: 3 },
    AnnexSEntry { bit1_vec: 1, bit1_idx: 8, bit0_vec: 2, bit0_idx: 1 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 20, bit0_vec: 0, bit0_idx: 2 },
    AnnexSEntry { bit1_vec: 1, bit1_idx: 7, bit0_vec: 2, bit0_idx: 0 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 19, bit0_vec: 0, bit0_idx: 1 },
    AnnexSEntry { bit1_vec: 1, bit1_idx: 6, bit0_vec: 3, bit0_idx: 13 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 18, bit0_vec: 0, bit0_idx: 0 },
    AnnexSEntry { bit1_vec: 1, bit1_idx: 5, bit0_vec: 3, bit0_idx: 12 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 17, bit0_vec: 1, bit0_idx: 22 },
    AnnexSEntry { bit1_vec: 1, bit1_idx: 4, bit0_vec: 3, bit0_idx: 11 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 16, bit0_vec: 1, bit0_idx: 21 },
    AnnexSEntry { bit1_vec: 1, bit1_idx: 3, bit0_vec: 3, bit0_idx: 10 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 15, bit0_vec: 1, bit0_idx: 20 },
    AnnexSEntry { bit1_vec: 1, bit1_idx: 2, bit0_vec: 3, bit0_idx: 9 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 14, bit0_vec: 1, bit0_idx: 19 },
    AnnexSEntry { bit1_vec: 1, bit1_idx: 1, bit0_vec: 3, bit0_idx: 8 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 13, bit0_vec: 1, bit0_idx: 18 },
    AnnexSEntry { bit1_vec: 1, bit1_idx: 0, bit0_vec: 3, bit0_idx: 7 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 12, bit0_vec: 1, bit0_idx: 17 },
    AnnexSEntry { bit1_vec: 2, bit1_idx: 10, bit0_vec: 3, bit0_idx: 6 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 11, bit0_vec: 1, bit0_idx: 16 },
    AnnexSEntry { bit1_vec: 2, bit1_idx: 9, bit0_vec: 3, bit0_idx: 5 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 10, bit0_vec: 1, bit0_idx: 15 },
    AnnexSEntry { bit1_vec: 2, bit1_idx: 8, bit0_vec: 3, bit0_idx: 4 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 9, bit0_vec: 1, bit0_idx: 14 },
    AnnexSEntry { bit1_vec: 2, bit1_idx: 7, bit0_vec: 3, bit0_idx: 3 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 8, bit0_vec: 1, bit0_idx: 13 },
    AnnexSEntry { bit1_vec: 2, bit1_idx: 6, bit0_vec: 3, bit0_idx: 2 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 7, bit0_vec: 1, bit0_idx: 12 },
    AnnexSEntry { bit1_vec: 2, bit1_idx: 5, bit0_vec: 3, bit0_idx: 1 },
    AnnexSEntry { bit1_vec: 0, bit1_idx: 6, bit0_vec: 1, bit0_idx: 11 },
    AnnexSEntry { bit1_vec: 2, bit1_idx: 4, bit0_vec: 3, bit0_idx: 0 },
];
