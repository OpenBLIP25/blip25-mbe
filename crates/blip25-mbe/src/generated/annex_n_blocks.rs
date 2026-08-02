// Annex N — AMBE+2 half-rate per-L block lengths.
//
// Normative TIA-102.BABA-A Annex data, fixed and verified against the
// published standard. Read verbatim: do not reformat, "clean up", or
// reconstruct these values from a formula.
//
// The dump_tables_csv example emits this table as CSV, for diffing against the
// standard. It dumps blip25-codec's copy of the table, not this file:
//     cargo run --release -p blip25-codec --example dump_tables_csv

/// AMBE+2 half-rate Annex N block-length table (48 L-values × 4 blocks).
pub const AMBE_BLOCK_LENGTHS: [[u8; 4]; 48] = [
    [2, 2, 2, 3], // L = 9
    [2, 2, 3, 3], // L = 10
    [2, 3, 3, 3], // L = 11
    [2, 3, 3, 4], // L = 12
    [3, 3, 3, 4], // L = 13
    [3, 3, 4, 4], // L = 14
    [3, 3, 4, 5], // L = 15
    [3, 4, 4, 5], // L = 16
    [3, 4, 5, 5], // L = 17
    [4, 4, 5, 5], // L = 18
    [4, 4, 5, 6], // L = 19
    [4, 4, 6, 6], // L = 20
    [4, 5, 6, 6], // L = 21
    [4, 5, 6, 7], // L = 22
    [5, 5, 6, 7], // L = 23
    [5, 5, 7, 7], // L = 24
    [5, 6, 7, 7], // L = 25
    [5, 6, 7, 8], // L = 26
    [5, 6, 8, 8], // L = 27
    [6, 6, 8, 8], // L = 28
    [6, 6, 8, 9], // L = 29
    [6, 7, 8, 9], // L = 30
    [6, 7, 9, 9], // L = 31
    [6, 7, 9, 10], // L = 32
    [7, 7, 9, 10], // L = 33
    [7, 8, 9, 10], // L = 34
    [7, 8, 10, 10], // L = 35
    [7, 8, 10, 11], // L = 36
    [8, 8, 10, 11], // L = 37
    [8, 9, 10, 11], // L = 38
    [8, 9, 11, 11], // L = 39
    [8, 9, 11, 12], // L = 40
    [8, 9, 11, 13], // L = 41
    [8, 9, 12, 13], // L = 42
    [8, 10, 12, 13], // L = 43
    [9, 10, 12, 13], // L = 44
    [9, 10, 12, 14], // L = 45
    [9, 10, 13, 14], // L = 46
    [9, 11, 13, 14], // L = 47
    [10, 11, 13, 14], // L = 48
    [10, 11, 13, 15], // L = 49
    [10, 11, 14, 15], // L = 50
    [10, 12, 14, 15], // L = 51
    [10, 12, 14, 16], // L = 52
    [11, 12, 14, 16], // L = 53
    [11, 12, 15, 16], // L = 54
    [11, 12, 15, 17], // L = 55
    [11, 13, 15, 17], // L = 56
];
