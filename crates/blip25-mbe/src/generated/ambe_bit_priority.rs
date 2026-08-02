// AMBE+2 half-rate bit-prioritization map: b-vector bit -> u-vector bit.
//
// TIA-102.BABA-A Annex data, with the reference hardware assignment where it
// deviates from the published standard. û₃ bits 5..8 (the four PRBA LSBs,
// b̂₃(0) and b̂₄(0..2)) follow the shipping the reference codec, NOT BABA-A Table 18 as
// printed: the published Table 18 rotates those four one position and does not
// match hardware (confirmed against two the reference implementations; the published
// order is also what JMBE carries). Do NOT "correct" them back to the spec.
// Read verbatim: do not reformat, "clean up", or reconstruct from a formula.
//
// The dump_tables_csv example emits this table as CSV, for diffing against the
// standard. It dumps blip25-codec's copy of the table, not this file:
//     cargo run --release -p blip25-codec --example dump_tables_csv

pub(crate) const AMBE_BIT_MAP: [BitMap; 49] = [
    BitMap { src_param: 2, src_bit: 1, dst_vec: 0, dst_bit: 0 },
    BitMap { src_param: 2, src_bit: 2, dst_vec: 0, dst_bit: 1 },
    BitMap { src_param: 2, src_bit: 3, dst_vec: 0, dst_bit: 2 },
    BitMap { src_param: 2, src_bit: 4, dst_vec: 0, dst_bit: 3 },
    BitMap { src_param: 1, src_bit: 1, dst_vec: 0, dst_bit: 4 },
    BitMap { src_param: 1, src_bit: 2, dst_vec: 0, dst_bit: 5 },
    BitMap { src_param: 1, src_bit: 3, dst_vec: 0, dst_bit: 6 },
    BitMap { src_param: 1, src_bit: 4, dst_vec: 0, dst_bit: 7 },
    BitMap { src_param: 0, src_bit: 3, dst_vec: 0, dst_bit: 8 },
    BitMap { src_param: 0, src_bit: 4, dst_vec: 0, dst_bit: 9 },
    BitMap { src_param: 0, src_bit: 5, dst_vec: 0, dst_bit: 10 },
    BitMap { src_param: 0, src_bit: 6, dst_vec: 0, dst_bit: 11 },
    BitMap { src_param: 4, src_bit: 3, dst_vec: 1, dst_bit: 0 },
    BitMap { src_param: 4, src_bit: 4, dst_vec: 1, dst_bit: 1 },
    BitMap { src_param: 4, src_bit: 5, dst_vec: 1, dst_bit: 2 },
    BitMap { src_param: 4, src_bit: 6, dst_vec: 1, dst_bit: 3 },
    BitMap { src_param: 3, src_bit: 1, dst_vec: 1, dst_bit: 4 },
    BitMap { src_param: 3, src_bit: 2, dst_vec: 1, dst_bit: 5 },
    BitMap { src_param: 3, src_bit: 3, dst_vec: 1, dst_bit: 6 },
    BitMap { src_param: 3, src_bit: 4, dst_vec: 1, dst_bit: 7 },
    BitMap { src_param: 3, src_bit: 5, dst_vec: 1, dst_bit: 8 },
    BitMap { src_param: 3, src_bit: 6, dst_vec: 1, dst_bit: 9 },
    BitMap { src_param: 3, src_bit: 7, dst_vec: 1, dst_bit: 10 },
    BitMap { src_param: 3, src_bit: 8, dst_vec: 1, dst_bit: 11 },
    BitMap { src_param: 8, src_bit: 2, dst_vec: 2, dst_bit: 0 },
    BitMap { src_param: 7, src_bit: 1, dst_vec: 2, dst_bit: 1 },
    BitMap { src_param: 7, src_bit: 2, dst_vec: 2, dst_bit: 2 },
    BitMap { src_param: 7, src_bit: 3, dst_vec: 2, dst_bit: 3 },
    BitMap { src_param: 6, src_bit: 1, dst_vec: 2, dst_bit: 4 },
    BitMap { src_param: 6, src_bit: 2, dst_vec: 2, dst_bit: 5 },
    BitMap { src_param: 6, src_bit: 3, dst_vec: 2, dst_bit: 6 },
    BitMap { src_param: 5, src_bit: 1, dst_vec: 2, dst_bit: 7 },
    BitMap { src_param: 5, src_bit: 2, dst_vec: 2, dst_bit: 8 },
    BitMap { src_param: 5, src_bit: 3, dst_vec: 2, dst_bit: 9 },
    BitMap { src_param: 5, src_bit: 4, dst_vec: 2, dst_bit: 10 },
    BitMap { src_param: 8, src_bit: 0, dst_vec: 3, dst_bit: 0 },
    BitMap { src_param: 8, src_bit: 1, dst_vec: 3, dst_bit: 1 },
    BitMap { src_param: 7, src_bit: 0, dst_vec: 3, dst_bit: 2 },
    BitMap { src_param: 6, src_bit: 0, dst_vec: 3, dst_bit: 3 },
    BitMap { src_param: 5, src_bit: 0, dst_vec: 3, dst_bit: 4 },
    BitMap { src_param: 3, src_bit: 0, dst_vec: 3, dst_bit: 5 },
    BitMap { src_param: 4, src_bit: 0, dst_vec: 3, dst_bit: 6 },
    BitMap { src_param: 4, src_bit: 1, dst_vec: 3, dst_bit: 7 },
    BitMap { src_param: 4, src_bit: 2, dst_vec: 3, dst_bit: 8 },
    BitMap { src_param: 0, src_bit: 0, dst_vec: 3, dst_bit: 9 },
    BitMap { src_param: 0, src_bit: 1, dst_vec: 3, dst_bit: 10 },
    BitMap { src_param: 0, src_bit: 2, dst_vec: 3, dst_bit: 11 },
    BitMap { src_param: 2, src_bit: 0, dst_vec: 3, dst_bit: 12 },
    BitMap { src_param: 1, src_bit: 0, dst_vec: 3, dst_bit: 13 },
];
