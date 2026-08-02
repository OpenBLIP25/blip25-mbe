// Annex O — AMBE+2 half-rate differential-gain levels.
//
// Normative TIA-102.BABA-A Annex data, fixed and verified against the
// published standard. Read verbatim: do not reformat, "clean up", or
// reconstruct these values from a formula.

/// AMBE+2 half-rate Annex O differential-gain levels (32 entries indexed by `b̂₂`).
pub const AMBE_GAIN_LEVELS: [f32; 32] = [
    -2.000000, // b̂₂ = 0
    -0.670000, // b̂₂ = 1
    0.297941, // b̂₂ = 2
    0.663728, // b̂₂ = 3
    1.036829, // b̂₂ = 4
    1.438136, // b̂₂ = 5
    1.890077, // b̂₂ = 6
    2.227_97, // b̂₂ = 7
    2.478289, // b̂₂ = 8
    2.667544, // b̂₂ = 9
    2.793619, // b̂₂ = 10
    2.893261, // b̂₂ = 11
    3.020_63, // b̂₂ = 12
    3.138586, // b̂₂ = 13
    3.237579, // b̂₂ = 14
    3.322_57, // b̂₂ = 15
    3.432367, // b̂₂ = 16
    3.571863, // b̂₂ = 17
    3.696_65, // b̂₂ = 18
    3.814917, // b̂₂ = 19
    3.920932, // b̂₂ = 20
    4.022503, // b̂₂ = 21
    4.123569, // b̂₂ = 22
    4.228291, // b̂₂ = 23
    4.370569, // b̂₂ = 24
    4.543_7, // b̂₂ = 25
    4.707695, // b̂₂ = 26
    4.848879, // b̂₂ = 27
    5.056757, // b̂₂ = 28
    5.326468, // b̂₂ = 29
    5.777581, // b̂₂ = 30
    6.874496, // b̂₂ = 31
];
