// Annex D — IMBE pitch-tracker FIR low-pass filter taps h_LPF(j).
//
// Normative TIA-102.BABA-A Annex data, fixed and verified against the
// published standard. Read verbatim: do not reformat, "clean up", or
// reconstruct these values from a formula.

/// Length of the Annex D FIR low-pass filter (n = −10..=10).
/// Annex D pitch-tracker LPF length (taps).
pub const ANNEX_D_LPF_LEN: usize = 21;

/// Annex D FIR low-pass filter h_LPF(n) used in initial pitch
/// autocorrelation. Indexed `[n + 10]`.
/// IMBE Annex D pitch-tracker LPF taps `h_LPF(j)` for j ∈ [-10, 10].
pub const IMBE_ANNEX_D_LPF: [f32; ANNEX_D_LPF_LEN] = [
    -0.002898, // n = -10
    -0.002831, // n = -9
    0.005666, // n = -8
    0.016601, // n = -7
    0.008800, // n = -6
    -0.026955, // n = -5
    -0.055990, // n = -4
    -0.015116, // n = -3
    0.118754, // n = -2
    0.278990, // n = -1
    0.351338, // n = 0
    0.278990, // n = 1
    0.118754, // n = 2
    -0.015116, // n = 3
    -0.055990, // n = 4
    -0.026955, // n = 5
    0.008800, // n = 6
    0.016601, // n = 7
    0.005666, // n = 8
    -0.002831, // n = 9
    -0.002898, // n = 10
];
