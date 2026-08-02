// Annex F — IMBE gain bit allocation and step sizes, indexed by L - 9.
//
// Normative TIA-102.BABA-A Annex data, fixed and verified against the
// published standard. Read verbatim: do not reformat, "clean up", or
// reconstruct these values from a formula.

pub(crate) const IMBE_GAIN_ALLOC: [[GainAlloc; 5]; 48] = [
    // L = 9
    [
        GainAlloc { b_m: 10, delta_m: 0.003100 },
        GainAlloc { b_m: 9, delta_m: 0.004020 },
        GainAlloc { b_m: 9, delta_m: 0.003360 },
        GainAlloc { b_m: 9, delta_m: 0.002900 },
        GainAlloc { b_m: 9, delta_m: 0.002640 },
    ],
    // L = 10
    [
        GainAlloc { b_m: 9, delta_m: 0.006200 },
        GainAlloc { b_m: 9, delta_m: 0.004020 },
        GainAlloc { b_m: 8, delta_m: 0.006720 },
        GainAlloc { b_m: 8, delta_m: 0.005800 },
        GainAlloc { b_m: 8, delta_m: 0.005280 },
    ],
    // L = 11
    [
        GainAlloc { b_m: 8, delta_m: 0.012400 },
        GainAlloc { b_m: 8, delta_m: 0.008040 },
        GainAlloc { b_m: 8, delta_m: 0.006720 },
        GainAlloc { b_m: 7, delta_m: 0.011600 },
        GainAlloc { b_m: 7, delta_m: 0.010560 },
    ],
    // L = 12
    [
        GainAlloc { b_m: 8, delta_m: 0.012400 },
        GainAlloc { b_m: 7, delta_m: 0.016080 },
        GainAlloc { b_m: 7, delta_m: 0.013440 },
        GainAlloc { b_m: 7, delta_m: 0.011600 },
        GainAlloc { b_m: 7, delta_m: 0.010560 },
    ],
    // L = 13
    [
        GainAlloc { b_m: 7, delta_m: 0.024800 },
        GainAlloc { b_m: 7, delta_m: 0.016080 },
        GainAlloc { b_m: 7, delta_m: 0.013440 },
        GainAlloc { b_m: 6, delta_m: 0.021750 },
        GainAlloc { b_m: 6, delta_m: 0.019800 },
    ],
    // L = 14
    [
        GainAlloc { b_m: 7, delta_m: 0.024800 },
        GainAlloc { b_m: 6, delta_m: 0.030150 },
        GainAlloc { b_m: 6, delta_m: 0.025200 },
        GainAlloc { b_m: 6, delta_m: 0.021750 },
        GainAlloc { b_m: 6, delta_m: 0.019800 },
    ],
    // L = 15
    [
        GainAlloc { b_m: 7, delta_m: 0.024800 },
        GainAlloc { b_m: 6, delta_m: 0.030150 },
        GainAlloc { b_m: 6, delta_m: 0.025200 },
        GainAlloc { b_m: 6, delta_m: 0.021750 },
        GainAlloc { b_m: 5, delta_m: 0.036960 },
    ],
    // L = 16
    [
        GainAlloc { b_m: 6, delta_m: 0.046500 },
        GainAlloc { b_m: 6, delta_m: 0.030150 },
        GainAlloc { b_m: 6, delta_m: 0.025200 },
        GainAlloc { b_m: 5, delta_m: 0.040600 },
        GainAlloc { b_m: 5, delta_m: 0.036960 },
    ],
    // L = 17
    [
        GainAlloc { b_m: 6, delta_m: 0.046500 },
        GainAlloc { b_m: 6, delta_m: 0.030150 },
        GainAlloc { b_m: 5, delta_m: 0.047040 },
        GainAlloc { b_m: 5, delta_m: 0.040600 },
        GainAlloc { b_m: 5, delta_m: 0.036960 },
    ],
    // L = 18
    [
        GainAlloc { b_m: 6, delta_m: 0.046500 },
        GainAlloc { b_m: 5, delta_m: 0.056280 },
        GainAlloc { b_m: 5, delta_m: 0.047040 },
        GainAlloc { b_m: 5, delta_m: 0.040600 },
        GainAlloc { b_m: 5, delta_m: 0.036960 },
    ],
    // L = 19
    [
        GainAlloc { b_m: 6, delta_m: 0.046500 },
        GainAlloc { b_m: 5, delta_m: 0.056280 },
        GainAlloc { b_m: 5, delta_m: 0.047040 },
        GainAlloc { b_m: 4, delta_m: 0.058000 },
        GainAlloc { b_m: 4, delta_m: 0.052800 },
    ],
    // L = 20
    [
        GainAlloc { b_m: 6, delta_m: 0.046500 },
        GainAlloc { b_m: 5, delta_m: 0.056280 },
        GainAlloc { b_m: 5, delta_m: 0.047040 },
        GainAlloc { b_m: 4, delta_m: 0.058000 },
        GainAlloc { b_m: 4, delta_m: 0.052800 },
    ],
    // L = 21
    [
        GainAlloc { b_m: 5, delta_m: 0.086800 },
        GainAlloc { b_m: 5, delta_m: 0.056280 },
        GainAlloc { b_m: 5, delta_m: 0.047040 },
        GainAlloc { b_m: 4, delta_m: 0.058000 },
        GainAlloc { b_m: 4, delta_m: 0.052800 },
    ],
    // L = 22
    [
        GainAlloc { b_m: 5, delta_m: 0.086800 },
        GainAlloc { b_m: 5, delta_m: 0.056280 },
        GainAlloc { b_m: 4, delta_m: 0.067200 },
        GainAlloc { b_m: 4, delta_m: 0.058000 },
        GainAlloc { b_m: 4, delta_m: 0.052800 },
    ],
    // L = 23
    [
        GainAlloc { b_m: 5, delta_m: 0.086800 },
        GainAlloc { b_m: 4, delta_m: 0.080400 },
        GainAlloc { b_m: 4, delta_m: 0.067200 },
        GainAlloc { b_m: 4, delta_m: 0.058000 },
        GainAlloc { b_m: 4, delta_m: 0.052800 },
    ],
    // L = 24
    [
        GainAlloc { b_m: 5, delta_m: 0.086800 },
        GainAlloc { b_m: 4, delta_m: 0.080400 },
        GainAlloc { b_m: 4, delta_m: 0.067200 },
        GainAlloc { b_m: 4, delta_m: 0.058000 },
        GainAlloc { b_m: 4, delta_m: 0.052800 },
    ],
    // L = 25
    [
        GainAlloc { b_m: 5, delta_m: 0.086800 },
        GainAlloc { b_m: 4, delta_m: 0.080400 },
        GainAlloc { b_m: 4, delta_m: 0.067200 },
        GainAlloc { b_m: 4, delta_m: 0.058000 },
        GainAlloc { b_m: 3, delta_m: 0.085800 },
    ],
    // L = 26
    [
        GainAlloc { b_m: 5, delta_m: 0.086800 },
        GainAlloc { b_m: 4, delta_m: 0.080400 },
        GainAlloc { b_m: 4, delta_m: 0.067200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 3, delta_m: 0.085800 },
    ],
    // L = 27
    [
        GainAlloc { b_m: 5, delta_m: 0.086800 },
        GainAlloc { b_m: 4, delta_m: 0.080400 },
        GainAlloc { b_m: 4, delta_m: 0.067200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 3, delta_m: 0.085800 },
    ],
    // L = 28
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 4, delta_m: 0.080400 },
        GainAlloc { b_m: 4, delta_m: 0.067200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 3, delta_m: 0.085800 },
    ],
    // L = 29
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 4, delta_m: 0.080400 },
        GainAlloc { b_m: 4, delta_m: 0.067200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 3, delta_m: 0.085800 },
    ],
    // L = 30
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 4, delta_m: 0.080400 },
        GainAlloc { b_m: 4, delta_m: 0.067200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 3, delta_m: 0.085800 },
    ],
    // L = 31
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 4, delta_m: 0.080400 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 3, delta_m: 0.085800 },
    ],
    // L = 32
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 4, delta_m: 0.080400 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 3, delta_m: 0.085800 },
    ],
    // L = 33
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 3, delta_m: 0.085800 },
    ],
    // L = 34
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 3, delta_m: 0.085800 },
    ],
    // L = 35
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 3, delta_m: 0.085800 },
    ],
    // L = 36
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 3, delta_m: 0.085800 },
    ],
    // L = 37
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 38
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 39
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 40
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 3, delta_m: 0.094250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 41
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 42
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 43
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 44
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 45
    [
        GainAlloc { b_m: 4, delta_m: 0.124000 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 46
    [
        GainAlloc { b_m: 3, delta_m: 0.201500 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 47
    [
        GainAlloc { b_m: 3, delta_m: 0.201500 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 48
    [
        GainAlloc { b_m: 3, delta_m: 0.201500 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 49
    [
        GainAlloc { b_m: 3, delta_m: 0.201500 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 50
    [
        GainAlloc { b_m: 3, delta_m: 0.201500 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 51
    [
        GainAlloc { b_m: 3, delta_m: 0.201500 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 3, delta_m: 0.109200 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 52
    [
        GainAlloc { b_m: 3, delta_m: 0.201500 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 2, delta_m: 0.142800 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 53
    [
        GainAlloc { b_m: 3, delta_m: 0.201500 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 2, delta_m: 0.142800 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 54
    [
        GainAlloc { b_m: 3, delta_m: 0.201500 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 2, delta_m: 0.142800 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 55
    [
        GainAlloc { b_m: 3, delta_m: 0.201500 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 2, delta_m: 0.142800 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
    // L = 56
    [
        GainAlloc { b_m: 3, delta_m: 0.201500 },
        GainAlloc { b_m: 3, delta_m: 0.130650 },
        GainAlloc { b_m: 2, delta_m: 0.142800 },
        GainAlloc { b_m: 2, delta_m: 0.123250 },
        GainAlloc { b_m: 2, delta_m: 0.112200 },
    ],
];
