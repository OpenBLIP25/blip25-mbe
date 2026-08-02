// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! Annex-T tone-frame synthesis for the AMBE+2 (r33) decode path.
//!
//! the reference's AMBE+2 half-rate frames carry, alongside voice, a compact
//! **tone-frame** encoding (TIA-102.BABA-A §2.10, Annex T): a signature in the
//! prioritized info vectors selects a fixed tone-parameter table row `(f0, l1,
//! l2)` and a 7-bit log-amplitude `A_D`. The reference decoder synthesizes a
//! clean 1-2 harmonic sinusoid from those; the plain voice path in
//! [`crate::Decoder::decode_pcm_fixed`] cannot (its pitch index is out of the
//! voice range), so without this module those frames drop to silence.
//!
//! The classification + parameter bridge lands on the plain-field
//! [`MbeParams`] and feeds the float harmonic synthesizer
//! ([`crate::synth::synthesize_frame`]). It touches ONLY tone/erasure frames;
//! voice frames are classified `Voice` and fall through unchanged.

use core::f64::consts::LOG2_10;
use core::f64::consts::PI as PI64;

use crate::dequantize::{MbeParams, L_MAX, L_MIN};

/// One Annex-T tone-table row: fundamental `f0` (Hz) and the two harmonic
/// indices whose sinusoids make up the tone (`l1 == l2` for a single tone).
#[derive(Clone, Copy, Debug)]
pub struct ToneParams {
    /// Fundamental frequency in Hz.
    pub f0: f32,
    /// Harmonic index of the first tone component.
    pub l1: u8,
    /// Harmonic index of the second tone component (equal to `l1` for a
    /// single-frequency tone).
    pub l2: u8,
}

/// Bits extracted from a tone frame per §2.10.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToneFrameFields {
    /// 8-bit tone ID (`I_D`).
    pub id: u8,
    /// 7-bit log-amplitude (`A_D ∈ [0, 127]`).
    pub amplitude: u8,
}

/// Classification of a received half-rate frame prior to decode.
///
/// The class is a function of the 6-bit escape field `û₀(11..6)` alone —
/// see [`classify`]. `Silence` is a distinct class from `Erasure`: both are
/// escapes, but only `Erasure` drives the repeat/mute concealment machinery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    /// Normal voice frame — dequantize + voice synth (unchanged path).
    Voice,
    /// Tone frame per §2.10 — parse ID/amplitude and synthesize a sinusoid.
    Tone,
    /// Erasure marker — repeat the previous frame (§2.8.2).
    Erasure,
    /// Silence marker (`û₀(11..6) == 0x3E`). Distinct from `Erasure`: it is
    /// not a channel-error indication and must not advance the repeat/mute
    /// counters.
    Silence,
}

/// Annex-T sinusoidal-amplitude scale factor (§2.10.3 Eq. 209):
/// `M̃_l = 16384 · 10^{0.03555·(A_D − 127)}` at the tone's harmonic indices.
pub(crate) const TONE_AMPLITUDE_PEAK: f64 = 16384.0;
/// `log10` exponent multiplier in Eq. 209 — 0.711 dB/step in linear form.
pub(crate) const TONE_AMPLITUDE_EXPONENT_STEP: f64 = 0.03555;

pub const ANNEX_T: [Option<ToneParams>; 256] = [
    None, // 0
    None, // 1
    None, // 2
    None, // 3
    None, // 4
    Some(ToneParams {
        f0: 156.25,
        l1: 1,
        l2: 1,
    }), // 5
    Some(ToneParams {
        f0: 187.5,
        l1: 1,
        l2: 1,
    }), // 6
    Some(ToneParams {
        f0: 218.75,
        l1: 1,
        l2: 1,
    }), // 7
    Some(ToneParams {
        f0: 250.0000,
        l1: 1,
        l2: 1,
    }), // 8
    Some(ToneParams {
        f0: 281.25,
        l1: 1,
        l2: 1,
    }), // 9
    Some(ToneParams {
        f0: 312.5,
        l1: 1,
        l2: 1,
    }), // 10
    Some(ToneParams {
        f0: 343.75,
        l1: 1,
        l2: 1,
    }), // 11
    Some(ToneParams {
        f0: 375.0000,
        l1: 1,
        l2: 1,
    }), // 12
    Some(ToneParams {
        f0: 203.125,
        l1: 2,
        l2: 2,
    }), // 13
    Some(ToneParams {
        f0: 218.75,
        l1: 2,
        l2: 2,
    }), // 14
    Some(ToneParams {
        f0: 234.375,
        l1: 2,
        l2: 2,
    }), // 15
    Some(ToneParams {
        f0: 250.0000,
        l1: 2,
        l2: 2,
    }), // 16
    Some(ToneParams {
        f0: 265.625,
        l1: 2,
        l2: 2,
    }), // 17
    Some(ToneParams {
        f0: 281.25,
        l1: 2,
        l2: 2,
    }), // 18
    Some(ToneParams {
        f0: 296.875,
        l1: 2,
        l2: 2,
    }), // 19
    Some(ToneParams {
        f0: 312.5,
        l1: 2,
        l2: 2,
    }), // 20
    Some(ToneParams {
        f0: 328.125,
        l1: 2,
        l2: 2,
    }), // 21
    Some(ToneParams {
        f0: 343.75,
        l1: 2,
        l2: 2,
    }), // 22
    Some(ToneParams {
        f0: 359.375,
        l1: 2,
        l2: 2,
    }), // 23
    Some(ToneParams {
        f0: 375.0000,
        l1: 2,
        l2: 2,
    }), // 24
    Some(ToneParams {
        f0: 390.625,
        l1: 2,
        l2: 2,
    }), // 25
    Some(ToneParams {
        f0: 270.842,
        l1: 3,
        l2: 3,
    }), // 26
    Some(ToneParams {
        f0: 281.259,
        l1: 3,
        l2: 3,
    }), // 27
    Some(ToneParams {
        f0: 291.676,
        l1: 3,
        l2: 3,
    }), // 28
    Some(ToneParams {
        f0: 302.093,
        l1: 3,
        l2: 3,
    }), // 29
    Some(ToneParams {
        f0: 312.51,
        l1: 3,
        l2: 3,
    }), // 30
    Some(ToneParams {
        f0: 322.927,
        l1: 3,
        l2: 3,
    }), // 31
    Some(ToneParams {
        f0: 333.344,
        l1: 3,
        l2: 3,
    }), // 32
    Some(ToneParams {
        f0: 343.761,
        l1: 3,
        l2: 3,
    }), // 33
    Some(ToneParams {
        f0: 354.178,
        l1: 3,
        l2: 3,
    }), // 34
    Some(ToneParams {
        f0: 364.595,
        l1: 3,
        l2: 3,
    }), // 35
    Some(ToneParams {
        f0: 375.012,
        l1: 3,
        l2: 3,
    }), // 36
    Some(ToneParams {
        f0: 385.429,
        l1: 3,
        l2: 3,
    }), // 37
    Some(ToneParams {
        f0: 395.846,
        l1: 3,
        l2: 3,
    }), // 38
    Some(ToneParams {
        f0: 304.6875,
        l1: 4,
        l2: 4,
    }), // 39
    Some(ToneParams {
        f0: 312.5,
        l1: 4,
        l2: 4,
    }), // 40
    Some(ToneParams {
        f0: 320.3125,
        l1: 4,
        l2: 4,
    }), // 41
    Some(ToneParams {
        f0: 328.125,
        l1: 4,
        l2: 4,
    }), // 42
    Some(ToneParams {
        f0: 335.9375,
        l1: 4,
        l2: 4,
    }), // 43
    Some(ToneParams {
        f0: 343.75,
        l1: 4,
        l2: 4,
    }), // 44
    Some(ToneParams {
        f0: 351.5625,
        l1: 4,
        l2: 4,
    }), // 45
    Some(ToneParams {
        f0: 359.375,
        l1: 4,
        l2: 4,
    }), // 46
    Some(ToneParams {
        f0: 367.1875,
        l1: 4,
        l2: 4,
    }), // 47
    Some(ToneParams {
        f0: 375.0000,
        l1: 4,
        l2: 4,
    }), // 48
    Some(ToneParams {
        f0: 382.8125,
        l1: 4,
        l2: 4,
    }), // 49
    Some(ToneParams {
        f0: 390.625,
        l1: 4,
        l2: 4,
    }), // 50
    Some(ToneParams {
        f0: 398.4375,
        l1: 4,
        l2: 4,
    }), // 51
    Some(ToneParams {
        f0: 325.0000,
        l1: 5,
        l2: 5,
    }), // 52
    Some(ToneParams {
        f0: 331.25,
        l1: 5,
        l2: 5,
    }), // 53
    Some(ToneParams {
        f0: 337.5,
        l1: 5,
        l2: 5,
    }), // 54
    Some(ToneParams {
        f0: 343.75,
        l1: 5,
        l2: 5,
    }), // 55
    Some(ToneParams {
        f0: 350.0000,
        l1: 5,
        l2: 5,
    }), // 56
    Some(ToneParams {
        f0: 356.25,
        l1: 5,
        l2: 5,
    }), // 57
    Some(ToneParams {
        f0: 362.5,
        l1: 5,
        l2: 5,
    }), // 58
    Some(ToneParams {
        f0: 368.75,
        l1: 5,
        l2: 5,
    }), // 59
    Some(ToneParams {
        f0: 375.0000,
        l1: 5,
        l2: 5,
    }), // 60
    Some(ToneParams {
        f0: 381.25,
        l1: 5,
        l2: 5,
    }), // 61
    Some(ToneParams {
        f0: 387.5,
        l1: 5,
        l2: 5,
    }), // 62
    Some(ToneParams {
        f0: 393.75,
        l1: 5,
        l2: 5,
    }), // 63
    Some(ToneParams {
        f0: 400.0000,
        l1: 5,
        l2: 5,
    }), // 64
    Some(ToneParams {
        f0: 343.2195,
        l1: 6,
        l2: 6,
    }), // 65
    Some(ToneParams {
        f0: 348.4998,
        l1: 6,
        l2: 6,
    }), // 66
    Some(ToneParams {
        f0: 353.7801,
        l1: 6,
        l2: 6,
    }), // 67
    Some(ToneParams {
        f0: 359.0604,
        l1: 6,
        l2: 6,
    }), // 68
    Some(ToneParams {
        f0: 364.3407,
        l1: 6,
        l2: 6,
    }), // 69
    Some(ToneParams {
        f0: 369.621,
        l1: 6,
        l2: 6,
    }), // 70
    Some(ToneParams {
        f0: 374.9013,
        l1: 6,
        l2: 6,
    }), // 71
    Some(ToneParams {
        f0: 380.1816,
        l1: 6,
        l2: 6,
    }), // 72
    Some(ToneParams {
        f0: 385.4619,
        l1: 6,
        l2: 6,
    }), // 73
    Some(ToneParams {
        f0: 390.7422,
        l1: 6,
        l2: 6,
    }), // 74
    Some(ToneParams {
        f0: 396.0225,
        l1: 6,
        l2: 6,
    }), // 75
    Some(ToneParams {
        f0: 401.3028,
        l1: 6,
        l2: 6,
    }), // 76
    Some(ToneParams {
        f0: 343.7511,
        l1: 7,
        l2: 7,
    }), // 77
    Some(ToneParams {
        f0: 348.2154,
        l1: 7,
        l2: 7,
    }), // 78
    Some(ToneParams {
        f0: 352.6797,
        l1: 7,
        l2: 7,
    }), // 79
    Some(ToneParams {
        f0: 357.144,
        l1: 7,
        l2: 7,
    }), // 80
    Some(ToneParams {
        f0: 361.6083,
        l1: 7,
        l2: 7,
    }), // 81
    Some(ToneParams {
        f0: 366.0726,
        l1: 7,
        l2: 7,
    }), // 82
    Some(ToneParams {
        f0: 370.5369,
        l1: 7,
        l2: 7,
    }), // 83
    Some(ToneParams {
        f0: 375.0012,
        l1: 7,
        l2: 7,
    }), // 84
    Some(ToneParams {
        f0: 379.4655,
        l1: 7,
        l2: 7,
    }), // 85
    Some(ToneParams {
        f0: 383.9298,
        l1: 7,
        l2: 7,
    }), // 86
    Some(ToneParams {
        f0: 388.3941,
        l1: 7,
        l2: 7,
    }), // 87
    Some(ToneParams {
        f0: 392.8584,
        l1: 7,
        l2: 7,
    }), // 88
    Some(ToneParams {
        f0: 397.3227,
        l1: 7,
        l2: 7,
    }), // 89
    Some(ToneParams {
        f0: 351.567,
        l1: 8,
        l2: 8,
    }), // 90
    Some(ToneParams {
        f0: 355.4733,
        l1: 8,
        l2: 8,
    }), // 91
    Some(ToneParams {
        f0: 359.3796,
        l1: 8,
        l2: 8,
    }), // 92
    Some(ToneParams {
        f0: 363.2859,
        l1: 8,
        l2: 8,
    }), // 93
    Some(ToneParams {
        f0: 367.1922,
        l1: 8,
        l2: 8,
    }), // 94
    Some(ToneParams {
        f0: 371.0985,
        l1: 8,
        l2: 8,
    }), // 95
    Some(ToneParams {
        f0: 375.0048,
        l1: 8,
        l2: 8,
    }), // 96
    Some(ToneParams {
        f0: 378.9111,
        l1: 8,
        l2: 8,
    }), // 97
    Some(ToneParams {
        f0: 382.8174,
        l1: 8,
        l2: 8,
    }), // 98
    Some(ToneParams {
        f0: 386.7237,
        l1: 8,
        l2: 8,
    }), // 99
    Some(ToneParams {
        f0: 390.63,
        l1: 8,
        l2: 8,
    }), // 100
    Some(ToneParams {
        f0: 394.5363,
        l1: 8,
        l2: 8,
    }), // 101
    Some(ToneParams {
        f0: 398.4426,
        l1: 8,
        l2: 8,
    }), // 102
    Some(ToneParams {
        f0: 357.6366,
        l1: 9,
        l2: 9,
    }), // 103
    Some(ToneParams {
        f0: 361.1088,
        l1: 9,
        l2: 9,
    }), // 104
    Some(ToneParams {
        f0: 364.581,
        l1: 9,
        l2: 9,
    }), // 105
    Some(ToneParams {
        f0: 368.0532,
        l1: 9,
        l2: 9,
    }), // 106
    Some(ToneParams {
        f0: 371.5254,
        l1: 9,
        l2: 9,
    }), // 107
    Some(ToneParams {
        f0: 374.9976,
        l1: 9,
        l2: 9,
    }), // 108
    Some(ToneParams {
        f0: 378.4698,
        l1: 9,
        l2: 9,
    }), // 109
    Some(ToneParams {
        f0: 381.942,
        l1: 9,
        l2: 9,
    }), // 110
    Some(ToneParams {
        f0: 385.4142,
        l1: 9,
        l2: 9,
    }), // 111
    Some(ToneParams {
        f0: 388.8864,
        l1: 9,
        l2: 9,
    }), // 112
    Some(ToneParams {
        f0: 392.3586,
        l1: 9,
        l2: 9,
    }), // 113
    Some(ToneParams {
        f0: 395.8308,
        l1: 9,
        l2: 9,
    }), // 114
    Some(ToneParams {
        f0: 399.303,
        l1: 9,
        l2: 9,
    }), // 115
    Some(ToneParams {
        f0: 362.5,
        l1: 10,
        l2: 10,
    }), // 116
    Some(ToneParams {
        f0: 365.625,
        l1: 10,
        l2: 10,
    }), // 117
    Some(ToneParams {
        f0: 368.75,
        l1: 10,
        l2: 10,
    }), // 118
    Some(ToneParams {
        f0: 371.875,
        l1: 10,
        l2: 10,
    }), // 119
    Some(ToneParams {
        f0: 375.0000,
        l1: 10,
        l2: 10,
    }), // 120
    Some(ToneParams {
        f0: 378.125,
        l1: 10,
        l2: 10,
    }), // 121
    Some(ToneParams {
        f0: 381.25,
        l1: 10,
        l2: 10,
    }), // 122
    None, // 123
    None, // 124
    None, // 125
    None, // 126
    None, // 127
    Some(ToneParams {
        f0: 78.5000,
        l1: 12,
        l2: 17,
    }), // 128
    Some(ToneParams {
        f0: 173.48,
        l1: 4,
        l2: 7,
    }), // 129
    Some(ToneParams {
        f0: 70.0000,
        l1: 10,
        l2: 19,
    }), // 130
    Some(ToneParams {
        f0: 87.0000,
        l1: 8,
        l2: 17,
    }), // 131
    Some(ToneParams {
        f0: 109.95,
        l1: 7,
        l2: 11,
    }), // 132
    Some(ToneParams {
        f0: 191.68,
        l1: 4,
        l2: 7,
    }), // 133
    Some(ToneParams {
        f0: 70.1700,
        l1: 11,
        l2: 21,
    }), // 134
    Some(ToneParams {
        f0: 71.0600,
        l1: 12,
        l2: 17,
    }), // 135
    Some(ToneParams {
        f0: 121.58,
        l1: 7,
        l2: 11,
    }), // 136
    Some(ToneParams {
        f0: 212.0000,
        l1: 4,
        l2: 7,
    }), // 137
    Some(ToneParams {
        f0: 116.41,
        l1: 6,
        l2: 14,
    }), // 138
    Some(ToneParams {
        f0: 96.1500,
        l1: 8,
        l2: 17,
    }), // 139
    Some(ToneParams {
        f0: 71.0000,
        l1: 12,
        l2: 23,
    }), // 140
    Some(ToneParams {
        f0: 234.26,
        l1: 4,
        l2: 7,
    }), // 141
    Some(ToneParams {
        f0: 134.38,
        l1: 7,
        l2: 9,
    }), // 142
    Some(ToneParams {
        f0: 134.35,
        l1: 7,
        l2: 11,
    }), // 143
    Some(ToneParams {
        f0: 68.3300,
        l1: 12,
        l2: 17,
    }), // 144
    Some(ToneParams {
        f0: 150.803,
        l1: 4,
        l2: 7,
    }), // 145
    Some(ToneParams {
        f0: 67.8200,
        l1: 9,
        l2: 17,
    }), // 146
    Some(ToneParams {
        f0: 86.5000,
        l1: 7,
        l2: 15,
    }), // 147
    Some(ToneParams {
        f0: 95.7900,
        l1: 7,
        l2: 11,
    }), // 148
    Some(ToneParams {
        f0: 166.92,
        l1: 4,
        l2: 7,
    }), // 149
    Some(ToneParams {
        f0: 67.7000,
        l1: 10,
        l2: 19,
    }), // 150
    Some(ToneParams {
        f0: 74.7400,
        l1: 10,
        l2: 14,
    }), // 151
    Some(ToneParams {
        f0: 105.9,
        l1: 7,
        l2: 11,
    }), // 152
    Some(ToneParams {
        f0: 92.7800,
        l1: 8,
        l2: 14,
    }), // 153
    Some(ToneParams {
        f0: 101.55,
        l1: 6,
        l2: 14,
    }), // 154
    Some(ToneParams {
        f0: 84.0200,
        l1: 8,
        l2: 17,
    }), // 155
    Some(ToneParams {
        f0: 67.8300,
        l1: 11,
        l2: 21,
    }), // 156
    Some(ToneParams {
        f0: 102.3,
        l1: 8,
        l2: 14,
    }), // 157
    Some(ToneParams {
        f0: 117.0000,
        l1: 7,
        l2: 9,
    }), // 158
    Some(ToneParams {
        f0: 117.49,
        l1: 7,
        l2: 11,
    }), // 159
    Some(ToneParams {
        f0: 87.7800,
        l1: 4,
        l2: 5,
    }), // 160
    Some(ToneParams {
        f0: 70.8300,
        l1: 6,
        l2: 7,
    }), // 161
    Some(ToneParams {
        f0: 122.0000,
        l1: 4,
        l2: 5,
    }), // 162
    Some(ToneParams {
        f0: 70.0000,
        l1: 5,
        l2: 7,
    }), // 163
    None, // 164
    None, // 165
    None, // 166
    None, // 167
    None, // 168
    None, // 169
    None, // 170
    None, // 171
    None, // 172
    None, // 173
    None, // 174
    None, // 175
    None, // 176
    None, // 177
    None, // 178
    None, // 179
    None, // 180
    None, // 181
    None, // 182
    None, // 183
    None, // 184
    None, // 185
    None, // 186
    None, // 187
    None, // 188
    None, // 189
    None, // 190
    None, // 191
    None, // 192
    None, // 193
    None, // 194
    None, // 195
    None, // 196
    None, // 197
    None, // 198
    None, // 199
    None, // 200
    None, // 201
    None, // 202
    None, // 203
    None, // 204
    None, // 205
    None, // 206
    None, // 207
    None, // 208
    None, // 209
    None, // 210
    None, // 211
    None, // 212
    None, // 213
    None, // 214
    None, // 215
    None, // 216
    None, // 217
    None, // 218
    None, // 219
    None, // 220
    None, // 221
    None, // 222
    None, // 223
    None, // 224
    None, // 225
    None, // 226
    None, // 227
    None, // 228
    None, // 229
    None, // 230
    None, // 231
    None, // 232
    None, // 233
    None, // 234
    None, // 235
    None, // 236
    None, // 237
    None, // 238
    None, // 239
    None, // 240
    None, // 241
    None, // 242
    None, // 243
    None, // 244
    None, // 245
    None, // 246
    None, // 247
    None, // 248
    None, // 249
    None, // 250
    None, // 251
    None, // 252
    None, // 253
    None, // 254
    Some(ToneParams {
        f0: 250.0000,
        l1: 0,
        l2: 0,
    }), // 255
];

// ===========================================================================
// Encode-side tone DETECTION
//
// Given a 160-sample PCM frame, decide whether the reference encoder would emit
// an Annex-T tone frame, and if so which `(I_D, A_D)`. Reproducing the reference's
// `(id, amplitude)` is sufficient for byte-exact output: `encode_tone_frame_info
// + encode_frame` reconstructs the reference r33 bytes exactly.
//
// Classification is a direct spectral match: audible frequency = `l·f0`, so the
// single-tone rows (ids 5..122) form a monotonic frequency grid and the dual
// rows (128..163) carry a `{l1·f0, l2·f0}` pair. We estimate the frame's 1–2
// dominant sinusoids (rectangular Goertzel + local refine), gate on tonality,
// nearest-match to the grid, and set `A_D` from the geometric mean of the two
// harmonic peaks (Eq. 209 inverse). All of this is generic DSP.
// ===========================================================================

/// Lowest analysis frequency (Hz) — below the lowest single-tone grid point.
const DETECT_F_LO: f64 = 60.0;
/// Highest analysis frequency (Hz) — above the highest dual harmonic.
const DETECT_F_HI: f64 = 3900.0;
/// Local-refine step (Hz) for peak-frequency estimation.
const DETECT_REFINE_HZ: f64 = 0.25;
/// Minimum separation (Hz) for a second peak to count as distinct.
const DETECT_PEAK_SEP_HZ: f64 = 60.0;
/// Second-peak / first-peak amplitude ratio at/above which the frame is a
/// DUAL tone (below → single). DTMF low/high groups can differ by ~8 dB.
const DETECT_DUAL_RATIO: f64 = 0.22;
/// A second peak within this fractional distance of an integer multiple of the
/// first is treated as harmonic distortion of a single tone, not a dual tone.
const DETECT_HARMONIC_TOL: f64 = 0.04;
/// Tonality gate for a SINGLE tone: fraction of frame variance the one harmonic
/// must explain. High, because a lone dominant harmonic is common in voiced
/// speech and must not be mistaken for a tone.
const DETECT_TONALITY_SINGLE: f64 = 0.97;
/// Tonality gate for a DUAL tone: relaxed, because two on-grid harmonics
/// forming an Annex-T pair almost never occur in speech, and close call-progress
/// pairs sit lower after matching-pursuit extraction.
const DETECT_TONALITY_DUAL: f64 = 0.88;
/// Two harmonics closer than this (Hz) are a "close pair" (call-progress tones)
/// that beats within the analysis frame. Kept tight so ordinary low-pitched
/// voiced harmonics don't qualify for the relaxed close-pair floor.
const DETECT_CLOSE_PAIR_HZ: f64 = 80.0;
/// Relaxed tonality floor for a close dual pair (its beat nulls dip low).
const DETECT_TONALITY_CLOSE: f64 = 0.80;
/// Max sub-harmonic energy (at f/2, f/3, 2f/3 of a single tone), as a fraction
/// of the tone's peak, before the frame is treated as a voiced harmonic (not a
/// tone). Real single tones sit near 0; voiced vowels run 0.2–0.5.
const DETECT_SUBHARM_MAX: f64 = 0.15;
/// Max genuine local-peak overtones (at 2f/3f/4f above the detected tone) before
/// the frame is treated as a voiced harmonic comb, not a tone. A pure Annex-T
/// tone has none (at most weak 2nd-harmonic distortion); voiced speech carries a
/// full comb. Measured: speech false-fires carry ≥2 (clean 26/29, dam 21/21);
/// real tones carry 0–1 (cp0 0/50, dtmf 965/1124 ≤1). Applies to single AND dual
/// (voiced frames masquerade as both). Rejecting ≥`DETECT_OVERTONE_PEAKS`.
const DETECT_OVERTONE_PEAKS: usize = 2;
/// Minimum overtone magnitude, as a fraction of the tone's own peak, for a
/// harmonic slot to count toward the comb test. Voiced-speech overtones are a
/// large fraction; a pure tone's window-leakage skirt at 2f/3f/4f is tiny, so
/// this stops leakage from being miscounted as a comb.
const DETECT_OVERTONE_MIN_RATIO: f64 = 0.008;
/// Half-width (Hz) of the local-peak test around a candidate sub-harmonic: it
/// counts only if it exceeds the magnitude this far to either side (a leakage
/// skirt slopes monotonically toward the main peak and fails this).
const DETECT_SUBHARM_NOTCH: f64 = 40.0;
/// Max nearest-grid frequency residual (Hz) to accept a single-tone match.
const DETECT_FREQ_TOL: f64 = 40.0;
/// Max per-tone residual (Hz) to accept a dual-tone match — more generous than
/// the single tolerance because the dual grid is sparse and the reference quantizes
/// off-grid dual tones (e.g. call-progress) to the nearest sparse row.
const DETECT_DUAL_FREQ_TOL: f64 = 55.0;
/// A_D model slope (= inverse of the Eq. 209 exponent 0.03555·20).
const DETECT_AD_SLOPE: f64 = 1.406;
/// A_D model offset. Calibrated for exact-`A_D` match against the reference tone
/// frames using the matching-pursuit LS harmonic amplitudes (94% exact, ≤±1 on
/// the rest across the tone vectors).
const DETECT_AD_OFFSET: f64 = -0.12;

/// Goertzel complex magnitude of a mean-removed signal at frequency `f` (Hz).
fn goertzel_mag(x: &[f64], f: f64) -> f64 {
    let w = 2.0 * PI64 * f / 8000.0;
    let (c, s) = (w.cos(), w.sin());
    let coeff = 2.0 * c;
    let (mut q1, mut q2) = (0.0, 0.0);
    for &v in x {
        let q0 = coeff * q1 - q2 + v;
        q2 = q1;
        q1 = q0;
    }
    let re = q1 - q2 * c;
    let im = q2 * s;
    (re * re + im * im).sqrt()
}

/// A detected spectral peak: refined frequency (Hz) and sinusoid peak amplitude.
#[derive(Clone, Copy, Debug)]
struct Peak {
    f: f64,
    amp: f64,
}

/// Least-squares real-sinusoid coefficients `(a, b)` of `x` at frequency `f`
/// (Hz): `x[n] ≈ a·cos(ωn) + b·sin(ωn)`, `ω = 2πf/8000`. Peak amplitude is
/// `hypot(a, b)`.
fn ls_sinusoid(x: &[f64], f: f64) -> (f64, f64) {
    let w = 2.0 * PI64 * f / 8000.0;
    let (mut sc, mut ss, mut scc, mut sss, mut scs) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (n, &xn) in x.iter().enumerate() {
        let (s, c) = (w * n as f64).sin_cos();
        sc += xn * c;
        ss += xn * s;
        scc += c * c;
        sss += s * s;
        scs += c * s;
    }
    // Solve the 2×2 normal equations [scc scs; scs sss][a;b] = [sc; ss].
    let det = scc * sss - scs * scs;
    if det.abs() < 1e-9 {
        return (0.0, 0.0);
    }
    let a = (sc * sss - ss * scs) / det;
    let b = (ss * scc - sc * scs) / det;
    (a, b)
}

/// Zero-padded FFT length for the coarse spectral search (bin ≈ 7.8 Hz).
const FFT_N: usize = 1024;

/// In-place iterative radix-2 Cooley–Tukey FFT (`re`/`im` length must be a power
/// of two). Generic DSP.
fn fft_inplace(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI64 / len as f64;
        let (wli, wlr) = ang.sin_cos();
        let half = len / 2;
        let mut i = 0;
        while i < n {
            let (mut wr, mut wi) = (1.0f64, 0.0f64);
            for k in 0..half {
                let a = i + k;
                let b = i + k + half;
                let tr = wr * re[b] - wi * im[b];
                let ti = wr * im[b] + wi * re[b];
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let nwr = wr * wlr - wi * wli;
                wi = wr * wli + wi * wlr;
                wr = nwr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Coarse magnitude spectrum of `x` (zero-padded to [`FFT_N`]), returning the
/// magnitude at each bin `k` (frequency `k·8000/FFT_N`) up to Nyquist.
fn coarse_spectrum(x: &[f64]) -> [f64; FFT_N / 2 + 1] {
    let mut re = [0.0f64; FFT_N];
    let mut im = [0.0f64; FFT_N];
    for (i, &v) in x.iter().take(FFT_N).enumerate() {
        re[i] = v;
    }
    fft_inplace(&mut re, &mut im);
    let mut mag = [0.0f64; FFT_N / 2 + 1];
    for (k, m) in mag.iter_mut().enumerate() {
        *m = (re[k] * re[k] + im[k] * im[k]).sqrt();
    }
    mag
}

/// Locate the dominant spectral peak of `x` via a zero-padded FFT coarse search
/// then a local Goertzel refine, avoiding a `±exclude_hz` band around `avoid`
/// (Hz; negative to disable).
fn dominant_peak(x: &[f64], avoid: f64, exclude_hz: f64) -> Option<f64> {
    let mag = coarse_spectrum(x);
    let bin_hz = 8000.0 / FFT_N as f64;
    let k_lo = (DETECT_F_LO / bin_hz).floor() as usize;
    let k_hi = ((DETECT_F_HI / bin_hz).ceil() as usize).min(FFT_N / 2);
    let (mut bk, mut bm) = (0usize, 0.0);
    for k in k_lo..=k_hi {
        let f = k as f64 * bin_hz;
        if avoid >= 0.0 && (f - avoid).abs() < exclude_hz {
            continue;
        }
        if mag[k] > bm {
            bm = mag[k];
            bk = k;
        }
    }
    if bm <= 0.0 {
        return None;
    }
    // Refine within ±1.5 bins of the coarse peak using local Goertzel.
    let f_coarse = bk as f64 * bin_hz;
    Some(refine_freq_wide(x, f_coarse, 2.0 * bin_hz))
}

/// Refine `f0` (Hz) to [`DETECT_REFINE_HZ`] over ±`half_hz` by maximizing the
/// Goertzel magnitude.
fn refine_freq_wide(x: &[f64], f0: f64, half_hz: f64) -> f64 {
    let mut bf = f0;
    let mut bm = goertzel_mag(x, f0);
    let mut ff = (f0 - half_hz).max(DETECT_F_LO);
    let top = f0 + half_hz;
    while ff <= top {
        let m = goertzel_mag(x, ff);
        if m > bm {
            bm = m;
            bf = ff;
        }
        ff += DETECT_REFINE_HZ;
    }
    bf
}

/// Extract up to two sinusoids from `frame` by matching pursuit: find the
/// dominant tone, subtract its least-squares fit, then find the second tone in
/// the residual. Resolves close tone pairs (call-progress ~88 Hz apart) that a
/// single windowed spectrum merges, and finds off-grid tones (the reference still
/// quantizes those to the nearest Annex-T row). Returns `(rms, peaks)` sorted
/// by amplitude.
fn spectral_peaks(frame: &[i16]) -> (f64, Vec<Peak>) {
    let n = frame.len();
    if n == 0 {
        return (0.0, Vec::new());
    }
    let mean = frame.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
    let mut x: Vec<f64> = frame.iter().map(|&s| s as f64 - mean).collect();
    let rms = (x.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
    if rms < 1.0 {
        return (rms, Vec::new());
    }

    let mut peaks: Vec<Peak> = Vec::new();
    // First tone.
    let Some(f1) = dominant_peak(&x, -1.0, 0.0) else {
        return (rms, peaks);
    };
    let (a1, b1) = ls_sinusoid(&x, f1);
    let amp1 = a1.hypot(b1);
    peaks.push(Peak { f: f1, amp: amp1 });
    // Subtract the first tone.
    let w1 = 2.0 * PI64 * f1 / 8000.0;
    for (nn, xn) in x.iter_mut().enumerate() {
        let (s, c) = (w1 * nn as f64).sin_cos();
        *xn -= a1 * c + b1 * s;
    }
    // Second tone from the residual, away from the first.
    if let Some(f2) = dominant_peak(&x, f1, DETECT_PEAK_SEP_HZ) {
        let (a2, b2) = ls_sinusoid(&x, f2);
        let amp2 = a2.hypot(b2);
        if amp2 > amp1 * 0.02 {
            peaks.push(Peak { f: f2, amp: amp2 });
        }
    }
    peaks.sort_by(|a, b| {
        b.amp
            .partial_cmp(&a.amp)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    (rms, peaks)
}

/// Nearest single-tone id (`l1==l2`, ids 5..122) to frequency `f`, with residual.
fn nearest_single(f: f64) -> Option<(u8, f64)> {
    let mut best: Option<(u8, f64)> = None;
    for id in 0..256usize {
        if id == 255 {
            continue;
        }
        if let Some(ToneParams { f0, l1, l2 }) = ANNEX_T[id] {
            if l1 != l2 || l1 == 0 {
                continue;
            }
            let fl = l1 as f64 * f0 as f64;
            let d = (fl - f).abs();
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((id as u8, d));
            }
        }
    }
    best
}

/// Nearest dual-tone id (`l1!=l2`, ids 128..163) to a `(flo, fhi)` pair, with
/// the summed residual.
fn nearest_dual(flo: f64, fhi: f64) -> Option<(u8, f64)> {
    let mut best: Option<(u8, f64)> = None;
    for id in 0..256usize {
        if let Some(ToneParams { f0, l1, l2 }) = ANNEX_T[id] {
            if l1 == l2 {
                continue;
            }
            let a = l1 as f64 * f0 as f64;
            let b = l2 as f64 * f0 as f64;
            let (glo, ghi) = if a < b { (a, b) } else { (b, a) };
            let d = (glo - flo).abs() + (ghi - fhi).abs();
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((id as u8, d));
            }
        }
    }
    best
}

/// Quantize the geometric mean of two harmonic peak amplitudes to `A_D`
/// (Eq. 209 inverse). Clamped to the 7-bit field.
fn amplitude_to_ad(p1: f64, p2: f64) -> u8 {
    let slope = DETECT_AD_SLOPE;
    let offset = DETECT_AD_OFFSET;
    let g = (p1.max(1.0) * p2.max(1.0)).sqrt();
    let ad = slope * 20.0 * g.log10() + offset;
    ad.round().clamp(0.0, 127.0) as u8
}

/// One frame's tone-detection decision with the internal metrics that produced
/// it — used by calibration/scoring tooling.
#[derive(Clone, Copy, Debug)]
pub struct ToneDecision {
    /// The emitted tone id/amplitude.
    pub fields: ToneFrameFields,
    /// Tonality ratio `(a1²+a2²)/(2·rms²)`.
    pub tonality: f64,
    /// Whether the frame was classified as a dual tone.
    pub dual: bool,
    /// Nearest-grid frequency residual (Hz).
    pub residual: f64,
    /// The two dominant peak frequencies (Hz); `f2==f1` if single.
    pub f1: f64,
    pub f2: f64,
    /// Geometric mean of the two harmonic amplitudes (pre-quantization), for
    /// temporal amplitude smoothing in the stateful detector.
    pub geom_amp: f64,
}

/// Detect whether the reference encoder would emit an Annex-T tone frame for this
/// 160-sample PCM `frame`, returning the `(I_D, A_D)` it would carry.
///
/// Returns `None` for voice / silence / non-tonal frames (caller runs the voice
/// analysis path). Pure DSP: estimates the 1–2 dominant sinusoids, gates on
/// tonality, nearest-matches the Annex-T grid, and sets `A_D` from the harmonic
/// peaks. Reproducing `(id, amplitude)` yields byte-exact tone frames.
pub fn detect_tone_frame(frame: &[i16]) -> Option<ToneFrameFields> {
    detect_tone_frame_dbg(frame, false).map(|d| d.fields)
}

/// Detection with optional stderr trace. Public for scoring tools; prefer
/// [`detect_tone_frame`] in production.
pub fn detect_tone_frame_dbg(frame: &[i16], dbg: bool) -> Option<ToneDecision> {
    // Asymmetric tonality gates. Voiced speech readily produces one dominant
    // harmonic (a lone tone) but almost never two on-grid harmonics forming an
    // Annex-T pair, so single tones are gated far harder than duals. Measured
    // over the reference vectors: at tonality ≥0.97 zero voice frame passes the DUAL
    // gate, and only ~1% of voice frames pass SINGLE — while ≥97% of real tones
    // clear either. A close dual pair (call-progress) sits lower, hence the
    // relaxed dual floor.
    let tonality_single = DETECT_TONALITY_SINGLE;
    let tonality_dual = DETECT_TONALITY_DUAL;
    let dual_ratio = DETECT_DUAL_RATIO;
    let freq_tol = DETECT_FREQ_TOL;

    let (rms, peaks) = spectral_peaks(frame);
    if rms < 1.0 || peaks.is_empty() {
        if dbg {
            eprintln!("  reject: rms={rms:.1} peaks={}", peaks.len());
        }
        return None;
    }
    // Extracted tones (matching pursuit): p1 strongest, p2 (if any) the second.
    let p1 = peaks[0];
    let p2 = peaks.get(1).copied();

    // Tonality: fraction of frame variance the 1–2 extracted tones explain.
    let a2 = p2.map_or(0.0, |p| p.amp);
    let tonality = ((p1.amp * p1.amp + a2 * a2) * 0.5) / (rms * rms);
    // Permissive early gate (reject broadband voice/noise cheaply); the exact
    // single/dual/close-pair gate is applied once the classification is known.
    let tonality_close = DETECT_TONALITY_CLOSE;
    if tonality < tonality_dual.min(tonality_single).min(tonality_close) {
        if dbg {
            eprintln!(
                "  reject tonality={tonality:.3} f1={:.0}({:.0}) f2={:.0}({:.0}) rms={rms:.0}",
                p1.f,
                p1.amp,
                p2.map_or(0.0, |p| p.f),
                a2
            );
        }
        return None;
    }

    // Second tone counts as a distinct tone (dual) if strong enough and not a
    // near-integer harmonic of the first (single-tone distortion).
    let dual_like = match p2 {
        Some(p) => {
            let ratio = p.amp / p1.amp;
            let mult = p.f / p1.f;
            let near_harmonic =
                (mult - mult.round()).abs() < DETECT_HARMONIC_TOL && mult.round() >= 2.0;
            ratio >= dual_ratio && !near_harmonic
        }
        None => false,
    };

    // Single- and dual-grid hypotheses. Single/dual is decided by the SIGNAL
    // (are there two genuine tones?), not by which grid row fits better — the reference
    // quantizes an off-grid dual (e.g. 338/554 Hz) to the nearest dual row
    // (id163 = 350/490) even though a single row would fit one tone tighter.
    let single = nearest_single(p1.f).map(|(id, resid)| (id, resid, p1.f, p1.f, p1.amp, p1.amp));
    let dual = if dual_like {
        let p = p2.unwrap();
        let (lo, hi) = if p1.f < p.f { (p1, p) } else { (p, p1) };
        nearest_dual(lo.f, hi.f).map(|(id, resid)| (id, resid / 2.0, lo.f, hi.f, lo.amp, hi.amp))
    } else {
        None
    };

    let dual_tol = DETECT_DUAL_FREQ_TOL;
    let chosen = match (single, dual) {
        // Two genuine tones → dual, on a generous per-tone tolerance.
        (_, Some(d)) if d.1 <= dual_tol => d,
        (Some(s), _) if s.1 <= freq_tol => s,
        (None, Some(d)) if d.1 <= dual_tol => d,
        _ => return None,
    };
    let (id, per_tone_resid, f1v, f2v, amp1, amp2) = chosen;
    if per_tone_resid > freq_tol {
        if dbg {
            eprintln!("  reject resid={per_tone_resid:.1} f1={f1v:.0} f2={f2v:.0}");
        }
        return None;
    }
    let is_dual = (f1v - f2v).abs() > 1.0;
    // Final tonality gate at the classification-specific threshold. A *close*
    // dual pair (e.g. call-progress tones ~70 Hz apart) beats within the frame,
    // dipping tonality at the beat nulls, so it gets a lower floor — still
    // safe because such a pair must also match a specific sparse dual grid row.
    let close_pair = is_dual && (f1v - f2v).abs() < DETECT_CLOSE_PAIR_HZ;
    let gate = if !is_dual {
        tonality_single
    } else if close_pair {
        tonality_close
    } else {
        tonality_dual
    };
    if tonality < gate {
        if dbg {
            eprintln!("  reject final tonality={tonality:.3} dual={is_dual} id{id}");
        }
        return None;
    }

    // Sub-harmonic guard — the key voice/tone discriminator. A sustained voiced
    // vowel produces a strong harmonic that looks like a pure tone (high
    // tonality, lands on the grid), but it is the l-th harmonic of a lower
    // fundamental, so a genuine separate PEAK sits at f/2, f/3 or 2f/3 below it.
    // A real Annex-T tone has nothing there — only its own window skirt, which
    // is a monotonic slope, NOT a local peak. So we don't just measure energy at
    // the sub-harmonic bin (a low tone's leakage would false-trigger); we require
    // it to be a genuine LOCAL MAXIMUM (energy at fs greater than at fs±40 Hz),
    // which the leakage skirt never is. Measured on a Hann-windowed frame.
    // Single tones only: a dual pair's two tones beat/intermodulate into
    // sub-harmonic bins that aren't a voicing cue.
    let subharm_max = DETECT_SUBHARM_MAX;
    let flo = f1v.min(f2v);
    if !is_dual {
        let n = frame.len();
        let mean = frame.iter().map(|&s| s as f64).sum::<f64>() / n.max(1) as f64;
        let w: Vec<f64> = (0..n)
            .map(|i| {
                let h = 0.5 - 0.5 * (2.0 * PI64 * i as f64 / (n.max(2) as f64 - 1.0)).cos();
                (frame[i] as f64 - mean) * h
            })
            .collect();
        let main = goertzel_mag(&w, flo).max(1e-9);
        // Strongest sub-harmonic that is a genuine local peak (not leakage).
        let sub = [flo / 2.0, flo / 3.0, 2.0 * flo / 3.0]
            .into_iter()
            .filter(|&f| f - DETECT_SUBHARM_NOTCH >= DETECT_F_LO)
            .filter(|&f| {
                let m = goertzel_mag(&w, f);
                m > goertzel_mag(&w, f + DETECT_SUBHARM_NOTCH)
                    && m > goertzel_mag(&w, f - DETECT_SUBHARM_NOTCH)
            })
            .map(|f| goertzel_mag(&w, f))
            .fold(0.0f64, f64::max);
        if sub / main > subharm_max {
            if dbg {
                eprintln!("  reject sub-harmonic {:.2} (voiced) id{id}", sub / main);
            }
            return None;
        }
    }

    // Overtone (harmonic-comb) guard — the other half of the voice/tone
    // discriminator, and the one that catches DUAL-classified vowels the
    // sub-harmonic guard skips. A voiced fundamental carries genuine local peaks
    // at 2f/3f/4f above it; a pure Annex-T tone has none. Reject when ≥
    // DETECT_OVERTONE_PEAKS are present. INTERIM: console tone-detection is being
    // reverse-engineered separately; this is the ground-truth-validated stopgap
    // (console emits 0 tone-frames on clean/dam/noisy speech).
    let overtone_max = DETECT_OVERTONE_PEAKS as f64 as usize;
    {
        let n = frame.len();
        let mean = frame.iter().map(|&s| s as f64).sum::<f64>() / n.max(1) as f64;
        let w: Vec<f64> = (0..n)
            .map(|i| {
                let h = 0.5 - 0.5 * (2.0 * PI64 * i as f64 / (n.max(2) as f64 - 1.0)).cos();
                (frame[i] as f64 - mean) * h
            })
            .collect();
        // The genuine second tone of a DUAL is not a voiced overtone — exclude
        // any harmonic slot that coincides with it, so a legit 2-tone (e.g. a
        // dual-harmonic Annex-T tone) is not mistaken for a comb.
        let fhi = f1v.max(f2v);
        let main = goertzel_mag(&w, flo).max(1e-9);
        let ot_ratio = DETECT_OVERTONE_MIN_RATIO;
        let mut peaks = 0usize;
        for k in [2.0, 3.0, 4.0] {
            let fk = flo * k;
            if fk > DETECT_F_HI {
                continue;
            }
            if (fk - fhi).abs() < DETECT_SUBHARM_NOTCH {
                continue;
            }
            let m = goertzel_mag(&w, fk);
            if m > main * ot_ratio
                && m > goertzel_mag(&w, fk + DETECT_SUBHARM_NOTCH)
                && m > goertzel_mag(&w, fk - DETECT_SUBHARM_NOTCH)
            {
                peaks += 1;
            }
        }
        if peaks >= overtone_max {
            if dbg {
                eprintln!("  reject overtone-comb {peaks} (voiced) id{id}");
            }
            return None;
        }
    }

    // A_D from the geometric mean of the two extracted harmonic amplitudes.
    let geom_amp = (amp1.max(1.0) * amp2.max(1.0)).sqrt();
    let amplitude = amplitude_to_ad(amp1, amp2);

    Some(ToneDecision {
        fields: ToneFrameFields { id, amplitude },
        tonality,
        dual: is_dual,
        residual: per_tone_resid,
        f1: f1v,
        f2: f2v,
        geom_amp,
    })
}

/// Quantize a pre-computed geometric-mean harmonic amplitude to `A_D`.
fn geom_amp_to_ad(g: f64) -> u8 {
    amplitude_to_ad(g, g)
}

/// Frames of agreement required before a tone is confirmed (onset lag).
const DETECT_CONFIRM: u32 = 2;
/// Frames a confirmed tone survives disagreement/absence (offset hangover +
/// beat-null / id-flip smoothing).
const DETECT_HANGOVER: u32 = 1;
/// EMA coefficient for the confirmed tone's amplitude. Low → strong smoothing
/// (steady `A_D` on a sustained tone); 1.0 → no smoothing (per-frame `A_D`).
const DETECT_AMP_ALPHA: f64 = 0.35;
/// Frames to switch between two confirmed tones (DTMF/sweep transitions). Equal
/// to the cold confirm: a shorter value gains nothing, because the residual
/// sweep loss is genuine mid-sweep mis-id (the tone sits between grid points),
/// not switch lag.
const DETECT_TRANS_CONFIRM: u32 = 2;

/// Streaming, stateful tone detector — the encode-side counterpart of the
/// decoder's tone path. Wraps the per-frame [`detect_tone_frame`] with a
/// confirm+hangover state machine so a tone is only emitted once it has
/// persisted (rejecting transient voiced-speech harmonics that momentarily look
/// tonal) and survives short dropouts / beat-null id flips (call-progress tone
/// pairs). This reproduces the reference's 1-frame onset lag and offset hangover.
///
/// Feed frames in stream order via [`Self::process`]; each call returns the
/// tone frame for that position, or `None` for a voice frame.
#[derive(Clone, Debug, Default)]
pub struct ToneDetector {
    confirmed: Option<u8>,
    /// EMA of the confirmed tone's geometric-mean amplitude (stabilizes `A_D`:
    /// the reference emits a rock-steady value on a sustained tone, but the per-frame
    /// amplitude estimate jitters ±1 quantization step around it).
    smooth_amp: f64,
    miss: u32,
    cand_id: Option<u8>,
    cand_count: u32,
}

impl ToneDetector {
    /// A fresh detector with cold state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset to cold state (e.g. on a stream discontinuity).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Emit the confirmed tone at the smoothed amplitude, folding in this
    /// frame's raw amplitude via an EMA.
    fn emit_held(&mut self, id: u8, raw_amp: f64, alpha: f64) -> Option<ToneFrameFields> {
        if raw_amp > 0.0 {
            self.smooth_amp = alpha * raw_amp + (1.0 - alpha) * self.smooth_amp;
        }
        Some(ToneFrameFields {
            id,
            amplitude: geom_amp_to_ad(self.smooth_amp),
        })
    }

    /// Process one 160-sample frame in stream order. Returns the tone frame to
    /// emit at this position, or `None` for a voice frame.
    pub fn process(&mut self, frame: &[i16]) -> Option<ToneFrameFields> {
        let confirm = DETECT_CONFIRM;
        let hangover = DETECT_HANGOVER;
        // Frames needed to switch from one confirmed tone to a *different* one.
        let trans_confirm: u32 = DETECT_TRANS_CONFIRM;
        let alpha = DETECT_AMP_ALPHA;

        let raw = detect_tone_frame_dbg(frame, false);

        // Confirmed state: keep emitting through short disruptions.
        if let Some(conf_id) = self.confirmed {
            match &raw {
                Some(r) if r.fields.id == conf_id => {
                    self.miss = 0;
                    self.cand_id = None;
                    self.cand_count = 0;
                    return self.emit_held(conf_id, r.geom_amp, alpha);
                }
                other => {
                    // Track a possible transition to a *different* sustained
                    // tone so a real id change switches quickly, while a lone
                    // beat-null flip is absorbed by the hangover.
                    if let Some(r) = other {
                        if self.cand_id == Some(r.fields.id) {
                            self.cand_count += 1;
                        } else {
                            self.cand_id = Some(r.fields.id);
                            self.cand_count = 1;
                        }
                        if self.cand_count >= trans_confirm {
                            self.confirmed = Some(r.fields.id);
                            self.smooth_amp = r.geom_amp;
                            self.miss = 0;
                            self.cand_id = None;
                            self.cand_count = 0;
                            return Some(r.fields);
                        }
                    } else {
                        self.cand_id = None;
                        self.cand_count = 0;
                    }
                    self.miss += 1;
                    if self.miss <= hangover {
                        // Hold the last confirmed tone at its smoothed amplitude.
                        return Some(ToneFrameFields {
                            id: conf_id,
                            amplitude: geom_amp_to_ad(self.smooth_amp),
                        });
                    }
                    self.confirmed = None;
                    self.miss = 0;
                }
            }
        }

        // Unconfirmed: build a candidate until it persists `confirm` frames.
        match raw {
            Some(r) => {
                if self.cand_id == Some(r.fields.id) {
                    self.cand_count += 1;
                } else {
                    self.cand_id = Some(r.fields.id);
                    self.cand_count = 1;
                }
                if self.cand_count >= confirm {
                    self.confirmed = Some(r.fields.id);
                    self.smooth_amp = r.geom_amp;
                    self.miss = 0;
                    self.cand_id = None;
                    self.cand_count = 0;
                    Some(r.fields)
                } else {
                    None
                }
            }
            None => {
                self.cand_id = None;
                self.cand_count = 0;
                None
            }
        }
    }
}

/// Classify a half-rate frame from its prioritized info vectors `û₀..û₃`
/// (§2.10.1). Signature-first: a tone frame is `û₀(11..6) == 0x3F` with a
/// fixed-zero trailer `û₃(3..0) == 0`. A legitimate voice frame cannot produce
/// this signature (voice-path `b̂₀(6..3) = 1111` would need `b̂₀ ≥ 120`, outside
/// Annex L's valid pitch range), so the check is unambiguous. Non-tone frames
/// deprioritize and split on `b̂₀`: `[0, 119]` is Voice, above is Erasure.
pub fn classify(u: &[u16; 4]) -> FrameKind {
    // The routing field is the 6-bit escape `û₀(11..6)`, which the
    // prioritization map composes as `b̂₀(6..3) ‖ b̂₁(4..3)` — four bits of
    // pitch and two of voicing. `b̂₀ ∈ [120,127]` (i.e. `b̂₀(6..3) == 0b1111`)
    // is only half the signature; the sub-class lives in the top two bits of
    // `b̂₁`, which the escape repurposes as a selector. Testing `b̂₀` alone
    // cannot separate a tone frame from an erasure.
    match (u[0] >> 6) & 0x3F {
        0x3F => FrameKind::Tone,
        0x3E => FrameKind::Silence,
        0x3D | 0x3C => FrameKind::Erasure,
        // `< 0x3C` ⇒ `b̂₀ <= 119`; b̂₁ carries no class information here.
        _ => FrameKind::Voice,
    }
}

/// The Annex-T parameter block, in the fixed-point engine's own terms.
///
/// This is what the reference writes at the synthesis boundary for a tone
/// frame, so that the tone reaches the *same* synthesis engine voice frames
/// drive and shares the output remainder it carries between frames.
pub(crate) struct ToneBlock {
    /// Harmonic count — the highest lit harmonic, floored at `L_MIN`.
    pub l: usize,
    /// Fundamental step word (ω₀ · 262144/π).
    pub step: i16,
    /// Packed 2-bit-per-band voicing word.
    pub mask_word: u32,
    /// Per-harmonic log₂ magnitudes in Q11, all 56 entries.
    pub ml: [i16; 56],
}

/// Q11 log₂ magnitude floor the reference word-fills the `M_l` array with
/// (`0xB001`), and the clamp bounds applied to the entries it then writes.
const ML_LOG2_FILL: i16 = -20479;
const ML_LOG2_MIN: i32 = -20480;
/// Ceiling is one bit lower when two components sum into the same array.
const ML_LOG2_MAX_SINGLE: i32 = 30719;
const ML_LOG2_MAX_MULTI: i32 = 28671;

/// Build the fixed-point Annex-T block for `(I_D, A_D)`.
///
/// Returns `None` for a reserved `I_D`. `I_D = 255` (silence) yields a block
/// with no lit harmonic, which synthesizes to nothing while still advancing
/// the carried remainder.
pub(crate) fn tone_site_block(id: u8, amplitude: u8) -> Option<ToneBlock> {
    let ToneParams { f0, l1, l2 } = ANNEX_T[id as usize]?;
    let f0 = f64::from(f0);
    if f0 <= 0.0 {
        return None;
    }
    let l = usize::from(l1.max(l2)).clamp(L_MIN as usize, L_MAX);

    // step = ω₀·262144/π with ω₀ = 2π·f0/8000, i.e. f0 · 65.536. This
    // reproduces the reference's word for both of its arms — the computed
    // `I_D·2048/(g+1)` of the single-tone path and the per-I_D table of the
    // multi-tone path — because `ANNEX_T` already carries the fundamental
    // each of those encodes (I_D 12 -> 24576; I_D 128 -> 5145 = 0x1419).
    let step = (f0 * 65.536).round() as i16;

    // M_l: word-filled with the log2 floor, then one entry per component.
    let mut ml = [ML_LOG2_FILL; 56];
    if id != 255 {
        let ceiling = if l1 == l2 {
            ML_LOG2_MAX_SINGLE
        } else {
            ML_LOG2_MAX_MULTI
        };
        // log2(16384 · 10^(0.03555·(A_D−127))) in Q11.
        let log2_mag =
            14.0 + TONE_AMPLITUDE_EXPONENT_STEP * (f64::from(amplitude) - 127.0) * LOG2_10;
        let q11 = (log2_mag * 2048.0).round() as i32;
        for &lt in &[l1, l2] {
            if lt >= 1 && usize::from(lt) <= l {
                ml[usize::from(lt) - 1] = q11.clamp(ML_LOG2_MIN, ceiling) as i16;
            }
        }
    }

    // Voicing. The reference writes a literal 1 into this field, but the code
    // that reads it is skipped on the tone path, so the value cannot be taken
    // at face value: `gen_vmask` maps harmonics to 16 bands of 250 Hz, and a
    // literal 1 voices only harmonics below 250 Hz — which would leave the
    // I_D 12 tone (375 Hz, band 1) unvoiced, contradicting its measured
    // spectral purity of 1.000. So the mask is constructed to meet the
    // measured contract instead: light the band each lit harmonic falls in.
    let mut mask_word = 0u32;
    if id != 255 {
        // Mirror gen_vmask's band walk: harmonic index i sits in band
        // ((i+1)·step_g) >> 16 with step_g = step·4.
        let step_g = i64::from(step) * 4;
        for &lt in &[l1, l2] {
            if lt >= 1 && usize::from(lt) <= l {
                let band = ((i64::from(lt) * step_g) >> 16).clamp(0, 15) as u32;
                mask_word |= 1 << (2 * band);
            }
        }
    }

    Some(ToneBlock {
        l,
        step,
        mask_word,
        ml,
    })
}

/// [`classify`], with a `Tone` that fails its own redundancy or range gates
/// demoted to `Erasure`.
///
/// The gates live in [`parse_tone_frame`] and [`tone_to_mbe_params`]; a frame
/// that trips one is not a voice frame and must not be decoded as one. The
/// reference returns the same state value from a rejected tone frame as from
/// an erasure, so a rejected tone repeats the previous frame rather than
/// muting or falling through to voice.
pub fn classify_decoded(u: &[u16; 4]) -> FrameKind {
    let kind = classify(u);
    if kind == FrameKind::Tone {
        let ok = parse_tone_frame(u)
            .and_then(|f| tone_to_mbe_params(f.id, f.amplitude))
            .is_some();
        if !ok {
            return FrameKind::Erasure;
        }
    }
    kind
}

/// The four redundant `I_D` copies of Table 20, in wire order.
///
/// Copy 0 and copy 3 are contiguous 8-bit blocks; copies 1 and 2 straddle
/// bit-vector boundaries. Copy 0 is the one the decoder actually uses — see
/// [`parse_tone_frame`].
fn id_copies(u: &[u16; 4]) -> [u8; 4] {
    [
        ((u[1] >> 4) & 0xFF) as u8,                          // û₁(11..4)
        (((u[1] & 0x0F) << 4) | ((u[2] >> 7) & 0x0F)) as u8, // û₁(3..0)‖û₂(10..7)
        (((u[2] & 0x7F) << 1) | ((u[3] >> 13) & 1)) as u8,   // û₂(6..0)‖û₃(13)
        ((u[3] >> 5) & 0xFF) as u8,                          // û₃(12..5)
    ]
}

/// Parse tone-frame fields per §2.10.2. Returns `None` if the signature or the
/// fixed trailer doesn't match (caller treats as erasure/silence).
pub fn parse_tone_frame(u: &[u16; 4]) -> Option<ToneFrameFields> {
    if (u[0] >> 6) & 0x3F != 0x3F {
        return None;
    }
    if u[3] & 0x0F != 0 {
        return None;
    }
    // Redundancy gates. These are *tolerance* tests, not unanimity: the
    // decoder accepts a frame whose copies disagree in up to five bits, and
    // it takes `I_D` from copy 0 verbatim rather than majority-voting. On a
    // frame where copy 0 is the corrupted one, voting would pick a different
    // ID than the reference does.
    let c = id_copies(u);
    // Copy 1's high nibble must match copy 0's. This test rejects in
    // isolation — a single differing bit in that nibble fails the frame even
    // though it is far under the total-disagreement tolerance below.
    if (c[1] >> 4) != (c[0] >> 4) {
        return None;
    }
    // Bounded total disagreement against copy 0, across the other three.
    let spread: u32 = c[1..]
        .iter()
        .map(|&x| u32::from(x ^ c[0]).count_ones())
        .sum();
    if spread >= 6 {
        return None;
    }
    let id = c[0];
    // A_D: 6 MSBs in û₀(5..0) (= A_D(6..1)), LSB in û₃(4) (= A_D(0)).
    let ad_hi = (u[0] & 0x3F) as u8;
    let ad_lo = ((u[3] >> 4) & 1) as u8;
    let amplitude = (ad_hi << 1) | ad_lo;
    Some(ToneFrameFields { id, amplitude })
}

/// Convert `(I_D, A_D)` to [`MbeParams`] via the MBE bridge of Eq. 206–209.
///
/// Returns `None` for reserved `I_D` (no row in [`ANNEX_T`]). For `I_D = 255`
/// (silence) the amplitude is forced to zero regardless of `A_D`.
pub(crate) fn tone_to_mbe_params(id: u8, amplitude: u8) -> Option<MbeParams> {
    let ToneParams { f0, l1, l2 } = ANNEX_T[id as usize]?;
    let f0 = f64::from(f0);
    if f0 <= 0.0 {
        return None;
    }
    let omega_0 = (2.0 * PI64 / 8000.0) * f0;

    // L is the *highest lit harmonic*, floored at L_MIN — not Eq. 207's
    // `floor(3812.5/f0)`.
    //
    // Both of the reference's arms reduce to this. For a single tone it uses
    // `clamp(g+1, 9, 56)` where `g+1` is the one lit harmonic; for a multi-tone
    // frame it uses `clamp(g2+1, 9, 56)`, taking the *second* (higher) index.
    // `ANNEX_T` stores 1-based harmonic numbers, so `g+1` is `l1` and `g2+1` is
    // `l2` — and single-tone rows carry `l1 == l2`, collapsing both arms to
    // `clamp(max(l1,l2), L_MIN, L_MAX)`.
    //
    // Eq. 207 is much larger nearly everywhere (I_D 5: 24 vs 9; I_D 128: 48 vs
    // 17; I_D 130: 54 vs 19). The surplus harmonics are unlit, so the float
    // back-end renders the same audio either way — but L drives the fixed-point
    // path's site-A resampler grid and its M_l tail padding, so the two are not
    // interchangeable there.
    let l = i64::from(l1.max(l2)).clamp(L_MIN as i64, L_MAX as i64) as u8;

    // Eq. 208–209: voicing + magnitude at l1 and l2 only.
    let tone_magnitude = if id == 255 {
        0.0 // silence override per §2.10.3 Step 3
    } else {
        TONE_AMPLITUDE_PEAK
            * 10f64.powf(TONE_AMPLITUDE_EXPONENT_STEP * (f64::from(amplitude) - 127.0))
    } as f32;
    // Reference clamp on the per-harmonic log-magnitude: [-10.0, +15.0] in log2
    // for a single tone, ceiling lowered to +14.0 when two components sum.
    let ceiling = if l1 == l2 { 32768.0 } else { 16384.0 };
    let tone_magnitude = tone_magnitude.min(ceiling);

    let mut voiced = vec![false; l as usize];
    let mut amplitudes = vec![0f32; l as usize];
    for &lt in &[l1, l2] {
        if lt >= 1 && lt <= l {
            voiced[(lt - 1) as usize] = true;
            amplitudes[(lt - 1) as usize] = tone_magnitude;
        }
    }

    Some(MbeParams {
        b0: id,
        omega_0: omega_0 as f32,
        l,
        voiced,
        amplitudes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Input peak amplitude (per harmonic) that the `A_D` model quantizes to a
    /// target `A_D` — inverse of [`amplitude_to_ad`].
    fn peak_for_ad(ad: u8) -> f64 {
        10f64.powf((ad as f64 - DETECT_AD_OFFSET) / (DETECT_AD_SLOPE * 20.0))
    }

    /// Synthesize `n` frames (160 samples each) of an Annex-T tone `id` at the
    /// given per-harmonic peak amplitude.
    fn synth_tone(id: u8, peak: f64, n: usize) -> Vec<i16> {
        let ToneParams { f0, l1, l2 } = ANNEX_T[id as usize].unwrap();
        let (fl1, fl2) = (l1 as f64 * f0 as f64, l2 as f64 * f0 as f64);
        let total = n * 160;
        (0..total)
            .map(|i| {
                let t = i as f64;
                let mut s = peak * (2.0 * PI64 * fl1 * t / 8000.0).sin();
                if l2 != l1 {
                    s += peak * (2.0 * PI64 * fl2 * t / 8000.0 + 0.7).sin();
                }
                s.round().clamp(-32768.0, 32767.0) as i16
            })
            .collect()
    }

    /// A single synthesized single-tone frame is classified as its Annex-T id
    /// with the expected `A_D`.
    #[test]
    fn detect_single_tone_id_and_ad() {
        // id30 (single, 937.5 Hz), target A_D = 110.
        let ad = 110u8;
        let pcm = synth_tone(30, peak_for_ad(ad), 3);
        let d = detect_tone_frame(&pcm[160..320]).expect("tone detected");
        assert_eq!(d.id, 30);
        assert!(
            (d.amplitude as i32 - ad as i32).abs() <= 1,
            "A_D {} vs {ad}",
            d.amplitude
        );
    }

    /// A synthesized dual tone (Knox id145) is classified with both harmonics
    /// and the expected `A_D`.
    #[test]
    fn detect_dual_tone_id_and_ad() {
        let ad = 99u8;
        let pcm = synth_tone(145, peak_for_ad(ad), 3);
        let d = detect_tone_frame(&pcm[160..320]).expect("dual tone detected");
        assert_eq!(d.id, 145);
        assert!((d.amplitude as i32 - ad as i32).abs() <= 1);
    }

    /// Broadband noise is never classified as a tone (voice/noise rejection).
    #[test]
    fn noise_is_not_a_tone() {
        // Deterministic pseudo-noise (LCG), full-band.
        let mut x = 0x1234_5678u32;
        let noise: Vec<i16> = (0..160)
            .map(|_| {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
                ((x >> 8) as i16) / 4
            })
            .collect();
        assert!(detect_tone_frame(&noise).is_none());
    }

    /// The streaming detector reproduces the reference's 1-frame onset lag: the first
    /// frame of a tone is voice, the tone is emitted from the second frame.
    #[test]
    fn detector_onset_lag() {
        let pcm = synth_tone(30, peak_for_ad(110), 5);
        let mut det = ToneDetector::new();
        let outs: Vec<_> = pcm.chunks_exact(160).map(|c| det.process(c)).collect();
        assert!(outs[0].is_none(), "onset frame should be voice");
        assert!(outs[1].is_some(), "tone confirmed on the second frame");
        assert_eq!(outs[1].unwrap().id, 30);
        assert!(outs[4].is_some());
    }

    /// A lone tonal frame between voice does not confirm (rejects transient
    /// voiced-speech harmonics).
    #[test]
    fn detector_rejects_single_frame_tone() {
        let tone = synth_tone(30, peak_for_ad(110), 1);
        let mut det = ToneDetector::new();
        let silence = [0i16; 160];
        assert!(det.process(&silence).is_none());
        assert!(
            det.process(&tone).is_none(),
            "one tonal frame must not confirm"
        );
        assert!(det.process(&silence).is_none());
    }

    /// Reserved IDs map to no row; block-formula single tones agree with the
    /// algebraic generator (5 -> 156.25, 64 -> 400.0, 122 -> 381.25).
    #[test]
    fn annex_t_spot_values() {
        assert!(ANNEX_T[0].is_none());
        assert!(ANNEX_T[4].is_none());
        assert!(ANNEX_T[123].is_none());
        assert!(ANNEX_T[254].is_none());
        assert_eq!(ANNEX_T[5].unwrap().f0, 156.25);
        assert_eq!(ANNEX_T[64].unwrap().f0, 400.0);
        assert_eq!(ANNEX_T[122].unwrap().f0, 381.25);
        // id255 = silence sentinel.
        assert_eq!(ANNEX_T[255].unwrap().l1, 0);
    }

    /// Measured-frequency correction for knox_1 (id145) is applied.
    #[test]
    fn id145_uses_measured_frequency() {
        assert_eq!(ANNEX_T[145].unwrap().f0, 150.803);
    }

    /// Build a well-formed Table 20 tone frame — signature, A_D split across
    /// û₀/û₃, **all four** I_D copies written, zero trailer.
    fn tone_frame(id: u8, amplitude: u8) -> [u16; 4] {
        let ad_hi = u16::from((amplitude >> 1) & 0x3F);
        let ad_lo = u16::from(amplitude & 1);
        [
            (0x3F << 6) | ad_hi,
            (u16::from(id) << 4) | u16::from(id >> 4),
            (u16::from(id & 0x0F) << 7) | u16::from(id >> 1),
            (u16::from(id & 1) << 13) | (u16::from(id) << 5) | (ad_lo << 4),
        ]
    }

    /// `tone_frame` really does write four agreeing copies — otherwise every
    /// redundancy test below would be vacuous.
    #[test]
    fn tone_frame_fixture_writes_four_agreeing_copies() {
        for id in [0u8, 1, 12, 145, 254, 255] {
            assert_eq!(id_copies(&tone_frame(id, 100)), [id; 4], "id {id}");
        }
    }

    /// The tone signature classifies as Tone; a plain low-b0 voice frame does
    /// not; and parse round-trips a hand-built (id, amplitude).
    #[test]
    fn classify_and_parse_signature() {
        let (id, amplitude) = (145u8, 100u8);
        let u = tone_frame(id, amplitude);
        assert_eq!(classify(&u), FrameKind::Tone);
        let f = parse_tone_frame(&u).expect("tone parses");
        assert_eq!(f.id, id);
        assert_eq!(f.amplitude, amplitude);

        // A frame without the signature is not a tone.
        let voice = [0u16; 4];
        assert_ne!(classify(&voice), FrameKind::Tone);
    }

    /// The escape field `û₀(11..6)` selects four classes, and `b̂₀` alone
    /// cannot distinguish them: a tone and an erasure can share a `b̂₀`.
    #[test]
    fn escape_field_selects_four_classes() {
        let of = |sig: u16| {
            let mut u = tone_frame(12, 82);
            u[0] = (sig << 6) | (u[0] & 0x3F);
            u
        };
        assert_eq!(classify(&of(0x3F)), FrameKind::Tone);
        assert_eq!(classify(&of(0x3E)), FrameKind::Silence);
        assert_eq!(classify(&of(0x3D)), FrameKind::Erasure);
        assert_eq!(classify(&of(0x3C)), FrameKind::Erasure);
        assert_eq!(classify(&of(0x3B)), FrameKind::Voice);

        // The tone and the erasure above share b̂₀ = 120 — the sub-class is
        // carried entirely by b̂₁'s top two bits.
        assert_eq!(crate::tables::deprioritize(&of(0x3F))[0], 120);
        assert_eq!(crate::tables::deprioritize(&of(0x3C))[0], 120);
    }

    /// I_D comes from copy 0 verbatim. It is **not** a majority vote: where
    /// copy 0 is the odd one out, copy 0 still wins.
    #[test]
    fn id_is_copy_zero_not_a_majority_vote() {
        // Copies [13,12,12,12]: majority 12, copy 0 says 13.
        let mut u = tone_frame(12, 82);
        u[1] = (u[1] & 0x000F) | (13u16 << 4);
        assert_eq!(id_copies(&u)[0], 13);
        assert_eq!(parse_tone_frame(&u).expect("accepted").id, 13);
    }

    /// The redundancy gates are a tolerance, not unanimity — and the
    /// copy-1 high-nibble test rejects in isolation, well under that
    /// tolerance.
    #[test]
    fn redundancy_gates_tolerate_disagreement_but_not_a_bad_high_nibble() {
        // Rewrite copy 1 (û₁(3..0) ‖ û₂(10..7)) to an arbitrary value.
        let with_copy1 = |v: u8| {
            let mut u = tone_frame(12, 82);
            u[1] = (u[1] & 0xFFF0) | u16::from(v >> 4);
            u[2] = (u[2] & 0x087F) | (u16::from(v & 0x0F) << 7);
            u
        };
        // One bit differing in the LOW nibble: accepted.
        assert_eq!(id_copies(&with_copy1(13))[1], 13);
        assert!(parse_tone_frame(&with_copy1(13)).is_some());
        // One bit differing in the HIGH nibble: rejected on its own.
        assert_eq!(id_copies(&with_copy1(140))[1], 140);
        assert!(parse_tone_frame(&with_copy1(140)).is_none());

        // Total disagreement is bounded at 6 bits against copy 0. Copy 3 is
        // free of the nibble test, so it isolates the tolerance: 255^12 has
        // popcount 6 and must fail; 13^12 has popcount 1 and must pass.
        let with_copy3 = |v: u8| {
            let mut u = tone_frame(12, 82);
            u[3] = (u[3] & 0xE01F) | (u16::from(v) << 5);
            u
        };
        assert_eq!((255u8 ^ 12).count_ones(), 6);
        assert!(parse_tone_frame(&with_copy3(255)).is_none());
        assert!(parse_tone_frame(&with_copy3(13)).is_some());
    }

    /// A tone frame that trips a gate is an *erasure*, not a voice frame and
    /// not a silence — it must reach the repeat path.
    #[test]
    fn rejected_tone_frames_demote_to_erasure() {
        let mut bad_nibble = tone_frame(12, 82);
        bad_nibble[1] = (bad_nibble[1] & 0xFFF0) | (140u16 >> 4);
        bad_nibble[2] = (bad_nibble[2] & 0x087F) | (u16::from(140u8 & 0x0F) << 7);
        assert_eq!(classify(&bad_nibble), FrameKind::Tone);
        assert_eq!(classify_decoded(&bad_nibble), FrameKind::Erasure);

        // An I_D outside every Annex-T bin is likewise an erasure.
        let reserved = tone_frame(125, 82);
        assert_eq!(classify(&reserved), FrameKind::Tone);
        assert_eq!(classify_decoded(&reserved), FrameKind::Erasure);

        // A well-formed one is untouched.
        assert_eq!(classify_decoded(&tone_frame(12, 82)), FrameKind::Tone);
    }

    /// A non-zero û₃ trailer fails the frame per §2.10.2.
    #[test]
    fn nonzero_trailer_is_rejected() {
        let mut u = tone_frame(12, 82);
        u[3] |= 0x0001;
        assert!(parse_tone_frame(&u).is_none());
        assert_eq!(classify_decoded(&u), FrameKind::Erasure);
    }

    /// tone_to_mbe_params lights exactly the l1/l2 harmonics, at ω₀ = 2π f0/8k,
    /// and forces silence for id255.
    #[test]
    fn tone_params_bridge() {
        let p = tone_to_mbe_params(145, 127).expect("id145 valid");
        // knox_1: l1=4, l2=7 -> harmonics 4 and 7 voiced, others not.
        assert!(p.voiced[3] && p.voiced[6]);
        assert!(!p.voiced[0]);
        let want_omega = (2.0 * PI64 / 8000.0) * 150.803;
        assert!((f64::from(p.omega_0) - want_omega).abs() < 1e-4);
        // A_D = 127 -> peak magnitude.
        assert!((p.amplitudes[3] - TONE_AMPLITUDE_PEAK as f32).abs() < 1.0);

        // id255 -> all-zero amplitudes (silence).
        let s = tone_to_mbe_params(255, 127).expect("silence sentinel valid");
        assert!(s.amplitudes.iter().all(|&a| a == 0.0));

        // reserved id -> None.
        assert!(tone_to_mbe_params(0, 64).is_none());
    }
}
