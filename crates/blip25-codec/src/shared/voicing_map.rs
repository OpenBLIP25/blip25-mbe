//! `(L, step)` derivation and the `L = 56` voicing override — shared between
//! the encoder's voicing quantizer and the decoder's from-bits synthesis path.
//!
//! The voicing codebook [`VOICING_CB`] lives here because both sides read it:
//! the encoder's `b1` VQ search (`enc::voicing_vq`) and the decoder's `L = 56`
//! gate ([`l56_gate_from_b1`]). Kept out of `enc/` so the decoder can reach it
//! without the encoder analysis pipeline.

/// Voicing codebook: 128 packed dwords, 16 bands x 2-bit selector.
/// The 5-bit `b1` index selects 32 candidates at stride 4 (`ncand = 1<<5`,
/// `stride = 1<<(7-5)`); each used dword is candidate `c` = dword `c*4`.
pub(crate) const VOICING_CB: [u32; 128] = [
    1431655765, 1431655761, 1431590229, 1431328085, 1431655764, 1431393621, 1431393617, 1431655766,
    1431590228, 1430607189, 1431328084, 1431655760, 1431655749, 1430279509, 1427461461, 1435850069,
    1431328080, 1431328000, 1426412805, 1498763605, 1364545877, 1431328064, 1427461397, 1364547925,
    1414878293, 1430541636, 1431655744, 1431393616, 1430607173, 1431590212, 1431590224, 1448432725,
    1426085120, 1426085200, 1426150657, 1430279492, 1430279488, 1430279504, 1430345025, 1430279508,
    1409307648, 1426084864, 1409307904, 1426085184, 1409373184, 1430279168, 1430279424, 1426085124,
    1342197760, 1342199040, 1342198784, 1342198272, 1409306624, 1342199808, 1426083840, 1358974976,
    1073758208, 1342193664, 1342177280, 1375752192, 1409286144, 21504, 1073762304, 1476415488, 0,
    256, 1073741824, 1, 16384, 268435456, 20480, 4096, 2147483648, 176160768, 2818572288, 44040192,
    2684354560, 2852126720, 44695552, 167772160, 2863267840, 8388608, 178913280, 2860515328,
    134217728, 2862612480, 715784192, 2863136768, 131072, 524288, 655360, 2097152, 2752512,
    11141120, 33554432, 41943040, 2863311530, 2863270570, 2818615296, 44696234, 178916010,
    2863310848, 2684395520, 134219776, 32768, 640, 512, 2728, 2048, 2560, 2688, 672, 43690, 43648,
    43008, 10922, 40960, 682, 2730, 43680, 2, 42, 8, 32, 10, 170, 128, 160,
];

/// The `L = 56` override gate.
///
/// The frame's `L`/`step` derivation works like this: it calls the gate over the
/// 32-bit voicing word at struct +8. When the gate returns 0 (normal) it falls
/// through to the pitch table. When the gate returns nonzero it forces
/// `step = 4217` (0x1079, NOT pitch[119]'s 0x10a4) and `L = 56`, writing both
/// into the two output structs.
///
/// The gate itself is NOT a numeric threshold on `b1`. It is a structural test on the
/// 2-bit-per-band voicing word `v = VOICING_CB[b1 * 4]`:
///
/// * `A = (v & 0x55555555) != 0` — some band carries code 1 (bit 0 set)
/// * `B = (v & 0xaaaaaaaa) != 0` — some band carries code 2 (bit 1 set)
/// * gate fires (returns 1) iff `A == 0 && B != 0`; both other exits return 0.
///
/// Over all 32 codebook entries that predicate selects **exactly** `b1 ∈ {18..=31}`,
/// which is why the gate looks like a "`b1 >= 18`" threshold. Do not implement it as
/// one: `b1 >= 18` is an EMERGENT consequence of the codebook's ordering (codes
/// 0..15 all have `A == 1`; code 16 is all-zero so `B == 0`; code 17 = `0x00004000`
/// has `A == 1`), not the mechanism. On a small sample it can look like `b1 >= 21`.
#[inline]
pub(crate) fn l56_gate_from_b1(b1: u16) -> bool {
    let v = match VOICING_CB.get((b1 as usize) * 4) {
        Some(v) => *v,
        None => return false,
    };
    (v & 0x5555_5555) == 0 && (v & 0xaaaa_aaaa) != 0
}

/// The override's fundamental-frequency step, `0x1079` (4217).
/// Out of the pitch table's range on purpose: `pitch[119]` is `0x10a4` (4260), `pitch[118]`
/// `0x10e5`. Applying `L = 56` WITHOUT this step measurably REGRESSES the amplitude bytes.
pub(crate) const L56_STEP: i16 = 0x1079;
/// The override's harmonic count, `0x38` (56).
pub(crate) const L56_L: usize = 56;

/// The encoder's `(L, step)` derivation for a frame — the Rust of the gated branch
/// in the `L`/`step` derivation. Returns `None` exactly when `b0` is not a decodable
/// pitch index.
///
/// When the gate fires this returns `(56, 0x1079)` — BOTH halves. Applying `L = 56` while
/// keeping the pitch table's own step is measurably WRONG (it regresses the amplitude bytes
/// below baseline), because the harmonic sampling of the spectrum moves with `omega_0`.
///
/// This is `pub` and pure: no fs, no env, no globals.
pub(crate) fn l_step_from_b0_b1(b0: u8, b1: u16) -> Option<(usize, i16)> {
    if l56_gate_from_b1(b1) {
        return Some((L56_L, L56_STEP));
    }
    let p = crate::dequantize::decode_pitch(b0)?;
    let step = ((p.omega_0 as f64) * (524288.0 / (2.0 * core::f64::consts::PI))).round() as i16;
    Some((p.l as usize, step))
}
