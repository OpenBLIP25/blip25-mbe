//! FROM-BITS voicing generators for the decode fixed-point path.
//!
//! * [`gen_vmask`] — the voicing expander: a 2-bit-per-band voicing
//!   `maskword` -> the 56-element per-harmonic voicing array the voiced/unvoiced
//!   synthesis reads. Bit-exact port, given the maskword.
//! * [`build_maskword`] — a FROM-BITS reconstruction of the band builder's
//!   `maskword`. It maps the b1 voicing decision (`expand_vuv`) onto the
//!   16-slot 2-bit band layout via the same `omega0` stepping `gen_vmask`
//!   inverts. NOT bit-exact: the DLL builder's exact band boundaries are
//!   unknown, and this agrees with the reference maskword on ~2609/3700 site-B
//!   subframes. It is the remaining from-bits gap on the voicing path.

/// Decode-side voicing maskword codebook, indexed by the 5-bit voicing index
/// `b1`. This is the 2-bit-per-band voicing word the maskword expander
/// consumes (struct field `+8`). The word is a pure function of `b1`: every
/// `b1 ∈ 0..=31` maps to exactly ONE maskword, independent of L/omega0, so it
/// is NOT an omega0-driven band expansion. Distinct from the ENCODER's
/// `VOICING_CB` (the DLL decoder never reads `VOICING_CB`); the two are
/// related only through the underlying per-band voicing they both encode.
pub(crate) const VMASK_CB: [u32; 32] = [
    0x55555555, 0x55555555, 0x05555555, 0x55555555, // 0..3
    0x00555555, 0x55555055, 0x55550555, 0x55055555, // 4..7
    0x00005555, 0x00055555, 0x00000555, 0x50000555, // 8..11
    0x00000055, 0x00000555, 0x00000005, 0x00000555, // 12..15
    0x00000000, 0x00000000, 0x0000000a, 0x000000aa, // 16..19
    0xaaaaaaaa, 0x00000a00, 0xa0000000, 0xaaa00000, // 20..23
    0xaaaaaaaa, 0xaaaaaa00, 0x00000000, 0x00000000, // 24..27
    0x00000000, 0x00000000, 0x00000000, 0x00000000, // 28..31
];

/// Decode-side **site-A** (mid-frame, odd c520 call) voicing maskword codebook,
/// indexed by the 5-bit voicing index `b1`. This is the SIBLING of [`VMASK_CB`]
/// (site B): the site-A mid-frame param struct carries
/// its OWN `+8` voicing word, distinct from site B's.
///
/// The word is a pure function of `b1`: prev-independent, NOT a
/// `combine(cur, prev)` (AND/prev/min all fail), and NOT the site-B word
/// (`VMASK_CB[b1]` agrees on only 3114/3700 site-A calls).
///
/// # Mechanism (why it differs from [`VMASK_CB`], and why it is still f(b1))
///
/// The site-A struct's voicing is the current-frame per-band voicing carried
/// onto site A's geometric-mean grid (`a_om`/`a_l`). Two regimes:
///   * **Normal bands (`b1` 0..17):** a near-identity resample of the site-B
///     word onto `a_om`; the small grid drift shifts a band or two (e.g.
///     `b1=1`: site-B `0x55555555` → site-A `0x05555555`, a one-band shift).
///   * **L=56 override bands (`b1` 18..31):** the mixed voicing 'a'-pattern is
///     carried by whichever site the override index selects — the SAME
///     site-split as [`p4e_from_bits`]: `b1∈18..23` → site B carries it (site A
///     = 0), `b1∈26..31` → site A carries it (site B = 0), `b1∈24,25` → both.
///     This is why site-A's 18..23 are 0 and its 26..31 hold the 'a' words that
///     site B zeroes.
///
/// Because `a_om`, the resample, and the override site-split are all determined
/// by `b1`, the repacked word is single-valued per `b1`.
///
/// Memoized codebook, matching the reference site-A `p4_maskword` on every
/// site-A call of the reference file — the same evidence class as site B's
/// [`VMASK_CB`]. Some `b1` values are thinly sampled (17: 3 rows, 30: 8,
/// 22: 12, 11: 14): single-valued, but lightly covered.
pub(crate) const SITEA_VMASK_CB: [u32; 32] = [
    0x55555555, 0x05555555, 0x05555555, 0x55055555, // 0..3
    0x00555555, 0x55555055, 0x55550555, 0x55055555, // 4..7
    0x00005555, 0x00055555, 0x00000555, 0x00000555, // 8..11
    0x00000055, 0x00000055, 0x00000005, 0x00000000, // 12..15
    0x00000000, 0x00000005, 0x00000000, 0x00000000, // 16..19
    0x00000000, 0x00000000, 0x00000000, 0x00000000, // 20..23
    0xaaaaaaaa, 0xaaaaaa00, 0x0000000a, 0x00000a00, // 24..27
    0xaaaaaaaa, 0x000000aa, 0xa0000000, 0xaa000000, // 28..31
];

/// FROM-BITS **site-A** voicing maskword: a pure lookup on the 5-bit voicing
/// index `b1` ([`SITEA_VMASK_CB`]). Signature-compatible with [`build_maskword`]
/// (site B); `b0`/`l`/`omega0` do not affect the result. See
/// [`SITEA_VMASK_CB`] for the mechanism.
pub(crate) fn sitea_maskword(_b0: u8, b1: u16, _l: usize, _omega0: i16) -> u32 {
    SITEA_VMASK_CB.get(b1 as usize).copied().unwrap_or(0)
}

/// Expand a 2-bit-per-band `maskword` into the 56-word per-
/// harmonic voicing array. `[L..56]` is zeroed. Bit-exact.
pub(crate) fn gen_vmask(maskword: u32, omega0: i32, l: i32) -> [i16; 56] {
    let v: [i32; 16] = std::array::from_fn(|b| ((maskword >> (2 * b)) & 3) as i32);
    let om16 = (omega0 as i16) as i32; // sign-extend the low 16 bits
    let step = (om16 << 6) >> 4; // shift left 6, then arithmetic (signed) shift right 4
    let mut out = [0i16; 56];
    let mut acc = step; // acc starts at step before the loop -> first iter uses (0+1)*step
    let ll = l.clamp(0, 56);
    for slot in out.iter_mut().take(ll as usize) {
        let raw = (acc >> 16) & 0xffff;
        let j = if (raw as i16 as i32) > 15 { 15 } else { raw };
        let jp = if j + 1 > 15 { 15 } else { j + 1 };
        let mut val = v[j as usize] | v[jp as usize];
        if val >= 3 {
            val = 1;
        }
        *slot = val as i16;
        acc = acc.wrapping_add(step);
    }
    out
}

/// FROM-BITS voicing maskword: a pure lookup on the 5-bit voicing index `b1`
/// ([`VMASK_CB`]). `b0`/`l`/`omega0` are accepted for signature compatibility
/// but do NOT affect the result (the DLL's maskword is omega0/L-independent).
/// Returns the 32-bit 2-bit-per-band voicing word `gen_vmask` / the gates
/// consume.
pub(crate) fn build_maskword(_b0: u8, b1: u16, _l: usize, _omega0: i16) -> u32 {
    VMASK_CB.get(b1 as usize).copied().unwrap_or(0)
}

/// FROM-BITS `p4.e` -- the `+0xe` c520 frame-struct scalar (`b00cap_meta`
/// `p4_e`), a pitch-derived quantized amplitude for the L=56 mixed-voicing
/// (override) band. Nonzero only on L=56 override subframes.
///
/// The L=56 voicing sub-band index `b1` in {18..31} selects which of the two
/// c520 sites carries the scalar `V = 260*b0 - 16510` (a linear function of the
/// 7-bit pitch index `b0`):
///   * `b1` in {18..23}: **site B** carries `V` (site A = 0)
///   * `b1` in {26..31}: **site A** carries `V` (site B = 0)
///   * `b1` in {24,25}:  BOTH sites carry an OFF grid value -- site B CLAMPS
///     `round((b0-60)/11)` ([`P4E_OFF_GRID`]); site A WRAPS the same pitch-index
///     residue mod 11 (`iA = fold11((b0-60) mod 11)`). The wrap-vs-clamp split is
///     the mid-frame band-split counterpart to site B (data-derived, 45/45).
///
/// `site`: 0 = A (odd c520 call, mid-frame), 1 = B (even).
pub(crate) const P4E_OFF_GRID: [i32; 11] = [
    -15127, -12102, -9077, -6051, -3026, 0, 3025, 6050, 9076, 12101, 15126,
];
pub(crate) fn p4e_from_bits(site: usize, b0: u16, b1: u16, l: usize) -> i32 {
    if l != 56 {
        return 0;
    }
    let v = 260 * (b0 as i32) - 16510;
    // site-B coarse-off: m = round((b0-60)/11) clamped to [-5,5]
    let off_b = |b0: i32| -> i32 {
        let num = 2 * (b0 - 60) + 11;
        let mut m = if num >= 0 {
            num / 22
        } else {
            -((-num + 21) / 22)
        };
        m = m.clamp(-5, 5);
        P4E_OFF_GRID[(m + 5) as usize]
    };
    if site == 1 {
        // site B (even call)
        if (18..=23).contains(&b1) {
            v
        } else if b1 == 24 || b1 == 25 {
            off_b(b0 as i32)
        } else {
            0
        }
    } else {
        // site A (odd call): V for the upper L=56 sub-band range.
        if (26..=31).contains(&b1) {
            v
        } else if b1 == 24 || b1 == 25 {
            // site-A off branch: the mid-frame band-split WRAPS the pitch-index
            // residue mod 11 (vs site B's CLAMP of round((b0-60)/11)):
            // iA = fold11((b0-60) mod 11); peA = P4E_OFF_GRID[iA+5].
            let r = (((b0 as i32 - 60) % 11) + 11) % 11; // 0..10
            let ia = if r > 5 { r - 11 } else { r }; // -5..5
            P4E_OFF_GRID[(ia + 5) as usize]
        } else {
            0
        }
    }
}
