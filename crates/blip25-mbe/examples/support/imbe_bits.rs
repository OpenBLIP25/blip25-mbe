//! Shared plumbing for the IMBE oracle instruments (`imbe_scoreboard`,
//! `imbe_divergence`).
//!
//! Not an example itself — it lives under `examples/support/` so cargo does not
//! try to build it as a binary, and is pulled in with `#[path = ...] mod`.
//!
//! The unit of comparison is the **parameter vector** `b̂₀..b̂_{L+2}`, not the
//! wire bytes: the 88 prioritized info bits are an `L`-dependent packing, so two
//! frames that disagree on `b̂₀` disagree on the *meaning* of every later bit.
//! Both sides are deprioritized with their own `L` (recovered from their own
//! `b̂₀`, whose position in `û` is fixed and `L`-independent), which is the only
//! way to name a field on a frame where the two sides chose different `L`.

#![allow(dead_code)]

use blip25_mbe::imbe7200::priority::{deprioritize, L_MAX, L_MIN};
use blip25_mbe::vocoder::{LiveEncoder, Rate, Vocoder};

/// `b̂₀..b̂_{L+2}` for `L = 56` ⇒ 59 slots (= `priority::IMBE_B_MAX`).
pub const IMBE_B_MAX: usize = 59;

/// Info-only IMBE payload width (`Rate::Imbe4400x4400`).
pub const PAYLOAD: usize = 11;

/// 20 ms at 8 kHz.
pub const FRAME: usize = 160;

/// Widths of `û₀..û₇`, MSB-first over the 11-byte payload.
const U_WIDTHS: [usize; 8] = [12, 12, 12, 12, 11, 11, 11, 7];

/// Unpack an 11-byte / 88-bit info payload into `û₀..û₇` (MSB-first).
pub fn unpack_u(payload: &[u8]) -> [u16; 8] {
    let bit = |p: usize| ((payload[p / 8] >> (7 - (p % 8))) & 1) as u16;
    let mut u = [0u16; 8];
    let mut p = 0;
    for (i, &w) in U_WIDTHS.iter().enumerate() {
        let mut v = 0u16;
        for _ in 0..w {
            v = (v << 1) | bit(p);
            p += 1;
        }
        u[i] = v;
    }
    u
}

/// `b̂₀` read straight from the info vectors. Its bit positions do not depend on
/// `L`, so it can be recovered before `L` is known — which is what makes the
/// whole comparison possible.
pub fn b0_of(u: &[u16; 8]) -> i16 {
    (((u[0] >> 4) & 0xFC) | ((u[7] >> 1) & 0x3)) as i16
}

/// One decoded frame in parameter space.
pub struct Dec {
    /// Raw 11-byte payload, as transmitted.
    pub raw: [u8; PAYLOAD],
    /// Pitch index 0..=207.
    pub b0: i16,
    /// Harmonic count `L` implied by `b0` (clamped when invalid).
    pub l: u8,
    /// Voicing-band count `K` implied by `L`; `b̂₁` is `K` bits wide.
    pub k: usize,
    /// `b̂₀..b̂_{L+2}`; slots above `L+2` are zero.
    pub b: [u16; IMBE_B_MAX],
    /// False when `b0`/`L` falls outside the codec's voice range.
    pub valid: bool,
}

impl Dec {
    /// Highest populated parameter index (`L+2`, the sync bit).
    pub fn top(&self) -> usize {
        (self.l as usize + 2).min(IMBE_B_MAX - 1)
    }
    /// Amplitude parameter range `b̂₃..=b̂_{L+1}` as a half-open index range.
    pub fn amp_range(&self) -> std::ops::Range<usize> {
        3..(self.l as usize + 2).min(IMBE_B_MAX)
    }
}

/// Decode an 11-byte payload into parameter space.
pub fn decode(payload: &[u8]) -> Dec {
    let mut raw = [0u8; PAYLOAD];
    raw.copy_from_slice(&payload[..PAYLOAD]);
    let u = unpack_u(&raw);
    let b0 = b0_of(&u);
    let (_fund, nh) = blip25_codec::imbe::pitch_cell(b0);
    if !(0..=207).contains(&b0) || !(i16::from(L_MIN)..=i16::from(L_MAX)).contains(&nh) {
        let l = nh.clamp(i16::from(L_MIN), i16::from(L_MAX)) as u8;
        return Dec {
            raw,
            b0,
            l,
            k: blip25_codec::imbe::num_bands(i16::from(l)),
            b: [0u16; IMBE_B_MAX],
            valid: false,
        };
    }
    let l = nh as u8;
    Dec {
        raw,
        b0,
        l,
        k: blip25_codec::imbe::num_bands(nh),
        b: deprioritize(&u, l),
        valid: true,
    }
}

/// Bit width of parameter `b̂_p` at harmonic count `l`.
///
/// Derived from the priority table itself rather than a second copy of the
/// allocation rule: every `û` bit maps to exactly one `(param, bit)` pair, so
/// deprioritizing an all-ones `û` yields each parameter's full-width mask.
pub fn field_width(l: u8, p: usize) -> u32 {
    let mask = deprioritize(&[0xFFFFu16; 8], l);
    16 - mask[p].leading_zeros()
}

/// Load reference bits. Accepts both shapes in the tree:
///
/// * `oracle_bits/<vec>.imbe.bits` — 11 bytes per input frame, pad stripped.
/// * `wave7k_gold_standard/imbe/<vec>_wave7k.bits` — raw 12-byte DLL units with
///   a trailing zero pad byte, including the priming frame.
///
/// `unit` forces the unit size; `None` auto-detects and reports what it chose.
pub fn load_reference(path: &str, unit: Option<usize>) -> (Vec<Dec>, usize) {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("reference bits {path}: {e}"));
    let unit = unit.unwrap_or_else(|| {
        let twelve = raw.len() % 12 == 0 && raw.chunks_exact(12).all(|u| u[11] == 0);
        let eleven = raw.len() % 11 == 0;
        match (eleven, twelve) {
            (true, false) => 11,
            (false, true) => 12,
            (true, true) => 11,
            (false, false) => panic!(
                "{path}: length {} is neither 11- nor 12-byte units",
                raw.len()
            ),
        }
    });
    let frames = raw
        .chunks_exact(unit)
        .map(|u| decode(&u[..PAYLOAD]))
        .collect();
    (frames, unit)
}

/// Our encoder, whole-buffer — the shipped IMBE path (`Vocoder::encode`).
pub fn ours_whole(pcm: &[i16]) -> Vec<Dec> {
    Vocoder::new(Rate::Imbe4400x4400)
        .encode(pcm)
        .iter()
        .map(|f| decode(f))
        .collect()
}

/// Our encoder, streaming (`LiveEncoder`). A different amplitude mechanism and
/// look-ahead depth, so it is scored separately rather than assumed equal.
pub fn ours_live(pcm: &[i16]) -> Vec<Dec> {
    let mut le = LiveEncoder::new(Rate::Imbe4400x4400);
    let mut out = Vec::new();
    for ch in pcm.chunks(FRAME) {
        for r in le.push(ch) {
            out.push(decode(&r.expect("live encode")));
        }
    }
    for f in le.flush().expect("live flush") {
        out.push(decode(&f));
    }
    out
}

/// Read a raw little-endian i16 PCM file.
pub fn read_pcm(path: &str) -> Vec<i16> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("pcm {path}: {e}"))
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// The pitch word our IMBE chain actually consumes, per emitted frame.
///
/// `Vocoder::encode` does not derive IMBE pitch from the IMBE analysis: it runs
/// the reverse-engineered AMBE+2 spectral-DP pitch track, applies a source-RMS
/// silence gate, and hands the result in as `forced_b0`. That word is an
/// **AMBE** 7-bit pitch index, requantized onto the 8-bit IMBE grid inside the
/// encoder — so the transmitted `b̂₀` is two quantizations deep, and the
/// transmitted field cannot show sub-step error in either of them.
///
/// This reproduces that sequence so the instruments can read it directly. It
/// mirrors `Vocoder::encode`'s IMBE arm; the two constants below are that arm's
/// private `IMBE_SILENCE_B0` / `IMBE_SILENCE_RMS`.
pub fn internal_ambe_b0(pcm: &[i16]) -> Vec<u8> {
    const IMBE_SILENCE_B0: u8 = 31;
    const IMBE_SILENCE_RMS: f64 = 400.0;
    let residue = pcm.len() % FRAME;
    let padded: Vec<i16> = if residue == 0 {
        pcm.to_vec()
    } else {
        pcm.iter()
            .copied()
            .chain(std::iter::repeat_n(0i16, FRAME - residue))
            .collect()
    };
    let track = blip25_codec::enc::pcm_encode::encode_pcm_b0_no_tone_branch(
        &padded,
        blip25_codec::enc::pcm_encode::EncodeOpts::default(),
    );
    if track.is_empty() {
        return Vec::new();
    }
    let nf = padded.len() / FRAME;
    (0..nf)
        .map(|f| {
            let chunk = &padded[f * FRAME..(f + 1) * FRAME];
            let ss: f64 = chunk.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            if (ss / FRAME as f64).sqrt() < IMBE_SILENCE_RMS {
                IMBE_SILENCE_B0
            } else {
                track[(f + 2).min(track.len() - 1)]
            }
        })
        .collect()
}

/// The `b̂₀` the reference packer writes when its voicing search returns the
/// all-unvoiced code (`FUN_1030bda0`). On those frames `b̂₀` is not a pitch.
pub const IMBE_UV_B0: i16 = 25;

/// The prequantiser pitch word per ANALYSIS frame — the quantity
/// `blip25_codec::imbe::quantize::imbe_b0_from_pitch_word` consumes when
/// `BLIP25_IMBE_NEXT` is on. Indexed by analysis frame, so a caller comparing
/// against emitted frames applies the injection lag itself.
pub fn internal_pitch_word(pcm: &[i16]) -> Vec<u16> {
    let residue = pcm.len() % FRAME;
    let padded: Vec<i16> = if residue == 0 {
        pcm.to_vec()
    } else {
        pcm.iter()
            .copied()
            .chain(std::iter::repeat_n(0i16, FRAME - residue))
            .collect()
    };
    blip25_codec::enc::pcm_encode::encode_pcm_pitch_no_tone_branch(
        &padded,
        blip25_codec::enc::pcm_encode::EncodeOpts::default(),
    )
    .word
}

/// ω₀ (rad/sample) the AMBE pitch word `b0` decodes to — the continuous value
/// the IMBE grid then requantizes.
pub fn ambe_omega0(b0: u8) -> Option<f64> {
    blip25_codec::dequantize::decode_pitch(b0).map(|p| f64::from(p.omega_0))
}

/// ω₀ (rad/sample) at the centre of IMBE pitch cell `b0`.
pub fn imbe_omega0(b0: i16) -> f64 {
    blip25_codec::imbe::pitch_omega0(b0)
}

/// The set of IMBE pitch cells our chain can ever emit: the image of the AMBE
/// pitch ladder under `decode_pitch → quantize_pitch`.
pub fn reachable_imbe_b0() -> Vec<bool> {
    let mut ok = vec![false; 256];
    for a in 0..=255u8 {
        if let Some(p) = blip25_codec::dequantize::decode_pitch(a) {
            let (b0, _) = blip25_codec::imbe::quantize_pitch(p.omega_0);
            ok[b0 as usize] = true;
        }
    }
    ok
}

/// Percentage helper.
pub fn pc(x: usize, n: usize) -> f64 {
    100.0 * x as f64 / n.max(1) as f64
}

/// Median of a slice that is about to be sorted in place.
pub fn median(v: &mut [i64]) -> i64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}
