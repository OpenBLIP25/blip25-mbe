//! Committed integrity checker for the `cargo fuzz` seed corpus.
//!
//! CI runs each fuzz target with `-max_total_time=60`, which loads whatever
//! bytes sit in `fuzz/corpus/<target>/`. libFuzzer does not care what a seed
//! *means*, so a truncated file, a stale regeneration, or a selector byte
//! pointing at the wrong rate is indistinguishable from a good seed and the job
//! stays green while the seed stops reaching the state it was written to reach.
//! This test is the only thing that notices.
//!
//! Three layers, all hermetic (no reference paths, no env vars, no network):
//!
//! 1. **Presence** — the corpus directory and all four target subdirectories
//!    exist and hold exactly the manifest below. A missing corpus is a panic,
//!    never a skip.
//! 2. **Framing** — every seed's selector byte(s) resolve to the rate its
//!    filename claims *under the `RATES` array of its own target*. The four
//!    arrays are not in the same order (`decode_soft` swaps the middle two) and
//!    `transcode` takes two selector bytes, so the order is re-read from the
//!    fuzz target sources and cross-checked. Payload lengths must be a whole
//!    number of frames for the resolved rate, and every seed is driven through
//!    the same public API its target drives.
//! 3. **Content** — each seed class is pinned to the property that makes it
//!    worth committing: Annex-T tone frames are rebuilt from `(I_D, A_D)`,
//!    FEC-error frames are re-derived by flipping a known bit pattern,
//!    concealment runs must walk repeat → mute, LLR extremes must be saturated,
//!    and so on. `seed_classes_are_all_pinned` fails if a seed appears that no
//!    content check covers.
//!
//! ## Cross-seed reference frames
//!
//! The corpus is built around four reference voice frames per rate, committed
//! as `decode_bits/valid_<rate>_4frames.bin`. Nearly every other class quotes
//! them: `valid_<rate>_1frame` is reference frame 0, the concealment runs wrap
//! six bad frames in references 0/1 … 2/3, the soft `valid_strong` /
//! `valid_weak` / `rescuable` seeds are LLR expansions of them, and the
//! transcode seeds re-use them as source frames. Checking those identities is
//! what makes a single edited seed detectable.
//!
//! ## Known corpus deviation — tone seeds omit `I_D` copies 1..3
//!
//! `rate33::dequantize::encode_tone_frame_info` spreads four redundant copies
//! of `I_D` across `û₀..û₃` per Table 20. The committed tone seeds carry only
//! the `û₀` signature and the copy-4 fields in `û₃`, leaving `û₁` / `û₂` at a
//! constant filler, so they are not byte-identical to that builder. They are
//! still valid tone frames — `classify` and `parse_tone_frame` read `û₀` and
//! `û₃` only — and `tone_info_matches_the_canonical_builder` proves the fields
//! agree. The frames are rebuilt here from the Annex-T field layout plus the
//! two frozen filler constants.
//!
//! ## Known corpus defect — `rescuable_*` does not rescue
//!
//! `rescuable_<rate>.bin` is byte-exactly reference frame 3 expanded to ±100
//! LLRs with three sign flips at channel bits 3, 17 and 41 carried at magnitude
//! 1, and that construction is asserted here. It does **not** achieve what its
//! name claims. Decoding it at magnitude 1 and at magnitude 100 yields
//! identical info vectors, so soft decision buys nothing over a hard slice, and
//! the recovered frame is not the clean one: at 3600 bps channel bit 3 lands in
//! the uncoded `û₂`, and at 7200 bps bits 17 and 41 both land in `û₆`, which is
//! Hamming(15,11) and already carries one error in the reference frame. The
//! assertions below therefore pin what is true — the Golay-protected vectors
//! are recovered, and `unrescuable_*` recovers none of them — rather than the
//! filename's claim.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use blip25_codec::tone::FrameKind;
use blip25_mbe::vocoder::{FrameDisposition, Rate, Transcoder, Vocoder, VocoderError};
use blip25_mbe::{imbe7200, rate33};

/// Package-root-relative so the path resolves from the working directory cargo
/// sets for test binaries.
const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fuzz/corpus");
const TARGET_SRC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fuzz/fuzz_targets");

const FRAME_SAMPLES: usize = 160;
/// Bytes per PCM frame in an `encode_pcm` seed: 160 samples, big-endian i16.
const PCM_FRAME_BYTES: usize = FRAME_SAMPLES * 2;
/// Largest valid IMBE pitch index (Annex L); above this is an erasure.
const IMBE_PITCH_INDEX_MAX: u8 = 207;
/// Widths of the IMBE info vectors `û₀..û₇`, for unpacking a no-FEC frame.
const IMBE_INFO_WIDTHS: [u8; 8] = [12, 12, 12, 12, 11, 11, 11, 7];
/// Peak sample above which a decoded frame is carrying real audio rather than
/// the mute gate's comfort noise, which is bounded to `[-5, 5]`.
const AUDIBLE_PEAK: i32 = 20;
/// Peak sample at or below which a decoded frame is the mute gate's output.
const MUTE_PEAK: i32 = 5;

// ---------------------------------------------------------------------------
// Selector-byte tables — one per target, mirroring `fuzz/fuzz_targets/*.rs`.
// `fuzz_target_rate_tables_match` re-reads the sources and proves these are
// still the same arrays and still in the same order.
// ---------------------------------------------------------------------------

const RATES_DECODE_BITS: [Rate; 4] = [
    Rate::Imbe7200x4400,
    Rate::Imbe4400x4400,
    Rate::AmbePlus2_3600x2450,
    Rate::AmbePlus2_2450x2450,
];

/// The middle two are swapped relative to every other target.
const RATES_DECODE_SOFT: [Rate; 4] = [
    Rate::Imbe7200x4400,
    Rate::AmbePlus2_3600x2450,
    Rate::Imbe4400x4400,
    Rate::AmbePlus2_2450x2450,
];

const RATES_ENCODE_PCM: [Rate; 4] = RATES_DECODE_BITS;
const RATES_TRANSCODE: [Rate; 4] = RATES_DECODE_BITS;

// ---------------------------------------------------------------------------
// Manifest — a seed vanishing must fail, so the names are frozen here.
// ---------------------------------------------------------------------------

const DECODE_BITS_SEEDS: [&str; 26] = [
    "conceal_ambe2450_erasure_run.bin",
    "conceal_ambe3600_erasure_run.bin",
    "conceal_imbe4400_badpitch_run.bin",
    "conceal_imbe7200_badpitch_run.bin",
    "fec_errors_ambe2450.bin",
    "fec_errors_ambe3600.bin",
    "fec_errors_imbe4400.bin",
    "fec_errors_imbe7200.bin",
    "ones_ambe2450.bin",
    "ones_ambe3600.bin",
    "ones_imbe4400.bin",
    "ones_imbe7200.bin",
    "tone_annex_t_ambe2450.bin",
    "tone_annex_t_ambe3600.bin",
    "valid_ambe2450_1frame.bin",
    "valid_ambe2450_4frames.bin",
    "valid_ambe3600_1frame.bin",
    "valid_ambe3600_4frames.bin",
    "valid_imbe4400_1frame.bin",
    "valid_imbe4400_4frames.bin",
    "valid_imbe7200_1frame.bin",
    "valid_imbe7200_4frames.bin",
    "zeros_ambe2450.bin",
    "zeros_ambe3600.bin",
    "zeros_imbe4400.bin",
    "zeros_imbe7200.bin",
];

const DECODE_SOFT_SEEDS: [&str; 18] = [
    "erasure_llr_ambe3600.bin",
    "max_llr_ambe3600.bin",
    "max_llr_imbe7200.bin",
    "min_llr_ambe3600.bin",
    "min_llr_imbe7200.bin",
    "nofec_rejected_ambe2450.bin",
    "nofec_rejected_imbe4400.bin",
    "rescuable_ambe3600.bin",
    "rescuable_imbe7200.bin",
    "tone_llr_ambe3600.bin",
    "unrescuable_ambe3600.bin",
    "unrescuable_imbe7200.bin",
    "valid_strong_ambe3600.bin",
    "valid_strong_imbe7200.bin",
    "valid_weak_ambe3600.bin",
    "valid_weak_imbe7200.bin",
    "zero_llr_ambe3600.bin",
    "zero_llr_imbe7200.bin",
];

const ENCODE_PCM_SEEDS: [&str; 19] = [
    "dc_negative_imbe7200.bin",
    "dc_positive_ambe2450.bin",
    "digital_silence_ambe2450.bin",
    "digital_silence_ambe3600.bin",
    "digital_silence_imbe4400.bin",
    "digital_silence_imbe7200.bin",
    "impulse_ambe3600.bin",
    "odd_length_imbe7200.bin",
    "rail_alternating_ambe3600.bin",
    "rail_alternating_imbe7200.bin",
    "rail_full_negative_imbe4400.bin",
    "rail_full_positive_imbe7200.bin",
    "ramp_imbe4400.bin",
    "selector_only_ambe2450.bin",
    "speech_ambe2450.bin",
    "speech_ambe3600.bin",
    "speech_imbe4400.bin",
    "speech_imbe7200.bin",
    "speech_silence_speech_ambe3600.bin",
];

const TRANSCODE_SEEDS: [&str; 32] = [
    "conceal_ambe2450_to_ambe3600.bin",
    "conceal_ambe3600_to_ambe2450.bin",
    "conceal_ambe3600_to_imbe7200.bin",
    "conceal_imbe4400_to_imbe7200.bin",
    "conceal_imbe7200_to_ambe3600.bin",
    "conceal_imbe7200_to_imbe4400.bin",
    "ones_ambe2450_to_ambe3600.bin",
    "ones_ambe3600_to_ambe2450.bin",
    "ones_ambe3600_to_imbe7200.bin",
    "ones_imbe4400_to_imbe7200.bin",
    "ones_imbe7200_to_ambe3600.bin",
    "ones_imbe7200_to_imbe4400.bin",
    "single_ambe2450_to_ambe3600.bin",
    "single_ambe3600_to_ambe2450.bin",
    "single_ambe3600_to_imbe7200.bin",
    "single_imbe4400_to_imbe7200.bin",
    "single_imbe7200_to_ambe3600.bin",
    "single_imbe7200_to_imbe4400.bin",
    "tone_annex_t_ambe3600_to_ambe2450.bin",
    "tone_annex_t_ambe3600_to_imbe7200.bin",
    "valid_ambe2450_to_ambe3600.bin",
    "valid_ambe3600_to_ambe2450.bin",
    "valid_ambe3600_to_imbe7200.bin",
    "valid_imbe4400_to_imbe7200.bin",
    "valid_imbe7200_to_ambe3600.bin",
    "valid_imbe7200_to_imbe4400.bin",
    "zeros_ambe2450_to_ambe3600.bin",
    "zeros_ambe3600_to_ambe2450.bin",
    "zeros_ambe3600_to_imbe7200.bin",
    "zeros_imbe4400_to_imbe7200.bin",
    "zeros_imbe7200_to_ambe3600.bin",
    "zeros_imbe7200_to_imbe4400.bin",
];

const TOTAL_SEEDS: usize = 95;

const RATE_TAGS: [&str; 4] = ["imbe7200", "imbe4400", "ambe3600", "ambe2450"];

// ---------------------------------------------------------------------------
// Corpus access — every failure mode is a panic, never a silent skip.
// ---------------------------------------------------------------------------

fn corpus_root() -> PathBuf {
    let root = PathBuf::from(CORPUS);
    assert!(
        root.is_dir(),
        "fuzz seed corpus missing at {} — CI's `cargo fuzz run` would start from an \
         empty corpus and still pass; the seeds must be committed and checked out",
        root.display()
    );
    root
}

/// Sorted `.bin` seed names in one target directory. libFuzzer writes its own
/// discovered inputs into these directories named by content SHA-1 with no
/// extension; those are gitignored and are not part of the manifest.
fn seed_names(target: &str) -> Vec<String> {
    let dir = corpus_root().join(target);
    assert!(
        dir.is_dir(),
        "fuzz corpus target directory missing: {}",
        dir.display()
    );
    let entries =
        fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    let mut names: Vec<String> = entries
        .map(|e| {
            e.unwrap_or_else(|err| panic!("cannot read entry in {}: {err}", dir.display()))
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|n| n.ends_with(".bin"))
        .collect();
    assert!(
        !names.is_empty(),
        "fuzz corpus target directory {} holds no committed seeds",
        dir.display()
    );
    names.sort();
    names
}

fn seed(target: &str, name: &str) -> Vec<u8> {
    let path = corpus_root().join(target).join(name);
    fs::read(&path).unwrap_or_else(|e| panic!("cannot read seed {}: {e}", path.display()))
}

/// Split a seed into its selector byte and payload, panicking on an empty file.
fn split_one_selector(target: &str, name: &str) -> (u8, Vec<u8>) {
    let data = seed(target, name);
    let (&sel, payload) = data
        .split_first()
        .unwrap_or_else(|| panic!("{target}/{name} is empty — the target reads a selector byte"));
    (sel, payload.to_vec())
}

fn split_two_selectors(target: &str, name: &str) -> (u8, u8, Vec<u8>) {
    let data = seed(target, name);
    assert!(
        data.len() >= 2,
        "{target}/{name} is {} bytes — transcode reads two selector bytes",
        data.len()
    );
    (data[0], data[1], data[2..].to_vec())
}

// ---------------------------------------------------------------------------
// Filename ↔ rate
// ---------------------------------------------------------------------------

fn rate_from_tag(tag: &str) -> Rate {
    match tag {
        "imbe7200" => Rate::Imbe7200x4400,
        "imbe4400" => Rate::Imbe4400x4400,
        "ambe3600" => Rate::AmbePlus2_3600x2450,
        "ambe2450" => Rate::AmbePlus2_2450x2450,
        other => panic!("unknown rate tag {other:?}"),
    }
}

/// Rate tags appearing in a seed filename, in order.
fn tags_in(name: &str) -> Vec<&str> {
    name.trim_end_matches(".bin")
        .split('_')
        .filter(|t| RATE_TAGS.contains(t))
        .collect()
}

/// The single rate tag a `decode_bits` / `decode_soft` / `encode_pcm` seed
/// name carries.
fn sole_tag(name: &str) -> &str {
    let tags = tags_in(name);
    assert_eq!(
        tags.len(),
        1,
        "{name} must carry exactly one rate tag, found {tags:?}"
    );
    tags[0]
}

/// The `<from>`, `<to>` pair a `transcode` seed name carries.
fn pair_tags(name: &str) -> (&str, &str) {
    let tags = tags_in(name);
    assert_eq!(
        tags.len(),
        2,
        "{name} must be named <class>_<from>_to_<to>.bin, found tags {tags:?}"
    );
    (tags[0], tags[1])
}

// ---------------------------------------------------------------------------
// Bit / frame helpers
// ---------------------------------------------------------------------------

fn bits_msb(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 8);
    for &b in bytes {
        for k in (0..8).rev() {
            out.push((b >> k) & 1);
        }
    }
    out
}

fn pack_msb(bits: &[u8]) -> Vec<u8> {
    assert_eq!(bits.len() % 8, 0, "packing needs whole bytes");
    let mut out = vec![0u8; bits.len() / 8];
    for (i, &b) in bits.iter().enumerate() {
        out[i / 8] |= b << (7 - (i % 8));
    }
    out
}

/// Expand a hard frame to one LLR per channel bit: `1 → +mag`, `0 → −mag`.
fn expand_llrs(frame: &[u8], mag: i8) -> Vec<i8> {
    bits_msb(frame)
        .iter()
        .map(|&b| if b == 1 { mag } else { -mag })
        .collect()
}

fn dibits_of<const N: usize>(frame: &[u8]) -> [u8; N] {
    let bits = bits_msb(frame);
    assert_eq!(bits.len(), 2 * N, "frame is the wrong width for {N} dibits");
    let mut out = [0u8; N];
    for (i, d) in out.iter_mut().enumerate() {
        *d = (bits[2 * i] << 1) | bits[2 * i + 1];
    }
    out
}

fn imbe_info_from_fec(frame: &[u8]) -> [u16; 8] {
    imbe7200::frame::decode_frame(&dibits_of::<72>(frame)).info
}

fn imbe_info_from_no_fec(frame: &[u8]) -> [u16; 8] {
    let bits = bits_msb(frame);
    let mut out = [0u16; 8];
    let mut idx = 0usize;
    for (i, &w) in IMBE_INFO_WIDTHS.iter().enumerate() {
        let mut v = 0u16;
        for _ in 0..w {
            v = (v << 1) | u16::from(bits[idx]);
            idx += 1;
        }
        out[i] = v;
    }
    out
}

/// `b̂₀`, the 8-bit IMBE pitch index: `û₀[11..6]` over `û₇[2..1]`.
fn imbe_pitch_index(info: &[u16; 8]) -> u8 {
    ((((info[0] >> 6) & 0x3F) << 2) | ((info[7] >> 1) & 0x03)) as u8
}

fn ambe_info(rate: Rate, frame: &[u8]) -> [u16; 4] {
    match rate {
        Rate::AmbePlus2_3600x2450 => rate33::frame::decode_frame(&dibits_of::<36>(frame)).info,
        Rate::AmbePlus2_2450x2450 => rate33::frame::unpack_no_fec(frame),
        other => panic!("{other:?} is not a half-rate AMBE+2 wire format"),
    }
}

/// Does this wire frame carry ordinary voice — neither an Annex-T tone
/// signature nor an out-of-range pitch index?
fn is_voice_frame(rate: Rate, frame: &[u8]) -> bool {
    match rate {
        Rate::Imbe7200x4400 => imbe_pitch_index(&imbe_info_from_fec(frame)) <= IMBE_PITCH_INDEX_MAX,
        Rate::Imbe4400x4400 => {
            imbe_pitch_index(&imbe_info_from_no_fec(frame)) <= IMBE_PITCH_INDEX_MAX
        }
        Rate::AmbePlus2_3600x2450 | Rate::AmbePlus2_2450x2450 => {
            blip25_codec::tone::classify(&ambe_info(rate, frame)) == FrameKind::Voice
        }
        other => panic!("no voice test wired up for {other:?}"),
    }
}

/// `û₁` / `û₂` filler the committed tone seeds carry. The Table 20 layout
/// spreads four redundant copies of `I_D` across `û₀..û₃`; these seeds write
/// the signature and copy 4 only, so the two middle vectors are constant across
/// every tone frame in the corpus regardless of `(I_D, A_D)`.
/// [`tone_info_matches_the_canonical_builder`] checks that the resulting frames
/// still carry the same fields as `rate33::dequantize::encode_tone_frame_info`.
/// Annex-T info vectors for a tone frame, from the one canonical packer.
///
/// This used to be a second, local builder that wrote `I_D` into û₃ (copy 3)
/// only and filled û₁/û₂ with constants. That produces a frame no encoder
/// emits: Table 20 carries `I_D` **four** times, the decoder reads copy 0
/// (û₁(11..4)) and gates on the other three agreeing with it, so a
/// filler-û₁ frame is rejected outright. There is exactly one packer now.
fn tone_info(id: u8, amplitude: u8) -> [u16; 4] {
    rate33::dequantize::encode_tone_frame_info(id, amplitude)
}

/// Build the wire frame for an Annex-T `(I_D, A_D)` tone at a half-rate format.
fn tone_frame(rate: Rate, id: u8, amplitude: u8) -> Vec<u8> {
    let info = tone_info(id, amplitude);
    match rate {
        Rate::AmbePlus2_3600x2450 => {
            let dibits = rate33::frame::encode_frame(&info);
            let bits: Vec<u8> = dibits.iter().flat_map(|&d| [(d >> 1) & 1, d & 1]).collect();
            pack_msb(&bits)
        }
        Rate::AmbePlus2_2450x2450 => rate33::frame::pack_no_fec(&info).to_vec(),
        other => panic!("Annex-T tone frames are half-rate only, not {other:?}"),
    }
}

/// The four reference voice frames every other class quotes.
fn reference_frames(tag: &str) -> Vec<Vec<u8>> {
    let rate = rate_from_tag(tag);
    let (_, payload) = split_one_selector("decode_bits", &format!("valid_{tag}_4frames.bin"));
    let n = rate.fec_frame_bytes();
    assert_eq!(
        payload.len(),
        4 * n,
        "valid_{tag}_4frames.bin must hold exactly 4 frames"
    );
    payload.chunks(n).map(<[u8]>::to_vec).collect()
}

/// The ten frames of a concealment run: two good, six bad, two good.
fn conceal_frames(tag: &str) -> Vec<Vec<u8>> {
    let rate = rate_from_tag(tag);
    let name = if tag.starts_with("imbe") {
        format!("conceal_{tag}_badpitch_run.bin")
    } else {
        format!("conceal_{tag}_erasure_run.bin")
    };
    let (_, payload) = split_one_selector("decode_bits", &name);
    let n = rate.fec_frame_bytes();
    assert_eq!(payload.len(), 10 * n, "{name} must hold exactly 10 frames");
    payload.chunks(n).map(<[u8]>::to_vec).collect()
}

fn pcm_samples(payload: &[u8]) -> Vec<i16> {
    assert_eq!(
        payload.len() % 2,
        0,
        "PCM payload must be a whole number of big-endian i16 samples"
    );
    payload
        .chunks_exact(2)
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect()
}

fn peak(pcm: &[i16]) -> i32 {
    pcm.iter()
        .map(|&s| i32::from(s).abs())
        .max()
        .expect("decoded frame is never empty")
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

// ===========================================================================
// 1. Presence
// ===========================================================================

#[test]
fn corpus_is_present_and_complete() {
    let root = corpus_root();
    for (target, manifest) in [
        ("decode_bits", DECODE_BITS_SEEDS.as_slice()),
        ("decode_soft", DECODE_SOFT_SEEDS.as_slice()),
        ("encode_pcm", ENCODE_PCM_SEEDS.as_slice()),
        ("transcode", TRANSCODE_SEEDS.as_slice()),
    ] {
        let found = seed_names(target);
        let want: Vec<String> = manifest.iter().map(|s| (*s).to_string()).collect();
        assert_eq!(
            found,
            want,
            "fuzz/corpus/{target} does not match the committed manifest \
             (missing: {:?}, unexpected: {:?})",
            want.iter()
                .filter(|n| !found.contains(n))
                .collect::<Vec<_>>(),
            found
                .iter()
                .filter(|n| !want.contains(n))
                .collect::<Vec<_>>(),
        );
    }
    let total = DECODE_BITS_SEEDS.len()
        + DECODE_SOFT_SEEDS.len()
        + ENCODE_PCM_SEEDS.len()
        + TRANSCODE_SEEDS.len();
    assert_eq!(total, TOTAL_SEEDS, "manifest total drifted");
    assert!(
        root.join("README.md").is_file(),
        "fuzz/corpus/README.md documents the selector layout and must ship with the seeds"
    );
    for name in DECODE_BITS_SEEDS
        .iter()
        .chain(DECODE_SOFT_SEEDS.iter())
        .chain(ENCODE_PCM_SEEDS.iter())
        .chain(TRANSCODE_SEEDS.iter())
    {
        let target = if DECODE_BITS_SEEDS.contains(name) {
            "decode_bits"
        } else if DECODE_SOFT_SEEDS.contains(name) {
            "decode_soft"
        } else if ENCODE_PCM_SEEDS.contains(name) {
            "encode_pcm"
        } else {
            "transcode"
        };
        assert!(
            !seed(target, name).is_empty(),
            "{target}/{name} is zero bytes — the target would return before touching the API"
        );
    }
}

/// The selector byte only means what the filename says while the target's
/// `RATES` array keeps its order. Re-read the four sources and prove it.
#[test]
fn fuzz_target_rate_tables_match() {
    for (target, table) in [
        ("decode_bits", RATES_DECODE_BITS),
        ("decode_soft", RATES_DECODE_SOFT),
        ("encode_pcm", RATES_ENCODE_PCM),
        ("transcode", RATES_TRANSCODE),
    ] {
        let path = PathBuf::from(TARGET_SRC).join(format!("{target}.rs"));
        let src = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read fuzz target {}: {e}", path.display()));
        let head = "const RATES: [Rate; 4] = [";
        let start = src
            .find(head)
            .unwrap_or_else(|| panic!("{target}.rs has no `{head}` declaration"));
        let body = &src[start + head.len()..];
        let end = body
            .find("];")
            .unwrap_or_else(|| panic!("{target}.rs RATES declaration is unterminated"));
        let body = &body[..end];
        let declared: Vec<&str> = body
            .match_indices("Rate::")
            .map(|(i, _)| {
                body[i + "Rate::".len()..]
                    .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .next()
                    .expect("variant name follows `Rate::`")
            })
            .collect();
        let expected: Vec<String> = table.iter().map(|r| format!("{r:?}")).collect();
        assert_eq!(
            declared, expected,
            "fuzz_targets/{target}.rs RATES order drifted from this test's table — \
             every seed's selector byte now resolves to a different rate"
        );
    }
}

// ===========================================================================
// 2. Framing + API drive, per target
// ===========================================================================

#[test]
fn decode_bits_seeds_drive_their_target() {
    for name in DECODE_BITS_SEEDS {
        let (sel, payload) = split_one_selector("decode_bits", name);
        let rate = RATES_DECODE_BITS[sel as usize % RATES_DECODE_BITS.len()];
        assert_eq!(
            rate,
            rate_from_tag(sole_tag(name)),
            "{name}: selector byte {sel} selects {rate:?}, filename says {}",
            sole_tag(name)
        );

        let n = rate.fec_frame_bytes();
        assert_eq!(
            payload.len() % n,
            0,
            "{name}: {} payload bytes is not a whole number of {n}-byte {rate:?} frames",
            payload.len()
        );
        let frames = payload.len() / n;
        assert!(frames >= 1, "{name}: payload holds no frames");

        let mut v = Vocoder::new(rate);
        let mut decoded = 0usize;
        for (i, chunk) in payload.chunks(n).enumerate() {
            let pcm = v
                .decode_bits(chunk)
                .unwrap_or_else(|e| panic!("{name}: frame {i} rejected: {e}"));
            assert_eq!(pcm.len(), FRAME_SAMPLES, "{name}: frame {i} wrong length");
            decoded += 1;
        }
        assert_eq!(decoded, frames, "{name}: not every frame was decoded");

        // The target resets mid-stream and re-feeds the whole payload.
        v.reset();
        match v.decode_bits(&payload) {
            Ok(pcm) => {
                assert_eq!(
                    frames, 1,
                    "{name}: multi-frame payload accepted as one frame"
                );
                assert_eq!(pcm.len(), FRAME_SAMPLES);
            }
            Err(VocoderError::WrongBitsLength { expected, got }) => {
                assert!(frames > 1, "{name}: single-frame payload rejected");
                assert_eq!(expected, n, "{name}");
                assert_eq!(got, payload.len(), "{name}");
            }
            Err(e) => panic!("{name}: unexpected error from whole-payload decode: {e}"),
        }
    }
}

#[test]
fn decode_soft_seeds_drive_their_target() {
    for name in DECODE_SOFT_SEEDS {
        let (sel, payload) = split_one_selector("decode_soft", name);
        let rate = RATES_DECODE_SOFT[sel as usize % RATES_DECODE_SOFT.len()];
        assert_eq!(
            rate,
            rate_from_tag(sole_tag(name)),
            "{name}: selector byte {sel} selects {rate:?} under the decode_soft table, \
             filename says {}",
            sole_tag(name)
        );

        let llrs: Vec<i8> = payload.iter().map(|&b| b as i8).collect();
        let mut v = Vocoder::new(rate);
        match rate.soft_frame_bits() {
            Some(n) => {
                assert_eq!(
                    llrs.len() % n,
                    0,
                    "{name}: {} LLRs is not a whole number of {n}-LLR {rate:?} frames",
                    llrs.len()
                );
                let frames = llrs.len() / n;
                assert!(frames >= 1, "{name}: payload holds no frames");
                let mut decoded = 0usize;
                for (i, chunk) in llrs.chunks(n).enumerate() {
                    let pcm = v
                        .decode_soft(chunk)
                        .unwrap_or_else(|e| panic!("{name}: frame {i} rejected: {e}"));
                    assert_eq!(pcm.len(), FRAME_SAMPLES, "{name}: frame {i} wrong length");
                    decoded += 1;
                }
                assert_eq!(decoded, frames, "{name}: not every frame was decoded");
            }
            None => {
                let err = match v.decode_soft(&llrs) {
                    Ok(_) => panic!("{name}: a no-FEC rate must reject soft decode"),
                    Err(e) => e,
                };
                assert!(
                    matches!(err, VocoderError::SoftUnsupported { rate: r } if r == rate),
                    "{name}: expected SoftUnsupported for {rate:?}, got {err}"
                );
            }
        }
    }
}

#[test]
fn encode_pcm_seeds_drive_their_target() {
    for name in ENCODE_PCM_SEEDS {
        let (sel, payload) = split_one_selector("encode_pcm", name);
        let rate = RATES_ENCODE_PCM[sel as usize % RATES_ENCODE_PCM.len()];
        assert_eq!(
            rate,
            rate_from_tag(sole_tag(name)),
            "{name}: selector byte {sel} selects {rate:?}, filename says {}",
            sole_tag(name)
        );

        // Two documented exceptions to whole-frame framing: `odd_length_*` is
        // one frame plus a 90-sample tail (the short-chunk rejection path) and
        // `selector_only_*` carries no payload at all.
        if name.starts_with("odd_length_") {
            assert_eq!(
                payload.len(),
                PCM_FRAME_BYTES + 90 * 2,
                "{name}: must be one whole frame plus a 90-sample tail"
            );
        } else if name.starts_with("selector_only_") {
            assert!(payload.is_empty(), "{name}: must carry no payload");
        } else {
            assert_eq!(
                payload.len() % PCM_FRAME_BYTES,
                0,
                "{name}: {} payload bytes is not a whole number of 160-sample frames",
                payload.len()
            );
        }

        let pcm = pcm_samples(&payload);
        let mut v = Vocoder::new(rate);
        let n = v.frame_samples();
        let mut whole = 0usize;
        for (i, frame) in pcm.chunks(n).enumerate() {
            match v.encode_pcm(frame) {
                Ok(bits) => {
                    assert_eq!(frame.len(), n, "{name}: short frame {i} was accepted");
                    assert_eq!(
                        bits.len(),
                        rate.fec_frame_bytes(),
                        "{name}: frame {i} wrong wire length"
                    );
                    whole += 1;
                }
                Err(VocoderError::WrongPcmLength { expected, got }) => {
                    assert_ne!(frame.len(), n, "{name}: whole frame {i} was rejected");
                    assert_eq!(expected, n, "{name}");
                    assert_eq!(got, frame.len(), "{name}");
                }
                Err(e) => panic!("{name}: unexpected error on frame {i}: {e}"),
            }
        }
        assert_eq!(
            whole,
            pcm.len() / n,
            "{name}: not every whole frame was encoded"
        );

        // The target then calls encode_pcm on the whole buffer and flushes.
        if pcm.len() != n {
            let err = v
                .encode_pcm(&pcm)
                .expect_err("whole-buffer encode must be rejected unless it is one frame");
            assert!(
                matches!(err, VocoderError::WrongPcmLength { expected, got }
                         if expected == n && got == pcm.len()),
                "{name}: expected WrongPcmLength, got {err}"
            );
        }
        for bits in v.flush_encode() {
            assert_eq!(
                bits.len(),
                rate.fec_frame_bytes(),
                "{name}: flushed frame wrong wire length"
            );
        }
    }
}

#[test]
fn transcode_seeds_drive_their_target() {
    for name in TRANSCODE_SEEDS {
        let (sel_from, sel_to, payload) = split_two_selectors("transcode", name);
        let from = RATES_TRANSCODE[sel_from as usize % RATES_TRANSCODE.len()];
        let to = RATES_TRANSCODE[sel_to as usize % RATES_TRANSCODE.len()];
        let (tag_from, tag_to) = pair_tags(name);
        assert_eq!(
            from,
            rate_from_tag(tag_from),
            "{name}: first selector byte {sel_from} selects {from:?}, filename says {tag_from}"
        );
        assert_eq!(
            to,
            rate_from_tag(tag_to),
            "{name}: second selector byte {sel_to} selects {to:?}, filename says {tag_to}"
        );

        let n = from.fec_frame_bytes();
        assert_eq!(
            payload.len() % n,
            0,
            "{name}: {} payload bytes is not a whole number of {n}-byte {from:?} frames",
            payload.len()
        );
        let frames = payload.len() / n;
        assert!(frames >= 1, "{name}: payload holds no frames");

        let mut t = Transcoder::new(from, to).unwrap_or_else(|e| {
            panic!(
                "{name}: {from:?} -> {to:?} is not a supported direction, so the seed \
                    would return before touching the API: {e}"
            )
        });
        let mut converted = 0usize;
        for (i, chunk) in payload.chunks(n).enumerate() {
            let out = t
                .transcode(chunk)
                .unwrap_or_else(|e| panic!("{name}: frame {i} rejected: {e}"));
            assert_eq!(
                out.len(),
                to.fec_frame_bytes(),
                "{name}: frame {i} wrong output length"
            );
            converted += 1;
        }
        assert_eq!(converted, frames, "{name}: not every frame was transcoded");

        t.reset();
        match t.transcode(&payload) {
            Ok(out) => {
                assert_eq!(
                    frames, 1,
                    "{name}: multi-frame payload accepted as one frame"
                );
                assert_eq!(out.len(), to.fec_frame_bytes());
            }
            Err(VocoderError::WrongBitsLength { expected, got }) => {
                assert!(frames > 1, "{name}: single-frame payload rejected");
                assert_eq!(expected, n, "{name}");
                assert_eq!(got, payload.len(), "{name}");
            }
            Err(e) => panic!("{name}: unexpected error from whole-payload transcode: {e}"),
        }
    }
}

// ===========================================================================
// 3. Content — decode_bits
// ===========================================================================

/// Reference frames are real voice frames and stay quoted verbatim by the
/// single-frame seed.
#[test]
fn decode_bits_valid_seeds_are_reference_voice_frames() {
    for tag in RATE_TAGS {
        let rate = rate_from_tag(tag);
        let refs = reference_frames(tag);
        let (_, one) = split_one_selector("decode_bits", &format!("valid_{tag}_1frame.bin"));
        assert_eq!(
            one, refs[0],
            "valid_{tag}_1frame.bin must be reference frame 0 of valid_{tag}_4frames.bin"
        );

        let mut v = Vocoder::new(rate);
        for (i, f) in refs.iter().enumerate() {
            assert!(
                is_voice_frame(rate, f),
                "valid_{tag}_4frames.bin frame {i} is not a voice frame"
            );
            let pcm = v
                .decode_bits(f)
                .unwrap_or_else(|e| panic!("valid_{tag}_4frames.bin frame {i}: {e}"));
            assert!(
                peak(&pcm) >= AUDIBLE_PEAK,
                "valid_{tag}_4frames.bin frame {i} decodes to near-silence (peak {}) — \
                 a reference voice frame must carry audible speech",
                peak(&pcm)
            );
        }
    }
}

/// Rewrite every committed seed whose payload contains Annex-T tone frames,
/// from the canonical builder. Ignored by default — run it only when the tone
/// wire format itself changes, then re-run the suite:
///
/// ```text
/// cargo test -p blip25-mbe --test fuzz_corpus -- --ignored regenerate_tone_seeds
/// ```
///
/// Every non-tone byte (selectors, trailing voice frames) is preserved from
/// the file on disk, so this only ever moves the bytes it is meant to.
#[test]
#[ignore = "regenerates committed corpus seeds; run explicitly"]
fn regenerate_tone_seeds() {
    let write = |target: &str, name: &str, bytes: Vec<u8>| {
        let path = corpus_root().join(target).join(name);
        let old = fs::read(&path).expect("seed exists");
        assert_eq!(
            old.len(),
            bytes.len(),
            "{name}: regenerated length {} != committed {}",
            bytes.len(),
            old.len()
        );
        if old != bytes {
            fs::write(&path, &bytes).expect("write seed");
            eprintln!("regenerated {target}/{name}");
        }
    };

    // decode_bits: [selector] + 5 tone frames + the 2 trailing voice frames.
    const BITS_TONES: [(u8, u8); 5] = [(5, 64), (5, 90), (5, 90), (40, 51), (255, 0)];
    for tag in ["ambe3600", "ambe2450"] {
        let rate = rate_from_tag(tag);
        let name = format!("tone_annex_t_{tag}.bin");
        let old = seed("decode_bits", &name);
        let n = rate.fec_frame_bytes();
        let mut out = vec![old[0]];
        for &(id, amplitude) in &BITS_TONES {
            out.extend_from_slice(&tone_frame(rate, id, amplitude));
        }
        out.extend_from_slice(&old[1 + BITS_TONES.len() * n..]);
        write("decode_bits", &name, out);
    }

    // decode_soft: [selector] + the +/-100 LLR expansion of 3 tone frames.
    const SOFT_TONES: [(u8, u8); 3] = [(5, 90), (5, 90), (40, 51)];
    {
        let rate = Rate::AmbePlus2_3600x2450;
        let old = seed("decode_soft", "tone_llr_ambe3600.bin");
        let mut out = vec![old[0]];
        out.extend(
            SOFT_TONES
                .iter()
                .flat_map(|&(id, a)| expand_llrs(&tone_frame(rate, id, a), 100))
                .map(|v| v as u8),
        );
        write("decode_soft", "tone_llr_ambe3600.bin", out);
    }

    // transcode: [selector, selector] + 3 tone frames at the source rate.
    for name in [
        "tone_annex_t_ambe3600_to_ambe2450.bin",
        "tone_annex_t_ambe3600_to_imbe7200.bin",
    ] {
        let (tag_from, _) = pair_tags(name);
        let rate = rate_from_tag(tag_from);
        let old = seed("transcode", name);
        let mut out = vec![old[0], old[1]];
        for &(id, a) in &SOFT_TONES {
            out.extend_from_slice(&tone_frame(rate, id, a));
        }
        write("transcode", name, out);
    }
}

/// The tone frames this file rebuilds must be **byte-identical** to
/// `rate33::dequantize::encode_tone_frame_info`, the crate's canonical builder.
///
/// The old form of this test compared `û₀` only, which is why a local builder
/// that never wrote `I_D` copies 0..2 passed it for as long as it existed.
/// Comparing all four vectors is what makes "one packer" enforceable.
#[test]
fn tone_info_matches_the_canonical_builder() {
    for (id, amplitude) in [(5u8, 64u8), (5, 90), (40, 51), (255, 0)] {
        let mine = tone_info(id, amplitude);
        let canonical = rate33::dequantize::encode_tone_frame_info(id, amplitude);
        assert_eq!(
            mine, canonical,
            "û₀..û₃ must match the canonical Annex-T builder for (I_D {id}, A_D {amplitude})"
        );
        assert_eq!(mine[3] & 0x0F, 0, "û₃ must keep the Table 20 zero trailer");
        // All four I_D copies must agree — the decoder gates on it.
        let u = mine;
        let copies = [
            ((u[1] >> 4) & 0xFF) as u8,
            (((u[1] & 0x0F) << 4) | ((u[2] >> 7) & 0x0F)) as u8,
            (((u[2] & 0x7F) << 1) | ((u[3] >> 13) & 1)) as u8,
            ((u[3] >> 5) & 0xFF) as u8,
        ];
        assert_eq!(copies, [id; 4], "all four I_D copies must carry I_D {id}");
        for info in [mine, canonical] {
            assert_eq!(
                blip25_codec::tone::classify(&info),
                FrameKind::Tone,
                "(I_D {id}, A_D {amplitude}) must classify as a tone, not an erasure"
            );
            let fields = blip25_codec::tone::parse_tone_frame(&info)
                .unwrap_or_else(|| panic!("(I_D {id}, A_D {amplitude}) has no tone fields"));
            assert_eq!((fields.id, fields.amplitude), (id, amplitude));
        }
    }
}

/// Annex-T tone frames, rebuilt from `(I_D, A_D)` and compared byte-for-byte.
/// This is the seed's whole reason to exist: a fuzzer will not synthesise the
/// `û₀(11..6) == 0x3F` signature by chance, and it must not be read as an
/// erasure.
#[test]
fn decode_bits_tone_seeds_carry_the_annex_t_signature() {
    const TONES: [(u8, u8); 5] = [(5, 64), (5, 90), (5, 90), (40, 51), (255, 0)];
    for tag in ["ambe3600", "ambe2450"] {
        let rate = rate_from_tag(tag);
        let refs = reference_frames(tag);
        let (_, payload) = split_one_selector("decode_bits", &format!("tone_annex_t_{tag}.bin"));
        let n = rate.fec_frame_bytes();
        let frames: Vec<&[u8]> = payload.chunks(n).collect();
        assert_eq!(
            frames.len(),
            TONES.len() + 2,
            "tone_annex_t_{tag}.bin must be {} tone frames plus two good voice frames",
            TONES.len()
        );

        let mut v = Vocoder::new(rate);
        for (i, &(id, amplitude)) in TONES.iter().enumerate() {
            let want = tone_frame(rate, id, amplitude);
            assert_eq!(
                frames[i],
                want.as_slice(),
                "tone_annex_t_{tag}.bin frame {i} is not the Annex-T frame for \
                 (I_D {id}, A_D {amplitude})"
            );
            let info = ambe_info(rate, frames[i]);
            assert_eq!(
                blip25_codec::tone::classify(&info),
                FrameKind::Tone,
                "tone_annex_t_{tag}.bin frame {i} does not classify as a tone"
            );
            let fields = blip25_codec::tone::parse_tone_frame(&info).unwrap_or_else(|| {
                panic!("tone_annex_t_{tag}.bin frame {i} has no parseable tone fields")
            });
            assert_eq!((fields.id, fields.amplitude), (id, amplitude));
            let pcm = v
                .decode_bits(frames[i])
                .unwrap_or_else(|e| panic!("tone_annex_t_{tag}.bin frame {i}: {e}"));
            assert_eq!(pcm.len(), FRAME_SAMPLES);
        }
        // Good frames on the far side, so the tone → voice transition is covered.
        assert_eq!(
            frames[TONES.len()],
            refs[0].as_slice(),
            "tone_annex_t_{tag}.bin frame {} must be reference frame 0",
            TONES.len()
        );
        assert_eq!(
            frames[TONES.len() + 1],
            refs[1].as_slice(),
            "tone_annex_t_{tag}.bin frame {} must be reference frame 1",
            TONES.len() + 1
        );
    }
}

/// One clean frame followed by the same frame with 1, 3 and 7 channel bits
/// flipped at stride 13 — re-derived here and compared byte-for-byte, so a
/// zeroed or regenerated seed cannot pass.
#[test]
fn decode_bits_fec_error_seeds_are_derived_bit_flips() {
    const FLIP_START: usize = 3;
    const FLIP_STRIDE: usize = 13;
    const FLIP_COUNTS: [usize; 4] = [0, 1, 3, 7];

    for tag in RATE_TAGS {
        let rate = rate_from_tag(tag);
        let name = format!("fec_errors_{tag}.bin");
        let (_, payload) = split_one_selector("decode_bits", &name);
        let n = rate.fec_frame_bytes();
        let frames: Vec<&[u8]> = payload.chunks(n).collect();
        assert_eq!(
            frames.len(),
            FLIP_COUNTS.len(),
            "{name} must hold one frame per flip count"
        );

        let clean = frames[0];
        assert!(
            is_voice_frame(rate, clean),
            "{name} frame 0 must be a clean voice frame"
        );
        let nbits = n * 8;
        let mut v = Vocoder::new(rate);
        for (i, &count) in FLIP_COUNTS.iter().enumerate() {
            let mut bits = bits_msb(clean);
            let mut flipped = BTreeSet::new();
            for k in 0..count {
                let pos = (FLIP_START + FLIP_STRIDE * k) % nbits;
                bits[pos] ^= 1;
                assert!(
                    flipped.insert(pos),
                    "{name}: flip stride collides at bit {pos}"
                );
            }
            let want = pack_msb(&bits);
            assert_eq!(
                frames[i],
                want.as_slice(),
                "{name} frame {i} is not frame 0 with {count} channel bits flipped at \
                 positions {flipped:?}"
            );
            let differing = bits_msb(clean)
                .iter()
                .zip(bits_msb(frames[i]))
                .filter(|(a, b)| **a != *b)
                .count();
            assert_eq!(
                differing, count,
                "{name} frame {i} carries the wrong error count"
            );
            let pcm = v
                .decode_bits(frames[i])
                .unwrap_or_else(|e| panic!("{name} frame {i}: {e}"));
            assert_eq!(pcm.len(), FRAME_SAMPLES);
        }
    }
}

#[test]
fn decode_bits_uniform_seeds_are_uniform() {
    for tag in RATE_TAGS {
        let rate = rate_from_tag(tag);
        for (class, byte) in [("zeros", 0x00u8), ("ones", 0xFFu8)] {
            let name = format!("{class}_{tag}.bin");
            let (_, payload) = split_one_selector("decode_bits", &name);
            assert_eq!(
                payload.len(),
                3 * rate.fec_frame_bytes(),
                "{name} must hold exactly 3 frames"
            );
            assert!(
                payload.iter().all(|&b| b == byte),
                "{name} must be entirely {byte:#04x}; found {:?}",
                payload
                    .iter()
                    .copied()
                    .collect::<BTreeSet<u8>>()
                    .iter()
                    .take(8)
                    .collect::<Vec<_>>()
            );
        }
    }
}

/// A concealment run must actually walk repeat → mute, not merely decode.
///
/// The half-rate path reports this through `DecodeStats::disposition`. Both
/// IMBE rates hard-code `FrameDisposition::Use` in `Vocoder::decode_via_codec`
/// (the engine conceals internally and the state is not surfaced), so their run
/// is pinned on the wire side — an out-of-range pitch index — plus the audible
/// signature of the mute gate: comfort noise on `[−5, 5]`.
#[test]
fn decode_bits_conceal_seeds_walk_repeat_then_mute() {
    const AMBE_ERASURE_B0: [u16; 6] = [120, 121, 122, 123, 120, 121];

    for tag in RATE_TAGS {
        let rate = rate_from_tag(tag);
        let refs = reference_frames(tag);
        let frames = conceal_frames(tag);
        for (slot, r) in [(0usize, 0usize), (1, 1), (8, 2), (9, 3)] {
            assert_eq!(
                frames[slot], refs[r],
                "conceal run for {tag}: frame {slot} must be reference frame {r}, so recovery \
                 on both sides of the run is exercised"
            );
        }

        for (k, f) in frames[2..8].iter().enumerate() {
            match rate {
                Rate::Imbe7200x4400 | Rate::Imbe4400x4400 => {
                    let info = if rate == Rate::Imbe7200x4400 {
                        imbe_info_from_fec(f)
                    } else {
                        imbe_info_from_no_fec(f)
                    };
                    assert!(
                        imbe_pitch_index(&info) > IMBE_PITCH_INDEX_MAX,
                        "conceal run for {tag}: frame {} has a valid pitch index and would \
                         not trip concealment",
                        k + 2
                    );
                }
                Rate::AmbePlus2_3600x2450 | Rate::AmbePlus2_2450x2450 => {
                    let b = rate33::priority::deprioritize(&ambe_info(rate, f));
                    assert_eq!(
                        b[0],
                        AMBE_ERASURE_B0[k],
                        "conceal run for {tag}: frame {} must carry b̂₀ in the BABA-1 erasure \
                         range [120, 123]",
                        k + 2
                    );
                }
                other => panic!("no erasure test wired up for {other:?}"),
            }
        }

        let mut v = Vocoder::new(rate);
        let mut dispositions = Vec::with_capacity(frames.len());
        let mut peaks = Vec::with_capacity(frames.len());
        for (i, f) in frames.iter().enumerate() {
            let pcm = v
                .decode_bits(f)
                .unwrap_or_else(|e| panic!("conceal run for {tag}: frame {i}: {e}"));
            let stats =
                v.last_stats().decode.clone().unwrap_or_else(|| {
                    panic!("conceal run for {tag}: frame {i} left no decode stats")
                });
            dispositions.push(stats.disposition);
            peaks.push(peak(&pcm));
        }
        assert_eq!(dispositions.len(), frames.len());
        assert_eq!(peaks.len(), frames.len());

        match rate {
            Rate::AmbePlus2_3600x2450 | Rate::AmbePlus2_2450x2450 => {
                use FrameDisposition::{Mute, Repeat, Use};
                assert_eq!(
                    dispositions,
                    vec![Use, Use, Repeat, Repeat, Repeat, Mute, Mute, Mute, Use, Use],
                    "conceal run for {tag}: the run must walk Use -> Repeat -> Mute -> Use"
                );
            }
            Rate::Imbe7200x4400 | Rate::Imbe4400x4400 => {
                assert!(
                    dispositions.iter().all(|&d| d == FrameDisposition::Use),
                    "the IMBE decode path does not surface a disposition yet; update this \
                     check if it starts to"
                );
            }
            other => panic!("no disposition expectation wired up for {other:?}"),
        }

        // Audible signature, identical on all four rates: the mute gate replaces
        // the last three frames of the run with comfort noise on [-5, 5], and
        // every frame outside the run stays loud.
        for i in [0usize, 1, 8, 9] {
            assert!(
                peaks[i] >= AUDIBLE_PEAK,
                "conceal run for {tag}: frame {i} is outside the run but decodes to \
                 near-silence (peak {})",
                peaks[i]
            );
        }
        for i in 5..8 {
            assert!(
                peaks[i] <= MUTE_PEAK,
                "conceal run for {tag}: frame {i} must be muted to comfort noise, peak is {}",
                peaks[i]
            );
        }
        assert!(
            peaks[2..5].iter().any(|&p| p >= AUDIBLE_PEAK),
            "conceal run for {tag}: the repeat phase must still emit the previous frame's \
             audio, not silence"
        );
    }
}

// ===========================================================================
// 3. Content — decode_soft
// ===========================================================================

#[test]
fn decode_soft_llr_extremes_are_saturated() {
    for (class, want) in [
        ("zero_llr", 0i8),
        ("min_llr", i8::MIN),
        ("max_llr", i8::MAX),
    ] {
        for tag in ["imbe7200", "ambe3600"] {
            let name = format!("{class}_{tag}.bin");
            let (_, payload) = split_one_selector("decode_soft", &name);
            let rate = rate_from_tag(tag);
            let n = rate
                .soft_frame_bits()
                .unwrap_or_else(|| panic!("{name}: {rate:?} has no soft frame width"));
            assert_eq!(payload.len(), 2 * n, "{name} must hold exactly 2 frames");
            let llrs: Vec<i8> = payload.iter().map(|&b| b as i8).collect();
            assert!(
                llrs.iter().all(|&v| v == want),
                "{name} must be entirely {want}; found {:?}",
                llrs.iter()
                    .copied()
                    .collect::<BTreeSet<i8>>()
                    .iter()
                    .take(8)
                    .collect::<Vec<_>>()
            );
        }
    }
}

/// `valid_strong_*` and `valid_weak_*` are the same reference frames at
/// opposite ends of the confidence scale, so they must decode identically.
#[test]
fn decode_soft_valid_seeds_are_reference_frame_expansions() {
    for tag in ["imbe7200", "ambe3600"] {
        let rate = rate_from_tag(tag);
        let refs = reference_frames(tag);
        for (class, mag, count) in [("valid_strong", 100i8, 3usize), ("valid_weak", 1, 2)] {
            let name = format!("{class}_{tag}.bin");
            let (_, payload) = split_one_selector("decode_soft", &name);
            let want: Vec<i8> = (0..count)
                .flat_map(|k| expand_llrs(&refs[k], mag))
                .collect();
            let got: Vec<i8> = payload.iter().map(|&b| b as i8).collect();
            assert_eq!(
                got, want,
                "{name} must be the +/-{mag} LLR expansion of reference frames 0..{count}"
            );
        }

        // Same signs, magnitude 1 vs 100: the decoded audio must not move.
        let strong: Vec<i8> = split_one_selector("decode_soft", &format!("valid_strong_{tag}.bin"))
            .1
            .iter()
            .map(|&b| b as i8)
            .collect();
        let weak: Vec<i8> = split_one_selector("decode_soft", &format!("valid_weak_{tag}.bin"))
            .1
            .iter()
            .map(|&b| b as i8)
            .collect();
        let n = rate.soft_frame_bits().expect("FEC-bearing rate");
        let mut a = Vocoder::new(rate);
        let mut b = Vocoder::new(rate);
        for k in 0..weak.len() / n {
            let pa = a
                .decode_soft(&strong[k * n..(k + 1) * n])
                .unwrap_or_else(|e| panic!("valid_strong_{tag} frame {k}: {e}"));
            let pb = b
                .decode_soft(&weak[k * n..(k + 1) * n])
                .unwrap_or_else(|e| panic!("valid_weak_{tag} frame {k}: {e}"));
            assert_eq!(
                pa, pb,
                "valid_weak_{tag} frame {k} must decode to the same audio as valid_strong_{tag}"
            );
            assert!(
                peak(&pa) >= AUDIBLE_PEAK,
                "valid_strong_{tag} frame {k} decodes to near-silence (peak {})",
                peak(&pa)
            );
        }
    }
}

/// The soft erasure run is the hard concealment run expanded to LLRs, and must
/// reach the same repeat → mute states.
#[test]
fn decode_soft_erasure_seed_walks_repeat_then_mute() {
    let rate = Rate::AmbePlus2_3600x2450;
    let frames = conceal_frames("ambe3600");
    // Good frame, the six erasures, good frame.
    let order = [0usize, 2, 3, 4, 5, 6, 7, 1];
    let want: Vec<i8> = order
        .iter()
        .flat_map(|&k| expand_llrs(&frames[k], 100))
        .collect();
    let (_, payload) = split_one_selector("decode_soft", "erasure_llr_ambe3600.bin");
    let got: Vec<i8> = payload.iter().map(|&b| b as i8).collect();
    assert_eq!(
        got, want,
        "erasure_llr_ambe3600.bin must be the +/-100 expansion of conceal run frames {order:?}"
    );

    let n = rate.soft_frame_bits().expect("FEC-bearing rate");
    let mut v = Vocoder::new(rate);
    let mut dispositions = Vec::with_capacity(order.len());
    for (i, chunk) in got.chunks(n).enumerate() {
        v.decode_soft(chunk)
            .unwrap_or_else(|e| panic!("erasure_llr_ambe3600.bin frame {i}: {e}"));
        let stats =
            v.last_stats().decode.clone().unwrap_or_else(|| {
                panic!("erasure_llr_ambe3600.bin frame {i} left no decode stats")
            });
        dispositions.push(stats.disposition);
    }
    assert_eq!(dispositions.len(), order.len());
    use FrameDisposition::{Mute, Repeat, Use};
    assert_eq!(
        dispositions,
        vec![Use, Repeat, Repeat, Repeat, Mute, Mute, Mute, Use],
        "erasure_llr_ambe3600.bin must walk Use -> Repeat -> Mute -> Use"
    );
}

#[test]
fn decode_soft_tone_seed_carries_the_annex_t_signature() {
    const TONES: [(u8, u8); 3] = [(5, 90), (5, 90), (40, 51)];
    let rate = Rate::AmbePlus2_3600x2450;
    let want: Vec<i8> = TONES
        .iter()
        .flat_map(|&(id, amplitude)| expand_llrs(&tone_frame(rate, id, amplitude), 100))
        .collect();
    let (_, payload) = split_one_selector("decode_soft", "tone_llr_ambe3600.bin");
    let got: Vec<i8> = payload.iter().map(|&b| b as i8).collect();
    assert_eq!(
        got, want,
        "tone_llr_ambe3600.bin must be the +/-100 expansion of the Annex-T frames {TONES:?}"
    );

    let n = rate.soft_frame_bits().expect("FEC-bearing rate");
    let mut v = Vocoder::new(rate);
    let mut kinds = Vec::with_capacity(TONES.len());
    for (i, chunk) in got.chunks(n).enumerate() {
        let soft: &[i8; 72] = chunk.try_into().expect("half-rate soft frame is 72 LLRs");
        kinds.push(blip25_codec::tone::classify(
            &rate33::frame::decode_frame_soft(soft).info,
        ));
        let pcm = v
            .decode_soft(chunk)
            .unwrap_or_else(|e| panic!("tone_llr_ambe3600.bin frame {i}: {e}"));
        assert!(
            peak(&pcm) >= AUDIBLE_PEAK,
            "tone_llr_ambe3600.bin frame {i} must synthesize an audible tone, peak is {}",
            peak(&pcm)
        );
    }
    assert_eq!(kinds.len(), TONES.len());
    assert!(
        kinds.iter().all(|&k| k == FrameKind::Tone),
        "tone_llr_ambe3600.bin must soft-decode to tone frames, got {kinds:?}"
    );
}

/// `rescuable_*` is reference frame 3 with three sign flips carried at
/// magnitude 1; `unrescuable_*` is a full-confidence corruption whose FEC error
/// counts saturate on every coded vector. See the module header for why the
/// rescue itself is not asserted.
#[test]
fn decode_soft_rescue_seeds_are_derived_corruptions() {
    const WEAK_FLIPS: [usize; 3] = [3, 17, 41];

    for tag in ["imbe7200", "ambe3600"] {
        let rate = rate_from_tag(tag);
        let n = rate.soft_frame_bits().expect("FEC-bearing rate");
        let clean_frame = reference_frames(tag).swap_remove(3);
        let clean = expand_llrs(&clean_frame, 100);
        assert_eq!(clean.len(), n);

        let mut want = clean.clone();
        for &p in &WEAK_FLIPS {
            want[p] = if clean[p] > 0 { -1 } else { 1 };
        }
        let (_, payload) = split_one_selector("decode_soft", &format!("rescuable_{tag}.bin"));
        let resc: Vec<i8> = payload.iter().map(|&b| b as i8).collect();
        assert_eq!(
            resc, want,
            "rescuable_{tag}.bin must be reference frame 3 at +/-100 with channel bits \
             {WEAK_FLIPS:?} sign-flipped at magnitude 1"
        );
        let weak: Vec<usize> = (0..n).filter(|&i| resc[i].abs() == 1).collect();
        assert_eq!(weak, WEAK_FLIPS.to_vec());
        assert!(
            (0..n).all(|i| resc[i].abs() == if WEAK_FLIPS.contains(&i) { 1 } else { 100 }),
            "rescuable_{tag}.bin carries an unexpected confidence magnitude"
        );

        let (_, payload) = split_one_selector("decode_soft", &format!("unrescuable_{tag}.bin"));
        let unre: Vec<i8> = payload.iter().map(|&b| b as i8).collect();
        assert_eq!(unre.len(), n, "unrescuable_{tag}.bin must hold one frame");
        assert!(
            unre.iter().all(|&v| v.abs() == 100),
            "unrescuable_{tag}.bin must carry full confidence everywhere — a low-confidence \
             bit is the marker of the rescuable class"
        );
        let corrupted = (0..n).filter(|&i| (unre[i] > 0) != (clean[i] > 0)).count();
        assert!(
            corrupted > n / 4,
            "unrescuable_{tag}.bin only flips {corrupted} of {n} bits against the clean \
             frame — that is inside the FEC's reach"
        );

        // Recovery: the Golay-protected vectors come back for the rescuable
        // seed and none of them do for the unrescuable one.
        let (clean_info, resc_info, unre_info, unre_errors, coded) = match rate {
            Rate::AmbePlus2_3600x2450 => {
                let f = |v: &[i8]| {
                    let s: &[i8; 72] = v.try_into().expect("half-rate soft frame is 72 LLRs");
                    rate33::frame::decode_frame_soft(s)
                };
                let (c, r, u) = (f(&clean), f(&resc), f(&unre));
                (
                    c.info.to_vec(),
                    r.info.to_vec(),
                    u.info.to_vec(),
                    u.errors.to_vec(),
                    // û₀ is Golay(24,12), û₁ Golay(23,12); û₂/û₃ are uncoded.
                    vec![(0usize, 3u8), (1, 3)],
                )
            }
            Rate::Imbe7200x4400 => {
                let f = |v: &[i8]| {
                    let s: &[i8; 144] = v.try_into().expect("full-rate soft frame is 144 LLRs");
                    imbe7200::frame::decode_frame_soft(s)
                };
                let (c, r, u) = (f(&clean), f(&resc), f(&unre));
                (
                    c.info.to_vec(),
                    r.info.to_vec(),
                    u.info.to_vec(),
                    u.errors.to_vec(),
                    // û₀..û₃ Golay(23,12); û₄..û₆ Hamming(15,11).
                    vec![
                        (0usize, 3u8),
                        (1, 3),
                        (2, 3),
                        (3, 3),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                    ],
                )
            }
            other => panic!("{other:?} has no soft FEC layer"),
        };
        assert!(!coded.is_empty(), "{rate:?} must have coded info vectors");
        for &(i, max_errors) in &coded {
            assert_eq!(
                unre_errors[i], max_errors,
                "unrescuable_{tag}.bin: coded vector {i} must saturate its FEC error count"
            );
        }
        let golay: Vec<usize> = coded
            .iter()
            .filter(|(_, e)| *e == 3)
            .map(|(i, _)| *i)
            .collect();
        assert!(
            !golay.is_empty(),
            "{rate:?} must have Golay-protected vectors"
        );
        for &i in &golay {
            assert_eq!(
                resc_info[i], clean_info[i],
                "rescuable_{tag}.bin: Golay-protected vector {i} was not recovered"
            );
        }
        assert!(
            golay.iter().all(|&i| unre_info[i] != clean_info[i]),
            "unrescuable_{tag}.bin recovered a Golay-protected vector, so it is not \
             unrescuable"
        );

        // The property the filename claims, in the direction that holds.
        let mut a = Vocoder::new(rate);
        let pcm_clean = a
            .decode_soft(&clean)
            .unwrap_or_else(|e| panic!("clean frame for {tag}: {e}"));
        let mut b = Vocoder::new(rate);
        let pcm_unre = b
            .decode_soft(&unre)
            .unwrap_or_else(|e| panic!("unrescuable_{tag}.bin: {e}"));
        assert_ne!(
            pcm_unre, pcm_clean,
            "unrescuable_{tag}.bin decodes to the clean frame's audio, so it is not \
             unrescuable"
        );
        let mut c = Vocoder::new(rate);
        let pcm_resc = c
            .decode_soft(&resc)
            .unwrap_or_else(|e| panic!("rescuable_{tag}.bin: {e}"));
        assert_ne!(
            pcm_resc, pcm_unre,
            "rescuable_{tag}.bin and unrescuable_{tag}.bin decode identically, so the two \
             classes are indistinguishable"
        );
    }
}

#[test]
fn decode_soft_nofec_seeds_are_rejected() {
    for tag in ["imbe4400", "ambe2450"] {
        let rate = rate_from_tag(tag);
        assert!(
            rate.soft_frame_bits().is_none(),
            "{rate:?} is supposed to be a no-FEC rate"
        );
        let name = format!("nofec_rejected_{tag}.bin");
        let (_, payload) = split_one_selector("decode_soft", &name);
        assert_eq!(payload.len(), 64, "{name} must carry 64 LLRs");
        assert!(
            payload.iter().all(|&b| b == 0x5A),
            "{name} must be a uniform +90 LLR run"
        );
        let llrs: Vec<i8> = payload.iter().map(|&b| b as i8).collect();
        let mut v = Vocoder::new(rate);
        let err = v
            .decode_soft(&llrs)
            .expect_err("a no-FEC rate must reject soft decode");
        assert!(
            matches!(err, VocoderError::SoftUnsupported { rate: r } if r == rate),
            "{name}: expected SoftUnsupported, got {err}"
        );
        // Any length must be rejected the same way, which is the point of the seed.
        for len in [0usize, 1, 71, 144] {
            let err = v
                .decode_soft(&vec![0i8; len])
                .expect_err("a no-FEC rate must reject soft decode at any length");
            assert!(
                matches!(err, VocoderError::SoftUnsupported { .. }),
                "{name}: {err}"
            );
        }
    }
}

// ===========================================================================
// 3. Content — encode_pcm
// ===========================================================================

/// FNV-1a of the shared 480-sample speech payload, so the deterministic signal
/// the `speech_*` seeds carry cannot drift unnoticed.
const SPEECH_PAYLOAD_HASH: u64 = 0x29fd_1a2c_69e9_dc6e;

#[test]
fn encode_pcm_seeds_carry_their_named_waveform() {
    let speech = split_one_selector("encode_pcm", "speech_imbe7200.bin").1;
    assert_eq!(
        speech.len(),
        3 * PCM_FRAME_BYTES,
        "speech seeds must hold exactly 3 frames"
    );
    assert_eq!(
        fnv1a64(&speech),
        SPEECH_PAYLOAD_HASH,
        "the deterministic speech-like signal shared by every speech_* seed changed"
    );
    let speech_pcm = pcm_samples(&speech);
    assert!(
        speech_pcm.chunks(FRAME_SAMPLES).all(|f| peak(f) >= 500),
        "every speech frame must carry audible level"
    );
    assert!(
        peak(&speech_pcm) < i16::MAX as i32 / 4,
        "the speech signal must stay well clear of the rails"
    );

    for tag in RATE_TAGS {
        // One signal, four selector bytes: the rate is the only variable.
        assert_eq!(
            split_one_selector("encode_pcm", &format!("speech_{tag}.bin")).1,
            speech,
            "speech_{tag}.bin must carry the same signal as speech_imbe7200.bin"
        );
        let (_, silence) = split_one_selector("encode_pcm", &format!("digital_silence_{tag}.bin"));
        assert_eq!(silence.len(), 3 * PCM_FRAME_BYTES);
        assert!(
            silence.iter().all(|&b| b == 0),
            "digital_silence_{tag}.bin must be all-zero PCM"
        );
    }

    let one_frame = |name: &str| -> Vec<i16> {
        let (_, payload) = split_one_selector("encode_pcm", name);
        assert_eq!(
            payload.len(),
            2 * PCM_FRAME_BYTES,
            "{name} must hold exactly 2 frames"
        );
        pcm_samples(&payload)
    };

    for name in ["rail_full_positive_imbe7200.bin"] {
        let pcm = one_frame(name);
        assert!(
            pcm.iter().all(|&s| s == i16::MAX),
            "{name} must be pinned at positive full scale"
        );
    }
    for name in ["rail_full_negative_imbe4400.bin"] {
        let pcm = one_frame(name);
        assert!(
            pcm.iter().all(|&s| s == i16::MIN),
            "{name} must be pinned at negative full scale"
        );
    }
    for name in [
        "rail_alternating_ambe3600.bin",
        "rail_alternating_imbe7200.bin",
    ] {
        let pcm = one_frame(name);
        assert!(
            pcm.iter()
                .enumerate()
                .all(|(i, &s)| s == if i % 2 == 0 { i16::MAX } else { i16::MIN }),
            "{name} must alternate between the rails every sample (full-scale Nyquist)"
        );
    }
    for (name, level) in [
        ("dc_positive_ambe2450.bin", 8192i16),
        ("dc_negative_imbe7200.bin", -8192),
    ] {
        let pcm = one_frame(name);
        assert!(
            pcm.iter().all(|&s| s == level),
            "{name} must be constant {level}"
        );
    }
    {
        let name = "impulse_ambe3600.bin";
        let pcm = one_frame(name);
        let nonzero: Vec<(usize, i16)> = pcm
            .iter()
            .enumerate()
            .filter(|(_, &s)| s != 0)
            .map(|(i, &s)| (i, s))
            .collect();
        assert_eq!(
            nonzero,
            vec![(80, i16::MAX), (240, i16::MIN)],
            "{name} must be one full-scale impulse per frame, at the frame centre"
        );
    }
    {
        let name = "ramp_imbe4400.bin";
        let pcm = one_frame(name);
        let want: Vec<i16> = (0..pcm.len())
            .map(|i| (-32768 + (65535 * i as i64) / (pcm.len() as i64 - 1)) as i16)
            .collect();
        assert_eq!(pcm, want, "{name} must sweep the full i16 range linearly");
    }
    {
        let name = "speech_silence_speech_ambe3600.bin";
        let (_, payload) = split_one_selector("encode_pcm", name);
        assert_eq!(
            payload.len(),
            6 * PCM_FRAME_BYTES,
            "{name} must hold 6 frames"
        );
        assert_eq!(
            &payload[..2 * PCM_FRAME_BYTES],
            &speech[..2 * PCM_FRAME_BYTES],
            "{name} must open with the first two speech frames"
        );
        assert!(
            payload[2 * PCM_FRAME_BYTES..4 * PCM_FRAME_BYTES]
                .iter()
                .all(|&b| b == 0),
            "{name} must gate to digital silence for two frames"
        );
        assert_eq!(
            &payload[4 * PCM_FRAME_BYTES..],
            &speech[..2 * PCM_FRAME_BYTES],
            "{name} must return to speech, so both silence-gate transitions are covered"
        );
    }
    {
        let name = "odd_length_imbe7200.bin";
        let (_, payload) = split_one_selector("encode_pcm", name);
        assert_eq!(
            payload,
            speech[..PCM_FRAME_BYTES + 90 * 2],
            "{name} must be the speech signal truncated to one frame plus 90 samples"
        );
    }
    {
        let name = "selector_only_ambe2450.bin";
        let (_, payload) = split_one_selector("encode_pcm", name);
        assert!(payload.is_empty(), "{name} must be a bare selector byte");
        assert_eq!(seed("encode_pcm", name).len(), 1);
    }
}

// ===========================================================================
// 3. Content — transcode
// ===========================================================================

#[test]
fn transcode_seeds_carry_their_named_frames() {
    const TONES: [(u8, u8); 3] = [(5, 90), (5, 90), (40, 51)];

    for name in TRANSCODE_SEEDS {
        let (tag_from, _) = pair_tags(name);
        let from = rate_from_tag(tag_from);
        let n = from.fec_frame_bytes();
        let (_, _, payload) = split_two_selectors("transcode", name);
        let frames: Vec<&[u8]> = payload.chunks(n).collect();

        if name.starts_with("zeros_") {
            assert_eq!(frames.len(), 2, "{name} must hold 2 frames");
            assert!(
                payload.iter().all(|&b| b == 0x00),
                "{name} must be all-zero"
            );
        } else if name.starts_with("ones_") {
            assert_eq!(frames.len(), 2, "{name} must hold 2 frames");
            assert!(
                payload.iter().all(|&b| b == 0xFF),
                "{name} must be all-ones"
            );
        } else if name.starts_with("tone_annex_t_") {
            assert_eq!(frames.len(), TONES.len(), "{name} must hold 3 tone frames");
            for (i, &(id, amplitude)) in TONES.iter().enumerate() {
                assert_eq!(
                    frames[i],
                    tone_frame(from, id, amplitude).as_slice(),
                    "{name} frame {i} is not the Annex-T frame for (I_D {id}, A_D {amplitude})"
                );
                assert_eq!(
                    blip25_codec::tone::classify(&ambe_info(from, frames[i])),
                    FrameKind::Tone,
                    "{name} frame {i} does not classify as a tone"
                );
            }
        } else if name.starts_with("single_") {
            let refs = reference_frames(tag_from);
            assert_eq!(frames.len(), 1, "{name} must hold 1 frame");
            assert_eq!(
                frames[0], refs[0],
                "{name} must be reference frame 0 for {tag_from}"
            );
        } else if name.starts_with("valid_") {
            let refs = reference_frames(tag_from);
            assert_eq!(frames.len(), 3, "{name} must hold 3 frames");
            for k in 0..2 {
                assert_eq!(
                    frames[k], refs[k],
                    "{name} frame {k} must be reference frame {k} for {tag_from}"
                );
            }
            for (i, f) in frames.iter().enumerate() {
                assert!(
                    is_voice_frame(from, f),
                    "{name} frame {i} is not a voice frame"
                );
            }
        } else if name.starts_with("conceal_") {
            let run = conceal_frames(tag_from);
            assert_eq!(frames.len(), 6, "{name} must hold 6 frames");
            assert!(
                is_voice_frame(from, frames[0]),
                "{name} frame 0 must be a good frame ahead of the run"
            );
            for k in 1..6 {
                assert_eq!(
                    frames[k],
                    run[k + 1].as_slice(),
                    "{name} frame {k} must be frame {} of the {tag_from} concealment run",
                    k + 1
                );
                assert!(
                    !is_voice_frame(from, frames[k]),
                    "{name} frame {k} must be a frame the decoder has to conceal"
                );
            }
        } else {
            panic!("{name}: unknown transcode seed class");
        }
    }
}

// ===========================================================================
// Coverage guard — no class may sit on a shape-only check
// ===========================================================================

/// Every seed must belong to a class that one of the content tests above
/// pins. A new seed with an unrecognised name fails here rather than
/// silently riding on the framing checks alone.
#[test]
fn seed_classes_are_all_pinned() {
    const DECODE_BITS_CLASSES: [&str; 6] = [
        "conceal_",
        "fec_errors_",
        "ones_",
        "tone_annex_t_",
        "valid_",
        "zeros_",
    ];
    const DECODE_SOFT_CLASSES: [&str; 9] = [
        "erasure_llr_",
        "max_llr_",
        "min_llr_",
        "nofec_rejected_",
        "rescuable_",
        "tone_llr_",
        "unrescuable_",
        "valid_strong_",
        "valid_weak_",
    ];
    const DECODE_SOFT_ZERO: &str = "zero_llr_";
    const ENCODE_PCM_CLASSES: [&str; 11] = [
        "dc_negative_",
        "dc_positive_",
        "digital_silence_",
        "impulse_",
        "odd_length_",
        "rail_alternating_",
        "rail_full_negative_",
        "rail_full_positive_",
        "ramp_",
        "selector_only_",
        "speech_",
    ];
    const TRANSCODE_CLASSES: [&str; 6] = [
        "conceal_",
        "ones_",
        "single_",
        "tone_annex_t_",
        "valid_",
        "zeros_",
    ];

    let covered = |name: &str, classes: &[&str]| classes.iter().any(|c| name.starts_with(c));
    for name in DECODE_BITS_SEEDS {
        assert!(
            covered(name, &DECODE_BITS_CLASSES),
            "decode_bits/{name} is unclassified"
        );
    }
    for name in DECODE_SOFT_SEEDS {
        assert!(
            covered(name, &DECODE_SOFT_CLASSES) || name.starts_with(DECODE_SOFT_ZERO),
            "decode_soft/{name} is unclassified"
        );
    }
    for name in ENCODE_PCM_SEEDS {
        assert!(
            covered(name, &ENCODE_PCM_CLASSES),
            "encode_pcm/{name} is unclassified"
        );
    }
    for name in TRANSCODE_SEEDS {
        assert!(
            covered(name, &TRANSCODE_CLASSES),
            "transcode/{name} is unclassified"
        );
    }
}
