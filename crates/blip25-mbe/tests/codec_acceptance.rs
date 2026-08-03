//! reference vocoder acceptance GATE — the permanent regression tripwire proving that
//! blip25-mbe's DEFAULT public path is byte / sample-identical to the vendored
//! `blip25_codec` reference codec, for ALL 4 rates.
//!
//! This is keyed on decode / encode **OUTPUT** equality (not field equality):
//! a future default-config divergence — e.g. re-enabling a post-decode
//! enhancement in `Vocoder::new` / `VocoderBuilder` — makes the DEFAULT output
//! differ from `blip25_codec` direct and FAILS this gate. A field-by-field
//! comparison is not enough: equal fields do not imply equal output, so a
//! default-config divergence can slip past one.
//!
//! Two tiers:
//!   * HERMETIC (always runs, no external data): a deterministic synthetic
//!     signal is encoded through `blip25_codec` direct to real frames, then the
//!     blip25 DEFAULT decode / encode / streaming / live / transcode paths are
//!     asserted byte/sample-equal to `blip25_codec` direct. Covers all 4 rates.
//!   * CORPUS (runs only when the reference `tv-rc` tree is reachable): the same
//!     OUTPUT-equality assertions on a representative set of REAL vectors
//!     (speech + tone + error), all 4 rates. The EXHAUSTIVE 105-vector × 4-rate
//!     sweep lives in `examples/codec_acceptance_sweep.rs` (too slow for the
//!     default `cargo test` run); this gate exercises a fast representative set.
//!
//! The reference is `blip25_codec` direct, NOT the reference `.pcm` files: the vendored
//! codec intentionally does not bit-reproduce the reference decode (the
//! synthesis phase back-end caveat documented in `crates/blip25-codec/src/lib.rs`).
//! The contract under test is "the blip25 façade reproduces the vendored codec
//! exactly."

use blip25_mbe::enhancement::{ClassicalConfig, EnhancementMode};
use blip25_mbe::vocoder::{
    imbe_fec_to_info_bytes, LiveDecoder, LiveEncoder, Rate, Transcoder, Vocoder,
};
use std::path::{Path, PathBuf};

const ALL_RATES: [Rate; 4] = [
    Rate::AmbePlus2_3600x2450,
    Rate::AmbePlus2_2450x2450,
    Rate::Imbe7200x4400,
    Rate::Imbe4400x4400,
];

fn rate_name(r: Rate) -> &'static str {
    match r {
        Rate::AmbePlus2_3600x2450 => "AMBE+2 r33 (9B)",
        Rate::AmbePlus2_2450x2450 => "AMBE+2 r34 (7B)",
        Rate::Imbe7200x4400 => "IMBE full (18B)",
        Rate::Imbe4400x4400 => "IMBE info (11B)",
        _ => "UNKNOWN",
    }
}

/// Rate-aware b-field decode (`[b0..b8]`) of one encoded AMBE+2 frame — the FEC
/// r33 (9B) or no-FEC r34 (7B) info layout. Lets the streaming/whole-buffer
/// parity check compare pitch/voicing (b0/b1) independently of the amplitude and
/// gain fields (b2..b8), which the two paths legitimately diverge on.
fn ambe_fields(rate: Rate, frame: &[u8]) -> [u16; 9] {
    match rate {
        Rate::AmbePlus2_3600x2450 => blip25_mbe::rate33::frame::fields_from_fec(frame),
        Rate::AmbePlus2_2450x2450 => blip25_mbe::rate33::frame::fields_from_no_fec(frame),
        _ => unreachable!("ambe_fields called on a non-AMBE+2 rate"),
    }
}

// ---------- blip25_codec-direct references (independent of blip25 routing) ----------

fn codec_direct_decode(rate: Rate, bits: &[u8]) -> Vec<i16> {
    let n = rate.fec_frame_bytes();
    let mut w_ambe = blip25_codec::Decoder::new();
    let mut w_imbe = blip25_codec::ImbeDecoder::new();
    let mut add_fec = Transcoder::new(Rate::Imbe4400x4400, Rate::Imbe7200x4400).unwrap();
    let mut out = Vec::new();
    for f in 0..(bits.len() / n) {
        let fr = &bits[f * n..(f + 1) * n];
        let pcm: Vec<i16> = match rate {
            // Concealed fixed path — the shipped default (BABA-1 §5.6/§5.7
            // repeat/mute), so this reference matches `Vocoder::decode_bits`.
            Rate::AmbePlus2_3600x2450 => w_ambe.decode_pcm_fixed_concealed(fr).0.to_vec(),
            Rate::AmbePlus2_2450x2450 => {
                let mut r34 = [0u8; 7];
                r34.copy_from_slice(&fr[..7]);
                let b = blip25_codec::enc::encode_frame::decode_r34(&r34);
                let r33 = blip25_codec::enc::encode_frame::encode_r33(&b);
                w_ambe.decode_pcm_fixed_concealed(&r33).0.to_vec()
            }
            Rate::Imbe7200x4400 => {
                let mut a = [0u8; 18];
                a.copy_from_slice(&fr[..18]);
                w_imbe.decode_pcm(&a).to_vec()
            }
            Rate::Imbe4400x4400 => {
                let full = add_fec.transcode(fr).unwrap();
                let mut a = [0u8; 18];
                a.copy_from_slice(&full[..18]);
                w_imbe.decode_pcm(&a).to_vec()
            }
            _ => unreachable!(),
        };
        out.extend(pcm);
    }
    out
}

/// blip25_codec-direct encode: ordered list of REAL emitted frames (placeholder-free).
fn codec_direct_encode(rate: Rate, pcm: &[i16]) -> Vec<Vec<u8>> {
    let mut e = blip25_codec::enc::Encoder::new();
    let mut out: Vec<Vec<u8>> = Vec::new();
    for f in 0..(pcm.len() / 160) {
        let frame: &[i16; 160] = (&pcm[f * 160..(f + 1) * 160]).try_into().unwrap();
        let emitted = match rate {
            Rate::AmbePlus2_3600x2450 => e.encode_frame_r33(frame).map(|b| b.to_vec()),
            Rate::AmbePlus2_2450x2450 => e.encode_frame_r34(frame).map(|b| b.to_vec()),
            Rate::Imbe7200x4400 => e.encode_imbe_frame(frame).map(|b| b.to_vec()),
            Rate::Imbe4400x4400 => e
                .encode_imbe_frame(frame)
                .map(|b| imbe_fec_to_info_bytes(&b).to_vec()),
            _ => unreachable!(),
        };
        if let Some(b) = emitted {
            out.push(b);
        }
    }
    let flush: Vec<Vec<u8>> = match rate {
        Rate::AmbePlus2_3600x2450 => e.flush_r33().into_iter().map(|b| b.to_vec()).collect(),
        Rate::AmbePlus2_2450x2450 => e.flush_r34().into_iter().map(|b| b.to_vec()).collect(),
        Rate::Imbe7200x4400 => e.flush_imbe().into_iter().map(|b| b.to_vec()).collect(),
        Rate::Imbe4400x4400 => e
            .flush_imbe()
            .into_iter()
            .map(|b| imbe_fec_to_info_bytes(&b).to_vec())
            .collect(),
        _ => unreachable!(),
    };
    out.extend(flush);
    out
}

// ---------- blip25 DEFAULT-path drivers (no config touched) ----------

fn blip25_default_decode(rate: Rate, bits: &[u8]) -> Vec<i16> {
    let n = rate.fec_frame_bytes();
    let mut v = Vocoder::new(rate);
    let mut out = Vec::new();
    for f in 0..(bits.len() / n) {
        out.extend(v.decode_bits(&bits[f * n..(f + 1) * n]).unwrap());
    }
    out
}

/// blip25 DEFAULT encode + flush, look-ahead aligned (placeholder dropped).
/// Returns `(real_frames, placeholder_is_zero)`.
fn blip25_default_encode(rate: Rate, pcm: &[i16]) -> (Vec<Vec<u8>>, bool) {
    let mut v = Vocoder::new(rate);
    let mut ours: Vec<Vec<u8>> = Vec::new();
    for f in 0..(pcm.len() / 160) {
        ours.push(v.encode_pcm(&pcm[f * 160..(f + 1) * 160]).unwrap());
    }
    let flush = v.flush_encode();
    let ph_zero = ours
        .first()
        .map(|p| p.iter().all(|&b| b == 0))
        .unwrap_or(true);
    let mut stream: Vec<Vec<u8>> = ours.into_iter().skip(1).collect();
    stream.extend(flush);
    (stream, ph_zero)
}

fn max_abs_diff(a: &[i16], b: &[i16]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).abs())
        .max()
        .unwrap_or(0)
}

// ---------- the OUTPUT-equality assertions, reused hermetic + corpus ----------

/// DECODE gate: DEFAULT `Vocoder`, DEFAULT `VocoderBuilder`, `decode_stream`,
/// and `LiveDecoder` all reproduce `blip25_codec` direct exactly on `bits`.
fn assert_decode_parity(rate: Rate, bits: &[u8], ctx: &str) {
    let refr = codec_direct_decode(rate, bits);

    // 1. Vocoder::new default (no set_enhancement).
    let ours = blip25_default_decode(rate, bits);
    assert_eq!(
        ours.len(),
        refr.len(),
        "{ctx} {}: decode_bits sample count {} != blip25_codec {}",
        rate_name(rate),
        ours.len(),
        refr.len()
    );
    assert_eq!(
        max_abs_diff(&ours, &refr),
        0,
        "{ctx} {}: decode_bits (DEFAULT) diverges from blip25_codec direct",
        rate_name(rate)
    );

    // 2. VocoderBuilder default build() must equal too (guards the builder path).
    let mut vb = Vocoder::builder(rate).build();
    let n = rate.fec_frame_bytes();
    let mut builder_out = Vec::new();
    for f in 0..(bits.len() / n) {
        builder_out.extend(vb.decode_bits(&bits[f * n..(f + 1) * n]).unwrap());
    }
    assert_eq!(
        builder_out,
        refr,
        "{ctx} {}: VocoderBuilder::build() default decode diverges from blip25_codec direct",
        rate_name(rate)
    );

    // 3. decode_stream (default) == the whole reference.
    let mut vs = Vocoder::new(rate);
    let stream: Vec<i16> = vs
        .decode_stream(bits)
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(
        stream,
        refr,
        "{ctx} {}: decode_stream diverges from blip25_codec direct",
        rate_name(rate)
    );

    // 4. LiveDecoder push(odd chunks) == the whole reference (zero samples lost).
    let mut live = LiveDecoder::new(rate);
    let mut lp = Vec::new();
    for c in bits.chunks(n * 3 + 5) {
        for r in live.push(c) {
            lp.extend(r.unwrap());
        }
    }
    assert_eq!(
        lp,
        refr,
        "{ctx} {}: LiveDecoder diverges from blip25_codec direct",
        rate_name(rate)
    );
    assert_eq!(
        live.pending_bytes(),
        (bits.len() % n),
        "{ctx} {}: LiveDecoder pending bytes wrong",
        rate_name(rate)
    );
}

/// ENCODE gate: DEFAULT `Vocoder` encode+flush, `encode_stream`, and
/// `LiveEncoder`+flush all reproduce `blip25_codec` direct exactly, and LiveEncoder
/// loses ZERO frames.
fn assert_encode_parity(rate: Rate, pcm: &[i16], ctx: &str) {
    assert_encode_parity_opt(rate, pcm, ctx, true);
}

/// `ambe_strict`: the AMBE+2 `LiveEncoder` (single-pass reference streamer) is
/// compared to whole-buffer `Vocoder::encode`. With Route A (the two-frame
/// look-ahead) the two now agree byte-for-byte on pitch (b0) AND amplitude/gain
/// (b2..b8), both checked every frame. Voicing (b1) is checked only when
/// `ambe_strict` (speech), since the whole-buffer global lag search makes
/// pure-tone b1 non-reproducible in a stream. The straight encode_pcm /
/// encode_stream paths stay byte-strict.
fn assert_encode_parity_opt(rate: Rate, pcm: &[i16], ctx: &str, ambe_strict: bool) {
    let refr = codec_direct_encode(rate, pcm);

    // 1. Vocoder::new default encode_pcm loop + flush_encode.
    let (ours, ph_zero) = blip25_default_encode(rate, pcm);
    assert!(
        ph_zero,
        "{ctx} {}: frame-0 look-ahead placeholder is not all-zero",
        rate_name(rate)
    );
    assert_eq!(
        ours.len(),
        refr.len(),
        "{ctx} {}: encode frame count {} != blip25_codec {} (lost/gained frames)",
        rate_name(rate),
        ours.len(),
        refr.len()
    );
    assert_eq!(
        ours,
        refr,
        "{ctx} {}: encode_pcm+flush (DEFAULT) diverges from blip25_codec direct",
        rate_name(rate)
    );

    // 2. encode_stream + flush == encode_pcm loop (== blip25_codec direct).
    let mut vs = Vocoder::new(rate);
    let mut stream: Vec<Vec<u8>> = vs
        .encode_stream(pcm)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert!(
        !stream.is_empty(),
        "{ctx} {}: encode_stream empty",
        rate_name(rate)
    );
    stream.remove(0); // drop placeholder
    stream.extend(vs.flush_encode());
    assert_eq!(
        stream,
        refr,
        "{ctx} {}: encode_stream+flush diverges from blip25_codec direct",
        rate_name(rate)
    );

    // 3. LiveEncoder push(odd chunks) + flush: byte-identical AND zero frames lost.
    let mut live = LiveEncoder::new(rate);
    let splits = [137usize, 400, 99, 160, 320, 211, 41];
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let (mut pos, mut i) = (0usize, 0usize);
    while pos < pcm.len() {
        let take = splits[i % splits.len()];
        i += 1;
        let end = (pos + take).min(pcm.len());
        for r in live.push(&pcm[pos..end]) {
            emitted.push(r.unwrap());
        }
        pos = end;
    }
    for b in live.flush().unwrap() {
        emitted.push(b);
    }
    assert!(
        !emitted.is_empty(),
        "{ctx} {}: LiveEncoder empty",
        rate_name(rate)
    );
    // `LiveEncoder` is the single-pass streaming replica of whole-buffer
    // `Vocoder::encode` at EVERY rate — `AmbeStream` for AMBE+2, `ImbeStream`
    // for IMBE. Both carry the reference pitch and voicing analysis causally,
    // both handle their own look-ahead internally (nothing to strip here), and
    // both emit one frame per 20 ms of input. So the reference is `encode`, not
    // the per-frame codec-direct stream that steps 1-3 above pin.
    let is_ambe = matches!(rate, Rate::AmbePlus2_3600x2450 | Rate::AmbePlus2_2450x2450);
    let expected = Vocoder::new(rate).encode(pcm);
    assert_eq!(
        emitted.len(),
        expected.len(),
        "{ctx} {}: LiveEncoder emitted {} real frames, expected {} — frames LOST",
        rate_name(rate),
        emitted.len(),
        expected.len()
    );
    if is_ambe {
        // Streaming AmbeStream vs whole-buffer `Vocoder::encode`. Route A gave
        // the streaming amplitude Encoder the same two-frame look-ahead the
        // whole-buffer path uses, so both feed `band_decompress` the identical
        // pitch-aligned live `gap2` window and produce byte-identical amplitude /
        // gain fields (b2..b8) — as well as the same reference spectral-DP pitch (b0).
        //
        // Pitch (b0) and amplitude/gain (b2..b8) are checked every frame. Voicing
        // (b1) comes from the streamer's bounded `b1_track` vs the whole-buffer
        // global ±3 lag search; the two agree on speech but the lag search makes
        // pure-tone b1 non-reproducible per-frame, so b1 is only asserted under
        // `ambe_strict`.
        let n = emitted.len();
        let ef: Vec<[u16; 9]> = emitted.iter().map(|f| ambe_fields(rate, f)).collect();
        let xf: Vec<[u16; 9]> = expected.iter().map(|f| ambe_fields(rate, f)).collect();
        let b0_mism: Vec<usize> = (0..n).filter(|&i| ef[i][0] != xf[i][0]).collect();
        assert!(
            b0_mism.is_empty(),
            "{ctx} {}: streaming vs whole-buffer pitch(b0) diverges at frames {:?} ({} of {n})",
            rate_name(rate),
            &b0_mism[..b0_mism.len().min(8)],
            b0_mism.len(),
        );
        // Route A parity: amplitude/gain fields must match byte-for-byte.
        let amp_mism: Vec<usize> = (0..n).filter(|&i| ef[i][2..9] != xf[i][2..9]).collect();
        assert!(
            amp_mism.is_empty(),
            "{ctx} {}: streaming vs whole-buffer amplitude/gain(b2..b8) diverges at frames {:?} ({} of {n}) — Route A parity broken",
            rate_name(rate),
            &amp_mism[..amp_mism.len().min(8)],
            amp_mism.len(),
        );
        if ambe_strict {
            let b1_mism: Vec<usize> = (0..n).filter(|&i| ef[i][1] != xf[i][1]).collect();
            assert!(
                b1_mism.is_empty(),
                "{ctx} {}: streaming vs whole-buffer voicing(b1) diverges at frames {:?} ({} of {n})",
                rate_name(rate),
                &b1_mism[..b1_mism.len().min(8)],
                b1_mism.len(),
            );
        }
    } else {
        // IMBE has no r33 field decomposition to compare through, so the
        // streaming replica is asserted on the packed wire bytes directly.
        assert_eq!(
            emitted,
            expected,
            "{ctx} {}: LiveEncoder (ImbeStream) diverges from Vocoder::encode",
            rate_name(rate)
        );
    }
    assert_eq!(
        live.pending_samples(),
        0,
        "{ctx} {}: LiveEncoder left samples pending after flush",
        rate_name(rate)
    );
}

// ---------- deterministic synthetic corpus (hermetic) ----------

/// A deterministic ~50-frame speech-like signal: onset from silence, a voiced
/// harmonic segment, an unvoiced (noise) segment, silence, and a second louder
/// voiced burst. Exercises the voiced / unvoiced / silence synthesis paths on
/// both codecs so the OUTPUT-equality gate isn't just proving silence.
fn synthetic_pcm() -> Vec<i16> {
    let mut lcg: u32 = 0x1234_5678;
    let mut rng = || {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((lcg >> 8) as i32 & 0xFFFF) - 0x8000 // ~[-32768, 32767]
    };
    let mut out: Vec<i16> = Vec::new();
    let sr = 8000.0_f64;

    let voiced = |out: &mut Vec<i16>, nframes: usize, f0: f64, amp: f64, env_on: bool, ph0: f64| {
        for fr in 0..nframes {
            let env = if env_on {
                (fr as f64 + 1.0) / nframes as f64
            } else {
                1.0
            };
            for s in 0..160 {
                let t = (fr * 160 + s) as f64 / sr;
                let mut v = 0.0;
                for h in 1..=6 {
                    v += (1.0 / h as f64)
                        * (2.0 * std::f64::consts::PI * f0 * h as f64 * t + ph0).sin();
                }
                let x = (v * amp * env).round();
                out.push(x.clamp(-30000.0, 30000.0) as i16);
            }
        }
    };
    let noise = |out: &mut Vec<i16>, rng: &mut dyn FnMut() -> i32, nframes: usize, amp: i32| {
        for _ in 0..(nframes * 160) {
            out.push(((rng() * amp) >> 15) as i16);
        }
    };
    let silence = |out: &mut Vec<i16>, nframes: usize| {
        out.extend(std::iter::repeat(0i16).take(nframes * 160))
    };

    silence(&mut out, 2);
    voiced(&mut out, 12, 140.0, 4000.0, true, 0.0); // onset voiced
    noise(&mut out, &mut rng, 8, 6000); // unvoiced
    silence(&mut out, 4);
    voiced(&mut out, 12, 220.0, 7000.0, false, 0.7); // louder voiced burst
    voiced(&mut out, 6, 110.0, 3000.0, false, 1.3); // low-pitch tail
    out
}

// ================================ HERMETIC ================================

#[test]
fn hermetic_decode_parity_all_rates() {
    let pcm = synthetic_pcm();
    for &rate in &ALL_RATES {
        // Real frames from blip25_codec direct = a valid bit stream for this rate.
        let bits: Vec<u8> = codec_direct_encode(rate, &pcm).concat();
        assert!(
            bits.len() >= rate.fec_frame_bytes() * 20,
            "{}: synthetic bit stream too short ({})",
            rate_name(rate),
            bits.len()
        );
        assert_decode_parity(rate, &bits, "hermetic");
    }
}

#[test]
fn hermetic_encode_parity_all_rates() {
    let pcm = synthetic_pcm();
    for &rate in &ALL_RATES {
        assert_encode_parity(rate, &pcm, "hermetic");
    }
}

/// The gate must be SENSITIVE: prove the one remaining default-config
/// divergence (a non-None post-decode enhancement) actually changes the
/// OUTPUT, so re-introducing it as a default would fail `assert_decode_parity`.
#[test]
fn tripwire_config_divergences_change_output() {
    let pcm = synthetic_pcm();
    let rate = Rate::Imbe7200x4400;
    let bits: Vec<u8> = codec_direct_encode(rate, &pcm).concat();
    let refr = codec_direct_decode(rate, &bits);
    let n = rate.fec_frame_bytes();

    // A non-None post-decode enhancement diverges from blip25_codec direct.
    let mut ve = Vocoder::new(rate);
    ve.set_enhancement(EnhancementMode::Classical(ClassicalConfig::default()));
    let mut enh = Vec::new();
    for f in 0..(bits.len() / n) {
        enh.extend(ve.decode_bits(&bits[f * n..(f + 1) * n]).unwrap());
    }
    assert_ne!(
        enh, refr,
        "enhancement=Classical did NOT change decode output — the decode gate \
         would be blind to a non-None enhancement default"
    );
}

/// Same-codec wire-transform invariants the streaming/decode routing relies on.
///
/// Note: `full -> info -> full` is deliberately NOT asserted to be identity —
/// stripping FEC drops the encoder's original parity bits, and re-adding FEC
/// recomputes them, so the reconstructed full frame can differ from the
/// original. That is harmless for the gate: BOTH the blip25 decode routing and
/// the blip25_codec-direct reference apply the SAME info->full re-add before decoding
/// the info rate, so decode parity holds regardless (proven above). What is
/// asserted here are the info-preserving identities the wire layer guarantees.
#[test]
fn transcoder_wire_transforms_consistent() {
    let pcm = synthetic_pcm();

    // IMBE: (1) the public Transcoder full->info strip == imbe_fec_to_info_bytes;
    //       (2) info->full->info is identity (re-add then strip preserves info).
    let full_frames = codec_direct_encode(Rate::Imbe7200x4400, &pcm);
    let mut strip = Transcoder::new(Rate::Imbe7200x4400, Rate::Imbe4400x4400).unwrap();
    let mut add = Transcoder::new(Rate::Imbe4400x4400, Rate::Imbe7200x4400).unwrap();
    for fr in &full_frames {
        let a: [u8; 18] = fr[..18].try_into().unwrap();
        let info_t = strip.transcode(&a).unwrap();
        assert_eq!(
            info_t,
            imbe_fec_to_info_bytes(&a).to_vec(),
            "IMBE full->info transcode != public imbe_fec_to_info_bytes"
        );
        let full2 = add.transcode(&info_t).unwrap();
        let info2 = strip.transcode(&full2).unwrap();
        assert_eq!(info2, info_t, "IMBE info->full->info is not identity");
    }

    // AMBE+2: r34(info)->r33(full)->r34(info) is identity.
    let r34_frames = codec_direct_encode(Rate::AmbePlus2_2450x2450, &pcm);
    let mut to33 = Transcoder::new(Rate::AmbePlus2_2450x2450, Rate::AmbePlus2_3600x2450).unwrap();
    let mut to34 = Transcoder::new(Rate::AmbePlus2_3600x2450, Rate::AmbePlus2_2450x2450).unwrap();
    for fr in &r34_frames {
        let full = to33.transcode(fr).unwrap();
        assert_eq!(full.len(), 9);
        let back = to34.transcode(&full).unwrap();
        assert_eq!(back, fr.clone(), "AMBE r34->r33->r34 is not identity");
    }
}

/// The r34 (2450, no-FEC) 49-bit wire layout is the reference's 3-way column interleave,
/// and BOTH layers must produce it.
///
/// A bijective mis-permutation still round-trips, so the `r34->r33->r34`
/// identity asserted above passes under *any* consistent ordering: it cannot
/// tell the engine's packer apart from `rate33::frame::pack_no_fec`, and cannot
/// tell either apart from the reference. Only `R34_BIT_ORDER` is interoperable — packing
/// in straight `b0..b8` parameter order with no permute matches the reference on **0**
/// of 110,840 vector frames, while the interleaved layout matches 100.00% of
/// all 97,177 clean-vector frames.
///
/// These assertions are hermetic: they need no reference material, because the
/// permutation table is the frozen record of that measurement.
#[test]
fn r34_wire_layout_is_reference_interleave_in_both_layers() {
    use blip25_codec::enc::encode_frame::{decode_r34, encode_r34, R34_BIT_ORDER};

    // 1. The two independently-maintained permutation tables agree.
    assert_eq!(
        R34_BIT_ORDER,
        blip25_mbe::rate33::frame::R34_BIT_ORDER,
        "engine and rate33 R34_BIT_ORDER have drifted apart"
    );

    // 2. It really is a permutation (every natural bit placed exactly once).
    let mut seen = [false; 49];
    for &src in R34_BIT_ORDER.iter() {
        assert!(!seen[src as usize], "R34_BIT_ORDER repeats bit {src}");
        seen[src as usize] = true;
    }
    assert!(
        seen.iter().all(|&s| s),
        "R34_BIT_ORDER does not cover all 49 bits"
    );

    // 3. The engine's packer and the spec-derived packer agree byte-for-byte on
    //    real encoder output, and the engine round-trips its own frames.
    let pcm = synthetic_pcm();
    let r33_frames = codec_direct_encode(Rate::AmbePlus2_3600x2450, &pcm);
    assert!(!r33_frames.is_empty());
    for fr in &r33_frames {
        let info = blip25_mbe::rate33::frame::fields_from_fec(fr);
        let b: [u16; 9] = info[..9].try_into().unwrap();

        let engine = encode_r34(&b);
        let spec =
            blip25_mbe::rate33::frame::pack_no_fec(&blip25_codec::frame::decode_bytes(fr).info);
        assert_eq!(
            engine.to_vec(),
            spec.to_vec(),
            "engine encode_r34 != rate33 pack_no_fec — the r34 layout split is back"
        );
        assert_eq!(
            decode_r34(&engine),
            b,
            "encode_r34/decode_r34 is not a round-trip"
        );
    }

    // 4. End-to-end: what `Vocoder` emits at 2450 equals what `Transcoder`
    //    produces from the same 3600 frame.
    let voc_r34 = codec_direct_encode(Rate::AmbePlus2_2450x2450, &pcm);
    let mut to34 = Transcoder::new(Rate::AmbePlus2_3600x2450, Rate::AmbePlus2_2450x2450).unwrap();
    for (i, fr33) in r33_frames.iter().enumerate() {
        let via_transcoder = to34.transcode(fr33).unwrap();
        assert_eq!(
            via_transcoder, voc_r34[i],
            "frame {i}: Transcoder(3600->2450) != direct 2450 encode"
        );
    }
}

// ================================= CORPUS =================================

/// Locate the reference `tv-rc` tree. `BLIP25_TVRC_DIR` wins; otherwise the
/// gitignored `reference-material/Vectors/tv-rc` symlink at the workspace root.
///
/// The fallback is resolved from `CARGO_MANIFEST_DIR`, not from the working
/// directory: cargo runs a test binary with the *package* root as its cwd, so a
/// plain relative path would look for `crates/blip25-mbe/reference-material/` and never find
/// anything.
///
/// Returns `None` (and the corpus tier then demands an explicit acknowledgement
/// rather than skipping quietly) when the vectors are absent. They are not
/// redistributable, so that is the normal state of a fresh clone and of CI.
fn corpus_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("BLIP25_TVRC_DIR") {
        let p = PathBuf::from(d);
        if p.is_dir() {
            return Some(p);
        }
    }
    let p = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../reference-material/Vectors/tv-rc"
    ));
    if p.is_dir() {
        Some(p)
    } else {
        None
    }
}

fn bit_subdir(r: Rate) -> &'static str {
    match r {
        Rate::AmbePlus2_3600x2450 => "r33",
        Rate::AmbePlus2_2450x2450 => "r34",
        Rate::Imbe7200x4400 => "p25",
        Rate::Imbe4400x4400 => "p25_nofec",
        _ => unreachable!(),
    }
}

fn read_pcm_file(p: &Path) -> Vec<i16> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Representative REAL-vector gate. The full 105-vector × 4-rate sweep is in
/// `examples/codec_acceptance_sweep.rs`; this gates a fast representative set
/// (speech + tone + error) on prefixes so `cargo test` stays quick.
#[test]
fn corpus_representative_parity_all_rates() {
    let Some(root) = corpus_dir() else {
        // A skip that prints to stderr and returns is reported by cargo as a
        // PASS, so in a green summary it is indistinguishable from a test that
        // actually verified something. Fail loudly instead, and require the
        // absence to be declared explicitly.
        assert!(
            std::env::var_os("BLIP25_ALLOW_MISSING_CORPUS").is_some(),
            "corpus_representative_parity_all_rates: the reference tv-rc not found. \
             Set BLIP25_TVRC_DIR to point at it, or set \
             BLIP25_ALLOW_MISSING_CORPUS=1 to acknowledge running without it \
             (that is expected on a fresh clone, in CI, and for consumers — the \
             hermetic gates still ran). Refusing to report a silent skip as a \
             pass."
        );
        eprintln!("SKIP corpus_representative_parity_all_rates: corpus absent, explicitly allowed");
        return;
    };
    // Names present across the rate subdirs; speech + tone + channel-error.
    let names = ["mark", "alltone", "dam_e5_hd"];
    const DEC_FRAMES: usize = 120; // prefix, keeps the slow IMBE synth fast
    const ENC_FRAMES: usize = 60; // encode is ~140 frames/s, so keep short

    let mut decode_checked = 0usize;
    let mut encode_checked = 0usize;
    for &rate in &ALL_RATES {
        let n = rate.fec_frame_bytes();
        for name in &names {
            // DECODE: this rate's own .bit vector (prefix).
            let bitp = root.join(bit_subdir(rate)).join(format!("{name}.bit"));
            if let Ok(bytes) = std::fs::read(&bitp) {
                let take = (DEC_FRAMES * n).min(bytes.len() / n * n);
                if take >= n {
                    assert_decode_parity(rate, &bytes[..take], "corpus");
                    decode_checked += 1;
                }
            }
            // ENCODE: shared r33/*.pcm prefix, fed identically to both encoders.
            let pcmp = root.join("r33").join(format!("{name}.pcm"));
            if pcmp.exists() {
                let pcm = read_pcm_file(&pcmp);
                let take = (ENC_FRAMES * 160).min(pcm.len() / 160 * 160);
                if take >= 160 {
                    // Pure-tone clips: the AMBE+2 streamer's per-frame voicing
                    // can't reproduce the whole-buffer global lag search, so its
                    // LiveEncoder check is tone-tolerant (speech stays strict).
                    let strict = !name.contains("tone");
                    assert_encode_parity_opt(rate, &pcm[..take], "corpus", strict);
                    encode_checked += 1;
                }
            }
        }
    }
    assert!(
        decode_checked >= 4 && encode_checked >= 4,
        "corpus present but too few vectors matched (decode {decode_checked}, encode {encode_checked})"
    );
    eprintln!(
        "corpus_representative_parity_all_rates: {decode_checked} decode + {encode_checked} \
         encode vector×rate checks passed at 0 diff (representative; full sweep = \
         codec_acceptance_sweep example)."
    );
}

/// The enhancement chain's boundary fade must actually fire on a
/// `Repeat`/`Mute` -> `Use` transition.
///
/// This drives the shipped `decode_bits` path on purpose. `enhancement.rs`'s
/// own unit test calls `enhancement::apply` directly, so it cannot prove that
/// `decode_bits` threads the real `FrameDisposition` through as `prev_was_use`
/// rather than hardcoding it — a unit test that bypasses the caller cannot
/// prove the caller wired it up.
///
/// Concealment is driven by forcing `b0` into the erasure range [120,123],
/// which is deterministic and needs no bit-error model. Hostile bytes do NOT
/// work: an all-`0xFF` or all-zero frame decodes as `Use`, because the FEC
/// corrects it into a valid codeword.
///
/// Mute returns before the voice state advances, so the frame after
/// concealment decodes from identical decoder state — which is what makes this
/// a clean isolation of the fade.
#[test]
fn enhancement_boundary_fade_fires_after_concealment() {
    let rate = Rate::AmbePlus2_3600x2450;
    let n = rate.fec_frame_bytes();
    let pcm = synthetic_pcm();
    let good: Vec<u8> = codec_direct_encode(rate, &pcm).concat();
    assert!(good.len() >= n * 6);

    // A frame whose b0 sits in the erasure range -> repeat, then mute.
    let erasure = {
        let mut b =
            blip25_codec::tables::deprioritize(&blip25_codec::frame::decode_bytes(&good[..n]).info);
        b[0] = 121;
        blip25_codec::enc::encode_frame::encode_r33(&b).to_vec()
    };

    // Isolate the fade by holding concealment IDENTICAL and varying only
    // `boundary_fade_samples`. Comparing "concealed" against "not concealed"
    // does NOT isolate it: `Repeat` re-decodes the last good frame and advances
    // the OLA ring, gain register and predictor, so the two runs decode the
    // following frame from different state — and that comparison makes the
    // concealed frame LOUDER, not quieter.
    let run = |fade: usize| -> (Vec<i16>, blip25_mbe::vocoder::FrameDisposition) {
        let mut v = Vocoder::new(rate);
        v.set_enhancement(EnhancementMode::Classical(ClassicalConfig {
            biquads: [None, None],
            compressor: None,
            boundary_fade_samples: fade,
            output_gain_db: 0.0,
        }));
        for i in 0..4 {
            v.decode_bits(&good[i * n..(i + 1) * n]).unwrap();
        }
        for _ in 0..6 {
            v.decode_bits(&erasure).unwrap();
        }
        let out = v.decode_bits(&good[4 * n..5 * n]).unwrap();
        let disp = v
            .last_stats()
            .decode
            .as_ref()
            .expect("decode stats")
            .disposition;
        (out, disp)
    };

    let (unfaded, disp) = run(0);
    let (faded, _) = run(80);

    assert_eq!(
        disp,
        blip25_mbe::vocoder::FrameDisposition::Use,
        "the frame after concealment should decode normally"
    );
    assert_ne!(
        unfaded, faded,
        "boundary fade did not fire: the frame following Mute is identical \
         with the fade on and off, so `prev_was_use` is not being threaded \
         from the real disposition"
    );
    let head = |v: &[i16]| -> i64 { v[..80].iter().map(|&s| i64::from(s).abs()).sum() };
    assert!(
        head(&faded) < head(&unfaded),
        "faded frame head should be attenuated: {} vs {}",
        head(&faded),
        head(&unfaded)
    );
    // Past the fade window the two must agree exactly — the fade is a head
    // taper, not a gain change.
    assert_eq!(
        &faded[80..],
        &unfaded[80..],
        "fade leaked past boundary_fade_samples"
    );
}
