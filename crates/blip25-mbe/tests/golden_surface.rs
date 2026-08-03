//! Pinned output for the shipped public surface that `golden_output.rs`
//! never reaches.
//!
//! `golden_output.rs` pins three encode entry points and one decode entry
//! point. Six other shipped surfaces produce audio or wire bytes that no
//! literal in this tree constrains, so any of them could change output
//! silently:
//!
//! 1. [`Vocoder::decode_soft`] — the soft-decision front end, in both the
//!    regime where it must agree with the hard path and the regime where it
//!    must not.
//! 2. [`Transcoder`] — every accepted `(from, to)` pair, including the two
//!    cross-codec parameter-domain directions, which were guarded by nothing
//!    but "does not panic".
//! 3. Annex-T tone frames through the decoder — decoded tone *audio*.
//! 4. [`EnhancementMode::Classical`], including the soft-knee compressor
//!    branch, which no other test in the tree executes.
//! 5. [`LiveDecoder`] at awkward chunk sizes.
//! 6. `flush_encode` — the tail frame of every transmission, which the
//!    `GOLDEN` encode column drops on the floor.
//!
//! The conventions are `golden_output.rs`'s, deliberately: hand-inlined
//! FNV-1a so no dependency can move a hash, comparison against literals, a
//! placeholder-zero check so an unpinned gate cannot read as passing, and a
//! re-bless block printed under `--nocapture`.
//!
//! Every generator here is a pure integer function of a fixed seed — no
//! libm, no floats — so the *input* is bit-identical on every target and
//! only the codec is under test. The generators are local to this file and
//! pinned by `test_inputs_are_platform_stable`; they deliberately do not
//! share code with `golden_output.rs`, so a change here can never force a
//! re-bless of the long-standing constants there.
//!
//! **If a change to the codec makes this fail, that is the test working.**

use blip25_mbe::enhancement::{ClassicalConfig, Compressor, EnhancementMode};
use blip25_mbe::vocoder::{LiveDecoder, Rate, Transcoder, Vocoder, VocoderError};
use blip25_mbe::{imbe7200, rate33};

const ALL_RATES: [Rate; 4] = [
    Rate::Imbe7200x4400,
    Rate::Imbe4400x4400,
    Rate::AmbePlus2_3600x2450,
    Rate::AmbePlus2_2450x2450,
];

/// The two FEC-bearing rates — the only ones with a soft-decision layer.
const FEC_RATES: [Rate; 2] = [Rate::Imbe7200x4400, Rate::AmbePlus2_3600x2450];

/// The two AMBE+2 rates — the only ones carrying Annex-T tone frames.
const AMBE_RATES: [Rate; 2] = [Rate::AmbePlus2_3600x2450, Rate::AmbePlus2_2450x2450];

fn rate_key(r: Rate) -> &'static str {
    match r {
        Rate::Imbe7200x4400 => "imbe_7200x4400",
        Rate::Imbe4400x4400 => "imbe_4400x4400",
        Rate::AmbePlus2_3600x2450 => "ambe2_3600x2450",
        Rate::AmbePlus2_2450x2450 => "ambe2_2450x2450",
        _ => "unknown",
    }
}

/// FNV-1a 64. Defined here rather than pulled in so the gate has no
/// dependency that could itself change the hash.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn hash_i16(samples: &[i16]) -> u64 {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    fnv1a(&bytes)
}

fn hash_i8(values: &[i8]) -> u64 {
    let bytes: Vec<u8> = values.iter().map(|&v| v as u8).collect();
    fnv1a(&bytes)
}

// ── Deterministic integer generators ──────────────────────────────────────

/// xorshift64*, fully specified integer arithmetic. Seeded per generator so
/// the several inputs in this file are independent of each other.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // A zero state is a fixed point of xorshift; no seed here is zero,
        // but pin the invariant rather than trusting the literals.
        assert_ne!(seed, 0, "xorshift64 seed must be non-zero");
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_u16(&mut self) -> u16 {
        (self.next_u64() >> 48) as u16
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }
}

/// Deterministic speech-like PCM: three summed integer triangle waves with a
/// per-frame envelope plus a reproducible noise floor, so both the voiced and
/// the unvoiced paths are driven. Distinct from `golden_output.rs`'s signal
/// (different periods, envelope period and seed) so the two files' pins stay
/// independent.
fn surface_pcm(frames: usize) -> Vec<i16> {
    let n = frames * 160;
    let mut out = Vec::with_capacity(n);
    let mut rng = Rng::new(0x1234_5678_9ABC_DEF0);
    for i in 0..n {
        let noise = ((rng.next_u64() >> 49) as i32) - 16384 / 2;

        // Integer triangle waves at three periods — deterministic, no libm.
        let tri = |period: i32| -> i32 {
            let p = (i as i32) % period;
            let half = period / 2;
            if p < half {
                p * 2 - half
            } else {
                (period - p) * 2 - half
            }
        };
        let env = 2 + ((i / 160) % 7) as i32; // 2..8, varies per frame
        let v = (tri(61) * 70 + tri(29) * 55 + tri(13) * 25) * env / 8 + noise / 20;
        out.push(v.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
    }
    out
}

/// Pack a dibit stream MSB-first into bytes: dibit `i`'s high bit is frame
/// bit `2i`, its low bit frame bit `2i+1`.
fn pack_dibits<const N: usize, const B: usize>(dibits: &[u8; N]) -> [u8; B] {
    let mut out = [0u8; B];
    let mut bit = 0usize;
    for &d in dibits {
        for pos in (0..2).rev() {
            out[bit / 8] |= ((d >> pos) & 1) << (7 - (bit % 8));
            bit += 1;
        }
    }
    out
}

/// Wire frames at `rate`, from a fixed integer sequence.
///
/// For the two FEC-bearing rates, random info vectors are run through this
/// crate's own wire encoder, so every frame is an **error-free codeword**.
/// That is what makes the clean-LLR equality assertion below a statement
/// about the soft FEC layer rather than about two decoders happening to
/// miscorrect the same way. For the two no-FEC rates every byte string is
/// already a valid frame (they carry packed info bits and nothing else), so
/// the sequence is used directly.
fn wire_frames(rate: Rate, frames: usize) -> Vec<u8> {
    let mut rng = Rng::new(0x0BAD_C0DE_1357_9BDF);
    let mut out = Vec::with_capacity(rate.fec_frame_bytes() * frames);
    for _ in 0..frames {
        match rate {
            Rate::Imbe7200x4400 => {
                let mut info = [0u16; 8];
                for slot in &mut info {
                    *slot = rng.next_u16();
                }
                let dibits = imbe7200::frame::encode_frame(&info);
                out.extend_from_slice(&pack_dibits::<72, 18>(&dibits));
            }
            Rate::AmbePlus2_3600x2450 => {
                let mut info = [0u16; 4];
                for slot in &mut info {
                    *slot = rng.next_u16();
                }
                let dibits = rate33::frame::encode_frame(&info);
                out.extend_from_slice(&pack_dibits::<36, 9>(&dibits));
            }
            _ => {
                for _ in 0..rate.fec_frame_bytes() {
                    out.push(rng.next_u8());
                }
            }
        }
    }
    out
}

/// One LLR per FEC channel bit, in raw frame-bit order (MSB-first over the
/// bytes of [`wire_frames`]) — the order `decode_soft` documents.
///
/// Sign is the hard decision (`>0` → 1), magnitude the confidence. Two
/// regimes:
///
/// * `flip_every == 0` — every LLR agrees with the true bit at a strong,
///   graded magnitude (33..=127). Graded rather than uniform so the
///   least-reliable ranking inside Chase-II is well defined instead of
///   depending on a sort's tie-break.
/// * `flip_every == k` — every `k`-th LLR has its sign inverted, and *every*
///   LLR takes a magnitude from the full 1..=127 range. Spreading confidence
///   over correct and errored bits alike is what makes this regime a real
///   test of the Chase search: the least-reliable set is then a mix, so the
///   search depth, the ranking and the soft-distance tie-break all decide
///   the outcome. (Giving only the errored bits low confidence instead makes
///   the weak set exactly the error set, and the search stops depending on
///   its own depth — that version of this generator could not tell
///   `CHASE_GOLAY_DEPTH = 6` from `5`.)
fn llrs_for(rate: Rate, frames: usize, flip_every: usize) -> Vec<i8> {
    let bits_per_frame = rate
        .soft_frame_bits()
        .expect("llrs_for is only meaningful for a FEC-bearing rate");
    let hard = wire_frames(rate, frames);
    let mut rng = Rng::new(0x7E57_C0DE_2468_ACE1);
    let mut out = Vec::with_capacity(bits_per_frame * frames);
    for i in 0..bits_per_frame * frames {
        let r = rng.next_u64();
        let bit = (hard[i / 8] >> (7 - (i % 8))) & 1;
        let noisy = flip_every != 0;
        let flip = noisy && i % flip_every == 2;
        let magnitude = if noisy {
            1 + (r % 127) as i8
        } else {
            33 + (r % 95) as i8
        };
        // `flip` inverts the hard decision the LLR carries.
        out.push(if (bit == 1) != flip {
            magnitude
        } else {
            -magnitude
        });
    }
    out
}

/// The Annex-T tone program driven through the decoder.
///
/// Ordered to exercise the decoder's tone tracker, not just the tone table:
/// a four-frame burst of one ID (the confirm counter and the burst-start
/// onset-phase seeding only engage on repeats), then single frames at other
/// valid IDs, then reserved IDs with no `ANNEX_T` row.
const TONE_PROGRAM: [(u8, u8); 13] = [
    (60, 100), // burst — 4 frames of the same ID
    (60, 100),
    (60, 100),
    (60, 100),
    (5, 127), // lowest valid ID, full amplitude
    (32, 96),
    (120, 64),
    (163, 110), // highest non-sentinel valid ID
    (60, 20),   // same ID as the burst, much quieter
    (60, 20),
    (0, 100),   // reserved — no ANNEX_T row
    (200, 100), // reserved
    (250, 30),  // reserved
];

/// [`TONE_PROGRAM`] as wire frames at an AMBE+2 rate.
fn tone_frames(rate: Rate) -> Vec<u8> {
    let mut out = Vec::with_capacity(rate.fec_frame_bytes() * TONE_PROGRAM.len());
    for &(id, amplitude) in TONE_PROGRAM.iter() {
        let info = rate33::dequantize::encode_tone_frame_info(id, amplitude);
        match rate {
            Rate::AmbePlus2_3600x2450 => {
                let dibits = rate33::frame::encode_frame(&info);
                out.extend_from_slice(&pack_dibits::<36, 9>(&dibits));
            }
            Rate::AmbePlus2_2450x2450 => {
                out.extend_from_slice(&rate33::frame::pack_no_fec(&info));
            }
            other => panic!("{} carries no Annex-T tone frames", rate_key(other)),
        }
    }
    out
}

fn decode_all(rate: Rate, bits: &[u8]) -> Vec<i16> {
    let mut v = Vocoder::new(rate);
    let n = rate.fec_frame_bytes();
    let mut all = Vec::new();
    for frame in bits.chunks_exact(n) {
        all.extend(v.decode_bits(frame).expect("decode a full frame"));
    }
    all
}

const FRAMES: usize = 20;

// ── 1. Vocoder::decode_soft ───────────────────────────────────────────────
//
// Two hashes per FEC-bearing rate:
//   clean — strong LLRs over error-free codewords. Also asserted equal to
//           the hard decode of the same frames, which is the stronger claim:
//           the soft path must be a no-op when there is nothing to rescue.
//   noisy — one LLR in five sign-flipped, with confidence spread over the
//           whole range. This is the only pin in the tree over the Chase-II
//           search: the `CHASE_GOLAY_DEPTH` / `CHASE_HAMMING_DEPTH`
//           constants, the least-reliable ranking, and the soft-distance
//           tie-break all move it, and move nothing in golden_output.rs.
//
// Measured sensitivity of the noisy column to `CHASE_GOLAY_DEPTH` (shipped
// value 6), because it is not uniform and the next person to touch that
// constant should not have to rediscover it: the IMBE column moves for every
// depth in 0..=5, the AMBE+2 column moves for 0..=4 and again at 7. Neither
// moves between 6 and 7-or-more for IMBE — the search has saturated, i.e.
// raising the depth buys nothing on this input and costs candidates. The
// depth-5 case is the reason `SOFT_FRAMES` is 64 rather than `FRAMES`: at 20
// frames the 6th least-reliable position never changed a winner in the whole
// run, and a 6 -> 5 mutation passed silently.
//
// (rate key, clean, noisy)
const GOLDEN_SOFT: [(&str, u64, u64); 2] = [
    ("imbe_7200x4400", 0xdc9a300d3d9e1786, 0xa0fa4b76a726a493),
    ("ambe2_3600x2450", 0x7a66009b1b4d6d96, 0xe7c3009749be8533),
];

/// One in `SOFT_FLIP_EVERY` channel bits is delivered with the wrong sign.
const SOFT_FLIP_EVERY: usize = 5;

/// Frames in the soft-decision runs. Longer than [`FRAMES`] on purpose: the
/// Chase-II search saturates quickly, and whether the *k*-th least-reliable
/// position ever changes a winner is a question about how many codewords the
/// run decodes. See the note on `CHASE_GOLAY_DEPTH` sensitivity below.
const SOFT_FRAMES: usize = 64;

#[test]
fn soft_decode_is_pinned() {
    let mut actual = Vec::new();
    let mut mismatches = Vec::new();

    for rate in FEC_RATES {
        let key = rate_key(rate);
        let n = rate
            .soft_frame_bits()
            .expect("FEC rate has a soft frame size");

        // (a) Clean regime. The soft path must reproduce the hard path
        // bit-for-bit — anything else means the soft FEC is inventing
        // corrections on error-free input.
        let clean_llrs = llrs_for(rate, SOFT_FRAMES, 0);
        let mut soft_rx = Vocoder::new(rate);
        let mut clean_pcm = Vec::new();
        for frame in clean_llrs.chunks_exact(n) {
            clean_pcm.extend(soft_rx.decode_soft(frame).expect("soft decode a frame"));
        }
        let hard_pcm = decode_all(rate, &wire_frames(rate, SOFT_FRAMES));
        assert_eq!(
            clean_pcm, hard_pcm,
            "{key}: decode_soft on strong, correct LLRs diverged from \
             decode_bits on the same error-free frames. The soft path is \
             documented to be bit-identical here and to differ only by \
             rescuing frames a hard decode would corrupt."
        );

        // (b) Noisy regime — what soft decoding is for.
        let noisy_llrs = llrs_for(rate, SOFT_FRAMES, SOFT_FLIP_EVERY);
        let mut noisy_rx = Vocoder::new(rate);
        let mut noisy_pcm = Vec::new();
        for frame in noisy_llrs.chunks_exact(n) {
            noisy_pcm.extend(noisy_rx.decode_soft(frame).expect("soft decode a frame"));
        }
        assert_ne!(
            noisy_pcm, clean_pcm,
            "{key}: sign-flipped LLRs decoded to exactly the clean audio. \
             Either the flips are not reaching the decoder or the soft input \
             is being ignored — either way this gate would be vacuous."
        );

        actual.push((key, hash_i16(&clean_pcm), hash_i16(&noisy_pcm)));

        let want = GOLDEN_SOFT
            .iter()
            .find(|(k, _, _)| *k == key)
            .map(|(_, c, e)| (*c, *e))
            .unwrap_or_else(|| panic!("no GOLDEN_SOFT row for rate {key}"));
        for (label, got, want) in [
            ("clean", hash_i16(&clean_pcm), want.0),
            ("noisy", hash_i16(&noisy_pcm), want.1),
        ] {
            if want != 0 && got != want {
                mismatches.push(format!(
                    "{key} soft/{label}: expected {want:#018x}, got {got:#018x}"
                ));
            }
        }
    }

    // The no-FEC rates have no soft layer. They must say so, loudly, rather
    // than being quietly skipped by whoever enumerates rates next.
    for rate in [Rate::Imbe4400x4400, Rate::AmbePlus2_2450x2450] {
        assert_eq!(
            rate.soft_frame_bits(),
            None,
            "{}: a no-FEC rate reports a soft frame size",
            rate_key(rate)
        );
        let mut rx = Vocoder::new(rate);
        let err = rx
            .decode_soft(&[64i8; 144])
            .expect_err("no-FEC rate must reject soft input");
        assert!(
            matches!(err, VocoderError::SoftUnsupported { rate: r } if r == rate),
            "{}: expected SoftUnsupported, got {err:?}",
            rate_key(rate)
        );
    }

    println!("\n// Re-bless by copying this block into GOLDEN_SOFT:");
    println!("const GOLDEN_SOFT: [(&str, u64, u64); 2] = [");
    for (k, c, e) in &actual {
        println!("    (\"{k}\", {c:#018x}, {e:#018x}),");
    }
    println!("];\n");

    assert!(
        GOLDEN_SOFT.iter().all(|(_, c, e)| *c != 0 && *e != 0),
        "GOLDEN_SOFT still holds placeholder zeros — an unpinned gate reads \
         as passing, which is worse than no gate."
    );
    assert!(
        mismatches.is_empty(),
        "soft-decode output changed from the pinned values:\n  {}",
        mismatches.join("\n  ")
    );
}

// ── 2. Transcoder ─────────────────────────────────────────────────────────
//
// The accepted `(from, to)` set is pinned as well as the bytes, so a pair
// silently appearing or disappearing from `Transcoder::new` fails here
// instead of shipping. The two cross-codec entries run the full
// parameter-domain converter (extract MBE params from one codec's bits,
// re-quantize into the other's) with no PCM detour; before this file their
// only guard was "does not panic".
const TRANSCODE_PAIRS: [&str; 6] = [
    "imbe_7200x4400->imbe_4400x4400",
    "imbe_7200x4400->ambe2_3600x2450",
    "imbe_4400x4400->imbe_7200x4400",
    "ambe2_3600x2450->imbe_7200x4400",
    "ambe2_3600x2450->ambe2_2450x2450",
    "ambe2_2450x2450->ambe2_3600x2450",
];

const GOLDEN_TRANSCODE: [(&str, u64); 6] = [
    ("imbe_7200x4400->imbe_4400x4400", 0x043c7920743caead),
    ("imbe_7200x4400->ambe2_3600x2450", 0x04c0cb81d1ee1606),
    ("imbe_4400x4400->imbe_7200x4400", 0xe626f3e03bae8b29),
    ("ambe2_3600x2450->imbe_7200x4400", 0xda56738d806e4555),
    ("ambe2_3600x2450->ambe2_2450x2450", 0x54a93d82533e8c3d),
    ("ambe2_2450x2450->ambe2_3600x2450", 0xcdfad3c6f2982de4),
];

fn pair_key(from: Rate, to: Rate) -> String {
    format!("{}->{}", rate_key(from), rate_key(to))
}

#[test]
fn transcoder_output_is_pinned() {
    let mut accepted = Vec::new();
    let mut actual = Vec::new();
    let mut mismatches = Vec::new();

    for from in ALL_RATES {
        for to in ALL_RATES {
            let key = pair_key(from, to);
            let mut tx = match Transcoder::new(from, to) {
                Ok(t) => t,
                Err(_) => continue,
            };
            accepted.push(key.clone());

            assert_eq!(tx.direction().from, from);
            assert_eq!(tx.direction().to, to);
            assert_eq!(tx.direction().input_frame_bytes(), from.fec_frame_bytes());
            assert_eq!(tx.direction().output_frame_bytes(), to.fec_frame_bytes());

            let input = wire_frames(from, FRAMES);
            let mut out = Vec::new();
            for frame in input.chunks_exact(from.fec_frame_bytes()) {
                let bytes = tx.transcode(frame).expect("transcode a full frame");
                assert_eq!(
                    bytes.len(),
                    to.fec_frame_bytes(),
                    "{key}: wrong output frame size"
                );
                out.extend_from_slice(&bytes);
            }
            assert_eq!(out.len(), to.fec_frame_bytes() * FRAMES);
            actual.push((key.clone(), fnv1a(&out)));

            let want = GOLDEN_TRANSCODE
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, h)| *h)
                .unwrap_or_else(|| panic!("no GOLDEN_TRANSCODE row for pair {key}"));
            if want != 0 && fnv1a(&out) != want {
                mismatches.push(format!(
                    "{key}: expected {want:#018x}, got {:#018x}",
                    fnv1a(&out)
                ));
            }

            // Wrong input length is an error, not a panic or a short read.
            assert!(tx.transcode(&[]).is_err(), "{key}: empty input accepted");
            assert!(
                tx.transcode(&vec![0u8; from.fec_frame_bytes() + 1])
                    .is_err(),
                "{key}: oversized input accepted"
            );
        }
    }

    println!("\n// Re-bless by copying these blocks into TRANSCODE_PAIRS / GOLDEN_TRANSCODE:");
    println!("const TRANSCODE_PAIRS: [&str; {}] = [", accepted.len());
    for k in &accepted {
        println!("    \"{k}\",");
    }
    println!("];");
    println!(
        "const GOLDEN_TRANSCODE: [(&str, u64); {}] = [",
        actual.len()
    );
    for (k, h) in &actual {
        println!("    (\"{k}\", {h:#018x}),");
    }
    println!("];\n");

    assert_eq!(
        accepted, TRANSCODE_PAIRS,
        "the set of (from, to) pairs Transcoder::new accepts changed. A pair \
         appearing or disappearing is an API change; re-bless only if that \
         was the intent."
    );
    assert!(
        GOLDEN_TRANSCODE.iter().all(|(_, h)| *h != 0),
        "GOLDEN_TRANSCODE still holds placeholder zeros — an unpinned gate \
         reads as passing, which is worse than no gate."
    );
    assert!(
        mismatches.is_empty(),
        "transcoder output changed from the pinned values:\n  {}\n\n\
         For the two cross-codec directions this is the parameter-domain \
         converter (extraction, re-quantization, cross-rate predictor); for \
         the four same-codec directions it is pure wire-layer bit shuffling \
         and any movement is a wire-format break.",
        mismatches.join("\n  ")
    );
}

// ── 3. Decoded Annex-T tone audio ─────────────────────────────────────────
//
// Tone *classification* is covered elsewhere; the synthesized tone audio is
// not pinned anywhere else in the tree. The program includes a repeated-ID
// burst (the tracker's confirm counter plus the burst-start onset-phase
// seeding) and reserved IDs with no ANNEX_T row.
// Moved 0x64bb…f5c2 -> 0x6379…58b7 when `tone_to_mbe_params` took L from the
// highest lit harmonic instead of Eq. 207's `floor(3812.5/f0)`. Verified frame
// by frame against the old rule: 11 of the 13 program frames are
// **bit-identical**, and the change is confined to frame 7 (I_D 163, where L
// goes 54 -> 9 — the program's largest) plus frame 8, which inherits its
// synthesis state. Level moves 0.04% and peak 0.6%: a waveform/phase
// difference, not a level or content change. Across all 256 IDs the new rule
// changes L on 120 of them and changes *which components are lit* on none.
// Moved again when tone frames were routed through the shared fixed-point
// engine instead of the float `synth`, so they overlap-add onto the carried
// output remainder rather than starting from a cold state. A different
// synthesizer, so a wholly different hash is expected. Measured against the
// the reference on off-air material:
//
//   * isolated mid-word tone frame (I_D 12 at t=17.10s): the 56-sample hole
//     at the transition is gone (leading zeros 56 -> 0), frame RMS 385 -> 737
//     against the reference's 765, the displaced tail at the next frame 966 -> 328
//     against 422, and the following frame's phase inversion (ncc -0.80)
//     becomes +0.976. Whole-call mean |RMS error| vs the reference 7.4 -> 1.98.
//   * sustained bursts: level now within ~1% of the reference (1399 vs 1386) where it
//     ran ~5% low, no collapse; that call's mean |RMS error| 14.7 -> 7.21.
//
// Residual is a constant phase offset on tones (ncc ~-0.5 at steady state with
// level and spectrum correct) — absolute tone phase is context-derived and
// still un-reverse-engineered. Inaudible; do not chase it by moving levels.
const GOLDEN_TONE: [(&str, u64); 2] = [
    ("ambe2_3600x2450", 0xcf39eefbabd89c49),
    ("ambe2_2450x2450", 0xcf39eefbabd89c49),
];

#[test]
fn decoded_tone_audio_is_pinned() {
    // Every frame the program builds must really be a tone frame on the
    // wire. Checked against the engine's own classifier, not this crate's,
    // so the two cannot drift into agreeing on something wrong.
    for &(id, amplitude) in TONE_PROGRAM.iter() {
        let info = rate33::dequantize::encode_tone_frame_info(id, amplitude);
        assert_eq!(
            blip25_codec::tone::classify(&info),
            blip25_codec::tone::FrameKind::Tone,
            "tone id {id} amp {amplitude} does not classify as Tone — the \
             frame would decode as voice or erasure and this gate would be \
             pinning the wrong thing entirely"
        );
    }

    let mut actual = Vec::new();
    let mut mismatches = Vec::new();
    let mut tone_audio: Vec<Vec<i16>> = Vec::new();

    for rate in AMBE_RATES {
        let key = rate_key(rate);
        let frames = tone_frames(rate);
        assert_eq!(frames.len(), rate.fec_frame_bytes() * TONE_PROGRAM.len());
        let pcm = decode_all(rate, &frames);
        assert_eq!(pcm.len(), 160 * TONE_PROGRAM.len());

        // A silent pass would be worthless: assert there is audio, and that
        // the burst specifically is loud, before hashing anything.
        let peak = pcm.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        assert!(
            peak > 1_000,
            "{key}: decoded tone program peaks at {peak} — that is silence, \
             so the pinned hash below would be guarding nothing"
        );
        let burst = &pcm[..160 * 4];
        let burst_peak = burst.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        assert!(
            burst_peak > 1_000,
            "{key}: the repeated-ID burst peaks at {burst_peak} — the tone \
             tracker never confirmed the tone"
        );

        tone_audio.push(pcm.clone());
        actual.push((key, hash_i16(&pcm)));
        let want = GOLDEN_TONE
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, h)| *h)
            .unwrap_or_else(|| panic!("no GOLDEN_TONE row for rate {key}"));
        if want != 0 && hash_i16(&pcm) != want {
            mismatches.push(format!(
                "{key} tone: expected {want:#018x}, got {:#018x}",
                hash_i16(&pcm)
            ));
        }
    }

    // Structural invariant, stated independently of the literals so a
    // regression fails with a diagnosis instead of an opaque hash mismatch.
    //
    // The two rates carry the same 49 info bits in different wire layouts,
    // and the r34 decode path detours through the parameter vector
    // (`decode_r34` → b → `encode_r33`). Identical audio is therefore a
    // statement that the prioritize/deprioritize round-trip is a bijection
    // on the Annex-T signature bits specifically — the one place where a
    // lossy round-trip would silently turn a tone frame into voice or
    // erasure.
    assert_eq!(
        tone_audio[0], tone_audio[1],
        "the two AMBE+2 rates decoded the same tone program to different \
         audio. The r34 wire layout or the prioritize/deprioritize round-trip \
         is dropping part of the tone signature."
    );

    println!("\n// Re-bless by copying this block into GOLDEN_TONE:");
    println!("const GOLDEN_TONE: [(&str, u64); 2] = [");
    for (k, h) in &actual {
        println!("    (\"{k}\", {h:#018x}),");
    }
    println!("];\n");

    assert!(
        GOLDEN_TONE.iter().all(|(_, h)| *h != 0),
        "GOLDEN_TONE still holds placeholder zeros — an unpinned gate reads \
         as passing, which is worse than no gate."
    );
    assert!(
        mismatches.is_empty(),
        "decoded tone audio changed from the pinned values:\n  {}\n\n\
         Movement here means the Annex-T table, the tone→MBE bridge, the \
         onset-phase seeding or the tone tracker moved. Judge by ear before \
         re-blessing.",
        mismatches.join("\n  ")
    );
}

// ── 4. EnhancementMode::Classical ─────────────────────────────────────────
//
// Two configurations per rate:
//   classical  — ClassicalConfig::default(): the two biquads and the
//                boundary fade, compressor off.
//   compressed — the same plus a Compressor, which is the only way to reach
//                the soft-knee DRC branch in enhancement::apply. Nothing
//                else in the tree executes it.
//
// Both are asserted different from the default (None) output, so a chain
// that silently became a pass-through fails with a diagnosis rather than a
// re-blessable hash.
//
// Both columns depend on libm: the biquad coefficients and the soft-knee
// compressor go through f32 `sin`, `cos`, `powf`, `log10` and `exp`. The
// inputs are integer-only, so the codec stays under test — but these two
// values are a function of the host's libm as well as of this tree, and a
// libm difference can move them with nothing here having changed.
//
// (rate key, classical, compressed)
// Every AMBE+2 literal in this file moved once, together, when the Annex-T
// redundancy gates went in: `wire_frames` builds frames from random info
// vectors, so roughly one frame in 64 lands on the `û₀(11..6) == 0x3F` tone
// signature by chance. Those used to be decoded as tones — a spurious beep
// out of garbage — and now fail the I_D copy gates (four random copies do not
// agree within the 6-bit tolerance) and demote to erasure, which repeats the
// previous frame instead. IMBE literals are untouched: it has no tone path.
const GOLDEN_ENHANCEMENT: [(&str, u64, u64); 4] = [
    // The IMBE rows carry the post-concealment fade. `wire_frames` builds its
    // stream from random info vectors, so ~19% of IMBE frames land outside the
    // valid pitch range and decode as erasures; the fade in `enhancement::apply`
    // arms on the first `Use` frame after each of those runs.
    ("imbe_7200x4400", 0x57333699d89bb564, 0x09217a389cd416f1),
    ("imbe_4400x4400", 0x9c9de7f9212f9f0b, 0x08633cfc8ff4d6d5),
    ("ambe2_3600x2450", 0x7c1a72d8369b6a58, 0x7d8ec558ec9a8424),
    ("ambe2_2450x2450", 0x574e01aa1880b831, 0x874c387803ea20ba),
];

/// Reaches the soft-knee compressor branch.
///
/// The threshold is set well under the decoded level on purpose. Above it the
/// gain-reduction arithmetic runs; below it the branch collapses to unity gain
/// and only the makeup multiply survives, which the pin cannot tell from no
/// compressor at all. At -24 dBFS three of the four rates crossed it and
/// `ambe2_3600x2450` did not, so a mutation to the over-threshold expression
/// moved only three of the four hashes; -40 dBFS is crossed at every rate.
fn compressor_config() -> ClassicalConfig {
    ClassicalConfig {
        compressor: Some(Compressor {
            threshold_db: -40.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 200.0,
            makeup_db: 6.0,
        }),
        ..ClassicalConfig::default()
    }
}

fn decode_with_enhancement(rate: Rate, bits: &[u8], mode: EnhancementMode) -> Vec<i16> {
    let mut v = Vocoder::builder(rate).enhancement(mode).build();
    let n = rate.fec_frame_bytes();
    let mut all = Vec::new();
    for frame in bits.chunks_exact(n) {
        all.extend(v.decode_bits(frame).expect("decode a full frame"));
    }
    all
}

#[test]
fn enhancement_output_is_pinned() {
    let mut actual = Vec::new();
    let mut mismatches = Vec::new();

    for rate in ALL_RATES {
        let key = rate_key(rate);
        let bits = wire_frames(rate, FRAMES);

        let plain = decode_with_enhancement(rate, &bits, EnhancementMode::None);
        let classical = decode_with_enhancement(
            rate,
            &bits,
            EnhancementMode::Classical(ClassicalConfig::default()),
        );
        let compressed =
            decode_with_enhancement(rate, &bits, EnhancementMode::Classical(compressor_config()));

        assert_eq!(
            plain,
            decode_all(rate, &bits),
            "{key}: EnhancementMode::None is not the untouched decode path"
        );
        assert_ne!(
            classical, plain,
            "{key}: EnhancementMode::Classical produced the same audio as \
             None. The chain has become a pass-through and the pin below \
             would be guarding the plain decoder twice."
        );
        assert_ne!(
            compressed, classical,
            "{key}: adding a Compressor changed nothing, so the soft-knee \
             DRC branch in enhancement::apply is not being executed"
        );

        actual.push((key, hash_i16(&classical), hash_i16(&compressed)));
        let want = GOLDEN_ENHANCEMENT
            .iter()
            .find(|(k, _, _)| *k == key)
            .map(|(_, c, d)| (*c, *d))
            .unwrap_or_else(|| panic!("no GOLDEN_ENHANCEMENT row for rate {key}"));
        for (label, got, want) in [
            ("classical", hash_i16(&classical), want.0),
            ("compressed", hash_i16(&compressed), want.1),
        ] {
            if want != 0 && got != want {
                mismatches.push(format!(
                    "{key} {label}: expected {want:#018x}, got {got:#018x}"
                ));
            }
        }
    }

    println!("\n// Re-bless by copying this block into GOLDEN_ENHANCEMENT:");
    println!("const GOLDEN_ENHANCEMENT: [(&str, u64, u64); 4] = [");
    for (k, c, d) in &actual {
        println!("    (\"{k}\", {c:#018x}, {d:#018x}),");
    }
    println!("];\n");

    assert!(
        GOLDEN_ENHANCEMENT
            .iter()
            .all(|(_, c, d)| *c != 0 && *d != 0),
        "GOLDEN_ENHANCEMENT still holds placeholder zeros — an unpinned gate \
         reads as passing, which is worse than no gate."
    );
    assert!(
        mismatches.is_empty(),
        "enhancement output changed from the pinned values:\n  {}\n\n\
         This is off the spec path by construction, so movement is not a \
         codec regression — but it is still a change in what callers who opt \
         in will hear.\n\n\
         CAVEAT before hunting for a regression: these two columns depend on \
         libm. The biquad and compressor chain calls f32 sin/cos/powf/log10/\
         exp, so a host whose libm rounds one of those differently moves them \
         with no change to this tree at all. Rule that out first: if the plain \
         decode pins over the same wire frames (GOLDEN_LIVE_DECODE) are green \
         and only these moved, the codec is reproducing bit-for-bit and the \
         difference is confined to the enhancement chain or the libm under it.",
        mismatches.join("\n  ")
    );
}

// ── 5. LiveDecoder ────────────────────────────────────────────────────────
//
// Chunk sizes chosen to straddle every frame size in the crate (7, 9, 11,
// 18): one byte at a time, a prime smaller than the smallest frame, a prime
// larger than the largest, and one bigger than the whole buffer. The
// concatenated output must be byte-identical to a whole-buffer decode at
// every one of them — the residue buffer must not be able to drop, duplicate
// or reorder a frame.
const LIVE_CHUNKS: [usize; 6] = [1, 5, 13, 23, 97, 4096];

const GOLDEN_LIVE_DECODE: [(&str, u64); 4] = [
    ("imbe_7200x4400", 0x85d137ef8f7c6ebc),
    ("imbe_4400x4400", 0x6140eac62499c673),
    ("ambe2_3600x2450", 0x2a8a99b9cbbaa5da),
    ("ambe2_2450x2450", 0x273f52cee3a6d801),
];

#[test]
fn live_decoder_output_is_pinned() {
    let mut actual = Vec::new();
    let mut mismatches = Vec::new();

    for rate in ALL_RATES {
        let key = rate_key(rate);
        let n = rate.fec_frame_bytes();
        // A deliberately ragged length: whole frames plus a maximal partial
        // one, so the residue is non-empty at the end of the run.
        let bits = {
            let mut b = wire_frames(rate, FRAMES);
            b.truncate(n * (FRAMES - 1) + (n - 1));
            b
        };
        let whole = decode_all(rate, &bits);
        assert_eq!(whole.len(), 160 * (FRAMES - 1));

        for &chunk in LIVE_CHUNKS.iter() {
            let mut dec = LiveDecoder::new(rate);
            assert_eq!(dec.rate(), rate);
            let mut out = Vec::new();
            for piece in bits.chunks(chunk) {
                for frame in dec.push(piece) {
                    out.extend(frame.expect("live decode a frame"));
                }
                assert!(
                    dec.pending_bytes() < n,
                    "{key}: {} bytes still buffered after push at chunk {chunk}",
                    dec.pending_bytes()
                );
            }
            assert_eq!(
                out, whole,
                "{key}: LiveDecoder at chunk size {chunk} does not match a \
                 whole-buffer decode_bits loop"
            );
            assert_eq!(
                dec.pending_bytes(),
                n - 1,
                "{key}: wrong residue at chunk size {chunk}"
            );
            dec.discard_pending();
            assert_eq!(dec.pending_bytes(), 0);
        }

        actual.push((key, hash_i16(&whole)));
        let want = GOLDEN_LIVE_DECODE
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, h)| *h)
            .unwrap_or_else(|| panic!("no GOLDEN_LIVE_DECODE row for rate {key}"));
        if want != 0 && hash_i16(&whole) != want {
            mismatches.push(format!(
                "{key} live: expected {want:#018x}, got {:#018x}",
                hash_i16(&whole)
            ));
        }
    }

    println!("\n// Re-bless by copying this block into GOLDEN_LIVE_DECODE:");
    println!("const GOLDEN_LIVE_DECODE: [(&str, u64); 4] = [");
    for (k, h) in &actual {
        println!("    (\"{k}\", {h:#018x}),");
    }
    println!("];\n");

    assert!(
        GOLDEN_LIVE_DECODE.iter().all(|(_, h)| *h != 0),
        "GOLDEN_LIVE_DECODE still holds placeholder zeros — an unpinned gate \
         reads as passing, which is worse than no gate."
    );
    assert!(
        mismatches.is_empty(),
        "live-decode output changed from the pinned values:\n  {}",
        mismatches.join("\n  ")
    );
}

// ── 6. flush_encode ───────────────────────────────────────────────────────
//
// The console codec holds a one-frame analysis look-ahead: `encode_pcm`
// emits an all-zero placeholder first and keeps the last real frame buffered
// until `flush_encode` drains it. `golden_output.rs`'s GOLDEN encode column
// never calls it, so the final real frame of every transmission — the frame
// a caller ships at end-of-PTT — is outside that hash entirely.
//
// (rate key, whole run including the flushed tail, tail alone)
const GOLDEN_FLUSH_ENCODE: [(&str, u64, u64); 4] = [
    // moved with the IMBE fricative-protect + per-band voicing veto (b1 V/UV
    // bits change). IMBE only; AMBE+2 rows below unmoved (default config).
    ("imbe_7200x4400", 0xd13c4a3b9a7275db, 0xbcb6521e0977324c),
    ("imbe_4400x4400", 0x990187dcec384a13, 0xe59723efb43477ca),
    // omega_0-refined STEP: amps_from_bins keys band_decompress STEP on the
    // continuous omega_0 count per L cell (step_for_omega). See golden_output.rs.
    ("ambe2_3600x2450", 0xbe0ff98723eb8dbc, 0x55813b1b4de75e6a),
    ("ambe2_2450x2450", 0x1aea4a268e485e2e, 0x8fbb002374930c63),
];

#[test]
fn flush_encode_tail_is_pinned() {
    let mut actual = Vec::new();
    let mut mismatches = Vec::new();

    let pcm = surface_pcm(FRAMES);

    for rate in ALL_RATES {
        let key = rate_key(rate);
        let n = rate.fec_frame_bytes();
        let mut v = Vocoder::new(rate);

        let mut run = Vec::new();
        for frame in pcm.chunks_exact(160) {
            let bytes = v.encode_pcm(frame).expect("encode a full frame");
            assert_eq!(bytes.len(), n, "{key}: wrong encoded frame size");
            run.extend_from_slice(&bytes);
        }

        let tail: Vec<u8> = v.flush_encode().concat();
        assert!(
            !tail.is_empty(),
            "{key}: flush_encode returned nothing. The encoder's look-ahead \
             frame is then either being dropped or was never buffered, and \
             the last real frame of every transmission is lost."
        );
        assert_eq!(
            tail.len() % n,
            0,
            "{key}: flushed tail is not a whole number of frames"
        );
        // Flushing twice must not invent frames.
        assert!(
            v.flush_encode().is_empty(),
            "{key}: a second flush_encode produced more frames"
        );

        run.extend_from_slice(&tail);

        // The look-ahead is exactly one frame deep, and these three
        // assertions are what make that observable rather than assumed: a
        // longer tail means frames are being held back, a shorter one means
        // the last real frame of the transmission is lost, and a tail that is
        // still the placeholder means the buffered frame was never encoded.
        assert_eq!(
            tail.len(),
            n,
            "{key}: flush_encode drained {} frames, not the single buffered \
             look-ahead frame",
            tail.len() / n
        );
        assert!(
            run[..n].iter().all(|&b| b == 0),
            "{key}: the first emitted frame is not the all-zero placeholder, \
             so the encoder is not running one frame behind its input"
        );
        assert!(
            tail.iter().any(|&b| b != 0),
            "{key}: the flushed tail is all zeros — it is another placeholder, \
             not the last real frame"
        );

        actual.push((key, fnv1a(&run), fnv1a(&tail)));
        let want = GOLDEN_FLUSH_ENCODE
            .iter()
            .find(|(k, _, _)| *k == key)
            .map(|(_, r, t)| (*r, *t))
            .unwrap_or_else(|| panic!("no GOLDEN_FLUSH_ENCODE row for rate {key}"));
        for (label, got, want) in [("run", fnv1a(&run), want.0), ("tail", fnv1a(&tail), want.1)] {
            if want != 0 && got != want {
                mismatches.push(format!(
                    "{key} flush/{label}: expected {want:#018x}, got {got:#018x}"
                ));
            }
        }
    }

    println!("\n// Re-bless by copying this block into GOLDEN_FLUSH_ENCODE:");
    println!("const GOLDEN_FLUSH_ENCODE: [(&str, u64, u64); 4] = [");
    for (k, r, t) in &actual {
        println!("    (\"{k}\", {r:#018x}, {t:#018x}),");
    }
    println!("];\n");

    assert!(
        GOLDEN_FLUSH_ENCODE
            .iter()
            .all(|(_, r, t)| *r != 0 && *t != 0),
        "GOLDEN_FLUSH_ENCODE still holds placeholder zeros — an unpinned gate \
         reads as passing, which is worse than no gate."
    );
    assert!(
        mismatches.is_empty(),
        "flush_encode output changed from the pinned values:\n  {}",
        mismatches.join("\n  ")
    );
}

// ── Generator pins ────────────────────────────────────────────────────────

const SURFACE_PCM_HASH: u64 = 0x7165d38e57e9257b;

/// (rate key, wire_frames, clean LLRs, noisy LLRs). The LLR columns are 0
/// for the no-FEC rates, which have no soft layer.
const SURFACE_INPUT_HASHES: [(&str, u64, u64, u64); 4] = [
    (
        "imbe_7200x4400",
        0xd9f0baf5473d4479,
        0x785dcd6ff7bc8d23,
        0x946f009e15dd828e,
    ),
    ("imbe_4400x4400", 0xcab3d28cfb36fde9, 0, 0),
    (
        "ambe2_3600x2450",
        0x6c558f45300e8cd8,
        0x10717c7bc83e5549,
        0x34acdec7d0ce5340,
    ),
    ("ambe2_2450x2450", 0x5d68e111d90d954c, 0, 0),
];

const SURFACE_TONE_HASHES: [(&str, u64); 2] = [
    ("ambe2_3600x2450", 0xfe99a684a7ae14c7),
    ("ambe2_2450x2450", 0x392be747c006236c),
];

/// The gates above are only meaningful if the inputs they hash are identical
/// on every target, so pin those too. Every generator here is integer-only —
/// no libm, no floats — which is what makes them portable; these constants
/// prove that property rather than assuming it.
#[test]
fn test_inputs_are_platform_stable() {
    let pcm = surface_pcm(FRAMES);
    assert_eq!(pcm.len(), FRAMES * 160);
    let pcm_hash = hash_i16(&pcm);

    let mut inputs = Vec::new();
    for rate in ALL_RATES {
        let frames = wire_frames(rate, FRAMES);
        assert_eq!(frames.len(), rate.fec_frame_bytes() * FRAMES);
        let (clean, noisy) = match rate.soft_frame_bits() {
            Some(n) => {
                let c = llrs_for(rate, SOFT_FRAMES, 0);
                let e = llrs_for(rate, SOFT_FRAMES, SOFT_FLIP_EVERY);
                assert_eq!(c.len(), n * SOFT_FRAMES);
                assert_eq!(e.len(), n * SOFT_FRAMES);
                // The two regimes must genuinely differ, or the "noisy"
                // column of GOLDEN_SOFT is a duplicate of the clean one.
                assert_ne!(c, e);
                (hash_i8(&c), hash_i8(&e))
            }
            None => (0, 0),
        };
        inputs.push((rate_key(rate), fnv1a(&frames), clean, noisy));
    }

    let mut tones = Vec::new();
    for rate in AMBE_RATES {
        tones.push((rate_key(rate), fnv1a(&tone_frames(rate))));
    }

    println!("\n// Re-bless test-input hashes:");
    println!("const SURFACE_PCM_HASH: u64 = {pcm_hash:#018x};");
    println!("const SURFACE_INPUT_HASHES: [(&str, u64, u64, u64); 4] = [");
    for (k, f, c, e) in &inputs {
        println!("    (\"{k}\", {f:#018x}, {c:#018x}, {e:#018x}),");
    }
    println!("];");
    println!("const SURFACE_TONE_HASHES: [(&str, u64); 2] = [");
    for (k, h) in &tones {
        println!("    (\"{k}\", {h:#018x}),");
    }
    println!("];\n");

    assert_ne!(
        SURFACE_PCM_HASH, 0,
        "SURFACE_PCM_HASH still holds a placeholder zero — an unpinned gate \
         reads as passing, which is worse than no gate."
    );
    assert!(
        SURFACE_INPUT_HASHES.iter().all(|(_, f, _, _)| *f != 0)
            && SURFACE_TONE_HASHES.iter().all(|(_, h)| *h != 0),
        "input hashes still hold placeholder zeros — an unpinned gate reads \
         as passing, which is worse than no gate."
    );

    assert_eq!(
        pcm_hash, SURFACE_PCM_HASH,
        "PCM generator is not platform-stable"
    );
    for (key, frames, clean, noisy) in inputs {
        let want = SURFACE_INPUT_HASHES
            .iter()
            .find(|(k, _, _, _)| *k == key)
            .copied()
            .expect("every rate has a pinned input row");
        assert_eq!(
            (frames, clean, noisy),
            (want.1, want.2, want.3),
            "input generators are not platform-stable for {key}"
        );
    }
    for (key, hash) in tones {
        let want = SURFACE_TONE_HASHES
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, h)| *h)
            .expect("every AMBE+2 rate has a pinned tone row");
        assert_eq!(hash, want, "tone-frame generator is not stable for {key}");
    }
}

// ── Pin-table coverage ────────────────────────────────────────────────────

/// Assert `table` is keyed by exactly `expected`, in either direction.
///
/// Missing means a rate this file drives has no pinned row; extra means a row
/// is keyed to something nothing drives, which is what a mistyped or renamed
/// key looks like.
fn assert_keyed_by(table: &str, rows: &[&str], expected: &[&str]) {
    for key in expected {
        assert!(
            rows.contains(key),
            "{table} has no row for {key}. Every gate here looks its expected \
             values up by that key, so an unlisted rate is an unpinned rate."
        );
    }
    for key in rows {
        assert!(
            expected.contains(key),
            "{table} has a row keyed {key:?}, which is not a rate this file \
             drives. A renamed or mistyped key leaves the rate it used to \
             cover unpinned."
        );
    }
}

/// [`Rate`] is `#[non_exhaustive]` and documents that more variants will
/// land. Adding one must fail here — loudly, in one place — rather than
/// leaving a rate silently unpinned in whichever gate forgot it.
#[test]
fn every_rate_has_a_pinned_row() {
    let mut all: Vec<&str> = Vec::new();
    for rate in ALL_RATES {
        let key = rate_key(rate);
        assert_ne!(
            key, "unknown",
            "{rate:?} has no rate_key arm. Every table here is keyed by that \
             string, so an unnamed rate collides with every other unnamed one."
        );
        assert!(!all.contains(&key), "two rates share the key {key}");
        all.push(key);
    }
    let fec: Vec<&str> = FEC_RATES.iter().map(|&r| rate_key(r)).collect();
    let ambe: Vec<&str> = AMBE_RATES.iter().map(|&r| rate_key(r)).collect();
    for subset in [&fec, &ambe] {
        for key in subset {
            assert!(
                all.contains(key),
                "{key} is in a subset but not in ALL_RATES"
            );
        }
    }

    let soft: Vec<&str> = GOLDEN_SOFT.iter().map(|(k, _, _)| *k).collect();
    let tone: Vec<&str> = GOLDEN_TONE.iter().map(|(k, _)| *k).collect();
    let tone_in: Vec<&str> = SURFACE_TONE_HASHES.iter().map(|(k, _)| *k).collect();
    let enh: Vec<&str> = GOLDEN_ENHANCEMENT.iter().map(|(k, _, _)| *k).collect();
    let live: Vec<&str> = GOLDEN_LIVE_DECODE.iter().map(|(k, _)| *k).collect();
    let flush: Vec<&str> = GOLDEN_FLUSH_ENCODE.iter().map(|(k, _, _)| *k).collect();
    let inputs: Vec<&str> = SURFACE_INPUT_HASHES.iter().map(|(k, _, _, _)| *k).collect();

    assert_keyed_by("GOLDEN_ENHANCEMENT", &enh, &all);
    assert_keyed_by("GOLDEN_LIVE_DECODE", &live, &all);
    assert_keyed_by("GOLDEN_FLUSH_ENCODE", &flush, &all);
    assert_keyed_by("SURFACE_INPUT_HASHES", &inputs, &all);
    assert_keyed_by("GOLDEN_SOFT", &soft, &fec);
    assert_keyed_by("GOLDEN_TONE", &tone, &ambe);
    assert_keyed_by("SURFACE_TONE_HASHES", &tone_in, &ambe);

    // The subset tables are only the right size if the subsets themselves are
    // still true of the crate, so re-derive them from the rates rather than
    // trusting the two literals above.
    for rate in ALL_RATES {
        assert_eq!(
            rate.soft_frame_bits().is_some(),
            FEC_RATES.contains(&rate),
            "{}: FEC_RATES no longer lists exactly the rates with a soft \
             layer, so GOLDEN_SOFT covers the wrong set",
            rate_key(rate)
        );
    }

    // Transcoder pairs, from the same double loop the gate itself walks.
    let pairs: Vec<String> = ALL_RATES
        .iter()
        .flat_map(|&from| ALL_RATES.iter().map(move |&to| (from, to)))
        .filter(|&(from, to)| Transcoder::new(from, to).is_ok())
        .map(|(from, to)| pair_key(from, to))
        .collect();
    let pairs: Vec<&str> = pairs.iter().map(|k| k.as_str()).collect();
    let transcode: Vec<&str> = GOLDEN_TRANSCODE.iter().map(|(k, _)| *k).collect();
    assert_keyed_by("GOLDEN_TRANSCODE", &transcode, &pairs);
    assert_keyed_by("TRANSCODE_PAIRS", &TRANSCODE_PAIRS, &pairs);
}
