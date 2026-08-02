//! No-panic garbage-input harness for the blip25-codec public surface.
//!
//! Production contract: hostile/malformed input must NEVER panic — the codec
//! degrades gracefully (concealment / silence / erasure handling). This
//! harness throws structured hostile inputs (all-zeros, all-ones, erasure-range
//! b0, invalid/reserved tone IDs, seeded-LCG random bytes, PCM rails / DC /
//! impulses / noise, wrong-length slices) at every public entry point and
//! asserts that no call panicked.
//!
//! Deterministic by construction: seeded LCG, fixed iteration counts, no env
//! vars, no external fixtures. One #[test] per input family; each family
//! collects every panic via `catch_unwind` (so one panic does not mask the
//! rest) and completes with an assertion listing all of them.

use std::cell::{Cell, RefCell};
use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

use blip25_codec::dequantize::MbeParams;
use blip25_codec::enc::encode_frame::{decode_r34, encode_r33, encode_r34};
use blip25_codec::enc::pcm_encode::{encode_pcm_b0, EncodeOpts};
use blip25_codec::synth::{self, SynthState};
use blip25_codec::{frame, tables, tone, Decoder, Encoder, ImbeDecoder};

// ---------------------------------------------------------------------------
// panic capture: record the panic message + file:line for every probed call
// without spamming the default hook's output for EXPECTED-probe panics.
// ---------------------------------------------------------------------------

thread_local! {
    static CAPTURING: Cell<bool> = const { Cell::new(false) };
    static LAST_PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
}
static HOOK: Once = Once::new();

fn install_hook() {
    HOOK.call_once(|| {
        let prev = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            if CAPTURING.with(|c| c.get()) {
                // `info` Display includes payload + panic site file:line.
                LAST_PANIC.with(|p| *p.borrow_mut() = Some(info.to_string()));
            } else {
                prev(info);
            }
        }));
    });
}

/// Run one probed call. On panic, log `label: <panic message @ file:line>` and
/// return 0; otherwise return the call's checksum contribution.
fn probe<F: FnOnce() -> u64>(log: &mut Vec<String>, label: String, f: F) -> u64 {
    install_hook();
    CAPTURING.with(|c| c.set(true));
    let r = panic::catch_unwind(AssertUnwindSafe(f));
    CAPTURING.with(|c| c.set(false));
    match r {
        Ok(v) => v,
        Err(_) => {
            let msg = LAST_PANIC
                .with(|p| p.borrow_mut().take())
                .unwrap_or_else(|| "<panic info not captured>".to_string());
            log.push(format!("{label}: {msg}"));
            0
        }
    }
}

/// Final family assertion: pass iff no probe panicked. On failure print every
/// distinct panic (with an example input label) plus the total count.
fn finish(family: &str, log: Vec<String>, sink: u64) {
    // `sink` folds every successful call's output so the whole family is
    // observable work, not dead code the optimizer (or a refactor) can drop.
    std::hint::black_box(sink);
    if log.is_empty() {
        return;
    }
    let mut distinct: Vec<(String, usize, String)> = Vec::new(); // (msg, count, first label)
    for entry in &log {
        let (label, msg) = entry.split_once(": ").unwrap_or(("", entry));
        match distinct.iter_mut().find(|(m, _, _)| m == msg) {
            Some((_, n, _)) => *n += 1,
            None => distinct.push((msg.to_string(), 1, label.to_string())),
        }
    }
    let mut report = format!(
        "[{family}] {} panic(s) across {} probed call(s); {} distinct:\n",
        log.len(),
        log.len(),
        distinct.len()
    );
    for (msg, n, first) in distinct.iter().take(40) {
        report.push_str(&format!("  x{n} (first: {first})\n      {msg}\n"));
    }
    if distinct.len() > 40 {
        report.push_str(&format!(
            "  ... and {} more distinct panics\n",
            distinct.len() - 40
        ));
    }
    panic!("{report}");
}

// ---------------------------------------------------------------------------
// deterministic LCG + input builders
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn u8(&mut self) -> u8 {
        (self.next() >> 33) as u8
    }
    fn u16(&mut self) -> u16 {
        (self.next() >> 33) as u16
    }
    fn i16(&mut self) -> i16 {
        (self.next() >> 33) as i16
    }
}

fn checksum_pcm(pcm: &[i16]) -> u64 {
    pcm.iter().fold(0u64, |a, &s| {
        a.wrapping_mul(31).wrapping_add(s as u16 as u64)
    })
}

/// AMBE+2 nominal b0..b8 widths (bits beyond these are ignored by encode).
const B_WIDTHS: [u32; 9] = [7, 5, 5, 9, 7, 5, 4, 4, 3];

/// Random in-width b-vector with a forced b0.
fn b_vec_with_b0(b0: u16, rng: &mut Lcg) -> [u16; 9] {
    let mut b = [0u16; 9];
    b[0] = b0;
    for (i, slot) in b.iter_mut().enumerate().skip(1) {
        *slot = rng.u16() & ((1 << B_WIDTHS[i]) - 1);
    }
    b
}

/// Byte-exact 9-byte r33 tone frame for any (I_D, A_D) — including reserved /
/// invalid IDs. Built from the §2.10.1 info-vector signature via the public
/// `tables::deprioritize` + `encode_r33` bijection.
fn tone_frame_bytes(id: u8, amp: u8, rng: &mut Lcg) -> [u8; 9] {
    let u0 = (0x3Fu16 << 6) | (u16::from(amp >> 1) & 0x3F);
    let u1 = rng.u16() & 0xFFF;
    let u2 = rng.u16() & 0x7FF;
    let u3 = (u16::from(id) << 5) | (u16::from(amp & 1) << 4); // trailer (3..0) = 0
    let b = tables::deprioritize(&[u0, u1, u2, u3]);
    encode_r33(&b)
}

/// The shared hostile 9-byte frame corpus for the AMBE decoder family.
fn hostile_ambe_frames() -> Vec<(String, [u8; 9])> {
    let mut rng = Lcg::new(0xB1_1425_0001);
    let mut frames: Vec<(String, [u8; 9])> = Vec::new();
    for (name, byte) in [
        ("all-zeros", 0x00u8),
        ("all-ones", 0xFF),
        ("0xAA", 0xAA),
        ("0x55", 0x55),
    ] {
        frames.push((format!("pattern:{name}"), [byte; 9]));
    }
    // Erasure-range + out-of-range voice pitch b0 (120..=127), random rest.
    for b0 in 120u16..=127 {
        frames.push((
            format!("erasure-b0:{b0}"),
            encode_r33(&b_vec_with_b0(b0, &mut rng)),
        ));
    }
    // Tone frames: every 8th ID (covers valid, reserved, KNOX, silence bands),
    // plus edge IDs, at amplitude extremes + midpoint.
    let mut ids: Vec<u8> = (0u16..=255).step_by(8).map(|i| i as u8).collect();
    ids.extend_from_slice(&[1, 5, 143, 144, 159, 160, 254, 255]);
    for id in ids {
        for amp in [0u8, 64, 127] {
            frames.push((
                format!("tone-id:{id}-amp:{amp}"),
                tone_frame_bytes(id, amp, &mut rng),
            ));
        }
    }
    // Random garbage frames (fixed count, seeded).
    for k in 0..48 {
        let mut f = [0u8; 9];
        for b in f.iter_mut() {
            *b = rng.u8();
        }
        frames.push((format!("random:{k}"), f));
    }
    frames
}

// ---------------------------------------------------------------------------
// families
// ---------------------------------------------------------------------------

/// AMBE+2 decode entry points over hostile-but-well-sized 9-byte frames:
/// structural patterns, erasure-range b0, tone frames with invalid/reserved
/// IDs, random bytes — each API on a fresh decoder AND all APIs interleaved on
/// one long-lived decoder (cross-frame state interactions).
#[test]
fn ambe_decoder_survives_hostile_frames() {
    let frames = hostile_ambe_frames();
    let mut log = Vec::new();
    let mut sink = 0u64;

    // Per-API fresh decoders, full corpus in order.
    let mut d_pcm = Decoder::new();
    let mut d_fix = Decoder::new();
    let mut d_con = Decoder::new();
    let mut d_par = Decoder::new();
    for (name, f) in &frames {
        sink ^= probe(&mut log, format!("Decoder::decode_pcm({name})"), || {
            checksum_pcm(&d_pcm.decode_pcm(f))
        });
        sink ^= probe(
            &mut log,
            format!("Decoder::decode_pcm_fixed({name})"),
            || checksum_pcm(&d_fix.decode_pcm_fixed(f)),
        );
        sink ^= probe(
            &mut log,
            format!("Decoder::decode_pcm_fixed_concealed({name})"),
            || {
                let (pcm, disp, e0, et) = d_con.decode_pcm_fixed_concealed(f);
                checksum_pcm(&pcm) ^ ((disp as u64) << 1) ^ u64::from(e0) ^ (u64::from(et) << 8)
            },
        );
        sink ^= probe(&mut log, format!("Decoder::decode_params({name})"), || {
            let (fr, p) = d_par.decode_params(f);
            u64::from(fr.epsilon_t()) ^ p.map(|m| u64::from(m.l)).unwrap_or(0xEE)
        });
    }

    // One decoder, all APIs interleaved across the corpus.
    let mut d_mix = Decoder::new();
    for (i, (name, f)) in frames.iter().enumerate() {
        sink ^= probe(
            &mut log,
            format!("Decoder(interleaved#{i})::*({name})"),
            || match i % 4 {
                0 => checksum_pcm(&d_mix.decode_pcm(f)),
                1 => checksum_pcm(&d_mix.decode_pcm_fixed(f)),
                2 => checksum_pcm(&d_mix.decode_pcm_fixed_concealed(f).0),
                _ => {
                    let (fr, _) = d_mix.decode_params(f);
                    u64::from(fr.epsilon_t())
                }
            },
        );
    }

    finish("ambe_decoder_survives_hostile_frames", log, sink);
}

/// IMBE decode entry points over hostile 18-byte frames.
#[test]
fn imbe_decoder_survives_hostile_frames() {
    let mut rng = Lcg::new(0x1B_BE_0002);
    let mut frames: Vec<(String, [u8; 18])> = vec![
        ("all-zeros".into(), [0x00; 18]),
        ("all-ones".into(), [0xFF; 18]),
        ("0xAA".into(), [0xAA; 18]),
        ("0x55".into(), [0x55; 18]),
    ];
    for k in 0..64 {
        let mut f = [0u8; 18];
        for b in f.iter_mut() {
            *b = rng.u8();
        }
        frames.push((format!("random:{k}"), f));
    }

    let mut log = Vec::new();
    let mut sink = 0u64;
    let mut d_pcm = ImbeDecoder::new();
    let mut d_par = ImbeDecoder::new();
    for (name, f) in &frames {
        sink ^= probe(&mut log, format!("ImbeDecoder::decode_pcm({name})"), || {
            checksum_pcm(&d_pcm.decode_pcm(f))
        });
        sink ^= probe(
            &mut log,
            format!("ImbeDecoder::decode_params({name})"),
            || {
                d_par
                    .decode_params(f)
                    .map(|p| u64::from(p.l))
                    .unwrap_or(0xEE)
            },
        );
    }
    finish("imbe_decoder_survives_hostile_frames", log, sink);
}

/// `tone::classify` / `tone::parse_tone_frame` over structured tone signatures
/// (every I_D, amplitude extremes), over-width info vectors, and random
/// full-range `[u16; 4]` patterns.
#[test]
fn tone_info_classify_parse_survive_all_patterns() {
    let mut log = Vec::new();
    let mut sink = 0u64;

    // Structured: full tone signature for every ID x amplitude extremes.
    for id in 0u16..=255 {
        for amp in [0u8, 127] {
            let u0 = (0x3Fu16 << 6) | (u16::from(amp >> 1) & 0x3F);
            let u3 = (id << 5) | (u16::from(amp & 1) << 4);
            let u = [u0, 0x0FFF, 0x07FF, u3];
            sink ^= probe(
                &mut log,
                format!("tone::classify/parse(id:{id},amp:{amp})"),
                || {
                    let k = tone::classify(&u) as u64;
                    let p = tone::parse_tone_frame(&u)
                        .map(|f| (u64::from(f.id) << 8) | u64::from(f.amplitude))
                        .unwrap_or(0xFFFF);
                    (k << 32) ^ p
                },
            );
        }
    }
    // Over-width vectors (bits above the nominal 12/12/11/14 widths set).
    for u in [
        [0xFFFFu16, 0xFFFF, 0xFFFF, 0xFFFF],
        [0x0000, 0x0000, 0x0000, 0x0000],
        [0xF000, 0xF000, 0xF800, 0xC000],
    ] {
        sink ^= probe(
            &mut log,
            format!("tone::classify/parse(overwidth:{u:04x?})"),
            || {
                let k = tone::classify(&u) as u64;
                tone::parse_tone_frame(&u)
                    .map(|f| u64::from(f.id))
                    .unwrap_or(0xEE)
                    ^ k
            },
        );
    }
    // Random full-range info vectors — cheap pure bit logic, wide sweep.
    let mut rng = Lcg::new(0x70_4E_0003);
    for k in 0..20_000u32 {
        let u = [rng.u16(), rng.u16(), rng.u16(), rng.u16()];
        sink ^= probe(
            &mut log,
            format!("tone::classify/parse(random:{k})"),
            || {
                let c = tone::classify(&u) as u64;
                let b = tables::deprioritize(&u); // exercised on the same patterns
                c ^ u64::from(b[0])
            },
        );
    }
    finish("tone_info_classify_parse_survive_all_patterns", log, sink);
}

/// Frame-layer byte APIs: `frame::decode_bytes` and the byte-slice `Decoder`
/// entry points at WRONG lengths (0..=16) as well as 9-byte garbage;
/// `decode_r34` on hostile 7-byte frames; `encode_r33`/`encode_r34` with
/// full-range (over-width) b-vectors.
#[test]
fn frame_layer_survives_wrong_lengths_and_garbage() {
    let mut rng = Lcg::new(0xF4_A3_0004);
    let mut log = Vec::new();
    let mut sink = 0u64;

    // decode_bytes at every length 0..=16 (9 = nominal), garbage content.
    for len in 0usize..=16 {
        let buf: Vec<u8> = (0..len).map(|_| rng.u8()).collect();
        sink ^= probe(&mut log, format!("frame::decode_bytes(len:{len})"), || {
            let f = frame::decode_bytes(&buf);
            u64::from(f.epsilon_t()) ^ u64::from(f.info[0])
        });
    }
    // Decoder byte-slice entry points at wrong lengths.
    for len in [0usize, 1, 4, 8, 10, 16] {
        let buf: Vec<u8> = (0..len).map(|_| rng.u8()).collect();
        let mut d = Decoder::new();
        sink ^= probe(&mut log, format!("Decoder::decode_pcm(len:{len})"), || {
            checksum_pcm(&d.decode_pcm(&buf))
        });
        let mut d = Decoder::new();
        sink ^= probe(
            &mut log,
            format!("Decoder::decode_pcm_fixed(len:{len})"),
            || checksum_pcm(&d.decode_pcm_fixed(&buf)),
        );
        let mut d = Decoder::new();
        sink ^= probe(
            &mut log,
            format!("Decoder::decode_pcm_fixed_concealed(len:{len})"),
            || checksum_pcm(&d.decode_pcm_fixed_concealed(&buf).0),
        );
        let mut d = Decoder::new();
        sink ^= probe(
            &mut log,
            format!("Decoder::decode_params(len:{len})"),
            || u64::from(d.decode_params(&buf).0.epsilon_t()),
        );
    }
    // r34 decode: hostile 7-byte frames (incl. nonzero padding bits).
    for (name, f) in [
        ("all-zeros".to_string(), [0x00u8; 7]),
        ("all-ones".to_string(), [0xFF; 7]),
    ]
    .into_iter()
    .chain((0..32).map(|k| {
        let mut f = [0u8; 7];
        for b in f.iter_mut() {
            *b = rng.u8();
        }
        (format!("random:{k}"), f)
    })) {
        sink ^= probe(
            &mut log,
            format!("encode_frame::decode_r34({name})"),
            || u64::from(decode_r34(&f)[0]),
        );
    }
    // Encode with over-width b-vectors (docs: excess bits ignored — no panic).
    for k in 0..32 {
        let b: [u16; 9] = std::array::from_fn(|_| rng.u16());
        sink ^= probe(
            &mut log,
            format!("encode_frame::encode_r33/r34(overwidth:{k})"),
            || {
                let r33 = encode_r33(&b);
                let r34 = encode_r34(&b);
                u64::from(r33[0]) ^ (u64::from(r34[0]) << 8)
            },
        );
    }
    finish("frame_layer_survives_wrong_lengths_and_garbage", log, sink);
}

/// The 160-sample hostile PCM pattern set shared by the encoder families.
fn hostile_pcm_patterns() -> Vec<(String, Vec<i16>)> {
    let mut rng = Lcg::new(0xEC_0D_0005);
    let n = 160 * 4; // 4 frames worth per pattern
    let mut noise = vec![0i16; n];
    for s in noise.iter_mut() {
        *s = rng.i16();
    }
    vec![
        ("silence".into(), vec![0i16; n]),
        ("pos-rail".into(), vec![i16::MAX; n]),
        ("neg-rail".into(), vec![i16::MIN; n]),
        ("dc-1000".into(), vec![1000i16; n]),
        (
            "impulse-train".into(),
            (0..n)
                .map(|i| if i % 17 == 0 { i16::MAX } else { 0 })
                .collect(),
        ),
        (
            "alt-rails".into(),
            (0..n)
                .map(|i| if i % 2 == 0 { i16::MAX } else { i16::MIN })
                .collect(),
        ),
        ("noise".into(), noise),
    ]
}

/// Streaming `Encoder` (r33 / r34 / IMBE) over hostile PCM: rails, DC,
/// impulse trains, alternating rails, seeded noise — plus end-of-stream flush.
#[test]
fn encoder_streaming_survives_hostile_pcm() {
    let mut log = Vec::new();
    let mut sink = 0u64;
    for (name, pcm) in hostile_pcm_patterns() {
        // r33 + r34 on one encoder each; IMBE on its own.
        let mut e33 = Encoder::new();
        let mut e34 = Encoder::new();
        let mut eim = Encoder::new();
        for (fi, chunk) in pcm.chunks_exact(160).enumerate() {
            let fr: &[i16; 160] = chunk.try_into().unwrap();
            sink ^= probe(
                &mut log,
                format!("Encoder::encode_frame_r33({name}#{fi})"),
                || {
                    e33.encode_frame_r33(fr)
                        .map(|b| u64::from(b[0]) + 1)
                        .unwrap_or(0)
                },
            );
            sink ^= probe(
                &mut log,
                format!("Encoder::encode_frame_r34({name}#{fi})"),
                || {
                    e34.encode_frame_r34(fr)
                        .map(|b| u64::from(b[0]) + 1)
                        .unwrap_or(0)
                },
            );
            sink ^= probe(
                &mut log,
                format!("Encoder::encode_imbe_frame({name}#{fi})"),
                || {
                    eim.encode_imbe_frame(fr)
                        .map(|b| u64::from(b[0]) + 1)
                        .unwrap_or(0)
                },
            );
        }
        sink ^= probe(&mut log, format!("Encoder::flush_r33({name})"), || {
            e33.flush_r33().len() as u64
        });
        sink ^= probe(&mut log, format!("Encoder::flush_r34({name})"), || {
            e34.flush_r34().len() as u64
        });
        sink ^= probe(&mut log, format!("Encoder::flush_imbe({name})"), || {
            eim.flush_imbe().len() as u64
        });
    }
    finish("encoder_streaming_survives_hostile_pcm", log, sink);
}

/// Batch `pcm_encode` entry points with hostile lengths (empty, sub-frame,
/// off-by-one around the 160-sample frame) and hostile content.
#[test]
fn pcm_encode_survives_hostile_lengths_and_pcm() {
    let mut rng = Lcg::new(0x9C_E0_0006);
    let mut noise = vec![0i16; 400];
    for s in noise.iter_mut() {
        *s = rng.i16();
    }
    let rails = vec![i16::MIN; 400];
    let cases: Vec<(String, Vec<i16>)> = vec![
        ("empty".into(), Vec::new()),
        ("one-sample".into(), vec![i16::MAX]),
        ("len-159-rails".into(), rails[..159].to_vec()),
        ("len-161-noise".into(), noise[..161].to_vec()),
        ("len-320-rails".into(), rails[..320].to_vec()),
        ("len-400-noise".into(), noise.clone()),
    ];
    let mut log = Vec::new();
    let mut sink = 0u64;
    for (name, pcm) in &cases {
        sink ^= probe(
            &mut log,
            format!("pcm_encode::encode_pcm_b0({name})"),
            || encode_pcm_b0(pcm, EncodeOpts::default()).len() as u64,
        );
    }
    // Conformance opts variant on one hostile input.
    sink ^= probe(
        &mut log,
        "pcm_encode::encode_pcm_b0(conformance,noise)".to_string(),
        || encode_pcm_b0(&noise, EncodeOpts::conformance()).len() as u64,
    );
    finish("pcm_encode_survives_hostile_lengths_and_pcm", log, sink);
}

/// `synth::synthesize_frame` with hostile `MbeParams`: length mismatches
/// between `l` and the voiced/amplitude vectors, non-finite and huge values,
/// out-of-range `l`, degenerate omega — on both AMBE and IMBE synth state,
/// fed as sequences (cross-frame state carry-over).
#[test]
fn synthesize_frame_survives_hostile_params() {
    let mk = |b0: u8, omega: f32, l: u8, voiced: Vec<bool>, amps: Vec<f32>| MbeParams {
        b0,
        omega_0: omega,
        l,
        voiced,
        amplitudes: amps,
    };
    let cases: Vec<(String, MbeParams)> = vec![
        ("l0-empty".into(), mk(0, 0.1, 0, vec![], vec![])),
        (
            "l56-nominal-huge-amps".into(),
            mk(40, 0.08, 56, vec![true; 56], vec![1e30; 56]),
        ),
        ("l56-empty-vecs".into(), mk(40, 0.08, 56, vec![], vec![])),
        (
            "l56-short-vecs".into(),
            mk(40, 0.08, 56, vec![true; 10], vec![1.0; 10]),
        ),
        (
            "l255-short-vecs".into(),
            mk(200, 0.08, 255, vec![true; 10], vec![1.0; 10]),
        ),
        (
            "nan-omega".into(),
            mk(40, f32::NAN, 20, vec![true; 20], vec![1.0; 20]),
        ),
        (
            "inf-amps".into(),
            mk(40, 0.1, 20, vec![true; 20], vec![f32::INFINITY; 20]),
        ),
        (
            "nan-amps".into(),
            mk(40, 0.1, 20, vec![false; 20], vec![f32::NAN; 20]),
        ),
        (
            "neg-amps".into(),
            mk(40, 0.1, 20, vec![true; 20], vec![-1e9; 20]),
        ),
        (
            "zero-omega".into(),
            mk(40, 0.0, 20, vec![true; 20], vec![100.0; 20]),
        ),
        (
            "neg-omega".into(),
            mk(40, -1.0, 20, vec![true; 20], vec![100.0; 20]),
        ),
        (
            "huge-omega".into(),
            mk(40, 1e9, 20, vec![true; 20], vec![100.0; 20]),
        ),
        ("l9-min".into(), mk(0, 0.5, 9, vec![false; 9], vec![0.0; 9])),
    ];
    let mut log = Vec::new();
    let mut sink = 0u64;
    // Fresh state per case, both synth-state flavors.
    for (name, p) in &cases {
        let mut st = SynthState::new();
        sink ^= probe(&mut log, format!("synthesize_frame[ambe]({name})"), || {
            checksum_pcm(&synth::synthesize_frame(p, &mut st))
        });
        let mut st = SynthState::new_imbe();
        sink ^= probe(&mut log, format!("synthesize_frame[imbe]({name})"), || {
            checksum_pcm(&synth::synthesize_frame(p, &mut st))
        });
    }
    // One long-lived state across the whole hostile sequence (incl. tone mode).
    let mut st = SynthState::new();
    for (i, (name, p)) in cases.iter().enumerate() {
        st.set_tone_frame(i % 3 == 0);
        sink ^= probe(
            &mut log,
            format!("synthesize_frame[seq#{i}]({name})"),
            || checksum_pcm(&synth::synthesize_frame(p, &mut st)),
        );
    }
    finish("synthesize_frame_survives_hostile_params", log, sink);
}

/// PCM-side tone detection (`detect_tone_frame`, `ToneDetector::process`) with
/// hostile PCM at nominal and wrong lengths.
#[test]
fn tone_detector_survives_hostile_pcm() {
    let mut rng = Lcg::new(0xDE_7E_0007);
    let mut noise = vec![0i16; 4096];
    for s in noise.iter_mut() {
        *s = rng.i16();
    }
    let mut log = Vec::new();
    let mut sink = 0u64;
    let contents: Vec<(&str, Vec<i16>)> = vec![
        ("rails", vec![i16::MIN; 4096]),
        ("dc", vec![1000; 4096]),
        ("noise", noise),
        ("zeros", vec![0; 4096]),
    ];
    for (cname, buf) in &contents {
        for len in [0usize, 1, 64, 159, 160, 161, 1024] {
            let slice = &buf[..len.min(buf.len())];
            sink ^= probe(
                &mut log,
                format!("tone::detect_tone_frame({cname},len:{len})"),
                || {
                    tone::detect_tone_frame(slice)
                        .map(|f| (u64::from(f.id) << 8) | u64::from(f.amplitude))
                        .unwrap_or(0xEE)
                },
            );
        }
    }
    // Streaming detector across mixed content + odd chunk sizes.
    let mut det = tone::ToneDetector::new();
    for (i, len) in [160usize, 160, 159, 161, 0, 1, 160, 1024]
        .iter()
        .enumerate()
    {
        let base = (i * 191) % 2048;
        let cont = &contents[i % contents.len()].1;
        let slice = &cont[base..(base + len).min(cont.len())];
        sink ^= probe(
            &mut log,
            format!("ToneDetector::process(chunk#{i},len:{len})"),
            || det.process(slice).map(|f| u64::from(f.id)).unwrap_or(0xEE),
        );
    }
    det.reset();
    finish("tone_detector_survives_hostile_pcm", log, sink);
}
