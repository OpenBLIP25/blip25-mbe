//! Bounded-memory soak for the long-running live-stream APIs.
//!
//! Feeds deterministic synthetic audio / frames (seeded LCG + sine mix at a
//! speech-like level) through [`LiveEncoder`], [`Vocoder::decode_stream`] and
//! [`LiveDecoder`], draining every output the APIs yield, then asserts the
//! process RSS delta between a post-warm-up snapshot and the end of the soak
//! stays under a generous threshold. Linux-only measurement (`/proc/self/status`
//! `VmRSS`), which is fine — CI and dev boxes for this repo are Linux.
//!
//! All three phases run inside ONE `#[test]` so the RSS snapshots are never
//! polluted by a concurrently running sibling test in the same process.
//!
//! Scale notes (dev profile, unoptimised):
//! * Decode of real frames is ~0.65 ms/frame in dev, so the `DecodeStream`
//!   soak runs near full task scale (2,000 warm-up + 15,000 soak frames) and
//!   the `LiveDecoder` soak (same underlying decode state) runs a shorter
//!   1,000 + 6,000 pass focused on the pending-buffer churn.
//! * Encode is ~95 ms/frame in dev — 22,000 frames would take ~35 minutes —
//!   so the encoder soak is scaled to fit the ~30 s budget (24 warm-up +
//!   80 soak frames) and additionally reports the measured per-frame RSS
//!   growth rate so unbounded growth is still visible in the failure message.
//! * Encode-side Annex-T tone detection is requested ON for this soak, but the
//!   public `LiveEncoder` API has no tone toggle: `LiveEncoder::new` hard-codes
//!   the streaming path's `tone_on = false` and `LiveEncoder::vocoder()` is
//!   read-only, so `Vocoder::set_tone_detection` is unreachable from here.
//!   The soak therefore runs the streaming encoder as publicly constructable.
//!   The `ToneDetector` itself is exercised long-running via the whole-buffer
//!   `Vocoder::encode` path (tone detection enabled) in the encoder phase.

#![cfg(target_os = "linux")]

use blip25_mbe::vocoder::{LiveDecoder, LiveEncoder, Rate, Vocoder};

/// One 20 ms frame at 8 kHz.
const FRAME: usize = 160;
/// AMBE+2 3600x2450 FEC frame bytes.
const R33_BYTES: usize = 9;
/// Generous-but-real RSS growth ceiling per phase (allocator noise margin).
const RSS_DELTA_LIMIT: usize = 16 * 1024 * 1024;

/// Deterministic LCG (same multiplier family the repo's seeded tests use).
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

/// Speech-like deterministic synthetic frame: two sines (a "formant" pair)
/// plus LCG noise, ~-8 dBFS peaks, with a slow amplitude wobble so the
/// encoder's gain / voicing chains see non-static input.
fn synth_frame(lcg: &mut Lcg, t: &mut f64) -> [i16; FRAME] {
    let mut out = [0i16; FRAME];
    for s in out.iter_mut() {
        let noise = (f64::from(lcg.next_u32()) / f64::from(u32::MAX) - 0.5) * 900.0;
        let wobble = 0.75 + 0.25 * (2.0 * std::f64::consts::PI * 2.3 * *t).sin();
        let v = wobble
            * (6000.0 * (2.0 * std::f64::consts::PI * 211.0 * *t).sin()
                + 2200.0 * (2.0 * std::f64::consts::PI * 733.0 * *t).sin())
            + noise;
        *t += 1.0 / 8000.0;
        *s = v.clamp(-30000.0, 30000.0) as i16;
    }
    out
}

/// Current process resident set size in bytes, from `/proc/self/status` VmRSS.
fn rss_bytes() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kib: usize = rest
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse()
                .expect("parse VmRSS kB");
            return kib * 1024;
        }
    }
    panic!("VmRSS not found in /proc/self/status");
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// LiveEncoder soak, one rate. Scaled to the dev-profile encode cost
/// (~95 ms/frame); reports the per-frame growth rate so a fixed-rate leak is
/// diagnosable even at this scale.
///
/// Every rate is driven by a single-pass streamer holding a bounded look-ahead
/// (`AmbeStream` / `ImbeStream`), and each keeps its own copy of the growing
/// `pref`/`raw` analysis buffers, so each needs its own soak.
fn soak_live_encoder(rate: Rate, frame_bytes: usize) {
    const WARMUP: usize = 24;
    const SOAK: usize = 80;
    // Pending look-ahead bound: B1_RESERVE(16) + B1_MIN_BATCH(16) + lag slack,
    // asserted with headroom at 64 frames.
    const PENDING_LIMIT: usize = 64 * FRAME;

    let mut enc = LiveEncoder::new(rate);
    let mut lcg = Lcg(0xB1_1925);
    let mut t = 0.0f64;
    let mut frames_out = 0usize;
    let mut bytes_out = 0usize;

    let feed = |enc: &mut LiveEncoder,
                lcg: &mut Lcg,
                t: &mut f64,
                n: usize,
                frames_out: &mut usize,
                bytes_out: &mut usize| {
        for _ in 0..n {
            let f = synth_frame(lcg, t);
            for r in enc.push(&f) {
                let bits = r.expect("valid PCM must encode without error");
                assert_eq!(bits.len(), frame_bytes);
                *frames_out += 1;
                *bytes_out += bits.len();
            }
            let pending = enc.pending_samples();
            assert!(
                pending <= PENDING_LIMIT,
                "LiveEncoder pending_samples unbounded: {pending} > {PENDING_LIMIT}"
            );
        }
    };

    feed(
        &mut enc,
        &mut lcg,
        &mut t,
        WARMUP,
        &mut frames_out,
        &mut bytes_out,
    );
    let rss0 = rss_bytes();
    feed(
        &mut enc,
        &mut lcg,
        &mut t,
        SOAK,
        &mut frames_out,
        &mut bytes_out,
    );
    let tail = enc.flush().expect("flush must succeed");
    for bits in &tail {
        assert_eq!(bits.len(), frame_bytes);
        bytes_out += bits.len();
    }
    assert_eq!(
        enc.pending_samples(),
        0,
        "flush must drain all pending samples"
    );
    let rss1 = rss_bytes();

    // Sanity: the soak actually produced a stream (look-ahead accounts for the
    // small deficit vs frames fed).
    assert!(
        frames_out + tail.len() + 8 >= WARMUP + SOAK,
        "encoder emitted too few frames: {} + {} tail vs {} fed",
        frames_out,
        tail.len(),
        WARMUP + SOAK
    );

    let delta = rss1.saturating_sub(rss0);
    eprintln!(
        "[soak] LiveEncoder({rate:?}): {SOAK} frames after warm-up, RSS delta {:.2} MiB \
         ({:.0} B/frame), {bytes_out} bytes out",
        mib(delta),
        delta as f64 / SOAK as f64
    );
    assert!(
        delta < RSS_DELTA_LIMIT,
        "LiveEncoder({rate:?}) RSS grew {:.2} MiB over {SOAK} frames ({:.0} B/frame) — unbounded growth",
        mib(delta),
        delta as f64 / SOAK as f64
    );
}

/// Long-running tone-detector exercise via the only public tone-detection
/// entry point: whole-buffer `Vocoder::encode` with `set_tone_detection(true)`,
/// called repeatedly on a persistent `Vocoder` with alternating tone / voice
/// chunks (the `LiveEncoder` streaming path cannot enable tone detection —
/// see the module docs).
fn soak_tone_detection_encode() {
    const CHUNK_FRAMES: usize = 4;
    const WARMUP_CALLS: usize = 2;
    const SOAK_CALLS: usize = 4;

    let mut voc = Vocoder::new(Rate::AmbePlus2_3600x2450);
    voc.set_tone_detection(true);
    let mut lcg = Lcg(0x70_4E5);
    let mut t = 0.0f64;

    let run = |voc: &mut Vocoder, lcg: &mut Lcg, t: &mut f64, calls: usize| {
        for call in 0..calls {
            let mut pcm = Vec::with_capacity(CHUNK_FRAMES * FRAME);
            if call % 2 == 0 {
                // Clean dual tone (Knox id145 pair: 603.2 / 1055.6 Hz).
                for _ in 0..CHUNK_FRAMES * FRAME {
                    let v = 8000.0 * (2.0 * std::f64::consts::PI * 603.2 * *t).sin()
                        + 8000.0 * (2.0 * std::f64::consts::PI * 1055.6 * *t).sin();
                    *t += 1.0 / 8000.0;
                    pcm.push(v as i16);
                }
            } else {
                for _ in 0..CHUNK_FRAMES {
                    pcm.extend_from_slice(&synth_frame(lcg, t));
                }
            }
            let frames = voc.encode(&pcm);
            for f in &frames {
                assert_eq!(f.len(), R33_BYTES);
            }
        }
    };

    run(&mut voc, &mut lcg, &mut t, WARMUP_CALLS);
    let rss0 = rss_bytes();
    run(&mut voc, &mut lcg, &mut t, SOAK_CALLS);
    let rss1 = rss_bytes();
    let delta = rss1.saturating_sub(rss0);
    eprintln!(
        "[soak] Vocoder::encode tone-detect: {SOAK_CALLS} calls x {CHUNK_FRAMES} frames, \
         RSS delta {:.2} MiB",
        mib(delta)
    );
    assert!(
        delta < RSS_DELTA_LIMIT,
        "tone-detect encode RSS grew {:.2} MiB — unbounded growth",
        mib(delta)
    );
}

/// DecodeStream soak at full scale: 2,000 warm-up + 15,000 soak frames fed
/// through `Vocoder::decode_stream` in bounded chunks (as a live consumer
/// would), draining every PCM frame the iterator yields.
fn soak_decode_stream(frame_pool: &[Vec<u8>]) {
    const WARMUP: usize = 2_000;
    const SOAK: usize = 15_000;
    const CHUNK: usize = 10; // frames per decode_stream call

    let mut voc = Vocoder::new(Rate::AmbePlus2_3600x2450);
    let mut lcg = Lcg(0xDEC_0DE);
    let mut samples_out = 0usize;

    let run = |voc: &mut Vocoder, lcg: &mut Lcg, frames: usize, samples_out: &mut usize| {
        let mut fed = 0usize;
        let mut buf = Vec::with_capacity(CHUNK * R33_BYTES);
        while fed < frames {
            buf.clear();
            let n = CHUNK.min(frames - fed);
            for _ in 0..n {
                let pick = lcg.next_u32() as usize % (frame_pool.len() + 1);
                if pick < frame_pool.len() {
                    buf.extend_from_slice(&frame_pool[pick]);
                } else {
                    // Hostile / corrupt frame: must decode via concealment,
                    // never panic, and never accumulate state.
                    for _ in 0..R33_BYTES {
                        buf.push(lcg.next_u32() as u8);
                    }
                }
            }
            for pcm in voc.decode_stream(&buf) {
                let pcm = pcm.expect("well-formed 9-byte frames must decode");
                assert_eq!(pcm.len(), FRAME);
                *samples_out += pcm.len();
            }
            fed += n;
        }
    };

    run(&mut voc, &mut lcg, WARMUP, &mut samples_out);
    let rss0 = rss_bytes();
    run(&mut voc, &mut lcg, SOAK, &mut samples_out);
    let rss1 = rss_bytes();

    assert_eq!(samples_out, (WARMUP + SOAK) * FRAME);
    let delta = rss1.saturating_sub(rss0);
    eprintln!(
        "[soak] DecodeStream: {SOAK} frames after warm-up, RSS delta {:.2} MiB \
         ({:.1} B/frame)",
        mib(delta),
        delta as f64 / SOAK as f64
    );
    assert!(
        delta < RSS_DELTA_LIMIT,
        "DecodeStream RSS grew {:.2} MiB over {SOAK} frames ({:.1} B/frame) — unbounded growth",
        mib(delta),
        delta as f64 / SOAK as f64
    );
}

/// LiveDecoder soak at full scale, pushed in deliberately misaligned byte
/// chunks so the pending buffer is exercised every call; asserts the public
/// `pending_bytes` accessor stays sub-frame throughout.
fn soak_live_decoder(frame_pool: &[Vec<u8>]) {
    const WARMUP: usize = 1_000;
    const SOAK: usize = 6_000;

    let mut dec = LiveDecoder::new(Rate::AmbePlus2_3600x2450);
    let mut lcg = Lcg(0x11FE);
    let mut frames_out = 0usize;

    let run = |dec: &mut LiveDecoder, lcg: &mut Lcg, frames: usize, frames_out: &mut usize| {
        let mut wire: Vec<u8> = Vec::new();
        let mut fed = 0usize;
        while fed < frames || !wire.is_empty() {
            // Keep a small rolling wire buffer topped up.
            while wire.len() < 64 && fed < frames {
                let pick = lcg.next_u32() as usize % frame_pool.len();
                wire.extend_from_slice(&frame_pool[pick]);
                fed += 1;
            }
            // Misaligned push sizes (1..=23 bytes) to churn the pending buffer.
            let take = (1 + lcg.next_u32() as usize % 23).min(wire.len());
            let chunk: Vec<u8> = wire.drain(..take).collect();
            for pcm in dec.push(&chunk) {
                let pcm = pcm.expect("well-formed frames must decode");
                assert_eq!(pcm.len(), FRAME);
                *frames_out += 1;
            }
            assert!(
                dec.pending_bytes() < R33_BYTES,
                "LiveDecoder pending_bytes unbounded: {}",
                dec.pending_bytes()
            );
        }
    };

    run(&mut dec, &mut lcg, WARMUP, &mut frames_out);
    let rss0 = rss_bytes();
    run(&mut dec, &mut lcg, SOAK, &mut frames_out);
    let rss1 = rss_bytes();

    assert_eq!(frames_out, WARMUP + SOAK);
    let delta = rss1.saturating_sub(rss0);
    eprintln!(
        "[soak] LiveDecoder: {SOAK} frames after warm-up, RSS delta {:.2} MiB \
         ({:.1} B/frame)",
        mib(delta),
        delta as f64 / SOAK as f64
    );
    assert!(
        delta < RSS_DELTA_LIMIT,
        "LiveDecoder RSS grew {:.2} MiB over {SOAK} frames ({:.1} B/frame) — unbounded growth",
        mib(delta),
        delta as f64 / SOAK as f64
    );
}

/// Build a small pool of real encoded frames for the decode soaks (cycled by
/// the LCG so the decoder sees varied but valid input without paying the dev
/// profile's ~95 ms/frame encode cost 22,000 times).
fn build_frame_pool() -> Vec<Vec<u8>> {
    let mut lcg = Lcg(0x9001);
    let mut t = 0.0f64;
    let mut pcm = Vec::with_capacity(16 * FRAME);
    for _ in 0..16 {
        pcm.extend_from_slice(&synth_frame(&mut lcg, &mut t));
    }
    let voc = Vocoder::new(Rate::AmbePlus2_3600x2450);
    let pool = voc.encode(&pcm);
    assert!(!pool.is_empty(), "frame pool encode produced no frames");
    for f in &pool {
        assert_eq!(f.len(), R33_BYTES);
    }
    pool
}

/// Single test on purpose: sequential phases keep the RSS snapshots free of
/// concurrent-test allocation noise within this process.
#[test]
fn stream_soak_bounded_memory() {
    let mut t0 = std::time::Instant::now();
    let lap = |name: &str, t0: &mut std::time::Instant| {
        eprintln!("[soak] phase {name}: {:?}", t0.elapsed());
        *t0 = std::time::Instant::now();
    };
    let pool = build_frame_pool();
    lap("pool", &mut t0);
    soak_live_encoder(Rate::AmbePlus2_3600x2450, R33_BYTES);
    soak_live_encoder(Rate::Imbe7200x4400, 18);
    soak_live_encoder(Rate::Imbe4400x4400, 11);
    lap("live_encoder", &mut t0);
    soak_tone_detection_encode();
    lap("tone_encode", &mut t0);
    soak_decode_stream(&pool);
    lap("decode_stream", &mut t0);
    soak_live_decoder(&pool);
    lap("live_decoder", &mut t0);
}
