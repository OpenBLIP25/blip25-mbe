//! Pinned codec output — the regression gate the parity tests cannot be.
//!
//! Every other test in this crate compares the `Vocoder` façade against
//! `blip25_codec` called directly. That verifies *routing*: it proves the
//! façade dispatches to the engine correctly. It cannot prove the engine is
//! right, because if the engine changes, both sides of the comparison change
//! together and the test still passes.
//!
//! This file closes that hole. The expected values below are hashes of actual
//! encode and decode output, captured on x86-64 Linux. They are compared
//! against literals, so nothing about the engine can move without failing
//! here — including a float divergence on a different architecture, which is
//! what makes the cross-platform CI matrix meaningful. The crate's core claim
//! is bit-exact reproduction of a reference vocoder; this is where that claim
//! is actually enforced.
//!
//! **If a change to the codec makes this fail, that is the test working.** Do
//! not re-bless the constants to make CI green without first establishing that
//! the new output is correct — the whole point is that a silent output change
//! is impossible. To re-bless deliberately, run with `--nocapture` and copy
//! the printed values.

use blip25_mbe::vocoder::{LiveEncoder, Rate, Vocoder};

const ALL_RATES: [Rate; 4] = [
    Rate::Imbe7200x4400,
    Rate::Imbe4400x4400,
    Rate::AmbePlus2_3600x2450,
    Rate::AmbePlus2_2450x2450,
];

fn rate_key(r: Rate) -> &'static str {
    match r {
        Rate::Imbe7200x4400 => "imbe_7200x4400",
        Rate::Imbe4400x4400 => "imbe_4400x4400",
        Rate::AmbePlus2_3600x2450 => "ambe2_3600x2450",
        Rate::AmbePlus2_2450x2450 => "ambe2_2450x2450",
        _ => "unknown",
    }
}

/// FNV-1a 64. Defined here rather than pulled in so the gate has no dependency
/// that could itself change the hash.
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

/// Floor of the square root, by integer Newton iteration. Written out rather
/// than using `f64::sqrt` (or `i64::isqrt`) because the envelope below is a
/// pinned constant: the result must be bit-identical on every target and
/// independent of the toolchain's libm and of std version drift.
fn isqrt_i64(n: i64) -> i64 {
    assert!(n >= 0, "isqrt_i64 of a negative value: {n}");
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Per-frame RMS of decoded PCM, in 160-sample (20 ms) frames.
///
/// Integer arithmetic end to end: sum of squares in `i64`, integer divide by
/// the frame length, then [`isqrt_i64`].
fn frame_rms(pcm: &[i16]) -> Vec<u32> {
    assert!(!pcm.is_empty(), "frame_rms on empty PCM");
    assert_eq!(
        pcm.len() % 160,
        0,
        "PCM length {} is not a whole number of 160-sample frames",
        pcm.len()
    );
    pcm.chunks_exact(160)
        .map(|f| {
            let ss: i64 = f.iter().map(|&s| (s as i64) * (s as i64)).sum();
            isqrt_i64(ss / 160) as u32
        })
        .collect()
}

/// How coarsely the per-frame RMS is quantised before hashing. Deliberately
/// lossy: the bucket hash must NOT move for sample-level drift, only for a
/// change in output *level*. That immunity is the hash's alone — `max` and
/// `mean` are unquantised and a one-LSB perturbation can shift them by a count.
/// See [`ENVELOPE_HASH_TRIAGE`].
const ENVELOPE_SHIFT: u32 = 6;

/// Coarse RMS envelope of decoded PCM: `(hash, max, mean)`.
///
/// The hash covers the shape of the whole envelope; `max` and `mean` are
/// carried alongside it because they are readable — a reviewer can see at a
/// glance whether output went quiet or hot without decoding anything. They are
/// taken over the *unquantised* per-frame RMS values, so they are finer-grained
/// than the hash and will move for level changes too small to shift a bucket.
fn envelope(pcm: &[i16]) -> (u64, u32, u32) {
    let rms = frame_rms(pcm);
    let buckets: Vec<u8> = rms
        .iter()
        .map(|&r| (r >> ENVELOPE_SHIFT).min(255) as u8)
        .collect();
    let max = rms.iter().copied().max().expect("at least one frame");
    let mean = (rms.iter().map(|&r| r as u64).sum::<u64>() / rms.len() as u64) as u32;
    (fnv1a(&buckets), max, mean)
}

/// Deterministic speech-like test signal: three summed harmonics with an
/// amplitude envelope, plus a reproducible pseudo-random component so the
/// unvoiced path is exercised too. No floating-point transcendentals in the
/// generator itself — a sine table would be re-derived per platform — so the
/// *input* is bit-identical everywhere and only the codec is under test.
fn test_pcm(frames: usize) -> Vec<i16> {
    let n = frames * 160;
    let mut out = Vec::with_capacity(n);
    // xorshift64*, fully specified integer arithmetic.
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    for i in 0..n {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let noise = ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 48) as i32) - 32768 / 2;

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
        let env = 1 + ((i / 160) % 5) as i32; // 1..5, varies per frame
        let v = (tri(51) * 90 + tri(23) * 40 + tri(11) * 18) * env / 5 + noise / 24;
        out.push(v.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
    }
    out
}

/// As [`test_pcm`], but with two quiet stretches: frames 6..10 attenuated to a
/// low noise floor (below the encode-side silence gate's threshold but
/// non-zero, the realistic inter-word case) and frames 17..19 true digital
/// silence. [`test_pcm`] itself never drops below the threshold — its quietest
/// frame sits around RMS 1000.
///
/// What that buys is attenuated and digital-zero frames driven through the
/// whole-buffer and streaming encode paths. It does not pin the silence gate
/// itself: on every frame this signal pushes under `SILENCE_RMS` the gain byte
/// is already at or below `SILENCE_GAIN_CAP`, so the clamp is a no-op and no
/// hash here observes it.
///
/// Kept separate from [`test_pcm`] so a test-only change to this signal can
/// never force a re-bless of the `GOLDEN` values.
fn test_pcm_with_silence(frames: usize) -> Vec<i16> {
    let mut pcm = test_pcm(frames);
    for (f, chunk) in pcm.chunks_mut(160).enumerate() {
        match f {
            6..=9 => chunk.iter_mut().for_each(|s| *s /= 64),
            17..=18 => chunk.fill(0),
            _ => {}
        }
    }
    pcm
}

/// Deterministic wire bits for the decode direction, derived from a fixed
/// integer sequence rather than from this crate's encoder, so a bug that
/// affects both directions cannot cancel out.
fn test_bits(rate: Rate, frames: usize) -> Vec<u8> {
    let n = rate.fec_frame_bytes() * frames;
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state >> 56) as u8
        })
        .collect()
}

// ── Expected values, captured on x86-64 Linux ─────────────────────────────
//
// Three hashes per rate, because they cover different machinery:
//   encode    — PCM in, wire bits out
//   conceal   — decode of random bytes; mostly drives the FEC-error /
//               erasure repeat+mute concealment path
//   roundtrip — decode of *valid* frames from this crate's encoder; this is
//               the one that actually exercises dequantization and synthesis.
//               Pinning it to a literal is what stops it from being a
//               self-consistent no-op.
//
// The r34 wire layout is a permutation, so encode->decode is self-inverse
// under any consistent ordering: a change to `R34_BIT_ORDER` moves the
// `ambe2_2450x2450` encode and conceal hashes but leaves roundtrip fixed. That
// asymmetry is how a pure relabeling is told apart from an audio change.
//
// (rate key, encode, conceal, roundtrip)
const GOLDEN: [(&str, u64, u64, u64); 4] = [
    (
        "imbe_7200x4400",
        // encode + roundtrip moved with the IMBE-scoped fricative-protect +
        // per-3-harmonic-band voicing veto: the b1 V/UV word now withholds the
        // snap-to-all-voiced on fricative frames (voiced low band, unvoiced high
        // SH/T) and unvoices whole 3-harmonic bands whose window-transform error
        // clears band_thr=0.5. Fewer voiced high bands -> different b1 bits
        // (encode) and less high-band buzz on decode (roundtrip). Conceal is
        // escape/erasure frames (voicing-independent), unmoved. IMBE only; the
        // AMBE+2 path uses VoicingConfig::default() (both flags false), unmoved.
        // encode + roundtrip moved again with the IMBE voicing freq_tilt = -0.75
        // (inverts the high-harmonic tilt, voicing more high bands to match the
        // reference's aggressive high-band voicing). Conceal unmoved.
        // (The bit-exact FFT spectrum mechanism amplitudes are whole-buffer-only,
        // so they do NOT move these `encode_pcm`/streaming hashes — see
        // GOLDEN_ENCODE_PATHS' whole_buffer column.)
        0xb37bddba299aca91,
        0x4632ab258479dae6,
        0xf21cd2f2487b3388,
    ),
    (
        "imbe_4400x4400",
        // encode + conceal moved when the info-only FEC strip was corrected: the
        // 11-byte frame now carries the encoder's u0..u7 losslessly (mbe's
        // Hamming[15,11] column order was reversed vs the reference engine, so
        // the strip's imbe7200 re-decode phantom-corrected u4..u6 in ~1139/3699
        // clean frames, corrupting the shipped b1/b2 bits). Encode = the corrected
        // shipped strip; conceal = decoding external info frames now uses the
        // engine-compatible Hamming layout. Roundtrip of our own output was already
        // lossless (the strip and re-add table-mismatches cancelled) so it is
        // unchanged.
        // encode + roundtrip moved again with the fricative-protect + per-band
        // voicing veto (see imbe_7200x4400); conceal unmoved.
        // encode + roundtrip moved again with the IMBE voicing freq_tilt = -0.75
        // (see imbe_7200x4400); conceal unmoved.
        // (bit-exact FFT mechanism amplitudes are whole-buffer-only; see
        // imbe_7200x4400.)
        0x1dd56369649fe063,
        0xdd34466dc38b7bf6,
        0xc3bbff341f6735dd,
    ),
    (
        // conceal moved 0xf3e9…b9b5 -> 0x3427…d9f3 when the escape space was
        // repartitioned on `û₀(11..6)` instead of `b̂₀ ∈ [120,123]`. Exactly
        // one frame of the random fixture changes class (frame 13: b̂₀ = 127,
        // b̂₁ = 20 -> Silence). Its own PCM is unchanged — both the old and the
        // new path emit zeros for it — but the old path reached that silence by
        // falling through the inert erasure gate into the normal exit, which
        // set `last_good_bytes` to the escape frame. A later erasure then
        // "repeated" a frame that decodes to silence. The level rise (RMS max
        // 1215 -> 1383, mean 203 -> 225) is later repeats recovering a real
        // voice frame as their source. Encode column unchanged.
        "ambe2_3600x2450",
        // b2 gain: replaced the frame-energy affine gain override
        // (gamma = 0.65*log2(E) - 5.74) with quantize_indices' spectral gamma
        // (gamma = mean(log2 M_l) + 0.5*log2(L)) to match the reference DLL gain
        // quantizer (sub_0x10313ec0), which works from the spectral
        // log-amplitude vector, not frame energy. Lifts b2 vs the reference
        // from 6.3->38.1 (clean) / 7.5->34.6 (dam) / 10.7->40.2 (mark) with
        // every other field (b0,b1,b3..b8) held exactly. Encode moves (the b2
        // wire bits change) and roundtrip follows (decode reflects the new
        // bits); conceal is escape frames (gain-independent), unmoved.
        //
        // b2 gain: constant -0.13 gamma offset correcting quantize_gain
        // rounding-convention vs the DLL fixed-point renorm. Replaced the
        // vf-gated -0.03 nudge with an ungated -0.13 bias on gamma, centring the
        // nearest-neighbour gain_o decision onto the reference cells. Lifts b2 vs the
        // reference clean 39.2->42.7 / dam 37.0->43.5 / mark 40.5->47.1 with
        // every other field (b0,b1,b3..b8) held. Encode moves (b2 wire bits) and
        // roundtrip follows; conceal is escape frames (gain-independent), unmoved.
        //
        // omega_0-refined STEP: amps_from_bins now keys band_decompress's STEP on
        // the continuous omega_0 count within each L cell (step_for_omega) instead
        // of the integer-L modal lookup. This is the finer STEP driver the modal
        // table doc flagged; it lifts the shared T'-shape amplitude fields vs the
        // reference across the board (b3/b4/b6/b7/b8 all up, ALL9 avg 0.47->0.77)
        // with b2/b5 essentially flat. Encode moves (amplitude wire bits) and
        // roundtrip follows; conceal (escape frames) unmoved. AMBE path only;
        // IMBE amps_from_bins call site unchanged.
        //
        // exact omega_0->STEP closed form: step_for_omega now returns
        // round(omega_0 * 2^18/pi) — the Q11 fixed-point of the harmonic spacing
        // in 256-pt-FFT bins — replacing the modal-table + continuous-count
        // approximation, which diverged from the reference STEP on 103/120 pitch
        // entries. Lifts the shared T'-shape fields across the board (b2/b3/b4/
        // b5/b6/b7/b8 all up, ALL9 avg 0.77->1.21). AMBE path only; IMBE unchanged.
        0x8b6a04427d63792f,
        0x3427e1faaeaad9f3,
        0x2abe4208e5b01f2b,
    ),
    (
        "ambe2_2450x2450",
        // exact omega_0->STEP closed form; see the note on ambe2_3600x2450.
        0x08ae27dd49763b4b,
        0x19da69148650bda0,
        0x6d0ebaeb4e316b2e,
    ),
];

// ── Coarse RMS envelope of the two decode columns ─────────────────────────
//
// These pins add no detection. `envelope()` is a deterministic function of the
// same decoded PCM that the `GOLDEN` conceal and roundtrip hashes already
// cover, so every envelope movement implies a hash movement; the hash is the
// gate, and it stays the gate. What the envelope adds is *resolution after the
// gate fires*. The `GOLDEN` hashes are all-or-nothing — one sample off by an
// LSB reports exactly the way the output going silent reports — and these
// literals say which of the two it was without decoding anything.
//
// Only the two *decode* columns get one. The `encode` column is wire bytes, not
// audio; an RMS envelope over it would be meaningless.
//
// The pair is read together, and the test says so on failure:
const ENVELOPE_HASH_TRIAGE: &str = "\
    TRIAGE RULE — the envelope grades a failure the hash has already caught;\n\
    it never catches one on its own. Read the two together:\n\
    \x20 * hash moved, envelope IDENTICAL  -> sample-level drift only (LSB /\n\
    \x20     phase / rounding). Level, timing and voicing are unchanged, so this\n\
    \x20     is consistent with a benign refactor. It is NOT proof of one: verify\n\
    \x20     the change was meant to be bit-affecting, then re-bless the hash and\n\
    \x20     leave the envelope literals alone.\n\
    \x20 * hash moved, envelope ALSO moved -> the output level or its shape over\n\
    \x20     time changed (frames gone quiet or hot, a dropped or muted frame, a\n\
    \x20     gain or level-constant change). Listen before re-blessing anything.\n\
    \x20     One exception, measured below: a bucket hash that holds while `max`\n\
    \x20     or `mean` shifts by a single count is within one-LSB dither range\n\
    \x20     and grades as drift, not as level.\n\
    \x20 * envelope moved, hash UNCHANGED  -> unreachable; the envelope is a pure\n\
    \x20     function of the same PCM the hash covers. Suspect the envelope\n\
    \x20     helper, not the codec.";
//
// `max` and `mean` are over the raw per-frame RMS, so they are sharper than the
// hash and can move on their own for a level change too small to cross an
// `ENVELOPE_SHIFT` bucket boundary. Envelope "moved" means any of the three.
//
// The split is mutation-verified, not assumed. Perturbations of the decode
// path, each reverted:
//
//   * XOR 1 into the shared final packer (`round_pack16` in blip25-codec) —
//     every decoded sample off by one LSB, uniformly. All eight decode hashes
//     moved; not one envelope value moved.
//
//     That result is specific to a uniform bit flip. It does NOT generalise to
//     one-LSB perturbations as a class, so it is not a "benign refactor
//     signature": swapping the XOR in that same function for a data-dependent
//     ±1 (magnitude shrunk by one LSB, sign preserved) leaves all eight bucket
//     hashes standing but moves `max` and/or `mean` on six of the eight
//     columns — including both AMBE+2 roundtrip means, 400 -> 399 and
//     391 -> 390. `mean` is an integer average of per-frame RMS values, and on
//     AMBE+2 those average near 390-400, so one count is about 0.25% —
//     resolution finer than full-LSB dither. A ±1 shift in `max`/`mean` with
//     the bucket hashes intact is therefore evidence of drift, not of level.
//   * `ImbeDecoder::ola_offset` (the additive IMBE level constant, Q4.11 log2)
//     0 -> 256, a +9.1% level change. Both IMBE rates moved hash, bucket hash,
//     max and mean together; the AMBE+2 rates correctly did not move at all,
//     since `ola_offset` is IMBE-only.
//
// How much level discrimination `ENVELOPE_SHIFT = 6` leaves depends strongly on
// the rate, because the eight columns do not share a dynamic range. Measured
// bucket occupancy (bucket 0 is the near-silent frames every column has):
//
//     rate               conceal                roundtrip
//     imbe_7200x4400     0..169, 13 distinct    0..28, 13 distinct
//     imbe_4400x4400     0..95,  20 distinct    0..29, 13 distinct
//     ambe2_3600x2450    0..18,   9 distinct    0..10,  8 distinct
//     ambe2_2450x2450    0..51,  17 distinct    0..10,  8 distinct
//
// IMBE reaches RMS 10869; the AMBE+2 roundtrip columns top out at 642 and
// resolve their whole range into eight bucket values. So the bucket hash is the
// weak member on AMBE+2 and the scalars carry the discrimination there — the
// reverse of IMBE, where `ola_offset` 0 -> 16 (+0.54%) moves the bucket hashes.
// Scaling decoded PCM by -0.39% (the envelope's view of a level change) moves
// neither AMBE+2 conceal bucket hash, only their `max` and `mean`; the
// ambe2_3600x2450 conceal bucket hash still holds at -0.54%. Tightening the
// shift would sharpen AMBE+2 at the cost of the LSB immunity above, so it stays
// at 6 and AMBE+2 level detection rests on `max`/`mean`.
//
// (rate key, conceal (hash, max, mean), roundtrip (hash, max, mean))
const GOLDEN_ENVELOPE: [(&str, (u64, u32, u32), (u64, u32, u32)); 4] = [
    (
        "imbe_7200x4400",
        (0x64f1cd73def011a4, 10869, 1420),
        // Roundtrip moved with the fricative-protect + per-band voicing veto
        // (see GOLDEN note): less voiced high-band energy on decode. Conceal
        // (escape/erasure frames) unmoved. (The bit-exact FFT mechanism
        // amplitudes are whole-buffer-only and do not reach this streaming
        // roundtrip envelope.)
        (0x3181127c93fe7385, 1575, 1172),
    ),
    (
        "imbe_4400x4400",
        // Conceal envelope moved with the corrected info-only FEC strip (see the
        // GOLDEN note): decoding external info frames now uses the engine-compatible
        // Hamming layout. Roundtrip of our own output is unchanged.
        (0xe080fe029f60f40e, 6139, 1782),
        // Roundtrip moved with the fricative-protect + per-band voicing veto.
        // (bit-exact FFT mechanism amplitudes are whole-buffer-only; see
        // imbe_7200x4400.)
        (0xf9b6d1bb5115fe10, 1574, 1163),
    ),
    (
        "ambe2_3600x2450",
        // Roundtrip column moved with the b2 gain change (see GOLDEN note):
        // dropping the frame-energy affine gain override lets quiet frames use
        // the spectral gamma, raising decoded level (RMS max 590->1531). Decoded
        // audio is not a promotion gate; the wire b2 match is the win. Conceal
        // is gain-independent escape frames, unmoved.
        (0x973563d15eecf938, 1383, 225),
        // Roundtrip moved with the exact omega_0->STEP closed form (see GOLDEN note).
        (0x489d02dfcc0f868e, 1380, 999),
    ),
    (
        "ambe2_2450x2450",
        (0xea4b9f810ca80c75, 3319, 859),
        (0x4c58aaf119ccc2ed, 1380, 979),
    ),
];

// ── The other two shipped encode entry points ─────────────────────────────
//
// `GOLDEN` above hashes `Vocoder::encode_pcm`, the low-level per-frame
// primitive. That is NOT the path the documentation tells callers to use, and
// for AMBE+2 it is not even the same analysis chain: `Vocoder::encode` runs the
// reverse-engineered reference pitch chain, the RE'd `b1_track` voicing with its ±3
// lag search, the silence gate and the optional tone overlay — none of which
// `encode_pcm` touches. `LiveEncoder` is a third path again: byte-exact to
// `Vocoder::encode` for AMBE+2 (via `AmbeStream`), but falling back to
// `encode_pcm` for the two IMBE rates.
//
// These hashes are the only guard on the reference pitch injection, the voicing
// tracker, the lag search, `BLAG`, and the `AmbeStream` duplicates — none of
// those move a `GOLDEN` hash.
//
// A silence gain-cap once clamped near-silent frames' b2, but it was removed:
// it was a documented no-op until the reference-exact b2 gain landed, at which point
// quiet frames began emitting b2 > 2 — the values the reference itself emits
// (reference does not silence-cap) — so the clamp had started to *deviate* from reference
// rather than track it. Removing it restored the reference match and moved b2 (only
// b2) on both AMBE+2 rates, which is why both AMBE+2 hashes below moved.
//
// Two facts are visible in the values themselves, and both are load-bearing:
//
//   * For BOTH AMBE+2 rates, `whole_buffer != live` — but with Route A the two
//     now agree on pitch (b0) AND amplitude/gain (b2..b8): the streaming
//     amplitude Encoder gained the same two-frame look-ahead, so both feed
//     `band_decompress` the identical pitch-aligned live `gap2` window. The
//     residual hash difference is voicing (b1) alone — whole-buffer's global ±3
//     lag search vs the streamer's bounded `b1_track` — on a few frames of this
//     silence-containing fixture. The structural invariant below enforces that
//     b0 and b2..b8 match byte-for-byte rather than asserting it only in prose.
//   * For BOTH IMBE rates, `whole_buffer != live`. That is NOT a property to
//     preserve — it is a known defect:
//     `Vocoder::encode` injects the RE'd reference pitch for IMBE while
//     `LiveEncoder` falls back to the clean-room estimator, so the same PCM
//     gives audibly different output depending on which API a caller reaches
//     for. Closing that WILL move the IMBE `live` hashes, and that re-bless is
//     the fix landing, not a regression.
//
// `whole_buffer` also differs from the `GOLDEN` encode hash for every rate,
// since that one covers the per-frame primitive.
//
// (rate key, whole_buffer, live)
const GOLDEN_ENCODE_PATHS: [(&str, u64, u64); 4] = [
    // both paths moved with the fricative-protect + per-band voicing veto (b1
    // V/UV bits change); IMBE only, AMBE+2 rows below unmoved (default config).
    // whole_buffer ONLY moved again with the bit-exact FFT spectrum -> mechanism
    // amps + re-fit gain DC biases (b2..b8 bits change): the whole-buffer path
    // opts into the two-frame look-ahead + mechanism amplitudes, while the live
    // (LiveEncoder) streaming path keeps the one-frame look-ahead + synthesized-
    // DFT proxy and is byte-unchanged. IMBE only, AMBE+2 unmoved.
    // whole_buffer ONLY moved again with the exact per-frame band_decompress
    // step: the IMBE mechanism amplitude path now derives band_decompress's STEP
    // from the quantized IMBE pitch index (STEP = fund_freq(b0) >> 13, pure
    // truncation on the Q1.31 fundamental) instead of round(omega_0*2^18/pi) fed
    // the continuous pitch estimate. This reproduces the reference DLL's captured
    // step 45/45 on the mark capture (vs ~0% for the round-of-continuous form) and
    // lifts the shared amplitude fields vs the reference across the board (amps|L
    // clean 60.0->60.7 / dam 65.2->66.1 / mark 77.5->77.7 / noisy 68.5->76.2; ALL
    // clean 0.05->0.24 / mark 0.11->0.54 / noisy 0.64->6.53). Amplitude wire bits
    // change (b3..b8) so whole_buffer moves; the live (LiveEncoder) path keeps the
    // synthesized-DFT proxy and is byte-unchanged. IMBE only: the step is derived
    // inside analyze_imbe_frame from b0_imbe; the shared omega-keyed step_for_omega
    // used by the frozen AMBE+2 Route A path is untouched, so both AMBE+2 rows hold.
    // whole_buffer ONLY moved again wiring IMBE voicing through the shared reference
    // metric: analyze_imbe_frame now expands `b1_track`'s 5-bit voicing word
    // (build_maskword + gen_vmask on the IMBE ω0/L grid, AND-binned per band)
    // instead of the float decide_voicing_cfg path, which over-voiced fricative
    // high bands. b1 V/UV bits change (broad b1 vs reference clean 64.8->76.5 /
    // dam 66.5->78.0 / mark 56.5->72.5 / noisy 65.9->76.2; clean SH fricative
    // frames now unvoice to match reference). The live (LiveEncoder) path keeps the
    // float voicing and is byte-unchanged; AMBE+2 rows unmoved (IMBE-only consumer).
    // both paths moved again with the inter-word gap ring-out fix: (1) the low-L
    // gain floor in sa_encode is now energy-gated (src_rms < IMBE_SILENCE_RMS &&
    // b_vec[1]==0, dropping the nh<15 proxy) so low-energy unvoiced word-edge
    // frames get the -2100 gain floor without touching high-energy fricatives, and
    // (2) analyze_imbe_frame forces all bands unvoiced when src_rms < the same
    // silence threshold. Both fire on the silence-containing fixture, changing the
    // gain (b2) and voicing (b1) wire bits on the low-energy gap frames; both the
    // whole_buffer and live paths run the shared quantizer/analysis so both move.
    // IMBE only (src_rms plumbed only through the IMBE analyze/quantize path);
    // AMBE+2 rows below unmoved.
    // whole_buffer ONLY moved again with the release-edge alignment: at a word's
    // trailing edge our boundary frame stays hot (b0/L/b2) one frame after its
    // voicing has released to fully unvoiced, seeding the decoder's differential-
    // amplitude predictor and ringing into the inter-word gap. analyze_imbe_frame
    // now detects that edge (previous frame voiced, this frame fully unvoiced,
    // still loud, NEXT frame already in the silence regime) and forces the same
    // silence release the next frame gets — pin b0 to the silence default (25/L14)
    // and route through the low-energy gain floor. b0/L/b1/b2 wire bits change on
    // the release-edge frames (e.g. clean f601 32/16/0/47 -> 25/14/0/41, matching
    // reference 25/14/0). The clip guard (frame must be already fully unvoiced) keeps
    // voiced word tails and loud unvoiced fricatives untouched; broad b1 holds
    // (clean 76.7 / dam 78.6 / mark 73.3 / noisy 76.2). The live (LiveEncoder)
    // path uses float voicing and the edge does not fire on the fixture, so it is
    // byte-unchanged; AMBE+2 rows below unmoved (IMBE-only analyze/quantize path).
    ("imbe_7200x4400", 0x65d9100dd278a904, 0x3a36cb44d0c9bbfb),
    ("imbe_4400x4400", 0x57ec6d93ce04289d, 0xd22380d77b32cbcb),
    // Route A: the streaming amplitude Encoder now runs the same two-frame
    // look-ahead as whole-buffer, so both feed `band_decompress` the identical
    // pitch-aligned live `gap2` window and agree on the amplitude/gain fields
    // (b2..b8). `whole_buffer != live` here survives only because voicing (b1)
    // still comes from two different trackers — whole-buffer's global ±3 lag
    // search vs the streamer's bounded `b1_track` — which differ on a few frames
    // of this silence-containing fixture. The `live` hashes moved (b2..b8 now
    // aligned) while `whole_buffer` held.
    ("ambe2_3600x2450", 0x42b6c4be42ebaa3d, 0x2b2e6aa6e5fd5547),
    ("ambe2_2450x2450", 0x6c0c03fee044490b, 0xc8b4d346ca730bd6),
];

fn golden_for(key: &str) -> (u64, u64, u64) {
    GOLDEN
        .iter()
        .find(|(k, _, _, _)| *k == key)
        .map(|(_, e, c, r)| (*e, *c, *r))
        .unwrap_or_else(|| panic!("no GOLDEN row for rate {key}"))
}

/// Assert `table` is keyed by exactly `expected`, in either direction.
///
/// Missing means a rate the gates drive has no pinned row; extra means a row
/// is keyed to something nothing drives, which is what a renamed or mistyped
/// key looks like.
fn assert_keyed_by(table: &str, rows: &[&str], expected: &[&str]) {
    for key in expected {
        assert!(
            rows.contains(key),
            "{table} has no row for {key}. The gates look their expected \
             values up by that key, so an unlisted rate is an unpinned rate."
        );
    }
    for key in rows {
        assert!(
            expected.contains(key),
            "{table} has a row keyed {key:?}, which is not a rate these tests \
             drive. A renamed or mistyped key leaves the rate it used to cover \
             unpinned."
        );
    }
}

/// [`Rate`] is `#[non_exhaustive]` and documents that more variants will
/// land. Adding one must fail here — with a diagnosis — rather than leaving a
/// rate silently unpinned in whichever table forgot it. Run from both gates,
/// so neither can be the one that skips the check.
fn assert_every_rate_is_pinned() {
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

    let golden: Vec<&str> = GOLDEN.iter().map(|(k, _, _, _)| *k).collect();
    let envelope: Vec<&str> = GOLDEN_ENVELOPE.iter().map(|(k, _, _)| *k).collect();
    let paths: Vec<&str> = GOLDEN_ENCODE_PATHS.iter().map(|(k, _, _)| *k).collect();

    assert_keyed_by("GOLDEN", &golden, &all);
    assert_keyed_by("GOLDEN_ENVELOPE", &envelope, &all);
    assert_keyed_by("GOLDEN_ENCODE_PATHS", &paths, &all);
}

const FRAMES: usize = 24;

fn encode_frames(rate: Rate) -> Vec<u8> {
    let pcm = test_pcm(FRAMES);
    let mut v = Vocoder::new(rate);
    let mut all = Vec::new();
    for frame in pcm.chunks_exact(160) {
        all.extend(v.encode_pcm(frame).expect("encode a full frame"));
    }
    all
}

/// `Vocoder::encode` — the whole-buffer method the docs recommend.
fn encode_whole_buffer(rate: Rate, pcm: &[i16]) -> Vec<u8> {
    Vocoder::new(rate).encode(pcm).concat()
}

/// `LiveEncoder` — the streaming path, fed in deliberately misaligned chunks so
/// the internal buffering and look-ahead are exercised rather than assumed.
fn encode_live(rate: Rate, pcm: &[i16]) -> Vec<u8> {
    let mut enc = LiveEncoder::new(rate);
    let mut all = Vec::new();
    for chunk in pcm.chunks(377) {
        for frame in enc.push(chunk) {
            all.extend(frame.expect("live encode a frame"));
        }
    }
    for frame in enc.flush().expect("live encoder flush") {
        all.extend(frame);
    }
    all
}

/// Per-frame b-fields (`[b0..b8]`) of a concatenated AMBE+2 bitstream — FEC r33
/// (9B) or no-FEC r34 (7B). Lets the streaming/whole-buffer structural invariant
/// compare pitch/voicing (b0/b1) apart from amplitude/gain (b2..b8).
fn ambe_fields_of(rate: Rate, bits: &[u8]) -> Vec<[u16; 9]> {
    let n = rate.fec_frame_bytes();
    bits.chunks_exact(n)
        .map(|f| match rate {
            Rate::AmbePlus2_3600x2450 => blip25_mbe::rate33::frame::fields_from_fec(f),
            Rate::AmbePlus2_2450x2450 => blip25_mbe::rate33::frame::fields_from_no_fec(f),
            _ => unreachable!("ambe_fields_of called on a non-AMBE+2 rate"),
        })
        .collect()
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

#[test]
fn codec_output_is_pinned() {
    assert_every_rate_is_pinned();

    let mut actual = Vec::new();
    let mut actual_env = Vec::new();
    let mut mismatches = Vec::new();
    let mut env_mismatches = Vec::new();
    let mut decode_hash_moved = false;

    for rate in ALL_RATES {
        let key = rate_key(rate);
        let encoded = encode_frames(rate);
        let enc = fnv1a(&encoded);
        let conceal_pcm = decode_all(rate, &test_bits(rate, FRAMES));
        let roundtrip_pcm = decode_all(rate, &encoded);
        let conceal = hash_i16(&conceal_pcm);
        let roundtrip = hash_i16(&roundtrip_pcm);
        actual.push((key, enc, conceal, roundtrip));

        let (w_enc, w_con, w_rt) = golden_for(key);
        for (label, got, want) in [
            ("encode", enc, w_enc),
            ("conceal", conceal, w_con),
            ("roundtrip", roundtrip, w_rt),
        ] {
            if want != 0 && got != want {
                mismatches.push(format!(
                    "{key} {label}: expected {want:#018x}, got {got:#018x}"
                ));
                decode_hash_moved |= label != "encode";
            }
        }

        // Envelope of the same two decode buffers the hashes above cover.
        let env_con = envelope(&conceal_pcm);
        let env_rt = envelope(&roundtrip_pcm);
        actual_env.push((key, env_con, env_rt));

        let want_env = GOLDEN_ENVELOPE
            .iter()
            .find(|(k, _, _)| *k == key)
            .map(|(_, c, r)| (*c, *r))
            .unwrap_or_else(|| panic!("no GOLDEN_ENVELOPE row for rate {key}"));
        for (label, got, want) in [
            ("conceal", env_con, want_env.0),
            ("roundtrip", env_rt, want_env.1),
        ] {
            if want == (0, 0, 0) {
                continue; // placeholder; the assertion below catches it
            }
            if got.0 != want.0 {
                env_mismatches.push(format!(
                    "{key} {label} envelope hash: expected {:#018x}, got {:#018x}",
                    want.0, got.0
                ));
            }
            if got.1 != want.1 {
                env_mismatches.push(format!(
                    "{key} {label} RMS max: expected {}, got {}",
                    want.1, got.1
                ));
            }
            if got.2 != want.2 {
                env_mismatches.push(format!(
                    "{key} {label} RMS mean: expected {}, got {}",
                    want.2, got.2
                ));
            }
        }
    }

    println!("\n// Re-bless by copying this block into GOLDEN:");
    println!("const GOLDEN: [(&str, u64, u64, u64); 4] = [");
    for (k, e, c, r) in &actual {
        println!("    (\"{k}\", {e:#018x}, {c:#018x}, {r:#018x}),");
    }
    println!("];\n");

    println!("// Re-bless by copying this block into GOLDEN_ENVELOPE:");
    println!("const GOLDEN_ENVELOPE: [(&str, (u64, u32, u32), (u64, u32, u32)); 4] = [");
    for (k, c, r) in &actual_env {
        println!(
            "    (\"{k}\", ({:#018x}, {}, {}), ({:#018x}, {}, {})),",
            c.0, c.1, c.2, r.0, r.1, r.2
        );
    }
    println!("];\n");

    assert!(
        GOLDEN
            .iter()
            .all(|(_, e, c, r)| *e != 0 && *c != 0 && *r != 0),
        "GOLDEN still holds placeholder zeros — run this test with --nocapture \
         and paste the printed block in. An unpinned gate is worse than none, \
         because it reads as passing."
    );

    assert!(
        GOLDEN_ENVELOPE
            .iter()
            .all(|(_, c, r)| *c != (0, 0, 0) && *r != (0, 0, 0)),
        "GOLDEN_ENVELOPE still holds placeholder zeros — run this test with \
         --nocapture and paste the printed block in. An unpinned gate is worse \
         than none, because it reads as passing."
    );

    if !mismatches.is_empty() || !env_mismatches.is_empty() {
        let verdict = match (decode_hash_moved, env_mismatches.is_empty()) {
            (true, true) => {
                "VERDICT: decode hashes moved but every envelope is IDENTICAL -> \
                 sample-level drift, level and timing intact."
            }
            (true, false) => {
                "VERDICT: decode hashes AND envelopes moved -> the audio level or \
                 its shape over time changed. Treat as an audible regression — \
                 unless every envelope line above is a ±1 shift in max or mean \
                 with the bucket hashes intact, which is one-LSB dither."
            }
            (false, false) => {
                "VERDICT: envelopes moved with the decode hashes unchanged -> \
                 unreachable from codec output alone, since the envelope is a \
                 function of the PCM the hashes cover; suspect the envelope helper."
            }
            (false, true) => {
                "VERDICT: only the encode column moved -> wire bytes, no decoded \
                 audio involved; the envelope columns are silent on this by design."
            }
        };
        panic!(
            "codec output changed from the pinned values:\n  {}\n  {}\n\n\
             {verdict}\n\n{ENVELOPE_HASH_TRIAGE}\n\n\
             If this is an intentional codec change, verify the new output is \
             correct FIRST, then re-bless from the printed blocks. If this fired \
             on a non-x86-64 target, the engine is not bit-portable — that is a \
             real defect in the crate's core claim, not a test to relax.",
            mismatches.join("\n  "),
            env_mismatches.join("\n  "),
        );
    }
}

/// Pin `Vocoder::encode` (whole-buffer) and `LiveEncoder` (streaming).
///
/// These are the paths a real caller uses. `codec_output_is_pinned` covers only
/// the per-frame primitive; without this test the reverse-engineered pitch and
/// voicing chains are not guarded by anything at all.
#[test]
fn shipped_encode_paths_are_pinned() {
    assert_every_rate_is_pinned();

    let mut actual = Vec::new();
    let mut mismatches = Vec::new();

    let pcm = test_pcm_with_silence(FRAMES);

    for rate in ALL_RATES {
        let key = rate_key(rate);
        let whole = fnv1a(&encode_whole_buffer(rate, &pcm));
        let live = fnv1a(&encode_live(rate, &pcm));
        actual.push((key, whole, live));

        let want = GOLDEN_ENCODE_PATHS
            .iter()
            .find(|(k, _, _)| *k == key)
            .map(|(_, w, l)| (*w, *l))
            .unwrap_or_else(|| panic!("no GOLDEN_ENCODE_PATHS row for rate {key}"));

        for (label, got, want) in [("whole_buffer", whole, want.0), ("live", live, want.1)] {
            if want != 0 && got != want {
                mismatches.push(format!(
                    "{key} {label}: expected {want:#018x}, got {got:#018x}"
                ));
            }
        }
    }

    println!("\n// Re-bless by copying this block into GOLDEN_ENCODE_PATHS:");
    println!("const GOLDEN_ENCODE_PATHS: [(&str, u64, u64); 4] = [");
    for (k, w, l) in &actual {
        println!("    (\"{k}\", {w:#018x}, {l:#018x}),");
    }
    println!("];\n");

    assert!(
        GOLDEN_ENCODE_PATHS
            .iter()
            .all(|(_, w, l)| *w != 0 && *l != 0),
        "GOLDEN_ENCODE_PATHS still holds placeholder zeros — an unpinned gate \
         reads as passing, which is worse than no gate."
    );

    assert!(
        mismatches.is_empty(),
        "shipped encode-path output changed from the pinned values:\n  {}\n\n\
         This gate covers the reference pitch chain, the RE'd voicing tracker, the \
         lag search, `BLAG` and the `AmbeStream` duplicates. It does NOT cover \
         the silence gate: disabling that clamp entirely leaves every hash \
         here green. If you did not intend to change any of the above, this is \
         a real regression — do not re-bless it. If you did, verify by ear \
         FIRST, then re-bless from the printed block.",
        mismatches.join("\n  ")
    );

    // Structural invariant, stated independently of the literals above so a
    // regression fails with a diagnosis instead of an opaque hash mismatch.
    //
    // `AmbeStream` is the single-pass streaming replica of the whole-buffer
    // AMBE+2 encoder. With Route A (the two-frame look-ahead) the two share the
    // identical pitch-aligned live `gap2` window, so on continuous speech they
    // are byte-for-byte IDENTICAL across all nine fields — pitch (b0), voicing
    // (b1) and amplitude/gain (b2..b8). This asserts that full equality. (b2..b8
    // parity is the Route A win; b0/b1 must hold as they did before it.) On a
    // silence-containing signal b1 can still differ — the whole-buffer `(-3..=3)`
    // lag search vs the streamer's bounded `b1_track`, `BLAG = 2` — so this runs
    // on the continuous signal only.
    let continuous = test_pcm(FRAMES);
    for rate in [Rate::AmbePlus2_3600x2450, Rate::AmbePlus2_2450x2450] {
        let wf = ambe_fields_of(rate, &encode_whole_buffer(rate, &continuous));
        let lf = ambe_fields_of(rate, &encode_live(rate, &continuous));
        assert_eq!(
            wf.len(),
            lf.len(),
            "{}: LiveEncoder (AmbeStream) and Vocoder::encode emit different \
             frame counts on continuous speech.",
            rate_key(rate)
        );
        for (i, (w, l)) in wf.iter().zip(&lf).enumerate() {
            assert_eq!(
                w, l,
                "{}: LiveEncoder (AmbeStream) diverges from Vocoder::encode at \
                 frame {i} on continuous speech — Route A makes the two identical \
                 across b0..b8. b2..b8 divergence = the pitch-aligned live gap2 \
                 window drifted (look-ahead depth / feed timing); b0/b1 divergence \
                 = the hdr30 VAD replica (vocoder.rs advance_hdr30) or the reference-b0 \
                 injection drifting.",
                rate_key(rate)
            );
        }
    }
}

/// The gate above is only meaningful if the inputs it hashes are themselves
/// identical everywhere, so pin those too. Both generators use only integer
/// arithmetic — no libm, no floats — which is what makes them portable; these
/// constants prove that property rather than assuming it.
#[test]
fn test_inputs_are_platform_stable() {
    const PCM_HASH: u64 = 0x199f5a4fb93a492b;
    const BIT_HASHES: [(&str, u64); 4] = [
        ("imbe_7200x4400", 0x9ffe31e4b8296b0a),
        ("imbe_4400x4400", 0x434a4fb290f09b01),
        ("ambe2_3600x2450", 0xfc463daa504c0223),
        ("ambe2_2450x2450", 0xb61d2dd6e48d6032),
    ];

    let bits: Vec<&str> = BIT_HASHES.iter().map(|(k, _)| *k).collect();
    let all: Vec<&str> = ALL_RATES.iter().map(|&r| rate_key(r)).collect();
    assert_keyed_by("BIT_HASHES", &bits, &all);

    let pcm = test_pcm(FRAMES);
    assert_eq!(pcm.len(), FRAMES * 160);
    let pcm_hash = hash_i16(&pcm);

    println!("\n// Re-bless test-input hashes:");
    println!("    const PCM_HASH: u64 = {pcm_hash:#018x};");
    println!("    const BIT_HASHES: [(&str, u64); 4] = [");
    for rate in ALL_RATES {
        let bits = test_bits(rate, FRAMES);
        assert_eq!(bits.len(), rate.fec_frame_bytes() * FRAMES);
        println!("        (\"{}\", {:#018x}),", rate_key(rate), fnv1a(&bits));
    }
    println!("    ];\n");

    if PCM_HASH != 0 {
        assert_eq!(pcm_hash, PCM_HASH, "PCM generator is not platform-stable");
        for rate in ALL_RATES {
            let want = BIT_HASHES
                .iter()
                .find(|(k, _)| *k == rate_key(rate))
                .map(|(_, h)| *h)
                .unwrap_or_else(|| panic!("no BIT_HASHES row for rate {}", rate_key(rate)));
            assert_eq!(
                fnv1a(&test_bits(rate, FRAMES)),
                want,
                "bit generator is not platform-stable for {}",
                rate_key(rate)
            );
        }
    }
}
