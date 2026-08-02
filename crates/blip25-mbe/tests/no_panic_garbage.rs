//! Deterministic garbage-input harness: the public API must NEVER panic on
//! hostile / malformed input — it degrades gracefully (erasure, concealment,
//! silence, or an `Err`). Every test drives an API family with seeded-LCG
//! random and adversarial inputs; completing without a panic IS the
//! assertion. Output lengths are additionally asserted where trivially
//! checkable.
//!
//! No env vars, no wall-clock, no reference: fully deterministic. Iteration
//! counts are sized so the whole file stays under ~20 s in the dev profile.

use blip25_mbe::vocoder::{LiveDecoder, LiveEncoder, Rate, Transcoder, Vocoder};
use blip25_mbe::{reference_soft_decision, fec, imbe7200, rate33, rate_conversion};

const ALL_RATES: [Rate; 4] = [
    Rate::Imbe7200x4400,
    Rate::Imbe4400x4400,
    Rate::AmbePlus2_3600x2450,
    Rate::AmbePlus2_2450x2450,
];

const FEC_RATES: [Rate; 2] = [Rate::Imbe7200x4400, Rate::AmbePlus2_3600x2450];

// ---------------------------------------------------------------------------
// Seeded LCG (PCG-style multiplier; state-only, no external deps).
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // xorshift the state for output so low bits aren't weak
        let x = self.0;
        (x ^ (x >> 31)).wrapping_mul(0x9E3779B97F4A7C15)
    }
    fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }
    fn next_u16(&mut self) -> u16 {
        (self.next_u64() >> 48) as u16
    }
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
    fn next_i16(&mut self) -> i16 {
        self.next_u16() as i16
    }
    /// Full-range i8 INCLUDING i8::MIN (-128) — the classic negation-overflow
    /// hazard for soft-decision code.
    fn next_i8(&mut self) -> i8 {
        self.next_u8() as i8
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_u8()).collect()
    }
    fn i8s(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| self.next_i8()).collect()
    }
    fn dibits<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        for d in out.iter_mut() {
            *d = self.next_u8() & 0x03;
        }
        out
    }
}

/// Pack 36 dibits into the 9-byte AMBE+2 r33 wire frame (MSB-first).
fn pack_dibits36(dibits: &[u8; 36]) -> [u8; 9] {
    let mut out = [0u8; 9];
    for (i, &d) in dibits.iter().enumerate() {
        let (hi, lo) = ((d >> 1) & 1, d & 1);
        out[(2 * i) / 8] |= hi << (7 - ((2 * i) % 8));
        out[(2 * i + 1) / 8] |= lo << (7 - ((2 * i + 1) % 8));
    }
    out
}

/// A random but field-width-valid AMBE+2 `b` vector, derived by wire-decoding
/// random FEC bytes (whatever survives Golay correction is by construction
/// in-range for every field).
fn random_valid_b(lcg: &mut Lcg) -> [u16; 9] {
    let bytes = lcg.bytes(9);
    rate33::fields_from_fec(&bytes)
}

// ===========================================================================
// 1. Vocoder::decode_bits — hard-decision garbage, every rate
// ===========================================================================

#[test]
fn decode_hard_garbage_all_rates() {
    for (ri, &rate) in ALL_RATES.iter().enumerate() {
        let mut lcg = Lcg::new(0xDEC0DE_0001 + ri as u64);
        let mut rx = Vocoder::new(rate);
        let n = rate.fec_frame_bytes();

        // Deterministic edge frames first, then a random stream through the
        // SAME stateful decoder (state carry-over is part of the surface).
        let mut frames: Vec<Vec<u8>> =
            vec![vec![0x00; n], vec![0xFF; n], vec![0xAA; n], vec![0x55; n]];
        for _ in 0..24 {
            frames.push(lcg.bytes(n));
        }
        for (fi, f) in frames.iter().enumerate() {
            let pcm = rx.decode_bits(f).unwrap_or_else(|e| {
                panic!("decode_bits errored on valid-length input (rate {rate:?}, frame {fi}): {e}")
            });
            assert_eq!(pcm.len(), 160, "rate {rate:?} frame {fi}");
        }

        // Wrong lengths must be an Err, not a panic.
        assert!(rx.decode_bits(&[]).is_err());
        assert!(rx.decode_bits(&vec![0u8; n + 1]).is_err());
        assert!(rx.decode_bits(&vec![0u8; n - 1]).is_err());
    }
}

// ===========================================================================
// 2. Vocoder::decode_bits — targeted b0 erasure range + Annex-T tone-ID space
// ===========================================================================

#[test]
fn decode_hard_erasure_range_and_tone_ids() {
    // b0 sweep 0..=127 (7-bit pitch field): covers the valid pitch range,
    // the erasure range [120,123], and the reserved tail — through both
    // AMBE+2 wire formats, one stateful decoder each.
    let mut lcg = Lcg::new(0xE7A5_0002);
    let mut rx33 = Vocoder::new(Rate::AmbePlus2_3600x2450);
    let mut rx34 = Vocoder::new(Rate::AmbePlus2_2450x2450);
    for b0 in 0u16..=127 {
        let mut b = random_valid_b(&mut lcg);
        b[0] = b0;
        let f33 = blip25_codec::enc::encode_frame::encode_r33(&b);
        let f34 = blip25_codec::enc::encode_frame::encode_r34(&b);
        let p33 = rx33.decode_bits(&f33).expect("r33 decode");
        let p34 = rx34.decode_bits(&f34).expect("r34 decode");
        assert_eq!(p33.len(), 160, "b0={b0}");
        assert_eq!(p34.len(), 160, "b0={b0}");
    }

    // Annex-T tone-ID space: every id 0..=255 (valid AND invalid IDs) with
    // amplitude cycling over the extremes, as byte-exact r33 tone frames.
    let amps = [0u8, 64, 127, 255];
    for id in 0u16..=255 {
        let amp = amps[(id as usize) % amps.len()];
        let info = rate33::dequantize::encode_tone_frame_info(id as u8, amp);
        let dibits = rate33::frame::encode_frame(&info);
        let frame = pack_dibits36(&dibits);
        let pcm = rx33.decode_bits(&frame).expect("tone-frame decode");
        assert_eq!(pcm.len(), 160, "tone id={id} amp={amp}");
    }
}

// ===========================================================================
// 3. Vocoder::decode_soft — LLR extremes and random, both FEC rates
// ===========================================================================

#[test]
fn decode_soft_garbage() {
    for (ri, &rate) in FEC_RATES.iter().enumerate() {
        let n = rate.soft_frame_bits().expect("FEC rate has soft bits");
        let mut lcg = Lcg::new(0x50F7_0003 + ri as u64);
        let mut rx = Vocoder::new(rate);

        // Extremes: saturated both ways, all-zero (pure erasure), i8::MIN
        // everywhere (negation-overflow bait), and MIN/MAX alternation.
        let mut patterns: Vec<Vec<i8>> = vec![
            vec![127i8; n],
            vec![-127i8; n],
            vec![0i8; n],
            vec![-128i8; n],
            (0..n)
                .map(|i| if i % 2 == 0 { -128 } else { 127 })
                .collect(),
        ];
        for _ in 0..16 {
            patterns.push(lcg.i8s(n)); // full range incl. -128
        }
        for (pi, p) in patterns.iter().enumerate() {
            let pcm = rx.decode_soft(p).unwrap_or_else(|e| {
                panic!(
                    "decode_soft errored on valid-length LLRs (rate {rate:?}, pattern {pi}): {e}"
                )
            });
            assert_eq!(pcm.len(), 160, "rate {rate:?} pattern {pi}");
        }

        // Wrong length -> Err, not panic.
        assert!(rx.decode_soft(&vec![0i8; n - 1]).is_err());
        assert!(rx.decode_soft(&vec![0i8; n + 1]).is_err());
        assert!(rx.decode_soft(&[]).is_err());
    }

    // No-FEC rates: explicit SoftUnsupported, not a panic.
    for rate in [Rate::Imbe4400x4400, Rate::AmbePlus2_2450x2450] {
        let mut rx = Vocoder::new(rate);
        assert!(rx.decode_soft(&[0i8; 72]).is_err(), "{rate:?}");
    }
}

// ===========================================================================
// 4. Streaming iterators — random bytes/LLRs, trailing partials
// ===========================================================================

#[test]
fn streaming_iterators_garbage() {
    for (ri, &rate) in ALL_RATES.iter().enumerate() {
        let mut lcg = Lcg::new(0x57ea_0004 + ri as u64);
        let n = rate.fec_frame_bytes();

        // decode_stream over random bytes with a trailing partial frame.
        let mut rx = Vocoder::new(rate);
        let bits = lcg.bytes(n * 5 + (n / 2).max(1));
        let frames: Result<Vec<Vec<i16>>, _> = rx.decode_stream(&bits).collect();
        let frames = frames.expect("decode_stream");
        assert_eq!(frames.len(), 5, "{rate:?}");
        assert!(frames.iter().all(|f| f.len() == 160));

        // decode_stream_soft over random LLRs (incl. -128) + trailing partial.
        let mut rx2 = Vocoder::new(rate);
        let want = rate.soft_frame_bits();
        let llrs = lcg.i8s(want.unwrap_or(72) * 3 + 11);
        let soft_frames: Result<Vec<Vec<i16>>, _> = rx2.decode_stream_soft(&llrs).collect();
        let soft_frames = soft_frames.expect("decode_stream_soft");
        match want {
            Some(_) => assert_eq!(soft_frames.len(), 3, "{rate:?}"),
            None => assert!(soft_frames.is_empty(), "{rate:?}"),
        }

        // encode_stream over hostile PCM (rails + noise) with trailing partial.
        let mut tx = Vocoder::new(rate);
        let mut pcm: Vec<i16> = Vec::new();
        for i in 0..(160 * 2 + 37) {
            pcm.push(match i % 4 {
                0 => i16::MIN,
                1 => i16::MAX,
                _ => lcg.next_i16(),
            });
        }
        let enc: Result<Vec<Vec<u8>>, _> = tx.encode_stream(&pcm).collect();
        let enc = enc.expect("encode_stream");
        assert_eq!(enc.len(), 2, "{rate:?}");
        assert!(enc.iter().all(|f| f.len() == n));
        let _ = tx.flush_encode();
    }
}

// ===========================================================================
// 5. LiveEncoder / LiveDecoder — odd-sized chunk pushes
// ===========================================================================

#[test]
fn live_streams_odd_chunks() {
    // LiveDecoder: random bytes pushed in odd-sized chunks, all rates.
    for (ri, &rate) in ALL_RATES.iter().enumerate() {
        let mut lcg = Lcg::new(0x11FE_0005 + ri as u64);
        let mut ld = LiveDecoder::new(rate);
        let chunks = [1usize, 3, 7, 2, 19, 31, 64, 5, 128, 11];
        for &c in &chunks {
            let data = lcg.bytes(c);
            for r in ld.push(&data) {
                let pcm = r.expect("live decode");
                assert_eq!(pcm.len(), 160);
            }
        }
        let _ = ld.pending_bytes();
        ld.discard_pending();
        ld.reset();
    }

    // LiveEncoder: hostile PCM pushed in odd-sized chunks. One AMBE rate
    // (exercises the single-pass AmbeStream) + one IMBE rate (per-frame
    // path). Kept short — the stream analysis is the slow part in dev.
    for (ri, rate) in [Rate::AmbePlus2_3600x2450, Rate::Imbe7200x4400]
        .into_iter()
        .enumerate()
    {
        let mut lcg = Lcg::new(0x11FE_1005 + ri as u64);
        let mut le = LiveEncoder::new(rate);
        let n = rate.fec_frame_bytes();
        // ~4 frames total, in deliberately awkward chunk sizes.
        let chunks = [1usize, 3, 159, 160, 161, 17, 139];
        for &c in &chunks {
            let pcm: Vec<i16> = (0..c)
                .map(|i| match i % 3 {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    _ => lcg.next_i16(),
                })
                .collect();
            for r in le.push(&pcm) {
                let f = r.expect("live encode");
                assert_eq!(f.len(), n);
            }
        }
        let _ = le.pending_samples();
        for f in le.flush().expect("live flush") {
            assert_eq!(f.len(), n);
        }
    }
}

// ===========================================================================
// 6. imbe7200 wire layer — frame / fec entry points
// ===========================================================================

#[test]
fn imbe7200_wire_garbage() {
    let mut lcg = Lcg::new(0x1BBE_0006);

    for _ in 0..64 {
        // decode_frame with proper dibits (0..=3)...
        let dibits: [u8; 72] = lcg.dibits();
        let fr = imbe7200::frame::decode_frame(&dibits);
        let _ = fr.error_total();
        // ...and with raw full-range bytes (hostile: the type doesn't
        // enforce the dibit domain).
        let mut raw = [0u8; 72];
        for b in raw.iter_mut() {
            *b = lcg.next_u8();
        }
        let _ = imbe7200::frame::decode_frame(&raw);

        // Soft decode with full-range LLRs incl. -128.
        let mut soft = [0i8; 144];
        for s in soft.iter_mut() {
            *s = lcg.next_i8();
        }
        let _ = imbe7200::frame::decode_frame_soft(&soft);

        // encode_frame: width-masked info (in-contract random)...
        let mut info = [0u16; 8];
        for (i, w) in imbe7200::frame::INFO_WIDTHS.iter().enumerate() {
            info[i] = lcg.next_u16() & ((1u16 << w) - 1);
        }
        let enc = imbe7200::frame::encode_frame(&info);
        assert_eq!(enc.len(), 72);
        // ...and full-range u16 info (hostile).
        let mut wild = [0u16; 8];
        for v in wild.iter_mut() {
            *v = lcg.next_u16();
        }
        let _ = imbe7200::frame::encode_frame(&wild);

        // fec module primitives with random codewords.
        let u0 = lcg.next_u16();
        let _ = imbe7200::fec::pn_sequence(u0);
        let masks = imbe7200::fec::modulation_masks(u0);
        let mut cw = [0u32; 8];
        for c in cw.iter_mut() {
            *c = lcg.next_u32();
        }
        let _ = imbe7200::fec::demodulate(cw, u0);
        let _ = imbe7200::fec::demodulate(masks, u0);
        let di = imbe7200::fec::deinterleave(&dibits);
        let _ = imbe7200::fec::interleave(&di);
        let _ = imbe7200::fec::interleave(&cw);
        let _ = imbe7200::fec::soft_deinterleave(&soft);
        let mut sv = [0i8; 23];
        for s in sv.iter_mut() {
            *s = lcg.next_i8();
        }
        imbe7200::fec::soft_demodulate_vector(&mut sv, lcg.next_u32());
    }

    // Extremes for soft-demodulate: all i8::MIN under an all-ones mask
    // (worst case for any negation).
    let mut sv = [i8::MIN; 23];
    imbe7200::fec::soft_demodulate_vector(&mut sv, u32::MAX);
    let soft_min = [i8::MIN; 144];
    let _ = imbe7200::frame::decode_frame_soft(&soft_min);
    let _ = imbe7200::fec::soft_deinterleave(&soft_min);
}

// ===========================================================================
// 7. rate33 wire layer — frame entry points
// ===========================================================================

#[test]
fn rate33_wire_garbage() {
    let mut lcg = Lcg::new(0x0A33_0007);

    for _ in 0..64 {
        let dibits: [u8; 36] = lcg.dibits();
        let fr = rate33::frame::decode_frame(&dibits);
        let _ = fr.error_total();
        // Raw bytes as "dibits" (hostile domain violation).
        let mut raw = [0u8; 36];
        for b in raw.iter_mut() {
            *b = lcg.next_u8();
        }
        let _ = rate33::frame::decode_frame(&raw);

        let mut soft = [0i8; 72];
        for s in soft.iter_mut() {
            *s = lcg.next_i8();
        }
        let _ = rate33::frame::decode_frame_soft(&soft);
        let _ = rate33::frame::soft_deinterleave(&soft);

        // encode_frame with width-masked info + full-range info.
        let mut info = [0u16; 4];
        for (i, w) in rate33::frame::INFO_WIDTHS.iter().enumerate() {
            info[i] = lcg.next_u16() & ((1u16 << w) - 1);
        }
        let _ = rate33::frame::encode_frame(&info);
        let packed = rate33::frame::pack_no_fec(&info);
        let _ = rate33::frame::unpack_no_fec(&packed);
        let mut wild = [0u16; 4];
        for v in wild.iter_mut() {
            *v = lcg.next_u16();
        }
        let _ = rate33::frame::encode_frame(&wild);
        let _ = rate33::frame::pack_no_fec(&wild);

        // Valid-length random bytes through every fields_from_* view.
        let b7 = lcg.bytes(7);
        let _ = rate33::frame::unpack_no_fec(&b7);
        let _ = rate33::fields_from_no_fec(&b7);
        let _ = rate33::fields_from_natural(&b7);
        let b9 = lcg.bytes(9);
        let _ = rate33::fields_from_fec(&b9);

        // PN / modulation / interleave primitives.
        let u0 = lcg.next_u16();
        let _ = rate33::frame::pn_sequence(u0);
        let _ = rate33::frame::modulation_masks(u0);
        let mut cw = [0u32; 4];
        for c in cw.iter_mut() {
            *c = lcg.next_u32();
        }
        let _ = rate33::frame::demodulate(cw, u0);
        let _ = rate33::frame::decode_code_vectors(cw);
        let di = rate33::frame::deinterleave(&dibits);
        let _ = rate33::frame::interleave(&di);
        let mut sv = [0i8; 24];
        for s in sv.iter_mut() {
            *s = lcg.next_i8();
        }
        rate33::frame::soft_demodulate_vector(&mut sv, lcg.next_u32());
    }

    let soft_min = [i8::MIN; 72];
    let _ = rate33::frame::decode_frame_soft(&soft_min);
    let mut sv = [i8::MIN; 24];
    rate33::frame::soft_demodulate_vector(&mut sv, u32::MAX);
}

// ===========================================================================
// 8. Dequantize entry points — width-masked random u-vectors, full sweeps
// ===========================================================================

#[test]
fn dequantize_garbage() {
    // -- rate33 (AMBE+2 half-rate) ------------------------------------------
    let mut lcg = Lcg::new(0xDE0A_0008);
    let mut st = rate33::dequantize::DecoderState::new();
    for _ in 0..256 {
        let mut u = [0u16; 4];
        for (i, w) in rate33::frame::INFO_WIDTHS.iter().enumerate() {
            u[i] = lcg.next_u16() & ((1u16 << w) - 1);
        }
        let _ = rate33::dequantize::classify_ambe_plus2_frame(&u);
        let _ = rate33::dequantize::parse_tone_frame(&u);
        let _ = rate33::dequantize::decode_to_params(&u, &mut st);
        let _ = rate33::dequantize::dequantize(&u, &mut st);
    }
    // Chained: random wire bytes -> frame decode -> dequantize (the
    // production-reachable path).
    for _ in 0..128 {
        let dibits: [u8; 36] = lcg.dibits();
        let fr = rate33::frame::decode_frame(&dibits);
        let _ = rate33::dequantize::decode_to_params(&fr.info, &mut st);
    }
    // Full pitch-index / tone-space sweeps.
    for b0 in 0u16..=255 {
        let _ = rate33::dequantize::decode_pitch(b0 as u8);
        let _ = rate33::dequantize::l56_gate_from_b1(b0 as u8);
    }
    for id in 0u16..=255 {
        for amp in [0u8, 1, 64, 127, 128, 255] {
            let _ = rate33::dequantize::tone_to_mbe_params(id as u8, amp);
            let _ = rate33::dequantize::encode_tone_frame_info(id as u8, amp);
        }
    }

    // -- imbe7200 (full-rate) ------------------------------------------------
    let mut lcg = Lcg::new(0xDE01_0008);
    let mut st = imbe7200::dequantize::FullrateDecoderState::new();
    for _ in 0..256 {
        let mut u = [0u16; 8];
        for (i, w) in imbe7200::frame::INFO_WIDTHS.iter().enumerate() {
            u[i] = lcg.next_u16() & ((1u16 << w) - 1);
        }
        let _ = imbe7200::dequantize::dequantize(&u, &mut st);
    }
    for _ in 0..128 {
        let dibits: [u8; 72] = lcg.dibits();
        let fr = imbe7200::frame::decode_frame(&dibits);
        let _ = imbe7200::dequantize::decode_to_params(&fr, &mut st);
    }
    for b0 in 0u16..=255 {
        let _ = imbe7200::dequantize::pitch_decode(b0 as u8);
        let _ = imbe7200::dequantize::decode_gain(b0 as u8);
    }
}

// ===========================================================================
// 9. FEC codeword garbage — hard + soft Golay/Hamming
// ===========================================================================

#[test]
fn fec_codeword_garbage() {
    let mut lcg = Lcg::new(0xFEC0_0009);

    for _ in 0..2000 {
        let w = lcg.next_u32();
        let _ = fec::golay_23_12_decode(w & 0x7F_FFFF);
        let _ = fec::golay_23_12_decode(w); // hostile: bits above 23
        let _ = fec::golay_24_12_decode(w & 0xFF_FFFF);
        let _ = fec::golay_24_12_decode(w); // hostile: bits above 24
        let h = lcg.next_u16();
        let _ = fec::hamming_15_11_decode(h & 0x7FFF);
        let _ = fec::hamming_15_11_decode(h); // hostile: bit 15 set
                                              // Encoders with full-range (hostile: info wider than 12/11 bits).
        let _ = fec::golay_23_12_encode(h);
        let _ = fec::golay_24_12_encode(h);
        let _ = fec::hamming_15_11_encode(h);
    }

    // Soft decoders: random incl. -128, plus saturated extremes.
    for _ in 0..200 {
        let mut s23 = [0i8; 23];
        for v in s23.iter_mut() {
            *v = lcg.next_i8();
        }
        let _ = fec::golay_23_12_decode_soft(&s23);
        let mut s24 = [0i8; 24];
        for v in s24.iter_mut() {
            *v = lcg.next_i8();
        }
        let _ = fec::golay_24_12_decode_soft(&s24);
        let mut s15 = [0i8; 15];
        for v in s15.iter_mut() {
            *v = lcg.next_i8();
        }
        let _ = fec::hamming_15_11_decode_soft(&s15);
    }
    for fill in [i8::MIN, -127, 0, 127] {
        let _ = fec::golay_23_12_decode_soft(&[fill; 23]);
        let _ = fec::golay_24_12_decode_soft(&[fill; 24]);
        let _ = fec::hamming_15_11_decode_soft(&[fill; 15]);
    }

    // Soft-decision nibble/LLR helpers over the full domains.
    for v in i8::MIN..=i8::MAX {
        let _ = reference_soft_decision::llr_to_sd_nibble(v);
    }
    for n in 0u8..=255 {
        let _ = reference_soft_decision::sd_nibble_to_llr(n);
    }
    let llrs = lcg.i8s(144);
    let packed = reference_soft_decision::pack_nibble_stream(&llrs);
    let _ = reference_soft_decision::unpack_nibble_stream(&packed);
    let _ = reference_soft_decision::unpack_nibble_stream(&lcg.bytes(101));
}

// ===========================================================================
// 10. Rate conversion — Transcoder over every supported pair
// ===========================================================================

#[test]
fn transcoder_garbage() {
    let pairs = [
        (Rate::Imbe7200x4400, Rate::AmbePlus2_3600x2450),
        (Rate::AmbePlus2_3600x2450, Rate::Imbe7200x4400),
        (Rate::Imbe7200x4400, Rate::Imbe4400x4400),
        (Rate::Imbe4400x4400, Rate::Imbe7200x4400),
        (Rate::AmbePlus2_3600x2450, Rate::AmbePlus2_2450x2450),
        (Rate::AmbePlus2_2450x2450, Rate::AmbePlus2_3600x2450),
    ];
    for (pi, &(from, to)) in pairs.iter().enumerate() {
        let mut lcg = Lcg::new(0x7C0D_000A + pi as u64);
        let mut tc = Transcoder::new(from, to).expect("supported pair");
        let in_n = tc.direction().input_frame_bytes();
        let out_n = tc.direction().output_frame_bytes();

        let mut frames: Vec<Vec<u8>> = vec![vec![0x00; in_n], vec![0xFF; in_n]];
        for _ in 0..16 {
            frames.push(lcg.bytes(in_n));
        }
        for (fi, f) in frames.iter().enumerate() {
            // Err (e.g. Quantize) is acceptable; a panic is not.
            if let Ok(out) = tc.transcode(f) {
                assert_eq!(out.len(), out_n, "{from:?}->{to:?} frame {fi}");
            }
        }
        // Wrong length -> Err.
        assert!(tc.transcode(&vec![0u8; in_n + 1]).is_err());
        tc.reset();
    }

    // Direct converter drive with random dibits (both proper 0..=3 dibits
    // and hostile raw bytes).
    let mut lcg = Lcg::new(0x7C0D_100A);
    let mut f2h = rate_conversion::FullToHalfConverter::new();
    let mut h2f = rate_conversion::HalfToFullConverter::new();
    for _ in 0..12 {
        let d72: [u8; 72] = lcg.dibits();
        let _ = f2h.convert(&d72);
        let d36: [u8; 36] = lcg.dibits();
        let _ = h2f.convert(&d36);
    }
    let mut raw72 = [0u8; 72];
    let mut raw36 = [0u8; 36];
    for b in raw72.iter_mut() {
        *b = lcg.next_u8();
    }
    for b in raw36.iter_mut() {
        *b = lcg.next_u8();
    }
    let _ = f2h.convert(&raw72);
    let _ = h2f.convert(&raw36);

    // Repeaters (FEC-transparent repeat path).
    let mut fr = rate_conversion::FullRateRepeater::new();
    let mut hr = rate_conversion::HalfRateRepeater::new();
    for _ in 0..8 {
        let d72: [u8; 72] = lcg.dibits();
        let _ = fr.repeat(&d72);
        let d36: [u8; 36] = lcg.dibits();
        let _ = hr.repeat(&d36);
    }
}

// ===========================================================================
// 11. Encode side — hostile PCM
// ===========================================================================

/// Hostile PCM generators, all deterministic.
fn hostile_pcm(kind: usize, len: usize, lcg: &mut Lcg) -> Vec<i16> {
    match kind {
        // Full-rail alternation at the Nyquist-adjacent worst case.
        0 => (0..len)
            .map(|i| if i % 2 == 0 { i16::MIN } else { i16::MAX })
            .collect(),
        // Hard DC offset (also exercises i16::MIN abs/negation hazards).
        1 => vec![i16::MIN; len],
        // Impulse train on a DC pedestal.
        2 => (0..len)
            .map(|i| if i % 40 == 0 { i16::MAX } else { -20_000 })
            .collect(),
        // Seeded LCG noise.
        _ => (0..len).map(|_| lcg.next_i16()).collect(),
    }
}

#[test]
fn encode_hostile_pcm_per_frame() {
    // Per-frame primitive path: encode_pcm + flush_encode, all rates,
    // all hostile signals.
    for (ri, &rate) in ALL_RATES.iter().enumerate() {
        let n = rate.fec_frame_bytes();
        for kind in 0..4 {
            let mut lcg = Lcg::new(0xE4C0_000B + (ri * 4 + kind) as u64);
            let mut tx = Vocoder::new(rate);
            let pcm = hostile_pcm(kind, 160 * 3, &mut lcg);
            for f in pcm.chunks_exact(160) {
                let bytes = tx.encode_pcm(f).expect("encode_pcm");
                assert_eq!(bytes.len(), n, "{rate:?} kind {kind}");
            }
            for bytes in tx.flush_encode() {
                assert_eq!(bytes.len(), n, "{rate:?} kind {kind} flush");
            }
            // Wrong PCM length -> Err, not panic.
            assert!(tx.encode_pcm(&[0i16; 159]).is_err());
            assert!(tx.encode_pcm(&[]).is_err());
        }
    }
}

#[test]
fn encode_hostile_pcm_whole_buffer() {
    // Whole-buffer `encode` (reference-voicing path for AMBE+2, reference-pitch
    // path for IMBE) — the heavy path, so short buffers and a reduced
    // signal matrix keep dev-profile runtime bounded.
    let cases = [
        (Rate::AmbePlus2_3600x2450, 0usize),
        (Rate::AmbePlus2_3600x2450, 3),
        (Rate::AmbePlus2_2450x2450, 1),
        (Rate::AmbePlus2_2450x2450, 2),
        (Rate::Imbe7200x4400, 0),
        (Rate::Imbe7200x4400, 3),
        (Rate::Imbe4400x4400, 1),
        (Rate::Imbe4400x4400, 2),
    ];
    for (ci, &(rate, kind)) in cases.iter().enumerate() {
        let mut lcg = Lcg::new(0xE4C0_100B + ci as u64);
        let tx = Vocoder::new(rate);
        let pcm = hostile_pcm(kind, 160 * 3 + 40, &mut lcg); // trailing partial too
        let frames = tx.encode(&pcm);
        let n = rate.fec_frame_bytes();
        assert!(
            frames.iter().all(|f| f.len() == n),
            "{rate:?} kind {kind}: frame length"
        );
    }

    // Tone detection ON over hostile PCM (the detector must not panic on
    // rails/noise), one short buffer.
    let mut lcg = Lcg::new(0xE4C0_200B);
    let mut tx = Vocoder::new(Rate::AmbePlus2_3600x2450);
    tx.set_tone_detection(true);
    let pcm = hostile_pcm(0, 160 * 2, &mut lcg);
    let _ = tx.encode(&pcm);

    // Empty and sub-frame inputs.
    for rate in ALL_RATES {
        let tx = Vocoder::new(rate);
        assert!(tx.encode(&[]).is_empty(), "{rate:?} empty");
        let _ = tx.encode(&[i16::MIN; 40]);
    }
}
