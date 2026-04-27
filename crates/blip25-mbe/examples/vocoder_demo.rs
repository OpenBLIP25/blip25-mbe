//! End-to-end demo of the [`blip25_mbe::vocoder`] API.
//!
//! Run:
//!   `cargo run --release --example vocoder_demo`
//!
//! The example walks through every public surface of `Vocoder`:
//! one-shot encode/decode, slice streaming, live chunk-driven
//! streaming, the parameter-layer (extract_params / synthesize_params),
//! the builder, and stats / disposition inspection.

use blip25_mbe::vocoder::{LiveEncoder, Rate, Transcoder, Vocoder};

const FRAMES: usize = 50; // 1.0 second of audio at 8 kHz / 20 ms/frame.

fn main() {
    let pcm = synthetic_speech(FRAMES);

    println!("=== One-shot encode/decode ===");
    one_shot(&pcm);

    println!("\n=== Slice streaming (Iterator) ===");
    slice_streaming(&pcm);

    println!("\n=== Live chunk-driven streaming ===");
    live_streaming(&pcm);

    println!("\n=== Parameter-layer (extract → mutate → synthesize) ===");
    parameter_layer(&pcm);

    println!("\n=== Builder + opt-in knobs ===");
    builder_demo(&pcm);

    println!("\n=== Transcoder: P25 Phase 1 → Phase 2 ===");
    transcode_demo(&pcm);
}

fn one_shot(pcm: &[i16]) {
    let mut tx = Vocoder::new(Rate::Imbe7200x4400);
    let mut rx = Vocoder::new(Rate::Imbe7200x4400);
    let mut encoded: Vec<u8> = Vec::new();
    let mut decoded: Vec<i16> = Vec::new();

    for chunk in pcm.chunks_exact(tx.frame_samples()) {
        let bits = tx.encode_pcm(chunk).expect("encode");
        let pcm_out = rx.decode_bits(&bits).expect("decode");
        encoded.extend(&bits);
        decoded.extend(&pcm_out);
    }

    let stats = tx.last_stats();
    let last_disposition = rx.last_disposition();
    println!(
        "encoded {} bytes, decoded {} samples; last enc kind = {:?}, last dec disposition = {:?}",
        encoded.len(),
        decoded.len(),
        stats.analysis.as_ref().map(|a| a.output),
        last_disposition,
    );
}

fn slice_streaming(pcm: &[i16]) {
    let mut tx = Vocoder::new(Rate::AmbePlus2_3600x2450); // half-rate for variety
    let bits: Vec<Vec<u8>> = tx
        .encode_stream(pcm)
        .collect::<Result<Vec<_>, _>>()
        .expect("encode-stream");
    println!("encode_stream: produced {} half-rate frames (9 bytes each)", bits.len());

    // Flatten bits + decode through a fresh channel.
    let bytes: Vec<u8> = bits.into_iter().flatten().collect();
    let mut rx = Vocoder::new(Rate::AmbePlus2_3600x2450);
    let frames: Vec<Vec<i16>> = rx
        .decode_stream(&bytes)
        .collect::<Result<Vec<_>, _>>()
        .expect("decode-stream");
    println!("decode_stream: produced {} PCM frames", frames.len());
}

fn live_streaming(pcm: &[i16]) {
    let mut enc = LiveEncoder::new(Rate::Imbe7200x4400);
    // Feed in odd-sized chunks (250, 50, 333, …) so residue
    // accumulates across calls.
    let splits = [250usize, 50, 333, 256, 128];
    let mut pos = 0;
    let mut bits: Vec<u8> = Vec::new();
    let mut chunks_pushed = 0;
    for n in splits.iter().cycle() {
        if pos >= pcm.len() {
            break;
        }
        let end = (pos + n).min(pcm.len());
        for f in enc.push(&pcm[pos..end]) {
            bits.extend(f.expect("encode-frame"));
        }
        chunks_pushed += 1;
        pos = end;
    }
    // Flush the residue tail with zero-pad.
    if let Some(tail) = enc.flush().expect("flush") {
        bits.extend(tail);
    }
    println!(
        "LiveEncoder: pushed {} odd-sized chunks → {} FEC bytes (residue flushed)",
        chunks_pushed,
        bits.len(),
    );
}

fn parameter_layer(pcm: &[i16]) {
    // Extract params on one channel, optionally tweak, synthesize on another.
    // Useful for transcoding, analysis-only tooling, or custom synth chains.
    let mut analyzer = Vocoder::new(Rate::Imbe7200x4400);
    let mut synth = Vocoder::new(Rate::Imbe7200x4400);
    let mut got_voice = 0;
    for chunk in pcm.chunks_exact(analyzer.frame_samples()) {
        let mut params = analyzer.extract_params(chunk).expect("extract");
        // (Could mutate params here — lower amps, bias V/UV, etc.)
        let _ = synth.synthesize_params(&params);
        if params.amplitudes_slice().iter().any(|&a| a > 0.0) {
            got_voice += 1;
        }
        // Touch params so it isn't elided.
        let _ = std::hint::black_box(&mut params);
    }
    println!(
        "extract_params + synthesize_params: {got_voice}/{} frames yielded voice params",
        pcm.len() / 160
    );
}

fn builder_demo(pcm: &[i16]) {
    let mut tx = Vocoder::builder(Rate::AmbePlus2_3600x2450)
        .tone_detection(true)            // opt-in: emit Annex T tone frames on detected tones
        .repeat_reset_after(Some(3))     // beyond-spec, JMBE-style chip-interop
        .silence_dispatch(false)         // spec default
        .pitch_silence_override(false)
        .build();

    println!("Built: rate={:?} tone_detection={} repeat_reset_after={:?}",
        tx.rate(), tx.tone_detection(), tx.repeat_reset_after());

    let mut frame_buf: Vec<u8> = Vec::new();
    for chunk in pcm.chunks_exact(tx.frame_samples()) {
        frame_buf.extend(tx.encode_pcm(chunk).expect("encode"));
    }
    println!("Encoded {} bytes via builder-configured channel", frame_buf.len());
}

fn transcode_demo(pcm: &[i16]) {
    // Encode at Phase 1 (full-rate, 18-byte FEC frames) and transcode
    // each frame to Phase 2 (half-rate, 9-byte). The bridge runs in
    // the parameter domain — no PCM round-trip — so quality stays at
    // the parameter-extraction floor instead of the lossy
    // analysis-encode → synthesis → analysis-encode chain.
    let mut enc = Vocoder::new(Rate::Imbe7200x4400);
    let mut tx = Transcoder::new(Rate::Imbe7200x4400, Rate::AmbePlus2_3600x2450)
        .expect("supported direction");
    let mut p1_total = 0usize;
    let mut p2_total = 0usize;
    for chunk in pcm.chunks_exact(enc.frame_samples()) {
        let p1 = enc.encode_pcm(chunk).expect("encode phase1");
        let p2 = tx.transcode(&p1).expect("transcode phase1 → phase2");
        p1_total += p1.len();
        p2_total += p2.len();
    }
    println!(
        "Transcoder: {} P1 bytes → {} P2 bytes ({}× compression)",
        p1_total,
        p2_total,
        p1_total as f32 / p2_total as f32,
    );
}

// Synthesize a 1-second test signal: a 312.5 Hz tone for the first
// half, silence for the second half. Demonstrates voice + silence
// transitions.
fn synthetic_speech(n_frames: usize) -> Vec<i16> {
    let total = n_frames * 160;
    let mut pcm = vec![0i16; total];
    let half = total / 2;
    for (n, slot) in pcm[..half].iter_mut().enumerate() {
        let s = 4000.0
            * (2.0 * core::f64::consts::PI * 312.5 * n as f64 / 8000.0).sin();
        *slot = s.round() as i16;
    }
    // Second half stays at 0.
    pcm
}
