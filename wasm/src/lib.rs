//! In-browser, decode-only proof-of-concept for the `blip25-mbe` P25 MBE
//! vocoder.
//!
//! This crate is a thin `#[wasm_bindgen]` shim over the public
//! [`blip25_mbe::vocoder`] decode API. It builds against the crate with
//! `default-features = false, features = ["decode"]`, so the encoder and its
//! analysis pipeline are compiled out entirely — the wasm binary proves that
//! the decoder stands alone.
//!
//! The JS side constructs a [`WasmDecoder`] with a rate code, then feeds it one
//! wire frame at a time (or a whole concatenated bitstream) and gets back 8 kHz
//! signed-16-bit PCM, one 20 ms / 160-sample block per frame.

use blip25_mbe::vocoder::{Rate, Vocoder};
use wasm_bindgen::prelude::*;

/// Map a small integer rate code (stable across the JS boundary) to a [`Rate`].
///
/// | code | rate                     | wire bytes/frame |
/// |------|--------------------------|------------------|
/// | 0    | `Imbe7200x4400`          | 18 (FEC)         |
/// | 1    | `Imbe4400x4400`          | 11 (info-only)   |
/// | 2    | `AmbePlus2_3600x2450`    | 9  (FEC)         |
/// | 3    | `AmbePlus2_2450x2450`    | 7  (info-only)   |
fn rate_from_code(code: u8) -> Result<Rate, JsError> {
    Ok(match code {
        0 => Rate::Imbe7200x4400,
        1 => Rate::Imbe4400x4400,
        2 => Rate::AmbePlus2_3600x2450,
        3 => Rate::AmbePlus2_2450x2450,
        other => {
            return Err(JsError::new(&format!(
                "unknown rate code {other} (expected 0..=3)"
            )))
        }
    })
}

/// A P25 MBE decoder, one per logical audio stream, exposed to JavaScript.
///
/// Wraps a stateful [`Vocoder`]: the decoder carries inter-frame state
/// (previous-frame parameters for concealment, enhancement history), so decode
/// a stream through a single instance in order — do not reuse one instance
/// across unrelated streams.
#[wasm_bindgen]
pub struct WasmDecoder {
    inner: Vocoder,
    rate: Rate,
}

#[wasm_bindgen]
impl WasmDecoder {
    /// Construct a decoder for the given rate code (see [`rate_from_code`]).
    #[wasm_bindgen(constructor)]
    pub fn new(rate_code: u8) -> Result<WasmDecoder, JsError> {
        let rate = rate_from_code(rate_code)?;
        Ok(WasmDecoder {
            inner: Vocoder::new(rate),
            rate,
        })
    }

    /// Bytes of one wire frame at this decoder's rate. The JS harness slices the
    /// bitstream on this boundary.
    #[wasm_bindgen(getter, js_name = frameBytes)]
    pub fn frame_bytes(&self) -> usize {
        self.rate.fec_frame_bytes()
    }

    /// PCM samples produced per decoded frame (always 160 at 8 kHz / 20 ms).
    #[wasm_bindgen(getter, js_name = frameSamples)]
    pub fn frame_samples(&self) -> usize {
        self.rate.frame_samples()
    }

    /// Decode exactly one wire frame ([`Self::frame_bytes`] bytes) to signed
    /// 16-bit PCM (160 samples).
    #[wasm_bindgen(js_name = decodeFrame)]
    pub fn decode_frame(&mut self, bits: &[u8]) -> Result<Vec<i16>, JsError> {
        self.inner
            .decode_bits(bits)
            .map_err(|e| JsError::new(&format!("decode error: {e:?}")))
    }

    /// Decode exactly one wire frame to `f32` PCM normalized to `[-1.0, 1.0]`
    /// — ready to drop straight into a Web Audio `AudioBuffer` channel.
    #[wasm_bindgen(js_name = decodeFrameF32)]
    pub fn decode_frame_f32(&mut self, bits: &[u8]) -> Result<Vec<f32>, JsError> {
        let pcm = self.decode_frame(bits)?;
        Ok(pcm.iter().map(|&s| s as f32 / 32768.0).collect())
    }

    /// Decode a whole concatenated bitstream (any whole number of frames;
    /// trailing partial bytes are ignored) to one contiguous `f32` PCM buffer
    /// normalized to `[-1.0, 1.0]`. This is the one call the harness needs for
    /// gapless playback: decoder state is carried frame-to-frame internally.
    #[wasm_bindgen(js_name = decodeStreamF32)]
    pub fn decode_stream_f32(&mut self, bits: &[u8]) -> Result<Vec<f32>, JsError> {
        let mut out: Vec<f32> = Vec::new();
        for frame in self.inner.decode_stream(bits) {
            let pcm = frame.map_err(|e| JsError::new(&format!("decode error: {e:?}")))?;
            out.extend(pcm.iter().map(|&s| s as f32 / 32768.0));
        }
        Ok(out)
    }
}

/// The native sample rate of every P25 MBE frame (Hz). Exposed so the JS side
/// can build its `AudioBuffer` without hard-coding a magic number.
#[wasm_bindgen(js_name = sampleRateHz)]
pub fn sample_rate_hz() -> u32 {
    8000
}
