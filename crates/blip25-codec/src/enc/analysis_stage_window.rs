//! Geometry of the per-frame "analysis staging buffer" -> the 40 per-frame
//! [`super::windowed_taper::windowed_taper`] calls.
//!
//! ## The buffer
//!
//! `windowed_taper`'s 32-sample input window (`srcptr`) comes from a 108-word
//! (216-byte) persistent region at a fixed offset from the encoder object
//! (`encoder_buf+0x15ce`). It is not the spectrum and not raw or scaled input
//! PCM -- all three were ruled out. The raw-audio pre-filter
//! (`enc::audio_prefilter`) writes at a fixed offset inside this same buffer,
//! `stage_base + 28` samples, which is exactly `108 - 80` for its 80-sample
//! write span.
//!
//! ## The windowing geometry
//!
//! Within a frame, `windowed_taper` is called exactly 40 times, in two
//! identical 20-call passes to two destination buffers 16 samples apart. (The
//! purpose of the second, content-identical pass is unresolved.) Each pass
//! slides a 32-sample window over the 108-word staging buffer in steps of 4
//! samples, from offset 0 to offset 76 (`(108-32)/4 + 1 = 20` positions) -- an
//! exhaustive, gap-free, 87.5%-overlap cover of the whole buffer, with the last
//! window's tail landing exactly on the buffer's last sample (`76 + 32 == 108`,
//! not a coincidence).
//!
//! ## Scope
//!
//! This module is pure slicing logic: which 32-sample slices of an EXTERNALLY
//! SUPPLIED 108-sample buffer feed each of the 20 per-pass `windowed_taper`
//! calls. The content comes from [`super::history_ring`], which is bit-exact --
//! see [`super::Encoder::current_stage`].
//!
//! [`super::Encoder::analyze_frame`]'s quantized bit output does **not** consume
//! this chain (same status as `windowed_taper`/`real_fft32`: tested and correct,
//! but `band_decompress`'s caller-side `step`/`outer` inputs are unresolved).

/// Length of the per-frame analysis staging buffer (`encoder_buf+0x15ce`).
pub(crate) const STAGE_LEN: usize = 108;
/// Width of each `windowed_taper` input window.
pub(crate) const WINDOW_LEN: usize = 32;
/// Sample stride between consecutive windows within one pass.
pub(crate) const WINDOW_STRIDE: usize = 4;
/// Number of windows in one pass: `(STAGE_LEN - WINDOW_LEN) / WINDOW_STRIDE + 1`,
/// which lands exactly on the buffer's last sample with no gap and no leftover
/// (`76 + 32 == 108`).
pub(crate) const NUM_WINDOWS: usize = (STAGE_LEN - WINDOW_LEN) / WINDOW_STRIDE + 1;

/// Slice the 108-sample staging buffer into the `NUM_WINDOWS` (20) overlapping
/// 32-sample windows one `windowed_taper` pass consumes, in call order.
///
/// Pure geometry: array slicing at the confirmed offsets/stride/count. It makes
/// no claim about how `stage` itself was produced -- see the module doc.
pub(crate) fn analysis_windows(stage: &[i16; STAGE_LEN]) -> [[i16; WINDOW_LEN]; NUM_WINDOWS] {
    let mut out = [[0i16; WINDOW_LEN]; NUM_WINDOWS];
    for (i, win) in out.iter_mut().enumerate() {
        let off = i * WINDOW_STRIDE;
        win.copy_from_slice(&stage[off..off + WINDOW_LEN]);
    }
    out
}
