// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! Encoder analysis front-end (AMBE+2 / IMBE).
//!
//! Streaming i16 PCM -> per-frame windowed short-time spectrum used by the
//! downstream pitch / voicing / spectral-amplitude estimators.
//!
//! Pipeline per 20 ms frame (160 samples @ 8 kHz):
//!   1. Anchor the analysis window at `frame_start + FRAME/2 + win_offset()`.
//!      The short-time analysis is taken around the frame midpoint rather than
//!      the frame edge (TIA-102.BABA-A §0.2 / IMBE Annex), plus the reference's
//!      forward shift into the look-ahead -- see [`win_offset`].
//!   2. Apply the analysis window `wA(n)` (the firmware 250-tap reference window,
//!      see [`window_coeff`]).
//!   3. Zero-phase (centred) real DFT of size `FFT_SIZE` (256, matching the
//!      decoder's `dft_256_windowed` convention) -> complex bins.
//!
//! Alongside the spectrum each frame also carries two cheap time-domain
//! features computed on the *un-windowed* 160-sample frame:
//!   * `energy`  — Σ s(n)²  (used to scale the voicing threshold Θ),
//!   * `crest`   — peak/RMS  (phase-sensitive voicing feature; the reference V/UV
//!     decision is phase sensitive so a time-domain crest factor
//!     is more faithful than a purely spectral one).
//!
//! ## Conventions
//! * `FFT_SIZE = 256` and the zero-phase centring match `synth::dft_256_*`,
//!   so encoder and decoder share a bin convention.
//! * Internals are `f64`.

/// Samples per 20 ms frame at 8 kHz.
pub(crate) const FRAME: usize = 160;

/// Half-support of the analysis window (samples either side of centre).
/// The firmware analysis window is the 250-pt reference window, so its support is
/// 125 samples either side of the geometric centre (even length → centre index
/// 125; `n = i - WIN_HALF` runs −125..=124).
pub(crate) const WIN_HALF: usize = 125;

/// Full analysis-window length = the firmware reference window length (250 taps).
pub(crate) const WIN_LEN: usize = crate::fw_tables::REFERENCE_WIN250_N;

/// DFT size (matches the decoder's 256-point convention).
pub(crate) const FFT_SIZE: usize = 256;

use core::f64::consts::PI;

/// Analysis-window coefficient at window index `i` in `0..WIN_LEN`.
///
/// The firmware's bit-exact 250-pt reference window (`fw_tables::reference_win250`, the
/// interior of a 256-pt Hann), normalised to peak ~1.0. This is the window the
/// reference analyses with — pitch, amplitude, and voicing all share this front-end,
/// so the encoder must use it for ALL three (not a Hamming stand-in) to match
/// the reference's spectrum.
#[inline]
pub(crate) fn window_coeff(i: usize) -> f64 {
    debug_assert!(i < WIN_LEN);
    f64::from(crate::fw_tables::reference_win250()[i]) / 32768.0
}

/// Forward shift (samples) applied to the shared analysis window's centre,
/// on top of the geometric frame midpoint. Shared by all analysis-window call
/// sites (e.g. `Encoder::prefiltered_spectrum`) so they use one value.
///
/// The reference centres its analysis window ~72 samples past the frame
/// midpoint, into the look-ahead -- NOT at the midpoint. 72 is the peak of a
/// sweep against per-parameter bit-match with the reference bits; pulling it
/// back toward the midpoint costs the amplitude fields b6/b7/b8 roughly ten
/// points of bit-match each. It is fitted to reference bits, not tuned by ear,
/// and the reference is retired -- do not re-derive it.
#[inline]
pub(crate) fn win_offset() -> i64 {
    72
}

/// One frame's short-time analysis result.
#[derive(Clone, Debug)]
pub(crate) struct Spectrum {
    /// Absolute index of the frame this spectrum was computed for
    /// (`0` = first 160-sample frame pushed).
    pub frame_index: usize,
    /// Real part of each DFT bin, length [`FFT_SIZE`].
    pub re: Vec<f64>,
    /// Imag part of each DFT bin, length [`FFT_SIZE`].
    pub im: Vec<f64>,
    /// Σ s(n)² over the un-windowed 160-sample frame.
    pub energy: f64,
    /// Time-domain crest factor (peak |s| / RMS) of the frame; 0 for silence.
    pub crest: f64,
}

impl Spectrum {
    /// Magnitude `|X[k]|` of bin `k`.
    #[inline]
    pub fn mag(&self, k: usize) -> f64 {
        (self.re[k] * self.re[k] + self.im[k] * self.im[k]).sqrt()
    }
}

/// Streaming PCM -> per-frame spectrum analyser.
///
/// Feed arbitrary-length i16 chunks via [`push`](Analyzer::push); whenever a
/// frame's full window support (incl. the 30-sample look-ahead past the frame
/// end) has arrived it is analysed and cached. Completed spectra queue up in
/// `ready`; [`spectrum_for`](Analyzer::spectrum_for) fetches one by frame index
/// and [`discard_before`](Analyzer::discard_before) prunes consumed ones. Use
/// [`flush`] to force the trailing (zero-padded) frames at end-of-stream.
pub(crate) struct Analyzer {
    /// Sliding sample buffer (f64), holding a suffix of the input stream.
    buf: Vec<f64>,
    /// Absolute stream index of `buf[0]`.
    buf_start: usize,
    /// Total samples ever pushed (absolute write head).
    total_pushed: usize,
    /// Next frame index awaiting analysis.
    next_frame: usize,
    /// Completed-but-not-yet-consumed spectra, in ascending `frame_index`
    /// order. A single slot suffices for a one-frame look-ahead, but the
    /// live-`gap2` amplitude path needs the encoder to hold frame `f`'s
    /// spectrum until frame `f+2` has been analysed, so up to two (plus the
    /// just-completed one) can be pending. The encoder prunes consumed
    /// entries via [`Self::discard_before`].
    ready: std::collections::VecDeque<Spectrum>,
}

impl Analyzer {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            buf_start: 0,
            total_pushed: 0,
            next_frame: 0,
            ready: std::collections::VecDeque::new(),
        }
    }

    /// Absolute stream index of the analysis-window centre for frame `f`:
    /// the geometric frame midpoint plus [`win_offset`]'s forward shift.
    #[inline]
    fn frame_center(f: usize) -> usize {
        ((f * FRAME + FRAME / 2) as i64 + win_offset()).max(0) as usize
    }

    /// Last absolute sample index the window for frame `f` reaches.
    #[inline]
    fn frame_last_needed(f: usize) -> usize {
        Self::frame_center(f) + WIN_HALF
    }

    /// Read stream sample at absolute index `abs`, zero outside what we hold
    /// or before the stream start (left/right zero-padding).
    #[inline]
    fn sample_at(&self, abs: i64) -> f64 {
        if abs < 0 {
            return 0.0;
        }
        let abs = abs as usize;
        if abs < self.buf_start {
            return 0.0;
        }
        let rel = abs - self.buf_start;
        if rel < self.buf.len() {
            self.buf[rel]
        } else {
            0.0
        }
    }

    /// Append PCM and analyse every frame whose window support is now
    /// fully available.
    pub fn push(&mut self, pcm: &[i16]) {
        self.buf.extend(pcm.iter().map(|&s| s as f64));
        self.total_pushed += pcm.len();
        self.process_ready(false);
    }

    /// Flush trailing frames at end-of-stream, zero-padding any missing
    /// look-ahead samples so the final partial-context frames are emitted.
    pub fn flush(&mut self) {
        self.process_ready(true);
    }

    fn process_ready(&mut self, eos: bool) {
        loop {
            let f = self.next_frame;
            // The frame's samples themselves (frame_start..frame_start+FRAME)
            // must exist; with eos we additionally pad the look-ahead.
            let frame_start = f * FRAME;
            let frame_end = frame_start + FRAME; // exclusive
            let have_frame = self.total_pushed >= frame_end;
            let have_lookahead = self.total_pushed > Self::frame_last_needed(f);
            let ready = if eos { have_frame } else { have_lookahead };
            if !ready {
                break;
            }
            let spec = self.analyze_frame(f);
            self.ready.push_back(spec);
            self.next_frame += 1;
            self.trim();
        }
    }

    /// Drop buffered samples no later frame will ever need.
    fn trim(&mut self) {
        // The next frame to analyse needs samples from here on. Keep from the
        // EARLIER of the window's left edge and the frame's own start: with a
        // large forward window offset the window can sit entirely past the
        // frame midpoint, but the crest/energy features still read the whole
        // un-windowed frame [frame_start, frame_start+FRAME), which begins
        // before the window's left edge -- so those samples must be preserved.
        let center = Self::frame_center(self.next_frame) as i64;
        let frame_start = (self.next_frame * FRAME) as i64;
        let keep_from = (center - WIN_HALF as i64).min(frame_start).max(0) as usize;
        if keep_from > self.buf_start {
            let drop = keep_from - self.buf_start;
            let drop = drop.min(self.buf.len());
            self.buf.drain(0..drop);
            self.buf_start += drop;
        }
    }

    fn analyze_frame(&self, f: usize) -> Spectrum {
        let center = Self::frame_center(f) as i64;

        // 1+2. Window, centred on frame midpoint.
        let mut windowed = vec![0.0f64; WIN_LEN];
        for i in 0..WIN_LEN {
            let n = i as i64 - WIN_HALF as i64; // -WIN_HALF..=WIN_HALF
            let s = self.sample_at(center + n);
            windowed[i] = s * window_coeff(i);
        }

        // 3. Zero-phase centred real DFT.
        let (re, im) = centered_rdft(&windowed);

        // Time-domain features on the un-windowed 160-sample frame.
        let frame_start = (f * FRAME) as i64;
        let mut energy = 0.0f64;
        let mut peak = 0.0f64;
        for j in 0..FRAME {
            let s = self.sample_at(frame_start + j as i64);
            energy += s * s;
            let a = s.abs();
            if a > peak {
                peak = a;
            }
        }
        let rms = (energy / FRAME as f64).sqrt();
        let crest = if rms > 1e-12 { peak / rms } else { 0.0 };

        Spectrum {
            frame_index: f,
            re,
            im,
            energy,
            crest,
        }
    }

    /// The completed spectrum for frame `f`, if it is still pending (not yet
    /// discarded). Returns `None` once [`Self::discard_before`] has dropped it.
    pub fn spectrum_for(&self, f: usize) -> Option<&Spectrum> {
        self.ready.iter().find(|s| s.frame_index == f)
    }

    /// Drop every pending spectrum with `frame_index < f` — the encoder calls
    /// this after consuming frame `f-1` so the deque stays bounded to the
    /// frames still needed (`[next_emit, frames_completed)`).
    pub fn discard_before(&mut self, f: usize) {
        while let Some(front) = self.ready.front() {
            if front.frame_index < f {
                self.ready.pop_front();
            } else {
                break;
            }
        }
    }

    /// Index of the next frame awaiting analysis (== number completed).
    pub fn frames_completed(&self) -> usize {
        self.next_frame
    }
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-phase real DFT of a centred window.
///
/// Input `windowed[i]` corresponds to time offset `n = i - WIN_HALF`, so the
/// transform uses `n` directly in the exponent (phase referenced to the
/// window centre, `n = 0`). This is the same centring the decoder uses, so
/// bin phases are directly comparable between encode and decode.
///
/// Returns `(re, im)`, each length [`FFT_SIZE`].
pub(crate) fn centered_rdft(windowed: &[f64]) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(windowed.len(), WIN_LEN);
    let (ctab, stab) = rdft_twiddles();
    let mut re = vec![0.0f64; FFT_SIZE];
    let mut im = vec![0.0f64; FFT_SIZE];
    for k in 0..FFT_SIZE {
        let mut ar = 0.0f64;
        let mut ai = 0.0f64;
        let base = k * WIN_LEN;
        for i in 0..WIN_LEN {
            // Twiddles precomputed from the IDENTICAL in-loop expression
            // (see `rdft_twiddles`); accumulation stays strictly
            // left-to-right so the result is bit-identical to the naive loop.
            let s = windowed[i];
            ar += s * ctab[base + i];
            ai += s * stab[base + i];
        }
        re[k] = ar;
        im[k] = ai;
    }
    (re, im)
}

/// Precomputed cos/sin twiddle tables for [`centered_rdft`].
///
/// `ctab[k*WIN_LEN + i] = (scale * k * n).cos()` and likewise `stab` with
/// `.sin()`, using the exact same `scale = -2*PI/FFT_SIZE` and
/// `n = i - WIN_HALF` as the original in-loop expression. Built once and
/// cached; the values are bit-identical to recomputing `ang.cos()`/`ang.sin()`
/// per iteration, so the DFT output — and every downstream bit — is unchanged.
fn rdft_twiddles() -> (&'static [f64], &'static [f64]) {
    use std::sync::OnceLock;
    type TwiddleTabs = (Box<[f64]>, Box<[f64]>);
    static TABS: OnceLock<TwiddleTabs> = OnceLock::new();
    let (c, s) = TABS.get_or_init(|| {
        let mut c = vec![0.0f64; FFT_SIZE * WIN_LEN];
        let mut s = vec![0.0f64; FFT_SIZE * WIN_LEN];
        let scale = -2.0 * PI / FFT_SIZE as f64;
        for k in 0..FFT_SIZE {
            let base = k * WIN_LEN;
            for i in 0..WIN_LEN {
                let n = i as i64 - WIN_HALF as i64;
                let ang = scale * (k as f64) * (n as f64);
                c[base + i] = ang.cos();
                s[base + i] = ang.sin();
            }
        }
        (c.into_boxed_slice(), s.into_boxed_slice())
    });
    (c, s)
}
