//! reference vocoder acceptance sweep — the EXHAUSTIVE proof behind the committed
//! regression gate (`tests/codec_acceptance.rs`).
//!
//! For EVERY reference `tv-rc` vector and ALL 4 rates, this asserts that
//! blip25-mbe's DEFAULT public path is byte / sample-identical to calling the
//! vendored `blip25_codec` reference codec directly:
//!
//!   * DECODE  (full corpus, every rate) — `Vocoder::new(rate).decode_bits`
//!     (fully default: no `set_enhancement` call) vs
//!     `blip25_codec::Decoder::decode_pcm_fixed` / `ImbeDecoder::decode_pcm`.
//!   * ENCODE  (full corpus, every rate) — `Vocoder::new(rate).encode_pcm` +
//!     `flush_encode()` (look-ahead aligned) vs
//!     `blip25_codec::enc::Encoder::encode_frame_* + flush_*`.
//!   * WRAPPERS (representative subset) — `decode_stream` / `encode_stream` /
//!     `LiveEncoder`+flush / `LiveDecoder` produce the same total bytes /
//!     samples as the direct path, and `LiveEncoder` loses ZERO frames.
//!     (Wrappers delegate to the same `decode_bits`/`encode_pcm` the core
//!     sweep already proves on every frame, so a representative set is
//!     sufficient here; the committed `#[test]` gate exercises them too.)
//!
//! The reference is `blip25_codec` direct — NOT the reference `.pcm` files. The
//! vendored codec deliberately does not bit-reproduce the reference decode
//! (the synthesis phase back-end caveat documented in
//! `crates/blip25-codec/src/lib.rs`); this gate proves the blip25 façade
//! reproduces the *vendored* codec exactly, so a default-config drift (e.g. a
//! non-None enhancement default) is caught.
//!
//! Usage: `cargo run -p blip25-mbe --release --example codec_acceptance_sweep -- [tv-rc dir]`
//! Exit code 0 iff every rate is 0 max-abs-diff and 0 byte / len mismatch.

use blip25_mbe::vocoder::{
    imbe_fec_to_info_bytes, LiveDecoder, LiveEncoder, Rate, Transcoder, Vocoder,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Default corpus location: the gitignored `reference-material/Vectors/tv-rc` symlink at the
/// workspace root. Resolved from `CARGO_MANIFEST_DIR` so it does not depend on
/// the working directory. Override by passing a path as the first argument.
const DEFAULT_TVRC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../reference-material/Vectors/tv-rc");

fn read_bytes(p: &Path) -> Vec<u8> {
    fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}
fn read_pcm(p: &Path) -> Vec<i16> {
    read_bytes(p)
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}
fn max_abs_diff(a: &[i16], b: &[i16]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).abs())
        .max()
        .unwrap_or(0)
}

const ALL_RATES: [Rate; 4] = [
    Rate::AmbePlus2_3600x2450,
    Rate::AmbePlus2_2450x2450,
    Rate::Imbe7200x4400,
    Rate::Imbe4400x4400,
];

fn rate_idx(r: Rate) -> usize {
    match r {
        Rate::AmbePlus2_3600x2450 => 0,
        Rate::AmbePlus2_2450x2450 => 1,
        Rate::Imbe7200x4400 => 2,
        Rate::Imbe4400x4400 => 3,
        _ => unreachable!(),
    }
}

fn label(r: Rate) -> &'static str {
    match r {
        Rate::AmbePlus2_3600x2450 => "AMBE+2 r33 (9B)  ",
        Rate::AmbePlus2_2450x2450 => "AMBE+2 r34 (7B)  ",
        Rate::Imbe7200x4400 => "IMBE  full (18B) ",
        Rate::Imbe4400x4400 => "IMBE  info (11B) ",
        _ => "UNKNOWN          ",
    }
}

/// Subdir holding the `.bit` vectors for a rate.
fn bit_subdir(r: Rate) -> &'static str {
    match r {
        Rate::AmbePlus2_3600x2450 => "r33",
        Rate::AmbePlus2_2450x2450 => "r34",
        Rate::Imbe7200x4400 => "p25",
        Rate::Imbe4400x4400 => "p25_nofec",
        _ => unreachable!(),
    }
}

// ---- blip25_codec-direct references (independent of the blip25 codec routing) ----

/// blip25_codec-direct decode of a whole bit stream for `rate`. Fresh decoder state,
/// frames fed in order — the same contract a fresh `Vocoder::new(rate)` gives.
fn codec_direct_decode(rate: Rate, bits: &[u8]) -> Vec<i16> {
    let n = rate.fec_frame_bytes();
    let frames = bits.len() / n;
    let mut w_ambe = blip25_codec::Decoder::new();
    let mut w_imbe = blip25_codec::ImbeDecoder::new();
    // info -> full FEC re-add uses the PUBLIC Transcoder wire transform.
    let mut add_fec = Transcoder::new(Rate::Imbe4400x4400, Rate::Imbe7200x4400).unwrap();
    let mut out: Vec<i16> = Vec::with_capacity(frames * 160);
    for f in 0..frames {
        let fr = &bits[f * n..(f + 1) * n];
        let pcm: Vec<i16> = match rate {
            Rate::AmbePlus2_3600x2450 => w_ambe.decode_pcm_fixed(fr).to_vec(),
            Rate::AmbePlus2_2450x2450 => {
                let mut r34 = [0u8; 7];
                r34.copy_from_slice(&fr[..7]);
                let b = blip25_codec::enc::encode_frame::decode_r34(&r34);
                let r33 = blip25_codec::enc::encode_frame::encode_r33(&b);
                w_ambe.decode_pcm_fixed(&r33).to_vec()
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

/// blip25_codec-direct encode of a whole PCM stream: per-frame emit + flush.
/// Returns the ordered list of REAL emitted frames (no look-ahead placeholder).
fn codec_direct_encode(rate: Rate, pcm: &[i16]) -> Vec<Vec<u8>> {
    let frames = pcm.len() / 160;
    let mut e = blip25_codec::enc::Encoder::new();
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(frames);
    for f in 0..frames {
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

// ---- blip25 DEFAULT-path drivers (no config touched) ----

/// blip25 DEFAULT decode of a whole bit stream via `decode_bits`.
fn blip25_default_decode(rate: Rate, bits: &[u8]) -> Vec<i16> {
    let n = rate.fec_frame_bytes();
    let mut v = Vocoder::new(rate); // DEFAULT: enhancement None, reference codec on
    let mut out: Vec<i16> = Vec::new();
    for f in 0..(bits.len() / n) {
        out.extend(v.decode_bits(&bits[f * n..(f + 1) * n]).unwrap());
    }
    out
}

/// blip25 DEFAULT encode of a whole PCM stream via `encode_pcm` + `flush_encode`,
/// aligned past the frame-0 look-ahead placeholder. Returns
/// `(real_frames_in_order, placeholder_is_zero)`.
fn blip25_default_encode(rate: Rate, pcm: &[i16]) -> (Vec<Vec<u8>>, bool) {
    let mut v = Vocoder::new(rate); // DEFAULT config
    let frames = pcm.len() / 160;
    let mut ours: Vec<Vec<u8>> = Vec::with_capacity(frames);
    for f in 0..frames {
        ours.push(v.encode_pcm(&pcm[f * 160..(f + 1) * 160]).unwrap());
    }
    let flush = v.flush_encode();
    let placeholder_zero = ours
        .first()
        .map(|p| p.iter().all(|&b| b == 0))
        .unwrap_or(true);
    let mut stream: Vec<Vec<u8>> = ours.into_iter().skip(1).collect();
    stream.extend(flush);
    (stream, placeholder_zero)
}

#[derive(Default, Clone)]
struct RateStat {
    dec_vectors: usize,
    dec_frames: usize,
    dec_max_abs: i32,
    dec_mismatch_vectors: usize,
    enc_vectors: usize,
    enc_byte_mismatch_vectors: usize,
    enc_len_mismatch_vectors: usize,
    placeholder_bad_vectors: usize,
    wrap_mismatch: usize,
    live_frames_lost: usize,
}

impl RateStat {
    fn merge(&mut self, o: &RateStat) {
        self.dec_vectors += o.dec_vectors;
        self.dec_frames += o.dec_frames;
        self.dec_max_abs = self.dec_max_abs.max(o.dec_max_abs);
        self.dec_mismatch_vectors += o.dec_mismatch_vectors;
        self.enc_vectors += o.enc_vectors;
        self.enc_byte_mismatch_vectors += o.enc_byte_mismatch_vectors;
        self.enc_len_mismatch_vectors += o.enc_len_mismatch_vectors;
        self.placeholder_bad_vectors += o.placeholder_bad_vectors;
        self.wrap_mismatch += o.wrap_mismatch;
        self.live_frames_lost += o.live_frames_lost;
    }
    fn fail(&self) -> bool {
        self.dec_max_abs != 0
            || self.dec_mismatch_vectors != 0
            || self.enc_byte_mismatch_vectors != 0
            || self.enc_len_mismatch_vectors != 0
            || self.placeholder_bad_vectors != 0
            || self.wrap_mismatch != 0
            || self.live_frames_lost != 0
    }
}

enum Job {
    Decode { rate: Rate, path: PathBuf },
    Encode { rate: Rate, path: PathBuf },
}

fn list_ext(dir: &Path, ext: &str) -> Vec<String> {
    let mut v: Vec<String> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| n.ends_with(ext))
            .collect(),
        Err(_) => Vec::new(),
    };
    v.sort();
    v
}

fn process(job: &Job) -> (usize, RateStat) {
    match job {
        Job::Decode { rate, path } => {
            let rate = *rate;
            let n = rate.fec_frame_bytes();
            let bits = read_bytes(path);
            let mut st = RateStat::default();
            if bits.len() < n {
                return (rate_idx(rate), st);
            }
            st.dec_vectors = 1;
            st.dec_frames = bits.len() / n;
            let ours = blip25_default_decode(rate, &bits);
            let refr = codec_direct_decode(rate, &bits);
            let m = ours.len().min(refr.len());
            st.dec_max_abs = max_abs_diff(&ours[..m], &refr[..m]);
            if st.dec_max_abs != 0 || ours.len() != refr.len() {
                st.dec_mismatch_vectors = 1;
            }
            (rate_idx(rate), st)
        }
        Job::Encode { rate, path } => {
            let rate = *rate;
            let pcm = read_pcm(path);
            let mut st = RateStat::default();
            if pcm.len() < 160 {
                return (rate_idx(rate), st);
            }
            st.enc_vectors = 1;
            let (ours, placeholder_zero) = blip25_default_encode(rate, &pcm);
            if !placeholder_zero {
                st.placeholder_bad_vectors = 1;
            }
            let refr = codec_direct_encode(rate, &pcm);
            if ours.len() != refr.len() {
                st.enc_len_mismatch_vectors = 1;
            }
            let cmp = ours.len().min(refr.len());
            if (0..cmp).any(|i| ours[i] != refr[i]) {
                st.enc_byte_mismatch_vectors = 1;
            }
            (rate_idx(rate), st)
        }
    }
}

/// Representative-subset wrapper checks: encode_stream / decode_stream /
/// LiveEncoder+flush (zero frames lost) / LiveDecoder against the core path.
fn wrapper_checks(root: &Path, names: &[&str]) -> [RateStat; 4] {
    let mut acc: [RateStat; 4] = Default::default();
    for &rate in &ALL_RATES {
        let ri = rate_idx(rate);
        let bit_dir = root.join(bit_subdir(rate));
        let pcm_dir = root.join("r33");
        for name in names {
            // ---- decode wrappers ----
            let bitp = bit_dir.join(format!("{name}.bit"));
            if bitp.exists() {
                let bits = read_bytes(&bitp);
                let core = blip25_default_decode(rate, &bits);
                // decode_stream
                let mut vs = Vocoder::new(rate);
                let stream: Vec<i16> = vs
                    .decode_stream(&bits)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap()
                    .into_iter()
                    .flatten()
                    .collect();
                if stream != core {
                    acc[ri].wrap_mismatch += 1;
                }
                // LiveDecoder push(odd chunks)
                let n = rate.fec_frame_bytes();
                let mut live = LiveDecoder::new(rate);
                let mut lp: Vec<i16> = Vec::new();
                for c in bits.chunks(n * 3 + 5) {
                    for r in live.push(c) {
                        lp.extend(r.unwrap());
                    }
                }
                if lp != core {
                    acc[ri].wrap_mismatch += 1;
                }
            }
            // ---- encode wrappers ----
            let pcmp = pcm_dir.join(format!("{name}.pcm"));
            if pcmp.exists() {
                let pcm = read_pcm(&pcmp);
                let (core, _) = blip25_default_encode(rate, &pcm);
                let refr = codec_direct_encode(rate, &pcm);
                // encode_stream + flush
                let mut vs = Vocoder::new(rate);
                let mut stream: Vec<Vec<u8>> = vs
                    .encode_stream(&pcm)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                if !stream.is_empty() {
                    stream.remove(0);
                }
                stream.extend(vs.flush_encode());
                if stream != core {
                    acc[ri].wrap_mismatch += 1;
                }
                // LiveEncoder push(odd chunks) + flush: byte-identical AND zero lost
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
                if !emitted.is_empty() {
                    emitted.remove(0);
                }
                if emitted.len() != refr.len() {
                    acc[ri].live_frames_lost += 1;
                }
                if emitted != core {
                    acc[ri].wrap_mismatch += 1;
                }
            }
        }
    }
    acc
}

fn main() -> ExitCode {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_TVRC.to_string());
    let root = Path::new(&dir);
    if !root.is_dir() {
        eprintln!("corpus dir not found: {}", root.display());
        return ExitCode::FAILURE;
    }

    // Build the full job list: DECODE over every .bit in each rate's subdir;
    // ENCODE over every r33/*.pcm (105) that has a matching r33/*.bit pair, at
    // all 4 rates (the .pcm is fed identically to both encoders, so its
    // rate-specific semantics don't matter for encoder-vs-encoder parity).
    let mut jobs: Vec<Job> = Vec::new();
    for &rate in &ALL_RATES {
        let sub = root.join(bit_subdir(rate));
        for name in list_ext(&sub, ".bit") {
            jobs.push(Job::Decode {
                rate,
                path: sub.join(name),
            });
        }
    }
    let pcm_dir = root.join("r33");
    let paired_pcm: Vec<String> = list_ext(&pcm_dir, ".pcm")
        .into_iter()
        .filter(|nm| {
            pcm_dir
                .join(format!("{}.bit", &nm[..nm.len() - 4]))
                .exists()
        })
        .collect();
    for &rate in &ALL_RATES {
        for name in &paired_pcm {
            jobs.push(Job::Encode {
                rate,
                path: pcm_dir.join(name),
            });
        }
    }

    let total = jobs.len();
    println!(
        "== reference vocoder acceptance sweep: blip25 DEFAULT == blip25_codec direct ==\ncorpus: {}\njobs: {total} (decode: every .bit, all rates; encode: {} paired .pcm × 4 rates)\n",
        root.display(),
        paired_pcm.len()
    );

    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let jobs_ref = &jobs;

    let partials: Vec<[RateStat; 4]> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for _ in 0..nthreads {
            let cursor = &cursor;
            let done = &done;
            handles.push(s.spawn(move || {
                let mut local: [RateStat; 4] = Default::default();
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs_ref.len() {
                        break;
                    }
                    let (ri, st) = process(&jobs_ref[i]);
                    local[ri].merge(&st);
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 50 == 0 {
                        eprintln!("  ...{d}/{total} jobs");
                    }
                }
                local
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut stats: [RateStat; 4] = Default::default();
    for p in &partials {
        for r in 0..4 {
            stats[r].merge(&p[r]);
        }
    }

    // Representative wrapper checks (delegate to the already-proven core path).
    let rep = [
        "mark",      // speech, male
        "clean",     // clean speech
        "lvl",       // level sweep
        "alltone",   // tone content
        "dam_e5_hd", // channel errors
        "cp1",       // comfort-noise / DTX-ish
    ];
    let wrap = wrapper_checks(root, &rep);
    for r in 0..4 {
        stats[r].merge(&wrap[r]);
    }

    // -------- report --------
    println!(
        "{:<17} {:>5} {:>8} | DEC max_abs bad | ENC vecs byte len ph | WRAP bad live_lost",
        "rate", "vecs", "frames"
    );
    let mut any_fail = false;
    for &rate in &ALL_RATES {
        let st = &stats[rate_idx(rate)];
        let fail = st.fail();
        any_fail |= fail;
        println!(
            "{:<17} {:>5} {:>8} | {:>10} {:>3} | {:>4} {:>4} {:>3} {:>2} | {:>7} {:>9}  {}",
            label(rate),
            st.dec_vectors,
            st.dec_frames,
            st.dec_max_abs,
            st.dec_mismatch_vectors,
            st.enc_vectors,
            st.enc_byte_mismatch_vectors,
            st.enc_len_mismatch_vectors,
            st.placeholder_bad_vectors,
            st.wrap_mismatch,
            st.live_frames_lost,
            if fail { "FAIL" } else { "ok" }
        );
    }
    println!("\nwrapper subset: {rep:?}");
    println!();
    if any_fail {
        println!("RESULT: FAIL — blip25 default path diverges from blip25_codec direct.");
        ExitCode::FAILURE
    } else {
        println!("RESULT: PASS — blip25 default path is byte/sample-identical to blip25_codec direct on every vector, all 4 rates.");
        ExitCode::SUCCESS
    }
}
