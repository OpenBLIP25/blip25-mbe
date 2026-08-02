//! blip25 blip25_codec conformance CLI.
//!
//! This binary has no subcommands of its own; the conformance tooling lives in
//! the examples of this crate:
//!
//!   * `reference_roundtrip_sanity` — decode / encode / full WAV->codec->WAV
//!     roundtrip vs the reference vocoder's ground truth, driven through the
//!     shipped [`blip25_mbe::vocoder`] API, with a phase-invariant per-frame
//!     energy-envelope metric.
//!   * `roundtrip_wav` — encode->decode a raw PCM file back to a WAV for an ear
//!     check of the shipped codec.

fn main() {
    eprintln!(
        "blip25 blip25_codec conformance — run the examples:\n  \
         cargo run -p blip25-conformance-roundtrip --example reference_roundtrip_sanity -- --help\n  \
         cargo run -p blip25-conformance-roundtrip --example roundtrip_wav -- <in.pcm> <out.wav>"
    );
}
