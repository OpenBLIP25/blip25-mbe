//! §2.1 decode probe: log-prediction recurrence convergence.
//!
//! The QUALITY_FINDINGS §2.1 suspect #1 is the half-rate log-magnitude
//! prediction recurrence (`apply_log_prediction`): on a steady signal the
//! decoded envelope should converge to a stable shape. If our recurrence
//! drifts on a *constant* input info vector, the whole envelope walks
//! frame-over-frame even on identical bits — a decode-side timbre bug.
//!
//! This probe feeds one fixed info vector (a steady voiced frame, taken
//! from a chip-encoded `.ambe9`) repeatedly through `dequantize` with a
//! single advancing `DecoderState`, and prints the per-frame envelope
//! energy + the frame-to-frame envelope delta. A healthy recurrence
//! settles: the delta -> 0 within a few frames.
//!
//! Usage:
//!   cargo run -p blip25-conformance-speech-quality \
//!     --example halfrate_envelope_converge -- 020e 0c00 01dd 0e62 [N]

use blip25_mbe::ambe_plus2_wire::dequantize::{dequantize, DecoderState};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 4 {
        eprintln!("usage: halfrate_envelope_converge u0 u1 u2 u3 [N]");
        std::process::exit(2);
    }
    let u: [u16; 4] = [
        u16::from_str_radix(&a[0], 16).unwrap(),
        u16::from_str_radix(&a[1], 16).unwrap(),
        u16::from_str_radix(&a[2], 16).unwrap(),
        u16::from_str_radix(&a[3], 16).unwrap(),
    ];
    let n: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(30);

    let mut state = DecoderState::new();
    let mut prev: Option<Vec<f32>> = None;
    println!("frame  L   env_rms      d_env(dB)   note");
    for f in 0..n {
        let params = match dequantize(&u, &mut state) {
            Ok(p) => p,
            Err(e) => {
                println!("{f:3}   dequantize error: {e:?}");
                return;
            }
        };
        let amps = params.amplitudes_slice().to_vec();
        let env_rms = (amps.iter().map(|&m| (m as f64) * (m as f64)).sum::<f64>()
            / amps.len() as f64)
            .sqrt();
        // Per-harmonic log delta vs previous frame (mean abs dB), over the
        // common harmonic count.
        let d_db = match &prev {
            Some(p) => {
                let k = p.len().min(amps.len());
                let s: f64 = (0..k)
                    .map(|i| {
                        let a0 = (p[i] as f64).max(1e-6);
                        let a1 = (amps[i] as f64).max(1e-6);
                        (20.0 * (a1 / a0).log10()).abs()
                    })
                    .sum();
                s / k as f64
            }
            None => f64::NAN,
        };
        let note = if f > 0 && d_db < 0.01 { "converged" } else { "" };
        println!("{f:3}  {:3}  {env_rms:10.2}   {d_db:8.4}   {note}", amps.len());
        prev = Some(amps);
    }
}
