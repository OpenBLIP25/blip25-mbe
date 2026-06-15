//! Probe 3: all-VOICED AMBE+2 frame to isolate the phase-regeneration path.
//!
//! With every band voiced, the output is pure harmonic synthesis (Eq. 127-141
//! voiced branch in our impl, US5701390 §5 phase regen for AMBE+/AMBE+2).
//! The unvoiced LCG path contributes only to ρ_l noise samples — but for
//! AMBE+2 phase mode we don't use Eq. 141's ρ_l at all (phase regen replaces
//! it). So a per-sample chip-vs-ours diff on this probe localizes any
//! disagreement in the phase-regen algorithm itself.
//!
//! Parameters:
//!   b̂₀ = 60     mid-range pitch index from Annex L
//!   b̂₁ = 0      AMBE_VUV_CODEBOOK[0] → all-voiced expansion
//!   b̂₂ = 16     mid-range Annex O gain index
//!   b̂₃..b̂₈ = 0  lowest-index codebooks (flat M̃ where possible)

use blip25_mbe::rate33::dequantize::{decode_to_params, Decoded, DecoderState};
use blip25_mbe::rate33::frame::{encode_frame, DIBITS_PER_FRAME};
use blip25_mbe::rate33::priority::{prioritize, AMBE_B_COUNT};

const N_FRAMES: usize = 300;

fn pack_dibits(dibits: &[u8; DIBITS_PER_FRAME]) -> [u8; 9] {
    let mut out = [0u8; 9];
    let mut bit = 0;
    for &d in dibits {
        for pos in (0..2).rev() {
            let b = (d >> pos) & 1;
            out[bit / 8] |= b << (7 - (bit % 8));
            bit += 1;
        }
    }
    out
}

fn main() {
    let binary = std::env::args().any(|a| a == "--binary");

    let mut b = [0u16; AMBE_B_COUNT];
    b[0] = 60; // pitch
    b[1] = 0; // V/UV → all-voiced
    b[2] = 16; // gain
               // b[3..=8] = 0

    let info = prioritize(&b);
    let mut decoder = DecoderState::new();
    match decode_to_params(&info, &mut decoder) {
        Ok(Decoded::Voice(p)) => {
            let l = p.harmonic_count();
            let omega = p.omega_0();
            let v = p.voiced_slice();
            let amps = p.amplitudes_slice();
            let v_count = v.iter().filter(|&&x| x).count();
            eprintln!(
                "Decoded all-voiced probe: L={} ω₀={:.4} V={}/{} amps[1..6]=[{}]",
                l,
                omega,
                v_count,
                l,
                amps[..6.min(amps.len())]
                    .iter()
                    .map(|a| format!("{:.0}", a))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            if v_count != l as usize {
                eprintln!("WARNING: not fully voiced — V={}/{}", v_count, l);
            }
        }
        other => {
            eprintln!("Probe decoded as {:?}", other);
            std::process::exit(1);
        }
    }

    let dibits = encode_frame(&info);
    let bytes = pack_dibits(&dibits);

    if binary {
        use std::io::Write;
        let stdout = std::io::stdout();
        let mut h = stdout.lock();
        for _ in 0..N_FRAMES {
            h.write_all(&bytes).unwrap();
        }
    } else {
        let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        for _ in 0..N_FRAMES {
            println!("{}", hex);
        }
    }
    eprintln!("Wrote {} frames ({} ms)", N_FRAMES, N_FRAMES * 20);
}
