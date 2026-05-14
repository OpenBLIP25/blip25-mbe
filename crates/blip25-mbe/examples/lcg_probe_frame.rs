//! Probe 1: synthesize an all-unvoiced AMBE+2 FEC frame for LCG isolation.
//!
//! Construct `b̂₀..b̂₈` such that the decoded synth output is pure
//! shaped LCG noise — no harmonic content — so chip vs ours per-sample
//! diff reveals only the LCG schedule difference.
//!
//! Parameters chosen:
//!   b̂₀ = 60     mid-range pitch index from Annex L (moderate ω₀, L)
//!   b̂₁ = 16     AMBE_VUV_CODEBOOK[16] → all-unvoiced expansion
//!   b̂₂ = 16     mid-range Annex O gain index
//!   b̂₃ = 0      lowest-index PRBA24 (flat low coefficients)
//!   b̂₄ = 0      lowest-index PRBA58
//!   b̂₅..b̂₈ = 0  lowest-index HOC codebooks (flat M̃)
//!
//! Output:
//!   stdout: 9-byte hex per line, 300 frames
//!   stderr: decoded (L, ω₀, M̃[1..=L], v_bar) for sanity check
//!
//! Use as:
//!   ./target/release/examples/lcg_probe_frame > /tmp/probe.hex
//!   xxd -r -p /tmp/probe.hex > /tmp/probe.ambe9      # to binary
//!   # or directly:
//!   ./target/release/examples/lcg_probe_frame --binary > /tmp/probe.ambe9
//!
//! Then:
//!   ./target/release/conformance-speech-quality decode-fec-halfrate \
//!     --binary --input /tmp/probe.ambe9 --out-wav /tmp/probe_ours.wav
//!   scp /tmp/probe.ambe9 root@192.168.1.6:/tmp/probe.ambe9
//!   ssh root@192.168.1.6 'dvsi decode-halfrate /tmp/probe.ambe9 /tmp/probe_chip.pcm'

use blip25_mbe::ambe_plus2_wire::dequantize::{
    DecoderState, Decoded, decode_to_params,
};
use blip25_mbe::ambe_plus2_wire::frame::{encode_frame, DIBITS_PER_FRAME};
use blip25_mbe::ambe_plus2_wire::priority::{prioritize, AMBE_B_COUNT};

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

    // Construct b̂₀..b̂₈ for an all-unvoiced, flat-spectrum frame.
    let mut b = [0u16; AMBE_B_COUNT];
    b[0] = 60;  // pitch index — mid range
    b[1] = 16;  // V/UV codebook → all-unvoiced
    b[2] = 16;  // gain index — mid range
    b[3] = 0;   // PRBA24 = 0
    b[4] = 0;   // PRBA58 = 0
    b[5] = 0;   // HOC_B5 = 0
    b[6] = 0;   // HOC_B6 = 0
    b[7] = 0;   // HOC_B7 = 0
    b[8] = 0;   // HOC_B8 = 0

    // Sanity-check what the decoder will see.
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
                "Decoded probe frame: L={} ω₀={:.4} V={}/{} amps[1..6]=[{}]",
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
        }
        other => {
            eprintln!("Probe frame decoded as {:?}", other);
            std::process::exit(1);
        }
    }

    // Encode + pack the frame. Then emit it N_FRAMES times. Each frame is
    // independent on the wire; decoder state evolves across them.
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
    eprintln!(
        "Wrote {} frames × 9 bytes = {} bytes ({} s of synthesized audio)",
        N_FRAMES,
        N_FRAMES * 9,
        N_FRAMES * 20 / 1000
    );
}
