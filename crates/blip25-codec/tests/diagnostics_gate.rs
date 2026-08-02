//! Regression guard for `Encoder::set_diagnostics` (the off-by-default gate
//! around the experimental per-frame research paths in `enc::mod`):
//!
//! (a) the flag must not change a single shipped output bit,
//! (b) the `*_x86_*` research logs must be empty with the flag off and
//!     one-entry-per-emitted-frame with it on,
//! (c) the LIVE logs (`gap2_mid`/`gap2_slot1`/`gap2_slot2`/`prefiltered`/
//!     `frame_energy`) must be identical in both modes.

use blip25_codec::enc::{encode_frame, Encoder};

const FRAME: usize = 160;

/// Deterministic pseudo-speech PCM: two summed sines with a slow amplitude
/// ramp plus fixed-seed LCG noise. No I/O, no external fixtures.
fn test_pcm(nframes: usize) -> Vec<i16> {
    let n = nframes * FRAME;
    let mut lcg: u32 = 0x1234_5678;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = i32::from((lcg >> 16) as i16) / 8;
        let t = i as f64 / 8000.0;
        let s = 6000.0 * (2.0 * std::f64::consts::PI * 155.0 * t).sin()
            + 2500.0 * (2.0 * std::f64::consts::PI * 470.0 * t).sin();
        let ramp = 0.25 + 0.75 * (i as f64 / n as f64);
        let v = (s * ramp) as i32 + noise;
        out.push(v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16);
    }
    out
}

/// Stream the clip through the r33 encoder (the shipped AMBE+2 on-air path),
/// draining the final look-ahead frame with `flush_r33`.
fn encode_all(pcm: &[i16], diagnostics: bool) -> (Encoder, Vec<[u8; encode_frame::R33_BYTES]>) {
    let mut enc = Encoder::new();
    enc.set_diagnostics(diagnostics);
    let mut frames = Vec::new();
    for ch in pcm.chunks_exact(FRAME) {
        let mut fr = [0i16; FRAME];
        fr.copy_from_slice(ch);
        if let Some(b) = enc.encode_frame_r33(&fr) {
            frames.push(b);
        }
    }
    frames.extend(enc.flush_r33());
    (enc, frames)
}

#[test]
fn diagnostics_gate_is_bit_transparent() {
    const NFRAMES: usize = 300;
    let pcm = test_pcm(NFRAMES);

    let (enc_off, frames_off) = encode_all(&pcm, false);
    let (enc_on, frames_on) = encode_all(&pcm, true);

    // (a) shipped output bits identical in both modes, frame for frame.
    assert!(!frames_off.is_empty(), "no frames emitted");
    assert_eq!(
        frames_off, frames_on,
        "set_diagnostics(true) changed shipped output bits"
    );
    let n = frames_off.len();

    // (b) diagnostic logs: empty with the flag off ...
    macro_rules! check_diag {
        ($($log:ident),+ $(,)?) => {$(
            assert!(
                enc_off.$log().is_empty(),
                concat!(stringify!($log), " not empty with diagnostics off")
            );
            assert_eq!(
                enc_on.$log().len(),
                n,
                concat!(stringify!($log), " not one-per-frame with diagnostics on")
            );
        )+};
    }
    // ... and exactly one entry per emitted frame with it on.
    check_diag!(
        arg2_x86_log,
        b1_x86_log,
        b2_x86_log,
        b2_x86_pf_log,
        b2_x86_wintaper_log,
        b2_x86_outeronly_log,
        b2_x86_realbins_log,
        realbins_meanraw_log,
        b2_x86_refexact_log,
        b2_x86_stepformula_log,
        b2_x86_stepformula_gateref_log,
        b2_x86_stepformula_realcount_log,
        b2_x86_realbins_realstep_log,
        realbins_realstep_meanraw_log,
        b2_x86_realbins_slot1_realstep_log,
        realbins_slot1_realstep_meanraw_log,
        b2_x86_refexact_slot1_log,
        b2_x86_realbins_slot2_realstep_log,
        realbins_slot2_realstep_meanraw_log,
        b2_x86_realbins_realstep_gateref_log,
        realbins_realstep_gateref_meanraw_log,
        b2_x86_realbins_realstep_realcount_log,
        realbins_realstep_realcount_meanraw_log,
        win_rbs_pre_log,
        l_us_rbs_log,
        array_a_stage1_log,
        array_a_stage2_log,
        b2_gamma_poly_log,
        b2_gamma_poly_param5_log,
    );

    // (b2) prefilter-cadence diagnostic logs: pushed once per SOURCE frame in
    // `run_prefilter` (not once per emitted frame), so they are checked
    // against each other rather than against `n`.
    for (name, len_off, len_on) in [
        (
            "gap2_pre_log",
            enc_off.gap2_pre_log().len(),
            enc_on.gap2_pre_log().len(),
        ),
        (
            "ring_full_pre_log",
            enc_off.ring_full_pre_log().len(),
            enc_on.ring_full_pre_log().len(),
        ),
        (
            "ring_full_mid_log",
            enc_off.ring_full_mid_log().len(),
            enc_on.ring_full_mid_log().len(),
        ),
    ] {
        assert_eq!(len_off, 0, "{name} not empty with diagnostics off");
        assert!(
            len_on >= n,
            "{name} shorter than emitted frames with diagnostics on"
        );
        assert_eq!(
            len_on,
            enc_on.gap2_pre_log().len(),
            "{name} length differs from its prefilter-cadence siblings"
        );
    }

    // (c) live logs identical in both modes.
    assert_eq!(enc_off.gap2_mid_log(), enc_on.gap2_mid_log());
    assert_eq!(enc_off.gap2_slot1_log(), enc_on.gap2_slot1_log());
    assert_eq!(enc_off.gap2_slot2_log(), enc_on.gap2_slot2_log());
    assert_eq!(enc_off.prefiltered_log(), enc_on.prefiltered_log());
    assert_eq!(enc_off.frame_energy_log(), enc_on.frame_energy_log());
    assert_eq!(
        enc_off.gap2_mid_log().len(),
        n,
        "live log length != emitted frames"
    );
}
