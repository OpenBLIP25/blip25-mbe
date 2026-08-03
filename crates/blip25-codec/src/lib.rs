//! Reference-validated reference codec for the reference P25 MBE vocoder family —
//! **both** IMBE (Phase 1) and AMBE+2 (Phase 2), **both** encode and decode.
//!
//! Decode pipeline (either codec): wire frame -> FEC deframe -> info bits ->
//! dequantize -> [`dequantize::MbeParams`] -> synthesis ([`synth`]/[`dec`]) ->
//! 8 kHz i16 PCM. Encode is the inverse through [`enc`].
//!
//! # Module layout — one shared MBE core, two wire front-ends
//!
//! IMBE and AMBE+2 are **peers**: two different wire encodings of the same
//! Multi-Band-Excitation parameter model ([`dequantize::MbeParams`]). Neither
//! is converted into the other and either stands alone ([`ImbeDecoder`] vs
//! [`Decoder`]/[`Encoder`]). They meet at `MbeParams`, then share the
//! analysis/synthesis internals.
//!
//! The file tree looks lopsided — there is an [`imbe`] folder but no `ambe`
//! folder — and that reflects a real asymmetry, not a hierarchy:
//!
//! * [`imbe`] is the IMBE **front-end**, cleanly self-contained (its own
//!   `frame`/`dequantize`/`quantize`/`tables`/`dsp`/`math`/`fixp`).
//! * The **AMBE+2 front-end is fused with the shared core**, not in a folder of
//!   its own: the engine is AMBE+2/MBE-core-first, so the AMBE+2 wire layer
//!   ([`dequantize`], [`frame`], [`tables`], [`tone`]) is entangled with the
//!   shared parts. In particular [`dequantize`] hosts
//!   `MbeParams` *itself* — the type IMBE also produces — so it is not a
//!   pure AMBE+2 module.
//! * **Shared by both codecs:** the analysis engine [`enc`] (its [`enc::Encoder`]
//!   does r33/r34 *and* IMBE), the synthesis back-end [`dec`]/[`synth`], and
//!   [`phase_regen`]/[`fw_tables`]/[`fec`] (plus the crate-internal `fixops`).
//!
//! So do not read a "primary vs. bolted-on" relationship into the folders: the
//! split is "self-contained IMBE front-end" vs. "AMBE+2 front-end fused with
//! the shared MBE core."
//!
//! The VQ codebooks (PRBA24/PRBA58/HOC b5..b8, Annex-O gain) are loaded
//! bit-exact from the reference vocoder's C55x firmware via the
//! `blip25_codebooks` crate. The normative spec tables (Annex L/M/N/S +
//! bit prioritization) are baked into source from TIA-102.BABA-A.
//!
//! See `README.md` for the reference-parity validation result and the
//! caveat about the synthesis phase back-end (the P-8 proprietary
//! frontier).

// Internal codec docs cross-reference `pub(crate)` helpers by name; those
// links are intentional. `broken_intra_doc_links` stays on for dead links.
#![allow(rustdoc::private_intra_doc_links)]
// In a decode-only build (`--no-default-features --features decode`) the encoder
// pipeline is gated out, so the encode-only helpers that live in `shared/`
// alongside the decoder's are legitimately unused. Silence dead-code noise only
// in that configuration; the default (encode+decode) build keeps the lint hot.
#![cfg_attr(not(feature = "encode"), allow(dead_code))]

pub mod dec;
pub mod dequantize;
/// Encoder analysis pipeline (PCM -> frame bits). Gated behind the `encode`
/// feature: a decode-only build omits it entirely. The DSP/BFP primitives the
/// decoder also needs were factored out into [`shared`] so `dec/` + the
/// synthesis path never reach into `enc/`.
#[cfg(feature = "encode")]
pub mod enc;
pub mod fec;
pub(crate) mod fixops;
pub mod frame;
pub mod fw_tables;
pub mod imbe;
pub mod phase_regen;
/// DSP/BFP primitives shared by [`enc`] and [`dec`]. Available whenever the
/// crate is built (both `encode` and `decode` depend on it).
pub mod shared;
pub mod synth;
pub mod tables;
pub mod tone;

use dequantize::{DecodeError, DecoderState, MbeParams};
use frame::Frame;
use synth::SynthState;

use dec::excitation::{lcg_next, NOISE_RING_LEN};
use dec::gen_ml::MlGen;
use dec::gen_resid::resid_from_ml;
use dec::gen_vmask::{build_maskword, gen_vmask, p4e_from_bits, sitea_maskword};
use dec::linamp::{amp_from_mllog, amp_from_mllog_ungained};
use dec::ola_body::{voiced_ola_body, FrameParams, FrameParams2, OlaState};
use dec::ola_driver::block_denorm;
use dec::unv_frame::{synth_unvoiced_subframe, STATE_RING, STATE_SEED, STATE_WORDS};
use dec::unvoiced::{biquad_postfilter, UnvoicedBandParams};
use shared::voicing_map::l_step_from_b0_b1;

/// Number of 80-sample per-site synthesis calls per 9-byte r33 subframe: site A
/// (mid-frame) + site B. Two calls -> 160 PCM samples, matching
/// [`Decoder::decode_pcm`]'s external contract.
const SITE_SAMPLE_COUNT: usize = 80;

#[inline]
fn sx16(v: i32) -> i32 {
    (v as u16) as i16 as i32
}

/// The frame loudness-gain slew. `n = popcount(mask&0x55555555)`;
/// `n>=8 -> 0.9*prev + 0.1*target`, else freeze. Computed once per subframe at
/// site A and shared to site B.
#[inline]
fn gain_slew(prev: i32, target: i32, mask: u32) -> i32 {
    let n = (mask & 0x5555_5555).count_ones() as i32;
    if n >= 8 {
        let target_w = sx16(target).wrapping_mul(0xcccc);
        let prev_w = sx16(prev).wrapping_mul(0xe666u32 as i32);
        ((target_w >> 3).wrapping_add(prev_w) >> 16) & 0xffff
    } else {
        (sx16(prev) as u32 & 0xffff) as i32
    }
}

/// Per-element pack: round + saturate a fixed shift-2 sample.
#[inline]
fn round_pack16(shifted: i32) -> i16 {
    let val = shifted;
    let rounded = val.wrapping_add(0x8000);
    let ovf = (rounded ^ val) & rounded;
    let sat = if ovf >= 0 {
        rounded
    } else if val < 0 {
        0x8000_0000u32 as i32
    } else {
        0x7fff_ffff
    };
    (sat >> 16) as i16
}

/// Cross-subframe state for the from-bits fixed-point decode path
/// ([`Decoder::decode_pcm_fixed`]). Cold-initialized to the DLL power-on state.
struct FixedState {
    mlgen: MlGen,
    /// Site-A loudness-gain register; DLL power-on seed 7316.
    gain_a: i32,
    /// Voiced OLA body state; cold = all-zero.
    ola: OlaState,
    /// Unvoiced synthesis state; cold-primed by drawing
    /// `NOISE_RING_LEN` LCG words from seed 0 (ending seed 34076).
    unv_state: Vec<i16>,
    /// Post-filter carry (6 words), cold = 0.
    fca: [i32; 6],
    /// Previous synthesis call's voicing word / omega0 (the `p5` operand).
    prev_vmask: u32,
    prev_omega0: i32,
    /// Previous SUBFRAME's omega0 / M_l / maskword (site-A interpolation
    /// source). Cold = 0.
    prev_sub_om: i16,
    prev_sub_ml: [i16; 56],
    prev_sub_mask: u32,
}

/// Site-A omega0 = geometric mean of cur/prev grids (Q15 sqrt),
/// with the three voicing-mask fallbacks.
fn sitea_omega0(a_mask: u32, cur_om: i32, prev_om: i32, cur_mask: u32, prev_mask: u32) -> i16 {
    use shared::gamma_poly::block_float_word_rescale;
    if (prev_mask & 0x5555_5555) == 0 {
        return cur_om as i16;
    }
    if (a_mask & 0xaaaa_aaaa) != 0 {
        return cur_om as i16;
    }
    if (cur_mask & 0x5555_5555) == 0 {
        return prev_om as i16;
    }
    let c = (cur_om as i16 as i32)
        .wrapping_mul(prev_om as i16 as i32)
        .wrapping_mul(2);
    block_float_word_rescale(c, -8, -4)
}

impl FixedState {
    fn new() -> Self {
        let mut unv_state = vec![0i16; STATE_WORDS];
        let mut seed: u16 = 0;
        for k in 0..NOISE_RING_LEN {
            unv_state[STATE_RING + k] = lcg_next(&mut seed);
        }
        unv_state[STATE_SEED] = seed as i16;
        FixedState {
            mlgen: MlGen::new(),
            gain_a: 7316,
            ola: OlaState::default(),
            unv_state,
            fca: [0i32; 6],
            prev_vmask: 0,
            prev_omega0: 16384,
            prev_sub_om: 0,
            prev_sub_ml: [0i16; 56],
            prev_sub_mask: 0,
        }
    }
}

/// Streaming PCM -> AMBE+2 frame encoder (the inverse of [`Decoder`]).
#[cfg(feature = "encode")]
pub use enc::Encoder;

/// Per-frame concealment disposition (TIA-102.BABA-1 §5.6/§5.7), surfaced to the
/// consumer so the framing layer can drive higher-level policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FrameDisposition {
    /// Frame decoded normally from its own bits.
    #[default]
    Use,
    /// FEC error gate / erasure fired — the previous good frame's audio was
    /// repeated in place of this frame.
    Repeat,
    /// Mute gate fired (≥4 consecutive invalid frames, or εR > 0.096) — output
    /// is uniform comfort noise on [−5, 5], not the (untrustworthy) frame.
    Mute,
    /// The frame was an explicit silence escape (`û₀(11..6) == 0x3E`), not a
    /// channel-error indication. Distinct from [`Self::Mute`], which is a
    /// concealment decision: silence is what the sender asked for, so it does
    /// not advance the repeat/mute counters.
    Silence,
}

/// A streaming decoder: holds the cross-frame dequant + synth state so
/// frames can be fed in order.
///
/// Also implements the TIA-102.BABA-1 half-rate error-smoothing / frame-
/// repeat / frame-mute logic (Addendum §5.5-5.7, Eq. 55-64): a running
/// `epsilon_R` error-rate estimate drives (a) a per-frame repeat-previous-
/// frame gate keyed on the Golay error counts, and (b) a mute-to-noise gate
/// when the smoothed error rate or a run of invalid frames gets too high.
pub struct Decoder {
    deq: DecoderState,
    syn: SynthState,
    /// Last successfully decoded frame's params (repeat-frame source, float path).
    last_params: Option<MbeParams>,
    /// Last successfully decoded frame's raw r33 bytes (repeat source for the
    /// shipped fixed-point path, [`Decoder::decode_pcm_fixed_concealed`]).
    last_good_bytes: Option<[u8; 9]>,
    /// Smoothed error rate epsilon_R(n) = 0.95*epsilon_R(n-1) + 0.001064*epsilon_T(n).
    epsilon_r: f64,
    /// Count of consecutive invalid (repeated/erasure) frames.
    consecutive_invalid: u32,
    /// LCG state for the mute-noise generator (same 173/13849/65536 LCG the
    /// unvoiced synthesis path uses, independently seeded/advanced).
    mute_lcg: u32,
    /// Lazily-created state for the from-bits fixed-point path
    /// ([`Decoder::decode_pcm_fixed`]). `None` until the first call.
    fixed: Option<Box<FixedState>>,
}

/// Everything one synthesis site consumes, in the engine's own terms.
///
/// The voice path derives these from the frame bits; the Annex-T path builds
/// them from `(I_D, A_D)`. That is the whole point of the split — the DLL has
/// one synthesis engine and reaches it from both classes, so the tail it
/// carries between frames is shared.
struct SiteParams<'a> {
    /// Harmonic count for this site.
    l: usize,
    /// Fundamental step word (ω₀ · 262144/π).
    step: i16,
    /// Packed voicing word.
    mask_word: u32,
    /// Per-harmonic log-magnitudes.
    ml: &'a [i16],
    /// `p4.e` — the L=56 mixed-band pitch scalar; 0 whenever `l != 56`.
    p4e: i32,
}

/// Which class is driving [`synth_site`].
///
/// The two differ only in what they touch on the way through: a tone frame
/// leaves the loudness-gain register alone and takes the short amplitude
/// chain. Both advance the OLA remainder, the phase carry
/// (`prev_vmask`/`prev_omega0`) and the unvoiced generator.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SiteMode {
    Voice,
    Tone,
}

/// Render one synthesis site (80 samples) into `out`, advancing `st`.
///
/// Extracted verbatim from the voice path so the Annex-T path can reach the
/// same engine. `SiteMode::Voice` is bit-for-bit what the loop did inline.
fn synth_site(
    st: &mut FixedState,
    out: &mut [i16; synth::FRAME_SAMPLES],
    site: usize,
    p: SiteParams<'_>,
    mode: SiteMode,
) {
    let SiteParams {
        l: sl,
        step: som16,
        mask_word: smask,
        ml: sml,
        p4e,
    } = p;
    let som = som16 as i32;

    // Site A slews the shared loudness gain from bits. Tone frames skip the
    // gain update outright (the reference's tone branch jumps past it), so
    // their level comes from A_D alone rather than from preceding speech.
    if site == 0 && mode == SiteMode::Voice {
        st.gain_a = gain_slew(st.gain_a, som, smask);
    }
    let gain_a = st.gain_a;

    if sl == 0 || sml.len() < sl {
        // degenerate site: emit the (self-carried) unvoiced-less silence.
        st.prev_vmask = smask;
        st.prev_omega0 = som;
        return;
    }

    let gen_mask = gen_vmask(smask, som, sl as i32);
    let resid = resid_from_ml(sml, sl);

    // Linear amplitude. The reference's tone branch skips the two stages that
    // fold in the gain register and the L=56 correction, so a tone takes the
    // short chain and its magnitudes stay absolute.
    let (amp_vec, amp_bexp) = match mode {
        SiteMode::Voice => amp_from_mllog(sml, sl, som, gain_a),
        SiteMode::Tone => amp_from_mllog_ungained(sml, sl, som),
    };
    let mut amp56 = [0i16; 56];
    amp56[..sl.min(amp_vec.len())].copy_from_slice(&amp_vec[..sl.min(amp_vec.len())]);

    // unvoiced accumulator (advances unv_state once per synthesis call).
    let vv: Vec<i16> = (0..sl).map(|k| gen_mask[k]).collect();
    let unv_amps: Vec<i16> = amp56[..sl].to_vec();
    let unv_st = UnvoicedBandParams {
        mode: 1,
        l: sl as i16,
        coef: som16,
        amps: &unv_amps,
        voicing: &vv,
        base_exp: amp_bexp as u16,
    };
    let mut out_acc = vec![0i32; SITE_SAMPLE_COUNT + 16];
    synth_unvoiced_subframe(
        &mut out_acc,
        SITE_SAMPLE_COUNT as i32,
        &mut st.unv_state,
        &unv_st,
        gain_a,
    );

    // voiced OLA body (self-carried OLA state; from-bits resid).
    st.ola.resid = [0i16; 56];
    st.ola.resid[..sl.min(resid.len())].copy_from_slice(&resid[..sl.min(resid.len())]);
    let maskv: Vec<i16> = gen_mask.to_vec();
    let p4 = FrameParams {
        hi: sl as i32,
        mask_word: smask,
        l: som,
        e: p4e,
        amp: amp56,
        mask: &maskv,
        base_exp: amp_bexp,
    };
    let p5 = FrameParams2 {
        mask_word: st.prev_vmask,
        l5: st.prev_omega0,
    };
    voiced_ola_body(&mut out_acc, SITE_SAMPLE_COUNT as i32, &mut st.ola, &p4, &p5, gain_a);

    // post-filter + pack -> 80 PCM samples.
    let inp: Vec<i32> = out_acc[..SITE_SAMPLE_COUNT].to_vec();
    let mut ofc = out_acc.clone();
    biquad_postfilter(&inp, &mut ofc, &mut st.fca, SITE_SAMPLE_COUNT as i16);
    let base = site * SITE_SAMPLE_COUNT;
    for i in 0..SITE_SAMPLE_COUNT {
        out[base + i] = round_pack16(block_denorm(ofc[i], 2));
    }

    st.prev_vmask = smask;
    st.prev_omega0 = som;
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            deq: DecoderState::new(),
            syn: SynthState::new(),
            last_params: None,
            last_good_bytes: None,
            epsilon_r: 0.0,
            consecutive_invalid: 0,
            mute_lcg: 1,
            fixed: None,
        }
    }

    /// Decode one 9-byte r33 frame to 160 PCM samples via the **from-bits
    /// fixed-point** synthesis path — the per-subframe dispatcher
    /// composed entirely in fixed point, driven from `bytes` alone.
    ///
    /// This is the parallel to [`Decoder::decode_pcm`] that replaces the float
    /// `synth::synthesize_frame` with the DLL's integer synthesis: for the
    /// subframe it generates `M_l` (the spectral-amplitude chain), the linear
    /// amplitude, the per-harmonic voicing array, the voiced-phase residual
    /// (Hilbert of `M_l`), and the unvoiced accumulator, then runs the voiced
    /// OLA body, the post-filter and the pack for the two synthesis sites
    /// (A mid-frame + B), emitting 2*80 = 160 samples. ALL decoder state
    /// (`M_l` predictor, gain register, OLA, unvoiced ring, `p5`) is
    /// self-carried from cold init.
    pub fn decode_pcm_fixed(&mut self, bytes: &[u8]) -> [i16; synth::FRAME_SAMPLES] {
        if self.fixed.is_none() {
            self.fixed = Some(Box::new(FixedState::new()));
        }
        // Framing from bits (no repeat/mute gate: raw per-frame synthesis).
        let fr = frame::decode_bytes(bytes);
        let b = tables::deprioritize(&fr.info);
        let mut out = [0i16; synth::FRAME_SAMPLES];

        // --- Annex-T tone synthesis.
        //
        // Tone frames run through the SAME fixed-point engine as voice, so the
        // ~168-sample output remainder the OLA carries between frames is
        // shared. That sharing is the whole point: a tone frame arriving
        // mid-word must overlap-add onto the outgoing voiced tail rather than
        // start from silence, or the transition punches a hole in the audio and
        // the displaced tail reappears a frame late.
        //
        // What a tone frame touches, and what it leaves alone:
        //   * OLA remainder                    — advances (via `synth_site`)
        //   * phase carry (prev_vmask/omega0)   — advances
        //   * unvoiced generator               — advances
        //   * M_l predictor (`mlgen`, prev_sub_*) — frozen: we return before it
        //   * loudness gain register           — frozen: `SiteMode::Tone`
        match tone::classify_decoded(&fr.info) {
            tone::FrameKind::Tone => {
                if let Some(f) = tone::parse_tone_frame(&fr.info) {
                    if let Some(block) = tone::tone_site_block(f.id, f.amplitude) {
                        let st = self.fixed.as_mut().unwrap();
                        for site in 0..2 {
                            synth_site(
                                st,
                                &mut out,
                                site,
                                SiteParams {
                                    l: block.l,
                                    step: block.step,
                                    mask_word: block.mask_word,
                                    ml: &block.ml,
                                    // `p4.e` is the L=56 mixed-band pitch
                                    // scalar; a tone's L is never 56.
                                    p4e: 0,
                                },
                                SiteMode::Tone,
                            );
                        }
                        // Return before the `prev_sub_*` carry at the end of
                        // the voice path — that is the M_l predictor's context,
                        // which a tone frame must not advance.
                        return out;
                    }
                }
                return out;
            }
            tone::FrameKind::Erasure | tone::FrameKind::Silence => {
                // Raw per-frame synthesis: no repeat machinery here, so both
                // escapes emit nothing. The repeat/mute distinction between
                // them is made by `decode_pcm_fixed_concealed`.
                //
                // KNOWN GAP: the reference's silence class is not digital
                // zero — it decodes as an ordinary parameter frame with
                // forced pitch/voicing, so its level tracks the frame's own
                // gain bits and the preceding decoder state (measured at up
                // to 21 dB apart for identical input frames). Muting is an
                // approximation with a level error that depends on what was
                // being said before the silence.
                return out;
            }
            tone::FrameKind::Voice => {} // fall through to existing voice synth
        }

        let (l, omega0_i16) = match l_step_from_b0_b1(b[0] as u8, b[1]) {
            Some(v) => v,
            None => return out, // tone/erasure -> silence
        };
        let omega0 = omega0_i16 as i32;
        // Site-B (even call) voicing word; site A (odd/mid-frame) has its OWN.
        let maskword = build_maskword(b[0] as u8, b[1], l, omega0_i16);
        let maskword_a = sitea_maskword(b[0] as u8, b[1], l, omega0_i16);

        // Site-B M_l from bits (advances predictor state once per subframe).
        let ml = self
            .fixed
            .as_mut()
            .unwrap()
            .mlgen
            .next_subframe(&b, &fr.info, l as i32)
            .unwrap_or_default();
        if l == 0 || ml.len() < l {
            return out;
        }

        // Pad the tail [L..56] with the LAST LIVE HARMONIC (ml[L-1]), matching
        // the DLL's site-B pstruct M_l array: the tail is last-harmonic-padded,
        // NOT zero. The site-A resampler reads into this tail when a_l > L (L
        // grows between subframes); zero-padding it is the dominant site-A M_l
        // residual.
        let mut cur_ml56 = [0i16; 56];
        let ml_fill = *ml.get(l.min(ml.len()).saturating_sub(1)).unwrap_or(&0);
        for i in 0..56 {
            cur_ml56[i] = if i < l.min(ml.len()) { ml[i] } else { ml_fill };
        }

        // ---- site-A (mid-frame) grid + interpolated M_l ----
        // Site A drives the first 80 samples of every subframe; its M_l is the
        // interpolation of this subframe's and the previous subframe's M_l onto
        // the geometric-mean grid a_om/a_l. (Using site B's M_l for site A
        // instead costs ~61pp of sample-exactness.)
        let (a_om, a_l, sa_ml, prev_sub_om, prev_sub_mask) = {
            let s = self.fixed.as_ref().unwrap();
            // a_mask = the site-A struct's own maskword (reads dest.mask);
            // cur_mask = the current site-B word.
            let a_om = sitea_omega0(
                maskword_a,
                omega0,
                s.prev_sub_om as i32,
                maskword,
                s.prev_sub_mask,
            );
            let a_l = (crate::shared::step_count::count_from_step(a_om) as usize).min(56);
            let sa_ml = dec::sitea_ml::build_sitea_magnitudes(
                a_om,
                a_l,
                omega0_i16,
                &cur_ml56,
                s.prev_sub_om,
                &s.prev_sub_ml,
            );
            (a_om, a_l, sa_ml, s.prev_sub_om, s.prev_sub_mask)
        };
        let _ = (prev_sub_om, prev_sub_mask);
        let sa_ml_vec: Vec<i16> = sa_ml[..a_l].to_vec();

        // Per-site (L, omega0_i16, omega0, M_l, maskword). Site A first (odd:
        // updates the gain register), then site B.
        let site_ml_a = sa_ml_vec;
        let site_ml_b = ml.clone();
        let sites: [(usize, i16, u32, &Vec<i16>); 2] = [
            (a_l, a_om, maskword_a, &site_ml_a),
            (l, omega0_i16, maskword, &site_ml_b),
        ];

        for (site, &(sl, som16, smask, sml)) in sites.iter().enumerate() {
            synth_site(
                self.fixed.as_mut().unwrap(),
                &mut out,
                site,
                SiteParams {
                    l: sl,
                    step: som16,
                    mask_word: smask,
                    ml: sml,
                    p4e: p4e_from_bits(site, b[0], b[1], l),
                },
                SiteMode::Voice,
            );
        }

        // carry this subframe as "prev" for the next subframe's site-A interp.
        let st = self.fixed.as_mut().unwrap();
        st.prev_sub_om = omega0_i16;
        st.prev_sub_ml = cur_ml56;
        st.prev_sub_mask = maskword;
        out
    }

    /// Deframe + dequantize one 9-byte r33 frame to [`MbeParams`], applying
    /// the BABA-1 §5.6 frame-repeat gate. Returns the FEC [`Frame`] alongside.
    /// `Err(BadPitch)` indicates a tone frame (b0 in [124,255]) with no prior
    /// frame to repeat.
    pub fn decode_params(&mut self, bytes: &[u8]) -> (Frame, Result<MbeParams, DecodeError>) {
        let f = frame::decode_bytes(bytes);
        let b = tables::deprioritize(&f.info);
        let b0_raw = b[0] as u8;

        // epsilon_0/epsilon_1 per BABA-1 Eq. 196 (u8::MAX = uncorrectable c0 -> 4).
        let epsilon_0 = if f.errors[0] == u8::MAX {
            4
        } else {
            f.errors[0]
        };
        let epsilon_1 = f.errors[1];
        let epsilon_t = epsilon_0.saturating_add(epsilon_1);
        self.epsilon_r = 0.95 * self.epsilon_r + 0.001064 * f64::from(epsilon_t);

        // b0 in [120,123] is the half-rate erasure signal (distinct from the
        // [124,255] tone-frame range, which stays out of scope here).
        let is_erasure = (120..=123).contains(&b0_raw);
        let do_repeat = epsilon_0 >= 4 || (epsilon_0 >= 2 && epsilon_t >= 6) || is_erasure;

        let p = if do_repeat {
            self.consecutive_invalid = self.consecutive_invalid.saturating_add(1);
            match &self.last_params {
                Some(last) => Ok(last.clone()),
                None => dequantize::dequantize(&f.info, &mut self.deq),
            }
        } else {
            let r = dequantize::dequantize(&f.info, &mut self.deq);
            if r.is_ok() {
                self.consecutive_invalid = 0;
            } else {
                self.consecutive_invalid = self.consecutive_invalid.saturating_add(1);
            }
            r
        };

        if let Ok(params) = &p {
            self.last_params = Some(params.clone());
        }

        (f, p)
    }

    /// True when BABA-1 §5.7's mute gate fires: smoothed error rate too high,
    /// or 4+ consecutive invalid (repeated/erasure) frames.
    fn muted(&self) -> bool {
        self.epsilon_r > 0.096 || self.consecutive_invalid >= 4
    }

    /// Current smoothed error rate epsilon_R(n) (diagnostic/test hook).
    pub fn epsilon_r(&self) -> f64 {
        self.epsilon_r
    }

    /// Current consecutive-invalid-frame count (diagnostic/test hook).
    pub fn consecutive_invalid(&self) -> u32 {
        self.consecutive_invalid
    }

    /// Uniform noise on [-5,5] (BABA-1 §5.7 frame-mute output).
    fn mute_noise_frame(&mut self) -> [i16; synth::FRAME_SAMPLES] {
        let mut out = [0i16; synth::FRAME_SAMPLES];
        for s in out.iter_mut() {
            self.mute_lcg = 173u32.wrapping_mul(self.mute_lcg).wrapping_add(13849) & 0xFFFF;
            *s = (self.mute_lcg % 11) as i16 - 5;
        }
        out
    }

    /// Decode one r33 frame all the way to 160 PCM samples. Tone/erasure
    /// frames synthesize silence (the tone path is out of scope here);
    /// muted frames (per BABA-1 §5.7) synthesize uniform noise instead.
    pub fn decode_pcm(&mut self, bytes: &[u8]) -> [i16; synth::FRAME_SAMPLES] {
        let (_f, p) = self.decode_params(bytes);
        let muted = self.muted();
        match (p, muted) {
            (_, true) => self.mute_noise_frame(),
            (Ok(params), false) => {
                // Voice float path: phase regen ON (the tone path may have left
                // `suppress_regen` set on this shared state; voice needs regen).
                self.syn.set_tone_frame(false);
                synth::synthesize_frame(&params, &mut self.syn)
            }
            (Err(_), false) => [0i16; synth::FRAME_SAMPLES],
        }
    }

    /// Shipped fixed-point decode WITH the BABA-1 §5.6/§5.7 concealment gate.
    ///
    /// Byte-identical to [`Decoder::decode_pcm_fixed`] on clean frames (the gate
    /// is inert when ε₀ = 0 and b0 is a voice index). On an FEC-error / erasure
    /// frame it **repeats** the previous good frame; after ≥4 consecutive invalid
    /// frames (or εR > 0.096) it **mutes** to comfort noise. Returns the PCM plus
    /// the [`FrameDisposition`] and (ε₀, εₜ) counts for the consumer / framing
    /// layer. The repeat/mute *policy* is the codec's per BABA-1; a framing layer
    /// that already knows a frame is lost can still drive it via those counts.
    pub fn decode_pcm_fixed_concealed(
        &mut self,
        bytes: &[u8],
    ) -> ([i16; synth::FRAME_SAMPLES], FrameDisposition, u8, u8) {
        let f = frame::decode_bytes(bytes);
        let b0_raw = tables::deprioritize(&f.info)[0] as u8;
        // ε₀ per BABA-1 Eq. 196 (uncorrectable c0 → 4); εₜ = ε₀ + ε₁.
        let epsilon_0 = if f.errors[0] == u8::MAX {
            4
        } else {
            f.errors[0]
        };
        let epsilon_t = epsilon_0.saturating_add(f.errors[1]);
        self.epsilon_r = 0.95 * self.epsilon_r + 0.001064 * f64::from(epsilon_t);
        // Tone frames (§2.10) carry a `b̂₀` in the erasure range [120,123] but are
        // NOT erasures — the tone signature (`classify`) is unambiguous. Route
        // them straight to synthesis; the FEC-error/erasure repeat+mute machinery
        // is for corrupted VOICE frames only. (Without this, a tone frame is
        // mis-read as an erasure and repeats the last voice frame, whose OLA rings
        // then collapses to silence — the "sustained tone collapse".)
        let kind = tone::classify_decoded(&f.info);
        if kind == tone::FrameKind::Tone {
            self.consecutive_invalid = 0;
            let pcm = self.decode_pcm_fixed(bytes);
            return (pcm, FrameDisposition::Use, epsilon_0, epsilon_t);
        }
        // A silence frame is an escape, not a channel-error indication: it
        // must not advance the repeat/mute counters.
        if kind == tone::FrameKind::Silence {
            self.consecutive_invalid = 0;
            let pcm = self.decode_pcm_fixed(bytes);
            return (pcm, FrameDisposition::Silence, epsilon_0, epsilon_t);
        }
        // Erasure is a *class*, not a `b̂₀` range. The old `b̂₀ ∈ [120,123]`
        // test was a different partition of the escape space: it called every
        // `b̂₀ ∈ [124,127]` escape a non-erasure regardless of b̂₁, and every
        // `b̂₀ ∈ [120,123]` escape an erasure even when b̂₁ marked it silence.
        let is_erasure = kind == tone::FrameKind::Erasure;
        let _ = b0_raw;
        let do_repeat = epsilon_0 >= 4 || (epsilon_0 >= 2 && epsilon_t >= 6) || is_erasure;
        if do_repeat {
            self.consecutive_invalid = self.consecutive_invalid.saturating_add(1);
        } else {
            self.consecutive_invalid = 0;
        }
        if self.muted() {
            return (
                self.mute_noise_frame(),
                FrameDisposition::Mute,
                epsilon_0,
                epsilon_t,
            );
        }
        if do_repeat {
            let pcm = match self.last_good_bytes {
                Some(b) => self.decode_pcm_fixed(&b),
                None => [0i16; synth::FRAME_SAMPLES],
            };
            return (pcm, FrameDisposition::Repeat, epsilon_0, epsilon_t);
        }
        let pcm = self.decode_pcm_fixed(bytes);
        // Hostile input class: truncated wire frames (len < 9) — retain the same
        // zero-padded 9 bytes `decode_bytes` decoded instead of slicing past `len`.
        let mut lb = [0u8; 9];
        let n = bytes.len().min(9);
        lb[..n].copy_from_slice(&bytes[..n]);
        self.last_good_bytes = Some(lb);
        (pcm, FrameDisposition::Use, epsilon_0, epsilon_t)
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// A streaming **IMBE** (P25 Phase 1, 7200 bps full-rate) decoder. Deframes
/// 18-byte IMBE frames, dequantizes via the bit-exact [`imbe`] front-end, and
/// drives the SAME MBE synthesis back-end as [`Decoder`] — only the front-end
/// differs between the two vocoders.
pub struct ImbeDecoder {
    deq: imbe::ImbeState,
    syn: SynthState,
    mute_lcg: u32,
    /// Shared DLL-exact OLA synthesis state (min-phase resid + amp chain + WOLA),
    /// lazily built. Present only when `ola_synth` routes IMBE through the OLA
    /// path instead of the f32 `synth::synthesize_frame`.
    fixed: Option<Box<FixedState>>,
    /// Render voiced frames through the shared DLL-exact OLA synthesis (the
    /// reference's deterministic minimum-phase-of-log-amplitude voiced path + 168-tap
    /// WOLA) — the console-grade IMBE decode.
    ola_synth: bool,
    /// Additive level constant in the log-amp bridge
    /// `sml = (sa_prev1 >> 11) + OFFSET` (Q4.11 log2).
    ola_offset: i32,
}

impl ImbeDecoder {
    pub fn new() -> Self {
        Self {
            deq: imbe::ImbeState::new(),
            syn: SynthState::new_imbe(),
            mute_lcg: 1,
            fixed: None,
            ola_synth: true,
            ola_offset: 0,
        }
    }

    /// Deframe + dequantize one 18-byte IMBE frame to [`MbeParams`], applying
    /// the TIA-102.BABA §7.6-7.7 error-rate repeat gate (see
    /// [`imbe::decode_params`]). Returns `None` only before the first valid
    /// frame.
    pub fn decode_params(&mut self, bytes: &[u8; imbe::FRAME_BYTES]) -> Option<MbeParams> {
        let fr = imbe::deframe(bytes);
        imbe::decode_params(&fr, &mut self.deq)
    }

    /// True when the §7.7 mute gate fires: smoothed error rate too high, or
    /// 4+ consecutive invalid (repeated) frames.
    fn muted(&self) -> bool {
        self.deq.epsilon_r() > 0.0875 || self.deq.consecutive_invalid() >= 4
    }

    /// Uniform noise on [-5,5] (§7.7 frame-mute output).
    fn mute_noise_frame(&mut self) -> [i16; synth::FRAME_SAMPLES] {
        let mut out = [0i16; synth::FRAME_SAMPLES];
        for s in out.iter_mut() {
            self.mute_lcg = 173u32.wrapping_mul(self.mute_lcg).wrapping_add(13849) & 0xFFFF;
            *s = (self.mute_lcg % 11) as i16 - 5;
        }
        out
    }

    /// Build the AMBE-style 2-bit-per-band voicing maskword from IMBE's
    /// per-harmonic V/UV by the same band-index stepping `gen_vmask` uses, so
    /// the shared gates (`popcount & 0x55555555`) and the per-harmonic expansion
    /// round-trip.
    fn imbe_maskword(v_uv: &[i16], som16: i16, l: usize) -> u32 {
        let step = (som16 as i32) << 2;
        let mut acc = step;
        let mut word = 0u32;
        for &v in v_uv.iter().take(l) {
            let j = ((acc >> 16) & 0xffff).min(15) as u32;
            if v != 0 {
                word |= 1 << (2 * j);
            }
            acc += step;
        }
        word
    }

    /// Decode one IMBE frame through the SHARED DLL-exact OLA synthesis — the
    /// path AMBE+2 uses (`resid_from_ml` minimum-phase voiced + `amp_from_mllog` +
    /// the 168-sample WOLA `voiced_ola_body`). The IMBE front-end supplies the
    /// OLA input contract (L, ω₀-step, voicing maskword, log-M_l) and drives the
    /// same two-site (2×80) loop as [`Decoder::decode_pcm_fixed`]. It uses the
    /// reference's deterministic minimum phase rather than the f32
    /// `synth::synthesize_frame` (Eq. 139 + random regen).
    fn decode_imbe_ola(&mut self, ola: &imbe::ImbeOlaParams) -> [i16; synth::FRAME_SAMPLES] {
        if self.fixed.is_none() {
            self.fixed = Some(Box::new(FixedState::new()));
        }
        let mut out = [0i16; synth::FRAME_SAMPLES];
        let l = ola.num_harms as usize;
        if l == 0 {
            return out;
        }
        // ω₀ step word: round(fund_freq / 8192) = (fund_freq + 0x1000) >> 13.
        let omega0_i16 = ((ola.fund_freq + 0x1000) >> 13) as i16;
        let omega0 = omega0_i16 as i32;
        // log-M_l bridge: sml[i] = clamp_i16((sa_prev1[i+1] >> 11) + OFFSET). Both
        // sides are base-2 log2 (IMBE Q10.22 → reference Q4.11); >>11 is the exact
        // scale change, OFFSET the (calibrated) level constant.
        let ml: Vec<i16> = (0..l)
            .map(|i| {
                let v = (ola.log_amps[i + 1] >> 11) + self.ola_offset;
                v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
            })
            .collect();
        let maskword = Self::imbe_maskword(&ola.v_uv_dsn, omega0_i16, l);

        // last-harmonic-padded M_l tail (the site-A resampler reads it when a_l>L).
        let mut cur_ml56 = [0i16; 56];
        let ml_fill = *ml.get(l.saturating_sub(1)).unwrap_or(&0);
        for i in 0..56 {
            cur_ml56[i] = if i < l { ml[i] } else { ml_fill };
        }

        // site A: mid-frame interpolated grid + M_l (shared AMBE interpolation).
        let (a_om, a_l, sa_ml) = {
            let s = self.fixed.as_ref().unwrap();
            let a_om = sitea_omega0(
                maskword,
                omega0,
                s.prev_sub_om as i32,
                maskword,
                s.prev_sub_mask,
            );
            let a_l = (crate::shared::step_count::count_from_step(a_om) as usize).min(56);
            let sa_ml = dec::sitea_ml::build_sitea_magnitudes(
                a_om,
                a_l,
                omega0_i16,
                &cur_ml56,
                s.prev_sub_om,
                &s.prev_sub_ml,
            );
            (a_om, a_l, sa_ml)
        };
        let sa_ml_vec: Vec<i16> = sa_ml[..a_l].to_vec();
        let site_ml_b = ml.clone();
        // IMBE has one per-frame voicing; both sites share the maskword.
        let sites: [(usize, i16, u32, &Vec<i16>); 2] = [
            (a_l, a_om, maskword, &sa_ml_vec),
            (l, omega0_i16, maskword, &site_ml_b),
        ];

        for (site, &(sl, som16, smask, sml)) in sites.iter().enumerate() {
            let som = som16 as i32;
            let st = self.fixed.as_mut().unwrap();
            if site == 0 {
                st.gain_a = gain_slew(st.gain_a, som, smask);
            }
            let gain_a = st.gain_a;
            if sl == 0 || sml.len() < sl {
                st.prev_vmask = smask;
                st.prev_omega0 = som;
                continue;
            }
            // Per-harmonic voicing. For the TRANSMITTED site (B, sl==L) use IMBE's
            // exact per-harmonic V/UV directly — the maskword→gen_vmask band-OR
            // round-trip over-voices ~26% of harmonics. The maskword is
            // still synthesized above for the popcount/OLA gates (which are exact).
            // The interpolated mid-frame site A keeps the grid-derived expansion.
            let gen_mask: [i16; 56] = if site == 1 {
                let mut m = [0i16; 56];
                m[..sl.min(56)].copy_from_slice(&ola.v_uv_dsn[..sl.min(56)]);
                m
            } else {
                gen_vmask(smask, som, sl as i32)
            };
            let resid = resid_from_ml(sml, sl);
            let (amp_vec, amp_bexp) = amp_from_mllog(sml, sl, som, gain_a);
            let mut amp56 = [0i16; 56];
            amp56[..sl.min(amp_vec.len())].copy_from_slice(&amp_vec[..sl.min(amp_vec.len())]);
            let vv: Vec<i16> = (0..sl).map(|k| gen_mask[k]).collect();
            let unv_amps: Vec<i16> = amp56[..sl].to_vec();
            let unv_st = UnvoicedBandParams {
                mode: 1,
                l: sl as i16,
                coef: som16,
                amps: &unv_amps,
                voicing: &vv,
                base_exp: amp_bexp as u16,
            };
            let mut out_acc = vec![0i32; SITE_SAMPLE_COUNT + 16];
            synth_unvoiced_subframe(
                &mut out_acc,
                SITE_SAMPLE_COUNT as i32,
                &mut st.unv_state,
                &unv_st,
                gain_a,
            );
            st.ola.resid = [0i16; 56];
            st.ola.resid[..sl.min(resid.len())].copy_from_slice(&resid[..sl.min(resid.len())]);
            let maskv: Vec<i16> = gen_mask.to_vec();
            let p4 = FrameParams {
                hi: sl as i32,
                mask_word: smask,
                l: som,
                e: 0, // IMBE: no AMBE L=56 mixed-band pitch scalar.
                amp: amp56,
                mask: &maskv,
                base_exp: amp_bexp,
            };
            let p5 = FrameParams2 {
                mask_word: st.prev_vmask,
                l5: st.prev_omega0,
            };
            voiced_ola_body(&mut out_acc, SITE_SAMPLE_COUNT as i32, &mut st.ola, &p4, &p5, gain_a);
            let inp: Vec<i32> = out_acc[..SITE_SAMPLE_COUNT].to_vec();
            let mut ofc = out_acc.clone();
            biquad_postfilter(&inp, &mut ofc, &mut st.fca, SITE_SAMPLE_COUNT as i16);
            let base = site * SITE_SAMPLE_COUNT;
            for i in 0..SITE_SAMPLE_COUNT {
                out[base + i] = round_pack16(block_denorm(ofc[i], 2));
            }
            st.prev_vmask = smask;
            st.prev_omega0 = som;
        }
        let st = self.fixed.as_mut().unwrap();
        st.prev_sub_om = omega0_i16;
        st.prev_sub_ml = cur_ml56;
        st.prev_sub_mask = maskword;
        out
    }

    /// Decode one 18-byte IMBE frame to 160 PCM samples via the shared
    /// synthesis. Muted frames (per §7.7) synthesize uniform noise instead.
    pub fn decode_pcm(&mut self, bytes: &[u8; imbe::FRAME_BYTES]) -> [i16; synth::FRAME_SAMPLES] {
        self.decode_pcm_concealed(bytes).0
    }

    /// [`Self::decode_pcm`] plus the concealment telemetry the framing layer
    /// needs: the [`FrameDisposition`] this frame took and its (ε₀, εₜ) Golay
    /// error counts.
    ///
    /// The PCM is byte-identical to [`Self::decode_pcm`] — the §7.6/§7.7 ladder
    /// runs either way, and this call only reports what it decided. A frame
    /// whose pitch index is outside the valid range (`b0 > 207`) is an erasure
    /// marker: it repeats the previous good frame, and 4+ consecutive ones (or
    /// a smoothed error rate above 0.0875) escalate to comfort-noise mute.
    ///
    /// IMBE has no counterpart to the half-rate silence escape, so this never
    /// returns [`FrameDisposition::Silence`].
    pub fn decode_pcm_concealed(
        &mut self,
        bytes: &[u8; imbe::FRAME_BYTES],
    ) -> ([i16; synth::FRAME_SAMPLES], FrameDisposition, u8, u8) {
        let params = self.decode_params(bytes);
        let (epsilon_0, epsilon_t) = self.deq.last_errors();
        if self.muted() {
            return (
                self.mute_noise_frame(),
                FrameDisposition::Mute,
                epsilon_0,
                epsilon_t,
            );
        }
        let disposition = if self.deq.last_was_repeat() {
            FrameDisposition::Repeat
        } else {
            FrameDisposition::Use
        };
        // Route the frame through the shared minimum-phase voiced OLA
        // synthesis when `ola_synth` is set; otherwise the f32 path.
        if self.ola_synth {
            let pcm = match self.deq.ola_params().cloned() {
                Some(ola) => self.decode_imbe_ola(&ola),
                None => [0i16; synth::FRAME_SAMPLES],
            };
            return (pcm, disposition, epsilon_0, epsilon_t);
        }
        let pcm = match params {
            Some(params) => synth::synthesize_frame(&params, &mut self.syn),
            None => [0i16; synth::FRAME_SAMPLES],
        };
        (pcm, disposition, epsilon_0, epsilon_t)
    }
}

impl Default for ImbeDecoder {
    fn default() -> Self {
        Self::new()
    }
}
