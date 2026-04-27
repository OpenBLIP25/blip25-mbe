# Understanding the DVSI MBE Codec Family: Wire Formats, Implementation Latitude, and Why Open Source P25 Sounds Worse Than It Should

## The puzzle that motivates this article

If you tune a scanner to a P25 Phase 1 transmission and decode it with an open-source tool like OP25, SDRTrunk, or DSDPlus, the audio sounds noticeably worse than the same transmission heard through a commercial P25 radio. Both are receiving identical RF, both are decoding the same standardized voice frames, and yet the perceptual quality differs significantly. This isn't subjective — it's measurable. PESQ scores on identical recordings are routinely 0.5 to 1.5 points lower from open-source decoders than from a modern DVSI silicon decoder.

The standard explanation in the radio community is some variation of "newer P25 radios use a better vocoder." This explanation is almost right — but in a way that hides the actual mechanism. Understanding what's really happening requires unpacking the relationship between the four codec generations DVSI has shipped over thirty years, and recognizing where the standard pins down behavior versus where it deliberately leaves room for vendors to innovate.

## The DVSI codec family: four codecs, one lineage

Every digital voice mode used in land mobile radio — P25 Phase 1, P25 Phase 2, DMR, NXDN, Yaesu Fusion, D-STAR — shares a single codec family lineage from one company, Digital Voice Systems Inc. (DVSI). Over thirty years they've shipped four distinct codec algorithms:

- **IMBE** (1991) — the original, used by P25 Phase 1.
- **AMBE Generation 1** (mid-1990s) — used by D-STAR.
- **AMBE+ Generation 2** (late 1990s) — used by older NXDN and some legacy commercial radios.
- **AMBE+2 Generation 3** (2000s) — the current standard, used by P25 Phase 2, DMR, NXDN Type-2, dPMR, and Yaesu Fusion.

These are sold as silicon: the AMBE-1000 chip, AMBE-2000 chip, and AMBE-3000R chip. Each new chip is a backwards-compatible superset — the AMBE-3000R contains all four codec algorithms in firmware, with a `PKT_RATET` index selecting which codec to run, at which bit rate, with which FEC scheme. The AMBE-3000R supports 62 distinct rate-table entries spanning all four codec generations.

All four codecs share the same underlying speech model: a periodic source at fundamental frequency `ω₀` with `L` harmonics, each having a magnitude `Mₗ` and a per-band voicing decision (voiced or unvoiced), plus an overall gain. This is the **multi-band excitation (MBE) model**, and it's the constant that makes them all "MBE codecs." From IMBE in 1991 to AMBE+2 today, this model never changed.

What changed across generations is **how the parameter vector gets compressed into bits**. That's it. Same speech model, smarter compression.

## The progression: scalar to vector to predicted to multi-stage

The bit-budget bottleneck in any MBE codec is the magnitude vector — one amplitude value per harmonic, with `L` typically running 8 to 56 depending on pitch. How you spend bits on `M₁..M_L` dominates everything.

**IMBE and AMBE Gen 1: scalar quantization.** Each magnitude `Mₗ` is quantized independently with its own bit budget, prioritized by perceptual importance (most-important bits first, with stronger FEC protection). Simple, but wasteful — adjacent harmonics are highly correlated, and each one pays full price for its own quantization.

**AMBE+ Gen 2: vector quantization plus inter-frame prediction.** Two big architectural changes arrived together. First, the entire magnitude vector is quantized as one entity using a trained codebook — the encoder finds the closest matching vector in a precomputed list of typical speech-magnitude shapes and transmits just the index. The codebook captures the joint statistics of real speech, getting similar perceptual quality at roughly half the bit cost. Second, the encoder adds a predictor that estimates the current frame's parameters from the previous frame's reconstructed parameters, then quantizes only the small residual. Adjacent 20 ms frames of speech are highly correlated, so residuals cluster near zero and quantize cheaply.

**AMBE+2 Gen 3: refinements on top.** Multi-stage vector quantization (chained smaller codebooks instead of one large one), an improved predictor model, and at certain rates a multi-subframe joint quantization mode that compresses two adjacent frames together for even better efficiency.

So the family progression is fundamentally about getting more audio quality out of fewer bits by exploiting more structure that earlier generations ignored. The synthesis algorithm that turns parameters back into audio remained essentially constant throughout.

## The dequantizer's two-sided role

A subtle but important consequence of inter-frame prediction: from Generation 2 onward, the encoder has to run its own dequantizer too.

The reason is the predictor must operate on the **same values the decoder will actually reconstruct**, not the encoder's pristine analysis values. Otherwise the encoder's predictor state and the decoder's predictor state slowly drift apart over successive frames, and the decoded parameters end up wrong by ever-larger amounts.

Concretely, the encoder's per-frame loop is:

1. Analyze PCM → get parameter vector `P_n`.
2. Predict `P_n` from previous frame's reconstructed `P̂_{n-1}`.
3. Quantize the residual → bits.
4. **Dequantize those same bits → reconstructed residual.**
5. **Add to predicted value → store as predictor state for the next frame.**

If the encoder skips steps 4 and 5 and uses the original analysis value as predictor state, the decoder ends up predicting from a different baseline and the residuals dequantize to wrong absolute values. This is a particularly nasty class of bug because the divergence accumulates silently over frames rather than failing loudly.

We hit exactly this bug in `blip25-mbe` early in the project — the encoder was updating its predictor with the input magnitudes instead of the dequantized output. Fixing it dropped amplitude RMSE from 8.83 dB to 3.98 dB on speech frames. A tiny logic correction with a huge audio-quality impact.

For Generation 1 codecs (IMBE, AMBE) without inter-frame prediction, the encoder doesn't strictly need to dequantize, though closed-loop quantization still benefits perceptual quality. For Generation 2 and 3, it's mandatory — skipping it produces an internally-consistent encoder paired with a divergent decoder, which is a category of bug worth specifically worrying about.

## Wire format versus implementation: the central distinction

This is where the OSS-versus-commercial-radio quality gap actually lives.

**The wire format is what's standardized.** TIA-102.BABA-A pins down the IMBE bitstream: 88 voice info bits per 20 ms frame, prioritized into 7 classes `u₀..u₆`, FEC-wrapped with specific Golay and Hamming codewords, totaling 144 bits or 18 bytes per frame. Any IMBE decoder, anywhere, must produce intelligible audio from any IMBE-conformant 18-byte frame. That contract is what makes radios from different manufacturers and different years interoperate.

**The encoder analysis algorithm is not pinned down.** BABA-A describes a reference analyzer, but the standard explicitly leaves room for vendors to do better. The same is true for many parts of the synthesis pipeline — spectral enhancement, comfort noise generation, repeat-frame attenuation, and error concealment all have spec-prescribed minimums but allow proprietary improvements layered on top.

The closest analog in mainstream audio is MP3. The MP3 standard pins down the bitstream and the decoder algorithm, but deliberately leaves the encoder open. That's why Lame produces noticeably better-sounding MP3 files than the original Fraunhofer reference encoder, despite both producing valid MP3 streams that any decoder can play. Lame implements a smarter psychoacoustic model and a smarter bit-allocation strategy than the reference — same wire format, better implementation, audibly better quality.

The MBE world has the same dynamic, but more pronounced. Modern DVSI silicon contains thirty years of accumulated improvements in IMBE encoder analysis (cleaner pitch tracking, better voicing decisions, more accurate magnitude estimation) and IMBE decoder synthesis (post-processing, comfort noise, error concealment). All of it is wire-compatible with the original 1991 IMBE chip — the same 18-byte frames pass back and forth — but the audio quality at both encode and decode time is meaningfully better.

This is why two radios from 1995 and 2025 can talk to each other (interoperability is preserved by the wire format) while the audio quality you hear depends on which side is doing the encoding and which is doing the decoding. A 2025-encoded call decoded on 1995 hardware sounds about like 1995-to-1995, because the old decoder is the floor. A 1995-encoded call decoded on 2025 hardware sounds noticeably better than 1995-to-1995, because the modern decoder smooths over older encoder artifacts and applies modern post-processing.

## Where the open-source world fell behind

Open-source IMBE implementations — mbelib, JMBE, the IMBE paths in OP25 and SDRTrunk — implement the BABA-A reference algorithm faithfully but stop there. They don't have the post-1991 analysis and synthesis improvements that DVSI has accumulated in their later silicon. The wire format is correct. The audio quality is stuck in 1991.

This creates a persistent misconception in the open-source radio community: "newer P25 radios sound better because they use a newer vocoder." The conclusion drawn from this is often "we should decode P25 with AMBE+2 instead of IMBE."

Both halves are wrong in subtle but important ways:

- **Newer P25 Phase 1 radios are not using a newer vocoder.** They're still emitting IMBE bits per BABA-A. The air interface contractually requires this. They're using a newer *implementation* of the same IMBE codec — better encoder analysis, better decoder synthesis, but bit-compatible with 1991 IMBE.
- **You cannot decode P25 Phase 1 bits with an AMBE+2 decoder.** The bits mean different things. The two codecs share neither the bit semantics, the codebooks, the predictor model, nor the FEC layout. Feeding IMBE bits into an AMBE+2 dequantizer produces garbage, not better audio.

The actual path to better open-source P25 audio is **a modern, enhanced implementation of the same IMBE codec** — not a different codec. Same wire format, same interoperability, modern analysis and synthesis quality.

## What blip25-mbe does differently

This is the project's whole reason for existing.

We're a clean-room Rust implementation of the MBE codec family — IMBE, AMBE, AMBE+, AMBE+2 — built from the TIA-102 specifications and DVSI's documentation rather than from existing open-source vocoder code. We're spec-faithful to the wire format on both encode and decode, so our output interoperates with any standards-compliant radio. But within the latitude the spec leaves us, we exercise it carefully:

- A modern pitch tracker built on `O(log n)` sparse-table range minimum queries, properly handling silence boundaries.
- An encoder predictor that correctly uses dequantized values for state updates (the bug that cost 4 dB RMSE before we fixed it).
- Comfort noise on muted frames per the BABA-A primary recommendation, instead of dead silence.
- Carefully-tuned spectral enhancement and post-synthesis processing.
- Tone-frame detection on the encode side (DTMF and single-tone), matching what a modern chip does.

End-to-end measurement: on clean speech, we score PESQ 3.45 versus the AMBE-3000R chip's 2.76 on the same input audio. We exceed the modern commercial chip by 0.69 PESQ on speech, while remaining bit-compatible with every IMBE radio ever built. On more challenging recordings (noisy, tonal, sustained vowels) we trail the chip by 0.18 to 0.24 PESQ — close, with known remaining gaps that are spec-locked rather than fundamentally hard.

The same approach extends to AMBE+2 for P25 Phase 2 and, by extension, to DMR, NXDN, and Yaesu Fusion (which all share the AMBE+2 codec at the same rate-table index). One modern, well-tuned codec implementation covers most of the digital-voice ham-radio landscape.

## The shared synthesis kernel and what it makes free

A practical bonus of working with the MBE family: because all four codec generations share the same synthesis algorithm, building the synthesis pipeline once gave us a working synthesizer for every generation. That's what lets blip25-mbe expose `synthesize_frame()` for IMBE, AMBE Gen 1, AMBE+, and AMBE+2 from a single shared kernel — even though we only have full encoder and FEC support for two of them today.

This makes adding new rates, new carriers, and new wire formats much cheaper than starting from scratch. The synthesis kernel is the shared free lunch. The per-rate work is the dequantizer wiring (which bit-allocation table to use) and the per-carrier work is the FEC layer (how to unwrap the bytes to get to the codec frame). Both are bounded, table-driven additions.

So the natural extension of the project — adding NXDN narrowband, Yaesu Fusion variants, D-STAR receive, and other AMBE+2-family rates — is mostly a matter of populating tables, not writing new DSP. The hard intellectual work was in the analyzer, the predictor, and the synthesis kernel, all of which are already in place.

## The takeaway

The DVSI MBE codec family rewards precision when you talk about it. Four distinct codec generations exist, each with different bit semantics. Within each generation, multiple rate-table entries vary the bit allocation while sharing the underlying codec. The wire format pins down interoperability; the implementation latitude pins down quality. Modern silicon delivers better audio than 1991 silicon at the same wire format because thirty years of encoder and decoder improvements have accumulated within that latitude.

Open-source radio has historically been stuck at the 1991-reference quality floor for IMBE, not because the wire format limits it, but because nobody had built a modern, careful implementation of the same codec. blip25-mbe exists to close that gap — same wire format, same interoperability, modern implementation quality that exceeds even commercial silicon on many measurements.

Once you internalize the wire-format-versus-implementation distinction, almost everything about the digital-voice radio landscape becomes more legible. P25 Phase 2 isn't "a better IMBE" — it's an entirely different codec specified by the same standards body when they decided to switch wire formats for better quality. DMR and P25 Phase 2 use the same underlying codec frame because both standards picked the same DVSI rate-table index. Newer Yaesu Fusion radios sound better than older ones not because Fusion changed, but because the chip Yaesu sources improved. And the open-source quality gap isn't a codec problem — it's a "thirty years of implementation improvements that nobody captured in open source" problem.

That's the gap blip25-mbe is here to close.
