# IMBE bit-exact campaign — log

Branch: `wip/imbe-bit-exact` (off main 83a4ad7)
Reference: reference x86 DLL IMBE (same-arch => bit-exact reachable).
Reference bits: `<reference_root>/imbe/{clean,dam,noisy,mark}.bits`
Sources: same dir, `<vec>.pcm` (raw i16 8k). Reference `.bit` vectors are for a different CPU — NOT used.

Harness: `crates/blip25-mbe/examples/imbe_field_compare.rs`
Run: `cargo run --release --example imbe_field_compare -- <src.pcm> <reference.bits>`

---

## 1. FIELD MAP — 88 prioritized info bits -> b-vector

Authorities: `blip25-codec/src/imbe/frame.rs` (u-word widths), `dequantize.rs`
`decode_frame_vector` (deprioritize), `quantize.rs` `encode_frame_vector` (inverse),
and the generated per-L table `blip25-mbe/src/generated/imbe_bit_priority.rs`
(exposed via `imbe7200::priority::{prioritize,deprioritize}`).

### 88 bits = u0..u7, MSB-first
`U_WIDTHS = [12,12,12,12,11,11,11,7]` (sum 88). The 11-byte info-only frame packs
u0..u7 MSB-first. (The 7200 frame adds FEC: c0..c3 [23,12] Golay, c4..c6 [15,11]
Hamming, c7 7 uncoded, Annex-H interleave + u0-seeded PN — stripped in 4400x4400.)

### b-vector `b[0..=L+2]` (L = num_harms)
| field | width | source in u-words | meaning |
|---|---|---|---|
| **b0 pitch** | 8 | `((u0>>4)&0xFC) \| ((u7>>1)&0x3)` — 6 MSB from u0[11:6], 2 LSB from u7[2:1] | pitch index 0..207; FIXED position, L-independent |
| **b1 V/UV** | K | stream bs[39 .. 39+K], MSB-first | per-band voiced/unvoiced, K bands |
| **b2 gain** | 6 | `(u0 & 0x38)` (bits5:3) \| `(bs[39+K]<<2 \| bs[39+K+1]<<1)` (2 stream) \| `((u7>>3)&1)` (bit0) | log-gain index (MONO65 DC + step) |
| **b3 .. b_{L+1}** | var (0..~4 each) | priority-rescan of the compacted stream, MSB-first by descending bit-threshold; widths from `get_bit_allocation(L)` | L-1 spectral-amplitude quantizer indices |
| **b_{L+2}** | 1 | `u7 & 1` | sync bit (encoder sets 0) |

### Priority bit-stream bs[0..75]  (75 = 3 + 3·12 + 3·11 + 3)
- bs[0..3]   = u0 bits 2,1,0
- bs[3..39]  = u1,u2,u3 (12 bits each, LSB-first fill)
- bs[39..72] = u4,u5,u6 (11 bits each, LSB-first fill)
- bs[72..75] = u7 bits 6,5,4
b1 (K bits) + b2's 2 stream bits sit at bs[39..], then the tail is compacted down
by (K+2) and priority-rescanned into b3..b_{L+1}.

### L-from-b0 dependence (the AMBE2 analogue)
`b0 -> fund_freq (ω0) -> L`:
- `tmp = 2*(b0 + 39.5)` (Q15.1); `fund_freq ≈ 1/tmp` (ω0/π, Q1.31)
- `L = num_harms = floor(0.9254 · (b0+40.5)/4)`  (`pitch_cell`, fixed-point exact)
- valid range b0∈[0,207], L∈[9,56]; out-of-range => frame repeat
- `K = num_bands = (L<=36) ? floor((L+2)·0.3333) : 12`  (K∈[1,12])
L drives: #amplitude coeffs (L-1), their per-coeff bit allocation, K (b1 width),
and therefore the whole placement of b1/b2/amps in the 75-bit stream. Higher b0 =>
lower pitch => larger L. Exactly like AMBE2's b0-driven variable layout.

Harness deprioritize: extract b0 by the fixed formula (L-independent), compute L via
`imbe::pitch_cell(b0)`, then `imbe7200::priority::deprioritize(&u, L)` -> b[0..=L+2].
Applied identically to ours and reference (internally consistent b-space).

---

## 2. ALIGNMENT (IMBE encoder delay)

Best frame offset = **-1 on all four vectors** (ours[k] <-> reference[k-1]): our encoder
emits one frame later than reference's indexing, consistent with the one-frame analysis
look-ahead (encode() drops the frame-0 placeholder + flushes the tail). Frame counts
match (clean/mark exact; dam/noisy off-by-1). Confidence: MODERATE — chosen on a weak
b0 signal (winning-offset b0 only 20-27%), but -1 beats all neighbours on 4 independent
vectors and 26% >> the ~1-2% an 8-bit exact match gives at a wrong offset. Reconfirm
once pitch improves.

---

## 3. BASELINE — per-field % agreement vs reference x86 reference

| field | clean | dam | mark | noisy |
|---|---|---|---|---|
| b0 pitch (exact) | 26.8 | 26.5 | 26.1 | 20.1 |
| L harms match | 39.9 | 43.9 | 53.8 | 31.3 |
| b1 vuv | 55.5 | 58.9 | 53.6 | 39.4 |
| b2 gain | 20.0 | 9.5 | 14.6 | 17.6 |
| amps aggregate | 35.1 | 40.1 | 60.2 | 28.3 |
| **ALL (b0..b_{L+2})** | **0.00** | **0.00** | **0.00** | **0.00** |

### Conditional on L matching (isolates IMBE-wire from the pitch root)
| field \| L-match | clean | dam | mark | noisy |
|---|---|---|---|---|
| b1 vuv \| L | 26.3 | 35.2 | 49.9 | 11.0 |
| b2 gain \| L | 18.6 | 15.4 | 17.6 | 11.6 |
| amps \| L | 55.2 | 59.8 | 72.6 | 52.9 |

### b0 closeness (exact / ±1 / ±2 / ±4 / gross≥8)
- clean: 26.8 / 42.7 / 43.3 / 43.6 / 56.1
- dam:   26.5 / 46.7 / 47.2 / 47.3 / 52.2
- mark:  26.1 / 57.8 / 62.9 / 63.3 / 36.3
- noisy: 20.1 / 33.1 / 33.7 / 34.3 / 65.1

### b0 range/mean (ours vs reference)
- clean: ours[10..207] mean 111.1  |  reference[10..207] mean 57.2
- mark:  ours[46..207] mean 154.7  |  reference[25..207] mean 124.0
- noisy: ours[2..207]  mean 108.1  |  reference[2..207]  mean 48.0
- dam:   ours[19..207] mean 110.0  |  reference[19..207] mean 65.7
Both sides use the FULL 8-bit range => NOT a 7-bit/8-bit convention error. But our
mean b0 is systematically HIGH (=> lower pitch / larger L), worst on low-energy/
unvoiced frames (clean/noisy gap ~55; mark, mostly voiced, gap ~30).

---

## 4. SHARED FRONT-END READ

IMBE does NOT inherit AMBE2's ~90% pitch/voicing parity — it starts far lower
(b0 26%, b1 55%, gain 15%). The shared analysis core DOES contribute, but unevenly:

- **Amplitudes (b3..)** — the clear shared-core beneficiary: 53-73% once L matches
  (vs 28-60% raw). Amplitude wire + analysis are mostly right; their raw deficit is
  a downstream victim of wrong pitch/L, not IMBE-wire breakage.
- **Pitch (b0)** — partial: the contour tracks (±2 reaches 43-63% on cleaner audio)
  so the shared estimate is in the ballpark, but exact b0 is far AND systematically
  biased high. The finer 8-bit IMBE grid explains ~2×, not 90%->26%. IMBE-specific:
  either the ω0->b0 cell quantization (`imbe/quantize.rs quantize_pitch`/`pitch_cell`)
  or the forced reference-b0 mapping into the IMBE cell diverges from reference.
- **Gain (b2)** — NOT inherited: 11-19% even conditioned on L. Deepest wire field,
  independent of pitch. Mirrors the AMBE2 finding that gain (b2) was THE gap.
- **Voicing (b1)** — NOT inherited: 11-50% conditioned on L. Real V/UV decision
  divergence (or K-band mapping), not a shared-core win.

Opposite of the AMBE2 starting point (b0/b1 already ~90%). Here only amps behave like
a shared-core beneficiary.

---

## 5. RANKED FIRST TARGETS (by deficit + leverage)

1. **b0 pitch / L (ROOT).** 20-27% exact, systematic high-mean bias. Highest leverage:
   ALL-match is pinned at 0% purely because b0/L is wrong; conditional numbers show
   amps jump to 53-73% the moment L is right. Two sub-leads: (a) `quantize_pitch`/
   `pitch_cell` ω0->b0 cell quantization; (b) the forced reference-b0 (`encode_pcm_b0` +
   `set_forced_b0`) mapping into the IMBE 8-bit cell. Attack first.
2. **b2 gain.** 9-20% raw, still 11-19% conditioned on L => decoupled from pitch, can
   be attacked in parallel. Known-hard field (AMBE2's THE-gap). Look at
   `sa_encode` gain path (`tbl_quant`/`MONO65_ENCODE` DC + `qnt_by_step` steps) and
   the log2/DCT gain chain vs reference.
3. **b1 vuv.** 39-59% raw, 11-50% conditioned. Voicing-decision / K-band divergence in
   the shared analysis + `quantize_to_u`'s per-band bit build.
4. **amps (b3..).** Lowest priority — already 53-73% once L matches; will largely fall
   out of fixing b0/L and gain. Revisit last for residual per-coeff step/DCT deltas.

---

## Notes / open items
- Alignment (-1) rests on a weak b0 signal; reconfirm after pitch work.
- Harness compares ours-vs-reference in a shared b-space via one deprioritize; if L differs
  the amplitude-index comparison only spans the L-overlap (honest, but amp% is only
  meaningful once L match rises).
