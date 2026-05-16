# Wire Formats and Storage — What Each Byte Layout Actually Means

This crate ships four `Rate` variants. To a casual reader they look like four
symmetric options — "with or without FEC, full-rate or half-rate." That
symmetry is misleading. Two of the four formats are normative P25 air-interface
formats; two are *conventions* whose exact byte layout depends on whose
implementation you're standing in. This article catalogues which is which and
recommends a storage format that won't trip future maintainers.

## The four formats

| `Rate` variant | Bytes / frame | Bits / frame | What it represents |
|---|---:|---:|---|
| `Imbe7200x4400` | 18 | 144 | **P25 Phase 1 full-rate IMBE with Annex H FEC.** Normative in TIA-102.BABA-A. This is what comes off the air on Phase 1 FDMA voice channels. |
| `Imbe4400x4400` | 11 | 88 | **IMBE info-only.** 88 prioritized info bits packed MSB-first. Not on the air anywhere — purely a storage convention. JMBE uses this layout, OP25 uses this layout, DVSI's `p25_nofec` test vectors use this layout. They all happen to agree. |
| `AmbePlus2_3600x2450` | 9 | 72 | **P25 Phase 2 half-rate AMBE+2 with Golay/Hamming/PN FEC.** Normative in BABA-A. Also the format DMR Tier II/III voice frames take at the vocoder layer (carrier-specific framing aside). DVSI chip rate index 33. |
| `AmbePlus2_2450x2450` | 7 | 49 + 7 pad | **AMBE+2 info-only.** 49 info bits in 7 bytes. **Byte layout is not standardized** — see the next section. |

## The `r34` nuance — read this before assuming it's just "r33 minus FEC"

DVSI's chip exposes a rate index 34: "AMBE+2 half-rate without FEC." Naively
you'd assume the chip's r34 output is the same 49 info bits the r33-with-FEC
frame carries, just stripped of the Golay/Hamming/PN parity. Mostly true —
*but the byte order differs.*

Concretely, for the same input PCM frame, DVSI's chip produces:

```
chip output at rate 33: [ … 9 bytes, FEC-bearing … ]
chip output at rate 34: [ … 7 bytes, info-only … ]
```

Both encode **the same 49 information bits** — verified by Hamming-weight
match plus a strict bit-permutation test across 2850 frames from
`DVSI/Vectors/tv-rc/{r33,r34}/`. Every r34 bit position maps to a unique
u-info bit position via a fixed permutation table; no bit is dropped, added,
or scrambled.

But the *byte layout* of those 49 bits differs:

```
ours       (49-bit info, u₀..u₃ MSB-first):  [98 02 b9 4f a4 d3 80]
DVSI r34   (same 49 bits, permuted):         [cd 4a c3 01 b6 e6 00]
```

The permutation looks like a 3-row interleave — likely a chip-internal
serialization quirk left over from how the AMBE-3000R's bit FIFOs feed the
USB protocol. **It is not specified anywhere in BABA-A or the AMBE-3000
protocol spec.** It's a private DVSI convention.

This crate's `AmbePlus2_2450x2450` packs the 49 bits in the natural
u₀..u₃ MSB-first order, which round-trips losslessly through our own
encoder/decoder pair but **does not byte-equal the chip's r34 output stream**.
If you ever need to feed raw bytes into a DVSI chip configured at rate index
34, you would need a permutation step — but you almost never need that path
(see "Recommended storage" below).

## What "without FEC" means in each codec — they're not symmetric

The IMBE no-FEC layout (`p25_nofec`, 11 bytes) **is** standardized by
convention. JMBE, OP25, SDRTrunk, and DVSI's own `tv-std/tv/p25_nofec/`
vectors all use the same layout: 88 prioritized info bits, MSB-first,
packed into 11 bytes with 8 bits of pad. We round-trip this bit-exactly
against DVSI (100% in `cross-rate-compare`). Anyone storing IMBE
info-only is using this layout, full stop.

The AMBE+2 no-FEC layout has no such consensus. JMBE doesn't decode AMBE+2
info-only (its AMBE module consumes 9-byte FEC frames). OP25 doesn't expose
an AMBE+2 info-only file format. The only "r34 wire format" that exists in
the wild is DVSI's chip-internal permutation, which has no second consumer.

So our two no-FEC variants are structurally different things despite the
parallel naming:
- `Imbe4400x4400`: a real conventional storage format with multi-tool consensus.
- `AmbePlus2_2450x2450`: our private convention, useful for ours-to-ours
  archival but not for interop with anything outside this library.

## Why we didn't pursue byte-exact DVSI r34 compatibility

The DVSI r34 byte permutation isn't documented in BABA-A and isn't a P25
air-interface format. It exists only as a DVSI chip-internal serialization
choice. Replicating it byte-for-byte would require either:

1. Reverse-engineering the permutation via single-bit b̂ probes against the
   AMBE-3000R chip — feasible (~1 hour of probe runs), but the resulting code
   serves exactly one consumer: another DVSI chip configured at rate 34.
2. Finding it in a DVSI document we don't have. None of BABA-A, BABG, or
   the AMBE-3000F manual specify it.

We chose to leave this as a "deferred to chip-shim crate" item. The future
`blip25-chip-shim` project — which emulates the AMBE-3000R serial protocol
for drop-in Blue-DV-style compatibility — is the right place for byte-exact
chip wire-format work. blip25-mbe itself is a codec library, not a chip
emulator; the project memory `Chip-shim is future separate project` codifies
this scope boundary.

## Recommended storage format (read this if you only read one section)

**Store raw FEC-bearing frames as received.** That means:

- P25 Phase 1: 18-byte IMBE frames (`Rate::Imbe7200x4400`), exactly as they
  arrive from the demod-and-deinterleave layer.
- P25 Phase 2: 9-byte AMBE+2 frames (`Rate::AmbePlus2_3600x2450`), exactly
  as they arrive from the burst payload.

Reasons:

1. **Universal interop in the smallest format that survives uncorrectable-frame
   analysis.** Every consumer in the P25 ecosystem speaks 18/9-byte FEC-bearing
   frames: JMBE, SDRTrunk, OP25, our own decoder, the DVSI chip via PKT_CHANP.
   No transformation needed at replay time.
2. **FEC errors stay visible.** If you decode-and-re-encode (the "repeater"
   pattern), the stored stream is always FEC-valid and you can't tell from
   the file whether a frame was clean on receive or rescued by Golay. Storing
   raw bits preserves channel-quality forensics.
3. **No information loss.** Soft-bits preserve more, but at 8× the storage
   cost and zero ecosystem interop. For voice archives that's a bad trade.
4. **Erasures are handled by the decoder, not the storage layer.** BABA-A
   §1.11 defines what happens when FEC is uncorrectable: the codec
   substitutes Mute (first uncorrectable frame), Repeat (run), or Comfort
   Noise (extended run). You don't need to mark erasures in the bits — the
   FEC decoder detects them at replay time.

The info-only variants (`Imbe4400x4400`, `AmbePlus2_2450x2450`) exist for
specialized use cases (compact ours-to-ours archive, JMBE-style export for
the IMBE side) but are not the recommended default. They drop the FEC-error
signal, and the AMBE+2 variant ties you to our private byte layout.

## Feeding stored frames back through the chip

A common workflow is "store FEC-bearing frames, later replay through a real
DVSI chip via the AMBE-3000R USB-3000 board for an A/B oracle." For both
IMBE and AMBE+2:

```
18-byte IMBE frame  →  PKT_CHANP envelope (n_bits=144) →  chip decodes
9-byte r33 frame    →  PKT_CHANP envelope (n_bits=72)  →  chip decodes
```

No padding, no permutation, no transformation. The bytes go in
byte-for-byte; the chip's serial protocol wraps them in a header but does
not modify the payload. This is implemented in
`conformance/chip/python/dvsi_driver.py:make_channel_packet`.

The one historical pitfall worth flagging: feeding info-only bytes
(`p25_nofec` 11-byte or `r34` 7-byte) directly to the chip configured at
rate 33 by zero-padding them to 18/9 bytes does *not* work. The chip's
Golay/Hamming decoders syndrome-check the parity bytes; zeroed parity is
not a valid codeword for arbitrary info bits, so the decoder either fails
or miscorrects to garbage. The correct path is `Transcoder::new(no_fec,
fec)` to re-apply Annex H FEC (IMBE) or Golay/Hamming/PN (AMBE+2) before
feeding the chip.

## Summary table for future maintainers

| If you want to … | Use |
|---|---|
| Store P25 Phase 1 voice for replay through anything | `Rate::Imbe7200x4400` (18 bytes/frame) |
| Store P25 Phase 2 / DMR voice for replay through anything | `Rate::AmbePlus2_3600x2450` (9 bytes/frame) |
| Compact archive, IMBE side, JMBE-compatible | `Rate::Imbe4400x4400` (11 bytes/frame) — matches DVSI / JMBE convention |
| Compact archive, AMBE+2 side, ours-to-ours only | `Rate::AmbePlus2_2450x2450` (7 bytes/frame) — private layout, not byte-equal to DVSI r34 |
| Feed bytes directly into a DVSI chip via PKT_CHANP at rate 34 | Not supported here; see future blip25-chip-shim |

If something downstream surprises you with "I expected r34 to match DVSI's
output and it didn't," the answer is in this article. Don't go looking for
a bug in `ambe_plus2_wire`; the bits are right, the byte order is ours by
design.
