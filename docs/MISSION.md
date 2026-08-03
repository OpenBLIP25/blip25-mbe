# blip25-mbe — Mission

## Goal
An **open-source dispatch console** for P25: encode and decode that are, by ear,
**indistinguishable from the reference codec that ships in professional consoles**
(the reference codec that runs in a professional dispatch console). 

## The reference
The correctness target is the **output of the reference codec as it runs in a
professional console** — not the TIA-102 spec text, and not any open-source
reimplementation. `blip25-codec` is this project's own engine; it is tuned so
that its output matches what the console codec produces on the same input, for
all four deliverables below. Where a fidelity metric and the console disagree,
the console is right by definition, because the console *is* the target.

The **Motorola AXS console** is the concrete instance of that target: it runs
the reference codec public-safety dispatch actually uses, which is why its output —
not a spec, and not the opaque AMBE-3000R chip — is the ruler.

Rationale and design goal are documented in
[`WHY_THE_REFERENCE_CODEC.md`](WHY_THE_REFERENCE_CODEC.md); patent and IP scope in
[`../PATENT_NOTICE.md`](../PATENT_NOTICE.md).

> Note on vectors: the reference SDK P25 vectors (`tv-std/tv/r33`) validate the
> decoder and supply realistic input audio, but they are **chip-produced**
> (AMBE-3000R, C55x fixed point) and so are not bit-exact reachable by x86 code.
> The console codec — the x86 Wave7k softclient — is the authoritative target and
> the only one against which bit-exactness is a coherent goal. `tv-rc` is a
> rate-conversion set and is not an encode reference for either codec. See
> [`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md) §1.

## The four deliverables (all essential)
| # | Codec | Direction | State |
|---|-------|-----------|-------|
| 1 | AMBE+2 (P25 Phase 2, rate 33) | **decode** | ships — ear-indistinguishable |
| 2 | AMBE+2 (P25 Phase 2, rate 33) | **encode** | ships — output bits frozen; the console A/B that set them is retired |
| 3 | IMBE full-rate (P25 Phase 1)  | **decode** | ships — the shared OLA back-end |
| 4 | IMBE full-rate (P25 Phase 1)  | **encode** | ships — the same analysis chain |

## Real-time invariant
The codec is **live, not batch**. A caller pushes 20 ms of PCM and gets that
frame's bits back; it never has to hold the whole utterance first. The target is
the oracle's own geometry — one 20 ms frame in, one frame out, with a one-frame
pipeline delay (three-frame batches for IMBE), drained by pushing silence because
there is no end-of-stream call. See [`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md)
§1.

This is a product constraint, not a preference. The codec sits behind a PTT
key-up: latency is added twice, once before the first audio leaves and again
before the operator can release the key. Hundreds of milliseconds are not
available. Whole-buffer helpers may exist for offline callers, but any analysis
that *requires* look-ahead beyond about one frame is out of scope — and, per
§1, is evidence the algorithm has not been recovered yet rather than a cost
worth paying.

## Interoperability invariant
The codec is a plain "bits in / bits out" function. Bits we encode must decode
correctly on **any** conformant AMBE+2 / IMBE decoder, and bits from any
conformant encoder must decode on ours. No special paired encoder/decoder, no
private side channel.
