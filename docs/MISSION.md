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

> Note on vectors: the reference SDK P25 vectors (`tv-std/tv/r33`) are a
> useful **proxy** and are what the decoder is validated against, but they come
> from a *different* reference product than the console codec and are not bit-identical to it.
> The console codec's output is the authoritative target. `tv-rc` is a
> rate-conversion set and is not a reference for either direction.

## The four deliverables (all essential)
| # | Codec | Direction | State |
|---|-------|-----------|-------|
| 1 | AMBE+2 (P25 Phase 2, rate 33) | **decode** | ships — ear-indistinguishable |
| 2 | AMBE+2 (P25 Phase 2, rate 33) | **encode** | ships — output bits frozen; the console A/B that set them is retired |
| 3 | IMBE full-rate (P25 Phase 1)  | **decode** | ships — the shared OLA back-end |
| 4 | IMBE full-rate (P25 Phase 1)  | **encode** | ships — the same analysis chain |

## Interoperability invariant
The codec is a plain "bits in / bits out" function. Bits we encode must decode
correctly on **any** conformant AMBE+2 / IMBE decoder, and bits from any
conformant encoder must decode on ours. No special paired encoder/decoder, no
private side channel.
