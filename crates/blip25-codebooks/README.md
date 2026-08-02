# blip25-codebooks — half-rate P25 vocoder codebooks

Bit-exact vector-quantizer codebooks for the half-rate P25 vocoder (AMBE+2,
Phase 2), with a typed loader. Tables are `static [i16; N]` compiled into the
binary, so accessors borrow rather than allocate — free to call per frame.

> **Depend on [`blip25-mbe`](https://crates.io/crates/blip25-mbe), not on this
> crate.** This is an implementation detail of the
> [`blip25-codec`](https://crates.io/crates/blip25-codec) engine, published
> only so `blip25-mbe` can be published. It carries no API stability promise
> and its contents are meaningless without the engine that indexes them.

## What is in here

| Table | Shape | Use |
|---|---|---|
| `prba24` | 512×3 | PRBA sub-vector VQ, indexed by b3 (9-bit) |
| `prba58` | 128×4 | PRBA sub-vector VQ, indexed by b4 (7-bit) |
| `hoc_b5` | 32×4 | Higher-order-coefficient VQ |
| `hoc_b6` | 16×4 | Higher-order-coefficient VQ |
| `hoc_b7` | 16×4 | Higher-order-coefficient VQ |
| `hoc_b8` | 8×4 | Higher-order-coefficient VQ |
| `gain_o` | 32 | Differential gain quantizer (Annex-O), indexed by b2 |

All Q11 — real value = `raw / 2048.0`. **Read the raw `i16` verbatim for
bit-exactness:** several tables were built at scale ~2045 rather than 2048 and
truncate toward zero rather than rounding, so reconstructing them from a
formula will not reproduce the reference vocoder.

Full-rate IMBE quantizer tables are *not* here. Those are spec-derived and live
as inline arrays in `blip25-codec`'s `imbe::tables`.

## Provenance

Every table in this crate is **firmware-recovered**, not published spec data —
which is why it needs its own crate. The values were extracted from a reference
vocoder image and verified bit-exact against that vocoder's own output.

The raw extract `.bin` files are **not** in version
control. `src/generated.rs` is the committed, reviewable form and is what
compiles, so a fresh clone builds without them. Regenerate with
`.github/scripts/gen_codebooks.py` if a `.bin` is ever available again.

This crate is part of a patent-encumbered codec. See `PATENT_NOTICE.md` and
`ATTRIBUTION.md`, both shipped in this package.

## License

MIT — see `LICENSE`. The MIT grant covers this source; it does not grant
patent rights. Read the patent notice.
