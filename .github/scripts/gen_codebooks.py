#!/usr/bin/env python3
"""Regenerate `crates/blip25-codebooks/src/generated.rs` from the raw `.bin`
extracts in `reference-material/codebooks/`.

**Maintainer tool. You do not need to run this to build the project.**

`reference-material/` is gitignored, so the `.bin` files are not in version control — the
*generated Rust source is*, and that is what compiles. A fresh clone builds
without ever seeing a `.bin`. This script exists to regenerate that source if a
table is ever re-extracted.

Emitting Rust `static [i16; N]` rather than `include_bytes!` + a runtime parse
buys three things: no heap allocation or parsing at run time, array sizes
checked by the compiler instead of a runtime `assert_eq!`, and — most
importantly for data recovered from a binary — values that are greppable,
diffable, and reviewable, because review is the only verification these have.

Usage:  python3 .github/scripts/gen_codebooks.py
"""

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = ROOT / "reference-material" / "codebooks"
OUT = ROOT / "crates" / "blip25-codebooks" / "src" / "generated.rs"

# (rust_const, filename, rows, cols) — cols=1 means a flat table.
# Only the tables that are actually used. MONO65 and HOC_SIGMA were dropped:
# both were unreferenced, and the live IMBE equivalents are spec-derived inline
# arrays in `blip25-codec/src/imbe/tables.rs`.
TABLES = [
    ("PRBA24_RAW", "PRBA24_q11.bin", 512, 3),
    ("PRBA58_RAW", "PRBA58_q11.bin", 128, 4),
    ("HOC_B5_RAW", "HOC_b5_q11.bin", 32, 4),
    ("HOC_B6_RAW", "HOC_b6_q11.bin", 16, 4),
    ("HOC_B7_RAW", "HOC_b7_q11.bin", 16, 4),
    ("HOC_B8_RAW", "HOC_b8_q11.bin", 8, 4),
    ("GAIN_O_RAW", "gain_O_q11.bin", 32, 1),
]

# Stamped verbatim at the top of OUT. Must stay byte-identical to the header
# already committed there, so a regeneration is a no-op on those lines.
HEADER = """// AMBE+2 vector-quantizer codebooks recovered from the reference firmware. Do not edit
// by hand.
//
// This committed source is what compiles. The firmware extract `.bin` files it
// derives from are gitignored and not redistributed, so a fresh
// clone has none of them and needs none. `.github/scripts/gen_codebooks.py`
// regenerates this file from those extracts on a machine that has them; it is a
// maintainer tool, not a build step.
//
// Storage convention: int16, row-major, Q11 unless noted. Real value =
// raw / 2048.0. Values are read verbatim for bit-exactness; do not "clean
// them up" — several tables use a scale of ~2045 rather than 2048 and truncate
// toward zero, and reconstructing them from a formula will not reproduce the
// reference vocoder.

"""


def load(path: Path, count: int) -> list[int]:
    raw = path.read_bytes()
    if len(raw) != count * 2:
        sys.exit(f"{path.name}: expected {count * 2} bytes, found {len(raw)}")
    return list(struct.unpack(f"<{count}h", raw))


def fmt(values: list[int], cols: int, indent: str = "    ") -> str:
    if cols == 1:
        # Flat table: wrap at a readable width.
        out, line = [], indent
        for v in values:
            piece = f"{v}, "
            if len(line) + len(piece) > 96:
                out.append(line.rstrip())
                line = indent
            line += piece
        if line.strip():
            out.append(line.rstrip())
        return "\n".join(out)
    # Row-major: one codebook row per line, so a diff points at a row index.
    rows = [values[i:i + cols] for i in range(0, len(values), cols)]
    width = max((len(str(v)) for v in values), default=1)
    return "\n".join(
        indent + ", ".join(f"{v:>{width}}" for v in row) + ","
        for row in rows
    )


def main() -> int:
    if not BIN_DIR.is_dir():
        sys.exit(
            f"{BIN_DIR} not found.\n"
            "This is a maintainer tool — the generated source is already "
            "committed, so you do not need it to build."
        )

    parts = [HEADER]
    for const, fname, rows, cols in TABLES:
        n = rows * cols
        values = load(BIN_DIR / fname, n)
        shape = f"{rows}x{cols}" if cols > 1 else f"{rows}"
        parts.append(
            f"/// `{fname}` — {shape}, Q11. {n} int16 values.\n"
            f"pub(crate) static {const}: [i16; {n}] = [\n"
            f"{fmt(values, cols)}\n];\n"
        )
        print(f"  {const:<12} {n:>5} values from {fname}")

    OUT.write_text("\n".join(parts), encoding="utf8")
    print(f"\nwrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
