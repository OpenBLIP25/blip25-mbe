# conformance/scripts

External-tool oracles used by the blip25-mbe conformance suite.
Deliberately not Rust crates: each script depends on a third-party
library that's GPL/restrictive (JMBE under GPLv3) or
ecosystem-specific (Python `pesq` / `pystoi` from `pip`). Keeping
them at the scripts layer lets the codec library itself stay
dep-free.

## Tools

### `score_ab_matrix.py`

PESQ-NB + STOI scoring of `(reference, candidate)` WAV pairs in the
layout produced by `conformance-speech-quality ab-matrix`. Runs in
the `~/.blip25-eval/` venv with `pesq` + `pystoi` installed.

```bash
~/.blip25-eval/bin/python3 conformance/scripts/score_ab_matrix.py \
    /tmp/ab_clean
```

Used by every PESQ-related memory entry under
`~/.claude/projects/-home-chance-blip25-mbe/memory/`.

### `JMBEDecode.java`

Black-box IMBE decoder driver. Decodes a stream of 18-byte P25 Phase 1
FEC frames into 8 kHz mono WAV using JMBE's `IAudioCodec` interface.
Used as the third-party reference channel in
`our_enc → JMBE → PESQ` measurements (see memory
`project_jmbe_validates_chip_bit_2026-04-24` for context).

Build:
```bash
mkdir -p /tmp/jmbe_iface_classes
javac -d /tmp/jmbe_iface_classes ~/jmbe/api/src/main/java/jmbe/iface/*.java
javac -cp ~/jmbe-1.0.9.jar:/tmp/jmbe_iface_classes \
      -d conformance/scripts \
      conformance/scripts/JMBEDecode.java
```

Run (full classpath includes JMBE runtime deps from the JMBE creator
distribution):
```bash
JMBE_LIB=/tmp/jmbe_creator/creator-linux-x86_64-v1.0.9/lib
java -cp ~/jmbe-1.0.9.jar:/tmp/jmbe_iface_classes:\
$JMBE_LIB/slf4j-api-1.7.25.jar:$JMBE_LIB/slf4j-simple-1.7.25.jar:\
$JMBE_LIB/JTransforms-3.1.jar:$JMBE_LIB/JLargeArrays-1.6.jar:\
$JMBE_LIB/commons-math3-3.5.jar:conformance/scripts \
JMBEDecode in.bit out.wav
```

### `JMBEDumpInfo.java`

A/B probe added during the gap 0021 investigation. Decodes a single
chip.bit frame through JMBE up to (but not including) the model-
parameter dequantize stage and prints the post-FEC `u₀..u₇` info
vectors plus per-codeword error counts. Used to confirm
bit-equivalence between blip25-mbe's spec-derived FEC and JMBE's
implementation — same compile + classpath as `JMBEDecode`.

```bash
java -cp <CP> JMBEDumpInfo in.bit <frame_index>
```

## Why this is here, not a sibling repo

These are cross-codec oracles, not codec primitives. They validate
blip25-mbe's outputs against external references; they don't
implement any codec functionality.

The directory currently has three files (~250 LOC total) and one
build dependency tree. If it grows beyond ~10 tools or starts
needing its own tracked classpath / venv lockfile, split into a
sibling `blip25-mbe-tools` repo at that point. Today: smaller blast
radius to keep here.

## License note

JMBE is GPLv3, Python `pesq` is BSD-3, `pystoi` is MIT. blip25-mbe
itself is MIT (per the workspace `Cargo.toml`). Wrapping JMBE in a
runtime-only black-box driver (no source copied, no derived work) is
clean — these scripts don't ship as part of the crate. Downstream
consumers of `blip25-mbe` are unaffected.
