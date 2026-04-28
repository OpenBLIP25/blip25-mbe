# Patent Notice

This source code is provided for **research and educational purposes only**.
It is a written description of how certain voice encoding and decoding
algorithms could be implemented.

Executable objects compiled or derived from this package, and the act of
running them, may be covered by one or more patents. Readers are strongly
advised to check for any patent restrictions or licensing requirements
before compiling, using, or distributing this source code.

The above patent notice is adapted from the verbatim form used by the
[mbelib](https://github.com/szechyjs/mbelib) and
[JMBE](https://github.com/DSheirer/jmbe) projects.

## Specific patents the maintainers are aware of

This list is not exhaustive and is not legal advice. It reflects the
clean-room patent audit at
`~/blip25-specs/DVSI/AMBE-3000/AMBE-3000_Patent_Reference.md`.

### Active patents that the half-rate (AMBE+2) implementation reads on

| Patent | Subject | Anticipated expiration |
|---|---|---|
| **US8359197** | Half-rate vocoder — mixed pitch+voicing+gain first parameter codeword + Golay FEC + scrambling. The parent grant of US8595002 with ~5 years patent term adjustment. | **2028-05-20** |

Per the spec-author's claim-by-claim analysis, **any BABA-A-compatible
half-rate (AMBE+2) decoder or encoder unavoidably reads on US8359197
claims 1, 8, 9, 13, 14, 15 (encode side) and 42, 47–51, 60, 72 (decode
side).** The wire format mandates the mixed first-codeword construction
that the patent claims; bitstream interoperability with the AMBE-3000
chip cannot be achieved with a different construction.

### Active patents that overlap potential future frontend improvements

These patents do **not** read on the current implementation, but they
constrain the design space for future analysis-frontend improvements
(e.g. closing the measured PESQ gaps on noisy / tonal / SCBA-mask
content).

| Patent | Subject | Anticipated expiration |
|---|---|---|
| US8265937 | Breathing-apparatus speech enhancement (fireground / SCBA noise) | ~2032 |
| US12254895 | Detecting and compensating for speaker mask | ~2045 |
| US11990144 | Reducing perceived effects of non-voice data | ~2041 |
| US12451151 | Tone frame detector (PTAB-confirmed) | ~2042 |

### Expired patents that the implementation derives from

The following are now public domain and serve as detailed algorithmic
documentation. Implementing them is unrestricted.

| Patent | Subject | Expired |
|---|---|---|
| US5701390 | MBE synthesis with regenerated phase | 2015-02-22 |
| US6199037 | Joint quantization of voicing and pitch | 2017-12-04 |
| US8315860 | Interoperable vocoder | 2022-11-13 |
| US8595002 | Half-rate vocoder (AMBE+2) — sibling of active US8359197 | 2023-04-01 |
| US7634399 | Voice transcoder | 2025-11-07 |

## Project policy

**This project is research / educational software.** The maintainers do
not distribute commercial product, do not sell licenses, and do not
charge for downloads, hosting, or integration.

**No commercial product is shipped from this codebase before
2028-05-20** (US8359197 anticipated expiration). Downstream consumers
who wish to use this code in a commercial product are responsible for
their own patent due diligence and licensing.

## Why this notice exists

The maintainers researched comparable open-source projects (JMBE,
mbelib, OP25, SDRTrunk, DSDcc, dsd-fme) and found a consistent pattern:
all carry a patent disclaimer of this form, all distribute under
permissive or copyleft open-source licenses without payment, and none
have been the subject of patent enforcement action by DVSI in
10+ years. DVSI's documented enforcement (Codec2 / David Rowe, 2017–2019)
targeted commercial use, not the existence of open-source
implementations.

This notice aligns blip25-mbe with that prevailing posture and signals
the maintainers' awareness of the patent landscape.
