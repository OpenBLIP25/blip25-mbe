# Design goal: match what dispatch consoles actually produce

*Design rationale.*

## The target is the console, not a metric

blip25-codec is this project's own codec engine for the P25 MBE vocoder family —
IMBE (Phase 1) and AMBE+2 (Phase 2). Its correctness target is a specific,
concrete thing: the **reference codec that public-safety dispatch consoles actually
run.** Success is defined as blip25-codec producing, on the same input, output
that matches what that console codec produces — close enough that a listener
copying the traffic cannot tell them apart.

That target is deliberately **not** "a codec that scores well on a metric."
Divergence is measurable with spectral correlation, PESQ, STOI, level and
envelope metrics, and per-parameter field agreement — and every one of them can
be satisfied while still missing the thing that matters. Those metrics are
useful for *locating* where two signals differ. They are not the definition of
correct.

## The target is intelligibility over a radio, not fidelity

The console codec is the product of **thousands of hours of subjective listening
panels, field trials, and deliberate engineering tradeoffs**, all aimed at one
goal: making speech **survive a degraded radio channel and be understood
correctly, the first time.** That is a fundamentally different objective from
"sounds good in a quiet room."

It is the same instinct as radio voice procedure. You say **"niner"** instead of
*nine* and **"fife"** instead of *five* — not because they are more pleasant, but
because they are unmistakable through noise, static, accent, and a clipped
transmission. The codec has its own subtle, hard-won version of that: choices a
fidelity metric marks *down*, or that sound less smooth on headphones, precisely
because they are tuned for the **channel**, not for **comfort**. The measure is
whether the message gets through — not whether it is easy on the ear.

Those decisions are **not in the spec.** The TIA-102 documents pin the wire
format, the bit ordering, and the FEC; they do not pin the analysis and
synthesis choices that make dispatch speech intelligible under a noisy channel.
A metric can tell you two signals differ; it cannot tell you which difference a
dispatcher's ear actually needs preserved. So a spec-plus-score approach was
never going to converge on the target — it optimizes a different thing and calls
it "better."

## How blip25-codec is tuned

blip25-codec is tuned so that its output matches the console codec on the same
input, for **both IMBE and AMBE+2**, on encode and decode. When a design choice
inside blip25-codec would score well on a fidelity metric but move its output
away from what the console produces, the console wins — because the console *is*
the target. "Different, and higher-scoring" is not "better"; the point is to land
on the console's output, not near it.

Do not try to "improve on" the codec's austere-sounding choices. They are not
outdated technology to be modernized; they are the result of tuning for a
channel, and reproducing them is the job.

## How correctness is judged

By **ear**, against the console, via stereo A/B (ours on one side, the console on
the other — centered means match). The measurement harnesses in this repository —
spectral correlation, field agreement, byte-exact percentages, PESQ/STOI — are
**diagnostics**: they locate a divergence once the ear has flagged one. They are
never promotion gates, because at these bit rates they have repeatedly failed to
predict what a dispatcher's ear actually needs.

## Patent and IP scope

Downstream use is subject to the same reference / P25 intellectual-property
considerations that apply to any P25 vocoder work — including active patents on
the AMBE+2 wire construction. See [`../PATENT_NOTICE.md`](../PATENT_NOTICE.md)
for the full accounting and project policy.

## The one-line version

blip25-codec is not tuned to sound good in a quiet room. It is tuned to match the
codec that dispatch consoles actually run — because the job is getting the call
through, and that is exactly what the target codec was built to do.
