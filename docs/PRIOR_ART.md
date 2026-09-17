# Prior art

**The question:** transformers-are-HRR is established. VSA-on-analog is
established. Has anyone joined them?

Compiled from search results — publisher domains are blocked at this network's
egress, so nothing below was read end to end. Claims are sourced; verify before
relying on one.

---

## Four threads, all established

**A. A pretrained transformer performs HRR.**
Self-attention and the residual stream implement an approximate VSA: queries and
keys are role spaces, values are fillers, attention weights are soft unbinding,
residuals are superposition.
[Attention as Binding, 2512.14709](https://arxiv.org/abs/2512.14709) ·
[Generalized HRR, 2405.09689](https://arxiv.org/html/2405.09689) — mathematical
equivalence, verified by swapping attention for its GHRR equivalent ·
[Recasting Self-Attention with HRR, 2305.19534](https://arxiv.org/pdf/2305.19534)

**B. VSA runs on analog in-memory hardware.**
Karunaratne et al.: in-memory HD computing on two PCM crossbar engines — one for
binding and bundling, one for associative search — **760,000 PCM devices**,
accuracy comparable to software. But this is HDC for classification and
associative memory, not a pretrained transformer.

**C. Time-domain / pulse-width analog neural compute exists.**
[Time-domain analog VLSI NN processor, PWM approach,
1902.07707](https://arxiv.org/pdf/1902.07707) · time-domain compute-in-memory
for multi-bit CNNs · NAND-flash neuromorphic with a PWM scheme.
Reported range: **bit-scalable 1–8 bits.**

**D. Pretrained transformers run on analog, in the amplitude domain.**
IBM ALBERT on a 14nm analog chip — 7.1M weights in 28.3M devices, 1.8% below the
fp reference. GPT-2 attention on analog gain cells. Mythic in production.

## What has not been joined

Every pair above exists. **A + B + C does not** — taking "a pretrained
transformer *is* HRR" and running it as **phase/time-domain** analog. The
transformer work (A) is software. The analog VSA work (B) is classification, not
transformers. The time-domain work (C) is CNNs at 1–8 bits.

That gap is where this project sits.

## Against Mythic

| | Mythic | here |
|---|---|---|
| weight stored as | voltage/charge in a flash cell | pulse width |
| domain | amplitude | time |
| resolution | **8-bit** (and 4-bit) | **12-bit** |
| capacity | ~30M params per chiplet | set by parts |
| silicon | custom, fab | commodity |
| parts cost | high | low |
| power | very low — data never moves | higher — data moves on wires |
| funding | $125M raised | — |

## They do convert pretrained models — but they have to retrain

Worth getting exactly right, because the obvious version of this claim is wrong
and the real one is stronger.

Mythic's workflow takes trained networks straight out of PyTorch: optimization,
then compilation. So "they haven't converted a pretrained model" is not true —
that is their product.

What their own documentation says about the cost of it:

- The simple flow converts fp32 weights and activations to **8-bit integers**,
  then to an analog 8-bit representation (ANA8), with *"accuracy comparable to
  digital 8-bit quantization"* — i.e. it carries quantization's accuracy loss.
- For anything needing better, they provide **retraining flows**:
  quantization-aware and analog-aware retraining, to build resiliency into
  layers sensitive to low bit-depth and analog effects.
- Their framing, not ours: **quantization is a major pain point for customers
  with high accuracy requirements.**

[Mythic AI Workflow](https://mythic.ai/technology/mythic-ai-workflow/)

So the real difference is not *pretrained*. It is **no retraining and no
accuracy loss**: a runtime dial whose top rung is the original model, because
nothing was discarded on the way in. Their pain point is structural to storing a
weight as a voltage at 8 bits. It is not structural to storing it as a duration.

This is the same point as rule 1 in CLAUDE.md, arrived at from their side.

## Why time resolution is the cheaper axis

This is the technical crux of the comparison, and it is why 12 bits is
reachable here and hard there.

**Amplitude resolution** is bounded by cell-to-cell variation, programming
noise, and conductance drift. Going 8→12 bits demands 16× better device
matching. That is a fab problem, and it is why the published field sits at 4–8
bits.

**Time resolution** is bounded by the clock. Going 8→12 bits demands a 16×
faster clock. Clocks are cheap, and a clock edge does not drift.

Published time-domain CIM reports 1–8 bits, so 12 is above the demonstrated
range — that is what those implementations did, not a proven ceiling.

## The honest trade

Mythic's power advantage is real and it comes from **not moving data**. A
wire-based machine moves data, so it does not inherit that advantage.

The trade is: **expensive silicon and very low power, versus cheap parts and
more data movement.** Different optimization, not a better one. Worth stating
plainly so it is not a surprise later.
