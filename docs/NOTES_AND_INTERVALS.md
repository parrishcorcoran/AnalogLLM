# Notes and intervals

How the model looks when you stop reading it as arithmetic and read it as notes.
Builds on FUNDAMENTALS.md, does not replace it.

Every claim below is tagged:

- **[owner]** measured by the owner on real weights
- **[measured]** measured here, code in `experiments/`
- **[arithmetic]** exact counting, no assumptions
- **[cited]** published elsewhere, source named
- **[open]** nobody has checked this

---

## 1. The multiply disappears

Write a number as an angle instead of a magnitude. Then multiplying two numbers
is going further around the circle — you add the angles. Rotate 1000 by 300 and
you are at 1300. Rotate backwards and you wrap: that is the complement, and it
is why negatives need no sign bit and no separate machinery. **[arithmetic]**

So the multiply is not made cheaper. It is gone. What is left is rotating and
summing, and a medium does both without being asked.

## 2. The dial is a change of base, not a quantization

Every number in the model, weights and activations both, carried as a position
on a 4096 dial plus a lap counter. The head is the exception and gets four
digits. This is **fp32 accurate**. **[owner]**

It is exact for a structural reason, not by luck: **[arithmetic]**

| | |
|---|---|
| one dial | 4096 = 2^12 |
| dial plus one lap | 4096 x 4096 = 2^24 |
| fp32 significand | 2^24 |

One dial and one lap *is* fp32's mantissa, to the bit. Which is why the number
is 4096 and not 1024 or 8192 — it is the square root of fp32.

So this is not 12-bit anything. It is the same number written in base 4096 with
two digits instead of base 2 with twenty-four. Nothing is rounded because
nothing is discarded, and the phrase "we will not quantize" is literal.

The head at four digits is 2^48, well past fp32 — headroom where 50,257
near-ties have to be separated.

## 3. A station is a rotation

Send a wave down a fixed length. It comes out further around its own circle, and
how much further depends on how fast it wiggles. The length is built once and
never changes. So the only free variable is which frequency you speak on, and
**picking the frequency is picking the rotation**. **[arithmetic]**

That is why a station is a weight. A station does not move. Set at build time,
set forever unless you train — which is the requirement the weights already had.

Go high enough and it comes all the way around and starts over. The wrap is the
mod, done by the medium, with no counter anywhere.

## 4. The weight has to be a delay

A tapped line is a convolution -- signal round a loop, each tap delaying and
scaling, all summed at a node. That is textbook FIR. What is worth writing down
is that it equals HRR binding exactly, and that reading it backwards is
unbinding on the same hardware, with one condition. **[measured]**

| taps | forward then backward recovers the input |
|---|---|
| random | fails, error 2.5 |
| all-pass (unit spectrum) | exact, 5.7e-15 |
| **a plain delay** | **exact, 0.00** |

Only an all-pass tap pattern inverts. A pure delay is the simplest all-pass
thing there is. So a weight being a delay is a **requirement**, not a
convenient choice — and an attenuator on a tap is lossy, therefore not
all-pass, therefore breaks the invertibility that makes unbinding free. Earlier
drafts here put a mask on every tap; that was the wrong part.

Scope: all-pass matters where the signal must be read *back*, which is
attention's retrieval. An MLP is never inverted in a forward pass and does not
carry the same constraint. **[open]** how far the requirement reaches.

Separately, unbinding is exact only at unit magnitude: **[measured]**

| | unbind recovers the input |
|---|---|
| gaussian HRR (Plate 1995) | 0.710 |
| FHRR, unit magnitude, 4096 phases | 1.000, phase error 1.8e-16 |

Magnitude 1 is not a simplification of the representation. It is what makes
unbinding lossless. "The lap matters, but not very much" and "unbinding is
exact" are the same statement.

## 5. A weight is an interval

A weight needs two addresses: it is the weight *from* note j *to* note i. A pitch
only has one. The second address comes from the pair — the interval between two
notes, which is what music puts the meaning in. Two notes meeting anywhere
non-linear give their difference, and the phase of that difference is the
difference of their phases. Two one-address things in, one two-address thing out,
no table. **[open]** — this is the framing, not a verified construction.

The counting is exact, though:

| | |
|---|---|
| ordered pairs among 768 notes | 589,824 |
| entries in one 768x768 matrix | 589,824 |

The same number. A dense projection is not *approximated* by intervals, it **is**
the set of intervals. Nothing dropped, nothing quantized. **[arithmetic]**

If the notes are evenly spaced this breaks — 768 evenly spaced notes give only
767 distinct intervals for 294,528 unordered pairs, 384 pairs colliding on each.
The notes have to be spaced so no two pairs sit the same distance apart. That
costs span, not precision: ~295,000 notches instead of 768. **[arithmetic]**

## 6. Tension and resolution

Not metaphor. Tension in music is beating — two things close but not equal,
wobbling at their difference, a real envelope you can put a meter on. Resolution
is that wobble going to zero.

The notes are rung at the embedding, they bloom with attention, and they resolve
at layer 12 into the next token. **[owner]**

The last layer already works this way. Picking the next word means holding the
state against 50,257 vocabulary vectors and taking the largest — which is fifty
thousand tuning forks and one of them singing. Embedding and LM head are the
same tied matrix used in both directions: word to chord, chord back to word.
**[arithmetic]**

**This removes softmax.** Softmax is what you need when you are *computing* which
fork rings. If the forks are physically present and the chord is physically
played, the loudest one is loudest. There is no argmax to perform.

## 7. The net is already doing superposition

The residual stream carries far more features than it has dimensions, by putting
them in nearly-orthogonal directions and tolerating the overlap. The published
name for this is superposition; a neuron carrying several unrelated signals is
called polysemantic. **[cited — Elhage et al., *Toy Models of Superposition*,
Anthropic 2022]**

What that buys, for 768 dimensions: **[arithmetic]**

| crosstalk allowed | stations that fit |
|---|---|
| none (orthogonal) | 768 |
| 20% | 2,160 |
| 30% | 32,000,000 |

Forty thousand times the capacity for tolerating interference. The network took
the deal. The crosstalk is bought, not suffered.

**The design rule that falls out:** the network already runs at a tolerated
interference level. Noise the medium adds *underneath* that level is invisible —
not acceptable, invisible. Only noise that pokes above it costs anything.

## 8. Why this does not speed up a normal computer

It is worth writing down why the same trick does not just make PyTorch faster.

First, matmul is multiply-*accumulate*, and angles only make the multiply free.
Adding two notes is not adding their angles — it is what happens when two waves
meet — so on a digital machine you have to leave angle-land, sum, and come back,
per term. One cheap multiply traded for an add and two conversions. This is old
and known; it is why logarithmic number systems never took over.

Second, and larger: the multiply was never the cost. **[cited — Horowitz, ISSCC
2014 keynote, 45 nm]**

| | |
|---|---|
| 32-bit float add | 0.9 pJ |
| 32-bit float multiply | 3.7 pJ |
| **read one weight from DRAM** | **640 pJ** |

Using one weight once costs 644.6 pJ. Make the multiply an add and it costs
641.8 pJ — a saving of **0.43%**. The fetch is 99% of it. **[arithmetic]**

So the claim is not "adds instead of multiplies." It is that **the weight never
travels**. A station is not fetched. The 640 pJ is not reduced, it is never
incurred, because there is nothing to move.

## 9. Tuning, and what it costs

A resonator does not read. It responds. And a bank of them responds *at once* —
4096 tuned things on one wire all ring simultaneously from the same passing
wave, so listening to 4096 stations costs what listening to one costs. That is
the fetch gone rather than made cheaper.

The price is selectivity: you must listen long enough to tell neighbours apart.

    listen per tick = 4096 / bandwidth

**The length of the medium cancels out.** **[arithmetic]**

| band | station gap | listen/tick | tokens/s at 12 ticks |
|---|---|---|---|
| 100 MHz | 24 kHz | 41 us | 2,035 |
| 1 GHz | 244 kHz | 4.1 us | 20,345 |
| 10 GHz | 2.4 MHz | 0.41 us | 203,451 |

At 10 GHz the loop that spaces the comb correctly is **2 cm** long. Holding
notes in frequency rather than strung out in space makes the medium small.

## 10. What the whole model costs, in these units

**[arithmetic]** GPT-2 small, standard, no trades:

| | |
|---|---|
| one dense 768x768 projection | one set of ordered intervals |
| one block | 12 of them (4 attention, 8 MLP) |
| twelve blocks | **144 interval sets** |
| LM head | not a matrix — the fork bank it rings against |

Early exit and early start are deliberately **not** used. They buy speed with
accuracy, which is the same trade as quantizing. Standard model first.

## 11. Open

**Answered, and it is two numbers that do not compose.** **[owner]**

*Across the twelve blocks:* three. Twelve station kernels, two generators cover
90%, three cover 99.2%, four exact. Stations 5-8 are nearly the same kernel
(cosine +.98, +.97). The delays are shared.

*Within one matrix:* 220-410, against a random control of 614 — a third to
two-thirds of available capacity. Participation ratio 216-407, spectral entropy
394-544, rank at 90% 399-530, three readings in the same band. The
Marchenko-Pastur fit lands at 800-920 superposed terms, **above** the 768
dimensions: overcomplete superposition, more bound pairs stored than there are
dimensions to store them in. That is the HRR capacity regime.

**That second number does not answer the generator question**, and it was
recorded here as if it did. Participation ratio, spectral entropy and rank at
90% are all functions of the singular values — which is why all three land in
the same band; they are one measurement. And the controls in
`experiments/weight_structure.py` show exactly that measurement is blind here:
a matrix built from ten mask-and-delay pairs reads **rank 94 of 192, generators
9**, and a pure convolution is full rank. So 220-410 cannot tell 4 generators
from 768.

What it does establish is real and worth keeping: one matrix is a dense
superposition of rank-1 terms, overcomplete, at HRR capacity.

**Still unknown:** the mask-and-delay generator count for a single matrix,
`rank(M)` where `M[k,i] = W[i, (i-k) % d]`. Which route is buildable —
streaming at $1,882/tok/s, or stationary weights — still turns on it.

(superseded) **Three.** Twelve station kernels, two generators cover 90%, three
cover 99.2%, four are exact. Stations 5 through 8 are nearly the same kernel
(pairwise cosine +.98, +.97). So the delays are four lengths of glass shared
across all twelve blocks, not 144. **[owner]**

This was measured as: one kernel fitted per station, then how many of the twelve
are independent. That is inter-block redundancy — a related but not identical
quantity to `rank(M)` per projection in `experiments/weight_structure.py`, which
asks how many mask-and-delay pairs build a single matrix. Both being small is
what a build needs; only the first is measured.

Precision is settled (section 2). The rest:

**How many generators?** `W = sum_k D_k S^k` is exact with d terms. Stack the
masks as `M[k,i] = W[i, i-k]` and `rank(M) = r` means exactly

    W = sum over r of  diag(u_m) @ C_m

— r amplitude masks, each followed by one convolution. A convolution is a
binding, so r is the generator count. Small r is a build: r masks and r
bindings instead of 589,824 stored values. **[arithmetic]**

**A holographic code is built to look random**, and that changes how a null
result reads. Bound vectors have random-looking entries by construction, so
singular values and diagonals are precisely the instruments they hide from.
A matrix made of ten mask-and-bind pairs reads as near-full rank and sits
exactly on the random null for conv-energy — and the generator test catches
it at 9. Verified against controls in `experiments/weight_structure.py`.
**[measured]**

So "most of the energy is in near-random directions" is not evidence against
structure in the bulk. It is what structure in the bulk looks like from a
second-order test. Every earlier test here would have called a ten-generator
matrix noise.

**Intrinsic dimension, measured on gpt2 by the owner with TwoNN** (Facco et al.
2017). Reported readings, plus de-biasing measured here against clouds of known
dimension at the same sample count: **[owner]** for the readings, **[measured]**
for the correction.

| | TwoNN read | true dim | of 768 |
|---|---|---|---|
| random control | 160.88 | 768 | 100% |
| block1 mlp.c_fc | 117.26 | 473 | 62% |
| block6 mlp.c_fc | 85.30 | 270 | 35% |
| block12 mlp.c_fc | 43.11 | 79 | 10% |
| embedding table | 35.16 | 58 | 8% |
| **block1 attn.c_attn** | **7.41** | **8** | **1%** |

Two things about reading these. The estimator needs samples exponential in the
true dimension, so it reads low — but the bias is 4.9x at true 768 and under 1%
below true 10. **A low reading is therefore trustworthy and a high one is a
floor**, which is the opposite of the usual worry: 7.41 is not a squashed 400,
it is about 8. The control landing at 160.88 where a synthetic full-dimensional
cloud reads 157.88 says the measurement itself is sound. **[measured]**

And the de-biasing matters for the rest: 117 against a control of 161 looks like
a small gap and is really 473 of 768. The shape that survives correction is
**intrinsic dimension collapsing with depth** — 473, 270, 79 across blocks 1, 6
and 12 — and block 1's attention sitting at 8.

One artifact fakes a low reading and only one. Near-duplicate rows do not bias
TwoNN, they annihilate it: 10% of them take a genuinely 768-dimensional cloud
from 157 to **0.27**, and it is a cliff rather than a slope. Separated clusters
— gpt2's fused Q/K/V is three of them — barely move it, 157 to 145. So the
duplicate fraction has to be reported next to any low dimension, and the probe
now does. **[measured]**

What a real intrinsic dimension of 8 would mean, if it survives that check:
each row of the projection is set by 8 numbers rather than 768. Per note that is
8 values and a shared shape, so 768 x 8 = 6,144 instead of 589,824. **[open]**
- Attention setting the key — which resolutions are available — is a guess and
  nothing has been measured. **[open]**
- How the interval is physically formed. **[open]**
