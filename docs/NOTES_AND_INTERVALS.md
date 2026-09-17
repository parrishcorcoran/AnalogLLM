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

## 2. A station is a rotation

Send a wave down a fixed length. It comes out further around its own circle, and
how much further depends on how fast it wiggles. The length is built once and
never changes. So the only free variable is which frequency you speak on, and
**picking the frequency is picking the rotation**. **[arithmetic]**

That is why a station is a weight. A station does not move. Set at build time,
set forever unless you train — which is the requirement the weights already had.

Go high enough and it comes all the way around and starts over. The wrap is the
mod, done by the medium, with no counter anywhere.

## 3. A weight is an interval

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

## 4. Tension and resolution

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

## 5. The net is already doing superposition

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

## 6. Why this does not speed up a normal computer

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

## 7. Tuning, and what it costs

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

## 8. What the whole model costs, in these units

**[arithmetic]** GPT-2 small, standard, no trades:

| | |
|---|---|
| one dense 768x768 projection | one set of ordered intervals |
| one block | 12 of them (4 attention, 8 MLP) |
| twelve blocks | **144 interval sets** |
| LM head | not a matrix — the fork bank it rings against |

Early exit and early start are deliberately **not** used. They buy speed with
accuracy, which is the same trade as quantizing. Standard model first.

## 9. Open

- Do GPT-2's trained values land on intervals a real medium can produce, or do
  they need 589,824 arbitrary numbers with no structure between them? This is
  the difference between a build and a wish. **[open]**
- Attention setting the key — which resolutions are available — is a guess and
  nothing has been measured. **[open]**
- How the interval is physically formed. **[open]**
