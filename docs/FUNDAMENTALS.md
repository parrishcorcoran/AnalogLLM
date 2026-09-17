# Fundamentals

The concepts, in the owner's terms. Minimum needed so they don't have to be
re-explained every conversation. Short on purpose.

## 1. Abelian

The whole network is addition on a cyclic group. Order does not matter.

That is why pulses can arrive in any order and the sum is still correct, and it
is why the sum is legal in a physical medium at all. **All we are is adds.**

## 2. The number: 0–4096 plus a lap counter

Every weight and every neuron value is a number between 0 and 4096, plus a lap
counter. 4096 positions on a circle.

The lap counter is amplitude. It matters, but not very much.

## 3. Complementary pairs

Negatives are complements, not a separate sign.

```
-1000 = 3096
```

No sign rail. No two passes. Subtraction is addition going the other way round
the circle.

## 4. Weights are thresholds, and they never move

A weight is a level sitting still. The clock's ramp crosses it and that makes the
pulse. Nothing is fetched.

Weights are set once and are **set forever** unless you train. Run 590k tokens,
same weights.

## 5. Depth collapses

Run the flow model and the 12 layers collapse into one. Depth is not a schedule,
it is where the pipes go.

## 6. Low weights matter

Low weights matter a lot. They are harmonic undertones. Non-resonant ones dampen,
important ones amplify.

## 7. The LM head is the exception

Everything is 0–4096 plus lap. The LM head is the only part that doesn't convert
to 0–4096 — it needs a resolution of 4096 × 4.

## 8. Encoding is free

Pulse width, odometer, radix, binary — doesn't matter. Digital pulse widths are
binary, and edges over cycles is one way to store data.

The code is a choice about how to spend time. It is not the machine.

## 9. Two alphabets

**Digital:** 0, -1, -0, 1.

**Analog:** frequency, pulse width, amplitude, duration, harmonics, ramp speed.

Analog also gives potential modes in wire, and potential cancellation in wire.

## 10. The harmonic model

The AI model just digitally encodes a digital medium for a harmonic system.

The first layers are a note or chord being played. Attention adds resonant nodes
— the meanings that are related. A version of music-theory tension builds. The
last layer is resolution, and the resolution note is the next token.

## 11. Parallelism is the key

Parallelism is the absolute, crucial key to making this work, digital or analog.

Every single weight, every neuron value, every add, in parallel. At most one tick
per layer — and that should collapse to one cycle too.

**The depth of a model is only the parallel processes possible on it.**

12 bits in a pulse width:

```
000000000001111111111
```

0 for 11 cycles of time, then 1 for 11 cycles. From there you can put a terminal
code, or flow straight into the next one, or run the entire stream in parallel
from 22 encoders each outputting a 0 or a 1.

Reference point: Mythic stores weights in FLASH as voltage at 8 bits. A pulse
width carries 12.

## 12. What this is called: HRR / FHRR / VSA

The number system above is not new — it has a name, a literature, and a
vocabulary. That is useful: it means other people's work can be read, and this
can be described to someone in one sentence instead of twenty.

**Holographic Reduced Representations (HRR)**, Tony Plate. Structured
information in fixed-width vectors, composed with algebraic binding. The broader
field is **Vector Symbolic Architectures (VSA)** / **hyperdimensional computing**.

**FHRR** — Fourier HRR — is the variant that matches this project almost
exactly. Every dimension is a **unit-magnitude complex phasor**: magnitude fixed
at 1, phase carries the information.

| here | FHRR |
|---|---|
| 0–4096 on a circle | a phase |
| lap counter is amplitude, matters but not much | magnitude fixed at 1 |
| all we are is adds | bundling = superposition = addition |
| complementary pairs, −1000 = 3096 | unbinding = complex conjugate = phase negation |
| abelian | phase addition is the circle group, commutative |

And there is a quantized version: **qFHRR** — *Rethinking Fourier Holographic
Reduced Representations through Quantized Phase and Integer Arithmetic*
([arXiv 2604.25939](https://arxiv.org/abs/2604.25939)). Each dimension is a
**discrete phase index**, with binding, unbinding, bundling and similarity done
as **integer modular arithmetic**. They report 3–4 bits per dimension is enough.

That is the 0–4096 dial with complement arithmetic, arrived at independently.

**This is established, not a conjecture.** A pretrained transformer performs
HRR. The literature:

- **Attention as Binding: A Vector-Symbolic Perspective on Transformer
  Reasoning** ([arXiv 2512.14709](https://arxiv.org/abs/2512.14709)) —
  self-attention and the residual stream implement an approximate VSA. Queries
  and keys define **role** spaces, values encode **fillers**, attention weights
  perform **soft unbinding**, and residual connections are the **superposition**
  of many bound structures.
- **Generalized Holographic Reduced Representations**
  ([arXiv 2405.09689](https://arxiv.org/html/2405.09689)) — GHRR binding
  implements attention with **mathematical equivalence**, verified by replacing
  the attention mechanism in a transformer with its GHRR equivalent and getting
  *better* performance on language modelling.
- **Recasting Self-Attention with Holographic Reduced Representations**
  ([arXiv 2305.19534](https://arxiv.org/pdf/2305.19534)).

So: the model *is* an encode/decode machine in a phase algebra. That part is
settled.

## 13. The analog connection

This is the step that is this project's own.

HRR's only primitives are **phase addition** and **superposition**.

A wire does both, natively. Superposition is what a linear medium does to two
signals sharing it — no hardware, no instruction, it is just what happens. Phase
addition is a delay.

So a pretrained transformer is not being *approximated* onto an analog medium.
It is the same algebra, running on a substrate whose native operations are that
algebra. The digital version is the translation; the analog version is the
original.

That is why it works.
