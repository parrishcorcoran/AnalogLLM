# Building blocks

What each piece of the machine is, and what it costs. Short on purpose.

---

## The parts list

Every operation the machine needs, and what performs it:

| operation | what does it | cost |
|---|---|---|
| **bundle** (superpose, add) | two signals sharing a conductor | **nothing.** it is what a wire does |
| **bind** | a filter (convolution in time = multiply in frequency) | one filter |
| **unbind** | the same filter reversed (a matched filter = correlation) | the same part, plus a sign |
| **normalise** | a limiter — keep phase, discard magnitude | **two diodes** |
| **nonlinearity** (GELU) | a diode or FET curve | one part |
| **weight value** | a counter compare value | exact, digital, no drift |
| **clock** | anything both ends can count | free |
| **readout** (argmax) | winner-take-all — which line fired first | known circuit |

Nothing on this list is hard. Two of them need no components at all.

## The one thing that is hard

**Where a note lives when nobody is playing it.**

A digital tap is a multiply-accumulate, so its weight is read from memory every
time it fires. That is the memory wall, back again. The question is not how to
avoid transferring a note — it is **what holds a note after it arrives.**

There are exactly three kinds of answer, and every one of them has a cost:

| where the note lives | set once? | cost |
|---|---|---|
| **a number in memory** | no — read every use | the memory wall |
| **physical geometry** (a tap position, a length) | yes, forever, exact | you must *build* it. fab at scale |
| **stored analog state** (charge on a cell) | yes | drifts, and bounded to ~8 bits |

That is the whole space. Mythic chose the third and spends $125M fighting the
drift. The digital machine chooses the first and pays the wall every token.

## What actually holds a note

Working down from worst to best:

- **a capacitor** — holds an amplitude. leaks, drifts with temperature.
- **a counter** — holds a number exactly, forever, no drift. but it is digital,
  and it needs a clock to become a note again.
- **a physical length** — holds a *delay* exactly and permanently. a delay is a
  phase. this does not drift because geometry does not drift.
- **a recirculating loop** — holds a whole waveform, going round a lap. real
  technology (mercury delay lines, 1940s) but small capacity.
- **a resonator** — a cavity, a crystal, a SAW device, a length of wire with
  reflections. **A resonator holds a note by definition.** No power, no refresh,
  no drift. A multimode structure holds a comb of notes at once.

## The honest gap

A resonator holds notes perfectly and for free. What it does not do is let you
*choose* which notes, after it is made — the modes are set by geometry, which
means manufacturing.

**Nobody has a cheap programmable version.** That is the open component problem
the whole neuromorphic field is stuck on, stated in this project's terms: a thing
that holds a note exactly, forever, and can be retuned.

Everything else on the parts list is solved.
