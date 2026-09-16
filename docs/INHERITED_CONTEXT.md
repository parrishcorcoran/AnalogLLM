# Inherited context — the timing-substrate results

Carried over from the `timing-substrate` project so this repo does not have to
re-derive them. Everything here was measured on real code by the repo owner,
against GPT-2 (124M) and Qwen2.5-0.5B. **Treat as ground truth.**

Tags follow the house rule: `[MEASURED]` on hardware or real code, `[DERIVED]`
by arithmetic, `[CONJECTURE]` untested.

---

## The core idea

Every weight is a signed integer **duration** — how many ticks of a cycle a line
stays high — not a magnitude. Each output column has its own cycle scale. One
runtime parameter `T` (ticks per cycle) sets the resolution of every duration in
the model, and therefore sets both accuracy and speed. `T` is changed between
forward passes on the same file, with no reconversion.

```
cyc[j]   = max over i of |W[i,j]|          # per OUTPUT column
Wq[i,j]  = round( W[i,j] / cyc[j] * T )    # int16 pulse width
```

This is **not quantization**, and the distinction is load-bearing: quantization
picks a bit width at conversion time and bakes it into both the file and the
kernel. Here one file runs at any `T` from 4 to 65536, and the top of the dial is
bit-exact agreement with fp32 — which quantization does not offer at any width.

## The dial, measured on GPT-2 [MEASURED]

| `T` (ticks/cycle) | perplexity vs fp32 | top-1 token agreement |
|---|---|---|
| 256 | 1.17–1.23× | 73–76% |
| 1024 | 1.00–1.02× | 92–96% |
| 4096 | 1.000–1.006× | 98.4–98.8% |
| 16384 | 1.000× | 99.6% |
| 65536 | **1.000×** | **100.0%** |

Qwen2.5-0.5B at `T=4096` produced identical generated tokens to fp32.

## Add-only execution works [MEASURED]

The whole model, no multiply in any layer, using an integrating clock that adds
the summed current each tick and subtracts a line's contribution when its width
expires: relative error 0.00028, correct English output, 6.1 tok/s single-threaded
(`clocks/clock_int.c`). Slower than a matmul on a CPU, which is expected — a
multiply instruction *is* the hardware that performs those repeated adds in one
cycle. The add-only form matters because it proves the multiply is unnecessary,
which is what makes the design portable to a substrate that has accumulators and
no multipliers. **That substrate is what this repo is about.**

## Flowing execution is more accurate than staged [MEASURED]

One global clock, every layer live on every tick, each stage reading what the one
before it emitted on the previous tick:

- 2-layer test: relative error 0.0251 flowing vs 0.0362 batch (cosine 0.9997 vs 0.9993)
- Whole GPT-2, 128-tick cycle: flowing final answer matches batch, and the token
  **locked at tick 15 of 128** — decided at 12% of the cycle, the rest confirmation
- Early exit on argmax stability (K=8 consecutive ticks) cost 23 ticks, not 128.
  Do **not** use confidence as the stopping signal; it flickers during the
  transient and can be high on a wrong guess.

## One-bit streams reproduce the model [MEASURED]

Every activation as a 1-bit pulse-density stream — the DSD encoding, what a MEMS
microphone and an audio DAC speak natively. At 1,024× oversampling the output is
character-for-character the fp32 model's. Precision comes from **rate**, not bit
width.

Why this works here when DSD audio cannot do it: multiplying two bitstreams is the
thing that field cannot do cleanly. This machine never needs to — the weights are
stationary constants, so a pulse means "add this pipe's aperture", which is
bitstream × constant, not bitstream × bitstream.

The converter is three lines of state:

```c
err += lvl;                      // integrate
bit  = (err >= 0) ? +1 : -1;     // comparator -> ONE BIT
cnt += bit;                      // what a counter downstream sees
err -= bit;                      // feedback: subtract what was emitted
```

Measured: 690 M ticks/s scalar on one core, 4.6 G ticks/s at 16 converters per
SIMD instruction, with **0 pulse difference** between them.

## Where the parallelism has to come from [MEASURED + DERIVED]

One token needs 65,280 values converted. At R=1,024 that is 66.8M tick-operations
— 10 tok/s on one core [MEASURED]. But every converter is independent, so the
work does not shrink and the *waiting* does: ~1,024 parallel converters is where
thousands of tokens per second appears [DERIVED].

**This is the number that scopes the analog track.** An audio cable gives you one
or two lines, not 1,024. The cable is an instrument for proving the physics, not
an engine for throughput — and the project's own discipline says to say so.

## Two traps already paid for

- **Positional beats unary by 341×** [MEASURED]. Pulse-width is a unary code: the
  value 2000 costs 2000 ticks. The same accuracy (rel. error 0.00028) comes from
  12 ticks base-2, 3 ticks base-16, 2 ticks base-64. The radix is a free dial on
  tick count. Its top rung is an ordinary integer matmul — which is why the
  software machine was already at its optimum.
- **The ODE framing is wrong** [MEASURED]. RK4 at 32 evaluations was no better
  than Euler at 32. Integer counting has no derivative to approximate and no
  truncation error; the error is purely input grain size, scales as 1/G, and does
  **not** compound with depth (7.1e-4 → 5.2e-4 across twelve layers).

## The open crux, restated for this repo

From the original handoff: *"weight" must enter twice — as delay (which input,
when) and as strength (how much it counts)*. In the audio-cable machine that is
resolved concretely: **strength is amplitude and delay is pulse width**, and their
product is charge. Whether that stays linear through a real AC-coupled codec is
exactly what `loopback/rig.py` measures.
