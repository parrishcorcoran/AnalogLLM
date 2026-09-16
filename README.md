# AnalogLLM

An analog GPT-2. The model run as pulses down a real wire, with the multiply and
the sum done by physics instead of arithmetic. The wire is an audio cable from the
laptop's output back into its input.

Companion to `timing-substrate`, which proved in software that a pretrained
transformer runs as durations on a clock and is bit-exact against fp32 at the top
of the dial. See [`docs/INHERITED_CONTEXT.md`](docs/INHERITED_CONTEXT.md).

**Accuracy first.** Speed is the thing that gets traded away, never accuracy. The
dial is a runtime parameter, not a bit width baked into a file.

## The machine, in one paragraph

Pulse `i` is emitted with **amplitude = wᵢ** and **width = xᵢ**. The charge it
deposits on a capacitor is `qᵢ ∝ wᵢ·xᵢ` — that is the multiply, performed by
holding a level for a duration. The cap holds the running total — that is the
addition, performed by accumulation. The final voltage is the dot product. No
arithmetic happens anywhere; the wire does it.

```
   LEFT out ---[ R ]---+--- LINE IN
                       |
                     [ C ]
                       |
   GROUND -------------+
```

## Start here

Everything downstream depends on what the cable actually gives you, so measure
that before writing a single layer into it.

```bash
pip install numpy               # sounddevice too, for real hardware
python3 -m loopback.rig --simulate       # model the path, no hardware
python3 -m loopback.rig --list           # find your device index
python3 -m loopback.rig --device 1       # measure the real cable
python3 -m pytest tests/ -q
```

The rig reports five things: round-trip latency, the noise floor in bits, how much
of the **start** of a pulse train still counts by the **end** of it, whether the
readback is linear in the true dot product, and what that buys in
multiply-accumulates per second.

## Findings so far

Nothing here has touched hardware yet. These are `[SIMULATED]` — produced by the
model of the path in `loopback/model.py`, which includes the passive RC, both
AC-coupling stages, converter quantization and noise. They are predictions about
what the rig will find, not measurements. The `--device` run replaces them.

**1. The naive circuit cannot compute a long dot product, and AC coupling is why.**
`[SIMULATED]` A laptop's output and input are both AC-coupled — a series capacitor
that is not allowed to pass a held level. But a held level is exactly what an
accumulated sum *is*. Measured against the channel's own impulse response, a pulse
placed at the start of a 256-tick train contributes **negatively** by the time the
train ends: the coupling cap has handed back more charge than it took. Driven raw,
R² against the true dot product falls from 0.997 at a 32-tick train to 0.008 at
4096 ticks. The train is not merely attenuated; its early elements change sign.

**2. Dividing the channel out fixes it, up to a hard capacity limit.**
`[SIMULATED]` The channel is linear and time-invariant, and reading one sample of
its output is an inner product with its time-reversed impulse response. That
response is measurable in a single capture, so it can simply be divided out at the
transmitter — pre-emphasis, the same move as a phono curve. With each slot
pre-scaled by the inverse of what the channel will do to it:

| train | slots | raw R² | raw bits | compensated R² | compensated bits |
|---|---|---|---|---|---|
| 32 ticks | 8 | 0.9968 | 4.1 | 0.9995 | 5.4 |
| 64 ticks | 8 | 0.9701 | 2.5 | **0.9999** | **6.4** |
| 128 ticks | 32 | 0.9080 | 1.7 | 0.9987 | 4.8 |
| 256 ticks | 27 of 32 | 0.3078 | 0.3 | 0.9924 | 3.5 |

The limit is not gradual. The reversed response passes through **zero** somewhere
in a long train, and a slot sitting on that null carries nothing at any drive
level — dividing by it only clips the drive. So the cable's capacity for one dot
product is a **count of usable slots**, and past ~256 ticks that count starts
falling. This is the number to design against, not the train length.

**3. The cable is an instrument, not an engine.** `[DERIVED]` One slot carries one
multiply-accumulate. At 48 kHz with an 8-tick slot that is 6,000 MAC/s on one
channel. GPT-2 needs ~124M MACs per token, so the whole model through one cable is
**~5.7 hours per token** — and no encoding fixes that, because the ceiling is
`sample_rate / slot_ticks` and the sample rate is the sample rate.

What the cable *can* do at that rate is one 768-input neuron every 128 ms. So the
honest shape of this project is a **hardware-in-the-loop GPT-2**: real dot products
computed in copper, the rest in software, and the question being answered is
whether the token survives. The timing-substrate numbers already say ~1,024
parallel converters is where thousands of tokens per second live. A cable gives
you one.

## Layout

```
loopback/
  model.py    the path modelled honestly — passive RC that forgets, AC coupling
              that cannot hold a level, converter quantization, noise
  rig.py      the measurement stages, the sweep, and the budget arithmetic
docs/
  INHERITED_CONTEXT.md   measured ground truth from timing-substrate
tests/        tests of the instrument, not of the cable
```

## Next

Run `--device` and replace every `[SIMULATED]` above with a measurement. The
model's job was to say what to look for; only the cable can say what is there.
