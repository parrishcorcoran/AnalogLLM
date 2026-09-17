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

**3. Speed is set by how many weights are physically in the circuit, not by the
clock alone.** `[DERIVED]` `python3 -m loopback.budget`. One input value in GPT-2
drives 1,892 multiply-accumulates — it reaches every output neuron of its layer
through a different weight. Whether the wire performs those 1,892 at once or one
at a time is the entire question, and it is decided by how many resistors are
soldered to it, not by the sample rate.

| tier | what is physically there | ticks/token, PDM-exact | @48 kHz | @DSD512 |
|---|---|---|---|---|
| 1 | one resistor, one cap | 126,496,800,768 | 31 days | 1.6 h |
| 2 | crossbar, 4-in/4-out interface | 16,711,680 | 5.8 min | 1.35 tok/s |
| 2 | crossbar, 256-in/256-out | 261,120 | 5.4 s | 86 tok/s |
| 3 | one wire per weight, flowing | 1,036 | 46 tok/s | 21,795 tok/s |

Every row scales linearly with the clock — so cycle speed *is* everything, within
a tier. But the whole span of audio clock rates, 48 kHz to DSD512, is worth 470×,
and moving one rung up the ladder is worth 7,569×. Tier 1 is the only rung that
is a wiring change; the rest are hardware builds, and tier 2 needs 133,201
reprogrammable conductances, which is the component the handoff already flags as
the open problem.

The useful consequence: **tier 1 will never beat the software machine** (23.8
tok/s [MEASURED]) at any audio clock that exists. It is not there to be fast. It
is there to answer whether charge on a wire stays linear enough to hold a
transformer, at a 20 µs tick — within a factor of 20 of the microsecond substrate
the handoff says to build first.

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
