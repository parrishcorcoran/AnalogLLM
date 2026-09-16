# CLAUDE.md — read this before doing anything in this repo

## What this project is

An **analog GPT-2**: the model run as pulses down a real wire, with the multiply
and the sum performed by physics rather than by arithmetic. The wire, for now, is
an audio cable from the laptop's output back into its input.

This repo is the analog track of the timing-substrate work. That project proved a
pretrained transformer can be run as durations on a clock, in software, with the
top of the dial bit-exact against fp32. See `docs/INHERITED_CONTEXT.md` — those
results are **ground truth**. Your job is to build on them, not re-derive them.

## Rules — these matter

1. **Accuracy is the point. This is not a quantization project.** The owner's
   position, and the one the measurements support: build the substrate exact and
   let speed be the thing you trade away, never the reverse. `T` (ticks) is a
   runtime dial on one file, not a bit width baked into it. If you find yourself
   about to reach for a smaller number format to go faster, stop.
2. **Report measured numbers only, and say which kind.** Every number carries a
   tag: `[MEASURED]` on hardware, `[SIMULATED]` from a model of the path,
   `[DERIVED]` by arithmetic from those, `[CONJECTURE]` not yet tested. A
   simulated number is not a measurement and must never be written as one.
3. **Model the circuit as it is, not as it would be convenient.** The summing
   network is a passive cap, so it forgets. The codec is AC-coupled, so it cannot
   pass a held level. Both of those are in `loopback/model.py` and both of them
   dominate the answer. Do not quietly replace them with an ideal integrator.
4. **When a metric jumps or a gate fails, stop and diagnose.** Do not continue
   and report it later. Two real bugs have already been found this way, and both
   looked like physics until they were read carefully.
5. **Search before proposing.** If you think something is impossible or already
   solved, check first, then say what you found.

## Where things are

```
loopback/    the characterization rig — what the cable actually gives you
  model.py     the path, modelled honestly: DAC, AC coupling, passive RC, ADC
  rig.py       the measurement stages and the sweep. Run this first, always.
docs/
  INHERITED_CONTEXT.md   the measured ground truth carried over from
                         timing-substrate. Treat as fact.
tests/       tests of the instrument, not of the cable
results/     rig output (gitignored — it is machine-specific)
```

## What to do first

Run the rig. Everything else depends on its numbers:

```bash
python3 -m loopback.rig --simulate          # no hardware, model the path
python3 -m loopback.rig --device 1          # measure the real cable
```

Do not write a layer of GPT-2 into the cable before the rig says how many
effective bits one dot product resolves and how many slots the channel can carry.
Those two numbers decide the whole design.

## Target hardware

A MacBook, its built-in codec, and ~$2 of passives. Built-in audio is typically
44.1/48 kHz — do not assume 192 kHz without checking; the rig reports what the
device actually accepted.

## Tone

The owner is a visual-spatial thinker with deep physics and music intuition and no
formal ML background. Explain in terms of the machine — pulses, charge, clocks,
accumulators, what the wire does — not in ML jargon. When something he says does
not parse, ask what he means rather than assuming the less charitable reading.
Build first, evaluate second.
