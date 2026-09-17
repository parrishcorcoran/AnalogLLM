# CLAUDE.md — read this before doing anything in this repo

## What this project is

An **analog GPT-2**: the model run as pulses down a real wire, with the multiply
and the sum performed by physics rather than by arithmetic. The wire, for now, is
an audio cable from the laptop's output back into its input.

This repo is the analog track of the timing-substrate work.

**On the inherited documents — read this carefully.** The prose in
`timing-substrate/specs/` and `docs/INHERITED_CONTEXT.md` was written by Claude,
not by the owner. The CODE ran and the numbers came out of it, so treat the
numbers as results to re-check. But the framings, the conclusions, the tone
notes and the claims about what the owner thinks are Claude's, and several were
wrong. Do NOT quote those documents back to the owner as his own position.
Ask him.

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

**An array of modems plus FPGA modules.** Many independent analog channels in
parallel, clocked and orchestrated by the FPGAs. Channel count is the point —
it is the answer to the parallelism limit, not a workaround for it.

The owner's computer is an HP. There is no EPYC server, no Strix Halo and no
MacBook — earlier documents asserted all three and were wrong. Do not assume
hardware. Ask.

The audio-jack loopback in `loopback/` is a single-channel bench instrument for
checking whether charge on a wire stays linear. It is not the target.

## Tone

The owner is a visual-spatial thinker with deep physics and music intuition and no
formal ML background. Explain in terms of the machine — pulses, charge, clocks,
accumulators, what the wire does — not in ML jargon. When something he says does
not parse, ask what he means rather than assuming the less charitable reading.
Build first, evaluate second.
