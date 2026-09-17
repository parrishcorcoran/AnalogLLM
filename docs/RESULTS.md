# Results

Measured only. Each entry says what was measured, on what, and what it does not
show. Tags: `[MEASURED]` on hardware · `[SIMULATED]` from a model ·
`[DERIVED]` arithmetic on those.

---

## Flowing settles exactly at depth `[SIMULATED]`

`experiments/flow_settling.py` — 12-deep stack, width 64, random weights,
MLP-only blocks. Every block fires every tick, each reading what the block
before it emitted on the **previous** tick.

```
tick 11   1.724e+01
tick 12   0.000e+00   <-- EXACT
tick 13   0.000e+00
tick 14   0.000e+00
```

**It converges to the staged answer exactly at tick 12, where the depth is 12.**
Error 0.0, not approximately zero, and it stays 0.0 forever after. State
movement hits exactly zero on the same tick.

Consequences:

- **"Run until the state stops moving" is a valid stopping rule**, and it stops
  at precisely the depth.
- It holds with **random** weights, so this is not a property of training. Depth
  is not a schedule — it is how far the signal has to travel.

Does not show: the early lock-in (answer correct at tick 15 of 128 on real
GPT-2). That is weight-dependent and needs trained weights to reproduce.

## Noise does not compound over ticks `[SIMULATED]`

Same stack, with independent noise added to every value on every tick — the
analog case.

| noise/tick | ~bits | err @ tick 12 | err @ tick 60 | growth |
|---|---|---|---|---|
| 1e-4 | 13 | 2.92e-4 | 3.14e-4 | **1.08×** |
| 1e-3 | 10 | 2.92e-3 | 3.14e-3 | **1.08×** |
| 1e-2 | 7 | 2.91e-2 | 3.14e-2 | **1.08×** |
| 1e-1 | 3 | 2.81e-1 | 3.07e-1 | **1.09×** |

**Running five times longer than needed grows the error 8%, not 500%.** The
error does not random-walk. LayerNorm pulls the state back onto its manifold
every tick — the floor cleans at the rate the arithmetic dirties.

This is the property an analog substrate needs and usually does not get: **time
is free.** Letting the machine run longer costs nothing in accuracy.

### The design number `[DERIVED]`

Error settles at **~2.9× the per-tick noise**, linear across four orders of
magnitude. So:

> a channel delivering **N bits** per value yields a settled answer with roughly
> **N − 1.5 bits**

8-bit channels (osmo-fl2k, video capture) therefore land around **6.5 effective
bits** at the output.

Does not show: behaviour with attention in the loop, at GPT-2's width and depth,
or with correlated rather than independent noise. Real channel noise is
correlated; this used white noise.
