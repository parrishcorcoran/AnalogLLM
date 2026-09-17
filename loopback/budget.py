#!/usr/bin/env python3
"""budget.py -- how fast can audio possibly go?

Cycle speed sets the rate. What it gets multiplied BY is set by how many weights
are physically present in the circuit, and that spans five orders of magnitude.
So the honest answer is a ladder, not a number.

    tok/s  =  cycle_rate / ticks_per_token

Cycle rate is the audio clock. Ticks-per-token is the tier below. Every row
scales linearly with the clock -- doubling the sample rate doubles every number
in this file. That is why cycle speed is everything. But the tier sets the
constant, and the constant spans 1e5.

THE THING THAT SETS THE TIER: fanout.

One input value in GPT-2 drives 1,892 multiply-accumulates on average -- it
reaches every output neuron of its layer through a different weight. If the
weights exist as physical components, one pulse down the wire performs all 1,892
at once, for free, because the wire fans out into them. If there is one resistor
and one capacitor, that pulse performs exactly one, and the other 1,891 have to
be transmitted separately.

That is the whole ladder. Not the clock -- the clock is the same in every row.

GPT-2 (124M) constants below are derived from the config in derive(), not quoted,
and n_in reproduces the 65,280 in WATER_TO_CONVERTER.pdf as a check.

All figures [DERIVED] from the measured numbers in docs/INHERITED_CONTEXT.md.
Nothing here has touched hardware.
"""
from __future__ import annotations

N_EMBD, N_LAYER, VOCAB = 768, 12, 50257
LAYERS = [("c_attn", 768, 2304), ("attn.c_proj", 768, 768),
          ("mlp.c_fc", 768, 3072), ("mlp.c_proj", 3072, 768)]

PDM_RATE = 1024      # oversampling at which PDM went bit-identical to fp32
GPT2_DEPTH = 12      # blocks, for the flowing arrangement
LINEAR_STAGES = 48   # 12 blocks x 4 linear layers, for the staged arrangement
SOFTWARE_TOK_S = 23.8   # water.py on the owner's laptop [MEASURED]


def derive() -> dict:
    """Every GPT-2 constant this file uses, from the config."""
    macs = sum(i * o for _, i, o in LAYERS) * N_LAYER + N_EMBD * VOCAB
    n_in = sum(i for _, i, _ in LAYERS) * N_LAYER + N_EMBD
    n_out = sum(o for _, _, o in LAYERS) * N_LAYER + VOCAB
    return {"macs": macs, "n_in": n_in, "n_out": n_out, "fanout": macs / n_in}


G = derive()


# ---------------------------------------------------------------- the ladder

def ticks_one_accumulator(slot_ticks: int = 1, **_) -> int:
    """TIER 1 -- one resistor, one cap. The $2 circuit.

    There is a single accumulator, so every multiply-accumulate has to be
    transmitted on its own. The wire's fanout is one. This is the tier the
    loopback rig measures, and it is the slowest arrangement that exists.
    """
    return G["macs"] * slot_ticks


def ticks_crossbar(slot_ticks: int = 1, emit_channels: int = 1,
                   read_channels: int = 1, **_) -> int:
    """TIER 2 -- weights as physical components. One wire, many resistors.

    The emitting wire fans out into one resistor per output neuron, each into
    its own cap. One pulse train of n_in slots now performs every MAC in the
    layer simultaneously -- the 1,892x is recovered, and it is recovered by
    copper, not by cleverness.

    What does not come free is the I/O: n_in values have to be emitted and
    n_out caps have to be read back, each through however many channels the
    interface has. Emitting and reading run at the same time on a duplex
    interface, so the cost is whichever side is slower. Note what this means --
    once the weights are physical, CHANNEL COUNT is the lever, not sample rate.
    """
    emit = G["n_in"] * slot_ticks / max(1, emit_channels)
    read = G["n_out"] / max(1, read_channels)
    return int(max(emit, read))


def ticks_substrate(flowing: bool = True, **_) -> int:
    """TIER 3 -- one wire per weight, every line live. The real machine.

    Neither the MAC count nor the neuron count appears. Depth stops multiplying
    and starts adding, which is the flowing result: cycle + depth, measured at
    lock-in tick 15 of 128 on real GPT-2.
    """
    return (PDM_RATE + GPT2_DEPTH) if flowing else (PDM_RATE * LINEAR_STAGES)


CLOCKS = [
    ("built-in jack", 48_000),
    ("built-in at 96k", 96_000),
    ("USB interface 192k", 192_000),
    ("top audio ADC 768k", 768_000),
    ("DSD64 bitstream 2.8M", 2_822_400),
    ("DSD512 bitstream 22.6M", 22_579_200),
]


def human(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds*1e6:.0f} us"
    if seconds < 1:
        return f"{seconds*1e3:.1f} ms"
    if seconds < 90:
        return f"{seconds:.2f} s"
    if seconds < 5400:
        return f"{seconds/60:.1f} min"
    if seconds < 86400 * 2:
        return f"{seconds/3600:.1f} h"
    return f"{seconds/86400:.0f} days"


def row(label: str, ticks: int) -> None:
    """One tier across every clock. Below 1 tok/s the useful unit is time."""
    cells = []
    for _, fs in CLOCKS:
        tps = fs / ticks
        cells.append(f"{tps:>10,.0f}" if tps >= 10 else
                     f"{tps:>10,.2f}" if tps >= 1 else
                     f"{human(ticks / fs):>10s}")
    print(f"  {label:34s}" + "".join(cells))


def main() -> None:
    print(f"""
GPT-2 (124M), derived from config:
    {G['macs']:>12,}  multiply-accumulates per token
    {G['n_in']:>12,}  input values   (WATER_TO_CONVERTER.pdf says 65,280 -- matches)
    {G['n_out']:>12,}  output values
    {G['fanout']:>12,.0f}  MACs driven by one input value  <-- the factor the tier decides
""")
    print("=" * 96)
    print("  TOKENS PER SECOND. Every row scales linearly with the clock.")
    print("=" * 96)
    print(f"  {'':34s}" + "".join(f"{n.split()[-1]:>10s}" for n, _ in CLOCKS))
    print(f"  {'tier':34s}" + "".join(f"{'':>10s}" for _ in CLOCKS))
    print("  " + "-" * 94)

    print("  DRAFT precision -- 1 tick per value, so a 1-bit activation.")
    row("1  one cap", ticks_one_accumulator(1))
    row("2  crossbar, 1 out + 1 in", ticks_crossbar(1, 1, 1))
    row("2  crossbar, 4 out + 4 in", ticks_crossbar(1, 4, 4))
    row("2  crossbar, 256 out + 256 in", ticks_crossbar(1, 256, 256))
    print()
    print(f"  EXACT precision -- {PDM_RATE} ticks per value, where PDM went")
    print("  bit-identical to fp32 [MEASURED]. This is the like-for-like row.")
    row("1  one cap", ticks_one_accumulator(PDM_RATE))
    row("2  crossbar, 1 out + 1 in", ticks_crossbar(PDM_RATE, 1, 1))
    row("2  crossbar, 4 out + 4 in", ticks_crossbar(PDM_RATE, 4, 4))
    row("2  crossbar, 256 out + 256 in", ticks_crossbar(PDM_RATE, 256, 256))
    row("3  substrate, staged", ticks_substrate(flowing=False))
    row("3  substrate, flowing", ticks_substrate(flowing=True))

    # Compare like for like: every tier at the precision PDM needs to be exact.
    t1 = ticks_one_accumulator(PDM_RATE)
    t2 = ticks_crossbar(PDM_RATE, 4, 4)
    t2big = ticks_crossbar(PDM_RATE, 256, 256)
    t3 = ticks_substrate()
    print(f"""
{'=' * 96}
  WHAT THE LADDER COSTS, AND WHAT IT BUYS
{'=' * 96}

  tier 1 -> tier 2   {t1/t2:>12,.0f}x     one resistor per output neuron instead of one
                                   resistor, on a 4-in/4-out interface. The wire
                                   does the fanout; the interface does the rest.
  tier 2 -> tier 3   {t2/t3:>12,.0f}x     one wire per weight instead of one wire.
  tier 1 -> tier 3   {t1/t3:>12,.0f}x

  Software reference: the water machine runs GPT-2 at {SOFTWARE_TOK_S} tok/s on the
  owner's laptop [MEASURED]. All rows below are at PDM-exact precision:

      tier 2, 4 out + 4 in  @ 192 kHz : {192_000/t2:>8,.2f} tok/s
      tier 2, 256 + 256     @ 48 kHz  : {48_000/t2big:>8,.2f} tok/s
      tier 2, 256 + 256     @ DSD512  : {22_579_200/t2big:>8,.1f} tok/s
      tier 3, flowing       @ 48 kHz  : {48_000/t3:>8,.1f} tok/s

  So cycle speed IS everything -- within a tier. Doubling the clock doubles
  every number above. But a tier is worth more than any clock audio can offer:
  going from one cap to a crossbar is worth {t1/t2:,.0f}x, and the entire span of audio
  clock rates, 48 kHz to DSD512, is worth {22_579_200/48_000:.0f}x. Both matter. The tier
  matters more, and only one of them is a wiring change.

  The catch, stated plainly: tier 2 needs {G['n_out']:,} physical weight elements
  to hold one token's worth of layers, and they have to be reprogrammable
  between tiles. Fixed resistors give you one layer, not a model. A
  programmable conductance that holds its value is exactly the component the
  handoff already flags as the open problem -- so the ladder is real, and the
  rung above tier 1 is a hardware build, not a wiring change.
""")


if __name__ == "__main__":
    main()
