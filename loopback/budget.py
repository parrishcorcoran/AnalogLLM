#!/usr/bin/env python3
"""budget.py -- how fast can audio possibly go?

There are two completely different questions hiding in "how fast is audio", and
they have answers four orders of magnitude apart. Keeping them separate is the
whole point of this file.

    SERIAL   -- what a cable can do. Every multiply-accumulate has to take its
                turn down the wire, so the limit is samples per second and
                nothing else. This is what you can build today.

    PARALLEL -- what an audio-rate CLOCK implies for the substrate the cable is
                a model of: one wire per weight, every line live at once. Here
                the MAC count does not appear at all. The limit is ticks per
                token, which the flowing machine measured at depth + cycle.

The cable is slow. The clock is not. The entire gap between them is parallelism,
which is exactly the distinction PARALLELISM_AND_THROUGHPUT.pdf was written to
make: tick-operations are total work, time-in-ticks is how long you wait.

All figures [DERIVED] -- arithmetic over the measured numbers in
docs/INHERITED_CONTEXT.md. Nothing here has touched hardware.
"""
from __future__ import annotations

GPT2_MACS_PER_TOKEN = 124_000_000    # ~124M params, each used once per token
GPT2_LINEAR_STAGES = 48              # 12 blocks x 4 linear layers
GPT2_DEPTH = 12                      # blocks, for the flowing arrangement
PDM_RATE = 1024                      # oversampling at which PDM was bit-identical

# What audio hardware actually carries, as total samples per second. Channel
# count times sample rate is the only thing that matters -- a tick is a sample,
# whichever wire it arrives on.
LINKS = [
    ("built-in jack, mono",        1,    48_000),
    ("built-in jack, stereo",      2,    48_000),
    ("built-in at 96k, stereo",    2,    96_000),
    ("USB interface, 2ch @ 192k",  2,   192_000),
    ("USB interface, 8ch @ 192k",  8,   192_000),
    ("MADI, 64ch @ 48k",          64,    48_000),
    ("Dante on 1GbE, ~512ch",    512,    48_000),
]

# What a tick buys, depending on how the activation is coded. This is the one
# real dial: the wire multiplies by holding an amplitude for a duration, so the
# duration IS the activation's resolution, and it is paid for in ticks.
CODINGS = [
    ("binary activation (1 bit)",        1),
    ("3-bit activation (rig's point)",   8),
    ("6-bit activation",                64),
]


def serial_rate(channels: int, fs: int, ticks_per_mac: int) -> float:
    """MACs per second down a wire. One slot carries one MAC."""
    return channels * fs / ticks_per_mac


def human_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1e3:.0f} ms"
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds/60:.1f} min"
    if seconds < 86400 * 2:
        return f"{seconds/3600:.1f} h"
    return f"{seconds/86400:.1f} days"


def parallel_tokens_per_s(fs: float, flowing: bool = True) -> float:
    """One wire per weight: the MAC count vanishes, only ticks per token remain.

    Flowing -- every stage live on the same clock -- costs the converter's own
    cycle plus the depth, because depth becomes an addition rather than a
    multiplication. Staged costs one full cycle per stage.
    """
    ticks = (PDM_RATE + GPT2_DEPTH) if flowing else (PDM_RATE * GPT2_LINEAR_STAGES)
    return fs / ticks


def main() -> None:
    print("=" * 78)
    print("  SERIAL -- GPT-2 down an audio cable. Every MAC takes its turn.")
    print("=" * 78)
    for name, ticks in CODINGS:
        print(f"\n  {name}: {ticks} tick(s) per MAC")
        print(f"    {'link':28s} {'samples/s':>12s} {'MAC/s':>12s} {'per token':>12s}")
        print(f"    {'-'*28} {'-'*12} {'-'*12} {'-'*12}")
        for link, ch, fs in LINKS:
            macs = serial_rate(ch, fs, ticks)
            print(f"    {link:28s} {ch*fs:12,d} {macs:12,.0f} "
                  f"{human_time(GPT2_MACS_PER_TOKEN / macs):>12s}")

    print()
    print("=" * 78)
    print("  PARALLEL -- one wire per weight, clocked at an audio rate.")
    print("  The MAC count does not appear. Only ticks per token.")
    print("=" * 78)
    print(f"\n    flowing: {PDM_RATE} + {GPT2_DEPTH} = {PDM_RATE + GPT2_DEPTH:,} ticks/token")
    print(f"    staged:  {PDM_RATE} x {GPT2_LINEAR_STAGES} = "
          f"{PDM_RATE * GPT2_LINEAR_STAGES:,} ticks/token\n")
    print(f"    {'tick rate':28s} {'staged tok/s':>14s} {'flowing tok/s':>15s}")
    print(f"    {'-'*28} {'-'*14} {'-'*15}")
    for label, fs in [("48 kHz  (built-in audio)", 48_000),
                      ("192 kHz (good interface)", 192_000),
                      ("768 kHz (top audio ADC)", 768_000),
                      ("1 MHz   (FPGA, the plan)", 1_000_000),
                      ("1 GHz   (copper geometry)", 1_000_000_000)]:
        print(f"    {label:28s} {parallel_tokens_per_s(fs, False):14,.1f} "
              f"{parallel_tokens_per_s(fs, True):15,.0f}")

    best = max(serial_rate(ch, fs, 1) for _, ch, fs in LINKS)
    print()
    print("=" * 78)
    print("  THE ANSWER")
    print("=" * 78)
    print(f"""
    Serial ceiling, best case, every favourable assumption granted -- the
    biggest audio network in the building and a 1-bit activation:

        {best:,.0f} MAC/s  ->  {human_time(GPT2_MACS_PER_TOKEN/best)} per token
                                  ({best/GPT2_MACS_PER_TOKEN:.2f} tok/s)

    The same laptop runs the software water machine at 23.8 tok/s [MEASURED].
    So audio as a transport is about {23.8/(best/GPT2_MACS_PER_TOKEN):.0f}x SLOWER than just doing it
    in software. No encoding fixes this. The ceiling is samples per second,
    and audio hardware does not have more samples per second.

    But the CLOCK is not the problem. At 48 kHz, a flowing machine with one
    wire per weight would turn out {parallel_tokens_per_s(48_000):,.0f} tokens per second, because
    depth is an addition and the MAC count never enters. The audio tick is
    already fast enough. What audio does not have is {65_280:,} wires.

    So the cable is not a slow computer. It is a correctly-clocked model of
    the real machine with the parallelism removed -- which is the cheapest
    honest way to test whether charge on a wire is linear enough to hold a
    transformer, at the exact tick rate the microsecond substrate will use.
""")


if __name__ == "__main__":
    main()
