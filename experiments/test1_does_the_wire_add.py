#!/usr/bin/env python3
"""TEST 1. Does the wire do the addition?

The claim, from docs/NOTES_AND_INTERVALS section 7: a digital machine pays
twice for a multiply-accumulate -- 0.9 pJ for the add and 640 pJ to fetch the
operand -- and a medium does the accumulate for nothing, because two signals in
the same place at the same time are already summed. Nothing performs it.

This is the smallest measurement that tests that. Play A out the left channel
and B out the right, join them, record the sum, ask whether it equals A+B to
twelve bits.

WIRING
    left out  ---[ R ]---+
                         +--- input
    right out ---[ R ]---+

    Two equal resistors, any value from 1k to 10k, matched by eye. A bare
    Y-cable also works and is louder, but it shorts the two outputs together
    and some interfaces dislike it; resistors are safer and cost nothing.

WHY AUDIO AND NOT THE FAST HARDWARE
    A 24-bit codec is exactly dial-plus-lap -- 2^24 -- so one sample holds a
    whole number in this system. The $5 FL2000 is 8 bits per channel and needs
    two channels summed through a resistor network just to reach the dial.
    Audio has more bits than the fast part and is already built. The only thing
    it cannot do is see a cable delay, because a 48 kHz wavelength is 4.2 km of
    coax -- that is test 2, and that one needs the scope.

THE CONTROL THAT MATTERS
    Each channel is also measured alone. If the sum is bad but the singles are
    clean, the addition is at fault. If a single is bad, the channel is, and the
    sum was never going to work. Without this the result is unreadable.

    python3 experiments/test1_does_the_wire_add.py --sim      # no hardware
    python3 experiments/test1_does_the_wire_add.py --device 3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loopback.model import effective_bits, fit_linear, quantize              # noqa: E402
from loopback.rig import PREAMBLE_TICKS, GUARD_TICKS, TAIL_PAD, align       # noqa: E402

N_PAIRS = 512
SLOT = 32          # ticks each value is held, so the path settles on it
FS = 48_000


class StereoSim:
    """A CODE CHECK, not a prediction of the hardware.

    Deliberately a clean channel -- a gain, a little mismatch, 16-bit
    quantisation -- so that a poor number here means the analysis is broken
    rather than the physics being hard. loopback/model.py's path is a leaky
    capacitor, which the repo says plainly is not this architecture, and using
    it here would bury the thing being measured under a bug we already know
    about.
    """

    def __init__(self, seed=0, bits=16):
        self.rng = np.random.default_rng(seed)
        self.bits = bits

    def play(self, left, right):
        l = 0.5000 * np.asarray(left, float)
        r = 0.4950 * np.asarray(right, float)        # 1% channel mismatch
        return quantize(l + r, self.bits)


class StereoDevice:
    """Left and right carry DIFFERENT signals. DeviceLink tiles one to both."""

    def __init__(self, device, fs=FS):
        import sounddevice as sd
        self.sd, self.device, self.fs = sd, device, fs

    def play(self, left, right):
        out = np.stack([np.asarray(left, np.float32),
                        np.asarray(right, np.float32)], axis=1)
        rec = self.sd.playrec(out, samplerate=self.fs, channels=1,
                              device=self.device, blocking=True)
        return rec[:, 0].astype(np.float64)


def framed(link, left, right):
    """Preamble, guard, payload, tail -- same framing the rest of the rig uses."""
    pre = np.concatenate([np.ones(PREAMBLE_TICKS), np.zeros(GUARD_TICKS)])
    tail = np.zeros(TAIL_PAD)
    l = np.concatenate([pre, left, tail])
    r = np.concatenate([pre, right, tail])
    return align(link.play(l, r), len(left))


def hold(v):
    return np.repeat(v, SLOT)


def levels(rec):
    """One number per slot, from the settled second half of each."""
    b = rec[:len(rec) // SLOT * SLOT].reshape(-1, SLOT)
    return b[:, SLOT // 2:].mean(axis=1)


def report(name, got, want):
    m, _, r2 = fit_linear(got, want)
    print("  %-26s R2 %.6f    %.2f bits" % (name, r2, effective_bits(r2)))
    return m, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--device", type=int)
    a = ap.parse_args()
    if not a.sim and a.device is None:
        print(__doc__)
        return 1

    link = StereoSim() if a.sim else StereoDevice(a.device)
    rng = np.random.default_rng(1)
    A = rng.uniform(-0.45, 0.45, N_PAIRS)
    B = rng.uniform(-0.45, 0.45, N_PAIRS)
    zero = np.zeros(N_PAIRS)

    print("\n%s, %d pairs, %d ticks per value\n" % (
        "simulated" if a.sim else "device %d" % a.device, N_PAIRS, SLOT))

    print("controls -- each channel alone:")
    ga, _ = report("left only", levels(framed(link, hold(A), hold(zero))), A)
    gb, _ = report("right only", levels(framed(link, hold(zero), hold(B))), B)

    print("\nthe question -- both at once:")
    both = levels(framed(link, hold(A), hold(B)))
    _, r2 = report("left + right", both, ga * A + gb * B)

    print("\nchannel gains %.4f and %.4f, mismatch %.1f%%" % (
        ga, gb, 100 * abs(ga - gb) / max(abs(ga), abs(gb))))
    print("\n%.2f bits on the sum. twelve or better means the wire is doing" %
          effective_bits(r2))
    print("the accumulate. if the singles are clean and this is not, the")
    print("addition is what failed -- which is the whole point of the test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
