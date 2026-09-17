#!/usr/bin/env python3
"""Bench sequence for a $5 FL2000 dongle, a coax run and a 12-bit scope.

Three measurements, in order. Each one is a claim from docs/NOTES_AND_INTERVALS
that has never been tested outside audio.

  0. HOW MANY BITS SURVIVE.  The FL2000 is 8 bits per channel, so 4096 levels
     means two channels summed through a 16:1 resistor pair -- one carries the
     top 8 bits, one carries the bottom 4. The ratio only has to hold to 1/16
     = 6% for the last bit to mean anything, so ordinary 1% parts have six
     times the margin needed and the DAC's own linearity is the suspect.
     Whether 12 clean bits actually come out is a measurement, not a theory.

  1. DOES THE WIRE ADD.  Two sources joined into one coax. If the scope sees
     A+B to 12 bits, the medium is doing the accumulate -- which is the half
     of multiply-accumulate that a digital machine pays for twice
     (NOTES_AND_INTERVALS section 7).

  2. IS A LENGTH A ROTATION.  Time-of-flight, not a phase sweep -- a DHO804
     samples at 800 ps, so a 10 m piece with the far end open gives a 100 ns
     round trip, 125 samples, velocity factor to 0.8%. A 5 cm piece is under
     one sample and cannot be measured directly. Measure the long one and
     scale. That turns "5 cm is one notch of a 4096 dial at 977 kHz" from
     arithmetic off a textbook 0.66 into a number for YOUR cable.

Generate the transmit files here, play them with osmo-fl2k, capture on the
scope, bring the CSV back:

    python3 experiments/bench_coax.py gen  out/
    fl2k_file -s 100e6 -d 0 out/step0_msb.u8 -c out/step0_lsb.u8
    python3 experiments/bench_coax.py step0 capture.csv
    python3 experiments/bench_coax.py step1 capture.csv

The CSV wants one column of samples, or two with time first; a plain scope
export works. `grab` pulls it straight off a DHO800 over LAN instead.

Settings for a DHO804 (70 MHz, 1.25 GSa/s, 25 Mpts, 12 bit, 4 ch):

  step 0  the ramp is 1.67 ms and wants 2.09 Mpts of the 25 available. Each
          level is held 408 ns against a 5 ns rise time, 82x settled, so the
          70 MHz does not bite here -- bandwidth only matters on fast edges.
  step 1  put A on CH1, B on CH2 and the summing node on CH3. The truth then
          comes off the scope instead of a saved file, which removes the
          alignment problem entirely.
  step 2  10 m piece, far end open, single shot on the edge.

The scope is 12 bit, which is the dial exactly. Use :WAV:FORM WORD, not BYTE,
or you throw away four of them at the last step.
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loopback.model import effective_bits, fit_linear   # noqa: E402

LEVELS = 4096
HOLD = 64              # samples per level, so the scope has something to average


def split(v):
    """4096-level value -> (high byte, low byte) for a 16:1 resistive sum.

    High channel carries bits 11..4 and is weighted 16x; low channel carries
    bits 3..0 left-justified. Recombined value is hi * 16 + (lo >> 4).
    """
    v = np.asarray(v, int)
    return (v >> 4).astype(np.uint8), ((v & 0xF) << 4).astype(np.uint8)


def gen(outdir):
    os.makedirs(outdir, exist_ok=True)
    # step 0: every level once, in order, so INL and missing codes both show
    ramp = np.repeat(np.arange(LEVELS), HOLD)
    msb, lsb = split(ramp)
    msb.tofile(os.path.join(outdir, "step0_msb.u8"))
    lsb.tofile(os.path.join(outdir, "step0_lsb.u8"))

    # step 1: two operands whose sum is known. A sweeps, B is a fixed pattern.
    rng = np.random.default_rng(0)
    a = np.repeat(rng.integers(0, LEVELS, 512), HOLD)
    b = np.repeat(rng.integers(0, LEVELS, 512), HOLD)
    for nm, arr in (("a", a), ("b", b)):
        m, l = split(arr)
        m.tofile(os.path.join(outdir, "step1_%s_msb.u8" % nm))
        l.tofile(os.path.join(outdir, "step1_%s_lsb.u8" % nm))
    np.save(os.path.join(outdir, "step1_truth.npy"),
            np.stack([a[::HOLD], b[::HOLD]]))
    print("wrote to %s/  -- step0 is %d levels x %d samples" % (outdir, LEVELS, HOLD))
    print("step1 truth saved; keep it, step1 needs it")


def grab(host, chan=1, points=2_500_000):
    """Pull a capture off a DHO800 over LAN. Untested here -- no scope."""
    import socket
    sk = socket.create_connection((host, 5555), timeout=10)

    def cmd(t, read=False):
        sk.sendall((t + "\n").encode())
        if not read:
            return None
        buf = b""
        while not buf.endswith(b"\n"):
            buf += sk.recv(1 << 20)
        return buf

    cmd(":STOP")
    cmd(":WAV:SOUR CHAN%d" % chan)
    cmd(":WAV:MODE RAW")
    cmd(":WAV:FORM WORD")                      # 12 bit; BYTE throws away four
    pre = cmd(":WAV:PRE?", read=True).decode().split(",")
    yinc, yorig, yref = float(pre[7]), float(pre[8]), float(pre[9])

    out = []
    step = 250_000
    for a in range(1, points + 1, step):
        cmd(":WAV:STAR %d" % a)
        cmd(":WAV:STOP %d" % min(a + step - 1, points))
        sk.sendall(b":WAV:DATA?\n")
        head = sk.recv(2)
        n = int(sk.recv(int(chr(head[1]))))
        raw = b""
        while len(raw) < n:
            raw += sk.recv(1 << 20)
        out.append(np.frombuffer(raw[:n], dtype="<u2"))
    sk.close()
    return (np.concatenate(out).astype(float) - yorig - yref) * yinc


def load(path):
    d = np.loadtxt(path, delimiter=",", skiprows=0, ndmin=2)
    return d[:, -1]              # last column is the signal either way


def blocks(sig, n):
    """Chop a capture into n equal blocks and take the middle of each."""
    e = np.array_split(sig, n)
    return np.array([b[len(b)//4: -len(b)//4 or None].mean() for b in e])


def step0(path):
    got = blocks(load(path), LEVELS)
    want = np.arange(LEVELS, dtype=float)
    _, _, r2 = fit_linear(got, want)
    bits = effective_bits(r2)
    d = np.diff(got)
    print("levels sent        %d" % LEVELS)
    print("R^2 against a ramp %.6f" % r2)
    print("EFFECTIVE BITS     %.2f" % bits)
    print("monotonic          %s  (%d steps go backwards)" % (
        "yes" if (d > 0).all() else "NO", int((d <= 0).sum())))
    print("worst step         %.2f LSB  (should be 1.00)" % (d.max()/np.median(d)))
    print()
    print("backwards steps landing every 16 levels means the 16:1 ratio is off;")
    print("backwards steps scattered anywhere means the DAC's own linearity.")


def step1(path, truthdir="out"):
    truth = np.load(os.path.join(truthdir, "step1_truth.npy"))
    want = truth[0].astype(float) + truth[1].astype(float)
    got = blocks(load(path), len(want))
    _, _, r2 = fit_linear(got, want)
    print("pairs              %d" % len(want))
    print("R^2 of A+B         %.6f" % r2)
    print("EFFECTIVE BITS     %.2f" % effective_bits(r2))
    print()
    print("this is the accumulate. if it holds at 12 bits the wire is doing")
    print("the half of multiply-accumulate that costs a CPU 0.9 pJ per term")
    print("and a DRAM fetch 640.")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    cmd, arg = sys.argv[1], sys.argv[2]
    {"gen": gen, "step0": step0, "step1": step1}[cmd](arg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
