#!/usr/bin/env python3
"""rig.py -- measure what the audio cable actually gives you.

Before any layer of GPT-2 is computed in copper, six numbers have to exist,
because every accuracy claim downstream rests on them:

    1. tick rate      what sample rate the codec really runs at. One tick = one
                      sample. This is the clock of the whole machine.
    2. latency        how many ticks from emitting to hearing. Alignment.
    3. noise floor    the input-referred noise, in bits. The accuracy ceiling.
    4. memory         how much of the START of a train still counts by the END
                      of it. An accumulator scores 1.00. Anything less is the
                      network forgetting, and it bounds the vector length.
    5. effective bits what the readback of a real dot product resolves, swept
                      over vector length and slot width.

Numbers 4 and 5 are the ones that decide whether this works, and they pull
against each other. The network only sums cleanly while the train is short
compared to its memory -- but a short train deposits little charge, and then the
noise floor from 3 eats the answer. The sweep finds where those two curves
cross. That crossing is the machine's operating point.

USAGE
    python3 -m loopback.rig --simulate               # no hardware, model the path
    python3 -m loopback.rig --list                   # find your device index
    python3 -m loopback.rig --device 1 --fs 48000    # measure the real cable

THE CIRCUIT (from timing-substrate/analog/analog_dot.py, ~$2)

    LEFT out ---[ R ]---+--- LINE IN
                        |
                      [ C ]
                        |
    GROUND -------------+

Start with R=10k, C=10nF (tau=100us). The rig will tell you if that is wrong for
your rate -- at 48 kHz that tau is only 4.8 ticks of history, which is almost
certainly too short, and the sweep will say so in the linearity column.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path as FsPath

import numpy as np

from .model import (PathConfig, build_train, effective_bits, fit_linear,
                    run_path, true_dot)

PREAMBLE_TICKS = 64      # full-scale burst that marks where the train begins
GUARD_TICKS = 256        # silence between preamble and train, so they do not blur
TAIL_PAD = 320           # slack after the train, so detection slip never truncates it
GPT2_MACS_PER_TOKEN = 124_000_000    # ~124M params, each used once per token


# ------------------------------------------------------------------- the link

class SimLink:
    """The cable, simulated. Same framing and alignment path as the real one."""

    measured = False

    def __init__(self, cfg: PathConfig, integrator: str = "passive", seed: int = 0):
        self.cfg = cfg
        self.fs = cfg.fs
        self.integrator = integrator
        self.rng = np.random.default_rng(seed)

    def transmit(self, frame: np.ndarray) -> np.ndarray:
        return run_path(frame, self.cfg, self.integrator, self.rng)

    def send(self, train, raw: bool = False):
        return _send(self, train, raw)


class DeviceLink:
    """The cable, real. Plays the frame and records it coming back."""

    measured = True

    def __init__(self, device, fs: float, channels_out: int = 2):
        import sounddevice as sd
        self.sd = sd
        self.device = device
        self.fs = fs
        self.channels_out = channels_out

    def transmit(self, frame: np.ndarray) -> np.ndarray:
        out = np.tile(frame.astype(np.float32)[:, None], (1, self.channels_out))
        rec = self.sd.playrec(out, samplerate=self.fs, channels=1,
                              device=self.device, blocking=True)
        return rec[:, 0].astype(np.float64)

    def send(self, train, raw: bool = False):
        return _send(self, train, raw)


def _send(link, train, raw: bool = False) -> np.ndarray:
    """Frame a train, push it down the path, return the window it occupied.

    raw=True skips the preamble and the alignment and hands back the whole
    recording -- used for the noise floor and the latency probe, which must not
    have a preamble decaying into them.
    """
    train = np.asarray(train, dtype=np.float64)
    if raw:
        return link.transmit(train)
    frame = np.concatenate([
        np.ones(PREAMBLE_TICKS),
        np.zeros(GUARD_TICKS),
        train,
        np.zeros(TAIL_PAD),
    ])
    return align(link.transmit(frame), len(train))


def align(rec: np.ndarray, want_len: int) -> np.ndarray:
    """Find the preamble, skip the guard, return exactly the train window.

    The preamble is the only full-scale thing in the frame, so a threshold on it
    is enough. Detection lands a tick or two late because the network takes time
    to charge; that slip is the same on every capture, so it shows up as a
    constant scale on the readback and the linear fit absorbs it. Jitter would
    not be absorbed -- that is a hardware property, and the sweep is where it
    would show up.
    """
    peak = float(np.max(np.abs(rec))) if rec.size else 0.0
    if peak <= 0:
        return np.full(want_len, np.nan)
    hits = np.flatnonzero(np.abs(rec) > 0.5 * peak)
    if hits.size == 0:
        return np.full(want_len, np.nan)
    start = int(hits[0]) + PREAMBLE_TICKS + GUARD_TICKS
    window = rec[start:start + want_len]
    if window.size < want_len:
        window = np.concatenate([window, np.full(want_len - window.size, np.nan)])
    return window



# -------------------------------------------------------------- exponent fits

def fit_exponential(y: np.ndarray, fs: float, rising: bool,
                    lo: float = 0.05, hi: float = 0.90) -> tuple[float, float]:
    """Fit a single pole to a rise or a decay. Returns (tau_seconds, fit_r2).

    Rising:  y = V * (1 - exp(-t/tau))  ->  ln(1 - y/V) is linear in t
    Falling: y = V * exp(-t/tau)        ->  ln(y/V)     is linear in t

    Only the middle of the curve is used; the ends are where the log blows up
    and where the noise floor lives. Returns (nan, nan) if the curve is not
    clean enough to fit, rather than a confident wrong number.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size < 8 or not np.all(np.isfinite(y)):
        return float("nan"), float("nan")
    V = float(np.max(np.abs(y)))
    if V <= 0:
        return float("nan"), float("nan")
    frac = np.abs(y) / V
    keep = (frac > lo) & (frac < hi)
    if keep.sum() < 5:
        return float("nan"), float("nan")
    t = np.arange(y.size, dtype=np.float64)[keep] / fs
    z = np.log(1.0 - frac[keep]) if rising else np.log(frac[keep])
    A = np.vstack([t, np.ones_like(t)]).T
    (slope, _), *_ = np.linalg.lstsq(A, z, rcond=None)
    if slope >= 0:
        return float("nan"), float("nan")
    resid = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return float(-1.0 / slope), r2


# ------------------------------------------------------------------- measures

def measure_latency(link, fs: float, pad_ticks: int = 4096) -> dict:
    """Send one burst into silence and count the ticks until it comes back.

    On hardware this is buffer latency and it is large -- thousands of ticks. It
    does not affect the answer, because every capture finds its own preamble, but
    it sets how long a single dot product takes wall-clock, and a machine that
    waits 20 ms per neuron is a different machine from one that waits 2 ms.
    """
    probe = np.concatenate([np.zeros(pad_ticks), np.ones(PREAMBLE_TICKS),
                            np.zeros(pad_ticks)])
    rec = link.send(probe, raw=True)
    if rec.size == 0 or not np.any(np.isfinite(rec)):
        return {"ticks": float("nan"), "ms": float("nan")}
    peak = float(np.max(np.abs(rec)))
    if peak <= 0:
        return {"ticks": float("nan"), "ms": float("nan")}
    hits = np.flatnonzero(np.abs(rec) > 0.1 * peak)
    if hits.size == 0:
        return {"ticks": float("nan"), "ms": float("nan")}
    ticks = int(hits[0]) - pad_ticks
    return {"ticks": ticks, "ms": 1e3 * ticks / fs}


def measure_noise(link, ticks: int = 4096) -> dict:
    """Record silence. Whatever comes back is the floor everything sits on.

    Sent raw -- no preamble -- because a preamble decaying through the network
    would land inside the window and be counted as noise.
    """
    rec = link.send(np.zeros(ticks), raw=True)
    rec = rec[np.isfinite(rec)]
    if rec.size == 0:
        return {"rms": float("nan"), "bits": float("nan")}
    rms = float(np.sqrt(np.mean(rec ** 2)))
    bits = float("inf") if rms <= 0 else float(np.log2(1.0 / rms))
    return {"rms": rms, "bits": bits, "peak": float(np.max(np.abs(rec)))}


def measure_memory(link, fs: float, lengths: list[int]) -> list[dict]:
    """How much of the start of a train survives to the end of it.

    This is the measurement the whole machine turns on, and it needs no curve
    fitting and no model of the circuit. For a window of L ticks, put ONE unit
    pulse at the very start and read the last tick; then put the SAME pulse at
    the very end and read the last tick. An accumulator would give the same
    reading both times -- charge is charge, whenever it arrived. The ratio of
    the two is how much the network has forgotten across L ticks.

        ratio = 1.00   a true accumulator. Position does not matter.
        ratio = 0.90   the first element of the train counts 10% less than the
                       last. That is a systematic error no calibration removes,
                       because it depends on WHERE each element sat.
        ratio = 0.01   the network is a lowpass on the last few slots. There is
                       no dot product here.

    A train of n elements at s ticks each is L = n*s ticks long, so this table
    says directly how long a vector the cable can hold.
    """
    rows = []
    for L in lengths:
        early = np.zeros(L); early[0] = 1.0
        late = np.zeros(L); late[-1] = 1.0
        re = send_baselined(link, early)
        rl = send_baselined(link, late)
        if not (np.isfinite(re[-1]) and np.isfinite(rl[-1])) or rl[-1] == 0:
            rows.append({"ticks": L, "ms": 1e3 * L / fs, "early": float("nan"),
                         "late": float("nan"), "ratio": float("nan")})
            continue
        rows.append({
            "ticks": L,
            "ms": 1e3 * L / fs,
            "early": float(re[-1]),
            "late": float(rl[-1]),
            "ratio": float(re[-1] / rl[-1]),
        })
    return rows


def max_train_ticks(memory: list[dict], tolerance: float = 0.99) -> int:
    """Longest train whose first element still counts within `tolerance`."""
    ok = [r["ticks"] for r in memory
          if r["ratio"] == r["ratio"] and r["ratio"] >= tolerance]
    return max(ok) if ok else 0


def send_baselined(link, train: np.ndarray) -> np.ndarray:
    """Capture a train, minus the capture of silence in the same frame.

    Every frame carries a preamble so the receiver can find the train, and that
    preamble is still decaying through the network while the train plays. It is
    identical on every capture, so subtracting a silent capture removes it
    exactly and leaves only what the train itself did. Without this, an impulse
    response measured here is mostly the preamble's tail.
    """
    train = np.asarray(train, dtype=np.float64)
    return link.send(train) - link.send(np.zeros(len(train)))


def measure_impulse(link, L: int) -> np.ndarray:
    """The channel's impulse response over a window of L ticks.

    Put a single unit tick at the start of the window and record the whole
    window. The path is linear and time-invariant, so that recording IS the
    impulse response, and it says everything about what the cable does to a
    pulse train -- including the part the memory probe finds: after a while the
    response goes NEGATIVE, because the coupling capacitors are not allowed to
    pass a held level and must give the charge back.
    """
    train = np.zeros(L)
    train[0] = 1.0
    return send_baselined(link, train)


def reversed_response(h: np.ndarray) -> np.ndarray:
    """g[t] = h[L-1-t]: what the channel applies to a tick at position t.

    Reading the last tick of the window is an inner product between what was
    emitted and the time-reversed impulse response. So a tick early in the train
    is not weighted the same as a tick late in it. An accumulator would give a
    flat g. This returns the real one.
    """
    return h[::-1]


def usable_slots(g: np.ndarray, n: int, slot_ticks: int,
                 floor_frac: float = 0.05) -> np.ndarray:
    """Which slots the channel can actually carry an element in.

    The reversed response passes through zero somewhere in a long train -- that
    is the coupling capacitor handing back exactly as much charge as it took.
    A slot sitting on that null contributes nothing at ANY drive level, so no
    pre-scale can rescue it; dividing by it only blows up the drive and clips.

    The count of usable slots is the cable's real capacity for one dot product,
    and it is a more useful number than the train length.
    """
    means = np.array([abs(float(np.mean(g[i * slot_ticks:(i + 1) * slot_ticks])))
                      for i in range(n)])
    peak = float(means.max()) if means.size else 0.0
    return means >= floor_frac * peak if peak > 0 else np.zeros(n, dtype=bool)


def comp_scale(g: np.ndarray, n: int, slot_ticks: int,
               usable: np.ndarray | None = None) -> float:
    """One fixed drive scale for a whole sweep cell.

    The pre-scale must NOT be renormalised per dot product -- a scale that
    varies with the input is a multiplicative error on the answer, which is
    exactly the linearity being measured. So it is fixed once from the channel,
    which is the only thing it is allowed to depend on.
    """
    means = np.array([abs(float(np.mean(g[i * slot_ticks:(i + 1) * slot_ticks])))
                      for i in range(n)])
    if usable is not None:
        means = means[usable]
    means = means[means > 0]
    return float(np.min(means) * slot_ticks) if means.size else 1.0


def widths_of(x: np.ndarray, slot_ticks: int) -> np.ndarray:
    return np.clip(np.round(np.abs(x) * slot_ticks), 0, slot_ticks).astype(int)


def analog_dot(link, w: np.ndarray, x: np.ndarray, slot_ticks: int,
               g: np.ndarray | None = None, scale: float = 1.0) -> float:
    """One dot product, done in the wire. Signed weights are two passes.

    Amplitude is unipolar, so the positive weights and the negative weights each
    get their own train and the readings are subtracted. Read at the last tick
    of the train -- that is the moment the network holds the whole sum.

    With `g` (the reversed impulse response), each slot is driven at

        amp_i = scale * w_i * width_i / (slot * G_i),    G_i = sum of g over the
                                                         ticks the pulse is on

    so that the channel's own weighting cancels and what arrives at the read
    point is the dot product. This is inverting a known linear channel, the same
    move as pre-emphasis on tape -- not a fitted correction.
    """
    n = len(w)
    amp = w
    if g is not None:
        widths = widths_of(x, slot_ticks)
        amp = np.zeros(n)
        for i in range(n):
            lo = i * slot_ticks
            G = float(np.sum(g[lo:lo + widths[i]]))
            if G != 0.0 and widths[i] > 0:
                amp[i] = scale * w[i] * widths[i] / (slot_ticks * G)
        amp = np.clip(amp, -1.0, 1.0)
    reads = []
    for wv in (np.clip(amp, 0, None), np.clip(-amp, 0, None)):
        rec = link.send(build_train(wv, x, slot_ticks))
        reads.append(float(rec[-1]))
    return reads[0] - reads[1]


def measure_linearity(link, n: int, slot_ticks: int, trials: int,
                      fs: float, rng: np.random.Generator,
                      compensate: bool = False) -> dict:
    """The measurement that matters: is the readback linear in the true dot?

    The reference is the dot product the TRAIN encodes (widths already rounded to
    whole ticks), not w @ x. That isolates what the cable did from rounding the
    activation onto the clock, which is not the cable's fault and is a dial the
    operator sets.
    """
    g, scale, usable = None, 1.0, np.ones(n, dtype=bool)
    if compensate:
        h = measure_impulse(link, n * slot_ticks)
        if np.all(np.isfinite(h)):
            g = reversed_response(h)
            usable = usable_slots(g, n, slot_ticks)
            scale = comp_scale(g, n, slot_ticks, usable)

    got, want = [], []
    for _ in range(trials):
        w = rng.uniform(-1.0, 1.0, n) * usable   # nothing is placed on a null
        x = rng.uniform(0.0, 1.0, n)
        v = analog_dot(link, w, x, slot_ticks, g, scale)
        if not np.isfinite(v):
            continue
        got.append(v)
        want.append(true_dot(w, x, slot_ticks))
    if len(got) < 4:
        return {"n": n, "slot_ticks": slot_ticks, "captured": len(got),
                "r2": float("nan"), "bits": float("nan")}
    got_a, want_a = np.array(got), np.array(want)
    m, b, r2 = fit_linear(got_a, want_a)
    train_ticks = n * slot_ticks
    return {
        "n": n,
        "slot_ticks": slot_ticks,
        "captured": len(got),
        "train_ticks": train_ticks,
        "train_ms": 1e3 * train_ticks / fs,
        "slope": m,
        "intercept": b,
        "r2": r2,
        "bits": effective_bits(r2),
        "usable": int(usable.sum()),
    }


# --------------------------------------------------------------------- budget

def tick_budget(fs: float, slot_ticks: int, compute_channels: int = 1,
                usable: int | None = None, n: int | None = None) -> dict:
    """What the cable can carry, in the only unit that matters here.

    One slot carries one element of one dot product: one multiply-accumulate,
    performed by holding a level for a duration. So the cable's throughput is
    usable slots per second, and GPT-2's appetite is ~124M MACs per token.

    `usable` and `n` account for slots lost to the channel's null: a train of n
    slots that only carries `usable` elements still costs n slots of time.
    """
    duty = 1.0 if (usable is None or not n) else usable / n
    macs_per_s = fs / slot_ticks * compute_channels * duty
    return {
        "fs": fs,
        "slot_ticks": slot_ticks,
        "compute_channels": compute_channels,
        "slot_duty": duty,
        "ticks_per_s": fs,
        "macs_per_s": macs_per_s,
        "gpt2_seconds_per_token": GPT2_MACS_PER_TOKEN / macs_per_s if macs_per_s else float("inf"),
        "neuron768_ms": 1e3 * 768 * slot_ticks / (fs * compute_channels * duty)
                        if duty else float("inf"),
    }


# ----------------------------------------------------------------------- main

def print_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> None:
    head = "  ".join(f"{title:>{w}}" for _, title, w in
                     [(k, t, len(t) if len(t) > 8 else 8) for k, t, _ in cols])
    widths = {k: max(8, len(t)) for k, t, _ in cols}
    print("  ".join(f"{t:>{widths[k]}}" for k, t, _ in cols))
    print("  ".join("-" * widths[k] for k, _, _ in cols))
    for r in rows:
        cells = []
        for k, _, fmt in cols:
            v = r.get(k, float("nan"))
            try:
                cells.append(f"{v:>{widths[k]}{fmt}}")
            except (ValueError, TypeError):
                cells.append(f"{str(v):>{widths[k]}}")
        print("  ".join(cells))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate", action="store_true",
                    help="model the path instead of touching hardware")
    ap.add_argument("--list", action="store_true", help="list audio devices and exit")
    ap.add_argument("--device", type=int, default=None, help="input/output device index")
    ap.add_argument("--fs", type=float, default=48_000.0, help="tick rate (sample rate)")
    ap.add_argument("--tau-us", type=float, default=100.0,
                    help="simulated RC time constant, microseconds")
    ap.add_argument("--hp-hz", type=float, default=20.0,
                    help="simulated AC-coupling corner at each end, Hz")
    ap.add_argument("--noise-rms", type=float, default=3e-5,
                    help="simulated input-referred noise, fraction of full scale")
    ap.add_argument("--integrator", choices=["passive", "ideal"], default="passive",
                    help="simulated summing network: the real cap, or an op-amp")
    ap.add_argument("--n", type=int, nargs="+", default=[8, 16, 32, 64, 128],
                    help="vector lengths to sweep")
    ap.add_argument("--slots", type=int, nargs="+", default=[4, 8, 16, 32],
                    help="slot widths in ticks to sweep")
    ap.add_argument("--trials", type=int, default=24, help="dot products per cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="results/loopback.json")
    a = ap.parse_args(argv)

    if a.list:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    simulate = a.simulate or a.device is None
    cfg = PathConfig(fs=a.fs, tau_s=a.tau_us * 1e-6, noise_rms=a.noise_rms,
                     out_hp_hz=a.hp_hz, in_hp_hz=a.hp_hz)

    if simulate:
        link = SimLink(cfg, integrator=a.integrator, seed=a.seed)
        print("=" * 72)
        print("  SIMULATED -- these numbers are a MODEL of the path, NOT measurements.")
        print("  Run with --device to measure the real cable.")
        print("=" * 72)
        print(f"  model: fs={cfg.fs:,.0f} Hz  tau_rc={cfg.tau_s*1e6:.1f} us "
              f"({cfg.tau_ticks:.1f} ticks)  hp={cfg.out_hp_hz:.0f} Hz  "
              f"noise={cfg.noise_rms:.1e}  integrator={a.integrator}")
    else:
        link = DeviceLink(a.device, a.fs)
        print("=" * 72)
        print(f"  MEASURED on device {a.device} at {a.fs:,.0f} Hz")
        print("=" * 72)

    rng = np.random.default_rng(a.seed)
    t_start = time.time()

    print("\n[1] round-trip latency")
    lat = measure_latency(link, a.fs)
    print(f"    {lat['ticks']} ticks = {lat['ms']:.2f} ms out-to-in")

    print("\n[2] noise floor")
    noise = measure_noise(link)
    print(f"    rms {noise['rms']:.3e} of full scale   ->  {noise['bits']:.1f} bits of headroom")

    print("\n[3] memory: how much of the start of a train survives to the end")
    print("    ratio 1.00 = a true accumulator, position does not matter.")
    print("    ratio 0.90 = the first element counts 10% less than the last.\n")
    lengths = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    memory = measure_memory(link, a.fs, lengths)
    print_table(memory, [
        ("ticks", "ticks", "d"), ("ms", "ms", ".2f"),
        ("early", "early", ".3e"), ("late", "late", ".3e"),
        ("ratio", "ratio", ".4f"),
    ])
    cap_1pct = max_train_ticks(memory, 0.99)
    cap_10pct = max_train_ticks(memory, 0.90)
    print(f"\n    longest train the channel weights flat to 1%:  {cap_1pct} ticks"
          f" ({1e3*cap_1pct/a.fs:.2f} ms)")
    print(f"    longest train the channel weights flat to 10%: {cap_10pct} ticks"
          f" ({1e3*cap_10pct/a.fs:.2f} ms)")
    print("    Zero here does not mean the cable is useless -- it means the raw")
    print("    channel is nowhere flat, so the train must be pre-shaped. That is")
    print("    what the 'comp' column in [5] measures.")

    print("\n[5] linearity sweep: did the wire compute the dot product?")
    print("    r2 > 0.99 means yes. bits is what the readback resolved.")
    print("    'raw' drives the train straight in. 'comp' divides out the channel's")
    print("    own impulse response first, so each slot lands with equal weight.")
    print("    'usable' is how many of the n slots are not sitting on a null.\n")
    rows = []
    for n in a.n:
        for slot in a.slots:
            raw = measure_linearity(link, n, slot, a.trials, a.fs, rng, compensate=False)
            comp = measure_linearity(link, n, slot, a.trials, a.fs, rng, compensate=True)
            raw["r2_comp"] = comp["r2"]
            raw["bits_comp"] = comp["bits"]
            raw["usable"] = comp["usable"]
            raw["over_cap"] = (raw["train_ticks"] / cap_1pct) if cap_1pct else float("inf")
            rows.append(raw)
    print_table(rows, [
        ("n", "n", "d"), ("slot_ticks", "slot", "d"),
        ("train_ticks", "ticks", "d"), ("train_ms", "ms", ".2f"),
        ("r2", "r2 raw", ".5f"), ("bits", "bits", ".1f"),
        ("usable", "usable", "d"),
        ("r2_comp", "r2 comp", ".5f"), ("bits_comp", "bits", ".1f"),
    ])

    ok = [r for r in rows if r["r2_comp"] == r["r2_comp"] and r["r2_comp"] > 0.99]
    best = max(rows, key=lambda r: r["bits_comp"] if r["bits_comp"] == r["bits_comp"] else -1)

    print("\n[6] what that buys, at the best cell measured")
    print(f"    best: n={best['n']} slot={best['slot_ticks']} ticks, compensated  ->  "
          f"r2 {best['r2_comp']:.5f}, {best['bits_comp']:.1f} effective bits, "
          f"{best['usable']} of {best['n']} slots usable")
    bud = tick_budget(a.fs, best["slot_ticks"], usable=best.get("usable"),
                      n=best["n"])
    print(f"    the cable carries {bud['macs_per_s']:,.0f} multiply-accumulates per second")
    print(f"    one 768-input neuron: {bud['neuron768_ms']:.1f} ms")
    print(f"    all of GPT-2 (~124M MACs/token): {bud['gpt2_seconds_per_token']/60:.1f} "
          f"minutes per token on one channel")
    print(f"    cells reaching r2 > 0.99: {len(ok)} of {len(rows)}")

    payload = {
        "mode": "measured" if link.measured else "simulated",
        "warning": None if link.measured else
                   "SIMULATED -- a model of the path, not a measurement of hardware",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.time() - t_start,
        "config": cfg.as_dict() if simulate else {"fs": a.fs, "device": a.device},
        "integrator": a.integrator if simulate else "hardware",
        "latency": lat,
        "noise": noise,
        "memory": memory,
        "max_train_ticks_1pct": cap_1pct,
        "max_train_ticks_10pct": cap_10pct,
        "sweep": rows,
        "best_cell": best,
        "budget_at_best": bud,
    }
    out = FsPath(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n    wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
