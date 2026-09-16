"""The loopback path, modelled honestly.

The chain is:  DAC -> output AC coupling -> RC network -> input AC coupling -> ADC.

Every stage here exists because it is something the real cable does to the pulse
train. Nothing is idealised away. In particular the RC network is modelled as the
PASSIVE circuit it is:

    v += (dt/tau) * (s - v)          <- a cap charging toward the drive

and NOT as

    v += (dt/tau) * s                <- an op-amp integrator (needs power)

The difference is the whole story of how long a pulse train may be. A passive cap
only behaves as an integrator while v is small compared to s; once it has charged
appreciably it starts forgetting the early part of the train at rate 1/tau. So the
train has to fit inside tau, and the fraction of tau it uses is exactly the
linearity error you pay. `ideal_integrator` is kept so the two can be compared --
that comparison is the point.
"""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class PathConfig:
    """Everything about the physical path that affects the readback.

    Defaults describe a plausible built-in laptop codec driving the $2 RC network
    from timing-substrate/analog/analog_dot.py (R=10k, C=10nF -> tau=100us).
    They are a STARTING POINT for simulation, not measurements. `rig.py --device`
    measures the real values and writes them back.
    """
    fs: float = 48_000.0        # tick rate: one sample is one tick
    tau_s: float = 100e-6       # RC time constant of the summing network
    out_hp_hz: float = 20.0     # output AC-coupling corner (headphone amp)
    in_hp_hz: float = 20.0      # input AC-coupling corner (line/mic in)
    dac_bits: int = 16
    adc_bits: int = 16
    noise_rms: float = 3e-5     # input-referred noise, fraction of full scale
    gain: float = 1.0           # end-to-end voltage gain, unknown and irrelevant

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def tau_ticks(self) -> float:
        """How many ticks of history the network actually holds."""
        return self.tau_s * self.fs

    def as_dict(self) -> dict:
        d = asdict(self)
        d["tau_ticks"] = self.tau_ticks
        return d


def quantize(sig: np.ndarray, bits: int) -> np.ndarray:
    """A converter has a finite number of rungs. Clip, then land on one."""
    levels = 2 ** (bits - 1) - 1
    return np.round(np.clip(sig, -1.0, 1.0) * levels) / levels


def highpass(sig: np.ndarray, fs: float, f_hz: float) -> np.ndarray:
    """First-order AC coupling. A series cap cannot pass a held level."""
    if f_hz <= 0:
        return sig.astype(np.float64)
    a = 1.0 / (1.0 + 2.0 * np.pi * f_hz / fs)
    out = np.empty_like(sig, dtype=np.float64)
    prev_in = 0.0
    prev_out = 0.0
    for k, x in enumerate(sig):
        prev_out = a * (prev_out + x - prev_in)
        prev_in = x
        out[k] = prev_out
    return out


def passive_rc(sig: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    """THE REAL CIRCUIT. A cap charging toward the drive through a resistor.

    Integrates while v << s, forgets with time constant tau. This is why the
    train length matters and why the readback is not a clean sum.
    """
    a = 1.0 / (tau_s * fs)
    out = np.empty_like(sig, dtype=np.float64)
    v = 0.0
    for k, s in enumerate(sig):
        v += a * (s - v)
        out[k] = v
    return out


def ideal_integrator(sig: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    """What an op-amp integrator would do -- and what analog_dot.py assumed.

    Never forgets. Included only so the cost of the passive approximation can be
    measured rather than argued about.
    """
    a = 1.0 / (tau_s * fs)
    return np.cumsum(sig.astype(np.float64)) * a


def run_path(sig: np.ndarray, cfg: PathConfig, integrator: str = "passive",
             rng: np.random.Generator | None = None) -> np.ndarray:
    """Push a pulse train down the simulated cable and return what comes back."""
    rng = rng or np.random.default_rng(0)
    x = quantize(sig, cfg.dac_bits)
    x = highpass(x, cfg.fs, cfg.out_hp_hz)
    if integrator == "passive":
        v = passive_rc(x, cfg.fs, cfg.tau_s)
    elif integrator == "ideal":
        v = ideal_integrator(x, cfg.fs, cfg.tau_s)
    else:
        raise ValueError(f"unknown integrator {integrator!r}")
    v = highpass(v, cfg.fs, cfg.in_hp_hz) * cfg.gain
    v = v + rng.normal(0.0, cfg.noise_rms, size=v.shape)
    return quantize(v, cfg.adc_bits)


# ---------------------------------------------------------------- pulse trains

def build_train(w: np.ndarray, x: np.ndarray, slot_ticks: int) -> np.ndarray:
    """One slot per element. AMPLITUDE = w, WIDTH = x. The wire does the rest.

    Charge deposited by element i is proportional to w_i * x_i -- the multiply is
    done by holding a level for a duration. The cap sums them. `w` must be
    non-negative here; signed weights are two passes (see analog_dot).
    """
    n = len(w)
    sig = np.zeros(n * slot_ticks, dtype=np.float64)
    widths = np.clip(np.round(np.abs(x) * slot_ticks), 0, slot_ticks).astype(int)
    for i in range(n):
        start = i * slot_ticks
        sig[start:start + widths[i]] = abs(w[i])
    return sig


def true_dot(w: np.ndarray, x: np.ndarray, slot_ticks: int) -> float:
    """The dot product the train actually encodes, after width rounding.

    Compared against this rather than against w @ x so that the metric isolates
    what the CABLE did, not the unavoidable rounding of x onto whole ticks.
    """
    widths = np.clip(np.round(np.abs(x) * slot_ticks), 0, slot_ticks)
    return float(np.sum(w * widths / slot_ticks))


# -------------------------------------------------------------------- metrics

def fit_linear(got: np.ndarray, want: np.ndarray) -> tuple[float, float, float]:
    """Fit got = m*want + b. Returns (m, b, r2).

    The absolute scale (1/RC, codec gain, input trim) is unknown and does not
    matter. What matters is whether the readback is LINEAR in the true dot
    product -- that is the claim being tested.
    """
    A = np.vstack([want, np.ones_like(want)]).T
    (m, b), *_ = np.linalg.lstsq(A, got, rcond=None)
    resid = got - A @ np.array([m, b])
    ss_tot = float(np.sum((got - got.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0
    return float(m), float(b), r2


def effective_bits(r2: float) -> float:
    """Bits of the dot product the cable actually resolved.

    Same definition used in timing-substrate/analog: an SNR read off the residual
    of the linear fit. r2 = 0.99 is ~3.3 bits, 0.9999 is ~6.6 bits.
    """
    return 0.5 * np.log2(1.0 / max(1e-15, 1.0 - r2))
