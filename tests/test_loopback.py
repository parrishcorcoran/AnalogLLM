"""Tests for the loopback rig.

These check the instrument, not the cable. If the rig cannot recover a dot
product from a channel whose behaviour we know exactly, then nothing it reports
about real hardware means anything.
"""
import numpy as np
import pytest

from loopback.model import (PathConfig, build_train, effective_bits, fit_linear,
                            highpass, ideal_integrator, passive_rc, true_dot)
from loopback import rig


# ------------------------------------------------------------ circuit models

def test_passive_rc_forgets_and_ideal_does_not():
    """The whole reason the train length matters."""
    fs, tau = 48_000.0, 1e-3          # 48 ticks of memory
    sig = np.zeros(2000); sig[0] = 1.0
    assert passive_rc(sig, fs, tau)[-1] == pytest.approx(0.0, abs=1e-9)
    assert ideal_integrator(sig, fs, tau)[-1] > 0.0


def test_passive_rc_integrates_while_train_is_short():
    """Inside its memory the cap is a good accumulator: order does not matter."""
    fs, tau = 48_000.0, 10.0           # effectively infinite memory
    a = np.zeros(64); a[0] = 1.0
    b = np.zeros(64); b[-1] = 1.0
    assert passive_rc(a, fs, tau)[-1] == pytest.approx(passive_rc(b, fs, tau)[-1], rel=1e-3)


def test_highpass_removes_a_held_level():
    """A series cap cannot pass DC. This is what kills a long accumulation."""
    fs = 48_000.0
    held = np.ones(int(fs))            # one second of DC
    out = highpass(held, fs, 20.0)
    assert abs(out[-1]) < 1e-3
    assert out[0] == pytest.approx(1.0, rel=0.01)


def test_highpass_passthrough_when_corner_is_zero():
    sig = np.linspace(-1, 1, 100)
    assert np.allclose(highpass(sig, 48_000.0, 0.0), sig)


# ---------------------------------------------------------------- pulse trains

def test_build_train_encodes_amplitude_and_width():
    w = np.array([0.5, 1.0])
    x = np.array([0.25, 1.0])
    train = build_train(w, x, slot_ticks=8)
    assert train.shape == (16,)
    assert np.count_nonzero(train[:8]) == 2          # 0.25 * 8
    assert np.count_nonzero(train[8:]) == 8
    assert train[0] == pytest.approx(0.5)
    assert train[8] == pytest.approx(1.0)


def test_true_dot_accounts_for_width_rounding():
    """The reference is what the train encodes, not w @ x."""
    w = np.array([1.0, 1.0])
    x = np.array([0.5, 0.1])                          # 0.1*4 rounds to 0 ticks
    assert true_dot(w, x, slot_ticks=4) == pytest.approx(0.5)


# -------------------------------------------------------------------- metrics

def test_fit_absorbs_unknown_gain_and_offset():
    """The absolute scale of the readback is unknown and must not matter."""
    want = np.linspace(-1, 1, 50)
    got = 37.0 * want - 4.0
    m, b, r2 = fit_linear(got, want)
    assert m == pytest.approx(37.0)
    assert b == pytest.approx(-4.0)
    assert r2 == pytest.approx(1.0)


def test_effective_bits_is_monotone_in_r2():
    assert effective_bits(0.99) < effective_bits(0.9999)
    assert effective_bits(0.99) == pytest.approx(3.32, abs=0.01)


# ------------------------------------------------------------------- the link

def _clean_link(**kw):
    """A path with no droop, no coupling loss and no noise: a pure accumulator."""
    cfg = PathConfig(fs=48_000.0, tau_s=10.0, out_hp_hz=0.0, in_hp_hz=0.0,
                     noise_rms=0.0, dac_bits=24, adc_bits=24, **kw)
    return rig.SimLink(cfg, integrator="ideal")


def test_align_recovers_the_train_window():
    link = _clean_link()
    train = np.zeros(128); train[64] = 1.0
    out = link.send(train)
    assert out.shape == (128,)
    assert np.all(np.isfinite(out))


def test_clean_path_computes_the_dot_product():
    """With an honest accumulator the wire gets it essentially exactly."""
    link = _clean_link()
    rng = np.random.default_rng(0)
    r = rig.measure_linearity(link, n=16, slot_ticks=8, trials=12,
                              fs=48_000.0, rng=rng)
    assert r["r2"] > 0.9999


def test_memory_probe_scores_one_on_a_true_accumulator():
    link = _clean_link()
    rows = rig.measure_memory(link, 48_000.0, [32, 128])
    for row in rows:
        assert row["ratio"] == pytest.approx(1.0, abs=1e-3)


def test_memory_probe_detects_forgetting():
    cfg = PathConfig(fs=48_000.0, tau_s=1e-3, out_hp_hz=0.0, in_hp_hz=0.0,
                     noise_rms=0.0)
    rows = rig.measure_memory(rig.SimLink(cfg), 48_000.0, [256])
    assert rows[0]["ratio"] < 0.1


# ------------------------------------------------------------- compensation

def _droopy_link():
    """A channel that is definitely not flat: short memory, real AC coupling."""
    cfg = PathConfig(fs=48_000.0, tau_s=10e-3, out_hp_hz=20.0, in_hp_hz=20.0,
                     noise_rms=1e-6, dac_bits=24, adc_bits=24)
    return rig.SimLink(cfg)


def test_compensation_recovers_linearity_a_droopy_channel_loses():
    """The regression test that matters: inverting the channel must win.

    Raw, the far end of the train counts for more than the near end and the
    readback is not the dot product. Pre-shaped by the measured impulse
    response, it is.
    """
    rng = np.random.default_rng(1)
    raw = rig.measure_linearity(_droopy_link(), 32, 8, 16, 48_000.0,
                                np.random.default_rng(1), compensate=False)
    comp = rig.measure_linearity(_droopy_link(), 32, 8, 16, 48_000.0,
                                 np.random.default_rng(1), compensate=True)
    assert raw["r2"] < 0.99
    assert comp["r2"] > 0.999
    assert comp["bits"] > raw["bits"] + 2


def test_impulse_response_is_not_the_preamble_tail():
    """Baselining must remove the preamble, or the response is mostly preamble."""
    link = _droopy_link()
    h = rig.measure_impulse(link, 128)
    silence = rig.send_baselined(link, np.zeros(128))
    assert abs(h[0]) > 20 * (np.abs(silence).max() + 1e-12)


# ---------------------------------------------------------------- the budget

def test_budget_arithmetic():
    b = rig.tick_budget(fs=48_000.0, slot_ticks=8, compute_channels=1)
    assert b["macs_per_s"] == pytest.approx(6_000.0)
    assert b["gpt2_seconds_per_token"] == pytest.approx(
        rig.GPT2_MACS_PER_TOKEN / 6_000.0)


# ----------------------------------------------------------------- the budget

def test_gpt2_constants_are_derived_not_quoted():
    """n_in must reproduce the 65,280 in WATER_TO_CONVERTER.pdf, or the model
    of the machine disagrees with the machine."""
    from loopback import budget
    g = budget.derive()
    assert g["n_in"] == 65_280
    assert g["macs"] == pytest.approx(123.5e6, rel=0.01)
    assert g["fanout"] == pytest.approx(g["macs"] / g["n_in"])


def test_one_accumulator_pays_for_every_mac_separately():
    from loopback import budget
    assert budget.ticks_one_accumulator(1) == budget.G["macs"]
    assert budget.ticks_one_accumulator(8) == 8 * budget.G["macs"]


def test_crossbar_recovers_the_fanout():
    """The whole point of tier 2: the wire performs 1,892 MACs per pulse."""
    from loopback import budget
    t1 = budget.ticks_one_accumulator(1)
    t2 = budget.ticks_crossbar(1, 1, 1)
    assert t1 / t2 > 900              # at least the fanout, minus readout cost


def test_crossbar_is_limited_by_whichever_side_is_slower():
    from loopback import budget
    g = budget.G
    assert budget.ticks_crossbar(1, 1, 10_000) == g["n_in"]      # emit-bound
    assert budget.ticks_crossbar(1, 10_000, 1) == g["n_out"]     # read-bound


def test_substrate_throughput_ignores_both_counts():
    """Depth is an addition when every stage is live, not a multiplication."""
    from loopback import budget
    flowing = budget.ticks_substrate(flowing=True)
    staged = budget.ticks_substrate(flowing=False)
    assert flowing == budget.PDM_RATE + budget.GPT2_DEPTH
    assert staged == budget.PDM_RATE * budget.LINEAR_STAGES
    assert staged > 40 * flowing


def test_the_ladder_is_ordered():
    """Each tier must beat the one below it at equal precision."""
    from loopback import budget
    R = budget.PDM_RATE
    t1 = budget.ticks_one_accumulator(R)
    t2 = budget.ticks_crossbar(R, 4, 4)
    t3 = budget.ticks_substrate()
    assert t1 > t2 > t3


def test_one_cap_cannot_beat_software_at_any_audio_clock():
    """Tier 1 is the cable on the desk. It loses to the laptop, always."""
    from loopback import budget
    best_clock = max(fs for _, fs in budget.CLOCKS)
    ticks = budget.ticks_one_accumulator(budget.PDM_RATE)
    assert best_clock / ticks < budget.SOFTWARE_TOK_S
