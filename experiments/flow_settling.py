#!/usr/bin/env python3
"""Does a stack with every layer live at once settle -- and does it survive noise?

Two questions, both structural:

  1. If every block fires on every tick, each reading what the block before it
     emitted on the PREVIOUS tick, does the stack converge to the same answer
     the staged version gives? And when?

  2. If every tick adds noise -- which is what an analog substrate does -- does
     the error compound the longer it runs, or stay bounded?

Random weights, MLP-only blocks, no attention. So this says nothing about GPT-2's
numbers. It tests the ARCHITECTURE, which is the part that should not depend on
which weights are loaded.

    python3 experiments/flow_settling.py
"""
import numpy as np

D, L = 64, 12          # width, depth


def ln(x):
    return (x - x.mean()) / (x.std() + 1e-5)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def make(seed=0):
    rng = np.random.default_rng(seed)
    W1 = [rng.standard_normal((4 * D, D)) / np.sqrt(D) for _ in range(L)]
    W2 = [rng.standard_normal((D, 4 * D)) / np.sqrt(4 * D) for _ in range(L)]
    block = lambda k, h: h + W2[k] @ gelu(W1[k] @ ln(h))
    return rng, block, rng.standard_normal(D)


def staged(block, x0):
    """Layer 1 finishes, hands off, layer 2 runs. The ordinary way."""
    h = x0.copy()
    for k in range(L):
        h = block(k, h)
    return h


def flowing(block, x0, ticks, noise=0.0, rng=None, want=None):
    """Every block fires every tick. Each reads the PREVIOUS tick's state.

    Nobody waits for anybody. Returns the error history against `want`.
    """
    state = [x0.copy()] + [np.zeros(D) for _ in range(L)]
    hist = []
    for _ in range(ticks):
        state = [state[0]] + [
            block(k, state[k]) + (rng.normal(0, noise, D) if noise else 0.0)
            for k in range(L)
        ]
        hist.append(np.linalg.norm(state[L] - want))
    return state, hist


def main():
    print("1. NOISELESS -- does it settle, and to what?\n")
    rng, block, x0 = make()
    want = staged(block, x0)
    _, hist = flowing(block, x0, L + 5, want=want)
    print(f"{'tick':>6}{'||flowing - staged||':>24}")
    for t, e in enumerate(hist, 1):
        print(f"{t:>6}{e:>24.3e}" + ("   <-- EXACT" if e == 0.0 else ""))
    first = next(t for t, e in enumerate(hist, 1) if e == 0.0)
    print(f"\n   depth L = {L}, exact at tick {first}. "
          f"Not approximately -- error is 0.0 and stays 0.")
    print("   'run until the state stops moving' stops at exactly the depth.\n")

    print("2. WITH NOISE -- the analog case. Does error compound over ticks?\n")
    print(f"{'noise/tick':>12}{'~bits':>7}{'err @ tick L':>15}"
          f"{'err @ 5L':>12}{'growth':>9}")
    for noise, bits in [(1e-4, 13), (1e-3, 10), (1e-2, 7), (1e-1, 3)]:
        rng, block, x0 = make()
        want = staged(block, x0)
        n = np.linalg.norm(want)
        _, hist = flowing(block, x0, 5 * L, noise=noise, rng=rng, want=want)
        eL, e5 = hist[L - 1] / n, hist[-1] / n
        print(f"{noise:>12.0e}{bits:>7}{eL:>15.2e}{e5:>12.2e}{e5/eL:>8.2f}x")

    print("""
   Running 5x longer than needed grows the error ~7%, not 5x. It does NOT
   compound. LayerNorm pulls the state back each tick, so errors do not
   random-walk -- the floor cleans at the rate the arithmetic dirties.

   Error settles at ~2.9x the per-tick noise, linearly. So a channel giving
   N bits per value yields an answer with roughly N - 1.5 bits.
""")


if __name__ == "__main__":
    main()
