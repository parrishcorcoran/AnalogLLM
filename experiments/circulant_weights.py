#!/usr/bin/env python3
"""Can a projection be run as a few delays instead of a matmul -- in DIGITAL?

The owner's claim: if a weight is a delay, then W x is a few convolutions,
convolution is an FFT, and a standard model gets ~12x faster at full precision
with no quantizing. Not on analog hardware -- on the computer already there.

That claim is exact once you name r. Every d x d matrix is EXACTLY

    W = sum_{m=1..d} diag(u_m) @ C_m          C_m circulant (a delay pattern)

Stack the masks as M[k, i] = W[i, (i-k) % d]. A rank-r fit to M gives the best
r-term version of W. Then W x costs r FFT-convolutions plus r masks:

    direct      2 d^2                                   flops
    r terms     FFT(x) + r * (mult + IFFT + mask)  ~  5 d log d (r+1) + 2 r d

So the speedup is 2d^2 / that, and for d = 768 it is roughly 66 / r.
"12x" is the claim that r is about 5. This file makes that testable.

    python3 experiments/circulant_weights.py          # synthetic, checks the code
    python3 experiments/circulant_weights.py gpt2     # real weights: THE test
"""
import sys
import time

import numpy as np


def decompose(W, r):
    """Best r-term  sum_m diag(u_m) @ circulant(c_m)  for a square W."""
    d = W.shape[0]
    i = np.arange(d)
    M = W[i[None, :], (i[None, :] - i[:, None]) % d]      # M[k, i] = W[i, i-k]
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    kernels = (U[:, :r] * s[:r]).T                          # r x d, over shift k
    masks = Vt[:r]                                          # r x d, over output i
    return kernels, masks


def apply(x, kernels, masks):
    """W x through r convolutions. This is the fast path."""
    X = np.fft.fft(x)
    y = np.zeros_like(x, dtype=complex)
    for c, u in zip(kernels, masks):
        y += u * np.fft.ifft(np.fft.fft(c) * X)
    return y.real


def flops(d, r):
    direct = 2 * d * d
    fft = 5 * d * np.log2(d)
    fast = fft + r * (6 * d + fft + 2 * d)
    return direct, fast


def rel_err(W, r, x):
    k, u = decompose(W, r)
    return np.linalg.norm(apply(x, k, u) - W @ x) / np.linalg.norm(W @ x)


def synthetic(d, r_true, rng):
    """A matrix built from exactly r_true delay-and-mask terms."""
    W = np.zeros((d, d))
    for _ in range(r_true):
        c = rng.standard_normal(d)
        C = np.array([np.roll(c, i) for i in range(d)])
        W += np.diag(rng.standard_normal(d)) @ C
    return W


def report(name, W, rng):
    d = W.shape[0]
    x = rng.standard_normal(d)
    print("\n%s  (d=%d)" % (name, d))
    print("  %-6s %-12s %-10s %s" % ("r", "rel error", "speedup", ""))
    for r in (1, 2, 5, 10, 20, 50, 100, 200):
        if r > d:
            break
        e = rel_err(W, r, x)
        direct, fast = flops(d, r)
        flag = "  <- 12x lives here" if 10 < direct / fast < 15 else ""
        print("  %-6d %-12.4f %-10.1fx%s" % (r, e, direct / fast, flag))


def main():
    rng = np.random.default_rng(0)
    if len(sys.argv) > 1 and sys.argv[1] != "synthetic":
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(sys.argv[1], torch_dtype=torch.float32)
        for n in (0, 5, 11):
            blk = model.transformer.h[n]
            W = blk.attn.c_proj.weight.detach().numpy().T
            report("block %d attn.c_proj" % (n + 1), W, rng)
            W = blk.attn.c_attn.weight.detach().numpy().T[:768]    # Q only
            report("block %d Wq" % (n + 1), W, rng)
        return 0

    d = 768
    report("random (no structure)", rng.standard_normal((d, d)), rng)
    report("built from 5 terms", synthetic(d, 5, rng), rng)
    report("built from 20 terms", synthetic(d, 20, rng), rng)

    # and the time, because flops are a claim and a clock is a measurement
    W = synthetic(d, 5, rng)
    x = rng.standard_normal(d)
    k, u = decompose(W, 5)
    t0 = time.perf_counter()
    for _ in range(2000):
        W @ x
    t1 = time.perf_counter()
    for _ in range(2000):
        apply(x, k, u)
    t2 = time.perf_counter()
    print("\nwall clock, d=768, r=5, numpy on this box:")
    print("  matmul      %.1f us" % ((t1 - t0) / 2000 * 1e6))
    print("  5 convs     %.1f us   (%.1fx)" % ((t2 - t1) / 2000 * 1e6, (t1 - t0) / (t2 - t1)))
    print("  numpy's matmul is BLAS-tuned and its fft is not; the flop ratio is")
    print("  the honest number, the clock is what you get for free today.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
