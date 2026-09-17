#!/usr/bin/env python3
"""Is a trained projection made of pair-relationships, or 589,824 loose numbers?

A dense 768x768 projection has exactly as many entries as there are ordered
pairs of 768 notes. That is a counting identity and it is free. What it does
NOT tell you is whether the trained values have the SHAPE a pair of notes can
make -- whether W_ij can be GENERATED from a relationship between note i and
note j, or whether it has to be STORED, one number at a time.

That is the difference between a build and a wish, and it is answerable from
the weights alone. No rig, no medium, no analog anything. Four measurements,
each with a control that pins it.

    conv-energy   fraction of W explained by depending on (i-j) ALONE.
                  a pure convolution reads 1.0; no structure reads 1/d.
                  this is the strong interval hypothesis: one delay, shared.

    shifts/90%    how many distinct shifts carry 90% of the energy.
                  W = sum_k D_k S^k is EXACT with d terms (D_k diagonal, S the
                  cyclic shift), so this is not a fit -- it is the real split.
                  a small number means few delay lengths, each with its own
                  per-note mask. that is the weak interval hypothesis, and it
                  is the one that would actually be buildable.

    rank/90%      basis-independent compressibility. included because it
                  separates "has structure" from "has INTERVAL structure".

    generators    the one that matters, and the one the others are blind to.
                  a holographic code is BUILT to look random -- bound vectors
                  have random-looking entries by construction, so singular
                  values are exactly the instrument it hides from. a matrix
                  made of 10 mask-and-bind pairs reads near-full rank. this
                  test rearranges W so the generators become a rank, and
                  recovers the count. small means buildable: r masks and r
                  bindings instead of 589,824 stored values.

    non-normal    ||WW' - W'W|| / ||W||^2. every circulant is normal, so a
                  large value rules out W being a convolution in ANY basis.
                  zero does not prove it is one; large does disprove it.

CAVEAT, and it matters. The 768 dimensions of the residual stream have no
canonical order -- permute them consistently and the model is unchanged. So
conv-energy and shifts are measured in an ARBITRARY basis and structure could
be present but hidden. A positive result is therefore strong and a negative
result is weak. rank and non-normality do not have this problem; they are the
same in every basis. Read them first.

    pip install torch transformers
    python3 experiments/weight_structure.py [model] [n_layers]

Runs on CPU. gpt2 takes a couple of minutes.
"""
import sys
from collections import defaultdict

import numpy as np


def squares(W):
    """Cut a 2-D weight into square blocks. A d x 4d MLP is 4 square maps."""
    a, b = W.shape
    if a == b:
        return [W]
    if a > b:
        W, a, b = W.T, b, a
    return [W[:, i * a:(i + 1) * a] for i in range(b // a)] if b % a == 0 else []


def diag_constancy(W):
    d = W.shape[0]
    i = np.arange(d)
    cols = (i[:, None] - i[None, :]) % d
    m = W[i[:, None], cols].mean(axis=0)        # best pure-convolution fit
    return d * (m ** 2).sum() / (W ** 2).sum()


def shift_concentration(W):
    d = W.shape[0]
    i = np.arange(d)
    cols = (i[:, None] - i[None, :]) % d
    e = (W[i[:, None], cols] ** 2).sum(axis=0)
    e = np.sort(e)[::-1] / e.sum()
    return int(np.searchsorted(np.cumsum(e), 0.90) + 1)


def rank90(W):
    s = np.linalg.svd(W, compute_uv=False) ** 2
    return int(np.searchsorted(np.cumsum(s / s.sum()), 0.90) + 1)


def generators(W):
    """How many mask-and-bind pairs generate W.

    W = sum_k D_k S^k is exact. Stack the masks: M[k,i] = W[i, i-k]. Then
    rank(M) = r means exactly

        W = sum_{m=1..r} diag(u_m) @ C_m

    -- r amplitude masks, each followed by one convolution. A convolution is a
    binding, so r IS the generator count. rank(W) cannot see this: a 10-
    generator matrix has near-full rank and reads as noise."""
    d = W.shape[0]
    i = np.arange(d)
    return rank90(W[i[None, :], (i[None, :] - i[:, None]) % d])


def nonnormality(W):
    return np.linalg.norm(W @ W.T - W.T @ W) / np.linalg.norm(W) ** 2


def measure(W):
    W = np.asarray(W, dtype=np.float64)
    return (diag_constancy(W), shift_concentration(W), rank90(W),
            generators(W), nonnormality(W))


HEAD = "%-26s %-12s %-11s %-9s %-11s %s" % (
    "", "conv-energy", "shifts/90%", "rank/90%", "generators", "non-normal")
ROW = "%-26s %-12.4f %-11.0f %-9.0f %-11.0f %.4f"


def _circulant(c):
    """One kernel, rolled. Drawing a fresh vector per row gives a plain random
    matrix and silently kills the control -- this bit twice while writing it."""
    return np.array([np.roll(c, i) for i in range(len(c))])


def controls(d, rng):
    print("CONTROLS  (what each test looks like when you already know the answer)\n")
    print(HEAD)
    rand = rng.standard_normal((d, d))
    circ = _circulant(rng.standard_normal(d))
    lowr = rng.standard_normal((d, 20)) @ rng.standard_normal((20, d))
    fewk = np.zeros((d, d))
    idx = np.arange(d)
    for k in (0, 5, 17, 60, 200):
        fewk[idx, (idx - k) % d] = rng.standard_normal(d)
    gen10 = sum(np.diag(rng.standard_normal(d)) @ _circulant(rng.standard_normal(d))
                for _ in range(10))
    for name, M in [("random -- THE NULL", rand), ("true convolution", circ),
                    ("rank-20", lowr), ("5 shifts + masks", fewk),
                    ("10 mask-and-bind pairs", gen10)]:
        print(ROW % ((name,) + measure(M)))
    print("\n  no structure at all reads conv-energy = 1/d = %.4f\n" % (1.0 / d))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    blocks = None
    for attr in ("transformer.h", "model.layers", "gpt_neox.layers"):
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            blocks = obj
            break
    if blocks is None:
        print("could not find the block list for %s" % name, file=sys.stderr)
        return 1

    d = model.config.hidden_size
    controls(d, np.random.default_rng(0))

    print("%s -- %d blocks, width %d\n" % (name, len(blocks), d))
    got = defaultdict(list)
    with torch.no_grad():
        for n, blk in enumerate(blocks[:limit]):
            for pname, p in blk.named_parameters():
                if p.ndim != 2:
                    continue
                # All four metrics are transpose-invariant (verified), so the
                # (in,out) vs (out,in) split between HF Conv1D and nn.Linear
                # does not need handling. gpt2's fused c_attn is 768x2304 and
                # squares() cuts it into exactly Q, K and V.
                for blkW in squares(p.detach().float().numpy()):
                    got[pname.replace(".weight", "")].append(measure(blkW))
            print("  block %d done" % n, file=sys.stderr)

    print(HEAD)
    for k in sorted(got):
        v = np.array(got[k])
        print(ROW % ((k + "  (n=%d)" % len(v),) + tuple(v.mean(axis=0))))

    print("\nread generators FIRST, then rank and non-normal")
    print("(read rank and non-normal FIRST -- they are basis-independent.")
    print("conv-energy and shifts are measured in an arbitrary basis, so a high")
    print("value is strong evidence and a low value proves nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
