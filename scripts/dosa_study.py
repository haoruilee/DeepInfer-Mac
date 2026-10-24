"""Would DoSA-style exact screening pay off on this model? Offline study on captured indexer inputs
(scripts/capture_indexer.py). Usage: dosa_study.py CAPTURE.npz [seed_keys]

For each index layer's last decode query: tau = the k-th best exact score among the most recent
`seed_keys` keys (a lower bound on the true k-th score), and per key a certified upper bound U on
its exact score built from m dot products -- all positive heads and the strongest negative heads
exactly, the remaining negative heads as one ReLU of their aggregated direction
(sum_h ReLU(x_h) >= ReLU(sum_h x_h)), inflated for the pipeline's bf16 roundings. Keys with
U < tau cannot enter the Top-K. Cost relative to full scoring ~ m/H (bounds) + survivors (rescoring).
"""
import sys

import mlx.core as mx
import numpy as np

z = np.load(sys.argv[1])
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1536
K = 512


def exact(q, w, k):
    bf = mx.bfloat16
    s = mx.einsum("lhd,td->lht", mx.array(q), mx.array(k)).astype(bf)
    s = (mx.maximum(s, 0) * mx.array(w).astype(bf)[..., None]).astype(bf)
    return np.array(s.astype(mx.float32).sum(axis=1).astype(bf).astype(mx.float32))


for L in sorted({int(x.split("_")[0][1:]) for x in z.files}):
    if L >= 1000 or z[f"L{L}_src"]:  # prefill blocks; the candidate-source layer needs all scores
        continue
    q, w, k = z[f"L{L}_q"][-1].astype(np.float64), z[f"L{L}_w"][-1].astype(np.float64), z[f"L{L}_k"]
    T = int(z[f"L{L}_visible"][-1])
    k = k[:T].astype(np.float64)
    s = exact(q[None].astype(np.float32), w[None].astype(np.float32), k.astype(np.float32))[0]
    if f"L{L}_cand" in z:
        s = np.where(z[f"L{L}_cand"][-1][:T], s, -np.inf)
    tau = np.sort(s[T - SEED:])[::-1][K - 1]
    X = k @ q.T  # [T, H]
    pos, neg = np.where(w > 0)[0], np.where(w < 0)[0]
    neg = neg[np.argsort(-np.abs(w[neg]) * np.linalg.norm(q[neg], axis=1))]
    out = [f"layer {L:2d} keys {T:6d} positive heads {len(pos):2d} |"]
    for extra in (0, 4, 8, 12):
        if extra > len(neg):
            continue
        ex, ag = neg[:extra], neg[extra:]
        P = (w[pos] * np.maximum(X[:, pos], 0)).sum(1)
        N = (np.abs(w[ex]) * np.maximum(X[:, ex], 0)).sum(1)
        if len(ag):
            N = N + np.maximum(k @ (np.abs(w[ag])[:, None] * q[ag]).sum(0), 0)
        U = P - N + 2**-7 * (P + N) + 1e-6
        surv = (U >= tau) & np.isfinite(s)
        assert np.all(surv | (s < tau)), "unsound bound"
        m = len(pos) + extra + (len(ag) > 0)
        cost = m / len(w) + surv.mean() * len(ag) / len(w)
        out.append(f"m={m}: {surv.mean() * 100:4.1f}% survive, cost {cost * 100:3.0f}%")
    print("  ".join(out))
