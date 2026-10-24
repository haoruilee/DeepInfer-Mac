"""Would DoSA-style exact screening pay off on this model? Offline study on captured indexer inputs
(scripts/capture_indexer.py). Usage: dosa_study.py CAPTURE.npz [seed_keys] [prefill_rows]

For each index layer (except layer 20, the candidate source, which needs every block maximum) and
for several queries -- the last decode query and `prefill_rows` queries spread over the captured
prefill block -- tau is the k-th best exact score among the `seed_keys` most recent eligible keys
(a lower bound on the true k-th score). Every other eligible key (reachable and, in layers 21-39,
inside the candidate blocks) gets a certified upper bound U on its exact score from m dot
products: all positive heads and the strongest negative heads exactly, the remaining negative
heads as one ReLU of their aggregated direction (sum_h ReLU(x_h) >= ReLU(sum_h x_h)), inflated for
the pipeline's bf16 roundings. Keys with U < tau cannot enter the Top-K.

Cost relative to full scoring of the eligible keys:
  seed keys fully scored + other keys x m/H (bound) + survivors x |aggregated heads|/H (a survivor
  reuses its exactly evaluated heads).
"""
import sys

import mlx.core as mx
import numpy as np

z = np.load(sys.argv[1])
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
K = 512


def exact(q, w, k):
    bf = mx.bfloat16
    s = mx.einsum("hd,td->ht", mx.array(q.astype(np.float32)), mx.array(k.astype(np.float32))).astype(bf)
    s = (mx.maximum(s, 0) * mx.array(w.astype(np.float32)).astype(bf)[:, None]).astype(bf)
    return np.array(s.astype(mx.float32).sum(axis=0).astype(bf).astype(mx.float32))


def row_costs(q, w, k, vis, cand):
    """-> {m: (survive fraction, cost)} for one query, or None if nothing can be screened."""
    T = vis
    k = k[:T].astype(np.float64)
    q, w = q.astype(np.float64), w.astype(np.float64)
    elig = np.ones(T, bool) if cand is None else cand[:T].astype(bool)
    s = np.where(elig, exact(q, w, k), -np.inf)
    idx = np.nonzero(elig)[0]
    if len(idx) <= K + SEED:
        return None
    seed = idx[-SEED:]
    rest = idx[:-SEED]
    tau = np.sort(s[seed])[::-1][K - 1]
    X = k[rest] @ q.T
    pos, neg = np.where(w > 0)[0], np.where(w < 0)[0]
    neg = neg[np.argsort(-np.abs(w[neg]) * np.linalg.norm(q[neg], axis=1))]
    H, out = len(w), {}
    for extra in (0, 4, 8, 12):
        if extra > len(neg):
            continue
        ex, ag = neg[:extra], neg[extra:]
        P = (w[pos] * np.maximum(X[:, pos], 0)).sum(1)
        N = (np.abs(w[ex]) * np.maximum(X[:, ex], 0)).sum(1)
        if len(ag):
            N = N + np.maximum(k[rest] @ (np.abs(w[ag])[:, None] * q[ag]).sum(0), 0)
        U = P - N + 2**-7 * (P + N) + 1e-6
        surv = U >= tau
        assert np.all(surv | (s[rest] < tau)), "unsound bound"
        m = len(pos) + extra + (len(ag) > 0)
        n = len(idx)
        cost = (len(seed) + len(rest) * m / H + surv.sum() * len(ag) / H) / n
        out[extra] = (surv.mean(), cost, m)
    return out


for L in sorted({int(x.split("_")[0][1:]) for x in z.files}):
    if L >= 1000 or z[f"L{L}_src"]:
        continue
    queries = [(z[f"L{L}_q"][-1], z[f"L{L}_w"][-1], z[f"L{L}_k"], int(z[f"L{L}_visible"][-1]),
                z[f"L{L}_cand"][-1] if f"L{L}_cand" in z else None)]
    P = L + 1000
    if f"L{P}_q" in z:
        nrow = z[f"L{P}_q"].shape[0]
        for r in np.linspace(0, nrow - 1, ROWS).astype(int):
            queries.append((z[f"L{P}_q"][r], z[f"L{P}_w"][r], z[f"L{P}_k"], int(z[f"L{P}_visible"][r]),
                            z[f"L{P}_cand"][r] if f"L{P}_cand" in z else None))
    res = [c for c in (row_costs(*qq) for qq in queries) if c]
    ms = sorted(set.intersection(*[set(c) for c in res]))  # numbers of exactly evaluated negative heads
    summary = {e: tuple(np.mean([c[e][i] for c in res]) for i in range(3)) for e in ms}
    best = min(summary, key=lambda e: summary[e][1])
    print(f"layer {L:2d} keys {queries[0][3]:6d} queries {len(res):2d} | " +
          "  ".join(f"e={e} (m~{mm:.0f}): {sv * 100:4.1f}% survive, cost {c * 100:3.0f}%" for e, (sv, c, mm) in summary.items()) +
          f" | best: e={best}, m~{summary[best][2]:.0f}, survive {summary[best][0] * 100:.1f}%, cost {summary[best][1] * 100:.0f}%")
