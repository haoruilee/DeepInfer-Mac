"""Kernel numbers quoted in the report, logged in one run (GPU only, no model load):
  1. indexer scoring: FP32 einsum + cast vs. BF16 GEMM, bit-identity over every captured score
  2. head reduction: three array passes vs. the fused Metal kernel
  3. one prefill query block of the indexer, before vs. after (scores and selected keys compared)
  4. MXFP8 quantized matmul vs. a BF16 GEMM of the same shape (prefill-sized)
Usage: bench_kernels.py CAPTURE.npz"""
import sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepinfer_mac.model.model import topk_positions
from deepinfer_mac.model.ops import index_head_sum

z = np.load(sys.argv[1])
bf = mx.bfloat16


def timeit(f, n=5):
    mx.eval(f())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(f())
    return (time.perf_counter() - t) / n * 1e3


# 1. bit-identity of the BF16 scoring over every captured score (decode rows and prefill blocks)
total = mism = 0
for key in sorted(k for k in z.files if k.endswith("_q")):
    L = key[:-2]
    q, k = mx.array(z[f"{L}_q"]), mx.array(z[f"{L}_k"])
    for r0 in range(0, q.shape[0], 32):
        a = mx.einsum("lhd,td->lht", q[r0:r0 + 32], k).astype(bf)
        b = mx.einsum("lhd,td->lht", q[r0:r0 + 32].astype(bf), k.astype(bf))
        mism += int((a != b).sum())
        total += a.size
print(f"1. BF16 vs FP32 scoring: {mism} differences in {total} scores ({total / 1e6:.0f}M)")

for L in ("L1024", "L1036", "L1002", "L1014"):
    q, w, k = mx.array(z[f"{L}_q"]), mx.array(z[f"{L}_w"]).astype(bf), mx.array(z[f"{L}_k"])
    vis = mx.array(z[f"{L}_visible"])
    T = k.shape[0]
    kb, qb = k.astype(bf), q.astype(bf)
    reach = mx.arange(T)[None] < vis[:, None]
    t32 = timeit(lambda: mx.einsum("lhd,td->lht", q, k).astype(bf))
    t16 = timeit(lambda: mx.einsum("lhd,td->lht", qb, kb))
    S = mx.einsum("lhd,td->lht", qb, kb)
    mx.eval(S)
    plain = lambda: (mx.maximum(S, 0) * w[..., None]).astype(bf).astype(mx.float32).sum(axis=1).astype(bf)  # noqa: E731
    same = bool(mx.all(plain() == index_head_sum(S, w)))
    tp, tk = timeit(plain), timeit(lambda: index_head_sum(S, w))

    def old():
        s = mx.einsum("lhd,td->lht", q, k).astype(bf)
        s = (mx.maximum(s, 0) * w[..., None]).astype(bf).astype(mx.float32).sum(axis=1).astype(bf).astype(mx.float32)
        return mx.sort(mx.argpartition(-mx.where(reach, s, -mx.inf), 511, axis=-1)[:, :512], axis=-1), s

    def new():
        s = index_head_sum(mx.einsum("lhd,td->lht", qb, kb), w).astype(mx.float32)
        return mx.sort(topk_positions(mx.where(reach, s, -mx.inf), 512), axis=-1), s

    (io, so), (inn, sn) = old(), new()
    to, tn = timeit(old, 3), timeit(new, 3)
    print(f"{L[1:].lstrip('0') or '0'} rows={q.shape[0]} keys={T}: 2. matmul FP32 {t32:.1f} ms vs BF16 {t16:.1f} ms ({t32 / t16:.1f}x) | "
          f"head sum 3 passes {tp:.1f} ms vs fused {tk:.1f} ms ({tp / tk:.0f}x, identical={same}) | "
          f"3. block {to:.1f} -> {tn:.1f} ms ({to / tn:.1f}x), scores identical={bool(mx.all(so == sn))}, keys identical={bool(mx.all(io == inn))}")

# 4. MXFP8 vs BF16 GEMM, prefill-sized (4096 tokens x a 5120 -> 5120 projection)
x = mx.random.normal((4096, 5120)).astype(bf)
w = mx.random.normal((5120, 5120)).astype(bf) * 0.02
wq, sc, *rest = mx.quantize(w, group_size=32, bits=8, mode="mxfp8")
tb = timeit(lambda: x @ w.T)
tq = timeit(lambda: mx.quantized_matmul(x, wq, sc, None, transpose=True, group_size=32, bits=8, mode="mxfp8"))
print(f"4. 4096x5120x5120: BF16 GEMM {tb:.1f} ms, MXFP8 quantized matmul {tq:.1f} ms ({tq / tb:.2f}x the BF16 time)")
