import sys, time, collections, numpy as np, mlx.core as mx
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
exec(open("scripts/bench_dense.py").read().split("t = time.time()\nhasher")[0])  # reuse setup (W, a, cfg)
from deepinfer_mac.model import model as M, ops
hasher = NgramHasher.load(a, root, cfg["engram_vocab_size"])
m = Transformer(a, W(), hasher)
T = collections.defaultdict(float)
def timed(name, f):
    def g(*args, **kw):
        mx.eval(args[1] if len(args) > 1 and isinstance(args[1], mx.array) else [])
        t = time.perf_counter(); out = f(*args, **kw)
        mx.eval(out if isinstance(out, mx.array) else [x for x in (out if isinstance(out, tuple) else ()) if isinstance(x, mx.array)])
        T[name] += time.perf_counter() - t; return out
    return g
M.Attention.__call__ = timed("attn", M.Attention.__call__)
M.MoE.__call__ = timed("moe", M.MoE.__call__)
M.Engram.__call__ = timed("engram", M.Engram.__call__)
M.Block.hc_mixes = timed("hc_mixes", M.Block.hc_mixes)
M.Attention._indexer = timed("  indexer", M.Attention._indexer)
ops.sparse_attn = timed("  sparse_attn", ops.sparse_attn); M.ops = ops
st = m.new_state(); toks = list(range(1000, 1300)); m.forward(st, toks)
T.clear(); t = time.perf_counter(); n = 5
for i in range(n): mx.eval(m.forward(st, [2000 + i]))
tot = (time.perf_counter() - t) / n
print(f"total {tot*1000:.0f} ms/token"); [print(f"{k:14s} {v/n*1000:7.1f} ms") for k, v in sorted(T.items(), key=lambda x: -x[1])]
