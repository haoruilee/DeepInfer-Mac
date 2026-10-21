"""Decode-step profile on the real converted model, Engram stubbed (no network), converter paused
by the caller. Times sub-parts of attention and MoE with a sync after each."""
import collections, sys, time
import mlx.core as mx, numpy as np
sys.path.insert(0, ".")
from deepinfer_mac import LLM
from deepinfer_mac.model import model as M
llm = LLM(expert_cache_gb=5)
class Zeros:
    def lookup(self, layer, ids): return mx.zeros((*ids.shape, 256), dtype=mx.bfloat16)
for b in llm.model.layers:
    if b.engram is not None: b.engram.table = Zeros()
T = collections.defaultdict(float)
def sync(x): mx.eval(x if isinstance(x, (mx.array, list)) else [v for v in (x if isinstance(x, tuple) else ()) if isinstance(v, mx.array)])
def wrap(obj, name, tag):
    f = getattr(obj, name)
    def g(*a, **k):
        t = time.perf_counter(); out = f(*a, **k); sync(out); T[tag] += time.perf_counter() - t; return out
    setattr(obj, name, g)
for b in llm.model.layers:
    at = b.attn
    at._compiled()
    wrap(at, "_proj_c", "attn.proj"); wrap(at, "_core_c", "attn.core")
    if at.is_kv_source: wrap(at, "_compress", "attn.compressor")
    if at.is_index_source: wrap(at, "_indexer", "attn.indexer")
    wrap(b.ffn, "_slab_moe", "moe.experts(get+gather_qmm)")
    if not hasattr(b.ffn, "_rs_c"): b.ffn._rs_c = mx.compile(b.ffn._route_shared)
    wrap(b.ffn, "_rs_c", "moe.route+shared")
wrap(M.Attention, "__call__", "ATTN total") if False else None
oa = M.Attention.__call__
def att(self, *a, **k):
    t = time.perf_counter(); out = oa(self, *a, **k); sync(out); T["ATTN total"] += time.perf_counter() - t; return out
M.Attention.__call__ = att
om = M.MoE.__call__
def moe(self, *a, **k):
    t = time.perf_counter(); out = om(self, *a, **k); sync(out); T["MOE total"] += time.perf_counter() - t; return out
M.MoE.__call__ = moe
ids, _ = llm.chat_prompt([{"role": "user", "content": "Count from one to twenty, separated by commas."}])
st, logits, _, _ = llm._prefill(ids, False)
tok = int(np.argmax(np.array(logits[-1])))
for i in range(3):  # warm compile caches
    lg = llm.model.forward(st, [tok]); tok = int(np.argmax(np.array(lg[-1])))
T.clear(); s0 = dict(llm.weights.experts.stats); n = 6; t = time.perf_counter()
for i in range(n):
    lg = llm.model.forward(st, [tok]); mx.eval(lg); tok = int(np.argmax(np.array(lg[-1])))
tot = (time.perf_counter() - t) / n; s = llm.weights.experts.stats
print(f"total {tot*1000:.0f} ms/token (sync after every part)")
for k, v in sorted(T.items(), key=lambda x: -x[1]): print(f"  {k:30s} {v/n*1000:7.1f} ms/token")
print(f"  SSD wait {(s['wait_s']-s0['wait_s'])/n*1000:.0f} ms/token, {(s['bytes']-s0['bytes'])/n/1e9:.2f} GB/token, hit {(s['hits']-s0['hits'])/max(1,(s['hits']-s0['hits'])+(s['misses']-s0['misses'])):.2f}")
