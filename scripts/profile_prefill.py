"""Where does long-prompt prefill time go? One 2048-token chunk, a sync after each component."""
import collections, sys, time
from pathlib import Path
import mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepinfer_mac import LLM
llm = LLM()
T = collections.defaultdict(float)
def sync(x): mx.eval(x if isinstance(x, (mx.array, list)) else [v for v in (x if isinstance(x, tuple) else ()) if isinstance(v, mx.array)])
def wrap(obj, name, tag):
    f = getattr(obj, name)
    def g(*a, **k):
        t = time.perf_counter(); out = f(*a, **k); sync(out); T[tag] += time.perf_counter() - t; return out
    setattr(obj, name, g)
for b in llm.model.layers:
    at = b.attn; at._compiled()
    wrap(at, "_proj_c", "attn: q/kv projections (mxfp8)"); wrap(at, "_core_c", "attn: sparse attention + wo_a/wo_b")
    if at.is_kv_source: wrap(at, "_compress", "attn: compressor")
    if at.is_index_source: wrap(at, "_indexer", "attn: indexer")
    b.ffn._rs_c = mx.compile(b.ffn._route_shared); wrap(b.ffn, "_rs_c", "moe: router + shared expert (mxfp8)")
    wrap(b.ffn, "_experts_into", "moe: routed experts (gather_qmm + SSD)")
    wrap(b, "hc_mixes", "mHC mixes")
    if b.engram is not None: wrap(b.engram, "__call__", "engram")
text = (Path("~/models/api-vectors/deepseek-v4.1-flash-20260910/prompts/case_014.txt").expanduser()).read_text()
ids = llm.encode(text)[:2048]
st = llm.model.new_state(); s0 = dict(llm.weights.experts.stats); t = time.perf_counter()
llm.model.forward(st, ids, replay_from=len(ids) - 140)
tot = time.perf_counter() - t; s = llm.weights.experts.stats
print(f"2048-token prefill chunk (bounded replay): {tot:.1f}s = {2048/tot:.0f} tok/s; SSD wait {s['wait_s']-s0['wait_s']:.1f}s, {(s['bytes']-s0['bytes'])/1e9:.0f} GB")
for k, v in sorted(T.items(), key=lambda x: -x[1]): print(f"  {k:42s} {v:6.1f}s  {100*v/tot:4.0f}%")
