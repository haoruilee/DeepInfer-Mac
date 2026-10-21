"""Plain end-to-end decode speed on the real model (no profiler syncs). Engram stubbed with zeros
unless --real-engram (the local tables, once converted)."""
import sys, time
import mlx.core as mx, numpy as np
sys.path.insert(0, ".")
from deepinfer_mac import LLM
import os
from deepinfer_mac.model import model as _M
_M.MoE.LOOKAHEAD = os.environ.get("LOOKAHEAD", "0") == "1"
llm = LLM(expert_cache_gb=float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1][0].isdigit() else None)
if "--real-engram" not in sys.argv:
    class Zeros:
        def lookup(self, layer, ids): return mx.zeros((*ids.shape, 256), dtype=mx.bfloat16)
    for b in llm.model.layers:
        if b.engram is not None: b.engram.table = Zeros()
ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long paragraph about the history of tea."}])
st, logits, _, _ = llm._prefill(ids, False)
tok = int(np.argmax(np.array(logits[-1])))
ts = []; s0 = dict(llm.weights.experts.stats)
for i in range(20):
    t = time.perf_counter(); lg = llm.model.forward(st, [tok]); tok = int(np.argmax(np.array(lg[-1]))); ts.append(time.perf_counter() - t)
s = llm.weights.experts.stats
n = s['hits'] - s0['hits'] + s['misses'] - s0['misses'] + s['prefetch_hits'] - s0['prefetch_hits']
print(f"decode: median {np.median(ts[5:])*1000:.0f} ms/token = {1/np.median(ts[5:]):.2f} tok/s | hit rate {(n - (s['misses']-s0['misses']))/n:.2f} | "
      f"{(s['bytes']-s0['bytes'])/20/1e9:.2f} GB/token | SSD wait {(s['wait_s']-s0['wait_s'])/20*1000:.0f} ms/token")
