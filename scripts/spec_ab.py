"""Idle-window speculation A/B on the real model: decode speed with SPEC_K = 0 (off) and k next-layer
guesses read after each layer's demand misses, interleaved in one process. Also reports how many
speculative reads were used. Usage: spec_ab.py CACHE_GB K1,K2,... [rounds]"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepinfer_mac import LLM
from deepinfer_mac.model import model as M

llm = LLM(expert_cache_gb=float(sys.argv[1]))
ks = [int(k) for k in sys.argv[2].split(",")]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 4
store = llm.weights.experts
ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long essay about the history of tea in China, Japan and Britain."}])
st, logits, _, _ = llm._prefill(ids, False)
tok = int(np.argmax(np.array(logits[-1])))


def step():
    global tok
    t = time.perf_counter()
    lg = llm.model.forward(st, [tok])
    tok = int(np.argmax(np.array(lg[-1])))
    return time.perf_counter() - t


for _ in range(6):
    step()
res = {k: [] for k in ks}
use = {k: [0, 0, 0.0, 0, 0] for k in ks}  # prefetch hits, spec bytes, wait s, misses, hits
for r in range(rounds):
    for k in (ks if r % 2 == 0 else ks[::-1]):
        M.MoE.SPEC_K = k
        s0 = dict(store.stats)
        res[k] += [step() for _ in range(8)]
        s = store.stats
        use[k][0] += s["prefetch_hits"] - s0["prefetch_hits"]
        use[k][2] += s["wait_s"] - s0["wait_s"]
        use[k][3] += s["misses"] - s0["misses"]
        use[k][4] += s["hits"] - s0["hits"]
        print(f"round {r} k={k}: median {np.median(res[k][-8:]) * 1e3:.0f} ms", flush=True)
for k in ks:
    n = len(res[k])
    print(f"SPEC_K={k}: median {np.median(res[k]) * 1e3:.0f} ms/token = {1 / np.median(res[k]):.3f} tok/s | "
          f"SSD wait {use[k][2] / n * 1e3:.0f} ms/token | misses {use[k][3] / n:.1f}, hits {use[k][4] / n:.1f}, "
          f"speculative hits {use[k][0] / n:.1f} per token")
