"""Prefill with and without overlapping the next expert batch's SSD reads with the current batch's
compute (MoE._slab_moe prefetches batch i+1 while batch i runs). Interleaved A/B on a ~1K-token
prompt. Usage: prefill_overlap.py [rounds]"""
import glob, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepinfer_mac import LLM

import os
llm = LLM(expert_cache_gb=float(os.environ["CACHE_GB"]) if "CACHE_GB" in os.environ else None)
store = llm.weights.experts
text = "\n\n".join(Path(p).read_text() for p in sorted(glob.glob(str(Path("~/models/api-vectors/*/prompts/*.txt").expanduser()))))
ids = llm.encode(text)[: int(sys.argv[2]) if len(sys.argv) > 2 else 1024]
orig = store.prefetch


def no_optional(key, group, optional=False):
    if not optional:
        orig(key, group)


res = {True: [], False: []}
for r in range(int(sys.argv[1]) if len(sys.argv) > 1 else 2):
    for on in (True, False) if r % 2 == 0 else (False, True):
        store.prefetch = orig if on else no_optional
        llm._state = llm._snap = None
        t = time.perf_counter()
        llm._prefill(ids, False)
        res[on].append(time.perf_counter() - t)
        print(f"pipelined {'on ' if on else 'off'}: {res[on][-1]:.1f}s", flush=True)
for on in (True, False):
    print(f"mean pipelined {'on ' if on else 'off'}: {sum(res[on]) / len(res[on]):.1f}s for {len(ids)} tokens")
