"""Decode schedules on the real model, interleaved in one process: fully serial (no overlap, no
speculation), overlap of cached and missing experts, overlap + idle-window speculation. Then checks
that speculation does not change greedy outputs. Usage: schedule_ab.py CACHE_GB [rounds]"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepinfer_mac import LLM
from deepinfer_mac.model import model as M

llm = LLM(expert_cache_gb=float(sys.argv[1]))
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long essay about the history of tea in China, Japan and Britain."}])
CONF = {"serial": (False, 0), "overlap": (True, 0), "overlap+spec": (True, 3)}


def greedy(n):
    st, logits, _, _ = llm._prefill(ids, False)
    tok, out, ts = int(np.argmax(np.array(logits[-1]))), [], []
    for _ in range(n):
        t = time.perf_counter()
        lg = llm.model.forward(st, [tok])
        tok = int(np.argmax(np.array(lg[-1])))
        ts.append(time.perf_counter() - t)
        out.append(tok)
    return out, ts, st


_, _, st = greedy(6)  # warm
tok = 0
res = {k: [] for k in CONF}
for r in range(rounds):
    for name in (CONF if r % 2 == 0 else list(CONF)[::-1]):
        M.MoE.OVERLAP, M.MoE.SPEC_K = CONF[name]
        for _ in range(8):
            t = time.perf_counter()
            lg = llm.model.forward(st, [tok])
            tok = int(np.argmax(np.array(lg[-1])))
            res[name].append(time.perf_counter() - t)
for name in CONF:
    m = np.median(res[name])
    print(f"{name:13s}: median {m * 1e3:.0f} ms/token = {1 / m:.3f} tok/s (n={len(res[name])})", flush=True)
outs = {}
for k in (0, 3):
    M.MoE.OVERLAP, M.MoE.SPEC_K = True, k
    llm._state = llm._snap = None
    outs[k], _, _ = greedy(24)
print("greedy tokens identical with and without speculation:", outs[0] == outs[3], f"({len(outs[0])} tokens)")
