"""Decode concurrency on the real model: (1) throughput with the hit/miss overlap on vs. off,
interleaved in one process; (2) a per-layer timeline of decode steps -- when the router's expert
ids reach the host, each SSD read (per I/O thread), the GPU work on resident experts and on the
misses. Writes the trace to OUT.json. Usage: trace_decode.py OUT.json [CACHE_GB] [SPEC_K]"""
import json, sys, threading, time
from pathlib import Path
import numpy as np, mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepinfer_mac import LLM
from deepinfer_mac.model import model as M

llm = LLM(expert_cache_gb=float(sys.argv[2]) if len(sys.argv) > 2 else None)
M.MoE.SPEC_K = int(sys.argv[3]) if len(sys.argv) > 3 else 0
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


for _ in range(4):  # warm the cache
    step()
res = {True: [], False: []}
for rnd in range(4):
    for ov in (True, False) if rnd % 2 == 0 else (False, True):
        M.MoE.OVERLAP = ov
        res[ov] += [step() for _ in range(8)]
M.MoE.OVERLAP = True
for ov in (True, False):
    print(f"overlap {'on ' if ov else 'off'}: median {np.median(res[ov]) * 1e3:.0f} ms/token = {1 / np.median(res[ov]):.2f} tok/s (n={len(res[ov])})", flush=True)

# ---- timeline
ev, T0 = [], [0.0]
lock = threading.Lock()
orig_read, orig_slab, orig_into, orig_slots = store._read, M.MoE._slab_moe, M.MoE._experts_into, store.get_slots


def rec(**k):
    with lock:
        ev.append(k)


def read(key, e, slot):
    a = time.perf_counter()
    orig_read(key, e, slot)
    rec(kind="read", layer=key, thread=threading.current_thread().name, t0=a - T0[0], t1=time.perf_counter() - T0[0])


def slab_moe(self, x, xq, w, idx_np, y):
    rec(kind="route", layer=self.key, t0=time.perf_counter() - T0[0],
        n=int((idx_np >= 0).sum()), hits=sum(store.resident(self.key, int(e)) for e in np.unique(idx_np) if e >= 0))
    out = orig_slab(self, x, xq, w, idx_np, y)
    return out


def into(self, y, x, xq, w, idx_np, group):
    a = time.perf_counter()
    y = orig_into(self, y, x, xq, w, idx_np, group)
    mx.eval(y)  # timestamp the GPU work (the reads keep running on the I/O threads meanwhile)
    rec(kind="gpu", layer=self.key, t0=a - T0[0], t1=time.perf_counter() - T0[0], n=len(group))
    return y


def slots(key, group):
    a = time.perf_counter()
    r = orig_slots(key, group)
    rec(kind="wait", layer=key, t0=a - T0[0], t1=time.perf_counter() - T0[0])
    return r


store._read, store.get_slots = read, slots
M.MoE._slab_moe, M.MoE._experts_into = slab_moe, into
tokens = []
for i in range(3):
    ev.clear()
    T0[0] = time.perf_counter()
    dt = step()
    tokens.append({"total": dt, "events": list(ev)})
    print(f"traced token {i}: {dt * 1e3:.0f} ms, {sum(e['kind'] == 'read' for e in ev)} reads", flush=True)
Path(sys.argv[1]).write_text(json.dumps({"overlap": {str(k): v for k, v in res.items()}, "tokens": tokens}))
