"""Decode speed on the converted layers with real SSD expert reads (extrapolated to 40 layers)."""
import sys, time, numpy as np, mlx.core as mx
sys.argv = [sys.argv[0], "6"]
sys.path.insert(0, "scripts")
import real_check as R
from deepinfer_mac.model import model as M
from deepinfer_mac.engine import LLM
mx.set_cache_limit(1 << 30)
ours = R.build_ours(6)
ids, _ = LLM.chat_prompt(type("X", (), {"root": R.ROOT})(), [{"role": "user", "content": "Write a short story about a lighthouse keeper who finds a message in a bottle."}])
toks = list(ids)
orig = M.Block.__call__
BT = []
def timed(self, *a, **k):
    t = time.perf_counter(); out = orig(self, *a, **k); mx.eval(out); BT.append(time.perf_counter() - t); return out
M.Block.__call__ = timed
# warm the remote engram rows for the exact token sequence first (greedy is deterministic)
st = ours.new_state(); ours.forward(st, toks); tok = 1000
for i in range(24):
    lg = ours.forward(st, [tok]); tok = int(np.argmax(np.array(lg[-1])))
for look in (False, True, False, True):
    M.MoE.LOOKAHEAD = look
    R.STORE.cache.clear(); R.STORE.used = 0
    st = ours.new_state(); ours.forward(st, toks)
    R.STORE.stats.update(hits=0, misses=0, bytes=0, wait_s=0.0)
    rng = np.random.default_rng(0); ts = []
    tok = 1000
    for i in range(24):
        t = time.perf_counter(); lg = ours.forward(st, [tok]); mx.eval(lg); ts.append(time.perf_counter() - t)
        tok = int(np.argmax(np.array(lg[-1])))
    BT.clear()
    st = ours.new_state(); ours.forward(st, toks); BT.clear(); tok = 1000
    R.STORE.cache.clear(); R.STORE.used = 0; R.STORE.stats.update(hits=0, misses=0, bytes=0, wait_s=0.0)
    for i in range(24):
        lg = ours.forward(st, [tok]); tok = int(np.argmax(np.array(lg[-1])))
    per = np.median(BT[6 * 4 :]) 
    s = R.STORE.stats
    print(f"lookahead={look}: {per*1000:.1f} ms/layer -> {per*40:.2f} s/token over 40 layers (~{1/(per*40):.2f} tok/s); "
          f"hit rate {s['hits']/(s['hits']+s['misses']):.2f}, IO wait {s['wait_s']/24/6*1000:.1f} ms/layer, read {s['bytes']/1e9:.1f} GB")
