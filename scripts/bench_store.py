"""SSD expert streaming throughput on converted layers: decode-like fetches of 6 random experts."""
import sys, time, numpy as np, mlx.core as mx
sys.path.insert(0, ".")
from deepinfer_mac.store import ExpertStore
s = ExpertStore("/Users/harvey/models/dsv41-mac", budget_bytes=64 << 20, threads=8)
rng = np.random.default_rng(0)
keys = ["L00", "L01"]
for ids_per in (6, 12, 32):
    t = time.perf_counter(); n = 0
    for step in range(20):
        ids = rng.choice(384, ids_per, replace=False).tolist()
        ews = s.get(keys[step % 2], ids); mx.eval([p[0] for ew in ews for p in (ew.w1, ew.w3, ew.w2)]); n += ids_per
    dt = time.perf_counter() - t
    print(f"{ids_per} experts/fetch: {dt/20*1000:.1f} ms per fetch, {n*s.layouts['L00'].nbytes/dt/1e9:.2f} GB/s")
