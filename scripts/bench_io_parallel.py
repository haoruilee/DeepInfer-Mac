"""SSD expert-read bandwidth vs. the number of reads in flight (I/O threads), F_NOCACHE, cold
experts drawn across all 40 layers so nothing is cached. Usage: bench_io_parallel.py"""
import sys, time
from pathlib import Path
import numpy as np, mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepinfer_mac.store import ExpertStore

root = Path("~/models/dsv41-mac").expanduser()
rng = np.random.default_rng(0)
for threads in [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2,4,8,16,32").split(",")]:
    s = ExpertStore(root, budget_bytes=2 << 30, threads=threads)
    keys = [k for k in s.layouts if k.startswith("L")]
    nbytes, t = 0, time.perf_counter()
    for step in range(30):
        key = keys[rng.integers(len(keys))]
        ids = rng.choice(384, 16, replace=False).tolist()
        ews = s.get(key, ids)
        mx.eval([p[0] for ew in ews for p in (ew.w1, ew.w3, ew.w2)])
        nbytes += 16 * s.layouts[key].nbytes
    dt = time.perf_counter() - t
    print(f"threads {threads:2d}: {nbytes / dt / 1e9:.2f} GB/s", flush=True)
    s.pool.shutdown()
