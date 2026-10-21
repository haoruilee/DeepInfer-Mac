"""Spot-check the converted Engram tables: random rows from the local 4-bit files vs. the original
fp8 rows range-read from the Hub. Catches offset/layout mistakes and measures quantization error."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlx.core as mx
from deepinfer_mac.model.engram import RemoteEngramTable, SSDEngramTable

root = Path("~/models/dsv41-mac").expanduser()
lay = json.loads((root / "layout.json").read_text())["engram"]
cfg = json.loads((root / "inference_config.json").read_text())
local = SSDEngramTable(root, cfg["engram_layer_ids"], dim=lay["dim"], bits=lay["bits"], group=lay.get("group", 64))
remote = RemoteEngramTable()
rng = np.random.default_rng(0)
worst = 0.0
for layer, rows in zip(cfg["engram_layer_ids"], cfg["engram_num_embeddings"]):
    ids = np.sort(rng.integers(0, rows, size=200)).reshape(1, -1)  # spread over the whole table
    a = np.array(local.lookup(layer, ids).astype(mx.float32))[0]
    b = np.array(remote.lookup(layer, ids).astype(mx.float32))[0]
    rel = ((a - b) ** 2).sum(1) / np.maximum((b ** 2).sum(1), 1e-12)
    cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    worst = max(worst, rel.max())
    print(f"layer {layer}: 200 rows | mean rel err {rel.mean():.4f}, max {rel.max():.4f} | min cos {cos.min():.4f}")
print("ENGRAM CHECK", "OK" if worst < 0.05 else "FAILED")
