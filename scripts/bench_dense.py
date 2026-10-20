"""Real dense weights, stand-in experts (one random expert of real shape, in memory) and a zero
engram: measures per-token compute and resident memory without the SSD."""
import json, sys, time, numpy as np, mlx.core as mx
sys.path.insert(0, ".")
from pathlib import Path
from deepinfer_mac.model.model import Args, Transformer, ExpertWeights
from deepinfer_mac.model.engram import NgramHasher
root = Path("~/models/dsv41-mac").expanduser()
cfg = json.loads((root / "inference_config.json").read_text()); a = Args.from_dict(cfg)
def q(o, i, bits):
    w = (mx.random.normal((o, i)) * 0.02).astype(mx.bfloat16)
    return (*mx.quantize(w, group_size=64, bits=bits), bits, "affine")
EW = ExpertWeights(q(2304, 5120, 2), q(2304, 5120, 2), q(5120, 2304, 3))
class Experts:
    def get(self, key, ids): return [EW] * len(ids)
class Engram:
    def lookup(self, layer, ids): return mx.zeros((*ids.shape, 256), dtype=mx.bfloat16)
class W:
    globals = mx.load(str(root / "dense/globals.safetensors"))
    experts, engram = Experts(), Engram()
    def layer(self, i): return mx.load(str(root / f"dense/L{i:02d}.safetensors"))
    _e = np.memmap(root / "embed.bf16", dtype=np.uint16, mode="r").reshape(-1, a.dim)
    def embed(self, ids): return mx.array(np.ascontiguousarray(self._e[ids])).view(mx.bfloat16)
t = time.time()
hasher = NgramHasher.load(a, root, cfg["engram_vocab_size"])
print(f"token map {time.time()-t:.1f}s"); t = time.time()
m = Transformer(a, W(), hasher)
print(f"load {time.time()-t:.1f}s  active mem {mx.get_active_memory()/2**30:.2f} GB  peak {mx.get_peak_memory()/2**30:.2f} GB")
st = m.new_state()
for L in (512, 2048):
    toks = np.random.default_rng(0).integers(1000, 100000, size=L).tolist()
    t = time.time(); m.forward(st, toks, replay_from=L - 140); print(f"prefill {L} (bounded replay): {time.time()-t:.2f}s  -> {L/(time.time()-t):.0f} tok/s compute-only")
    t = time.time(); m.forward(m.new_state(), toks); print(f"prefill {L} (full): {time.time()-t:.2f}s")
st = m.new_state(); m.forward(st, toks[:300])
ts = []
for i in range(8):
    t = time.time(); mx.eval(m.forward(st, [toks[300 + i]])); ts.append(time.time() - t)
print(f"decode compute per token: {np.median(ts)*1000:.0f} ms (median of 8)  peak mem {mx.get_peak_memory()/2**30:.2f} GB")
