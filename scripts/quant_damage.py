"""How much do the converted (2/3-bit) experts change the model vs. the original lossless FP4
experts? Runs the first N real layers twice on a real prompt -- once with the converted expert
store, once with the routed experts' original FP4 tensors fetched from the Hub (bit-exact mxfp4) --
and reports per-layer hidden-state cosine."""
import json, sys
from pathlib import Path
import numpy as np, mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepinfer_mac import quant as Q
from deepinfer_mac.convert import hf
from deepinfer_mac.engine import build_chat
from deepinfer_mac.model.model import Args, ExpertWeights, Transformer
from deepinfer_mac.model.engram import NgramHasher
from deepinfer_mac.weights import DiskWeights

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
root = Path("~/models/dsv41-mac").expanduser()
cfg = json.loads((root / "inference_config.json").read_text())
INDEX = json.loads((Path(__file__).resolve().parent.parent / "deepinfer_mac/convert/tensor_index.json").read_text())


class Original:
    """Routed experts as shipped (FP4 + E8M0 == MLX mxfp4, lossless), fetched on demand."""
    def __init__(self): self.cache = {}
    def get(self, key, ids):
        out = []
        for e in ids:
            k = (key, int(e))
            if k not in self.cache:
                parts = []
                for w in ("w1", "w3", "w2"):
                    mw, ms = INDEX[f"layers.{int(key[1:])}.ffn.experts.{e}.{w}.weight"], INDEX[f"layers.{int(key[1:])}.ffn.experts.{e}.{w}.scale"]
                    q, s = Q.fp4_to_mxfp4(hf.as_array(hf.get_range(mw["shard"], mw["start"], mw["end"]), mw),
                                          hf.as_array(hf.get_range(ms["shard"], ms["start"], ms["end"]), ms))
                    parts.append((mx.array(q), mx.array(s), None, 4, "mxfp4"))
                self.cache[k] = ExpertWeights(*parts)
            out.append(self.cache[k])
        return out


a = Args.from_dict(cfg); a.n_layers = N
w = DiskWeights(root, a.dim, a.engram_layer_ids, expert_budget=3 << 30)
a.engram_layer_ids = tuple(l for l in a.engram_layer_ids if l < N)
hasher = NgramHasher.load(a, root, cfg["engram_vocab_size"])
ids, _, _ = build_chat(root, [{"role": "user", "content": "Explain in two sentences why rain leaves a fresh smell."}])
toks = list(ids)

def run(experts):
    from deepinfer_mac.model import model as M
    m = Transformer(a, w, hasher)
    for b in m.layers: b.ffn.experts = experts
    caps = {}
    orig = M.Block.__call__
    def wrapped(self, *args, **kw):
        out = orig(self, *args, **kw); caps[self.layer_id] = np.array(out[0].astype(mx.float32)); return out
    M.Block.__call__ = wrapped
    try:
        m.forward(m.new_state(), toks)
    finally:
        M.Block.__call__ = orig
    return caps

conv = run(w.experts)
orig = run(Original())
for i in range(N):
    x, y = conv[i].ravel(), orig[i].ravel()
    lay = json.loads((root / "layout.json").read_text())["experts"][f"L{i:02d}"]
    print(f"layer {i} ({lay['bits_w1']}/{lay['bits_w3']}/{lay['bits_w2']}-bit experts): hidden cos vs original FP4 = {x @ y / np.linalg.norm(x) / np.linalg.norm(y):.5f}", flush=True)
