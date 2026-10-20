import sys, numpy as np, torch, mlx.core as mx
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import test_parity as T
from deepinfer_mac.model import model as M
rm, mm = T.build(index_topk=64, candidate_topk_blocks=64)
toks = np.random.default_rng(0).integers(100, 120000, size=41).tolist()
ref_out = {}
def hook(name):
    def f(mod, inp, out):
        ref_out[name] = (out[0] if isinstance(out, tuple) else out).detach().float().numpy()
    return f
for i, b in enumerate(rm.layers):
    b.register_forward_hook(hook(f"L{i}"))
    b.attn.register_forward_hook(hook(f"A{i}"))
    b.ffn.register_forward_hook(hook(f"F{i}"))
    if b.engram is not None: b.engram.register_forward_hook(hook(f"E{i}"))
rm(torch.tensor([toks]), 0)
mine = {}
for i, b in enumerate(mm.layers):
    orig_call = b.__class__.__call__
    att, ffn = b.attn, b.ffn
    b.attn = (lambda f, i: (lambda *a, **k: mine.__setitem__(f"A{i}", f(*a, **k)) or mine[f"A{i}"]))(att, i)
    b.ffn = (lambda f, i: (lambda *a, **k: mine.__setitem__(f"F{i}", f(*a, **k)) or mine[f"F{i}"]))(ffn, i)
    if b.engram is not None:
        eg = b.engram
        b.engram = (lambda f, i: (lambda *a, **k: mine.__setitem__(f"E{i}", f(*a, **k)) or mine[f"E{i}"]))(eg, i)
orig = M.Block.__call__
def wrapped(self, *a, **k):
    out = orig(self, *a, **k); mine[f"L{self.layer_id}"] = out[0]; return out
M.Block.__call__ = wrapped
mm.forward(mm.new_state(), toks)
for k in ref_out:
    if k not in mine: continue
    a = ref_out[k].reshape(-1); b = np.array(mine[k].astype(mx.float32)).reshape(-1)
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b)); print(f"{k:4s} cos={cos:.6f}  relmax={np.abs(a-b).max()/np.abs(a).max():.4f}")
