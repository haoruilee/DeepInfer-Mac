import sys, numpy as np, torch, mlx.core as mx
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import test_parity as T
from deepinfer_mac.model import model as M
rm, mm = T.build(index_topk=64, candidate_topk_blocks=64)
toks = np.random.default_rng(0).integers(100, 120000, size=60).tolist()
P = 41
rm(torch.tensor([toks[:P]]), 0); st = mm.new_state(); mm.forward(st, toks[:P])
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 42
for i in range(P, STEP):
    rm(torch.tensor([[toks[i]]]), i); mm.forward(st, [toks[i]])
ref_out, mine = {}, {}
def hook(name):
    def f(mod, inp, out): ref_out[name] = (out[0] if isinstance(out, tuple) else out).detach().float().numpy()
    return f
for i, b in enumerate(rm.layers):
    b.register_forward_hook(hook(f"L{i}")); b.attn.register_forward_hook(hook(f"A{i}"))
orig = M.Block.__call__
def wrapped(self, *a, **k):
    out = orig(self, *a, **k); mine[f"L{self.layer_id}"] = out[0]; return out
M.Block.__call__ = wrapped
origA = M.Attention.__call__
def wa(self, *a, **k):
    o = origA(self, *a, **k); mine[f"A{self.layer_id}"] = o; return o
M.Attention.__call__ = wa
rm(torch.tensor([[toks[STEP]]]), STEP); mm.forward(st, [toks[STEP]])
for k in ref_out:
    a = ref_out[k].reshape(-1); b = np.array(mine[k].astype(mx.float32)).reshape(-1)
    print(f"{k:4s} cos={a@b/np.linalg.norm(a)/np.linalg.norm(b):.6f}", end="  ")
print()
# compare compressor caches after step
for L in (2, 4):
    c = st.comp[L]; n = c.kv.n
    a = rm.layers[L].attn.compress_kv_cache[0, :n].float().numpy().ravel(); b = np.array(c.kv.view().astype(mx.float32)).ravel()
    print(f"L{L} comp n={n} cos={a@b/np.linalg.norm(a)/np.linalg.norm(b):.6f}", "last rows diff:", np.abs(rm.layers[L].attn.compress_kv_cache[0, :n].float().numpy() - np.array(c.kv.view().astype(mx.float32))).max(axis=1)[-4:])
# isolate: capture layer-2 sparse_attn inputs on the next step
cap = {}
orig_sa = T.ref.sparse_attn
def sa(q, kv, sink, idx, scale):
    if "r" not in cap and kv.shape[1] > 8: cap["r"] = (q.clone(), kv.clone(), idx.clone())
    return orig_sa(q, kv, sink, idx, scale)
T.ref.sparse_attn = sa
origm = M.ops.sparse_attn
def msa(q, kv, sink, idx, scale, **k):
    if "m" not in cap and q.shape[1] == 8: pass
    cap.setdefault("ms", []).append((q, kv, idx)); return origm(q, kv, sink, idx, scale, **k)
M.ops.sparse_attn = msa
S2 = STEP + 1
rm(torch.tensor([[toks[S2]]]), S2); mm.forward(st, [toks[S2]])
rq, rkv, ridx = cap["r"]; mq, mkv, midx = [c for c in cap["ms"] if c[1].shape[0] > 9][0]
print("ref idx", ridx[0, 0].tolist()); print("my  idx", np.array(midx[0]).tolist())
print("ref kv rows", rkv.shape, "mine", mkv.shape)

def cmp(n, a, b):
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    print(f"{n}: cos={a@b/np.linalg.norm(a)/np.linalg.norm(b):.7f} maxdiff={np.abs(a-b).max():.5f}")
cmp("q", rq[0,0].float().numpy(), np.array(mq[0].astype(mx.float32)))
ri = ridx[0,0].numpy(); mi = np.array(midx[0])
rk = rkv[0].float().numpy(); mk = np.array(mkv.astype(mx.float32))
rows_r = rk[ri[ri>=0]]; rows_m = mk[mi[mi>=0]]
print("n sel", len(rows_r), len(rows_m))
# compare as multisets: sort rows by first element
cmp("selected kv (sorted)", rows_r[np.lexsort(rows_r.T[::-1])], rows_m[np.lexsort(rows_m.T[::-1])])
cmp("compressed part", rk[8:], mk[9:])
