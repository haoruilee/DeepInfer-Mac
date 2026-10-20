import sys, numpy as np, torch, mlx.core as mx
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import test_parity as T
from deepinfer_mac.model import model as M
rm, mm = T.build(index_topk=64, candidate_topk_blocks=64)
toks = np.random.default_rng(0).integers(100, 120000, size=41).tolist()
cap = {}
def pre_hook(mod, args): cap["x"] = args[0].clone()
def hook(mod, args, out): cap["o"] = out.clone()
rm.layers[2].attn.register_forward_pre_hook(pre_hook); rm.layers[2].attn.register_forward_hook(hook)
# also capture sparse_attn inputs
orig_sa = T.ref.sparse_attn
def sa(q, kv, sink, idx, scale):
    o = orig_sa(q, kv, sink, idx, scale)
    if q.shape[1] == 41 and kv.shape[1] > 41 and "sa" not in cap: cap["sa"] = (q.clone(), kv.clone(), idx.clone(), o.clone())
    return o
T.ref.sparse_attn = sa
rm(torch.tensor([toks]), 0)
x = T._mx(cap["x"][0])
st = mm.new_state(); sh = M.Shared()
att = mm.layers[2].attn
mine_cap = {}
orig_msa = M.ops.sparse_attn
def msa(q, kv, sink, idx, scale, **k):
    o = orig_msa(q, kv, sink, idx, scale, **k); mine_cap["sa"] = (q, kv, idx, o); return o
M.ops.sparse_attn = msa
o = att(x, mx.arange(41), st, st.rings[2], sh)
def cmp(n, a, b):
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    print(f"{n}: cos={a@b/np.linalg.norm(a)/np.linalg.norm(b):.7f} maxdiff={np.abs(a-b).max():.5f}")
cmp("attn out", cap["o"][0].float().numpy(), np.array(o.astype(mx.float32)))
rq, rkv, ridx, ro = cap["sa"]; mq, mkv, midx, mo = mine_cap["sa"]
cmp("q", rq[0].float().numpy(), np.array(mq.astype(mx.float32)))
cmp("kv window", rkv[0, :41].float().numpy(), np.array(mkv[8:49].astype(mx.float32)))
cmp("kv comp", rkv[0, 41:].float().numpy(), np.array(mkv[49:].astype(mx.float32)))
cmp("sa out", ro[0].float().numpy(), np.array(mo.astype(mx.float32)))
print("ridx row 30", ridx[0,30].tolist()); print("midx row 30", np.array(midx[30]).tolist())
# same inputs through both implementations
with torch.no_grad(): o_ref = T.ref_kernels.sparse_attn(rq, rkv, rm.layers[2].attn.attn_sink, ridx, att.scale)
o_mine = orig_msa(T._mx(rq[0]), T._mx(rkv[0]), mx.array(rm.layers[2].attn.attn_sink.detach().float().numpy()), mx.array(ridx[0].numpy()), att.scale)
cmp("same-input ref-emul vs mlx", o_ref[0].float().numpy(), np.array(o_mine.astype(mx.float32)))
# brute force in float64
q64, kv64, sink = rq[0].double(), rkv[0].double(), rm.layers[2].attn.attn_sink.detach().double()
out = torch.zeros_like(q64)
for t in range(41):
    ii = ridx[0, t]; ii = ii[ii >= 0]
    g = kv64[ii]; s = q64[t] @ g.T * att.scale
    m = s.max(-1, keepdim=True).values
    p = torch.exp(s - m); den = p.sum(-1, keepdim=True) + torch.exp(sink[:, None] - m)
    out[t] = (p @ g) / den
cmp("float64 vs ref-emul", out.numpy(), o_ref[0].float().numpy())
cmp("float64 vs mlx", out.numpy(), np.array(o_mine.astype(mx.float32)))
r = ridx[0].numpy(); m = np.array(midx)
conv = lambda x: sorted(int(v - 8) if v < 49 else int(v - 8) for v in x if v >= 0)
bad = [t for t in range(41) if sorted(r[t][r[t] >= 0].tolist()) != conv(m[t])]
print("rows differing:", bad[:10], len(bad))
for t in bad[:3]: print(t, sorted(r[t][r[t]>=0].tolist()), conv(m[t]))
