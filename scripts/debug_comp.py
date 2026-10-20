import sys, numpy as np, torch, mlx.core as mx
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import test_parity as T
from deepinfer_mac.model import model as M
rm, mm = T.build(index_topk=64, candidate_topk_blocks=64)
toks = np.random.default_rng(0).integers(100, 120000, size=41).tolist()
cap = {}
orig_idx = T.ref.Indexer.forward
def idx_fwd(self, x, qr, latent, start_pos, offset):
    r = orig_idx(self, x, qr, latent, start_pos, offset); cap.setdefault("ref_topk", []).append(r.clone()); return r
T.ref.Indexer.forward = idx_fwd
rm(torch.tensor([toks]), 0)
st = mm.new_state()
orig_m = M.Attention._indexer
def my_idx(self, *a, **k):
    r = orig_m(self, *a, **k); cap.setdefault("my_topk", []).append(r); return r
M.Attention._indexer = my_idx
mm.forward(st, toks)
for L, li in ((2, 2), (4, 4)):
    ref_kv = rm.layers[L].attn.compress_kv_cache[0].float().numpy()
    my_kv = np.array(st.comp[L].kv.view().astype(mx.float32))
    n = my_kv.shape[0]
    a, b = ref_kv[:n].ravel(), my_kv.ravel()
    print(f"L{L} comp kv n={n} cos={a@b/np.linalg.norm(a)/np.linalg.norm(b):.6f} maxdiff={np.abs(a-b).max():.4f} absmax={np.abs(a).max():.3f}")
    ref_k = rm.layers[L].attn.indexer.k_cache[0].float().numpy()[:n]; my_k = np.array(st.comp[L].index_k.view().astype(mx.float32))
    a, b = ref_k.ravel(), my_k.ravel(); print(f"L{L} index k cos={a@b/np.linalg.norm(a)/np.linalg.norm(b):.6f} maxdiff={np.abs(a-b).max():.4f}")
for j, (r, m) in enumerate(zip(cap["ref_topk"], cap["my_topk"])):
    r = r[0].numpy(); m = np.array(m)
    off_r = r.max(); 
    print(f"indexer call {j}: ref shape {r.shape} mine {m.shape}")
    # compare sets per query, after removing offsets
    rs = [set(x[x >= 0].tolist()) for x in r]; ms = [set(x[x >= 0].tolist()) for x in m]
    print("  ref sample row 20:", sorted(rs[20])[:8], " mine:", sorted(ms[20])[:8])
