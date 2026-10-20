import sys, numpy as np, torch, mlx.core as mx
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import test_parity as T
rm, mm = T.build(index_topk=64, candidate_topk_blocks=64)
toks = np.random.default_rng(0).integers(100, 120000, size=60).tolist()
rm(torch.tensor([toks[:41]]), 0); st = mm.new_state(); mm.forward(st, toks[:41])
for L in range(6):
    r = rm.layers[L].attn.window_kv_cache[0].float().numpy(); m = np.array(st.rings[L].kv.astype(mx.float32))
    print(L, "slot maxdiff", np.abs(r - m).max(axis=1).round(3), "pos", st.rings[L].pos)
