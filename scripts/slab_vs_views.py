import sys, numpy as np, mlx.core as mx
sys.argv = [sys.argv[0], "6"]; sys.path.insert(0, "scripts")
import real_check as R
from deepinfer_mac.model import model as M
from deepinfer_mac.engine import LLM
ids, _ = LLM.chat_prompt(type("X", (), {"root": R.ROOT})(), [{"role": "user", "content": "用一句话介绍一下杭州。Then say it in English."}])
toks = list(ids)
import test_parity; test_parity._no_quant(True)
ours = R.build_ours(6)
class Views:  # exposes only get(): forces MoE's per-expert path over the same store
    def get(self, key, ids): return R.STORE.get(key, ids)
def run(slab):
    for blk in ours.layers: blk.ffn.experts = R.STORE if slab else Views()
    out, st = [], ours.new_state()
    lg = ours.forward(st, toks[:12]); out.append(np.array(lg[-1]))
    for t in toks[12:]: out.append(np.array(ours.forward(st, [t])[-1]))
    return out
a, b = run(True), run(False)
for i, (x, y) in enumerate(zip(a, b)):
    print(f"step {i}: slab vs per-expert logits cos {float(x@y/np.linalg.norm(x)/np.linalg.norm(y)):.6f} top1 {x.argmax()} {y.argmax()}")
print("hit rate", R.STORE.hit_rate(), R.STORE.stats)
REC = {}
orig = M.MoE._route_shared
def rs(self, x, bias):
    out = orig(self, x, bias)
    scores = np.array(mx.sqrt(mx.logaddexp(x.astype(mx.float32) @ self.gate_w.T, 0.0)) + bias)
    REC.setdefault(TAG, []).append((self.key, sorted(np.array(out[1])[-1].tolist()), np.sort(scores[-1])[::-1][:8]))
    return out
for blk in ours.layers: blk.ffn._rs_c = lambda x, b, f=blk.ffn: rs(f, x, b)
TAG = "slab"; run(True); TAG = "views"; run(False)
n = len(REC["slab"]) // 4
for (k, a, sa), (_, b, sb) in zip(REC["slab"][-6 * 1:], REC["views"][-6 * 1:]):
    print(k, a, b, "FLIP" if a != b else "", "6th/7th score gap", round(float(sa[5] - sa[6]), 5))
print("records", len(REC["slab"]), len(REC["views"]))
for i, ((k, a, sa), (_, b, sb)) in enumerate(zip(REC["slab"], REC["views"])):
    if a != b: print("FLIP at record", i, k, a, b, "gap", round(float(sa[5] - sa[6]), 6))
