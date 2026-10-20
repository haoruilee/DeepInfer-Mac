import sys, tempfile, numpy as np
from pathlib import Path
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import test_parity as T, tiny_ckpt
from deepinfer_mac.engine import LLM
rm, mm = T.build()
root = tiny_ckpt.write(rm, T.tiny_args(), Path(tempfile.mkdtemp()), T.TOKENIZER)
cos = lambda a, b: float(np.ravel(a) @ np.ravel(b) / np.linalg.norm(a) / np.linalg.norm(b))
toks = np.random.default_rng(4).integers(100, 120000, size=40).tolist()
def check(name, model):
    full = np.array(model.forward(model.new_state(), toks))
    st = model.new_state(); model.forward(st, toks[:30])
    for t in toks[30:]: out = np.array(model.forward(st, [t]))
    print(f"{name}: decode-vs-prefill cos={cos(full, out):.6f}")
llm = LLM(root, expert_cache_gb=1.0, bounded_replay=False)
check("disk all", llm.model)
mem_engram = mm.w.engram
for b in llm.model.layers:
    if b.engram is not None: b.engram.table = mem_engram
check("disk experts + memory engram", llm.model)
for b in llm.model.layers:
    if b.engram is not None: b.engram.table = llm.weights.engram
    b.ffn.experts = mm.w
check("memory experts + disk engram", llm.model)
from deepinfer_mac.model import model as M
for b in llm.model.layers: b.ffn.experts = llm.weights.experts
rec = []
orig = M.MoE.route
def route(self, x, image_mask=None):
    w, idx = orig(self, x, image_mask)
    sc = mx.sqrt(mx.logaddexp(x.astype(mx.float32) @ self.gate_w.T, 0.0)) + self.bias
    rec.append((self.key, np.array(idx), np.array(sc)))
    return w, idx
import mlx.core as mx
M.MoE.route = route

rec.clear(); llm.model.forward(llm.model.new_state(), toks); A = {k: (i, s) for k, i, s in rec}
st = llm.model.new_state(); llm.model.forward(st, toks[:30])
B = {}
for pos, t in enumerate(toks[30:], start=30):
    rec.clear(); llm.model.forward(st, [t])
    for k, i, s in rec: B.setdefault(k, []).append((pos, i[0], s[0]))
flips = 0
for k, rows in B.items():
    for pos, ib, sb in rows:
        ia = A[k][0][pos]; sa = A[k][1][pos]
        if set(ia.tolist()) != set(ib.tolist()):
            flips += 1
            srt = np.sort(sa)[::-1]
            print(f"{k} pos {pos}: prefill {sorted(ia.tolist())} decode {sorted(ib.tolist())} prefill top-3 scores {np.round(srt[:3],4)} decode {np.round(np.sort(sb)[::-1][:3],4)}")
print("flips:", flips)
