import sys, numpy as np
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import test_parity as T
rm, mm = T.build()
toks = np.random.default_rng(4).integers(100, 120000, size=40).tolist()
cos = lambda a, b: float(np.ravel(a) @ np.ravel(b) / np.linalg.norm(a) / np.linalg.norm(b))
full = np.array(mm.forward(mm.new_state(), toks))
for split in (30, 37, 39):
    st = mm.new_state(); mm.forward(st, toks[:split])
    for t in toks[split:-3] if split < 37 else []: mm.forward(st, [t])
    rest = toks[st.pos:]
    out = np.array(mm.forward(st, rest))
    print(f"prefill {split} + decode to {st.pos - len(rest)} + chunk {len(rest)}: cos={cos(full, out):.6f} top1 {full.argmax()} vs {out.argmax()}")
st = mm.new_state(); mm.forward(st, toks[:30])
for t in toks[30:]: out = np.array(mm.forward(st, [t]))
print(f"prefill 30 + 10 decode: cos={cos(full, out):.6f}")
