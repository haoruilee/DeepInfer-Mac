import sys, tempfile, numpy as np
from pathlib import Path
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import test_parity as T, tiny_ckpt
from deepinfer_mac.engine import LLM, SamplingParams
rm, mm = T.build()
root = tiny_ckpt.write(rm, T.tiny_args(), Path(tempfile.mkdtemp()), T.TOKENIZER)
cos = lambda a, b: float(np.ravel(a) @ np.ravel(b) / np.linalg.norm(a) / np.linalg.norm(b))
for gb in (0.01, 1.0):
    llm = LLM(root, expert_cache_gb=gb, bounded_replay=False)
    p = np.random.default_rng(4).integers(100, 120000, size=30).tolist()
    sp = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    out1 = [o.token_id for o in llm.stream(p, sp)]
    p2 = p + out1 + [500, 600]
    st = llm._state
    lr = np.array(llm.model.forward(st, p2[len(st.tokens):]))
    llm.reset(); _, lf, _, _ = llm._prefill(p2, False)
    print(f"cache {gb} GB: reuse vs fresh cos={cos(lr, np.array(lf)):.6f}")
    # same without engine: fresh state, decode path
    st2 = llm.model.new_state(); llm.model.forward(st2, p)
    for t in out1[:-1]: llm.model.forward(st2, [t])
    l2 = np.array(llm.model.forward(st2, p2[len(st2.tokens):]))
    print(f"   manual decode-path vs fresh cos={cos(l2, np.array(lf)):.6f}  manual vs reuse {cos(l2, lr):.6f}")
