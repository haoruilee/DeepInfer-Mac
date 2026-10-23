"""Save the indexer's inputs (queries, head weights, keys, candidate mask) of a long real prompt,
for studying exact pruning (DoSA) offline. Usage: capture_indexer.py OUT.npz [n_tokens]"""
import sys, glob
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepinfer_mac import LLM
from deepinfer_mac.model.model import Attention

out, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8192
llm = LLM()
text = "\n\n".join(Path(p).read_text() for p in sorted(glob.glob(str(Path("~/models/api-vectors/*/prompts/*.txt").expanduser()))))
ids = llm.encode(text)[:n]
saved, layer_of = {}, {id(b.attn): i for i, b in enumerate(llm.model.layers)}
orig = Attention._index_block

def hook(self, q, w, visible, index_k, T, sh, b0, b1):
    i = layer_of[id(self)]
    if q.shape[0] > 1:  # prefill blocks: keep the last one of size >= 64 rows as layer i + 1000
        if q.shape[0] < 64:
            return orig(self, q, w, visible, index_k, T, sh, b0, b1)
        i += 1000
    c = sh.candidates[b0:b1] if self.uses_candidates else None
    saved[i] = dict(q=np.array(q.astype(mx.float32)), w=np.array(w.astype(mx.float32)), visible=np.array(visible),
                    k=np.array(index_k), ratio=self.ratio, src=int(self.is_candidate_source),
                    **({"cand": np.array(c)} if c is not None else {}))
    return orig(self, q, w, visible, index_k, T, sh, b0, b1)

Attention._index_block = hook
llm._prefill(ids, False)
np.savez(out, **{f"L{i}_{k}": v for i, d in saved.items() for k, v in d.items()})
print("saved layers", sorted(saved), {i: d["q"].shape + d["k"].shape for i, d in saved.items()})
