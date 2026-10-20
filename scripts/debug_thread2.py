import sys, tempfile, threading, traceback
from pathlib import Path
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import test_parity as T, tiny_ckpt, mlx.core as mx, numpy as np
from deepinfer_mac.engine import LLM
rm, _ = T.build()
root = tiny_ckpt.write(rm, T.tiny_args(), Path(tempfile.mkdtemp()), T.TOKENIZER)
llm = LLM(root, expert_cache_gb=0.05)
def run():
    f = llm.model.layers[0].ffn; nxt = f.next_moe
    x = mx.ones((1, llm.args.dim), dtype=mx.bfloat16)
    for name, fn in [("gate_w", lambda: nxt.gate_w), ("bias", lambda: nxt.bias), ("matmul", lambda: x.astype(mx.float32) @ nxt.gate_w.T),
                     ("plus bias", lambda: x.astype(mx.float32) @ nxt.gate_w.T + nxt.bias),
                     ("argpartition", lambda: mx.argpartition(-(x.astype(mx.float32) @ nxt.gate_w.T + nxt.bias), kth=1, axis=-1))]:
        try: np.array(fn()); print(name, "ok")
        except Exception as e: print(name, "FAIL", e)
t = threading.Thread(target=run); t.start(); t.join()
from deepinfer_mac.model import model as M
orig = M.MoE.__call__
def wrapped(self, x, image_mask=None):
    for name, a in [("x", x)]:
        try: mx.eval(a)
        except Exception as e: print(self.key, "input x FAIL", e)
    return orig(self, x, image_mask)
M.MoE.__call__ = wrapped
oa = M.Attention.__call__
def wa(self, x, positions, st, ring, sh, start):
    for name, a in [("x", x), ("positions", positions), ("ring", ring.kv)]:
        try: mx.eval(a)
        except Exception as e: print(self.layer_id, "attn", name, "FAIL", e)
    return oa(self, x, positions, st, ring, sh, start)
M.Attention.__call__ = wa
def run2():
    try: list(llm.stream([0, 5, 6, 7], __import__("deepinfer_mac.engine", fromlist=["x"]).SamplingParams(temperature=0, max_tokens=2)))
    except Exception as e: print("stream FAIL", e)
t = threading.Thread(target=run2); t.start(); t.join()
