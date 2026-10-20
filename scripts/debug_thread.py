import sys, tempfile, threading, traceback
from pathlib import Path
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import test_parity as T, tiny_ckpt
from deepinfer_mac.engine import LLM, SamplingParams
rm, _ = T.build()
root = tiny_ckpt.write(rm, T.tiny_args(), Path(tempfile.mkdtemp()), T.TOKENIZER)
llm = LLM(root, expert_cache_gb=0.05)
def run():
    try: print([o.token_id for o in llm.stream([0, 5, 6, 7], SamplingParams(temperature=0, max_tokens=3))])
    except Exception: traceback.print_exc()
t = threading.Thread(target=run); t.start(); t.join()
def probe():
    f = llm.model.layers[1].ffn
    for name in ("gate_w", "bias"):
        try: _ = (getattr(f, name) + 0).tolist() if False else mx.eval(getattr(f, name) * 1); print(name, "ok")
        except Exception as e: print(name, "FAIL", e)
import mlx.core as mx
t = threading.Thread(target=probe); t.start(); t.join()
print("bias_vl" , type(llm.model.layers[1].ffn.bias_vl))
