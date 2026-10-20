import sys, tempfile, threading
from pathlib import Path
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import test_parity as T, tiny_ckpt, mlx.core as mx
from deepinfer_mac.engine import LLM
rm, _ = T.build()
root = tiny_ckpt.write(rm, T.tiny_args(), Path(tempfile.mkdtemp()), T.TOKENIZER)
llm = LLM(root, expert_cache_gb=0.05)
def walk(o, path, out, seen, d=0):
    if d > 5 or id(o) in seen: return
    seen.add(id(o))
    if isinstance(o, mx.array): out.append((path, o)); return
    if isinstance(o, dict): [walk(v, f"{path}[{k}]", out, seen, d+1) for k, v in o.items()]
    elif isinstance(o, (list, tuple)): [walk(v, f"{path}[{i}]", out, seen, d+1) for i, v in enumerate(o)]
    elif hasattr(o, "__dict__"): [walk(v, f"{path}.{k}", out, seen, d+1) for k, v in vars(o).items() if k not in ("next_moe", "experts", "table")]
def run():
    arrs = []; walk(llm.model.layers[2], "L2", arrs, set())
    for p, a in arrs:
        try: mx.eval(a)
        except Exception as e: print("FAIL", p)
    print("checked", len(arrs))
t = threading.Thread(target=run); t.start(); t.join()
