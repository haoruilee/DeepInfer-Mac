"""Decode schedule ablation on the real model, all configurations interleaved in one process, over
several prompts and rounds; raw per-token times and store counters go to OUT.json.
Usage: spec_ablation.py OUT.json CACHE_GB [rounds] [tokens_per_round] [configs]"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepinfer_mac import LLM
from deepinfer_mac.model import model as M
from deepinfer_mac.store import ExpertStore

out_path, cache = sys.argv[1], float(sys.argv[2])
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 4
per = int(sys.argv[4]) if len(sys.argv) > 4 else 6
# name: (OVERLAP, LOOKAHEAD_K (naive prefetch), SPEC_K, SPEC_WAIT, SPEC_COLD)
ALL = {
    "serial": (False, 0, 0, True, True),
    "overlap": (True, 0, 0, True, True),
    "naive-prefetch": (True, 3, 0, True, True),
    "spec-neither": (True, 0, 3, False, False),
    "spec-wait-only": (True, 0, 3, True, False),
    "spec-cold-only": (True, 0, 3, False, True),
    "idle-window": (True, 0, 3, True, True),
}
names = sys.argv[5].split(",") if len(sys.argv) > 5 else list(ALL)
PROMPTS = [
    "Write a long essay about the history of tea in China, Japan and Britain.",
    "Explain step by step how a compiler turns C code into machine code, with examples.",
    "用中文详细介绍一下长江流域的地理、历史和经济。",
]
llm = LLM(expert_cache_gb=cache)
store = llm.weights.experts


def setconf(name):
    ov, la, sk, wait, cold = ALL[name]
    M.MoE.OVERLAP, M.MoE.LOOKAHEAD, M.MoE.LOOKAHEAD_K, M.MoE.SPEC_K = ov, la > 0, la, sk
    ExpertStore.SPEC_WAIT, ExpertStore.SPEC_COLD = wait, cold


data = {"cache_gb": cache, "configs": {n: ALL[n] for n in names}, "runs": []}
for pi, prompt in enumerate(PROMPTS):
    llm._state = llm._snap = None
    ids, _ = llm.chat_prompt([{"role": "user", "content": prompt}])
    st, logits, _, _ = llm._prefill(ids, False)
    tok = int(np.argmax(np.array(logits[-1])))
    setconf("idle-window")
    for _ in range(4):  # warm the cache for this prompt
        tok = int(np.argmax(np.array(llm.model.forward(st, [tok])[-1])))
    for r in range(rounds):
        order = names[r % len(names):] + names[: r % len(names)]  # rotate the order every round
        for name in order:
            setconf(name)
            s0 = dict(store.stats)
            ts = []
            for _ in range(per):
                t = time.perf_counter()
                tok = int(np.argmax(np.array(llm.model.forward(st, [tok])[-1])))
                ts.append(time.perf_counter() - t)
            s = store.stats
            data["runs"].append({"prompt": pi, "round": r, "config": name, "ms": [x * 1e3 for x in ts],
                                 **{k: s[k] - s0[k] for k in ("hits", "misses", "prefetch_hits", "bytes", "spec_bytes", "wait_s")}})
        Path(out_path).write_text(json.dumps(data))
        print(f"prompt {pi} round {r} done", flush=True)
for name in names:
    rs = [x for x in data["runs"] if x["config"] == name]
    med = [np.median(x["ms"]) for x in rs]
    n = sum(len(x["ms"]) for x in rs)
    print(f"{name:15s} tok/s mean {np.mean([1e3 / m for m in med]):.3f} ± {np.std([1e3 / m for m in med]):.3f} (n_blocks={len(rs)}) | "
          f"wait {sum(x['wait_s'] for x in rs) / n * 1e3:.0f} ms/tok | demand GB/tok {(sum(x['bytes'] - x['spec_bytes'] for x in rs)) / n / 1e9:.2f} | "
          f"spec GB/tok {sum(x['spec_bytes'] for x in rs) / n / 1e9:.2f} | spec hits/tok {sum(x['prefetch_hits'] for x in rs) / n:.1f}")
