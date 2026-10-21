"""Long-prompt prefill: time, SSD bytes and peak memory per chunk size. Chunked prefill re-reads
the encoder's experts once per chunk, so bigger chunks should approach one sweep of the experts."""
import sys, time
from pathlib import Path
import mlx.core as mx, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepinfer_mac import LLM

V = Path("~/models/api-vectors/deepseek-v4.1-flash-20260910/prompts/case_014.txt").expanduser()  # 16383-token retrieval case
llm = LLM()
ids, _ = llm.chat_prompt([{"role": "user", "content": V.read_text()}])
print(f"prompt {len(ids)} tokens; expert cache {llm.weights.experts.budget / 2**30:.1f} GB", flush=True)
for chunk in [int(c) for c in (sys.argv[1:] or ["4096", "16384"])]:
    llm.reset(); llm.prefill_chunk = max(chunk, llm.keep_tokens)
    mx.reset_peak_memory(); s0 = dict(llm.weights.experts.stats); t = time.perf_counter()
    st, logits, _, _ = llm._prefill(ids, False); mx.eval(logits)
    dt = time.perf_counter() - t; s = llm.weights.experts.stats
    ans = llm.tokenizer.decode([int(np.argmax(np.array(logits[-1])))])
    print(f"chunk {chunk:6d}: {dt:6.0f}s ({len(ids)/dt:.0f} tok/s) | experts read {(s['bytes']-s0['bytes'])/1e9:.0f} GB | "
          f"peak MLX {mx.get_peak_memory()/2**30:.1f} GB | next token {ans!r}", flush=True)
