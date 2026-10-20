"""Offline inference, in the shape of DeepInfer's examples/offline_inference.py.

    python examples/offline_inference.py [--model ~/models/dsv41-mac] [--spec]
"""

import argparse
import math

from deepinfer_mac import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None, help="converted checkpoint (default: $DEEPINFER_HF_MODEL_DIR or ~/models/dsv41-mac)")
ap.add_argument("--spec", action="store_true", help="DSpark speculative decoding")
args = ap.parse_args()

llm = LLM(**({"model": args.model} if args.model else {}), speculative=args.spec or None)

conversations = [
    [{"role": "user", "content": "用三句话介绍一下杭州。"}],
    [{"role": "user", "content": "Write a haiku about solid-state drives."}],
]
for messages in conversations:
    prompt_ids, _ = llm.chat_prompt(messages)
    params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=200, seq_seed=0)
    output = llm.generate(prompt_token_ids=[prompt_ids], sampling_params=params)[0]
    text = llm.tokenizer.decode([o.token_id for o in output], skip_special_tokens=True)
    logps = [o.token_logprobs for o in output]
    n_target = sum(1 for o in output if o.draft_tok > 0)
    n_draft = sum(o.draft_tok for o in output) - n_target
    accepted = sum(o.accept_tok for o in output) - n_target
    rate = f"{100 * accepted / n_draft:.0f}%" if n_draft else "N/A"
    print(f"INPUT: {messages[-1]['content']!r}\nOUTPUT: {text!r}")
    print(f"output_tokens={len(output)}; accept_rate={rate}; decode_ppl={math.exp(-sum(logps) / len(logps)):.3f}")
    print(f"stats: {llm.last_stats}\n")
