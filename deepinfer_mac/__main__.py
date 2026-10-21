"""deepinfer-mac command line.

  python -m deepinfer_mac convert --out ~/models/dsv41-mac      # stream + convert from the Hub
  python -m deepinfer_mac chat [--think]                         # interactive multi-turn chat
  python -m deepinfer_mac serve [--port 8000]                    # OpenAI/Anthropic-compatible API
  python -m deepinfer_mac bench [--prompt-len 512 --gen 64]      # throughput numbers
  python -m deepinfer_mac doctor                                 # readiness + conversion status
"""

import argparse
import json
import sys
import time

from .engine import DEFAULT_MODEL_DIR


def _llm(a):
    from .engine import LLM
    from .model.model import MoE

    MoE.SKIP_MISS_BELOW = a.skip_miss
    MoE.PREFILL_TOPK = a.prefill_topk
    MoE.LOOKAHEAD, MoE.LOOKAHEAD_K = a.lookahead > 0, a.lookahead
    return LLM(a.model, expert_cache_gb=a.cache_gb, speculative=a.spec or None, moe_topk=a.moe_topk)


def chat(a):
    llm = _llm(a)
    from .engine import EOS_ID, SamplingParams

    msgs = []
    print("DeepSeek-V4.1-Flash on DeepInfer-Mac. Empty line = quit, /reset = new conversation.\n")
    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            break
        if not user:
            break
        if user == "/reset":
            msgs = []
            llm.reset()
            continue
        msgs.append({"role": "user", "content": user})
        extra = {"reasoning_effort": a.effort} if a.think and a.effort else {}
        ids, conv = llm.chat_prompt(msgs, thinking=a.think, **extra)
        sp = SamplingParams(temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens)
        out, shown = [], ""
        print("ds> ", end="", flush=True)
        for o in llm.stream(ids, sp):
            if o.token_id == EOS_ID:
                break
            out.append(o.token_id)
            text = llm.tokenizer.decode(out, skip_special_tokens=False)
            if not text.endswith("�"):
                print(text[len(shown):], end="", flush=True)
                shown = text
        print(f"\n   [{json.dumps(llm.last_stats)}]\n")
        reply = shown.split("</think>")[-1].strip()
        msgs.append({"role": "assistant", "content": reply})


def serve(a):
    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(_llm(a)), host=a.host, port=a.port, log_level="info")


def bench(a):
    import numpy as np

    from .engine import SamplingParams

    llm = _llm(a)
    text = ("The history of computing is a story of abstraction layered on abstraction. " * 400)
    ids = llm.encode(text)[: a.prompt_len]
    for rnd in range(a.rounds):
        llm.reset()
        t = time.perf_counter()
        n = sum(1 for _ in llm.stream(ids, SamplingParams(temperature=0, max_tokens=a.gen, ignore_eos=True)))
        print(json.dumps({"round": rnd, **llm.last_stats, "wall_s": round(time.perf_counter() - t, 1)}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="deepinfer_mac")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("chat", "serve", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
        p.add_argument("--cache-gb", type=float, default=None, help="expert cache size (default: free RAM after dense weights - 7 GB)")
        p.add_argument("--spec", action="store_true", help="DSpark speculative decoding")
        p.add_argument("--moe-topk", type=int, default=None, help="experts per token (model: 6); fewer = faster, lossy")
        p.add_argument("--skip-miss", type=float, default=0.0, help="skip a non-cached expert under this routing-weight share (e.g. 0.08); lossy")
        p.add_argument("--prefill-topk", type=int, default=0, help="experts per prompt token outside the last 140 (e.g. 4); faster TTFT, lossy")
        p.add_argument("--lookahead", type=int, default=0, help="prefetch the k most confident next-layer experts (0 = off)")
        if name == "chat":
            p.add_argument("--think", action="store_true")
            p.add_argument("--effort", choices=["high", "max"], default=None, help="reasoning effort in thinking mode (API default: high = 75)")
            p.add_argument("--temperature", type=float, default=0.6)
            p.add_argument("--max-tokens", type=int, default=2048)
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
        if name == "bench":
            p.add_argument("--prompt-len", type=int, default=256)
            p.add_argument("--gen", type=int, default=32)
            p.add_argument("--rounds", type=int, default=2)
    d = sub.add_parser("doctor")
    d.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    c = sub.add_parser("convert")
    c.add_argument("--out", default=str(DEFAULT_MODEL_DIR))
    c.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    if a.cmd == "doctor":
        from pathlib import Path

        from .doctor import main as doctor

        return doctor(Path(a.model).expanduser())
    if a.cmd == "convert":
        from .convert.remote import main as conv

        sys.argv = ["convert", "--out", a.out, "--workers", str(a.workers)]
        return conv()
    {"chat": chat, "serve": serve, "bench": bench}[a.cmd](a)


if __name__ == "__main__":
    main()
