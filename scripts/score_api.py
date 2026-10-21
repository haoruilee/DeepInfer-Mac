"""Score the converted model against continuations collected from DeepSeek's official API
(the V4.1-Flash vectors published in antirez/ds4, gguf-tools/quality-testing).

For each case: render the prompt as the API does (deepseek-recipe, thinking disabled), teacher-force
the API's sampled continuation, and report
  * NLL of the continuation under our model vs. the API's own logprobs,
  * top-1 agreement: our argmax == the API's most likely token, per position,
  * for the retrieval cases (expected answer "074"): a short greedy answer.

    python scripts/score_api.py /path/to/ds4/gguf-tools/quality-testing/deepseek-v4.1-flash-20260910 [--max-cases 5] [--check-tokens]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_cases(d: Path):
    cases = []
    for line in (d / "manifest.tsv").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cid = line.split("\t")[0]
        prompt = (d / "prompts" / f"{cid}.txt").read_text(encoding="utf-8")
        resp = json.loads((d / "responses" / f"{cid}.json").read_text())
        cases.append((cid, prompt, resp))
    return cases


def api_tokens(resp, tok):
    """The API's tokens as our ids, via their exact bytes; plus the API's logprobs and top token."""
    ids, lps, tops = [], [], []
    for e in resp["choices"][0]["logprobs"]["content"]:
        ids.append(_byte_id(tok, bytes(e["bytes"]), e["token"]))
        lps.append(e["logprob"])
        top = e["top_logprobs"][0]
        tops.append(_byte_id(tok, bytes(top["bytes"]), top["token"]))
    return ids, lps, tops


def _byte_id(tok, b: bytes, s: str):
    i = tok.token_to_id(s)
    if i is not None:
        return i
    enc = tok.encode(b.decode("utf-8", "replace"), add_special_tokens=False).ids
    return enc[0] if len(enc) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--check-tokens", action="store_true", help="only check the token mapping (no model)")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    cases = load_cases(a.dir)[: a.max_cases]
    from tokenizers import Tokenizer

    from deepinfer_mac.engine import DEFAULT_MODEL_DIR

    root = Path(a.model).expanduser() if a.model else DEFAULT_MODEL_DIR
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    if a.check_tokens:
        for cid, prompt, resp in cases:
            ids, lps, tops = api_tokens(resp, tok)
            bad = sum(i is None for i in ids) + sum(i is None for i in tops)
            text = tok.decode([i for i in ids if i is not None], skip_special_tokens=False)
            same = text == resp["choices"][0]["message"]["content"]
            print(f"{cid}: {len(ids)} tokens, unmapped {bad}, round-trip text identical: {same}")
        return

    import os

    from deepinfer_mac.engine import LLM
    from deepinfer_mac.model.model import MoE

    MoE.SKIP_MISS_BELOW = float(os.environ.get("SKIP_MISS", "0"))
    MoE.PREFILL_TOPK = int(os.environ.get("PREFILL_TOPK", "0"))
    llm = LLM(root)
    rows = []
    for cid, prompt, resp in cases:
        ids, lps, tops = api_tokens(resp, tok)
        if any(i is None for i in ids):
            print(f"{cid}: skipped (unmapped API token)")
            continue
        pids, _ = llm.chat_prompt([{"role": "user", "content": prompt}], thinking=False)
        llm.reset()
        t = time.time()
        st, logits, _, _ = llm._prefill(pids, want_hidden=False)
        first = np.array(logits[-1], dtype=np.float64)
        if os.environ.get("STEPWISE") == "1":  # one token at a time, like decoding (decode-time knobs act)
            rest = np.array([np.array(llm.model.forward(st, [t])[-1], dtype=np.float64) for t in ids[:-1]]).reshape(-1, first.size)
        else:
            rest = np.array(llm.model.forward(st, ids[:-1], all_logits=True), dtype=np.float64) if len(ids) > 1 else np.zeros((0, first.size))
        allg = np.vstack([first[None], rest])
        lse = np.log(np.exp(allg - allg.max(1, keepdims=True)).sum(1)) + allg.max(1)
        ours = -(allg[np.arange(len(ids)), ids] - lse)
        agree = np.mean([int(np.argmax(allg[j])) == tops[j] for j in range(len(ids)) if tops[j] is not None])
        api_nll = -np.mean(lps)
        rows.append((cid, len(pids), len(ids), ours.mean(), api_nll, agree, time.time() - t))
        print(f"{cid}: prompt {len(pids)} tok, cont {len(ids)} tok | NLL ours {ours.mean():.3f} vs API {api_nll:.3f} | "
              f"top-1 agreement {agree:.1%} | {time.time() - t:.0f}s", flush=True)
        if "074" in json.dumps(resp["choices"][0]["message"]["content"]) or int(cid.split("_")[1]) >= 5:
            llm.reset()
            from deepinfer_mac.engine import SamplingParams

            ans = llm.tokenizer.decode([o.token_id for o in llm.stream(pids, SamplingParams(temperature=0, max_tokens=12))], skip_special_tokens=True)
            print(f"    greedy answer: {ans!r}  {'OK' if '074' in ans else 'MISS'}", flush=True)
    if rows:
        r = np.array([x[3:6] for x in rows], dtype=float)
        print(f"\nmean NLL ours {r[:, 0].mean():.3f} vs API {r[:, 1].mean():.3f}; mean top-1 agreement {r[:, 2].mean():.1%} over {len(rows)} cases")


if __name__ == "__main__":
    main()
