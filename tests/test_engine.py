"""End-to-end engine on a tiny on-disk checkpoint: SSD expert streaming, engram table, prefix
reuse, chunked prefill, bounded replay, structured output."""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_parity as T  # noqa: E402  (skips the module when the reference is unavailable)
import tiny_ckpt  # noqa: E402

from deepinfer_mac.engine import LLM, SamplingParams  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    rm, mm = T.build()
    root = tiny_ckpt.write(rm, rm_args(rm), tmp_path_factory.mktemp("tiny"), T.TOKENIZER)
    llm = LLM(root, expert_cache_gb=0.01, bounded_replay=False, prefill_chunk=16)
    return mm, llm


def rm_args(rm):
    return T.tiny_args()


def _cos(a, b):
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return a @ b / np.linalg.norm(a) / np.linalg.norm(b)


def test_disk_matches_memory(setup):
    mm, llm = setup
    toks = np.random.default_rng(3).integers(100, 120000, size=40).tolist()
    a = mm.forward(mm.new_state(), toks)
    st, b, _, _ = llm._prefill(toks, want_hidden=False)
    c = _cos(np.array(a), np.array(b))
    print("disk vs memory (4-bit experts, 4-bit engram, chunked):", c, llm.weights.experts.stats)
    assert c > 0.98


def test_prefix_reuse_and_generate(setup):
    """Reuse must equal building the same state by hand. (It need not equal a one-shot prefill:
    decode and prefill differ in the last ulps, and this tiny random router -- top-2 of 8 --
    has near-ties that such ulps flip.)"""
    _, llm = setup
    llm.reset()
    p = np.random.default_rng(4).integers(100, 120000, size=30).tolist()
    sp = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    out1 = [o.token_id for o in llm.stream(p, sp)]
    assert llm.last_stats["reused_tokens"] == 0
    p2 = p + out1 + [500, 600]
    out2 = [o.token_id for o in llm.stream(p2, sp)]
    assert llm.last_stats["reused_tokens"] == len(p) + len(out1) - 1  # the final token is never fed
    m = llm.model
    st = m.new_state()
    logits = m.forward(st, p)
    for t in out1[:-1]:
        logits = m.forward(st, [t])
    logits = m.forward(st, p2[len(st.tokens):])
    manual = [int(np.argmax(np.array(logits[-1])))]
    print("reuse", out2[:1], "manual", manual, llm.last_stats)
    assert out2[:1] == manual


def test_bounded_replay_runs(setup):
    _, llm = setup
    llm.reset()
    llm.bounded_replay, llm.keep_tokens = True, 12
    p = np.random.default_rng(5).integers(100, 120000, size=60).tolist()
    out = [o.token_id for o in llm.stream(p, SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))]
    llm.bounded_replay = False
    assert len(out) == 4


def test_structured_regex(setup):
    _, llm = setup
    llm.reset()
    sp = SamplingParams(temperature=1.0, max_tokens=12, seed=0, so_args={"schema_type": "regex", "schema": r"[0-9]{3}-[a-z]{2}"})
    text = "".join(o.text for o in llm.stream([0, 1000, 2000], sp))
    print("constrained:", repr(text))
    assert re.fullmatch(r"[0-9]{3}-[a-z]{2}(<｜end▁of▁sentence｜>)?", text)
