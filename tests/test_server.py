"""The OpenAI-compatible server on the tiny checkpoint (protocol plumbing, not text quality)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_parity as T  # noqa: E402
import tiny_ckpt  # noqa: E402

from deepinfer_mac.engine import LLM  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from deepinfer_mac.server import create_app

    rm, _ = T.build()
    root = tiny_ckpt.write(rm, T.tiny_args(), tmp_path_factory.mktemp("tiny"), T.TOKENIZER)
    return TestClient(create_app(LLM(root, expert_cache_gb=0.05)))


def test_chat_completion(client):
    r = client.post("/v1/chat/completions", json={
        "model": "deepseek-flash", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 6, "temperature": 0,
        "thinking": {"type": "disabled"},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    print(json.dumps(body)[:400])
    assert body["object"] == "chat.completion" and body["usage"]["completion_tokens"] >= 1


def test_streaming_and_prefix_cache(client):
    msgs = [{"role": "user", "content": "hello there"}]
    with client.stream("POST", "/v1/chat/completions", json={"model": "deepseek-flash", "messages": msgs, "max_tokens": 5, "stream": True,
                                                             "temperature": 0, "thinking": {"type": "disabled"}}) as r:
        lines = [l for l in r.iter_lines() if l.startswith("data:")]
    assert lines[-1].strip() == "data: [DONE]"
    stats = client.get("/stats").json()
    print(stats)
    assert stats["output_tokens"] >= 1


def test_logprobs(client):
    r = client.post("/v1/chat/completions", json={
        "model": "deepseek-flash", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "temperature": 0.8,
        "logprobs": True, "top_logprobs": 3, "thinking": {"type": "disabled"}, "seed": 1,
    })
    assert r.status_code == 200, r.text
    lp = r.json()["choices"][0]["logprobs"]["content"]
    print(json.dumps(lp[0])[:300])
    assert len(lp) >= 1 and len(lp[0]["top_logprobs"]) == 3 and lp[0]["logprob"] <= 0
