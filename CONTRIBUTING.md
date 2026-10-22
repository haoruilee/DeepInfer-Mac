# Contributing

Thanks for helping! Start with the pinned [Roadmap](../../issues/1) and the issues labeled
[`good first issue`](../../issues?q=is%3Aopen+label%3A%22good+first+issue%22). Comment on an issue to claim it.

> 欢迎贡献！可以先看置顶的 Roadmap 和带 `good first issue` 标签的任务，在 issue 下留言认领即可。

## Setup

```bash
git clone https://github.com/haoruilee/DeepInfer-Mac && cd DeepInfer-Mac
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[test]"
```

You do **not** need the converted 340 GB model to work on most of the code: the test suite runs on
tiny random models.

## Tests

The tests use DeepSeek's official PyTorch implementation as the oracle. Fetch it (and the tokenizer)
once:

```bash
mkdir -p ref/inference ref/tok
B=https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main
for f in model.py kernel.py engram.py vision.py image_processor.py; do curl -sfL $B/inference/$f -o ref/inference/$f; done
curl -sfL $B/tokenizer.json -o ref/tok/tokenizer.json
DSV41_REF=ref/inference DSV41_TOKENIZER=ref/tok/tokenizer.json pytest -q tests/
python -m pyflakes deepinfer_mac scripts
```

If you have the converted model, `scripts/real_check.py 6 --noquant` compares real layers against the
reference and `scripts/score_api.py` scores against the official API vectors.

## Principles

- **Parity first.** Model changes must keep `tests/test_parity.py` passing; when in doubt, compare
  against the reference with activation rounding disabled (`--noquant`) to separate logic from noise.
- **Measure before claiming a speedup.** Report the numbers (`scripts/decode_speed.py`, repeated and
  alternated runs — single runs vary by ±15%) and the machine. Several well-known tricks turned out
  slower on our hardware; negative results are welcome too.
- **Quality changes need a score.** Anything lossy needs `scripts/score_api.py` (use `STEPWISE=1` for
  decode-time changes) before/after.
- Match the surrounding code style; keep comments about *why*.
