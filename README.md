# DeepInfer-Mac

<p align="left">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <img alt="Platform" src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <a href="https://github.com/ml-explore/mlx"><img alt="MLX" src="https://img.shields.io/badge/backend-MLX-orange"></a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017"><img alt="Model" src="https://img.shields.io/badge/model-DeepSeek--V4.1--Flash-536af5"></a>
  <a href="../../actions/workflows/ci.yml"><img alt="CI" src="../../actions/workflows/ci.yml/badge.svg"></a>
</p>

[English](README.md) | [简体中文](README_zh.md)

DeepInfer-Mac runs **DeepSeek-V4.1-Flash** (754B parameters) on a single Apple Silicon Mac — built and
tested on a **MacBook Air M4 with 24 GB** of unified memory. It re-implements the model-side features
of DeepSeek's [DeepInfer](https://github.com/deepseek-ai/DeepInfer) on [MLX](https://github.com/ml-explore/mlx),
and replaces DeepInfer's multi-GPU expert parallelism with **SSD expert streaming**: the resident dense
weights take 8 GB, and the 6 routed experts each layer picks per token are read from the SSD on demand.

> This is an independent community project. It is not affiliated with or endorsed by DeepSeek.

Supported models:

- DeepSeek-V4.1-Flash (`deepseek-ai/DeepSeek-V4.1-Flash-1017`)

Supported features:

- MLX port of the official reference model, parity-tested against it
- SSD expert streaming (zero-copy reads into MLX slabs, `gather_qmm` over slots)
- DSpark and confidence-scheduled verification
- SWA bounded replay
- Prefix reuse across chat turns
- Structured outputs
- Image input (ViT + aligner, VL routing bias)
- OpenAI / Anthropic / Responses-compatible server on [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)

<details>
<summary>Feature details and configuration</summary>

- **Model:** window + compressed sparse attention with the two-level indexer and cross-layer KV
  sharing, Engram n-gram memory, Hyper-Connections (Sinkhorn), 384-expert MoE with sqrt-softplus
  routing, and the training-time fp8/fp4 activation and cache rounding — ported 1:1 from
  `inference/model.py`. Prefill and decode share one code path that runs any chunk at any position.
- **SSD expert streaming:** experts live in page-aligned records; a miss takes a slot in a
  preallocated per-layout slab and one `preadv` (`F_NOCACHE`) scatters the record straight into
  unified memory. All routed experts of a layer then run as one `gather_qmm` per projection.
  `--cache-gb` sizes the slabs (default: RAM − dense weights − 9.5 GB).
- **DSpark and confidence-scheduled verification:** draft stages, Markov head and confidence head,
  verified by speculative rejection sampling (exact at any temperature). Enable with `--spec` or
  `DEEPINFER_ENABLE_MTP_VARLEN=1`; `DEEPINFER_USE_NEXTN_DECODE` sets the maximum draft length
  (default: 5). On SSD-bound machines it helps only with a warm expert cache.
- **SWA bounded replay:** layers up to the last KV source see the whole prompt; the decoder layers
  replay only a bounded suffix. `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` and
  `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140` are the defaults, as in DeepInfer.
- **Prefix reuse:** a request whose prompt extends the previous one feeds only the new tokens
  (DeepInfer's open-source release ships no KV cache system).
- **Structured outputs:** pass `so_args={"schema_type": "json" | "regex" | "ebnf" | "json_object",
  "schema": ...}` to `SamplingParams` (via [xgrammar](https://github.com/mlc-ai/xgrammar)); see
  [examples/structured_output.py](examples/structured_output.py).
- **Deterministic sampling:** pass a fixed `seq_seed` to `SamplingParams`.

</details>

Supported hardware:

- Apple Silicon Macs (M1 or later), 24 GB of unified memory or more, ~360 GB of free SSD for the conversion

Not included (compared with DeepInfer):

- Multi-device parallelism (Attention DP/CP, MoE EP + EPLB), batch-invariant multi-request batching

## What it looks like

Real output of `python -m deepinfer_mac chat` on the MacBook Air M4 (24 GB), greedy decoding:

```text
you> 用一句话介绍一下杭州。
ds>  杭州是中国浙江省的省会，以西湖美景、深厚的历史文化底蕴和蓬勃发展的数字经济而闻名。
     [decode 1.23 tok/s]
you> What is 17 * 23? Answer with the number only.
ds>  391
you> Write a haiku about the sea.
ds>  Waves crash on the shore,
     Salt and endless blue expanse,
     Moon pulls the tide home.
you> /img carrots.jpeg What is in this image? One sentence.
ds>  A small pile of four fresh orange carrots with green stems rests on a white background.
```

## Getting started

### Prerequisites

- macOS 14+ on Apple Silicon
- Python 3.11+ ([uv](https://github.com/astral-sh/uv) recommended)

### Install

```bash
git clone https://github.com/haoruilee/DeepInfer-Mac && cd DeepInfer-Mac
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
```

### Convert the checkpoint

The converter range-reads every tensor straight from the Hugging Face Hub, converts it on the GPU
and writes it to its final place — the 510 GB release is never stored. It is resumable at any point;
stalled transfers resume from the last byte and alternate between `huggingface.co` and `hf-mirror.com`.

```bash
python -m deepinfer_mac convert --out ~/models/dsv41-mac   # ~340 GB written
python -m deepinfer_mac doctor                             # progress, disk, memory, cache size
```

### Offline inference

```bash
python examples/offline_inference.py
```

```python
from deepinfer_mac import LLM, SamplingParams

llm = LLM()  # ~/models/dsv41-mac, or $DEEPINFER_HF_MODEL_DIR
prompt_ids, _ = llm.chat_prompt([{"role": "user", "content": "Hello!"}])
outputs = llm.generate(prompt_token_ids=[prompt_ids], sampling_params=SamplingParams(temperature=0.6, max_tokens=256))
print(llm.tokenizer.decode([o.token_id for o in outputs[0]]))
```

Each output is a list of per-token `RequestOutput(token_id, token_logprobs, prompt_len, draft_tok, accept_tok)`,
as in DeepInfer.

### Chat and server

```bash
python -m deepinfer_mac chat [--think]      # multi-turn chat; /img photo.jpg <question> for images
python -m deepinfer_mac serve --port 8000   # /v1/chat/completions, /v1/responses, /v1/messages
```

The server renders prompts and parses outputs (thinking, tool calls, JSON) with deepseek-recipe, so
clients see the same API as DeepSeek's. See [examples/openai_client.py](examples/openai_client.py).

## Weight format

| Part | Release | DeepInfer-Mac | Size | Where |
| :--- | :--- | :--- | ---: | :--- |
| Dense: attention, shared experts, gates, mHC, Engram projections | FP8 e4m3, 32×32 E8M0 | MLX **mxfp8**, bit-exact | 7.7 GB | RAM |
| Output head | BF16 | affine 8-bit | 0.7 GB | RAM |
| Embedding | BF16 | BF16 (memory-mapped) | 1.3 GB | SSD |
| Routed experts, 40 × 384 | FP4 e2m1 + E8M0/32 | affine g64 with clipping search: 3-bit in layers 0–3, 8, 14 and the decoder band 20–38; 4-bit in layer 39; 2-bit gate/up + 3-bit down in layers 4–7, 9–13, 15–19 | 213 GB | SSD |
| DSpark draft experts, 3 × 128 | FP4 | 2-bit gate/up, 3-bit down | 4.8 GB | SSD |
| Engram tables, 2 × 384M rows | FP8 + E8M0/32 | 4-bit g32 with clipping search, 160-byte rows | 123 GB | SSD |

Lossless experts (MXFP4, ~296 GB) are a one-line change in `deepinfer_mac/format.py` for machines with more disk.

## Performance benchmarks

MacBook Air M4, 24 GB, internal SSD, macOS 26 — everything else closed:

| | |
| :--- | :--- |
| Decode | **1.25 tok/s** (median of 3 runs; 0.80 s/token, of which 0.55 s is SSD wait) |
| Time to first token: 14-token prompt / 1K / 16K | 10 s / ~49 s / ~8 min |
| Follow-up chat turns | prefill only the new tokens |
| SSD expert streaming | 2.8–3.15 GB/s; ~1.8 GB of expert reads per token at a 49% cache hit rate |
| Resident memory | 7.7 GB dense weights + the expert cache (auto-sized) |

Decode is bounded by SSD bandwidth: 40 layers × 6 experts × 12–20 MB per token, minus cache hits.
We measured the tricks other SSD-streaming projects use, on this model and machine: letting macOS's
page cache act as a second-level expert cache (0.20–0.97 tok/s vs 1.12), splitting each expert read
4-way (1.11 vs 1.12) and next-layer prefetch (−6%) all lose here — 7.7 GB of resident dense weights
and 12–20 MB experts leave no RAM headroom for them. Long-prompt prefill is GPU-compute bound.
Prefill streams the experts once per layer per chunk — even a short prompt routes to most of them.
Reproduce with `python scripts/decode_speed.py --real-engram` and `python -m deepinfer_mac bench`.

## Tuning for your Mac

| Flag | Default | Effect |
| :--- | :--- | :--- |
| `--cache-gb` | auto (free RAM − dense − 4 GB, keeping ≥ 9 GB for macOS) | Expert cache size — the main decode lever: 1.5× capacity lifts the hit rate 0.38 → 0.47 on real routing traces. Close memory-heavy apps first. |
| `--skip-miss 0.08` | off | Skip an uncached expert carrying < 8% of a token's routing weight. Measured: +19% decode speed, top-1 agreement 91.2% → 88.8%. |
| `--skip-miss 0.12` | off | +29% decode speed, agreement 88.8%, NLL +13%. |
| `--prefill-topk 4` | off | Route prompt tokens outside the last 140 to their top-4 experts (lossy; helps only short-to-medium prompts). |
| `--lookahead K` | off | Prefetch the K most confident next-layer experts. Measured 6% slower at K=2 on this SSD-bound machine; may help on faster SSDs. |
| `--moe-topk N` | 6 | Experts per token for every token (lossy). |
| `--spec` | off | DSpark speculative decoding (cumulative-confidence draft ranking, as DeepInfer). A throughput chooser measures speculative vs. plain steps and keeps the faster: on this 24 GB Air speculation loses (verification streams the union of the drafts' experts), so it settles on plain decoding — 1.14 tok/s. |

See [docs/architecture.md](docs/architecture.md) for how the pieces fit together.

## Validation

**Against the official DeepSeek API.** Continuations sampled from DeepSeek's API (`deepseek-flash`,
the V4.1 vectors published by [antirez/ds4](https://github.com/antirez/ds4)), teacher-forced through
the converted model ([`scripts/score_api.py`](scripts/score_api.py)):

| Cases | Top-1 agreement with the API | Notes |
| :--- | :---: | :--- |
| Prose, code, arithmetic, Italian (5) | 90.6% | short continuations; the gap is the 2/3-bit expert compression |
| Needle retrieval at 127 / 128 / 129 / 511 / 512 / 513 / 1023 / 1024 / 1025 / 16383 / 16384 / 16385 tokens (12) | **100%** | answer `074` correct in all 12 — crosses the sliding window, compressed-key selection and the hierarchical candidate-block boundaries |
| **All 17** | **96.7%** | mean NLL 0.127 (API 0.046) |

**Against the official PyTorch reference.** [`tests/`](tests) runs DeepSeek's `inference/model.py` —
with CPU emulations of its TileLang kernels — as the oracle:

- **Model parity** (`test_parity.py`): prefill and 19 decode steps on small random configurations
  covering every mechanism. With activation rounding disabled on both sides, median logit cosine
  ≥ 0.9999; query-blocked indexer identical to the one-shot version.
- **Vision** (`test_vision.py`): ViT + aligner identical in fp32 (cosine 1.0000000); multimodal
  forward ≥ 0.9999.
- **DSpark** (`test_dspark.py`): draft tokens identical to the reference `forward_spec`; greedy
  speculative decoding reproduces plain greedy decoding on both the reject and full-accept paths.
- **Engine, store, server** (`test_engine.py`, `test_store.py`, `test_server.py`): the converted
  on-disk format end to end, slab/LRU invariants and regressions, the HTTP API.
- **Real weights** ([`scripts/real_check.py`](scripts/real_check.py)): converted layers of the actual
  checkpoint against the reference on a real chat prompt — every layer ≥ 0.99997 cosine with activation
  rounding disabled; Engram hash ids identical; converted Engram rows within 0.64% of the originals.

```bash
DSV41_REF=/path/to/DeepSeek-V4.1-Flash-1017/inference pytest tests/
```

## Contributing

See the pinned [Roadmap](../../issues/1), the [good first issues](../../issues?q=is%3Aopen+label%3A%22good+first+issue%22) and [CONTRIBUTING.md](CONTRIBUTING.md). Benchmark results from other Macs are especially welcome.

## License

DeepInfer-Mac is released under the Apache License 2.0. See [LICENSE](LICENSE) for details. The
DeepSeek-V4.1-Flash weights are released by DeepSeek under the MIT License.

## Acknowledgments

DeepInfer-Mac follows the design of DeepSeek's [DeepInfer](https://github.com/deepseek-ai/DeepInfer)
and [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe), and ports the official
DeepSeek-V4.1-Flash reference implementation. It is built on [MLX](https://github.com/ml-explore/mlx)
and [xgrammar](https://github.com/mlc-ai/xgrammar). Ideas and corrections came from
[PipeNetwork/deepseek-v41-mlx](https://github.com/PipeNetwork/deepseek-v41-mlx),
[ssd-moe/deepseek-v4-flash-mlx](https://github.com/ssd-moe/deepseek-v4-flash-mlx),
[antirez/ds4](https://github.com/antirez/ds4), [flash-moe](https://github.com/danveloper/flash-moe)
and the rapid-mlx quantization study. We thank their developers.

## Citation

```bibtex
@misc{deepinfermac2026,
  title = {DeepInfer-Mac: DeepSeek-V4.1-Flash on a single Apple Silicon Mac},
  author = {haoruilee},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/haoruilee/DeepInfer-Mac}},
}
```
