# DeepInfer-Mac

DeepSeek's [DeepInfer](https://github.com/deepseek-ai/DeepInfer) serves **DeepSeek-V4.1-Flash**
(754B parameters: 40 layers × 384 experts, plus 196B of Engram n-gram tables) on NVIDIA SM100 and
Huawei Ascend clusters. This is a re-implementation for **one Apple Silicon Mac**, built and
tested on a **MacBook Air M4 with 24 GB** of unified memory, in Python + [MLX](https://github.com/ml-explore/mlx).

The model is 30× larger than the Mac's memory. It runs because only ~16B parameters are active per
token: dense weights stay resident (8 GB), and the 6 routed experts each layer picks per token are
streamed from the SSD into an LRU cache. Engram rows are fetched from the SSD by hash.

| DeepInfer feature | Here |
|---|---|
| FlashMLA / DeepGEMM / TileLang kernels | MLX `quantized_matmul` (mxfp8, affine 2/3/4-bit), gather-based sparse attention |
| Compressed sparse attention, 2-level indexer, cross-layer KV | ported 1:1 from the official `inference/model.py`, parity-tested |
| **SWA Bounded Replay** (`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`) | same: layers ≤ 20 see the whole prompt, layers 21-39 replay only the last 140 tokens |
| **DSpark speculative decoding** (`DEEPINFER_ENABLE_MTP_VARLEN=1`) | draft stages + Markov/confidence heads, rejection-sampling verify, opt-in (`--spec`) |
| **Structured output** (regex / JSON schema / EBNF) | `SamplingParams(so_args={"schema_type": ..., "schema": ...})` via xgrammar |
| **deepseek-recipe** API alignment | OpenAI Chat Completions, Responses and Anthropic Messages server on deepseek-recipe |
| Expert parallelism / EPLB / DP / CP | n/a (one machine) -- replaced by SSD expert streaming |
| KV cache system (not open-sourced in DeepInfer) | prefix reuse across chat turns |
| Vision | not yet (weights are converted; the ViT is not wired in) |

## Quick start

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install mlx numpy safetensors tokenizers xgrammar deepseek-recipe fastapi uvicorn requests

# 1. Stream-convert the checkpoint straight from the Hub (no 510 GB download sitting on disk).
#    ~323 GB written, resumable; at ~12 MB/s this takes ~12 hours.
python -m deepinfer_mac convert --out ~/models/dsv41-mac

# 2. Use it
python -m deepinfer_mac chat            # multi-turn chat (add --think for thinking mode)
python -m deepinfer_mac serve           # http://127.0.0.1:8000/v1/chat/completions
python -m deepinfer_mac bench
```

```python
from deepinfer_mac import LLM, SamplingParams

llm = LLM()  # ~/models/dsv41-mac, or $DEEPINFER_HF_MODEL_DIR
ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a haiku about SSDs."}])
for out in llm.stream(ids, SamplingParams(temperature=0.6, max_tokens=200)):
    print(out.text, end="", flush=True)
print(llm.last_stats)

# DeepInfer-style batch call; outputs are per-token RequestOutput(token_id, token_logprobs, prompt_len, draft_tok, accept_tok)
outs = llm.generate(["<｜User｜>1+1=<｜Assistant｜></think>"], SamplingParams(temperature=0, max_tokens=8))

# structured output
sp = SamplingParams(so_args={"schema_type": "json", "schema": '{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}'})
```

## How the weights are stored

| Part | Checkpoint | Here | Size |
|---|---|---|---|
| Dense (attention, shared experts, gates, mHC, Engram projections) | FP8 e4m3, 32×32 E8M0 blocks | MLX **mxfp8**, bit-exact repack | 8.2 GB, resident |
| Output head | BF16 | affine 8-bit | 0.7 GB, resident |
| Embedding | BF16 | BF16, memory-mapped | 1.3 GB, on SSD |
| Routed experts (40 × 384) | FP4 e2m1 + E8M0/32 (= MLX mxfp4) | affine: w2 3-bit; w1/w3 3-bit in layers 0-3 and 36-39, 2-bit elsewhere; per-group clipping search (MSE −25% vs plain) | ~202 GB, streamed |
| DSpark draft experts (3 × 128) | FP4 | w1/w3 2-bit, w2 3-bit | 4.8 GB, streamed |
| Engram tables (2 × 384M rows) | FP8 + E8M0/32 | affine 4-bit, 144-byte rows | 110 GB, random reads |

Lossless experts (MXFP4, 296 GB) are a one-line change in `format.expert_layout_for` if your disk
has ~420 GB free; decode is then ~1.5× slower (more bytes per expert).

## Performance on a MacBook Air M4 / 24 GB

SSD reads ~2.8 GB/s; one decode step needs 40 layers × 6 experts × 12.5 MB = 3 GB minus cache hits.

* Decode: ~1-1.5 tok/s (measured numbers: see `python -m deepinfer_mac bench`).
* Time to first token on a fresh prompt: ~1 minute. Even a 100-token prompt routes to ~75% of
  all experts, so prefill is one sweep over most of the expert files. Follow-up chat turns reuse
  the conversation state and pay only for their new tokens.
* `--moe-topk 4` trades quality for a third fewer expert reads.
* `--spec` (DSpark) helps only when the expert cache is warm: verifying k drafts reads the union of
  their experts.

## Validation

`tests/` runs the official PyTorch `inference/model.py` (with CPU emulations of its TileLang
kernels) against this port on small random configurations:

* `test_parity.py` -- prefill + 19 decode steps across every mechanism (window + compressed sparse
  attention, both indexer levels, cross-layer KV sharing, Engram, mHC, fp4 experts). With
  activation fake-quant disabled on both sides, median logit cosine ≥ 0.9999; with it enabled,
  ≥ 0.99 (single fp8 rounding or routing flips on near-ties move individual tokens).
* `test_dspark.py` -- draft tokens identical to the reference's `forward_spec`; greedy speculative
  decoding reproduces plain greedy decoding, on both the reject and the full-accept paths.
* `test_engine.py` / `test_server.py` -- the converted on-disk format end to end (SSD expert store,
  Engram table, prefix reuse, bounded replay, xgrammar), and the HTTP API.

```bash
DSV41_REF=/path/to/DeepSeek-V4.1-Flash/inference pytest tests/
```

## Layout

```
deepinfer_mac/
  convert/remote.py   HTTP-range streaming converter (resumable, no shard ever stored)
  format.py           on-disk layout
  quant.py            fp8/fp4 codecs, bit-exact repacks, clipping-search requantization
  model/model.py      the network (single sequence, any chunk at any position)
  model/ops.py        fake-quant, RoPE/YaRN, Sinkhorn, sparse attention
  model/engram.py     n-gram hashing + SSD table
  store.py            SSD expert streaming (F_NOCACHE preads, thread pool, LRU)
  engine.py           LLM / SamplingParams / RequestOutput, bounded replay, prefix reuse, xgrammar
  dspark.py           DSpark speculative decoding
  server.py           OpenAI / Anthropic / Responses server on deepseek-recipe
```

License: Apache-2.0 for this code (as DeepInfer). Model weights: MIT (DeepSeek).
