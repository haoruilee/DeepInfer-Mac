# Changelog

## Unreleased

- **Indexer 2x faster in long prefill, bit-identical.** The indexer's queries and keys are fp4 values
  with power-of-two scales, so a bf16 matmul (fp32 accumulation, one rounding) equals the reference's
  fp32 einsum + cast; the ReLU / head-weight / head-sum chain is one Metal pass instead of three over
  the [rows, heads, keys] tensor (16x). Checked on 436M real scores at 16-32K: zero differences.
- **Deterministic Top-K** (#10): higher score first, earlier position on ties, in both indexer levels.
- **DoSA study** (#8): exact screening in the style of DoSA (arXiv 2610.13990) measured on real
  V4.1 indexer activations (`scripts/capture_indexer.py`, `scripts/dosa_study.py`). At 16-32K keys a
  sound certificate costs 83-160% of plain scoring (Top-K 512 needs ~20 of 32 heads exact before
  anything can be rejected; the layers after 20 are already restricted to 2048 candidate blocks),
  so it is not enabled. Worth revisiting at >=128K keys.

## v0.1.1

Alignment with DeepInfer and deepseek-recipe, and fixes found by reviewing them and peer projects.

- **JSON mode** on the server: `json_object` (no schema) now compiles; in thinking mode the grammar
  starts after `</think>` (DeepInfer's `trigger_token_ids`), so reasoning is no longer forced into JSON.
- **Thinking-mode multi-turn chat** reuses the conversation: a snapshot one token before each prompt's
  end (as DeepInfer stores its cache) survives the API template dropping earlier reasoning. Restoring it
  is bit-exact.
- **Images on the server** go through deepseek-recipe's resolver (the API's preprocessing); image URLs
  may only reach public addresses, and the engine receives bytes, never paths.
- **Sampling** as DeepInfer: top-p before top-k, logprobs of the processed distribution,
  `logprobs`/`top_logprobs` in the server; BOS, pad and image specials are never generated.
- **DSpark**: drafts ranked by the running product of confidences; a measured throughput chooser falls
  back to plain decoding when speculation loses (here: 1.14 vs. 0.35 tok/s before).
- Bounded replay: replayed decoder queries see the window only from the replay start, as in DeepInfer.
- Measured and rejected on this machine: OS page cache as a second-level expert cache, split reads,
  larger prefill chunks (all documented in the README).
- Prefill expert compute as one quantized matmul per expert (1.67x).


## v0.1.0

First public release: DeepSeek-V4.1-Flash (`deepseek-ai/DeepSeek-V4.1-Flash-1017`) on a single
Apple Silicon Mac, developed and validated on a MacBook Air M4 with 24 GB.

**Model**
- MLX port of the official reference implementation: window + compressed sparse attention with the
  two-level indexer and cross-layer KV sharing, Engram, Hyper-Connections, 384-expert MoE, DSpark
  draft stages, and the training-time fp8/fp4 activation and cache rounding.
- Prefill and decode share one path that runs any chunk at any position (chunked prefill, prefix
  reuse, bounded replay, speculative verification). Indexer scoring is query-blocked, so long
  prompts stay within memory.

**Runtime**
- SSD expert streaming: page-aligned records, `preadv` with `F_NOCACHE` straight into preallocated
  MLX slabs (zero copy), one `gather_qmm` per projection, LRU over slots; resident experts compute
  while misses load. The cache sizes itself from free memory.
- SWA bounded replay (140-token suffix, DeepInfer's default), prefix reuse across chat turns.
- DSpark speculative decoding with confidence-scheduled verification (opt-in).
- Structured outputs (regex / JSON schema / EBNF via xgrammar).
- Image input: MLX ViT + aligner with the official preprocessing, VL routing bias, image tokens
  excluded from Engram; in the Python API, the chat CLI (`/img`) and the server.
- OpenAI Chat Completions / Responses / Anthropic Messages server on deepseek-recipe.
- Opt-in byte savers for SSD-bound machines: `--skip-miss`, `--prefill-topk`, `--lookahead`.

**Conversion**
- Streams the 510 GB release from the Hub without storing it: range reads, GPU requantization,
  resumable per task. ModelScope (byte-identical shards) first, reached directly even behind a TUN
  proxy; Hugging Face as fallback. Stalled transfers resume from the last byte.
- Weights: dense mxfp8 (bit-exact); experts 3-bit in 25 layers (0-3, 8, 14, 20-38), 4-bit in layer 39,
  2/3-bit elsewhere, affine with a clipping search; Engram 4-bit g32. ~340 GB in total.

**Measured on a MacBook Air M4, 24 GB**
- Decode 1.25 tok/s; first token 10 s (short prompt) / ~49 s (1K) / ~8 min (16K).
- Prefill expert compute as one quantized matmul per expert (1.67x over per-pair gather);
  long-prompt prefill is now bound by the GPU's dense throughput (MXFP8 projections run within
  ~15% of a bf16 GEMM of the same shape).

**Validation**
- Parity against the official PyTorch reference on random configurations covering every mechanism,
  and on the real converted layers (every layer ≥ 0.99997 cosine with activation rounding disabled).
- DSpark drafts identical to the reference; greedy speculative decoding equals greedy decoding.
- Against continuations sampled from the official API: 96.7% top-1 agreement over 17 cases; the
  long-context retrieval answer is correct at every boundary from 127 to 16,385 tokens.
- Vision tower identical to the reference in fp32; converted Engram rows within 0.64% of the originals.
