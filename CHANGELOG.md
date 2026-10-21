# Changelog

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
- OpenAI Chat Completions / Responses / Anthropic Messages server on deepseek-recipe.
- Opt-in byte savers for SSD-bound machines: `--skip-miss`, `--prefill-topk`, `--lookahead`.

**Conversion**
- Streams the 510 GB release from the Hub without storing it: range reads, GPU requantization,
  resumable per task. ModelScope (byte-identical shards) first, reached directly even behind a TUN
  proxy; Hugging Face as fallback. Stalled transfers resume from the last byte.
- Weights: dense mxfp8 (bit-exact), experts 2/3/4-bit affine with a clipping search placed by layer
  sensitivity, Engram 4-bit g32.

**Validation**
- Parity against the official PyTorch reference on random configurations covering every mechanism,
  and on the real converted layers (every layer ≥ 0.99997 cosine with activation rounding disabled).
- DSpark drafts identical to the reference; greedy speculative decoding equals greedy decoding.
- Scoring against continuations from the official API (teacher-forced NLL, top-1 agreement, and
  long-context retrieval up to 16K tokens).
