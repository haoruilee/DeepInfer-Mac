# DeepInfer-Mac — plan

Goal: run **DeepSeek-V4.1-Flash-1017** (the model DeepInfer serves) on a single
**MacBook Air M4 / 24 GB / 1 TB**, re-implementing DeepInfer's model-side features
(SWA Bounded Replay, DSpark speculative decoding, structured output, deepseek-recipe
API alignment) on MLX + Metal.

## Hard facts (measured 2026-10-20)

| | |
|---|---|
| Checkpoint | 510 GB, 48 shards. Routed experts 296 GB (FP4 e2m1 + E8M0/32), Engram 203 GB (FP8), dense 7 GB (FP8 32x32), embed/head 2.6 GB, vision 0.8 GB, DSpark/MTP 0.7 GB |
| Model | 40 layers, d=5120, 384 routed + 1 shared expert (top-6, inter 2304), MLA-style 1-KV-head attn (head 512), SWA 128, compressed sparse attention + 2-level indexer, cross-layer KV (layer 20 feeds 21-39 = the "encoder/decoder" split), mHC x4, Engram on layers 1/14 |
| Mac | 24 GB unified RAM, ~16 GB usable; SSD 396 GB free, **2.8 GB/s** uncached reads (8 threads, 11-19 MB chunks), ~20K IOPS random small reads |
| Network | **~13.7 MB/s** to HF (parallelism does not help) -> ~10.5 h download |

No existing project runs V4.1-Flash below a 256 GB Mac. The only way on 24 GB is
**SSD expert streaming**: keep dense weights resident (~5-8 GB), stream the 6 routed
experts/layer/token from SSD into an LRU cache, mmap Engram rows on demand.

## Key design decisions

1. **Remote-streaming converter.** The 510 GB checkpoint cannot land on a 396 GB disk.
   Using the safetensors headers (already fetched), the converter range-GETs each tensor
   straight from HF, repacks it, and appends to the MLX-native layout. No raw shards ever
   stored; resumable per tensor via a manifest.
2. **Experts are already MXFP4.** FP4 e2m1 + one E8M0 scale per 32 along K == MLX
   `mode="mxfp4"` (group 32). Repack is bit-exact. Lossy 2/3-bit is optional for disk/speed.
3. **Dense FP8 (32x32 block, E8M0) -> MLX mxfp8** (e4m3 + E8M0 per 32 along K, block scale
   broadcast over rows): bit-exact. Optional 4-bit dense to free RAM.
4. **Expert file layout:** one file per layer, each expert's w1|w3|w2 (+scales) contiguous,
   16 KB aligned -> one `pread` per expert, `F_NOCACHE` so the OS page cache does not
   double-buffer. Thread pool of 8 readers, LRU cache sized from free RAM.
5. **Engram on SSD:** tables mmap'd; hash ids are a deterministic function of tokens, so rows
   are prefetched on a thread before layer 1 runs (DeepSeek's own design intent).
6. **Prefill is expert-major:** per layer, each needed expert loaded once and applied to all
   its tokens (chunks of 4-16K tokens) so long prompts cost one sweep of the experts.
7. **SWA Bounded Replay (DeepInfer's YOCO prefill truncation):** layers 0-20 run on the full
   prompt; layers 21-39 (which only read layer 20's compressed KV + their own 128-token
   window) replay only the last `keep_tokens=140`. Same default as DeepInfer.
8. **Prefix cache** across turns (KV is ~890 B/token, so a whole conversation stays in RAM):
   only new tokens are prefilled each chat turn.
9. **DSpark speculative decoding** (3 resident draft layers, 128 experts, block 5) with the
   confidence head deciding how many drafts to verify; verification is one target forward on
   <=6 tokens, so the experts of all drafted tokens are fetched in one parallel batch.

## Expected performance (estimates, to be replaced by measurements)

| Expert precision | Disk total | Per-expert | Decode | Notes |
|---|---|---|---|---|
| MXFP4 (lossless) | ~410 GB | 18.8 MB | ~0.8-1.2 tok/s | does not fit 396 GB free |
| mixed 2/3-bit | ~300 GB | ~12.5 MB | ~1.5-2.5 tok/s | lossy |

Time-to-first-token on a fresh long prompt ~30-60 s (one sweep of the experts); follow-up
chat turns only pay for the new tokens.

## Package layout

```
deepinfer_mac/
  config.py            ModelArgs from config.json
  quant.py             e2m1/e4m3/e8m0 codecs, repack to MLX mxfp4/mxfp8, requant
  model/
    layers.py          RMSNorm, YaRN RoPE, quantized Linear, mHC sinkhorn
    attention.py       window ring, Compressor, Indexer, candidate blocks, sparse attn
    moe.py             Gate (sqrtsoftplus, bias/bias_vl), streamed experts
    engram.py          n-gram hashing + SSD table with prefetch
    dspark.py          draft blocks, markov + confidence heads
    transformer.py     full model, bounded replay, expert-major prefill
  store/expert_store.py   pread/F_NOCACHE pool + LRU + prefetch
  convert/remote.py       HTTP-range streaming converter (resumable)
  engine/              LLM / SamplingParams / RequestOutput (DeepInfer-shaped API),
                       sampler, prefix cache, spec decode loop, structured output (xgrammar)
  server/openai.py     OpenAI-compatible server via deepseek-recipe
examples/  tests/  scripts/
```

## Phases

0. **Env** — uv venv (py3.12), mlx, torch (CPU, for the reference), safetensors, tokenizers,
   xgrammar, deepseek-recipe, fastapi. Private repo `haoruilee/DeepInfer-Mac`.
1. **Converter first** (download is the long pole): codecs + bit-exactness tests on real
   tensors fetched by range; then launch the full streaming conversion in the background.
2. **MLX port** of `inference/model.py`; parity test against the official PyTorch reference
   on a tiny random config (CPU emulations of the TileLang kernels: act_quant, fp4/fp8 gemm,
   sparse_attn, hc_split_sinkhorn).
3. **Expert store** + streamed MoE; RAM budget manager.
4. **Engine**: sampling, prefix cache, bounded replay, chunked expert-major prefill,
   DeepInfer-shaped `LLM.generate`.
5. **DSpark** speculative decoding.
6. **Structured output** (regex / JSON schema / EBNF via xgrammar) + **OpenAI server**
   with deepseek-recipe for prompt rendering and output parsing (thinking, tool calls).
7. **Real-weight validation** as the download lands: per-layer parity against the torch
   reference on real weights (few tokens, CPU), coherent generation, benchmarks.
8. Stretch: vision tower, mlx.distributed across multiple Macs (DeepInfer's EP analogue).

Not ported (no meaning on one Mac): multi-GPU EP/DP/CP, EPLB, batch-invariant kernels
across GPUs (single-sequence decode here is deterministic anyway with a seed).
