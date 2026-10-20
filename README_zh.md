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

DeepInfer-Mac 在一台 Apple Silicon Mac 上运行 **DeepSeek-V4.1-Flash**（7540 亿参数）——在
**24 GB 统一内存的 MacBook Air M4** 上开发与测试。它基于 [MLX](https://github.com/ml-explore/mlx)
复现了 DeepSeek [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 的模型侧能力，并用 **SSD 专家流式加载**
替代 DeepInfer 的多卡专家并行：常驻的稠密权重占 8 GB，每层每个 token 选中的 6 个路由专家按需从 SSD 读取。

> 本项目为独立的社区项目，与 DeepSeek 官方无关，也未获其背书。

支持的模型：

- DeepSeek-V4.1-Flash（`deepseek-ai/DeepSeek-V4.1-Flash-1017`）

支持的特性：

- 官方参考实现的 MLX 移植，并与之逐层对齐测试
- SSD 专家流式加载（零拷贝读入 MLX slab，按槽位 `gather_qmm`）
- DSpark 与基于置信度的验证调度
- SWA Bounded Replay
- 多轮对话前缀复用
- 结构化输出
- 基于 [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) 的 OpenAI / Anthropic / Responses 兼容服务

<details>
<summary>特性细节与配置</summary>

- **模型：** 滑动窗口 + 压缩稀疏注意力（两级 indexer、跨层 KV 共享）、Engram n-gram 记忆、Hyper-Connections（Sinkhorn）、
  384 专家 MoE（sqrt-softplus 路由），以及训练时的 fp8/fp4 激活与缓存舍入——均按 `inference/model.py` 一一移植。
  prefill 与 decode 共用一条代码路径，可在任意位置处理任意长度的 chunk。
- **SSD 专家流式加载：** 专家以页对齐记录存储；未命中时在预分配的 slab 中取一个槽位，一次 `preadv`（`F_NOCACHE`）
  直接写入统一内存。每层所有路由专家以每个投影一次 `gather_qmm` 完成。`--cache-gb` 设置 slab 大小
  （默认：内存 − 稠密权重 − 9.5 GB）。
- **DSpark 与置信度验证调度：** 草稿层、Markov 头与置信度头，通过投机拒绝采样验证（任意温度下均精确）。
  用 `--spec` 或 `DEEPINFER_ENABLE_MTP_VARLEN=1` 开启；`DEEPINFER_USE_NEXTN_DECODE` 设置最大草稿长度（默认 5）。
  在受限于 SSD 带宽的机器上，只有专家缓存较热时才有收益。
- **SWA Bounded Replay：** 最后一个 KV 源层及之前的层处理完整 prompt；之后的 decoder 层只重放有界后缀。
  默认 `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`、`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`，与 DeepInfer 一致。
- **前缀复用：** 若新请求的 prompt 是上一次的延续，只计算新增 token（DeepInfer 开源版不含 KV cache 系统）。
- **结构化输出：** 向 `SamplingParams` 传入 `so_args={"schema_type": "json" | "regex" | "ebnf" | "json_object", "schema": ...}`
  （基于 [xgrammar](https://github.com/mlc-ai/xgrammar)）；见 [examples/structured_output.py](examples/structured_output.py)。
- **确定性采样：** 向 `SamplingParams` 传入固定的 `seq_seed`。

</details>

支持的硬件：

- Apple Silicon Mac（M1 及以上），统一内存 ≥ 24 GB，SSD 空闲约 350 GB

相比 DeepInfer 未包含：

- 多设备并行（Attention DP/CP、MoE EP + EPLB）、多请求 batch 不变性
- 图像输入（视觉塔权重已转换，尚未接入）

## 快速开始

### 环境要求

- Apple Silicon 上的 macOS 14+
- Python 3.11+（推荐 [uv](https://github.com/astral-sh/uv)）

### 安装

```bash
git clone https://github.com/haoruilee/DeepInfer-Mac && cd DeepInfer-Mac
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
```

### 转换权重

转换器直接从 Hugging Face Hub 按区间读取每个张量，在 GPU 上转换后写入最终位置——510 GB 的原始权重从不落盘。
可随时中断续传；卡住的传输从断点续传，并在 `huggingface.co` 与 `hf-mirror.com` 之间轮换。

```bash
python -m deepinfer_mac convert --out ~/models/dsv41-mac   # 写入约 345 GB
```

### 离线推理

```bash
python examples/offline_inference.py
```

```python
from deepinfer_mac import LLM, SamplingParams

llm = LLM()  # ~/models/dsv41-mac，或 $DEEPINFER_HF_MODEL_DIR
prompt_ids, _ = llm.chat_prompt([{"role": "user", "content": "你好！"}])
outputs = llm.generate(prompt_token_ids=[prompt_ids], sampling_params=SamplingParams(temperature=0.6, max_tokens=256))
print(llm.tokenizer.decode([o.token_id for o in outputs[0]]))
```

每个输出是逐 token 的 `RequestOutput(token_id, token_logprobs, prompt_len, draft_tok, accept_tok)` 列表，与 DeepInfer 相同。

### 对话与服务

```bash
python -m deepinfer_mac chat [--think]      # 终端多轮对话
python -m deepinfer_mac serve --port 8000   # /v1/chat/completions、/v1/responses、/v1/messages
```

服务端用 deepseek-recipe 渲染 prompt、解析输出（思考、工具调用、JSON），客户端看到的 API 与 DeepSeek 官方一致。
见 [examples/openai_client.py](examples/openai_client.py)。

## 权重格式

| 部分 | 官方发布 | DeepInfer-Mac | 大小 | 位置 |
| :--- | :--- | :--- | ---: | :--- |
| 稠密权重：注意力、共享专家、门控、mHC、Engram 投影 | FP8 e4m3，32×32 E8M0 | MLX **mxfp8**，比特级无损 | 7.7 GB | 内存 |
| 输出头 | BF16 | affine 8-bit | 0.7 GB | 内存 |
| 词嵌入 | BF16 | BF16（内存映射） | 1.3 GB | SSD |
| 路由专家，40 × 384 | FP4 e2m1 + E8M0/32 | affine g64 + 裁剪搜索：index 源层与靠后层 3-bit，最后一层 4-bit，其余 gate/up 2-bit、down 3-bit | ~210 GB | SSD |
| DSpark 草稿专家，3 × 128 | FP4 | gate/up 2-bit，down 3-bit | 4.8 GB | SSD |
| Engram 表，2 × 3.84 亿行 | FP8 + E8M0/32 | 4-bit g32 + 裁剪搜索，每行 160 字节 | 123 GB | SSD |

磁盘充足时，在 `deepinfer_mac/format.py` 中改一行即可使用无损专家（MXFP4，约 296 GB）。

## 性能

MacBook Air M4，24 GB，内置 SSD：

| | |
| :--- | :--- |
| SSD 专家流式读取 | 2.8–3.15 GB/s；单层 6 个专家约 33 ms |
| Decode（在已转换的前几层上实测，外推到 40 层） | 约 0.7 tok/s |
| 新 prompt 的首 token 延迟 | 约 1 分钟（100 token 的 prompt 已路由到约 75% 的专家） |
| 多轮对话的后续轮次 | 只 prefill 新增 token |

Decode 受 SSD 带宽限制：每个 token 需 40 层 × 6 个专家 × 约 13 MB，减去缓存命中。端到端数据：`python -m deepinfer_mac bench`。

## 验证

[`tests/`](tests) 以 DeepSeek 官方 PyTorch `inference/model.py`（其 TileLang kernel 用 CPU 模拟）作为基准：

- **模型对齐**（`test_parity.py`）：在覆盖全部机制的小规模随机配置上测试 prefill 与 19 步 decode；两侧关闭激活舍入时，logits 余弦相似度中位数 ≥ 0.9999。
- **DSpark**（`test_dspark.py`）：草稿 token 与参考 `forward_spec` 完全一致；贪心投机解码在拒绝与全接受两条路径上都与普通贪心解码一致。
- **引擎、存储、服务**（`test_engine.py`、`test_store.py`、`test_server.py`）：端到端的磁盘格式、slab/LRU 不变量、HTTP API。
- **真实权重**（[`scripts/real_check.py`](scripts/real_check.py)）：用真实 checkpoint 已转换的前几层在真实对话 prompt 上与参考实现逐层比对——关闭激活舍入时每层余弦 ≥ 0.99997；Engram 哈希 id 完全一致。

```bash
DSV41_REF=/path/to/DeepSeek-V4.1-Flash-1017/inference pytest tests/
```

## 许可证

DeepInfer-Mac 以 Apache License 2.0 发布，详见 [LICENSE](LICENSE)。DeepSeek-V4.1-Flash 权重由 DeepSeek 以 MIT 许可证发布。

## 致谢

DeepInfer-Mac 参照 DeepSeek 的 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 与
[deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) 设计，并移植了官方 DeepSeek-V4.1-Flash 参考实现；
基于 [MLX](https://github.com/ml-explore/mlx) 与 [xgrammar](https://github.com/mlc-ai/xgrammar) 构建。
[PipeNetwork/deepseek-v41-mlx](https://github.com/PipeNetwork/deepseek-v41-mlx)、
[ssd-moe/deepseek-v4-flash-mlx](https://github.com/ssd-moe/deepseek-v4-flash-mlx)、
[antirez/ds4](https://github.com/antirez/ds4)、[flash-moe](https://github.com/danveloper/flash-moe)
以及 rapid-mlx 的量化研究为本项目提供了思路与修正。感谢各位开发者。

## 引用

```bibtex
@misc{deepinfermac2026,
  title = {DeepInfer-Mac: DeepSeek-V4.1-Flash on a single Apple Silicon Mac},
  author = {haoruilee},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/haoruilee/DeepInfer-Mac}},
}
```
