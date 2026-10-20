"""Parity of the MLX port against the official PyTorch reference (inference/model.py) on a tiny
random model that exercises every mechanism: window + compressed sparse attention, both indexer
levels, cross-layer KV sharing, engram, mHC and fp4 experts.

    DSV41_REF=~/dl/dsv41-ref/inference DSV41_TOKENIZER=~/models/dsv41-mac/tokenizer.json pytest -s tests/test_parity.py
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import mlx.core as mx  # noqa: E402

HERE = Path(__file__).parent
REF = Path(os.environ.get("DSV41_REF", "~/dl/dsv41-ref/inference")).expanduser()
TOKENIZER = Path(os.environ.get("DSV41_TOKENIZER", "~/models/dsv41-mac/tokenizer.json")).expanduser()
if not (REF / "model.py").exists() or not TOKENIZER.exists():
    pytest.skip("reference code or tokenizer not available", allow_module_level=True)

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import ref_kernels  # noqa: E402

sys.modules["kernel"] = ref_kernels
sys.path.insert(0, str(REF))
import engram as ref_engram  # noqa: E402
import model as ref  # noqa: E402

from deepinfer_mac import quant as Q  # noqa: E402
from deepinfer_mac.model.engram import MemoryEngramTable, NgramHasher, build_compressed_token_map  # noqa: E402
from deepinfer_mac.model.model import Args, ExpertWeights, Transformer  # noqa: E402


class _Tok:
    def __init__(self, path):
        from tokenizers import Tokenizer

        self.backend_tokenizer = Tokenizer.from_file(str(path))

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


def tiny_args(**over):
    base = dict(
        max_batch_size=1, max_seq_len=256, dim=256, moe_inter_dim=128, n_layers=6, n_heads=8,
        n_routed_experts=8, n_activated_experts=2, route_scale=1.5, swiglu_limit=10.0,
        q_lora_rank=128, head_dim=128, rope_head_dim=32, o_groups=4, o_lora_rank=64, window_size=8,
        compress_ratios=(0, 0, 2, 2, 1, 1), kv_source_layers=(2, 4), index_source_layers=(2, 4, 5),
        original_seq_len=64, rope_factor=4, compress_rope_theta=40000.0,
        index_n_heads=4, index_head_dim=64, index_topk=6,
        candidate_source_layer=4, candidate_topk_blocks=2, candidate_block_size=4,
        engram_layer_ids=(1,), engram_vocab_size=1000, engram_n_heads=2, engram_head_dim=64,
        engram_max_ngram_size=4, engram_compressed_vocab_size=99092, vision_n_layers=0, dspark_block_size=0,
    )
    base.update(over)
    layout = ref_engram.EngramLayout.from_args(type("A", (), {**base, "engram_num_embeddings": (0,)})())
    base["engram_num_embeddings"] = tuple(sum(p for per in layer for p in per) for layer in layout.primes)
    return ref.ModelArgs(**base)


def fp8_quantize(w: torch.Tensor, block=32):
    """float [out, in] -> (e4m3 codes, e8m0 block scales) like the checkpoint."""
    o, i = w.shape
    blocks = w.float().reshape(o // block, block, i // block, block)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-8)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448)))
    codes = (blocks / s).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(o, i)
    return codes, s.reshape(o // block, i // block).to(torch.float8_e8m0fnu)


def init_reference(m: "ref.Transformer", seed=0):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    for name, mod in m.named_modules():
        if isinstance(mod, ref.Linear):
            o, i = mod.out_features, mod.in_features
            if mod.weight.dtype == torch.float8_e4m3fn:
                codes, s = fp8_quantize(rn(o, i) / i**0.5)
                mod.weight.data.copy_(codes)
                mod.scale.data.copy_(s)
            elif mod.weight.dtype == torch.float4_e2m1fn_x2:
                mod.weight.data.copy_(torch.randint(0, 256, (o, i // 2), generator=g, dtype=torch.uint8).view(torch.float4_e2m1fn_x2))
                exp = torch.randint(118, 123, (o, i // 32), generator=g).to(torch.float32)
                mod.scale.data.copy_(torch.exp2(exp - 127).to(torch.float8_e8m0fnu))
            else:
                mod.weight.data.copy_((rn(o, i) / i**0.5).to(mod.weight.dtype))
        elif isinstance(mod, ref.RMSNorm):
            mod.weight.data.copy_(1 + 0.1 * rn(mod.dim))
        elif isinstance(mod, ref.ParallelEngramEmbedding):
            w = rn(mod.weight.size(0), mod.dim)
            amax = w.reshape(w.size(0), -1, 32).abs().amax(-1, keepdim=True)
            s = torch.exp2(torch.ceil(torch.log2(amax / 448)))
            mod.weight.data.copy_((w.reshape(w.size(0), -1, 32) / s).to(torch.float8_e4m3fn).reshape_as(w))
            mod.scale.data.copy_(s.squeeze(-1).to(torch.float8_e8m0fnu))
    for name, p in m.named_parameters(recurse=True):
        leaf = name.split(".")[-1]
        if leaf == "attn_sink":
            p.data.copy_(rn(*p.shape))
        elif leaf.startswith("hc_") and leaf.endswith("_fn"):
            p.data.copy_(0.05 * rn(*p.shape))
        elif leaf.startswith("hc_") and leaf.endswith("_base"):
            p.data.copy_(0.5 * rn(*p.shape))
        elif leaf.startswith("hc_") and leaf.endswith("_scale"):
            p.data.copy_(0.5 + torch.rand(*p.shape, generator=g))
        elif name.endswith("gate.weight"):
            p.data.copy_((rn(*p.shape) / p.shape[1] ** 0.5).to(p.dtype))
        elif name.endswith("gate.bias"):
            p.data.copy_(0.1 * rn(*p.shape))
        elif leaf in ("q_weight", "k_weight"):
            p.data.copy_((1 + 0.1 * rn(*p.shape)).to(p.dtype))
        elif name == "embed.weight":
            p.data.copy_(rn(*p.shape).to(p.dtype))
        elif name == "head.weight":
            p.data.copy_(rn(*p.shape) / p.shape[1] ** 0.5)


def _np(t: torch.Tensor) -> np.ndarray:
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2):
        return t.view(torch.uint8).numpy()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np.uint16)
    return t.numpy()


def _mx(t: torch.Tensor) -> mx.array:
    if t.dtype == torch.bfloat16:
        return mx.array(_np(t)).view(mx.bfloat16)
    return mx.array(t.numpy())


class TinyWeights:
    """deepinfer_mac's weight interface over a reference state dict (fp4 experts kept bit-exact
    as mxfp4, fp8 linears as mxfp8, engram dequantized to bf16 -- all lossless)."""

    def __init__(self, sd: dict, args):
        self.sd, self.args = sd, args
        self.globals = {"norm.weight": _mx(sd["norm.weight"]), "head.weight": _mx(sd["head.weight"])}
        self.experts = self
        tables = {}
        for lid in args.engram_layer_ids:
            w, s = sd[f"layers.{lid}.engram.embed.weight"], sd[f"layers.{lid}.engram.embed.scale"]
            tables[lid] = Q.dequant_fp8_rowscale(mx.array(_np(w)), mx.array(_np(s))).astype(mx.bfloat16)
        self.engram = MemoryEngramTable(tables)
        self._embed = _mx(sd["embed.weight"])

    def layer(self, i: int, prefix: str | None = None) -> dict:
        p = prefix or f"layers.{i}."
        out = {}
        for k, t in self.sd.items():
            if not k.startswith(p) or ".experts." in k or ".engram.embed." in k:
                continue
            key = k[len(p) :]
            if key in ("embed.weight", "head.weight"):
                continue  # the draft stages share the backbone's
            if key.endswith(".scale"):
                continue
            if t.dtype == torch.float8_e4m3fn:
                q, s = Q.fp8_block_to_mxfp8(_np(t), _np(self.sd[k[: -len(".weight")] + ".scale"]))
                out[key], out[key[: -len(".weight")] + ".scales"] = mx.array(q), mx.array(s)
            else:
                out[key] = _mx(t)
        return out

    def mtp(self, j: int) -> dict:
        return self.layer(j, prefix=f"mtp.{j}.")

    def get(self, key: str, ids: list[int]):
        li = int(key[1:])
        base = f"mtp.{li}." if key[0] == "M" else f"layers.{li}."
        res = []
        for e in ids:
            parts = []
            for w in ("w1", "w3", "w2"):
                pre = f"{base}ffn.experts.{e}.{w}."
                q, s = Q.fp4_to_mxfp4(_np(self.sd[pre + "weight"]), _np(self.sd[pre + "scale"]))
                parts.append((mx.array(q), mx.array(s), None, 4, "mxfp4"))
            res.append(ExpertWeights(*parts))
        return res

    def embed(self, ids: np.ndarray) -> mx.array:
        return self._embed[mx.array(ids)]


# "logic": every compressed position selected and activation fake-quant switched off on both
# sides, so nothing amplifies ulp-level differences (fp32 matmul order on Accelerate vs Metal):
# checks the wiring tightly. "full": real sparse selection and the real fp8/fp4 activation
# rounding, where a single rounding or routing flip on a near-tie moves one token's logits.
CONFIGS = {
    "logic": (dict(index_topk=64, candidate_topk_blocks=64), True, 0.9999),
    "full": ({}, False, 0.975),  # ulp-level reorderings (batched matmuls) shift which fp8 roundings flip
}


def _no_quant(enable: bool):
    from deepinfer_mac.model import model as M

    saved = (ref.act_quant, ref.fp4_act_quant, ref.fp8_gemm, ref.fp4_gemm, M.act_quant, M.fp4_act_quant)
    if not enable:
        return saved

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        return x if inplace else (x, torch.ones(*x.shape[:-1], x.shape[-1] // block_size))

    def fp8_gemm(a, a_s, b, b_s, scale_dtype, block_size=32):
        w = b.float() * ref_kernels._expand(b_s, b.size(0), a.size(-1), block_size, block_size)
        return (a.float() @ w.T).to(torch.get_default_dtype())

    def fp4_gemm(a, a_s, b, b_s, scale_dtype, act_block_size=32):
        raw = b.view(torch.uint8)
        w = torch.stack([ref_kernels._FP4[(raw & 0xF).long()], ref_kernels._FP4[(raw >> 4).long()]], -1).flatten(-2)
        return (a.float() @ (w * b_s.float().repeat_interleave(32, -1)).T).to(torch.get_default_dtype())

    ref.act_quant, ref.fp4_act_quant, ref.fp8_gemm, ref.fp4_gemm = act_quant, (lambda x, *a, **k: x), fp8_gemm, fp4_gemm
    M.act_quant = lambda x, block=32: x
    M.fp4_act_quant = lambda x, block=32, e4m3_scale=False: x
    return saved


def _restore(saved):
    from deepinfer_mac.model import model as M

    ref.act_quant, ref.fp4_act_quant, ref.fp8_gemm, ref.fp4_gemm, M.act_quant, M.fp4_act_quant = saved


@pytest.fixture(scope="module", params=list(CONFIGS))
def models(request):
    over, noquant, tol = CONFIGS[request.param]
    saved = _no_quant(noquant)
    yield (*build(**over), tol)
    _restore(saved)


def build(**over):
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    args = tiny_args(**over)
    rm = ref.Transformer(args, _Tok(TOKENIZER))
    init_reference(rm)
    rm.temperature = 0
    sd = {k: v for k, v in rm.state_dict().items()}
    tm, cv = build_compressed_token_map(str(TOKENIZER))
    margs = Args.from_dict({**asdict(args), "compress_ratios": tuple(args.compress_ratios)})
    hasher = NgramHasher(margs, tm, cv, args.engram_vocab_size)
    mm = Transformer(margs, TinyWeights(sd, args), hasher)
    return rm, mm


def _cmp(a: np.ndarray, b: np.ndarray):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    cos = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    rel = np.abs(a - b).max() / (np.abs(a).max() + 1e-9)
    return cos, rel, int(a.argmax()) == int(b.argmax())


def test_prefill_and_decode(models):
    rm, mm, tol = models
    rng = np.random.default_rng(0)
    toks = rng.integers(100, 120000, size=60).tolist()
    P = 41  # odd, so ratio-2 groups straddle the prefill/decode boundary
    _, rl, _ = rm(torch.tensor([toks[:P]]), 0)
    st = mm.new_state()
    ml = mm.forward(st, toks[:P])
    cos, rel, top = _cmp(rl.float().numpy(), np.array(ml))
    print(f"prefill: cos={cos:.6f} rel={rel:.4f} top1={top}")
    assert cos > tol and top
    coss = []
    for i in range(P, len(toks)):
        _, rl, _ = rm(torch.tensor([[toks[i]]]), i)
        ml = mm.forward(st, [toks[i]])
        coss.append(_cmp(rl.float().numpy(), np.array(ml))[0])
    print("decode cos:", " ".join(f"{c:.5f}" for c in coss))
    assert np.median(coss) > tol and min(coss) > 0.95


def test_chunked_prefill_matches_full(models):
    """Generic chunks (no reference equivalent): two chunks == one prefill on our side."""
    _, mm, tol = models
    toks = np.random.default_rng(1).integers(100, 120000, size=50).tolist()
    a = mm.forward(mm.new_state(), toks)
    st = mm.new_state()
    mm.forward(st, toks[:23])
    b = mm.forward(st, toks[23:])
    cos, rel, top = _cmp(np.array(a), np.array(b))
    print(f"chunked: cos={cos:.6f} rel={rel:.4f}")
    assert cos > max(tol, 0.98) and top
