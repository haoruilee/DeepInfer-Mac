"""Numerics shared by the model: QAT fake-quant, RoPE/YaRN, mHC Sinkhorn, sparse attention.

Each function mirrors a TileLang kernel of the official inference/kernel.py; the docstrings say
which, and tests/test_parity.py checks them against a CPU emulation of those kernels.
"""

import math
from functools import lru_cache

import mlx.core as mx
import numpy as np

FP8_MAX = 448.0
FP4_MAX = 6.0


def _pow2_ceil_log2(x: mx.array) -> mx.array:
    """2**ceil(log2(x)) for positive normal float32 x, by the same bit manipulation as the
    kernels' fast_round_scale."""
    bits = x.astype(mx.float32).view(mx.uint32)
    exp = (bits >> 23) & 0xFF
    exp = exp + ((bits & 0x7FFFFF) != 0).astype(mx.uint32)
    return (exp << 23).view(mx.float32)


def round_e4m3(x: mx.array) -> mx.array:
    """Round float32 to the nearest e4m3 value (inputs are pre-clamped to +-448)."""
    return mx.from_fp8(mx.to_fp8(x), dtype=mx.float32)


def round_e2m1(x: mx.array) -> mx.array:
    """Round to the nearest e2m1 value, ties to even mantissa (inputs pre-clamped to +-6)."""
    a = mx.abs(x)
    q = mx.where(
        a <= 0.25, 0.0,
        mx.where(a < 0.75, 0.5,
        mx.where(a <= 1.25, 1.0,
        mx.where(a < 1.75, 1.5,
        mx.where(a <= 2.5, 2.0,
        mx.where(a < 3.5, 3.0,
        mx.where(a <= 5.0, 4.0, 6.0)))))))
    return mx.where(x < 0, -q, q)


def act_quant(x: mx.array, block: int = 32) -> mx.array:
    return _act_quant_c[block](x)


def _act_quant(x: mx.array, block: int = 32) -> mx.array:
    """kernel.act_quant(..., scale_fmt='ue8m0', inplace=True): fp8 fake-quant per `block` along
    the last dim with power-of-two scales; returns x's dtype."""
    shape, dtype = x.shape, x.dtype
    xf = x.astype(mx.float32).reshape(-1, shape[-1] // block, block)
    amax = mx.maximum(mx.abs(xf).max(axis=-1, keepdims=True), 1e-4)
    s = _pow2_ceil_log2(amax / FP8_MAX)
    y = round_e4m3(mx.clip(xf / s, -FP8_MAX, FP8_MAX)) * s
    return y.reshape(shape).astype(dtype)


def fp4_act_quant(x: mx.array, block: int = 32, e4m3_scale: bool = False) -> mx.array:
    return _fp4_c[(block, e4m3_scale)](x)


def _fp4_act_quant(x: mx.array, block: int = 32, e4m3_scale: bool = False) -> mx.array:
    """kernel.fp4_act_quant(..., inplace=True): fp4 fake-quant. Power-of-two scales (indexer) or,
    with e4m3_scale, scales rounded to e4m3 (compressed KV, block 16)."""
    shape, dtype = x.shape, x.dtype
    xf = x.astype(mx.float32).reshape(-1, shape[-1] // block, block)
    amax = mx.abs(xf).max(axis=-1, keepdims=True)
    if e4m3_scale:
        s = round_e4m3(mx.maximum(amax, FP4_MAX * 2.0**-9) / FP4_MAX)
    else:
        s = _pow2_ceil_log2(mx.maximum(amax, FP4_MAX * 2.0**-126) / FP4_MAX)
    y = round_e2m1(mx.clip(xf / s, -FP4_MAX, FP4_MAX)) * s
    return y.reshape(shape).astype(dtype)


def rms_norm(x: mx.array, w: mx.array, eps: float) -> mx.array:
    """RMSNorm in fp32, weight applied before casting back (matches the reference exactly; the
    model's eps is 1e-20, below what mx.fast.rms_norm guarantees to honour)."""
    return _rms_c(x, w, mx.array(eps, dtype=mx.float32))


@mx.compile
def _rms_c(x, w, eps):
    xf = x.astype(mx.float32)
    y = xf * mx.rsqrt(mx.square(xf).mean(-1, keepdims=True) + eps)
    return (w.astype(mx.float32) * y).astype(x.dtype)


# ------------------------------------------------------------------------------ rope


@lru_cache(8)
def rope_table(dim: int, original_seq_len: int, base: float, factor: float, beta_fast: int, beta_slow: int):
    """Per-pair inverse frequencies with YaRN ramp (precompute_freqs_cis), as numpy float32."""
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rot):
            return dim * math.log(original_seq_len / (rot * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 1e-3), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs.astype(np.float32)


class Rope:
    def __init__(self, dim, original_seq_len, base, factor, beta_fast, beta_slow):
        self.inv = mx.array(rope_table(dim, original_seq_len, base, factor, beta_fast, beta_slow))

    def cos_sin(self, positions: mx.array):
        # angles in float32 like torch.outer(arange, freqs) followed by torch.polar
        ang = positions.astype(mx.float32)[:, None] * self.inv[None, :]
        return mx.cos(ang), mx.sin(ang)

    @staticmethod
    def apply(x: mx.array, cs, inverse: bool = False) -> mx.array:
        """Rotate adjacent pairs of the last dim. x: [L, ..., rd]; cos/sin: [L, rd/2]."""
        cos, sin = cs
        if inverse:
            sin = -sin
        shape = x.shape
        xf = x.astype(mx.float32).reshape(*shape[:-1], -1, 2)
        while cos.ndim < xf.ndim - 1:
            cos, sin = cos[:, None], sin[:, None]
        a, b = xf[..., 0], xf[..., 1]
        y = mx.stack([a * cos - b * sin, a * sin + b * cos], axis=-1)
        return y.reshape(shape).astype(x.dtype)


def rope_tail(x: mx.array, cs, rd: int, inverse: bool = False) -> mx.array:
    """Apply RoPE to the last `rd` dims only (the model's `x[..., -rd:]` in-place rotations)."""
    return mx.concatenate([x[..., :-rd], Rope.apply(x[..., -rd:], cs, inverse)], axis=-1)


# ------------------------------------------------------------------------------ mHC


def hc_split_sinkhorn(mixes: mx.array, scale: mx.array, base: mx.array, hc: int, iters: int, eps: float):
    key = (hc, iters, eps)
    if key not in _sinkhorn_c:
        _sinkhorn_c[key] = mx.compile(lambda m, s, b: _hc_split_sinkhorn(m, s, b, hc, iters, eps))
    return _sinkhorn_c[key](mixes, scale, base)


_sinkhorn_c: dict = {}


def _hc_split_sinkhorn(mixes: mx.array, scale: mx.array, base: mx.array, hc: int, iters: int, eps: float):
    """kernel.hc_split_sinkhorn. mixes: [..., (2+hc)*hc] fp32 -> pre [..., hc], post [..., hc],
    comb [..., hc, hc] (doubly stochastic)."""
    pre = mx.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * mx.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]
    comb = comb.reshape(*comb.shape[:-1], hc, hc)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + eps)
        comb = comb / (comb.sum(-2, keepdims=True) + eps)
    return pre, post, comb


# ------------------------------------------------------------------------------ attention


def sparse_attn(q: mx.array, kv: mx.array, sink: mx.array, idxs: mx.array, scale: float, q_block: int = 128) -> mx.array:
    """kernel.sparse_attn for one sequence. q: [L, H, D]; kv: [N, D] (keys == values, MQA with a
    single latent head); idxs: [L, K] int32 into kv, -1 = empty slot; sink: [H] fp32 extra logit
    that only joins the softmax denominator. A query with no valid slot outputs zeros."""
    L = q.shape[0]
    outs = []
    for a in range(0, L, q_block):
        qi, ii = q[a : a + q_block], idxs[a : a + q_block]
        valid = ii >= 0
        g = kv[mx.maximum(ii, 0)]  # [l, K, D]
        s = (qi.astype(mx.float32) @ g.astype(mx.float32).swapaxes(-1, -2)) * scale  # [l, H, K]
        s = mx.where(valid[:, None, :], s, -mx.inf)
        m = mx.maximum(s.max(-1, keepdims=True), -1e30)
        p = mx.exp(s - m)
        denom = p.sum(-1, keepdims=True) + mx.exp(sink[None, :, None] - m)
        # probabilities go through bf16 before the PV product, as in the kernel
        o = (p.astype(mx.bfloat16).astype(mx.float32) @ g.astype(mx.float32)) / denom
        outs.append(o.astype(q.dtype))
    return mx.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]


_act_quant_c = {b: mx.compile(lambda x, b=b: _act_quant(x, b)) for b in (16, 32)}
_fp4_c = {(b, e): mx.compile(lambda x, b=b, e=e: _fp4_act_quant(x, b, e)) for b in (16, 32) for e in (False, True)}
