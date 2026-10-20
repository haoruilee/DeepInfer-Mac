"""CPU PyTorch stand-ins for the official TileLang kernels (inference/kernel.py), so the reference
model.py can run on a Mac as the parity oracle. Written from the kernel source, independently of
deepinfer_mac's MLX versions."""

import torch

FP8_MAX, FP4_MAX = 448.0, 6.0
_FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _round_scale(x):
    return torch.exp2(torch.ceil(torch.log2(x)))


def _to_fp4(x):
    """Round to e2m1, ties to even, by nearest-grid search with an even-code tie break."""
    grid = _FP4[:8]
    a = x.abs().unsqueeze(-1)
    d = (a - grid).abs()
    best = d.min(dim=-1, keepdim=True).values
    cand = (d == best)
    code = torch.where(cand & (torch.arange(8) % 2 == 0), torch.arange(8), 99).min(-1).values
    code = torch.where(code == 99, cand.int().argmax(-1), code)
    return torch.where(x < 0, -grid[code], grid[code])


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    shp = x.shape
    xf = x.float().reshape(-1, shp[-1] // block_size, block_size)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _round_scale(amax / FP8_MAX) if scale_fmt is not None else amax / FP8_MAX
    y = (xf / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((y.float() * s).reshape(shp).to(x.dtype))
        return x
    return y.reshape(shp), s.squeeze(-1).reshape(*shp[:-1], -1).to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert inplace
    shp = x.shape
    xf = x.float().reshape(-1, shp[-1] // block_size, block_size)
    amax = xf.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6 * 2**-9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        s = _round_scale(amax.clamp_min(6 * 2**-126) / FP4_MAX)
    y = _to_fp4((xf / s).clamp(-FP4_MAX, FP4_MAX)) * s
    x.copy_(y.reshape(shp).to(x.dtype))
    return x


def _expand(s, rows, cols, br, bc):
    return s.float().repeat_interleave(br, 0)[:rows].repeat_interleave(bc, 1)[:, :cols]


def fp8_gemm(a, a_s, b, b_s, scale_dtype, block_size=32):
    K = a.size(-1)
    af = a.float().reshape(-1, K) * a_s.float().reshape(-1, K // block_size).repeat_interleave(block_size, -1)
    bf = b.float() * _expand(b_s, b.size(0), K, block_size, block_size)
    return (af @ bf.T).reshape(*a.shape[:-1], -1).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype, act_block_size=32):
    K = a.size(-1)
    raw = b.view(torch.uint8)
    w = torch.stack([_FP4[(raw & 0xF).long()], _FP4[(raw >> 4).long()]], -1).flatten(-2)
    w = w * b_s.float().repeat_interleave(32, -1)
    af = a.float().reshape(-1, K) * a_s.float().reshape(-1, K // act_block_size).repeat_interleave(act_block_size, -1)
    return (af @ w.T).reshape(*a.shape[:-1], -1).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    g = torch.stack([kv[i][idx[i].clamp_min(0)] for i in range(b)])  # [b, m, k, d]
    s = torch.einsum("bmhd,bmkd->bmhk", q.float(), g.float()) * softmax_scale
    s = s.masked_fill(~valid[:, :, None, :], float("-inf"))
    mx = s.amax(-1, keepdim=True).clamp_min(-1e30)
    p = torch.exp(s - mx)
    denom = p.sum(-1, keepdim=True) + torch.exp(attn_sink.float()[None, None, :, None] - mx)
    o = torch.einsum("bmhk,bmkd->bmhd", p.bfloat16().float(), g.float()) / denom
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
