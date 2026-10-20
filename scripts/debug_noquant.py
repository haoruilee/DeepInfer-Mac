"""Decode parity with all activation fake-quant disabled on both sides: isolates logic from
fp8/fp4 rounding-flip amplification."""
import sys, numpy as np, torch, mlx.core as mx
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import test_parity as T
from deepinfer_mac.model import model as M
def ref_act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    if inplace: return x
    return x, torch.ones(*x.shape[:-1], x.shape[-1] // block_size)
T.ref.act_quant = ref_act_quant
T.ref.fp4_act_quant = lambda x, *a, **k: x
M.act_quant = lambda x, block=32: x
M.fp4_act_quant = lambda x, block=32, e4m3_scale=False: x
import deepinfer_mac.model.ops as O
def fp8_gemm(a, a_s, b, b_s, scale_dtype, block_size=32):
    K = a.size(-1); bf = b.float() * T.ref_kernels._expand(b_s, b.size(0), K, block_size, block_size)
    return (a.float() @ bf.T).to(torch.get_default_dtype())
def fp4_gemm(a, a_s, b, b_s, scale_dtype, act_block_size=32):
    raw = b.view(torch.uint8); F4 = T.ref_kernels._FP4
    w = torch.stack([F4[(raw & 0xF).long()], F4[(raw >> 4).long()]], -1).flatten(-2) * b_s.float().repeat_interleave(32, -1)
    return (a.float() @ w.T).to(torch.get_default_dtype())
T.ref.fp8_gemm, T.ref.fp4_gemm = fp8_gemm, fp4_gemm
for name, over in (("exact", dict(index_topk=64, candidate_topk_blocks=64)), ("selective", {})):
    rm, mm = T.build(**over)
    toks = np.random.default_rng(0).integers(100, 120000, size=60).tolist()
    _, rl, _ = rm(torch.tensor([toks[:41]]), 0); st = mm.new_state(); ml = mm.forward(st, toks[:41])
    out = [T._cmp(rl.float().numpy(), np.array(ml))[0]]
    for i in range(41, 60):
        _, rl, _ = rm(torch.tensor([[toks[i]]]), i); ml = mm.forward(st, [toks[i]])
        out.append(T._cmp(rl.float().numpy(), np.array(ml))[0])
    print(name, "min cos", min(out), " ".join(f"{c:.5f}" for c in out))
