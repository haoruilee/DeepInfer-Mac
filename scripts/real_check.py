"""Real-weight check on the first N converted layers, before the full conversion finishes.

Runs layers 0..N-1 of DeepSeek-V4.1-Flash with the *real* weights through both the official
PyTorch reference (inference/model.py, CPU kernel emulations) and this MLX port, on a real chat
prompt, and compares every layer's output -- prefill and a few decode steps.

Both sides use the same converted experts (the reference gets them dequantized, with the fp8
activation rounding of its fp4 path) and the same original fp8 Engram rows (range-read from the Hub,
since the Engram tables convert last). Dense weights: the reference gets the original fp8 values
recovered bit-exactly from the mxfp8 files.

    python scripts/real_check.py [N_LAYERS]
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path("~/models/dsv41-mac").expanduser()
REF = Path("~/dl/dsv41-ref/inference").expanduser()
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "tests"), str(REF)]

import torch  # noqa: E402

import ref_kernels  # noqa: E402

sys.modules["kernel"] = ref_kernels
import model as ref  # noqa: E402

from deepinfer_mac import quant as Q  # noqa: E402
from deepinfer_mac.convert import hf  # noqa: E402
from deepinfer_mac.model.engram import NgramHasher  # noqa: E402
from deepinfer_mac.model.model import Args, Transformer  # noqa: E402
from deepinfer_mac.store import ExpertStore  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6
NOQUANT = "--noquant" in sys.argv  # switch off activation fake-quant on both sides: logic-only check
cfg = json.loads((ROOT / "inference_config.json").read_text())
INDEX = json.loads((HERE.parent / "deepinfer_mac/convert/tensor_index.json").read_text())


# ------------------------------------------------------------------------------ shared sources


class RemoteEngram:
    """Original fp8 Engram rows, range-read from the Hub and cached; dequantized to bf16."""

    def __init__(self):
        self.rows: dict[tuple[int, int], np.ndarray] = {}
        self.pool = ThreadPoolExecutor(16)

    def _fetch(self, layer, r):
        w, s = INDEX[f"layers.{layer}.engram.embed.weight"], INDEX[f"layers.{layer}.engram.embed.scale"]
        a = hf.get_range(w["shard"], w["start"] + r * 256, w["start"] + (r + 1) * 256)
        b = hf.get_range(s["shard"], s["start"] + r * 8, s["start"] + (r + 1) * 8)
        deq = Q.dequant_fp8_rowscale(mx.array(np.frombuffer(a, np.uint8)[None]), mx.array(np.frombuffer(b, np.uint8)[None]))
        return np.array(deq.astype(mx.float32))[0]

    def ensure(self, layer, ids: np.ndarray):
        need = sorted({int(i) for i in ids.reshape(-1)} - {r for (l, r) in self.rows if l == layer})
        for r, v in zip(need, self.pool.map(lambda r: self._fetch(layer, r), need)):
            self.rows[(layer, r)] = v

    def get(self, layer, ids: np.ndarray) -> np.ndarray:
        self.ensure(layer, ids)
        return np.stack([self.rows[(layer, int(i))] for i in ids.reshape(-1)]).reshape(*ids.shape, 256)

    def lookup(self, layer_id, ids):  # deepinfer_mac's table interface
        return mx.array(self.get(layer_id, ids)).astype(mx.bfloat16)


ENGRAM = RemoteEngram()
STORE = ExpertStore(ROOT, budget_bytes=8 << 30)
EMBED = np.memmap(ROOT / "embed.bf16", dtype=np.uint16, mode="r").reshape(-1, cfg["dim"])


def dequant_expert(layer: int, e: int) -> list[torch.Tensor]:
    ew = STORE.get(f"L{layer:02d}", [e])[0]
    out = []
    for q, s, b, bits, mode in (ew.w1, ew.w2, ew.w3):
        w = mx.dequantize(q, s.view(mx.bfloat16), b.view(mx.bfloat16), group_size=64, bits=bits).astype(mx.float32)
        out.append(torch.from_numpy(np.array(w)).bfloat16())
    return out


# ------------------------------------------------------------------------------ reference


def build_reference():
    torch.set_default_dtype(torch.bfloat16)
    names = {f.name for f in fields(ref.ModelArgs)}
    a = ref.ModelArgs(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items() if k in names})
    a.n_layers, a.dspark_block_size, a.vision_n_layers, a.max_seq_len, a.max_batch_size = N, 0, 0, 4096, 1
    a.engram_layer_ids = tuple(l for l in a.engram_layer_ids if l < N)

    class Tok:
        def __init__(self):
            from tokenizers import Tokenizer

            self.backend_tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer.json"))

        def __len__(self):
            return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

    # experts: allocate nothing up front; each expert pulls its converted weights on first use
    def lazy_forward(self, x, weights=None):
        if not hasattr(self, "_w"):
            self._w = dequant_expert(*self._key)
        w1, w2, w3 = self._w
        xq = x if NOQUANT else ref_kernels.act_quant(x.clone(), 32, "ue8m0", inplace=True)  # the fp4 path's fp8 input
        gate, up = torch.nn.functional.linear(xq, w1).float(), torch.nn.functional.linear(xq, w3).float()
        up, gate = up.clamp(-self.swiglu_limit, self.swiglu_limit), gate.clamp(max=self.swiglu_limit)
        h = torch.nn.functional.silu(gate) * up
        if weights is not None:
            h = weights * h
        h = h.to(x.dtype) if NOQUANT else ref_kernels.act_quant(h.to(x.dtype), 32, "ue8m0", inplace=True)
        return torch.nn.functional.linear(h, w2)

    class LazyExperts(torch.nn.Module):
        def __init__(self, layer, n, limit):
            super().__init__()
            self.layer, self.n, self.limit, self.cache = layer, n, limit, {}

        def __getitem__(self, i):
            if i not in self.cache:
                m = torch.nn.Module()
                m._key, m.swiglu_limit = (self.layer, i), self.limit
                m.forward = lazy_forward.__get__(m)
                self.cache[i] = m
            return self.cache[i]

    orig_moe_init = ref.MoE.__init__

    def moe_init(self, layer_id, args):
        n = args.n_routed_experts
        args.n_routed_experts = 0  # build without routed experts...
        try:
            orig_moe_init(self, layer_id, args)
        finally:
            args.n_routed_experts = n
        self.n_routed_experts = self.n_local_experts = self.experts_end_idx = n
        self.experts = LazyExperts(layer_id, n, args.swiglu_limit)  # ...then attach the lazy ones
        self.gate = ref.Gate(layer_id, args)

    ref.MoE.__init__ = moe_init

    def engram_embed_forward(self, indices):
        return torch.from_numpy(ENGRAM.get(self._layer, indices.numpy())).bfloat16()

    orig_ee = ref.ParallelEngramEmbedding.__init__

    def ee_init(self, num_embeddings, dim):
        torch.nn.Module.__init__(self)
        self.num_embeddings, self.dim = num_embeddings, dim

    ref.ParallelEngramEmbedding.__init__ = ee_init
    ref.ParallelEngramEmbedding.forward = engram_embed_forward

    with torch.device("cpu"):
        m = ref.Transformer(a, Tok())
    ref.MoE.__init__, ref.ParallelEngramEmbedding.__init__ = orig_moe_init, orig_ee
    for i, blk in enumerate(m.layers):
        if blk.engram is not None:
            blk.engram.embed._layer = i
        load_reference_layer(m, i)
    m.embed.weight.data = torch.from_numpy(np.array(EMBED).view(np.int16)).view(torch.bfloat16)
    m.temperature = 0
    return m, a


def load_reference_layer(m, i):
    """Dense tensors of layer i, from our mxfp8 files back to the checkpoint's fp8 (bit-exact)."""
    d = mx.load(str(ROOT / "dense" / f"L{i:02d}.safetensors"))
    sd = m.layers[i].state_dict()
    for name, t in sd.items():
        if ".experts." in name or "engram.embed" in name:
            continue
        if name.endswith(".scale"):
            base = name[: -len(".scale")]
            q, s = np.array(d[base + ".weight"]), np.array(d[base + ".scales"])
            codes = q.view(np.uint8).reshape(q.shape[0], -1)
            block = s[::32]  # the 32x32 block scale, repeated over rows by the converter
            w = torch.from_numpy(codes.copy()).view(torch.float8_e4m3fn)
            if t.shape != tuple(block.shape):  # wo_a: dense bf16 in the reference
                continue
            m.layers[i].get_parameter(base + ".weight").data = w
            m.layers[i].get_parameter(name).data = torch.from_numpy(block.copy()).view(torch.float8_e8m0fnu)
        elif name.endswith(".weight") and name[: -len(".weight")] + ".scale" in sd:
            continue
        elif name.endswith("wo_a.weight"):
            deq = mx.dequantize(d["attn.wo_a.weight"], d["attn.wo_a.scales"], None, group_size=32, bits=8, mode="mxfp8")
            m.layers[i].get_parameter(name).data = torch.from_numpy(np.array(deq.astype(mx.float32))).bfloat16()
        else:
            a = d[name]
            v = torch.from_numpy(np.array(a.astype(mx.float32)))
            m.layers[i].get_parameter(name).data = v.to(t.dtype)


# ------------------------------------------------------------------------------ ours


def build_ours(n_layers):
    a = Args.from_dict(cfg)
    a.n_layers = n_layers
    a.engram_layer_ids = tuple(l for l in a.engram_layer_ids if l < n_layers)

    class W:
        globals = mx.load(str(ROOT / "dense" / "globals.safetensors"))
        experts = STORE
        engram = ENGRAM

        def layer(self, i):
            return mx.load(str(ROOT / "dense" / f"L{i:02d}.safetensors"))

        def embed(self, ids):
            return mx.array(np.ascontiguousarray(EMBED[ids])).view(mx.bfloat16)

    return Transformer(a, W(), NgramHasher.load(a, ROOT, cfg["engram_vocab_size"]))


# ------------------------------------------------------------------------------ compare


def main():
    from deepinfer_mac.engine import LLM

    ids, _ = LLM.chat_prompt(type("X", (), {"root": ROOT})(), [{"role": "user", "content": "用一句话介绍一下杭州。Then say it in English."}])
    toks = list(ids)
    print(f"prompt: {len(toks)} tokens, first ids {toks[:8]}")
    if NOQUANT:
        import test_parity

        test_parity._no_quant(True)
    t = time.time()
    rm, ra = build_reference()
    print(f"reference built in {time.time() - t:.0f}s")
    ours = build_ours(N)

    # engram hash ids must agree exactly
    st = ours.new_state()
    h_ours = ours.ngram(st, toks)
    h_ref = rm.engram_hash(torch.tensor([toks]), 0)[0].numpy()
    print("engram hash ids identical:", np.array_equal(h_ours, h_ref), "max id", h_ours.max(), "< rows", cfg["engram_num_embeddings"])

    cap_ref, cap_ours = {}, {}
    for i, b in enumerate(rm.layers):
        b.register_forward_hook(lambda mod, inp, out, i=i: cap_ref.__setitem__(i, out[0].detach().float().numpy()))
    from deepinfer_mac.model import model as M

    orig = M.Block.__call__

    def wrapped(self, *args, **kw):
        out = orig(self, *args, **kw)
        cap_ours[self.layer_id] = np.array(out[0].astype(mx.float32))
        return out

    M.Block.__call__ = wrapped

    def compare(tag):
        worst = 1.0
        row = []
        for i in range(N):
            a, b = cap_ref[i].reshape(-1), cap_ours[i].reshape(-1)
            cos = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
            worst = min(worst, cos)
            row.append(f"L{i}={cos:.5f}")
        print(f"{tag}: " + " ".join(row), f"| |h| ref {np.abs(cap_ref[N-1]).mean():.3f} ours {np.abs(cap_ours[N-1]).mean():.3f}")
        return worst

    with torch.inference_mode():
        P = len(toks) - 3
        t = time.time()
        rm(torch.tensor([toks[:P]]), 0)
        print(f"reference prefill {time.time() - t:.0f}s")
        st = ours.new_state()
        t = time.time()
        ours.forward(st, toks[:P])
        print(f"ours prefill {time.time() - t:.1f}s  (expert hit rate {STORE.hit_rate():.2f})")
        worst = compare(f"prefill {P} tok")
        for i in range(P, len(toks)):
            rm(torch.tensor([[toks[i]]]), i)
            ours.forward(st, [toks[i]])
            worst = min(worst, compare(f"decode @{i}"))
    finite = all(np.isfinite(v).all() for v in cap_ours.values())
    print(f"\nRESULT: worst layer cosine {worst:.5f}; finite={finite}; experts read {STORE.stats['bytes']/1e9:.1f} GB")


if __name__ == "__main__":
    main()
