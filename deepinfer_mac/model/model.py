"""DeepSeek-V4.1-Flash in MLX, for one sequence at a time.

A port of the official inference/model.py. The reference only knows two shapes of call --
a prefill from position 0 and single-token decode -- whereas this one runs any chunk of tokens
at any start position (chunked prefill, prefix reuse across chat turns, DeepInfer's SWA bounded
replay, speculative verification). All per-sequence state lives in `SeqState`, so the weights
are shared and stateless.

Routed experts never live in this module: `Weights.experts.get(layer_key, ids)` hands them over
(from the SSD stream in production, from memory in tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from . import ops
from .ops import Rope, act_quant, fp4_act_quant, rms_norm, rope_tail


@dataclass
class Args:
    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_mtp_layers: int = 3
    n_heads: int = 64
    n_routed_experts: int = 384
    n_activated_experts: int = 6
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    kv_source_layers: tuple = (2, 8, 14, 20)
    index_source_layers: tuple = (2, 8, 14, 20, 24, 28, 32, 36)
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: tuple = (1, 14)
    engram_max_ngram_size: int = 4
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    compress_rope_theta: float = 160000.0
    compress_ratios: tuple = ()
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: tuple = (37, 38, 39)
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_n_activated_experts: int = 3

    @classmethod
    def from_dict(cls, d: dict) -> "Args":
        names = cls.__dataclass_fields__
        kw = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in names}
        return cls(**kw)

    def moe(self, layer_id: int) -> tuple[int, int]:
        if layer_id < self.n_layers:
            return self.n_routed_experts, self.n_activated_experts
        return self.dspark_n_routed_experts, self.dspark_n_activated_experts


# ------------------------------------------------------------------------------ linear layers


class Linear:
    """bf16/fp32 weight, MLX mxfp8 (bit-exact from the fp8 checkpoint) or affine-quantized.
    fp8-quantized weights get their activations fp8-fake-quantized first, as the reference's
    fp8_gemm path does."""

    def __init__(self, d: dict, name: str, dtype=None):
        self.w = d[name + ".weight"]
        self.scales = d.get(name + ".scales")
        self.biases = d.get(name + ".biases")
        self.mode = "affine" if self.biases is not None else "mxfp8"
        if self.scales is None and dtype is not None:
            self.w = self.w.astype(dtype)
        if self.scales is not None:
            self.bits = 32 * self.w.shape[1] // (self.scales.shape[1] * (32 if self.mode == "mxfp8" else 64)) if self.mode == "affine" else 8
            self.group = 32 if self.mode == "mxfp8" else 64

    def __call__(self, x: mx.array) -> mx.array:
        if self.scales is None:
            return x.astype(self.w.dtype) @ self.w.T
        if self.mode == "mxfp8":
            x = act_quant(x)
        return mx.quantized_matmul(
            x, self.w, self.scales, self.biases, transpose=True, group_size=self.group, bits=self.bits, mode=self.mode
        )

    def rows(self, a: int, b: int) -> "Linear":
        """The output rows [a, b) as their own Linear (for the grouped wo_a)."""
        out = object.__new__(Linear)
        out.__dict__.update(self.__dict__)
        out.w = self.w[a:b]
        out.scales = None if self.scales is None else self.scales[a:b]
        out.biases = None if self.biases is None else self.biases[a:b]
        return out


class GroupedLinear:
    """wo_a: block-diagonal over groups (each group projects only its own heads). One batched
    gather_qmm over the groups instead of a matmul per group."""

    def __init__(self, lin: Linear, groups: int):
        self.lin, self.g = lin, groups
        rows = lin.w.shape[0] // groups
        if lin.scales is not None:
            self.w = lin.w.reshape(groups, rows, -1)
            self.scales = lin.scales.reshape(groups, rows, -1)
            self.biases = None if lin.biases is None else lin.biases.reshape(groups, rows, -1)
            self.idx = mx.arange(groups, dtype=mx.uint32)
        else:
            self.w = lin.w.reshape(groups, rows, -1)

    def __call__(self, o: mx.array) -> mx.array:
        """o: [L, groups, in] -> [L, groups * rows]"""
        L = o.shape[0]
        x = o.swapaxes(0, 1)[:, None]  # [g, 1, L, in]
        if self.lin.scales is None:
            y = x.astype(self.w.dtype) @ self.w.swapaxes(-1, -2)[:, None]
        else:
            y = mx.gather_qmm(x, self.w, self.scales, self.biases, lhs_indices=self.idx, rhs_indices=self.idx,
                              transpose=True, group_size=self.lin.group, bits=self.lin.bits, mode=self.lin.mode)
        return y.reshape(self.g, L, -1).swapaxes(0, 1).reshape(L, -1)


@dataclass
class ExpertWeights:
    """One routed expert: per projection (packed, scales, biases-or-None, bits, mode)."""

    w1: tuple
    w3: tuple
    w2: tuple


def _qmm(x, p):
    q, s, b, bits, mode = p
    group = 32 if mode.startswith("mx") else 64
    return mx.quantized_matmul(x, q, s, b, transpose=True, group_size=group, bits=bits, mode=mode)


def swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    return _swiglu_c(gate, up, mx.array(limit, dtype=mx.float32))


@mx.compile
def _swiglu_c(gate, up, limit):
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    up = mx.where(limit > 0, mx.clip(up, -limit, limit), up)
    gate = mx.where(limit > 0, mx.minimum(gate, limit), gate)
    return gate * mx.sigmoid(gate) * up


def run_expert(xq: mx.array, e: ExpertWeights, weights: mx.array | None, limit: float) -> mx.array:
    """xq: the fp8-fake-quantized input rows (quantized once per MoE call)."""
    key = tuple(p[3:] for p in (e.w1, e.w3, e.w2)) + (e.w1[2] is None,)
    f = _expert_fns.get(key)
    if f is None:
        f = _expert_fns[key] = mx.compile(lambda xq, w, lim, a1, b1, c1, a3, b3, c3, a2, b2, c2: _expert(
            xq, w, lim, (a1, b1, c1, *key[0]), (a3, b3, c3, *key[1]), (a2, b2, c2, *key[2])))
    w = weights if weights is not None else mx.ones((xq.shape[0], 1), dtype=mx.float32)
    return f(xq, w, mx.array(limit, dtype=mx.float32), *e.w1[:3], *e.w3[:3], *e.w2[:3])


_expert_fns: dict = {}


def _expert(xq, w, limit, p1, p3, p2):
    h = _swiglu_c(_qmm(xq, p1), _qmm(xq, p3), limit)
    return _qmm(act_quant(((w * h).astype(xq.dtype))), p2)


# ------------------------------------------------------------------------------ caches


class Ring:
    """A layer's sliding-window KV: `window` slots plus the position each one holds (-1 empty)."""

    def __init__(self, window: int, dim: int):
        self.kv = mx.zeros((window, dim), dtype=mx.bfloat16)
        self.pos = np.full(window, -1, dtype=np.int64)

    def copy(self) -> "Ring":
        r = object.__new__(Ring)
        r.kv, r.pos = self.kv, self.pos.copy()
        return r


class Growable:
    """Append-only [n, dim] buffer, grown in steps (like mlx-lm's KVCache)."""

    STEP = 1024

    def __init__(self, dim: int, dtype=mx.bfloat16):
        self.buf = mx.zeros((0, dim), dtype=dtype)
        self.n = 0

    def append(self, x: mx.array):
        need = self.n + x.shape[0]
        if need > self.buf.shape[0]:
            grow = max(self.STEP, need - self.buf.shape[0])
            self.buf = mx.concatenate([self.buf, mx.zeros((grow, self.buf.shape[1]), dtype=self.buf.dtype)])
        self.buf[self.n : need] = x.astype(self.buf.dtype)
        self.n = need

    def view(self, n: int | None = None) -> mx.array:
        return self.buf[: self.n if n is None else n]

    def truncate(self, n: int):
        self.n = min(self.n, n)


@dataclass
class CompressorState:
    ratio: int
    kv: Growable  # RoPE'd, fp4-fake-quantized compressed latents
    index_k: Growable | None  # indexer keys derived from the same latents
    tail_kv: mx.array | None = None  # incomplete group carried over (fp32, pre-pool)
    tail_score: mx.array | None = None


@dataclass
class SeqState:
    """Everything one sequence carries between forward calls."""

    rings: list[Ring]
    comp: dict[int, CompressorState]
    tokens: list[int] = field(default_factory=list)
    compressed_ids: list[int] = field(default_factory=list)  # engram token history
    pos: int = 0  # tokens fed so far
    mtp_rings: list[Ring] = field(default_factory=list)

    def snapshot(self) -> dict:
        """Cheap rollback point for speculative decoding: arrays are immutable values."""
        return {
            "rings": [r.copy() for r in self.rings],
            "mtp_rings": [r.copy() for r in self.mtp_rings],
            "comp": {k: (c.kv.n, None if c.index_k is None else c.index_k.n, c.tail_kv, c.tail_score) for k, c in self.comp.items()},
            "tokens": len(self.tokens),
            "compressed": len(self.compressed_ids),
            "pos": self.pos,
        }

    def restore(self, snap: dict):
        self.rings = snap["rings"]
        self.mtp_rings = snap["mtp_rings"]
        for k, (n, nk, tk, ts) in snap["comp"].items():
            c = self.comp[k]
            c.kv.truncate(n)
            if c.index_k is not None:
                c.index_k.truncate(nk)
            c.tail_kv, c.tail_score = tk, ts
        del self.tokens[snap["tokens"] :]
        del self.compressed_ids[snap["compressed"] :]
        self.pos = snap["pos"]


class Shared:
    """What attention layers hand down the stack within one forward (SharedAttentionRuntime)."""

    def __init__(self):
        self.comp: CompressorState | None = None
        self.index_k: Growable | None = None
        self.topk: mx.array | None = None
        self.candidates: mx.array | None = None


# ------------------------------------------------------------------------------ attention


class Attention:
    def __init__(self, a: Args, layer_id: int, d: dict, is_backbone: bool = True):
        self.a, self.layer_id = a, layer_id
        self.ratio = a.compress_ratios[layer_id]
        self.is_kv_source = is_backbone and layer_id in a.kv_source_layers
        self.is_index_source = is_backbone and layer_id in a.index_source_layers
        self.owns_index_k = self.is_index_source and layer_id in a.kv_source_layers
        self.is_candidate_source = layer_id == a.candidate_source_layer
        self.uses_candidates = 0 <= a.candidate_source_layer < layer_id
        self.sink = d["attn.attn_sink"].astype(mx.float32)
        self.wq_a = Linear(d, "attn.wq_a")
        self.q_norm = d["attn.q_norm.weight"]
        self.wq_b = Linear(d, "attn.wq_b")
        self.wkv = Linear(d, "attn.wkv")
        self.kv_norm = d["attn.kv_norm.weight"]
        self.wo_a = GroupedLinear(Linear(d, "attn.wo_a"), a.o_groups)
        self.wo_b = Linear(d, "attn.wo_b")
        self.scale = a.head_dim**-0.5
        if self.ratio:
            self.rope = Rope(a.rope_head_dim, a.original_seq_len, a.compress_rope_theta, a.rope_factor, a.beta_fast, a.beta_slow)
        else:
            self.rope = Rope(a.rope_head_dim, 0, a.rope_theta, a.rope_factor, a.beta_fast, a.beta_slow)
        if self.is_kv_source:
            self.c_wkv = Linear(d, "attn.compressor.wkv", dtype=mx.float32 if self.ratio > 1 else None)
            self.c_wgate = Linear(d, "attn.compressor.wgate", dtype=mx.float32) if self.ratio > 1 else None
            self.c_norm = d["attn.compressor.norm.weight"]
        if self.is_index_source:
            self.i_wq_b = Linear(d, "attn.indexer.wq_b")
            self.i_weights_proj = Linear(d, "attn.indexer.weights_proj")
            if self.owns_index_k:
                self.i_wk = Linear(d, "attn.indexer.wk")
                self.i_k_norm = d["attn.indexer.k_norm.weight"]

    # -- compressor

    def _compress(self, x: mx.array, c: CompressorState) -> mx.array | None:
        """New complete groups' pooled latents (pre-RoPE, normed, bf16), or None."""
        eps = self.a.norm_eps
        if self.ratio == 1:
            return rms_norm(self.c_wkv(x), self.c_norm, eps)
        xf = x.astype(mx.float32)
        kv, score = self.c_wkv(xf), self.c_wgate(xf)
        if c.tail_kv is not None:
            kv, score = mx.concatenate([c.tail_kv, kv]), mx.concatenate([c.tail_score, score])
        n = kv.shape[0]
        full = n // self.ratio * self.ratio
        c.tail_kv = kv[full:] if full < n else None
        c.tail_score = score[full:] if full < n else None
        if full == 0:
            return None
        kv = kv[:full].reshape(-1, self.ratio, kv.shape[-1])
        score = score[:full].reshape(-1, self.ratio, score.shape[-1])
        pooled = (kv * mx.softmax(score, axis=1, precise=True)).sum(axis=1)
        return rms_norm(pooled.astype(x.dtype), self.c_norm, eps)

    # -- indexer

    def _indexer(self, x, qr, latent, first_group, positions, sh: Shared, c: CompressorState, offset: int, end: int):
        a, r, rd = self.a, self.ratio, self.a.rope_head_dim
        if self.owns_index_k:
            if latent is not None:
                gpos = mx.arange(first_group, first_group + latent.shape[0]) * r
                k = rms_norm(self.i_wk(latent), self.i_k_norm, a.norm_eps)
                k = fp4_act_quant(rope_tail(k, self.rope.cos_sin(gpos), rd))
                c.index_k.append(k)
            sh.index_k = c.index_k
        L = x.shape[0]
        q = self.i_wq_b(qr).reshape(L, a.index_n_heads, a.index_head_dim)
        q = fp4_act_quant(rope_tail(q, self.rope.cos_sin(positions), rd))
        visible = (positions + 1) // r  # compressed positions each query may see
        T = end // r
        index_k = sh.index_k.view(T)
        # bf16 roundings at the same points as the reference's torch ops, so near-ties in the
        # top-k resolve the same way
        bf = mx.bfloat16
        w = (self.i_weights_proj(x).astype(mx.float32) * (a.index_head_dim**-0.5 * a.index_n_heads**-0.5)).astype(bf)
        score = mx.einsum("lhd,td->lht", q.astype(mx.float32), index_k.astype(mx.float32)).astype(bf)
        score = (mx.maximum(score, 0) * w[..., None]).astype(bf)
        score = score.astype(mx.float32).sum(axis=1).astype(bf).astype(mx.float32)  # [L, T]
        reach = mx.arange(T)[None, :] < visible[:, None]
        score = mx.where(reach, score, -mx.inf)
        if self.is_candidate_source:
            sh.candidates = select_candidate_blocks(score, visible, a.candidate_topk_blocks, a.candidate_block_size)
        elif self.uses_candidates:
            score = mx.where(sh.candidates, score, -mx.inf)
        k = min(a.index_topk, T)
        idx = mx.argpartition(-score, kth=k - 1, axis=-1)[:, :k] if k < T else mx.broadcast_to(mx.arange(T), (L, T))
        idx = mx.sort(idx.astype(mx.int32), axis=-1)
        ok = mx.take_along_axis(score, idx, axis=-1) > -mx.inf
        return mx.where(ok & (idx < visible[:, None]), idx + offset, -1).astype(mx.int32)

    # -- forward

    def __call__(self, x: mx.array, positions: mx.array, st: SeqState, ring: Ring, sh: Shared, start: int) -> mx.array:
        a, rd = self.a, self.a.rope_head_dim
        L = x.shape[0]
        cs = self.rope.cos_sin(positions)
        qr = rms_norm(self.wq_a(x), self.q_norm, a.norm_eps)
        q = rope_tail(self.wq_b(qr).reshape(L, a.n_heads, a.head_dim), cs, rd)
        kv = act_quant(rope_tail(rms_norm(self.wkv(x), self.kv_norm, a.norm_eps), cs, rd))

        # sliding window over [ring (older positions) | this chunk]
        win = a.window_size
        pos_np = np.arange(start, start + L)
        wpos = pos_np[:, None] - win + 1 + np.arange(win)[None, :]
        slot = wpos % win
        in_ring = (wpos >= 0) & (wpos < start) & (ring.pos[slot] == wpos)
        widx = np.where(wpos >= start, win + wpos - start, np.where(in_ring, slot, -1))
        keys = [ring.kv, kv]
        idxs = [mx.array(widx.astype(np.int32))]

        if self.ratio:
            offset = win + L
            c = st.comp[self.layer_id] if self.is_kv_source else sh.comp
            first_group = c.kv.n
            latent = self._compress(x, c) if self.is_kv_source else None
            if self.is_kv_source:
                sh.comp = c
            if self.is_index_source:
                if (start + L) // self.ratio == 0:
                    sh.topk = mx.full((L, 0), -1, dtype=mx.int32)
                else:
                    sh.topk = self._indexer(x, qr, latent, first_group, positions, sh, c, offset, start + L)
            if latent is not None:
                gpos = mx.arange(first_group, first_group + latent.shape[0]) * self.ratio
                latent = fp4_act_quant(rope_tail(latent, self.rope.cos_sin(gpos), rd), block=16, e4m3_scale=True)
                c.kv.append(latent)
            keys.append(sh.comp.kv.view())
            idxs.append(sh.topk)

        o = ops.sparse_attn(q, mx.concatenate(keys), self.sink, mx.concatenate(idxs, axis=-1), self.scale)
        o = rope_tail(o, cs, rd, inverse=True)

        # update the ring with this chunk's newest `win` tokens. Built functionally (no in-place
        # writes) so ring snapshots taken for speculative rollback stay valid.
        keep = min(L, win)
        src = np.full(win, -1, dtype=np.int32)
        src[pos_np[-keep:] % win] = np.arange(L - keep, L)
        has = mx.array(src >= 0)
        ring.kv = mx.where(has[:, None], kv[mx.array(np.maximum(src, 0))], ring.kv)
        ring.pos = ring.pos.copy()
        ring.pos[pos_np[-keep:] % win] = pos_np[-keep:]

        o = act_quant(o.reshape(L, a.o_groups, -1))
        return self.wo_b(self.wo_a(o))


def select_candidate_blocks(logits: mx.array, visible: mx.array, topk_blocks: int, block: int) -> mx.array:
    """Level one of the two-level top-k: a [L, T] bool mask of the best `topk_blocks` blocks."""
    L, T = logits.shape
    pad = -T % block
    if pad:
        logits = mx.concatenate([logits, mx.full((L, pad), -mx.inf)], axis=-1)
    scores = logits.reshape(L, -1, block).max(axis=-1)
    nb = scores.shape[-1]
    last = (visible - 1) // block
    scores = mx.where(mx.arange(nb)[None, :] == last[:, None], mx.inf, scores)
    k = min(topk_blocks, nb)
    if k < nb:
        top = mx.argpartition(-scores, kth=k - 1, axis=-1)[:, :k]
        keep = mx.zeros((L, nb), dtype=mx.bool_)
        keep = mx.put_along_axis(keep, top, mx.take_along_axis(scores, top, axis=-1) > -mx.inf, axis=-1)
    else:
        keep = scores > -mx.inf
    return mx.repeat(keep, block, axis=-1)[:, :T]


# ------------------------------------------------------------------------------ MoE


class MoE:
    EXPERT_BATCH = 32

    def __init__(self, a: Args, layer_id: int, d: dict, experts, key: str):
        self.a, self.key, self.experts = a, key, experts
        self.n_routed, self.topk = a.moe(layer_id)
        self.gate_w = d["ffn.gate.weight"].astype(mx.float32)
        self.bias = d["ffn.gate.bias"].astype(mx.float32)
        self.bias_vl = d.get("ffn.gate.bias_vl")
        self.shared = [Linear(d, f"ffn.shared_experts.{w}") for w in ("w1", "w2", "w3")]

    def route(self, x: mx.array, image_mask: mx.array | None = None):
        scores = mx.sqrt(mx.logaddexp(x.astype(mx.float32) @ self.gate_w.T, 0.0))  # sqrt(softplus)
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = mx.where(image_mask[:, None], self.bias_vl.astype(mx.float32), bias)
        idx = mx.argpartition(-(scores + bias), kth=self.topk - 1, axis=-1)[:, : self.topk]
        w = mx.take_along_axis(scores, idx, axis=-1)
        if self.topk > 1:
            w = w / (w.sum(-1, keepdims=True) + 1e-20)
        return w * self.a.route_scale, idx

    def __call__(self, x: mx.array, image_mask=None) -> mx.array:
        w, idx = self.route(x, image_mask)
        idx_np = np.array(idx)
        w1, w2, w3 = self.shared
        xq = act_quant(x)
        y = w2(swiglu(w1(x), w3(x), self.a.swiglu_limit).astype(x.dtype)).astype(mx.float32)
        ids = np.unique(idx_np).tolist()
        # expert-major: each needed expert is fetched once for all of its tokens. Batches bound the
        # working set during long prefills (a chunk can touch every expert of the layer) and let
        # the next batch's reads overlap this batch's compute.
        B = self.EXPERT_BATCH
        batches = [ids[i : i + B] for i in range(0, len(ids), B)]
        prefetch = getattr(self.experts, "prefetch", None)
        for bi, batch in enumerate(batches):
            if prefetch is not None and bi + 1 < len(batches):
                prefetch(self.key, batches[bi + 1])
            for e, ew in zip(batch, self.experts.get(self.key, batch)):
                rows, col = np.nonzero(idx_np == e)
                r = mx.array(rows.astype(np.int32))
                out = run_expert(xq[r], ew, w[r, mx.array(col.astype(np.int32))][:, None], self.a.swiglu_limit)
                y = y.at[r].add(out.astype(mx.float32))
            if len(batches) > 1:
                mx.eval(y)
        return y.astype(x.dtype)


# ------------------------------------------------------------------------------ engram


class Engram:
    def __init__(self, a: Args, layer_id: int, d: dict, table):
        self.a, self.layer_id, self.table = a, layer_id, table
        self.wkv = Linear(d, "engram.wkv")
        self.qk = (d["engram.q_weight"].astype(mx.float32) * d["engram.k_weight"].astype(mx.float32))

    def __call__(self, h: mx.array, hash_ids: np.ndarray, token_mask: mx.array | None) -> mx.array:
        a = self.a
        rows = self.table.lookup(self.layer_id, hash_ids)  # [L, n_cols, head_dim] bf16
        kv = self.wkv(rows.reshape(rows.shape[0], -1))
        key = kv[:, : a.hc_mult * a.dim].astype(mx.float32).reshape(-1, a.hc_mult, a.dim)
        value = kv[:, a.hc_mult * a.dim :].astype(mx.float32)
        hf = h.astype(mx.float32)
        eps = a.norm_eps
        rstd = mx.rsqrt(mx.square(hf).mean(-1) + eps) * mx.rsqrt(mx.square(key).mean(-1) + eps)
        dot = (hf * self.qk * key).sum(-1) * rstd * a.dim**-0.5
        gate = mx.sigmoid(mx.sign(dot) * mx.sqrt(mx.maximum(mx.abs(dot), 1e-6)))
        if token_mask is not None:
            gate = mx.where(token_mask[:, None], gate, 0.0)
        return (hf + gate[..., None] * value[:, None, :]).astype(h.dtype)


# ------------------------------------------------------------------------------ blocks


class Block:
    def __init__(self, a: Args, layer_id: int, d: dict, experts, engram_table=None, key=None, is_backbone=True):
        self.a, self.layer_id = a, layer_id
        self.attn = Attention(a, layer_id, d, is_backbone)
        self.ffn = MoE(a, layer_id, d, experts, key or f"L{layer_id:02d}")
        self.engram = Engram(a, layer_id, d, engram_table) if "engram.wkv.weight" in d else None
        self.attn_norm, self.ffn_norm = d["attn_norm.weight"], d["ffn_norm.weight"]
        self.hc = {k: d[f"hc_{k}"].astype(mx.float32) for k in ("attn_fn", "ffn_fn", "attn_base", "ffn_base", "attn_scale", "ffn_scale")}

    def hc_mixes(self, x, kind):
        a = self.a
        xf = x.reshape(x.shape[0], -1).astype(mx.float32)
        rs = mx.rsqrt(mx.square(xf).mean(-1, keepdims=True) + a.norm_eps)
        mixes = (xf @ self.hc[f"{kind}_fn"].T) * rs
        return ops.hc_split_sinkhorn(mixes, self.hc[f"{kind}_scale"], self.hc[f"{kind}_base"], a.hc_mult, a.hc_sinkhorn_iters, a.hc_eps)

    @staticmethod
    def hc_pre(x, pre):
        return _hc_pre_c(x, pre)

    @staticmethod
    def hc_post(x, residual, post, comb):
        return _hc_post_c(x, residual, post, comb)

    @staticmethod
    def _hc_post(x, residual, post, comb):
        # fp32 products summed over the residual streams in order, like the reference's torch.sum
        prod = comb[..., None] * residual.astype(mx.float32)[:, :, None, :]  # [L, j, k, d]
        acc = prod[:, 0]
        for j in range(1, prod.shape[1]):
            acc = acc + prod[:, j]
        y = post[..., None] * x.astype(mx.float32)[:, None, :] + acc
        return y.astype(x.dtype)

    def __call__(self, x, positions, pre_mix, st: SeqState, ring: Ring, sh: Shared, image_mask=None, start: int = 0):
        eps = self.a.norm_eps
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, "attn")
        h = rms_norm(self.hc_pre(x, pre_mix), self.attn_norm, eps)
        h = self.attn(h, positions, st, ring, sh, start)
        x = self.hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, "ffn")
        h = rms_norm(self.hc_pre(x, attn_pre), self.ffn_norm, eps)
        h = self.ffn(h, image_mask)
        x = self.hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre


@mx.compile
def _hc_pre_c(x, pre):
    return (pre[..., None] * x.astype(mx.float32)).sum(axis=1).astype(x.dtype)


_hc_post_c = mx.compile(Block._hc_post)


# ------------------------------------------------------------------------------ the model


def materialize(*objs) -> None:
    """Evaluate every mx.array reachable through the attributes of `objs`. mx.load is lazy and
    pins arrays to the loading thread's stream; materializing makes them plain data any thread
    can use, and front-loads the reads."""
    seen, arrays = set(), []

    def walk(o, depth=0):
        if depth > 6 or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, mx.array):
            arrays.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, depth + 1)
        elif hasattr(o, "__dict__") and not isinstance(o, type):
            for v in vars(o).values():
                walk(v, depth + 1)

    for o in objs:
        walk(o)
    for i in range(0, len(arrays), 64):
        mx.eval(arrays[i : i + 64])


class Transformer:
    def __init__(self, a: Args, weights, ngram=None):
        self.a, self.w, self.ngram = a, weights, ngram
        self.layers = [
            Block(a, i, weights.layer(i), weights.experts, weights.engram) for i in range(a.n_layers)
        ]
        g = weights.globals
        self.norm = g["norm.weight"]
        self.head = Linear(g, "head")
        self.head.w = self.head.w if self.head.scales is not None else self.head.w.astype(mx.float32)
        materialize(self.layers, self.head, self.norm)

    def new_state(self) -> SeqState:
        a = self.a
        comp = {}
        for i in a.kv_source_layers:
            r = a.compress_ratios[i]
            comp[i] = CompressorState(r, Growable(a.head_dim), Growable(a.index_head_dim) if i in a.index_source_layers else None)
        return SeqState(rings=[Ring(a.window_size, a.head_dim) for _ in range(a.n_layers)], comp=comp)

    def embed_tokens(self, tokens: list[int]) -> mx.array:
        return self.w.embed(np.asarray(tokens, dtype=np.int64))

    def forward(
        self,
        st: SeqState,
        tokens: list[int],
        replay_from: int | None = None,
        all_logits: bool = False,
        want_hidden: bool = False,
    ):
        """Feed `tokens` at positions st.pos.. and return logits (last position, or every position).

        replay_from: SWA Bounded Replay. Layers up to the last KV source (the "encoder") see every
        token; the layers after it (the "decoder", which read only that source's compressed KV
        plus their own window) see only tokens at index >= replay_from of this chunk.
        """
        a = self.a
        L = len(tokens)
        start = st.pos
        positions = mx.arange(start, start + L)
        hash_ids = self.ngram(st, tokens) if self.ngram is not None else None
        st.tokens.extend(tokens)

        h = self.embed_tokens(tokens)
        h = mx.repeat(h[:, None, :], a.hc_mult, axis=1)
        pre_mix = mx.zeros((L, a.hc_mult), dtype=mx.float32)
        pre_mix[:, 0] = 1.0
        sh = Shared()
        last_source = max(a.kv_source_layers)
        hiddens = []
        for i, layer in enumerate(self.layers):
            if replay_from and i == last_source + 1 and replay_from >= L:
                st.pos = start + L  # a non-final prefill chunk: the decoder replays nothing
                return (None, None) if want_hidden else None
            if replay_from and i == last_source + 1:
                h, pre_mix, positions = h[replay_from:], pre_mix[replay_from:], positions[replay_from:]
                if sh.topk is not None:
                    sh.topk = sh.topk[replay_from:]
                if sh.candidates is not None:
                    sh.candidates = sh.candidates[replay_from:]
                hash_ids = None if hash_ids is None else hash_ids[replay_from:]
            if layer.engram is not None:
                h = layer.engram(h, hash_ids[:, a.engram_layer_ids.index(i)], None)
            if i in a.dspark_target_layer_ids:
                hiddens.append(h.mean(axis=1))
            h, pre_mix = layer(h, positions, pre_mix, st, st.rings[i], sh, start=start + ((replay_from or 0) if i > last_source else 0))
        st.pos = start + L
        h = Block.hc_pre(h, pre_mix)
        if not all_logits:
            h = h[-1:]
        logits = self.head(rms_norm(h, self.norm, a.norm_eps)).astype(mx.float32)
        if want_hidden:
            # DSpark's input: the target layers' (mean over hc copies) inputs, for every position
            # the decoder processed -- the last rows of this chunk
            return logits, mx.concatenate(hiddens, axis=-1) if hiddens else None
        return logits
