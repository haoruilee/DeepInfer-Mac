"""DSpark speculative decoding (the mtp.* stages of DeepSeek-V4.1-Flash).

The reference implements only the draft forward (DSparkBlock / forward_spec); the loop around it is
DeepInfer's. Here:

  draft:   from the newest target token t (at position p+1) and the target's hidden at position
           p, the three draft stages run a block of `block_size` positions at once -- t followed
           by noise tokens -- attending to a sliding window of *target* hidden states (seeded by
           `seed`) plus the whole block. Tokens then come out one by one through the Markov head
           (logits bias from the previous token) and a confidence head scores each.
  verify:  the target runs [t, d1..dk] in one forward; drafts are accepted by speculative
           rejection sampling (exact for any temperature; argmax match when greedy). k is the
           longest prefix whose confidence clears `threshold` (DeepInfer: "confidence estimates
           deciding when to verify").
  rollback: on a partial accept the target state is restored and only the accepted tokens are
           re-fed -- their experts are still in the LRU, so this costs dense compute only.
"""

from __future__ import annotations

import time

import numpy as np
import mlx.core as mx

from . import engine as E
from .model import ops
from .model import model as M
from .model.model import Args, Block, Linear, Ring, rms_norm, rope_tail


class DSparkBlock(Block):
    def attn_dspark(self, x: mx.array, p: int, main_x: mx.array, ring: Ring) -> mx.array:
        at, a, rd = self.attn, self.a, self.a.rope_head_dim
        win, n = a.window_size, x.shape[0]
        # the target's state at position p enters this stage's window
        main_kv = M.act_quant(rope_tail(rms_norm(at.wkv(main_x), at.kv_norm, a.norm_eps), at.rope.cos_sin(mx.array([p])), rd))
        seed_ring(ring, main_kv, [p], win)
        cs = at.rope.cos_sin(mx.arange(p + 1, p + 1 + n))
        qr = rms_norm(at.wq_a(x), at.q_norm, a.norm_eps)
        q = rope_tail(at.wq_b(qr).reshape(n, a.n_heads, a.head_dim), cs, rd)
        kv = M.act_quant(rope_tail(rms_norm(at.wkv(x), at.kv_norm, a.norm_eps), cs, rd))
        valid = np.nonzero((ring.pos >= 0) & (ring.pos > p - win) & (ring.pos <= p))[0]
        idx = np.concatenate([valid, win + np.arange(n)]).astype(np.int32)  # whole block, both ways
        o = ops.sparse_attn(q, mx.concatenate([ring.kv, kv]), at.sink, mx.broadcast_to(mx.array(idx), (n, idx.size)), at.scale)
        o = M.act_quant(rope_tail(o, cs, rd, inverse=True).reshape(n, a.o_groups, -1))
        return at.wo_b(at.wo_a(o))

    def step(self, x, p, pre_mix, main_x, ring):
        eps = self.a.norm_eps
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, "attn")
        h = rms_norm(self.hc_pre(x, pre_mix), self.attn_norm, eps)
        x = self.hc_post(self.attn_dspark(h, p, main_x, ring), residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, "ffn")
        h = rms_norm(self.hc_pre(x, attn_pre), self.ffn_norm, eps)
        return self.hc_post(self.ffn(h), residual, ffn_post, ffn_comb), ffn_pre


def seed_ring(ring: Ring, kv: mx.array, positions, win: int):
    positions = np.asarray(positions)
    keep = positions[-win:]
    src = np.full(win, -1, dtype=np.int32)
    src[keep % win] = np.arange(len(positions) - len(keep), len(positions))
    has = mx.array(src >= 0)
    ring.kv = mx.where(has[:, None], kv[mx.array(np.maximum(src, 0))].astype(ring.kv.dtype), ring.kv)
    ring.pos = ring.pos.copy()
    ring.pos[keep % win] = keep


class DSpark:
    ADAPTIVE = True  # fall back to plain decoding when it measures faster

    def __init__(self, args: Args, weights, model, threshold: float = 0.5):
        self.a, self.model, self.threshold = args, model, threshold
        n = args.n_mtp_layers
        ds = [weights.mtp(j) for j in range(n)]
        self.blocks = [DSparkBlock(args, args.n_layers + j, ds[j], weights.experts, key=f"M{j}", is_backbone=False) for j in range(n)]
        self.main_proj = Linear(ds[0], "main_proj")
        self.main_norm = ds[0]["main_norm.weight"]
        last = ds[-1]
        self.norm = last["norm.weight"]
        self.markov_embed = last["markov_head.embed.weight"]
        self.markov_head = last["markov_head.head.weight"].astype(mx.float32)
        self.conf = last["confidence_head.proj.weight"].astype(mx.float32)
        M.materialize(self.blocks, self.main_proj, self.main_norm, self.norm, self.markov_embed, self.markov_head, self.conf)

    def rings(self, st):
        if not st.mtp_rings:
            st.mtp_rings = [Ring(self.a.window_size, self.a.head_dim) for _ in self.blocks]
        return st.mtp_rings

    def _main_x(self, hidden: mx.array) -> mx.array:
        return rms_norm(self.main_proj(hidden.astype(mx.bfloat16)), self.main_norm, self.a.norm_eps)

    def seed(self, st, hidden: mx.array, positions) -> None:
        """Write target hidden states at `positions` into every stage's window (prefill seeding)."""
        if len(positions) == 0:
            return
        main_x = self._main_x(hidden)
        cs = None
        for blk, ring in zip(self.blocks, self.rings(st)):
            at = blk.attn
            cs = at.rope.cos_sin(mx.array(np.asarray(positions)))
            kv = M.act_quant(rope_tail(rms_norm(at.wkv(main_x), at.kv_norm, self.a.norm_eps), cs, self.a.rope_head_dim))
            seed_ring(ring, kv, positions, self.a.window_size)

    def draft(self, st, token: int, hidden: mx.array, p: int, sp, rng):
        """-> draft ids [B], draft logits [B, V] (with Markov bias), confidences [B]."""
        a = self.a
        B = a.dspark_block_size
        main_x = self._main_x(hidden[None] if hidden.ndim == 1 else hidden)
        ids = [token] + [a.dspark_noise_token_id] * (B - 1)
        x = mx.repeat(self.model.embed_tokens(ids)[:, None, :], a.hc_mult, axis=1)
        pre_mix = mx.zeros((B, a.hc_mult), dtype=mx.float32)
        pre_mix[:, 0] = 1.0
        for blk, ring in zip(self.blocks, self.rings(st)):
            x, pre_mix = blk.step(x, p, pre_mix, main_x, ring)
        h = Block.hc_pre(x, pre_mix)
        logits = self.model.head(rms_norm(h, self.norm, a.norm_eps)).astype(mx.float32)
        out, prev, membs, lg_all = [], token, [], []
        for i in range(B):
            emb = self.markov_embed[prev]
            lg = logits[i] + self.markov_head @ emb.astype(mx.float32)
            membs.append(emb)
            lg_np = np.array(lg, dtype=np.float32)
            lg_all.append(lg_np)
            prev = E.sample(lg_np, sp, rng)
            out.append(prev)
        conf = (mx.concatenate([h.astype(mx.float32), mx.stack(membs).astype(mx.float32)], axis=-1) @ self.conf.T)[:, 0]
        return out, np.stack(lg_all), np.array(mx.sigmoid(conf))

    # ------------------------------------------------------------------------------ loop

    PROBE_EVERY = 16  # re-measure the losing mode this often

    def _choose(self, rate: dict, step: int) -> str:
        """Speculate or not, by measured throughput. On an SSD-bound machine verifying drafts reads
        the union of their experts, so plain decoding can win; this keeps --spec from being slower
        than plain beyond the occasional probe."""
        if not self.ADAPTIVE:
            return "spec"
        if rate["spec"] is None:
            return "spec"
        if rate["plain"] is None:
            return "plain"
        best, other = ("spec", "plain") if rate["spec"] >= rate["plain"] else ("plain", "spec")
        return other if step % self.PROBE_EVERY == 0 else best

    @staticmethod
    def _rate(rate: dict, mode: str, tokens: int, seconds: float) -> None:
        r = tokens / max(seconds, 1e-6)
        rate[mode] = r if rate[mode] is None else 0.7 * rate[mode] + 0.3 * r


    def generate(self, eng, st, logits, hidden, sp, rng, grammar, stops, n_prompt):
        if grammar is not None or hidden is None:
            yield from eng._plain_decode(st, logits, sp, rng, grammar, stops, n_prompt)
            return
        m = self.model
        n_h = hidden.shape[0]
        positions = list(range(st.pos - n_h, st.pos))
        self.seed(st, hidden[:-1], positions[:-1])
        mh, b = hidden[-1], st.pos - 1  # last fed position and its hidden
        lg = np.array(logits[-1], dtype=np.float32)
        t = E.sample(lg, sp, rng)
        yield E.RequestOutput(t, float(E.log_softmax(lg)[t]), n_prompt, text=eng.tokenizer.decode([t], skip_special_tokens=False))
        produced = 1
        mode_rate = {"spec": None, "plain": None}  # measured tokens/second per mode (EMA)
        step = 0
        while t not in stops and produced < sp.max_tokens:
            mode = self._choose(mode_rate, step)
            step += 1
            t0 = time.perf_counter()
            if mode == "plain":
                # no drafting: one target step, with the draft window kept seeded so speculation
                # can resume at any point
                self.seed(st, mh[None] if mh.ndim == 1 else mh, [b])
                lg, h = m.forward(st, [t], want_hidden=True)
                lg = np.array(lg[-1], dtype=np.float32)
                nxt, lp, top = E.sample_lp(lg, sp, rng)
                self._rate(mode_rate, "plain", 1, time.perf_counter() - t0)
                yield E.RequestOutput(nxt, lp, n_prompt, text=eng.tokenizer.decode([nxt], skip_special_tokens=False), top_logprobs=top)
                produced += 1
                b, mh, t = b + 1, h[-1], nxt
                continue
            drafts, dlog, conf = self.draft(st, t, mh, b, sp, rng)
            # DeepInfer ranks drafts by the running product of confidences
            prod = np.cumprod(conf)
            k = 0
            while k < min(len(drafts), eng.draft_len, sp.max_tokens - produced - 1) and prod[k] >= self.threshold:
                k += 1
            snap = st.snapshot() if k else None
            vlog, vhid = m.forward(st, [t] + drafts[:k], all_logits=True, want_hidden=True)
            vlog = np.array(vlog, dtype=np.float32)
            acc, nxt = accept(vlog, dlog[:k], drafts[:k], sp, rng)
            if acc < k:
                st.restore(snap)
                _, vhid = m.forward(st, [t] + drafts[:acc], all_logits=True, want_hidden=True)
            self.seed(st, vhid[:acc], list(range(b + 1, b + 1 + acc)))
            emitted = drafts[:acc] + [nxt]
            self._rate(mode_rate, "spec", len(emitted), time.perf_counter() - t0)
            for j, tok in enumerate(emitted):
                lp = float(E.log_softmax(vlog[j])[tok])
                last = j == len(emitted) - 1
                yield E.RequestOutput(tok, lp, n_prompt, draft_tok=(k + 1) if last else 0, accept_tok=(acc + 1) if last else 0,
                                      text=eng.tokenizer.decode([tok], skip_special_tokens=False))
                produced += 1
                if tok in stops or produced >= sp.max_tokens:
                    return
            b, mh, t = b + 1 + acc, vhid[acc], nxt


def _probs(logits: np.ndarray, sp) -> np.ndarray:
    x = logits.astype(np.float64) / max(sp.temperature, 1e-5)
    if sp.top_k and sp.top_k < x.size:
        x[x < np.partition(x, -sp.top_k)[-sp.top_k]] = -np.inf
    p = np.exp(x - x.max())
    p /= p.sum()
    if sp.top_p < 1.0:
        order = np.argsort(-p)
        c = np.cumsum(p[order])
        p[order[np.searchsorted(c, sp.top_p) + 1 :]] = 0
        p /= p.sum()
    return p


def accept(vlog: np.ndarray, dlog: np.ndarray, drafts: list[int], sp, rng) -> tuple[int, int]:
    """Speculative sampling. vlog[j] is the target's distribution for the token after the j-th
    fed token, so it judges drafts[j]; vlog[k] supplies the bonus token. Returns (#accepted, next)."""
    k = len(drafts)
    for j in range(k):
        d = drafts[j]
        if sp.temperature <= 0:
            if int(np.argmax(vlog[j])) != d:
                return j, int(np.argmax(vlog[j]))
            continue
        p, q = _probs(vlog[j], sp), _probs(dlog[j], sp)
        if rng.random() * q[d] > p[d]:  # reject with prob 1 - p/q
            r = np.maximum(p - q, 0)
            r = r / r.sum() if r.sum() > 0 else p
            return j, int(rng.choice(r.size, p=r))
    last = vlog[k]
    return k, int(np.argmax(last)) if sp.temperature <= 0 else int(rng.choice(last.size, p=_probs(last, sp)))
