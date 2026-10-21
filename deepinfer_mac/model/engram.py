"""Engram n-gram hashing (numpy port of inference/engram.py) and table backends.

The hash ids of a position depend only on the tokens, so they are computed on the CPU before the
forward pass and the table rows can be fetched from SSD while the first layers run.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

from .. import format as F

DEAD = -1


def _isprime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def layout_primes(layer_ids, max_ngram_size: int, n_heads: int, vocab_size: int) -> list:
    """[layer][ngram-1][head] prime bucket sizes, drawn in order and never reused (EngramLayout)."""
    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, cur = [], vocab_size - 1
            for _ in range(n_heads):
                cur += 1
                while not _isprime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                sizes.append(cur)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    return primes


def hash_multipliers(layer_ids, max_ngram_size: int, compressed_vocab: int) -> np.ndarray:
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab) // 2)
    rows = []
    for layer_id in layer_ids:
        g = np.random.default_rng(10007 * layer_id)
        rows.append(g.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64) * 2 + 1)
    return np.stack(rows)


def build_compressed_token_map(tokenizer_json: str) -> tuple[np.ndarray, int]:
    """Same normalization as inference/engram.py, on a raw `tokenizers.Tokenizer`."""
    from tokenizers import Regex, Tokenizer, normalizers

    tok = Tokenizer.from_file(tokenizer_json)
    sentinel = ""
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    n = tok.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = np.zeros(n, dtype=np.int64)
    for tid in range(n):
        text = tok.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(tid)
        else:
            nt = norm.normalize_str(text)
            key = nt if nt else text
        lookup[tid] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)


class NgramHasher:
    """Maps the tokens fed to a sequence onto engram hash ids [L, n_layers, n_hash_cols]."""

    def __init__(self, args, token_map: np.ndarray, compressed_vocab: int, engram_vocab_size: int, pad_id: int = 2):
        assert compressed_vocab == args.engram_compressed_vocab_size if hasattr(args, "engram_compressed_vocab_size") else True
        self.n = args.engram_max_ngram_size
        self.token_map = token_map
        self.pad = int(token_map[pad_id])
        primes = layout_primes(args.engram_layer_ids, self.n, args.engram_n_heads, engram_vocab_size)
        self.primes = np.array(primes, dtype=np.int64)  # [layers, n-1, heads]
        flat = [[p for per in layer for p in per] for layer in primes]
        self.offsets = np.array([np.cumsum([0, *f[:-1]]) for f in flat], dtype=np.int64)  # [layers, cols]
        self.mult = hash_multipliers(args.engram_layer_ids, self.n, compressed_vocab)  # [layers, n]

    @classmethod
    def load(cls, args, root: Path, engram_vocab_size: int):
        cache = root / "engram_token_map.npy"
        if cache.exists():
            tm = np.load(cache)
            cv = int(tm.max()) + 1
        else:
            tm, cv = build_compressed_token_map(str(root / "tokenizer.json"))
            np.save(cache, tm)
        return cls(args, tm, cv, engram_vocab_size)

    def __call__(self, st, tokens: list[int], token_mask: np.ndarray | None = None) -> np.ndarray:
        comp = self.token_map[np.asarray(tokens, dtype=np.int64)]
        if token_mask is not None:
            comp = np.where(token_mask, comp, DEAD)
        hist = np.asarray(st.compressed_ids[-(self.n - 1):] if st.compressed_ids else [], dtype=np.int64)
        st.compressed_ids.extend(comp.tolist())
        seq = np.concatenate([hist, comp])
        h0 = len(hist)
        start = len(st.compressed_ids) - len(comp)  # absolute position of comp[0]
        L = len(comp)
        toks, blocked = [], np.zeros(L, dtype=bool)
        for shift in range(self.n):
            idx = np.arange(L) + h0 - shift
            pos = start + np.arange(L) - shift
            src = np.where(idx >= 0, seq[np.maximum(idx, 0)], self.pad)
            blocked = blocked | (pos < 0) | (src == DEAD)
            toks.append(np.where(blocked, self.pad, src))
        toks = np.stack(toks, axis=-1)  # [L, n]
        prod = toks[:, None, :] * self.mult[None]  # [L, layers, n] (int64 wraparound as torch)
        rolling = prod[..., 0]
        out = []
        for i in range(1, self.n):
            rolling = np.bitwise_xor(rolling, prod[..., i])
            out.append(rolling[..., None] % self.primes[None, :, i - 1])
        return np.concatenate(out, axis=-1) + self.offsets[None]


class MemoryEngramTable:
    """Tables held as dense bf16 arrays (tests, tiny models)."""

    def __init__(self, tables: dict[int, mx.array]):
        self.tables = tables

    def lookup(self, layer_id: int, ids: np.ndarray) -> mx.array:
        return self.tables[layer_id][mx.array(ids.astype(np.int64))]


class SSDEngramTable:
    """The converted 4-bit tables on SSD: fixed-size rows read with parallel preads on F_NOCACHE
    descriptors (no read-ahead or page-cache pollution on a 24 GB machine), behind a small LRU of
    recently used rows -- n-grams repeat, and a row is only ~160 bytes. Each position needs
    (n-1) * heads = 24 rows per engram layer."""

    def __init__(self, root: Path, layer_ids, dim: int = F.ENGRAM_DIM, bits: int = F.ENGRAM_BITS,
                 group: int = F.ENGRAM_GROUP, threads: int = 16, cache_rows: int = 1 << 20):
        import fcntl
        from collections import OrderedDict

        self.fds = {}
        for l in layer_ids:
            fd = os.open(root / "engram" / f"L{l:02d}.bin", os.O_RDONLY)
            fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
            self.fds[l] = fd
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix="engram-io")
        self.dim, self.bits, self.group = dim, bits, group
        self.qbytes = dim * bits // 8
        self.row = self.qbytes + 2 * (dim // group) * 2
        self.cache: OrderedDict = OrderedDict()
        self.cache_rows = cache_rows

    def _read(self, fd: int, rows: np.ndarray) -> np.ndarray:
        out = np.empty((len(rows), self.row), dtype=np.uint8)
        for i, r in enumerate(rows):
            out[i] = np.frombuffer(os.pread(fd, self.row, int(r) * self.row), dtype=np.uint8)
        return out

    def lookup(self, layer_id: int, ids: np.ndarray) -> mx.array:
        uniq, inv = np.unique(ids.reshape(-1), return_inverse=True)
        raw = np.empty((len(uniq), self.row), dtype=np.uint8)
        miss = []
        for i, r in enumerate(uniq.tolist()):
            hit = self.cache.get((layer_id, r))
            if hit is None:
                miss.append(i)
            else:
                self.cache.move_to_end((layer_id, r))
                raw[i] = hit
        if miss:
            miss = np.array(miss)
            parts = np.array_split(miss, max(1, min(16, len(miss) // 16)))
            futs = [(p, self.pool.submit(self._read, self.fds[layer_id], uniq[p])) for p in parts]
            for p, f in futs:
                raw[p] = f.result()
            for i in miss.tolist():
                self.cache[(layer_id, int(uniq[i]))] = raw[i].copy()
            while len(self.cache) > self.cache_rows:
                self.cache.popitem(last=False)
        g, qb = self.dim // self.group, self.qbytes
        q = mx.array(raw[:, :qb].copy()).view(mx.uint32)
        s = mx.array(raw[:, qb : qb + 2 * g].copy()).view(mx.bfloat16)
        b = mx.array(raw[:, qb + 2 * g :].copy()).view(mx.bfloat16)
        rows = mx.dequantize(q, s, b, group_size=self.group, bits=self.bits)
        return rows[mx.array(inv.astype(np.int32))].reshape(*ids.shape, self.dim)


class RemoteEngramTable:
    """Original fp8 Engram rows range-read from the Hub (ModelScope first) and cached in memory.
    Lets the model run -- slowly, ~48 small requests per token -- before the 200 GB of local
    tables finish converting. Used automatically while engram conversion is incomplete."""

    def __init__(self, threads: int = 32):
        import json as _json

        from ..convert import hf
        from .. import quant as Q

        self.hf, self.Q = hf, Q
        self.index = _json.loads((Path(__file__).parent.parent / "convert" / "tensor_index.json").read_text())
        self.rows: dict[tuple[int, int], np.ndarray] = {}
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix="engram-remote")

    def _fetch(self, layer: int, r: int) -> np.ndarray:
        w = self.index[f"layers.{layer}.engram.embed.weight"]
        s = self.index[f"layers.{layer}.engram.embed.scale"]
        a = self.hf.get_range(w["shard"], w["start"] + r * 256, w["start"] + (r + 1) * 256)
        b = self.hf.get_range(s["shard"], s["start"] + r * 8, s["start"] + (r + 1) * 8)
        return np.concatenate([np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)])

    def lookup(self, layer_id: int, ids: np.ndarray) -> mx.array:
        uniq, inv = np.unique(ids.reshape(-1), return_inverse=True)
        need = [int(r) for r in uniq if (layer_id, int(r)) not in self.rows]
        for r, v in zip(need, self.pool.map(lambda r: self._fetch(layer_id, r), need)):
            self.rows[(layer_id, r)] = v
        raw = np.stack([self.rows[(layer_id, int(r))] for r in uniq])
        rows = self.Q.dequant_fp8_rowscale(mx.array(raw[:, :256].copy()), mx.array(raw[:, 256:].copy())).astype(mx.bfloat16)
        return rows[mx.array(inv.astype(np.int32))].reshape(*ids.shape, 256)
