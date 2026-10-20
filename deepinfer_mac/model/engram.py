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
    """The converted 4-bit tables on SSD: 144-byte rows, read with parallel preads. Each
    position needs (n-1) * heads = 24 rows per engram layer; duplicates are read once."""

    def __init__(self, root: Path, layer_ids, threads: int = 16):
        self.fds = {l: os.open(root / "engram" / f"L{l:02d}.bin", os.O_RDONLY) for l in layer_ids}
        self.pool = ThreadPoolExecutor(threads)
        self.row = F.ENGRAM_ROW

    def _read(self, fd: int, rows: np.ndarray) -> np.ndarray:
        out = np.empty((len(rows), self.row), dtype=np.uint8)
        for i, r in enumerate(rows):
            out[i] = np.frombuffer(os.pread(fd, self.row, int(r) * self.row), dtype=np.uint8)
        return out

    def fetch(self, layer_id: int, ids: np.ndarray):
        """Start reading; returns a future of (unique ids, raw rows, inverse)."""
        uniq, inv = np.unique(ids.reshape(-1), return_inverse=True)
        fd = self.fds[layer_id]
        parts = np.array_split(uniq, max(1, min(16, len(uniq) // 32)))
        futs = [self.pool.submit(self._read, fd, p) for p in parts]
        return uniq, futs, inv

    def lookup(self, layer_id: int, ids: np.ndarray, pending=None) -> mx.array:
        uniq, futs, inv = pending if pending is not None else self.fetch(layer_id, ids)
        raw = np.concatenate([f.result() for f in futs])
        g = F.ENGRAM_DIM // F.GROUP
        q = mx.array(raw[:, :128].copy()).view(mx.uint32)
        s = mx.array(raw[:, 128 : 128 + 2 * g].copy()).view(mx.bfloat16)
        b = mx.array(raw[:, 128 + 2 * g :].copy()).view(mx.bfloat16)
        rows = mx.dequantize(q, s, b, group_size=F.GROUP, bits=F.ENGRAM_BITS)
        return rows[mx.array(inv.astype(np.int32))].reshape(*ids.shape, F.ENGRAM_DIM)
