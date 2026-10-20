"""Routed experts streamed from SSD into a bounded LRU cache in unified memory.

Each expert is one page-aligned record (format.ExpertLayout), read with a single pread on a
descriptor opened with F_NOCACHE: the OS page cache would otherwise keep a second copy of every
expert we already hold, which a 24 GB machine cannot afford. Reads run on a thread pool so the six
experts of a decode step (or a prefetch for the next layer) load in parallel; measured on an M4
MacBook Air this sustains ~2.8 GB/s.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import format as F
from .model.model import ExpertWeights

F_NOCACHE = 48


class ExpertStore:
    def __init__(self, root: Path, budget_bytes: int, threads: int = 8):
        root = Path(root)
        meta = json.loads((root / "layout.json").read_text())["experts"]
        self.layouts = {k: F.ExpertLayout(v["bits_w1"], v["bits_w3"], v["bits_w2"], v.get("dim", F.DIM), v.get("inter", F.INTER)) for k, v in meta.items()}
        self.paths = {k: root / "experts" / f"{k}.bin" for k in self.layouts}
        self.fds: dict[str, int] = {}
        self.budget = budget_bytes
        self.cache: OrderedDict[tuple[str, int], ExpertWeights] = OrderedDict()
        self.used = 0
        self.inflight: dict[tuple[str, int], Future] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix="expert-io")
        self.stats = {"hits": 0, "misses": 0, "bytes": 0, "wait_s": 0.0}

    def _fd(self, key: str) -> int:
        fd = self.fds.get(key)
        if fd is None:
            fd = os.open(self.paths[key], os.O_RDONLY)
            fcntl.fcntl(fd, F_NOCACHE, 1)
            self.fds[key] = fd
        return fd

    def _read(self, key: str, e: int) -> np.ndarray:
        lay = self.layouts[key]
        buf = np.empty(lay.nbytes, dtype=np.uint8)
        n = os.preadv(self._fd(key), [buf], e * lay.stride)
        if n != lay.nbytes:
            raise IOError(f"short read on {key} expert {e}: {n} != {lay.nbytes}")
        return buf

    def _to_mx(self, key: str, buf: np.ndarray) -> ExpertWeights:
        lay = self.layouts[key]
        arrs = {}
        for name, shape, dtype, off, n in lay.parts():
            a = buf[off : off + n]
            arrs[name] = mx.array(a.view(np.uint32).reshape(shape)) if dtype == "uint32" else mx.array(a.view(np.uint16).reshape(shape)).view(mx.bfloat16)
        bits = {"w1": lay.bits_w1, "w3": lay.bits_w3, "w2": lay.bits_w2}
        return ExpertWeights(*[(arrs[f"{w}.q"], arrs[f"{w}.s"], arrs[f"{w}.b"], bits[w], "affine") for w in ("w1", "w3", "w2")])

    def prefetch(self, key: str, ids) -> None:
        with self.lock:
            for e in ids:
                k = (key, int(e))
                if k not in self.cache and k not in self.inflight:
                    self.inflight[k] = self.pool.submit(self._read, key, int(e))

    def get(self, key: str, ids) -> list[ExpertWeights]:
        self.prefetch(key, ids)
        out = []
        for e in ids:
            k = (key, int(e))
            with self.lock:
                hit = self.cache.get(k)
                if hit is not None:
                    self.cache.move_to_end(k)
                    self.stats["hits"] += 1
                    out.append(hit)
                    continue
                fut = self.inflight.pop(k)
            t = time.perf_counter()
            buf = fut.result()
            self.stats["wait_s"] += time.perf_counter() - t
            ew = self._to_mx(key, buf)
            with self.lock:
                self.stats["misses"] += 1
                self.stats["bytes"] += buf.nbytes
                self.cache[k] = ew
                self.used += self.layouts[key].stride
                self._evict(keep={(key, int(x)) for x in ids})
            out.append(ew)
        return out

    def _evict(self, keep: set):
        while self.used > self.budget and self.cache:
            k, _ = next(iter(self.cache.items()))
            if k in keep:
                self.cache.move_to_end(k)
                if all(x in keep for x in self.cache):
                    break
                continue
            self.cache.popitem(last=False)
            self.used -= self.layouts[k[0]].stride

    def hit_rate(self) -> float:
        n = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / n if n else 0.0
