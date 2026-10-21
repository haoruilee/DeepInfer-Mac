"""Routed experts streamed from SSD into preallocated slabs in unified memory.

Each expert is one page-aligned record on disk (format.ExpertLayout). For every distinct layout
the store owns one *slab*: nine MLX arrays [slots, ...] (w1/w3/w2 x packed/scales/biases),
allocated once. A miss takes an LRU slot and a single `preadv` on an F_NOCACHE descriptor scatters
the record straight into that slot of the nine arrays -- unified memory makes MLX allocations
directly writable, so there is no staging copy, no per-miss allocation and no MLX call off the
compute thread (MLX streams are per-thread). The model then runs all routed experts of a layer
with one `gather_qmm` per projection over slot indices.

The OS page cache is bypassed (F_NOCACHE): it would hold a second copy of experts we already hold,
which a 24 GB machine cannot afford. Measured on an M4 MacBook Air: 2.8-3.15 GB/s.

Slot safety: slots served by the last two `get_slots` calls stay pinned, because the compute that
reads them is lazy until the next routing sync; eviction never hands those out.
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

from . import format as F
from .model.model import ExpertWeights

F_NOCACHE = 48
_DT = {"uint32": mx.uint32, "bfloat16": mx.uint16}  # bf16 slabs are stored as uint16 bits


class Slab:
    def __init__(self, layout: F.ExpertLayout, slots: int):
        self.layout, self.slots = layout, slots
        self.parts = {name: mx.zeros((slots, *shape), dtype=_DT[dtype]) for name, shape, dtype, _, _ in layout.parts()}
        mx.eval(list(self.parts.values()))
        self.nbytes = {name: n for name, _, _, _, n in layout.parts()}
        self.order = [name for name, *_ in layout.parts()]
        self.mem = {name: memoryview(a).cast("B") for name, a in self.parts.items()}
        self.bf16 = {name: a.view(mx.bfloat16) for name, a in self.parts.items() if a.dtype == mx.uint16}
        mx.eval(list(self.bf16.values()))  # evaluated views share the slab's memory (SSD writes stay visible)
        self.lru: OrderedDict[tuple[str, int], int] = OrderedDict()  # (key, expert) -> slot
        self.free = list(range(slots))
        self.pins: list[set[int]] = [set(), set()]

    def iovecs(self, slot: int) -> list[memoryview]:
        return [self.mem[n][slot * self.nbytes[n] : (slot + 1) * self.nbytes[n]] for n in self.order]

    def proj(self, w: str):
        """(packed, scales, biases, bits) of projection w1/w3/w2 across all slots."""
        bits = {"w1": self.layout.bits_w1, "w3": self.layout.bits_w3, "w2": self.layout.bits_w2}[w]
        return self.parts[f"{w}.q"], self.bf16[f"{w}.s"], self.bf16[f"{w}.b"], bits


class ExpertStore:
    def __init__(self, root: Path, budget_bytes: int, threads: int = 8):
        root = Path(root)
        meta = json.loads((root / "layout.json").read_text())["experts"]
        self.layouts = {
            k: F.ExpertLayout(v["bits_w1"], v["bits_w3"], v["bits_w2"], v.get("dim", F.DIM), v.get("inter", F.INTER))
            for k, v in meta.items()
        }
        self.paths = {k: root / "experts" / f"{k}.bin" for k in self.layouts}
        self.fds: dict[str, int] = {}
        self.budget = budget_bytes
        # one slab per distinct layout, sized by its share of the experts' bytes
        groups: dict[F.ExpertLayout, list[str]] = {}
        for k, lay in self.layouts.items():
            groups.setdefault(lay, []).append(k)
        total = sum(len(ks) * lay.stride for lay, ks in groups.items())
        self.slabs: dict[str, Slab] = {}
        for lay, ks in groups.items():
            slots = max(96, int(budget_bytes * len(ks) * lay.stride / total) // lay.stride)
            slab = Slab(lay, slots)
            for k in ks:
                self.slabs[k] = slab
        self.used = sum(s.slots * s.layout.stride for s in set(self.slabs.values()))
        self.inflight: dict[tuple[str, int], Future] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix="expert-io")
        self._pad = threading.local()
        self.stats = {"hits": 0, "misses": 0, "prefetch_hits": 0, "bytes": 0, "wait_s": 0.0}
        self._prefetched: set[tuple[str, int]] = set()

    # -- IO

    def _fd(self, key: str) -> int:
        fd = self.fds.get(key)
        if fd is None:
            fd = os.open(self.paths[key], os.O_RDONLY)
            fcntl.fcntl(fd, F_NOCACHE, 1)
            self.fds[key] = fd
        return fd

    def _read(self, key: str, e: int, slot: int) -> None:
        """IO thread: one preadv of the whole page-aligned record, scattered into the slab."""
        slab, lay = self.slabs[key], self.layouts[key]
        pad = lay.stride - lay.nbytes
        if getattr(self._pad, "buf", None) is None or len(self._pad.buf) < pad:
            self._pad.buf = bytearray(max(pad, 1))
        bufs = slab.iovecs(slot) + ([memoryview(self._pad.buf)[:pad]] if pad else [])
        off, want = e * lay.stride, lay.stride
        while want:
            n = os.preadv(self._fd(key), bufs, off)
            if n <= 0:
                raise IOError(f"read failed on {key} expert {e} at {off}")
            off, want = off + n, want - n
            while bufs and n >= len(bufs[0]):  # short read: drop what was filled, continue
                n -= len(bufs[0])
                bufs.pop(0)
            if bufs and n:
                bufs[0] = bufs[0][n:]

    # -- slots

    def _take_slot(self, slab: Slab, optional: bool, protect: set[int] = frozenset()) -> int | None:
        if slab.free:
            return slab.free.pop()
        pinned = slab.pins[0] | slab.pins[1] | protect
        for k, slot in slab.lru.items():  # oldest first
            if slot in pinned:
                continue
            fut = self.inflight.get(k)
            if fut is not None:
                if not fut.done():
                    continue  # a read still landing in this slot
                del self.inflight[k]  # finished read nobody asked for (a lookahead misprediction)
            del slab.lru[k]
            self._prefetched.discard(k)
            return slot
        if optional:
            return None
        raise MemoryError("expert cache too small for one layer's experts; raise --cache-gb")

    def prefetch(self, key: str, ids, optional: bool = False) -> None:
        """Start reads for experts not resident. `optional` (lookahead) never evicts pinned
        slots and silently skips when the slab is full. Experts of this same request that are
        already resident are never evicted to make room for the others."""
        slab = self.slabs[key]
        ids = [int(e) for e in ids]
        with self.lock:
            protect = {slab.lru[(key, e)] for e in ids if (key, e) in slab.lru}
            for e in ids:
                k = (key, e)
                if k in slab.lru:
                    continue
                slot = self._take_slot(slab, optional, protect)
                if slot is None:
                    return
                slab.lru[k] = slot
                protect.add(slot)
                self.inflight[k] = self.pool.submit(self._read, key, e, slot)
                self.stats["bytes"] += self.layouts[key].nbytes
                if optional:
                    self._prefetched.add(k)

    def resident(self, key: str, e: int) -> bool:
        k = (key, int(e))
        fut = self.inflight.get(k)
        return k in self.slabs[key].lru and (fut is None or fut.done())

    def get_slots(self, key: str, ids) -> tuple[Slab, list[int]]:
        """Make `ids` resident; return their slab and slots (blocking until loaded)."""
        slab = self.slabs[key]
        ids = [int(e) for e in ids]
        with self.lock:
            for e in ids:
                k = (key, e)
                if k in slab.lru and k not in self.inflight:
                    self.stats["prefetch_hits" if k in self._prefetched else "hits"] += 1
                else:
                    self.stats["misses"] += 1
                self._prefetched.discard(k)
        self.prefetch(key, ids)
        slots = []
        for e in ids:
            k = (key, e)
            fut = self.inflight.get(k)
            if fut is not None:
                t = time.perf_counter()
                fut.result()
                self.stats["wait_s"] += time.perf_counter() - t
                with self.lock:
                    self.inflight.pop(k, None)
            with self.lock:
                slab.lru.move_to_end(k)
                slots.append(slab.lru[k])
        slab.pins = [slab.pins[1], set(slots)]
        return slab, slots

    def get(self, key: str, ids) -> list[ExpertWeights]:
        """Per-expert views (tests, benchmarks); the model uses get_slots + gather_qmm."""
        slab, slots = self.get_slots(key, ids)
        out = []
        for s in slots:
            out.append(ExpertWeights(*[
                (q[s], sc[s], b[s], bits, "affine") for q, sc, b, bits in (slab.proj("w1"), slab.proj("w3"), slab.proj("w2"))
            ]))
        return out

    def hit_rate(self) -> float:
        s = self.stats
        n = s["hits"] + s["misses"] + s["prefetch_hits"]
        return (s["hits"] + s["prefetch_hits"]) / n if n else 0.0
