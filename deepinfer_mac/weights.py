"""Weights of a converted checkpoint (see format.py): dense resident, experts and engram on SSD."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np

from .model.engram import SSDEngramTable
from .store import ExpertStore


def total_memory() -> int:
    return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())


def dense_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in (root / "dense").glob("*.safetensors"))


def default_expert_budget(root: Path, reserve_gb: float = 9.5) -> int:
    """Whatever unified memory is left after the resident dense weights and a reserve for macOS,
    other apps and activations; at least 2 GB."""
    return max(2 << 30, int(total_memory() - dense_bytes(root) - reserve_gb * (1 << 30)))


class DiskWeights:
    def __init__(self, root: Path, dim: int, engram_layers, expert_budget: int | None = None, io_threads: int = 8):
        self.root = root = Path(root)
        layout = json.loads((root / "layout.json").read_text())
        self.globals = mx.load(str(root / "dense" / "globals.safetensors"))
        budget = expert_budget if expert_budget is not None else default_expert_budget(root)
        self.experts = ExpertStore(root, budget, io_threads)
        eg = layout.get("engram", {})
        self.engram = SSDEngramTable(root, engram_layers, dim=eg.get("dim", 256), bits=eg.get("bits", 4), group=eg.get("group", 64)) if engram_layers else None
        self._embed = np.memmap(root / "embed.bf16", dtype=np.uint16, mode="r").reshape(-1, dim)

    def layer(self, i: int) -> dict:
        return mx.load(str(self.root / "dense" / f"L{i:02d}.safetensors"))

    def mtp(self, i: int) -> dict:
        return mx.load(str(self.root / "dense" / f"M{i}.safetensors"))

    def has_mtp(self, n: int) -> bool:
        return all((self.root / "dense" / f"M{i}.safetensors").exists() for i in range(n)) and all(
            (self.root / "experts" / f"M{i}.bin").exists() for i in range(n)
        )

    def embed(self, ids: np.ndarray) -> mx.array:
        return mx.array(np.ascontiguousarray(self._embed[ids])).view(mx.bfloat16)
