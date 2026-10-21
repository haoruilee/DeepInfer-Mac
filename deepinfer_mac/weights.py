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


def available_memory() -> int:
    """Memory macOS can hand out without pressure: free + inactive + purgeable + speculative pages."""
    out = subprocess.check_output(["vm_stat"], text=True)
    page = int(out.split("page size of")[1].split()[0])
    pages = {l.split(":")[0].strip(): int(l.split(":")[1].strip().rstrip(".")) for l in out.splitlines()[1:] if ":" in l}
    return page * sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages purgeable", "Pages speculative"))


def default_expert_budget(root: Path, margin_gb: float = 2.5, os_reserve_gb: float = 6.0) -> int:
    """Size the expert slabs from what is actually free right now (after the dense weights about to
    load and a safety margin), capped so macOS keeps `os_reserve_gb` even on an idle machine.
    A larger cache is the main decode lever: 1.5x capacity lifts the hit rate 0.38 -> 0.47 on real
    routing traces. Between 2 GB and (RAM - dense - os_reserve)."""
    dense = sum(p.stat().st_size for p in (root / "dense").glob("L*.safetensors")) + (root / "dense" / "globals.safetensors").stat().st_size
    cap = total_memory() - dense - int(os_reserve_gb * (1 << 30))
    want = available_memory() - dense - int(margin_gb * (1 << 30))
    return int(max(2 << 30, min(cap, want)))


class DiskWeights:
    def __init__(self, root: Path, dim: int, engram_layers, expert_budget: int | None = None, io_threads: int = 8):
        self.root = root = Path(root)
        layout = json.loads((root / "layout.json").read_text())
        self.globals = mx.load(str(root / "dense" / "globals.safetensors"))
        budget = expert_budget if expert_budget is not None else default_expert_budget(root)
        self.experts = ExpertStore(root, budget, io_threads)
        eg = layout.get("engram", {})
        self.engram = None
        if engram_layers:
            done = (root / "progress.log").read_text().split() if (root / "progress.log").exists() else []
            n_eg = sum(t.startswith("EG-") for t in done)
            total = sum(-(-rows // (1 << 20)) for rows in json.loads((root / "inference_config.json").read_text())["engram_num_embeddings"])
            if (root / "progress.log").exists() and n_eg < total:
                from .model.engram import RemoteEngramTable

                print(f"[weights] engram tables {n_eg}/{total} converted: reading rows from the Hub meanwhile", flush=True)
                self.engram = RemoteEngramTable()
            else:
                self.engram = SSDEngramTable(root, engram_layers, dim=eg.get("dim", 256), bits=eg.get("bits", 4), group=eg.get("group", 64))
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
