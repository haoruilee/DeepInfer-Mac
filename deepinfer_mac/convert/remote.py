"""Streaming converter: Hugging Face Hub -> DeepInfer-Mac layout, without ever storing a shard.

The released checkpoint is 510 GB; the target Mac has ~400 GB free. Every tensor is range-read
straight from the shards (offsets from the safetensors headers), converted on the GPU, and
written to its final place. Each task is idempotent and logged on completion, so the run can be
killed and restarted at any point.

    python -m deepinfer_mac.convert.remote --out ~/models/dsv41-mac
"""

import argparse
import json
import os
import queue
import re
import shutil
import threading
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from .. import format as F
from .. import quant as Q
from . import hf

INDEX = Path(__file__).with_name("tensor_index.json")
EXPERT_CHUNK = 8  # experts per download task (~100-150 MB)
ENGRAM_CHUNK = 1 << 20  # rows per download task (~264 MB)
MAX_SPAN = 256 << 20
MAX_GAP = 1 << 20


# ------------------------------------------------------------------------------ fetching


def plan_spans(metas: list[dict]) -> list[tuple[str, int, int]]:
    """Merge tensors that sit close together in the same shard into single range requests."""
    spans = []
    for m in sorted(metas, key=lambda m: (m["shard"], m["start"])):
        if spans and spans[-1][0] == m["shard"] and m["start"] - spans[-1][2] <= MAX_GAP and m["end"] - spans[-1][1] <= MAX_SPAN:
            spans[-1][2] = max(spans[-1][2], m["end"])
        else:
            spans.append([m["shard"], m["start"], m["end"]])
    return [tuple(s) for s in spans]


def fetch(named: dict[str, dict]) -> dict[str, np.ndarray]:
    """{name: meta} -> {name: array}, reading merged spans."""
    out = {}
    for shard, a, b in plan_spans(list(named.values())):
        buf = hf.get_range(shard, a, b)
        for name, m in named.items():
            if m["shard"] == shard and a <= m["start"] and m["end"] <= b:
                out[name] = hf.as_array(buf[m["start"] - a : m["end"] - a], m)
    assert len(out) == len(named), set(named) - set(out)
    return out


def fetch_rows(m: dict, r0: int, r1: int) -> np.ndarray:
    """Rows [r0, r1) of a 2D tensor (the engram tables are single 98 GB tensors)."""
    row = m["end"] - m["start"]
    row //= m["shape"][0]
    buf = hf.get_range(m["shard"], m["start"] + r0 * row, m["start"] + r1 * row)
    return np.frombuffer(buf, dtype=hf.DTYPES[m["dtype"]]).reshape(r1 - r0, *m["shape"][1:])


# ------------------------------------------------------------------------------ converting


def to_mx(name: str, a: np.ndarray, dtype: str) -> mx.array:
    if dtype == "BF16":
        return mx.array(a.view(np.uint16)).view(mx.bfloat16)
    return mx.array(a)


def convert_dense(arrays: dict[str, np.ndarray], metas: dict[str, dict], strip: str) -> dict[str, mx.array]:
    """fp8 Linears -> mxfp8 (bit-exact); head -> affine 8-bit; everything else as shipped."""
    out = {}
    for name, a in arrays.items():
        m = metas[name]
        key = name[len(strip) :]
        if name.endswith(".scale") and name[: -len(".scale")] + ".weight" in arrays:
            continue  # consumed with its weight
        if m["dtype"] == "F8_E4M3":
            scale = arrays[name[: -len(".weight")] + ".scale"]
            q, s = Q.fp8_block_to_mxfp8(a, scale)
            base = key[: -len(".weight")]
            out[base + ".weight"], out[base + ".scales"] = mx.array(q), mx.array(s)
        elif key == "head.weight":
            q, s, b = mx.quantize(to_mx(name, a, m["dtype"]), group_size=F.GROUP, bits=8)
            out["head.weight"], out["head.scales"], out["head.biases"] = q, s, b
        else:
            out[key] = to_mx(name, a, m["dtype"])
    return out


def convert_expert(w: dict[str, np.ndarray], layout: F.ExpertLayout) -> bytes:
    buf = bytearray(layout.stride)
    bits = {"w1": layout.bits_w1, "w3": layout.bits_w3, "w2": layout.bits_w2}
    parts = {}
    for k in ("w1", "w3", "w2"):
        deq = Q.dequant_fp4(mx.array(w[k + ".weight"]), mx.array(w[k + ".scale"]))
        q, s, b = Q.quantize_mse(deq, bits[k])
        mx.eval(q, s, b)
        parts[k + ".q"], parts[k + ".s"], parts[k + ".b"] = q, s, b
    for name, shape, dtype, off, nbytes in layout.parts():
        a = parts[name]
        assert tuple(a.shape) == shape and str(a.dtype).endswith(dtype), (name, a.shape, a.dtype, shape, dtype)
        raw = np.array(a.view(mx.uint16) if dtype == "bfloat16" else a).tobytes()
        assert len(raw) == nbytes
        buf[off : off + nbytes] = raw
    return bytes(buf)


def convert_engram_rows(codes: np.ndarray, scale: np.ndarray) -> bytes:
    """[R, 256] e4m3 + [R, 8] e8m0 -> [R, 144] (4-bit affine g64 + bf16 scales + bf16 biases)."""
    deq = Q.dequant_fp8_rowscale(mx.array(codes), mx.array(scale)).astype(mx.bfloat16)
    q, s, b = mx.quantize(deq, group_size=F.GROUP, bits=F.ENGRAM_BITS)
    rows = mx.concatenate([q.view(mx.uint8), s.view(mx.uint8), b.view(mx.uint8)], axis=1)
    assert rows.shape[1] == F.ENGRAM_ROW
    return np.array(rows).tobytes()


# ------------------------------------------------------------------------------ tasks


class Converter:
    def __init__(self, out: Path, workers: int = 4):
        self.out = out
        self.idx = json.load(open(INDEX))
        self.inf = json.loads(self._small("inference/config.json"))
        self.workers = workers
        self.log = out / "progress.log"
        self.done = set(self.log.read_text().split()) if self.log.exists() else set()
        self.lock = threading.Lock()
        self.bytes_in = 0
        self.t0 = time.time()

    def _small(self, path: str) -> bytes:
        import requests

        r = requests.get(hf.url(path), timeout=60)
        r.raise_for_status()
        return r.content

    # -- task list

    def tasks(self) -> list[tuple[str, str, dict]]:
        by = defaultdict(dict)
        for name, m in self.idx.items():
            if (mm := re.match(r"(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.", name)):
                kind = "XL" if mm[1] == "layers" else "XM"
                by[(kind, int(mm[2]), int(mm[3]) // EXPERT_CHUNK)][name] = m
            elif ".engram.embed." in name:
                continue
            elif (mm := re.match(r"(layers|mtp)\.(\d+)\.", name)):
                by[("L" if mm[1] == "layers" else "M", int(mm[2]))][name] = m
            elif name.startswith(("vision.", "aligner.", "image_")):
                by[("vision",)][name] = m
            elif name == "embed.weight":
                by[("embed",)][name] = m
            else:
                by[("globals",)][name] = m
        order = [("globals",)] + sorted(k for k in by if k[0] == "L") + sorted(k for k in by if k[0] == "M")
        order += [("embed",), ("vision",)]
        order += sorted(k for k in by if k[0] == "XL") + sorted(k for k in by if k[0] == "XM")
        tasks = [("-".join(map(str, k)), k[0], by[k]) for k in order]
        for li, layer in enumerate(self.inf["engram_layer_ids"]):
            w = self.idx[f"layers.{layer}.engram.embed.weight"]
            s = self.idx[f"layers.{layer}.engram.embed.scale"]
            rows = w["shape"][0]
            for c in range(-(-rows // ENGRAM_CHUNK)):
                r0, r1 = c * ENGRAM_CHUNK, min(rows, (c + 1) * ENGRAM_CHUNK)
                tasks.append((f"EG-{layer}-{c}", "EG", {"w": w, "s": s, "layer": layer, "r0": r0, "r1": r1, "rows": rows}))
        return tasks

    # -- download side (threads)

    def download(self, kind: str, payload: dict):
        if kind == "EG":
            return fetch_rows(payload["w"], payload["r0"], payload["r1"]), fetch_rows(payload["s"], payload["r0"], payload["r1"])
        if kind == "embed":
            m = payload["embed.weight"]
            rows, out = m["shape"][0], []
            step = 8192
            for r0 in range(0, rows, step):
                out.append(fetch_rows(m, r0, min(rows, r0 + step)))
            return {"embed.weight": np.concatenate(out)}
        return fetch(payload)

    # -- convert/write side (main thread: MLX)

    def _file(self, path: Path, size: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        if os.fstat(fd).st_size < size:
            os.ftruncate(fd, size)
        return fd

    def write(self, tid: str, kind: str, payload: dict, data):
        out = self.out
        if kind in ("globals", "L", "M"):
            strip = "" if kind == "globals" else re.match(r"(layers|mtp)\.\d+\.", next(iter(payload)))[0]
            arrays = convert_dense(data, payload, strip)
            name = "globals" if kind == "globals" else f"L{int(tid.split('-')[1]):02d}" if kind == "L" else f"M{tid.split('-')[1]}"
            (out / "dense").mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(out / "dense" / f"{name}.safetensors"), arrays)
        elif kind == "embed":
            a = data["embed.weight"]
            (out / "embed.bf16").write_bytes(np.ascontiguousarray(a).tobytes())
        elif kind == "vision":
            mx.save_safetensors(str(out / "vision.safetensors"), {k: to_mx(k, a, payload[k]["dtype"]) for k, a in data.items()})
        elif kind in ("XL", "XM"):
            _, layer, _ = tid.split("-")
            layer = int(layer)
            is_mtp = kind == "XM"
            layout = F.expert_layout_for(layer, is_mtp=is_mtp)
            n_exp = self.inf["dspark_n_routed_experts"] if is_mtp else self.inf["n_routed_experts"]
            path = out / "experts" / (f"M{layer}.bin" if is_mtp else f"L{layer:02d}.bin")
            fd = self._file(path, n_exp * layout.stride)
            try:
                experts = defaultdict(dict)
                for name, a in data.items():
                    mm = re.search(r"experts\.(\d+)\.(w\d\.(?:weight|scale))$", name)
                    experts[int(mm[1])][mm[2]] = a
                for e, w in sorted(experts.items()):
                    os.pwrite(fd, convert_expert(w, layout), e * layout.stride)
                os.fsync(fd)
            finally:
                os.close(fd)
        elif kind == "EG":
            codes, scale = data
            path = out / "engram" / f"L{payload['layer']:02d}.bin"
            fd = self._file(path, payload["rows"] * F.ENGRAM_ROW)
            try:
                os.pwrite(fd, convert_engram_rows(codes, scale), payload["r0"] * F.ENGRAM_ROW)
                os.fsync(fd)
            finally:
                os.close(fd)
        mx.clear_cache()
        with open(self.log, "a") as f:
            f.write(tid + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_meta(self):
        out = self.out
        out.mkdir(parents=True, exist_ok=True)
        for p in ("config.json", "tokenizer.json", "tokenizer_config.json", "inference/config.json", "encoding/encoding.py"):
            dst = out / ("inference_config.json" if p == "inference/config.json" else Path(p).name)
            if not dst.exists():
                dst.write_bytes(self._small(p))
        layout = {
            "format": "deepinfer-mac/1",
            "group_size": F.GROUP,
            "dense": "mxfp8 (bit-exact from fp8 32x32 blocks); head affine 8-bit",
            "experts": {f"L{i:02d}": F.expert_layout_for(i).to_json() for i in range(self.inf["n_layers"])}
            | {f"M{i}": F.expert_layout_for(i, is_mtp=True).to_json() for i in range(self.inf["n_mtp_layers"])},
            "engram": {"bits": F.ENGRAM_BITS, "row_bytes": F.ENGRAM_ROW, "layers": self.inf["engram_layer_ids"]},
        }
        (out / "layout.json").write_text(json.dumps(layout, indent=1))

    def run(self, only: str | None = None, limit: int | None = None):
        self.write_meta()
        todo = [t for t in self.tasks() if t[0] not in self.done and (only is None or re.match(only, t[0]))]
        if limit:
            todo = todo[:limit]
        print(f"[convert] {len(self.done)} done, {len(todo)} to go", flush=True)
        jobs: queue.Queue = queue.Queue()
        for t in todo:
            jobs.put(t)
        results: dict[str, object] = {}
        cond = threading.Condition()

        def worker():
            while True:
                with cond:  # bound read-ahead when the head-of-line task is slow
                    cond.wait_for(lambda: len(results) < 2 * self.workers)
                try:
                    tid, kind, payload = jobs.get_nowait()
                except queue.Empty:
                    return
                data = self.download(kind, payload)
                n = sum(a.nbytes for a in (data.values() if isinstance(data, dict) else data))
                with cond:
                    self.bytes_in += n
                    results[tid] = data
                    cond.notify_all()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(self.workers)]
        for th in threads:
            th.start()
        # convert strictly in task order so the log stays a prefix-friendly record, but let
        # downloads run ahead by up to `workers` tasks
        for i, (tid, kind, payload) in enumerate(todo):
            with cond:
                cond.wait_for(lambda: tid in results)
                data = results.pop(tid)
                cond.notify_all()
            self.write(tid, kind, payload, data)
            dt = time.time() - self.t0
            free = shutil.disk_usage(self.out).free / 1e9
            print(f"[convert] {i + 1}/{len(todo)} {tid}  {self.bytes_in / 1e9:.1f} GB in, "
                  f"{self.bytes_in / dt / 1e6:.1f} MB/s, disk free {free:.0f} GB", flush=True)
            if free < 15:
                raise SystemExit("stopping: less than 15 GB free on disk")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", help="regex over task ids, e.g. 'globals|L-3|XL-3-0'")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    Converter(a.out.expanduser(), a.workers).run(a.only, a.limit)


if __name__ == "__main__":
    main()
