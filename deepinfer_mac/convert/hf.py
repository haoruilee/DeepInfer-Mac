"""Range-read tensors straight out of the safetensors shards on the Hugging Face Hub."""

import json
import struct
import threading
import time

import numpy as np
import requests

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
ENDPOINT = "https://huggingface.co"

DTYPES = {
    "BF16": np.uint16, "F32": np.float32, "F16": np.float16, "I8": np.uint8,
    "F8_E4M3": np.uint8, "F8_E8M0": np.uint8, "I32": np.int32, "I64": np.int64,
}

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def url(path: str, repo: str = REPO) -> str:
    return f"{ENDPOINT}/{repo}/resolve/main/{path}"


PIECE = 32 << 20  # one HTTP request; long single transfers occasionally stall on the CDN
MIN_RATE = 100 << 10  # bytes/s: only truly stuck transfers; a slow line must not abort every piece (livelock)


def _get_piece(path: str, start: int, end: int) -> bytes:
    r = _session().get(url(path), headers={"Range": f"bytes={start}-{end - 1}"}, timeout=30, stream=True)
    r.raise_for_status()
    buf, t0 = bytearray(), time.time()
    for chunk in r.iter_content(1 << 20):
        buf += chunk
        dt = time.time() - t0
        if dt > 30 and len(buf) / dt < MIN_RATE:
            r.close()
            raise IOError(f"stalled at {len(buf) / dt / 1e6:.2f} MB/s")
    if len(buf) != end - start:
        raise IOError(f"short read {len(buf)} != {end - start}")
    return bytes(buf)


def get_range(path: str, start: int, end: int, retries: int = 50) -> bytes:
    """Bytes [start, end) of a repo file, in PIECE-sized requests, each retried with backoff."""
    out = []
    for a in range(start, end, PIECE):
        b = min(end, a + PIECE)
        for attempt in range(retries):
            try:
                out.append(_get_piece(path, a, b))
                break
            except Exception as e:  # noqa: BLE001 - network: anything retries
                if attempt == retries - 1:
                    raise
                print(f"[hf] retry {attempt + 1} {path} [{a},{b}): {e}", flush=True)
                _local.__dict__.pop("s", None)  # new connection: a stalled one tends to stay on a slow CDN node
                time.sleep(min(30, 2**attempt))
    return b"".join(out)


def fetch_headers(shards: list[str]) -> dict:
    """{tensor: {dtype, shape, shard, start, end}} with absolute file offsets."""
    out = {}
    for s in shards:
        n = struct.unpack("<Q", get_range(s, 0, 8))[0]
        h = json.loads(get_range(s, 8, 8 + n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            a, b = v.pop("data_offsets")
            out[k] = {**v, "shard": s, "start": 8 + n + a, "end": 8 + n + b}
    return out


def as_array(buf: bytes, meta: dict) -> np.ndarray:
    return np.frombuffer(buf, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"])
