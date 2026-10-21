"""Range-read tensors straight out of the safetensors shards on the Hugging Face Hub."""

import json
import struct
import threading
import time

import numpy as np
import requests

ENDPOINT = "https://huggingface.co"
REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"

DTYPES = {
    "BF16": np.uint16, "F32": np.float32, "F16": np.float16, "I8": np.uint8,
    "F8_E4M3": np.uint8, "F8_E8M0": np.uint8, "I32": np.int32, "I64": np.int64,
}

_local = threading.local()

# --- direct routing for China-hosted mirrors -------------------------------------------------
# A system-wide proxy in TUN mode (e.g. Clash) captures every socket and answers DNS with fake
# IPs. ModelScope is reachable directly from mainland China at full line speed, so for its hosts
# we resolve through a public resolver and bind the socket to the physical interface: that
# traffic bypasses the proxy (and its metered node). Everything else is left untouched.
DIRECT_SUFFIXES = (".modelscope.cn",)
DIRECT_DNS = "223.5.5.5"
_dns_cache: dict[str, tuple[str, float]] = {}


def _physical_interface() -> str | None:
    import subprocess

    try:
        out = subprocess.run(["route", "-n", "get", "default"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "interface:" in line:
                name = line.split()[-1]
                return None if name.startswith("utun") else name
    except Exception:  # noqa: BLE001
        pass
    return None


def _resolve_direct(host: str) -> str:
    import subprocess

    hit = _dns_cache.get(host)
    if hit and time.time() - hit[1] < 300:
        return hit[0]
    out = subprocess.run(["dig", "+short", "+time=3", f"@{DIRECT_DNS}", host, "A"], capture_output=True, text=True, timeout=10).stdout
    ips = [l for l in out.split() if l.count(".") == 3 and l.replace(".", "").isdigit()]
    if not ips:
        raise OSError(f"direct DNS failed for {host}")
    _dns_cache[host] = (ips[0], time.time())
    return ips[0]


def _install_direct_routing():
    import socket

    import urllib3.util.connection as uconn

    iface = _physical_interface()
    if iface is None:
        return
    ifindex = socket.if_nametoindex(iface)
    orig = uconn.create_connection

    def create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, socket_options=None):
        host, port = address
        if not host.endswith(DIRECT_SUFFIXES):
            return orig(address, timeout, source_address, socket_options)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_IP, 25, ifindex)  # IP_BOUND_IF: leave via the physical NIC
        for opt in socket_options or []:
            sock.setsockopt(*opt)
        if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
            sock.settimeout(timeout)
        sock.connect((_resolve_direct(host), port))
        return sock

    uconn.create_connection = create_connection


_install_direct_routing()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


# Sources, in preference order. ModelScope hosts byte-identical shards (same size and sha256 for
# all 48) on a CDN in China; from there it delivers full line speed without a proxy, where the
# Hugging Face CDN can drop under 1 MB/s.
ENDPOINTS = ["https://www.modelscope.cn", "https://huggingface.co"]


def url(path: str, repo: str = REPO, endpoint: str = ENDPOINT) -> str:
    if "modelscope" in endpoint:
        return f"{endpoint}/models/{repo}/resolve/master/{path}"
    return f"{endpoint}/{repo}/resolve/main/{path}"


PIECE = 32 << 20  # one HTTP request
MIN_RATE = 100 << 10  # bytes/s: only truly stuck transfers (a slow line must not abort every piece)


def _stream_into(buf: bytearray, path: str, start: int, end: int, endpoint: str) -> None:
    """Append bytes [start, end) to buf; raises on a stall, keeping what arrived."""
    r = _session().get(url(path, endpoint=endpoint), headers={"Range": f"bytes={start}-{end - 1}"}, timeout=30, stream=True)
    r.raise_for_status()
    t0, n0 = time.time(), len(buf)
    for chunk in r.iter_content(1 << 20):
        buf += chunk
        dt = time.time() - t0
        if dt > 30 and (len(buf) - n0) / dt < MIN_RATE:
            r.close()
            raise IOError(f"stalled at {(len(buf) - n0) / dt / 1e6:.2f} MB/s")


def get_range(path: str, start: int, end: int, retries: int = 200) -> bytes:
    """Bytes [start, end) of a repo file in PIECE-sized requests. A failed or stalled request
    resumes from the last byte received (nothing is downloaded twice), on a fresh connection and
    alternating endpoints."""
    out = bytearray()
    for a in range(start, end, PIECE):
        b = min(end, a + PIECE)
        piece = bytearray()
        for attempt in range(retries):
            if len(piece) == b - a:
                break  # complete, even if the request that delivered the last bytes then failed
            try:
                _stream_into(piece, path, a + len(piece), b, ENDPOINTS[1] if attempt % 3 == 2 else ENDPOINTS[0])  # fallback every third retry
                if len(piece) != b - a:
                    raise IOError(f"short read {len(piece)} != {b - a}")
                break
            except Exception as e:  # noqa: BLE001 - network: anything retries
                if len(piece) > b - a:
                    piece = bytearray()  # a server ignored the range: start the piece over
                if attempt == retries - 1:
                    raise
                print(f"[hf] retry {attempt + 1} {path} [{a + len(piece)},{b}): {str(e)[:120]}", flush=True)
                _local.__dict__.pop("s", None)  # new connection
                time.sleep(min(20, 1 + attempt))
        out += piece
    return bytes(out)


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
