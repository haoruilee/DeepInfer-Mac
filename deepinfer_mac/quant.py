"""Codecs between the DeepSeek checkpoint formats and MLX quantized layouts.

Checkpoint formats (see the official inference/convert.py):
  * fp8 Linear: e4m3 bytes [out, in] + one e8m0 scale per 32x32 block [out/32, in/32]
  * fp4 expert: e2m1 nibbles packed two per byte, low nibble first [out, in/2] + one e8m0
    scale per 32 elements along `in` [out, in/32]
  * engram table: e4m3 bytes [rows, 256] + e8m0 per 32 along the row [rows, 8]

MLX layouts:
  * mxfp8 (group 32): uint32 words of 4 e4m3 codes (low byte first) + uint8 e8m0 per group.
    A 32x32-block fp8 weight is exactly this with the block scale repeated over its 32 rows,
    so dense weights repack bit-exactly.
  * mxfp4 (group 32): uint32 words of 8 e2m1 codes (low nibble first) + uint8 e8m0 -- the same
    bytes as the checkpoint experts.
  * affine (bits 2..8, group 64): what lossy expert / engram builds use.
"""

import mlx.core as mx
import numpy as np

FP4_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32
)


def _e4m3_table() -> np.ndarray:
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 0xF, b & 0x7
        if e == 0xF and m == 0x7:
            v = np.nan
        elif e == 0:
            v = m / 8 * 2.0**-6
        else:
            v = (1 + m / 8) * 2.0 ** (e - 7)
        t[b] = -v if s else v
    return t


E4M3_TABLE = _e4m3_table()
_FP4_MX = mx.array(FP4_TABLE)
_E4M3_MX = mx.array(np.nan_to_num(E4M3_TABLE))


def e8m0_to_float(s: mx.array) -> mx.array:
    """e8m0 byte -> 2**(s-127) as float32."""
    return mx.power(mx.array(2.0, mx.float32), s.astype(mx.float32) - 127.0)


# ---------------------------------------------------------------- dequantize checkpoint formats


def dequant_fp4(packed: mx.array, scale: mx.array) -> mx.array:
    """[out, in/2] uint8 + [out, in/32] e8m0 -> [out, in] float32."""
    packed = packed.astype(mx.uint8)
    lo = _FP4_MX[(packed & 0x0F).astype(mx.uint32)]
    hi = _FP4_MX[(packed >> 4).astype(mx.uint32)]
    out, half = packed.shape
    w = mx.stack([lo, hi], axis=-1).reshape(out, half * 2)
    return (w.reshape(out, -1, 32) * e8m0_to_float(scale)[..., None]).reshape(out, half * 2)


def dequant_fp8_rowscale(codes: mx.array, scale: mx.array, block: int = 32) -> mx.array:
    """[rows, d] e4m3 + [rows, d/block] e8m0 -> float32 (engram tables)."""
    v = _E4M3_MX[codes.astype(mx.uint32)]
    rows, d = codes.shape
    return (v.reshape(rows, -1, block) * e8m0_to_float(scale)[..., None]).reshape(rows, d)


def dequant_fp8_block(codes: mx.array, scale: mx.array, block: int = 32) -> mx.array:
    """[out, in] e4m3 + [out/block, in/block] e8m0 -> float32."""
    v = _E4M3_MX[codes.astype(mx.uint32)]
    out, inn = codes.shape
    s = mx.repeat(mx.repeat(e8m0_to_float(scale), block, axis=0)[:out], block, axis=1)[:, :inn]
    return v * s


# ---------------------------------------------------------------- bit-exact repacks


def fp8_block_to_mxfp8(codes: np.ndarray, scale: np.ndarray, block: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Lossless: [out, in] e4m3 bytes + [out/32, in/32] e8m0 -> mxfp8 (uint32 [out, in/4], uint8 [out, in/32])."""
    out, inn = codes.shape
    assert inn % 32 == 0
    words = np.ascontiguousarray(codes.view(np.uint8)).view(np.uint32).reshape(out, inn // 4)
    rows = np.repeat(scale.view(np.uint8), block, axis=0)[:out]
    rows = np.repeat(rows, block // 32, axis=1)[:, : inn // 32]
    return words, np.ascontiguousarray(rows)


def fp4_to_mxfp4(packed: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lossless: [out, in/2] e2m1 nibbles + [out, in/32] e8m0 -> mxfp4 (uint32 [out, in/8], uint8)."""
    out, half = packed.shape
    words = np.ascontiguousarray(packed.view(np.uint8)).view(np.uint32).reshape(out, half // 4)
    return words, np.ascontiguousarray(scale.view(np.uint8))


# ---------------------------------------------------------------- lossy requantization


def quantize_mse(w: mx.array, bits: int, group_size: int = 64, ratios=(1.0, 0.9, 0.8, 0.7, 0.6)):
    """MLX affine quantization with a per-group clipping search.

    For each candidate ratio every group is shrunk symmetrically about its midpoint before
    quantizing; per group we keep whichever candidate dequantizes with the lowest squared error.
    Groups are independent in MLX's packing (a group of 64 is 2*bits whole uint32 words), so the
    winners can be stitched together without repacking.
    """
    out, inn = w.shape
    ng = inn // group_size
    wg = w.reshape(out, ng, group_size)
    lo, hi = wg.min(axis=-1, keepdims=True), wg.max(axis=-1, keepdims=True)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    best = None
    for r in ratios:
        wc = mx.clip(wg, mid - half * r, mid + half * r).reshape(out, inn).astype(mx.bfloat16)
        q, s, b = mx.quantize(wc, group_size=group_size, bits=bits)
        err = mx.square(mx.dequantize(q, s, b, group_size=group_size, bits=bits).astype(mx.float32).reshape(out, ng, group_size) - wg).sum(-1)
        if best is None:
            best = [q.reshape(out, ng, -1), s, b, err]
            continue
        better = err < best[3]
        best[0] = mx.where(better[..., None], q.reshape(out, ng, -1), best[0])
        best[1] = mx.where(better, s, best[1])
        best[2] = mx.where(better, b, best[2])
        best[3] = mx.minimum(err, best[3])
        mx.eval(best)
    return best[0].reshape(out, -1), best[1], best[2]
