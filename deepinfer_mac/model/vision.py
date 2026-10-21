"""Vision tower of DeepSeek-V4.1-Flash in MLX: image preprocessing, ViT, aligner.

A port of the official inference/vision.py and inference/image_processor.py. An image becomes an
n_vit_h x n_vit_w grid of 14-px patches for the ViT (full bidirectional attention, 2D RoPE) and,
after the aligner's 3x3 downsample, an n_llm_h x n_llm_w token grid that the LLM sees as

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

Every position of that span carries `image_token_id`; only the token type tells them apart.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from urllib.request import urlopen

import mlx.core as mx
import numpy as np

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


@dataclass
class ImageInput:
    start: int  # position of IMAGE_START in the prompt
    patches: np.ndarray  # [n_vit_h * n_vit_w, 3, p, p] float32 in [-1, 1]
    n_vit_h: int
    n_vit_w: int
    types: np.ndarray  # token types of the span


# ------------------------------------------------------------------------------ preprocessing


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height, best_width, patch, ratio):
    return math.ceil((best_height // patch) / ratio), math.ceil((best_width // patch) / ratio)


def solve_resize_ratio(height, width, patch, ratio, max_n_token):
    r = height / width
    max_w = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h = max_w * r
    cell = patch * ratio
    if max_w < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
    return math.floor(height * beta / patch) * patch, math.floor(width * beta / patch) * patch


def plan_image_grid(width: int, height: int, a):
    p = a.vision_patch_size
    if a.vision_max_wh_ratio is not None and width > height * a.vision_max_wh_ratio:
        width = height * a.vision_max_wh_ratio
    if 0 < width * height < a.vision_min_pixels:
        ratio = (a.vision_min_pixels / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    best_w, best_h = math.ceil(width / p) * p, math.ceil(height / p) * p
    n_llm_h, n_llm_w = llm_grid(best_h, best_w, p, a.vision_downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > a.vision_max_n_token:
        best_h, best_w = solve_resize_ratio(height, width, p, a.vision_downsample_ratio, a.vision_max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_h, best_w, p, a.vision_downsample_ratio)
    return n_llm_h, n_llm_w, best_h, best_w


def load_image_bytes(record) -> bytes:
    """Raw/base64 bytes, an Anthropic `source`, a data/http URL, or a path."""
    if isinstance(record, (bytes, bytearray)):
        return bytes(record)
    if isinstance(record, str):
        record = {"url": record}
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)
    src = record.get("source")
    if isinstance(src, dict):
        if src.get("data") is not None:
            return base64.b64decode(src["data"])
        if src.get("url"):
            return load_image_bytes({"url": src["url"]})
    url = record.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            return base64.b64decode(payload)
        if url.startswith(("http://", "https://")):
            with urlopen(url, timeout=30) as r:
                return r.read()
        with open(url, "rb") as f:
            return f.read()
    raise ValueError(f"cannot load image from {record!r:.80}")


def load_image(record, a):
    """-> (patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w), as image_processor.load_image."""
    from PIL import Image, ImageOps

    p = a.vision_patch_size
    with Image.open(io.BytesIO(load_image_bytes(record))) as src:
        image = src.convert("RGB")
    n_llm_h, n_llm_w, best_h, best_w = plan_image_grid(image.width, image.height, a)
    n_vit_h, n_vit_w = best_h // p, best_w // p
    if a.vision_max_wh_ratio is not None and image.width >= a.vision_max_wh_ratio * image.height:
        image = image.resize((best_w, best_h))
    else:
        image = ImageOps.pad(image, (best_w, best_h), color=(127, 127, 127))
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
    x = (x - 0.5) / 0.5
    x = mx.array(x).astype(mx.bfloat16).astype(mx.float32)  # the reference rounds pixels to bf16
    patches = np.array(x).reshape(3, n_vit_h, p, n_vit_w, p).transpose(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> np.ndarray:
    return np.array([IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END], dtype=np.int64)


def expand_images(tokens: list[int], images: list, a) -> tuple[list[int], np.ndarray, list[ImageInput]]:
    """Expand each image placeholder token into its span (prepare_vl_inputs)."""
    n = sum(t == a.image_token_id for t in tokens)
    if n != len(images):
        raise ValueError(f"prompt has {n} image placeholders but {len(images)} images were given")
    out, types, inputs, it = [], [], [], iter(images)
    for t in tokens:
        if t != a.image_token_id:
            out.append(t)
            types.append(TEXT)
            continue
        patches, vh, vw, lh, lw = load_image(next(it), a)
        ty = image_token_types(lh, lw)
        inputs.append(ImageInput(len(out), patches, vh, vw, ty))
        out += [a.image_token_id] * len(ty)
        types += ty.tolist()
    return out, np.array(types, dtype=np.int64), inputs


# ------------------------------------------------------------------------------ ViT + aligner


def _lin(d, name, x):
    y = x @ d[name + ".weight"].T
    b = d.get(name + ".bias")
    return y if b is None else y + b


def _rms(x, w, eps=1e-6):
    xf = x.astype(mx.float32)
    return (w.astype(mx.float32) * (xf * mx.rsqrt(mx.square(xf).mean(-1, keepdims=True) + eps))).astype(x.dtype)


def vision_cos_sin(n_h, n_w, dim, theta):
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    hpos = np.repeat(np.arange(n_h)[:, None], n_w, axis=1)
    wpos = np.repeat(np.arange(n_w)[None, :], n_h, axis=0)
    freqs = (np.stack([hpos, wpos], axis=-1).reshape(-1, 2, 1).astype(np.float32) * inv).reshape(n_h * n_w, -1)
    return mx.array(np.cos(freqs))[:, None, :], mx.array(np.sin(freqs))[:, None, :]


def _rotary(x, cos, sin):
    dtype = x.dtype
    x1, x2 = mx.split(x.astype(mx.float32), 2, axis=-1)
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(dtype)


class VisionTower:
    def __init__(self, a, d: dict):
        """d: vision.* / aligner.* / image_* tensors (bf16) with their checkpoint names."""
        self.a, self.d = a, d
        self.n_heads = a.vision_n_heads
        self.head_dim = a.vision_dim // a.vision_n_heads
        self.rope_dim = self.head_dim // 2

    def encode(self, img: ImageInput) -> mx.array:
        """-> aligner rows [n_llm_h * n_llm_w, dim] in reading order."""
        a, d = self.a, self.d
        x = mx.array(img.patches.reshape(img.patches.shape[0], -1)).astype(mx.bfloat16)
        x = _lin(d, "vision.patch_embed.proj", x)
        cos, sin = vision_cos_sin(img.n_vit_h, img.n_vit_w, self.rope_dim, a.vision_rope_theta)
        n = x.shape[0]
        for i in range(a.vision_n_layers):
            p = f"vision.blocks.{i}."
            h = _rms(x, d[p + "norm1.weight"])
            q, k, v = mx.split(_lin(d, p + "attn.wqkv", h).reshape(n, 3, self.n_heads, self.head_dim), 3, axis=1)
            q, k, v = (t[:, 0] for t in (q, k, v))
            q, k = _rotary(q, cos, sin), _rotary(k, cos, sin)
            o = mx.fast.scaled_dot_product_attention(
                q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None], v.transpose(1, 0, 2)[None], scale=self.head_dim**-0.5
            )[0]
            x = x + _lin(d, p + "attn.wo", o.transpose(1, 0, 2).reshape(n, -1))
            h = _rms(x, d[p + "norm2.weight"])
            gate, up = mx.split(_lin(d, p + "mlp.w1", h), 2, axis=-1)
            x = x + _lin(d, p + "mlp.w2", (gate * mx.sigmoid(gate)) * up)
        x = _rms(x, d["vision.norm.weight"])
        return self._align(x, img.n_vit_h, img.n_vit_w)

    def _align(self, x, n_h, n_w):
        r = self.a.vision_downsample_ratio
        c = x.shape[-1]
        x = x.reshape(n_h, n_w, c)
        ph, pw = -n_h % r, -n_w % r
        if ph or pw:
            x = mx.pad(x, [(0, ph), (0, pw), (0, 0)])
        H, W = x.shape[0] // r, x.shape[1] // r
        # F.unfold(kernel=r, stride=r) on [C, H, W]: each output row is (C, kh, kw) flattened
        x = x.reshape(H, r, W, r, c).transpose(0, 2, 4, 1, 3).reshape(H * W, c * r * r)
        h = _lin(self.d, "aligner.w1", x)
        h = 0.5 * h * (1 + mx.erf(h / math.sqrt(2)))  # exact gelu, as F.gelu
        return _lin(self.d, "aligner.w2", h)

    def merge(self, h: mx.array, images: list[ImageInput]) -> mx.array:
        """Write each image span into the token embeddings h [L, dim]."""
        rows = []
        for img in images:
            span = np.zeros((len(img.types), h.shape[-1]), dtype=np.float32)
            emb = self.encode(img).astype(mx.float32)
            span = mx.array(span)
            ty = mx.array(img.types)
            span = mx.where((ty == IMAGE_START)[:, None], self.d["image_start"].astype(mx.float32), span)
            span = mx.where((ty == IMAGE_END)[:, None], self.d["image_end"].astype(mx.float32), span)
            span = mx.where((ty == IMAGE_NEW_LINE)[:, None], self.d["image_newline"].astype(mx.float32), span)
            pos = np.nonzero(img.types == IMAGE)[0]
            span[mx.array(pos)] = emb
            rows.append((img.start, span.astype(h.dtype)))
        for start, span in rows:
            h = mx.concatenate([h[:start], span, h[start + span.shape[0]:]])
        return h
