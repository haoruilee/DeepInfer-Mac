"""Vision parity against the official reference (inference/vision.py + image_processor.py) on a tiny
random model: ViT + aligner output, and a full multimodal forward (span merge, VL routing bias,
image tokens excluded from Engram)."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_parity as T  # noqa: E402

torch = T.torch
import mlx.core as mx  # noqa: E402

VISION = dict(
    vision_n_layers=2, vision_dim=64, vision_n_heads=4, vision_inter_dim=96, vision_patch_size=14,
    vision_downsample_ratio=3, vision_max_n_token=40, vision_min_pixels=42 * 42, vision_rope_theta=10000.0,
    image_token_id=129264, index_topk=64, candidate_topk_blocks=64,
)


def _png(w=100, h=70):
    from PIL import Image

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def models():
    saved = T._no_quant(True)
    rm, mm = T.build(**VISION)
    g = torch.Generator().manual_seed(1)
    for name, mod in rm.named_modules():  # vision modules are plain torch layers: init them too
        if isinstance(mod, torch.nn.Linear):
            mod.weight.data.copy_((torch.randn(mod.weight.shape, generator=g) / mod.weight.shape[1] ** 0.5).to(mod.weight.dtype))
            if mod.bias is not None:
                mod.bias.data.copy_((0.02 * torch.randn(mod.bias.shape, generator=g)).to(mod.bias.dtype))
    for name, p in rm.named_parameters():
        if name.startswith("vision.") and name.endswith("norm1.weight") or name.endswith("norm2.weight") or name == "vision.norm.weight":
            p.data.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
        if name in ("image_start", "image_end", "image_newline"):
            p.data.copy_(torch.randn(p.shape, generator=g).to(p.dtype))
        if name.endswith("gate.bias_vl"):
            p.data.copy_(0.3 * torch.randn(p.shape, generator=g))
    sd = rm.state_dict()
    vis = {k: T._mx(v) for k, v in sd.items() if k.startswith(("vision.", "aligner.", "image_"))}
    mm.w.vision = lambda: vis
    for i, blk in enumerate(mm.layers):  # reload gate biases (bias_vl was re-initialized above)
        blk.ffn.bias_vl = T._mx(sd[f"layers.{i}.ffn.gate.bias_vl"]).astype(mx.float32)
    yield rm, mm
    T._restore(saved)


def test_vision_tower(models):
    import image_processor as ref_ip

    rm, mm = models
    args = rm_args = T.tiny_args(**VISION)
    patches, vh, vw, lh, lw = ref_ip.load_image({"data": _png()}, args)
    with torch.inference_mode():
        ref = rm.encode_image(patches, vh, vw).float().numpy()
    from deepinfer_mac.model.vision import ImageInput, image_token_types, load_image

    p2, vh2, vw2, lh2, lw2 = load_image({"data": _png()}, mm.a)
    assert (vh, vw, lh, lw) == (vh2, vw2, lh2, lw2)
    assert np.allclose(patches.float().numpy(), p2, atol=1e-6)
    ours = np.array(mm.vision_tower().encode(ImageInput(0, p2, vh2, vw2, image_token_types(lh2, lw2))).astype(mx.float32))
    cos, rel, _ = T._cmp(ref, ours)
    print(f"vision tower: {ref.shape} cos={cos:.6f} rel={rel:.4f}")
    assert ref.shape == ours.shape and cos > 0.9995  # bf16 accumulation (both sides); fp32 on both sides gives 1.0000000


def test_multimodal_forward(models):
    import image_processor as ref_ip

    rm, mm = models
    args = T.tiny_args(**VISION)
    text = np.random.default_rng(2).integers(100, 120000, size=10).tolist()
    prompt = text[:4] + [args.image_token_id] + text[4:]
    from deepinfer_mac.model.vision import expand_images

    toks, types, imgs = expand_images(prompt, [{"data": _png()}], mm.a)
    patches, vh, vw, lh, lw = ref_ip.load_image({"data": _png()}, args)
    ref_img = ref_ip.ImageInput(imgs[0].start, patches, vh, vw, torch.tensor(imgs[0].types))
    with torch.inference_mode():
        _, rl, _ = rm(torch.tensor([toks]), 0, images=[[ref_img]], token_types=torch.tensor([types]))
    ml = mm.forward(mm.new_state(), toks, images=imgs, token_types=types)
    cos, rel, top = T._cmp(rl.float().numpy(), np.array(ml))
    print(f"multimodal forward ({len(toks)} tokens, {int((types >= 0).sum())} image): cos={cos:.6f} top1={top}")
    assert cos > 0.9999 and top
