"""Write the tiny parity model to disk in the converted layout, through the converter's own
functions, so the SSD paths (expert store, engram table, embed memmap) run end to end."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from deepinfer_mac import format as F
from deepinfer_mac.convert.remote import convert_dense, convert_engram_rows
from deepinfer_mac.model.model import ExpertWeights

DT = {torch.float8_e4m3fn: "F8_E4M3", torch.float8_e8m0fnu: "F8_E8M0", torch.bfloat16: "BF16", torch.float32: "F32", torch.float4_e2m1fn_x2: "I8"}


def write(rm, args, out: Path, tokenizer: Path, bits=(4, 4, 4)):
    import test_parity as T
    from deepinfer_mac import quant as Q

    out.mkdir(parents=True, exist_ok=True)
    (out / "dense").mkdir(exist_ok=True)
    (out / "experts").mkdir(exist_ok=True)
    (out / "engram").mkdir(exist_ok=True)
    sd = rm.state_dict()
    arrays = {k: T._np(v) for k, v in sd.items()}
    metas = {k: {"dtype": DT[v.dtype]} for k, v in sd.items()}
    glob = {k: arrays[k] for k in ("norm.weight", "head.weight")}
    mx.save_safetensors(str(out / "dense" / "globals.safetensors"), convert_dense(glob, metas, ""))
    for i in range(args.n_layers):
        p = f"layers.{i}."
        sel = {k: a for k, a in arrays.items() if k.startswith(p) and ".experts." not in k and ".engram.embed." not in k}
        mx.save_safetensors(str(out / "dense" / f"L{i:02d}.safetensors"), convert_dense(sel, metas, p))
        lay = F.ExpertLayout(*bits, dim=args.dim, inter=args.moe_inter_dim)
        with open(out / "experts" / f"L{i:02d}.bin", "wb") as f:
            for e in range(args.n_routed_experts):
                rec = bytearray(lay.stride)
                parts = {}
                for w in ("w1", "w3", "w2"):
                    deq = Q.dequant_fp4(mx.array(arrays[f"{p}ffn.experts.{e}.{w}.weight"]), mx.array(arrays[f"{p}ffn.experts.{e}.{w}.scale"]))
                    parts[w + ".q"], parts[w + ".s"], parts[w + ".b"] = mx.quantize(deq.astype(mx.bfloat16), group_size=64, bits=lay.bits_w1)
                for name, shape, dtype, off, n in lay.parts():
                    a = parts[name]
                    rec[off : off + n] = np.array(a.view(mx.uint16) if dtype == "bfloat16" else a).tobytes()
                f.write(rec)
    for lid in args.engram_layer_ids:
        rows = convert_engram_rows(arrays[f"layers.{lid}.engram.embed.weight"], arrays[f"layers.{lid}.engram.embed.scale"])
        (out / "engram" / f"L{lid:02d}.bin").write_bytes(rows)
    (out / "embed.bf16").write_bytes(np.ascontiguousarray(arrays["embed.weight"]).tobytes())
    cfg = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(args).items()}
    (out / "inference_config.json").write_text(json.dumps(cfg))
    lay = F.ExpertLayout(*bits, dim=args.dim, inter=args.moe_inter_dim)
    (out / "layout.json").write_text(json.dumps({
        "experts": {f"L{i:02d}": lay.to_json() for i in range(args.n_layers)},
        "engram": {"bits": 4, "group": 32, "dim": args.engram_head_dim, "layers": list(args.engram_layer_ids)},
    }))
    shutil.copy(tokenizer, out / "tokenizer.json")
    return out
