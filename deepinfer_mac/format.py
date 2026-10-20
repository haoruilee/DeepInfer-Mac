"""On-disk layout of a converted checkpoint, shared by the converter and the runtime.

<root>/
  layout.json                 everything below, as written by the converter
  config.json, tokenizer*.json
  dense/globals.safetensors   norm, head (affine 8-bit), markov head, ...
  dense/L{nn}.safetensors     one per backbone layer: attention, shared expert, gate, mHC, engram wkv
  dense/M{n}.safetensors      one per DSpark (mtp) layer
  embed.bf16                  raw [vocab, dim] bf16, mmap'd -- only looked-up rows are read
  experts/L{nn}.bin           fixed-stride records, one per routed expert (see ExpertLayout)
  experts/M{n}.bin
  engram/L{nn}.bin            fixed 144-byte rows: 4-bit affine g64 weights + bf16 scales + biases
  vision.safetensors          ViT + aligner, bf16 as shipped
"""

from dataclasses import asdict, dataclass

ALIGN = 16384  # page size on Apple Silicon; records start page-aligned for F_NOCACHE preads
GROUP = 64

DIM = 5120
INTER = 2304
ENGRAM_DIM = 256


def affine_sizes(out: int, inn: int, bits: int, group: int = GROUP) -> tuple[int, int]:
    """Bytes of (packed weights, scales) -- biases are the same size as scales (bf16 each)."""
    return out * inn * bits // 8, out * (inn // group) * 2


@dataclass(frozen=True)
class ExpertLayout:
    """One routed expert = w1 | w3 | w2, each as (q, scales, biases), packed back to back."""

    bits_w1: int
    bits_w3: int
    bits_w2: int
    dim: int = DIM
    inter: int = INTER

    def parts(self) -> list[tuple[str, tuple[int, ...], str, int, int]]:
        """(name, shape, dtype, offset, nbytes) for each array inside a record."""
        out, off = [], 0
        for w, (o, i), bits in (
            ("w1", (self.inter, self.dim), self.bits_w1),
            ("w3", (self.inter, self.dim), self.bits_w3),
            ("w2", (self.dim, self.inter), self.bits_w2),
        ):
            qb, sb = affine_sizes(o, i, bits)
            out.append((f"{w}.q", (o, i * bits // 32), "uint32", off, qb))
            off += qb
            out.append((f"{w}.s", (o, i // GROUP), "bfloat16", off, sb))
            off += sb
            out.append((f"{w}.b", (o, i // GROUP), "bfloat16", off, sb))
            off += sb
        return out

    @property
    def nbytes(self) -> int:
        _, _, _, off, n = self.parts()[-1]
        return off + n

    @property
    def stride(self) -> int:
        return -(-self.nbytes // ALIGN) * ALIGN

    def to_json(self) -> dict:
        return {**asdict(self), "stride": self.stride, "nbytes": self.nbytes}


ENGRAM_BITS = 4
ENGRAM_GROUP = 32  # g32 + clipping search (other V4.1 ports: plain g64 4-bit costs +7.3% divergence)
ENGRAM_ROW = ENGRAM_DIM * ENGRAM_BITS // 8 + 2 * (ENGRAM_DIM // ENGRAM_GROUP) * 2  # 160


# Index-source layers (their indexer picks what every following layer attends to) and the late
# layers carry the most weight for output quality (rapid-mlx's per-layer KL study; a 2-bit index layer
# cut MTP acceptance 84% -> 58% in another port), the last layer most of all.
HIGH_LAYERS = {8, 14, 20, 24, 28, 32, 35, 36, 37, 38}


def expert_layout_for(layer: int, n_layers: int = 40, is_mtp: bool = False) -> ExpertLayout:
    """Mixed 2/3/4-bit build. w2 (down proj) is always >= 3-bit. Layers 0-3 are 3-bit for
    historical reasons (converted first; the converter keeps a started layer's layout)."""
    if is_mtp:
        return ExpertLayout(2, 2, 3)
    if layer == n_layers - 1:
        return ExpertLayout(4, 4, 4)
    if layer < 4 or layer in HIGH_LAYERS:
        return ExpertLayout(3, 3, 3)
    return ExpertLayout(2, 2, 3)
