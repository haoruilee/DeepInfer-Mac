"""Indexer Top-K: a total order (score desc, position asc), independent of the partition kernel."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlx.core as mx  # noqa: E402

from deepinfer_mac.model.model import select_candidate_blocks, topk_positions  # noqa: E402


def reference_topk(score: np.ndarray, k: int) -> list[set]:
    """Stable sort by descending score: ties keep the earlier position."""
    return [set(np.argsort(-row, kind="stable")[:k].tolist()) for row in score]


def bf16(x: np.ndarray) -> np.ndarray:
    return np.array(mx.array(x).astype(mx.bfloat16).astype(mx.float32))


def test_ties_prefer_earlier_positions():
    rng = np.random.default_rng(0)
    for T, k in ((64, 7), (1000, 512), (4096, 512)):
        # few distinct values -> many ties straddle the k-th place; plus -inf / +-0 / negatives
        s = bf16(rng.integers(-4, 5, size=(9, T)).astype(np.float32) * 0.375)
        s[:, ::5] = -np.inf
        s[0, :] = 0.0
        s[1, ::2] = -0.0
        got = np.array(topk_positions(mx.array(s), k))
        assert [set(r.tolist()) for r in got] == reference_topk(s, k)


def test_random_scores_match_value_order():
    rng = np.random.default_rng(1)
    s = bf16(rng.standard_normal((16, 3000)).astype(np.float32) * 50)
    got = np.array(topk_positions(mx.array(s), 512))
    assert [set(r.tolist()) for r in got] == reference_topk(s, 512)


def test_candidate_blocks_ties():
    # equal block maxima: the earlier blocks win, the query's own (last) block is always kept
    T, block = 64, 8
    s = np.zeros((1, T), dtype=np.float32)
    keep = np.array(select_candidate_blocks(mx.array(s), mx.array([T]), 3, block))[0].reshape(-1, block)[:, 0]
    assert keep.tolist() == [True, True, False, False, False, False, False, True]


def test_index_head_sum_matches_elementwise():
    from deepinfer_mac.model.ops import index_head_sum

    bf = mx.bfloat16
    for R, H, T in ((1, 32, 1), (3, 32, 1000), (17, 8, 4099)):
        s = (mx.random.normal((R, H, T), key=mx.random.key(T)) * 7).astype(bf)
        w = mx.random.normal((R, H), key=mx.random.key(H)).astype(bf)
        ref = (mx.maximum(s, 0) * w[..., None]).astype(bf).astype(mx.float32).sum(axis=1).astype(bf)
        assert bool(mx.all(index_head_sum(s, w) == ref))
        sv = s[:, :, ::2]  # non-contiguous input
        ref = (mx.maximum(sv, 0) * w[..., None]).astype(bf).astype(mx.float32).sum(axis=1).astype(bf)
        assert bool(mx.all(index_head_sum(sv, w) == ref))
