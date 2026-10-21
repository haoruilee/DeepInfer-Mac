"""ExpertStore slab/LRU invariants on a synthetic expert file."""

import json

import numpy as np
import pytest

from deepinfer_mac import format as F
from deepinfer_mac.store import ExpertStore


def _make(tmp_path, n):
    lay = F.ExpertLayout(2, 2, 3, dim=128, inter=64)
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(n, lay.stride), dtype=np.uint8)
    for e in range(n):
        data[e, :8] = np.frombuffer(np.int64(e).tobytes(), np.uint8)  # expert id in its first bytes
    (tmp_path / "experts").mkdir()
    data.tofile(tmp_path / "experts" / "L00.bin")
    (tmp_path / "layout.json").write_text(json.dumps({"experts": {"L00": lay.to_json()}}))
    s = ExpertStore(tmp_path, budget_bytes=0)  # minimum slab (96 slots)
    return s, data, lay


@pytest.fixture()
def store(tmp_path):
    return _make(tmp_path, 64)


@pytest.fixture()
def big_store(tmp_path):
    return _make(tmp_path, 384)  # a real layer's expert count, 4x the minimum slab


def _check(s, data, lay, key, ids):
    slab, slots = s.get_slots(key, ids)
    first = lay.parts()[0]
    q = np.array(slab.parts[first[0]])
    for e, sl in zip(ids, slots):
        got = q[sl].view(np.uint8).reshape(-1)[: first[4]]
        assert np.array_equal(got, data[e, : first[4]]), (e, sl)


def test_contents_and_eviction(store):
    s, data, lay = store
    rng = np.random.default_rng(1)
    for step in range(300):  # far more distinct experts over time than slots
        ids = rng.choice(64, 6, replace=False).tolist()
        _check(s, data, lay, "L00", ids)
    # a request mixing many residents with misses must not evict its own residents
    big = list(range(64))
    _check(s, data, lay, "L00", big[:40])
    _check(s, data, lay, "L00", big[20:64])
    assert 0 < s.hit_rate() < 1


def test_optional_prefetch_never_breaks_requests(store):
    s, data, lay = store
    for i in range(50):
        s.prefetch("L00", [(i * 7 + j) % 64 for j in range(30)], optional=True)
        _check(s, data, lay, "L00", [(i * 5 + j) % 64 for j in range(6)])


def test_prefill_batches_fit_minimum_slab(big_store):
    """Expert-major prefill: batches of EXPERT_BATCH with the next batch prefetched (optional),
    as MoE._slab_moe does, over every expert -- must never exhaust a minimum-size slab."""
    from deepinfer_mac.model.model import MoE

    s, data, lay = big_store
    B = MoE.EXPERT_BATCH
    for rep in range(3):
        ids = list(range(384))
        batches = [ids[i : i + B] for i in range(0, len(ids), B)]
        for bi, batch in enumerate(batches):
            if bi + 1 < len(batches):
                s.prefetch("L00", batches[bi + 1], optional=True)
            _check(s, data, lay, "L00", batch)


def test_unrequested_lookahead_reads_do_not_leak(big_store):
    """Lookahead mispredictions (optional prefetches never requested) must become evictable once
    their reads finish, or they slowly eat the slab."""
    import time

    s, data, lay = big_store
    for i in range(200):
        s.prefetch("L00", [(i * 13 + j) % 384 for j in range(6)], optional=True)  # never requested
        time.sleep(0.001)
        _check(s, data, lay, "L00", [(i * 7 + j) % 384 for j in range(6)])
