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
    """Every byte of every requested record, across all slab parts, equals the file."""
    slab, slots = s.get_slots(key, ids)
    for e, sl in zip(ids, slots):
        for name, shape, dtype, off, n in lay.parts():
            got = np.array(slab.parts[name][sl]).view(np.uint8).reshape(-1)[:n]
            assert np.array_equal(got, data[e, off : off + n]), (e, sl, name)


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


def test_split_reads_and_page_cache_give_identical_contents(big_store):
    from deepinfer_mac.store import ExpertStore

    s, data, lay = big_store
    saved = (ExpertStore.IO_SPLIT, ExpertStore.PAGE_CACHE)
    try:
        ExpertStore.IO_SPLIT, ExpertStore.PAGE_CACHE = 4, True
        s.fds.clear()
        for i in range(60):
            _check(s, data, lay, "L00", [(i * 11 + j) % 384 for j in range(6)])
    finally:
        ExpertStore.IO_SPLIT, ExpertStore.PAGE_CACHE = saved


def test_speculation_is_cold_and_never_evicts_used_experts(big_store):
    """Idle-window speculation: guesses enter at the cold LRU end, so filling the slab with wrong
    guesses evicts guesses first and never an expert a request used; contents stay correct."""
    s, data, lay = big_store
    slab = s.slabs["L00"]
    used = list(range(len(slab.free) + len(slab.lru) - 20))  # nearly fill the slab with used experts
    for i in range(0, len(used), 6):
        _check(s, data, lay, "L00", used[i : i + 6])
    for i in range(30):  # many more wrong guesses than free slots
        s.speculate("L00", [300 + (i * 3 + j) % 80 for j in range(3)], after=[])
        s.spec_pool.submit(lambda: None).result()  # let the speculation thread run
        for f in list(s.inflight.values()):
            f.result()
    resident = {e for (_, e) in slab.lru}
    assert set(used) <= resident, "a guess evicted an expert that was used"
    _check(s, data, lay, "L00", [300, 301, 302] + used[:3])  # guesses that hit are correct data


def test_speculation_waits_for_demand_reads(big_store):
    s, data, lay = big_store
    order = []
    import concurrent.futures as cf
    gate = cf.Future()
    gate.add_done_callback(lambda _: order.append("demand"))
    s.speculate("L00", [10, 11], after=[gate])
    s.prefetch("L00", [1])  # demand traffic proceeds meanwhile
    assert ("L00", 10) not in s.slabs["L00"].lru  # nothing speculative before the demand read lands
    gate.set_result(None)
    s.spec_pool.submit(lambda: None).result()
    assert ("L00", 10) in s.slabs["L00"].lru and order == ["demand"]
    _check(s, data, lay, "L00", [10, 11, 1])
