"""GPU byte and lifetime contracts for the selected immutable-file backend.

Set B12X_DISK_BACKEND=gds and place pytest's basetemp on a GDS filesystem to
qualify cuFile. A selected but unavailable transport fails instead of skipping.
"""

from __future__ import annotations

import os
import random
import threading
import time

import numpy as np
import pytest
import torch

from b12x.sequence._shared.disk_table import DiskRowCache
from tests.experimental.b12x.conftest import require_b12x


def _cache(**kwargs):
    queue_depth = kwargs.pop("queue_depth", 2)
    return DiskRowCache(
        device=require_b12x(),
        max_lookups=40,
        table_rows=300,
        shard_start=7,
        shard_end=290,
        shard_rows=100,
        weight_row_bytes=257,
        scale_row_bytes=9,
        queue_depth=queue_depth,
        **kwargs,
    )


def _write(path, payload, offset):
    with path.open("wb", buffering=0) as stream:
        stream.seek(offset)
        stream.write(payload)
        os.fsync(stream.fileno())


@torch.inference_mode()
@pytest.mark.parametrize("queue_depth", [2, 3])
def test_rows_match_bytes_across_planes_shards_streams_and_frozen_gather(
    tmp_path, monkeypatch, queue_depth
):
    cache = _cache(queue_depth=queue_depth)
    payloads = []
    try:
        for scale, width in [(False, 257), (True, 9)]:
            data = (
                (
                    torch.arange(300 * width, dtype=torch.int64) * 17
                    + torch.arange(300).repeat_interleave(width)
                )
                .remainder(251)
                .to(torch.uint8)
                .reshape(300, width)
            )
            payloads.append(data)
            for shard in range(3):
                path = tmp_path / f"{scale}-{shard}.bin"
                _write(
                    path, data[shard * 100 : (shard + 1) * 100].numpy().tobytes(), 4093
                )
                cache.add_shard(shard, str(path), 4093, scale=scale)
        cache.freeze()
        if cache._gds:
            from b12x.sequence._shared._gds import _gather

            monkeypatch.setattr(
                _gather,
                "run",
                lambda *a, **k: pytest.fail(
                    "live counts must reuse the compiled gather"
                ),
            )
            cache._gds.native.start_stats()
        pointers = [cache.weight.data_ptr(), cache.scale.data_ptr()]
        footprint = cache.stats()["owned_staging_bytes"]
        streams = [torch.cuda.Stream(device=cache.device) for _ in range(2)]
        ids = torch.tensor(
            [
                7,
                7,
                99,
                100,
                199,
                200,
                289,
                290,
                -1,
                6,
                299,
                300,
                2**40,
                15,
                16,
                31,
                32,
                47,
                48,
                79,
                80,
                127,
                128,
                191,
                192,
                255,
                256,
            ],
            device=cache.device,
        )
        out = torch.empty_like(cache.weight)
        for iteration, count in enumerate([27, 1, 0, 13, 27, 5]):
            stream = streams[iteration % 2]
            with torch.cuda.stream(stream), cache.transaction():
                cache.read_rows(ids, count)
                out[:count].copy_(cache.weight[:count])
                for actual, source in [(out, payloads[0]), (cache.scale, payloads[1])]:
                    host_ids = ids[:count].cpu()
                    expected = torch.zeros((count, source.shape[1]), dtype=torch.uint8)
                    valid = (host_ids >= 7) & (host_ids < 290)
                    expected[valid] = source[host_ids[valid]]
                    torch.testing.assert_close(
                        actual[:count].cpu(), expected, rtol=0, atol=0
                    )
            assert pointers == [cache.weight.data_ptr(), cache.scale.data_ptr()]
            assert cache.stats()["owned_staging_bytes"] == footprint
        if cache._gds:
            assert cache.weight_host is cache.scale_host is None
            stats = cache._gds.native.transport_stats()
            assert stats["nvfs_ops"] + stats["p2p_ops"] > 0
            assert all(
                stats[k] == 0
                for k in ("posix_ops", "aio_ops", "iouring_ops", "read_errors")
            )
    finally:
        cache.close()
    cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        with cache.transaction():
            pass


@torch.inference_mode()
def test_file_offsets_and_global_ids_exceed_signed_32bit_products(tmp_path):
    device = require_b12x()
    row = (1 << 32) // 256 + 3
    offset = 4093
    path = tmp_path / "large-offset.bin"
    data = torch.arange(256, dtype=torch.uint8).numpy().tobytes()
    _write(path, data, offset + row * 256)
    cache = DiskRowCache(
        device=device,
        max_lookups=2,
        table_rows=row + 1,
        shard_start=row,
        shard_end=row + 1,
        shard_rows=row + 1,
        weight_row_bytes=256,
        queue_depth=1,
    )
    try:
        cache.add_shard(0, str(path), offset)
        cache.freeze()
        with cache.transaction():
            cache.read_rows(torch.tensor([row, row - 1], device=device), 2)
            torch.testing.assert_close(
                cache.weight[0].cpu(),
                torch.arange(256, dtype=torch.uint8),
                rtol=0,
                atol=0,
            )
            assert cache.weight[1].count_nonzero().item() == 0
    finally:
        cache.close()


def test_source_and_read_failures_are_reported_before_reuse(tmp_path):
    device = require_b12x()
    path = tmp_path / "source.bin"
    _write(path, bytes(range(256)) * 64, 0)
    cache = DiskRowCache(
        device=device,
        max_lookups=2,
        table_rows=64,
        shard_start=0,
        shard_end=64,
        shard_rows=64,
        weight_row_bytes=256,
        queue_depth=2,
    )
    try:
        with pytest.raises(RuntimeError, match="open"):
            cache.add_shard(0, str(tmp_path / "missing"), 0)
        cache.add_shard(0, str(path), 0)
        cache.freeze()
        # Violating the immutable-file contract must report a short read.
        with path.open("r+b") as stream:
            stream.truncate(4096)
        with pytest.raises(RuntimeError, match="short|failed|unexpected"):
            with cache.transaction():
                cache.read_rows(torch.tensor([31, 63], device=device), 2)
        assert cache._transaction_thread is None
    finally:
        cache.close()


def test_backend_is_fixed_at_construction_and_optional_dependency_is_lazy(monkeypatch):
    from b12x.loader import _gds_native

    def unavailable():
        raise RuntimeError("injected unavailable cuFile development files")

    monkeypatch.setattr(_gds_native, "load", unavailable)
    monkeypatch.delenv("B12X_DISK_BACKEND", raising=False)
    cache = _cache()
    try:
        assert cache._backend == "io_uring"
        monkeypatch.setenv("B12X_DISK_BACKEND", "gds")
        assert cache._backend == "io_uring"
        with pytest.raises(RuntimeError, match="unavailable cuFile"):
            _cache()
        monkeypatch.setenv("B12X_DISK_BACKEND", "invalid")
        with pytest.raises(ValueError, match="B12X_DISK_BACKEND"):
            _cache()
    finally:
        cache.close()


def test_close_waits_for_a_consumer_on_another_stream(tmp_path):
    device = require_b12x()
    path = tmp_path / "pending.bin"
    _write(path, bytes(range(256)) * 16, 0)
    cache = DiskRowCache(
        device=device,
        max_lookups=1,
        table_rows=16,
        shard_start=0,
        shard_end=16,
        shard_rows=16,
        weight_row_bytes=256,
        queue_depth=1,
    )
    cache.add_shard(0, str(path), 0)
    cache.freeze()
    stream = torch.cuda.Stream(device=device)
    result = torch.empty((256,), dtype=torch.uint8, device=device)
    ids = torch.tensor([1], device=device)
    with torch.cuda.stream(stream), cache.transaction():
        cache.read_rows(ids, 1)
        torch.cuda._sleep(10_000_000)
        result.copy_(cache.weight[0])
    cache.close()
    assert stream.query()
    torch.testing.assert_close(
        result.cpu(), torch.arange(256, dtype=torch.uint8), rtol=0, atol=0
    )


def test_full_batch_uses_registered_slots_above_one_mib(tmp_path):
    device = require_b12x()
    path = tmp_path / "batch-slots.bin"
    data = ((torch.arange(4096)[:, None] * 17 + torch.arange(256)) % 251).to(
        torch.uint8
    )
    _write(path, data.numpy().tobytes(), 0)
    cache = DiskRowCache(
        device=device,
        max_lookups=64,
        table_rows=4096,
        shard_start=0,
        shard_end=4096,
        shard_rows=4096,
        weight_row_bytes=256,
        queue_depth=64,
    )
    try:
        cache.add_shard(0, str(path), 0)
        cache.freeze()
        for count in [64, 1, 63]:
            ids = torch.arange(count, device=device) * 32
            with cache.transaction():
                cache.read_rows(ids, count)
                torch.testing.assert_close(
                    cache.weight[:count].cpu(), data[ids.cpu()], rtol=0, atol=0
                )
            assert cache.stats()["read_calls"] == count
    finally:
        cache.close()


def _payloads():
    return [
        (
            torch.arange(300 * width, dtype=torch.int64) * 17
            + torch.arange(300).repeat_interleave(width)
        )
        .remainder(251)
        .to(torch.uint8)
        .reshape(300, width)
        for width in (257, 9)
    ]


def _expected(source, ids):
    expected = torch.zeros((len(ids), source.shape[1]), dtype=torch.uint8)
    valid = (ids >= 7) & (ids < 290)
    expected[valid] = source[ids[valid]]
    return expected


def _byte_identity_round_trip(tmp_path, monkeypatch, policy, tier, capacity, source):
    """Serve one batch pattern through ``policy`` and check every staged byte.

    Cached staging must equal the source for repeats, in-batch duplicates,
    invalid IDs, and capacities that evict rows the same batch still reads.
    Both replacement policies run this identical fixture, so choosing a policy
    cannot change what the read path returns.
    """
    monkeypatch.setenv("B12X_DISK_BACKEND", "io_uring")
    payloads = _payloads()
    host_rows = None
    if source == "host":
        host_rows = tuple(p[7:290].contiguous() for p in payloads)
    cache = _cache(
        cache_rows=capacity, cache_tier=tier, host_rows=host_rows, policy=policy
    )
    try:
        if source == "disk":
            for scale, data in zip((False, True), payloads):
                for shard in range(3):
                    path = tmp_path / f"{scale}-{shard}.bin"
                    _write(
                        path, data[shard * 100 : (shard + 1) * 100].numpy().tobytes(), 0
                    )
                    cache.add_shard(shard, str(path), 0, scale=scale)
        cache.freeze()
        assert cache.stats()["row_cache_policy"] == policy
        assert (cache.weight.device.type == "cuda") and (
            (cache.weight_host is None) == (tier == "device")
        )
        generator = torch.Generator().manual_seed(capacity)
        hot = torch.tensor([7, 8, 150, 151, 289, 6, 290, 2**40])
        for count in [40, 40, 1, 0, 17, 40, 40]:
            ids = torch.where(
                torch.rand(count, generator=generator) < 0.6,
                hot[torch.randint(len(hot), (count,), generator=generator)],
                torch.randint(0, 300, (count,), generator=generator),
            )
            with cache.transaction():
                cache.read_rows(ids.to(cache.device), count)
                for actual, data in zip((cache.weight, cache.scale), payloads):
                    torch.testing.assert_close(
                        actual[:count].cpu(), _expected(data, ids), rtol=0, atol=0
                    )
        stats = cache.stats()
        assert stats["row_cache_tier"] == tier
        assert 0 < stats["row_cache_resident"] <= capacity
        assert stats["row_cache_hits"] > 0
    finally:
        cache.close()


@torch.inference_mode()
@pytest.mark.parametrize("tier", ["host", "device"])
@pytest.mark.parametrize("capacity", [1, 5, 64, 400])
@pytest.mark.parametrize("source", ["disk", "host"])
def test_sieve_cached_rows_match_uncached_bytes(
    tmp_path, monkeypatch, tier, capacity, source
):
    """The default policy is still SIEVE, byte for byte as before the choice."""
    _byte_identity_round_trip(tmp_path, monkeypatch, "sieve", tier, capacity, source)


@torch.inference_mode()
@pytest.mark.parametrize("policy", ["sieve", "s3fifo"])
@pytest.mark.parametrize("tier", ["host", "device"])
@pytest.mark.parametrize("capacity", [1, 5, 64, 400])
@pytest.mark.parametrize("source", ["disk", "host"])
def test_s3fifo_cached_rows_match_uncached_bytes(
    tmp_path, monkeypatch, policy, tier, capacity, source
):
    """S3-FIFO is a drop-in sibling: same fixture, same bytes.

    Eviction order differs, so which row is still resident when a batch reads
    it differs; what must not change is that a resident row holds the source's
    bytes and that a victim never clobbers a row the same batch reads.
    """
    _byte_identity_round_trip(tmp_path, monkeypatch, policy, tier, capacity, source)


def test_sieve_keeps_visited_rows_over_newer_one_hit_rows():
    from b12x.sequence._shared.row_cache import SieveIndex

    index = SieveIndex(3)
    index.plan([1, 2, 3])
    index.plan([1])  # 1 visited
    index.plan([4])  # evicts oldest unvisited (2); 1's bit is cleared on the sweep
    assert 1 in index and 2 not in index and 3 in index and 4 in index
    index.plan([5])  # hand continues from 3: 3 unvisited -> evicted
    assert 1 in index and 3 not in index and 4 in index and 5 in index
    index.plan([1, 1, 6])  # duplicates count once per position, fetch once
    assert index.hits == 1 + 2 and 6 in index and 1 in index


def test_s3fifo_small_queue_promotes_hits_and_ghosts_re_admit():
    """Hand-checked S3-FIFO sequence over a 10-row cache (S 1, M 9, G 1)."""
    from b12x.sequence._shared.row_cache import S3FifoIndex

    index = S3FifoIndex(10)
    assert index._small_cap == 1 and index._main_cap == 9 and index._ghost_cap == 1

    # Every new row joins S. Each is read once before the next insert recycles
    # it, so it is promoted into M with a reset counter instead of dropped: S
    # holds only the row that has not been recycled yet.
    for key in range(1, 10):
        index.insert(key)
        index.get(key)
    assert [index._key[slot] for slot in index._small] == [9]
    assert [index._key[slot] for slot in index._main] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert index._freq[index._slot[9]] == 1
    assert all(index._freq[index._slot[key]] == 0 for key in range(1, 9))
    assert not index._ghost

    # A row re-read while in M refills its counter, and the next S row promotes
    # beside it rather than displacing anything: the pool still had a slot.
    index.get(1)
    index.insert(20)
    assert [index._key[slot] for slot in index._main] == list(range(1, 10))
    assert [index._key[slot] for slot in index._small] == [20]
    assert index._freq[index._slot[1]] == 1
    assert len(index) == 10

    # 20 is never read, so recycling it from S demotes it to a ghost: the key
    # stays, the payload slot goes.
    index.insert(21)
    assert 20 not in index
    assert list(index._ghost) == [20]
    assert [index._key[slot] for slot in index._small] == [21]

    # A miss on a ghost is admitted straight into M, skipping S entirely.
    # Making room pops M's front: 1 still holds the credit its re-read earned,
    # so it is reinserted at the tail and 2 -- out of credit -- is the row that
    # leaves. A plain FIFO would have evicted 1 and kept 2.
    index.insert(20)
    assert index._slot[20] not in index._small
    assert index._queue[index._slot[20]] == S3FifoIndex._MAIN
    assert 20 not in index._ghost
    assert [index._key[slot] for slot in index._main] == [3, 4, 5, 6, 7, 8, 9, 1, 20]
    assert index._freq[index._slot[1]] == 0
    assert 1 in index and 2 not in index
    assert len(index) == 10


def test_s3fifo_churn_keeps_every_slot_owned_by_one_row():
    """Randomised access over both queues: budgets hold, no slot gains a second
    owner, and a batch reads its hits before its own stores land."""
    from b12x.sequence._shared.row_cache import S3FifoIndex

    index = S3FifoIndex(64)
    # What DiskRowCache would have written into each pool row, so a hit slot
    # quietly reassigned within the batch fails the read below rather than
    # serving another row's bytes.
    pool: dict[int, int] = {}
    rng = random.Random(1234)
    hot = [7, 8, 150, 151, 289, 6, 290, 2**40]
    for batch in range(200):
        ids = [
            rng.choice(hot) if rng.random() < 0.6 else rng.randrange(300)
            for _ in range(rng.choice([1, 5, 17, 40]))
        ]
        plan = index.plan(ids)
        for position, slot in zip(plan.hit_pos, plan.hit_slot):
            assert pool.get(slot) == ids[position]
        for slot, row in zip(plan.store_slot, plan.store_row):
            pool[slot] = plan.miss_ids[row]
        assert len(index._small) <= index._small_cap
        assert len(index._main) <= index._main_cap
        assert len(index._ghost) <= index._ghost_cap
        assert not set(index._small) & set(index._main)
        assert set(index._slot.values()) == set(index._small) | set(index._main)
        assert len(set(index._slot.values())) == len(index._slot)
        assert (
            len(index._free) + len(set(index._slot.values())) + len(index._loading)
            == index._used
        )
        for key, slot in index._slot.items():
            assert index._key[slot] == key
            assert index._state[slot] == S3FifoIndex._READY
            assert index._freq[slot] <= 3
        for key, slot in index._loading.items():
            # A prefetch in flight is invisible and out of every pop's reach.
            assert key not in index
            assert index._state[slot] == S3FifoIndex._LOADING
            assert slot not in index._small and slot not in index._main
        if batch % 7 == 3:
            key = 1000 + batch
            slot = index.reserve(key)
            if slot is not None:
                assert index.get(key) is None and not index.is_ready(key)
                if batch % 3 == 0:
                    index.cancel(slot)
                    assert slot in index._free
                else:
                    pool[slot] = key
                    assert index.confirm(slot) == slot
                    assert index.is_ready(key)


def test_s3fifo_policy_selected_by_env_or_argument(monkeypatch):
    """Env picks the policy, the argument beats it, and an unknown name fails.

    Surrounding space and case are normalised because operators set this from
    a shell or a manifest; an unrecognised name is a construction error rather
    than a silent fallback to the default.
    """
    from b12x.sequence._shared.row_cache import S3FifoIndex, SieveIndex

    host_rows = _host_rows(_payloads())

    def build(**kwargs):
        return _cache(cache_rows=8, cache_tier="host", host_rows=host_rows, **kwargs)

    monkeypatch.delenv("B12X_ROW_CACHE_POLICY", raising=False)
    default = build()
    try:
        assert type(default._index) is SieveIndex
        assert default.stats()["row_cache_policy"] == "sieve"
    finally:
        default.close()

    monkeypatch.setenv("B12X_ROW_CACHE_POLICY", "  S3Fifo ")
    selected = build()
    try:
        assert type(selected._index) is S3FifoIndex
        assert selected.stats()["row_cache_policy"] == "s3fifo"
    finally:
        selected.close()
    forced = build(policy="SIEVE")
    try:
        assert type(forced._index) is SieveIndex
        assert forced.stats()["row_cache_policy"] == "sieve"
    finally:
        forced.close()
    with pytest.raises(ValueError, match="B12X_ROW_CACHE_POLICY"):
        build(policy="lru")
    monkeypatch.setenv("B12X_ROW_CACHE_POLICY", "lru")
    with pytest.raises(ValueError, match="B12X_ROW_CACHE_POLICY"):
        build()

    # A disabled cache reports no policy, exactly as it reports no tier.
    uncached = _cache(cache_rows=0, cache_tier="host", host_rows=host_rows)
    try:
        assert uncached._index is None
        assert uncached.row_cache_policy is None
        assert "row_cache_policy" not in uncached.stats()
    finally:
        uncached.close()


def test_cache_requires_io_uring_or_host_source(monkeypatch):
    monkeypatch.setenv("B12X_DISK_BACKEND", "gds")
    with pytest.raises(ValueError, match="GDS backend does not support a row cache"):
        _cache(cache_rows=8, cache_tier="device")


def _host_rows(payloads):
    """CPU uint8 host-resident shard tables over the cache's ``[7, 290)`` range."""
    return tuple(p[7:290].contiguous() for p in payloads)


def _fake_fetch(payloads, block=None, fail_first=0, record=None):
    """Deterministic stand-in for the source read, independent of disk/native.

    It writes ``payloads[plane][gid]`` for in-shard ids and zeros otherwise, so
    prefetched and synchronously fetched bytes are byte-identical for a given
    id (last write wins). ``block`` holds the worker inside the fetch so a test
    can observe loading slots; ``fail_first`` raises for the leading calls to
    exercise the cancel path; ``record`` collects the raw call arguments.
    """
    weight_src, scale_src = payloads[0].numpy(), payloads[1].numpy()
    state = {"calls": 0}

    def fetch(self, ids, weights, scales, count):
        state["calls"] += 1
        if record is not None:
            record.append((ids, weights, scales, count))
        if block is not None:
            block.wait(timeout=5)
        if state["calls"] <= fail_first:
            raise IOError("injected prefetch read failure")
        id_arr = np.frombuffer(ids, dtype=np.int64, count=count)
        for buffer, source in ((weights, weight_src), (scales, scale_src)):
            if buffer is None:
                continue
            width = source.shape[1]
            out = np.frombuffer(buffer, dtype=np.uint8, count=count * width).reshape(
                count, width
            )
            out.flags.writeable = True
            for row, gid in enumerate(id_arr):
                gid = int(gid)
                out[row] = source[gid] if 7 <= gid < 290 else 0

    return fetch


def _drain(cache, ids, timeout=3.0):
    index = cache._index
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cache._prefetch_queue.empty() and all(index.is_ready(gid) for gid in ids):
            return
        time.sleep(0.005)
    raise AssertionError(f"prefetch did not drain: {ids}")


def _wait_loading(cache, ids, timeout=3.0):
    index = cache._index
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cache._prefetch_queue.empty() and all(gid in index._loading for gid in ids):
            return
        time.sleep(0.005)
    raise AssertionError(f"ids never entered loading: {ids}")


def _wait_jobs(cache, record, at_least, timeout=3.0):
    index = cache._index
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (
            len(record) >= at_least
            and cache._prefetch_queue.empty()
            and not index._loading
        ):
            return
        time.sleep(0.005)
    raise AssertionError(f"prefetch jobs did not settle (calls={len(record)})")


def test_prefetch_disabled_returns_false(monkeypatch):
    """Env unset / "0" and a zero-row cache never fetch and never spawn a thread."""
    monkeypatch.delenv("B12X_PLE_PREFETCH", raising=False)
    payloads = _payloads()
    host_rows = _host_rows(payloads)
    before = threading.active_count()
    cache = _cache(cache_rows=16, cache_tier="host", host_rows=host_rows)
    try:
        assert cache.prefetch([7, 8, 9]) is False
        assert cache._prefetch_thread is None
        monkeypatch.setenv("B12X_PLE_PREFETCH", "0")
        assert cache.prefetch([7, 8, 9]) is False
        assert cache._prefetch_thread is None
    finally:
        cache.close()
    monkeypatch.setenv("B12X_PLE_PREFETCH", "1")
    zero = _cache(cache_rows=0, cache_tier="host", host_rows=host_rows)
    try:
        assert zero.prefetch([7, 8]) is False
        assert zero._prefetch_thread is None
    finally:
        zero.close()
    assert threading.active_count() == before


@torch.inference_mode()
def test_prefetch_ready_hits_after_drain(monkeypatch):
    monkeypatch.setenv("B12X_PLE_PREFETCH", "1")
    payloads = _payloads()
    monkeypatch.setattr(DiskRowCache, "_fetch", _fake_fetch(payloads))
    cache = _cache(cache_rows=32, cache_tier="host", host_rows=_host_rows(payloads))
    try:
        cache.freeze()
        ids = [7, 8, 100, 150, 200, 289]
        assert cache.prefetch(ids) is True
        assert cache._prefetch_thread is not None
        _drain(cache, ids)
        assert all(cache._index.is_ready(gid) for gid in ids)
        with cache._lock:
            plan = cache._index.plan(list(ids))
        assert plan.miss_ids == []
        assert sorted(plan.hit_pos) == list(range(len(ids)))
        host_ids = torch.tensor(ids, dtype=torch.int64)
        with cache.transaction():
            cache.read_rows(host_ids.to(cache.device), len(ids))
            for actual, source in zip((cache.weight, cache.scale), payloads):
                torch.testing.assert_close(
                    actual[: len(ids)].cpu(),
                    _expected(source, host_ids),
                    rtol=0,
                    atol=0,
                )
    finally:
        cache.close()


def _assert_loading_slot_immune(monkeypatch, policy, cache_rows):
    """A row a prefetch is still fetching is invisible and never a victim.

    ``cache_rows`` is policy-sized: S3-FIFO keeps only its small-queue budget
    of rows that have not earned a second read yet, so 32 rows (S = 3) is the
    smallest cache that can hold all three prefetched ids once the
    synchronous batch has run through S.
    """
    monkeypatch.setenv("B12X_PLE_PREFETCH", "1")
    payloads = _payloads()
    gate = threading.Event()
    monkeypatch.setattr(DiskRowCache, "_fetch", _fake_fetch(payloads, block=gate))
    cache = _cache(
        cache_rows=cache_rows,
        cache_tier="host",
        host_rows=_host_rows(payloads),
        policy=policy,
    )
    try:
        cache.freeze()
        assert cache.stats()["row_cache_policy"] == policy
        loading_ids = [7, 8, 9]
        assert cache.prefetch(loading_ids) is True
        _wait_loading(cache, loading_ids)
        slots = {gid: cache._index._loading[gid] for gid in loading_ids}
        # Synchronous misses must evict committed rows, never a loading one.
        with cache._lock:
            plan = cache._index.plan([100, 101, 102, 103, 104, 105, 106, 107, 108])
        assert len(plan.miss_ids) == 9
        assert len(cache._index._slot) > 0
        # A store slot is a row this batch owns; a loading slot is in neither
        # queue, so the two can never name the same pool row.
        assert set(plan.store_slot).isdisjoint(slots.values())
        for gid in loading_ids:
            assert gid in cache._index._loading
            assert not cache._index.is_ready(gid)
            assert cache._index._loading[gid] == slots[gid]
        gate.set()
        _drain(cache, loading_ids)
        assert all(cache._index.is_ready(gid) for gid in loading_ids)
    finally:
        gate.set()
        cache.close()


@torch.inference_mode()
def test_loading_slot_immune_to_eviction(monkeypatch):
    _assert_loading_slot_immune(monkeypatch, "sieve", 6)


@torch.inference_mode()
def test_s3fifo_loading_slot_immune_to_eviction(monkeypatch):
    _assert_loading_slot_immune(monkeypatch, "s3fifo", 32)


def test_duplicate_invalid_noop(monkeypatch):
    monkeypatch.setenv("B12X_PLE_PREFETCH", "1")
    payloads = _payloads()
    record = []
    monkeypatch.setattr(DiskRowCache, "_fetch", _fake_fetch(payloads, record=record))
    cache = _cache(cache_rows=16, cache_tier="host", host_rows=_host_rows(payloads))
    try:
        cache.freeze()
        request = np.array([7, 7, 8, 8, 6, 290, 300, -5, 2**40, 100], dtype=np.int64)
        assert cache.prefetch(request) is True
        _drain(cache, [7, 8, 100])
        assert cache._index.is_ready(7) and cache._index.is_ready(8)
        assert cache._index.is_ready(100)
        for bad in (6, 290, 300, -5, 2**40):
            assert not cache._index.is_ready(bad)
        # One job, exactly the distinct in-shard ids, fetched once each.
        assert len(record) == 1
        fetched = set(
            np.frombuffer(record[0][0], dtype=np.int64, count=record[0][3]).tolist()
        )
        assert fetched == {7, 8, 100}
        # Re-prefetching already-ready rows is a no-op: still a single fetch.
        assert cache.prefetch([7, 8, 100]) is True
        assert len(record) == 1
    finally:
        cache.close()


def test_close_stops_worker(monkeypatch):
    monkeypatch.setenv("B12X_PLE_PREFETCH", "1")
    payloads = _payloads()
    monkeypatch.setattr(DiskRowCache, "_fetch", _fake_fetch(payloads))
    cache = _cache(cache_rows=16, cache_tier="host", host_rows=_host_rows(payloads))
    cache.freeze()
    assert cache.prefetch([7, 8, 9]) is True
    worker = cache._prefetch_thread
    assert worker is not None
    started = time.monotonic()
    cache.close()
    elapsed = time.monotonic() - started
    assert not worker.is_alive()
    assert elapsed <= 2.5


@torch.inference_mode()
def test_rejected_prefetch_sync_identical(monkeypatch):
    monkeypatch.setenv("B12X_PLE_PREFETCH", "1")
    payloads = _payloads()
    record = []
    # The worker's first fetch fails; the synchronous read must still succeed.
    monkeypatch.setattr(
        DiskRowCache, "_fetch", _fake_fetch(payloads, fail_first=1, record=record)
    )
    cache = _cache(cache_rows=16, cache_tier="host", host_rows=_host_rows(payloads))
    try:
        cache.freeze()
        ids = [7, 8, 100]
        assert cache.prefetch(ids) is True
        _wait_jobs(cache, record, 1)
        # A rejected prefetch leaves nothing cached and nothing in flight.
        for gid in ids:
            assert not cache._index.is_ready(gid)
            assert gid not in cache._index._loading
        host_ids = torch.tensor(ids, dtype=torch.int64)
        with cache.transaction():
            cache.read_rows(host_ids.to(cache.device), len(ids))
            for actual, source in zip((cache.weight, cache.scale), payloads):
                torch.testing.assert_close(
                    actual[: len(ids)].cpu(),
                    _expected(source, host_ids),
                    rtol=0,
                    atol=0,
                )
    finally:
        cache.close()
