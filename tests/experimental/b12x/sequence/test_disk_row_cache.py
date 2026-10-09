"""GPU byte and lifetime contracts for the selected immutable-file backend.

Set B12X_DISK_BACKEND=gds and place pytest's basetemp on a GDS filesystem to
qualify cuFile. A selected but unavailable transport fails instead of skipping.
"""

from __future__ import annotations

import os

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


@torch.inference_mode()
@pytest.mark.parametrize("tier", ["host", "device"])
@pytest.mark.parametrize("capacity", [1, 5, 64, 400])
@pytest.mark.parametrize("source", ["disk", "host"])
def test_sieve_cached_rows_match_uncached_bytes(
    tmp_path, monkeypatch, tier, capacity, source
):
    """Cached staging must equal the source for repeats, in-batch duplicates,
    invalid IDs, and capacities that evict rows the same batch still reads."""
    monkeypatch.setenv("B12X_DISK_BACKEND", "io_uring")
    payloads = _payloads()
    host_rows = None
    if source == "host":
        host_rows = tuple(p[7:290].contiguous() for p in payloads)
    cache = _cache(cache_rows=capacity, cache_tier=tier, host_rows=host_rows)
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


def test_cache_requires_io_uring_or_host_source(monkeypatch):
    monkeypatch.setenv("B12X_DISK_BACKEND", "gds")
    with pytest.raises(ValueError, match="GDS backend does not support a row cache"):
        _cache(cache_rows=8, cache_tier="device")
