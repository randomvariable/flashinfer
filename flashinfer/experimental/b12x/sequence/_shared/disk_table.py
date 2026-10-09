"""Bounded disk row staging shared by PLE and Engram.

The native reader owns bounded request planning and I/O. This module owns
storage, source registration and the host/GPU transaction boundary; hashing
and numerical decoding belong to each embedding operation.
"""

from __future__ import annotations

import math
import operator
import os
import queue
import sys
import threading
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Iterator

import torch

if TYPE_CHECKING:
    from cuda.bindings import runtime as cudart

# Poison pill handed to the prefetch worker's queue by ``close`` to unblock and
# stop its daemon thread. Distinct by identity from any id list a job may hold.
_PREFETCH_SENTINEL = object()


def _check_cuda(error: cudart.cudaError_t, operation: str) -> None:
    from cuda.bindings import runtime as cudart

    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {error}")


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= int(extent)
    return tuple(reversed(result))


def _tensor_from_pointer(
    pointer: int,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    nbytes: int,
) -> torch.Tensor:
    constructor = getattr(torch._C, "_construct_storage_from_data_pointer", None)
    if constructor is None:
        raise RuntimeError(
            "mapped-host storage requires torch._C._construct_storage_from_data_pointer"
        )
    storage = constructor(int(pointer), device, int(nbytes))
    return torch.empty(0, dtype=dtype, device=device).set_(
        storage, 0, shape, _contiguous_strides(shape)
    )


class MappedHostAllocation:
    """Own a mapped page-locked allocation and its CPU/CUDA tensor aliases."""

    def __init__(
        self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> None:
        from cuda.bindings import runtime as cudart

        self._host_pointer = 0
        self._closed = True
        if device.type != "cuda" or device.index is None:
            raise ValueError(
                f"mapped-host storage requires an indexed CUDA device, got {device}"
            )
        nbytes = math.prod(shape) * dtype.itemsize
        if nbytes <= 0:
            raise ValueError(
                f"mapped-host allocation size must be positive, got {nbytes}"
            )
        self.device = device
        self.nbytes = nbytes
        with torch.cuda.device(device):
            error, pointer = cudart.cudaHostAlloc(
                nbytes, cudart.cudaHostAllocMapped | cudart.cudaHostAllocWriteCombined
            )
            _check_cuda(error, "cudaHostAlloc")
            self._host_pointer = int(pointer)
            self._closed = False
            try:
                error, device_pointer = cudart.cudaHostGetDevicePointer(pointer, 0)
                _check_cuda(error, "cudaHostGetDevicePointer")
                self.host_view = _tensor_from_pointer(
                    self._host_pointer,
                    shape=shape,
                    dtype=dtype,
                    device=torch.device("cpu"),
                    nbytes=nbytes,
                )
                self.device_view = _tensor_from_pointer(
                    int(device_pointer),
                    shape=shape,
                    dtype=dtype,
                    device=device,
                    nbytes=nbytes,
                )
            except Exception:
                cudart.cudaFreeHost(pointer)
                self._host_pointer = 0
                self._closed = True
                raise

    def close(self) -> None:
        from cuda.bindings import runtime as cudart

        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        _check_cuda(cudart.cudaFreeHost(self._host_pointer)[0], "cudaFreeHost")
        self._host_pointer = 0
        self._closed = True

    def __del__(self) -> None:
        if (
            self._closed
            or self._host_pointer == 0
            or sys is None
            or sys.is_finalizing()
        ):
            return
        with suppress(Exception):
            self.close()


class DiskRowCache:
    """Read immutable row planes into a reusable, batch-sized staging buffer.

    Source rows have one weight byte plane and an optional independent scale
    byte plane. Hold ``transaction`` across ID production, ``read_rows`` and
    GPU decoding. Downstream graphs consume the decoder's fixed device output,
    not disk I/O.

    Rows come from checkpoint shards via io_uring/GDS, or from ``host_rows``:
    CPU tensors holding this shard's rows ``[shard_start, shard_end)`` in id
    order. An optional SIEVE row cache (``cache_rows`` > 0) keeps recently
    read rows so repeated n-grams skip the source:

    * ``cache_tier="host"`` keeps them in host memory and stages through the
      mapped-host buffer. On unified-memory parts (GB10) this is the only copy.
    * ``cache_tier="device"`` keeps them in device memory, stages into a device
      buffer, and transfers only misses (discrete GPUs).

    ``cache_rows``/``cache_tier`` default to ``B12X_ROW_CACHE_ROWS`` (0, off)
    and ``B12X_ROW_CACHE_TIER`` (``auto``: host on integrated GPUs, device
    otherwise). The GDS backend stages on the device itself and is not cached.

    ``prefetch(ids)`` warms the cache ahead of the read path from a background
    worker, driven by ``B12X_PLE_PREFETCH`` (off by default). Reserved rows sit
    in a loading slot that a lookup never hits and eviction never reclaims;
    only a confirmed row becomes a ready hit. The synchronous read path treats
    an in-flight prefetch as a plain miss and takes its own fetch, so the same
    row can be written by both the worker and the batch, always with identical
    bytes (last write wins). Both the disk and ``host_rows`` sources are
    prefetchable through the same ``_fetch`` primitive.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        max_lookups: int,
        table_rows: int,
        shard_start: int,
        shard_end: int,
        shard_rows: int,
        weight_row_bytes: int,
        scale_row_bytes: int = 0,
        queue_depth: int = 64,
        cache_rows: int | None = None,
        cache_tier: str | None = None,
        host_rows: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    ) -> None:
        from b12x.loader._native import load

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("disk row staging requires CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.max_lookups = operator.index(max_lookups)
        self.table_rows = operator.index(table_rows)
        self.shard_start = operator.index(shard_start)
        self.shard_end = operator.index(shard_end)
        self.shard_rows = operator.index(shard_rows)
        self.weight_row_bytes = operator.index(weight_row_bytes)
        self.scale_row_bytes = operator.index(scale_row_bytes)
        queue_depth = operator.index(queue_depth)
        if not 0 < self.shard_rows <= (1 << 63) - 1:
            raise ValueError("shard_rows must be a positive signed int64")
        self._backend = (
            "host"
            if host_rows is not None
            else os.environ.get("B12X_DISK_BACKEND", "io_uring")
        )
        if self._backend not in ("io_uring", "gds", "host"):
            raise ValueError("B12X_DISK_BACKEND must be io_uring or gds")
        if cache_rows is None:
            cache_rows = int(os.environ.get("B12X_ROW_CACHE_ROWS", "0"))
        cache_rows = operator.index(cache_rows)
        if cache_rows < 0:
            raise ValueError("cache_rows must be nonnegative")
        if cache_tier is None:
            cache_tier = os.environ.get("B12X_ROW_CACHE_TIER", "auto")
        if cache_tier == "auto":
            integrated = torch.cuda.get_device_properties(device).is_integrated
            cache_tier = "host" if integrated else "device"
        if cache_tier not in ("host", "device"):
            raise ValueError("cache_tier must be host, device or auto")
        if cache_rows and self._backend == "gds":
            raise ValueError("the GDS backend does not support a row cache")
        self.cache_rows = cache_rows
        self.cache_tier = cache_tier if cache_rows else None
        self._gds = None
        self._native = None
        self._reader = None
        self._host_weight = self._host_scale = None
        if self._backend == "host":
            self._host_weight, self._host_scale = self._check_host_rows(host_rows)
        elif self._backend == "io_uring":
            self._native = load()
            self._reader = self._native.ple_reader(
                self.shard_rows,
                self.table_rows,
                self.shard_start,
                self.shard_end,
                self.weight_row_bytes,
                self.scale_row_bytes,
                self.max_lookups,
                queue_depth,
            )
        self.shard_count = (self.table_rows + self.shard_rows - 1) // self.shard_rows
        self.ids_host = torch.empty(
            (self.max_lookups,), dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._ids_buffer = memoryview(self.ids_host.numpy())
        self._weight_allocation = self._scale_allocation = None
        self.weight_host = self.scale_host = None
        self._weight_buffer = self._scale_buffer = None
        self.scale = None
        if self._backend == "gds":
            from ._gds import GdsRows

            self._gds = GdsRows(self, queue_depth)
            self.weight, self.scale = self._gds.weight, self._gds.scale
        elif self.cache_tier == "device":
            self.weight = torch.empty(
                (self.max_lookups, self.weight_row_bytes),
                dtype=torch.uint8,
                device=device,
            )
            if self.scale_row_bytes:
                self.scale = torch.empty(
                    (self.max_lookups, self.scale_row_bytes),
                    dtype=torch.uint8,
                    device=device,
                )
        else:
            self._weight_allocation = MappedHostAllocation(
                (self.max_lookups, self.weight_row_bytes), torch.uint8, device
            )
            self.weight = self._weight_allocation.device_view
            self.weight_host = self._weight_allocation.host_view
            self._weight_buffer = memoryview(self.weight_host.numpy())
            if self.scale_row_bytes:
                self._scale_allocation = MappedHostAllocation(
                    (self.max_lookups, self.scale_row_bytes), torch.uint8, device
                )
                self.scale = self._scale_allocation.device_view
                self.scale_host = self._scale_allocation.host_view
                self._scale_buffer = memoryview(self.scale_host.numpy())
        self._index = None
        if cache_rows:
            self._init_row_cache()
        self._sources: set[tuple[bool, int]] = set()
        self._frozen = False
        self._lock = threading.RLock()
        self._transaction_thread: int | None = None
        self._transaction_stream: torch.cuda.Stream | None = None
        self._ids_ready = torch.cuda.Event()
        self._cache_done = torch.cuda.Event()
        self._cache_used = False
        self._closed = False
        # Speculative PLE row prefetch. The worker thread and its bounded queue
        # are created lazily on the first enabled ``prefetch`` so a disabled
        # cache never spawns a thread; both stay ``None`` until then.
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_queue: queue.Queue | None = None
        self._prefetch_stop = False

    def _check_host_rows(self, host_rows):
        weight, scale = host_rows
        rows = self.shard_end - self.shard_start
        planes = [(weight, self.weight_row_bytes, "weight")]
        if self.scale_row_bytes:
            planes.append((scale, self.scale_row_bytes, "scale"))
        elif scale is not None:
            raise ValueError("host_rows has a scale plane but scale_row_bytes is 0")
        for tensor, width, name in planes:
            if (
                tensor is None
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.uint8
                or tuple(tensor.shape) != (rows, width)
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"host_rows {name} must be a contiguous CPU uint8 tensor of "
                    f"shape ({rows}, {width})"
                )
        return weight.numpy(), (scale.numpy() if self.scale_row_bytes else None)

    def _init_row_cache(self) -> None:
        from .row_cache import SieveIndex

        rows = self.cache_rows
        widths = [self.weight_row_bytes] + (
            [self.scale_row_bytes] if self.scale_row_bytes else []
        )
        # vLLM constructs models under an ambient torch.device context (meta at
        # initialize_model time), so every CPU allocation here names the device
        # explicitly; a bare torch.empty(pin_memory=True) would build a meta
        # tensor and fail "Only dense CPU tensors can be pinned".
        self._index = SieveIndex(rows)
        # Miss rows land here first; the CPU reads them back, so they are plain
        # pinned memory rather than write-combined.
        self._miss_ids = torch.empty(
            (self.max_lookups,), dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._miss_planes = [
            torch.empty(
                (self.max_lookups, width),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            for width in widths
        ]
        if self.cache_tier == "host":
            self._pool = [
                torch.empty((rows, width), dtype=torch.uint8, device="cpu").numpy()
                for width in widths
            ]
            self._stage_host = [self.weight_host.numpy()] + (
                [self.scale_host.numpy()] if self.scale_row_bytes else []
            )
        else:
            self._pool = [
                torch.empty((rows, width), dtype=torch.uint8, device=self.device)
                for width in widths
            ]
            self._stage_device = [self.weight] + (
                [self.scale] if self.scale_row_bytes else []
            )
            # hit_pos, hit_slot, miss_pos, miss_row, store_slot, store_row.
            self._plan_host = torch.empty(
                (6, self.max_lookups),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            self._plan_device = torch.empty(
                (6, self.max_lookups), dtype=torch.int64, device=self.device
            )

    def _require_open(self):
        if self._closed:
            raise RuntimeError("disk row cache is closed")

    def __del__(self):
        if not getattr(self, "_closed", True):
            with suppress(Exception):
                self.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._transaction_thread is not None:
                raise RuntimeError("cannot close a disk row cache during a transaction")
            self._prefetch_stop = True
            worker = self._prefetch_thread
            jobs = self._prefetch_queue
            if self._cache_used:
                with torch.cuda.device(self.device):
                    self._cache_done.synchronize()
            if self._gds is not None:
                self._gds.close()
            for allocation in (self._scale_allocation, self._weight_allocation):
                if allocation is not None:
                    allocation.close()
            self._reader = self._native = None
            self._closed = True
        # Drain the worker outside the lock: a running job reacquires it in its
        # reserve/confirm sections, so joining while holding it would deadlock.
        if worker is not None and jobs is not None:
            with suppress(queue.Full):
                jobs.put_nowait(_PREFETCH_SENTINEL)
            worker.join(timeout=2)

    def prefetch(self, ids) -> bool:
        """Speculatively fetch ``ids`` into the row cache without blocking.

        Advisory only: it never touches the staging buffer, never raises, and
        hands the work to a single lazily-started daemon worker over a bounded
        queue, so callers are never stalled behind disk I/O. Returns ``False``
        when the row cache is disabled (``cache_rows == 0``) or the
        ``B12X_PLE_PREFETCH`` environment variable is unset or ``"0"``;
        otherwise ``True``. ``ids`` may be a list of ints, a CPU int64 tensor,
        or a numpy array; duplicate, out-of-shard, and already-handled ids are
        no-ops. When the queue is saturated the newest request is dropped.
        """
        if not self._prefetch_ok():
            return False
        job = self._normalize_prefetch_ids(ids)
        if not job:
            return True
        self._ensure_prefetch_worker()
        if self._prefetch_queue is None:
            return True
        try:
            self._prefetch_queue.put_nowait(job)
        except queue.Full:
            pass
        return True

    def _prefetch_ok(self) -> bool:
        """Gate speculative prefetch: cache enabled, disk/host backend, opt-in env.

        ``b12x`` has no ``vllm`` env module, so the flag is read straight from
        ``os.environ``. It defaults to off; only a set, nonzero value turns
        prefetch on. Returns ``False`` for a disabled cache (``cache_rows == 0``)
        or a closed one so no worker is ever spawned for a cache that cannot
        hold rows.
        """
        if self._closed or not self.cache_rows or self._index is None:
            return False
        if self._backend == "gds":
            return False
        flag = os.environ.get("B12X_PLE_PREFETCH")
        return flag is not None and flag != "0"

    def _normalize_prefetch_ids(self, ids) -> list[int]:
        """Coerce ``ids`` to the distinct in-shard, not-yet-cached row ids.

        Accepts a list of ints, a CPU int64 tensor, or a numpy array; every
        element is converted with ``int``. Duplicates collapse to first
        occurrence, out-of-shard ids are dropped (nothing to prefetch), and
        ids already committed are dropped as no-ops. Rows merely loading are
        left in: their ``reserve`` returns ``None`` in the worker, so they cost
        a cheap no-op rather than a second fetch.
        """
        import numpy as np

        if isinstance(ids, torch.Tensor):
            if ids.numel() == 0:
                return []
            ids = ids.detach().cpu()
            if ids.dtype != torch.int64:
                ids = ids.to(torch.int64)
            raw = ids.reshape(-1).tolist()
        elif isinstance(ids, np.ndarray):
            if ids.size == 0:
                return []
            raw = ids.reshape(-1).tolist()
        else:
            raw = list(ids)
        low, high = self.shard_start, self.shard_end
        index = self._index
        seen: set[int] = set()
        result: list[int] = []
        with self._lock:
            for value in raw:
                key = int(value)
                if key in seen:
                    continue
                seen.add(key)
                if not low <= key < high:
                    continue
                if index.is_ready(key):
                    continue
                result.append(key)
        return result

    def _ensure_prefetch_worker(self) -> None:
        with self._lock:
            if self._prefetch_thread is not None or self._closed:
                return
            self._prefetch_queue = queue.Queue(maxsize=64)
            self._prefetch_stop = False
            worker = threading.Thread(
                target=self._prefetch_loop,
                name="b12x-ple-prefetch",
                daemon=True,
            )
            self._prefetch_thread = worker
        worker.start()

    def _prefetch_loop(self) -> None:
        while not self._prefetch_stop:
            try:
                job = self._prefetch_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if job is _PREFETCH_SENTINEL:
                break
            try:
                self._run_prefetch_job(job)
            except Exception:
                # Prefetch is advisory: a source/transport error must never
                # surface to, or destabilize, the caller or the worker.
                pass

    def _run_prefetch_job(self, ids) -> None:
        import numpy as np

        index = self._index
        reserved: list[tuple[int, int]] = []
        with self._lock:
            if self._closed:
                return
            for key in ids:
                slot = index.reserve(key)
                if slot is not None:
                    reserved.append((key, slot))
        if not reserved:
            return
        count = len(reserved)
        weight_np = np.zeros((count, self.weight_row_bytes), dtype=np.uint8)
        scale_np = (
            np.zeros((count, self.scale_row_bytes), dtype=np.uint8)
            if self.scale_row_bytes
            else None
        )
        ids_np = np.fromiter((key for key, _ in reserved), dtype=np.int64, count=count)
        try:
            self._fetch(
                memoryview(ids_np),
                memoryview(weight_np),
                memoryview(scale_np) if scale_np is not None else None,
                count,
            )
            fetched = True
        except Exception:
            fetched = False
        planes = [weight_np] + ([scale_np] if scale_np is not None else [])
        with self._lock:
            if self._closed:
                for _, slot in reserved:
                    index.cancel(slot)
                return
            if not fetched:
                for _, slot in reserved:
                    index.cancel(slot)
                return
            if self.cache_tier == "host":
                for row, (_, slot) in enumerate(reserved):
                    if index._state[slot] != index._LOADING:
                        continue
                    for plane, pool in zip(planes, self._pool):
                        pool[slot] = plane[row]
            else:
                with torch.cuda.device(self.device):
                    for row, (_, slot) in enumerate(reserved):
                        if index._state[slot] != index._LOADING:
                            continue
                        for plane, pool in zip(planes, self._pool):
                            pool[slot].copy_(torch.from_numpy(plane[row]))
                    torch.cuda.current_stream(self.device).synchronize()
            for _, slot in reserved:
                index.confirm(slot)

    def add_shard(
        self, shard_index: int, path: str, offset: int, *, scale: bool = False
    ) -> None:
        self._require_open()
        if self._backend == "host":
            raise RuntimeError("host-sourced rows have no checkpoint shards")
        with self._lock:
            if self._frozen:
                raise RuntimeError("cannot change disk shards after binding")
            shard_index = operator.index(shard_index)
            offset = operator.index(offset)
            if not 0 <= shard_index < self.shard_count:
                raise ValueError("checkpoint shard index is out of range")
            if offset < 0:
                raise ValueError("checkpoint file offset must be nonnegative")
            if scale and not self.scale_row_bytes:
                raise ValueError("disk table has no row scale plane")
            key = (scale, shard_index)
            if key in self._sources:
                raise ValueError("checkpoint shard is already registered")
            start = shard_index * self.shard_rows
            end = min(start + self.shard_rows, self.table_rows)
            if end <= self.shard_start or start >= self.shard_end:
                return
            if self._gds is not None:
                self._gds.native.add(
                    self._gds.reader, shard_index, os.fspath(path), offset, scale
                )
            else:
                self._native.ple_reader_add(
                    self._reader, shard_index, os.fspath(path), offset, scale
                )
            self._sources.add(key)

    def require_complete(self) -> None:
        self._require_open()
        with self._lock:
            if self._backend == "host":
                return
            first = self.shard_start // self.shard_rows
            last = (self.shard_end + self.shard_rows - 1) // self.shard_rows
            for shard in range(first, last):
                if (False, shard) not in self._sources:
                    raise ValueError(f"missing disk weight shard {shard}")
                if self.scale_row_bytes and (True, shard) not in self._sources:
                    raise ValueError(f"missing disk scale shard {shard}")

    def freeze(self) -> None:
        with self._lock:
            self.require_complete()
            self._frozen = True

    @contextmanager
    def transaction(self) -> Iterator[DiskRowCache]:
        self._require_open()
        if torch.compiler.is_compiling():
            raise RuntimeError("disk table preparation cannot run under torch.compile")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "disk table preparation must run outside CUDA graph capture"
                )
            with self._lock:
                if self._transaction_thread is not None:
                    raise RuntimeError("disk row transactions cannot be nested")
                stream = torch.cuda.current_stream(self.device)
                if self._cache_used:
                    stream.wait_event(self._cache_done)
                self._transaction_thread = threading.get_ident()
                self._transaction_stream = stream
                try:
                    yield self
                finally:
                    self._cache_done.record(stream)
                    self._cache_used = True
                    self._transaction_stream = None
                    self._transaction_thread = None

    def read_rows(self, ids: torch.Tensor, count: int) -> None:
        self._stage_ids(ids, count)
        self._read_staged(count)

    def _stage_ids(self, ids: torch.Tensor, count: int) -> None:
        if self._transaction_thread != threading.get_ident():
            raise RuntimeError("read_rows requires an active disk row transaction")
        count = operator.index(count)
        if not 0 <= count <= self.max_lookups:
            raise ValueError("disk lookup count exceeds batch capacity")
        if (
            ids.device != self.device
            or ids.dtype != torch.int64
            or not ids.is_contiguous()
        ):
            raise ValueError(
                "disk row IDs must be contiguous CUDA int64 on the cache device"
            )
        if ids.numel() < count:
            raise ValueError("disk row IDs do not cover the requested count")
        self.ids_host[:count].copy_(ids.view(-1)[:count], non_blocking=True)
        self._ids_ready.record(self._transaction_stream)

    def _read_staged(self, count: int) -> None:
        # Completes ID production and all prior cache readers before host writes.
        with torch.cuda.device(self.device):
            self._ids_ready.synchronize()
            if self._gds is not None:
                self._gds.read(self._ids_buffer, count, self._transaction_stream)
                return
            if self._index is None:
                self._fetch(
                    self._ids_buffer, self._weight_buffer, self._scale_buffer, count
                )
            elif count:
                self._read_cached(count)

    def _fetch(self, ids, weights, scales, count: int) -> None:
        """Write source rows for ``ids[:count]`` into C-contiguous row buffers."""
        if self._backend == "io_uring":
            self._native.ple_reader_run(self._reader, ids, weights, scales, count)
            return
        import numpy as np

        ids = np.frombuffer(ids, dtype=np.int64, count=count)
        valid = np.flatnonzero((ids >= self.shard_start) & (ids < self.shard_end))
        rows = ids[valid] - self.shard_start
        planes = [(weights, self._host_weight)]
        if self.scale_row_bytes:
            planes.append((scales, self._host_scale))
        for out, table in planes:
            out = np.frombuffer(out, dtype=np.uint8).reshape(-1, table.shape[1])
            out[:count] = 0
            out[valid] = table[rows]

    def _read_cached(self, count: int) -> None:
        plan = self._index.plan(self.ids_host[:count].tolist())
        misses = len(plan.miss_ids)
        if misses:
            self._miss_ids[:misses] = torch.tensor(plan.miss_ids, dtype=torch.int64)
            self._fetch(
                memoryview(self._miss_ids.numpy()),
                memoryview(self._miss_planes[0].numpy()),
                memoryview(self._miss_planes[1].numpy())
                if self.scale_row_bytes
                else None,
                misses,
            )
        if self.cache_tier == "host":
            for stage, pool, fetched in zip(
                self._stage_host, self._pool, self._miss_planes
            ):
                fetched = fetched.numpy()
                # Hits read the pool before any store can reuse their slots.
                if plan.hit_pos:
                    stage[plan.hit_pos] = pool[plan.hit_slot]
                if misses:
                    stage[plan.miss_pos] = fetched[plan.miss_row]
                    pool[plan.store_slot] = fetched[plan.store_row]
            return
        lists = (
            plan.hit_pos,
            plan.hit_slot,
            plan.miss_pos,
            plan.miss_row,
            plan.store_slot,
            plan.store_row,
        )
        host = self._plan_host.numpy()
        for row, values in enumerate(lists):
            host[row, : len(values)] = values
        width = max(map(len, lists))
        stream = self._transaction_stream
        with torch.cuda.stream(stream):
            index = self._plan_device[:, :width]
            index.copy_(self._plan_host[:, :width], non_blocking=True)
            hits = len(plan.hit_pos)
            stores = len(plan.store_slot)
            for stage, pool, fetched in zip(
                self._stage_device, self._pool, self._miss_planes
            ):
                # Hits read the pool before any store can reuse their slots.
                if hits:
                    stage.index_copy_(
                        0, index[0, :hits], pool.index_select(0, index[1, :hits])
                    )
                if misses:
                    rows = fetched[:misses].to(self.device, non_blocking=True)
                    positions = len(plan.miss_pos)
                    stage.index_copy_(
                        0,
                        index[2, :positions],
                        rows.index_select(0, index[3, :positions]),
                    )
                    pool.index_copy_(
                        0, index[4, :stores], rows.index_select(0, index[5, :stores])
                    )

    def stats(self) -> dict[str, int | float]:
        self._require_open()
        with self._lock:
            if self._gds is not None:
                result = dict(self._gds.native.stats(self._gds.reader))
            elif self._native is not None:
                result = dict(self._native.ple_reader_stats(self._reader))
            else:
                result = {"staging_bytes": 0, "metadata_bytes": 0}
            result["ids_host_bytes"] = (
                self.ids_host.numel() * self.ids_host.element_size()
            )
            result["weight_cache_bytes"] = (
                self.weight.numel() * self.weight.element_size()
            )
            result["scale_cache_bytes"] = (
                self.scale.numel() * self.scale.element_size()
                if self.scale is not None
                else 0
            )
            result["cache_bytes"] = (
                result["weight_cache_bytes"] + result["scale_cache_bytes"]
            )
            result["owned_staging_bytes"] = (
                result["staging_bytes"]
                + result["ids_host_bytes"]
                + result["cache_bytes"]
            )
            descriptors = result.get("descriptor_bytes", 0)
            result["gds_enabled"] = int(self._gds is not None)
            device_staged = self._gds is not None or self.cache_tier == "device"
            result["device_staging_bytes"] = (
                (result["staging_bytes"] + result["cache_bytes"] + descriptors)
                if device_staged
                else 0
            )
            result["owned_host_bytes"] = (
                result["ids_host_bytes"] + descriptors + result["metadata_bytes"]
            )
            if not device_staged:
                result["owned_host_bytes"] += (
                    result["staging_bytes"] + result["cache_bytes"]
                )
            result["owned_staging_bytes"] += 2 * descriptors
            if self._index is not None:
                pool_bytes = self.cache_rows * (
                    self.weight_row_bytes + self.scale_row_bytes
                )
                lookups = self._index.hits + self._index.misses
                result.update(
                    row_cache_tier=self.cache_tier,
                    row_cache_rows=self.cache_rows,
                    row_cache_resident=len(self._index),
                    row_cache_bytes=pool_bytes,
                    row_cache_hits=self._index.hits,
                    row_cache_misses=self._index.misses,
                    row_cache_hit_rate=(self._index.hits / lookups if lookups else 0.0),
                )
            return result
