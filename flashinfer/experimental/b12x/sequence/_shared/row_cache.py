"""SIEVE replacement for immutable embedding-table rows.

Hashed n-gram tables (PLE, Engram) are read row-by-row with a Zipfian access
pattern: on llm-inference-bench traces of Qwen3.8-Flash-Next PLE, ~65% of row
reads repeat and a 10^4-10^5 row working set captures most of the reuse.
SIEVE (Zhang et al., NSDI'24) keeps a single insertion-ordered queue and a
moving hand; a hit only sets a bit, so the per-step bookkeeping stays a few
dictionary operations per looked-up row.

``SieveIndex`` owns only the id->slot mapping. Payload placement (host RAM,
unified RAM, or device memory) and the miss source (disk or a host-resident
table) belong to ``DiskRowCache``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass


@dataclass
class RowPlan:
    """One batch resolved against the index.

    ``hit_pos[i]`` reads cached slot ``hit_slot[i]``. ``miss_ids`` are the
    distinct uncached rows to fetch, in fetch order; position ``miss_pos[i]``
    receives fetched row ``miss_row[i]``. ``store_slot[i]`` receives fetched
    row ``store_row[i]`` after every hit has been read, so an eviction made
    for this batch never clobbers a row the batch still reads.
    """

    hit_pos: list[int]
    hit_slot: list[int]
    miss_ids: list[int]
    miss_pos: list[int]
    miss_row: list[int]
    store_slot: list[int]
    store_row: list[int]


class SieveIndex:
    """Fixed-capacity SIEVE over slot numbers ``0..capacity-1``."""

    def __init__(self, capacity: int) -> None:
        capacity = operator.index(capacity)
        if capacity <= 0:
            raise ValueError("row cache capacity must be positive")
        self.capacity = capacity
        self._slot: dict[int, int] = {}
        self._key = [0] * capacity
        self._visited = bytearray(capacity)
        # Doubly linked queue over slots: head is newest, tail is oldest.
        self._newer = [-1] * capacity
        self._older = [-1] * capacity
        self._head = self._tail = self._hand = -1
        self._used = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return self._used

    def __contains__(self, key: int) -> bool:
        return key in self._slot

    def _unlink(self, slot: int) -> None:
        newer, older = self._newer[slot], self._older[slot]
        if newer >= 0:
            self._older[newer] = older
        else:
            self._head = older
        if older >= 0:
            self._newer[older] = newer
        else:
            self._tail = newer

    def _push_head(self, slot: int) -> None:
        self._older[slot] = self._head
        self._newer[slot] = -1
        if self._head >= 0:
            self._newer[self._head] = slot
        else:
            self._tail = slot
        self._head = slot

    def _evict(self) -> int:
        slot = self._hand if self._hand >= 0 else self._tail
        while self._visited[slot]:
            self._visited[slot] = 0
            slot = self._newer[slot]
            if slot < 0:
                slot = self._tail
        self._hand = self._newer[slot]
        self._unlink(slot)
        del self._slot[self._key[slot]]
        return slot

    def get(self, key: int) -> int | None:
        slot = self._slot.get(key)
        if slot is not None:
            self._visited[slot] = 1
        return slot

    def insert(self, key: int) -> int:
        if key in self._slot:
            raise KeyError("row is already cached")
        if self._used < self.capacity:
            slot = self._used
            self._used += 1
        else:
            slot = self._evict()
        self._slot[key] = slot
        self._key[slot] = key
        self._visited[slot] = 0
        self._push_head(slot)
        return slot

    def plan(self, ids) -> RowPlan:
        """Resolve one batch; duplicates in the batch fetch and store once."""
        hit_pos: list[int] = []
        hit_slot: list[int] = []
        miss_ids: list[int] = []
        miss_pos: list[int] = []
        miss_row: list[int] = []
        fetched: dict[int, int] = {}
        for position, key in enumerate(ids):
            slot = self.get(key)
            if slot is not None:
                hit_pos.append(position)
                hit_slot.append(slot)
                continue
            row = fetched.get(key)
            if row is None:
                row = fetched[key] = len(miss_ids)
                miss_ids.append(key)
            miss_pos.append(position)
            miss_row.append(row)
        self.hits += len(hit_pos)
        self.misses += len(miss_pos)
        # Insert after every lookup so this batch's hits keep their visited bit
        # through the evictions its misses cause. A slot reassigned within the
        # batch keeps only its final owner.
        stored: dict[int, int] = {}
        for row, key in enumerate(miss_ids):
            stored[self.insert(key)] = row
        return RowPlan(
            hit_pos,
            hit_slot,
            miss_ids,
            miss_pos,
            miss_row,
            list(stored),
            list(stored.values()),
        )
