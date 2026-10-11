"""Replacement policies for immutable embedding-table rows.

Hashed n-gram tables (PLE, Engram) are read row-by-row with a Zipfian access
pattern: on llm-inference-bench traces of Qwen3.8-Flash-Next PLE, ~65% of row
reads repeat and a 10^4-10^5 row working set captures most of the reuse.
SIEVE (Zhang et al., NSDI'24) keeps a single insertion-ordered queue and a
moving hand; a hit only sets a bit, so the per-step bookkeeping stays a few
dictionary operations per looked-up row.

``S3FifoIndex`` is a drop-in sibling implementing S3-FIFO (Yang et al.,
SOSP'23): a small frequency-filtered FIFO, a main FIFO that reinserts rows
still holding counter credit, and a ghost list of recently demoted keys.
``WTinyLfuIndex`` is a second sibling implementing W-TinyLFU (Einziger et
al., WWW'18): an LRU admission window feeding a segmented-LRU main cache
(probation/protected), with a minimal-increment count-bloom filter deciding
which victim to demote. ``DiskRowCache`` selects among the policies with
``B12X_ROW_CACHE_POLICY``.

``SieveIndex`` owns only the id->slot mapping. Payload placement (host RAM,
unified RAM, or device memory) and the miss source (disk or a host-resident
table) belong to ``DiskRowCache``.

Every slot lives in one of three states. ``FREE`` slots are unclaimed. A
speculative prefetch ``reserve``\\ s a ``LOADING`` slot for a row it is still
fetching: a loading slot consumes capacity, is never an eviction victim, and
is never returned by ``get`` so an in-flight prefetch is invisible to the
read path. ``confirm`` promotes a loading slot to ``READY``, the only state the
synchronous ``plan`` observes. ``cancel`` drops a loading reservation without
touching the queue. ``DiskRowCache.prefetch`` drives that lifecycle; ordinary
reads only ever see ``READY`` slots and behave exactly as before prefetch
existed.
"""

from __future__ import annotations

import operator
from collections import deque
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
    """Fixed-capacity SIEVE over slot numbers ``0..capacity-1``.

    Loading (in-flight prefetch) slots are tracked apart from the ready queue:
    they are excluded from ``_slot`` so lookups miss them, and excluded from
    the doubly linked queue so the eviction hand can never reclaim one. A
    cancelled loading slot returns to a small free pool that ``reserve`` and
    ``insert`` reuse before bumping the allocation high-water mark, so dropped
    prefetches do not leak capacity.
    """

    _FREE = 0
    _LOADING = 1
    _READY = 2

    def __init__(self, capacity: int) -> None:
        capacity = operator.index(capacity)
        if capacity <= 0:
            raise ValueError("row cache capacity must be positive")
        self.capacity = capacity
        self._slot: dict[int, int] = {}
        self._key = [0] * capacity
        self._visited = bytearray(capacity)
        # Per-slot state: _FREE / _LOADING / _READY.
        self._state = bytearray(capacity)
        # key -> slot for rows a prefetch reserved but has not committed. They
        # are deliberately kept out of ``_slot`` so ``get`` and ``plan`` miss.
        self._loading: dict[int, int] = {}
        # Slots freed by cancel(); handed back out before ``_used`` grows.
        self._free: list[int] = []
        # Doubly linked queue over slots: head is newest, tail is oldest. Only
        # READY slots are linked, so the eviction hand never reaches a loading
        # slot.
        self._newer = [-1] * capacity
        self._older = [-1] * capacity
        self._head = self._tail = self._hand = -1
        self._used = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        # Occupied slots, not the high-water mark: a cancelled loading slot
        # sits in the free pool and must not count as resident.
        return len(self._slot) + len(self._loading)

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

    def _evict(self) -> int | None:
        # Walk the ready queue from the hand for an unvisited slot. A FULL
        # index whose every slot is still loading has an empty queue and no
        # victim; report that rather than indexing a -1 tail.
        slot = self._hand if self._hand >= 0 else self._tail
        if slot < 0:
            return None
        while self._visited[slot]:
            self._visited[slot] = 0
            slot = self._newer[slot]
            if slot < 0:
                slot = self._tail
        self._hand = self._newer[slot]
        self._unlink(slot)
        del self._slot[self._key[slot]]
        return slot

    def _claim(self, protect_ready: bool = False) -> int | None:
        """Take an unowned slot index, or return ``None`` when none is free."""
        if self._free:
            return self._free.pop()
        if self._used < self.capacity:
            slot = self._used
            self._used += 1
            return slot
        if protect_ready and len(self._slot) <= 1:
            # Keep at least one committed row evictable so the synchronous
            # read path can never starve behind a queue of in-flight prefetches.
            return None
        return self._evict()

    def get(self, key: int) -> int | None:
        slot = self._slot.get(key)
        if slot is not None:
            self._visited[slot] = 1
        return slot

    def is_ready(self, key: int) -> bool:
        """Whether ``key`` holds a committed (READY) row."""
        return key in self._slot

    def reserve(self, key: int) -> int | None:
        """Claim a loading slot for ``key``; ``None`` if it cannot be taken.

        A row already committed or already loading is a no-op (``None``), as is
        one that cannot be evicted for without stripping the last ready row.
        """
        if key in self._slot or key in self._loading:
            return None
        slot = self._claim(protect_ready=True)
        if slot is None:
            return None
        self._key[slot] = key
        self._state[slot] = self._LOADING
        self._loading[key] = slot
        return slot

    def confirm(self, slot: int) -> int:
        """Promote a loading slot to ready; idempotent if already committed."""
        key = self._key[slot]
        if self._state[slot] != self._LOADING:
            return slot
        self._loading.pop(key, None)
        self._slot[key] = slot
        self._visited[slot] = 0
        self._state[slot] = self._READY
        self._push_head(slot)
        return slot

    def cancel(self, slot: int) -> None:
        """Drop a loading reservation, returning the slot to the free pool."""
        if self._state[slot] != self._LOADING:
            return
        self._loading.pop(self._key[slot], None)
        self._state[slot] = self._FREE
        self._free.append(slot)

    def insert(self, key: int) -> int | None:
        if key in self._loading:
            # The synchronous path is finishing a row a prefetch is also
            # fetching: adopt the reserved slot so the two never own separate
            # copies of the same id.
            return self.confirm(self._loading[key])
        if key in self._slot:
            raise KeyError("row is already cached")
        slot = self._claim()
        if slot is None:
            return None
        self._slot[key] = slot
        self._key[slot] = key
        self._visited[slot] = 0
        self._state[slot] = self._READY
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
        # batch keeps only its final owner. A miss that cannot obtain a slot
        # (every slot still loading) is simply not cached this time; its
        # staging is unaffected, since misses are staged from the fetched
        # planes rather than from the pool.
        stored: dict[int, int] = {}
        for row, key in enumerate(miss_ids):
            slot = self.insert(key)
            if slot is not None:
                stored[slot] = row
        return RowPlan(
            hit_pos,
            hit_slot,
            miss_ids,
            miss_pos,
            miss_row,
            list(stored),
            list(stored.values()),
        )


class S3FifoIndex:
    """Fixed-capacity S3-FIFO over the same slot pool.

    A drop-in sibling of :class:`SieveIndex`: the constructor and the
    ``get``/``insert``/``plan``/``is_ready``/``reserve``/``confirm``/``cancel``
    contract match, so ``DiskRowCache`` may run either replacement policy over
    one payload pool. Slots are pool indices ``0..capacity-1`` and the queues
    are sequences of slots, so a slot handed out by ``insert`` keeps addressing
    the row its key owns.

    The bookkeeping mirrors libCacheSim's ``S3FIFO.c`` (the post-2024 variant,
    which is what the ``cachesim S3FIFO-0.1000-2`` reference numbers are), so
    parity with the C is the contract, not just the paper's:

    * small ``S`` = 10% of capacity, main ``M`` = the rest, ghost ``G`` (keys
      only) = 90% of capacity, each a FIFO.
    * A hit raises the 2-bit counter in whichever queue holds the row; a miss
      removes the key from ``G`` and remembers ``hit_on_ghost``. A resident
      hit never touches ``G``.
    * Admission: a ghost miss joins ``M`` directly; otherwise ``S`` — except
      that before the cache has ever evicted, an already-full ``S`` admits to
      ``M`` instead (the C's ``has_evicted`` fast path).
    * Recycling ``S`` promotes to ``M`` when the counter reaches
      ``move-to-main-threshold`` (2), *keeping* the counter; anything else
      demotes to ``G`` and frees its slot. Recycling ``M`` is a 2-bit clock:
      front rows with credit are reinserted with ``min(freq, 3) - 1`` until
      one without credit leaves.
    * The eviction source is ``M`` when ``M`` is over budget or ``S`` is
      empty, else ``S``; a promotion that overfills ``M`` is paid back from
      ``M`` at once, because the next eviction would pick ``M`` anyway.

    Loading (in-flight prefetch) slots are separate state, exactly as in
    :class:`SieveIndex`: a reserved key is invisible to ``get``, ``plan`` and
    ``is_ready``, its slot joins no queue, and no queue pop can reclaim one.
    A confirmed prefetch is admitted as a fresh row -- into ``M`` when its key
    is a ghost, else ``S`` (or ``M`` under the pre-first-eviction fast path).
    """

    _FREE = 0
    _LOADING = 1
    _READY = 2

    _UNQUEUED = 0
    _SMALL = 1
    _MAIN = 2

    # 2-bit saturation, and the capacity percentages of S and of the ghost
    # list, matching the C's DEFAULT_CACHE_PARAMS
    # ("small-size-ratio=0.10,ghost-size-ratio=0.90,move-to-main-threshold=2").
    _MAX_FREQ = 3
    _SMALL_SHARE = 10
    _GHOST_SHARE = 90
    _MOVE_TO_MAIN_THRESHOLD = 2

    def __init__(self, capacity: int) -> None:
        capacity = operator.index(capacity)
        if capacity <= 0:
            raise ValueError("row cache capacity must be positive")
        self.capacity = capacity
        # A 10% share rounds up to at least one row, so even a tiny cache has
        # a small queue to filter one-hit rows; M takes what is left. A
        # capacity of one leaves M with no budget at all, and the code below
        # treats an empty budget as "no promotions, no ghost fast-path"
        # rather than as a queue that may transiently hold a row nobody owns.
        self._small_cap = max(1, capacity * self._SMALL_SHARE // 100)
        self._main_cap = capacity - self._small_cap
        self._ghost_cap = capacity * self._GHOST_SHARE // 100
        self._slot: dict[int, int] = {}
        self._key = [0] * capacity
        self._freq = bytearray(capacity)
        # Per-slot state: _FREE / _LOADING / _READY.
        self._state = bytearray(capacity)
        # Per-slot queue tag. A loading slot is _UNQUEUED, which is what keeps
        # an in-flight prefetch out of reach of every pop.
        self._queue = bytearray(capacity)
        # key -> slot for rows a prefetch reserved but has not committed. They
        # are deliberately kept out of ``_slot`` so ``get`` and ``plan`` miss.
        self._loading: dict[int, int] = {}
        # Slots freed by a queue pop or cancel(); handed out before ``_used``
        # grows, so recycled rows and dropped prefetches do not leak capacity.
        self._free: list[int] = []
        self._small: deque[int] = deque()
        self._main: deque[int] = deque()
        # Ghost keys in demotion order, as a dict for O(1) membership: every
        # miss asks whether its key is a ghost.
        self._ghost: dict[int, None] = {}
        # Set once the cache has evicted its first row; gates the C's
        # "S full before first eviction admits to M" fast path.
        self._has_evicted = False
        self._used = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        # Occupied slots, not the high-water mark: a cancelled loading slot
        # sits in the free pool and must not count as resident.
        return len(self._slot) + len(self._loading)

    def __contains__(self, key: int) -> bool:
        # Resident means queued: S or M. A ghost is not resident, and neither
        # is a row whose prefetch is still in flight.
        return key in self._slot

    def _bind(self, slot: int, key: int, *, main: bool) -> None:
        """Make ``slot`` the ready home of ``key``, at the tail of one queue."""
        self._slot[key] = slot
        self._key[slot] = key
        self._freq[slot] = 0
        self._state[slot] = self._READY
        self._queue[slot] = self._MAIN if main else self._SMALL
        (self._main if main else self._small).append(slot)

    def _prepare_admission(self, *, main: bool) -> None:
        """Evict until the pool can take one more committed row.

        ``main`` is the row's *planned* destination: M's overfill test is
        evaluated as if the newcomer had already been appended, reproducing
        the C's single ``S3FIFO_evict`` call (the parent checks
        ``occupied_main > size_main`` with the insert pending). Loading slots
        are excluded -- they are neither queued nor counted by the C's
        sub-cache occupancy -- and a release that leaves a free slot ends the
        loop, exactly as the C's ``while (space_after_add > cache_size)``.
        """
        while len(self._slot) >= self.capacity:
            self._evict_one(main=main)
            if self._free:
                return

    def _settle_main_budget(self) -> None:
        """Charge M's overfill the pop C defers to its next eviction.

        The C allows M to sit one over budget between an M admission and the
        next ``S3FIFO_evict``; settling right after the bind keeps the queue
        budgets a hard invariant with the same victim -- the newcomer is at
        the tail, so it is never the popped row.
        """
        while len(self._main) > self._main_cap:
            self._pop_main()

    def _release(self, slot: int) -> None:
        """Drop ``slot``'s row and hand the pool index back."""
        del self._slot[self._key[slot]]
        self._freq[slot] = 0
        self._queue[slot] = self._UNQUEUED
        self._state[slot] = self._FREE
        self._free.append(slot)

    def _demote_ghost(self, key: int) -> None:
        self._ghost[key] = None
        while len(self._ghost) > self._ghost_cap:
            self._ghost.pop(next(iter(self._ghost)))

    def _pop_small(self) -> None:
        """Recycle S's front: threshold credit promotes into M keeping its
        counter, anything else demotes to a ghost and frees its slot.

        Promotions loop (the C's ``while !has_evicted``): only a demotion --
        or an exhausted S -- frees a pool slot for the caller.
        """
        while self._small:
            slot = self._small.popleft()
            if self._freq[slot] >= self._MOVE_TO_MAIN_THRESHOLD and self._main_cap:
                self._queue[slot] = self._MAIN
                self._main.append(slot)
                # Overfilling M is paid back now: the C's next eviction would
                # pick M anyway (``occupied > size``), so settling at the
                # budget is equivalent and keeps the queues testable.
                while len(self._main) > self._main_cap:
                    self._pop_main()
                continue
            key = self._key[slot]
            self._release(slot)
            self._demote_ghost(key)
            return

    def _pop_main(self) -> None:
        """2-bit clock over M: front rows with credit are reinserted with
        ``min(freq, 3) - 1`` until one without credit leaves."""
        while self._main:
            slot = self._main.popleft()
            freq = self._freq[slot]
            if freq >= 1:
                self._freq[slot] = min(freq, self._MAX_FREQ) - 1
                self._main.append(slot)
                continue
            self._release(slot)
            return

    def _evict_one(self, *, main: bool = False) -> None:
        """One eviction, choosing the source exactly as S3FIFO_evict does.

        ``main`` is the newcomer's planned destination: the C evaluates
        ``occupied_main > size_main`` with the pending insert counted, so an
        M admission at full M budget picks M, not S.
        """
        self._has_evicted = True
        if len(self._main) + main > self._main_cap or not self._small:
            self._pop_main()
        else:
            self._pop_small()

    def get(self, key: int) -> int | None:
        # A pure lookup: never consumes the ghost entry. In the C, the ghost
        # is removed by the miss path's find() only to raise hit_on_ghost, and
        # admission (insert) is where it is consumed; a lookup must not erase
        # the ghost fast path.
        slot = self._slot.get(key)
        if slot is None:
            return None
        if self._freq[slot] < self._MAX_FREQ:
            # One counter table serves both queues: S reads it to decide
            # promotion, M reads it to decide eviction or reinsertion.
            self._freq[slot] += 1
        return slot

    def is_ready(self, key: int) -> bool:
        """Whether ``key`` holds a committed (READY) row."""
        return key in self._slot

    def reserve(self, key: int) -> int | None:
        """Claim a loading slot for ``key``; ``None`` if it cannot be taken.

        A row already committed or already loading is a no-op (``None``), as is
        one that cannot be evicted for without stripping the last ready row.
        The slot joins neither queue, so no pop can reclaim an in-flight
        prefetch and no lookup can read a half-filled row.
        """
        if key in self._slot or key in self._loading:
            return None
        slot = self._claim_for_loading()
        if slot is None:
            return None
        self._key[slot] = key
        self._freq[slot] = 0
        self._queue[slot] = self._UNQUEUED
        self._state[slot] = self._LOADING
        self._loading[key] = slot
        return slot

    def _claim_for_loading(self) -> int | None:
        if self._free:
            return self._free.pop()
        if self._used < self.capacity:
            slot = self._used
            self._used += 1
            return slot
        if len(self._slot) <= 1:
            # Keep at least one committed row evictable so the synchronous
            # read path can never starve behind a queue of in-flight prefetches.
            return None
        self._evict_one()
        return self._free.pop() if self._free else None

    def confirm(self, slot: int) -> int:
        """Promote a loading slot to ready; idempotent if already committed."""
        key = self._key[slot]
        if self._state[slot] != self._LOADING:
            return slot
        self._loading.pop(key, None)
        # Room is made against committed rows only -- this slot is already
        # owned, and it sits in no queue. The ghost decision uses the C's
        # find()-time rule (a member of G admits to M), consumed at admission
        # exactly as the C's hit_on_ghost flag.
        main, ghost = self._admit(key)
        self._bind_loaded(slot, key, main=main)
        if ghost:
            self._ghost.pop(key, None)
        self._settle_main_budget()
        return slot

    def _bind_loaded(self, slot: int, key: int, *, main: bool) -> None:
        """Attach an already-owned loading slot as a ready row in one queue."""
        self._slot[key] = slot
        self._key[slot] = key
        self._freq[slot] = 0
        self._state[slot] = self._READY
        self._queue[slot] = self._MAIN if main else self._SMALL
        (self._main if main else self._small).append(slot)

    def _admit(
        self, key: int, *, hit_on_ghost: bool | None = None
    ) -> tuple[bool, bool]:
        """Plan one admission: evict for space, then resolve the queue.

        Returns ``(main, ghost_hit)``. Mirrors S3FIFO.c ordering: the ghost
        fast path is the parent's capacity loop evaluated with the pending
        insert counted into M (``occupied_main > size_main`` picks M); the
        plain path evicts for an S admission and only then reads the
        ``has_evicted`` gate, as insert() runs after the loop.
        """
        if hit_on_ghost is None:
            hit_on_ghost = key in self._ghost
        if hit_on_ghost and self._main_cap > 0:
            self._prepare_admission(main=True)
            return True, True
        self._prepare_admission(main=False)
        return self._ghost_admits_main(key) and self._main_cap > 0, hit_on_ghost

    def cancel(self, slot: int) -> None:
        """Drop a loading reservation, returning the slot to the free pool."""
        if self._state[slot] != self._LOADING:
            return
        self._loading.pop(self._key[slot], None)
        self._freq[slot] = 0
        self._queue[slot] = self._UNQUEUED
        self._state[slot] = self._FREE
        self._free.append(slot)

    def insert(self, key: int, *, hit_on_ghost: bool | None = None) -> int | None:
        """Admit ``key`` as a fresh row; ``None`` when no slot can be taken.

        ``hit_on_ghost`` is the C's latched flag: ``None`` (the direct API)
        reads the ghost set here, while :meth:`plan` passes the value observed
        at lookup time, before any eviction may have re-demoted this key.
        """
        if key in self._loading:
            # The synchronous path is finishing a row a prefetch is also
            # fetching: adopt the reserved slot so the two never own separate
            # copies of the same id.
            return self.confirm(self._loading[key])
        if key in self._slot:
            raise KeyError("row is already cached")
        main, ghost = self._admit(key, hit_on_ghost=hit_on_ghost)
        if ghost:
            self._ghost.pop(key, None)
        slot = self._free.pop() if self._free else None
        if slot is None:
            if self._used < self.capacity:
                slot = self._used
                self._used += 1
            else:
                return None
        self._bind(slot, key, main=main)
        self._settle_main_budget()
        return slot

    def _ghost_admits_main(self, key: int) -> bool:
        """The C's non-ghost rule: S already full before the first eviction."""
        return not self._has_evicted and len(self._small) >= self._small_cap

    def plan(self, ids) -> RowPlan:
        """Resolve one batch; duplicates in the batch fetch and store once."""
        hit_pos: list[int] = []
        hit_slot: list[int] = []
        miss_ids: list[int] = []
        miss_pos: list[int] = []
        miss_row: list[int] = []
        fetched: dict[int, int] = {}
        hit_on_ghost: dict[int, bool] = {}
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
                # Latch the C's find() flag at lookup time: an eviction during
                # this batch's insertions may otherwise re-demote the key and
                # grant it the ghost fast path it did not earn.
                hit_on_ghost[key] = key in self._ghost
            miss_pos.append(position)
            miss_row.append(row)
        self.hits += len(hit_pos)
        self.misses += len(miss_pos)
        # Insert after every lookup, so this batch's hits raise their counters
        # before any of its misses recycles a queue. A slot reassigned within
        # the batch keeps only its final owner. A miss that cannot obtain a
        # slot (every slot still loading) is simply not cached this time; its
        # staging is unaffected, since misses are staged from the fetched
        # planes rather than from the pool.
        stored: dict[int, int] = {}
        for row, key in enumerate(miss_ids):
            slot = self.insert(key, hit_on_ghost=hit_on_ghost[key])
            if slot is not None:
                stored[slot] = row
        return RowPlan(
            hit_pos,
            hit_slot,
            miss_ids,
            miss_pos,
            miss_row,
            list(stored),
            list(stored.values()),
        )


# XXH64 primes; libCacheSim's minimalIncrementCBF hashes each obj id as an
# 8-byte little-endian buffer with XXH64 under two fixed seeds
# (minimalIncrementCBF.c:88-91, hash.h:15-16: 0x12345678 / 0x12344321).
_XXH64_P1 = 0x9E3779B185EBCA87
_XXH64_P2 = 0xC2B2AE3D27D4EB4F
_XXH64_P3 = 0x165667B19E3779F9
_XXH64_P4 = 0x85EBCA77C2B2AE63
_XXH64_P5 = 0x27D4EB2F165667C5
_U64 = (1 << 64) - 1
_U32 = 0xFFFFFFFF


def _xxh64_8(value: int, seed: int) -> int:
    """XXH64 of the eight little-endian bytes of ``value``.

    The 8-byte length is the only input shape the filter sees (``obj_id_t``
    is ``uint64_t``, config.h:63), so this is the exact tail-only branch of
    the reference algorithm: single accumulator ``seed + P5``, one 8-byte
    round, then the avalanche. Verified against libCacheSim's vendored
    xxhash.c on boundary ids (0, 2**64-1, 2**63+5, ...).
    """
    acc = (seed + _XXH64_P5 + 8) & _U64
    k = (value * _XXH64_P2) & _U64
    k = ((k << 31) | (k >> 33)) & _U64
    k = (k * _XXH64_P1) & _U64
    acc ^= k
    acc = ((acc << 27) | (acc >> 37)) & _U64
    acc = (acc * _XXH64_P1 + _XXH64_P4) & _U64
    acc ^= acc >> 33
    acc = (acc * _XXH64_P2) & _U64
    acc ^= acc >> 29
    acc = (acc * _XXH64_P3) & _U64
    acc ^= acc >> 32
    return acc


class WTinyLfuIndex:
    """Fixed-capacity W-TinyLFU (Einziger et al., WWW'18) over the same slot pool.

    A drop-in sibling of :class:`SieveIndex` and :class:`S3FifoIndex`: same
    constructor and ``get``/``insert``/``plan``/``is_ready``/``reserve``/
    ``confirm``/``cancel`` contract, one payload pool, slots owned by their
    key until a pop releases them.

    The bookkeeping mirrors libCacheSim's ``WTinyLFU.c`` under its
    ``DEFAULT_PARAMS`` (``main-cache=SLRU,window-size=0.01``, WTinyLFU.c:43),
    which is what ``cachesim <trace> txt WTinyLFU <size>B`` runs, so parity
    with the C -- quirks included -- is the contract:

    * window = ``int(capacity * 0.01)`` rows (double multiply, truncated as
      in WTinyLFU.c:112-115), an LRU; main = the rest, a two-segment SLRU
      with ``seg-size=1:4`` -- probation ``int(main/5)``, protected
      ``int(4*main/5)``, each truncated in double arithmetic (SLRU.c:483-491)
      so prob+prot may sit a row below main and overflow is policed by the
      composite's fullness test, not the segment budgets.
    * A miss enters the window and the filter immediately (WTinyLFU.c:256-269).
      A *main* hit feeds the filter and ticks the decay clock; a window hit
      only re-orders the LRU -- neither the filter nor the counter is touched
      (WTinyLFU.c:225-254). The clock decays every ``32 * main_cap`` **main
      hits**; insertions and eviction adds never reach it.
    * One composite evict (WTinyLFU.c:277-358) drains the window's oldest
      rows into main while main has overall room -- a drain step frees no
      slot and adds the moved row to the filter again (:350 runs per step
      iteration) -- then, once main is full, makes one TinyLFU decision:
      window-tail estimate vs the SLRU victim's, *strictly greater* wins
      promotion, ties keep the main victim (:318-323). An empty window
      evicts straight from main with no filter add (:352-356).
    * The SLRU victim is the first non-empty segment's tail, probation first
      (SLRU.c:340-353); a protected hit moves to its head, a probation hit
      promotes and then cools the entered segment back to budget, cascading
      into probation evictions (SLRU.c:264-276, 524-549). Placement of an
      admitted row is the lowest segment with room, and when neither fits
      its own budget the C still bills it to probation (SLRU.c:296-313).
    * The filter is a minimal-increment CBF sized for ``main_cap`` entries at
      0.1% error (WTinyLFU.c:172-179, minimalIncrementCBF.c:29-73): 10
      uint32 counters per row probed by ``(a + i*b) mod 2**32`` over
      ``counter_num`` cells, add bumps every probe of the minimum, decay
      right-shifts all cells. Probe plans ride on the owning slot and are
      recomputed when a new row takes it.
    * ``can_insert`` is part of the C too (WTinyLFU.c:423-434, SLRU.c:510-515):
      a row must fit the window *and* the probation segment alone. A capacity
      truncated to a zero window (< 100 rows here) or zero probation turns
      every miss into an uncached miss -- cachesim's degenerate 1.0000 miss
      ratio at 50 rows, replicated rather than papered over.

    Loading (in-flight prefetch) slots are separate state, exactly as in the
    siblings: a reserved key is invisible to ``get``, ``plan`` and
    ``is_ready``, its slot joins no queue, and no queue pop can reclaim one.
    A confirmed prefetch enters as a fresh row: window head plus a filter
    add, the same admission a miss takes.
    """

    _FREE = 0
    _LOADING = 1
    _READY = 2

    _UNQUEUED = 0
    _WINDOW = 1
    _PROBATION = 2
    _PROTECTED = 3

    def __init__(self, capacity: int) -> None:
        import math

        capacity = operator.index(capacity)
        if capacity <= 0:
            raise ValueError("row cache capacity must be positive")
        self.capacity = capacity
        # WTinyLFU.c:112-115 truncates the 0.01 window share in double
        # arithmetic and hands main the remainder; SLRU.c:483-491 truncates
        # each "seg-size=1:4" share of main the same way.
        self._window_cap = int(capacity * 0.01)
        self._main_cap = capacity - self._window_cap
        self._prob_cap = int((1.0 / 5.0) * self._main_cap)
        self._prot_cap = int((4.0 / 5.0) * self._main_cap)
        # minimalIncrementCBF_init(CBF, main_cap, 0.001), WTinyLFU.c:172-179
        # and minimalIncrementCBF.c:29-44: bpe = -ln(error)/ln(2)^2, d =
        # ceil(ln(2)*bpe) probes, counter_num = ceil(bpe*entries) clamped to
        # INT_MAX and floored at 2*d.
        bpe = -math.log(0.001) / 0.480453013918201  # ln(2)^2
        self._hashes = math.ceil(0.693147180559945 * bpe)  # ln(2)
        self._counter_num = min(math.ceil(bpe * self._main_cap), 2**31 - 1)
        if self._counter_num < self._hashes:
            self._counter_num = 2 * self._hashes
        self._bf = [0] * self._counter_num
        # WTinyLFU.c:170,246-250: the decay clock is ticked by main-cache
        # hits only -- window hits and inserts never reach it.
        self._max_request_num = 32 * self._main_cap
        self._request_counter = 0
        self._slot: dict[int, int] = {}
        self._key = [0] * capacity
        # Per-slot state: _FREE / _LOADING / _READY.
        self._state = bytearray(capacity)
        # Per-slot queue tag: _UNQUEUED (loading) or one of the three lists.
        self._loc = bytearray(capacity)
        # Filter probe plan of the row currently owning each slot; recomputed
        # when a new row binds the slot, so memory stays bounded by capacity.
        self._probes: list[tuple[int, ...] | None] = [None] * capacity
        # key -> slot for rows a prefetch reserved but has not committed. They
        # are deliberately kept out of ``_slot`` so ``get`` and ``plan`` miss.
        self._loading: dict[int, int] = {}
        # Slots freed by a queue pop or cancel(); handed out before ``_used``
        # grows, so recycled rows and dropped prefetches do not leak capacity.
        self._free: list[int] = []
        # One shared pair of link arrays serves the three doubly linked queues
        # (0=window, 1=probation, 2=protected); every ready slot sits in
        # exactly one, head is most recent, tail is the eviction end, as in
        # libCacheSim's mmo lists.
        self._prev = [-1] * capacity
        self._next = [-1] * capacity
        self._head = [-1, -1, -1]
        self._tail = [-1, -1, -1]
        self._nq = [0, 0, 0]
        self._used = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        # Occupied slots, not the high-water mark: a cancelled loading slot
        # sits in the free pool and must not count as resident.
        return len(self._slot) + len(self._loading)

    def __contains__(self, key: int) -> bool:
        return key in self._slot

    # -- minimal-increment filter ---------------------------------------------

    def _probes_for(self, key: int) -> tuple[int, ...]:
        # minimalIncrementCBF.c:88-91: two 32-bit-truncated XXH64 hashes
        # folded Kirsch-Mitzenmacher style over the counter array; obj ids
        # live in the uint64 domain the C buffer has.
        a = _xxh64_8(key & _U64, 0x12345678) & _U32
        b = _xxh64_8(key & _U64, 0x12344321) & _U32
        n = self._counter_num
        return tuple(((a + i * b) & _U32) % n for i in range(self._hashes))

    def _cbf_add(self, probes: tuple[int, ...]) -> None:
        # minimalIncrementCBF.c:113-125: the estimate is the per-probe
        # minimum, and an add bumps *every* probe by one -- the minimal
        # increment keeps all probes at their old minimum plus one -- unless
        # that minimum already saturated a uint32.
        bf = self._bf
        lo = min(map(bf.__getitem__, probes))
        if lo < _U32:
            for x in probes:
                bf[x] += 1

    def _cbf_estimate(self, probes: tuple[int, ...]) -> int:
        return min(map(self._bf.__getitem__, probes))

    # -- queue helpers ----------------------------------------------------------

    def _push_head(self, q: int, slot: int) -> None:
        head = self._head[q]
        self._prev[slot] = -1
        self._next[slot] = head
        if head >= 0:
            self._prev[head] = slot
        else:
            self._tail[q] = slot
        self._head[q] = slot
        self._nq[q] += 1

    def _unlink(self, q: int, slot: int) -> None:
        p = self._prev[slot]
        n = self._next[slot]
        if p >= 0:
            self._next[p] = n
        else:
            self._head[q] = n
        if n >= 0:
            self._prev[n] = p
        else:
            self._tail[q] = p
        self._prev[slot] = -1
        self._next[slot] = -1
        self._nq[q] -= 1

    def _release(self, slot: int) -> None:
        """Drop ``slot``'s row (already unlinked) and hand the pool index back."""
        del self._slot[self._key[slot]]
        self._loc[slot] = self._UNQUEUED
        self._probes[slot] = None
        self._state[slot] = self._FREE
        self._free.append(slot)

    def _slru_place(self, slot: int) -> None:
        # SLRU.c:296-313: the lowest segment with room takes the row at its
        # head; when neither fits its own budget the C bills it to probation
        # anyway, because the composite only calls this once main's *total*
        # has room, which keeps SLRU's own evict loop from triggering.
        if self._nq[1] + 1 <= self._prob_cap:
            q = 1
        elif self._nq[2] + 1 <= self._prot_cap:
            q = 2
        else:
            q = 1
        self._loc[slot] = self._PROBATION if q == 1 else self._PROTECTED
        self._push_head(q, slot)

    def _cool(self, q: int) -> None:
        # SLRU.c:524-549: push ``q``'s tail one segment down; the bottom
        # segment (probation) pays for its own overflow with a real eviction.
        slot = self._tail[q]
        self._unlink(q, slot)
        if q == 2:
            self._loc[slot] = self._PROBATION
            self._push_head(1, slot)
            while self._nq[1] > self._prob_cap:
                self._cool(1)
        else:
            self._release(slot)

    # -- the composite eviction step ----------------------------------------------

    def _evict_step(self) -> bool:
        """One WTinyLFU_evict call (WTinyLFU.c:277-358).

        Drains the window's oldest rows into main while main has overall
        room -- a drain moves the row, frees no slot, and re-adds it to the
        filter -- and once main is full makes exactly one TinyLFU decision
        that releases one slot. Returns whether a slot was freed; ``False``
        means every ready row is already gone (only possible with loading
        rows holding the pool).
        """
        while True:
            if self._nq[0] > 0:
                w_slot = self._tail[0]
                w_key = self._key[w_slot]
                wp = self._probes[w_slot]
                if wp is None:
                    wp = self._probes[w_slot] = self._probes_for(w_key)
                if self._nq[1] + self._nq[2] + 1 <= self._main_cap:
                    # WTinyLFU.c:294-311: main has room, the window's oldest
                    # row just moves in and the step keeps draining.
                    self._unlink(0, w_slot)
                    self._slru_place(w_slot)
                    self._cbf_add(wp)
                    continue
                # WTinyLFU.c:313-348: main is full; the window tail competes
                # with the SLRU victim -- first non-empty segment's tail,
                # probation first (SLRU.c:340-353). Strictly greater wins,
                # ties keep the main victim (:318-323).
                if self._nq[1] > 0:
                    mv_q = 1
                elif self._nq[2] > 0:
                    mv_q = 2
                else:
                    return False
                mv_slot = self._tail[mv_q]
                mv_key = self._key[mv_slot]
                mp = self._probes[mv_slot]
                if mp is None:
                    mp = self._probes[mv_slot] = self._probes_for(mv_key)
                if self._cbf_estimate(wp) > self._cbf_estimate(mp):
                    self._unlink(mv_q, mv_slot)
                    self._release(mv_slot)
                    self._unlink(0, w_slot)
                    self._slru_place(w_slot)
                else:
                    self._unlink(0, w_slot)
                    self._release(w_slot)
                # The compared victim is added to the filter at the end of
                # every window step, whichever side won (:349-351).
                self._cbf_add(wp)
                return True
            # WTinyLFU.c:352-356: an empty window evicts straight from main,
            # with no filter add at all.
            if self._nq[1] > 0:
                slot = self._tail[1]
                self._unlink(1, slot)
                self._release(slot)
                return True
            if self._nq[2] > 0:
                slot = self._tail[2]
                self._unlink(2, slot)
                self._release(slot)
                return True
            return False

    def _make_room(self) -> None:
        # cache_get_base (cache.c:269-273) evicts one slot per step until the
        # incoming row fits; a step that frees nothing means only loading
        # rows hold the pool, and the caller's claim will fail.
        while len(self) + 1 > self.capacity:
            if not self._evict_step():
                return

    def _admissible(self) -> bool:
        # WTinyLFU_can_insert (WTinyLFU.c:423-434) plus SLRU_can_insert
        # (SLRU.c:510-515): a row enters through the window and must fit the
        # probation segment alone. A row is one slot here, so a truncated
        # budget of zero -- window for capacity < 100, probation for a tiny
        # main share -- makes every miss an uncached miss, exactly as
        # cachesim's degenerate 1.0000 miss ratio at 50 rows.
        return self._window_cap >= 1 and self._prob_cap >= 1

    # -- public contract -----------------------------------------------------------

    def get(self, key: int) -> int | None:
        slot = self._slot.get(key)
        if slot is None:
            return None
        loc = self._loc[slot]
        if loc == self._WINDOW:
            # A window hit only re-orders the LRU: WTinyLFU.c:241 gates every
            # filter touch on a *main* hit.
            self._unlink(0, slot)
            self._push_head(0, slot)
            return slot
        if loc == self._PROBATION:
            # SLRU.c:267-276: promote, then cool the entered segment back to
            # its budget; cooling protected can cascade into evicting the
            # probation tail.
            self._unlink(1, slot)
            self._loc[slot] = self._PROTECTED
            self._push_head(2, slot)
            while self._nq[2] > self._prot_cap:
                self._cool(2)
        else:
            self._unlink(2, slot)
            self._push_head(2, slot)
        wp = self._probes[slot]
        if wp is None:
            wp = self._probes[slot] = self._probes_for(key)
        # WTinyLFU.c:241-250: a main hit feeds the filter and ticks the
        # decay clock; decay right-shifts every counter (the C resets the
        # clock first, so a boundary add is counted against the halved
        # state).
        self._cbf_add(wp)
        self._request_counter += 1
        if self._request_counter >= self._max_request_num:
            self._request_counter = 0
            self._bf = [v >> 1 for v in self._bf]
        return slot

    def is_ready(self, key: int) -> bool:
        """Whether ``key`` holds a committed (READY) row."""
        return key in self._slot

    def reserve(self, key: int) -> int | None:
        """Claim a loading slot for ``key``; ``None`` if it cannot be taken.

        A row already committed or already loading is a no-op (``None``), as is
        one that cannot be evicted for without stripping the last ready row.
        The slot joins no queue, so no pop can reclaim an in-flight prefetch
        and no lookup can read a half-filled row.
        """
        if key in self._slot or key in self._loading:
            return None
        slot = self._claim_for_loading()
        if slot is None:
            return None
        self._key[slot] = key
        self._loc[slot] = self._UNQUEUED
        self._probes[slot] = None
        self._state[slot] = self._LOADING
        self._loading[key] = slot
        return slot

    def _claim_for_loading(self) -> int | None:
        if self._free:
            return self._free.pop()
        if self._used < self.capacity:
            slot = self._used
            self._used += 1
            return slot
        if len(self._slot) <= 1:
            # Keep at least one committed row evictable so the synchronous
            # read path can never starve behind a queue of in-flight prefetches.
            return None
        self._evict_step()
        return self._free.pop() if self._free else None

    def confirm(self, slot: int) -> int:
        """Promote a loading slot to ready; idempotent if already committed."""
        key = self._key[slot]
        if self._state[slot] != self._LOADING:
            return slot
        self._loading.pop(key, None)
        if not self._admissible():
            # The C never caches a row under a degenerate budget; a prefetched
            # row is no exception, and its slot goes straight back.
            self._state[slot] = self._FREE
            self._free.append(slot)
            return slot
        # A confirmed prefetch is a fresh row: it enters through the window
        # and feeds the filter exactly like a miss (WTinyLFU.c:256-269). No
        # eviction here -- ``reserve`` already paid for this slot, and with
        # it counted as loading, binding it ready cannot breach capacity.
        self._slot[key] = slot
        self._key[slot] = key
        self._state[slot] = self._READY
        self._loc[slot] = self._WINDOW
        probes = self._probes_for(key)
        self._probes[slot] = probes
        self._push_head(0, slot)
        self._cbf_add(probes)
        return slot

    def cancel(self, slot: int) -> None:
        """Drop a loading reservation, returning the slot to the free pool."""
        if self._state[slot] != self._LOADING:
            return
        self._loading.pop(self._key[slot], None)
        self._loc[slot] = self._UNQUEUED
        self._probes[slot] = None
        self._state[slot] = self._FREE
        self._free.append(slot)

    def insert(self, key: int) -> int | None:
        if key in self._loading:
            # The synchronous path is finishing a row a prefetch is also
            # fetching: adopt the reserved slot so the two never own separate
            # copies of the same id.
            return self.confirm(self._loading[key])
        if key in self._slot:
            raise KeyError("row is already cached")
        if not self._admissible():
            return None
        self._make_room()
        slot = self._free.pop() if self._free else None
        if slot is None:
            if self._used < self.capacity:
                slot = self._used
                self._used += 1
            else:
                return None
        self._slot[key] = slot
        self._key[slot] = key
        self._state[slot] = self._READY
        self._loc[slot] = self._WINDOW
        probes = self._probes_for(key)
        self._probes[slot] = probes
        self._push_head(0, slot)
        # WTinyLFU.c:256-269: every missed row enters through the window and
        # feeds the filter immediately.
        self._cbf_add(probes)
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
        # Insert after every lookup, so this batch's hits move and feed the
        # filter before any of its misses evicts, keeping the C's per-request
        # find-then-evict-then-insert order at batch=1. A slot reassigned
        # within the batch keeps only its final owner. A miss that cannot
        # obtain a slot (every slot still loading) is simply not cached this
        # time; its staging is unaffected, since misses are staged from the
        # fetched planes rather than from the pool.
        stored: dict[int, int] = {}
        for row, key in enumerate(miss_ids):
            slot = self.insert(key)
            if slot is not None:
                stored[slot] = row
        return RowPlan(
            hit_pos,
            hit_slot,
            miss_ids,
            miss_pos,
            miss_row,
            list(stored),
            list(stored.values()),
        )
