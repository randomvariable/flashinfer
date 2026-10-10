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
``DiskRowCache`` selects between the two policies with
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
    """Fixed-capacity S3-FIFO (Yang et al., SOSP'23) over the same slot pool.

    A drop-in sibling of :class:`SieveIndex`: the constructor and the
    ``get``/``insert``/``plan``/``is_ready``/``reserve``/``confirm``/``cancel``
    contract match, so ``DiskRowCache`` may run either replacement policy over
    one payload pool. Slots are pool indices ``0..capacity-1`` and the queues
    are sequences of slots, so a slot handed out by ``insert`` keeps addressing
    the row its key owns.

    Residency is earned in stages. A new row joins the small FIFO ``S`` (10% of
    capacity, never empty) carrying a 2-bit counter that every hit raises, up to
    three. When ``S`` recycles its front, a counter buys the row a promotion
    into the main FIFO ``M`` (the rest of the capacity); no counter demotes it
    to the ghost list ``G`` -- keys only, no payload, ``S``'s budget. ``M``
    recycles its front too but reinserts a row that still holds credit instead
    of dropping it, so a hot main-queue row cycles rather than leaves. A later
    miss on a ghost is admitted straight into ``M``: a row already proven worth
    keeping does not pay for a second stay in ``S``. Promotion resets the
    counter and hits raise it in either queue, so ``M``'s reinsertion is driven
    by main-queue reuse rather than by leftover small-queue credit.

    Loading (in-flight prefetch) slots are separate state, exactly as in
    :class:`SieveIndex`: a reserved key is invisible to ``get``, ``plan`` and
    ``is_ready``, its slot joins no queue, and no queue pop can reclaim one.
    A confirmed prefetch is admitted as a fresh row -- into ``S``, or straight
    into ``M`` when its key is a ghost.
    """

    _FREE = 0
    _LOADING = 1
    _READY = 2

    _UNQUEUED = 0
    _SMALL = 1
    _MAIN = 2

    # 2-bit saturation, and the percentage of capacity S -- and so G -- gets.
    _MAX_FREQ = 3
    _SMALL_SHARE = 10

    def __init__(self, capacity: int) -> None:
        capacity = operator.index(capacity)
        if capacity <= 0:
            raise ValueError("row cache capacity must be positive")
        self.capacity = capacity
        # A 10% share rounds up to at least one row, so even a tiny cache has a
        # small queue to filter one-hit rows; M takes what is left. A capacity
        # of one leaves M with no budget at all, and the pops below treat an
        # empty budget as "no promotions, no ghost fast-path" rather than as a
        # queue that may transiently hold a row nobody owns.
        self._small_cap = max(1, capacity * self._SMALL_SHARE // 100)
        self._main_cap = capacity - self._small_cap
        self._ghost_cap = self._small_cap
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
        # Ghost keys in demotion order: a dict rather than a deque because
        # every miss asks whether its key is a ghost.
        self._ghost: dict[int, None] = {}
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

    def _release(self, slot: int) -> None:
        """Drop ``slot``'s row and hand the pool index back."""
        del self._slot[self._key[slot]]
        self._freq[slot] = 0
        self._queue[slot] = self._UNQUEUED
        self._state[slot] = self._FREE
        self._free.append(slot)

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
        # The pool is exhausted. M gives up the slot first: it holds the bulk
        # of the cache, and its reinsertion sweep is the frequency filter that
        # keeps the rows worth keeping. S is the fallback, and its recycled row
        # lives on as a ghost. Either pop lands a slot in ``_free``: S always
        # frees one, directly or through the M sweep a promotion triggers.
        for pop in (self._pop_main, self._pop_small):
            pop()
            if self._free:
                return self._free.pop()
        return None

    def _pop_small(self) -> None:
        """Recycle the front of S: credit promotes into M, none demotes to G.

        Counter credit buys the row a promotion into M, which may itself evict
        from M; no credit -- or no room in M -- demotes it to a ghost and frees
        its slot. Either way S loses exactly one row and the caller gains a
        pool slot, which is what lets ``_make_room`` size the queues by budget.
        """
        if not self._small:
            return
        slot = self._small.popleft()
        if self._freq[slot] and self._main_cap:
            self._freq[slot] = 0
            self._queue[slot] = self._MAIN
            self._main.append(slot)
            self._enforce_main()
            return
        key = self._key[slot]
        self._release(slot)
        self._ghost[key] = None
        while len(self._ghost) > self._ghost_cap:
            self._ghost.pop(next(iter(self._ghost)))

    def _pop_main(self) -> None:
        """Evict the front of M, reinserting rows that still hold credit.

        Every reinsertion spends one unit of a row's counter, and nothing
        raises a counter while this runs, so the sweep cannot cycle forever:
        it ends on the first row with no credit left, freeing its slot.
        """
        while self._main:
            slot = self._main.popleft()
            if self._freq[slot]:
                self._freq[slot] -= 1
                self._main.append(slot)
                continue
            self._release(slot)
            return

    def _enforce_main(self) -> None:
        while len(self._main) > self._main_cap:
            self._pop_main()

    def _make_room(self, *, main: bool) -> None:
        # Recycle a member of the target queue *before* the new row joins it.
        # Appending first and sweeping after could hand the new row's own slot
        # back as the victim -- M's reinsertion rotates the queue -- and a slot
        # handed to ``DiskRowCache`` must stay owned by its key for as long as
        # the batch reads it. Callers ask for M only when M has a budget, so a
        # queue at its budget always has a front to recycle.
        if main:
            if len(self._main) >= self._main_cap:
                self._pop_main()
        elif len(self._small) >= self._small_cap:
            self._pop_small()

    def get(self, key: int) -> int | None:
        slot = self._slot.get(key)
        if slot is not None and self._freq[slot] < self._MAX_FREQ:
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
        slot = self._claim(protect_ready=True)
        if slot is None:
            return None
        self._key[slot] = key
        self._freq[slot] = 0
        self._queue[slot] = self._UNQUEUED
        self._state[slot] = self._LOADING
        self._loading[key] = slot
        return slot

    def confirm(self, slot: int) -> int:
        """Promote a loading slot to ready; idempotent if already committed."""
        key = self._key[slot]
        if self._state[slot] != self._LOADING:
            return slot
        self._loading.pop(key, None)
        main = key in self._ghost and self._main_cap > 0
        self._make_room(main=main)
        self._ghost.pop(key, None)
        self._bind(slot, key, main=main)
        return slot

    def cancel(self, slot: int) -> None:
        """Drop a loading reservation, returning the slot to the free pool."""
        if self._state[slot] != self._LOADING:
            return
        self._loading.pop(self._key[slot], None)
        self._freq[slot] = 0
        self._queue[slot] = self._UNQUEUED
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
        main = key in self._ghost and self._main_cap > 0
        self._make_room(main=main)
        slot = self._claim()
        if slot is None:
            return None
        self._ghost.pop(key, None)
        self._bind(slot, key, main=main)
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
        # Insert after every lookup, so this batch's hits raise their counters
        # before any of its misses recycles a queue. A slot reassigned within
        # the batch keeps only its final owner. A miss that cannot obtain a
        # slot (every slot still loading) is simply not cached this time; its
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
