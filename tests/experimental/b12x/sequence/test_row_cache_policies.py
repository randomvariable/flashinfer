"""Pure-Python contract tests for the row-cache policy indexes.

These exercise :class:`WTinyLfuIndex` bookkeeping without a GPU: the class is
stdlib-only, like its SIEVE and S3-FIFO siblings, so this file runs on any
workstation. The hard-coded expectations are taken from libCacheSim's
``cachesim`` WTinyLFU replay of the same request sequences (per-request hit
vectors and end-of-trace membership probed through the C; see
``probe/wtinylfu_parity.py`` for the full-trace gate). Where a test reaches
into privates it cites the C line it mirrors, not a whim of this module.
"""

from __future__ import annotations

import random

import pytest

# rows 1..201 each once, then 199, 202, 203, 204, 205, and re-reads 1, 2, 1.
MICRO_SEQ = list(range(1, 202)) + [199, 202, 203, 204, 205, 1, 2, 1]
# Per-request hit vector as reported by ``cachesim micro.txt txt WTinyLFU 200B``:
# every request misses except the final two (the re-reads of 2 and 1).
MICRO_HITS = "0" * 207 + "11"
# Membership of rows 1..210 after the sequence, likewise taken from cachesim
# (a row is resident iff a fresh replay of MICRO_SEQ + [row] hits its appended
# probe): everything from 1..199 except 3, plus 205; 199 re-admitted and then
# promoted to main, 200..204 all demoted out of the window.
MICRO_RESIDENT = [1, 2, *range(4, 200), 205]


def _wtl(capacity):
    from b12x.sequence._shared.row_cache import WTinyLfuIndex

    return WTinyLfuIndex(capacity)


def _replay(index, seq):
    hits = []
    for oid in seq:
        plan = index.plan((oid,))
        hits.append("1" if plan.hit_pos else "0")
    return "".join(hits)


# ---------------------------------------------------------------------------
# window admission + the TinyLFU demote/promote decision on a known micro trace
# ---------------------------------------------------------------------------


def test_wtinylfu_micro_trace_matches_cachesim_reference():
    """Window admission, drain, and both TinyLFU outcomes on one known trace.

    Replays MICRO_SEQ at capacity 200 (window 2, main 198, probation 39,
    protected 158 -- each share truncated in double arithmetic exactly as
    WTinyLFU.c:112-115 and SLRU.c:483-491 do) and asserts the exact
    per-request hit vector, the counters, and the final membership the C
    reference produced. The narrative, cross-checked against the C:

    * rows 1..201 enter through the window and the filter immediately
      (WTinyLFU.c:256-269); nothing overflows the pool until row 201, whose
      evict step drains rows 1..198 into main (each drain add, :350) before
      the window tail 199 -- estimate 1 -- loses to probation tail 1 --
      estimate 2 -- and is demoted out of the cache (:339-347).
    * the re-request of 199 re-admits it (the compared-victim add at :349-351
      plus the insert add take it to estimate 3); at row 203's competition it
      beats probation tail 1 (2), so 199 is promoted into main and 1 leaves
      -- the promotion path (:318-337).
    * rows 200..204 all lose as window tails (estimate 1 vs 2); 205 survives
      the trace in the window beside the re-inserted 1.
    * the late hit on 2 is a probation hit: it promotes to protected and the
      cooling cascade (SLRU.c:264-276, 524-549) evicts probation tail 3. The
      late hit on 1 is a window hit: it re-orders the LRU only.
    """
    index = _wtl(200)
    assert _replay(index, MICRO_SEQ) == MICRO_HITS
    assert index.hits == 2 and index.misses == len(MICRO_SEQ) - 2
    assert sorted(index._slot) == MICRO_RESIDENT
    assert index._nq == [2, 39, 158]


def test_wtinylfu_window_rows_do_not_feed_the_filter():
    """A window hit only re-orders the LRU (WTinyLFU.c:225-254)."""
    index = _wtl(200)
    index.plan((7,))
    slot = index._slot[7]
    assert index._loc[slot] == index._WINDOW
    before = index._cbf_estimate(index._probes[slot])
    for _ in range(3):
        index.plan((7,))  # window hits
    assert index._cbf_estimate(index._probes[slot]) == before
    assert index.hits == 3 and index.misses == 1


# ---------------------------------------------------------------------------
# minimal-increment CBF
# ---------------------------------------------------------------------------


def test_wtinylfu_cbf_init_matches_minimal_increment_cbf():
    """Filter geometry from minimalIncrementCBF.c:29-44 at error=0.001.

    main=198 -> bpe = -ln(0.001)/ln(2)^2, d = ceil(ln2*bpe) = 10 probes,
    counter_num = ceil(bpe*198); and the 8-byte XXH64 probe fold matches
    libCacheSim's vendored xxhash on the same seeds (minimalIncrementCBF.c:88-91,
    hash.h:15-16). The vectors were compiled from the very xxhash.c the
    cachesim binary links.
    """
    import math

    from b12x.sequence._shared.row_cache import WTinyLfuIndex, _xxh64_8

    index = WTinyLfuIndex(200)
    bpe = -math.log(0.001) / 0.480453013918201
    assert index._hashes == math.ceil(0.693147180559945 * bpe) == 10
    assert index._counter_num == math.ceil(bpe * index._main_cap)
    assert _xxh64_8(0, 0x12345678) == 0x67BB774D39F5D473
    assert _xxh64_8(0, 0x12344321) == 0xC2A64453B85A5E79
    assert _xxh64_8(18446744073709551615, 0x12345678) == 0x60B38A7D6D1E7EA9
    assert _xxh64_8(1000447, 0x12344321) == 0x863CAB7DF837B397
    probes = index._probes_for(42)
    assert len(probes) == 10
    assert all(0 <= x < index._counter_num for x in probes)


def test_wtinylfu_add_bumps_every_probe_and_estimate_is_the_min():
    """minimalIncrementCBF.c:113-125: add is unconditional on all d probes.

    Unlike a classic minimal-increment bloom that touches only the minimum
    cells, this C increments every probed counter when the minimum has not
    saturated, so the estimate rises by exactly one per add and an unseen key
    estimates zero.
    """
    index = _wtl(200)
    probes = index._probes_for(1234)
    assert index._cbf_estimate(probes) == 0
    index._cbf_add(probes)
    assert index._cbf_estimate(probes) == 1
    index._cbf_add(probes)
    assert index._cbf_estimate(probes) == 2
    assert all(index._bf[x] == 2 for x in probes)


def test_wtinylfu_decay_halves_estimates_and_only_main_hits_tick_the_clock():
    """WTinyLFU.c:246-250 + minimalIncrementCBF.c:162-178.

    The clock is ticked by main-cache hits only -- window hits and inserts
    never reach it -- and the decay is a right shift of every counter. The
    boundary hit adds first, then halves. Constants verified request by
    request against ``cachesim micro2.txt txt WTinyLFU 200B``.
    """
    fill = [oid for oid in range(1, 202) if oid not in (7, 8)]
    seq = [7, 8] + [8] * 20 + fill + [7, 7]
    index = _wtl(200)
    assert _replay(index, seq) == "00" + "1" * 20 + "0" * len(fill) + "11"
    # The 20 window hits on 8 did not tick; exactly the two main hits on 7
    # (probation promotion, then protected move-to-head) did.
    assert index._request_counter == 2
    slot7 = index._slot[7]
    assert index._loc[slot7] == index._PROTECTED
    # insert + drain + two main-hit adds
    partner = next(
        key
        for key, s in sorted(index._slot.items())
        if index._loc[s] == index._PROBATION
    )
    slot_p = index._slot[partner]
    # insert + drain (verified against the C replay)
    assert index._cbf_estimate(index._probes[slot_p]) == 2

    index._request_counter = index._max_request_num - 1
    index.plan((7,))  # main hit: add to 5, then the boundary decay
    assert index._request_counter == 0
    assert index._cbf_estimate(index._probes[slot7]) == 5 >> 1
    assert index._cbf_estimate(index._probes[slot_p]) == 2 >> 1


# ---------------------------------------------------------------------------
# SLRU segment behaviour
# ---------------------------------------------------------------------------


def test_wtinylfu_probation_hit_promotes_to_protected():
    """SLRU.c:264-276: a probation re-access moves the row to protected head."""
    index = _wtl(200)
    _replay(index, MICRO_SEQ)
    prob = [key for key, s in index._slot.items() if index._loc[s] == index._PROBATION]
    prot = [key for key, s in index._slot.items() if index._loc[s] == index._PROTECTED]
    assert prob and prot
    assert sum(index._nq) == len(index._slot)
    victim = prob[0]
    slot = index._slot[victim]
    index.plan((victim,))
    assert index._loc[slot] == index._PROTECTED
    # one row left probation and entered protected; the cooling cascade (if
    # protected was at budget) hands a row straight back, so probation moves
    # by at most one and protected never exceeds its budget.
    assert abs(index._nq[1] - 39) <= 1
    assert index._nq[2] <= 158
    assert sum(index._nq) == len(index._slot)


def test_wtinylfu_ties_keep_the_main_victim():
    """WTinyLFU.c:318-323 compares with a strict ``>``.

    Equalise the filter (all counters to one value) so both competitors
    estimate identically; then the window tail -- never the main victim --
    must lose its slot, and the compared victim takes the step-end add of
    :349-351 on top.
    """
    index = _wtl(200)
    for oid in range(1, 202):
        index.plan((oid,))
    index._bf = [5] * index._counter_num  # every estimate is 5
    window_tail_key = index._key[index._tail[0]]
    prob_tail_key = index._key[index._tail[1]]
    prob_tail_slot = index._tail[1]
    assert window_tail_key != prob_tail_key
    index.plan((999,))
    assert window_tail_key not in index  # the tie evicted the *window* row
    assert prob_tail_key in index
    assert index._slot[prob_tail_key] == prob_tail_slot  # main victim untouched
    assert index._cbf_estimate(index._probes_for(window_tail_key)) == 6


# ---------------------------------------------------------------------------
# the composite's degenerate small-capacity corner, straight from the C
# ---------------------------------------------------------------------------


def test_wtinylfu_degenerate_budget_caches_nothing():
    """WTinyLFU_can_insert / SLRU_can_insert gate the miss path (cited).

    ``cachesim probe/xxhcheck/smoke.txt txt WTinyLFU 50B`` reports miss ratio
    1.0000: the 0.01 window share truncates to zero rows, so no object fits
    the window and the reference never caches anything. The index reproduces
    that rather than papering over the quirk, and the first capacity whose
    window share rounds to one row (100) caches normally again.
    """
    for cap in (1, 5, 50, 99):
        index = _wtl(cap)
        _replay(index, list(range(1, 40)))
        assert len(index) == 0
        assert index._nq == [0, 0, 0]
        slot = index.reserve(5)
        if slot is not None:  # a loading slot may exist but never binds
            assert index.confirm(slot) == slot
            assert not index.is_ready(5)
            assert 5 not in index._slot
    index = _wtl(100)  # window = int(100 * 0.01) = 1 row
    index.plan((7,))
    assert 7 in index


def test_wtinylfu_rejects_bad_capacity():
    for bad in (0, -1):
        with pytest.raises(ValueError, match="capacity must be positive"):
            _wtl(bad)


# ---------------------------------------------------------------------------
# loading lifecycle (identical contract to the siblings)
# ---------------------------------------------------------------------------


def test_wtinylfu_loading_rows_are_invisible_and_immune():
    """A loading key never hits, never evicts, and never leaks capacity."""
    index = _wtl(200)
    for oid in range(1, 181):
        index.plan((oid,))
    slot = index.reserve(500)
    assert slot is not None
    assert index.get(500) is None
    assert not index.is_ready(500)
    assert 500 not in index
    assert index._state[slot] == index._LOADING
    assert index._loc[slot] == index._UNQUEUED

    # Churn until every ready row has cycled: the loading slot must never
    # surface in a queue nor be handed to another key.
    for batch in range(30):
        ids = list(range(600 * batch, 600 * batch + 120))
        index.plan(ids)
        assert index._state[slot] == index._LOADING
        assert index._loc[slot] == index._UNQUEUED
        assert 500 not in index
        assert slot not in index._slot.values()

    # The synchronous path finishing what the prefetch was fetching adopts
    # the same slot (no second copy of the row).
    assert index.insert(500) == slot
    assert index.is_ready(500)
    assert index.get(500) == slot
    assert index._state[slot] == index._READY

    # A cancelled reservation returns its slot to the pool for reuse.
    cancelled = index.reserve(501)
    if cancelled is not None:
        index.cancel(cancelled)
        assert index._state[cancelled] == index._FREE
        assert 501 not in index
        assert not index.is_ready(501)


def test_wtinylfu_reserve_confirm_admits_as_fresh_window_row():
    """A confirmed prefetch enters like a miss: window head plus filter add."""
    index = _wtl(200)
    slot = index.reserve(42)
    assert slot is not None
    assert index.confirm(slot) == slot
    assert index.is_ready(42)
    assert index._loc[slot] == index._WINDOW
    assert index._cbf_estimate(index._probes[slot]) == 1  # the insert add
    assert index.get(42) == slot


def test_wtinylfu_churn_keeps_slot_accounting_and_queue_invariants():
    """Randomised access over all three queues, as the sibling churn tests do.

    Budgets hold (with SLRU's documented one-row probation transient from
    SLRU.c:306-313), every slot has one owner, the queues are disjoint walks
    over the pool, and the accounting identity from the S3-FIFO churn test
    holds: ``len(_free) + resident + loading == _used``.
    """
    # capacity < 100 would truncate the 1% window to zero rows and cache
    # nothing at all (see test_wtinylfu_degenerate_budget_caches_nothing).
    index = _wtl(200)
    pool: dict[int, int] = {}
    rng = random.Random(4321)
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
        assert index._nq[1] <= index._prob_cap + 1
        assert index._nq[2] <= index._prot_cap
        assert index._nq[0] <= index.capacity
        assert sum(index._nq) == len(index._slot)
        seen: set[int] = set()
        for q in range(3):
            slot, walked = index._head[q], 0
            while slot >= 0:
                assert slot not in seen  # a slot sits in at most one queue
                seen.add(slot)
                assert index._loc[slot] == q + 1
                assert index._state[slot] == index._READY
                slot, walked = index._next[slot], walked + 1
            assert walked == index._nq[q]
        assert seen == set(index._slot.values())
        assert len(index._free) + len(seen) + len(index._loading) == index._used
        for key, slot in index._slot.items():
            assert index._key[slot] == key
        for key, slot in index._loading.items():
            assert key not in index
            assert index._state[slot] == index._LOADING
            assert index._loc[slot] == index._UNQUEUED
            assert slot not in seen
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


# ---------------------------------------------------------------------------
# DiskRowCache wiring (GPU lane; skipped on machines without SM12x)
# ---------------------------------------------------------------------------


def _host_payloads():
    import torch

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


def test_wtinylfu_policy_selected_by_env_and_unknown_names_rejected(monkeypatch):
    """``B12X_ROW_CACHE_POLICY=wtinylfu`` selects the index; junk is an error.

    The env value is space- and case-normalised (operators set it from shells
    and manifests); an unrecognised name raises rather than silently falling
    back, and ``stats()`` reports the selected policy.
    """
    from tests.experimental.b12x.conftest import require_b12x

    device = require_b12x()
    from b12x.sequence._shared.disk_table import DiskRowCache
    from b12x.sequence._shared.row_cache import WTinyLfuIndex

    host_rows = tuple(p[7:290].contiguous() for p in _host_payloads())

    def build(**kwargs):
        return DiskRowCache(
            device=device,
            max_lookups=40,
            table_rows=300,
            shard_start=7,
            shard_end=290,
            shard_rows=100,
            weight_row_bytes=257,
            scale_row_bytes=9,
            queue_depth=2,
            cache_rows=8,
            cache_tier="host",
            host_rows=host_rows,
            **kwargs,
        )

    monkeypatch.delenv("B12X_ROW_CACHE_POLICY", raising=False)
    with pytest.raises(ValueError, match="B12X_ROW_CACHE_POLICY"):
        build(policy="least-frequently-used")
    monkeypatch.setenv("B12X_ROW_CACHE_POLICY", "lru")
    with pytest.raises(ValueError, match="B12X_ROW_CACHE_POLICY"):
        build()

    monkeypatch.setenv("B12X_ROW_CACHE_POLICY", "  WTinyLFU ")
    selected = build()
    try:
        assert type(selected._index) is WTinyLfuIndex
        assert selected.stats()["row_cache_policy"] == "wtinylfu"
    finally:
        selected.close()
    forced = build(policy="wtinylfu")
    try:
        assert type(forced._index) is WTinyLfuIndex
        assert forced.stats()["row_cache_policy"] == "wtinylfu"
    finally:
        forced.close()
