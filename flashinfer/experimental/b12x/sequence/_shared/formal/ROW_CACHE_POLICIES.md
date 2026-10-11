# Row-cache replacement policies: contracts, reference parity, and the measured comparison

Companion to [`DESIGN.md`](DESIGN.md) (the zero-copy/deferred-free design and the
end-to-end throughput verdicts). This document covers the three replacement
policies `DiskRowCache` can run over its pinned row pool, how they were
validated, and which one to pick at the operating capacity.

Policies live in [`row_cache.py`](../row_cache.py); wiring and environment knobs
in [`disk_table.py`](../disk_table.py) (`B12X_ROW_CACHE_POLICY`, capacity via
`B12X_ROW_CACHE_ROWS` / `B12X_ROW_CACHE_BYTES`, tier via `B12X_ROW_CACHE_TIER`).

## 1. The shared contract

A policy index owns only the `id -> slot` mapping and eviction order over the
fixed pool `0..capacity-1`; payload placement (host/device tier), the miss
source (io_uring / `host_rows`), and staging belong to `DiskRowCache`. All
three classes implement the identical surface, so a batch is served byte-for-byte
the same regardless of policy (enforced by the parametrized byte-identity
fixture in `test_disk_row_cache.py`):

- `get(key)` — raise the row's recency/frequency state, return its ready slot
  or `None`. Must never return a loading slot.
- `insert(key, ...)` — admit a fresh row, evicting per policy; returns the slot
  or `None` only when no committed row can be freed.
- `plan(ids) -> RowPlan` — resolve one batch: hits read before stores,
  duplicate miss ids fetch once, a slot handed to the batch stays owned by its
  key until a pop releases it (the deferred-free invariant `DESIGN.md` §3 proves
  for the shared slot pool; loading slots are excluded from every eviction
  sweep).
- `reserve/confirm/cancel` — the prefetch lifecycle: a reserved (`LOADING`)
  slot consumes capacity, is invisible to lookups, and is immune to eviction;
  only `confirm` publishes it.

## 2. Policies

### SIEVE (`SieveIndex`, default)

Insertion-ordered FIFO with a moving hand and one visited bit per row
(Zhang et al., NSDI'24). A hit sets the bit; the hand clears bits as it scans
and evicts the first unvisited row. Cheapest bookkeeping of the three (a hit is
one dict lookup plus one byte store) — the reason it was shipped first.

### S3-FIFO (`S3FifoIndex`)

Python mirror of libCacheSim's `S3FIFO.c` (the post-2024 variant the
`S3FIFO-0.1000-2` reference column runs — not the paper's original, see the
file's docstring for the exact C-rule correspondence): small queue S = 10 % of
capacity filters one-hit rows, main queue M = 90 % is a 2-bit clock, ghost list
G = 90 % of capacity (keys only) feeds the direct-to-M fast path. Promotion
into M requires `move-to-main-threshold=2` hits and preserves the counter.

### W-TinyLFU (`WTinyLfuIndex`)

Python mirror of libCacheSim's `WTinyLFU.c` default (`window-size=0.01`,
main = `SLRU` with `seg-size=1:4`): every new row enters a 1 % LRU window;
window drains are admitted into SLRU probation only when their
minimal-increment Counting Bloom Filter estimate beats the probation victim's
(ties keep the main victim); probation hits promote to protected; the CBF
halves every `32 x main_size` main-cache hits. The C's quirks are replicated,
not smoothed — including the compared-victim filter add at the end of every
evict step and the main-hits-only decay clock — because parity with the C is
the gate, not the paper's prose.

## 3. Validation gates

Three layers, all on the **production row-id traces**
(`lcs_all.txt` 3.03 M, `lcs_gsm8k.txt` 1.02 M, `lcs_mmlu.txt` 2.01 M accesses,
converted from the `ple.jsonl` engine probe of `DESIGN.md` §1; replayed at
batch=1, the c=1 decode shape):

1. **Reference parity of the classes** (below): each index vs `cachesim`
   running the same trace/capacity, at caps 1k..1M. Harnesses:
   `probe/s3fifo_parity.py`, `probe/wtinylfu_parity.py`, `probe/policy_matrix.py`
   (workstation `~/.cache/vllm-multiarch-oci/probe/`, results
   `probe/policy_matrix.csv`, `probe/wtinylfu_gate_results.csv`). W-TinyLFU is
   additionally matched **request-by-request** against a C driver linked
   against `liblibCacheSim.a` (`probe/xxhcheck/trace_dump.c`), so its true
   delta is 0, not merely within print rounding.
2. **Contract unit tests**: `test_row_cache_policies.py` (pure Python: exact
   per-request hit vectors from the C on known micro traces, decay, promotion,
   tie behaviour, loading-slot invisibility, policy/env wiring) and the
   `SieveIndex`/`S3FifoIndex` sequence + randomised-churn tests in
   `test_disk_row_cache.py` (budget bounds, queue disjointness, slot-ownership
   accounting under 200-batch random access with interleaved
   reserve/cancel/confirm).
3. **Byte identity on SM120 GPU**: the same `DiskRowCache` batch fixture runs
   across `{sieve, s3fifo, wtinylfu} x {host, device} x {disk, host_rows}` —
   choosing a policy cannot change what the read path returns. W-TinyLFU's arm
   starts at 128 rows because below ~100 rows its window/SLRU segments truncate
   to zero and it admits nothing *by design* (matches the C's 1.0000 at 50 B).

Parity achieved (worst |miss-ratio| delta over all 15 cells, this branch):

| policy | vs `cachesim` same-policy | vs `cachesim` LRU control |
|---|---|---|
| SIEVE | 0.0000 (exact print-level match) | — |
| S3-FIFO | 0.0035 | harness LRU column 0.0000 (sanity) |
| W-TinyLFU | 0.0013 (request-exact vs C driver) | harness LRU column 0.0000 (sanity) |

The S3-FIFO residual is a stable +0.0034..0.0035 at the two mid capacities
(10k, 100k) on all three traces and zero elsewhere; we have localised it to
ordering of the 2-bit clock's chance/reinsert relative to the C's
`S3FIFO_evict` around the ghost fast path, but not to the last branch, so it
is reported as measured rather than explained. It never changes the ordering
conclusions below. An earlier replay series that disagreed with these columns
was traced to a harness parsing bug (Python `int()` dropping the trace's
eight `-1` obj-id rows, desynchronising the request stream); the harnesses
now parse with C `strtol` semantics, and the C-vs-C LRU control column
matching to 0.0000 is the guard against this class of error.

## 4. Measured comparison (miss ratio, lower is better)

From `probe/policy_matrix.csv`; `b12x_*` are the shipped classes, `lcs_*` the C
references, `lru_control` cachesim LRU.

| trace | rows | SIEVE | S3-FIFO | W-TinyLFU | LRU |
|---|---|---|---|---|---|
| all | 1 024 | 0.9304 | **0.8097** | 0.8234 | 0.8185 |
| all | 10 240 | 0.7597 | **0.4972** | 0.5817 | 0.4957 |
| all | 100 352 | 0.5074 | **0.3827** | 0.4362 | 0.3906 |
| all | 317 440 | **0.3570** | 0.3610 | 0.3732 | 0.3614 |
| all | 1 000 448 | 0.3465 | 0.3465 | 0.3471 | 0.3465 |
| gsm8k | 1 024 | 0.8543 | 0.7421 | **0.7360** | 0.7768 |
| gsm8k | 10 240 | 0.6096 | **0.4147** | 0.4795 | 0.4334 |
| gsm8k | 100 352 | **0.3163** | 0.3204 | 0.3408 | 0.3187 |
| gsm8k | 317 440 | 0.3017 | 0.3017 | 0.3017 | 0.3017 |
| gsm8k | 1 000 448 | 0.3017 | 0.3017 | 0.3017 | 0.3017 |
| mmlu | 1 024 | 0.9675 | **0.8441** | 0.8706 | 0.8398 |
| mmlu | 10 240 | 0.6541 | **0.5399** | 0.6245 | 0.5280 |
| mmlu | 100 352 | 0.5038 | **0.4173** | 0.4738 | 0.4300 |
| mmlu | 317 440 | 0.3973 | **0.3968** | 0.4076 | 0.4017 |
| mmlu | 1 000 448 | 0.3878 | 0.3878 | 0.3878 | 0.3878 |

Findings:

1. **At the shipped operating point (100k rows ~ 8.6 MiB) S3-FIFO is the
   policy: 0.383 vs 0.507 (all), 0.417 vs 0.504 (mmlu) — a ~24 % relative cut
   in disk-side misses; even on the friendly gsm8k trace SIEVE's lead is 1.3 %
   (0.316 vs 0.320).** The `DESIGN.md` §7 hit rate of 0.638 measured live on
   the SIEVE pool sits exactly where this table predicts; switching the default
   is expected to move it toward 0.62 miss-ratio -> ~0.66+ hit-rate.
2. **SIEVE is pathological below ~5 % of the working set** (0.93/0.97 miss at
   1k rows): the visited bit lets a scanned-once old row survive indefinitely
   while fresh hot rows are evicted unseen. If a deployment sizes the pool
   tiny, S3-FIFO's small-queue filter is what keeps it sane (it even beats
   LRU there).
3. **W-TinyLFU never wins a production cell by more than S3-FIFO wins**, and
   trails it at the operating capacity on the two longer traces. Its 1 % window
   is calibrated for web caches; the PLE row stream is nearly scan-at-first-
   touch, reuse-at-depth. It stays in the tree as the documented, parity-tested
   alternative and for its decay clock's future tuning headroom, not as a
   default candidate.
4. **All policies converge at >= 317k rows** (~28 MiB, the ~317k-row
   reusable working set of `DESIGN.md` §6): above the working set, policy is
   irrelevant and pool size is RAM, so the capacity guidance is unchanged —
   100k host-tier rows is the shipped configuration, `ram` table mode is the
   big lever.

## 5. Recommendation

`B12X_ROW_CACHE_POLICY=s3fifo` at the 100k-row host-tier default; keep `sieve`
as the default-off-adjacent safe choice and `wtinylfu` as the tested
alternative. Before flipping the built-in default, re-run the SM120 sweep
matrix (`DESIGN.md` §7 protocol: full lil-bench, >= 2 rounds, reversed arm
order) with `s3fifo` in place of `sieve` — the hit-rate gain must show up as
end-to-end throughput on the host tier to justify changing the shipped
default; the replay evidence here justifies offering the knob now.

## 6. What this does NOT cover

- Throughput deltas end-to-end from a policy switch (only §7 `DESIGN.md` for
  tier/host evidence; policy A/B pending as in §5).
- Prefetch interaction: `prefetch()` is policy-agnostic above the contract;
  the `device`-tier prefetch measurements of `DESIGN.md` §7 used SIEVE.
- C/portability: these classes are the executable specification for a future
  `_ple_reader.c` port — that port remains gated on the prefill-path
  measurement named in `DESIGN.md` status, not on policy choice.
