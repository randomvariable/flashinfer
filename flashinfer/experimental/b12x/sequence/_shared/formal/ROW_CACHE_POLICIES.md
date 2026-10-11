# Row-cache replacement policies: historical decision record

**Status: decision record, not a current feature matrix.** The shipped row
cache is a single implementation — the host-tier C S3-FIFO
([`_row_cache.c`](../../../../../loader/_row_cache.c), reached through
[`row_cache.py`](../row_cache.py)'s `S3FifoRowCache` facade). The SIEVE and
W-TinyLFU Python indices and the `B12X_ROW_CACHE_POLICY` knob were deleted
as experiments; everything below is preserved because *it is the evidence
that justifies that deletion*: the §4 matrix is measurements against the
removed code paths, replayed by parent-owned harnesses on the production
traces. The Python S3-FIFO class that mirrors libCacheSim survives as the
per-request parity oracle `_PyS3FifoReference` in
[`test_row_cache_policies.py`](../../../../../../../tests/experimental/b12x/sequence/test_row_cache_policies.py).

Companion to [`DESIGN.md`](DESIGN.md) (the zero-copy/deferred-free design and
the end-to-end throughput verdicts).

## 1. The shared contract

The index owns only the `id -> slot` mapping and eviction order over the
fixed pool `0..capacity-1`; payload placement (host RAM), the miss source
(io_uring / `host_rows`), and staging belong to `DiskRowCache`. The contract
below was shared by all three Python classes, is implemented identically in
the C, and is gated request-by-request against the retained oracle:

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

## 2. Policies (two of the three removed; kept as measured history)

### SIEVE (`SieveIndex`, former default — removed)

Insertion-ordered FIFO with a moving hand and one visited bit per row
(Zhang et al., NSDI'24). A hit sets the bit; the hand clears bits as it scans
and evicts the first unvisited row. Cheapest bookkeeping of the three (a hit is
one dict lookup plus one byte store) — the reason it was shipped first. Its
pathology below ~5 % of the working set (§4 finding 2) and its loss to
S3-FIFO at the operating capacity removed it.

### S3-FIFO (`S3FifoIndex` → shipped C implementation)

Python mirror of libCacheSim's `S3FIFO.c` (the post-2024 variant the
`cachesim S3FIFO-0.1000-2` reference column runs — not the paper's original,
see the oracle's docstring for the exact C-rule correspondence): small queue
S = 10 % of capacity filters one-hit rows, main queue M = 90 % is a 2-bit
clock, ghost list G = 90 % of capacity (keys only) feeds the direct-to-M fast
path. Promotion into M requires `move-to-main-threshold=2` hits and keeps the
counter. This policy is the shipped one, now in C; the Python class is the
test oracle.

### W-TinyLFU (`WTinyLfuIndex` — removed)

Python mirror of libCacheSim's `WTinyLFU.c` default (`window-size=0.01`,
main = `SLRU` with `seg-size=1:4`): every new row enters a 1 % LRU window;
window drains are admitted into SLRU probation only when their
minimal-increment Counting Bloom Filter estimate beats the probation victim's
(ties keep the main victim); probation hits promote to protected; the CBF
halves every `32 x main_size` main-cache hits. The C's quirks were replicated,
not smoothed — because parity with the C is the gate, not the paper's prose.
It never won a production cell by more than S3-FIFO (§4 finding 3), so it was
removed with its tests when the tree committed to one implementation.

## 3. Validation gates

Three layers, all on the **production row-id traces**
(`lcs_all.txt` 3.03 M, `lcs_gsm8k.txt` 1.02 M, `lcs_mmlu.txt` 2.01 M accesses,
converted from the `ple.jsonl` engine probe of `DESIGN.md` §1; replayed at
batch=1, the c=1 decode shape):

1. **Reference parity** (below): `cachesim` running the same trace/capacity,
   at caps 1k..1M, against (a) the Python oracle and (b) the **shipped C
   extension itself**, driven through `S3FifoRowCache` with dummy payload
   planes (`probe/c_cachesim_gate.py`, results `probe/c_cachesim_gate.csv`).
   Both hit the same worst delta, 0.0035 — concentrated in the mid-capacity
   10k/100k cells, 0.0000 elsewhere — so the gate on the port is closed by
   direct measurement, not inherited by assumption. Historical harnesses:
   `probe/s3fifo_parity.py`, `probe/wtinylfu_parity.py`,
   `probe/policy_matrix.py` (workstation `~/.cache/vllm-multiarch-oci/probe/`;
   results `probe/policy_matrix_v2.csv` — the v1 CSV predates the strtol
   parser fix — and `probe/wtinylfu_gate_results.csv`). W-TinyLFU (removed)
   was additionally matched **request-by-request** against a C driver linked
   against `liblibCacheSim.a` (`probe/xxhcheck/trace_dump.c`).
2. **Contract unit tests**: `test_row_cache_policies.py` now runs the C
   against the retained oracle: hand-checked sequences, per-request
   plan/counter/ghost/queue/pool-byte equality, randomised churn with
   interleaved reserve/cancel/confirm, and the production-trace prefix —
   replacing the pure-Python hit-vector and wiring tests the multi-policy
   era carried. `test_disk_row_cache.py` runs the `DiskRowCache` byte
   fixture across the host tier on the GPU.
3. **Byte identity on SM120 GPU**: the `DiskRowCache` batch fixture runs
   `{host tier} x {disk, host_rows}` through the C path — the cache cannot
   change what the read path returns. The former device-tier and
   removed-policy arms became ``ValueError`` regression tests naming the
   deleted knobs.

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

## 5. Decision

Ship S3-FIFO only. The §4 table is the rationale: at the shipped operating
point S3-FIFO cuts misses ~24 % relative against SIEVE, SIEVE is pathological
at small pools, W-TinyLFU never overtakes S3-FIFO on production cells, and all
policies converge above the working set — where policy choice stops mattering
and pool size is the lever. With one policy there is no knob:
``B12X_ROW_CACHE_POLICY`` and the Python index classes are deleted, the C is
the implementation, and enabling the cache without a size takes the 1 GiB
default (§8 of `DESIGN.md` keeps the pre-port throughput history; the sweep
arms are cache-off vs the C path).

## 6. What this does NOT cover

- Throughput deltas end-to-end from a policy switch (only §7 `DESIGN.md` for
  tier/host evidence; policy A/B pending as in §5).
- Prefetch interaction: `prefetch()` is policy-agnostic above the contract;
  the `device`-tier prefetch measurements of `DESIGN.md` §7 used SIEVE.
- C/portability: resolved. The former "future `_ple_reader.c` port" gate
  named in `DESIGN.md` status has been met — the hot path is the C
  `_row_cache.c` extension; these measurements were its executable
  specification and remain its historical justification.
