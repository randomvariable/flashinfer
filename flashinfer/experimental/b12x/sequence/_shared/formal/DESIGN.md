# Zero-copy SIEVE row cache for disk-resident PLE / Engram tables

Design + formal-model artifact for `DiskRowCache` (`../disk_table.py`) and
`SieveIndex` (`../row_cache.py`). Status: **implemented in Python (opt-in, off by
default); the C/zero-copy reclamation path is NOT built and is gated on the
SM120 measurement described in §6.** The correctness of its central invariant is
proven (see `results.md`).

## 1. Problem

A PLE (n-gram embedding) table is **320,001,536 rows × (80 B packed-NVFP4 weight +
10 B E4M3 group-64 scale) = 26.9 GiB**, in 128 shard files of 2,500,012 rows each
(`split_ngram_parts: 128`; measured from the CSF QAD checkpoint headers), far too
large for VRAM. `DiskRowCache` reads the
rows a step needs over io_uring (O_DIRECT, fixed buffers/files, 4 KiB block
coalescing) into a batch-sized mapped-host staging buffer; a Triton decode
kernel then gathers rows from it. Every step re-reads its rows — the cache
stages exactly one batch and retains nothing. Access is Zipfian: measured over
3.03 M row reads (1.05 M distinct) from a real `llm-inference-bench` run, 65 %
of reads are repeats, the top 1 % of rows carry 42 %, and ~72 % of distinct rows
are read exactly once. A small persistent row cache can skip most of the
re-reads.

The Python SIEVE cache added this session (tier `host`, `B12X_ROW_CACHE_ROWS`)
resolves ids→slots in Python per step, then copies hit rows `pool[slot] →
staging[i]`. Measured cost of `_read_cached` on the board fits
**8.4 µs + 181 ns/id**. At decode batch sizes (n≈24–64) that is ≤20 µs/step —
irrelevant. At prefill chunk sizes (n = tokens×head_count, up to 96304) it is
**~17 ms/step of Python**, which is where the added latency lands.

## 2. Zero-copy design (the proposed C path — not yet built)

Make the pool the single store and let the decode kernel gather directly from
it, so hits move no bytes and misses DMA straight into their final slot:

- **Pool = registered DMA target.** Allocate the pool once (fixed base address);
  register its pages with `IORING_REGISTER_BUFFERS`. A miss issues
  `io_uring_prep_read_fixed` writing the row *into its pool slot* — there is no
  staging buffer and no bounce copy on the miss path.
- **Per-position slot index.** A persistent `slot[max_lookups]` int buffer (fixed
  address, so it is CUDA-graph-capturable) maps each of a step's `count` gather
  positions to a pool slot. The decode kernel's row-index operand becomes
  `slot[i]` against the pool base instead of the contiguous batch index against
  the staging buffer. This unifies the disk path with the existing resident
  (`table_memory in {device, mapped_host}`) gather, which already indexes the
  table by id.
- **C owns the plan.** Hit/miss resolution (`SieveIndex`) runs in C inside
  `_ple_reader.c`: open-addressed key→slot map, FIFO queue + hand + visited bit,
  writing `slot[]` and the miss id list, then submitting the miss wave. This
  removes the 181 ns/id Python term entirely.

### Placements (the three architectures)

| Hardware | source | tier | hit cost | miss cost |
|---|---|---|---|---|
| GB10 / DGX Spark (unified) | disk | host pool | kernel reads host-mapped slot | DMA → pool slot |
| Discrete GPU (SM120/100) | disk | host-RAM pool | host copy to device slot | DMA → host slot, 1× H2D |
| Discrete GPU, table in host RAM | host table | VRAM pool | kernel reads VRAM slot | H2D → VRAM slot |

`row_cache.py`/`disk_table.py` today implement the two host-pool tiers and the
host-table source; only the **C plan + direct-DMA-into-pool-slot** is unbuilt.

## 3. The one dangerous invariant — and its proof

Everything is bookkeeping except this: **a slot an outstanding decode is
gathering from must never be recycled (freed → reallocated to a different row)
until that decode has retired.** With a batch-sized staging buffer this is free
(overwrite on the next step is fenced by the step's event). With a persistent
pool it is a real deferred-reclamation protocol: the SIEVE hand may pick a
victim that is `inflight == ∅` per the last step but a still-in-flight captured
decode from the *previous* step is reading it.

This is modelled as an epoch/RCU reclamation and proven in Quint + Apalache
(`slotpool.qnt`, evidence in `results.md`):

- **`verified` (deferred free: only a slot with no inflight holders *and* no
  outstanding reader reference may be recycled):** `NoUseAfterFree` holds,
  proven by **inductive invariant** (unbounded over the finite abstraction).
- **`broken` (recycle a resident slot regardless of outstanding readers):**
  Apalache returns a concrete 5-state **counterexample** where a reader still
  references a slot that has been reset to free — the checker rejects it.

The `DEFER_FREE` boolean is the exact code-level knob: `free_slot` in
`row_cache.py`/`disk_table.py` must carry the `inflight == ∅ ∧ no-reader-references`
guard, and the retirement path must clear a slot's `inflight` only after the
reading decode's CUDA event retires. That guard is what the proof pins.

## 4. What the formal model does NOT cover

- SIEVE eviction *quality* (hit-rate optimality) — that is the paper's claim, not
  verified here; only structural correctness of the lifecycle is proven.
- The Triton gather kernel's byte-level correctness — covered by the 25
  byte-identity unit tests in `test_disk_row_cache.py`.
- Real CUDA event ordering / driver semantics — the model abstracts the
  outstanding-reader set (`reading`) and inflight sets; the implementation must
  honour the same protocol with per-step `Event`s.
- DMA failure / short-read paths (`_ple_reader.c` already guards these).

## 5. Correctness gates (must pass before this ships on)

1. `quint verify slotpool.qnt --main verified --invariant=NoUseAfterFree
   --inductive-invariant=IndInv` → `[ok]` (unbounded).
2. `quint verify slotpool.qnt --main broken --invariant=NoUseAfterFree` →
   `[violation]` (negative test; proves the gate is non-vacuous).
3. `test_disk_row_cache.py` byte-identity suite green on GB10 across both tiers
   and both sources (already green: 25 passed).

## 6. Performance verdict (measured, 2026-10-09)

Method: cloned the live production pod spec on server21 (native SM120, RTX PRO
6000 96 GiB, full stack: LMCache recurrent connector, 1 MiB context,
`--gpu-memory-utilization 0.985`, MTP k=3) via the llm-d ArgoCD app scale-down;
arms = same image + env, differing only in overlay files and `B12X_ROW_CACHE_*` /
`ple_table_memory`. Metrics from `llm-inference-bench` (pinned v0.7.5) inside the
modelserver container: sustained c=1 decode (45 s) and GSM8K 64-item profile
(64×256 tokens).

| arm | decode tok/s | gsm8k tok/s | cache hit rate |
|---|---|---|---|
| disk, no cache (4 boots) | 147.6 ± 2.7 | 193.6 ± 3.1 | — |
| disk + host-tier pool 100k (3) | 153.2 ± 5.4 | 200.1 ± 1.9 | 0.44–0.46 |
| disk + host-tier pool 5M (1) | 142.6 | 200.5 | 0.45 (resident 317k) |
| disk + device-tier pool 100k (2) | 146.5 ± 4.2 | 192.6 ± 2.5 | ~0.46 |
| `ple_table_memory=ram` (1) | **168.8 (+14.4 %)** | **224.5 (+16.0 %)** | — |

Paired host−none deltas: gsm8k +7.8 / +7.1 / +8.9 (mean **+7.9 ± 1.0, t=14.3,
3/3 positive**, ≈ +4 %); decode +10.3 / +6.6 / +1.4 (all positive, high variance
in 45 s windows). Acceptance length 3.02–3.13 across all arms: a throughput
effect, not a numerics effect.

Findings:

1. **The disk read IS exposed on SM120 — ~14 %.** The ram ceiling proves it.
2. **The row cache recovers only ~25 % of that exposure, and pool size is not
   the lever:** hit rate saturates at ~0.45 whether the pool holds 100k or 5M
   rows (the whole re-usable working set is ~317k rows; 100k rows = 8.6 MiB
   already captures it). The remaining misses are rows read once per session —
   no eviction policy recovers them.
3. **Zero-copy/C-plan is therefore REJECTED on evidence.** The unclaimed ~10 %
   sits on the MISS path (8.3 ms per 1024-row io_uring wave, measured), which
   still touches NVMe no matter who owns the bookkeeping; the C plan addresses
   ~20 µs/step of Python and one host memcpy — the wrong 10 %. The device tier
   additionally pays GPU `index_select`/`index_copy` work per step that zeroes
   its own miss savings at 64-row waves (measured wash).
4. **Ship:** the Python host-tier cache as-is, opt-in, default off, pool 100k
   rows (8.6 MiB) — a stable +4 % on prefill-heavy profiles at ~zero cost and
   zero VRAM. **For users wanting the full disk win on SM120-class boxes, the
   answer is `ple_table_memory=ram`** (+14 %, 26.9 GiB host RAM — affordable
   everywhere this model runs, which already spends 24–32 GiB on lmcache L1),
   not new kernel code. WSL2 will differ (virtio-blk raises miss cost further;
   ram/cache matter more there).

Not measured: WSL2 SM120 (end-user board); single boots for 5m/ram arms
(ceiling direction is unambiguous, magnitude ±3 %).
