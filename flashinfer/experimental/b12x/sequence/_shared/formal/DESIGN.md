# Zero-copy SIEVE row cache for disk-resident PLE / Engram tables

Design + formal-model artifact for `DiskRowCache` (`../disk_table.py`) and
`SieveIndex` (`../row_cache.py`). Status: **implemented in Python (opt-in, off by
default); the C/zero-copy reclamation path is NOT built and is gated on the
SM120 measurement described in §6.** The correctness of its central invariant is
proven (see `results.md`).

## 1. Problem

A PLE (n-gram embedding) table is 20,000,000 rows × (80 B NVFP4 weight + 10 B
group-16 scale) = 26.8 GiB, far too large for VRAM. `DiskRowCache` reads the
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

## 6. Performance verdict and the go/no-go gate

**GB10: dead.** A 100k-row host-tier SIEVE cache produced no throughput change
at any concurrency (lil-bench c=1,4,8,16, 3 interleaved rounds): hit-rate
*decays* to ~0.42 at c=16 (16 concurrent sequences blow up the per-step working
set), and the io_uring reads are fully hidden behind compute. c=1 decode A/B was
noise (t=−1.07); GSM8K was consistently −0.78 tok/s — that penalty is the Python
plan in prefill chunks, not a missed disk win. So the Python cache already
cannot win on GB10, and zero-copy (which only removes the plan + copy overhead
of the same cache) can at best claw the −0.78 back to ~0, never positive.

**SM120/WSL2: untested, and the only place the feature could pay.** The
end-user's original symptom (RTX PRO 6000 under WSL2) had a 4.5 ms/step host
gap and disk PLE reads a suspected contributor. The pinned-memory fix
(`VLLM_WSL2_ENABLE_PIN_MEMORY=1`, +21 % c=1) already collapsed the *host-sync*
part of that gap, which argues the disk specifically was not the bottleneck —
but that is not conclusive for SM120. **Gate:** before building the C/zero-copy
path, run the disk-off vs cache-on vs `ple_table_memory=ram` A/B on server21
(native SM120; requires scaling down the production `rtx6000-qwen38-flash-next`
via ArgoCD first). If `ram` ≫ `disk` and cache-on closes most of that gap, the
disk is exposed and the C path is worth building; if not, ship the cache as
formally-proven-but-off-by-default and do not build C.
