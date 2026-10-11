# Zero-copy SIEVE row cache for disk-resident PLE / Engram tables

Design + formal-model artifact for `DiskRowCache` (`../disk_table.py`) and
`SieveIndex` (`../row_cache.py`). Status: **implemented in Python (opt-in, off by
default); the C/zero-copy reclamation path is NOT built. Its gate is
§6/§8: the prefill exposure measurement landed (§8 — host work overlapped,
no win today), the GB10 ARM Python constant is open. The correctness of its
central invariant is
proven (see `results.md`). The full policy set (SIEVE / S3-FIFO / W-TinyLFU),
their libCacheSim reference-parity gates, and the measured comparison at the
operating capacity are in [`ROW_CACHE_POLICIES.md`](ROW_CACHE_POLICIES.md).

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
3. **Zero-copy/C-plan is REJECTED on evidence, scoped to what was measured:
   discrete-GPU decode waves on server21.** There, the unclaimed ~10 % sits on
   the MISS path (8.3 ms per 1024-row io_uring wave, measured), which still
   touches NVMe no matter who owns the bookkeeping; the C plan addresses
   ~20 µs/step of Python and one host memcpy — the wrong 10 %. Two regimes
   were **unmeasured** at that date; (a) has since been measured (see §8):
   prefill chunks (n = tokens x head_count, up to 96 304) cost ~32 ms/call of
   Python on x86, but cold-prefill calibration shows the host work is
   overlapped and only ~0.5-1 % exposed, so the C plan's CPU term is the
   right one and the prize is the exposed fraction, not the 17 ms estimate;
   (b) GB10's ARM cores have never been profiled for `_read_cached`
   -- the 181 ns/id constant is measured on server21's Neoverse-N1 host (at ~1.17 GHz, 39% of its 3 GHz ceiling), NOT x86; GB10's cores are newer but fewer-per-process budget,
   while the bounce term is ~0 on a unified board. Both numbers must be taken
   before quoting a cross-board C verdict. The device tier additionally pays
   GPU `index_select`/`index_copy` work per step that zeroes its own miss
   savings at 64-row waves (measured wash).
4. **Ship:** the Python host-tier cache as-is, opt-in, default off — enable
   with `B12X_ROW_CACHE_BYTES=8.6MiB` (== `B12X_ROW_CACHE_ROWS=100000` at the
   PLE geometry's 90 B/row; the byte knob rounds down to whole rows) and
   `B12X_ROW_CACHE_TIER=host` — a stable +4 % on prefill-heavy profiles at
   ~zero cost and zero VRAM. **For users wanting the full disk win on
   SM120-class boxes, the answer is `ple_table_memory=ram`** (+14 %, 26.9 GiB
   host RAM — affordable everywhere this model runs, which already spends
   24–32 GiB on lmcache L1), not new kernel code. WSL2 will differ
   (virtio-blk raises miss cost further; ram/cache matter more there).

## 7. Full-sweep replication with the prefetch hook (2026-10-10/11)

The §6 matrix ran one point per arm: c=1, context 0, 45 s decode + GSM8K. The
verdict was re-measured as a full lil-bench sweep — `--concurrency 1,2,4,8,16`
× `--contexts 0,16k,64k,128k`, 45 s/cell — on the same cloned production spec,
**two rounds with reversed arm order** (round 2 `none,host,device,device-pf`,
round 3 reversed), pooled per cell. Tree under test: this branch at
`16c596af9` (SIEVE only; `B12X_ROW_CACHE_POLICY`/`B12X_PLE_PREFETCH` postdate
it — the prefetch arm drove the same `prefetch()` API through the vLLM-side
hook, gated by `VLLM_PLE_PREFETCH=1`). Harness `~/ab/run_sm120.sh` on server21,
logs `~/ab/ple_ab_r{2,3}.log`.

Pooled Δ vs the pooled no-cache baseline (gsm8k = pooled c=1 profile):

| arm | decode c=1 | decode c≥2 | gsm8k tok/s | acceptance | hit rate |
|---|---|---|---|---|---|
| none (disk, no cache) | baseline | baseline | 195.6 | 3.10 | — |
| host tier 100k | **+3.6 %** | **+3.4 %** | 198.7 | 3.08 | 0.638 |
| device tier 100k (`auto` on discrete) | −7.9 % | +0.6 % | 191.6 | 3.09 | 0.649 |
| device tier + prefetch | −2.9 % | −0.0 % | 190.2 | 3.09 | 0.650 |

Revisions and additions to the §6 verdict:

1. **Hit rate is traffic-dependent, and §6 under-measured it.** Under mixed
   concurrency the pool reaches 0.64–0.65 (vs 0.44–0.46 at c=1-only), because
   context-stable cells re-read the same rows. The working-set claim in §6
   finding 2 stands (size is still not the lever); the ceiling value does not.
2. **The decisive knob is the tier, not the pool.** On a discrete GPU, `auto`
   selects the device tier and that costs −7.9 % at c=1; explicit `host` gains
   +3.6 %/−3.4 % (§6's ≈ +4 % holds, now on the full surface).
3. **The prefetch hook recovers +5.4 pp of the device-tier penalty** (−7.9 →
   −2.9 at c=1): the disk miss wave is genuinely hideable under GPU compute.
   It never beats `host`, so §6's rejection of extra machinery on the miss path
   is unchanged — with prefetch the gap to baseline is within round noise, not
   positive. Prefetch under vLLM needs **both** gates set: `VLLM_PLE_PREFETCH=1`
   (runner hook) and `B12X_PLE_PREFETCH=1` (worker, added in `65138cc48`).
4. **Protocol rule: single 45 s c=1 cells are below the noise floor for ≤5 %
   effects.** The first c=1-only round of this series reported the host cache
   as ~6 % *slower* than no cache — the sign inverted under the replicated
   sweep. Any future row-cache verdict must be a sweep, ≥ 2 rounds, reversed
   arm order.

Not measured here: the `s3fifo` policy and the in-b12x `B12X_PLE_PREFETCH`
worker path end-to-end (both postdate the tree under test). Offline replay of
all three policies on the production traces is now measured and gate-verified
against libCacheSim in [`ROW_CACHE_POLICIES.md`](ROW_CACHE_POLICIES.md);
re-run this sweep matrix on the branch head before quoting end-to-end policy
or worker numbers.

Not measured: WSL2 SM120 (end-user board); single boots for 5m/ram arms
(ceiling direction is unambiguous, magnitude ±3 %).

## 8. Prefill exposure calibration — the C-port gate measurement (server21, 2026-10-11)

§6 finding 3 left open whether the ~17 ms/step Python of `DiskRowCache._read_cached`
at prefill chunk size (n = tokens x head_count) is *exposed* or hidden under GPU
compute. Measured with the calibration overlay (`~/ab/b12x/sequence/_shared/disk_table_cal.py`
mounted over `disk_table.py`; `B12X_CAL_SLEEP_NS` injects a `time.sleep()` at the
top of every cached read — CPU time on the host step thread inside the timed
window; `B12X_CAL_TIMING=1` self-times the path): cloned production spec, GPU
frozen off serving, `--prefill-only --prefill-contexts 131072` cold prompts x3
passes, one arm per boot. Harness `~/ab/run_prefill.sh`, log `~/ab/prefill_r1.log`.

| arm | prefill tok/s | Δ vs mean(cache..cal4) | injected CPU/call |
|---|---|---|---|
| nocache (disk, no pool) | 6536 | — | — |
| cache host-tier 100k | 6603 (+1.0 % vs nocache) | +0.2…−0.6 % | 0 (measured path: 32.1 ms/call mean, n≈207, max 167 ms) |
| cal0 / cal2 / cal4 | 6631 / 6578 / 6651 | ±0.5 % (noise band) | +0 / +2 / +4 ms |
| cal8 | 6314 | **−4.6 %** | +8 ms |

Readings:

1. **Host work in the cached path is overlapped, not exposed, on discrete SM120
   prefill.** Injections up to 4 ms/call move throughput inside the arm-to-arm
   noise; the 8 ms/call injection — 20 % of the ~40.6 ms chunk cycle
   (96 304 ids = 268 tokens at the measured 6.6k tok/s) — costs 4.6 %. If the
   host were the critical path, 8 ms on a 32 ms base would be ~25 %.
2. **Self-time, n=209 calls / 12.2 M ids: mean 32-33 ms, max 167 ms** (first
   chunks populate the 100k pool; steady-state hits land near the 181 ns/id
   constant from §1). The 17 ms §1 estimate scaled to the warm full 96 304-id
   chunk; the pool-resident miss wave (~0.42 ms at 0.985 hit rate) rides the
   io_uring thread, not the step thread.
3. **C-port verdict, prefill regime: no measurable win today.** Removing the
   Python entirely could return at most the exposed fraction, ~0.5-1 % and
   inside noise. The gate stays open with an explicit *trigger* rather than
   staying vaguely open: the host currently fits in ~79 % of the chunk cycle;
   when GPU-side work (b12x MoE/dense targets, §census in the RCA gist) pushes
   the cycle under ~32 ms, Python becomes the critical path and the port must
   be revisited.
4. **Caveats:** single arm boot per cal point (±0.5 % observed spread); hit
   rate 0.985 reflects the repeated 131k prompt (pass 1 warms the pool), so
   the miss-wave term is understated — but it is the Python term the C port
   targets, and that term was measured across the full id stream; cold-cache
   prefill is a separate regime and is NOT covered here (its disk time is
   NVMe, already ruled non-bookkeeping in §6); GB10's ARM per-id Python
   constant remains unmeasured — the §6(b) half of the gate.

**Trigger analysis (theoretical, from the §8 constants; not measured).** The
gate is a race between the GPU chunk cycle and the host's per-call Python
(32.1 ms at 96 304 ids). Break-even is a 1.26x faster GPU **on this host**:

| board | GPU cycle (ms) | host Python (ms) | exposed | C-port prize |
|---|---|---|---|---|
| Max-Q 300 W (server21, measured) | 40.6 | 32.1 | 0 % | 0 % |
| Workstation 600 W / Server 400-600 W¹ | 32.5-38.7 | 32.1 | 0-0.5 % | ≈ nothing — same die, clocks-only ≈ 1.05-1.25x² |
| VR200 + 88-core Vera host (40 ns/id³) | 8.1-14.5 | 12.3 | 0 % (HBM4-capped 2.8x) / 51 % (5x) | up to ~10 % of wall |
| VR200 + old N1-class host | 8.1-14.5 | 32.1 | 121-295 % | the host *is* the bottleneck |

¹ NVIDIA lists 600 W Workstation / 400-600 W Server editions vs the 300 W
   Max-Q in server21's `nvidia-smi`; same GB100 die + 96 GB GDDR7.
² GDDR7 spec is unchanged across SKUs, so a memory-bound prefill phase gains
   little from TDP; only the compute-bound part scales.
³ 4.5x better than the Neoverse-N1 constant, justified by modern OoO cores;
   server21's own constant would drop ~2.5x just by holding its 3 GHz ceiling
   (`CPU max MHz: 3000`, observed scaling 39 %) — the Python term is
   host-throttled here, and a cheap re-check on a pinned-clock host is due
   before trusting the 1.26x break-even.

Honest ceiling of this table: every cell scales the §8 cycle by announced
FLOPS/bandwidth ratios (VR200: 50 PFLOPS NVFP4, 22 TB/s HBM4) — theoretical,
no board in the last two rows has run this stack. The *decision* it supports:
on the 600 W part, do not build the C port; on Vera-class hardware the per-id
host constant, not the GPU, sets whether prefill is host-bound, and that
number must be measured on the actual host before the gate can flip.
