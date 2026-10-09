# Formal verification results — deferred-free row-cache reclamation

Property under test: **NoUseAfterFree** — a slot an outstanding decode references
is `resident` and lists that reader among its inflight holders. Protocol: a miss
may recycle a slot only after every outstanding reader of it has retired.

## Toolchain (recorded so the run is reproducible)

| Tool | Version | Source |
|---|---|---|
| `quint` | 0.32.0 | Homebrew (`/home/linuxbrew/.linuxbrew/bin/quint`), Rust backend |
| Apalache | pinned, auto-fetched on demand | via `quint verify` (ADR-008 managing-apalache) |
| `java` (JRE for Apalache) | OpenJDK 27 | system |

Spec: `slotpool.qnt`. Two instantiation modules differ only in `DEFER_FREE`:
`verified = {DEFER_FREE=true}`, `broken = {DEFER_FREE=false}`.

## Commands and results

```sh
# Unbounded proof (inductive invariant) of the honest reclamation rule:
quint verify slotpool.qnt --main verified --invariant=NoUseAfterFree --inductive-invariant=IndInv
#   > [1/3] Checking whether the inductive invariant 'IndInv' holds in the initial state(s) defined by 'init'...
#   > [2/3] Checking whether 'step' preserves the inductive invariant 'IndInv'...
#   > [3/3] Checking whether the inductive invariant 'IndInv' implies 'NoUseAfterFree'...
#   [ok] No violation found (1886ms).

# Negative test: the checker MUST reject the broken rule with a counterexample:
quint verify slotpool.qnt --main broken --invariant=NoUseAfterFree --max-steps=10 --out-itf broken-use-after-free.itf.json
#   [violation] Found an issue (1299ms).

# Random-sim sanity (both arms), quick smoke:
quint run slotpool.qnt --main verified --invariant=NoUseAfterFree --max-steps=30 --seed=0x1   # [ok]
```

## Per-property table

| Module | DEFER_FREE | Property | Method | Result |
|---|---|---|---|---|
| verified | true | NoUseAfterFree | inductive invariant (Apalache) | **PROVED** (unbounded) |
| broken | false | NoUseAfterFree | bounded model check (Apalache, 10 steps) | **REFUTED** — counterexample |

The model is genuinely discriminating: identical spec, one boolean, opposite
verdicts. A gate that only ever passes is worthless; this one fails on the
broken rule.

## Counterexample narrative (broken arm, 5 states)

Slot 0 (of 3) exercises the whole bug; reader 1 (of 2), key `"a"`:

| # | slot 0 | reading[1] | transition |
|---|---|---|---|
| 0 | `{free,"",∅}` | ∅ | init |
| 1 | `{loading,"a",∅}` | ∅ | miss (allocate + DMA) |
| 2 | `{resident,"a",∅}` | ∅ | dma_complete |
| 3 | `{resident,"a",{1}}` | `{0}` | reader 1 launches gather |
| 4 | `{free,"",∅}` | **`{0}`** | **free ignores the outstanding reader** |

At state 4, reader 1 still references slot 0 but slot 0 is `free` (and would be
reallocated to a different row on the next miss). `NoUseAfterFree` fails: the
referenced slot is not `resident` and does not list reader 1. This is exactly the
use-after-free a real GPU would hit if `free_slot` dropped the retire fence. Full
machine-readable trace: `broken-use-after-free.itf.json` (ITF; view in the VSCode
Trace Viewer or `jq`).

## Inductive invariant (`IndInv`)

`IndInv = ∧ { frame, inflightLinksReaders, NoUseAfterFree }` where
- `frame`: `pool ∈ setOfMaps(Slots, cellUniverse)`, `reading ∈ setOfMaps(Readers,
  Slots.powerset())` — the Apalache assignment-discipline domain guards (each var
  constrained before read; without them VCGen errors "pool is used before it is
  assigned").
- `inflightLinksReaders`: a `resident` slot's `inflight` set equals *exactly* the
  set of readers whose `reading[r]` contains it. This two-directional link is what
  makes the reclamation deferral inductive — it forbids the spurious
  "referenced-but-empty-inflight" and "inflight-but-unreferenced" states that
  otherwise break `2/3`.

`NoUseAfterFree` is the safety property; `IndInv ⇒ NoUseAfterFree` is check `3/3`.

## Modelling bounds and abstraction gaps

- Universes: `Slots = 0.to(2)` (3), `Readers = 1.to(2)` (2), `Keys = {a,b,c}`.
  The inductive proof is **unbounded in time** (all reachable states of the finite
  abstraction) but the state space is bounded by these constants; the property is
  structural (per-slot lifetime), not size-dependent, so small bounds are adequate
  — the counterexample needs only 1 slot, 1 reader, 1 key.
- **Not modelled (out of scope for this proof):** SIEVE eviction quality /
  hit-rate optimality; the Triton gather kernel's byte correctness (covered by the
  25 byte-identity tests in `test_disk_row_cache.py`); real CUDA event ordering
  and driver `IORING_REGISTER_BUFFERS` behaviour; io_uring DMA short-read/failure
  paths (`_ple_reader.c` guards these separately). The model captures the logical
  slot-lifecycle protocol the implementation must honour with per-step events.

## Gate to ship the C/zero-copy path on

All three of: (1) verified inductive `[ok]` above; (2) broken `[violation]` above;
(3) `test_disk_row_cache.py` byte-identity green on target. Plus the SM120
perf gate in `DESIGN.md` §6 before building the C reclamation path.
