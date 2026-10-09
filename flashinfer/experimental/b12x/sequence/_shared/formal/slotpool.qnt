/**
 * Zero-copy SIEVE row cache: deferred-free slot-reclamation safety.
 *
 * The GPU decode gathers rows directly from persistent pool slots (no staging
 * copy). A miss may recycle a slot ONLY after every outstanding decode that
 * referenced it has retired. Safety = NoUseAfterFree: a slot an outstanding
 * reader is gathering from stays "resident" and still lists that reader among
 * its inflight holders.
 *
 * Run:
 *   quint verify slotpool.qnt --main verified --invariant=NoUseAfterFree --inductive-invariant=IndInv
 *   quint verify slotpool.qnt --main broken   --invariant=NoUseAfterFree --max-steps=15
 */
module slotpool {
  type Key = str
  type Slot = int
  type Reader = int
  type Cell = { status: str, key: Key, inflight: Set[Reader] }

  // Minimal universes that still exhibit a use-after-free.
  pure val Slots: Set[Slot] = 0.to(2)
  pure val Keys: Set[Key] = Set("a", "b", "c")
  pure val Readers: Set[Reader] = 1.to(2)
  pure val Statuses: Set[str] = Set("free", "loading", "resident")
  pure val KeyU: Set[Key] = Keys.union(Set(""))

  pure val cellUniverse: Set[Cell] =
    tuples(Statuses, KeyU, Readers.powerset())
      .map(((st, k, inf)) => { status: st, key: k, inflight: inf })

  const DEFER_FREE: bool

  var pool: Slot -> Cell
  var reading: Reader -> Set[Slot]

  pure def freeCell: Cell = { status: "free", key: "", inflight: Set() }

  // Frame: Apalache requires each var constrained before it is read.
  def frame = and {
    pool.in(setOfMaps(Slots, cellUniverse)),
    reading.in(setOfMaps(Readers, Slots.powerset())),
  }

  action init = all {
    pool' = Slots.mapBy(_ => freeCell),
    reading' = Readers.mapBy(_ => Set()),
  }

  action step = {
    nondet r = Readers.oneOf()
    nondet k = Keys.oneOf()
    nondet s = Slots.oneOf()
    all {
    frame,
    any {
      // MISS: recycle a free slot, begin async DMA into it.
      all {
        pool.get(s).status == "free",
        pool' = pool.set(s, { status: "loading", key: k, inflight: Set() }),
        reading' = reading,
      },
      // DMA completed: row present, readable, still zero holders.
      all {
        pool.get(s) == { status: "loading", key: k, inflight: Set() },
        pool' = pool.set(s, { status: "resident", key: k, inflight: Set() }),
        reading' = reading,
      },
      // HIT + launch: reader r begins gathering from a slot resident on k.
      all {
        pool.get(s).status == "resident" and pool.get(s).key == k,
        pool' = pool.set(s,
          { status: "resident", key: k, inflight: pool.get(s).inflight.union(Set(r)) }),
        reading' = reading.set(r, reading.get(r).union(Set(s))),
      },
      // RETIRE: r's decode finished; drop r from every slot it referenced.
      all {
        pool' = Slots.mapBy(t =>
          if (reading.get(r).contains(t))
            { status: "resident", key: pool.get(t).key,
              inflight: pool.get(t).inflight.exclude(Set(r)) }
          else pool.get(t)),
        reading' = reading.set(r, Set()),
      },
      // FREE: recycle a resident slot. THE RULE UNDER TEST.
      all {
        pool.get(s).status == "resident" and (not(DEFER_FREE) or
          (pool.get(s).inflight == Set() and
            Readers.forall(rr => not(reading.get(rr).contains(s))))),
        pool' = pool.set(s, freeCell),
        reading' = reading,
      },
      // tick (stutter).
      all { pool' = pool, reading' = reading },
    }
    }
  }

  /// A resident slot's inflight set is EXACTLY the readers pointing at it.
  val inflightLinksReaders = Slots.forall(s =>
    (pool.get(s).status == "resident") implies
      (pool.get(s).inflight == Readers.filter(rr => reading.get(rr).contains(s))))

  /// SAFETY: a slot some reader references is resident and lists that reader.
  val NoUseAfterFree = Readers.forall(r => reading.get(r).forall(s =>
    pool.get(s).status == "resident" and pool.get(s).inflight.contains(r)))

  val IndInv = and { frame, inflightLinksReaders, NoUseAfterFree }
}

module verified { import slotpool(DEFER_FREE = true).* }
module broken { import slotpool(DEFER_FREE = false).* }
