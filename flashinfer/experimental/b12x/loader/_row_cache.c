// S3-FIFO host-tier index: a per-request mirror of the libCacheSim
// S3FIFO.c mirror (the S3FIFO-0.1000-2 variant) the repository shipped as
// S3FifoIndex in sequence/_shared/row_cache.py. That class is the
// behavioural oracle, now test-private as ``_PyS3FifoReference`` in
// tests/experimental/b12x/sequence/test_row_cache_policies.py; the
// request-facing contract here is byte-identical plan/store/reserve/
// confirm/cancel semantics, verified per-request against that oracle.
//
// Hot path: row_cache_read(count) resolves ids[:count] against the bound
// id plane, applies every hit/freq/ghost/queue mutation of plan(), and copies
// hit rows from the bound pool plane into the bound stage plane; then the
// caller fetches the miss wave into the bound miss planes and calls
// row_cache_store() for the miss staging and pool stores. No Python per-id
// work happens between create() and the counters. Every allocation is done at
// row_cache_create; read/store/reuse their persistent arrays and the stamped
// dedup tables, so the steady state is allocation-free.
//
// Reclamation follows formal/slotpool.qnt: a LOADING slot joins no queue and
// is never an eviction victim (no pop can reach it), so an outstanding decode
// that staged from a READY slot keeps its bytes until the transaction fence
// (disk_table.py transaction() / _cache_done) retires it; the C port changes
// no release ordering: releases happen only from the same pop paths the
// Python class uses, and stores land after all hit copies of their batch.
//
// Threading: all mutation happens under DiskRowCache._lock (the transaction
// holds it across the read; the prefetch worker takes it around its
// reserve/confirm/cancel sections), and every entry point runs with the GIL
// held, so the structure needs no lock of its own — the same discipline the
// Python class relies on.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef char require_64_bit_pointers[(sizeof(size_t) == 8) ? 1 : -1];

#define RC_ROW_CACHE_CAPSULE "b12x.row_cache"
#define RC_ROW_CACHE_MAX_CAPACITY (INT32_MAX - 16)
#define RC_MAX_PLANES 2
#define RC_MAX_FREQ 3u
#define RC_SMALL_SHARE 10
#define RC_GHOST_SHARE 90
#define RC_MOVE_TO_MAIN_THRESHOLD 2u

// Slot states and queue tags, numerically identical to S3FifoIndex._FREE /
// _LOADING / _READY and _UNQUEUED / _SMALL / _MAIN.
enum { RC_FREE = 0, RC_LOADING = 1, RC_READY = 2 };
enum { RC_UNQUEUED = 0, RC_SMALL = 1, RC_MAIN = 2 };

typedef struct {
  int nplanes;
  size_t row_bytes[RC_MAX_PLANES];
  int64_t capacity, small_cap, main_cap, ghost_cap, max_lookups;
  // Ghost entry universe: _demote_ghost inserts *before* trimming (the
  // Python dict momentarily holds cap+1 keys), so entries are sized cap + 2.
  int64_t ghost_entries;
  // Bound host buffers; the Py_buffer refs keep them alive (contract with
  // DiskRowCache._init_row_cache: ids/miss planes are pinned, pool/stage are
  // the host-tier numpy/mapped-host planes the Python path addresses).
  Py_buffer ids, miss_ids;
  Py_buffer miss[RC_MAX_PLANES], stage[RC_MAX_PLANES], pool[RC_MAX_PLANES];
  int64_t* ids_buf;
  int64_t* miss_ids_buf;
  uint8_t* miss_plane[RC_MAX_PLANES];
  uint8_t* stage_plane[RC_MAX_PLANES];
  uint8_t* pool_plane[RC_MAX_PLANES];
  // Pool state, one entry per slot (S3FifoIndex._key/_freq/_state/_queue).
  int64_t* key;
  uint8_t* freq;
  uint8_t* state;
  uint8_t* queue;
  // key -> slot map for READY and LOADING rows: open-chained buckets. A key
  // is never READY and LOADING at once (reserve rejects either, confirm is
  // state-guarded), so one table answers both ``key in _slot`` and
  // ``key in _loading``.
  int32_t* hash_next;
  int32_t* buckets;
  int64_t bucket_mask;
  // FIFO rings over slot ids (S3FifoIndex._small/_main deques) and the LIFO
  // free list (_free: list.pop() takes the most recently released slot).
  int32_t* small_ring;
  int32_t* main_ring;
  int64_t small_head, small_count, main_head, main_count;
  int32_t* free_stack;
  int64_t free_top;
  int64_t used, ready_count, loading_count;
  int has_evicted;
  // Ghost dict (_ghost): insertion-ordered key set with delete-any. Entries
  // are indices into the ghost arrays; order links + hash chains.
  int64_t* ghost_key;
  int32_t* ghost_prev;
  int32_t* ghost_next;
  int32_t* ghost_hash_next;
  int32_t* ghost_buckets;
  int64_t ghost_bucket_mask;
  int32_t* ghost_free_stack;
  int64_t ghost_free_top, ghost_count, ghost_head, ghost_tail;
  // Persistent per-batch arrays sized max_lookups (RowPlan fields) plus the
  // stamped dedup tables that replace the plan() ``fetched``/``hit_on_ghost``
  // and ``stored`` dicts without clearing between steps.
  int32_t *hit_pos, *hit_slot, *miss_pos, *miss_row, *store_slot, *store_row;
  uint8_t* ghost_latch;
  int32_t n_hit, n_miss_ids, n_miss_pos, n_store;
  int64_t* dedup_key;
  int32_t* dedup_val;
  uint32_t* dedup_stamp;
  uint32_t dedup_gen;
  int32_t* store_probe_slot;
  int32_t* store_probe_idx;
  uint32_t* store_probe_stamp;
  uint32_t store_gen;
  int64_t probe_mask;
  int64_t hits, misses;
} row_cache_t;

static uint64_t rc_mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

static int64_t rc_pow2_ceil(int64_t value) {
  int64_t n = 1;
  while (n < value) n <<= 1;
  return n;
}

// ---------------------------------------------------------------------------
// key -> slot map (mirrors S3FifoIndex._slot plus _loading)
// ---------------------------------------------------------------------------

static int32_t rc_tbl_find(row_cache_t* rc, int64_t key) {
  int64_t i = rc->buckets[rc_mix64((uint64_t)key) & rc->bucket_mask];
  while (i >= 0) {
    if (rc->key[i] == key) return i;
    i = rc->hash_next[i];
  }
  return -1;
}

static void rc_tbl_add(row_cache_t* rc, int64_t key, int32_t slot) {
  int64_t h = rc_mix64((uint64_t)key) & rc->bucket_mask;
  rc->hash_next[slot] = rc->buckets[h];
  rc->buckets[h] = slot;
}

static void rc_tbl_del(row_cache_t* rc, int64_t key) {
  int64_t h = rc_mix64((uint64_t)key) & rc->bucket_mask;
  int32_t* link = &rc->buckets[h];
  while (*link >= 0) {
    int32_t slot = *link;
    if (rc->key[slot] == key) {
      *link = rc->hash_next[slot];
      rc->hash_next[slot] = -1;
      return;
    }
    link = &rc->hash_next[slot];
  }
}

// ---------------------------------------------------------------------------
// FIFO rings (S3FifoIndex._small / _main append + popleft)
// ---------------------------------------------------------------------------

static void rc_ring_push(int32_t* ring, int64_t size, int64_t* head, int64_t* count, int32_t slot) {
  int64_t tail = (*head + *count) % size;
  ring[tail] = slot;
  *count += 1;
}

static int32_t rc_ring_pop(int32_t* ring, int64_t size, int64_t* head, int64_t* count) {
  int32_t slot = ring[*head];
  *head = (*head + 1) % size;
  *count -= 1;
  return slot;
}

// ---------------------------------------------------------------------------
// ghost dict (S3FifoIndex._ghost: dict-of-None, insertion order)
// ---------------------------------------------------------------------------

static bool rc_ghost_contains(row_cache_t* rc, int64_t key) {
  if (rc->ghost_cap == 0) return false;
  int32_t e = rc->ghost_buckets[rc_mix64((uint64_t)key) & rc->ghost_bucket_mask];
  while (e >= 0) {
    if (rc->ghost_key[e] == key) return true;
    e = rc->ghost_hash_next[e];
  }
  return false;
}

static void rc_ghost_unlink_hash(row_cache_t* rc, int32_t entry) {
  int64_t h = rc_mix64((uint64_t)rc->ghost_key[entry]) & rc->ghost_bucket_mask;
  int32_t* link = &rc->ghost_buckets[h];
  while (*link >= 0) {
    if (*link == entry) {
      *link = rc->ghost_hash_next[entry];
      rc->ghost_hash_next[entry] = -1;
      return;
    }
    link = &rc->ghost_hash_next[*link];
  }
}

// self._ghost[key] = None: an existing key keeps its position (dict
// overwrite), a new one appends at the tail.
static void rc_ghost_push(row_cache_t* rc, int64_t key) {
  if (rc->ghost_cap == 0) return;  // trimmed to zero immediately anyway
  if (rc_ghost_contains(rc, key)) return;
  int32_t e = rc->ghost_free_stack[--rc->ghost_free_top];
  rc->ghost_key[e] = key;
  rc->ghost_prev[e] = rc->ghost_tail;
  rc->ghost_next[e] = -1;
  if (rc->ghost_tail >= 0) {
    rc->ghost_next[rc->ghost_tail] = e;
  } else {
    rc->ghost_head = e;
  }
  rc->ghost_tail = e;
  int64_t h = rc_mix64((uint64_t)key) & rc->ghost_bucket_mask;
  rc->ghost_hash_next[e] = rc->ghost_buckets[h];
  rc->ghost_buckets[h] = e;
  rc->ghost_count += 1;
}

// pop(next(iter(self._ghost))): the oldest entry leaves.
static void rc_ghost_pop_first(row_cache_t* rc) {
  int32_t e = rc->ghost_head;
  rc->ghost_head = rc->ghost_next[e];
  if (rc->ghost_head < 0) {
    rc->ghost_tail = -1;
  } else {
    rc->ghost_prev[rc->ghost_head] = -1;
  }
  rc_ghost_unlink_hash(rc, e);
  rc->ghost_free_stack[rc->ghost_free_top++] = e;
  rc->ghost_count -= 1;
}

static void rc_ghost_del(row_cache_t* rc, int64_t key) {
  if (rc->ghost_cap == 0) return;
  int32_t e = rc->ghost_buckets[rc_mix64((uint64_t)key) & rc->ghost_bucket_mask];
  while (e >= 0) {
    if (rc->ghost_key[e] == key) break;
    e = rc->ghost_hash_next[e];
  }
  if (e < 0) return;
  if (rc->ghost_prev[e] >= 0) {
    rc->ghost_next[rc->ghost_prev[e]] = rc->ghost_next[e];
  } else {
    rc->ghost_head = rc->ghost_next[e];
  }
  if (rc->ghost_next[e] >= 0) {
    rc->ghost_prev[rc->ghost_next[e]] = rc->ghost_prev[e];
  } else {
    rc->ghost_tail = rc->ghost_prev[e];
  }
  rc_ghost_unlink_hash(rc, e);
  rc->ghost_free_stack[rc->ghost_free_top++] = e;
  rc->ghost_count -= 1;
}

// S3FifoIndex._demote_ghost.
static void rc_demote_ghost(row_cache_t* rc, int64_t key) {
  rc_ghost_push(rc, key);
  while (rc->ghost_count > rc->ghost_cap) rc_ghost_pop_first(rc);
}

// ---------------------------------------------------------------------------
// pool slot lifecycle (S3FifoIndex._release / _bind / _bind_loaded)
// ---------------------------------------------------------------------------

static void rc_release(row_cache_t* rc, int32_t slot) {
  rc_tbl_del(rc, rc->key[slot]);
  rc->freq[slot] = 0;
  rc->queue[slot] = RC_UNQUEUED;
  rc->state[slot] = RC_FREE;
  rc->free_stack[rc->free_top++] = slot;
  rc->ready_count -= 1;
}

static void rc_bind(row_cache_t* rc, int32_t slot, int64_t key, bool main) {
  rc_tbl_add(rc, key, slot);
  rc->key[slot] = key;
  rc->freq[slot] = 0;
  rc->state[slot] = RC_READY;
  rc->queue[slot] = main ? RC_MAIN : RC_SMALL;
  if (main) {
    rc_ring_push(rc->main_ring, rc->capacity, &rc->main_head, &rc->main_count, slot);
  } else {
    rc_ring_push(rc->small_ring, rc->capacity, &rc->small_head, &rc->small_count, slot);
  }
  rc->ready_count += 1;
}

// S3FifoIndex._pop_main: the 2-bit clock over M.
static void rc_pop_main(row_cache_t* rc) {
  while (rc->main_count) {
    int32_t slot = rc_ring_pop(rc->main_ring, rc->capacity, &rc->main_head, &rc->main_count);
    uint8_t f = rc->freq[slot];
    if (f >= 1) {
      rc->freq[slot] = (f < RC_MAX_FREQ ? f : RC_MAX_FREQ) - 1;
      rc_ring_push(rc->main_ring, rc->capacity, &rc->main_head, &rc->main_count, slot);
      continue;
    }
    rc_release(rc, slot);
    return;
  }
}

// S3FifoIndex._pop_small: recycle S's front; threshold credit promotes into M
// keeping its counter (overfill paid back at once), anything else demotes to
// a ghost and frees its slot. Promotions loop; only a demotion returns.
static void rc_pop_small(row_cache_t* rc) {
  while (rc->small_count) {
    int32_t slot = rc_ring_pop(rc->small_ring, rc->capacity, &rc->small_head, &rc->small_count);
    if (rc->freq[slot] >= RC_MOVE_TO_MAIN_THRESHOLD && rc->main_cap > 0) {
      rc->queue[slot] = RC_MAIN;
      rc_ring_push(rc->main_ring, rc->capacity, &rc->main_head, &rc->main_count, slot);
      while (rc->main_count > rc->main_cap) rc_pop_main(rc);
      continue;
    }
    int64_t key = rc->key[slot];
    rc_release(rc, slot);
    rc_demote_ghost(rc, key);
    return;
  }
}

// S3FifoIndex._evict_one: the eviction source rule, with the newcomer's
// planned destination counted into M's overfill test.
static void rc_evict_one(row_cache_t* rc, bool main) {
  rc->has_evicted = 1;
  if (rc->main_count + (main ? 1 : 0) > rc->main_cap || rc->small_count == 0) {
    rc_pop_main(rc);
  } else {
    rc_pop_small(rc);
  }
}

// S3FifoIndex._prepare_admission.
static void rc_prepare_admission(row_cache_t* rc, bool main) {
  while (rc->ready_count >= rc->capacity) {
    rc_evict_one(rc, main);
    if (rc->free_top) return;
  }
}

// S3FifoIndex._settle_main_budget.
static void rc_settle_main_budget(row_cache_t* rc) {
  while (rc->main_count > rc->main_cap) rc_pop_main(rc);
}

// S3FifoIndex._admit. latch == -1 mirrors hit_on_ghost=None (read the ghost
// set now); otherwise it is the value plan() latched at lookup time.
static void rc_admit(row_cache_t* rc, int64_t key, int latch, bool* out_main, bool* out_ghost) {
  bool hit_on_ghost = latch >= 0 ? (bool)latch : rc_ghost_contains(rc, key);
  if (hit_on_ghost && rc->main_cap > 0) {
    rc_prepare_admission(rc, true);
    *out_main = true;
    *out_ghost = true;
    return;
  }
  rc_prepare_admission(rc, false);
  // S3FifoIndex._ghost_admits_main: the C's has_evicted fast path.
  *out_main =
      !rc->has_evicted && rc->small_count >= rc->small_cap && rc->main_cap > 0;
  *out_ghost = hit_on_ghost;
}

static void rc_copy_row(row_cache_t* rc, uint8_t* const* dst_planes, const uint8_t* const* src_planes,
                        int64_t dst_row, int64_t src_row) {
  for (int p = 0; p < rc->nplanes; p++) {
    memcpy(dst_planes[p] + (size_t)dst_row * rc->row_bytes[p],
           src_planes[p] + (size_t)src_row * rc->row_bytes[p], rc->row_bytes[p]);
  }
}

// S3FifoIndex.confirm. Idempotent off a non-loading slot, exactly like the
// Python method.
static int32_t rc_confirm(row_cache_t* rc, int32_t slot) {
  int64_t key = rc->key[slot];
  if (rc->state[slot] != RC_LOADING) return slot;
  rc_tbl_del(rc, key);
  rc->loading_count -= 1;
  bool main, ghost;
  rc_admit(rc, key, -1, &main, &ghost);
  // _bind_loaded: same shape as _bind for an already-owned, unqueued slot.
  rc_bind(rc, slot, key, main);
  if (ghost) rc_ghost_del(rc, key);
  rc_settle_main_budget(rc);
  return slot;
}

// S3FifoIndex.insert. Returns the slot, -1 when no slot can be taken, or -2
// for the "already cached" condition the Python method raises KeyError on
// (unreachable from plan because miss ids are distinct, still surfaced).
static int32_t rc_insert(row_cache_t* rc, int64_t key, int latch) {
  int32_t slot = rc_tbl_find(rc, key);
  if (slot >= 0) {
    if (rc->state[slot] == RC_LOADING) return rc_confirm(rc, slot);
    return -2;
  }
  bool main, ghost;
  rc_admit(rc, key, latch, &main, &ghost);
  if (ghost) rc_ghost_del(rc, key);
  if (rc->free_top) {
    slot = rc->free_stack[--rc->free_top];
  } else if (rc->used < rc->capacity) {
    slot = (int32_t)rc->used++;
  } else {
    return -1;
  }
  rc_bind(rc, slot, key, main);
  rc_settle_main_budget(rc);
  return slot;
}

// S3FifoIndex._claim_for_loading.
static int32_t rc_claim_for_loading(row_cache_t* rc) {
  if (rc->free_top) return rc->free_stack[--rc->free_top];
  if (rc->used < rc->capacity) return (int32_t)rc->used++;
  if (rc->ready_count <= 1) {
    // Keep at least one committed row evictable so the synchronous read path
    // can never starve behind a queue of in-flight prefetches.
    return -1;
  }
  rc_evict_one(rc, false);
  return rc->free_top ? rc->free_stack[--rc->free_top] : -1;
}

// S3FifoIndex.reserve.
static int32_t rc_reserve(row_cache_t* rc, int64_t key) {
  if (rc_tbl_find(rc, key) >= 0) return -1;
  int32_t slot = rc_claim_for_loading(rc);
  if (slot < 0) return -1;
  rc->key[slot] = key;
  rc->freq[slot] = 0;
  rc->queue[slot] = RC_UNQUEUED;
  rc->state[slot] = RC_LOADING;
  rc_tbl_add(rc, key, slot);
  rc->loading_count += 1;
  return slot;
}

// S3FifoIndex.cancel.
static void rc_cancel(row_cache_t* rc, int32_t slot) {
  if (rc->state[slot] != RC_LOADING) return;
  rc_tbl_del(rc, rc->key[slot]);
  rc->loading_count -= 1;
  rc->freq[slot] = 0;
  rc->queue[slot] = RC_UNQUEUED;
  rc->state[slot] = RC_FREE;
  rc->free_stack[rc->free_top++] = slot;
}

// S3FifoIndex.get: pure lookup, raises the 2-bit counter on a resident hit
// and never consumes the ghost entry.
static int32_t rc_get(row_cache_t* rc, int64_t key) {
  int32_t slot = rc_tbl_find(rc, key);
  if (slot < 0 || rc->state[slot] != RC_READY) return -1;
  if (rc->freq[slot] < RC_MAX_FREQ) rc->freq[slot] += 1;
  return slot;
}

// ---------------------------------------------------------------------------
// stamped dedup tables (plan's ``fetched`` / latched ``hit_on_ghost`` and the
// ``stored`` slot dict: first-insert order kept, value updated)
// ---------------------------------------------------------------------------

#define RC_NOT_FOUND INT32_MIN

static int32_t rc_dedup_find(row_cache_t* rc, int64_t key) {
  int64_t h = rc_mix64((uint64_t)key) & rc->probe_mask;
  while (rc->dedup_stamp[h] == rc->dedup_gen) {
    if (rc->dedup_key[h] == key) return rc->dedup_val[h];
    h = (h + 1) & rc->probe_mask;
  }
  return RC_NOT_FOUND;
}

static void rc_dedup_add(row_cache_t* rc, int64_t key, int32_t val) {
  int64_t h = rc_mix64((uint64_t)key) & rc->probe_mask;
  while (rc->dedup_stamp[h] == rc->dedup_gen) h = (h + 1) & rc->probe_mask;
  rc->dedup_stamp[h] = rc->dedup_gen;
  rc->dedup_key[h] = key;
  rc->dedup_val[h] = val;
}

static int32_t rc_store_find(row_cache_t* rc, int32_t slot) {
  int64_t h = rc_mix64((uint64_t)(uint32_t)slot) & rc->probe_mask;
  while (rc->store_probe_stamp[h] == rc->store_gen) {
    if (rc->store_probe_slot[h] == slot) return rc->store_probe_idx[h];
    h = (h + 1) & rc->probe_mask;
  }
  return RC_NOT_FOUND;
}

static void rc_store_add(row_cache_t* rc, int32_t slot, int32_t idx) {
  int64_t h = rc_mix64((uint64_t)(uint32_t)slot) & rc->probe_mask;
  while (rc->store_probe_stamp[h] == rc->store_gen) h = (h + 1) & rc->probe_mask;
  rc->store_probe_stamp[h] = rc->store_gen;
  rc->store_probe_slot[h] = slot;
  rc->store_probe_idx[h] = idx;
}

// ---------------------------------------------------------------------------
// module glue
// ---------------------------------------------------------------------------

static void row_cache_release(row_cache_t* rc) {
  if (!rc) return;
  PyBuffer_Release(&rc->ids);
  PyBuffer_Release(&rc->miss_ids);
  for (int p = 0; p < rc->nplanes; p++) {
    PyBuffer_Release(&rc->miss[p]);
    PyBuffer_Release(&rc->stage[p]);
    PyBuffer_Release(&rc->pool[p]);
  }
  free(rc->key);
  free(rc->freq);
  free(rc->state);
  free(rc->queue);
  free(rc->hash_next);
  free(rc->buckets);
  free(rc->small_ring);
  free(rc->main_ring);
  free(rc->free_stack);
  free(rc->ghost_key);
  free(rc->ghost_prev);
  free(rc->ghost_next);
  free(rc->ghost_hash_next);
  free(rc->ghost_buckets);
  free(rc->ghost_free_stack);
  free(rc->hit_pos);
  free(rc->hit_slot);
  free(rc->miss_pos);
  free(rc->miss_row);
  free(rc->store_slot);
  free(rc->store_row);
  free(rc->ghost_latch);
  free(rc->dedup_key);
  free(rc->dedup_val);
  free(rc->dedup_stamp);
  free(rc->store_probe_slot);
  free(rc->store_probe_idx);
  free(rc->store_probe_stamp);
  free(rc);
}

static void rc_delete(PyObject* capsule) {
  row_cache_t* rc = PyCapsule_GetPointer(capsule, RC_ROW_CACHE_CAPSULE);
  if (rc) row_cache_release(rc);
}

static int rc_check_1d_int64(const Py_buffer* b, const char* name) {
  if (b->ndim != 1 || b->itemsize != 8 || !PyBuffer_IsContiguous(b, 'C') ||
      (b->format && strcmp(b->format, "l") != 0 && strcmp(b->format, "q") != 0) ||
      (b->strides && b->strides[0] != b->itemsize)) {
    PyErr_Format(PyExc_ValueError,
                 "row cache %s must be a C-contiguous signed int64 1-D buffer", name);
    return -1;
  }
  return 0;
}

static int rc_check_plane(const Py_buffer* b, int64_t rows, const char* name) {
  bool row_major = !b->strides || (b->strides[1] == b->itemsize &&
                                   b->strides[0] == b->shape[1] * b->itemsize);
  if (b->ndim != 2 || b->itemsize != 1 || !PyBuffer_IsContiguous(b, 'C') || !row_major ||
      b->shape[0] != rows) {
    PyErr_Format(PyExc_ValueError,
                 "row cache %s must be a C-contiguous uint8 plane of shape (%lld, width)", name,
                 (long long)rows);
    return -1;
  }
  return 0;
}

static int rc_bind_plane(PyObject* tuple, int plane, Py_buffer* out, const char* name) {
  (void)name;  // diagnostics live in rc_check_plane's caller-side validation
  PyObject* item = PyTuple_GET_ITEM(tuple, plane);
  // PyBUF_STRIDED so rc_check_plane sees the exporter's real strides: a
  // strided view of a larger tensor must fail, not be assumed row-major.
  if (PyObject_GetBuffer(item, out, PyBUF_STRIDED) < 0) return -1;
  return 0;
}


static PyObject* py_row_cache_create(PyObject* self, PyObject* args) {
  (void)self;
  long long capacity, max_lookups;
  PyObject *ids_object, *miss_ids_object, *miss_tuple, *stage_tuple, *pool_tuple;
  if (!PyArg_ParseTuple(args, "LLOOOOO", &capacity, &max_lookups, &ids_object, &miss_ids_object,
                        &miss_tuple, &stage_tuple, &pool_tuple))
    return NULL;
  if (capacity <= 0) {
    PyErr_SetString(PyExc_ValueError, "row cache capacity must be positive");
    return NULL;
  }
  if (capacity > RC_ROW_CACHE_MAX_CAPACITY) {
    PyErr_SetString(PyExc_ValueError, "row cache capacity exceeds the 32-bit slot limit");
    return NULL;
  }
  if (max_lookups <= 0 || max_lookups > RC_ROW_CACHE_MAX_CAPACITY) {
    PyErr_SetString(PyExc_ValueError, "row cache batch capacity must be positive and fit a "
                                      "32-bit slot index");
    return NULL;
  }
  if (!PyTuple_Check(miss_tuple) || !PyTuple_Check(stage_tuple) || !PyTuple_Check(pool_tuple) ||
      PyTuple_GET_SIZE(miss_tuple) != PyTuple_GET_SIZE(stage_tuple) ||
      PyTuple_GET_SIZE(miss_tuple) != PyTuple_GET_SIZE(pool_tuple) ||
      PyTuple_GET_SIZE(miss_tuple) < 1 || PyTuple_GET_SIZE(miss_tuple) > RC_MAX_PLANES) {
    PyErr_SetString(PyExc_ValueError,
                    "row cache planes must be matching 1- or 2-tuples of uint8 buffers");
    return NULL;
  }
  row_cache_t* rc = calloc(1, sizeof(row_cache_t));
  if (!rc) return PyErr_NoMemory();
  rc->capacity = capacity;
  rc->max_lookups = max_lookups;
  rc->small_cap = capacity * RC_SMALL_SHARE / 100;
  if (rc->small_cap < 1) rc->small_cap = 1;
  rc->main_cap = capacity - rc->small_cap;
  rc->ghost_cap = capacity * RC_GHOST_SHARE / 100;
  rc->nplanes = (int)PyTuple_GET_SIZE(miss_tuple);
  rc->hash_next = rc->buckets = rc->small_ring = rc->main_ring = rc->free_stack = NULL;
  rc->key = NULL;
  rc->freq = rc->state = rc->queue = NULL;
  rc->ghost_key = NULL;
  rc->ghost_prev = rc->ghost_next = rc->ghost_hash_next = rc->ghost_buckets =
      rc->ghost_free_stack = NULL;
  rc->hit_pos = rc->hit_slot = rc->miss_pos = rc->miss_row = rc->store_slot = rc->store_row = NULL;
  rc->ghost_latch = NULL;
  rc->dedup_key = NULL;
  rc->dedup_val = NULL;
  rc->dedup_stamp = NULL;
  rc->store_probe_slot = rc->store_probe_idx = NULL;
  rc->store_probe_stamp = NULL;
  rc->ghost_head = rc->ghost_tail = -1;
  rc->small_head = rc->main_head = 0;
  int64_t nbuckets = rc_pow2_ceil(capacity < 4 ? 4 : capacity);
  rc->ghost_entries = rc->ghost_cap + 2;
  int64_t nghost_buckets = rc_pow2_ceil(rc->ghost_entries < 4 ? 4 : rc->ghost_entries);
  int64_t probes = rc_pow2_ceil(max_lookups * 2 < 8 ? 8 : max_lookups * 2);
  rc->bucket_mask = nbuckets - 1;
  rc->ghost_bucket_mask = nghost_buckets - 1;
  rc->probe_mask = probes - 1;

  bool ok = true;
  ok &= PyObject_GetBuffer(ids_object, &rc->ids, PyBUF_STRIDED_RO) == 0;
  if (ok && rc_check_1d_int64(&rc->ids, "ids") != 0) ok = false;
  if (ok && rc->ids.len < max_lookups * 8) {
    PyErr_SetString(PyExc_ValueError, "row cache ids buffer must cover max_lookups entries");
    ok = false;
  }
  if (ok && PyObject_GetBuffer(miss_ids_object, &rc->miss_ids, PyBUF_STRIDED_RO) == 0) {
    if (rc_check_1d_int64(&rc->miss_ids, "miss ids") != 0) ok = false;
  } else {
    ok = false;
  }
  if (ok && rc->miss_ids.len < max_lookups * 8) {
    PyErr_SetString(PyExc_ValueError, "row cache miss ids buffer must cover max_lookups entries");
    ok = false;
  }
  for (int p = 0; ok && p < rc->nplanes; p++) {
    ok &= rc_bind_plane(miss_tuple, p, &rc->miss[p], "miss") == 0;
    ok &= rc_check_plane(&rc->miss[p], max_lookups, "miss") == 0;
    if (ok) {
      ok &= rc_bind_plane(stage_tuple, p, &rc->stage[p], "stage") == 0;
      if (ok) ok &= rc_check_plane(&rc->stage[p], max_lookups, "stage") == 0;
    }
    if (ok) {
      ok &= rc_bind_plane(pool_tuple, p, &rc->pool[p], "pool") == 0;
      if (ok) ok &= rc_check_plane(&rc->pool[p], capacity, "pool") == 0;
    }
    if (ok) {
      // Planes are independent row geometries (weight 257 B, scale 9 B in
      // the shipped PLE layout); the invariant is that plane p of the three
      // tuples describes the same rows and width.
      ok &= rc->stage[p].shape[1] == rc->miss[p].shape[1] &&
            rc->pool[p].shape[1] == rc->miss[p].shape[1];
      if (!ok)
        PyErr_Format(PyExc_ValueError,
                     "row cache plane %d widths differ across miss/stage/pool", p);
    }
    if (ok) rc->row_bytes[p] = (size_t)rc->pool[p].shape[1];
  }

#define RC_ALLOC(type, field, count, zero)                                                       \
  if (ok) {                                                                                      \
    size_t bytes = (size_t)(count) * sizeof(type);                                               \
    rc->field = zero ? calloc(1, bytes ? bytes : sizeof(type)) : malloc(bytes ? bytes : 1);      \
    if (!rc->field) {                                                                            \
      PyErr_NoMemory();                                                                          \
      ok = false;                                                                                \
    }                                                                                            \
  }
  RC_ALLOC(int64_t, key, capacity, true)
  RC_ALLOC(uint8_t, freq, capacity, true)
  RC_ALLOC(uint8_t, state, capacity, true)
  RC_ALLOC(uint8_t, queue, capacity, true)
  RC_ALLOC(int32_t, hash_next, capacity, false)
  RC_ALLOC(int32_t, small_ring, capacity, false)
  RC_ALLOC(int32_t, main_ring, capacity, false)
  RC_ALLOC(int32_t, free_stack, capacity, false)
  RC_ALLOC(int32_t, buckets, nbuckets, false)
  RC_ALLOC(int64_t, ghost_key, rc->ghost_entries, false)
  RC_ALLOC(int32_t, ghost_prev, rc->ghost_entries, false)
  RC_ALLOC(int32_t, ghost_next, rc->ghost_entries, false)
  RC_ALLOC(int32_t, ghost_hash_next, rc->ghost_entries, false)
  RC_ALLOC(int32_t, ghost_buckets, nghost_buckets, false)
  RC_ALLOC(int32_t, ghost_free_stack, rc->ghost_entries, false)
  RC_ALLOC(int32_t, hit_pos, max_lookups, false)
  RC_ALLOC(int32_t, hit_slot, max_lookups, false)
  RC_ALLOC(int32_t, miss_pos, max_lookups, false)
  RC_ALLOC(int32_t, miss_row, max_lookups, false)
  RC_ALLOC(int32_t, store_slot, max_lookups, false)
  RC_ALLOC(int32_t, store_row, max_lookups, false)
  RC_ALLOC(uint8_t, ghost_latch, max_lookups, false)
  RC_ALLOC(int64_t, dedup_key, probes, false)
  RC_ALLOC(int32_t, dedup_val, probes, false)
  RC_ALLOC(uint32_t, dedup_stamp, probes, true)
  RC_ALLOC(int32_t, store_probe_slot, probes, false)
  RC_ALLOC(int32_t, store_probe_idx, probes, false)
  RC_ALLOC(uint32_t, store_probe_stamp, probes, true)
#undef RC_ALLOC

  if (ok) {
    for (int64_t i = 0; i < nbuckets; i++) rc->buckets[i] = -1;
    for (int64_t i = 0; i < nghost_buckets; i++) rc->ghost_buckets[i] = -1;
    for (int64_t i = 0; i < rc->ghost_entries; i++) rc->ghost_free_stack[i] = (int32_t)i;
    rc->ghost_free_top = rc->ghost_entries;
    rc->ids_buf = (int64_t*)rc->ids.buf;
    rc->miss_ids_buf = (int64_t*)rc->miss_ids.buf;
    for (int p = 0; p < rc->nplanes; p++) {
      rc->miss_plane[p] = (uint8_t*)rc->miss[p].buf;
      rc->stage_plane[p] = (uint8_t*)rc->stage[p].buf;
      rc->pool_plane[p] = (uint8_t*)rc->pool[p].buf;
    }
  }
  if (!ok) {
    row_cache_release(rc);
    return NULL;
  }
  PyObject* capsule = PyCapsule_New(rc, RC_ROW_CACHE_CAPSULE, rc_delete);
  if (!capsule) row_cache_release(rc);
  return capsule;
}

static row_cache_t* rc_from(PyObject* capsule) {
  return PyCapsule_GetPointer(capsule, RC_ROW_CACHE_CAPSULE);
}

// _PyS3FifoReference.plan: resolve ids[:count]; latches the ghost flag at
// lookup time, counts every miss position, then admits each distinct miss
// after all lookups so hits raise their counters before any recycling. Hit
// copies run here (after all inserts, so this batch's eviction never clobbers
// a row the batch still reads -- the same ordering as _read_cached); miss
// staging and pool stores run in row_cache_store after the caller's fetch
// filled the miss planes.
static PyObject* py_row_cache_read(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  long long count;
  if (!PyArg_ParseTuple(args, "OL", &capsule, &count)) return NULL;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  if (count < 0 || count > rc->max_lookups) {
    PyErr_SetString(PyExc_ValueError, "row cache batch count exceeds batch capacity");
    return NULL;
  }
  rc->n_hit = rc->n_miss_ids = rc->n_miss_pos = rc->n_store = 0;
  rc->dedup_gen += 1;
  rc->store_gen += 1;
  const int64_t* ids = rc->ids_buf;
  for (int64_t position = 0; position < count; position++) {
    int64_t key = ids[position];
    int32_t slot = rc_get(rc, key);
    if (slot >= 0) {
      rc->hit_pos[rc->n_hit] = (int32_t)position;
      rc->hit_slot[rc->n_hit] = slot;
      rc->n_hit += 1;
      continue;
    }
    int32_t row = rc_dedup_find(rc, key);
    if (row == RC_NOT_FOUND) {
      row = rc->n_miss_ids;
      rc->miss_ids_buf[row] = key;
      // Latch the C's find() flag at lookup time (plan's hit_on_ghost dict).
      rc->ghost_latch[row] = rc_ghost_contains(rc, key) ? 1 : 0;
      rc_dedup_add(rc, key, row);
      rc->n_miss_ids += 1;
    }
    rc->miss_pos[rc->n_miss_pos] = (int32_t)position;
    rc->miss_row[rc->n_miss_pos] = row;
    rc->n_miss_pos += 1;
  }
  rc->hits += rc->n_hit;
  rc->misses += rc->n_miss_pos;
  for (int32_t row = 0; row < rc->n_miss_ids; row++) {
    int32_t slot = rc_insert(rc, rc->miss_ids_buf[row], rc->ghost_latch[row]);
    if (slot == -2) {
      PyErr_SetString(PyExc_KeyError, "row is already cached");
      return NULL;
    }
    if (slot < 0) continue;  // no slot could be taken: not cached this time
    int32_t existing = rc_store_find(rc, slot);
    if (existing != RC_NOT_FOUND) {
      rc->store_row[existing] = row;  // dict update keeps first-insert order
    } else {
      rc_store_add(rc, slot, rc->n_store);
      rc->store_slot[rc->n_store] = slot;
      rc->store_row[rc->n_store] = row;
      rc->n_store += 1;
    }
  }
  for (int32_t i = 0; i < rc->n_hit; i++) {
    rc_copy_row(rc, rc->stage_plane, (const uint8_t* const*)rc->pool_plane, rc->hit_pos[i],
                rc->hit_slot[i]);
  }
  return Py_BuildValue("ii", rc->n_hit, rc->n_miss_ids);
}

// The store half of the batch: misses stage from the fetched planes and the
// pool absorbs the rows plan() reserved store slots for.
static PyObject* py_row_cache_store(PyObject* self, PyObject* capsule) {
  (void)self;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  for (int32_t i = 0; i < rc->n_miss_pos; i++) {
    rc_copy_row(rc, rc->stage_plane, (const uint8_t* const*)rc->miss_plane, rc->miss_pos[i],
                rc->miss_row[i]);
  }
  for (int32_t i = 0; i < rc->n_store; i++) {
    rc_copy_row(rc, rc->pool_plane, (const uint8_t* const*)rc->miss_plane, rc->store_slot[i],
                rc->store_row[i]);
  }
  Py_RETURN_NONE;
}

static PyObject* py_row_cache_reserve(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  long long key;
  if (!PyArg_ParseTuple(args, "OL", &capsule, &key)) return NULL;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  int32_t slot = rc_reserve(rc, (int64_t)key);
  if (slot < 0) Py_RETURN_NONE;
  return PyLong_FromLong(slot);
}

static PyObject* py_row_cache_confirm(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  int slot;
  if (!PyArg_ParseTuple(args, "Oi", &capsule, &slot)) return NULL;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  if (slot < 0 || (int64_t)slot >= rc->capacity) {
    PyErr_SetString(PyExc_ValueError, "row cache slot index out of range");
    return NULL;
  }
  return PyLong_FromLong(rc_confirm(rc, slot));
}

static PyObject* py_row_cache_cancel(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  int slot;
  if (!PyArg_ParseTuple(args, "Oi", &capsule, &slot)) return NULL;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  if (slot < 0 || (int64_t)slot >= rc->capacity) {
    PyErr_SetString(PyExc_ValueError, "row cache slot index out of range");
    return NULL;
  }
  rc_cancel(rc, slot);
  Py_RETURN_NONE;
}

static PyObject* py_row_cache_get(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  long long key;
  if (!PyArg_ParseTuple(args, "OL", &capsule, &key)) return NULL;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  int32_t slot = rc_get(rc, (int64_t)key);
  if (slot < 0) Py_RETURN_NONE;
  return PyLong_FromLong(slot);
}

static PyObject* py_row_cache_is_ready(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  long long key;
  if (!PyArg_ParseTuple(args, "OL", &capsule, &key)) return NULL;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  int32_t slot = rc_tbl_find(rc, (int64_t)key);
  return PyBool_FromLong(slot >= 0 && rc->state[slot] == RC_READY);
}

static PyObject* py_row_cache_state(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  int slot;
  if (!PyArg_ParseTuple(args, "Oi", &capsule, &slot)) return NULL;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  if (slot < 0 || (int64_t)slot >= rc->capacity) {
    PyErr_SetString(PyExc_ValueError, "row cache slot index out of range");
    return NULL;
  }
  return PyLong_FromLong(rc->state[slot]);
}

static PyObject* int_list(const int32_t* values, int64_t count) {
  PyObject* list = PyList_New(count);
  if (!list) return NULL;
  for (int64_t i = 0; i < count; i++) {
    PyObject* item = PyLong_FromLong(values[i]);
    if (!item) {
      Py_DECREF(list);
      return NULL;
    }
    PyList_SET_ITEM(list, i, item);
  }
  return list;
}

static PyObject* key_list(const int64_t* values, int64_t count) {
  PyObject* list = PyList_New(count);
  if (!list) return NULL;
  for (int64_t i = 0; i < count; i++) {
    PyObject* item = PyLong_FromLongLong(values[i]);
    if (!item) {
      Py_DECREF(list);
      return NULL;
    }
    PyList_SET_ITEM(list, i, item);
  }
  return list;
}

// The RowPlan the last row_cache_read produced, as lists: the parity tests
// compare these against _PyS3FifoReference.plan's fields. Diagnostic only:
// the hot path never calls it.
static PyObject* py_row_cache_plan_arrays(PyObject* self, PyObject* capsule) {
  (void)self;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  PyObject* result = PyDict_New();
  if (!result) return NULL;
  PyObject* values[7];
  values[0] = int_list(rc->hit_pos, rc->n_hit);
  values[1] = int_list(rc->hit_slot, rc->n_hit);
  values[2] = key_list(rc->miss_ids_buf, rc->n_miss_ids);
  values[3] = int_list(rc->miss_pos, rc->n_miss_pos);
  values[4] = int_list(rc->miss_row, rc->n_miss_pos);
  values[5] = int_list(rc->store_slot, rc->n_store);
  values[6] = int_list(rc->store_row, rc->n_store);
  // (miss_ids is values[2]; no duplicate.)
  static const char* names[7] = {"hit_pos", "hit_slot", "miss_ids", "miss_pos", "miss_row",
                                 "store_slot", "store_row"};
  for (int i = 0; i < 7; i++) {
    if (!values[i] || PyDict_SetItemString(result, names[i], values[i]) != 0) {
      Py_XDECREF(values[i]);
      Py_DECREF(result);
      return NULL;
    }
    Py_DECREF(values[i]);
  }
  return result;
}

// _PyS3FifoReference queue/ghost state for per-request parity: (key, freq)
// pairs in queue order, ghost keys in demotion order, resident/loading keys.
static PyObject* py_row_cache_debug_state(PyObject* self, PyObject* capsule) {
  (void)self;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  PyObject* result = PyDict_New();
  if (!result) return NULL;
  PyObject* queues[2];
  const int32_t* rings[2] = {rc->small_ring, rc->main_ring};
  const int32_t heads[2] = {(int32_t)rc->small_head, (int32_t)rc->main_head};
  const int64_t counts[2] = {rc->small_count, rc->main_count};
  for (int q = 0; q < 2; q++) {
    PyObject* list = PyList_New(counts[q]);
    if (!list) {
      Py_DECREF(result);
      return NULL;
    }
    for (int64_t i = 0; i < counts[q]; i++) {
      int32_t slot = rings[q][(heads[q] + i) % rc->capacity];
      PyObject* pair = Py_BuildValue("Li", rc->key[slot], (int)rc->freq[slot]);
      if (!pair) {
        Py_DECREF(list);
        Py_DECREF(result);
        return NULL;
      }
      PyList_SET_ITEM(list, i, pair);
    }
    queues[q] = list;
  }
  if (PyDict_SetItemString(result, "small", queues[0]) != 0 ||
      PyDict_SetItemString(result, "main", queues[1]) != 0) {
    Py_DECREF(queues[0]);
    Py_DECREF(queues[1]);
    Py_DECREF(result);
    return NULL;
  }
  Py_DECREF(queues[0]);
  Py_DECREF(queues[1]);
  PyObject* ghost = PyList_New(rc->ghost_count);
  if (!ghost) {
    Py_DECREF(result);
    return NULL;
  }
  int64_t i = 0;
  for (int32_t e = rc->ghost_head; e >= 0; e = rc->ghost_next[e]) {
    PyObject* item = PyLong_FromLongLong(rc->ghost_key[e]);
    if (!item) {
      Py_DECREF(ghost);
      Py_DECREF(result);
      return NULL;
    }
    PyList_SET_ITEM(ghost, i++, item);
  }
  if (PyDict_SetItemString(result, "ghost", ghost) != 0) {
    Py_DECREF(ghost);
    Py_DECREF(result);
    return NULL;
  }
  Py_DECREF(ghost);
  PyObject* ready = PyList_New(rc->ready_count);
  PyObject* loading = PyDict_New();  // key -> slot, mirroring _loading
  if (!ready || !loading) {
    Py_XDECREF(ready);
    Py_XDECREF(loading);
    Py_DECREF(result);
    return NULL;
  }
  int64_t n_ready = 0;
  for (int64_t slot = 0; slot < rc->used; slot++) {
    if (rc->state[slot] == RC_READY) {
      PyObject* item = PyLong_FromLongLong(rc->key[slot]);
      if (!item) {
        Py_DECREF(ready);
        Py_DECREF(loading);
        Py_DECREF(result);
        return NULL;
      }
      PyList_SET_ITEM(ready, n_ready++, item);
    } else if (rc->state[slot] == RC_LOADING) {
      PyObject* item = PyLong_FromLongLong(rc->key[slot]);
      PyObject* value = PyLong_FromLong((long)slot);
      int added = !item || !value || PyDict_SetItem(loading, item, value) != 0;
      Py_XDECREF(item);
      Py_XDECREF(value);
      if (added) {
        Py_DECREF(ready);
        Py_DECREF(loading);
        Py_DECREF(result);
        return NULL;
      }
    }
  }
  if (PyDict_SetItemString(result, "ready", ready) != 0 ||
      PyDict_SetItemString(result, "loading", loading) != 0) {
    Py_DECREF(ready);
    Py_DECREF(loading);
    Py_DECREF(result);
    return NULL;
  }
  Py_DECREF(ready);
  Py_DECREF(loading);
  // Slot-allocation state: the parity tests compare the free list (order
  // matters: Python _free.pop() takes the most recently released slot), the
  // high-water mark, and the eviction gate against the oracle's privates.
  PyObject* free = PyList_New(rc->free_top);
  if (!free) {
    Py_DECREF(result);
    return NULL;
  }
  for (int64_t i = 0; i < rc->free_top; i++) {
    PyObject* item = PyLong_FromLong((long)rc->free_stack[i]);
    if (!item) {
      Py_DECREF(free);
      Py_DECREF(result);
      return NULL;
    }
    PyList_SET_ITEM(free, i, item);
  }
  if (PyDict_SetItemString(result, "free", free) != 0) {
    Py_DECREF(free);
    Py_DECREF(result);
    return NULL;
  }
  Py_DECREF(free);
  PyObject* scalars = Py_BuildValue("{s:L,s:L}", "used", rc->used, "has_evicted",
                                    (long long)rc->has_evicted);
  if (!scalars || PyDict_Update(result, scalars) != 0) {
    Py_XDECREF(scalars);
    Py_DECREF(result);
    return NULL;
  }
  Py_DECREF(scalars);
  return result;
}

// Diagnostic: raw arrays so the parity harness can diff C internals against
// the oracle's privates when the derived debug_state disagrees (never used by
// the hot path or DiskRowCache).
static PyObject* py_row_cache_raw_dump(PyObject* self, PyObject* capsule) {
  (void)self;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  PyObject* result = PyDict_New();
  if (!result) return NULL;
  PyObject* items[10];
  items[0] = Py_NewRef(result);  // placeholder, replaced below
  int64_t n = rc->used < rc->capacity ? rc->used : rc->capacity;
  items[1] = PyList_New(n);      // state
  items[2] = PyList_New(n);      // queue
  items[3] = PyList_New(n);      // freq
  items[4] = PyList_New(n);      // key
  items[5] = PyList_New(rc->capacity);  // small ring raw
  items[6] = PyList_New(rc->capacity);  // main ring raw
  items[7] = PyList_New(rc->free_top);  // free stack raw
  items[8] = PyList_New(rc->ghost_cap);  // ghost key raw (order via next links)
  items[9] = PyList_New(rc->ghost_cap);  // ghost next raw
  for (int i = 1; i < 10; i++) {
    if (!items[i]) {
      for (int k = 1; k < 10; k++) Py_XDECREF(items[k]);
      Py_DECREF(result);
      return NULL;
    }
  }
  for (int64_t i = 0; i < n; i++) {
    PyList_SET_ITEM(items[1], i, PyLong_FromLong(rc->state[i]));
    PyList_SET_ITEM(items[2], i, PyLong_FromLong(rc->queue[i]));
    PyList_SET_ITEM(items[3], i, PyLong_FromLong(rc->freq[i]));
    PyList_SET_ITEM(items[4], i, PyLong_FromLongLong(rc->key[i]));
  }
  for (int64_t i = 0; i < rc->capacity; i++) {
    PyList_SET_ITEM(items[5], i, PyLong_FromLong(rc->small_ring[i]));
    PyList_SET_ITEM(items[6], i, PyLong_FromLong(rc->main_ring[i]));
  }
  for (int64_t i = 0; i < rc->free_top; i++)
    PyList_SET_ITEM(items[7], i, PyLong_FromLong(rc->free_stack[i]));
  for (int64_t i = 0; i < rc->ghost_cap; i++) {
    PyList_SET_ITEM(items[8], i, PyLong_FromLongLong(rc->ghost_key[i]));
    PyList_SET_ITEM(items[9], i, PyLong_FromLong(rc->ghost_next[i]));
  }
  int rc_set = 0;
  rc_set |= PyDict_SetItemString(result, "state", items[1]);
  rc_set |= PyDict_SetItemString(result, "queue", items[2]);
  rc_set |= PyDict_SetItemString(result, "freq", items[3]);
  rc_set |= PyDict_SetItemString(result, "keys", items[4]);
  rc_set |= PyDict_SetItemString(result, "small_ring", items[5]);
  rc_set |= PyDict_SetItemString(result, "main_ring", items[6]);
  rc_set |= PyDict_SetItemString(result, "free_raw", items[7]);
  rc_set |= PyDict_SetItemString(result, "ghost_keys", items[8]);
  rc_set |= PyDict_SetItemString(result, "ghost_next", items[9]);
  PyObject* hs = PyLong_FromLongLong(rc->small_head);
  PyObject* hm = PyLong_FromLongLong(rc->main_head);
  PyObject* hg = PyLong_FromLong(rc->ghost_head);
  PyObject* tg = PyLong_FromLong(rc->ghost_tail);
  if (!hs || !hm || !hg || !tg) {
    rc_set = -1;
  } else {
    rc_set |= PyDict_SetItemString(result, "head_small", hs);
    rc_set |= PyDict_SetItemString(result, "head_main", hm);
    rc_set |= PyDict_SetItemString(result, "head_ghost", hg);
    rc_set |= PyDict_SetItemString(result, "tail_ghost", tg);
  }
  Py_XDECREF(hs);
  Py_XDECREF(hm);
  Py_XDECREF(hg);
  Py_XDECREF(tg);
  for (int i = 1; i < 10; i++) Py_DECREF(items[i]);
  if (rc_set != 0) {
    Py_DECREF(result);
    return NULL;
  }
  return result;
}

// Hit/miss counters and queue sizes for DiskRowCache.stats; the s/m/g fields
// keep the probe surface the Python class exposed through privates.
static PyObject* py_row_cache_stats(PyObject* self, PyObject* capsule) {
  (void)self;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  return Py_BuildValue("{s:L,s:L,s:L,s:l,s:l,s:l,s:L,s:L,s:L,s:L}",  //
                       "hits", rc->hits, "misses", rc->misses, "resident",
                       rc->ready_count + rc->loading_count, "s", rc->small_count, "m",
                       rc->main_count, "g", rc->ghost_count, "capacity", rc->capacity,
                       "small_cap", rc->small_cap, "main_cap", rc->main_cap, "ghost_cap",
                       rc->ghost_cap);
}

static PyMethodDef methods[] = {
    {"row_cache_create", py_row_cache_create, METH_VARARGS, NULL},
    {"row_cache_read", py_row_cache_read, METH_VARARGS, NULL},
    {"row_cache_store", py_row_cache_store, METH_O, NULL},
    {"row_cache_reserve", py_row_cache_reserve, METH_VARARGS, NULL},
    {"row_cache_confirm", py_row_cache_confirm, METH_VARARGS, NULL},
    {"row_cache_cancel", py_row_cache_cancel, METH_VARARGS, NULL},
    {"row_cache_get", py_row_cache_get, METH_VARARGS, NULL},
    {"row_cache_is_ready", py_row_cache_is_ready, METH_VARARGS, NULL},
    {"row_cache_state", py_row_cache_state, METH_VARARGS, NULL},
    {"row_cache_plan_arrays", py_row_cache_plan_arrays, METH_O, NULL},
    {"row_cache_debug_state", py_row_cache_debug_state, METH_O, NULL},
    {"row_cache_stats", py_row_cache_stats, METH_O, NULL},
    {"row_cache_raw_dump", py_row_cache_raw_dump, METH_O, NULL},
    {NULL, NULL, 0, NULL},
};
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_b12x_row_cache",
    .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit__b12x_row_cache(void) {
  PyObject* result = PyModule_Create(&module);
  if (result && PyModule_AddIntConstant(result, "ABI_VERSION", 1) < 0) {
    Py_DECREF(result);
    return NULL;
  }
  return result;
}
