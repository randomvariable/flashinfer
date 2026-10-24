"""Host row-id helpers must equal the ``reference.py`` oracles, CPU only."""

from __future__ import annotations

import pytest
import torch

from b12x.sequence.engram import host as engram_host
from b12x.sequence.engram.geometry import build_geometry
from b12x.sequence.engram.reference import hash_reference
from b12x.sequence.ple_hash import host as ple_host
from b12x.sequence.ple_hash import reference as ple_reference

VOCAB_SIZE = 1000
EOS = 7
HEADS_PER_ORDER = 8


def _ple_case(requests, queries, max_order, geometry):
    """Deterministic history/query tokens with EOS in both halves."""
    multipliers, prime_sizes, table_offsets = geometry
    history_len = max_order - 1
    generator = torch.Generator().manual_seed(0x5EED)
    query = torch.randint(
        0, VOCAB_SIZE, (requests * queries,), generator=generator
    ).reshape(requests, queries)
    history = torch.randint(0, VOCAB_SIZE, (requests, history_len), generator=generator)
    history[0, :] = EOS
    if requests > 1 and history_len > 1:
        history[1, history_len // 2] = EOS
    query[0, 1] = EOS
    if requests > 2 and queries > 2:
        query[2, queries - 2] = EOS
    complete = torch.cat((history, query), dim=1)
    starts = torch.arange(0, requests * queries + 1, queries, dtype=torch.int32)
    expected = ple_reference.ple_hash_packed_reference(
        query.reshape(-1).contiguous(),
        starts,
        history,
        eos_token_id=EOS,
        multipliers=multipliers,
        prime_sizes=prime_sizes,
        table_offsets=table_offsets,
        heads_per_order=HEADS_PER_ORDER,
    )
    actual = ple_host.query_row_ids(
        complete,
        eos_token_id=EOS,
        max_order=max_order,
        multipliers=multipliers,
        prime_sizes=prime_sizes,
        table_offsets=table_offsets,
        heads_per_order=HEADS_PER_ORDER,
    )
    assert actual.shape == expected.reshape(-1).shape
    assert torch.equal(actual, expected.reshape(-1))


@pytest.mark.parametrize("max_order", [2, 3, 4])
def test_ple_query_row_ids_match_packed_reference(max_order):
    """Vectorized host windows hash exactly like the per-request oracle."""
    geometry = (
        ple_reference.ple_multipliers(
            vocab_size=VOCAB_SIZE,
            max_order=max_order,
            dense_layer_ordinal=1,
        ),
        *ple_reference.ple_table_geometry(
            base_size=97,
            dense_layer_ordinal=1,
            total_heads=(max_order - 1) * HEADS_PER_ORDER,
        ),
    )
    _ple_case(requests=5, queries=4, max_order=max_order, geometry=geometry)
    _ple_case(requests=1, queries=37, max_order=max_order, geometry=geometry)


def _build_engram_geometry():
    for base_table_size in (1000, 4096, 65536):
        try:
            return build_geometry(
                layer_ids=(1, 14),
                base_table_size=base_table_size,
                compressed_vocab_size=50,
            )
        except ValueError:
            continue
    raise AssertionError("no candidate base_table_size builds Engram geometry")


def test_engram_query_row_ids_match_reference():
    """Dead-token propagation and per-order heads match the CPU oracle."""
    geometry = _build_engram_geometry()
    compressed_vocab = geometry.compressed_vocab_size
    generator = torch.Generator().manual_seed(0xC0FFEE)
    token_map = torch.randint(0, compressed_vocab, (200,), generator=generator).tolist()
    requests, queries = 4, 5
    tokens = torch.randint(
        0, len(token_map), (requests * queries,), generator=generator
    )
    mask = [True] * (requests * queries)
    for dead in (2, 9, 17):
        mask[dead] = False
    query = torch.tensor(
        [token_map[t] if keep else -1 for t, keep in zip(tokens.tolist(), mask)],
        dtype=torch.int64,
    ).reshape(requests, queries)
    history = torch.randint(-1, compressed_vocab, (requests, 3), generator=generator)
    history[0, 0] = -1
    history[2, 2] = -1
    starts = list(range(0, requests * queries + 1, queries))
    slots = range(requests)
    for layer_id in geometry.layer_ids:
        expected = hash_reference(
            tokens.tolist(),
            mask,
            starts,
            slots,
            [list(row) for row in history.tolist()],
            token_map,
            geometry,
            layer_id,
        )
        actual = engram_host.query_row_ids(
            torch.cat((history, query), dim=1),
            geometry=geometry,
            layer_id=layer_id,
            pad=token_map[2],
        )
        assert actual.shape == expected.reshape(-1).shape
        assert torch.equal(actual, expected.reshape(-1))
