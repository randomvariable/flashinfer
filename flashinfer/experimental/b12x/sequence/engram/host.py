"""Vectorized host row ids for advisory prefetch; must equal reference.py."""

from __future__ import annotations

import torch

from .geometry import Geometry

ORDERS = 3
HEADS_PER_ORDER = 8


def query_row_ids(
    compressed: torch.Tensor,
    *,
    geometry: Geometry,
    layer_id: int,
    pad: int,
) -> torch.Tensor:
    """Return ``(R * Q * 24,)`` global row ids for the query positions.

    ``compressed`` is ``(ORDERS + Q)`` columns wide per request: the committed
    compressed history followed by the compressed queries, with ``-1`` marking
    a dead token. That is the same per-request ``cache`` that
    ``reference.hash_reference`` builds, so this must keep matching it.
    """
    if compressed.ndim != 2 or compressed.dtype != torch.int64:
        raise TypeError("compressed must be a rank-2 torch.int64 tensor")
    requests, length = compressed.shape
    if length <= ORDERS:
        raise ValueError(
            f"compressed must carry {ORDERS} history plus at least one query column"
        )
    if layer_id not in geometry.layer_ids:
        raise ValueError(f"geometry has no layer id {layer_id}")
    layer = geometry.layer_ids.index(layer_id)
    device = compressed.device
    multipliers = torch.tensor(
        geometry.multipliers[layer], dtype=torch.int64, device=device
    )
    primes = torch.tensor(geometry.primes[layer], dtype=torch.int64, device=device)
    offsets = torch.tensor(geometry.offsets[layer], dtype=torch.int64, device=device)
    current = torch.arange(ORDERS, length, dtype=torch.int64, device=device)
    rolling = torch.zeros(requests, current.numel(), dtype=torch.int64, device=device)
    blocked = torch.zeros(requests, current.numel(), dtype=torch.bool, device=device)
    blocks = []
    for lag in range(ORDERS + 1):
        source = compressed[:, current - lag]
        blocked = blocked | (source == -1)
        # Once any token in the window is dead, every term reads as the pad.
        rolling = torch.bitwise_xor(
            rolling, torch.where(blocked, int(pad), source) * multipliers[lag]
        )
        if lag:
            first = (lag - 1) * HEADS_PER_ORDER
            blocks.append(
                rolling[..., None] % primes[first : first + HEADS_PER_ORDER]
                + offsets[first : first + HEADS_PER_ORDER]
            )
    return torch.cat(blocks, dim=-1).reshape(-1)


__all__ = ["query_row_ids"]
