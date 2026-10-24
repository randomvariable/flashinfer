"""Vectorized host row ids for advisory prefetch; must equal reference.py."""

from __future__ import annotations

import torch

from .reference import ple_hash_ids_reference


def eos_bounded_query_windows(
    complete: torch.Tensor,
    *,
    eos_token_id: int,
    max_order: int,
) -> dict[int, torch.Tensor]:
    """Build the query-position n-gram windows of ``complete``.

    ``complete`` is ``(R, H + Q)`` int64, where ``H = max_order - 1`` committed
    history columns are followed by ``Q`` query columns. Tokens run oldest to
    current, so each row of the ``(R * Q, order)`` result is one window and rows
    are request-major, then query position. Predecessors behind the nearest
    preceding EOS, or behind the start of the request, become ``eos_token_id``.
    """
    if complete.ndim != 2 or complete.dtype != torch.int64:
        raise TypeError("complete must be a rank-2 torch.int64 tensor")
    max_order = int(max_order)
    if max_order < 2:
        raise ValueError(f"max_order must be at least 2, got {max_order}")
    requests, length = complete.shape
    history = max_order - 1
    if length < history:
        raise ValueError(
            f"complete must carry at least {history} history columns, got {length}"
        )
    device = complete.device
    position = torch.arange(length, dtype=torch.int64, device=device)
    current = torch.arange(history, length, dtype=torch.int64, device=device)
    last_eos = torch.cummax(
        torch.where(complete == int(eos_token_id), position, -1), dim=1
    ).values
    seg_start = torch.cat(
        (
            torch.zeros(requests, 1, dtype=torch.int64, device=device),
            last_eos[:, :-1] + 1,
        ),
        dim=1,
    )
    start = seg_start[:, current]
    windows: dict[int, torch.Tensor] = {}
    for order in range(2, max_order + 1):
        source = (
            current[:, None]
            + torch.arange(1 - order, 1, dtype=torch.int64, device=device)[None, :]
        )
        valid = (source >= 0)[None] & (source[None] >= start[:, :, None])
        windows[order] = torch.where(
            valid, complete[:, source.clamp_min(0)], int(eos_token_id)
        ).reshape(requests * current.numel(), order)
    return windows


def query_row_ids(
    complete: torch.Tensor,
    *,
    eos_token_id: int,
    max_order: int,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    heads_per_order: int,
) -> torch.Tensor:
    """Return ``(R * Q * (max_order - 1) * heads_per_order,)`` global row ids.

    Identical to ``reference.ple_hash_packed_reference`` over the same
    ``(R, H + Q)`` inputs, which is the oracle this must keep matching.
    """
    windows = eos_bounded_query_windows(
        complete, eos_token_id=eos_token_id, max_order=max_order
    )
    return ple_hash_ids_reference(
        windows,
        multipliers=multipliers,
        prime_sizes=prime_sizes,
        table_offsets=table_offsets,
        heads_per_order=heads_per_order,
    ).reshape(-1)


__all__ = ["eos_bounded_query_windows", "query_row_ids"]
