"""PLE hashing/decoding over shared batch-bounded io_uring row storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .._shared.disk_table import DiskRowCache

if TYPE_CHECKING:
    from ._contracts import Binding, TableLayout


class DiskTable:
    """Own PLE row sources and decode a compact batch cache outside graphs.

    Rows come from checkpoint shards (``add_shard``) or, with ``host_rows``,
    from this TP shard's rows already resident in host RAM. ``cache_rows``
    and ``cache_tier`` configure the host-tier C S3-FIFO row cache; see
    ``DiskRowCache``.
    """

    def __init__(
        self,
        layout: TableLayout,
        shard_rows: int,
        *,
        queue_depth: int = 64,
        cache_rows: int | None = None,
        cache_tier: str | None = None,
        host_rows: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    ) -> None:
        """Build the shared disk row cache for one bound PLE table layout.

        Row widths come from ``layout`` (nvfp4 group-16 adds a scale plane),
        and the staging planes are exposed reinterpreted with the table's
        weight and scale dtypes. ``host_rows`` sources this TP shard's rows
        from host RAM instead of checkpoint shards.
        """
        from ._contracts import TableLayout

        if not isinstance(layout, TableLayout):
            raise TypeError("layout must be TableLayout")
        if layout.caps.table_memory != "io_uring":
            raise ValueError("DiskTable requires table_memory='io_uring'")
        self.layout = layout
        self._cache = DiskRowCache(
            device=layout.caps.device,
            max_lookups=layout.caps.max_tokens * layout.head_count,
            table_rows=layout.padded_vocab_size,
            shard_start=layout.shard_start,
            shard_end=layout.shard_end,
            shard_rows=shard_rows,
            weight_row_bytes=layout.weight_shape[1] * layout.weight_dtype.itemsize,
            scale_row_bytes=(
                layout.head_dim // 16
                if layout.caps.quant_mode == "nvfp4_group16"
                else 0
            ),
            queue_depth=queue_depth,
            cache_rows=cache_rows,
            cache_tier=cache_tier,
            host_rows=host_rows,
        )
        self.weight = self._cache.weight.view(layout.weight_dtype)
        self.weight_host = (
            self._cache.weight_host.view(layout.weight_dtype)
            if self._cache.weight_host is not None
            else None
        )
        self.weight_scale = (
            self._cache.scale.view(torch.float8_e4m3fn)
            if self._cache.scale is not None
            else None
        )
        self.weight_scale_host = (
            self._cache.scale_host.view(torch.float8_e4m3fn)
            if self._cache.scale_host is not None
            else None
        )

    def add_shard(
        self, shard_index: int, path: str, offset: int, *, scale: bool = False
    ) -> None:
        """Register one checkpoint shard file with the underlying row cache."""
        self._cache.add_shard(shard_index, path, offset, scale=scale)

    def _require_complete(self) -> None:
        """Raise unless every shard the bound row range needs is registered."""
        self._cache.require_complete()

    def _freeze(self) -> None:
        """Freeze shard registration after completeness is verified."""
        self._cache.freeze()

    def stats(self) -> dict[str, int | float]:
        """Byte budgets and row-cache counters, delegated to ``DiskRowCache``."""
        return self._cache.stats()

    def _run(self, binding: Binding, *, state, token_count: int) -> None:
        """Hash tokens to row ids, read those rows, and decode in one transaction.

        Runs the table hash, stages ``token_count * head_count`` row ids
        through ``read_rows``, then the lookup kernel over the staged planes.
        The whole sequence sits inside one cache ``transaction`` so staging
        stays pinned until decoding completes.
        """
        layout = self.layout
        with self._cache.transaction():
            state.hash_state.run(binding._hash_binding, token_count=token_count)
            self._cache.read_rows(binding._ids, token_count * layout.head_count)
            if token_count:
                weight_scale = (
                    self.weight_scale
                    if self._cache.scale_row_bytes
                    else binding.weight_scale
                )
                state.run_lookup(
                    self.weight,
                    weight_scale,
                    binding.weight_scale_2,
                    binding._ids,
                    binding.num_tokens,
                    binding.out,
                    token_count=token_count,
                )
