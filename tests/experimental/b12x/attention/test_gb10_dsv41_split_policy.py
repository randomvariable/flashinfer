"""Measured DSV41 split defaults preserve precision, producers, and overrides."""

from types import SimpleNamespace

import pytest
import torch

from b12x.attention._shared.mla.kernel import (
    _dsv41_spark_default_num_splits,
    prepare_unified_decode_launch,
)
from b12x.attention._shared.mla.traits import ModelType


def _policy(**changes):
    arguments = dict(
        rows=13,
        heads=64,
        topk=128,
        extra_topk=512,
        swa_page_size=128,
        indexed_page_size=128,
        compute_mode="bf16",
        sm_count=48,
        compute_capability=(12, 1),
        workspace_splits=54,
    )
    arguments.update(changes)
    return _dsv41_spark_default_num_splits(**arguments)


@pytest.mark.parametrize("rows", [7, 13, 14])
def test_gb10_dsv41_measured_rows_use_three_splits(rows):
    assert _policy(rows=rows) == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"compute_capability": (12, 0)},
        {"compute_capability": None},
        {"sm_count": 188},
        {"sm_count": 64},
        {"heads": 32},
        {"heads": 16},
        {"rows": 1},
        {"rows": 4},
        {"rows": 8},
        {"rows": 15},
        {"topk": 64},
        {"extra_topk": 0},
        {"extra_topk": 1024},
        {"swa_page_size": 64},
        {"indexed_page_size": 64},
        {"indexed_page_size": None},
        {"compute_mode": "fp8"},
    ],
)
def test_gb10_dsv41_policy_does_not_leak_to_other_contracts(changes):
    assert _policy(**changes) == 54


@pytest.mark.parametrize("splits", range(1, 10))
def test_gb10_dsv41_retains_smaller_prepared_workspace_choices(splits):
    assert _policy(workspace_splits=splits) == splits


@pytest.mark.parametrize("rows", [7, 13, 14])
@pytest.mark.parametrize("context", [1, 128, 512, 131072])
def test_gb10_dsv41_policy_matches_reference_and_graph_replay(rows, context):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor, properties.multi_processor_count) != (
        12,
        1,
        48,
    ):
        pytest.skip("This policy is specific to the 48-SM GB10")

    from benchmarks.experimental.b12x.benchmark_deepseek_v41_mla import (
        Case,
        check_reference,
        make_inputs,
        prepare_case,
    )
    from b12x.preparation import PreparationSession

    case = Case(
        name=f"gb10-splits-{rows}-{context}",
        mode="decode",
        form="c1",
        batch=rows,
        query_width=1,
        rows=rows,
        swa_width=128,
        swa_page_size=128,
        indexed_width=512,
        indexed_page_size=128,
        page_table_width=2048,
        num_q_heads=64,
        context_length=context,
        context_capacity=262144,
        compute_mode="bf16",
        cache_pages=2048,
    )
    inputs = make_inputs(case, device)
    arguments = dict(
        q_all=inputs.q,
        swa_k_cache=inputs.swa_cache,
        swa_indices=inputs.swa_indices,
        workspace=SimpleNamespace(max_chunks_per_row=54, num_chunks_value=54),
        swa_page_size=128,
        sm_count=48,
        indexed_k_cache=inputs.indexed_cache,
        indexed_indices=inputs.indexed_indices,
        indexed_page_size=128,
        attn_sink=inputs.sink,
        model_type_override=ModelType.DSV41,
        v41_compute_mode="bf16",
    )
    launch = prepare_unified_decode_launch(**arguments)
    assert launch.num_splits == 3
    assert not launch.native_dsv4_h8 and not launch.native_dsv4_h16
    assert not launch.native_dsv41_fp8
    for forced in (1, 10):
        explicit = prepare_unified_decode_launch(
            **arguments, forced_num_splits=forced
        )
        assert explicit.num_splits == forced

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        prepared = prepare_case(case, inputs, session)
        check_reference(prepared, inputs, case)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prepared.run()
        inputs.q.mul_(0.75)
        inputs.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check_reference(prepared, inputs, case, run=False)
