"""GB10 CSF defaults keep the native dense ladder and RTX behavior."""

from dataclasses import replace

import pytest

from b12x.moe.fused_moe._impl import _heuristic_moe_decode_config
from b12x.moe.fused_moe._tuning import (
    TUNING,
    _csf_recipe_defaults_enabled,
)
from b12x.preparation import DeviceIdentity, FrozenMapping
from tests.experimental.b12x.preparation.test_precision_choices import _nvfp4_query

GB10 = DeviceIdentity("nvidia", (12, 1), 48, "GB10")
RTX = DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000")


def _query(tokens=768, **changes):
    arguments = dict(
        num_experts=384,
        route_num_experts=384,
        hidden_size=5120,
        intermediate_size=2304,
        top_k=6,
        num_tokens=tokens,
        routed_rows=tokens * 6,
        csf_inline_scales=True,
        swiglu_limit=10.0,
    )
    arguments.update(changes)
    return replace(_nvfp4_query(), **arguments)


@pytest.mark.parametrize(
    "tokens,backend,tile,inline,materialized",
    [
        (1, "micro", None, False, False),
        (2, "micro", None, False, False),
        (3, "micro", None, False, False),
        (4, "dynamic", 16, False, False),
        (8, "dynamic", 16, False, False),
        (9, "dynamic", 16, False, False),
        (15, "dynamic", 16, False, False),
        (16, "dynamic", 16, True, False),
        (128, "dynamic", 16, True, False),
        (767, "dynamic", 16, True, False),
        (768, "dynamic", 32, True, False),
        (769, "dynamic", 32, True, False),
        (959, "dynamic", 32, True, False),
        (960, "dynamic", 32, True, False),
        (961, "dynamic", 32, True, False),
        (1023, "dynamic", 32, True, False),
        (1024, "dynamic", 32, True, False),
        (1025, "dynamic", 32, False, False),
        (2304, "dynamic", 32, False, False),
        (3071, "dynamic", 32, False, False),
        (3072, "dynamic", 64, False, False),
        (3073, "dynamic", 64, False, False),
        (6143, "dynamic", 64, False, False),
        (6144, "dynamic", 128, False, True),
        (6145, "dynamic", 128, False, True),
        (8192, "dynamic", 128, False, True),
    ],
)
def test_gb10_csf_defaults_keep_the_native_dense_ladder(
    tokens, backend, tile, inline, materialized
):
    config = TUNING.configure(_query(tokens), device=GB10, search=False).default
    assert (
        config.backend,
        config.dynamic_tile_m,
        config.csf_inline_scales,
        config.nvfp4_materialize_intermediate,
    ) == (backend, tile, inline, materialized)


@pytest.mark.parametrize("tokens", [3072, 3073, 6143, 6144, 6145, 8192])
def test_rtx_retains_early_materialization(tokens):
    config = TUNING.configure(_query(tokens), device=RTX, search=False).default
    assert config.dynamic_tile_m == 128
    assert config.nvfp4_materialize_intermediate
    assert not config.csf_inline_scales


@pytest.mark.parametrize(
    "changes",
    [
        {"num_experts": 256},
        {"hidden_size": 4096},
        {"intermediate_size": 1024},
        {"top_k": 8},
        {"csf_inline_scales": False},
        {
            "quant_mode": "w4a8_mx",
            "quant_modes": ("w4a8_mx",),
            "source_format": "fp4_e8m0_k32",
        },
        {"quant_mode": "w4a16", "quant_modes": ("w4a16",)},
        # TP4 partitions moe_intermediate_size 2304 into 576 per rank.
        {"intermediate_size": 576},
    ],
)
def test_gb10_csf_does_not_change_unmeasured_contracts(changes):
    query = _query(**changes)
    assert not _csf_recipe_defaults_enabled(query, GB10)
    expected = _heuristic_moe_decode_config(query, GB10)
    config = TUNING.configure(query, device=GB10, search=False).default
    assert (
        config.backend,
        config.dynamic_tile_m,
        config.csf_inline_scales,
    ) == (expected.backend, expected.dynamic_tile_m, False)


@pytest.mark.parametrize(
    "changes",
    [
        {"io_dtype": "float16"},
        {"shared_input_scales": False},
        {"deterministic_output": True},
        {"apply_router_weight_on_input": True},
        {"collect_activation_amax": True},
        {"activation": "relu2"},
        {"source_format": "fp4_e8m0_k32"},
    ],
)
def test_gb10_scope_requires_the_measured_execution_contract(changes):
    assert not _csf_recipe_defaults_enabled(_query(**changes), GB10)


def test_gb10_scope_requires_48_sms():
    device = DeviceIdentity("nvidia", (12, 1), 188, "GB10-like")
    assert not _csf_recipe_defaults_enabled(_query(768), device)


def test_sm120_scope_is_unchanged():
    assert _csf_recipe_defaults_enabled(_query(768), RTX)
    assert not _csf_recipe_defaults_enabled(
        _query(768, csf_inline_scales=False), RTX
    )


@pytest.mark.parametrize("device", [GB10, RTX])
def test_explicit_tile_control_blocks_the_density_promotion(device):
    config = TUNING.configure(
        _query(768, controls=FrozenMapping({"dynamic_tile_mn": (16, 128)})),
        device=device,
        search=False,
    ).default
    assert config.dynamic_tile_m == 16
    assert not config.nvfp4_materialize_intermediate
