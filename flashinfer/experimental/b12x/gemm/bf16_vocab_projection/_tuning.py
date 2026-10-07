"""Configuration contract for BF16 vocabulary projection preparation."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.preparation import (
    DeviceIdentity,
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)

MAX_IN_FEATURES = 8_192
MIN_NATIVE_OUT_FEATURES = 16_384
# Rows one bf16_gemv SIMT tile serves from a single weight read (its SMALL_M_MAX).
# The Triton row kernels stream the weight once per row, so beyond one row the
# CuTe kernel is the only native choice that keeps the projection bandwidth-bound.
MAX_CUTE_TOKENS = 8
_TRITON_WARPS = frozenset((1, 2, 4, 8))
_LOOP_BLOCKS = frozenset((256, 512, 1_024))
_ALGORITHMS = {"torch": ("torch",), "triton": ("row", "loop"), "cute": ("simt",)}
# Largest row capacity whose default is the CuTe SIMT GEMV, from graph-replay
# timings of a 248320x2560 head. GB10 (12, 1): SIMT beats cuBLAS at 2..8 rows.
# RTX 5090 (12, 0): SIMT wins at 2 rows, ties at 4 and loses at 8, so wider
# capacities keep cuBLAS unless autotuning measures otherwise.
_DEFAULT_CUTE_TOKENS = {(12, 0): 2, (12, 1): MAX_CUTE_TOKENS}


@dataclass(frozen=True, kw_only=True)
class Bf16VocabProjectionQuery:
    dtype: str
    max_tokens: int
    in_features: int
    out_features: int


@dataclass(frozen=True, kw_only=True)
class Bf16VocabProjectionConfig:
    backend: str
    algorithm: str
    block_k: int
    num_warps: int

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "Bf16VocabProjectionConfig":
        expected = {"backend", "algorithm", "block_k", "num_warps"}
        if set(payload) != expected:
            raise ValueError(
                "BF16 vocabulary projection configs require backend, "
                "algorithm, block_k, and num_warps"
            )
        return cls(
            backend=str(payload["backend"]),
            algorithm=str(payload["algorithm"]),
            block_k=int(payload["block_k"]),
            num_warps=int(payload["num_warps"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "algorithm": self.algorithm,
            "block_k": self.block_k,
            "num_warps": self.num_warps,
        }


def _encode(query: Bf16VocabProjectionQuery) -> dict[str, object]:
    return {
        name: getattr(query, name)
        for name in Bf16VocabProjectionQuery.__dataclass_fields__
    }


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _default_config(
    query: Bf16VocabProjectionQuery,
    device: DeviceIdentity | None,
) -> Bf16VocabProjectionConfig:
    cute_tokens = (
        0 if device is None else _DEFAULT_CUTE_TOKENS.get(device.compute_capability, 0)
    )
    if (
        cute_tokens
        and query.dtype == "bfloat16"
        and 0 < query.in_features <= MAX_IN_FEATURES
        and query.out_features >= MIN_NATIVE_OUT_FEATURES
    ):
        if query.max_tokens == 1:
            return Bf16VocabProjectionConfig(
                backend="triton",
                algorithm="row",
                block_k=_next_power_of_two(query.in_features),
                num_warps=8,
            )
        if query.max_tokens <= cute_tokens:
            return Bf16VocabProjectionConfig(
                backend="cute", algorithm="simt", block_k=0, num_warps=0
            )
    return Bf16VocabProjectionConfig(
        backend="torch",
        algorithm="torch",
        block_k=0,
        num_warps=0,
    )


def _validate_query(
    query: Bf16VocabProjectionQuery,
    _device: DeviceIdentity | None,
) -> None:
    if not isinstance(query, Bf16VocabProjectionQuery):
        raise TypeError("query must be Bf16VocabProjectionQuery")
    if query.dtype != "bfloat16":
        raise ValueError(f"unsupported vocabulary projection dtype {query.dtype!r}")
    if query.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if query.in_features <= 0 or query.out_features <= 0:
        raise ValueError("projection dimensions must be positive")


def _validate_config(
    query: Bf16VocabProjectionQuery,
    config: Bf16VocabProjectionConfig,
    _device: DeviceIdentity | None,
) -> None:
    if not isinstance(config, Bf16VocabProjectionConfig):
        raise TypeError("config must be Bf16VocabProjectionConfig")
    if config.backend == "torch":
        if (config.algorithm, config.block_k, config.num_warps) != ("torch", 0, 0):
            raise ValueError("torch projection configs cannot carry native knobs")
        return
    if config.backend == "cute":
        if (config.algorithm, config.block_k, config.num_warps) != ("simt", 0, 0):
            raise ValueError("CuTe projection configs select only the SIMT GEMV")
        if query.max_tokens > MAX_CUTE_TOKENS:
            raise ValueError(
                f"the CuTe vocabulary GEMV supports max_tokens <= {MAX_CUTE_TOKENS}"
            )
        return
    if config.backend != "triton":
        raise ValueError(f"unsupported projection backend {config.backend!r}")
    if query.max_tokens != 1:
        raise ValueError("the Triton vocabulary GEMV requires max_tokens=1")
    if query.in_features > MAX_IN_FEATURES:
        raise ValueError(f"the Triton vocabulary GEMV supports K <= {MAX_IN_FEATURES}")
    if config.num_warps not in _TRITON_WARPS:
        raise ValueError(f"unsupported Triton warp count {config.num_warps}")
    if config.algorithm == "row":
        if (
            config.block_k < query.in_features
            or config.block_k > MAX_IN_FEATURES
            or config.block_k & (config.block_k - 1)
        ):
            raise ValueError("row block_k must be a covering power of two")
    elif config.algorithm == "loop":
        if config.block_k not in _LOOP_BLOCKS:
            raise ValueError(f"unsupported loop block_k {config.block_k}")
    else:
        raise ValueError(f"unsupported Triton algorithm {config.algorithm!r}")


def _paired_algorithm(choice) -> bool:
    return choice["algorithm"] in _ALGORITHMS[choice["backend"]]


def _tuning_parameters(query, device):
    row_blocks = tuple(
        1 << exponent
        for exponent in range(MAX_IN_FEATURES.bit_length())
        if 1 << exponent >= query.in_features
    )
    backends = ["torch"]
    if query.max_tokens == 1 and query.in_features <= MAX_IN_FEATURES:
        backends.append("triton")
    if query.max_tokens <= MAX_CUTE_TOKENS:
        backends.append("cute")
    return ParameterSpace.create(
        TUNING.knobs,
        values={
            "backend": tuple(backends),
            "block_k": tuple(sorted(set(row_blocks) | _LOOP_BLOCKS)),
        },
        predicates=(_paired_algorithm,),
    )


TUNING = TuningContract(
    component_id="gemm.bf16_vocab_projection",
    query_schema_version=1,
    config_schema_version=2,
    query_fields=frozenset(Bf16VocabProjectionQuery.__dataclass_fields__),
    config_fields=frozenset(Bf16VocabProjectionConfig.__dataclass_fields__),
    encode_query=_encode,
    encode_config=Bf16VocabProjectionConfig.to_dict,
    decode_config=Bf16VocabProjectionConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default_config,
    candidate_contract_version=3,
    knobs=(
        Knob(
            name="backend",
            values=("torch", "triton", "cute"),
            binding=ParameterBinding.COMPILE,
        ),
        Knob(
            name="algorithm",
            values=("torch", "row", "loop", "simt"),
            binding=ParameterBinding.COMPILE,
        ),
        Knob(
            name="block_k",
            values=None,
            binding=ParameterBinding.COMPILE,
            when=FrozenMapping({"backend": "triton"}),
            otherwise=0,
        ),
        Knob(
            name="num_warps",
            values=(1, 2, 4, 8),
            binding=ParameterBinding.COMPILE,
            when=FrozenMapping({"backend": "triton"}),
            otherwise=0,
        ),
    ),
    parameters=_tuning_parameters,
)


__all__ = [
    "TUNING",
    "Bf16VocabProjectionConfig",
    "Bf16VocabProjectionQuery",
    "MAX_CUTE_TOKENS",
    "MAX_IN_FEATURES",
    "MIN_NATIVE_OUT_FEATURES",
]
