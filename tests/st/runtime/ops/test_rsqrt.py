# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Reciprocal Square Root (rsqrt) System Tests.

Covers both DSL layers exposed by commit 8b8b055e (pto.trsqrt high-precision mode):

  TileRsqrtTest              : Tile level, 1-operand basic form.      out = rsqrt(a)
  TileRsqrtHighPrecisionTest : Tile level, 2-operand (user-managed    out = rsqrt(a) (high precision)
                               scratch tile via pl.tile.rsqrt(src, tmp=...)).
  TensorRsqrtTest            : Tensor level, basic form               out = pl.rsqrt(x)
  TensorRsqrtHighPrecisionTest : Tensor level, compiler-inserted      out = pl.rsqrt(x, high_precision=True)
                               scratch allocation.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

M = 16
N = 16

# rsqrt on hardware uses a Newton-Raphson approximation whose precision is looser
# than torch.rsqrt; the high-precision path additionally involves a scratch-tile
# refinement step.  Use relaxed tolerances similar to the qwen3 decode tests.
_RSQRT_RTOL = 1e-2
_RSQRT_ATOL = 1e-2


def _positive_input(shape: list[int]) -> torch.Tensor:
    """Strictly-positive tensor so rsqrt input stays in a well-defined range."""
    return torch.rand(shape) + 0.5


def _relaxed_config() -> RunConfig:
    return RunConfig(rtol=_RSQRT_RTOL, atol=_RSQRT_ATOL)


# ---------------------------------------------------------------------------
# Tile level: basic (1-operand) pto.trsqrt
# ---------------------------------------------------------------------------


@pl.program
class TileRsqrtProgram:
    """Tile-level rsqrt using the basic 1-operand form."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        tile_a: pl.Tile[[M, N], pl.FP32] = pl.load(a, [0, 0], [M, N])
        tile_c: pl.Tile[[M, N], pl.FP32] = pl.tile.rsqrt(tile_a)
        out = pl.store(tile_c, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        out = self.kernel(a, out)
        return out


class TileRsqrtTest(PTOTestCase):
    """Tile rsqrt basic: out = rsqrt(a)."""

    __test__ = False

    def __init__(self, config=None):
        super().__init__(config or _relaxed_config())

    def get_name(self) -> str:
        return "tile_rsqrt_basic"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.FP32, init_value=_positive_input([M, N])),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileRsqrtProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.rsqrt(tensors["a"])


# ---------------------------------------------------------------------------
# Tile level: high-precision (2-operand) pto.trsqrt with user-managed tmp tile
# ---------------------------------------------------------------------------


@pl.program
class TileRsqrtHighPrecisionProgram:
    """Tile-level rsqrt using the 2-operand high-precision form (tmp allocated by the user)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        tile_a: pl.Tile[[M, N], pl.FP32] = pl.load(a, [0, 0], [M, N])
        tmp: pl.Tile[[M, N], pl.FP32] = pl.tile.create(
            [M, N], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        tile_c: pl.Tile[[M, N], pl.FP32] = pl.tile.rsqrt(tile_a, tmp=tmp)
        out = pl.store(tile_c, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        out = self.kernel(a, out)
        return out


class TileRsqrtHighPrecisionTest(PTOTestCase):
    """Tile rsqrt high-precision: out = rsqrt(a) via pl.tile.rsqrt(src, tmp=...)."""

    __test__ = False

    def __init__(self, config=None):
        super().__init__(config or _relaxed_config())

    def get_name(self) -> str:
        return "tile_rsqrt_high_precision"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.FP32, init_value=_positive_input([M, N])),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileRsqrtHighPrecisionProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.rsqrt(tensors["a"])


# ---------------------------------------------------------------------------
# Tensor level: basic pl.rsqrt
# ---------------------------------------------------------------------------


@pl.program
class TensorRsqrtProgram:
    """Tensor-level rsqrt using the basic form (compiler lowers to 1-operand pto.trsqrt)."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        x: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            y = pl.rsqrt(x)
            out = pl.assemble(out, y, [0, 0])
        return out


class TensorRsqrtTest(PTOTestCase):
    """Tensor rsqrt basic: out = pl.rsqrt(x)."""

    __test__ = False

    def __init__(self, config=None):
        super().__init__(config or _relaxed_config())

    def get_name(self) -> str:
        return "tensor_rsqrt_basic"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [M, N], DataType.FP32, init_value=_positive_input([M, N])),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TensorRsqrtProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.rsqrt(tensors["x"])


# ---------------------------------------------------------------------------
# Tensor level: high-precision pl.rsqrt(x, high_precision=True)
# ---------------------------------------------------------------------------


@pl.program
class TensorRsqrtHighPrecisionProgram:
    """Tensor-level rsqrt with high_precision=True (compiler inserts the scratch tile)."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        x: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            y = pl.rsqrt(x, high_precision=True)
            out = pl.assemble(out, y, [0, 0])
        return out


class TensorRsqrtHighPrecisionTest(PTOTestCase):
    """Tensor rsqrt high-precision: out = pl.rsqrt(x, high_precision=True)."""

    __test__ = False

    def __init__(self, config=None):
        super().__init__(config or _relaxed_config())

    def get_name(self) -> str:
        return "tensor_rsqrt_high_precision"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [M, N], DataType.FP32, init_value=_positive_input([M, N])),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TensorRsqrtHighPrecisionProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.rsqrt(tensors["x"])


# ---------------------------------------------------------------------------
# pytest wrappers
# ---------------------------------------------------------------------------


class TestRsqrt:
    """End-to-end rsqrt tests at both Tile and Tensor DSL levels."""

    def test_tile_rsqrt(self, test_runner):
        """Tile-level basic 1-operand pto.trsqrt."""
        result = test_runner.run(TileRsqrtTest())
        assert result.passed, f"Tile rsqrt basic failed: {result.error}"

    def test_tile_rsqrt_high_precision(self, test_runner):
        """Tile-level 2-operand high-precision pto.trsqrt."""
        result = test_runner.run(TileRsqrtHighPrecisionTest())
        assert result.passed, f"Tile rsqrt high-precision failed: {result.error}"

    def test_tensor_rsqrt(self, test_runner):
        """Tensor-level basic pl.rsqrt."""
        result = test_runner.run(TensorRsqrtTest())
        assert result.passed, f"Tensor rsqrt basic failed: {result.error}"

    def test_tensor_rsqrt_high_precision(self, test_runner):
        """Tensor-level pl.rsqrt(high_precision=True) — compiler-inserted scratch tile."""
        result = test_runner.run(TensorRsqrtHighPrecisionTest())
        assert result.passed, f"Tensor rsqrt high-precision failed: {result.error}"


@pl.jit
def _a5_tensor_rsqrt_precision(
    x: pl.Tensor[[M, N], pl.FP32],
    precise: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    default: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        precise[:, :] = pl.rsqrt(x, high_precision=True)
        default[:, :] = pl.rsqrt(x)
    return precise, default


@pl.jit.incore
def _a5_tile_rsqrt_precision(
    x: pl.Tensor[[M, N], pl.FP32],
    precise: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    default: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    a = pl.load(x, [0, 0], [M, N])
    tmp = pl.tile.create([M, N], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    precise = pl.store(pl.tile.rsqrt(a, tmp=tmp), [0, 0], precise)
    default = pl.store(pl.tile.rsqrt(a), [0, 0], default)
    return precise, default


def _a5_rounding_inputs() -> torch.Tensor:
    # Normal positive FP32 values around rounding boundaries across 41 binades.
    # Use fixed bit patterns so device results do not depend on RNG state.
    i = torch.arange(M * N, dtype=torch.int64)
    mantissa = (i * 1664525 + 1013904223) & 0x7FFFFF
    bits = (((i % 41) + 107) << 23) | mantissa
    return bits.to(torch.int32).view(torch.float32).reshape(M, N)


def _a5_rsqrt_golden(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Float64 avoids making the CPU's FP32 rsqrt approximation our oracle.
    reference = torch.rsqrt(tensors["x"].double()).float()
    return {"precise": reference, "default": reference}


def _compare_a5_rsqrt_ulp(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    errors = {}
    for name in ("precise", "default"):
        value = actual[name]
        assert torch.isfinite(value).all() and (value > 0).all(), name
        # Positive finite floats are ordered by their integer bit patterns.
        errors[name] = (
            value.view(torch.int32).to(torch.int64) - expected[name].view(torch.int32).to(torch.int64)
        ).abs()
    # A5 high precision uses a rounded sqrt followed by IEEE754 division, so
    # allow one ULP against the correctly rounded reciprocal square root.
    assert errors["precise"].max().item() <= 1, errors["precise"]
    # This is essential: with the missing selector both outputs use DEFAULT
    # and even a tight tolerance can accidentally let the regression pass.
    assert (errors["precise"] < errors["default"]).any(), "high precision never improves DEFAULT"
    assert errors["precise"].sum() < errors["default"].sum(), "high precision must reduce total ULP error"


@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            kernel,
            name=f"a5_rsqrt_{entry}_precision_ulp",
            tensors=[
                TensorSpec("x", [M, N], DataType.FP32, init_value=_a5_rounding_inputs),
                TensorSpec("precise", [M, N], DataType.FP32, is_output=True),
                TensorSpec("default", [M, N], DataType.FP32, is_output=True),
            ],
            golden=_a5_rsqrt_golden,
            compare=_compare_a5_rsqrt_ulp,
            enable_buffer_ir=False,
        )
        for entry, kernel in (("tensor", _a5_tensor_rsqrt_precision), ("tile", _a5_tile_rsqrt_precision))
    ]
)
def test_a5_rsqrt_precision_ulp(case_run):
    """Both API entries must select an observably more accurate A5 algorithm."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
