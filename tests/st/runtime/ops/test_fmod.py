# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile element-wise floating-point remainder operations.

Covers two tile-level ops:
- ``tile.fmod``  (tile vs tile)   -> ``pto.tfmod``
- ``tile.fmods`` (tile vs scalar) -> ``pto.tfmods``

The result matches ``torch.fmod`` (the result takes the sign of the dividend).
The divisor is kept strictly non-zero in every case.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

M = 16
N = 16
# Non-zero scalar divisors covering negatives and positives.
SCALARS = [-2.5, 2.5, 3.0]


def _lhs() -> torch.Tensor:
    """Dividend range covering negatives, zero, and positives."""
    return (torch.arange(M * N, dtype=torch.float32).reshape(M, N).remainder(13) - 6).contiguous()


def _rhs() -> torch.Tensor:
    """Strictly non-zero divisor (1.5 .. 5.5)."""
    return (torch.arange(M * N, dtype=torch.float32).reshape(M, N).remainder(5) + 1.5).contiguous()


@pl.program
class TileFmodProgram:
    """Element-wise floating-point remainder of two FP32 tiles."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[M, N], pl.FP32],
        rhs: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        lhs_tile: pl.Tile[[M, N], pl.FP32] = pl.load(lhs, [0, 0], [M, N], valid_shapes=[M, N])
        rhs_tile: pl.Tile[[M, N], pl.FP32] = pl.load(rhs, [0, 0], [M, N], valid_shapes=[M, N])
        out_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.fmod(lhs_tile, rhs_tile)
        out = pl.store(out_tile, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        lhs: pl.Tensor[[M, N], pl.FP32],
        rhs: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        out = self.kernel(lhs, rhs, out)
        return out


def _make_tile_fmods_program(scalar: float):
    """Build a tile.fmods program parametrized by scalar divisor."""

    @pl.program
    class TileFmodsProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            lhs_tile: pl.Tile[[M, N], pl.FP32] = pl.load(lhs, [0, 0], [M, N], valid_shapes=[M, N])
            out_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.fmods(lhs_tile, scalar)
            out = pl.store(out_tile, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            lhs: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            out = self.kernel(lhs, out)
            return out

    return TileFmodsProgram


class TileFmodTestCase(PTOTestCase):
    """tile.fmod: element-wise floating-point remainder of two FP32 tiles."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_fmod"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_lhs()),
            TensorSpec("rhs", [M, N], DataType.FP32, init_value=_rhs()),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileFmodProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.fmod(tensors["lhs"], tensors["rhs"])


class TileFmodsTestCase(PTOTestCase):
    """tile.fmods: element-wise floating-point remainder of an FP32 tile with a scalar."""

    __test__ = False

    def __init__(self, scalar: float, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self._scalar = scalar

    def get_name(self) -> str:
        return f"tile_fmods_s{self._scalar}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_lhs()),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _make_tile_fmods_program(self._scalar)

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        scalar = torch.tensor(self._scalar, dtype=tensors["lhs"].dtype)
        tensors["out"][:] = torch.fmod(tensors["lhs"], scalar)


class TestTileFmodOperations:
    """Test tile element-wise floating-point remainder ops across supported platforms."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tile_fmod(self, test_runner, platform):
        result = test_runner.run(TileFmodTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("scalar", SCALARS)
    def test_tile_fmods(self, test_runner, platform, scalar):
        result = test_runner.run(TileFmodsTestCase(scalar, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


# ---------------------------------------------------------------------------
# Tensor-level ops: rely on ConvertTensorToTileOps to dispatch
# tensor.fmod -> tile.fmod (tensor rhs) and tensor.fmods -> tile.fmods (scalar
# rhs). Use Opaque + pl.at(CORE_GROUP) + pl.assemble to write back to Out.
# ---------------------------------------------------------------------------


@pl.program
class TensorFmodProgram:
    """Element-wise floating-point remainder of two FP32 tensors; lowers to tile.fmod."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        lhs: pl.Tensor[[M, N], pl.FP32],
        rhs: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.assemble(out, pl.fmod(lhs, rhs), [0, 0])
        return out


def _make_tensor_fmods_program(scalar: float):
    @pl.program
    class TensorFmodsProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            lhs: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.assemble(out, pl.fmods(lhs, scalar), [0, 0])
            return out

    return TensorFmodsProgram


class TensorFmodTestCase(PTOTestCase):
    """tensor.fmod (tensor-tensor): lowers to tile.fmod."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "tensor_fmod"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_lhs()),
            TensorSpec("rhs", [M, N], DataType.FP32, init_value=_rhs()),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TensorFmodProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.fmod(tensors["lhs"], tensors["rhs"])


class TensorFmodsScalarTestCase(PTOTestCase):
    """tensor.fmods (tensor-scalar): lowers to tile.fmods."""

    __test__ = False

    def __init__(self, scalar: float, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self._scalar = scalar

    def get_name(self) -> str:
        return f"tensor_fmods_s{self._scalar}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_lhs()),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _make_tensor_fmods_program(self._scalar)

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        scalar = torch.tensor(self._scalar, dtype=tensors["lhs"].dtype)
        tensors["out"][:] = torch.fmod(tensors["lhs"], scalar)


class TestTensorFmodOperations:
    """Test tensor-level fmod ops (lowered by ConvertTensorToTileOps)."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tensor_fmod(self, test_runner, platform):
        result = test_runner.run(TensorFmodTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("scalar", SCALARS)
    def test_tensor_fmods(self, test_runner, platform, scalar):
        result = test_runner.run(TensorFmodsScalarTestCase(scalar, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
