# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile element-wise maximum/minimum operations.

Covers four tile-level ops:
- ``tile.maximum``  (tile vs tile)  -> ``pto.tmax``
- ``tile.maximums`` (tile vs scalar) -> ``pto.tmaxs``
- ``tile.minimum``  (tile vs tile)  -> ``pto.tmin``
- ``tile.minimums`` (tile vs scalar) -> ``pto.tmins``
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

M = 16
N = 16
# Cover negative, zero, and positive scalars.
SCALARS = [-2.5, 0.0, 2.5]


def _lhs() -> torch.Tensor:
    """Range covering negatives, zero, and positives."""
    return (torch.arange(M * N, dtype=torch.float32).reshape(M, N).remainder(9) - 4).contiguous()


def _rhs() -> torch.Tensor:
    """Different distribution to ensure max/min produce a mix from both sides."""
    return (torch.arange(M * N, dtype=torch.float32).reshape(M, N).remainder(7) - 3).contiguous()


@pl.program
class TileMaximumProgram:
    """Element-wise maximum of two FP32 tiles."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[M, N], pl.FP32],
        rhs: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        lhs_tile: pl.Tile[[M, N], pl.FP32] = pl.load(lhs, [0, 0], [M, N], valid_shapes=[M, N])
        rhs_tile: pl.Tile[[M, N], pl.FP32] = pl.load(rhs, [0, 0], [M, N], valid_shapes=[M, N])
        out_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.maximum(lhs_tile, rhs_tile)
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


@pl.program
class TileMinimumProgram:
    """Element-wise minimum of two FP32 tiles."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[M, N], pl.FP32],
        rhs: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        lhs_tile: pl.Tile[[M, N], pl.FP32] = pl.load(lhs, [0, 0], [M, N], valid_shapes=[M, N])
        rhs_tile: pl.Tile[[M, N], pl.FP32] = pl.load(rhs, [0, 0], [M, N], valid_shapes=[M, N])
        out_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.minimum(lhs_tile, rhs_tile)
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


def _make_tile_maximums_program(scalar: float):
    """Build a program parametrized by scalar value."""

    @pl.program
    class TileMaximumsProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            lhs_tile: pl.Tile[[M, N], pl.FP32] = pl.load(lhs, [0, 0], [M, N], valid_shapes=[M, N])
            out_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.maximums(lhs_tile, scalar)
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

    return TileMaximumsProgram


def _make_tile_minimums_program(scalar: float):
    @pl.program
    class TileMinimumsProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            lhs_tile: pl.Tile[[M, N], pl.FP32] = pl.load(lhs, [0, 0], [M, N], valid_shapes=[M, N])
            out_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.minimums(lhs_tile, scalar)
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

    return TileMinimumsProgram


class TileMaximumTestCase(PTOTestCase):
    """tile.maximum: element-wise max of two FP32 tiles."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_maximum"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_lhs()),
            TensorSpec("rhs", [M, N], DataType.FP32, init_value=_rhs()),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileMaximumProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.maximum(tensors["lhs"], tensors["rhs"])


class TileMinimumTestCase(PTOTestCase):
    """tile.minimum: element-wise min of two FP32 tiles."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_minimum"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_lhs()),
            TensorSpec("rhs", [M, N], DataType.FP32, init_value=_rhs()),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileMinimumProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.minimum(tensors["lhs"], tensors["rhs"])


class TileMaximumsTestCase(PTOTestCase):
    """tile.maximums: element-wise max of an FP32 tile with a scalar."""

    __test__ = False

    def __init__(self, scalar: float, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self._scalar = scalar

    def get_name(self) -> str:
        return f"tile_maximums_s{self._scalar}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_lhs()),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _make_tile_maximums_program(self._scalar)

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        scalar = torch.tensor(self._scalar, dtype=tensors["lhs"].dtype)
        tensors["out"][:] = torch.maximum(tensors["lhs"], scalar)


class TileMinimumsTestCase(PTOTestCase):
    """tile.minimums: element-wise min of an FP32 tile with a scalar."""

    __test__ = False

    def __init__(self, scalar: float, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self._scalar = scalar

    def get_name(self) -> str:
        return f"tile_minimums_s{self._scalar}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_lhs()),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _make_tile_minimums_program(self._scalar)

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        scalar = torch.tensor(self._scalar, dtype=tensors["lhs"].dtype)
        tensors["out"][:] = torch.minimum(tensors["lhs"], scalar)


class TestTileMaxMinOperations:
    """Test tile element-wise max/min operations across supported platforms."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tile_maximum(self, test_runner, platform):
        result = test_runner.run(TileMaximumTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tile_minimum(self, test_runner, platform):
        result = test_runner.run(TileMinimumTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("scalar", SCALARS)
    def test_tile_maximums(self, test_runner, platform, scalar):
        result = test_runner.run(TileMaximumsTestCase(scalar, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("scalar", SCALARS)
    def test_tile_minimums(self, test_runner, platform, scalar):
        result = test_runner.run(TileMinimumsTestCase(scalar, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


# ---------------------------------------------------------------------------
# Tensor-level ops: rely on ConvertTensorToTileOps to dispatch to
# tile.maximum/minimum (tensor rhs) or tile.maximums/minimums (scalar rhs).
# Use Opaque + pl.at(CORE_GROUP) + pl.assemble to write back to Out params.
# ---------------------------------------------------------------------------


@pl.program
class TensorMaximumProgram:
    """Element-wise maximum of two FP32 tensors; lowers to tile.maximum."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        lhs: pl.Tensor[[M, N], pl.FP32],
        rhs: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.assemble(out, pl.tensor.maximum(lhs, rhs), [0, 0])
        return out


@pl.program
class TensorMinimumProgram:
    """Element-wise minimum of two FP32 tensors; lowers to tile.minimum."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        lhs: pl.Tensor[[M, N], pl.FP32],
        rhs: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.assemble(out, pl.tensor.minimum(lhs, rhs), [0, 0])
        return out


def _make_tensor_maximum_scalar_program(scalar: float):
    @pl.program
    class TensorMaximumScalarProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            lhs: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.assemble(out, pl.tensor.maximum(lhs, scalar), [0, 0])
            return out

    return TensorMaximumScalarProgram


def _make_tensor_minimum_scalar_program(scalar: float):
    @pl.program
    class TensorMinimumScalarProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            lhs: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.assemble(out, pl.tensor.minimum(lhs, scalar), [0, 0])
            return out

    return TensorMinimumScalarProgram


class TensorMaximumTestCase(PTOTestCase):
    """tensor.maximum (tensor-tensor): lowers to tile.maximum."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "tensor_maximum"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_lhs()),
            TensorSpec("rhs", [M, N], DataType.FP32, init_value=_rhs()),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TensorMaximumProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.maximum(tensors["lhs"], tensors["rhs"])


class TensorMinimumTestCase(PTOTestCase):
    """tensor.minimum (tensor-tensor): lowers to tile.minimum."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "tensor_minimum"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_lhs()),
            TensorSpec("rhs", [M, N], DataType.FP32, init_value=_rhs()),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TensorMinimumProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.minimum(tensors["lhs"], tensors["rhs"])


class TensorMaximumScalarTestCase(PTOTestCase):
    """tensor.maximum (tensor-scalar): lowers to tile.maximums."""

    __test__ = False

    def __init__(self, scalar: float, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self._scalar = scalar

    def get_name(self) -> str:
        return f"tensor_maximum_scalar_s{self._scalar}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_lhs()),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _make_tensor_maximum_scalar_program(self._scalar)

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        scalar = torch.tensor(self._scalar, dtype=tensors["lhs"].dtype)
        tensors["out"][:] = torch.maximum(tensors["lhs"], scalar)


class TensorMinimumScalarTestCase(PTOTestCase):
    """tensor.minimum (tensor-scalar): lowers to tile.minimums."""

    __test__ = False

    def __init__(self, scalar: float, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self._scalar = scalar

    def get_name(self) -> str:
        return f"tensor_minimum_scalar_s{self._scalar}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_lhs()),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _make_tensor_minimum_scalar_program(self._scalar)

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        scalar = torch.tensor(self._scalar, dtype=tensors["lhs"].dtype)
        tensors["out"][:] = torch.minimum(tensors["lhs"], scalar)


class TestTensorMaxMinOperations:
    """Test tensor-level max/min ops (lowered by ConvertTensorToTileOps)."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tensor_maximum(self, test_runner, platform):
        result = test_runner.run(TensorMaximumTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tensor_minimum(self, test_runner, platform):
        result = test_runner.run(TensorMinimumTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("scalar", SCALARS)
    def test_tensor_maximum_scalar(self, test_runner, platform, scalar):
        result = test_runner.run(TensorMaximumScalarTestCase(scalar, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("scalar", SCALARS)
    def test_tensor_minimum_scalar(self, test_runner, platform, scalar):
        result = test_runner.run(TensorMinimumScalarTestCase(scalar, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
