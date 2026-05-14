# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile comparison operations."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

M = 16
N = 16

_OUTPUT_NAMES = ("eq", "ne", "lt", "le", "gt", "ge")


def _cmp_lhs() -> torch.Tensor:
    values = torch.arange(M * N, dtype=torch.float32).reshape(M, N)
    return values.remainder(9) - 4


def _cmp_rhs() -> torch.Tensor:
    values = torch.arange(M * N, dtype=torch.float32).reshape(M, N)
    return values.remainder(7) - 3


def _write_expected_outputs(outputs: dict[str, torch.Tensor], lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    comparisons = {
        "eq": lhs == rhs,
        "ne": lhs != rhs,
        "lt": lhs < rhs,
        "le": lhs <= rhs,
        "gt": lhs > rhs,
        "ge": lhs >= rhs,
    }
    for name, result in comparisons.items():
        outputs[name][:] = result.to(outputs[name].dtype)


@pl.program
class TileCmpProgram:
    """Tile-to-tile comparison for all cmp_type modes."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[M, N], pl.FP32],
        rhs: pl.Tensor[[M, N], pl.FP32],
        eq: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ne: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        lt: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        le: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        gt: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ge: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
    ]:
        lhs_tile: pl.Tile[[M, N], pl.FP32] = pl.load(lhs, [0, 0], [M, N])
        rhs_tile: pl.Tile[[M, N], pl.FP32] = pl.load(rhs, [0, 0], [M, N])
        one_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.full([M, N], dtype=pl.FP32, value=1.0)
        zero_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.full([M, N], dtype=pl.FP32, value=0.0)
        tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
        eq_mask: pl.Tile[[M, 32], pl.UINT8] = pl.tile.cmp(lhs_tile, rhs_tile, cmp_type=0)
        ne_mask: pl.Tile[[M, 32], pl.UINT8] = pl.tile.cmp(lhs_tile, rhs_tile, cmp_type=1)
        lt_mask: pl.Tile[[M, 32], pl.UINT8] = pl.tile.cmp(lhs_tile, rhs_tile, cmp_type=2)
        le_mask: pl.Tile[[M, 32], pl.UINT8] = pl.tile.cmp(lhs_tile, rhs_tile, cmp_type=3)
        gt_mask: pl.Tile[[M, 32], pl.UINT8] = pl.tile.cmp(lhs_tile, rhs_tile, cmp_type=4)
        ge_mask: pl.Tile[[M, 32], pl.UINT8] = pl.tile.cmp(lhs_tile, rhs_tile, cmp_type=5)
        eq = pl.store(pl.tile.sel(eq_mask, one_tile, zero_tile, tmp), [0, 0], eq)
        ne = pl.store(pl.tile.sel(ne_mask, one_tile, zero_tile, tmp), [0, 0], ne)
        lt = pl.store(pl.tile.sel(lt_mask, one_tile, zero_tile, tmp), [0, 0], lt)
        le = pl.store(pl.tile.sel(le_mask, one_tile, zero_tile, tmp), [0, 0], le)
        gt = pl.store(pl.tile.sel(gt_mask, one_tile, zero_tile, tmp), [0, 0], gt)
        ge = pl.store(pl.tile.sel(ge_mask, one_tile, zero_tile, tmp), [0, 0], ge)
        return eq, ne, lt, le, gt, ge

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        lhs: pl.Tensor[[M, N], pl.FP32],
        rhs: pl.Tensor[[M, N], pl.FP32],
        eq: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ne: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        lt: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        le: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        gt: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ge: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
    ]:
        eq, ne, lt, le, gt, ge = self.kernel(lhs, rhs, eq, ne, lt, le, gt, ge)
        return eq, ne, lt, le, gt, ge


@pl.program
class TileCmpsProgram:
    """Tile-to-scalar comparison for all cmp_type modes."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[M, N], pl.FP32],
        eq: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ne: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        lt: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        le: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        gt: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ge: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
    ]:
        lhs_tile: pl.Tile[[M, N], pl.FP32] = pl.load(lhs, [0, 0], [M, N])
        one_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.full([M, N], dtype=pl.FP32, value=1.0)
        zero_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.full([M, N], dtype=pl.FP32, value=0.0)
        tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
        eq_mask: pl.Tile[[M, 32], pl.UINT8] = pl.tile.cmps(lhs_tile, 0.0, cmp_type=0)
        ne_mask: pl.Tile[[M, 32], pl.UINT8] = pl.tile.cmps(lhs_tile, 0.0, cmp_type=1)
        lt_mask: pl.Tile[[M, 32], pl.UINT8] = pl.tile.cmps(lhs_tile, 0.0, cmp_type=2)
        le_mask: pl.Tile[[M, 32], pl.UINT8] = pl.tile.cmps(lhs_tile, 0.0, cmp_type=3)
        gt_mask: pl.Tile[[M, 32], pl.UINT8] = pl.tile.cmps(lhs_tile, 0.0, cmp_type=4)
        ge_mask: pl.Tile[[M, 32], pl.UINT8] = pl.tile.cmps(lhs_tile, 0.0, cmp_type=5)
        eq = pl.store(pl.tile.sel(eq_mask, one_tile, zero_tile, tmp), [0, 0], eq)
        ne = pl.store(pl.tile.sel(ne_mask, one_tile, zero_tile, tmp), [0, 0], ne)
        lt = pl.store(pl.tile.sel(lt_mask, one_tile, zero_tile, tmp), [0, 0], lt)
        le = pl.store(pl.tile.sel(le_mask, one_tile, zero_tile, tmp), [0, 0], le)
        gt = pl.store(pl.tile.sel(gt_mask, one_tile, zero_tile, tmp), [0, 0], gt)
        ge = pl.store(pl.tile.sel(ge_mask, one_tile, zero_tile, tmp), [0, 0], ge)
        return eq, ne, lt, le, gt, ge

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        lhs: pl.Tensor[[M, N], pl.FP32],
        eq: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ne: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        lt: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        le: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        gt: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ge: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
    ]:
        eq, ne, lt, le, gt, ge = self.kernel(lhs, eq, ne, lt, le, gt, ge)
        return eq, ne, lt, le, gt, ge


class TileCmpTestCase(PTOTestCase):
    """Tile cmp: compare two FP32 tiles."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "tile_cmp"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_cmp_lhs()),
            TensorSpec("rhs", [M, N], DataType.FP32, init_value=_cmp_rhs()),
            *(TensorSpec(name, [M, N], DataType.FP32, is_output=True) for name in _OUTPUT_NAMES),
        ]

    def get_program(self) -> Any:
        return TileCmpProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        _write_expected_outputs(tensors, tensors["lhs"], tensors["rhs"])


class TileCmpsTestCase(PTOTestCase):
    """Tile cmps: compare an FP32 tile with scalar zero."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "tile_cmps"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_cmp_lhs()),
            *(TensorSpec(name, [M, N], DataType.FP32, is_output=True) for name in _OUTPUT_NAMES),
        ]

    def get_program(self) -> Any:
        return TileCmpsProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        _write_expected_outputs(tensors, tensors["lhs"], torch.tensor(0.0, dtype=tensors["lhs"].dtype))


@pl.program
class TensorCmpProgram:
    """Tensor-to-tensor comparison for all cmp_type modes; lowers to tile.cmp + tile.full + tile.sel."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        lhs: pl.Tensor[[M, N], pl.FP32],
        rhs: pl.Tensor[[M, N], pl.FP32],
        eq: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ne: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        lt: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        le: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        gt: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ge: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
    ]:
        with pl.at(level=pl.Level.CORE_GROUP):
            eq = pl.assemble(eq, pl.cmp(lhs, rhs, cmp_type=0), [0, 0])
            ne = pl.assemble(ne, pl.cmp(lhs, rhs, cmp_type=1), [0, 0])
            lt = pl.assemble(lt, pl.cmp(lhs, rhs, cmp_type=2), [0, 0])
            le = pl.assemble(le, pl.cmp(lhs, rhs, cmp_type=3), [0, 0])
            gt = pl.assemble(gt, pl.cmp(lhs, rhs, cmp_type=4), [0, 0])
            ge = pl.assemble(ge, pl.cmp(lhs, rhs, cmp_type=5), [0, 0])
        return eq, ne, lt, le, gt, ge


@pl.program
class TensorCmpsProgram:
    """Tensor-to-scalar comparison; lowers to tile.cmps + tile.full + tile.sel."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        lhs: pl.Tensor[[M, N], pl.FP32],
        eq: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ne: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        lt: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        le: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        gt: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ge: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
    ]:
        with pl.at(level=pl.Level.CORE_GROUP):
            eq = pl.assemble(eq, pl.cmp(lhs, 0.0, cmp_type=0), [0, 0])
            ne = pl.assemble(ne, pl.cmp(lhs, 0.0, cmp_type=1), [0, 0])
            lt = pl.assemble(lt, pl.cmp(lhs, 0.0, cmp_type=2), [0, 0])
            le = pl.assemble(le, pl.cmp(lhs, 0.0, cmp_type=3), [0, 0])
            gt = pl.assemble(gt, pl.cmp(lhs, 0.0, cmp_type=4), [0, 0])
            ge = pl.assemble(ge, pl.cmp(lhs, 0.0, cmp_type=5), [0, 0])
        return eq, ne, lt, le, gt, ge


class TensorCmpTestCase(PTOTestCase):
    """Tensor cmp: compare two FP32 tensors; pass-driven tile lowering."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "tensor_cmp"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_cmp_lhs()),
            TensorSpec("rhs", [M, N], DataType.FP32, init_value=_cmp_rhs()),
            *(TensorSpec(name, [M, N], DataType.FP32, is_output=True) for name in _OUTPUT_NAMES),
        ]

    def get_program(self) -> Any:
        return TensorCmpProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        _write_expected_outputs(tensors, tensors["lhs"], tensors["rhs"])


class TensorCmpsTestCase(PTOTestCase):
    """Tensor cmps: compare an FP32 tensor with scalar zero; pass-driven tile lowering."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "tensor_cmps"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("lhs", [M, N], DataType.FP32, init_value=_cmp_lhs()),
            *(TensorSpec(name, [M, N], DataType.FP32, is_output=True) for name in _OUTPUT_NAMES),
        ]

    def get_program(self) -> Any:
        return TensorCmpsProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        _write_expected_outputs(tensors, tensors["lhs"], torch.tensor(0.0, dtype=tensors["lhs"].dtype))


class TestTileCmpOperations:
    """Test tile comparison operations across supported platforms."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tile_cmp(self, test_runner, platform):
        result = test_runner.run(TileCmpTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tile_cmps(self, test_runner, platform):
        result = test_runner.run(TileCmpsTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


class TestTensorCmpOperations:
    """Test tensor-level comparison ops (lowered by ConvertTensorToTileOps)."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tensor_cmp(self, test_runner, platform):
        result = test_runner.run(TensorCmpTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tensor_cmps(self, test_runner, platform):
        result = test_runner.run(TensorCmpsTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
