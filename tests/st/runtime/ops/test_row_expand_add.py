# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for row-wise broadcast add: row_expand_add (TROWEXPANDADD).

Adds a per-row COLUMN vector ``row_vec`` of shape ``[M, 1]`` to every column of
the main tile: ``dst[i, j] = tile[i, j] + row_vec[i, 0]``. The op declares
``forbid_output_alias(1)`` (the output may not alias the row_vec operand), so the
result is a fresh tile. Golden: ``a + row_vec`` (broadcast along columns).

Covers two usage scenarios of the op:
- tile-level ``tile.row_expand_add(tile, row_vec)`` (InCore + Orchestration), with
  ``row_vec`` fed as a separate ``[M, 1]`` input tile.
- tensor-level ``tensor.row_expand_add(tensor, row_vec)`` (Opaque + pl.at(CORE_GROUP)
  + pl.assemble), lowered by ConvertTensorToTileOps.

Coverage: multiple shapes (square/tall/wide), FP32 + FP16, and a narrowed
valid_shape (reduce over valid cols / keep only valid rows). row_expand_add is a
single FP16-rounded add per element, so FP16 stays at the default tolerance.

Scope is a2a3 only (``@pytest.mark.platforms("a2a3")``); a5 coverage is a separate PR.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

_PL_DT = {DataType.FP32: pl.FP32, DataType.FP16: pl.FP16}


def _signed(m, n):
    return torch.randn(m, n, dtype=torch.float32)


# =============================================================================
# Tile-level: tile.row_expand_add(tile, row_vec) — row_vec is a separate [M, 1] input.
# =============================================================================


class _TileRowExpandAddBase(PTOTestCase):
    __test__ = False

    def __init__(self, *, m=16, n=16, valid_shapes=None, dtype=DataType.FP32, config=None):
        super().__init__(config)
        self._m, self._n, self._valid, self._dtype = m, n, valid_shapes, dtype

    def get_name(self) -> str:
        v = f"_v{self._valid[0]}x{self._valid[1]}" if self._valid else ""
        return f"tile_row_expand_add_{self._m}x{self._n}_{self._dtype.value}{v}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._m, self._n], self._dtype, init_value=lambda: _signed(self._m, self._n)),
            TensorSpec("row_vec", [self._m, 1], self._dtype, init_value=lambda: _signed(self._m, 1)),
            TensorSpec("out", [self._m, self._n], self._dtype, is_output=True),
        ]

    def get_program(self) -> Any:
        m, n = self._m, self._n
        vshape = list(self._valid) if self._valid else [m, n]
        dt = _PL_DT[self._dtype]

        @pl.program
        class RowExpandAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[m, n], dt],
                row_vec: pl.Tensor[[m, 1], dt],
                out: pl.Out[pl.Tensor[[m, n], dt]],
            ) -> pl.Tensor[[m, n], dt]:
                a_tile: pl.Tile[[m, n], dt] = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                rv_tile: pl.Tile[[m, 1], dt] = pl.load(row_vec, [0, 0], [m, 1])
                res: pl.Tile[[m, n], dt] = pl.tile.row_expand_add(a_tile, rv_tile)
                out = pl.store(res, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[m, n], dt],
                row_vec: pl.Tensor[[m, 1], dt],
                out: pl.Out[pl.Tensor[[m, n], dt]],
            ) -> pl.Tensor[[m, n], dt]:
                out = self.kernel(a, row_vec, out)
                return out

        return RowExpandAddProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        rv = tensors["row_vec"]  # [M, 1]
        out = torch.zeros_like(tensors["out"])  # [M, N]
        if self._valid:
            vm, vn = self._valid
            # Only the valid sub-region is defined: valid rows, valid cols.
            out[:vm, :vn] = a[:vm, :vn] + rv[:vm, :]
        else:
            out[:] = a + rv
        tensors["out"][:] = out


# =============================================================================
# Tensor-level: tensor.row_expand_add(tensor, row_vec) — lowered to tile op.
# =============================================================================


class _TensorRowExpandAddBase(PTOTestCase):
    __test__ = False

    def __init__(self, *, m=16, n=16, dtype=DataType.FP32, config=None):
        super().__init__(config)
        self._m, self._n, self._dtype = m, n, dtype

    def get_name(self) -> str:
        return f"tensor_row_expand_add_{self._m}x{self._n}_{self._dtype.value}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._m, self._n], self._dtype, init_value=lambda: _signed(self._m, self._n)),
            TensorSpec("row_vec", [self._m, 1], self._dtype, init_value=lambda: _signed(self._m, 1)),
            TensorSpec("out", [self._m, self._n], self._dtype, is_output=True),
        ]

    def get_program(self) -> Any:
        m, n = self._m, self._n
        dt = _PL_DT[self._dtype]

        @pl.program
        class TensorRowExpandAddProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                a: pl.Tensor[[m, n], dt],
                row_vec: pl.Tensor[[m, 1], dt],
                out: pl.Out[pl.Tensor[[m, n], dt]],
            ) -> pl.Tensor[[m, n], dt]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.assemble(out, pl.row_expand_add(a, row_vec), [0, 0])
                return out

        return TensorRowExpandAddProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["a"] + tensors["row_vec"]


# Full-shape cases: square / tall / wide.
_SHAPE_CFGS = [
    ("16x16", 16, 16),
    ("32x64", 32, 64),
    ("8x128", 8, 128),
]


class TestTileRowExpandAdd:
    """tile.row_expand_add: per-row broadcast add across shapes, dtypes, valid_shape."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_row_expand_add(self, test_runner, label, m, n):
        result = test_runner.run(_TileRowExpandAddBase(m=m, n=n))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_row_expand_add_fp16(self, test_runner):
        result = test_runner.run(_TileRowExpandAddBase(m=32, n=64, dtype=DataType.FP16))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_row_expand_add_valid(self, test_runner):
        # Narrowed valid_shape: valid rows 0:8, valid cols 0:12 of a 16x16 tile.
        result = test_runner.run(_TileRowExpandAddBase(m=16, n=16, valid_shapes=(8, 12)))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_row_expand_add_valid_fp16(self, test_runner):
        result = test_runner.run(_TileRowExpandAddBase(m=16, n=16, valid_shapes=(8, 12), dtype=DataType.FP16))
        assert result.passed, f"Test failed: {result.error}"


class TestTensorRowExpandAdd:
    """tensor.row_expand_add: lowered via ConvertTensorToTileOps."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tensor_row_expand_add(self, test_runner, label, m, n):
        result = test_runner.run(_TensorRowExpandAddBase(m=m, n=n))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tensor_row_expand_add_fp16(self, test_runner):
        result = test_runner.run(_TensorRowExpandAddBase(m=32, n=64, dtype=DataType.FP16))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
