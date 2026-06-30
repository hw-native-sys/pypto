# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for row-wise min reduction: row_min (TROWMIN).

Covers two usage scenarios of the op:
- tile-level ``tile.row_min(tile, tmp)`` (InCore + Orchestration). The op REQUIRES
  an explicit scratch ``tmp`` tile of the SAME shape/dtype as the input, created in
  Vec memory (mirrors ``tile.row_prod`` in test_prod_reduction.py / row_max).
- tensor-level ``tensor.row_min(input)`` (Opaque + pl.at(CORE_GROUP) + pl.assemble);
  no tmp at tensor level — it is auto-created by ConvertTensorToTileOps.

The reduction is along axis=1: ``[M, N] -> [M, 1]`` (a per-row COLUMN vector).
Golden: ``torch.min(a, dim=1, keepdim=True).values``.

Coverage: multiple shapes (square/tall/wide), FP32 + FP16, and narrowed
valid_shape (cols-only — reduce over valid cols, all rows valid; combined — reduce
over valid cols AND keep only valid output rows). row_min is an exact selection
(no arithmetic) so FP16 stays at the default tolerance.

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
# Tile-level: tile.row_min(tile, tmp) — tmp is [M, N], same dtype, in Vec memory
# (mirrors tile.row_prod / row_max which also require the scratch tile).
# =============================================================================


class _TileRowMinBase(PTOTestCase):
    __test__ = False

    def __init__(self, *, m=16, n=16, valid_shapes=None, dtype=DataType.FP32, config=None):
        super().__init__(config)
        self._m, self._n, self._valid, self._dtype = m, n, valid_shapes, dtype

    def get_name(self) -> str:
        v = f"_v{self._valid[0]}x{self._valid[1]}" if self._valid else ""
        return f"tile_row_min_{self._m}x{self._n}_{self._dtype.value}{v}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._m, self._n], self._dtype, init_value=lambda: _signed(self._m, self._n)),
            TensorSpec("out", [self._m, 1], self._dtype, is_output=True),
        ]

    def get_program(self) -> Any:
        m, n = self._m, self._n
        vshape = list(self._valid) if self._valid else [m, n]
        dt = _PL_DT[self._dtype]

        @pl.program
        class RowMinProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[m, 1], dt]]
            ) -> pl.Tensor[[m, 1], dt]:
                a_tile: pl.Tile[[m, n], dt] = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                tmp: pl.Tile[[m, n], dt] = pl.tile.create([m, n], dtype=dt, target_memory=pl.MemorySpace.Vec)
                res: pl.Tile[[m, 1], dt] = pl.tile.row_min(a_tile, tmp)
                out = pl.store(res, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[m, 1], dt]]
            ) -> pl.Tensor[[m, 1], dt]:
                out = self.kernel(a, out)
                return out

        return RowMinProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        out = torch.zeros_like(tensors["out"])  # [M, 1]
        if self._valid:
            vm, vn = self._valid
            # Reduce over the valid columns only; only the valid rows are defined.
            out[:vm, :] = torch.min(a[:vm, :vn], dim=1, keepdim=True).values
        else:
            out[:] = torch.min(a, dim=1, keepdim=True).values
        tensors["out"][:] = out


# =============================================================================
# Tensor-level: tensor.row_min(input) — no tmp (auto-created on lowering).
# =============================================================================


class _TensorRowMinBase(PTOTestCase):
    __test__ = False

    def __init__(self, *, m=16, n=16, dtype=DataType.FP32, config=None):
        super().__init__(config)
        self._m, self._n, self._dtype = m, n, dtype

    def get_name(self) -> str:
        return f"tensor_row_min_{self._m}x{self._n}_{self._dtype.value}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._m, self._n], self._dtype, init_value=lambda: _signed(self._m, self._n)),
            TensorSpec("out", [self._m, 1], self._dtype, is_output=True),
        ]

    def get_program(self) -> Any:
        m, n = self._m, self._n
        dt = _PL_DT[self._dtype]

        @pl.program
        class TensorRowMinProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[m, 1], dt]]
            ) -> pl.Tensor[[m, 1], dt]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.assemble(out, pl.row_min(a), [0, 0])
                return out

        return TensorRowMinProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.min(tensors["a"], dim=1, keepdim=True).values


# Full-shape cases: square / tall / wide.
_SHAPE_CFGS = [
    ("16x16", 16, 16),
    ("32x64", 32, 64),
    ("8x128", 8, 128),
]

# Narrowed valid_shape cases.
_VALID_CFGS = [
    # cols-only narrow: reduce over the first 48 of 64 cols; all 32 rows valid.
    ("32x64_narrow_cols", 32, 64, (32, 48)),
    # combined narrow: reduce over the first 12 of 16 cols; only the first 8 rows valid.
    ("16x16_narrow_both", 16, 16, (8, 12)),
]


class TestTileRowMin:
    """tile.row_min: row-wise min across shapes, dtypes, and narrowed valid_shape."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_row_min(self, test_runner, label, m, n):
        result = test_runner.run(_TileRowMinBase(m=m, n=n))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_row_min_fp16(self, test_runner):
        result = test_runner.run(_TileRowMinBase(m=32, n=64, dtype=DataType.FP16))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _VALID_CFGS, ids=[c[0] for c in _VALID_CFGS])
    def test_tile_row_min_valid(self, test_runner, label, m, n, valid):
        result = test_runner.run(_TileRowMinBase(m=m, n=n, valid_shapes=valid))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_row_min_valid_fp16(self, test_runner):
        result = test_runner.run(_TileRowMinBase(m=16, n=16, valid_shapes=(8, 12), dtype=DataType.FP16))
        assert result.passed, f"Test failed: {result.error}"


class TestTensorRowMin:
    """tensor.row_min: row-wise min lowered via ConvertTensorToTileOps."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tensor_row_min(self, test_runner, label, m, n):
        result = test_runner.run(_TensorRowMinBase(m=m, n=n))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tensor_row_min_fp16(self, test_runner):
        result = test_runner.run(_TensorRowMinBase(m=32, n=64, dtype=DataType.FP16))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
