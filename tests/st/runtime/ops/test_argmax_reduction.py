# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile argmax/argmin reductions.

Covers four tile-level ops (and the tensor-level mirrors of the max variants):
- ``tile.row_argmax`` / ``tile.row_argmin`` -> ``pto.trowargmax`` / ``pto.trowargmin``
- ``tile.col_argmax`` / ``tile.col_argmin`` -> ``pto.tcolargmax`` / ``pto.tcolargmin``

row variants reduce the last axis ([M, N] -> [M, 1]) and return, per row, the
column index of the max/min. col variants reduce axis 0 ([M, N] -> [1, N]) and
return, per column, the row index of the max/min. The index output dtype is
INT32. All four require a tmp scratch tile (unlike col_max/col_min).

Coverage per op: an ``aligned`` (fully valid) case AND a ``valid_shape`` case, in
both FP32 and FP16. The valid_shape case narrows the *reduced* dimension only
(columns for the row ops, rows for the col ops) so the output region stays fully
valid and exactly comparable, while still exercising the partial-reduction /
tail path (e.g. FP32 valid cols 72 > the 64-element repeat -> the tmp scratch is
actually used).

Inputs are a random permutation of distinct integers (``torch.randperm``), so
every row/column has a unique extremum (no tie-break ambiguity vs torch) and the
values are exactly representable in FP16 (all < 2048). The integer index outputs
are compared exactly under the default tolerance.

The DSL parser requires a literal ``pl.tile.<op>`` call, so there is one TestCase
subclass per op (the op name cannot be an alias); the dtype and valid_shape are
closure vars in a ``get_program``-nested ``@pl.program``.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType

_PL_DT = {DataType.FP32: pl.FP32, DataType.FP16: pl.FP16}

M = 16
N = 128


def _distinct(m: int, n: int):
    """No-arg init: a permutation of 0..m*n-1 (distinct -> unique extrema, FP16-exact)."""
    return lambda: torch.randperm(m * n).reshape(m, n).to(torch.float32)


# ---------------------------------------------------------------------------
# Tile-level base + one subclass per op (literal pl.tile.<op> call).
# valid narrows only the reduced dim, so the output stays fully valid.
#   row ops: valid = (M, vc)   -> reduce cols, output [M, 1]
#   col ops: valid = (vr, N)   -> reduce rows, output [1, N]
# ---------------------------------------------------------------------------


class _ArgBase(PTOTestCase):
    __test__ = False
    op_name = ""
    reduce_dim = 1  # 1 = row (reduce cols), 0 = col (reduce rows)
    is_max = True

    def __init__(self, *, valid=None, dtype=DataType.FP32, config=None):
        super().__init__(config)
        self._valid = valid
        self._dtype = dtype

    def get_name(self) -> str:
        v = f"_v{self._valid[0]}x{self._valid[1]}" if self._valid else "_aligned"
        return f"{self.op_name}_{M}x{N}_{self._dtype.value}{v}"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def _out_shape(self) -> list[int]:
        return [M, 1] if self.reduce_dim == 1 else [1, N]

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], self._dtype, init_value=_distinct(M, N)),
            TensorSpec("out", self._out_shape(), DataType.INT32, is_output=True),
        ]

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        vr, vc = self._valid if self._valid else (M, N)
        sub = a[:vr, :vc]
        fn = torch.argmax if self.is_max else torch.argmin
        tensors["out"][:] = fn(sub, dim=self.reduce_dim, keepdim=True).to(torch.int32)


class TileRowArgmax(_ArgBase):
    op_name = "row_argmax"
    reduce_dim = 1
    is_max = True

    def get_program(self) -> Any:
        dt = _PL_DT[self._dtype]
        vshape = list(self._valid) if self._valid else [M, N]

        @pl.program
        class RowArgmaxProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[M, N], dt], out: pl.Out[pl.Tensor[[M, 1], pl.INT32]]
            ) -> pl.Tensor[[M, 1], pl.INT32]:
                t: pl.Tile[[M, N], dt] = pl.load(a, [0, 0], [M, N], valid_shapes=vshape)
                tmp: pl.Tile[[M, N], dt] = pl.tile.create([M, N], dtype=dt, target_memory=pl.MemorySpace.Vec)
                r: pl.Tile[[M, 1], pl.INT32] = pl.tile.row_argmax(t, tmp)
                return pl.store(r, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, N], dt], out: pl.Out[pl.Tensor[[M, 1], pl.INT32]]
            ) -> pl.Tensor[[M, 1], pl.INT32]:
                return self.kernel(a, out)

        return RowArgmaxProgram


class TileRowArgmin(_ArgBase):
    op_name = "row_argmin"
    reduce_dim = 1
    is_max = False

    def get_program(self) -> Any:
        dt = _PL_DT[self._dtype]
        vshape = list(self._valid) if self._valid else [M, N]

        @pl.program
        class RowArgminProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[M, N], dt], out: pl.Out[pl.Tensor[[M, 1], pl.INT32]]
            ) -> pl.Tensor[[M, 1], pl.INT32]:
                t: pl.Tile[[M, N], dt] = pl.load(a, [0, 0], [M, N], valid_shapes=vshape)
                tmp: pl.Tile[[M, N], dt] = pl.tile.create([M, N], dtype=dt, target_memory=pl.MemorySpace.Vec)
                r: pl.Tile[[M, 1], pl.INT32] = pl.tile.row_argmin(t, tmp)
                return pl.store(r, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, N], dt], out: pl.Out[pl.Tensor[[M, 1], pl.INT32]]
            ) -> pl.Tensor[[M, 1], pl.INT32]:
                return self.kernel(a, out)

        return RowArgminProgram


class TileColArgmax(_ArgBase):
    op_name = "col_argmax"
    reduce_dim = 0
    is_max = True

    def get_program(self) -> Any:
        dt = _PL_DT[self._dtype]
        vshape = list(self._valid) if self._valid else [M, N]

        @pl.program
        class ColArgmaxProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[M, N], dt], out: pl.Out[pl.Tensor[[1, N], pl.INT32]]
            ) -> pl.Tensor[[1, N], pl.INT32]:
                t: pl.Tile[[M, N], dt] = pl.load(a, [0, 0], [M, N], valid_shapes=vshape)
                tmp: pl.Tile[[M, N], dt] = pl.tile.create([M, N], dtype=dt, target_memory=pl.MemorySpace.Vec)
                r: pl.Tile[[1, N], pl.INT32] = pl.tile.col_argmax(t, tmp)
                return pl.store(r, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, N], dt], out: pl.Out[pl.Tensor[[1, N], pl.INT32]]
            ) -> pl.Tensor[[1, N], pl.INT32]:
                return self.kernel(a, out)

        return ColArgmaxProgram


class TileColArgmin(_ArgBase):
    op_name = "col_argmin"
    reduce_dim = 0
    is_max = False

    def get_program(self) -> Any:
        dt = _PL_DT[self._dtype]
        vshape = list(self._valid) if self._valid else [M, N]

        @pl.program
        class ColArgminProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[M, N], dt], out: pl.Out[pl.Tensor[[1, N], pl.INT32]]
            ) -> pl.Tensor[[1, N], pl.INT32]:
                t: pl.Tile[[M, N], dt] = pl.load(a, [0, 0], [M, N], valid_shapes=vshape)
                tmp: pl.Tile[[M, N], dt] = pl.tile.create([M, N], dtype=dt, target_memory=pl.MemorySpace.Vec)
                r: pl.Tile[[1, N], pl.INT32] = pl.tile.col_argmin(t, tmp)
                return pl.store(r, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, N], dt], out: pl.Out[pl.Tensor[[1, N], pl.INT32]]
            ) -> pl.Tensor[[1, N], pl.INT32]:
                return self.kernel(a, out)

        return ColArgminProgram


# ---------------------------------------------------------------------------
# Tensor-level (lowered by ConvertTensorToTileOps, which injects the tmp tile).
# Aligned only: DDR tensors cannot express a partial valid region.
# ---------------------------------------------------------------------------


class TensorRowArgmax(_ArgBase):
    op_name = "tensor_row_argmax"
    reduce_dim = 1
    is_max = True

    def get_program(self) -> Any:
        @pl.program
        class TensorRowArgmaxProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[M, N], pl.FP32], out: pl.Out[pl.Tensor[[M, 1], pl.INT32]]
            ) -> pl.Tensor[[M, 1], pl.INT32]:
                r: pl.Tensor[[M, 1], pl.INT32] = pl.row_argmax(a)
                return pl.assemble(out, r, [0, 0])

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, N], pl.FP32], out: pl.Out[pl.Tensor[[M, 1], pl.INT32]]
            ) -> pl.Tensor[[M, 1], pl.INT32]:
                return self.kernel(a, out)

        return TensorRowArgmaxProgram


class TensorColArgmax(_ArgBase):
    op_name = "tensor_col_argmax"
    reduce_dim = 0
    is_max = True

    def get_program(self) -> Any:
        @pl.program
        class TensorColArgmaxProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[M, N], pl.FP32], out: pl.Out[pl.Tensor[[1, N], pl.INT32]]
            ) -> pl.Tensor[[1, N], pl.INT32]:
                r: pl.Tensor[[1, N], pl.INT32] = pl.col_argmax(a)
                return pl.assemble(out, r, [0, 0])

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, N], pl.FP32], out: pl.Out[pl.Tensor[[1, N], pl.INT32]]
            ) -> pl.Tensor[[1, N], pl.INT32]:
                return self.kernel(a, out)

        return TensorColArgmaxProgram


_DTYPES = [DataType.FP32, DataType.FP16]
_ROW_OPS = [TileRowArgmax, TileRowArgmin]
_COL_OPS = [TileColArgmax, TileColArgmin]
# (label, valid) — narrow the reduced dim only so the output stays fully valid.
_ROW_CASES = [("aligned", None), ("valid_cols", (M, 72))]
_COL_CASES = [("aligned", None), ("valid_rows", (10, N))]


class TestTileArgReduce:
    """Tile-level row/col argmax/argmin: aligned + valid_shape, FP32 + FP16."""

    @pytest.mark.parametrize("dtype", _DTYPES, ids=[d.value for d in _DTYPES])
    @pytest.mark.parametrize("label,valid", _ROW_CASES, ids=[c[0] for c in _ROW_CASES])
    @pytest.mark.parametrize("op_cls", _ROW_OPS, ids=[c.op_name for c in _ROW_OPS])
    def test_row(self, test_runner, op_cls, label, valid, dtype):
        result = test_runner.run(op_cls(valid=valid, dtype=dtype))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("dtype", _DTYPES, ids=[d.value for d in _DTYPES])
    @pytest.mark.parametrize("label,valid", _COL_CASES, ids=[c[0] for c in _COL_CASES])
    @pytest.mark.parametrize("op_cls", _COL_OPS, ids=[c.op_name for c in _COL_OPS])
    def test_col(self, test_runner, op_cls, label, valid, dtype):
        result = test_runner.run(op_cls(valid=valid, dtype=dtype))
        assert result.passed, f"Test failed: {result.error}"


class TestTensorArgReduce:
    """Tensor-level pl.row_argmax / pl.col_argmax (lowered via tensor->tile)."""

    def test_tensor_row_argmax(self, test_runner):
        result = test_runner.run(TensorRowArgmax())
        assert result.passed, f"Test failed: {result.error}"

    def test_tensor_col_argmax(self, test_runner):
        result = test_runner.run(TensorColArgmax())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
