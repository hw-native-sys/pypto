# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile argmax/argmin reductions.

Covers four tile-level ops (plus tensor-level mirrors of the max variants):
- ``tile.row_argmax`` / ``tile.row_argmin`` -> ``pto.trowargmax`` / ``pto.trowargmin``
- ``tile.col_argmax`` / ``tile.col_argmin`` -> ``pto.tcolargmax`` / ``pto.tcolargmin``

row variants reduce the last axis ([M, N] -> [M, 1]) and return, per row, the
column index of the max/min. col variants reduce axis 0 ([M, N] -> [1, N]) and
return, per column, the row index of the max/min. The index output dtype is
INT32. All four require a tmp scratch tile (unlike col_max/col_min). Per the
pto-isa contract the source dtype is half or float only, so coverage is FP32+FP16.

Full-scenario coverage per op (shared config matrix, both FP32 and FP16):
- shapes: 16x16 (square), 32x64, 8x128 (wide) — varies the reduced and the
  kept extent, exercising the single-repeat and multi-repeat reduction paths;
- valid_shape: aligned, narrow rows, narrow cols, narrow both, and a wide
  narrow-cols case (8x128 valid cols 72 > the FP32 64-element repeat, which
  forces the tmp scratch into use). Narrowing the *reduced* dim exercises the
  partial-reduction tail; narrowing the *kept* dim shrinks the valid output
  region (the untouched output elements stay zero, matching the golden).

Inputs are a random permutation of distinct integers (``torch.randperm``), so
every row/column has a unique extremum (no tie-break ambiguity vs torch) and the
values are exactly representable in FP16 (all < 2048). The integer index outputs
are compared exactly under the default tolerance.

The DSL parser requires a literal ``pl.tile.<op>`` call, so there is one TestCase
subclass per op; dtype, shape, and valid_shape are closure vars in a
``get_program``-nested ``@pl.program``.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType

_PL_DT = {DataType.FP32: pl.FP32, DataType.FP16: pl.FP16}
_DTYPES = [DataType.FP32, DataType.FP16]

# (label, m, n, valid=(vr, vc) or None) — shared across all four ops.
_CFGS = [
    ("16x16", 16, 16, None),
    ("32x64", 32, 64, None),
    ("8x128", 8, 128, None),
    ("16x16_nrows", 16, 16, (8, 16)),
    ("16x16_ncols", 16, 16, (16, 8)),
    ("16x16_nboth", 16, 16, (8, 12)),
    ("8x128_ncols72", 8, 128, (8, 72)),
]


def _distinct(m: int, n: int):
    """No-arg init: a permutation of 0..m*n-1 (distinct -> unique extrema, FP16-exact)."""
    return lambda: torch.randperm(m * n).reshape(m, n).to(torch.float32)


class _ArgBase(PTOTestCase):
    __test__ = False
    op_name = ""
    reduce_dim = 1  # 1 = row (reduce cols -> [m, 1]); 0 = col (reduce rows -> [1, n])
    is_max = True

    def __init__(self, *, m=16, n=16, valid=None, dtype=DataType.FP32, config=None):
        super().__init__(config)
        self._m, self._n, self._valid, self._dtype = m, n, valid, dtype

    def get_name(self) -> str:
        v = f"_v{self._valid[0]}x{self._valid[1]}" if self._valid else "_aligned"
        return f"{self.op_name}_{self._m}x{self._n}_{self._dtype.value}{v}"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def _out_shape(self) -> list[int]:
        return [self._m, 1] if self.reduce_dim == 1 else [1, self._n]

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._m, self._n], self._dtype, init_value=_distinct(self._m, self._n)),
            TensorSpec("out", self._out_shape(), DataType.INT32, is_output=True, init_value=torch.zeros),
        ]

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        m, n = self._m, self._n
        vr, vc = self._valid if self._valid else (m, n)
        fn = torch.argmax if self.is_max else torch.argmin
        if self.reduce_dim == 1:
            out = torch.zeros((m, 1), dtype=torch.int32)
            out[:vr, 0] = fn(a[:vr, :vc], dim=1).to(torch.int32)
        else:
            out = torch.zeros((1, n), dtype=torch.int32)
            out[0, :vc] = fn(a[:vr, :vc], dim=0).to(torch.int32)
        tensors["out"][:] = out


# One subclass per op. The DSL parser requires a literal ``pl.tile.<op>`` call
# (it cannot resolve an aliased/closure op), so each get_program inlines its own
# @pl.program; the shape/dtype/valid_shape are closure vars.


class TileRowArgmax(_ArgBase):
    op_name = "row_argmax"
    reduce_dim = 1
    is_max = True

    def get_program(self) -> Any:
        m, n = self._m, self._n
        dt = _PL_DT[self._dtype]
        vshape = list(self._valid) if self._valid else [m, n]

        @pl.program
        class RowArgmaxProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.InOut[pl.Tensor[[m, 1], pl.INT32]]
            ) -> pl.Tensor[[m, 1], pl.INT32]:
                t: pl.Tile[[m, n], dt] = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                tmp: pl.Tile[[m, n], dt] = pl.tile.create([m, n], dtype=dt, target_memory=pl.MemorySpace.Vec)
                r: pl.Tile[[m, 1], pl.INT32] = pl.tile.row_argmax(t, tmp)
                return pl.store(r, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.InOut[pl.Tensor[[m, 1], pl.INT32]]
            ) -> pl.Tensor[[m, 1], pl.INT32]:
                return self.kernel(a, out)

        return RowArgmaxProgram


class TileRowArgmin(_ArgBase):
    op_name = "row_argmin"
    reduce_dim = 1
    is_max = False

    def get_program(self) -> Any:
        m, n = self._m, self._n
        dt = _PL_DT[self._dtype]
        vshape = list(self._valid) if self._valid else [m, n]

        @pl.program
        class RowArgminProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.InOut[pl.Tensor[[m, 1], pl.INT32]]
            ) -> pl.Tensor[[m, 1], pl.INT32]:
                t: pl.Tile[[m, n], dt] = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                tmp: pl.Tile[[m, n], dt] = pl.tile.create([m, n], dtype=dt, target_memory=pl.MemorySpace.Vec)
                r: pl.Tile[[m, 1], pl.INT32] = pl.tile.row_argmin(t, tmp)
                return pl.store(r, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.InOut[pl.Tensor[[m, 1], pl.INT32]]
            ) -> pl.Tensor[[m, 1], pl.INT32]:
                return self.kernel(a, out)

        return RowArgminProgram


class TileColArgmax(_ArgBase):
    op_name = "col_argmax"
    reduce_dim = 0
    is_max = True

    def get_program(self) -> Any:
        m, n = self._m, self._n
        dt = _PL_DT[self._dtype]
        vshape = list(self._valid) if self._valid else [m, n]

        @pl.program
        class ColArgmaxProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.InOut[pl.Tensor[[1, n], pl.INT32]]
            ) -> pl.Tensor[[1, n], pl.INT32]:
                t: pl.Tile[[m, n], dt] = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                tmp: pl.Tile[[m, n], dt] = pl.tile.create([m, n], dtype=dt, target_memory=pl.MemorySpace.Vec)
                r: pl.Tile[[1, n], pl.INT32] = pl.tile.col_argmax(t, tmp)
                return pl.store(r, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.InOut[pl.Tensor[[1, n], pl.INT32]]
            ) -> pl.Tensor[[1, n], pl.INT32]:
                return self.kernel(a, out)

        return ColArgmaxProgram


class TileColArgmin(_ArgBase):
    op_name = "col_argmin"
    reduce_dim = 0
    is_max = False

    def get_program(self) -> Any:
        m, n = self._m, self._n
        dt = _PL_DT[self._dtype]
        vshape = list(self._valid) if self._valid else [m, n]

        @pl.program
        class ColArgminProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.InOut[pl.Tensor[[1, n], pl.INT32]]
            ) -> pl.Tensor[[1, n], pl.INT32]:
                t: pl.Tile[[m, n], dt] = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                tmp: pl.Tile[[m, n], dt] = pl.tile.create([m, n], dtype=dt, target_memory=pl.MemorySpace.Vec)
                r: pl.Tile[[1, n], pl.INT32] = pl.tile.col_argmin(t, tmp)
                return pl.store(r, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.InOut[pl.Tensor[[1, n], pl.INT32]]
            ) -> pl.Tensor[[1, n], pl.INT32]:
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

    def __init__(self, **kw):
        super().__init__(m=32, n=64, **kw)

    def get_program(self) -> Any:
        @pl.program
        class TensorRowArgmaxProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[32, 64], pl.FP32], out: pl.InOut[pl.Tensor[[32, 1], pl.INT32]]
            ) -> pl.Tensor[[32, 1], pl.INT32]:
                r: pl.Tensor[[32, 1], pl.INT32] = pl.row_argmax(a)
                return pl.assemble(out, r, [0, 0])

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 64], pl.FP32], out: pl.InOut[pl.Tensor[[32, 1], pl.INT32]]
            ) -> pl.Tensor[[32, 1], pl.INT32]:
                return self.kernel(a, out)

        return TensorRowArgmaxProgram


class TensorColArgmax(_ArgBase):
    op_name = "tensor_col_argmax"
    reduce_dim = 0
    is_max = True

    def __init__(self, **kw):
        super().__init__(m=32, n=64, **kw)

    def get_program(self) -> Any:
        @pl.program
        class TensorColArgmaxProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[32, 64], pl.FP32], out: pl.InOut[pl.Tensor[[1, 64], pl.INT32]]
            ) -> pl.Tensor[[1, 64], pl.INT32]:
                r: pl.Tensor[[1, 64], pl.INT32] = pl.col_argmax(a)
                return pl.assemble(out, r, [0, 0])

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 64], pl.FP32], out: pl.InOut[pl.Tensor[[1, 64], pl.INT32]]
            ) -> pl.Tensor[[1, 64], pl.INT32]:
                return self.kernel(a, out)

        return TensorColArgmaxProgram


_ROW_OPS = [TileRowArgmax, TileRowArgmin]
_COL_OPS = [TileColArgmax, TileColArgmin]


class TestTileArgReduce:
    """Tile-level row/col argmax/argmin: full shape x valid_shape x dtype matrix."""

    @pytest.mark.parametrize("dtype", _DTYPES, ids=[d.value for d in _DTYPES])
    @pytest.mark.parametrize("label,m,n,valid", _CFGS, ids=[c[0] for c in _CFGS])
    @pytest.mark.parametrize("op_cls", _ROW_OPS, ids=[c.op_name for c in _ROW_OPS])
    def test_row(self, test_runner, op_cls, label, m, n, valid, dtype):
        result = test_runner.run(op_cls(m=m, n=n, valid=valid, dtype=dtype))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("dtype", _DTYPES, ids=[d.value for d in _DTYPES])
    @pytest.mark.parametrize("label,m,n,valid", _CFGS, ids=[c[0] for c in _CFGS])
    @pytest.mark.parametrize("op_cls", _COL_OPS, ids=[c.op_name for c in _COL_OPS])
    def test_col(self, test_runner, op_cls, label, m, n, valid, dtype):
        result = test_runner.run(op_cls(m=m, n=n, valid=valid, dtype=dtype))
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
