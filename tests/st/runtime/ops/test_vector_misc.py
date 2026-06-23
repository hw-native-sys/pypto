# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for assorted tile-level vector ops:

muls            tile * scalar
divs            tile / scalar
col_expand_add  tile + broadcast(col_vec[1, N]) over rows

Coverage per op: multiple shapes (square/tall/wide), aligned + narrow valid_shape
(combined / rows-only / cols-only), FP16, non-zero store offset; muls/divs sweep
the scalar operand.

Scope is a2a3 only (``@pytest.mark.platforms("a2a3")``); a5 coverage is a
separate PR.

(expands and sum are omitted: no codegen registered — KNOWN_ISSUES.)
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

_PL_DT = {DataType.FP32: pl.FP32, DataType.FP16: pl.FP16}

_SHAPE_CFGS = [
    ("16x16", 16, 16, None),
    ("32x64", 32, 64, None),
    ("8x128", 8, 128, None),
    ("16x16_narrow_both", 16, 16, (8, 12)),
    ("16x16_narrow_rows", 16, 16, (8, 16)),
    ("16x16_narrow_cols", 16, 16, (16, 8)),
]


def _randn(m, n):
    return torch.randn(m, n, dtype=torch.float32)


class _ScalarOpBase(PTOTestCase):
    """tile (op) scalar, with shape / valid / dtype / scalar / offset config."""

    __test__ = False
    op_name = ""

    def __init__(
        self,
        *,
        m=16,
        n=16,
        valid_shapes=None,
        dtype=DataType.FP32,
        rhs=2.0,
        out_m=None,
        out_n=None,
        off=(0, 0),
        config=None,
    ):
        super().__init__(config)
        self._m, self._n, self._valid, self._dtype = m, n, valid_shapes, dtype
        self._rhs, self._out_m, self._out_n, self._off = rhs, out_m or m, out_n or n, off

    def get_name(self) -> str:
        v = f"_v{self._valid[0]}x{self._valid[1]}" if self._valid else ""
        o = f"_off{self._off[0]}x{self._off[1]}" if self._off != (0, 0) else ""
        return f"tile_{self.op_name}_{self._m}x{self._n}_{self._dtype.value}_r{self._rhs}{v}{o}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._m, self._n], self._dtype, init_value=lambda: _randn(self._m, self._n)),
            TensorSpec("out", [self._out_m, self._out_n], self._dtype, is_output=True),
        ]

    def _ref(self, a):
        raise NotImplementedError

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        out = torch.zeros_like(tensors["out"])
        r, c = self._off
        if self._valid:
            vm, vn = self._valid
            res = torch.zeros_like(a)
            res[:vm, :vn] = self._ref(a[:vm, :vn])
        else:
            res = self._ref(a)
        out[r : r + self._m, c : c + self._n] = res
        tensors["out"][:] = out


class TileMulsTestCase(_ScalarOpBase):
    op_name = "muls"

    def _ref(self, a):
        return a * self._rhs

    def get_program(self) -> Any:
        m, n, om, on, off = self._m, self._n, self._out_m, self._out_n, list(self._off)
        vshape = list(self._valid) if self._valid else [m, n]
        dt, rhs = _PL_DT[self._dtype], self._rhs

        @pl.program
        class MulsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                out = pl.store(pl.tile.muls(a_tile, rhs), off, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                out = self.kernel(a, out)
                return out

        return MulsProgram


class TileDivsTestCase(_ScalarOpBase):
    op_name = "divs"

    def _ref(self, a):
        return a / self._rhs

    def get_program(self) -> Any:
        m, n, om, on, off = self._m, self._n, self._out_m, self._out_n, list(self._off)
        vshape = list(self._valid) if self._valid else [m, n]
        dt, rhs = _PL_DT[self._dtype], self._rhs

        @pl.program
        class DivsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                out = pl.store(pl.tile.divs(a_tile, rhs), off, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                out = self.kernel(a, out)
                return out

        return DivsProgram


class ColExpandAddTestCase(PTOTestCase):
    """tile[m,n] + broadcast(col_vec[1,n]) over rows, with shape/valid/dtype/offset config."""

    __test__ = False

    def __init__(
        self,
        *,
        m=16,
        n=16,
        valid_shapes=None,
        dtype=DataType.FP32,
        out_m=None,
        out_n=None,
        off=(0, 0),
        config=None,
    ):
        super().__init__(config)
        self._m, self._n, self._valid, self._dtype = m, n, valid_shapes, dtype
        self._out_m, self._out_n, self._off = out_m or m, out_n or n, off

    def get_name(self) -> str:
        v = f"_v{self._valid[0]}x{self._valid[1]}" if self._valid else ""
        o = f"_off{self._off[0]}x{self._off[1]}" if self._off != (0, 0) else ""
        return f"tile_col_expand_add_{self._m}x{self._n}_{self._dtype.value}{v}{o}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._m, self._n], self._dtype, init_value=lambda: _randn(self._m, self._n)),
            TensorSpec("col_vec", [1, self._n], self._dtype, init_value=lambda: _randn(1, self._n)),
            TensorSpec("out", [self._out_m, self._out_n], self._dtype, is_output=True),
        ]

    def get_program(self) -> Any:
        m, n, om, on, off = self._m, self._n, self._out_m, self._out_n, list(self._off)
        vshape = list(self._valid) if self._valid else [m, n]
        col_vshape = [1, self._valid[1]] if self._valid else [1, n]
        dt = _PL_DT[self._dtype]

        @pl.program
        class ColExpandAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[m, n], dt],
                col_vec: pl.Tensor[[1, n], dt],
                out: pl.Out[pl.Tensor[[om, on], dt]],
            ) -> pl.Tensor[[om, on], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                col_tile = pl.load(col_vec, [0, 0], [1, n], valid_shapes=col_vshape)
                out = pl.store(pl.tile.col_expand_add(a_tile, col_tile), off, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[m, n], dt],
                col_vec: pl.Tensor[[1, n], dt],
                out: pl.Out[pl.Tensor[[om, on], dt]],
            ) -> pl.Tensor[[om, on], dt]:
                out = self.kernel(a, col_vec, out)
                return out

        return ColExpandAddProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a, col = tensors["a"], tensors["col_vec"]
        out = torch.zeros_like(tensors["out"])
        r, c = self._off
        if self._valid:
            vm, vn = self._valid
            res = torch.zeros_like(a)
            res[:vm, :vn] = a[:vm, :vn] + col[:, :vn]
        else:
            res = a + col
        out[r : r + self._m, c : c + self._n] = res
        tensors["out"][:] = out


_MULS_RHS = [2.5, -2.5, 0.0, 1.0]
_DIVS_RHS = [2.0, 3.0, -2.0, 0.5]


class TestVectorMisc:
    """Tile-level muls/divs/col_expand_add on a2a3 across shapes, valid, dtype, scalar, offset."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_muls(self, test_runner, label, m, n, valid):
        result = test_runner.run(TileMulsTestCase(m=m, n=n, valid_shapes=valid, rhs=2.5))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_divs(self, test_runner, label, m, n, valid):
        result = test_runner.run(TileDivsTestCase(m=m, n=n, valid_shapes=valid, rhs=2.0))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("rhs", _MULS_RHS, ids=[f"r{r}" for r in _MULS_RHS])
    def test_tile_muls_scalars(self, test_runner, rhs):
        result = test_runner.run(TileMulsTestCase(rhs=rhs))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("rhs", _DIVS_RHS, ids=[f"r{r}" for r in _DIVS_RHS])
    def test_tile_divs_scalars(self, test_runner, rhs):
        result = test_runner.run(TileDivsTestCase(rhs=rhs))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_muls_fp16(self, test_runner):
        result = test_runner.run(TileMulsTestCase(m=32, n=64, dtype=DataType.FP16, rhs=2.5))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_divs_fp16(self, test_runner):
        result = test_runner.run(TileDivsTestCase(m=32, n=64, dtype=DataType.FP16, rhs=2.0))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_muls_offset(self, test_runner):
        result = test_runner.run(TileMulsTestCase(out_m=32, out_n=32, off=(16, 16), rhs=2.5))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_col_expand_add(self, test_runner, label, m, n, valid):
        result = test_runner.run(ColExpandAddTestCase(m=m, n=n, valid_shapes=valid))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_col_expand_add_fp16(self, test_runner):
        result = test_runner.run(ColExpandAddTestCase(m=32, n=64, dtype=DataType.FP16))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_col_expand_add_offset(self, test_runner):
        result = test_runner.run(ColExpandAddTestCase(out_m=32, out_n=32, off=(16, 16)))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
