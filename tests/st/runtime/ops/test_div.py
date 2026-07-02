# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for element-wise division ops.

Tile-level:
- ``tile.div``  (tile vs tile)   -> ``pto.tdiv``
- ``tile.divs`` (tile vs scalar) -> ``pto.tdivs`` (``tile.div`` with a scalar rhs auto-dispatches)

Tensor-level (lowered by ConvertTensorToTileOps):
- ``tensor.div`` (tensor vs tensor) and ``tensor.div`` (tensor vs scalar).

The result matches ``torch.div``. The divisor is kept strictly non-zero in every
case so the golden numerics are well-defined. Coverage: multiple shapes
(square/tall/wide), aligned + narrow valid_shape (combined / rows-only /
cols-only), scalar sweep incl. a negative, and FP32 + FP16.

Scope is a2a3 only (``@pytest.mark.platforms("a2a3")``); a5 coverage is a
separate PR.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

_PL_DT = {DataType.FP32: pl.FP32, DataType.FP16: pl.FP16}

_SHAPE_CFGS = [
    ("16x16", 16, 16, None),
    ("32x64", 32, 64, None),
    ("8x128", 8, 128, None),
    ("16x16_narrow_both", 16, 16, (8, 12)),
    ("16x16_narrow_rows", 16, 16, (8, 16)),
    ("16x16_narrow_cols", 16, 16, (16, 8)),
]

# Tensor-level ops compute over the whole tensor; there is no per-tile
# valid_shape to narrow, so tensor tests use the full-shape configs only.
_FULL_SHAPE_CFGS = [c for c in _SHAPE_CFGS if c[3] is None]

# Non-zero scalar divisors covering negatives and positives.
_SCALARS = [2.5, -3.0, 4.0]


def _lhs(m: int, n: int) -> torch.Tensor:
    """Dividend covering negatives, zero, and positives."""
    return (torch.arange(m * n, dtype=torch.float32).reshape(m, n).remainder(13) - 6).contiguous()


def _rhs(m: int, n: int) -> torch.Tensor:
    """Strictly non-zero divisor (1.5 .. 5.5)."""
    return (torch.arange(m * n, dtype=torch.float32).reshape(m, n).remainder(5) + 1.5).contiguous()


class _DivBase(PTOTestCase):
    """Shared scaffolding: a 2D dividend ``a``, an optional divisor ``b`` (tile/tile)
    or scalar (tile/scalar), and a valid_shape-aware golden via ``torch.div``."""

    __test__ = False
    op_name = ""

    def __init__(
        self,
        *,
        m: int = 16,
        n: int = 16,
        valid_shapes=None,
        dtype: DataType = DataType.FP32,
        scalar: float | None = None,
        config=None,
    ):
        super().__init__(config)
        self._m, self._n, self._valid, self._dtype, self._scalar = m, n, valid_shapes, dtype, scalar

    def get_name(self) -> str:
        v = f"_v{self._valid[0]}x{self._valid[1]}" if self._valid else ""
        s = f"_s{self._scalar}" if self._scalar is not None else ""
        return f"{self.op_name}_{self._m}x{self._n}_{self._dtype.value}{v}{s}"

    def define_tensors(self) -> list[TensorSpec]:
        m, n = self._m, self._n
        specs = [TensorSpec("a", [m, n], self._dtype, init_value=lambda: _lhs(m, n))]
        if self._scalar is None:
            specs.append(TensorSpec("b", [m, n], self._dtype, init_value=lambda: _rhs(m, n)))
        specs.append(TensorSpec("out", [m, n], self._dtype, is_output=True))
        return specs

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        out = torch.zeros_like(tensors["out"])
        if self._scalar is None:
            b = tensors["b"]
            if self._valid:
                vm, vn = self._valid
                res = torch.zeros_like(a)
                res[:vm, :vn] = torch.div(a[:vm, :vn], b[:vm, :vn])
            else:
                res = torch.div(a, b)
        else:
            scalar = torch.tensor(self._scalar, dtype=a.dtype)
            if self._valid:
                vm, vn = self._valid
                res = torch.zeros_like(a)
                res[:vm, :vn] = torch.div(a[:vm, :vn], scalar)
            else:
                res = torch.div(a, scalar)
        out[: self._m, : self._n] = res
        tensors["out"][:] = out


class TileDivTestCase(_DivBase):
    """tile.div (tile/tile) and tile.div(tile, scalar) -> tile.divs (auto-dispatch)."""

    op_name = "tile_div"

    def get_program(self) -> Any:
        m, n = self._m, self._n
        vshape = list(self._valid) if self._valid else [m, n]
        dt = _PL_DT[self._dtype]
        scalar = self._scalar

        if scalar is None:

            @pl.program
            class TileDivProgram:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    a: pl.Tensor[[m, n], dt],
                    b: pl.Tensor[[m, n], dt],
                    out: pl.Out[pl.Tensor[[m, n], dt]],
                ) -> pl.Tensor[[m, n], dt]:
                    a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                    b_tile = pl.load(b, [0, 0], [m, n], valid_shapes=vshape)
                    out = pl.store(pl.tile.div(a_tile, b_tile), [0, 0], out)
                    return out

                @pl.function(type=pl.FunctionType.Orchestration)
                def orchestrator(
                    self,
                    a: pl.Tensor[[m, n], dt],
                    b: pl.Tensor[[m, n], dt],
                    out: pl.Out[pl.Tensor[[m, n], dt]],
                ) -> pl.Tensor[[m, n], dt]:
                    out = self.kernel(a, b, out)
                    return out

            return TileDivProgram

        @pl.program
        class TileDivsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[m, n], dt],
                out: pl.Out[pl.Tensor[[m, n], dt]],
            ) -> pl.Tensor[[m, n], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                # Scalar rhs auto-dispatches tile.div -> tile.divs.
                out = pl.store(pl.tile.div(a_tile, scalar), [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[m, n], dt],
                out: pl.Out[pl.Tensor[[m, n], dt]],
            ) -> pl.Tensor[[m, n], dt]:
                out = self.kernel(a, out)
                return out

        return TileDivsProgram


class TensorDivTestCase(_DivBase):
    """tensor.div (tensor/tensor) and tensor.div(tensor, scalar); lowers to tile.div / tile.divs."""

    op_name = "tensor_div"

    def get_program(self) -> Any:
        m, n = self._m, self._n
        dt = _PL_DT[self._dtype]
        scalar = self._scalar

        if scalar is None:

            @pl.program
            class TensorDivProgram:
                @pl.function(type=pl.FunctionType.Opaque)
                def main(
                    self,
                    a: pl.Tensor[[m, n], dt],
                    b: pl.Tensor[[m, n], dt],
                    out: pl.Out[pl.Tensor[[m, n], dt]],
                ) -> pl.Tensor[[m, n], dt]:
                    with pl.at(level=pl.Level.CORE_GROUP):
                        out = pl.assemble(out, pl.div(a, b), [0, 0])
                    return out

            return TensorDivProgram

        @pl.program
        class TensorDivsProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                a: pl.Tensor[[m, n], dt],
                out: pl.Out[pl.Tensor[[m, n], dt]],
            ) -> pl.Tensor[[m, n], dt]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.assemble(out, pl.div(a, scalar), [0, 0])
                return out

        return TensorDivsProgram


# FP16 division needs relaxed tolerance, not a workaround: FP16 machine epsilon
# is 2**-10 ~= 9.8e-4, so a 1e-5 comparison is below FP16's representable
# precision. division rounds the quotient to ~1 ULP, so 2e-3 (~2 FP16 ULP) is
# the right bar.
_FP16_CFG = RunConfig(rtol=2e-3, atol=2e-3)


class TestTileDiv:
    """Tile-level div/divs on a2a3."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_div(self, test_runner, label, m, n, valid):
        result = test_runner.run(TileDivTestCase(m=m, n=n, valid_shapes=valid))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("scalar", _SCALARS, ids=[f"s{s}" for s in _SCALARS])
    def test_tile_divs(self, test_runner, scalar):
        result = test_runner.run(TileDivTestCase(scalar=scalar))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_divs_narrow(self, test_runner):
        result = test_runner.run(TileDivTestCase(scalar=-3.0, valid_shapes=(8, 12)))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_div_fp16(self, test_runner):
        result = test_runner.run(TileDivTestCase(dtype=DataType.FP16, config=_FP16_CFG))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_divs_fp16(self, test_runner):
        result = test_runner.run(TileDivTestCase(scalar=2.5, dtype=DataType.FP16, config=_FP16_CFG))
        assert result.passed, f"Test failed: {result.error}"


class TestTensorDiv:
    """Tensor-level div/divs (lowered by ConvertTensorToTileOps) on a2a3."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _FULL_SHAPE_CFGS, ids=[c[0] for c in _FULL_SHAPE_CFGS])
    def test_tensor_div(self, test_runner, label, m, n, valid):
        result = test_runner.run(TensorDivTestCase(m=m, n=n, valid_shapes=valid))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("scalar", _SCALARS, ids=[f"s{s}" for s in _SCALARS])
    def test_tensor_divs(self, test_runner, scalar):
        result = test_runner.run(TensorDivTestCase(scalar=scalar))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tensor_div_fp16(self, test_runner):
        result = test_runner.run(TensorDivTestCase(dtype=DataType.FP16, config=_FP16_CFG))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tensor_divs_fp16(self, test_runner):
        result = test_runner.run(TensorDivTestCase(scalar=-3.0, dtype=DataType.FP16, config=_FP16_CFG))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
