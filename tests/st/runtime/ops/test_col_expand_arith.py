# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for the column-broadcast arithmetic ops div / sub:

- ``tile.col_expand_div`` -> ``pto.tcolexpanddiv``   (dst[i,j] = a[i,j] / v[0,j])
- ``tile.col_expand_sub`` -> ``pto.tcolexpandsub``   (dst[i,j] = a[i,j] - v[0,j])

``v`` is a per-column scalar row vector of shape [1, N] broadcast down every
column.  The divisor row vector is kept strictly non-zero and well away from 0
so FP16/FP32 rounding stays bounded.

Each op is exercised on:
- the tile path (``@pl.function(InCore)`` kernel + ``Orchestration``), with both
  a full valid_shape and a narrowed valid_shape (tail / valid handling), and
- the tensor path (``pl.col_expand_*`` whole-Tensor, lowered to the tile op by
  ``ConvertTensorToTileOps``),

across FP32 and FP16.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

M, N = 32, 64

_PL_DT = {DataType.FP32: pl.FP32, DataType.FP16: pl.FP16}

# Square shape plus narrowed valid_shape variants (rows-only / cols-only / both).
# (label, m, n, valid_shape | None)
_SHAPE_CFGS = [
    ("32x64", M, N, None),
    ("32x64_narrow_both", M, N, (20, 48)),
    ("32x64_narrow_rows", M, N, (20, N)),
    ("32x64_narrow_cols", M, N, (M, 48)),
]


def _a(m: int, n: int) -> torch.Tensor:
    """Source covering negatives, zero, and positives."""
    return (torch.arange(m * n, dtype=torch.float32).reshape(m, n).remainder(13) - 6).contiguous()


def _v(n: int) -> torch.Tensor:
    """Strictly non-zero per-column scalar row vector (1.5 .. 5.5)."""
    return (torch.arange(n, dtype=torch.float32).remainder(5) + 1.5).reshape(1, n).contiguous()


# =============================================================================
# Tile-level cases (per-op @pl.program factory; distinct class names)
# =============================================================================


class _TileColExpandBase(PTOTestCase):
    __test__ = False
    op_name = ""

    def __init__(self, *, m=M, n=N, valid_shapes=None, dtype=DataType.FP32, config=None, platform=None):
        super().__init__(config, platform=platform)
        self._m, self._n, self._valid, self._dtype = m, n, valid_shapes, dtype

    def get_name(self) -> str:
        v = f"_v{self._valid[0]}x{self._valid[1]}" if self._valid else ""
        return f"tile_{self.op_name}_{self._m}x{self._n}_{self._dtype.value}{v}"

    def define_tensors(self) -> list[TensorSpec]:
        m, n = self._m, self._n
        return [
            TensorSpec("a", [m, n], self._dtype, init_value=lambda: _a(m, n)),
            TensorSpec("v", [1, n], self._dtype, init_value=lambda: _v(n)),
            TensorSpec("out", [m, n], self._dtype, is_output=True),
        ]

    def _ref(self, a, v):
        raise NotImplementedError

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a, v = tensors["a"], tensors["v"]
        res = torch.zeros_like(a)
        if self._valid:
            vm, vn = self._valid
            res[:vm, :vn] = self._ref(a[:vm, :vn], v[:, :vn])
        else:
            res = self._ref(a, v)
        tensors["out"][:] = res


class TileColExpandDivCase(_TileColExpandBase):
    op_name = "col_expand_div"

    def _ref(self, a, v):
        return a / v

    def get_program(self) -> Any:
        m, n = self._m, self._n
        vshape = list(self._valid) if self._valid else [m, n]
        # The column vector valid columns must match the destination valid columns.
        v_cols = vshape[1]
        dt = _PL_DT[self._dtype]

        @pl.program
        class ColExpandDivProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[m, n], dt],
                v: pl.Tensor[[1, n], dt],
                out: pl.Out[pl.Tensor[[m, n], dt]],
            ) -> pl.Tensor[[m, n], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                v_tile = pl.load(v, [0, 0], [1, n], valid_shapes=[1, v_cols])
                out_tile = pl.tile.col_expand_div(a_tile, v_tile)
                return pl.store(out_tile, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[m, n], dt],
                v: pl.Tensor[[1, n], dt],
                out: pl.Out[pl.Tensor[[m, n], dt]],
            ) -> pl.Tensor[[m, n], dt]:
                out = self.kernel(a, v, out)
                return out

        return ColExpandDivProgram


class TileColExpandSubCase(_TileColExpandBase):
    op_name = "col_expand_sub"

    def _ref(self, a, v):
        return a - v

    def get_program(self) -> Any:
        m, n = self._m, self._n
        vshape = list(self._valid) if self._valid else [m, n]
        v_cols = vshape[1]
        dt = _PL_DT[self._dtype]

        @pl.program
        class ColExpandSubProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[m, n], dt],
                v: pl.Tensor[[1, n], dt],
                out: pl.Out[pl.Tensor[[m, n], dt]],
            ) -> pl.Tensor[[m, n], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                v_tile = pl.load(v, [0, 0], [1, n], valid_shapes=[1, v_cols])
                out_tile = pl.tile.col_expand_sub(a_tile, v_tile)
                return pl.store(out_tile, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[m, n], dt],
                v: pl.Tensor[[1, n], dt],
                out: pl.Out[pl.Tensor[[m, n], dt]],
            ) -> pl.Tensor[[m, n], dt]:
                out = self.kernel(a, v, out)
                return out

        return ColExpandSubProgram


# FP16 needs a relaxed tolerance: FP16 eps ~= 9.8e-4, so a 1e-5 bar is below the
# representable precision. 2e-3 (~2 FP16 ULP) is the right bar for div/sub.
_FP16_CFG = RunConfig(rtol=2e-3, atol=2e-3)


class TestTileColExpandArith:
    """tile.col_expand_div / col_expand_sub on a2a3 (FP32 + FP16, full + narrowed valid)."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("label,m,n,valid", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_div(self, test_runner, platform, label, m, n, valid):
        result = test_runner.run(TileColExpandDivCase(m=m, n=n, valid_shapes=valid, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("label,m,n,valid", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_sub(self, test_runner, platform, label, m, n, valid):
        result = test_runner.run(TileColExpandSubCase(m=m, n=n, valid_shapes=valid, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tile_div_fp16(self, test_runner, platform):
        result = test_runner.run(
            TileColExpandDivCase(dtype=DataType.FP16, config=_FP16_CFG, platform=platform)
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tile_sub_fp16(self, test_runner, platform):
        result = test_runner.run(
            TileColExpandSubCase(dtype=DataType.FP16, config=_FP16_CFG, platform=platform)
        )
        assert result.passed, f"Test failed: {result.error}"


# =============================================================================
# Tensor-level cases — pl.col_expand_div / col_expand_sub on whole Tensors,
# lowered to the tile ops by ConvertTensorToTileOps.
# =============================================================================


@pl.program
class TensorColExpandDivProg:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[1, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        result: pl.Tensor[[M, N], pl.FP32] = pl.col_expand_div(a, v)
        return pl.assemble(output, result, [0, 0])

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[1, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        output = self.kernel(a, v, output)
        return output


@pl.program
class TensorColExpandSubProg:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[1, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        result: pl.Tensor[[M, N], pl.FP32] = pl.col_expand_sub(a, v)
        return pl.assemble(output, result, [0, 0])

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        v: pl.Tensor[[1, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        output = self.kernel(a, v, output)
        return output


class _TensorColExpandBase(PTOTestCase):
    __test__ = False
    op_name = ""
    program: Any = None

    def __init__(self, *, platform=None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"tensor_{self.op_name}_fp32"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.FP32, init_value=lambda: _a(M, N)),
            TensorSpec("v", [1, N], DataType.FP32, init_value=lambda: _v(N)),
            TensorSpec("output", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return self.program

    def _ref(self, a, v):
        raise NotImplementedError

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["output"][:] = self._ref(tensors["a"], tensors["v"])


class TensorColExpandDivCase(_TensorColExpandBase):
    op_name = "col_expand_div"
    program = TensorColExpandDivProg

    def _ref(self, a, v):
        return a / v


class TensorColExpandSubCase(_TensorColExpandBase):
    op_name = "col_expand_sub"
    program = TensorColExpandSubProg

    def _ref(self, a, v):
        return a - v


class TestTensorColExpandArith:
    """Tensor-level pl.col_expand_div / col_expand_sub (lowered via tensor->tile)."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tensor_div(self, test_runner, platform):
        result = test_runner.run(TensorColExpandDivCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tensor_sub(self, test_runner, platform):
        result = test_runner.run(TensorColExpandSubCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
