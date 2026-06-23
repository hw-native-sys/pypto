# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile-level activation ops: relu, lrelu.

Both use signed inputs to exercise the negative branch. lrelu computes
``max(t, slope*t)`` with a scalar slope (swept over identity / relu-equivalent /
negative / >1 values).

Coverage per op: multiple shapes (square/tall/wide), aligned + narrow valid_shape
(combined / rows-only / cols-only), FP16, non-zero store offset; lrelu also sweeps
the slope arg.

Scope is a2a3 only (``@pytest.mark.platforms("a2a3")``); a5 coverage is a
separate PR.

(prelu is omitted: its 3-arg DSL form mismatches codegen pto.tprelu — KNOWN_ISSUES.)
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


def _signed(m, n):
    return torch.randn(m, n, dtype=torch.float32)


class _ActBase(PTOTestCase):
    __test__ = False
    op_name = ""

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
        return f"tile_{self.op_name}_{self._m}x{self._n}_{self._dtype.value}{v}{o}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._m, self._n], self._dtype, init_value=lambda: _signed(self._m, self._n)),
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


class TileReluTestCase(_ActBase):
    op_name = "relu"

    def _ref(self, a):
        return torch.relu(a)

    def get_program(self) -> Any:
        m, n, om, on, off = self._m, self._n, self._out_m, self._out_n, list(self._off)
        vshape = list(self._valid) if self._valid else [m, n]
        dt = _PL_DT[self._dtype]

        @pl.program
        class ReluProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                out = pl.store(pl.tile.relu(a_tile), off, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                out = self.kernel(a, out)
                return out

        return ReluProgram


class TileLreluTestCase(_ActBase):
    op_name = "lrelu"

    def __init__(self, *, slope=0.1, **kw):
        super().__init__(**kw)
        self._slope = slope

    def get_name(self) -> str:
        return super().get_name() + f"_s{self._slope}"

    def _ref(self, a):
        return torch.maximum(a, self._slope * a)

    def get_program(self) -> Any:
        m, n, om, on, off = self._m, self._n, self._out_m, self._out_n, list(self._off)
        vshape = list(self._valid) if self._valid else [m, n]
        dt = _PL_DT[self._dtype]
        slope = self._slope

        @pl.program
        class LreluProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                out = pl.store(pl.tile.lrelu(a_tile, slope), off, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                out = self.kernel(a, out)
                return out

        return LreluProgram


_LRELU_SLOPES = [0.0, 0.1, 1.0, -0.5, 2.0]


class TestActivation:
    """Tile-level relu/lrelu on a2a3 across shapes, valid_shapes, dtypes, offset, slope."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_relu(self, test_runner, label, m, n, valid):
        result = test_runner.run(TileReluTestCase(m=m, n=n, valid_shapes=valid))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_lrelu(self, test_runner, label, m, n, valid):
        result = test_runner.run(TileLreluTestCase(m=m, n=n, valid_shapes=valid))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("slope", _LRELU_SLOPES, ids=[f"s{s}" for s in _LRELU_SLOPES])
    def test_tile_lrelu_slopes(self, test_runner, slope):
        result = test_runner.run(TileLreluTestCase(slope=slope))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_relu_fp16(self, test_runner):
        result = test_runner.run(TileReluTestCase(dtype=DataType.FP16))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_lrelu_fp16(self, test_runner):
        result = test_runner.run(TileLreluTestCase(dtype=DataType.FP16))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_relu_offset(self, test_runner):
        result = test_runner.run(TileReluTestCase(out_m=32, out_n=32, off=(16, 16)))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_lrelu_offset(self, test_runner):
        result = test_runner.run(TileLreluTestCase(out_m=32, out_n=32, off=(16, 16)))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
