# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile-level unary math ops: sin, cos, sqrt.

sin/cos are FP32-only and interpret their input as radians; they are exercised
on both positive and signed/multi-period input. sqrt needs a non-negative
domain and is also exercised at the exact-0 boundary and in FP16.

Coverage dimensions per op: multiple shapes (square/tall/wide), aligned + narrow
valid_shape (combined / rows-only / cols-only), non-zero store offset, dtype, and
op-specific input ranges.

sin/cos hold at the strict 1e-5 default (the hardware approximation is accurate);
their wide 8x128 case is skipped because a2a3 miscomputes columns >= 64 on a
width-128 tile (a real codegen/ISA defect, not a precision issue — sqrt at the
same shape is correct). See KNOWN_ISSUES.

Scope is a2a3 only (``@pytest.mark.platforms("a2a3")``); a5 coverage is a
separate PR.
"""

import math
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

_PL_DT = {DataType.FP32: pl.FP32, DataType.FP16: pl.FP16}

# (label, m, n, valid_shapes) — the full shape/valid sweep (used by sqrt).
_SHAPE_CFGS = [
    ("16x16", 16, 16, None),
    ("32x64", 32, 64, None),
    ("8x128", 8, 128, None),
    ("16x16_narrow_both", 16, 16, (8, 12)),
    ("16x16_narrow_rows", 16, 16, (8, 16)),
    ("16x16_narrow_cols", 16, 16, (16, 8)),
]

# sin/cos sweep: the wide 8x128 case is omitted — a2a3 miscomputes its columns
# >= 64 (a ptoas/pto-isa defect, see KNOWN_ISSUES). It will be added once the
# underlying ISA path is fixed.
_TRIG_SHAPE_CFGS = [
    ("16x16", 16, 16, None),
    ("32x64", 32, 64, None),
    ("16x16_narrow_both", 16, 16, (8, 12)),
    ("16x16_narrow_rows", 16, 16, (8, 16)),
    ("16x16_narrow_cols", 16, 16, (16, 8)),
]


def _positive(m, n):
    return torch.rand(m, n, dtype=torch.float32) * 3.0 + 0.1


def _signed_periods(m, n):
    # negatives + a full period each way; kept within +/-2pi so hardware range
    # reduction stays accurate (large-arg sin/cos accuracy is out of scope).
    return torch.rand(m, n, dtype=torch.float32) * (4.0 * math.pi) - (2.0 * math.pi)


def _sqrt_domain(m, n):
    t = torch.rand(m, n, dtype=torch.float32) * 100.0
    t.view(-1)[0] = 0.0  # exact-0 boundary
    return t


class _UnaryMathBase(PTOTestCase):
    """Shared scaffolding: load [m,n] (optionally narrow valid), apply one op, store at offset."""

    __test__ = False
    op_name = ""  # subclass sets; used in get_name

    def __init__(
        self,
        *,
        m=16,
        n=16,
        valid_shapes=None,
        dtype=DataType.FP32,
        input_fn=None,
        out_m=None,
        out_n=None,
        off=(0, 0),
        config=None,
    ):
        super().__init__(config)
        self._m, self._n, self._valid, self._dtype = m, n, valid_shapes, dtype
        self._input_fn = input_fn or _positive
        self._out_m, self._out_n, self._off = out_m or m, out_n or n, off

    def get_name(self) -> str:
        v = f"_v{self._valid[0]}x{self._valid[1]}" if self._valid else ""
        o = f"_off{self._off[0]}x{self._off[1]}" if self._off != (0, 0) else ""
        return f"tile_{self.op_name}_{self._m}x{self._n}_{self._dtype.value}{v}{o}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "a", [self._m, self._n], self._dtype, init_value=lambda: self._input_fn(self._m, self._n)
            ),
            TensorSpec("out", [self._out_m, self._out_n], self._dtype, is_output=True),
        ]

    def _ref(self, a: torch.Tensor) -> torch.Tensor:
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


class TileSinTestCase(_UnaryMathBase):
    op_name = "sin"

    def _ref(self, a):
        return torch.sin(a)

    def get_program(self) -> Any:
        m, n, om, on, off = self._m, self._n, self._out_m, self._out_n, list(self._off)
        vshape = list(self._valid) if self._valid else [m, n]
        dt = _PL_DT[self._dtype]

        @pl.program
        class SinProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                out = pl.store(pl.tile.sin(a_tile), off, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                out = self.kernel(a, out)
                return out

        return SinProgram


class TileCosTestCase(_UnaryMathBase):
    op_name = "cos"

    def _ref(self, a):
        return torch.cos(a)

    def get_program(self) -> Any:
        m, n, om, on, off = self._m, self._n, self._out_m, self._out_n, list(self._off)
        vshape = list(self._valid) if self._valid else [m, n]
        dt = _PL_DT[self._dtype]

        @pl.program
        class CosProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                out = pl.store(pl.tile.cos(a_tile), off, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                out = self.kernel(a, out)
                return out

        return CosProgram


class TileSqrtTestCase(_UnaryMathBase):
    op_name = "sqrt"

    def _ref(self, a):
        return torch.sqrt(a)

    def get_program(self) -> Any:
        m, n, om, on, off = self._m, self._n, self._out_m, self._out_n, list(self._off)
        vshape = list(self._valid) if self._valid else [m, n]
        dt = _PL_DT[self._dtype]

        @pl.program
        class SqrtProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                out = pl.store(pl.tile.sqrt(a_tile), off, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                out = self.kernel(a, out)
                return out

        return SqrtProgram


class TestUnaryMath:
    """Tile-level sin/cos/sqrt on a2a3 across shapes, valid_shapes, dtypes, offset, input ranges."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _TRIG_SHAPE_CFGS, ids=[c[0] for c in _TRIG_SHAPE_CFGS])
    def test_tile_sin(self, test_runner, label, m, n, valid):
        result = test_runner.run(TileSinTestCase(m=m, n=n, valid_shapes=valid))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _TRIG_SHAPE_CFGS, ids=[c[0] for c in _TRIG_SHAPE_CFGS])
    def test_tile_cos(self, test_runner, label, m, n, valid):
        result = test_runner.run(TileCosTestCase(m=m, n=n, valid_shapes=valid))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_sqrt(self, test_runner, label, m, n, valid):
        result = test_runner.run(TileSqrtTestCase(m=m, n=n, valid_shapes=valid))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_sin_signed(self, test_runner):
        result = test_runner.run(TileSinTestCase(input_fn=_signed_periods))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_cos_signed(self, test_runner):
        result = test_runner.run(TileCosTestCase(input_fn=_signed_periods))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_sqrt_zero_boundary(self, test_runner):
        result = test_runner.run(TileSqrtTestCase(input_fn=_sqrt_domain))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_sqrt_fp16(self, test_runner):
        result = test_runner.run(TileSqrtTestCase(m=32, n=64, dtype=DataType.FP16))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_sqrt_offset(self, test_runner):
        result = test_runner.run(TileSqrtTestCase(out_m=32, out_n=32, off=(16, 16)))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
