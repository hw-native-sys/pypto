# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for the element-wise natural-log op.

Tile-level:  ``tile.log``  -> ``pto.tlog``
Tensor-level: ``tensor.log`` (lowered by ConvertTensorToTileOps -> tile.log).

The result matches ``torch.log``. The input is kept strictly positive in every
case (arange + 1) so the golden numerics are well-defined (``log`` of <= 0 is
undefined). ``log`` supports FP16/FP32 only (int is rejected). Coverage:
multiple shapes (square/tall/wide), aligned + narrow valid_shape (combined /
rows-only / cols-only), and FP32 + FP16.

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


def _pos(m: int, n: int) -> torch.Tensor:
    """Strictly positive input (1 .. m*n) so log is well-defined."""
    return (torch.arange(m * n, dtype=torch.float32).reshape(m, n) + 1.0).contiguous()


class _LogBase(PTOTestCase):
    """Shared scaffolding: a strictly-positive 2D input and a valid_shape-aware
    golden via ``torch.log``."""

    __test__ = False
    op_name = ""

    def __init__(
        self,
        *,
        m: int = 16,
        n: int = 16,
        valid_shapes=None,
        dtype: DataType = DataType.FP32,
        config=None,
    ):
        super().__init__(config)
        self._m, self._n, self._valid, self._dtype = m, n, valid_shapes, dtype

    def get_name(self) -> str:
        v = f"_v{self._valid[0]}x{self._valid[1]}" if self._valid else ""
        return f"{self.op_name}_{self._m}x{self._n}_{self._dtype.value}{v}"

    def define_tensors(self) -> list[TensorSpec]:
        m, n = self._m, self._n
        return [
            TensorSpec("a", [m, n], self._dtype, init_value=lambda: _pos(m, n)),
            TensorSpec("out", [m, n], self._dtype, is_output=True),
        ]

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        out = torch.zeros_like(tensors["out"])
        if self._valid:
            vm, vn = self._valid
            res = torch.zeros_like(a)
            res[:vm, :vn] = torch.log(a[:vm, :vn])
        else:
            res = torch.log(a)
        out[: self._m, : self._n] = res
        tensors["out"][:] = out


class TileLogTestCase(_LogBase):
    """tile.log: element-wise natural log of a tile."""

    op_name = "tile_log"

    def get_program(self) -> Any:
        m, n = self._m, self._n
        vshape = list(self._valid) if self._valid else [m, n]
        dt = _PL_DT[self._dtype]

        @pl.program
        class TileLogProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[m, n], dt],
                out: pl.Out[pl.Tensor[[m, n], dt]],
            ) -> pl.Tensor[[m, n], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                out = pl.store(pl.tile.log(a_tile), [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[m, n], dt],
                out: pl.Out[pl.Tensor[[m, n], dt]],
            ) -> pl.Tensor[[m, n], dt]:
                out = self.kernel(a, out)
                return out

        return TileLogProgram


class TensorLogTestCase(_LogBase):
    """tensor.log: lowers to tile.log via ConvertTensorToTileOps."""

    op_name = "tensor_log"

    def get_program(self) -> Any:
        m, n = self._m, self._n
        dt = _PL_DT[self._dtype]

        @pl.program
        class TensorLogProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                a: pl.Tensor[[m, n], dt],
                out: pl.Out[pl.Tensor[[m, n], dt]],
            ) -> pl.Tensor[[m, n], dt]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.assemble(out, pl.log(a), [0, 0])
                return out

        return TensorLogProgram


# FP16 log needs relaxed tolerance, not a workaround: FP16 machine epsilon is
# 2**-10 ~= 9.8e-4, so a 1e-5 comparison is below FP16's representable
# precision. log's transcendental approximation rounds to ~1 ULP, so 2e-3
# (~2 FP16 ULP) is the right bar.
_FP16_CFG = RunConfig(rtol=2e-3, atol=2e-3)


class TestTileLog:
    """Tile-level log on a2a3."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_log(self, test_runner, label, m, n, valid):
        result = test_runner.run(TileLogTestCase(m=m, n=n, valid_shapes=valid))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_log_fp16(self, test_runner):
        result = test_runner.run(TileLogTestCase(dtype=DataType.FP16, config=_FP16_CFG))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_log_fp16_narrow(self, test_runner):
        result = test_runner.run(TileLogTestCase(dtype=DataType.FP16, valid_shapes=(8, 12), config=_FP16_CFG))
        assert result.passed, f"Test failed: {result.error}"


class TestTensorLog:
    """Tensor-level log (lowered by ConvertTensorToTileOps) on a2a3."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _FULL_SHAPE_CFGS, ids=[c[0] for c in _FULL_SHAPE_CFGS])
    def test_tensor_log(self, test_runner, label, m, n, valid):
        result = test_runner.run(TensorLogTestCase(m=m, n=n, valid_shapes=valid))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tensor_log_fp16(self, test_runner):
        result = test_runner.run(TensorLogTestCase(dtype=DataType.FP16, config=_FP16_CFG))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
