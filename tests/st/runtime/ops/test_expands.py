# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile/tensor scalar broadcast (expands).

Covers two ops:
- ``tile.expands(target, scalar)``   -> ``pto.texpands``
- ``tensor.expands(target, scalar)`` -> lowers to ``tile.expands``

Semantics: broadcast ``scalar`` to the shape of ``target``; the output is
``torch.full(target.shape, scalar)``. The ``target`` operand contributes only
its shape -- its element values never affect the result -- but a real input
tile is still loaded as the target. The scalar is swept over negatives, zero,
and positives. The op is a pure fill (no arithmetic), so it is exact in FP16
and stays at the strict default tolerance.

Scope is a2a3 only (``@pytest.mark.platforms("a2a3")``); a5 coverage is a
separate PR.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

_PL_DT = {DataType.FP32: pl.FP32, DataType.FP16: pl.FP16}

# Scalar sweep: negative, zero, positive.
_SCALARS = [-2.5, 0.0, 3.0]

# (label, m, n, valid) — last three narrow the valid_shape below the physical
# tile (combined / rows-only / cols-only).
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


# ---------------------------------------------------------------------------
# Tile-level: InCore kernel (pl.load with a possibly-narrowed valid_shape) +
# Orchestration wrapper. The loaded tile is only a shape carrier for expands.
# ---------------------------------------------------------------------------


class TileExpandsTestCase(PTOTestCase):
    __test__ = False
    op_name = "expands"

    def __init__(self, *, scalar=3.0, m=16, n=16, valid_shapes=None, dtype=DataType.FP32, config=None):
        super().__init__(config)
        self._scalar = scalar
        self._m, self._n, self._valid, self._dtype = m, n, valid_shapes, dtype

    def get_name(self) -> str:
        v = f"_v{self._valid[0]}x{self._valid[1]}" if self._valid else ""
        return f"tile_{self.op_name}_{self._m}x{self._n}_{self._dtype.value}_s{self._scalar}{v}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._m, self._n], self._dtype, init_value=lambda: _signed(self._m, self._n)),
            TensorSpec("out", [self._m, self._n], self._dtype, is_output=True),
        ]

    def _ref(self, a):
        # Output is all-scalar; a only provides the shape/dtype.
        return torch.full_like(a, self._scalar)

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        out = torch.zeros_like(tensors["out"])
        if self._valid:
            vm, vn = self._valid
            res = torch.zeros_like(a)
            res[:vm, :vn] = self._ref(a[:vm, :vn])
        else:
            res = self._ref(a)
        out[: self._m, : self._n] = res
        tensors["out"][:] = out

    def get_program(self) -> Any:
        m, n = self._m, self._n
        vshape = list(self._valid) if self._valid else [m, n]
        dt = _PL_DT[self._dtype]
        scalar = self._scalar

        @pl.program
        class ExpandsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[m, n], dt]]
            ) -> pl.Tensor[[m, n], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                out = pl.store(pl.tile.expands(a_tile, scalar), [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[m, n], dt]]
            ) -> pl.Tensor[[m, n], dt]:
                out = self.kernel(a, out)
                return out

        return ExpandsProgram


# ---------------------------------------------------------------------------
# Tensor-level: Opaque + pl.at(CORE_GROUP) + pl.assemble; lowered to
# tile.expands by ConvertTensorToTileOps.
# ---------------------------------------------------------------------------


def _make_tensor_expands_program(m, n, dt, scalar):
    @pl.program
    class TensorExpandsProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[m, n], dt]]) -> pl.Tensor[[m, n], dt]:
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.assemble(out, pl.expands(a, scalar), [0, 0])
            return out

    return TensorExpandsProgram


class TensorExpandsTestCase(PTOTestCase):
    __test__ = False
    op_name = "expands"

    def __init__(self, *, scalar=3.0, m=16, n=16, dtype=DataType.FP32, config=None):
        super().__init__(config)
        self._scalar = scalar
        self._m, self._n, self._dtype = m, n, dtype

    def get_name(self) -> str:
        return f"tensor_{self.op_name}_{self._m}x{self._n}_{self._dtype.value}_s{self._scalar}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._m, self._n], self._dtype, init_value=lambda: _signed(self._m, self._n)),
            TensorSpec("out", [self._m, self._n], self._dtype, is_output=True),
        ]

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.full_like(tensors["out"], self._scalar)

    def get_program(self) -> Any:
        return _make_tensor_expands_program(self._m, self._n, _PL_DT[self._dtype], self._scalar)


class TestTileExpands:
    """Tile-level expands on a2a3 across shapes, valid_shapes, scalars, dtypes."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_expands(self, test_runner, label, m, n, valid):
        result = test_runner.run(TileExpandsTestCase(m=m, n=n, valid_shapes=valid))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("scalar", _SCALARS, ids=[f"s{s}" for s in _SCALARS])
    def test_tile_expands_scalars(self, test_runner, scalar):
        result = test_runner.run(TileExpandsTestCase(scalar=scalar))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_expands_fp16(self, test_runner):
        # expands is a pure fill (no arithmetic) -> exact in FP16, default tol.
        result = test_runner.run(TileExpandsTestCase(dtype=DataType.FP16))
        assert result.passed, f"Test failed: {result.error}"


class TestTensorExpands:
    """Tensor-level expands (lowered by ConvertTensorToTileOps)."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("scalar", _SCALARS, ids=[f"s{s}" for s in _SCALARS])
    def test_tensor_expands(self, test_runner, scalar):
        result = test_runner.run(TensorExpandsTestCase(scalar=scalar))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tensor_expands_fp16(self, test_runner):
        result = test_runner.run(TensorExpandsTestCase(dtype=DataType.FP16))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
