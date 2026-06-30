# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for the tile-level ``tile.sels`` op (TSELS).

``dst[i, j] = mask[i, j] ? src[i, j] : scalar`` — per-element selection between a
source tile and a scalar, driven by a predicate mask tile. The mask is produced
on-device by ``tile.cmps`` (compare src against a threshold), mirroring how
``tile.sel`` consumes a ``tile.cmp`` mask; ``tmp`` is the mandatory TSELS scratch
tile.

``tile.sels`` is tile-only (no tensor-level counterpart). a2a3 supports the
shared src/dst element types i16/i32/f16/f32 (per the PTO IR manual), so coverage
sweeps FP32 and FP16, several comparison modes, the false-branch scalar, multiple
shapes, and aligned + narrow ``valid_shape``. The INT32 case is skipped pending a
fix to integer ``cmps`` scalar codegen (the mask builder, not ``sels`` itself);
see ``test_tile_sels_int32``.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

# cmp/cmps modes (src/backend/common/pto_ops_common.cpp): index → predicate.
CMP_EQ, CMP_NE, CMP_LT, CMP_LE, CMP_GT, CMP_GE = range(6)
_CMP_FN = {
    CMP_EQ: lambda a, t: a == t,
    CMP_NE: lambda a, t: a != t,
    CMP_LT: lambda a, t: a < t,
    CMP_LE: lambda a, t: a <= t,
    CMP_GT: lambda a, t: a > t,
    CMP_GE: lambda a, t: a >= t,
}

_PL_DT = {DataType.FP32: pl.FP32, DataType.FP16: pl.FP16, DataType.INT32: pl.INT32}
_TORCH_DT = {DataType.FP32: torch.float32, DataType.FP16: torch.float16, DataType.INT32: torch.int32}


def _signed(m, n, dtype):
    """Signed values spanning negatives/zero/positives so the threshold splits the tile."""
    base = torch.arange(m * n, dtype=torch.float32).reshape(m, n) % 17 - 8
    return base.to(_TORCH_DT[dtype]).contiguous()


class TileSelsTestCase(PTOTestCase):
    """tile.sels: dst = mask ? src : scalar, with mask = cmps(src, threshold, mode)."""

    __test__ = False

    def __init__(
        self,
        *,
        m=16,
        n=16,
        valid_shapes=None,
        dtype=DataType.FP32,
        cmp_mode=CMP_GT,
        threshold=0.0,
        scalar=7.0,
        config=None,
    ):
        super().__init__(config)
        self._m, self._n, self._valid, self._dtype = m, n, valid_shapes, dtype
        self._mode, self._thr, self._scalar = cmp_mode, threshold, scalar

    def get_name(self) -> str:
        v = f"_v{self._valid[0]}x{self._valid[1]}" if self._valid else ""
        return f"tile_sels_{self._m}x{self._n}_{self._dtype.value}_m{self._mode}_s{self._scalar}{v}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [self._m, self._n],
                self._dtype,
                init_value=lambda: _signed(self._m, self._n, self._dtype),
            ),
            TensorSpec("out", [self._m, self._n], self._dtype, is_output=True),
        ]

    def get_program(self) -> Any:
        m, n = self._m, self._n
        vshape = list(self._valid) if self._valid else [m, n]
        dt = _PL_DT[self._dtype]
        mode, thr, scalar = self._mode, self._thr, self._scalar
        # cmps packs the predicate into a UINT8 tile of shape [m, ceil(n/16)*16] with a
        # narrowed valid_shape; the exact view is inferred by the type checker, so the
        # mask is bound without an explicit annotation via an intermediate assignment.

        @pl.program
        class SelsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[m, n], dt]]
            ) -> pl.Tensor[[m, n], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                mask = pl.cmps(a_tile, thr, cmp_type=mode)
                tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
                out = pl.store(pl.tile.sels(mask, a_tile, tmp, scalar), [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.Out[pl.Tensor[[m, n], dt]]
            ) -> pl.Tensor[[m, n], dt]:
                out = self.kernel(a, out)
                return out

        return SelsProgram

    def _ref(self, a):
        cond = _CMP_FN[self._mode](a, a.new_tensor(self._thr))
        scalar = a.new_tensor(self._scalar)
        return torch.where(cond, a, scalar)

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


_SHAPE_CFGS = [
    ("16x16", 16, 16, None),
    ("32x64", 32, 64, None),
    ("8x128", 8, 128, None),
    ("16x16_narrow_both", 16, 16, (8, 12)),
    ("16x16_narrow_rows", 16, 16, (8, 16)),
    ("16x16_narrow_cols", 16, 16, (16, 8)),
]
_SCALARS = [-3.0, 0.0, 7.0]
_MODES = [CMP_GT, CMP_LT, CMP_GE, CMP_EQ]


class TestTileSels:
    """tile.sels on a2a3: shapes, valid_shapes, dtypes, cmp modes, false-branch scalar."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_sels_shapes(self, test_runner, label, m, n, valid):
        result = test_runner.run(TileSelsTestCase(m=m, n=n, valid_shapes=valid))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("scalar", _SCALARS, ids=[f"s{s}" for s in _SCALARS])
    def test_tile_sels_scalars(self, test_runner, scalar):
        result = test_runner.run(TileSelsTestCase(scalar=scalar))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("mode", _MODES, ids=[f"m{m}" for m in _MODES])
    def test_tile_sels_modes(self, test_runner, mode):
        result = test_runner.run(TileSelsTestCase(cmp_mode=mode))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_sels_fp16(self, test_runner):
        result = test_runner.run(TileSelsTestCase(dtype=DataType.FP16))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.skip(
        reason="Integer cmps codegen gap: the cmps scalar threshold is emitted as MLIR "
        "'index' instead of the tile dtype, so ptoas rejects pto.tcmps ('operand must be "
        "numeric, got index'). Pre-existing and unrelated to tile.sels — test_cmp.py only "
        "ever exercises FP32 cmps. Re-enable once integer cmps scalar codegen is fixed."
    )
    def test_tile_sels_int32(self, test_runner):
        # Integer select is exact; threshold/scalar are integers.
        result = test_runner.run(TileSelsTestCase(dtype=DataType.INT32, threshold=0, scalar=5))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
