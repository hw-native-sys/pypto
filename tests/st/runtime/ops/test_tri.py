# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for ``tile.tri`` -> ``pto.ttri`` (triangular mask generation).

``tri`` writes a triangular 0/1 mask over the destination region ``[R, C]`` with
a diagonal offset ``d``:
- Lower (``upper=False``): ``dst[i, j] = 1`` if ``j <= i + d`` else ``0``
- Upper (``upper=True``):  ``dst[i, j] = 1`` if ``j >= i + d`` else ``0``

This matches ``torch.tril`` / ``torch.triu`` with ``diagonal=d``, which is the
golden.

``tri`` is a generator (no tensor inputs), so the destination *shape* is its
valid region — there is no separate ``valid_shape`` load parameter. Coverage
therefore sweeps the shape instead, which drives the same validRow/validCol
codegen paths. Columns are kept a multiple of 16 because ptoas requires a tile's
row byte size (``cols * sizeof(dtype)``) to be 32-byte aligned (16 cols covers
FP16/FP32/INT32); rows are unconstrained, so they carry the odd / tail-path
coverage:
- ``16x16`` — square baseline (lower + upper).
- ``8x16``  — validRow < 16 (narrow-row tail path).
- ``13x16`` — odd row count, validRow < 16.
- ``32x16`` — more than one block of rows.
- ``16x32`` — wider columns.

Each dtype needs its own ``@pl.program`` (a program hardcodes its dtype at parse
time) and the op call must be a literal ``pl.tile.tri``; ``R``/``C``/``diagonal``/
``upper`` are passed as closure values (operand values are fine as closures).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

# dtype label -> (harness DataType, torch dtype)
_DTYPES = {
    "int32": (DataType.INT32, torch.int32),
    "fp16": (DataType.FP16, torch.float16),
    "fp32": (DataType.FP32, torch.float32),
}


def _prog_int32(rows: int, cols: int, diagonal: int, upper: bool):
    @pl.program
    class TriInt32:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, out: pl.Out[pl.Tensor[[rows, cols], pl.INT32]]) -> pl.Tensor[[rows, cols], pl.INT32]:
            o: pl.Tile[[rows, cols], pl.INT32] = pl.tile.tri(
                diagonal, [rows, cols], dtype=pl.INT32, upper=upper
            )
            out = pl.store(o, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self, out: pl.Out[pl.Tensor[[rows, cols], pl.INT32]]
        ) -> pl.Tensor[[rows, cols], pl.INT32]:
            out = self.kernel(out)
            return out

    return TriInt32


def _prog_fp16(rows: int, cols: int, diagonal: int, upper: bool):
    @pl.program
    class TriFP16:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, out: pl.Out[pl.Tensor[[rows, cols], pl.FP16]]) -> pl.Tensor[[rows, cols], pl.FP16]:
            o: pl.Tile[[rows, cols], pl.FP16] = pl.tile.tri(
                diagonal, [rows, cols], dtype=pl.FP16, upper=upper
            )
            out = pl.store(o, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self, out: pl.Out[pl.Tensor[[rows, cols], pl.FP16]]
        ) -> pl.Tensor[[rows, cols], pl.FP16]:
            out = self.kernel(out)
            return out

    return TriFP16


def _prog_fp32(rows: int, cols: int, diagonal: int, upper: bool):
    @pl.program
    class TriFP32:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]]) -> pl.Tensor[[rows, cols], pl.FP32]:
            o: pl.Tile[[rows, cols], pl.FP32] = pl.tile.tri(
                diagonal, [rows, cols], dtype=pl.FP32, upper=upper
            )
            out = pl.store(o, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self, out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]]
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            out = self.kernel(out)
            return out

    return TriFP32


_PROG_FACTORY = {"int32": _prog_int32, "fp16": _prog_fp16, "fp32": _prog_fp32}


def _golden(rows: int, cols: int, upper: bool, diagonal: int, torch_dtype: torch.dtype) -> torch.Tensor:
    ones = torch.ones(rows, cols)
    mask = torch.triu(ones, diagonal=diagonal) if upper else torch.tril(ones, diagonal=diagonal)
    return mask.to(torch_dtype)


class TriTestCase(PTOTestCase):
    """tile.tri triangular mask generation (no inputs)."""

    __test__ = False

    def __init__(self, dtype_key, rows, cols, upper, diagonal, label, *, platform=None, config=None):
        super().__init__(config, platform=platform)
        self._dtype_key = dtype_key
        self._rows = rows
        self._cols = cols
        self._upper = upper
        self._diagonal = diagonal
        self._label = label

    def get_name(self) -> str:
        side = "upper" if self._upper else "lower"
        return f"tri_{self._dtype_key}_{self._label}_{side}_d{self._diagonal}".replace("-", "m")

    def define_tensors(self) -> list[TensorSpec]:
        hdtype, _ = _DTYPES[self._dtype_key]
        return [TensorSpec("out", [self._rows, self._cols], hdtype, is_output=True)]

    def get_program(self) -> Any:
        return _PROG_FACTORY[self._dtype_key](self._rows, self._cols, self._diagonal, self._upper)

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        _, tdtype = _DTYPES[self._dtype_key]
        tensors["out"][:] = _golden(self._rows, self._cols, self._upper, self._diagonal, tdtype)


_DTYPE_KEYS = ["int32", "fp16", "fp32"]
# (label, rows, cols, upper, diagonal) — cols kept a multiple of 16 (32B row align)
_CASES = [
    ("16x16_d0", 16, 16, False, 0),
    ("16x16_d0", 16, 16, True, 0),
    ("8x16_d1", 8, 16, False, 1),  # validRow < 16 tail path
    ("13x16_dm1", 13, 16, True, -1),  # odd row count, validRow < 16
    ("32x16_d3", 32, 16, False, 3),  # multiple row blocks
    ("16x32_d0", 16, 32, True, 0),  # wider columns
]


class TestTri:
    """Test tile.tri across supported platforms, dtypes, shapes, sides, and diagonals."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("dtype_key", _DTYPE_KEYS)
    @pytest.mark.parametrize(("label", "rows", "cols", "upper", "diagonal"), _CASES)
    def test_tri(self, test_runner, platform, dtype_key, label, rows, cols, upper, diagonal):
        result = test_runner.run(
            TriTestCase(dtype_key, rows, cols, upper, diagonal, label, platform=platform)
        )
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
