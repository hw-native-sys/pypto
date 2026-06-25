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
golden. The op is a generator (no tensor inputs); each dtype needs its own
``@pl.program`` (a program hardcodes its dtype at parse time) and the op call
must be a literal ``pl.tile.tri``. ``diagonal`` and ``upper`` are passed as
closure values (operand values are fine as closures; only the op name must be
literal).

The compile-time ``upper`` selector is carried to ptoas via the generic op form
(the pretty ``ins/outs`` form only encodes the lower-triangular variant), so the
upper cases are the regression gate for that codegen path.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

M = 16
N = 16

# dtype label -> (harness DataType, torch dtype)
_DTYPES = {
    "int32": (DataType.INT32, torch.int32),
    "fp16": (DataType.FP16, torch.float16),
    "fp32": (DataType.FP32, torch.float32),
}


def _prog_int32(diagonal: int, upper: bool):
    @pl.program
    class TriInt32:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, out: pl.Out[pl.Tensor[[M, N], pl.INT32]]) -> pl.Tensor[[M, N], pl.INT32]:
            o: pl.Tile[[M, N], pl.INT32] = pl.tile.tri(diagonal, [M, N], dtype=pl.INT32, upper=upper)
            out = pl.store(o, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(self, out: pl.Out[pl.Tensor[[M, N], pl.INT32]]) -> pl.Tensor[[M, N], pl.INT32]:
            out = self.kernel(out)
            return out

    return TriInt32


def _prog_fp16(diagonal: int, upper: bool):
    @pl.program
    class TriFP16:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, out: pl.Out[pl.Tensor[[M, N], pl.FP16]]) -> pl.Tensor[[M, N], pl.FP16]:
            o: pl.Tile[[M, N], pl.FP16] = pl.tile.tri(diagonal, [M, N], dtype=pl.FP16, upper=upper)
            out = pl.store(o, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(self, out: pl.Out[pl.Tensor[[M, N], pl.FP16]]) -> pl.Tensor[[M, N], pl.FP16]:
            out = self.kernel(out)
            return out

    return TriFP16


def _prog_fp32(diagonal: int, upper: bool):
    @pl.program
    class TriFP32:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, out: pl.Out[pl.Tensor[[M, N], pl.FP32]]) -> pl.Tensor[[M, N], pl.FP32]:
            o: pl.Tile[[M, N], pl.FP32] = pl.tile.tri(diagonal, [M, N], dtype=pl.FP32, upper=upper)
            out = pl.store(o, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(self, out: pl.Out[pl.Tensor[[M, N], pl.FP32]]) -> pl.Tensor[[M, N], pl.FP32]:
            out = self.kernel(out)
            return out

    return TriFP32


_PROG_FACTORY = {"int32": _prog_int32, "fp16": _prog_fp16, "fp32": _prog_fp32}


def _golden(upper: bool, diagonal: int, torch_dtype: torch.dtype) -> torch.Tensor:
    ones = torch.ones(M, N)
    mask = torch.triu(ones, diagonal=diagonal) if upper else torch.tril(ones, diagonal=diagonal)
    return mask.to(torch_dtype)


class TriTestCase(PTOTestCase):
    """tile.tri triangular mask generation (no inputs)."""

    __test__ = False

    def __init__(self, dtype_key: str, upper: bool, diagonal: int, *, platform=None, config=None):
        super().__init__(config, platform=platform)
        self._dtype_key = dtype_key
        self._upper = upper
        self._diagonal = diagonal

    def get_name(self) -> str:
        side = "upper" if self._upper else "lower"
        return f"tri_{self._dtype_key}_{side}_d{self._diagonal}".replace("-", "m")

    def define_tensors(self) -> list[TensorSpec]:
        hdtype, _ = _DTYPES[self._dtype_key]
        return [TensorSpec("out", [M, N], hdtype, is_output=True)]

    def get_program(self) -> Any:
        return _PROG_FACTORY[self._dtype_key](self._diagonal, self._upper)

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        _, tdtype = _DTYPES[self._dtype_key]
        tensors["out"][:] = _golden(self._upper, self._diagonal, tdtype)


_DTYPE_KEYS = ["int32", "fp16", "fp32"]
# (label, upper, diagonal)
_CASES = [
    ("lower_d0", False, 0),
    ("upper_d0", True, 0),
    ("lower_d2", False, 2),
    ("upper_dm1", True, -1),
]


class TestTri:
    """Test tile.tri across supported platforms, dtypes, sides, and diagonals."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("dtype_key", _DTYPE_KEYS)
    @pytest.mark.parametrize(("label", "upper", "diagonal"), _CASES)
    def test_tri(self, test_runner, platform, dtype_key, label, upper, diagonal):
        result = test_runner.run(TriTestCase(dtype_key, upper, diagonal, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
