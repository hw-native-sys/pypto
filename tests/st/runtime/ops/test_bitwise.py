# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for the tile-level integer bitwise NOT op (TNOT).

``not_`` is the only integer bitwise/remainder tile op that currently assembles
and runs correctly on a2a3 (TNOT supports int16/uint16). Coverage: multiple
shapes (square/tall/wide), aligned + narrow valid_shape (combined / rows-only /
cols-only), int16 and uint16 dtypes, and non-zero store offset.

Blocked on a2a3 (tracked in KNOWN_ISSUES; the tile.rem/rems DSL+codegen already
carry the scratch tmp operand the ISA requires, ready to re-enable):
  * rem  — TREM returns wrong values on int32.
  * rems — TREMS alloc_tile element-type error.
  * xor/xors — TXOR/TXORS require int16/uint16 element type.
  * and_/or_/shl/shr (+scalar) — ptoas rejects pto.tand/tor/tshl/tshr.

Scope is a2a3 only (``@pytest.mark.platforms("a2a3")``); a5 coverage is a
separate PR.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

_PL_DT = {DataType.INT16: pl.INT16, DataType.UINT16: pl.UINT16}

_SHAPE_CFGS = [
    ("16x16", 16, 16, None),
    ("32x64", 32, 64, None),
    ("8x128", 8, 128, None),
    ("16x16_narrow_both", 16, 16, (8, 12)),
    ("16x16_narrow_rows", 16, 16, (8, 16)),
    ("16x16_narrow_cols", 16, 16, (16, 8)),
]


def _a_input(m, n):
    """Non-negative spread 0..(m*n-1), cast to the spec dtype by create_tensor."""
    return torch.arange(m * n, dtype=torch.int32).reshape(m, n)


class BitwiseNotTestCase(PTOTestCase):
    """Unary not_ with shape / valid / dtype / offset config."""

    __test__ = False

    def __init__(
        self,
        *,
        m=16,
        n=16,
        valid_shapes=None,
        dtype=DataType.INT16,
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
        return f"tile_not_{self._m}x{self._n}_{self._dtype.value}{v}{o}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._m, self._n], self._dtype, init_value=lambda: _a_input(self._m, self._n)),
            TensorSpec(
                "out", [self._out_m, self._out_n], self._dtype, is_output=True, init_value=torch.zeros
            ),
        ]

    def get_program(self) -> Any:
        m, n, om, on, off = self._m, self._n, self._out_m, self._out_n, list(self._off)
        vshape = list(self._valid) if self._valid else [m, n]
        dt = _PL_DT[self._dtype]

        @pl.program
        class BitwiseNotProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], dt], out: pl.InOut[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                a_tile = pl.load(a, [0, 0], [m, n], valid_shapes=vshape)
                out = pl.store(pl.tile.not_(a_tile), off, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], dt], out: pl.InOut[pl.Tensor[[om, on], dt]]
            ) -> pl.Tensor[[om, on], dt]:
                out = self.kernel(a, out)
                return out

        return BitwiseNotProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        out = torch.zeros_like(tensors["out"])
        r, c = self._off
        if self._valid:
            vm, vn = self._valid
            res = torch.zeros_like(a)
            res[:vm, :vn] = torch.bitwise_not(a[:vm, :vn])
        else:
            res = torch.bitwise_not(a)
        out[r : r + self._m, c : c + self._n] = res
        tensors["out"][:] = out


class TestBitwise:
    """Tile-level integer bitwise NOT on a2a3 across shapes, valid_shapes, dtypes, offset."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("label,m,n,valid", _SHAPE_CFGS, ids=[c[0] for c in _SHAPE_CFGS])
    def test_tile_not(self, test_runner, label, m, n, valid):
        result = test_runner.run(BitwiseNotTestCase(m=m, n=n, valid_shapes=valid))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_not_uint16(self, test_runner):
        result = test_runner.run(BitwiseNotTestCase(dtype=DataType.UINT16))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_not_offset(self, test_runner):
        result = test_runner.run(BitwiseNotTestCase(out_m=32, out_n=32, off=(16, 16)))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
