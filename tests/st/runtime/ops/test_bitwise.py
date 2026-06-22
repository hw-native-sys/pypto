# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for the tile-level integer bitwise NOT op.

Only ``not_`` (TNOT, INT16) is covered: it is the one integer bitwise/remainder
tile op that currently assembles and runs correctly on a2a3. The rest of the
family is blocked by a2a3 assembler/ISA defects (tracked in KNOWN_ISSUES):
  * rem  — TREM returns wrong values on int32 (e.g. 0 where a%b != 0).
  * rems — TREMS alloc_tile element-type error.
  * xor/xors — TXOR/TXORS require int16/uint16 element type (int32 rejected).
  * and_/or_/shl/shr (+scalar) — ptoas rejects pto.tand/tor/tshl/tshr
    ("invalid kind of type specified").

(The tile.rem/rems DSL/codegen now carry the scratch ``tmp`` operand that the
ISA requires — see the elementwise op registration — so they are ready to
re-enable once the a2a3 TREM/TXOR paths are fixed.)

not_ runs aligned (valid_shape == [M, N]) and narrow (valid_shape
[VALID_M, VALID_N] < [M, N]; invalid output stays zero).

Scope is a2a3 only (``@pytest.mark.platforms("a2a3")``); a5 coverage is a
separate PR.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 16
N = 16
VALID_M = 8
VALID_N = 12


def _a_input() -> torch.Tensor:
    """Non-negative INT spread 0..255 (cast to the spec dtype by create_tensor)."""
    return torch.arange(M * N, dtype=torch.int32).reshape(M, N)


class BitwiseNotTestCase(PTOTestCase):
    """Unary not_ on INT16, aligned or narrow valid_shape."""

    __test__ = False

    def __init__(self, *, valid_shapes: tuple[int, int] | None = None, config=None):
        super().__init__(config)
        self._valid = valid_shapes

    def get_name(self) -> str:
        return "tile_bitwise_not_narrow" if self._valid else "tile_bitwise_not"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.INT16, init_value=_a_input),
            TensorSpec("not_o", [M, N], DataType.INT16, is_output=True),
        ]

    def get_program(self) -> Any:
        vshape = list(self._valid) if self._valid else [M, N]

        @pl.program
        class BitwiseNotProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, N], pl.INT16],
                not_o: pl.Out[pl.Tensor[[M, N], pl.INT16]],
            ) -> pl.Tensor[[M, N], pl.INT16]:
                a_tile = pl.load(a, [0, 0], [M, N], valid_shapes=vshape)
                not_o = pl.store(pl.tile.not_(a_tile), [0, 0], not_o)
                return not_o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, N], pl.INT16],
                not_o: pl.Out[pl.Tensor[[M, N], pl.INT16]],
            ) -> pl.Tensor[[M, N], pl.INT16]:
                not_o = self.kernel(a, not_o)
                return not_o

        return BitwiseNotProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        if self._valid:
            vm, vn = self._valid
            out = torch.zeros_like(tensors["not_o"])
            out[:vm, :vn] = torch.bitwise_not(a[:vm, :vn])
            tensors["not_o"][:] = out
        else:
            tensors["not_o"][:] = torch.bitwise_not(a)


class TestBitwise:
    """Tile-level integer bitwise NOT on a2a3."""

    @pytest.mark.platforms("a2a3")
    def test_tile_bitwise_not(self, test_runner):
        result = test_runner.run(BitwiseNotTestCase())
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_bitwise_not_narrow(self, test_runner):
        result = test_runner.run(BitwiseNotTestCase(valid_shapes=(VALID_M, VALID_N)))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
