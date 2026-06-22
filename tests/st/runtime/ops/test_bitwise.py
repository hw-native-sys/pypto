# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile-level integer remainder / xor / not ops.

Tile-tile:   rem, xor    (INT32, both need a scratch tmp tile)
Tile-scalar: rems, xors  (INT32, both need a scratch tmp tile)
Unary:       not_        (INT16 — TNOT only supports int16/uint16)

These map to TREM/TXOR-family intrinsics that take a scratch ``tmp`` tile as a
third operand; a single scratch tile is reused across the ops in each program.

``and_``/``or_``/``shl``/``shr`` (and their scalar forms) are intentionally NOT
covered here: ptoas rejects ``pto.tand``/``tor``/``tshl``/``tshr`` with "invalid
kind of type specified" on a2a3 (assembler-side gap, tracked in KNOWN_ISSUES);
they can be added once ptoas supports them.

Inputs are kept non-negative so remainder is unambiguous. Each program runs
aligned (valid_shape == [M, N]) and narrow (valid_shape [VALID_M, VALID_N] <
[M, N]; invalid output stays zero).

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

REMS_RHS = 7
XORS_RHS = 0xFF


def _a_input() -> torch.Tensor:
    """Non-negative INT32 spread 0..255."""
    return torch.arange(M * N, dtype=torch.int32).reshape(M, N)


def _b_input() -> torch.Tensor:
    """Positive INT32 divisor in [1, 4] — non-zero for remainder."""
    return (torch.arange(M * N, dtype=torch.int32) % 4 + 1).reshape(M, N)


def _fill_valid(tensors: dict[str, torch.Tensor], fns: dict, valid: tuple[int, int] | None) -> None:
    """Write each output: aligned = full op result; narrow = op result in the
    valid sub-region only, zero elsewhere."""
    if valid:
        vm, vn = valid
        for name, fn in fns.items():
            out = torch.zeros_like(tensors[name])
            out[:vm, :vn] = fn()[:vm, :vn]
            tensors[name][:] = out
    else:
        for name, fn in fns.items():
            tensors[name][:] = fn()


# ---------------------------------------------------------------------------
# Tile-tile rem / xor (INT32, scratch tmp)
# ---------------------------------------------------------------------------


class BitwiseTileTestCase(PTOTestCase):
    """Tile-tile rem/xor on INT32, aligned or narrow valid_shape."""

    __test__ = False

    def __init__(self, *, valid_shapes: tuple[int, int] | None = None, config=None):
        super().__init__(config)
        self._valid = valid_shapes

    def get_name(self) -> str:
        return "tile_bitwise_tile_narrow" if self._valid else "tile_bitwise_tile"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.INT32, init_value=_a_input),
            TensorSpec("b", [M, N], DataType.INT32, init_value=_b_input),
            TensorSpec("rem_o", [M, N], DataType.INT32, is_output=True),
            TensorSpec("xor_o", [M, N], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        vshape = list(self._valid) if self._valid else [M, N]

        @pl.program
        class BitwiseTileProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, N], pl.INT32],
                b: pl.Tensor[[M, N], pl.INT32],
                rem_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                xor_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> tuple[pl.Tensor[[M, N], pl.INT32], pl.Tensor[[M, N], pl.INT32]]:
                a_tile = pl.load(a, [0, 0], [M, N], valid_shapes=vshape)
                b_tile = pl.load(b, [0, 0], [M, N], valid_shapes=vshape)
                tmp: pl.Tile[[M, N], pl.INT32] = pl.tile.create(
                    [M, N], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
                )
                rem_o = pl.store(pl.tile.rem(a_tile, b_tile, tmp), [0, 0], rem_o)
                xor_o = pl.store(pl.tile.xor(a_tile, b_tile, tmp), [0, 0], xor_o)
                return rem_o, xor_o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, N], pl.INT32],
                b: pl.Tensor[[M, N], pl.INT32],
                rem_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                xor_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> tuple[pl.Tensor[[M, N], pl.INT32], pl.Tensor[[M, N], pl.INT32]]:
                rem_o, xor_o = self.kernel(a, b, rem_o, xor_o)
                return rem_o, xor_o

        return BitwiseTileProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a, b = tensors["a"], tensors["b"]
        fns = {
            "rem_o": lambda: torch.remainder(a, b),
            "xor_o": lambda: torch.bitwise_xor(a, b),
        }
        _fill_valid(tensors, fns, self._valid)


# ---------------------------------------------------------------------------
# Tile-scalar rems / xors (INT32, scratch tmp)
# ---------------------------------------------------------------------------


class BitwiseScalarTestCase(PTOTestCase):
    """Tile-scalar rems/xors on INT32, aligned or narrow valid_shape."""

    __test__ = False

    def __init__(self, *, valid_shapes: tuple[int, int] | None = None, config=None):
        super().__init__(config)
        self._valid = valid_shapes

    def get_name(self) -> str:
        return "tile_bitwise_scalar_narrow" if self._valid else "tile_bitwise_scalar"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.INT32, init_value=_a_input),
            TensorSpec("rems_o", [M, N], DataType.INT32, is_output=True),
            TensorSpec("xors_o", [M, N], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        vshape = list(self._valid) if self._valid else [M, N]

        @pl.program
        class BitwiseScalarProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, N], pl.INT32],
                rems_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                xors_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> tuple[pl.Tensor[[M, N], pl.INT32], pl.Tensor[[M, N], pl.INT32]]:
                a_tile = pl.load(a, [0, 0], [M, N], valid_shapes=vshape)
                tmp: pl.Tile[[M, N], pl.INT32] = pl.tile.create(
                    [M, N], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
                )
                rems_o = pl.store(pl.tile.rems(a_tile, REMS_RHS, tmp), [0, 0], rems_o)
                xors_o = pl.store(pl.tile.xors(a_tile, XORS_RHS, tmp), [0, 0], xors_o)
                return rems_o, xors_o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, N], pl.INT32],
                rems_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                xors_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> tuple[pl.Tensor[[M, N], pl.INT32], pl.Tensor[[M, N], pl.INT32]]:
                rems_o, xors_o = self.kernel(a, rems_o, xors_o)
                return rems_o, xors_o

        return BitwiseScalarProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        fns = {
            "rems_o": lambda: torch.remainder(a, REMS_RHS),
            "xors_o": lambda: torch.bitwise_xor(a, XORS_RHS),
        }
        _fill_valid(tensors, fns, self._valid)


# ---------------------------------------------------------------------------
# Unary not_ (INT16)
# ---------------------------------------------------------------------------


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
        _fill_valid(tensors, {"not_o": lambda: torch.bitwise_not(a)}, self._valid)


class TestBitwise:
    """Tile-level integer rem / xor / not ops on a2a3."""

    @pytest.mark.platforms("a2a3")
    def test_tile_bitwise_tile(self, test_runner):
        result = test_runner.run(BitwiseTileTestCase())
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_bitwise_tile_narrow(self, test_runner):
        result = test_runner.run(BitwiseTileTestCase(valid_shapes=(VALID_M, VALID_N)))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_bitwise_scalar(self, test_runner):
        result = test_runner.run(BitwiseScalarTestCase())
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_bitwise_scalar_narrow(self, test_runner):
        result = test_runner.run(BitwiseScalarTestCase(valid_shapes=(VALID_M, VALID_N)))
        assert result.passed, f"Test failed: {result.error}"

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
