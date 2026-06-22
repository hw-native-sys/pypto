# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile-level integer bitwise / shift / remainder ops.

Tile-tile forms:  rem, and_, or_, shl, shr   (INT32)
Tile-scalar forms: rems, ands, ors, shls, shrs  (INT32)
Unary:             not_                       (INT16 — TNOT only supports int16/uint16)

Inputs are kept non-negative so remainder and right-shift are unambiguous, and
shift amounts stay in [1, 4]. Each program is exercised aligned (valid_shape ==
[M, N]) and narrow (valid_shape [VALID_M, VALID_N] < [M, N]; the invalid output
region stays zero).

(xor/xors are intentionally omitted: their 3-arg DSL form ``xor(a, b, tmp)``
mismatches codegen ``pto.txor`` which expects 2 arguments — known gap.)
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
ANDS_RHS = 0x0F
ORS_RHS = 0x10
SHLS_RHS = 2
SHRS_RHS = 1

# This test set targets a2a3 only; a5 coverage is handled in a separate PR.
A2A3 = [pytest.param("a2a3", id="a2a3")]


def _a_input() -> torch.Tensor:
    """Non-negative INT32 spread 0..255."""
    return torch.arange(M * N, dtype=torch.int32).reshape(M, N)


def _b_input() -> torch.Tensor:
    """Small positive INT32 in [1, 4] — safe shift amounts and non-zero divisors."""
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
# Tile-tile bitwise forms (INT32)
# ---------------------------------------------------------------------------


class BitwiseTileTestCase(PTOTestCase):
    """Tile-tile rem/and/or/shl/shr on INT32, aligned or narrow valid_shape."""

    __test__ = False

    def __init__(
        self, *, valid_shapes: tuple[int, int] | None = None, platform: str | None = None, config=None
    ):
        super().__init__(config, platform=platform)
        self._valid = valid_shapes

    def get_name(self) -> str:
        return "tile_bitwise_tile_narrow" if self._valid else "tile_bitwise_tile"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.INT32, init_value=_a_input()),
            TensorSpec("b", [M, N], DataType.INT32, init_value=_b_input()),
            TensorSpec("rem_o", [M, N], DataType.INT32, is_output=True),
            TensorSpec("and_o", [M, N], DataType.INT32, is_output=True),
            TensorSpec("or_o", [M, N], DataType.INT32, is_output=True),
            TensorSpec("shl_o", [M, N], DataType.INT32, is_output=True),
            TensorSpec("shr_o", [M, N], DataType.INT32, is_output=True),
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
                and_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                or_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                shl_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                shr_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> tuple[
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
            ]:
                a_tile = pl.load(a, [0, 0], [M, N], valid_shapes=vshape)
                b_tile = pl.load(b, [0, 0], [M, N], valid_shapes=vshape)
                rem_o = pl.store(pl.tile.rem(a_tile, b_tile), [0, 0], rem_o)
                and_o = pl.store(pl.tile.and_(a_tile, b_tile), [0, 0], and_o)
                or_o = pl.store(pl.tile.or_(a_tile, b_tile), [0, 0], or_o)
                shl_o = pl.store(pl.tile.shl(a_tile, b_tile), [0, 0], shl_o)
                shr_o = pl.store(pl.tile.shr(a_tile, b_tile), [0, 0], shr_o)
                return rem_o, and_o, or_o, shl_o, shr_o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, N], pl.INT32],
                b: pl.Tensor[[M, N], pl.INT32],
                rem_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                and_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                or_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                shl_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                shr_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> tuple[
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
            ]:
                rem_o, and_o, or_o, shl_o, shr_o = self.kernel(a, b, rem_o, and_o, or_o, shl_o, shr_o)
                return rem_o, and_o, or_o, shl_o, shr_o

        return BitwiseTileProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a, b = tensors["a"], tensors["b"]
        fns = {
            "rem_o": lambda: torch.remainder(a, b),
            "and_o": lambda: torch.bitwise_and(a, b),
            "or_o": lambda: torch.bitwise_or(a, b),
            "shl_o": lambda: torch.bitwise_left_shift(a, b),
            "shr_o": lambda: torch.bitwise_right_shift(a, b),
        }
        _fill_valid(tensors, fns, self._valid)


# ---------------------------------------------------------------------------
# Tile-scalar bitwise forms (INT32)
# ---------------------------------------------------------------------------


class BitwiseScalarTestCase(PTOTestCase):
    """Tile-scalar rems/ands/ors/shls/shrs on INT32, aligned or narrow valid_shape."""

    __test__ = False

    def __init__(
        self, *, valid_shapes: tuple[int, int] | None = None, platform: str | None = None, config=None
    ):
        super().__init__(config, platform=platform)
        self._valid = valid_shapes

    def get_name(self) -> str:
        return "tile_bitwise_scalar_narrow" if self._valid else "tile_bitwise_scalar"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.INT32, init_value=_a_input()),
            TensorSpec("rems_o", [M, N], DataType.INT32, is_output=True),
            TensorSpec("ands_o", [M, N], DataType.INT32, is_output=True),
            TensorSpec("ors_o", [M, N], DataType.INT32, is_output=True),
            TensorSpec("shls_o", [M, N], DataType.INT32, is_output=True),
            TensorSpec("shrs_o", [M, N], DataType.INT32, is_output=True),
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
                ands_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                ors_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                shls_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                shrs_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> tuple[
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
            ]:
                a_tile = pl.load(a, [0, 0], [M, N], valid_shapes=vshape)
                rems_o = pl.store(pl.tile.rems(a_tile, REMS_RHS), [0, 0], rems_o)
                ands_o = pl.store(pl.tile.ands(a_tile, ANDS_RHS), [0, 0], ands_o)
                ors_o = pl.store(pl.tile.ors(a_tile, ORS_RHS), [0, 0], ors_o)
                shls_o = pl.store(pl.tile.shls(a_tile, SHLS_RHS), [0, 0], shls_o)
                shrs_o = pl.store(pl.tile.shrs(a_tile, SHRS_RHS), [0, 0], shrs_o)
                return rems_o, ands_o, ors_o, shls_o, shrs_o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, N], pl.INT32],
                rems_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                ands_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                ors_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                shls_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
                shrs_o: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> tuple[
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
                pl.Tensor[[M, N], pl.INT32],
            ]:
                rems_o, ands_o, ors_o, shls_o, shrs_o = self.kernel(a, rems_o, ands_o, ors_o, shls_o, shrs_o)
                return rems_o, ands_o, ors_o, shls_o, shrs_o

        return BitwiseScalarProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        fns = {
            "rems_o": lambda: torch.remainder(a, REMS_RHS),
            "ands_o": lambda: torch.bitwise_and(a, ANDS_RHS),
            "ors_o": lambda: torch.bitwise_or(a, ORS_RHS),
            "shls_o": lambda: torch.bitwise_left_shift(a, SHLS_RHS),
            "shrs_o": lambda: torch.bitwise_right_shift(a, SHRS_RHS),
        }
        _fill_valid(tensors, fns, self._valid)


# ---------------------------------------------------------------------------
# Unary not_ (INT16)
# ---------------------------------------------------------------------------


class BitwiseNotTestCase(PTOTestCase):
    """Unary not_ on INT16, aligned or narrow valid_shape."""

    __test__ = False

    def __init__(
        self, *, valid_shapes: tuple[int, int] | None = None, platform: str | None = None, config=None
    ):
        super().__init__(config, platform=platform)
        self._valid = valid_shapes

    def get_name(self) -> str:
        return "tile_bitwise_not_narrow" if self._valid else "tile_bitwise_not"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.INT16, init_value=_a_input().to(torch.int16)),
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
    """Tile-level integer bitwise / shift / remainder ops on a2a3."""

    @pytest.mark.parametrize("platform", A2A3)
    def test_tile_bitwise_tile(self, test_runner, platform):
        result = test_runner.run(BitwiseTileTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", A2A3)
    def test_tile_bitwise_tile_narrow(self, test_runner, platform):
        result = test_runner.run(BitwiseTileTestCase(valid_shapes=(VALID_M, VALID_N), platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", A2A3)
    def test_tile_bitwise_scalar(self, test_runner, platform):
        result = test_runner.run(BitwiseScalarTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", A2A3)
    def test_tile_bitwise_scalar_narrow(self, test_runner, platform):
        result = test_runner.run(BitwiseScalarTestCase(valid_shapes=(VALID_M, VALID_N), platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", A2A3)
    def test_tile_bitwise_not(self, test_runner, platform):
        result = test_runner.run(BitwiseNotTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", A2A3)
    def test_tile_bitwise_not_narrow(self, test_runner, platform):
        result = test_runner.run(BitwiseNotTestCase(valid_shapes=(VALID_M, VALID_N), platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
