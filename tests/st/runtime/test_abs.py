# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Element-wise absolute value (abs) System Tests.

Covers both DSL layers exposed by issue #1138 (tensor.abs):

  TileAbsTest         : Tile level,   pl.tile.abs(t)      out = abs(a)
  TensorAbsTestFP16   : Tensor level, pl.abs(x)  FP16     out = abs(x)
  TensorAbsTestFP32   : Tensor level, pl.abs(x)  FP32     out = abs(x)
  TensorAbsTestLarge  : Tensor level, pl.abs(x)  FP32 64x128 — exercises chunked_loop split
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 16
N = 16


def _signed_input(shape: list[int]) -> torch.Tensor:
    """Random tensor centered around zero so abs has work to do."""
    return torch.randn(shape)


# ---------------------------------------------------------------------------
# Tile level: pl.tile.abs
# ---------------------------------------------------------------------------


@pl.program
class TileAbsProgram:
    """Tile-level absolute value."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        tile_a: pl.Tile[[M, N], pl.FP32] = pl.load(a, [0, 0], [M, N])
        tile_c: pl.Tile[[M, N], pl.FP32] = pl.tile.abs(tile_a)
        out = pl.store(tile_c, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        out = self.kernel(a, out)
        return out


class TileAbsTest(PTOTestCase):
    """Tile abs: out = abs(a)."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_abs"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.FP32, init_value=_signed_input([M, N])),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAbsProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.abs(tensors["a"])


# ---------------------------------------------------------------------------
# Tensor level: pl.abs (issue #1138)
# ---------------------------------------------------------------------------


@pl.program
class TensorAbsProgramFP16:
    """Tensor-level absolute value, FP16 (lowered to tile.abs by ConvertTensorToTileOps)."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        x: pl.Tensor[[M, N], pl.FP16],
        out: pl.Out[pl.Tensor[[M, N], pl.FP16]],
    ) -> pl.Tensor[[M, N], pl.FP16]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN),
        ):
            y = pl.abs(x)
            out = pl.assemble(out, y, [0, 0])
        return out


class TensorAbsTestFP16(PTOTestCase):
    """Tensor abs FP16: out = pl.abs(x). Issue #1138 requested BF16, but pto.tabs only
    supports f16/f32, so we use FP16 here."""

    __test__ = False

    def get_name(self) -> str:
        return "tensor_abs_fp16"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [M, N], DataType.FP16, init_value=_signed_input([M, N]).to(torch.float16)),
            TensorSpec("out", [M, N], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return TensorAbsProgramFP16

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.abs(tensors["x"])


@pl.program
class TensorAbsProgramFP32:
    """Tensor-level absolute value, FP32."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        x: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN),
        ):
            y = pl.abs(x)
            out = pl.assemble(out, y, [0, 0])
        return out


class TensorAbsTestFP32(PTOTestCase):
    """Tensor abs FP32: out = pl.abs(x)."""

    __test__ = False

    def get_name(self) -> str:
        return "tensor_abs_fp32"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [M, N], DataType.FP32, init_value=_signed_input([M, N])),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TensorAbsProgramFP32

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.abs(tensors["x"])


LARGE_M = 64
LARGE_N = 128


@pl.program
class TensorAbsProgramLarge:
    """Tensor-level abs on a larger shape (64x128) to exercise the chunked-loop split path."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        x: pl.Tensor[[LARGE_M, LARGE_N], pl.FP32],
        out: pl.Out[pl.Tensor[[LARGE_M, LARGE_N], pl.FP32]],
    ) -> pl.Tensor[[LARGE_M, LARGE_N], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN),
        ):
            y = pl.abs(x)
            out = pl.assemble(out, y, [0, 0])
        return out


class TensorAbsTestLarge(PTOTestCase):
    """Tensor abs on 64x128 FP32 — validates codegen under chunked-loop splitting."""

    __test__ = False

    def get_name(self) -> str:
        return "tensor_abs_large"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "x",
                [LARGE_M, LARGE_N],
                DataType.FP32,
                init_value=_signed_input([LARGE_M, LARGE_N]),
            ),
            TensorSpec("out", [LARGE_M, LARGE_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TensorAbsProgramLarge

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.abs(tensors["x"])


# ---------------------------------------------------------------------------
# pytest wrappers
# ---------------------------------------------------------------------------


class TestAbs:
    """End-to-end abs tests at both Tile and Tensor DSL levels."""

    def test_tile_abs(self, test_runner):
        """Tile-level pl.tile.abs."""
        result = test_runner.run(TileAbsTest())
        assert result.passed, f"Tile abs failed: {result.error}"

    def test_tensor_abs_fp16(self, test_runner):
        """Tensor-level pl.abs, FP16 (issue #1138)."""
        result = test_runner.run(TensorAbsTestFP16())
        assert result.passed, f"Tensor abs FP16 failed: {result.error}"

    def test_tensor_abs_fp32(self, test_runner):
        """Tensor-level pl.abs, FP32."""
        result = test_runner.run(TensorAbsTestFP32())
        assert result.passed, f"Tensor abs FP32 failed: {result.error}"

    def test_tensor_abs_large(self, test_runner):
        """Tensor-level pl.abs on 64x128 — exercises chunked-loop split."""
        result = test_runner.run(TensorAbsTestLarge())
        assert result.passed, f"Tensor abs large failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
