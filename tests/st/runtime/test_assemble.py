# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile.assemble (write a source tile into a target tile at a specified offset).

Hardware semantics (PTO backend):
  tile.assemble maps to TINSERT. The mode is inferred from operand memory spaces:

  Acc→Mat (TInsertMode::NZ):
    source: Acc (L0C), FP32, fractal layout  [output of tile.matmul]
    target: Mat (L1), FP32, fractal layout
    Data flow: a, b (GM) → Mat → Left/Right → matmul → Acc → TINSERT → Mat → Vec → GM

  Vec→Vec (TInsertMode::ND_VEC):
    source: Vec (UB), FP32, ND/RowMajor layout
    target: Vec (UB), FP32, ND/RowMajor layout
    Data flow: x, src (GM) → Vec → TINSERT → Vec → GM
"""

from typing import Any

import pytest
import torch
from examples.kernels.assemble import (
    TileAssembleAccMatProgram,
    TileAssembleDoubleLoopBroadcastProgram,
    TileAssembleDoubleLoopProgram,
    TileAssembleLoopColBroadcastProgram,
    TileAssembleRowByRowProgram,
    TileAssembleVecProgram,
)
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy

# ---------------------------------------------------------------------------
# Acc→Mat (NZ mode): matmul result assembled into a Mat target
# ---------------------------------------------------------------------------


class TileAssembleAccMatTestCase(PTOTestCase):
    """Acc→Mat: matmul(a[32,16], b[16,16]) assembled into the right half of x[32,32] at [0, 16]."""

    def get_name(self) -> str:
        return "tile_assemble_acc_mat"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 32], DataType.FP32, init_value=torch.rand),
            TensorSpec("a", [32, 16], DataType.FP32, init_value=torch.rand),
            TensorSpec("b", [16, 16], DataType.FP32, init_value=torch.rand),
            TensorSpec("y", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAssembleAccMatProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def compute_expected(self, tensors, params=None):
        # matmul(a, b) overwrites the right half; left half (columns 0..15) remains x (1.0)
        src = tensors["a"] @ tensors["b"]
        tensors["y"][:] = tensors["x"]
        tensors["y"][:, 16:] = src


# ---------------------------------------------------------------------------
# Vec→Vec single-shot (ND_VEC mode)
# ---------------------------------------------------------------------------


class TileAssembleVecTestCase(PTOTestCase):
    """Vec→Vec single-shot: src[32,16] assembled into the left half of x[32,32] at [0, 0]."""

    def get_name(self) -> str:
        return "tile_assemble_vec"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 32], DataType.FP32, init_value=torch.rand),
            TensorSpec("src", [32, 16], DataType.FP32, init_value=torch.rand),
            TensorSpec("y", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAssembleVecProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def compute_expected(self, tensors, params=None):
        tensors["y"][:] = tensors["x"]
        tensors["y"][:, :16] = tensors["src"]


# ---------------------------------------------------------------------------
# Vec→Vec single loop + pl.slice: dynamic row gather
# ---------------------------------------------------------------------------


class TileAssembleRowByRowTestCase(PTOTestCase):
    """Vec→Vec row-by-row: for each row i, pl.slice src[i,:] and assemble at [i, 0].

    Semantically equivalent to TileAssembleVecTestCase but exercises the
    loop + pl.slice + dynamic-offset assemble code path.
    """

    def get_name(self) -> str:
        return "tile_assemble_row_by_row"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 32], DataType.FP32, init_value=torch.rand),
            TensorSpec("src", [32, 16], DataType.FP32, init_value=torch.rand),
            TensorSpec("y", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAssembleRowByRowProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def compute_expected(self, tensors, params=None):
        tensors["y"][:] = tensors["x"]
        tensors["y"][:, :16] = tensors["src"]


# ---------------------------------------------------------------------------
# Vec→Vec nested loops + pl.slice: batch×head two-level index
# ---------------------------------------------------------------------------


class TileAssembleDoubleLoopTestCase(PTOTestCase):
    """Vec→Vec nested loops: outer b in range(4), inner i in range(8).

    Row index row = b*8+i; pl.slice src[row,:] assembled at [row, 0].
    Models the batch×head two-level indexing pattern in real workloads.
    """

    def get_name(self) -> str:
        return "tile_assemble_double_loop"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 32], DataType.FP32, init_value=torch.rand),
            TensorSpec("src", [32, 16], DataType.FP32, init_value=torch.rand),
            TensorSpec("y", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAssembleDoubleLoopProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def compute_expected(self, tensors, params=None):
        tensors["y"][:] = tensors["x"]
        tensors["y"][:, :16] = tensors["src"]


# ---------------------------------------------------------------------------
# Vec→Vec single loop, no pl.slice: dynamic column broadcast
# ---------------------------------------------------------------------------


class TileAssembleLoopColBroadcastTestCase(PTOTestCase):
    """Vec→Vec column broadcast: loop c in range(4), same src[32,8] assembled at [0, c*8].

    No pl.slice — the entire source is loaded once and written to each column-block.
    Result: all column-blocks of y equal src.
    """

    def get_name(self) -> str:
        return "tile_assemble_loop_col_broadcast"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 32], DataType.FP32, init_value=torch.rand),
            TensorSpec("src", [32, 8], DataType.FP32, init_value=torch.rand),
            TensorSpec("y", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAssembleLoopColBroadcastProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def compute_expected(self, tensors, params=None):
        for c in range(4):
            tensors["y"][:, c * 8 : (c + 1) * 8] = tensors["src"]


# ---------------------------------------------------------------------------
# Vec→Vec nested loops, no pl.slice: 2-D position broadcast
# ---------------------------------------------------------------------------


class TileAssembleDoubleLoopBroadcastTestCase(PTOTestCase):
    """Vec→Vec 2-D broadcast: nested b×c in range(2)×range(2), src[16,16] at [b*16, c*16].

    No pl.slice — same source tile fills all four [16,16] quadrants of y.
    Result: all quadrants of y equal src.
    """

    def get_name(self) -> str:
        return "tile_assemble_double_loop_broadcast"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 32], DataType.FP32, init_value=torch.rand),
            TensorSpec("src", [16, 16], DataType.FP32, init_value=torch.rand),
            TensorSpec("y", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAssembleDoubleLoopBroadcastProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def compute_expected(self, tensors, params=None):
        for b in range(2):
            for c in range(2):
                tensors["y"][b * 16 : (b + 1) * 16, c * 16 : (c + 1) * 16] = tensors["src"]


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------


# tile.assemble lowers to TINSERT, which is only available on Ascend 950.
@pytest.mark.platforms("a5", "a5sim")
class TestAssembleOperations:
    """Test suite for tile.assemble: one test per distinct pattern."""

    @pytest.mark.skip(reason="Codegen bug: MemRef not found in mapping for Acc→Mat assemble")
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_tile_assemble_acc_mat(self, test_runner, platform):
        """Acc→Mat (NZ mode): matmul result assembled into right half of Mat target."""
        result = test_runner.run(TileAssembleAccMatTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_tile_assemble_vec(self, test_runner, platform):
        """Vec→Vec single-shot (ND_VEC mode): src assembled into left half of target."""
        result = test_runner.run(TileAssembleVecTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.skip(
        reason="Sim bug: Vec→Vec assemble with pl.slice produces wrong output (496/1024 mismatch)"
    )
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_tile_assemble_row_by_row(self, test_runner, platform):
        """Vec→Vec single loop + pl.slice: dynamic row gather into left half."""
        result = test_runner.run(TileAssembleRowByRowTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.skip(
        reason="Sim bug: Vec→Vec assemble with pl.slice produces wrong output (496/1024 mismatch)"
    )
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_tile_assemble_double_loop(self, test_runner, platform):
        """Vec->Vec nested loops + pl.slice: batch x head two-level index (b*8+i)."""
        result = test_runner.run(TileAssembleDoubleLoopTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_tile_assemble_loop_col_broadcast(self, test_runner, platform):
        """Vec→Vec single loop, no pl.slice: same src column-block at each c*8 offset."""
        result = test_runner.run(TileAssembleLoopColBroadcastTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_tile_assemble_double_loop_broadcast(self, test_runner, platform):
        """Vec→Vec nested loops, no pl.slice: same src[16,16] fills all four quadrants."""
        result = test_runner.run(TileAssembleDoubleLoopBroadcastTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
