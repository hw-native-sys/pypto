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
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

from examples.language.beginner.assemble import (
    TileAddThenAssembleRightOffsetProgram,
    TileAddThenAssembleZeroOffsetProgram,
    TileAssembleRightOffsetProgram,
    TileAssembleVecRightOffsetProgram,
    TileAssembleVecZeroOffsetProgram,
    TileAssembleZeroOffsetProgram,
)


class TileAssembleZeroOffsetTestCase(PTOTestCase):
    """Test case for tile.assemble at [0, 0]: matmul(a,b)[32x16] overwrites the left half of x[32x32]."""

    def get_name(self) -> str:
        return "tile_assemble_zero_offset"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 32], DataType.FP32, init_value=1.0),
            TensorSpec("a", [32, 16], DataType.FP32, init_value=1.0),
            TensorSpec("b", [16, 16], DataType.FP32, init_value=1.0),
            TensorSpec("y", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAssembleZeroOffsetProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def compute_expected(self, tensors, params=None):
        # assemble at [0, 0]: matmul(a, b) overwrites columns 0..15; columns 16..31 remain x (1.0)
        src = tensors["a"] @ tensors["b"]
        tensors["y"][:] = tensors["x"]
        tensors["y"][:, :16] = src


class TileAssembleRightOffsetTestCase(PTOTestCase):
    """Test case for tile.assemble at [0, 16]: matmul(a,b)[32x16] overwrites the right half of x[32x32]."""

    def get_name(self) -> str:
        return "tile_assemble_right_offset"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 32], DataType.FP32, init_value=1.0),
            TensorSpec("a", [32, 16], DataType.FP32, init_value=1.0),
            TensorSpec("b", [16, 16], DataType.FP32, init_value=1.0),
            TensorSpec("y", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAssembleRightOffsetProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def compute_expected(self, tensors, params=None):
        # assemble at [0, 16]: columns 0..15 remain x (1.0); matmul(a, b) overwrites columns 16..31
        src = tensors["a"] @ tensors["b"]
        tensors["y"][:] = tensors["x"]
        tensors["y"][:, 16:] = src


@pytest.mark.skip(reason="Pending adaptation for A5")
class TestAssembleOperations:
    """Test suite for tile.assemble operations."""

    def test_tile_assemble_zero_offset(self, test_runner):
        """Test tile.assemble: write matmul result into left half of target at offset [0, 0]."""
        test_case = TileAssembleZeroOffsetTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_assemble_right_offset(self, test_runner):
        """Test tile.assemble: write matmul result into right half of target at offset [0, 16]."""
        test_case = TileAssembleRightOffsetTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


class TileAssembleVecZeroOffsetTestCase(PTOTestCase):
    """Test case for Vec-to-Vec tile.assemble at [0, 0]: src[32x16] overwrites the left half of x[32x32]."""

    def get_name(self) -> str:
        return "tile_assemble_vec_zero_offset"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 32], DataType.FP32, init_value=1.0),
            TensorSpec("src", [32, 16], DataType.FP32, init_value=2.0),
            TensorSpec("y", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAssembleVecZeroOffsetProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def compute_expected(self, tensors, params=None):
        # assemble at [0, 0]: src overwrites columns 0..15; columns 16..31 remain x (1.0)
        tensors["y"][:] = tensors["x"]
        tensors["y"][:, :16] = tensors["src"]


class TileAssembleVecRightOffsetTestCase(PTOTestCase):
    """Test case for Vec-to-Vec tile.assemble at [0, 16]: src[32x16] overwrites the right half of x[32x32]."""

    def get_name(self) -> str:
        return "tile_assemble_vec_right_offset"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 32], DataType.FP32, init_value=1.0),
            TensorSpec("src", [32, 16], DataType.FP32, init_value=2.0),
            TensorSpec("y", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAssembleVecRightOffsetProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def compute_expected(self, tensors, params=None):
        # assemble at [0, 16]: columns 0..15 remain x (1.0); src overwrites columns 16..31
        tensors["y"][:] = tensors["x"]
        tensors["y"][:, 16:] = tensors["src"]


@pytest.mark.skip(reason="Pending adaptation for A5")
class TestVecAssembleOperations:
    """Test suite for UB-to-UB (Vec→Vec) tile.assemble operations (TInsertMode::ND_VEC)."""

    def test_tile_assemble_vec_zero_offset(self, test_runner):
        """Test Vec tile.assemble: write src into left half of target at offset [0, 0]."""
        test_case = TileAssembleVecZeroOffsetTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_assemble_vec_right_offset(self, test_runner):
        """Test Vec tile.assemble: write src into right half of target at offset [0, 16]."""
        test_case = TileAssembleVecRightOffsetTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


class TileAddThenAssembleZeroOffsetTestCase(PTOTestCase):
    """Test case for add-then-assemble at [0, 0].

    add(src, delta)[32x16] overwrites the left half of x[32x32].
    """

    def get_name(self) -> str:
        return "tile_add_then_assemble_zero_offset"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 32], DataType.FP32, init_value=1.0),
            TensorSpec("src", [32, 16], DataType.FP32, init_value=2.0),
            TensorSpec("delta", [32, 16], DataType.FP32, init_value=3.0),
            TensorSpec("y", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAddThenAssembleZeroOffsetProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def compute_expected(self, tensors, params=None):
        # add then assemble at [0, 0]: (src + delta) overwrites columns 0..15; columns 16..31 remain x (1.0)
        tensors["y"][:] = tensors["x"]
        tensors["y"][:, :16] = tensors["src"] + tensors["delta"]


class TileAddThenAssembleRightOffsetTestCase(PTOTestCase):
    """Test case for add-then-assemble at [0, 16].

    add(src, delta)[32x16] overwrites the right half of x[32x32].
    """

    def get_name(self) -> str:
        return "tile_add_then_assemble_right_offset"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 32], DataType.FP32, init_value=1.0),
            TensorSpec("src", [32, 16], DataType.FP32, init_value=2.0),
            TensorSpec("delta", [32, 16], DataType.FP32, init_value=3.0),
            TensorSpec("y", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileAddThenAssembleRightOffsetProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def compute_expected(self, tensors, params=None):
        # add then assemble at [0, 16]: columns 0..15 remain x (1.0); (src + delta) overwrites columns 16..31
        tensors["y"][:] = tensors["x"]
        tensors["y"][:, 16:] = tensors["src"] + tensors["delta"]


@pytest.mark.skip(reason="Pending adaptation for A5")
class TestAddThenAssembleOperations:
    """Test suite for add-then-assemble: pl.add on source tile before Vec→Vec tile.assemble."""

    def test_tile_add_then_assemble_zero_offset(self, test_runner):
        """Test add-then-assemble: write add(src, delta) into left half of target at offset [0, 0]."""
        test_case = TileAddThenAssembleZeroOffsetTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_add_then_assemble_right_offset(self, test_runner):
        """Test add-then-assemble: write add(src, delta) into right half of target at offset [0, 16]."""
        test_case = TileAddThenAssembleRightOffsetTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
