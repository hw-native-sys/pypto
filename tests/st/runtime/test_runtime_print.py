# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for pl.runtime_print() — verifies that pto.tprint codegen works
end-to-end without breaking kernel correctness.

runtime_print is a debugging utility that emits pto.tprint instructions.
These tests verify that inserting runtime_print into a kernel does NOT
affect the computed results (it is a pure side-effect operation).
"""

from typing import Any

import pypto.language as pl
import pytest
from harness.core.harness import DataType, PTOTestCase, TensorSpec

# =============================================================================
# Program definitions
# =============================================================================


@pl.program
class RuntimePrintTileProgram:
    """Element-wise add with a runtime_print of the intermediate tile."""

    @pl.function(type=pl.FunctionType.InCore)
    def tile_add_with_print(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
        tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
        tile_c: pl.Tile[[128, 128], pl.FP32] = pl.add(tile_a, tile_b)
        pl.runtime_print(tile_c)
        out_c = pl.store(tile_c, [0, 0], c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        out_c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_c = self.tile_add_with_print(a, b, out_c)
        return out_c


@pl.program
class RuntimePrintTensorProgram:
    """Element-wise add with a runtime_print of the input tensor."""

    @pl.function(type=pl.FunctionType.InCore)
    def tile_add_with_tensor_print(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        pl.runtime_print(a)
        tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
        tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
        tile_c: pl.Tile[[128, 128], pl.FP32] = pl.add(tile_a, tile_b)
        out_c = pl.store(tile_c, [0, 0], c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        out_c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_c = self.tile_add_with_tensor_print(a, b, out_c)
        return out_c


# =============================================================================
# Test cases
# =============================================================================


class RuntimePrintTileTestCase(PTOTestCase):
    """Test: element-wise add with pl.runtime_print(tile) after computation."""

    __test__ = False

    def get_name(self) -> str:
        return "runtime_print_tile_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RuntimePrintTileProgram

    def compute_expected(self, tensors, params=None):
        """Expected: c = a + b. runtime_print does not affect the result."""
        tensors["c"][:] = tensors["a"] + tensors["b"]


class RuntimePrintTensorTestCase(PTOTestCase):
    """Test: element-wise add with pl.runtime_print(tensor) on input."""

    __test__ = False

    def get_name(self) -> str:
        return "runtime_print_tensor_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RuntimePrintTensorProgram

    def compute_expected(self, tensors, params=None):
        """Expected: c = a + b. runtime_print does not affect the result."""
        tensors["c"][:] = tensors["a"] + tensors["b"]


# =============================================================================
# pytest test functions
# =============================================================================


class TestRuntimePrint:
    """Test suite for runtime_print — verifies codegen and correctness."""

    def test_runtime_print_tile(self, test_runner):
        """runtime_print(tile) should compile and run without affecting results."""
        test_case = RuntimePrintTileTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"runtime_print tile test failed: {result.error}"

    def test_runtime_print_tensor(self, test_runner):
        """runtime_print(tensor) should compile and run without affecting results."""
        test_case = RuntimePrintTensorTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"runtime_print tensor test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
