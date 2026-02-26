# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end test for orchestration function CCE codegen.

This test verifies the complete compilation pipeline for an orchestration program
implementing the formula: f = (a + b + 1)(a + b + 2)

Task Graph:
  task0: c = a + b          (kernel_add, func_id=0)
  task1: d = c + 1          (kernel_add_scalar, func_id=1)
  task2: e = c + 2          (kernel_add_scalar, func_id=1)
  task3: f = d * e          (kernel_mul, func_id=2)

Dependencies: t0→t1, t0→t2, t1→t3, t2→t3
"""

from typing import Any

import pypto.language as pl
import pytest
from harness.core.harness import DataType, PTOTestCase, TensorSpec


class TestAddMulOrchestration(PTOTestCase):
    """Test case for orchestration function with multiple InCore kernels.

    Implements formula: f = (a + b + 1)(a + b + 2)

    Task graph:
      - kernel_add: c = a + b
      - kernel_add_scalar: d = c + 1
      - kernel_add_scalar: e = c + 2
      - kernel_mul: f = d * e
    """

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "add_mul_orchestration"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [16, 16], DataType.FP32, init_value=2.0),
            TensorSpec("b", [16, 16], DataType.FP32, init_value=3.0),
            TensorSpec("output", [16, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class ExampleOrchProgram:
            """Example orchestration program with InCore kernels."""

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                """Adds two tensors element-wise: result = a + b"""
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                output_new: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
                return output_new

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add_scalar(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                scalar: pl.Scalar[pl.FP32],
                output: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                """Adds a scalar to each element: result = a + scalar"""
                x: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(x, scalar)
                output_new: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
                return output_new

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_mul(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                """Multiplies two tensors element-wise: result = a * b"""
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.mul(a_tile, b_tile)
                output_new: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
                return output_new

            @pl.function(type=pl.FunctionType.Orchestration)
            def BuildExampleGraph(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                """Build BuildExampleGraph orchestration function.

                Orchestration function for formula: f = (a + b + 1)(a + b + 2)
                Uses load/store pattern: InCore kernels take input + output tensors.

                Calls InCore functions to build the task graph:
                  - task0: c = a + b (kernel_add writes to c)
                  - task1: d = c + 1 (kernel_add_scalar writes to d)
                  - task2: e = c + 2 (kernel_add_scalar writes to e)
                  - task3: f = d * e (kernel_mul writes to f)

                Args:
                    a: Input tensor A
                    b: Input tensor B

                Returns:
                    Final result tensor
                """
                # Task 0: c = a + b (call kernel_add with output buffer c)
                c: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)

                # Task 1: d = c + 1 (call kernel_add_scalar with output buffer d)
                d: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add_scalar(c, 1.0)  # type: ignore[reportArgumentType]

                # Task 2: e = c + 2 (call kernel_add_scalar with output buffer e)
                e: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add_scalar(c, 2.0)  # type: ignore[reportArgumentType]

                # Task 3: f = d * e (call kernel_mul with output buffer)
                f_result: pl.Tensor[[16, 16], pl.FP32] = self.kernel_mul(d, e)
                return f_result

        return ExampleOrchProgram

    def compute_expected(self, tensors, params=None):
        """Compute expected output: f = (a + b + 1)(a + b + 2)"""
        a = tensors["a"]
        b = tensors["b"]
        c = a + b
        d = c + 1.0
        e = c + 2.0
        tensors["output"][:] = d * e


# =============================================================================
# pytest test suite
# =============================================================================


class TestOrchestrationCodegen:
    """Test suite for orchestration codegen."""

    def test_add_mul_orch_cce_codegen(self, test_runner):
        """Test end-to-end CCE codegen for orchestration function.

        Verifies that:
        - IR program is built successfully with 4 functions (3 InCore + 1 Orchestration)
        - Compilation with PassManager and CCECodegen completes
        - Output directory is created
        - Required files are generated (orchestration and kernel files)
        - Generated files are not empty
        """
        test_case = TestAddMulOrchestration()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
