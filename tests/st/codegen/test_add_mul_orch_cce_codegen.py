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

import os

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.backend import BackendType


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
            c: Temp buffer for a + b
            d: Temp buffer for c + 1
            e: Temp buffer for c + 2
            output: Final output buffer for d * e

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


def build_example_orch_program(dtype: DataType = DataType.FP32):
    """Build the complete example_orch program.

    Creates a program with:
      - 3 InCore functions (kernel_add, kernel_add_scalar, kernel_mul)
      - 1 Orchestration function (BuildExampleGraph)

    Args:
        dtype: Data type for tensors (currently only FP32 supported)

    Returns:
        Program object
    """
    if dtype != DataType.FP32:
        raise ValueError(f"Only FP32 is currently supported, got {dtype}")

    return ExampleOrchProgram


def test_add_mul_orch_cce_codegen(tmp_path):
    """Test end-to-end CCE codegen for orchestration function.

    Verifies that:
    - IR program is built successfully
    - Compilation with PassManager and CCECodegen completes
    - Output directory is created
    - Required files are generated (orchestration and kernel files)
    - Generated files are not empty
    """
    # Build IR program
    dtype = DataType.FP32
    program = build_example_orch_program(dtype)

    # Verify program structure
    assert program is not None, "Program should be created"
    assert len(program.functions) == 4, "Program should have 4 functions"
    function_names = [f.name for f in program.functions.values()]
    assert "kernel_add" in function_names, "Should have kernel_add function"
    assert "kernel_add_scalar" in function_names, "Should have kernel_add_scalar function"
    assert "kernel_mul" in function_names, "Should have kernel_mul function"
    assert "BuildExampleGraph" in function_names, "Should have BuildExampleGraph function"

    # Compile with ir.compile API
    output_dir = ir.compile(
        program,
        output_dir=str(tmp_path / "output"),
        strategy=ir.OptimizationStrategy.Default,
        dump_passes=True,
        backend_type=BackendType.CCE,
    )

    # Verify output directory exists
    assert os.path.exists(output_dir), "Output directory should exist"
    assert os.path.isdir(output_dir), "Output path should be a directory"

    # Verify orchestration file exists
    orch_file = os.path.join(output_dir, "orchestration", "BuildExampleGraph.cpp")
    assert os.path.exists(orch_file), "Orchestration file BuildExampleGraph.cpp should exist"
    assert os.path.getsize(orch_file) > 0, "Orchestration file should not be empty"

    # Verify kernel files exist
    kernel_names = ["kernel_add", "kernel_add_scalar", "kernel_mul"]
    for kernel_name in kernel_names:
        kernel_file = os.path.join(output_dir, "kernels", "aiv", f"{kernel_name}.cpp")
        assert os.path.exists(kernel_file), f"Kernel file {kernel_name}.cpp should exist"
        assert os.path.getsize(kernel_file) > 0, f"Kernel file {kernel_name}.cpp should not be empty"

    # Verify passes_dump directory exists
    passes_dump_dir = os.path.join(output_dir, "passes_dump")
    assert os.path.exists(passes_dump_dir), "Passes dump directory should exist"
    assert os.path.isdir(passes_dump_dir), "Passes dump path should be a directory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
