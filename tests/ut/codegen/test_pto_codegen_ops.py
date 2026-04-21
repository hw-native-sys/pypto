# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
910B PTO Backend: Block-level Operations Codegen Test.

This test validates code generation for all supported tile-level operations
(Tile-Tile and Tile-Scalar) in the 910B PTO backend. It creates kernels for
each operation type, compiles them through the PassManager and PTOCodegen,
and verifies the generated orchestration code.
"""

import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

# ============================================================================
# Operation to PTO API Mapping
# ============================================================================

# Mapping from kernel operation name to expected PTO API call
BINARY_TILE_TILE_OPS = {
    "add": "pto.tadd",
    "sub": "pto.tsub",
    "mul": "pto.tmul",
    "div": "pto.tdiv",
    "maximum": "pto.tmax",
    "minimum": "pto.tmin",
}

UNARY_TILE_OPS = {
    "neg": "pto.tneg",
    "exp": "pto.texp",
    "sqrt": "pto.tsqrt",
    "rsqrt": "pto.trsqrt",
    "recip": "pto.trecip",
    "log": "pto.tlog",
    "abs": "pto.tabs",
    "relu": "pto.trelu",
}

TILE_SCALAR_OPS = {
    "adds": "pto.tadds",
    "subs": "pto.tsubs",
    "muls": "pto.tmuls",
    "divs": "pto.tdivs",
}

COMPARISON_OPS = {
    "cmp": "pto.tcmp",
}

MATMUL_OPS = {
    "matmul": "pto.tmatmul",
    "matmul_acc": "pto.tmatmul.acc",
}


# ============================================================================
# Helper Functions for Validation
# ============================================================================


def get_operation_category(kernel_name: str) -> str:
    """Determine operation category from kernel name.

    Args:
        kernel_name: Kernel function name (e.g., "kernel_add", "kernel_neg").

    Returns:
        Operation category: "binary_tile_tile", "unary_tile", "tile_scalar", "comparison", or "matmul".

    Raises:
        ValueError: If operation is not recognized.
    """
    # Remove "kernel_" prefix to get operation name
    if not kernel_name.startswith("kernel_"):
        raise ValueError(f"Invalid kernel name format: {kernel_name}")

    op_name = kernel_name[7:]  # Strip "kernel_" prefix

    if op_name in BINARY_TILE_TILE_OPS:
        return "binary_tile_tile"
    elif op_name in UNARY_TILE_OPS:
        return "unary_tile"
    elif op_name in TILE_SCALAR_OPS:
        return "tile_scalar"
    elif op_name in COMPARISON_OPS:
        return "comparison"
    elif op_name in MATMUL_OPS:
        return "matmul"
    else:
        raise ValueError(f"Unknown operation: {op_name}")


def get_expected_pto_api(kernel_name: str) -> str:
    """Get expected PTO API call for a kernel.

    Args:
        kernel_name: Kernel function name (e.g., "kernel_add", "kernel_neg").

    Returns:
        Expected PTO API name (e.g., "pto.tadd", "pto.tneg").

    Raises:
        ValueError: If operation is not recognized.
    """
    op_name = kernel_name[7:]  # Strip "kernel_" prefix

    # Check all operation mappings
    all_ops = {**BINARY_TILE_TILE_OPS, **UNARY_TILE_OPS, **TILE_SCALAR_OPS, **COMPARISON_OPS, **MATMUL_OPS}

    if op_name not in all_ops:
        raise ValueError(f"Unknown operation: {op_name}")

    return all_ops[op_name]


def validate_kernel_codegen(kernel_name: str, mlir_code: str) -> None:
    """Validate that kernel generates correct PTO API calls.

    Args:
        kernel_name: Kernel function name (e.g., "kernel_add").
        mlir_code: Generated MLIR code string.

    Raises:
        AssertionError: If validation fails.
    """
    category = get_operation_category(kernel_name)
    expected_api = get_expected_pto_api(kernel_name)

    # Validate expected PTO API is present
    assert expected_api in mlir_code, f"Kernel {kernel_name} should generate {expected_api} call"

    # Validate memory operations are present
    assert "pto.tload" in mlir_code, f"Kernel {kernel_name} should contain pto.tload operation"
    assert "pto.tstore" in mlir_code, f"Kernel {kernel_name} should contain pto.tstore operation"
    assert "pto.partition_view" in mlir_code, (
        f"Kernel {kernel_name} should contain pto.partition_view operation"
    )

    # Category-specific validations
    if category == "binary_tile_tile":
        # Binary ops should load two tiles
        tload_count = mlir_code.count("pto.tload")
        assert tload_count >= 2, (
            f"Binary tile-tile op {kernel_name} should have at least 2 tload operations, got {tload_count}"
        )

    elif category == "unary_tile":
        # Unary ops should load one tile
        tload_count = mlir_code.count("pto.tload")
        assert tload_count >= 1, (
            f"Unary tile op {kernel_name} should have at least 1 tload operation, got {tload_count}"
        )

    elif category == "tile_scalar":
        # Tile-scalar ops should load one tile (scalar is a parameter)
        tload_count = mlir_code.count("pto.tload")
        assert tload_count >= 1, (
            f"Tile-scalar op {kernel_name} should have at least 1 tload operation, got {tload_count}"
        )

    elif category == "comparison":
        # Comparison ops should load two tiles
        tload_count = mlir_code.count("pto.tload")
        assert tload_count >= 2, (
            f"Comparison op {kernel_name} should have at least 2 tload operations, got {tload_count}"
        )


@pl.program
class BlockOperationsTest:
    """Test program containing kernels for all supported tile-level operations."""

    # Tile-Tile Binary Operations

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add(
        self,
        lhs: pl.Tensor[[16, 16], pl.FP32],
        rhs: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise addition: output = lhs + rhs."""
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, 0], [16, 16])
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.add(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_sub(
        self,
        lhs: pl.Tensor[[16, 16], pl.FP32],
        rhs: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise subtraction: output = lhs - rhs."""
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, 0], [16, 16])
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.sub(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_mul(
        self,
        lhs: pl.Tensor[[16, 16], pl.FP32],
        rhs: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise multiplication: output = lhs * rhs."""
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, 0], [16, 16])
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.mul(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_div(
        self,
        lhs: pl.Tensor[[16, 16], pl.FP32],
        rhs: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise division: output = lhs / rhs."""
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, 0], [16, 16])
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.div(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    # Tile-Scalar Binary Operations

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_adds(
        self,
        tensor: pl.Tensor[[16, 16], pl.FP32],
        scalar: pl.Scalar[pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise scalar addition: output = tensor + scalar."""
        tensor_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.adds(tensor_tile, scalar)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_subs(
        self,
        tensor: pl.Tensor[[16, 16], pl.FP32],
        scalar: pl.Scalar[pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise scalar subtraction: output = tensor - scalar."""
        tensor_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.subs(tensor_tile, scalar)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_muls(
        self,
        tensor: pl.Tensor[[16, 16], pl.FP32],
        scalar: pl.Scalar[pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise scalar multiplication: output = tensor * scalar."""
        tensor_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.muls(tensor_tile, scalar)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_divs(
        self,
        tensor: pl.Tensor[[16, 16], pl.FP32],
        scalar: pl.Scalar[pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise scalar division: output = tensor / scalar."""
        tensor_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.divs(tensor_tile, scalar)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    # Unary Operations

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_neg(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise negation: output = -input."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.neg(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_exp(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise exponential: output = exp(input)."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.exp(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_sqrt(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise square root: output = sqrt(input)."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.sqrt(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_rsqrt(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise reciprocal square root: output = 1/sqrt(input)."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.rsqrt(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_recip(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise reciprocal: output = 1/input."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.recip(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_log(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise natural logarithm: output = log(input)."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.log(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_abs(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise absolute value: output = abs(input)."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.abs(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_relu(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise ReLU: output = max(0, input)."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.relu(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    # Comparison Operations

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_maximum(
        self,
        lhs: pl.Tensor[[16, 16], pl.FP32],
        rhs: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise maximum: output = max(lhs, rhs)."""
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, 0], [16, 16])
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.maximum(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_minimum(
        self,
        lhs: pl.Tensor[[16, 16], pl.FP32],
        rhs: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise minimum: output = min(lhs, rhs)."""
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, 0], [16, 16])
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.minimum(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_cmp(
        self,
        lhs: pl.Tensor[[16, 16], pl.FP32],
        rhs: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise comparison: output = cmp(lhs, rhs)."""
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, 0], [16, 16])
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.cmp(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_matmul(
        self,
        lhs: pl.Tensor[[16, 16], pl.FP32],
        rhs: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Matmul: output = matmul(lhs, rhs)."""
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
            lhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
        )
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
            rhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
        )
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.matmul(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_matmul_acc(
        self,
        lhs: pl.Tensor[[16, 16], pl.FP32],
        rhs: pl.Tensor[[16, 16], pl.FP32],
        factor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Matmul_acc: output = matmul_acc(factor, lhs, rhs)."""
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
            lhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
        )
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
            rhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
        )
        factor_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
            factor, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
        )
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.matmul_acc(factor_tile, lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
        return updated_output


def build_tile_ops_test_program(dtype: DataType = DataType.FP32):
    """Build the tile operations test program.

    Args:
        dtype: Data type for tensors (currently only FP32 is supported).

    Returns:
        BlockOperationsTest program containing all tile-level operation kernels.

    Raises:
        ValueError: If dtype is not FP32.
    """
    if dtype != DataType.FP32:
        raise ValueError(f"Only FP32 is currently supported, got {dtype}")
    return BlockOperationsTest


class Test910BBlockOpsCodegen:
    """Tests for 910B PTO backend tile-level operations code generation."""

    def test_tile_ops_codegen(self):
        """Test code generation for all tile-level operations."""
        # Set backend type for testing
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        dtype = DataType.FP32

        # Build IR program
        program = build_tile_ops_test_program(dtype)

        # Validate program structure
        assert program is not None, "Program should not be None"
        assert hasattr(program, "functions"), "Program should have functions attribute"
        assert len(program.functions) > 0, "Program should have at least one function"

        # Collect function names for validation
        function_names = [func.name for func in program.functions.values()]

        # Validate that all functions start with "kernel_" prefix
        for func_name in function_names:
            assert func_name.startswith("kernel_"), f"Function {func_name} should start with 'kernel_' prefix"

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized_program = pm.run_passes(program)

        # Generate PTO MLIR code for each function individually
        codegen_instance = codegen.PTOCodegen()

        for func in optimized_program.functions.values():
            func_name = func.name

            # Create a single-function program for code generation
            single_func_program = ir.Program([func], func_name, optimized_program.span)

            # Generate MLIR code for this function
            mlir_code = codegen_instance.generate(single_func_program)

            # Validate that MLIR code was generated
            assert mlir_code is not None, f"MLIR code should be generated for {func_name}"
            assert len(mlir_code) > 0, f"MLIR code for {func_name} should not be empty"

            # Validate kernel codegen using abstract validation
            validate_kernel_codegen(func_name, mlir_code)


class TestRsqrtHighPrecisionCodegen:
    """Tests for the high-precision path of tile.rsqrt (2-arg form)."""

    def _generate_mlir(self, program_cls) -> str:
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program_cls)
        codegen_instance = codegen.PTOCodegen()
        funcs = list(optimized.functions.values())
        assert funcs, "Program has no functions"
        single = ir.Program([funcs[0]], funcs[0].name, optimized.span)
        return codegen_instance.generate(single)

    def test_tile_rsqrt_with_tmp_emits_two_operand_trsqrt(self):
        """pl.tile.rsqrt(src, tmp=...) emits pto.trsqrt with two ins operands."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_rsqrt_hp(
                self,
                input_tensor: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
                tmp_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.create(
                    [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                result_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.rsqrt(input_tile, tmp=tmp_tile)
                updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], output)
                return updated_output

        mlir = self._generate_mlir(Prog)
        assert "pto.trsqrt" in mlir
        # Two ins operands separated by a comma must appear inside a single ins(...) clause.
        trsqrt_line = next((line for line in mlir.splitlines() if "pto.trsqrt" in line), "")
        assert trsqrt_line, f"pto.trsqrt not found in MLIR:\n{mlir}"
        ins_start = trsqrt_line.find("ins(")
        assert ins_start != -1, f"ins(...) clause not found in: {trsqrt_line}"
        ins_end = trsqrt_line.find(")", ins_start)
        ins_body = trsqrt_line[ins_start + len("ins(") : ins_end]
        # 2-operand form: "<operand1>, <operand2> : ..." — check for a separator before the ":".
        operand_str = ins_body.split(":", 1)[0]
        assert operand_str.count(",") >= 1, (
            f"Expected 2 ins operands for high-precision pto.trsqrt, got: {trsqrt_line}"
        )

    def test_tensor_rsqrt_high_precision_end_to_end(self):
        """pl.rsqrt(x, high_precision=True) at the tensor level also emits the 2-operand form."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_rsqrt_hp_tensor(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                y: pl.Tensor[[16, 16], pl.FP32] = pl.rsqrt(x, high_precision=True)
                return y

        mlir = self._generate_mlir(Prog)
        assert "pto.trsqrt" in mlir
        trsqrt_line = next((line for line in mlir.splitlines() if "pto.trsqrt" in line), "")
        ins_start = trsqrt_line.find("ins(")
        ins_end = trsqrt_line.find(")", ins_start)
        ins_body = trsqrt_line[ins_start + len("ins(") : ins_end]
        operand_str = ins_body.split(":", 1)[0]
        assert operand_str.count(",") >= 1, (
            f"Expected 2 ins operands for tensor-level high_precision rsqrt, got: {trsqrt_line}"
        )


class TestTileReadWriteOffsetCodegen:
    """Tests verifying tile.read/write multi-dimensional indices generate correct flat offsets."""

    def _generate_mlir(self, program_cls) -> str:
        """Run PassManager and PTOCodegen on the given program, return MLIR string."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program_cls)
        codegen_instance = codegen.PTOCodegen()
        funcs = list(optimized.functions.values())
        assert funcs, "Program has no functions"
        single = ir.Program([funcs[0]], funcs[0].name, optimized.span)
        return codegen_instance.generate(single)

    def test_tile_read_constant_1d(self):
        """1D-like slice of 2D tile [1, 16], tile.read(t, [0, 3]) -> flat offset 3 -> pto.tgetval."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                dst: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(src, [0, 0], [16, 16])
                val: pl.Scalar[pl.FP32] = pl.tile.read(t, [0, 3])
                pl.tile.write(t, [0, 0], val)
                return pl.store(t, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.tgetval" in mlir
        assert "3" in mlir

    def test_tile_read_constant_2d(self):
        """2D tile [4, 8], tile.read(t, [1, 3]) -> flat offset 11 -> pto.tgetval."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[4, 8], pl.FP32],
                dst: pl.Tensor[[4, 8], pl.FP32],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                t: pl.Tile[[4, 8], pl.FP32] = pl.load(src, [0, 0], [4, 8])
                val: pl.Scalar[pl.FP32] = pl.tile.read(t, [1, 3])
                pl.tile.write(t, [0, 0], val)
                return pl.store(t, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.tgetval" in mlir
        assert "11" in mlir

    def test_tile_write_constant_2d(self):
        """2D tile [4, 8], tile.write(t, [1, 3], val) -> flat offset 11 -> pto.tsetval."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[4, 8], pl.FP32],
                dst: pl.Tensor[[4, 8], pl.FP32],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                t: pl.Tile[[4, 8], pl.FP32] = pl.load(src, [0, 0], [4, 8])
                val: pl.Scalar[pl.FP32] = pl.tile.read(t, [0, 0])
                pl.tile.write(t, [1, 3], val)
                return pl.store(t, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.tsetval" in mlir
        assert "11" in mlir


class TestBroadcastOpsCodegen:
    """Tests for broadcast (expand) operations PTO code generation."""

    def _generate_mlir(self, program_cls) -> str:
        """Run PassManager and PTOCodegen on the given program, return MLIR string."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program_cls)
        codegen_instance = codegen.PTOCodegen()
        funcs = list(optimized.functions.values())
        assert funcs, "Program has no functions"
        single = ir.Program([funcs[0]], funcs[0].name, optimized.span)
        return codegen_instance.generate(single)

    def test_col_expand_mul_codegen(self):
        """tile.col_expand_mul(tile[M,N], col_vec[1,N]) should generate pto.tcolexpandmul."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                col_vec_tensor: pl.Tensor[[1, 16], pl.FP32],
                dst: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                src_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(src, [0, 0], [16, 16])
                col_tile: pl.Tile[[1, 16], pl.FP32] = pl.load(col_vec_tensor, [0, 0], [1, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.col_expand_mul(src_tile, col_tile)
                return pl.store(result, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.tcolexpandmul" in mlir, f"col_expand_mul should generate pto.tcolexpandmul, got:\n{mlir}"

    def test_col_expand_codegen(self):
        """tile.col_expand(target, col_vec) should emit pto.tcolexpand with only col_vec in ins()."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                col_vec_tensor: pl.Tensor[[1, 16], pl.FP32],
                dst: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                src_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(src, [0, 0], [16, 16])
                col_tile: pl.Tile[[1, 16], pl.FP32] = pl.load(col_vec_tensor, [0, 0], [1, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.col_expand(src_tile, col_tile)
                return pl.store(result, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.tcolexpand" in mlir, f"col_expand should generate pto.tcolexpand, got:\n{mlir}"
        # ptoas expects unary ins; two SSA values look like "ins(%a, %b : ...)".
        for line in mlir.splitlines():
            if "pto.tcolexpand" in line and "ins(" in line:
                after_ins = line.split("ins(", 1)[1]
                value_list = after_ins.split(" : ", 1)[0]
                assert "," not in value_list, f"tcolexpand ins should be single SSA operand, got: {line!r}"
                break
        else:
            raise AssertionError("no pto.tcolexpand ins(...) line in MLIR")

    def test_row_expand_codegen(self):
        """tile.row_expand(target, row_vec) should emit pto.trowexpand with only row_vec in ins()."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                row_vec_tensor: pl.Tensor[[16, 1], pl.FP32],
                dst: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                src_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(src, [0, 0], [16, 16])
                row_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(row_vec_tensor, [0, 0], [16, 1])
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.row_expand(src_tile, row_tile)
                return pl.store(result, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.trowexpand" in mlir, f"row_expand should generate pto.trowexpand, got:\n{mlir}"
        # ptoas expects unary ins; two SSA values look like "ins(%a, %b : ...)".
        for line in mlir.splitlines():
            if "pto.trowexpand" in line and "ins(" in line:
                after_ins = line.split("ins(", 1)[1]
                value_list = after_ins.split(" : ", 1)[0]
                assert "," not in value_list, f"trowexpand ins should be single SSA operand, got: {line!r}"
                break
        else:
            raise AssertionError("no pto.trowexpand ins(...) line in MLIR")


class TestTileSliceCodegen:
    """Tests for tile.slice PTO code generation (pto.textract)."""

    def _generate_mlir(self, program_cls) -> str:
        """Run PassManager and PTOCodegen on the given program, return MLIR string."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program_cls)
        codegen_instance = codegen.PTOCodegen()
        funcs = list(optimized.functions.values())
        assert funcs, "Program has no functions"
        single = ir.Program([funcs[0]], funcs[0].name, optimized.span)
        return codegen_instance.generate(single)

    def test_tile_slice_codegen(self):
        """tile.slice(tile[32,32], [16,16], [0,0]) should generate pto.textract."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[32, 32], pl.FP32],
                dst: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                src_tile: pl.Tile[[32, 32], pl.FP32] = pl.load(src, [0, 0], [32, 32])
                sliced: pl.Tile[[16, 16], pl.FP32] = pl.tile.slice(src_tile, [16, 16], [0, 0])
                return pl.store(sliced, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.textract" in mlir, f"tile.slice should generate pto.textract, got:\n{mlir}"

    def test_tile_slice_codegen_with_valid_shape(self):
        """tile.slice(..., valid_shape=...) should still generate pto.textract."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[32, 32], pl.FP32],
                valid_cols: pl.Scalar[pl.INDEX],
                dst: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                src_tile: pl.Tile[[32, 32], pl.FP32] = pl.load(src, [0, 0], [32, 32])
                sliced: pl.Tile[[16, 16], pl.FP32] = pl.tile.slice(
                    src_tile, [16, 16], [0, 0], valid_shape=[16, valid_cols]
                )
                return pl.store(sliced, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.textract" in mlir, (
            f"tile.slice with valid_shape should generate pto.textract, got:\n{mlir}"
        )

    def test_tile_slice_multiple_slices_have_correct_types(self):
        """Multiple tile.slice from one reshape must produce correct type annotations.

        Reproduces the Qwen3 decode pattern:
            load [1,512] → reshape [4,128] → slice [4,64]@[0,0] + slice [4,64]@[0,64]
        Both slices share the same MemRef and PTO buffer (sequential execution),
        but their ins()/outs() type annotations must reflect the [4,64] slice
        shape, not the [1,512] root alloc shape.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[1, 512], pl.FP32],
                dst: pl.Tensor[[4, 128], pl.FP32],
            ) -> pl.Tensor[[4, 128], pl.FP32]:
                tile_a: pl.Tile[[1, 512], pl.FP32] = pl.load(src, [0, 0], [1, 512])
                reshaped: pl.Tile[[4, 128], pl.FP32] = pl.tile.reshape(tile_a, [4, 128])
                lo: pl.Tile[[4, 64], pl.FP32] = pl.tile.slice(reshaped, [4, 64], [0, 0])
                hi: pl.Tile[[4, 64], pl.FP32] = pl.tile.slice(reshaped, [4, 64], [0, 64])
                combined: pl.Tile[[4, 128], pl.FP32] = pl.tile.concat(lo, hi)
                return pl.store(combined, [0, 0], dst)

        mlir = self._generate_mlir(Prog)

        textract_lines = [line.strip() for line in mlir.splitlines() if "pto.textract" in line]
        assert len(textract_lines) == 2, (
            f"Expected 2 pto.textract (lo+hi slices), got {len(textract_lines)}:\n"
            + "\n".join(textract_lines)
        )
        for line in textract_lines:
            assert "rows=4" in line and "cols=64" in line, (
                f"textract outs() type should be rows=4,cols=64 (slice shape), got:\n{line}"
            )


class TestSetValidShapeCodegen:
    """Tests for tile.set_validshape PTO code generation."""

    def _generate_mlir(self, program_cls) -> str:
        """Run PassManager and PTOCodegen on the given program, return MLIR string."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program_cls)
        codegen_instance = codegen.PTOCodegen()
        funcs = list(optimized.functions.values())
        assert funcs, "Program has no functions"
        single = ir.Program([funcs[0]], funcs[0].name, optimized.span)
        return codegen_instance.generate(single)

    def test_set_validshape_codegen(self):
        """tile.set_validshape should generate pto.set_validshape."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[32, 32], pl.FP32],
                valid_rows: pl.Scalar[pl.INDEX],
                valid_cols: pl.Scalar[pl.INDEX],
                dst: pl.Tensor[[32, 32], pl.FP32],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                src_tile: pl.Tile[[32, 32], pl.FP32] = pl.load(src, [0, 0], [32, 32])
                narrowed: pl.Tile[[32, 32], pl.FP32] = pl.tile.set_validshape(
                    src_tile, valid_rows, valid_cols
                )
                return pl.store(narrowed, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.set_validshape" in mlir, (
            f"tile.set_validshape should generate pto.set_validshape, got:\n{mlir}"
        )


class TestMrgSortCodegen:
    """Tests for mrgsort format1 code generation with constant and variable block_len."""

    def _generate_mlir(self, program_cls) -> str:
        """Run PassManager and PTOCodegen on the given program, return MLIR string."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program_cls)
        codegen_instance = codegen.PTOCodegen()
        funcs = list(optimized.functions.values())
        assert funcs, "Program has no functions"
        single = ir.Program([funcs[0]], funcs[0].name, optimized.span)
        return codegen_instance.generate(single)

    def test_mrgsort_format1_const_block_len(self):
        """mrgsort with constant block_len=64 should generate pto.tmrgsort with i32 operand."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[1, 256], pl.FP32],
                idx: pl.Tensor[[1, 256], pl.UINT32],
            ) -> pl.Tensor[[1, 256], pl.FP32]:
                src_tile: pl.Tile[[1, 256], pl.FP32] = pl.load(src, [0, 0], [1, 256])
                idx_tile: pl.Tile[[1, 256], pl.UINT32] = pl.load(idx, [0, 0], [1, 256])
                sorted_tile: pl.Tile[[1, 512], pl.FP32] = pl.tile.sort32(src_tile, idx_tile)
                merged: pl.Tile[[1, 512], pl.FP32] = pl.tile.mrgsort(sorted_tile, block_len=64)
                vals: pl.Tile[[1, 256], pl.FP32] = pl.tile.gather(
                    merged, mask_pattern=pl.tile.MaskPattern.P0101
                )
                return pl.store(vals, [0, 0], src)

        mlir = self._generate_mlir(Prog)
        assert "pto.tmrgsort" in mlir, f"Expected pto.tmrgsort in codegen output:\n{mlir}"
        # Constant block_len should appear as an i32 constant
        tmrgsort_lines = [line for line in mlir.splitlines() if "pto.tmrgsort" in line]
        assert tmrgsort_lines, "No pto.tmrgsort line found"
        assert "i32" in tmrgsort_lines[0], f"block_len type annotation should be i32: {tmrgsort_lines[0]}"

    def test_mrgsort_format1_variable_block_len(self):
        """mrgsort with variable block_len (function parameter) should generate pto.tmrgsort."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[1, 256], pl.FP32],
                idx: pl.Tensor[[1, 256], pl.UINT32],
                block_len: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[1, 256], pl.FP32]:
                src_tile: pl.Tile[[1, 256], pl.FP32] = pl.load(src, [0, 0], [1, 256])
                idx_tile: pl.Tile[[1, 256], pl.UINT32] = pl.load(idx, [0, 0], [1, 256])
                sorted_tile: pl.Tile[[1, 512], pl.FP32] = pl.tile.sort32(src_tile, idx_tile)
                merged: pl.Tile[[1, 512], pl.FP32] = pl.tile.mrgsort(sorted_tile, block_len=block_len)
                vals: pl.Tile[[1, 256], pl.FP32] = pl.tile.gather(
                    merged, mask_pattern=pl.tile.MaskPattern.P0101
                )
                return pl.store(vals, [0, 0], src)

        mlir = self._generate_mlir(Prog)
        assert "pto.tmrgsort" in mlir, f"Expected pto.tmrgsort in codegen output:\n{mlir}"
        tmrgsort_lines = [line for line in mlir.splitlines() if "pto.tmrgsort" in line]
        assert tmrgsort_lines, "No pto.tmrgsort line found"
        assert "i32" in tmrgsort_lines[0], f"block_len type annotation should be i32: {tmrgsort_lines[0]}"


class TestConstDtypeCodegen:
    """Regression tests for const dtype emission (Issue #934).

    Previously, codegen hardcoded 'f32' for all float consts and 'index' for
    all int consts. These tests verify the actual dtype is emitted.
    """

    def _generate_mlir(self, program_cls) -> str:
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program_cls)
        codegen_instance = codegen.PTOCodegen()
        funcs = list(optimized.functions.values())
        assert funcs
        single = ir.Program([funcs[0]], funcs[0].name, optimized.span)
        return codegen_instance.generate(single)

    def test_full_bf16_const_emits_bf16(self):
        """tile.full with a bf16 fill value must emit bf16, not f32 (Issue #934)."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 16], pl.BF16],
            ) -> pl.Tensor[[16, 16], pl.BF16]:
                t = pl.tile.full([16, 16], dtype=pl.BF16, value=1.0)
                return pl.store(t, [0, 0], a)

        mlir = self._generate_mlir(Prog)
        assert "bf16" in mlir, f"Expected bf16 in MLIR output:\n{mlir}"
        assert "1.00000000000000000e+00 : bf16" in mlir, f"Expected bf16 float constant in MLIR:\n{mlir}"
        # Ensure no f32 constant was emitted for the fill value
        assert "1.00000000000000000e+00 : f32" not in mlir, f"f32 constant leaked into MLIR:\n{mlir}"

    def test_full_f16_const_emits_f16(self):
        """tile.full with an f16 fill value must emit f16, not f32 (Issue #934)."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[16, 16], pl.FP16],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                t = pl.tile.full([16, 16], dtype=pl.FP16, value=0.0)
                return pl.store(t, [0, 0], a)

        mlir = self._generate_mlir(Prog)
        assert "f16" in mlir, f"Expected f16 in MLIR output:\n{mlir}"
        assert "0.00000000000000000e+00 : f16" in mlir, f"Expected f16 float constant in MLIR:\n{mlir}"
        assert "0.00000000000000000e+00 : f32" not in mlir, f"f32 constant leaked into MLIR:\n{mlir}"


class TestColReductionCodegen:
    """Tests for column-wise reduction operations codegen (Issue #881)."""

    def _generate_mlir(self, program_cls) -> str:
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program_cls)
        codegen_instance = codegen.PTOCodegen()
        funcs = list(optimized.functions.values())
        assert funcs
        single = ir.Program([funcs[0]], funcs[0].name, optimized.span)
        return codegen_instance.generate(single)

    def test_col_sum_codegen(self):
        """tile.col_sum without tmp_tile emits pto.tcolsum with no isBinary attribute."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[1, 16], pl.FP32],
            ) -> pl.Tensor[[1, 16], pl.FP32]:
                tile_in: pl.Tile[[16, 16], pl.FP32] = pl.load(input, [0, 0], [16, 16])
                result: pl.Tile[[1, 16], pl.FP32] = pl.tile.col_sum(tile_in)
                return pl.store(result, [0, 0], output)

        mlir = self._generate_mlir(Prog)
        assert "pto.tcolsum" in mlir, f"Expected pto.tcolsum in codegen output:\n{mlir}"
        assert "isBinary" not in mlir, f"Expected no isBinary attribute in codegen output:\n{mlir}"

    def test_col_sum_codegen_binary(self):
        """tile.col_sum with tmp_tile emits isBinary = true."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[1, 16], pl.FP32],
            ) -> pl.Tensor[[1, 16], pl.FP32]:
                tile_in: pl.Tile[[16, 16], pl.FP32] = pl.load(input, [0, 0], [16, 16])
                tmp_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.create(
                    [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                result: pl.Tile[[1, 16], pl.FP32] = pl.tile.col_sum(tile_in, tmp_tile)
                return pl.store(result, [0, 0], output)

        mlir = self._generate_mlir(Prog)
        assert "pto.tcolsum" in mlir, f"Expected pto.tcolsum in codegen output:\n{mlir}"
        assert "isBinary = true" in mlir, f"Expected isBinary = true in codegen output:\n{mlir}"

    def test_col_max_codegen(self):
        """tile.col_max emits pto.tcolmax."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[1, 16], pl.FP32],
            ) -> pl.Tensor[[1, 16], pl.FP32]:
                tile_in: pl.Tile[[16, 16], pl.FP32] = pl.load(input, [0, 0], [16, 16])
                result: pl.Tile[[1, 16], pl.FP32] = pl.tile.col_max(tile_in)
                return pl.store(result, [0, 0], output)

        mlir = self._generate_mlir(Prog)
        assert "pto.tcolmax" in mlir, f"Expected pto.tcolmax in codegen output:\n{mlir}"

    def test_col_min_codegen(self):
        """tile.col_min emits pto.tcolmin."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Tensor[[1, 16], pl.FP32],
            ) -> pl.Tensor[[1, 16], pl.FP32]:
                tile_in: pl.Tile[[16, 16], pl.FP32] = pl.load(input, [0, 0], [16, 16])
                result: pl.Tile[[1, 16], pl.FP32] = pl.tile.col_min(tile_in)
                return pl.store(result, [0, 0], output)

        mlir = self._generate_mlir(Prog)
        assert "pto.tcolmin" in mlir, f"Expected pto.tcolmin in codegen output:\n{mlir}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
