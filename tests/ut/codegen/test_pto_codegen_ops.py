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

This test validates code generation for all supported block-level operations
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
    """Test program containing kernels for all supported block-level operations."""

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
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.add(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
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
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.sub(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
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
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.mul(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
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
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.div(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
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
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.adds(tensor_tile, scalar)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
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
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.subs(tensor_tile, scalar)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
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
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.muls(tensor_tile, scalar)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
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
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.divs(tensor_tile, scalar)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
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
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.neg(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_exp(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise exponential: output = exp(input)."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.exp(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_sqrt(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise square root: output = sqrt(input)."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.sqrt(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_rsqrt(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise reciprocal square root: output = 1/sqrt(input)."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.rsqrt(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_recip(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise reciprocal: output = 1/input."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.recip(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_log(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise natural logarithm: output = log(input)."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.log(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_abs(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise absolute value: output = abs(input)."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.abs(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_relu(
        self,
        input_tensor: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise ReLU: output = max(0, input)."""
        input_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.relu(input_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
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
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.maximum(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
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
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.minimum(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
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
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.cmp(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
        return updated_output

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_matmul(
        self,
        lhs: pl.Tensor[[16, 16], pl.FP32],
        rhs: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Matmul: output = matmul(lhs, rhs)."""
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, 0], [16, 16])
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.matmul(lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
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
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, 0], [16, 16])
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [0, 0], [16, 16])
        factor_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(factor, [0, 0], [16, 16])
        result_tile: pl.Tile[[16, 16], pl.FP32] = pl.block.matmul_acc(factor_tile, lhs_tile, rhs_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(result_tile, [0, 0], [16, 16], output)
        return updated_output


def build_block_ops_test_program(dtype: DataType = DataType.FP32):
    """Build the block operations test program.

    Args:
        dtype: Data type for tensors (currently only FP32 is supported).

    Returns:
        BlockOperationsTest program containing all block-level operation kernels.

    Raises:
        ValueError: If dtype is not FP32.
    """
    if dtype != DataType.FP32:
        raise ValueError(f"Only FP32 is currently supported, got {dtype}")
    return BlockOperationsTest


class Test910BBlockOpsCodegen:
    """Tests for 910B PTO backend block-level operations code generation."""

    def test_block_ops_codegen(self):
        """Test code generation for all block-level operations."""
        # Set backend type for testing
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.PTO)

        dtype = DataType.FP32

        # Build IR program
        program = build_block_ops_test_program(dtype)

        # Validate program structure
        assert program is not None, "Program should not be None"
        assert hasattr(program, "functions"), "Program should have functions attribute"
        assert len(program.functions) > 0, "Program should have at least one function"

        # Collect function names for validation
        function_names = [func.name for func in program.functions.values()]

        # Validate that all functions start with "kernel_" prefix
        for func_name in function_names:
            assert func_name.startswith("kernel_"), f"Function {func_name} should start with 'kernel_' prefix"

        # Run PassManager optimization with PTOAS strategy (PTO assembly without sync)
        pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
