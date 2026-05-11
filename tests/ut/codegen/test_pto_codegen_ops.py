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
        result_tile: pl.Tile[[16, 32], pl.UINT8] = pl.tile.cmp(lhs_tile, rhs_tile)
        one_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.full([16, 16], dtype=pl.FP32, value=1.0)
        zero_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.full([16, 16], dtype=pl.FP32, value=0.0)
        tmp_tile: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
        selected_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.sel(result_tile, one_tile, zero_tile, tmp_tile)
        updated_output: pl.Tensor[[16, 16], pl.FP32] = pl.store(selected_tile, [0, 0], output)
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
    """Tests for tile.slice PTO code generation (pto.subview).

    tile.slice lowers to pto.subview — a pure view alias of the source tile —
    rather than the historical pto.textract data-movement op.  The result tile
    inherits the source's tile_buf configuration (loc/dtype/blayout/slayout/
    fractal/pad) and only the shape/valid_shape change.
    """

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
        """tile.slice(tile[32,32], [16,16], [0,0]) should generate pto.subview."""

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
        assert "pto.subview" in mlir, f"tile.slice should generate pto.subview, got:\n{mlir}"
        assert "pto.textract" not in mlir, f"tile.slice no longer emits pto.textract, got:\n{mlir}"
        # Subview line must carry both source and result tile_buf types and a
        # static `sizes [R, C]` attribute matching the slice shape.
        subview_lines = [line for line in mlir.splitlines() if "pto.subview" in line]
        assert subview_lines, "no pto.subview line emitted"
        line = subview_lines[0]
        assert "sizes [16, 16]" in line, f"sizes attribute must be [16, 16], got:\n{line}"
        assert "rows=16, cols=16" in line, f"result tile_buf must carry rows=16, cols=16, got:\n{line}"

    def test_tile_slice_codegen_with_valid_shape(self):
        """tile.slice(..., valid_shape=...) emits pto.subview + pto.tmov + pto.set_validshape."""

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
        assert "pto.subview" in mlir, f"tile.slice with valid_shape should generate pto.subview, got:\n{mlir}"
        # With explicit valid_shape, tile.slice emits subview + tmov + set_validshape
        # instead of a subview with `valid [...]` clause.
        tmov_lines = [line for line in mlir.splitlines() if "pto.tmov" in line]
        assert tmov_lines, f"Expected pto.tmov after subview for valid_shape slice, got:\n{mlir}"
        set_vs_lines = [line for line in mlir.splitlines() if "pto.set_validshape" in line]
        assert set_vs_lines, f"Expected pto.set_validshape for valid_shape slice, got:\n{mlir}"

    def test_tile_slice_multiple_slices_have_correct_types(self):
        """Multiple tile.slice from one reshape must produce correct type annotations.

        Reproduces the Qwen3 decode pattern:
            load [1,512] → reshape [4,128] → slice [4,64]@[0,0] + slice [4,64]@[0,64]
        Both slices share the same MemRef and PTO buffer (sequential execution),
        but their pto.subview result tile_buf types must reflect the [4,64]
        slice shape, not the [1,512] root alloc shape.
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

        subview_lines = [line.strip() for line in mlir.splitlines() if "pto.subview" in line]
        assert len(subview_lines) == 2, (
            f"Expected 2 pto.subview (lo+hi slices), got {len(subview_lines)}:\n" + "\n".join(subview_lines)
        )
        # Both subviews must declare the [4, 64] sub-shape on their result type.
        for line in subview_lines:
            result_type = line.split("->", 1)[-1]
            assert "rows=4" in result_type and "cols=64" in result_type, (
                f"subview result type should be rows=4,cols=64 (slice shape), got:\n{line}"
            )

    def test_tile_slice_static_valid_inferred_with_dynamic_offset(self):
        """tile.slice with dynamic offset should infer static v_row/v_col when source valid >= slice size.

        Regression test for: InferSubviewTileTypeComponents was missing the case where
        the source valid extent is a static constant >= the requested slice size but the
        offset is dynamic. Before the fix, v_row/v_col were left as dynamic ('?') even
        though any offset within bounds leaves exactly `size` valid elements.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[32, 32], pl.FP32],
                row_off: pl.Scalar[pl.INDEX],
                col_off: pl.Scalar[pl.INDEX],
                dst: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                src_tile: pl.Tile[[32, 32], pl.FP32] = pl.load(src, [0, 0], [32, 32])
                sliced: pl.Tile[[16, 16], pl.FP32] = pl.tile.slice(src_tile, [16, 16], [row_off, col_off])
                return pl.store(sliced, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        subview_lines = [line for line in mlir.splitlines() if "pto.subview" in line]
        assert subview_lines, "no pto.subview line emitted"
        result_type = subview_lines[0].split("->", 1)[-1]
        assert "v_row=16" in result_type and "v_col=16" in result_type, (
            f"Source valid [32,32] >= slice size [16,16] with dynamic offset must yield "
            f"static v_row=16, v_col=16 in result tile_buf type, got:\n{subview_lines[0]}"
        )
        assert "v_row=?" not in result_type and "v_col=?" not in result_type, (
            f"v_row/v_col must not be dynamic ('?') when source valid >= slice size, got:\n{subview_lines[0]}"
        )


class TestTileAssembleCodegen:
    """Tests for tile.assemble PTO code generation (pto.subview + pto.tmov).

    tile.assemble lowers to:
      1. (optional) pto.tmov target → dst when buffer reuse did not merge them.
      2. %dst_view = pto.subview %dst[row, col] sizes [...] : ... -> ...
      3. pto.tmov ins(%src) outs(%dst_view)
    The historical pto.tinsert copy op is no longer emitted.
    """

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

    def test_tile_assemble_codegen(self):
        """tile.assemble lowers to pto.subview + pto.tmov, never pto.tinsert."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                target_in: pl.Tensor[[16, 128], pl.FP32],
                source_in: pl.Tensor[[16, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                target: pl.Tile[[16, 128], pl.FP32] = pl.load(target_in, [0, 0], [16, 128])
                source: pl.Tile[[16, 64], pl.FP32] = pl.load(source_in, [0, 0], [16, 64])
                result: pl.Tile[[16, 128], pl.FP32] = pl.tile.assemble(target, source, [0, 64])
                return pl.store(result, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        assert "pto.tinsert" not in mlir, f"tile.assemble no longer emits pto.tinsert, got:\n{mlir}"
        # Exactly one pto.subview should carve out the destination window.
        subview_lines = [line.strip() for line in mlir.splitlines() if "pto.subview" in line]
        assert len(subview_lines) == 1, (
            f"Expected one pto.subview for tile.assemble, got {len(subview_lines)}:\n"
            + "\n".join(subview_lines)
        )
        sv = subview_lines[0]
        assert "sizes [16, 64]" in sv, f"subview sizes must equal source shape: {sv}"
        assert "valid [" in sv, f"subview must carry explicit valid operands: {sv}"
        valid_clause = sv.split("valid [", 1)[1].split("]", 1)[0]
        assert "%" in valid_clause, f"subview valid operands must be SSA/index values, got: {sv}"
        assert "v_row=16" in sv and "v_col=64" in sv, (
            f"subview result type must preserve the source tile valid_shape: {sv}"
        )
        # The actual data write is a pto.tmov from src into the subview SSA.
        view_ssa = sv.split(" = ", 1)[0].strip()
        assert any("pto.tmov" in line and f"outs({view_ssa}" in line for line in mlir.splitlines()), (
            f"tile.assemble should pto.tmov into the subview SSA {view_ssa!r}, got:\n{mlir}"
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

    def test_set_validshape_on_gather_output_codegen(self):
        """Regression for issue #1174: tile.set_validshape on a tile.gather output
        must allocate the gather output with dynamic validShape (v_row=?, v_col=?)
        and physical-dim valid_row/valid_col operands, so the downstream
        pto.set_validshape sees a dynamic-validShape source (PTOAS requirement)."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[1, 2048], pl.FP32],
                valid_cols: pl.Scalar[pl.INDEX],
                dst: pl.Tensor[[1, 1024], pl.FP32],
            ) -> pl.Tensor[[1, 1024], pl.FP32]:
                src_tile: pl.Tile[[1, 2048], pl.FP32] = pl.load(src, [0, 0], [1, 2048])
                gathered: pl.Tile[[1, 1024], pl.FP32] = pl.tile.gather_mask(
                    src_tile, mask_pattern=pl.tile.MaskPattern.P1010
                )
                narrowed: pl.Tile[[1, 1024], pl.FP32] = pl.tile.set_validshape(gathered, 1, valid_cols)
                return pl.store(narrowed, [0, 0], dst)

        mlir = self._generate_mlir(Prog)

        gathered_alloc_lines = [
            line for line in mlir.splitlines() if "pto.alloc_tile" in line and "%gathered" in line
        ]
        assert gathered_alloc_lines, f"Expected pto.alloc_tile for 'gathered' var, got:\n{mlir}"
        assert len(gathered_alloc_lines) == 1, (
            f"Expected exactly one alloc_tile for 'gathered', got {len(gathered_alloc_lines)}:\n"
            + "\n".join(gathered_alloc_lines)
        )

        gathered_line = gathered_alloc_lines[0]
        assert "v_row=?" in gathered_line and "v_col=?" in gathered_line, (
            f"alloc_tile for gather output must use dynamic validShape (v_row=?, v_col=?)\n"
            f"so pto.set_validshape sees a dynamic-validShape source (PTOAS requirement);\n"
            f"got line:\n{gathered_line}\n\nfull MLIR:\n{mlir}"
        )
        assert "valid_row =" in gathered_line and "valid_col =" in gathered_line, (
            f"alloc_tile for gather output must carry valid_row / valid_col operands\n"
            f"(initialized to physical dims so tgather writes the full region);\n"
            f"got line:\n{gathered_line}\n\nfull MLIR:\n{mlir}"
        )

        set_vs_lines = [line for line in mlir.splitlines() if "pto.set_validshape" in line]
        assert set_vs_lines, f"Expected pto.set_validshape in codegen output:\n{mlir}"
        for line in set_vs_lines:
            assert "v_row=?" in line and "v_col=?" in line, (
                f"pto.set_validshape source tile_buf type must be dynamic "
                f"(v_row=?, v_col=?); got line:\n{line}\n\nfull MLIR:\n{mlir}"
            )

    def test_set_validshape_updates_source_tile_and_aliases_result(self):
        """tile.set_validshape should mutate the source tile handle and alias the result to it."""

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
        set_validshape_line = next(line.strip() for line in mlir.splitlines() if "pto.set_validshape" in line)
        narrowed_alloc_lines = [
            line.strip() for line in mlir.splitlines() if "%narrowed" in line and "pto.alloc_tile" in line
        ]
        store_line = next(
            line.strip() for line in mlir.splitlines() if "pto.tstore" in line and "%src_tile" in line
        )

        assert "%src_tile" in set_validshape_line, (
            f"Expected pto.set_validshape to target the source tile SSA, got:\n{set_validshape_line}"
        )
        assert "v_row=?" in set_validshape_line and "v_col=?" in set_validshape_line, (
            f"Expected dynamic source tile type on pto.set_validshape, got:\n{set_validshape_line}"
        )
        assert not narrowed_alloc_lines, (
            "Expected set_validshape result to alias the source tile without a second alloc_tile, got:\n"
            + "\n".join(narrowed_alloc_lines)
        )
        assert "%src_tile" in store_line, (
            f"Expected subsequent narrowed uses to resolve to the source tile SSA, got:\n{store_line}"
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
                vals: pl.Tile[[1, 256], pl.FP32] = pl.tile.gather_mask(
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
                vals: pl.Tile[[1, 256], pl.FP32] = pl.tile.gather_mask(
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


class TestTileExtractCodegen:
    """Tests for tile.extract PTO code generation (pto.textract)."""

    def _generate_mlir(self, program_cls) -> str:
        """Run PassManager and PTOCodegen on the given program, return MLIR string."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program_cls)
        codegen_instance = codegen.PTOCodegen()
        funcs = list(optimized.functions.values())
        assert funcs, "Program has no functions"
        # ExpandMixedKernel may wrap the kernel in a Group function plus AIC/AIV
        # variants. PTOCodegen rejects Group; pick the first InCore-variant func.
        target = next((f for f in funcs if ir.is_incore_type(f.func_type)), funcs[0])
        single = ir.Program([target], target.name, optimized.span)
        return codegen_instance.generate(single)

    def test_tile_extract_acc_to_mat_emits_pto_textract(self):
        """tile.extract from an Acc-source tile to Mat lowers to pto.textract."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[32, 128], pl.BF16],
                y: pl.Tensor[[128, 32], pl.BF16],
                dst: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                x_mat = pl.load(x, [0, 0], [32, 128], target_memory=pl.MemorySpace.Mat)
                y_mat = pl.load(y, [0, 0], [128, 32], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                acc: pl.Tile[[32, 32], pl.FP32] = pl.matmul(x_left, y_right)
                sub: pl.Tile[[16, 16], pl.FP32] = pl.tile.extract(
                    acc, 0, 0, shape=[16, 16], target_memory=pl.MemorySpace.Mat
                )
                vec = pl.move(sub, target_memory=pl.MemorySpace.Vec)
                return pl.store(vec, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.textract" in mlir, f"Expected pto.textract in:\n{mlir}"

        textract_lines = [line.strip() for line in mlir.splitlines() if "pto.textract" in line]
        assert textract_lines, "no pto.textract line emitted"
        line = textract_lines[0]
        assert "ins(" in line and "outs(" in line, f"DPS form expected, got: {line}"
        ins_clause = line.split("ins(", 1)[1].split(")", 1)[0]
        operand_count = ins_clause.split(":", 1)[0].count(",") + 1
        assert operand_count == 3, (
            f"pto.textract should have 3 ins operands (src, row, col), got {operand_count}: {line}"
        )


class TestTileScatterUpdateCodegen:
    """Tests for tile.scatter_update PTO code generation.

    tile.scatter_update(input, index, src) should update the input tile in place.
    PTO codegen lowers this to nested scf.for loops plus pto.tgetval/pto.textract/pto.tinsert.
    """

    def _generate_mlir(self, program_cls) -> str:
        """Run PassManager and PTOCodegen on the given program, return MLIR string."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program_cls)
        codegen_instance = codegen.PTOCodegen()
        funcs = list(optimized.functions.values())
        assert funcs, "Program has no functions"
        # ExpandMixedKernel may wrap the kernel in a Group function plus AIC/AIV
        # variants. PTOCodegen rejects Group; pick the first InCore-variant func.
        target = next((f for f in funcs if ir.is_incore_type(f.func_type)), funcs[0])
        single = ir.Program([target], target.name, optimized.span)
        return codegen_instance.generate(single)

    def test_tile_scatter_update_emits_scf_for(self):
        """tile.scatter_update should generate a scf.for loop over b*s iterations."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                input_t: pl.Tensor[[16, 32], pl.FP32],
                index_t: pl.Tensor[[2, 4], pl.INT32],
                src_t: pl.Tensor[[8, 32], pl.FP32],
                dst_t: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                input_tile: pl.Tile[[16, 32], pl.FP32] = pl.load(input_t, [0, 0], [16, 32])
                index_tile: pl.Tile[[2, 4], pl.INT32] = pl.load(index_t, [0, 0], [2, 4])
                src_tile: pl.Tile[[8, 32], pl.FP32] = pl.load(src_t, [0, 0], [8, 32])
                scratch_tile: pl.Tile[[1, 32], pl.FP32] = pl.tile.create(
                    [1, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                result: pl.Tile[[16, 32], pl.FP32] = pl.tile.scatter_update(
                    input_tile, index_tile, src_tile, scratch_tile, dim=-2
                )
                return pl.store(result, [0, 0], dst_t)

        mlir = self._generate_mlir(Prog)
        assert "scf.for" in mlir, f"scatter_update should generate scf.for loop, got:\n{mlir}"

    def test_tile_scatter_update_emits_row_extract_and_insert(self):
        """tile.scatter_update should read index scalar, extract src row, and insert row into dst."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                input_t: pl.Tensor[[16, 32], pl.FP32],
                index_t: pl.Tensor[[2, 4], pl.INT32],
                src_t: pl.Tensor[[8, 32], pl.FP32],
                dst_t: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                input_tile: pl.Tile[[16, 32], pl.FP32] = pl.load(input_t, [0, 0], [16, 32])
                index_tile: pl.Tile[[2, 4], pl.INT32] = pl.load(index_t, [0, 0], [2, 4])
                src_tile: pl.Tile[[8, 32], pl.FP32] = pl.load(src_t, [0, 0], [8, 32])
                scratch_tile: pl.Tile[[1, 32], pl.FP32] = pl.tile.create(
                    [1, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                result: pl.Tile[[16, 32], pl.FP32] = pl.tile.scatter_update(
                    input_tile, index_tile, src_tile, scratch_tile, dim=-2
                )
                return pl.store(result, [0, 0], dst_t)

        mlir = self._generate_mlir(Prog)
        assert "pto.tgetval" in mlir, f"scatter_update should emit pto.tgetval for index read, got:\n{mlir}"
        assert "pto.textract" in mlir, (
            f"scatter_update should emit pto.textract for src row read, got:\n{mlir}"
        )
        assert "pto.tinsert" in mlir, (
            f"scatter_update should emit pto.tinsert for dst row write, got:\n{mlir}"
        )
        assert "pto.tsetval" not in mlir, f"scatter_update should not emit scalar row writes, got:\n{mlir}"

    def test_tensor_scatter_update_scratch_has_alloc_addr(self):
        """tensor.scatter_update lowering should materialize an addressed scratch row tile."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                input_t: pl.Tensor[[16, 32], pl.FP32],
                index_t: pl.Tensor[[2, 4], pl.INT32],
                src_t: pl.Tensor[[8, 32], pl.FP32],
                dst_t: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                result: pl.Tensor[[16, 32], pl.FP32] = pl.scatter_update(input_t, -2, index_t, src_t)
                return result

        mlir = self._generate_mlir(Prog)
        scatter_allocs = [
            line.strip() for line in mlir.splitlines() if "scatter_row" in line and "pto.alloc_tile" in line
        ]
        assert scatter_allocs, f"Expected scatter_row alloc_tile, got:\n{mlir}"
        assert all("addr =" in line for line in scatter_allocs), (
            f"scatter_row alloc_tile must carry addr operand, got: {scatter_allocs}"
        )

    def test_tile_scatter_update_loop_bound(self):
        """scf.for upper bound should equal b*s (here 2*4=8)."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                input_t: pl.Tensor[[16, 32], pl.FP32],
                index_t: pl.Tensor[[2, 4], pl.INT32],
                src_t: pl.Tensor[[8, 32], pl.FP32],
                dst_t: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                input_tile: pl.Tile[[16, 32], pl.FP32] = pl.load(input_t, [0, 0], [16, 32])
                index_tile: pl.Tile[[2, 4], pl.INT32] = pl.load(index_t, [0, 0], [2, 4])
                src_tile: pl.Tile[[8, 32], pl.FP32] = pl.load(src_t, [0, 0], [8, 32])
                scratch_tile: pl.Tile[[1, 32], pl.FP32] = pl.tile.create(
                    [1, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                result: pl.Tile[[16, 32], pl.FP32] = pl.tile.scatter_update(
                    input_tile, index_tile, src_tile, scratch_tile, dim=-2
                )
                return pl.store(result, [0, 0], dst_t)

        mlir = self._generate_mlir(Prog)
        # b=2, s=4 → b*s=8; constant "8" should appear as loop upper bound
        assert "8" in mlir, f"Expected loop bound 8 (b*s=2*4) in MLIR:\n{mlir}"

    def test_tile_scatter_update_index_cast(self):
        """arith.index_cast should be emitted to convert i32 index to index type."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                input_t: pl.Tensor[[16, 32], pl.FP32],
                index_t: pl.Tensor[[2, 4], pl.INT32],
                src_t: pl.Tensor[[8, 32], pl.FP32],
                dst_t: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                input_tile: pl.Tile[[16, 32], pl.FP32] = pl.load(input_t, [0, 0], [16, 32])
                index_tile: pl.Tile[[2, 4], pl.INT32] = pl.load(index_t, [0, 0], [2, 4])
                src_tile: pl.Tile[[8, 32], pl.FP32] = pl.load(src_t, [0, 0], [8, 32])
                scratch_tile: pl.Tile[[1, 32], pl.FP32] = pl.tile.create(
                    [1, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                result: pl.Tile[[16, 32], pl.FP32] = pl.tile.scatter_update(
                    input_tile, index_tile, src_tile, scratch_tile, dim=-2
                )
                return pl.store(result, [0, 0], dst_t)

        mlir = self._generate_mlir(Prog)
        assert "arith.index_cast" in mlir, f"Expected arith.index_cast in MLIR:\n{mlir}"

    def test_tile_scatter_update_is_inplace(self):
        """tile.scatter_update should reuse the input tile buffer instead of copying it."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                input_t: pl.Tensor[[16, 32], pl.FP32],
                index_t: pl.Tensor[[2, 4], pl.INT32],
                src_t: pl.Tensor[[8, 32], pl.FP32],
                dst_t: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                input_tile: pl.Tile[[16, 32], pl.FP32] = pl.load(input_t, [0, 0], [16, 32])
                index_tile: pl.Tile[[2, 4], pl.INT32] = pl.load(index_t, [0, 0], [2, 4])
                src_tile: pl.Tile[[8, 32], pl.FP32] = pl.load(src_t, [0, 0], [8, 32])
                scratch_tile: pl.Tile[[1, 32], pl.FP32] = pl.tile.create(
                    [1, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                result: pl.Tile[[16, 32], pl.FP32] = pl.tile.scatter_update(
                    input_tile, index_tile, src_tile, scratch_tile, dim=-2
                )
                return pl.store(result, [0, 0], dst_t)

        mlir = self._generate_mlir(Prog)
        assert "pto.tmov" not in mlir, f"scatter_update should update in place, got:\n{mlir}"
        assert "result = pto.alloc_tile" not in mlir, (
            f"scatter_update result should alias input tile, got:\n{mlir}"
        )

    def test_tile_scatter_update_avoids_runtime_treshape(self):
        """tile.scatter_update should access original 2D tile buffers directly."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                input_t: pl.Tensor[[32, 32], pl.FP32],
                index_t: pl.Tensor[[2, 8], pl.INT32],
                src_t: pl.Tensor[[16, 32], pl.FP32],
                dst_t: pl.Tensor[[32, 32], pl.FP32],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                input_tile: pl.Tile[[32, 32], pl.FP32] = pl.load(input_t, [0, 0], [32, 32])
                index_tile: pl.Tile[[2, 8], pl.INT32] = pl.load(index_t, [0, 0], [2, 8])
                src_tile: pl.Tile[[16, 32], pl.FP32] = pl.load(src_t, [0, 0], [16, 32])
                scratch_tile: pl.Tile[[1, 32], pl.FP32] = pl.tile.create(
                    [1, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                result: pl.Tile[[32, 32], pl.FP32] = pl.tile.scatter_update(
                    input_tile, index_tile, src_tile, scratch_tile, dim=-2
                )
                return pl.store(result, [0, 0], dst_t)

        mlir = self._generate_mlir(Prog)
        assert "pto.treshape" not in mlir, f"scatter_update should not emit runtime treshape, got:\n{mlir}"
        assert "rows=2, cols=8" in mlir, f"Expected original index tile shape in MLIR:\n{mlir}"
        assert "rows=16, cols=32" in mlir, f"Expected original src tile shape in MLIR:\n{mlir}"
        assert "rows=32, cols=32" in mlir, f"Expected original dst tile shape in MLIR:\n{mlir}"


class TestTileMoveAccNoopElision:
    """Regression test for #1310: pto.tmov acc→acc must be elided.

    When AutoTileMatmulL0 rewrites matmul_acc into an inner K-loop, the fresh
    IterArg Vars carry different MemRef bases than the outer loop's accumulator.
    MemoryReuse's YieldFixupMutator inserts tile.move(target_memory=Acc) for
    these, but after AllocateMemoryAddr both sides share the same physical Acc
    address. Codegen must elide the no-op pto.tmov to avoid the unsupported
    acc→acc address-space pair on Ascend 910B.
    """

    def _generate_mlir(self, program_cls) -> str:
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program_cls)
        codegen_instance = codegen.PTOCodegen()
        funcs = list(optimized.functions.values())
        assert funcs, "Program has no functions"
        target = next((f for f in funcs if ir.is_incore_type(f.func_type)), funcs[0])
        single = ir.Program([target], target.name, optimized.span)
        return codegen_instance.generate(single)

    @staticmethod
    def _has_acc_to_acc_tmov(mlir: str) -> bool:
        """Check if MLIR contains pto.tmov where both ins and outs are loc=acc."""
        for line in mlir.splitlines():
            stripped = line.strip()
            if "pto.tmov" not in stripped:
                continue
            ins_part = stripped.split("ins(", 1)
            outs_part = stripped.split("outs(", 1)
            if len(ins_part) < 2 or len(outs_part) < 2:
                continue
            ins_clause = ins_part[1].split(")", 1)[0]
            outs_clause = outs_part[1].split(")", 1)[0]
            if "loc=acc" in ins_clause and "loc=acc" in outs_clause:
                return True
        return False

    def test_matmul_acc_in_outer_loop_no_acc_to_acc_tmov(self):
        """matmul_acc inside an outer accumulation loop must not produce pto.tmov acc→acc.

        K=512 exceeds L0 capacity (256 for BF16), so AutoTileMatmulL0 rewrites
        each matmul into an inner K-loop with fresh IterArg Vars. MemoryReuse's
        YieldFixupMutator may insert tile.move acc→acc when the inner loop's
        yield MemRef has a different base_ pointer than the outer iter-arg init.
        The codegen must elide this no-op move.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 1024], pl.BF16],
                w: pl.Tensor[[1024, 64], pl.BF16],
                dst: pl.Tensor[[16, 64], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                # First block: K=512, triggers AutoTileMatmulL0 (K > 256)
                x0 = pl.load(x, [0, 0], [16, 512], target_memory=pl.MemorySpace.Mat)
                w0 = pl.load(w, [0, 0], [512, 64], target_memory=pl.MemorySpace.Mat)
                x0_left = pl.move(x0, target_memory=pl.MemorySpace.Left)
                w0_right = pl.move(w0, target_memory=pl.MemorySpace.Right)
                acc: pl.Tile[[16, 64], pl.FP32] = pl.matmul(x0_left, w0_right)
                # Second block: matmul_acc accumulating into the same Acc tile
                x1 = pl.load(x, [0, 512], [16, 512], target_memory=pl.MemorySpace.Mat)
                w1 = pl.load(w, [512, 0], [512, 64], target_memory=pl.MemorySpace.Mat)
                x1_left = pl.move(x1, target_memory=pl.MemorySpace.Left)
                w1_right = pl.move(w1, target_memory=pl.MemorySpace.Right)
                acc2: pl.Tile[[16, 64], pl.FP32] = pl.matmul_acc(acc, x1_left, w1_right)
                vec = pl.move(acc2, target_memory=pl.MemorySpace.Vec)
                return pl.store(vec, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert not self._has_acc_to_acc_tmov(mlir), (
            f"Generated MLIR contains invalid pto.tmov acc→acc (regression #1310):\n{mlir}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
