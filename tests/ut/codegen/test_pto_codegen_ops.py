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

import re
import warnings

import pypto
import pypto.language as pl
import pytest
from _pto_loc_common import strip_loc
from pypto import DataType, backend, codegen, ir, passes
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
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Matmul_acc: output = matmul_acc(matmul(lhs, rhs), lhs, rhs).

        The accumulator comes from ``tile.matmul``, which lands it in Acc (L0C) by
        construction. It cannot be loaded from GM: ``tile.matmul_acc``'s first
        operand is its own output buffer, and nothing but the matrix unit writes
        L0C -- there is no load or move into it from any other space.
        """
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
            lhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
        )
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
            rhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
        )
        acc_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.tile.matmul(lhs_tile, rhs_tile)
        result_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.tile.matmul_acc(
            acc_tile, lhs_tile, rhs_tile
        )
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


class TestB01PrecisionAndRowExpandAddCodegen:
    """Exact PTOAS forms added for the first op batch."""

    def _generate_mlir(self, program_cls, backend_type=BackendType.Ascend910B) -> str:
        backend.reset_for_testing()
        backend.set_backend_type(backend_type)

        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
        funcs = list(optimized.functions.values())
        assert funcs, "Program has no functions"
        single = ir.Program([funcs[0]], funcs[0].name, optimized.span)
        return codegen.PTOCodegen().generate(single)

    @staticmethod
    def _op_line(mlir: str, op_name: str) -> str:
        line = next((line for line in mlir.splitlines() if op_name in line), "")
        assert line, f"{op_name} not found in MLIR:\n{mlir}"
        return line

    @staticmethod
    def _ins_operand_count(line: str) -> int:
        ins_start = line.find("ins(")
        ins_end = line.find(")", ins_start)
        assert ins_start != -1 and ins_end != -1, f"ins(...) clause not found in: {line}"
        operands = line[ins_start + len("ins(") : ins_end].split(":", 1)[0]
        return operands.count(",") + 1

    def test_tdiv_default_omits_and_high_precision_appends_exact_attr(self):
        @pl.program
        class DefaultProg:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 16], pl.FP32],
                rhs: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, 0], [16, 16])
                rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.div(lhs_tile, rhs_tile)
                return pl.store(result, [0, 0], out)

        @pl.program
        class HighPrecisionProg:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 16], pl.FP32],
                rhs: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, 0], [16, 16])
                rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.div(lhs_tile, rhs_tile, high_precision=True)
                return pl.store(result, [0, 0], out)

        default_line = self._op_line(self._generate_mlir(DefaultProg), "pto.tdiv")
        high_precision_line = self._op_line(self._generate_mlir(HighPrecisionProg), "pto.tdiv")
        assert "precisionType" not in default_line
        assert strip_loc(high_precision_line).endswith("{precisionType = #pto<div_precision high_precision>}")
        assert ") outs(" in high_precision_line
        assert high_precision_line.index("outs(") < high_precision_line.index("precisionType")

    def test_tlog_emits_exact_high_precision_attr(self):
        @pl.program
        class LogProg:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                result: pl.Tensor[[16, 16], pl.FP32] = pl.log(src, high_precision=True)
                return result

        log_line = self._op_line(self._generate_mlir(LogProg), "pto.tlog")
        assert strip_loc(log_line).endswith("{precisionType = #pto<log_precision high_precision>}")

    def test_trecip_default_omits_and_high_precision_appends_exact_attr_on_a5(self):
        @pl.program
        class DefaultProg:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                result: pl.Tensor[[16, 16], pl.FP32] = pl.recip(src)
                return result

        @pl.program
        class HighPrecisionProg:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                result: pl.Tensor[[16, 16], pl.FP32] = pl.recip(src, high_precision=True)
                return result

        backend_type = BackendType.Ascend950
        default_line = self._op_line(self._generate_mlir(DefaultProg, backend_type), "pto.trecip")
        high_precision_line = self._op_line(
            self._generate_mlir(HighPrecisionProg, backend_type), "pto.trecip"
        )
        assert "precisionType" not in default_line
        assert strip_loc(high_precision_line).endswith(
            "{precisionType = #pto<recip_precision high_precision>}"
        )

    def test_trowexpandadd_emits_two_and_three_operand_forms(self):
        @pl.program
        class WithoutTmpProg:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                row: pl.Tensor[[16, 1], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                src_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(src, [0, 0], [16, 16])
                row_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(row, [0, 0], [16, 1])
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.row_expand_add(src_tile, row_tile)
                return pl.store(result, [0, 0], out)

        @pl.program
        class WithTmpProg:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                row: pl.Tensor[[16, 1], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                src_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(src, [0, 0], [16, 16])
                row_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(row, [0, 0], [16, 1])
                tmp: pl.Tile[[16, 16], pl.FP32] = pl.tile.create(
                    [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.row_expand_add(src_tile, row_tile, tmp)
                return pl.store(result, [0, 0], out)

        two_operand_line = self._op_line(self._generate_mlir(WithoutTmpProg), "pto.trowexpandadd")
        three_operand_line = self._op_line(self._generate_mlir(WithTmpProg), "pto.trowexpandadd")
        assert self._ins_operand_count(two_operand_line) == 2
        assert self._ins_operand_count(three_operand_line) == 3


class TestB02SelectionAndPreluCodegen:
    """Exact PTOAS forms for TSELS and TPRELU."""

    def _generate_mlir(self, program_cls, backend_type=BackendType.Ascend910B) -> str:
        backend.reset_for_testing()
        backend.set_backend_type(backend_type)

        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
        funcs = list(optimized.functions.values())
        assert funcs, "Program has no functions"
        single = ir.Program([funcs[0]], funcs[0].name, optimized.span)
        return codegen.PTOCodegen().generate(single)

    @staticmethod
    def _op_line(mlir: str, op_name: str) -> str:
        line = next((line for line in mlir.splitlines() if op_name in line), "")
        assert line, f"{op_name} not found in MLIR:\n{mlir}"
        return line

    @staticmethod
    def _ins_outs_ssas(line: str) -> tuple[list[str], list[str]]:
        match = re.fullmatch(r"\s*pto\.\w+\s+ins\((.*?)\)\s+outs\((.*?)\)\s*", line)
        assert match, f"expected exact PTO ins(...)/outs(...) form, got: {line}"
        ins = re.findall(r"%[\w.$-]+", match.group(1).split(":", 1)[0])
        outs = re.findall(r"%[\w.$-]+", match.group(2).split(":", 1)[0])
        return ins, outs

    @staticmethod
    def _assert_named_ssas(ssas: list[str], expected_names: list[str]) -> None:
        assert len(ssas) == len(expected_names)
        for ssa, expected_name in zip(ssas, expected_names, strict=True):
            assert expected_name in ssa, f"expected {expected_name!r} SSA at this position, got {ssa!r}"

    @staticmethod
    def _alloc_addr_for_named_ssa(mlir: str, expected_name: str) -> str:
        lines = [
            line
            for line in mlir.splitlines()
            if "= pto.alloc_tile" in line and expected_name in line.split("=", 1)[0]
        ]
        assert len(lines) == 1, f"expected one alloc_tile for {expected_name!r}, got {lines}:\n{mlir}"
        match = re.search(r"addr = (%[\w.$-]+)", lines[0])
        assert match, f"expected a baked PyPTO address in: {lines[0]}"
        return match.group(1)

    def test_tsels_emits_mask_src_tmp_and_typed_scalar(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.INT32],
                out: pl.Tensor[[16, 16], pl.INT32],
            ) -> pl.Tensor[[16, 16], pl.INT32]:
                src_tile: pl.Tile[[16, 16], pl.INT32] = pl.load(src, [0, 0], [16, 16])
                mask: pl.Tile[[16, 32], pl.UINT8] = pl.tile.cmps(src_tile, 0, cmp_type=4)
                tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
                result: pl.Tile[[16, 16], pl.INT32] = pl.tile.sels(mask, src_tile, tmp, -3)
                return pl.store(result, [0, 0], out)

        for backend_type in (BackendType.Ascend910B, BackendType.Ascend950):
            mlir = self._generate_mlir(Prog, backend_type)
            line = self._op_line(mlir, "pto.tsels")
            ins, outs = self._ins_outs_ssas(line)
            scalar = re.search(r"^\s*(%[\w.$-]+)\s*=\s*arith\.constant -3 : i32\s*$", mlir, re.MULTILINE)
            assert scalar
            self._assert_named_ssas(ins[:3], ["mask", "src_tile", "tmp"])
            assert ins[3:] == [scalar.group(1)]
            self._assert_named_ssas(outs, ["result"])
            assert "i32" in line

    def test_tsels_unsigned_src_emits_signed_bit_compatible_scalar(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.UINT32],
                out: pl.Tensor[[16, 16], pl.UINT32],
            ) -> pl.Tensor[[16, 16], pl.UINT32]:
                src_tile: pl.Tile[[16, 16], pl.UINT32] = pl.load(src, [0, 0], [16, 16])
                mask: pl.Tile[[16, 32], pl.UINT8] = pl.tile.cmps(src_tile, 0, cmp_type=4)
                tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
                result: pl.Tile[[16, 16], pl.UINT32] = pl.tile.sels(mask, src_tile, tmp, 0x8000000B)
                return pl.store(result, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        line = self._op_line(mlir, "pto.tsels")
        ins, outs = self._ins_outs_ssas(line)
        scalar = re.search(
            r"^\s*(%[\w.$-]+)\s*=\s*arith\.constant -2147483637 : i32\s*$",
            mlir,
            re.MULTILINE,
        )
        assert scalar
        self._assert_named_ssas(ins[:3], ["mask", "src_tile", "tmp"])
        assert ins[3:] == [scalar.group(1)]
        self._assert_named_ssas(outs, ["result"])
        assert "ui32" in line
        assert "i32" in line

    def test_tsels_rejects_int8_on_a2a3(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.INT8],
                out: pl.Tensor[[16, 16], pl.INT8],
            ) -> pl.Tensor[[16, 16], pl.INT8]:
                src_tile: pl.Tile[[16, 16], pl.INT8] = pl.load(src, [0, 0], [16, 16])
                mask: pl.Tile[[16, 32], pl.UINT8] = pl.tile.cmps(src_tile, 0, cmp_type=4)
                tmp: pl.Tile[[1, 1], pl.UINT8] = pl.tile.create([1, 1], dtype=pl.UINT8)
                result: pl.Tile[[16, 16], pl.INT8] = pl.tile.sels(mask, src_tile, tmp, -3)
                return pl.store(result, [0, 0], out)

        with pytest.raises(ValueError, match="only supported on the 'a5' backend"):
            self._generate_mlir(Prog)

    def test_tsels_tmp_may_alias_src_only_on_a5(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mask_in: pl.Tensor[[2, 16], pl.INT32],
                src: pl.Tensor[[2, 16], pl.INT32],
                out: pl.Tensor[[2, 16], pl.INT32],
            ) -> pl.Tensor[[2, 16], pl.INT32]:
                mask: pl.Tile[[2, 16], pl.INT32] = pl.load(mask_in, [0, 0], [2, 16])
                src_tile: pl.Tile[[2, 16], pl.INT32] = pl.load(src, [0, 0], [2, 16])
                result: pl.Tile[[2, 16], pl.INT32] = pl.tile.sels(mask, src_tile, src_tile, -3)
                return pl.store(result, [0, 0], out)

        with pytest.raises(ValueError, match="tmp overlaps src"):
            self._generate_mlir(Prog, BackendType.Ascend910B)

        a5_line = self._op_line(self._generate_mlir(Prog, BackendType.Ascend950), "pto.tsels")
        a5_ins, a5_outs = self._ins_outs_ssas(a5_line)
        assert a5_ins[1] == a5_ins[2]
        self._assert_named_ssas(a5_outs, ["result"])

    def test_tsels_tmp_may_alias_mask_only_on_a5(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mask_in: pl.Tensor[[2, 16], pl.INT32],
                src: pl.Tensor[[2, 16], pl.INT32],
                out: pl.Tensor[[2, 16], pl.INT32],
            ) -> pl.Tensor[[2, 16], pl.INT32]:
                mask: pl.Tile[[2, 16], pl.INT32] = pl.load(mask_in, [0, 0], [2, 16])
                src_tile: pl.Tile[[2, 16], pl.INT32] = pl.load(src, [0, 0], [2, 16])
                result: pl.Tile[[2, 16], pl.INT32] = pl.tile.sels(mask, src_tile, mask, -3)
                return pl.store(result, [0, 0], out)

        with pytest.raises(ValueError, match="tmp overlaps mask"):
            self._generate_mlir(Prog, BackendType.Ascend910B)

        a5_line = self._op_line(self._generate_mlir(Prog, BackendType.Ascend950), "pto.tsels")
        a5_ins, a5_outs = self._ins_outs_ssas(a5_line)
        assert a5_ins[0] == a5_ins[2]
        self._assert_named_ssas(a5_outs, ["result"])

    def test_tsels_a2a3_rejects_overlapping_tmp_view(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mask_in: pl.Tensor[[2, 16], pl.INT32],
                src: pl.Tensor[[2, 24], pl.INT32],
                out: pl.Tensor[[2, 16], pl.INT32],
            ) -> pl.Tensor[[2, 16], pl.INT32]:
                mask: pl.Tile[[2, 16], pl.INT32] = pl.load(mask_in, [0, 0], [2, 16])
                base: pl.Tile[[2, 24], pl.INT32] = pl.load(src, [0, 0], [2, 24])
                src_view: pl.Tile[[2, 16], pl.INT32] = pl.tile.slice(base, [2, 16], [0, 0])
                tmp_view: pl.Tile[[2, 16], pl.INT32] = pl.tile.slice(base, [2, 16], [0, 8])
                result: pl.Tile[[2, 16], pl.INT32] = pl.tile.sels(mask, src_view, tmp_view, -3)
                return pl.store(result, [0, 0], out)

        with pytest.raises(ValueError, match="tmp overlaps src"):
            self._generate_mlir(Prog, BackendType.Ascend910B)

        assert "pto.tsels" in self._generate_mlir(Prog, BackendType.Ascend950)

    def test_tsels_a2a3_keeps_caller_tmp(self):
        """A2/A3 preserves an explicit caller tmp instead of synthesizing sels_tmp."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mask_in: pl.Tensor[[2, 16], pl.INT32],
                src_in: pl.Tensor[[2, 16], pl.INT32],
                tmp_in: pl.Tensor[[2, 16], pl.INT32],
                out: pl.Tensor[[2, 16], pl.INT32],
            ) -> pl.Tensor[[2, 16], pl.INT32]:
                mask: pl.Tile[[2, 16], pl.INT32] = pl.load(mask_in, [0, 0], [2, 16])
                src: pl.Tile[[2, 16], pl.INT32] = pl.load(src_in, [0, 0], [2, 16])
                tmp: pl.Tile[[2, 16], pl.INT32] = pl.load(tmp_in, [0, 0], [2, 16])
                result: pl.Tile[[2, 16], pl.INT32] = pl.tile.sels(mask, src, tmp, -3)
                keep_src_live: pl.Tile[[2, 16], pl.INT32] = pl.tile.add(src, result)
                return pl.store(keep_src_live, [0, 0], out)

        mlir = self._generate_mlir(Prog, BackendType.Ascend910B)
        line = self._op_line(mlir, "pto.tsels")
        ins, outs = self._ins_outs_ssas(line)
        self._assert_named_ssas(ins[:3], ["mask", "src", "tmp"])
        self._assert_named_ssas(outs, ["result"])

    def test_tprelu_emits_target_specific_exact_operands(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                slope: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                src_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(src, [0, 0], [16, 16])
                slope_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(slope, [0, 0], [16, 16])
                tmp: pl.Tile[[17, 32], pl.UINT8] = pl.tile.create([17, 32], dtype=pl.UINT8)
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.prelu(src_tile, slope_tile, tmp)
                return pl.store(result, [0, 0], out)

        a3_line = self._op_line(self._generate_mlir(Prog, BackendType.Ascend910B), "pto.tprelu")
        a3_ins, a3_outs = self._ins_outs_ssas(a3_line)
        self._assert_named_ssas(a3_ins, ["prelu_src_view", "prelu_slope_view", "prelu_tmp_view"])
        self._assert_named_ssas(a3_outs, ["prelu_dst_view"])
        assert "v_row=16" in a3_line and "v_row=?" not in a3_line.split("pto.tprelu", 1)[1]

        a5_line = self._op_line(self._generate_mlir(Prog, BackendType.Ascend950), "pto.tprelu")
        a5_ins, a5_outs = self._ins_outs_ssas(a5_line)
        self._assert_named_ssas(a5_ins, ["src_tile", "slope_tile", "tmp"])
        self._assert_named_ssas(a5_outs, ["result"])

    def test_tprelu_signed_scratch_is_a5_only(self):
        """A2/A3 requires UINT8 scratch; caller-provided signed tmp is rejected there."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                slope: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                src_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(src, [0, 0], [16, 16])
                slope_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(slope, [0, 0], [16, 16])
                tmp: pl.Tile[[17, 32], pl.INT8] = pl.tile.create([17, 32], dtype=pl.INT8)
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.prelu(src_tile, slope_tile, tmp)
                return pl.store(result, [0, 0], out)

        with pytest.raises(ValueError, match="UINT8 tmp scratch"):
            self._generate_mlir(Prog, BackendType.Ascend910B)

        a5_line = self._op_line(self._generate_mlir(Prog, BackendType.Ascend950), "pto.tprelu")
        a5_ins, a5_outs = self._ins_outs_ssas(a5_line)
        self._assert_named_ssas(a5_ins, ["src_tile", "slope_tile", "tmp"])
        self._assert_named_ssas(a5_outs, ["result"])

    def test_tprelu_a3_rejects_overlapping_views_but_a5_accepts_them(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 24], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                base: pl.Tile[[16, 24], pl.FP32] = pl.load(src, [0, 0], [16, 24])
                src_view: pl.Tile[[16, 16], pl.FP32] = pl.tile.slice(base, [16, 16], [0, 0])
                slope_view: pl.Tile[[16, 16], pl.FP32] = pl.tile.slice(base, [16, 16], [0, 8])
                tmp: pl.Tile[[17, 32], pl.UINT8] = pl.tile.create([17, 32], dtype=pl.UINT8)
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.prelu(src_view, slope_view, tmp)
                return pl.store(result, [0, 0], out)

        with pytest.raises(ValueError, match="src overlaps slope"):
            self._generate_mlir(Prog, BackendType.Ascend910B)

        mlir = self._generate_mlir(Prog, BackendType.Ascend950)
        line = self._op_line(mlir, "pto.tprelu")
        ins, outs = self._ins_outs_ssas(line)
        subview_results = [
            subview_line.split("=", 1)[0].strip()
            for subview_line in mlir.splitlines()
            if "= pto.subview " in subview_line
        ]
        assert ins[:2] == subview_results[:2]
        self._assert_named_ssas(ins[2:], ["tmp"])
        self._assert_named_ssas(outs, ["result"])

    def test_tprelu_undersized_tmp_is_rejected_on_a2a3(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                slope: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                src_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(src, [0, 0], [16, 16])
                slope_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(slope, [0, 0], [16, 16])
                tmp: pl.Tile[[1, 1], pl.UINT8] = pl.tile.create([1, 1], dtype=pl.UINT8)
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.prelu(src_tile, slope_tile, tmp)
                return pl.store(result, [0, 0], out)

        with pytest.raises(ValueError, match="physical rows"):
            self._generate_mlir(Prog, BackendType.Ascend910B)

        line = self._op_line(self._generate_mlir(Prog, BackendType.Ascend950), "pto.tprelu")
        ins, outs = self._ins_outs_ssas(line)
        self._assert_named_ssas(ins, ["src_tile", "slope_tile", "tmp"])
        self._assert_named_ssas(outs, ["result"])

    def test_tprelu_a3_rejects_tmp_with_insufficient_valid_columns(self):
        """Exercise the packed-column bound independently of the row bound."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                slope: pl.Tensor[[16, 16], pl.FP32],
                tmp_in: pl.Tensor[[17, 32], pl.UINT8],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                src_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(src, [0, 0], [16, 16])
                slope_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(slope, [0, 0], [16, 16])
                tmp: pl.Tile[[17, 32], pl.UINT8] = pl.load(tmp_in, [0, 0], [17, 32], valid_shape=[17, 1])
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.prelu(src_tile, slope_tile, tmp)
                return pl.store(result, [0, 0], out)

        with pytest.raises(ValueError, match="valid columns"):
            self._generate_mlir(Prog, BackendType.Ascend910B)

        line = self._op_line(self._generate_mlir(Prog, BackendType.Ascend950), "pto.tprelu")
        ins, outs = self._ins_outs_ssas(line)
        self._assert_named_ssas(ins, ["src_tile", "slope_tile", "tmp"])
        self._assert_named_ssas(outs, ["result"])

    def test_tprelu_a2a3_rejects_unproven_dynamic_tmp_rows(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                slope: pl.Tensor[[16, 16], pl.FP32],
                tmp_in: pl.Tensor[[17, 32], pl.UINT8],
                out: pl.Tensor[[16, 16], pl.FP32],
                rows: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                src_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(src, [0, 0], [16, 16], valid_shape=[rows, 16])
                slope_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
                    slope, [0, 0], [16, 16], valid_shape=[rows, 16]
                )
                tmp: pl.Tile[[17, 32], pl.UINT8] = pl.load(tmp_in, [0, 0], [17, 32])
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.prelu(src_tile, slope_tile, tmp)
                return pl.store(result, [0, 0], out)

        with pytest.raises(ValueError, match="physical rows"):
            self._generate_mlir(Prog, BackendType.Ascend910B)

    def test_tprelu_a2a3_rejects_unproven_dynamic_tmp_columns(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                slope: pl.Tensor[[16, 16], pl.FP32],
                tmp_in: pl.Tensor[[17, 32], pl.UINT8],
                out: pl.Tensor[[16, 16], pl.FP32],
                cols: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                src_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(src, [0, 0], [16, 16], valid_shape=[16, cols])
                slope_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
                    slope, [0, 0], [16, 16], valid_shape=[16, cols]
                )
                tmp: pl.Tile[[17, 32], pl.UINT8] = pl.load(tmp_in, [0, 0], [17, 32], valid_shape=[17, 2])
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.prelu(src_tile, slope_tile, tmp)
                return pl.store(result, [0, 0], out)

        with pytest.raises(ValueError, match="valid columns"):
            self._generate_mlir(Prog, BackendType.Ascend910B)


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

    def test_tile_read_variable_2d(self):
        """2D tile [4, 8], tile.read(t, [row, col]) with variable indices -> arith.muli/arith.addi SSA."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[4, 8], pl.FP32],
                config: pl.Tensor[[2], pl.INT64],
                dst: pl.Tensor[[4, 8], pl.FP32],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                t: pl.Tile[[4, 8], pl.FP32] = pl.load(src, [0, 0], [4, 8])
                row: pl.Scalar[pl.INT64] = pl.read(config, [0])
                col: pl.Scalar[pl.INT64] = pl.read(config, [1])
                val: pl.Scalar[pl.FP32] = pl.tile.read(t, [row, col])
                pl.tile.write(t, [0, 0], val)
                return pl.store(t, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.tgetval" in mlir, f"Expected pto.tgetval for tile.read, got:\n{mlir}"
        assert "arith.muli" in mlir, f"Expected arith.muli for dynamic flat offset, got:\n{mlir}"
        assert "arith.addi" in mlir, f"Expected arith.addi for dynamic flat offset, got:\n{mlir}"
        assert "arith.index_cast" in mlir, f"Expected arith.index_cast for INT64 -> index, got:\n{mlir}"

    def test_tile_write_variable_2d(self):
        """2D tile.write with variable indices emits arith.muli/arith.addi SSA."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[4, 8], pl.FP32],
                config: pl.Tensor[[2], pl.INT64],
                dst: pl.Tensor[[4, 8], pl.FP32],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                t: pl.Tile[[4, 8], pl.FP32] = pl.load(src, [0, 0], [4, 8])
                row: pl.Scalar[pl.INT64] = pl.read(config, [0])
                col: pl.Scalar[pl.INT64] = pl.read(config, [1])
                val: pl.Scalar[pl.FP32] = pl.tile.read(t, [0, 0])
                pl.tile.write(t, [row, col], val)
                return pl.store(t, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.tsetval" in mlir, f"Expected pto.tsetval for tile.write, got:\n{mlir}"
        assert "arith.muli" in mlir, f"Expected arith.muli for dynamic flat offset, got:\n{mlir}"
        assert "arith.addi" in mlir, f"Expected arith.addi for dynamic flat offset, got:\n{mlir}"

    def test_tile_read_variable_2d_dynamic_i64_shape_stride(self):
        """2D tile.read with an i64 runtime shape dim must cast the stride operand to index."""

        span = ir.Span.unknown()
        rows = ir.Var("rows", ir.ScalarType(pl.INT64), span)
        cols = ir.Var("cols", ir.ScalarType(pl.INT64), span)
        row = ir.Var("row", ir.ScalarType(pl.INT64), span)
        col = ir.Var("col", ir.ScalarType(pl.INT64), span)
        memref = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, pl.INT64, span), 128 * 128 * 4, 0)
        tile_view = ir.TileView(valid_shape=[rows, cols])
        tile_type = ir.TileType([rows, cols], pl.FP32, memref, tile_view, ir.MemorySpace.Vec)
        tile = ir.Var("tile", tile_type, span)
        val = ir.Var("val", ir.ScalarType(pl.FP32), span)
        indices = ir.MakeTuple([row, col], span)
        read_call = ir.Call(ir.Op("tile.read"), [tile, indices], {}, ir.ScalarType(pl.FP32), span)
        body = ir.SeqStmts([ir.AssignStmt(val, read_call, span)], span)
        func = ir.Function(
            "dynamic_shape_tile_read",
            [
                (tile, ir.ParamDirection.In),
                (rows, ir.ParamDirection.In),
                (cols, ir.ParamDirection.In),
                (row, ir.ParamDirection.In),
                (col, ir.ParamDirection.In),
            ],
            [],
            body,
            span,
            ir.FunctionType.AIV,
        )

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        mlir = codegen.PTOCodegen().generate(ir.Program([func], "dynamic_shape_tile_read_program", span))

        assert "pto.tgetval" in mlir, f"Expected pto.tgetval for tile.read, got:\n{mlir}"
        assert "arith.muli" in mlir, f"Expected arith.muli for dynamic flat offset, got:\n{mlir}"
        assert "arith.addi" in mlir, f"Expected arith.addi for dynamic flat offset, got:\n{mlir}"
        assert "%cols_idx = arith.index_cast" in mlir, f"Expected i64 shape dim cast to index, got:\n{mlir}"
        assert "%cols_idx" in next(
            (line for line in mlir.splitlines() if "arith.muli" in line and "flat_offset_mul" in line),
            "",
        ), f"Expected dynamic stride multiply to use the casted cols index, got:\n{mlir}"

    def test_tile_read_variable_2d_partial(self):
        """2D tile [1, 8], tile.read(t, [0, col]) with variable col -> arith.muli/arith.addi still emitted.

        Even though the row index is constant 0, the index tuple has 2 elements so
        EmitFlatOffsetSSAFromValues computes offset = 0 * 8 + col via arith.muli/arith.addi.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[1, 8], pl.FP32],
                config: pl.Tensor[[1], pl.INT64],
                dst: pl.Tensor[[1, 8], pl.FP32],
            ) -> pl.Tensor[[1, 8], pl.FP32]:
                t: pl.Tile[[1, 8], pl.FP32] = pl.load(src, [0, 0], [1, 8])
                col: pl.Scalar[pl.INT64] = pl.read(config, [0])
                val: pl.Scalar[pl.FP32] = pl.tile.read(t, [0, col])
                pl.tile.write(t, [0, 0], val)
                return pl.store(t, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.tgetval" in mlir, f"Expected pto.tgetval for tile.read, got:\n{mlir}"
        # 2D index [0, col] has 2 elements, so EmitFlatOffsetSSAFromValues still
        # emits arith.muli/arith.addi (offset = 0 * stride + col).
        assert "arith.muli" in mlir, f"Expected arith.muli for 2D partial-constant index, got:\n{mlir}"
        assert "arith.addi" in mlir, f"Expected arith.addi for 2D partial-constant index, got:\n{mlir}"

    def test_store_dynamic_offset_clamped_to_non_negative(self):
        """pl.store with a dynamic offset emits arith.maxsi clamp against zero.

        A runtime INT32 scalar used as a store offset (e.g. a rank-derived ring
        index) could evaluate to a negative value.  The offset must be clamped
        to non-negative before it reaches pto.partition_view, which does not
        handle negative indices.  Regression for issue #2201.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[1, 8], pl.FP32],
                config: pl.Tensor[[1], pl.INT32],
                out: pl.Tensor[[1, 32], pl.FP32],
            ) -> pl.Tensor[[1, 32], pl.FP32]:
                offset: pl.Scalar[pl.INT32] = pl.read(config, [0])
                ones: pl.Tile[[1, 8], pl.FP32] = pl.load(src, [0, 0], [1, 8])
                return pl.store(ones, [0, offset], out)

        mlir = self._generate_mlir(Prog)
        assert "arith.maxsi" in mlir, f"Expected arith.maxsi clamp for dynamic store offset, got:\n{mlir}"


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

    def test_col_expand_add_codegen(self):
        """tile.col_expand_add(tile[M,N], col_vec[1,N]) should generate pto.tcolexpandadd."""

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
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.col_expand_add(src_tile, col_tile)
                return pl.store(result, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.tcolexpandadd" in mlir, f"col_expand_add should generate pto.tcolexpandadd, got:\n{mlir}"

    def test_col_expand_div_codegen(self):
        """tile.col_expand_div(tile[M,N], col_vec[1,N]) should generate pto.tcolexpanddiv."""

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
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.col_expand_div(src_tile, col_tile)
                return pl.store(result, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.tcolexpanddiv" in mlir, f"col_expand_div should generate pto.tcolexpanddiv, got:\n{mlir}"

    def test_col_expand_sub_codegen(self):
        """tile.col_expand_sub(tile[M,N], col_vec[1,N]) should generate pto.tcolexpandsub."""

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
                result: pl.Tile[[16, 16], pl.FP32] = pl.tile.col_expand_sub(src_tile, col_tile)
                return pl.store(result, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.tcolexpandsub" in mlir, f"col_expand_sub should generate pto.tcolexpandsub, got:\n{mlir}"

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

    def test_unaligned_vec_column_slice_into_muls_uses_aligned_extract(self):
        """Regression for #1789: an FP32 column subview at offset 1 starts four
        bytes past its source allocation and must not feed ``pto.tmuls``
        directly. The pass materializes it into a fresh aligned tile first."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 8], pl.FP32],
                dst: pl.Tensor[[16, 1], pl.FP32],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[16, 8], pl.FP32] = pl.load(src, [0, 0], [16, 8])
                head: pl.Tile[[16, 1], pl.FP32] = pl.tile.slice(local, [16, 1], [0, 1])
                scaled: pl.Tile[[16, 1], pl.FP32] = pl.tile.muls(head, 0.5)
                return pl.store(scaled, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.subview" not in mlir, (
            f"unaligned Vec slice must be canonicalized to tile.extract, got:\n{mlir}"
        )
        textract_lines = [line.strip() for line in mlir.splitlines() if "pto.textract" in line]
        tmuls_lines = [line.strip() for line in mlir.splitlines() if "pto.tmuls" in line]
        assert len(textract_lines) == 1 and len(tmuls_lines) == 1, (
            f"expected one aligned extract followed by one tmuls, got:\n{mlir}"
        )

        textract_out = textract_lines[0].split("outs(", 1)[1].split(":", 1)[0].strip()
        tmuls_in = tmuls_lines[0].split("ins(", 1)[1].split(":", 1)[0].split(",", 1)[0].strip()
        assert mlir.index("pto.textract") < mlir.index("pto.tmuls"), (
            f"the aligned extract must precede pto.tmuls, got:\n{mlir}"
        )

        # Layout resolution may insert a row-major conversion between textract
        # and tmuls. Both the materialized tile and the final tmuls operand must
        # therefore have independent, aligned allocations; direct SSA identity
        # is not required.
        for value in (textract_out, tmuls_in):
            alloc_line = next(
                (line for line in mlir.splitlines() if line.strip().startswith(f"{value} = pto.alloc_tile")),
                None,
            )
            assert alloc_line is not None, f"{value} must have its own allocation, got:\n{mlir}"
            address = alloc_line.split("addr = ", 1)[1].split()[0]
            constant = re.fullmatch(r"%c(-?\d+)_i64", address)
            assert constant is not None, f"expected a static allocation address, got: {alloc_line}"
            assert int(constant.group(1)) % 32 == 0, (
                f"Vec operand address must be 32-byte aligned: {alloc_line}"
            )

    @pytest.mark.parametrize(
        "op_name, pto_op",
        [
            ("col_expand_mul", "pto.tcolexpandmul"),
            ("col_expand_add", "pto.tcolexpandadd"),
            ("col_expand_div", "pto.tcolexpanddiv"),
            ("col_expand_sub", "pto.tcolexpandsub"),
        ],
    )
    def test_tile_slice_into_col_expand_materializes_via_extract(self, op_name, pto_op):
        """Regression for #1640: a dynamic-offset Vec ``tile.slice`` feeding
        ``tile.col_expand_mul`` / ``tile.col_expand_add`` must NOT be materialized
        into the slice's own (source-aliasing) result buffer.

        ``pto.tcolexpandmul`` / ``pto.tcolexpandadd`` cannot consume a
        ``pto.subview`` operand, so codegen used to lazily emit
        ``pto.textract ins(src, off) outs(slice_buf)``.  For a dynamic offset the
        slice buffer inherits the source allocation base, so the materialization
        overwrote the source tile's row 0.  ``CanonicalizeTileSlice`` now rewrites
        the operand to a fresh ``tile.extract`` (own non-inherited allocation), so
        the slice's ``pto.subview`` disappears and the materialization lands in a
        distinct buffer.
        """
        if op_name == "col_expand_mul":

            @pl.program
            class ProgMul:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    scores: pl.Tensor[[16, 256], pl.FP32],
                    gamma: pl.Tensor[[1, 256], pl.FP32],
                    row_off: pl.Scalar[pl.INDEX],
                    dst: pl.Tensor[[1, 256], pl.FP32],
                ) -> pl.Tensor[[1, 256], pl.FP32]:
                    local: pl.Tile[[16, 256], pl.FP32] = pl.load(scores, [0, 0], [16, 256])
                    gamma_t: pl.Tile[[1, 256], pl.FP32] = pl.load(gamma, [0, 0], [1, 256])
                    # Dynamic-offset slice of a local tile — the #1640 hazard.
                    row: pl.Tile[[1, 256], pl.FP32] = pl.tile.slice(local, [1, 256], [row_off, 0])
                    scaled: pl.Tile[[1, 256], pl.FP32] = pl.tile.col_expand_mul(row, gamma_t)
                    return pl.store(scaled, [0, 0], dst)

            prog = ProgMul
        elif op_name == "col_expand_add":

            @pl.program
            class ProgAdd:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    scores: pl.Tensor[[16, 256], pl.FP32],
                    gamma: pl.Tensor[[1, 256], pl.FP32],
                    row_off: pl.Scalar[pl.INDEX],
                    dst: pl.Tensor[[1, 256], pl.FP32],
                ) -> pl.Tensor[[1, 256], pl.FP32]:
                    local: pl.Tile[[16, 256], pl.FP32] = pl.load(scores, [0, 0], [16, 256])
                    gamma_t: pl.Tile[[1, 256], pl.FP32] = pl.load(gamma, [0, 0], [1, 256])
                    # Dynamic-offset slice of a local tile — the #1640 hazard.
                    row: pl.Tile[[1, 256], pl.FP32] = pl.tile.slice(local, [1, 256], [row_off, 0])
                    scaled: pl.Tile[[1, 256], pl.FP32] = pl.tile.col_expand_add(row, gamma_t)
                    return pl.store(scaled, [0, 0], dst)

            prog = ProgAdd
        elif op_name == "col_expand_div":

            @pl.program
            class ProgDiv:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    scores: pl.Tensor[[16, 256], pl.FP32],
                    gamma: pl.Tensor[[1, 256], pl.FP32],
                    row_off: pl.Scalar[pl.INDEX],
                    dst: pl.Tensor[[1, 256], pl.FP32],
                ) -> pl.Tensor[[1, 256], pl.FP32]:
                    local: pl.Tile[[16, 256], pl.FP32] = pl.load(scores, [0, 0], [16, 256])
                    gamma_t: pl.Tile[[1, 256], pl.FP32] = pl.load(gamma, [0, 0], [1, 256])
                    # Dynamic-offset slice of a local tile — the #1640 hazard.
                    row: pl.Tile[[1, 256], pl.FP32] = pl.tile.slice(local, [1, 256], [row_off, 0])
                    scaled: pl.Tile[[1, 256], pl.FP32] = pl.tile.col_expand_div(row, gamma_t)
                    return pl.store(scaled, [0, 0], dst)

            prog = ProgDiv
        else:

            @pl.program
            class ProgSub:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    scores: pl.Tensor[[16, 256], pl.FP32],
                    gamma: pl.Tensor[[1, 256], pl.FP32],
                    row_off: pl.Scalar[pl.INDEX],
                    dst: pl.Tensor[[1, 256], pl.FP32],
                ) -> pl.Tensor[[1, 256], pl.FP32]:
                    local: pl.Tile[[16, 256], pl.FP32] = pl.load(scores, [0, 0], [16, 256])
                    gamma_t: pl.Tile[[1, 256], pl.FP32] = pl.load(gamma, [0, 0], [1, 256])
                    # Dynamic-offset slice of a local tile — the #1640 hazard.
                    row: pl.Tile[[1, 256], pl.FP32] = pl.tile.slice(local, [1, 256], [row_off, 0])
                    scaled: pl.Tile[[1, 256], pl.FP32] = pl.tile.col_expand_sub(row, gamma_t)
                    return pl.store(scaled, [0, 0], dst)

            prog = ProgSub

        mlir = self._generate_mlir(prog)
        assert pto_op in mlir, f"{op_name} should still lower to {pto_op}, got:\n{mlir}"
        # The slice is canonicalized to tile.extract, so its subview is gone and
        # the materialization is a pto.textract into a fresh buffer. With the bug
        # present, the slice emits a pto.subview and the textract writes into the
        # (source-aliasing) slice buffer instead.
        assert "pto.subview" not in mlir, (
            f"slice feeding {op_name} must be canonicalized to tile.extract (no subview), got:\n{mlir}"
        )
        assert "pto.textract" in mlir, f"materialization should be a pto.textract, got:\n{mlir}"

        # The col-expand operand must be the pto.textract's fresh output buffer
        # (materialized into a distinct tile), not the source tile.
        textract_lines = [ln.strip() for ln in mlir.splitlines() if "pto.textract" in ln]
        colexpand_lines = [ln.strip() for ln in mlir.splitlines() if pto_op in ln]
        assert textract_lines and colexpand_lines, f"expected both ops, got:\n{mlir}"

        def _ssa_tokens(clause: str) -> list[str]:
            # Operands before the optional `: type` annotation, split on commas.
            head = clause.split(":", 1)[0]
            return [tok.strip() for tok in head.split(",") if tok.strip().startswith("%")]

        tex = textract_lines[0]
        tex_src = _ssa_tokens(tex.split("ins(", 1)[1])[0]
        tex_out = _ssa_tokens(tex.split("outs(", 1)[1])[0]
        assert tex_out != tex_src, f"pto.textract must not write into its own source, got:\n{tex}"

        cem_ins = _ssa_tokens(colexpand_lines[0].split("ins(", 1)[1])
        assert tex_out in cem_ins, (
            f"{pto_op} must consume the freshly-materialized extract buffer {tex_out}, got:\n"
            f"{colexpand_lines[0]}"
        )

    def test_column_slice_of_multirow_tile_into_col_expand_uses_disjoint_buffer(self):
        """Regression for #2010: ``t[:, a:b]`` on a multi-row tile feeding
        ``col_expand_mul`` must materialize into a buffer disjoint from its source.

        The slice's own buffer is dense (row pitch 64) yet aliases the source (row
        pitch 128), so the lazy ``pto.textract`` used to repack strided -> dense on
        top of its own live source and destroy it — only row 0 survived. The fix
        canonicalizes the slice into a ``tile.extract``, which gets a fresh
        allocation. Both the SSA identity *and* the allocated addresses are checked:
        the pre-fix codegen already emitted a distinct destination SSA, and it was
        the *address* the destination landed on that made it corrupt.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 128], pl.FP32],
                gamma: pl.Tensor[[1, 64], pl.FP32],
                dst: pl.Tensor[[16, 64], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                t: pl.Tile[[16, 128], pl.FP32] = pl.load(x, [0, 0], [16, 128])
                gamma_t: pl.Tile[[1, 64], pl.FP32] = pl.load(gamma, [0, 0], [1, 64])
                hi: pl.Tile[[16, 64], pl.FP32] = t[:, 64:128]
                scaled: pl.Tile[[16, 64], pl.FP32] = pl.tile.col_expand_mul(hi, gamma_t)
                return pl.store(scaled, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        assert "pto.subview" not in mlir, (
            f"a column slice feeding col_expand_mul must be canonicalized to tile.extract, got:\n{mlir}"
        )

        textract_lines = [ln.strip() for ln in mlir.splitlines() if "pto.textract" in ln]
        assert len(textract_lines) == 1, f"expected exactly one pto.textract, got:\n{mlir}"

        def _first_ssa(clause: str) -> str:
            return clause.split(":", 1)[0].split(",")[0].strip()

        src = _first_ssa(textract_lines[0].split("ins(", 1)[1])
        dst_ssa = _first_ssa(textract_lines[0].split("outs(", 1)[1])
        assert dst_ssa != src, f"pto.textract must not write into its own source, got:\n{textract_lines[0]}"

        # The destination must also be allocated at a different address: sharing the
        # source's base is exactly the #2010 corruption, whatever the SSA names say.
        addrs = {}
        for line in mlir.splitlines():
            if "pto.alloc_tile" not in line:
                continue
            name = line.strip().split("=", 1)[0].strip()
            addrs[name] = line.split("addr = ", 1)[1].split()[0]
        assert src in addrs and dst_ssa in addrs, f"both tiles must be allocated, got {sorted(addrs)}"
        assert addrs[src] != addrs[dst_ssa], (
            f"pto.textract destination {dst_ssa} is allocated at the source's address "
            f"{addrs[src]} — the repack would corrupt its own source (#2010)"
        )

    def test_dynamic_offset_slice_escaping_the_pass_into_col_expand_is_rejected(self):
        """A dynamic-offset slice that ``CanonicalizeTileSlice`` cannot rewrite must be
        rejected by codegen, not silently materialized onto its own source.

        The pass only canonicalizes plain 3-arg windows, so a rank-reducing ``t[row]``
        (a 5-arg slice carrying drop_dims) escapes it. Its window is a single row —
        contiguous — but the offset is dynamic, and a dynamic offset cannot be folded
        into the source-inherited buffer's address: the destination falls back to the
        source base, so the lazy ``pto.textract`` would extract row ``row`` on top of
        the source's row 0 while the source is still live (#1640). The contiguity
        guard alone lets this through, so the offset must be checked too.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 32], pl.FP32],
                gamma: pl.Tensor[[1, 32], pl.FP32],
                row: pl.Scalar[pl.INDEX],
                dst: pl.Tensor[[1, 32], pl.FP32],
            ) -> pl.Tensor[[1, 32], pl.FP32]:
                t: pl.Tile[[16, 32], pl.FP32] = pl.load(x, [0, 0], [16, 32])
                gamma_t: pl.Tile[[1, 32], pl.FP32] = pl.load(gamma, [0, 0], [1, 32])
                # Rank-reducing subscript -> tile.slice with drop_dims: not a plain
                # 3-arg window, so CanonicalizeTileSlice leaves it alone.
                row_tile: pl.Tile[[1, 32], pl.FP32] = t[row]
                scaled: pl.Tile[[1, 32], pl.FP32] = pl.tile.col_expand_mul(row_tile, gamma_t)
                # Keep the source live past the materialization.
                out: pl.Tile[[1, 32], pl.FP32] = pl.tile.add(scaled, t[0])
                return pl.store(out, [0, 0], dst)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError, match="dynamic-offset tile.slice"):
                self._generate_mlir(Prog)

    def test_tile_slice_codegen_rank_reducing(self):
        """A rank-reducing tile subscript `t[i]` (→ tile.slice with drop_dims) reaches
        PTO codegen and emits pto.subview — the result is clamped to 2D [1, N]."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[32, 32], pl.FP32],
                dst: pl.Tensor[[1, 32], pl.FP32],
                row: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[1, 32], pl.FP32]:
                src_tile: pl.Tile[[32, 32], pl.FP32] = pl.load(src, [0, 0], [32, 32])
                row_tile: pl.Tile[[1, 32], pl.FP32] = src_tile[row]
                return pl.store(row_tile, [0, 0], dst)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlir = self._generate_mlir(Prog)
        assert "pto.subview" in mlir, f"rank-reducing tile.slice should generate pto.subview, got:\n{mlir}"
        subview_lines = [line for line in mlir.splitlines() if "pto.subview" in line]
        assert subview_lines and "sizes [1, 32]" in subview_lines[0], (
            f"subview sizes attribute must be [1, 32], got:\n{subview_lines}"
        )

    def test_tile_slice_codegen_with_valid_shape(self):
        """tile.slice(..., valid_shape=...) emits a single pto.subview with a `valid [...]` clause.

        tile.slice is a pure view; the explicit valid_shape is encoded directly
        into pto.subview's `valid [...]` operand list. No materializing pto.tmov
        or follow-up pto.set_validshape is emitted — that earlier lowering
        aliased the source allocation and corrupted source rows for dynamic
        offsets (issue #1622). See also the matching codegen test
        ``test_pto_codegen_slice_full_window_dynamic_valid_shape_uses_subview_valid``.
        """

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
        subview_lines = [line for line in mlir.splitlines() if "pto.subview" in line]
        assert len(subview_lines) == 1, (
            f"tile.slice with valid_shape should emit exactly one pto.subview, got:\n{mlir}"
        )
        assert "valid [" in subview_lines[0], (
            f"pto.subview should carry a `valid [...]` clause when valid_shape is explicit, got:\n"
            f"{subview_lines[0]}"
        )
        tmov_lines = [line for line in mlir.splitlines() if "pto.tmov" in line]
        assert not tmov_lines, "tile.slice must not emit pto.tmov (pure view), got:\n" + "\n".join(tmov_lines)
        set_vs_lines = [line for line in mlir.splitlines() if "pto.set_validshape" in line]
        assert not set_vs_lines, (
            "tile.slice must not emit pto.set_validshape (valid encoded in subview), got:\n"
            + "\n".join(set_vs_lines)
        )

    def test_tile_slice_mixed_dynamic_static_valid_shape_per_dim_type(self):
        """tile.slice(valid_shape=[dynamic_row, static_col]) must keep the static
        col dim static in the result tile_buf type.

        Regression for the pypto-lib qwen3-14b `lm_head_store` PTOAS failure:
        when valid_shape mixes a dynamic dim (e.g. `pl.min(...)` row count) with
        a static dim, the result tile_buf type must reflect each dim's
        static/dynamic-ness independently. A previous lowering promoted *both*
        dims to dynamic when either was dynamic, emitting `v_col=?` while the
        subview's `valid [...]` operand for that dim was a ConstInt — which
        PTOAS rejects with "'pto.subview' op expects result valid_shape[1] to
        match inferred/explicit valid_col". The static valid operand must pair
        with a static `v_col=<N>` result dim.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 64], pl.FP32],
                valid_rows: pl.Scalar[pl.INDEX],
                dst: pl.Tensor[[16, 64], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                src_tile: pl.Tile[[16, 64], pl.FP32] = pl.load(src, [0, 0], [16, 64])
                sliced: pl.Tile[[16, 64], pl.FP32] = pl.tile.slice(
                    src_tile, [16, 64], [0, 0], valid_shape=[valid_rows, 64]
                )
                return pl.store(sliced, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        subview_lines = [line for line in mlir.splitlines() if "pto.subview" in line]
        assert len(subview_lines) == 1, f"expected one pto.subview, got:\n{mlir}"
        # The result tile_buf type is the right-hand side of the `->`.
        result_type = subview_lines[0].split("->", 1)[-1]
        assert "v_col=64" in result_type, (
            f"static valid col (64) must yield a static v_col=64 result dim (PTOAS "
            f"requires the valid operand and result type to agree per-dim), got:\n{subview_lines[0]}"
        )
        assert "v_col=?" not in result_type, (
            f"static valid col must NOT be promoted to dynamic v_col=? (this is the "
            f"lm_head_store PTOAS rejection), got:\n{subview_lines[0]}"
        )
        # The dynamic row dim stays dynamic.
        assert "v_row=?" in result_type, (
            f"dynamic valid row must yield a dynamic v_row=? result dim, got:\n{subview_lines[0]}"
        )

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
        # A dynamic column cannot be proved aligned, so #1789 canonicalization
        # materializes this slice. Its inferred static valid shape must survive
        # on the fresh allocation.
        assert "pto.subview" not in mlir, f"dynamic unaligned slice must be materialized, got:\n{mlir}"
        textract_lines = [line.strip() for line in mlir.splitlines() if "pto.textract" in line]
        assert len(textract_lines) == 1, f"expected one pto.textract, got:\n{mlir}"
        textract_out = textract_lines[0].split("outs(", 1)[1].split(":", 1)[0].strip()
        alloc_line = next(
            line for line in mlir.splitlines() if line.strip().startswith(f"{textract_out} = pto.alloc_tile")
        )
        assert "valid_row = %c16_index" in alloc_line and "valid_col = %c16_index" in alloc_line, (
            f"Source valid [32,32] >= slice size [16,16] must preserve static valid [16,16], got:\n"
            f"{alloc_line}"
        )

    def test_tile_slice_preserves_non_sentinel_parent_valid_shape(self):
        """Counterpart to the [0, 0] sentinel regression: when the parent's
        IR carries a genuine narrow valid_shape (not the lane1 sentinel),
        InferSubviewTileTypeComponents must still honour it so the subview's
        static v_row/v_col reflects the narrower extent.

        Parent valid [12, 12], slice sizes [8, 8] at offset [6, 6]:
          remain = 12 - 6 = 6;  result v = min(8, 6) = 6.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                dst: pl.Tensor[[8, 8], pl.FP32],
            ) -> pl.Tensor[[8, 8], pl.FP32]:
                narrow_parent: pl.Tile[
                    [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[12, 12])
                ] = pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                sliced: pl.Tile[[8, 8], pl.FP32] = pl.tile.slice(narrow_parent, [8, 8], [6, 6])
                return pl.store(sliced, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        # Offset [6,6] is not 32-byte aligned, so #1789 canonicalization
        # materializes the slice. The narrow valid extent must be preserved on
        # that fresh allocation.
        assert "pto.subview" not in mlir, f"unaligned slice must be materialized, got:\n{mlir}"
        textract_lines = [line.strip() for line in mlir.splitlines() if "pto.textract" in line]
        assert len(textract_lines) == 1, f"expected one pto.textract, got:\n{mlir}"
        textract_out = textract_lines[0].split("outs(", 1)[1].split(":", 1)[0].strip()
        alloc_line = next(
            line for line in mlir.splitlines() if line.strip().startswith(f"{textract_out} = pto.alloc_tile")
        )
        assert "valid_row = %c6_index" in alloc_line and "valid_col = %c6_index" in alloc_line, (
            "Parent narrow valid [12, 12] with offset [6, 6] and sizes [8, 8] must preserve "
            f"valid [6,6] (min(size, valid - offset)) on the materialized result; got:\n{alloc_line}"
        )

    def test_tile_slice_with_zero_valid_sentinel_parent_does_not_emit_zero_result_valid(self):
        """Regression for issue #1507: subview must not emit static v_row=0, v_col=0
        when the parent's IR carries the lane1 [0, 0] valid_shape sentinel.

        The parent's static type string always renders v_row=?, v_col=? (per
        ExtractTileTypeInfo in pto_type_utils.cpp), so PTOAS infers the
        subview's result valid from the slice's `sizes`. Our inference must
        align — reading the parent's tile_view_.valid_shape (which can be the
        sentinel [0, 0] after SplitVectorKernel's WithZeroValidShape) would
        produce v_row=0, v_col=0 that PTOAS rejects.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                dst: pl.Tensor[[16, 8], pl.FP32],
            ) -> pl.Tensor[[16, 8], pl.FP32]:
                sentinel_tile: pl.Tile[
                    [16, 32], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                ] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                sliced: pl.Tile[[16, 8], pl.FP32] = pl.tile.slice(sentinel_tile, [16, 8], [0, 0])
                return pl.store(sliced, [0, 0], dst)

        mlir = self._generate_mlir(Prog)
        subview_lines = [line for line in mlir.splitlines() if "pto.subview" in line]
        assert subview_lines, f"no pto.subview line emitted; got:\n{mlir}"
        result_type = subview_lines[0].split("->", 1)[-1]
        assert "v_row=0" not in result_type and "v_col=0" not in result_type, (
            "Sentinel [0, 0] parent valid_shape must NOT propagate to a static v_row=0/v_col=0 "
            f"on the subview result (ptoas would reject); got:\n{subview_lines[0]}"
        )

    def _generate_mlir_all_incore(self, program_cls) -> str:
        """Like ``_generate_mlir`` but concatenates PTOCodegen output for every
        InCore (AIC/AIV) leaf function, skipping the Group/orchestration wrapper that
        a mixed (cube+vector) kernel splits into (those are not PTOCodegen targets)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
        return "\n".join(
            codegen.PTOCodegen().generate(ir.Program([func], func.name, optimized.span))
            for func in optimized.functions.values()
            if ir.is_incore_type(func.func_type)
        )

    def test_tile_slice_mat_subview_emits_loc_mat(self):
        """A Mat tile.slice that survives the full pass pipeline (consumed by
        tile.move Mat→Vec, not extract/matmul) lowers to pto.subview with
        loc=mat on both source and result types."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[32, 64], pl.FP32],
                dst: pl.Tensor[[16, 64], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                src_mat = pl.tile.load(src, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
                sliced = pl.tile.slice(src_mat, [16, 64], [16, 0])
                vec_tile = pl.tile.move(sliced, target_memory=pl.Mem.Vec)
                return pl.store(vec_tile, [0, 0], dst)

        mlir = self._generate_mlir_all_incore(Prog)
        subview_lines = [line.strip() for line in mlir.splitlines() if "pto.subview" in line]
        assert subview_lines, f"Expected pto.subview for Mat tile.slice, got:\n{mlir}"
        sv = subview_lines[0]
        assert "loc=mat" in sv, f"pto.subview source must be loc=mat: {sv}"
        result_type = sv.split("->", 1)[-1] if "->" in sv else ""
        assert "loc=mat" in result_type, f"pto.subview result must be loc=mat: {sv}"
        assert "sizes [16, 64]" in sv, f"subview sizes must match slice shape: {sv}"


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

    def _generate_mlir_all_incore(self, program_cls) -> str:
        """Like ``_generate_mlir`` but concatenates PTOCodegen output for every
        InCore (AIC/AIV) leaf function, skipping the Group/orchestration wrapper that
        a mixed (cube+vector) kernel splits into (those are not PTOCodegen targets)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
        return "\n".join(
            codegen.PTOCodegen().generate(ir.Program([func], func.name, optimized.span))
            for func in optimized.functions.values()
            if ir.is_incore_type(func.func_type)
        )

    def test_tile_assemble_acc_to_mat_full_window_codegen(self):
        """A whole-tile Acc->Mat ``tile.assemble`` (a matmul result drained into an
        L1/Mat scratch — the representative Mat-scratch pattern) lowers to a
        converting ``pto.subview`` + ``pto.tmov``. The subview is typed from the
        **Mat result** (the dst — ``loc=mat``, ``fractal=512``); the ``pto.tmov``
        moves the Acc source (``loc=acc``, ``fractal=1024``) into it.

        Because the insert covers the whole result tile, no out-of-window
        preservation copy is emitted, so the output carries no unsupported Mat->Mat
        tmov and is accepted by PTOAS's verifier (``TMovOp::isAccToMat`` only
        requires the Mat dst fractal to be 512). The IR enforces
        ``source.dtype == result.dtype`` (DeduceTileAssembleType), so this is a
        layout/space conversion, not a dtype cast. Regression for the codegen the
        Mat-scratch path depends on (PTOCodegen previously aborted here)."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                a: pl.Tensor[[32, 16], pl.FP16],
                b: pl.Tensor[[16, 32], pl.FP16],
                y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                target = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
                tile_a = pl.load(a, [0, 0], [32, 16], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [16, 32], target_memory=pl.MemorySpace.Mat)
                src = pl.matmul(
                    pl.move(tile_a, target_memory=pl.MemorySpace.Left),
                    pl.move(tile_b, target_memory=pl.MemorySpace.Right),
                )  # Acc (L0C) [32, 32]
                result = pl.tile.assemble(target, src, [0, 0])  # full-window Acc -> Mat
                return pl.store(pl.move(result, target_memory=pl.MemorySpace.Vec), [0, 0], y)

        mlir = self._generate_mlir_all_incore(Prog)
        subview = next((line for line in mlir.splitlines() if "pto.subview" in line and "->" in line), None)
        assert subview is not None, f"expected an Acc->Mat assemble pto.subview, got:\n{mlir}"
        view_ty = subview.split("->", 1)[1]
        assert "loc=mat" in view_ty and "fractal=512" in view_ty, (
            f"Acc->Mat assemble subview must be typed from the Mat result (loc=mat, fractal=512): {view_ty}"
        )
        # The data write is a converting pto.tmov from the Acc source into the Mat view.
        view_ssa = subview.split(" = ", 1)[0].strip()
        tmov = next(
            (line for line in mlir.splitlines() if "pto.tmov" in line and f"outs({view_ssa}" in line),
            None,
        )
        assert tmov is not None, f"expected pto.tmov into the assemble subview {view_ssa!r}, got:\n{mlir}"
        tmov_src = tmov.split("outs", 1)[0]
        assert "loc=acc" in tmov_src and "fractal=1024" in tmov_src, (
            f"the assemble tmov source must be the Acc matmul result (loc=acc, fractal=1024): {tmov}"
        )
        # A full-window insert overwrites the whole tile, so the dead out-of-window
        # preservation copy is skipped — and therefore no unsupported Mat->Mat tmov.
        mat_to_mat = [
            line
            for line in mlir.splitlines()
            if "pto.tmov" in line
            and "loc=mat" in line.split("ins(", 1)[-1].split(")", 1)[0]
            and "loc=mat" in line.split("outs(", 1)[-1]
        ]
        assert not mat_to_mat, "full-window Acc->Mat must not emit a Mat->Mat pre-copy, got:\n" + "\n".join(
            mat_to_mat
        )

    def test_tile_assemble_acc_to_mat_partial_in_place(self):
        """A *partial* Acc->Mat ``tile.assemble`` whose Mat target is reused in-place
        as the result (memory reuse merges them, so ``target == dst``) lowers to a
        clean Acc->Mat ``pto.subview`` + ``pto.tmov`` with **no out-of-window
        preservation copy** — the codegen aliases the result to the target buffer, so
        the window write is in place and there is no unsupported Mat->Mat move. (A
        genuinely un-mergeable target still hits the fail-loud guard; the Mat-scratch
        autotiler's in-place chain keeps target == result.)"""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                a: pl.Tensor[[32, 16], pl.FP16],
                b: pl.Tensor[[16, 16], pl.FP16],
                y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
                tile_a = pl.load(a, [0, 0], [32, 16], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
                src = pl.matmul(
                    pl.move(tile_a, target_memory=pl.MemorySpace.Left),
                    pl.move(tile_b, target_memory=pl.MemorySpace.Right),
                )  # Acc (L0C) [32, 16]
                result = pl.tile.assemble(tile_x, src, [0, 16])  # partial Acc -> Mat, merged in place
                return pl.store(pl.move(result, target_memory=pl.MemorySpace.Vec), [0, 0], y)

        mlir = self._generate_mlir_all_incore(Prog)
        subview = next((line for line in mlir.splitlines() if "pto.subview" in line and "->" in line), None)
        assert subview is not None, f"expected an Acc->Mat assemble pto.subview, got:\n{mlir}"
        assert "loc=mat" in subview.split("->", 1)[1], f"the assemble subview must be a Mat view: {subview}"
        # In-place: no out-of-window preservation copy, hence no unsupported Mat->Mat tmov.
        mat_to_mat = [
            line
            for line in mlir.splitlines()
            if "pto.tmov" in line
            and "loc=mat" in line.split("ins(", 1)[-1].split(")", 1)[0]
            and "loc=mat" in line.split("outs(", 1)[-1]
        ]
        assert not mat_to_mat, (
            "in-place partial Acc->Mat must not emit a Mat->Mat pre-copy, got:\n" + "\n".join(mat_to_mat)
        )

    def test_chained_matmul_mat_scratch_codegen(self):
        """End-to-end: an oversized chained matmul whose bf16 result is consumed on-chip
        tiles into an L1/Mat scratch via the Acc->Mat **FIXPIPE writeback** — each
        per-sub-tile assemble lowers to ``pto.tinsert`` (the offset Acc->Mat path on
        A2/A3, which downcasts f32->bf16), filling a bf16 Mat scratch. Under the
        drain-count cost model (#1912) the 256x256x256 producer picks (256,128,64) OS
        split-K (wider m halves the drain count) → a 1x2 grid → 2 tinserts. (Assembles
        green through ptoas v0.45.)"""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[256, 256], pl.BF16],
                b: pl.Tensor[[256, 256], pl.BF16],
                e: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                c = pl.matmul(
                    a, b, out_dtype=pl.FP32
                )  # [256, 256] > L0c, consumed on-chip (K-split, both OS -> packs)
                cb = pl.cast(c, pl.BF16, mode="rint")  # rint -> bf16 Mat scratch (FIXPIPE tie-even)
                d = pl.matmul(cb, e, out_dtype=pl.FP32)
                out = pl.assemble(out, d, [0, 0])
                return out

        mlir = self._generate_mlir_all_incore(Prog)
        tinserts = [line for line in mlir.splitlines() if "pto.tinsert" in line]
        assert len(tinserts) == 2, f"1x2 grid -> 2 Acc->Mat tinserts, got {len(tinserts)}:\n{mlir}"
        assert "loc=mat, dtype=bf16" in mlir, (
            f"the chained-matmul intermediate must be a bf16 Mat scratch:\n{mlir}"
        )

    def test_chained_matmul_full_k_mat_scratch_codegen(self):
        """End-to-end full-K Mat-scratch: a K-fits-L0 oversized chained matmul (bf16
        intermediate) tiles into a Mat scratch via the *pipelined* emitter; the
        loop-variable-offset Acc->Mat assembles lower to ``pto.tinsert`` filling a bf16
        Mat scratch. (Assembles green through ptoas v0.45.)"""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[256, 64], pl.BF16],
                b: pl.Tensor[[64, 256], pl.BF16],
                e: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                c = pl.matmul(a, b, out_dtype=pl.FP32)  # K=64 fits L0 (k == K) -> full-K; on-chip
                cb = pl.cast(c, pl.BF16, mode="rint")  # rint -> bf16 Mat scratch (FIXPIPE tie-even)
                d = pl.matmul(cb, e, out_dtype=pl.FP32)
                out = pl.assemble(out, d, [0, 0])
                return out

        mlir = self._generate_mlir_all_incore(Prog)
        tinserts = [line for line in mlir.splitlines() if "pto.tinsert" in line]
        assert len(tinserts) == 2, (
            f"full-M, N-tiled (1x2 grid) -> 2 Acc->Mat tinserts, got {len(tinserts)}:\n{mlir}"
        )
        assert "loc=mat, dtype=bf16" in mlir, (
            f"the full-K chained-matmul intermediate must be a bf16 Mat scratch:\n{mlir}"
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
        set_validshape_line = next(
            strip_loc(line) for line in mlir.splitlines() if "pto.set_validshape" in line
        )
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

    def test_set_validshape_on_a_slice_is_rejected(self):
        """A view operand has no valid_row / valid_col operands to update.

        `tile.slice` lowers to a `pto.subview`, which carries its valid extent in
        its own type. ptoas rejects a runtime `pto.set_validshape` against it
        ("expects source tile_buf to have dynamic validShape (?, ?)"), so PyPTO
        must reject it first with a message that names the alternative instead of
        letting the failure surface in the assembler.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[32, 8], pl.FP32],
                dst: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            ) -> pl.Tensor[[8, 8], pl.FP32]:
                packed: pl.Tile[[32, 8], pl.FP32] = pl.load(
                    src, [0, 0], [32, 8], target_memory=pl.MemorySpace.Vec
                )
                blk: pl.Tile[[8, 8], pl.FP32] = packed[8:16, 0:8]
                narrowed: pl.Tile[[8, 8], pl.FP32] = pl.tile.set_validshape(blk, 8, 4)
                padded: pl.Tile[[8, 8], pl.FP32] = pl.tile.fillpad(narrowed, pad_value=pl.PadValue.zero)
                return pl.store(padded, [0, 0], dst)

        with pytest.raises(ValueError, match="cannot narrow a tile view"):
            self._generate_mlir(Prog)

    def test_set_validshape_rejects_a_dynamic_valid_slice_by_view_provenance(self):
        """A view is recognised by provenance, not by its rendered valid dims.

        `tile.slice` given a runtime `valid_shape` emits a `pto.subview` typed
        `v_row=?, v_col=?` — the same rendering an alloc-backed handle uses. It is
        still a view with no operands to update, so it must be rejected too;
        reading the type tokens instead would emit a `pto.set_validshape` that
        ptoas refuses with "requires a locally bound tile source".
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[32, 8], pl.FP32],
                slice_rows: pl.Scalar[pl.INDEX],
                slice_cols: pl.Scalar[pl.INDEX],
                dst: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            ) -> pl.Tensor[[8, 8], pl.FP32]:
                packed: pl.Tile[[32, 8], pl.FP32] = pl.load(
                    src, [0, 0], [32, 8], target_memory=pl.MemorySpace.Vec
                )
                blk: pl.Tile[[8, 8], pl.FP32] = pl.tile.slice(
                    packed, [8, 8], [8, 0], valid_shape=[slice_rows, slice_cols]
                )
                narrowed: pl.Tile[[8, 8], pl.FP32] = pl.tile.set_validshape(blk, 8, 4)
                padded: pl.Tile[[8, 8], pl.FP32] = pl.tile.fillpad(narrowed, pad_value=pl.PadValue.zero)
                return pl.store(padded, [0, 0], dst)

        with pytest.raises(ValueError, match="cannot narrow a tile view"):
            self._generate_mlir(Prog)


_ACC_M = 64
_ACC_K = 128
_ACC_N = 256

# tile.assemble cases keep the accumulator small enough for the 128 KiB L0C arena.
_ASM_M = 32
_ASM_K = 128
_ASM_N = 64


@pl.program
class AssembleNarrowedAccIntoFullWindow:
    """A compact matmul result assembled into a *non*-narrowed Acc buffer.

    The two windows address L0C at different fractal strides, so the copy is not
    expressible as a `pto.subview`. Before #2470 both sides read as non-compact
    and the mismatch was a silent miscompile.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[_ASM_M, _ASM_K], pl.INT8],
        rhs: pl.Tensor[[_ASM_K, _ASM_N], pl.INT8],
        valid_rows: pl.Scalar[pl.INDEX],
        output: pl.Out[pl.Tensor[[_ASM_M, 2 * _ASM_N], pl.INT32]],
    ) -> pl.Tensor[[_ASM_M, 2 * _ASM_N], pl.INT32]:
        lhs_tile: pl.Tile[[_ASM_M, _ASM_K], pl.INT8] = pl.load(
            lhs,
            [0, 0],
            [_ASM_M, _ASM_K],
            valid_shape=[valid_rows, _ASM_K],
            target_memory=pl.MemorySpace.Mat,
        )
        rhs_tile: pl.Tile[[_ASM_K, _ASM_N], pl.INT8] = pl.load(
            rhs, [0, 0], [_ASM_K, _ASM_N], target_memory=pl.MemorySpace.Mat
        )
        part: pl.Tile[[_ASM_M, _ASM_N], pl.INT32] = pl.tile.matmul(lhs_tile, rhs_tile)
        big: pl.Tile[[_ASM_M, 2 * _ASM_N], pl.INT32] = pl.tile.create(
            [_ASM_M, 2 * _ASM_N], pl.INT32, target_memory=pl.MemorySpace.Acc
        )
        joined: pl.Tile[[_ASM_M, 2 * _ASM_N], pl.INT32] = pl.tile.assemble(big, part, [0, 0])
        return pl.store(joined, [0, 0], output)


@pl.program
class AssembleFullAccIntoFullWindow:
    """The remedy the diagnostic names: compute the full tile, no narrowing."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[_ASM_M, _ASM_K], pl.INT8],
        rhs: pl.Tensor[[_ASM_K, _ASM_N], pl.INT8],
        output: pl.Out[pl.Tensor[[_ASM_M, 2 * _ASM_N], pl.INT32]],
    ) -> pl.Tensor[[_ASM_M, 2 * _ASM_N], pl.INT32]:
        lhs_tile: pl.Tile[[_ASM_M, _ASM_K], pl.INT8] = pl.load(
            lhs, [0, 0], [_ASM_M, _ASM_K], target_memory=pl.MemorySpace.Mat
        )
        rhs_tile: pl.Tile[[_ASM_K, _ASM_N], pl.INT8] = pl.load(
            rhs, [0, 0], [_ASM_K, _ASM_N], target_memory=pl.MemorySpace.Mat
        )
        part: pl.Tile[[_ASM_M, _ASM_N], pl.INT32] = pl.tile.matmul(lhs_tile, rhs_tile)
        big: pl.Tile[[_ASM_M, 2 * _ASM_N], pl.INT32] = pl.tile.create(
            [_ASM_M, 2 * _ASM_N], pl.INT32, target_memory=pl.MemorySpace.Acc
        )
        joined: pl.Tile[[_ASM_M, 2 * _ASM_N], pl.INT32] = pl.tile.assemble(big, part, [0, 0])
        return pl.store(joined, [0, 0], output)


@pl.program
class MatmulAccRuntimeNarrowedRows:
    """Issue #2470: the matmul's lhs carries a runtime valid row count."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[_ACC_M, _ACC_K], pl.INT8],
        rhs: pl.Tensor[[_ACC_K, _ACC_N], pl.INT8],
        valid_rows: pl.Scalar[pl.INDEX],
        output: pl.Out[pl.Tensor[[_ACC_M, _ACC_N], pl.INT32]],
    ) -> pl.Tensor[[_ACC_M, _ACC_N], pl.INT32]:
        lhs_tile: pl.Tile[[_ACC_M, _ACC_K], pl.INT8] = pl.load(
            lhs,
            [0, 0],
            [_ACC_M, _ACC_K],
            valid_shape=[valid_rows, _ACC_K],
            target_memory=pl.MemorySpace.Mat,
        )
        rhs_tile: pl.Tile[[_ACC_K, _ACC_N], pl.INT8] = pl.load(
            rhs, [0, 0], [_ACC_K, _ACC_N], target_memory=pl.MemorySpace.Mat
        )
        acc_tile: pl.Tile[[_ACC_M, _ACC_N], pl.INT32] = pl.tile.matmul(lhs_tile, rhs_tile)
        return pl.store(acc_tile, [0, 0], output)


@pl.program
class MatmulAccFullRows:
    """The same reduction with no narrowing anywhere -- the control case."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[_ACC_M, _ACC_K], pl.INT8],
        rhs: pl.Tensor[[_ACC_K, _ACC_N], pl.INT8],
        output: pl.Out[pl.Tensor[[_ACC_M, _ACC_N], pl.INT32]],
    ) -> pl.Tensor[[_ACC_M, _ACC_N], pl.INT32]:
        lhs_tile: pl.Tile[[_ACC_M, _ACC_K], pl.INT8] = pl.load(
            lhs, [0, 0], [_ACC_M, _ACC_K], target_memory=pl.MemorySpace.Mat
        )
        rhs_tile: pl.Tile[[_ACC_K, _ACC_N], pl.INT8] = pl.load(
            rhs, [0, 0], [_ACC_K, _ACC_N], target_memory=pl.MemorySpace.Mat
        )
        acc_tile: pl.Tile[[_ACC_M, _ACC_N], pl.INT32] = pl.tile.matmul(lhs_tile, rhs_tile)
        return pl.store(acc_tile, [0, 0], output)


class TestMatmulAccCompactCodegen:
    """Issue #2470: a runtime-narrowed accumulator must reach PTOAS as compact."""

    def _generate_mlir(self, program_cls) -> str:
        """Run PassManager and PTOCodegen on the given program, return MLIR string."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
        funcs = list(optimized.functions.values())
        assert funcs, "Program has no functions"
        single = ir.Program([funcs[0]], funcs[0].name, optimized.span)
        return codegen.PTOCodegen().generate(single)

    def test_runtime_narrowed_matmul_stores_from_a_compact_accumulator(self):
        """The store that reads L0C must use the stride ``mad`` wrote it at.

        ``mad`` takes M from the L0A operand's *valid* rows and lays the result
        out with an N-fractal stride of ceil(M/16)*16.  ``TSTORE`` recovers that
        stride only for a compact tile; otherwise it walks L0C at the physical
        ``Rows``, so every N-fractal above the first came back scrambled.
        """
        mlir = self._generate_mlir(MatmulAccRuntimeNarrowedRows)

        store_lines = [line for line in mlir.splitlines() if "pto.tstore" in line]
        assert store_lines, f"expected a pto.tstore for the accumulator, got:\n{mlir}"
        for line in store_lines:
            assert re.search(r"!pto\.tile_buf<loc=acc,[^>]*compact=1>", line), (
                "a runtime-narrowed accumulator must be stored as compact, or TSTORE reads "
                f"L0C at the physical Rows pitch instead of ceil(validRow/16)*16;\ngot:\n{line}"
            )

        # The accumulator's own allocation must agree with the store's view, or
        # one L0C buffer would be declared at two different strides.
        alloc_lines = [line for line in mlir.splitlines() if "pto.alloc_tile" in line and "loc=acc" in line]
        assert alloc_lines, f"expected an Acc pto.alloc_tile, got:\n{mlir}"
        for line in alloc_lines:
            assert re.search(r"!pto\.tile_buf<loc=acc,[^>]*compact=1>", line), line

    def test_full_width_matmul_accumulator_stays_noncompact(self):
        """Without narrowing, the emitted accumulator keeps its historical form."""
        mlir = self._generate_mlir(MatmulAccFullRows)

        assert not re.search(r"!pto\.tile_buf<loc=acc,[^>]*compact=1>", mlir), (
            f"a fully valid accumulator must not be stamped compact:\n{mlir}"
        )

    def test_assembling_a_narrowed_accumulator_into_a_full_window_is_rejected(self):
        """Two Acc windows at different L0C strides cannot share a ``pto.subview``.

        Before the accumulator carried compact this compiled and silently copied
        at the wrong pitch. Rejecting it is the fix; the message has to name the
        user's construct and a remedy rather than ``pto.subview``'s invariant.
        """
        with pytest.raises(ValueError, match="L0C fractal stride") as excinfo:
            self._generate_mlir(AssembleNarrowedAccIntoFullWindow)

        assert "Drop the narrowing from the matmul operand" in str(excinfo.value), str(excinfo.value)

    def test_assembling_a_full_accumulator_still_compiles(self):
        """The remedy the diagnostic names works, and the ordinary case is untouched."""
        mlir = self._generate_mlir(AssembleFullAccIntoFullWindow)

        assert "pto.subview" in mlir, mlir
        assert not re.search(r"!pto\.tile_buf<loc=acc,[^>]*compact=1>", mlir), mlir


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
        tsort32_line = next(line for line in mlir.splitlines() if "pto.tsort32" in line)
        assert "%sort32_src_view" in tsort32_line and "%sort32_idx_view" in tsort32_line, tsort32_line
        assert "%sort32_dst_view" in tsort32_line, tsort32_line
        assert "v_row=?" not in tsort32_line and "v_col=?" not in tsort32_line, tsort32_line
        assert "pto.tmrgsort" in mlir, f"Expected pto.tmrgsort in codegen output:\n{mlir}"
        # Constant block_len should appear as an i32 constant
        tmrgsort_lines = [line for line in mlir.splitlines() if "pto.tmrgsort" in line]
        assert tmrgsort_lines, "No pto.tmrgsort line found"
        assert "i32" in tmrgsort_lines[0], f"block_len type annotation should be i32: {tmrgsort_lines[0]}"
        tgather_line = next(line for line in mlir.splitlines() if "pto.tgather" in line)
        assert '"row"' in tgather_line and tgather_line.index('"row"') < tgather_line.index("outs("), (
            f"mask-form pto.tgather must emit the PTOAS row axis before outs(...), got:\n{tgather_line}"
        )

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

    @pytest.mark.parametrize(
        ("src_dtype", "output_cols"),
        [(pl.FP32, 128), (pl.FP16, 256)],
    )
    def test_sort32_dynamic_valid_width_emits_level3_scratch(self, src_dtype, output_cols):
        """A runtime valid width keeps logical output bounds and uses static TSORT capacity."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[1, 64], src_dtype],
                idx: pl.Tensor[[1, 64], pl.UINT32],
                out: pl.Tensor[[1, output_cols], src_dtype],
                valid_cols: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[1, output_cols], src_dtype]:
                src_tile = pl.load(src, [0, 0], [1, 64], valid_shape=[1, valid_cols])
                idx_tile = pl.load(idx, [0, 0], [1, 64], valid_shape=[1, valid_cols])
                sorted_tile = pl.tile.sort32(src_tile, idx_tile)
                return pl.store(sorted_tile, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        line = next(line for line in mlir.splitlines() if "pto.tsort32" in line)
        ins = line.split("ins(", 1)[1].split(":", 1)[0]
        assert ins.count(",") == 2, line
        assert "%sort32_tmp_view" in line and "%sort32_dst_view" in line, line
        outs = line.split("outs(", 1)[1]
        assert f"cols={output_cols}" in outs and f"v_col={output_cols}" in outs, line
        assert "v_col=?" not in outs, line
        tstore_line = next(line for line in mlir.splitlines() if "pto.tstore" in line)
        assert "%sorted_tile" in tstore_line and "%sort32_dst_view" not in tstore_line, tstore_line
        assert "arith.muli" in mlir, mlir


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


class TestLevel3StaticViewCodegen:
    """Codegen static-view bridges for explicit level-3 tmp forms on Ascend910B."""

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

    def test_tci_emits_static_view_for_explicit_tmp(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                out: pl.Tensor[[1, 32], pl.INT32],
            ) -> pl.Tensor[[1, 32], pl.INT32]:
                tmp: pl.Tile[[1, 192], pl.FP32, pl.Mem.Vec] = pl.tile.create(
                    [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                )
                seq: pl.Tile[[1, 32], pl.INT32, pl.Mem.Vec] = pl.tile.ci(0, [1, 32], dtype=pl.INT32, tmp=tmp)
                return pl.store(seq, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        tci_line = next(line for line in mlir.splitlines() if "pto.tci" in line)
        assert "%ci_tmp_view" in tci_line and "%ci_dst_view" in tci_line, tci_line
        assert "v_row=?" not in tci_line and "v_col=?" not in tci_line, tci_line

    def test_tcvt_emits_static_view_for_explicit_tmp(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.INT16],
            ) -> pl.Tensor[[16, 16], pl.INT16]:
                tile_in: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.load(
                    src, [0, 0], [16, 16], target_memory=pl.Mem.Vec
                )
                result: pl.Tile[[16, 16], pl.INT16, pl.Mem.Vec] = pl.tile.cast(
                    tile_in, target_type=pl.INT16, mode="round"
                )
                return pl.store(result, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        tcvt_line = next(line for line in mlir.splitlines() if "pto.tcvt" in line)
        assert "%tcvt_tmp_view" in tcvt_line and "%tcvt_dst_view" in tcvt_line, tcvt_line
        assert "v_row=?" not in tcvt_line and "v_col=?" not in tcvt_line, tcvt_line

    def test_tcolsum_binary_emits_static_view_for_tmp(self):
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
        tcolsum_line = next(line for line in mlir.splitlines() if "pto.tcolsum" in line)
        assert "%colsum_tmp_view" in tcolsum_line and "%colsum_dst_view" in tcolsum_line, tcolsum_line
        assert "isBinary = true" in tcolsum_line, tcolsum_line


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


class TestTileMoveAccAddressInvariant:
    """Regressions for #1310/#1352: lowering must not leave Acc→Acc moves.

    When AutoTileMatmulL0 rewrites matmul_acc into an inner K-loop, the fresh
    IterArg Vars carry different MemRef bases than the outer loop's accumulator.
    The semantic-alias and MemoryReuse pipeline must keep the accumulator chain
    on one buffer without leaving a redundant tile.move. Baked-address codegen
    rejects any same-address move instead of masking an upstream mismatch.
    """

    def _generate_mlir(self, program_cls) -> str:
        """Generate MLIR for all InCore functions, joined by blank lines.

        Programs that lower to split AIC/AIV kernels (e.g. ``split=UP_DOWN``)
        produce multiple InCore functions; the bad acc→acc ``pto.tmov`` may
        appear on the AIC side only.  Codegen each InCore function and
        concatenate their MLIR so regression assertions cover every variant.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program_cls)
        codegen_instance = codegen.PTOCodegen()
        funcs = list(optimized.functions.values())
        assert funcs, "Program has no functions"
        incore = [f for f in funcs if ir.is_incore_type(f.func_type)]
        targets = incore if incore else [funcs[0]]
        return "\n\n".join(
            codegen_instance.generate(ir.Program([f], f.name, optimized.span)) for f in targets
        )

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
        semantic-alias handling must align the inner yield with the outer
        iter-arg so no redundant same-address tile.move reaches codegen.
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

    def test_pipeline_matmul_acc_no_acc_to_acc_tmov(self):
        """Dual-accumulator pl.pipeline(stage=2) matmul_acc must not produce
        pto.tmov acc→acc (#1352).

        Reproduces the Qwen3-32B gate_up_silu shape that triggered the bug: two
        independent accumulators (gate, up) built with prolog-then-pipeline
        matmul_acc under ``pl.at(CORE_GROUP, optimizations=[pl.split(UP_DOWN)])``.

        Why this shape triggers it:
        - Mat-resident inputs + K_CHUNK=128/N=256 make AutoTileMatmulL0 insert an
          inner K-loop (K_L0=64) with a fresh ``_l0_c`` accumulator IterArg.
        - pl.pipeline(stage=2) replicates the loop body, nesting that IterArg.
        - split=UP_DOWN forces the AIC/AIV split, so MemoryReuse runs on the AIC
          function where gate is consumed before up, so up *reuses* gate's freed
          Acc buffer.

        The bug (before the Path 2 fix): MemoryReuse's greedy pass retyped up's
        producer/init AssignStmt vars onto gate's buffer but left up's
        loop-carried iter_arg/return_var on its own buffer, splitting the chain.
        YieldFixupMutator (bottom-up) then bridged the two Acc buffers with
        ``tile.move`` ops that lowered to acc→acc ``pto.tmov`` — which ptoas
        rejects on Ascend 910B.

        The fix (AlignLoopCarriesToInitMutator) re-aligns loop-carried MemRefs to
        their reused init top-down, so the whole up chain lands on the single
        reused Acc buffer and no acc→acc move is ever inserted.  This test
        asserts the generated MLIR (across all split InCore functions) carries no
        acc→acc ``pto.tmov``; it fails without the fix.
        """
        BATCH, K_CHUNK, N, NUM_CHUNKS = 16, 128, 256, 4

        @pl.program
        class PipelineGateUpProg:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[BATCH, K_CHUNK * NUM_CHUNKS], pl.BF16],
                wg: pl.Tensor[[K_CHUNK * NUM_CHUNKS, N], pl.BF16],
                wu: pl.Tensor[[K_CHUNK * NUM_CHUNKS, N], pl.BF16],
                out: pl.Out[pl.Tensor[[BATCH, N], pl.BF16]],
            ) -> pl.Tensor[[BATCH, N], pl.BF16]:
                out_tile = pl.create_tensor([BATCH, N], dtype=pl.BF16)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
                    name_hint="gate_up_pipeline_acc",
                ):
                    x0 = pl.slice(x, [BATCH, K_CHUNK], [0, 0])
                    x1 = pl.slice(x, [BATCH, K_CHUNK], [0, K_CHUNK])
                    wg0 = pl.slice(wg, [K_CHUNK, N], [0, 0])
                    wg1 = pl.slice(wg, [K_CHUNK, N], [K_CHUNK, 0])
                    wu0 = pl.slice(wu, [K_CHUNK, N], [0, 0])
                    wu1 = pl.slice(wu, [K_CHUNK, N], [K_CHUNK, 0])
                    gate_acc = pl.matmul(x0, wg0, out_dtype=pl.FP32)
                    gate_acc = pl.matmul_acc(gate_acc, x1, wg1)
                    for kb in pl.pipeline(2, NUM_CHUNKS, stage=2):
                        k0 = kb * K_CHUNK
                        gate_acc = pl.matmul_acc(
                            gate_acc,
                            pl.slice(x, [BATCH, K_CHUNK], [0, k0]),
                            pl.slice(wg, [K_CHUNK, N], [k0, 0]),
                        )
                    # Consume gate (to Vec) before the up pipeline so up reuses
                    # gate's freed Acc buffer — the condition that exposed #1352.
                    gate_fp32 = pl.add(gate_acc, 0.0)
                    up_acc = pl.matmul(x0, wu0, out_dtype=pl.FP32)
                    up_acc = pl.matmul_acc(up_acc, x1, wu1)
                    for kb in pl.pipeline(2, NUM_CHUNKS, stage=2):
                        k0 = kb * K_CHUNK
                        up_acc = pl.matmul_acc(
                            up_acc,
                            pl.slice(x, [BATCH, K_CHUNK], [0, k0]),
                            pl.slice(wu, [K_CHUNK, N], [k0, 0]),
                        )
                    combined = pl.mul(gate_fp32, up_acc)
                    out_tile = pl.assemble(out_tile, pl.cast(combined, pl.BF16), [0, 0])
                out = pl.assemble(out, out_tile, [0, 0])
                return out

        # Guard: assert split=UP_DOWN actually produced split InCore kernels.
        # The acc→acc regression only exists on the AIC side after the AIC/AIV
        # split, so without this guard a split-lowering regression would make
        # the test pass vacuously and stop covering #1352.
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(PipelineGateUpProg)
        incore = [f for f in optimized.functions.values() if ir.is_incore_type(f.func_type)]
        assert len(incore) >= 2, (
            f"Expected split=UP_DOWN to produce split InCore variants for #1352, "
            f"got {len(incore)} InCore function(s)"
        )

        mlir = self._generate_mlir(PipelineGateUpProg)
        assert not self._has_acc_to_acc_tmov(mlir), (
            f"Generated MLIR contains invalid pto.tmov acc→acc (regression #1352):\n{mlir}"
        )


class TestTileMoveAddressValidation:
    """Baked-address tile.move rejects in-place source/destination addresses."""

    @staticmethod
    def _vec_tile_move_program(
        *,
        dst_view: ir.TileView | None,
        name: str,
        dst_byte_offset: int = 0,
        dst_space: ir.MemorySpace = ir.MemorySpace.Vec,
        shared_base: bool = False,
    ) -> ir.Program:
        span = ir.Span.unknown()
        size = 64
        nbytes = size * size * 2  # BF16
        src_base = ir.Var(f"{name}_src_base", ir.PtrType(), span)
        dst_base = src_base if shared_base else ir.Var(f"{name}_dst_base", ir.PtrType(), span)
        src_memref = ir.MemRef(src_base, ir.ConstInt(0, DataType.INT64, span), nbytes, span)
        dst_memref = ir.MemRef(
            dst_base,
            ir.ConstInt(dst_byte_offset, DataType.INT64, span),
            nbytes,
            span,
        )

        inp = ir.Var("inp", ir.TensorType([size, size], DataType.BF16), span)
        out = ir.Var("out", ir.TensorType([size, size], DataType.BF16), span)

        src_ty = ir.TileType([size, size], DataType.BF16, src_memref, None, ir.MemorySpace.Vec)
        dst_ty = ir.TileType([size, size], DataType.BF16, dst_memref, dst_view, dst_space)
        src = ir.Var("src_nd", src_ty, span)
        dst = ir.Var("dst_layout", dst_ty, span)
        result = ir.Var("result", ir.TensorType([size, size], DataType.BF16), span)

        zero = ir.ConstInt(0, DataType.INDEX, span)
        dim = ir.ConstInt(size, DataType.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shapes = ir.MakeTuple([dim, dim], span)

        load = ir.Call(ir.Op("tile.load"), [inp, offsets, shapes], {}, src_ty, span)
        move = ir.Call(
            ir.Op("tile.move"),
            [src],
            {"target_memory": dst_space},
            dst_ty,
            span,
        )
        store = ir.Call(ir.Op("tile.store"), [dst, offsets, out], result.type, span)

        body = ir.SeqStmts(
            [
                ir.SeqStmts(
                    [
                        ir.AssignStmt(src, load, span),
                        ir.AssignStmt(dst, move, span),
                        ir.AssignStmt(result, store, span),
                    ],
                    span,
                ),
                ir.ReturnStmt([result], span),
            ],
            span,
        )
        func = ir.Function(
            name,
            [(inp, ir.ParamDirection.In), (out, ir.ParamDirection.Out)],
            [ir.TensorType([size, size], DataType.BF16)],
            body,
            span,
            ir.FunctionType.InCore,
        )
        return ir.Program([func], f"{name}_program", span)

    @staticmethod
    def _generate_mlir(program: ir.Program) -> str:
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        return codegen.PTOCodegen().generate(program)

    @pytest.mark.parametrize(
        ("dst_view", "name"),
        [
            pytest.param(None, "move_same_layout_same_addr", id="same-layout"),
            pytest.param(
                ir.TileView(
                    blayout=ir.TileLayout.col_major,
                    slayout=ir.TileLayout.row_major,
                    fractal=1024,
                ),
                "move_nd_to_nz_same_addr",
                id="different-layout",
            ),
            pytest.param(
                ir.TileView(compact=ir.CompactMode.normal),
                "move_noncompact_to_compact_same_addr",
                id="different-compact",
            ),
            pytest.param(
                ir.TileView(pad=ir.PadValue.max),
                "move_unpadded_to_padded_same_addr",
                id="different-pad",
            ),
        ],
    )
    def test_same_address_rejected(self, dst_view: ir.TileView | None, name: str):
        """TMOV cannot use equal source and destination addresses."""
        program = self._vec_tile_move_program(dst_view=dst_view, name=name)
        with pytest.raises(ValueError, match="tile.move requires distinct source and destination addresses"):
            self._generate_mlir(program)

    def test_same_base_distinct_addresses_emit_tmov(self):
        """Different offsets in one allocation remain a functional move."""
        program = self._vec_tile_move_program(
            dst_view=None,
            name="move_same_base_distinct_addresses",
            dst_byte_offset=64 * 64 * 2,
            shared_base=True,
        )
        mlir = self._generate_mlir(program)
        tmovs = [line for line in mlir.splitlines() if "pto.tmov" in line]
        assert len(tmovs) == 1
        assert "loc=vec" in tmovs[0]

    def test_equal_numeric_addresses_in_different_spaces_emit_tmov(self):
        """Equal offsets in independent memory spaces are not an in-place move."""
        program = self._vec_tile_move_program(
            dst_view=None,
            name="move_vec_to_mat_at_zero",
            dst_space=ir.MemorySpace.Mat,
        )
        mlir = self._generate_mlir(program)
        tmovs = [line for line in mlir.splitlines() if "pto.tmov" in line]
        assert len(tmovs) == 1
        assert "loc=vec" in tmovs[0]
        assert "loc=mat" in tmovs[0]

    @pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
    def test_declared_same_address_rejected_after_planning(self, planner, ascend_backend):
        """A pinned allocation cannot bypass the baked-address invariant."""

        @pl.program
        class SameAddressMove:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                inp: pl.Tensor[[32, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[32, 128], pl.FP32]],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                src: pl.Tile[[32, 128], pl.FP32, pl.MemRef("shared"), pl.Mem.Vec] = pl.tile.load(
                    inp, [0, 0], [32, 128]
                )
                dst: pl.Tile[[32, 128], pl.FP32, pl.MemRef("shared"), pl.Mem.Vec] = pl.tile.move(
                    src, target_memory=pl.Mem.Vec
                )
                return pl.tile.store(dst, [0, 0], out)

        with passes.PassContext([], memory_planner=planner):
            optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(SameAddressMove)
        function = next(func for func in optimized.functions.values() if ir.is_incore_type(func.func_type))
        program = ir.Program([function], function.name, optimized.span)

        with pytest.raises(ValueError, match="tile.move requires distinct source and destination addresses"):
            self._generate_mlir(program)


class TestTileStoreAtomicCodegen:
    """Tests for tile.store atomic-add codegen (pto.tstore atomicType attr)."""

    def _generate_mlir(self, program_cls) -> str:
        """Run PassManager and PTOCodegen on the given program, return MLIR string."""
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

    # -- Vector (AIV) atomic-add store: one hardware atomic-add dtype per test. --
    # Every hardware atomic-add dtype (set_atomic_{f32,f16,bf16,s32,s16,s8}) is a
    # plain loaded Vec tile stored to a GM tensor of the same dtype -> a `loc=vec`
    # atomic store on the AIV UB->GM (MTE3) pipe. These vector-path dtypes are not
    # constrained by the Acc->GM whitelist (that only bounds the cube path).
    def _assert_vec_atomic_store(self, dtype, mlir_dt, cols=16):
        # `cols` widens the tile so the row byte size (cols * sizeof(dtype)) meets
        # ptoas' 32-byte row alignment — int8 needs 32 cols (16 would be 16 bytes).
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[16, cols], dtype], out: pl.Tensor[[16, cols], dtype]):
                t = pl.load(x, [0, 0], [16, cols])
                pl.store(t, [0, 0], out, atomic=pl.AtomicType.Add)

        mlir = self._generate_mlir(Prog)
        tstore_lines = [line.strip() for line in mlir.splitlines() if "pto.tstore" in line]
        assert tstore_lines, f"no pto.tstore line emitted:\n{mlir}"
        assert all("{atomicType = #pto<atomic_type atomic_add>}" in line for line in tstore_lines), (
            f"expected atomic_add on every {mlir_dt} pto.tstore, got:\n{tstore_lines}"
        )
        assert all("loc=vec" in line and f"dtype={mlir_dt}" in line for line in tstore_lines), (
            f"expected a {mlir_dt} vector (loc=vec) store, got:\n{tstore_lines}"
        )

    def test_atomic_add_store_fp32_emits_atomic_type(self):
        """fp32 vector (AIV) atomic-add store (set_atomic_f32)."""
        self._assert_vec_atomic_store(pl.FP32, "f32")

    def test_atomic_add_store_fp16_emits_atomic_type(self):
        """fp16 vector (AIV) atomic-add store (set_atomic_f16)."""
        self._assert_vec_atomic_store(pl.FP16, "f16")

    def test_atomic_add_store_bf16_emits_atomic_type(self):
        """bf16 vector (AIV) atomic-add store (set_atomic_bf16; A2/A3 only)."""
        self._assert_vec_atomic_store(pl.BF16, "bf16")

    def test_atomic_add_store_int32_emits_atomic_type(self):
        """int32 vector (AIV) atomic-add store (set_atomic_s32)."""
        self._assert_vec_atomic_store(pl.INT32, "i32")

    def test_atomic_add_store_int16_emits_atomic_type(self):
        """int16 vector (AIV) atomic-add store (set_atomic_s16)."""
        self._assert_vec_atomic_store(pl.INT16, "i16")

    def test_atomic_add_store_int8_emits_atomic_type(self):
        """int8 vector (AIV) atomic-add store (set_atomic_s8); 32 cols for row alignment."""
        self._assert_vec_atomic_store(pl.INT8, "i8", cols=32)

    def test_plain_store_omits_atomic_type(self):
        """A plain pl.store emits no atomicType attribute (byte-identical codegen)."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Tensor[[16, 16], pl.FP32]):
                t = pl.load(x, [0, 0], [16, 16])
                pl.store(t, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        tstore_lines = [line.strip() for line in mlir.splitlines() if "pto.tstore" in line]
        assert tstore_lines, f"no pto.tstore line emitted:\n{mlir}"
        assert all("atomicType" not in line for line in tstore_lines), (
            f"plain store must not emit atomicType, got:\n{tstore_lines}"
        )

    def test_atomic_add_bf16_rejected_on_ascend950(self):
        """bf16 atomic-add is A2/A3-only; on Ascend950 (A5) the program is rejected.

        The IR-level dtype gate is backend-agnostic (a program may target A2/A3),
        so bf16 atomic passes op validation. The backend-aware guard is the
        ``AtomicAddDtypeValid`` property verifier, which ``PassPipeline`` runs at
        pipeline input — the error carries the user's own span instead of
        deferring to a downstream pto-isa ``static_assert``. Per-verifier
        coverage of all four atomic sites lives in
        ``tests/ut/ir/verifier/test_atomic_add_dtype.py``.
        """
        # Capture the prior backend so the A5 override does not leak; it may be
        # unset (get_backend_type raises when no backend is configured yet).
        try:
            prev_backend = backend.get_backend_type()
        except Exception:
            prev_backend = None
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend950)
        try:

            @pl.program
            class Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(self, x: pl.Tensor[[16, 16], pl.BF16], out: pl.Tensor[[16, 16], pl.BF16]):
                    t = pl.load(x, [0, 0], [16, 16])
                    pl.store(t, [0, 0], out, atomic=pl.AtomicType.Add)

            with pytest.raises(pypto.Error, match="AtomicAddDtypeValid"):
                PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
        finally:
            backend.reset_for_testing()
            if prev_backend is not None:
                backend.set_backend_type(prev_backend)


class TestTensorAssembleAtomicCodegen:
    """Tests for tensor.assemble atomic-add — lowers to tile.store with atomicType."""

    def _generate_mlir(self, program_cls) -> str:
        """Run PassManager and PTOCodegen on the given program, return MLIR string."""
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

    def test_atomic_add_assemble_emits_atomic_type(self):
        """pl.assemble(..., atomic=AtomicType.Add) into a GM output lowers to an atomic-add store."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Tensor[[16, 16], pl.FP32]):
                y = pl.add(x, x)
                out = pl.assemble(out, y, [0, 0], atomic=pl.AtomicType.Add)
                return out

        mlir = self._generate_mlir(Prog)
        tstore_lines = [line.strip() for line in mlir.splitlines() if "pto.tstore" in line]
        assert tstore_lines, f"no pto.tstore line emitted:\n{mlir}"
        assert all("{atomicType = #pto<atomic_type atomic_add>}" in line for line in tstore_lines), (
            f"expected atomic_add attribute on the lowered pto.tstore, got:\n{tstore_lines}"
        )

    def test_plain_assemble_omits_atomic_type(self):
        """A plain pl.assemble lowers to a plain store — no atomicType attribute."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Tensor[[16, 16], pl.FP32]):
                y = pl.add(x, x)
                out = pl.assemble(out, y, [0, 0])
                return out

        mlir = self._generate_mlir(Prog)
        tstore_lines = [line.strip() for line in mlir.splitlines() if "pto.tstore" in line]
        assert tstore_lines, f"no pto.tstore line emitted:\n{mlir}"
        assert all("atomicType" not in line for line in tstore_lines), (
            f"plain assemble must not emit atomicType, got:\n{tstore_lines}"
        )

    def test_atomic_add_bf16_target_emits_atomic_type(self):
        """atomic=Add into a bf16 GM target is a hardware atomic-add dtype on A2/A3.

        The bf16 assemble lowers to a bf16 atomic-add tile.store (set_atomic_bf16
        on pto-isa), so the emitted pto.tstore must carry the atomic_add attribute.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[16, 16], pl.BF16], out: pl.Tensor[[16, 16], pl.BF16]):
                y = pl.add(x, x)
                out = pl.assemble(out, y, [0, 0], atomic=pl.AtomicType.Add)
                return out

        mlir = self._generate_mlir(Prog)
        tstore_lines = [line.strip() for line in mlir.splitlines() if "pto.tstore" in line]
        assert tstore_lines, f"no pto.tstore line emitted:\n{mlir}"
        assert all("{atomicType = #pto<atomic_type atomic_add>}" in line for line in tstore_lines), (
            f"expected atomic_add attribute on the lowered bf16 pto.tstore, got:\n{tstore_lines}"
        )

    def test_atomic_add_tile_target_rejected(self):
        """atomic=Add into an on-chip tile is rejected — no global-memory destination."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, src_in: pl.Tensor[[16, 64], pl.FP32], out: pl.Tensor[[16, 128], pl.FP32]):
                target = pl.create_tensor([16, 128], dtype=pl.FP32)
                source = pl.add(src_in, src_in)
                target = pl.assemble(target, source, [0, 64], atomic=pl.AtomicType.Add)
                out = pl.assemble(out, target, [0, 0])
                return out

        with pytest.raises(ValueError, match="global-memory destination"):
            self._generate_mlir(Prog)


class TestScatterCodegen:
    """Tests for tile.scatter / tile.scatter_mask PTO code generation (DPS).

    Both forms map to pto.tscatter and are destination-passing-style: the
    destination tile is the first operand and the result aliases it via
    set_output_reuses_input(0). Codegen must therefore emit the dst (the result
    target) inside outs(...) and the remaining operands inside ins(...).
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

    def test_tile_scatter_index_form_codegen(self):
        """tile.scatter emits pto.tscatter ins(src, idx) outs(dst) with dst aliased.

        ``indexes`` is the per-element *flattened* destination index tile, so it
        has the same [rows, cols] shape as ``src`` (pto.tscatter writes
        dst.flat[idx[i, j]] = src[i, j]).
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                dst_in: pl.Tensor[[16, 32], pl.FP32],
                src: pl.Tensor[[4, 32], pl.FP32],
                idx: pl.Tensor[[4, 32], pl.INT32],
                out: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                dst_tile: pl.Tile[[16, 32], pl.FP32] = pl.load(dst_in, [0, 0], [16, 32])
                src_tile: pl.Tile[[4, 32], pl.FP32] = pl.load(src, [0, 0], [4, 32])
                idx_tile: pl.Tile[[4, 32], pl.INT32] = pl.load(idx, [0, 0], [4, 32])
                scattered: pl.Tile[[16, 32], pl.FP32] = pl.tile.scatter(dst_tile, src_tile, idx_tile)
                return pl.store(scattered, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        tscatter_lines = [line for line in mlir.splitlines() if "pto.tscatter" in line]
        assert len(tscatter_lines) == 1, f"Expected exactly one pto.tscatter, got:\n{mlir}"
        line = tscatter_lines[0]
        assert "ins(" in line and "outs(" in line, (
            f"pto.tscatter must use the ins(...) outs(...) DPS form, got:\n{line}"
        )

    def test_tile_scatter_mask_form_codegen(self):
        """tile.scatter_mask emits row-axis pto.tscatter with a maskPattern attribute (DPS)."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                dst_in: pl.Tensor[[4, 16], pl.FP32],
                src: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Tensor[[4, 16], pl.FP32],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                dst_tile: pl.Tile[[4, 16], pl.FP32] = pl.load(dst_in, [0, 0], [4, 16])
                src_tile: pl.Tile[[4, 8], pl.FP32] = pl.load(src, [0, 0], [4, 8])
                scattered: pl.Tile[[4, 16], pl.FP32] = pl.tile.scatter_mask(
                    dst_tile, src_tile, mask_pattern=pl.tile.MaskPattern.P0101
                )
                return pl.store(scattered, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        tscatter_lines = [line for line in mlir.splitlines() if "pto.tscatter" in line]
        assert len(tscatter_lines) == 1, f"Expected exactly one pto.tscatter, got:\n{mlir}"
        line = tscatter_lines[0]
        assert "maskPattern" in line and "P0101" in line, (
            f"mask form must emit a #pto.mask_pattern<P0101> attribute, got:\n{line}"
        )
        assert "ins(" in line and "outs(" in line, (
            f"pto.tscatter mask form must use the ins(...) outs(...) DPS form, got:\n{line}"
        )
        # maskPattern must ride inside ins() after src (like pto.tgather), not as
        # a trailing attr after outs() — PTOAS rejects a bare ins(%src ...) with
        # "expected ',' after src operand".
        assert line.index("maskPattern") < line.index("outs("), (
            f"maskPattern must appear inside ins(...) before outs(...), got:\n{line}"
        )
        assert '"row"' in line and line.index('"row"') < line.index("outs("), (
            f"mask-form pto.tscatter must emit the PTOAS row axis before outs(...), got:\n{line}"
        )


class TestCrossCoreSyncCodegen:
    """Tests for explicit pto.sync.set/wait emission."""

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

    def test_static_event_id_and_ffts_mode(self):
        """Static DSL operands map to PTO attributes and literals."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
                pl.system.sync_set(event_id=3, pipe=pl.PipeType.MTE3, ffts_mode=1)
                pl.system.sync_wait(event_id=3, pipe=pl.PipeType.MTE2)
                return x

        mlir = self._generate_mlir(Prog)
        assert "pto.sync.set <PIPE_MTE3>, 3 {ffts_mode = 1 : i32}" in mlir
        assert "pto.sync.wait <PIPE_MTE2>, 3" in mlir

    def test_a3_ffts_workspace_setup(self):
        """An FFTS tensor stays a raw pointer and lowers to the required PTO setup op."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                ffts_workspace: pl.Tensor[[256], pl.INT64],
                x: pl.Tensor[[16], pl.FP32],
            ) -> pl.Tensor[[16], pl.FP32]:
                pl.system.set_ffts(ffts_workspace)
                pl.system.sync_wait(event_id=3, pipe=pl.PipeType.MTE2)
                return x

        mlir = self._generate_mlir(Prog)
        assert "%arg0: !pto.ptr<i64>" in mlir
        assert "pto.set_ffts %arg0 : !pto.ptr<i64>" in mlir
        assert "ffts_workspace_view" not in mlir

    def test_dynamic_event_id(self):
        """An index scalar lowers to the PTO dynamic event-id operand."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self, x: pl.Tensor[[16], pl.FP32], event_id: pl.Scalar[pl.INDEX]
            ) -> pl.Tensor[[16], pl.FP32]:
                pl.system.sync_set(event_id, pipe=pl.PipeType.MTE3)
                pl.system.sync_wait(event_id, pipe=pl.PipeType.MTE2)
                return x

        mlir = self._generate_mlir(Prog)
        assert "pto.sync.set <PIPE_MTE3>, %" in mlir
        assert "pto.sync.wait <PIPE_MTE2>, %" in mlir


class TestSyncAllCodegen:
    """Tests that pl.system.syncall lowers to pto.syncall (hard/FFTS form)."""

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

    def test_syncall_emits_hard_barrier_with_core_type(self):
        """pl.system.syncall(core_type=...) emits pto.syncall() mode=<hard>."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_syncall(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                pl.system.syncall(core_type=pl.KernelType.AIV)
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(tile, [0, 0], out)
                return updated

        mlir = self._generate_mlir(Prog)
        assert "pto.syncall()" in mlir, f"pto.syncall not found in MLIR:\n{mlir}"
        line = next((ln for ln in mlir.splitlines() if "pto.syncall" in ln), "")
        assert "mode = #pto.sync_all_mode<hard>" in line, f"hard mode missing:\n{line}"
        assert "core_type = #pto.sync_core_type<aiv_only>" in line, f"core_type missing:\n{line}"

    def test_syncall_soft_emits_gm_polling_barrier(self):
        """soft syncall emits pto.syncall(%gm_pview, %used : ...) mode=<soft>."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_syncall_soft(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
                ws: pl.Tensor[[16], pl.INT32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                pl.system.syncall(
                    mode=pl.SyncAllMode.SOFT,
                    core_type=pl.KernelType.AIV,
                    gm_workspace=ws,
                    used_cores=4,
                )
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(tile, [0, 0], out)
                return updated

        mlir = self._generate_mlir(Prog)
        line = next((ln for ln in mlir.splitlines() if "pto.syncall(" in ln), "")
        assert line, f"soft pto.syncall not found in MLIR:\n{mlir}"
        assert "mode = #pto.sync_all_mode<soft>" in line, f"soft mode missing:\n{line}"
        assert "core_type = #pto.sync_core_type<aiv_only>" in line, f"core_type missing:\n{line}"
        # PTOAS v0.60 takes the raw GM pointer + optional used_cores.
        assert "!pto.ptr<i32>" in line, f"gm pointer missing:\n{line}"
        assert "tile_buf" not in line, f"legacy scratch operand still emitted:\n{line}"
        assert line.split(" : ", 1)[0].count(",") == 1, f"unexpected soft operand count:\n{line}"
        assert not any("partition_view" in ln and "syncgm" in ln for ln in mlir.splitlines()), mlir

    def test_syncall_soft_omits_launch_derived_participant_count(self):
        """used_cores=0 emits the canonical single-operand soft form."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_syncall_soft(self, ws: pl.Tensor[[4, 4], pl.INT32]) -> pl.Tensor[[4, 4], pl.INT32]:
                pl.system.syncall(
                    mode=pl.SyncAllMode.SOFT,
                    core_type=pl.KernelType.MIX,
                    gm_workspace=ws,
                    used_cores=0,
                )
                return ws

        mlir = self._generate_mlir(Prog)
        line = next((ln for ln in mlir.splitlines() if "pto.syncall(" in ln), "")
        assert line, f"soft pto.syncall not found in MLIR:\n{mlir}"
        assert "!pto.ptr<i32>" in line, f"gm pointer missing:\n{line}"
        assert line.split(" : ", 1)[0].count(",") == 0, f"unexpected used_cores operand:\n{line}"
        assert "mode = #pto.sync_all_mode<soft>" in line, f"soft mode missing:\n{line}"

    def test_syncall_soft_accepts_dynamic_participant_count(self):
        """An INT32 scalar lowers as the optional soft used_cores operand."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_syncall_soft(
                self,
                ws: pl.Tensor[[16], pl.INT32],
                participants: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[16], pl.INT32]:
                pl.system.syncall(
                    mode=pl.SyncAllMode.SOFT,
                    core_type=pl.KernelType.AIV,
                    gm_workspace=ws,
                    used_cores=participants,
                )
                return ws

        mlir = self._generate_mlir(Prog)
        line = next((ln for ln in mlir.splitlines() if "pto.syncall(" in ln), "")
        assert line, f"soft pto.syncall not found in MLIR:\n{mlir}"
        assert "!pto.ptr<i32>, i32" in line, f"dynamic used_cores missing:\n{line}"
        assert line.split(" : ", 1)[0].count(",") == 1, f"unexpected soft operand count:\n{line}"

    def test_syncall_soft_rejects_workspace_smaller_than_cache_line(self):
        """A static GM workspace must contain at least 16 INT32 elements."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_syncall_soft(self, ws: pl.Tensor[[15], pl.INT32]) -> pl.Tensor[[15], pl.INT32]:
                pl.system.syncall(
                    mode=pl.SyncAllMode.SOFT,
                    core_type=pl.KernelType.AIV,
                    gm_workspace=ws,
                    used_cores=0,
                )
                return ws

        with pytest.raises(ValueError, match="at least 16 INT32 elements"):
            self._generate_mlir(Prog)


class TestFenceCodegen:
    """Tests that pl.system.fence lowers to pto.fence.barrier_all."""

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

    def test_fence_emits_barrier_all_gm(self):
        """pl.system.fence() emits pto.fence.barrier_all #pto.fence_scope<gm>."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_fence(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                pl.system.fence()
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(tile, [0, 0], out)
                return updated

        mlir = self._generate_mlir(Prog)
        assert "pto.fence.barrier_all #pto.fence_scope<gm>" in mlir, (
            f"pto.fence.barrier_all not found in MLIR:\n{mlir}"
        )


class TestPipeBarrierCodegen:
    """Tests that pl.system.bar_* lower to pto.barrier on the expected pipe."""

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

    def test_bar_all_emits_pipe_all_barrier(self):
        """pl.system.bar_all() emits pto.barrier <PIPE_ALL>."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_bar_all(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                pl.system.bar_all()
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(tile, [0, 0], out)
                return updated

        mlir = self._generate_mlir(Prog)
        assert "pto.barrier <PIPE_ALL>" in mlir, f"pto.barrier <PIPE_ALL> not found in MLIR:\n{mlir}"

    def test_bar_v_emits_pipe_v_barrier(self):
        """pl.system.bar_v() emits pto.barrier <PIPE_V>."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_bar_v(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                pl.system.bar_v()
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(tile, [0, 0], out)
                return updated

        mlir = self._generate_mlir(Prog)
        assert "pto.barrier <PIPE_V>" in mlir, f"pto.barrier <PIPE_V> not found in MLIR:\n{mlir}"

    def test_bar_m_emits_pipe_m_barrier(self):
        """pl.system.bar_m() emits pto.barrier <PIPE_M>."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_bar_m(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                pl.system.bar_m()
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(tile, [0, 0], out)
                return updated

        mlir = self._generate_mlir(Prog)
        assert "pto.barrier <PIPE_M>" in mlir, f"pto.barrier <PIPE_M> not found in MLIR:\n{mlir}"


def _cmo_cacheinvalid_line(mlir: str) -> str:
    """Return the single `pto.cmo.cacheinvalid` line (the whole module contains a
    partition_tensor_view from the surrounding tile.store, so cmo-form assertions
    must inspect this line, not the module)."""
    lines = [line.strip() for line in mlir.splitlines() if "pto.cmo.cacheinvalid" in line]
    assert len(lines) == 1, f"expected exactly one pto.cmo.cacheinvalid line, got {lines}"
    return lines[0]


class TestCacheInvalidCodegen:
    """Tests that pl.system.cacheinvalid lowers to a partition-view or whole-GM cmo."""

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

    @pytest.mark.parametrize(
        "shapes, offsets, expected_dims",
        [
            # A single element is not special-cased: it is a 1x1 region.
            ([1, 1], [0, 8], "1x1"),
            ([16, 16], [0, 0], "16x16"),
        ],
        ids=["single_element", "full_region"],
    )
    def test_cacheinvalid_region_emits_partition_view(self, shapes, offsets, expected_dims):
        """Every region size lowers through the same partition_tensor_view cmo.

        A raw `!pto.ptr` operand is rejected by ptoas — at parse without a type
        annotation, and by the lowering pass with one — so the all-ones case must
        take the partition-view path too, not a scalar `pto.addptr` shortcut.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_cacheinvalid_region(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(tile, [0, 0], out)
                pl.system.cacheinvalid(updated, shapes, offsets)
                return updated

        mlir = self._generate_mlir(Prog)
        assert "pto.partition_view" in mlir, f"pto.partition_view not found in MLIR:\n{mlir}"
        cmo_line = _cmo_cacheinvalid_line(mlir)
        assert "single_cache_line" in cmo_line
        assert f"!pto.partition_tensor_view<{expected_dims}xf32>" in cmo_line, (
            f"expected a {expected_dims} partition view operand, got: {cmo_line}"
        )

    def test_cacheinvalid_dynamic_offset(self):
        """A runtime offset expression (loop-var arithmetic) reaches the partition view."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_cacheinvalid_dyn(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(tile, [0, 0], out)
                for i in pl.range(4):
                    # offset row is computed at runtime from the loop index
                    pl.system.cacheinvalid(updated, [1, 1], [i, 0])
                return updated

        mlir = self._generate_mlir(Prog)
        # The dynamic row index feeds the partition view's offsets operand.
        assert "pto.partition_view" in mlir, f"pto.partition_view not found in MLIR:\n{mlir}"
        cmo_line = _cmo_cacheinvalid_line(mlir)
        assert "single_cache_line" in cmo_line
        assert "!pto.partition_tensor_view<1x1xf32>" in cmo_line, f"unexpected cmo operand: {cmo_line}"

    def test_cacheinvalid_whole_gm_no_args_emits_all(self):
        """The no-argument form invalidates the whole GM address space."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_cacheinvalid_all(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(tile, [0, 0], out)
                pl.system.cacheinvalid()
                return updated

        mlir = self._generate_mlir(Prog)
        cmo_line = _cmo_cacheinvalid_line(mlir)
        # Whole-GM form: `all #pto.address_space<gm>`, no ptr / partition view.
        assert "all #pto.address_space<gm>" in cmo_line, f"whole-GM form not emitted: {cmo_line}"
        assert "single_cache_line" not in cmo_line
        assert "partition_tensor_view" not in cmo_line


class TestB03TriAndGatherCodegen:
    """Exact PTOAS spellings and attributes for the B03 operators."""

    @staticmethod
    def _generate_mlir(program_cls, backend_type=BackendType.Ascend910B) -> str:
        backend.reset_for_testing()
        backend.set_backend_type(backend_type)
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
        funcs = list(optimized.functions.values())
        assert funcs
        target = next((func for func in funcs if ir.is_incore_type(func.func_type)), funcs[0])
        single = ir.Program([target], target.name, optimized.span)
        return codegen.PTOCodegen().generate(single)

    @staticmethod
    def _ssa_pattern(logical: str) -> re.Pattern[str]:
        name = logical.removeprefix("%")
        suffix = ""
        for candidate in ("_pview", "_view"):
            if name.endswith(candidate):
                name = name[: -len(candidate)]
                suffix = candidate
                break
        return re.compile(rf"%{re.escape(name)}(?:__ssa_v\d+)?{re.escape(suffix)}")

    @classmethod
    def _resolve_ssa(cls, mlir: str, logical: str) -> str:
        matches = cls._ssa_pattern(logical).findall(mlir)
        assert matches, f"SSA corresponding to {logical} not found"
        return matches[0]

    @classmethod
    def _alloc_tile_type(cls, mlir: str, ssa: str) -> str:
        actual_ssa = cls._resolve_ssa(mlir, ssa)
        line = next(
            strip_loc(line)
            for line in mlir.splitlines()
            if line.strip().startswith(f"{actual_ssa} = pto.alloc_tile")
        )
        return line.rsplit(" : ", 1)[1]

    @classmethod
    def _assert_partition_view(
        cls,
        mlir: str,
        *,
        result: str,
        source: str,
        result_type: str,
    ) -> None:
        actual_result = cls._resolve_ssa(mlir, result)
        actual_source = cls._resolve_ssa(mlir, source)
        line = next(
            line.strip()
            for line in mlir.splitlines()
            if line.strip().startswith(f"{actual_result} = pto.partition_view")
        )
        assert line.startswith(f"{actual_result} = pto.partition_view {actual_source}, offsets = [")
        assert strip_loc(line).endswith(f" -> {result_type}")

    @classmethod
    def _assert_mgather(
        cls,
        mlir: str,
        *,
        operands: list[str],
        operand_types: list[str],
        result: str,
        result_type: str,
        attributes: str,
    ) -> None:
        line = next(strip_loc(line) for line in mlir.splitlines() if "pto.mgather" in line)
        actual_operands = [cls._resolve_ssa(mlir, operand) for operand in operands]
        actual_result = cls._resolve_ssa(mlir, result)
        expected = (
            f"pto.mgather ins({', '.join(actual_operands)} : {', '.join(operand_types)}) "
            f"outs({actual_result} : {result_type}) {{{attributes}}}"
        )
        assert line == expected

    def test_ttri_emits_upper_selector_and_partial_destination(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                out: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                mask: pl.Tile[[16, 32], pl.FP32] = pl.tile.tri(
                    1, [16, 32], valid_shape=[9, 21], dtype=pl.FP32, upper=True
                )
                return pl.store(mask, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        mask_ssa = self._resolve_ssa(mlir, "%mask")
        mask_type = self._alloc_tile_type(mlir, "%mask")
        line = next(strip_loc(line) for line in mlir.splitlines() if '"pto.ttri"' in line)
        assert "%c1_i32 = arith.constant 1 : i32" in mlir
        assert line == (
            f'"pto.ttri"(%c1_i32, {mask_ssa}) {{upperOrLower = 1 : i32}} : (i32, {mask_type}) -> ()'
        )
        alloc = next(
            line.strip()
            for line in mlir.splitlines()
            if line.strip().startswith(f"{mask_ssa} = pto.alloc_tile")
        )
        assert " valid_row = %c9_index valid_col = %c21_index : " in alloc

    @pytest.mark.parametrize("dtype", [pl.INT8, pl.UINT8, pl.BF16])
    def test_ttri_rejects_a5_only_dtypes_on_a2a3(self, dtype):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, out: pl.Tensor[[16, 32], dtype]):
                mask: pl.Tile[[16, 32], dtype] = pl.tile.tri(0, [16, 32], valid_shape=[9, 21], dtype=dtype)
                pl.store(mask, [0, 0], out)

        with pytest.raises(ValueError, match="not supported on the 'a2a3' backend"):
            self._generate_mlir(Prog)

    @pytest.mark.parametrize("dtype", [pl.INT8, pl.UINT8, pl.BF16])
    def test_ttri_accepts_extended_dtypes_on_a5(self, dtype):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, out: pl.Tensor[[16, 32], dtype]):
                mask: pl.Tile[[16, 32], dtype] = pl.tile.tri(0, [16, 32], valid_shape=[9, 21], dtype=dtype)
                pl.store(mask, [0, 0], out)

        assert '"pto.ttri"' in self._generate_mlir(Prog, BackendType.Ascend950)

    def test_tgatherb_emits_exact_two_input_name(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 128], pl.FP16],
                offset: pl.Tensor[[8, 8], pl.UINT32],
                out: pl.Tensor[[8, 128], pl.FP16],
            ) -> pl.Tensor[[8, 128], pl.FP16]:
                src_tile: pl.Tile[[16, 128], pl.FP16] = pl.load(src, [0, 0], [16, 128])
                offset_tile: pl.Tile[[8, 8], pl.UINT32] = pl.load(offset, [0, 0], [8, 8], valid_shape=[5, 5])
                gathered: pl.Tile[[8, 128], pl.FP16] = pl.tile.gatherb(src_tile, offset_tile)
                return pl.store(gathered, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        src_type = self._alloc_tile_type(mlir, "%src_tile")
        offset_type = self._alloc_tile_type(mlir, "%offset_tile")
        gathered_type = self._alloc_tile_type(mlir, "%gathered")
        src_ssa = self._resolve_ssa(mlir, "%src_tile")
        offset_ssa = self._resolve_ssa(mlir, "%offset_tile")
        gathered_ssa = self._resolve_ssa(mlir, "%gathered")
        line = next(strip_loc(line) for line in mlir.splitlines() if "pto.tgatherb" in line)
        assert line == (
            f"pto.tgatherb ins({src_ssa}, {offset_ssa} : {src_type}, {offset_type}) "
            f"outs({gathered_ssa} : {gathered_type})"
        )

    @pytest.mark.parametrize(
        ("dtype", "output_cols", "column", "aligned"),
        [
            (pl.UINT8, 256, 31, False),
            (pl.UINT8, 256, 32, True),
            (pl.FP16, 128, 15, False),
            (pl.FP16, 128, 16, True),
            (pl.FP32, 64, 7, False),
            (pl.FP32, 64, 8, True),
        ],
    )
    def test_tgatherb_materializes_unaligned_source_subview(self, dtype, output_cols, column, aligned):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 256], dtype],
                offset: pl.Tensor[[8, 8], pl.UINT32],
                out: pl.Tensor[[8, output_cols], dtype],
            ):
                src_tile: pl.Tile[[16, 256], dtype] = pl.load(src, [0, 0], [16, 256])
                src_view: pl.Tile[[16, 128], dtype] = pl.tile.slice(src_tile, [16, 128], [0, column])
                offset_tile: pl.Tile[[8, 8], pl.UINT32] = pl.load(offset, [0, 0], [8, 8])
                gathered: pl.Tile[[8, output_cols], dtype] = pl.tile.gatherb(src_view, offset_tile)
                pl.store(gathered, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        assert "pto.tgatherb" in mlir
        if aligned:
            assert "pto.subview" in mlir
            assert "pto.textract" not in mlir
        else:
            assert "pto.subview" not in mlir
            assert "pto.textract" in mlir

    def test_tgatherb_materializes_dynamic_column_subview_alignment(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 256], pl.FP16],
                offset: pl.Tensor[[8, 8], pl.UINT32],
                out: pl.Tensor[[8, 128], pl.FP16],
                column: pl.Scalar[pl.INDEX],
            ):
                src_tile: pl.Tile[[16, 256], pl.FP16] = pl.load(src, [0, 0], [16, 256])
                src_view: pl.Tile[[16, 128], pl.FP16] = pl.tile.slice(src_tile, [16, 128], [0, column])
                offset_tile: pl.Tile[[8, 8], pl.UINT32] = pl.load(offset, [0, 0], [8, 8])
                gathered: pl.Tile[[8, 128], pl.FP16] = pl.tile.gatherb(src_view, offset_tile)
                pl.store(gathered, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        assert "pto.tgatherb" in mlir
        assert "pto.subview" not in mlir
        assert "pto.textract" in mlir

    @pytest.mark.parametrize(
        ("dtype", "output_cols"),
        [
            (pl.UINT8, 256),
            (pl.FP16, 128),
            (pl.FP32, 64),
        ],
    )
    def test_tgatherb_accepts_dynamic_row_subview_with_aligned_stride(self, dtype, output_cols):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[32, 256], dtype],
                offset: pl.Tensor[[8, 8], pl.UINT32],
                out: pl.Tensor[[8, output_cols], dtype],
                row: pl.Scalar[pl.INDEX],
            ):
                src_tile: pl.Tile[[32, 256], dtype] = pl.load(src, [0, 0], [32, 256])
                src_view: pl.Tile[[16, 128], dtype] = pl.tile.slice(src_tile, [16, 128], [row, 0])
                offset_tile: pl.Tile[[8, 8], pl.UINT32] = pl.load(offset, [0, 0], [8, 8])
                gathered: pl.Tile[[8, output_cols], dtype] = pl.tile.gatherb(src_view, offset_tile)
                pl.store(gathered, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        subview_line = next(line for line in mlir.splitlines() if "pto.subview" in line)
        gather_line = next(line for line in mlir.splitlines() if "pto.tgatherb" in line)
        subview_ssa = subview_line.split("=", maxsplit=1)[0].strip()
        assert subview_ssa in gather_line

    @pytest.mark.parametrize(
        ("dtype", "output_cols"),
        [
            (pl.UINT8, 256),
            (pl.FP16, 128),
            (pl.FP32, 64),
        ],
    )
    def test_tgatherb_materializes_dynamic_row_subview_with_unaligned_stride(self, dtype, output_cols):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[32, 255], dtype],
                offset: pl.Tensor[[8, 8], pl.UINT32],
                out: pl.Tensor[[8, output_cols], dtype],
                row: pl.Scalar[pl.INDEX],
            ):
                src_tile: pl.Tile[[32, 255], dtype] = pl.load(src, [0, 0], [32, 255])
                src_view: pl.Tile[[16, 128], dtype] = pl.tile.slice(src_tile, [16, 128], [row, 0])
                offset_tile: pl.Tile[[8, 8], pl.UINT32] = pl.load(offset, [0, 0], [8, 8])
                gathered: pl.Tile[[8, output_cols], dtype] = pl.tile.gatherb(src_view, offset_tile)
                pl.store(gathered, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        assert "pto.tgatherb" in mlir
        assert "pto.subview" not in mlir
        assert "pto.textract" in mlir

    def test_tgatherb_materializes_nested_dynamic_column_subview_alignment(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 256], pl.FP16],
                offset: pl.Tensor[[8, 8], pl.UINT32],
                out: pl.Tensor[[8, 128], pl.FP16],
                column: pl.Scalar[pl.INDEX],
            ):
                src_tile: pl.Tile[[16, 256], pl.FP16] = pl.load(src, [0, 0], [16, 256])
                outer: pl.Tile[[16, 128], pl.FP16] = pl.tile.slice(src_tile, [16, 128], [0, column])
                inner: pl.Tile[[16, 128], pl.FP16] = pl.tile.slice(outer, [16, 128], [0, 0])
                offset_tile: pl.Tile[[8, 8], pl.UINT32] = pl.load(offset, [0, 0], [8, 8])
                gathered: pl.Tile[[8, 128], pl.FP16] = pl.tile.gatherb(inner, offset_tile)
                pl.store(gathered, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        assert "pto.tgatherb" in mlir
        assert "pto.subview" not in mlir
        assert "pto.textract" in mlir

    def test_tgatherb_accepts_nested_static_offsets_with_aligned_total(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 256], pl.FP16],
                offset: pl.Tensor[[8, 8], pl.UINT32],
                out: pl.Tensor[[8, 128], pl.FP16],
            ):
                src_tile: pl.Tile[[16, 256], pl.FP16] = pl.load(src, [0, 0], [16, 256])
                outer: pl.Tile[[16, 240], pl.FP16] = pl.tile.slice(src_tile, [16, 240], [0, 1])
                inner: pl.Tile[[16, 128], pl.FP16] = pl.tile.slice(outer, [16, 128], [0, 15])
                offset_tile: pl.Tile[[8, 8], pl.UINT32] = pl.load(offset, [0, 0], [8, 8])
                gathered: pl.Tile[[8, 128], pl.FP16] = pl.tile.gatherb(inner, offset_tile)
                pl.store(gathered, [0, 0], out)

        assert "pto.tgatherb" in self._generate_mlir(Prog)

    def test_mgather_row_emits_correct_name_and_explicit_coalesce(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[64, 32], pl.FP32],
                idx: pl.Tensor[[1, 16], pl.INT32],
                out: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                idx_tile: pl.Tile[[1, 16], pl.INT32] = pl.load(idx, [0, 0], [1, 16])
                gathered: pl.Tile[[16, 32], pl.FP32] = pl.tile.mgather(mem, idx_tile, coalesce="row")
                return pl.store(gathered, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        assert "pto.tmgather" not in mlir
        mem_type = "!pto.partition_tensor_view<64x32xf32>"
        self._assert_partition_view(mlir, result="%mem_pview", source="%mem_view", result_type=mem_type)
        self._assert_mgather(
            mlir,
            operands=["%mem_pview", "%idx_tile"],
            operand_types=[mem_type, self._alloc_tile_type(mlir, "%idx_tile")],
            result="%gathered",
            result_type=self._alloc_tile_type(mlir, "%gathered"),
            attributes="coalesce = #pto<coalesce row>",
        )

    def test_mgather_elem_emits_explicit_coalesce(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[256], pl.FP32],
                idx: pl.Tensor[[8, 32], pl.INT32],
                out: pl.Tensor[[8, 32], pl.FP32],
            ) -> pl.Tensor[[8, 32], pl.FP32]:
                idx_tile: pl.Tile[[8, 32], pl.INT32] = pl.load(idx, [0, 0], [8, 32])
                gathered: pl.Tile[[8, 32], pl.FP32] = pl.tile.mgather(mem, idx_tile, coalesce="elem")
                return pl.store(gathered, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        mem_type = "!pto.partition_tensor_view<256xf32>"
        self._assert_partition_view(mlir, result="%mem_pview", source="%mem_view", result_type=mem_type)
        self._assert_mgather(
            mlir,
            operands=["%mem_pview", "%idx_tile"],
            operand_types=[mem_type, self._alloc_tile_type(mlir, "%idx_tile")],
            result="%gathered",
            result_type=self._alloc_tile_type(mlir, "%gathered"),
            attributes="coalesce = #pto<coalesce elem>",
        )

    def test_mgather_a2a3_rejects_column_vector_row_indices(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[64, 32], pl.FP16],
                idx: pl.Tensor[[16, 1], pl.INT32],
                out: pl.Tensor[[16, 32], pl.FP16],
            ):
                idx_tile: pl.Tile[[16, 1], pl.INT32] = pl.load(idx, [0, 0], [16, 1])
                gathered: pl.Tile[[16, 32], pl.FP16] = pl.tile.mgather(mem, idx_tile, coalesce="row")
                pl.store(gathered, [0, 0], out)

        with pytest.raises(ValueError, match=r"requires idx shape \[1, R\]"):
            self._generate_mlir(Prog)
        assert "pto.mgather" in self._generate_mlir(Prog, BackendType.Ascend950)

    def test_mgather_a2a3_requires_aligned_vec_rows(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[64, 17], pl.FP16],
                idx: pl.Tensor[[1, 16], pl.INT32],
                out: pl.Tensor[[16, 17], pl.FP16],
            ):
                idx_tile: pl.Tile[[1, 16], pl.INT32] = pl.load(idx, [0, 0], [1, 16])
                gathered: pl.Tile[[16, 17], pl.FP16] = pl.tile.mgather(mem, idx_tile, coalesce="row")
                pl.store(gathered, [0, 0], out)

        with pytest.raises(ValueError, match="32-byte aligned"):
            self._generate_mlir(Prog)
        assert "pto.mgather" in self._generate_mlir(Prog, BackendType.Ascend950)

    def test_mgather_a2a3_requires_nd_source_for_vec_output(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[32, 64], pl.FP16],
                idx: pl.Tensor[[1, 16], pl.INT32],
                out: pl.Tensor[[16, 32], pl.FP16],
            ):
                mem_dn: pl.Tensor[
                    [64, 32],
                    pl.FP16,
                    pl.TensorView(stride=[1, 64], layout=pl.TensorLayout.DN),
                ] = pl.tensor.view(mem, layout=pl.TensorLayout.DN)
                idx_tile: pl.Tile[[1, 16], pl.INT32] = pl.load(idx, [0, 0], [1, 16])
                gathered: pl.Tile[[16, 32], pl.FP16] = pl.tile.mgather(mem_dn, idx_tile, coalesce="row")
                pl.store(gathered, [0, 0], out)

        with pytest.raises(ValueError, match="requires an ND source tensor"):
            self._generate_mlir(Prog)

    def test_mgather_emits_nondefault_out_of_bounds_mode(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[64, 32], pl.FP32],
                idx: pl.Tensor[[1, 16], pl.INT32],
                out: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                idx_tile: pl.Tile[[1, 16], pl.INT32] = pl.load(idx, [0, 0], [1, 16])
                gathered: pl.Tile[[16, 32], pl.FP32] = pl.tile.mgather(
                    mem, idx_tile, coalesce="row", gather_oob="zero"
                )
                return pl.store(gathered, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        mem_type = "!pto.partition_tensor_view<64x32xf32>"
        self._assert_partition_view(mlir, result="%mem_pview", source="%mem_view", result_type=mem_type)
        self._assert_mgather(
            mlir,
            operands=["%mem_pview", "%idx_tile"],
            operand_types=[mem_type, self._alloc_tile_type(mlir, "%idx_tile")],
            result="%gathered",
            result_type=self._alloc_tile_type(mlir, "%gathered"),
            attributes=("coalesce = #pto<coalesce row>, gatherOob = #pto<gather_oob zero>"),
        )

    def test_mgather_mat_row_emits_gm_index_partition(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[64, 32], pl.FP16],
                idx: pl.Tensor[[1, 16], pl.INT32],
                eye: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                gathered: pl.Tile[[16, 32], pl.FP16] = pl.tile.mgather(
                    mem,
                    idx,
                    coalesce="row",
                    gather_oob="zero",
                    target_memory=pl.MemorySpace.Mat,
                    valid_shape=[9, 21],
                )
                eye_tile: pl.Tile[[16, 16], pl.FP16] = pl.load(
                    eye,
                    [0, 0],
                    [16, 16],
                    valid_shape=[16, 9],
                    target_memory=pl.MemorySpace.Mat,
                )
                product: pl.Tile[[16, 32], pl.FP32] = pl.matmul(eye_tile, gathered)
                return pl.store(product, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        mem_type = "!pto.partition_tensor_view<64x32xf16>"
        idx_type = "!pto.partition_tensor_view<1x16xi32>"
        self._assert_partition_view(mlir, result="%mem_pview", source="%mem_view", result_type=mem_type)
        self._assert_partition_view(mlir, result="%idx_pview", source="%idx_view", result_type=idx_type)
        self._assert_mgather(
            mlir,
            operands=["%mem_pview", "%idx_pview"],
            operand_types=[mem_type, idx_type],
            result="%gathered",
            result_type=self._alloc_tile_type(mlir, "%gathered"),
            attributes=("coalesce = #pto<coalesce row>, gatherOob = #pto<gather_oob zero>"),
        )
        gathered_alloc = next(
            line for line in mlir.splitlines() if "pto.alloc_tile" in line and "%gathered" in line
        )
        assert "loc=mat" in self._alloc_tile_type(mlir, "%gathered")
        assert "valid_row = %c9_index" in gathered_alloc
        assert "valid_col = %c21_index" in gathered_alloc

    @pytest.mark.parametrize(
        ("dtype", "dtype_text"),
        [
            (pl.FP8E4M3FN, "f8E4M3FN"),
            (pl.FP8E5M2, "f8E5M2"),
            (pl.HF8, "!pto.hif8"),
        ],
    )
    def test_mgather_a5_low_precision_reaches_mat_codegen(self, dtype, dtype_text):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[64, 32], dtype],
                idx: pl.Tensor[[1, 16], pl.INT32],
                eye: pl.Tensor[[16, 16], dtype],
                out: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                gathered: pl.Tile[[16, 32], dtype] = pl.tile.mgather(
                    mem,
                    idx,
                    target_memory=pl.MemorySpace.Mat,
                )
                eye_tile: pl.Tile[[16, 16], dtype] = pl.load(
                    eye, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
                )
                product: pl.Tile[[16, 32], pl.FP32] = pl.matmul(eye_tile, gathered)
                return pl.store(product, [0, 0], out)

        mlir = self._generate_mlir(Prog, BackendType.Ascend950)
        mem_type = f"!pto.partition_tensor_view<64x32x{dtype_text}>"
        idx_type = "!pto.partition_tensor_view<1x16xi32>"
        self._assert_partition_view(mlir, result="%mem_pview", source="%mem_view", result_type=mem_type)
        self._assert_partition_view(mlir, result="%idx_pview", source="%idx_view", result_type=idx_type)
        gathered_type = self._alloc_tile_type(mlir, "%gathered")
        assert "loc=mat" in gathered_type
        self._assert_mgather(
            mlir,
            operands=["%mem_pview", "%idx_pview"],
            operand_types=[mem_type, idx_type],
            result="%gathered",
            result_type=gathered_type,
            attributes="coalesce = #pto<coalesce row>",
        )

    @pytest.mark.parametrize(
        ("dtype", "dtype_text"),
        [
            (pl.FP8E4M3FN, "f8E4M3FN"),
            (pl.FP8E5M2, "f8E5M2"),
            (pl.HF8, "!pto.hif8"),
        ],
    )
    def test_mgather_a5_low_precision_reaches_mat_elem_codegen(self, dtype, dtype_text):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[512], dtype],
                idx: pl.Tensor[[16, 32], pl.INT32],
                scratch: pl.Tensor[[512], dtype],
                out: pl.Tensor[[16, 32], dtype],
            ) -> pl.Tensor[[16, 32], dtype]:
                gathered: pl.Tile[[16, 32], dtype] = pl.tile.mgather(
                    mem,
                    idx,
                    coalesce="elem",
                    target_memory=pl.MemorySpace.Mat,
                    scratch=scratch,
                )
                gathered_vec: pl.Tile[[16, 32], dtype] = pl.move(gathered, target_memory=pl.MemorySpace.Vec)
                return pl.store(gathered_vec, [0, 0], out)

        mlir = self._generate_mlir(Prog, BackendType.Ascend950)
        mem_type = f"!pto.partition_tensor_view<512x{dtype_text}>"
        idx_type = "!pto.partition_tensor_view<16x32xi32>"
        scratch_type = mem_type
        self._assert_partition_view(mlir, result="%mem_pview", source="%mem_view", result_type=mem_type)
        self._assert_partition_view(mlir, result="%idx_pview", source="%idx_view", result_type=idx_type)
        self._assert_partition_view(
            mlir,
            result="%scratch_pview",
            source="%scratch_view",
            result_type=scratch_type,
        )
        gathered_type = self._alloc_tile_type(mlir, "%gathered")
        assert "loc=mat" in gathered_type
        self._assert_mgather(
            mlir,
            operands=["%mem_pview", "%idx_pview", "%scratch_pview"],
            operand_types=[mem_type, idx_type, scratch_type],
            result="%gathered",
            result_type=gathered_type,
            attributes="coalesce = #pto<coalesce elem>",
        )

    @pytest.mark.parametrize("dtype", [pl.FP8E4M3FN, pl.FP8E5M2, pl.HF8])
    def test_mgather_rejects_a5_only_dtypes_on_a2a3(self, dtype):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[64, 32], dtype],
                idx: pl.Tensor[[1, 16], pl.INT32],
                eye: pl.Tensor[[16, 16], dtype],
                out: pl.Tensor[[16, 32], pl.FP32],
            ):
                gathered: pl.Tile[[16, 32], dtype] = pl.tile.mgather(
                    mem, idx, target_memory=pl.MemorySpace.Mat
                )
                eye_tile: pl.Tile[[16, 16], dtype] = pl.load(
                    eye, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
                )
                product: pl.Tile[[16, 32], pl.FP32] = pl.matmul(eye_tile, gathered)
                pl.store(product, [0, 0], out)

        with pytest.raises(ValueError, match="not supported on the 'a2a3' backend"):
            self._generate_mlir(Prog)

    def test_mgather_mat_elem_emits_gm_scratch_partition(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[256], pl.FP16],
                idx: pl.Tensor[[16, 16], pl.INT32],
                scratch: pl.Tensor[[256], pl.FP16],
                eye: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                gathered: pl.Tile[[16, 16], pl.FP16] = pl.tile.mgather(
                    mem,
                    idx,
                    coalesce="elem",
                    gather_oob="wrap",
                    target_memory=pl.MemorySpace.Mat,
                    scratch=scratch,
                )
                eye_tile: pl.Tile[[16, 16], pl.FP16] = pl.load(
                    eye, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
                )
                product: pl.Tile[[16, 16], pl.FP32] = pl.matmul(eye_tile, gathered)
                return pl.store(product, [0, 0], out)

        mlir = self._generate_mlir(Prog)
        mem_type = "!pto.partition_tensor_view<256xf16>"
        idx_type = "!pto.partition_tensor_view<16x16xi32>"
        scratch_type = "!pto.partition_tensor_view<256xf16>"
        self._assert_partition_view(mlir, result="%mem_pview", source="%mem_view", result_type=mem_type)
        self._assert_partition_view(mlir, result="%idx_pview", source="%idx_view", result_type=idx_type)
        self._assert_partition_view(
            mlir,
            result="%scratch_pview",
            source="%scratch_view",
            result_type=scratch_type,
        )
        gathered_type = self._alloc_tile_type(mlir, "%gathered")
        assert "loc=mat" in gathered_type
        self._assert_mgather(
            mlir,
            operands=["%mem_pview", "%idx_pview", "%scratch_pview"],
            operand_types=[mem_type, idx_type, scratch_type],
            result="%gathered",
            result_type=gathered_type,
            attributes=("coalesce = #pto<coalesce elem>, gatherOob = #pto<gather_oob wrap>"),
        )

    def test_mgather_mat_elem_rejects_overlapping_scratch_view(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[16, 16], pl.FP16],
                idx: pl.Tensor[[16, 16], pl.INT32],
                eye: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Tensor[[16, 16], pl.FP32],
            ):
                scratch: pl.Tensor[[256], pl.FP16] = pl.tensor.view(mem, [256])
                gathered: pl.Tile[[16, 16], pl.FP16] = pl.tile.mgather(
                    mem,
                    idx,
                    coalesce="elem",
                    target_memory=pl.MemorySpace.Mat,
                    scratch=scratch,
                )
                eye_tile: pl.Tile[[16, 16], pl.FP16] = pl.load(
                    eye, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
                )
                product: pl.Tile[[16, 16], pl.FP32] = pl.matmul(eye_tile, gathered)
                pl.store(product, [0, 0], out)

        with pytest.raises(ValueError, match="scratch must not overlap mem"):
            self._generate_mlir(Prog)

    def test_mgather_mat_elem_rejects_scratch_view_overlapping_idx(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[256], pl.INT32],
                idx: pl.Tensor[[16, 16], pl.INT32],
                out: pl.Tensor[[16, 16], pl.INT32],
            ):
                scratch: pl.Tensor[[256], pl.INT32] = pl.tensor.view(idx, [256])
                gathered: pl.Tile[[16, 16], pl.INT32] = pl.tile.mgather(
                    mem,
                    idx,
                    coalesce="elem",
                    target_memory=pl.MemorySpace.Mat,
                    scratch=scratch,
                )
                gathered_vec: pl.Tile[[16, 16], pl.INT32] = pl.move(
                    gathered, target_memory=pl.MemorySpace.Vec
                )
                pl.store(gathered_vec, [0, 0], out)

        with pytest.raises(ValueError, match="scratch must not overlap idx"):
            self._generate_mlir(Prog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
