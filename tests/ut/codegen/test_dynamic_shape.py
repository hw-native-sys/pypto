# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for codegen with dynamic shape tensor parameters."""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors
# from type-checking annotations that reference module-level DynVar names.
# pyright: reportUndefinedVariable=false

import pypto.language as pl
from pypto import backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

M = pl.dynamic("M")
N = pl.dynamic("N")


@pl.program
class AddKernelDynamic:
    """Add kernel with dynamic shape tensor parameters."""

    @pl.function(type=pl.FunctionType.InCore)
    def add_kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        b: pl.Tensor[[M, N], pl.FP32],
        output: pl.Tensor[[M, N], pl.FP32],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        """Adds two tensors element-wise with dynamic shapes: result = a + b"""
        a_tile = pl.load(a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec)
        b_tile = pl.load(b, [0, 0], [128, 128])
        result = pl.add(a_tile, b_tile)
        out = pl.store(result, [0, 0], output)
        return out


@pl.program
class AddKernelValidShape:
    """Add kernel with static tensors but dynamic valid_shapes passed as scalars."""

    @pl.function(type=pl.FunctionType.InCore)
    def add_kernel(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        output: pl.Tensor[[128, 128], pl.FP32],
        M: pl.Scalar[pl.INDEX],
        N: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        """Loads 128x128 tiles but marks only [M, N] as valid: result = a + b"""
        a_tile = pl.load(a, [0, 0], [128, 128], valid_shapes=[M, N])
        b_tile = pl.load(b, [0, 0], [128, 128], valid_shapes=[M, N])
        result = pl.add(a_tile, b_tile)
        out = pl.store(result, [0, 0], output)
        return out


@pl.program
class AddKernelValidShapeExpr:
    """Add kernel with valid_shapes computed from a runtime expression (regression for #707)."""

    @pl.function(type=pl.FunctionType.InCore)
    def add_kernel(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        output: pl.Tensor[[128, 128], pl.FP32],
        M: pl.Scalar[pl.INDEX],
        N: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        """valid_shape elements come from pl.min(...) — i.e. ir::Call, not ir::Var."""
        valid_m = pl.min(M, N)
        a_tile = pl.load(a, [0, 0], [128, 128], valid_shapes=[valid_m, N])
        b_tile = pl.load(b, [0, 0], [128, 128], valid_shapes=[valid_m, N])
        result = pl.add(a_tile, b_tile)
        out = pl.store(result, [0, 0], output)
        return out


@pl.program
class AddKernelLoopDynamic:
    """Add kernel with dynamic shape tensor parameters."""

    @pl.function(type=pl.FunctionType.InCore)
    def add_kernel(
        self,
        a: pl.Tensor[[M, 128], pl.FP32],
        b: pl.Tensor[[M, 128], pl.FP32],
        output: pl.Tensor[[M, 128], pl.FP32],
    ) -> pl.Tensor[[M, 128], pl.FP32]:
        """Adds two tensors element-wise with dynamic shapes: result = a + b"""
        M = pl.tensor.dim(a, 0)
        for i, (out,) in pl.range(0, M, 2, init_values=(output,)):
            offset_1 = i * 2
            a_tile = pl.load(a, [offset_1, 0], [2, 128], target_memory=pl.MemorySpace.Vec)
            b_tile = pl.load(b, [offset_1, 0], [2, 128], target_memory=pl.MemorySpace.Vec)
            result = pl.add(a_tile, b_tile)
            updated: pl.Tensor[[M, 128], pl.FP32] = pl.store(result, [offset_1, 0], out)
            loop_out = pl.yield_(updated)
        return loop_out


def test_add_kernel_dynamic_shape_pto_codegen():
    """Test PTO codegen generates correct signature and tensor views for dynamic shapes."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    func = AddKernelDynamic.get_function("add_kernel")
    assert func is not None
    program = ir.Program([func], "test_add_kernel", ir.Span.unknown())
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    optimized = pm.run_passes(program)

    gen = codegen.PTOCodegen()
    mlir_code = gen.generate(optimized)

    # Dynamic index params appended to function signature
    assert "%arg3: index" in mlir_code
    assert "%arg4: index" in mlir_code
    # Dynamic dim variables used in make_tensor_view shape and strides
    assert "shape = [%arg3, %arg4]" in mlir_code
    assert "strides = [%arg4, %c1_index]" in mlir_code
    # Dynamic type annotation uses wildcard
    assert "!pto.tensor_view<?x?xf32>" in mlir_code
    # Dynamic dims must not appear as zero constants in make_tensor_view shape
    assert "shape = [%c0_index" not in mlir_code


def test_add_kernel_valid_shape_pto_codegen():
    """Test PTO codegen handles load with valid_shapes: tile allocated from shapes, M/N as index scalars."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    func = AddKernelValidShape.get_function("add_kernel")
    assert func is not None
    program = ir.Program([func], "test_add_kernel_valid_shape", ir.Span.unknown())
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    optimized = pm.run_passes(program)

    gen = codegen.PTOCodegen()
    mlir_code = gen.generate(optimized)

    # Scalar params M and N appear in the function signature as index type
    assert "%arg3: index" in mlir_code
    assert "%arg4: index" in mlir_code
    # Static tensor views use constant dims from the 128x128 tensor type
    assert "shape = [%c128_index, %c128_index]" in mlir_code
    assert "strides = [%c128_index, %c1_index]" in mlir_code
    assert "!pto.tensor_view<?x?xf32>" in mlir_code
    # partition_view follows valid_shapes (dynamic %arg3, %arg4) so the DMA
    # only fetches the valid region from GM. The partition_view type therefore
    # uses dynamic dims and its sizes use the valid_shape SSA values directly.
    assert "partition_tensor_view<?x?xf32>" in mlir_code
    assert "sizes = [%arg3, %arg4]" in mlir_code
    # tload is generated for each load
    assert "pto.tload" in mlir_code
    # alloc_tile has dynamic type (v_row=?, v_col=?) with dynamic operands
    assert "v_row=?" in mlir_code
    assert "v_col=?" in mlir_code
    # alloc_tile valid_row/valid_col use the dynamic valid_shape operands
    # (%arg3, %arg4), matching the partition_view sizes above.
    assert "valid_row = %arg3" in mlir_code
    assert "valid_col = %arg4" in mlir_code
    # No set_validshape: alloc_tile carries the valid_row/valid_col operands
    # and partition_view already reflects the same valid region.
    assert "pto.set_validshape" not in mlir_code


def test_add_kernel_valid_shape_expr_pto_codegen():
    """Regression for #707: alloc_tile must emit valid_row/valid_col operands when the
    valid_shape element is an arbitrary expression (e.g. pl.min(...)), not just an ir::Var.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    func = AddKernelValidShapeExpr.get_function("add_kernel")
    assert func is not None
    program = ir.Program([func], "test_add_kernel_valid_shape_expr", ir.Span.unknown())
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    optimized = pm.run_passes(program)

    gen = codegen.PTOCodegen()
    mlir_code = gen.generate(optimized)

    # The runtime min(M, N) must lower to an MLIR arith.minsi op.
    assert "arith.minsi" in mlir_code, "pl.min should lower to arith.minsi; codegen did not visit the expr"
    # alloc_tile type must be dynamic in the row dim (computed from min result).
    assert "v_row=?" in mlir_code
    # alloc_tile must carry a valid_row operand referencing the min SSA value
    # (without the fix this was empty, causing PTOAS verification to fail).
    assert "valid_row = %" in mlir_code, (
        "alloc_tile must emit valid_row operand even when the source is ir::Call"
    )
    # The N dimension still uses the scalar arg directly.
    assert "valid_col = %" in mlir_code


def test_add_kernel_loop_dynamic_pto_codegen():
    """Test that tensor.dim result variable is correctly mapped to the MLIR shape arg."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    func = AddKernelLoopDynamic.get_function("add_kernel")
    assert func is not None
    program = ir.Program([func], "test_add_kernel_loop", ir.Span.unknown())
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    optimized = pm.run_passes(program)

    gen = codegen.PTOCodegen()
    mlir_code = gen.generate(optimized)

    # M is the dynamic dim passed as trailing index arg (%arg3)
    assert "%arg3: index" in mlir_code
    # scf.for loop bound should use %arg3 (M resolved via tensor.dim)
    assert "to %arg3" in mlir_code
    # offset_1 = i * 2 must appear as a real SSA value in partition_view offsets (not blank)
    assert "offsets = [, " not in mlir_code


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
