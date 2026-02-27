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
        a_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
        b_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
        result: pl.Tile[[128, 128], pl.FP32] = pl.add(a_tile, b_tile)
        out: pl.Tensor[[M, N], pl.FP32] = pl.store(result, [0, 0], [128, 128], output)
        return out


def test_add_kernel_dynamic_shape_pto_codegen():
    """Test PTO codegen generates correct signature and tensor views for dynamic shapes."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)
    func = AddKernelDynamic.get_function("add_kernel")
    assert func is not None
    program = ir.Program([func], "test_add_kernel", ir.Span.unknown())
    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    optimized = pm.run_passes(program)

    gen = codegen.PTOCodegen()
    mlir_code = gen.generate(optimized)
    print(mlir_code)

    # Dynamic index params appended to function signature
    assert "%arg3: index" in mlir_code
    assert "%arg4: index" in mlir_code
    # Dynamic dim variables used in make_tensor_view shape and strides
    assert "shape = [%arg3, %arg4]" in mlir_code
    assert "strides = [%arg4, %c1]" in mlir_code
    # Dynamic type annotation uses wildcard
    assert "!pto.tensor_view<?x?xf32>" in mlir_code
    # Dynamic dims must not appear as zero constants in make_tensor_view shape
    assert "shape = [%c0" not in mlir_code
