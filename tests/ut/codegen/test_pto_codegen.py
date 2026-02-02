# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for PTOCodegen - MLIR generation from PyPTO IR.

The new PTOCodegen generates PTO-ISA MLIR dialect instead of PTO assembly.
Tests verify:
- Correct MLIR module structure
- Proper function signatures with tensor pointers
- make_tensor_view generation for tensor parameters
- alloc_tile generation for tile buffers
- Operator lowering (block.load/store/mul/adds -> pto.tload/tstore/tmul/tadds)
- SSA form with correct variable naming
"""

import pypto.language as pl
from pypto.ir import OptimizationStrategy, PassManager
from pypto.pypto_core.codegen import PTOCodegen


def _get_mlir_code(result):
    """Normalize generate() result to MLIR string (support both str and dict)."""
    return result if isinstance(result, str) else "".join(result.values())


def test_pto_codegen_basic_mlir_structure():
    """Test that PTOCodegen generates valid MLIR module structure."""

    @pl.program
    class BasicProgram:
        @pl.function
        def test_func(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.op.block.load(a, 0, 0, 32, 32)
            tile_b = pl.op.block.adds(tile_a, 1.0)
            pl.op.block.store(tile_b, 0, 0, 32, 32, b)

    # Compile with PTOAS strategy (applies necessary passes + codegen)
    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(BasicProgram)

    # Generate MLIR
    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify MLIR module structure
    assert "module {" in mlir_code
    assert "func.func @test_func" in mlir_code
    assert "return" in mlir_code
    assert "}" in mlir_code


def test_pto_codegen_tensor_parameters():
    """Test that tensor parameters generate correct make_tensor_view."""

    @pl.program
    class TensorParamProgram:
        @pl.function
        def tensor_param_func(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ):
            tile_a = pl.op.block.load(input_a, 0, 0, 32, 32)
            tile_b = pl.op.block.load(input_b, 0, 0, 32, 32)
            tile_c = pl.op.block.mul(tile_a, tile_b)
            pl.op.block.store(tile_c, 0, 0, 32, 32, output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(TensorParamProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify function signature with pointer types
    assert "%arg0: !pto.ptr<f32>" in mlir_code
    assert "%arg1: !pto.ptr<f32>" in mlir_code
    assert "%arg2: !pto.ptr<f32>" in mlir_code

    # Verify make_tensor_view generation
    assert "pto.make_tensor_view" in mlir_code
    assert "shape = [%c64, %c64]" in mlir_code or "shape = [%c32, %c32]" in mlir_code
    assert "strides = " in mlir_code
    assert "!pto.tensor_view<2xf32>" in mlir_code


def test_pto_codegen_alloc_tile():
    """Test that tile buffers generate alloc_tile operations."""

    @pl.program
    class AllocTileProgram:
        @pl.function
        def alloc_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.op.block.load(a, 0, 0, 32, 32)
            tile_b = pl.op.block.load(a, 0, 0, 32, 32)
            tile_c = pl.op.block.mul(tile_a, tile_b)
            pl.op.block.store(tile_c, 0, 0, 32, 32, b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(AllocTileProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify alloc_tile operations
    assert "pto.alloc_tile" in mlir_code
    assert "loc=ub" in mlir_code  # Unified buffer
    assert "dtype=f32" in mlir_code
    assert "rows=32, cols=32" in mlir_code


def test_pto_codegen_block_load_lowering():
    """Test that block.load generates subview + tload."""

    @pl.program
    class LoadProgram:
        @pl.function
        def load_test(self, input: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]):
            tile = pl.op.block.load(input, 0, 0, 32, 32)
            pl.op.block.store(tile, 0, 0, 32, 32, output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(LoadProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify subview generation
    assert "pto.subview" in mlir_code
    assert "offsets = [%c0, %c0]" in mlir_code
    assert "sizes = [%c32, %c32]" in mlir_code
    assert "!pto.tile_view<32x32xf32>" in mlir_code

    # Verify tload generation
    assert "pto.tload" in mlir_code
    assert "ins(" in mlir_code
    assert "outs(" in mlir_code
    assert "!pto.tile_buf<" in mlir_code


def test_pto_codegen_block_store_lowering():
    """Test that block.store generates subview + tstore."""

    @pl.program
    class StoreProgram:
        @pl.function
        def store_test(self, input: pl.Tensor[[32, 32], pl.FP32], output: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.op.block.load(input, 0, 0, 32, 32)
            pl.op.block.store(tile, 0, 0, 32, 32, output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(StoreProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify tstore generation
    assert "pto.tstore" in mlir_code
    assert "ins(" in mlir_code
    assert "outs(" in mlir_code


def test_pto_codegen_block_mul():
    """Test that block.mul generates pto.tmul."""

    @pl.program
    class MulProgram:
        @pl.function
        def mul_test(
            self,
            a: pl.Tensor[[32, 32], pl.FP32],
            b: pl.Tensor[[32, 32], pl.FP32],
            c: pl.Tensor[[32, 32], pl.FP32],
        ):
            tile_a = pl.op.block.load(a, 0, 0, 32, 32)
            tile_b = pl.op.block.load(b, 0, 0, 32, 32)
            tile_c = pl.op.block.mul(tile_a, tile_b)
            pl.op.block.store(tile_c, 0, 0, 32, 32, c)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(MulProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify tmul generation
    assert "pto.tmul" in mlir_code
    assert "ins(" in mlir_code
    assert "outs(" in mlir_code


def test_pto_codegen_block_adds():
    """Test that block.adds generates pto.tadds with scalar constant."""

    @pl.program
    class AddsProgram:
        @pl.function
        def adds_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.op.block.load(a, 0, 0, 32, 32)
            tile_b = pl.op.block.adds(tile_a, 3.14)
            pl.op.block.store(tile_b, 0, 0, 32, 32, b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(AddsProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify tadds generation
    assert "pto.tadds" in mlir_code

    # Verify scalar constant generation
    assert "arith.constant" in mlir_code
    assert ": f32" in mlir_code


def test_pto_codegen_constants():
    """Test that constants are generated correctly."""

    @pl.program
    class ConstantProgram:
        @pl.function
        def const_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.op.block.load(a, 0, 0, 32, 32)
            pl.op.block.store(tile_a, 0, 0, 32, 32, b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(ConstantProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify index constants
    assert "arith.constant" in mlir_code
    assert ": index" in mlir_code
    assert "%c0" in mlir_code or "%c32" in mlir_code


def test_pto_codegen_ssa_naming():
    """Test that SSA value names are correct."""

    @pl.program
    class SSAProgram:
        @pl.function
        def ssa_test(
            self,
            a: pl.Tensor[[32, 32], pl.FP32],
            b: pl.Tensor[[32, 32], pl.FP32],
            c: pl.Tensor[[32, 32], pl.FP32],
        ):
            tile_a = pl.op.block.load(a, 0, 0, 32, 32)
            tile_b = pl.op.block.load(b, 0, 0, 32, 32)
            tile_c = pl.op.block.mul(tile_a, tile_b)
            pl.op.block.store(tile_c, 0, 0, 32, 32, c)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(SSAProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify SSA value naming pattern
    assert "%arg0" in mlir_code  # Function parameters
    assert "%0" in mlir_code or "%1" in mlir_code  # Temporary values
    assert "%c" in mlir_code  # Constants


def test_pto_codegen_code_generation_order():
    """Test that code is generated in correct order: constants, views, allocs, body."""

    @pl.program
    class OrderProgram:
        @pl.function
        def order_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.op.block.load(a, 0, 0, 32, 32)
            pl.op.block.store(tile, 0, 0, 32, 32, b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(OrderProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    lines = mlir_code.split("\n")

    # Find indices of key operations
    const_idx = next((i for i, line in enumerate(lines) if "arith.constant" in line), -1)
    view_idx = next((i for i, line in enumerate(lines) if "make_tensor_view" in line), -1)
    alloc_idx = next((i for i, line in enumerate(lines) if "alloc_tile" in line), -1)
    load_idx = next((i for i, line in enumerate(lines) if "tload" in line), -1)

    # Verify order: constants < make_tensor_view < alloc_tile < operations
    assert const_idx < view_idx, "Constants should come before make_tensor_view"
    assert view_idx < alloc_idx, "make_tensor_view should come before alloc_tile"
    assert alloc_idx < load_idx, "alloc_tile should come before tload"


def test_pto_codegen_multiple_functions():
    """Test PTOCodegen with multiple functions."""

    @pl.program
    class MultiFunc:
        @pl.function
        def func1(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.op.block.load(a, 0, 0, 32, 32)
            pl.op.block.store(tile, 0, 0, 32, 32, b)

        @pl.function
        def func2(self, x: pl.Tensor[[32, 32], pl.FP32], y: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.op.block.load(x, 0, 0, 32, 32)
            pl.op.block.store(tile, 0, 0, 32, 32, y)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(MultiFunc)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify both functions are present
    assert "func.func @func1" in mlir_code
    assert "func.func @func2" in mlir_code


def test_pto_codegen_reusability():
    """Test that the same PTOCodegen instance can be used multiple times."""

    @pl.program
    class ReusableProgram:
        @pl.function
        def test_func(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.op.block.load(a, 0, 0, 32, 32)
            pl.op.block.store(tile, 0, 0, 32, 32, b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(ReusableProgram)

    # Use the same codegen instance multiple times
    codegen = PTOCodegen()

    code1 = _get_mlir_code(codegen.generate(transformed_program))
    code2 = _get_mlir_code(codegen.generate(transformed_program))

    # Verify both calls produce valid code
    assert isinstance(code1, str)
    assert isinstance(code2, str)
    assert "func.func @test_func" in code1
    assert "func.func @test_func" in code2
    assert code1 == code2  # Should produce identical output
