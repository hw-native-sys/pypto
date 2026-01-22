# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for PTOCodegen - PTO assembly generation from PyPTO IR."""

from pypto import DataType, ir
from pypto.ir import IRBuilder


def test_pto_codegen_basic():
    """Test basic PTOCodegen functionality - generates PTO assembly."""
    # Create a simple program
    ib = IRBuilder()

    with ib.function("test_func") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        result = ib.let("result", ir.op.block.muls(x, 2.0))
        ib.return_stmt(result)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    # Apply codegen
    codegen = ir.PTOCodegen()
    pto_code = codegen.generate(program)

    # Verify the result is PTO assembly string
    assert isinstance(pto_code, str)

    # Verify generated code contains expected elements
    assert "func @test_func" in pto_code
    assert "alloc_tile" in pto_code
    assert "tmuls" in pto_code
    assert "return" in pto_code


def test_pto_codegen_tile_declarations():
    """Test that tile declarations are correctly generated."""
    ib = IRBuilder()

    with ib.function("test_tiles") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([16, 32], DataType.FP16))

        y = ib.let("y", ir.op.block.mul(x, x))  # 8x8 tile
        z = ib.let("z", ir.op.block.adds(y, 1.0))  # 8x8 tile
        ib.return_stmt(z)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    codegen = ir.PTOCodegen()
    pto_code = codegen.generate(program)

    # Verify tile declarations
    assert "%y = alloc_tile" in pto_code
    assert "%z = alloc_tile" in pto_code


def test_pto_codegen_for_loop():
    """Test that ForStmt is correctly converted to FOR/ENDFOR."""
    ib = IRBuilder()

    with ib.function("test_loop") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        # Create loop variable
        i = ib.var("i", ir.ScalarType(DataType.INT32))

        # Create for loop
        with ib.for_loop(i, 0, 10, 1):
            ib.let("y", ir.op.block.muls(x, 2.0))

        ib.return_stmt(x)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    codegen = ir.PTOCodegen()
    pto_code = codegen.generate(program)

    # Verify for loop structure
    assert "FOR %i:" in pto_code
    assert "ENDFOR" in pto_code


def test_pto_codegen_scalar_operations():
    """Test that scalar operations use correct PTO instructions."""
    ib = IRBuilder()

    with ib.function("test_scalar_ops") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        y = ib.let("y", ir.op.block.muls(x, 2.0))
        z = ib.let("z", ir.op.block.divs(y, 3.0))
        w = ib.let("w", ir.op.block.adds(z, 1.0))
        ib.return_stmt(w)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    codegen = ir.PTOCodegen()
    pto_code = codegen.generate(program)

    # Verify scalar operation names
    assert "tmuls" in pto_code
    assert "tdivs" in pto_code
    assert "tadds" in pto_code
    # Verify scalar values are present
    assert "2" in pto_code
    assert "3" in pto_code
    assert "1" in pto_code


def test_pto_codegen_binary_operations():
    """Test that binary tile operations are correctly generated."""
    ib = IRBuilder()

    with ib.function("test_binary_ops") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        y = f.param("y", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        z1 = ib.let("z1", ir.op.block.mul(x, y))
        z2 = ib.let("z2", ir.op.block.add(z1, x))
        z3 = ib.let("z3", ir.op.block.sub(z2, y))
        ib.return_stmt(z3)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    codegen = ir.PTOCodegen()
    pto_code = codegen.generate(program)

    # Verify binary operation names
    assert "tmul" in pto_code
    assert "tadd" in pto_code
    assert "tsub" in pto_code
    # Verify result variables
    assert "%z1" in pto_code
    assert "%z2" in pto_code
    assert "%z3" in pto_code


def test_pto_codegen_data_types():
    """Test that different data types are correctly converted to PTO types."""
    ib = IRBuilder()

    with ib.function("test_types") as f:
        x_fp32 = f.param("x_fp32", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        y_fp16 = ib.let("y_fp16", ir.op.block.mul(x_fp32, x_fp32))
        ib.return_stmt(y_fp16)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    codegen = ir.PTOCodegen()
    pto_code = codegen.generate(program)

    # Verify PTO type conversions
    assert "f32" in pto_code


def test_pto_codegen_multiple_functions():
    """Test PTOCodegen with multiple functions."""
    # Create program with two functions
    ib1 = IRBuilder()
    with ib1.function("func1") as f1:
        x = f1.param("x", ir.TileType([8, 8], DataType.FP32))
        f1.return_type(ir.TileType([8, 8], DataType.FP32))
        result = ib1.let("result", ir.op.block.muls(x, 2.0))
        ib1.return_stmt(result)
    func1 = f1.get_result()

    ib2 = IRBuilder()
    with ib2.function("func2") as f2:
        y = f2.param("y", ir.TileType([8, 8], DataType.FP32))
        f2.return_type(ir.TileType([8, 8], DataType.FP32))
        result = ib2.let("result", ir.op.block.adds(y, 1.0))
        ib2.return_stmt(result)
    func2 = f2.get_result()

    program = ir.Program([func1, func2], "multi_func_program", ir.Span.unknown())

    # Apply codegen
    codegen = ir.PTOCodegen()
    pto_code = codegen.generate(program)

    # Verify both functions are generated
    assert "func @func1" in pto_code
    assert "func @func2" in pto_code


def test_pto_codegen_reusability():
    """Test that the same PTOCodegen instance can be used multiple times."""
    ib = IRBuilder()

    with ib.function("test_func") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))
        y = ib.let("y", ir.op.block.muls(x, 2.0))
        ib.return_stmt(y)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    # Use the same codegen instance multiple times
    codegen = ir.PTOCodegen()

    code1 = codegen.generate(program)
    code2 = codegen.generate(program)

    # Verify both calls produce valid code
    assert isinstance(code1, str)
    assert isinstance(code2, str)
    assert "func @test_func" in code1
    assert "func @test_func" in code2


def test_pto_codegen_with_dtype_target_isa():
    """Test that generated PTO assembly ignores dtype and target_isa metadata params."""
    ib = IRBuilder()

    with ib.function("test_func") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))
        y = ib.let("y", ir.op.block.muls(x, 2.0))
        ib.return_stmt(y)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    codegen = ir.PTOCodegen()
    pto_code = codegen.generate(program)

    # Verify generated code structure
    assert "func @test_func" in pto_code
    assert "%y = alloc_tile" in pto_code
    assert "return" in pto_code


def test_pto_codegen_scalar_declarations():
    """Test that scalar declarations are correctly generated."""
    ib = IRBuilder()

    with ib.function("test_scalars") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        # Add some scalar variables
        ib.let("count", ir.ConstInt(10, DataType.INT32, ir.Span.unknown()))
        ib.let("offset", ir.ConstInt(0, DataType.INT32, ir.Span.unknown()))

        # Add a tile operation
        y = ib.let("y", ir.op.block.muls(x, 2.0))
        ib.return_stmt(y)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    codegen = ir.PTOCodegen()
    pto_code = codegen.generate(program)

    # Verify scalar declarations are generated
    assert "%count = alloc_scalar : i32" in pto_code
    assert "%offset = alloc_scalar : i32" in pto_code
    # Verify tile declaration is also present
    assert "%y = alloc_tile" in pto_code


def test_pto_codegen_comparison_expressions():
    """Test that scalar comparison expressions generate CMP instructions."""
    ib = IRBuilder()

    with ib.function("test_comparisons") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        # Create scalar variables
        count = ib.let("count", ir.ConstInt(10, DataType.INT32, ir.Span.unknown()))
        threshold = ib.let("threshold", ir.ConstInt(5, DataType.INT32, ir.Span.unknown()))

        # Create comparison expressions
        is_greater = ib.let("is_greater", count >= threshold)  # GE
        ib.let("is_less", count < threshold)  # LT
        ib.let("is_equal", count == threshold)  # EQ

        # Use comparison in if statement
        with ib.if_stmt(is_greater):
            ib.let("y", ir.op.block.muls(x, 2.0))

        ib.return_stmt(x)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    codegen = ir.PTOCodegen()
    pto_code = codegen.generate(program)

    # Verify CMP instructions are generated with correct format
    assert "CMP %is_greater:u1" in pto_code
    assert "CMP %is_less:u1" in pto_code
    assert "CMP %is_equal:u1" in pto_code

    # Verify comparison operators
    assert ", GE" in pto_code  # Greater or equal
    assert ", LT" in pto_code  # Less than
    assert ", EQ" in pto_code  # Equal

    # Verify scalar declarations for comparison results
    assert "%is_greater = alloc_scalar : u1" in pto_code
    assert "%is_less = alloc_scalar : u1" in pto_code
    assert "%is_equal = alloc_scalar : u1" in pto_code
