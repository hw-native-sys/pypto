# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for PTOCodegen - PTO assembly generation from PyPTO IR."""

import pytest
from pypto import DataType, ir
from pypto.ir import IRBuilder
from pypto.pypto_core import codegen as codegen_mod


def _get_pto_content(files):
    """Return combined PTO content from generate() result (dict) for assertions."""
    if isinstance(files, str):
        return files
    return "".join(files.values())


def test_pto_codegen_basic():
    """Test basic PTOCodegen functionality - generates PTO assembly."""
    # Create a simple program
    ib = IRBuilder()

    with ib.function("test_func", type=ir.FunctionType.InCore) as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        result = ib.let("result", ir.op.block.muls(x, 2.0))
        ib.return_stmt(result)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    # Apply codegen
    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)

    # Verify the result is a dict mapping file paths to content
    assert isinstance(files, dict)
    assert len(files) >= 1
    pto_code = _get_pto_content(files)

    # Verify generated code contains expected elements
    assert "func @test_func" in pto_code
    assert "alloc_tile" in pto_code
    assert "tmuls" in pto_code
    assert "return" in pto_code


def test_pto_codegen_tile_declarations():
    """Test that tile declarations are correctly generated."""
    ib = IRBuilder()

    with ib.function("test_tiles", type=ir.FunctionType.InCore) as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([16, 32], DataType.FP16))

        y = ib.let("y", ir.op.block.mul(x, x))  # 8x8 tile
        z = ib.let("z", ir.op.block.adds(y, 1.0))  # 8x8 tile
        ib.return_stmt(z)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)
    pto_code = _get_pto_content(files)

    # Verify tile declarations
    assert "%y = alloc_tile" in pto_code
    assert "%z = alloc_tile" in pto_code


def test_pto_codegen_for_loop():
    """Test that ForStmt is correctly converted to FOR/ENDFOR."""
    ib = IRBuilder()

    with ib.function("test_loop", type=ir.FunctionType.InCore) as f:
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

    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)
    pto_code = _get_pto_content(files)

    # Verify for loop structure
    assert "FOR %i:" in pto_code
    assert "ENDFOR" in pto_code


def test_pto_codegen_scalar_operations():
    """Test that scalar operations use correct PTO instructions."""
    ib = IRBuilder()

    with ib.function("test_scalar_ops", type=ir.FunctionType.InCore) as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        y = ib.let("y", ir.op.block.muls(x, 2.0))
        z = ib.let("z", ir.op.block.divs(y, 3.0))
        w = ib.let("w", ir.op.block.adds(z, 1.0))
        ib.return_stmt(w)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)
    pto_code = _get_pto_content(files)

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

    with ib.function("test_binary_ops", type=ir.FunctionType.InCore) as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        y = f.param("y", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        z1 = ib.let("z1", ir.op.block.mul(x, y))
        z2 = ib.let("z2", ir.op.block.add(z1, x))
        z3 = ib.let("z3", ir.op.block.sub(z2, y))
        ib.return_stmt(z3)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)
    pto_code = _get_pto_content(files)

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

    with ib.function("test_types", type=ir.FunctionType.InCore) as f:
        x_fp32 = f.param("x_fp32", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        y_fp16 = ib.let("y_fp16", ir.op.block.mul(x_fp32, x_fp32))
        ib.return_stmt(y_fp16)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)
    pto_code = _get_pto_content(files)

    # Verify PTO type conversions
    assert "f32" in pto_code


def test_pto_codegen_multiple_functions():
    """Test PTOCodegen with multiple functions."""
    # Create program with two functions
    ib1 = IRBuilder()
    with ib1.function("func1", type=ir.FunctionType.InCore) as f1:
        x = f1.param("x", ir.TileType([8, 8], DataType.FP32))
        f1.return_type(ir.TileType([8, 8], DataType.FP32))
        result = ib1.let("result", ir.op.block.muls(x, 2.0))
        ib1.return_stmt(result)
    func1 = f1.get_result()

    ib2 = IRBuilder()
    with ib2.function("func2", type=ir.FunctionType.InCore) as f2:
        y = f2.param("y", ir.TileType([8, 8], DataType.FP32))
        f2.return_type(ir.TileType([8, 8], DataType.FP32))
        result = ib2.let("result", ir.op.block.adds(y, 1.0))
        ib2.return_stmt(result)
    func2 = f2.get_result()

    program = ir.Program([func1, func2], "multi_func_program", ir.Span.unknown())

    # Apply codegen
    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)
    pto_code = _get_pto_content(files)

    # Verify both functions are generated
    assert "func @func1" in pto_code
    assert "func @func2" in pto_code


def test_pto_codegen_reusability():
    """Test that the same PTOCodegen instance can be used multiple times."""
    ib = IRBuilder()

    with ib.function("test_func", type=ir.FunctionType.InCore) as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))
        y = ib.let("y", ir.op.block.muls(x, 2.0))
        ib.return_stmt(y)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    # Use the same codegen instance multiple times
    codegen = codegen_mod.PTOCodegen()

    code1 = codegen.generate(program)
    code2 = codegen.generate(program)

    # Verify both calls produce valid code (dict of path -> content)
    assert isinstance(code1, dict)
    assert isinstance(code2, dict)
    assert "func @test_func" in _get_pto_content(code1)
    assert "func @test_func" in _get_pto_content(code2)


def test_pto_codegen_with_dtype_target_isa():
    """Test that generated PTO assembly ignores dtype and target_isa metadata params."""
    ib = IRBuilder()

    with ib.function("test_func", type=ir.FunctionType.InCore) as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))
        y = ib.let("y", ir.op.block.muls(x, 2.0))
        ib.return_stmt(y)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)
    pto_code = _get_pto_content(files)

    # Verify generated code structure
    assert "func @test_func" in pto_code
    assert "%y = alloc_tile" in pto_code
    assert "return" in pto_code


def test_pto_codegen_scalar_declarations():
    """Test that scalar declarations are correctly generated."""
    ib = IRBuilder()

    with ib.function("test_scalars", type=ir.FunctionType.InCore) as f:
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

    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)
    pto_code = _get_pto_content(files)

    # Verify scalar declarations are generated
    assert "%count = alloc_scalar : i32" in pto_code
    assert "%offset = alloc_scalar : i32" in pto_code
    # Verify tile declaration is also present
    assert "%y = alloc_tile" in pto_code


def test_pto_codegen_comparison_expressions():
    """Test that scalar comparison expressions generate CMP instructions."""
    ib = IRBuilder()

    with ib.function("test_comparisons", type=ir.FunctionType.InCore) as f:
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

    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)
    pto_code = _get_pto_content(files)

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


def test_pto_codegen_simple_orchestration():
    """Test orchestration code generation with a simple linear task graph: c = add(a,b), result = mul(c,a)."""
    ib = IRBuilder()

    # Step 1: Build kernel functions (InCore)
    with ib.function("kernel_add", type=ir.FunctionType.InCore) as f_add:
        a = f_add.param("a", ir.TileType([16, 16], DataType.FP32))
        b = f_add.param("b", ir.TileType([16, 16], DataType.FP32))
        f_add.return_type(ir.TileType([16, 16], DataType.FP32))
        result = ib.let("result", ir.op.block.add(a, b))
        ib.return_stmt(result)
    kernel_add = f_add.get_result()

    with ib.function("kernel_mul", type=ir.FunctionType.InCore) as f_mul:
        a = f_mul.param("a", ir.TileType([16, 16], DataType.FP32))
        b = f_mul.param("b", ir.TileType([16, 16], DataType.FP32))
        f_mul.return_type(ir.TileType([16, 16], DataType.FP32))
        result = ib.let("result", ir.op.block.mul(a, b))
        ib.return_stmt(result)
    kernel_mul = f_mul.get_result()

    # Step 2: Build orchestration function
    with ib.function("simple_orch", type=ir.FunctionType.Orchestration) as f:
        a = f.param("a", ir.TensorType([128], DataType.FP32))
        b = f.param("b", ir.TensorType([128], DataType.FP32))
        _d = f.param("d", ir.TensorType([128], DataType.FP32))  # extra param, not used in task graph
        f.return_type(ir.TensorType([128], DataType.FP32))

        tensor_128_fp32 = ir.TensorType([128], DataType.FP32)
        add_op = ir.GlobalVar("kernel_add")
        kwargs = {"func_id": 0, "device_type": 1}
        add_call = ir.Call(add_op, [a, b], kwargs, tensor_128_fp32, ir.Span.unknown())
        c = ib.let("c", add_call)

        mul_op = ir.GlobalVar("kernel_mul")
        kwargs = {"func_id": 1, "device_type": 1}
        mul_call = ir.Call(mul_op, [c, a], kwargs, tensor_128_fp32, ir.Span.unknown())
        result = ib.let("result", mul_call)

        ib.return_stmt(result)

    func = f.get_result()

    # Step 3: Create Program with ALL functions (kernels + orchestration)
    program = ir.Program([kernel_add, kernel_mul, func], "simple_orch", ir.Span.unknown())

    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)

    assert isinstance(files, dict)
    # Program with kernels: should have both .pto files and .cpp file
    assert "orchestration/simple_orch.cpp" in files
    cpp_content = files["orchestration/simple_orch.cpp"]

    assert "// Orchestration Function: simple_orch" in cpp_content
    assert "// Generated by PyPTO IR Compiler" in cpp_content
    # Build function name is Build + capitalized func name
    assert "BuildSimple_orch" in cpp_content
    # Task kernels referenced in generated code
    assert "kernel_add" in cpp_content
    assert "kernel_mul" in cpp_content

    # Verify data-flow based dependencies
    # c = add(a, b)  -> Task t0 produces c
    # result = mul(c, a) -> Task t1 consumes c (depends on t0)
    # Expected: t0 -> t1 dependency
    assert "// Dependencies (data-flow based)" in cpp_content
    assert "runtime->add_successor(t0, t1)" in cpp_content


def test_pto_codegen_orchestration_missing_function():
    """Test that orchestration fails when referenced function is missing from the Program."""
    ib = IRBuilder()

    # Create orchestration function that references non-existent kernel
    with ib.function("simple_orch", type=ir.FunctionType.Orchestration) as f:
        a = f.param("a", ir.TensorType([128], DataType.FP32))
        b = f.param("b", ir.TensorType([128], DataType.FP32))
        f.return_type(ir.TensorType([128], DataType.FP32))

        # Call non-existent kernel
        add_op = ir.GlobalVar("kernel_add")  # This function doesn't exist in the Program
        tensor_128_fp32 = ir.TensorType([128], DataType.FP32)
        kwargs = {"func_id": 0, "device_type": 1}
        add_call = ir.Call(add_op, [a, b], kwargs, tensor_128_fp32, ir.Span.unknown())
        c = ib.let("c", add_call)
        ib.return_stmt(c)

    func = f.get_result()

    # Program does NOT contain kernel_add
    program = ir.Program([func], "simple_orch", ir.Span.unknown())

    codegen = codegen_mod.PTOCodegen()

    # Should raise ValueError with clear error message
    with pytest.raises(ValueError, match=r"references undefined functions[\s\S]*kernel_add"):
        codegen.generate(program)


def test_pto_codegen_parallel_tasks():
    """Test data-flow dependency analysis with independent parallel tasks."""
    ib = IRBuilder()

    # Build kernel functions
    with ib.function("kernel_add", type=ir.FunctionType.InCore) as f_add:
        a = f_add.param("a", ir.TileType([16, 16], DataType.FP32))
        b = f_add.param("b", ir.TileType([16, 16], DataType.FP32))
        f_add.return_type(ir.TileType([16, 16], DataType.FP32))
        result = ib.let("result", ir.op.block.add(a, b))
        ib.return_stmt(result)
    kernel_add = f_add.get_result()

    with ib.function("kernel_sub", type=ir.FunctionType.InCore) as f_sub:
        a = f_sub.param("a", ir.TileType([16, 16], DataType.FP32))
        b = f_sub.param("b", ir.TileType([16, 16], DataType.FP32))
        f_sub.return_type(ir.TileType([16, 16], DataType.FP32))
        result = ib.let("result", ir.op.block.sub(a, b))
        ib.return_stmt(result)
    kernel_sub = f_sub.get_result()

    # Build orchestration with independent tasks
    with ib.function("parallel_orch", type=ir.FunctionType.Orchestration) as f:
        a = f.param("a", ir.TensorType([128], DataType.FP32))
        b = f.param("b", ir.TensorType([128], DataType.FP32))
        f.return_type(ir.TensorType([128], DataType.FP32))

        tensor_128_fp32 = ir.TensorType([128], DataType.FP32)

        # Task t0: c = add(a, b) - produces c
        add_op = ir.GlobalVar("kernel_add")
        kwargs = {"func_id": 0, "device_type": 1}
        add_call = ir.Call(add_op, [a, b], kwargs, tensor_128_fp32, ir.Span.unknown())
        c = ib.let("c", add_call)

        # Task t1: d = sub(a, b) - produces d (independent of t0, uses only inputs a, b)
        sub_op = ir.GlobalVar("kernel_sub")
        kwargs = {"func_id": 1, "device_type": 1}
        sub_call = ir.Call(sub_op, [a, b], kwargs, tensor_128_fp32, ir.Span.unknown())
        _d = ib.let("d", sub_call)  # Intentionally unused - testing parallel task creation

        ib.return_stmt(c)

    func = f.get_result()
    program = ir.Program([kernel_add, kernel_sub, func], "parallel_orch", ir.Span.unknown())

    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)
    cpp_content = files["orchestration/parallel_orch.cpp"]

    # Verify tasks are created
    assert "int t0 = runtime->add_task" in cpp_content
    assert "int t1 = runtime->add_task" in cpp_content

    # Verify NO dependency between t0 and t1 (they are independent)
    # Both tasks use only function inputs (a, b), not outputs from each other
    assert "runtime->add_successor(t0, t1)" not in cpp_content
    assert "runtime->add_successor(t1, t0)" not in cpp_content


def test_pto_codegen_diamond_dependencies():
    """Test data-flow dependency analysis with diamond pattern (convergent data flow)."""
    ib = IRBuilder()

    # Build kernel functions
    with ib.function("kernel_add", type=ir.FunctionType.InCore) as f_add:
        a = f_add.param("a", ir.TileType([16, 16], DataType.FP32))
        b = f_add.param("b", ir.TileType([16, 16], DataType.FP32))
        f_add.return_type(ir.TileType([16, 16], DataType.FP32))
        result = ib.let("result", ir.op.block.add(a, b))
        ib.return_stmt(result)
    kernel_add = f_add.get_result()

    with ib.function("kernel_mul", type=ir.FunctionType.InCore) as f_mul:
        a = f_mul.param("a", ir.TileType([16, 16], DataType.FP32))
        b = f_mul.param("b", ir.TileType([16, 16], DataType.FP32))
        f_mul.return_type(ir.TileType([16, 16], DataType.FP32))
        result = ib.let("result", ir.op.block.mul(a, b))
        ib.return_stmt(result)
    kernel_mul = f_mul.get_result()

    with ib.function("kernel_sub", type=ir.FunctionType.InCore) as f_sub:
        a = f_sub.param("a", ir.TileType([16, 16], DataType.FP32))
        b = f_sub.param("b", ir.TileType([16, 16], DataType.FP32))
        f_sub.return_type(ir.TileType([16, 16], DataType.FP32))
        result = ib.let("result", ir.op.block.sub(a, b))
        ib.return_stmt(result)
    kernel_sub = f_sub.get_result()

    # Build orchestration with diamond dependency pattern
    with ib.function("diamond_orch", type=ir.FunctionType.Orchestration) as f:
        a = f.param("a", ir.TensorType([128], DataType.FP32))
        b = f.param("b", ir.TensorType([128], DataType.FP32))
        f.return_type(ir.TensorType([128], DataType.FP32))

        tensor_128_fp32 = ir.TensorType([128], DataType.FP32)

        # Task t0: c = add(a, b) - produces c
        add_op = ir.GlobalVar("kernel_add")
        kwargs = {"func_id": 0, "device_type": 1}
        add_call = ir.Call(add_op, [a, b], kwargs, tensor_128_fp32, ir.Span.unknown())
        c = ib.let("c", add_call)

        # Task t1: d = mul(c, a) - consumes c from t0
        mul_op = ir.GlobalVar("kernel_mul")
        kwargs = {"func_id": 1, "device_type": 1}
        mul_call = ir.Call(mul_op, [c, a], kwargs, tensor_128_fp32, ir.Span.unknown())
        d = ib.let("d", mul_call)

        # Task t2: e = mul(c, b) - also consumes c from t0
        kwargs = {"func_id": 1, "device_type": 1}
        mul_call2 = ir.Call(mul_op, [c, b], kwargs, tensor_128_fp32, ir.Span.unknown())
        e = ib.let("e", mul_call2)

        # Task t3: result = sub(d, e) - consumes both d (from t1) and e (from t2)
        sub_op = ir.GlobalVar("kernel_sub")
        kwargs = {"func_id": 2, "device_type": 1}
        sub_call = ir.Call(sub_op, [d, e], kwargs, tensor_128_fp32, ir.Span.unknown())
        result = ib.let("result", sub_call)

        ib.return_stmt(result)

    func = f.get_result()
    program = ir.Program([kernel_add, kernel_mul, kernel_sub, func], "diamond_orch", ir.Span.unknown())

    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)
    cpp_content = files["orchestration/diamond_orch.cpp"]

    # Verify all dependencies in diamond pattern
    # Expected: t0 -> t1, t0 -> t2, t1 -> t3, t2 -> t3
    assert "runtime->add_successor(t0, t1)" in cpp_content  # c flows to d
    assert "runtime->add_successor(t0, t2)" in cpp_content  # c flows to e
    assert "runtime->add_successor(t1, t3)" in cpp_content  # d flows to result
    assert "runtime->add_successor(t2, t3)" in cpp_content  # e flows to result

    # t1 and t2 should be independent of each other
    assert "runtime->add_successor(t1, t2)" not in cpp_content
    assert "runtime->add_successor(t2, t1)" not in cpp_content


def test_pto_codegen_chain_dependencies():
    """Test data-flow dependency analysis with sequential chain of tasks."""
    ib = IRBuilder()

    # Build kernel functions
    with ib.function("kernel_add", type=ir.FunctionType.InCore) as f_add:
        a = f_add.param("a", ir.TileType([16, 16], DataType.FP32))
        b = f_add.param("b", ir.TileType([16, 16], DataType.FP32))
        f_add.return_type(ir.TileType([16, 16], DataType.FP32))
        result = ib.let("result", ir.op.block.add(a, b))
        ib.return_stmt(result)
    kernel_add = f_add.get_result()

    with ib.function("kernel_mul", type=ir.FunctionType.InCore) as f_mul:
        a = f_mul.param("a", ir.TileType([16, 16], DataType.FP32))
        b = f_mul.param("b", ir.TileType([16, 16], DataType.FP32))
        f_mul.return_type(ir.TileType([16, 16], DataType.FP32))
        result = ib.let("result", ir.op.block.mul(a, b))
        ib.return_stmt(result)
    kernel_mul = f_mul.get_result()

    with ib.function("kernel_sub", type=ir.FunctionType.InCore) as f_sub:
        a = f_sub.param("a", ir.TileType([16, 16], DataType.FP32))
        b = f_sub.param("b", ir.TileType([16, 16], DataType.FP32))
        f_sub.return_type(ir.TileType([16, 16], DataType.FP32))
        result = ib.let("result", ir.op.block.sub(a, b))
        ib.return_stmt(result)
    kernel_sub = f_sub.get_result()

    # Build orchestration with sequential chain
    with ib.function("chain_orch", type=ir.FunctionType.Orchestration) as f:
        a = f.param("a", ir.TensorType([128], DataType.FP32))
        b = f.param("b", ir.TensorType([128], DataType.FP32))
        f.return_type(ir.TensorType([128], DataType.FP32))

        tensor_128_fp32 = ir.TensorType([128], DataType.FP32)

        # Task t0: c = add(a, b) - produces c
        add_op = ir.GlobalVar("kernel_add")
        kwargs = {"func_id": 0, "device_type": 1}
        add_call = ir.Call(add_op, [a, b], kwargs, tensor_128_fp32, ir.Span.unknown())
        c = ib.let("c", add_call)

        # Task t1: d = mul(c, b) - consumes c from t0
        mul_op = ir.GlobalVar("kernel_mul")
        kwargs = {"func_id": 1, "device_type": 1}
        mul_call = ir.Call(mul_op, [c, b], kwargs, tensor_128_fp32, ir.Span.unknown())
        d = ib.let("d", mul_call)

        # Task t2: result = sub(d, a) - consumes d from t1
        sub_op = ir.GlobalVar("kernel_sub")
        kwargs = {"func_id": 2, "device_type": 1}
        sub_call = ir.Call(sub_op, [d, a], kwargs, tensor_128_fp32, ir.Span.unknown())
        result = ib.let("result", sub_call)

        ib.return_stmt(result)

    func = f.get_result()
    program = ir.Program([kernel_add, kernel_mul, kernel_sub, func], "chain_orch", ir.Span.unknown())

    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)
    cpp_content = files["orchestration/chain_orch.cpp"]

    # Verify linear chain of dependencies
    # Expected: t0 -> t1 -> t2 (sequential dependency chain)
    assert "runtime->add_successor(t0, t1)" in cpp_content  # c flows to d
    assert "runtime->add_successor(t1, t2)" in cpp_content  # d flows to result

    # No direct dependency from t0 to t2 (only transitive through t1)
    assert "runtime->add_successor(t0, t2)" not in cpp_content


def test_pto_codegen_multiple_producers():
    """Test data-flow dependency analysis with task depending on multiple producers."""
    ib = IRBuilder()

    # Build kernel functions
    with ib.function("kernel_add", type=ir.FunctionType.InCore) as f_add:
        a = f_add.param("a", ir.TileType([16, 16], DataType.FP32))
        b = f_add.param("b", ir.TileType([16, 16], DataType.FP32))
        f_add.return_type(ir.TileType([16, 16], DataType.FP32))
        result = ib.let("result", ir.op.block.add(a, b))
        ib.return_stmt(result)
    kernel_add = f_add.get_result()

    with ib.function("kernel_mul", type=ir.FunctionType.InCore) as f_mul:
        a = f_mul.param("a", ir.TileType([16, 16], DataType.FP32))
        b = f_mul.param("b", ir.TileType([16, 16], DataType.FP32))
        f_mul.return_type(ir.TileType([16, 16], DataType.FP32))
        result = ib.let("result", ir.op.block.mul(a, b))
        ib.return_stmt(result)
    kernel_mul = f_mul.get_result()

    # Build orchestration with multiple independent producers
    with ib.function("multi_producer_orch", type=ir.FunctionType.Orchestration) as f:
        a = f.param("a", ir.TensorType([128], DataType.FP32))
        b = f.param("b", ir.TensorType([128], DataType.FP32))
        f.return_type(ir.TensorType([128], DataType.FP32))

        tensor_128_fp32 = ir.TensorType([128], DataType.FP32)

        # Task t0: c = add(a, b) - produces c
        add_op = ir.GlobalVar("kernel_add")
        kwargs = {"func_id": 0, "device_type": 1}
        add_call = ir.Call(add_op, [a, b], kwargs, tensor_128_fp32, ir.Span.unknown())
        c = ib.let("c", add_call)

        # Task t1: d = mul(a, b) - produces d (independent of t0)
        mul_op = ir.GlobalVar("kernel_mul")
        kwargs = {"func_id": 1, "device_type": 1}
        mul_call = ir.Call(mul_op, [a, b], kwargs, tensor_128_fp32, ir.Span.unknown())
        d = ib.let("d", mul_call)

        # Task t2: result = add(c, d) - consumes both c (from t0) and d (from t1)
        kwargs = {"func_id": 0, "device_type": 1}
        add_call2 = ir.Call(add_op, [c, d], kwargs, tensor_128_fp32, ir.Span.unknown())
        result = ib.let("result", add_call2)

        ib.return_stmt(result)

    func = f.get_result()
    program = ir.Program([kernel_add, kernel_mul, func], "multi_producer_orch", ir.Span.unknown())

    codegen = codegen_mod.PTOCodegen()
    files = codegen.generate(program)
    cpp_content = files["orchestration/multi_producer_orch.cpp"]

    # Verify dependencies to task with multiple producers
    # Expected: t0 -> t2 (c flows to result), t1 -> t2 (d flows to result)
    assert "runtime->add_successor(t0, t2)" in cpp_content
    assert "runtime->add_successor(t1, t2)" in cpp_content

    # t0 and t1 should be independent of each other
    assert "runtime->add_successor(t0, t1)" not in cpp_content
    assert "runtime->add_successor(t1, t0)" not in cpp_content
