# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Python IR printer with type annotations and SSA-style syntax."""

import pytest
from pypto import DataType, ir


def test_python_print_basic_expressions():
    """Test Python-style printing of basic expressions."""
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Variables should include just the name
    x = ir.Var("x", ir.ScalarType(dtype), span)
    assert "x" in ir.python_print(x)

    # Constants
    c = ir.ConstInt(42, dtype, span)
    assert "42" in ir.python_print(c)

    # Boolean constants
    b_true = ir.ConstBool(True, span)
    assert "True" in ir.python_print(b_true)
    b_false = ir.ConstBool(False, span)
    assert "False" in ir.python_print(b_false)

    # Binary operations
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)
    c = ir.ConstInt(42, dtype, span)
    add = ir.Add(ir.Add(a, b, dtype, span), c, dtype, span)
    result = ir.python_print(add)
    assert "a + b + 42" in result


def test_python_print_assignment_with_type_annotation():
    """Test assignment statements include type annotations."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    c = ir.ConstInt(42, dtype, span)
    assign = ir.AssignStmt(x, c, span)

    result = ir.python_print(assign)
    # Should have type annotation with default "pl" prefix
    assert "x:" in result or "x :" in result
    assert "pl.INT64" in result
    assert "42" in result


def test_python_print_tensor_type_annotation():
    """Test tensor type annotations."""
    span = ir.Span.unknown()
    dim1 = ir.ConstInt(64, DataType.INT32, span)
    dim2 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim1, dim2], DataType.FP32)
    a = ir.Var("a", tensor_type, span)
    b = ir.Var("b", tensor_type, span)

    # Create an assignment to see the type annotation
    assign = ir.AssignStmt(a, b, span)
    result = ir.python_print(assign)

    assert "a:" in result or "a :" in result
    assert "pl.Tensor[[64, 128], pl.FP32]" in result


def test_python_print_function_with_annotations():
    """Test function definitions include type annotations."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)

    # Simple function body
    add = ir.Add(x, y, dtype, span)
    z = ir.Var("z", ir.ScalarType(dtype), span)
    assign = ir.AssignStmt(z, add, span)
    yield_stmt = ir.YieldStmt([z], span)
    body = ir.SeqStmts([assign, yield_stmt], span)

    func = ir.Function("add_func", [x, y], [ir.ScalarType(dtype)], body, span)
    result = ir.python_print(func)

    # Check for function signature with type annotations
    assert "def add_func" in result
    assert "x:" in result or "x :" in result
    assert "y:" in result or "y :" in result
    assert "pl.INT64" in result
    assert "->" in result  # Return type annotation


def test_python_print_program():
    """Test program printing with header."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)

    assign = ir.AssignStmt(y, x, span)
    func = ir.Function("simple_func", [x], [ir.ScalarType(dtype)], assign, span)
    program = ir.Program([func], "test_program", span)

    result = ir.python_print(program)

    # Check for program header with default "pl" prefix
    assert "# pypto.program: test_program" in result
    assert "import pypto.language as pl" in result
    assert "def simple_func" in result


def test_python_print_if_stmt_basic():
    """Test basic if statement printing."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    zero = ir.ConstInt(0, dtype, span)
    condition = ir.Gt(x, zero, dtype, span)

    y = ir.Var("y", ir.ScalarType(dtype), span)
    c1 = ir.ConstInt(1, dtype, span)
    assign = ir.AssignStmt(y, c1, span)

    if_stmt = ir.IfStmt(condition, assign, None, [], span)
    result = ir.python_print(if_stmt)

    assert "if" in result
    assert "x > 0" in result or "x>0" in result


def test_python_print_for_stmt_basic():
    """Test basic for loop printing."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    i = ir.Var("i", ir.ScalarType(dtype), span)
    start = ir.ConstInt(0, dtype, span)
    stop = ir.ConstInt(10, dtype, span)
    step = ir.ConstInt(1, dtype, span)

    x = ir.Var("x", ir.ScalarType(dtype), span)
    c2 = ir.ConstInt(2, dtype, span)
    mul = ir.Mul(i, c2, dtype, span)
    assign = ir.AssignStmt(x, mul, span)

    for_stmt = ir.ForStmt(i, start, stop, step, [], assign, [], span)
    result = ir.python_print(for_stmt)

    assert "for" in result
    assert "for i in pl.range" in result  # No type annotation in for loop header
    assert "pl.INT64" in result  # Type annotation in body assignment
    assert "range" in result
    assert "0" in result
    assert "10" in result


def test_python_print_all_binary_operators():
    """Test all binary operators are printed correctly."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)

    # Arithmetic operators
    ops_and_symbols = [
        (ir.Add(a, b, dtype, span), "+"),
        (ir.Sub(a, b, dtype, span), "-"),
        (ir.Mul(a, b, dtype, span), "*"),
        (ir.FloorDiv(a, b, dtype, span), "//"),
        (ir.FloorMod(a, b, dtype, span), "%"),
        (ir.FloatDiv(a, b, dtype, span), "/"),
        (ir.Pow(a, b, dtype, span), "**"),
        # Comparison operators
        (ir.Eq(a, b, dtype, span), "=="),
        (ir.Ne(a, b, dtype, span), "!="),
        (ir.Lt(a, b, dtype, span), "<"),
        (ir.Le(a, b, dtype, span), "<="),
        (ir.Gt(a, b, dtype, span), ">"),
        (ir.Ge(a, b, dtype, span), ">="),
        # Logical operators
        (ir.And(a, b, dtype, span), "and"),
        (ir.Or(a, b, dtype, span), "or"),
        # Bitwise operators
        (ir.BitAnd(a, b, dtype, span), "&"),
        (ir.BitOr(a, b, dtype, span), "|"),
        (ir.BitXor(a, b, dtype, span), "^"),
        (ir.BitShiftLeft(a, b, dtype, span), "<<"),
        (ir.BitShiftRight(a, b, dtype, span), ">>"),
    ]

    for expr, symbol in ops_and_symbols:
        result = ir.python_print(expr)
        assert symbol in result, f"Symbol {symbol} not found in {result}"


def test_python_print_all_unary_operators():
    """Test all unary operators are printed correctly."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)

    # Negation
    neg = ir.Neg(x, dtype, span)
    result = ir.python_print(neg)
    assert "-x" in result or "- x" in result

    # Bitwise not
    bitnot = ir.BitNot(x, dtype, span)
    result = ir.python_print(bitnot)
    assert "~x" in result or "~ x" in result

    # Logical not
    not_expr = ir.Not(x, dtype, span)
    result = ir.python_print(not_expr)
    assert "not" in result

    # Abs
    abs_expr = ir.Abs(x, dtype, span)
    result = ir.python_print(abs_expr)
    assert "abs" in result


def test_python_print_min_max():
    """Test min/max function-style operators."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)

    min_expr = ir.Min(a, b, dtype, span)
    result = ir.python_print(min_expr)
    assert "min(a, b)" in result or "min( a, b )" in result or "min(a,b)" in result

    max_expr = ir.Max(a, b, dtype, span)
    result = ir.python_print(max_expr)
    assert "max(a, b)" in result or "max( a, b )" in result or "max(a,b)" in result


def test_python_print_call_expression():
    """Test function call expressions."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)

    op = ir.Op("my_op")
    call = ir.Call(op, [a, b], span)
    result = ir.python_print(call)

    assert "my_op" in result
    assert "(" in result
    assert ")" in result


def test_python_print_op_with_attributes():
    """Test Op calls with attributes are printed as keyword arguments."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)

    op = ir.Op("tensor_add")
    # Note: We can't easily set attributes from Python bindings without proper support
    # This is a basic structure test
    call = ir.Call(op, [a, b], span)
    result = ir.python_print(call)

    assert "tensor_add" in result
    assert "a" in result
    assert "b" in result


def test_python_print_yield_stmt():
    """Test yield statement printing."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)

    # Yield with no values
    yield_empty = ir.YieldStmt(span)
    result = ir.python_print(yield_empty)
    assert "yield_" in result

    # Yield with single value
    yield_single = ir.YieldStmt([x], span)
    result = ir.python_print(yield_single)
    assert "yield_" in result
    assert "x" in result

    # Yield with multiple values
    yield_multi = ir.YieldStmt([x, y], span)
    result = ir.python_print(yield_multi)
    assert "yield_" in result
    assert "x" in result
    assert "y" in result


def test_python_print_seq_stmts():
    """Test sequence of statements."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    z = ir.Var("z", ir.ScalarType(dtype), span)

    c1 = ir.ConstInt(1, dtype, span)
    c2 = ir.ConstInt(2, dtype, span)

    assign1 = ir.AssignStmt(x, c1, span)
    assign2 = ir.AssignStmt(y, c2, span)
    add = ir.Add(x, y, dtype, span)
    assign3 = ir.AssignStmt(z, add, span)

    seq = ir.SeqStmts([assign1, assign2, assign3], span)
    result = ir.python_print(seq)

    # All assignments should be present
    assert "x:" in result
    assert "y:" in result
    assert "z:" in result


def test_python_print_op_stmts():
    """Test OpStmts (sequence of assignments)."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)

    c1 = ir.ConstInt(1, dtype, span)
    c2 = ir.ConstInt(2, dtype, span)

    assign1 = ir.AssignStmt(x, c1, span)
    assign2 = ir.AssignStmt(y, c2, span)

    op_stmts = ir.OpStmts([assign1, assign2], span)
    result = ir.python_print(op_stmts)

    assert "x:" in result
    assert "y:" in result


def test_python_print_tile_type():
    """Test tile type annotations."""
    span = ir.Span.unknown()
    dim1 = ir.ConstInt(16, DataType.INT32, span)
    dim2 = ir.ConstInt(16, DataType.INT32, span)
    tile_type = ir.TileType([dim1, dim2], DataType.FP16)
    t = ir.Var("t", tile_type, span)

    assign = ir.AssignStmt(t, t, span)
    result = ir.python_print(assign)

    assert "t:" in result
    assert "pl.Tile[[16, 16], pl.FP16]" in result


def test_python_print_all_scalar_types():
    """Test all scalar type annotations."""
    span = ir.Span.unknown()

    type_map = [
        (DataType.INT8, "pl.INT8"),
        (DataType.INT16, "pl.INT16"),
        (DataType.INT32, "pl.INT32"),
        (DataType.INT64, "pl.INT64"),
        (DataType.UINT8, "pl.UINT8"),
        (DataType.UINT16, "pl.UINT16"),
        (DataType.UINT32, "pl.UINT32"),
        (DataType.UINT64, "pl.UINT64"),
        (DataType.FP16, "pl.FP16"),
        (DataType.FP32, "pl.FP32"),
        (DataType.BF16, "pl.BFLOAT16"),
    ]

    for dtype, expected_str in type_map:
        x = ir.Var("x", ir.ScalarType(dtype), span)
        c = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(x, c, span)
        result = ir.python_print(assign)
        assert expected_str in result, f"Expected {expected_str} in output for {dtype}"


def test_python_print_complex_nested_function():
    """Test complex function with nested control flow."""
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Parameters
    n = ir.Var("n", ir.ScalarType(dtype), span)

    # Initialize sum
    sum_var = ir.Var("sum", ir.ScalarType(dtype), span)
    zero = ir.ConstInt(0, dtype, span)
    init_sum = ir.AssignStmt(sum_var, zero, span)

    # Loop variable
    i = ir.Var("i", ir.ScalarType(dtype), span)
    start = ir.ConstInt(0, dtype, span)
    step = ir.ConstInt(1, dtype, span)

    # Loop body: sum = sum + i
    sum_copy = ir.Var("sum", ir.ScalarType(dtype), span)
    add_expr = ir.Add(sum_copy, i, dtype, span)
    update_sum = ir.AssignStmt(sum_var, add_expr, span)

    # For loop
    for_stmt = ir.ForStmt(i, start, n, step, [], update_sum, [], span)

    # Yield sum
    yield_stmt = ir.YieldStmt([sum_var], span)

    # Function body
    body = ir.SeqStmts([init_sum, for_stmt, yield_stmt], span)
    func = ir.Function("loop_sum", [n], [ir.ScalarType(dtype)], body, span)

    result = ir.python_print(func)

    # Verify structure
    assert "def loop_sum" in result
    assert "n:" in result
    assert "pl.INT64" in result
    assert "for" in result
    assert "range" in result
    assert "return" in result  # Functions use return, not yield


def test_python_print_program_with_multiple_functions():
    """Test program with multiple functions."""
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Function 1
    x1 = ir.Var("x", ir.ScalarType(dtype), span)
    y1 = ir.Var("y", ir.ScalarType(dtype), span)
    assign1 = ir.AssignStmt(y1, x1, span)
    func1 = ir.Function("func1", [x1], [ir.ScalarType(dtype)], assign1, span)

    # Function 2
    x2 = ir.Var("a", ir.ScalarType(dtype), span)
    y2 = ir.Var("b", ir.ScalarType(dtype), span)
    assign2 = ir.AssignStmt(y2, x2, span)
    func2 = ir.Function("func2", [x2], [ir.ScalarType(dtype)], assign2, span)

    program = ir.Program([func1, func2], "multi_func_program", span)
    result = ir.python_print(program)

    # Check program structure with default "pl" prefix
    assert "# pypto.program: multi_func_program" in result
    assert "import pypto.language as pl" in result
    assert "def func1" in result
    assert "def func2" in result


def test_python_print_str_method():
    """Test that str() uses the Python printer."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    c = ir.ConstInt(42, dtype, span)
    assign = ir.AssignStmt(x, c, span)

    # str() should use Python printer with default "pl" prefix
    str_result = str(assign)
    # Should include type annotation
    assert "pl.INT64" in str_result


def test_python_print_custom_prefix():
    """Test configurable prefix for type annotations."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    c = ir.ConstInt(42, dtype, span)
    assign = ir.AssignStmt(x, c, span)

    # Test default prefix "pl"
    result_pi = ir.python_print(assign)
    assert "pl.INT64" in result_pi

    # Test "ir" prefix
    result_ir = ir.python_print(assign, "ir")
    assert "ir.INT64" in result_ir

    # Test custom prefix
    result_custom = ir.python_print(assign, "myir")
    assert "myir.INT64" in result_custom

    # Test with program to check import statement
    func = ir.Function("test", [x], [ir.ScalarType(dtype)], assign, span)
    program = ir.Program([func], "test_prog", span)

    # Default "pl" should use "import pypto.language as pl"
    prog_pi = ir.python_print(program)
    assert "import pypto.language as pl" in prog_pi
    assert "pl.INT64" in prog_pi

    # "ir" prefix should use "from pypto import ir"
    prog_ir = ir.python_print(program, "language")
    assert "from pypto import language" in prog_ir
    assert "language.INT64" in prog_ir

    # Custom prefix should use "import pypto.ir as <prefix>"
    prog_custom = ir.python_print(program, "custom")
    assert "from pypto import language as custom" in prog_custom
    assert "custom.INT64" in prog_custom


def test_python_print_block_load_store():
    """Test printing of block.load and block.store operations with tuple arguments."""
    span = ir.Span.unknown()

    # Create tensor and tile types
    dim1 = ir.ConstInt(64, DataType.INT32, span)
    dim2 = ir.ConstInt(64, DataType.INT32, span)
    tensor_type = ir.TensorType([dim1, dim2], DataType.FP32)
    tile_type = ir.TileType([dim1, dim2], DataType.FP32)

    # Create variables
    input_tensor = ir.Var("input_tensor", tensor_type, span)
    output_tensor = ir.Var("output_tensor", tensor_type, span)
    tile = ir.Var("tile", tile_type, span)

    # Create offset and shape tuples
    zero = ir.ConstInt(0, DataType.INT32, span)
    size = ir.ConstInt(64, DataType.INT32, span)
    offsets_tuple = ir.MakeTuple([zero, zero], span)
    shapes_tuple = ir.MakeTuple([size, size], span)

    # Test block.load
    load_op = ir.Op("block.load")
    load_call = ir.Call(load_op, [input_tensor, offsets_tuple, shapes_tuple], span)

    load_result = ir.python_print(load_call)
    print("\nblock.load output:")
    print(load_result)

    # Should contain operation name
    assert "pl.block.load" in load_result
    # Should contain tensor name
    assert "input_tensor" in load_result
    # Should contain tuple representation of offsets
    assert "[0, 0]" in load_result
    # Should contain tuple representation of shapes
    assert "[64, 64]" in load_result

    # Test block.store
    store_op = ir.Op("block.store")
    store_call = ir.Call(store_op, [tile, offsets_tuple, shapes_tuple, output_tensor], span)

    store_result = ir.python_print(store_call)
    print("\nblock.store output:")
    print(store_result)

    # Should contain operation name
    assert "pl.block.store" in store_result
    # Should contain tile name
    assert "tile" in store_result
    # Should contain tuple representation
    assert "[0, 0]" in store_result
    assert "[64, 64]" in store_result
    # Should contain output tensor
    assert "output_tensor" in store_result

    # Test with target_memory kwarg
    # Correct signature: Call(op, args, kwargs, span)
    load_call_with_kwargs = ir.Call(
        load_op, [input_tensor, offsets_tuple, shapes_tuple], {"target_memory": 1}, span
    )

    load_kwargs_result = ir.python_print(load_call_with_kwargs)
    print("\nblock.load with kwargs output:")
    print(load_kwargs_result)

    assert "pl.block.load" in load_kwargs_result
    assert "target_memory=1" in load_kwargs_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
