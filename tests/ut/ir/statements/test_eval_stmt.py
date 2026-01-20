# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pypto import DataType, ir


def test_eval_stmt_creation():
    """Test creating an EvalStmt."""
    span = ir.Span("test.py", 1, 1)
    
    # Create a simple expression (e.g., a binary op)
    lhs = ir.ConstInt(1, DataType.INT32, span)
    rhs = ir.ConstInt(2, DataType.INT32, span)
    expr = ir.Add(lhs, rhs, DataType.INT32, span)
    
    # Create EvalStmt
    stmt = ir.EvalStmt(expr, span)
    
    assert stmt.expr.same_as(expr)
    assert stmt.span.filename == "test.py"


def test_eval_stmt_python_print():
    """Test printing an EvalStmt as Python code."""
    span = ir.Span("test.py", 1, 1)
    
    # Create: print(42)
    op = ir.Op("print")
    arg = ir.ConstInt(42, DataType.INT32, span)
    # We need to register the op first if we use create_op_call with deducer
    # But for raw Call creation via constructor if exposed, or we can use a known op.
    # Since we don't have a registry in this test, let's use manual Call construction if possible
    # or rely on the fact that `create_op_call` might fail if not registered.
    
    # Wait, create_op_call requires registration. Let's use a lower level approach or mock.
    # Actually, we can just use any expression, e.g. a binary op, although it's weird as a statement.
    # "1 + 2" is a valid statement in Python (evaluates and discards result).
    
    lhs = ir.ConstInt(1, DataType.INT32, span)
    rhs = ir.ConstInt(2, DataType.INT32, span)
    expr = ir.Add(lhs, rhs, DataType.INT32, span)
    
    stmt = ir.EvalStmt(expr, span)
    
    # Print
    code = ir.python_print(stmt)
    assert code.strip() == "1 + 2"


def test_eval_stmt_serialization():
    """Test serializing and deserializing an EvalStmt."""
    span = ir.Span("test.py", 1, 1)
    
    lhs = ir.ConstInt(1, DataType.INT32, span)
    rhs = ir.ConstInt(2, DataType.INT32, span)
    expr = ir.Add(lhs, rhs, DataType.INT32, span)
    
    stmt = ir.EvalStmt(expr, span)
    
    # Serialize
    data = ir.serialize(stmt)
    
    # Deserialize
    restored_stmt = ir.deserialize(data)
    
    assert isinstance(restored_stmt, ir.EvalStmt)
    assert ir.structural_equal(stmt, restored_stmt)
