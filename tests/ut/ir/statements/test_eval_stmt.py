# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pytest
from pypto import ir


def test_eval_stmt_creation():
    """Test creating an EvalStmt using a registered system op."""
    span = ir.Span("test.py", 1, 1)

    # Create system.bar_all() which takes no arguments but has an optional pipe attribute
    # Here we create it without attributes first
    call = ir.create_op_call("system.bar_all", [], span)

    stmt = ir.EvalStmt(call, span)

    assert stmt.expr.same_as(call)
    assert stmt.span.filename == "test.py"


def test_eval_stmt_python_print():
    """Test printing an EvalStmt as Python code using system.sync_src."""
    span = ir.Span("test.py", 1, 1)

    # Create system.sync_src(set_pipe=MTE2, wait_pipe=V, event_id=1)
    sync_call = ir.create_op_call(
        "system.sync_src", [], {"set_pipe": ir.PipeType.MTE2, "wait_pipe": ir.PipeType.V, "event_id": 1}, span
    )

    stmt = ir.EvalStmt(sync_call, span)

    # Print
    code = ir.python_print(stmt)
    assert "system.sync_src(" in code
    assert "set_pipe=pl.PipeType.MTE2" in code
    assert "wait_pipe=pl.PipeType.V" in code
    assert "event_id=1" in code
    assert "(, " not in code  # no stray comma for no-argument ops


def test_eval_stmt_serialization():
    """Test serializing and deserializing an EvalStmt."""
    span = ir.Span("test.py", 1, 1)

    # Create system.bar_v()
    call = ir.create_op_call("system.bar_v", [], {}, span)

    stmt = ir.EvalStmt(call, span)

    # Serialize
    data = ir.serialize(stmt)

    # Deserialize
    restored_stmt = ir.deserialize(data)

    assert isinstance(restored_stmt, ir.EvalStmt)
    ir.assert_structural_equal(stmt, restored_stmt)


def test_sync_ops_enum_usage():
    """Test that PipeType and CoreType enums can be used correctly."""
    # This test verifies the binding of enums
    assert ir.PipeType.MTE2 is not None
    assert ir.PipeType.V is not None
    assert ir.CoreType.VECTOR is not None
    assert ir.CoreType.CUBE is not None

    # Verify we can pass them to create_op_call
    span = ir.Span("test.py", 1, 1)
    # system.bar_all has no attributes now, so we pass empty dict.
    # But to test enum passing, let's use system.sync_src
    ir.create_op_call(
        "system.sync_src", [], {"set_pipe": ir.PipeType.MTE2, "wait_pipe": ir.PipeType.V, "event_id": 0}, span
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
