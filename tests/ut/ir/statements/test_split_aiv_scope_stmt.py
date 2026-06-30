# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the SplitAivScopeStmt IR node."""

import pypto
import pytest
from pypto import DataType, ir


def _make_body(span: ir.Span, var_x: ir.Var, var_y: ir.Var) -> ir.Stmt:
    return ir.AssignStmt(var_y, var_x, span)


def _vars(span: ir.Span) -> tuple[ir.Var, ir.Var]:
    var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
    var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)
    return var_x, var_y


def test_construct():
    """SplitAivScopeStmt exposes split/count fields and reports ScopeKind.SplitAiv."""
    span = ir.Span("test.py", 1, 1, 1, 10)
    body = _make_body(span, *_vars(span))
    scope = ir.SplitAivScopeStmt(split=ir.SplitMode.UP_DOWN, body=body, span=span)

    assert isinstance(scope, ir.ScopeStmt)
    assert scope.scope_kind == ir.ScopeKind.SplitAiv
    assert scope.split == ir.SplitMode.UP_DOWN
    assert scope.count == 2
    assert isinstance(scope.body, ir.AssignStmt)


def test_construct_accepts_none_mode():
    """The ctor accepts SplitMode.NONE — a task-parallel (no-halve) AIV region.

    Both AIV lanes run the full body, dispatched by aiv_id; count stays 2.
    """
    span = ir.Span("test.py", 1, 1, 1, 10)
    body = _make_body(span, *_vars(span))
    scope = ir.SplitAivScopeStmt(split=ir.SplitMode.NONE, body=body, span=span)
    assert scope.split == ir.SplitMode.NONE
    assert scope.count == 2
    assert scope.scope_kind == ir.ScopeKind.SplitAiv


def test_construct_rejects_bad_count():
    """count must be 2 (the two AIV sub-cores) regardless of mode."""
    span = ir.Span("test.py", 1, 1, 1, 10)
    body = _make_body(span, *_vars(span))
    with pytest.raises(pypto.InternalError):
        ir.SplitAivScopeStmt(split=ir.SplitMode.NONE, count=4, body=body, span=span)


def test_none_mode_serialize_roundtrip():
    """A task-parallel (None) region survives .pto serialize -> deserialize."""
    span = ir.Span("test.py", 1, 1, 1, 10)
    scope = ir.SplitAivScopeStmt(split=ir.SplitMode.NONE, body=_make_body(span, *_vars(span)), span=span)
    data = ir.serialize(scope)
    restored = ir.deserialize(data)
    assert isinstance(restored, ir.SplitAivScopeStmt)
    assert restored.split == ir.SplitMode.NONE
    assert restored.count == 2
    assert ir.serialize(restored) == data


def test_structural_equal_same():
    """Two nodes with identical mode/body compare structurally equal."""
    span = ir.Span("test.py", 1, 1, 1, 10)
    var_x, var_y = _vars(span)
    scope1 = ir.SplitAivScopeStmt(split=ir.SplitMode.UP_DOWN, body=_make_body(span, var_x, var_y), span=span)
    scope2 = ir.SplitAivScopeStmt(split=ir.SplitMode.UP_DOWN, body=_make_body(span, var_x, var_y), span=span)
    assert ir.structural_equal(scope1, scope2)


def test_structural_unequal_mode():
    """Different split modes make the nodes structurally unequal."""
    span = ir.Span("test.py", 1, 1, 1, 10)
    var_x, var_y = _vars(span)
    scope1 = ir.SplitAivScopeStmt(split=ir.SplitMode.UP_DOWN, body=_make_body(span, var_x, var_y), span=span)
    scope2 = ir.SplitAivScopeStmt(
        split=ir.SplitMode.LEFT_RIGHT, body=_make_body(span, var_x, var_y), span=span
    )
    assert not ir.structural_equal(scope1, scope2)


def test_serialize_roundtrip():
    """A .pto serialize -> deserialize round-trip is a byte-level fixpoint.

    (Free Vars in the body get fresh identities on deserialize, so
    structural_equal is not a reliable cross-roundtrip check for any scope
    node — re-serialize and compare bytes instead.)
    """
    span = ir.Span("test.py", 1, 1, 1, 10)
    scope = ir.SplitAivScopeStmt(
        split=ir.SplitMode.LEFT_RIGHT, body=_make_body(span, *_vars(span)), span=span
    )
    data = ir.serialize(scope)
    restored = ir.deserialize(data)

    assert isinstance(restored, ir.SplitAivScopeStmt)
    assert restored.split == ir.SplitMode.LEFT_RIGHT
    assert restored.count == 2
    assert ir.serialize(restored) == data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
