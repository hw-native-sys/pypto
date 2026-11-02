# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""RegScope is a fieldless ScopeStmt with ordinary traversal and persistence."""

import pytest
from pypto import DataType, ir, passes
from pypto.ir.printer import python_print


def _scope():
    span = ir.Span("reg_scope.py", 2, 1, 4, 8)
    value = ir.ConstInt(1, DataType.INT32, span)
    return ir.RegScopeStmt("vf", ir.EvalStmt(value, span), span, ["keep"], {"tag": "local"})


def test_reg_scope_fields_and_roundtrip():
    scope = _scope()
    assert isinstance(scope, ir.ScopeStmt)
    assert scope.scope_kind == ir.ScopeKind.Reg
    assert ir.ScopeKind.Reg.value == 9
    assert scope.name_hint == "vf"
    assert scope.attrs == {"tag": "local"}
    assert list(scope.leading_comments) == ["keep"]
    assert not hasattr(scope, "captures")
    restored = ir.deserialize(ir.serialize(scope))
    assert isinstance(restored, ir.RegScopeStmt)
    ir.assert_structural_equal(scope, restored)
    assert ir.structural_hash(scope) == ir.structural_hash(restored)
    assert ir.serialize(scope) == ir.serialize(restored)
    other = ir.ClusterScopeStmt("vf", scope.body, scope.span)
    assert not ir.structural_equal(scope, other)


def test_reg_scope_visitor_and_mutator_preserve_metadata():
    class Visitor(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.values = []

        def visit_const_int(self, op):
            self.values.append(op.value)

    class Mutator(ir.IRMutator):
        def visit_const_int(self, op):
            return ir.ConstInt(op.value + 1, op.dtype, op.span)

    scope = _scope()
    visitor = Visitor()
    visitor.visit_stmt(scope)
    assert visitor.values == [1]
    changed = Mutator().visit_stmt(scope)
    assert isinstance(changed, ir.RegScopeStmt)
    assert isinstance(changed.body, ir.EvalStmt)
    assert isinstance(changed.body.expr, ir.ConstInt)
    assert changed.body.expr.value == 2
    assert changed.attrs == scope.attrs
    assert list(changed.leading_comments) == ["keep"]
    assert changed.span.filename == "reg_scope.py"
    assert not ir.structural_equal(scope, changed)


def test_flatten_call_expr_keeps_nested_calls_inside_reg_scope():
    span = ir.Span.unknown()
    scalar = ir.ScalarType(DataType.INT32)
    inner = ir.Call(ir.Op("inner"), [], scalar, span)
    outer = ir.Call(ir.Op("outer"), [inner], scalar, span)
    result = ir.Var("result", scalar, span)
    scope = ir.RegScopeStmt("vf", ir.AssignStmt(result, outer, span), span, ["keep"], {"tag": "local"})
    program = ir.Program(
        [ir.Function("main", [], [], scope, span, type=ir.FunctionType.InCore)], "test", span
    )
    with passes.PassContext([]):
        after = passes.flatten_call_expr()(program)
    body = next(iter(after.functions.values())).body
    if isinstance(body, ir.SeqStmts):
        assert len(body.stmts) == 1, "Nested calls must not escape the register scope"
        body = body.stmts[0]
    assert isinstance(body, ir.RegScopeStmt)
    assert body.name_hint == "vf"
    assert body.attrs == {"tag": "local"}
    assert list(body.leading_comments) == ["keep"]
    assert isinstance(body.body, ir.SeqStmts)
    first, second = body.body.stmts
    assert isinstance(first, ir.AssignStmt)
    assert isinstance(second, ir.AssignStmt)
    ir.assert_structural_equal(first.value, inner)
    assert isinstance(second.value, ir.Call)
    assert second.value.op.name == "outer"
    ir.assert_structural_equal(second.value.args[0], first.var)
    ir.assert_structural_equal(second.var, result)


@pytest.mark.parametrize("dead_body", ["assignment", "yield_slot"])
def test_simplify_preserves_reg_scope_when_eliminating_dead_values(dead_body):
    span = ir.Span("reg_scope.py", 2, 1, 4, 8)
    scalar = ir.ScalarType(DataType.INT32)
    condition = ir.Var("condition", ir.ScalarType(DataType.BOOL), span)
    value = ir.ConstInt(1, DataType.INT32, span)
    unused = ir.Var("unused", scalar, span)
    if dead_body == "assignment":
        body = ir.AssignStmt(unused, value, span)
    else:
        body = ir.IfStmt(
            condition,
            ir.YieldStmt([value], span),
            ir.YieldStmt([value], span),
            [unused],
            span,
        )
    scope = ir.RegScopeStmt("vf", body, span, ["keep"], {"tag": "local"})
    program = ir.Program(
        [ir.Function("main", [condition], [], scope, span, type=ir.FunctionType.InCore)], "test", span
    )
    with passes.PassContext([]):
        after = passes.simplify()(program)
    result = next(iter(after.functions.values())).body
    assert isinstance(result, ir.RegScopeStmt)
    assert result.name_hint == "vf"
    assert result.attrs == {"tag": "local"}
    assert list(result.leading_comments) == ["keep"]
    assert result.span.filename == span.filename
    if dead_body == "assignment":
        assert isinstance(result.body, ir.SeqStmts)
        assert not result.body.stmts
    else:
        assert isinstance(result.body, ir.IfStmt)
        assert not result.body.return_vars


def test_reg_scope_text_printing_does_not_erase_boundary():
    scope = _scope()
    printable = ir.RegScopeStmt("", scope.body, scope.span)
    assert "with pl.reg_scope():" in python_print(printable, format=False)


def test_reg_scope_empty_body_prints_parseable_pass():
    span = ir.Span("reg_scope.py", 2, 1, 2, 8)
    scope = ir.RegScopeStmt("", ir.SeqStmts([], span), span)
    printed = python_print(scope, format=False)
    assert printed == "with pl.reg_scope():\n    pass"


def test_reg_scope_metadata_is_rejected_in_python_printer():
    with pytest.raises(ValueError, match="name_hint or attrs"):
        python_print(_scope(), format=False)


def test_reg_scope_rejects_unknown_serialized_node_kind():
    payload = ir.serialize(_scope())
    assert payload.count(b"RegScopeStmt") == 1
    with pytest.raises(TypeError, match="Unknown|registered"):
        ir.deserialize(payload.replace(b"RegScopeStmt", b"BadScopeStmt"))


def test_reg_scope_builder_context_preserves_body_and_attrs():
    builder = ir.IRBuilder()
    span = ir.Span.unknown()
    value = ir.ConstInt(7, DataType.INT32, span)
    with builder.function("kernel", span=span, type=ir.FunctionType.InCore) as function:
        with builder.scope(ir.ScopeKind.Reg, span=span, name_hint="vf", attrs=[("tag", "local")]) as region:
            builder.emit(ir.EvalStmt(value, span))
    scope = region.get_result()
    assert isinstance(scope, ir.RegScopeStmt)
    assert scope.scope_kind == ir.ScopeKind.Reg
    assert scope.name_hint == "vf"
    assert scope.attrs == {"tag": "local"}
    assert isinstance(scope.body, ir.EvalStmt)
    ir.assert_structural_equal(scope.body.expr, value)
    built = function.get_result()
    assert built is not None
    ir.assert_structural_equal(built.body, scope)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
