# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Framework wire-up tests for :class:`WindowBuffer`.

After N2, ``WindowBuffer`` is a :class:`Var` subclass with its own
``ObjectKind``. The dispatch tables in ``ExprFunctor`` / ``IRVisitor`` /
``IRMutator`` must each have an explicit ``WindowBuffer`` entry — otherwise
any pass that visits IR containing a constructed ``WindowBuffer`` would
fall through and raise ``TypeError("Unknown expression type")``.

These tests construct a ``WindowBuffer`` directly and drive it through the
base ``IRVisitor`` and ``IRMutator`` to confirm the wire-up is in place
before the ``MaterializeCommDomainScopes`` pass (which materialises ``WindowBuffer``
instances during normal compilation) lands.
"""

import pytest
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import (
    ConstInt,
    IRMutator,
    IRVisitor,
    PtrType,
    Span,
    Var,
    WindowBuffer,
)


def _const(value: int) -> ConstInt:
    return ConstInt(value, DataType.INT64, Span.unknown())


def _ptr(name: str) -> Var:
    return Var(name, PtrType(), Span.unknown())


def test_visitor_dispatch_does_not_throw():
    """``IRVisitor::VisitExpr(window_buffer)`` must route to the dedicated
    overload, not fall through to the catch-all ``TypeError``."""
    wb = WindowBuffer(_ptr("data"), _const(256))
    visitor = IRVisitor()
    # Default impl is a no-op; the key invariant is "no TypeError".
    visitor.visit_expr(wb)


def test_visitor_custom_subclass_sees_window_buffer():
    """A Python-side subclass overriding ``visit_window_buffer`` must be
    invoked when the dispatcher sees a ``WindowBuffer`` instance."""

    class Recorder(IRVisitor):
        def __init__(self):
            super().__init__()
            self.seen: list[WindowBuffer] = []

        def visit_window_buffer(self, op):  # noqa: D401 - test hook
            self.seen.append(op)

    wb = WindowBuffer(_ptr("data"), _const(256))
    rec = Recorder()
    rec.visit_expr(wb)
    assert len(rec.seen) == 1
    assert rec.seen[0] is wb


def test_mutator_identity_returns_same_instance():
    """The base ``IRMutator`` is copy-on-write: if no child changed, it must
    return the original ``WindowBuffer`` instance, not mint a fresh one."""
    wb = WindowBuffer(_ptr("data"), _const(256))
    mutator = IRMutator()
    result = mutator.visit_expr(wb)
    assert result is wb


def test_mutator_remaps_base_via_var_substitution():
    """When the mutator's ``var_remap_`` (seeded through a subclass) swaps
    the underlying Ptr ``base``, the WindowBuffer override must mint a fresh
    ``WindowBuffer`` referencing the substituted base."""

    old_base = _ptr("data")
    new_base = _ptr("data")  # Different Var identity, same name_hint
    wb = WindowBuffer(old_base, _const(256))

    class SubstituteBaseMutator(IRMutator):
        def visit_var(self, op):
            if op is old_base:
                return new_base
            return super().visit_var(op)

    fresh = SubstituteBaseMutator().visit_expr(wb)
    assert fresh is not wb
    assert isinstance(fresh, WindowBuffer)
    assert fresh.base is new_base
    assert fresh.load_from_host == wb.load_from_host
    assert fresh.store_to_host == wb.store_to_host


def test_mutator_preserves_host_staging_flags():
    """Host-staging flags must round-trip through identity mutation."""
    wb = WindowBuffer(_ptr("data"), _const(64), load_from_host=True, store_to_host=True)
    mutator = IRMutator()
    result = mutator.visit_expr(wb)
    assert result is wb  # identity (no children changed)
    assert isinstance(result, WindowBuffer)
    assert result.load_from_host is True
    assert result.store_to_host is True


def _window_rebind_fixture(child_name, owner_name):
    """Include all type-bearing result paths without invoking parser/lowering."""
    span = Span.unknown()
    old = WindowBuffer(_ptr("deferred"), _const(0))
    n = _ir.Var("n", _ir.ScalarType(DataType.INDEX), span)
    size = _ir.Var("size", _ir.ScalarType(DataType.INDEX), span)
    base = _ptr("buf")
    actual = WindowBuffer(base, size, True, True, span)
    storage = _ir.MemRef("buf", 8, 12, span)
    view = _ir.TensorView(stride=[1, 2], layout=_ir.TensorLayout.ND)
    ty = _ir.DistributedTensorType([_const(2), _const(3)], DataType.FP16, storage, view, old)
    x = _ir.Var("x", ty, span)
    # A nested tuple parameter exercises recursive type rebuilding separately
    # from the ordinary distributed parameter and scalar TaskId suffix.
    nested_type = _ir.TupleType([ty, _ir.TupleType([ty, _ir.ScalarType(DataType.INDEX)])])
    nested = _ir.Var("nested", nested_type, span)
    child = _ir.Function(child_name, [x, nested], [ty], _ir.ReturnStmt([x], span), span)
    seed = _ir.Var("seed", ty, span)
    dep = _ir.Var("dep", _ir.ScalarType(DataType.TASK_ID), span)
    owner_nested = _ir.Var("owner_nested", nested_type, span)
    called = _ir.Call(_ir.GlobalVar(child_name), [seed, owner_nested], {}, {"saved": seed}, ty, span)
    call_result = _ir.Var("called", ty, span)
    submit_type = _ir.TupleType([ty, _ir.ScalarType(DataType.TASK_ID)])
    submitted = _ir.Submit(
        _ir.GlobalVar(child_name),
        [seed, owner_nested],
        [dep],
        {},
        {"saved": seed},
        submit_type,
        span,
        core_num=_const(2),
        sync_start=True,
        allow_early_resolve=True,
        predicate=_ir.ConstBool(True, span),
    )
    packed = _ir.Var("packed", submit_type, span)
    first = _ir.Var("first", ty, span)
    scope = _ir.CommDomainScopeStmt(
        [],
        [old],
        "global",
        body=_ir.SeqStmts(
            [
                _ir.AssignStmt(size, _ir.Mul(n, _const(4), DataType.INDEX, span), span),
                _ir.AssignStmt(
                    base,
                    _ir.create_op_call("pld.tensor.alloc_window_buffer", [size], {"name": "buf"}, span),
                    span,
                ),
                _ir.AssignStmt(call_result, called, span),
                _ir.AssignStmt(packed, submitted, span),
                _ir.AssignStmt(first, _ir.TupleGetItemExpr(packed, 0, span), span),
                _ir.EvalStmt(old, span),
                _ir.ReturnStmt([first, owner_nested], span),
            ],
            span,
        ),
        span=span,
    )
    owner = _ir.Function(
        owner_name, [n, seed, dep, owner_nested], [ty, nested_type], scope, span, attrs={"saved": seed}
    )
    return _ir.Program([child, owner], "WindowRebind", span), old, actual, storage, view


@pytest.mark.parametrize("names", [("a_child", "z_owner"), ("z_child", "a_owner")])
def test_private_window_rebind_updates_all_types_without_rebinding_owner_storage(names):
    """The opt-in final traversal changes only explicit WindowBuffer identities."""
    child_name, owner_name = names
    program, old, actual, storage, view = _window_rebind_fixture(*names)
    changed = _ir._rebind_window_buffers(program, [(old, actual)])
    child = changed.get_function(child_name)
    owner = changed.get_function(owner_name)
    expected_type = _ir.DistributedTensorType([_const(2), _const(3)], DataType.FP16, storage, view, actual)
    for ty in (child.params[0].type, child.return_types[0], owner.params[1].type, owner.return_types[0]):
        assert ty.window_buffer is actual
        assert ty.memref is storage
        # TensorView is stored by value; compare its complete metadata while
        # retaining exact identity checks for MemRef and WindowBuffer.
        _ir.assert_structural_equal(ty, expected_type)
        assert ty.dtype == DataType.FP16
        assert [dim.value for dim in ty.shape] == [2, 3]
    for ty in (child.params[1].type, owner.params[3].type, owner.return_types[1]):
        assert ty.types[0].window_buffer is actual
        assert ty.types[1].types[0].window_buffer is actual
    returned = child.body.value
    assert returned[0] is child.params[0]
    scope = owner.body
    assert scope.slots[0] is actual
    size, alloc, call, submit, projection, direct, returned = scope.body.stmts
    assert actual.base is alloc.var
    assert actual.size is size.var
    assert size.value.left is owner.params[0]
    assert alloc.value.args[0] is size.var
    assert actual.load_from_host is True and actual.store_to_host is True
    assert storage.base_ is not actual.base
    assert owner.attrs["saved"] is owner.params[1]
    for assignment in (call, submit):
        launch = assignment.value
        assert launch.args[0] is owner.params[1]
        assert launch.args[1] is owner.params[3]
        assert dict(launch.attrs)["saved"] is owner.params[1]
    assert call.var.type.window_buffer is actual
    assert call.value.type.window_buffer is actual
    assert submit.var.type.types[0].window_buffer is actual
    assert submit.value.type.types[0].window_buffer is actual
    assert submit.value.type.types[1].dtype == DataType.TASK_ID
    assert submit.value.deps[0] is owner.params[2]
    assert submit.value.core_num.value == 2
    assert submit.value.sync_start and submit.value.allow_early_resolve
    assert submit.value.predicate.value is True
    assert projection.var.type.window_buffer is actual
    assert projection.value.type.window_buffer is actual
    assert projection.value.tuple is submit.var
    assert direct.expr is actual
    assert returned.value[0] is projection.var
    assert returned.value[1] is owner.params[3]
    assert program.get_function(child_name).params[0].type.window_buffer is old
    assert _ir._rebind_window_buffers(changed, []) is changed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
