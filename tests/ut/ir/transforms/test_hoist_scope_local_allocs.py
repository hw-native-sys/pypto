# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the HoistScopeLocalAllocs pass.

The pass stamps ``attrs['hoistable_alloc'] = True`` onto every ``tensor.create``
that sits directly in a ``pl.manual_scope`` body (``RuntimeScopeStmt`` with
``manual == True``) and whose result shape references no Var defined inside that
body — the buffer is then enclosing-scope-valid and orchestration codegen hoists
its declaration one level out (issue #1697). A create nested in a for/if within
the scope, a shape-local create, or a create outside any manual scope is left
unmarked.

Tests assert on the stamped attr rather than a full Expected program: the pass
rewrites nothing but the create Call's ``attrs``.
"""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import passes as _passes

HOISTABLE_ATTR = "hoistable_alloc"
FP32 = ir.DataType.FP32
INDEX = ir.DataType.INDEX


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _create_call(shape):
    return ir.Call(ir.Op("tensor.create"), [], ir.TensorType(shape, FP32), ir.Span.unknown())


def _orch_program(body):
    span = ir.Span.unknown()
    func = ir.Function("orch", [], [], body, span, type=ir.FunctionType.Orchestration)
    return ir.Program([func], "prog", span)


def _run_pass(program):
    """Run the pass on hand-built IR. Verification is disabled so the pass's
    declared prerequisites (SSA / RuntimeScopesMaterialized / ...) need not be
    materialised on the minimal fixtures — we exercise the stamping logic only."""
    with passes.PassContext([], passes.VerificationLevel.NONE):
        return passes.hoist_scope_local_allocs()(program)


def _creates_in_program(program):
    """Map ``var name_hint -> tensor.create Call`` for every create in the program."""
    found = {}

    def walk(node):
        if node is None:
            return
        if isinstance(node, ir.RuntimeScopeStmt):
            walk(node.body)
        elif isinstance(node, ir.SeqStmts):
            for c in node.stmts:
                walk(c)
        elif isinstance(node, ir.ForStmt):
            walk(node.body)
        elif isinstance(node, ir.IfStmt):
            walk(node.then_body)
            walk(node.else_body)
        elif isinstance(node, ir.AssignStmt):
            v = node.value
            if isinstance(v, ir.Call) and v.op.name == ir.get_op("tensor.create").name:
                found[node.var.name_hint] = v

    for func in program.functions.values():
        walk(func.body)
    return found


def _is_marked(call) -> bool:
    return bool(dict(call.attrs).get(HOISTABLE_ATTR, False))


def test_static_create_in_manual_scope_is_marked():
    """A create with a constant shape directly in a manual-scope body is hoistable."""
    span = ir.Span.unknown()
    scratch = ir.Var("scratch", ir.TensorType([64], FP32), span)
    manual_body = ir.SeqStmts([ir.AssignStmt(scratch, _create_call([64]), span)], span)
    manual = ir.RuntimeScopeStmt(True, "m", body=manual_body, span=span)
    body = ir.SeqStmts([manual, ir.ReturnStmt(span)], span)

    out = _run_pass(_orch_program(body))
    creates = _creates_in_program(out)
    assert _is_marked(creates["scratch"]), "static manual-scope create must be marked hoistable"


def test_create_outside_manual_scope_not_marked():
    """A top-level create (no enclosing manual scope) is never marked."""
    span = ir.Span.unknown()
    t = ir.Var("t", ir.TensorType([64], FP32), span)
    body = ir.SeqStmts([ir.AssignStmt(t, _create_call([64]), span), ir.ReturnStmt(span)], span)

    out = _run_pass(_orch_program(body))
    assert not _is_marked(_creates_in_program(out)["t"]), "create outside a manual scope must stay unmarked"


def test_create_nested_in_for_within_manual_scope_not_marked():
    """A create nested inside a for-loop within the manual scope is not a direct-body
    statement, so it is not hoisted (it would fall out of the loop-body C++ scope)."""
    span = ir.Span.unknown()
    inner = ir.Var("inner", ir.TensorType([64], FP32), span)
    loop_var = ir.Var("i", ir.ScalarType(INDEX), span)
    for_body = ir.SeqStmts([ir.AssignStmt(inner, _create_call([64]), span)], span)
    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, INDEX, span),
        ir.ConstInt(4, INDEX, span),
        ir.ConstInt(1, INDEX, span),
        [],
        for_body,
        [],
        span,
    )
    manual = ir.RuntimeScopeStmt(True, "m", body=ir.SeqStmts([for_stmt], span), span=span)
    body = ir.SeqStmts([manual, ir.ReturnStmt(span)], span)

    out = _run_pass(_orch_program(body))
    assert not _is_marked(_creates_in_program(out)["inner"]), "for-nested create must stay unmarked"


def test_shape_local_create_not_marked():
    """A create whose shape references a Var defined inside the manual-scope body is
    not enclosing-scope-valid, so it is not hoisted."""
    span = ir.Span.unknown()
    dyn_n = ir.Var("n", ir.ScalarType(INDEX), span)
    dim = ir.Mul(dyn_n, ir.ConstInt(16, INDEX, span), INDEX, span)

    static_t = ir.Var("static_t", ir.TensorType([16, 8], FP32), span)
    dyn_t = ir.Var("dyn_t", ir.TensorType([dim, ir.ConstInt(8, INDEX, span)], FP32), span)
    manual_body = ir.SeqStmts(
        [
            ir.AssignStmt(static_t, _create_call([16, 8]), span),
            ir.AssignStmt(dyn_n, ir.ConstInt(4, INDEX, span), span),  # body-local shape source
            ir.AssignStmt(dyn_t, _create_call([dim, ir.ConstInt(8, INDEX, span)]), span),
        ],
        span,
    )
    manual = ir.RuntimeScopeStmt(True, "m", body=manual_body, span=span)
    body = ir.SeqStmts([manual, ir.ReturnStmt(span)], span)

    out = _run_pass(_orch_program(body))
    creates = _creates_in_program(out)
    assert _is_marked(creates["static_t"]), "constant-shape create must be marked"
    assert not _is_marked(creates["dyn_t"]), "shape-local create must stay unmarked"


def test_idempotent():
    """Re-running the pass does not duplicate or flip the attr."""
    span = ir.Span.unknown()
    scratch = ir.Var("scratch", ir.TensorType([64], FP32), span)
    manual = ir.RuntimeScopeStmt(
        True, "m", body=ir.SeqStmts([ir.AssignStmt(scratch, _create_call([64]), span)], span), span=span
    )
    program = _orch_program(ir.SeqStmts([manual, ir.ReturnStmt(span)], span))

    once = _run_pass(program)
    twice = _run_pass(once)
    call = _creates_in_program(twice)["scratch"]
    keys = [k for k in dict(call.attrs) if k == HOISTABLE_ATTR]
    assert keys == [HOISTABLE_ATTR], "attr must appear exactly once after a second run"
    assert _is_marked(call)


def test_pass_metadata():
    """Required/produced properties."""
    p = passes.hoist_scope_local_allocs()
    required = p.get_required_properties()
    produced = p.get_produced_properties()
    assert required.contains(_passes.IRProperty.RuntimeScopesMaterialized)
    assert required.contains(_passes.IRProperty.CallDirectionsResolved)
    assert produced.contains(_passes.IRProperty.HoistableAllocsMarked)


def test_full_pipeline_marks_manual_scope_create():
    """End-to-end: a create inside a pl.manual_scope, consumed by a submit, is marked
    by the default pipeline (which runs HoistScopeLocalAllocs after
    MaterializeRuntimeScopes)."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def k(
            self, x: pl.Tensor[[64], pl.FP32], scratch: pl.Out[pl.Tensor[[64], pl.FP32]]
        ) -> pl.Tensor[[64], pl.FP32]:
            t = pl.load(x, [0], [64])
            scratch = pl.store(t, [0], scratch)
            return scratch

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.manual_scope():
                scratch = pl.create_tensor([64], dtype=pl.FP32)
                a, a_tid = pl.submit(self.k, x, scratch)
            b, _ = pl.submit(self.k, a, x)
            return b

    transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
    creates = list(_creates_in_program(transformed).values())
    assert creates, "expected a tensor.create in the compiled program"
    assert all(_is_marked(c) for c in creates), "manual-scope create must be marked hoistable end-to-end"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
