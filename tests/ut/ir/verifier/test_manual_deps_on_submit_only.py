# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the ManualDepsOnSubmitOnly property verifier.

Manual dependency edges belong in the typed ``Submit::deps_`` field. A plain
cross-function ``Call`` (GlobalVar callee) carrying ``attrs["manual_dep_edges"]``
is a verifier error; op callees (e.g. ``system.task_dummy``) legitimately keep
the attr as their codegen fanin contract and are exempt.
"""

import pytest
from pypto import DataType, ir
from pypto.pypto_core import passes


def _verify(prog):
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.ManualDepsOnSubmitOnly)
    return passes.PropertyVerifierRegistry.verify(props, prog)


def _span():
    return ir.Span.unknown()


def _t64():
    return ir.TensorType([64], DataType.FP32)


def _task_id():
    return ir.ScalarType(DataType.TASK_ID)


def _build_worker(span):
    """Callee ``worker(x: In) -> Tensor[[64], FP32]`` returning its input."""
    x = ir.Var("x", _t64(), span)
    return ir.Function(
        "worker",
        [(x, ir.ParamDirection.In)],
        [_t64()],
        ir.ReturnStmt([x], span),
        span,
        ir.FunctionType.InCore,
    )


def test_plain_call_with_manual_dep_edges_is_rejected():
    """A plain cross-function Call (GlobalVar callee) carrying
    ``attrs["manual_dep_edges"]`` fails verification."""
    span = _span()
    a = ir.Var("a", _t64(), span)
    t = ir.Var("t", _task_id(), span)
    res = ir.Var("res", _t64(), span)
    call = ir.Call(ir.GlobalVar("worker"), [a], {}, {"manual_dep_edges": [t]}, _t64(), span)
    main = ir.Function(
        "main",
        [(a, ir.ParamDirection.In), (t, ir.ParamDirection.In)],
        [_t64()],
        ir.SeqStmts([ir.AssignStmt(res, call, span), ir.ReturnStmt([res], span)], span),
        span,
        ir.FunctionType.Orchestration,
    )
    prog = ir.Program([_build_worker(span), main], "bad_manual_deps", span)

    diags = _verify(prog)
    assert len(diags) == 1
    assert diags[0].rule_name == "ManualDepsOnSubmitOnly"
    assert "manual_dep_edges" in diags[0].message
    assert "pl.submit" in diags[0].message


def test_task_dummy_call_with_manual_dep_edges_is_exempt():
    """``system.task_dummy`` is an Op callee (not GlobalVar) — its
    ``manual_dep_edges`` fanin contract is exempt."""
    span = _span()
    t = ir.Var("t", _task_id(), span)
    dummy = ir.Var("dummy", _task_id(), span)
    call = ir.Call(ir.Op("system.task_dummy"), [], {}, {"manual_dep_edges": [t]}, _task_id(), span)
    main = ir.Function(
        "main",
        [(t, ir.ParamDirection.In)],
        [_task_id()],
        ir.SeqStmts([ir.AssignStmt(dummy, call, span), ir.ReturnStmt([dummy], span)], span),
        span,
        ir.FunctionType.Orchestration,
    )
    prog = ir.Program([main], "dummy_exempt", span)

    assert _verify(prog) == []


def test_submit_with_deps_is_legal():
    """A Submit carrying deps in its first-class ``deps_`` field passes."""
    span = _span()
    a = ir.Var("a", _t64(), span)
    t = ir.Var("t", _task_id(), span)
    submit_ret = ir.TupleType([_t64(), _task_id()])
    res = ir.Var("res", submit_ret, span)
    submit = ir.Submit(ir.GlobalVar("worker"), [a], [t], submit_ret, span)
    main = ir.Function(
        "main",
        [(a, ir.ParamDirection.In), (t, ir.ParamDirection.In)],
        [submit_ret],
        ir.SeqStmts([ir.AssignStmt(res, submit, span), ir.ReturnStmt([res], span)], span),
        span,
        ir.FunctionType.Orchestration,
    )
    prog = ir.Program([_build_worker(span), main], "submit_deps_ok", span)

    assert _verify(prog) == []


def test_submit_with_stray_manual_dep_edges_attr_is_rejected():
    """``deps_`` is the single source of truth on a Submit; a stray
    ``attrs["manual_dep_edges"]`` (legal only inside the transient
    SubmitToCallView) fails verification."""
    span = _span()
    a = ir.Var("a", _t64(), span)
    t = ir.Var("t", _task_id(), span)
    submit_ret = ir.TupleType([_t64(), _task_id()])
    res = ir.Var("res", submit_ret, span)
    submit = ir.Submit(ir.GlobalVar("worker"), [a], [t], {}, {"manual_dep_edges": [t]}, submit_ret, span)
    main = ir.Function(
        "main",
        [(a, ir.ParamDirection.In), (t, ir.ParamDirection.In)],
        [submit_ret],
        ir.SeqStmts([ir.AssignStmt(res, submit, span), ir.ReturnStmt([res], span)], span),
        span,
        ir.FunctionType.Orchestration,
    )
    prog = ir.Program([_build_worker(span), main], "submit_stray_attr", span)

    diags = _verify(prog)
    assert len(diags) == 1
    assert diags[0].rule_name == "ManualDepsOnSubmitOnly"
    assert "deps_" in diags[0].message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
