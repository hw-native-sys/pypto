# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for the ``device=`` kwarg on Orchestration dispatch calls.

``self.<orch>(..., device=r)`` selects the physical device for a per-rank
dispatch. The kwarg is only legal when the callee is an
``ir.FunctionType.Orchestration`` function. The parser is intentionally
permissive about the value form — any IR expression is parsed and stashed
on ``call.attrs['device']``. Downstream passes (simplify, comm-collection)
inspect the resolved expression to derive the per-dispatch device subset:

* ``ConstInt`` → fixed-device dispatch
* IR Var that's a ``for``-loop induction variable → subset / kAll
* Anything else → rejected by the comm-collection pass with full
  def-use context
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core import ir


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _find_callee_call(func: ir.Function, callee_name: str) -> ir.Call:
    found: list[ir.Call] = []

    def maybe_record(expr: ir.Expr | None) -> None:
        if isinstance(expr, ir.Call) and expr.op.name == callee_name:
            found.append(expr)

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.AssignStmt):
            maybe_record(stmt.value)
        if isinstance(stmt, ir.EvalStmt):
            maybe_record(stmt.expr)
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
        if isinstance(stmt, ir.ForStmt):
            walk(stmt.body)

    walk(func.body)
    assert found, f"no call to '{callee_name}' found in function body"
    return found[0]


# ---------------------------------------------------------------------------
# Positive cases — parser stashes the IR expression on ``call.attrs['device']``
# ---------------------------------------------------------------------------


def test_device_int_literal():
    """``device=0`` lifts to a ConstInt stored on call.attrs['device']."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, x: pl.Tensor[[64], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
            self.chip_orch(x, device=0)
            return 0

    func = _get_func(P, "host_orch")
    call = _find_callee_call(func, "chip_orch")
    device = call.attrs["device"]
    assert isinstance(device, ir.ConstInt)
    assert device.value == 0


def test_device_uses_for_loop_induction_var():
    """``device=r`` parses to the IR Var of the enclosing ``for r in pl.range(...)``."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, x: pl.Tensor[[64], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
            for r in pl.range(pld.world_size()):  # type: ignore[arg-type]
                self.chip_orch(x, device=r)
            return 0

    func = _get_func(P, "host_orch")
    call = _find_callee_call(func, "chip_orch")
    device = call.attrs["device"]
    assert isinstance(device, ir.Var)
    assert device.name_hint == "r"
    assert isinstance(device.type, ir.ScalarType)


def test_device_accepts_locally_bound_constant():
    """``other = 3; self.chip_orch(..., device=other)`` is accepted — ``other``
    parses to a Var bound to ConstInt(3); the comm-collection pass folds it
    via simplify before deriving the device subset."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, x: pl.Tensor[[64], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
            other = 3
            self.chip_orch(x, device=other)
            return 0

    func = _get_func(P, "host_orch")
    call = _find_callee_call(func, "chip_orch")
    device = call.attrs["device"]
    # The parser hands back the Var; downstream simplify can fold it to
    # ConstInt(3) when the comm-collection pass needs a literal.
    assert isinstance(device, ir.Var)
    assert device.name_hint == "other"


def test_device_accepts_arithmetic_expression():
    """``device=r + 1`` parses to an arithmetic IR expression. The
    comm-collection pass is responsible for any further validation."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, x: pl.Tensor[[64], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
            for r in pl.range(pld.world_size()):  # type: ignore[arg-type]
                self.chip_orch(x, device=r + 1)
            return 0

    func = _get_func(P, "host_orch")
    call = _find_callee_call(func, "chip_orch")
    device = call.attrs["device"]
    # Parser does not constrain shape — comm-collection pass will validate.
    assert isinstance(device, ir.Expr)


def test_device_accepts_explicit_level_role_orchestrator():
    """``FunctionType.Orchestration`` and ``level=..., role=Orchestrator`` are
    two ways to declare the same role. Both must accept ``device=`` on
    dispatch sites — the parser predicate is on ``role``, not ``func_type``."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip_orch(self, x: pl.Tensor[[64], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
            self.chip_orch(x, device=2)
            return 0

    func = _get_func(P, "host_orch")
    call = _find_callee_call(func, "chip_orch")
    device = call.attrs["device"]
    assert isinstance(device, ir.ConstInt)
    assert device.value == 2


# ---------------------------------------------------------------------------
# Negative cases — only the callee-type check stays in the parser
# ---------------------------------------------------------------------------


def test_device_rejected_on_non_orchestrator_callee():
    """A callee that is not an Orchestrator (default ``Opaque`` /
    ``SubWorker``) does not accept ``device=``. Predicate is on ``role``,
    not ``func_type``."""
    with pytest.raises(Exception, match="'device'"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
                self.kernel(x, device=0)
                return 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
