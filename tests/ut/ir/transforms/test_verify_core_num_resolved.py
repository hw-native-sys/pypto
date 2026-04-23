# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the CoreNumResolved property verifier.

The verifier enforces three invariants on ``SpmdScopeStmt::core_num_``:

1. The expression folds to a ``ConstInt`` (no runtime IR values).
2. The value is positive (``> 0``).
3. The value fits in an ``int32`` (so downstream ``static_cast<int>`` is safe
   when the integer is emplaced as a Function attribute consumed by
   orchestration codegen).
"""

import pytest
from pypto import DataType, ir, passes

INT32_MAX = 2**31 - 1


def _make_program_with_spmd(core_num_expr: ir.Expr) -> ir.Program:
    """Build a minimal program wrapping a single ``SpmdScopeStmt``."""
    span = ir.Span.unknown()
    var = ir.Var("x", ir.ScalarType(DataType.INDEX), span)
    body = ir.EvalStmt(var, span)
    spmd = ir.SpmdScopeStmt(core_num=core_num_expr, body=body, span=span)
    fn = ir.Function("main", [], [], body=spmd, span=span, type=ir.FunctionType.Opaque)
    return ir.Program([fn], "test_program", span)


def _verify(program: ir.Program) -> list:
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.CoreNumResolved)
    return passes.PropertyVerifierRegistry.verify(props, program)


def test_core_num_resolved_accepts_positive_const_int():
    span = ir.Span.unknown()
    program = _make_program_with_spmd(ir.ConstInt(4, DataType.INDEX, span))
    assert _verify(program) == []


def test_core_num_resolved_rejects_non_const_int():
    """A runtime Var is not a compile-time integer — verifier must reject."""
    span = ir.Span.unknown()
    runtime_var = ir.Var("ctx_blocks", ir.ScalarType(DataType.INDEX), span)
    program = _make_program_with_spmd(runtime_var)
    diagnostics = _verify(program)
    assert len(diagnostics) == 1
    assert diagnostics[0].severity == passes.DiagnosticSeverity.Error
    assert diagnostics[0].rule_name == "CoreNumResolved"
    assert "did not fold to a compile-time integer" in diagnostics[0].message


def test_core_num_resolved_rejects_zero():
    span = ir.Span.unknown()
    program = _make_program_with_spmd(ir.ConstInt(0, DataType.INDEX, span))
    diagnostics = _verify(program)
    assert len(diagnostics) == 1
    assert "must be positive" in diagnostics[0].message


def test_core_num_resolved_rejects_negative():
    span = ir.Span.unknown()
    program = _make_program_with_spmd(ir.ConstInt(-4, DataType.INDEX, span))
    diagnostics = _verify(program)
    assert len(diagnostics) == 1
    assert "must be positive" in diagnostics[0].message


def test_core_num_resolved_rejects_overflow_int32():
    """A value that does not fit in int32 would overflow the downstream cast."""
    span = ir.Span.unknown()
    program = _make_program_with_spmd(ir.ConstInt(INT32_MAX + 1, DataType.INDEX, span))
    diagnostics = _verify(program)
    assert len(diagnostics) == 1
    assert "int32 range" in diagnostics[0].message


def test_core_num_resolved_accepts_int32_max():
    span = ir.Span.unknown()
    program = _make_program_with_spmd(ir.ConstInt(INT32_MAX, DataType.INDEX, span))
    assert _verify(program) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
