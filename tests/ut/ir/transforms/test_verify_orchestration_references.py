# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the OrchestrationReferencesResolved property verifier.

Replaces the codegen-side ``ValidateOrchestrationReferences`` check that used
to throw at codegen time. The check now lives as a ``PropertyVerifier``
auto-invoked by the pass pipeline (registered as a property produced by
``OutlineHierarchyScopes``).
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _orch_refs_props():
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.OrchestrationReferencesResolved)
    return props


class TestOrchestrationReferencesResolvedVerifier:
    def test_well_formed_program_passes(self):
        """An Orchestration function whose every non-builtin callee exists in the Program."""

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.kernel(x)
                return y

        diagnostics = passes.PropertyVerifierRegistry.verify(_orch_refs_props(), P)
        errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
        assert errors == []

    def test_orchestration_calling_missing_function_fails(self):
        """An Orchestration function that references a non-existent callee."""
        # Build a minimal hand-rolled Orchestration function whose body calls
        # an undefined GlobalVar 'ghost'. The Program intentionally only
        # contains this orchestration function — 'ghost' is never registered.
        span = ir.Span.unknown()
        x = ir.Var("x", ir.TensorType([64], ir.DataType.FP32), span)
        ghost_call = ir.Call(ir.GlobalVar("ghost"), [x], ir.TupleType([]), span)
        body = ir.SeqStmts(
            [ir.EvalStmt(ghost_call, span), ir.ReturnStmt([], span)],
            span,
        )
        orch = ir.Function("orch_main", [x], [], body, span, type=ir.FunctionType.Orchestration)
        prog = ir.Program([orch], "test_orch_missing", span)

        diagnostics = passes.PropertyVerifierRegistry.verify(_orch_refs_props(), prog)
        errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
        assert len(errors) == 1
        assert "ghost" in errors[0].message
        assert "undefined function" in errors[0].message

    def test_builtin_calls_are_skipped(self):
        """Builtin ops, including the distributed context query, are not undefined callees."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.TensorType([64], ir.DataType.FP32), span)
        # tensor.print is a builtin — must be ignored by the verifier.
        builtin_call = ir.Call(ir.GlobalVar("tensor.print"), [x], ir.TupleType([]), span)
        get_ctx_call = ir.Call(ir.GlobalVar("pld.system.get_comm_ctx"), [x], ir.CommCtxType.get(), span)
        body = ir.SeqStmts(
            [ir.EvalStmt(builtin_call, span), ir.EvalStmt(get_ctx_call, span), ir.ReturnStmt([], span)],
            span,
        )
        orch = ir.Function("orch_main", [x], [], body, span, type=ir.FunctionType.Orchestration)
        prog = ir.Program([orch], "test_orch_builtin", span)

        diagnostics = passes.PropertyVerifierRegistry.verify(_orch_refs_props(), prog)
        errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
        assert errors == []

    @pytest.mark.parametrize("op_name", ["pld.tensor.put_async", "pld.tensor.get_async"])
    def test_async_transfer_in_host_body_fails(self, op_name):
        """put_async/get_async must run in InCore, not a host (Opaque) body."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.TensorType([64], ir.DataType.FP32), span)
        put = ir.Call(ir.GlobalVar(op_name), [x], ir.TupleType([]), span)
        body = ir.SeqStmts([ir.EvalStmt(put, span), ir.ReturnStmt([], span)], span)
        host = ir.Function("host_main", [x], [], body, span)  # default Opaque
        prog = ir.Program([host], "test_async_host_placement", span)

        diagnostics = passes.PropertyVerifierRegistry.verify(_orch_refs_props(), prog)
        errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
        assert len(errors) == 1, [e.message for e in errors]
        assert "host_main" in errors[0].message
        assert op_name in errors[0].message
        assert "non-InCore" in errors[0].message

    @pytest.mark.parametrize("op_name", ["pld.tensor.put_async", "pld.tensor.get_async"])
    @pytest.mark.parametrize("func_type", [ir.FunctionType.Orchestration, ir.FunctionType.Graph])
    def test_async_transfer_in_orchestration_body_fails(self, op_name, func_type):
        """Placement errors must not also report the op as an undefined function."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.TensorType([64], ir.DataType.FP32), span)
        get = ir.Call(ir.GlobalVar(op_name), [x], ir.TupleType([]), span)
        body = ir.SeqStmts([ir.EvalStmt(get, span), ir.ReturnStmt([], span)], span)
        orch = ir.Function("orch_main", [x], [], body, span, type=func_type)
        prog = ir.Program([orch], "test_async_orch_placement", span)

        diagnostics = passes.PropertyVerifierRegistry.verify(_orch_refs_props(), prog)
        errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
        assert len(errors) == 1, [e.message for e in errors]
        assert op_name in errors[0].message
        assert "non-InCore" in errors[0].message

    @pytest.mark.parametrize("op_name", ["pld.tensor.put_async", "pld.tensor.get_async"])
    def test_async_transfer_in_incore_body_passes(self, op_name):
        """The placement guard must continue to allow the owning InCore body."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.TensorType([64], ir.DataType.FP32), span)
        call = ir.Call(ir.GlobalVar(op_name), [x], ir.TupleType([]), span)
        body = ir.SeqStmts([ir.EvalStmt(call, span), ir.ReturnStmt([], span)], span)
        kernel = ir.Function("kernel", [x], [], body, span, type=ir.FunctionType.InCore)
        prog = ir.Program([kernel], "test_async_incore_placement", span)

        diagnostics = passes.PropertyVerifierRegistry.verify(_orch_refs_props(), prog)
        errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
        assert errors == []

    @pytest.mark.parametrize("op_name", ["pld.tensor.put_async", "pld.tensor.get_async"])
    def test_async_transfer_submit_requires_defined_callee(self, op_name):
        """Exempting a direct op Call must not exempt an undefined Submit target."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.TensorType([64], ir.DataType.FP32), span)
        submit = ir.Submit(ir.GlobalVar(op_name), [x], [], ir.ScalarType(ir.DataType.TASK_ID), span)
        body = ir.SeqStmts([ir.EvalStmt(submit, span), ir.ReturnStmt([], span)], span)
        graph = ir.Function("graph_main", [x], [], body, span, type=ir.FunctionType.Graph)
        prog = ir.Program([graph], "test_async_submit_callee", span)

        diagnostics = passes.PropertyVerifierRegistry.verify(_orch_refs_props(), prog)
        errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
        assert len(errors) == 1, [e.message for e in errors]
        assert op_name in errors[0].message
        assert "undefined function" in errors[0].message

    def test_non_orchestration_functions_are_skipped(self):
        """Non-Orchestration (e.g. Opaque) functions are not checked for undefined callees."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.TensorType([64], ir.DataType.FP32), span)
        ghost_call = ir.Call(ir.GlobalVar("ghost"), [x], ir.TupleType([]), span)
        body = ir.SeqStmts(
            [ir.EvalStmt(ghost_call, span), ir.ReturnStmt([], span)],
            span,
        )
        # Default function type is Opaque — verifier should ignore it.
        opaque = ir.Function("opaque_main", [x], [], body, span)
        prog = ir.Program([opaque], "test_opaque", span)

        diagnostics = passes.PropertyVerifierRegistry.verify(_orch_refs_props(), prog)
        errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
        assert errors == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
