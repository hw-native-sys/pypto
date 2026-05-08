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
        """Builtin ops (tile.*, tensor.*, system.*) must NOT be flagged as undefined."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.TensorType([64], ir.DataType.FP32), span)
        # tensor.print is a builtin — must be ignored by the verifier.
        builtin_call = ir.Call(ir.GlobalVar("tensor.print"), [x], ir.TupleType([]), span)
        body = ir.SeqStmts(
            [ir.EvalStmt(builtin_call, span), ir.ReturnStmt([], span)],
            span,
        )
        orch = ir.Function("orch_main", [x], [], body, span, type=ir.FunctionType.Orchestration)
        prog = ir.Program([orch], "test_orch_builtin", span)

        diagnostics = passes.PropertyVerifierRegistry.verify(_orch_refs_props(), prog)
        errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
        assert errors == []

    def test_non_orchestration_functions_are_skipped(self):
        """Non-Orchestration (e.g. Opaque) functions are NOT verified by this property."""
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
