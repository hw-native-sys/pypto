# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the InlineFunctions pass.

Verifies that ``FunctionType::Inline`` functions are spliced into every call
site (alpha-renamed, with formal params substituted by actual args) and then
removed from the program.

Tests use the Before/Expected pattern with ``ir.assert_structural_equal``,
which compares programs under alpha-equivalence (Var name mismatches are OK
as long as the LHS↔RHS Var mapping is consistent throughout)."""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir import OptimizationStrategy, PassManager
from pypto.pypto_core import passes as core_passes


class TestInlineFunctionsBasic:
    """Single-call-site, single-return cases."""

    def test_single_call_site(self):
        """One Inline function called once: body spliced, function removed."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
                return y

            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                z: pl.Tensor[[1], pl.INT32] = self.helper(a)
                return z

        @pl.program
        class Expected:
            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y_inline: pl.Tensor[[1], pl.INT32] = pl.mul(a, a)
                z: pl.Tensor[[1], pl.INT32] = y_inline
                return z

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_inline_function_dropped_from_program(self):
        """After splicing, the Inline function is removed from the program."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                z: pl.Tensor[[1], pl.INT32] = self.helper(a)
                return z

        After = passes.inline_functions()(Before)
        names = [f.name for f in After.functions.values()]
        assert "helper" not in names
        assert "main" in names

    def test_no_inline_functions_is_noop(self):
        """Programs with no Inline functions pass through unchanged."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = pl.add(x, x)
                return y

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Before)


class TestInlineFunctionsCallSiteForms:
    """The three top-level call-site forms: AssignStmt, EvalStmt, self-aliasing.

    The AssignStmt form is exercised throughout the file; these pin the two
    less-common forms handled by ``HandleTopLevelInlineCall``."""

    def test_eval_stmt_call_site_drops_return(self):
        """A bare ``self.writeout(...)`` (EvalStmt, no LHS) splices only the
        pre-return body and drops the trailing return value.

        ``SpliceInlineCallAsEval`` (src lines 293-296) calls ``CloneInlineBody``
        and returns ``body.stmts`` only — the trailing ``return out`` value is
        discarded since there is no LHS to bind it to. The in-place rebinding
        ``out = pl.assemble(out, x, ...)`` collapses to ``ext = pl.assemble(ext,
        a, ...)`` under the ``x→a``, ``out→ext`` param substitution. The caller
        deliberately does not read ``ext`` back (that would trip the
        InOutUseDiscipline structural verifier); it returns an independent
        value, so the dropped return is genuinely unused."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def writeout(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                out = pl.tensor.assemble(out, x, [0])
                return out

            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                ext: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                self.writeout(a, ext)  # EvalStmt call site — no LHS
                b: pl.Tensor[[4], pl.FP32] = pl.tensor.assemble(a, a, [0])
                return b

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                ext: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                ext = pl.tensor.assemble(ext, a, [0])  # spliced; trailing return dropped
                b: pl.Tensor[[4], pl.FP32] = pl.tensor.assemble(a, a, [0])
                return b

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_self_aliasing_assign_skips_redundant_copy(self):
        """``a = self.passthrough(a)`` where the inline returns its param
        verbatim emits NO assignment — the ``lhs = lhs`` no-op is elided.

        ``SpliceInlineCallAsAssign`` (src lines 313-318): the substituted return
        value is the Var ``a`` and the call-site LHS is also ``a``, so
        ``var_expr.get() == lhs.get()`` holds and the body's stmts (empty here)
        are returned without appending ``a = a``. ``main`` collapses to a bare
        ``return a``."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def passthrough(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                return x

            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                a = self.passthrough(a)  # arg == LHS Var → redundant copy elided
                return a

        @pl.program
        class Expected:
            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                return a

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)


class TestInlineFunctionsMultiCallSite:
    """Multiple call sites of the same Inline function: each gets a fresh expansion."""

    def test_multiple_call_sites_independent_expansion(self):
        """Same Inline called twice → two independently alpha-renamed copies."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def square(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
                return y

            @pl.function
            def main(
                self,
                a: pl.Tensor[[1], pl.INT32],
                b: pl.Tensor[[1], pl.INT32],
            ) -> pl.Tensor[[1], pl.INT32]:
                a2: pl.Tensor[[1], pl.INT32] = self.square(a)
                b2: pl.Tensor[[1], pl.INT32] = self.square(b)
                s: pl.Tensor[[1], pl.INT32] = pl.add(a2, b2)
                return s

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[1], pl.INT32],
                b: pl.Tensor[[1], pl.INT32],
            ) -> pl.Tensor[[1], pl.INT32]:
                y_a_inline: pl.Tensor[[1], pl.INT32] = pl.mul(a, a)
                a2: pl.Tensor[[1], pl.INT32] = y_a_inline
                y_b_inline: pl.Tensor[[1], pl.INT32] = pl.mul(b, b)
                b2: pl.Tensor[[1], pl.INT32] = y_b_inline
                s: pl.Tensor[[1], pl.INT32] = pl.add(a2, b2)
                return s

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)


class TestInlineFunctionsNested:
    """Inline calls Inline: pass iterates to fixpoint."""

    def test_inline_calls_inline(self):
        """A → B (both Inline) → caller. Both inlined."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def square(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
                return y

            @pl.function(type=pl.FunctionType.Inline)
            def quad(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                sq: pl.Tensor[[1], pl.INT32] = self.square(x)
                sq2: pl.Tensor[[1], pl.INT32] = self.square(sq)
                return sq2

            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                r: pl.Tensor[[1], pl.INT32] = self.quad(a)
                return r

        After = passes.inline_functions()(Before)

        # After: both Inline functions gone; main has the fully-expanded body.
        names = [f.name for f in After.functions.values()]
        assert names == ["main"]

        # Body has 5 statements: 2 mul (one per square call) + 2 sq* assigns
        # (from the quad body) + 1 r assign (the call site result) + return.
        # Exact shape verified below via structural equality.
        @pl.program
        class Expected:
            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                # First square (called from inlined quad on a)
                y0_inline: pl.Tensor[[1], pl.INT32] = pl.mul(a, a)
                sq_inline: pl.Tensor[[1], pl.INT32] = y0_inline
                # Second square (called from inlined quad on sq_inline)
                y1_inline: pl.Tensor[[1], pl.INT32] = pl.mul(sq_inline, sq_inline)
                sq2_inline: pl.Tensor[[1], pl.INT32] = y1_inline
                # quad's return → main's call-site LHS
                r: pl.Tensor[[1], pl.INT32] = sq2_inline
                return r

        ir.assert_structural_equal(After, Expected)


class TestInlineFunctionsCycles:
    """Cycle detection in the Inline → Inline call graph."""

    def test_self_recursion_errors(self):
        """An Inline function calling itself raises ValueError."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def loop(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = self.loop(x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                r: pl.Tensor[[1], pl.INT32] = self.loop(x)
                return r

        with pytest.raises(ValueError, match="Cycle detected"):
            passes.inline_functions()(Before)

    def test_mutual_recursion_errors(self):
        """A → B → A (both Inline) raises ValueError naming the cycle."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def a(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = self.b(x)
                return y

            @pl.function(type=pl.FunctionType.Inline)
            def b(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = self.a(x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                r: pl.Tensor[[1], pl.INT32] = self.a(x)
                return r

        with pytest.raises(ValueError, match="Cycle detected.*Inline"):
            passes.inline_functions()(Before)


class TestInlineFunctionsBodyShapes:
    """Inline bodies containing pl.at, pl.range, and other constructs.

    The pass must preserve the body verbatim (modulo alpha-rename + param
    substitution); downstream passes (OutlineIncoreScopes, UnrollLoops, etc.)
    handle the spliced constructs as if they had been written inline.
    """

    def test_inline_body_with_pl_at(self):
        """An Inline body containing ``with pl.at(...)`` splices the scope intact."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.helper(a)
                return r

        @pl.program
        class Expected:
            @pl.function
            def main(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y_inline: pl.Tensor[[64], pl.FP32] = pl.add(a, a)
                r: pl.Tensor[[64], pl.FP32] = y_inline
                return r

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_inline_body_with_pl_range(self):
        """An Inline body containing ``for i in pl.range(...)`` splices the loop
        intact, with the loop body alpha-renamed and params substituted.

        Uses an in-place ``pl.Out`` rebinding inside the loop (no loop-carried
        return var) so the spliced shape is hand-derivable: ``CloneInlineBody``
        deep-clones the For body with ``x→a``, ``out→ext`` (param substitution
        carried into both use- and def-sites, src lines 224-242), the base
        IRMutator mints a fresh loop var, and the trailing ``return out`` aliases
        to the call-site LHS (``r = ext``)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                for i in pl.range(4):
                    out = pl.tensor.assemble(out, x, [0])
                return out

            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                ext: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                r: pl.Tensor[[4], pl.FP32] = self.helper(a, ext)
                return r

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                ext: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                for i in pl.range(4):
                    ext = pl.tensor.assemble(ext, a, [0])
                r: pl.Tensor[[4], pl.FP32] = ext  # trailing return aliased to LHS
                return r

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)


class TestInlineFunctionsDeadCode:
    """Inline functions with no callers."""

    def test_no_callers_silently_dropped(self):
        """An Inline function with no call sites is removed from the program and
        the surviving caller body is left byte-for-byte unchanged.

        ``unused`` has no Call site, so the fixpoint loop never splices it (no
        ``any_changed``); the cleanup phase (src lines 596-603) drops it purely
        because ``func_type_ == Inline``. ``main`` carries no inline call, so it
        passes through verbatim — hence Expected is just ``main`` alone."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def unused(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
                return y

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)


class TestInlineFunctionsInDefaultPipeline:
    """Verify the pass is wired into the default pipeline at position 0."""

    def test_inline_runs_in_default_pipeline(self):
        """End-to-end: inline functions disappear after PassManager.Default runs."""

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                z: pl.Tensor[[1], pl.INT32] = self.helper(x)
                return z

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        After = pm.run_passes(P)
        names = [f.name for f in After.functions.values()]
        assert "helper" not in names


class TestInlineFunctionsEliminatedVerifier:
    """The PropertyVerifier catches surviving Inline functions / Calls."""

    def _make_property_set(self):
        ps = core_passes.IRPropertySet()
        ps.insert(core_passes.IRProperty.InlineFunctionsEliminated)
        return ps

    def test_verifier_flags_surviving_inline_function(self):
        """If an Inline function survives, the verifier reports an error."""

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                z: pl.Tensor[[1], pl.INT32] = self.helper(x)
                return z

        # Don't run the inline pass — feed P directly to the verifier.
        ps = self._make_property_set()
        diagnostics = core_passes.PropertyVerifierRegistry.verify(ps, P)
        errors = [d for d in diagnostics if d.severity == core_passes.DiagnosticSeverity.Error]
        # Expect at least: 1 error for the surviving Inline function, 1 for the Call.
        assert len(errors) >= 2, (
            f"Expected verifier to flag survivors, got {[(d.severity, d.message) for d in diagnostics]}"
        )
        messages = " | ".join(d.message for d in errors)
        assert "helper" in messages

    def test_verifier_silent_after_inline_pass(self):
        """After inline_functions(), the verifier produces no errors."""

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                z: pl.Tensor[[1], pl.INT32] = self.helper(x)
                return z

        After = passes.inline_functions()(P)
        ps = self._make_property_set()
        diagnostics = core_passes.PropertyVerifierRegistry.verify(ps, After)
        errors = [d for d in diagnostics if d.severity == core_passes.DiagnosticSeverity.Error]
        assert errors == [], f"Verifier should be silent post-pass, got {[d.message for d in errors]}"


class TestInlineFunctionsParamRebinding:
    """Regression coverage for issue #1281.

    An ``@pl.jit.inline`` callee that rebinds one of its ``pl.Out`` parameters
    (the typical ``out = pl.tensor.assemble(out, ...)`` pattern) used to leave
    the LHS of the rebinding pointing at the callee's original param Var after
    splicing. The post-call alias was then synthesised from the substituted
    use-site and ended up as ``lhs = actual_arg`` instead of ``lhs = rebound``,
    which downstream codegen lowered to a self-referential ``auto X = X;`` in
    C++. With substitution carried into def-sites, the rebinding lands in the
    caller scope as ``actual_arg = pl.tensor.assemble(actual_arg, ...)`` and
    the post-call alias correctly plumbs the rebound value.
    """

    def test_single_callsite_pl_out_rebinding(self):
        """Param rebinding survives through to the caller's scope and the
        post-call alias is no longer a self-reference."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def proj(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                out = pl.tensor.assemble(out, x, [0])
                return out

            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                ext: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                v: pl.Tensor[[4], pl.FP32] = self.proj(a, ext)
                return v

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                ext: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                ext = pl.tensor.assemble(ext, a, [0])  # in-place rebinding of the pl.Out
                v: pl.Tensor[[4], pl.FP32] = ext  # alias to rebound value, NOT pre-call ext
                return v

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multi_callsite_distinct_rebindings(self):
        """Three call sites of an inline callee that rebinds its pl.Out param
        each emit an independent ``actual = assemble(actual, ...)`` rebinding
        of their own caller-side actual arg, never aliasing across sites."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def proj(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                out = pl.tensor.assemble(out, x, [0])
                return out

            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                qo: pl.Out[pl.Tensor[[4], pl.FP32]],
                ko: pl.Out[pl.Tensor[[4], pl.FP32]],
                vo: pl.Out[pl.Tensor[[4], pl.FP32]],
            ):
                q: pl.Tensor[[4], pl.FP32] = self.proj(a, qo)
                k: pl.Tensor[[4], pl.FP32] = self.proj(a, ko)
                v: pl.Tensor[[4], pl.FP32] = self.proj(a, vo)
                return q, k, v

        After = passes.inline_functions()(Before)

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                qo: pl.Out[pl.Tensor[[4], pl.FP32]],
                ko: pl.Out[pl.Tensor[[4], pl.FP32]],
                vo: pl.Out[pl.Tensor[[4], pl.FP32]],
            ):
                qo = pl.tensor.assemble(qo, a, [0])
                q: pl.Tensor[[4], pl.FP32] = qo
                ko = pl.tensor.assemble(ko, a, [0])
                k: pl.Tensor[[4], pl.FP32] = ko
                vo = pl.tensor.assemble(vo, a, [0])
                v: pl.Tensor[[4], pl.FP32] = vo
                return q, k, v

        ir.assert_structural_equal(After, Expected)


class TestInlineReturnAndMultiReturn:
    """`return inline_call(...)` and tuple-unpack of multi-return inline calls.

    Issue #1304 — these forms previously slipped through InlineFunctions because
    the mutator only handled ``AssignStmt`` and ``EvalStmt`` call sites and
    emitted dead ``LHS = MakeTuple(...)`` bindings for multi-return.
    """

    def test_return_inline_call_single_return(self):
        """`return inline_call(...)` with a single-return inline: body spliced,
        outer ReturnStmt rewritten to return the cloned return value directly."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def proj(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                out = pl.tensor.assemble(out, x, [0])
                return out

            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                qo: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                return self.proj(a, qo)

        After = passes.inline_functions()(Before)

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                qo: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                qo = pl.tensor.assemble(qo, a, [0])
                return qo

        ir.assert_structural_equal(After, Expected)

    def test_return_inline_call_multi_return(self):
        """`return inline_call(...)` with a multi-return inline: outer
        ReturnStmt rewritten to return the cloned values directly — no
        intermediate MakeTuple."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def proj(
                self,
                x: pl.Tensor[[4], pl.FP32],
                o0: pl.Out[pl.Tensor[[4], pl.FP32]],
                o1: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4], pl.FP32]]:
                o0 = pl.tensor.assemble(o0, x, [0])
                o1 = pl.tensor.assemble(o1, x, [0])
                return o0, o1

            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                q: pl.Out[pl.Tensor[[4], pl.FP32]],
                k: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4], pl.FP32]]:
                return self.proj(a, q, k)

        After = passes.inline_functions()(Before)

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                q: pl.Out[pl.Tensor[[4], pl.FP32]],
                k: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4], pl.FP32]]:
                q = pl.tensor.assemble(q, a, [0])
                k = pl.tensor.assemble(k, a, [0])
                return q, k

        ir.assert_structural_equal(After, Expected)

    def test_tuple_unpack_inline_call_multi_return(self):
        """`y0, y1 = inline_call(...)` — multi-return inline call site
        substitutes TupleGetItemExpr uses with the return values, leaves no
        live MakeTuple binding."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def proj(
                self,
                x: pl.Tensor[[4], pl.FP32],
                o0: pl.Out[pl.Tensor[[4], pl.FP32]],
                o1: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4], pl.FP32]]:
                o0 = pl.tensor.assemble(o0, x, [0])
                o1 = pl.tensor.assemble(o1, x, [0])
                return o0, o1

            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                q: pl.Out[pl.Tensor[[4], pl.FP32]],
                k: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4], pl.FP32]]:
                y0, y1 = self.proj(a, q, k)
                return y0, y1

        After = passes.inline_functions()(Before)

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                q: pl.Out[pl.Tensor[[4], pl.FP32]],
                k: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4], pl.FP32]]:
                q = pl.tensor.assemble(q, a, [0])
                k = pl.tensor.assemble(k, a, [0])
                y0: pl.Tensor[[4], pl.FP32] = q
                y1: pl.Tensor[[4], pl.FP32] = k
                return y0, y1

        ir.assert_structural_equal(After, Expected)

    def test_tuple_unpack_inline_call_returning_tuple_temporary(self):
        """A tuple temporary returned by an inline helper is expanded before
        tuple-get-item substitution, so no MakeTuple reaches codegen."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def proj(
                self,
                x: pl.Tensor[[4], pl.FP32],
                o0: pl.Out[pl.Tensor[[4], pl.FP32]],
                o1: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4], pl.FP32]]:
                o0 = pl.tensor.assemble(o0, x, [0])
                o1 = pl.tensor.assemble(o1, x, [0])
                tmp = (o0, o1)
                return tmp

            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                q: pl.Out[pl.Tensor[[4], pl.FP32]],
                k: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4], pl.FP32]]:
                y0, y1 = self.proj(a, q, k)
                return y0, y1

        After = passes.inline_functions()(Before)

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                q: pl.Out[pl.Tensor[[4], pl.FP32]],
                k: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4], pl.FP32]]:
                q = pl.tensor.assemble(q, a, [0])
                k = pl.tensor.assemble(k, a, [0])
                y0: pl.Tensor[[4], pl.FP32] = q
                y1: pl.Tensor[[4], pl.FP32] = k
                return y0, y1

        ir.assert_structural_equal(After, Expected)

    def test_inline_with_bare_tensor_params_multi_return(self):
        """Bare `pl.Tensor` inline params (no `pl.Out` wrapper) splice the
        same way as `pl.Out`-annotated params: rebindings retarget the
        actual-arg Var and tuple-unpack uses get substituted directly.

        Issue #1304 deprecates `pl.Out` on `@pl.jit.inline` helpers — this
        test pins the equivalent behavior at the IR level."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def proj(
                self,
                x: pl.Tensor[[4], pl.FP32],
                o0: pl.Tensor[[4], pl.FP32],
                o1: pl.Tensor[[4], pl.FP32],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4], pl.FP32]]:
                o0 = pl.tensor.assemble(o0, x, [0])
                o1 = pl.tensor.assemble(o1, x, [0])
                return o0, o1

            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                q: pl.Out[pl.Tensor[[4], pl.FP32]],
                k: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4], pl.FP32]]:
                y0, y1 = self.proj(a, q, k)
                return y0, y1

        After = passes.inline_functions()(Before)

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                q: pl.Out[pl.Tensor[[4], pl.FP32]],
                k: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4], pl.FP32]]:
                q = pl.tensor.assemble(q, a, [0])
                k = pl.tensor.assemble(k, a, [0])
                y0: pl.Tensor[[4], pl.FP32] = q
                y1: pl.Tensor[[4], pl.FP32] = k
                return y0, y1

        ir.assert_structural_equal(After, Expected)


class TestInlineFunctionsSubmitCallSite:
    """Inline callee launched via ``pl.submit`` inside a ``pl.manual_scope``.

    InlineFunctions drops Inline functions unconditionally (``func_type_ ==
    Inline``). A ``pl.submit(self.helper, ...)`` of a dropped Inline function
    would therefore be left dangling. Per
    ``.claude/rules/pass-submit-awareness.md`` (rule 1: "When walking calls,
    walk Submit too"), the InlineFunctionsEliminated verifier is Submit-aware
    and flags such a dangling submit (fixed in #1615).
    """

    def test_submit_of_inline_eliminates_reference(self):
        """After the pass, no reference (Call OR Submit) to a dropped Inline
        function may survive — the documented ``InlineFunctionsEliminated``
        contract (doc §Verification: "No Call whose callee resolves to one
        survives"), extended to Submit per the submit-awareness rule.

        Regression test for #1615: the InlineFunctionsEliminated verifier is
        Submit-aware and reports an error for the surviving
        ``pl.submit(self.helper, ...)`` after ``helper`` is dropped. (Inlining a
        submit is not meaningful — the task launch / TASK_ID result would
        vanish — so flagging it loudly is the correct contract.)"""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, a_tid = pl.submit(self.helper, x)
                return a

        # inline_functions PRODUCES the InlineFunctionsEliminated property, so
        # with the now-Submit-aware verifier its own post-pass verification
        # throws on the dangling pl.submit. Run under VerificationLevel.NONE to
        # obtain `After` and inspect the diagnostics explicitly below (the throw
        # path is itself the correct loud-failure behavior).
        with passes.PassContext([], passes.VerificationLevel.NONE):
            After = passes.inline_functions()(Before)

        # The Inline function `helper` is dropped, so any surviving reference to
        # it (here a Submit) is a dangling reference and must be reported by the
        # InlineFunctionsEliminated verifier.
        ps = core_passes.IRPropertySet()
        ps.insert(core_passes.IRProperty.InlineFunctionsEliminated)
        diagnostics = core_passes.PropertyVerifierRegistry.verify(ps, After)
        errors = [d for d in diagnostics if d.severity == core_passes.DiagnosticSeverity.Error]
        assert errors, (
            "Expected the verifier to flag the surviving pl.submit(self.helper, ...) "
            "after `helper` was dropped, but it reported no errors."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
