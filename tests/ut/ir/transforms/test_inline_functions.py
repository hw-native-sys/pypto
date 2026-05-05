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
        """An Inline body containing ``for i in pl.range(...)`` splices the loop intact."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                acc: pl.Tensor[[1], pl.INT32] = x
                for i in pl.range(4):
                    acc = pl.add(acc, x)
                return acc

            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                r: pl.Tensor[[1], pl.INT32] = self.helper(a)
                return r

        After = passes.inline_functions()(Before)
        # helper is gone; main contains the spliced loop.
        names = [f.name for f in After.functions.values()]
        assert names == ["main"]


class TestInlineFunctionsDeadCode:
    """Inline functions with no callers."""

    def test_no_callers_silently_dropped(self):
        """An Inline function with no call sites is removed from the program."""

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

        After = passes.inline_functions()(Before)
        names = [f.name for f in After.functions.values()]
        assert names == ["main"]


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
