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

import pypto
import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir import OptimizationStrategy, PassManager
from pypto.pypto_core import passes as core_passes
from pypto.runtime import RunConfig


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

        ``SpliceInlineCallAsEval`` calls ``CloneInlineBody`` and keeps
        ``body.stmts``; the trailing ``return out`` value is discarded because
        there is no LHS to bind it to *and* a bare ``Var`` produces a value and
        nothing else — dropping it loses no side effect (contrast
        ``test_eval_stmt_call_site_preserves_nested_write`` below, where the
        discarded value is a cross-function Call). The in-place rebinding
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

    def test_eval_stmt_call_site_preserves_nested_write(self):
        """Regression test for #2705: an ignored wrapper whose body is
        ``return self.writeout(x, out)`` must still perform ``writeout``'s write.

        Dropping a return VALUE is not the same as dropping its EVALUATION.
        ``CloneInlineBody`` strips the trailing ReturnStmt's expression into
        ``return_values``, so before the fix ``SpliceInlineCallAsEval`` returned
        an empty ``body.stmts`` for ``forward`` and the nested cross-function
        Call — together with its write through the ``pl.Out`` arg — vanished
        without a diagnostic. ``main`` collapsed to a bare ``return x``.

        Now the discarded ``Call(writeout, x, out)`` is re-emitted as an
        EvalStmt, which the fixpoint loop expands on the next iteration into
        ``writeout``'s own spliced body — the assemble below."""

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

            @pl.function(type=pl.FunctionType.Inline)
            def forward(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                return self.writeout(x, out)

            @pl.function
            def main(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                self.forward(x, out)  # EvalStmt call site — return value ignored
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                out = pl.tensor.assemble(out, x, [0])
                return x

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_eval_stmt_call_site_keeps_non_inline_dispatch(self):
        """Same shape as above, but the wrapper forwards to a NON-Inline
        function: the re-emitted EvalStmt stays an ordinary cross-function
        dispatch, exactly as if the author had written ``self.inner(...)`` at the
        call site. Nothing in the fixpoint loop expands it (``inner`` is not
        Inline), and nothing may delete it either — ``inner`` writes ``out``."""

        @pl.program
        class Before:
            @pl.function
            def inner(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                out = pl.tensor.assemble(out, x, [0])
                return out

            @pl.function(type=pl.FunctionType.Inline)
            def forward(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                return self.inner(x, out)

            @pl.function
            def main(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                self.forward(x, out)
                return x

        @pl.program
        class Expected:
            @pl.function
            def inner(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                out = pl.tensor.assemble(out, x, [0])
                return out

            @pl.function
            def main(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                self.inner(x, out)
                return x

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_eval_stmt_call_site_preserves_returned_builtin_store(self):
        """An ignored wrapper that directly returns an effectful *builtin* keeps
        its write too — the callee does not have to be a cross-function call.

        ``return pl.tile.store(t, [0, 0], out)`` is a supported return form, and
        ``tile.store`` declares argument 2 as ``ArgEffect::Write`` /
        ``ReadWrite`` (``src/ir/op/tile_ops/memory.cpp``). The write lives in the
        return expression itself, not in a preceding ``AssignStmt``, so the
        discarded-value classification has to consult the operator registry
        rather than assume every builtin call is a pure value."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def writeout(
                self,
                t: pl.Tile[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                return pl.tile.store(t, [0, 0], out)

            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(a, [0, 0], [64, 64])
                self.writeout(t, out)  # EvalStmt call site — return value ignored
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(a, [0, 0], [64, 64])
                pl.tile.store(t, [0, 0], out)
                return out

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_eval_stmt_call_site_keeps_returned_sync_op(self):
        """An operator that writes through no argument is still not deletable.

        ``system.set_ffts`` declares ``no_arg_writes()`` — it moves no data —
        yet it hands the FFTS unit its workspace pointer, and the same holds for
        ``pld.system.wait`` (blocks on a signal threshold) and
        ``pld.system.defer_wait`` (registers a completion condition). "Writes
        through no argument" is not "safe to delete", so the discarded-value
        classification cannot key deletion on
        ``OpRegistryEntry::WritesAnyArg``; the pass keeps every call until a real
        deletability property exists."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def setup(self, ws: pl.Tensor[[256], pl.INT64]) -> pl.Tensor[[256], pl.INT64]:
                # The DSL return annotation is parser metadata for the IR, while
                # the Python surface types `set_ffts` as `-> Call`; the two never
                # meet at runtime because the body is parsed, not executed.
                return pl.system.set_ffts(ws)  # type: ignore[reportReturnType]

            @pl.function(type=pl.FunctionType.AIV)
            def main(
                self,
                ws: pl.Tensor[[256], pl.INT64],
                x: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                self.setup(ws)  # EvalStmt call site — return value ignored
                return x

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV)
            def main(
                self,
                ws: pl.Tensor[[256], pl.INT64],
                x: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                pl.system.set_ffts(ws)
                return x

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_eval_stmt_call_site_keeps_side_effect_free_builtin(self):
        """Even a builtin that happens to be pure is kept, because nothing in the
        IR can prove it.

        No operator property answers "is this call safe to delete".
        ``OpRegistryEntry::WritesAnyArg`` answers a narrower question and is
        wrong in both directions here: 263 of 315 operators are unclassified
        (among them ``tile.tpush_to_aiv`` and ``system.aic_initialize_pipe``,
        which ``dce::IsSideEffectOp`` lists as side-effecting), and a positive
        ``no_arg_writes()`` verdict covers synchronization ops as well — see
        ``test_eval_stmt_call_site_keeps_returned_sync_op``. So the pass keeps
        every call. ``tensor.add`` is genuinely pure and its `EvalStmt` is dead
        but harmless; that is the deliberate cost of not guessing."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def compute(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                return pl.add(x, x)

            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                self.compute(a)  # EvalStmt call site — return value ignored
                return a

        @pl.program
        class Expected:
            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                pl.add(a, a)
                return a

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_eval_stmt_call_site_keeps_call_wrapping_a_call(self):
        """A discarded builtin op Call that *wraps* a cross-function call is kept
        whole, so the nested call's write rides along inside it.

        ``pl.add`` is an unclassified operator, so the outer Call is preserved by
        the same rule as any other non-provably-pure call — and preserving it
        preserves the nested ``Call(inner, x, out)`` verbatim. Nothing is
        deleted and nothing has to be rejected."""

        @pl.program
        class Before:
            @pl.function
            def inner(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                out = pl.tensor.assemble(out, x, [0])
                return out

            @pl.function(type=pl.FunctionType.Inline)
            def forward(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                return pl.add(self.inner(x, out), x)

            @pl.function
            def main(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                self.forward(x, out)
                return x

        @pl.program
        class Expected:
            @pl.function
            def inner(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                out = pl.tensor.assemble(out, x, [0])
                return out

            @pl.function
            def main(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                pl.add(self.inner(x, out), x)
                return x

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_eval_stmt_call_site_wrapping_a_call_errors(self):
        """A discarded return value that is *not itself call-like* but hides a
        call cannot keep that call's evaluation, so the pass raises instead of
        deleting it.

        Scalar arithmetic is a dedicated Expr kind (`Add`), not a `Call`, so
        ``self.bump(n) + 1`` cannot be re-emitted as an `EvalStmt` the way a
        call-like value can. Dropping it would drop the nested
        ``Call(bump, n)`` with no verifier to catch it (``bump`` is not Inline,
        so ``InlineFunctionsEliminated`` sees nothing wrong). Reject it loudly
        instead; the message names both workarounds."""

        @pl.program
        class Before:
            @pl.function
            def bump(self, n: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                m: pl.Scalar[pl.INDEX] = n + 1
                return m

            @pl.function(type=pl.FunctionType.Inline)
            def forward(self, n: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                return self.bump(n) + 1

            @pl.function
            def main(
                self,
                x: pl.Tensor[[4], pl.FP32],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4], pl.FP32]:
                self.forward(n)
                return x

        with pytest.raises(ValueError, match="wraps a call"):
            passes.inline_functions()(Before)

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


class TestInlineFunctionsDumpMarks:
    """Selective-dump marks (``dump_vars``) on the scopes an Inline body splices in."""

    def test_helper_tag_on_cluster_follows_param_substitution(self):
        """A helper-local ``pl.dump_tag(x)`` lands on its ``pl.cluster`` scope as
        well as on the inner ``pl.at`` carrier; splicing must rename ``x`` to the
        caller's arg on both, or the Cluster's mark names a Var the caller never
        binds and the outliner silently drops it."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.dump_tag(x)
                with pl.cluster():
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
                with pl.cluster(dumps=[a]):
                    with pl.at(level=pl.Level.CORE_GROUP, dumps=[a]):
                        y_inline: pl.Tensor[[64], pl.FP32] = pl.add(a, a)
                r: pl.Tensor[[64], pl.FP32] = y_inline
                return r

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_helper_tag_on_graph_follows_param_substitution(self):
        """Same as the Cluster case for a ``pl.graph`` region, whose mark marks
        the graph task itself."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.dump_tag(x)
                with pl.graph("g"):
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
                with pl.graph("g", dumps=[a]):
                    with pl.at(level=pl.Level.CORE_GROUP, dumps=[a]):
                        y_inline: pl.Tensor[[64], pl.FP32] = pl.add(a, a)
                r: pl.Tensor[[64], pl.FP32] = y_inline
                return r

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_call_site_tag_skips_split_aiv_region(self):
        """A call-site tag is transferred onto the spliced ``pl.at`` carrier but
        not onto the ``pl.split_aiv`` region inside it: that region is never
        outlined into a dispatch, and ``pl.split_aiv`` has no ``dumps=`` to print
        a mark as."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(
                self, x: pl.Tensor[[128, 128], pl.FP32], out: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    for aiv in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        t: pl.Tile[[64, 128], pl.FP32] = pl.load(x, [aiv * 64, 0], [64, 128])
                        out = pl.store(t, [aiv * 64, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, a: pl.Tensor[[128, 128], pl.FP32], out: pl.Out[pl.Tensor[[128, 128], pl.FP32]]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                pl.dump_tag(a)
                out = self.helper(a, out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, a: pl.Tensor[[128, 128], pl.FP32], out: pl.Out[pl.Tensor[[128, 128], pl.FP32]]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, dumps=[a]):
                    for aiv in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        t: pl.Tile[[64, 128], pl.FP32] = pl.load(a, [aiv * 64, 0], [64, 128])
                        out = pl.store(t, [aiv * 64, 0], out)
                return out

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


class TestInlineFunctionsNestedCallSites:
    """A call in a nested expression position is hoisted, then spliced.

    `HandleTopLevelInlineCall` only recognises a Call that *is* the whole
    statement value, but the pass drops every Inline function regardless — so a
    nested call used to survive as a reference to a deleted function and fail
    only at `GenerateOrchestration preconditions`. Each case below is the shape
    the parser produces from ordinary DSL.
    """

    def test_nested_in_call_argument(self):
        """`arr[i] = helper(x)` desugars to `array.update_element(arr, i, helper(x))`."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
                return y

            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                z: pl.Tensor[[1], pl.INT32] = pl.add(self.helper(a), a)
                return z

        @pl.program
        class Expected:
            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y_inline: pl.Tensor[[1], pl.INT32] = pl.mul(a, a)
                t_arg: pl.Tensor[[1], pl.INT32] = y_inline
                z: pl.Tensor[[1], pl.INT32] = pl.add(t_arg, a)
                return z

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_nested_in_binary_operand(self):
        """`k = helper(a) + a` — the call is an operand, not the whole value."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def half(self, n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
                m: pl.Scalar[pl.INT32] = n // 2
                return m

            @pl.function
            def main(self, n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
                k: pl.Scalar[pl.INT32] = self.half(n) + 1
                return k

        @pl.program
        class Expected:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
                m_inline: pl.Scalar[pl.INT32] = n // 2
                t_arg: pl.Scalar[pl.INT32] = m_inline
                k: pl.Scalar[pl.INT32] = t_arg + 1
                return k

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_inline_call_as_argument_of_inline_call(self):
        """`half(half(n))` — the pass's transitive-inline feature, written as one expression."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def half(self, n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
                m: pl.Scalar[pl.INT32] = n // 2
                return m

            @pl.function
            def main(self, n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
                k: pl.Scalar[pl.INT32] = self.half(self.half(n))
                return k

        After = passes.inline_functions()(Before)
        printed = ir.python_print(After)
        assert "half" not in printed, printed
        assert printed.count("// 2") == 2, printed

    def test_nested_in_loop_bound(self):
        """A loop bound is evaluated once before the loop, so hoisting it is semantics-preserving."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def half(self, n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
                m: pl.Scalar[pl.INT32] = n // 2
                return m

            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32], n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
                for _i in pl.range(self.half(n)):
                    _t: pl.Tensor[[1], pl.INT32] = pl.mul(a, a)
                return n

        After = passes.inline_functions()(Before)
        printed = ir.python_print(After)
        assert "half" not in printed, printed
        # The hoisted bound precedes the loop it feeds.
        assert printed.index("// 2") < printed.index("pl.range"), printed

    def test_top_level_call_site_gains_no_temporary(self):
        """`z = helper(a)` already splices; hoisting must not add a redundant copy."""

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

    def test_tuple_returning_callee_is_not_hoisted(self):
        """A tuple-returning callee must stay put — hoisting it leaves an undefined temp.

        `SpliceInlineCallAsTupleSub` deliberately emits no `tmp = ...` binding: it
        records the cloned return values against the LHS `Var` and rewrites
        downstream `TupleGetItemExpr(tmp, i)` uses instead. A nested consumer holds
        `tmp` itself, not a `TupleGetItemExpr`, so a hoist would print
        `t__inline_arg_v0__FREE_VAR`. Leaving the Call in place keeps the pre-hoist
        behaviour, and `InlineFunctionsEliminated` reports it.

        Written as source text because the IR shape — a multi-value `ReturnStmt`
        whose first value is a tuple — has no Python-typeable annotation: the DSL
        rejects a nested `tuple[...]` return, and a flat one contradicts what the
        callee returns.
        """
        source = """
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.Inline)
    def pair(self, x: pl.Tensor[[1], pl.INT32]) -> tuple[pl.Tensor[[1], pl.INT32], pl.Tensor[[1], pl.INT32]]:
        a: pl.Tensor[[1], pl.INT32] = pl.tensor.mul(x, x)
        b: pl.Tensor[[1], pl.INT32] = pl.tensor.add(x, x)
        return a, b

    @pl.function
    def main(self, x: pl.Tensor[[1], pl.INT32]) -> tuple[
            pl.Tensor[[1], pl.INT32], pl.Tensor[[1], pl.INT32], pl.Tensor[[1], pl.INT32]]:
        y: pl.Tensor[[1], pl.INT32] = pl.tensor.mul(x, x)
        return self.pair(x), y
"""
        Before = pl.parse_program(source)

        # InlineFunctionsEliminated is in GetVerifiedProperties(), so the
        # un-spliced Call is reported by the pass's own post-verification — at
        # the `return self.pair(x), y` line, not 40 passes later in codegen.
        with pytest.raises(pypto.Error, match=r"Dangling Call to function 'pair'"):
            passes.inline_functions()(Before)

        # The IR itself is intact: the Call stays where it was, and no temp is
        # left dangling (`t__inline_arg_vN__FREE_VAR`).
        with passes.PassContext([], passes.VerificationLevel.NONE):
            After = passes.inline_functions()(Before)
        printed = ir.python_print(After)
        assert "FREE_VAR" not in printed, printed
        assert "self.pair(x)" in printed, printed

    def test_tuple_unpack_call_site_still_splices(self):
        """The ordinary `a, b = self.pair(x)` form is unaffected by the hoist skip."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def pair(
                self, x: pl.Tensor[[1], pl.INT32]
            ) -> tuple[pl.Tensor[[1], pl.INT32], pl.Tensor[[1], pl.INT32]]:
                a: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
                b: pl.Tensor[[1], pl.INT32] = pl.add(x, x)
                return a, b

            @pl.function
            def main(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                p, q = self.pair(x)
                z: pl.Tensor[[1], pl.INT32] = pl.add(p, q)
                return z

        After = passes.inline_functions()(Before)
        printed = ir.python_print(After)
        assert "pair" not in printed, printed
        assert "FREE_VAR" not in printed, printed

    def test_verifier_silent_after_nested_call_site(self):
        """The nested form leaves nothing for InlineFunctionsEliminated to report."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                y: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
                return y

            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                z: pl.Tensor[[1], pl.INT32] = pl.add(self.helper(a), a)
                return z

        After = passes.inline_functions()(Before)
        ps = core_passes.IRPropertySet()
        ps.insert(core_passes.IRProperty.InlineFunctionsEliminated)
        diagnostics = core_passes.PropertyVerifierRegistry.verify(ps, After)
        errors = [d for d in diagnostics if d.severity == core_passes.DiagnosticSeverity.Error]
        assert errors == [], f"Expected no survivors, got {[d.message for d in errors]}"


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

    def test_rebound_param_with_slice_arg_binds_temporary(self):
        """A rebound param whose actual arg is a slice is bound to a temporary.

        Substituting ``ext[r]`` for ``out`` at the rebinding's def-site would put
        a ``tensor.slice`` Call on the LHS of an AssignStmt, which used to fail
        with ``InternalError: AssignStmt var is not a Var after mutation``. The
        splice instead binds the slice to a fresh Var first, which is the IR the
        parser already produces when the caller names the slice itself.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def write0(self, x: pl.Tensor[[4], pl.FP32], out: pl.Tensor[[4], pl.FP32]):
                out = pl.tensor.assemble(out, x, [0])

            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                ext: pl.Out[pl.Tensor[[2, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 4], pl.FP32]:
                for r in pl.range(2):
                    self.write0(a, ext[r])
                return ext

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                ext: pl.Out[pl.Tensor[[2, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 4], pl.FP32]:
                for r in pl.range(2):
                    out: pl.Tensor[[4], pl.FP32] = ext[r]
                    out = pl.tensor.assemble(out, a, [0])
                return ext

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_rebound_param_with_iter_arg_binds_temporary(self):
        """An IterArg actual arg is not an assignable Var either, so a rebound
        param bound to one gets the same temporary instead of crashing."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def write0(self, x: pl.Tensor[[4], pl.FP32], out: pl.Tensor[[4], pl.FP32]):
                out = pl.tensor.assemble(out, x, [0])

            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                ext: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                for i, (acc,) in pl.range(2, init_values=(ext,)):
                    self.write0(a, acc)
                    acc = pl.yield_(acc)
                return ext

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                ext: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                for i, (acc,) in pl.range(2, init_values=(ext,)):
                    out: pl.Tensor[[4], pl.FP32] = acc
                    out = pl.tensor.assemble(out, a, [0])
                    acc = pl.yield_(acc)
                return ext

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_read_only_param_with_slice_arg_binds_temporary(self):
        """A computed tensor arg is evaluated once at the call site even when
        the callee only reads it.

        Substituting ``a[r]`` verbatim would re-evaluate the slice at every use
        inside the callee body, moving it into whatever ``pl.spmd`` /
        ``pl.pipeline`` / ``pl.at`` scope holds that use and therefore into the
        outlined kernel.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def write0(self, x: pl.Tensor[[4], pl.FP32], out: pl.Tensor[[4], pl.FP32]):
                out = pl.tensor.assemble(out, x, [0])

            @pl.function
            def main(
                self,
                a: pl.Tensor[[2, 4], pl.FP32],
                ext: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                for r in pl.range(2):
                    self.write0(a[r], ext)
                return ext

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[2, 4], pl.FP32],
                ext: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                for r in pl.range(2):
                    x: pl.Tensor[[4], pl.FP32] = a[r]
                    ext = pl.tensor.assemble(ext, x, [0])
                return ext

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_scalar_expression_arg_is_substituted_in_place(self):
        """A read-only scalar arg is not bound: only computed tensor / tile args
        are, so scalar expressions keep folding into the callee body."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def dbl(self, n: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                m: pl.Scalar[pl.INDEX] = n * 2
                return m

            @pl.function
            def main(self, k: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                y: pl.Scalar[pl.INDEX] = self.dbl(k + 1)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, k: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                m: pl.Scalar[pl.INDEX] = (k + 1) * 2
                y: pl.Scalar[pl.INDEX] = m
                return y

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_rebound_scalar_param_does_not_clobber_caller_arg(self):
        """A rebound pass-by-value scalar param binds a call-site temporary.

        ``n`` is an ``In`` scalar — Python pass-by-value, no in-place contract.
        Substituting the caller's ``k`` at the *def*-site would splice
        ``k = k + 1`` into the caller, so the later ``k + m`` would read the
        bumped value and compute ``2k + 2`` instead of ``2k + 1``."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def bump(self, n: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                n = n + 1
                return n

            @pl.function
            def main(self, k: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                m: pl.Scalar[pl.INDEX] = self.bump(k)
                s: pl.Scalar[pl.INDEX] = k + m
                return s

        @pl.program
        class Expected:
            @pl.function
            def main(self, k: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                n_inline0: pl.Scalar[pl.INDEX] = k  # call-site temporary
                n_inline0 = n_inline0 + 1  # rebinding stays callee-local
                m: pl.Scalar[pl.INDEX] = n_inline0
                s: pl.Scalar[pl.INDEX] = k + m  # reads the ORIGINAL k
                return s

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_rebound_tensor_param_still_aliases_caller_arg(self):
        """A rebound tensor param keeps aliasing the caller's Var.

        Guards the counterpart of the test above: an inline callee's shaped
        params *are* in-place aliases of the caller's handles — that is what
        lets ``c[...] = v`` write through — so ``@pl.jit.inline`` strips
        ``pl.Out`` from them. Direction alone therefore cannot decide, and a
        tensor param must not gain a call-site temporary."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def fill(self, x: pl.Tensor[[4], pl.FP32], c: pl.Tensor[[4], pl.FP32]) -> pl.Tensor[[4], pl.FP32]:
                c = pl.tensor.assemble(c, x, [0])
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                ext: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                v: pl.Tensor[[4], pl.FP32] = self.fill(a, ext)
                return v

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[4], pl.FP32],
                ext: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                ext = pl.tensor.assemble(ext, a, [0])  # no temporary: writes through
                v: pl.Tensor[[4], pl.FP32] = ext
                return v

        After = passes.inline_functions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_rebound_array_param_gains_no_bare_alias(self):
        """A rebound ``pl.Array`` param keeps aliasing the caller's Var.

        An Array is a handle like a tensor: ``a[i] = v`` parses as
        ``a = pl.array.update_element(a, i, v)``, so the rebinding *is* the
        update and the caller must see it. Binding a call-site temporary would
        emit a bare ``vals_inline0 = vals`` alias, which orchestration codegen
        cannot declare — an array Var comes from ``array.create`` or an alias
        onto a backing array, never from its type, so it aborts with
        ``GetCppType called for ArrayType``."""

        @pl.jit.inline
        def fill(vals: pl.Array[2, pl.INDEX], base: pl.Scalar[pl.INDEX]):
            for i in pl.range(2):
                vals[i] = base + i

        @pl.jit
        def drv(
            x: pl.Tensor[[64], pl.FP32],
            k: pl.Scalar[pl.INDEX],
            o: pl.Out[pl.Tensor[[64], pl.FP32]],
        ):
            vals = pl.array.create(2, pl.INDEX)
            fill(vals, k)
            o[0:64] = pl.add(x[0:64], x[0:64])
            return o

        # An Inline function with an Array param trips the ArrayNotEscaped
        # verifier on the *input* program. The real pipeline never sees it —
        # InlineFunctions runs first and splices the callee away before any
        # verification — so run the pass the same way here.
        with core_passes.PassContext([], core_passes.VerificationLevel.NONE):
            spliced = passes.inline_functions()(drv.specialize())
        orch = next(f for f in spliced.functions.values() if f.func_type == ir.FunctionType.Orchestration)

        class BareArrayAliasCollector(ir.IRVisitor):
            def __init__(self):
                super().__init__()
                self.aliases: list[str] = []

            def visit_assign_stmt(self, op):
                if isinstance(op.value, ir.Var) and isinstance(op.value.type, ir.ArrayType):
                    self.aliases.append(f"{op.var.name_hint} = {op.value.name_hint}")
                super().visit_assign_stmt(op)

        collector = BareArrayAliasCollector()
        collector.visit_function(orch)
        assert collector.aliases == [], (
            f"InlineFunctions emitted a bare array alias codegen cannot declare: {collector.aliases}"
        )

    def test_jit_inline_loop_carried_scalar_rebind_keeps_caller_arg(self):
        """End-to-end: a ``@pl.jit.inline`` scalar rebind inside a loop.

        The JIT specializer alpha-renames a rebinding only at the scope depth
        where the name was first bound, so a rebind inside ``pl.range`` reaches
        InlineFunctions as a genuine def-site of the param. The loop carry must
        run over a call-site temporary, leaving the caller's ``k`` readable
        afterwards (``s == k + (k + 6)``, not ``(k + 6) * 2``)."""

        @pl.jit.inline
        def acc(n: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
            for i in pl.range(4):
                n = n + i
            return n

        @pl.jit
        def drv(
            a: pl.Tensor[[256], pl.FP32],
            k: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ):
            m = acc(k)
            s = k + m
            out[0:64] = pl.add(a[s : s + 64], a[s : s + 64])
            return out

        spliced = passes.inline_functions()(drv.specialize())
        orch = next(f for f in spliced.functions.values() if f.func_type == ir.FunctionType.Orchestration)
        caller_k = orch.params[1]

        # The loop must rebind a temporary, never the caller's `k` param.
        class LoopCarryCollector(ir.IRVisitor):
            def __init__(self):
                super().__init__()
                self.rebound: list[ir.Var] = []

            def visit_assign_stmt(self, op):
                self.rebound.append(op.var)
                super().visit_assign_stmt(op)

        collector = LoopCarryCollector()
        collector.visit_function(orch)
        rebound_ids = {v.unique_id for v in collector.rebound}
        assert caller_k.unique_id not in rebound_ids, (
            "InlineFunctions spliced the callee's rebinding onto the caller's `k`"
        )

    def test_jit_subscript_write_through_sliced_arg_lowers(self):
        """End-to-end repro: a ``@pl.jit.inline`` helper writing ``c[...] = v``
        into a tensor param, called with ``c[r]``, lowers through the default
        pipeline and still writes into the caller's ``c``."""

        @pl.jit.inline
        def add_into(x: pl.Tensor[[64, 128], pl.FP32], c: pl.Tensor[[64, 128], pl.FP32]):
            for i in pl.spmd(2):
                c0 = i * 64
                c[0:64, c0 : c0 + 64] = pl.add(x[0:64, c0 : c0 + 64], x[0:64, c0 : c0 + 64])

        @pl.jit
        def drv(a: pl.Tensor[[2, 64, 128], pl.FP32], c: pl.Out[pl.Tensor[[2, 64, 128], pl.FP32]]):
            for r in pl.range(2):
                add_into(a[r], c[r])
            return c

        # tests/ut/ir/transforms/conftest.py pins the Ascend950 backend.
        lowered = drv.lower(config=RunConfig(platform="a5"))
        assert ir.FunctionType.Inline not in {f.func_type for f in lowered.functions.values()}

        class DispatchCollector(ir.IRVisitor):
            def __init__(self):
                super().__init__()
                self.calls: list[ir.Call] = []

            def visit_call(self, op):
                if isinstance(op.op, ir.GlobalVar):
                    self.calls.append(op)
                super().visit_call(op)

        orch = next(f for f in lowered.functions.values() if f.func_type == ir.FunctionType.Orchestration)
        collector = DispatchCollector()
        collector.visit_function(orch)
        assert len(collector.calls) == 1
        # The kernel's written operand is the bound slice: a view into c's buffer.
        written_type = collector.calls[0].args[-1].type
        c_type = orch.params[1].type
        assert isinstance(written_type, ir.TensorType) and isinstance(c_type, ir.TensorType)
        assert written_type.memref is not None and c_type.memref is not None
        assert written_type.memref.base_.unique_id == c_type.memref.base_.unique_id


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


class TestInlineFunctionsReservedDelimiter:
    """A local whose name ends in `_` must not fuse with the `_inlineN` suffix.

    `assert_structural_equal` compares under alpha-equivalence, so it cannot see
    a name-shape regression. These assert the produced name text directly, and
    that the first downstream renamer (ConvertToSSA, the first pass to re-derive
    an auto-name from a Var) accepts the result.
    """

    def test_underscore_local_does_not_fuse(self):
        """`_` is Python's throwaway name — inlining must not yield `__inlineN`."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                _: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
                return _

            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                z: pl.Tensor[[1], pl.INT32] = self.helper(a)
                return z

        After = passes.inline_functions()(Before)
        printed = ir.python_print(After)
        assert "__" not in printed, printed
        passes.convert_to_ssa()(After)  # must not raise

    def test_trailing_underscore_local_does_not_fuse(self):
        """Any base ending in `_`, not only the bare `_`."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                t_: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
                return t_

            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                z: pl.Tensor[[1], pl.INT32] = self.helper(a)
                return z

        After = passes.inline_functions()(Before)
        printed = ir.python_print(After)
        assert "__" not in printed, printed
        assert "t_inline" in printed, printed
        passes.convert_to_ssa()(After)  # must not raise

    def test_author_written_double_underscore_still_rejected(self):
        """Inlining must not launder a base that was already invalid.

        Trimming the tail is about not *creating* the reserved delimiter; a name
        the author wrote with `__` in it stays a user-facing error (see
        test_convert_to_ssa_pass.py::test_reserved_auto_name_delimiter_in_base_raises).
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                a__b: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
                return a__b

            @pl.function
            def main(self, a: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                z: pl.Tensor[[1], pl.INT32] = self.helper(a)
                return z

        After = passes.inline_functions()(Before)
        with pytest.raises(ValueError, match="reserved delimiter '__'"):
            passes.convert_to_ssa()(After)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
