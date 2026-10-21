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
from pypto import backend, codegen, ir, passes
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

        Nested inline expansion preserves ``writeout``'s spliced body — the
        assemble below — even when the outer helper's result is discarded."""

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
        call site. The inliner does not expand it (``inner`` is not
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
    """Inline calls Inline: recursive cloning preserves transitive calls."""

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


class TestInlineTileArgumentContracts:
    """Validate declared Tile invariants before replacing formals with actuals."""

    @pytest.mark.parametrize("discard_return", [False, True])
    @pytest.mark.parametrize(
        "shape, dtype, space, tensor, mismatch",
        [
            ([16, 64], pl.FP32, ir.MemorySpace.Vec, False, "physical shape dimension"),
            ([8, 64], pl.BF16, ir.MemorySpace.Vec, False, "dtype"),
            ([8, 64], pl.FP32, ir.MemorySpace.Acc, False, "memory space"),
            ([8, 64, 1], pl.FP32, ir.MemorySpace.Vec, False, "physical rank"),
            ([8, 64], pl.FP32, None, True, "expected Tile"),
        ],
        ids=["shape", "dtype", "memory-space", "rank", "tensor"],
    )
    def test_rejects_conflicting_actual_argument(self, shape, dtype, space, tensor, mismatch, discard_return):
        """Direct IR and discarded returns must not bypass the parameter contract."""
        span = ir.Span.unknown()
        formal_type = ir.TileType([8, 64], pl.FP32, memory_space=ir.MemorySpace.Vec)
        actual_type = ir.TensorType(shape, dtype) if tensor else ir.TileType(shape, dtype, memory_space=space)
        formal = ir.Var("tile", formal_type, span)
        actual = ir.Var("source", actual_type, span)
        helper = ir.Function(
            "keep_tile", [formal], [formal_type], ir.ReturnStmt([formal], span), span, ir.FunctionType.Inline
        )
        call = ir.Call(ir.GlobalVar("keep_tile"), [actual], formal_type, span)
        if discard_return:
            body = ir.SeqStmts([ir.EvalStmt(call, span), ir.ReturnStmt([], span)], span)
        else:
            result = ir.Var("result", formal_type, span)
            body = ir.SeqStmts([ir.AssignStmt(result, call, span), ir.ReturnStmt([result], span)], span)
        main = ir.Function("main", [actual], [], body, span)
        before = ir.Program([helper, main], "Before", span)

        with pytest.raises(ValueError, match=f"Inline call to 'keep_tile' parameter 'tile': .*{mismatch}"):
            passes.inline_functions()(before)

    def test_return_stmt_call_checks_declared_shape(self):
        """Returning an inline call directly must validate before substituting its result."""
        span = ir.Span.unknown()
        formal_type = ir.TileType([8, 64], pl.FP32, memory_space=ir.MemorySpace.Vec)
        actual_type = ir.TileType([16, 64], pl.FP32, memory_space=ir.MemorySpace.Vec)
        formal = ir.Var("tile", formal_type, span)
        source = ir.Var("source", actual_type, span)
        helper = ir.Function(
            "keep_tile", [formal], [formal_type], ir.ReturnStmt([formal], span), span, ir.FunctionType.Inline
        )
        call = ir.Call(ir.GlobalVar("keep_tile"), [source], formal_type, span)
        main = ir.Function("main", [source], [formal_type], ir.ReturnStmt([call], span), span)
        before = ir.Program([helper, main], "Before", span)

        with pytest.raises(
            ValueError, match="Inline call to 'keep_tile' parameter 'tile': .*physical shape dimension"
        ):
            passes.inline_functions()(before)

    @pytest.mark.parametrize("rows", [8, 16], ids=["matching", "conflicting"])
    def test_scalar_parameter_controls_physical_shape(self, rows):
        """An explicit scalar argument is authoritative, rather than a free shape placeholder."""
        span = ir.Span.unknown()
        scalar = ir.Var("rows", ir.ScalarType(pl.INDEX), span)
        formal_type = ir.TileType([scalar, 64], pl.FP32, memory_space=ir.MemorySpace.Vec)
        actual_type = ir.TileType([8, 64], pl.FP32, memory_space=ir.MemorySpace.Vec)
        formal = ir.Var("tile", formal_type, span)
        source = ir.Var("source", actual_type, span)
        helper = ir.Function(
            "keep_tile",
            [scalar, formal],
            [formal_type],
            ir.ReturnStmt([formal], span),
            span,
            ir.FunctionType.Inline,
        )
        call = ir.Call(
            ir.GlobalVar("keep_tile"), [ir.ConstInt(rows, pl.INDEX, span), source], actual_type, span
        )
        result = ir.Var("result", actual_type, span)
        body = ir.SeqStmts([ir.AssignStmt(result, call, span), ir.ReturnStmt([result], span)], span)
        before = ir.Program([helper, ir.Function("main", [source], [], body, span)], "Before", span)

        if rows != 8:
            with pytest.raises(
                ValueError, match="Inline call to 'keep_tile' parameter 'tile': .*physical shape dimension"
            ):
                passes.inline_functions()(before)
            return

        @pl.program
        class Expected:
            @pl.function
            def main(self, source: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec]):
                result = source
                return result

        ir.assert_structural_equal(passes.inline_functions()(before), Expected)

    def test_nested_inline_call_checks_its_own_contract(self):
        """An outer helper cannot erase the inner helper's fixed physical shape."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def keep_tile(self, tile: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec]):
                return tile

            @pl.function(type=pl.FunctionType.Inline)
            def forward(self, tile: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]):
                return self.keep_tile(tile)

            @pl.function
            def main(self, source: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]):
                return self.forward(source)

        with pytest.raises(
            ValueError, match="Inline call to 'keep_tile' parameter 'tile': .*physical shape dimension"
        ):
            passes.inline_functions()(Before)

    @pytest.mark.parametrize("actual_rows", [8, 16], ids=["matching", "conflicting"])
    def test_nested_generic_wrapper_checks_final_actual(self, actual_rows):
        """An outer placeholder is checked only after the final caller supplies its Tile."""
        rows = pl.dynamic("TILE_CONTRACT_WRAPPER_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def keep_tile(self, tile: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec]):
                return tile

            @pl.function(type=pl.FunctionType.Inline)
            def forward(self, tile: pl.Tile[[rows, 64], pl.FP32, pl.Mem.Vec]):
                return self.keep_tile(tile)

            @pl.function
            def main(self, source: pl.Tile[[actual_rows, 64], pl.FP32, pl.Mem.Vec]):
                result = self.forward(source)
                return result

        if actual_rows != 8:
            with pytest.raises(
                ValueError, match="Inline call to 'keep_tile' parameter 'tile': .*physical shape dimension"
            ):
                passes.inline_functions()(Before)
            return

        @pl.program
        class Expected:
            @pl.function
            def main(self, source: pl.Tile[[actual_rows, 64], pl.FP32, pl.Mem.Vec]):
                result = source
                return result

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_repeated_physical_dimension_within_one_tile_must_agree(self):
        """A square Tile declaration cannot bind its two equal axes to 8 and 16."""
        rows = pl.dynamic("TILE_CONTRACT_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def consume(self, tile: pl.Tile[[rows, rows], pl.FP32]):
                _doubled = pl.add(tile, tile)

            @pl.function
            def main(self, source: pl.Tile[[8, 16], pl.FP32]):
                self.consume(source)

        with pytest.raises(
            ValueError, match="Inline call to 'consume' parameter 'tile': .*physical shape dimension"
        ):
            passes.inline_functions()(Before)

    def test_generic_dimensions_remain_independent_across_tile_arguments(self):
        """An inline placeholder describes each operand without equating separate buffers."""
        rows = pl.dynamic("TILE_CONTRACT_GENERIC_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def consume(self, first: pl.Tile[[rows, 64], pl.FP32], second: pl.Tile[[rows, 64], pl.FP32]):
                _first_doubled = pl.add(first, first)
                _second_doubled = pl.add(second, second)

            @pl.function
            def main(self, first: pl.Tile[[8, 64], pl.FP32], second: pl.Tile[[16, 64], pl.FP32]):
                self.consume(first, second)

        @pl.program
        class Expected:
            @pl.function
            def main(self, first: pl.Tile[[8, 64], pl.FP32], second: pl.Tile[[16, 64], pl.FP32]):
                _first_doubled = pl.add(first, first)
                _second_doubled = pl.add(second, second)

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    @pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
    @pytest.mark.parametrize("tensor_rows", [8, 16], ids=["matching", "conflicting"])
    def test_void_shared_tile_tensor_dimension_is_checked(self, nested, tensor_rows):
        """A void call must retain shared Tensor constraints through generic wrappers."""
        rows = pl.dynamic("VOID_SHARED_TILE_TENSOR_ROWS")
        wrapper_tile_rows = pl.dynamic("VOID_WRAPPER_TILE_ROWS")
        wrapper_tensor_rows = pl.dynamic("VOID_WRAPPER_TENSOR_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def consume(self, tile: pl.Tile[[rows, 64], pl.FP32], tensor: pl.Tensor[[rows, 64], pl.FP32]):
                _tile_alias = tile
                _tensor_alias = tensor

            @pl.function(type=pl.FunctionType.Inline)
            def forward(
                self,
                tile: pl.Tile[[wrapper_tile_rows, 64], pl.FP32],
                tensor: pl.Tensor[[wrapper_tensor_rows, 64], pl.FP32],
            ):
                self.consume(tile, tensor)

            @pl.function
            def main(self, tile: pl.Tile[[8, 64], pl.FP32], tensor: pl.Tensor[[tensor_rows, 64], pl.FP32]):
                self.forward(tile, tensor)

        if not nested:
            span = ir.Span.unknown()
            main = Before.get_function("main")
            assert main is not None
            call = ir.Call(ir.GlobalVar("consume"), main.params, ir.UnknownType(), span)
            main = ir.Function("main", main.params, [], ir.EvalStmt(call, span), span)
            consume = Before.get_function("consume")
            forward = Before.get_function("forward")
            assert consume is not None and forward is not None
            Before = ir.Program([consume, forward, main], "Before", span)

        if tensor_rows != 8:
            with pytest.raises(ValueError, match="Dynamic type variable.*conflicting bindings"):
                passes.inline_functions()(Before)
            return

        @pl.program
        class Expected:
            @pl.function
            def main(self, tile: pl.Tile[[8, 64], pl.FP32], tensor: pl.Tensor[[tensor_rows, 64], pl.FP32]):
                _tile_alias = tile
                _tensor_alias = tensor

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_unrelated_tile_keeps_void_tensor_placeholders_independent(self):
        """Adding a fixed Tile must not tighten legacy logical/padded Tensor helpers."""
        rows = pl.dynamic("VOID_UNRELATED_TENSOR_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def consume(
                self,
                tile: pl.Tile[[8, 64], pl.FP32],
                logical: pl.Tensor[[rows, 64], pl.FP32],
                padded: pl.Tensor[[rows, 64], pl.FP32],
            ):
                _tile_alias = tile
                _logical_alias = logical
                _padded_alias = padded

            @pl.function
            def main(
                self,
                tile: pl.Tile[[8, 64], pl.FP32],
                logical: pl.Tensor[[8, 64], pl.FP32],
                padded: pl.Tensor[[16, 64], pl.FP32],
            ):
                self.consume(tile, logical, padded)

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                tile: pl.Tile[[8, 64], pl.FP32],
                logical: pl.Tensor[[8, 64], pl.FP32],
                padded: pl.Tensor[[16, 64], pl.FP32],
            ):
                _tile_alias = tile
                _logical_alias = logical
                _padded_alias = padded

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    @pytest.mark.parametrize("tensor_rows", [8, 16], ids=["matching", "conflicting"])
    def test_void_shared_scalar_tile_tensor_dimension_is_checked(self, tensor_rows):
        """An explicit scalar dimension remains shared with a void helper's Tensor."""
        span = ir.Span.unknown()
        rows = ir.Var("rows", ir.ScalarType(pl.INDEX), span)
        tile = ir.Var("tile", ir.TileType([rows, 64], pl.FP32), span)
        tensor = ir.Var("tensor", ir.TensorType([rows, 64], pl.FP32), span)
        source = ir.Var("source", ir.TileType([8, 64], pl.FP32), span)
        storage = ir.Var("storage", ir.TensorType([tensor_rows, 64], pl.FP32), span)
        helper = ir.Function(
            "consume", [rows, tile, tensor], [], ir.ReturnStmt([], span), span, ir.FunctionType.Inline
        )
        args = [ir.ConstInt(8, pl.INDEX, span), source, storage]
        call = ir.Call(ir.GlobalVar("consume"), args, ir.UnknownType(), span)
        body = ir.SeqStmts([ir.EvalStmt(call, span), ir.ReturnStmt([source], span)], span)
        main = ir.Function("main", [source, storage], [source.type], body, span)
        before = ir.Program([helper, main], "Before", span)

        if tensor_rows != 8:
            with pytest.raises(ValueError, match="Dynamic type variable.*conflicting bindings"):
                ir.deduce_call_return_type(helper.params, args, [], is_inline=True)
            with pytest.raises(ValueError, match="Dynamic type variable.*conflicting bindings"):
                passes.inline_functions()(before)
            return

        assert ir.deduce_call_return_type(helper.params, args, [], is_inline=True) == []

        @pl.program
        class Expected:
            @pl.function
            def main(
                self, source: pl.Tile[[8, 64], pl.FP32], storage: pl.Tensor[[tensor_rows, 64], pl.FP32]
            ) -> pl.Tile[[8, 64], pl.FP32]:
                return source

        ir.assert_structural_equal(passes.inline_functions()(before), Expected)

    @pytest.mark.parametrize("metadata", ["valid_shape", "stride", "start_offset"])
    @pytest.mark.parametrize("returns_tile", [False, True], ids=["void", "returning"])
    @pytest.mark.parametrize("scalar_last", [False, True], ids=["scalar-first", "scalar-last"])
    @pytest.mark.parametrize("scalar_value", [3, 8], ids=["matching", "conflicting"])
    def test_scalar_argument_controls_explicit_tile_view(
        self, metadata, returns_tile, scalar_last, scalar_value
    ):
        """An actual Scalar value must agree with every declared Tile view occurrence."""
        span = ir.Span("inline_scalar_view.py", 12, 1)
        extent = ir.Var("extent", ir.ScalarType(pl.INDEX), span)
        if metadata == "valid_shape":
            formal_view = ir.TileView(valid_shape=[extent, 64])
            actual_view = ir.TileView(valid_shape=[3, 64])
            context = "valid shape"
        elif metadata == "stride":
            formal_view = ir.TileView(stride=[extent, 1])
            actual_view = ir.TileView(stride=[3, 1])
            context = "tile stride"
        else:
            formal_view = ir.TileView(start_offset=extent)
            actual_view = ir.TileView(start_offset=3)
            context = "tile start_offset"
        formal_type = ir.TileType([8, 64], pl.FP32, tile_view=formal_view)
        actual_type = ir.TileType([8, 64], pl.FP32, tile_view=actual_view)
        tile = ir.Var("tile", formal_type, span)
        source = ir.Var("source", actual_type, span)
        value = ir.ConstInt(scalar_value, pl.INDEX, span)
        params, args = ([tile, extent], [source, value]) if scalar_last else ([extent, tile], [value, source])
        return_types = [formal_type] if returns_tile else []
        helper = ir.Function(
            "consume",
            params,
            return_types,
            ir.ReturnStmt([tile] if returns_tile else [], span),
            span,
            ir.FunctionType.Inline,
        )
        call = ir.Call(ir.GlobalVar("consume"), args, actual_type if returns_tile else ir.UnknownType(), span)
        body = (
            ir.ReturnStmt([call], span)
            if returns_tile
            else ir.SeqStmts([ir.EvalStmt(call, span), ir.ReturnStmt([source], span)], span)
        )
        before = ir.Program(
            [helper, ir.Function("main", [source], [actual_type], body, span)], "Before", span
        )

        if scalar_value != 3:
            message = rf"Dynamic type variable 'extent'.*conflicting bindings 8 and 3.*{context}"
            with pytest.raises(ValueError, match=message) as error:
                ir.deduce_call_return_type(params, args, return_types, is_inline=True)
            assert "inline_scalar_view.py:12:1" in str(error.value)
            with pytest.raises(ValueError, match=message):
                passes.inline_functions()(before)
            # Surviving-call inference keeps its existing metadata-only binding policy.
            legacy = ir.deduce_call_return_type(params, args, [formal_type])
            ir.assert_structural_equal(legacy[0], actual_type)
            return

        inferred = ir.deduce_call_return_type(params, args, return_types, is_inline=True)
        if returns_tile:
            ir.assert_structural_equal(inferred[0], actual_type)
        else:
            assert inferred == []
        expected = ir.Program(
            [ir.Function("main", [source], [actual_type], ir.ReturnStmt([source], span), span)],
            "Expected",
            span,
        )
        ir.assert_structural_equal(passes.inline_functions()(before), expected)

    @pytest.mark.parametrize("scalar_value", [2, 3], ids=["matching", "conflicting"])
    def test_tile_view_constraint_uses_transitive_actual_scalar_value(self, scalar_value):
        """A view M=4 must also agree with Tensor M=2*n and the actual scalar n."""
        span = ir.Span.unknown()
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        rows = ir.Var("rows", ir.ScalarType(pl.INDEX), span)
        tile_type = ir.TileType([8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[rows, 64]))
        actual_type = ir.TileType([8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[4, 64]))
        tile = ir.Var("tile", tile_type, span)
        tensor = ir.Var("tensor", ir.TensorType([rows, 64], pl.FP32), span)
        actual_tile = ir.Var("actual_tile", actual_type, span)
        twice_scalar = ir.Mul(ir.ConstInt(2, pl.INDEX, span), scalar, pl.INDEX, span)
        actual_tensor = ir.Var("actual_tensor", ir.TensorType([twice_scalar, 64], pl.FP32), span)
        params = [tile, tensor, scalar]
        args = [actual_tile, actual_tensor, ir.ConstInt(scalar_value, pl.INDEX, span)]
        if scalar_value != 2:
            with pytest.raises(
                ValueError, match="Dynamic type variable 'rows'.*conflicting bindings 4 and .*3"
            ):
                ir.deduce_call_return_type(params, args, [], is_inline=True)
            return
        assert ir.deduce_call_return_type(params, args, [], is_inline=True) == []

    @pytest.mark.parametrize("metadata", ["valid_shape", "stride", "start_offset"])
    @pytest.mark.parametrize("returns_tile", [False, True], ids=["void", "returning"])
    @pytest.mark.parametrize("matching", [False, True], ids=["conflicting", "matching"])
    @pytest.mark.parametrize("constant_first", [False, True], ids=["symbolic_binding", "constant_binding"])
    def test_transitive_scalar_tile_view_expression(self, metadata, returns_tile, matching, constant_first):
        """Shared dimension bindings retain Scalar provenance even when a constant won first."""
        span = ir.Span("indirect_scalar_view.py", 51, 2)
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        rows = ir.Var("rows", ir.ScalarType(pl.INDEX), span)
        declared = ir.Sub(rows, ir.ConstInt(1, pl.INDEX, span), pl.INDEX, span)
        observed = 5 if matching else 4
        if metadata == "valid_shape":
            formal_view = ir.TileView(valid_shape=[declared, 64])
            actual_view = ir.TileView(valid_shape=[observed, 64])
        elif metadata == "stride":
            formal_view = ir.TileView(stride=[declared, 1])
            actual_view = ir.TileView(stride=[observed, 1])
        else:
            formal_view = ir.TileView(start_offset=declared)
            actual_view = ir.TileView(start_offset=observed)
        formal_type = ir.TileType([8, 64], pl.FP32, tile_view=formal_view)
        actual_type = ir.TileType([8, 64], pl.FP32, tile_view=actual_view)
        tile = ir.Var("tile", formal_type, span)
        source = ir.Var("source", actual_type, span)
        extent = ir.Var("extent", ir.TensorType([rows, 64], pl.FP32), span)
        twice_scalar = ir.Mul(ir.ConstInt(2, pl.INDEX, span), scalar, pl.INDEX, span)
        tensor = ir.Var("tensor", ir.TensorType([twice_scalar, 64], pl.FP32), span)
        params = [tile]
        args: list[ir.Expr] = [source]
        if constant_first:
            fixed = ir.Var("fixed", ir.TensorType([rows, 64], pl.FP32), span)
            actual_fixed = ir.Var("actual_fixed", ir.TensorType([6, 64], pl.FP32), span)
            params.append(fixed)
            args.append(actual_fixed)
        params.extend([extent, scalar])
        args.extend([tensor, ir.ConstInt(3, pl.INDEX, span)])
        returns = [formal_type] if returns_tile else []
        if matching:
            ir.deduce_call_return_type(params, args, returns, is_inline=True)
        else:
            with pytest.raises(
                ValueError, match=rf"Scalar-dependent Tile {metadata}.*expected 5.*got 4"
            ) as error:
                ir.deduce_call_return_type(params, args, returns, is_inline=True)
            assert "indirect_scalar_view.py:51:2" in str(error.value)
        # Legacy non-inline inference does not interpret Scalar argument values.
        if not constant_first:
            ir.deduce_call_return_type(params, args, [formal_type])

    @pytest.mark.parametrize("matching", [False, True], ids=["conflicting", "matching"])
    def test_scalar_view_provenance_created_during_refinement(self, matching):
        """A deferred view binding can make another complete expression Scalar-dependent."""
        span = ir.Span.unknown()
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        rows = ir.Var("rows", ir.ScalarType(pl.INDEX), span)
        one, three = ir.ConstInt(1, pl.INDEX, span), ir.ConstInt(3, pl.INDEX, span)
        twice_scalar = ir.Mul(ir.ConstInt(2, pl.INDEX, span), scalar, pl.INDEX, span)
        formal_view = ir.TileView(
            valid_shape=[ir.Sub(rows, one, pl.INDEX, span), 64],
            stride=[ir.Add(scalar, rows, pl.INDEX, span), 1],
        )
        actual_view = ir.TileView(
            valid_shape=[5 if matching else 4, 64],
            stride=[ir.Add(three, twice_scalar, pl.INDEX, span), 1],
        )
        tile = ir.Var("tile", ir.TileType([8, 64], pl.FP32, tile_view=formal_view), span)
        source = ir.Var("source", ir.TileType([8, 64], pl.FP32, tile_view=actual_view), span)
        if matching:
            assert ir.deduce_call_return_type([tile, scalar], [source, three], [], is_inline=True) == []
        else:
            with pytest.raises(ValueError, match="Scalar-dependent Tile valid_shape.*expected 5.*got 4"):
                ir.deduce_call_return_type([tile, scalar], [source, three], [], is_inline=True)

    def test_inline_scalar_identity_cannot_be_rebound_by_tile_view(self):
        """A caller-owned scalar identity cannot silently become an unrelated valid extent."""
        span = ir.Span.unknown()
        extent = ir.Var("extent", ir.ScalarType(pl.INDEX), span)
        tile = ir.Var(
            "tile", ir.TileType([8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[extent, 64])), span
        )
        source = ir.Var(
            "source", ir.TileType([8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[3, 64])), span
        )
        with pytest.raises(ValueError, match="Dynamic type variable 'extent'.*conflicting bindings"):
            ir.deduce_call_return_type([extent, tile], [extent, source], [], is_inline=True)

    def test_inline_scalar_binding_rejects_wrong_argument_count(self):
        """The scalar prepass must reject malformed public calls before indexing arguments."""
        span = ir.Span.unknown()
        scalar = ir.Var("extent", ir.ScalarType(pl.INDEX), span)
        with pytest.raises(ValueError, match="callee_params size.*must match args size"):
            ir.deduce_call_return_type([scalar], [], [], is_inline=True)

    @pytest.mark.parametrize("metadata", ["valid_shape", "stride", "start_offset"])
    @pytest.mark.parametrize("expression", ["subtract", "double", "product"])
    @pytest.mark.parametrize("returns_tile", [False, True], ids=["void", "returning"])
    @pytest.mark.parametrize("matching", [False, True], ids=["conflicting", "matching"])
    def test_composite_scalar_tile_view_contract(self, metadata, expression, returns_tile, matching):
        """Whole view expressions must agree after substituting all actual scalar values."""
        span = ir.Span("composite_scalar_view.py", 45, 1)
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        factor = ir.Var("a", ir.ScalarType(pl.INDEX), span)
        if expression == "subtract":
            formal_extent = ir.Sub(scalar, ir.ConstInt(1, pl.INDEX, span), pl.INDEX, span)
            scalar_value, expected_value = 8, 7
        else:
            coefficient = factor if expression == "product" else ir.ConstInt(2, pl.INDEX, span)
            formal_extent = ir.Mul(coefficient, scalar, pl.INDEX, span)
            scalar_value, expected_value = 3, 6
        actual_extent = expected_value if matching else 3
        if metadata == "valid_shape":
            formal_view = ir.TileView(valid_shape=[formal_extent, 64])
            actual_view = ir.TileView(valid_shape=[actual_extent, 64])
        elif metadata == "stride":
            formal_view = ir.TileView(stride=[formal_extent, 1])
            actual_view = ir.TileView(stride=[actual_extent, 1])
        else:
            formal_view = ir.TileView(start_offset=formal_extent)
            actual_view = ir.TileView(start_offset=actual_extent)
        formal_type = ir.TileType([8, 64], pl.FP32, tile_view=formal_view)
        actual_type = ir.TileType([8, 64], pl.FP32, tile_view=actual_view)
        tile = ir.Var("tile", formal_type, span)
        source = ir.Var("source", actual_type, span)
        params = [tile, scalar, factor]
        args = [source, ir.ConstInt(scalar_value, pl.INDEX, span), ir.ConstInt(2, pl.INDEX, span)]
        returns = [formal_type] if returns_tile else []
        helper = ir.Function(
            "consume",
            params,
            returns,
            ir.ReturnStmt([tile] if returns_tile else [], span),
            span,
            ir.FunctionType.Inline,
        )
        call = ir.Call(ir.GlobalVar("consume"), args, actual_type if returns_tile else ir.UnknownType(), span)
        body = (
            ir.ReturnStmt([call], span)
            if returns_tile
            else ir.SeqStmts([ir.EvalStmt(call, span), ir.ReturnStmt([source], span)], span)
        )
        before = ir.Program(
            [helper, ir.Function("main", [source], [actual_type], body, span)], "Before", span
        )
        if not matching:
            message = rf"Scalar-dependent Tile {metadata}.*expected {expected_value}.*got 3"
            with pytest.raises(ValueError, match=message) as error:
                ir.deduce_call_return_type(params, args, returns, is_inline=True)
            assert "composite_scalar_view.py:45:1" in str(error.value)
            with pytest.raises(ValueError, match=message):
                passes.inline_functions()(before)
            legacy = ir.deduce_call_return_type(params, args, [formal_type])
            ir.assert_structural_equal(legacy[0], formal_type)
            return
        ir.deduce_call_return_type(params, args, returns, is_inline=True)
        expected = ir.Program(
            [ir.Function("main", [source], [actual_type], ir.ReturnStmt([source], span), span)],
            "Expected",
            span,
        )
        ir.assert_structural_equal(passes.inline_functions()(before), expected)

    @pytest.mark.parametrize(
        "case",
        [
            "caller_expr",
            "identity",
            "swapped_add",
            "swapped_product",
            "different_ast",
            "unknown",
            "known_unknown",
        ],
    )
    def test_symbolic_scalar_tile_view_expression(self, case):
        """Equivalent caller expressions pass; unrelated runtime dimensions require a guard."""
        span = ir.Span.unknown()
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        other = ir.Var("a", ir.ScalarType(pl.INDEX), span)
        left = ir.Var("left", ir.ScalarType(pl.INDEX), span)
        right = ir.Var("right", ir.ScalarType(pl.INDEX), span)
        one = ir.ConstInt(1, pl.INDEX, span)
        scalar_actual = left
        if case in ("swapped_add", "swapped_product", "different_ast"):
            operation = ir.Mul if case == "swapped_product" else ir.Add
            formal = operation(scalar, other, pl.INDEX, span)
            actual = operation(right, left, pl.INDEX, span)
            if case == "different_ast":
                scalar_actual = ir.Add(left, one, pl.INDEX, span)
                actual = ir.Add(left, ir.Add(right, one, pl.INDEX, span), pl.INDEX, span)
        else:
            formal = ir.Sub(scalar, one, pl.INDEX, span)
            actual = right if case in ("unknown", "known_unknown") else left
            if case == "known_unknown":
                scalar_actual = ir.ConstInt(8, pl.INDEX, span)
            if case == "identity":
                scalar_actual = scalar
                actual = ir.Sub(scalar, one, pl.INDEX, span)
            elif case == "caller_expr":
                scalar_actual = ir.Add(left, one, pl.INDEX, span)
        tile = ir.Var(
            "tile", ir.TileType([8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[formal, 64])), span
        )
        source = ir.Var(
            "source", ir.TileType([8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[actual, 64])), span
        )
        params, args = [tile, scalar, other], [source, scalar_actual, right]
        if case in ("unknown", "known_unknown"):
            with pytest.raises(ValueError, match="Scalar-dependent Tile valid_shape.*not provably equal"):
                ir.deduce_call_return_type(params, args, [], is_inline=True)
            return
        assert ir.deduce_call_return_type(params, args, [], is_inline=True) == []

    @pytest.mark.parametrize("missing", ["valid_shape", "stride", "tile_kind"])
    def test_scalar_view_contract_rejects_missing_actual_field(self, missing):
        """A missing axis or a non-Tile value cannot establish an explicit scalar contract."""
        span = ir.Span.unknown()
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        extent = ir.Sub(scalar, ir.ConstInt(1, pl.INDEX, span), pl.INDEX, span)
        if missing == "stride":
            formal_view, actual_view = ir.TileView(stride=[64, extent]), ir.TileView(stride=[3])
        else:
            formal_view, actual_view = ir.TileView(valid_shape=[8, extent]), ir.TileView(valid_shape=[3])
        tile = ir.Var("tile", ir.TileType([8, 64], pl.FP32, tile_view=formal_view), span)
        actual_type = (
            ir.TensorType([8, 64], pl.FP32)
            if missing == "tile_kind"
            else ir.TileType([8, 64], pl.FP32, tile_view=actual_view)
        )
        source = ir.Var("source", actual_type, span)
        message = "Scalar-dependent Tile.*got " + (
            "a non-Tile argument" if missing == "tile_kind" else "unspecified"
        )
        with pytest.raises(ValueError, match=message):
            ir.deduce_call_return_type(
                [tile, scalar], [source, ir.ConstInt(8, pl.INDEX, span)], [], is_inline=True
            )

    def test_scalar_view_equivalence_preserves_free_dimension_binding(self):
        """A reordered scalar product must not overwrite a free dimension supplied by another Tile."""
        span = ir.Span.unknown()
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        rows = ir.Var("rows", ir.ScalarType(pl.INDEX), span)
        left = ir.Var("left", ir.ScalarType(pl.INDEX), span)
        right = ir.Var("right", ir.ScalarType(pl.INDEX), span)
        formal_product = ir.Mul(scalar, rows, pl.INDEX, span)
        actual_product = ir.Mul(right, left, pl.INDEX, span)
        tile = ir.Var(
            "tile",
            ir.TileType([8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[formal_product, 64])),
            span,
        )
        physical = ir.Var("physical", ir.TileType([rows, 64], pl.FP32), span)
        source = ir.Var(
            "source",
            ir.TileType([8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[actual_product, 64])),
            span,
        )
        actual_physical = ir.Var("actual_physical", ir.TileType([right, 64], pl.FP32), span)
        scalar_type = ir.ScalarType(pl.INDEX)
        helper = ir.Function(
            "dimension",
            [tile, scalar, physical],
            [scalar_type],
            ir.ReturnStmt([rows], span),
            span,
            ir.FunctionType.Inline,
        )
        call = ir.Call(ir.GlobalVar("dimension"), [source, left, actual_physical], scalar_type, span)
        caller_params = [left, source, actual_physical]
        before = ir.Program(
            [helper, ir.Function("main", caller_params, [scalar_type], ir.ReturnStmt([call], span), span)],
            "Before",
            span,
        )
        expected = ir.Program(
            [ir.Function("main", caller_params, [scalar_type], ir.ReturnStmt([right], span), span)],
            "Expected",
            span,
        )
        ir.assert_structural_equal(passes.inline_functions()(before), expected)

    @pytest.mark.parametrize("bound_value", [3, 4], ids=["matching", "conflicting"])
    def test_scalar_view_anchor_uses_later_tensor_binding(self, bound_value):
        """A later Tensor can establish the Scalar anchor needed to infer a free view dimension."""
        span = ir.Span.unknown()
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        rows = ir.Var("M", ir.ScalarType(pl.INDEX), span)
        anchor = ir.Var("K", ir.ScalarType(pl.INDEX), span)
        caller_rows = ir.Var("P", ir.ScalarType(pl.INDEX), span)
        declared = ir.Add(scalar, rows, pl.INDEX, span)
        observed = ir.Add(anchor, caller_rows, pl.INDEX, span)
        tile = ir.Var(
            "tile", ir.TileType([8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[declared, 64])), span
        )
        source = ir.Var(
            "source", ir.TileType([8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[observed, 64])), span
        )
        tensor = ir.Var("tensor", ir.TensorType([anchor, 64], pl.FP32), span)
        storage = ir.Var("storage", ir.TensorType([bound_value, 64], pl.FP32), span)
        params, args = [tile, scalar, tensor], [source, ir.ConstInt(3, pl.INDEX, span), storage]
        scalar_type = ir.ScalarType(pl.INDEX)
        helper = ir.Function(
            "dimension", params, [scalar_type], ir.ReturnStmt([rows], span), span, ir.FunctionType.Inline
        )
        call = ir.Call(ir.GlobalVar("dimension"), args, scalar_type, span)
        before = ir.Program(
            [
                helper,
                ir.Function("main", [source, storage], [scalar_type], ir.ReturnStmt([call], span), span),
            ],
            "Before",
            span,
        )
        if bound_value != 3:
            with pytest.raises(ValueError, match="Scalar-dependent Tile valid_shape.*not provably equal"):
                passes.inline_functions()(before)
            return
        assert ir.deduce_call_return_type(params, args, [], is_inline=True) == []
        expected = ir.Program(
            [ir.Function("main", [source, storage], [scalar_type], ir.ReturnStmt([caller_rows], span), span)],
            "Expected",
            span,
        )
        ir.assert_structural_equal(passes.inline_functions()(before), expected)

    def test_scalar_view_bindings_wake_dependent_fields(self):
        """A deferred field can unlock an earlier field through a Scalar argument expression."""
        span = ir.Span.unknown()
        n, m, rows, anchor, caller_rows, offset, base = [
            ir.Var(name, ir.ScalarType(pl.INDEX), span) for name in ("n", "m", "M", "K", "P", "L", "T")
        ]
        three = ir.ConstInt(3, pl.INDEX, span)
        one = ir.ConstInt(1, pl.INDEX, span)
        first = ir.Var(
            "first",
            ir.TileType(
                [8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[ir.Add(n, rows, pl.INDEX, span), 64])
            ),
            span,
        )
        second = ir.Var(
            "second",
            ir.TileType(
                [8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[ir.Add(m, anchor, pl.INDEX, span), 64])
            ),
            span,
        )
        actual_first = ir.Var(
            "actual_first",
            ir.TileType(
                [8, 64],
                pl.FP32,
                tile_view=ir.TileView(valid_shape=[ir.Add(anchor, caller_rows, pl.INDEX, span), 64]),
            ),
            span,
        )
        actual_second = ir.Var(
            "actual_second",
            ir.TileType(
                [8, 64],
                pl.FP32,
                tile_view=ir.TileView(valid_shape=[ir.Add(offset, three, pl.INDEX, span), 64]),
            ),
            span,
        )
        base_tensor = ir.Var("base_tensor", ir.TensorType([base, 64], pl.FP32), span)
        offset_tensor = ir.Var("offset_tensor", ir.TensorType([offset, 64], pl.FP32), span)
        actual_base = ir.Var("actual_base", ir.TensorType([3, 64], pl.FP32), span)
        actual_offset = ir.Var("actual_offset", ir.TensorType([4, 64], pl.FP32), span)
        params = [first, second, n, m, base_tensor, offset_tensor]
        args = [
            actual_first,
            actual_second,
            three,
            ir.Add(base, one, pl.INDEX, span),
            actual_base,
            actual_offset,
        ]
        inferred = ir.deduce_call_return_type(
            params, args, [ir.TileType([rows, 64], pl.FP32)], is_inline=True
        )
        ir.assert_structural_equal(inferred[0], ir.TileType([caller_rows, 64], pl.FP32))

    def test_zero_scalar_view_does_not_constrain_an_independent_factor(self):
        """A whole expression already equal to zero must not infer a conflicting free factor."""
        span = ir.Span.unknown()
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        rows = ir.Var("M", ir.ScalarType(pl.INDEX), span)
        caller_rows = ir.Var("P", ir.ScalarType(pl.INDEX), span)
        zero = ir.ConstInt(0, pl.INDEX, span)
        declared = ir.Mul(scalar, rows, pl.INDEX, span)
        observed = ir.Mul(zero, caller_rows, pl.INDEX, span)
        tile = ir.Var(
            "tile", ir.TileType([rows, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[declared, 64])), span
        )
        source = ir.Var(
            "source", ir.TileType([16, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[observed, 64])), span
        )
        assert ir.deduce_call_return_type([tile, scalar], [source, zero], [], is_inline=True) == []

    @pytest.mark.parametrize(
        "metadata,scalar_value,matching",
        [
            ("valid_shape", 9, True),
            ("valid_shape", 8, False),
            ("start_offset", 1, True),
            ("start_offset", 2, False),
            ("stride", 65, False),
        ],
    )
    def test_scalar_view_contract_uses_actual_default_metadata(self, metadata, scalar_value, matching):
        """Full valid shape and zero offset have defaults; missing strides remain unspecified."""
        span = ir.Span.unknown()
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        extent = ir.Sub(scalar, ir.ConstInt(1, pl.INDEX, span), pl.INDEX, span)
        if metadata == "valid_shape":
            view = ir.TileView(valid_shape=[extent, 64])
        elif metadata == "stride":
            view = ir.TileView(stride=[extent, 1])
        else:
            view = ir.TileView(start_offset=extent)
        tile = ir.Var("tile", ir.TileType([8, 64], pl.FP32, tile_view=view), span)
        source = ir.Var("source", ir.TileType([8, 64], pl.FP32), span)
        params, args = [tile, scalar], [source, ir.ConstInt(scalar_value, pl.INDEX, span)]
        if not matching:
            message = rf"Scalar-dependent Tile {metadata}.*got " + (
                "unspecified" if metadata == "stride" else ""
            )
            with pytest.raises(ValueError, match=message):
                ir.deduce_call_return_type(params, args, [], is_inline=True)
            return
        assert ir.deduce_call_return_type(params, args, [], is_inline=True) == []

    @pytest.mark.parametrize("metadata", ["valid_shape", "stride", "start_offset"])
    def test_scalar_view_check_preserves_constant_metadata_policy(self, metadata):
        """An unrelated Scalar must not make fixed metadata an inline equality contract."""
        span = ir.Span.unknown()
        scalar = ir.Var("unused", ir.ScalarType(pl.INDEX), span)
        if metadata == "valid_shape":
            formal_view, actual_view = ir.TileView(valid_shape=[7, 64]), ir.TileView(valid_shape=[3, 64])
        elif metadata == "stride":
            formal_view, actual_view = ir.TileView(stride=[7, 1]), ir.TileView(stride=[3, 1])
        else:
            formal_view, actual_view = ir.TileView(start_offset=7), ir.TileView(start_offset=3)
        tile = ir.Var("tile", ir.TileType([8, 64], pl.FP32, tile_view=formal_view), span)
        source = ir.Var("source", ir.TileType([8, 64], pl.FP32, tile_view=actual_view), span)
        assert (
            ir.deduce_call_return_type(
                [tile, scalar], [source, ir.ConstInt(8, pl.INDEX, span)], [], is_inline=True
            )
            == []
        )

    @pytest.mark.parametrize(
        "mismatch", ["shape", "dtype", "memory_space", "rank", "tile_kind", "tuple_kind", "tuple_arity"]
    )
    def test_nested_tuple_tile_physical_contract(self, mismatch):
        """Discarding a tuple argument must not erase a nested Tile's declared contract."""
        span = ir.Span("tuple_tile_contract.py", 29, 3)
        scalar_type = ir.ScalarType(pl.INDEX)
        formal_tile = ir.TileType([8, 64], pl.FP32, memory_space=ir.MemorySpace.Vec)
        formal_type = ir.TupleType([scalar_type, ir.TupleType([formal_tile])])
        shape = [16, 64] if mismatch == "shape" else [8, 64, 1] if mismatch == "rank" else [8, 64]
        dtype = pl.BF16 if mismatch == "dtype" else pl.FP32
        space = ir.MemorySpace.Acc if mismatch == "memory_space" else ir.MemorySpace.Vec
        actual_tile = (
            ir.TensorType(shape, dtype)
            if mismatch == "tile_kind"
            else ir.TileType(shape, dtype, memory_space=space)
        )
        actual_type = (
            actual_tile
            if mismatch == "tuple_kind"
            else ir.TupleType([scalar_type, ir.TupleType([] if mismatch == "tuple_arity" else [actual_tile])])
        )
        pair, source = ir.Var("pair", formal_type, span), ir.Var("source", actual_type, span)
        helper = ir.Function("consume", [pair], [], ir.ReturnStmt([], span), span, ir.FunctionType.Inline)
        call = ir.Call(ir.GlobalVar("consume"), [source], ir.UnknownType(), span)
        body = ir.SeqStmts([ir.EvalStmt(call, span), ir.ReturnStmt([source], span)], span)
        before = ir.Program(
            [helper, ir.Function("main", [source], [actual_type], body, span)], "Before", span
        )
        detail = {
            "shape": "physical shape dimension",
            "dtype": "dtype",
            "memory_space": "memory space",
            "rank": "physical rank",
            "tile_kind": "expected Tile",
            "tuple_kind": "expected tuple",
            "tuple_arity": "tuple arity",
        }[mismatch]
        with pytest.raises(
            ValueError, match=f"Inline call to 'consume' parameter 'pair': .*{detail}"
        ) as error:
            passes.inline_functions()(before)
        assert "tuple_tile_contract.py:29:3" in str(error.value)

    def test_nested_tuple_tiles_keep_independent_generic_dimensions(self):
        """Each generic Tile uses its own physical extents and preserves its caller-owned view."""
        span = ir.Span.unknown()
        rows = ir.Var("rows", ir.ScalarType(pl.INDEX), span)
        formal_tile = ir.TileType([rows, 64], pl.FP32)
        formal_type = ir.TupleType([formal_tile, ir.TupleType([formal_tile])])
        actual_type = ir.TupleType(
            [
                ir.TileType(
                    [8, 64],
                    pl.FP32,
                    memory_space=ir.MemorySpace.Vec,
                    tile_view=ir.TileView(valid_shape=[3, 64]),
                ),
                ir.TupleType([ir.TileType([16, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[5, 64]))]),
            ]
        )
        pair, source = ir.Var("pair", formal_type, span), ir.Var("source", actual_type, span)
        helper = ir.Function("consume", [pair], [], ir.ReturnStmt([], span), span, ir.FunctionType.Inline)
        call = ir.Call(ir.GlobalVar("consume"), [source], ir.UnknownType(), span)
        before = ir.Program(
            [
                helper,
                ir.Function(
                    "main",
                    [source],
                    [actual_type],
                    ir.SeqStmts([ir.EvalStmt(call, span), ir.ReturnStmt([source], span)], span),
                    span,
                ),
            ],
            "Before",
            span,
        )
        expected = ir.Program(
            [ir.Function("main", [source], [actual_type], ir.ReturnStmt([source], span), span)],
            "Expected",
            span,
        )
        ir.assert_structural_equal(passes.inline_functions()(before), expected)

    @pytest.mark.parametrize("scalar_first", [False, True], ids=["scalar_last", "scalar_first"])
    @pytest.mark.parametrize("scalar_value", [8, 16], ids=["matching", "conflicting"])
    def test_nested_tuple_tile_shape_uses_actual_scalar(self, scalar_first, scalar_value):
        """Scalar values remain authoritative for physical dimensions inside tuple parameters."""
        span = ir.Span.unknown()
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        formal_type = ir.TupleType([ir.TupleType([ir.TileType([scalar, 64], pl.FP32)])])
        actual_type = ir.TupleType(
            [ir.TupleType([ir.TileType([8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[3, 64]))])]
        )
        pair, source = ir.Var("pair", formal_type, span), ir.Var("source", actual_type, span)
        params, args = [pair, scalar], [source, ir.ConstInt(scalar_value, pl.INDEX, span)]
        if scalar_first:
            params.reverse()
            args.reverse()
        helper = ir.Function("consume", params, [], ir.ReturnStmt([], span), span, ir.FunctionType.Inline)
        call = ir.Call(ir.GlobalVar("consume"), args, ir.UnknownType(), span)
        before = ir.Program(
            [
                helper,
                ir.Function(
                    "main",
                    [source],
                    [actual_type],
                    ir.SeqStmts([ir.EvalStmt(call, span), ir.ReturnStmt([source], span)], span),
                    span,
                ),
            ],
            "Before",
            span,
        )
        if scalar_value != 8:
            with pytest.raises(ValueError, match="parameter 'pair': .*physical shape dimension"):
                passes.inline_functions()(before)
            return
        expected = ir.Program(
            [ir.Function("main", [source], [actual_type], ir.ReturnStmt([source], span), span)],
            "Expected",
            span,
        )
        ir.assert_structural_equal(passes.inline_functions()(before), expected)

    @pytest.mark.parametrize("scalar_value", [4, 8], ids=["matching", "conflicting"])
    def test_void_tuple_tile_scalar_view_contract(self, scalar_value):
        """Tuple-contained Tiles must reach the same void-call contract validation."""
        span = ir.Span.unknown()
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        extent = ir.Sub(scalar, ir.ConstInt(1, pl.INDEX, span), pl.INDEX, span)
        formal_type = ir.TupleType(
            [ir.TileType([8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[extent, 64]))]
        )
        actual_type = ir.TupleType(
            [ir.TileType([8, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[3, 64]))]
        )
        pair, source = ir.Var("pair", formal_type, span), ir.Var("source", actual_type, span)
        helper = ir.Function(
            "consume", [pair, scalar], [], ir.ReturnStmt([], span), span, ir.FunctionType.Inline
        )
        call = ir.Call(
            ir.GlobalVar("consume"),
            [source, ir.ConstInt(scalar_value, pl.INDEX, span)],
            ir.UnknownType(),
            span,
        )
        body = ir.SeqStmts([ir.EvalStmt(call, span), ir.ReturnStmt([source], span)], span)
        before = ir.Program(
            [helper, ir.Function("main", [source], [actual_type], body, span)], "Before", span
        )
        if scalar_value != 4:
            with pytest.raises(ValueError, match="Scalar-dependent Tile valid_shape.*expected 7.*got 3"):
                passes.inline_functions()(before)
            return
        expected = ir.Program(
            [ir.Function("main", [source], [actual_type], ir.ReturnStmt([source], span), span)],
            "Expected",
            span,
        )
        ir.assert_structural_equal(passes.inline_functions()(before), expected)

    @pytest.mark.parametrize("last_rows", [8, 9], ids=["matching", "conflicting"])
    def test_void_shared_tile_constraint_checks_transitive_tensor_bindings(self, last_rows):
        """Every binding feeding the shared Tile dimension must agree, including dependencies."""
        rows = pl.dynamic("VOID_TRANSITIVE_TILE_ROWS")
        dependent = pl.dynamic("VOID_TRANSITIVE_DEPENDENT_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def consume(
                self,
                tile: pl.Tile[[rows, 64], pl.FP32],
                tensor: pl.Tensor[[rows, 64], pl.FP32],
                first: pl.Tensor[[dependent, 64], pl.FP32],
                second: pl.Tensor[[dependent, 64], pl.FP32],
            ):
                _tile_alias = tile
                _tensor_alias = tensor

            @pl.function
            def main(
                self,
                tile: pl.Tile[[16, 64], pl.FP32],
                tensor: pl.Tensor[[2 * dependent, 64], pl.FP32],
                first: pl.Tensor[[8, 64], pl.FP32],
                second: pl.Tensor[[last_rows, 64], pl.FP32],
            ):
                self.consume(tile, tensor, first, second)

        if last_rows != 8:
            with pytest.raises(ValueError, match="Dynamic type variable.*conflicting bindings"):
                passes.inline_functions()(Before)
            return

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                tile: pl.Tile[[16, 64], pl.FP32],
                tensor: pl.Tensor[[2 * dependent, 64], pl.FP32],
                first: pl.Tensor[[8, 64], pl.FP32],
                second: pl.Tensor[[last_rows, 64], pl.FP32],
            ):
                _tile_alias = tile
                _tensor_alias = tensor

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    @pytest.mark.parametrize("valid_rows", [8, 4], ids=["matching", "conflicting"])
    def test_void_explicit_view_keeps_shared_dimension_strict(self, valid_rows):
        """An explicit valid extent still constrains the same physical placeholder."""
        rows = pl.dynamic("VOID_SHARED_TILE_VIEW_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def consume(
                self,
                tile: pl.Tile[[rows, 64], pl.FP32],
                viewed: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[rows, 64])],
            ):
                _tile_alias = tile
                _viewed_alias = viewed

            @pl.function
            def main(
                self,
                tile: pl.Tile[[8, 64], pl.FP32],
                viewed: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[valid_rows, 64])],
            ):
                self.consume(tile, viewed)

        if valid_rows != 8:
            with pytest.raises(ValueError, match="Dynamic type variable.*conflicting bindings"):
                passes.inline_functions()(Before)
            return

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                tile: pl.Tile[[8, 64], pl.FP32],
                viewed: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[valid_rows, 64])],
            ):
                _tile_alias = tile
                _viewed_alias = viewed

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    @pytest.mark.parametrize("affine", [False, True], ids=["plain", "affine"])
    def test_returned_tile_arguments_keep_independent_generic_dimensions(self, affine):
        """First, second, and tuple returns must retain their own actual shapes through later ops."""
        rows = pl.dynamic("TILE_CONTRACT_RETURNED_GENERIC_ROWS")
        extent = 2 * rows if affine else rows

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def select_first(
                self, first: pl.Tile[[extent, 64], pl.FP32], second: pl.Tile[[extent, 64], pl.FP32]
            ):
                return first

            @pl.function(type=pl.FunctionType.Inline)
            def select_second(
                self, first: pl.Tile[[extent, 64], pl.FP32], second: pl.Tile[[extent, 64], pl.FP32]
            ):
                return second

            @pl.function(type=pl.FunctionType.Inline)
            def select_pair(
                self, first: pl.Tile[[extent, 64], pl.FP32], second: pl.Tile[[extent, 64], pl.FP32]
            ):
                return first, second

            @pl.function
            def main(self, first: pl.Tile[[16, 64], pl.FP32], second: pl.Tile[[32, 64], pl.FP32]):
                selected_first = self.select_first(first, second)
                selected_second = self.select_second(first, second)
                left, right = self.select_pair(first, second)
                first_sum = pl.add(selected_first, left)
                second_sum = pl.add(selected_second, right)
                return first_sum, second_sum

        @pl.program
        class Expected:
            @pl.function
            def main(self, first: pl.Tile[[16, 64], pl.FP32], second: pl.Tile[[32, 64], pl.FP32]):
                selected_first = first
                selected_second = second
                left = first
                right = second
                first_sum = pl.add(selected_first, left)
                second_sum = pl.add(selected_second, right)
                return first_sum, second_sum

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_unspecified_formal_memory_space_accepts_vec(self):
        """An omitted formal space leaves the actual Tile's resolved space intact."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def keep_tile(self, tile: pl.Tile[[8, 64], pl.FP32]):
                return tile

            @pl.function
            def main(self, source: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec]):
                result = self.keep_tile(source)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, source: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec]):
                result = source
                return result

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_unresolved_actual_memory_space_remains_inferable(self):
        """A formal Vec declaration must not force premature actual-space inference."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def keep_tile(self, tile: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec]):
                return tile

            @pl.function
            def main(self, source: pl.Tile[[8, 64], pl.FP32]):
                result = self.keep_tile(source)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, source: pl.Tile[[8, 64], pl.FP32]):
                result = source
                return result

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    @pytest.mark.parametrize("dynamic_physical", [False, True])
    @pytest.mark.parametrize("dynamic_valid", [False, True])
    def test_partial_valid_shape_is_preserved(self, dynamic_physical, dynamic_valid):
        """Only physical dimensions bind the formal; a partial valid extent is independent."""
        physical_rows = pl.dynamic("TILE_CONTRACT_PHYSICAL_ROWS") if dynamic_physical else 8
        valid_rows = pl.dynamic("TILE_CONTRACT_VALID_ROWS") if dynamic_valid else 3

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def keep_tile(self, tile: pl.Tile[[physical_rows, 64], pl.FP32, pl.Mem.Vec]):
                return tile

            @pl.function
            def main(
                self,
                source: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])],
            ):
                result = self.keep_tile(source)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                source: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])],
            ):
                result = source
                return result

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_composite_generic_physical_dimension_binds_to_actual(self):
        """Bind M inside 2*M to the caller's N while retaining its smaller valid extent."""
        formal_rows = pl.dynamic("TILE_CONTRACT_COMPOSITE_FORMAL_ROWS")
        caller_rows = pl.dynamic("TILE_CONTRACT_COMPOSITE_CALLER_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def consume(self, tile: pl.Tile[[2 * formal_rows, 64], pl.FP32, pl.Mem.Vec]):
                _alias = tile

            @pl.function
            def main(
                self,
                source: pl.Tile[
                    [2 * caller_rows, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[caller_rows, 64])
                ],
            ):
                self.consume(source)
                return source

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                source: pl.Tile[
                    [2 * caller_rows, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[caller_rows, 64])
                ],
            ):
                _alias = source
                return source

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_provably_equal_physical_arithmetic_is_accepted(self):
        """Equivalent physical expressions need not have identical AST spelling."""
        rows = pl.dynamic("TILE_CONTRACT_ARITH_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def consume(self, tile: pl.Tile[[(rows + 8) - rows, 64], pl.FP32]):
                _alias = tile

            @pl.function
            def main(self, source: pl.Tile[[8, 64], pl.FP32]):
                self.consume(source)
                return source

        @pl.program
        class Expected:
            @pl.function
            def main(self, source: pl.Tile[[8, 64], pl.FP32]):
                _alias = source
                return source

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_affine_generic_dimension_binds_to_concrete_identity_argument(self):
        """The reported 2*M -> 16 identity call also specializes the parser's return type."""
        rows = pl.dynamic("TILE_CONTRACT_STATIC_AFFINE_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def keep_tile(self, tile: pl.Tile[[2 * rows, 64], pl.FP32]):
                return tile

            @pl.function
            def main(self, source: pl.Tile[[16, 64], pl.FP32]):
                result = self.keep_tile(source)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, source: pl.Tile[[16, 64], pl.FP32]):
                result = source
                return result

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    @pytest.mark.parametrize("operation", ["add", "subtract", "multiply"])
    @pytest.mark.parametrize("scalar_first", [False, True], ids=["scalar_last", "scalar_first"])
    def test_actual_scalar_is_substituted_before_generic_extent_inference(self, operation, scalar_first):
        """Known Scalar values leave one free dimension for return inference and clone seeding."""
        span = ir.Span.unknown()
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        rows = ir.Var("M", ir.ScalarType(pl.INDEX), span)
        scalar_value = 24 if operation == "subtract" else 2 if operation == "multiply" else 8
        operator = {"add": ir.Add, "subtract": ir.Sub, "multiply": ir.Mul}[operation]
        extent = operator(scalar, rows, pl.INDEX, span)
        formal_type: ir.Type = ir.TileType([extent, 64], pl.FP32)
        actual_type: ir.Type = ir.TileType([16, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[3, 64]))
        if operation == "multiply":
            formal_type, actual_type = ir.TupleType([formal_type]), ir.TupleType([actual_type])
        formal = ir.Var("tile", formal_type, span)
        source = ir.Var("source", actual_type, span)
        params, args = [formal, scalar], [source, ir.ConstInt(scalar_value, pl.INDEX, span)]
        if scalar_first:
            params.reverse()
            args.reverse()
        result_type = ir.ScalarType(pl.INDEX)
        helper = ir.Function(
            "dimension", params, [result_type], ir.ReturnStmt([rows], span), span, ir.FunctionType.Inline
        )
        call = ir.Call(ir.GlobalVar("dimension"), args, result_type, span)
        before = ir.Program(
            [helper, ir.Function("main", [source], [result_type], ir.ReturnStmt([call], span), span)],
            "Before",
            span,
        )
        expected = ir.Program(
            [
                ir.Function(
                    "main",
                    [source],
                    [result_type],
                    ir.ReturnStmt([ir.ConstInt(8, pl.INDEX, span)], span),
                    span,
                )
            ],
            "Expected",
            span,
        )
        ir.assert_structural_equal(passes.inline_functions()(before), expected)
        # The shared parser deduction must recover the same free extent.
        (deduced,) = ir.deduce_call_return_type(
            params, args, [ir.TileType([rows, 64], pl.FP32)], is_inline=True
        )
        ir.assert_structural_equal(deduced, ir.TileType([8, 64], pl.FP32))

    @pytest.mark.parametrize("columns", [8, 9], ids=["consistent", "conflicting"])
    def test_scalar_specialized_affine_binding_agrees_across_axes(self, columns):
        """Scalar substitution cannot make repeated physical dimensions independent."""
        span = ir.Span.unknown()
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        rows = ir.Var("M", ir.ScalarType(pl.INDEX), span)
        formal = ir.Var("tile", ir.TileType([ir.Add(scalar, rows, pl.INDEX, span), rows], pl.FP32), span)
        source = ir.Var("source", ir.TileType([16, columns], pl.FP32), span)
        helper = ir.Function(
            "consume", [scalar, formal], [], ir.ReturnStmt([], span), span, ir.FunctionType.Inline
        )
        call = ir.Call(
            ir.GlobalVar("consume"), [ir.ConstInt(8, pl.INDEX, span), source], ir.UnknownType(), span
        )
        before = ir.Program(
            [helper, ir.Function("main", [source], [], ir.EvalStmt(call, span), span)], "Before", span
        )
        if columns != 8:
            with pytest.raises(ValueError, match="parameter 'tile': .*physical shape dimension"):
                passes.inline_functions()(before)
            return
        expected = ir.Program(
            [ir.Function("main", [source], [], ir.SeqStmts([], span), span)], "Expected", span
        )
        ir.assert_structural_equal(passes.inline_functions()(before), expected)

    @pytest.mark.parametrize("valid_rows", [7, 4], ids=["matching", "conflicting"])
    def test_scalar_specialized_generic_extent_keeps_view_provenance(self, valid_rows):
        """Solving M from n+M must retain its Scalar origin for explicit view checks."""
        span = ir.Span("scalar_affine_view.py", 19, 2)
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        rows = ir.Var("M", ir.ScalarType(pl.INDEX), span)
        extent = ir.Add(scalar, rows, pl.INDEX, span)
        valid = ir.Sub(rows, ir.ConstInt(1, pl.INDEX, span), pl.INDEX, span)
        formal = ir.Var(
            "tile", ir.TileType([extent, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[valid, 64])), span
        )
        source = ir.Var(
            "source",
            ir.TileType([16, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[valid_rows, 64])),
            span,
        )
        args = [source, ir.ConstInt(8, pl.INDEX, span)]
        if valid_rows != 7:
            with pytest.raises(
                ValueError, match="Scalar-dependent Tile valid_shape.*expected 7.*got 4"
            ) as error:
                ir.deduce_call_return_type([formal, scalar], args, [], is_inline=True)
            assert "scalar_affine_view.py:19:2" in str(error.value)
            return
        assert ir.deduce_call_return_type([formal, scalar], args, [], is_inline=True) == []

    def test_affine_inference_does_not_solve_an_unknown_scalar_actual(self):
        """A caller-owned runtime dimension cannot become a free affine variable after substitution."""
        span = ir.Span.unknown()
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        rows = ir.Var("M", ir.ScalarType(pl.INDEX), span)
        extent = ir.Add(scalar, rows, pl.INDEX, span)
        formal = ir.Var("tile", ir.TileType([extent, 64], pl.FP32), span)
        source = ir.Var("source", ir.TileType([16, 64], pl.FP32), span)
        returned_type = ir.TileType([rows, 64], pl.FP32)
        (deduced,) = ir.deduce_call_return_type(
            [formal, scalar], [source, rows], [returned_type], is_inline=True
        )
        ir.assert_structural_equal(deduced, returned_type)
        helper = ir.Function(
            "consume", [formal, scalar], [], ir.ReturnStmt([], span), span, ir.FunctionType.Inline
        )
        call = ir.Call(ir.GlobalVar("consume"), [source, rows], ir.UnknownType(), span)
        before = ir.Program(
            [helper, ir.Function("main", [source, rows], [], ir.EvalStmt(call, span), span)], "Before", span
        )
        with pytest.raises(ValueError, match="parameter 'tile': .*physical shape dimension"):
            passes.inline_functions()(before)

    def test_scalar_specialized_tiles_keep_independent_free_dimensions(self):
        """A known Scalar does not merge the remaining generic extents of separate Tile arguments."""
        span = ir.Span.unknown()
        scalar = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        rows = ir.Var("M", ir.ScalarType(pl.INDEX), span)
        formal_type = ir.TileType([ir.Add(scalar, rows, pl.INDEX, span), 64], pl.FP32)
        first, second = ir.Var("first", formal_type, span), ir.Var("second", formal_type, span)
        left = ir.Var("left", ir.TileType([16, 64], pl.FP32), span)
        right = ir.Var("right", ir.TileType([24, 64], pl.FP32), span)
        helper = ir.Function(
            "consume", [first, second, scalar], [], ir.ReturnStmt([], span), span, ir.FunctionType.Inline
        )
        call = ir.Call(
            ir.GlobalVar("consume"), [left, right, ir.ConstInt(8, pl.INDEX, span)], ir.UnknownType(), span
        )
        before = ir.Program(
            [helper, ir.Function("main", [left, right], [], ir.EvalStmt(call, span), span)], "Before", span
        )
        expected = ir.Program(
            [ir.Function("main", [left, right], [], ir.SeqStmts([], span), span)], "Expected", span
        )
        ir.assert_structural_equal(passes.inline_functions()(before), expected)

    @pytest.mark.parametrize("valid_rows", [17, 4], ids=["fully_valid", "partially_valid"])
    def test_affine_binding_specializes_inline_local_type(self, valid_rows):
        """Cloned locals use the physical binding even when the argument is only partially valid."""
        rows = pl.dynamic("TILE_CONTRACT_AFFINE_LOCAL_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def make_tile(self, tile: pl.Tile[[2 * rows + 1, 64], pl.FP32]):
                scratch: pl.Tile[[rows, 64], pl.FP32] = pl.tile.tpop_from_aiv()
                return scratch

            @pl.function
            def main(self, source: pl.Tile[[17, 64], pl.FP32, pl.TileView(valid_shape=[valid_rows, 64])]):
                result = self.make_tile(source)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, source: pl.Tile[[17, 64], pl.FP32, pl.TileView(valid_shape=[valid_rows, 64])]):
                scratch: pl.Tile[[8, 64], pl.FP32] = pl.tile.tpop_from_aiv()
                result = scratch
                return result

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    @pytest.mark.parametrize(
        "valid_rows", [3, 4, "composite"], ids=["odd_static", "even_static", "composite_dynamic"]
    )
    def test_affine_concrete_physical_binding_ignores_partial_valid_shape(self, valid_rows):
        """An identity return preserves validity without rebinding the physical symbol M = 8."""
        rows = pl.dynamic("TILE_CONTRACT_AFFINE_PARTIAL_ROWS")
        valid_factor = 2 if isinstance(valid_rows, str) else 1
        if isinstance(valid_rows, str):
            # A matching 2*N tree must not bind M from an undeclared valid shape.
            valid_rows = pl.dynamic("TILE_CONTRACT_AFFINE_VALID_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def keep_tile(self, tile: pl.Tile[[2 * rows, 64], pl.FP32]):
                return tile

            @pl.function
            def main(
                self,
                source: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[valid_factor * valid_rows, 64])],
            ):
                result = self.keep_tile(source)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                source: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[valid_factor * valid_rows, 64])],
            ):
                result = source
                return result

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_explicit_tile_valid_shape_still_binds_returned_local_type(self):
        """An explicitly declared valid symbol still specializes metadata independently of physical shape."""
        valid_rows = pl.dynamic("TILE_CONTRACT_EXPLICIT_VALID_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def receive(self, tile: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[valid_rows, 64])]):
                received: pl.Tile[[valid_rows, 64], pl.FP32] = pl.tile.tpop_from_aiv()
                return received

            @pl.function
            def main(self, source: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[4, 64])]):
                result = self.receive(source)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, source: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[4, 64])]):
                received: pl.Tile[[4, 64], pl.FP32] = pl.tile.tpop_from_aiv()
                result = received
                return result

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_affine_generic_dimension_requires_integral_binding(self):
        """2*M = 15 has no integer solution and must not truncate to M = 7."""
        rows = pl.dynamic("TILE_CONTRACT_AFFINE_ODD_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def consume(self, tile: pl.Tile[[2 * rows, 64], pl.FP32]):
                _alias = tile

            @pl.function
            def main(self, source: pl.Tile[[15, 64], pl.FP32]):
                self.consume(source)

        with pytest.raises(
            ValueError, match="Inline call to 'consume' parameter 'tile': .*physical shape dimension"
        ):
            passes.inline_functions()(Before)

    @pytest.mark.parametrize("columns", [32, 36], ids=["consistent", "conflicting"])
    def test_affine_binding_is_consistent_across_tile_axes(self, columns):
        """The axes 2*M and 4*M must describe one M even when both actual axes are static."""
        rows = pl.dynamic("TILE_CONTRACT_AFFINE_SHARED_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def consume(self, tile: pl.Tile[[2 * rows, 4 * rows], pl.FP32]):
                _alias = tile

            @pl.function
            def main(self, source: pl.Tile[[16, columns], pl.FP32]):
                self.consume(source)

        if columns != 32:
            with pytest.raises(
                ValueError, match="Inline call to 'consume' parameter 'tile': .*physical shape dimension"
            ):
                passes.inline_functions()(Before)
            return

        @pl.program
        class Expected:
            @pl.function
            def main(self, source: pl.Tile[[16, columns], pl.FP32]):
                _alias = source

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)


class TestInlineFunctionsDynamicTypes:
    """Inlined tensor types must reference the caller's values and dimensions."""

    def test_dynamic_tpop_preserves_declared_result_type(self):
        """Unknown op inference must retain the remapped explicit tile type."""
        formal_rows = pl.dynamic("FORMAL_ROWS")
        rows = pl.dynamic("ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def receive(self, shape_source: pl.Tile[[formal_rows, 16], pl.FP32]):
                received: pl.Tile[[formal_rows, 16], pl.FP32] = pl.tile.tpop_from_aiv()
                return received

            @pl.function
            def main(self, source: pl.Tile[[rows, 16], pl.FP32]):
                result = self.receive(source)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, source: pl.Tile[[rows, 16], pl.FP32]):
                received: pl.Tile[[rows, 16], pl.FP32] = pl.tile.tpop_from_aiv()
                result = received
                return result

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(after, pl.parse_program(after.as_python()))

    def test_alias_type_follows_rhs_through_ssa(self):
        """A stale LHS-only view cannot override the actual alias source."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, source: pl.Tensor[[4, 8], pl.FP32]):
                viewed: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[16, 1])] = source
                _rows = pl.tensor.dim(viewed, 0)

            @pl.function
            def main(self, source: pl.Tensor[[4, 8], pl.FP32]):
                self.helper(source)
                caller_alias: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[16, 1])] = source
                _caller_rows = pl.tensor.dim(caller_alias, 0)

        @pl.program
        class Expected:
            @pl.function
            def main(self, source: pl.Tensor[[4, 8], pl.FP32]):
                viewed = source
                _rows = pl.tensor.dim(viewed, 0)
                caller_alias = source
                _caller_rows = pl.tensor.dim(caller_alias, 0)

        # This regression deliberately supplies stale alias types to the
        # normalization pass; verify its repaired output, not that input.
        instrument = passes.VerificationInstrument(passes.VerificationMode.AFTER)
        with passes.PassContext([instrument]):
            after = passes.inline_functions()(Before)
        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.AssignTypeSymmetry)
        for actual, expected in (
            (after, Expected),
            (passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected)),
        ):
            ir.assert_structural_equal(actual, expected)
            assert not passes.PropertyVerifierRegistry.verify(props, actual)

    def test_rhs_view_is_preserved_through_ssa(self):
        """RHS view metadata survives inlining, including downstream aliases."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(
                self,
                source: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND)],
            ):
                viewed: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[32, 1])] = source
                _rows = pl.tensor.dim(viewed, 0)

            @pl.function
            def main(
                self,
                source: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND)],
            ):
                self.helper(source)
                caller_alias: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[32, 1])] = source
                _caller_rows = pl.tensor.dim(caller_alias, 0)

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                source: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND)],
            ):
                viewed = source
                _rows = pl.tensor.dim(viewed, 0)
                caller_alias = source
                _caller_rows = pl.tensor.dim(caller_alias, 0)

        # This regression deliberately supplies stale alias types to the
        # normalization pass; verify its repaired output, not that input.
        instrument = passes.VerificationInstrument(passes.VerificationMode.AFTER)
        with passes.PassContext([instrument]):
            after = passes.inline_functions()(Before)
        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.AssignTypeSymmetry)
        for actual, expected in (
            (after, Expected),
            (passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected)),
        ):
            ir.assert_structural_equal(actual, expected)
            assert not passes.PropertyVerifierRegistry.verify(props, actual)

    def test_tile_alias_memory_space_follows_rhs(self):
        """Assignment retyping must not reapply stale LHS storage metadata."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, source: pl.Tile[[4, 8], pl.FP32]):
                _alias: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = source

            @pl.function
            def main(self, source: pl.Tile[[4, 8], pl.FP32]):
                self.helper(source)

        @pl.program
        class Expected:
            @pl.function
            def main(self, source: pl.Tile[[4, 8], pl.FP32]):
                _alias = source

        # This regression deliberately supplies stale alias types to the
        # normalization pass; verify its repaired output, not that input.
        instrument = passes.VerificationInstrument(passes.VerificationMode.AFTER)
        with passes.PassContext([instrument]):
            after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected))

    def test_inline_alias_inherits_actual_memref(self):
        """Structural type equality alone must not discard the RHS allocation."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, source: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec]):
                _alias = source

            @pl.function
            def main(self, data: pl.Tensor[[4, 8], pl.FP32]):
                source: pl.Tile[[4, 8], pl.FP32, pl.MemRef("scratch"), pl.Mem.Vec] = pl.load(
                    data, [0, 0], [4, 8], target_memory=pl.Mem.Vec
                )
                self.helper(source)

        @pl.program
        class Expected:
            @pl.function
            def main(self, data: pl.Tensor[[4, 8], pl.FP32]):
                source: pl.Tile[[4, 8], pl.FP32, pl.MemRef("scratch"), pl.Mem.Vec] = pl.load(
                    data, [0, 0], [4, 8], target_memory=pl.Mem.Vec
                )
                _alias = source

        after = passes.inline_functions()(Before)
        for actual, expected in (
            (after, Expected),
            (passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected)),
        ):
            ir.assert_structural_equal(actual, expected)
            main = next(iter(actual.functions.values()))
            assert isinstance(main.body, ir.SeqStmts)
            source, alias = main.body.stmts
            assert isinstance(source, ir.AssignStmt)
            assert isinstance(alias, ir.AssignStmt)
            assert isinstance(source.var.type, ir.TileType)
            assert isinstance(alias.var.type, ir.TileType)
            assert source.var.type.memref is not None
            assert alias.var.type.memref is source.var.type.memref

    def test_shared_formal_dimension_keeps_each_actual_tensor_shape(self):
        formal_rows = pl.dynamic("formal_rows")
        first_rows = pl.dynamic("first_rows")
        second_rows = pl.dynamic("second_rows")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def fill(
                self,
                first: pl.Tensor[[formal_rows, 16], pl.FP32],
                second: pl.Tensor[[formal_rows, 16], pl.FP32],
                row: pl.Tensor[[1, 16], pl.FP32],
            ):
                first_update = pl.tensor.assemble(first, row, [0, 0])
                first = first_update
                second_update = pl.tensor.assemble(second, row, [0, 0])
                second = second_update

            @pl.function
            def main(
                self,
                first: pl.Tensor[[first_rows, 16], pl.FP32],
                second: pl.Tensor[[second_rows, 16], pl.FP32],
                row: pl.Tensor[[1, 16], pl.FP32],
            ):
                self.fill(first, second, row)

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                first: pl.Tensor[[first_rows, 16], pl.FP32],
                second: pl.Tensor[[second_rows, 16], pl.FP32],
                row: pl.Tensor[[1, 16], pl.FP32],
            ):
                first_update = pl.tensor.assemble(first, row, [0, 0])
                first = first_update
                second_update = pl.tensor.assemble(second, row, [0, 0])
                second = second_update

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_computed_scalar_argument_preserves_caller_tensor_identity(self):
        shared_rows = pl.dynamic("shared_rows")
        caller_rows = pl.dynamic("caller_rows")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def fill(
                self,
                out: pl.Tensor[[shared_rows, 16], pl.FP32],
                row: pl.Tensor[[1, 16], pl.FP32],
                count: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[shared_rows, 16], pl.FP32]:
                if count > 0:
                    out = pl.tensor.assemble(out, row, [0, 0])
                return out

            @pl.function
            def main(
                self,
                ids: pl.Tensor[[shared_rows], pl.INT32],
                out: pl.Tensor[[caller_rows, 16], pl.FP32],
                row: pl.Tensor[[1, 16], pl.FP32],
            ) -> pl.Tensor[[caller_rows, 16], pl.FP32]:
                out = self.fill(out, row, pl.tensor.dim(ids, 0))
                return out

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                ids: pl.Tensor[[shared_rows], pl.INT32],
                out: pl.Tensor[[caller_rows, 16], pl.FP32],
                row: pl.Tensor[[1, 16], pl.FP32],
            ) -> pl.Tensor[[caller_rows, 16], pl.FP32]:
                if pl.tensor.dim(ids, 0) > 0:
                    out = pl.tensor.assemble(out, row, [0, 0])
                return out

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_shape_expression_binding_preserves_caller_symbols(self):
        shared_rows = pl.dynamic("shared_rows")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def fill(
                self,
                out: pl.Tensor[[shared_rows, 16], pl.FP32],
                row: pl.Tensor[[1, 16], pl.FP32],
            ):
                updated = pl.tensor.assemble(out, row, [0, 0])
                out = updated

            @pl.function
            def main(
                self,
                out: pl.Tensor[[shared_rows + 1, 16], pl.FP32],
                row: pl.Tensor[[1, 16], pl.FP32],
            ) -> pl.Tensor[[shared_rows + 1, 16], pl.FP32]:
                self.fill(out, row)
                return out

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                out: pl.Tensor[[shared_rows + 1, 16], pl.FP32],
                row: pl.Tensor[[1, 16], pl.FP32],
            ) -> pl.Tensor[[shared_rows + 1, 16], pl.FP32]:
                updated = pl.tensor.assemble(out, row, [0, 0])
                out = updated
                return out

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    @pytest.mark.parametrize("via_alias", [False, True])
    def test_conditional_assemble_uses_actual_tensor_shape(self, via_alias):
        formal_rows = pl.dynamic("formal_rows")
        caller_rows = pl.dynamic("caller_rows")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def fill(
                self,
                out: pl.Tensor[[formal_rows, 16], pl.FP32],
                row: pl.Tensor[[1, 16], pl.FP32],
                active: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[formal_rows, 16], pl.FP32]:
                if active > 0:
                    if via_alias:
                        updated = pl.tensor.assemble(out, row, [0, 0])
                        out = updated
                    else:
                        out = pl.tensor.assemble(out, row, [0, 0])
                return out

            @pl.function
            def main(
                self,
                out: pl.Tensor[[caller_rows, 16], pl.FP32],
                row: pl.Tensor[[1, 16], pl.FP32],
                active: pl.Scalar[pl.INDEX],
            ):
                result = self.fill(out, row, active)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                out: pl.Tensor[[caller_rows, 16], pl.FP32],
                row: pl.Tensor[[1, 16], pl.FP32],
                active: pl.Scalar[pl.INDEX],
            ):
                if active > 0:
                    if via_alias:
                        updated = pl.tensor.assemble(out, row, [0, 0])
                        out = updated
                    else:
                        out = pl.tensor.assemble(out, row, [0, 0])
                result = out
                return result

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_local_dimensions_and_valid_shapes_are_fresh_at_each_call_site(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[16, 16], pl.FP32], n: pl.Scalar[pl.INDEX]):
                rows = n + 1
                out = pl.tensor.create([rows, 16], dtype=pl.FP32)
                view = pl.tensor.slice(x, [8, 16], [0, 0], valid_shape=[rows, 16])
                tile = pl.tile.load(view, [0, 0], [8, 16])
                out = pl.tile.store(tile, [0, 0], out)
                return out

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                first_rows: pl.Scalar[pl.INDEX],
                second_rows: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                self.helper(x, first_rows)
                self.helper(x, second_rows)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                first_rows: pl.Scalar[pl.INDEX],
                second_rows: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                rows_first = first_rows + 1
                out_first = pl.tensor.create([rows_first, 16], dtype=pl.FP32)
                view_first = pl.tensor.slice(x, [8, 16], [0, 0], valid_shape=[rows_first, 16])
                tile_first = pl.tile.load(view_first, [0, 0], [8, 16])
                out_first = pl.tile.store(tile_first, [0, 0], out_first)
                rows_second = second_rows + 1
                out_second = pl.tensor.create([rows_second, 16], dtype=pl.FP32)
                view_second = pl.tensor.slice(x, [8, 16], [0, 0], valid_shape=[rows_second, 16])
                tile_second = pl.tile.load(view_second, [0, 0], [8, 16])
                out_second = pl.tile.store(tile_second, [0, 0], out_second)
                return x

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_for_loop_carried_dimensions_reference_fresh_iter_args(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, n: pl.Scalar[pl.INDEX]):
                for i, (rows,) in pl.range(2, init_values=(n,)):
                    _out = pl.tensor.create([rows, 16], dtype=pl.FP32)
                    _done = pl.yield_(rows + 1)

            @pl.function
            def main(self, initial_rows: pl.Scalar[pl.INDEX]):
                self.helper(initial_rows)

        @pl.program
        class Expected:
            @pl.function
            def main(self, initial_rows: pl.Scalar[pl.INDEX]):
                for i, (rows,) in pl.range(2, init_values=(initial_rows,)):
                    _out = pl.tensor.create([rows, 16], dtype=pl.FP32)
                    _done = pl.yield_(rows + 1)

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_while_loop_carried_dimensions_reference_fresh_iter_args(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, n: pl.Scalar[pl.INDEX]):
                for (rows,) in pl.while_(init_values=(n,)):
                    pl.cond(rows < 16)
                    _out = pl.tensor.create([rows, 16], dtype=pl.FP32)
                    _done = pl.yield_(rows + 1)

            @pl.function
            def main(self, initial_rows: pl.Scalar[pl.INDEX]):
                self.helper(initial_rows)

        @pl.program
        class Expected:
            @pl.function
            def main(self, initial_rows: pl.Scalar[pl.INDEX]):
                for (rows,) in pl.while_(init_values=(initial_rows,)):
                    pl.cond(rows < 16)
                    _out = pl.tensor.create([rows, 16], dtype=pl.FP32)
                    _done = pl.yield_(rows + 1)

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)

    def test_inline_submit_preserves_task_result_and_dependencies(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[16], pl.FP32], n: pl.Scalar[pl.INDEX]):
                with pl.manual_scope():
                    prior = pl.system.task_dummy(deps=[])
                    out, tid = pl.spmd_submit(self.kernel, x, core_num=n, deps=[prior])
                    _fence = pl.system.task_dummy(deps=[tid])
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[16], pl.FP32], blocks: pl.Scalar[pl.INDEX]):
                result = self.helper(x, blocks)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[16], pl.FP32], blocks: pl.Scalar[pl.INDEX]):
                with pl.manual_scope():
                    prior = pl.system.task_dummy(deps=[])
                    out, tid = pl.spmd_submit(self.kernel, x, core_num=blocks, deps=[prior])
                    _fence = pl.system.task_dummy(deps=[tid])
                result = out
                return result

        ir.assert_structural_equal(passes.inline_functions()(Before), Expected)


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

    def test_call_site_tag_follows_local_binding_into_scope(self):
        """A tagged argument stays selected when the inline parameter is localized."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    x = pl.add(x, x)
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.dump_tag(a)
                r = self.helper(a)
                return r

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local = a
                with pl.at(level=pl.Level.CORE_GROUP, dumps=[local]):
                    local = pl.add(local, local)
                r = local
                return r

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(after, pl.parse_program(after.as_python()))

    def test_call_site_tag_follows_local_binding_into_dispatch(self):
        """A void inline call preserves tags on a nested cross-function dispatch."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def double(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return pl.add(x, x)

            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[64], pl.FP32]):
                x = self.double(x)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, a: pl.Tensor[[64], pl.FP32]):
                pl.dump_tag(a)
                self.helper(a)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def double(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return pl.add(x, x)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, a: pl.Tensor[[64], pl.FP32]):
                local = a
                pl.dump_tag(local)
                local = self.double(local)

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(after, pl.parse_program(after.as_python()))

    def test_call_site_tag_follows_local_binding_into_submits(self):
        """Both submit forms preserve dependencies, existing dumps, and repeated args."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
                z: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                return pl.add(x, pl.add(y, z))

            @pl.function(type=pl.FunctionType.Inline)
            def helper(
                self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    prior = pl.system.task_dummy(deps=[])
                    first, first_tid = pl.submit(self.kernel, x, y, x, deps=[prior], dumps=[y, y])
                    x = first
                    second, second_tid = pl.spmd_submit(
                        self.kernel, x, y, x, core_num=2, deps=[first_tid], dumps=[y, y]
                    )
                    x = second
                    _fence = pl.system.task_dummy(deps=[second_tid])
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                pl.dump_tag(a)
                result = self.helper(a, b)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
                z: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                return pl.add(x, pl.add(y, z))

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                local = a
                with pl.manual_scope():
                    prior = pl.system.task_dummy(deps=[])
                    first, first_tid = pl.submit(self.kernel, local, b, local, deps=[prior], dumps=[b, local])
                    local = first
                    second, second_tid = pl.spmd_submit(
                        self.kernel, local, b, local, core_num=2, deps=[first_tid], dumps=[b, local]
                    )
                    local = second
                    _fence = pl.system.task_dummy(deps=[second_tid])
                result = local
                return result

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(after, pl.parse_program(after.as_python()))
        ir.assert_structural_equal(passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected))

    def test_call_site_tag_follows_nested_local_bindings(self):
        """A return-position inline call keeps the tag through a second local handle."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def leaf(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    x = pl.add(x, x)
                return x

            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                x = self.leaf(x)
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.dump_tag(a)
                return self.helper(a)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                outer = a
                inner = outer
                with pl.at(level=pl.Level.CORE_GROUP, dumps=[inner]):
                    inner = pl.add(inner, inner)
                outer = inner
                return outer

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(after, pl.parse_program(after.as_python()))

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

        ``unused`` has no Call site; cleanup drops it because its
        ``func_type_ == Inline``. ``main`` carries no inline call, so it
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

    def test_live_returned_tuple_reaches_orchestration_codegen(self):
        """Keeping a used tuple must preserve both its body write and returned operands."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self, x: pl.Tensor[[16], pl.FP32], out: pl.Out[pl.Tensor[[16], pl.FP32]]
            ) -> pl.Tensor[[16], pl.FP32]:
                return pl.store(pl.load(x, [0], [16]), [0], out)

            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[16], pl.FP32], out: pl.Tensor[[16], pl.FP32]):
                tmp = (x, x)
                out = self.consume(tmp[0], out)
                return tmp

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, x: pl.Tensor[[16], pl.FP32], out: pl.Out[pl.Tensor[[16], pl.FP32]]
            ) -> pl.Tensor[[16], pl.FP32]:
                a, b = self.helper(x, out)
                out = self.consume(a, out)
                out = self.consume(b, out)
                return out

        lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Program)
        orch = next(f for f in lowered.functions.values() if f.func_type == ir.FunctionType.Orchestration)
        generated = codegen.generate_orchestration(lowered, orch).code
        assert generated.count(".add_input(ext_x)") == 3, generated
        assert "FREE_VAR" not in generated


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

    def test_tensor_value_rebinding_keeps_caller_argument(self):
        """A callee-local add result must not replace the caller's input."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def twice(self, t: pl.Tensor[[4], pl.FP32]) -> pl.Tensor[[4], pl.FP32]:
                t = pl.add(t, t)
                return t

            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32]):
                u = self.twice(a)
                v = self.twice(a)
                return pl.add(a, pl.add(u, v))

        @pl.program
        class Expected:
            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32]):
                t0 = a
                t0 = pl.add(t0, t0)
                u = t0
                t1 = a
                t1 = pl.add(t1, t1)
                v = t1
                return pl.add(a, pl.add(u, v))

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected))

    def test_tile_value_rebinding_keeps_caller_argument(self):
        """Tile parameters have the same local binding semantics as tensors."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def twice(self, t: pl.Tile[[8, 8], pl.FP32]) -> pl.Tile[[8, 8], pl.FP32]:
                t = pl.tile.add(t, t)
                return t

            @pl.function
            def main(self, a: pl.Tile[[8, 8], pl.FP32]):
                u = self.twice(a)
                return pl.tile.add(a, u)

        @pl.program
        class Expected:
            @pl.function
            def main(self, a: pl.Tile[[8, 8], pl.FP32]):
                t = a
                t = pl.tile.add(t, t)
                u = t
                return pl.tile.add(a, u)

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_alias_changed_on_later_iteration_cannot_retarget_caller(self):
        """Alias classification must account for loop backedges, not just DFS order."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def replace(self, t: pl.Tensor[[4], pl.FP32], other: pl.Tensor[[4], pl.FP32]):
                alias = t
                for i in pl.range(2):
                    t = alias
                    alias = other
                return t

            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]):
                u = self.replace(a, b)
                return a, u

        @pl.program
        class Expected:
            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]):
                local = a
                alias = local
                for i in pl.range(2):
                    local = alias
                    alias = b
                u = local
                return a, u

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected))

    def test_rebinding_from_explicit_iter_arg_keeps_caller_argument(self):
        """An IterArg's value can come from another buffer and change on each yield."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def replace(self, t: pl.Tensor[[4], pl.FP32], other: pl.Tensor[[4], pl.FP32]):
                for i, (carry,) in pl.range(2, init_values=(other,)):
                    t = carry
                    _result = pl.yield_(pl.add(carry, carry))
                return t

            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]):
                u = self.replace(a, b)
                return a, u

        @pl.program
        class Expected:
            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]):
                local = a
                for i, (carry,) in pl.range(2, init_values=(b,)):
                    local = carry
                    _result = pl.yield_(pl.add(carry, carry))
                u = local
                return a, u

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected))

    def test_inout_tensor_value_rebinding_stays_local(self):
        """Direction describes storage effects, not a reference to the caller's name."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def twice(self, t: pl.InOut[pl.Tensor[[4], pl.FP32]]):
                t = pl.add(t, t)
                return t

            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32]):
                u = self.twice(a)
                return pl.add(a, u)

        @pl.program
        class Expected:
            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32]):
                local = a
                local = pl.add(local, local)
                u = local
                return pl.add(a, u)

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_rebinding_to_another_write_target_keeps_caller_argument(self):
        """A writeback to a different buffer is still a local name rebinding."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def replace(self, t: pl.Tensor[[4], pl.FP32], other: pl.Tensor[[4], pl.FP32]):
                t = pl.tensor.assemble(other, t, [0])
                return t

            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]):
                u = self.replace(a, b)
                return a, u

        @pl.program
        class Expected:
            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]):
                t = a
                t = pl.tensor.assemble(b, t, [0])
                u = t
                return a, u

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)

    @pytest.mark.parametrize("count", [0, 1, 3])
    def test_jit_tensor_loop_rebinding_compiles_with_local_carry(self, count, tmp_path):
        """Specialization, inlining, SSA and codegen preserve the caller's input."""

        @pl.jit.inline
        def twice(t: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.FP32]:
            for i in pl.range(count):
                t = pl.add(t, t)
            return t

        @pl.jit
        def main(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(pl.Level.CORE_GROUP):
                u = twice(a)
                out[:] = pl.add(a, u)
            return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def main(
                self, a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]
            ) -> pl.Tensor[[8, 8], pl.FP32]:
                with pl.at(pl.Level.CORE_GROUP):
                    local = a
                    for i in pl.range(count):
                        local = pl.add(local, local)
                    u = local
                    out[:] = pl.add(a, u)
                return out

        before = main.specialize()
        after = passes.inline_functions()(before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected))
        actual = ir.compile(
            before,
            output_dir=str(tmp_path / "actual"),
            backend_type=backend.BackendType.Ascend950,
            skip_ptoas=True,
        )
        expected = ir.compile(
            Expected,
            output_dir=str(tmp_path / "expected"),
            backend_type=backend.BackendType.Ascend950,
            skip_ptoas=True,
        )
        actual_dump = sorted((actual.output_dir / "passes_dump").glob("*.py"))[-1]
        expected_dump = sorted((expected.output_dir / "passes_dump").glob("*.py"))[-1]
        ir.assert_structural_equal(pl.parse(actual_dump.read_text()), pl.parse(expected_dump.read_text()))

    @pytest.mark.parametrize("take_branch", [False, True])
    def test_mixed_writeback_and_value_rebinding_compiles(self, take_branch, tmp_path):
        """Writes before detaching affect the argument; later writes affect the local value."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def update(
                self,
                t: pl.Tensor[[8, 8], pl.FP32],
                x: pl.Tensor[[8, 8], pl.FP32],
                flag: pl.Scalar[pl.BOOL],
            ):
                t = pl.tensor.assemble(t, pl.add(x, x), [0, 0])
                if flag:
                    t = pl.add(t, t)
                t = pl.tensor.assemble(t, pl.add(x, pl.add(x, x)), [0, 0])
                return t

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
                x: pl.Tensor[[8, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            ):
                with pl.at(pl.Level.CORE_GROUP):
                    u = self.update(a, x, take_branch)
                    out[:] = pl.add(a, u)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
                x: pl.Tensor[[8, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            ):
                with pl.at(pl.Level.CORE_GROUP):
                    local = a
                    local = pl.tensor.assemble(local, pl.add(x, x), [0, 0])
                    if take_branch:
                        local = pl.add(local, local)
                    local = pl.tensor.assemble(local, pl.add(x, pl.add(x, x)), [0, 0])
                    u = local
                    out[:] = pl.add(a, u)
                return out

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected))
        actual = ir.compile(
            Before,
            output_dir=str(tmp_path / "actual"),
            backend_type=backend.BackendType.Ascend950,
            skip_ptoas=True,
        )
        expected = ir.compile(
            Expected,
            output_dir=str(tmp_path / "expected"),
            backend_type=backend.BackendType.Ascend950,
            skip_ptoas=True,
        )
        actual_dump = sorted((actual.output_dir / "passes_dump").glob("*.py"))[-1]
        expected_dump = sorted((expected.output_dir / "passes_dump").glob("*.py"))[-1]
        ir.assert_structural_equal(pl.parse(actual_dump.read_text()), pl.parse(expected_dump.read_text()))

    def test_rebound_tensor_param_still_aliases_caller_arg(self):
        """A parameter used only for self-writeback keeps direct substitution.

        Unlike value rebinding, ``assemble(c, ...)`` preserves the destination
        storage. A bare tensor annotation must still support that writeback
        without introducing a redundant call-site handle."""

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

    @pytest.mark.parametrize("rebind_source", [False, True])
    def test_returned_tuple_temporary_keeps_body_uses(self, rebind_source):
        """A live tuple definition and its captured elements must survive inlining."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4], pl.FP32]]:
                tmp = (x, x)
                out = pl.tensor.assemble(out, tmp[0], [0])
                if rebind_source:
                    x = pl.add(x, x)
                return tmp

            @pl.function
            def main(self, x: pl.Tensor[[4], pl.FP32], out: pl.Out[pl.Tensor[[4], pl.FP32]]):
                a, b = self.helper(x, out)
                return a, b, out

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[4], pl.FP32], out: pl.Out[pl.Tensor[[4], pl.FP32]]):
                local = x
                tmp = (local, local)
                out = pl.tensor.assemble(out, tmp[0], [0])
                if rebind_source:
                    local = pl.add(local, local)
                a = tmp[0]
                b = tmp[1]
                return a, b, out

        after = passes.inline_functions()(Before)
        assert "FREE_VAR" not in after.as_python()
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(after, pl.parse_program(after.as_python()))
        ir.assert_structural_equal(passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected))

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


class TestInlineFunctionsDynamicShapes:
    def test_specialized_result_reaches_next_inline_call(self):
        """The next inline call must receive its argument's refined view metadata."""
        rows = pl.dynamic("FORMAL_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def refine(self, x: pl.Tensor[[rows, 4], pl.FP32]) -> pl.Tensor[[rows, 4], pl.FP32]:
                refined = pl.tensor.set_validshape(x, 3, 4)
                return refined

            @pl.function(type=pl.FunctionType.Inline)
            def double(self, x: pl.Tensor[[rows, 4], pl.FP32]):
                result = pl.add(x, x)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[8, 4], pl.FP32]):
                first = self.refine(x)
                second = self.double(first)
                return pl.add(second, second)

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[8, 4], pl.FP32]):
                refined = pl.tensor.set_validshape(x, 3, 4)
                first = refined
                result = pl.add(first, first)
                second = result
                return pl.add(second, second)

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected))

    def test_nested_inline_tuple_results_keep_independent_shapes(self):
        """Nested tuple substitutions and their caller uses share the updated types."""
        rows = pl.dynamic("FORMAL_ROWS")
        padded_rows = pl.dynamic("PADDED_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def pair(self, x: pl.Tensor[[rows, 4], pl.FP32], padded: pl.Tensor[[padded_rows, 4], pl.FP32]):
                first = pl.add(x, x)
                second = pl.add(padded, padded)
                return first, second

            @pl.function(type=pl.FunctionType.Inline)
            def forward(self, x: pl.Tensor[[rows, 4], pl.FP32], padded: pl.Tensor[[padded_rows, 4], pl.FP32]):
                return self.pair(x, padded)

            @pl.function
            def main(self, x: pl.Tensor[[8, 4], pl.FP32], padded: pl.Tensor[[16, 4], pl.FP32]):
                a, b = self.forward(x, padded)
                return pl.add(a, x), pl.add(b, padded)

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[8, 4], pl.FP32], padded: pl.Tensor[[16, 4], pl.FP32]):
                first = pl.add(x, x)
                second = pl.add(padded, padded)
                a = first
                b = second
                return pl.add(a, x), pl.add(b, padded)

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected))

    def test_nested_alias_return_refines_callers_valid_shape(self):
        """Factoring a metadata update into an inline helper keeps its caller binding."""
        rows = pl.dynamic("NESTED_ALIAS_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def refine(self, t: pl.Tensor[[rows, 4], pl.FP32]) -> pl.Tensor[[rows, 4], pl.FP32]:
                refined = pl.tensor.set_validshape(t, 3, 4)
                return refined

            @pl.function(type=pl.FunctionType.Inline)
            def outer(self, t: pl.Tensor[[rows, 4], pl.FP32]):
                t = self.refine(t)

            @pl.function
            def main(self, t: pl.Tensor[[8, 4], pl.FP32]):
                self.outer(t)
                return t

        @pl.program
        class Expected:
            @pl.function
            def main(self, t: pl.Tensor[[8, 4], pl.FP32]):
                refined = pl.tensor.set_validshape(t, 3, 4)
                t = refined
                return t

        after = passes.inline_functions()(Before)
        ssa = passes.convert_to_ssa()(after)
        main = ssa.get_function("main")
        assert main is not None and isinstance(main.body, ir.SeqStmts)
        result = main.body.stmts[-1]
        assert isinstance(result, ir.ReturnStmt)
        result_type = result.value[0].type
        assert isinstance(result_type, ir.TensorType)
        view = result_type.tensor_view
        valid_shape = view.valid_shape if view is not None and view.valid_shape else result_type.shape
        assert len(valid_shape) == 2
        for dim, expected in zip(valid_shape, [3, 4]):
            assert isinstance(dim, ir.ConstInt)
            assert dim.value == expected
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(ssa, passes.convert_to_ssa()(Expected))

    def test_nested_fresh_return_keeps_callers_binding(self):
        """Expanding a generic nested helper must not turn fresh results into writebacks."""
        rows = pl.dynamic("NESTED_FRESH_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def double(self, t: pl.Tensor[[rows, 4], pl.FP32]) -> pl.Tensor[[rows, 4], pl.FP32]:
                doubled = pl.add(t, t)
                return doubled

            @pl.function(type=pl.FunctionType.Inline)
            def outer(self, t: pl.Tensor[[rows, 4], pl.FP32]):
                t = self.double(t)

            @pl.function
            def main(self, t: pl.Tensor[[8, 4], pl.FP32]):
                self.outer(t)
                return t

        @pl.program
        class Expected:
            @pl.function
            def main(self, t: pl.Tensor[[8, 4], pl.FP32]):
                local = t
                doubled = pl.add(local, local)
                local = doubled
                return t

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected))

    def test_nested_identity_return_materializes_original_scalar_def(self):
        """An analysis-only no-op elimination must not expose a non-Var assignment target."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def identity(self, n: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                return n

            @pl.function(type=pl.FunctionType.Inline)
            def outer(self, n: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                n = self.identity(n)
                return n

            @pl.function
            def main(self, k: pl.Scalar[pl.INDEX]):
                constant = self.outer(3)
                computed = self.outer(k + 1)
                return constant + computed

        @pl.program
        class Expected:
            @pl.function
            def main(self, k: pl.Scalar[pl.INDEX]):
                local_constant = 3
                constant = local_constant
                local_computed = k + 1
                computed = local_computed
                return constant + computed

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected))

    def test_inline_view_refinement_preserves_caller_rebinding(self):
        """A void helper must update the binding read after the inline call."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def refine(self, out: pl.Tensor[[8, 4], pl.FP32]):
                out = pl.tensor.set_validshape(out, 3, 4)

            @pl.function
            def main(self, out: pl.Tensor[[8, 4], pl.FP32]):
                self.refine(out)
                return out

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8, 4], pl.FP32]):
                out = pl.tensor.set_validshape(out, 3, 4)
                return out

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected))

    def test_inline_tile_refinement_preserves_caller_rebinding(self):
        """Tile metadata updates use the same direct writeback contract as tensors."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def refine(self, out: pl.Tile[[8, 4], pl.FP32, pl.TileView(valid_shape=[3, 4])]):
                out = pl.tile.set_validshape(out, 3, 4)

            @pl.function
            def main(self, out: pl.Tile[[8, 4], pl.FP32, pl.TileView(valid_shape=[3, 4])]):
                self.refine(out)
                return out

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tile[[8, 4], pl.FP32, pl.TileView(valid_shape=[3, 4])]):
                out = pl.tile.set_validshape(out, 3, 4)
                return out

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        ir.assert_structural_equal(passes.convert_to_ssa()(after), passes.convert_to_ssa()(Expected))

    def test_conflicting_result_type_uses_operand_even_with_caller_dimension(self):
        """An unrelated caller signature dimension cannot mask a stale result extent."""
        rows = pl.dynamic("SHARED_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def fill(self, out: pl.Tensor[[rows, 4], pl.FP32], padded: pl.Tensor[[rows, 4], pl.FP32]):
                updated = pl.add(out, out)
                out = updated

            @pl.function
            def main(
                self,
                marker: pl.Tensor[[rows, 4], pl.FP32],
                out: pl.Tensor[[8, 4], pl.FP32],
                padded: pl.Tensor[[16, 4], pl.FP32],
            ):
                self.fill(out, padded)
                return out

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                marker: pl.Tensor[[rows, 4], pl.FP32],
                out: pl.Tensor[[8, 4], pl.FP32],
                padded: pl.Tensor[[16, 4], pl.FP32],
            ):
                local = out
                updated = pl.add(local, local)
                local = updated
                return out

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        passes.convert_to_ssa()(after)
        ir.assert_structural_equal(after, pl.parse_program(after.as_python()))

    def test_conflicting_tensor_carries_follow_for_and_while_initializers(self):
        """Both loop kinds specialize tensor carries and the results consumed after them."""
        rows = pl.dynamic("FORMAL_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def fill(self, out: pl.Tensor[[rows, 4], pl.FP32], padded: pl.Tensor[[rows, 4], pl.FP32]):
                for i, (carry,) in pl.range(2, init_values=(out,)):
                    result = pl.yield_(pl.add(carry, carry))
                out = pl.add(result, result)
                for carry, count in pl.while_(init_values=(out, 0)):
                    pl.cond(count < 2)
                    result, _count = pl.yield_(pl.add(carry, carry), count + 1)
                out = pl.add(result, result)

            @pl.function
            def main(self, out: pl.Tensor[[8, 4], pl.FP32], padded: pl.Tensor[[16, 4], pl.FP32]):
                self.fill(out, padded)
                return out

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8, 4], pl.FP32], padded: pl.Tensor[[16, 4], pl.FP32]):
                local = out
                for i, (carry,) in pl.range(2, init_values=(local,)):
                    result = pl.yield_(pl.add(carry, carry))
                local = pl.add(result, result)
                for carry, count in pl.while_(init_values=(local, 0)):
                    pl.cond(count < 2)
                    result, _count = pl.yield_(pl.add(carry, carry), count + 1)
                local = pl.add(result, result)
                return out

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        passes.convert_to_ssa()(after)
        ir.assert_structural_equal(after, pl.parse_program(after.as_python()))

    def test_caller_signature_dimension_not_present_in_actual_arguments(self):
        """A signature dimension stays in scope when only static buffers reach the helper."""
        rows = pl.dynamic("SHARED_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def fill(self, out: pl.Tensor[[rows, 4], pl.FP32], padded: pl.Tensor[[rows, 4], pl.FP32]):
                _scratch = pl.create_tensor([rows, 4], pl.FP32)

            @pl.function
            def main(
                self,
                marker: pl.Tensor[[rows, 4], pl.FP32],
                out: pl.Tensor[[8, 4], pl.FP32],
                padded: pl.Tensor[[16, 4], pl.FP32],
            ):
                self.fill(out, padded)
                return marker

        after = passes.inline_functions()(Before)
        passes.convert_to_ssa()(after)

    @pytest.mark.parametrize("loop_carried", [False, True])
    @pytest.mark.parametrize("separate_symbols", [False, True])
    def test_logical_and_padded_local_types(self, loop_carried, separate_symbols):
        """Local/loop types follow their operands even when annotations share a symbol."""
        rows = pl.dynamic("CALLEE_ROWS")
        padded_rows = pl.dynamic("PADDED_ROWS") if separate_symbols else rows

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def fill(self, out: pl.Tensor[[rows, 4], pl.FP32], padded: pl.Tensor[[padded_rows, 4], pl.FP32]):
                if loop_carried:
                    for i in pl.range(2):
                        out = pl.add(out, out)
                else:
                    _updated = pl.add(padded, padded)

            @pl.function
            def main(self, out: pl.Tensor[[8, 4], pl.FP32], padded: pl.Tensor[[16, 4], pl.FP32]):
                self.fill(out, padded)
                return out

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8, 4], pl.FP32], padded: pl.Tensor[[16, 4], pl.FP32]):
                local = out
                if loop_carried:
                    for i in pl.range(2):
                        local = pl.add(local, local)
                else:
                    _updated = pl.add(padded, padded)
                return out

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        passes.convert_to_ssa()(after)
        ir.assert_structural_equal(after, pl.parse_program(after.as_python()))

    def test_void_inline_call_preserves_logical_and_padded_arguments(self):
        """Inline helpers may use one placeholder for logical and padded buffers."""
        rows = pl.dynamic("CALLEE_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def fill(self, logical: pl.Tensor[[rows, 4], pl.FP32], padded: pl.Tensor[[rows, 4], pl.FP32]):
                n = pl.tensor.dim(logical, 0)
                _value = pl.tensor.read(padded, [n, 0])

            @pl.function
            def main(self, logical: pl.Tensor[[8, 4], pl.FP32], padded: pl.Tensor[[16, 4], pl.FP32]):
                self.fill(logical, padded)
                return padded

        @pl.program
        class Expected:
            @pl.function
            def main(self, logical: pl.Tensor[[8, 4], pl.FP32], padded: pl.Tensor[[16, 4], pl.FP32]):
                n = pl.tensor.dim(logical, 0)
                _value = pl.tensor.read(padded, [n, 0])
                return padded

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        passes.convert_to_ssa()(after)

    def test_shared_shape_symbols_preserve_caller_arguments(self):
        """Specialize callee locals without rewriting an argument's shared shape symbol."""
        rows = pl.dynamic("CALLEE_ROWS")
        shared = pl.dynamic("SHARED_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def double(self, x: pl.Tensor[[rows, 4], pl.FP32], marker: pl.Tensor[[shared, 4], pl.FP32]):
                y = pl.add(x, x)
                return y

            @pl.function
            def main(self, a: pl.Tensor[[shared + 4, 4], pl.FP32], b: pl.Tensor[[16, 4], pl.FP32]):
                result = self.double(a, b)
                return result

        after = passes.inline_functions()(Before)
        main = after.get_function("main")
        assert main is not None
        assert isinstance(main.body, ir.SeqStmts)
        assign = main.body.stmts[0]
        assert isinstance(assign, ir.AssignStmt)
        assert isinstance(assign.var.type, ir.TensorType)
        assert isinstance(main.params[0].type, ir.TensorType)
        ir.assert_structural_equal(assign.var.type, main.params[0].type)
        assert isinstance(assign.value, ir.Call)
        assert assign.value.args[0] is main.params[0]
        passes.convert_to_ssa()(after)
        ir.assert_structural_equal(after, pl.parse_program(after.as_python()))

    def test_local_extent_through_loop_carried_tensor(self):
        """Callee-only dimensions must be bound to the caller's runtime extent (#2936)."""
        rows = pl.dynamic("CALLEE_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def fill(self, out: pl.Tensor[[rows, 4], pl.FP32], src: pl.Tensor[[4, 4], pl.FP32]):
                for i in pl.range(2):
                    updated = pl.tensor.assemble(out, src, [0, 0])
                    out = updated
                return out

            @pl.function
            def main(self, n: pl.Scalar[pl.INDEX], src: pl.Tensor[[4, 4], pl.FP32]):
                extent = n + 4
                out = pl.create_tensor([extent, 4], pl.FP32)
                result = self.fill(out, src)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, n: pl.Scalar[pl.INDEX], src: pl.Tensor[[4, 4], pl.FP32]):
                extent = n + 4
                out = pl.create_tensor([extent, 4], pl.FP32)
                for i in pl.range(2):
                    updated = pl.tensor.assemble(out, src, [0, 0])
                    out = updated
                result = out
                return result

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        passes.convert_to_ssa()(after)

    def test_void_helper_specializes_submit_result_and_preserves_output_storage(self):
        """A Submit result can use a local handle while preserving the output storage."""
        rows = pl.dynamic("CALLEE_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def store(
                self, src: pl.Tensor[[4, 4], pl.FP32], out: pl.Out[pl.Tensor[[rows, 4], pl.FP32]]
            ) -> pl.Tensor[[rows, 4], pl.FP32]:
                out = pl.tensor.assemble(out, src, [0, 0])
                return out

            @pl.function(type=pl.FunctionType.Inline)
            def fill(self, out: pl.Tensor[[rows, 4], pl.FP32], src: pl.Tensor[[4, 4], pl.FP32]):
                with pl.manual_scope():
                    updated, tid = pl.submit(self.store, src, out)
                    out = updated

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, n: pl.Scalar[pl.INDEX], src: pl.Tensor[[4, 4], pl.FP32]):
                extent = n + 4
                out = pl.create_tensor([extent, 4], pl.FP32)
                self.fill(out, src)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def store(
                self, src: pl.Tensor[[4, 4], pl.FP32], out: pl.Out[pl.Tensor[[rows, 4], pl.FP32]]
            ) -> pl.Tensor[[rows, 4], pl.FP32]:
                out = pl.tensor.assemble(out, src, [0, 0])
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, n: pl.Scalar[pl.INDEX], src: pl.Tensor[[4, 4], pl.FP32]):
                extent = n + 4
                out = pl.create_tensor([extent, 4], pl.FP32)
                local = out
                with pl.manual_scope():
                    updated, tid = pl.submit(self.store, src, local)
                    local = updated
                return out

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        passes.convert_to_ssa()(after)

    def test_each_call_binds_its_own_static_extent(self):
        """A shared inline helper must specialize local types independently at each call."""
        rows = pl.dynamic("CALLEE_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Inline)
            def double(self, x: pl.Tensor[[rows, 4], pl.FP32]):
                y = pl.add(x, x)
                return y

            @pl.function
            def main(self, a: pl.Tensor[[8, 4], pl.FP32], b: pl.Tensor[[16, 4], pl.FP32]):
                first = self.double(a)
                second = self.double(b)
                return first, second

        @pl.program
        class Expected:
            @pl.function
            def main(self, a: pl.Tensor[[8, 4], pl.FP32], b: pl.Tensor[[16, 4], pl.FP32]):
                y0 = pl.add(a, a)
                first = y0
                y1 = pl.add(b, b)
                second = y1
                return first, second

        after = passes.inline_functions()(Before)
        ir.assert_structural_equal(after, Expected)
        passes.convert_to_ssa()(after)


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
