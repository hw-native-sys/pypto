# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""OutlineGraphScopes — ``with pl.graph("name"):`` becomes a FunctionType.Graph function.

The point of the pass is that the scope form and ``@pl.jit.graph`` converge here:
everything downstream (LegalizeGraphBoundary, the Graph verifier, codegen) sees a
Graph function plus a Call either way, and none of it needs to know which surface
the user wrote. The tests assert that convergence, not just that something was
outlined.
"""

import pypto
import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.language.parser.diagnostics import ParserSyntaxError


def _graph_funcs(prog):
    """Graph-typed functions in ``prog``, in program order."""
    return [f for f in prog.functions.values() if f.func_type == ir.FunctionType.Graph]


class TestOutlineGraphScopes:
    """Outlining ``pl.graph`` regions into Graph functions."""

    def test_outline_simple_graph_scope(self):
        """A Graph region becomes a Graph function named after the region."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.graph("layer"):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.layer(x)
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_graph_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_scope_form_matches_decorator_form(self):
        """The outlined program is the one ``@pl.jit.graph`` would have produced.

        This is the pass's whole contract. Asserting it structurally means a
        later change that makes the scope form diverge — a different function
        type, a missing level, an extra wrapper — fails here rather than in some
        distant codegen assertion.

        Full structural equality holds here because the region captures its one
        value in the same position the decorator form declares it. In general the
        outliner appends parameters in *capture* order, so a region capturing
        several values can produce a permutation of the declared signature; the
        ST asserts boundary arity rather than order for that reason.
        """

        @pl.program
        class ScopeForm:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.graph("layer"):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        @pl.program
        class DecoratorForm:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.layer(x)
                return y

        outlined = passes.outline_graph_scopes()(passes.convert_to_ssa()(ScopeForm))
        reference = passes.convert_to_ssa()(DecoratorForm)
        ir.assert_structural_equal(outlined, reference)

    def test_outlined_function_carries_orchestration_level_and_role(self):
        """A Graph body is chip-level orchestration, same as the decorator form.

        The pass stamps nothing — ``Function``'s constructor derives both for any
        orchestration-like type. Pinning it here means a change to that
        derivation shows up as a Graph losing its level rather than as a codegen
        surprise much later.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.graph("layer"):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        After = passes.outline_graph_scopes()(passes.convert_to_ssa()(Before))
        graphs = _graph_funcs(After)
        assert len(graphs) == 1
        assert graphs[0].name == "layer"
        assert graphs[0].level == ir.Level.CHIP
        assert graphs[0].role == ir.Role.Orchestrator

    def test_opaque_parent_is_not_promoted(self):
        """Deliberately *not* the ``Opaque -> Orchestration`` promotion InCore does.

        Promoting here would make any Opaque helper that happens to carry a Graph
        region eligible to be picked as the compiled entry (the backend takes the
        first Orchestration function), so which function a program compiles to
        could change with an unrelated edit.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.graph("layer"):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        After = passes.outline_graph_scopes()(passes.convert_to_ssa()(Before))
        main = After.get_function("main")
        assert main is not None
        assert main.func_type == ir.FunctionType.Opaque

    def test_orchestration_parent_stays_orchestration(self):
        """The real entry shape: a @pl.jit body keeps its type across the pass."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.graph("layer"):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        After = passes.outline_graph_scopes()(passes.convert_to_ssa()(Before))
        main = After.get_function("main")
        assert main is not None
        assert main.func_type == ir.FunctionType.Orchestration

    def test_incore_scopes_inside_a_graph_region_outline_afterwards(self):
        """The InCore scope inside the region outlines on the pass-order path.

        OutlineGraphScopes runs immediately before OutlineIncoreScopes, and that
        pass keeps a Graph function Graph. If either half regressed, the region
        would reach codegen either un-outlined or as a plain Orchestration
        function, both of which lose the feature silently.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.graph("layer"):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        ssa = passes.convert_to_ssa()(Before)
        After = passes.outline_incore_scopes()(passes.outline_graph_scopes()(ssa))

        graphs = _graph_funcs(After)
        assert len(graphs) == 1
        assert graphs[0].name == "layer"
        incore = [f for f in After.functions.values() if f.func_type == ir.FunctionType.InCore]
        assert len(incore) == 1

    def test_capture_written_only_by_an_inner_kernel_is_not_left_In(self):
        """A capture handed to a callee's ``InOut`` slot must not come out ``In``.

        The region never touches ``out`` itself — its only use is the argument of
        a kernel *call*. Write evidence for a call is answered from the operator
        registry, which knows nothing about a ``GlobalVar`` callee, so without
        resolving the callee the capture keeps the seeded ``In``.

        That is the silent direction: an ``In`` boundary tensor is not a writer,
        so it loses the RAW edge against whoever reads it next, where an
        over-declared one would only over-order. ``ConvertTensorToTileOps``
        happens to repair this later, but it is a tensor->tile pass, not a
        direction-inference one; this pass must not depend on that.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer(
                self,
                src: pl.Tensor[[64], pl.FP32],
                dst: pl.InOut[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(src, [0], [64])
                dst = pl.store(t, [0], dst)
                return dst

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.InOut[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.graph("layer"):
                    out = self.writer(x, out)
                return out

        After = passes.outline_graph_scopes()(passes.convert_to_ssa()(Before))

        graphs = _graph_funcs(After)
        assert len(graphs) == 1
        directions = dict(zip([p.name_hint for p in graphs[0].params], graphs[0].param_directions))
        written = [n for n in directions if n.startswith("out")]
        assert len(written) == 1, f"expected one 'out' capture, got {list(directions)}"
        assert directions[written[0]] == ir.ParamDirection.InOut, (
            f"capture written by the inner kernel came out {directions[written[0]]}, "
            f"not InOut; all directions: {directions}"
        )

    def test_capture_read_by_a_submit_predicate_is_not_left_Out(self):
        """A predicate read counts, even when the same launch overwrites the value.

        ``predicate`` and ``core_num`` are first-class SSA operands on ``Submit``,
        not metadata. The read collector replaces the base walk rather than
        extending it, so anything reachable only through them has to be visited
        explicitly or it is invisible.

        Here ``rc`` goes to ``gate``'s ``Out`` slot — so the argument walk skips
        it as a pure write — and is read by that same launch's predicate, which
        the runtime evaluates on the *incoming* contents before deciding whether
        to launch. The predicate is therefore ``rc``'s only read, and dropping it
        yields ``Out``: a boundary tensor with no input edge feeding the very
        value the predicate tests. The argument skip is what exposes this, so the
        two are only ever wrong together.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def gate(self, g: pl.Out[pl.Tensor[[512, 128], pl.INT32]]) -> pl.Tensor[[512, 128], pl.INT32]:
                t = pl.load(g, [0, 0], [128, 128])
                g = pl.store(t, [0, 0], g)
                return g

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, rc: pl.InOut[pl.Tensor[[512, 128], pl.INT32]]) -> pl.Tensor[[512, 128], pl.INT32]:
                with pl.graph("layer"):
                    with pl.manual_scope():
                        rc, _ = pl.spmd_submit(self.gate, rc, core_num=1, predicate=(rc[0, 0] > 0))
                return rc

        After = passes.outline_graph_scopes()(passes.convert_to_ssa()(Before))

        graphs = _graph_funcs(After)
        assert len(graphs) == 1
        directions = dict(zip([p.name_hint for p in graphs[0].params], graphs[0].param_directions))
        read = [n for n in directions if n.startswith("rc")]
        assert len(read) == 1, f"expected one 'rc' capture, got {list(directions)}"
        assert directions[read[0]] == ir.ParamDirection.InOut, (
            f"capture read by the submit predicate came out {directions[read[0]]}, "
            f"not InOut; all directions: {directions}"
        )

    def test_two_regions_sharing_a_name_get_distinct_functions(self):
        """Distinct regions must not collapse onto one graph key.

        The emitted symbol comes from the function name, and the runtime caches
        one Definition per symbol — so two regions that both asked to be called
        "layer" must end up with different names, or the second would replay the
        first one's recorded topology.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.graph("layer"):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.graph("layer"):
                    z: pl.Tensor[[64], pl.FP32] = pl.add(y, y)
                return z

        After = passes.outline_graph_scopes()(passes.convert_to_ssa()(Before))
        names = [f.name for f in _graph_funcs(After)]
        assert len(names) == 2
        assert len(set(names)) == 2, f"graph regions collapsed onto one name: {names}"

    def test_region_inside_a_loop_is_outlined_once(self):
        """The N-layer case: one recording, N call sites — one function either way."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for _ in pl.range(4):
                    with pl.graph("layer"):
                        x = pl.add(x, x)
                return x

        After = passes.outline_graph_scopes()(passes.convert_to_ssa()(Before))
        assert len(_graph_funcs(After)) == 1


class TestOutlineGraphScopesRejections:
    """Cases the pass must reject rather than lower into a silent runtime fallback."""

    def test_nested_graph_regions_are_rejected(self):
        """The runtime cannot record a graph inside a graph, and falls back silently.

        The parser rejects the textually nested form, so this drives the pass's
        own guard by building the IR directly — the guard is the invariant, the
        parser check is the good error message.
        """

        @pl.program
        class Single:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.graph("inner"):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        class WrapInOuterGraph(ir.IRMutator):
            """Wrap the one Graph region in a second one, which the parser forbids."""

            def visit_graph_scope_stmt(self, op):
                return ir.GraphScopeStmt("outer", body=op, span=op.span)

        nested = ir.Program(
            [
                ir.Function(
                    f.name,
                    list(f.params),
                    list(f.return_types),
                    WrapInOuterGraph().visit_stmt(f.body),
                    f.span,
                    type=f.func_type,
                    level=f.level,
                    role=f.role,
                )
                for f in Single.functions.values()
            ],
            Single.name,
            Single.span,
        )

        with pytest.raises(ValueError, match=r"nested inside pl\.graph"):
            passes.outline_graph_scopes()(nested)

    @staticmethod
    def _inline_function_holding_a_graph_region():
        """An SSA program whose ``Inline`` body still carries a ``pl.graph`` region.

        The parser permits this deliberately: ``InlineFunctions`` splices such a body
        into its orchestration caller before ``OutlineGraphScopes`` runs. Reaching the
        pass with it means the pipeline was ordered wrong.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Inline)
            def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.graph("layer"):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = self.helper(x)
                return z

        return passes.convert_to_ssa()(Prog)

    def test_inline_graph_region_is_rejected_by_pre_verification(self):
        """With verification on, the declared requirement reports the mis-ordering."""
        prog = self._inline_function_holding_a_graph_region()
        with pytest.raises(pypto.Error, match=r"InlineFunctionsEliminated"):
            passes.outline_graph_scopes()(prog)

    def test_inline_graph_region_is_rejected_without_verification(self):
        """The pass rejects it on its own, with verification disabled.

        This is the case the declared requirement cannot cover: ``required`` is checked
        only by ``VerificationInstrument``, so without one a skipped function would
        otherwise keep its ``GraphScopeStmt`` while the pass still advertises
        ``GraphOutlined`` — a false property reaching codegen unnoticed.
        """
        prog = self._inline_function_holding_a_graph_region()
        with passes.PassContext([]):
            with pytest.raises(ValueError, match=r"survives in function 'helper'"):
                passes.outline_graph_scopes()(prog)

    def test_inline_functions_eliminated_is_declared_required(self):
        """The ordering the rejection enforces is also declared as a pass property.

        With verification enabled this reports the mis-ordered pipeline before the pass
        runs, instead of leaving the rejection above as the only signal.
        """
        required = passes.outline_graph_scopes().get_required_properties()
        assert required.contains(passes.IRProperty.InlineFunctionsEliminated)
        assert required.contains(passes.IRProperty.SSAForm)


class TestOutlineGraphScopesFastPath:
    """A program with no Graph region must not pay for the outliner."""

    def test_graph_free_program_is_returned_unchanged(self):
        """No ``pl.graph`` means the ScopeOutliner never runs.

        ScopeOutliner recomputes the used-after set over the whole statement suffix once
        per statement, so running it costs O(M^2) in a function of M statements. This
        pass runs on every default compilation, so a Graph-free body has to take the
        linear path — and taking it must not perturb the IR.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_graph_scopes()(Before)
        ir.assert_structural_equal(After, Before)
        assert _graph_funcs(After) == []


class TestGraphScopeParsing:
    """Frontend surface: what ``pl.graph`` accepts and what it refuses."""

    def test_region_name_is_required(self):
        with pytest.raises(ParserSyntaxError, match=r"exactly one positional argument"):

            @pl.program
            class Bad:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.graph():
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    return y

    def test_region_name_must_be_a_string_literal(self):
        with pytest.raises(ParserSyntaxError, match=r"must be a string literal"):

            @pl.program
            class Bad:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    # A non-string literal is exactly what this test feeds the parser.
                    with pl.graph(4):  # type: ignore[arg-type]
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    return y

    def test_rejected_inside_an_incore_scope(self):
        """A device scope is one task; a Graph region is a topology of them."""
        with pytest.raises(ParserSyntaxError, match=r"cannot be nested inside `pl\.at"):

            @pl.program
            class Bad:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.at(level=pl.Level.CORE_GROUP):
                        with pl.graph("layer"):
                            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    return y

    def test_rejected_inside_a_hierarchy_scope(self):
        """Graph x distributed is out of scope for now (RFC #2399, open question 4).

        Rejected at the surface rather than left to reach a pass that has never
        seen the combination.
        """
        with pytest.raises(ParserSyntaxError, match=r"cannot be nested inside `pl\.at"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.at(level=pl.Level.HOST):
                        with pl.graph("layer"):
                            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    return y

    def test_rejected_inside_an_incore_function(self):
        with pytest.raises(ParserSyntaxError, match=r"not valid inside a InCore function"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.graph("layer"):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    return y

    def test_textually_nested_regions_are_rejected_at_parse_time(self):
        with pytest.raises(ParserSyntaxError, match=r"cannot be nested inside another pl\.graph"):

            @pl.program
            class Bad:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.graph("outer"):
                        with pl.graph("inner"):
                            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    return y

    def test_as_clause_is_rejected(self):
        with pytest.raises(ParserSyntaxError, match=r"as \.\.\.:` is not supported"):

            @pl.program
            class Bad:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.graph("layer") as _tid:
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    return y


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
