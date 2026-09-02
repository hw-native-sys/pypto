# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""LegalizeGraphBoundary — Step A (scalar hoisting) and Step D (boundary legality).

Step A exists because a boundary scalar is tracked by the *address* of its
argument slot. A value the Graph body derives (``base = layer * 5120``) has no
slot, so the runtime classifies it as static data and freezes the first call's
value into the recorded graph — silently, on every later replay. Hoisting the
computation to the call site turns it back into a real pass-through scalar.

Step D rejects boundaries the runtime would decline to cache. Those failures are
silent non-graph fallbacks in a release build: the program stays correct and the
feature simply does nothing, which no numerical test can detect. Hence the
negative cases below assert on the message, not just the raise.
"""

import re

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.printer import python_print


def _legalize(program: ir.Program) -> ir.Program:
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        return passes.legalize_graph_boundary()(passes.convert_to_ssa()(program))


def _legalize_outlined(program: ir.Program) -> ir.Program:
    """``_legalize`` with the scope outliner in front, as the real pipeline has it.

    ``OutlineIncoreScopes`` rewrites an in-place device scope into a call, which
    rebinds the InOut parameter the body writes through::

        c_1 = layer_incore_0(a, c)
        return c_1

    That is the shape every Graph body actually has by the time this pass runs,
    so a check written against the pre-outlining shape can reject the entire
    feature while ``_legalize`` still passes.

    ``NormalizeReturnOrder`` is in the chain for the same reason. It runs at
    pass 26 and this pass at 45, and it is what establishes
    ``IRProperty::ReturnParamsExplicit`` — the pointer-identity return -> param
    map the pass reads to carry boundary provenance across an in-place call
    (``tmp = kernel(a, tmp)``). Without it that map answers nullopt for every
    callee, the propagation cannot fire, and a test asserting on it passes for
    the wrong reason.
    """
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        ssa = passes.convert_to_ssa()(program)
        outlined = passes.outline_incore_scopes()(ssa)
        return passes.legalize_graph_boundary()(passes.normalize_return_order()(outlined))


def _graph_func(program: ir.Program, name: str) -> ir.Function:
    func = program.get_function(name)
    assert func is not None
    return func


def _scalar_param_names(func: ir.Function) -> list[str]:
    """Scalar parameter names with ConvertToSSA's ``__ssa_vN`` suffix stripped."""
    return [re.sub(r"__ssa_v\d+$", "", p.name_hint) for p in func.params if isinstance(p.type, ir.ScalarType)]


# ---------------------------------------------------------------------------
# Step A — derived boundary scalars are hoisted to the call sites
# ---------------------------------------------------------------------------


class TestDerivedScalarHoisting:
    def test_derived_scalar_becomes_a_parameter(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                layer_idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                base = layer_idx * 128
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [base, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for i in pl.range(4):
                    self.layer(a, c, i)
                return c

        After = _legalize_outlined(Before)
        layer = _graph_func(After, "layer")
        # The derived value now arrives as a parameter instead of being computed
        # inside the region, so the runtime can anchor its argument slot.
        assert "base" in _scalar_param_names(layer)

    def test_passthrough_scalar_is_left_alone(self):
        """A bare parameter reference already has a slot; nothing to hoist."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                offset: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                self.layer(a, c, 0)
                return c

        After = _legalize_outlined(Before)
        assert _scalar_param_names(_graph_func(After, "layer")) == ["offset"]

    def test_multi_level_derived_scalars_are_fully_expanded(self):
        """A hoist defined in terms of an earlier hoist must not name it at the call site.

        Step A erases the body definitions it hoists, so an expression captured
        as ``end = base + 128`` would leave the caller passing a ``base`` that
        exists nowhere. Each substitution has to feed the next.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                base = idx * 128
                end = base + 128
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [base, 0], [128, 128])
                    u: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [end - 128, 0], [128, 128])
                    pl.store(pl.add(t, u), [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for i in pl.range(4):
                    self.layer(a, c, i)
                return c

        After = _legalize_outlined(Before)
        assert _scalar_param_names(_graph_func(After, "layer")) == ["idx", "base", "end"]

        # The caller must name only things it has: its own loop variable and
        # constants. A leftover reference to the callee's `base` would be
        # defined nowhere at all, since Step A erased its definition.
        #
        # Asserted on the whole caller rather than the call line, because the
        # argument may be spelled inline or bound to a local first — only the
        # absence of the erased name is contractual.
        caller = python_print(_graph_func(After, "main"))
        assert not re.search(r"\bbase__ssa_v\d+\b", caller)
        # The hoisted arithmetic really did move here, in the caller's own terms.
        assert re.search(r"i__\w+ \* 128", caller)

    def test_inline_computed_scalar_argument_is_rejected(self):
        """Step A hoists named bindings; an inline expression has no name to hoist.

        `self.kernel(a, d, idx * 128)` reaches the task as a computed value with
        no boundary slot, so the runtime freezes the first call's number into the
        recording — the same silent failure as the named case, but previously
        waved through because the argument is not a Var.

        (An inline offset inside a `pl.at` scope is *not* this case: the scope
        outliner moves the arithmetic into the outlined kernel and passes the
        bare parameter, so it keeps its slot.)
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                off: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    a, [off, 0], [128, 128], target_memory=pl.MemorySpace.Vec
                )
                return pl.store(t, [0, 0], out)

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                d: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                d = self.kernel(a, d, idx * 128)
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                d: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c, d, idx)
                return c

        with pytest.raises(ValueError, match="computes a scalar inline in a task argument"):
            _legalize(Before)

    def test_literal_scalar_argument_is_allowed(self):
        """A literal is the same on every call, so freezing it is harmless."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [128, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c)
                return c

        assert _graph_func(_legalize_outlined(Before), "layer").func_type == ir.FunctionType.Graph

    def test_alias_of_a_parameter_does_not_leak_to_the_caller(self):
        """A hoisted expression written through an alias must resolve to the param.

        `alias = idx` is not hoisted — it names a parameter that already has a
        slot — but `base = alias * 128` is, and the substitution seeded with only
        parameters and hoisted vars leaves `alias` in place. The caller then
        references a name that exists only inside the Graph.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                alias = idx
                base = alias * 128
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(w, [base, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for i in pl.range(4):
                    self.layer(w, c, i)
                return c

        caller = python_print(_graph_func(_legalize_outlined(Before), "main"))
        # The printer marks an unbound reference with __FREE_VAR, which is
        # exactly what a leaked Graph-local alias produces here.
        assert "__FREE_VAR" not in caller, caller
        assert not re.search(r"\balias__ssa_v\d+\b", caller), caller

    def test_a_pass_through_defined_after_the_last_hoist(self):
        """Definition order can end on an alias, with no hoist left to compare it against.

        The inverse of the test above: `base` is hoisted and `col` merely names
        it, so the call site's merge runs out of hoists while a pass-through is
        still pending. Deciding whether the alias came next read
        `by_definition[next_hoist]` before checking a hoist was left — one past
        the end of a non-empty vector, then dereferenced through `->original`.

        The merge *result* was never wrong (the guard one line below forces the
        alias to be taken once the hoists are gone), so this asserts the shape
        the merge must produce and pins the path a sanitizer build watches.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                base = idx * 128
                col = base
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(w, [col, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for i in pl.range(4):
                    self.layer(w, c, i)
                return c

        After = _legalize_outlined(Before)
        assert "base" in _scalar_param_names(_graph_func(After, "layer"))
        caller = python_print(_graph_func(After, "main"))
        # `col` names a value only the Graph has; the caller must pass `base`.
        assert "__FREE_VAR" not in caller, caller
        assert not re.search(r"\bcol__ssa_v\d+\b", caller), caller

    def test_non_graph_program_is_untouched(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

        ssa = passes.convert_to_ssa()(Before)
        with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
            After = passes.legalize_graph_boundary()(ssa)
        ir.assert_structural_equal(ssa, After)


class TestRuntimeGate:
    """A Graph only exists under `host_build_graph`.

    Codegen emits `GraphTaskArgs` and `rt_submit_graph` unconditionally, and the
    `tensormap_and_ringbuffer` orchestration API declares neither, so compiling a
    Graph against the default runtime yields orchestration C++ that names
    undeclared symbols. Every other test here sets the runtime explicitly, which
    is exactly why the default path had no coverage.
    """

    @staticmethod
    def _program():
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c)
                return c

        return Before

    def test_graph_under_the_default_runtime_is_rejected(self):
        program = self._program()
        with passes.PassContext([]):  # default: tensormap_and_ringbuffer
            ssa = passes.convert_to_ssa()(program)
            outlined = passes.outline_incore_scopes()(ssa)
            with pytest.raises(ValueError, match="requires the host_build_graph runtime"):
                passes.legalize_graph_boundary()(outlined)

    def test_graph_under_host_build_graph_is_accepted(self):
        _legalize_outlined(self._program())


class TestLaunchSpec:
    """`core_num` is frozen into the recorded node, not patched on replay.

    Recording stores `logical_block_num = block_num` and replay copies it back
    from the Definition (`slot.logical_block_num = source.logical_block_num`);
    only boundary tensors and scalar slots are refreshed. A `core_num` read from
    a boundary scalar therefore replays call one's block count for every later
    call — the rest of the blocks are never scheduled, with no diagnostic.
    """

    def test_core_num_from_a_boundary_scalar_is_rejected(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                x: pl.Tensor[[128, 128], pl.FP32],
                o: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(x, [0, 0], [128, 128])
                o = pl.store(t, [0, 0], o)
                return o

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                blocks: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.manual_scope():
                    c, _ = pl.spmd_submit(self.kernel, a, c, core_num=blocks)
                return c

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c, 4)
                return c

        with pytest.raises(ValueError, match="core_num is not a compile-time constant"):
            _legalize_outlined(Before)

    def test_core_num_derived_from_a_boundary_scalar_is_rejected(self):
        """Step A would hoist `blocks * 2` into a parameter, which makes it a
        replay-patchable *argument* but not a replay-patchable *launch spec*."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                x: pl.Tensor[[128, 128], pl.FP32],
                o: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(x, [0, 0], [128, 128])
                o = pl.store(t, [0, 0], o)
                return o

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                blocks: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.manual_scope():
                    c, _ = pl.spmd_submit(self.kernel, a, c, core_num=blocks * 2)
                return c

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c, 4)
                return c

        with pytest.raises(ValueError, match="core_num is not a compile-time constant"):
            _legalize_outlined(Before)

    def test_constant_core_num_is_accepted(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                x: pl.Tensor[[128, 128], pl.FP32],
                o: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(x, [0, 0], [128, 128])
                o = pl.store(t, [0, 0], o)
                return o

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                blocks: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.manual_scope():
                    c, _ = pl.spmd_submit(self.kernel, a, c, core_num=4)
                return c

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c, 4)
                return c

        _legalize_outlined(Before)


# ---------------------------------------------------------------------------
# Step B — derived slices of a boundary tensor are taken at the call site
# ---------------------------------------------------------------------------


def _tensor_param_names(func: ir.Function) -> list[str]:
    return [
        re.sub(r"__ssa_v\d+$", "", p.name_hint) for p in func.params if not isinstance(p.type, ir.ScalarType)
    ]


def _param_directions(func: ir.Function) -> dict[str, ir.ParamDirection]:
    """Parameter name (SSA suffix stripped) -> declared direction."""
    return {re.sub(r"__ssa_v\d+$", "", p.name_hint): d for p, d in zip(func.params, func.param_directions)}


class TestDerivedSliceHoisting:
    def test_slice_of_a_boundary_tensor_becomes_a_parameter(self):
        """Replay patches a boundary tensor's address, not a view derived inside.

        A view taken in the region is re-derived from whatever the recording
        froze, so it has to be taken at the call site and passed in.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                layer_idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                base = layer_idx * 128
                wl: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(w, [128, 128], [base, 0])
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(wl, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for i in pl.range(4):
                    c = self.layer(w, c, i)
                return c

        layer = _graph_func(_legalize_outlined(Before), "layer")
        assert "wl" in _tensor_param_names(layer)
        # Within the appended parameters, tensors precede scalars. IR order as a
        # whole need not be tensors-first — codegen's stable reorder produces the
        # tensors-before-scalars `CoreTaskArgs` order the runtime requires, and
        # the graph body's `args.tensor(i)` / `args.scalar(k)` indices are
        # assigned by counting each kind separately, so they agree either way.
        appended = [isinstance(p.type, ir.ScalarType) for p in layer.params[3:]]
        assert appended == sorted(appended), f"appended tensors must come first, got {appended}"

    def test_a_bare_alias_of_a_scalar_parameter_is_not_rejected(self):
        """A pass-through already has a slot; there is nothing to reconstruct.

        Step A classifies `alias = layer_idx` as a pass-through rather than a
        hoist, so the assignment survives the rewrite. `ConvertToSSA` produces
        this shape routinely (`layer_idx__ssa_v1 = layer_idx`), so treating the
        survivor as unhoistable rejects ordinary correct programs.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                layer_idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                alias = layer_idx
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [alias, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for i in pl.range(4):
                    c = self.layer(a, c, i)
                return c

        _legalize_outlined(Before)

    def test_a_scalar_alias_is_substituted_away_rather_than_left_in_the_body(self):
        """A rename does not inherit the parameter's argument slot.

        Accepting the alias is not enough — it has to stop existing. Orchestration
        codegen emits a survivor as a value copy (`int64_t n = batch;`) and then
        `add_scalar(n)`, but recording matches a boundary scalar by the *address*
        its value came from (`&boundary_args->scalar(i)`). The copy lives at a
        different address, so it is recorded as static data and every later
        replay reuses the first call's number — the exact silent freeze Step A
        exists to prevent, reintroduced by a bare rename.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                layer_idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                alias = layer_idx
                again = alias
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [again, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for i in pl.range(4):
                    c = self.layer(a, c, i)
                return c

        layer = _graph_func(_legalize_outlined(Before), "layer")
        printed = python_print(layer)
        # The whole chain collapses onto the parameter, so the task reads the
        # slot itself. Both names must be gone, not just the last one.
        assert "alias" not in printed, printed
        assert "again" not in printed, printed
        assert "layer_idx" in printed, printed

    def test_a_view_of_a_hoisted_allocation_moves_out_with_it(self):
        """A region allocation is a boundary tensor, so its views are too.

        Once Step C hoists ``local``, it is a boundary parameter — and a view of
        a boundary tensor taken *inside* the region is exactly what Step B
        exists to move out. Leaving ``lv`` behind would also make this pass
        produce IR its own ``GraphBoundaryLegalized`` verifier rejects: that
        verifier treats every tensor parameter as a boundary root and holds any
        in-region view of one to the replay-invariant-window rule.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                local: pl.Tensor[[256, 128], pl.FP32] = pl.create_tensor([256, 128], pl.FP32)
                lv: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(local, [128, 128], [0, 0])
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(lv, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(w, c)
                return c

        layer = _graph_func(_legalize_outlined(Before), "layer")
        # The allocation first, then the view of it — both appended as InOut.
        assert _tensor_param_names(layer) == ["w", "c", "local", "lv"]
        assert _param_directions(layer)["local"] == ir.ParamDirection.InOut
        assert _param_directions(layer)["lv"] == ir.ParamDirection.InOut

    def test_slice_of_a_local_tensor_is_left_alone(self):
        """Only views *of a boundary tensor* need hoisting."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(w, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(w, c)
                return c

        layer = _graph_func(_legalize_outlined(Before), "layer")
        assert _tensor_param_names(layer) == ["w", "c"]

    def test_a_view_of_a_hoisted_view_is_hoisted_too(self):
        """Once `wl` moves out it is a boundary tensor, so `wr` is one too.

        Leaving `wr` behind is silent: recording classifies it as a view of
        `wl` and freezes the offset computed on the first call, so replay
        patches `wl`'s address but keeps call one's window.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                layer_idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                base = layer_idx * 128
                wl: pl.Tensor[[256, 128], pl.FP32] = pl.tensor.slice(w, [256, 128], [base, 0])
                wr: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(wl, [128, 128], [base, 0])
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(wr, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for i in pl.range(2):
                    c = self.layer(w, c, i)
                return c

        layer = _graph_func(_legalize_outlined(Before), "layer")
        names = _tensor_param_names(layer)
        assert "wl" in names and "wr" in names, names

    def test_a_view_through_a_tensor_alias_is_hoisted(self):
        """Provenance follows a bare tensor alias.

        `alias = w` then `slice(alias, ...)`: the immediate source is neither an
        original parameter nor a collected view, so without provenance the view
        stays in the region and the recording freezes the first call's offset.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                layer_idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                alias = w
                wl: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(alias, [128, 128], [layer_idx * 128, 0])
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(wl, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for i in pl.range(4):
                    c = self.layer(w, c, i)
                return c

        After = _legalize_outlined(Before)
        layer = _graph_func(After, "layer")
        assert "wl" in _tensor_param_names(layer), _tensor_param_names(layer)

        # The caller must name only things it has. `alias` exists solely inside
        # the Graph, so a leftover reference is printed `__FREE_VAR` — asserting
        # on the parameter list alone would not catch it, which is how this got
        # through the first time.
        caller = python_print(_graph_func(After, "main"))
        assert "__FREE_VAR" not in caller, caller
        assert "alias" not in caller, caller
        assert "pl.tensor.slice(w" in caller, caller

    def test_an_alias_of_a_hoisted_view_resolves_at_the_call_site(self):
        """An alias may name an earlier *hoisted view*, not just a parameter.

        That is why the call site replays hoists and aliases in one definition
        order rather than binding scalars then tensors: `mid` is only bound once
        `wl`'s own hoist has run.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                layer_idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                wl: pl.Tensor[[256, 128], pl.FP32] = pl.tensor.slice(w, [256, 128], [layer_idx * 128, 0])
                mid = wl
                inner: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(mid, [128, 128], [0, 0])
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(inner, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for i in pl.range(2):
                    c = self.layer(w, c, i)
                return c

        After = _legalize_outlined(Before)
        names = _tensor_param_names(_graph_func(After, "layer"))
        assert "wl" in names and "inner" in names, names
        caller = python_print(_graph_func(After, "main"))
        assert "__FREE_VAR" not in caller, caller
        assert "mid" not in caller, caller

    def test_a_hoisted_view_takes_its_roots_direction(self):
        """A view of an `InOut` tensor is `InOut`, not `In`.

        Declared `In`, codegen emits `add_input(view)` and the graph launch is
        never registered as a writer of that buffer, so a consumer downstream can
        be ordered against the pre-write contents.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
                layer_idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                ov: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(out, [128, 128], [layer_idx * 128, 0])
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], ov)
                return out

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.range(4):
                    out = self.layer(a, out, i)
                return out

        layer = _graph_func(_legalize_outlined(Before), "layer")
        # Index over *all* params: `param_directions` is parallel to `params`,
        # while `_tensor_param_names` drops the scalars.
        idx = [re.sub(r"__ssa_v\d+$", "", p.name_hint) for p in layer.params].index("ov")
        assert layer.param_directions[idx] == ir.ParamDirection.InOut, layer.param_directions[idx]

    def test_a_view_with_a_non_constant_shape_is_rejected(self):
        """`graph_rebind_tensor` patches a view's address, never its shape.

        The recorded template's `shapes`/`strides` are replayed as-is, so an
        extent read from a boundary scalar would apply call one's shape to a
        later call's buffer.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                rows: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                wl: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(w, [rows, 128], [0, 0])
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(wl, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                w: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(w, c, 128)
                return c

        with pytest.raises(ValueError, match="shape is not a compile-time constant"):
            _legalize_outlined(Before)


# ---------------------------------------------------------------------------
# Step C — allocations the region makes for itself
# ---------------------------------------------------------------------------


class TestRegionAllocationHoisting:
    """A `pl.create_tensor` in the region comes off the graph heap.

    `task_allocator.h` reclaims nothing there until the run ends, so a region
    that allocates for itself holds one buffer per *submission* rather than one
    per program: a decoder layer with 14 intermediates recorded over N layers
    holds 14 x N. Step C moves each one to the call site, where it is an
    ordinary reclaimable allocation.
    """

    def test_region_allocation_becomes_an_inout_parameter(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tmp: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], tmp)
                with pl.at(level=pl.Level.CORE_GROUP):
                    u: pl.Tile[[128, 128], pl.FP32] = pl.load(tmp, [0, 0], [128, 128])
                    pl.store(u, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c)
                return c

        layer = _graph_func(_legalize_outlined(Before), "layer")
        assert _tensor_param_names(layer) == ["a", "c", "tmp"]
        # `InOut`, not `In`: the region writes it. Declared `In`, codegen emits
        # `add_input`, the launch never registers as a writer of the buffer, and
        # a caller that hoisted the allocation out of its own loop would get no
        # ordering between successive launches over it. `Out` is illegal on a
        # Graph boundary — it means the *runtime* allocates.
        assert _param_directions(layer)["tmp"] == ir.ParamDirection.InOut
        # The region no longer allocates at all.
        assert "pl.tensor.create" not in python_print(layer), python_print(layer)

    def test_the_call_site_takes_over_the_allocation(self):
        """The buffer has to be created somewhere; the caller is that somewhere.

        Bound to a local ahead of the launch rather than written inline into the
        argument list: orchestration codegen accepts only a Var or a literal as
        an argument.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tmp: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], tmp)
                with pl.at(level=pl.Level.CORE_GROUP):
                    u: pl.Tile[[128, 128], pl.FP32] = pl.load(tmp, [0, 0], [128, 128])
                    pl.store(u, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c)
                return c

        After = _legalize_outlined(Before)
        main = _graph_func(After, "main")
        printed = python_print(main)
        assert "pl.tensor.create" in printed, printed
        assert "__graph_arg" in printed, printed
        # Every parameter supplied: a Graph has no runtime-allocated tail.
        assert len(_graph_func(After, "layer").params) == 3

    def test_a_view_of_a_hoisted_allocation_with_a_moving_window_is_rejected(self):
        """Hoisting the allocation subjects its views to the Step B rule.

        Recording stores a `BOUNDARY_VIEW`'s offset as the delta seen on the
        first call and patches only the buffer address, so an offset that can
        differ between calls replays call one's window. That is now reachable
        for a view of a region allocation, because the allocation is a boundary
        tensor — and it is rejected here rather than left for the runtime to get
        silently wrong.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tmp: pl.Tensor[[512, 128], pl.FP32] = pl.create_tensor([512, 128], pl.FP32)
                for i in pl.range(4):
                    off = n + i * 128
                    tv: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(tmp, [128, 128], [off, 0])
                    with pl.at(level=pl.Level.CORE_GROUP):
                        t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                        pl.store(t, [0, 0], tv)
                with pl.at(level=pl.Level.CORE_GROUP):
                    u: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(u, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c, n)
                return c

        with pytest.raises(ValueError, match="neither reconstructible at the call site nor the same"):
            _legalize_outlined(Before)

    def test_an_allocation_under_a_loop_stays_in_the_region(self):
        """A loop-nested create is a *fresh* buffer per iteration.

        Hoisting it would collapse N buffers into one parameter, so iterations
        that used to write disjoint memory would alias — and the cross-task
        edges that would have to re-serialise them were derived by
        `AutoDeriveTaskDependencies`, well upstream of this pass.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec
                )
                return pl.store(t, [0, 0], out)

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for _ in pl.range(4):
                    tmp: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
                    tmp = self.kernel(a, tmp)
                    c = self.kernel(tmp, c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c)
                return c

        layer = _graph_func(_legalize_outlined(Before), "layer")
        assert _tensor_param_names(layer) == ["a", "c"]
        assert "pl.tensor.create" in python_print(layer), python_print(layer)

    def test_a_view_of_an_inout_rebound_allocation_moves_out_with_it(self):
        """Provenance has to survive the SSA rebind an in-place call creates.

        ``tmp = self.kernel(a, tmp)`` binds a *fresh* name to the same buffer.
        Tracking only bare `alias = var` assignments loses the boundary root
        there, so Step B skips the view of the rebound name outright — no hoist
        and no check — and a call-varying offset stays in the region with the
        first call's window frozen into the recording.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec
                )
                out = pl.store(t, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tmp: pl.Tensor[[512, 128], pl.FP32] = pl.create_tensor([512, 128], pl.FP32)
                tmp = self.kernel(a, tmp)
                tv: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(tmp, [128, 128], [n, 0])
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(tv, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c, n)
                return c

        layer = _graph_func(_legalize_outlined(Before), "layer")
        # Both the allocation and the view of its rebound name are boundary
        # tensors now; neither is left for the recording to freeze.
        assert _tensor_param_names(layer) == ["a", "c", "tmp", "tv"]
        assert "pl.tensor.slice" not in python_print(layer), python_print(layer)

    def test_a_view_of_an_inout_rebound_parameter_moves_out_with_it(self):
        """The same rebind on a plain boundary parameter — no allocation involved.

        Step B has always had this hole: it is the `tensor_root_` lookup that
        fails, not anything specific to Step C. Pinned separately so a future
        change to the allocation path cannot quietly take the parameter path
        back down with it.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec
                )
                out = pl.store(t, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                w: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                w = self.kernel(a, w)
                wl: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(w, [128, 128], [n, 0])
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(wl, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                w: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, w, c, n)
                return c

        layer = _graph_func(_legalize_outlined(Before), "layer")
        assert _tensor_param_names(layer) == ["a", "w", "c", "wl"]
        assert "pl.tensor.slice" not in python_print(layer), python_print(layer)

    def test_a_view_of_a_prefix_submit_result_moves_out_with_it(self):
        """A `Submit` may omit a runtime-allocated `Out` tail, and still writes back.

        `pl.submit(self.kernel, a, w)` against a callee declaring a trailing
        `pl.Out` is ordinary DSL: the runtime allocates the omitted parameter.
        The caller-supplied prefix still maps positionally, so the returned `w`
        is the boundary tensor the callee wrote through.

        Demanding full arity before reading that mapping would drop provenance
        for every such submit — the exact shape `Submit::args_` in `ir/expr.h`
        documents as legal — and the dynamic view below would stay in the
        region with the first invocation's window frozen.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                w: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
                scratch: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec
                )
                scratch = pl.store(t, [0, 0], scratch)
                w = pl.store(t, [0, 0], w)
                return w

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                w: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.manual_scope():
                    # `scratch` omitted — the runtime allocates it.
                    w, _tid = pl.submit(self.kernel, a, w)
                wl: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(w, [128, 128], [n, 0])
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(wl, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                w: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, w, c, n)
                return c

        layer = _graph_func(_legalize_outlined(Before), "layer")
        assert _tensor_param_names(layer) == ["a", "w", "c", "wl"]
        assert "pl.tensor.slice" not in python_print(layer), python_print(layer)

    def test_a_rebound_view_with_a_moving_window_is_rejected(self):
        """Seeing through the rebind also restores the *check*, not just the hoist.

        ``n + i * 128`` can neither be rebuilt at the call site (``i`` does not
        exist there) nor frozen (``n`` is patched every call). Before provenance
        crossed the rebind this was skipped in silence; now it reaches the same
        three-way decision every other boundary view faces, and is rejected.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec
                )
                out = pl.store(t, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                w: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                w = self.kernel(a, w)
                for i in pl.range(2):
                    off = n + i * 128
                    wl: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(w, [128, 128], [off, 0])
                    with pl.at(level=pl.Level.CORE_GROUP):
                        t: pl.Tile[[128, 128], pl.FP32] = pl.load(wl, [0, 0], [128, 128])
                        pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                w: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, w, c, n)
                return c

        with pytest.raises(ValueError, match="neither reconstructible at the call site nor the same"):
            _legalize_outlined(Before)

    def test_a_graph_return_names_its_parameter_directly(self):
        """Guards the dependency that makes a second `InOut` parameter legal.

        `NormalizeReturnOrder` canonicalizing a Graph's returns is not this
        pass's work — it landed in #2618 — but Steps B and C are unusable
        without it. Orchestration codegen maps a call result onto one of the
        callee's Out/InOut params through `ExplicitReturnedParamIndices`, a
        pointer-identity read, and falls back to "the single Out/InOut param"
        only when that map yields nothing; a Graph body is
        `c_1 = layer_incore_0(a, c); return c_1` after the scope outliner, so
        the fallback is all it ever had, and it stops existing at the second
        `InOut`.

        Pinned here, in the consumer, rather than left to #2618's own tests:
        narrowing that gate would break the hoists silently, and this is where
        that shows up.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c)
                return c

        with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
            ssa = passes.convert_to_ssa()(Before)
            outlined = passes.outline_incore_scopes()(ssa)
            before = _graph_func(outlined, "layer")
            normalized = passes.normalize_return_order()(outlined)

        # The outliner leaves the return on the rebind, which resolves to
        # nothing under a pointer-identity read.
        assert "return c__ssa_v1" in python_print(before), python_print(before)

        layer = _graph_func(normalized, "layer")
        printed = python_print(layer)
        # Now the parameter itself, by name and so by pointer identity.
        assert f"return {layer.params[1].name_hint}" in printed, printed
        # The rebind is still computed — it is a task launch with side effects.
        assert "= layer_incore_0(" in printed, printed


# ---------------------------------------------------------------------------
# Step D — boundaries the runtime could not cache
# ---------------------------------------------------------------------------


class TestBoundaryLegality:
    def test_graph_without_tensor_parameters_is_rejected(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(self, n: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                return n

            @pl.function
            def main(self) -> pl.Scalar[pl.INDEX]:
                return self.layer(0)

        with pytest.raises(ValueError, match="empty boundary"):
            _legalize(Before)

    def test_graph_returning_a_computed_value_is_rejected(self):
        """A return that aliases an InOut parameter is fine; a computed one is not.

        ``rt_submit_graph`` yields a valid task id only on a cache hit, so a
        graph cannot hand a computed value back to its caller.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Scalar[pl.INDEX]:
                m = pl.tensor.dim(a, 0)
                return m

            @pl.function
            def main(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Scalar[pl.INDEX]:
                return self.layer(a)

        with pytest.raises(ValueError, match="returns a value it computed"):
            _legalize(Before)

    def test_in_place_return_survives_outlining(self):
        """The in-place idiom must still be accepted once it lowers to a call.

        The returned value is then a *rebind* of the InOut parameter rather than
        the parameter node, so matching parameters by pointer identity would
        reject every Graph body with a device scope — which is all of them.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c)
                return c

        After = _legalize_outlined(Before)
        layer = _graph_func(After, "layer")
        # The body really was outlined, so this does not pass by being skipped.
        assert After.get_function("layer_incore_0") is not None
        assert layer.func_type == ir.FunctionType.Graph

    def test_computed_return_is_still_rejected_after_outlining(self):
        """Following the rebind must not turn the check into a rubber stamp."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Scalar[pl.INDEX]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                m = pl.tensor.dim(a, 0)
                return m

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Scalar[pl.INDEX]:
                return self.layer(a, c)

        with pytest.raises(ValueError, match="returns a value it computed"):
            _legalize_outlined(Before)

    def test_launch_in_a_loop_counts_per_iteration(self):
        """A lexical count would wave 2000 launches past the runtime's 1024 limit."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec
                )
                return pl.store(t, [0, 0], out)

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for _ in pl.range(2000):
                    c = self.kernel(a, c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c)
                return c

        with pytest.raises(ValueError, match="launches 2000 tasks, over the runtime's per-graph limit"):
            _legalize(Before)

    def test_launch_in_a_loop_with_runtime_bounds_is_rejected(self):
        """A launch count that can differ between calls cannot be recorded once."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec
                )
                return pl.store(t, [0, 0], out)

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for _ in pl.range(n):
                    c = self.kernel(a, c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c, n)
                return c

        with pytest.raises(ValueError, match="trip count is not a compile-time constant"):
            _legalize(Before)

    def test_launch_under_a_conditional_is_rejected(self):
        """Which arm ran on call one would be replayed for every later call."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec
                )
                return pl.store(t, [0, 0], out)

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                d: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                flag: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                if flag > 0:
                    d = self.kernel(a, d)
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                d: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                flag: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c, d, flag)
                return c

        with pytest.raises(ValueError, match="inside a conditional"):
            _legalize(Before)

    def test_loop_without_a_launch_keeps_runtime_bounds(self):
        """The topology rules bind launches only — plain compute loops are fine."""
        NR = pl.dynamic("NR")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[NR, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                n = pl.tensor.dim(a, 0)
                for _ in pl.range(n):
                    pass
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[NR, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c)
                return c

        assert _graph_func(_legalize_outlined(Before), "layer").func_type == ir.FunctionType.Graph

    def test_dummy_tasks_count_toward_the_node_limit(self):
        """`system.task_dummy` becomes `rt_submit_dummy_task` — a real node.

        Counting only calls to functions under-reports the topology, and
        `ExpandManualPhaseFence` inserts these automatically, so a Graph carries
        nodes its author never wrote.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                with pl.manual_scope():
                    tids = pl.array.create(4, pl.TASK_ID)
                    for _ in pl.range(2000):
                        pl.system.task_dummy(deps=[tids])
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c)
                return c

        with pytest.raises(ValueError, match="over the runtime's per-graph limit"):
            _legalize_outlined(Before)

    def test_graph_launching_nothing_is_rejected(self):
        """The runtime refuses a node count of zero, so the region never caches."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c)
                return c

        with pytest.raises(ValueError, match="launches no tasks"):
            _legalize_outlined(Before)

    def test_allocation_in_a_dynamic_loop_is_rejected(self):
        """An allocation records a node, so a varying count is a varying topology."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                for _ in pl.range(n):
                    _s: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c, n)
                return c

        with pytest.raises(ValueError, match="trip count is not a compile-time constant"):
            _legalize_outlined(Before)

    def test_allocation_with_a_runtime_shape_is_rejected(self):
        """Recording freezes the shape; replay never re-runs the body.

        A later call with a larger extent would be handed the first call's
        buffer — a wrong address layout, not a fallback.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                _s: pl.Tensor[[n], pl.FP32] = pl.create_tensor([n], pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c, n)
                return c

        with pytest.raises(ValueError, match="shape is not a compile-time constant"):
            _legalize_outlined(Before)

    def test_tensor_full_inside_a_graph_is_rejected(self):
        """Orchestration codegen has no lowering for it, so reject it here."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                _s: pl.Tensor[[16], pl.FP32] = pl.full([16], dtype=pl.FP32, value=0.0)
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c)
                return c

        with pytest.raises(ValueError, match="calls tensor.full inside the region"):
            _legalize_outlined(Before)

    def test_allocations_count_toward_the_node_limit(self):
        """A Graph at the launch limit is over it once it also allocates.

        1024 launches plus a two-trip loop holding one create is 1026 recorded
        nodes. Leaving allocations out of the total entirely would pass this and
        leave the runtime to decline the cache silently.

        The create sits under a loop so that it stays in the region: Step C
        hoists a *top-level* allocation to the call site, and one it hoists is a
        node the emitted region no longer has. Two trips rather than one because
        `Simplify` collapses a single-trip loop into its body, which would put
        the create back at the top level in the real pipeline.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec
                )
                return pl.store(t, [0, 0], out)

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for _ in pl.range(2):
                    _s: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                for _ in pl.range(1024):
                    c = self.kernel(a, c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c)
                return c

        with pytest.raises(ValueError, match="launches 1026 tasks, over the runtime's per-graph limit"):
            _legalize_outlined(Before)

    def test_interleaved_allocations_batch_across_the_statement_list(self):
        """A launch between two creates does not close the batch.

        Codegen collects every eligible create in the statement list before
        packing them 16 to an `alloc_tensors`, so 20 creates are 2 allocation
        nodes however many launches sit between them.

        The count is pinned through the over-limit message rather than by
        passing, because both this and the adjacent-run counting it replaced
        accept a small Graph — the two only diverge in the number. Here:

        The interleaved run sits under a two-trip loop so that Step C leaves it
        in the region — it hoists a *top-level* allocation to the call site, and
        one it hoists is a node the emitted region no longer has. The loop body
        is still one statement list, which is what the batching rule is about.
        Two trips rather than one because `Simplify` collapses a single-trip
        loop into its body.

            per iteration = 20 interleaved launches + ceil(20 / 16) = 2 batched
            total         = 2 * 22 + 1024 in the loop + 1 for the scope -> 1069
            per-create    = 2 * 40 + 1025                               -> 1105

        so the reported total is what distinguishes them.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec
                )
                return pl.store(t, [0, 0], out)

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                d: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for _ in pl.range(2):
                    _s0: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s1: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s2: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s3: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s4: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s5: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s6: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s7: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s8: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s9: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s10: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s11: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s12: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s13: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s14: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s15: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s16: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s17: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s18: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                    _s19: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                    d = self.kernel(a, d)
                for _ in pl.range(1024):
                    d = self.kernel(a, d)
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                d: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c, d)
                return c

        with pytest.raises(ValueError, match="launches 1069 tasks"):
            _legalize_outlined(Before)

    def test_batchable_gm_pipe_allocations_are_not_charged_individually(self):
        """A GM pipe buffer that keeps its place in the batch counts as batched.

        The emitter pulls one out of the shared `alloc_tensors` only when its
        `core_num` reads a value defined earlier in the same statement list.
        Charging every `gm_pipe_buffer_*` create its own node instead would make
        these 40 into 40 nodes rather than 3, and reject a Graph the runtime
        accepts.

        The names carry the `gm_pipe_buffer_` prefix on purpose: that is what
        `InjectGmPipeBuffer` produces and what the shared predicate matches, so
        they cannot be renamed to satisfy the unused-local lint.

        Pinned through the over-limit total, since both countings accept a small
        Graph:

        The run sits under a two-trip loop so that Step C leaves it in the
        region — it hoists a *top-level* allocation to the call site, and one it
        hoists is a node the emitted region no longer has. The loop body is
        still one statement list, and two trips rather than one because
        `Simplify` collapses a single-trip loop into its body.

            batched    = 2 * ceil(40 / 16) = 6 + 1024 + 1 for the scope -> 1031
            per-create = 2 * 40 = 80 + 1025                             -> 1105
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec
                )
                return pl.store(t, [0, 0], out)

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                d: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for _ in pl.range(2):
                    gm_pipe_buffer_0: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_1: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_2: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_3: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_4: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_5: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_6: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_7: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_8: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_9: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_10: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_11: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_12: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_13: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_14: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_15: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_16: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_17: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_18: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_19: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_20: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_21: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_22: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_23: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_24: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_25: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_26: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_27: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_28: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_29: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_30: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_31: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_32: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_33: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_34: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_35: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_36: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_37: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_38: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                    gm_pipe_buffer_39: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)  # noqa: F841
                for _ in pl.range(1024):
                    d = self.kernel(a, d)
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                d: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c, d)
                return c

        with pytest.raises(ValueError, match="launches 1031 tasks"):
            _legalize_outlined(Before)

    def test_constant_shaped_allocation_is_allowed(self):
        """A literal shape is replay-invariant, so the recording reproduces it."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                _s: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = self.layer(a, c)
                return c

        assert _graph_func(_legalize_outlined(Before), "layer").func_type == ir.FunctionType.Graph

    def test_nested_graph_is_rejected(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def inner(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function(type=pl.FunctionType.Graph)
            def outer(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                self.inner(a, c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                self.outer(a, c)
                return c

        with pytest.raises(ValueError, match="Nested graphs are not supported"):
            _legalize_outlined(Before)


# ---------------------------------------------------------------------------
# Replay-invariant values — frozen, but frozen at the right value
# ---------------------------------------------------------------------------


class TestReplayInvariantScalars:
    """Values Step A cannot hoist, yet replay reproduces exactly.

    Hoistability answers "can the call site recompute this?"; the runtime only
    needs "is this the same on every call?", which is strictly weaker. A value in
    the gap has no argument slot and *is* frozen into the recording — and that is
    harmless, because the frozen copy is the value every later replay would have
    computed anyway. Rejecting these excluded every tiled kernel, since a slab
    offset `i * TILE` cannot be hoisted: it does not exist at the call site.
    """

    def test_a_constant_trip_loop_offset_is_accepted(self):
        """Recording bakes each iteration's literal into that iteration's node.

        Constant bounds mean every later call walks the identical sequence, so
        the baked literals stay correct. The offset must also stay *in* the
        region — there is no call-site name to hoist it to.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def tiled(
                self,
                w: pl.Tensor[[512, 256], pl.FP32],
                acc: pl.InOut[pl.Tensor[[128, 256], pl.FP32]],
                layer_idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 256], pl.FP32]:
                base = layer_idx * 128
                for i in pl.range(2):
                    off = i * 128
                    with pl.at(level=pl.Level.CORE_GROUP):
                        band: pl.Tile[[128, 128], pl.FP32] = pl.load(w, [base, off], [128, 128])
                        cur: pl.Tile[[128, 128], pl.FP32] = pl.load(acc, [0, off], [128, 128])
                        pl.store(pl.add(cur, band), [0, off], acc)
                return acc

            @pl.function
            def main(
                self,
                w: pl.Tensor[[512, 256], pl.FP32],
                acc: pl.InOut[pl.Tensor[[128, 256], pl.FP32]],
            ) -> pl.Tensor[[128, 256], pl.FP32]:
                self.tiled(w, acc, 0)
                return acc

        scalars = _scalar_param_names(_graph_func(_legalize_outlined(Before), "tiled"))
        # `base` derives from a parameter, so it hoists as before...
        assert "base" in scalars, scalars
        # ...while the induction-derived offset stays put: hoisting it would ask
        # the caller for a value that only exists inside the loop.
        assert "off" not in scalars, scalars

    def test_a_boundary_tensor_dim_read_is_accepted(self):
        """`graph_boundary_matches` pins every boundary tensor's shape.

        It compares `ndims`, `shapes` and `strides` against the recorded
        `GraphBoundarySignature` and declines the cached graph on any mismatch,
        so within one recording a boundary extent cannot change.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def dim_read(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                rows = pl.tensor.dim(a, 0)
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [rows - 128, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                self.dim_read(a, c)
                return c

        scalars = _scalar_param_names(_graph_func(_legalize_outlined(Before), "dim_read"))
        assert "rows" not in scalars, scalars

    def test_a_task_id_stored_through_an_array_is_accepted(self):
        """A TaskId is topology, never a boundary scalar.

        The same id written straight into `deps=[...]` always compiled; only the
        `pl.array` store looked like a task consuming a boundary scalar, because
        the check runs over every call's arguments and an array store is a call.
        The DSL has no other way to accumulate ids produced across a loop.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def fenced(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.manual_scope():
                    seed = pl.system.task_dummy(deps=[])
                    seeds = pl.array.create(1, pl.TASK_ID)
                    seeds[0] = seed
                    with pl.at(level=pl.Level.CORE_GROUP, deps=[seeds[0]]):
                        t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                        pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                self.fenced(a, c)
                return c

        # Compiling at all is the assertion: this used to raise on the array store.
        assert _graph_func(_legalize_outlined(Before), "fenced") is not None

    def test_a_scalar_read_from_a_tensor_is_still_rejected(self):
        """The guard the widening must not dissolve.

        A value read out of a tensor genuinely differs between calls — nothing
        pins a buffer's *contents* — so it has neither an argument slot nor a
        reproducible value, and freezing it is the silent wrong answer this whole
        step exists to prevent.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                idx: pl.Tensor[[4], pl.INT32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                row: pl.Scalar[pl.INDEX] = pl.tensor.read(idx, [0])
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [row, 0], [128, 128])
                    pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                idx: pl.Tensor[[4], pl.INT32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                self.layer(a, idx, c)
                return c

        with pytest.raises(ValueError, match="can differ between calls"):
            _legalize_outlined(Before)

    def test_a_view_offset_by_a_loop_variable_stays_in_the_region(self):
        """Step B's third outcome: neither hoisted nor rejected.

        Replay restores a `BOUNDARY_VIEW` as `boundary.start_offset +
        packed_offset`, with `packed_offset` recorded on the first call. An
        invariant offset makes that frozen delta the right one every time, and
        the call site has no name for a loop variable to hoist it with.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def tiled(
                self,
                w: pl.Tensor[[512, 256], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for i in pl.range(2):
                    off = i * 128
                    band: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(w, [128, 128], [off, 0])
                    with pl.at(level=pl.Level.CORE_GROUP):
                        t: pl.Tile[[128, 128], pl.FP32] = pl.load(band, [0, 0], [128, 128])
                        pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                w: pl.Tensor[[512, 256], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                self.tiled(w, c)
                return c

        tensors = _tensor_param_names(_graph_func(_legalize_outlined(Before), "tiled"))
        assert "band" not in tensors, tensors

    def test_a_view_offset_mixing_a_boundary_scalar_and_a_loop_variable_is_rejected(self):
        """The case that would genuinely freeze.

        `layer_idx + i * 128` can be neither hoisted (`i` does not exist at the
        call site) nor frozen (`layer_idx` is patched per call), so the recorded
        delta would be call one's while the address is call two's.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def tiled(
                self,
                w: pl.Tensor[[512, 256], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
                layer_idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for i in pl.range(2):
                    off = layer_idx + i * 128
                    band: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(w, [128, 128], [off, 0])
                    with pl.at(level=pl.Level.CORE_GROUP):
                        t: pl.Tile[[128, 128], pl.FP32] = pl.load(band, [0, 0], [128, 128])
                        pl.store(t, [0, 0], c)
                return c

            @pl.function
            def main(
                self,
                w: pl.Tensor[[512, 256], pl.FP32],
                c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                self.tiled(w, c, 0)
                return c

        with pytest.raises(ValueError, match="nor the same on every replay"):
            _legalize_outlined(Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
