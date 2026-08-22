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
    with passes.PassContext([]):
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
    """
    with passes.PassContext([]):
        ssa = passes.convert_to_ssa()(program)
        return passes.legalize_graph_boundary()(passes.outline_incore_scopes()(ssa))


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

        After = _legalize(Before)
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

        After = _legalize(Before)
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

        After = _legalize(Before)
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
        with passes.PassContext([]):
            After = passes.legalize_graph_boundary()(ssa)
        ir.assert_structural_equal(ssa, After)


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

        assert _graph_func(_legalize(Before), "layer").func_type == ir.FunctionType.Graph

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
            _legalize(Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
