# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for OutlineHierarchyScopes pass.

The inline ``with pl.at(level>=HOST, role=pl.Role.SubWorker)`` form was removed
(SubWorkers are now declared via ``@pl.function(level=..., role=...)``), so the
cases here exercise Hierarchy scopes through the still-supported inline forms:
``with pl.at(level=..., role=pl.Role.Orchestrator)`` and the level-only
``with pl.at(level=...)`` (for any non-CORE_GROUP level — CORE_GROUP is special
cased to InCore by the parser).

Every transform case follows the canonical Before/After/Expected style: the
Expected IR is the post-outline program written by hand from the pass's
documented semantics (``docs/en/dev/passes/09-outline_hierarchy_scopes.md`` and
``src/ir/transforms/outline_hierarchy_scopes_pass.cpp``), then run through
``convert_to_ssa`` so SSA naming/phi insertion matches the pass output.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


class TestOutlineHierarchyScopes:
    """Test OutlineHierarchyScopes pass."""

    def test_outline_single_orchestrator_scope(self):
        """A single Hierarchy(Orchestrator) scope is lifted into an Opaque
        function carrying level/role; the scope is replaced by a Call. Parent
        stays Opaque (doc Example, lines 113-139; source: outlined_func_type_ =
        FunctionType::Opaque, MutableCopy preserves parent type)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def main_host_orch_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_host_orch_0(x)
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_hierarchy_level_only(self):
        """Outlining a Hierarchy scope with only level (no role) — the role
        suffix is omitted from the name (doc Naming table; source
        GenerateHierarchySuffix appends role only when present)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.GLOBAL):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.GLOBAL)
            def main_global_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_global_0(x)
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_multiple_sequential_scopes_independent_counter(self):
        """Two sibling Hierarchy scopes are outlined into two functions with a
        per-function counter (``_0`` then ``_1``). The first is level-only
        (no role suffix), the second is Orchestrator (source: scope_counter_
        increments per outlined scope; naming via GenerateHierarchySuffix)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CHIP):
                    a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator):
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                return z

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.CHIP)
            def main_chip_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return a

            @pl.function(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator)
            def main_global_orch_1(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                return z

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = self.main_chip_0(x)
                z: pl.Tensor[[64], pl.FP32] = self.main_global_orch_1(a)
                return z

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_nested_orchestrator_scopes_chained_name(self):
        """Nested Hierarchy scopes are outlined recursively: the inner scope is
        extracted first and replaced with a Call inside the outer outlined
        function, producing chained names like
        ``main_global_orch_0_host_orch_0`` (doc Nested Hierarchy Example,
        lines 153-189)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
                        z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def main_global_orch_0_host_orch_0(self, y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

            @pl.function(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator)
            def main_global_orch_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                z: pl.Tensor[[64], pl.FP32] = self.main_global_orch_0_host_orch_0(y)
                return z

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = self.main_global_orch_0(x)
                return z

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_multiple_inputs(self):
        """A scope that reads two outer variables produces a 2-param outlined
        function; the captured Vars are the call args in first-use order
        (source: input_vars built from body_collector.var_uses_ordered)."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, y)
                with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
                    c: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return c

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def main_host_orch_0(
                self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                c: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return c

            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, y)
                c: pl.Tensor[[64], pl.FP32] = self.main_host_orch_0(a, b)
                return c

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_multi_output_tuple_get_item(self):
        """A scope that defines two values used after it returns a 2-tuple; the
        call site binds a temp ``ret`` var, then projects each output with a
        ``TupleGetItem`` (source OutlineScope output_vars.size() > 1 path,
        lines 984-996)."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator):
                    a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    b: pl.Tensor[[64], pl.FP32] = pl.mul(w, w)
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return y

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator)
            def main_global_orch_0(
                self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(w, w)
                return a, b

            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                a, b = self.main_global_orch_0(x, w)
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_with_intermediate_computation(self):
        """Computation before and after the scope is left in the parent; the
        scope itself is replaced by a single Call (single-output AssignStmt
        path, source line 981-983)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
                    c: pl.Tensor[[64], pl.FP32] = pl.add(b, b)
                e: pl.Tensor[[64], pl.FP32] = pl.add(c, c)
                return e

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def main_host_orch_0(self, b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                c: pl.Tensor[[64], pl.FP32] = pl.add(b, b)
                return c

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                c: pl.Tensor[[64], pl.FP32] = self.main_host_orch_0(b)
                e: pl.Tensor[[64], pl.FP32] = pl.add(c, c)
                return e

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_in_control_flow(self):
        """A Hierarchy scope nested inside an ``if`` branch is outlined in place
        (source: SeqStmts visitor recurses into control-flow bodies via
        required_outputs_ propagation). SSA phi insertion for the joined ``y``
        is produced identically on both sides by convert_to_ssa."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], cond: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
                if cond:
                    with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
                        y0: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    y: pl.Tensor[[64], pl.FP32] = pl.add(y0, x)  # type: ignore[no-redef]
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)  # type: ignore[no-redef,unreachable]
                return y

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def main_host_orch_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y0: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], cond: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
                if cond:
                    y0: pl.Tensor[[64], pl.FP32] = self.main_host_orch_0(x)
                    y: pl.Tensor[[64], pl.FP32] = pl.add(y0, x)  # type: ignore[no-redef]
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)  # type: ignore[no-redef,unreachable]
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_submit_emission_with_deps(self):
        """``with pl.at(...) as tid:`` makes the scope carry a ``task_id_var``
        attr, so the call site becomes an ``ir.Submit`` (not a plain Call) whose
        return type is the augmented ``Tuple{<scope output>, Scalar[TASK_ID]}``;
        a trailing ``TupleGetItem`` binds the producer TaskId. ``deps=[t1]`` is
        folded into ``Submit.deps_``. The parser's transient
        ``system.task_invalid()`` placeholder before each scope is dropped
        (source lines 317-345, 836, 871-909, 961-978)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator, name_hint="s1") as t1:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator, name_hint="s2", deps=[t1]) as _t2:
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator)
            def s1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def s2(self, y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

            @pl.function(type=pl.FunctionType.Opaque)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y, t1 = pl.submit(self.s1, x)
                z, _t2 = pl.submit(self.s2, y, deps=[t1])
                return z

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_submit_single_output(self):
        """Even with a single scope output, the ``as tid:`` path always uses the
        temp+unpack form: the TaskId is an extra trailing tuple element needing
        its own ``TupleGetItem`` (source lines 961-978 — "always goes through
        the temp+unpack path even for <=1 output")."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator, name_hint="s1") as t1:  # noqa: F841
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator)
            def s1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Opaque)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y, t1 = pl.submit(self.s1, x)  # noqa: F841
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_hierarchy_does_not_affect_incore_scopes(self):
        """OutlineHierarchyScopes only targets ScopeKind::Hierarchy; InCore
        scopes are left untouched (source: target_scope_kind_ == Hierarchy)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Before)

    def test_hierarchy_does_not_affect_cluster_scopes(self):
        """Cluster scopes are likewise untouched by the hierarchy pass."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Before)

    def test_no_hierarchy_scopes_passthrough(self):
        """Functions without any Hierarchy scope are passed through unchanged."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Before)

    def test_outline_preserves_parent_function_type(self):
        """Parent stays Opaque after outlining — hierarchy is orthogonal to
        FunctionType (doc lines 31-34; source: MutableCopy preserves
        func_type_, no Orchestration promotion)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)

        main_func = After.get_function("main")
        assert main_func is not None
        assert main_func.func_type == ir.FunctionType.Opaque

    def test_outline_skips_non_opaque_functions(self):
        """Non-Opaque functions are emitted unchanged; only the Opaque ``main``
        is processed (source: ``if func_type_ != Opaque: continue``)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)

        # InCore function preserved unchanged.
        compute = After.get_function("compute")
        assert compute is not None
        assert compute.func_type == ir.FunctionType.InCore

        # Hierarchy scope in main was outlined.
        hierarchy_func = After.get_function("main_host_orch_0")
        assert hierarchy_func is not None
        assert hierarchy_func.level == ir.Level.HOST
        assert hierarchy_func.role == ir.Role.Orchestrator

    def test_outline_independent_counter_across_functions(self):
        """Each Opaque function starts its own scope counter at 0 (source:
        scope_counter_ is constructed per-function in the pass loop)."""

        @pl.program
        class Before:
            @pl.function
            def func1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def func2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator):
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)

        func1_outlined = After.get_function("func1_host_orch_0")
        assert func1_outlined is not None
        assert func1_outlined.level == ir.Level.HOST
        assert func1_outlined.role == ir.Role.Orchestrator

        func2_outlined = After.get_function("func2_global_orch_0")
        assert func2_outlined is not None
        assert func2_outlined.level == ir.Level.GLOBAL
        assert func2_outlined.role == ir.Role.Orchestrator

    def test_outline_hierarchy_with_alias_level(self):
        """Level aliases (``POD = CLUSTER_0``) resolve to the canonical name in
        both the function suffix (``cluster0``) and the stored ``level``
        (doc Level aliases, lines 107-109; the binding returns the canonical
        enum member)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(pl.Level.POD):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)

        func = After.get_function("main_cluster0_0")
        assert func is not None
        assert func.level == ir.Level.CLUSTER_0

    def test_outline_hierarchy_round_trip(self):
        """Outlined hierarchy program survives a print → parse round-trip."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)

        printed = After.as_python()
        Reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(After, Reparsed)


class TestHierarchyOutlinedVerifier:
    """Tests for the HierarchyOutlined property verifier."""

    @staticmethod
    def _hierarchy_outlined_props():
        ps = passes.IRPropertySet()
        ps.insert(passes.IRProperty.HierarchyOutlined)
        return ps

    def test_clean_program_passes_verification(self):
        """Outlined program with no Hierarchy scopes passes verification."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        ctx = passes.PassContext([], passes.VerificationLevel.NONE)
        with ctx:
            program = passes.convert_to_ssa()(Input)
            program = passes.outline_hierarchy_scopes()(program)

        # Should not throw — no Hierarchy scopes remain.
        passes.verify_properties(self._hierarchy_outlined_props(), program, "test")

    def test_remaining_hierarchy_scope_fails_verification(self):
        """Leftover Hierarchy ScopeStmt causes verification failure."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        # Don't outline — just convert to SSA, leaving Hierarchy scope intact.
        ctx = passes.PassContext([], passes.VerificationLevel.NONE)
        with ctx:
            program = passes.convert_to_ssa()(Input)

        # verify_properties should throw because Hierarchy scope remains.
        with pytest.raises(Exception, match="Hierarchy ScopeStmt"):
            passes.verify_properties(self._hierarchy_outlined_props(), program, "test")

    def test_program_without_hierarchy_passes_verification(self):
        """Program that never had Hierarchy scopes passes verification."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        # No hierarchy scopes at all — verification should pass.
        passes.verify_properties(self._hierarchy_outlined_props(), Input, "test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
