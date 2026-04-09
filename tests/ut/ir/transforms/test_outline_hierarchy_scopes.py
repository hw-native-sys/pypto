# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for OutlineHierarchyScopes pass."""

import pypto.language as pl
import pytest
from pypto import ir, passes


class TestOutlineHierarchyScopes:
    """Test OutlineHierarchyScopes pass."""

    def test_outline_simple_hierarchy_scope(self):
        """Test outlining a simple Hierarchy scope with level and role."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def main_host_worker_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_host_worker_0(x)
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_hierarchy_level_only(self):
        """Test outlining a Hierarchy scope with only level (no role)."""

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

    def test_outline_multiple_hierarchy_scopes(self):
        """Test outlining multiple Hierarchy scopes in one function."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator):
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def main_host_worker_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator)
            def main_global_orch_1(self, y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_host_worker_0(x)
                z: pl.Tensor[[64], pl.FP32] = self.main_global_orch_1(y)
                return z

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_nested_hierarchy_scopes(self):
        """Test outlining nested Hierarchy scopes."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                        z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def main_global_orch_0_host_worker_0(
                self, y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

            @pl.function(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator)
            def main_global_orch_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                z: pl.Tensor[[64], pl.FP32] = self.main_global_orch_0_host_worker_0(y)
                return z

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = self.main_global_orch_0(x)
                return z

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_hierarchy_with_incore_preserved(self):
        """Test that InCore scope inside Hierarchy scope is preserved (not outlined by this pass)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def main_host_worker_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_host_worker_0(x)
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_hierarchy_multiple_inputs(self):
        """Test outlining scope that uses multiple outer variables."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, y)
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return result

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def main_host_worker_0(
                self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return result

            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, y)
                result: pl.Tensor[[64], pl.FP32] = self.main_host_worker_0(a, b)
                return result

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_hierarchy_multiple_outputs(self):
        """Test outlining scope that produces multiple values."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                result: pl.Tensor[[64], pl.FP32] = pl.add(y, z)
                return result

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)

        # Verify outlined function has level/role and 2 return types
        hierarchy_func = After.get_function("main_host_worker_0")
        assert hierarchy_func is not None
        assert hierarchy_func.level == ir.Level.HOST
        assert hierarchy_func.role == ir.Role.Worker
        assert len(hierarchy_func.return_types) == 2

    def test_outline_hierarchy_no_outputs(self):
        """Test outlining a Hierarchy scope with no variables used after."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    _y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return x

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)

        hierarchy_func = After.get_function("main_host_worker_0")
        assert hierarchy_func is not None
        assert hierarchy_func.level == ir.Level.HOST
        assert hierarchy_func.role == ir.Role.Worker
        assert len(hierarchy_func.return_types) == 0

    def test_outline_hierarchy_in_control_flow(self):
        """Test outlining Hierarchy scope inside conditional statement."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], cond: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
                if cond:
                    with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)  # type: ignore[no-redef]
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)  # type: ignore[no-redef,unreachable]
                return y

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def main_host_worker_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], cond: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
                if cond:
                    y: pl.Tensor[[64], pl.FP32] = self.main_host_worker_0(x)  # type: ignore[no-redef]
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)  # type: ignore[no-redef,unreachable]
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_hierarchy_does_not_affect_incore_scopes(self):
        """Test that OutlineHierarchyScopes does not outline InCore scopes."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)
        # InCore scopes should remain untouched by the hierarchy pass
        ir.assert_structural_equal(After, Before)

    def test_hierarchy_does_not_affect_cluster_scopes(self):
        """Test that OutlineHierarchyScopes does not outline Cluster scopes."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)
        # Cluster scopes should remain untouched by the hierarchy pass
        ir.assert_structural_equal(After, Before)

    def test_no_hierarchy_scopes_passthrough(self):
        """Test that functions without Hierarchy scopes are passed through unchanged."""

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
        """Test that parent function keeps its original type after outlining."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)

        main_func = After.get_function("main")
        assert main_func is not None
        # Parent remains Opaque (not promoted to Orchestration)
        assert main_func.func_type == ir.FunctionType.Opaque

    def test_outline_hierarchy_different_levels(self):
        """Test various level/role combinations."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CHIP, role=pl.Role.Orchestrator):
                    a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.CLUSTER_0, role=pl.Role.Worker):
                    b: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                return b

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)

        # Verify levels are correctly propagated
        func_0 = After.get_function("main_chip_orch_0")
        assert func_0 is not None
        assert func_0.level == ir.Level.CHIP
        assert func_0.role == ir.Role.Orchestrator

        func_1 = After.get_function("main_cluster0_worker_1")
        assert func_1 is not None
        assert func_1.level == ir.Level.CLUSTER_0
        assert func_1.role == ir.Role.Worker

    def test_outline_skips_non_opaque_functions(self):
        """Test that non-Opaque functions (InCore, Orchestration) are skipped."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)

        # InCore function preserved unchanged
        compute = After.get_function("compute")
        assert compute is not None
        assert compute.func_type == ir.FunctionType.InCore

        # Hierarchy scope in main was outlined
        hierarchy_func = After.get_function("main_host_worker_0")
        assert hierarchy_func is not None
        assert hierarchy_func.level == ir.Level.HOST

    def test_outline_hierarchy_with_intermediate_computation(self):
        """Test outlining with computation before, inside, and after the scope."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    c: pl.Tensor[[64], pl.FP32] = pl.add(b, b)
                    d: pl.Tensor[[64], pl.FP32] = pl.mul(c, c)
                e: pl.Tensor[[64], pl.FP32] = pl.add(d, d)
                return e

        @pl.program
        class Expected:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def main_host_worker_0(self, b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                c: pl.Tensor[[64], pl.FP32] = pl.add(b, b)
                d: pl.Tensor[[64], pl.FP32] = pl.mul(c, c)
                return d

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                d: pl.Tensor[[64], pl.FP32] = self.main_host_worker_0(b)
                e: pl.Tensor[[64], pl.FP32] = pl.add(d, d)
                return e

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_hierarchy_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_multiple_functions_with_hierarchy(self):
        """Test outlining in multiple functions (independent counter per function)."""

        @pl.program
        class Before:
            @pl.function
            def func1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def func2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator):
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)

        # Each function gets its own counter
        func1_outlined = After.get_function("func1_host_worker_0")
        assert func1_outlined is not None
        assert func1_outlined.level == ir.Level.HOST
        assert func1_outlined.role == ir.Role.Worker

        func2_outlined = After.get_function("func2_global_orch_0")
        assert func2_outlined is not None
        assert func2_outlined.level == ir.Level.GLOBAL
        assert func2_outlined.role == ir.Role.Orchestrator

    def test_outline_hierarchy_round_trip(self):
        """Test that outlined hierarchy program survives print-parse round-trip."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_hierarchy_scopes()(Before)

        # Print and re-parse
        printed = After.as_python()
        Reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(After, Reparsed)

    def test_outline_then_incore(self):
        """Test hierarchy outlined first, then InCore outlined from inside hierarchy function."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)

        # Step 1: Outline hierarchy scopes
        After1 = passes.outline_hierarchy_scopes()(Before)

        # The outlined hierarchy function should contain the InCore scope
        hierarchy_func = After1.get_function("main_host_worker_0")
        assert hierarchy_func is not None
        assert hierarchy_func.level == ir.Level.HOST
        printed1 = After1.as_python()
        assert "pl.at(level=pl.Level.CORE_GROUP)" in printed1

        # Step 2: Outline incore scopes (processes Opaque functions including hierarchy-outlined ones)
        After2 = passes.outline_incore_scopes()(After1)

        # The InCore scope should now be outlined from the hierarchy function
        incore_func = After2.get_function("main_host_worker_0_incore_0")
        assert incore_func is not None
        assert incore_func.func_type == ir.FunctionType.InCore

    def test_outline_hierarchy_with_alias_level(self):
        """Test that level aliases (POD = CLUSTER_0) resolve to canonical name."""

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
        # POD is an alias for CLUSTER_0 — both have underlying value 6.
        # The binding returns the canonical enum member (CLUSTER_0).
        assert func.level == ir.Level.CLUSTER_0


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
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        ctx = passes.PassContext([], passes.VerificationLevel.NONE)
        with ctx:
            program = passes.convert_to_ssa()(Input)
            program = passes.outline_hierarchy_scopes()(program)

        # Should not throw — no Hierarchy scopes remain
        passes.verify_properties(self._hierarchy_outlined_props(), program, "test")

    def test_remaining_hierarchy_scope_fails_verification(self):
        """Leftover Hierarchy ScopeStmt causes verification failure."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        # Don't outline — just convert to SSA, leaving Hierarchy scope intact
        ctx = passes.PassContext([], passes.VerificationLevel.NONE)
        with ctx:
            program = passes.convert_to_ssa()(Input)

        # verify_properties should throw because Hierarchy scope remains
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

        # No hierarchy scopes at all — verification should pass
        passes.verify_properties(self._hierarchy_outlined_props(), Input, "test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
