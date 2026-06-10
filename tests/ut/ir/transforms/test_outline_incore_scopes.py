# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for OutlineIncoreScopes pass."""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes


class TestOutlineIncoreScopes:
    """Test OutlineIncoreScopes pass."""

    def test_outline_simple_incore_scope(self):
        """Test outlining a simple InCore scope."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        # Convert to SSA first (required by outline pass)
        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)

        # Apply outline pass
        After = passes.outline_incore_scopes()(Before)

        # Should be structurally equal
        ir.assert_structural_equal(After, Expected)

    def test_outline_multiple_incore_scopes(self):
        """Test outlining multiple InCore scopes in one function."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.CORE_GROUP):
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_1(self, y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                z: pl.Tensor[[64], pl.FP32] = self.main_incore_1(y)
                return z

        # Convert to SSA first
        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)

        # Apply outline pass
        After = passes.outline_incore_scopes()(Before)

        # Should be structurally equal
        ir.assert_structural_equal(After, Expected)

    def test_outline_preserves_non_incore_functions(self):
        """Test that non-InCore functions are preserved unchanged."""

        @pl.program
        class Before:
            @pl.function
            def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return result

            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        # Convert to SSA first
        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)

        # Apply outline pass
        After = passes.outline_incore_scopes()(Before)

        # Should be structurally equal
        ir.assert_structural_equal(After, Expected)

    def test_outline_scope_with_multiple_inputs(self):
        """Test outlining scope that uses multiple outer variables."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, y)
                with pl.at(level=pl.Level.CORE_GROUP):
                    result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, y)
                result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(a, b)
                return result

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_scope_with_multiple_outputs(self):
        """Test outlining scope that produces multiple values.

        The Before/After pattern can't express TupleGetItem in the DSL,
        so we verify properties directly.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                result: pl.Tensor[[64], pl.FP32] = pl.add(y, z)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, x: pl.Tensor[[64], pl.FP32]
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                z: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return (y, z)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                ret = self.main_incore_0(x)
                y = ret[0]
                z = ret[1]
                result: pl.Tensor[[64], pl.FP32] = pl.add(y, z)
                return result

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_incore_scopes()(Before)

        ir.assert_structural_equal(After, Expected)

    def test_nested_incore_scopes_rejected_by_verifier(self):
        """Nested InCore scopes are rejected by the NoNestedInCore structural verifier."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    with pl.at(level=pl.Level.CORE_GROUP):
                        z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        # Verify directly (no pass pipeline) — nested InCore is a structural invariant violation
        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.NoNestedInCore)
        diagnostics = passes.PropertyVerifierRegistry.verify(props, Before)
        errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
        assert len(errors) >= 1
        assert "Nested InCore scope" in errors[0].message

    def test_outline_scope_with_single_input_single_output(self):
        """Test outlining scope with simple single input/output."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                result: pl.Tensor[[64], pl.FP32] = pl.add(y, y)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(a)
                result: pl.Tensor[[64], pl.FP32] = pl.add(y, y)
                return result

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_multiple_functions_with_scopes(self):
        """Test outlining scopes in multiple functions (independent numbering)."""

        @pl.program
        class Before:
            @pl.function
            def func1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def func2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def func1_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def func1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.func1_incore_0(x)
                return y

            @pl.function(type=pl.FunctionType.InCore)
            def func2_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def func2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.func2_incore_0(x)
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_scope_in_control_flow(self):
        """Test outlining scope inside conditional statement."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], cond: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
                if cond:
                    with pl.at(level=pl.Level.CORE_GROUP):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)  # type: ignore[no-redef]
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)  # type: ignore[no-redef,unreachable]
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32], cond: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
                if cond:
                    y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)  # type: ignore[no-redef]
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)  # type: ignore[no-redef,unreachable]
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_incore_with_if_yield(self):
        """Test outline_incore_scopes with IfStmt containing unannotated yields (issue #233)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], cond: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    if cond:
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                        z = pl.yield_(y)  # Unannotated - should infer type
                    else:
                        y2: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                        z = pl.yield_(y2)
                return z

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, cond: pl.Scalar[pl.BOOL], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                if cond:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    z = pl.yield_(y)  # type: ignore[no-redef]
                else:
                    y2: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                    z = pl.yield_(y2)  # type: ignore[no-redef]
                return z

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32], cond: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(cond, x)
                return z

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_scope_with_intermediate_computation(self):
        """Test outlining scope with computation before, inside, and after."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                with pl.at(level=pl.Level.CORE_GROUP):
                    c: pl.Tensor[[64], pl.FP32] = pl.add(b, b)
                    d: pl.Tensor[[64], pl.FP32] = pl.mul(c, c)
                e: pl.Tensor[[64], pl.FP32] = pl.add(d, d)
                return e

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                c: pl.Tensor[[64], pl.FP32] = pl.add(b, b)
                d: pl.Tensor[[64], pl.FP32] = pl.mul(c, c)
                return d

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                d: pl.Tensor[[64], pl.FP32] = self.main_incore_0(b)
                e: pl.Tensor[[64], pl.FP32] = pl.add(d, d)
                return e

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_scope_with_store_only_outputs(self):
        """Test outlining scope where the only outputs are store targets.

        When an InCore scope only writes to external tensors via tile.store
        (no new variable definitions used after the scope), the store targets
        must be recognised as outputs and returned.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
                buf: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP):
                    tile = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                    pl.store(tile, [0, 0], buf)
                result: pl.Tensor[[16, 128], pl.FP32] = pl.add(buf, x)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, buf: pl.InOut[pl.Tensor[[16, 128], pl.FP32]]
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                tile = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                buf_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(tile, [0, 0], buf)
                return buf_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
                buf: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                buf2: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(buf)
                result: pl.Tensor[[16, 128], pl.FP32] = pl.add(buf2, x)
                return result

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_scope_with_multiple_store_targets(self):
        """Test outlining scope with multiple store targets as outputs.

        Multiple external tensors modified via tile.store should all appear
        as return values of the outlined function.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
                buf_a: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                buf_b: pl.Tensor[[16, 1], pl.FP32] = pl.create_tensor([16, 1], dtype=pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP):
                    tile_a = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                    tile_b = pl.tile.full([16, 1], dtype=pl.FP32, value=0.0)
                    pl.store(tile_a, [0, 0], buf_a)
                    pl.store(tile_b, [0, 0], buf_b)
                result: pl.Tensor[[16, 128], pl.FP32] = pl.add(buf_a, x)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                buf_a: pl.InOut[pl.Tensor[[16, 128], pl.FP32]],
                buf_b: pl.InOut[pl.Tensor[[16, 1], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 128], pl.FP32]]:
                tile_a = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                tile_b = pl.tile.full([16, 1], dtype=pl.FP32, value=0.0)
                buf_a_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(tile_a, [0, 0], buf_a)
                buf_b_store: pl.Tensor[[16, 1], pl.FP32] = pl.store(tile_b, [0, 0], buf_b)
                return (buf_b_store, buf_a_store)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
                buf_a: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                buf_b: pl.Tensor[[16, 1], pl.FP32] = pl.create_tensor([16, 1], dtype=pl.FP32)
                ret = self.main_incore_0(buf_a, buf_b)
                buf_b2 = ret[0]
                buf_a2 = ret[1]
                result: pl.Tensor[[16, 128], pl.FP32] = pl.add(buf_a2, x)
                return result

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_repeated_store_target_keeps_body_in_sync(self):
        """N same-kind scopes storing to one tensor must thread the store target.

        Regression test for issue #1462. Each scope writing the shared store
        target ``out`` is outlined into a function that takes it as a parameter
        and returns a fresh renamed value (``out -> out_v1 -> out_v2 -> ...``).
        ScopeOutliner must keep every reference consistent:

        - the outlined body's store use-site must resolve to that function's
          own parameter (else a Var is used outside its defining function — a
          whole-program SSAForm violation);
        - each scope's synthesised call must pass the value current as of that
          scope, and the ReturnStmt must read the final value (else a renamed
          store target is read pre-call — an InOutUseDiscipline violation).

        Four scopes are used so the test exercises both the body use-site
        (visible with two scopes) and the call-argument threading across the
        full chain (call-argument staleness only becomes visible by the fourth
        scope). ``outline_incore_scopes`` post-verifies the structural
        properties, so a stale call argument throws here; the explicit check
        covers SSAForm.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    tile0 = pl.load(a, [0, 0], [32, 32])
                    pl.store(tile0, [0, 0], out)
                with pl.at(level=pl.Level.CORE_GROUP):
                    tile1 = pl.load(a, [0, 0], [32, 32])
                    pl.store(tile1, [0, 0], out)
                with pl.at(level=pl.Level.CORE_GROUP):
                    tile2 = pl.load(a, [0, 0], [32, 32])
                    pl.store(tile2, [0, 0], out)
                with pl.at(level=pl.Level.CORE_GROUP):
                    tile3 = pl.load(a, [0, 0], [32, 32])
                    pl.store(tile3, [0, 0], out)
                return out

        After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

        # Four InCore scopes outlined into four functions plus the orchestrator.
        assert len(After.functions) == 5

        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.SSAForm)
        errors = [
            d
            for d in passes.PropertyVerifierRegistry.verify(props, After)
            if d.severity == passes.DiagnosticSeverity.Error
        ]
        assert not errors, f"SSAForm violated: {[d.message for d in errors]}"

    def test_outline_scope_with_loop_carried_init_values(self):
        """Test outlining scope where inner loop references outer loop-carried variable via init_values.

        Regression test for issue #369: OutlineIncoreScopes failed to include
        outer loop-carried variables as incore function parameters when they
        appeared only inside IterArg.initValue_ expressions.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(3, init_values=(x,)):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for j, (inner,) in pl.range(2, init_values=(acc,)):
                            updated: pl.Tensor[[64], pl.FP32] = pl.add(inner, y)
                            inner_rv = pl.yield_(updated)
                    acc_rv = pl.yield_(inner_rv)
                return acc_rv

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, acc: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                for j, (inner,) in pl.range(2, init_values=(acc,)):
                    updated: pl.Tensor[[64], pl.FP32] = pl.add(inner, y)
                    inner_rv = pl.yield_(updated)
                return inner_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(3, init_values=(x,)):
                    inner_rv: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, y)
                    acc_rv = pl.yield_(inner_rv)
                return acc_rv

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_scope_does_not_capture_outer_init_value(self):
        """Outer loop's init value must NOT become a parameter of the outlined incore function.

        When an incore scope uses a loop-carried variable (IterArg) from an
        outer ForStmt, only the IterArg itself should be captured as a
        parameter, not its initValue_ expression.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self, init: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                for sb, (acc,) in pl.range(4, init_values=(init,)):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        result: pl.Tensor[[64], pl.FP32] = pl.add(acc, y)
                    acc_rv = pl.yield_(result)
                return acc_rv

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, acc: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(acc, y)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, init: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                for sb, (acc,) in pl.range(4, init_values=(init,)):
                    result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, y)
                    acc_rv = pl.yield_(result)
                return acc_rv

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_assemble_dest_upgraded_to_inout(self):
        """``tensor.assemble`` destination captured by a scope becomes an InOut param.

        ``tensor.assemble`` is SSA-pure (returns a fresh Tensor), but its first
        argument is a destination the result aliases. When the destination is a
        captured outer variable, the outlined function effectively reads and
        writes the same backing buffer, so ``InferParamDirections`` Step 2
        (``AssembleDestUpgrader``, scope_outline_utils.h:1129-1153) upgrades that
        parameter from ``In`` to ``InOut``. The ``src`` argument is only read, so
        it stays ``In`` — pinning both directions makes the InOut upgrade the
        load-bearing assertion (``assert_structural_equal`` is direction-aware).
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self, dst: pl.Tensor[[64], pl.FP32], src: pl.Tensor[[32], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    updated: pl.Tensor[[64], pl.FP32] = pl.assemble(dst, src, [0])
                return updated

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, dst: pl.InOut[pl.Tensor[[64], pl.FP32]], src: pl.Tensor[[32], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                updated: pl.Tensor[[64], pl.FP32] = pl.assemble(dst, src, [0])
                return updated

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, dst: pl.Tensor[[64], pl.FP32], src: pl.Tensor[[32], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                updated: pl.Tensor[[64], pl.FP32] = self.main_incore_0(dst, src)
                return updated

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_scope_inside_while_captures_iter_arg(self):
        """Scope inside a WhileStmt body captures the loop-carried IterArg, not its init.

        WhileStmt iter-args are SSA values bound by the loop, so a scope that
        reads ``acc`` (the while iter-arg) must take ``acc`` as a parameter while
        the loop's init value (``x``) stays out of the callee signature. This is
        the while-loop counterpart of ``test_outline_scope_with_loop_carried_init_values``
        and exercises ``VarCollector::VisitStmt_(WhileStmtPtr)``
        (scope_outline_utils.h:200-212), which seeds the symbol table with the
        while's iter-args / return-vars so they resolve as captured inputs.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64], pl.FP32]:
                for acc, c in pl.while_(init_values=(x, 0)):
                    pl.cond(c < n)
                    with pl.at(level=pl.Level.CORE_GROUP):
                        updated: pl.Tensor[[64], pl.FP32] = pl.add(acc, y)
                    c_new: pl.Scalar[pl.INDEX] = c + 1
                    acc_rv, c_rv = pl.yield_(updated, c_new)
                return acc_rv

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, acc: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                updated: pl.Tensor[[64], pl.FP32] = pl.add(acc, y)
                return updated

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64], pl.FP32]:
                for acc, c in pl.while_(init_values=(x, 0)):
                    pl.cond(c < n)
                    updated: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, y)
                    c_new: pl.Scalar[pl.INDEX] = c + 1
                    acc_rv, c_rv = pl.yield_(updated, c_new)
                return acc_rv

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)


class TestOutlineSubmitTaskId:
    """``with pl.at(...) as tid:`` emits an ``ir.Submit`` (not a plain Call).

    When a scope captures a producer TaskId via the ``as tid`` binding (or
    carries ``deps=[...]``), the outliner emits an ``ir.Submit`` whose return
    type is augmented with a trailing ``Scalar[TASK_ID]`` and whose call site
    unpacks the flat tuple: ``out = ret[0]`` ... ``tid = ret[last]``. This
    matches the explicit ``out, tid = pl.submit(self.kernel, ...)`` DSL surface,
    so the Expected programs author the equivalent ``pl.submit`` form directly.
    See scope_outline_utils.h:836-931 (Submit emission) and 961-978 (TaskId
    tuple-unpack call site).
    """

    def test_outline_scope_with_task_id_emits_submit(self):
        """A lone ``as tid`` scope outlines to a deps-free ``ir.Submit``."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP) as tid:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # ``y, tid = pl.submit(...)`` desugars to AssignStmt(_submit_tmp,
                # Submit) + TupleGetItem unpack — exactly the outliner's Submit
                # emission. ``tid`` is the trailing TaskId tuple element.
                y, tid = pl.submit(self.main_incore_0, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_chained_task_id_deps_fold_into_submit_deps(self):
        """``deps=[tid0]`` on a second scope folds into the second ``Submit``'s deps_.

        Two ordered scopes: the first binds ``tid0``; the second declares
        ``deps=[tid0]``. The outliner emits two ``ir.Submit`` nodes — the second
        carries ``tid0`` in its first-class ``deps_`` field (folded from the
        scope's ``manual_dep_edges`` attr because ``task_id_var`` is set;
        scope_outline_utils.h:896-901). The Expected expresses this as
        ``pl.submit(self.main_incore_1, y, deps=[tid0])``.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP) as tid0:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.CORE_GROUP, deps=[tid0]) as tid1:
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_1(self, y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y, tid0 = pl.submit(self.main_incore_0, x)
                z, tid1 = pl.submit(self.main_incore_1, y, deps=[tid0])
                return z

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_deps_only_scope_emits_submit(self):
        """A ``deps=[tid0]`` scope WITHOUT ``as tid`` still outlines to an ``ir.Submit``.

        The second scope consumes ``tid0`` via ``deps=[...]`` but does not bind
        a TaskId of its own. The outliner must still emit an ``ir.Submit`` (not
        a plain Call): its first-class ``deps_`` carries the single ``tid0``
        edge, and the return type is augmented with the trailing
        ``Scalar[TASK_ID]`` — the synthesized TaskId name is internal, so only
        kind/deps/type-shape are asserted (no exact tid name).
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP) as tid0:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.CORE_GROUP, deps=[tid0]):
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_incore_scopes()(Before)

        submits: list[ir.Submit] = []

        class _Collector(ir.IRVisitor):
            def visit_submit(self, op):
                submits.append(op)
                super().visit_submit(op)

        _Collector().visit_program(After)

        # Both scopes outline to Submits; the deps-only scope is the second.
        assert len(submits) == 2
        call = submits[1]
        assert isinstance(call, ir.Submit)
        assert len(call.deps) == 1
        assert call.op.name == "main_incore_1"
        # Return type carries the trailing Scalar[TASK_ID].
        assert isinstance(call.type, ir.TupleType)
        tail = call.type.types[-1]
        assert isinstance(tail, ir.ScalarType)
        assert tail.dtype == DataType.TASK_ID


class TestOutlineSpmdScope:
    """InCore scopes nested inside a ``SpmdScopeStmt`` are outlined, wrapper kept.

    ``for i in pl.spmd(n):`` desugars to ``SpmdScopeStmt(InCoreScopeStmt(...))``
    in an Orchestration function. ``OutlineIncoreScopes`` processes Orchestration
    functions (outline_incore_scopes_pass.cpp:42-44): it descends into the
    SpmdScopeStmt with ``inside_nested_scope_body_`` set, outlines the inner
    InCore body into a separate function, and replaces it with a Call while
    leaving the ``with pl.spmd(...)`` wrapper in place around the Call.
    """

    def test_outline_inner_incore_keeps_spmd_wrapper(self):
        """The inner InCore body is outlined; the SpmdScope wrapper survives.

        The scope body writes ``out`` via ``tile.store``, so ``out`` is a store
        target: it becomes an ``InOut`` callee param and a returned (renamed)
        value (``out_v1`` -> ``out_store``), mirroring
        ``test_outline_scope_with_store_only_outputs``.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4):
                    offset = i * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                i: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                offset = i * 128
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                out_v1: pl.Tensor[[512, 128], pl.FP32] = pl.store(t, [offset, 0], out)
                out_store: pl.Tensor[[512, 128], pl.FP32] = out_v1
                return out_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4):
                    out_v2: pl.Tensor[[512, 128], pl.FP32] = self.main_incore_0(a, out)
                return out_v2

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)


class TestSplitIncoreOrchVerifier:
    """Regression tests for the SplitIncoreOrch property verifier."""

    def _build_outlined_program(self, input_program):
        """Run convert_to_ssa + outline_incore_scopes."""
        program = passes.convert_to_ssa()(input_program)
        program = passes.outline_incore_scopes()(program)
        return program

    @staticmethod
    def _split_incore_orch_props():
        ps = passes.IRPropertySet()
        ps.insert(passes.IRProperty.SplitIncoreOrch)
        return ps

    def test_clean_orchestration_passes_verification(self):
        """Outlined program with all compute in InCore passes property verification."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        After = self._build_outlined_program(Input)
        # Should not throw — no InCore scopes remain, no errors
        passes.verify_properties(self._split_incore_orch_props(), After, "test")

    def test_remaining_incore_scope_fails_verification(self):
        """Leftover InCore ScopeStmt in non-InCore function causes verification failure."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        # Don't outline — just convert to SSA, leaving InCore scope intact
        program = passes.convert_to_ssa()(Input)

        # verify_properties should throw because InCore scope remains in Opaque function
        with pytest.raises(Exception, match="InCore ScopeStmt"):
            passes.verify_properties(self._split_incore_orch_props(), program, "test")

    def test_compute_op_in_orchestration_does_not_fail(self):
        """Compute tensor op in Orchestration produces warning (not error), verification passes."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                return y

        After = self._build_outlined_program(Input)
        # Orchestration has tensor.add — but it's a warning, not an error
        # verify_properties should NOT throw
        passes.verify_properties(self._split_incore_orch_props(), After, "test")

    def test_outline_does_not_throw_for_clean_program(self):
        """Running outline_incore_scopes on a clean program does not throw."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        # Run with full verification enabled — should not throw
        program = passes.convert_to_ssa()(Input)
        passes.outline_incore_scopes()(program)

    def test_outline_with_compute_outside_incore_verification_passes(self):
        """Compute ops outside incore in explicit pl.incore() usage: verification passes (warning only)."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                result: pl.Tensor[[64], pl.FP32] = pl.add(y, y)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(a)
                result: pl.Tensor[[64], pl.FP32] = pl.add(y, y)
                return result

        # Run with full verification — should pass despite compute ops in orchestration
        program = passes.convert_to_ssa()(Input)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(program)
        ir.assert_structural_equal(After, Expected)

    def test_full_pipeline_with_verification_passes(self):
        """Full pipeline with auto_incore: no compute ops leak into Orchestration."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    x = pl.add(x, 1.0)
                    for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, 2.0)
                return x

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                x1: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return x1

            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i, (xi,) in pl.parallel(
                    4, init_values=(x,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
                ):
                    x4: pl.Tensor[[64], pl.FP32] = pl.add(xi, 2.0)
                    xrv = pl.yield_(x4)
                return xrv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                x1: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                for i, (xi,) in pl.parallel(
                    2, init_values=(x1,), attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}
                ):
                    xrv: pl.Tensor[[64], pl.FP32] = self.main_incore_1(xi)
                    xorv = pl.yield_(xrv)
                return xorv

        # Run the full pipeline with verification enabled — should not throw
        program = passes.unroll_loops()(Input)
        program = passes.convert_to_ssa()(program)
        program = passes.flatten_call_expr()(program)
        program = passes.split_chunked_loops()(program)
        program = passes.interchange_chunk_loops()(program)
        program = passes.outline_incore_scopes()(program)

        ir.assert_structural_equal(program, Expected)


class TestOutlineNamedIncoreScopes:
    """Test OutlineIncoreScopes pass with user-provided scope names."""

    def test_outline_named_incore_scope(self):
        """Test that user-provided name is used for the outlined function."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="fused_add"):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def fused_add(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.fused_add(x)
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_mixed_named_and_unnamed_scopes(self):
        """Test that unnamed scopes still get auto-generated names when mixed with named scopes."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="first_kernel"):
                    a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.CORE_GROUP):
                    b: pl.Tensor[[64], pl.FP32] = pl.add(y, a)
                return b

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def first_kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return a

            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_1(
                self,
                y: pl.Tensor[[64], pl.FP32],
                a: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                b: pl.Tensor[[64], pl.FP32] = pl.add(y, a)
                return b

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = self.first_kernel(x)
                b: pl.Tensor[[64], pl.FP32] = self.main_incore_1(y, a)
                return b

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_duplicate_name_hint_auto_dedup(self):
        """Test that duplicate name_hints are auto-deduplicated."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="my_kernel"):
                    a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="my_kernel"):
                    b: pl.Tensor[[64], pl.FP32] = pl.add(y, a)
                return b

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def my_kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return a

            @pl.function(type=pl.FunctionType.InCore)
            def my_kernel_0(
                self,
                y: pl.Tensor[[64], pl.FP32],
                a: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                b: pl.Tensor[[64], pl.FP32] = pl.add(y, a)
                return b

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = self.my_kernel(x)
                b: pl.Tensor[[64], pl.FP32] = self.my_kernel_0(y, a)
                return b

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_duplicate_name_hint_across_functions(self):
        """Sibling functions reusing the same ``name_hint`` must not collide.

        Regression test for issue #1711: composing independently-runnable child
        kernels (e.g. two kernels reusing one ``@pl.jit.inline`` helper) yields a
        program where multiple Orchestration functions each outline an InCore
        scope carrying the *same* ``name_hint``. The outlined functions land in a
        single namespace, so the bare hint would clash at Program construction.
        The pass disambiguates a *cross-function* collision by namespacing it
        under the originating function: ``fn_a`` keeps ``dup`` (first seen,
        stable, matching its standalone compilation), ``fn_b`` gets the
        source-derived ``fn_b_dup``. This differs from the *in-function* dedup
        above, which keeps the historical numeric ``_0`` suffix.
        """

        @pl.program
        class Before:
            @pl.function
            def fn_a(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dup"):
                    a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return a

            @pl.function
            def fn_b(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dup"):
                    b: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return b

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def dup(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return a

            @pl.function(type=pl.FunctionType.InCore)
            def fn_b_dup(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                b: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return b

            @pl.function(type=pl.FunctionType.Orchestration)
            def fn_a(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = self.dup(x)
                return a

            @pl.function(type=pl.FunctionType.Orchestration)
            def fn_b(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                b: pl.Tensor[[64], pl.FP32] = self.fn_b_dup(x)
                return b

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)


class TestOutlineNoDepArgs:
    """``pl.at(no_dep_args=[t])`` lowering: ScopeStmt.attrs[arg_direction_overrides_vars]
    is translated by the outliner into per-call positional indices stored as
    ``Call.attrs[arg_direction_overrides]``, which DeriveCallDirections then
    consumes to overwrite the auto-derived direction at each slot to NoDep.

    These tests run under the default RoundtripInstrument (print/reparse after
    every pass). The Call printer now surfaces ``attrs[arg_direction_overrides]``
    generically (``PrintAttrValue``) and the parser recovers it
    (``_parse_attr_value``), so the synthesised no-dep dispatch round-trips.
    """

    @staticmethod
    def _outlined_user_call(program: ir.Program) -> ir.Call:
        """Return the synthesised Call inside main that targets the outlined kernel."""
        main = program.get_function("main")
        assert main is not None
        body = main.body
        stmts = list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]
        for s in stmts:
            value = getattr(s, "value", None)
            if isinstance(value, ir.Call) and isinstance(value.op, ir.GlobalVar):
                return value
            if isinstance(s, ir.EvalStmt) and isinstance(s.expr, ir.Call):
                if isinstance(s.expr.op, ir.GlobalVar):
                    return s.expr
        raise AssertionError(f"no outlined kernel Call found in main, stmts={stmts}")

    def test_outline_translates_no_dep_args_to_indices(self):
        """Captured-Var order → positional indices on the synthesised Call."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                w: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, no_dep_args=[w]):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, w)
                return y

        After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

        call = self._outlined_user_call(After)
        # Captured order: x first (referenced before w), w second.
        # The outlined function's signature reflects that order, so the
        # NoDep override for w lands at index 1.
        overrides = call.attrs.get("arg_direction_overrides")
        assert overrides == [1], f"expected [1], got {overrides!r}"
        # The scope-level marker has been consumed — it must NOT survive on
        # the synthesised Call (it is exclusively a ScopeStmt-level attr).
        assert "arg_direction_overrides_vars" not in call.attrs

    def test_outline_plus_derive_marks_slot_no_dep(self):
        """Indices recorded by the outliner are consumed by DeriveCallDirections
        to overwrite the slot's direction to ``ArgDirection.NoDep``.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                w: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, no_dep_args=[w]):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, w)
                return y

        After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
        After = passes.derive_call_directions()(After)

        call = self._outlined_user_call(After)
        dirs = list(call.arg_directions)
        # The unmarked tensor (x) keeps its auto-derived direction (Input);
        # the marked tensor (w) is forced to NoDep regardless of how the
        # auto-deriver would otherwise classify it.
        assert dirs[1] == ir.ArgDirection.NoDep, f"expected NoDep at slot 1, got {dirs}"
        assert dirs[0] != ir.ArgDirection.NoDep, f"slot 0 should keep auto-direction, got {dirs}"

    def test_outline_plus_derive_no_dep_on_mutated_capture(self):
        """``pl.at(no_dep_args=[k])`` is legal when the scope body mutates ``k``
        via ``pl.assemble`` — i.e. the synthesised kernel param direction for
        ``k`` is ``InOut`` rather than ``In``.

        Mirrors the qwen3-style paged-KV-cache pattern: ``k_cache`` and
        ``v_cache`` are written at a data-dependent offset inside a parallel
        fan-out, so the compiler cannot prove sibling writes are disjoint;
        the user opts the slots out of OverlapMap tracking via
        ``no_dep_args=`` because the runtime slot allocation protocol
        guarantees disjointness.
        """
        from pypto.pypto_core import passes as _core_passes  # noqa: PLC0415

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                k_cache: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, no_dep_args=[k_cache]):
                    # The scope body writes into ``k_cache`` via pl.assemble —
                    # the outliner infers ``InOut`` for the synthesised
                    # callee's k_cache param. Without the relaxed Out+NoDep
                    # rule, the post-pass verifier would reject the resulting
                    # ``InOut`` (callee) + ``NoDep`` (call-site) combination.
                    k_cache = pl.assemble(k_cache, x, [0])
                return k_cache

        After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
        After = passes.derive_call_directions()(After)

        call = self._outlined_user_call(After)
        # Locate the k_cache slot. SSA conversion renames k_cache to a
        # ``k_cache__rv_N``-style version, so match by name prefix rather
        # than exact identity. (Captured-Var order depends on outliner
        # traversal — we don't pin the position.)
        k_cache_idx = next(
            (
                i
                for i, a in enumerate(call.args)
                if isinstance(a, ir.Var) and (a.name_hint == "k_cache" or a.name_hint.startswith("k_cache"))
            ),
            None,
        )
        assert k_cache_idx is not None, (
            f"k_cache not found in outlined call args: "
            f"{[a.name_hint for a in call.args if isinstance(a, ir.Var)]}"
        )

        dirs = list(call.arg_directions)
        # The marked tensor (k_cache) is forced to NoDep even though the
        # synthesised callee declares it as InOut (because pl.assemble inside
        # the body writes into it).
        assert dirs[k_cache_idx] == ir.ArgDirection.NoDep, (
            f"expected NoDep at k_cache slot {k_cache_idx}, got {dirs}"
        )
        # The synthesised callee classifies the mutated capture as InOut
        # (asserted indirectly via the verifier below, which rejects
        # NoDep on a callee `In` param if some sibling argument got
        # re-classified).

        # And the post-pass property verifier must accept the InOut+NoDep
        # combination on the synthesised Call.
        props = _core_passes.IRPropertySet()
        props.insert(_core_passes.IRProperty.CallDirectionsResolved)
        _core_passes.PropertyVerifierRegistry.verify_or_throw(props, After)

        # Assert directly that the synthesised callee declares the marked
        # param as InOut — this is the load-bearing precondition that makes
        # this an InOut+NoDep test (rather than the trivial In+NoDep case
        # covered by ``test_outline_plus_derive_marks_slot_no_dep``).
        outlined = next(f for gv, f in After.functions.items() if gv.name != "main")
        assert outlined.param_directions[k_cache_idx] == ir.ParamDirection.InOut, (
            f"expected InOut at outlined callee param {k_cache_idx}, got {list(outlined.param_directions)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
