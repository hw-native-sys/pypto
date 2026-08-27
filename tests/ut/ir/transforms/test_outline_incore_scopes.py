# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for OutlineIncoreScopes pass."""

import pypto
import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import DataType, ir, passes
from pypto.ir.printer import python_print
from pypto.language.parser.diagnostics.exceptions import ParserSyntaxError
from pypto.language.parser.text_parser import parse as text_parse


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
                self, buf: pl.Out[pl.Tensor[[16, 128], pl.FP32]]
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                tile = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                buf_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(tile, [0, 0], buf)
                return buf

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
                buf_a: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
                buf_b: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 128], pl.FP32]]:
                tile_a = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                tile_b = pl.tile.full([16, 1], dtype=pl.FP32, value=0.0)
                buf_a_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(tile_a, [0, 0], buf_a)
                buf_b_store: pl.Tensor[[16, 1], pl.FP32] = pl.store(tile_b, [0, 0], buf_b)
                return (buf_b, buf_a)

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
        and returns that param; the call site still binds a fresh renamed value
        (``out -> out_v1 -> out_v2 -> ...``).
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
                return acc

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

    @staticmethod
    def _outlined_directions(program, callee="main_incore_0"):
        """Map the outlined callee's parameter names to their derived directions.

        Keys drop the ``__ssa_vN`` suffix ``ConvertToSSA`` appends, so a test can
        name the parameter as it was written in the source.
        """
        after = passes.outline_incore_scopes()(passes.convert_to_ssa()(program))
        func = after.get_function(callee)
        assert func is not None, f"{callee} was not outlined"
        return {p.name_hint.split("__ssa_v")[0]: d for p, d in zip(func.params, func.param_directions)}

    def test_outline_scalar_write_dest_becomes_out(self):
        """``tensor.write`` writes its destination, so a captured tensor written
        only through it becomes ``Out``.

        The outliner used to recognise exactly two writers, ``tile.store`` and
        ``tensor.assemble``. Every other write operator — ``tensor.write`` here
        — left the captured tensor looking untouched, so the parameter stayed
        ``In`` and the caller got no dependency on the write. The operator now
        declares which argument it writes and the outliner reads that.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self, dst: pl.Tensor[[64], pl.FP32], value: pl.Scalar[pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    updated: pl.Tensor[[64], pl.FP32] = pl.write(dst, [0], value)
                return updated

        assert self._outlined_directions(Before)["dst"] == ir.ParamDirection.Out

    def test_outline_expand_clone_target_becomes_out(self):
        """``tensor.expand_clone`` stores into its ``target`` on every lowering
        branch, so a captured target is written, not read."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self, src: pl.Tensor[[1, 1, 32], pl.FP32], dst: pl.Tensor[[1, 8, 32], pl.FP32]
            ) -> pl.Tensor[[1, 8, 32], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    expanded: pl.Tensor[[1, 8, 32], pl.FP32] = pl.expand_clone(src, dst)
                return expanded

        directions = self._outlined_directions(Before)
        assert directions["dst"] == ir.ParamDirection.Out
        assert directions["src"] == ir.ParamDirection.In

    def test_outline_read_then_write_dest_is_inout(self):
        """A captured tensor the scope reads *and* writes stays ``InOut``. The
        write widens the direction; it does not replace the read."""

        @pl.program
        class Before:
            @pl.function
            def main(self, dst: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    first: pl.Scalar[pl.FP32] = pl.read(dst, [0])
                    updated: pl.Tensor[[64], pl.FP32] = pl.write(dst, [1], first)
                return updated

        assert self._outlined_directions(Before)["dst"] == ir.ParamDirection.InOut

    def test_outline_write_only_assemble_dest_becomes_out(self):
        """``tensor.assemble`` destination captured by a scope becomes an Out param.

        ``tensor.assemble`` is SSA-pure (returns a fresh Tensor), but its first
        argument is a destination the result aliases in place. When the
        destination is a captured outer variable, the outlined function writes
        the caller's backing buffer, so ``InferParamDirections`` lifts that
        parameter off ``In`` from the operator's declared write. Here the body
        never reads ``dst`` — the assemble destination slot is the only use — so
        the direction is ``Out``, not ``InOut`` (issue #2415: a false ``InOut``
        reaches ``DistributedCodegen`` and manufactures a cross-rank edge). The
        ``src`` argument is only read, so it stays ``In`` — pinning both
        directions makes the derived direction the load-bearing assertion
        (``assert_structural_equal`` is direction-aware).
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
                self, dst: pl.Out[pl.Tensor[[64], pl.FP32]], src: pl.Tensor[[32], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                updated: pl.Tensor[[64], pl.FP32] = pl.assemble(dst, src, [0])
                return dst

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

    def test_outline_read_assemble_dest_stays_inout(self):
        """The same scope that also *reads* the destination keeps ``InOut``.

        The write-only case above earns ``Out``; here ``dst`` additionally
        feeds a ``pl.tensor.slice``, which is a genuine read of the incoming
        buffer, so the direction must stay ``InOut``. Pairing the two pins the
        distinction rather than the blanket upgrade that issue #2415 reported.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self, dst: pl.Tensor[[64], pl.FP32], src: pl.Tensor[[32], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    head: pl.Tensor[[32], pl.FP32] = pl.tensor.slice(dst, [32], [0], [], [])
                    mixed: pl.Tensor[[32], pl.FP32] = pl.add(head, src)
                    updated: pl.Tensor[[64], pl.FP32] = pl.assemble(dst, mixed, [0])
                return updated

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, dst: pl.InOut[pl.Tensor[[64], pl.FP32]], src: pl.Tensor[[32], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                head: pl.Tensor[[32], pl.FP32] = pl.tensor.slice(dst, [32], [0], [], [])
                mixed: pl.Tensor[[32], pl.FP32] = pl.add(head, src)
                updated: pl.Tensor[[64], pl.FP32] = pl.assemble(dst, mixed, [0])
                return dst

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

    def test_outline_read_through_post_write_alias_is_inout(self):
        """A read of the store's SSA result is a read of the destination.

        Under SSA the post-write state binds to a fresh Var (``out_v1 =
        tile.store(t, off, out_v0)``), which names the *same* backing buffer —
        the aliasing ``PostStoreAliasCollector`` records for export bookkeeping.
        Reading that alias over a region the scope never wrote genuinely needs
        the incoming contents, so the direction must be ``InOut``. Recognising
        only the original captured Var would derive ``Out`` and drop the
        dependency on those contents.

        The scope writes rows 0 and 384 and reads row 256, so the read cannot be
        satisfied by anything the scope itself produced.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    out2 = pl.store(t, [0, 0], out)
                    back: pl.Tile[[128, 128], pl.FP32] = pl.load(out2, [256, 0], [128, 128])
                    out3 = pl.store(back, [384, 0], out2)
                return out3

        After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

        outlined = next(f for gv, f in After.functions.items() if gv.name != "main")
        out_idx = next(i for i, p in enumerate(outlined.params) if p.name_hint.startswith("out"))
        assert outlined.param_directions[out_idx] == ir.ParamDirection.InOut, (
            f"expected InOut at outlined callee param {out_idx}, got {list(outlined.param_directions)}"
        )

    def test_outline_atomic_store_dest_is_inout(self):
        """An atomic-add store reads its destination, so it stays ``InOut``.

        ``pl.store(t, off, out, atomic=pl.AtomicType.Add)`` is ``out += t``: the
        accumulator's existing contents are an operand. Treating the store
        target as a pure overwrite would derive ``Out``, the runtime would skip
        host->device staging for it, and the accumulation would start from
        allocator garbage instead of the caller's zeros — NaN for a float
        accumulator, silently wrong sums for an integer one.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    out = pl.store(t, [0, 0], out, atomic=pl.AtomicType.Add)
                return out

        After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

        outlined = next(f for gv, f in After.functions.items() if gv.name != "main")
        out_idx = next(i for i, p in enumerate(outlined.params) if p.name_hint.startswith("out"))
        assert outlined.param_directions[out_idx] == ir.ParamDirection.InOut, (
            f"expected InOut at outlined callee param {out_idx}, got {list(outlined.param_directions)}"
        )

    def test_outline_atomic_assemble_dest_is_inout(self):
        """The ``tensor.assemble`` half of the same rule.

        ``pl.assemble(c, partial, off, atomic=pl.AtomicType.Add)`` is how split-K
        matmul accumulates each core's partial product into the shared output, so
        the destination is read there too.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                src: pl.Tensor[[32], pl.FP32],
                dst: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    dst = pl.assemble(dst, src, [0], atomic=pl.AtomicType.Add)
                return dst

        After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

        outlined = next(f for gv, f in After.functions.items() if gv.name != "main")
        dst_idx = next(i for i, p in enumerate(outlined.params) if p.name_hint.startswith("dst"))
        assert outlined.param_directions[dst_idx] == ir.ParamDirection.InOut, (
            f"expected InOut at outlined callee param {dst_idx}, got {list(outlined.param_directions)}"
        )

    def test_outline_bookkeeping_attr_is_not_a_read(self):
        """``dumps=[out]`` names a tensor without reading it.

        ``dump_vars`` (and ``arg_direction_overrides_vars``) are bookkeeping
        references, so counting them as reads would promote a write-only capture
        back to ``InOut`` — re-creating the false cross-rank dependency of issue
        #2415 for exactly the programs that asked for dump or ``NoDep``
        treatment. The scope here only stores into ``out``.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, dumps=[out]):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    out = pl.store(t, [0, 0], out)
                return out

        After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

        outlined = next(f for gv, f in After.functions.items() if gv.name != "main")
        out_idx = next(i for i, p in enumerate(outlined.params) if p.name_hint.startswith("out"))
        assert outlined.param_directions[out_idx] == ir.ParamDirection.Out, (
            f"expected Out at outlined callee param {out_idx}, got {list(outlined.param_directions)}"
        )

    def test_outline_mgather_scratch_stays_out(self):
        """Writing an argument does not make the result name it.

        ``tile.mgather`` stages a Mat *elem* gather through the GM ``scratch``
        operand — its only written argument — but returns a **fresh** tile.
        Inferring the result alias from "the one write slot" registered the
        gathered tile as a second name for ``scratch``, so reading the tile
        marked a write-only ``scratch`` as read and promoted it to ``InOut``.
        That is the false read of issue #2415, on a capture whose contents the
        scope never needs staged in.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                mem: pl.Tensor[[512], pl.FP32],
                idx: pl.Tensor[[16, 32], pl.INT32],
                scratch: pl.Out[pl.Tensor[[512], pl.FP32]],
                out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    gathered: pl.Tile[[16, 32], pl.FP32] = pl.tile.mgather(
                        mem, idx, coalesce="elem", target_memory=pl.MemorySpace.Mat, scratch=scratch
                    )
                    # Reading the gathered tile must not read `scratch`.
                    vec: pl.Tile[[16, 32], pl.FP32] = pl.move(gathered, target_memory=pl.MemorySpace.Vec)
                    out = pl.store(vec, [0, 0], out)
                return out

        After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

        outlined = next(f for gv, f in After.functions.items() if gv.name != "main")
        directions = {
            p.name_hint.split("__ssa_v")[0]: d for p, d in zip(outlined.params, outlined.param_directions)
        }
        assert directions["scratch"] == ir.ParamDirection.Out, (
            f"mgather scratch is written, never read; got {directions}"
        )
        assert directions["out"] == ir.ParamDirection.Out, f"expected Out for out, got {directions}"

    def test_outline_spmd_assemble_keeps_out_param_out(self):
        """Regression for issue #2415: a ``pl.Out`` formal stays ``Out``.

        ``for row in pl.spmd(ROWS): dst[row] = src[row]`` lowers to an
        ``InCoreScopeStmt`` whose body assembles into ``dst`` and reads nothing
        of it. Before the fix the outlined callee came out ``InOut`` while the
        caller and the top-level formal both stayed ``pl.Out``. That false read
        reaches ``DistributedCodegen::EmitCallToWorker``, which tags each
        per-rank chip dispatch from the callee direction — so two ranks writing
        *disjoint* rows of one rank-major ``pl.Out`` tensor were given a
        cross-rank write dependency and the model deadlocked on the first
        rendezvous.

        This is the issue's minimal case, checked at the level it breaks: the
        outlined callee's ``param_directions``.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def outline_out_direction(
                self,
                src: pl.Tensor[[8, 8], pl.INT32],
                dst: pl.Out[pl.Tensor[[8, 8], pl.INT32]],
            ) -> pl.Tensor[[8, 8], pl.INT32]:
                for row in pl.spmd(8, name_hint="fill_rows"):
                    t: pl.Tensor[[1, 8], pl.INT32] = pl.tensor.slice(src, [1, 8], [row, 0], [], [])
                    dst = pl.assemble(dst, t, [row, 0])
                return dst

        After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

        outlined = next(f for gv, f in After.functions.items() if gv.name == "fill_rows")
        dst_idx = next(i for i, p in enumerate(outlined.params) if p.name_hint.startswith("dst"))
        assert outlined.param_directions[dst_idx] == ir.ParamDirection.Out, (
            f"expected Out at outlined callee param {dst_idx}, got {list(outlined.param_directions)}"
        )
        # The caller's own formal is untouched — the point of the bug was that
        # the two disagreed across the boundary the pass had just introduced.
        # Resolve the caller's slot on its own params: the outliner orders the
        # callee signature by capture order, which need not match the caller's.
        caller = After.get_function("outline_out_direction")
        assert caller is not None
        caller_dst_idx = next(i for i, p in enumerate(caller.params) if p.name_hint.startswith("dst"))
        assert caller.param_directions[caller_dst_idx] == ir.ParamDirection.Out

    def test_out_direction_survives_to_the_chip_formal(self):
        """Regression for issue #2415, checked where the deadlock came from.

        The outliner's direction does not stay local: ``ConvertTensorToTileOps``
        propagates a callee ``InOut`` onto the caller's parameter
        (convert_tensor_to_tile_ops_pass.cpp:2677-2678). So a false ``InOut`` on
        the outlined InCore kernel rewrites the *chip-level* formal from ``Out``
        to ``InOut`` — and that formal is exactly what
        ``DistributedCodegen::EmitCallToWorker`` reads to tag each per-rank chip
        dispatch, turning disjoint per-rank slices into a cross-rank write
        dependency.

        Pinning both functions after both passes is what makes this a test of
        the whole chain rather than of one pass's output.
        """

        @pl.program
        class Before:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self,
                src: pl.Tensor[[8, 128], pl.FP32],
                dst: pl.Out[pl.Tensor[[8, 128], pl.FP32]],
            ) -> pl.Tensor[[8, 128], pl.FP32]:
                for row in pl.spmd(8, name_hint="fill_rows"):
                    t: pl.Tile[[1, 128], pl.FP32] = pl.load(src, [row, 0], [1, 128])
                    dst = pl.store(t, [row, 0], dst)
                return dst

        After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
        After = passes.convert_tensor_to_tile_ops()(After)

        for name in ("chip_orch", "fill_rows"):
            func = After.get_function(name)
            assert func is not None, f"{name} missing from {list(After.functions)}"
            dst_idx = next(i for i, p in enumerate(func.params) if p.name_hint.startswith("dst"))
            assert func.param_directions[dst_idx] == ir.ParamDirection.Out, (
                f"{name}: expected Out at param {dst_idx}, got {list(func.param_directions)}"
            )

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

    def test_deferred_wait_does_not_order_later_notify_behind_waiter(self):
        """Notify writes its signal, and still carries no waiter dependency.

        The deferred waiter is logically live after its AIV kernel returns, so
        a spurious waiter -> notifier edge would recreate the physical-core
        saturation deadlock. That edge is what this test guards, and a MANUAL
        scope leaves the later notifier unordered regardless of direction.

        The directions themselves are not symmetric. ``pld.system.wait`` /
        ``defer_wait`` poll a signal they never write, so the waiter's parameter
        is ``Input``. ``pld.system.notify`` deposits a value into the peer's
        slot — that is the write whose absence dropped the RAW edge a waiter
        needs and deadlocked the communication card, so the notifier's parameter
        is ``Out`` and its call site ``OutputExisting``. The operator declares
        both facts on the registry, so the outliner and ``ConvertTensorToTileOps``
        read the same answer instead of disagreeing about the same call.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                signal: pld.DistributedTensor[[1, 1], pl.INT32],
                peer: pl.Scalar[pl.INT32],
                payload: pl.Tensor[[1], pl.INT32],
            ):
                with pl.manual_scope():
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="deferred_wait") as wait_tid:
                        pld.system.defer_wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="notifier"):
                        pld.system.notify(
                            signal,
                            peer=peer,
                            offsets=[0, 0],
                            value=1,
                            op=pld.NotifyOp.Set,
                        )
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="consumer",
                        deps=[wait_tid],
                    ):
                        payload_value = pl.read(payload, [0])

        after = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
        after = passes.derive_call_directions()(after)
        after = passes.auto_derive_task_dependencies(analyze_auto_scopes=True)(after)

        main = after.get_function("main")
        assert main is not None
        calls: list[ir.Call | ir.Submit] = []

        class _Collector(ir.IRVisitor):
            def visit_call(self, op):
                if isinstance(op.op, ir.GlobalVar):
                    calls.append(op)
                super().visit_call(op)

            def visit_submit(self, op):
                calls.append(op)
                super().visit_submit(op)

        _Collector().visit_stmt(main.body)
        assert len(calls) == 3
        waiter, notifier, consumer = calls
        assert isinstance(waiter, ir.Submit)
        assert list(waiter.arg_directions) == [ir.ArgDirection.Input]
        assert isinstance(notifier, ir.Call)
        assert list(notifier.arg_directions) == [
            ir.ArgDirection.OutputExisting,
            ir.ArgDirection.Scalar,
        ]
        assert "manual_dep_edges" not in notifier.attrs
        assert "compiler_manual_dep_edges" not in notifier.attrs
        assert isinstance(consumer, ir.Submit)
        assert len(consumer.deps) == 1

    def test_deferred_wait_rejects_allow_early_resolve(self):
        """A deferred-completion TaskId cannot also opt into early resolve."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="deferred_wait",
                    allow_early_resolve=True,
                ) as wait_tid:
                    pld.system.defer_wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

        with pytest.raises(ValueError, match="defer_wait.*cannot use.*allow_early_resolve=True"):
            passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

    def test_terminal_deferred_waiter_needs_no_task_id_capture(self):
        """A fire-and-forget terminal waiter is a valid ordinary task dispatch."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="terminal_waiter"):
                    pld.system.defer_wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

        after = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
        waiter = after.get_function("terminal_waiter")
        assert waiter is not None
        assert waiter.attrs["deferred_completion_waiter"] is True

        main = after.get_function("main")
        assert main is not None
        calls: list[ir.Call] = []

        class _CallCollector(ir.IRVisitor):
            def visit_call(self, op):
                if isinstance(op.op, ir.GlobalVar):
                    calls.append(op)
                super().visit_call(op)

        _CallCollector().visit_stmt(main.body)
        assert len(calls) == 1

    def test_deferred_waiter_rejects_nested_early_resolve_launch(self):
        """An outer launch must not silently own waiter scheduling semantics."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
                with pl.spmd(1, allow_early_resolve=True):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="nested_waiter"):
                        pld.system.defer_wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

        with pytest.raises(ValueError, match="defer_wait must be in a task-level.*nesting.*unsupported"):
            passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

    def test_deferred_waiter_rejects_nested_predicated_launch(self):
        """A predicated outer launch cannot carry a deferred waiter."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                signal: pld.DistributedTensor[[1, 1], pl.INT32],
                gate: pl.Tensor[[1], pl.INT32],
            ):
                with pl.spmd(1, predicate=(gate[0] > 0)):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="nested_waiter"):
                        pld.system.defer_wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

        with pytest.raises(ValueError, match="defer_wait must be in a task-level.*nesting.*unsupported"):
            passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

    def test_deferred_wait_consumer_may_allow_early_resolve(self):
        """The hint is producer-side; it does not pre-stage this consumer."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                signal: pld.DistributedTensor[[1, 1], pl.INT32],
                payload: pl.Tensor[[1], pl.INT32],
            ):
                with pl.at(level=pl.Level.CORE_GROUP) as wait_tid:
                    pld.system.defer_wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    deps=[wait_tid],
                    allow_early_resolve=True,
                ):
                    payload_value = pl.read(payload, [0])

        after = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
        main = after.get_function("main")
        assert main is not None
        submits: list[ir.Submit] = []

        class _SubmitCollector(ir.IRVisitor):
            def visit_submit(self, op):
                submits.append(op)
                super().visit_submit(op)

        _SubmitCollector().visit_stmt(main.body)
        assert len(submits) == 2
        assert submits[0].allow_early_resolve is False
        assert submits[1].allow_early_resolve is True
        assert len(submits[1].deps) == 1

    def test_deferred_waiter_rejects_scalar_returning_helper_call(self):
        """Scalar bookkeeping cannot hide arbitrary kernel side effects."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def scalar_helper(self, value: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
                return value

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                signal: pld.DistributedTensor[[1, 1], pl.INT32],
                expected: pl.Scalar[pl.INT32],
            ):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="deferred_wait"):
                    hidden: pl.Scalar[pl.INT32] = self.scalar_helper(expected)
                    pld.system.defer_wait(signal, offsets=[0, 0], expected=hidden, cmp=pld.WaitCmp.Ge)

        with pytest.raises(ValueError, match="supports only a pre-registration tensor.read"):
            passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

    def test_deferred_wait_accepts_static_conditional_registration_loop(self):
        """The MoE waiter is bounded and consumers use ordinary deps unchanged.

        ``Expected`` (after both outlining passes) pins the whole shape at once:
        the waiter is outlined carrying ``deferred_completion_waiter``, the
        gather body carries no ``system.cacheinvalid``, and ``main`` ends up
        with two launches where the second takes the waiter's TaskId as its
        single dep.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                signal: pld.DistributedTensor[[8, 1], pl.INT32],
                indices: pl.Tensor[[1, 1], pl.INT32],
                payload: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
                my_rank: pl.Scalar[pl.INT32],
                epoch: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_wait") as wait_tid:
                    idx_anchor = pl.read(indices, [0, 0])
                    for src in pl.range(8):
                        if src != my_rank:
                            pld.system.defer_wait(
                                signal,
                                offsets=[src, 0],
                                expected=epoch,
                                cmp=pld.WaitCmp.Ge,
                            )
                with pl.spmd(
                    4,
                    name_hint="dispatch_gather",
                    deps=[wait_tid],
                ) as _gather_tid:
                    block = pl.tile.get_block_idx()
                    offset = block * 16
                    value = pl.load(payload, [offset], [16])
                    out = pl.store(value, [offset], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def dispatch_gather(
                self,
                payload: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                block: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                offset: pl.Scalar[pl.INDEX] = block * 16
                value: pl.Tile[[16], pl.FP32] = pl.tile.load(payload, [offset], [16], [16])
                out_1: pl.Tensor[[64], pl.FP32] = pl.tile.store(value, [offset], out)
                out_store: pl.Tensor[[64], pl.FP32] = out_1
                return out

            @pl.function(type=pl.FunctionType.Spmd, strict_ssa=True)
            def dispatch_gather_spmd(
                self,
                payload: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out_2: pl.Tensor[[64], pl.FP32] = self.dispatch_gather(payload, out)
                return out

            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def dispatch_wait(
                self,
                indices: pl.Tensor[[1, 1], pl.INT32],
                my_rank: pl.Scalar[pl.INT32],
                signal: pld.DistributedTensor[[8, 1], pl.INT32],
                epoch: pl.Scalar[pl.INT32],
            ):
                pl.func_attr({"deferred_completion_waiter": True})
                idx_anchor: pl.Scalar[pl.INT32] = pl.tensor.read(indices, [0, 0])
                for src in pl.range(8):
                    if src != pl.cast(my_rank, pl.INDEX):
                        pld.system.defer_wait(signal, [src, 0], epoch, cmp=pld.WaitCmp.Ge)

            @pl.function(type=pl.FunctionType.Orchestration, strict_ssa=True)
            def main(
                self,
                signal: pld.DistributedTensor[[8, 1], pl.INT32],
                indices: pl.Tensor[[1, 1], pl.INT32],
                payload: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
                my_rank: pl.Scalar[pl.INT32],
                epoch: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[64], pl.FP32]:
                wait_ret: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.submit(
                    self.dispatch_wait, indices, my_rank, signal, epoch
                )
                wait_tid: pl.Scalar[pl.TASK_ID] = wait_ret[0]
                gather_ret: pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.dispatch_gather_spmd, payload, out, deps=[wait_tid], core_num=4
                )
                out_2: pl.Tensor[[64], pl.FP32] = gather_ret[0]
                gather_tid: pl.Scalar[pl.TASK_ID] = gather_ret[1]
                return out_2

        after = passes.outline_cluster_scopes()(
            passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
        )
        ir.assert_structural_equal(after, Expected)

    def test_deferred_wait_accepts_pure_scalar_temporary_in_registration_loop(self):
        """Hoisting pure expected-value arithmetic must not change legality."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                signal: pld.DistributedTensor[[4, 1], pl.INT32],
                epoch: pl.Scalar[pl.INT32],
            ):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="waiter"):
                    for src in pl.range(4):
                        wanted = epoch * 4
                        pld.system.defer_wait(
                            signal,
                            offsets=[src, 0],
                            expected=wanted,
                            cmp=pld.WaitCmp.Ge,
                        )

        after = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
        waiter = after.get_function("waiter")
        assert waiter is not None
        assert waiter.attrs["deferred_completion_waiter"] is True

    def test_deferred_wait_rejects_tensor_read_in_registration_loop(self):
        """A later iteration must not read tensor state after registration began."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                signal: pld.DistributedTensor[[2, 1], pl.INT32],
                expected_values: pl.Tensor[[2], pl.INT32],
            ):
                with pl.at(level=pl.Level.CORE_GROUP):
                    for src in pl.range(2):
                        wanted = pl.read(expected_values, [src])
                        pld.system.defer_wait(
                            signal,
                            offsets=[src, 0],
                            expected=wanted,
                            cmp=pld.WaitCmp.Ge,
                        )

        with pytest.raises(ValueError, match="loop may execute tensor reads.*after registering"):
            passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

    def test_deferred_wait_accepts_phi_carried_scalar_threshold(self):
        """A branch-merged scalar is SSA bookkeeping, not continuation work."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                signal: pld.DistributedTensor[[4, 1], pl.INT32],
                my_rank: pl.Scalar[pl.INT32],
                low: pl.Scalar[pl.INT32],
                high: pl.Scalar[pl.INT32],
            ):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="waiter"):
                    if my_rank == 0:
                        wanted = low
                    else:
                        wanted = high
                    pld.system.defer_wait(signal, offsets=[0, 0], expected=wanted, cmp=pld.WaitCmp.Ge)

        after = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
        waiter = after.get_function("waiter")
        assert waiter is not None
        assert waiter.attrs["deferred_completion_waiter"] is True

    def test_deferred_wait_accepts_iter_arg_carried_scalar_threshold(self):
        """A threshold advanced across iterations rides an iter_arg yield."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                signal: pld.DistributedTensor[[4, 1], pl.INT32],
                epoch: pl.Scalar[pl.INT32],
                step: pl.Scalar[pl.INT32],
            ):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="waiter"):
                    wanted = epoch
                    for src in pl.range(4):
                        pld.system.defer_wait(
                            signal,
                            offsets=[src, 0],
                            expected=wanted,
                            cmp=pld.WaitCmp.Ge,
                        )
                        wanted = wanted + step

        after = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
        waiter = after.get_function("waiter")
        assert waiter is not None
        assert waiter.attrs["deferred_completion_waiter"] is True

    def test_deferred_wait_accepts_pure_scalar_loop_that_registers_nothing(self):
        """A loop contributing zero conditions needs no registration of its own."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                signal: pld.DistributedTensor[[8, 1], pl.INT32],
                my_rank: pl.Scalar[pl.INT32],
                epoch: pl.Scalar[pl.INT32],
            ):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="waiter"):
                    coord = my_rank
                    for _ in pl.range(3):
                        coord = coord + my_rank
                    pld.system.defer_wait(signal, offsets=[coord, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

        after = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
        waiter = after.get_function("waiter")
        assert waiter is not None
        assert waiter.attrs["deferred_completion_waiter"] is True

    def test_deferred_wait_rejects_tensor_carried_across_iterations(self):
        """Accepting scalar yields must not open a path for carrying tensor state.

        The refusal comes from the AssignStmt tensor guard, which fires on the
        binding that would define the carry -- before SSA can turn it into a
        yield. ValidateScalarExpr's tensor-type guard on the yield itself is
        therefore defence in depth, unreachable from the DSL today.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                signal: pld.DistributedTensor[[4, 1], pl.INT32],
                payload: pl.Tensor[[4], pl.INT32],
            ):
                with pl.at(level=pl.Level.CORE_GROUP):
                    carried = payload
                    for src in pl.range(4):
                        pld.system.defer_wait(signal, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)
                        carried = carried  # noqa: PLW0127  # intentional tensor carry under test
                    _ = carried

        with pytest.raises(ValueError, match="cannot create or update payload tensors"):
            passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

    def test_deferred_wait_accepts_exactly_64_static_conditions(self):
        """The runtime capacity is inclusive: 64 conditions are supported."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, signal: pld.DistributedTensor[[64, 1], pl.INT32]):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="waiter"):
                    for src in pl.range(64):
                        pld.system.defer_wait(
                            signal,
                            offsets=[src, 0],
                            expected=1,
                            cmp=pld.WaitCmp.Ge,
                        )

        after = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
        waiter = after.get_function("waiter")
        assert waiter is not None
        assert waiter.attrs["deferred_completion_waiter"] is True

    def test_deferred_wait_rejects_more_than_64_static_conditions(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                signal: pld.DistributedTensor[[65, 1], pl.INT32],
                my_rank: pl.Scalar[pl.INT32],
            ):
                with pl.at(level=pl.Level.CORE_GROUP) as wait_tid:
                    for src in pl.range(65):
                        if src != my_rank:
                            pld.system.defer_wait(
                                signal,
                                offsets=[src, 0],
                                expected=1,
                                cmp=pld.WaitCmp.Ge,
                            )
                with pl.at(level=pl.Level.CORE_GROUP, deps=[wait_tid]):
                    pl.system.cacheinvalid()

        with pytest.raises(ValueError, match="at most 64 conditions"):
            passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

    def test_deferred_wait_rejects_dynamic_registration_loop(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                signal: pld.DistributedTensor[[64, 1], pl.INT32],
                limit: pl.Scalar[pl.INDEX],
            ):
                with pl.at(level=pl.Level.CORE_GROUP):
                    for src in pl.range(limit):
                        pld.system.defer_wait(
                            signal,
                            offsets=[src, 0],
                            expected=1,
                            cmp=pld.WaitCmp.Ge,
                        )

        with pytest.raises(ValueError, match="statically known positive trip count"):
            passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

    def test_deferred_wait_rejects_zero_trip_registration_loop(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, signal: pld.DistributedTensor[[1, 1], pl.INT32]):
                with pl.at(level=pl.Level.CORE_GROUP):
                    for src in pl.range(0):
                        pld.system.defer_wait(
                            signal,
                            offsets=[src, 0],
                            expected=1,
                            cmp=pld.WaitCmp.Ge,
                        )

        with pytest.raises(ValueError, match="statically known positive trip count"):
            passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

    def test_deferred_wait_rejects_nested_tensor_read_after_registration(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                signal: pld.DistributedTensor[[1, 1], pl.INT32],
                expected_values: pl.Tensor[[1], pl.INT32],
                my_rank: pl.Scalar[pl.INT32],
            ):
                with pl.at(level=pl.Level.CORE_GROUP) as wait_tid:
                    pld.system.defer_wait(
                        signal,
                        offsets=[0, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )
                    if my_rank >= 0:
                        next_expected = pl.read(expected_values, [0])
                with pl.at(level=pl.Level.CORE_GROUP, deps=[wait_tid]):
                    pl.system.cacheinvalid()

        with pytest.raises(ValueError, match="cannot execute tensor reads.*after.*registration"):
            passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))


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

        The scope body writes ``out`` via ``tile.store`` and never reads it, so
        ``out`` is a write-only store target: it becomes an ``Out`` callee param
        and the outlined function returns the param itself (param-writeback
        alias return), mirroring ``test_outline_scope_with_store_only_outputs``.
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
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                i: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                offset = i * 128
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                out_v1: pl.Tensor[[512, 128], pl.FP32] = pl.store(t, [offset, 0], out)
                out_store: pl.Tensor[[512, 128], pl.FP32] = out_v1
                return out

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
        with pytest.raises(pypto.Error, match="InCore ScopeStmt"):
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

    def test_reinterpret_view_is_classified_as_metadata(self):
        """reinterpret_view is metadata-only and does not trigger the compute-op warning."""

        @pl.program
        class Input:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8, 32], pl.INT16]:
                return pl.tensor.reinterpret_view(x, pl.INT16)

        diagnostics = passes.PropertyVerifierRegistry.verify(self._split_incore_orch_props(), Input)
        assert all("tensor.reinterpret_view" not in diagnostic.message for diagnostic in diagnostics)

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
        """Compute ops outside an explicit pl.at(CORE_GROUP) scope: verification passes (warning only)."""

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
        ``k`` is a write direction rather than ``In``.

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
                    # The scope body writes into ``k_cache`` via pl.assemble
                    # and never reads it, so the outliner infers ``Out`` for the
                    # synthesised callee's k_cache param. The relaxed Out+NoDep
                    # rule is what lets the post-pass verifier accept the
                    # resulting ``Out`` (callee) + ``NoDep`` (call-site) pair.
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
        # synthesised callee declares it as a write param (because pl.assemble
        # inside the body writes into it).
        assert dirs[k_cache_idx] == ir.ArgDirection.NoDep, (
            f"expected NoDep at k_cache slot {k_cache_idx}, got {dirs}"
        )

        # And the post-pass property verifier must accept the Out+NoDep
        # combination on the synthesised Call.
        props = _core_passes.IRPropertySet()
        props.insert(_core_passes.IRProperty.CallDirectionsResolved)
        _core_passes.PropertyVerifierRegistry.verify_or_throw(props, After)

        # Assert directly that the synthesised callee declares the marked
        # param as a write direction — this is the load-bearing precondition
        # that makes this a write+NoDep test (rather than the trivial In+NoDep
        # case covered by ``test_outline_plus_derive_marks_slot_no_dep``). The
        # body writes ``k_cache`` and never reads it, so the direction is
        # ``Out``; issue #2415 is why it is not ``InOut``.
        outlined = next(f for gv, f in After.functions.items() if gv.name != "main")
        assert outlined.param_directions[k_cache_idx] == ir.ParamDirection.Out, (
            f"expected Out at outlined callee param {k_cache_idx}, got {list(outlined.param_directions)}"
        )

    def test_outline_propagates_split_aiv_attr(self):
        """A pl.split_aiv InCore scope carries split + split_aiv onto the outlined function.

        OutlineIncoreScopes must propagate the manual AIV-split marker
        (``split_aiv``) — not just the ``split`` mode — so the downstream
        SplitVectorKernel bypass can find it on the
        outlined function.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                b: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    offset = aiv_id * 128
                    tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [offset, 0], [128, 128])
                    out = pl.store(pl.add(tile_a, tile_b), [offset, 0], out)
                return out

        # Use BEFORE_AND_AFTER property verification (not the default `roundtrip`
        # level): the python printer does not yet emit the InCoreScopeStmt
        # `split_aiv` marker, so a print->parse roundtrip of a split_aiv program
        # spuriously fails on the scope's attrs. That printer gap is unrelated to
        # the outliner propagation this test exercises.
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            Before = passes.convert_to_ssa()(Before)
            After = passes.outline_incore_scopes()(Before)

        outlined = next(f for gv, f in After.functions.items() if f.func_type == ir.FunctionType.InCore)
        assert outlined.attrs["split"] == pl.SplitMode.UP_DOWN.value
        assert outlined.attrs["split_aiv"] is True

    def test_function_split_with_split_aiv_region_rejected(self):
        """A function-level pl.split (optimizations=[pl.split(mode)]) on a scope
        that ALSO contains pl.split_aiv region(s) is rejected: the two AIV-split
        mechanisms are mutually exclusive, and the function-level split would
        otherwise be silently dropped (the per-region split governs the lanes).

        Detected by the *parser*, the only layer that sees the literal
        ``pl.split(...)`` the user wrote — see
        test_function_split_none_with_split_aiv_region_rejected for why that
        matters, and ..._rejected_at_outline for the pass-level backstop."""
        with pytest.raises(ParserSyntaxError, match="mutually exclusive"):

            @pl.program
            class Before:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="k",
                        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
                    ):
                        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            offset = aiv_id * 128
                            t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                            out = pl.store(t, [offset, 0], out)
                    return out

    def test_function_split_none_with_split_aiv_region_rejected(self):
        """``optimizations=[pl.split(pl.SplitMode.NONE)]`` on a scope holding
        pl.split_aiv region(s) is rejected too (RFC #1820).

        NONE carries no split of its own, but writing it still reads as "auto and
        manual split mixed on one scope". The exemption that once existed was
        only because the cross-core slot count had no carrier other than
        ``pl.split(..., slot_num=N)``; it now has one (see
        test_cross_core_slot_with_split_aiv_region_accepted).

        This spelling is why the check lives in the parser. Since issue #2205
        ``InCoreScopeStmt.split_`` has a single encoding of "no split"
        (``SplitMode.NONE``), so by the time ``OutlineIncoreScopes`` runs a literal
        ``pl.split(pl.SplitMode.NONE)`` is indistinguishable from no ``pl.split``
        at all — the two encodings that used to distinguish them also broke
        print -> parse round-tripping."""
        with pytest.raises(ParserSyntaxError, match="mutually exclusive"):

            @pl.program
            class Before:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(
                    self,
                    a: pl.Tensor[[512, 128], pl.FP32],
                    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                ) -> pl.Tensor[[512, 128], pl.FP32]:
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="k",
                        optimizations=[pl.split(pl.SplitMode.NONE)],
                    ):
                        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                            offset = aiv_id * 128
                            t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                            out = pl.store(t, [offset, 0], out)
                    return out

    def test_function_split_with_split_aiv_region_rejected_at_outline(self):
        """OutlineIncoreScopes is the backstop for IR that bypassed the parser.

        Deserialized ``.pto`` and programmatically built scopes never hit the
        parse-time check, so the pass still rejects any scope whose ``split_`` is
        set (i.e. not ``SplitMode.NONE``) while its body holds a region. Stamping
        the mode on with a mutator reproduces exactly that shape."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="k"):
                    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        offset = aiv_id * 128
                        t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                        out = pl.store(t, [offset, 0], out)
                return out

        class _StampIncoreSplit(ir.IRMutator):
            def visit_in_core_scope_stmt(self, op):
                rewritten = super().visit_in_core_scope_stmt(op)
                assert isinstance(rewritten, ir.InCoreScopeStmt)
                return ir.InCoreScopeStmt(
                    ir.SplitMode.UP_DOWN,
                    rewritten.name_hint,
                    body=rewritten.body,
                    span=rewritten.span,
                )

        # `Before` is valid and round-trippable — the invalid combination is
        # introduced only by the mutator below, after ConvertToSSA has run.
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            ssa = passes.convert_to_ssa()(Before)
            stamped = _StampIncoreSplit().visit_program(ssa)
            with pytest.raises(ValueError, match="mutually exclusive"):
                passes.outline_incore_scopes()(stamped)

    def test_cross_core_slot_with_split_aiv_region_accepted(self):
        """``optimizations=[pl.cross_core_slot(slot_num=N)]`` coexists with
        pl.split_aiv region(s): sizing the cross-core ring is orthogonal to
        partitioning work across the AIV lanes.

        This is the migration path off the rejected
        ``pl.split(pl.SplitMode.NONE, slot_num=N)`` idiom — the outlined function
        carries both the slot count and the region-derived split_aiv marker."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="k",
                    optimizations=[pl.cross_core_slot(slot_num=4)],
                ):
                    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        offset = aiv_id * 128
                        t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                        out = pl.store(t, [offset, 0], out)
                return out

        # BEFORE_AND_AFTER rather than the default roundtrip level: the python
        # printer does not emit the InCoreScopeStmt `split_aiv` marker, so a
        # print->parse roundtrip of a split_aiv program spuriously fails — same
        # caveat as test_outline_propagates_split_aiv_attr.
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            ssa = passes.convert_to_ssa()(Before)
            After = passes.outline_incore_scopes()(ssa)

        outlined = next(f for gv, f in After.functions.items() if f.func_type == ir.FunctionType.InCore)
        assert outlined.attrs["slot_num"] == 4
        assert outlined.attrs["split_aiv"] is True
        # The region's mode is still bridged to a function-level representative.
        assert outlined.attrs["split"] == pl.SplitMode.UP_DOWN.value

    def test_outline_multi_mode_regions_omits_func_split(self):
        """Two sibling pl.split_aiv regions with DIFFERING modes (UP_DOWN +
        LEFT_RIGHT) in one CORE_GROUP scope: the outlined function gets
        split_aiv=True but NO function-level ``split`` mode — there is no single
        representative mode. The authoritative per-region mode rides each
        SplitAivScopeStmt (consumed at LowerAutoVectorSplit, pass 21); downstream
        readers key on the split_aiv marker / per-op split, not a func mode.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                out_ud: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
                out_lr: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="k"):
                    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        off_ud = aiv_id * 64
                        ta: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                        out_ud = pl.store(pl.add(ta, ta), [off_ud, 0], out_ud)
                    for aiv_id2 in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                        off_lr = aiv_id2 * 64
                        tb: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
                        out_lr = pl.store(pl.add(tb, tb), [0, off_lr], out_lr)
                return out_ud

        # BEFORE_AND_AFTER verification only (the python printer does not yet emit
        # the split_aiv marker, so a roundtrip would spuriously fail — see
        # test_outline_propagates_split_aiv_attr).
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            Before = passes.convert_to_ssa()(Before)
            After = passes.outline_incore_scopes()(Before)

        outlined = next(f for gv, f in After.functions.items() if f.func_type == ir.FunctionType.InCore)
        assert outlined.attrs["split_aiv"] is True
        # Differing sibling modes -> no single representative -> func-level mode unset.
        assert "split" not in outlined.attrs

    @staticmethod
    def _strip_outlined_incore_wrapper(program):
        """Author-side adapter for the SplitAiv outlined Expected.

        OutlineIncoreScopes emits the ``SplitAivScopeStmt`` region *directly* in
        the outlined InCore body — the InCore scope it absorbed became the
        function boundary, so no ``InCoreScopeStmt`` wrapper remains. The
        ``pl.split_aiv`` DSL, however, always nests its region inside a
        CORE_GROUP InCore scope, so a DSL-authored Expected carries one extra
        wrapper. Strip that single redundant ``InCoreScopeStmt`` from every
        InCore function so the authored Expected matches the outlined shape.
        """
        new_funcs = []
        for func in program.functions.values():
            body = func.body
            if func.func_type == ir.FunctionType.InCore and isinstance(body, ir.SeqStmts):
                flat = []
                for stmt in body.stmts:
                    if isinstance(stmt, ir.InCoreScopeStmt):
                        inner = stmt.body
                        flat.extend(inner.stmts if isinstance(inner, ir.SeqStmts) else [inner])
                    else:
                        flat.append(stmt)
                func = ir.Function(
                    func.name,
                    list(zip(func.params, func.param_directions)),
                    func.return_types,
                    ir.SeqStmts(flat, func.span),
                    func.span,
                    func.func_type,
                    func.level,
                    func.role,
                    dict(func.attrs),
                )
            new_funcs.append(func)
        return ir.Program(new_funcs, program.name, program.span)

    def test_split_aiv_preserved_in_outlined_func(self):
        """OutlineIncoreScopes outlines the enclosing InCore scope but preserves
        the nested ``SplitAivScopeStmt`` region inside the outlined function body
        (SplitAiv is never an outline target — it is lowered in place at pass 21).

        The Expected pins the outlined two-function form: the region lives in the
        InCore ``main_incore_0`` body and the Orchestration ``main`` only carries
        the synthesised Call (no region), which a structural match subsumes.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                b: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    offset = aiv_id * 128
                    tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [offset, 0], [128, 128])
                    out = pl.store(pl.add(tile_a, tile_b), [offset, 0], out)
                return out

        @pl.program
        class ExpectedRaw:
            @pl.function(
                type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True}
            )
            def main_incore_0(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                b: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    offset: pl.Scalar[pl.INDEX] = aiv_id * 128
                    tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [offset, 0], [128, 128])
                    out_v1: pl.Tensor[[512, 128], pl.FP32] = pl.store(
                        pl.add(tile_a, tile_b), [offset, 0], out
                    )
                    out_store: pl.Tensor[[512, 128], pl.FP32] = out_v1
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                b: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                out2: pl.Tensor[[512, 128], pl.FP32] = self.main_incore_0(a, b, out)
                return out2

        # BEFORE_AND_AFTER verification (not the default `roundtrip` level): the
        # python printer does not yet emit the InCoreScopeStmt `split_aiv` marker,
        # so a print->parse roundtrip of a split_aiv program spuriously fails on
        # the scope's attrs — unrelated to the outlining this test exercises.
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
            Expected = self._strip_outlined_incore_wrapper(passes.convert_to_ssa()(ExpectedRaw))

        ir.assert_structural_equal(After, Expected)

    def test_split_aiv_aiv_id_not_outlined_param(self):
        """``aiv_id`` is bound inside the region body (``tile.get_subblock_idx``),
        so the outliner's def/use analysis must treat it as locally-defined — the
        outlined function signature must gain NO ``aiv_id`` parameter.

        The Expected's ``main_incore_0`` signature is ``(a, b, out)`` with no
        ``aiv_id`` param, so the structural match pins that guarantee; an explicit
        param-name assertion keeps the test's named intent self-evident.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                b: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    offset = aiv_id * 128
                    tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [offset, 0], [128, 128])
                    out = pl.store(pl.add(tile_a, tile_b), [offset, 0], out)
                return out

        @pl.program
        class ExpectedRaw:
            @pl.function(
                type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True}
            )
            def main_incore_0(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                b: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    offset: pl.Scalar[pl.INDEX] = aiv_id * 128
                    tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [offset, 0], [128, 128])
                    out_v1: pl.Tensor[[512, 128], pl.FP32] = pl.store(
                        pl.add(tile_a, tile_b), [offset, 0], out
                    )
                    out_store: pl.Tensor[[512, 128], pl.FP32] = out_v1
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                b: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                out2: pl.Tensor[[512, 128], pl.FP32] = self.main_incore_0(a, b, out)
                return out2

        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
            Expected = self._strip_outlined_incore_wrapper(passes.convert_to_ssa()(ExpectedRaw))

        ir.assert_structural_equal(After, Expected)

        # Focused guard on the test's named purpose: aiv_id stays region-local.
        outlined = next(f for gv, f in After.functions.items() if f.func_type == ir.FunctionType.InCore)
        param_names = [p.name_hint for p in outlined.params]
        assert not any("aiv_id" in name for name in param_names), (
            f"aiv_id must not be promoted to an outlined param, got params {param_names}"
        )

    @staticmethod
    def _outline_incore(program: ir.Program) -> ir.Function:
        """Outline ``program`` and return its single InCore function."""
        with passes.PassContext([]):
            After = passes.outline_incore_scopes()(program)
        return next(f for f in After.functions.values() if f.func_type == ir.FunctionType.InCore)

    def test_split_aiv_none_region_stamps_no_function_level_split(self):
        """A uniform ``mode=NONE`` region set stamps ``split_aiv`` alone.

        "No split" has a single canonical encoding at the function-attr level: an
        absent key. ``Function::GetSplitMode`` maps a stored 0 to ``None`` exactly
        as it does an absent key, so an explicit ``split=SplitMode.NONE`` was
        invisible to every consumer — while the parser dropped it on the way back
        in, making print -> parse lossy (``Kwargs size mismatch``). Asserted as
        exact dict equality: the defect was an EXTRA entry, which a per-key
        ``.get()`` check would not have caught.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    base = aiv_id * 64
                    c = pl.store(pl.exp(pl.load(a, [base, 0], [64, 128])), [base, 0], c)
                return c

        outlined = self._outline_incore(Before)

        assert dict(outlined.attrs) == {"split_aiv": True}
        assert outlined.split is None

    def test_split_aiv_up_down_region_stamps_function_level_split(self):
        """Control for the NONE case: a real mode is still bridged to the function.

        Pins that the NONE exclusion above is narrow — dropping a genuine
        representative mode would silently strip the whole-function marker that
        SplitVectorKernel keys on.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    c = pl.store(pl.exp(pl.load(a, [0, 0], [128, 128])), [0, 0], c)
                return c

        outlined = self._outline_incore(Before)

        assert dict(outlined.attrs) == {"split_aiv": True, "split": ir.SplitMode.UP_DOWN.value}
        assert outlined.split == ir.SplitMode.UP_DOWN


class TestOutlineReboundOutCapture:
    """A captured ``pl.Out`` tensor the scope rebinds under its own name.

    ``c = pl.store(t, off, c)`` is what the parser emits before ConvertToSSA
    splits the definition from the use, so ``c`` is one Var that is both stored
    into and assigned inside the scope. The outlined function must capture it as
    a write parameter and bind the store result to a *distinct* Var, so the body
    neither leaves ``c`` free nor rebinds its own parameter. The store target is
    the only use, so the direction is ``Out`` (issue #2415) — and the same
    program run through ConvertToSSA first must agree, which
    ``test_matches_the_convert_to_ssa_prefixed_result`` pins.

    These run the pass directly on parser output (no ConvertToSSA) — that is
    precisely the input shape the flow-insensitive ``var_uses \\ var_defs``
    partition mishandled.
    """

    @staticmethod
    def _outlined(program: ir.Program) -> ir.Function:
        incore = [f for f in program.functions.values() if f.func_type == ir.FunctionType.InCore]
        assert len(incore) == 1, f"expected exactly one outlined InCore function, got {len(incore)}"
        return incore[0]

    @staticmethod
    def _stmts(func: ir.Function) -> list[ir.Stmt]:
        body = func.body
        assert isinstance(body, ir.SeqStmts), f"expected a SeqStmts body, got {type(body).__name__}"
        return list(body.stmts)

    def test_rebound_out_capture_becomes_out_param(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    c = pl.store(pl.exp(pl.load(a, [0, 0], [128, 128])), [0, 0], c)
                return c

        with passes.PassContext([]):
            After = passes.outline_incore_scopes()(Before)

        outlined = self._outlined(After)
        assert [p.name_hint for p in outlined.params] == ["a", "c"]
        assert outlined.param_directions[1] == ir.ParamDirection.Out

        # The captured tensor is threaded through the call site, not dropped.
        main = After.get_function("main")
        assert main is not None
        calls = [
            s.value
            for s in self._stmts(main)
            if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Call)
        ]
        assert len(calls) == 1, f"expected one outlined-kernel call in main, got {len(calls)}"
        arg_names = [arg.name_hint for arg in calls[0].args if isinstance(arg, ir.Var)]
        assert arg_names == ["a", "c"]

    def test_rebound_out_capture_output_round_trips(self):
        """The whole point: the outlined program survives print -> parse."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    c = pl.store(pl.exp(pl.load(a, [0, 0], [128, 128])), [0, 0], c)
                return c

        with passes.PassContext([]):
            After = passes.outline_incore_scopes()(Before)

        reparsed = pl.parse_program(ir.python_print(After))
        ir.assert_structural_equal(reparsed, After)

    def test_rebound_out_capture_in_split_aiv_region_reparses(self):
        """The originally-reported shape: the region form used to leave ``c`` free.

        Asserts parseability only, not structural equality — but no longer over the
        ``split=pl.SplitMode.NONE`` attr, which the outliner has stopped stamping.
        What remains is the surviving region node itself: the parser wraps a
        top-level ``pl.split_aiv`` in an ``InCoreScopeStmt`` whenever an InCore
        scope is open, so re-parsing this *outlined* function (which the pass
        declines to re-enter, being already InCore) re-introduces a
        ``with pl.at(level=pl.Level.CORE_GROUP):`` the original does not have. That
        is the separate ``pl.split_aiv``-inside-InCore gap, unrelated to the capture
        fix. Once the region is erased the output does round-trip structurally —
        asserted by ``test_lower_auto_vector_split.py::
        test_outlined_region_still_lowers_and_stamps``.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    base = aiv_id * 64
                    c = pl.store(pl.exp(pl.load(a, [base, 0], [64, 128])), [base, 0], c)
                return c

        with passes.PassContext([]):
            After = passes.outline_incore_scopes()(Before)

        outlined = self._outlined(After)
        assert [p.name_hint for p in outlined.params] == ["a", "c"]
        pl.parse_program(ir.python_print(After))

    def test_rebound_out_capture_body_is_single_assignment(self):
        """Each store binds its own result Var; the parameter is never rebound."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    c = pl.store(pl.exp(pl.load(a, [0, 0], [64, 128])), [0, 0], c)
                    c = pl.store(pl.exp(pl.load(a, [64, 0], [64, 128])), [64, 0], c)
                return c

        with passes.PassContext([]):
            After = passes.outline_incore_scopes()(Before)

        outlined = self._outlined(After)
        param_c = outlined.params[1]
        assigned = [s.var for s in self._stmts(outlined) if isinstance(s, ir.AssignStmt)]
        assert len(assigned) == 2, f"expected two store results, got {len(assigned)}"
        assert not any(v.same_as(param_c) for v in assigned), "must not rebind the InOut parameter"
        assert not assigned[0].same_as(assigned[1]), "repeated stores need distinct result Vars"

        # ...and the unchanged SSAForm verifier accepts the result.
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE)]):
            passes.outline_cluster_scopes()(After)

    def test_matches_the_convert_to_ssa_prefixed_result(self):
        """Outlining parser output agrees with outlining its SSA form.

        The pass declares ``IRProperty::SSAForm`` as a precondition; this pins
        that honouring the rebound capture produces the same signature shape the
        SSA path already produced, rather than a second, divergent lowering.
        """

        def build():
            @pl.program
            class Prog:
                @pl.function
                def main(
                    self,
                    a: pl.Tensor[[128, 128], pl.FP32],
                    c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
                ) -> pl.Tensor[[128, 128], pl.FP32]:
                    with pl.at(level=pl.Level.CORE_GROUP):
                        c = pl.store(pl.exp(pl.load(a, [0, 0], [128, 128])), [0, 0], c)
                    return c

            return Prog

        with passes.PassContext([]):
            direct = self._outlined(passes.outline_incore_scopes()(build()))
            via_ssa = self._outlined(passes.outline_incore_scopes()(passes.convert_to_ssa()(build())))

        # ConvertToSSA versions the names (``c`` -> ``c__ssa_v0``), so compare
        # the base names; the signature itself must otherwise be identical.
        def base_names(func: ir.Function) -> list[str]:
            return [p.name_hint.split("__ssa_v")[0] for p in func.params]

        assert base_names(direct) == base_names(via_ssa) == ["a", "c"]
        assert list(direct.param_directions) == list(via_ssa.param_directions)


class TestPromotionFoldsParamDimReads:
    """Promoting Opaque -> Orchestration folds ``tensor.dim`` on a param dyn-dim axis.

    A tensor's declared extent *is* its runtime extent, so reading it back mints a
    second scalar for one quantity. The DSL parser folds that read onto the symbol
    the signature already names, but only in an Orchestration body. A body written
    as Opaque keeps the read, so the promotion here must fold it — otherwise the IR
    this pass emits no longer parses back to itself.
    """

    M_DYN = pl.dynamic("M_DYN")

    @staticmethod
    def _main_stmts(program: ir.Program) -> list[ir.Stmt]:
        main = program.get_function("main")
        assert main is not None
        body = main.body
        assert isinstance(body, ir.SeqStmts), f"expected a SeqStmts body, got {type(body).__name__}"
        return list(body.stmts)

    @staticmethod
    def _dim_reads(stmts: list[ir.Stmt]) -> list[ir.Stmt]:
        dim = ir.get_op("tensor.dim").name
        return [
            s
            for s in stmts
            if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Call) and s.value.op.name == dim
        ]

    def test_promotion_matches_the_parser_folded_form(self):
        """The promoted body equals what the parser produces for the same source.

        ``Expected`` is the identical program declared Orchestration, where the
        parser folds ``m`` at parse time. Both go through the same passes, so any
        divergence is the promotion failing to establish the same normal form.
        """
        M_DYN = self.M_DYN

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                a: pl.Tensor[[M_DYN, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[M_DYN, 128], pl.FP32]],
            ) -> pl.Tensor[[M_DYN, 128], pl.FP32]:
                m = pl.tensor.dim(a, 0)
                with pl.spmd(m // 16):
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[16, 128], pl.FP32] = pl.load(a, [i * 16, 0], [16, 128])
                    out = pl.store(pl.add(t, t), [i * 16, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[M_DYN, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[M_DYN, 128], pl.FP32]],
            ) -> pl.Tensor[[M_DYN, 128], pl.FP32]:
                m = pl.tensor.dim(a, 0)
                with pl.spmd(m // 16):
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[16, 128], pl.FP32] = pl.load(a, [i * 16, 0], [16, 128])
                    out = pl.store(pl.add(t, t), [i * 16, 0], out)
                return out

        with passes.PassContext([]):
            After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
            ExpectedAfter = passes.outline_incore_scopes()(passes.convert_to_ssa()(Expected))

        assert self._dim_reads(self._main_stmts(After)) == []
        ir.assert_structural_equal(After, ExpectedAfter)

    def test_promotion_folds_transitive_dim_read(self):
        """A read exposed *by* an earlier fold folds too, in the same pass.

        Folding ``m`` retypes the local ``tmp`` from ``[m, 128]`` to ``[M_DYN, 128]``,
        which makes ``n = tensor.dim(tmp, 0)`` foldable in turn. Missing that second
        read leaves the printed IR parsing back to one statement fewer -- exactly the
        roundtrip mismatch this fold exists to remove.
        """
        M_DYN = self.M_DYN

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                a: pl.Tensor[[M_DYN, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[M_DYN, 128], pl.FP32]],
            ) -> pl.Tensor[[M_DYN, 128], pl.FP32]:
                m = pl.tensor.dim(a, 0)
                tmp: pl.Tensor[[M_DYN, 128], pl.FP32] = pl.create_tensor([m, 128], dtype=pl.FP32)
                n = pl.tensor.dim(tmp, 0)
                with pl.spmd(n // 16):
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[16, 128], pl.FP32] = pl.load(a, [i * 16, 0], [16, 128])
                    out = pl.store(pl.add(t, t), [i * 16, 0], out)
                return out

        with passes.PassContext([]):
            After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

        assert self._dim_reads(self._main_stmts(After)) == []

        # The invariant itself: the emitted IR parses back to itself.
        printed = python_print(After, format=False)
        ir.assert_structural_equal(After, text_parse(printed, filename="<roundtrip>"))

    def test_promotion_folds_symbol_used_inside_incore_body(self):
        """A read consumed *inside* the scope folds too, and is captured as the symbol."""
        M_DYN = self.M_DYN

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                a: pl.Tensor[[M_DYN, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[M_DYN, 128], pl.FP32]],
            ) -> pl.Tensor[[M_DYN, 128], pl.FP32]:
                m = pl.tensor.dim(a, 0)
                with pl.spmd(m // 16):
                    i = pl.tile.get_block_idx()
                    lim = m - 16
                    t: pl.Tile[[16, 128], pl.FP32] = pl.load(a, [pl.min(i * 16, lim), 0], [16, 128])
                    out = pl.store(pl.add(t, t), [i * 16, 0], out)
                return out

        with passes.PassContext([]):
            After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

        assert self._dim_reads(self._main_stmts(After)) == []

        # The extent crosses into the outlined kernel as a scalar parameter, and
        # the caller passes the symbol itself rather than a copy of it.
        incore = [f for f in After.functions.values() if f.func_type == ir.FunctionType.InCore]
        assert len(incore) == 1, f"expected one outlined InCore function, got {len(incore)}"
        assert isinstance(incore[0].params[0].type, ir.ScalarType)
        assert incore[0].params[0].name_hint == "M_DYN"

    def test_static_extent_dim_read_is_kept(self):
        """A statically-shaped axis has no symbol to fold onto — the read must stay."""
        M_DYN = self.M_DYN

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                a: pl.Tensor[[M_DYN, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[M_DYN, 128], pl.FP32]],
            ) -> pl.Tensor[[M_DYN, 128], pl.FP32]:
                cols = pl.tensor.dim(a, 1)  # 128 — a constant extent, not a symbol
                with pl.spmd(cols // 16):
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[16, 128], pl.FP32] = pl.load(a, [i * 16, 0], [16, 128])
                    out = pl.store(pl.add(t, t), [i * 16, 0], out)
                return out

        with passes.PassContext([]):
            After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

        assert len(self._dim_reads(self._main_stmts(After))) == 1

    def test_opaque_without_incore_scope_keeps_dim_read(self):
        """No InCore scope means no promotion — an Opaque body keeps its read."""
        M_DYN = self.M_DYN

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(self, a: pl.Tensor[[M_DYN, 128], pl.FP32]) -> pl.Scalar[pl.INDEX]:
                m = pl.tensor.dim(a, 0)
                return m

        with passes.PassContext([]):
            After = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))

        main = After.get_function("main")
        assert main is not None
        assert main.func_type == ir.FunctionType.Opaque
        assert len(self._dim_reads(self._main_stmts(After))) == 1


class TestGraphFunctionTypeIsPreserved:
    """A Graph body is scanned for InCore scopes without losing its identity.

    The promotion that turns an Opaque body into Orchestration once it has been
    outlined is an unconditional overwrite of ``func_type_``. That was safe only
    while the pass admitted exactly Opaque and Orchestration; admitting Graph
    without guarding the promotion would erase the marker silently, leaving a
    function indistinguishable from a plain Orchestration entry downstream.
    """

    @staticmethod
    def _outline(program):
        with passes.PassContext([]):
            return passes.outline_incore_scopes()(passes.convert_to_ssa()(program))

    def test_graph_with_incore_scope_is_outlined_and_stays_graph(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        After = self._outline(Before)

        layer = After.get_function("layer")
        assert layer is not None
        assert layer.func_type == ir.FunctionType.Graph, "outlining must not overwrite the Graph marker"
        # The scope really was outlined, so this does not pass by being skipped.
        assert After.get_function("layer_incore_0") is not None

    def test_graph_without_incore_scope_stays_graph(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Graph)
            def layer(self, x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
                return x

        layer = self._outline(Before).get_function("layer")
        assert layer is not None
        assert layer.func_type == ir.FunctionType.Graph

    def test_opaque_promotion_still_works(self):
        """Guarding the promotion must not disable it for the Opaque case."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        main = self._outline(Before).get_function("main")
        assert main is not None
        assert main.func_type == ir.FunctionType.Orchestration


class TestOutlineCachePolicy:
    """``pl.set_cache_policy(t, policy)`` lowering: ``ScopeStmt.attrs['cache_policy_vars']``
    holds the captured Vars the declaration names, and the outliner translates
    them into positional indices into the outlined function's params, re-emitted
    as the function attr ``cache_policy``.

    The Var form is *consumed* here — it must not survive onto the synthesised
    call or onto either function. Param indices are a carrier with a deliberately
    short life: they stay valid only until ConvertTensorToTileOps (pass 10) turns
    them into per-``tile.load`` ``cache`` kwargs, because passes after that both
    append to param lists (InjectGMPipeBuffer, MaterializeDistTensorCtx) and
    prepend onto them (MaterializeValidShapeSymbols).

    The two rejections below are author errors rather than compiler bugs — a
    declaration naming a tensor the scope never reads, and a bypassing read of
    bytes the same kernel writes — so they surface as ``ValueError`` from
    ``CHECK_SPAN``, not as an internal error.
    """

    @staticmethod
    def _outline(program: ir.Program) -> ir.Program:
        return passes.outline_incore_scopes()(passes.convert_to_ssa()(program))

    @staticmethod
    def _kernel(program: ir.Program, name: str) -> ir.Function:
        func = program.get_function(name)
        assert func is not None, f"outlined kernel '{name}' not found"
        return func

    def test_declaration_becomes_a_param_index_on_the_outlined_kernel(self):
        """The declared Var is resolved to its slot in the outlined signature."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 128], pl.FP32],
                b: pl.Tensor[[128, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 256], pl.FP32]],
            ) -> pl.Tensor[[256, 256], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
                    pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
                    c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                    out = pl.assemble(out, c, [0, 0])
                return out

        mm = self._kernel(self._outline(Before), "mm")
        # Capture order is a, b, out — so b's declaration lands at index 1,
        # paired with CachePolicy.BYPASS's underlying int (the form the
        # ``tile.load`` ``cache`` kwarg is registered with).
        assert dict(mm.attrs)["cache_policy"] == [(1, int(pl.CachePolicy.BYPASS))]
        assert mm.params[1].name_hint.startswith("b"), "index 1 must be the declared tensor's slot"

    def test_scope_attr_is_consumed_by_outlining(self):
        """The Var-keyed scope attr is translated, never propagated."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 128], pl.FP32],
                b: pl.Tensor[[128, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 256], pl.FP32]],
            ) -> pl.Tensor[[256, 256], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
                    pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
                    c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                    out = pl.assemble(out, c, [0, 0])
                return out

        After = self._outline(Before)

        # Not on the synthesised dispatch: the declaration is a property of the
        # callee's parameters, and nothing at the call site consumes it.
        call = TestOutlineNoDepArgs._outlined_user_call(After)
        assert "cache_policy_vars" not in call.attrs
        assert "cache_policy" not in call.attrs
        # Not on the orchestrator either — only the outlined kernel carries it.
        main = self._kernel(After, "main")
        assert "cache_policy_vars" not in dict(main.attrs)
        assert "cache_policy" not in dict(main.attrs)
        # And the Var form is gone from the kernel: indices replaced it.
        assert "cache_policy_vars" not in dict(self._kernel(After, "mm").attrs)

    def test_declarations_are_recorded_in_param_index_order(self):
        """Declaration order is irrelevant; the attr is sorted by param index.

        The declaration set is order-independent, so two spellings that differ
        only in the order the author wrote them must produce structurally equal
        IR (and identical dumps).
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 128], pl.FP32],
                b: pl.Tensor[[128, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 256], pl.FP32]],
            ) -> pl.Tensor[[256, 256], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
                    # Written b-then-a; a is captured first, so the attr must
                    # come back a-then-b.
                    pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
                    pl.set_cache_policy(a, pl.CachePolicy.BYPASS)
                    c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                    out = pl.assemble(out, c, [0, 0])
                return out

        mm = self._kernel(self._outline(Before), "mm")
        bypass = int(pl.CachePolicy.BYPASS)
        assert dict(mm.attrs)["cache_policy"] == [(0, bypass), (1, bypass)]

    def test_default_policy_is_allowed_on_a_written_tensor(self):
        """Only BYPASS carries the no-concurrent-write promise, so only BYPASS
        is restricted to ``In`` params. DEFAULT states nothing and is legal
        anywhere the tensor is captured."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 128], pl.FP32],
                b: pl.Tensor[[128, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 256], pl.FP32]],
            ) -> pl.Tensor[[256, 256], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
                    pl.set_cache_policy(out, pl.CachePolicy.DEFAULT)
                    c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                    out = pl.assemble(out, c, [0, 0])
                return out

        mm = self._kernel(self._outline(Before), "mm")
        assert dict(mm.attrs)["cache_policy"] == [(2, int(pl.CachePolicy.DEFAULT))]

    def test_declaration_on_an_uncaptured_tensor_is_rejected(self):
        """A tensor the scope body never reads has no parameter to resolve to."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                w: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    pl.set_cache_policy(w, pl.CachePolicy.BYPASS)
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        with pytest.raises(ValueError, match="not captured by the scope body"):
            self._outline(Before)

    def test_bypass_on_an_out_param_is_rejected(self):
        """BYPASS promises nothing writes those bytes while the kernel runs; a
        tensor the scope itself writes breaks that promise by construction."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 128], pl.FP32],
                b: pl.Tensor[[128, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 256], pl.FP32]],
            ) -> pl.Tensor[[256, 256], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
                    pl.set_cache_policy(out, pl.CachePolicy.BYPASS)
                    c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                    out = pl.assemble(out, c, [0, 0])
                return out

        with pytest.raises(ValueError, match=r"not allowed on a tensor this scope writes \(Out\)"):
            self._outline(Before)

    def test_bypass_on_an_inout_param_is_rejected(self):
        """Same rejection for a captured tensor that is both read and updated."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                k: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kk"):
                    pl.set_cache_policy(k, pl.CachePolicy.BYPASS)
                    t: pl.Tensor[[64, 64], pl.FP32] = pl.add(k, x)
                    k = pl.assemble(k, t, [0, 0])
                return k

        with pytest.raises(ValueError, match=r"not allowed on a tensor this scope writes \(InOut\)"):
            self._outline(Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
