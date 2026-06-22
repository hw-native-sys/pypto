# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for OutlineClusterScopes pass."""

import pypto.language as pl
import pytest

from pypto import ir, passes


class TestOutlineClusterScopes:
    """Test OutlineClusterScopes pass."""

    def test_outline_simple_cluster_scope(self):
        """Test outlining a simple Cluster scope into a Group function."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Group)
            def main_cluster_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_cluster_0(x)
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_cluster_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_cluster_with_incore(self):
        """Test outlining a Cluster scope containing an InCore scope."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster():
                    with pl.at(level=pl.Level.CORE_GROUP):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Group)
            def main_cluster_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_cluster_0(x)
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_cluster_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_multiple_cluster_scopes(self):
        """Test outlining multiple Cluster scopes in one function."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.cluster():
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Group)
            def main_cluster_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Group)
            def main_cluster_1(self, y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_cluster_0(x)
                z: pl.Tensor[[64], pl.FP32] = self.main_cluster_1(y)
                return z

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_cluster_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_no_cluster_scopes_passthrough(self):
        """Test that functions without Cluster scopes are passed through unchanged."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_cluster_scopes()(Before)
        ir.assert_structural_equal(After, Before)

    def test_cluster_does_not_affect_incore_scopes(self):
        """Test that OutlineClusterScopes does not outline InCore scopes."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_cluster_scopes()(Before)
        # InCore scopes should remain untouched by the cluster pass
        ir.assert_structural_equal(After, Before)

    def test_outline_standalone_spmd_scope(self):
        """Standalone pl.spmd() should outline to a Spmd wrapper, not a Group."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.add(x_tile, x_tile)
                out: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.spmd(4, sync_start=True):
                    out = self.kernel(x, out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.add(x_tile, x_tile)
                out: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out)
                return out

            @pl.function(type=pl.FunctionType.Spmd, attrs={"core_num": 4, "sync_start": True})
            def main_spmd_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                # Wrapper returns its own `out` param (the kernel writes through
                # the Out arg), not the post-call SSA result var.
                out_call = self.kernel(x, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out = self.main_spmd_0(x, out)
                return out

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_cluster_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_cluster_in_orchestration_function(self):
        """Test that Cluster scopes in Orchestration functions are also outlined."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster():
                    y: pl.Tensor[[64], pl.FP32] = self.compute(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Group)
            def main_cluster_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.compute(x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_cluster_0(x)
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_cluster_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_cluster_scope_no_outputs(self):
        """Test outlining a Cluster scope with no variables used after."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster():
                    _y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return x

        @pl.program
        class Expected:
            # _y is not used after the cluster scope, so the outlined Group
            # function has no return types.
            @pl.function(type=pl.FunctionType.Group)
            def main_cluster_0(self, x: pl.Tensor[[64], pl.FP32]):
                _y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                self.main_cluster_0(x)
                return x

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_cluster_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_cluster_round_trip(self):
        """Test that cluster scope survives print -> parse round-trip."""

        @pl.program
        class Original:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        printed = Original.as_python()
        Reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Original, Reparsed)

    def test_cluster_multiple_outputs(self):
        """Test outlining a Cluster scope with multiple output variables."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                w: pl.Tensor[[64], pl.FP32] = pl.add(y, z)
                return w

        @pl.program
        class Expected:
            # y and z are both used after the cluster scope, so the outlined
            # Group function returns a 2-tuple.
            @pl.function(type=pl.FunctionType.Group)
            def main_cluster_0(
                self, x: pl.Tensor[[64], pl.FP32]
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                z: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return y, z

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y, z = self.main_cluster_0(x)
                w: pl.Tensor[[64], pl.FP32] = pl.add(y, z)
                return w

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_cluster_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_spmd_for_loop_auto_outlines_incore(self):
        """`for i in pl.spmd(N)` auto-outlines into Spmd + InCore.

        The new loop form binds the iteration variable to
        ``pl.tile.get_block_idx()`` inside an implicit InCoreScopeStmt,
        giving inline tile ops direct access to the block index without a
        separate ``@pl.function(type=InCore)`` declaration.
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
                    tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(tile_a, [offset, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                i = pl.tile.get_block_idx()
                offset = i * 128
                tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                out_store = pl.store(tile_a, [offset, 0], out)
                # The outline pass keeps the post-store SSA alias but returns
                # the InOut param itself (param-identity returns, #1702).
                out_final: pl.Tensor[[512, 128], pl.FP32] = out_store
                return out

            @pl.function(type=pl.FunctionType.Spmd, attrs={"core_num": 4})
            def main_spmd_0(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                out_call = self.main_incore_0(a, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                out = self.main_spmd_0(a, out)
                return out

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        After = passes.outline_cluster_scopes()(After)
        ir.assert_structural_equal(After, Expected)

    def test_outline_spmd_for_loop_uses_name_hint_on_incore(self):
        """``for ... in pl.spmd(..., name_hint=...)`` names the outlined InCore kernel."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4, name_hint="q_proj_spmd"):
                    offset = i * 128
                    tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(tile_a, [offset, 0], out)
                return out

        @pl.program
        class Expected:
            # name_hint="q_proj_spmd" names the outlined Spmd wrapper, and the
            # InCore kernel drops the "_spmd" suffix -> "q_proj" (instead of the
            # default "main_incore_0" / "main_spmd_0").
            @pl.function(type=pl.FunctionType.InCore)
            def q_proj(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                i = pl.tile.get_block_idx()
                offset = i * 128
                tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                out_store = pl.store(tile_a, [offset, 0], out)
                # The outline pass keeps the post-store SSA alias but returns
                # the InOut param itself (param-identity returns, #1702).
                out_final: pl.Tensor[[512, 128], pl.FP32] = out_store
                return out

            @pl.function(type=pl.FunctionType.Spmd, attrs={"core_num": 4})
            def q_proj_spmd(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                out_call = self.q_proj(a, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                out = self.q_proj_spmd(a, out)
                return out

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        After = passes.outline_cluster_scopes()(After)
        ir.assert_structural_equal(After, Expected)

    def test_outline_spmd_for_loop_marks_assemble_dest_as_inout(self):
        """`for n0 in pl.spmd(N): out = pl.assemble(out, slice, [n0, ...])`
        must make `out` an InOut parameter on both the outlined InCore and
        Spmd wrapper. Without this, the orchestration codegen later drops the
        SSA-result alias for the inout call and emits a use of an undeclared
        ``out__ssa_vN`` C++ identifier.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for n0 in pl.spmd(4):
                    offset = n0 * 128
                    chunk: pl.Tensor[[128, 128], pl.FP32] = pl.slice(a, [128, 128], [offset, 0])
                    out = pl.assemble(out, chunk, [offset, 0])
                return out

        @pl.program
        class Expected:
            # `out` is the assemble destination, so it becomes InOut on both
            # the outlined InCore kernel and the Spmd wrapper; `a` stays In.
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                n0 = pl.tile.get_block_idx()
                offset = n0 * 128
                chunk: pl.Tensor[[128, 128], pl.FP32] = pl.slice(a, [128, 128], [offset, 0])
                out_asm = pl.assemble(out, chunk, [offset, 0])
                return out

            @pl.function(type=pl.FunctionType.Spmd, attrs={"core_num": 4})
            def main_spmd_0(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.InOut[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                out_call = self.main_incore_0(a, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                out = self.main_spmd_0(a, out)
                return out

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        After = passes.outline_cluster_scopes()(After)
        ir.assert_structural_equal(After, Expected)

    def test_nested_spmd_in_cluster_propagates_sync_start(self):
        """`with pl.cluster(): with pl.spmd(N, sync_start=True): ...` keeps a
        single Group function and moves ``core_num``/``sync_start`` onto the
        Group's attrs.

        Semantics (outline_cluster_scopes_pass.cpp:41-70, UnwrapNestedSpmd):
        the Cluster scope is outlined into a Group function whose body still
        contains the nested ``SpmdScopeStmt``. The post-outline UnwrapNestedSpmd
        sweep then (a) copies ``core_num`` and ``sync_start`` from the Spmd
        scope onto the Group function's attrs and (b) replaces the
        ``ScopeStmt(Spmd)`` with its body (the single kernel call). No separate
        Spmd wrapper is emitted — the doc's "Unwrap Nested Spmd in Group" step.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.add(x_tile, x_tile)
                out: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster():
                    with pl.spmd(4, sync_start=True):
                        out = self.kernel(x, out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.add(x_tile, x_tile)
                out: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out)
                return out

            # Single Group function (NOT a Spmd wrapper): the nested Spmd scope
            # was unwrapped, so core_num/sync_start ride on the Group's attrs.
            @pl.function(type=pl.FunctionType.Group, attrs={"core_num": 4, "sync_start": True})
            def main_cluster_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                # Group wrapper returns its own `out` param (the kernel writes
                # through the Out arg), not the post-call SSA result var.
                out_call = self.kernel(x, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out = self.main_cluster_0(x, out)
                return out

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_cluster_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_nested_spmd_in_cluster_omits_sync_start_when_false(self):
        """Without ``sync_start=True`` the unwrapped Group carries only
        ``core_num`` — the ``sync_start`` attr is not added.

        Pins the conditional in UnwrapNestedSpmd
        (outline_cluster_scopes_pass.cpp:66-68): ``sync_start`` is appended to
        the Group attrs only when present AND true. With the default
        ``sync_start=False`` the attr must be absent, not ``False``.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.add(x_tile, x_tile)
                out: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster():
                    with pl.spmd(8):
                        out = self.kernel(x, out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.add(x_tile, x_tile)
                out: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out)
                return out

            # core_num only — no sync_start attr (default False is dropped).
            @pl.function(type=pl.FunctionType.Group, attrs={"core_num": 8})
            def main_cluster_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                # Group wrapper returns its own `out` param (the kernel writes
                # through the Out arg), not the post-call SSA result var.
                out_call = self.kernel(x, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out = self.main_cluster_0(x, out)
                return out

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_cluster_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_cluster_store_target_becomes_inout(self):
        """A Cluster scope that writes an external tensor via ``pl.store`` marks
        that tensor as an ``InOut`` parameter on the outlined Group function.

        Semantics: ScopeOutliner treats tile.store targets as side-effect
        outputs (scope_outline_utils.h:560-571) and InferParamDirections Step 1
        upgrades the corresponding param to ``InOut``
        (scope_outline_utils.h:1118-1123). The store result is captured into a
        fresh ``_store`` SSA var, but the Group returns the InOut param itself
        (param-identity returns, #1702); the call site re-binds the external
        tensor to that return value (``buf`` -> ``buf2``). Mirrors the
        InCore-pass store-only case, but the parent here is NOT promoted — the
        cluster pass leaves the caller's function type unchanged.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
                buf: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                with pl.cluster():
                    tile = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                    pl.store(tile, [0, 0], buf)
                result: pl.Tensor[[16, 128], pl.FP32] = pl.add(buf, x)
                return result

        @pl.program
        class Expected:
            # buf is a tile.store target, so it becomes InOut on the Group; the
            # store result lands in a fresh buf_store var, but the Group
            # returns the InOut param itself (param-identity returns, #1702).
            @pl.function(type=pl.FunctionType.Group)
            def main_cluster_0(
                self, buf: pl.InOut[pl.Tensor[[16, 128], pl.FP32]]
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                tile = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                buf_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(tile, [0, 0], buf)
                return buf

            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
                buf: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                buf2: pl.Tensor[[16, 128], pl.FP32] = self.main_cluster_0(buf)
                result: pl.Tensor[[16, 128], pl.FP32] = pl.add(buf2, x)
                return result

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_cluster_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_cluster_duplicate_name_hint_dedups(self):
        """Two Cluster scopes sharing a ``name_hint`` deduplicate the second
        outlined function name with a numeric suffix.

        Semantics (scope_outline_utils.h:464-472): the first scope takes the
        verbatim hint ``grp`` and inserts it into ``known_names_``; the second
        scope's identical hint collides, so the outliner appends ``_0`` ->
        ``grp_0``. This keeps program function names unique even when the user
        reuses a hint across sibling scopes.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster(name_hint="grp"):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.cluster(name_hint="grp"):
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Group)
            def grp(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            # Second "grp" collides -> deduplicated to "grp_0".
            @pl.function(type=pl.FunctionType.Group)
            def grp_0(self, y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.grp(x)
                z: pl.Tensor[[64], pl.FP32] = self.grp_0(y)
                return z

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_cluster_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_cluster_outlined_verifier_rejects_cluster_in_incore(self):
        """Test that ClusterOutlined verifier flags Cluster scopes in InCore functions."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.ClusterOutlined)

        with pytest.raises(Exception, match="Verification failed"):
            passes.verify_properties(props, Program, "OutlineClusterScopes")


class TestOutlineSpmdScopeTaskId:
    """``with pl.spmd(...) as tid:`` outlines to an ``ir.Submit`` (grid dispatch + producer TaskId).

    The captured-TaskId Spmd scope rides the same ``kAttrTaskIdVar`` rail as
    ``pl.at(...) as tid:``: ``OutlineIncoreScopes`` outlines the inner InCore body
    into a kernel (preserving the outer scope's attrs), then
    ``OutlineClusterScopes``' Spmd outliner — seeing ``kAttrTaskIdVar`` — emits an
    ``ir.Submit`` whose return type ends in ``Scalar[TASK_ID]`` instead of a plain
    Call. ``core_num`` rides on the outlined Spmd ``Function`` attrs, so the
    Submit's own ``core_num`` is ``None`` (codegen reads it via the launch-function
    fallback). Explicit ``deps=[tid]`` fold into the consumer Submit's ``deps``.
    """

    @staticmethod
    def _run(prog):
        prog = passes.convert_to_ssa()(prog)
        prog = passes.outline_incore_scopes()(prog)
        prog = passes.outline_cluster_scopes()(prog)
        return prog

    @staticmethod
    def _submit_values(func):
        """All ``ir.Submit`` exprs bound by an AssignStmt in ``func`` (in body order)."""
        found = []

        def walk(n):
            if isinstance(n, ir.SeqStmts):
                for s in n.stmts:
                    walk(s)
            elif isinstance(n, ir.AssignStmt):
                if isinstance(n.value, ir.Submit):
                    found.append(n.value)
            elif hasattr(n, "body") and n.body is not None:
                walk(n.body)

        walk(func.body)
        return found

    @staticmethod
    def _funcs_by_type(prog, ftype):
        return [f for f in prog.functions.values() if f.func_type == ftype]

    def test_as_tid_outlines_to_submit(self):
        """A captured Spmd dispatch lowers to a deps-free ``ir.Submit`` (not a plain Call)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="stage1") as tid:
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(pl.add(t, t), [i * 128, 0], out)
                return out

        After = self._run(Before)

        # A Spmd wrapper function was synthesised carrying the launch spec.
        spmd_fns = self._funcs_by_type(After, ir.FunctionType.Spmd)
        assert len(spmd_fns) == 1
        assert "core_num" in spmd_fns[0].attrs

        # The orchestration entry lowers the dispatch to exactly one Submit.
        orch = self._funcs_by_type(After, ir.FunctionType.Orchestration)[0]
        submits = self._submit_values(orch)
        assert len(submits) == 1
        submit = submits[0]
        # core_num rides on the Spmd Function attrs, NOT on the Submit.
        assert submit.core_num is None
        # No explicit deps on a lone captured dispatch.
        assert len(submit.deps) == 0

    def test_as_tid_deps_fold_into_submit_deps(self):
        """``deps=[tid0]`` on a second captured Spmd folds into that Submit's ``deps``."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="stage1") as tid0:
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(t, [i * 128, 0], out)
                with pl.spmd(4, name_hint="stage2", deps=[tid0]) as tid1:
                    j = pl.tile.get_block_idx()
                    u: pl.Tile[[128, 128], pl.FP32] = pl.load(out, [j * 128, 0], [128, 128])
                    out = pl.store(pl.add(u, u), [j * 128, 0], out)
                return out

        After = self._run(Before)
        orch = self._funcs_by_type(After, ir.FunctionType.Orchestration)[0]
        submits = self._submit_values(orch)
        assert len(submits) == 2
        # First dispatch has no explicit deps; second carries the first's producer TaskId.
        assert len(submits[0].deps) == 0
        assert len(submits[1].deps) == 1
        assert isinstance(submits[1].deps[0], ir.Var)
        # Two distinct Spmd wrapper functions were synthesised.
        assert len(self._funcs_by_type(After, ir.FunctionType.Spmd)) == 2

    def test_allow_early_resolve_threads_onto_submit(self):
        """``pl.spmd(..., allow_early_resolve=True) as tid`` sets the Submit's flag."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="stage1", allow_early_resolve=True) as tid:
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(pl.add(t, t), [i * 128, 0], out)
                return out

        After = self._run(Before)
        orch = self._funcs_by_type(After, ir.FunctionType.Orchestration)[0]
        (submit,) = self._submit_values(orch)
        assert submit.allow_early_resolve is True

    def test_allow_early_resolve_without_as_tid_still_submits(self):
        """A plain ``with pl.spmd(..., allow_early_resolve=True):`` forces an ``ir.Submit``.

        Without ``as tid`` the dispatch would normally lower to a plain Call; the
        early-dispatch hint forces the Submit shape (the outliner synthesises an
        unused producer TaskId) so the flag can ride to codegen.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                t = pl.load(a, [0, 0], [512, 128])
                out = pl.store(t, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, allow_early_resolve=True):
                    out = self.kernel(a, out)
                return out

        After = self._run(Before)
        orch = self._funcs_by_type(After, ir.FunctionType.Orchestration)[0]
        (submit,) = self._submit_values(orch)
        assert submit.allow_early_resolve is True
        # core_num rides on the Spmd Function attrs, not on the Submit.
        assert submit.core_num is None


class TestOutlinedReturnParamsExplicit:
    """Outlined wrapper functions return their params by pointer identity.

    Param-identity returns make the return->param mapping a lookup for
    orchestration codegen, so multi-output wrappers can never alias the
    returned value to the wrong output (issue #1702).
    """

    @staticmethod
    def _first_return(func):
        ret = func.body.stmts[-1]
        assert isinstance(ret, ir.ReturnStmt)
        return ret

    def test_spmd_wrapper_tuple_destructure_returns_param(self):
        """Spmd wrapper around a multi-out kernel returns the actual out param.

        Mirrors #1702: the kernel writes two InOut + one Out, the scope's only
        consumed result is the Out (last). The wrapper's return must be its own
        ``out`` param, not the TupleGetItem SSA alias.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                pre: pl.InOut[pl.Tensor[[64, 64], pl.FP32]],
                post: pl.InOut[pl.Tensor[[64, 64], pl.FP32]],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[64, 64], pl.FP32],
                pl.Tensor[[64, 64], pl.FP32],
                pl.Tensor[[64, 64], pl.FP32],
            ]:
                tile = pl.load(a, [0, 0], [64, 64])
                pre = pl.store(tile, [0, 0], pre)
                post = pl.store(tile, [0, 0], post)
                t2 = pl.add(tile, tile)
                out = pl.store(t2, [0, 0], out)
                return pre, post, out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                pre: pl.InOut[pl.Tensor[[64, 64], pl.FP32]],
                post: pl.InOut[pl.Tensor[[64, 64], pl.FP32]],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.spmd(4):
                    pre, post, out = self.kernel(a, pre, post, out)
                return out

        with passes.PassContext([], passes.VerificationLevel.NONE):
            after = passes.outline_cluster_scopes()(passes.convert_to_ssa()(Before))

        spmd_fns = [f for f in after.functions.values() if f.func_type == ir.FunctionType.Spmd]
        assert len(spmd_fns) == 1
        wrapper = spmd_fns[0]
        ret = self._first_return(wrapper)
        # Bind the projected values once: each `ret.value[i]` access creates a
        # fresh Python wrapper, so `id()` is only stable on a held reference.
        ret_values = list(ret.value)
        params = list(wrapper.params)
        assert len(ret_values) == 1
        # Pointer identity with the wrapper's own `out` param (params are
        # [a, pre, post, out] in first-use order).
        param_ids = {id(p): p.name_hint for p in params}
        assert id(ret_values[0]) in param_ids, "wrapper must return a param by identity"
        assert param_ids[id(ret_values[0])].startswith("out")

    def test_store_target_output_returns_inout_param(self):
        """A store-target output of a cluster scope returns the InOut param itself."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4) as tid:  # noqa: F841
                    i = pl.tile.get_block_idx()
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
                    out = pl.store(pl.add(t, t), [i * 128, 0], out)
                return out

        after = passes.outline_cluster_scopes()(
            passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
        )

        spmd_fns = [f for f in after.functions.values() if f.func_type == ir.FunctionType.Spmd]
        assert len(spmd_fns) == 1
        wrapper = spmd_fns[0]
        ret = self._first_return(wrapper)
        # Hold references so wrapper identity (`id`) stays stable across checks.
        ret_values = list(ret.value)
        params = list(wrapper.params)
        param_ids = {id(p) for p in params}
        for value in ret_values:
            assert id(value) in param_ids, "store-target output must be returned as the param itself"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
