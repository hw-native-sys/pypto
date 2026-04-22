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
                with pl.spmd(core_num=4, sync_start=True):
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
                out = self.kernel(x, out)
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

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_cluster_scopes()(Before)

        # The outlined function should have Group type
        group_func = After.get_function("main_cluster_0")
        assert group_func is not None
        assert group_func.func_type == ir.FunctionType.Group
        # y is not used after the cluster scope, so no return types
        assert len(group_func.return_types) == 0

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

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_cluster_scopes()(Before)

        # Verify the outlined function has Group type
        group_func = After.get_function("main_cluster_0")
        assert group_func is not None
        assert group_func.func_type == ir.FunctionType.Group

        # Verify it has 2 return types (y and z)
        assert len(group_func.return_types) == 2

    def test_outline_spmd_for_loop_auto_outlines_incore(self):
        """`for i in pl.spmd(core_num=N)` auto-outlines into Spmd + InCore.

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
                b: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(core_num=4):
                    offset = i * 128
                    tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [offset, 0], [128, 128])
                    out = pl.store(pl.add(tile_a, tile_b), [offset, 0], out)
                return out

        After = passes.convert_to_ssa()(Before)
        After = passes.outline_incore_scopes()(After)
        After = passes.outline_cluster_scopes()(After)

        # The pipeline produces three functions: main (Orchestration), a Spmd
        # wrapper, and an InCore leaf.  Check the structural properties that
        # matter rather than pinning exact stmt counts, since downstream SSA
        # and store-alias normalization inserts book-keeping assignments that
        # are not user-visible.
        func_names = sorted(f.name for f in After.functions.values())
        assert func_names == ["main", "main_incore_0", "main_spmd_0"]

        main_incore = next(f for f in After.functions.values() if f.name == "main_incore_0")
        assert main_incore.func_type == pl.FunctionType.InCore

        main_spmd = next(f for f in After.functions.values() if f.name == "main_spmd_0")
        assert main_spmd.func_type == pl.FunctionType.Spmd
        assert main_spmd.attrs.get("core_num") == 4

        main_fn = next(f for f in After.functions.values() if f.name == "main")
        assert main_fn.func_type == pl.FunctionType.Orchestration

        # The InCore leaf must start by binding the block index.
        body = main_incore.body
        first = body.stmts[0] if isinstance(body, ir.SeqStmts) else body
        assert isinstance(first, ir.AssignStmt)
        assert isinstance(first.value, ir.Call)
        assert first.value.op.name == "tile.get_block_idx"

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
