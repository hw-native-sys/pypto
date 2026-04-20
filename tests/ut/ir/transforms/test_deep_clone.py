# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for DeepClone utility.

Tests that deep cloning creates fresh Var/IterArg objects at definition sites
while preserving IR structure and SSA consistency.
"""

import pypto.language as pl
import pytest
from pypto import DataType, backend, ir, passes
from pypto.backend import BackendType


def _get_function(program: ir.Program, name: str) -> ir.Function:
    """Get a function from a program by name."""
    func = program.get_function(name)
    assert func is not None, f"Function '{name}' not found in program"
    return func


class TestDeepCloneBasic:
    """Basic deep clone tests."""

    def test_clone_simple_body(self):
        """Deep-cloned body should preserve structure."""

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

        func = _get_function(P, "main")
        cloned_body, var_map = ir.deep_clone(func.body)
        assert cloned_body is not None
        # Simple return has no definition sites, so var_map should be empty
        assert len(var_map) == 0

    def test_clone_assign_stmt(self):
        """Deep clone creates fresh Var for AssignStmt LHS."""

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                return y

        func = _get_function(P, "main")
        _cloned_body, var_map = ir.deep_clone(func.body)
        # var_map should contain at least the AssignStmt LHS var 'y'
        assert len(var_map) > 0
        # Each pair should have two distinct Var objects
        for orig, clone in var_map:
            assert orig is not clone

    def test_clone_body_is_distinct(self):
        """Cloned body should be a different object from the original."""

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                return y

        func = _get_function(P, "main")
        cloned_body, _var_map = ir.deep_clone(func.body)
        assert cloned_body is not func.body


class TestDeepCloneNoSharedIdentity:
    """Tests that deep clone produces no shared Var identity."""

    def test_two_clones_no_shared_vars(self):
        """Two deep clones of the same body should have distinct Var objects."""

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                return y

        func = _get_function(P, "main")
        _, map1 = ir.deep_clone(func.body)
        _, map2 = ir.deep_clone(func.body)

        # Both clones should have mappings
        assert len(map1) > 0
        assert len(map2) > 0

        # The cloned Vars from each clone should be different objects
        clones1 = {id(clone) for _, clone in map1}
        clones2 = {id(clone) for _, clone in map2}
        assert clones1.isdisjoint(clones2)


class TestDeepCloneWithExpandMixedKernel:
    """Integration test: ExpandMixedKernel uses DeepClone internally."""

    @pytest.fixture(autouse=True)
    def _setup_backend(self):
        """Configure Ascend950 backend for expand_mixed_kernel tests."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend950)
        yield
        backend.reset_for_testing()

    def test_expand_mixed_kernel_whole_program(self):
        """After ExpandMixedKernel, whole-program structural equality should work."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                x_left: pl.Tile[[16, 128], pl.BF16] = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_left, y_right)
                z_vec: pl.Tile[[16, 128], pl.FP32] = pl.move(z_tile, target_memory=pl.MemorySpace.Vec)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                z: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        After = passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(Before))

        # Run expand again on the same input — should produce structurally equal result
        After2 = passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(Before))

        # Whole-program structural equality should work now that DeepClone is used
        ir.assert_structural_equal(After, After2)


class TestDeepCloneTypeRemap:
    """Type annotations embed Exprs (shape dims, TileView/TensorView fields,
    MemRef byte_offset) that may reference scope Vars. DeepClone must rewrite
    those embedded Exprs through its expr_map_ — otherwise a cloned Var carries
    a type pointing at out-of-scope originals, which the printer surfaces as
    `pl.dynamic(...)` forward declarations.
    """

    @staticmethod
    def _make_assign_with_tile_var_referencing(loop_var: ir.Var) -> tuple[ir.AssignStmt, ir.Var]:
        """Build `t = loop_var` where t has TileType whose tile_view.valid_shape[0] is loop_var."""
        span = ir.Span.unknown()
        tile_view = ir.TileView(
            valid_shape=[loop_var],
            stride=[ir.ConstInt(1, DataType.INDEX, span)],
            start_offset=ir.ConstInt(0, DataType.INDEX, span),
        )
        tile_type = ir.TileType(
            [ir.ConstInt(4, DataType.INDEX, span)],
            DataType.FP32,
            None,  # memref
            tile_view,
            None,  # memory_space
        )
        t_var = ir.Var("t", tile_type, span)
        # The assignment value isn't the focus — we only inspect the LHS Var's type.
        return ir.AssignStmt(t_var, loop_var, span), t_var

    def test_cloned_tile_type_references_substituted_var(self):
        """valid_shape referencing a scope var gets remapped to the substitute."""
        span = ir.Span.unknown()
        i_orig = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        j_sub = ir.Var("j", ir.ScalarType(DataType.INDEX), span)

        assign, t_var = self._make_assign_with_tile_var_referencing(i_orig)
        cloned, var_map = ir.deep_clone(assign, var_map=[(i_orig, j_sub)])

        assert isinstance(cloned, ir.AssignStmt)
        cloned_t = cloned.var
        assert cloned_t is not t_var  # fresh Var at the def site
        cloned_tile_type = cloned_t.type
        assert isinstance(cloned_tile_type, ir.TileType)
        assert cloned_tile_type.tile_view is not None

        # The embedded valid_shape[0] must now reference j_sub, not i_orig.
        valid_shape_0 = cloned_tile_type.tile_view.valid_shape[0]
        assert valid_shape_0 is j_sub
        assert valid_shape_0 is not i_orig

        # Sanity: the def-site var_map contains the t clone.
        assert any(orig is t_var and clone is cloned_t for orig, clone in var_map)

    def test_cloned_type_unchanged_when_no_substitution(self):
        """Without substitution, the embedded Expr identity survives intact."""
        span = ir.Span.unknown()
        i_orig = ir.Var("i", ir.ScalarType(DataType.INDEX), span)

        assign, _ = self._make_assign_with_tile_var_referencing(i_orig)
        cloned, _ = ir.deep_clone(assign)  # empty var_map

        assert isinstance(cloned, ir.AssignStmt)
        cloned_tile_type = cloned.var.type
        assert isinstance(cloned_tile_type, ir.TileType)
        assert cloned_tile_type.tile_view is not None
        # External vars pass through VisitExpr_(VarPtr) unchanged (no substitution).
        assert cloned_tile_type.tile_view.valid_shape[0] is i_orig


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
