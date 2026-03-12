# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ExpandMixedKernel pass.

Note on test strategy:
  Non-mixed cases (pure vector, pure cube) verify that the FunctionType is
  converted to AIV or AIC respectively.  Orchestration functions pass through
  unchanged.

  Split cases use property-based checks (function types, printed body content,
  parameter lists, cross-core op presence) because TileView information on Var
  types in the C++ pass output cannot be expressed in the DSL, which blocks
  ir.assert_structural_equal.  tpop ops now use zero positional arguments with
  explicit type (no SSA self-reference).
"""

import re

import pypto.language as pl
import pytest
from pypto import ir, passes

# ---------------------------------------------------------------------------
# Shared helpers: program builders and pass invocation
# ---------------------------------------------------------------------------


def _expand(program):
    """Run infer_tile_memory_space then expand_mixed_kernel."""
    return passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(program))


def _make_matmul_program():
    """Standard mixed kernel: load→Mat→Left/Right, matmul, move→Vec, store."""

    @pl.program
    class P:
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

    return P


def _make_matmul_exp_program():
    """Mixed kernel with post-matmul exp: matmul → move→Vec → exp → store."""

    @pl.program
    class P:
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
            w_tile: pl.Tile[[16, 128], pl.FP32] = pl.exp(z_vec)
            out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(w_tile, [0, 0], out_0)
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

    return P


# ---------------------------------------------------------------------------
# Pass-through: programs that should NOT be split
# ---------------------------------------------------------------------------


class TestPassthrough:
    """Tests where the program is not split (pure vector, orchestration, pure cube).

    Non-mixed InCore functions get their FunctionType converted to AIC or AIV.
    """

    def test_pure_vector_becomes_aiv(self):
        """InCore with only vector ops → no split, FunctionType becomes AIV."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.add(x_tile, x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        Inferred = passes.infer_tile_memory_space()(Before)
        After = passes.expand_mixed_kernel()(Inferred)
        func = After.get_function("main_incore_0")
        assert func is not None
        assert func.func_type == pl.FunctionType.AIV

    def test_orchestration_unchanged(self):
        """Non-InCore functions pass through unchanged."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

        Inferred = passes.infer_tile_memory_space()(Before)
        After = passes.expand_mixed_kernel()(Inferred)
        ir.assert_structural_equal(After, Inferred)

    def test_pure_cube_becomes_aic(self):
        """InCore with only cube ops (no Acc→Vec boundary) → no split, FunctionType becomes AIC.

        Uses load(Mat) + move(Mat→Left/Right) + matmul + store(Acc tile directly).
        """

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
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
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

        Inferred = passes.infer_tile_memory_space()(Before)
        After = passes.expand_mixed_kernel()(Inferred)

        # No CV boundary move → not mixed → FunctionType becomes AIC
        func = After.get_function("main_incore_0")
        assert func is not None
        assert func.func_type == pl.FunctionType.AIC

    def test_pure_vector_inside_loop_becomes_aiv(self):
        """InCore with only vector ops inside a loop → no split, FunctionType becomes AIV."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(4):
                    x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                    y_tile: pl.Tile[[64], pl.FP32] = pl.add(x_tile, x_tile)
                    out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        Inferred = passes.infer_tile_memory_space()(Before)
        After = passes.expand_mixed_kernel()(Inferred)
        func = After.get_function("main_incore_0")
        assert func is not None
        assert func.func_type == pl.FunctionType.AIV


# ---------------------------------------------------------------------------
# Split structure: AIC / AIV / Group function properties
# ---------------------------------------------------------------------------


class TestSplitStructure:
    """Test the structure of generated AIC, AIV, and Group functions."""

    @pytest.fixture()
    def expanded(self):
        """Standard matmul program expanded into AIC + AIV + Group."""
        return _expand(_make_matmul_program())

    def test_aic_has_no_return_types(self, expanded):
        aic_func = expanded.get_function("main_incore_0_aic")
        assert aic_func is not None
        assert len(aic_func.return_types) == 0

    def test_aiv_preserves_return_types(self, expanded):
        aiv_func = expanded.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        assert len(aiv_func.return_types) > 0

    def test_aic_contains_cube_ops_not_vector_store(self, expanded):
        aic_func = expanded.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "matmul" in aic_str
        assert "tile.load(" in aic_str  # load(Mat) is CUBE
        assert "tile.store(" not in aic_str

    def test_aiv_contains_store_not_matmul(self, expanded):
        aiv_func = expanded.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "tile.store(" in aiv_str
        assert "tile.matmul(" not in aiv_str

    def test_group_calls_aic_then_aiv(self, expanded):
        group_func = expanded.get_function("main_incore_0")
        assert group_func is not None
        group_str = group_func.as_python()
        assert "main_incore_0_aic" in group_str
        assert "main_incore_0_aiv" in group_str

    def test_group_replaces_original_name(self, expanded):
        group_func = expanded.get_function("main_incore_0")
        assert group_func is not None
        assert group_func.func_type == pl.FunctionType.Group

    def test_orchestration_call_site_unchanged(self, expanded):
        main_func = expanded.get_function("main")
        assert main_func is not None
        assert main_func.func_type == pl.FunctionType.Orchestration
        main_str = main_func.as_python()
        assert "main_incore_0" in main_str

    def test_params_preserved_on_aic(self, expanded):
        aic_func = expanded.get_function("main_incore_0_aic")
        assert aic_func is not None
        param_names = [p.name for p in aic_func.params]
        assert "x" in param_names
        assert "y" in param_names
        assert "out_0" in param_names

    def test_params_preserved_on_aiv(self, expanded):
        aiv_func = expanded.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        param_names = [p.name for p in aiv_func.params]
        assert "x" in param_names
        assert "y" in param_names
        assert "out_0" in param_names

    def test_params_preserved_on_group(self, expanded):
        group_func = expanded.get_function("main_incore_0")
        assert group_func is not None
        param_names = [p.name for p in group_func.params]
        assert "x" in param_names
        assert "y" in param_names
        assert "out_0" in param_names

    def test_function_count_after_split(self):
        """After splitting 1 mixed InCore: original 2 funcs → 4 (AIC + AIV + Group + Orch)."""
        Before = _make_matmul_program()
        assert len(Before.functions) == 2

        After = _expand(Before)
        assert len(After.functions) == 4


# ---------------------------------------------------------------------------
# Cross-core boundary detection and TPUSH/TPOP insertion
# ---------------------------------------------------------------------------


class TestCrossCoreBoundaries:
    """Test C↔V boundary detection and cross-core communication ops."""

    @pytest.fixture()
    def expanded_matmul_exp(self):
        """matmul + exp program → C→V boundary at Acc→Vec move."""
        return _expand(_make_matmul_exp_program())

    def test_c2v_boundary_tpush_tpop(self, expanded_matmul_exp):
        """matmul result used by exp → tpush_to_aiv / tpop_from_aic."""
        aic_func = expanded_matmul_exp.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "tpush_to_aiv" in aic_str

        aiv_func = expanded_matmul_exp.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "tpop_from_aic" in aiv_str

    def test_aiv_idx_is_zero(self, expanded_matmul_exp):
        """All TPUSH/TPOP should use aiv_idx=0."""
        result_str = expanded_matmul_exp.as_python()
        assert "aiv_idx=0" in result_str
        aiv_idx_vals = re.findall(r"aiv_idx=(\d+)", result_str)
        assert all(v == "0" for v in aiv_idx_vals)

    def test_v2c_boundary_add_to_matmul(self):
        """Pre-matmul vector op: add(x,x) produces V→C boundary to matmul."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
                x_sum: pl.Tile[[16, 128], pl.BF16] = pl.add(x_tile, x_tile)
                x_sum_mat: pl.Tile[[16, 128], pl.BF16] = pl.move(x_sum, target_memory=pl.MemorySpace.Mat)
                x_sum_left: pl.Tile[[16, 128], pl.BF16] = pl.move(
                    x_sum_mat, target_memory=pl.MemorySpace.Left
                )
                y_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_sum_left, y_right)
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

        After = _expand(Before)

        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "add" in aiv_str
        assert "tpush_to_aic" in aiv_str

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "matmul" in aic_str
        assert "tpop_from_aiv" in aic_str


# ---------------------------------------------------------------------------
# Cube op variant classification
# ---------------------------------------------------------------------------


class TestCubeOpVariants:
    """Test that all cube op variants are correctly classified and placed in AIC."""

    def test_matmul_acc_in_aic(self):
        """matmul + matmul_acc → both in AIC, none in AIV."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                a_left: pl.Tile[[16, 128], pl.BF16] = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                b_right: pl.Tile[[128, 128], pl.BF16] = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                c_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(a_left, b_right)
                d_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul_acc(c_tile, a_left, b_right)
                d_vec: pl.Tile[[16, 128], pl.FP32] = pl.move(d_tile, target_memory=pl.MemorySpace.Vec)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(d_vec, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                c: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(a, b, out_0)
                return c

        After = _expand(Before)

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        assert aic_func.func_type == pl.FunctionType.AIC
        aic_str = aic_func.as_python()
        assert "tile.matmul(" in aic_str
        assert "matmul_acc" in aic_str

        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "matmul" not in aiv_str

    def test_matmul_bias_in_aic(self):
        """tile.matmul_bias is a CUBE op → triggers split."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                bias: pl.Tensor[[1, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                a_left: pl.Tile[[16, 128], pl.BF16] = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                b_right: pl.Tile[[128, 128], pl.BF16] = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                bias_tile: pl.Tile[[1, 128], pl.FP32] = pl.load(bias, [0, 0], [1, 128])
                bias_mat: pl.Tile[[1, 128], pl.FP32] = pl.move(bias_tile, target_memory=pl.MemorySpace.Mat)
                c_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul_bias(a_left, b_right, bias_mat)
                c_vec: pl.Tile[[16, 128], pl.FP32] = pl.move(c_tile, target_memory=pl.MemorySpace.Vec)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(c_vec, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                bias: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                c: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(a, b, bias, out_0)
                return c

        After = _expand(Before)
        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "matmul_bias" in aic_str

    def test_gemv_in_aic(self):
        """tile.gemv is a CUBE op → triggers split."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                a_left: pl.Tile[[16, 128], pl.BF16] = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                b_right: pl.Tile[[128, 128], pl.BF16] = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                c_tile: pl.Tile[[16, 128], pl.FP32] = pl.gemv(a_left, b_right)
                c_vec: pl.Tile[[16, 128], pl.FP32] = pl.move(c_tile, target_memory=pl.MemorySpace.Vec)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(c_vec, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                c: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(a, b, out_0)
                return c

        After = _expand(Before)
        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        assert aic_func.func_type == pl.FunctionType.AIC
        assert "gemv" in aic_func.as_python()

    def test_gemv_acc_in_aic(self):
        """tile.gemv_acc is a CUBE op → triggers split."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                a_left: pl.Tile[[16, 128], pl.BF16] = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                b_right: pl.Tile[[128, 128], pl.BF16] = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                c_tile: pl.Tile[[16, 128], pl.FP32] = pl.gemv(a_left, b_right)
                a_left2: pl.Tile[[16, 128], pl.BF16] = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_right2: pl.Tile[[128, 128], pl.BF16] = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                d_tile: pl.Tile[[16, 128], pl.FP32] = pl.gemv_acc(c_tile, a_left2, b_right2)
                d_vec: pl.Tile[[16, 128], pl.FP32] = pl.move(d_tile, target_memory=pl.MemorySpace.Vec)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(d_vec, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                c: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(a, b, out_0)
                return c

        After = _expand(Before)
        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        assert "gemv_acc" in aic_func.as_python()

    def test_gemv_bias_in_aic(self):
        """tile.gemv_bias is a CUBE op → triggers split."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                bias: pl.Tensor[[1, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                a_left: pl.Tile[[16, 128], pl.BF16] = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                b_right: pl.Tile[[128, 128], pl.BF16] = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                bias_tile: pl.Tile[[1, 128], pl.FP32] = pl.load(bias, [0, 0], [1, 128])
                bias_mat: pl.Tile[[1, 128], pl.FP32] = pl.move(bias_tile, target_memory=pl.MemorySpace.Mat)
                c_tile: pl.Tile[[16, 128], pl.FP32] = pl.gemv_bias(a_left, b_right, bias_mat)
                c_vec: pl.Tile[[16, 128], pl.FP32] = pl.move(c_tile, target_memory=pl.MemorySpace.Vec)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(c_vec, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                bias: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                c: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(a, b, bias, out_0)
                return c

        After = _expand(Before)
        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        assert "gemv_bias" in aic_func.as_python()


# ---------------------------------------------------------------------------
# Vector op classification
# ---------------------------------------------------------------------------


class TestVectorOpClassification:
    """Test that vector ops are correctly classified and placed in AIV."""

    def test_exp_is_vector(self):
        """tile.exp should be in AIV, not AIC."""
        After = _expand(_make_matmul_exp_program())

        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "tile.exp(" in aiv_str

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "tile.exp(" not in aic_str

    def test_sub_is_vector(self):
        """tile.sub should be in AIV, not AIC."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
                x_sub: pl.Tile[[16, 128], pl.BF16] = pl.sub(x_tile, x_tile)
                x_sub_mat: pl.Tile[[16, 128], pl.BF16] = pl.move(x_sub, target_memory=pl.MemorySpace.Mat)
                x_sub_left: pl.Tile[[16, 128], pl.BF16] = pl.move(
                    x_sub_mat, target_memory=pl.MemorySpace.Left
                )
                y_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_sub_left, y_right)
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

        After = _expand(Before)

        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "tile.sub(" in aiv_str

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "tile.sub(" not in aic_str

    def test_dn_transpose_moves_in_aic(self):
        """Cube moves (Mat→Left/Right) with DN layout and transpose stay in AIC.

        Merges coverage from same-side-cube-move and with-move-before-matmul tests.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16, pl.DN],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_l1: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                x_left: pl.Tile[[16, 128], pl.BF16] = pl.move(x_l1, target_memory=pl.MemorySpace.Left)
                y_l1: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    y,
                    [0, 0],
                    [128, 128],
                    target_memory=pl.MemorySpace.Mat,
                    transpose=True,
                )
                y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_l1, target_memory=pl.MemorySpace.Right)
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_left, y_right)
                z_vec: pl.Tile[[16, 128], pl.FP32] = pl.move(z_tile, target_memory=pl.MemorySpace.Vec)
                z_exp: pl.Tile[[16, 128], pl.FP32] = pl.exp(z_vec)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_exp, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16, pl.DN],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                z: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        After = _expand(Before)

        # AIC: cube moves + matmul + load(Mat) with transpose
        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "tile.move(" in aic_str
        assert "matmul" in aic_str
        assert "tile.load(" in aic_str
        assert "transpose=True" in aic_str

        # AIV: exp + store, no matmul
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "exp" in aiv_str
        assert "tile.store(" in aiv_str
        assert "tile.matmul(" not in aiv_str

        # DN layout preserved in orchestration
        main_func = After.get_function("main")
        assert main_func is not None
        main_str = ir.python_print(main_func)
        assert "TensorLayout.DN" in main_str


# ---------------------------------------------------------------------------
# Realistic computation patterns
# ---------------------------------------------------------------------------


class TestRealisticPatterns:
    """Test realistic computation patterns (attention, post-processing chains)."""

    @pytest.fixture()
    def expanded_matmul_exp_add(self):
        """matmul → exp → add pattern (attention-like)."""

        @pl.program
        class P:
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
                exp_tile: pl.Tile[[16, 128], pl.FP32] = pl.exp(z_vec)
                sum_tile: pl.Tile[[16, 128], pl.FP32] = pl.add(exp_tile, exp_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(sum_tile, [0, 0], out_0)
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

        return _expand(P)

    def test_attention_pattern_split(self, expanded_matmul_exp_add):
        """matmul → exp → add: AIC gets matmul, AIV gets exp+add+store."""
        After = expanded_matmul_exp_add

        aic_func = After.get_function("main_incore_0_aic")
        aiv_func = After.get_function("main_incore_0_aiv")
        group_func = After.get_function("main_incore_0")
        assert aic_func is not None
        assert aiv_func is not None
        assert group_func is not None
        assert aic_func.func_type == pl.FunctionType.AIC
        assert aiv_func.func_type == pl.FunctionType.AIV
        assert group_func.func_type == pl.FunctionType.Group

        aic_str = aic_func.as_python()
        assert "tile.matmul(" in aic_str
        assert "tile.exp(" not in aic_str
        assert "tile.add(" not in aic_str

        aiv_str = aiv_func.as_python()
        assert "tile.exp(" in aiv_str
        assert "tile.store(" in aiv_str

    def test_dce_removes_vector_ops_from_aic(self, expanded_matmul_exp_add):
        """DCE: vector-only vars (exp, add) should not appear in AIC."""
        aic_func = expanded_matmul_exp_add.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "matmul" in aic_str
        assert "exp" not in aic_str
        assert "add" not in aic_str
        assert "tile.store(" not in aic_str

    def test_matmul_chain_vector_postprocessing(self):
        """matmul → exp → mul → store: multiple vector post-ops in AIV."""

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
                z_exp: pl.Tile[[16, 128], pl.FP32] = pl.exp(z_vec)
                z_mul: pl.Tile[[16, 128], pl.FP32] = pl.mul(z_exp, z_exp)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_mul, [0, 0], out_0)
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

        After = _expand(Before)

        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "exp" in aiv_str
        assert "mul" in aiv_str
        assert "tile.store(" in aiv_str

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "tile.matmul(" in aic_str
        assert "tpush_to_aiv" in aic_str
        assert "tile.exp(" not in aic_str
        assert "tile.mul(" not in aic_str


# ---------------------------------------------------------------------------
# Multiple InCore functions
# ---------------------------------------------------------------------------


class TestMultipleInCore:
    """Test behavior with multiple InCore functions in a program."""

    def test_multiple_mixed_functions(self):
        """Two mixed InCore functions → both are split independently."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def compute_a_incore_0(
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

            @pl.function(type=pl.FunctionType.InCore)
            def compute_b_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                a_left: pl.Tile[[16, 128], pl.BF16] = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
                b_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                b_right: pl.Tile[[128, 128], pl.BF16] = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
                c_tile: pl.Tile[[16, 128], pl.FP32] = pl.gemv(a_left, b_right)
                c_vec: pl.Tile[[16, 128], pl.FP32] = pl.move(c_tile, target_memory=pl.MemorySpace.Vec)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(c_vec, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                z: pl.Tensor[[16, 128], pl.FP32] = self.compute_a_incore_0(x, y, out_0)
                return z

        After = _expand(Before)

        assert After.get_function("compute_a_incore_0_aic") is not None
        assert After.get_function("compute_a_incore_0_aiv") is not None
        group_a = After.get_function("compute_a_incore_0")
        assert group_a is not None
        assert group_a.func_type == pl.FunctionType.Group

        assert After.get_function("compute_b_incore_0_aic") is not None
        assert After.get_function("compute_b_incore_0_aiv") is not None
        group_b = After.get_function("compute_b_incore_0")
        assert group_b is not None
        assert group_b.func_type == pl.FunctionType.Group

    def test_mixed_plus_pure_incore(self):
        """One mixed + one pure vector InCore → only mixed is split."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def pure_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.add(x_tile, x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.InCore)
            def mixed_incore_0(
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
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.pure_incore_0(x, out_0)
                return y

        After = _expand(Before)

        pure_func = After.get_function("pure_incore_0")
        assert pure_func is not None
        assert pure_func.func_type == pl.FunctionType.AIV

        assert After.get_function("mixed_incore_0_aic") is not None
        assert After.get_function("mixed_incore_0_aiv") is not None
        mixed_group = After.get_function("mixed_incore_0")
        assert mixed_group is not None
        assert mixed_group.func_type == pl.FunctionType.Group


# ---------------------------------------------------------------------------
# Property verification
# ---------------------------------------------------------------------------


class TestPropertyVerification:
    """Test property verification behavior with ExpandMixedKernel."""

    def test_produces_mixed_kernel_expanded_property(self):
        """After pass runs, MixedKernelExpanded property should be verifiable."""
        After = _expand(_make_matmul_program())

        prop_set = passes.IRPropertySet()
        prop_set.insert(passes.IRProperty.MixedKernelExpanded)
        passes.verify_properties(prop_set, After, "test")

    def test_verification_with_after_mode_instrument(self):
        """Property verification instrument works after expand."""
        Before = _make_matmul_program()

        instrument = passes.VerificationInstrument(passes.VerificationMode.AFTER)
        with passes.PassContext([instrument]):
            After = _expand(Before)

        assert After.get_function("main_incore_0_aic") is not None


# ---------------------------------------------------------------------------
# Nested structures (for loops)
# ---------------------------------------------------------------------------


class TestNestedStructures:
    """Test that mixed ops inside ForStmt are handled recursively."""

    def test_for_loop_split_and_boundaries(self):
        """Mixed ops inside a for loop → AIC/AIV each get the loop; TPUSH/TPOP inside."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                for i in pl.range(4):
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

        After = _expand(Before)

        aic_func = After.get_function("main_incore_0_aic")
        aiv_func = After.get_function("main_incore_0_aiv")
        group_func = After.get_function("main_incore_0")
        assert aic_func is not None
        assert aiv_func is not None
        assert group_func is not None
        assert aic_func.func_type == pl.FunctionType.AIC
        assert aiv_func.func_type == pl.FunctionType.AIV
        assert group_func.func_type == pl.FunctionType.Group

        # AIC: loop with matmul, no store
        aic_str = aic_func.as_python()
        assert "matmul" in aic_str
        assert "tile.store(" not in aic_str
        assert "pl.range" in aic_str
        assert "tpush_to_aiv" in aic_str

        # AIV: loop with store, no matmul
        aiv_str = aiv_func.as_python()
        assert "tile.store(" in aiv_str
        assert "tile.matmul(" not in aiv_str
        assert "pl.range" in aiv_str
        assert "tpop_from_aic" in aiv_str

    def test_bidirectional_inside_for_loop(self):
        """V→C and C→V boundaries inside same loop body.

        Pattern: load(Vec) → add (V) → move(Vec→Mat→Left) → matmul (C) → move(Acc→Vec) → exp (V) → store
        V→C: add result flows to matmul via tpush_to_aic / tpop_from_aiv
        C→V: matmul result flows to exp via tpush_to_aiv / tpop_from_aic
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                for i in pl.range(4):
                    x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
                    x_sum: pl.Tile[[16, 128], pl.BF16] = pl.add(x_tile, x_tile)
                    x_sum_mat: pl.Tile[[16, 128], pl.BF16] = pl.move(x_sum, target_memory=pl.MemorySpace.Mat)
                    x_sum_left: pl.Tile[[16, 128], pl.BF16] = pl.move(
                        x_sum_mat, target_memory=pl.MemorySpace.Left
                    )
                    y_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(
                        y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                    )
                    y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                    z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_sum_left, y_right)
                    z_vec: pl.Tile[[16, 128], pl.FP32] = pl.move(z_tile, target_memory=pl.MemorySpace.Vec)
                    w_tile: pl.Tile[[16, 128], pl.FP32] = pl.exp(z_vec)
                    out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(w_tile, [0, 0], out_0)
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

        After = _expand(Before)
        result_str = After.as_python()

        # C→V: AIC pushes matmul result to AIV for exp
        assert "tpush_to_aiv" in result_str
        assert "tpop_from_aic" in result_str

        # V→C: AIV pushes add result to AIC for matmul
        assert "tpush_to_aic" in result_str
        assert "tpop_from_aiv" in result_str

    def test_mixed_loop_plus_flat_ops(self):
        """load(Mat) outside loop + mixed ops inside loop."""

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
                for i in pl.range(2):
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

        After = _expand(Before)

        aic_func = After.get_function("main_incore_0_aic")
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aic_func is not None
        assert aiv_func is not None
        assert aic_func.func_type == pl.FunctionType.AIC
        assert aiv_func.func_type == pl.FunctionType.AIV

        aic_str = aic_func.as_python()
        assert "matmul" in aic_str
        assert "pl.range" in aic_str
        assert "tile.load(" in aic_str

        aiv_str = aiv_func.as_python()
        assert "tile.store(" in aiv_str
        assert "pl.range" in aiv_str


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_existing_group_from_cluster_outline(self):
        """Existing Group caller is rewritten to call AIC+AIV; no redundant Group wrapper."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def compute_incore_0(
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

            @pl.function(type=pl.FunctionType.Group)
            def compute_group(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                result: pl.Tensor[[16, 128], pl.FP32] = self.compute_incore_0(x, y, out_0)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                z: pl.Tensor[[16, 128], pl.FP32] = self.compute_group(x, y, out_0)
                return z

        After = _expand(Before)

        # AIC and AIV functions are created
        aic = After.get_function("compute_incore_0_aic")
        assert aic is not None
        assert aic.func_type == pl.FunctionType.AIC

        aiv = After.get_function("compute_incore_0_aiv")
        assert aiv is not None
        assert aiv.func_type == pl.FunctionType.AIV

        # No redundant Group wrapper with the InCore name
        assert After.get_function("compute_incore_0") is None

        # Original Group is preserved and rewritten to call AIC + AIV
        group = After.get_function("compute_group")
        assert group is not None
        assert group.func_type == pl.FunctionType.Group

        # Verify the Group body calls AIC then AIV directly
        group_str = str(group)
        assert "compute_incore_0_aic" in group_str
        assert "compute_incore_0_aiv" in group_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
