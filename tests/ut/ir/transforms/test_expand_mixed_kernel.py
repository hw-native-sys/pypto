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
  Pass-through cases (pure vector, orchestration-only) use the preferred
  Before/Expected + ir.assert_structural_equal pattern.

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


class TestExpandMixedKernelBasics:
    """Test basic ExpandMixedKernel pass functionality."""

    def test_pure_vector_unchanged(self):
        """InCore with only vector ops (tile.load + tile.add + tile.store) -> no split."""

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

        After = passes.expand_mixed_kernel()(Before)
        ir.assert_structural_equal(After, Before)

    def test_orchestration_unchanged(self):
        """Non-InCore functions pass through unchanged."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

        After = passes.expand_mixed_kernel()(Before)
        ir.assert_structural_equal(After, Before)

    def test_simple_mixed_split(self):
        """tile.load + tile.matmul + tile.store -> AIC + AIV + Group with TPUSH/TPOP."""

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
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
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

        After = passes.expand_mixed_kernel()(Before)

        # Should have: main_incore_0_aic, main_incore_0_aiv, main_incore_0 (Group), main
        aic_func = After.get_function("main_incore_0_aic")
        aiv_func = After.get_function("main_incore_0_aiv")
        group_func = After.get_function("main_incore_0")
        main_func = After.get_function("main")

        assert aic_func is not None
        assert aiv_func is not None
        assert group_func is not None
        assert main_func is not None

        assert aic_func.func_type == pl.FunctionType.AIC
        assert aiv_func.func_type == pl.FunctionType.AIV
        assert group_func.func_type == pl.FunctionType.Group


class TestExpandMixedKernelFunctionStructure:
    """Test the structure of generated AIC, AIV, and Group functions."""

    @pytest.fixture()
    def simple_mixed_result(self):
        """A simple load→matmul→store mixed kernel expanded result."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def compute_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                z: pl.Tensor[[16, 128], pl.FP32] = self.compute_incore_0(x, y, out_0)
                return z

        return passes.expand_mixed_kernel()(Before)

    def test_aic_has_no_return_types(self, simple_mixed_result):
        """AIC function should have empty return_types."""
        aic_func = simple_mixed_result.get_function("compute_incore_0_aic")
        assert aic_func is not None
        assert len(aic_func.return_types) == 0

    def test_aiv_preserves_return_types(self, simple_mixed_result):
        """AIV function should have the same return types as the original InCore."""
        aiv_func = simple_mixed_result.get_function("compute_incore_0_aiv")
        assert aiv_func is not None
        assert len(aiv_func.return_types) > 0

    def test_aic_contains_matmul_not_load_store(self, simple_mixed_result):
        """AIC body should contain matmul but not load/store."""
        aic_func = simple_mixed_result.get_function("compute_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "matmul" in aic_str
        assert "tile.load(" not in aic_str
        assert "tile.store(" not in aic_str

    def test_aiv_contains_load_store_not_matmul(self, simple_mixed_result):
        """AIV body should contain load/store but not matmul."""
        aiv_func = simple_mixed_result.get_function("compute_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "tile.load(" in aiv_str
        assert "tile.store(" in aiv_str
        assert "tile.matmul(" not in aiv_str

    def test_group_calls_aic_then_aiv(self, simple_mixed_result):
        """Group function body should reference both AIC and AIV."""
        group_func = simple_mixed_result.get_function("compute_incore_0")
        assert group_func is not None
        group_str = group_func.as_python()
        assert "compute_incore_0_aic" in group_str
        assert "compute_incore_0_aiv" in group_str

    def test_group_replaces_original_name(self, simple_mixed_result):
        """Group function keeps the original InCore function name."""
        group_func = simple_mixed_result.get_function("compute_incore_0")
        assert group_func is not None
        assert group_func.func_type == pl.FunctionType.Group

    def test_orchestration_call_site_unchanged(self, simple_mixed_result):
        """Orchestration still calls the same function name (now Group)."""
        main_func = simple_mixed_result.get_function("main")
        assert main_func is not None
        assert main_func.func_type == pl.FunctionType.Orchestration
        main_str = main_func.as_python()
        assert "compute_incore_0" in main_str

    def test_params_preserved_on_aic(self, simple_mixed_result):
        """AIC function should have the same parameters as the original InCore."""
        aic_func = simple_mixed_result.get_function("compute_incore_0_aic")
        assert aic_func is not None
        param_names = [p.name for p in aic_func.params]
        assert "x" in param_names
        assert "y" in param_names
        assert "out_0" in param_names

    def test_params_preserved_on_aiv(self, simple_mixed_result):
        """AIV function should have the same parameters as the original InCore."""
        aiv_func = simple_mixed_result.get_function("compute_incore_0_aiv")
        assert aiv_func is not None
        param_names = [p.name for p in aiv_func.params]
        assert "x" in param_names
        assert "y" in param_names
        assert "out_0" in param_names

    def test_params_preserved_on_group(self, simple_mixed_result):
        """Group function should have the same parameters as the original InCore."""
        group_func = simple_mixed_result.get_function("compute_incore_0")
        assert group_func is not None
        param_names = [p.name for p in group_func.params]
        assert "x" in param_names
        assert "y" in param_names
        assert "out_0" in param_names


class TestExpandMixedKernelBoundaries:
    """Test cross-core boundary detection and TPUSH/TPOP insertion."""

    def test_v2c_boundary_load_to_matmul(self):
        """Vector→Cube: tile.load result used by tile.matmul → tpush_to_aic / tpop_from_aiv."""

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
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
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

        After = passes.expand_mixed_kernel()(Before)

        # AIV side: tpush_to_aic for the loaded tiles flowing to matmul
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "tpush_to_aic" in aiv_str

        # AIC side: tpop_from_aiv to receive loaded tiles
        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "tpop_from_aiv" in aic_str

    def test_c2v_boundary_matmul_to_exp(self):
        """Cube→Vector: tile.matmul result used by tile.exp → tpush_to_aiv / tpop_from_aic."""

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
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
                w_tile: pl.Tile[[16, 128], pl.FP32] = pl.exp(z_tile)
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

        After = passes.expand_mixed_kernel()(Before)

        # AIC side: tpush_to_aiv for matmul result flowing to exp
        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "tpush_to_aiv" in aic_str

        # AIV side: tpop_from_aic to receive matmul result
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "tpop_from_aic" in aiv_str

    def test_bidirectional_boundaries(self):
        """V→C and C→V boundaries in the same function."""

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
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
                w_tile: pl.Tile[[16, 128], pl.FP32] = pl.exp(z_tile)
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

        After = passes.expand_mixed_kernel()(Before)
        result_str = After.as_python()

        # Both directions of cross-core communication should be present
        assert "tpush_to_aic" in result_str  # V→C: load tiles to matmul
        assert "tpop_from_aiv" in result_str  # V→C: matmul receives from load
        assert "tpush_to_aiv" in result_str  # C→V: matmul result to exp
        assert "tpop_from_aic" in result_str  # C→V: exp receives from matmul


class TestExpandMixedKernelCubeOpVariants:
    """Test that all Cube op variants are correctly classified."""

    def test_matmul_acc_classified_as_cube(self):
        """tile.matmul_acc is a CUBE op → triggers split."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(a, [0, 0], [16, 128])
                b_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(b, [0, 0], [128, 128])
                c_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(a_tile, b_tile)
                d_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul_acc(c_tile, a_tile, b_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(d_tile, [0, 0], out_0)
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

        After = passes.expand_mixed_kernel()(Before)

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        assert aic_func.func_type == pl.FunctionType.AIC

        # Both matmul and matmul_acc should be in AIC
        aic_str = aic_func.as_python()
        assert "matmul_acc" in aic_str
        assert "matmul" in aic_str

    def test_matmul_bias_classified_as_cube(self):
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
                a_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(a, [0, 0], [16, 128])
                b_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(b, [0, 0], [128, 128])
                bias_tile: pl.Tile[[1, 128], pl.FP32] = pl.load(bias, [0, 0], [1, 128])
                c_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul_bias(a_tile, b_tile, bias_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(c_tile, [0, 0], out_0)
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

        After = passes.expand_mixed_kernel()(Before)

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "matmul_bias" in aic_str

        # load should NOT be in AIC (it's VECTOR)
        assert "tile.load(" not in aic_str

    def test_gemv_classified_as_cube(self):
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
                a_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(a, [0, 0], [16, 128])
                b_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(b, [0, 0], [128, 128])
                c_tile: pl.Tile[[16, 128], pl.FP32] = pl.gemv(a_tile, b_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(c_tile, [0, 0], out_0)
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

        After = passes.expand_mixed_kernel()(Before)

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        assert aic_func.func_type == pl.FunctionType.AIC
        aic_str = aic_func.as_python()
        assert "gemv" in aic_str

    def test_gemv_acc_classified_as_cube(self):
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
                a_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(a, [0, 0], [16, 128])
                b_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(b, [0, 0], [128, 128])
                c_tile: pl.Tile[[16, 128], pl.FP32] = pl.gemv(a_tile, b_tile)
                d_tile: pl.Tile[[16, 128], pl.FP32] = pl.gemv_acc(c_tile, a_tile, b_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(d_tile, [0, 0], out_0)
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

        After = passes.expand_mixed_kernel()(Before)

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "gemv_acc" in aic_str

    def test_gemv_bias_classified_as_cube(self):
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
                a_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(a, [0, 0], [16, 128])
                b_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(b, [0, 0], [128, 128])
                bias_tile: pl.Tile[[1, 128], pl.FP32] = pl.load(bias, [0, 0], [1, 128])
                c_tile: pl.Tile[[16, 128], pl.FP32] = pl.gemv_bias(a_tile, b_tile, bias_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(c_tile, [0, 0], out_0)
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

        After = passes.expand_mixed_kernel()(Before)

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "gemv_bias" in aic_str


class TestExpandMixedKernelVectorOpClassification:
    """Test that various vector ops are correctly classified as VECTOR (not CUBE)."""

    def test_tile_move_is_vector(self):
        """tile.move should be classified as VECTOR, not CUBE."""

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
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_l1, target_memory=pl.MemorySpace.Right)
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_left, y_right)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
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

        After = passes.expand_mixed_kernel()(Before)

        # move is VECTOR → should be in AIV, not AIC
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "tile.move(" in aiv_str

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "tile.move(" not in aic_str

        # Verify DN layout and transpose metadata are preserved in split functions
        main_func = After.get_function("main")
        assert main_func is not None
        main_str = ir.python_print(main_func)
        assert "TensorLayout.DN" in main_str
        assert "transpose=True" in aiv_str

    def test_tile_sub_is_vector(self):
        """tile.sub should be classified as VECTOR, not CUBE."""

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
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_sub, y_tile)
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

        After = passes.expand_mixed_kernel()(Before)

        # sub is VECTOR → should be in AIV, not AIC
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "tile.sub(" in aiv_str

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "tile.sub(" not in aic_str

    def test_tile_exp_is_vector(self):
        """tile.exp should be classified as VECTOR."""

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
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
                e_tile: pl.Tile[[16, 128], pl.FP32] = pl.exp(z_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(e_tile, [0, 0], out_0)
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

        After = passes.expand_mixed_kernel()(Before)

        # exp is VECTOR → should be in AIV, not AIC
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "tile.exp(" in aiv_str

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "tile.exp(" not in aic_str


class TestExpandMixedKernelRealisticPatterns:
    """Test realistic computation patterns similar to attention mechanisms."""

    def test_attention_qk_softmax_pattern(self):
        """Realistic attention: load Q,K → matmul(QK) → exp → add → store.

        Pattern: V→V→C→V→V→V (load, load, matmul, exp, add, store)
        Boundaries: V→C (loaded tiles to matmul), C→V (matmul result to exp)
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def attn_incore_0(
                self,
                q: pl.Tensor[[16, 128], pl.BF16],
                k: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                q_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(q, [0, 0], [16, 128])
                k_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(k, [0, 0], [128, 128])
                qk_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(q_tile, k_tile)
                exp_tile: pl.Tile[[16, 128], pl.FP32] = pl.exp(qk_tile)
                norm_tile: pl.Tile[[16, 128], pl.FP32] = pl.add(exp_tile, exp_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(norm_tile, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                q: pl.Tensor[[16, 128], pl.BF16],
                k: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                z: pl.Tensor[[16, 128], pl.FP32] = self.attn_incore_0(q, k, out_0)
                return z

        After = passes.expand_mixed_kernel()(Before)

        # Verify split
        aic_func = After.get_function("attn_incore_0_aic")
        aiv_func = After.get_function("attn_incore_0_aiv")
        group_func = After.get_function("attn_incore_0")
        assert aic_func is not None
        assert aiv_func is not None
        assert group_func is not None
        assert aic_func.func_type == pl.FunctionType.AIC
        assert aiv_func.func_type == pl.FunctionType.AIV
        assert group_func.func_type == pl.FunctionType.Group

        # AIC: only matmul + cross-core communication
        aic_str = aic_func.as_python()
        assert "tile.matmul(" in aic_str
        assert "tile.load(" not in aic_str
        assert "tile.exp(" not in aic_str
        assert "tile.add(" not in aic_str

        # AIV: load, exp, add, store + cross-core communication
        aiv_str = aiv_func.as_python()
        assert "tile.load(" in aiv_str
        assert "tile.exp(" in aiv_str
        assert "tile.store(" in aiv_str

    def test_load_add_matmul_pattern(self):
        """Pre-matmul vector processing: load → add → matmul → store.

        add(x,x) produces a V2C boundary var used by matmul.
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
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
                x_sum: pl.Tile[[16, 128], pl.BF16] = pl.add(x_tile, x_tile)
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_sum, y_tile)
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

        After = passes.expand_mixed_kernel()(Before)

        # add is VECTOR → should be in AIV, its result flows to matmul (V→C)
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "add" in aiv_str
        assert "tpush_to_aic" in aiv_str  # x_sum flows V→C

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "matmul" in aic_str
        assert "tpop_from_aiv" in aic_str  # receive x_sum

    def test_matmul_chain_with_vector_postprocessing(self):
        """matmul → exp → mul → store: C→V chain with multiple vector post-ops."""

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
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
                z_exp: pl.Tile[[16, 128], pl.FP32] = pl.exp(z_tile)
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

        After = passes.expand_mixed_kernel()(Before)

        # All post-matmul ops should be in AIV
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "exp" in aiv_str
        assert "mul" in aiv_str
        assert "tile.store(" in aiv_str

        # AIC should only have matmul + tpush for result
        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "tile.matmul(" in aic_str
        assert "tpush_to_aiv" in aic_str
        assert "tile.exp(" not in aic_str
        assert "tile.mul(" not in aic_str

    def test_with_move_before_matmul(self):
        """Realistic pattern: load → move(Left) → move(Right) → matmul → store."""

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
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_l1, target_memory=pl.MemorySpace.Right)
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_left, y_right)
                z_vec: pl.Tile[[16, 128], pl.FP32] = pl.exp(z_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
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

        After = passes.expand_mixed_kernel()(Before)

        # move is VECTOR: load+move should be in AIV, matmul in AIC
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "tile.load(" in aiv_str
        assert "tile.move(" in aiv_str
        assert "exp" in aiv_str

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "matmul" in aic_str
        assert "tile.move(" not in aic_str
        assert "tile.load(" not in aic_str

        # Verify DN layout and transpose metadata are preserved in split functions
        main_func = After.get_function("main")
        assert main_func is not None
        main_str = ir.python_print(main_func)
        assert "TensorLayout.DN" in main_str
        assert "transpose=True" in aiv_str


class TestExpandMixedKernelMultipleInCore:
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
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.InCore)
            def compute_b_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(a, [0, 0], [16, 128])
                b_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(b, [0, 0], [128, 128])
                c_tile: pl.Tile[[16, 128], pl.FP32] = pl.gemv(a_tile, b_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(c_tile, [0, 0], out_0)
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

        After = passes.expand_mixed_kernel()(Before)

        # Both should be split
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
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.pure_incore_0(x, out_0)
                return y

        After = passes.expand_mixed_kernel()(Before)

        # Pure InCore should remain unchanged
        pure_func = After.get_function("pure_incore_0")
        assert pure_func is not None
        assert pure_func.func_type == pl.FunctionType.InCore

        # Mixed InCore should be split
        assert After.get_function("mixed_incore_0_aic") is not None
        assert After.get_function("mixed_incore_0_aiv") is not None
        mixed_group = After.get_function("mixed_incore_0")
        assert mixed_group is not None
        assert mixed_group.func_type == pl.FunctionType.Group

    def test_function_count_after_split(self):
        """After splitting 1 mixed InCore: original 2 funcs → 4 (AIC + AIV + Group + Orch)."""

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
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
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

        assert len(Before.functions) == 2

        After = passes.expand_mixed_kernel()(Before)
        # 1 InCore (mixed) → 3 (AIC + AIV + Group) + 1 Orch = 4 total
        assert len(After.functions) == 4


class TestExpandMixedKernelDeadCodeElimination:
    """Test that DCE removes unused SHARED statements from AIC/AIV."""

    def test_unused_vector_var_not_in_aic(self):
        """Vector var that's only used by other vector ops shouldn't appear in AIC."""

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
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
                z_exp: pl.Tile[[16, 128], pl.FP32] = pl.exp(z_tile)
                z_add: pl.Tile[[16, 128], pl.FP32] = pl.add(z_exp, z_exp)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_add, [0, 0], out_0)
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

        After = passes.expand_mixed_kernel()(Before)

        # AIC should only have matmul + tpush, no vector ops
        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "matmul" in aic_str
        assert "exp" not in aic_str
        assert "add" not in aic_str
        assert "tile.store(" not in aic_str


class TestExpandMixedKernelPropertyVerification:
    """Test property verification behavior with ExpandMixedKernel."""

    def test_pass_produces_mixed_kernel_expanded_property(self):
        """After pass runs, MixedKernelExpanded property should be verifiable."""

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
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
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

        After = passes.expand_mixed_kernel()(Before)

        # Verify the MixedKernelExpanded property (should not raise)
        prop_set = passes.IRPropertySet()
        prop_set.insert(passes.IRProperty.MixedKernelExpanded)
        passes.verify_properties(prop_set, After, "test")

    def test_verification_with_after_mode_instrument(self):
        """Property verification instrument works after expand."""

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
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
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

        # Run with verification instrument — should not raise
        instrument = passes.VerificationInstrument(passes.VerificationMode.AFTER)
        with passes.PassContext([instrument]):
            After = passes.expand_mixed_kernel()(Before)

        assert After.get_function("main_incore_0_aic") is not None


class TestExpandMixedKernelEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pure_cube_incore_unchanged(self):
        """InCore with only cube ops and no vector ops → not mixed, no split.

        NOTE: In practice, a pure-cube InCore is unlikely (no load/store) but
        the pass should handle it gracefully by not splitting.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tile[[16, 128], pl.BF16],
                y: pl.Tile[[128, 128], pl.BF16],
            ):
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x, y)  # noqa: F841  # DSL statement

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                return x

        After = passes.expand_mixed_kernel()(Before)

        # Should not be split since there are no VECTOR ops
        func = After.get_function("main_incore_0")
        assert func is not None
        assert func.func_type == pl.FunctionType.InCore

    def test_aiv_idx_is_zero(self):
        """All TPUSH/TPOP should use aiv_idx=0."""

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
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
                w_tile: pl.Tile[[16, 128], pl.FP32] = pl.exp(z_tile)
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

        After = passes.expand_mixed_kernel()(Before)
        result_str = After.as_python()

        # All cross-core ops should have aiv_idx=0
        assert "aiv_idx=0" in result_str
        # No other aiv_idx values
        aiv_idx_vals = re.findall(r"aiv_idx=(\d+)", result_str)
        assert all(v == "0" for v in aiv_idx_vals)

    def test_two_matmuls_stay_in_aic(self):
        """Multiple CUBE ops in one InCore → all go to AIC, none duplicated."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(a, [0, 0], [16, 128])
                b_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(b, [0, 0], [128, 128])
                c_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(a_tile, b_tile)
                d_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul_acc(c_tile, a_tile, b_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(d_tile, [0, 0], out_0)
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

        After = passes.expand_mixed_kernel()(Before)

        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "tile.matmul(" in aic_str
        assert "matmul_acc" in aic_str

        # No matmul in AIV
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "matmul" not in aiv_str

    def test_existing_group_from_cluster_outline(self):
        """Group function from OutlineClusterScopes passes through unchanged.

        When OutlineClusterScopes has already created a Group function calling
        an InCore, the pass should split the InCore but leave the Group alone.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def compute_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
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

        After = passes.expand_mixed_kernel()(Before)

        # The InCore should be split
        assert After.get_function("compute_incore_0_aic") is not None
        assert After.get_function("compute_incore_0_aiv") is not None

        # The new Group function replaces the original InCore name
        new_group = After.get_function("compute_incore_0")
        assert new_group is not None
        assert new_group.func_type == pl.FunctionType.Group

        # The existing outer Group from cluster outline should remain
        outer_group = After.get_function("compute_group")
        assert outer_group is not None
        assert outer_group.func_type == pl.FunctionType.Group


class TestExpandMixedKernelNestedStructures:
    """Test that mixed ops inside ForStmt/IfStmt are handled recursively."""

    def test_mixed_ops_inside_for_loop(self):
        """Mixed ops inside a for loop → both AIC and AIV get the loop with pruned body."""

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
                    y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                    z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
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

        After = passes.expand_mixed_kernel()(Before)

        # Should produce AIC, AIV, Group
        aic_func = After.get_function("main_incore_0_aic")
        aiv_func = After.get_function("main_incore_0_aiv")
        group_func = After.get_function("main_incore_0")

        assert aic_func is not None
        assert aiv_func is not None
        assert group_func is not None

        assert aic_func.func_type == pl.FunctionType.AIC
        assert aiv_func.func_type == pl.FunctionType.AIV
        assert group_func.func_type == pl.FunctionType.Group

        # AIC should have for loop with matmul but not load/store
        aic_str = aic_func.as_python()
        assert "matmul" in aic_str
        assert "tile.load(" not in aic_str
        assert "tile.store(" not in aic_str
        assert "pl.range" in aic_str  # Loop should be preserved

        # AIV should have for loop with load/store but not matmul
        aiv_str = aiv_func.as_python()
        assert "tile.load(" in aiv_str
        assert "tile.store(" in aiv_str
        assert "tile.matmul(" not in aiv_str
        assert "pl.range" in aiv_str  # Loop should be preserved

    def test_boundaries_inside_for_loop(self):
        """Cross-core boundaries detected inside loops produce TPUSH/TPOP inside the loop."""

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
                    y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                    z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
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

        After = passes.expand_mixed_kernel()(Before)

        # AIV should have tpush_to_aic inside the loop (V→C: load tiles to matmul)
        aiv_func = After.get_function("main_incore_0_aiv")
        assert aiv_func is not None
        aiv_str = aiv_func.as_python()
        assert "tpush_to_aic" in aiv_str

        # AIC should have tpop_from_aiv inside the loop (receiving loaded tiles)
        aic_func = After.get_function("main_incore_0_aic")
        assert aic_func is not None
        aic_str = aic_func.as_python()
        assert "tpop_from_aiv" in aic_str

    def test_bidirectional_inside_for_loop(self):
        """V→C and C→V boundaries inside same loop body."""

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
                    y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                    z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
                    w_tile: pl.Tile[[16, 128], pl.FP32] = pl.exp(z_tile)
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

        After = passes.expand_mixed_kernel()(Before)
        result_str = After.as_python()

        # Both directions present in the overall output
        assert "tpush_to_aic" in result_str  # V→C: load tiles to matmul
        assert "tpop_from_aiv" in result_str  # V→C: matmul receives from load
        assert "tpush_to_aiv" in result_str  # C→V: matmul result to exp
        assert "tpop_from_aic" in result_str  # C→V: exp receives from matmul

    def test_pure_vector_inside_loop_unchanged(self):
        """InCore with only vector ops inside a loop should not be split."""

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

        After = passes.expand_mixed_kernel()(Before)
        ir.assert_structural_equal(After, Before)

    def test_mixed_loop_plus_flat_ops(self):
        """Mixed loop at top level plus flat ops should all be handled."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                # Flat vector op before the loop
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
                for i in pl.range(2):
                    y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                    z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
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

        After = passes.expand_mixed_kernel()(Before)

        aic_func = After.get_function("main_incore_0_aic")
        aiv_func = After.get_function("main_incore_0_aiv")

        assert aic_func is not None
        assert aiv_func is not None
        assert aic_func.func_type == pl.FunctionType.AIC
        assert aiv_func.func_type == pl.FunctionType.AIV

        # AIC should have the loop with matmul
        aic_str = aic_func.as_python()
        assert "matmul" in aic_str
        assert "pl.range" in aic_str

        # AIV should have the top-level load AND the loop with load/store
        aiv_str = aiv_func.as_python()
        assert "tile.load(" in aiv_str
        assert "pl.range" in aiv_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
