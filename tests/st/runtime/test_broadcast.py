# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile row/col broadcast operations using the PyPTO frontend.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec

MN_CASES: list[tuple[int, int]] = [(8, 16), (16, 16), (16, 8)]


class TestTileRowExpand(PTOTestCase):
    """Test case for tile.row_expand."""

    __test__ = False

    def __init__(self, m: int = 16, n: int = 16, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M = m
        self.N = n

    def get_name(self) -> str:
        return f"tile_row_expand_{self.M}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("row_vec", [self.M, 1], DataType.FP32, init_value=torch.randn),
            TensorSpec("y", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, N = self.M, self.N

        @pl.program
        class TileRowExpandProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def row_expand_kernel(
                self,
                row_vec: pl.Tensor[[M, 1], pl.FP32],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                row_tile: pl.Tile[[M, 1], pl.FP32] = pl.load(row_vec, [0, 0], [M, 1])
                target_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.create([M, N], dtype=pl.FP32)
                expanded: pl.Tile[[M, N], pl.FP32] = pl.tile.row_expand(target_tile, row_tile)
                out: pl.Tensor[[M, N], pl.FP32] = pl.store(expanded, [0, 0], y)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                row_vec: pl.Tensor[[M, 1], pl.FP32],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                y = self.row_expand_kernel(row_vec, y)
                return y

        return TileRowExpandProgram

    def compute_expected(self, tensors, params=None):
        tensors["y"][:] = tensors["row_vec"].repeat(1, self.N)


class TestTileColExpand(PTOTestCase):
    """Test case for tile.col_expand."""

    __test__ = False

    def __init__(self, m: int = 16, n: int = 16, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M = m
        self.N = n

    def get_name(self) -> str:
        return f"tile_col_expand_{self.M}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("col_vec", [1, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("y", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, N = self.M, self.N

        @pl.program
        class TileColExpandProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def col_expand_kernel(
                self,
                col_vec: pl.Tensor[[1, N], pl.FP32],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                col_tile: pl.Tile[[1, N], pl.FP32] = pl.load(col_vec, [0, 0], [1, N])
                target_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.create([M, N], dtype=pl.FP32)
                expanded: pl.Tile[[M, N], pl.FP32] = pl.tile.col_expand(target_tile, col_tile)
                out: pl.Tensor[[M, N], pl.FP32] = pl.store(expanded, [0, 0], y)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                col_vec: pl.Tensor[[1, N], pl.FP32],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                y = self.col_expand_kernel(col_vec, y)
                return y

        return TileColExpandProgram

    def compute_expected(self, tensors, params=None):
        tensors["y"][:] = tensors["col_vec"].repeat(self.M, 1)


class TestTensorExpandClone(PTOTestCase):
    """Test case for tensor.expand_clone."""

    __test__ = False

    def __init__(
        self,
        b: int = 8,
        n: int = 8,
        k: int = 8,
        broadcast_dim: int = 0,
        *,
        platform: str | None = None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        self.B = b
        self.N = n
        self.K = k
        self.broadcast_dim = broadcast_dim

    def get_name(self) -> str:
        return f"tensor_expand_clone_d{self.broadcast_dim}_{self.B}x{self.N}x{self.K}"

    def _input_shape(self) -> list[int]:
        if self.broadcast_dim == -1:
            return [self.B, self.N, self.K]
        if self.broadcast_dim == 0:
            return [1, self.N, self.K]
        if self.broadcast_dim == 1:
            return [self.B, 1, self.K]
        if self.broadcast_dim == 2:
            return [self.B, self.N, 1]
        raise ValueError(f"Unsupported broadcast_dim: {self.broadcast_dim}")

    def define_tensors(self) -> list[TensorSpec]:
        input_shape = self._input_shape()
        return [
            TensorSpec("x", input_shape, DataType.FP32, init_value=torch.randn),
            TensorSpec("y", [self.B, self.N, self.K], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        B, N, K = self.B, self.N, self.K
        input_shape = self._input_shape()

        @pl.program
        class ExpandCloneProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def expand_clone_kernel(
                self,
                x: pl.Tensor[input_shape, pl.FP32],
                y: pl.Out[pl.Tensor[[B, N, K], pl.FP32]],
            ) -> pl.Tensor[[B, N, K], pl.FP32]:
                out: pl.Tensor[[B, N, K], pl.FP32] = pl.expand_clone(x, y)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[input_shape, pl.FP32],
                y: pl.Out[pl.Tensor[[B, N, K], pl.FP32]],
            ) -> pl.Tensor[[B, N, K], pl.FP32]:
                y = self.expand_clone_kernel(x, y)
                return y

        return ExpandCloneProgram

    def compute_expected(self, tensors, params=None):
        if self.broadcast_dim == -1:
            tensors["y"][:] = tensors["x"]
        elif self.broadcast_dim == 0:
            tensors["y"][:] = tensors["x"].repeat(self.B, 1, 1)
        elif self.broadcast_dim == 1:
            tensors["y"][:] = tensors["x"].repeat(1, self.N, 1)
        elif self.broadcast_dim == 2:
            tensors["y"][:] = tensors["x"].repeat(1, 1, self.K)
        else:
            raise ValueError(f"Unsupported broadcast_dim: {self.broadcast_dim}")


class TestRowExpandMulFromNdColumnSlice(PTOTestCase):
    """Regression for #1230: row_expand_mul with a column slice of a non-column-vector ND tensor.

    Reproduces the ``pl.slice(post_2d, [T, 1], [0, h])`` pattern that used to be
    lowered to a ``ColMajor`` Vec tile from an ``ND`` GM view, failing the
    runtime TLOAD ``isSameLayout`` check (ND2ND/DN2DN/NZ2NZ only) on real a2a3.
    With the fix, the slice keeps RowMajor and rides ND2ND.
    """

    __test__ = False

    def __init__(
        self,
        t: int = 16,
        d: int = 64,
        hc: int = 4,
        col_idx: int = 2,
        *,
        platform: str | None = None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        self.T = t
        self.D = d
        self.HC = hc
        self.col_idx = col_idx

    def get_name(self) -> str:
        return f"row_expand_mul_nd_col_slice_T{self.T}_D{self.D}_HC{self.HC}_h{self.col_idx}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [self.T, self.D], DataType.FP32, init_value=torch.randn),
            TensorSpec("w", [self.T, self.HC], DataType.FP32, init_value=torch.randn),
            TensorSpec("y", [self.T, self.D], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        T, D, HC = self.T, self.D, self.HC
        col_idx = self.col_idx

        @pl.program
        class RowExpandMulFromNdColumnSliceProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[T, D], pl.FP32],
                w: pl.Tensor[[T, HC], pl.FP32],
                y: pl.Out[pl.Tensor[[T, D], pl.FP32]],
            ) -> pl.Tensor[[T, D], pl.FP32]:
                x_tile: pl.Tile[[T, D], pl.FP32] = pl.load(x, [0, 0], [T, D])
                # The bug: loading a [T, 1] slice from a non-column-vector ND
                # tensor previously emitted a ColMajor tile.load against an ND
                # GM view, which the a2a3 runtime rejects at incore compile
                # time. (Equivalent to the issue's ``pl.slice(post_2d, [T, 1],
                # ...)`` pattern after ConvertTensorToTileOps lowering.)
                w_col: pl.Tile[[T, 1], pl.FP32] = pl.load(w, [0, col_idx], [T, 1])
                y_tile: pl.Tile[[T, D], pl.FP32] = pl.tile.row_expand_mul(x_tile, w_col)
                out: pl.Tensor[[T, D], pl.FP32] = pl.store(y_tile, [0, 0], y)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[T, D], pl.FP32],
                w: pl.Tensor[[T, HC], pl.FP32],
                y: pl.Out[pl.Tensor[[T, D], pl.FP32]],
            ) -> pl.Tensor[[T, D], pl.FP32]:
                y = self.kernel(x, w, y)
                return y

        return RowExpandMulFromNdColumnSliceProgram

    def compute_expected(self, tensors, params=None):
        # y[i, j] = x[i, j] * w[i, col_idx]
        tensors["y"][:] = tensors["x"] * tensors["w"][:, self.col_idx : self.col_idx + 1]


class TestBroadcastOperations:
    """Test suite for tile broadcast operations."""

    @pytest.mark.parametrize("m, n", MN_CASES)
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_tile_row_expand(self, test_runner, platform, m, n):
        """Test tile.row_expand across platforms."""
        result = test_runner.run(TestTileRowExpand(m=m, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("m, n", MN_CASES)
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_tile_col_expand(self, test_runner, platform, m, n):
        """Test tile.col_expand across platforms."""
        result = test_runner.run(TestTileColExpand(m=m, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("broadcast_dim", [-1, 0, 1, 2])
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_tensor_expand_clone(self, test_runner, platform, broadcast_dim):
        """Test tensor.expand_clone across platforms."""
        if platform in ("a5", "a5sim") and broadcast_dim == 2:
            pytest.skip("Skip broadcast_dim=2 for Ascend 950 due to pto-isa bug.")
        result = test_runner.run(TestTensorExpandClone(broadcast_dim=broadcast_dim, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("col_idx", [0, 1, 2, 3])
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_row_expand_mul_from_nd_column_slice(self, test_runner, platform, col_idx):
        """Regression for #1230: row_expand_mul fed by a column slice of a [T, HC] ND tensor.

        Sweep ``col_idx`` so the inner offset is non-zero (stresses partition_view
        ND2ND TLOAD on a non-leading column).
        """
        result = test_runner.run(TestRowExpandMulFromNdColumnSlice(col_idx=col_idx, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
