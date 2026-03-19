# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for elementwise operations on 4D tiles.

These tests exercise the FlattenTileNdTo2D pass end-to-end: programs are
written with 4D tile shapes which the pass flattens to 2D before code
generation.  Shape [2, 3, 8, 64] flattens to [48, 64].
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# --- Programs (partial coverage) ---


@pl.program
class Tile4DMulPartialProgram:
    """Partial-coverage 4D tile: load first half of dim-0, store to second half.

    Tensor shape [4, 3, 8, 64]; tile shape [2, 3, 8, 64] (half of dim-0).
    Exercises the case where tile.store offset is non-zero: the partition_view
    sizes must reflect the tile shape [2,3,8,64], NOT the full tensor [4,3,8,64].
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[4, 3, 8, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[4, 3, 8, 64], pl.FP32]],
    ) -> pl.Tensor[[4, 3, 8, 64], pl.FP32]:
        # Load first half of outer dim: [0:2, 0:3, 0:8, 0:64]
        a_tile = pl.load(a, [0, 0, 0, 0], [2, 3, 8, 64])
        c_tile = pl.tile.mul(a_tile, a_tile)
        # Store to second half: offset [2,0,0,0], tile covers [2,3,8,64] elements
        out = pl.store(c_tile, [2, 0, 0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[4, 3, 8, 64], pl.FP32],
    ) -> pl.Tensor[[4, 3, 8, 64], pl.FP32]:
        out = pl.create_tensor([4, 3, 8, 64], dtype=pl.FP32)
        out = self.kernel(a, out)
        return out


@pl.program
class Tile4DQuadrantProgram:
    """4D tensor [2,2,8,16] divided into 4 blocks of [1,1,8,16].

    Loads the top-right block (offset [0,1,0,0]), squares it, then stores
    the result into the bottom-left block (offset [1,0,0,0]).

    This is the key partial-coverage test: the tile [1,1,8,16] flattens to
    [8,16], but the store offset [1,0,0,0] is non-zero in dim-0.  The
    partition_view sizes for the store must be [1,1,8,16] (tile shape), NOT
    [2,2,8,16] (full tensor shape).  With the wrong sizes, offset[0]+size[0]
    = 1+2 = 3 > 2, which is out-of-bounds and produces incorrect results.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[2, 2, 8, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[2, 2, 8, 16], pl.FP32]],
    ) -> pl.Tensor[[2, 2, 8, 16], pl.FP32]:
        # Load top-right quadrant: a[0, 1, :, :]
        tile = pl.load(a, [0, 1, 0, 0], [1, 1, 8, 16])
        result_tile = pl.tile.mul(tile, tile)
        # Store to bottom-left quadrant: out[1, 0, :, :]
        out = pl.store(result_tile, [1, 0, 0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[2, 2, 8, 16], pl.FP32],
    ) -> pl.Tensor[[2, 2, 8, 16], pl.FP32]:
        out = pl.create_tensor([2, 2, 8, 16], dtype=pl.FP32)
        out = self.kernel(a, out)
        return out


@pl.program
class Tile4DTopToBottomProgram:
    """4D tensor [2,2,8,16] divided into 4 blocks of [1,1,8,16].

    Computes a*b for the entire top row via a single [1,2,8,16] tile and
    stores the result into the bottom row:
      a[0,:,:,:] * b[0,:,:,:] -> out[1,:,:,:]

    Uses a single [1,2,8,16] tile (load offset [0,0,0,0], store offset
    [1,0,0,0]).  The mul op lets ResolveBackendOpLayouts infer TileView;
    the store offset is non-zero in dim-0 so partition_view sizes must be
    [1,2,8,16] (tile shape), not [2,2,8,16] (full tensor shape).
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[2, 2, 8, 16], pl.FP32],
        b: pl.Tensor[[2, 2, 8, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[2, 2, 8, 16], pl.FP32]],
    ) -> pl.Tensor[[2, 2, 8, 16], pl.FP32]:
        # Load entire top row as one tile: a[0, :, :, :] and b[0, :, :, :]
        a_tile = pl.load(a, [0, 0, 0, 0], [1, 2, 8, 16])
        b_tile = pl.load(b, [0, 0, 0, 0], [1, 2, 8, 16])
        # Multiply element-wise so ResolveBackendOpLayouts can infer TileView
        result_tile = pl.tile.mul(a_tile, b_tile)
        # Store to bottom row in one shot: out[1, :, :, :]
        out = pl.store(result_tile, [1, 0, 0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[2, 2, 8, 16], pl.FP32],
        b: pl.Tensor[[2, 2, 8, 16], pl.FP32],
    ) -> pl.Tensor[[2, 2, 8, 16], pl.FP32]:
        out = pl.create_tensor([2, 2, 8, 16], dtype=pl.FP32)
        out = self.kernel(a, b, out)
        return out


@pl.program
class Tile2DStoreTo3DProgram:
    """2D tile [1, 16] mul then stored into a 3D tensor [2, 4, 16].

    The tile is natively 2D — no ND tile involved, so FlattenTileNdTo2D
    would previously skip injecting the shapes tuple (it only checked tile rank).
    This test verifies the fix: shapes are injected based on tensor rank.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[4, 16], pl.FP32],
        b: pl.Tensor[[4, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[2, 4, 16], pl.FP32]],
    ) -> pl.Tensor[[2, 4, 16], pl.FP32]:
        # Load row 0 from each input: natively 2D tiles [1, 16]
        a_tile = pl.load(a, [0, 0], [1, 16])
        b_tile = pl.load(b, [0, 0], [1, 16])
        c_tile = pl.tile.mul(a_tile, b_tile)
        # Store into slot [1, 2, 0] of the 3D tensor
        out = pl.store(c_tile, [1, 2, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[4, 16], pl.FP32],
        b: pl.Tensor[[4, 16], pl.FP32],
    ) -> pl.Tensor[[2, 4, 16], pl.FP32]:
        out = pl.create_tensor([2, 4, 16], dtype=pl.FP32)
        out = self.kernel(a, b, out)
        return out


# --- Test Cases ---


class Tile4DMulPartialTestCase(PTOTestCase):
    """4D tile partial coverage: tile [2,3,8,64] stores to offset [2,0,0,0] of a [4,3,8,64] tensor."""

    def get_name(self) -> str:
        return "tile_4d_mul_partial"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [4, 3, 8, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [4, 3, 8, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Tile4DMulPartialProgram

    def compute_expected(self, tensors, params=None):
        # First half of dim-0 is unsquared (out[:2] unchanged / zero-initialized)
        # Second half of dim-0 = a[:2, ...] ** 2
        tensors["out"][2:, ...] = tensors["a"][:2, ...] * tensors["a"][:2, ...]


class Tile4DTopToBottomTestCase(PTOTestCase):
    """4D tensor [2,2,8,16]; mul top row a*b via one [1,2,8,16] tile, store to bottom row."""

    def get_name(self) -> str:
        return "tile_4d_top_to_bottom"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [2, 2, 8, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [2, 2, 8, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [2, 2, 8, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Tile4DTopToBottomProgram

    def compute_expected(self, tensors, params=None):
        # Only bottom row is written; top row stays zero-initialized.
        tensors["out"][1] = tensors["a"][0] * tensors["b"][0]


class Tile4DQuadrantTestCase(PTOTestCase):
    """4D tensor [2,2,8,16] split into 4 blocks; load top-right, store squared to bottom-left."""

    def get_name(self) -> str:
        return "tile_4d_quadrant"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [2, 2, 8, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [2, 2, 8, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Tile4DQuadrantProgram

    def compute_expected(self, tensors, params=None):
        # Only bottom-left quadrant is written; the rest stays zero-initialized.
        tensors["out"][1, 0] = tensors["a"][0, 1] ** 2


class Tile2DStoreTo3DTestCase(PTOTestCase):
    """2D tile [1, 16] mul then stored into a 3D tensor [2, 4, 16].

    Verifies that FlattenTileNdTo2D injects the correct shapes tuple [1, 1, 16]
    (tile coverage left-padded to tensor rank) rather than the full tensor shape
    [2, 4, 16]. Before the fix, this would crash with
    'tile.store on ND tensor requires shapes tuple (args[3])'.
    """

    def get_name(self) -> str:
        return "tile_2d_store_to_3d"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [4, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [4, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [2, 4, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Tile2DStoreTo3DProgram

    def compute_expected(self, tensors, params=None):
        tensors["out"][1, 2, :] = tensors["a"][0, :] * tensors["b"][0, :]


# --- Tests ---


class TestElementwise4D:
    """End-to-end tests for elementwise ops on 4D tiles (exercises FlattenTileNdTo2D pass)."""

    def test_tile_4d_top_to_bottom(self, test_runner):
        """4D tensor [2,2,8,16]; a*b on top row via a single [1,2,8,16] tile, store to bottom row.

        Loads a[0,:,:,:] and b[0,:,:,:] with tile shape [1,2,8,16] from offset
        [0,0,0,0], multiplies them, and stores to out[1,:,:,:] at offset
        [1,0,0,0].  Partition_view sizes for the store must be [1,2,8,16]
        (tile shape), not [2,2,8,16] (full tensor shape).
        """
        test_case = Tile4DTopToBottomTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_4d_quadrant(self, test_runner):
        """4D tensor [2,2,8,16] divided into 4 blocks of [1,1,8,16].

        Loads top-right block a[0,1,:,:], squares it, stores to bottom-left
        block out[1,0,:,:].  Partition_view sizes for the store must be
        [1,1,8,16] (tile shape).  With the current bug, sizes=[2,2,8,16] and
        offset[0]=1 gives offset+size=3 > tensor_dim=2 (out of bounds).
        """
        test_case = Tile4DQuadrantTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_4d_mul_partial(self, test_runner):
        """Partial-coverage 4D tile store: tile [2,3,8,64] at offset [2,0,0,0] of [4,3,8,64] tensor.

        Verifies that partition_view sizes match the tile shape [2,3,8,64] (not the full
        tensor [4,3,8,64]). With the bug, sizes=[4,3,8,64] and offset=[2,0,0,0] would
        produce offset+size=[6,...] > tensor_dim[4,...], causing ptoas to reject the IR.
        """
        test_case = Tile4DMulPartialTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_2d_store_to_3d(self, test_runner):
        """2D tile [1, 16] stored into a 3D tensor [2, 4, 16].

        Regression test for the bug where FlattenTileNdTo2D only injected the
        shapes tuple when the *tile* was ND, missing the case where the tile is
        natively 2D but the output tensor is ND (rank > 2).
        """
        test_case = Tile2DStoreTo3DTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
