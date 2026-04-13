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
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType

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
        a_tile = pl.load(a, [0, 0, 0, 0], [2, 3, 8, 64])
        c_tile = pl.tile.mul(a_tile, a_tile)
        out = pl.store(c_tile, [2, 0, 0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[4, 3, 8, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[4, 3, 8, 64], pl.FP32]],
    ) -> pl.Tensor[[4, 3, 8, 64], pl.FP32]:
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
        tile = pl.load(a, [0, 1, 0, 0], [1, 1, 8, 16])
        result_tile = pl.tile.mul(tile, tile)
        out = pl.store(result_tile, [1, 0, 0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[2, 2, 8, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[2, 2, 8, 16], pl.FP32]],
    ) -> pl.Tensor[[2, 2, 8, 16], pl.FP32]:
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
        a_tile = pl.load(a, [0, 0, 0, 0], [1, 2, 8, 16])
        b_tile = pl.load(b, [0, 0, 0, 0], [1, 2, 8, 16])
        result_tile = pl.tile.mul(a_tile, b_tile)
        out = pl.store(result_tile, [1, 0, 0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[2, 2, 8, 16], pl.FP32],
        b: pl.Tensor[[2, 2, 8, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[2, 2, 8, 16], pl.FP32]],
    ) -> pl.Tensor[[2, 2, 8, 16], pl.FP32]:
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
        a_tile = pl.load(a, [0, 0], [1, 16])
        b_tile = pl.load(b, [0, 0], [1, 16])
        c_tile = pl.tile.mul(a_tile, b_tile)
        out = pl.store(c_tile, [1, 2, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[4, 16], pl.FP32],
        b: pl.Tensor[[4, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[2, 4, 16], pl.FP32]],
    ) -> pl.Tensor[[2, 4, 16], pl.FP32]:
        out = self.kernel(a, b, out)
        return out


# --- Test Cases ---


class Tile4DMulPartialTestCase(PTOTestCase):
    """4D tile partial coverage: tile [2,3,8,64] stores to offset [2,0,0,0] of a [4,3,8,64] tensor."""

    __test__ = False

    def __init__(self, *, backend_type: BackendType | None = None, config=None):
        super().__init__(config, backend_type=backend_type)

    def get_name(self) -> str:
        return "tile_4d_mul_partial"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [4, 3, 8, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [4, 3, 8, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Tile4DMulPartialProgram

    def compute_expected(self, tensors, params=None):
        tensors["out"][2:, ...] = tensors["a"][:2, ...] * tensors["a"][:2, ...]


class Tile4DTopToBottomTestCase(PTOTestCase):
    """4D tensor [2,2,8,16]; mul top row a*b via one [1,2,8,16] tile, store to bottom row."""

    __test__ = False

    def __init__(self, *, backend_type: BackendType | None = None, config=None):
        super().__init__(config, backend_type=backend_type)

    def get_name(self) -> str:
        return "tile_4d_top_to_bottom"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [2, 2, 8, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [2, 2, 8, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [2, 2, 8, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Tile4DTopToBottomProgram

    def compute_expected(self, tensors, params=None):
        tensors["out"][1] = tensors["a"][0] * tensors["b"][0]


class Tile4DQuadrantTestCase(PTOTestCase):
    """4D tensor [2,2,8,16] split into 4 blocks; load top-right, store squared to bottom-left."""

    __test__ = False

    def __init__(self, *, backend_type: BackendType | None = None, config=None):
        super().__init__(config, backend_type=backend_type)

    def get_name(self) -> str:
        return "tile_4d_quadrant"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [2, 2, 8, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [2, 2, 8, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return Tile4DQuadrantProgram

    def compute_expected(self, tensors, params=None):
        tensors["out"][1, 0] = tensors["a"][0, 1] ** 2


class Tile2DStoreTo3DTestCase(PTOTestCase):
    """2D tile [1, 16] mul then stored into a 3D tensor [2, 4, 16].

    Verifies that FlattenTileNdTo2D injects the correct shapes tuple [1, 1, 16]
    (tile coverage left-padded to tensor rank) rather than the full tensor shape
    [2, 4, 16]. Before the fix, this would crash with
    'tile.store on ND tensor requires shapes tuple (args[3])'.
    """

    __test__ = False

    def __init__(self, *, backend_type: BackendType | None = None, config=None):
        super().__init__(config, backend_type=backend_type)

    def get_name(self) -> str:
        return "tile_2d_store_to_3d"

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

    @pytest.mark.parametrize("backend", PLATFORMS)
    def test_tile_4d_top_to_bottom(self, test_runner, backend):
        """4D tensor [2,2,8,16]; a*b on top row via a single [1,2,8,16] tile, store to bottom row."""
        result = test_runner.run(Tile4DTopToBottomTestCase(backend_type=backend))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("backend", PLATFORMS)
    def test_tile_4d_quadrant(self, test_runner, backend):
        """4D tensor [2,2,8,16] divided into 4 blocks of [1,1,8,16]."""
        result = test_runner.run(Tile4DQuadrantTestCase(backend_type=backend))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("backend", PLATFORMS)
    def test_tile_4d_mul_partial(self, test_runner, backend):
        """Partial-coverage 4D tile store: tile [2,3,8,64] at offset [2,0,0,0] of [4,3,8,64] tensor."""
        result = test_runner.run(Tile4DMulPartialTestCase(backend_type=backend))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("backend", PLATFORMS)
    def test_tile_2d_store_to_3d(self, test_runner, backend):
        """2D tile [1, 16] stored into a 3D tensor [2, 4, 16]."""
        result = test_runner.run(Tile2DStoreTo3DTestCase(backend_type=backend))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
