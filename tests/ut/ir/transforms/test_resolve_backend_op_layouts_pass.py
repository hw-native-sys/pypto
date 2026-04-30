# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ResolveBackendOpLayouts pass."""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType


def _run_pass(program):
    """Run ResolveBackendOpLayouts with the Ascend910B backend active."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    try:
        return passes.resolve_backend_op_layouts()(program)
    finally:
        backend.reset_for_testing()


class TestResolveBackendOpLayouts:
    """Test backend-driven layout repair for constrained tile ops.

    On Ascend910B, elementwise tile ops on `[N, 1]` column vectors are
    repaired by reshaping to `[1, N]` row-major before the op and
    reshaping back to `[N, 1]` afterwards.
    """

    def test_rewrites_column_vector_add_through_row_major_reshape(self):
        """`tile.add` on `[N, 1]` vectors should be repaired through `[1, N] row_major`."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def repro(
                self,
                data: pl.Tensor[[16, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                acc_0: pl.Tile[[16, 1], pl.FP32] = pl.tile.create(
                    [16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                acc_1: pl.Tile[[16, 1], pl.FP32] = pl.tile.muls(acc_0, 0.0)
                chunk: pl.Tile[[16, 256], pl.FP32] = pl.load(data, [0, 0], [16, 256])
                tmp: pl.Tile[[16, 256], pl.FP32] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                partial: pl.Tile[[16, 1], pl.FP32] = pl.tile.row_sum(chunk, tmp)
                updated: pl.Tile[[16, 1], pl.FP32] = pl.tile.add(acc_1, partial)
                stored: pl.Tensor[[16, 1], pl.FP32] = pl.store(updated, [0, 0], out)
                return stored

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def repro(
                self,
                data: pl.Tensor[[16, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                acc_0: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                acc_0_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(acc_0, [1, 16])
                acc_1_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.muls(acc_0_rm, 0.0)
                acc_1: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(acc_1_rm, [16, 1])
                chunk: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.load(data, [0, 0], [16, 256])
                tmp: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                partial: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.row_sum(chunk, tmp)
                acc_1_rm2: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(acc_1, [1, 16])
                partial_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(partial, [1, 16])
                updated_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.add(acc_1_rm2, partial_rm)
                updated: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(updated_rm, [16, 1])
                stored: pl.Tensor[[16, 1], pl.FP32] = pl.store(updated, [0, 0], out)
                return stored

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_rewrites_column_vector_abs_through_row_major_reshape(self):
        """`tile.abs` (unary) on `[N, 1]` col_major vector should be repaired."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def repro(
                self,
                data: pl.Tensor[[16, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                chunk: pl.Tile[[16, 256], pl.FP32] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tmp: pl.Tile[[16, 256], pl.FP32] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                partial: pl.Tile[[16, 1], pl.FP32] = pl.tile.row_sum(chunk, tmp)
                result: pl.Tile[[16, 1], pl.FP32] = pl.tile.abs(partial)
                stored: pl.Tensor[[16, 1], pl.FP32] = pl.store(result, [0, 0], out)
                return stored

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def repro(
                self,
                data: pl.Tensor[[16, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                chunk: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tmp: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                partial: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.row_sum(chunk, tmp)
                partial_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(partial, [1, 16])
                result_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.abs(partial_rm)
                result: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(result_rm, [16, 1])
                stored: pl.Tensor[[16, 1], pl.FP32] = pl.store(result, [0, 0], out)
                return stored

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_rewrites_column_vector_muls_through_row_major_reshape(self):
        """`tile.muls` (tile x scalar) on `[N, 1]` col_major should repair only the tile input."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def repro(
                self,
                data: pl.Tensor[[16, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                chunk: pl.Tile[[16, 256], pl.FP32] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tmp: pl.Tile[[16, 256], pl.FP32] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                partial: pl.Tile[[16, 1], pl.FP32] = pl.tile.row_sum(chunk, tmp)
                scaled: pl.Tile[[16, 1], pl.FP32] = pl.tile.muls(partial, 2.0)
                stored: pl.Tensor[[16, 1], pl.FP32] = pl.store(scaled, [0, 0], out)
                return stored

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def repro(
                self,
                data: pl.Tensor[[16, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                chunk: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tmp: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                partial: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.row_sum(chunk, tmp)
                partial_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(partial, [1, 16])
                scaled_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.muls(partial_rm, 2.0)
                scaled: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(scaled_rm, [16, 1])
                stored: pl.Tensor[[16, 1], pl.FP32] = pl.store(scaled, [0, 0], out)
                return stored

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_rewrites_matrix_exp_through_row_major_move(self):
        """`tile.exp` on a non-vector col_major tile should be repaired through `tile.move`."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def repro(
                self,
                out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
            ) -> pl.Tensor[[16, 256], pl.FP32]:
                src: pl.Tile[[16, 256], pl.FP32] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                col_major: pl.Tile[
                    [16, 256],
                    pl.FP32,
                    pl.MemorySpace.Vec,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.tile.move(
                    src,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                result: pl.Tile[
                    [16, 256],
                    pl.FP32,
                    pl.MemorySpace.Vec,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.tile.exp(col_major)
                stored: pl.Tensor[[16, 256], pl.FP32] = pl.store(result, [0, 0], out)
                return stored

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def repro(
                self,
                out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
            ) -> pl.Tensor[[16, 256], pl.FP32]:
                src: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                col_major: pl.Tile[
                    [16, 256],
                    pl.FP32,
                    pl.MemorySpace.Vec,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.tile.move(
                    src,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                col_major_rm: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.tile.move(
                    col_major,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                result_rm: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.tile.exp(col_major_rm)
                result: pl.Tile[
                    [16, 256],
                    pl.FP32,
                    pl.MemorySpace.Vec,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.tile.move(
                    result_rm,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                )
                stored: pl.Tensor[[16, 256], pl.FP32] = pl.store(result, [0, 0], out)
                return stored

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
