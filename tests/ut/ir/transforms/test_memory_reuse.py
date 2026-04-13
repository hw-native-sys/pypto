# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for MemoryReusePass.

Most tests use the Before/Expected pattern with
``ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)``.
``enable_auto_mapping=True`` aligns MemRef objects consistently across the
comparison: if two tiles share a MemRef in ``After``, the corresponding tiles
in ``Expected`` must also share (i.e. use the same ``mem_*`` pointer name).
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _run_pipeline(program: ir.Program) -> ir.Program:
    """Run init_mem_ref + memory_reuse pipeline, return resulting Program."""
    return passes.memory_reuse()(passes.init_mem_ref()(program))


class TestBasic:
    """Core reuse logic: chain reuse, producer-consumer, size/shape, transitive conflicts."""

    def test_simple(self):
        """tile_c, tile_d, tile_e all chain-reuse tile_a; tile_b remains independent."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                input_b: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_b, [0, 0], [64, 64])
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_b)
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.mul(tile_c, tile_c)
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_d, tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        # tile_a/c/d/e all share mem_vec_3; tile_b uses mem_vec_4 (independent).
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                input_b: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_b, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_b
                )
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.mul(
                    tile_c, tile_c
                )
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_d, tile_d
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)] = pl.tile.store(
                    tile_e, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_sequential(self):
        """Sequential chain: tile_a/c/e share one buffer, tile_b/d share another."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_b, tile_b)
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_c, tile_c)
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_d, tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        # All five tiles end up in mem_vec_2 — full producer-consumer reuse chain
        # collapses everything into a single buffer.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_a
                )
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_b, tile_b
                )
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_c, tile_c
                )
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_d, tile_d
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_e, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_different_sizes(self):
        """Different-shaped tiles cannot reuse each other's buffer."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                input_b: pl.Tensor[[32, 32], pl.FP32],
                output_a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                output_b: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                _result_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_a, [0, 0], output_a)
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_b, [0, 0], [32, 32])
                _result_b: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output_b)
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_f: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_b, [0, 0], [32, 32])
                _result_e: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output_a)
                result_f: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_f, [0, 0], output_b)
                return result_f

        # tile_a/tile_e share mem_vec_4 (16384 bytes). tile_b/tile_f share mem_vec_5
        # (4096 bytes). Different sizes never alias.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                input_b: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_1", 0, 4096)],
                output_a: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)]],
                output_b: pl.Out[pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_3", 0, 4096)]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                _result_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)] = pl.tile.store(
                    tile_a, [0, 0], output_a
                )
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_5, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_b, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                _result_b: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_3", 0, 4096)] = pl.tile.store(
                    tile_b, [0, 0], output_b
                )
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_f: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_5, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_b, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                _result_e: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)] = pl.tile.store(
                    tile_e, [0, 0], output_a
                )
                result_f: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_3", 0, 4096)] = pl.tile.store(
                    tile_f, [0, 0], output_b
                )
                return result_f

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_empty_function(self):
        """Empty function (no TileType) should pass through unchanged."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                return output

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                return output

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_transitive_conflict(self):
        """Transitive conflict: tile_c and tile_d cannot share."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_b, tile_b)
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_c, tile_c)
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_c, tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        # tile_a/b/c/e share mem_vec_2; tile_d gets its own mem_vec_5 because
        # tile_c is still live when tile_d is defined (tile_e reads tile_c).
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_a
                )
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_b, tile_b
                )
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_c, tile_c
                )
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_c, tile_d
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_e, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


class TestAllocCleanup:
    """Tests for redundant tile.alloc removal after memory reuse."""

    def test_unused_alloc_removed_after_reuse(self):
        """Alloc stmts for MemRefs replaced by reuse should be removed.

        Before reuse there are 3 allocs (tile_a/b/c each have one).
        After chain reuse, all three tiles share mem_vec_2 — only one alloc remains.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_b, tile_b)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_a
                )
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_b, tile_b
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_c, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_partial_reuse_with_overlapping_lifetimes(self):
        """When some lifetimes truly overlap, only partial reuse happens.

        tile_a and tile_b are both live at tile_c's def, so tile_b cannot
        reuse tile_a. tile_c reuses tile_a (greedy first-fit). 2 allocs remain.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_b)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_b
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_c, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


class TestDtype:
    """Tests that tiles with different dtypes do NOT reuse each other's memory."""

    def test_cross_dtype_no_reuse_same_dtype_reuse(self):
        """Cross-dtype reuse forbidden; same-dtype tiles reuse within their group."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                tile_cast: pl.Tile[[64, 64], pl.BF16, pl.MemorySpace.Vec] = pl.cast(
                    tile_b, target_type=pl.BF16
                )
                tile_d: pl.Tile[[64, 64], pl.BF16, pl.MemorySpace.Vec] = pl.add(tile_cast, tile_cast)
                tile_e: pl.Tile[[64, 64], pl.BF16, pl.MemorySpace.Vec] = pl.add(tile_d, tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        # FP32 group (tile_a, tile_b) shares mem_vec_2 (16384 bytes).
        # BF16 group (tile_cast, tile_d, tile_e) shares mem_vec_4 (8192 bytes).
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_a
                )
                tile_cast: pl.Tile[[64, 64], pl.BF16, pl.MemRef(mem_vec_4, 0, 8192), pl.Mem.Vec] = (
                    pl.tile.cast(tile_b, target_type=pl.BF16, mode="round")
                )
                tile_d: pl.Tile[[64, 64], pl.BF16, pl.MemRef(mem_vec_4, 0, 8192), pl.Mem.Vec] = pl.tile.add(
                    tile_cast, tile_cast
                )
                tile_e: pl.Tile[[64, 64], pl.BF16, pl.MemRef(mem_vec_4, 0, 8192), pl.Mem.Vec] = pl.tile.add(
                    tile_d, tile_d
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_e, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


class TestFillpad:
    """Tests that fillpad output does NOT reuse input due to TileView differences."""

    def test_fillpad_output_incompatible_with_input(self):
        """fillpad changes valid_shape and pad: output cannot reuse input."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_a, [0, 0], [64, 64], valid_shapes=[48, 64]
                )
                padded: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.fillpad(
                    tile_a, pad_value=pl.PadValue.max
                )
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded, [0, 0], output)
                return result

        # tile_a uses mem_vec_2 (valid_shape=[48, 64]); padded uses mem_vec_3
        # because the TileView changes from valid_shape=[48,64] to a padded view.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_2, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(valid_shape=[48, 64]),
                ] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [48, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                padded: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_3, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(pad=pl.PadValue.max),
                ] = pl.tile.fillpad(tile_a, pad_value=pl.PadValue.max)
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    padded, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_fillpad_different_pad_no_reuse(self):
        """Two fillpad outputs with different pad values cannot reuse each other."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output_a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                output_b: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_a, [0, 0], [64, 64], valid_shapes=[48, 64]
                )
                padded_max: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.fillpad(
                    tile_a, pad_value=pl.PadValue.max
                )
                _res_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_max, [0, 0], output_a)
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_a, [0, 0], [64, 64], valid_shapes=[48, 64]
                )
                padded_min: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.fillpad(
                    tile_b, pad_value=pl.PadValue.min
                )
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_min, [0, 0], output_b)
                return result

        # tile_a/tile_b share mem_vec_3 (same valid_shape view).
        # padded_max uses mem_vec_4 (PadValue.max). padded_min uses mem_vec_6
        # (PadValue.min) — different padding views can't share.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output_a: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
                output_b: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_3, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(valid_shape=[48, 64]),
                ] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [48, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                padded_max: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_4, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(pad=pl.PadValue.max),
                ] = pl.tile.fillpad(tile_a, pad_value=pl.PadValue.max)
                _res_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    padded_max, [0, 0], output_a
                )
                tile_b: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_3, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(valid_shape=[48, 64]),
                ] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [48, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                padded_min: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_6, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(pad=pl.PadValue.min),
                ] = pl.tile.fillpad(tile_b, pad_value=pl.PadValue.min)
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)] = pl.tile.store(
                    padded_min, [0, 0], output_b
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_fillpad_same_pad_can_reuse(self):
        """Two fillpad outputs with identical TileView attributes CAN reuse."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output_a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                output_b: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_a, [0, 0], [64, 64], valid_shapes=[48, 64]
                )
                padded_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.fillpad(
                    tile_a, pad_value=pl.PadValue.max
                )
                _res_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_a, [0, 0], output_a)
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_a, [0, 0], [64, 64], valid_shapes=[48, 64]
                )
                padded_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.fillpad(
                    tile_b, pad_value=pl.PadValue.max
                )
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_b, [0, 0], output_b)
                return result

        # tile_a/tile_b share mem_vec_3 (same view).
        # padded_a/padded_b share mem_vec_4 (same PadValue.max view).
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output_a: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
                output_b: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_3, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(valid_shape=[48, 64]),
                ] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [48, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                padded_a: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_4, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(pad=pl.PadValue.max),
                ] = pl.tile.fillpad(tile_a, pad_value=pl.PadValue.max)
                _res_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    padded_a, [0, 0], output_a
                )
                tile_b: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_3, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(valid_shape=[48, 64]),
                ] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [48, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                padded_b: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_4, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(pad=pl.PadValue.max),
                ] = pl.tile.fillpad(tile_b, pad_value=pl.PadValue.max)
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)] = pl.tile.store(
                    padded_b, [0, 0], output_b
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


class TestViewOps:
    """Tests for view operations (reshape) with memory reuse."""

    def test_reshape_chain_shares_memref(self):
        """Chained reshapes should all share the same MemRef."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[4096, 1], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(tile_a, [4096, 1])
                tile_c: pl.Tile[[1, 4096], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(tile_b, [1, 4096])
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(tile_c, [64, 64])
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[4096, 1], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.reshape(tile_a, [4096, 1])
                )
                tile_c: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.reshape(tile_b, [1, 4096])
                )
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.reshape(tile_c, [64, 64])
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_d, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_reshape_not_broken_by_memory_reuse(self):
        """MemoryReuse should propagate reuse to ALL variables sharing MemRef.

        tile_a and _tile_b share MemRef (reshape = view alias). When tile_a
        is reused with tile_c, _tile_b must also pick up tile_c's MemRef.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                _tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_c, tile_c)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                _tile_b: pl.Tile[[4096, 1], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(tile_a, [4096, 1])
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        # All five tiles end up sharing mem_vec_2 — chain reuse plus view alias propagation.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                _tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_c, tile_c
                )
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                _tile_b: pl.Tile[[4096, 1], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.reshape(tile_a, [4096, 1])
                )
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_a
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_e, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_reshape_shared_buffer_can_be_reused_after_all_dead(self):
        """After all aliases are dead, shared buffer can be reused."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                _tile_b: pl.Tile[[4096, 1], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(tile_a, [4096, 1])
                _tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_d, tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                _tile_b: pl.Tile[[4096, 1], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.reshape(tile_a, [4096, 1])
                )
                _tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_a
                )
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_d, tile_d
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_e, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


class TestInplaceOps:
    """Tests verifying that ops marked not_inplace_safe block producer-consumer reuse."""

    def test_inplace_unsafe_op_no_producer_consumer_reuse(self):
        """tile.recip must NOT reuse its input's buffer."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.recip(tile_a)
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output)
                return result

        # tile_a uses mem_vec_2; tile_b uses mem_vec_3 (recip is inplace-unsafe).
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_0", 0, 4096)],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_1", 0, 4096)]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                tile_a: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_2, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_3, 0, 4096), pl.Mem.Vec] = pl.tile.recip(
                    tile_a
                )
                result: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_1", 0, 4096)] = pl.tile.store(
                    tile_b, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_inplace_unsafe_op_allows_non_producer_consumer_reuse(self):
        """tile.recip output must never share a buffer with its input.

        tile_a/tile_c/tile_x share mem_vec_4 (chain reuse — they're not
        consumed by tile_b's recip). tile_b uses mem_vec_7 (separate buffer
        because recip is inplace-unsafe w.r.t. tile_x).
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP32],
                input_c: pl.Tensor[[32, 32], pl.FP32],
                input_x: pl.Tensor[[32, 32], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                _s1: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_a, [0, 0], output)
                tile_c: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_c, [0, 0], [32, 32])
                _s2: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_c, [0, 0], output)
                tile_x: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_x, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.recip(tile_x)
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_0", 0, 4096)],
                input_c: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_1", 0, 4096)],
                input_x: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_3", 0, 4096)]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                tile_a: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_4, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                _s1: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_3", 0, 4096)] = pl.tile.store(
                    tile_a, [0, 0], output
                )
                tile_c: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_4, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_c, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                _s2: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_3", 0, 4096)] = pl.tile.store(
                    tile_c, [0, 0], output
                )
                tile_x: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_4, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_x, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_7, 0, 4096), pl.Mem.Vec] = pl.tile.recip(
                    tile_x
                )
                result: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_3", 0, 4096)] = pl.tile.store(
                    tile_b, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_inplace_safe_op_allows_producer_consumer_reuse(self):
        """tile.add (inplace-safe) CAN reuse its input's buffer."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_0", 0, 4096)],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_1", 0, 4096)]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                tile_a: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_2, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_2, 0, 4096), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_a
                )
                result: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_1", 0, 4096)] = pl.tile.store(
                    tile_b, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_ands_no_producer_consumer_reuse(self):
        """tile.ands must NOT reuse its input's buffer."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.INT32],
                output: pl.Out[pl.Tensor[[32, 32], pl.INT32]],
            ) -> pl.Tensor[[32, 32], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.INT32, pl.MemorySpace.Vec] = pl.ands(tile_a, 255)
                result: pl.Tensor[[32, 32], pl.INT32] = pl.store(tile_b, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.INT32, pl.MemRef("mem_ddr_0", 0, 4096)],
                output: pl.Out[pl.Tensor[[32, 32], pl.INT32, pl.MemRef("mem_ddr_1", 0, 4096)]],
            ) -> pl.Tensor[[32, 32], pl.INT32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                tile_a: pl.Tile[[32, 32], pl.INT32, pl.MemRef(mem_vec_2, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[32, 32], pl.INT32, pl.MemRef(mem_vec_3, 0, 4096), pl.Mem.Vec] = pl.tile.ands(
                    tile_a, 255
                )
                result: pl.Tensor[[32, 32], pl.INT32, pl.MemRef("mem_ddr_1", 0, 4096)] = pl.tile.store(
                    tile_b, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_xors_no_producer_consumer_reuse(self):
        """tile.xors must NOT reuse its input's buffer."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.INT32],
                input_b: pl.Tensor[[32, 32], pl.INT32],
                output: pl.Out[pl.Tensor[[32, 32], pl.INT32]],
            ) -> pl.Tensor[[32, 32], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                tile_tmp: pl.Tile[[32, 32], pl.INT32, pl.MemorySpace.Vec] = pl.load(input_b, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.INT32, pl.MemorySpace.Vec] = pl.xors(tile_a, 255, tile_tmp)
                result: pl.Tensor[[32, 32], pl.INT32] = pl.store(tile_b, [0, 0], output)
                return result

        # tile_a, tile_tmp, tile_b each get their own buffer — xors is inplace-unsafe.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.INT32, pl.MemRef("mem_ddr_0", 0, 4096)],
                input_b: pl.Tensor[[32, 32], pl.INT32, pl.MemRef("mem_ddr_1", 0, 4096)],
                output: pl.Out[pl.Tensor[[32, 32], pl.INT32, pl.MemRef("mem_ddr_2", 0, 4096)]],
            ) -> pl.Tensor[[32, 32], pl.INT32]:
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                tile_a: pl.Tile[[32, 32], pl.INT32, pl.MemRef(mem_vec_3, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_tmp: pl.Tile[[32, 32], pl.INT32, pl.MemRef(mem_vec_4, 0, 4096), pl.Mem.Vec] = (
                    pl.tile.load(
                        input_b, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                    )
                )
                tile_b: pl.Tile[[32, 32], pl.INT32, pl.MemRef(mem_vec_5, 0, 4096), pl.Mem.Vec] = pl.tile.xors(
                    tile_a, 255, tile_tmp
                )
                result: pl.Tensor[[32, 32], pl.INT32, pl.MemRef("mem_ddr_2", 0, 4096)] = pl.tile.store(
                    tile_b, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_inplace_unsafe_two_level_transitive_chain(self):
        """tile.recip must not reuse a buffer occupied by its input via a two-level chain.

        tile_a/tile_b/tile_x/tile_c all share mem_vec_3 (chain reuse).
        tile_d uses mem_vec_6 — recip(tile_d) cannot reuse tile_d's buffer.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP32],
                input_u: pl.Tensor[[32, 32], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                _s1: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output)
                tile_u: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_u, [0, 0], [32, 32])
                tile_d: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_u, tile_u)
                _s2: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_u, [0, 0], output)
                tile_c: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.recip(tile_d)
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_0", 0, 4096)],
                input_u: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_1", 0, 4096)],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                tile_a: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_3, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_3, 0, 4096), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_a
                )
                _s1: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)] = pl.tile.store(
                    tile_b, [0, 0], output
                )
                tile_u: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_3, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_u, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_d: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_6, 0, 4096), pl.Mem.Vec] = pl.tile.add(
                    tile_u, tile_u
                )
                _s2: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)] = pl.tile.store(
                    tile_u, [0, 0], output
                )
                tile_c: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_3, 0, 4096), pl.Mem.Vec] = pl.tile.recip(
                    tile_d
                )
                result: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)] = pl.tile.store(
                    tile_c, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


class TestYieldFixup:
    """Yield fixup for ForStmt and IfStmt -- ensuring loop-carry and return variables share correct MemRef."""

    def test_tile_move_inserted_when_memrefs_diverge(self):
        """When initValue and yield value start with different MemRefs,
        a tile.move is inserted to unify all loop-carry vars to one MemRef.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                init_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                for _i, (acc_0,) in pl.range(0, 4, init_values=(init_0,)):
                    extra_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_0, acc_0)
                    next_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(extra_0, acc_0)
                    out_0 = pl.yield_(next_0)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(out_0, [0, 0], output)
                return result

        # init_0 uses mem_vec_2; loop body uses mem_vec_3 for extra_0/next_0;
        # tile.move converts next_0 -> next_0_mv with mem_vec_2 so the yield
        # value matches the iter_arg's MemRef.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                init_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                for _i, (acc_0,) in pl.range(4, init_values=(init_0,)):
                    extra_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(acc_0, acc_0)
                    )
                    next_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(extra_0, acc_0)
                    )
                    next_0_mv: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.move(next_0, target_memory=pl.Mem.Vec)
                    )
                    out_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.yield_(
                        next_0_mv
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    out_0, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_simple_loop_memrefs_unified(self):
        """Simple loop: after reuse, iter_arg/initValue/return_var share MemRef.

        Even with no extra ops, tile.move is still inserted because the
        MemoryReuse pass first allocates a separate buffer for next_0 and
        then unifies via move.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                init_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                for _i, (acc_0,) in pl.range(0, 4, init_values=(init_0,)):
                    next_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_0, acc_0)
                    out_0 = pl.yield_(next_0)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(out_0, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                init_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                for _i, (acc_0,) in pl.range(4, init_values=(init_0,)):
                    next_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(acc_0, acc_0)
                    )
                    next_0_mv: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.move(next_0, target_memory=pl.Mem.Vec)
                    )
                    out_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.yield_(
                        next_0_mv
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    out_0, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_multiple_iter_args_partial_mismatch(self):
        """With 2 iter_args, tile.move is inserted for each mismatched pair."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                init_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                init_1: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                for _i, (acc_0, acc_1) in pl.range(0, 4, init_values=(init_0, init_1)):
                    extra_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_0, acc_0)
                    next_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(extra_0, acc_0)
                    extra_1: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_1, acc_1)
                    next_1: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(extra_1, acc_1)
                    out_0, out_1 = pl.yield_(next_0, next_1)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(out_0, [0, 0], output)
                return result

        # init_0/init_1 each get their own buffer (mem_vec_2, mem_vec_3).
        # Loop body uses mem_vec_4/mem_vec_6 for the two intermediate chains.
        # Two tile.move ops unify next_0/next_1 to init buffers.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                init_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                init_1: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                for _i, (acc_0, acc_1) in pl.range(4, init_values=(init_0, init_1)):
                    extra_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(acc_0, acc_0)
                    )
                    next_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(extra_0, acc_0)
                    )
                    extra_1: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_6, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(acc_1, acc_1)
                    )
                    next_1: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_6, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(extra_1, acc_1)
                    )
                    next_0_mv: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.move(next_0, target_memory=pl.Mem.Vec)
                    )
                    next_1_mv: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.move(next_1, target_memory=pl.Mem.Vec)
                    )
                    out_0, out_1 = pl.yield_(next_0_mv, next_1_mv)
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    out_0, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_if_stmt_return_var_memref_patched(self):
        """tile_b/tile_c reuse tile_a's MemRef; if_result picks up the patched MemRef."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                _: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_a, [0, 0], output)
                if cond_param < 2:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        input_tensor, [0, 0], [64, 64]
                    )
                    if_result = pl.yield_(tile_b)
                else:
                    tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        input_tensor, [0, 0], [64, 64]
                    )
                    if_result = pl.yield_(tile_c)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(if_result, [0, 0], output)
                return result

        # tile_a is dead before the IfStmt, so tile_b/tile_c both reuse mem_vec_2.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                _: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_a, [0, 0], output
                )
                if cond_param < 2:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.load(
                            input_tensor,
                            [0, 0],
                            [64, 64],
                            [64, 64],
                            target_memory=pl.Mem.Vec,
                            transpose=False,
                        )
                    )
                    if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(tile_b)
                    )
                else:
                    tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.load(
                            input_tensor,
                            [0, 0],
                            [64, 64],
                            [64, 64],
                            target_memory=pl.Mem.Vec,
                            transpose=False,
                        )
                    )
                    if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(tile_c)
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    if_result, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_if_stmt_tile_move_when_branch_memrefs_differ(self):
        """When IfStmt branches yield tiles with different MemRefs, the pass
        unifies them. In this case t3 already gets reused into tile_a's MemRef.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                input_b: pl.Tensor[[64, 64], pl.FP32],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_b, [0, 0], [64, 64])
                if cond_param < 2:
                    alias_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = tile_a
                    if_result = pl.yield_(alias_a)
                else:
                    t1: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_b)
                    t2: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(t1, tile_a)
                    t3: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(t2, tile_a)
                    if_result = pl.yield_(t3)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(if_result, [0, 0], output)
                return result

        # tile_a/alias_a/if_result share mem_vec_3 (then-branch). tile_b uses
        # mem_vec_4. In the else, t1/t2 use mem_vec_4 (reused via tile_b's
        # buffer), and t3 reuses mem_vec_3 because tile_a is at last use.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                input_b: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_b, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                if cond_param < 2:
                    alias_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = tile_a
                    if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(alias_a)
                    )
                else:
                    t1: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                        tile_a, tile_b
                    )
                    t2: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                        t1, tile_a
                    )
                    t3: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                        t2, tile_a
                    )
                    if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(t3)
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)] = pl.tile.store(
                    if_result, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


class TestControlFlow:
    """Tests for correct lifetime analysis across control flow boundaries."""

    def test_var_used_in_nested_if_not_reused_in_loop(self):
        """tile_a is used inside loop body so it stays live across the loop;
        tile_c gets a separate buffer; tile.move unifies the loop-carry."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                for i, (acc,) in pl.range(0, 4, init_values=(tile_a,)):
                    if i < 2:
                        tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc, tile_a)
                        if_result = pl.yield_(tile_c)
                    else:
                        if_result = pl.yield_(acc)
                    loop_out = pl.yield_(if_result)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(loop_out, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                for i, (acc,) in pl.range(4, init_values=(tile_a,)):
                    if i < 2:
                        tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                            pl.tile.add(acc, tile_a)
                        )
                        if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                            pl.yield_(tile_c)
                        )
                    else:
                        if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                            pl.yield_(acc)
                        )
                    if_result_mv: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.move(if_result, target_memory=pl.Mem.Vec)
                    )
                    loop_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(if_result_mv)
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    loop_out, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_different_if_branches_can_share(self):
        """Variables in different IfStmt branches CAN share MemRef (non-overlapping lifetimes)."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                if cond_param < 2:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        input_tensor, [0, 0], [64, 64]
                    )
                    if_result = pl.yield_(tile_b)
                else:
                    tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        input_tensor, [0, 0], [64, 64]
                    )
                    if_result = pl.yield_(tile_c)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(if_result, [0, 0], output)
                return result

        # tile_b/tile_c/if_result all share mem_vec_2.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                if cond_param < 2:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.load(
                            input_tensor,
                            [0, 0],
                            [64, 64],
                            [64, 64],
                            target_memory=pl.Mem.Vec,
                            transpose=False,
                        )
                    )
                    if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(tile_b)
                    )
                else:
                    tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.load(
                            input_tensor,
                            [0, 0],
                            [64, 64],
                            [64, 64],
                            target_memory=pl.Mem.Vec,
                            transpose=False,
                        )
                    )
                    if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(tile_c)
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    if_result, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_loop_local_var_can_be_reused(self):
        """Variables defined AND used entirely within a single loop iteration
        can still be reused with other loop-local variables."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                init_tile: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                for _i, (acc,) in pl.range(0, 4, init_values=(init_tile,)):
                    tile_x: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        input_tensor, [0, 0], [64, 64]
                    )
                    tile_y: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_x, tile_x)
                    tile_z: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_y, tile_y)
                    loop_out = pl.yield_(tile_z)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(loop_out, [0, 0], output)
                return result

        # init_tile uses mem_vec_2; loop body uses mem_vec_3 for the chain;
        # tile.move inserts a copy at the yield to reset to init's MemRef.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                init_tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                )
                for _i, (acc,) in pl.range(4, init_values=(init_tile,)):
                    tile_x: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.load(
                            input_tensor,
                            [0, 0],
                            [64, 64],
                            [64, 64],
                            target_memory=pl.Mem.Vec,
                            transpose=False,
                        )
                    )
                    tile_y: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(tile_x, tile_x)
                    )
                    tile_z: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(tile_y, tile_y)
                    )
                    tile_z_mv: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.move(tile_z, target_memory=pl.Mem.Vec)
                    )
                    loop_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(tile_z_mv)
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    loop_out, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_nested_for_loops_outer_var_extends_to_outer_end(self):
        """Variable defined before nested loops, used in inner loop body --
        lifetime extends to the END of the OUTER loop (not just inner)."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                init_outer: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                for _i, (acc_outer,) in pl.range(0, 4, init_values=(init_outer,)):
                    init_inner: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                        [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                    )
                    for _j, (acc_inner,) in pl.range(0, 4, init_values=(init_inner,)):
                        tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_inner, tile_a)
                        inner_out = pl.yield_(tile_b)
                    tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_outer, inner_out)
                    outer_out = pl.yield_(tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(outer_out, [0, 0], output)
                return result

        # tile_a (mem_vec_2), init_outer (mem_vec_3), init_inner (mem_vec_4),
        # tile_b (mem_vec_5) — tile_a stays live across both loops, so it
        # never gets reused. tile.move pairs unify yields back to outer/inner
        # iter_arg buffers.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                init_outer: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                )
                for _i, (acc_outer,) in pl.range(4, init_values=(init_outer,)):
                    init_inner: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    )
                    for _j, (acc_inner,) in pl.range(4, init_values=(init_inner,)):
                        tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, 0, 16384), pl.Mem.Vec] = (
                            pl.tile.add(acc_inner, tile_a)
                        )
                        tile_b_mv: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                            pl.tile.move(tile_b, target_memory=pl.Mem.Vec)
                        )
                        inner_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                            pl.yield_(tile_b_mv)
                        )
                    tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(acc_outer, inner_out)
                    )
                    tile_d_mv: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.move(tile_d, target_memory=pl.Mem.Vec)
                    )
                    outer_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(tile_d_mv)
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    outer_out, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_if_without_else_branch(self):
        """IfStmt with only then branch (no else): tile_a is alive through the
        IfStmt and reused only by tile_c (after the IfStmt, when tile_a is at
        last use). tile_b inside the then branch needs its own buffer.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                if cond_param < 2:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                    _: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], output)
                    pl.yield_()
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                if cond_param < 2:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(tile_a, tile_a)
                    )
                    _: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                        tile_b, [0, 0], output
                    )
                    pl.yield_()
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_a
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_c, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_for_with_if_multiple_vars_competing(self):
        """ForStmt with IfStmt inside, multiple variables from before the loop
        used inside the if -- all outer variables stay live across the loop."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                init_tile: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                for i, (acc,) in pl.range(0, 4, init_values=(init_tile,)):
                    if i < 2:
                        tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_b)
                        if_result = pl.yield_(tile_c)
                    else:
                        tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_b, tile_a)
                        if_result = pl.yield_(tile_d)
                    loop_out = pl.yield_(if_result)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(loop_out, [0, 0], output)
                return result

        # tile_a → mem_vec_2, tile_b → mem_vec_3 (both live across loop).
        # init_tile → mem_vec_4 (loop-carry buffer).
        # tile_c/tile_d → mem_vec_5 (different branches share).
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                init_tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                )
                for i, (acc,) in pl.range(4, init_values=(init_tile,)):
                    if i < 2:
                        tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, 0, 16384), pl.Mem.Vec] = (
                            pl.tile.add(tile_a, tile_b)
                        )
                        if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, 0, 16384), pl.Mem.Vec] = (
                            pl.yield_(tile_c)
                        )
                    else:
                        tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, 0, 16384), pl.Mem.Vec] = (
                            pl.tile.add(tile_b, tile_a)
                        )
                        if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, 0, 16384), pl.Mem.Vec] = (
                            pl.yield_(tile_d)
                        )
                    if_result_mv: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.move(if_result, target_memory=pl.Mem.Vec)
                    )
                    loop_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(if_result_mv)
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    loop_out, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_branch_local_var_does_not_leak(self):
        """A variable defined and consumed entirely inside one IfStmt branch
        has a short lifetime and does not block reuse after the IfStmt."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                if cond_param < 2:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                    if_result = pl.yield_(tile_b)
                else:
                    if_result = pl.yield_(tile_a)
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(if_result, if_result)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        # tile_a → mem_vec_2 (and tile_e reuses it). tile_b → mem_vec_3
        # (in then-branch), unified with else-branch via tile.move on tile_a.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                if cond_param < 2:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(tile_a, tile_a)
                    )
                    if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(tile_b)
                    )
                else:
                    tile_a_mv: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.move(tile_a, target_memory=pl.Mem.Vec)
                    )
                    if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(tile_a_mv)
                    )
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    if_result, if_result
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_e, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_loop_return_var_blocks_init_memref_reuse(self):
        """Return_var used after loop must block reuse of initValue's MemRef.

        Regression test for issue #768: MemoryReuse allowed a post-loop
        variable to reuse the initValue's MemRef, but YieldFixup later
        placed the loop output in the same buffer, causing the accumulated
        result to be overwritten before the final add consumed it.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                input_b: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                o_acc: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                o_acc_z: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.muls(o_acc, 0.0)
                for _kb, (acc,) in pl.range(0, 4, init_values=(o_acc_z,)):
                    chunk: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                    acc_next: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc, chunk)
                    loop_out = pl.yield_(acc_next)
                resid: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_b, [0, 0], [64, 64])
                final: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(loop_out, resid)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(final, [0, 0], output)
                return result

        # o_acc/o_acc_z/loop_out/final all share mem_vec_3 (loop-carry buffer).
        # chunk/acc_next reuse mem_vec_5 inside the loop, and resid reuses
        # mem_vec_5 because chunk/acc_next are dead by then. Crucially, resid
        # does NOT take mem_vec_3 — that would clobber the loop result.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                input_b: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                o_acc: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                )
                o_acc_z: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.muls(o_acc, 0.0)
                )
                for _kb, (acc,) in pl.range(4, init_values=(o_acc_z,)):
                    chunk: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.load(
                            input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                        )
                    )
                    acc_next: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(acc, chunk)
                    )
                    acc_next_mv: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.move(acc_next, target_memory=pl.Mem.Vec)
                    )
                    loop_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(acc_next_mv)
                    )
                resid: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_b, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                final: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    loop_out, resid
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)] = pl.tile.store(
                    final, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


class TestMetadata:
    """Function metadata should survive MemoryReuse rewrites."""

    def test_preserves_split_metadata(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def vector_producer(
                self,
                input_tensor: pl.Tensor[[16, 16], pl.FP16],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                tile_a: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [16, 16]
                )
                tile_b: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                result: pl.Tensor[[16, 16], pl.FP16] = pl.store(tile_b, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def vector_producer(
                self,
                input_tensor: pl.Tensor[[16, 16], pl.FP16, pl.MemRef("mem_ddr_0", 0, 512)],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP16, pl.MemRef("mem_ddr_1", 0, 512)]],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
                tile_a: pl.Tile[[16, 16], pl.FP16, pl.MemRef(mem_vec_2, 0, 512), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [16, 16], [16, 16], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[16, 16], pl.FP16, pl.MemRef(mem_vec_2, 0, 512), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_a
                )
                result: pl.Tensor[[16, 16], pl.FP16, pl.MemRef("mem_ddr_1", 0, 512)] = pl.tile.store(
                    tile_b, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

        # Sanity: split metadata round-trips through the pass.
        after_vp = After.get_function("vector_producer")
        assert after_vp is not None
        assert after_vp.split == ir.SplitMode.UP_DOWN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
