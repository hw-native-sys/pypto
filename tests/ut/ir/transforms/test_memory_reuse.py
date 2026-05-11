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
``ir.assert_structural_equal(After, Expected)``.
DefFields always auto-map, so ``enable_auto_mapping=True`` is unnecessary.
This aligns MemRef objects consistently: if two tiles share a MemRef in
``After``, the corresponding tiles in ``Expected`` must also share.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir.op import tile


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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)


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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)


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
        ir.assert_structural_equal(After, Expected)


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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)


class TestValidShapeDivergence:
    """Tiles with identical storage but divergent ``valid_shape`` can share a MemRef.

    Reproduces the scenario from issue #1094: unrolled / partially-unrolled
    kernels produce sibling branches whose tiles differ only in ``valid_shape``
    (driven by per-branch boundary guards). Those tiles should share a backing
    allocation; each variable keeps its own ``valid_shape`` at every use site.
    """

    def test_different_valid_shape_can_reuse(self):
        """Two sequential loads with different static ``valid_shape`` share one MemRef."""

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
                _res_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_a, [0, 0], output_a)
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_a, [0, 0], [64, 64], valid_shapes=[32, 64]
                )
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], output_b)
                return result

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
                tile_a: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_3, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(valid_shape=[48, 64]),
                ] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [48, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                _res_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_a, [0, 0], output_a
                )
                tile_b: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_3, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(valid_shape=[32, 64]),
                ] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [32, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)] = pl.tile.store(
                    tile_b, [0, 0], output_b
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_non_2d_divergent_valid_shape_blocks_reuse(self):
        """3D tiles with divergent ``valid_shape`` must NOT reuse (set_validshape is 2D-only)."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[4, 64, 64], pl.FP32],
                output_a: pl.Out[pl.Tensor[[4, 64, 64], pl.FP32]],
                output_b: pl.Out[pl.Tensor[[4, 64, 64], pl.FP32]],
            ) -> pl.Tensor[[4, 64, 64], pl.FP32]:
                tile_a: pl.Tile[[4, 64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_a, [0, 0, 0], [4, 64, 64], valid_shapes=[4, 48, 64]
                )
                _res_a: pl.Tensor[[4, 64, 64], pl.FP32] = pl.store(tile_a, [0, 0, 0], output_a)
                tile_b: pl.Tile[[4, 64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_a, [0, 0, 0], [4, 64, 64], valid_shapes=[4, 32, 64]
                )
                result: pl.Tensor[[4, 64, 64], pl.FP32] = pl.store(tile_b, [0, 0, 0], output_b)
                return result

        After = _run_pipeline(Before)
        # Collect base_ptr names from every tile AssignStmt in the After IR.
        # 3D tiles with divergent valid_shape must NOT share a MemRef — the
        # compatibility check's 2D guard keeps them on the strict path.
        bases = _collect_tile_memref_bases(After)
        tile_a_base = bases.get("tile_a")
        tile_b_base = bases.get("tile_b")
        assert tile_a_base is not None and tile_b_base is not None, (
            f"Expected tile_a and tile_b in After IR; got bases: {bases}"
        )
        assert tile_a_base != tile_b_base, (
            f"3D divergent-valid_shape tiles should NOT share a MemRef, but both bind to {tile_a_base}"
        )


def _collect_tile_memref_bases(program: ir.Program) -> dict[str, str]:
    """Return ``{tile_var_name: memref_base_name}`` for every AssignStmt in the program.

    Walks the first function's body using a small IRVisitor subclass that
    records the MemRef base name when a tile-typed variable is assigned.
    """
    result: dict[str, str] = {}
    main_func = next(iter(program.functions.values()))

    class _Collector(ir.IRVisitor):
        def visit_assign_stmt(self, stmt):  # type: ignore[override]
            var_type = stmt.var.type
            if isinstance(var_type, ir.TileType) and var_type.memref is not None:
                result[stmt.var.name_hint] = var_type.memref.base_.name_hint
            super().visit_assign_stmt(stmt)

    visitor = _Collector()
    visitor.visit_stmt(main_func.body)
    return result


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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)


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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)


class TestYieldFixup:
    """Yield fixup for ForStmt and IfStmt -- ensuring loop-carry and return variables share correct MemRef."""

    def test_producer_retyped_to_iter_arg_buffer(self):
        """The yield producer is retyped directly to the iter_arg's MemRef
        (no tile.move inserted). Intermediate 'extra_0' keeps its own buffer.
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

        # init_0/acc_0/next_0/out_0 all share mem_vec_2 (the iter_arg buffer).
        # The retargeter places next_0 directly on mem_vec_2; extra_0 (not the
        # yield value) keeps its own buffer mem_vec_3. No tile.move is needed.
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
                    next_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(extra_0, acc_0)
                    )
                    out_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.yield_(
                        next_0
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    out_0, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected)

    def test_simple_loop_memrefs_unified(self):
        """Simple loop: iter_arg/initValue/return_var/next_0 all land in a
        single MemRef. The retargeter retypes next_0 directly, so no
        intermediate buffer or tile.move is needed.
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
                init_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                for _i, (acc_0,) in pl.range(4, init_values=(init_0,)):
                    next_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(acc_0, acc_0)
                    )
                    out_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.yield_(
                        next_0
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    out_0, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multiple_iter_args_producers_retyped_independently(self):
        """With 2 iter_args, the retargeter retypes each yield producer
        directly to its own iter_arg buffer. Intermediate chains share a
        single scratch buffer.
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
                init_1: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                for _i, (acc_0, acc_1) in pl.range(0, 4, init_values=(init_0, init_1)):
                    extra_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_0, acc_0)
                    next_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(extra_0, acc_0)
                    extra_1: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_1, acc_1)
                    next_1: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(extra_1, acc_1)
                    out_0, _out_1 = pl.yield_(next_0, next_1)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(out_0, [0, 0], output)
                return result

        # init_0 -> mem_vec_2 and init_1 -> mem_vec_3 (loop-carry buffers).
        # next_0/next_1 retyped directly to mem_vec_2/mem_vec_3; extra_0 and
        # extra_1 share a single scratch buffer mem_vec_4. No tile.move ops.
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
                    next_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(extra_0, acc_0)
                    )
                    extra_1: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(acc_1, acc_1)
                    )
                    next_1: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(extra_1, acc_1)
                    )
                    out_0, _out_1 = pl.yield_(next_0, next_1)
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    out_0, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)


class TestControlFlow:
    """Tests for correct lifetime analysis across control flow boundaries."""

    def test_var_used_in_nested_if_shares_iter_arg_buffer(self):
        """The iter_arg `acc` and its initValue `tile_a` share MemRef via
        InitMemRef. The retargeter further propagates that MemRef through
        the IfStmt's return_var and both branches' yield values, so every
        tile in the yield chain lands on the iter_arg buffer.
        """

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

        # tile_a/acc/tile_c/if_result/loop_out all share mem_vec_2. The else
        # branch already yields `acc` (already on mem_vec_2), and the then
        # branch's tile_c is retargeted onto mem_vec_2 since mem_vec_2 is
        # not read after tile_c's write inside the branch body.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                for i, (acc,) in pl.range(4, init_values=(tile_a,)):
                    if i < 2:
                        tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                            pl.tile.add(acc, tile_a)
                        )
                        if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                            pl.yield_(tile_c)
                        )
                    else:
                        if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                            pl.yield_(acc)
                        )
                    loop_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(if_result)
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    loop_out, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)

    def test_loop_local_var_can_be_reused(self):
        """Loop-local vars share a scratch buffer; the yield producer is
        retyped directly to the iter_arg buffer. tile_x/tile_y share a
        scratch, tile_z (the yield value) lands on init_tile's buffer.
        """

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

        # init_tile/acc/tile_z/loop_out on mem_vec_2; tile_x/tile_y share
        # scratch mem_vec_3. The retargeter retypes tile_z directly to
        # mem_vec_2, so no tile.move is needed at yield.
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
                    tile_z: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(tile_y, tile_y)
                    )
                    loop_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(tile_z)
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    loop_out, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected)

    def test_nested_for_loops_outer_var_extends_to_outer_end(self):
        """Variable defined before nested loops, used in inner loop body --
        lifetime extends to the END of the OUTER loop. With the retargeter,
        each level's yield producer is retyped directly onto that level's
        iter_arg buffer; no tile.move ops are inserted.
        """

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

        # tile_a is live across both loops on mem_vec_2. init_outer/acc_outer
        # share mem_vec_3; init_inner/acc_inner share mem_vec_4. tile_b
        # (inner yield) is retyped to mem_vec_4, tile_d (outer yield) to
        # mem_vec_3. No scratch buffer for tile_b is allocated.
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
                        tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                            pl.tile.add(acc_inner, tile_a)
                        )
                        inner_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                            pl.yield_(tile_b)
                        )
                    tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(acc_outer, inner_out)
                    )
                    outer_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(tile_d)
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    outer_out, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)

    def test_for_with_if_multiple_vars_competing(self):
        """ForStmt with IfStmt inside: `tile_a` and `tile_b` are live across
        the loop on distinct buffers. The retargeter propagates the
        iter_arg buffer through the IfStmt's return_var and both branches'
        yield producers (both unconstrained adds with liveness OK).
        """

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

        # tile_a -> mem_vec_2, tile_b -> mem_vec_3 (both live across loop).
        # init_tile/acc/tile_c/tile_d/if_result/loop_out all share mem_vec_4
        # via the retargeter.
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
                        tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                            pl.tile.add(tile_a, tile_b)
                        )
                        if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                            pl.yield_(tile_c)
                        )
                    else:
                        tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                            pl.tile.add(tile_b, tile_a)
                        )
                        if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                            pl.yield_(tile_d)
                        )
                    loop_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(if_result)
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    loop_out, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected)

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
        ir.assert_structural_equal(After, Expected)

    def test_loop_return_var_blocks_init_memref_reuse(self):
        """Return_var used after loop must block reuse of initValue's MemRef.

        Regression test for issue #768: MemoryReuse allowed a post-loop
        variable to reuse the initValue's MemRef, causing the accumulated
        result to be overwritten before the final add consumed it. The
        critical invariant — `resid` must NOT take the loop-carry buffer
        `mem_vec_3` — is still enforced. The retargeter additionally
        retypes `acc_next` directly to `mem_vec_3`, eliminating the
        tile.move the old pipeline emitted, but the #768 guard is
        unchanged.
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
        # acc_next is retyped directly to mem_vec_3 by the retargeter.
        # chunk lives on mem_vec_5 inside the loop; resid reuses mem_vec_5
        # because chunk is dead by then. Crucially, resid does NOT take
        # mem_vec_3 -- that would clobber the loop result (#768 regression).
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
                    acc_next: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(acc, chunk)
                    )
                    loop_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(acc_next)
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
        ir.assert_structural_equal(After, Expected)


class TestTopDownRetargeter:
    """Tests for the Step-0 top-down retargeter inside MemoryReuse.

    The retargeter walks each ForStmt's iter_arg -> yield chain and
    rewrites the producer's MemRef to the iter_arg's MemRef when the
    source tile is dead at the producer's write. These tests exercise
    its happy path (pinned accumulator chain) and its safety check
    (must decline when target is still live).
    """

    def test_acc_chain_pinned_producer_shares_iter_arg_buffer(self):
        """A matmul_acc chain over Acc memory: the retargeter follows the
        pinned `acc` input up to the iter_arg (already on target) and
        retypes `acc_next` onto the same single Acc allocation. No
        tile.move ops are inserted.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP16],
                input_b: pl.Tensor[[32, 32], pl.FP16],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a_l1: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Mat] = pl.load(
                    input_a, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
                )
                tile_b_l1: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Mat] = pl.load(
                    input_b, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
                )
                tile_a_l0a: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Left] = pl.move(
                    tile_a_l1, target_memory=pl.MemorySpace.Left
                )
                tile_b_l0b: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Right] = pl.move(
                    tile_b_l1, target_memory=pl.MemorySpace.Right
                )
                # Use matmul (not tile.create) so init_acc's TileView
                # matches matmul_acc's — keeps the pre-verified IR well-formed.
                init_acc: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(tile_a_l0a, tile_b_l0b)
                for _k, (acc,) in pl.range(0, 4, init_values=(init_acc,)):
                    acc_next: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Acc] = pl.matmul_acc(
                        acc, tile_a_l0a, tile_b_l0b
                    )
                    loop_out = pl.yield_(acc_next)
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(loop_out, [0, 0], output)
                return result

        # init_acc, acc, acc_next, loop_out all share the single Acc
        # allocation mem_acc_7. No tile.move op appears anywhere in the
        # loop body — the retargeter collapses the chain. (matmul_acc is
        # already pinned to its acc input by set_output_reuses_input(0), so
        # the retargeter recurses through the pin to the iter_arg, which
        # is already on the target MemRef, then retypes acc_next.)
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP16, pl.MemRef("mem_ddr_0", 0, 2048)],
                input_b: pl.Tensor[[32, 32], pl.FP16, pl.MemRef("mem_ddr_1", 0, 2048)],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 2048)
                mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 2048)
                mem_left_5: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 2048)
                mem_right_6: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 2048)
                mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
                tile_a_l1: pl.Tile[[32, 32], pl.FP16, pl.MemRef(mem_mat_3, 0, 2048), pl.Mem.Mat] = (
                    pl.tile.load(
                        input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Mat, transpose=False
                    )
                )
                tile_b_l1: pl.Tile[[32, 32], pl.FP16, pl.MemRef(mem_mat_4, 0, 2048), pl.Mem.Mat] = (
                    pl.tile.load(
                        input_b, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Mat, transpose=False
                    )
                )
                tile_a_l0a: pl.Tile[[32, 32], pl.FP16, pl.MemRef(mem_left_5, 0, 2048), pl.Mem.Left] = (
                    pl.tile.move(tile_a_l1, target_memory=pl.Mem.Left)
                )
                tile_b_l0b: pl.Tile[[32, 32], pl.FP16, pl.MemRef(mem_right_6, 0, 2048), pl.Mem.Right] = (
                    pl.tile.move(tile_b_l1, target_memory=pl.Mem.Right)
                )
                init_acc: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_acc_7, 0, 4096), pl.Mem.Acc] = (
                    pl.tile.matmul(tile_a_l0a, tile_b_l0b)
                )
                for _k, (acc,) in pl.range(4, init_values=(init_acc,)):
                    acc_next: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_acc_7, 0, 4096), pl.Mem.Acc] = (
                        pl.tile.matmul_acc(acc, tile_a_l0a, tile_b_l0b)
                    )
                    loop_out: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_acc_7, 0, 4096), pl.Mem.Acc] = (
                        pl.yield_(acc_next)
                    )
                result: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)] = pl.tile.store(
                    loop_out, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected)

    def test_retargeter_declines_when_target_still_live(self):
        """Safety check: if target's base is read after the candidate
        producer (here, via another op that reads the iter_arg), the
        retargeter must leave the producer alone so that the iter_arg's
        value is preserved until its last read. YieldFixup then inserts
        a tile.move to unify the yield to the iter_arg buffer.
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
                    tmp: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_0, acc_0)
                    # `other` reads acc_0 AFTER tmp is written. If the
                    # retargeter retyped tmp onto acc_0's buffer here, the
                    # subsequent read of acc_0 would see the already-
                    # clobbered value. So the retargeter must decline.
                    other: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.mul(tmp, acc_0)
                    _use: pl.Tensor[[64, 64], pl.FP32] = pl.store(other, [0, 0], output)
                    loop_out = pl.yield_(tmp)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(loop_out, [0, 0], output)
                return result

        # tmp stays on its own buffer mem_vec_3 (retargeter declined).
        # YieldFixup inserts tmp_mv = tile.move(tmp, ...) onto the iter_arg
        # buffer mem_vec_2, and loop_out yields tmp_mv.
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
                init_0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                for _i, (acc_0,) in pl.range(4, init_values=(init_0,)):
                    tmp: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                        acc_0, acc_0
                    )
                    other: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.mul(tmp, acc_0)
                    )
                    _use: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                        other, [0, 0], output
                    )
                    tmp_mv: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.move(tmp, target_memory=pl.Mem.Vec)
                    )
                    loop_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(tmp_mv)
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    loop_out, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected)

    def test_retargeter_declines_when_read_after_nested_if(self):
        """Regression test for the ancestor-walking liveness check.

        The yield producer (``tile_c``) sits inside an IfStmt branch, but
        a subsequent op reads ``acc_0`` *after* the IfStmt in the
        enclosing loop body.  An innermost-branch-only liveness check
        would miss this read and incorrectly retype ``tile_c`` onto
        ``acc_0``'s buffer, clobbering the iter_arg before the post-
        IfStmt read runs.  The ancestor-walking check sees the read and
        declines.

        The post-IfStmt read is expressed as ``pl.store(acc_0, ...)``
        directly rather than via an intermediate ``side = op(acc_0)`` so
        the assertion is not muddied by any lifetime-coalescing that
        could place ``side`` and ``if_result`` in the same buffer.
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
                for i, (acc_0,) in pl.range(0, 4, init_values=(init_0,)):
                    if i < 2:
                        tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_0, acc_0)
                        if_result = pl.yield_(tile_c)
                    else:
                        if_result = pl.yield_(acc_0)
                    # Reads acc_0 (target base) AFTER the IfStmt.
                    _use: pl.Tensor[[64, 64], pl.FP32] = pl.store(acc_0, [0, 0], output)
                    loop_out = pl.yield_(if_result)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(loop_out, [0, 0], output)
                return result

        # acc_0/init_0 share mem_vec_2.  tile_c stays on mem_vec_3 (NOT
        # retargeted onto mem_vec_2) because the liveness check detects
        # the post-IfStmt read of acc_0.  YieldFixup then inserts a
        # tile.move to unify if_result to the iter_arg buffer at the yield.
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
                for i, (acc_0,) in pl.range(4, init_values=(init_0,)):
                    if i < 2:
                        tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                            pl.tile.add(acc_0, acc_0)
                        )
                        if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                            pl.yield_(tile_c)
                        )
                    else:
                        if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                            pl.yield_(acc_0)
                        )
                    _use: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                        acc_0, [0, 0], output
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
        ir.assert_structural_equal(After, Expected)

    def test_retargeter_declines_for_not_inplace_safe_op(self):
        """Regression test for the not_inplace_safe check.

        ``tile.mrgsort_format1`` is registered ``.not_inplace_safe()`` —
        its implementation requires distinct src/dst buffers.  In a
        merge-sort accumulator loop the yield producer both reads and
        (would) write ``tile_iter``'s buffer, so retargeting ``merged``
        onto that buffer creates an in-place execution the op cannot
        handle and fails at runtime with NPU error 507017
        (``rtStreamSynchronize AICPU failed``).  The retargeter must
        decline, and YieldFixup then inserts a ``tile.move`` to unify
        the yield with the iter_arg buffer.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                src_tensor: pl.Tensor[[1, 2048], pl.FP32],
                idx_tensor: pl.Tensor[[1, 2048], pl.UINT32],
                val_output: pl.Out[pl.Tensor[[1, 2048], pl.FP32]],
            ) -> pl.Tensor[[1, 2048], pl.FP32]:
                src_tile: pl.Tile[[1, 2048], pl.FP32] = pl.load(src_tensor, [0, 0], [1, 2048])
                idx_tile: pl.Tile[[1, 2048], pl.UINT32] = pl.load(idx_tensor, [0, 0], [1, 2048])
                sorted_tile: pl.Tile[[1, 4096], pl.FP32] = pl.tile.sort32(src_tile, idx_tile)
                for i, (tile_iter,) in pl.range(3, init_values=(sorted_tile,)):
                    block_len = 1 << (6 + i * 2)
                    merged: pl.Tile[[1, 4096], pl.FP32] = pl.tile.mrgsort(tile_iter, block_len=block_len)
                    result = pl.yield_(merged)
                vals: pl.Tile[[1, 2048], pl.FP32] = pl.tile.gather_mask(
                    result, mask_pattern=pl.tile.MaskPattern.P0101
                )
                out_val: pl.Tensor[[1, 2048], pl.FP32] = pl.store(vals, [0, 0], val_output)
                return out_val

        # tile_iter/sorted_tile/result live on mem_vec_5 (loop-carry buffer).
        # `merged` is allocated on its own buffer mem_vec_6 so src (tile_iter
        # on mem_vec_5) and dst (merged on mem_vec_6) differ.  YieldFixup
        # inserts merged_mv on mem_vec_5 so the yield matches the iter_arg.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                src_tensor: pl.Tensor[[1, 2048], pl.FP32, pl.MemRef("mem_ddr_0", 0, 8192)],
                idx_tensor: pl.Tensor[[1, 2048], pl.UINT32, pl.MemRef("mem_ddr_1", 0, 8192)],
                val_output: pl.Out[pl.Tensor[[1, 2048], pl.FP32, pl.MemRef("mem_ddr_2", 0, 8192)]],
            ) -> pl.Tensor[[1, 2048], pl.FP32]:
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
                mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                src_tile: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_3, 0, 8192), pl.Mem.Vec] = (
                    pl.tile.load(
                        src_tensor, [0, 0], [1, 2048], [1, 2048], target_memory=pl.Mem.Vec, transpose=False
                    )
                )
                idx_tile: pl.Tile[[1, 2048], pl.UINT32, pl.MemRef(mem_vec_4, 0, 8192), pl.Mem.Vec] = (
                    pl.tile.load(
                        idx_tensor, [0, 0], [1, 2048], [1, 2048], target_memory=pl.Mem.Vec, transpose=False
                    )
                )
                sorted_tile: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_5, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.sort32(src_tile, idx_tile)
                )
                for i, (tile_iter,) in pl.range(3, init_values=(sorted_tile,)):
                    block_len = 1 << (6 + i * 2)
                    merged: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_6, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.mrgsort(tile_iter, block_len=block_len)
                    )
                    merged_mv: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_5, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.move(merged, target_memory=pl.Mem.Vec)
                    )
                    result: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_5, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(merged_mv)
                    )
                vals: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_3, 0, 8192), pl.Mem.Vec] = (
                    pl.tile.gather_mask(result, mask_pattern=pl.tile.MaskPattern.P0101)
                )
                out_val: pl.Tensor[[1, 2048], pl.FP32, pl.MemRef("mem_ddr_2", 0, 8192)] = pl.tile.store(
                    vals, [0, 0], val_output
                )
                return out_val

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected)


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
        ir.assert_structural_equal(After, Expected)

        # Sanity: split metadata round-trips through the pass.
        after_vp = After.get_function("vector_producer")
        assert after_vp is not None
        assert after_vp.split == ir.SplitMode.UP_DOWN


class TestStructuralShapeEquality:
    """Structural-equality tile compatibility.

    ``AreTileTypesCompatible`` used to compare shape/TileView expressions via
    pointer identity (with a ConstInt value-equality fallback). That missed
    freshly-allocated non-ConstInt expressions that were structurally identical
    — e.g. tiles produced by DeepClone — and blocked legitimate reuse. The pass
    now uses ``structural_equal`` so such tiles are recognised as compatible.
    """

    def test_pointer_distinct_but_structurally_equal_shape_reuses_memref(self):
        """Two tiles whose shape contains pointer-distinct composite expressions
        that are structurally identical must share a MemRef after memory_reuse.

        Constructing fresh ``Add(ConstInt(32), ConstInt(32))`` nodes for each
        tile simulates what DeepClone produces: identical tree shape, but
        freshly-allocated ``ExprPtr``s. The old pointer-equality check (with a
        ConstInt value-equality fallback) missed these non-ConstInt composite
        expressions and blocked reuse; ``structural_equal`` recurses into the
        tree and correctly recognises them as compatible.
        """
        span = ir.Span.unknown()
        c64 = ir.ConstInt(64, DataType.INT64, span)

        def make_add64() -> ir.Add:
            # Fresh Add(ConstInt(32), ConstInt(32)) — non-ConstInt expression
            # that is structurally equal across calls but pointer-distinct.
            return ir.Add(
                ir.ConstInt(32, DataType.INT64, span),
                ir.ConstInt(32, DataType.INT64, span),
                DataType.INT64,
                span,
            )

        add_1 = make_add64()
        add_2 = make_add64()
        assert add_1 is not add_2

        memref_a = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, DataType.INT64, span), 16384, 0)
        memref_b = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(16384, DataType.INT64, span), 16384, 1)

        input_x = ir.Var("input_x", ir.TensorType([64, 64], DataType.FP32), span)
        output_x = ir.Var("output_x", ir.TensorType([64, 64], DataType.FP32), span)

        tile_a = ir.Var(
            "tile_a",
            ir.TileType([add_1, c64], DataType.FP32, memref_a, memory_space=ir.MemorySpace.Vec),
            span,
        )
        tile_b = ir.Var(
            "tile_b",
            ir.TileType([add_2, c64], DataType.FP32, memref_b, memory_space=ir.MemorySpace.Vec),
            span,
        )
        store_a = ir.Var("store_a", ir.TensorType([64, 64], DataType.FP32), span)
        store_b = ir.Var("store_b", ir.TensorType([64, 64], DataType.FP32), span)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(tile_a, tile.load(input_x, offsets=[0, 0], shapes=[64, 64]), span),
                ir.AssignStmt(
                    store_a,
                    tile.store(tile_a, offsets=[0, 0], output_tensor=output_x),
                    span,
                ),
                ir.AssignStmt(tile_b, tile.load(input_x, offsets=[0, 0], shapes=[64, 64]), span),
                ir.AssignStmt(
                    store_b,
                    tile.store(tile_b, offsets=[0, 0], output_tensor=output_x),
                    span,
                ),
                ir.ReturnStmt(span),
            ],
            span,
        )
        func = ir.Function("main", [input_x, output_x], [], body, span, ir.FunctionType.InCore)
        Before = ir.Program([func], "test_struct_shape_reuse", span)

        with passes.PassContext([], passes.VerificationLevel.NONE):
            After = passes.memory_reuse()(Before)

        after_func = After.get_function("main")
        assert after_func is not None
        after_body = after_func.body
        assert isinstance(after_body, ir.SeqStmts)
        assign_a = after_body.stmts[0]
        assign_b = after_body.stmts[2]
        assert isinstance(assign_a, ir.AssignStmt)
        assert isinstance(assign_b, ir.AssignStmt)
        tile_a_type = assign_a.var.type
        tile_b_type = assign_b.var.type
        assert isinstance(tile_a_type, ir.TileType)
        assert isinstance(tile_b_type, ir.TileType)
        assert tile_a_type.shares_memref_with(tile_b_type)


class TestParallelPlaceholdersInIfThen:
    """Regression: two parallel `tile.create` placeholders inside an IfStmt
    then-branch, each feeding a sibling inner ForStmt, must NOT be aliased
    to the same buffer when both inner loops' results are simultaneously
    consumed at the if-then yield. Mirrors the kv_proj pattern that
    surfaces in qwen3_decode."""

    def test_parallel_placeholders_must_not_alias(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                cond_param: pl.Scalar[pl.INDEX],
                output_a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                output_b: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64, 64], pl.FP32], pl.Tensor[[64, 64], pl.FP32]]:
                outer_init_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                outer_init_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                if cond_param < 2:
                    inner_init_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                        [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                    )
                    for _i, (acc_a,) in pl.range(0, 4, init_values=(inner_init_a,)):
                        next_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_a, acc_a)
                        loop_a_out = pl.yield_(next_a)
                    inner_init_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                        [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                    )
                    for _j, (acc_b,) in pl.range(0, 4, init_values=(inner_init_b,)):
                        next_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_b, acc_b)
                        loop_b_out = pl.yield_(next_b)
                    phi_a, phi_b = pl.yield_(loop_a_out, loop_b_out)
                else:
                    phi_a, phi_b = pl.yield_(outer_init_a, outer_init_b)
                result_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(phi_a, [0, 0], output_a)
                result_b: pl.Tensor[[64, 64], pl.FP32] = pl.store(phi_b, [0, 0], output_b)
                return result_a, result_b

        After = _run_pipeline(Before)

        # Walk the IR after MemoryReuse and confirm inner_init_a and
        # inner_init_b are NOT on the same MemRef base.  They are simultaneously
        # consumed at the if-then yield, so aliasing them is a correctness bug
        # (the second loop's writes would clobber the first loop's value before
        # the if-then yield reads both).
        func = After.get_function("main")
        assert func is not None
        # Find the two inner_init AssignStmts inside the if-then body
        inits: dict[str, ir.MemRef] = {}

        def visit(stmt: ir.Stmt) -> None:
            if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint in ("inner_init_a", "inner_init_b"):
                t = stmt.var.type
                assert isinstance(t, ir.TileType)
                assert t.memref is not None
                inits[stmt.var.name_hint] = t.memref
            if isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    visit(s)
            elif isinstance(stmt, ir.IfStmt):
                visit(stmt.then_body)
                if stmt.else_body is not None:
                    visit(stmt.else_body)
            elif isinstance(stmt, ir.ForStmt):
                visit(stmt.body)

        visit(func.body)
        assert "inner_init_a" in inits, "inner_init_a not found in After IR"
        assert "inner_init_b" in inits, "inner_init_b not found in After IR"
        assert inits["inner_init_a"].base_ is not inits["inner_init_b"].base_, (
            f"inner_init_a and inner_init_b must NOT share MemRef base; both at "
            f"{inits['inner_init_a'].base_.name_hint}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
