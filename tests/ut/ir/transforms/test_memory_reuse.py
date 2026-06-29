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
from pypto import DataType, backend, ir, passes
from pypto.backend import BackendType
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
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_b, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
                )
                _result_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)] = pl.tile.store(
                    tile_a, [0, 0], output_a
                )
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_5, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_b, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec
                )
                _result_b: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_3", 0, 4096)] = pl.tile.store(
                    tile_b, [0, 0], output_b
                )
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
                )
                tile_f: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_5, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_b, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec
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
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
    """Tiles with different dtypes CAN reuse each other's memory.

    PTO codegen binds a per-var alloc_tile to each tile, so a BF16 tile may
    alias the buffer of a now-dead FP32 tile (each alloc_tile carries its own
    dtype/shape at the shared base). The former dtype-match reuse gate has
    been removed; in-place read-while-write hazards are handled by
    not_inplace_safe()/forbid_output_alias() instead.
    """

    def test_cross_dtype_can_reuse(self):
        """All tiles collapse onto one buffer regardless of FP32/BF16 dtype."""

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

        # With the dtype gate removed, all tiles chain-reuse one buffer:
        # tile_a/tile_b (FP32) and tile_cast/tile_d/tile_e (BF16) all share
        # mem_vec_2 (16384 bytes — sized for the largest, FP32, occupant).
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
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_a
                )
                tile_cast: pl.Tile[[64, 64], pl.BF16, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.cast(tile_b, target_type=pl.BF16, mode="round")
                )
                tile_d: pl.Tile[[64, 64], pl.BF16, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_cast, tile_cast
                )
                tile_e: pl.Tile[[64, 64], pl.BF16, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_d, tile_d
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_e, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected)


class TestFillpad:
    """fillpad outputs CAN reuse memory across differing TileView attributes.

    fillpad is a view/in-place-safe op (tile.fillpad aliases its input MemRef),
    so its padded output may share the input tile's buffer, and two padded
    tiles with different pad values may share one buffer too — differing
    TileView fields no longer block reuse now that the storage-attribute gate
    is gone. Each tile keeps its own view on its own alloc_tile at the shared
    base.
    """

    def test_fillpad_output_can_reuse_input(self):
        """fillpad output (pad view) reuses the input tile's buffer."""

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

        # tile_a (valid_shape=[48, 64]) and padded (pad view) both bind to
        # mem_vec_2: the differing TileView no longer blocks reuse, and fillpad
        # is in-place-safe so the output may alias its consumed input's buffer.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_2, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(valid_shape=[48, 64]),
                ] = pl.tile.load(input_a, [0, 0], [64, 64], [48, 64], target_memory=pl.Mem.Vec)
                padded: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_2, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(pad=pl.PadValue.max),
                ] = pl.tile.fillpad(tile_a, pad_value=pl.PadValue.max)
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    padded, [0, 0], output
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected)

    def test_fillpad_different_pad_can_reuse(self):
        """Two fillpad outputs with different pad values share one buffer."""

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

        # All four tiles chain-reuse mem_vec_3: tile_a/tile_b (valid_shape view)
        # and padded_max/padded_min (different pad views) have non-overlapping
        # lifetimes, and the differing TileView no longer blocks sharing.
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
                ] = pl.tile.load(input_a, [0, 0], [64, 64], [48, 64], target_memory=pl.Mem.Vec)
                padded_max: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_3, 0, 16384),
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
                ] = pl.tile.load(input_a, [0, 0], [64, 64], [48, 64], target_memory=pl.Mem.Vec)
                padded_min: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_3, 0, 16384),
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

        # All four tiles share mem_vec_3: tile_a/tile_b (valid_shape view) and
        # padded_a/padded_b (identical PadValue.max view) chain-reuse one buffer.
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
                ] = pl.tile.load(input_a, [0, 0], [64, 64], [48, 64], target_memory=pl.Mem.Vec)
                padded_a: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_3, 0, 16384),
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
                ] = pl.tile.load(input_a, [0, 0], [64, 64], [48, 64], target_memory=pl.Mem.Vec)
                padded_b: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_3, 0, 16384),
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
                ] = pl.tile.load(input_a, [0, 0], [64, 64], [48, 64], target_memory=pl.Mem.Vec)
                _res_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_a, [0, 0], output_a
                )
                tile_b: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_3, 0, 16384),
                    pl.Mem.Vec,
                    pl.TileView(valid_shape=[32, 64]),
                ] = pl.tile.load(input_a, [0, 0], [64, 64], [32, 64], target_memory=pl.Mem.Vec)
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)] = pl.tile.store(
                    tile_b, [0, 0], output_b
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_non_2d_divergent_valid_shape_can_reuse(self):
        """3D tiles with divergent ``valid_shape`` share a MemRef (gate removed)."""

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
        # With the reuse-compatibility gate removed, 3D tiles with divergent
        # valid_shape share a MemRef: each keeps its own valid_shape on its own
        # alloc_tile at the shared base (per-use metadata, not storage identity).
        bases = _collect_tile_memref_bases(After)
        tile_a_base = bases.get("tile_a")
        tile_b_base = bases.get("tile_b")
        assert tile_a_base is not None and tile_b_base is not None, (
            f"Expected tile_a and tile_b in After IR; got bases: {bases}"
        )
        assert tile_a_base == tile_b_base, (
            f"3D divergent-valid_shape tiles should share a MemRef, but bind to "
            f"{tile_a_base} and {tile_b_base}"
        )

    def test_view_present_vs_absent_can_reuse(self):
        """A tile carrying a storage-trivial view and a no-view tile share a MemRef.

        Reproduces the scenario from issue #1547: after SplitVectorKernel the
        two mutually-exclusive arms of a dual-AIV ``if`` are structural clones,
        but one arm's tiles carry a trivial ``valid_shape`` view while the
        other's carry none. A tile with no TileView has default physical
        storage; a tile whose view sets only ``valid_shape`` (default stride /
        offset / layout / fractal / pad) is physically identical, so the two
        must be allowed to share a backing allocation. Here ``tile_a`` (view)
        and the later ``tile_b`` (no view) have non-overlapping lifetimes and
        reuse one MemRef -- before the fix the ``has_view`` mismatch blocked it.
        """

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
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
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
                ] = pl.tile.load(input_a, [0, 0], [64, 64], [48, 64], target_memory=pl.Mem.Vec)
                _res_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_a, [0, 0], output_a
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)] = pl.tile.store(
                    tile_b, [0, 0], output_b
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


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

    def test_subview_group_keeps_offsets_on_reuse(self):
        """Retargeting a sharing group must preserve per-member subview offsets (issue #1723).

        ``dead`` dies before ``src``, so ``src`` retargets onto ``dead``'s buffer.
        ``srcT`` transposes ``src``; tile.transpose is not in-place safe, so it
        gets a buffer distinct from ``src``, and its slice/reshape view group
        shares that fresh base. The two per-row slices sit at byte offsets 0 and
        64 within the group; they must keep those distinct offsets, not collapse
        onto the base offset.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                inp: pl.Tensor[[16, 8], pl.FP32],
                dead_in: pl.Tensor[[16, 8], pl.FP32],
                out_dead: pl.Out[pl.Tensor[[16, 8], pl.FP32]],
                out0: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
                out1: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                dead: pl.Tile[[16, 8], pl.FP32, pl.MemorySpace.Vec] = pl.load(dead_in, [0, 0], [16, 8])
                _sd: pl.Tensor[[16, 8], pl.FP32] = pl.store(dead, [0, 0], out_dead)
                src: pl.Tile[[16, 8], pl.FP32, pl.MemorySpace.Vec] = pl.load(inp, [0, 0], [16, 8])
                srcT: pl.Tile[[8, 16], pl.FP32, pl.MemorySpace.Vec] = pl.transpose(src, axis1=0, axis2=1)
                # Slices authored as separate stmts: the isolated pipeline skips
                # FlattenCallExpr, so an inline slice would not join the group.
                s0: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.slice(srcT, [1, 16], [0, 0])
                r0: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(s0, [16, 1])
                s1: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.slice(srcT, [1, 16], [1, 0])
                r1: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(s1, [16, 1])
                _o0: pl.Tensor[[16, 1], pl.FP32] = pl.store(r0, [0, 0], out0)
                result: pl.Tensor[[16, 1], pl.FP32] = pl.store(r1, [0, 0], out1)
                return result

        After = _run_pipeline(Before)
        func = After.get_function("main")
        assert func is not None
        body = func.body
        assert isinstance(body, ir.SeqStmts)
        members = {}
        for stmt in body.stmts:
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.var.type, ir.TileType):
                mr = stmt.var.type.memref
                assert mr is not None
                off = mr.byte_offset_
                assert isinstance(off, ir.ConstInt)
                members[stmt.var.name_hint] = (mr.base_.name_hint, off.value, mr.size_)

        # src retargets onto dead's buffer (reuse actually happened).
        assert members["src"][0] == members["dead"][0]
        # tile.transpose is not in-place safe, so srcT gets a buffer distinct
        # from src; the whole view group (srcT + its slices/reshapes) shares
        # that one fresh base.
        base = members["srcT"][0]
        assert base != members["src"][0], (
            "srcT must not reuse the src buffer (transpose is not in-place safe)"
        )
        for name in ("srcT", "s0", "r0", "s1", "r1"):
            assert members[name][0] == base, f"{name} not on shared base {base}"
        # Row 0 slice/reshape at offset 0; row 1 slice/reshape at offset 64 — the
        # offsets must NOT collapse (pre-fix bug put all four at 0).
        assert members["s0"][1] == 0 and members["r0"][1] == 0
        assert members["s1"][1] == 64 and members["r1"][1] == 64
        # Each member keeps its own 64-byte size, not the target's 512.
        assert members["r0"][2] == 64 and members["r1"][2] == 64

    def test_loop_carry_retarget_keeps_slice_view_aliased(self):
        """A sub-region view must follow its input when the input is loop-carry
        retargeted (issue #1776 follow-up).

        ``ti`` is reloaded each iteration and yielded back, so the loop-carry
        retargeter (``PropagateFromForStmt``) aligns it onto the iter_arg buffer
        (``t0``'s). The ``si = slice(ti)`` sub-region view is an off-chain
        consumer: without forward propagation its declared MemRef stays on
        ``ti``'s original (now-orphaned) buffer, so the view would read a buffer
        the reload never wrote. After the fix ``si`` re-anchors onto ``ti``'s
        retargeted buffer.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                inp: pl.Tensor[[16, 16], pl.FP32],
                out_acc: pl.Out[pl.Tensor[[16, 8], pl.FP32]],
            ) -> pl.Tensor[[16, 8], pl.FP32]:
                t0: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(inp, [0, 0], [16, 16])
                s0: pl.Tile[[16, 8], pl.FP32, pl.MemorySpace.Vec] = pl.slice(t0, [16, 8], [0, 0])
                # Independent accumulator so t0's buffer is free for the reload to reuse.
                acc0: pl.Tile[[16, 8], pl.FP32, pl.MemorySpace.Vec] = pl.add(s0, s0)
                for _i, (t_c, acc_c) in pl.range(0, 4, init_values=(t0, acc0)):
                    ti: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(inp, [0, 0], [16, 16])
                    si: pl.Tile[[16, 8], pl.FP32, pl.MemorySpace.Vec] = pl.slice(ti, [16, 8], [0, 0])
                    acc_n: pl.Tile[[16, 8], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_c, si)
                    _t_y, acc_y = pl.yield_(ti, acc_n)
                result: pl.Tensor[[16, 8], pl.FP32] = pl.store(acc_y, [0, 0], out_acc)
                return result

        bases = _collect_tile_memref_bases(_run_pipeline(Before))
        # The scenario is real: the reload retargeted onto the carried buffer.
        assert bases["ti"] == bases["t0"], "expected loop-carry retarget of the reload"
        # The fix: the sub-region view follows its retargeted input (not orphaned).
        assert bases["si"] == bases["ti"], (
            f"slice view orphaned: si on {bases['si']} but its input ti on {bases['ti']}"
        )

    def test_loop_carry_retarget_keeps_reshape_view_aliased(self):
        """Full-alias view (reshape) variant of the loop-carry orphan guard.

        Same shape as the slice test but the off-chain view is a whole-buffer
        ``reshape`` (the pure-alias path shared with ``tile.transpose_view``,
        the original #1776 regression). It must also follow the retargeted input.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                inp: pl.Tensor[[16, 16], pl.FP32],
                out_acc: pl.Out[pl.Tensor[[256, 1], pl.FP32]],
            ) -> pl.Tensor[[256, 1], pl.FP32]:
                t0: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(inp, [0, 0], [16, 16])
                r0: pl.Tile[[256, 1], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(t0, [256, 1])
                acc0: pl.Tile[[256, 1], pl.FP32, pl.MemorySpace.Vec] = pl.add(r0, r0)
                for _i, (t_c, acc_c) in pl.range(0, 4, init_values=(t0, acc0)):
                    ti: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(inp, [0, 0], [16, 16])
                    ri: pl.Tile[[256, 1], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(ti, [256, 1])
                    acc_n: pl.Tile[[256, 1], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_c, ri)
                    _t_y, acc_y = pl.yield_(ti, acc_n)
                result: pl.Tensor[[256, 1], pl.FP32] = pl.store(acc_y, [0, 0], out_acc)
                return result

        bases = _collect_tile_memref_bases(_run_pipeline(Before))
        assert bases["ti"] == bases["t0"], "expected loop-carry retarget of the reload"
        assert bases["ri"] == bases["ti"], (
            f"reshape view orphaned: ri on {bases['ri']} but its input ti on {bases['ti']}"
        )

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
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
                )
                _tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_c, tile_c
                )
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
                )
                _tile_b: pl.Tile[[4096, 1], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.reshape(tile_a, [4096, 1])
                )
                _tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_a
                )
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec
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
                    input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec
                )
                _s1: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_3", 0, 4096)] = pl.tile.store(
                    tile_a, [0, 0], output
                )
                tile_c: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_4, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_c, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec
                )
                _s2: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_3", 0, 4096)] = pl.tile.store(
                    tile_c, [0, 0], output
                )
                tile_x: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_4, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_x, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec
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
                    input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec
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
                    input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec
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
                    input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec
                )
                tile_tmp: pl.Tile[[32, 32], pl.INT32, pl.MemRef(mem_vec_4, 0, 4096), pl.Mem.Vec] = (
                    pl.tile.load(input_b, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec)
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
                    input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec
                )
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_3, 0, 4096), pl.Mem.Vec] = pl.tile.add(
                    tile_a, tile_a
                )
                _s1: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)] = pl.tile.store(
                    tile_b, [0, 0], output
                )
                tile_u: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_vec_3, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                    input_u, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec
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
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
                )
                init_1: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_b, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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

    def test_if_phi_read_after_branch_blocks_reuse(self):
        """An scf.if return_var (phi) read *after* the branch keeps its buffer
        live; a later temporary must not be packed onto it.

        Regression test for issue #1821: the lifetime analyzer did not track
        IfStmt return_vars, so reads of a phi after the if were dropped and its
        buffer's live range collapsed at the yield.  A later temporary (`e`) was
        then packed onto the still-live phi buffer, corrupting the value `f`
        reads (manifested as ~17.5% wrong output in the dsv4 prefill_sparse_attn
        merge_norm kernel).

        Here `a` stays live across the whole body (used by `e` and `g`), so its
        buffer is occupied and `e` would otherwise fall back onto the phi's
        buffer.  Pre-fix, `e` took the phi buffer `mem_vec_3` (clobbering `r`
        before `f = r + e` read it); post-fix, `e` gets its own `mem_vec_6`.  The
        branch-local sources `b`/`c` still share one buffer (`mem_vec_3`),
        proving the fix keeps legitimate mutually-exclusive branch sharing.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                if cond_param < 2:
                    b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(a, a)
                    r = pl.yield_(b)
                else:
                    c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.mul(a, a)
                    r = pl.yield_(c)
                e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(a, a)
                f: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(r, e)
                g: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(a, f)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(g, [0, 0], output)
                return result

        # a → mem_vec_2 (live to the end). b/c/r (phi) share mem_vec_3. e must
        # NOT take mem_vec_3 (the phi is live until f reads it) → e gets mem_vec_6.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
                )
                if cond_param < 2:
                    b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.add(
                        a, a
                    )
                    r: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.yield_(b)
                else:
                    c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.mul(
                        a, a
                    )
                    r: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.yield_(c)
                e: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_6, 0, 16384), pl.Mem.Vec] = pl.tile.add(a, a)
                f: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.add(r, e)
                g: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.add(a, f)
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    g, [0, 0], output
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
                        pl.tile.load(input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec)
                    )
                    acc_next: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(acc, chunk)
                    )
                    loop_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(acc_next)
                    )
                resid: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_b, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    pl.tile.load(input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Mat)
                )
                tile_b_l1: pl.Tile[[32, 32], pl.FP16, pl.MemRef(mem_mat_4, 0, 2048), pl.Mem.Mat] = (
                    pl.tile.load(input_b, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Mat)
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
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
        # on mem_vec_5) and dst (merged on mem_vec_6) differ — the retargeter
        # still declines because mrgsort cannot run src==dst.  YieldFixup
        # inserts merged_mv on mem_vec_5 so the yield matches the iter_arg.
        # Global largest-first packing additionally lets the two 8 KB tiles
        # src_tile and vals (both lifetime-disjoint from `merged`) reuse
        # `merged`'s larger 16 KB mem_vec_6 buffer, so only three Vec buffers
        # are allocated instead of four.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                src_tensor: pl.Tensor[[1, 2048], pl.FP32, pl.MemRef("mem_ddr_0", 0, 8192)],
                idx_tensor: pl.Tensor[[1, 2048], pl.UINT32, pl.MemRef("mem_ddr_1", 0, 8192)],
                val_output: pl.Out[pl.Tensor[[1, 2048], pl.FP32, pl.MemRef("mem_ddr_2", 0, 8192)]],
            ) -> pl.Tensor[[1, 2048], pl.FP32]:
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
                mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                src_tile: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_6, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.load(src_tensor, [0, 0], [1, 2048], [1, 2048], target_memory=pl.Mem.Vec)
                )
                idx_tile: pl.Tile[[1, 2048], pl.UINT32, pl.MemRef(mem_vec_4, 0, 8192), pl.Mem.Vec] = (
                    pl.tile.load(idx_tensor, [0, 0], [1, 2048], [1, 2048], target_memory=pl.Mem.Vec)
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
                vals: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_6, 0, 16384), pl.Mem.Vec] = (
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
                    input_tensor, [0, 0], [16, 16], [16, 16], target_memory=pl.Mem.Vec
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


class TestL0CrossShapeReuse:
    """L0 cube-input buffers (Left/Right) hold sub-tiles produced by view ops
    (tile.extract), which codegen materialises per tile var at the buffer base.

    Two such buffers in the same L0 space, with non-overlapping lifetimes and
    sufficient byte size, may therefore share one slot even when their *shapes*
    differ — unlike Vec/Acc/Mat buffers, which keep the strict shape match.
    This is what lets fused-attention reuse the QK Right buffer ([k, SEQ]) for
    the PV Right buffer ([k', HEAD]) (issue #1595)."""

    def test_right_buffers_different_shapes_reuse(self):
        """``rb`` ([64, 256] Right) is dead before ``rd`` ([128, 128] Right) is
        born; both are 32 KB extract sub-tiles, so ``rd`` reuses ``rb``'s buffer
        despite the differing shape.  ``la`` ([16, 64] Left, 2 KB) is dead before
        ``lc`` ([16, 128] Left, 4 KB) is born; global largest-first packing makes
        the larger ``lc`` the buffer representative and lets the smaller, earlier
        ``la`` share it — cross-shape L0 reuse is bidirectional."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 64], pl.BF16],
                b: pl.Tensor[[64, 256], pl.BF16],
                c: pl.Tensor[[16, 128], pl.BF16],
                d: pl.Tensor[[128, 128], pl.BF16],
                out1: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
                out2: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_mat: pl.Tile[[16, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a, [0, 0], [16, 64], target_memory=pl.Mem.Mat
                )
                b_mat: pl.Tile[[64, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b, [0, 0], [64, 256], target_memory=pl.Mem.Mat
                )
                la: pl.Tile[[16, 64], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    a_mat, 0, 0, [16, 64], target_memory=pl.Mem.Left
                )
                rb: pl.Tile[[64, 256], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    b_mat, 0, 0, [64, 256], target_memory=pl.Mem.Right
                )
                m1: pl.Tile[[16, 256], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(la, rb)
                out1 = pl.store(m1, [0, 0], out1)
                c_mat: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    c, [0, 0], [16, 128], target_memory=pl.Mem.Mat
                )
                d_mat: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    d, [0, 0], [128, 128], target_memory=pl.Mem.Mat
                )
                lc: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    c_mat, 0, 0, [16, 128], target_memory=pl.Mem.Left
                )
                rd: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    d_mat, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                m2: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lc, rd)
                out2 = pl.store(m2, [0, 0], out2)
                return out2

        After = _run_pipeline(Before)

        # Collect the MemRef base of each extract-produced L0 tile.
        func = After.get_function("kernel")
        assert func is not None
        bases: dict[str, ir.Var] = {}

        def visit(stmt: ir.Stmt) -> None:
            if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint in ("la", "rb", "lc", "rd"):
                t = stmt.var.type
                assert isinstance(t, ir.TileType)
                assert t.memref is not None
                bases[stmt.var.name_hint] = t.memref.base_
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
        for name in ("la", "rb", "lc", "rd"):
            assert name in bases, f"{name} not found in After IR"

        # rb ([64,256]) and rd ([128,128]) are different shapes but reuse the
        # same Right buffer — the cross-shape L0 reuse this pass now allows.
        assert bases["rb"] is bases["rd"], (
            f"rd ([128,128] Right) must reuse rb's ([64,256] Right) buffer; "
            f"got rb@{bases['rb'].name_hint} vs rd@{bases['rd'].name_hint}"
        )
        # la ([16,64] Left, 2 KB) is dead before lc ([16,128] Left, 4 KB) is born.
        # Global largest-first packing makes the larger lc the representative and
        # lets the smaller, earlier la share its buffer — the former one-directional
        # size gate (source.size >= target.size) could never capture this. Saves 2 KB
        # of L0A; lc's 4 KB buffer is large enough to hold la.
        assert bases["la"] is bases["lc"], (
            "la ([16,64]) should reuse lc's ([16,128]) larger L0A buffer under global "
            f"packing; got la@{bases['la'].name_hint} vs lc@{bases['lc'].name_hint}"
        )


class TestAscend910BLoadTpopHazard:
    """MemoryReuse must not coalesce a writer that consumes a tile.load result
    and a tile.tpop_from_aic value into the load's buffer on Ascend910B split-AIV
    functions — that in-place sharing is a silent hardware hazard.  This guard
    folds in the responsibility formerly owned by LegalizePTOBufferReuse.
    """

    @staticmethod
    def _build_program():
        """down_next = tile.add(down_prev=tile.load, pipe_chunk=tile.tpop_from_aic).

        Each tile starts in its own buffer (pre-MemoryReuse state).  ``down_prev``
        and ``pipe_chunk`` are both last-used at the ``tile.add``, so without the
        hazard guard MemoryReuse would in-place-reuse ``down_prev``'s buffer for
        ``down_next``.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main(self, down: pl.InOut[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                mem_vec_0: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                down_prev: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_0, 0, 4096), pl.Mem.Vec] = (
                    pl.tile.load(down, [0, 0], [8, 128], [8, 128], target_memory=pl.Mem.Vec)
                )
                pipe_chunk: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_1, 0, 4096), pl.Mem.Vec] = (
                    pl.tile.tpop_from_aic(split=1)
                )
                down_next: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_2, 0, 4096), pl.Mem.Vec] = (
                    pl.tile.add(down_prev, pipe_chunk)
                )
                result: pl.Tensor[[16, 128], pl.FP32] = pl.tile.store(down_next, [0, 0], down)
                return result

        return Prog

    def test_ascend910b_split_aiv_does_not_reuse_load_buffer(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        try:
            After = passes.memory_reuse()(self._build_program())
        finally:
            backend.reset_for_testing()

        bases = _collect_tile_memref_bases(After)
        assert "down_prev" in bases and "down_next" in bases, f"missing tile vars; got {bases}"
        assert bases["down_next"] != bases["down_prev"], (
            "Ascend910B split-AIV: tile.add output must NOT reuse the tile.load buffer "
            f"(load+tpop_from_aic hazard), but both bind to {bases['down_prev']}"
        )

    def test_ascend950_allows_load_buffer_reuse(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend950)
        try:
            After = passes.memory_reuse()(self._build_program())
        finally:
            backend.reset_for_testing()

        bases = _collect_tile_memref_bases(After)
        assert "down_prev" in bases and "down_next" in bases, f"missing tile vars; got {bases}"
        assert bases["down_next"] == bases["down_prev"], (
            "Ascend950 has no load+tpop hazard, so MemoryReuse should in-place-reuse the "
            f"load buffer for the tile.add output; got down_next={bases['down_next']} "
            f"down_prev={bases['down_prev']}"
        )


class TestForbidOutputAlias:
    """A tile.sel output must not alias its mask (arg 0) or tmp (arg 3) buffer.

    The TSEL intrinsic reads the predicate mask and the tmp scratch while
    writing dst, so an in-place write onto either would corrupt the op
    mid-flight (wrong select results on Ascend a2a3). tile.sel declares these
    via OpRegistryEntry::forbid_output_alias(); MemoryReuse honours the marker
    even when shape/dtype would otherwise permit the reuse.
    """

    def test_sel_output_does_not_alias_mask_or_tmp(self):
        """dst skips the mask buffer (large enough to hold it) and reuses a value operand."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                tmp_in: pl.Tensor[[1, 32], pl.UINT8],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                # t1 (FP32 16x16, 1024B) dies at the cmp, so its buffer is free
                # and large enough for the sel output. The mask reuses it; the
                # forbid_output_alias marker is the only thing keeping dst off it.
                t0: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(a, [0, 0], [16, 16])
                t1: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.add(t0, t0)
                t2: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(b, [0, 0], [16, 16])
                mask: pl.Tile[[16, 32], pl.UINT8, pl.MemorySpace.Vec] = pl.cmp(t1, t2, cmp_type=0)
                tmp: pl.Tile[[1, 32], pl.UINT8, pl.MemorySpace.Vec] = pl.load(tmp_in, [0, 0], [1, 32])
                dst: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.sel(mask, t2, t2, tmp)
                res: pl.Tensor[[16, 16], pl.FP32] = pl.store(dst, [0, 0], out)
                return res

        After = _run_pipeline(Before)
        bases = _collect_tile_memref_bases(After)
        for name in ("dst", "mask", "tmp"):
            assert name in bases, f"Expected {name} in After IR; got bases: {bases}"

        # The mask reuses the dead 1024B FP32 buffer — big enough to hold dst —
        # so without the marker the greedy allocator would place dst there.
        assert bases["dst"] != bases["mask"], (
            f"tile.sel output must not alias its mask buffer, but both bind to {bases['dst']}"
        )
        assert bases["dst"] != bases["tmp"], (
            f"tile.sel output must not alias its tmp buffer, but both bind to {bases['dst']}"
        )

    def test_row_sum_output_does_not_alias_input_or_tmp(self):
        """A row reduction output must not share a buffer with its input or tmp.

        ``tile.row_sum`` reads the full input row and the tmp scratch while
        writing the reduced ``[M, 1]`` output, so it is ``not_inplace_safe``.
        Here ``sq`` (the squared input, reusing ``t0``) and ``tmp`` both die at
        the reduction and are large enough to hold the small output, so without
        the marker the greedy allocator would place ``s`` on one of them.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                tmp_in: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                t0: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(a, [0, 0], [16, 16])
                sq: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.mul(t0, t0)
                tmp: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(tmp_in, [0, 0], [16, 16])
                s: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.row_sum(sq, tmp)
                res: pl.Tensor[[16, 1], pl.FP32] = pl.store(s, [0, 0], out)
                return res

        After = _run_pipeline(Before)
        bases = _collect_tile_memref_bases(After)
        for name in ("s", "sq", "tmp"):
            assert name in bases, f"Expected {name} in After IR; got bases: {bases}"
        assert bases["s"] != bases["sq"], (
            f"row_sum output must not alias its input buffer, but both bind to {bases['s']}"
        )
        assert bases["s"] != bases["tmp"], (
            f"row_sum output must not alias its tmp buffer, but both bind to {bases['s']}"
        )

    def test_forbidden_input_reached_through_view_is_honored(self):
        """A not_inplace_safe op reading a VIEW of its input must still not alias it.

        ``tile.recip`` is ``not_inplace_safe``. Its input ``v`` is a reshape
        *view* of ``t0`` (sharing ``t0``'s MemRef base), and ``t0`` dies at the
        recip, so the recip output ``r`` is the same size and would greedily
        reuse ``t0``'s buffer. A Var-identity-only guard misses this (``v`` is a
        view with no reuse-map entry); the guard must resolve the operand to its
        physical base and keep ``r`` off it. Mirrors the on-device gather /
        qk_recip corruption the gate removal exposed.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[8, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            ) -> pl.Tensor[[8, 8], pl.FP32]:
                t0: pl.Tile[[8, 8], pl.FP32, pl.MemorySpace.Vec] = pl.load(x, [0, 0], [8, 8])
                v: pl.Tile[[64, 1], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(t0, [64, 1])
                r: pl.Tile[[64, 1], pl.FP32, pl.MemorySpace.Vec] = pl.recip(v)
                r2: pl.Tile[[8, 8], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(r, [8, 8])
                res: pl.Tensor[[8, 8], pl.FP32] = pl.store(r2, [0, 0], out)
                return res

        After = _run_pipeline(Before)
        bases = _collect_tile_memref_bases(After)
        for name in ("r", "t0", "v"):
            assert name in bases, f"Expected {name} in After IR; got bases: {bases}"
        # ``v`` shares ``t0``'s base (it is a view); the recip output must not
        # land on that physical buffer even though ``v`` itself is the operand.
        assert bases["r"] != bases["t0"], (
            f"recip output must not alias its (viewed) input's buffer, but both bind to {bases['r']}"
        )

    def test_widening_cast_output_does_not_alias_input(self):
        """A dtype-widening cast output must not alias its (narrower) input.

        Element i is read at ``i*in_bytes`` but written at ``i*out_bytes``; with
        the output wider, the write cursor outruns the read cursor and clobbers
        input elements not yet converted. The bf16 input here reuses a dead FP32
        buffer (cross-dtype reuse) so it is large enough to hold the FP32 output,
        making the in-place upcast reachable — the guard must forbid it.
        Narrowing / same-width casts stay in-place-safe.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[8, 16], pl.FP32],
                b: pl.Tensor[[8, 16], pl.BF16],
                out: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
            ) -> pl.Tensor[[8, 16], pl.FP32]:
                t0: pl.Tile[[8, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(a, [0, 0], [8, 16])
                _dead: pl.Tile[[8, 16], pl.FP32, pl.MemorySpace.Vec] = pl.add(t0, t0)
                bf: pl.Tile[[8, 16], pl.BF16, pl.MemorySpace.Vec] = pl.load(b, [0, 0], [8, 16])
                r: pl.Tile[[8, 16], pl.FP32, pl.MemorySpace.Vec] = pl.cast(bf, target_type=pl.FP32)
                res: pl.Tensor[[8, 16], pl.FP32] = pl.store(r, [0, 0], out)
                return res

        After = _run_pipeline(Before)
        bases = _collect_tile_memref_bases(After)
        for name in ("r", "bf"):
            assert name in bases, f"Expected {name} in After IR; got bases: {bases}"
        assert bases["r"] != bases["bf"], (
            f"widening cast output must not alias its input buffer, but both bind to {bases['r']}"
        )

    def test_col_expand_mul_output_does_not_alias_col_vector(self):
        """col_expand_mul output must not alias its broadcast column vector.

        ``out[i, j] = target[i, j] * col[0, j]`` re-reads the column vector for
        every output row, so an output that aliases the column buffer overwrites
        it after row 0 and multiplies later rows by garbage. ``col`` here is a
        view of a dead [8, 16] tile, so its buffer is large enough for the output
        to greedily reuse — the forbid_output_alias(1) marker must prevent it.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[8, 16], pl.FP32],
                c: pl.Tensor[[8, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
            ) -> pl.Tensor[[8, 16], pl.FP32]:
                t0: pl.Tile[[8, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(a, [0, 0], [8, 16])
                tgt: pl.Tile[[8, 16], pl.FP32, pl.MemorySpace.Vec] = pl.add(t0, t0)
                cbig: pl.Tile[[8, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(c, [0, 0], [8, 16])
                col_src: pl.Tile[[8, 16], pl.FP32, pl.MemorySpace.Vec] = pl.add(cbig, cbig)
                col: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.slice(col_src, [1, 16], [0, 0])
                r: pl.Tile[[8, 16], pl.FP32, pl.MemorySpace.Vec] = pl.col_expand_mul(tgt, col)
                res: pl.Tensor[[8, 16], pl.FP32] = pl.store(r, [0, 0], out)
                return res

        After = _run_pipeline(Before)
        bases = _collect_tile_memref_bases(After)
        for name in ("r", "col", "col_src"):
            assert name in bases, f"Expected {name} in After IR; got bases: {bases}"
        # ``col`` is a view of ``col_src``; the expand output must not land on
        # that physical buffer (it re-reads the column for every row).
        assert bases["r"] != bases["col_src"], (
            f"col_expand_mul output must not alias its column vector's buffer, but both bind to {bases['r']}"
        )

    def test_rsqrt_output_does_not_alias_input(self):
        """tile.rsqrt output must not alias its input (``not_inplace_safe``).

        Like ``tile.recip``, ``rsqrt``'s high-precision lowering reads the input
        while writing the output, so it is marked ``not_inplace_safe`` (the tmp
        scratch is injected by a later pass, so at MemoryReuse the only operand
        is the input). ``sq`` (reusing ``t0``) dies at the rsqrt and is the same
        size as the output, so without the marker the output would reuse it.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t0: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(a, [0, 0], [16, 16])
                sq: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.mul(t0, t0)
                r: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.rsqrt(sq)
                res: pl.Tensor[[16, 16], pl.FP32] = pl.store(r, [0, 0], out)
                return res

        After = _run_pipeline(Before)
        bases = _collect_tile_memref_bases(After)
        for name in ("r", "sq"):
            assert name in bases, f"Expected {name} in After IR; got bases: {bases}"
        assert bases["r"] != bases["sq"], (
            f"rsqrt output must not alias its input buffer, but both bind to {bases['r']}"
        )


class TestPipelineStageSeparation:
    """``pl.pipeline`` stage tiles must stay on distinct buffers (ping-pong).

    ``LowerPipelineLoops`` tags each replicated clone's tile-producing ``Call``
    with a ``pipeline_membership`` ``(group, stage)`` attr; ``MemoryReuse`` then
    refuses to coalesce two tiles that share a group with different stages, even
    when their program-order lifetimes are disjoint (the disjointness pipelining
    deliberately hides). These tests run the lower → reuse chain WITHOUT
    ``CanonicalizeIOOrder`` so the separation is attributable to the explicit
    constraint alone, not to clustering-induced lifetime overlap.
    """

    @staticmethod
    def _lower_then_reuse(program: ir.Program) -> ir.Program:
        # Deliberately skip CanonicalizeIOOrder: the two clones' loads are then
        # NOT co-live (program-order-disjoint lifetimes), the exact condition
        # under which MemoryReuse would otherwise coalesce them into one buffer.
        p = passes.lower_pipeline_loops()(program)
        p = passes.materialize_tensor_strides()(p)
        p = passes.init_mem_ref()(p)
        return passes.memory_reuse()(p)

    @staticmethod
    def _tile_def_bases(program: ir.Program) -> list[str]:
        """Return the MemRef base name of every tile-typed AssignStmt, in order.

        Unlike ``_collect_tile_memref_bases``, this keeps duplicates: the two
        pipeline clones share a ``name_hint`` (``t``) — the printer only adds the
        ``_1`` suffix for display — so a name-keyed dict would collapse them.
        """
        bases: list[str] = []
        main_func = next(iter(program.functions.values()))

        class _Collector(ir.IRVisitor):
            def visit_assign_stmt(self, stmt):  # type: ignore[override]
                var_type = stmt.var.type
                if isinstance(var_type, ir.TileType) and var_type.memref is not None:
                    bases.append(var_type.memref.base_.name_hint)
                super().visit_assign_stmt(stmt)

        _Collector().visit_stmt(main_func.body)
        return bases

    def test_stage_tiles_get_distinct_buffers(self):
        """The two stage clones of a ``stage=2`` pipeline never share a buffer."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(
                self,
                x: pl.Tensor[[512], pl.FP32],
                out: pl.Out[pl.Tensor[[512], pl.FP32]],
            ) -> pl.Tensor[[512], pl.FP32]:
                for i in pl.pipeline(0, 4, 1, stage=2):
                    t: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec] = pl.tile.load(
                        x, [i * 128], [128], [128], target_memory=pl.MemorySpace.Vec
                    )
                    _r: pl.Tensor[[512], pl.FP32] = pl.tile.store(t, [i * 128], out)
                return x

        After = self._lower_then_reuse(Before)
        bases = self._tile_def_bases(After)
        # Two clones (stage 0 and stage 1) → two tile defs on distinct buffers.
        assert len(bases) == 2, f"expected two tile defs (one per stage clone), got {bases}"
        assert len(set(bases)) == 2, (
            f"pipeline stage clones must occupy distinct buffers, but bind to {bases}"
        )
        # MemoryReuse consumes pipeline_membership and must strip it so the attr
        # does not ride downstream into codegen.
        assert "pipeline_membership" not in ir.python_print(After), (
            "pipeline_membership attr must be stripped after MemoryReuse"
        )

    @staticmethod
    def _tile_defs_with_role(program: ir.Program) -> list[tuple[bool, str]]:
        """Return ``(is_load, memref_base_name)`` for every tile-typed AssignStmt.

        ``is_load`` is True when the defining op is ``tile.load`` / ``tile.read``.
        """
        defs: list[tuple[bool, str]] = []
        main_func = next(iter(program.functions.values()))

        class _Collector(ir.IRVisitor):
            def visit_assign_stmt(self, stmt):  # type: ignore[override]
                var_type = stmt.var.type
                if isinstance(var_type, ir.TileType) and var_type.memref is not None:
                    val = stmt.value
                    is_load = isinstance(val, ir.Call) and val.op.name in ("tile.load", "tile.read")
                    defs.append((is_load, var_type.memref.base_.name_hint))
                super().visit_assign_stmt(stmt)

        _Collector().visit_stmt(main_func.body)
        return defs

    def test_load_buffers_separate_but_compute_may_coalesce(self):
        """Role-aware granularity: load buffers stay per-stage; compute relaxes.

        Forbidding *all* cross-stage reuse (depth = F) overflows real kernels, so
        only load buffers are kept private. Here a ``stage=2`` body has one load
        and one compute per clone: the two load buffers must differ (ping-pong),
        but the total buffer count stays below ``depth = F`` because the compute
        tile coalesces (here in-place onto its own stage's load) instead of
        demanding a fourth independent buffer.
        """

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(
                self,
                x: pl.Tensor[[512], pl.FP32],
                out: pl.Out[pl.Tensor[[512], pl.FP32]],
            ) -> pl.Tensor[[512], pl.FP32]:
                for i in pl.pipeline(0, 4, 1, stage=2):
                    ld: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec] = pl.tile.load(
                        x, [i * 128], [128], [128], target_memory=pl.MemorySpace.Vec
                    )
                    c: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec] = pl.tile.exp(ld)
                    _r: pl.Tensor[[512], pl.FP32] = pl.tile.store(c, [i * 128], out)
                return x

        After = self._lower_then_reuse(Before)
        defs = self._tile_defs_with_role(After)
        load_bases = [base for is_load, base in defs if is_load]
        all_bases = [base for _, base in defs]
        # Both stage loads present and on distinct buffers (ping-pong preserved).
        assert len(load_bases) == 2, f"expected one load per stage, got {defs}"
        assert len(set(load_bases)) == 2, f"stage load buffers must differ, got {load_bases}"
        # Compute relaxation: fewer buffers than depth = F (4 tile defs → 4 buffers).
        assert len(set(all_bases)) < len(defs), (
            f"compute tiles should coalesce instead of forcing depth=F separation, got {defs}"
        )

    def test_disjoint_non_pipeline_tiles_still_merge(self):
        """Control: identical disjoint tiles WITHOUT pipeline tags do coalesce.

        Confirms the harness merges lifetime-disjoint tiles by default, so the
        separation asserted above is the pipeline constraint at work — not an
        artifact of the tiles being inherently unmergeable.
        """

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(
                self,
                x: pl.Tensor[[512], pl.FP32],
                out: pl.Out[pl.Tensor[[512], pl.FP32]],
            ) -> pl.Tensor[[512], pl.FP32]:
                a: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec] = pl.tile.load(
                    x, [0], [128], [128], target_memory=pl.MemorySpace.Vec
                )
                _r0: pl.Tensor[[512], pl.FP32] = pl.tile.store(a, [0], out)
                b: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec] = pl.tile.load(
                    x, [128], [128], [128], target_memory=pl.MemorySpace.Vec
                )
                _r1: pl.Tensor[[512], pl.FP32] = pl.tile.store(b, [128], out)
                return x

        After = passes.memory_reuse()(passes.init_mem_ref()(passes.materialize_tensor_strides()(Before)))
        bases = _collect_tile_memref_bases(After)
        assert "a" in bases and "b" in bases, f"Expected both tiles; got {bases}"
        assert bases["a"] == bases["b"], (
            f"disjoint non-pipeline tiles should merge, but bind to {bases['a']} and {bases['b']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
