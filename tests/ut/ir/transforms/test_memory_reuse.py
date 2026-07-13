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
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _run_pipeline(program: ir.Program) -> ir.Program:
    """Run init_mem_ref + materialize_semantic_aliases + memory_reuse pipeline.

    The loop-carry / in-place must-alias retarget (formerly MemoryReuse "Step 0")
    lives in materialize_semantic_aliases, which runs between init_mem_ref and
    memory_reuse in the real pipeline — compose all three here so these unit
    tests exercise the same combined transformation.
    """
    return passes.memory_reuse()(passes.materialize_semantic_aliases()(passes.init_mem_ref()(program)))


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

    def test_concat_output_must_not_alias_either_source(self):
        """tile.concat's output must get a buffer distinct from both sources.

        pto.tconcat copies row by row, and dst's row stride (cols0 + cols1)
        differs from each source's. A dst sharing a source's base therefore
        overwrites source rows before they are read — the concat silently
        returns rows of the wrong data on both the simulator and the device.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[32, 16], pl.FP32],
                b: pl.Tensor[[32, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a: pl.Tile[[32, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(a, [0, 0], [32, 16])
                tile_b: pl.Tile[[32, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(b, [0, 0], [32, 16])
                tile_c: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.concat(tile_a, tile_b)
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_c, [0, 0], out)
                return result

        After = _run_pipeline(Before)
        bases = _collect_tile_memref_bases(After)

        assert bases["tile_c"] != bases["tile_a"], (
            "concat output must not reuse src0's buffer (tile.concat is not in-place safe)"
        )
        assert bases["tile_c"] != bases["tile_b"], (
            "concat output must not reuse src1's buffer (tile.concat is not in-place safe)"
        )

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

    def test_pipelined_kloop_accumulator_coalesces_to_one_acc_buffer(self):
        """A stage-2 pipelined K-loop matmul (as AutoTileMatmulL0 emits) whose
        L0C accumulator is large (176x176x4 = 121KB, fp32). After
        LowerPipelineLoops peels it into the multi-if-block shape, MemoryReuse
        must coalesce the whole accumulator chain (tile.create init + first-block
        matmul seed + the per-block matmul_acc + the if phis + the loop yield)
        onto ONE Acc allocation.

        Regression (fixed by TopDownRetargeter::CoalesceAccumulatorIfPhis): the
        peeled epilogue if-phi has a live in-place ``matmul_acc`` branch (on the
        accumulator buffer) and a dead ``if k==0`` fresh-``matmul`` seed branch on
        a different buffer. YieldFixupMutator used to reconcile them by copying the
        accumulator onto the seed buffer via an acc->acc ``tile.move`` -- a 2nd
        co-live 121KB L0C buffer that overflows the 128KB L0C, and an Acc->Acc tmov
        that ptoas rejects on every target. This reproduced the 512x512x192 bf16
        compile failure. The fix retargets the seed onto the accumulator buffer so
        both branches share it and no move is emitted (mad_acc's shared-%dst
        semantics). Runs the real ``lower_pipeline_loops`` so the peeled SSA shape
        matches production (hand-authored post-peel IR coalesces fine, so the real
        pass is required to trigger the gap). Runs with BASIC verification: the
        coalescing makes the peeled IR round-trip-clean, so the check is on legal
        IR, not just buffer count.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[176, 192], pl.BF16],
                rhs: pl.Tensor[[192, 176], pl.BF16],
                out: pl.Out[pl.Tensor[[176, 176], pl.FP32]],
            ) -> pl.Tensor[[176, 176], pl.FP32]:
                lhs_mat: pl.Tile[[176, 192], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [176, 192], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[192, 176], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [192, 176], target_memory=pl.Mem.Mat
                )
                c_init: pl.Tile[[176, 176], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [176, 176], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for ko, (c_iter,) in pl.pipeline(0, 192, 64, init_values=(c_init,), stage=2):
                    sa: pl.Tile[[176, 64], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, 0, ko, shape=[176, 64], target_memory=pl.Mem.Left
                    )
                    sb: pl.Tile[[64, 176], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko, 0, shape=[64, 176], target_memory=pl.Mem.Right
                    )
                    if ko == 0:
                        c_first: pl.Tile[[176, 176], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(sa, sb)
                        c_phi: pl.Tile[[176, 176], pl.FP32, pl.Mem.Acc] = pl.yield_(c_first)
                    else:
                        c_acc: pl.Tile[[176, 176], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(c_iter, sa, sb)
                        c_phi: pl.Tile[[176, 176], pl.FP32, pl.Mem.Acc] = pl.yield_(c_acc)
                    c: pl.Tile[[176, 176], pl.FP32, pl.Mem.Acc] = pl.yield_(c_phi)
                result: pl.Tensor[[176, 176], pl.FP32] = pl.store(c, [0, 0], out)
                return result

        # BASIC verification: the coalescing fix makes the peeled IR round-trip
        # clean, so this exercises the legality check, not just the buffer count.
        with passes.PassContext([], passes.VerificationLevel.BASIC):
            After = passes.memory_reuse()(
                passes.materialize_semantic_aliases()(
                    passes.init_mem_ref()(passes.lower_pipeline_loops()(Before))
                )
            )
        # The whole accumulator chain must coalesce onto ONE Acc allocation. A
        # phantom acc->acc tile.move (failed coalescing) leaves a 2nd Acc base.
        acc_bases = {b for b in _collect_tile_memref_bases(After).values() if "acc" in b}
        assert len(acc_bases) == 1, (
            f"expected ONE Acc allocation (accumulator coalesced), got {len(acc_bases)}: "
            f"{sorted(acc_bases)}\n{ir.python_print(After)}"
        )
        # Self-documenting: an in-place accumulator kernel needs no tile.move; a
        # surviving one here would be the illegal Acc->Acc copy.
        assert "tile.move" not in ir.python_print(After), (
            f"expected no tile.move in a coalesced accumulator chain:\n{ir.python_print(After)}"
        )

    def test_accumulator_if_phi_seed_retargets_to_accumulator_buffer(self):
        """Structural before/after for CoalesceAccumulatorIfPhis on a minimal
        accumulator if-phi (no loop/peel needed to reproduce).

        ``then`` is a fresh ``matmul`` seed on its own Acc buffer; ``else`` is an
        in-place ``matmul_acc`` on the accumulator buffer (aliasing ``prev``).
        Pre-fix, MemoryReuse reconciled them with a phantom Acc->Acc ``tile.move``
        onto a 2nd Acc buffer. The fix retargets the seed onto the accumulator
        buffer, so both branches and the phi share ONE Acc allocation and no move
        is emitted. Pinned structurally.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 64], pl.BF16],
                rhs: pl.Tensor[[64, 64], pl.BF16],
                cond: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                sa: pl.Tile[[16, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 64], target_memory=pl.Mem.Mat
                )
                sb: pl.Tile[[64, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [64, 64], target_memory=pl.Mem.Mat
                )
                prev: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(sa, sb)
                if cond < 1:
                    seed: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(sa, sb)
                    phi: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(seed)
                else:
                    acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(prev, sa, sb)
                    phi: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(acc)
                result: pl.Tensor[[16, 64], pl.FP32] = pl.store(phi, [0, 0], out)
                return result

        # Both branches' matmul/matmul_acc AND the phi land on mem_acc_5; the seed's
        # own buffer (mem_acc_6) is retargeted away and its alloc dropped; no tile.move.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 64], pl.BF16, pl.MemRef("mem_ddr_0", 0, 2048)],
                rhs: pl.Tensor[[64, 64], pl.BF16, pl.MemRef("mem_ddr_1", 0, 8192)],
                cond: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 2048)
                mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
                mem_acc_5: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
                sa: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_mat_3, 0, 2048), pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Mat
                )
                sb: pl.Tile[[64, 64], pl.BF16, pl.MemRef(mem_mat_4, 0, 8192), pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Mat
                )
                prev: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, 0, 4096), pl.Mem.Acc] = pl.tile.matmul(
                    sa, sb
                )
                if cond < 1:
                    seed: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, 0, 4096), pl.Mem.Acc] = (
                        pl.tile.matmul(sa, sb)
                    )
                    phi: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, 0, 4096), pl.Mem.Acc] = pl.yield_(
                        seed
                    )
                else:
                    acc: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, 0, 4096), pl.Mem.Acc] = (
                        pl.tile.matmul_acc(prev, sa, sb)
                    )
                    phi: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, 0, 4096), pl.Mem.Acc] = pl.yield_(
                        acc
                    )
                result: pl.Tensor[[16, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)] = pl.tile.store(
                    phi, [0, 0], out
                )
                return result

        After = _run_pipeline(Before)
        ir.assert_structural_equal(After, Expected)

    def test_gating_vec_inplace_if_phi_is_not_acc_coalesced(self):
        """Gating: the accumulator coalescer is Acc-scoped. A structurally
        identical if-phi in Vec space -- ``else`` is an in-place
        ``fillpad_inplace`` (reuses input, output aliases it), ``then`` a fresh
        seed on a different Vec buffer -- must NOT be coalesced by the seed
        retarget. ``IsInplaceAccumulatorProducer`` returns true for the Vec
        branch, but the ``MemorySpace::Acc`` gate skips it, so the existing
        ``YieldFixupMutator`` reconciles it with a legal Vec->Vec ``tile.move``
        (Vec->Vec is a legal tmov pair, unlike Acc->Acc). Two Vec buffers survive.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                cond: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                base: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                prev: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.fillpad_inplace(
                    base, pad_value=pl.PadValue.zero
                )
                if cond < 1:
                    # Fresh, non-in-place producer (tile.fillpad, not _inplace) so the
                    # seed is NOT an accumulator, while still carrying a tile_view to
                    # match the else branch (IfStmt requires consistent tile_view).
                    seed: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.fillpad(
                        base, pad_value=pl.PadValue.max
                    )
                    phi: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(seed)
                else:
                    acc: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.fillpad_inplace(
                        prev, pad_value=pl.PadValue.max
                    )
                    phi: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(acc)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(phi, [0, 0], out)
                return result

        After = _run_pipeline(Before)
        bases = {b for b in _collect_tile_memref_bases(After).values() if "vec" in b}
        # Not acc-coalesced: the two branch buffers survive and the existing path
        # inserts a (legal, Vec->Vec) tile.move to reconcile the phi.
        assert len(bases) == 2, f"Vec accumulator-shaped phi must not be acc-coalesced; bases={sorted(bases)}"
        assert "tile.move" in ir.python_print(After), (
            f"expected the pre-existing Vec->Vec move (coalescer must skip non-Acc):\n{ir.python_print(After)}"
        )

    def test_pre_if_acc_seed_not_coalesced_onto_accumulator(self):
        """Branch-locality guard: the accumulator coalescer must only retarget a
        seed that is defined *inside* the non-accumulator branch.

        Here ``then`` yields ``pre`` — a *pre-if* Acc value (computed before the
        ``if``, so it runs unconditionally) — and ``else`` accumulates in place
        into ``prev``. Coalescing would retarget ``pre`` onto ``prev``'s buffer,
        writing it before the ``if`` and clobbering the accumulator that the else
        branch reads. Branch exclusivity does NOT hold for a pre-if producer, so
        the coalescer must skip this phi (leaving it to YieldFixup) and keep
        ``pre`` and ``prev`` on distinct buffers.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 64], pl.BF16],
                rhs: pl.Tensor[[64, 64], pl.BF16],
                cond: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                sa: pl.Tile[[16, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 64], target_memory=pl.Mem.Mat
                )
                sb: pl.Tile[[64, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [64, 64], target_memory=pl.Mem.Mat
                )
                prev: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(sa, sb)  # the accumulator
                pre: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(sa, sb)  # pre-if seed
                if cond < 1:
                    phi: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(pre)
                else:
                    acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(prev, sa, sb)
                    phi: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(acc)
                result: pl.Tensor[[16, 64], pl.FP32] = pl.store(phi, [0, 0], out)
                return result

        # Verification off: the un-coalesced divergent Acc phi is the documented
        # out-of-scope case (YieldFixup emits its usual move); we only assert the
        # safety property — the pre-if seed was NOT retargeted onto the accumulator.
        with passes.PassContext([], passes.VerificationLevel.NONE):
            After = _run_pipeline(Before)
        bases = _collect_tile_memref_bases(After)
        assert bases["pre"] != bases["prev"], (
            f"pre-if seed must not be coalesced onto the accumulator buffer (clobber): "
            f"pre={bases.get('pre')} prev={bases.get('prev')}\n{ir.python_print(After)}"
        )

    def test_seed_branch_write_only_clobber_blocks_acc_coalesce(self):
        """Safety gate for the accumulator-if-phi coalescer's branch-tail
        liveness scan: a *write* to the accumulator buffer after the seed (with
        no read) must block coalescing, not just a read.

        The seed branch computes the fresh ``seed`` (its own Acc buffer) and then
        a *write-only* op ``_clob`` that lands on the accumulator buffer
        (``mem_acc_5``) — a fresh matmul reading only Mat operands, so it never
        *reads* ``mem_acc_5``.  If the coalescer retargeted ``seed`` onto
        ``mem_acc_5`` (as the read-only scan would allow), ``_clob`` — sequenced
        after ``seed`` — would overwrite the yielded value before the phi is read.
        The scan must therefore reject a later write of the target base, not only
        a later read (``SubtreeReadsBase`` alone misses the write-only clobber
        because the read collector skips the LHS definition of each stmt).

        Authored in fully-lowered form (explicit allocs + MemRefs) and run through
        ``memory_reuse`` alone: the clobbering alias is a specific buffer layout
        that only surfaces after allocation, so it cannot be expressed through the
        high-level ``init_mem_ref`` path (which hands every tile a distinct base).
        Verification off: the declined phi is reconciled by YieldFixup's usual
        (legal, distinct-buffer) move — we assert only the safety property, that
        ``seed`` was NOT retargeted onto the accumulator buffer.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 64], pl.BF16, pl.MemRef("mem_ddr_0", 0, 2048)],
                rhs: pl.Tensor[[64, 64], pl.BF16, pl.MemRef("mem_ddr_1", 0, 8192)],
                cond: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 2048)
                mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
                mem_acc_5: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
                mem_acc_6: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
                sa: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_mat_3, 0, 2048), pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Mat
                )
                sb: pl.Tile[[64, 64], pl.BF16, pl.MemRef(mem_mat_4, 0, 8192), pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Mat
                )
                prev: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, 0, 4096), pl.Mem.Acc] = pl.tile.matmul(
                    sa, sb
                )
                if cond < 1:
                    seed: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_6, 0, 4096), pl.Mem.Acc] = (
                        pl.tile.matmul(sa, sb)
                    )
                    # Write-only clobber of the accumulator buffer mem_acc_5: a
                    # fresh matmul reading only Mat operands, so it never *reads*
                    # mem_acc_5. Sequenced after `seed`, before the yield. Left
                    # deliberately unread (``_`` prefix) — only its write to the
                    # accumulator base matters; a read here would already trip the
                    # read-only scan and mask the write-only gap this test guards.
                    _clob: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, 0, 4096), pl.Mem.Acc] = (
                        pl.tile.matmul(sa, sb)
                    )
                    phi: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_6, 0, 4096), pl.Mem.Acc] = pl.yield_(
                        seed
                    )
                else:
                    acc: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, 0, 4096), pl.Mem.Acc] = (
                        pl.tile.matmul_acc(prev, sa, sb)
                    )
                    phi: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, 0, 4096), pl.Mem.Acc] = pl.yield_(
                        acc
                    )
                result: pl.Tensor[[16, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)] = pl.tile.store(
                    phi, [0, 0], out
                )
                return result

        with passes.PassContext([], passes.VerificationLevel.NONE):
            After = passes.memory_reuse()(Before)
        bases = _collect_tile_memref_bases(After)
        assert bases["seed"] != bases["prev"], (
            f"seed must not be coalesced onto the accumulator buffer when a later write-only op "
            f"clobbers it in the branch tail: seed={bases.get('seed')} prev={bases.get('prev')}\n"
            f"{ir.python_print(After)}"
        )

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


class TestCapacityGatedReuse:
    """Capacity-gated reuse (now the unconditional default, #1475 L0b fix) keeps
    cross-stage pipeline operands in separate L0 buffers when the space can afford
    it — instead of the legacy ``is_l0_space`` exemption that merges them and
    serialises the matmuls.

    Asserted by the **buffer signature**, not by synchronisation count: the two
    cross-stage ``Right`` operands get distinct buffers (the depth-2 ping-pong that
    shows up downstream as the ``0 A 0 A`` address stream) whenever the space can
    afford it; when it cannot, the shed / force_legacy floor merges them (the
    fa_fused 8->1 collapse in miniature). The success metric is WAR distance /
    overlap, *never* sync-flag count (see the pipeline-stage guard in
    docs/en/dev/passes/29-memory_reuse.md). The operands are ``tile.move``
    results (not loads), so the legacy load-only guard never protected them either.
    """

    @staticmethod
    def _collect_bases(program: ir.Program, names: tuple[str, ...]) -> dict[str, ir.Var]:
        """MemRef ``base_`` of each named Right operand in the result IR."""
        func = program.get_function("kernel")
        assert func is not None
        bases: dict[str, ir.Var] = {}

        def visit(stmt: ir.Stmt) -> None:
            if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint in names:
                t = stmt.var.type
                assert isinstance(t, ir.TileType) and t.memref is not None
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
        missing = [n for n in names if n not in bases]
        assert not missing, f"operands {missing} not found in After IR: {list(bases)}"
        return bases

    @staticmethod
    def _two_stage_matmuls(
        a_shape: tuple[int, int] = (32, 32), b_shape: tuple[int, int] = (32, 32)
    ) -> ir.Program:
        """Two matmuls whose Right operands ``r0``/``r1`` are cross-stage pipeline
        clones (``pipeline_membership`` ``"0:0"`` vs ``"0:1"`` — same group, distinct
        stage) with disjoint lifetimes: the minimal fa_fused shape. ``a_shape`` is the
        Left ``[M, K]`` and ``b_shape`` the Right ``[K, N]``; the default ``[32, 32]``
        BF16 => 2 KB each fits L0b (64 KB) with room. Pass a larger ``b_shape`` to force
        the depth-2 overflow that pins the group to a single buffer."""
        a_m, a_k = a_shape
        b_k, b_n = b_shape

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a0: pl.Tensor[[a_m, a_k], pl.BF16],
                b0: pl.Tensor[[b_k, b_n], pl.BF16],
                a1: pl.Tensor[[a_m, a_k], pl.BF16],
                b1: pl.Tensor[[b_k, b_n], pl.BF16],
                out0: pl.Out[pl.Tensor[[a_m, b_n], pl.FP32]],
                out1: pl.Out[pl.Tensor[[a_m, b_n], pl.FP32]],
            ) -> pl.Tensor[[a_m, b_n], pl.FP32]:
                a0m: pl.Tile[[a_m, a_k], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a0, [0, 0], [a_m, a_k], target_memory=pl.Mem.Mat
                )
                b0m: pl.Tile[[b_k, b_n], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b0, [0, 0], [b_k, b_n], target_memory=pl.Mem.Mat
                )
                l0: pl.Tile[[a_m, a_k], pl.BF16, pl.Mem.Left] = pl.tile.move(a0m, target_memory=pl.Mem.Left)
                r0: pl.Tile[[b_k, b_n], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b0m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:0"}
                )
                m0: pl.Tile[[a_m, b_n], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(l0, r0)
                out0 = pl.store(m0, [0, 0], out0)
                a1m: pl.Tile[[a_m, a_k], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a1, [0, 0], [a_m, a_k], target_memory=pl.Mem.Mat
                )
                b1m: pl.Tile[[b_k, b_n], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b1, [0, 0], [b_k, b_n], target_memory=pl.Mem.Mat
                )
                l1: pl.Tile[[a_m, a_k], pl.BF16, pl.Mem.Left] = pl.tile.move(a1m, target_memory=pl.Mem.Left)
                r1: pl.Tile[[b_k, b_n], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b1m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:1"}
                )
                m1: pl.Tile[[a_m, b_n], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(l1, r1)
                out1 = pl.store(m1, [0, 0], out1)
                return out1

        return Before

    def test_separates_affordable_cross_stage_right_operands(self):
        """Affordable: L0b (64 KB) holds both 2 KB operands, so the gate keeps them
        in distinct buffers — the depth-2 ping-pong. This is the ->2 address
        signature (the ``0 A 0 A`` stream) capacity-gated reuse produces."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        Before = self._two_stage_matmuls()
        After = passes.memory_reuse()(passes.init_mem_ref()(Before))
        bases = self._collect_bases(After, ("r0", "r1"))
        assert bases["r0"] is not bases["r1"], (
            "capacity-gated reuse should keep affordable cross-stage Right operands in separate L0b buffers"
        )

    def test_merges_when_slot_too_large_to_double_buffer(self):
        """Slot too large to double-buffer: each 48 KB Right operand leaves
        room for only k = min(2, floor(64/48)) = 1 buffer in the 64 KB L0b, so the two
        stages share it (depth-1). Depth-2 would need 96 KB > 64 KB. This is the
        capacity-pinned projection behaviour, preventing the overflow that blind
        separation would cause."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        # [128, 192] BF16 Right operand => 48 KB, so depth-2 (96 KB) overflows the 64 KB L0b.
        Before = self._two_stage_matmuls(a_shape=(16, 128), b_shape=(128, 192))
        After = passes.memory_reuse()(passes.init_mem_ref()(Before))
        bases = self._collect_bases(After, ("r0", "r1"))
        assert bases["r0"] is bases["r1"], (
            "L0b that fits only one 48 KB buffer (k=1) must merge the two stages"
        )

    def test_emits_perf_hint_when_pipeline_depth_capacity_reduced(self, tmp_path):
        """Explicit ``pl.pipeline`` intent that cannot fit must NOT degrade silently. The same
        48 KB Right operand caps ``F_g`` to 1 in the 64 KB L0b while the programmer requested
        depth 2, so the pass emits a loud ``PH-MR-001`` perf hint naming requested-vs-achieved
        depth and the fix. Routed through the diagnostic channel and captured via a
        ``ReportInstrument``'s ``perf_hints.log`` — the serialization is no longer silent."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        Before = self._two_stage_matmuls(a_shape=(16, 128), b_shape=(128, 192))
        with passes.PassContext([passes.ReportInstrument(str(tmp_path))]):
            passes.memory_reuse()(passes.init_mem_ref()(Before))
        log = tmp_path / "perf_hints.log"
        assert log.exists(), "a capacity-reduced pipeline depth must emit a perf hint, not serialize silently"
        text = log.read_text()
        assert "PH-MR-001" in text, f"expected the capacity-gate perf hint PH-MR-001, got: {text!r}"
        assert "requested depth 2" in text and "only 1 of 2 buffers" in text, (
            f"the hint must name the requested vs achieved depth, got: {text!r}"
        )
        # This shed is slot-bound (48 KB operand can't be double-buffered in 64 KB), so the fix is the
        # exact byte threshold — not the space-pressure wording.
        assert "shrink the per-stage tile" in text, (
            f"an operand-too-large shed must give the byte-threshold fix, got: {text!r}"
        )

    def test_finds_max_affordable_double_buffer_depth(self):
        """Depth-aware: a 3-stage group whose full separation (3 x 32 = 96 KB)
        exceeds L0b (64 KB) is capped to the max-affordable double-buffering depth
        k = min(3, floor(64/32)) = 2 — NOT collapsed to depth-1. The stages ping-pong
        through 2 buffers (stage mod 2): r0 and r2 share, r1 is separate. This is the
        proper modulo-variable-expansion an all-or-nothing gate would miss."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a0: pl.Tensor[[16, 128], pl.BF16],
                b0: pl.Tensor[[128, 128], pl.BF16],
                a1: pl.Tensor[[16, 128], pl.BF16],
                b1: pl.Tensor[[128, 128], pl.BF16],
                a2: pl.Tensor[[16, 128], pl.BF16],
                b2: pl.Tensor[[128, 128], pl.BF16],
                out0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
                out1: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
                out2: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a0m: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a0, [0, 0], [16, 128], target_memory=pl.Mem.Mat
                )
                b0m: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b0, [0, 0], [128, 128], target_memory=pl.Mem.Mat
                )
                l0: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(a0m, target_memory=pl.Mem.Left)
                r0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b0m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:0"}
                )
                m0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(l0, r0)
                out0 = pl.store(m0, [0, 0], out0)
                a1m: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a1, [0, 0], [16, 128], target_memory=pl.Mem.Mat
                )
                b1m: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b1, [0, 0], [128, 128], target_memory=pl.Mem.Mat
                )
                l1: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(a1m, target_memory=pl.Mem.Left)
                r1: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b1m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:1"}
                )
                m1: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(l1, r1)
                out1 = pl.store(m1, [0, 0], out1)
                a2m: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a2, [0, 0], [16, 128], target_memory=pl.Mem.Mat
                )
                b2m: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b2, [0, 0], [128, 128], target_memory=pl.Mem.Mat
                )
                l2: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(a2m, target_memory=pl.Mem.Left)
                r2: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b2m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:2"}
                )
                m2: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(l2, r2)
                out2 = pl.store(m2, [0, 0], out2)
                return out2

        After = passes.memory_reuse()(passes.init_mem_ref()(Before))
        bases = self._collect_bases(After, ("r0", "r1", "r2"))
        distinct = {b.name_hint for b in bases.values()}
        assert len(distinct) == 2, (
            f"depth-aware gate must keep the max-affordable depth-2 (2 buffers), got {len(distinct)}: {distinct}"
        )
        assert bases["r0"] is bases["r2"], "stages 0 and 2 (0 mod 2 == 2 mod 2) must share a ping-pong buffer"
        assert bases["r0"] is not bases["r1"], "stages 0 and 1 must occupy different ping-pong buffers"

    def test_merges_same_stage_operands(self):
        """Within-stage coalescing (the other half of the §5 tie-break): two operands
        tagged the SAME (group, stage) map to the same ping-pong residue, so they merge
        — only *cross-stage* operands are kept apart."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a0: pl.Tensor[[32, 32], pl.BF16],
                b0: pl.Tensor[[32, 32], pl.BF16],
                a1: pl.Tensor[[32, 32], pl.BF16],
                b1: pl.Tensor[[32, 32], pl.BF16],
                out0: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
                out1: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                a0m: pl.Tile[[32, 32], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a0, [0, 0], [32, 32], target_memory=pl.Mem.Mat
                )
                b0m: pl.Tile[[32, 32], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b0, [0, 0], [32, 32], target_memory=pl.Mem.Mat
                )
                l0: pl.Tile[[32, 32], pl.BF16, pl.Mem.Left] = pl.tile.move(a0m, target_memory=pl.Mem.Left)
                r0: pl.Tile[[32, 32], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b0m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:0"}
                )
                m0: pl.Tile[[32, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(l0, r0)
                out0 = pl.store(m0, [0, 0], out0)
                a1m: pl.Tile[[32, 32], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a1, [0, 0], [32, 32], target_memory=pl.Mem.Mat
                )
                b1m: pl.Tile[[32, 32], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b1, [0, 0], [32, 32], target_memory=pl.Mem.Mat
                )
                l1: pl.Tile[[32, 32], pl.BF16, pl.Mem.Left] = pl.tile.move(a1m, target_memory=pl.Mem.Left)
                r1: pl.Tile[[32, 32], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b1m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:0"}
                )
                m1: pl.Tile[[32, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(l1, r1)
                out1 = pl.store(m1, [0, 0], out1)
                return out1

        After = passes.memory_reuse()(passes.init_mem_ref()(Before))
        bases = self._collect_bases(After, ("r0", "r1"))
        assert bases["r0"] is bases["r1"], "same-stage operands must share a buffer"

    def test_separates_sparse_stage_ids(self):
        """Sparse stage IDs: stages {0, 2} are two distinct stages, so with k=2 they
        must stay separate. The fix compares the dense stage *ordinal* (0, 1) mod k,
        not the raw stage value — raw `2 mod 2 == 0 mod 2` would wrongly merge them."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a0: pl.Tensor[[32, 32], pl.BF16],
                b0: pl.Tensor[[32, 32], pl.BF16],
                a1: pl.Tensor[[32, 32], pl.BF16],
                b1: pl.Tensor[[32, 32], pl.BF16],
                out0: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
                out1: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                a0m: pl.Tile[[32, 32], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a0, [0, 0], [32, 32], target_memory=pl.Mem.Mat
                )
                b0m: pl.Tile[[32, 32], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b0, [0, 0], [32, 32], target_memory=pl.Mem.Mat
                )
                l0: pl.Tile[[32, 32], pl.BF16, pl.Mem.Left] = pl.tile.move(a0m, target_memory=pl.Mem.Left)
                r0: pl.Tile[[32, 32], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b0m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:0"}
                )
                m0: pl.Tile[[32, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(l0, r0)
                out0 = pl.store(m0, [0, 0], out0)
                a1m: pl.Tile[[32, 32], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a1, [0, 0], [32, 32], target_memory=pl.Mem.Mat
                )
                b1m: pl.Tile[[32, 32], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b1, [0, 0], [32, 32], target_memory=pl.Mem.Mat
                )
                l1: pl.Tile[[32, 32], pl.BF16, pl.Mem.Left] = pl.tile.move(a1m, target_memory=pl.Mem.Left)
                r1: pl.Tile[[32, 32], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b1m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:2"}
                )
                m1: pl.Tile[[32, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(l1, r1)
                out1 = pl.store(m1, [0, 0], out1)
                return out1

        After = passes.memory_reuse()(passes.init_mem_ref()(Before))
        bases = self._collect_bases(After, ("r0", "r1"))
        assert bases["r0"] is not bases["r1"], (
            "sparse stages {0,2} are distinct; dense-ordinal mod k must keep them apart"
        )

    def test_sheds_depth_when_coresident_tile_would_overflow(self, tmp_path):
        """Whole-space footprint safety with the *exact* SpaceFootprint: two 32 KB
        pipeline operands fill L0b at depth 2 (64 KB) on their own; a **co-live**
        non-pipeline Right tile ``np0`` (defined before stage 0, used after stage 1)
        cannot reuse either pipeline buffer, so it adds real capacity and the space
        overflows. The gate sheds the pipeline group's depth (2 -> 1) so the two
        operands merge, and AllocateMemoryAddr completes without overflow.

        Note ``np0`` must be *co-live*: a disjoint-lifetime non-pipeline tile would
        reuse a pipeline buffer for free (the exact footprint sees this), so the
        operands would correctly stay separated — the old conservative Sum(size)
        estimate over-counted a disjoint tile as its own buffer."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a0: pl.Tensor[[16, 128], pl.BF16],
                b0: pl.Tensor[[128, 128], pl.BF16],
                a1: pl.Tensor[[16, 128], pl.BF16],
                b1: pl.Tensor[[128, 128], pl.BF16],
                a2: pl.Tensor[[16, 128], pl.BF16],
                b2: pl.Tensor[[128, 128], pl.BF16],
                out0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
                out1: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
                out2: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                # np0 sourced + defined at the top and consumed last, so it is live
                # across both pipeline stages — a genuine co-resident that cannot reuse.
                a2m: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a2, [0, 0], [16, 128], target_memory=pl.Mem.Mat
                )
                b2m: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b2, [0, 0], [128, 128], target_memory=pl.Mem.Mat
                )
                l2: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(a2m, target_memory=pl.Mem.Left)
                np0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b2m, target_memory=pl.Mem.Right
                )
                a0m: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a0, [0, 0], [16, 128], target_memory=pl.Mem.Mat
                )
                b0m: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b0, [0, 0], [128, 128], target_memory=pl.Mem.Mat
                )
                l0: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(a0m, target_memory=pl.Mem.Left)
                r0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b0m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:0"}
                )
                m0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(l0, r0)
                out0 = pl.store(m0, [0, 0], out0)
                a1m: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a1, [0, 0], [16, 128], target_memory=pl.Mem.Mat
                )
                b1m: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b1, [0, 0], [128, 128], target_memory=pl.Mem.Mat
                )
                l1: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(a1m, target_memory=pl.Mem.Left)
                r1: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b1m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:1"}
                )
                m1: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(l1, r1)
                out1 = pl.store(m1, [0, 0], out1)
                m2: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(l2, np0)
                out2 = pl.store(m2, [0, 0], out2)
                return out2

        with passes.PassContext([passes.ReportInstrument(str(tmp_path))]):
            after_reuse = passes.memory_reuse()(passes.init_mem_ref()(Before))
        bases = self._collect_bases(after_reuse, ("r0", "r1"))
        assert bases["r0"] is bases["r1"], "co-resident non-pipeline tile must force fallback to merge"
        # The merged allocation fits; AllocateMemoryAddr must complete without overflow.
        allocated = passes.allocate_memory_addr()(after_reuse)
        assert allocated.get_function("kernel") is not None
        # This shed is *space-pressure*, not slot-bound: each 32 KB operand fits depth 2 in the 64 KB L0b on
        # its own — the co-live np tile is what overflows. The hint must therefore blame the co-residents and
        # NOT hand out the misleading per-slot byte threshold (the tile already satisfies slot <= cap/depth).
        text = (tmp_path / "perf_hints.log").read_text()
        assert "PH-MR-001" in text, f"a co-live shed must still emit the capacity hint, got: {text!r}"
        assert "over-subscribe the space" in text, (
            f"a space-pressure shed must point at co-residents, got: {text!r}"
        )
        assert "shrink the per-stage tile" not in text, (
            f"the per-slot byte threshold is misleading when the operand fits alone, got: {text!r}"
        )

    def test_is_deterministic(self):
        """Same IR in => identical buffer assignment out (the direct depth-cap is
        order-free — no ratio-greedy merge ordering to diverge)."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        Before = self._two_stage_matmuls()
        first = self._collect_bases(passes.memory_reuse()(passes.init_mem_ref()(Before)), ("r0", "r1"))
        second = self._collect_bases(passes.memory_reuse()(passes.init_mem_ref()(Before)), ("r0", "r1"))
        assert (first["r0"] is first["r1"]) == (second["r0"] is second["r1"])
        assert {b.name_hint for b in first.values()} == {b.name_hint for b in second.values()}

    def test_composes_with_matmul_acc_carry(self):
        """Carry composition (#1352; see the loop-carry re-alignment in
        docs/en/dev/passes/29-memory_reuse.md): the gate only ever *adds* separation and
        excludes loop carries from the packer, so capacity-gated reuse must not disturb a
        matmul_acc accumulator chain. These operands carry no pipeline_membership tags,
        so they never trip the gated residue constraint and behave like legacy. The pass
        + allocation must complete cleanly. (Note: this exercises the *untagged* bypass,
        NOT the shed-loop `force_legacy` fallback branch, which needs a tagged,
        capacity-overflowing space to fire.)"""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

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
                init_acc: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(tile_a_l0a, tile_b_l0b)
                for _k, (acc,) in pl.range(0, 4, init_values=(init_acc,)):
                    acc_next: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Acc] = pl.matmul_acc(
                        acc, tile_a_l0a, tile_b_l0b
                    )
                    loop_out = pl.yield_(acc_next)
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(loop_out, [0, 0], output)
                return result

        after = passes.memory_reuse()(passes.init_mem_ref()(Before))
        allocated = passes.allocate_memory_addr()(after)
        assert allocated.get_function("main") is not None, (
            "capacity-gated reuse must compose cleanly with an untagged matmul_acc carry chain"
        )

    def test_real_pipeline_membership_tags_reach_the_gate(self):
        """End-to-end tag flow on a REAL same-core ``pl.pipeline``: AutoTileMatmulL0
        turns a full-K matmul into ``pl.pipeline(stage=2)``, LowerPipelineLoops stamps
        real ``pipeline_membership`` tags, and they must reach MemoryReuse — verified
        by running the actual Default pipeline (truncated at MemoryReuse), not by
        hand-stamping tags.

        Note: same-core stage clones are *co-live*, so they are already double-buffered
        (OFF and ON both keep 2). The fa 8->1 *collapse* is cross-core-skew-specific —
        SkewCrossCorePipeline leaves the operands disjoint in each core's local order,
        which is what the legacy gate over-merges — and is modeled by the hand-tagged
        tests above. This guards the real tag production/consumption flow and that the
        gate yields a valid <=L0b allocation on a real pipeline."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        def pm_upto(upto: str) -> PassManager:
            pm = PassManager.get_strategy(OptimizationStrategy.Default)
            idx = pm.pass_names.index(upto)
            pipe = passes.PassPipeline()
            for p in pm.passes[: idx + 1]:
                pipe.add_pass(p)
            pm._pipeline = pipe
            return pm

        # Real LowerPipelineLoops tags reach MemoryReuse's input (they are stripped in
        # MemoryReuse's own output, so assert on the pre-pass IR).
        before_reuse = ir.python_print(pm_upto("InitMemRef").run_passes(Prog))
        assert '"pipeline_membership"' in before_reuse, "LowerPipelineLoops tags must reach MemoryReuse"
        assert "Mem.Right" in before_reuse, "expected pipelined Right operands in the lowered IR"

        # The gate runs on the real tags and yields a valid ≤ L0b allocation (no overflow).
        after_reuse = pm_upto("MemoryReuse").run_passes(Prog)
        allocated = passes.allocate_memory_addr()(after_reuse)
        assert allocated.get_function("kernel") is not None

    def test_stage4_uses_balanced_modulo_coloring(self):
        """Stage-4 group, room for 2 buffers (F = min(4, 64/32) = 2): the surviving
        coloring must be the balanced modulo-2 residues {0,2},{1,3} — every adjacent
        clone pair in different buffers — not a def-order adjacent collapse. This is the
        adjacency guarantee at depth > 2 that a distance-blind shed would break; it is
        why mod F is kept rather than recovered from a scalar ShedScore."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                # One Left / one Mat load reused; the four stage clones r0..r3 are distinct
                # disjoint-lifetime moves (each its own buffer candidate), tagged 0:0..0:3.
                am: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a, [0, 0], [16, 128], target_memory=pl.Mem.Mat
                )
                bm: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b, [0, 0], [128, 128], target_memory=pl.Mem.Mat
                )
                lt: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(am, target_memory=pl.Mem.Left)
                r0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    bm, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:0"}
                )
                m0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, r0)
                out = pl.store(m0, [0, 0], out)
                r1: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    bm, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:1"}
                )
                m1: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, r1)
                out = pl.store(m1, [0, 0], out)
                r2: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    bm, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:2"}
                )
                m2: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, r2)
                out = pl.store(m2, [0, 0], out)
                r3: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    bm, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:3"}
                )
                m3: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, r3)
                out = pl.store(m3, [0, 0], out)
                return out

        # Empty-instruments context suppresses the autouse SSA verification: this kernel
        # intentionally reassigns `out` per stage (non-SSA input) to model the stage clones.
        with passes.PassContext([]):
            after = passes.memory_reuse()(passes.init_mem_ref()(Before))
        bases = self._collect_bases(after, ("r0", "r1", "r2", "r3"))
        assert len({b.name_hint for b in bases.values()}) == 2, "exactly 2 ping-pong buffers at F=2"
        assert bases["r0"] is bases["r2"], "residue 0 = stages {0,2} share a buffer"
        assert bases["r1"] is bases["r3"], "residue 1 = stages {1,3} share a buffer"
        assert bases["r0"] is not bases["r1"], "adjacent stages 0,1 must be in different buffers"
        allocated = passes.allocate_memory_addr()(after)
        assert allocated.get_function("kernel") is not None

    def test_sequential_groups_time_share_without_false_shed(self):
        """Two *sequential* pipeline groups in L0b — group A (16 KB slots, stages 0:0/0:1)
        then group B (24 KB slots, 1:0/1:1). The naive per-group sum is 2·16 + 2·24 =
        80 KB > 64 KB, but the groups are disjoint in time, so the exact SpaceFootprint
        lets them **cross-merge diagonally** (A's operands reuse B's freed buffers) and
        both keep depth 2 in just 2 physical buffers (48 KB). No false shed — this is the
        co-resident/whole-space accuracy that the old Sum(size) estimate lacked. (The shed
        loop itself is exercised by the co-resident test, where a co-live tile forces it.)"""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                la: pl.Tensor[[16, 128], pl.BF16],
                ba0: pl.Tensor[[128, 64], pl.BF16],
                ba1: pl.Tensor[[128, 64], pl.BF16],
                bb0: pl.Tensor[[128, 96], pl.BF16],
                bb1: pl.Tensor[[128, 96], pl.BF16],
                outa0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
                outa1: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
                outb0: pl.Out[pl.Tensor[[16, 96], pl.FP32]],
                outb1: pl.Out[pl.Tensor[[16, 96], pl.FP32]],
            ) -> pl.Tensor[[16, 96], pl.FP32]:
                lam: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    la, [0, 0], [16, 128], target_memory=pl.Mem.Mat
                )
                lt: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(lam, target_memory=pl.Mem.Left)
                # group A (16 KB Right slots), stages 0:0 / 0:1
                ba0m: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    ba0, [0, 0], [128, 64], target_memory=pl.Mem.Mat
                )
                ra0: pl.Tile[[128, 64], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    ba0m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:0"}
                )
                ma0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, ra0)
                outa0 = pl.store(ma0, [0, 0], outa0)
                ba1m: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    ba1, [0, 0], [128, 64], target_memory=pl.Mem.Mat
                )
                ra1: pl.Tile[[128, 64], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    ba1m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:1"}
                )
                ma1: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, ra1)
                outa1 = pl.store(ma1, [0, 0], outa1)
                # group B (24 KB Right slots), stages 1:0 / 1:1
                bb0m: pl.Tile[[128, 96], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    bb0, [0, 0], [128, 96], target_memory=pl.Mem.Mat
                )
                rb0: pl.Tile[[128, 96], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    bb0m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:0"}
                )
                mb0: pl.Tile[[16, 96], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, rb0)
                outb0 = pl.store(mb0, [0, 0], outb0)
                bb1m: pl.Tile[[128, 96], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    bb1, [0, 0], [128, 96], target_memory=pl.Mem.Mat
                )
                rb1: pl.Tile[[128, 96], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    bb1m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:1"}
                )
                mb1: pl.Tile[[16, 96], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, rb1)
                outb1 = pl.store(mb1, [0, 0], outb1)
                return outb1

        after = passes.memory_reuse()(passes.init_mem_ref()(Before))
        bases = self._collect_bases(after, ("ra0", "ra1", "rb0", "rb1"))
        assert bases["ra0"] is not bases["ra1"], "group A keeps depth 2 — its operands stay separate"
        assert bases["rb0"] is not bases["rb1"], "group B keeps depth 2 — its operands stay separate"
        assert len({b.name_hint for b in bases.values()}) == 2, "both groups fit in 2 buffers by time-sharing"
        allocated = passes.allocate_memory_addr()(after)
        assert allocated.get_function("kernel") is not None

    def test_reserved_region_reduces_available_capacity(self):
        """The fit check begins free allocation at reserved_start — the top of any
        system.reserve_buffer region — matching AllocateMemoryAddr. Two 48 KB Vec
        pipeline operands fit at depth 2 (96 KB) on their own, but a 128 KB reserved
        buffer leaves too little Vec room, so the gate sheds them to a shared buffer.
        Without the reserve they stay separate: the only difference is the
        reserved_start accounting. (reserve_buffer lives in Vec for InCore functions;
        L0 — the #1475 L0b target — has no reserve region, so it is unaffected.)"""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class WithReserve:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x0: pl.Tensor[[128, 96], pl.FP32],
                x1: pl.Tensor[[128, 96], pl.FP32],
                out0: pl.Out[pl.Tensor[[128, 96], pl.FP32]],
                out1: pl.Out[pl.Tensor[[128, 96], pl.FP32]],
            ) -> pl.Tensor[[128, 96], pl.FP32]:
                pl.reserve_buffer(name="scratch", size=131072)
                r0: pl.Tile[[128, 96], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x0, [0, 0], [128, 96], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"}
                )
                s0: pl.Tile[[128, 96], pl.FP32, pl.Mem.Vec] = pl.tile.add(r0, r0)
                out0 = pl.store(s0, [0, 0], out0)
                r1: pl.Tile[[128, 96], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x1, [0, 0], [128, 96], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"}
                )
                s1: pl.Tile[[128, 96], pl.FP32, pl.Mem.Vec] = pl.tile.add(r1, r1)
                out1 = pl.store(s1, [0, 0], out1)
                return out1

        @pl.program
        class NoReserve:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x0: pl.Tensor[[128, 96], pl.FP32],
                x1: pl.Tensor[[128, 96], pl.FP32],
                out0: pl.Out[pl.Tensor[[128, 96], pl.FP32]],
                out1: pl.Out[pl.Tensor[[128, 96], pl.FP32]],
            ) -> pl.Tensor[[128, 96], pl.FP32]:
                r0: pl.Tile[[128, 96], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x0, [0, 0], [128, 96], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"}
                )
                s0: pl.Tile[[128, 96], pl.FP32, pl.Mem.Vec] = pl.tile.add(r0, r0)
                out0 = pl.store(s0, [0, 0], out0)
                r1: pl.Tile[[128, 96], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x1, [0, 0], [128, 96], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"}
                )
                s1: pl.Tile[[128, 96], pl.FP32, pl.Mem.Vec] = pl.tile.add(r1, r1)
                out1 = pl.store(s1, [0, 0], out1)
                return out1

        no_r = self._collect_bases(passes.memory_reuse()(passes.init_mem_ref()(NoReserve)), ("r0", "r1"))
        with_r = self._collect_bases(passes.memory_reuse()(passes.init_mem_ref()(WithReserve)), ("r0", "r1"))
        assert no_r["r0"] is not no_r["r1"], "without the reserve, 96 KB of Vec operands fit at depth 2"
        assert with_r["r0"] is with_r["r1"], "the 128 KB reserve reduces free Vec room, forcing the merge"

    def test_multi_group_shed_prefers_largest_slot(self):
        """The hardcoded ``max_relief`` shed policy: when a space overflows and *two*
        pipeline groups can each give up double-buffering, shed the **larger-slot**
        group first (freeing the most bytes per level ⇒ fewest levels lost).

        Two sheddable groups alone cannot force this — being lifetime-disjoint within a
        group is precisely what lets them *diagonal cross-merge* and time-share, so they
        never overflow (see ``test_on_sequential_groups_time_share_without_false_shed``;
        this is provable, not incidental: a group sheddable enough to merge its own
        stages always has a stage disjoint from the other group's, so a diagonal merge
        exists). The selection only becomes reachable with a **co-live non-pipeline
        blocker** that adds fixed pressure on top of the time-shared groups.

        Here group A (24 KB slots, ``0:0``/``0:1``) and B (16 KB slots, ``1:0``/``1:1``)
        are sequential (so they diagonal-merge to 2×24 KB), and ``np0`` (24 KB) is live
        across both. Full: 24 (A/B shared) + 24 (A/B shared) + 24 (np0) = 72 KB > 64 KB
        L0b ⇒ shed. ``max_relief`` drops A (24 KB > 16 KB) to depth 1; the space then
        fits at exactly 64 KB with **B still double-buffered**. A min-relief / arrival
        policy would instead shed B first — which does *not* relieve enough (A's two
        24 KB buffers + np0 still overflow), so it would go on to shed A too and lose
        B's depth as well. So the assertion below discriminates ``max_relief`` from the
        alternatives, not merely "some group shed"."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                la: pl.Tensor[[16, 128], pl.BF16],
                bnp: pl.Tensor[[128, 96], pl.BF16],
                ba: pl.Tensor[[128, 96], pl.BF16],
                bb: pl.Tensor[[128, 64], pl.BF16],
                onp: pl.Out[pl.Tensor[[16, 96], pl.FP32]],
                oa: pl.Out[pl.Tensor[[16, 96], pl.FP32]],
                ob: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 96], pl.FP32]:
                lam: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    la, [0, 0], [16, 128], target_memory=pl.Mem.Mat
                )
                lt: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(lam, target_memory=pl.Mem.Left)
                # co-live blocker np0: defined at the top, consumed last — cannot reuse a pipeline buffer.
                bnpm: pl.Tile[[128, 96], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    bnp, [0, 0], [128, 96], target_memory=pl.Mem.Mat
                )
                np0: pl.Tile[[128, 96], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    bnpm, target_memory=pl.Mem.Right
                )
                # group A (24 KB Right slots), sequential stages 0:0 / 0:1 (both loads reuse `ba`)
                ba0m: pl.Tile[[128, 96], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    ba, [0, 0], [128, 96], target_memory=pl.Mem.Mat
                )
                ra0: pl.Tile[[128, 96], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    ba0m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:0"}
                )
                ma0: pl.Tile[[16, 96], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, ra0)
                oa = pl.store(ma0, [0, 0], oa)
                ba1m: pl.Tile[[128, 96], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    ba, [0, 0], [128, 96], target_memory=pl.Mem.Mat
                )
                ra1: pl.Tile[[128, 96], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    ba1m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:1"}
                )
                ma1: pl.Tile[[16, 96], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, ra1)
                oa = pl.store(ma1, [0, 0], oa)
                # group B (16 KB Right slots), sequential stages 1:0 / 1:1 (both loads reuse `bb`)
                bb0m: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    bb, [0, 0], [128, 64], target_memory=pl.Mem.Mat
                )
                rb0: pl.Tile[[128, 64], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    bb0m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:0"}
                )
                mb0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, rb0)
                ob = pl.store(mb0, [0, 0], ob)
                bb1m: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    bb, [0, 0], [128, 64], target_memory=pl.Mem.Mat
                )
                rb1: pl.Tile[[128, 64], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    bb1m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "1:1"}
                )
                mb1: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, rb1)
                ob = pl.store(mb1, [0, 0], ob)
                mnp: pl.Tile[[16, 96], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, np0)
                onp = pl.store(mnp, [0, 0], onp)
                return onp

        # Empty-instruments context suppresses the autouse SSA verification: this kernel
        # intentionally reassigns `oa`/`ob` per stage (non-SSA input) to model the stage clones.
        with passes.PassContext([]):
            after = passes.memory_reuse()(passes.init_mem_ref()(Before))
        bases = self._collect_bases(after, ("ra0", "ra1", "rb0", "rb1"))
        assert bases["ra0"] is bases["ra1"], "max_relief sheds the larger-slot group A to depth 1"
        assert bases["rb0"] is not bases["rb1"], "the smaller group B keeps its depth-2 double-buffering"
        # After the shed the space fits at exactly cap — AllocateMemoryAddr must complete.
        allocated = passes.allocate_memory_addr()(after)
        assert allocated.get_function("kernel") is not None

    def test_unknown_capacity_matches_legacy_not_merge_all(self):
        """Unknown capacity (`cap == 0` — here no backend configured) must fall through to the
        legacy predicate, **not** gate every group to F_g == 1. Gating to F_g == 1 would merge
        everything and silently drop the legacy non-L0 load-only separation (#1900's Mat/L1 fix),
        separating strictly *less* than legacy. With an unknown budget the capacity-gated path is a
        no-op equivalent to legacy, so "never worse than legacy" holds for separation, not only for
        overflow. Two cross-stage Vec (non-L0) *load* tiles must stay apart."""
        backend.reset_for_testing()  # deliberately NO backend → GetMemSize == 0 for every space

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x0: pl.Tensor[[128, 96], pl.FP32],
                x1: pl.Tensor[[128, 96], pl.FP32],
                o0: pl.Out[pl.Tensor[[128, 96], pl.FP32]],
                o1: pl.Out[pl.Tensor[[128, 96], pl.FP32]],
            ) -> pl.Tensor[[128, 96], pl.FP32]:
                r0: pl.Tile[[128, 96], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x0, [0, 0], [128, 96], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"}
                )
                s0: pl.Tile[[128, 96], pl.FP32, pl.Mem.Vec] = pl.tile.add(r0, r0)
                o0 = pl.store(s0, [0, 0], o0)
                r1: pl.Tile[[128, 96], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x1, [0, 0], [128, 96], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"}
                )
                s1: pl.Tile[[128, 96], pl.FP32, pl.Mem.Vec] = pl.tile.add(r1, r1)
                o1 = pl.store(s1, [0, 0], o1)
                return o1

        bases = self._collect_bases(passes.memory_reuse()(passes.init_mem_ref()(Before)), ("r0", "r1"))
        assert bases["r0"] is not bases["r1"], (
            "unknown capacity must NOT merge the two cross-stage Vec load tiles (legacy fallthrough)"
        )

    def test_fallback_repacks_legacy_on_genuine_overflow(self, capfd):
        """The ``force_legacy`` shed-loop floor (§8.4): when a tagged space genuinely cannot fit
        at *any* depth, the shed exhausts (every group at F_g == 1) and the packer re-runs the
        legacy predicate + logs a diagnostic. One pipeline group, 4 co-live 20 KB stages (all four
        defined before any use ⇒ lifetimes overlap): 4×20 = 80 KB > 64 KB L0b, and co-liveness
        keeps ``can_share`` false at every depth, so the shed can never reduce the footprint and
        lands in ``force_legacy``. This is a genuine overflow legacy would also hit, so we don't
        assert allocation succeeds — we assert the branch's contract: the fallback re-runs exactly
        the legacy packing, which keeps the 4 co-live operands in distinct buffers."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                la: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 80], pl.BF16],
                o0: pl.Out[pl.Tensor[[16, 80], pl.FP32]],
                o1: pl.Out[pl.Tensor[[16, 80], pl.FP32]],
                o2: pl.Out[pl.Tensor[[16, 80], pl.FP32]],
                o3: pl.Out[pl.Tensor[[16, 80], pl.FP32]],
            ) -> pl.Tensor[[16, 80], pl.FP32]:
                lam: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    la, [0, 0], [16, 128], target_memory=pl.Mem.Mat
                )
                lt: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(lam, target_memory=pl.Mem.Left)
                # all four stages of one group defined before any use → mutually co-live (20 KB each)
                b0m: pl.Tile[[128, 80], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b, [0, 0], [128, 80], target_memory=pl.Mem.Mat
                )
                r0: pl.Tile[[128, 80], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b0m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:0"}
                )
                b1m: pl.Tile[[128, 80], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b, [0, 0], [128, 80], target_memory=pl.Mem.Mat
                )
                r1: pl.Tile[[128, 80], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b1m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:1"}
                )
                b2m: pl.Tile[[128, 80], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b, [0, 0], [128, 80], target_memory=pl.Mem.Mat
                )
                r2: pl.Tile[[128, 80], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b2m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:2"}
                )
                b3m: pl.Tile[[128, 80], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    b, [0, 0], [128, 80], target_memory=pl.Mem.Mat
                )
                r3: pl.Tile[[128, 80], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    b3m, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:3"}
                )
                m0: pl.Tile[[16, 80], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, r0)
                o0 = pl.store(m0, [0, 0], o0)
                m1: pl.Tile[[16, 80], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, r1)
                o1 = pl.store(m1, [0, 0], o1)
                m2: pl.Tile[[16, 80], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, r2)
                o2 = pl.store(m2, [0, 0], o2)
                m3: pl.Tile[[16, 80], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lt, r3)
                o3 = pl.store(m3, [0, 0], o3)
                return o3

        names = ("r0", "r1", "r2", "r3")
        bases = self._collect_bases(passes.memory_reuse()(passes.init_mem_ref()(Before)), names)
        # co-live ⇒ the force_legacy floor reproduces the legacy packing, keeping all four separate.
        assert len({bases[n].name_hint for n in names}) == 4, (
            "force_legacy fallback must keep the 4 co-live operands in distinct buffers"
        )
        # The fallback is not silent: it emits a Warning through the unified diagnostic channel (stderr).
        err = capfd.readouterr().err
        assert "fell back to the legacy packing" in err, (
            f"force_legacy must warn through the diagnostic channel, got stderr: {err!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
