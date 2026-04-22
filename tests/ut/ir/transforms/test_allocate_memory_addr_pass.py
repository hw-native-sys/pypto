# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.backend import is_backend_configured, reset_for_testing


def test_allocate_memory_addr_simple():
    """Simple function: Vec tiles get 32-byte aligned addresses at offsets 0 and 16384."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], output)
            return result

    @pl.program
    class Expected:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
            output: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
            mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
            tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
            )
            tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 16384, 16384), pl.Mem.Vec] = pl.tile.add(
                tile_a, tile_a
            )
            result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                tile_b, [0, 0], output
            )
            return result

    After = passes.init_mem_ref()(Before)
    After = passes.allocate_memory_addr()(After)
    ir.assert_structural_equal(After, Expected)


def test_allocate_memory_addr_multiple_tiles():
    """Three tiles each get their own MemRef at 32-byte aligned offsets 0, 16384, 32768."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_b, tile_b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
            return result

    @pl.program
    class Expected:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
            output: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
            mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
            mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
            tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
            )
            tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 16384, 16384), pl.Mem.Vec] = pl.tile.add(
                tile_a, tile_a
            )
            tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 32768, 16384), pl.Mem.Vec] = pl.tile.add(
                tile_b, tile_b
            )
            result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                tile_c, [0, 0], output
            )
            return result

    After = passes.init_mem_ref()(Before)
    After = passes.allocate_memory_addr()(After)
    ir.assert_structural_equal(After, Expected)


def test_allocate_memory_addr_resolves_auto_reserve_buffer_before_tiles():
    """AUTO reserve_buffer should consume the low address range before tile allocation."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            _ = pl.reserve_buffer(name="c2v_slot_buffer", size=4096)
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], output)
            return result

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
            output: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
            mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
            _: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0)
            tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 4096, 16384), pl.Mem.Vec] = pl.tile.load(
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
            )
            tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 20480, 16384), pl.Mem.Vec] = pl.tile.add(
                tile_a, tile_a
            )
            result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                tile_b, [0, 0], output
            )
            return result

    After = passes.init_mem_ref()(Before)
    After = passes.allocate_memory_addr()(After)
    ir.assert_structural_equal(After, Expected)


def test_allocate_memory_addr_rejects_overlapping_reserve_buffer_ranges():
    """Explicit reserve_buffer bases must not overlap previously reserved ranges."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(self):
            _first_buf = pl.reserve_buffer(name="first_slot_buffer", size=4096)
            _overlap_buf = pl.reserve_buffer(name="overlap_slot_buffer", size=1024, base=2048)

    with pytest.raises(
        Exception, match=re.escape("AllocateMemoryAddr found overlapping reserve_buffer ranges")
    ):
        program = passes.init_mem_ref()(Before)
        passes.allocate_memory_addr()(program)


def test_allocate_memory_addr_reuses_right_buffer_when_moves_sink_to_consumer():
    """Right buffers should share one address window when matmul moves do not overlap."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            lhs: pl.Tensor[[4, 128], pl.BF16],
            rhs0: pl.Tensor[[128, 64], pl.BF16],
            rhs1: pl.Tensor[[128, 64], pl.BF16],
            out_0: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
        ) -> pl.Tensor[[4, 64], pl.FP32]:
            lhs_tile: pl.Tile[[4, 128], pl.BF16] = pl.load(
                lhs, [0, 0], [4, 128], target_memory=pl.MemorySpace.Mat
            )
            rhs0_tile: pl.Tile[[128, 64], pl.BF16] = pl.load(
                rhs0, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat
            )
            rhs1_tile: pl.Tile[[128, 64], pl.BF16] = pl.load(
                rhs1, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat
            )
            _acc0: pl.Tile[[4, 64], pl.FP32] = pl.matmul(lhs_tile, rhs0_tile)
            acc1: pl.Tile[[4, 64], pl.FP32] = pl.matmul(lhs_tile, rhs1_tile)
            result: pl.Tensor[[4, 64], pl.FP32] = pl.store(acc1, [0, 0], out_0)
            return result

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            lhs: pl.Tensor[[4, 128], pl.BF16, pl.MemRef("mem_ddr_0", 0, 1024)],
            rhs0: pl.Tensor[[128, 64], pl.BF16, pl.MemRef("mem_ddr_1", 0, 16384)],
            rhs1: pl.Tensor[[128, 64], pl.BF16, pl.MemRef("mem_ddr_2", 0, 16384)],
            out_0: pl.Out[pl.Tensor[[4, 64], pl.FP32, pl.MemRef("mem_ddr_3", 0, 1024)]],
        ) -> pl.Tensor[[4, 64], pl.FP32]:
            mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 1024)
            mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
            mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
            mem_left_7: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 1024)
            mem_right_8: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 16384)
            mem_acc_9: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 1024)
            lhs_tile: pl.Tile[[4, 128], pl.BF16, pl.MemRef(mem_mat_4, 0, 1024), pl.Mem.Mat] = pl.tile.load(
                lhs, [0, 0], [4, 128], [4, 128], target_memory=pl.Mem.Mat, transpose=False
            )
            rhs0_tile: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_mat_5, 1024, 16384), pl.Mem.Mat] = (
                pl.tile.load(rhs0, [0, 0], [128, 64], [128, 64], target_memory=pl.Mem.Mat, transpose=False)
            )
            rhs1_tile: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_mat_6, 17408, 16384), pl.Mem.Mat] = (
                pl.tile.load(rhs1, [0, 0], [128, 64], [128, 64], target_memory=pl.Mem.Mat, transpose=False)
            )
            lhs_tile_Left: pl.Tile[[4, 128], pl.BF16, pl.MemRef(mem_left_7, 0, 1024), pl.Mem.Left] = (
                pl.tile.move(lhs_tile, target_memory=pl.Mem.Left)
            )
            # Both rhs*_tile_Right share mem_right_8 at offset 0 (memory reuse).
            rhs0_tile_Right: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_right_8, 0, 16384), pl.Mem.Right] = (
                pl.tile.move(rhs0_tile, target_memory=pl.Mem.Right)
            )
            _acc0: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_acc_9, 0, 1024), pl.Mem.Acc] = pl.tile.matmul(
                lhs_tile_Left, rhs0_tile_Right
            )
            rhs1_tile_Right: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_right_8, 0, 16384), pl.Mem.Right] = (
                pl.tile.move(rhs1_tile, target_memory=pl.Mem.Right)
            )
            acc1: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_acc_9, 0, 1024), pl.Mem.Acc] = pl.tile.matmul(
                lhs_tile_Left, rhs1_tile_Right
            )
            result: pl.Tensor[[4, 64], pl.FP32, pl.MemRef("mem_ddr_3", 0, 1024)] = pl.tile.store(
                acc1, [0, 0], out_0
            )
            return result

    After = passes.infer_tile_memory_space()(Before)
    After = passes.init_mem_ref()(After)
    After = passes.memory_reuse()(After)
    After = passes.allocate_memory_addr()(After)
    ir.assert_structural_equal(After, Expected)


def test_allocate_memory_addr_empty_function():
    """Functions with no TileType variables: pass is a no-op."""

    @pl.program
    class Before:
        @pl.function
        def main(self, output: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
            return output

    @pl.program
    class Expected:
        @pl.function
        def main(self, output: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
            return output

    After = passes.allocate_memory_addr()(Before)
    ir.assert_structural_equal(After, Expected)


def test_allocate_memory_addr_allocs_are_prepended_to_body():
    """Alloc statements are prepended as direct children of the top-level SeqStmts before tile ops."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], output)
            return result

    @pl.program
    class Expected:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
            output: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            # Allocs are prepended before all tile ops.
            mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
            mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
            tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
            )
            tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 16384, 16384), pl.Mem.Vec] = pl.tile.add(
                tile_a, tile_a
            )
            result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                tile_b, [0, 0], output
            )
            return result

    After = passes.init_mem_ref()(Before)
    After = passes.allocate_memory_addr()(After)
    ir.assert_structural_equal(After, Expected)


def test_allocate_memory_addr_raw_pointer_uniqueness():
    """Each unique MemRef gets its own alloc with distinct addresses (no reuse)."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_b, tile_b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
            return result

    @pl.program
    class Expected:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
            output: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            # Three distinct allocs for three distinct MemRefs, at three distinct offsets.
            mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
            mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
            mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
            tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
            )
            tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 16384, 16384), pl.Mem.Vec] = pl.tile.add(
                tile_a, tile_a
            )
            tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 32768, 16384), pl.Mem.Vec] = pl.tile.add(
                tile_b, tile_b
            )
            result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                tile_c, [0, 0], output
            )
            return result

    After = passes.init_mem_ref()(Before)
    After = passes.allocate_memory_addr()(After)
    ir.assert_structural_equal(After, Expected)


def test_allocated_memory_addr_verifier_passes_after_add_alloc():
    """After init_mem_ref + allocate_memory_addr, non-DDR memrefs have valid (non-negative) addresses."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], output)
            return result

    @pl.program
    class Expected:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
            output: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            # Non-DDR memrefs are allocated at non-negative byte offsets (0, 16384).
            mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
            mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
            tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
            )
            tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 16384, 16384), pl.Mem.Vec] = pl.tile.add(
                tile_a, tile_a
            )
            result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                tile_b, [0, 0], output
            )
            return result

    After = passes.init_mem_ref()(Before)
    After = passes.allocate_memory_addr()(After)
    ir.assert_structural_equal(After, Expected)


def test_memrefs_before_allocate_have_unallocated_addr():
    """Before AllocateMemoryAddr (only init_mem_ref), MemRef byte_offsets are 0 (uninitialized).

    This is a precondition check on init_mem_ref — not a test of allocate_memory_addr.
    It's kept here (rather than in test_init_memref.py) to document the contract this
    pass depends on. Kept in non-declarative form because it asserts a specific field
    value after a different pass.
    """

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], output)
            return result

    program = passes.init_mem_ref()(Before)
    func = next(iter(program.functions.values()))

    memref_addrs = {}
    assert isinstance(func.body, ir.SeqStmts)
    for stmt in func.body.stmts:
        if isinstance(stmt, ir.AssignStmt):
            var_type = stmt.var.type
            if isinstance(var_type, ir.TileType) and var_type.memref is not None:
                memref = var_type.memref
                if isinstance(memref.byte_offset_, ir.ConstInt):
                    memref_addrs[stmt.var.name_hint] = memref.byte_offset_.value

    assert len(memref_addrs) > 0, "Should have MemRef addresses after init_mem_ref"
    for var_name, addr in memref_addrs.items():
        assert addr == 0, (
            f"MemRef byte_offset for '{var_name}' should be 0 before AllocateMemoryAddr, got {addr}"
        )


def test_allocated_memory_addr_verifier_via_pipeline():
    """Test that the AllocatedMemoryAddr property is verified through the PassPipeline.

    Uses VerificationInstrument in AFTER mode to confirm that add_alloc
    correctly produces the AllocatedMemoryAddr property.
    """

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], output)
            return result

    pipeline = passes.PassPipeline()
    pipeline.add_pass(passes.init_mem_ref())
    pipeline.add_pass(passes.allocate_memory_addr())

    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
        result = pipeline.run(Before)
        assert result is not None


def test_allocate_memory_addr_uses_default_policy_without_backend():
    """Test that AllocateMemoryAddr falls back to DefaultMemoryAllocatorPolicy when no backend is configured.

    Without a backend, the pass should still produce correct 32-byte aligned
    addresses using the default policy (skip DDR, sort by id, 32-byte alignment).
    """
    was_configured = is_backend_configured()
    if was_configured:
        reset_for_testing()
    try:
        assert not is_backend_configured(), "Backend must not be configured for this test"

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
                tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_b, tile_b)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                output: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a,
                    [0, 0],
                    [64, 64],
                    [64, 64],
                    target_memory=pl.Mem.Vec,
                    transpose=False,
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 16384, 16384), pl.Mem.Vec] = (
                    pl.tile.add(tile_a, tile_a)
                )
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 32768, 16384), pl.Mem.Vec] = (
                    pl.tile.add(tile_b, tile_b)
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_c, [0, 0], output
                )
                return result

        After = passes.init_mem_ref()(Before)
        After = passes.allocate_memory_addr()(After)
        ir.assert_structural_equal(After, Expected)
    finally:
        if was_configured:
            reset_for_testing()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
