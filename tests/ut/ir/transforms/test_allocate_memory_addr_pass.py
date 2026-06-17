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
from pypto.backend import (
    BackendType,
    get_backend_type,
    is_backend_configured,
    reset_for_testing,
    set_backend_type,
)


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


def test_allocated_memory_addr_verifier_errors_when_vec_exceeds_safe_cap():
    """A Vec footprint above the safe UB cap must be rejected (pto-isa#170).

    The 910B Vec UB physical size is 192KB, but only ~184KB is usable: PTO-ISA
    reserves the top ~8KB and silently corrupts any tile placed there (pto-isa#170).
    soc.cpp therefore caps the *safe* Vec UB at 184KB (188416 bytes), so the
    AllocatedMemoryAddr verifier (which compares the Vec high-water against
    backend.get_mem_size(Vec)) raises when usage exceeds the safe cap.

    A single 64x752 FP32 tile is 192512 bytes: above the 184KB safe cap but below
    the 192KB physical size. Under the old 192KB limit it passed; it must now error.
    Regression guard so the cap is not silently raised back to 192KB without also
    restoring the physical size in soc.cpp.
    """
    # 64 * 752 * 4 = 192512 bytes; 184KB = 188416, 192KB = 196608.
    assert 184 * 1024 < 64 * 752 * 4 < 192 * 1024

    was_configured = is_backend_configured()
    prior_type = get_backend_type() if was_configured else None
    if was_configured:
        reset_for_testing()
    try:
        set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 752], pl.FP32],
                output: pl.Tensor[[64, 752], pl.FP32],
            ) -> pl.Tensor[[64, 752], pl.FP32]:
                # One live Vec tile of 192512 bytes -> Vec high-water exceeds the
                # 184KB safe cap (but not the 192KB physical size).
                tile_a: pl.Tile[[64, 752], pl.FP32] = pl.load(input_a, [0, 0], [64, 752])
                result: pl.Tensor[[64, 752], pl.FP32] = pl.store(tile_a, [0, 0], output)
                return result

        program = passes.init_mem_ref()(Before)
        pipeline = passes.PassPipeline()
        pipeline.add_pass(passes.allocate_memory_addr())
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
            with pytest.raises(ValueError, match=r"Vec buffer usage .* exceeds platform limit"):
                pipeline.run(program)
    finally:
        reset_for_testing()
        if prior_type is not None:
            set_backend_type(prior_type)


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


def test_allocate_memory_addr_preserves_sibling_slice_offsets():
    """Sibling slice/reshape views keep distinct per-view addresses (base + slice offset).

    Regression for issue #1510. All views share one ``base_`` Ptr (the root
    ``mem_vec_3`` alloc), so they form a single base_ group co-located in one slot
    (pass src lines 300-307). The slot base is ``current_addr = 0`` (no reserve),
    and each member keeps its own relative offset: ``new_addr = slot_base + member
    offset`` (pass src lines 339-365, doc line 58-60). InitMemRef already recorded
    each view's relative offset (root/row-0 at 0, row-1 at 1*16*4 = 64 bytes), so:

      tile_a, s0, c0 -> 0   (root and row-0 views sit at the slot base)
      s1, c1         -> 64  (row-1 views carry the slice offset; must NOT collapse
                             onto base 0 — the #1510 bug a reshape-of-slice chain
                             would otherwise alias to row 0)

    Because slot_base is 0, the post-pass offsets equal the pre-pass relative
    offsets: the pass is a no-op on offsets here, which is exactly the invariant.
    """

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[8, 16], pl.FP32],
            out0: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            out1: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            tile_a: pl.Tile[[8, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [8, 16])
            # Two sibling slices at rows 0 and 1, each reshaped to a column vector.
            s0: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.slice(tile_a, [1, 16], [0, 0])
            s1: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.slice(tile_a, [1, 16], [1, 0])
            c0: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(s0, [16, 1])
            c1: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(s1, [16, 1])
            r0: pl.Tensor[[16, 1], pl.FP32] = pl.store(c0, [0, 0], out0)
            _r1: pl.Tensor[[16, 1], pl.FP32] = pl.store(c1, [0, 0], out1)
            return r0

    @pl.program
    class Expected:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[8, 16], pl.FP32, pl.MemRef("mem_ddr_0", 0, 512)],
            out0: pl.Out[pl.Tensor[[16, 1], pl.FP32, pl.MemRef("mem_ddr_1", 0, 64)]],
            out1: pl.Out[pl.Tensor[[16, 1], pl.FP32, pl.MemRef("mem_ddr_2", 0, 64)]],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
            # Root sits at the slot base 0, sized to the full 512-byte alloc.
            tile_a: pl.Tile[[8, 16], pl.FP32, pl.MemRef(mem_vec_3, 0, 512), pl.Mem.Vec] = pl.tile.load(
                input_a, [0, 0], [8, 16], [8, 16], target_memory=pl.Mem.Vec, transpose=False
            )
            # Row-0 slice + its reshape land on the slot base (offset 0).
            s0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_3, 0, 64), pl.Mem.Vec] = pl.tile.slice(
                tile_a, [1, 16], [0, 0]
            )
            # Row-1 slice + its reshape carry the +64 byte slice offset.
            s1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_3, 64, 64), pl.Mem.Vec] = pl.tile.slice(
                tile_a, [1, 16], [1, 0]
            )
            c0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_3, 0, 64), pl.Mem.Vec] = pl.tile.reshape(
                s0, [16, 1]
            )
            c1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_3, 64, 64), pl.Mem.Vec] = pl.tile.reshape(
                s1, [16, 1]
            )
            r0: pl.Tensor[[16, 1], pl.FP32, pl.MemRef("mem_ddr_1", 0, 64)] = pl.tile.store(c0, [0, 0], out0)
            _r1: pl.Tensor[[16, 1], pl.FP32, pl.MemRef("mem_ddr_2", 0, 64)] = pl.tile.store(c1, [0, 0], out1)
            return r0

    After = passes.init_mem_ref()(Before)
    After = passes.allocate_memory_addr()(After)
    ir.assert_structural_equal(After, Expected)


def test_allocate_memory_addr_resolves_aic_reserve_buffer_in_mat_space():
    """AIC reserve_buffer reserves the Mat space (not Vec) before Mat tile allocation.

    GetReserveBufferMemorySpace maps AIC -> MemorySpace::Mat (pass src lines 62-64),
    whereas AIV/InCore map to Vec. So an AUTO reserve_buffer of 4096 bytes consumes
    the low Mat window: reserved_end[Mat] = align32(0 + 4096) = 4096 (pass src
    lines 133-155). The Mat tile then starts at current_addr = reserved_end = 4096
    (pass src lines 309-313), and the buffer's base kwarg is rewritten 0 (lines
    230-247). This is the Mat-space dual of the AIV/Vec reserve test above.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIC)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.BF16],
            out_0: pl.Out[pl.Tensor[[64, 64], pl.BF16]],
        ) -> pl.Tensor[[64, 64], pl.BF16]:
            _ = pl.reserve_buffer(name="aic_slot_buffer", size=4096)
            tile_a: pl.Tile[[64, 64], pl.BF16] = pl.load(
                input_a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat
            )
            result: pl.Tensor[[64, 64], pl.BF16] = pl.store(tile_a, [0, 0], out_0)
            return result

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.BF16, pl.MemRef("mem_ddr_0", 0, 8192)],
            out_0: pl.Out[pl.Tensor[[64, 64], pl.BF16, pl.MemRef("mem_ddr_1", 0, 8192)]],
        ) -> pl.Tensor[[64, 64], pl.BF16]:
            mem_mat_2: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
            _: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name="aic_slot_buffer", size=4096, base=0)
            # Mat tile is pushed past the 4096-byte reserved Mat window.
            tile_a: pl.Tile[[64, 64], pl.BF16, pl.MemRef(mem_mat_2, 4096, 8192), pl.Mem.Mat] = pl.tile.load(
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Mat, transpose=False
            )
            result: pl.Tensor[[64, 64], pl.BF16, pl.MemRef("mem_ddr_1", 0, 8192)] = pl.tile.store(
                tile_a, [0, 0], out_0
            )
            return result

    After = passes.init_mem_ref()(Before)
    After = passes.allocate_memory_addr()(After)
    ir.assert_structural_equal(After, Expected)


def test_allocate_memory_addr_honors_explicit_reserve_buffer_base():
    """An explicit reserve_buffer base= is honored verbatim and bounds tile placement.

    With base provided (>= 0), resolved_base = base (pass src lines 122-123) — the
    pass does NOT fill the [0, base) gap below it. The reserved end is
    align32(base + size); tiles start there (pass src lines 309-313). Here
    base=8192, size=4096 -> reserved_end[Vec] = align32(12288) = 12288, so:

      tile_a -> 12288, tile_b -> align32(12288 + 16384) = 28672

    The base kwarg is left as the supplied 8192 (pass src lines 234-242).
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            _ = pl.reserve_buffer(name="explicit_slot_buffer", size=4096, base=8192)
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
            _: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(
                name="explicit_slot_buffer", size=4096, base=8192
            )
            # Reserved window is [8192, 12288); the [0, 8192) gap below base stays unused.
            tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 12288, 16384), pl.Mem.Vec] = pl.tile.load(
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
            )
            tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 28672, 16384), pl.Mem.Vec] = pl.tile.add(
                tile_a, tile_a
            )
            result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                tile_b, [0, 0], output
            )
            return result

    After = passes.init_mem_ref()(Before)
    After = passes.allocate_memory_addr()(After)
    ir.assert_structural_equal(After, Expected)


def test_allocate_memory_addr_skips_non_incore_function():
    """Non-InCore functions (Spmd/Group/Orchestration/Opaque) are returned unchanged.

    TransformAllocateMemoryAddr early-returns for any function where
    !IsInCoreType(func_type_) (pass src lines 396-398) — only InCore/AIC/AIV use
    on-chip tile buffers. An Opaque function whose tile allocs still sit at the
    InitMemRef placeholder offset (0, unallocated) must therefore come out
    byte-for-byte identical: the pass does NOT bump them to 0 / 16384 the way it
    would for an InCore function (cf. test_allocate_memory_addr_simple).

    Expected is the InitMemRef output itself: asserting the pass is a no-op
    directly encodes the skip semantics without hand-snapshotting InitMemRef.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], output)
            return result

    Initialized = passes.init_mem_ref()(Before)
    After = passes.allocate_memory_addr()(Initialized)
    # Pass is a no-op for the Opaque (non-InCore) function: addresses untouched.
    ir.assert_structural_equal(After, Initialized)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
