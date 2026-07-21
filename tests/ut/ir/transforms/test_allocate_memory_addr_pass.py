# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
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

requires_dsa = pytest.mark.skipif(
    not passes.is_dsa_solver_available(), reason="PyPTO was built without PYPTO_ENABLE_DSA_SOLVER"
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
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
            )
            tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 20480, 16384), pl.Mem.Vec] = pl.tile.add(
                tile_a, tile_a
            )
            result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                tile_b, [0, 0], output
            )
            return result

    Initialized = passes.init_mem_ref()(Before)
    After = passes.allocate_memory_addr()(Initialized)
    ir.assert_structural_equal(After, Expected)
    if passes.is_dsa_solver_available():
        DsaAfter = _allocate_with_dsa(Initialized)
        assert _vec_peak(DsaAfter) == 4096 + 16384


def test_allocate_memory_addr_rejects_overlapping_reserve_buffer_ranges():
    """Explicit reserve_buffer bases must not overlap previously reserved ranges."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(self):
            _first_buf = pl.reserve_buffer(name="first_slot_buffer", size=4096)
            _overlap_buf = pl.reserve_buffer(name="overlap_slot_buffer", size=1024, base=2048)

    with pytest.raises(
        # Message is now emitted by the shared reserve_buffer_utils resolver (used by both
        # AllocateMemoryAddr and MemoryReuse), so match the pass-agnostic substring.
        Exception,
        match=re.escape("overlapping reserve_buffer ranges"),
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
                lhs, [0, 0], [4, 128], [4, 128], target_memory=pl.Mem.Mat
            )
            rhs0_tile: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_mat_5, 1024, 16384), pl.Mem.Mat] = (
                pl.tile.load(rhs0, [0, 0], [128, 64], [128, 64], target_memory=pl.Mem.Mat)
            )
            rhs1_tile: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_mat_6, 17408, 16384), pl.Mem.Mat] = (
                pl.tile.load(rhs1, [0, 0], [128, 64], [128, 64], target_memory=pl.Mem.Mat)
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
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
            @pl.function(type=pl.FunctionType.AIV)
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
        if passes.is_dsa_solver_available():
            with pytest.raises(ValueError, match=r"standalone DSA solver could not fit"):
                _allocate_with_dsa(program)

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
                input_a, [0, 0], [8, 16], [8, 16], target_memory=pl.Mem.Vec
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


@requires_dsa
def test_dsa_writeback_preserves_relative_view_offsets(tmp_path):
    """A standalone placement moves a base without collapsing its views."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[8, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            tile_a: pl.Tile[[8, 16], pl.FP32] = pl.load(input_a, [0, 0], [8, 16])
            row_1: pl.Tile[[1, 16], pl.FP32] = pl.tile.slice(tile_a, [1, 16], [1, 0])
            column: pl.Tile[[16, 1], pl.FP32] = pl.tile.reshape(row_1, [16, 1])
            result = pl.store(column, [0, 0], out)
            return result

    initialized = passes.init_mem_ref()(Before)
    legacy = passes.allocate_memory_addr()(initialized)
    planned = _allocate_with_dsa(initialized, str(tmp_path))
    ir.assert_structural_equal(planned, legacy)

    document = json.loads((tmp_path / "pypto_main.dsa.json").read_text())
    assert document["problem"]["pypto_structure"]["alias_classes"] == [
        {"buffer": 0, "members": ["tile_a", "row_1", "column"]}
    ]


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
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Mat
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
                input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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


def _vec_peak(func) -> int:
    """Max (offset + size) over Vec-space MemRefs in a function's printed IR."""
    text = ir.python_print(func)
    peak = 0
    for off, size in re.findall(
        r"MemRef\([^,]+,\s*pl\.const\((\d+),[^)]*\),\s*(\d+)\),\s*pl\.Mem\.Vec", text
    ):
        peak = max(peak, int(off) + int(size))
    return peak


def _dsa_chain_program():
    """InCore (AIV) kernel with a chain a->b->c; tile_a[def..b] and tile_c are
    lifetime-disjoint, so tile_c can reuse tile_a's slot. AIV is required — a
    plain @pl.function is non-InCore and AllocateMemoryAddr no-ops on it."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def read_before_write_chain(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_b, tile_b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
            return result

    return passes.init_mem_ref()(Before)


def _allocate_with_dsa(
    base,
    export_dir: str | None = None,
    solution_dir: str | None = None,
    reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.DISABLED,
    reference_placement=passes.DsaReferencePlacement.DEFAULT,
    reference_target: str | None = None,
):
    """Run the standalone planner through its PassContext-owned adapter."""
    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.DSA,
        dsa_export_dir=export_dir,
        dsa_solution_dir=solution_dir,
        dsa_reuse_penalty_recognizer=reuse_penalty_recognizer,
        dsa_reference_placement=reference_placement,
        dsa_reference_target=reference_target,
    ):
        return passes.allocate_memory_addr()(base)


@requires_dsa
def test_dsa_planner_reuses_at_read_before_write_boundary():
    """The standalone planner jointly reuses and places unmerged buffers."""
    base = _dsa_chain_program()
    bump = passes.allocate_memory_addr()(base)
    planned = _allocate_with_dsa(base)

    bump_peak = _vec_peak(bump)
    plan_peak = _vec_peak(planned)
    assert bump_peak == 3 * 16384  # bump: three distinct 16 KB slots
    # Every producer's last read precedes its consumer's write at the same
    # statement point, so all three buffers may use one physical slot.
    assert plan_peak == 16384


@requires_dsa
def test_dsa_loose_reference_spreads_only_the_selected_function(tmp_path):
    """Compact and loose endpoints are derived and validated in one compile.

    An unmatched exact target must retain the compact baseline so multi-kernel
    programs can vary one DSA instance without perturbing siblings.
    """
    base = _dsa_chain_program()
    compact_dir = tmp_path / "compact"
    loose_dir = tmp_path / "loose"
    unmatched_dir = tmp_path / "unmatched"

    compact = _allocate_with_dsa(
        base,
        str(compact_dir),
        reference_placement=passes.DsaReferencePlacement.COMPACT,
    )
    loose = _allocate_with_dsa(
        base,
        str(loose_dir),
        reference_placement=passes.DsaReferencePlacement.LOOSE,
        reference_target="read_before_write_chain",
    )
    unmatched = _allocate_with_dsa(
        base,
        str(unmatched_dir),
        reference_placement=passes.DsaReferencePlacement.LOOSE,
        reference_target="some_other_kernel",
    )

    assert _vec_peak(compact) == 16384
    assert _vec_peak(loose) == 3 * 16384
    assert _vec_peak(unmatched) == 16384

    compact_problem = compact_dir / "pypto_read_before_write_chain.dsa.json"
    loose_problem = loose_dir / "pypto_read_before_write_chain.dsa.json"
    assert compact_problem.read_text() == loose_problem.read_text()

    compact_solution = json.loads(
        (compact_dir / "pypto_read_before_write_chain.dsa.solution.json").read_text()
    )
    loose_solution = json.loads((loose_dir / "pypto_read_before_write_chain.dsa.solution.json").read_text())
    unmatched_solution = json.loads(
        (unmatched_dir / "pypto_read_before_write_chain.dsa.solution.json").read_text()
    )
    assert compact_solution["problem_fingerprint"] == loose_solution["problem_fingerprint"]
    assert compact_solution["metadata"]["reference_placement"] == "compact"
    assert loose_solution["metadata"]["reference_placement"] == "loose"
    assert unmatched_solution["metadata"]["reference_placement"] == "compact"


@requires_dsa
def test_dsa_export_is_deterministic_pypto_hard_v1(tmp_path):
    """A real IR function exports a stable schema-v1 benchmark document."""
    base = _dsa_chain_program()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _allocate_with_dsa(base, str(first_dir))
    _allocate_with_dsa(base, str(second_dir))

    first = first_dir / "pypto_read_before_write_chain.dsa.json"
    second = second_dir / "pypto_read_before_write_chain.dsa.json"
    assert first.read_text() == second.read_text()
    first_solution = first_dir / "pypto_read_before_write_chain.dsa.solution.json"
    second_solution = second_dir / "pypto_read_before_write_chain.dsa.solution.json"
    assert first_solution.read_text() == second_solution.read_text()

    document = json.loads(first.read_text())
    assert document["schema_version"] == 1
    assert document["profile"] == "pypto_hard_v1"
    assert document["instance"] == "read_before_write_chain"
    assert document["metadata"]["solver_input"] == "pre_memory_reuse"
    buffers = document["problem"]["buffers"]
    assert len(buffers) == 3
    assert [buffer["id"] for buffer in buffers] == [0, 1, 2]
    assert [buffer["size"] for buffer in buffers] == [16384, 16384, 16384]
    assert [buffer["live_intervals"] for buffer in buffers] == [
        [{"lower": 7, "upper": 9}],
        [{"lower": 9, "upper": 11}],
        [{"lower": 11, "upper": 13}],
    ]
    assert document["problem"]["constraints"] == {
        "colocations": [],
        "pinned_allocations": [],
        "separations": [],
        "temporal_exclusions": [],
    }
    assert len(document["problem"]["pools"]) == 1
    vec_pool = document["problem"]["pools"][0]
    assert vec_pool["id"] == 1
    assert vec_pool["name"] == "Vec"
    assert vec_pool["capacity"] > 0
    assert vec_pool["reserved_ranges"] == []
    assert all(buffer["alignment"] == 32 for buffer in document["problem"]["buffers"])
    assert document["problem"]["pypto_structure"] == {
        "alias_classes": [
            {"buffer": 0, "members": ["tile_a"]},
            {"buffer": 1, "members": ["tile_b"]},
            {"buffer": 2, "members": ["tile_c"]},
        ],
        "pipeline_groups": [],
    }


@requires_dsa
def test_dsa_replays_fingerprinted_solution(tmp_path):
    """A saved placement is revalidated against a fresh export before writeback."""
    base = _dsa_chain_program()
    artifact_dir = tmp_path / "artifacts"
    solved = _allocate_with_dsa(base, export_dir=str(artifact_dir))
    replayed = _allocate_with_dsa(base, solution_dir=str(artifact_dir))

    ir.assert_structural_equal(solved, replayed)
    solution = json.loads((artifact_dir / "pypto_read_before_write_chain.dsa.solution.json").read_text())
    assert solution["schema_version"] == 1
    assert solution["profile"] == "pypto_hard_v1"
    assert solution["instance"] == "read_before_write_chain"
    assert solution["metadata"]["solver"] == "first_fit"
    assert len(solution["problem_fingerprint"]) == 16
    assert solution["placements"] == [
        {"buffer": 0, "offset": 0, "pool": 1},
        {"buffer": 1, "offset": 0, "pool": 1},
        {"buffer": 2, "offset": 0, "pool": 1},
    ]


def _dsa_pipeline_separation_program():
    """Two disjoint pipeline clones whose stage provenance forbids one address."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def pipeline_stage_separation(
            self,
            input_0: pl.Tensor[[64, 64], pl.FP32],
            input_1: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            stage_0: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(
                input_0,
                [0, 0],
                [64, 64],
                [64, 64],
                target_memory=pl.Mem.Vec,
                attrs={"pipeline_membership": "7:0"},
            )
            _stored_0 = pl.tile.store(stage_0, [0, 0], output)
            stage_1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(
                input_1,
                [0, 0],
                [64, 64],
                [64, 64],
                target_memory=pl.Mem.Vec,
                attrs={"pipeline_membership": "7:1"},
            )
            result = pl.tile.store(stage_1, [0, 0], output)
            return result

    return passes.init_mem_ref()(Before)


def _dsa_reuse_recognizer_program():
    """Independent load after one adjacent and one non-adjacent dead buffer."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def reuse_recognizer(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a = pl.load(input_a, [0, 0], [64, 64])
            tile_b = pl.add(tile_a, tile_a)
            tile_b2 = pl.add(tile_b, tile_b)
            tile_c = pl.load(input_b, [0, 0], [64, 64])
            tile_d = pl.add(tile_b2, tile_c)
            result = pl.store(tile_d, [0, 0], output)
            return result

    return passes.init_mem_ref()(Before)


def _dsa_same_resource_waw_program():
    """Two independent Vec definitions form a same-resource WAW candidate."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def same_resource_waw(
            self,
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            _unused: pl.Tile[[64, 64], pl.FP32] = pl.tile.full([64, 64], dtype=pl.FP32, value=0.0)
            kept: pl.Tile[[64, 64], pl.FP32] = pl.tile.full([64, 64], dtype=pl.FP32, value=1.0)
            result = pl.store(kept, [0, 0], output)
            return result

    return passes.init_mem_ref()(Before)


def _dsa_multi_resource_frontier_program():
    """The last access on one resource must not hide an earlier async access."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def multi_resource_frontier(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a = pl.load(input_a, [0, 0], [64, 64])
            _stored = pl.store(tile_a, [0, 0], output)
            # tile.abs was intentionally absent from the old operation-name
            # allowlist; the route classifier derives Vector from UB geometry.
            _last_vector_use = pl.abs(tile_a)
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.tile.full([64, 64], dtype=pl.FP32, value=1.0)
            result = pl.store(tile_b, [0, 0], output)
            return result

    return passes.init_mem_ref()(Before)


def _dsa_abstract_route_table_program():
    """Exercise inbound, L1-to-L0, matrix, and L0-drain routes."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIC)
        def abstract_route_table(
            self,
            lhs: pl.Tensor[[16, 16], pl.BF16],
            rhs: pl.Tensor[[16, 16], pl.BF16],
            output: pl.Tensor[[16, 16], pl.FP32],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            lhs_l1: pl.Tile[[16, 16], pl.BF16] = pl.load(lhs, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            rhs_l1: pl.Tile[[16, 16], pl.BF16] = pl.load(rhs, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            lhs_l0: pl.Tile[[16, 16], pl.BF16] = pl.tile.move(lhs_l1, target_memory=pl.Mem.Left)
            rhs_l0: pl.Tile[[16, 16], pl.BF16] = pl.tile.move(rhs_l1, target_memory=pl.Mem.Right)
            acc: pl.Tile[[16, 16], pl.FP32] = pl.matmul(lhs_l0, rhs_l0)
            result = pl.store(acc, [0, 0], output)
            return result

    return passes.init_mem_ref()(Before)


def _dsa_multiple_view_operands_program():
    """Two operands from distinct ranges of one allocation must both survive."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def multiple_view_operands(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[32, 64], pl.FP32],
        ) -> pl.Tensor[[32, 64], pl.FP32]:
            tile_a = pl.load(input_a, [0, 0], [64, 64])
            upper = tile_a[0:32, 0:64]
            lower = tile_a[32:64, 0:64]
            _combined = pl.add(upper, lower)
            later: pl.Tile[[32, 64], pl.FP32] = pl.tile.full([32, 64], dtype=pl.FP32, value=1.0)
            return pl.store(later, [0, 0], output)

    return passes.init_mem_ref()(Before)


def _dsa_directly_ordered_handoff_program():
    """A store result orders the next load independently of tile addresses."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def directly_ordered_handoff(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            scratch: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a = pl.load(input_a, [0, 0], [64, 64])
            updated = pl.store(tile_a, [0, 0], scratch)
            tile_b = pl.load(updated, [0, 0], [64, 64])
            result = pl.store(tile_b, [0, 0], output)
            return result

    return passes.init_mem_ref()(Before)


def _dsa_transitively_ordered_handoff_program():
    """A GM chain orders non-adjacent allocation handoffs transitively."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def transitively_ordered_handoff(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            scratch_a: pl.Tensor[[64, 64], pl.FP32],
            scratch_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a = pl.load(input_a, [0, 0], [64, 64])
            updated_a = pl.store(tile_a, [0, 0], scratch_a)
            tile_middle = pl.load(updated_a, [0, 0], [64, 64])
            updated_b = pl.store(tile_middle, [0, 0], scratch_b)
            tile_b = pl.load(updated_b, [0, 0], [64, 64])
            result = pl.store(tile_b, [0, 0], output)
            return result

    return passes.init_mem_ref()(Before)


def _dsa_partial_view_program():
    """A sliced allocation is deliberately outside recognizer v1."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def partial_view(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[32, 64], pl.FP32],
        ) -> pl.Tensor[[32, 64], pl.FP32]:
            tile_a = pl.load(input_a, [0, 0], [64, 64])
            view = tile_a[0:32, 0:64]
            result = pl.store(view, [0, 0], output)
            return result

    return passes.init_mem_ref()(Before)


def _dsa_nested_control_program():
    """A loop-containing region is deliberately outside recognizer v1."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def nested_control(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a = pl.load(input_a, [0, 0], [64, 64])
            for _i in pl.range(1):
                _first = pl.tile.full([64, 64], dtype=pl.FP32, value=0.0)
                _second = pl.tile.full([64, 64], dtype=pl.FP32, value=1.0)
            result = pl.store(tile_a, [0, 0], output)
            return result

    return passes.init_mem_ref()(Before)


def _dsa_opposite_branch_loop_program():
    """Opposite branches can execute in consecutive loop iterations."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def opposite_branch_loop(
            self,
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            for i in pl.range(2):
                if i == 0:
                    _first = pl.tile.full([64, 64], dtype=pl.FP32, value=0.0)
                else:
                    _second = pl.tile.full([64, 64], dtype=pl.FP32, value=1.0)
            result = pl.tile.full([64, 64], dtype=pl.FP32, value=2.0)
            return pl.store(result, [0, 0], output)

    return passes.init_mem_ref()(Before)


def _dsa_inner_to_outer_loop_program():
    """An inner-loop access can hand off across its enclosing-loop backedge."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def inner_to_outer_loop(
            self,
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            for _i in pl.range(2):
                for _j in pl.range(1):
                    _inner = pl.tile.full([64, 64], dtype=pl.FP32, value=0.0)
                _outer = pl.tile.full([64, 64], dtype=pl.FP32, value=1.0)
            result = pl.tile.full([64, 64], dtype=pl.FP32, value=2.0)
            return pl.store(result, [0, 0], output)

    return passes.init_mem_ref()(Before)


def _dsa_assemble_hazard_program():
    """A mutating inherit-input op must contribute a physical write."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def assemble_hazard(
            self,
            input_a: pl.Tensor[[32, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            target = pl.tile.create([64, 64], pl.FP32)
            source = pl.load(input_a, [0, 0], [32, 64])
            _assembled = pl.tile.assemble(target, source, [0, 0])
            later = pl.load(input_b, [0, 0], [64, 64])
            return pl.store(later, [0, 0], output)

    return passes.init_mem_ref()(Before)


def _dsa_tuple_result_hazard_program():
    """Tuple elements written by one operation are physical DSA accesses."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def tuple_result_hazard(
            self,
            src: pl.Tensor[[16, 64], pl.INT32],
            kvalue: pl.Scalar[pl.INT32],
            out_dst: pl.Tensor[[16, 8], pl.INT32],
            out_count: pl.Tensor[[1, 16], pl.INT32],
        ) -> tuple[pl.Tensor[[16, 8], pl.INT32], pl.Tensor[[1, 16], pl.INT32]]:
            source = pl.load(src, [0, 0], [16, 64])
            tmp = pl.tile.create([16, 64], pl.UINT8)
            dst, count = pl.tile.gather_compare(
                source,
                kvalue,
                tmp,
                cmp_mode="eq",
                out_cols=8,
            )
            dst_result = pl.store(dst, [0, 0], out_dst)
            count_result = pl.store(count, [0, 0], out_count)
            return dst_result, count_result

    return passes.init_mem_ref()(Before)


def _dsa_no_access_definition_program():
    """A tile declaration is not a hardware write endpoint."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def no_access_definition(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            _declaration_only = pl.tile.create([64, 64], pl.FP32)
            actual = pl.load(input_a, [0, 0], [64, 64])
            return pl.store(actual, [0, 0], output)

    return passes.init_mem_ref()(Before)


@requires_dsa
def test_dsa_quadratic_reuse_recognizer_exports_cross_resource_edges(tmp_path):
    """The coverage recognizer finds adjacent and non-adjacent handoffs."""
    expected_pairs = {(0, 3), (1, 3)}
    export_dir = tmp_path / "quadratic"
    _allocate_with_dsa(
        _dsa_reuse_recognizer_program(),
        str(export_dir),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    document = json.loads((export_dir / "pypto_reuse_recognizer.dsa.json").read_text())
    assert document["profile"] == "pypto_research_v1"
    penalties = document["problem"]["cost_model"]["reuse_penalties"]
    assert {(entry["first"], entry["second"]) for entry in penalties} == expected_pairs
    assert all(entry["cost"] == 1 and entry["reason"] == "cross_pipe" for entry in penalties)
    assert document["problem"]["objective"]["terms"] == [
        "capacity_overflow",
        "reuse_cost",
        "total_peak",
        "max_peak",
    ]
    assert document["metadata"]["recognized_reuse_penalties"] == str(len(expected_pairs))
    assert int(document["metadata"]["recognized_reuse_candidates"]) >= len(expected_pairs)
    assert int(document["metadata"]["recognized_cross_resource_candidates"]) >= len(expected_pairs)
    assert int(document["metadata"]["recognized_write_after_read_candidates"]) >= len(expected_pairs)
    assert document["metadata"]["recognized_nested_control_candidates"] == "0"
    assert document["metadata"]["reuse_penalty_promotion_policy"] == "cross_resource_unit_v3"
    assert "external->ub@inbound_dma" in document["metadata"]["recognized_reuse_candidate_records_v3"]

    solution = json.loads((export_dir / "pypto_reuse_recognizer.dsa.solution.json").read_text())
    assert solution["metadata"]["solver"] == "pypto_structured_search"
    offsets = {entry["buffer"]: entry["offset"] for entry in solution["placements"]}
    sizes = {entry["id"]: entry["size"] for entry in document["problem"]["buffers"]}
    for first, second in expected_pairs:
        assert (
            offsets[first] + sizes[first] <= offsets[second]
            or offsets[second] + sizes[second] <= offsets[first]
        )


@requires_dsa
def test_dsa_quadratic_recognizer_records_tile_assemble_write(tmp_path):
    """Memory-space inheritance must not hide tile.assemble's real write."""
    _allocate_with_dsa(
        _dsa_assemble_hazard_program(),
        str(tmp_path),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    document = json.loads((tmp_path / "pypto_assemble_hazard.dsa.json").read_text())
    records = document["metadata"]["recognized_reuse_candidate_records_v3"]
    assert "ub->ub@vector_compute=>external->ub@inbound_dma" in records
    assert "write_after_write" in records


@requires_dsa
def test_dsa_quadratic_recognizer_records_tuple_result_writes(tmp_path):
    """Both tile.gather_compare outputs participate in access-frontier construction."""
    _allocate_with_dsa(
        _dsa_tuple_result_hazard_program(),
        str(tmp_path),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    document = json.loads((tmp_path / "pypto_tuple_result_hazard.dsa.json").read_text())
    routes = set(document["metadata"]["recognized_access_routes_v1"].split(";"))
    assert "ub->ub@vector_compute" in routes
    assert document["metadata"]["reuse_penalty_partially_supported_allocations"] == "0"
    assert document["metadata"]["reuse_penalty_supported_allocations"] == "4"


@requires_dsa
def test_dsa_quadratic_recognizer_ignores_no_access_definitions(tmp_path):
    """tile.create allocates storage but does not create a WAW execution hazard."""
    _allocate_with_dsa(
        _dsa_no_access_definition_program(),
        str(tmp_path),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    document = json.loads((tmp_path / "pypto_no_access_definition.dsa.json").read_text())
    assert document["metadata"]["reuse_penalty_supported_allocations"] == "1"
    assert document["metadata"]["recognized_reuse_candidates"] == "0"
    assert "cost_model" not in document["problem"]


@requires_dsa
def test_dsa_reuse_recognizer_records_same_resource_waw_without_promoting_it(tmp_path):
    """Same-resource candidates remain report-only under the experimental v3 policy."""
    _allocate_with_dsa(
        _dsa_same_resource_waw_program(),
        str(tmp_path),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    document = json.loads((tmp_path / "pypto_same_resource_waw.dsa.json").read_text())
    assert document["profile"] == "pypto_hard_v1"
    assert "cost_model" not in document["problem"]
    assert document["metadata"]["recognized_reuse_candidates"] == "1"
    assert document["metadata"]["recognized_cross_resource_candidates"] == "0"
    assert document["metadata"]["recognized_same_resource_candidates"] == "1"
    assert document["metadata"]["recognized_write_after_read_candidates"] == "0"
    assert document["metadata"]["recognized_write_after_write_candidates"] == "1"
    assert document["metadata"]["recognized_nested_control_candidates"] == "0"
    assert document["metadata"]["recognized_reuse_candidate_records_v1"] == (
        "0,1,same_pipe,write_after_write,flat"
    )
    assert document["metadata"].get("recognized_reuse_penalties", "0") == "0"


@requires_dsa
def test_dsa_reuse_recognizer_records_logically_ordered_handoff_without_promoting_it(tmp_path):
    """SSA reachability is retained as evidence, not mistaken for completion."""
    export_dir = tmp_path / "quadratic"
    _allocate_with_dsa(
        _dsa_directly_ordered_handoff_program(),
        str(export_dir),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    document = json.loads((export_dir / "pypto_directly_ordered_handoff.dsa.json").read_text())
    assert document["metadata"]["reuse_penalty_candidate_pairs"] != "0"
    assert document["metadata"]["reuse_penalty_already_ordered_pairs"] != "0"
    assert document["metadata"]["recognized_reuse_candidates"] != "0"
    assert (
        document["metadata"]["recognized_ordered_evidence_candidates"]
        == document["metadata"]["recognized_reuse_candidates"]
    )
    assert "cost_model" not in document["problem"]


@requires_dsa
def test_dsa_quadratic_recognizer_records_transitive_order_as_evidence(tmp_path):
    """Transitive SSA order is reported while the async completion question remains open."""
    _allocate_with_dsa(
        _dsa_transitively_ordered_handoff_program(),
        str(tmp_path),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    document = json.loads((tmp_path / "pypto_transitively_ordered_handoff.dsa.json").read_text())
    assert int(document["metadata"]["reuse_penalty_candidate_pairs"]) >= 3
    assert int(document["metadata"]["reuse_penalty_already_ordered_pairs"]) >= 3
    assert document["metadata"]["recognized_reuse_candidates"] != "0"
    assert (
        document["metadata"]["recognized_ordered_evidence_candidates"]
        == document["metadata"]["recognized_reuse_candidates"]
    )
    assert "cost_model" not in document["problem"]


@requires_dsa
def test_dsa_reuse_recognizer_does_not_invent_partial_view_pairs(tmp_path):
    """A lone view-backed allocation does not create a synthetic reuse pair."""
    _allocate_with_dsa(
        _dsa_partial_view_program(),
        str(tmp_path),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    document = json.loads((tmp_path / "pypto_partial_view.dsa.json").read_text())
    assert document["metadata"]["reuse_penalty_supported_allocations"] == "1"
    assert document["metadata"]["recognized_reuse_candidates"] == "0"
    assert "cost_model" not in document["problem"]


@requires_dsa
def test_dsa_quadratic_recognizer_reports_nested_candidates_without_promoting_them(tmp_path):
    """Quadratic mode exposes forward and distance-one loop reuse handoffs."""
    _allocate_with_dsa(
        _dsa_nested_control_program(),
        str(tmp_path),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    document = json.loads((tmp_path / "pypto_nested_control.dsa.json").read_text())
    assert document["profile"] == "pypto_hard_v1"
    assert document["metadata"]["recognized_reuse_candidates"] == "2"
    assert document["metadata"]["recognized_same_resource_candidates"] == "2"
    assert document["metadata"]["recognized_nested_control_candidates"] == "2"
    assert document["metadata"]["recognized_in_loop_candidates"] == "2"
    assert document["metadata"]["recognized_loop_carried_candidates"] == "1"
    assert "distance_0" in document["metadata"]["recognized_reuse_candidate_records_v3"]
    assert "distance_1" in document["metadata"]["recognized_reuse_candidate_records_v3"]
    assert "cost_model" not in document["problem"]


@requires_dsa
def test_dsa_quadratic_recognizer_keeps_one_terminal_access_per_resource(tmp_path):
    """A later vector access does not hide an earlier outbound-DMA access."""
    _allocate_with_dsa(
        _dsa_multi_resource_frontier_program(),
        str(tmp_path),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    document = json.loads((tmp_path / "pypto_multi_resource_frontier.dsa.json").read_text())
    records = document["metadata"]["recognized_reuse_candidate_records_v3"]
    assert "ub->external@outbound_dma=>ub->ub@vector_compute" in records
    assert "ub->ub@vector_compute=>ub->ub@vector_compute" in records


@requires_dsa
def test_dsa_quadratic_recognizer_uses_abstract_route_table(tmp_path):
    """Route evidence comes from memory classes rather than operation-name pipe lists."""
    _allocate_with_dsa(
        _dsa_abstract_route_table_program(),
        str(tmp_path),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    document = json.loads((tmp_path / "pypto_abstract_route_table.dsa.json").read_text())
    routes = set(document["metadata"]["recognized_access_routes_v1"].split(";"))
    assert routes == {
        "external->l1@inbound_dma",
        "l1->l0@l1_to_l0",
        "l0->l0@matrix_compute",
        "l0->external@l0_to_external",
    }


@requires_dsa
def test_dsa_quadratic_recognizer_preserves_multiple_view_operands(tmp_path):
    """Distinct subranges read by one operation remain distinct access sites."""
    _allocate_with_dsa(
        _dsa_multiple_view_operands_program(),
        str(tmp_path),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    document = json.loads((tmp_path / "pypto_multiple_view_operands.dsa.json").read_text())
    records = document["metadata"]["recognized_reuse_candidate_records_v3"]
    assert "ranges=0+8192->0+8192" in records
    assert "ranges=8192+8192->0+8192" in records
    assert document["metadata"]["recognized_partial_access_candidates"] == "4"


@requires_dsa
def test_dsa_quadratic_recognizer_keeps_opposite_branches_across_iterations(tmp_path):
    """Branch exclusivity in one iteration does not suppress a backedge handoff."""
    _allocate_with_dsa(
        _dsa_opposite_branch_loop_program(),
        str(tmp_path),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    document = json.loads((tmp_path / "pypto_opposite_branch_loop.dsa.json").read_text())
    assert int(document["metadata"]["recognized_loop_carried_candidates"]) >= 1
    assert "distance_1" in document["metadata"]["recognized_reuse_candidate_records_v3"]
    assert "cost_model" not in document["problem"]


@requires_dsa
def test_dsa_quadratic_recognizer_uses_common_enclosing_loop(tmp_path):
    """Different loop stacks still share the outer-loop backedge."""
    _allocate_with_dsa(
        _dsa_inner_to_outer_loop_program(),
        str(tmp_path),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    document = json.loads((tmp_path / "pypto_inner_to_outer_loop.dsa.json").read_text())
    records = document["metadata"]["recognized_reuse_candidate_records_v3"]
    assert "distance_1" in records
    assert ",loop=" in records
    assert "cost_model" not in document["problem"]


@requires_dsa
def test_dsa_replays_recognized_reuse_problem_by_fingerprint(tmp_path):
    """Research-profile recognition is replayed as the strict recognized problem."""
    export_dir = tmp_path / "recognized"
    base = _dsa_reuse_recognizer_program()
    planned = _allocate_with_dsa(
        base,
        str(export_dir),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )
    replayed = _allocate_with_dsa(
        base,
        solution_dir=str(export_dir),
        reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
    )

    ir.assert_structural_equal(planned, replayed)
    with pytest.raises(ValueError, match="matches neither the strict recognized problem"):
        _allocate_with_dsa(
            base,
            solution_dir=str(export_dir),
            reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.DISABLED,
        )


@requires_dsa
def test_dsa_export_and_solver_preserve_pipeline_stage_separation(tmp_path):
    """PyPTO pipeline provenance becomes a checked standalone separation."""
    planned = _allocate_with_dsa(_dsa_pipeline_separation_program(), str(tmp_path))
    assert _vec_peak(planned) == 2 * 16384

    corpus_file = tmp_path / "pypto_pipeline_stage_separation.dsa.json"
    document = json.loads(corpus_file.read_text())
    assert document["instance"] == "pipeline_stage_separation"
    assert document["profile"] == "pypto_hard_v1"
    assert document["metadata"]["pipeline_intent_policy"] == "hard_requested_depth"
    assert document["problem"]["constraints"]["separations"] == [
        {"first": 0, "second": 1, "reasons": ["pipeline_stage"]}
    ]
    assert document["problem"]["pypto_structure"] == {
        "alias_classes": [
            {"buffer": 0, "members": ["stage_0"]},
            {"buffer": 1, "members": ["stage_1"]},
        ],
        "pipeline_groups": [
            {
                "group": 7,
                "pool": 1,
                "slot_size": 16384,
                "depth": 2,
                "effective_depth": 2,
                "members": [
                    {"buffer": 0, "stage": 0, "residue": 0},
                    {"buffer": 1, "stage": 1, "residue": 1},
                ],
            }
        ],
    }


def _dsa_capacity_gated_pipeline_cost_program():
    """Three sequential 240 KiB stages collapse to one capacity residue."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def capacity_gated_pipeline_cost(
            self,
            input_0: pl.Tensor[[128, 480], pl.FP32],
            input_1: pl.Tensor[[128, 480], pl.FP32],
            input_2: pl.Tensor[[128, 480], pl.FP32],
            output_0: pl.Out[pl.Tensor[[128, 480], pl.FP32]],
            output_1: pl.Out[pl.Tensor[[128, 480], pl.FP32]],
            output_2: pl.Out[pl.Tensor[[128, 480], pl.FP32]],
        ) -> pl.Tensor[[128, 480], pl.FP32]:
            stage_0 = pl.tile.load(
                input_0,
                [0, 0],
                [128, 480],
                [128, 480],
                target_memory=pl.Mem.Vec,
                attrs={"pipeline_membership": "11:0"},
            )
            _stored_0 = pl.tile.store(stage_0, [0, 0], output_0)
            stage_1 = pl.tile.load(
                input_1,
                [0, 0],
                [128, 480],
                [128, 480],
                target_memory=pl.Mem.Vec,
                attrs={"pipeline_membership": "11:1"},
            )
            _stored_1 = pl.tile.store(stage_1, [0, 0], output_1)
            stage_2 = pl.tile.load(
                input_2,
                [0, 0],
                [128, 480],
                [128, 480],
                target_memory=pl.Mem.Vec,
                attrs={"pipeline_membership": "11:2"},
            )
            result = pl.tile.store(stage_2, [0, 0], output_2)
            return result

    return passes.init_mem_ref()(Before)


@requires_dsa
def test_dsa_pipeline_intent_falls_back_with_reuse_cost_and_warning(tmp_path):
    """Full pipeline intent is tried first; capacity fallback is explicit and loud."""
    report_dir = tmp_path / "report"
    export_dir = tmp_path / "corpus"
    report_dir.mkdir()
    with passes.PassContext(
        [passes.ReportInstrument(str(report_dir))],
        memory_planner=passes.MemoryPlanner.DSA,
        dsa_export_dir=str(export_dir),
    ):
        planned = passes.allocate_memory_addr()(_dsa_capacity_gated_pipeline_cost_program())
    assert _vec_peak(planned) == 240 * 1024
    replayed = _allocate_with_dsa(
        _dsa_capacity_gated_pipeline_cost_program(),
        solution_dir=str(export_dir),
    )
    ir.assert_structural_equal(planned, replayed)

    document = json.loads((export_dir / "pypto_capacity_gated_pipeline_cost.dsa.json").read_text())
    assert document["profile"] == "pypto_research_v1"
    assert document["metadata"]["experimental_features"] == "pipeline_intent_fallback"
    assert document["metadata"]["pipeline_intent_policy"] == "soft_after_strict_no_fit"
    assert document["problem"]["constraints"]["separations"] == []
    group = document["problem"]["pypto_structure"]["pipeline_groups"][0]
    assert (group["depth"], group["effective_depth"], group["slot_size"]) == (3, 3, 240 * 1024)
    assert [member["residue"] for member in group["members"]] == [0, 1, 2]
    assert document["problem"]["cost_model"]["reuse_penalties"] == [
        {"first": 0, "second": 1, "cost": 1, "reason": "pipeline_serialization"},
        {"first": 0, "second": 2, "cost": 1, "reason": "pipeline_serialization"},
        {"first": 1, "second": 2, "cost": 1, "reason": "pipeline_serialization"},
    ]
    assert document["metadata"]["reuse_cost_model"] == "pipeline_stage_overlap_pairs_v1"
    hint = (report_dir / "perf_hints.log").read_text()
    assert "PH-DSA-001" in hint
    assert "could not find a capacity-fitting placement" in hint
    assert "software-pipeline overlap may be reduced" in hint


@requires_dsa
def test_dsa_pipeline_fallback_accounts_for_reserved_space(tmp_path):
    """Reserved space participates in the strict-fit decision before fallback."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def reserved_pipeline_depth(
            self,
            input_0: pl.Tensor[[128, 240], pl.FP32],
            input_1: pl.Tensor[[128, 240], pl.FP32],
            output_0: pl.Out[pl.Tensor[[128, 240], pl.FP32]],
            output_1: pl.Out[pl.Tensor[[128, 240], pl.FP32]],
        ) -> pl.Tensor[[128, 240], pl.FP32]:
            _ = pl.reserve_buffer(name="runtime_window", size=32768)
            stage_0 = pl.tile.load(
                input_0,
                [0, 0],
                [128, 240],
                [128, 240],
                target_memory=pl.Mem.Vec,
                attrs={"pipeline_membership": "17:0"},
            )
            _stored_0 = pl.tile.store(stage_0, [0, 0], output_0)
            stage_1 = pl.tile.load(
                input_1,
                [0, 0],
                [128, 240],
                [128, 240],
                target_memory=pl.Mem.Vec,
                attrs={"pipeline_membership": "17:1"},
            )
            result = pl.tile.store(stage_1, [0, 0], output_1)
            return result

    planned = _allocate_with_dsa(passes.init_mem_ref()(Before), str(tmp_path))
    assert _vec_peak(planned) == (32 + 120) * 1024

    document = json.loads((tmp_path / "pypto_reserved_pipeline_depth.dsa.json").read_text())
    group = document["problem"]["pypto_structure"]["pipeline_groups"][0]
    assert document["profile"] == "pypto_research_v1"
    assert (group["depth"], group["effective_depth"]) == (2, 2)
    assert document["problem"]["constraints"]["separations"] == []
    assert document["problem"]["cost_model"]["reuse_penalties"] == [
        {"first": 0, "second": 1, "cost": 1, "reason": "pipeline_serialization"}
    ]
    assert document["problem"]["pools"][0]["reserved_ranges"] == [{"begin": 0, "end": 32 * 1024}]


@requires_dsa
def test_dsa_export_preserves_ascend910b_target_hazard_reason(tmp_path):
    """The split-AIV load+tpop keep-apart edge remains identifiable offline."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
        def target_hazard(self, down: pl.InOut[pl.Tensor[[16, 128], pl.FP32]]):
            mem_vec_0: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
            mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
            mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
            down_prev: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_0, 0, 4096), pl.Mem.Vec] = pl.tile.load(
                down, [0, 0], [8, 128], [8, 128], target_memory=pl.Mem.Vec
            )
            pipe_chunk: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_1, 0, 4096), pl.Mem.Vec] = (
                pl.tile.tpop_from_aic(split=1)
            )
            down_next: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_2, 0, 4096), pl.Mem.Vec] = pl.tile.add(
                down_prev, pipe_chunk
            )
            result = pl.tile.store(down_next, [0, 0], down)
            return result

    was_configured = is_backend_configured()
    prior_type = get_backend_type() if was_configured else None
    if was_configured:
        reset_for_testing()
    try:
        set_backend_type(BackendType.Ascend910B)
        _allocate_with_dsa(Before, str(tmp_path))
    finally:
        reset_for_testing()
        if prior_type is not None:
            set_backend_type(prior_type)

    document = json.loads((tmp_path / "pypto_target_hazard.dsa.json").read_text())
    assert document["metadata"]["target"] == "Ascend910B"
    assert document["problem"]["constraints"]["separations"] == [
        {"first": 0, "second": 2, "reasons": ["target_hazard"]}
    ]


def _dsa_fragmentation_program():
    """Real IR form of #1908: a freed 64 KB region must hold two 32 KB tiles."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def issue_1908_fragmentation(
            self,
            large_input: pl.Tensor[[128, 128], pl.FP32],
            left_input: pl.Tensor[[64, 128], pl.FP32],
            right_input: pl.Tensor[[64, 128], pl.FP32],
            large_output: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            output: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            producer: pl.Tile[[128, 128], pl.FP32] = pl.load(large_input, [0, 0], [128, 128])
            _stored = pl.store(producer, [0, 0], large_output)
            left: pl.Tile[[64, 128], pl.FP32] = pl.load(left_input, [0, 0], [64, 128])
            right: pl.Tile[[64, 128], pl.FP32] = pl.load(right_input, [0, 0], [64, 128])
            combined: pl.Tile[[64, 128], pl.FP32] = pl.add(left, right)
            result = pl.store(combined, [0, 0], output)
            return result

    return passes.init_mem_ref()(Before)


@requires_dsa
def test_dsa_planner_subdivides_a_freed_larger_region(tmp_path):
    """Regression for #1908: later co-live buffers subdivide one freed region."""
    base = _dsa_fragmentation_program()
    bump = passes.allocate_memory_addr()(base)
    planned = _allocate_with_dsa(base, str(tmp_path))

    assert _vec_peak(bump) == (64 + 32 + 32 + 32) * 1024
    assert _vec_peak(planned) == 64 * 1024

    corpus_file = tmp_path / "pypto_issue_1908_fragmentation.dsa.json"
    document = json.loads(corpus_file.read_text())
    assert document["instance"] == "issue_1908_fragmentation"
    assert [buffer["size"] for buffer in document["problem"]["buffers"]] == [
        64 * 1024,
        32 * 1024,
        32 * 1024,
        32 * 1024,
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
