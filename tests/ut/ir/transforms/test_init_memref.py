# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for InitMemRefPass.

Most tests use the Before/Expected pattern with
``ir.assert_structural_equal(After, Expected)``.
DefFields always auto-map, so ``enable_auto_mapping=True`` is unnecessary.
This aligns MemRef objects consistently: if two tiles share a MemRef in
``After``, the corresponding tiles in ``Expected`` must also share.

Two tests are kept as raw-IR / diagnostic tests because the inputs cannot be
expressed via the DSL:
  * ``test_rejects_dynamic_tile_shape`` — verifies ``pytest.raises`` on B3.
  * ``test_if_phi_preserves_dynamic_valid_shape_vars`` — a regression test for
    issue #870 that constructs a ``TileView`` with dynamic ``valid_shape``
    Vars; this has no DSL syntax.
"""

from typing import cast

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir import MemorySpace


class TestBasic:
    """Basic MemRef creation, memory space assignment, and alloc generation."""

    def test_simple_load_add_store(self):
        """load-add-store sequence: Vec tiles get unique MemRefs, params get DDR."""

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
                tile_sum: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_b)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_sum, [0, 0], output)
                return result

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
                mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_b, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                tile_sum: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.add(tile_a, tile_b)
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_2", 0, 16384)] = pl.tile.store(
                    tile_sum, [0, 0], output
                )
                return result

        After = passes.init_mem_ref()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_matmul_pipeline(self):
        """load→move→matmul→store: Vec/Mat/Left/Right/Acc memory spaces each get their own MemRef."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP16],
                input_b: pl.Tensor[[32, 32], pl.FP16],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a_ub: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                tile_b_l1: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Mat] = pl.load(
                    input_b, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
                )
                tile_a_l0a: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Left] = pl.move(
                    tile_a_ub, target_memory=pl.MemorySpace.Left
                )
                tile_b_l0b: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Right] = pl.move(
                    tile_b_l1, target_memory=pl.MemorySpace.Right
                )
                tile_result: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(
                    tile_a_l0a, tile_b_l0b
                )
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_result, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP16, pl.MemRef("mem_ddr_0", 0, 2048)],
                input_b: pl.Tensor[[32, 32], pl.FP16, pl.MemRef("mem_ddr_1", 0, 2048)],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
                mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 2048)
                mem_left_5: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 2048)
                mem_right_6: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 2048)
                mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
                tile_a_ub: pl.Tile[[32, 32], pl.FP16, pl.MemRef(mem_vec_3, 0, 2048), pl.Mem.Vec] = (
                    pl.tile.load(
                        input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                    )
                )
                tile_b_l1: pl.Tile[[32, 32], pl.FP16, pl.MemRef(mem_mat_4, 0, 2048), pl.Mem.Mat] = (
                    pl.tile.load(
                        input_b, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Mat, transpose=False
                    )
                )
                tile_a_l0a: pl.Tile[[32, 32], pl.FP16, pl.MemRef(mem_left_5, 0, 2048), pl.Mem.Left] = (
                    pl.tile.move(tile_a_ub, target_memory=pl.Mem.Left)
                )
                tile_b_l0b: pl.Tile[[32, 32], pl.FP16, pl.MemRef(mem_right_6, 0, 2048), pl.Mem.Right] = (
                    pl.tile.move(tile_b_l1, target_memory=pl.Mem.Right)
                )
                tile_result: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_acc_7, 0, 4096), pl.Mem.Acc] = (
                    pl.tile.matmul(tile_a_l0a, tile_b_l0b)
                )
                result: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)] = pl.tile.store(
                    tile_result, [0, 0], output
                )
                return result

        After = passes.init_mem_ref()(Before)
        ir.assert_structural_equal(After, Expected)


class TestMemRefSharing:
    """MemRef sharing: tile.store shares with output param, view ops share with input."""

    def test_store_shares_memref_with_output_param(self):
        """tile.store result shares MemRef with the output tensor parameter."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_a, [0, 0], output)
                return result

        # ``output`` and ``result`` share the same "mem_ddr_1" pointer — this is
        # the store-shares-with-output relationship the test verifies.
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
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    tile_a, [0, 0], output
                )
                return result

        After = passes.init_mem_ref()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_view_op_shares_memref_with_input(self):
        """tile.reshape chain shares a single MemRef (only 1 alloc needed)."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                reshaped: pl.Tile[[4096, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(tile_a, [4096, 1])
                flat: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(reshaped, [64, 64])
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(flat, [0, 0], output)
                return result

        # All three tiles share ``mem_vec_2`` — reshape is a view op.
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
                reshaped: pl.Tile[[4096, 1], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.reshape(tile_a, [4096, 1])
                )
                flat: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.reshape(reshaped, [64, 64])
                )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    flat, [0, 0], output
                )
                return result

        After = passes.init_mem_ref()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_matmul_acc_shares_memref_with_accumulator(self):
        """tile.matmul_acc output shares MemRef with its accumulator input (arg[0])."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP16],
                input_b: pl.Tensor[[32, 32], pl.FP16],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a_ub: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                tile_b_l1: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Mat] = pl.load(
                    input_b, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
                )
                tile_a_l0a: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Left] = pl.move(
                    tile_a_ub, target_memory=pl.MemorySpace.Left
                )
                tile_b_l0b: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Right] = pl.move(
                    tile_b_l1, target_memory=pl.MemorySpace.Right
                )
                acc: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(tile_a_l0a, tile_b_l0b)
                acc_next: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Acc] = pl.matmul_acc(
                    acc, tile_a_l0a, tile_b_l0b
                )
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(acc_next, [0, 0], output)
                return result

        # ``acc`` and ``acc_next`` share ``mem_acc_7`` — matmul_acc reuses the
        # accumulator's storage.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP16, pl.MemRef("mem_ddr_0", 0, 2048)],
                input_b: pl.Tensor[[32, 32], pl.FP16, pl.MemRef("mem_ddr_1", 0, 2048)],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
                mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 2048)
                mem_left_5: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 2048)
                mem_right_6: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 2048)
                mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
                tile_a_ub: pl.Tile[[32, 32], pl.FP16, pl.MemRef(mem_vec_3, 0, 2048), pl.Mem.Vec] = (
                    pl.tile.load(
                        input_a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                    )
                )
                tile_b_l1: pl.Tile[[32, 32], pl.FP16, pl.MemRef(mem_mat_4, 0, 2048), pl.Mem.Mat] = (
                    pl.tile.load(
                        input_b, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Mat, transpose=False
                    )
                )
                tile_a_l0a: pl.Tile[[32, 32], pl.FP16, pl.MemRef(mem_left_5, 0, 2048), pl.Mem.Left] = (
                    pl.tile.move(tile_a_ub, target_memory=pl.Mem.Left)
                )
                tile_b_l0b: pl.Tile[[32, 32], pl.FP16, pl.MemRef(mem_right_6, 0, 2048), pl.Mem.Right] = (
                    pl.tile.move(tile_b_l1, target_memory=pl.Mem.Right)
                )
                acc: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_acc_7, 0, 4096), pl.Mem.Acc] = pl.tile.matmul(
                    tile_a_l0a, tile_b_l0b
                )
                acc_next: pl.Tile[[32, 32], pl.FP32, pl.MemRef(mem_acc_7, 0, 4096), pl.Mem.Acc] = (
                    pl.tile.matmul_acc(acc, tile_a_l0a, tile_b_l0b)
                )
                result: pl.Tensor[[32, 32], pl.FP32, pl.MemRef("mem_ddr_2", 0, 4096)] = pl.tile.store(
                    acc_next, [0, 0], output
                )
                return result

        After = passes.init_mem_ref()(Before)
        ir.assert_structural_equal(After, Expected)


class TestYieldMemRef:
    """MemRef propagation through yield in ForStmt and IfStmt."""

    def test_for_loop_carry_memref_relationships(self):
        """ForStmt: initValue/iter_arg share MemRef, yield/return_var share MemRef.

        Group A (initValue↔iter_arg) uses ``mem_vec_2``; Group B (yield↔return_var)
        uses ``mem_vec_4``. The two groups have different MemRefs — the yield-to-
        iter_arg mismatch is resolved later by MemoryReuse.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                init_tile: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                other_tile: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                for _i, (acc,) in pl.range(0, 4, init_values=(init_tile,)):
                    acc_next: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc, other_tile)
                    acc_out = pl.yield_(acc_next)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(acc_out, [0, 0], output)
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
                mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                init_tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.load(
                        input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                )
                other_tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                    pl.tile.load(
                        input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                )
                for _i, (acc,) in pl.range(0, 4, init_values=(init_tile,)):
                    acc_next: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(acc, other_tile)
                    )
                    acc_out: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_4, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(acc_next)
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    acc_out, [0, 0], output
                )
                return result

        After = passes.init_mem_ref()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_if_yield_return_var_shares_memref(self):
        """IfStmt: return_var shares MemRef with the then-branch yield value."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                cond: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                if cond < 2:
                    tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        input_tensor, [0, 0], [64, 64]
                    )
                    if_result = pl.yield_(tile_a)
                else:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        input_tensor, [0, 0], [64, 64]
                    )
                    if_result = pl.yield_(tile_b)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(if_result, [0, 0], output)
                return result

        # then-yield (tile_a) uses mem_vec_2; return_var (if_result) also uses
        # mem_vec_2 — shared per InitMemRef's phi-resolution rule. The else-
        # branch yield (tile_b) uses a separate mem_vec_3 — it's a distinct
        # temporary that MemoryReuse will later merge.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                cond: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                if cond < 2:
                    tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
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
                        pl.yield_(tile_a)
                    )
                else:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
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
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    if_result, [0, 0], output
                )
                return result

        After = passes.init_mem_ref()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_tile_alias_shares_source_memref(self):
        """Tile alias (``a = b``) shares MemRef with source tile, not a fresh one."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                cond: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                if cond < 2:
                    alias_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = tile_a
                    if_result = pl.yield_(alias_a)
                else:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                    if_result = pl.yield_(tile_b)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(if_result, [0, 0], output)
                return result

        # tile_a → alias_a → then-yield all share mem_vec_2 (alias chain).
        # The else-branch computation (tile_b) uses a fresh mem_vec_3.
        # The phi return_var (if_result) picks up the then-branch's mem_vec_2.
        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_0", 0, 16384)],
                cond: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = pl.tile.load(
                    input_tensor, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                if cond < 2:
                    alias_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = tile_a
                    if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(alias_a)
                    )
                else:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_3, 0, 16384), pl.Mem.Vec] = (
                        pl.tile.add(tile_a, tile_a)
                    )
                    if_result: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_2, 0, 16384), pl.Mem.Vec] = (
                        pl.yield_(tile_b)
                    )
                result: pl.Tensor[[64, 64], pl.FP32, pl.MemRef("mem_ddr_1", 0, 16384)] = pl.tile.store(
                    if_result, [0, 0], output
                )
                return result

        After = passes.init_mem_ref()(Before)
        ir.assert_structural_equal(After, Expected)


class TestDynamicValidShape:
    """Regression tests for dynamic valid_shape Var handling in phi-node return vars."""

    def test_if_phi_preserves_dynamic_valid_shape_vars(self):
        """IfStmt phi return vars must not clone Vars in TileView.valid_shape (issue #870).

        When PatchReturnVarsFromYield updates the return var's MemRef, it must not
        re-remap expressions that were already remapped by the base IRMutator visit.
        Double-remapping creates a fresh, undefined Var clone that fails UseAfterDef.

        Kept as raw-IR construction because ``TileView`` with dynamic ``valid_shape``
        Vars has no DSL syntax — cannot be expressed in the Before/Expected pattern.
        """
        span = ir.Span.unknown()
        idx = ir.DataType.INDEX

        # Params: flag (condition) and ctx_len (used to compute valid_len)
        flag = ir.Var("flag", ir.ScalarType(idx), span)
        ctx_len = ir.Var("ctx_len", ir.ScalarType(idx), span)

        # valid_len = ctx_len + 0  (defined before IfStmt)
        valid_len = ir.Var("valid_len", ir.ScalarType(idx), span)
        assign_valid_len = ir.AssignStmt(
            valid_len, ir.Add(ctx_len, ir.ConstInt(0, idx, span), idx, span), span
        )

        # TileType with dynamic valid_shape=[1, valid_len]
        tile_view = ir.TileView(
            [ir.ConstInt(1, idx, span), valid_len],
            [ir.ConstInt(1, idx, span), ir.ConstInt(120, idx, span)],
            ir.ConstInt(0, idx, span),
        )
        tile_type = ir.TileType([1, 120], ir.DataType.FP32, None, tile_view, MemorySpace.Vec)

        # Two tile vars: seed and updated
        seed = ir.Var("seed", tile_type, span)
        updated = ir.Var("updated", tile_type, span)
        tpop_call = ir.Call(ir.Op("tile.tpop_from_aic"), [], {"split": 0}, tile_type, span)
        muls_call = ir.Call(ir.Op("tile.muls"), [seed], {"scalar": 1.0}, tile_type, span)

        # Phi return var
        phi_var = ir.Var("result_phi", tile_type, span)

        # IfStmt: if flag == 0 then yield seed else yield updated
        condition = ir.Eq(flag, ir.ConstInt(0, idx, span), ir.DataType.BOOL, span)
        if_stmt = ir.IfStmt(
            condition,
            ir.YieldStmt([seed], span),
            ir.YieldStmt([updated], span),
            [phi_var],
            span,
        )

        body = ir.SeqStmts(
            [
                assign_valid_len,
                ir.AssignStmt(seed, tpop_call, span),
                ir.AssignStmt(updated, muls_call, span),
                if_stmt,
                ir.ReturnStmt([phi_var], span),
            ],
            span,
        )
        func = ir.Function("repro", [flag, ctx_len], [tile_type], body, span, type=ir.FunctionType.AIV)
        program = ir.Program([func], "test_program", span)

        # Run InitMemRef with verification but without roundtrip (raw IR may not
        # survive print→parse because TileView with dynamic Vars has no DSL syntax).
        with passes.PassContext(
            [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)],
        ):
            after = passes.init_mem_ref()(program)

        # Explicitly verify UseAfterDef — the bug caused this property to fail
        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.UseAfterDef)
        diagnostics = passes.PropertyVerifierRegistry.verify(props, after)
        errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
        assert not errors, f"UseAfterDef errors after InitMemRef: {[d.message for d in errors]}"

        # Double-check: return var's valid_shape must reference a defined Var
        func_after = next(iter(after.functions.values()))
        if_after = next(
            stmt for stmt in cast(ir.SeqStmts, func_after.body).stmts if isinstance(stmt, ir.IfStmt)
        )
        rv = if_after.return_vars[0]
        assert isinstance(rv.type, ir.TileType)
        assert rv.type.tile_view is not None
        vs = rv.type.tile_view.valid_shape
        assert len(vs) == 2
        assert isinstance(vs[1], ir.Var), "valid_shape[1] should be a Var, not a fresh clone"


class TestEdgeCases:
    """Edge cases requiring raw IR construction."""

    def test_rejects_dynamic_tile_shape(self):
        """InitMemRef must fail fast when allocation shape is still dynamic.

        Kept as a ``pytest.raises`` test: dynamic shape error paths do not fit
        the Before/Expected pattern.
        """
        span = ir.Span.unknown()

        dynamic_len = ir.Var("dynamic_len", ir.ScalarType(ir.DataType.INDEX), span)
        dynamic_tile_type = ir.TileType(
            [ir.ConstInt(1, ir.DataType.INDEX, span), dynamic_len],
            ir.DataType.FP32,
            memory_space=MemorySpace.Vec,
        )
        dynamic_tile = ir.Var("dynamic_tile", dynamic_tile_type, span)

        tpop_call = ir.Call(ir.Op("tile.tpop_from_aic"), [], {"split": 0}, dynamic_tile_type, span)
        body = ir.SeqStmts(
            [ir.AssignStmt(dynamic_tile, tpop_call, span), ir.ReturnStmt([dynamic_tile], span)], span
        )
        func = ir.Function("test_func", [], [dynamic_tile_type], body, span)
        program = ir.Program([func], "test_program", span)

        with pytest.raises(Exception, match="InitMemRef requires static shape"):
            passes.init_mem_ref()(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
