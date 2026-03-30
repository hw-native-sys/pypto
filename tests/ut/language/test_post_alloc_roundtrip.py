# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Simplified softmax-rescale DSL program exercising MemRefType, Tuple, and Scalar ==.

This is a stripped-down version of the SoftmaxRescaleProgram pass dump that
hits the three constructs whose pyright typing was fixed:

  1. pl.MemRefType / pl.tile.alloc  — printed by AllocateMemoryAddr pass
  2. pl.Tuple[...] with indexing     — multi-return incore functions
  3. Scalar == int in if-condition   — branch on loop index
"""

import pypto.language as pl
import pytest


class TestSoftmaxRescaleDSL:
    """Simplified softmax-rescale written directly in DSL form."""

    def test_softmax_rescale_incore_init(self):
        """Incore function that allocates tiles and returns a Tuple."""

        @pl.program
        class SoftmaxInit:
            @pl.function(type=pl.FunctionType.AIV)
            def init(
                self,
                ret0_out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
                ret1_out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 128], pl.FP32]]:
                li: pl.Tile[[1, 16], pl.FP32] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
                oi: pl.Tile[[16, 128], pl.FP32] = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                r0: pl.Tensor[[16, 1], pl.FP32] = pl.tile.store(
                    pl.tile.reshape(li, [16, 1]), [0, 0], ret0_out
                )
                r1: pl.Tensor[[16, 128], pl.FP32] = pl.tile.store(oi, [0, 0], ret1_out)
                return r0, r1

        assert SoftmaxInit is not None
        code = SoftmaxInit.as_python()
        assert "tile.full" in code
        assert "tile.store" in code

    def test_softmax_rescale_branch_on_index(self):
        """Incore function that branches on Scalar == 0 (if idx == 0)."""

        @pl.program
        class SoftmaxBranch:
            @pl.function(type=pl.FunctionType.AIV)
            def body(
                self,
                cur_li: pl.Tensor[[16, 1], pl.FP32],
                li_prev: pl.Tensor[[16, 1], pl.FP32],
                oi_prev: pl.Tensor[[16, 128], pl.FP32],
                oi_tmp: pl.Tensor[[16, 128], pl.FP32],
                idx: pl.Scalar[pl.INDEX],
                ret0_out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
                ret1_out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 128], pl.FP32]]:
                cur_li_t: pl.Tile[[16, 1], pl.FP32] = pl.tile.load(cur_li, [0, 0], [16, 1])
                li_t: pl.Tile[[16, 1], pl.FP32] = pl.tile.load(li_prev, [0, 0], [16, 1])
                oi_t: pl.Tile[[16, 128], pl.FP32] = pl.tile.load(oi_prev, [0, 0], [16, 128])
                oi_tmp_t: pl.Tile[[16, 128], pl.FP32] = pl.tile.load(oi_tmp, [0, 0], [16, 128])
                # Fix #5: Scalar == int in if-condition
                if idx == 0:
                    oi_out: pl.Tile[[16, 128], pl.FP32] = oi_tmp_t
                    li_out: pl.Tile[[16, 1], pl.FP32] = cur_li_t
                    li_phi, oi_phi = pl.yield_(li_out, oi_out)
                else:
                    li_new: pl.Tile[[16, 1], pl.FP32] = pl.tile.add(li_t, cur_li_t)
                    oi_new: pl.Tile[[16, 128], pl.FP32] = pl.tile.add(oi_t, oi_tmp_t)
                    li_phi, oi_phi = pl.yield_(li_new, oi_new)
                r0: pl.Tensor[[16, 1], pl.FP32] = pl.tile.store(li_phi, [0, 0], ret0_out)
                r1: pl.Tensor[[16, 128], pl.FP32] = pl.tile.store(oi_phi, [0, 0], ret1_out)
                return r0, r1

        assert SoftmaxBranch is not None
        code = SoftmaxBranch.as_python()
        assert "idx == 0" in code

    def test_softmax_rescale_orchestration_with_tuple_return(self):
        """Orchestration that calls incore and indexes the Tuple result."""

        @pl.program
        class SoftmaxOrch:
            @pl.function(type=pl.FunctionType.AIV)
            def init(
                self,
                x: pl.Tensor[[64, 128], pl.FP32],
                ret0_out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
                ret1_out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 128], pl.FP32]]:
                li: pl.Tile[[1, 16], pl.FP32] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
                oi: pl.Tile[[16, 128], pl.FP32] = pl.tile.load(x, [0, 0], [16, 128])
                r0: pl.Tensor[[16, 1], pl.FP32] = pl.tile.store(
                    pl.tile.reshape(li, [16, 1]), [0, 0], ret0_out
                )
                r1: pl.Tensor[[16, 128], pl.FP32] = pl.tile.store(oi, [0, 0], ret1_out)
                return r0, r1

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                input: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                ret0_out: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
                ret1_out: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32)
                # Fix #3: Tuple return, then index with [0] / [1]
                ret: pl.Tuple[
                    pl.Tensor[[16, 1], pl.FP32],
                    pl.Tensor[[16, 128], pl.FP32],
                ] = self.init(input, ret0_out, ret1_out)
                oi: pl.Tensor[[16, 128], pl.FP32] = ret[1]
                return oi

        assert SoftmaxOrch is not None
        assert len(SoftmaxOrch.functions) == 2

    def test_alloc_memref_directly_in_dsl(self):
        """pl.tile.alloc and pl.MemRefType used directly, as the pass dump emits them."""

        @pl.program
        class AllocProg:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                x: pl.Tensor[[16, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                # Fix #1: exactly what the printer emits after AllocateMemoryAddr
                mem_vec_0: pl.MemRefType = pl.tile.alloc(pl.Mem.Vec, 0, 8192, 0)
                mem_vec_1: pl.MemRefType = pl.tile.alloc(pl.Mem.Vec, 8192, 8192, 1)
                # mem_vec_* referenced in Tile annotations just like the real dump
                t: pl.Tile[[16, 128], pl.FP32, mem_vec_0, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [16, 128])
                t2: pl.Tile[[16, 128], pl.FP32, mem_vec_1, pl.Mem.Vec] = pl.tile.add(t, t)
                r: pl.Tensor[[16, 128], pl.FP32] = pl.tile.store(t2, [0, 0], out)
                return r

        assert AllocProg is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
