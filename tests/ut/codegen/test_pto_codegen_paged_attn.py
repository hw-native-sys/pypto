# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for PTO backend codegen for paged attention operations."""

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen


@pl.program
class PagedAttention:
    """
    Case1:
    batch 256
    num_heads 16
    kv_head_num 1
    head_dim 128
    block_size 128
    max_num_blocks_per_req 256
    scale_value 1
    Q(256, 16, 128) BF16
    K(16384, 128, 1, 128) BF16
    V(16384, 128, 1, 128) BF16
    block_table(256, 256) INT32
    context_lens(256, ) INT32
    out(524288, ) FP32
    """

    """
    orchestration config

    q_tile_size 16

    num_head_tiles 1
    sij_size 16 * 128 float
    pij_size 16 * 128 uint16
    mij_size 16 float
    lij_size 16 float
    oi_new_size 16 * 128 float

    mi_size 16 float
    li_size 16 float
    oi_size 16 * 128 float

    qi(256, 16, 128) BF16
    out()
    """

    # M K N 16 128 128

    # AIC kernels

    @pl.function(type=pl.FunctionType.InCore)
    def qk_matmul(
        self,
        qi: pl.Tensor[[16, 128], pl.BF16],
        kj: pl.Tensor[[128, 128], pl.BF16, pl.DN],
        s_ij: pl.Tensor[[16, 128], pl.FP32],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        q_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(qi, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
        k_tile_T: pl.Tile[[128, 128], pl.BF16] = pl.load(
            kj, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat, transpose=True
        )
        s_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.matmul(q_tile, k_tile_T)
        updated_sij: pl.Tensor[[16, 128], pl.FP32] = pl.store(s_tile, [0, 0], s_ij)
        return updated_sij

    @pl.function(type=pl.FunctionType.InCore)
    def pv_matmul(
        self,
        pij: pl.Tensor[[16, 128], pl.BF16],
        vj: pl.Tensor[[128, 128], pl.BF16],
        oij: pl.Tensor[[16, 128], pl.FP32],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        p_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(
            pij, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
        )
        v_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(
            vj, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
        )
        o_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.matmul(p_tile, v_tile)
        updated_oij: pl.Tensor[[16, 128], pl.FP32] = pl.store(o_tile, [0, 0], oij)
        return updated_oij

    # AIV kernels

    @pl.function(type=pl.FunctionType.InCore)
    def softmax_prepare(
        self,
        sij: pl.Tensor[[16, 128], pl.FP32],
        pij: pl.Tensor[[16, 128], pl.BF16],
        mij: pl.Tensor[[16, 1], pl.FP32],
        lij: pl.Tensor[[16, 1], pl.FP32],
        scale_value: pl.Scalar[pl.FP32],
    ):
        sij_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(sij, [0, 0], [16, 128])
        # sij_dyn_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(
        #     sij, [0, 0], [16, 128]
        # )
        # TODO: <TileType::Vec, float, M, N, BLayout::RowMajor, M, -1
        # sij_pad_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(
        #     sij, [0, 0], [16, 128]
        # )
        # TODO: <TileType::Vec, float, M, N, BLayout::RowMajor, M, N, SLayout::NoneBox, 512, PadValue::Min>
        pij_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(pij, [0, 0], [16, 128])
        tmp_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.sub(sij_tile, sij_tile)
        sij_tile = pl.tile.fillpad(sij_tile)
        sij_tile = pl.tile.muls(sij_tile, scale_value)
        max_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.row_max(sij_tile, tmp_tile)
        pij_tile = pl.tile.row_expand_sub(sij_tile, max_tile)
        pij_tile = pl.tile.exp(pij_tile)
        pij_bf16_tile = pl.tile.cast(pij_tile, mode="round", target_type=pl.BF16)
        pij_tile = pl.tile.cast(pij_bf16_tile, mode="round", target_type=pl.FP16)
        sum_tile: pl.Tile[[16, 1], pl.FP16] = pl.tile.row_sum(pij_tile, tmp_tile)
        pl.store(max_tile, [0, 0], mij)
        pl.store(sum_tile, [0, 0], lij)
        pl.store(pij_bf16_tile, [0, 0], pij)
        return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

    @pl.function(type=pl.FunctionType.InCore)
    def online_update(
        self,
        mij: pl.Tensor[[16, 1], pl.FP32],
        lij: pl.Tensor[[16, 1], pl.FP32],
        oi_new: pl.Tensor[[16, 128], pl.FP32],
        mi: pl.Tensor[[16, 1], pl.FP32],
        li: pl.Tensor[[16, 1], pl.FP32],
        oi: pl.Tensor[[16, 128], pl.FP32],
        dst: pl.Tensor[[16, 128], pl.FP32],
    ):
        oi_new_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(oi_new, [0, 0], [16, 128])
        oi_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(oi, [0, 0], [16, 128])
        mij_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(mij, [0, 0], [16, 1])
        lij_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(lij, [0, 0], [16, 1])
        mi_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(mi, [0, 0], [16, 1])
        li_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(li, [0, 0], [16, 1])

        mi_new_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.maximum(mi_tile, mij_tile)

        alpha_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.sub(mi_tile, mi_new_tile)
        alpha_tile = pl.tile.exp(alpha_tile)

        beta_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.sub(mij_tile, mi_new_tile)
        beta_tile = pl.tile.exp(beta_tile)

        li_scaled: pl.Tile[[16, 1], pl.FP32] = pl.tile.mul(alpha_tile, li_tile)
        lij_scaled: pl.Tile[[16, 1], pl.FP32] = pl.tile.mul(beta_tile, lij_tile)
        li_new_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.add(li_scaled, lij_scaled)

        oi_scaled: pl.Tile[[16, 128], pl.FP32] = pl.tile.row_expand_mul(oi_tile, alpha_tile)
        oi_new_scaled: pl.Tile[[16, 128], pl.FP32] = pl.tile.row_expand_mul(oi_new_tile, beta_tile)
        oi_updated_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.add(oi_scaled, oi_new_scaled)

        dst_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.row_expand_div(oi_updated_tile, li_new_tile)

        pl.store(mi_new_tile, [0, 0], mi)
        pl.store(li_new_tile, [0, 0], li)
        pl.store(oi_updated_tile, [0, 0], oi)
        pl.store(dst_tile, [0, 0], dst)
        return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement


def test_tile_ops_codegen():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    program = PagedAttention
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    optimized_program = pm.run_passes(program)
    codegen_instance = codegen.PTOCodegen()

    for func in optimized_program.functions.values():
        func_name = func.name
        single_func_program = ir.Program([func], func_name, optimized_program.span)
        mlir_code = codegen_instance.generate(single_func_program)
        assert mlir_code, f"Generated MLIR code for {func_name} should not be empty"


if __name__ == "__main__":
    pytest.main([__file__])
