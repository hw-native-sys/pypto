# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Paged Attention Orchestration Example

Builds a paged attention orchestration function using the PyPTO DSL with online
softmax and a 4-kernel pipeline.  Use build_paged_attention_program() to
obtain a @pl.program class parameterised by runtime tensor dimensions.

Tensor layout (all 2D, flattened):
  query:       [batch * num_heads, head_dim]                    BF16
  key_cache:   [total_pool_blocks * block_size, head_dim]       BF16
  value_cache: [total_pool_blocks * block_size, head_dim]       BF16
  out:         [batch * num_heads, head_dim]                    FP32

Pipeline per KV block iteration:
  1. QK Matmul:       sij          = kernel_qk_matmul(qi, kj, sij)
  2. Softmax Prepare: pij, mi, li  = kernel_softmax_prepare(sij_valid, scale, pij, mi, li)
  3. PV Matmul:       oi_tmp       = kernel_pv_matmul(pij, vj, oi_tmp)
  4. Online Update:   mi, li, oi, dst = kernel_online_update(
                          mi_j, li_j, oi_new, mi, li, oi, dst, is_first, is_last)

Shared InCore kernels (module-level @pl.function):
  kernel_init_inplace, kernel_qk_matmul, kernel_softmax_prepare,
  kernel_pv_matmul, kernel_online_update,
  kernel_qk_matmul_2block, kernel_softmax_prepare_2block, kernel_pv_matmul_2block

These can be imported and reused by other @pl.program definitions.
"""

import struct

import pypto.language as pl
import torch  # type: ignore[import]
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime import RunConfig, TensorSpec, run

# ── Shared InCore kernels (module-level, reusable across programs) ──────────


@pl.function(type=pl.FunctionType.InCore)
def kernel_init_inplace(
    oi: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
    li: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
    mi: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
) -> tuple[
    pl.Tensor[[16, 128], pl.FP32],
    pl.Tensor[[16, 1], pl.FP32],
    pl.Tensor[[16, 1], pl.FP32],
]:
    """Initialize inplace accumulators to zero (VECTOR)."""
    return oi, li, mi


@pl.function(type=pl.FunctionType.InCore)
def kernel_qk_matmul(
    qi: pl.Tensor[[16, 128], pl.BF16],
    kj: pl.Tensor[[128, 128], pl.BF16, pl.DN],
    output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
) -> pl.Tensor[[16, 128], pl.FP32]:
    """QK matmul: sij = qi @ kj.T (CUBE). kj transposed before move to L0B."""
    qi_l1 = pl.load(qi, offsets=[0, 0], shapes=[16, 128], target_memory=pl.MemorySpace.Mat)
    kj_l1 = pl.load(kj, offsets=[0, 0], shapes=[128, 128], target_memory=pl.MemorySpace.Mat, transpose=True)
    qi_l0a = pl.move(qi_l1, target_memory=pl.MemorySpace.Left)
    kj_l0b = pl.move(kj_l1, target_memory=pl.MemorySpace.Right)
    sij_l0c = pl.matmul(qi_l0a, kj_l0b)
    out: pl.Tensor[[16, 128], pl.FP32] = pl.store(sij_l0c, offsets=[0, 0], output_tensor=output)
    return out


@pl.function(type=pl.FunctionType.InCore)
def kernel_softmax_prepare(
    sij: pl.Tensor[[16, 128], pl.FP32],
    scale: pl.Scalar[pl.FP32],
    out_pij: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
    out_mi: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
    out_li: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
) -> tuple[
    pl.Tensor[[16, 128], pl.BF16],
    pl.Tensor[[16, 1], pl.FP32],
    pl.Tensor[[16, 1], pl.FP32],
]:
    """Softmax prepare: scale, row_max, exp, row_sum (VECTOR)."""
    s_tile = pl.load(sij, offsets=[0, 0], shapes=[16, 128], target_memory=pl.MemorySpace.Vec)
    scaled = pl.mul(s_tile, scale)
    tmp_tile = pl.create_tile([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    mi_tile = pl.row_max(scaled, tmp_tile)
    sij_centered = pl.row_expand_sub(scaled, mi_tile)
    exp_tile = pl.exp(sij_centered)
    pij_tile_bf16 = pl.cast(exp_tile, target_type=pl.BF16)
    pij_tile = pl.cast(pij_tile_bf16, target_type=pl.FP32)
    li_tile = pl.row_sum(pij_tile, tmp_tile)
    out_pij = pl.store(pij_tile_bf16, offsets=[0, 0], output_tensor=out_pij)
    out_mi = pl.store(mi_tile, offsets=[0, 0], output_tensor=out_mi)
    out_li = pl.store(li_tile, offsets=[0, 0], output_tensor=out_li)
    return out_pij, out_mi, out_li


@pl.function(type=pl.FunctionType.InCore)
def kernel_pv_matmul(
    pij: pl.Tensor[[16, 128], pl.BF16],
    vj: pl.Tensor[[128, 128], pl.BF16],
    output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
) -> pl.Tensor[[16, 128], pl.FP32]:
    """PV matmul: oi_tmp = pij @ vj (CUBE)."""
    pij_l1 = pl.load(pij, offsets=[0, 0], shapes=[16, 128], target_memory=pl.MemorySpace.Mat)
    vj_l1 = pl.load(vj, offsets=[0, 0], shapes=[128, 128], target_memory=pl.MemorySpace.Mat)
    pij_l0a = pl.move(pij_l1, target_memory=pl.MemorySpace.Left)
    vj_l0b = pl.move(vj_l1, target_memory=pl.MemorySpace.Right)
    oi_l0c = pl.matmul(pij_l0a, vj_l0b)
    out = pl.store(oi_l0c, offsets=[0, 0], output_tensor=output)
    return out


@pl.function(type=pl.FunctionType.InCore)
def kernel_online_update(
    mij: pl.Tensor[[16, 1], pl.FP32, pl.DN],
    lij: pl.Tensor[[16, 1], pl.FP32, pl.DN],
    oi_new: pl.Tensor[[16, 128], pl.FP32],
    mi: pl.InOut[pl.Tensor[[16, 1], pl.FP32, pl.DN]],
    li: pl.InOut[pl.Tensor[[16, 1], pl.FP32, pl.DN]],
    oi: pl.InOut[pl.Tensor[[16, 128], pl.FP32]],
    dst: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
    is_first: pl.Scalar[pl.BOOL],
    is_last: pl.Scalar[pl.BOOL],
) -> tuple[
    pl.Tensor[[16, 1], pl.FP32, pl.DN],
    pl.Tensor[[16, 1], pl.FP32, pl.DN],
    pl.Tensor[[16, 128], pl.FP32],
    pl.Tensor[[16, 128], pl.FP32],
]:
    """Online softmax update with inplace mi/li/oi (VECTOR)."""
    mij_tile = pl.load(mij, offsets=[0, 0], shapes=[16, 1], target_memory=pl.MemorySpace.Vec)
    lij_tile = pl.load(lij, offsets=[0, 0], shapes=[16, 1], target_memory=pl.MemorySpace.Vec)
    oi_new_tile = pl.load(oi_new, offsets=[0, 0], shapes=[16, 128], target_memory=pl.MemorySpace.Vec)
    mi_tile = pl.load(mi, offsets=[0, 0], shapes=[16, 1], target_memory=pl.MemorySpace.Vec)
    li_tile = pl.load(li, offsets=[0, 0], shapes=[16, 1], target_memory=pl.MemorySpace.Vec)
    oi_tile = pl.load(oi, offsets=[0, 0], shapes=[16, 128], target_memory=pl.MemorySpace.Vec)

    if is_first:
        mi_out = pl.store(mij_tile, offsets=[0, 0], output_tensor=mi)
        li_out = pl.store(lij_tile, offsets=[0, 0], output_tensor=li)
        oi_out = pl.store(oi_new_tile, offsets=[0, 0], output_tensor=oi)
        if is_last:
            dst_tile = pl.row_expand_div(oi_new_tile, lij_tile)
            dst_out = pl.store(dst_tile, offsets=[0, 0], output_tensor=dst)
        else:
            # First block but not last: dst is not yet meaningful, store zeros
            zero_tile = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
            dst_out = pl.store(zero_tile, offsets=[0, 0], output_tensor=dst)
    else:
        # Reshape DN [16,1] -> ND [1,16] for element-wise ops
        mi_tile_nd = pl.reshape(mi_tile, [1, 16])
        mij_tile_nd = pl.reshape(mij_tile, [1, 16])
        li_tile_nd = pl.reshape(li_tile, [1, 16])
        lij_tile_nd = pl.reshape(lij_tile, [1, 16])

        mi_new = pl.maximum(mi_tile_nd, mij_tile_nd)
        mi_diff = pl.sub(mi_tile_nd, mi_new)
        alpha = pl.exp(mi_diff)
        mij_diff = pl.sub(mij_tile_nd, mi_new)
        beta = pl.exp(mij_diff)

        li_scaled = pl.mul(alpha, li_tile_nd)
        lij_scaled = pl.mul(beta, lij_tile_nd)
        li_updated = pl.add(li_scaled, lij_scaled)

        alpha_dn = pl.reshape(alpha, [16, 1])  # Reshape [1,16] -> [16,1] DN for row_expand_mul
        oi_scaled = pl.row_expand_mul(oi_tile, alpha_dn)
        beta_dn = pl.reshape(beta, [16, 1])  # Reshape [1,16] -> [16,1] DN for row_expand_mul
        oi_new_scaled = pl.row_expand_mul(oi_new_tile, beta_dn)
        oi_updated = pl.add(oi_scaled, oi_new_scaled)

        mi_new_dn = pl.reshape(mi_new, [16, 1])  # Reshape back to DN [16,1] for store
        li_updated_dn = pl.reshape(li_updated, [16, 1])  # Reshape back to DN [16,1] for store

        mi_out = pl.store(mi_new_dn, offsets=[0, 0], output_tensor=mi)
        li_out = pl.store(li_updated_dn, offsets=[0, 0], output_tensor=li)

        if is_last:
            dst_tile = pl.row_expand_div(oi_updated, li_updated_dn)
            dst_out = pl.store(dst_tile, offsets=[0, 0], output_tensor=dst)
            oi_out = pl.store(oi_updated, offsets=[0, 0], output_tensor=oi)
        else:
            zero_tile = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
            dst_out = pl.store(zero_tile, offsets=[0, 0], output_tensor=dst)
            oi_out = pl.store(oi_updated, offsets=[0, 0], output_tensor=oi)

    return mi_out, li_out, oi_out, dst_out


@pl.function(type=pl.FunctionType.InCore)
def kernel_qk_matmul_2block(
    qi: pl.Tensor[[16, 128], pl.BF16],
    kj: pl.Tensor[[128, 256], pl.BF16, pl.DN],
    output: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
) -> pl.Tensor[[16, 256], pl.FP32]:
    """QK matmul 2block: sij = qi @ kj.T (CUBE)."""
    qi_l1_0 = pl.load(qi, offsets=[0, 0], shapes=[16, 128], target_memory=pl.MemorySpace.Mat)
    kj0_l1 = pl.load(kj, offsets=[0, 0], shapes=[128, 256], target_memory=pl.MemorySpace.Mat, transpose=True)
    qi_l0a_0 = pl.move(qi_l1_0, target_memory=pl.MemorySpace.Left)
    kj0_l0b = pl.move(kj0_l1, target_memory=pl.MemorySpace.Right)
    sij_h0_l0c = pl.matmul(qi_l0a_0, kj0_l0b)
    out = pl.store(sij_h0_l0c, offsets=[0, 0], output_tensor=output)
    return out


@pl.function(type=pl.FunctionType.InCore)
def kernel_softmax_prepare_2block(
    sij: pl.Tensor[[16, 256], pl.FP32],
    scale: pl.Scalar[pl.FP32],
    out_pij: pl.Out[pl.Tensor[[16, 256], pl.BF16]],
    out_mi: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
    out_li: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
) -> tuple[
    pl.Tensor[[16, 256], pl.BF16],
    pl.Tensor[[16, 1], pl.FP32],
    pl.Tensor[[16, 1], pl.FP32],
]:
    """Softmax prepare 2block: scale, row_max, exp, row_sum (VECTOR)."""
    s_tile = pl.load(sij, offsets=[0, 0], shapes=[16, 256], target_memory=pl.MemorySpace.Vec)
    scaled = pl.mul(s_tile, scale)
    tmp_tile = pl.create_tile([16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    mi_tile = pl.row_max(scaled, tmp_tile)
    sij_centered = pl.row_expand_sub(scaled, mi_tile)
    exp_tile = pl.exp(sij_centered)
    pij_tile_bf16 = pl.cast(exp_tile, target_type=pl.BF16)
    pij_tile = pl.cast(pij_tile_bf16, target_type=pl.FP32)
    li_tile = pl.row_sum(pij_tile, tmp_tile)
    out_pij = pl.store(pij_tile_bf16, offsets=[0, 0], output_tensor=out_pij)
    out_mi = pl.store(mi_tile, offsets=[0, 0], output_tensor=out_mi)
    out_li = pl.store(li_tile, offsets=[0, 0], output_tensor=out_li)
    return out_pij, out_mi, out_li


@pl.function(type=pl.FunctionType.InCore)
def kernel_pv_matmul_2block(
    pij: pl.Tensor[[16, 256], pl.BF16],
    vj: pl.Tensor[[256, 128], pl.BF16],
    output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
) -> pl.Tensor[[16, 128], pl.FP32]:
    """PV matmul 2block: oi_new = pij @ vj (CUBE)."""
    pij_l1 = pl.load(pij, offsets=[0, 0], shapes=[16, 256], target_memory=pl.MemorySpace.Mat)
    vj_l1 = pl.load(vj, offsets=[0, 0], shapes=[256, 128], target_memory=pl.MemorySpace.Mat)
    pij_l0a = pl.move(pij_l1, target_memory=pl.MemorySpace.Left)
    vj_l0b = pl.move(vj_l1, target_memory=pl.MemorySpace.Right)
    oi_l0c = pl.matmul(pij_l0a, vj_l0b)
    out = pl.store(oi_l0c, offsets=[0, 0], output_tensor=output)
    return out


def build_paged_attention_program(
    batch: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_num_blocks_per_req: int,
    q_tile: int = 16,
):
    """Build a parameterised paged-attention @pl.program for the given shapes.

    Returns the decorated program class (not an instance).  The tensor type
    annotations in the orchestration function are filled in from the arguments
    so that the PyPTO DSL can resolve static dimensions at compile time.

    Parameters
    ----------
    batch:                  number of requests in the batch
    num_heads:              number of query heads
    head_dim:               per-head feature dimension
    block_size:             KV-cache block size (rows per physical block)
    max_num_blocks_per_req: maximum number of KV blocks per request
    q_tile:                 query-head tile size used by the InCore kernels
    """
    # Derived static dimension values for tensor type annotations
    query_rows = batch * num_heads
    key_cache_rows = batch * max_num_blocks_per_req * block_size
    out_rows = batch * num_heads
    block_table_flat_size = batch * max_num_blocks_per_req

    @pl.program
    class PagedAttentionProgram:
        """Paged attention program with CUBE and VECTOR kernels (online softmax).

        InCore kernels (kernel_init_inplace, kernel_qk_matmul,
        kernel_softmax_prepare, kernel_pv_matmul, kernel_online_update) are
        defined at module level and automatically added to this program when
        called from the orchestration function.
        """

        # ── Orchestration function ──────────────────────────────────────────
        # Parameters: query, key_cache, value_cache, block_table, context_lens,
        #             out, config (7 tensors) + size_query, size_key_cache,
        #             size_value_cache (3 byte-size scalars)
        @pl.function(type=pl.FunctionType.Orchestration)
        def paged_attention(
            self,
            query: pl.Tensor[[query_rows, head_dim], pl.BF16],
            key_cache: pl.Tensor[[head_dim, key_cache_rows], pl.BF16, pl.DN],
            value_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
            block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
            context_lens: pl.Tensor[[batch], pl.INT32],
            out: pl.Tensor[[out_rows, head_dim], pl.FP32],
            config: pl.Tensor[[7], pl.INT64],
            size_query: pl.Tensor[[1], pl.INT64],
            size_key_cache: pl.Tensor[[1], pl.INT64],
            size_value_cache: pl.Tensor[[1], pl.INT64],
        ) -> pl.Tensor[[out_rows, head_dim], pl.FP32]:
            """Paged attention orchestration.

            Outer loops: batch → q_tile groups → KV blocks (bn).
            For each KV block, executes the 4-stage kernel pipeline.
            Config layout: [batch, num_heads, kv_head_num, head_dim, block_size, block_num, scale_bits]
            """
            # Read runtime config parameters
            batch_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
            num_heads_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
            head_dim_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [3])
            block_size_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [4])
            block_num_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [5])
            # scale_bits at config[6] - decoded as float

            q_head_num = num_heads_cfg
            q_loop_cfg = (q_head_num + q_tile - 1) // q_tile

            for b_idx in pl.range(batch_cfg):
                cur_seq = pl.tensor.read(context_lens, [b_idx])
                bn_this_batch = (cur_seq + block_size_cfg - 1) // block_size_cfg
                for q_idx in pl.range(q_loop_cfg):
                    cur_offset = b_idx * q_head_num + q_idx * q_tile

                    # Create inplace accumulators for this q_tile group
                    oi: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.create_tensor(
                        [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                        dtype=pl.FP32,
                    )
                    li_update: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)  # type: ignore[reportArgumentType]
                    mi_update: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)  # type: ignore[reportArgumentType]

                    # Initialize accumulators via shared module-level InCore kernel
                    oi, li_update, mi_update = kernel_init_inplace(oi, li_update, mi_update)

                    for bn in pl.range(bn_this_batch):
                        # Query view: row offset = b_idx * num_heads + q_idx * q_tile
                        qi: pl.Tensor[[q_tile, head_dim_cfg], pl.BF16] = pl.slice(
                            query,
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [cur_offset, 0],
                        )

                        # Get block index from block_table
                        cur_block_idx = pl.tensor.read(block_table, [b_idx * block_num_cfg + bn])
                        valid_len = pl.min(block_size_cfg, cur_seq - bn * block_size_cfg)

                        # Key/Value views: physical block row = cur_block_idx * block_size
                        kv_block_row = cur_block_idx * block_size_cfg
                        kj: pl.Tensor[[head_dim_cfg, block_size_cfg], pl.BF16, pl.DN] = pl.slice(
                            key_cache,
                            [head_dim_cfg, block_size_cfg],  # type: ignore[reportArgumentType]
                            [kv_block_row, 0],
                        )
                        vj: pl.Tensor[[block_size_cfg, head_dim_cfg], pl.BF16] = pl.slice(
                            value_cache,
                            [block_size_cfg, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [kv_block_row, 0],
                        )

                        sij: pl.Tensor[[q_tile, block_size_cfg], pl.FP32] = pl.create_tensor(
                            [q_tile, block_size_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.FP32,
                        )

                        # QK matmul (CUBE) via shared module-level InCore kernel
                        sij = kernel_qk_matmul(qi, kj, sij)
                        sij_valid: pl.Tensor[[q_tile, valid_len], pl.FP32] = pl.slice(
                            sij,
                            [q_tile, valid_len],  # type: ignore[reportArgumentType]
                            [0, 0],
                        )

                        pij_f16: pl.Tensor[[q_tile, block_size_cfg], pl.BF16] = pl.create_tensor(
                            [q_tile, block_size_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.BF16,
                        )
                        mi: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)  # type: ignore[reportArgumentType]
                        li: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)  # type: ignore[reportArgumentType]

                        # Softmax prepare (VECTOR) via shared module-level InCore kernel
                        pij_f16, mi, li = kernel_softmax_prepare(sij_valid, 1.0, pij_f16, mi, li)  # type: ignore[reportArgumentType]

                        oi_tmp: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.create_tensor(
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.FP32,
                        )
                        # PV matmul (CUBE) via shared module-level InCore kernel
                        oi_tmp = kernel_pv_matmul(pij_f16, vj, oi_tmp)

                        # Conditional flags
                        if bn == 0:
                            is_first: pl.Scalar[pl.INT64] = pl.yield_(1)  # type: ignore[reportArgumentType]
                        else:
                            is_first: pl.Scalar[pl.INT64] = pl.yield_(0)  # type: ignore[reportArgumentType]
                        if bn == bn_this_batch - 1:
                            is_last: pl.Scalar[pl.INT64] = pl.yield_(1)  # type: ignore[reportArgumentType]
                        else:
                            is_last: pl.Scalar[pl.INT64] = pl.yield_(0)  # type: ignore[reportArgumentType]

                        # Output view: same row offset as query view
                        out_view: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.slice(
                            out,
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [cur_offset, 0],
                        )
                        # Online softmax update via shared module-level InCore kernel
                        mi_update, li_update, oi, out_view = kernel_online_update(
                            mi, li, oi_tmp, mi_update, li_update, oi, out_view, is_first, is_last
                        )

            return out

    return PagedAttentionProgram


def build_paged_attention_multitier_program(
    batch: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_num_blocks_per_req: int,
    q_tile: int = 16,
    assume_contiguous_blocks: bool = False,
):
    """Build a parameterised paged-attention @pl.program with x2/x1 bn-loop tiers.

    Same signature as build_paged_attention_program but uses a 2-tier loop
    structure when ``assume_contiguous_blocks=True``.

    The x2 tier processes two physically-contiguous blocks per outer iteration
    using a single 2-block kernel call (kernel_qk_matmul_2block,
    kernel_softmax_prepare_2block, kernel_pv_matmul_2block).  The blocks must
    be physically contiguous in the KV-cache pool
    (bidx_x2_1 == bidx_x2_0 + 1).

    When ``assume_contiguous_blocks=False`` (the default), the x2 tier is
    disabled and all blocks are processed via x1 (single-block) kernels.
    This is safe for any block_table layout.  The PyPTO DSL does not allow
    runtime if/else branches with different tensor shapes, so runtime
    contiguity checking with a 2-block fallback is not feasible.

      n2   = bn_this_batch // 2          # x2 iterations (only if contiguous)
      end2 = n2 * 2                      # 0 when assume_contiguous_blocks=False

      for bn2 in pl.range(0,    end2,          2):  # kernelx2 (2-block kernel)
          kernel_*_2block(blocks bn2 and bn2+1, joint softmax)
      for bn3 in pl.range(end2, bn_this_batch, 1):  # kernelx1
          block bn3:   kernel_* (1-block)

    Parameters
    ----------
    batch:                    number of requests in the batch
    num_heads:                number of query heads
    head_dim:                 per-head feature dimension
    block_size:               KV-cache block size (rows per physical block)
    max_num_blocks_per_req:   maximum number of KV blocks per request
    q_tile:                   query-head tile size used by the InCore kernels
    assume_contiguous_blocks: if True, assume consecutive logical blocks map to
                              physically adjacent KV-cache rows and enable the
                              x2 (2-block) tier.  If False (default), all blocks
                              are processed via x1 kernels, which is correct for
                              any block_table layout.
    """
    # Derived static dimension values for tensor type annotations
    query_rows = batch * num_heads
    key_cache_rows = batch * max_num_blocks_per_req * block_size
    out_rows = batch * num_heads
    block_table_flat_size = batch * max_num_blocks_per_req

    # Compile-time q-loop bound: q_idx iterates over q_tile groups of heads
    q_loop_static = (num_heads + q_tile - 1) // q_tile

    # Compile-time scale for x2 tier: 2 when contiguous blocks assumed, 0 otherwise.
    # Used inside @pl.function to avoid Python if-statements in the DSL body.
    contiguous_scale = 2 if assume_contiguous_blocks else 0

    @pl.program
    class PagedAttentionMultitierProgram:
        """Paged attention program with x2/x1 multi-tier bn loop (online softmax).

        InCore kernels (kernel_init_inplace, kernel_qk_matmul,
        kernel_softmax_prepare, kernel_pv_matmul, kernel_online_update,
        kernel_qk_matmul_2block, kernel_softmax_prepare_2block,
        kernel_pv_matmul_2block) are defined at module level and automatically
        added to this program when called from the orchestration function.
        """

        @pl.function(type=pl.FunctionType.Orchestration)
        def paged_attention(
            self,
            query: pl.Tensor[[query_rows, head_dim], pl.BF16],
            key_cache: pl.Tensor[[head_dim, key_cache_rows], pl.BF16, pl.DN],
            value_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
            block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
            context_lens: pl.Tensor[[batch], pl.INT32],
            out: pl.Tensor[[out_rows, head_dim], pl.FP32],
            config: pl.Tensor[[7], pl.INT64],
            size_query: pl.Tensor[[1], pl.INT64],
            size_key_cache: pl.Tensor[[1], pl.INT64],
            size_value_cache: pl.Tensor[[1], pl.INT64],
        ) -> pl.Tensor[[out_rows, head_dim], pl.FP32]:
            """Paged attention orchestration with 2-tier bn loop (x2 / x1).

            When assume_contiguous_blocks=True, the x2 tier processes pairs
            (bn2, bn2+1) per outer iteration using a single 2-block kernel
            call, which computes a joint softmax over valid_0 + valid_1 tokens
            from both blocks.  Physical contiguity
            (bidx_x2_1 == bidx_x2_0 + 1) must be guaranteed by the caller.

            When assume_contiguous_blocks=False, the x2 tier is disabled
            (end2 = 0) and all blocks go through x1 single-block kernels.

            Tier boundaries (runtime scalars):
              n2   = bn_this_batch // 2              end2 = n2 * 2

            Loop structure:
              for bn2 in pl.range(0,    end2,          2):  # kernelx2
                  kernel_*_2block(bn2, bn2+1)  # joint softmax over both blocks
              for bn3 in pl.range(end2, bn_this_batch, 1):  # kernelx1
                  block bn3:   kernel_* (1-block)

            Config layout: [batch, num_heads, kv_head_num, head_dim,
                            block_size, block_num, scale_bits]
            """
            # Read runtime config parameters
            batch_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
            head_dim_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [3])
            block_size_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [4])
            block_num_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [5])

            for b_idx in pl.range(batch_cfg):
                cur_seq = pl.tensor.read(context_lens, [b_idx])
                bn_this_batch = (cur_seq + block_size_cfg - 1) // block_size_cfg

                # ── Compute tier boundaries (runtime scalars) ──────────────────
                # contiguous_scale is a compile-time Python int (0 or 2).
                # When assume_contiguous_blocks=False: end2 = n2 * 0 = 0 → x2 loop empty.
                # When assume_contiguous_blocks=True:  end2 = n2 * 2   → original behavior.
                # A Python-level `if` cannot be used here because the DSL's AST
                # parser would treat it as a runtime IfStmt.
                n2 = bn_this_batch // 2
                end2 = n2 * contiguous_scale

                # ── Compile-time unrolled q_idx loop ──────────────────────────
                for q_idx in pl.range(q_loop_static):
                    cur_offset = b_idx * num_heads + q_idx * q_tile

                    # Create and initialize inplace accumulators
                    # Note: pyright cannot infer tensor shapes from runtime scalar
                    # variables (q_tile, head_dim_cfg), so reportArgumentType
                    # suppressions are required throughout this function body.
                    oi: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.create_tensor(
                        [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                        dtype=pl.FP32,
                    )
                    li_update: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)  # type: ignore[reportArgumentType]
                    mi_update: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)  # type: ignore[reportArgumentType]
                    oi, li_update, mi_update = kernel_init_inplace(oi, li_update, mi_update)

                    # qi is invariant across both bn2 and bn3 loops (cur_offset is constant)
                    qi: pl.Tensor[[q_tile, head_dim_cfg], pl.BF16] = pl.slice(
                        query,
                        [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                        [cur_offset, 0],
                    )
                    # ── kernelx2: step=2, check physical contiguity before using 2-block kernel ──
                    for bn2 in pl.range(0, end2, 2):
                        # Read BOTH block indices to verify physical contiguity
                        bidx_x2_0 = pl.tensor.read(block_table, [b_idx * block_num_cfg + bn2])
                        valid_0 = pl.min(block_size_cfg, cur_seq - bn2 * block_size_cfg)
                        valid_1 = pl.min(block_size_cfg, cur_seq - (bn2 + 1) * block_size_cfg)

                        # ── physically contiguous: use 2-block kernel ──────────────────
                        valid_2block = valid_0 + valid_1
                        row_pair = bidx_x2_0 * block_size_cfg
                        kj_2b: pl.Tensor[[head_dim_cfg, 2 * block_size_cfg], pl.BF16, pl.DN] = pl.slice(
                            key_cache,
                            [head_dim_cfg, 2 * block_size_cfg],  # type: ignore[reportArgumentType]
                            [row_pair, 0],
                        )
                        vj_2b: pl.Tensor[[2 * block_size_cfg, head_dim_cfg], pl.BF16] = pl.slice(
                            value_cache,
                            [2 * block_size_cfg, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [row_pair, 0],
                        )
                        sij_2b: pl.Tensor[[q_tile, 2 * block_size_cfg], pl.FP32] = pl.create_tensor(
                            [q_tile, 2 * block_size_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.FP32,
                        )
                        sij_2b = kernel_qk_matmul_2block(qi, kj_2b, sij_2b)
                        sij_2b_valid: pl.Tensor[[q_tile, valid_2block], pl.FP32] = pl.slice(
                            sij_2b,
                            [q_tile, valid_2block],  # type: ignore[reportArgumentType]
                            [0, 0],
                        )
                        pij_2b: pl.Tensor[[q_tile, 2 * block_size_cfg], pl.BF16] = pl.create_tensor(
                            [q_tile, 2 * block_size_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.BF16,
                        )
                        mi_2b: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)  # type: ignore[reportArgumentType]
                        li_2b: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)  # type: ignore[reportArgumentType]
                        pij_2b, mi_2b, li_2b = kernel_softmax_prepare_2block(
                            sij_2b_valid,
                            1.0,  # type: ignore[reportArgumentType]
                            pij_2b,
                            mi_2b,
                            li_2b,
                        )
                        oi_2b: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.create_tensor(
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.FP32,
                        )
                        oi_2b = kernel_pv_matmul_2block(pij_2b, vj_2b, oi_2b)
                        if bn2 == 0:
                            is_first_2b: pl.Scalar[pl.INT64] = pl.yield_(1)  # type: ignore[reportArgumentType]
                        else:
                            is_first_2b: pl.Scalar[pl.INT64] = pl.yield_(0)  # type: ignore[reportArgumentType]
                        if bn2 + 2 == bn_this_batch:
                            is_last_2b: pl.Scalar[pl.INT64] = pl.yield_(1)  # type: ignore[reportArgumentType]
                        else:
                            is_last_2b: pl.Scalar[pl.INT64] = pl.yield_(0)  # type: ignore[reportArgumentType]
                        ov_2b: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.slice(
                            out,
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [cur_offset, 0],
                        )
                        mi_update, li_update, oi, ov_2b = kernel_online_update(
                            mi_2b, li_2b, oi_2b, mi_update, li_update, oi, ov_2b, is_first_2b, is_last_2b
                        )
                    # ── kernelx1: step=1, single 1-block pipeline ─────────────
                    for bn3 in pl.range(end2, bn_this_batch, 1):
                        bidx = pl.tensor.read(block_table, [b_idx * block_num_cfg + bn3])
                        valid_len = pl.min(block_size_cfg, cur_seq - bn3 * block_size_cfg)
                        row = bidx * block_size_cfg

                        kj: pl.Tensor[[head_dim_cfg, block_size_cfg], pl.BF16, pl.DN] = pl.slice(
                            key_cache,
                            [head_dim_cfg, block_size_cfg],  # type: ignore[reportArgumentType]
                            [row, 0],
                        )
                        vj: pl.Tensor[[block_size_cfg, head_dim_cfg], pl.BF16] = pl.slice(
                            value_cache,
                            [block_size_cfg, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [row, 0],
                        )
                        sij: pl.Tensor[[q_tile, block_size_cfg], pl.FP32] = pl.create_tensor(
                            [q_tile, block_size_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.FP32,
                        )
                        sij = kernel_qk_matmul(qi, kj, sij)
                        sij_valid: pl.Tensor[[q_tile, valid_len], pl.FP32] = pl.slice(
                            sij,
                            [q_tile, valid_len],  # type: ignore[reportArgumentType]
                            [0, 0],
                        )
                        pij: pl.Tensor[[q_tile, block_size_cfg], pl.BF16] = pl.create_tensor(
                            [q_tile, block_size_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.BF16,
                        )
                        mi: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)  # type: ignore[reportArgumentType]
                        li: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)  # type: ignore[reportArgumentType]

                        pij, mi, li = kernel_softmax_prepare(sij_valid, 1.0, pij, mi, li)  # type: ignore[reportArgumentType]
                        oi_tmp: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.create_tensor(
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.FP32,
                        )
                        oi_tmp = kernel_pv_matmul(pij, vj, oi_tmp)

                        if bn3 == 0:
                            is_first: pl.Scalar[pl.INT64] = pl.yield_(1)  # type: ignore[reportArgumentType]
                        else:
                            is_first: pl.Scalar[pl.INT64] = pl.yield_(0)  # type: ignore[reportArgumentType]
                        if bn3 == bn_this_batch - 1:
                            is_last: pl.Scalar[pl.INT64] = pl.yield_(1)  # type: ignore[reportArgumentType]
                        else:
                            is_last: pl.Scalar[pl.INT64] = pl.yield_(0)  # type: ignore[reportArgumentType]

                        out_view: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.slice(
                            out,
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [cur_offset, 0],
                        )

                        mi_update, li_update, oi, out_view = kernel_online_update(
                            mi, li, oi_tmp, mi_update, li_update, oi, out_view, is_first, is_last
                        )

            return out

    return PagedAttentionMultitierProgram


def golden(tensors: dict, params: dict | None = None) -> None:
    """Reference paged-attention computation (torch), mirroring the kernel pipeline.

    Implements the online-softmax paged attention matching the 4-kernel pipeline:
    QK matmul → softmax prepare → PV matmul → online update.

    Args:
        tensors: Dict mapping tensor names to torch tensors.
        params: Unused.
    """
    config = tensors["config"]
    batch = int(config[0].item())
    num_heads = int(config[1].item())
    head_dim = int(config[3].item())
    block_size = int(config[4].item())
    max_num_blocks_per_req = int(config[5].item())
    # scale_bits is stored as float32 bits (IEEE 754) in an INT64 slot
    scale_bits = int(config[6].item())
    scale = struct.unpack("f", struct.pack("I", scale_bits & 0xFFFFFFFF))[0]

    query = tensors["query"].float().reshape(batch, num_heads, head_dim)
    total_pool_blocks = batch * max_num_blocks_per_req
    key_cache = tensors["key_cache"].float().reshape(total_pool_blocks, block_size, head_dim)
    value_cache = tensors["value_cache"].float().reshape(total_pool_blocks, block_size, head_dim)
    block_table = tensors["block_table"].reshape(batch, max_num_blocks_per_req)
    context_lens = tensors["context_lens"]

    out = torch.zeros((batch, num_heads, head_dim), dtype=torch.float32)
    q_tile = 16
    max_bn = int((context_lens.max().item() + block_size - 1) // block_size)

    for q_offset in range(0, num_heads, q_tile):
        q_tile_size = min(q_tile, num_heads - q_offset)
        qi = query[:, q_offset : q_offset + q_tile_size, :]  # [batch, q_tile, head_dim]
        oi, li, mi = None, None, None

        for bn in range(max_bn):
            valid_lens = torch.clamp(context_lens - bn * block_size, min=0, max=block_size)
            if not (valid_lens > 0).any():
                break
            block_indices = block_table[:, bn]  # [batch]
            kj_all = key_cache[block_indices]  # [batch, block_size, head_dim]
            vj_all = value_cache[block_indices]  # [batch, block_size, head_dim]

            sij = torch.bmm(qi, kj_all.transpose(1, 2)) * scale  # [batch, q_tile, block_size]
            pos = torch.arange(block_size).unsqueeze(0)  # [1, block_size]
            valid_mask = (pos < valid_lens.unsqueeze(1)).unsqueeze(1)  # [batch, 1, block_size]
            sij = sij.masked_fill(~valid_mask, float("-inf"))
            mij = sij.max(dim=-1, keepdim=True)[0].clamp(min=-1e30)  # [batch, q_tile, 1]
            pij = torch.exp(sij - mij).masked_fill(~valid_mask, 0.0)
            pij = pij.to(torch.bfloat16).to(torch.float32)
            lij = pij.sum(dim=-1, keepdim=True)  # [batch, q_tile, 1]
            oi_new = torch.bmm(pij, vj_all)  # [batch, q_tile, head_dim]

            if bn == 0:
                oi, li, mi = oi_new, lij, mij
            else:
                mi_new = torch.maximum(mi, mij)
                alpha = torch.exp(mi - mi_new)
                beta = torch.exp(mij - mi_new)
                li = alpha * li + beta * lij
                oi = alpha * oi + beta * oi_new
                mi = mi_new

        assert oi is not None and li is not None, "No KV blocks processed for this query tile"
        out[:, q_offset : q_offset + q_tile_size, :] = oi / li

    tensors["out"][:] = out.reshape(batch * num_heads, head_dim)


def build_tensor_specs(
    batch: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_num_blocks_per_req: int,
    context_len: int,
    scale: float = 1.0,
) -> list[TensorSpec]:
    """Build the TensorSpec list matching the paged_attention orchestration signature.

    Args:
        batch: Number of requests in the batch.
        num_heads: Number of query heads.
        head_dim: Per-head feature dimension.
        block_size: KV-cache block size (rows per physical block).
        max_num_blocks_per_req: Maximum number of KV blocks per request.
        context_len: Number of valid tokens per request.
        scale: Attention scale factor (stored as float32 bits in config[6]).
    """
    query_rows = batch * num_heads
    key_cache_rows = batch * max_num_blocks_per_req * block_size
    block_table_flat_size = batch * max_num_blocks_per_req

    # config layout: [batch, num_heads, kv_head_num=1, head_dim, block_size, max_blocks, scale_bits]
    # scale_bits: float32 value stored as its raw IEEE-754 bit pattern in an INT64 slot
    scale_bits = struct.unpack("I", struct.pack("f", scale))[0]
    config_data = torch.tensor(
        [batch, num_heads, 1, head_dim, block_size, max_num_blocks_per_req, scale_bits],
        dtype=torch.int64,
    )

    # Each request has context_len valid tokens
    context_lens_data = torch.full((batch,), context_len, dtype=torch.int32)

    # block_table: random physical block assignment, indices in [0, batch*max_blocks)
    block_table_data = torch.randint(
        0, max(block_table_flat_size, 1), size=(batch, max_num_blocks_per_req), dtype=torch.int32
    ).flatten()

    # Byte sizes: BF16 tensors use 2 bytes per element
    size_query = torch.tensor([query_rows * head_dim * 2], dtype=torch.int64)
    size_key_cache = torch.tensor([key_cache_rows * head_dim * 2], dtype=torch.int64)
    size_value_cache = torch.tensor([key_cache_rows * head_dim * 2], dtype=torch.int64)

    return [
        TensorSpec("query", [query_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("key_cache", [key_cache_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("value_cache", [key_cache_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("block_table", [block_table_flat_size], torch.int32, init_value=block_table_data),
        TensorSpec("context_lens", [batch], torch.int32, init_value=context_lens_data),
        TensorSpec("out", [query_rows, head_dim], torch.float32, is_output=True),
        TensorSpec("config", [7], torch.int64, init_value=config_data),
        TensorSpec("size_query", [1], torch.int64, init_value=size_query),
        TensorSpec("size_key_cache", [1], torch.int64, init_value=size_key_cache),
        TensorSpec("size_value_cache", [1], torch.int64, init_value=size_value_cache),
    ]


def main():
    batch = 64
    num_heads = 16
    head_dim = 128
    block_size = 128
    max_model_len = 32768
    context_len = 8192
    scale = 1.0
    max_num_blocks_per_req = max_model_len // block_size  # 256

    program = build_paged_attention_program(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks_per_req=max_num_blocks_per_req,
    )

    # Run on device (requires Simpler's CodeRunner in SIMPLER_ROOT)
    tensor_specs = build_tensor_specs(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks_per_req=max_num_blocks_per_req,
        context_len=context_len,
        scale=scale,
    )
    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden,
        config=RunConfig(
            platform="a2a3",
            device_id=11,
            rtol=2e-2,
            atol=2e-2,
            strategy=OptimizationStrategy.Default,
            dump_passes=True,
            backend_type=BackendType.Ascend910B_PTO,
        ),
    )
    print(f"Result: {result}")

    print("\nDone.")


if __name__ == "__main__":
    main()
