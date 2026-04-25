# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
SPMD Paged Attention Example

Demonstrates multi-block SPMD dispatch of paged attention using pl.spmd().
Each SPMD block processes a subset of batch items via a stride loop,
parallelizing the batch dimension across hardware blocks.

Architecture (per KV-block iteration, 4 sequential SPMD submissions):
  1. QK Matmul (SPMD):        sij = SpmdQkMatmul(...)           [CUBE]
  2. Softmax Prepare (SPMD):   pij, mij, lij = SpmdSoftmaxPrepare(...)  [VECTOR]
  3. PV Matmul (SPMD):         oi_new = SpmdPvMatmul(...)        [CUBE]
  4. Online Update (SPMD):     mi, li, oi = SpmdOnlineUpdate(...) [VECTOR]

After the KV-block loop, a final SPMD normalization (SpmdNormalize) divides
the accumulated output by the softmax denominator and scatters to the output tensor.

Work partitioning: each HW block processes batch items in a stride loop where
  b starts at block_idx and increments by block_num (core_num=4).

Orchestration structure:
  - Outer loop over q_tile groups (tiling the num_heads dimension)
  - Inner loop over KV blocks (up to max_bn, the maximum number of
    KV blocks across all requests, clamped by active_block_num)
  - Each kernel stage is dispatched via pl.spmd(core_num=4)
  - SpmdSoftmaxPrepare and SpmdOnlineUpdate are additionally wrapped
    in pl.cluster() for cooperative execution

Tensor layout:
  query:       [batch * num_heads, head_dim]                    BF16
  key_cache:   [batch * max_num_blocks_per_req * block_size, head_dim]  BF16
  value_cache: [batch * max_num_blocks_per_req * block_size, head_dim]  BF16
  block_table: [batch * max_num_blocks_per_req]                 INT32
  context_lens:[batch]                                          INT32
  scale:       [1]                                              FP32
  out:         [batch * num_heads, head_dim]                    FP32
  config:      [7]  (batch, num_heads, kv_head_num, head_dim,
                     block_size, block_num_capacity, active_block_num)  INT64
"""

import argparse
import os
from collections.abc import Sequence

import pypto.language as pl
import torch
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime import RunConfig, run


def _get_default_device_id() -> int:
    """Return the runtime device id from env, defaulting to 0."""
    for env_name in ("TILE_FWK_DEVICE_ID", "ASCEND_DEVICE_ID", "ACL_DEVICE_ID", "DEVICE_ID"):
        env_value = os.environ.get(env_name)
        if env_value is None:
            continue
        try:
            return int(env_value)
        except (TypeError, ValueError):
            continue
    return 0


def build_paged_attention_spmd_program(
    batch: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_num_blocks_per_req: int,
    q_tile: int = 16,
):
    """Build a parameterised SPMD paged-attention @pl.program.

    Returns the decorated program class (not an instance). Uses pl.spmd()
    to dispatch each kernel stage across multiple hardware blocks, with
    batch-parallel stride partitioning inside InCore kernels.

    Parameters
    ----------
    batch:                  number of requests in the batch
    num_heads:              number of query heads
    head_dim:               per-head feature dimension
    block_size:             KV-cache block size (rows per physical block)
    max_num_blocks_per_req: maximum number of KV blocks per request
    q_tile:                 query-head tile size used by the InCore kernels
    """
    query_rows = batch * num_heads
    key_cache_rows = batch * max_num_blocks_per_req * block_size
    out_rows = batch * num_heads
    block_table_flat_size = batch * max_num_blocks_per_req
    batch_q_tile = batch * q_tile

    @pl.program
    class SpmdPagedAttentionProgram:
        """SPMD paged attention with batch-parallel stride partitioning."""

        # ── CUBE kernel: QK matmul ─────────────────────────────────────
        # For each batch item b (stride-partitioned across HW blocks):
        #   1. Compute the valid KV length for this block index
        #   2. Look up the physical block from block_table
        #   3. Load query tile [q_tile, head_dim] and key tile [block_size, head_dim]
        #   4. Compute sij = Q @ K^T and store to sij_batch at batch offset
        @pl.function(type=pl.FunctionType.InCore)
        def SpmdQkMatmul(  # noqa: PLR0913
            self,
            query: pl.Tensor[[query_rows, head_dim], pl.BF16],
            key_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
            sij_batch: pl.Out[pl.Tensor[[batch_q_tile, block_size], pl.FP32]],
            block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
            context_lens: pl.Tensor[[batch], pl.INT32],
            batch_count: pl.Scalar[pl.INDEX],
            block_idx_kv: pl.Scalar[pl.INDEX],
            q_offset: pl.Scalar[pl.INDEX],
            block_num: pl.Scalar[pl.INDEX],
            block_size_param: pl.Scalar[pl.INDEX],
            num_heads_param: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[batch_q_tile, block_size], pl.FP32]:
            """QK matmul with SPMD stride partitioning over batch (CUBE)."""
            spmd_idx = pl.tile.get_block_idx()
            spmd_block_num = pl.tile.get_block_num()
            for b in pl.range(spmd_idx, batch_count, spmd_block_num):
                qi_row = b * num_heads_param + q_offset
                cur_seq = pl.tensor.read(context_lens, [b])
                start = block_idx_kv * block_size_param
                remaining = cur_seq - start
                valid_len = pl.max(pl.min(remaining, block_size_param), 0)

                if valid_len > 0:
                    phys_block_raw: pl.Scalar[pl.INT32] = pl.read(block_table, b * block_num + block_idx_kv)
                    phys_block_idx: pl.Scalar[pl.INDEX] = pl.cast(phys_block_raw, pl.INDEX)
                    kj_row = phys_block_idx * block_size_param

                    qi_l1 = pl.load(
                        query,
                        [qi_row, 0],
                        [q_tile, head_dim],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    kj_l1 = pl.load(
                        key_cache,
                        [kj_row, 0],
                        [block_size, head_dim],
                        target_memory=pl.MemorySpace.Mat,
                        transpose=True,
                    )
                    qi_l0a = pl.move(qi_l1, target_memory=pl.MemorySpace.Left)
                    kj_l0b = pl.move(kj_l1, target_memory=pl.MemorySpace.Right)
                    sij_l0c = pl.matmul(qi_l0a, kj_l0b)
                    sij_batch = pl.store(sij_l0c, [b * q_tile, 0], sij_batch)
            return sij_batch

        # ── VECTOR kernel: softmax prepare ─────────────────────────────
        # For each batch item b (stride-partitioned across HW blocks):
        #   If valid_len == 0: produce zero pij, -inf mij, zero lij
        #   Else:
        #     1. Load sij scores, pad invalid positions with min value
        #     2. Scale scores by scale_value
        #     3. Compute row-wise max (mij) and center scores
        #     4. Compute exp(sij - mij) -> pij, and row-wise sum -> lij
        @pl.function(type=pl.FunctionType.InCore)
        def SpmdSoftmaxPrepare(
            self,
            sij_batch: pl.Tensor[[batch_q_tile, block_size], pl.FP32],
            pij_batch: pl.Out[pl.Tensor[[batch_q_tile, block_size], pl.BF16]],
            mij_batch: pl.Out[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
            lij_batch: pl.Out[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
            scale_value: pl.Scalar[pl.FP32],
            context_lens: pl.Tensor[[batch], pl.INT32],
            batch_count: pl.Scalar[pl.INDEX],
            block_idx_kv: pl.Scalar[pl.INDEX],
            block_size_param: pl.Scalar[pl.INDEX],
        ) -> tuple[
            pl.Tensor[[batch_q_tile, block_size], pl.BF16],
            pl.Tensor[[batch_q_tile, 1], pl.FP32],
            pl.Tensor[[batch_q_tile, 1], pl.FP32],
        ]:
            """Softmax prepare with SPMD stride over batch (VECTOR)."""
            spmd_idx = pl.tile.get_block_idx()
            spmd_block_num = pl.tile.get_block_num()
            for b in pl.range(spmd_idx, batch_count, spmd_block_num):
                cur_seq = pl.tensor.read(context_lens, [b])
                start = block_idx_kv * block_size_param
                remaining = cur_seq - start
                valid_len = pl.max(pl.min(remaining, block_size_param), 0)

                if valid_len == 0:
                    zero_pij_f32_tile = pl.tile.full([q_tile, block_size], dtype=pl.FP32, value=0.0)
                    zero_pij_tile = pl.cast(zero_pij_f32_tile, target_type=pl.BF16)
                    zero_tmp_tile = pl.create_tile(
                        [q_tile, block_size],
                        dtype=pl.FP32,
                        target_memory=pl.MemorySpace.Vec,
                    )
                    zero_li_tile = pl.row_sum(zero_pij_f32_tile, zero_tmp_tile)

                    neg_inf_scores_tile = pl.tile.full([q_tile, block_size], dtype=pl.FP32, value=-1e30)
                    neg_inf_tmp_tile = pl.create_tile(
                        [q_tile, block_size],
                        dtype=pl.FP32,
                        target_memory=pl.MemorySpace.Vec,
                    )
                    neg_inf_mi_tile = pl.row_max(neg_inf_scores_tile, neg_inf_tmp_tile)
                    pij_batch = pl.store(zero_pij_tile, [b * q_tile, 0], pij_batch, [q_tile, block_size])
                    mij_batch = pl.store(neg_inf_mi_tile, [b * q_tile, 0], mij_batch, [q_tile, 1])
                    lij_batch = pl.store(zero_li_tile, [b * q_tile, 0], lij_batch, [q_tile, 1])
                else:
                    s_tile = pl.load(
                        sij_batch,
                        [b * q_tile, 0],
                        [q_tile, block_size],
                        target_memory=pl.MemorySpace.Vec,
                        valid_shapes=[q_tile, valid_len],
                    )
                    s_padded = pl.tile.fillpad(s_tile, pad_value=pl.PadValue.min)
                    scaled = pl.mul(s_padded, scale_value)
                    tmp_tile = pl.create_tile(
                        [q_tile, block_size],
                        dtype=pl.FP32,
                        target_memory=pl.MemorySpace.Vec,
                    )
                    mi_tile = pl.row_max(scaled, tmp_tile)
                    sij_centered = pl.row_expand_sub(scaled, mi_tile)
                    exp_tile = pl.exp(sij_centered)
                    pij_tile_f16 = pl.cast(exp_tile, target_type=pl.BF16)
                    pij_tile = pl.cast(pij_tile_f16, target_type=pl.FP32)
                    li_tile = pl.row_sum(pij_tile, tmp_tile)
                    pij_batch = pl.store(pij_tile_f16, [b * q_tile, 0], pij_batch, [q_tile, block_size])
                    mij_batch = pl.store(mi_tile, [b * q_tile, 0], mij_batch, [q_tile, 1])
                    lij_batch = pl.store(li_tile, [b * q_tile, 0], lij_batch, [q_tile, 1])
            return pij_batch, mij_batch, lij_batch

        # ── CUBE kernel: PV matmul ─────────────────────────────────────
        # For each batch item b (stride-partitioned across HW blocks):
        #   1. Look up physical block from block_table (same logic as QK)
        #   2. Load pij tile [q_tile, block_size] and value tile [block_size, head_dim]
        #   3. Compute oi_new = pij @ V and store to oi_new_batch
        @pl.function(type=pl.FunctionType.InCore)
        def SpmdPvMatmul(
            self,
            pij_batch: pl.Tensor[[batch_q_tile, block_size], pl.BF16],
            value_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
            oi_new_batch: pl.Out[pl.Tensor[[batch_q_tile, head_dim], pl.FP32]],
            block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
            context_lens: pl.Tensor[[batch], pl.INT32],
            batch_count: pl.Scalar[pl.INDEX],
            block_idx_kv: pl.Scalar[pl.INDEX],
            block_num: pl.Scalar[pl.INDEX],
            block_size_param: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[batch_q_tile, head_dim], pl.FP32]:
            """PV matmul with SPMD stride partitioning over batch (CUBE)."""
            spmd_idx = pl.tile.get_block_idx()
            spmd_block_num = pl.tile.get_block_num()
            for b in pl.range(spmd_idx, batch_count, spmd_block_num):
                cur_seq = pl.tensor.read(context_lens, [b])
                start = block_idx_kv * block_size_param
                remaining = cur_seq - start
                valid_len = pl.max(pl.min(remaining, block_size_param), 0)

                if valid_len > 0:
                    phys_block_raw: pl.Scalar[pl.INT32] = pl.read(block_table, b * block_num + block_idx_kv)
                    phys_block_idx: pl.Scalar[pl.INDEX] = pl.cast(phys_block_raw, pl.INDEX)
                    vj_row = phys_block_idx * block_size_param

                    pij_l1 = pl.load(
                        pij_batch,
                        [b * q_tile, 0],
                        [q_tile, block_size],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    vj_l1 = pl.load(
                        value_cache,
                        [vj_row, 0],
                        [block_size, head_dim],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    pij_l0a = pl.move(pij_l1, target_memory=pl.MemorySpace.Left)
                    vj_l0b = pl.move(vj_l1, target_memory=pl.MemorySpace.Right)
                    oi_l0c = pl.matmul(pij_l0a, vj_l0b)
                    oi_new_batch = pl.store(oi_l0c, [b * q_tile, 0], oi_new_batch)
            return oi_new_batch

        # ── VECTOR kernel: online softmax update (in-place accumulators) ─
        # For each batch item b (stride-partitioned across HW blocks):
        #   If is_first (first KV block): directly copy mij, lij, oi_new
        #   Else: apply online softmax update:
        #     mi_new = max(mi, mij)
        #     alpha = exp(mi - mi_new), beta = exp(mij - mi_new)
        #     li = alpha * li + beta * lij
        #     oi = alpha * oi + beta * oi_new
        @pl.function(type=pl.FunctionType.InCore)
        def SpmdOnlineUpdate(  # noqa: PLR0913
            self,
            mij_batch: pl.Tensor[[batch_q_tile, 1], pl.FP32],
            lij_batch: pl.Tensor[[batch_q_tile, 1], pl.FP32],
            oi_new_batch: pl.Tensor[[batch_q_tile, head_dim], pl.FP32],
            mi_batch: pl.InOut[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
            li_batch: pl.InOut[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
            oi_batch: pl.InOut[pl.Tensor[[batch_q_tile, head_dim], pl.FP32]],
            is_first: pl.Scalar[pl.INDEX],
            batch_count: pl.Scalar[pl.INDEX],
        ) -> tuple[
            pl.Tensor[[batch_q_tile, 1], pl.FP32],
            pl.Tensor[[batch_q_tile, 1], pl.FP32],
            pl.Tensor[[batch_q_tile, head_dim], pl.FP32],
        ]:
            """Online softmax accumulator update (VECTOR)."""
            spmd_idx = pl.tile.get_block_idx()
            spmd_block_num = pl.tile.get_block_num()
            if is_first != 0:
                for b in pl.range(spmd_idx, batch_count, spmd_block_num):
                    mij_tile = pl.load(
                        mij_batch,
                        [b * q_tile, 0],
                        [q_tile, 1],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    lij_tile = pl.load(
                        lij_batch,
                        [b * q_tile, 0],
                        [q_tile, 1],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    oi_new_tile = pl.load(
                        oi_new_batch,
                        [b * q_tile, 0],
                        [q_tile, head_dim],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    mi_batch = pl.store(mij_tile, [b * q_tile, 0], mi_batch, [q_tile, 1])
                    li_batch = pl.store(lij_tile, [b * q_tile, 0], li_batch, [q_tile, 1])
                    oi_batch = pl.store(oi_new_tile, [b * q_tile, 0], oi_batch, [q_tile, head_dim])
            else:
                for b in pl.range(spmd_idx, batch_count, spmd_block_num):
                    mij_tile = pl.load(
                        mij_batch,
                        [b * q_tile, 0],
                        [q_tile, 1],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    lij_tile = pl.load(
                        lij_batch,
                        [b * q_tile, 0],
                        [q_tile, 1],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    oi_new_tile = pl.load(
                        oi_new_batch,
                        [b * q_tile, 0],
                        [q_tile, head_dim],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    mi_tile = pl.load(
                        mi_batch,
                        [b * q_tile, 0],
                        [q_tile, 1],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    li_tile = pl.load(
                        li_batch,
                        [b * q_tile, 0],
                        [q_tile, 1],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    oi_tile = pl.load(
                        oi_batch,
                        [b * q_tile, 0],
                        [q_tile, head_dim],
                        target_memory=pl.MemorySpace.Vec,
                    )

                    mi_tile_nd = pl.reshape(mi_tile, [1, q_tile])
                    mij_tile_nd = pl.reshape(mij_tile, [1, q_tile])
                    li_tile_nd = pl.reshape(li_tile, [1, q_tile])
                    lij_tile_nd = pl.reshape(lij_tile, [1, q_tile])

                    mi_new = pl.maximum(mi_tile_nd, mij_tile_nd)
                    alpha = pl.exp(pl.sub(mi_tile_nd, mi_new))
                    beta = pl.exp(pl.sub(mij_tile_nd, mi_new))
                    li_updated = pl.add(pl.mul(alpha, li_tile_nd), pl.mul(beta, lij_tile_nd))

                    mi_new_dn = pl.reshape(mi_new, [q_tile, 1])
                    li_updated_dn = pl.reshape(li_updated, [q_tile, 1])

                    mi_batch = pl.store(mi_new_dn, [b * q_tile, 0], mi_batch, [q_tile, 1])
                    li_batch = pl.store(li_updated_dn, [b * q_tile, 0], li_batch, [q_tile, 1])

                    alpha_dn = pl.reshape(alpha, [q_tile, 1])
                    beta_dn = pl.reshape(beta, [q_tile, 1])
                    oi_scaled = pl.row_expand_mul(oi_tile, alpha_dn)
                    oi_new_scaled = pl.row_expand_mul(oi_new_tile, beta_dn)
                    oi_updated = pl.add(oi_scaled, oi_new_scaled)
                    oi_batch = pl.store(oi_updated, [b * q_tile, 0], oi_batch, [q_tile, head_dim])

            return mi_batch, li_batch, oi_batch

        # ── VECTOR kernel: final normalization ─────────────────────────
        # For each batch item b (stride-partitioned across HW blocks):
        #   Compute out = oi / li and scatter to the output tensor at
        #   row (b * num_heads + q_offset)
        @pl.function(type=pl.FunctionType.InCore)
        def SpmdNormalize(
            self,
            oi_batch: pl.Tensor[[batch_q_tile, head_dim], pl.FP32],
            li_batch: pl.Tensor[[batch_q_tile, 1], pl.FP32],
            out_tensor: pl.Out[pl.Tensor[[out_rows, head_dim], pl.FP32]],
            batch_count: pl.Scalar[pl.INDEX],
            q_offset: pl.Scalar[pl.INDEX],
            num_heads_param: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[out_rows, head_dim], pl.FP32]:
            """Normalize accumulators and scatter to output (VECTOR)."""
            spmd_idx = pl.tile.get_block_idx()
            spmd_block_num = pl.tile.get_block_num()
            for b in pl.range(spmd_idx, batch_count, spmd_block_num):
                oi_tile = pl.load(
                    oi_batch,
                    [b * q_tile, 0],
                    [q_tile, head_dim],
                    target_memory=pl.MemorySpace.Vec,
                )
                li_tile = pl.load(
                    li_batch,
                    [b * q_tile, 0],
                    [q_tile, 1],
                    target_memory=pl.MemorySpace.Vec,
                )
                dst_tile = pl.row_expand_div(oi_tile, li_tile)
                dst_row = b * num_heads_param + q_offset
                out_tensor = pl.store(dst_tile, [dst_row, 0], out_tensor, [q_tile, head_dim])
            return out_tensor

        # ── Orchestration ──────────────────────────────────────────────
        # Reads runtime config, computes max KV blocks across all requests,
        # then runs a two-level loop:
        #   Outer: iterate over q_tile groups (tiling num_heads)
        #   Inner: iterate over KV blocks, dispatching 4 SPMD kernels per block
        # After inner loop, dispatches SpmdNormalize to produce final output.
        @pl.function(type=pl.FunctionType.Orchestration)
        def paged_attention_spmd(
            self,
            query: pl.Tensor[[query_rows, head_dim], pl.BF16],
            key_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
            value_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
            block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
            context_lens: pl.Tensor[[batch], pl.INT32],
            scale: pl.Tensor[[1], pl.FP32],
            out: pl.Out[pl.Tensor[[out_rows, head_dim], pl.FP32]],
            config: pl.Tensor[[7], pl.INT64],
        ) -> pl.Tensor[[out_rows, head_dim], pl.FP32]:
            """SPMD paged attention orchestration.

            Outer loop over q_tile groups, inner loop over KV blocks.
            Each kernel stage dispatched as SPMD across multiple hardware blocks.
            Config: [batch, num_heads, kv_head_num, head_dim,
                     block_size, block_num_capacity, active_block_num]
            """
            batch_cfg = pl.tensor.read(config, [0])
            num_heads_cfg = pl.tensor.read(config, [1])
            block_size_cfg = pl.tensor.read(config, [4])
            block_num_capacity_cfg = pl.tensor.read(config, [5])
            active_block_num_cfg = pl.tensor.read(config, [6])
            scale_value: pl.Scalar[pl.FP32] = pl.tensor.read(scale, [0])
            zero_bn_cfg: pl.Scalar[pl.INT64] = 0
            if batch_cfg == 0:
                max_bn_cfg: pl.Scalar[pl.INT64] = pl.yield_(zero_bn_cfg)
            else:
                first_seq = pl.tensor.read(context_lens, [0])
                first_bn_cfg: pl.Scalar[pl.INT64] = (first_seq + block_size_cfg - 1) // block_size_cfg
                max_bn_cfg: pl.Scalar[pl.INT64] = pl.yield_(first_bn_cfg)
            for b in pl.range(1, batch_cfg):
                cur_seq_b = pl.tensor.read(context_lens, [b])
                bn_b = (cur_seq_b + block_size_cfg - 1) // block_size_cfg
                max_bn_cfg = pl.max(max_bn_cfg, bn_b)

            max_bn_cfg = pl.min(max_bn_cfg, active_block_num_cfg)
            q_loop_cfg = (num_heads_cfg + q_tile - 1) // q_tile

            for q_idx in pl.range(q_loop_cfg):
                q_offset = q_idx * q_tile

                # Per-batch accumulators for online softmax across KV blocks.
                mi_a = pl.create_tensor([batch_q_tile, 1], dtype=pl.FP32)
                li_a = pl.create_tensor([batch_q_tile, 1], dtype=pl.FP32)
                oi_a = pl.create_tensor([batch_q_tile, head_dim], dtype=pl.FP32)

                for bn in pl.range(max_bn_cfg):
                    # Temporary tensors for this KV block iteration.
                    sij_b = pl.create_tensor([batch_q_tile, block_size], dtype=pl.FP32)
                    pij_b = pl.create_tensor([batch_q_tile, block_size], dtype=pl.BF16)
                    mij_b = pl.create_tensor([batch_q_tile, 1], dtype=pl.FP32)
                    lij_b = pl.create_tensor([batch_q_tile, 1], dtype=pl.FP32)
                    oi_new_b = pl.create_tensor([batch_q_tile, head_dim], dtype=pl.FP32)

                    # sync_start ensures all HW blocks begin together before
                    # the cooperative SPMD wave.
                    with pl.spmd(core_num=4, sync_start=True):
                        sij_b = self.SpmdQkMatmul(
                            query,
                            key_cache,
                            sij_b,
                            block_table,
                            context_lens,
                            batch_cfg,
                            bn,
                            q_offset,
                            block_num_capacity_cfg,
                            block_size_cfg,
                            num_heads_cfg,
                        )

                    with pl.cluster():
                        with pl.spmd(core_num=4):
                            pij_b, mij_b, lij_b = self.SpmdSoftmaxPrepare(
                                sij_b,
                                pij_b,
                                mij_b,
                                lij_b,
                                scale_value,
                                context_lens,
                                batch_cfg,
                                bn,
                                block_size_cfg,
                            )

                    with pl.spmd(core_num=4):
                        oi_new_b = self.SpmdPvMatmul(
                            pij_b,
                            value_cache,
                            oi_new_b,
                            block_table,
                            context_lens,
                            batch_cfg,
                            bn,
                            block_num_capacity_cfg,
                            block_size_cfg,
                        )

                    if bn == 0:
                        is_first = pl.yield_(1)
                    else:
                        is_first = pl.yield_(0)

                    with pl.cluster():
                        with pl.spmd(core_num=4):
                            mi_a, li_a, oi_a = self.SpmdOnlineUpdate(
                                mij_b,
                                lij_b,
                                oi_new_b,
                                mi_a,
                                li_a,
                                oi_a,
                                is_first,
                                batch_cfg,
                            )

                # Final normalization: out = oi / li, scattered to output tensor.
                with pl.spmd(core_num=4):
                    out = self.SpmdNormalize(
                        oi_a,
                        li_a,
                        out,
                        batch_cfg,
                        q_offset,
                        num_heads_cfg,
                    )

            return out

    return SpmdPagedAttentionProgram


def golden(tensors: dict, params: dict | None = None) -> None:
    """Reference paged-attention computation (torch), mirroring the kernel pipeline."""
    config = tensors["config"]
    batch_size = int(config[0].item())
    num_heads = int(config[1].item())
    head_dim = int(config[3].item())
    block_size = int(config[4].item())
    max_num_blocks_per_req = int(config[5].item())
    active_num_blocks = int(config[6].item())
    scale = float(tensors["scale"][0].item())

    query = tensors["query"].float().reshape(batch_size, num_heads, head_dim)
    total_pool_blocks = batch_size * max_num_blocks_per_req
    key_cache = tensors["key_cache"].float().reshape(total_pool_blocks, block_size, head_dim)
    value_cache = tensors["value_cache"].float().reshape(total_pool_blocks, block_size, head_dim)
    block_table = tensors["block_table"].reshape(batch_size, max_num_blocks_per_req)
    context_lens = tensors["context_lens"]

    out = torch.zeros((batch_size, num_heads, head_dim), dtype=torch.float32)
    q_tile = 16
    max_bn = min(
        active_num_blocks,
        int((context_lens.max().item() + block_size - 1) // block_size),
    )

    for q_offset in range(0, num_heads, q_tile):
        q_tile_size = min(q_tile, num_heads - q_offset)
        qi = query[:, q_offset : q_offset + q_tile_size, :]
        oi = torch.zeros(
            (batch_size, q_tile_size, head_dim),
            dtype=value_cache.dtype,
            device=qi.device,
        )
        li = torch.zeros(
            (batch_size, q_tile_size, 1),
            dtype=qi.dtype,
            device=qi.device,
        )
        mi = torch.zeros_like(li)

        for bn in range(max_bn):
            valid_lens = torch.clamp(context_lens - bn * block_size, min=0, max=block_size)
            if not (valid_lens > 0).any():
                break
            block_indices = block_table[:, bn]
            kj_all = key_cache[block_indices]
            vj_all = value_cache[block_indices]

            sij = torch.bmm(qi, kj_all.transpose(1, 2)) * scale
            pos = torch.arange(block_size).unsqueeze(0)
            valid_mask = (pos < valid_lens.unsqueeze(1)).unsqueeze(1)
            sij = sij.masked_fill(~valid_mask, float("-inf"))
            mij = sij.max(dim=-1, keepdim=True)[0].clamp(min=-1e30)
            pij = torch.exp(sij - mij).masked_fill(~valid_mask, 0.0)
            pij = pij.to(torch.bfloat16).to(torch.float32)
            lij = pij.sum(dim=-1, keepdim=True)
            oi_new = torch.bmm(pij, vj_all)

            if bn == 0:
                oi, li, mi = oi_new, lij, mij
            else:
                mi_new = torch.maximum(mi, mij)
                alpha = torch.exp(mi - mi_new)
                beta = torch.exp(mij - mi_new)
                li = alpha * li + beta * lij
                oi = alpha * oi + beta * oi_new
                mi = mi_new

        out[:, q_offset : q_offset + q_tile_size, :] = torch.where(
            li > 0,
            oi / li,
            torch.zeros_like(oi),
        )

    tensors["out"][:] = out.reshape(batch_size * num_heads, head_dim)


def build_tensor_specs(
    batch: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_num_blocks_per_req: int,
    active_num_blocks: int,
    context_len: int | Sequence[int] | torch.Tensor,
    scale: float = 1.0,
):
    """Build TensorSpec objects matching the paged_attention_spmd signature."""
    from pypto.runtime import TensorSpec  # noqa: PLC0415

    query_rows = batch * num_heads
    key_cache_rows = batch * max_num_blocks_per_req * block_size
    block_table_flat_size = batch * max_num_blocks_per_req

    if isinstance(context_len, torch.Tensor):
        context_lens_init = context_len.to(dtype=torch.int32)
    elif isinstance(context_len, Sequence) and not isinstance(context_len, (str, bytes)):
        context_lens_init = torch.tensor(list(context_len), dtype=torch.int32)
    else:
        context_lens_init = torch.full((batch,), int(context_len), dtype=torch.int32)

    if context_lens_init.numel() != batch:
        raise ValueError(
            f"context_len must provide exactly {batch} elements, got {context_lens_init.numel()}"
        )

    max_context_len = int(context_lens_init.max().item()) if batch > 0 else 0
    max_blocks_from_context = (max_context_len + block_size - 1) // block_size if block_size > 0 else 0

    if active_num_blocks > max_num_blocks_per_req:
        raise ValueError("active_num_blocks must not exceed max_num_blocks_per_req")
    active_num_blocks = min(active_num_blocks, max_blocks_from_context)

    def init_config():
        return torch.tensor(
            [
                batch,
                num_heads,
                1,
                head_dim,
                block_size,
                max_num_blocks_per_req,
                active_num_blocks,
            ],
            dtype=torch.int64,
        )

    def init_scale():
        return torch.tensor([scale], dtype=torch.float32)

    def init_context_lens():
        return context_lens_init.clone()

    def init_block_table():
        return torch.randint(
            0,
            max(block_table_flat_size, 1),
            size=(batch, max_num_blocks_per_req),
            dtype=torch.int32,
        ).flatten()

    return [
        TensorSpec("query", [query_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("key_cache", [key_cache_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("value_cache", [key_cache_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("block_table", [block_table_flat_size], torch.int32, init_value=init_block_table),
        TensorSpec("context_lens", [batch], torch.int32, init_value=init_context_lens),
        TensorSpec("scale", [1], torch.float32, init_value=init_scale),
        TensorSpec("out", [query_rows, head_dim], torch.float32, is_output=True),
        TensorSpec("config", [7], torch.int64, init_value=init_config),
    ]


def _build_runtime_tensors(tensor_specs):
    tensors = {spec.name: spec.create_tensor() for spec in tensor_specs}
    input_tensors = tuple(tensors[spec.name] for spec in tensor_specs if not spec.is_output)
    return tensors, input_tensors


def main():
    parser = argparse.ArgumentParser(description="SPMD paged attention example")
    default_device = _get_default_device_id()
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=default_device)
    parser.add_argument(
        "--runtime-profiling",
        action="store_true",
        default=False,
        help="Enable on-device runtime profiling and generate swimlane JSON",
    )
    args = parser.parse_args()
    print(f"Using platform={args.platform}, device_id={args.device}")

    batch = 4
    num_heads = 16
    head_dim = 128
    block_size = 128
    max_model_len = 32768
    context_len = 8192
    scale = 1.0
    max_num_blocks_per_req = max_model_len // block_size
    active_num_blocks = (context_len + block_size - 1) // block_size

    program = build_paged_attention_spmd_program(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks_per_req=max_num_blocks_per_req,
    )

    tensor_specs = build_tensor_specs(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks_per_req=max_num_blocks_per_req,
        active_num_blocks=active_num_blocks,
        context_len=context_len,
        scale=scale,
    )
    tensors, input_tensors = _build_runtime_tensors(tensor_specs)
    query = tensors["query"]
    key_cache = tensors["key_cache"]
    value_cache = tensors["value_cache"]
    block_table = tensors["block_table"]
    context_lens = tensors["context_lens"]
    scale_tensor = tensors["scale"]
    config = tensors["config"]
    run_config = RunConfig(
        platform=args.platform,
        device_id=args.device,
        strategy=OptimizationStrategy.Default,
        dump_passes=True,
        backend_type=BackendType.Ascend950 if args.platform.startswith("a5") else BackendType.Ascend910B,
        runtime_profiling=args.runtime_profiling,
    )
    compiled = run(program, config=run_config)
    output = compiled(*input_tensors, config=run_config)
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor output from compiled program, got {type(output).__name__}")

    expected_out = torch.zeros_like(output)
    golden(
        {
            "query": query,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "block_table": block_table,
            "context_lens": context_lens,
            "scale": scale_tensor,
            "out": expected_out,
            "config": config,
        },
    )
    assert torch.allclose(output, expected_out, rtol=2e-2, atol=2e-2), (
        f"Validation failed: max diff = {(output - expected_out).abs().max().item()}"
    )
    print("PASSED")


if __name__ == "__main__":
    main()
