# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Batch Paged Attention Example (16x16 tiles, BF16)

Builds a batch paged attention program using the PyPTO DSL with online
softmax and a 6-kernel pipeline (AIC/AIV split).  The batch loop is moved
inside kernels so that the orchestration only has q_idx and bn loops,
matching the C++ implementation in paged_attention_orch.cpp.

Tensor layout (all 2D, flattened):
  query:       [batch * num_heads, head_dim]                    BF16
  key_cache:   [total_pool_blocks * block_size, head_dim]       BF16
  value_cache: [total_pool_blocks * block_size, head_dim]       BF16
  out:         [batch * num_heads, head_dim]                    FP32

Intermediate batched tensors (contiguous across batch dimension):
  sij_batch:     (batch * q_tile, block_size)  FP32
  pij_batch:     (batch * q_tile, block_size)  BF16
  mij/lij_batch: (batch * q_tile, 1)           FP32
  oi_new_batch:  (batch * q_tile, head_dim)    FP32
  oi_batch:      (batch * q_tile, head_dim)    FP32  accumulator
  mi/li_batch:   (batch * q_tile, 1)           FP32  accumulator

Pipeline per KV block iteration (all batches processed in a single kernel call):
  1. QK Matmul (AIC):        sij            = KernelQkMatmul(query, key_cache, sij, ...)
  2. Softmax Prepare (AIV):  pij, mij, lij  = KernelSoftmaxPrepare(sij, pij, mij, lij, ...)
  3. PV Matmul (AIC):        oi_new         = KernelPvMatmul(pij, value_cache, oi_new, ...)
  4. Online Update (AIV):    mi, li, oi, out = KernelOnlineUpdate(mij, lij, oi_new, mi, li, oi, out, ...)

Kernel mapping to C++ implementations:
  KernelAicHub         -> kernels/aic/aic_hub.cpp          (func_id=4)
  KernelAivHub         -> kernels/aiv/aiv_hub.cpp          (func_id=5)
  KernelQkMatmul       -> kernels/aic/aic_qk_matmul.cpp   (func_id=0)
  KernelSoftmaxPrepare -> kernels/aiv/aiv_softmax_prepare.cpp (func_id=1)
  KernelPvMatmul       -> kernels/aic/aic_pv_matmul.cpp   (func_id=2)
  KernelOnlineUpdate   -> kernels/aiv/aiv_online_update.cpp (func_id=3)
"""

import os

import pypto.language as pl
from pypto import ir
from pypto.backend import BackendType


def BuildBatchPagedAttentionProgram(
    batch: int,
    num_heads: int,
    head_dim: int = 16,
    block_size: int = 16,
    max_num_blocks_per_req: int = 16,
    q_tile: int = 16,
):
    """Build a parameterised batch-paged-attention @pl.program for the given shapes.

    Returns the decorated program class (not an instance).  The tensor type
    annotations in the orchestration function are filled in from the arguments
    so that the PyPTO DSL can resolve static dimensions at compile time.

    Parameters
    ----------
    batch:                  number of requests in the batch
    num_heads:              number of query heads
    head_dim:               per-head feature dimension (default 16)
    block_size:             KV-cache block size (default 16)
    max_num_blocks_per_req: maximum number of KV blocks per request
    q_tile:                 query-head tile size used by InCore kernels (default 16)
    """
    query_rows = batch * num_heads
    key_cache_rows = batch * max_num_blocks_per_req * block_size
    out_rows = batch * num_heads
    block_table_flat_size = batch * max_num_blocks_per_req
    batch_q_tile = batch * q_tile

    @pl.program
    class BatchPagedAttentionProgram:
        """Batch paged attention with AIC (CUBE) and AIV (VECTOR) kernels."""

        # ── AIC hub kernel (placeholder for AIC-side synchronisation) ────
        @pl.function(type=pl.FunctionType.InCore)
        def KernelAicHub(self) -> None:
            """AIC hub: empty placeholder (func_id=4)."""

        # ── AIV hub kernel: zero-initialise batch-sized accumulators ─────
        @pl.function(type=pl.FunctionType.InCore)
        def KernelAivHub(
            self,
            oi_batch: pl.Out[pl.Tensor[[batch_q_tile, head_dim], pl.FP32]],
            li_batch: pl.Out[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
            mi_batch: pl.Out[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[batch_q_tile, head_dim], pl.FP32],
            pl.Tensor[[batch_q_tile, 1], pl.FP32],
            pl.Tensor[[batch_q_tile, 1], pl.FP32],
        ]:
            """AIV hub: zero-initialise inplace accumulators (func_id=5)."""
            return oi_batch, li_batch, mi_batch

        # ── CUBE kernel: QK matmul ──────────────────────────────────────
        # C++ params: query(in), key_cache(in), sij_batch(out),
        #   block_table(tensor), batch_count, block_idx, q_offset,
        #   block_num, num_heads  (9 params)
        @pl.function(type=pl.FunctionType.InCore)
        def KernelQkMatmul(
            self,
            query: pl.Tensor[[query_rows, head_dim], pl.BF16],
            key_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
            sij_batch: pl.Out[pl.Tensor[[batch_q_tile, block_size], pl.FP32]],
            block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
            batch_count: pl.Scalar[pl.INT64],
            block_idx: pl.Scalar[pl.INT64],
            q_offset: pl.Scalar[pl.INT64],
            block_num: pl.Scalar[pl.INT64],
            num_heads_param: pl.Scalar[pl.INT64],
        ) -> pl.Tensor[[batch_q_tile, block_size], pl.FP32]:
            """QK matmul: sij = qi @ kj.T (CUBE, func_id=0).

            Loops over batch internally, computing per-batch qi and kj
            addresses from block_table lookup and q_offset.
            """
            for b in pl.range(batch_count):
                qi_row = b * num_heads_param + q_offset
                phys_block = pl.read(block_table, b * block_num + block_idx)
                kj_row = phys_block * block_size

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
                )
                qi_l0a = pl.move(qi_l1, target_memory=pl.MemorySpace.Left)
                kj_l0b = pl.move(kj_l1, target_memory=pl.MemorySpace.Right, transpose=True)
                sij_l0c = pl.matmul(qi_l0a, kj_l0b)
                sij_batch_new = pl.store(sij_l0c, [b * q_tile, 0], sij_batch)
            return sij_batch_new

        # ── VECTOR kernel: softmax prepare ──────────────────────────────
        # C++ params: sij_batch(in), pij_batch(out), mij_batch(out),
        #   lij_batch(out), scale_value, context_lens(tensor),
        #   batch_count, block_idx  (8 params)
        @pl.function(type=pl.FunctionType.InCore)
        def KernelSoftmaxPrepare(
            self,
            sij_batch: pl.Tensor[[batch_q_tile, block_size], pl.FP32],
            pij_batch: pl.Out[pl.Tensor[[batch_q_tile, block_size], pl.BF16]],
            mij_batch: pl.Out[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
            lij_batch: pl.Out[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
            scale_value: pl.Scalar[pl.FP32],
            context_lens: pl.Tensor[[batch], pl.INT32],
            batch_count: pl.Scalar[pl.INT64],
            block_idx: pl.Scalar[pl.INT64],
        ) -> tuple[
            pl.Tensor[[batch_q_tile, block_size], pl.BF16],
            pl.Tensor[[batch_q_tile, 1], pl.FP32],
            pl.Tensor[[batch_q_tile, 1], pl.FP32],
        ]:
            """Softmax prepare: scale, row_max, exp, row_sum (VECTOR, func_id=1).

            Loops over batch internally, masking invalid positions with -inf
            using context_lens and block_idx to compute per-batch valid_len.
            """
            for b in pl.range(batch_count):
                cur_seq = pl.tensor.read(context_lens, [b])
                start = block_idx * block_size
                remaining = cur_seq - start
                valid_len = pl.max(pl.min(remaining, block_size), 0)

                s_tile = pl.load(
                    sij_batch,
                    [b * q_tile, 0],
                    [q_tile, block_size],
                    target_memory=pl.MemorySpace.Vec,
                )

                # Keep the allocated tile static and narrow only the logical valid columns.
                sij_dyn = pl.tile.slice(s_tile, [q_tile, block_size], [0, 0], valid_shape=[q_tile, valid_len])
                s_tile = pl.tile.fillpad(sij_dyn)

                scaled = pl.mul(s_tile, scale_value)
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

        # ── CUBE kernel: PV matmul ──────────────────────────────────────
        # C++ params: pij_batch(in), value_cache(in), oi_new_batch(out),
        #   block_table(tensor), batch_count, block_idx, block_num  (7 params)
        @pl.function(type=pl.FunctionType.InCore)
        def KernelPvMatmul(
            self,
            pij_batch: pl.Tensor[[batch_q_tile, block_size], pl.BF16],
            value_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
            oi_new_batch: pl.Out[pl.Tensor[[batch_q_tile, head_dim], pl.FP32]],
            block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
            batch_count: pl.Scalar[pl.INT64],
            block_idx: pl.Scalar[pl.INT64],
            block_num: pl.Scalar[pl.INT64],
        ) -> pl.Tensor[[batch_q_tile, head_dim], pl.FP32]:
            """PV matmul: oi_tmp = pij @ vj (CUBE, func_id=2).

            Loops over batch internally, computing per-batch pij offset
            and vj address from block_table lookup.
            """
            for b in pl.range(batch_count):
                phys_block = pl.read(block_table, b * block_num + block_idx)
                vj_row = phys_block * block_size

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

        # ── VECTOR kernel: online update (inplace) ──────────────────────
        # C++ params: mij_batch(in), lij_batch(in), oi_new_batch(in),
        #   mi_batch(inout), li_batch(inout), oi_batch(out), out(out),
        #   is_first, is_last, batch_count, q_offset, num_heads  (12 params)
        @pl.function(type=pl.FunctionType.InCore)
        def KernelOnlineUpdate(  # noqa: PLR0913
            self,
            mij_batch: pl.Tensor[[batch_q_tile, 1], pl.FP32],
            lij_batch: pl.Tensor[[batch_q_tile, 1], pl.FP32],
            oi_new_batch: pl.Tensor[[batch_q_tile, head_dim], pl.FP32],
            mi_batch: pl.InOut[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
            li_batch: pl.InOut[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
            oi_batch: pl.InOut[pl.Tensor[[batch_q_tile, head_dim], pl.FP32]],
            out_tensor: pl.Out[pl.Tensor[[out_rows, head_dim], pl.FP32]],
            is_first: pl.Scalar[pl.INT64],
            is_last: pl.Scalar[pl.INT64],
            batch_count: pl.Scalar[pl.INT64],
            q_offset: pl.Scalar[pl.INT64],
            num_heads_param: pl.Scalar[pl.INT64],
        ) -> tuple[
            pl.Tensor[[batch_q_tile, 1], pl.FP32],
            pl.Tensor[[batch_q_tile, 1], pl.FP32],
            pl.Tensor[[batch_q_tile, head_dim], pl.FP32],
            pl.Tensor[[out_rows, head_dim], pl.FP32],
        ]:
            """Online softmax update with inplace mi/li/oi (VECTOR, func_id=3).

            Loops over batch internally, updating accumulators per batch and
            writing normalised output to out_tensor at the correct batch offset
            on the last KV-block iteration.
            """
            for b in pl.range(batch_count):
                dst_row = b * num_heads_param + q_offset

                if is_first:
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

                    if is_last:
                        dst_tile = pl.row_expand_div(oi_new_tile, lij_tile)
                        out_tensor = pl.store(dst_tile, [dst_row, 0], out_tensor, [q_tile, head_dim])
                else:
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

                    # Reshape DN [q_tile,1] -> ND [1,q_tile] for element-wise ops
                    mi_tile_nd = pl.reshape(mi_tile, [1, q_tile])
                    mij_tile_nd = pl.reshape(mij_tile, [1, q_tile])
                    li_tile_nd = pl.reshape(li_tile, [1, q_tile])
                    lij_tile_nd = pl.reshape(lij_tile, [1, q_tile])

                    mi_new = pl.maximum(mi_tile_nd, mij_tile_nd)
                    alpha = pl.exp(pl.sub(mi_tile_nd, mi_new))
                    beta = pl.exp(pl.sub(mij_tile_nd, mi_new))
                    li_updated = pl.add(pl.mul(alpha, li_tile_nd), pl.mul(beta, lij_tile_nd))

                    # Reshape back to DN [q_tile,1] for store and row_expand ops
                    mi_new_dn = pl.reshape(mi_new, [q_tile, 1])
                    li_updated_dn = pl.reshape(li_updated, [q_tile, 1])

                    mi_batch = pl.store(mi_new_dn, [b * q_tile, 0], mi_batch, [q_tile, 1])
                    li_batch = pl.store(li_updated_dn, [b * q_tile, 0], li_batch, [q_tile, 1])

                    # Reshape ND [1,q_tile] -> DN [q_tile,1] for row_expand_mul
                    alpha_dn = pl.reshape(alpha, [q_tile, 1])
                    beta_dn = pl.reshape(beta, [q_tile, 1])
                    oi_scaled = pl.row_expand_mul(oi_tile, alpha_dn)
                    oi_new_scaled = pl.row_expand_mul(oi_new_tile, beta_dn)
                    oi_updated = pl.add(oi_scaled, oi_new_scaled)

                    if is_last:
                        dst_tile = pl.row_expand_div(oi_updated, li_updated_dn)
                        out_tensor = pl.store(dst_tile, [dst_row, 0], out_tensor, [q_tile, head_dim])
                    else:
                        oi_batch = pl.store(
                            oi_updated,
                            [b * q_tile, 0],
                            oi_batch,
                            [q_tile, head_dim],
                        )

            return mi_batch, li_batch, oi_batch, out_tensor

        # ── Orchestration function ──────────────────────────────────────
        # Mirrors aicpu_orchestration_entry() in paged_attention_orch.cpp:
        #   - No batch loop (batch dimension is inside kernels)
        #   - Loops: for q_idx -> for bn
        #   - Intermediate tensors are batch-sized (batch * q_tile)
        #   - Kernels receive full global tensors + scalar metadata
        @pl.function(type=pl.FunctionType.Orchestration)
        def BatchPagedAttention(
            self,
            query: pl.Tensor[[query_rows, head_dim], pl.BF16],
            key_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
            value_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
            block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
            context_lens: pl.Tensor[[batch], pl.INT32],
            out: pl.Out[pl.Tensor[[out_rows, head_dim], pl.FP32]],
            config: pl.Tensor[[7], pl.INT64],
            size_query: pl.Tensor[[1], pl.INT64],
            size_key_cache: pl.Tensor[[1], pl.INT64],
            size_value_cache: pl.Tensor[[1], pl.INT64],
        ) -> pl.Tensor[[out_rows, head_dim], pl.FP32]:
            """Batch paged attention orchestration.

            Strictly follows aicpu_orchestration_entry() logic:
            no batch loop, q_idx outer loop, bn inner loop.
            Config layout: [batch, num_heads, kv_head_num, head_dim,
                            block_size, block_num, scale_bits]
            """
            batch_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
            num_heads_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
            head_dim_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [3])
            block_size_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [4])
            block_num_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [5])

            q_loop_cfg = (num_heads_cfg + q_tile - 1) // q_tile

            # Compute max_bn across all batches (mirrors C++ max_bn loop)
            max_bn: pl.Scalar[pl.INT64] = pl.yield_(0)
            for b in pl.range(batch_cfg):
                cur_seq_b = pl.tensor.read(context_lens, [b])
                bn_b = (cur_seq_b + block_size_cfg - 1) // block_size_cfg
                max_bn = pl.max(max_bn, bn_b)

            for q_idx in pl.range(q_loop_cfg):
                q_offset = q_idx * q_tile

                # Batch-sized accumulators (mirrors C++ oi_batch/li_batch/mi_batch)
                oi_batch: pl.Tensor[[batch_cfg * q_tile, head_dim_cfg], pl.FP32] = pl.create_tensor(
                    [batch_cfg * q_tile, head_dim_cfg],
                    dtype=pl.FP32,
                )
                li_batch: pl.Tensor[[batch_cfg * q_tile, 1], pl.FP32] = pl.create_tensor(
                    [batch_cfg * q_tile, 1],
                    dtype=pl.FP32,
                )
                mi_batch: pl.Tensor[[batch_cfg * q_tile, 1], pl.FP32] = pl.create_tensor(
                    [batch_cfg * q_tile, 1],
                    dtype=pl.FP32,
                )

                # Zero-init accumulators via AIV hub (FUNC_AIV_HUB)
                oi_batch, li_batch, mi_batch = self.KernelAivHub(oi_batch, li_batch, mi_batch)

                for bn in pl.range(max_bn):
                    # Batch-sized intermediate tensors (mirrors C++ sij_b/pij_b/etc.)
                    sij_b: pl.Tensor[[batch_cfg * q_tile, block_size_cfg], pl.FP32] = pl.create_tensor(
                        [batch_cfg * q_tile, block_size_cfg],
                        dtype=pl.FP32,
                    )
                    pij_b: pl.Tensor[[batch_cfg * q_tile, block_size_cfg], pl.FP16] = pl.create_tensor(
                        [batch_cfg * q_tile, block_size_cfg],
                        dtype=pl.FP16,
                    )
                    mij_b: pl.Tensor[[batch_cfg * q_tile, 1], pl.FP32] = pl.create_tensor(
                        [batch_cfg * q_tile, 1],
                        dtype=pl.FP32,
                    )
                    lij_b: pl.Tensor[[batch_cfg * q_tile, 1], pl.FP32] = pl.create_tensor(
                        [batch_cfg * q_tile, 1],
                        dtype=pl.FP32,
                    )
                    oi_new_b: pl.Tensor[[batch_cfg * q_tile, head_dim_cfg], pl.FP32] = pl.create_tensor(
                        [batch_cfg * q_tile, head_dim_cfg],
                        dtype=pl.FP32,
                    )

                    # Stage 1: QK matmul (FUNC_QK_MATMUL, AIC / CUBE)
                    sij_b = self.KernelQkMatmul(
                        query,
                        key_cache,
                        sij_b,
                        block_table,
                        batch_cfg,
                        bn,
                        q_offset,
                        block_num_cfg,
                        num_heads_cfg,
                    )

                    # Stage 2: Softmax prepare (FUNC_SOFTMAX_PREPARE, AIV / VECTOR)
                    pij_b, mij_b, lij_b = self.KernelSoftmaxPrepare(
                        sij_b,
                        pij_b,
                        mij_b,
                        lij_b,
                        1.0,  # type: ignore[reportArgumentType]
                        context_lens,
                        batch_cfg,
                        bn,
                    )

                    # Stage 3: PV matmul (FUNC_PV_MATMUL, AIC / CUBE)
                    oi_new_b = self.KernelPvMatmul(
                        pij_b,
                        value_cache,
                        oi_new_b,
                        block_table,
                        batch_cfg,
                        bn,
                        block_num_cfg,
                    )

                    # Conditional flags (mirrors C++ is_first/is_last)
                    if bn == 0:
                        is_first: pl.Scalar[pl.INT64] = pl.yield_(1)
                    else:
                        is_first: pl.Scalar[pl.INT64] = pl.yield_(0)
                    if bn == max_bn - 1:
                        is_last: pl.Scalar[pl.INT64] = pl.yield_(1)
                    else:
                        is_last: pl.Scalar[pl.INT64] = pl.yield_(0)

                    # Stage 4: Online update (FUNC_ONLINE_UPDATE, AIV / VECTOR)
                    mi_batch, li_batch, oi_batch, out = self.KernelOnlineUpdate(
                        mij_b,
                        lij_b,
                        oi_new_b,
                        mi_batch,
                        li_batch,
                        oi_batch,
                        out,
                        is_first,
                        is_last,
                        batch_cfg,
                        q_offset,
                        num_heads_cfg,
                    )

            return out

    return BatchPagedAttentionProgram


def main():
    """Build IR, compile, and display generated orchestration C++ code."""
    print("=" * 70)
    print("Batch Paged Attention Orchestration Code Generation (16x16)")
    print("=" * 70)

    program = BuildBatchPagedAttentionProgram(
        batch=1,
        num_heads=16,
        head_dim=16,
        block_size=16,
        max_num_blocks_per_req=16,
    )
    print(f"\nProgram: {program.name}")
    print(f"Functions: {[f.name for f in program.functions.values()]}")

    print("\n[1] IR Preview:")
    print("-" * 70)
    ir_text = program.as_python()
    lines = ir_text.split("\n")
    preview = min(60, len(lines))
    print("\n".join(lines[:preview]))
    if len(lines) > preview:
        print(f"\n... ({len(lines) - preview} more lines)")
    print("-" * 70)

    print("\n[2] Compiling...")
    output_dir = ir.compile(
        program,
        strategy=ir.OptimizationStrategy.CCE,
        dump_passes=True,
        backend_type=BackendType.Ascend910B_CCE,
    )
    print(f"Output: {output_dir}")

    print("\n[3] Generated files:")
    for root, _dirs, files in os.walk(output_dir):
        for f in files:
            path = os.path.join(root, f)
            rel = os.path.relpath(path, output_dir)
            print(f"  - {rel} ({os.path.getsize(path)} bytes)")

    orch_file = os.path.join(output_dir, "orchestration", "batch_paged_attention.cpp")
    if os.path.exists(orch_file):
        print("\n[4] Generated Orchestration C++ path:")
        print(orch_file)

    print("\nDone.")


if __name__ == "__main__":
    main()
