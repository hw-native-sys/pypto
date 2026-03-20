# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Paged Attention 64 N_UNROLL Orchestration Example

64divisions configuration matching simper Case2:
  q_tile=64, block_size=64, head_dim=128, batch=64, num_heads=64, kv_head_num=1

Tile dimensions:
  QK Matmul:       qi(64, 128) @ kj.T(128, 64) → sij(64, 64)
  Softmax:         sij(64, N) → pij(64, N) bf16, mi(64, 1), li(64, 1)
  PV Matmul:       pij(64, 64) @ vj(64, 128) → oi(64, 128)
  Online Update:   operates on (64, 128) data tiles, (64, 1) scalar tiles

Uses single-loop N_UNROLL structure (matching C++ paged_attention_unroll):
  N_UNROLL = 64
  for bn in range(0, bn_this_batch, N_UNROLL):
      n_blocks = min(N_UNROLL, bn_this_batch - bn)
      KernelQkMatmul       — multi-block QK with internal loop
      KernelSoftmaxPrepare — two-pass softmax (global row_max, then exp+sum)
      KernelPvMatmul       — SplitK PV with matmul + matmul_acc
      KernelOnlineUpdate   — online softmax merge

Module-level InCore kernels (reusable, importable):
  kernel_aiv_hub, kernel_softmax_prepare_64, kernel_online_update_64

Factory functions for batch-dynamic kernels:
  make_kernel_qk_matmul_64(key_cache_rows, block_table_flat_size)
  make_kernel_pv_matmul_64(key_cache_rows, block_table_flat_size)
"""

import struct

import pypto.language as pl
import torch  # type: ignore[import]
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime import RunConfig, TensorSpec, run

# ── Constants ────────────────────────────────────────────────────────────────
Q_TILE = 16
BLOCK_SIZE = 64
HEAD_DIM = 128
N_UNROLL = 2
N_UNROLL_Q = N_UNROLL * Q_TILE  # 4096 — static sij/pij buffer height


# ── Module-level InCore kernels ───────────────────────────────────────────────


@pl.function(type=pl.FunctionType.InCore)
def kernel_aiv_hub(
    oi: pl.Out[pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32]],
    li: pl.Out[pl.Tensor[[Q_TILE, 1], pl.FP32]],
    mi: pl.Out[pl.Tensor[[Q_TILE, 1], pl.FP32]],
) -> tuple[
    pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32],
    pl.Tensor[[Q_TILE, 1], pl.FP32],
    pl.Tensor[[Q_TILE, 1], pl.FP32],
]:
    """Zero-initialise inplace accumulators (VECTOR)."""
    return oi, li, mi


@pl.function(type=pl.FunctionType.InCore)
def kernel_softmax_prepare_64(
    sij_buf: pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.FP32],
    scale: pl.Scalar[pl.FP32],
    pij_buf: pl.Out[pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.BF16]],
    mi_out: pl.InOut[pl.Tensor[[Q_TILE, 1], pl.FP32]],
    li_out: pl.InOut[pl.Tensor[[Q_TILE, 1], pl.FP32]],
    n_blocks: pl.Scalar[pl.INDEX],
) -> tuple[
    pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.BF16],
    pl.Tensor[[Q_TILE, 1], pl.FP32],
    pl.Tensor[[Q_TILE, 1], pl.FP32],
]:
    """Two-pass softmax: pass 1 finds global row_max, pass 2 computes exp+sum (VECTOR).

    Uses mi_out/li_out as GM scratch for cross-iteration state via store/load round-trips.
    Last block is masked to valid_len_last columns via slice+fillpad.
    """
    # Pass 1: find global row_max across all blocks
    for i, (mi_out_iter,) in pl.range(n_blocks, init_values=(mi_out,)):
        s_tile = pl.load(
            sij_buf,
            [i * Q_TILE, 0],
            [Q_TILE, BLOCK_SIZE],
            target_memory=pl.MemorySpace.Vec,
        )
        scaled = pl.mul(s_tile, scale)
        tmp_tile = pl.create_tile(
            [Q_TILE, BLOCK_SIZE],
            dtype=pl.FP32,
            target_memory=pl.MemorySpace.Vec,
        )
        local_max = pl.row_max(scaled, tmp_tile)

        if i == 0:
            mi_out_updated: pl.Tensor[[Q_TILE, 1], pl.FP32] = pl.store(local_max, [0, 0], mi_out_iter)
        else:
            global_max = pl.load(
                mi_out_iter,
                [0, 0],
                [Q_TILE, 1],
                target_memory=pl.MemorySpace.Vec,
            )
            gm_nd = pl.reshape(global_max, [1, Q_TILE])
            lm_nd = pl.reshape(local_max, [1, Q_TILE])
            new_max = pl.reshape(pl.maximum(gm_nd, lm_nd), [Q_TILE, 1])
            mi_out_updated: pl.Tensor[[Q_TILE, 1], pl.FP32] = pl.store(new_max, [0, 0], mi_out_iter)
        (mi_out_carry,) = pl.yield_(mi_out_updated)

    # Pass 2: exp(s - global_max), cast to bf16, row_sum accumulation
    # NOTE: use distinct variable names (s_tile_p2, scaled_p2, tmp_tile_p2)
    # to avoid SSA creating a carry chain from loop 1 → loop 2.
    for i, (pij_buf_iter, li_out_iter) in pl.range(n_blocks, init_values=(pij_buf, li_out)):
        global_max = pl.load(
            mi_out_carry,
            [0, 0],
            [Q_TILE, 1],
            target_memory=pl.MemorySpace.Vec,
        )
        s_tile_p2 = pl.load(
            sij_buf,
            [i * Q_TILE, 0],
            [Q_TILE, BLOCK_SIZE],
            target_memory=pl.MemorySpace.Vec,
        )
        scaled_p2 = pl.mul(s_tile_p2, scale)
        centered = pl.row_expand_sub(scaled_p2, global_max)
        exp_tile = pl.exp(centered)
        pij_bf16 = pl.cast(exp_tile, target_type=pl.BF16)
        pij_f32 = pl.cast(pij_bf16, target_type=pl.FP32)
        pij_buf_updated = pl.store(pij_bf16, [i * Q_TILE, 0], pij_buf_iter)

        tmp_tile_p2 = pl.create_tile(
            [Q_TILE, BLOCK_SIZE],
            dtype=pl.FP32,
            target_memory=pl.MemorySpace.Vec,
        )
        li_local = pl.row_sum(pij_f32, tmp_tile_p2)
        li_local_nd = pl.reshape(li_local, [1, Q_TILE])

        if i == 0:
            li_out_updated: pl.Tensor[[Q_TILE, 1], pl.FP32] = pl.store(li_local, [0, 0], li_out_iter)
        else:
            li_acc = pl.load(li_out_iter, [0, 0], [Q_TILE, 1])
            li_acc_nd = pl.reshape(li_acc, [1, Q_TILE])
            li_sum = pl.reshape(pl.add(li_acc_nd, li_local_nd), [Q_TILE, 1])
            li_out_updated: pl.Tensor[[Q_TILE, 1], pl.FP32] = pl.store(li_sum, [0, 0], li_out_iter)
        pij_buf_carry, li_out_carry = pl.yield_(pij_buf_updated, li_out_updated)

    return pij_buf_carry, mi_out_carry, li_out_carry


@pl.function(type=pl.FunctionType.InCore)
def kernel_online_update_64(  # noqa: PLR0913
    mij: pl.Tensor[[Q_TILE, 1], pl.FP32],
    lij: pl.Tensor[[Q_TILE, 1], pl.FP32],
    oi_new: pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32],
    mi: pl.InOut[pl.Tensor[[Q_TILE, 1], pl.FP32]],
    li: pl.InOut[pl.Tensor[[Q_TILE, 1], pl.FP32]],
    oi: pl.InOut[pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32]],
    dst: pl.Out[pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32]],
    is_first: pl.Scalar[pl.INDEX],
    is_last: pl.Scalar[pl.INDEX],
) -> tuple[
    pl.Tensor[[Q_TILE, 1], pl.FP32],
    pl.Tensor[[Q_TILE, 1], pl.FP32],
    pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32],
    pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32],
]:
    """Online softmax update with inplace mi/li/oi (VECTOR).

    Merges current group's (mij, lij, oi_new) into running accumulators
    (mi, li, oi). On last iteration, writes normalised output to dst.
    """
    mij_tile = pl.load(mij, [0, 0], [Q_TILE, 1], target_memory=pl.MemorySpace.Vec)
    lij_tile = pl.load(lij, [0, 0], [Q_TILE, 1], target_memory=pl.MemorySpace.Vec)
    oi_new_tile = pl.load(oi_new, [0, 0], [Q_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec)
    mi_tile = pl.load(mi, [0, 0], [Q_TILE, 1], target_memory=pl.MemorySpace.Vec)
    li_tile = pl.load(li, [0, 0], [Q_TILE, 1], target_memory=pl.MemorySpace.Vec)
    oi_tile = pl.load(oi, [0, 0], [Q_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec)

    if is_first == 1:
        mi_out = pl.store(mij_tile, [0, 0], mi)
        li_out = pl.store(lij_tile, [0, 0], li)
        oi_out = pl.store(oi_new_tile, [0, 0], oi)
        if is_last == 1:
            dst_tile = pl.row_expand_div(oi_new_tile, lij_tile)
            dst_out = pl.store(dst_tile, [0, 0], dst)
        else:
            zero_tile = pl.tile.full([Q_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
            dst_out = pl.store(zero_tile, [0, 0], dst)
    else:
        # Reshape DN [Q_TILE,1] → ND [1,Q_TILE] for element-wise ops
        mi_tile_nd = pl.reshape(mi_tile, [1, Q_TILE])
        mij_tile_nd = pl.reshape(mij_tile, [1, Q_TILE])
        li_tile_nd = pl.reshape(li_tile, [1, Q_TILE])
        lij_tile_nd = pl.reshape(lij_tile, [1, Q_TILE])

        mi_new = pl.maximum(mi_tile_nd, mij_tile_nd)
        mi_diff = pl.sub(mi_tile_nd, mi_new)
        alpha = pl.exp(mi_diff)
        mij_diff = pl.sub(mij_tile_nd, mi_new)
        beta = pl.exp(mij_diff)

        li_scaled = pl.mul(alpha, li_tile_nd)
        lij_scaled = pl.mul(beta, lij_tile_nd)
        li_updated = pl.add(li_scaled, lij_scaled)

        alpha_dn = pl.reshape(alpha, [Q_TILE, 1])
        oi_scaled = pl.row_expand_mul(oi_tile, alpha_dn)
        beta_dn = pl.reshape(beta, [Q_TILE, 1])
        oi_new_scaled = pl.row_expand_mul(oi_new_tile, beta_dn)
        oi_updated = pl.add(oi_scaled, oi_new_scaled)

        mi_new_dn = pl.reshape(mi_new, [Q_TILE, 1])
        li_updated_dn = pl.reshape(li_updated, [Q_TILE, 1])

        mi_out = pl.store(mi_new_dn, [0, 0], mi)
        li_out = pl.store(li_updated_dn, [0, 0], li)

        if is_last == 1:
            dst_tile = pl.row_expand_div(oi_updated, li_updated_dn)
            dst_out = pl.store(dst_tile, [0, 0], dst)
            oi_out = pl.store(oi_updated, [0, 0], oi)
        else:
            zero_tile = pl.tile.full([Q_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
            dst_out = pl.store(zero_tile, [0, 0], dst)
            oi_out = pl.store(oi_updated, [0, 0], oi)

    return mi_out, li_out, oi_out, dst_out


# ── Factory functions for batch-dynamic kernels ───────────────────────────────


def make_kernel_qk_matmul_64(key_cache_rows: int, block_table_flat_size: int):
    """Create a multi-block QK matmul InCore kernel for given cache dimensions.

    Parameters
    ----------
    key_cache_rows:        total rows in the key cache (batch * max_blocks * block_size)
    block_table_flat_size: total entries in the block table (batch * max_blocks)
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_qk_matmul_64(
        qi: pl.Tensor[[Q_TILE, HEAD_DIM], pl.BF16],
        key_cache: pl.Tensor[[key_cache_rows, HEAD_DIM], pl.BF16],
        sij_buf: pl.Out[pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.FP32]],
        block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
        n_blocks: pl.Scalar[pl.INDEX],
        bt_offset: pl.Scalar[pl.INT64],
    ) -> pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.FP32]:
        """Multi-block QK matmul: sij[i] = qi @ kj[i].T, vertically stacked (CUBE).

        Loops over n_blocks, looking up physical block indices via block_table.
        key_cache is stored as (rows, head_dim); transpose at load to get (head_dim, block_size).
        """
        for i, (sij_buf_iter,) in pl.range(n_blocks, init_values=(sij_buf,)):
            phys_block = pl.read(block_table, pl.cast(bt_offset, pl.INDEX) + i)

            kj_row = phys_block * BLOCK_SIZE

            qi_l1 = pl.load(
                qi,
                [0, 0],
                [Q_TILE, HEAD_DIM],
                target_memory=pl.MemorySpace.Mat,
            )
            kj_l1 = pl.load(
                key_cache,
                [0, kj_row],
                [HEAD_DIM, BLOCK_SIZE],
                target_memory=pl.MemorySpace.Mat,
                transpose=True,
            )
            qi_l0a = pl.move(qi_l1, target_memory=pl.MemorySpace.Left)
            kj_l0b = pl.move(kj_l1, target_memory=pl.MemorySpace.Right)
            sij_l0c = pl.matmul(qi_l0a, kj_l0b)
            sij_buf_updated = pl.store(sij_l0c, [i * Q_TILE, 0], sij_buf_iter)
            (sij_buf_out,) = pl.yield_(sij_buf_updated)
        return sij_buf_out

    return kernel_qk_matmul_64


def make_kernel_pv_matmul_64(key_cache_rows: int, block_table_flat_size: int):
    """Create a SplitK PV matmul InCore kernel for given cache dimensions.

    Parameters
    ----------
    key_cache_rows:        total rows in the value cache (batch * max_blocks * block_size)
    block_table_flat_size: total entries in the block table (batch * max_blocks)
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_pv_matmul_64(
        pij_buf: pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.BF16],
        value_cache: pl.Tensor[[key_cache_rows, HEAD_DIM], pl.BF16],
        oi_new: pl.Out[pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32]],
        block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
        n_blocks: pl.Scalar[pl.INDEX],
        bt_offset: pl.Scalar[pl.INT64],
    ) -> pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32]:
        """SplitK PV matmul: first block via matmul, rest via matmul_acc (CUBE).

        Accumulates pij[i] @ vj[i] across n_blocks on L0C, then stores result.
        """
        # First block: matmul (creates L0C accumulator)
        phys_block_0 = pl.read(block_table, pl.cast(bt_offset, pl.INDEX))
        vj_row_0 = phys_block_0 * BLOCK_SIZE

        pij_l1 = pl.load(
            pij_buf,
            [0, 0],
            [Q_TILE, BLOCK_SIZE],
            target_memory=pl.MemorySpace.Mat,
        )
        vj_l1 = pl.load(
            value_cache,
            [vj_row_0, 0],
            [BLOCK_SIZE, HEAD_DIM],
            target_memory=pl.MemorySpace.Mat,
        )
        pij_l0a = pl.move(pij_l1, target_memory=pl.MemorySpace.Left)
        vj_l0b = pl.move(vj_l1, target_memory=pl.MemorySpace.Right)
        oi_l0c = pl.matmul(pij_l0a, vj_l0b)

        # Remaining blocks: matmul_acc (accumulate onto L0C)
        for i, (oi_l0c_iter,) in pl.range(1, n_blocks, init_values=(oi_l0c,)):
            phys_block = pl.read(block_table, pl.cast(bt_offset, pl.INDEX) + i)
            vj_row = phys_block * BLOCK_SIZE

            pij_l1 = pl.load(
                pij_buf,
                [i * Q_TILE, 0],
                [Q_TILE, BLOCK_SIZE],
                target_memory=pl.MemorySpace.Mat,
            )
            vj_l1 = pl.load(
                value_cache,
                [vj_row, 0],
                [BLOCK_SIZE, HEAD_DIM],
                target_memory=pl.MemorySpace.Mat,
            )
            pij_l0a = pl.move(pij_l1, target_memory=pl.MemorySpace.Left)
            vj_l0b = pl.move(vj_l1, target_memory=pl.MemorySpace.Right)
            oi_l0c_acc = pl.matmul_acc(oi_l0c_iter, pij_l0a, vj_l0b)
            (oi_l0c_out,) = pl.yield_(oi_l0c_acc)

        oi_new = pl.store(oi_l0c_out, [0, 0], oi_new)
        return oi_new

    return kernel_pv_matmul_64


# ── Program builder ──────────────────────────────────────────────────────────


def build_paged_attention_64_program(
    batch: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_num_blocks_per_req: int,
    context_len: int,
    q_tile: int = Q_TILE,
):
    """Build paged-attention @pl.program with 64divisions (q_tile=64, block_size=64).

    Uses single-loop N_UNROLL=64 structure matching the C++ paged_attention_unroll
    pattern: one loop processing up to N_UNROLL blocks per group with generic
    multi-block kernels.

    Parameters
    ----------
    batch:                  number of requests in the batch
    num_heads:              number of query heads (typically 64 for this config)
    head_dim:               per-head feature dimension (128)
    block_size:             KV-cache block size (64)
    max_num_blocks_per_req: maximum number of KV blocks per request
    q_tile:                 query-head tile size (64)
    """
    query_rows = batch * num_heads
    key_cache_rows = batch * max_num_blocks_per_req * block_size
    out_rows = batch * num_heads
    block_table_flat_size = batch * max_num_blocks_per_req
    q_loop_static = (num_heads + q_tile - 1) // q_tile
    max_bn = (context_len + block_size - 1) // block_size

    kernel_qk = make_kernel_qk_matmul_64(key_cache_rows, block_table_flat_size)
    kernel_pv = make_kernel_pv_matmul_64(key_cache_rows, block_table_flat_size)

    @pl.program
    class PagedAttention64Program:
        """Paged attention 64divisions with N_UNROLL=64 single-loop structure."""

        @pl.function(type=pl.FunctionType.Orchestration)
        def paged_attention(
            self,
            query: pl.Tensor[[query_rows, head_dim], pl.BF16],
            key_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
            value_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
            block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
            context_lens: pl.Tensor[[batch], pl.INT32],
            out: pl.Tensor[[out_rows, head_dim], pl.FP32],
            config: pl.Tensor[[7], pl.INT64],
            size_query: pl.Tensor[[1], pl.INT64],
            size_key_cache: pl.Tensor[[1], pl.INT64],
            size_value_cache: pl.Tensor[[1], pl.INT64],
        ) -> pl.Tensor[[out_rows, head_dim], pl.FP32]:
            """Paged attention orchestration with single N_UNROLL loop.

            Config: [batch, num_heads, kv_head_num, head_dim, block_size, block_num, scale_bits]
            """
            for b_idx in pl.range(batch):
                for q_idx in pl.range(q_loop_static):
                    cur_offset = b_idx * num_heads + q_idx * q_tile

                    oi: pl.Tensor[[q_tile, head_dim], pl.FP32] = pl.create_tensor(
                        [q_tile, head_dim],  # type: ignore[reportArgumentType]
                        dtype=pl.FP32,
                    )
                    li_update: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor(
                        [q_tile, 1],
                        dtype=pl.FP32,  # type: ignore[reportArgumentType]
                    )
                    mi_update: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor(
                        [q_tile, 1],
                        dtype=pl.FP32,  # type: ignore[reportArgumentType]
                    )
                    oi, li_update, mi_update = kernel_aiv_hub(oi, li_update, mi_update)

                    qi: pl.Tensor[[q_tile, head_dim], pl.BF16] = pl.slice(
                        query,
                        [q_tile, head_dim],  # type: ignore[reportArgumentType]
                        [cur_offset, 0],
                    )

                    # ── Single N_UNROLL loop over KV blocks ──────────
                    for bn in pl.range(0, max_bn, N_UNROLL):  # type: ignore[reportArgumentType]
                        n_blocks = pl.min(N_UNROLL, max_bn - bn)  # type: ignore[reportArgumentType]
                        bt_offset = b_idx * max_num_blocks_per_req + bn
                        # valid_len_last = pl.min(  # unused when context_len % 4096 == 0
                        #     block_size,
                        #     cur_seq - (bn + n_blocks - 1) * block_size,
                        # )

                        # 1. QK matmul (CUBE)
                        sij_buf: pl.Tensor[[N_UNROLL_Q, block_size], pl.FP32] = pl.create_tensor(
                            [N_UNROLL_Q, block_size],  # type: ignore[reportArgumentType]
                            dtype=pl.FP32,
                        )
                        sij_buf = kernel_qk(
                            qi,
                            key_cache,
                            sij_buf,
                            block_table,
                            n_blocks,
                            bt_offset,
                        )

                        # 2. Softmax prepare (VECTOR)
                        pij_buf: pl.Tensor[[N_UNROLL_Q, block_size], pl.BF16] = pl.create_tensor(
                            [N_UNROLL_Q, block_size],  # type: ignore[reportArgumentType]
                            dtype=pl.BF16,
                        )
                        mi: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor(
                            [q_tile, 1],
                            dtype=pl.FP32,  # type: ignore[reportArgumentType]
                        )
                        li: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor(
                            [q_tile, 1],
                            dtype=pl.FP32,  # type: ignore[reportArgumentType]
                        )
                        pij_buf, mi, li = kernel_softmax_prepare_64(
                            sij_buf,
                            1.0,  # type: ignore[reportArgumentType]
                            pij_buf,
                            mi,
                            li,
                            n_blocks,  # type: ignore[reportArgumentType]
                        )

                        # 3. PV matmul (CUBE)
                        oi_new: pl.Tensor[[q_tile, head_dim], pl.FP32] = pl.create_tensor(
                            [q_tile, head_dim],  # type: ignore[reportArgumentType]
                            dtype=pl.FP32,
                        )
                        oi_new = kernel_pv(
                            pij_buf,
                            value_cache,
                            oi_new,
                            block_table,
                            n_blocks,
                            bt_offset,
                        )

                        # 4. Online update flags
                        if bn == 0:
                            is_first: pl.Scalar[pl.INT64] = pl.yield_(1)
                        else:
                            is_first: pl.Scalar[pl.INT64] = pl.yield_(0)
                        if bn + n_blocks == max_bn:
                            is_last: pl.Scalar[pl.INT64] = pl.yield_(1)
                        else:
                            is_last: pl.Scalar[pl.INT64] = pl.yield_(0)

                        # 5. Online update (VECTOR)
                        out_view: pl.Tensor[[q_tile, head_dim], pl.FP32] = pl.slice(
                            out,
                            [q_tile, head_dim],  # type: ignore[reportArgumentType]
                            [cur_offset, 0],
                        )
                        mi_update, li_update, oi, out_view = kernel_online_update_64(
                            mi,
                            li,
                            oi_new,
                            mi_update,
                            li_update,
                            oi,
                            out_view,
                            is_first,
                            is_last,
                        )

            return out

    return PagedAttention64Program


# ── Golden reference ─────────────────────────────────────────────────────────


def golden_64(tensors: dict, params: dict | None = None) -> None:
    """Golden reference for 64divisions paged attention with N_UNROLL grouping.

    Mirrors the single-loop N_UNROLL structure of the orchestration:
    each group of up to N_UNROLL blocks uses a two-pass softmax (global
    row_max across all blocks in the group, then exp with that max).
    """
    config = tensors["config"]
    batch = int(config[0].item())
    num_heads = int(config[1].item())
    head_dim = int(config[3].item())
    block_size = int(config[4].item())
    max_num_blocks_per_req = int(config[5].item())
    scale_bits = int(config[6].item())
    scale = struct.unpack("f", struct.pack("I", scale_bits & 0xFFFFFFFF))[0]

    query = tensors["query"].float().reshape(batch, num_heads, head_dim)
    total_pool_blocks = batch * max_num_blocks_per_req
    key_cache = tensors["key_cache"].float().reshape(total_pool_blocks, block_size, head_dim)
    value_cache = tensors["value_cache"].float().reshape(total_pool_blocks, block_size, head_dim)
    block_table = tensors["block_table"].reshape(batch, max_num_blocks_per_req)
    context_lens = tensors["context_lens"]

    out = torch.zeros((batch, num_heads, head_dim), dtype=torch.float32)
    q_tile = Q_TILE

    def _update(
        oi_a: torch.Tensor | None,
        li_a: torch.Tensor | None,
        mi_a: torch.Tensor | None,
        oi_new: torch.Tensor,
        li_new: torch.Tensor,
        mi_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Online softmax update."""
        if oi_a is None or li_a is None or mi_a is None:
            return oi_new, li_new, mi_new
        mi_u = torch.maximum(mi_a, mi_new)
        a = torch.exp(mi_a - mi_u)
        b_ = torch.exp(mi_new - mi_u)
        return a * oi_a + b_ * oi_new, a * li_a + b_ * li_new, mi_u

    for b in range(batch):
        cur_seq = int(context_lens[b].item())
        max_bn_b = (cur_seq + block_size - 1) // block_size

        for q_idx in range(num_heads // q_tile):
            q_off = q_idx * q_tile
            qi = query[b, q_off : q_off + q_tile, :]

            oi_acc, li_acc, mi_acc = None, None, None

            for bn in range(0, max_bn_b, N_UNROLL):
                n_blocks = min(N_UNROLL, max_bn_b - bn)

                # QK matmul for each block in the group
                all_sij = []
                for i in range(n_blocks):
                    v = min(block_size, cur_seq - (bn + i) * block_size)
                    bidx = int(block_table[b, bn + i].item())
                    kj = key_cache[bidx, :v]
                    sij = torch.mm(qi, kj.T) * scale
                    all_sij.append(sij)

                # Two-pass softmax: global row_max across all blocks in group
                global_max = all_sij[0].max(dim=-1, keepdim=True)[0]
                for sij in all_sij[1:]:
                    local_max = sij.max(dim=-1, keepdim=True)[0]
                    global_max = torch.maximum(global_max, local_max)
                global_max = global_max.clamp(min=-1e30)

                # Exp with global max, sum, PV matmul
                li_group = torch.zeros(q_tile, 1)
                oi_group = torch.zeros(q_tile, head_dim, dtype=torch.float32)
                for i, sij in enumerate(all_sij):
                    pij = torch.exp(sij - global_max).to(torch.bfloat16).to(torch.float32)
                    li_group += pij.sum(dim=-1, keepdim=True)
                    v = min(block_size, cur_seq - (bn + i) * block_size)
                    bidx = int(block_table[b, bn + i].item())
                    vj = value_cache[bidx, :v]
                    oi_group += torch.mm(pij, vj)

                # Online update
                oi_acc, li_acc, mi_acc = _update(oi_acc, li_acc, mi_acc, oi_group, li_group, global_max)

            assert oi_acc is not None and li_acc is not None, f"No valid blocks for b={b} q={q_off}"
            out[b, q_off : q_off + q_tile, :] = oi_acc / li_acc

    tensors["out"][:] = out.reshape(batch * num_heads, head_dim)


# ── TensorSpec builder ───────────────────────────────────────────────────────


def build_tensor_specs_64(
    batch: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_num_blocks_per_req: int,
    context_len: int,
    scale: float = 1.0,
) -> list[TensorSpec]:
    """Build TensorSpec list for 64divisions paged attention."""
    query_rows = batch * num_heads
    key_cache_rows = batch * max_num_blocks_per_req * block_size
    block_table_flat_size = batch * max_num_blocks_per_req

    scale_bits = struct.unpack("I", struct.pack("f", scale))[0]
    config_data = torch.tensor(
        [batch, num_heads, 1, head_dim, block_size, max_num_blocks_per_req, scale_bits],
        dtype=torch.int64,
    )
    context_lens_data = torch.full((batch,), context_len, dtype=torch.int32)
    block_table_data = torch.randint(
        0, max(block_table_flat_size, 1), size=(batch, max_num_blocks_per_req), dtype=torch.int32
    ).flatten()

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


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    batch = 64
    num_heads = 64
    head_dim = HEAD_DIM
    block_size = BLOCK_SIZE
    max_model_len = 32768
    context_len = 8192
    scale = 1.0
    max_num_blocks_per_req = max_model_len // block_size  # 512

    program = build_paged_attention_64_program(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks_per_req=max_num_blocks_per_req,
        context_len=context_len,
    )

    tensor_specs = build_tensor_specs_64(
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
        golden=golden_64,
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
