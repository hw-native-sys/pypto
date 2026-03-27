# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Paged Attention Multi-Config Example

Generates output compatible with multi-config paged_attention_unroll test interface:
- N_UNROLL = 8 (matching multi-config)
- Orchestration pre-extracts block_indices via pl.slice() from block_table
- Kernels receive block_indices tensor view (no block_table + bt_offset indirection)
- Golden function uses N_UNROLL=8 grouping with two-pass softmax

Key interface difference vs multi-config:
  multi-config passes 8 individual scalar block_indices to each kernel.
  PyPTO passes a single block_indices tensor view (functionally equivalent —
  same data access pattern: block_indices[i] → physical block index).

Tile dimensions (matching multi-config Case2, q_tile=16):
  QK Matmul:       qi(16, 128) @ kj.T(128, 64) → sij(16, 64)
  Softmax:         sij(16, N) → pij(16, N) bf16, mi(16, 1), li(16, 1)
  PV Matmul:       pij(16, 64) @ vj(64, 128) → oi(16, 128)
  Online Update:   operates on (16, 128) data tiles, (16, 1) scalar tiles

Module-level InCore kernels (reusable, importable):
  kernel_aiv_hub, kernel_softmax_prepare, kernel_online_update

Factory functions for batch-dynamic kernels:
  make_kernel_qk_matmul(key_cache_rows)
  make_kernel_pv_matmul(key_cache_rows)
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
N_UNROLL = 64
N_UNROLL_Q = N_UNROLL * Q_TILE  # 128 — static sij/pij buffer height


# ── Kernel factory functions ──────────────────────────────────────────────────


def make_kernel_aiv_hub(q_tile: int, head_dim: int):
    """Create aiv_hub InCore kernel with parameterised tile dimensions."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_aiv_hub(
        oi: pl.Out[pl.Tensor[[q_tile, head_dim], pl.FP32]],
        li: pl.Out[pl.Tensor[[q_tile, 1], pl.FP32]],
        mi: pl.Out[pl.Tensor[[q_tile, 1], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[q_tile, head_dim], pl.FP32],
        pl.Tensor[[q_tile, 1], pl.FP32],
        pl.Tensor[[q_tile, 1], pl.FP32],
    ]:
        """Initialize inplace accumulators to zero (VECTOR)."""
        return oi, li, mi

    return kernel_aiv_hub


def make_kernel_softmax_prepare(q_tile: int, block_size: int, n_unroll_q: int):
    """Create softmax_prepare InCore kernel with parameterised tile dimensions."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_softmax_prepare(
        sij_buf: pl.Tensor[[n_unroll_q, block_size], pl.FP32],
        scale: pl.Scalar[pl.FP32],
        pij_buf: pl.Out[pl.Tensor[[n_unroll_q, block_size], pl.BF16]],
        mi_out: pl.Out[pl.Tensor[[q_tile, 1], pl.FP32]],
        li_out: pl.Out[pl.Tensor[[q_tile, 1], pl.FP32]],
        n_blocks: pl.Scalar[pl.INDEX],
    ) -> tuple[
        pl.Tensor[[n_unroll_q, block_size], pl.BF16],
        pl.Tensor[[q_tile, 1], pl.FP32],
        pl.Tensor[[q_tile, 1], pl.FP32],
    ]:
        """Two-pass softmax: pass 1 finds global row_max, pass 2 computes exp+sum (VECTOR).

        Uses mi_out/li_out as GM scratch for cross-iteration state via store/load round-trips.
        """
        # Pass 1: find global row_max across all blocks
        for i, (mi_out_iter,) in pl.range(n_blocks, init_values=(mi_out,)):
            s_tile = pl.load(
                sij_buf,
                [i * q_tile, 0],
                [q_tile, block_size],
                target_memory=pl.MemorySpace.Vec,
            )
            scaled = pl.mul(s_tile, scale)
            tmp_tile = pl.create_tile(
                [q_tile, block_size],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            local_max = pl.row_max(scaled, tmp_tile)

            if i == 0:
                mi_out_updated: pl.Tensor[[q_tile, 1], pl.FP32] = pl.store(local_max, [0, 0], mi_out_iter)
            else:
                global_max = pl.load(
                    mi_out_iter,
                    [0, 0],
                    [q_tile, 1],
                    target_memory=pl.MemorySpace.Vec,
                )
                gm_nd = pl.reshape(global_max, [1, q_tile])
                lm_nd = pl.reshape(local_max, [1, q_tile])
                new_max = pl.reshape(pl.maximum(gm_nd, lm_nd), [q_tile, 1])
                mi_out_updated: pl.Tensor[[q_tile, 1], pl.FP32] = pl.store(new_max, [0, 0], mi_out_iter)
            (mi_out_carry,) = pl.yield_(mi_out_updated)

        # Pass 2: exp(s - global_max), cast to bf16, row_sum accumulation
        for i, (pij_buf_iter, li_out_iter) in pl.range(n_blocks, init_values=(pij_buf, li_out)):
            global_max = pl.load(
                mi_out_carry,
                [0, 0],
                [q_tile, 1],
                target_memory=pl.MemorySpace.Vec,
            )
            s_tile_p2 = pl.load(
                sij_buf,
                [i * q_tile, 0],
                [q_tile, block_size],
                target_memory=pl.MemorySpace.Vec,
            )
            scaled_p2 = pl.mul(s_tile_p2, scale)
            centered = pl.row_expand_sub(scaled_p2, global_max)
            exp_tile = pl.exp(centered)
            pij_bf16 = pl.cast(exp_tile, target_type=pl.BF16)
            pij_f32 = pl.cast(pij_bf16, target_type=pl.FP32)
            pij_buf_updated = pl.store(pij_bf16, [i * q_tile, 0], pij_buf_iter)

            tmp_tile_p2 = pl.create_tile(
                [q_tile, block_size],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            li_local = pl.row_sum(pij_f32, tmp_tile_p2)
            li_local_nd = pl.reshape(li_local, [1, q_tile])

            if i == 0:
                li_out_updated: pl.Tensor[[q_tile, 1], pl.FP32] = pl.store(li_local, [0, 0], li_out_iter)
            else:
                li_acc = pl.load(li_out_iter, [0, 0], [q_tile, 1])
                li_acc_nd = pl.reshape(li_acc, [1, q_tile])
                li_sum = pl.reshape(pl.add(li_acc_nd, li_local_nd), [q_tile, 1])
                li_out_updated: pl.Tensor[[q_tile, 1], pl.FP32] = pl.store(li_sum, [0, 0], li_out_iter)
            pij_buf_carry, li_out_carry = pl.yield_(pij_buf_updated, li_out_updated)

        return pij_buf_carry, mi_out_carry, li_out_carry

    return kernel_softmax_prepare


def make_kernel_online_update(q_tile: int, head_dim: int):
    """Create online_update InCore kernel with parameterised tile dimensions."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_online_update(  # noqa: PLR0913
        mij: pl.Tensor[[q_tile, 1], pl.FP32],
        lij: pl.Tensor[[q_tile, 1], pl.FP32],
        oi_new: pl.Tensor[[q_tile, head_dim], pl.FP32],
        mi: pl.InOut[pl.Tensor[[q_tile, 1], pl.FP32]],
        li: pl.InOut[pl.Tensor[[q_tile, 1], pl.FP32]],
        oi: pl.InOut[pl.Tensor[[q_tile, head_dim], pl.FP32]],
        dst: pl.Out[pl.Tensor[[q_tile, head_dim], pl.FP32]],
        is_first: pl.Scalar[pl.INDEX],
        is_last: pl.Scalar[pl.INDEX],
    ) -> tuple[
        pl.Tensor[[q_tile, 1], pl.FP32],
        pl.Tensor[[q_tile, 1], pl.FP32],
        pl.Tensor[[q_tile, head_dim], pl.FP32],
        pl.Tensor[[q_tile, head_dim], pl.FP32],
    ]:
        """Online softmax update with inplace mi/li/oi (VECTOR).

        Merges current group's (mij, lij, oi_new) into running accumulators
        (mi, li, oi). On last iteration, writes normalised output to dst.
        """
        mij_tile = pl.load(mij, [0, 0], [q_tile, 1], target_memory=pl.MemorySpace.Vec)
        lij_tile = pl.load(lij, [0, 0], [q_tile, 1], target_memory=pl.MemorySpace.Vec)
        oi_new_tile = pl.load(oi_new, [0, 0], [q_tile, head_dim], target_memory=pl.MemorySpace.Vec)
        mi_tile = pl.load(mi, [0, 0], [q_tile, 1], target_memory=pl.MemorySpace.Vec)
        li_tile = pl.load(li, [0, 0], [q_tile, 1], target_memory=pl.MemorySpace.Vec)
        oi_tile = pl.load(oi, [0, 0], [q_tile, head_dim], target_memory=pl.MemorySpace.Vec)

        if is_first == 1:
            mi_out = pl.store(mij_tile, [0, 0], mi)
            li_out = pl.store(lij_tile, [0, 0], li)
            oi_out = pl.store(oi_new_tile, [0, 0], oi)
            if is_last == 1:
                dst_tile = pl.row_expand_div(oi_new_tile, lij_tile)
                dst_out = pl.store(dst_tile, [0, 0], dst)
            else:
                zero_tile = pl.tile.full([q_tile, head_dim], dtype=pl.FP32, value=0.0)
                dst_out = pl.store(zero_tile, [0, 0], dst)
        else:
            # Reshape DN [q_tile,1] -> ND [1,q_tile] for element-wise ops
            mi_tile_nd = pl.reshape(mi_tile, [1, q_tile])
            mij_tile_nd = pl.reshape(mij_tile, [1, q_tile])
            li_tile_nd = pl.reshape(li_tile, [1, q_tile])
            lij_tile_nd = pl.reshape(lij_tile, [1, q_tile])

            mi_new = pl.maximum(mi_tile_nd, mij_tile_nd)
            mi_diff = pl.sub(mi_tile_nd, mi_new)
            alpha = pl.exp(mi_diff)
            mij_diff = pl.sub(mij_tile_nd, mi_new)
            beta = pl.exp(mij_diff)

            li_scaled = pl.mul(alpha, li_tile_nd)
            lij_scaled = pl.mul(beta, lij_tile_nd)
            li_updated = pl.add(li_scaled, lij_scaled)

            alpha_dn = pl.reshape(alpha, [q_tile, 1])
            oi_scaled = pl.row_expand_mul(oi_tile, alpha_dn)
            beta_dn = pl.reshape(beta, [q_tile, 1])
            oi_new_scaled = pl.row_expand_mul(oi_new_tile, beta_dn)
            oi_updated = pl.add(oi_scaled, oi_new_scaled)

            mi_new_dn = pl.reshape(mi_new, [q_tile, 1])
            li_updated_dn = pl.reshape(li_updated, [q_tile, 1])

            mi_out = pl.store(mi_new_dn, [0, 0], mi)
            li_out = pl.store(li_updated_dn, [0, 0], li)

            oi_out = pl.store(oi_updated, [0, 0], oi)
            if is_last == 1:
                dst_tile = pl.row_expand_div(oi_updated, li_updated_dn)
                dst_out = pl.store(dst_tile, [0, 0], dst)
            else:
                zero_tile = pl.tile.full([q_tile, head_dim], dtype=pl.FP32, value=0.0)
                dst_out = pl.store(zero_tile, [0, 0], dst)

        return mi_out, li_out, oi_out, dst_out

    return kernel_online_update


# ── Module-level kernel instances (backward-compatible imports) ───────────────
kernel_aiv_hub = make_kernel_aiv_hub(Q_TILE, HEAD_DIM)
kernel_softmax_prepare = make_kernel_softmax_prepare(Q_TILE, BLOCK_SIZE, N_UNROLL_Q)
kernel_online_update = make_kernel_online_update(Q_TILE, HEAD_DIM)


# ── Factory functions for multi-config kernels ──────────────────────────


def make_kernel_qk_matmul(
    key_cache_rows: int,
    q_tile: int = Q_TILE,
    head_dim: int = HEAD_DIM,
    block_size: int = BLOCK_SIZE,
    n_unroll: int = N_UNROLL,
    n_unroll_q: int = N_UNROLL_Q,
):
    """Create a multi-block QK matmul InCore kernel using pre-extracted block indices.

    Multi-config: receives block_indices tensor (pre-sliced by orchestration)
    instead of full block_table + bt_offset.

    Parameters
    ----------
    key_cache_rows: total rows in the key cache (batch * max_blocks * block_size)
    q_tile: query tile height
    head_dim: per-head feature dimension
    block_size: KV-cache block size
    n_unroll: number of blocks per unroll group
    n_unroll_q: n_unroll * q_tile (static buffer height)
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_qk_matmul(
        qi: pl.Tensor[[q_tile, head_dim], pl.BF16],
        key_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
        sij_buf: pl.Out[pl.Tensor[[n_unroll_q, block_size], pl.FP32]],
        block_indices: pl.Tensor[[n_unroll], pl.INT32],
        n_blocks: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[n_unroll_q, block_size], pl.FP32]:
        """Multi-block QK matmul: sij[i] = qi @ kj[i].T, vertically stacked (CUBE).

        Loops over n_blocks, looking up physical block indices via block_indices
        (pre-extracted by orchestration from block_table).
        key_cache is stored as (rows, head_dim); transpose at load to get (head_dim, block_size).
        """
        for i, (sij_buf_iter,) in pl.range(n_blocks, init_values=(sij_buf,)):
            phys_block = pl.read(block_indices, i)

            kj_row = phys_block * block_size

            qi_l1 = pl.load(
                qi,
                [0, 0],
                [q_tile, head_dim],
                target_memory=pl.MemorySpace.Mat,
            )
            kj_l1 = pl.load(
                key_cache,
                [kj_row, 0],
                [head_dim, block_size],
                target_memory=pl.MemorySpace.Mat,
                transpose=True,
            )
            qi_l0a = pl.move(qi_l1, target_memory=pl.MemorySpace.Left)
            kj_l0b = pl.move(kj_l1, target_memory=pl.MemorySpace.Right)
            sij_l0c = pl.matmul(qi_l0a, kj_l0b)
            sij_buf_updated = pl.store(sij_l0c, [i * q_tile, 0], sij_buf_iter)
            (sij_buf_out,) = pl.yield_(sij_buf_updated)
        return sij_buf_out

    return kernel_qk_matmul


def make_kernel_pv_matmul(
    key_cache_rows: int,
    q_tile: int = Q_TILE,
    head_dim: int = HEAD_DIM,
    block_size: int = BLOCK_SIZE,
    n_unroll: int = N_UNROLL,
    n_unroll_q: int = N_UNROLL_Q,
):
    """Create a SplitK PV matmul InCore kernel using pre-extracted block indices.

    Multi-config: receives block_indices tensor (pre-sliced by orchestration)
    instead of full block_table + bt_offset.

    Parameters
    ----------
    key_cache_rows: total rows in the value cache (batch * max_blocks * block_size)
    q_tile: query tile height
    head_dim: per-head feature dimension
    block_size: KV-cache block size
    n_unroll: number of blocks per unroll group
    n_unroll_q: n_unroll * q_tile (static buffer height)
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_pv_matmul(
        pij_buf: pl.Tensor[[n_unroll_q, block_size], pl.BF16],
        value_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
        oi_new: pl.Out[pl.Tensor[[q_tile, head_dim], pl.FP32]],
        block_indices: pl.Tensor[[n_unroll], pl.INT32],
        n_blocks: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[q_tile, head_dim], pl.FP32]:
        """SplitK PV matmul: first block via matmul, rest via matmul_acc (CUBE).

        Accumulates pij[i] @ vj[i] across n_blocks on L0C, then stores result.
        block_indices pre-extracted by orchestration from block_table.
        """
        # First block: matmul (creates L0C accumulator)
        # Use (n_blocks - n_blocks) to get an INDEX-typed zero for the first read
        first_idx = n_blocks - n_blocks
        phys_block_0 = pl.read(block_indices, first_idx)
        vj_row_0 = phys_block_0 * block_size

        pij_l1 = pl.load(
            pij_buf,
            [0, 0],
            [q_tile, block_size],
            target_memory=pl.MemorySpace.Mat,
        )
        vj_l1 = pl.load(
            value_cache,
            [vj_row_0, 0],
            [block_size, head_dim],
            target_memory=pl.MemorySpace.Mat,
        )
        pij_l0a = pl.move(pij_l1, target_memory=pl.MemorySpace.Left)
        vj_l0b = pl.move(vj_l1, target_memory=pl.MemorySpace.Right)
        oi_l0c = pl.matmul(pij_l0a, vj_l0b)
        oi_l0c_out = oi_l0c

        # Remaining blocks: matmul_acc (accumulate onto L0C)
        for i, (oi_l0c_iter,) in pl.range(1, n_blocks, init_values=(oi_l0c,)):
            phys_block = pl.read(block_indices, i)
            vj_row = phys_block * block_size

            pij_l1_i = pl.load(
                pij_buf,
                [i * q_tile, 0],
                [q_tile, block_size],
                target_memory=pl.MemorySpace.Mat,
            )
            vj_l1_i = pl.load(
                value_cache,
                [vj_row, 0],
                [block_size, head_dim],
                target_memory=pl.MemorySpace.Mat,
            )
            pij_l0a_i = pl.move(pij_l1_i, target_memory=pl.MemorySpace.Left)
            vj_l0b_i = pl.move(vj_l1_i, target_memory=pl.MemorySpace.Right)
            oi_l0c_acc = pl.matmul_acc(oi_l0c_iter, pij_l0a_i, vj_l0b_i)
            (oi_l0c_out,) = pl.yield_(oi_l0c_acc)

        oi_new = pl.store(oi_l0c_out, [0, 0], oi_new)
        return oi_new

    return kernel_pv_matmul


# ── Program builder ──────────────────────────────────────────────────────────


def build_paged_attention_multi_config_program(
    batch: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_num_blocks_per_req: int,
    context_len: int,
    q_tile: int = Q_TILE,
    n_unroll: int = N_UNROLL,
):
    """Build paged-attention @pl.program with multi-config interface.

    Orchestration pre-extracts block_indices via pl.slice() before kernel calls.

    Parameters
    ----------
    batch:                  number of requests in the batch
    num_heads:              number of query heads
    head_dim:               per-head feature dimension (128)
    block_size:             KV-cache block size (64)
    max_num_blocks_per_req: maximum number of KV blocks per request
    context_len:            context length (should be multiple of block_size for no-mask mode)
    q_tile:                 query-head tile size (16)
    n_unroll:               number of blocks per unroll group
    """
    n_unroll_q = n_unroll * q_tile
    query_rows = batch * num_heads
    key_cache_rows = batch * max_num_blocks_per_req * block_size
    out_rows = batch * num_heads
    block_table_flat_size = batch * max_num_blocks_per_req
    q_loop_static = (num_heads + q_tile - 1) // q_tile
    max_bn = (context_len + block_size - 1) // block_size

    _hub = make_kernel_aiv_hub(q_tile, head_dim)
    _sf = make_kernel_softmax_prepare(q_tile, block_size, n_unroll_q)
    _up = make_kernel_online_update(q_tile, head_dim)
    _qk = make_kernel_qk_matmul(key_cache_rows, q_tile, head_dim, block_size, n_unroll, n_unroll_q)
    _pv = make_kernel_pv_matmul(key_cache_rows, q_tile, head_dim, block_size, n_unroll, n_unroll_q)

    @pl.program
    class PagedAttentionMultiConfigProgram:
        """Paged attention with multi-config N_UNROLL=8 interface."""

        @pl.function(type=pl.FunctionType.Orchestration)
        def paged_attention(
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
            """Paged attention orchestration with N_UNROLL=8 block grouping.

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
                    oi, li_update, mi_update = _hub(oi, li_update, mi_update)

                    qi: pl.Tensor[[q_tile, head_dim], pl.BF16] = pl.slice(
                        query,
                        [q_tile, head_dim],  # type: ignore[reportArgumentType]
                        [cur_offset, 0],
                    )

                    # ── n_unroll loop over KV blocks ──────────
                    for bn in pl.range(0, max_bn, n_unroll):  # type: ignore[reportArgumentType]
                        n_blocks = pl.min(n_unroll, max_bn - bn)  # type: ignore[reportArgumentType]
                        bt_offset = b_idx * max_num_blocks_per_req + bn

                        # Pre-extract block indices from block_table (multi-config)
                        block_indices: pl.Tensor[[n_unroll], pl.INT32] = pl.slice(
                            block_table,
                            [n_unroll],  # type: ignore[reportArgumentType]
                            [bt_offset],
                        )

                        # 1. QK matmul (CUBE)
                        sij_buf: pl.Tensor[[n_unroll_q, block_size], pl.FP32] = pl.create_tensor(
                            [n_unroll_q, block_size],  # type: ignore[reportArgumentType]
                            dtype=pl.FP32,
                        )
                        sij_buf = _qk(
                            qi,
                            key_cache,
                            sij_buf,
                            block_indices,
                            n_blocks,
                        )

                        # 2. Softmax prepare (VECTOR)
                        pij_buf: pl.Tensor[[n_unroll_q, block_size], pl.BF16] = pl.create_tensor(
                            [n_unroll_q, block_size],  # type: ignore[reportArgumentType]
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
                        pij_buf, mi, li = _sf(
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
                        oi_new = _pv(
                            pij_buf,
                            value_cache,
                            oi_new,
                            block_indices,
                            n_blocks,
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
                        mi_update, li_update, oi, out_view = _up(
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

    return PagedAttentionMultiConfigProgram


# ── Golden reference ─────────────────────────────────────────────────────────


def golden_multi_config(tensors: dict, params: dict | None = None) -> None:
    """Golden reference for multi-config paged attention.

    Mirrors the orchestration structure: each group of up to n_unroll blocks
    uses a two-pass softmax (global row_max across all blocks in the group,
    then exp with that max).
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
    q_tile = min(num_heads, 128)
    n_unroll = (params or {}).get("n_unroll", 8)

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

            for bn in range(0, max_bn_b, n_unroll):
                n_blocks = min(n_unroll, max_bn_b - bn)

                # QK matmul for each block in the group
                all_sij = []
                for i in range(n_blocks):
                    bidx = int(block_table[b, bn + i].item())
                    kj = key_cache[bidx]
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
                    bidx = int(block_table[b, bn + i].item())
                    vj = value_cache[bidx]
                    oi_group += torch.mm(pij, vj)

                # Online update
                oi_acc, li_acc, mi_acc = _update(oi_acc, li_acc, mi_acc, oi_group, li_group, global_max)

            assert oi_acc is not None and li_acc is not None, f"No valid blocks for b={b} q={q_off}"
            out[b, q_off : q_off + q_tile, :] = oi_acc / li_acc

    tensors["out"][:] = out.reshape(batch * num_heads, head_dim)


# ── TensorSpec builder ───────────────────────────────────────────────────────


def build_tensor_specs_multi_config(
    batch: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_num_blocks_per_req: int,
    context_len: int,
    scale: float = 1.0,
) -> list[TensorSpec]:
    """Build TensorSpec list for multi-config paged attention."""
    query_rows = batch * num_heads
    key_cache_rows = batch * max_num_blocks_per_req * block_size
    block_table_flat_size = batch * max_num_blocks_per_req
    total_cache_blocks = key_cache_rows // block_size

    scale_bits = struct.unpack("I", struct.pack("f", scale))[0]
    config_data = torch.tensor(
        [batch, num_heads, 1, head_dim, block_size, max_num_blocks_per_req, scale_bits],
        dtype=torch.int64,
    )
    context_lens_data = torch.full((batch,), context_len, dtype=torch.int32)
    block_table_data = torch.randint(
        0, max(total_cache_blocks, 1), size=(batch, max_num_blocks_per_req), dtype=torch.int32
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
    # Small config for simulation testing
    # Use context_len as multiple of block_size to avoid partial-block masking
    batch = 4
    num_heads = 16
    head_dim = HEAD_DIM
    block_size = BLOCK_SIZE
    max_model_len = 2048
    context_len = 1024  # 1024 / 64 = 16 blocks, 16 / 8 = 2 N_UNROLL iterations
    scale = 1.0
    max_num_blocks_per_req = max_model_len // block_size  # 32

    program = build_paged_attention_multi_config_program(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks_per_req=max_num_blocks_per_req,
        context_len=context_len,
    )

    tensor_specs = build_tensor_specs_multi_config(
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
        golden=golden_multi_config,
        config=RunConfig(
            platform="a2a3sim",
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
