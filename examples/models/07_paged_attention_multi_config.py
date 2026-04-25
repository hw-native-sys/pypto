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

Builds a multi-config paged attention program using the PyPTO DSL with:
- N_UNROLL block grouping per unroll iteration
- Dynamic runtime shapes derived from input tensor dimensions via pl.tensor.dim()
- Per-request context length from context_lens tensor
- valid_len support for partial blocks (last block may have fewer valid columns)
- Two-pass softmax within each unroll group + online update across groups

Orchestration pre-extracts block_indices via pl.slice() from block_table.
Kernels receive block_indices tensor view (no block_table + bt_offset indirection).

Dynamic axis description follows the dynamic paged-attention example: tensor
type annotations use module-level pl.dynamic() variables, while InCore kernel
bodies use builder closures (q_tile, head_dim, block_size) for fixed-size loads.
Runtime values (batch, num_heads, head_dim, block_size, block_num) are derived
from input tensor shapes inside the orchestration via pl.tensor.dim().

Key interface difference vs multi-config C++ impl:
  multi-config passes N_UNROLL individual scalar block_indices to each kernel.
  PyPTO passes a single block_indices tensor view (functionally equivalent —
  same data access pattern: block_indices[i] -> physical block index).

Tile dimensions (matching multi-config Case2, q_tile=16):
  QK Matmul:       qi(16, 128) @ kj.T(128, 64) -> sij(16, 64)
  Softmax:         sij(16, N) -> pij(16, N) bf16, mi(16, 1), li(16, 1)
  PV Matmul:       pij(16, 64) @ vj(64, 128) -> oi(16, 128)
  Online Update:   operates on (16, 128) data tiles, (16, 1) scalar tiles

Module-level InCore kernels (reusable, importable):
  kernel_softmax_prepare, kernel_online_update

Factory functions for multi-block kernels:
  make_kernel_qk_matmul()
  make_kernel_pv_matmul()
"""

# DSL function bodies are parsed as AST — dynamic var names look undefined to pyright.
# pyright: reportUndefinedVariable=false

import argparse

import pypto.language as pl
import torch
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime import RunConfig, run

# ── Constants ────────────────────────────────────────────────────────────────
Q_TILE = 16
BLOCK_SIZE = 64
HEAD_DIM = 128
N_UNROLL = 64
N_UNROLL_Q = N_UNROLL * Q_TILE  # 1024 — static sij/pij buffer height

# ── Module-level dynamic variables ───────────────────────────────────────────
# Used only in tensor type annotations; InCore kernel bodies and orchestration
# bodies use builder closure ints / pl.tensor.dim() Scalars for actual sizes.
Q_HEADS = pl.dynamic("Q_HEADS")  # query tile rows (= q_tile)
HEAD_DIM_DYN = pl.dynamic("HEAD_DIM_DYN")  # per-head feature dim on the Q/output side
# KV-side head_dim is given a distinct dynamic var so that within a single InCore
# kernel call (e.g. PV matmul with value_cache + oi_new outputs), the type-system
# unification does not conflict between the orchestration-typed value_cache
# (KEY_CACHE_ROWS_DYN, HEAD_DIM_DYN) and the create_tensor-typed oi_new buffer
# whose head_dim is the concrete pl.tensor.dim() Scalar.  At call sites both
# resolve to the same runtime value (head_dim_cfg).
KV_HEAD_DIM_DYN = pl.dynamic("KV_HEAD_DIM_DYN")
BLOCK_SIZE_DYN = pl.dynamic("BLOCK_SIZE_DYN")  # KV-cache block size (= block_size)
BATCH_DYN = pl.dynamic("BATCH_DYN")  # number of requests
QUERY_ROWS_DYN = pl.dynamic("QUERY_ROWS_DYN")  # batch * num_heads
KEY_CACHE_ROWS_DYN = pl.dynamic("KEY_CACHE_ROWS_DYN")  # batch * max_blocks_per_req * block_size
BLOCK_TABLE_FLAT_DYN = pl.dynamic("BLOCK_TABLE_FLAT_DYN")  # batch * max_blocks_per_req


# ── Kernel factory functions ──────────────────────────────────────────────────


def make_kernel_softmax_prepare(q_tile: int, block_size: int, n_unroll_q: int):
    """Create softmax_prepare InCore kernel with parameterised tile dimensions."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_softmax_prepare(
        sij_buf: pl.Tensor[[n_unroll_q, BLOCK_SIZE_DYN], pl.FP32],
        scale: pl.Scalar[pl.FP32],
        pij_buf: pl.Out[pl.Tensor[[n_unroll_q, BLOCK_SIZE_DYN], pl.BF16]],
        mi_out: pl.Out[pl.Tensor[[Q_HEADS, 1], pl.FP32]],
        li_out: pl.Out[pl.Tensor[[Q_HEADS, 1], pl.FP32]],
        n_blocks: pl.Scalar[pl.INDEX],
        last_valid_len: pl.Scalar[pl.INDEX],
    ) -> tuple[
        pl.Tensor[[n_unroll_q, BLOCK_SIZE_DYN], pl.BF16],
        pl.Tensor[[Q_HEADS, 1], pl.FP32],
        pl.Tensor[[Q_HEADS, 1], pl.FP32],
    ]:
        """Two-pass softmax with partial-block support (VECTOR).

        Pass 1 finds global row_max, pass 2 computes exp+sum.
        The last block (i == n_blocks - 1) uses valid_shapes + fillpad to mask
        out invalid columns with -inf so they don't affect row_max or row_sum.
        Uses mi_out/li_out as GM scratch for cross-iteration state via store/load round-trips.
        """
        # Pass 1: find global row_max across all blocks
        for i, (mi_out_iter,) in pl.range(n_blocks, init_values=(mi_out,)):
            if i == n_blocks - 1:
                valid_len: pl.Scalar[pl.INDEX] = pl.yield_(last_valid_len)
            else:
                valid_len: pl.Scalar[pl.INDEX] = pl.yield_(block_size)
            s_tile = pl.load(
                sij_buf,
                [i * q_tile, 0],
                [q_tile, block_size],
                valid_shapes=[q_tile, valid_len],
                target_memory=pl.MemorySpace.Vec,
            )
            s_tile_padded = pl.tile.fillpad(s_tile, pad_value=pl.PadValue.min)
            scaled = pl.mul(s_tile_padded, scale)
            tmp_tile = pl.create_tile(
                [q_tile, block_size],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            local_max = pl.row_max(scaled, tmp_tile)

            if i == 0:
                mi_out_updated = pl.store(local_max, [0, 0], mi_out_iter)
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
                mi_out_updated = pl.store(new_max, [0, 0], mi_out_iter)
            (mi_out_carry,) = pl.yield_(mi_out_updated)

        # Pass 2: exp(s - global_max), cast to bf16, row_sum accumulation
        for i, (pij_buf_iter, li_out_iter) in pl.range(n_blocks, init_values=(pij_buf, li_out)):
            global_max = pl.load(
                mi_out_carry,
                [0, 0],
                [q_tile, 1],
                target_memory=pl.MemorySpace.Vec,
            )
            if i == n_blocks - 1:
                valid_len_p2: pl.Scalar[pl.INDEX] = pl.yield_(last_valid_len)
            else:
                valid_len_p2: pl.Scalar[pl.INDEX] = pl.yield_(block_size)
            s_tile_raw = pl.load(
                sij_buf,
                [i * q_tile, 0],
                [q_tile, block_size],
                valid_shapes=[q_tile, valid_len_p2],
                target_memory=pl.MemorySpace.Vec,
            )
            s_tile_p2 = pl.tile.fillpad(s_tile_raw, pad_value=pl.PadValue.min)
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
                li_out_updated = pl.store(li_local, [0, 0], li_out_iter)
            else:
                li_acc = pl.load(li_out_iter, [0, 0], [q_tile, 1])
                li_acc_nd = pl.reshape(li_acc, [1, q_tile])
                li_sum = pl.reshape(pl.add(li_acc_nd, li_local_nd), [q_tile, 1])
                li_out_updated = pl.store(li_sum, [0, 0], li_out_iter)
            pij_buf_carry, li_out_carry = pl.yield_(pij_buf_updated, li_out_updated)

        return pij_buf_carry, mi_out_carry, li_out_carry

    return kernel_softmax_prepare


def make_kernel_online_update(q_tile: int, head_dim: int):
    """Create online_update InCore kernel with parameterised tile dimensions."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_online_update(  # noqa: PLR0913
        mij: pl.Tensor[[Q_HEADS, 1], pl.FP32],
        lij: pl.Tensor[[Q_HEADS, 1], pl.FP32],
        oi_new: pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32],
        mi: pl.InOut[pl.Tensor[[Q_HEADS, 1], pl.FP32]],
        li: pl.InOut[pl.Tensor[[Q_HEADS, 1], pl.FP32]],
        oi: pl.InOut[pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32]],
        dst: pl.Out[pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32]],
        is_first: pl.Scalar[pl.INDEX],
        is_last: pl.Scalar[pl.INDEX],
    ) -> tuple[
        pl.Tensor[[Q_HEADS, 1], pl.FP32],
        pl.Tensor[[Q_HEADS, 1], pl.FP32],
        pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32],
        pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32],
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
kernel_softmax_prepare = make_kernel_softmax_prepare(Q_TILE, BLOCK_SIZE, N_UNROLL_Q)
kernel_online_update = make_kernel_online_update(Q_TILE, HEAD_DIM)


# ── Factory functions for multi-config kernels ──────────────────────────


def make_kernel_qk_matmul(
    q_tile: int = Q_TILE,
    head_dim: int = HEAD_DIM,
    block_size: int = BLOCK_SIZE,
    n_unroll: int = N_UNROLL,
    n_unroll_q: int = N_UNROLL_Q,
):
    """Create a multi-block QK matmul InCore kernel using pre-extracted block indices.

    Multi-config: receives block_indices tensor (pre-sliced by orchestration)
    instead of full block_table + bt_offset.

    Tensor type annotations use module-level pl.dynamic() variables so the kernel
    accepts any key_cache row count at runtime; load shapes use the builder
    closure ints (q_tile / head_dim / block_size).

    Parameters
    ----------
    q_tile: query tile height
    head_dim: per-head feature dimension
    block_size: KV-cache block size
    n_unroll: number of blocks per unroll group
    n_unroll_q: n_unroll * q_tile (static buffer height)
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_qk_matmul(
        qi: pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.BF16],
        key_cache: pl.Tensor[[KEY_CACHE_ROWS_DYN, KV_HEAD_DIM_DYN], pl.BF16],
        sij_buf: pl.Out[pl.Tensor[[n_unroll_q, BLOCK_SIZE_DYN], pl.FP32]],
        block_indices: pl.Tensor[[n_unroll], pl.INT32],
        n_blocks: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[n_unroll_q, BLOCK_SIZE_DYN], pl.FP32]:
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
                [block_size, head_dim],
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
    q_tile: int = Q_TILE,
    head_dim: int = HEAD_DIM,
    block_size: int = BLOCK_SIZE,
    n_unroll: int = N_UNROLL,
    n_unroll_q: int = N_UNROLL_Q,
):
    """Create a SplitK PV matmul InCore kernel using pre-extracted block indices.

    Multi-config: receives block_indices tensor (pre-sliced by orchestration)
    instead of full block_table + bt_offset.

    Tensor type annotations use module-level pl.dynamic() variables so the kernel
    accepts any value_cache row count at runtime; load shapes use the builder
    closure ints (q_tile / head_dim / block_size).

    Parameters
    ----------
    q_tile: query tile height
    head_dim: per-head feature dimension
    block_size: KV-cache block size
    n_unroll: number of blocks per unroll group
    n_unroll_q: n_unroll * q_tile (static buffer height)
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_pv_matmul(
        pij_buf: pl.Tensor[[n_unroll_q, BLOCK_SIZE_DYN], pl.BF16],
        value_cache: pl.Tensor[[KEY_CACHE_ROWS_DYN, KV_HEAD_DIM_DYN], pl.BF16],
        oi_new: pl.Out[pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32]],
        block_indices: pl.Tensor[[n_unroll], pl.INT32],
        n_blocks: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32]:
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
    q_tile: int = Q_TILE,
    head_dim: int = HEAD_DIM,
    block_size: int = BLOCK_SIZE,
    n_unroll: int = N_UNROLL,
):
    """Build paged-attention @pl.program with multi-config interface.

    Orchestration derives runtime shapes (batch, num_heads, head_dim, block_size,
    block_num) from input tensor dimensions via pl.tensor.dim(); per-request
    context lengths come from context_lens.  Block indices are pre-extracted
    via pl.slice() before kernel calls.

    Parameters
    ----------
    q_tile:     query-head tile size (compile-time constant for InCore kernels)
    head_dim:   per-head feature dimension (compile-time constant for loads)
    block_size: KV-cache block size (compile-time constant for loads)
    n_unroll:   number of blocks per unroll group
    """
    n_unroll_q = n_unroll * q_tile

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_init_inplace(
        oi: pl.Out[pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32]],
        li: pl.Out[pl.Tensor[[Q_HEADS, 1], pl.FP32]],
        mi: pl.Out[pl.Tensor[[Q_HEADS, 1], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32],
        pl.Tensor[[Q_HEADS, 1], pl.FP32],
        pl.Tensor[[Q_HEADS, 1], pl.FP32],
    ]:
        """No-op passthrough that binds concrete create_tensor shapes to dynamic types.

        pl.create_tensor zero-initialises the buffers before this call; this
        function exists solely to propagate dynamic-shape types at the call site
        so downstream kernel returns can be reassigned to the same variables.
        """
        return oi, li, mi

    _sf = make_kernel_softmax_prepare(q_tile, block_size, n_unroll_q)
    _up = make_kernel_online_update(q_tile, head_dim)
    _qk = make_kernel_qk_matmul(q_tile, head_dim, block_size, n_unroll, n_unroll_q)
    _pv = make_kernel_pv_matmul(q_tile, head_dim, block_size, n_unroll, n_unroll_q)

    @pl.program
    class PagedAttentionMultiConfigProgram:
        """Paged attention with dynamic-shape orchestration and valid_len support."""

        @pl.function(type=pl.FunctionType.Orchestration)
        def paged_attention(
            self,
            query: pl.Tensor[[QUERY_ROWS_DYN, HEAD_DIM_DYN], pl.BF16],
            key_cache: pl.Tensor[[KEY_CACHE_ROWS_DYN, HEAD_DIM_DYN], pl.BF16],
            value_cache: pl.Tensor[[KEY_CACHE_ROWS_DYN, HEAD_DIM_DYN], pl.BF16],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            context_lens: pl.Tensor[[BATCH_DYN], pl.INT32],
            out: pl.Out[pl.Tensor[[QUERY_ROWS_DYN, HEAD_DIM_DYN], pl.FP32]],
        ) -> pl.Tensor[[QUERY_ROWS_DYN, HEAD_DIM_DYN], pl.FP32]:
            """Paged attention orchestration with shapes derived from tensor dims.

            Shape derivations: batch = context_lens.dim(0),
            head_dim = query.dim(1), num_heads = query.dim(0) // batch,
            block_num = block_table.dim(0) // batch,
            block_size = value_cache.dim(0) // block_table.dim(0).
            """
            batch_cfg = pl.tensor.dim(context_lens, 0)
            query_rows = pl.tensor.dim(query, 0)
            head_dim_cfg = pl.tensor.dim(query, 1)
            value_cache_rows = pl.tensor.dim(value_cache, 0)
            block_table_size = pl.tensor.dim(block_table, 0)
            num_heads_cfg = query_rows // batch_cfg
            block_size_cfg = value_cache_rows // block_table_size
            block_num_cfg = block_table_size // batch_cfg
            q_loop_cfg = (num_heads_cfg + q_tile - 1) // q_tile

            for b_idx in pl.range(batch_cfg):
                cur_seq = pl.tensor.read(context_lens, [b_idx])
                bn_this_batch = (cur_seq + block_size_cfg - 1) // block_size_cfg
                for q_idx in pl.range(q_loop_cfg):
                    cur_offset = b_idx * num_heads_cfg + q_idx * q_tile

                    # Allocate accumulators with concrete shapes, then bind to
                    # dynamic-shape types via init_inplace so they can be carried
                    # across iterations of the bn loop without a type mismatch.
                    oi_buf = pl.create_tensor([q_tile, head_dim_cfg], dtype=pl.FP32)
                    li_buf = pl.create_tensor([q_tile, 1], dtype=pl.FP32)
                    mi_buf = pl.create_tensor([q_tile, 1], dtype=pl.FP32)
                    oi, li_update, mi_update = kernel_init_inplace(oi_buf, li_buf, mi_buf)

                    qi = pl.slice(query, [q_tile, head_dim_cfg], [cur_offset, 0])

                    # ── n_unroll loop over KV blocks ──────────
                    for bn in pl.range(0, bn_this_batch, n_unroll):  # type: ignore[reportArgumentType]
                        n_blocks = pl.min(n_unroll, bn_this_batch - bn)  # type: ignore[reportArgumentType]
                        bt_offset = b_idx * block_num_cfg + bn

                        # Pre-extract block indices from block_table (multi-config)
                        block_indices = pl.slice(block_table, [n_unroll], [bt_offset])  # type: ignore[reportArgumentType]

                        # valid columns for the last block in this unroll group
                        last_valid_len = pl.min(
                            block_size_cfg, cur_seq - (bn + n_blocks - 1) * block_size_cfg
                        )

                        # 1. QK matmul (CUBE)
                        sij_buf_in = pl.create_tensor([n_unroll_q, block_size_cfg], dtype=pl.FP32)
                        sij_buf = _qk(
                            qi,
                            key_cache,
                            sij_buf_in,
                            block_indices,
                            n_blocks,
                        )

                        # 2. Softmax prepare (VECTOR)
                        pij_buf_in = pl.create_tensor([n_unroll_q, block_size_cfg], dtype=pl.BF16)
                        mi_buf_in = pl.create_tensor([q_tile, 1], dtype=pl.FP32)
                        li_buf_in = pl.create_tensor([q_tile, 1], dtype=pl.FP32)
                        pij_buf, mi, li = _sf(
                            sij_buf,
                            1.0,
                            pij_buf_in,
                            mi_buf_in,
                            li_buf_in,
                            n_blocks,  # type: ignore[reportArgumentType]
                            last_valid_len,  # type: ignore[reportArgumentType]
                        )

                        # 3. PV matmul (CUBE)
                        oi_new_in = pl.create_tensor([q_tile, head_dim_cfg], dtype=pl.FP32)
                        oi_new = _pv(
                            pij_buf,
                            value_cache,
                            oi_new_in,
                            block_indices,
                            n_blocks,
                        )

                        # 4. Online update flags
                        if bn == 0:
                            is_first = pl.yield_(1)
                        else:
                            is_first = pl.yield_(0)
                        if bn + n_blocks == bn_this_batch:
                            is_last = pl.yield_(1)
                        else:
                            is_last = pl.yield_(0)

                        # 5. Online update (VECTOR)
                        out_view_buf = pl.slice(out, [q_tile, head_dim_cfg], [cur_offset, 0])
                        mi_update, li_update, oi, out_view = _up(
                            mi,
                            li,
                            oi_new,
                            mi_update,
                            li_update,
                            oi,
                            out_view_buf,
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
    then exp with that max).  Supports partial blocks via valid_len.

    Shape derivations match the orchestration's pl.tensor.dim() logic.
    Scale is hardcoded to 1.0 to match the orchestration function.
    """
    context_lens = tensors["context_lens"]
    query = tensors["query"]
    key_cache_t = tensors["key_cache"]
    value_cache_t = tensors["value_cache"]
    block_table_flat = tensors["block_table"]

    batch = context_lens.shape[0]
    num_heads = query.shape[0] // batch
    head_dim = query.shape[1]
    block_size = value_cache_t.shape[0] // block_table_flat.shape[0]
    max_num_blocks_per_req = block_table_flat.shape[0] // batch
    scale = 1.0

    query = query.float().reshape(batch, num_heads, head_dim)
    total_pool_blocks = batch * max_num_blocks_per_req
    key_cache = key_cache_t.float().reshape(total_pool_blocks, block_size, head_dim)
    value_cache = value_cache_t.float().reshape(total_pool_blocks, block_size, head_dim)
    block_table = block_table_flat.reshape(batch, max_num_blocks_per_req)

    out = torch.zeros((batch, num_heads, head_dim), dtype=torch.float32)
    params = params or {}
    # Default q_tile clamps to num_heads so callers using num_heads < Q_TILE still
    # get a divisible q-row tiling instead of running zero outer iterations.
    q_tile = params.get("q_tile", min(num_heads, Q_TILE))
    n_unroll = params.get("n_unroll", N_UNROLL)

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
                    # Mask invalid columns on the last block
                    if i == n_blocks - 1:
                        last_valid = min(block_size, cur_seq - (bn + i) * block_size)
                        if last_valid < block_size:
                            sij[:, last_valid:] = float("-inf")
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


def build_tensors_multi_config(
    batch: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_num_blocks_per_req: int,
    context_len: int,
) -> tuple[torch.Tensor, ...]:
    """Build torch tensors for multi-config paged attention.

    Returns:
        (query, key_cache, value_cache, block_table, context_lens, out)
    """
    query_rows = batch * num_heads
    key_cache_rows = batch * max_num_blocks_per_req * block_size
    total_cache_blocks = key_cache_rows // block_size

    context_lens = torch.full((batch,), context_len, dtype=torch.int32)
    block_table = torch.randint(
        0, max(total_cache_blocks, 1), size=(batch, max_num_blocks_per_req), dtype=torch.int32
    ).flatten()

    query = torch.randn(query_rows, head_dim, dtype=torch.bfloat16)
    key_cache = torch.randn(key_cache_rows, head_dim, dtype=torch.bfloat16)
    value_cache = torch.randn(key_cache_rows, head_dim, dtype=torch.bfloat16)
    out = torch.zeros(query_rows, head_dim, dtype=torch.float32)

    return (
        query,
        key_cache,
        value_cache,
        block_table,
        context_lens,
        out,
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Multi-config paged attention example")
    parser.add_argument(
        "--runtime-profiling",
        action="store_true",
        default=False,
        help="Enable on-device runtime profiling and generate swimlane JSON",
    )
    args = parser.parse_args()

    batch = 4
    num_heads = 16
    head_dim = HEAD_DIM
    block_size = BLOCK_SIZE
    max_model_len = 2048
    context_len = 1024
    max_num_blocks_per_req = max_model_len // block_size  # 32

    program = build_paged_attention_multi_config_program(
        head_dim=head_dim,
        block_size=block_size,
    )

    (
        query,
        key_cache,
        value_cache,
        block_table,
        context_lens,
        out,
    ) = build_tensors_multi_config(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks_per_req=max_num_blocks_per_req,
        context_len=context_len,
    )
    run(
        program,
        query,
        key_cache,
        value_cache,
        block_table,
        context_lens,
        out,
        config=RunConfig(
            platform="a2a3sim",
            device_id=11,
            strategy=OptimizationStrategy.Default,
            dump_passes=True,
            backend_type=BackendType.Ascend910B,
            compile_profiling=True,
            runtime_profiling=args.runtime_profiling,
        ),
    )

    # Golden validation
    expected_out = out.clone()
    golden_multi_config(
        {
            "query": query,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "block_table": block_table,
            "context_lens": context_lens,
            "out": expected_out,
        },
    )
    assert torch.allclose(out, expected_out, rtol=2e-2, atol=2e-2), (
        f"Validation failed: max diff = {(out - expected_out).abs().max().item()}"
    )
    print("PASSED")


if __name__ == "__main__":
    main()
