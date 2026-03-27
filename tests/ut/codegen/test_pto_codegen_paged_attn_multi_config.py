# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for PTO backend codegen for multi-config paged attention.

Tests that the multi-config paged attention program (with real QK/PV matmul
and online update kernels) compiles through the full pass pipeline and
generates non-empty MLIR for each InCore function.
"""

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

# ── Constants ────────────────────────────────────────────────────────────────
Q_TILE = 16
BLOCK_SIZE = 64
HEAD_DIM = 128
N_UNROLL = 64
N_UNROLL_Q = N_UNROLL * Q_TILE


# ── Kernel factory functions ─────────────────────────────────────────────────


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
        """Two-pass softmax: pass 1 finds global row_max, pass 2 computes exp+sum (VECTOR)."""
        for i, (mi_out_iter,) in pl.range(n_blocks, init_values=(mi_out,)):
            s_tile = pl.load(
                sij_buf,
                [i * q_tile, 0],
                [q_tile, block_size],
                target_memory=pl.MemorySpace.Vec,
            )
            scaled = pl.mul(s_tile, scale)
            tmp_tile = pl.create_tile([q_tile, block_size], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            local_max = pl.row_max(scaled, tmp_tile)
            if i == 0:
                mi_out_updated: pl.Tensor[[q_tile, 1], pl.FP32] = pl.store(local_max, [0, 0], mi_out_iter)
            else:
                global_max = pl.load(mi_out_iter, [0, 0], [q_tile, 1], target_memory=pl.MemorySpace.Vec)
                gm_nd = pl.reshape(global_max, [1, q_tile])
                lm_nd = pl.reshape(local_max, [1, q_tile])
                new_max = pl.reshape(pl.maximum(gm_nd, lm_nd), [q_tile, 1])
                mi_out_updated: pl.Tensor[[q_tile, 1], pl.FP32] = pl.store(new_max, [0, 0], mi_out_iter)
            (mi_out_carry,) = pl.yield_(mi_out_updated)

        for i, (pij_buf_iter, li_out_iter) in pl.range(n_blocks, init_values=(pij_buf, li_out)):
            global_max = pl.load(mi_out_carry, [0, 0], [q_tile, 1], target_memory=pl.MemorySpace.Vec)
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
                [q_tile, block_size], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
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
        """Online softmax update with inplace mi/li/oi (VECTOR)."""
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


def make_kernel_qk_matmul(
    key_cache_rows: int,
    q_tile: int = Q_TILE,
    head_dim: int = HEAD_DIM,
    block_size: int = BLOCK_SIZE,
    n_unroll: int = N_UNROLL,
    n_unroll_q: int = N_UNROLL_Q,
):
    """Create a multi-block QK matmul InCore kernel using pre-extracted block indices."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_qk_matmul(
        qi: pl.Tensor[[q_tile, head_dim], pl.BF16],
        key_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
        sij_buf: pl.Out[pl.Tensor[[n_unroll_q, block_size], pl.FP32]],
        block_indices: pl.Tensor[[n_unroll], pl.INT32],
        n_blocks: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[n_unroll_q, block_size], pl.FP32]:
        """Multi-block QK matmul: sij[i] = qi @ kj[i].T, vertically stacked (CUBE)."""
        for i, (sij_buf_iter,) in pl.range(n_blocks, init_values=(sij_buf,)):
            phys_block = pl.read(block_indices, i)
            kj_row = phys_block * block_size
            qi_l1 = pl.load(qi, [0, 0], [q_tile, head_dim], target_memory=pl.MemorySpace.Mat)
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
    """Create a SplitK PV matmul InCore kernel using pre-extracted block indices."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_pv_matmul(
        pij_buf: pl.Tensor[[n_unroll_q, block_size], pl.BF16],
        value_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
        oi_new: pl.Out[pl.Tensor[[q_tile, head_dim], pl.FP32]],
        block_indices: pl.Tensor[[n_unroll], pl.INT32],
        n_blocks: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[q_tile, head_dim], pl.FP32]:
        """SplitK PV matmul: first block via matmul, rest via matmul_acc (CUBE)."""
        first_idx = n_blocks - n_blocks
        phys_block_0 = pl.read(block_indices, first_idx)
        vj_row_0 = phys_block_0 * block_size

        pij_l1 = pl.load(pij_buf, [0, 0], [q_tile, block_size], target_memory=pl.MemorySpace.Mat)
        vj_l1 = pl.load(
            value_cache,
            [vj_row_0, 0],
            [block_size, head_dim],
            target_memory=pl.MemorySpace.Mat,
        )
        pij_l0a = pl.move(pij_l1, target_memory=pl.MemorySpace.Left)
        vj_l0b = pl.move(vj_l1, target_memory=pl.MemorySpace.Right)
        oi_l0c = pl.matmul(pij_l0a, vj_l0b)

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
    """Build paged-attention program with multi-config interface."""
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
        """Paged attention with multi-config interface."""

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
            for b_idx in pl.range(batch):
                for q_idx in pl.range(q_loop_static):
                    cur_offset = b_idx * num_heads + q_idx * q_tile

                    oi: pl.Tensor[[q_tile, head_dim], pl.FP32] = pl.create_tensor(
                        [q_tile, head_dim],
                        dtype=pl.FP32,  # type: ignore[reportArgumentType]
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

                    for bn in pl.range(0, max_bn, n_unroll):  # type: ignore[reportArgumentType]
                        n_blocks = pl.min(n_unroll, max_bn - bn)  # type: ignore[reportArgumentType]
                        bt_offset = b_idx * max_num_blocks_per_req + bn

                        block_indices: pl.Tensor[[n_unroll], pl.INT32] = pl.slice(
                            block_table,
                            [n_unroll],  # type: ignore[reportArgumentType]
                            [bt_offset],
                        )

                        sij_buf: pl.Tensor[[n_unroll_q, block_size], pl.FP32] = pl.create_tensor(
                            [n_unroll_q, block_size],
                            dtype=pl.FP32,  # type: ignore[reportArgumentType]
                        )
                        sij_buf = _qk(qi, key_cache, sij_buf, block_indices, n_blocks)

                        pij_buf: pl.Tensor[[n_unroll_q, block_size], pl.BF16] = pl.create_tensor(
                            [n_unroll_q, block_size],
                            dtype=pl.BF16,  # type: ignore[reportArgumentType]
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
                            li,  # type: ignore[reportArgumentType]
                            n_blocks,  # type: ignore[reportArgumentType]
                        )

                        oi_new: pl.Tensor[[q_tile, head_dim], pl.FP32] = pl.create_tensor(
                            [q_tile, head_dim],
                            dtype=pl.FP32,  # type: ignore[reportArgumentType]
                        )
                        oi_new = _pv(pij_buf, value_cache, oi_new, block_indices, n_blocks)

                        if bn == 0:
                            is_first: pl.Scalar[pl.INT64] = pl.yield_(1)
                        else:
                            is_first: pl.Scalar[pl.INT64] = pl.yield_(0)
                        if bn + n_blocks == max_bn:
                            is_last: pl.Scalar[pl.INT64] = pl.yield_(1)
                        else:
                            is_last: pl.Scalar[pl.INT64] = pl.yield_(0)

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


# ── Test parameters ──────────────────────────────────────────────────────────
# Small parameters for fast UT execution:
#   batch=1, num_heads=16, context_len=128 → max_bn=2 ≤ N_UNROLL=64
_BATCH = 1
_NUM_HEADS = 16
_MAX_NUM_BLOCKS_PER_REQ = 2
_CONTEXT_LEN = BLOCK_SIZE * _MAX_NUM_BLOCKS_PER_REQ  # 128 tokens


# ── Test ─────────────────────────────────────────────────────────────────────


def test_paged_attention_multi_config_codegen():
    """Verify multi-config paged attention compiles and generates MLIR for all InCore functions."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    program = build_paged_attention_multi_config_program(
        batch=_BATCH,
        num_heads=_NUM_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        max_num_blocks_per_req=_MAX_NUM_BLOCKS_PER_REQ,
        context_len=_CONTEXT_LEN,
    )

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    optimized_program = pm.run_passes(program)
    codegen_instance = codegen.PTOCodegen()

    incore_funcs = {
        func.name: func for func in optimized_program.functions.values() if ir.is_incore_type(func.func_type)
    }
    expected_incore = {
        "kernel_aiv_hub",
        "kernel_softmax_prepare",
        "kernel_online_update",
        "kernel_qk_matmul",
        "kernel_pv_matmul",
    }
    missing = expected_incore - set(incore_funcs)
    assert not missing, f"Missing InCore functions after passes: {sorted(missing)}"

    for func_name, func in incore_funcs.items():
        single_func_program = ir.Program([func], func_name, optimized_program.span)
        mlir_code = codegen_instance.generate(single_func_program)
        assert mlir_code, f"Generated MLIR code for {func_name} should not be empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
