# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Dynamic Paged Attention Example

InCore kernel type annotations use pl.dynamic() variables (Q_HEADS, HEAD_DIM_DYN,
BLOCK_SIZE_DYN), while load operations use closure variables (_Q_TILE, _HEAD_DIM,
_BLOCK_SIZE) captured from build_dynamic_paged_attention_program().
Orchestration-level tensor shapes also use pl.dynamic() variables so the program
accepts any batch size at runtime.
"""

# DSL function bodies are parsed as AST — dynamic var names look undefined to pyright.
# pyright: reportUndefinedVariable=false

import pypto.language as pl
import torch  # type: ignore[import]
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime import RunConfig, TensorSpec, run

# ---------------------------------------------------------------------------
# Module-level dynamic variables — used only in InCore kernel type annotations.
# Load operations inside the kernels use closure variables from the builder instead.
# ---------------------------------------------------------------------------

Q_HEADS = pl.dynamic("Q_HEADS")  # query tile rows   (e.g. 16)
HEAD_DIM_DYN = pl.dynamic("HEAD_DIM_DYN")  # head dimension    (e.g. 128)
BLOCK_SIZE_DYN = pl.dynamic("BLOCK_SIZE_DYN")  # KV block size     (e.g. 128)
BATCH_DYN = pl.dynamic("BATCH_DYN")  # batch size (number of requests)
QUERY_ROWS_DYN = pl.dynamic("QUERY_ROWS_DYN")  # batch * num_heads
KEY_CACHE_ROWS_DYN = pl.dynamic("KEY_CACHE_ROWS_DYN")  # batch * max_num_blocks_per_req * block_size
BLOCK_TABLE_FLAT_DYN = pl.dynamic("BLOCK_TABLE_FLAT_DYN")  # batch * max_num_blocks_per_req


# ---------------------------------------------------------------------------
# Program builders
# ---------------------------------------------------------------------------


def build_dynamic_paged_attention_program(
    q_tile: int,
    head_dim: int,
    block_size: int,
):
    """Build a paged-attention @pl.program whose InCore kernels use dynamic shapes.

    InCore kernel type annotations reference module-level pl.dynamic() variables so
    shapes are resolved at runtime.  Load operations use closure variables captured
    from the builder parameters (_Q_TILE, _HEAD_DIM, _BLOCK_SIZE).

    Parameters
    ----------
    q_tile:     query-head tile size (number of query heads processed per InCore call)
    head_dim:   per-head feature dimension
    block_size: KV-cache block size (rows per physical block)
    """
    # Tile-size constants captured as closures by the InCore kernels below.
    _Q_TILE: int = q_tile
    _HEAD_DIM: int = head_dim
    _BLOCK_SIZE: int = block_size

    # -----------------------------------------------------------------------
    # InCore kernels — defined here to capture _Q_TILE, _HEAD_DIM, _BLOCK_SIZE
    # as closure variables; type annotations use module-level pl.dynamic() vars.
    # -----------------------------------------------------------------------

    @pl.function(type=pl.FunctionType.InCore)
    def dyn_kernel_init_inplace(
        oi: pl.Out[pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32]],
        li: pl.Out[pl.Tensor[[Q_HEADS, 1], pl.FP32]],
        mi: pl.Out[pl.Tensor[[Q_HEADS, 1], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32],
        pl.Tensor[[Q_HEADS, 1], pl.FP32],
        pl.Tensor[[Q_HEADS, 1], pl.FP32],
    ]:
        """No-op passthrough: binds concrete tensor shapes to dynamic type annotations.

        pl.create_tensor zero-initialises the buffers before this call; this
        function exists solely to propagate the dynamic shape at the call site.
        """
        return oi, li, mi

    @pl.function(type=pl.FunctionType.InCore)
    def dyn_kernel_qk_matmul(
        qi: pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.BF16],
        kj: pl.Tensor[[BLOCK_SIZE_DYN, HEAD_DIM_DYN], pl.BF16],
        output: pl.Out[pl.Tensor[[Q_HEADS, BLOCK_SIZE_DYN], pl.FP32]],
    ) -> pl.Tensor[[Q_HEADS, BLOCK_SIZE_DYN], pl.FP32]:
        """QK matmul: output = qi @ kj.T (CUBE). kj transposed on load."""
        qi_l1 = pl.load(qi, [0, 0], [_Q_TILE, _HEAD_DIM], target_memory=pl.MemorySpace.Mat)
        kj_l1 = pl.load(
            kj, [0, 0], [_HEAD_DIM, _BLOCK_SIZE], target_memory=pl.MemorySpace.Mat, transpose=True
        )
        qi_l0a = pl.move(qi_l1, target_memory=pl.MemorySpace.Left)
        kj_l0b = pl.move(kj_l1, target_memory=pl.MemorySpace.Right)
        sij_l0c = pl.matmul(qi_l0a, kj_l0b)
        out = pl.store(sij_l0c, [0, 0], output)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def dyn_kernel_softmax_prepare(
        sij: pl.Tensor[[Q_HEADS, BLOCK_SIZE_DYN], pl.FP32],
        scale: pl.Scalar[pl.FP32],
        out_pij: pl.Out[pl.Tensor[[Q_HEADS, BLOCK_SIZE_DYN], pl.BF16]],
        out_mi: pl.Out[pl.Tensor[[Q_HEADS, 1], pl.FP32]],
        out_li: pl.Out[pl.Tensor[[Q_HEADS, 1], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[Q_HEADS, BLOCK_SIZE_DYN], pl.BF16],
        pl.Tensor[[Q_HEADS, 1], pl.FP32],
        pl.Tensor[[Q_HEADS, 1], pl.FP32],
    ]:
        """Scale sij, compute row_max (mi), exp(sij-mi), cast to BF16, row_sum (li). VECTOR."""
        s_tile = pl.load(sij, [0, 0], [_Q_TILE, _BLOCK_SIZE], target_memory=pl.MemorySpace.Vec)
        scaled = pl.mul(s_tile, scale)
        tmp_tile = pl.create_tile([_Q_TILE, _BLOCK_SIZE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        mi_tile = pl.row_max(scaled, tmp_tile)
        sij_centered = pl.row_expand_sub(scaled, mi_tile)
        exp_tile = pl.exp(sij_centered)
        pij_tile_bf16 = pl.cast(exp_tile, target_type=pl.BF16)
        pij_tile = pl.cast(pij_tile_bf16, target_type=pl.FP32)
        li_tile = pl.row_sum(pij_tile, tmp_tile)
        out_pij = pl.store(pij_tile_bf16, [0, 0], out_pij)
        out_mi = pl.store(mi_tile, [0, 0], out_mi)
        out_li = pl.store(li_tile, [0, 0], out_li)
        return out_pij, out_mi, out_li

    @pl.function(type=pl.FunctionType.InCore)
    def dyn_kernel_pv_matmul(
        pij: pl.Tensor[[Q_HEADS, BLOCK_SIZE_DYN], pl.BF16],
        vj: pl.Tensor[[BLOCK_SIZE_DYN, HEAD_DIM_DYN], pl.BF16],
        output: pl.Out[pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32]],
    ) -> pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32]:
        """PV matmul: output = pij @ vj (CUBE)."""
        pij_l1 = pl.load(pij, [0, 0], [_Q_TILE, _BLOCK_SIZE], target_memory=pl.MemorySpace.Mat)
        vj_l1 = pl.load(vj, [0, 0], [_BLOCK_SIZE, _HEAD_DIM], target_memory=pl.MemorySpace.Mat)
        pij_l0a = pl.move(pij_l1, target_memory=pl.MemorySpace.Left)
        vj_l0b = pl.move(vj_l1, target_memory=pl.MemorySpace.Right)
        oi_l0c = pl.matmul(pij_l0a, vj_l0b)
        out = pl.store(oi_l0c, [0, 0], output)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def dyn_kernel_online_update(
        mij: pl.Tensor[[Q_HEADS, 1], pl.FP32],
        lij: pl.Tensor[[Q_HEADS, 1], pl.FP32],
        oi_new: pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32],
        mi: pl.InOut[pl.Tensor[[Q_HEADS, 1], pl.FP32]],
        li: pl.InOut[pl.Tensor[[Q_HEADS, 1], pl.FP32]],
        oi: pl.InOut[pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32]],
        dst: pl.Out[pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32]],
        is_first: pl.Scalar[pl.BOOL],
        is_last: pl.Scalar[pl.BOOL],
    ) -> tuple[
        pl.Tensor[[Q_HEADS, 1], pl.FP32],
        pl.Tensor[[Q_HEADS, 1], pl.FP32],
        pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32],
        pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32],
    ]:
        """Merge current block's (mij, lij, oi_new) into running accumulators (mi, li, oi). VECTOR.

        First block (is_first=True): directly assign accumulators, no rescaling.
        Subsequent blocks: mi_new=max(mi,mij), alpha=exp(mi-mi_new), beta=exp(mij-mi_new),
          li <- alpha*li + beta*lij,  oi <- alpha*oi + beta*oi_new.
        Last block (is_last=True): write oi/li to dst; otherwise dst=zeros.

        Reshape between [Q_TILE,1] and [1,Q_TILE] is required because row_expand_mul
        broadcasts along the column axis while element-wise ops need consistent layout.
        """
        mij_tile = pl.load(mij, [0, 0], [_Q_TILE, 1], target_memory=pl.MemorySpace.Vec)
        lij_tile = pl.load(lij, [0, 0], [_Q_TILE, 1], target_memory=pl.MemorySpace.Vec)
        oi_new_tile = pl.load(oi_new, [0, 0], [_Q_TILE, _HEAD_DIM], target_memory=pl.MemorySpace.Vec)
        mi_tile = pl.load(mi, [0, 0], [_Q_TILE, 1], target_memory=pl.MemorySpace.Vec)
        li_tile = pl.load(li, [0, 0], [_Q_TILE, 1], target_memory=pl.MemorySpace.Vec)
        oi_tile = pl.load(oi, [0, 0], [_Q_TILE, _HEAD_DIM], target_memory=pl.MemorySpace.Vec)

        if is_first:
            mi_out = pl.store(mij_tile, [0, 0], mi)
            li_out = pl.store(lij_tile, [0, 0], li)
            oi_out = pl.store(oi_new_tile, [0, 0], oi)
            if is_last:
                dst_tile = pl.row_expand_div(oi_new_tile, lij_tile)
                dst_out = pl.store(dst_tile, [0, 0], dst)
            else:
                zero_tile = pl.tile.full([_Q_TILE, _HEAD_DIM], dtype=pl.FP32, value=0.0)
                dst_out = pl.store(zero_tile, [0, 0], dst)
        else:
            mi_tile_nd = pl.reshape(mi_tile, [1, _Q_TILE])
            mij_tile_nd = pl.reshape(mij_tile, [1, _Q_TILE])
            li_tile_nd = pl.reshape(li_tile, [1, _Q_TILE])
            lij_tile_nd = pl.reshape(lij_tile, [1, _Q_TILE])

            mi_new = pl.maximum(mi_tile_nd, mij_tile_nd)
            mi_diff = pl.sub(mi_tile_nd, mi_new)
            alpha = pl.exp(mi_diff)
            mij_diff = pl.sub(mij_tile_nd, mi_new)
            beta = pl.exp(mij_diff)

            li_scaled = pl.mul(alpha, li_tile_nd)
            lij_scaled = pl.mul(beta, lij_tile_nd)
            li_updated = pl.add(li_scaled, lij_scaled)

            alpha_dn = pl.reshape(alpha, [_Q_TILE, 1])
            oi_scaled = pl.row_expand_mul(oi_tile, alpha_dn)
            beta_dn = pl.reshape(beta, [_Q_TILE, 1])
            oi_new_scaled = pl.row_expand_mul(oi_new_tile, beta_dn)
            oi_updated = pl.add(oi_scaled, oi_new_scaled)

            mi_new_dn = pl.reshape(mi_new, [_Q_TILE, 1])
            li_updated_dn = pl.reshape(li_updated, [_Q_TILE, 1])

            mi_out = pl.store(mi_new_dn, [0, 0], mi)
            li_out = pl.store(li_updated_dn, [0, 0], li)
            oi_out = pl.store(oi_updated, [0, 0], oi)

            if is_last:
                dst_tile = pl.row_expand_div(oi_updated, li_updated_dn)
                dst_out = pl.store(dst_tile, [0, 0], dst)
            else:
                zero_tile = pl.tile.full([_Q_TILE, _HEAD_DIM], dtype=pl.FP32, value=0.0)
                dst_out = pl.store(zero_tile, [0, 0], dst)

        return mi_out, li_out, oi_out, dst_out

    # -----------------------------------------------------------------------
    # Program definition
    # -----------------------------------------------------------------------

    @pl.program
    class DynamicPagedAttentionProgram:
        """Paged attention with dynamic-shape InCore kernels (online softmax).

        InCore kernels are defined inside build_dynamic_paged_attention_program()
        and capture _Q_TILE, _HEAD_DIM, _BLOCK_SIZE as closure variables for
        load tile sizes, while their type annotations reference module-level
        pl.dynamic() variables (Q_HEADS, HEAD_DIM_DYN, BLOCK_SIZE_DYN).
        The 5-kernel pipeline and orchestration loops are identical in structure
        to the static version in paged_attention_example.py.
        """

        @pl.function(type=pl.FunctionType.Orchestration)
        def paged_attention(
            self,
            query: pl.Tensor[[QUERY_ROWS_DYN, HEAD_DIM_DYN], pl.BF16],
            key_cache: pl.Tensor[[KEY_CACHE_ROWS_DYN, HEAD_DIM_DYN], pl.BF16],
            value_cache: pl.Tensor[[KEY_CACHE_ROWS_DYN, HEAD_DIM_DYN], pl.BF16],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            context_lens: pl.Tensor[[BATCH_DYN], pl.INT32],
            out: pl.Tensor[[QUERY_ROWS_DYN, HEAD_DIM_DYN], pl.FP32],
        ) -> pl.Tensor[[QUERY_ROWS_DYN, HEAD_DIM_DYN], pl.FP32]:
            """Paged attention orchestration with dynamic-shape InCore kernels.

            Same 5-stage pipeline as the static version (init_inplace,
            qk_matmul, softmax_prepare, pv_matmul, online_update).  InCore
            kernels are closures defined inside build_dynamic_paged_attention_program()
            and referenced here by their local names (dyn_kernel_*).
            Shape derivations: batch = context_lens.dim(0),
            head_dim = query.dim(1), num_heads = query.dim(0) // batch,
            block_num = block_table.dim(0) // batch,
            block_size = value_cache.dim(0) // block_table.dim(0).
            """
            batch_cfg: pl.Scalar[pl.INT64] = pl.tensor.dim(context_lens, 0)
            query_rows: pl.Scalar[pl.INT64] = pl.tensor.dim(query, 0)
            head_dim_cfg: pl.Scalar[pl.INT64] = pl.tensor.dim(query, 1)
            value_cache_rows: pl.Scalar[pl.INT64] = pl.tensor.dim(value_cache, 0)
            block_table_size: pl.Scalar[pl.INT64] = pl.tensor.dim(block_table, 0)
            num_heads_cfg: pl.Scalar[pl.INT64] = query_rows // batch_cfg
            block_size_cfg: pl.Scalar[pl.INT64] = value_cache_rows // block_table_size
            block_num_cfg: pl.Scalar[pl.INT64] = block_table_size // batch_cfg

            q_head_num = num_heads_cfg
            # ceil-divide: number of q_tile-sized chunks needed to cover all query heads
            q_loop_cfg = (q_head_num + q_tile - 1) // q_tile

            for b_idx in pl.range(batch_cfg):
                cur_seq = pl.tensor.read(context_lens, [b_idx])
                # ceil-divide: number of KV blocks touched by this request
                bn_this_batch = (cur_seq + block_size_cfg - 1) // block_size_cfg
                for q_idx in pl.range(q_loop_cfg):
                    # Row offset into the flat query tensor for this (batch, q_tile) tile
                    cur_offset = b_idx * q_head_num + q_idx * q_tile

                    # Allocate zero-initialised accumulators for the online softmax state
                    oi_buf: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.create_tensor(
                        [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                        dtype=pl.FP32,
                    )
                    li_buf: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor(
                        [q_tile, 1],
                        dtype=pl.FP32,  # type: ignore[reportArgumentType]
                    )
                    mi_buf: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor(
                        [q_tile, 1],
                        dtype=pl.FP32,  # type: ignore[reportArgumentType]
                    )
                    # Bind concrete tensor shapes to dynamic type annotations (no-op passthrough)
                    oi, li_update, mi_update = dyn_kernel_init_inplace(oi_buf, li_buf, mi_buf)

                    for bn in pl.range(bn_this_batch):
                        # Slice the query tile for this head-tile and batch entry
                        qi: pl.Tensor[[q_tile, head_dim_cfg], pl.BF16] = pl.slice(
                            query,
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [cur_offset, 0],
                        )

                        # Look up the physical block index in the page table
                        cur_block_idx = pl.tensor.read(block_table, [b_idx * block_num_cfg + bn])
                        # Number of valid tokens in this block (last block may be partial)
                        valid_len = pl.min(block_size_cfg, cur_seq - bn * block_size_cfg)
                        # Starting row of this physical block in the flat KV-cache pool
                        kv_block_row = cur_block_idx * block_size_cfg

                        # Slice the key block: key_cache is [KV_rows, head_dim]
                        kj: pl.Tensor[[block_size_cfg, head_dim_cfg], pl.BF16] = pl.slice(
                            key_cache,
                            [block_size_cfg, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [kv_block_row, 0],
                        )
                        # Slice the value block: value_cache is stored as [KV_rows, head_dim]
                        vj: pl.Tensor[[block_size_cfg, head_dim_cfg], pl.BF16] = pl.slice(
                            value_cache,
                            [block_size_cfg, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [kv_block_row, 0],
                        )

                        # Stage 1: QK matmul — sij[q_tile, block_size] = qi @ kj.T
                        sij_buf: pl.Tensor[[q_tile, block_size_cfg], pl.FP32] = pl.create_tensor(
                            [q_tile, block_size_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.FP32,
                        )
                        sij = dyn_kernel_qk_matmul(qi, kj, sij_buf)

                        # Mask out padding columns beyond valid_len to avoid
                        # including out-of-range tokens in the softmax
                        sij_valid: pl.Tensor[[q_tile, valid_len], pl.FP32] = pl.slice(
                            sij,
                            [q_tile, valid_len],  # type: ignore[reportArgumentType]
                            [0, 0],
                        )

                        # Stage 2: softmax prepare — scale, row_max (mi), exp, row_sum (li)
                        pij_f16_buf: pl.Tensor[[q_tile, block_size_cfg], pl.BF16] = pl.create_tensor(
                            [q_tile, block_size_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.BF16,
                        )
                        mi_sm_buf: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor(
                            [q_tile, 1],
                            dtype=pl.FP32,  # type: ignore[reportArgumentType]
                        )
                        li_sm_buf: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor(
                            [q_tile, 1],
                            dtype=pl.FP32,  # type: ignore[reportArgumentType]
                        )
                        pij_f16, mi, li = dyn_kernel_softmax_prepare(
                            sij_valid,
                            1.0,  # type: ignore[reportArgumentType]
                            pij_f16_buf,
                            mi_sm_buf,
                            li_sm_buf,  # type: ignore[reportArgumentType]
                        )

                        # Stage 3: PV matmul — oi_tmp[q_tile, head_dim] = pij @ vj
                        oi_tmp_buf: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.create_tensor(
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.FP32,
                        )
                        oi_tmp = dyn_kernel_pv_matmul(pij_f16, vj, oi_tmp_buf)

                        # Determine position flags for the online softmax update kernel
                        if bn == 0:
                            is_first: pl.Scalar[pl.INT64] = pl.yield_(1)  # type: ignore[reportArgumentType]
                        else:
                            is_first: pl.Scalar[pl.INT64] = pl.yield_(0)  # type: ignore[reportArgumentType]
                        if bn == bn_this_batch - 1:
                            is_last: pl.Scalar[pl.INT64] = pl.yield_(1)  # type: ignore[reportArgumentType]
                        else:
                            is_last: pl.Scalar[pl.INT64] = pl.yield_(0)  # type: ignore[reportArgumentType]

                        # Stage 4: online update — merge (mij, lij, oi_new) into (mi, li, oi)
                        # and write final normalised output on the last block
                        out_view_buf: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.slice(
                            out,
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [cur_offset, 0],
                        )
                        mi_update, li_update, oi, out_view = dyn_kernel_online_update(
                            mi,
                            li,
                            oi_tmp,
                            mi_update,
                            li_update,
                            oi,
                            out_view_buf,
                            is_first,
                            is_last,
                        )

            return out

    return DynamicPagedAttentionProgram


def golden(tensors: dict, params: dict | None = None) -> None:
    """Reference paged-attention computation (torch) for the dynamic-shape program.

    Derives all shape parameters from tensor dimensions instead of a config tensor,
    matching the orchestration function's pl.tensor.dim() derivations.
    Scale is hardcoded to 1.0 to match the orchestration function.

    Args:
        tensors: Dict mapping tensor names to torch tensors.
        params: Unused.
    """
    context_lens = tensors["context_lens"]
    query = tensors["query"]
    key_cache = tensors["key_cache"]
    value_cache = tensors["value_cache"]
    block_table_flat = tensors["block_table"]

    batch = context_lens.shape[0]
    num_heads = query.shape[0] // batch
    head_dim = query.shape[1]
    # Mirrors the orchestration function's pl.tensor.dim() derivation:
    # block_size = value_cache.rows / block_table.rows  (rows per physical block)
    block_size = value_cache.shape[0] // block_table_flat.shape[0]
    max_num_blocks_per_req = block_table_flat.shape[0] // batch
    scale = 1.0

    # Reshape flat tensors into 3-D views for batch-matmul
    query_3d = query.float().reshape(batch, num_heads, head_dim)
    total_pool_blocks = batch * max_num_blocks_per_req
    key_cache_3d = key_cache.float().reshape(total_pool_blocks, block_size, head_dim)
    value_cache_3d = value_cache.float().reshape(total_pool_blocks, block_size, head_dim)
    block_table = block_table_flat.reshape(batch, max_num_blocks_per_req)

    out = torch.zeros((batch, num_heads, head_dim), dtype=torch.float32)
    q_tile = 16
    # Maximum number of KV blocks across all requests in the batch
    max_bn = int((context_lens.max().item() + block_size - 1) // block_size)

    for q_offset in range(0, num_heads, q_tile):
        q_tile_size = min(q_tile, num_heads - q_offset)
        qi = query_3d[:, q_offset : q_offset + q_tile_size, :]  # [batch, q_tile, head_dim]
        # Online softmax state: sentinel init so the first block is handled by the
        # general update formula (mi=-inf makes alpha=0, beta=1 on the first merge).
        mi = torch.full((batch, q_tile_size, 1), float("-inf"))
        li = torch.zeros((batch, q_tile_size, 1))
        oi = torch.zeros((batch, q_tile_size, head_dim))

        for bn in range(max_bn):
            # valid_lens[b]: how many tokens of block bn are valid for request b
            valid_lens = torch.clamp(context_lens - bn * block_size, min=0, max=block_size)
            if not (valid_lens > 0).any():
                break
            block_indices = block_table[:, bn]  # [batch]
            kj_all = key_cache_3d[block_indices]  # [batch, block_size, head_dim]
            vj_all = value_cache_3d[block_indices]  # [batch, block_size, head_dim]

            # Stage 1: QK matmul — sij[batch, q_tile, block_size] = qi @ kj.T * scale
            sij = torch.bmm(qi, kj_all.transpose(1, 2)) * scale  # [batch, q_tile, block_size]
            # Mask padding columns so they don't affect row_max / row_sum
            pos = torch.arange(block_size).unsqueeze(0)  # [1, block_size]
            valid_mask = (pos < valid_lens.unsqueeze(1)).unsqueeze(1)  # [batch, 1, block_size]
            sij = sij.masked_fill(~valid_mask, float("-inf"))

            # Stage 2: softmax prepare — row_max, exp, cast, row_sum
            mij = sij.max(dim=-1, keepdim=True)[0].clamp(min=-1e30)  # [batch, q_tile, 1]
            pij = torch.exp(sij - mij).masked_fill(~valid_mask, 0.0)
            # Simulate BF16 cast to match the InCore kernel's precision
            pij = pij.to(torch.bfloat16).to(torch.float32)
            lij = pij.sum(dim=-1, keepdim=True)  # [batch, q_tile, 1]

            # Stage 3: PV matmul — oi_new[batch, q_tile, head_dim] = pij @ vj
            oi_new = torch.bmm(pij, vj_all)  # [batch, q_tile, head_dim]

            # Stage 4: online update — unconditional merge; the sentinel mi=-inf ensures
            # alpha=0/beta=1 on the first valid block, making explicit bn==0 branching
            # unnecessary and correct for heterogeneous per-request starting points.
            mi_new = torch.maximum(mi, mij)
            alpha = torch.exp(mi - mi_new)  # rescale factor for old accumulator
            beta = torch.exp(mij - mi_new)  # rescale factor for new block
            li = alpha * li + beta * lij
            oi = alpha * oi + beta * oi_new
            mi = mi_new

        assert (li > 0).any(), "No KV blocks processed for this query tile"
        out[:, q_offset : q_offset + q_tile_size, :] = oi / li

    tensors["out"][:] = out.reshape(batch * num_heads, head_dim)


def build_tensor_specs(
    batch: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_num_blocks_per_req: int,
    context_len: int,
) -> list[TensorSpec]:
    """Build the TensorSpec list matching the dynamic paged_attention orchestration signature.

    The dynamic program derives all shape parameters from tensor dimensions at
    runtime, so there is no config tensor (unlike the static version).

    Args:
        batch: Number of requests in the batch.
        num_heads: Number of query heads.
        head_dim: Per-head feature dimension.
        block_size: KV-cache block size (rows per physical block).
        max_num_blocks_per_req: Maximum number of KV blocks per request.
        context_len: Number of valid tokens per request.
    """
    query_rows = batch * num_heads
    key_cache_rows = batch * max_num_blocks_per_req * block_size
    block_table_flat_size = batch * max_num_blocks_per_req

    context_lens_data = torch.full((batch,), context_len, dtype=torch.int32)
    block_table_data = torch.randint(
        0, max(block_table_flat_size, 1), size=(batch, max_num_blocks_per_req), dtype=torch.int32
    ).flatten()

    return [
        TensorSpec("query", [query_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("key_cache", [key_cache_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("value_cache", [key_cache_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("block_table", [block_table_flat_size], torch.int32, init_value=block_table_data),
        TensorSpec("context_lens", [batch], torch.int32, init_value=context_lens_data),
        TensorSpec("out", [query_rows, head_dim], torch.float32, is_output=True),
    ]


def main():
    """Run a single dynamic paged-attention example and verify against the golden reference.

    Uses a large batch (64) with 16 query heads, 128-dim heads, and 128-token KV
    blocks.  The program is built once via build_dynamic_paged_attention_program()
    and executed on the target device with tolerance atol/rtol=2e-2.
    """
    batch = 64
    num_heads = 16
    head_dim = 128
    block_size = 128
    max_model_len = 32768
    context_len = 8192
    q_tile = 16
    max_num_blocks_per_req = max_model_len // block_size  # 256

    program = build_dynamic_paged_attention_program(
        q_tile=q_tile,
        head_dim=head_dim,
        block_size=block_size,
    )

    tensor_specs = build_tensor_specs(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks_per_req=max_num_blocks_per_req,
        context_len=context_len,
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
