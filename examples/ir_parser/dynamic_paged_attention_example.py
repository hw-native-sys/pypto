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

Dynamic shapes — InCore kernel type annotations use pl.dynamic() variables
(Q_HEADS, HEAD_DIM_DYN, BLOCK_SIZE_DYN) instead of literal numbers, while
load operations use closure variables (_Q_TILE, _HEAD_DIM, _BLOCK_SIZE) captured
from build_dynamic_paged_attention_program() parameters for tile sizes.
Use build_dynamic_paged_attention_program() to obtain a @pl.program class.

Dynamic InCore kernels are defined inside build_dynamic_paged_attention_program()
and capture _Q_TILE, _HEAD_DIM, _BLOCK_SIZE as closure variables so that the same
builder can produce programs with different tile sizes.
"""

# DSL function bodies are parsed as AST — dynamic var names look undefined to pyright.
# pyright: reportUndefinedVariable=false

import pypto.language as pl

# ---------------------------------------------------------------------------
# Module-level dynamic variables — used only in InCore kernel type annotations.
# Load operations inside the kernels use closure variables from the builder instead.
# ---------------------------------------------------------------------------

Q_HEADS = pl.dynamic("Q_HEADS")  # query tile rows   (e.g. 16)
HEAD_DIM_DYN = pl.dynamic("HEAD_DIM_DYN")  # head dimension    (e.g. 128)
BLOCK_SIZE_DYN = pl.dynamic("BLOCK_SIZE_DYN")  # KV block size     (e.g. 128)


# ---------------------------------------------------------------------------
# Program builders
# ---------------------------------------------------------------------------


def build_dynamic_paged_attention_program(
    batch: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_num_blocks_per_req: int,
    q_tile: int = 16,
):
    """Build a paged-attention @pl.program whose InCore kernels use dynamic shapes.

    InCore kernel tensor type annotations reference module-level pl.dynamic()
    variables (Q_HEADS, HEAD_DIM_DYN, BLOCK_SIZE_DYN) so their shapes are
    resolved at runtime from the concrete tensors passed by the orchestration
    function.  Load operations use closure variables (_Q_TILE, _HEAD_DIM,
    _BLOCK_SIZE) captured from the builder parameters for the tile sizes.

    The orchestration function is identical in structure to the static version
    in paged_attention_example.py (same pl.slice masking, same pipeline).

    Parameters
    ----------
    batch:                  number of requests in the batch
    num_heads:              number of query heads
    head_dim:               per-head feature dimension
    block_size:             KV-cache block size (rows per physical block)
    max_num_blocks_per_req: maximum number of KV blocks per request
    q_tile:                 query-head tile size
    """
    # Tile-size constants captured as closures by the InCore kernels below.
    _Q_TILE: int = q_tile
    _HEAD_DIM: int = head_dim
    _BLOCK_SIZE: int = block_size

    # -----------------------------------------------------------------------
    # Dynamic InCore kernels — defined here to capture _Q_TILE, _HEAD_DIM,
    # _BLOCK_SIZE as closure variables.  Type annotations still use the
    # module-level pl.dynamic() variables; only the load sizes differ.
    # -----------------------------------------------------------------------

    @pl.function(type=pl.FunctionType.InCore)
    def dyn_kernel_init_inplace(
        oi: pl.Out[pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32]],
        li: pl.Out[pl.Tensor[[Q_HEADS, 1], pl.FP32, pl.DN]],
        mi: pl.Out[pl.Tensor[[Q_HEADS, 1], pl.FP32, pl.DN]],
    ) -> tuple[
        pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32],
        pl.Tensor[[Q_HEADS, 1], pl.FP32, pl.DN],
        pl.Tensor[[Q_HEADS, 1], pl.FP32, pl.DN],
    ]:
        """No-op passthrough for type/shape binding.

        Returns oi, li, mi unchanged.  Actual zero-initialization of the
        accumulators is performed by pl.create_tensor before this function
        is called; this function exists solely to bind the concrete tensor
        shapes to the dynamic type annotations at the call site.
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
        """Softmax prepare: scale, row_max, exp, row_sum (VECTOR)."""
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
        mij: pl.Tensor[[Q_HEADS, 1], pl.FP32, pl.DN],
        lij: pl.Tensor[[Q_HEADS, 1], pl.FP32, pl.DN],
        oi_new: pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32],
        mi: pl.InOut[pl.Tensor[[Q_HEADS, 1], pl.FP32, pl.DN]],
        li: pl.InOut[pl.Tensor[[Q_HEADS, 1], pl.FP32, pl.DN]],
        oi: pl.InOut[pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32]],
        dst: pl.Out[pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32]],
        is_first: pl.Scalar[pl.BOOL],
        is_last: pl.Scalar[pl.BOOL],
    ) -> tuple[
        pl.Tensor[[Q_HEADS, 1], pl.FP32, pl.DN],
        pl.Tensor[[Q_HEADS, 1], pl.FP32, pl.DN],
        pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32],
        pl.Tensor[[Q_HEADS, HEAD_DIM_DYN], pl.FP32],
    ]:
        """Online softmax update with inplace mi/li/oi (VECTOR)."""
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

    query_rows = batch * num_heads
    key_cache_rows = batch * max_num_blocks_per_req * block_size
    out_rows = batch * num_heads
    block_table_flat_size = batch * max_num_blocks_per_req

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
            query: pl.Tensor[[query_rows, head_dim], pl.BF16],
            key_cache: pl.Tensor[[head_dim, key_cache_rows], pl.BF16, pl.DN],
            value_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
            block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
            context_lens: pl.Tensor[[batch], pl.INT32],
            out: pl.Tensor[[out_rows, head_dim], pl.FP32],
            config: pl.Tensor[[7], pl.INT64],
        ) -> pl.Tensor[[out_rows, head_dim], pl.FP32]:
            """Paged attention orchestration with dynamic-shape InCore kernels.

            Same 5-stage pipeline as the static version (init_inplace,
            qk_matmul, softmax_prepare, pv_matmul, online_update).  InCore
            kernels are closures defined inside build_dynamic_paged_attention_program()
            and referenced here by their local names (dyn_kernel_*).
            Config: [batch, num_heads, kv_head_num (unused), head_dim,
                     block_size, block_num, scale_bits (unused)]
            """
            batch_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
            num_heads_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
            head_dim_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [3])
            block_size_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [4])
            block_num_cfg: pl.Scalar[pl.INT64] = pl.tensor.read(config, [5])

            q_head_num = num_heads_cfg
            q_loop_cfg = (q_head_num + q_tile - 1) // q_tile

            for b_idx in pl.range(batch_cfg):
                cur_seq = pl.tensor.read(context_lens, [b_idx])
                bn_this_batch = (cur_seq + block_size_cfg - 1) // block_size_cfg
                for q_idx in pl.range(q_loop_cfg):
                    cur_offset = b_idx * q_head_num + q_idx * q_tile

                    oi: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.create_tensor(
                        [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
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
                    oi, li_update, mi_update = dyn_kernel_init_inplace(oi, li_update, mi_update)

                    for bn in pl.range(bn_this_batch):
                        qi: pl.Tensor[[q_tile, head_dim_cfg], pl.BF16] = pl.slice(
                            query,
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [cur_offset, 0],
                        )

                        cur_block_idx = pl.tensor.read(block_table, [b_idx * block_num_cfg + bn])
                        valid_len = pl.min(block_size_cfg, cur_seq - bn * block_size_cfg)
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
                        sij = dyn_kernel_qk_matmul(qi, kj, sij)

                        sij_valid: pl.Tensor[[q_tile, valid_len], pl.FP32] = pl.slice(
                            sij,
                            [q_tile, valid_len],  # type: ignore[reportArgumentType]
                            [0, 0],
                        )

                        pij_f16: pl.Tensor[[q_tile, block_size_cfg], pl.BF16] = pl.create_tensor(
                            [q_tile, block_size_cfg],  # type: ignore[reportArgumentType]
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
                        pij_f16, mi, li = dyn_kernel_softmax_prepare(
                            sij_valid,
                            1.0,  # type: ignore[reportArgumentType]
                            pij_f16,
                            mi,
                            li,  # type: ignore[reportArgumentType]
                        )

                        oi_tmp: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.create_tensor(
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.FP32,
                        )
                        oi_tmp = dyn_kernel_pv_matmul(pij_f16, vj, oi_tmp)

                        # Conditional flags
                        if bn == 0:
                            is_first: pl.Scalar[pl.INT64] = pl.yield_(1)  # type: ignore[reportArgumentType]
                        else:
                            is_first: pl.Scalar[pl.INT64] = pl.yield_(0)  # type: ignore[reportArgumentType]
                        if bn == bn_this_batch - 1:
                            is_last: pl.Scalar[pl.INT64] = pl.yield_(1)  # type: ignore[reportArgumentType]
                        else:
                            is_last: pl.Scalar[pl.INT64] = pl.yield_(0)  # type: ignore[reportArgumentType]

                        out_view: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.slice(
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
                            out_view,
                            is_first,
                            is_last,
                        )

            return out

    return DynamicPagedAttentionProgram


def main():
    batch = 64
    num_heads = 16
    head_dim = 128
    block_size = 128
    max_model_len = 32768
    max_num_blocks_per_req = max_model_len // block_size  # 256

    program = build_dynamic_paged_attention_program(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks_per_req=max_num_blocks_per_req,
    )
    print(f"Built program: {program}")
    print("\nDone.")


if __name__ == "__main__":
    main()
