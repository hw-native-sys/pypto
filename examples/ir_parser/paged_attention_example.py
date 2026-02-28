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
"""

import os

import pypto.language as pl
from pypto import ir
from pypto.backend import BackendType


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
        """Paged attention program with CUBE and VECTOR kernels (online softmax)."""

        # ── VECTOR kernel: init inplace tensors ─────────────────────────────
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_init_inplace(
            self,
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

        # ── CUBE kernel: QK matmul ──────────────────────────────────────────
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_qk_matmul(
            self,
            qi: pl.Tensor[[16, 128], pl.BF16],
            kj: pl.Tensor[[128, 128], pl.BF16],
            output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            """QK matmul: sij = qi @ kj.T (CUBE). kj transposed before move to L0B."""
            qi_l1 = pl.load(qi, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            kj_l1 = pl.load(kj, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            qi_l0a = pl.move(qi_l1, target_memory=pl.MemorySpace.Left)
            kj_l0b = pl.move(kj_l1, target_memory=pl.MemorySpace.Right, transpose=True)
            sij_l0c = pl.matmul(qi_l0a, kj_l0b)
            out: pl.Tensor[[16, 128], pl.FP32] = pl.l0c_store(sij_l0c, [0, 0], [16, 128], output)
            return out

        # ── VECTOR kernel: softmax prepare ──────────────────────────────────
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_softmax_prepare(
            self,
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
            s_tile = pl.load(sij, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec)
            scaled = pl.mul(s_tile, scale)
            tmp_tile = pl.create_tile([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            mi_tile = pl.row_max(scaled, tmp_tile)
            sij_centered = pl.row_expand_sub(scaled, mi_tile)
            exp_tile = pl.exp(sij_centered)
            pij_tile_bf16 = pl.cast(exp_tile, target_type=pl.BF16)
            pij_tile = pl.cast(pij_tile_bf16, target_type=pl.FP32)
            li_tile = pl.row_sum(pij_tile, tmp_tile)
            out_pij = pl.store(pij_tile_bf16, [0, 0], [16, 128], out_pij)
            out_mi = pl.store(mi_tile, [0, 0], [16, 1], out_mi)
            out_li = pl.store(li_tile, [0, 0], [16, 1], out_li)
            return out_pij, out_mi, out_li

        # ── CUBE kernel: PV matmul ──────────────────────────────────────────
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_pv_matmul(
            self,
            pij: pl.Tensor[[16, 128], pl.BF16],
            vj: pl.Tensor[[128, 128], pl.BF16],
            output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            """PV matmul: oi_tmp = pij @ vj (CUBE)."""
            pij_l1 = pl.load(pij, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            vj_l1 = pl.load(vj, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            pij_l0a = pl.move(pij_l1, target_memory=pl.MemorySpace.Left)
            vj_l0b = pl.move(vj_l1, target_memory=pl.MemorySpace.Right)
            oi_l0c = pl.matmul(pij_l0a, vj_l0b)
            out = pl.l0c_store(oi_l0c, [0, 0], [16, 128], output)
            return out

        # ── VECTOR kernel: online update (inplace) ──────────────────────────
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_online_update(
            self,
            mij: pl.Tensor[[16, 1], pl.FP32],
            lij: pl.Tensor[[16, 1], pl.FP32],
            oi_new: pl.Tensor[[16, 128], pl.FP32],
            mi: pl.InOut[pl.Tensor[[16, 1], pl.FP32]],
            li: pl.InOut[pl.Tensor[[16, 1], pl.FP32]],
            oi: pl.InOut[pl.Tensor[[16, 128], pl.FP32]],
            dst: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            is_first: pl.Scalar[pl.BOOL],
            is_last: pl.Scalar[pl.BOOL],
        ) -> tuple[
            pl.Tensor[[16, 1], pl.FP32],
            pl.Tensor[[16, 1], pl.FP32],
            pl.Tensor[[16, 128], pl.FP32],
            pl.Tensor[[16, 128], pl.FP32],
        ]:
            """Online softmax update with inplace mi/li/oi (VECTOR)."""
            mij_tile = pl.load(mij, [0, 0], [16, 1], target_memory=pl.MemorySpace.Vec)
            lij_tile = pl.load(lij, [0, 0], [16, 1], target_memory=pl.MemorySpace.Vec)
            oi_new_tile = pl.load(oi_new, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec)
            mi_tile = pl.load(mi, [0, 0], [16, 1], target_memory=pl.MemorySpace.Vec)
            li_tile = pl.load(li, [0, 0], [16, 1], target_memory=pl.MemorySpace.Vec)
            oi_tile = pl.load(oi, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec)

            if is_first:
                mi_out = pl.store(mij_tile, [0, 0], [16, 1], mi)
                li_out = pl.store(lij_tile, [0, 0], [16, 1], li)
                oi_out = pl.store(oi_new_tile, [0, 0], [16, 128], oi)
                if is_last:
                    dst_tile = pl.row_expand_div(oi_new_tile, lij_tile)
                    dst_out = pl.store(dst_tile, [0, 0], [16, 128], dst)
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

                mi_out = pl.store(mi_new_dn, [0, 0], [16, 1], mi)
                li_out = pl.store(li_updated_dn, [0, 0], [16, 1], li)

                if is_last:
                    dst_tile = pl.row_expand_div(oi_updated, li_updated_dn)
                    dst_out = pl.store(dst_tile, [0, 0], [16, 128], dst)
                    oi_out = pl.store(oi_updated, [0, 0], [16, 128], oi)
                else:
                    zero_tile = pl.block.full([16, 128], dtype=pl.FP32, value=0.0)
                    dst_out = pl.store(zero_tile, [0, 0], [16, 128], dst)
                    oi_out = pl.store(oi_updated, [0, 0], [16, 128], oi)

            return mi_out, li_out, oi_out, dst_out

        # ── Orchestration function ──────────────────────────────────────────
        # Parameters: query, key_cache, value_cache, block_table, context_lens,
        #             out, config (7 tensors) + size_query, size_key_cache,
        #             size_value_cache (3 byte-size scalars)
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

                    # Initialize accumulators
                    oi, li_update, mi_update = self.kernel_init_inplace(oi, li_update, mi_update)

                    for bn in pl.range(bn_this_batch):
                        # Query view: row offset = b_idx * num_heads + q_idx * q_tile
                        qi: pl.Tensor[[q_tile, head_dim_cfg], pl.BF16] = pl.view(
                            query,
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [cur_offset, 0],
                        )

                        # Get block index from block_table
                        cur_block_idx = pl.tensor.read(block_table, [b_idx * block_num_cfg + bn])
                        valid_len = pl.min(block_size_cfg, cur_seq - bn * block_size_cfg)

                        # Key/Value views: physical block row = cur_block_idx * block_size
                        kv_block_row = cur_block_idx * block_size_cfg
                        kj: pl.Tensor[[block_size_cfg, head_dim_cfg], pl.BF16] = pl.view(
                            key_cache,
                            [block_size_cfg, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [kv_block_row, 0],
                        )
                        vj: pl.Tensor[[block_size_cfg, head_dim_cfg], pl.BF16] = pl.view(
                            value_cache,
                            [block_size_cfg, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [kv_block_row, 0],
                        )

                        sij: pl.Tensor[[q_tile, block_size_cfg], pl.FP32] = pl.create_tensor(
                            [q_tile, block_size_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.FP32,
                        )

                        # QK matmul (CUBE)
                        sij = self.kernel_qk_matmul(qi, kj, sij)
                        sij_valid: pl.Tensor[[q_tile, valid_len], pl.FP32] = pl.view(
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

                        # Softmax prepare (VECTOR) — use valid slice of sij
                        pij_f16, mi, li = self.kernel_softmax_prepare(sij_valid, 1.0, pij_f16, mi, li)  # type: ignore[reportArgumentType]

                        oi_tmp: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.create_tensor(
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            dtype=pl.FP32,
                        )
                        # PV matmul (CUBE)
                        oi_tmp = self.kernel_pv_matmul(pij_f16, vj, oi_tmp)

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
                        out_view: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.view(
                            out,
                            [q_tile, head_dim_cfg],  # type: ignore[reportArgumentType]
                            [cur_offset, 0],
                        )
                        mi_update, li_update, oi, out_view = self.kernel_online_update(
                            mi, li, oi_tmp, mi_update, li_update, oi, out_view, is_first, is_last
                        )

            return out

    return PagedAttentionProgram


def main():
    """Build IR, compile, and display generated orchestration C++ code."""
    print("=" * 70)
    print("Paged Attention Orchestration Code Generation")
    print("=" * 70)

    program = build_paged_attention_program(
        batch=2,
        num_heads=16,
        head_dim=128,
        block_size=128,
        max_num_blocks_per_req=32,
    )
    print(f"\nProgram: {program.name}")
    print(f"Functions: {[f.name for f in program.functions.values()]}")

    # Print IR preview
    print("\n[1] IR Preview:")
    print("-" * 70)
    ir_text = ir.python_print(program)
    lines = ir_text.split("\n")
    preview = min(60, len(lines))
    print("\n".join(lines[:preview]))
    if len(lines) > preview:
        print(f"\n... ({len(lines) - preview} more lines)")
    print("-" * 70)

    # Compile
    print("\n[2] Compiling...")
    output_dir = ir.compile(
        program,
        strategy=ir.OptimizationStrategy.Default,
        dump_passes=True,
        backend_type=BackendType.CCE,
    )
    print(f"Output: {output_dir}")

    # List generated files
    print("\n[3] Generated files:")
    for root, _dirs, files in os.walk(output_dir):
        for f in files:
            path = os.path.join(root, f)
            rel = os.path.relpath(path, output_dir)
            print(f"  - {rel} ({os.path.getsize(path)} bytes)")

    # Show orchestration code path
    orch_file = os.path.join(output_dir, "orchestration", "paged_attention.cpp")
    if os.path.exists(orch_file):
        print("\n[4] Generated Orchestration C++ path:")
        print(orch_file)

    print("\nDone.")


if __name__ == "__main__":
    main()
