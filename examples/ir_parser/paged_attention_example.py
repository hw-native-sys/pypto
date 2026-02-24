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

Builds a simplified paged attention orchestration function using the PyPTO DSL.
Generates C++ orchestration code similar to target_page_attn_codegen.cpp.

Simplified 16x16 version:
  - Query:       [batch * num_heads, head_dim] BF16
  - Key cache:   [total_blocks * block_size, head_dim] BF16
  - Value cache: [total_blocks * block_size, head_dim] BF16
  - Output:      [batch * num_heads, head_dim] FP32

Task graph per (batch, block) iteration:
  task0: sij     = qi @ kj^T           (kernel_qk_matmul,       CUBE)
  task1: pij, mi, li = softmax(sij)    (kernel_softmax_prepare, VECTOR)
  task2: oi_tmp  = pij @ vj            (kernel_pv_matmul,       CUBE)
  task3: online_update(mi, li, oi_tmp) (kernel_online_update,   VECTOR)
"""

import os

import pypto.language as pl
from pypto import ir
from pypto.backend import BackendType


@pl.program
class PagedAttentionProgram:
    """Paged attention program with CUBE and VECTOR kernels."""

    # ── VECTOR kernel: init inplace tensors ─────────────────────────────
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_init_inplace(
        self,
        oi: pl.Tensor[[16, 16], pl.FP32],
        li: pl.Tensor[[16], pl.FP32],
        mi: pl.Tensor[[16], pl.FP32],
    ) -> tuple[
        pl.Tensor[[16, 128], pl.FP32],
        pl.Tensor[[16], pl.FP32],
        pl.Tensor[[16], pl.FP32],
    ]:
        """Initialize inplace accumulators to zero (VECTOR)."""
        return oi, li, mi

    # ── CUBE kernel: QK matmul ──────────────────────────────────────────
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_qk_matmul(
        self,
        qi: pl.Tensor[[16, 128], pl.BF16],
        kj: pl.Tensor[[128, 128], pl.BF16],
        output: pl.Tensor[[16, 128], pl.FP32],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        """QK matmul: sij = qi @ kj (CUBE)."""
        qi_tile: pl.Tile[[16, 16], pl.BF16] = pl.load(qi, [0, 0], [16, 16])
        kj_tile: pl.Tile[[16, 16], pl.BF16] = pl.load(kj, [0, 0], [16, 16])
        result: pl.Tile[[16, 16], pl.FP32] = pl.matmul(qi_tile, kj_tile)
        out: pl.Tensor[[16, 128], pl.FP32] = pl.l0c_store(result, [0, 0], [16, 16], output)
        return out

    # ── VECTOR kernel: softmax prepare ──────────────────────────────────
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_softmax_prepare(
        self,
        sij: pl.Tensor[[16, 128], pl.FP32],
        scale: pl.Scalar[pl.FP32],
        out_pij: pl.Tensor[[16, 128], pl.BF16],
        out_mi: pl.Tensor[[16], pl.FP32],
        out_li: pl.Tensor[[16], pl.FP32],
    ) -> tuple[
        pl.Tensor[[16, 128], pl.BF16],
        pl.Tensor[[16], pl.FP32],
        pl.Tensor[[16], pl.FP32],
    ]:
        """Softmax prepare: scale, row_max, exp, row_sum (VECTOR).

        Simplified kernel body — the orchestration codegen only uses the
        function signature, not the body implementation.
        """
        s_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(sij, [0, 0], [16, 16])
        scaled: pl.Tile[[16, 16], pl.FP32] = pl.mul(s_tile, scale)
        tmp_mi: pl.Tile[[16], pl.FP32] = pl.create_tile([16], dtype=pl.FP32)
        mi_tile: pl.Tile[[16], pl.FP32] = pl.row_max(scaled, tmp_mi)
        sub_tile: pl.Tile[[16, 16], pl.FP32] = pl.sub(scaled, mi_tile)
        exp_tile: pl.Tile[[16, 16], pl.FP32] = pl.exp(sub_tile)
        tmp_li: pl.Tile[[16], pl.FP32] = pl.create_tile([16], dtype=pl.FP32)
        li_tile: pl.Tile[[16], pl.FP32] = pl.row_sum(exp_tile, tmp_li)
        out_pij: pl.Tensor[[16, 128], pl.BF16] = pl.store(exp_tile, [0, 0], [16, 16], out_pij)
        out_mi: pl.Tensor[[16], pl.FP32] = pl.store(mi_tile, [0], [16], out_mi)
        out_li: pl.Tensor[[16], pl.FP32] = pl.store(li_tile, [0], [16], out_li)
        return out_pij, out_mi, out_li

    # ── CUBE kernel: PV matmul ──────────────────────────────────────────
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_pv_matmul(
        self,
        pij: pl.Tensor[[16, 128], pl.BF16],
        vj: pl.Tensor[[128, 128], pl.BF16],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """PV matmul: oi_tmp = pij @ vj (CUBE)."""
        p_tile: pl.Tile[[16, 16], pl.BF16] = pl.load(pij, [0, 0], [16, 16])
        v_tile: pl.Tile[[16, 16], pl.BF16] = pl.load(vj, [0, 0], [16, 16])
        result: pl.Tile[[16, 16], pl.FP32] = pl.matmul(p_tile, v_tile)
        out: pl.Tensor[[16, 128], pl.FP32] = pl.l0c_store(result, [0, 0], [16, 16], output)
        return out

    # ── VECTOR kernel: online update (inplace) ──────────────────────────
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_online_update(
        self,
        mi: pl.Tensor[[16], pl.FP32],
        li: pl.Tensor[[16], pl.FP32],
        oi_tmp: pl.Tensor[[16, 128], pl.FP32],
        mi_update: pl.Tensor[[16], pl.FP32],
        li_update: pl.Tensor[[16], pl.FP32],
        oi: pl.Tensor[[16, 128], pl.FP32],
        out_view: pl.Tensor[[16, 128], pl.FP32],
        is_first: pl.Scalar[pl.INT64],
        is_last: pl.Scalar[pl.INT64],
    ) -> tuple[
        pl.Tensor[[16], pl.FP32],
        pl.Tensor[[16], pl.FP32],
        pl.Tensor[[16, 128], pl.FP32],
        pl.Tensor[[16, 128], pl.FP32],
    ]:
        """Online softmax update with inplace mi/li/oi (VECTOR)."""
        mi_tile: pl.Tile[[16], pl.FP32] = pl.load(mi_update, [0], [16])
        li_tile: pl.Tile[[16], pl.FP32] = pl.load(li_update, [0], [16])
        oi_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(oi, [0, 0], [16, 16])
        out_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(out_view, [0, 0], [16, 16])
        mi_r: pl.Tensor[[16], pl.FP32] = pl.store(mi_tile, [0], [16], mi_update)
        li_r: pl.Tensor[[16], pl.FP32] = pl.store(li_tile, [0], [16], li_update)
        oi_r: pl.Tensor[[16, 128], pl.FP32] = pl.store(oi_tile, [0, 0], [16, 16], oi)
        out_r: pl.Tensor[[16, 128], pl.FP32] = pl.store(out_tile, [0, 0], [16, 16], out_view)
        return mi_r, li_r, oi_r, out_r

    # ── Orchestration function ──────────────────────────────────────────
    @pl.function(type=pl.FunctionType.Orchestration)
    def paged_attention(
        self,
        query: pl.Tensor[[32, 128], pl.BF16],  # 2 * 16, 128
        key_cache: pl.Tensor[[8192, 128], pl.BF16],  # [2 * 2048 * 128, 128]
        value_cache: pl.Tensor[[8192, 128], pl.BF16],  # [2 * 2048 * 128, 128]
        host_block_table: pl.Tensor[[2, 2048], pl.INT32],
        context_lens: pl.Tensor[[2], pl.INT32],
        out: pl.Tensor[[32, 128], pl.FP32],
        config: pl.Tensor[[7], pl.UINT64],
    ) -> pl.Tensor[[32, 128], pl.FP32]:
        """Paged attention orchestration.

        Config layout: [batch, num_heads, kv_head_num, head_dim, block_size, block_num, scale]
        Loops over batch and KV blocks, calling CUBE/VECTOR kernels.
        """
        batch: pl.Scalar[pl.UINT64] = pl.tensor.read(config, [0])
        num_heads: pl.Scalar[pl.UINT64] = pl.tensor.read(config, [1])
        head_dim: pl.Scalar[pl.UINT64] = pl.tensor.read(config, [3])
        block_size: pl.Scalar[pl.UINT64] = pl.tensor.read(config, [4])
        _block_num: pl.Scalar[pl.UINT64] = pl.tensor.read(config, [5])  # noqa: F841
        q_tile = pl.min(num_heads, 128)
        q_loop = (num_heads + q_tile - 1) // q_tile

        for b_idx in pl.range(batch):
            cur_seq = pl.tensor.read(context_lens, [b_idx])
            bn_this_batch = (cur_seq + block_size - 1) // block_size
            for q_idx in pl.range(q_loop):
                cur_offset = b_idx * num_heads + q_idx * q_tile

                # Create inplace accumulators for this q_tile group
                oi: pl.Tensor[[q_tile, head_dim], pl.FP32] = pl.create_tensor(
                    [q_tile, head_dim],  # type: ignore[reportArgumentType]
                    dtype=pl.FP32,
                )
                li_update: pl.Tensor[[q_tile], pl.FP32] = pl.create_tensor([q_tile], dtype=pl.FP32)  # type: ignore[reportArgumentType]
                mi_update: pl.Tensor[[q_tile], pl.FP32] = pl.create_tensor([q_tile], dtype=pl.FP32)  # type: ignore[reportArgumentType]

                # Initialize accumulators
                oi, li_update, mi_update = self.kernel_init_inplace(oi, li_update, mi_update)

                for bn in pl.range(bn_this_batch):
                    # Dynamic views into Q/K/V for current batch and block
                    qi: pl.Tensor[[q_tile, head_dim], pl.BF16] = pl.view(
                        query,
                        [q_tile, head_dim],  # type: ignore[reportArgumentType]
                        [cur_offset, 0],
                    )
                    cur_block_idx = pl.tensor.read(host_block_table, [b_idx, bn])
                    valid_len = pl.min(block_size, cur_seq - bn * block_size)
                    kj: pl.Tensor[[valid_len, head_dim], pl.BF16] = pl.view(
                        key_cache,
                        [valid_len, head_dim],  # type: ignore[reportArgumentType]
                        [cur_block_idx * block_size, 0],
                    )
                    vj: pl.Tensor[[valid_len, head_dim], pl.BF16] = pl.view(
                        value_cache,
                        [valid_len, head_dim],  # type: ignore[reportArgumentType]
                        [cur_block_idx * block_size, 0],
                    )

                    sij: pl.Tensor[[q_tile, valid_len], pl.FP32] = pl.create_tensor(
                        [q_tile, valid_len],  # type: ignore[reportArgumentType]
                        dtype=pl.FP32,
                    )

                    # QK matmul (CUBE)
                    sij = self.kernel_qk_matmul(qi, kj, sij)

                    pij: pl.Tensor[[q_tile, valid_len], pl.BF16] = pl.create_tensor(
                        [q_tile, valid_len],  # type: ignore[reportArgumentType]
                        dtype=pl.BF16,
                    )
                    mi: pl.Tensor[[q_tile], pl.FP32] = pl.create_tensor([q_tile], dtype=pl.FP32)  # type: ignore[reportArgumentType]
                    li: pl.Tensor[[q_tile], pl.FP32] = pl.create_tensor([q_tile], dtype=pl.FP32)  # type: ignore[reportArgumentType]
                    # Softmax prepare (VECTOR) — outputs are per-block mi/li, not the inplace accumulators
                    pij, mi, li = self.kernel_softmax_prepare(sij, 1.0, pij, mi, li)  # type: ignore[reportArgumentType]

                    oi_tmp: pl.Tensor[[q_tile, head_dim], pl.FP32] = pl.create_tensor(
                        [q_tile, head_dim],  # type: ignore[reportArgumentType]
                        dtype=pl.FP32,
                    )
                    # PV matmul (CUBE)
                    oi_tmp = self.kernel_pv_matmul(pij, vj, oi_tmp)

                    # Conditional flags
                    if bn == 0:
                        is_first: pl.Scalar[pl.UINT64] = pl.yield_(1)  # type: ignore[reportArgumentType]
                    else:
                        is_first: pl.Scalar[pl.UINT64] = pl.yield_(0)  # type: ignore[reportArgumentType]
                    if bn == bn_this_batch - 1:
                        is_last: pl.Scalar[pl.UINT64] = pl.yield_(1)  # type: ignore[reportArgumentType]
                    else:
                        is_last: pl.Scalar[pl.UINT64] = pl.yield_(0)  # type: ignore[reportArgumentType]

                    # Online update (VECTOR): mi/li are inputs from softmax,
                    # mi_update/li_update/oi are inplace accumulators, out_view is output
                    out_view: pl.Tensor[[q_tile, head_dim], pl.FP32] = pl.view(
                        out,
                        [q_tile, head_dim],  # type: ignore[reportArgumentType]
                        [cur_offset, 0],
                    )
                    mi_update, li_update, oi, out_view = self.kernel_online_update(
                        mi, li, oi_tmp, mi_update, li_update, oi, out_view, is_first, is_last
                    )

        return out


def main():
    """Build IR, compile, and display generated orchestration C++ code."""
    print("=" * 70)
    print("Paged Attention Orchestration Code Generation")
    print("=" * 70)

    program = PagedAttentionProgram
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
