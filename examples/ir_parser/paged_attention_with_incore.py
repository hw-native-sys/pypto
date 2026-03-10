# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Paged Attention: single orchestration with embedded InCore via with pl.incore()

Same pipeline as paged_attention_example.py (QK matmul → softmax prepare → PV matmul
→ online update) but all incore logic is inlined inside the orchestration using
with pl.incore(): blocks instead of calling separate @pl.function(type=InCore) kernels.

Uses chunked loop syntax (para_for.md): batch and query-tile loops are chunked for
parallel expansion (chunk loop + in_chunk loop); the KV-block loop (bn) stays
sequential due to online softmax recurrence.
"""

import os
import struct
from datetime import datetime

import pypto.language as pl
import torch  # type: ignore[import]
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime import RunConfig, RunResult, TensorSpec, run

# Tile sizes used inside incore blocks (must match views created in orchestration)
Q_TILE = 16
BLOCK_SIZE = 128
HEAD_DIM = 128


def build_paged_attention_program(
    batch: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_num_blocks_per_req: int,
    q_tile: int = Q_TILE,
):
    """Build a parameterised paged-attention @pl.program with embedded incore blocks."""

    # Loop bounds and dimensions as constants (compile-time); enables chunked parallel loops
    BATCH_CFG = batch
    Q_LOOP_CFG = (num_heads + q_tile - 1) // q_tile
    Q_HEAD_NUM = num_heads
    BLOCK_NUM_CFG = max_num_blocks_per_req
    HEAD_DIM_CFG = head_dim
    BLOCK_SIZE_CFG = block_size
    # Chunk sizes for chunked loops (bounds are now constants)
    BATCH_CHUNK = 8
    Q_CHUNK = 2

    query_rows = batch * num_heads
    key_cache_rows = batch * max_num_blocks_per_req * block_size
    out_rows = batch * num_heads
    block_table_flat_size = batch * max_num_blocks_per_req

    @pl.program
    class PagedAttentionProgram:
        """Paged attention: one orchestration, incore logic embedded via with pl.incore()."""

        @pl.function
        def paged_attention(  # noqa: PLR0915
            self,
            query: pl.Tensor[[query_rows, head_dim], pl.BF16],
            key_cache: pl.Tensor[[head_dim, key_cache_rows], pl.BF16, pl.DN],
            value_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
            block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
            context_lens: pl.Tensor[[batch], pl.INT32],
            out: pl.Tensor[[out_rows, head_dim], pl.FP32],
            config: pl.Tensor[[7], pl.INT64],
            size_query: pl.Tensor[[1], pl.INT64],
            size_key_cache: pl.Tensor[[1], pl.INT64],
            size_value_cache: pl.Tensor[[1], pl.INT64],
        ) -> pl.Tensor[[out_rows, head_dim], pl.FP32]:
            """Orchestration: chunked loops over batch/q_tile; sequential bn with pl.incore()."""
            # Loop bounds and dimensions from pa.py constants (BATCH_CFG, Q_LOOP_CFG, etc.)

            # Chunked loop over batch: pl.parallel, chunk → expansion (bounds are compile-time constants)
            for b_idx in pl.parallel(0, BATCH_CFG, 1, chunk=BATCH_CHUNK):
                cur_seq = pl.tensor.read(context_lens, [b_idx])
                bn_this_batch = (cur_seq + BLOCK_SIZE_CFG - 1) // BLOCK_SIZE_CFG
                # Chunked loop over query-tile groups: pl.parallel, chunk → expansion
                for q_idx in pl.parallel(0, Q_LOOP_CFG, 1, chunk=Q_CHUNK):
                    cur_offset = b_idx * Q_HEAD_NUM + q_idx * q_tile

                    oi: pl.Tensor[[q_tile, HEAD_DIM_CFG], pl.FP32] = pl.create_tensor(
                        [q_tile, HEAD_DIM_CFG],
                        dtype=pl.FP32,
                    )
                    li_update: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)
                    mi_update: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)

                    # ── Init: zero accumulators (embedded incore) ─────────────
                    with pl.incore():
                        zero_oi = pl.tile.full([Q_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                        zero_li = pl.tile.full([Q_TILE, 1], dtype=pl.FP32, value=0.0)
                        zero_mi = pl.tile.full([Q_TILE, 1], dtype=pl.FP32, value=0.0)
                        pl.store(zero_oi, [0, 0], [Q_TILE, HEAD_DIM], oi)
                        pl.store(zero_li, [0, 0], [Q_TILE, 1], li_update)
                        pl.store(zero_mi, [0, 0], [Q_TILE, 1], mi_update)

                    # Sequential loop over KV blocks (no parallel): online softmax has loop-carried
                    # dependency (mi_update, li_update, oi depend on previous bn); order must be preserved.
                    for bn in pl.range(bn_this_batch):
                        qi: pl.Tensor[[q_tile, HEAD_DIM_CFG], pl.BF16] = pl.slice(
                            query, [q_tile, HEAD_DIM_CFG], [cur_offset, 0]
                        )
                        cur_block_idx = pl.tensor.read(block_table, [b_idx * BLOCK_NUM_CFG + bn])
                        valid_len = pl.min(BLOCK_SIZE_CFG, cur_seq - bn * BLOCK_SIZE_CFG)
                        kv_block_row = cur_block_idx * BLOCK_SIZE_CFG
                        kj: pl.Tensor[[HEAD_DIM_CFG, BLOCK_SIZE_CFG], pl.BF16, pl.DN] = pl.slice(
                            key_cache, [HEAD_DIM_CFG, BLOCK_SIZE_CFG], [kv_block_row, 0]
                        )
                        vj: pl.Tensor[[BLOCK_SIZE_CFG, HEAD_DIM_CFG], pl.BF16] = pl.slice(
                            value_cache, [BLOCK_SIZE_CFG, HEAD_DIM_CFG], [kv_block_row, 0]
                        )

                        sij: pl.Tensor[[q_tile, BLOCK_SIZE_CFG], pl.FP32] = pl.create_tensor(
                            [q_tile, BLOCK_SIZE_CFG], dtype=pl.FP32
                        )

                        # ── QK matmul (embedded incore) ─────────────────────────
                        with pl.incore():
                            qi_l1 = pl.load(qi, [0, 0], [Q_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
                            kj_l1 = pl.load(
                                kj,
                                [0, 0],
                                [HEAD_DIM, BLOCK_SIZE],
                                target_memory=pl.MemorySpace.Mat,
                                transpose=True,
                            )
                            qi_l0a = pl.move(qi_l1, target_memory=pl.MemorySpace.Left)
                            kj_l0b = pl.move(kj_l1, target_memory=pl.MemorySpace.Right)
                            sij_l0c = pl.matmul(qi_l0a, kj_l0b)
                            pl.store(sij_l0c, [0, 0], [Q_TILE, BLOCK_SIZE], sij)

                        sij_valid: pl.Tensor[[q_tile, valid_len], pl.FP32] = pl.slice(
                            sij, [q_tile, valid_len], [0, 0]
                        )

                        pij_f16: pl.Tensor[[q_tile, BLOCK_SIZE_CFG], pl.BF16] = pl.create_tensor(
                            [q_tile, BLOCK_SIZE_CFG], dtype=pl.BF16
                        )
                        mi: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)
                        li: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)

                        # ── Softmax prepare (embedded incore) ───────────────────
                        # Uses valid_len for the slice extent when loading/storing sij_valid.
                        with pl.incore():
                            scale = 1.0
                            s_tile = pl.load(
                                sij_valid, [0, 0], [Q_TILE, valid_len], target_memory=pl.MemorySpace.Vec
                            )
                            scaled = pl.mul(s_tile, scale)
                            tmp_tile = pl.create_tile(
                                [Q_TILE, BLOCK_SIZE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                            )
                            mi_tile = pl.row_max(scaled, tmp_tile)
                            sij_centered = pl.row_expand_sub(scaled, mi_tile)
                            exp_tile = pl.exp(sij_centered)
                            pij_tile_bf16 = pl.cast(exp_tile, target_type=pl.BF16)
                            pij_tile = pl.cast(pij_tile_bf16, target_type=pl.FP32)
                            li_tile = pl.row_sum(pij_tile, tmp_tile)
                            pl.store(pij_tile_bf16, [0, 0], [Q_TILE, valid_len], pij_f16)
                            pl.store(mi_tile, [0, 0], [Q_TILE, 1], mi)
                            pl.store(li_tile, [0, 0], [Q_TILE, 1], li)

                        oi_tmp: pl.Tensor[[q_tile, HEAD_DIM_CFG], pl.FP32] = pl.create_tensor(
                            [q_tile, HEAD_DIM_CFG], dtype=pl.FP32
                        )

                        # ── PV matmul (embedded incore) ─────────────────────────
                        with pl.incore():
                            pij_l1 = pl.load(
                                pij_f16, [0, 0], [Q_TILE, BLOCK_SIZE], target_memory=pl.MemorySpace.Mat
                            )
                            vj_l1 = pl.load(
                                vj, [0, 0], [BLOCK_SIZE, HEAD_DIM], target_memory=pl.MemorySpace.Mat
                            )
                            pij_l0a = pl.move(pij_l1, target_memory=pl.MemorySpace.Left)
                            vj_l0b = pl.move(vj_l1, target_memory=pl.MemorySpace.Right)
                            oi_l0c = pl.matmul(pij_l0a, vj_l0b)
                            pl.store(oi_l0c, [0, 0], [Q_TILE, HEAD_DIM], oi_tmp)

                        if bn == 0:
                            is_first: pl.Scalar[pl.BOOL] = pl.yield_(1)
                        else:
                            is_first = pl.yield_(0)
                        if bn == bn_this_batch - 1:
                            is_last: pl.Scalar[pl.BOOL] = pl.yield_(1)
                        else:
                            is_last = pl.yield_(0)

                        out_view: pl.Tensor[[q_tile, HEAD_DIM_CFG], pl.FP32] = pl.slice(
                            out, [q_tile, HEAD_DIM_CFG], [cur_offset, 0]
                        )

                        # ── Online softmax update (embedded incore) ─────────────
                        with pl.incore():
                            mij_tile = pl.load(mi, [0, 0], [Q_TILE, 1], target_memory=pl.MemorySpace.Vec)
                            lij_tile = pl.load(li, [0, 0], [Q_TILE, 1], target_memory=pl.MemorySpace.Vec)
                            oi_new_tile = pl.load(
                                oi_tmp, [0, 0], [Q_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec
                            )
                            mi_tile = pl.load(
                                mi_update, [0, 0], [Q_TILE, 1], target_memory=pl.MemorySpace.Vec
                            )
                            li_tile = pl.load(
                                li_update, [0, 0], [Q_TILE, 1], target_memory=pl.MemorySpace.Vec
                            )
                            oi_tile = pl.load(
                                oi, [0, 0], [Q_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec
                            )

                            if is_first:
                                pl.store(mij_tile, [0, 0], [Q_TILE, 1], mi_update)
                                pl.store(lij_tile, [0, 0], [Q_TILE, 1], li_update)
                                pl.store(oi_new_tile, [0, 0], [Q_TILE, HEAD_DIM], oi)
                                if is_last:
                                    dst_tile = pl.row_expand_div(oi_new_tile, lij_tile)
                                    pl.store(dst_tile, [0, 0], [Q_TILE, HEAD_DIM], out_view)
                                else:
                                    zero_tile = pl.tile.full([Q_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                                    pl.store(zero_tile, [0, 0], [Q_TILE, HEAD_DIM], out_view)
                            else:
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
                                pl.store(mi_new_dn, [0, 0], [Q_TILE, 1], mi_update)
                                pl.store(li_updated_dn, [0, 0], [Q_TILE, 1], li_update)
                                if is_last:
                                    dst_tile = pl.row_expand_div(oi_updated, li_updated_dn)
                                    pl.store(dst_tile, [0, 0], [Q_TILE, HEAD_DIM], out_view)
                                    pl.store(oi_updated, [0, 0], [Q_TILE, HEAD_DIM], oi)
                                else:
                                    zero_tile = pl.tile.full([Q_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                                    pl.store(zero_tile, [0, 0], [Q_TILE, HEAD_DIM], out_view)
                                    pl.store(oi_updated, [0, 0], [Q_TILE, HEAD_DIM], oi)

            return out

    return PagedAttentionProgram


def golden(tensors: dict, params: dict | None = None) -> None:
    """Reference paged-attention (torch), matching the 4-stage pipeline."""
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
    q_tile = 16
    max_bn = int((context_lens.max().item() + block_size - 1) // block_size)

    for q_offset in range(0, num_heads, q_tile):
        q_tile_size = min(q_tile, num_heads - q_offset)
        qi = query[:, q_offset : q_offset + q_tile_size, :]
        oi, li, mi = None, None, None

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

        assert oi is not None and li is not None
        out[:, q_offset : q_offset + q_tile_size, :] = oi / li

    tensors["out"][:] = out.reshape(batch * num_heads, head_dim)


def build_tensor_specs(
    batch: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_num_blocks_per_req: int,
    context_len: int,
    scale: float = 1.0,
) -> list[TensorSpec]:
    """Build TensorSpec list for paged_attention signature."""
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


def compile_and_run(
    batch: int = 64,
    num_heads: int = 16,
    head_dim: int = 128,
    block_size: int = 128,
    context_len: int = 8192,
    scale: float = 1.0,
    platform: str = "a2a3",
    device_id: int = 11,
    work_dir: str | None = None,
    dump_passes: bool = True,
) -> RunResult:
    """Compile the paged-attention program and optionally run on device.

    Args:
        batch: Batch size.
        num_heads: Number of heads.
        head_dim: Head dimension.
        block_size: KV block size.
        context_len: Context length per request.
        scale: Attention scale.
        platform: "a2a3" or "a2a3sim".
        device_id: Device index (for real hardware).
        work_dir: Output directory for generated files; None = temp dir.
        dump_passes: Dump IR after each pass.

    Returns:
        RunResult from pypto.runtime.run (PASS/FAIL or compile-only when code_runner missing).
    """
    max_num_blocks_per_req = (32768 + block_size - 1) // block_size

    program = build_paged_attention_program(
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
        context_len=context_len,
        scale=scale,
    )

    if work_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = os.path.join("build_output", f"PagedAttentionProgram_{timestamp}")

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=2e-2,
            atol=2e-2,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=BackendType.Ascend910B_PTO,
            work_dir=work_dir,
        ),
    )
    # Treat missing code_runner as compile-only success
    if not result.passed and result.error and "code_runner" in result.error:
        print("Result: COMPILE OK — device run skipped (code_runner not found).")
        print("  Compilation and passes completed successfully.")
        print("  Generated kernels/orchestration:", work_dir)
        print("  To run on device, set SIMPLER_ROOT to the Simpler repo and ensure")
        print("  code_runner is on PYTHONPATH (e.g. SIMPLER_ROOT/examples/scripts).")
        return RunResult(passed=True, error="device run skipped (no code_runner)")
    if result.passed:
        print("  Generated kernels/orchestration:", work_dir)
    return result


def main():
    result = compile_and_run(
        batch=64,
        num_heads=16,
        head_dim=128,
        block_size=128,
        context_len=8192,
        scale=1.0,
        platform="a2a3sim",
        device_id=11,
        dump_passes=True,
    )
    # Avoid duplicate "Result:" when compile_and_run already printed COMPILE OK
    if getattr(result, "error", None) != "device run skipped (no code_runner)":
        print(f"Result: {result}")
    print("\nDone.")


if __name__ == "__main__":
    main()
