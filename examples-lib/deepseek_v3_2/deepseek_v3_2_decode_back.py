# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

"""
DeepSeek V3.2-EXP single-layer decode BACK part (batch=16, max_seq=4096).

BACK boundary:
- start from combine buffer read
- run full residual + MLP + output path
"""

import os

import pypto.language as pl


BATCH = 16
MAX_SEQ = 4096

HIDDEN = 7168
INTERMEDIATE = 18432
NUM_HEADS = 128
V_HEAD_DIM = 128
ATTN_OUT = NUM_HEADS * V_HEAD_DIM
EP_NODES = 128

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# Increase tile sizes to encourage larger mixed-kernel fusion regions
# (notably for decode_back_layer_incore_0/1).
K_CHUNK = 512
Q_OUT_CHUNK = 128
MLP_OUT_CHUNK = 512
BATCH_TILE = 4


def build_deepseek_v3_2_decode_back_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    attn_out_size: int = ATTN_OUT,
    ep_nodes: int = EP_NODES,
):
    BATCH_CFG = batch
    HIDDEN_CFG = hidden_size
    INTER_CFG = intermediate_size
    ATTN_OUT_CFG = attn_out_size
    EP_NODES_CFG = ep_nodes

    ATTN_BLOCKS = (ATTN_OUT_CFG + K_CHUNK - 1) // K_CHUNK
    HIDDEN_BLOCKS = (HIDDEN_CFG + K_CHUNK - 1) // K_CHUNK
    Q_OUT_BLOCKS = (HIDDEN_CFG + Q_OUT_CHUNK - 1) // Q_OUT_CHUNK
    MLP_OUT_BLOCKS = (INTER_CFG + MLP_OUT_CHUNK - 1) // MLP_OUT_CHUNK

    @pl.program
    class DeepSeekV32DecodeBack:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_back_layer(
            self,
            hidden_states: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            node_id_t: pl.Tensor[[1], pl.INT32],
            # combine buffer from cross-node communication
            combine_buf: pl.Tensor[[EP_NODES_CFG, BATCH_CFG, ATTN_OUT_CFG], pl.BF16],
            wo: pl.Tensor[[ATTN_OUT_CFG, HIDDEN_CFG], pl.BF16],
            post_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_down: pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.BF16],
            out: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
        ) -> pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16]:
            with pl.auto_incore():
                node_id = pl.tensor.read(node_id_t, [0])
                combined = pl.create_tensor([BATCH_CFG, ATTN_OUT_CFG], dtype=pl.FP32)
                # Read combine results from this node view.
                for b in pl.parallel(0, BATCH_CFG, 1, chunk=4):
                    row = pl.cast(pl.slice(combine_buf, [1, ATTN_OUT_CFG], [node_id, b, 0]), target_type=pl.FP32)
                    combined = pl.assemble(combined, row, [b, 0])

                # Scope: output projection + residual + post-rms + MLP + residual.
                for b0 in pl.range(0, BATCH_CFG, BATCH_TILE):
                    resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)

                    # O projection and residual.
                    for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=8):
                        o0 = ob * Q_OUT_CHUNK
                        o_acc = pl.create_tensor([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                        o_acc = pl.mul(o_acc, 0.0)
                        for kb in pl.range(ATTN_BLOCKS):
                            k0 = kb * K_CHUNK
                            a_chunk = pl.cast(pl.slice(combined, [BATCH_TILE, K_CHUNK], [b0, k0]), target_type=pl.BF16)
                            w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                            o_acc = pl.add(o_acc, pl.matmul(a_chunk, w_chunk))
                        resid = pl.cast(pl.slice(hidden_states, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0]), target_type=pl.FP32)
                        resid1_tile = pl.assemble(resid1_tile, pl.add(o_acc, resid), [0, o0])

                    # Post RMSNorm.
                    sq_sum = pl.create_tensor([BATCH_TILE, 1], dtype=pl.FP32)
                    sq_sum = pl.mul(sq_sum, 0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        sq_sum = pl.add(sq_sum, pl.row_sum(pl.mul(x_chunk, x_chunk)))
                    inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))

                    post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.BF16)
                    down_proj_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)
                    down_proj_tile = pl.mul(down_proj_tile, 0.0)

                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                        post_norm_tile = pl.assemble(post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0])

                    # MLP.
                    for ob in pl.range(MLP_OUT_BLOCKS):
                        o0 = ob * MLP_OUT_CHUNK
                        gate_acc = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                        up_acc = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                        gate_acc = pl.mul(gate_acc, 0.0)
                        up_acc = pl.mul(up_acc, 0.0)

                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            gate_acc = pl.add(gate_acc, pl.matmul(post_chunk, wg))
                            up_acc = pl.add(up_acc, pl.matmul(post_chunk, wu))

                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)

                        for dob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=8):
                            d0 = dob * Q_OUT_CHUNK
                            down_prev = pl.slice(down_proj_tile, [BATCH_TILE, Q_OUT_CHUNK], [0, d0])
                            w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, Q_OUT_CHUNK], [o0, d0])
                            down_next = pl.add(down_prev, pl.matmul(mlp_chunk_bf16, w_down_chunk))
                            down_proj_tile = pl.assemble(down_proj_tile, down_next, [0, d0])

                    # Final residual and output.
                    for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=8):
                        o0 = ob * Q_OUT_CHUNK
                        down_acc = pl.add(
                            pl.slice(down_proj_tile, [BATCH_TILE, Q_OUT_CHUNK], [0, o0]),
                            pl.slice(resid1_tile, [BATCH_TILE, Q_OUT_CHUNK], [0, o0]),
                        )
                        out = pl.assemble(out, pl.cast(down_acc, target_type=pl.BF16), [b0, o0])

            return out

    return DeepSeekV32DecodeBack


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    attn_out_size: int = ATTN_OUT,
    ep_nodes: int = EP_NODES,
):
    import torch  # type: ignore[import]
    from pypto.runtime import TensorSpec

    node_id_data = torch.tensor([0], dtype=torch.int32)

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("node_id_t", [1], torch.int32, init_value=node_id_data),
        TensorSpec("combine_buf", [ep_nodes, batch, attn_out_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wo", [attn_out_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32, init_value=torch.randn),
        TensorSpec("w_gate", [hidden_size, intermediate_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("w_up", [hidden_size, intermediate_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("w_down", [intermediate_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("out", [batch, hidden_size], torch.bfloat16, is_output=True),
    ]


def compile_and_run(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    attn_out_size: int = ATTN_OUT,
    ep_nodes: int = EP_NODES,
    platform: str = "a2a3",
    device_id: int = 11,
    work_dir: str | None = None,
    dump_passes: bool = True,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    program = build_deepseek_v3_2_decode_back_program(
        batch=batch,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        attn_out_size=attn_out_size,
        ep_nodes=ep_nodes,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        attn_out_size=attn_out_size,
        ep_nodes=ep_nodes,
    )

    if work_dir is None:
        work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "deepseek_v3_2_decode_back_dump"))

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=None,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=2e-2,
            atol=2e-2,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=BackendType.CCE,
            work_dir=work_dir,
        ),
    )
    if not result.passed and result.error and "code_runner" in result.error:
        print("Result: COMPILE OK — device run skipped (code_runner not found).")
        print("  Generated kernels/orchestration:", work_dir)
        return result
    if not result.passed and result.error:
        print(f"Result: {result.error}")
        print("  Pass dumps may still have been written to:", work_dir)
    else:
        print("  Generated kernels/orchestration:", work_dir)
    return result


if __name__ == "__main__":
    compile_and_run()
