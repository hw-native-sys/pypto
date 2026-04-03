# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Qwen3 Scope 3: output projection + residual + post RMSNorm + MLP + residual.

Extracted from the full single-layer decode program to test this scope in
isolation. Covers:
  1. Output projection: attn_out × wo, accumulated in Q_OUT_CHUNK tiles
  2. Residual addition with hidden_states
  3. Post-attention RMSNorm
  4. MLP: gate/up projections, SiLU activation, down projection
  5. Final residual addition
"""

import pypto.language as pl

BATCH = 16
HIDDEN = 5120
INTERMEDIATE = 25600

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

K_CHUNK = 128
Q_OUT_CHUNK = 64
MLP_OUT_CHUNK = 64
BATCH_TILE = 16


def build_qwen3_scope3_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    BATCH_CFG = batch
    HIDDEN_CFG = hidden_size
    INTER_CFG = intermediate_size

    HIDDEN_BLOCKS = HIDDEN_CFG // K_CHUNK
    Q_OUT_BLOCKS = HIDDEN_CFG // Q_OUT_CHUNK
    MLP_OUT_BLOCKS = INTER_CFG // MLP_OUT_CHUNK

    @pl.program
    class Qwen3Scope3:
        @pl.function(type=pl.FunctionType.Opaque)
        def scope3(
            self,
            attn_out: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            hidden_states: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            wo: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.BF16],
            post_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_down: pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.BF16],
            out: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
        ) -> pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16]:
            with pl.auto_incore(split=pl.SplitMode.UP_DOWN):
                for b0 in pl.range(0, BATCH_CFG, BATCH_TILE):
                    resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)
                    for ob in pl.parallel(0, Q_OUT_BLOCKS):
                        o0 = ob * Q_OUT_CHUNK
                        zero_resid1 = pl.full([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        resid1_tile = pl.assemble(resid1_tile, zero_resid1, [0, o0])

                    # Output projection: attn_out × wo, tiled by Q_OUT_CHUNK.
                    for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=8):
                        o0 = ob * Q_OUT_CHUNK
                        o_acc = pl.full([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            a_chunk = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, k0])
                            w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                            o_acc = pl.add(o_acc, pl.matmul(a_chunk, w_chunk))
                        resid = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0]),
                            target_type=pl.FP32,
                        )
                        resid1_tile = pl.assemble(resid1_tile, pl.add(o_acc, resid), [0, o0])

                    # Post-attention RMSNorm: compute inv_rms over resid1_tile.
                    sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH_TILE]))
                    inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))

                    # Normalize and zero-init down_proj accumulator.
                    post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.BF16)
                    down_proj_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)
                    for zi in pl.range(HIDDEN_BLOCKS):
                        z0 = zi * K_CHUNK
                        down_zero_chunk = pl.full([BATCH_TILE, K_CHUNK], dtype=pl.FP32, value=0.0)
                        down_proj_tile = pl.assemble(down_proj_tile, down_zero_chunk, [0, z0])

                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, pl.reshape(inv_rms, [BATCH_TILE, 1])), gamma)
                        post_norm_tile = pl.assemble(
                            post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0]
                        )

                    # MLP: gate/up projections + SiLU + down projection.
                    for ob in pl.range(MLP_OUT_BLOCKS):
                        o0 = ob * MLP_OUT_CHUNK
                        gate_acc = pl.full([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        up_acc = pl.full([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32, value=0.0)

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

                        for dob in pl.parallel(0, HIDDEN_BLOCKS, 1, chunk=4):
                            d0 = dob * K_CHUNK
                            down_prev = pl.slice(down_proj_tile, [BATCH_TILE, K_CHUNK], [0, d0])
                            w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [o0, d0])
                            down_next = pl.add(down_prev, pl.matmul(mlp_chunk_bf16, w_down_chunk))
                            down_proj_tile = pl.assemble(down_proj_tile, down_next, [0, d0])

                    # Final residual: down_proj + resid1, write to output.
                    for ob in pl.parallel(0, HIDDEN_BLOCKS, 1, chunk=4):
                        o0 = ob * K_CHUNK
                        down_acc = pl.add(
                            pl.slice(down_proj_tile, [BATCH_TILE, K_CHUNK], [0, o0]),
                            pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, o0]),
                        )
                        out = pl.assemble(out, pl.cast(down_acc, target_type=pl.BF16), [b0, o0])

            return out

    return Qwen3Scope3


def golden(tensors: dict, params: dict | None = None) -> None:
    """Reference computation for Scope 3.

    Steps:
      1. Output projection: attn_out (cast BF16) × wo, FP32 accumulation + residual
      2. Post-attention RMSNorm
      3. SwiGLU MLP: gate/up projections → silu(gate) * up → down projection
      4. Final residual addition → BF16 output
    """
    import torch

    attn_out = tensors["attn_out"]  # [B, H], FP32
    hidden_states = tensors["hidden_states"]  # [B, H], BF16
    wo = tensors["wo"]  # [H, H], BF16
    post_rms_weight = tensors["post_rms_weight"]  # [1, H], FP32
    w_gate = tensors["w_gate"]  # [H, I], BF16
    w_up = tensors["w_up"]  # [H, I], BF16
    w_down = tensors["w_down"]  # [I, H], BF16

    H = attn_out.shape[1]
    eps = 1e-6

    # 1. Output projection (BF16 inputs, FP32 accumulation) + residual.
    o_proj = torch.matmul(attn_out.float(), wo.float())
    resid1 = o_proj + hidden_states.float()

    # 2. Post-attention RMSNorm.
    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    # 3. SwiGLU MLP: gate/up projections, silu activation, down projection.
    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = torch.matmul(mlp_bf16.float(), w_down.float())

    # 4. Final residual + cast to BF16.
    tensors["out"][:] = (down + resid1).bfloat16()


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    import torch  # type: ignore[import]
    from pypto.runtime import TensorSpec

    def xavier_bf16(shape: list[int]) -> torch.Tensor:
        """Generate in FP32 with Xavier (1/sqrt(fan_in)) scaling, then cast to BF16."""
        fan_in = shape[1]
        return (torch.randn(shape, dtype=torch.float32) / (fan_in ** 0.5)).to(torch.bfloat16)

    def xavier_fp32(shape: list[int]) -> torch.Tensor:
        """Generate in FP32 with Xavier (1/sqrt(fan_in)) scaling."""
        fan_in = shape[1]
        return torch.randn(shape, dtype=torch.float32) / (fan_in ** 0.5)

    return [
        TensorSpec("attn_out", [batch, hidden_size], torch.bfloat16,
                   init_value=xavier_bf16([batch, hidden_size])),
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16,
                   init_value=xavier_bf16([batch, hidden_size])),
        TensorSpec("wo", [hidden_size, hidden_size], torch.bfloat16,
                   init_value=xavier_bf16([hidden_size, hidden_size])),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32,
                   init_value=xavier_fp32([1, hidden_size])),
        TensorSpec("w_gate", [hidden_size, intermediate_size], torch.bfloat16,
                   init_value=xavier_bf16([hidden_size, intermediate_size])),
        TensorSpec("w_up", [hidden_size, intermediate_size], torch.bfloat16,
                   init_value=xavier_bf16([hidden_size, intermediate_size])),
        TensorSpec("w_down", [intermediate_size, hidden_size], torch.bfloat16,
                   init_value=xavier_bf16([intermediate_size, hidden_size])),
        TensorSpec("out", [batch, hidden_size], torch.bfloat16, is_output=True),
    ]


def compile_and_run(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    platform: str = "a5",
    device_id: int = 0,
    work_dir: str | None = None,
    dump_passes: bool = True,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    program = build_qwen3_scope3_program(
        batch=batch,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    tensor_specs = build_tensor_specs(
        batch=batch,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-3,
            atol=1e-3,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=BackendType.Ascend950,
        ),
    )
    if not result.passed and result.error and "code_runner" in result.error:
        print("Result: COMPILE OK — device run skipped (code_runner not found).")
    if not result.passed and result.error:
        print(f"Result: {result.error}")
    return result


if __name__ == "__main__":
    compile_and_run()
