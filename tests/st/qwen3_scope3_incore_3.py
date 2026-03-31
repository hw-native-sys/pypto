# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Qwen3 Scope 3 — incore_3: MLP gate/up projections + SiLU activation.

Isolated test for a single iteration of the MLP loop in scope3 orchestration:
  gate_acc = matmul(post_norm, w_gate)   (accumulated over K_CHUNK blocks)
  up_acc   = matmul(post_norm, w_up)     (accumulated over K_CHUNK blocks)
  mlp_chunk = silu(gate_acc) * up_acc    (SiLU = x * sigmoid(x))
  mlp_chunk_bf16 = cast(mlp_chunk, BF16)

Uses w_gate/w_up of shape [HIDDEN, MLP_OUT_CHUNK] to test a single output chunk.
"""

import pypto.language as pl

HIDDEN = 5120
K_CHUNK = 128
MLP_OUT_CHUNK = 64
BATCH_TILE = 16

HIDDEN_BLOCKS = HIDDEN // K_CHUNK


def build_program():
    @pl.program
    class Scope3Incore3:
        @pl.function(type=pl.FunctionType.Opaque)
        def scope3_incore_3(
            self,
            post_norm_tile: pl.Tensor[[BATCH_TILE, HIDDEN], pl.BF16],
            w_gate: pl.Tensor[[HIDDEN, MLP_OUT_CHUNK], pl.BF16],
            w_up: pl.Tensor[[HIDDEN, MLP_OUT_CHUNK], pl.BF16],
            mlp_chunk_bf16_out: pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.BF16],
        ) -> pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.BF16]:
            with pl.auto_incore():
                gate_acc = pl.full([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                up_acc = pl.full([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32, value=0.0)

                for kb in pl.range(HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                    wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, 0])
                    wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, 0])
                    gate_acc = pl.add(gate_acc, pl.matmul(post_chunk, wg))
                    up_acc = pl.add(up_acc, pl.matmul(post_chunk, wu))

                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                mlp_chunk_bf16_out = pl.assemble(
                    mlp_chunk_bf16_out, pl.cast(mlp_chunk, target_type=pl.BF16), [0, 0]
                )
            return mlp_chunk_bf16_out

    return Scope3Incore3


def golden(tensors: dict, params: dict | None = None) -> None:
    import torch

    post_norm = tensors["post_norm_tile"]  # [4, 5120] BF16
    w_gate = tensors["w_gate"]  # [5120, 64] BF16
    w_up = tensors["w_up"]  # [5120, 64] BF16

    gate = torch.matmul(post_norm.float(), w_gate.float())
    up = torch.matmul(post_norm.float(), w_up.float())
    mlp = (gate * torch.sigmoid(gate) * up).bfloat16()
    tensors["mlp_chunk_bf16_out"][:] = mlp


def build_tensor_specs():
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("post_norm_tile", [BATCH_TILE, HIDDEN], torch.bfloat16, init_value=torch.randn),
        TensorSpec("w_gate", [HIDDEN, MLP_OUT_CHUNK], torch.bfloat16, init_value=torch.randn),
        TensorSpec("w_up", [HIDDEN, MLP_OUT_CHUNK], torch.bfloat16, init_value=torch.randn),
        TensorSpec("mlp_chunk_bf16_out", [BATCH_TILE, MLP_OUT_CHUNK], torch.bfloat16, is_output=True),
    ]


def compile_and_run(
    platform: str = "a5sim",
    device_id: int = 11,
    dump_passes: bool = True,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    result = run(
        program=build_program(),
        tensor_specs=build_tensor_specs(),
        golden=golden,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=2e-2,
            atol=2e-2,
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
