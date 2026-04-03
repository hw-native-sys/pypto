# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Qwen3 Scope 3 — incore_4: Down projection accumulation.

Isolated test for a single iteration of the down-projection in scope3 orchestration:
  down_proj_tile += matmul(mlp_chunk_bf16, w_down_chunk)  (parallel over HIDDEN_BLOCKS, chunk=4)

Uses w_down of shape [MLP_OUT_CHUNK, HIDDEN] to test a single MLP output slice.
"""

import pypto.language as pl

HIDDEN = 5120
K_CHUNK = 128
MLP_OUT_CHUNK = 64
BATCH_TILE = 16

HIDDEN_BLOCKS = HIDDEN // K_CHUNK


def build_program():
    @pl.program
    class Scope3Incore4:
        @pl.function(type=pl.FunctionType.Opaque)
        def scope3_incore_4(
            self,
            mlp_chunk_bf16: pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.BF16],
            w_down: pl.Tensor[[MLP_OUT_CHUNK, HIDDEN], pl.BF16],
            down_proj_tile: pl.Tensor[[BATCH_TILE, HIDDEN], pl.FP32],
        ) -> pl.Tensor[[BATCH_TILE, HIDDEN], pl.FP32]:
            with pl.auto_incore(split=pl.SplitMode.UP_DOWN):
                for dob in pl.parallel(0, HIDDEN_BLOCKS, 1, chunk=4):
                    d0 = dob * K_CHUNK
                    down_prev = pl.slice(down_proj_tile, [BATCH_TILE, K_CHUNK], [0, d0])
                    w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [0, d0])
                    down_next = pl.add(down_prev, pl.matmul(mlp_chunk_bf16, w_down_chunk))
                    down_proj_tile = pl.assemble(down_proj_tile, down_next, [0, d0])
            return down_proj_tile

    return Scope3Incore4


def golden(tensors: dict, params: dict | None = None) -> None:
    import torch

    mlp_chunk = tensors["mlp_chunk_bf16"]  # [4, 64] BF16
    w_down = tensors["w_down"]  # [64, 5120] BF16
    down_prev = tensors["down_proj_tile"]  # [4, 5120] FP32

    tensors["down_proj_tile"][:] = down_prev + torch.matmul(mlp_chunk.float(), w_down.float())


def build_tensor_specs():
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("mlp_chunk_bf16", [BATCH_TILE, MLP_OUT_CHUNK], torch.bfloat16, init_value=torch.randn),
        TensorSpec("w_down", [MLP_OUT_CHUNK, HIDDEN], torch.bfloat16, init_value=torch.randn),
        TensorSpec(
            "down_proj_tile", [BATCH_TILE, HIDDEN], torch.float32, is_output=True, init_value=torch.randn
        ),
    ]


def compile_and_run(
    platform: str = "a5",
    device_id: int = 0,
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
