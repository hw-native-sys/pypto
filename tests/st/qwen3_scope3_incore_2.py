# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Qwen3 Scope 3 — incore_2: Post-RMSNorm application + zero-init down_proj.

Isolated test for the third incore call in scope3 orchestration:
  1. Zero-init down_proj_tile (chunked loop)
  2. Apply RMSNorm: post_norm = cast(x * inv_rms * gamma, BF16)
"""

import pypto.language as pl

HIDDEN = 256
K_CHUNK = 128
BATCH_TILE = 16

HIDDEN_BLOCKS = HIDDEN // K_CHUNK


def build_program():
    @pl.program
    class Scope3Incore2:
        @pl.function(type=pl.FunctionType.Opaque)
        def scope3_incore_2(
            self,
            resid1_tile: pl.Tensor[[BATCH_TILE, HIDDEN], pl.FP32],
            inv_rms: pl.Tensor[[BATCH_TILE, 1], pl.FP32],
            post_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            post_norm_tile: pl.Tensor[[BATCH_TILE, HIDDEN], pl.BF16],
            down_proj_tile: pl.Tensor[[BATCH_TILE, HIDDEN], pl.FP32],
        ) -> tuple[pl.Tensor[[BATCH_TILE, HIDDEN], pl.FP32], pl.Tensor[[BATCH_TILE, HIDDEN], pl.BF16]]:
            with pl.auto_incore():
                # Zero-init down_proj_tile.
                for zi in pl.range(HIDDEN_BLOCKS):
                    z0 = zi * K_CHUNK
                    down_zero_chunk = pl.full([BATCH_TILE, K_CHUNK], dtype=pl.FP32, value=0.0)
                    down_proj_tile = pl.assemble(down_proj_tile, down_zero_chunk, [0, z0])

                # Apply RMSNorm: x * inv_rms * gamma → BF16.
                for kb in pl.range(HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                    gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                    normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                    post_norm_tile = pl.assemble(
                        post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0]
                    )
            return down_proj_tile, post_norm_tile

    return Scope3Incore2


def golden(tensors: dict, params: dict | None = None) -> None:
    import torch

    resid1_tile = tensors["resid1_tile"]  # [4, 5120] FP32
    inv_rms = tensors["inv_rms"]  # [4, 1] FP32
    post_rms_weight = tensors["post_rms_weight"]  # [1, 5120] FP32

    tensors["down_proj_tile"][:] = torch.zeros_like(tensors["down_proj_tile"])
    tensors["post_norm_tile"][:] = (resid1_tile * inv_rms * post_rms_weight).bfloat16()


def build_tensor_specs():
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("resid1_tile", [BATCH_TILE, HIDDEN], torch.float32, init_value=torch.randn),
        TensorSpec("inv_rms", [BATCH_TILE, 1], torch.float32, init_value=torch.randn),
        TensorSpec("post_rms_weight", [1, HIDDEN], torch.float32, init_value=torch.randn),
        TensorSpec("post_norm_tile", [BATCH_TILE, HIDDEN], torch.bfloat16, is_output=True),
        TensorSpec("down_proj_tile", [BATCH_TILE, HIDDEN], torch.float32, is_output=True),
    ]


def compile_and_run(
    platform: str = "a2a3",
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
            backend_type=BackendType.Ascend910B,
        ),
    )
    if not result.passed and result.error and "code_runner" in result.error:
        print("Result: COMPILE OK — device run skipped (code_runner not found).")
    if not result.passed and result.error:
        print(f"Result: {result.error}")
    return result


if __name__ == "__main__":
    compile_and_run()
