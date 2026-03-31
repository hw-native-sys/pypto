# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Qwen3 Scope 3 — incore_1: RMSNorm (compute inv_rms).

Isolated test for the second incore call in scope3 orchestration:
  sq_sum = sum(x_chunk^2)  over K_CHUNK blocks
  inv_rms = rsqrt(sq_sum / HIDDEN + eps)
"""

import pypto.language as pl

HIDDEN = 5120
K_CHUNK = 128
BATCH_TILE = 16

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN
HIDDEN_BLOCKS = HIDDEN // K_CHUNK


def build_program():
    @pl.program
    class Scope3Incore1:
        @pl.function(type=pl.FunctionType.Opaque)
        def scope3_incore_1(
            self,
            resid1_tile: pl.Tensor[[BATCH_TILE, HIDDEN], pl.FP32],
            inv_rms_out: pl.Tensor[[1, BATCH_TILE], pl.FP32],
        ) -> pl.Tensor[[1, BATCH_TILE], pl.FP32]:
            with pl.auto_incore():
                sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                for kb in pl.range(HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                    tmp = pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, 16])
                    sq_sum = pl.add(sq_sum, tmp)
                inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))
                inv_rms_out = pl.assemble(inv_rms_out, inv_rms, [0, 0])
            return inv_rms_out

    return Scope3Incore1


def golden(tensors: dict, params: dict | None = None) -> None:
    import torch

    resid1_tile = tensors["resid1_tile"]  # [4, 5120] FP32
    variance = resid1_tile.pow(2).mean(dim=-1, keepdim=True)
    tensors["inv_rms_out"][:] = torch.rsqrt(variance + EPS)


def build_tensor_specs():
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("resid1_tile", [BATCH_TILE, HIDDEN], torch.float32, init_value=torch.randn),
        TensorSpec("inv_rms_out", [BATCH_TILE, 1], torch.float32, is_output=True),
    ]


def compile_and_run(
    platform: str = "a2a3",
    device_id: int = 14,
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
