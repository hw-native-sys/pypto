# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Qwen3 Scope 3 — incore_0: Output projection + residual.

Isolated test for the first incore call in scope3 orchestration:
  resid1_tile = matmul(cast(attn_out, BF16), wo) + cast(hidden_states, FP32)

Tiled as: parallel(Q_OUT_BLOCKS, chunk=8) × range(HIDDEN_BLOCKS) matmul accumulation.
"""

import pypto.language as pl

HIDDEN = 5120
K_CHUNK = 128
Q_OUT_CHUNK = 64
BATCH_TILE = 16

HIDDEN_BLOCKS = HIDDEN // K_CHUNK
Q_OUT_BLOCKS = HIDDEN // Q_OUT_CHUNK


def build_program():
    @pl.program
    class Scope3Incore0:
        @pl.function(type=pl.FunctionType.Opaque)
        def scope3_incore_0(
            self,
            attn_out: pl.Tensor[[BATCH_TILE, HIDDEN], pl.BF16],
            hidden_states: pl.Tensor[[BATCH_TILE, HIDDEN], pl.BF16],
            wo: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
            resid1_tile: pl.Tensor[[BATCH_TILE, HIDDEN], pl.FP32],
        ) -> pl.Tensor[[BATCH_TILE, HIDDEN], pl.FP32]:
            with pl.auto_incore(split=pl.SplitMode.UP_DOWN):
                for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=8):
                    o0 = ob * Q_OUT_CHUNK
                    o_acc = pl.full([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        a_chunk = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [0, k0])
                        w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                        o_acc = pl.add(o_acc, pl.matmul(a_chunk, w_chunk))
                    resid = pl.cast(
                        pl.slice(hidden_states, [BATCH_TILE, Q_OUT_CHUNK], [0, o0]),
                        target_type=pl.FP32,
                    )
                    resid1_tile = pl.assemble(resid1_tile, pl.add(o_acc, resid), [0, o0])
            return resid1_tile

    return Scope3Incore0


def golden(tensors: dict, params: dict | None = None) -> None:
    import torch

    attn_out = tensors["attn_out"]  # [4, 5120] FP32
    hidden_states = tensors["hidden_states"]  # [4, 5120] BF16
    wo = tensors["wo"]  # [5120, 5120] BF16

    o_proj = torch.matmul(attn_out.float(), wo.float())
    tensors["resid1_tile"][:] = o_proj + hidden_states.float()


def build_tensor_specs():
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("attn_out", [BATCH_TILE, HIDDEN], torch.bfloat16, init_value=torch.randn),
        TensorSpec("hidden_states", [BATCH_TILE, HIDDEN], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wo", [HIDDEN, HIDDEN], torch.bfloat16, init_value=torch.randn),
        TensorSpec("resid1_tile", [BATCH_TILE, HIDDEN], torch.float32, is_output=True),
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
