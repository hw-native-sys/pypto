# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Qwen3 Scope 3 — incore_5: Final residual addition.

Isolated test for the last incore call in scope3 orchestration:
  out = cast(down_proj_tile + resid1_tile, BF16)

Parallel over HIDDEN_BLOCKS with chunk=4.
"""

import pypto.language as pl

HIDDEN = 5120
K_CHUNK = 128
BATCH_TILE = 16

HIDDEN_BLOCKS = HIDDEN // K_CHUNK


def build_program():
    @pl.program
    class Scope3Incore5:
        @pl.function(type=pl.FunctionType.Opaque)
        def scope3_incore_5(
            self,
            down_proj_tile: pl.Tensor[[BATCH_TILE, HIDDEN], pl.FP32],
            resid1_tile: pl.Tensor[[BATCH_TILE, HIDDEN], pl.FP32],
            out: pl.Tensor[[BATCH_TILE, HIDDEN], pl.BF16],
        ) -> pl.Tensor[[BATCH_TILE, HIDDEN], pl.BF16]:
            with pl.auto_incore():
                for ob in pl.parallel(0, HIDDEN_BLOCKS, 1, chunk=4):
                    o0 = ob * K_CHUNK
                    down_acc = pl.add(
                        pl.slice(down_proj_tile, [BATCH_TILE, K_CHUNK], [0, o0]),
                        pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, o0]),
                    )
                    out = pl.assemble(out, pl.cast(down_acc, target_type=pl.BF16), [0, o0])
            return out

    return Scope3Incore5


def golden(tensors: dict, params: dict | None = None) -> None:
    down_proj = tensors["down_proj_tile"]  # [4, 5120] FP32
    resid1 = tensors["resid1_tile"]  # [4, 5120] FP32

    tensors["out"][:] = (down_proj + resid1).bfloat16()


def build_tensor_specs():
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("down_proj_tile", [BATCH_TILE, HIDDEN], torch.float32, init_value=torch.randn),
        TensorSpec("resid1_tile", [BATCH_TILE, HIDDEN], torch.float32, init_value=torch.randn),
        TensorSpec("out", [BATCH_TILE, HIDDEN], torch.bfloat16, is_output=True),
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
