# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hello World — the simplest PyPTO-Lib example.

Demonstrates the simplest use of auto_incore with a single parallel loop:
a large matrix is split into row chunks, and each chunk adds 1 elementwise.

    output[r, c] = input[r, c] + 1     for all (r, c)

The parallel loop with chunk= lets the compiler split the iteration space
into (chunk_loop, in_chunk_loop) and place the incore boundary automatically.
"""
from __future__ import annotations

import pypto.language as pl

ROWS = 1024
COLS = 512
ROW_CHUNK = 128


def build_hello_world_program(
    rows: int = ROWS,
    cols: int = COLS,
    row_chunk: int = ROW_CHUNK,
):
    @pl.program
    class HelloWorldProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def add_one(
            self,
            x: pl.Tensor[[rows, cols], pl.FP32],
            y: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            with pl.auto_incore():
                for r in pl.parallel(0, rows, 1, chunk=row_chunk):
                    tile_x = pl.slice(x, [1, cols], [r, 0])
                    tile_y = pl.add(tile_x, 1.0)
                    y = pl.assemble(y, tile_y, [r, 0])

            return y

    return HelloWorldProgram


def build_tensor_specs(
    rows: int = ROWS,
    cols: int = COLS,
):
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("x", [rows, cols], torch.float32, init_value=torch.randn),
        TensorSpec("y", [rows, cols], torch.float32, is_output=True),
    ]


def golden_hello_world(tensors, params):
    tensors["y"][:] = tensors["x"] + 1.0


def compile_and_run(
    rows: int = ROWS,
    cols: int = COLS,
    row_chunk: int = ROW_CHUNK,
    platform: str = "a2a3",
    device_id: int = 11,
    work_dir: str | None = None,
    dump_passes: bool = True,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    program = build_hello_world_program(
        rows=rows,
        cols=cols,
        row_chunk=row_chunk,
    )
    tensor_specs = build_tensor_specs(
        rows=rows,
        cols=cols,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_hello_world,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-5,
            atol=1e-5,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=BackendType.Ascend910B_PTO,
        ),
    )
    if not result.passed and result.error and "code_runner" in result.error:
        print("Result: COMPILE OK — device run skipped (code_runner not found).\n")
        print(result.error)
    elif not result.passed and result.error:
        print(f"Result: {result.error}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    result = compile_and_run(
        platform="a2a3sim" if args.sim else "a2a3",
        device_id=args.device,
    )
    if not result.passed:
        raise SystemExit(1)
