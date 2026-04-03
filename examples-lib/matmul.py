# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Matmul — tiled matrix multiplication with M/N blocking (no K tiling).

    C[m, n] = A[m, k] @ B[k, n]

M and N are parallelised via pl.parallel; K is consumed in a single matmul
(no reduction loop, no pl.add accumulation).

Input matrices are FP16; the output is FP32 (mixed-precision).
"""
from __future__ import annotations

import pypto.language as pl

# ---------------------------------------------------------------------------
# Matmul parameters — edit these to change problem size and tiling
# ---------------------------------------------------------------------------
M = 256         # total rows of A / C
N = 256         # total cols of B / C
K = 256         # total cols of A / rows of B
M_TILE = 64     # tile size along M dimension
N_TILE = 64     # tile size along N dimension
M_CHUNK = 2     # M-tiles grouped per incore chunk
N_CHUNK = 2     # N-tiles grouped per incore chunk


def build_matmul_program(
    m: int = M,
    n: int = N,
    k: int = K,
    m_tile: int = M_TILE,
    n_tile: int = N_TILE,
    m_chunk: int = M_CHUNK,
    n_chunk: int = N_CHUNK,
):
    @pl.program
    class MatmulProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def matmul(
            self,
            a: pl.Tensor[[m, k], pl.FP16],
            b: pl.Tensor[[k, n], pl.FP16],
            c: pl.Out[pl.Tensor[[m, n], pl.FP32]],
        ) -> pl.Tensor[[m, n], pl.FP32]:
            with pl.auto_incore():
                for mb in pl.parallel(0, m, m_tile, chunk=m_chunk):
                    for nb in pl.parallel(0, n, n_tile, chunk=n_chunk):
                        tile_a = pl.slice(a, [m_tile, k], [mb, 0])
                        tile_b = pl.slice(b, [k, n_tile], [0, nb])
                        tile_c = pl.matmul(tile_a, tile_b)
                        c = pl.assemble(c, tile_c, [mb, nb])

            return c

    return MatmulProgram


def build_tensor_specs(
    m: int = M,
    n: int = N,
    k: int = K,
):
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("a", [m, k], torch.float16, init_value=torch.randn),
        TensorSpec("b", [k, n], torch.float16, init_value=torch.randn),
        TensorSpec("c", [m, n], torch.float32, is_output=True),
    ]


def golden_matmul(tensors, params):
    tensors["c"][:] = tensors["a"].float() @ tensors["b"].float()


def compile_and_run(
    m: int = M,
    n: int = N,
    k: int = K,
    m_tile: int = M_TILE,
    n_tile: int = N_TILE,
    m_chunk: int = M_CHUNK,
    n_chunk: int = N_CHUNK,
    platform: str = "a2a3",
    device_id: int = 11,
    dump_passes: bool = True,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    program = build_matmul_program(
        m=m, n=n, k=k,
        m_tile=m_tile, n_tile=n_tile,
        m_chunk=m_chunk, n_chunk=n_chunk,
    )
    tensor_specs = build_tensor_specs(m=m, n=n, k=k)

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_matmul,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-3,
            atol=1e-3,
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
