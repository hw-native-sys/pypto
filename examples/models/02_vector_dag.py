# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Vector DAG computation with 3 InCore kernels and 1 JIT orchestration entry.

Implements: f = (a + b + 1)(a + b + 2) + (a + b)

Task graph:
  t0: c = kernel_add(a, b)
  t1: d = kernel_add_scalar(c, 1.0)
  t2: e = kernel_add_scalar(c, 2.0)
  t3: g = kernel_mul(d, e)
  t4: f = kernel_add(g, c)

Dependencies: t0->t1, t0->t2, t1->t3, t2->t3, t3->t4, t0->t4

Concepts introduced:
  - Multi-kernel orchestration with task dependencies
  - pl.Scalar parameter type
  - Intermediate tensors allocated via pl.create_tensor in the orchestration entry
  - golden() reference for runtime verification

Run:  python examples/models/02_vector_dag.py  (requires hardware)
Next: examples/models/03_flash_attention.py
"""

import argparse

import pypto.language as pl
import torch
from pypto.runtime import RunConfig

# ── Vector DAG (128x128) kernels ─────────────────────────────────────────────


@pl.jit.incore
def kernel_add_128(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
    """Adds two tensors element-wise: result = a + b"""
    a_tile = pl.load(a, [0, 0], [128, 128])
    b_tile = pl.load(b, [0, 0], [128, 128])
    result = pl.add(a_tile, b_tile)
    pl.store(result, [0, 0], output)
    return output


@pl.jit.incore
def kernel_add_scalar_128(
    a: pl.Tensor,
    scalar: pl.Scalar[pl.FP32],
    output: pl.Out[pl.Tensor],
):
    """Adds a scalar to each element: result = a + scalar"""
    x = pl.load(a, [0, 0], [128, 128])
    result = pl.add(x, scalar)
    pl.store(result, [0, 0], output)
    return output


@pl.jit.incore
def kernel_mul_128(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
    """Multiplies two tensors element-wise: result = a * b"""
    a_tile = pl.load(a, [0, 0], [128, 128])
    b_tile = pl.load(b, [0, 0], [128, 128])
    result = pl.mul(a_tile, b_tile)
    pl.store(result, [0, 0], output)
    return output


@pl.jit
def vector_dag(a: pl.Tensor, b: pl.Tensor, f: pl.Out[pl.Tensor]):
    """Orchestration for formula: f = (a + b + 1)(a + b + 2) + (a + b)

    Task graph:
      t0: c = kernel_add(a, b)
      t1: d = kernel_add_scalar(c, 1.0)
      t2: e = kernel_add_scalar(c, 2.0)
      t3: g = kernel_mul(d, e)
      t4: f = kernel_add(g, c)
    """
    c = pl.create_tensor([128, 128], dtype=pl.FP32)
    c = kernel_add_128(a, b, c)
    d = pl.create_tensor([128, 128], dtype=pl.FP32)
    d = kernel_add_scalar_128(c, 1.0, d)
    e = pl.create_tensor([128, 128], dtype=pl.FP32)
    e = kernel_add_scalar_128(c, 2.0, e)
    g = pl.create_tensor([128, 128], dtype=pl.FP32)
    g = kernel_mul_128(d, e, g)
    f = kernel_add_128(g, c, f)
    return f


# ── Smaller orchestration DAG (16x16) used by codegen tests ──────────────────


@pl.jit.incore
def kernel_add_16(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
    """Adds two tensors element-wise: result = a + b"""
    a_tile = pl.load(a, [0, 0], [16, 16])
    b_tile = pl.load(b, [0, 0], [16, 16])
    result = pl.add(a_tile, b_tile)
    pl.store(result, [0, 0], output)
    return output


@pl.jit.incore
def kernel_add_scalar_16(
    a: pl.Tensor,
    scalar: pl.Scalar[pl.FP32],
    output: pl.Out[pl.Tensor],
):
    """Adds a scalar to each element: result = a + scalar"""
    x = pl.load(a, [0, 0], [16, 16])
    result = pl.add(x, scalar)
    pl.store(result, [0, 0], output)
    return output


@pl.jit.incore
def kernel_mul_16(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
    """Multiplies two tensors element-wise: result = a * b"""
    a_tile = pl.load(a, [0, 0], [16, 16])
    b_tile = pl.load(b, [0, 0], [16, 16])
    result = pl.mul(a_tile, b_tile)
    pl.store(result, [0, 0], output)
    return output


@pl.jit
def example_orch(a: pl.Tensor, b: pl.Tensor, f_result: pl.Out[pl.Tensor]):
    """Simpler orchestration DAG (16x16): f = (a + b + 1)(a + b + 2)

    Used by codegen tests. 4 tasks, 3 InCore kernels.
    """
    c = pl.create_tensor([16, 16], dtype=pl.FP32)
    c = kernel_add_16(a, b, c)
    d = pl.create_tensor([16, 16], dtype=pl.FP32)
    d = kernel_add_scalar_16(c, 1.0, d)
    e = pl.create_tensor([16, 16], dtype=pl.FP32)
    e = kernel_add_scalar_16(c, 2.0, e)
    f_result = kernel_mul_16(d, e, f_result)
    return f_result


def golden(tensors: dict, params: dict | None = None) -> None:
    """Reference computation: f = (a + b + 1)(a + b + 2) + (a + b)."""
    a = tensors["a"].float()
    b = tensors["b"].float()
    c = a + b
    tensors["f"][:] = (c + 1.0) * (c + 2.0) + c


def main():
    parser = argparse.ArgumentParser(description="Vector DAG example")
    parser.add_argument(
        "--runtime-profiling",
        action="store_true",
        default=False,
        help="Enable on-device runtime profiling and generate swimlane JSON",
    )
    args = parser.parse_args()

    a = torch.full((128, 128), 2.0, dtype=torch.float32)
    b = torch.full((128, 128), 3.0, dtype=torch.float32)
    f = torch.zeros((128, 128), dtype=torch.float32)

    vector_dag(
        a,
        b,
        f,
        config=RunConfig(runtime_profiling=args.runtime_profiling),
    )

    # Golden validation
    tensors = {"a": a, "b": b, "f": f.clone()}
    golden(tensors)
    expected_f = tensors["f"]
    assert torch.allclose(f, expected_f, rtol=1e-5, atol=1e-5), (
        f"Validation failed: max diff = {(f - expected_f).abs().max().item()}"
    )
    print("OK")


if __name__ == "__main__":
    main()
