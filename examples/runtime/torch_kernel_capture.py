# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Torch kernel calls and warmed graph capture through JIT or torch.ops.

Requires the kernel-mode integration branch, its pinned Simpler build, and the
optional native torch_npu adapter. Verified target: A2/A3 TRB, Torch 2.6,
torch_npu 2.6.0.post2, CANN 9.

Run from the repository root:
    python examples/runtime/torch_kernel_capture.py --device 0
    python examples/runtime/torch_kernel_capture.py --device 0 --entry jit

Call pypto.torch.init once per process before the first kernel call. Both
operators borrow caller-owned tensors and share PyPTO's process Worker.
Both entry points use the same warmup and graph-lifetime contract.
"""

import argparse
import importlib

import pypto.language as pl
import torch
from pypto.torch import init, register


@pl.jit
def accumulate(
    x: pl.Tensor[[16, 16], pl.FP32], step: pl.Scalar[pl.FP32], acc: pl.InOut[pl.Tensor[[16, 16], pl.FP32]]
):
    """Update acc in place: acc += x * step."""
    with pl.at(level=pl.Level.CORE_GROUP):
        value = pl.add(pl.load(acc, [0, 0], [16, 16]), pl.mul(pl.load(x, [0, 0], [16, 16]), step))
        pl.store(value, [0, 0], acc)
    return acc


@pl.jit
def add_bias(x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
    """Write out = x + 4 into caller-provided storage."""
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(x, [0, 0], [16, 16]), 4), [0, 0], out)
    return out


def main(device: int, entry: str) -> None:
    """Run eager and captured calls and check their numerical results."""
    torch_npu = importlib.import_module("torch_npu")
    torch_npu.npu.set_device(device)
    # Execution information is process state: never passed per call, captured or registered.
    init()
    x = torch.full((16, 16), 2.0, dtype=torch.float32, device=f"npu:{device}")
    acc = torch.zeros_like(x)
    out = torch.empty_like(x)

    # First calls implicitly compile and register each operator. They also warm
    # up every specialization that will appear in the captured graph.
    accumulate(x, 3.0, acc)
    add_bias(acc, out)
    torch.testing.assert_close(out.cpu(), torch.full((16, 16), 10.0))

    # Optional eager torch.ops entry: same JIT operator and process Worker.
    register(accumulate, "pypto_kernel_example::accumulate")
    register(add_bias, "pypto_kernel_example::add_bias")
    torch.ops.pypto_kernel_example.accumulate(x, 2.0, acc)
    torch.testing.assert_close(acc.cpu(), torch.full((16, 16), 10.0))

    update = torch.ops.pypto_kernel_example.accumulate if entry == "torch_ops" else accumulate
    write_output = torch.ops.pypto_kernel_example.add_bias if entry == "torch_ops" else add_bias

    torch_npu.npu.synchronize()
    # Warmup executed the computation: restore InOut state before capture.
    acc.zero_()
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        update(x, 3.0, acc)
        write_output(acc, out)

    expected = 0.0
    for value in (2.0, 5.0, 1.0):
        # Keep the captured storage addresses; update their contents in place.
        # The captured Scalar remains 3.0 for every replay.
        x.fill_(value)
        graph.replay()
        expected += value * 3.0
        torch.testing.assert_close(acc.cpu(), torch.full((16, 16), expected))
        torch.testing.assert_close(out.cpu(), torch.full((16, 16), expected + 4.0))
        print(f"input={value}, accumulator={expected}, output={expected + 4.0}")

    # Beyond init(), no explicit compile(), Worker handle, or close() is required.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Visible NPU device index")
    parser.add_argument("--entry", choices=("jit", "torch_ops"), default="torch_ops")
    args = parser.parse_args()
    main(args.device, args.entry)
