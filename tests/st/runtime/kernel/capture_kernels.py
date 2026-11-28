# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU preparation of graph replay kernels for TMR and HBG."""

import argparse
import importlib.util
import sys
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto.pypto_core import passes

NR = pl.dynamic("NR")


@pl.jit.incore
def capture_barrier_step(signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]]):
    signal = pld.tensor.barrier(signal)
    return signal


@pl.jit
def capture_barrier_chip(signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]]):
    signal = capture_barrier_step(signal)
    return signal


def prepare(directory: Path, runtime: str) -> None:
    examples = Path(__file__).resolve().parents[4] / "examples/distributed"
    sys.path.insert(0, str(examples))
    from gloo_shmem_utils import compile_eager_kernel  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "capture_collectives", examples / "08_gloo_shmem_collectives.py"
    )
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    kind = {
        "tensormap_and_ringbuffer": passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER,
        "host_build_graph": passes.RuntimeKind.HOST_BUILD_GRAPH,
    }[runtime]
    with passes.PassContext([], runtime=kind):
        meta = torch.empty((1, example.SIZE), dtype=example.TORCH_DTYPE)
        for kernel, count in example.collective_kernels():
            compile_eager_kernel(kernel, *((meta,) * count), output_dir=str(directory / kernel.__name__))
        compile_eager_kernel(
            capture_barrier_chip,
            torch.empty((2, 1), dtype=torch.int32),
            output_dir=str(directory / "capture_barrier_chip"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--runtime",
        choices=("tensormap_and_ringbuffer", "host_build_graph"),
        default="tensormap_and_ringbuffer",
    )
    args = parser.parse_args()
    prepare(args.artifacts, args.runtime)
