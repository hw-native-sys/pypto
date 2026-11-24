# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Apply attention probabilities to values using the public Mega-MLA Cube kernel.

Requires CANNBotDSL 0.8.4, PyPTO, and the public CANNBotDSL operator sources:
    git clone https://github.com/hicann/cannbot-dsl.git
    git -C cannbot-dsl checkout 1b82f62c81a51998e49d1958ad8582755d6a385d
    export PYTHONPATH="$PWD/cannbot-dsl/samples/attention/mega_mla:${PYTHONPATH}"
    python examples/advanced/cannbotdsl/pv_context.py --compile-only

The compile-only path builds the device kernel and orchestration without a device.
Omit --compile-only on A5 to compare the FP32 context against PyTorch.
"""

import argparse

import pypto.language as pl
import torch
from attention_matmul import ProbabilityValueMatmulKernel
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig

# One external task computes one FP32 context tile.
pv = ProbabilityValueMatmulKernel(m=16, n=64, k=128, base_n=64).probability_value_matmul_kernel


@pl.jit
def attention_context(
    probabilities: pl.Tensor[[16, 128], pl.BF16],
    values: pl.Tensor[[128, 64], pl.BF16],
    context: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
):
    pv(context, probabilities, values)
    return context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=args.device_id
    )
    program = attention_context.compile(config=config)
    if args.compile_only:
        program.load()
        print(f"Built attention context program from {program.output_dir}")
        return
    # Each row is an exactly representable distribution over two distinct values.
    rows = torch.arange(16)
    probabilities = torch.zeros((16, 128), dtype=torch.bfloat16)
    probabilities[rows, rows] = 0.25
    probabilities[rows, 127 - rows] = 0.75
    values = torch.randint(-8, 9, (128, 64), generator=torch.Generator().manual_seed(13)).bfloat16()
    context = torch.full((16, 64), float("nan"))
    program(probabilities, values, context, config=config)
    torch.testing.assert_close(context, probabilities.float() @ values.float(), rtol=0, atol=0)
    print("Attention context matches the FP32 reference")


if __name__ == "__main__":
    main()
