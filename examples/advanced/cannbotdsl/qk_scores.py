# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compute BF16 query/key attention scores with the public Mega-MLA Cube kernel.

Requires CANNBotDSL 0.8.4, PyPTO, and the public CANNBotDSL operator sources:
    git clone https://github.com/hicann/cannbot-dsl.git
    git -C cannbot-dsl checkout 1b82f62c81a51998e49d1958ad8582755d6a385d
    export PYTHONPATH="$PWD/cannbot-dsl/samples/attention/mega_mla:${PYTHONPATH}"
    python examples/advanced/cannbotdsl/qk_scores.py --compile-only

The compile-only path builds the device kernel and orchestration without a device.
Omit --compile-only on A5 to compare the FP32 score matrix against PyTorch.
"""

import argparse

import cannbotdsl as cb
import pypto.language as pl
import torch
from attention_matmul import QueryKeyMatmulKernel
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig

qk = QueryKeyMatmulKernel(m=16, n=64, k=128, base_n=64, base_k=128)


@cb.kernel
def qk_one_block(q: cb.Tensor, k: cb.Tensor, scores: cb.Tensor):
    # The upstream standalone entry reads the launch grid. A PyPTO external
    # task has one logical block, so pass its coordinates to the same tile body.
    qk.compute_flat_tiles(scores, q, k, 0, 1, 1)


@pl.jit
def attention_scores(
    q: pl.Tensor[[16, 128], pl.BF16],
    k: pl.Tensor[[64, 128], pl.BF16],
    scores: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
):
    qk_one_block(q, k, scores)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=args.device_id
    )
    program = attention_scores.compile(config=config)
    if args.compile_only:
        program.load()
        print(f"Built QK scores program from {program.output_dir}")
        return
    # Small integers make the BF16 inputs and FP32 accumulation exact.
    rng = torch.Generator().manual_seed(11)
    q = torch.randint(-3, 4, (16, 128), generator=rng).bfloat16()
    k = torch.randint(-3, 4, (64, 128), generator=rng).bfloat16()
    scores = torch.full((16, 64), float("nan"))
    program(q, k, scores, config=config)
    torch.testing.assert_close(scores, q.float() @ k.float().T, rtol=0, atol=0)
    print("QK scores match the FP32 reference")


if __name__ == "__main__":
    main()
