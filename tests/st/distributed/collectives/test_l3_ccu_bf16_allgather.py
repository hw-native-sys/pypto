# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Two-rank BF16 CCU allgather through generated modules and native registration."""

from contextlib import nullcontext

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig


def _program(size):
    @pl.jit.incore
    def stage_in(
        inp: pl.Tensor[[1, size], pl.BF16],
        data: pl.InOut[pld.DistributedTensor[[1, size], pl.BF16]],
    ) -> pld.DistributedTensor[[1, size], pl.BF16]:
        tile = pl.load(inp, [0, 0], [1, size])
        return pl.store(tile, [0, 0], data)

    @pl.jit.incore
    def stage_out(
        data: pld.DistributedTensor[[2, size], pl.BF16],
        out: pl.Out[pl.Tensor[[2, size], pl.BF16]],
    ) -> pl.Tensor[[2, size], pl.BF16]:
        for r in pl.range(2):
            tile = pl.load(data, [r, 0], [1, size])
            out = pl.store(tile, [r, 0], out)
        return out

    @pl.jit
    def chip(
        inp: pl.Tensor[[1, size], pl.BF16],
        out: pl.Out[pl.Tensor[[2, size], pl.BF16]],
        data: pl.InOut[pld.DistributedTensor[[1, size], pl.BF16]],
        target: pl.InOut[pld.DistributedTensor[[2, size], pl.BF16]],
        signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
    ) -> pl.Tensor[[2, size], pl.BF16]:
        data = stage_in(inp, data)
        target = pld.tensor.allgather(data, target, signal, engine="ccu")
        target = pld.tensor.allgather(data, target, signal, engine="ccu")
        return stage_out(target, out)

    @pl.jit.host
    def host(
        inputs: pl.Tensor[[2, 1, size], pl.BF16],
        outputs: pl.Out[pl.Tensor[[2, 2, size], pl.BF16]],
    ) -> pl.Tensor[[2, 2, size], pl.BF16]:
        input_buf = pld.alloc_window_buffer(size * pl.BF16.get_byte())
        target_buf = pld.alloc_window_buffer(2 * size * pl.BF16.get_byte())
        signal_buf = pld.alloc_window_buffer(2 * pl.INT32.get_byte())
        for r in pl.range(2):
            data = pld.window(input_buf, [1, size], dtype=pl.BF16)
            target = pld.window(target_buf, [2, size], dtype=pl.BF16)
            signal = pld.window(signal_buf, [2, 1], dtype=pl.INT32)
            chip(inputs[r], outputs[r], data, target, signal, device=r)
        return outputs

    return host


@pytest.mark.platforms("a5")
@pytest.mark.parametrize("size", [64, 1024])
@pytest.mark.parametrize("prepared", [False, True], ids=["one-shot", "prepared"])
def test_ccu_bf16_allgather(test_config, device_ids, size, prepared):
    if len(device_ids) < 2:
        pytest.skip("CCU allgather requires two devices")
    inputs = (torch.arange(2 * size, dtype=torch.float32).reshape(2, 1, size) / 8 - size / 8).bfloat16()
    outputs = torch.zeros((2, 2, size), dtype=torch.bfloat16)
    if prepared:
        inputs.share_memory_()
        outputs.share_memory_()
    compiled = _program(size).compile(
        inputs,
        outputs,
        config=RunConfig(
            platform="a5",
            distributed_config=DistributedConfig(device_ids=device_ids[:2], num_sub_workers=0),
        ),
    )
    config = RunConfig(platform="a5")
    context = compiled.prepare(config=config, persistent=True) if prepared else nullcontext()
    with context as worker:
        for round_id in range(2):
            inputs.add_(round_id)
            outputs.fill_(float("nan"))
            if worker is None:
                compiled(inputs, outputs, config=config)
            else:
                worker.run(compiled, inputs, outputs, config=config)
            expected = inputs[:, 0].unsqueeze(0).expand(2, -1, -1)
            assert torch.equal(outputs, expected)
