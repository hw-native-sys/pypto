# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 Host-source domain construction, URMA context, and local MTE smoke test.

No peer read is issued: allocation must establish the URMA resources, and each
rank writes then reads its own mapped Host window through separate tasks.
"""

from contextlib import ExitStack

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import DistributedWorker, RunConfig


@pl.jit.incore
def fill_host(
    inp: pl.Tensor[[1, 64], pl.FP32],
    table: pl.Out[pld.DistributedTensor[[1, 64], pl.FP32]],
) -> pld.DistributedTensor[[1, 64], pl.FP32]:
    return pl.store(pl.load(inp, [0, 0], [1, 64]), [0, 0], table)


@pl.jit.incore
def read_host(
    table: pld.DistributedTensor[[1, 64], pl.FP32],
    out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
) -> pl.Tensor[[1, 64], pl.FP32]:
    return pl.store(pl.load(table, [0, 0], [1, 64]), [0, 0], out)


@pl.jit
def host_table_chip(
    inp: pl.Tensor[[1, 64], pl.FP32],
    out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
    table: pl.InOut[pld.DistributedTensor[[1, 64], pl.FP32]],
) -> pl.Tensor[[1, 64], pl.FP32]:
    table = fill_host(inp, table)
    return read_host(table, out)


@pl.jit.host
def host_table_program(
    inputs: pl.Tensor[[2, 1, 64], pl.FP32],
    outputs: pl.Out[pl.Tensor[[2, 1, 64], pl.FP32]],
) -> pl.Tensor[[2, 1, 64], pl.FP32]:
    buf = pld.alloc_window_buffer([1, 64], dtype=pl.FP32, memory_source="host")
    for r in pl.range(2):
        table = pld.window(buf, [1, 64], dtype=pl.FP32)
        host_table_chip(inputs[r], outputs[r], table, device=r)
    return outputs


@pytest.mark.platforms("a5")
def test_host_domain_local_mte(test_config, device_ids):
    if len(device_ids) < 2:
        pytest.skip("Host domain smoke test requires two A5 devices")
    inputs = torch.arange(128, dtype=torch.float32).reshape(2, 1, 64)
    outputs = torch.zeros_like(inputs)
    compiled = host_table_program.compile(
        inputs,
        outputs,
        config=RunConfig(
            platform=test_config.platform,
            distributed_config=DistributedConfig(device_ids=device_ids[:2], num_sub_workers=0),
        ),
    )
    compiled(inputs, outputs, config=RunConfig(platform=test_config.platform))
    assert torch.equal(outputs, inputs)


@pl.jit
def initialized_host_chip(
    table: pld.DistributedTensor[[1, 64], pl.FP32],
    out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
) -> pl.Tensor[[1, 64], pl.FP32]:
    return read_host(table, out)


@pl.jit.host
def initialized_host_program(
    outputs: pl.Out[pl.Tensor[[2, 1, 64], pl.FP32]],
) -> pl.Tensor[[2, 1, 64], pl.FP32]:
    buf = pld.alloc_window_buffer([1, 64], dtype=pl.FP32, memory_source="host", init="engram_l1")
    for r in pl.range(2):
        table = pld.window(buf, [1, 64], dtype=pl.FP32)
        initialized_host_chip(table, outputs[r], device=r)
    return outputs


@pytest.mark.platforms("a5")
@pytest.mark.parametrize("source_kind", ["file", "tensor"])
@pytest.mark.parametrize("reset", [None, False, True])
def test_host_domain_runtime_file_initialization(test_config, device_ids, tmp_path, source_kind, reset):
    if len(device_ids) < 2:
        pytest.skip("Host initialization smoke test requires two A5 devices")
    expected = torch.arange(128, dtype=torch.float32).reshape(2, 1, 64)
    outputs = torch.zeros_like(expected).share_memory_()
    compiled = initialized_host_program.compile(
        outputs,
        config=RunConfig(
            platform=test_config.platform,
            distributed_config=DistributedConfig(device_ids=device_ids[:2], num_sub_workers=0),
        ),
    )
    if test_config.codegen_only:
        pytest.skip("--codegen-only disables distributed runtime execution")
    # The files need not even exist until compilation has completed.
    with ExitStack() as stack:
        sources = []
        for rank in range(2):
            if source_kind == "tensor":
                sources.append(expected[rank].clone())
            else:
                path = tmp_path / f"rank{rank}.bin"
                path.write_bytes(expected[rank].numpy().tobytes())
                sources.append(stack.enter_context(path.open("rb")))
        with DistributedWorker(
            compiled, persistent=True, reset_persistent_windows=reset, window_sources={"engram_l1": sources}
        ) as worker:
            worker(outputs)
            assert torch.equal(outputs, expected)
            for source in sources:
                if isinstance(source, torch.Tensor):
                    source.zero_()
                else:
                    source.close()
            outputs.fill_(-1)
            worker(outputs)
            expected_reuse = expected if reset is False else torch.zeros_like(expected)
            assert torch.equal(outputs, expected_reuse)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
