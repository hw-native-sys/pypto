# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: host-orchestrator ``pld.tensor.reduce_scatter`` builtin dispatch."""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
NR = pl.dynamic("world_size")


def _expected_reduce_scatter(inputs: torch.Tensor, n_ranks: int) -> torch.Tensor:
    chunks = [sum(inputs[r, j] for r in range(n_ranks)) for j in range(n_ranks)]
    return torch.stack(chunks).reshape(n_ranks, 1, SIZE)


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    rows = [
        torch.arange(r * 100.0, r * 100.0 + n_ranks * SIZE, dtype=torch.float32).reshape(n_ranks, SIZE)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


@pl.program
class HostTensorReduceScatter:
    @pl.function(type=pl.FunctionType.InCore)
    def publish_step(
        self,
        inp: pl.Tensor[[NR, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[NR, SIZE], pl.FP32]:
        for j in pl.range(NR):
            chunk = pl.load(inp, [j, 0], [1, SIZE])
            data = pl.store(chunk, [j, 0], data)
        return data

    @pl.function(type=pl.FunctionType.Orchestration)
    def publish_orch(
        self,
        inp: pl.Tensor[[NR, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[NR, SIZE], pl.FP32]:
        return self.publish_step(inp, data)

    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        data: pld.DistributedTensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        acc = pl.load(data, [my_rank, 0], [1, SIZE])
        return pl.store(acc, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def consume_orch(
        self,
        data: pld.DistributedTensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        return self.consume_step(data, out, my_rank)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, NR, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, 1, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
            self.publish_orch(inputs[r], data, device=r)

        data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)
        data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)

        for r in pl.range(pld.world_size()):
            self.consume_orch(data, outputs[r], r, device=r)

        return outputs


class TestL3HostTensorReduceScatter:
    @pytest.mark.parametrize("n_ranks", [2])
    def test_host_tensor_reduce_scatter(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"host reduce_scatter P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            HostTensorReduceScatter,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.reduce_scatter__sum__fp32"
        assert variant_dir.is_dir()
        assert (variant_dir / "kernel_config.py").is_file()

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_reduce_scatter(inputs, n_ranks)
        assert torch.allclose(outputs, expected), (
            f"host reduce_scatter P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
