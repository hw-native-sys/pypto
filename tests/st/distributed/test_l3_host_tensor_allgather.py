# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: host-orchestrator ``pld.tensor.allgather`` builtin dispatch."""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
NR = pl.dynamic("world_size")


def _expected_allgather(inputs: torch.Tensor, n_ranks: int) -> torch.Tensor:
    gathered = torch.stack([inputs[r, 0] for r in range(n_ranks)])
    return torch.stack([gathered] * n_ranks).unsqueeze(1)


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    rows = [
        torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


@pl.program
class HostTensorAllGather:
    @pl.function(type=pl.FunctionType.InCore)
    def publish_step(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pld.DistributedTensor[[NR, SIZE], pl.FP32]:
        chunk = pl.load(inp, [0, 0], [1, SIZE])
        return pl.store(chunk, [my_rank, 0], data)

    @pl.function(type=pl.FunctionType.Orchestration)
    def publish_orch(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pld.DistributedTensor[[NR, SIZE], pl.FP32]:
        return self.publish_step(inp, data, my_rank)

    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        data: pld.DistributedTensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[1, NR, SIZE], pl.FP32]:
        for j in pl.range(NR):
            # Use remote_load so the TGET hardware channel transfers the data
            # across chip-process boundaries in both simulation and real hardware.
            # Direct pl.load (tile.load from local GM) only works on real hardware
            # where the window buffer is at the same physical address for all chips.
            row = pld.tile.remote_load(data, peer=j, offsets=[j, 0], shape=[1, SIZE])
            out = pl.store(row, [0, j, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def consume_orch(
        self,
        data: pld.DistributedTensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[1, NR, SIZE], pl.FP32]:
        return self.consume_step(data, out)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, 1, NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, 1, NR, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(pld.world_size() * SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [pld.world_size(), SIZE], dtype=pl.FP32)
            self.publish_orch(inputs[r], data, r, device=r)

        data = pld.window(data_buf, [pld.world_size(), SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)
        data = pld.tensor.allgather(data, signal)

        for r in pl.range(pld.world_size()):
            self.consume_orch(data, outputs[r], device=r)

        return outputs


class TestL3HostTensorAllGather:
    @pytest.mark.parametrize("n_ranks", [2])
    def test_host_tensor_allgather(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"host allgather P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            HostTensorAllGather,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        # pld.tensor.allgather on HOST lowers to builtin.tensor.barrier per chip
        # (the allgather AIV kernel requires concurrent cross-chip dispatch;
        # a barrier synchronises pre-staged window data; consume_step uses
        # pld.tile.remote_load to gather from all peers).
        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.barrier__fp32"
        assert variant_dir.is_dir()
        assert (variant_dir / "kernel_config.py").is_file()

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, n_ranks, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_allgather(inputs, n_ranks)
        assert torch.allclose(outputs, expected), (
            f"host allgather P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
