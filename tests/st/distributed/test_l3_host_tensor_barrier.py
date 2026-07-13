# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: host-orchestrator ``pld.tensor.barrier`` builtin dispatch."""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
NR = pl.dynamic("NR")


def _expected_peer_swap(inputs: torch.Tensor) -> torch.Tensor:
    return torch.stack([inputs[1], inputs[0]])


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    rows = [
        torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


@pl.program
class HostTensorBarrier:
    @pl.function(type=pl.FunctionType.InCore)
    def publish_step(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        local = pl.load(inp, [0, 0], [1, SIZE])
        return pl.store(local, [0, 0], data)

    @pl.function(type=pl.FunctionType.Orchestration)
    def publish_orch(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        sig: pld.DistributedTensor[[NR], pl.INT32],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        return self.publish_step(inp, data)

    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        data: pld.DistributedTensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        peer: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
        return pl.store(recv, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def consume_orch(
        self,
        data: pld.DistributedTensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        peer: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        return self.consume_step(data, out, peer)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, 1, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())
        signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            self.publish_orch(inputs[r], data, signal, device=r)

        signal = pld.tensor.barrier(signal)

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            peer = (r + 1) % pld.world_size()
            self.consume_orch(data, outputs[r], peer, device=r)

        return outputs


class TestL3HostTensorBarrier:
    @pytest.mark.parametrize("n_ranks", [2])
    def test_host_tensor_barrier(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"host barrier P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            HostTensorBarrier,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.barrier__fp32"
        assert variant_dir.is_dir()
        assert (variant_dir / "kernel_config.py").is_file()

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_peer_swap(inputs)
        assert torch.allclose(outputs, expected), (
            f"host barrier P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
