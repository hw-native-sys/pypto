# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: host-orchestrator ``pld.tensor.allgather`` builtin dispatch.

Validates the HOST-level allgather collective lowers through
``LowerHostTensorCollectives`` and produces correct rank-ordered gathered data.

The HOST lowering path detects ``pld.tensor.allgather`` in ``host_orch`` and
lowers it to ``builtin.tensor.allgather`` per chip.  The exchange uses a
push-based TPUT pattern with TWO DISTINCT windows (same constraint as
``all_to_all``):

  1. **Publish** (``publish_step``): each rank stores its single chunk at
     ``stage_buf[0, :]`` — a per-rank ``[1, SIZE]`` window used ONLY as a
     TPUT source.
  2. **Allgather** (``builtin.tensor.allgather``): kernel pushes
     ``stage_buf[0, :]`` to every peer's ``data_buf[my_rank, :]``
     via in-kernel TPUT and synchronises visibility.
  3. **Consume** (``consume_step``): each rank reads its own ``data_buf``
     window via ``pl.load`` (peers already placed their chunks there).

``stage_buf`` (``[1, SIZE]``) and ``data_buf`` (``[NR, SIZE]``) must be
separate windows — reusing one buffer for both roles is a genuine
cross-process data race.

ST coverage: P=2 and P=4 (skips when fewer devices are available).
"""

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
        stage: pl.Out[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        my_rank: pl.Scalar[pl.INT32],
        nranks: pl.Scalar[pl.INT32],
    ):
        # Stage local chunk at row 0 of this rank's [1, SIZE] window; kernel
        # pushes stage[0,:] to every peer's target[my_rank,:].
        chunk = pl.load(inp, [0, 0], [1, SIZE])
        stage = pl.store(chunk, [0, 0], stage)

    @pl.function(type=pl.FunctionType.Orchestration)
    def publish_orch(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        stage: pl.Out[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        my_rank: pl.Scalar[pl.INT32],
        nranks: pl.Scalar[pl.INT32],
    ):
        self.publish_step(inp, stage, my_rank, nranks)

    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        data: pld.DistributedTensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, NR, SIZE], pl.FP32]],
        nranks: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, NR, SIZE], pl.FP32]:
        for j in pl.range(nranks):
            # Local read — data was already pushed into our window by peers
            # via in-kernel TPUT.  No TGET needed.
            row = pl.load(data, [j, 0], [1, SIZE])
            out = pl.store(row, [0, j, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def consume_orch(
        self,
        data: pld.DistributedTensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, NR, SIZE], pl.FP32]],
        nranks: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, NR, SIZE], pl.FP32]:
        return self.consume_step(data, out, nranks)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, 1, NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, 1, NR, SIZE], pl.FP32]:
        # stage is a per-rank [1, SIZE] staging window (this rank's chunk only);
        # data is the [NR, SIZE] result window peers push into.
        stage_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        data_buf = pld.alloc_window_buffer(pld.world_size() * SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())

        for r in pl.range(pld.world_size()):
            stage = pld.window(stage_buf, [1, SIZE], dtype=pl.FP32)
            self.publish_orch(inputs[r], stage, r, pld.world_size(), device=r)

        stage = pld.window(stage_buf, [1, SIZE], dtype=pl.FP32)
        data = pld.window(data_buf, [pld.world_size(), SIZE], dtype=pl.FP32)
        # 1-D signal matches the NPU-passing host all_to_all ST.
        signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)
        data = pld.tensor.allgather(stage, data, signal)

        for r in pl.range(pld.world_size()):
            self.consume_orch(data, outputs[r], pld.world_size(), device=r)

        return outputs


class TestL3HostTensorAllGather:
    @pytest.mark.parametrize("n_ranks", [2, 4])
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

        # pld.tensor.allgather on HOST lowers to builtin.tensor.allgather
        # per chip (concurrent cross-chip TPUT + barrier).
        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.allgather__fp32"
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
    sys.exit(pytest.main([__file__, "-v", "-s"] + sys.argv[1:]))
