# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: host-orchestrator ``pld.tensor.all_to_all`` builtin dispatch.

Validates the HOST-level all-to-all collective lowers through
``LowerHostTensorCollectives`` and produces correct rank-ordered personalized
exchange.

The HOST lowering path detects ``pld.tensor.all_to_all`` in ``host_orch`` and
lowers it to ``builtin.tensor.all_to_all`` per chip.  The exchange uses a
push-based TPUT pattern with TWO DISTINCT windows:

  1. **Stage** (``stage_step``): each rank writes its per-destination chunks
     into ``stage_buf`` — a window used ONLY as a TPUT source, never as an
     incoming-push destination.
  2. **All-to-all** (``builtin.tensor.all_to_all``): kernel pushes
     ``stage_buf[dest, :]`` to each peer's ``data_buf`` window via in-kernel
     TPUT and synchronises visibility.
  3. **Consume** (``consume_step``): each rank reads its own ``data_buf``
     window via ``pl.load`` (peers already placed their chunks there via
     in-kernel TPUT).

``stage_buf`` and ``data_buf`` must be separate windows — reusing one buffer
for both roles is a genuine cross-process data race (see the builtin kernel
template's kernel.cpp.in for the full explanation).

TPUT auto-chunks rows larger than the staging tile via the pto-isa DMA engine
and works across chip-process boundaries in both simulation and real hardware.

ST coverage: P=2 (default CI / 2-device hosts).
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


def _expected_all_to_all(inputs: torch.Tensor) -> torch.Tensor:
    """Golden: output[rank, src, j] = src * 1000 + rank * 100 + j."""
    nranks = inputs.shape[0]
    rank_idx = torch.arange(nranks, dtype=torch.float32).view(-1, 1, 1)
    src_idx = torch.arange(nranks, dtype=torch.float32).view(1, -1, 1)
    j = torch.arange(SIZE, dtype=torch.float32).view(1, 1, -1)
    return src_idx * 1000 + rank_idx * 100 + j


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Each rank r fills input[r, d, j] = r * 1000 + d * 100 + j."""
    r = torch.arange(n_ranks, dtype=torch.float32).view(-1, 1, 1)
    d = torch.arange(n_ranks, dtype=torch.float32).view(1, -1, 1)
    j = torch.arange(SIZE, dtype=torch.float32).view(1, 1, -1)
    return r * 1000 + d * 100 + j


@pl.program
class HostTensorAllToAll:
    @pl.function(type=pl.FunctionType.InCore)
    def stage_step(
        self,
        inp: pl.Tensor[[NR, SIZE], pl.FP32],
        stage: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
        my_rank: pl.Scalar[pl.INT32],
    ):
        for dest in pl.range(NR):
            chunk = pl.load(inp, [dest, 0], [1, SIZE])
            stage = pl.store(chunk, [dest, 0], stage)

    @pl.function(type=pl.FunctionType.Orchestration)
    def stage_orch(
        self,
        inp: pl.Tensor[[NR, SIZE], pl.FP32],
        stage: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
        my_rank: pl.Scalar[pl.INT32],
    ):
        self.stage_step(inp, stage, my_rank)

    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        data: pld.DistributedTensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, SIZE], pl.FP32]:
        for src in pl.range(NR):
            # Local read — data was already pushed into our window by peers
            # via in-kernel TPUT in the builtin kernel.  No TGET needed.
            row = pl.load(data, [src, 0], [1, SIZE])
            out = pl.store(row, [src, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def consume_orch(
        self,
        data: pld.DistributedTensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, SIZE], pl.FP32]:
        return self.consume_step(data, out)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, NR, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, NR, SIZE], pl.FP32]:
        stage_buf = pld.alloc_window_buffer(pld.world_size() * SIZE * 4)
        data_buf = pld.alloc_window_buffer(pld.world_size() * SIZE * 4)
        signal_buf = pld.alloc_window_buffer(pld.world_size() * 4)

        # Stage: each rank writes its per-destination chunks into stage_buf —
        # a window used ONLY as a TPUT source, never an incoming-push
        # destination (see module docstring for why this must be separate
        # from data_buf).
        for r in pl.range(pld.world_size()):
            stage = pld.window(stage_buf, [pld.world_size(), SIZE], dtype=pl.FP32)
            self.stage_orch(inputs[r], stage, r, device=r)

        # Builtin collective: pushes stage_buf[dest,:] into each peer's
        # data_buf window, then synchronises visibility.
        stage = pld.window(stage_buf, [pld.world_size(), SIZE], dtype=pl.FP32)
        data = pld.window(data_buf, [pld.world_size(), SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)
        data = pld.tensor.all_to_all(stage, data, signal)

        # Consume: read back the result from the window (peers already placed
        # their chunks in our window via TPUT).
        for r in pl.range(pld.world_size()):
            self.consume_orch(data, outputs[r], device=r)

        return outputs


class TestL3HostTensorAllToAll:
    """L3 distributed runtime: HOST-level all-to-all via builtin dispatch.

    Validates that ``pld.tensor.all_to_all`` in a HOST orchestrator lowers
    through ``LowerHostTensorCollectives`` to ``builtin.tensor.all_to_all`` per
    chip, with in-kernel TPUT (TSTORE via CommRemotePtr) pushing from the
    separate ``stage_buf`` staging window into the ``data_buf`` result window.
    """

    @pytest.mark.parametrize("n_ranks", [2])
    def test_host_tensor_all_to_all(self, test_config, device_ids, n_ranks):
        """Compile and run host-level all-to-all for P=2."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"host all-to-all P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            HostTensorAllToAll,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        # pld.tensor.all_to_all on HOST lowers to builtin.tensor.all_to_all
        # per chip.  The kernel pushes stage_buf chunks to peers via in-kernel
        # TPUT (TSTORE to CommRemotePtr) and synchronises visibility.
        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.all_to_all__fp32"
        assert variant_dir.is_dir(), f"expected {variant_dir}"
        assert (variant_dir / "kernel_config.py").is_file()

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_all_to_all(inputs)
        assert torch.allclose(outputs, expected), (
            f"host all-to-all P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
