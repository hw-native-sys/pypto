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
exchange via the hand-written ``builtin.tensor.all_to_all`` kernel.

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


def _expected_all_to_all(inputs: torch.Tensor) -> torch.Tensor:
    """Golden: output[rank, src, j] = inputs[src, rank, j] (rank src's chunk
    destined for rank ``rank`` lands in rank's slot ``src``). Input-dependent so
    distinct per-round data propagates to the expected output."""
    return inputs.permute(1, 0, 2)


def _make_rank_inputs(n_ranks: int, round_offset: float = 0.0) -> torch.Tensor:
    """Each rank r fills input[r, d, j] = r * 1000 + d * 100 + j (+ round_offset)."""
    r = torch.arange(n_ranks, dtype=torch.float32).view(-1, 1, 1)
    d = torch.arange(n_ranks, dtype=torch.float32).view(1, -1, 1)
    j = torch.arange(SIZE, dtype=torch.float32).view(1, 1, -1)
    return round_offset + r * 1000 + d * 100 + j


@pl.program
class HostTensorAllToAll:
    @pl.function(type=pl.FunctionType.InCore)
    def stage_step(
        self,
        inp: pl.Tensor[[NR, SIZE], pl.FP32],
        stage: pl.Out[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
        my_rank: pl.Scalar[pl.INT32],
    ):
        for dest in pl.range(NR):
            chunk = pl.load(inp, [dest, 0], [1, SIZE])
            stage = pl.store(chunk, [dest, 0], stage)

    @pl.function(type=pl.FunctionType.Orchestration)
    def stage_orch(
        self,
        inp: pl.Tensor[[NR, SIZE], pl.FP32],
        stage: pl.Out[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
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
        stage_buf = pld.alloc_window_buffer(pld.world_size() * SIZE * pl.FP32.get_byte())
        data_buf = pld.alloc_window_buffer(pld.world_size() * SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())

        for r in pl.range(pld.world_size()):
            stage = pld.window(stage_buf, [pld.world_size(), SIZE], dtype=pl.FP32)
            self.stage_orch(inputs[r], stage, r, device=r)

        stage = pld.window(stage_buf, [pld.world_size(), SIZE], dtype=pl.FP32)
        data = pld.window(data_buf, [pld.world_size(), SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)
        data = pld.tensor.all_to_all(stage, data, signal)

        for r in pl.range(pld.world_size()):
            self.consume_orch(data, outputs[r], device=r)

        return outputs


def _build_host_all_to_all_signal_reuse_program():
    """Host all_to_all reusing ONE signal buffer across 2 back-to-back calls.

    The self-clearing epilogue restores the AtomicAdd(+1) cells to 0 after each
    call; without it the second call's Ge(1) wait passes on the stale
    satisfied cell.
    """
    ROUNDS = 2

    @pl.program
    class HostTensorAllToAllSignalReuse:
        @pl.function(type=pl.FunctionType.InCore)
        def stage_step(
            self,
            inp: pl.Tensor[[NR, SIZE], pl.FP32],
            stage: pl.Out[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
        ):
            for dest in pl.range(NR):
                chunk = pl.load(inp, [dest, 0], [1, SIZE])
                stage = pl.store(chunk, [dest, 0], stage)

        @pl.function(type=pl.FunctionType.Orchestration)
        def stage_orch(
            self,
            inp: pl.Tensor[[NR, SIZE], pl.FP32],
            stage: pl.Out[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
        ):
            self.stage_step(inp, stage)

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            data: pld.DistributedTensor[[NR, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[NR, SIZE], pl.FP32]],
        ) -> pl.Tensor[[NR, SIZE], pl.FP32]:
            for src in pl.range(NR):
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
            inputs: pl.Tensor[[ROUNDS, NR, NR, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[ROUNDS, NR, NR, SIZE], pl.FP32]],
        ) -> pl.Tensor[[ROUNDS, NR, NR, SIZE], pl.FP32]:
            stage_buf_1 = pld.alloc_window_buffer(pld.world_size() * SIZE * pl.FP32.get_byte())
            stage_buf_2 = pld.alloc_window_buffer(pld.world_size() * SIZE * pl.FP32.get_byte())
            data_buf_1 = pld.alloc_window_buffer(pld.world_size() * SIZE * pl.FP32.get_byte())
            data_buf_2 = pld.alloc_window_buffer(pld.world_size() * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())
            signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)

            # Round 1 — distinct stage/data windows per round, ONE shared signal.
            for r in pl.range(pld.world_size()):
                stage = pld.window(stage_buf_1, [pld.world_size(), SIZE], dtype=pl.FP32)
                self.stage_orch(inputs[0, r], stage, device=r)
            stage = pld.window(stage_buf_1, [pld.world_size(), SIZE], dtype=pl.FP32)
            data = pld.window(data_buf_1, [pld.world_size(), SIZE], dtype=pl.FP32)
            data = pld.tensor.all_to_all(stage, data, signal)
            for r in pl.range(pld.world_size()):
                self.consume_orch(data, outputs[0, r], device=r)

            # Round 2 — reuse the same signal.
            for r in pl.range(pld.world_size()):
                stage = pld.window(stage_buf_2, [pld.world_size(), SIZE], dtype=pl.FP32)
                self.stage_orch(inputs[1, r], stage, device=r)
            stage = pld.window(stage_buf_2, [pld.world_size(), SIZE], dtype=pl.FP32)
            data = pld.window(data_buf_2, [pld.world_size(), SIZE], dtype=pl.FP32)
            data = pld.tensor.all_to_all(stage, data, signal)
            for r in pl.range(pld.world_size()):
                self.consume_orch(data, outputs[1, r], device=r)
            return outputs

    return HostTensorAllToAllSignalReuse


class TestL3HostTensorAllToAll:
    """L3 distributed runtime: HOST-level all-to-all via builtin dispatch."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_host_tensor_all_to_all(self, test_config, device_ids, n_ranks):
        """Compile and run host-level all-to-all for P in {2, 4}."""
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

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_host_tensor_all_to_all_signal_reuse(self, test_config, device_ids, n_ranks):
        """Reuse ONE signal buffer across 2 back-to-back all_to_all calls."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"host all_to_all P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        rounds = 2
        compiled = ir.compile(
            _build_host_all_to_all_signal_reuse_program(),
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )
        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.all_to_all__fp32"
        assert variant_dir.is_dir()

        # Each round carries a distinct offset so a stale earlier-round result
        # (a missed epilogue reset) cannot match the round's golden.
        inputs = torch.stack([_make_rank_inputs(n_ranks, round_offset=rd * 10000.0) for rd in range(rounds)])
        outputs = torch.zeros((rounds, n_ranks, n_ranks, SIZE), dtype=torch.float32)
        compiled(inputs, outputs)

        for rd in range(rounds):
            expected = _expected_all_to_all(inputs[rd])
            assert torch.allclose(outputs[rd], expected), (
                f"host all_to_all signal-reuse round {rd} P={n_ranks} mismatch: "
                f"max diff = {(outputs[rd] - expected).abs().max().item()}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
