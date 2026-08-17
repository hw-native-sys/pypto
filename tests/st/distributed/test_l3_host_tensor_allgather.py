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

import re
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


def _make_rank_inputs(n_ranks: int, round_offset: float = 0.0) -> torch.Tensor:
    """Build distinct per-rank rows; ``round_offset`` distinguishes reuse rounds."""
    rows = [
        torch.arange(r * 100.0 + round_offset, r * 100.0 + round_offset + SIZE, dtype=torch.float32).reshape(
            1, SIZE
        )
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


def _build_host_allgather_signal_reuse_program():
    """Host allgather reusing ONE signal buffer across 2 back-to-back calls.

    The self-clearing epilogue restores the AtomicAdd(+1) cells to 0 after each
    call; without it the second call's Ge(1) wait passes on the stale
    satisfied cell.
    """
    ROUNDS = 2

    @pl.program
    class HostTensorAllGatherSignalReuse:
        @pl.function(type=pl.FunctionType.InCore)
        def publish_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            stage: pl.Out[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            my_rank: pl.Scalar[pl.INT32],
            nranks: pl.Scalar[pl.INT32],
        ):
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
            inputs: pl.Tensor[[ROUNDS, NR, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[ROUNDS, NR, 1, NR, SIZE], pl.FP32]],
        ) -> pl.Tensor[[ROUNDS, NR, 1, NR, SIZE], pl.FP32]:
            stage_buf_1 = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
            stage_buf_2 = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
            data_buf_1 = pld.alloc_window_buffer(pld.world_size() * SIZE * pl.FP32.get_byte())
            data_buf_2 = pld.alloc_window_buffer(pld.world_size() * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())
            signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)

            # Round 1 — distinct stage/data windows per round, ONE shared signal.
            for r in pl.range(pld.world_size()):
                stage = pld.window(stage_buf_1, [1, SIZE], dtype=pl.FP32)
                self.publish_orch(inputs[0, r], stage, r, pld.world_size(), device=r)
            stage = pld.window(stage_buf_1, [1, SIZE], dtype=pl.FP32)
            data = pld.window(data_buf_1, [pld.world_size(), SIZE], dtype=pl.FP32)
            data = pld.tensor.allgather(stage, data, signal)
            for r in pl.range(pld.world_size()):
                self.consume_orch(data, outputs[0, r], pld.world_size(), device=r)

            # Round 2 — reuse the same signal.
            for r in pl.range(pld.world_size()):
                stage = pld.window(stage_buf_2, [1, SIZE], dtype=pl.FP32)
                self.publish_orch(inputs[1, r], stage, r, pld.world_size(), device=r)
            stage = pld.window(stage_buf_2, [1, SIZE], dtype=pl.FP32)
            data = pld.window(data_buf_2, [pld.world_size(), SIZE], dtype=pl.FP32)
            data = pld.tensor.allgather(stage, data, signal)
            for r in pl.range(pld.world_size()):
                self.consume_orch(data, outputs[1, r], pld.world_size(), device=r)
            return outputs

    return HostTensorAllGatherSignalReuse


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

        # The builtin collective must join the per-rank comm ordering chain. It submits via
        # its own orch.submit_next_level in EmitBuiltinWindowCollectiveDispatch rather than
        # the custom-kernel path, so without the ordering token it would sit outside the
        # chain -- and a rank mixing a builtin collective with a custom comm kernel (exactly
        # this program: publish_orch -> allgather -> consume_orch) could then have a waiting
        # dispatch routed ahead of its producer and deadlock.
        orch_src = (compiled.output_dir / "orchestration" / "host_orch.py").read_text()
        # One separate allocation per rank, not rows of a single tensor. The runtime keys a
        # host tensor's dependencies on its storage base, so every view of one storage fuses
        # into a single node -- rows of one tensor would collapse the per-rank chains into one
        # global order, which a program whose ranks must be in flight together cannot satisfy.
        assert re.search(r'_ord"\] = \[torch\.zeros\(.*for _ in range\(', orch_src), (
            "comm ordering token is not a per-rank list of separate allocations"
        )
        submits = [
            line
            for line in orch_src.splitlines()
            if "_submit_chip(" in line or "orch.submit_next_level(" in line
        ]
        assert any("builtin.tensor.allgather" in line for line in submits), submits
        for submit in submits:
            ta_var = re.search(r"(_ta_\d+)", submit).group(1)
            token_lines = [
                line.strip()
                for line in orch_src.splitlines()
                if line.strip().startswith(f"{ta_var}.add_tensor(") and '_ord"][' in line
            ]
            assert token_lines, f"dispatch without an ordering token: {submit.strip()}"
            # The token goes through the same make_tensor_arg(worker, tensor) form as every
            # other host tensor arg. Asserting the form, not just the presence, is what
            # separates "no token" from "token emitted as an uncallable line".
            for token in token_lines:
                assert "make_tensor_arg(orch._worker, " in token, (
                    f"ordering token is not a well-formed make_tensor_arg call: {token}"
                )

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, n_ranks, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_allgather(inputs, n_ranks)
        assert torch.allclose(outputs, expected), (
            f"host allgather P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_host_tensor_allgather_signal_reuse(self, test_config, device_ids, n_ranks):
        """Reuse ONE signal buffer across 2 back-to-back allgather calls."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"host allgather P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        rounds = 2
        compiled = ir.compile(
            _build_host_allgather_signal_reuse_program(),
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )
        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.allgather__fp32"
        assert variant_dir.is_dir()

        # Each round carries a distinct offset so a stale earlier-round result
        # (a missed epilogue reset) cannot match the round's golden.
        inputs = torch.stack([_make_rank_inputs(n_ranks, round_offset=rd * 10000.0) for rd in range(rounds)])
        outputs = torch.zeros((rounds, n_ranks, 1, n_ranks, SIZE), dtype=torch.float32)
        compiled(inputs, outputs)

        for rd in range(rounds):
            expected = _expected_allgather(inputs[rd], n_ranks)
            assert torch.allclose(outputs[rd], expected), (
                f"host allgather signal-reuse round {rd} P={n_ranks} mismatch: "
                f"max diff = {(outputs[rd] - expected).abs().max().item()}"
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"] + sys.argv[1:]))
