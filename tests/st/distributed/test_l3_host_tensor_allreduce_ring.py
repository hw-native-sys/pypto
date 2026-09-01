# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: host-orchestrator ``pld.tensor.allreduce(mode="ring")`` builtin dispatch."""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir import DistributedConfig

SIZE = 256


def _expected_allreduce(inputs: torch.Tensor) -> torch.Tensor:
    reduced = inputs.sum(dim=0)
    return torch.stack([reduced] * inputs.shape[0])


def _make_rank_inputs(n_ranks: int, round_offset: float = 0.0) -> torch.Tensor:
    """Build distinct per-rank rows; ``round_offset`` distinguishes reuse rounds."""
    rows = [
        torch.arange(r * 100.0 + round_offset, r * 100.0 + round_offset + SIZE, dtype=torch.float32).reshape(
            1, SIZE
        )
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _build_host_ring_allreduce_program(n_ranks: int):
    total_rounds = 2 * (n_ranks - 1) + 1

    @pl.program
    class HostTensorAllReduceRing:
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
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            return self.publish_step(inp, data)

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            data: pld.DistributedTensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            reduced = pl.load(data, [0, 0], [1, SIZE])
            return pl.store(reduced, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            data: pld.DistributedTensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.consume_step(data, out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[n_ranks, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[n_ranks, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[n_ranks, 1, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(total_rounds * n_ranks * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
                self.publish_orch(inputs[r], data, device=r)

            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [total_rounds, n_ranks], dtype=pl.INT32)
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")

            for r in pl.range(pld.world_size()):
                self.consume_orch(data, outputs[r], device=r)

            return outputs

    return HostTensorAllReduceRing


def _build_host_ring_allreduce_signal_reuse_program(n_ranks: int):
    """Host ring allreduce reusing ONE signal buffer across 3 back-to-back calls.

    Same self-clearing-credit-barrier intent as the mesh reuse test; the ring
    signal is the [2*(NR-1)+1, NR] per-round matrix and every used row is reset by
    the epilogue."""

    total_rounds = 2 * (n_ranks - 1) + 1
    rounds = 3

    @pl.program
    class HostTensorAllReduceRingSignalReuse:
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
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            return self.publish_step(inp, data)

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            data: pld.DistributedTensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            reduced = pl.load(data, [0, 0], [1, SIZE])
            return pl.store(reduced, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            data: pld.DistributedTensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.consume_step(data, out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[rounds, n_ranks, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[rounds, n_ranks, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[rounds, n_ranks, 1, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(total_rounds * n_ranks * pl.INT32.get_byte())
            signal = pld.window(signal_buf, [total_rounds, n_ranks], dtype=pl.INT32)

            # Round 1 — every round below reuses the shared ``signal``.
            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
                self.publish_orch(inputs[0, r], data, device=r)
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
            for r in pl.range(pld.world_size()):
                self.consume_orch(data, outputs[0, r], device=r)

            # Round 2 — reuse the same signal.
            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
                self.publish_orch(inputs[1, r], data, device=r)
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
            for r in pl.range(pld.world_size()):
                self.consume_orch(data, outputs[1, r], device=r)

            # Round 3 — reuse the same signal again.
            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
                self.publish_orch(inputs[2, r], data, device=r)
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
            for r in pl.range(pld.world_size()):
                self.consume_orch(data, outputs[2, r], device=r)

            return outputs

    return HostTensorAllReduceRingSignalReuse


class TestL3HostTensorAllReduceRing:
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_host_tensor_allreduce_ring(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"host ring allreduce P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_host_ring_allreduce_program(n_ranks)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.allreduce_ring__sum__fp32"
        assert variant_dir.is_dir()
        assert (variant_dir / "kernel_config.py").is_file()

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_allreduce(inputs)
        assert torch.allclose(outputs, expected), (
            f"host ring allreduce P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_host_tensor_allreduce_ring_signal_reuse(self, test_config, device_ids, n_ranks):
        """Reuse ONE ring signal matrix across 3 back-to-back calls."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"host ring allreduce P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        rounds = 3
        compiled = ir.compile(
            _build_host_ring_allreduce_signal_reuse_program(n_ranks),
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )
        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.allreduce_ring__sum__fp32"
        assert variant_dir.is_dir()

        # Each round carries a distinct offset so a stale round-1 result in a
        # later round (a missed epilogue reset) cannot match the round's golden.
        inputs = torch.stack([_make_rank_inputs(n_ranks, round_offset=rd * 10000.0) for rd in range(rounds)])
        outputs = torch.zeros_like(inputs)
        compiled(inputs, outputs)

        for rd in range(rounds):
            expected = _expected_allreduce(inputs[rd])
            assert torch.allclose(outputs[rd], expected), (
                f"host ring allreduce signal-reuse round {rd} P={n_ranks} mismatch: "
                f"max diff = {(outputs[rd] - expected).abs().max().item()}"
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
