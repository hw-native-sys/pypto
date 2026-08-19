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


def _make_rank_inputs(n_ranks: int, round_offset: float = 0.0) -> torch.Tensor:
    """Build distinct per-rank rows; ``round_offset`` distinguishes reuse rounds."""
    rows = [
        torch.arange(
            r * 100.0 + round_offset, r * 100.0 + round_offset + n_ranks * SIZE, dtype=torch.float32
        ).reshape(n_ranks, SIZE)
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


def _build_host_reduce_scatter_signal_reuse_program():
    """Host reduce_scatter reusing ONE signal buffer across 2 back-to-back calls.

    The self-clearing epilogue restores the AtomicAdd cells to 0 after each
    call; without it the second call's Ge(1) wait passes on stale credits.
    """
    ROUNDS = 2

    @pl.program
    class HostTensorReduceScatterSignalReuse:
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
            inputs: pl.Tensor[[ROUNDS, NR, NR, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[ROUNDS, NR, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[ROUNDS, NR, 1, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())
            signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)

            for rd in pl.range(ROUNDS):
                for r in pl.range(pld.world_size()):
                    data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
                    self.publish_orch(inputs[rd, r], data, device=r)
                data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
                data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
                for r in pl.range(pld.world_size()):
                    self.consume_orch(data, outputs[rd, r], r, device=r)
            return outputs

    return HostTensorReduceScatterSignalReuse


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

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_host_tensor_reduce_scatter_signal_reuse(self, test_config, device_ids, n_ranks):
        """Reuse ONE signal buffer across 2 back-to-back reduce_scatter calls."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"host reduce_scatter P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        rounds = 2
        compiled = ir.compile(
            _build_host_reduce_scatter_signal_reuse_program(),
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )
        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.reduce_scatter__sum__fp32"
        assert variant_dir.is_dir()

        # Each round carries a distinct offset so a stale earlier-round result
        # (a missed epilogue reset) cannot match the round's golden.
        inputs = torch.stack([_make_rank_inputs(n_ranks, round_offset=rd * 10000.0) for rd in range(rounds)])
        outputs = torch.zeros((rounds, n_ranks, 1, SIZE), dtype=torch.float32)
        compiled(inputs, outputs)

        for rd in range(rounds):
            expected = _expected_reduce_scatter(inputs[rd], n_ranks)
            assert torch.allclose(outputs[rd], expected), (
                f"host reduce_scatter signal-reuse round {rd} P={n_ranks} mismatch: "
                f"max diff = {(outputs[rd] - expected).abs().max().item()}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
