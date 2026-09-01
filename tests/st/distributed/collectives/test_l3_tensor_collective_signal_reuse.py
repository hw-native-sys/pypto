# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: back-to-back composite collectives sharing one signal.

Regression coverage for the reusable barrier-signal protocol (issue #2156).
Each program invokes the same collective **twice on the same signal buffer**
with different inputs and validates that the second result is complete and
reflects only the second invocation's inputs.

Under the old ``Set(1)`` + ``Wait(>= 1)`` protocol every signal cell stayed at
``1`` after the first call, so the second call's waits were satisfied from stale
state before peers had pushed the second invocation's data — producing mixed or
truncated output. The self-clearing credit barrier (``AtomicAdd(+1)`` →
``Wait(>= g)`` plus a per-call epilogue ``AtomicAdd(-N)``) makes each call a
stateless cycle, so a back-to-back call on the same signal restarts at
generation 1 with no stale state.

ST coverage: **P=2** (default CI / 2-device hosts) and **P=4** (any four
devices). All programs use the same N-rank body.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir import DistributedConfig

SIZE = 64

# Offset separating the two invocations' inputs. Large enough that any element
# leaking from the first call is far outside the second call's value range.
SECOND_CALL_OFFSET = 100000.0


# ---------------------------------------------------------------------------
# Golden helpers
# ---------------------------------------------------------------------------


def _all_to_all_inputs(n_ranks: int, offset: float) -> torch.Tensor:
    """``input[r, d, j] = offset + r*1000 + d*100 + j`` — chunk for dest ``d``."""
    r = torch.arange(n_ranks, dtype=torch.float32).view(-1, 1, 1)
    d = torch.arange(n_ranks, dtype=torch.float32).view(1, -1, 1)
    j = torch.arange(SIZE, dtype=torch.float32).view(1, 1, -1)
    return offset + r * 1000 + d * 100 + j


def _expected_all_to_all(n_ranks: int, offset: float) -> torch.Tensor:
    """``output[rank, src, j] = offset + src*1000 + rank*100 + j``."""
    src = torch.arange(n_ranks, dtype=torch.float32).view(1, -1, 1)
    rank = torch.arange(n_ranks, dtype=torch.float32).view(-1, 1, 1)
    j = torch.arange(SIZE, dtype=torch.float32).view(1, 1, -1)
    return offset + src * 1000 + rank * 100 + j


def _allgather_inputs(n_ranks: int, offset: float) -> torch.Tensor:
    """``input[r, 0, j] = offset + r*100 + j`` — this rank's single chunk."""
    r = torch.arange(n_ranks, dtype=torch.float32).view(-1, 1, 1)
    j = torch.arange(SIZE, dtype=torch.float32).view(1, 1, -1)
    return offset + r * 100 + j


def _expected_allgather(n_ranks: int, offset: float) -> torch.Tensor:
    """Every rank ends with the same gathered ``[NR, SIZE]`` block."""
    gathered = _allgather_inputs(n_ranks, offset).reshape(n_ranks, SIZE)
    return gathered.unsqueeze(0).expand(n_ranks, n_ranks, SIZE).contiguous()


def _reduce_scatter_inputs(n_ranks: int, offset: float) -> torch.Tensor:
    """``input[r, 0, :]`` holds ``n_ranks`` contiguous chunks of ``SIZE``."""
    rows = [
        offset + torch.arange(r * 100.0, r * 100.0 + n_ranks * SIZE, dtype=torch.float32)
        for r in range(n_ranks)
    ]
    return torch.stack(rows).reshape(n_ranks, 1, n_ranks * SIZE)


def _expected_reduce_scatter(inputs: torch.Tensor) -> torch.Tensor:
    """Rank ``r`` receives the element-wise sum of chunk ``r`` across ranks."""
    n_ranks = inputs.shape[0]
    chunks = [inputs[:, 0, r * SIZE : (r + 1) * SIZE].sum(dim=0) for r in range(n_ranks)]
    return torch.stack(chunks).reshape(n_ranks, 1, SIZE)


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------


def _build_all_to_all_twice(n_ranks: int):
    """All-to-all twice on one signal, bracketed by bare barriers.

    The surrounding ``pld.tensor.barrier`` calls chain onto the same signal, so
    the signal also has to return to all-zero after each call when reset by a
    *different* collective in the family.
    """
    nr = n_ranks

    @pl.program
    class AllToAllSignalReuse:
        @pl.function(type=pl.FunctionType.InCore)
        def exchange_twice(
            self,
            first: pl.Tensor[[nr, SIZE], pl.FP32],
            second: pl.Tensor[[nr, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
            signal = pld.tensor.barrier(signal)
            data = pld.tensor.all_to_all(first, data, signal)
            data = pld.tensor.all_to_all(second, data, signal)
            signal = pld.tensor.barrier(signal)
            for src in pl.range(nr):
                chunk = pl.load(data, [src, 0], [1, SIZE])
                pl.store(chunk, [src, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            first: pl.Tensor[[nr, SIZE], pl.FP32],
            second: pl.Tensor[[nr, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
            return self.exchange_twice(first, second, out, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            firsts: pl.Tensor[[nr, nr, SIZE], pl.FP32],
            seconds: pl.Tensor[[nr, nr, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, nr, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, nr, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(nr * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(firsts[r], seconds[r], outputs[r], data, sig, device=r)
            return outputs

    return AllToAllSignalReuse


def _build_allgather_twice(n_ranks: int):
    """All-gather twice on one signal, second call with different local data."""
    nr = n_ranks

    @pl.program
    class AllGatherSignalReuse:
        @pl.function(type=pl.FunctionType.InCore)
        def gather_twice(
            self,
            first: pl.Tensor[[1, SIZE], pl.FP32],
            second: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
            data = pld.tensor.allgather(first, data, signal)
            data = pld.tensor.allgather(second, data, signal)
            for src in pl.range(nr):
                chunk = pl.load(data, [src, 0], [1, SIZE])
                pl.store(chunk, [src, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            first: pl.Tensor[[1, SIZE], pl.FP32],
            second: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
            return self.gather_twice(first, second, out, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            firsts: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            seconds: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, nr, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, nr, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(nr * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(firsts[r], seconds[r], outputs[r], data, sig, device=r)
            return outputs

    return AllGatherSignalReuse


def _build_reduce_scatter_twice(n_ranks: int):
    """Reduce-scatter twice on one signal, re-staging between the two calls."""
    nr = n_ranks

    @pl.program
    class ReduceScatterSignalReuse:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_twice(
            self,
            first: pl.Tensor[[1, nr * SIZE], pl.FP32],
            second: pl.Tensor[[1, nr * SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            for j in pl.range(nr):
                chunk = pl.load(first, [0, j * SIZE], [1, SIZE])
                pl.store(chunk, [j, 0], data)
            data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)

            # Re-stage the second invocation's chunks over the first result and
            # reduce again on the same signal.
            for j in pl.range(nr):
                chunk = pl.load(second, [0, j * SIZE], [1, SIZE])
                pl.store(chunk, [j, 0], data)
            data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)

            acc = pl.load(data, [my_rank, 0], [1, SIZE])
            return pl.store(acc, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            first: pl.Tensor[[1, nr * SIZE], pl.FP32],
            second: pl.Tensor[[1, nr * SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.reduce_twice(first, second, out, data, signal, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            firsts: pl.Tensor[[nr, 1, nr * SIZE], pl.FP32],
            seconds: pl.Tensor[[nr, 1, nr * SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(nr * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(firsts[r], seconds[r], outputs[r], data, sig, r, device=r)
            return outputs

    return ReduceScatterSignalReuse


def _compile(program, test_config, device_ids, n_ranks):
    return ir.compile(
        program,
        platform=test_config.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids[:n_ranks],
            num_sub_workers=0,
        ),
    )


class TestL3TensorCollectiveSignalReuse:
    """Back-to-back collectives on a shared signal buffer stay correct."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_all_to_all_back_to_back(self, test_config, device_ids, n_ranks):
        """The second all-to-all must reflect only the second inputs.

        The program also brackets the pair with ``pld.tensor.barrier`` on the
        same signal, so the signal is reset by two different collectives'
        epilogues.
        """
        if len(device_ids) < n_ranks:
            pytest.skip(f"all-to-all P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = _compile(_build_all_to_all_twice(n_ranks), test_config, device_ids, n_ranks)
        firsts = _all_to_all_inputs(n_ranks, 0.0)
        seconds = _all_to_all_inputs(n_ranks, SECOND_CALL_OFFSET)
        outputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)

        compiled(firsts, seconds, outputs)

        expected = _expected_all_to_all(n_ranks, SECOND_CALL_OFFSET)
        assert torch.allclose(outputs, expected), (
            f"back-to-back all-to-all P={n_ranks} leaked first-call data: "
            f"max diff = {(outputs - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_allgather_back_to_back(self, test_config, device_ids, n_ranks):
        """The second all-gather must reflect only the second local chunks."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"allgather P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = _compile(_build_allgather_twice(n_ranks), test_config, device_ids, n_ranks)
        firsts = _allgather_inputs(n_ranks, 0.0)
        seconds = _allgather_inputs(n_ranks, SECOND_CALL_OFFSET)
        outputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)

        compiled(firsts, seconds, outputs)

        expected = _expected_allgather(n_ranks, SECOND_CALL_OFFSET)
        assert torch.allclose(outputs, expected), (
            f"back-to-back allgather P={n_ranks} leaked first-call data: "
            f"max diff = {(outputs - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_reduce_scatter_back_to_back(self, test_config, device_ids, n_ranks):
        """The second reduce-scatter must reduce only the re-staged chunks."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"reduce-scatter P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = _compile(_build_reduce_scatter_twice(n_ranks), test_config, device_ids, n_ranks)
        firsts = _reduce_scatter_inputs(n_ranks, 0.0)
        seconds = _reduce_scatter_inputs(n_ranks, SECOND_CALL_OFFSET)
        outputs = torch.zeros((n_ranks, 1, SIZE), dtype=torch.float32)

        compiled(firsts, seconds, outputs)

        expected = _expected_reduce_scatter(seconds)
        assert torch.allclose(outputs, expected), (
            f"back-to-back reduce-scatter P={n_ranks} mixed in first-call data: "
            f"max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
