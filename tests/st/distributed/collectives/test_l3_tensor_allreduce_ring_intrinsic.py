# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank ring allreduce via ``pld.tensor.allreduce(mode="ring")``.

Same on-board semantics as ``test_l3_allreduce_ring.py`` — but the InCore body
calls the new composite intrinsic ``pld.tensor.allreduce(data, signal, mode="ring")``
rather than hand-rolling the 2(P−1)-step RS+AG ring loops.  After
``LowerCompositeOps`` (pass 14) expands the intrinsic into the chunked
reduce-scatter + allgather primitive tree, the kernel IR should match the
hand-written reference op-for-op.

The ring algorithm uses O(1) HCCL windows per rank (vs. O(P) for mesh),
2(P−1) per-round barriers (AtomicAdd→WaitGe), and chunk_size = SIZE // NR.

Signal shape for ring: ``[2 * (NR − 1), NR]`` — one row per ring round,
one cell per rank.  ``alloc_window_buffer`` zero-initialises every cell,
so per-round ``AtomicAdd(0→1)`` / ``WaitGe(1)`` is monotonic single-shot.

ST coverage: **P=2** (default CI / 2-device hosts) and **P=4** (any four
devices).  Both use the same N-rank program body.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 256  # matches ALLREDUCE_COUNT in simpler allreduce_ring_kernel.cpp


def _expected_allreduce(inputs: torch.Tensor) -> torch.Tensor:
    """Replicate the element-wise sum of all rank inputs on every rank."""
    reduced = inputs.sum(dim=0)
    return torch.stack([reduced] * inputs.shape[0])


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Distinct per-rank tensors so the golden sum is non-trivial."""
    rows = [
        torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _build_ring_allreduce_program(n_ranks: int):
    """Build an N-rank ring allreduce program using the composite intrinsic.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """
    nr = n_ranks
    total_rounds = 2 * (nr - 1)

    @pl.program
    class RingAllReduceIntrinsicNRank:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, nr], pl.INT32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """One-call ring allreduce via ``pld.tensor.allreduce(mode="ring")``.

            The intrinsic lowers in ``LowerCompositeOps`` to the chunked
            reduce-scatter + allgather ring schedule — the user writes one
            call and the compiler emits all the per-round barriers, remote
            loads, and accumulation loops.
            """
            # Stage-in: copy local input into this rank's HCCL window slot.
            local = pl.load(inp, [0, 0], [1, SIZE])
            data = pl.store(local, [0, 0], data)

            # One call — the composite rebinds ``data`` (in-place semantics,
            # same as ``pl.store``) so subsequent reads see the reduced slice.
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")

            # Stage-out — reduced result → local output.
            acc = pl.load(data, [0, 0], [1, SIZE])
            return pl.store(acc, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, nr], pl.INT32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """Per-device orchestration wrapper around ``reduce_step``."""
            return self.reduce_step(inp, out, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, SIZE], pl.FP32]:
            """Launch one chip orchestration per rank with shared window buffers.

            Ring signal shape is ``[2*(NR−1), NR]`` (rounds × ranks) — one
            row per ring round, one cell per rank for the per-round barrier.
            ``alloc_window_buffer`` zero-initialises so per-round
            ``AtomicAdd(0→1)`` / ``WaitGe(1)`` works without explicit reset.
            """
            data_buf = pld.alloc_window_buffer(SIZE * 4)  # 1 x SIZE x FP32
            signal_buf = pld.alloc_window_buffer(total_rounds * nr * 4)  # rounds × NR × INT32

            for r in pl.range(nr):
                data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [total_rounds, nr], dtype=pl.INT32)
                self.chip_orch(
                    inputs[r],
                    outputs[r],
                    data,
                    signal,
                    device=r,
                )
            return outputs

    return RingAllReduceIntrinsicNRank


class TestL3TensorRingAllReduceIntrinsic:
    """L3 distributed runtime: ring allreduce via the composite intrinsic.

    Validates that the lowered ring composite produces an on-board result
    bit-identical to the hand-written ``test_l3_allreduce_ring.py`` reference.
    """

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_ring_allreduce_intrinsic(self, test_config, device_ids, n_ranks):
        """Compile and run ring allreduce for P=2 or P=4; skip when devices are scarce."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"ring allreduce P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_ring_allreduce_program(n_ranks)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_allreduce(inputs)
        assert torch.allclose(outputs, expected), (
            f"ring allreduce intrinsic P={n_ranks} mismatch: "
            f"max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
