# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank all-to-all via ``pld.tensor.all_to_all`` intrinsic.

Validates the composite all-to-all intrinsic produces the same rank-ordered
personalized exchange as the simpler ``all_to_all_distributed`` reference.

The intrinsic accepts four arguments: ``input`` (Tensor [NR, SIZE]),
``target`` (DistributedTensor [NR, SIZE] staging window), ``signal``, and
``out`` (plain Tensor [NR, SIZE]).  It handles the stage-in, synchronises,
uses ``pld.tile.get`` to transfer from peers, and writes directly into
``out``.

ST coverage: **P=2** (default CI / 2-device hosts) and **P=4** (any four
devices). Both use the same N-rank program body.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64


def _expected_all_to_all(inputs: torch.Tensor) -> torch.Tensor:
    """Golden matching simpler: output[src, j] = src*1000 + rank*100 + j.

    inputs shape: [nranks, nranks, SIZE] — inputs[r, d, :] is what rank r
    sends to rank d.
    """
    nranks = inputs.shape[0]
    outputs = torch.zeros((nranks, nranks, SIZE), dtype=torch.float32)
    for rank in range(nranks):
        for src in range(nranks):
            for j in range(SIZE):
                outputs[rank, src, j] = float(src * 1000 + rank * 100 + j)
    return outputs


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Each rank r fills input[r, d, j] = r*1000 + d*100 + j (chunk for dest d)."""
    rows = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    for r in range(n_ranks):
        for d in range(n_ranks):
            for j in range(SIZE):
                rows[r, d, j] = float(r * 1000 + d * 100 + j)
    return rows


def _build_all_to_all_program(n_ranks: int):
    """Build an N-rank all-to-all program at call time using the intrinsic."""
    nr = n_ranks

    @pl.program
    class AllToAllIntrinsicNRank:
        @pl.function(type=pl.FunctionType.InCore)
        def exchange_step(
            self,
            inp: pl.Tensor[[nr, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
            # All-to-all — intrinsic handles stage-in, sync, pld.tile.get
            # from peers, and writes directly into out.
            result = pld.tensor.all_to_all(inp, data, signal, out)
            return result

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[nr, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
            return self.exchange_step(inp, out, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, nr, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, nr, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, nr, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(nr * SIZE * 4)
            signal_buf = pld.alloc_window_buffer(nr * 4)

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, sig, device=r)
            return outputs

    return AllToAllIntrinsicNRank


class TestL3TensorAllToAllIntrinsic:
    """L3 distributed runtime: N-rank all-to-all via ``pld.tensor.all_to_all``.

    Validates that the lowered composite produces an on-board result
    bit-identical to the simpler ``all_to_all_distributed`` reference.
    """

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_all_to_all_intrinsic(self, test_config, device_ids, n_ranks):
        """Compile and run mesh all-to-all for P=2 or P=4; skip when devices are scarce."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"all-to-all P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_all_to_all_program(n_ranks)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_all_to_all(inputs)
        assert torch.allclose(outputs, expected), (
            f"all-to-all intrinsic P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
