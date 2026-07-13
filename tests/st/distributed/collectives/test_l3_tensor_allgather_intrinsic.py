# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank allgather via ``pld.tensor.allgather`` intrinsic.

Validates the composite allgather intrinsic produces the same rank-ordered
concatenation on every rank as the hand-written ``test_l3_allgather.py``.

Push-based (3-arg): ``pld.tensor.allgather(local_data, target, signal)`` —
each rank pushes its chunk to every peer's window slot via
``pld.tile.put`` (TPUT-based); after the notify/wait barrier, the window itself
holds the gathered ``[NR, SIZE]`` result (window-as-result).  The InCore
function then reads from the window and writes into the output tensor for
host-side verification.

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


def _expected_allgather(inputs: torch.Tensor) -> torch.Tensor:
    """Rank-ordered concatenation; identical on every rank."""
    gathered = torch.cat([inputs[r, 0] for r in range(inputs.shape[0])])
    return torch.stack([gathered] * inputs.shape[0]).unsqueeze(1)


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Distinct per-rank tensors so the golden concat is non-trivial."""
    rows = [
        torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _build_allgather_program(n_ranks: int):
    """Build an N-rank allgather program at call time using the intrinsic.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """
    nr = n_ranks

    @pl.program
    class AllGatherIntrinsicNRank:
        @pl.function(type=pl.FunctionType.InCore)
        def gather_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
            # Push-based allgather: window becomes the gathered [NR, SIZE] result.
            data = pld.tensor.allgather(inp, data, signal)
            # Stage-out: read from the gathered window into the output tensor.
            for r in pl.range(nr):
                chunk = pl.load(data, [r, 0], [1, SIZE])
                pl.store(chunk, [0, r * SIZE], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
            return self.gather_step(inp, out, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(nr * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, sig, device=r)
            return outputs

    return AllGatherIntrinsicNRank


class TestL3TensorAllGatherIntrinsic:
    """L3 distributed runtime: N-rank push-based allgather via ``pld.tensor.allgather``.

    Validates that the lowered composite produces an on-board result
    bit-identical to the hand-written ``test_l3_allgather.py`` reference.
    """

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_allgather_intrinsic(self, test_config, device_ids, n_ranks):
        """Compile and run mesh allgather for P=2 or P=4; skip when devices are scarce."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"allgather P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_allgather_program(n_ranks)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, n_ranks * SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_allgather(inputs)
        assert torch.allclose(outputs, expected), (
            f"allgather intrinsic P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
