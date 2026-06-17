# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank broadcast via ``pld.tensor.broadcast`` intrinsic.

Same on-board semantics as ``test_l3_broadcast.py`` — but the InCore body
calls the new composite intrinsic rather than hand-rolling notify/wait/remote_load.

Golden: every rank's output equals root's input.  Non-root inputs must not
appear in outputs.

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
ROOT_RANK = 0


def _expected_broadcast(inputs: torch.Tensor, root: int = ROOT_RANK) -> torch.Tensor:
    """Root row replicated on every rank."""
    root_row = inputs[root, 0]
    return torch.stack([root_row] * inputs.shape[0]).unsqueeze(1)


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Distinct per-rank tensors so root-only broadcast is non-trivial."""
    rows = [
        torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _build_broadcast_program(n_ranks: int):
    """Build an N-rank broadcast program at call time using the intrinsic.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """
    nr = n_ranks

    @pl.program
    class BroadcastIntrinsicNRank:
        @pl.function(type=pl.FunctionType.InCore)
        def broadcast_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Phase 1: root only stages data.
            if my_rank == ROOT_RANK:
                local = pl.load(inp, [0, 0], [1, SIZE])
                pl.store(local, [0, 0], data)

            # Phases 2-3: barrier + broadcast — one call.
            data = pld.tensor.broadcast(data, signal, root=ROOT_RANK)

            # Stage-out: every rank reads root's data.
            acc = pl.load(data, [0, 0], [1, SIZE])
            return pl.store(acc, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.broadcast_step(inp, out, data, signal, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(SIZE * 4)
            signal_buf = pld.alloc_window_buffer(nr * 4)  # nr x 1 x INT32

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, sig, r, device=r)
            return outputs

    return BroadcastIntrinsicNRank


class TestL3TensorBroadcastIntrinsic:
    """L3 distributed runtime: N-rank broadcast via ``pld.tensor.broadcast``.

    Validates that the lowered composite produces an on-board result
    bit-identical to the hand-written ``test_l3_broadcast.py`` reference.
    """

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_broadcast_intrinsic(self, test_config, device_ids, n_ranks):
        """Compile and run mesh broadcast for P=2 or P=4; skip when devices are scarce."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"broadcast P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_broadcast_program(n_ranks)
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

        expected = _expected_broadcast(inputs)
        assert torch.allclose(outputs, expected), (
            f"broadcast intrinsic P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
