# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank reduce-scatter via ``pld.tensor.reduce_scatter`` intrinsic.

Same on-board semantics as ``test_l3_reduce_scatter.py``.  Target shape
[NR, SIZE] — each rank stages all NR chunks before the call; after the call
rank r's row holds the element-wise sum of chunk r across all ranks.

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


def _expected_reduce_scatter(inputs: torch.Tensor) -> torch.Tensor:
    """Per-rank golden: sum of chunk r across all ranks."""
    n_ranks = inputs.shape[0]
    chunks = [inputs[:, 0, r * SIZE : (r + 1) * SIZE].sum(dim=0) for r in range(n_ranks)]
    return torch.stack(chunks).reshape(n_ranks, 1, SIZE)


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Distinct per-rank tensors with n_ranks contiguous chunks of SIZE each."""
    rows = [
        torch.arange(r * 100.0, r * 100.0 + n_ranks * SIZE, dtype=torch.float32).reshape(1, n_ranks * SIZE)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _build_reduce_scatter_program(n_ranks: int):
    """Build an N-rank reduce-scatter program at call time using the intrinsic.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """
    nr = n_ranks

    @pl.program
    class ReduceScatterIntrinsicNRank:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[1, nr * SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Stage-in: write each chunk at its row.
            for j in pl.range(nr):
                chunk = pl.load(inp, [0, j * SIZE], [1, SIZE])
                pl.store(chunk, [j, 0], data)

            # Reduce-scatter — one call.
            data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)

            # Stage-out: read my reduced chunk.
            acc = pl.load(data, [my_rank, 0], [1, SIZE])
            return pl.store(acc, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, nr * SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.reduce_step(inp, out, data, signal, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, nr * SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(nr * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, sig, r, device=r)
            return outputs

    return ReduceScatterIntrinsicNRank


class TestL3TensorReduceScatterIntrinsic:
    """L3 distributed runtime: N-rank reduce-scatter via ``pld.tensor.reduce_scatter``.

    Validates that the lowered composite produces an on-board result
    bit-identical to the hand-written ``test_l3_reduce_scatter.py`` reference.
    """

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_reduce_scatter_intrinsic(self, test_config, device_ids, n_ranks):
        """Compile and run mesh reduce-scatter for P=2 or P=4; skip when devices are scarce."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"reduce-scatter P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_reduce_scatter_program(n_ranks)
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

        expected = _expected_reduce_scatter(inputs)
        assert torch.allclose(outputs, expected), (
            f"reduce-scatter intrinsic P={n_ranks} mismatch: "
            f"max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
