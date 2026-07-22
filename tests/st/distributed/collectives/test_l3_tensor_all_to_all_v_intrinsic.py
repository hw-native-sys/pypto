# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: variable-size all-to-all via ``pld.tensor.all_to_all_v`` intrinsic.

Validates the variable-size composite all-to-all intrinsic (MPI_Alltoallv pattern)
produces the correct rank-ordered personalized exchange.

The intrinsic takes three arguments with flat 2D layouts for ptoas compatibility:
  - ``input`` (Tensor [NR*MAX_RECV, SIZE]) — per-destination chunks, zero-padded
  - ``target`` (DistributedTensor [NR*MAX_RECV, SIZE]) — flat 2D staging window
  - ``signal`` (DistributedTensor INT32) — barrier

Window-as-result pattern: the intrinsic returns the target window, and the caller
reads back with ``pl.load`` — exactly the same pattern as the symmetric
``pld.tensor.all_to_all``.

ST coverage: **P=2** (default CI / 2-device hosts) and **P=4** (any four devices).
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
MAX_RECV = 4


def _build_all_to_all_v_program(n_ranks: int, max_recv: int):
    """Build an N-rank variable-size all-to-all program."""
    nr = n_ranks
    mr = max_recv
    total = nr * mr

    @pl.program
    class AllToAllVIntrinsicNRank:
        @pl.function(type=pl.FunctionType.InCore)
        def exchange_step(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[total, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[total, SIZE], pl.FP32]:
            # Push-based all_to_all_v — intrinsic pushes chunks to peers and
            # returns data in-place (window-as-result).
            result = pld.tensor.all_to_all_v(inp, data, signal)
            # Read back from the window into out for host-side verification.
            for src in pl.range(nr):
                base = src * mr
                for r in pl.range(mr):
                    flat_row = base + r
                    chunk = pl.load(result, [flat_row, 0], [1, SIZE])
                    pl.store(chunk, [flat_row, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[total, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[total, SIZE], pl.FP32]:
            return self.exchange_step(inp, out, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, total, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, total, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, total, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(total * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [total, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, sig, device=r)
            return outputs

    return AllToAllVIntrinsicNRank


class TestL3TensorAllToAllVIntrinsic:
    """L3 distributed runtime: variable-size all-to-all via ``pld.tensor.all_to_all_v``."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_all_to_all_v_intrinsic(self, test_config, device_ids, n_ranks):
        """Compile and run variable-size all-to-all for P=2 or P=4.

        Each rank sends ``n_ranks - dest`` rows to each peer (variable counts).
        MAX_RECV=4 is the compile-time capacity, actual sends ≤ MAX_RECV.
        The test validates that the push-based decomposition produces the
        correct per-src per-dest exchange.
        """
        if len(device_ids) < n_ranks:
            pytest.skip(f"all_to_all_v P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        nr = n_ranks
        mr = MAX_RECV
        total = nr * mr

        program = _build_all_to_all_v_program(nr, mr)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:nr],
                num_sub_workers=0,
            ),
        )

        # Build inputs: 3D host view [nr, total, SIZE] = per-rank flat 2D
        # Rank r sends to dest d: rows dest*mr+k for k=0..n_rows-1
        # Value = r*1000 + d*100 + k*10 + j%10
        inputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        for r in range(nr):
            for d in range(nr):
                n_rows = nr - d  # variable send count
                base = d * mr
                for k in range(n_rows):
                    for j in range(SIZE):
                        inputs[r, base + k, j] = float(r * 1000 + d * 100 + k * 10 + j % 10)

        outputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        # Golden validation:
        # Rank rank receives from src the chunk that src sent to dest=rank.
        # Flat 2D layout: rows src*mr+k hold what src pushed for peer dest=rank.
        for rank in range(nr):
            for src in range(nr):
                n_rows = nr - rank  # what src sent to dest=rank
                base = src * mr
                for k in range(n_rows):
                    expected_row = inputs[src, rank * mr + k, :]
                    got_row = outputs[rank, base + k, :]
                    assert torch.allclose(got_row, expected_row, atol=1e-5), (
                        f"P={nr} rank={rank} src={src} row={k}: "
                        f"max diff = {(got_row - expected_row).abs().max().item()}"
                    )
                # Pad rows beyond n_rows should be zero
                for k in range(n_rows, mr):
                    got_row = outputs[rank, base + k, :]
                    assert torch.allclose(got_row, torch.zeros(SIZE), atol=1e-5), (
                        f"P={nr} rank={rank} src={src} pad row={k}: expected zeros, "
                        f"max = {got_row.abs().max().item()}"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
