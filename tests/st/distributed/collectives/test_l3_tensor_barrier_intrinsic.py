# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank peer-swap via ``pld.tensor.barrier`` intrinsic.

Validates that the composite barrier correctly serialises cross-rank
access: each rank writes its own data, calls ``pld.tensor.barrier(signal)``,
then reads the peer's data.  Without the barrier, the read could observe
stale / zero data.  With the intrinsic, the peer swap is guaranteed.

Golden: for rank r, ``outputs[r] == inputs[(r + 1) % n_ranks]``.

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


def _expected_peer_swap(inputs: torch.Tensor) -> torch.Tensor:
    """Cyclic shift: rank r gets rank (r+1)%N input."""
    n = inputs.shape[0]
    return torch.stack([inputs[(r + 1) % n] for r in range(n)])


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Distinct per-rank tensors so the swap is non-trivial."""
    rows = [
        torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _build_barrier_peer_swap_program(n_ranks: int):
    """Build an N-rank peer-swap program at call time using the barrier intrinsic.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """
    nr = n_ranks

    @pl.program
    class BarrierPeerSwapNRank:
        @pl.function(type=pl.FunctionType.InCore)
        def swap_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Stage-in: write my data into the window.
            local = pl.load(inp, [0, 0], [1, SIZE])
            pl.store(local, [0, 0], data)

            # Barrier — ensure all ranks have staged before anyone reads.
            signal = pld.tensor.barrier(signal)

            # Read peer's data after the barrier guarantees it's staged.
            recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.swap_step(inp, out, data, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(
                    inputs[r],
                    outputs[r],
                    data,
                    sig,
                    (r + 1) % pld.world_size(),
                    device=r,
                )
            return outputs

    return BarrierPeerSwapNRank


class TestL3TensorBarrierIntrinsic:
    """L3 distributed runtime: N-rank peer swap via ``pld.tensor.barrier``.

    Validates that the lowered composite produces an on-board result
    bit-identical to the hand-written ``test_l3_barrier.py`` reference.
    """

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_barrier_intrinsic(self, test_config, device_ids, n_ranks):
        """Compile and run peer swap for P=2 or P=4; skip when devices are scarce."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"barrier P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_barrier_peer_swap_program(n_ranks)
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

        expected = _expected_peer_swap(inputs)
        assert torch.allclose(outputs, expected), (
            f"barrier intrinsic P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
