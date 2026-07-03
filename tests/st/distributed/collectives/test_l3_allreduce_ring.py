# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank ring allreduce — chunked reduce-scatter + allgather schedule.

Ports the ring algorithm from simpler ``allreduce_ring_distributed``
(`#975 <https://github.com/hw-native-sys/simpler/pull/975>`_) into the PyPTO
DSL as a monolithic InCore kernel — one AIV task dispatch per rank.

Schedule (NCCL-style RS + AG, matching ``allreduce_ring_kernel.cpp``):

* **Phase 1 (stage-in)** — partition each rank's input into ``NR`` equal
  chunks in the HCCL-window ``scratch`` buffer via ``pl.range`` loop.
* **Phase 2 (reduce-scatter)** — ``(NR−1)`` ring steps. Per step: barrier
  (notify-all / wait-all on per-round signal row), ``pld.tile.remote_load``
  left neighbour's chunk, ``pl.add`` into local accumulator chunk.
* **Phase 3 (allgather)** — ``(NR−1)`` ring steps. Per step: barrier,
  ``pld.tile.remote_load`` left neighbour's chunk, store into local chunk.
* **Phase 4 (stage-out)** — concatenate ``chunks[]`` → output via loop.

Golden matches mesh allreduce: ``output[i] = P·i + 100·P·(P−1)/2`` (element-wise
sum of all rank inputs).  Ring uses ``2(P−1)`` per-round barriers vs mesh's single
global barrier; remote traffic is ``O(N/P)`` per step vs mesh's ``O(N)``.

ST coverage: **P=2** (default CI / 2-device hosts) and **P=4** (any four devices,
e.g. ``--device=0,1,2,3`` or ``--device=0-3``).  P=8 / P=16 are valid
(``256 % P == 0``) and can be added to the parametrize list.
"""

# pyright: reportUndefinedVariable=false

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 256  # matches ALLREDUCE_COUNT in simpler allreduce_ring_kernel.cpp


def _expected_allreduce(inputs: torch.Tensor) -> torch.Tensor:
    """Element-wise sum of all rank inputs, replicated on every rank."""
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
    """Build a fixed-P ring allreduce ``@pl.program`` for the given rank count.

    Static chunk size and round count are computed from *n_ranks* at factory
    time, avoiding dynamic-shape arithmetic in type annotations.
    """
    total_rounds = 2 * (n_ranks - 1)
    chunk = SIZE // n_ranks  # Python int — used in tile shapes to avoid SSA renaming

    @pl.program
    class RingAllReduce:
        """Ring allreduce with chunked reduce-scatter + allgather schedule."""

        @pl.function(type=pl.FunctionType.InCore)
        def ring_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, n_ranks], pl.INT32]],
            chunk_elems: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """Monolithic ring allreduce: stage-in → RS loop → AG loop → stage-out.

            *scratch* holds ``NR`` chunks laid out flat in ``[1, SIZE]``;
            chunk *c* starts at offset ``c * chunk_elems``.  *signal* carries
            ``2(NR−1)`` rows, one per RS/AG round; each row has ``NR`` cells.
            ``alloc_window_buffer`` zero-initialises every cell, so per-round
            ``AtomicAdd(0→1)`` / ``WaitGe(1)`` is safe without explicit reset.
            """
            ctx = pld.get_comm_ctx(scratch)
            my_rank = pld.rank(ctx)
            nranks = pld.nranks(ctx)
            left = (my_rank - 1 + nranks) % nranks

            # Phase 1: stage-in — copy each local input chunk into scratch.
            for c in pl.range(nranks):
                src_tile = pl.load(inp, [0, c * chunk_elems], [1, chunk])
                pl.store(src_tile, [0, c * chunk_elems], scratch)

            # Phase 2: reduce-scatter — (NR−1) ring steps.
            # s is 0-indexed; step = s + 1 is the 1-indexed ring step.
            for s in pl.range(nranks - 1):
                step = s + 1
                recv_add_idx = (my_rank - step - 1 + nranks) % nranks
                left_send_idx = (left - step + nranks) % nranks
                rs_round = s

                # Barrier — every rank notifies every peer, then waits on every peer.
                for peer in pl.range(nranks):
                    if peer != my_rank:
                        pld.system.notify(
                            signal,
                            peer=peer,
                            offsets=[rs_round, my_rank],
                            value=1,
                            op=pld.NotifyOp.AtomicAdd,
                        )
                for peer in pl.range(nranks):
                    if peer != my_rank:
                        pld.system.wait(
                            signal=signal,
                            offsets=[rs_round, peer],
                            expected=1,
                            cmp=pld.WaitCmp.Ge,
                        )

                # Remote-load left neighbour's send chunk, add into local accumulator.
                recv = pld.tile.remote_load(
                    scratch,
                    peer=left,
                    offsets=[0, left_send_idx * chunk_elems],
                    shape=[1, chunk],
                )
                acc = pl.load(scratch, [0, recv_add_idx * chunk_elems], [1, chunk])
                acc = pl.add(acc, recv)
                pl.store(acc, [0, recv_add_idx * chunk_elems], scratch)

            # Phase 3: allgather — (NR−1) ring steps.
            for s in pl.range(nranks - 1):
                step = s + 1
                recv_idx = (my_rank - step + nranks) % nranks
                left_send_idx = (left - step + 1 + nranks) % nranks
                ag_round = (nranks - 1) + s

                # Barrier.
                for peer in pl.range(nranks):
                    if peer != my_rank:
                        pld.system.notify(
                            signal,
                            peer=peer,
                            offsets=[ag_round, my_rank],
                            value=1,
                            op=pld.NotifyOp.AtomicAdd,
                        )
                for peer in pl.range(nranks):
                    if peer != my_rank:
                        pld.system.wait(
                            signal=signal,
                            offsets=[ag_round, peer],
                            expected=1,
                            cmp=pld.WaitCmp.Ge,
                        )

                # Remote-load left neighbour's send chunk, store into local chunk.
                recv = pld.tile.remote_load(
                    scratch,
                    peer=left,
                    offsets=[0, left_send_idx * chunk_elems],
                    shape=[1, chunk],
                )
                pl.store(recv, [0, recv_idx * chunk_elems], scratch)

            # Phase 4: stage-out — write concatenated chunks into output.
            for c in pl.range(nranks):
                src_tile = pl.load(scratch, [0, c * chunk_elems], [1, chunk])
                pl.store(src_tile, [0, c * chunk_elems], out)

            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, n_ranks], pl.INT32]],
            chunk_elems: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """Per-device orchestration wrapper around ``ring_step``."""
            return self.ring_step(inp, out, scratch, signal, chunk_elems)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[n_ranks, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[n_ranks, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[n_ranks, 1, SIZE], pl.FP32]:
            """Launch one chip orchestration per rank with shared window buffers."""
            scratch_buf = pld.alloc_window_buffer(SIZE * 4)  # NR × chunk × FP32
            signal_buf = pld.alloc_window_buffer(total_rounds * n_ranks * 4)  # rounds × NR × INT32

            chunk_elems = SIZE // n_ranks
            for r in pl.range(n_ranks):
                scratch = pld.window(scratch_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [total_rounds, n_ranks], dtype=pl.INT32)
                self.chip_orch(
                    inputs[r],
                    outputs[r],
                    scratch,
                    signal,
                    chunk_elems,
                    device=r,
                )
            return outputs

    return RingAllReduce


class TestL3RingAllReduce:
    """L3 distributed runtime: N-rank ring allreduce via chunked RS+AG + notify/wait + remote_load."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_ring_allreduce(self, test_config, device_ids, n_ranks):
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
            f"ring allreduce P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
