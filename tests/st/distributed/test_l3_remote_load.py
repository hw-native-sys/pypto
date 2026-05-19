# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: ``pld.tile.remote_load`` ring-shuffle.

End-to-end exercise of the N6 cross-rank tile load primitive:

* Rank ``r`` is given local input ``inputs[r]``.
* Each rank stages ``inputs[r]`` into its own slice of the window-bound
  ``data`` buffer (a plain local ``pl.store`` into the DistributedTensor).
* All ranks barrier on ``data`` being staged (``pld.system.notify`` +
  ``pld.system.wait`` against an INT32 signal slot).
* Rank ``r`` then ``pld.tile.remote_load``s the peer slice belonging to
  ``(r + 1) % nranks`` and writes it to ``outputs[r]``.

Golden: ``outputs[r, i] == inputs[(r + 1) % nranks, i]``.

The test is currently **skipped** — the InCore PTO codegen for
``pld.tile.remote_load`` / ``pld.system.notify`` / ``pld.system.wait``
is in place (N6 P1), but the host-side glue still has open work:

* ``tile.store(tile, offsets, dst)`` verifier must accept a
  ``DistributedTensorType`` ``dst`` so the Phase-1 stage-in works.
* **N7** distributed_codegen.cpp must emit one
  ``chip_args.add_scalar(ctx.device_ctx[group_idx])`` per
  ``DistributedTensor`` formal parameter (in IR-parameter order),
  plus the ``ContinuousTensor.make(..., child_memory=True)`` wrapper
  for each ``DistributedTensor`` arg.
* **N8** distributed_runner.py must thread ``HostBufferStaging`` /
  ``ChipBootstrapConfig`` for the inferred CommGroup so the runtime
  knows which physical buffer to bind to each rank's window slot.

Drop ``pytest.mark.skip`` (and inline the @pl.program decorator at
module top-level) once the above land — the program below and the
golden check are the canonical e2e contract for N6 ops.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64


def _build_ring_shuffle_program():
    """Build the ring-shuffle program at call time.

    Deferred construction lets this test file collect even when the IR
    parser rejects the embedded body (e.g. the Phase-1 ``pl.store`` into
    a ``DistributedTensor`` is not yet accepted by ``tile.store``'s
    verifier). The skip marker on ``TestL3RemoteLoad`` ensures the body
    never runs until the pending work lands.
    """

    @pl.program
    class RemoteLoadRingShuffle:
        @pl.function(type=pl.FunctionType.InCore)
        def ring_step(
            self,
            inp: pl.Tensor[[SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[SIZE], pl.FP32]],
            data: pld.DistributedTensor[[SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[SIZE], pl.FP32]:
            # Phase 1: stage-in — local input → this rank's window slice.
            local = pl.load(inp, [0], [SIZE])
            _ = pl.store(local, [0], data)

            # Phase 2: signal that this rank has staged, then wait for the peer's
            # stage to land. AtomicAdd on a single signal cell is sufficient for
            # the 2-rank ring; nranks > 2 would size the signal to nranks and
            # let each rank set a distinct slot.
            pld.system.notify(
                target=signal,
                peer=peer,
                offsets=[0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(
                signal=signal,
                offsets=[0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

            # Phase 3: read the peer's slice via the cross-rank load.
            recv = pld.tile.remote_load(data, peer=peer, offsets=[0], shape=[SIZE])
            return pl.store(recv, [0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[SIZE], pl.FP32]],
            data: pld.DistributedTensor[[SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[SIZE], pl.FP32]:
            return self.ring_step(inp, out, data, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(SIZE * 4)  # SIZE × FP32 (4 bytes)
            signal_buf = pld.alloc_window_buffer(4)  # 1 × INT32

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1], dtype=pl.INT32)
                # Ring partner: rank r reads from peer = (r + 1) % nranks.
                self.chip_orch(inputs[r], outputs[r], data, signal, (r + 1) % pld.world_size(), device=r)
            return outputs

    return RemoteLoadRingShuffle


@pytest.mark.skip(
    reason=(
        "pld.tile.remote_load end-to-end requires: (a) tile.store accepting "
        "DistributedTensor destinations (Phase-1 stage-in), (b) N7 host_orch "
        "python codegen emitting add_scalar(ctx) per DistributedTensor, "
        "(c) N8 driver wiring CommGroup window buffers. The InCore PTO "
        "codegen (N6 P1) is in place — drop this skip once (a)-(c) land."
    )
)
class TestL3RemoteLoad:
    """L3 distributed runtime: ring-shuffle via pld.tile.remote_load."""

    def test_ring_shuffle(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"ring-shuffle needs 2 devices, got {device_ids}")

        program = _build_ring_shuffle_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        # Per-rank input: rank 0 holds [0, 1, …, SIZE-1]; rank 1 holds
        # [100, 101, …]. After the ring shuffle, rank 0's output should
        # contain rank 1's input (peer=1) and vice versa.
        inputs = torch.stack(
            [
                torch.arange(SIZE, dtype=torch.float32),
                torch.arange(100.0, 100.0 + SIZE, dtype=torch.float32),
            ]
        )
        outputs = torch.zeros((2, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        # rank r reads peer = (r + 1) % nranks → outputs[0] = inputs[1], etc.
        expected = torch.stack([inputs[1], inputs[0]])
        assert torch.allclose(outputs, expected), (
            f"ring-shuffle mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
