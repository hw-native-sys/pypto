# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: ``pld.system.notify`` / ``pld.system.wait`` handshake.

End-to-end exercise of the N6 cross-rank signalling primitives, isolated
from the data path (``remote_load`` is used only to read a signal cell
back for the golden — there is no data shuffle here):

* Each rank ``r`` writes a distinct ``tag = r + 1`` into its **peer**'s
  signal cell via ``pld.system.notify(..., op=pld.NotifyOp.Set)`` with
  ``peer = (r + 1) % nranks``.
* Each rank then ``pld.system.wait(..., cmp=pld.WaitCmp.Ge, expected=1)``
  on its **own** signal cell — blocking until the rank that targets it
  (``r' = (r - 1) % nranks``) has run its notify. This is the barrier the
  test is really validating.
* After the barrier, rank ``r`` reads its own cell with a local
  ``pl.load(signal, ...)`` (``tile.load`` accepts a ``DistributedTensor``
  source) and writes the received tag to ``outputs[r]``.

Golden: ``outputs[r] == ((r - 1) % nranks) + 1``. For 2 ranks rank 0 reads
rank 1's tag (2) and rank 1 reads rank 0's tag (1), so ``outputs == [2, 1]``.
A missing/incorrect wait would let the read race ahead of the peer's notify
and observe the zero-initialised cell.

The test is currently **skipped** — the InCore PTO codegen for
``pld.system.notify`` / ``pld.system.wait`` is in place (N6 P1), but the
host-side glue still has open work:

* **N7** distributed_codegen.cpp must emit one
  ``chip_args.add_scalar(ctx.device_ctx[group_idx])`` per
  ``DistributedTensor`` formal parameter, plus the
  ``ContinuousTensor.make(..., child_memory=True)`` wrapper for each
  ``DistributedTensor`` arg.
* **N8** distributed_runner.py must thread ``HostBufferStaging`` /
  ``ChipBootstrapConfig`` for the inferred CommGroup so the runtime knows
  which physical buffer to bind to each rank's window slot.

Drop ``pytest.mark.skip`` once the above land — the program below and the
golden check are the canonical e2e contract for ``pld.system.notify`` /
``pld.system.wait``.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig


def _build_signal_handshake_program():
    """Build the notify/wait handshake program at call time.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser; the skip marker on ``TestL3NotifyWait``
    ensures the body never runs until the pending host-side work lands.
    """

    @pl.program
    class SignalHandshake:
        @pl.function(type=pl.FunctionType.InCore)
        def barrier_step(
            self,
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            tag: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            # Phase 1: write our tag into the peer's signal cell (Set).
            pld.system.notify(
                target=signal,
                peer=peer,
                offsets=[0, 0],
                value=tag,
                op=pld.NotifyOp.Set,
            )

            # Phase 2: barrier — block until our own cell has been written by
            # the rank whose peer is us (its tag is >= 1).
            pld.system.wait(
                signal=signal,
                offsets=[0, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

            # Phase 3: read our own cell back locally and surface the received
            # tag as the output.
            recv = pl.load(signal, [0, 0], [1, 1])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            tag: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            return self.barrier_step(out, signal, peer, tag)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            outputs: pl.Out[pl.Tensor[[2, 1, 1], pl.INT32]],
        ) -> pl.Tensor[[2, 1, 1], pl.INT32]:
            signal_buf = pld.alloc_window_buffer(4)  # 1×1 × INT32

            for r in pl.range(pld.world_size()):
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                # peer = (r + 1) % nranks; tag = r + 1.
                self.chip_orch(outputs[r], signal, (r + 1) % pld.world_size(), r + 1, device=r)
            return outputs

    return SignalHandshake


@pytest.mark.skip(
    reason=(
        "pld.system.notify / pld.system.wait end-to-end requires: (a) N7 "
        "host_orch python codegen emitting add_scalar(ctx) per "
        "DistributedTensor, plus the ContinuousTensor.make(..., "
        "child_memory=True) wrapper; (b) N8 driver wiring CommGroup window "
        "buffers. The InCore PTO codegen (N6 P1) is in place — drop this "
        "skip once (a)-(b) land."
    )
)
class TestL3NotifyWait:
    """L3 distributed runtime: cross-rank notify/wait handshake."""

    def test_signal_exchange(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"notify/wait handshake needs 2 devices, got {device_ids}")

        program = _build_signal_handshake_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        outputs = torch.zeros((2, 1, 1), dtype=torch.int32)
        compiled(outputs)

        # rank r reads its own cell, written by rank (r - 1) % nranks with
        # tag = ((r - 1) % nranks) + 1 → rank 0 sees 2, rank 1 sees 1.
        expected = torch.tensor([[[2]], [[1]]], dtype=torch.int32)
        got = outputs.flatten().tolist()
        assert torch.equal(outputs, expected), f"notify/wait handshake mismatch: got {got}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
