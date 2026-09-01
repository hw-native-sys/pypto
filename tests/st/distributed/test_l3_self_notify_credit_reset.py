# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: self-clearing credit-barrier ISA primitives.

Validates, with no compiler involvement, the two riskiest assumptions behind
the self-clearing credit-barrier protocol proposed for issue #2156 (see
``pld.tensor.*`` collectives' ``EmitEpilogueReset``):

1. **Negative ``AtomicAdd`` is legal.** ``pld.system.notify`` accepts a
   negative ``value`` and it is applied as a genuine signed atomic add.
2. **Self-addressed notify (``peer == my_rank``) lands on the same cell as a
   remote notify.** If it resolved to a different address, a rank's own
   epilogue reset would not affect the value its peers read, and this test's
   golden would read back the un-reset credit total instead of 0.

Each rank:

* notifies its peer's signal cell with ``AtomicAdd(+1)``, ``REPS`` times —
  mirroring one ``EmitBarrier`` notify per barrier a composite collective
  issues;
* waits for its own cell to reach ``>= REPS`` (an ordinary barrier, so the
  self-notify below is provably not racing the peer's writes);
* self-notifies its own cell with a single ``AtomicAdd(-REPS)`` — exactly the
  shape of ``LoweringBuilder::EmitEpilogueReset``'s reset notify;
* reads its own cell back.

Golden: ``outputs[r] == 0`` for every rank — the peer's ``REPS`` credits and
the rank's own ``-REPS`` self-notify must land on the identical memory cell
and cancel exactly.

Runs on 2 devices via ``DistributedConfig(device_ids=device_ids[:2], ...)``.
Pytest skips only when fewer than 2 devices are available.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir import DistributedConfig

_REPS = 4


def _build_self_notify_credit_reset_program():
    """Build the self-notify credit-reset program at call time.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """

    @pl.program
    class SelfNotifyCreditReset:
        @pl.function(type=pl.FunctionType.InCore)
        def credit_and_reset(
            self,
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            # Phase 1: notify the peer REPS times — one AtomicAdd(+1) per
            # notify, mirroring one barrier's worth of credit per iteration.
            for _ in pl.unroll(_REPS):
                pld.system.notify(
                    target=signal,
                    peer=peer,
                    offsets=[0, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

            # Phase 2: ordinary barrier — wait until our own cell has
            # accumulated all REPS credits from the rank whose peer is us.
            # This rules out the self-notify below racing the peer's writes.
            pld.system.wait(
                signal=signal,
                offsets=[0, 0],
                expected=_REPS,
                cmp=pld.WaitCmp.Ge,
            )

            # Phase 3: self-notify — subtract REPS from our OWN cell in one
            # AtomicAdd, exactly the shape of EmitEpilogueReset's reset
            # notify. peer=my_rank means this targets the local window, not
            # a remote one.
            # The value must have explicit INT32 dtype to match the signal
            # element type (pto.comm.tnotify verifies this at ptoas level).
            neg_reps: pl.Scalar[pl.INT32] = pl.const(-4, pl.INT32)  # -_REPS typed
            pld.system.notify(
                target=signal,
                peer=my_rank,
                offsets=[0, 0],
                value=neg_reps,
                op=pld.NotifyOp.AtomicAdd,
            )

            # Phase 4: read our own cell back. If the self-notify landed on a
            # different address than the peer's remote adds (the one risk
            # this ST exists to rule out), this reads REPS instead of 0.
            val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            return self.credit_and_reset(out, signal, peer, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            outputs: pl.Out[pl.Tensor[[2, 1, 1], pl.INT32]],
        ) -> pl.Tensor[[2, 1, 1], pl.INT32]:
            signal_buf = pld.alloc_window_buffer(pl.INT32.get_byte())  # 1x1 x INT32

            for r in pl.range(pld.world_size()):
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                self.chip_orch(outputs[r], signal, (r + 1) % pld.world_size(), r, device=r)
            return outputs

    return SelfNotifyCreditReset


class TestL3SelfNotifyCreditReset:
    """L3 distributed runtime: self-clearing credit-barrier ISA primitives."""

    @staticmethod
    def _compile(test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"self-notify credit-reset needs 2 devices, got {device_ids}")

        program = _build_self_notify_credit_reset_program()
        return ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

    @staticmethod
    def _assert_zeroed(outputs):
        expected = torch.zeros((2, 1, 1), dtype=torch.int32)
        got = outputs.flatten().tolist()
        assert torch.equal(outputs, expected), (
            f"self-notify credit reset mismatch: got {got}, want all-zero "
            f"(peer's {_REPS} credits must cancel this rank's own -{_REPS} self-notify)"
        )

    def test_self_notify_credit_reset(self, test_config, device_ids):
        compiled = self._compile(test_config, device_ids)
        outputs = torch.zeros((2, 1, 1), dtype=torch.int32)
        compiled(outputs)
        self._assert_zeroed(outputs)

    def test_persistent_self_notify_credit_reset_without_window_reset(self, test_config, device_ids):
        """Repeated persistent dispatches stay zeroed with no host-side reset.

        ``reset_persistent_windows=False`` is exactly the mode the self-
        clearing protocol is meant to make safe: the signal must return to
        all-zero on-device after every call, with no synchronous host memset
        between dispatches.
        """
        compiled = self._compile(test_config, device_ids)
        outputs = torch.zeros((2, 1, 1), dtype=torch.int32).share_memory_()

        with compiled.prepare(persistent=True, reset_persistent_windows=False) as worker:
            for _ in range(3):
                outputs.zero_()
                worker(outputs)
                self._assert_zeroed(outputs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
