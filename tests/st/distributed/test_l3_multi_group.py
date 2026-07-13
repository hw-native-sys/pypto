# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: 3-device, two-OVERLAPPING-comm-domain peer-swap.

End-to-end exercise of the multi-comm-domain codegen. The N4 ``MaterializeCommDomainScopes``
pass clusters allocs by their dispatch device subset; this test produces two
distinct (and intentionally *overlapping*) comm domains and validates that:

* Host_orch python emits two nested ``with orch.allocate_domain(...)`` blocks
  (one per group; verified at codegen-text level by
  ``test_two_groups_emit_nested_allocate_domain``).
* The simpler runtime brings up two independent HCCL communicators (one per
  group). The shared device participates in BOTH communicators without window
  aliasing — each group's window is fully isolated.
* DistributedTensor args route through the group-matching ``ChipContext``:
  group-A buffers live in ``__comm_d0``, group-B buffers live in ``__comm_d1``,
  with their own ``device_ctx`` per chip. The shared device (1) holds two
  distinct ``device_ctx`` handles, one per dispatch / group.

**Workload — two overlapping intra-group peer-swaps on 3 devices:**

* Group A = devices ``{0, 1}`` with ``scratch_a``/``signal_a`` window buffers.
* Group B = devices ``{1, 2}`` with ``scratch_b``/``signal_b`` window buffers.
* Device 1 participates in BOTH groups — it stages its input into both
  windows and runs two independent ``swap_step`` dispatches.
* Each rank stages its input into its window slice, ``pld.system.notify``s
  its single in-group peer (Add-1), ``wait``s on its own signal cell, then
  ``pld.tile.remote_load``s the peer's slice into its output.

**Golden:**

* ``outputs_a[0] == inputs[1]``, ``outputs_a[1] == inputs[0]`` (group A swap)
* ``outputs_b[1] == inputs[2]``, ``outputs_b[2] == inputs[1]`` (group B swap)
* Untouched slots (``outputs_a[2]``, ``outputs_b[0]``) remain zero.

If group B's window aliased group A's (the bug class this test rules out),
device 1's two dispatches would read each other's signal/scratch instead of
their respective peers' — either the goldens would fail or one of the
notify/wait handshakes would hang.

Needs 3 physical devices.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64


def _build_two_group_swap_program():
    """Build the 3-device overlapping-two-group peer-swap program at call time.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser at import time.
    """

    @pl.program
    class TwoGroupSwap:
        @pl.function(type=pl.FunctionType.InCore)
        def swap_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Phase 1: stage-in — local input → this rank's scratch slot.
            local = pl.load(inp, [0, 0], [1, SIZE])
            pl.store(local, [0, 0], scratch)

            # Phase 2: barrier — AtomicAdd peer's signal cell, wait on ours.
            pld.system.notify(
                signal,
                peer=peer,
                offsets=[0, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(
                signal=signal,
                offsets=[0, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

            # Phase 3: remote_load — read peer's scratch slice into out.
            recv = pld.tile.remote_load(scratch, peer=peer, offsets=[0, 0], shape=[1, SIZE])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.swap_step(inp, out, scratch, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[3, 1, SIZE], pl.FP32],
            outputs_a: pl.Out[pl.Tensor[[3, 1, SIZE], pl.FP32]],
            outputs_b: pl.Out[pl.Tensor[[3, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[3, 1, SIZE], pl.FP32]:
            # Per-group window buffers land in distinct CommGroups because
            # MaterializeCommDomainScopes clusters by dispatch device subset.
            scratch_a_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
            signal_a_buf = pld.alloc_window_buffer(pl.INT32.get_byte())
            scratch_b_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
            signal_b_buf = pld.alloc_window_buffer(pl.INT32.get_byte())

            # Group A: devices {0, 1}. Within-group rank == r, peer = 1 - r.
            for r in pl.range(2):
                scratch = pld.window(scratch_a_buf, [SIZE], dtype=pl.FP32)
                signal = pld.window(signal_a_buf, [1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs_a[r], scratch, signal, 1 - r, device=r)

            # Group B: devices {1, 2} — overlaps group A on device 1.
            # Within-group rank = r - 1, peer = 2 - r.
            for r in pl.range(1, 3):
                scratch = pld.window(scratch_b_buf, [SIZE], dtype=pl.FP32)
                signal = pld.window(signal_b_buf, [1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs_b[r], scratch, signal, 2 - r, device=r)
            return outputs_a

    return TwoGroupSwap


class TestL3MultiGroup:
    """L3 distributed runtime: two disjoint 2-rank CommGroups peer-swap."""

    def test_two_groups_independent_swap(self, test_config, device_ids):
        if len(device_ids) < 3:
            pytest.skip(f"overlapping two-group swap needs 3 devices, got {device_ids}")

        program = _build_two_group_swap_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:3],
                num_sub_workers=0,
            ),
        )

        # Per-device constant payloads — disjoint value ranges across devices
        # make any cross-group aliasing visible in the golden check.
        inputs = torch.stack(
            [
                torch.full((1, SIZE), 10.0, dtype=torch.float32),  # device 0
                torch.full((1, SIZE), 11.0, dtype=torch.float32),  # device 1
                torch.full((1, SIZE), 12.0, dtype=torch.float32),  # device 2
            ]
        )
        outputs_a = torch.zeros((3, 1, SIZE), dtype=torch.float32)
        outputs_b = torch.zeros((3, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs_a, outputs_b)

        # Group A swap on devices {0, 1}: dev r reads from dev (1 - r).
        expected_a = torch.stack(
            [
                inputs[1],  # outputs_a[0] = dev 1's input
                inputs[0],  # outputs_a[1] = dev 0's input
                torch.zeros((1, SIZE), dtype=torch.float32),  # outputs_a[2] untouched
            ]
        )
        # Group B swap on devices {1, 2}: dev r reads from dev (3 - r) globally.
        expected_b = torch.stack(
            [
                torch.zeros((1, SIZE), dtype=torch.float32),  # outputs_b[0] untouched
                inputs[2],  # outputs_b[1] = dev 2's input
                inputs[1],  # outputs_b[2] = dev 1's input
            ]
        )

        assert torch.allclose(outputs_a, expected_a), (
            f"group A swap mismatch: max diff = "
            f"{(outputs_a - expected_a).abs().max().item()}\n"
            f"outputs_a={outputs_a}\nexpected_a={expected_a}"
        )
        assert torch.allclose(outputs_b, expected_b), (
            f"group B swap mismatch: max diff = "
            f"{(outputs_b - expected_b).abs().max().item()}\n"
            f"outputs_b={outputs_b}\nexpected_b={expected_b}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
