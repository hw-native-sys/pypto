# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: ``pld.tensor.put`` cross-rank write (TPUT).

End-to-end exercise of the N6 cross-rank bulk write primitive. ``put`` is the
*push* dual of ``pld.tile.remote_load`` (*pull*): rank ``r`` writes its own
window slice into a **peer**'s slice instead of reading the peer's slice.

Two scenarios are covered:

* :meth:`test_ring_shuffle` — ``atomic=AtomicType.None_`` overwrite. Rank ``r``
  pushes its input to ``peer = (r + 1) % nranks``, so each rank's slice is
  overwritten by ``(r - 1) % nranks``. Golden:
  ``outputs[r] == inputs[(r - 1) % nranks]``.
* :meth:`test_atomic_add_accumulate` — ``atomic=AtomicType.Add``. Every rank
  adds its scalar contribution into rank 0's single cell concurrently; the
  golden is the *sum* over all ranks. This is the case the device-side atomic
  combine exists for — a plain store would race and drop contributions.

Both use the ``pld.system.notify`` / ``pld.system.wait`` handshake as the
barrier that orders the synchronous put against the local read-back (see
:file:`test_l3_notify_wait.py` for the handshake contract in isolation).

The tests are currently **skipped** — the InCore PTO codegen for
``pld.tensor.put`` (plus notify/wait) is in place (N6 P1), but the host-side
glue still has open work, identical to the gap blocking
:file:`test_l3_remote_load.py` / :file:`test_l3_notify_wait.py`:

* ``tile.store(tile, offsets, dst)`` verifier must accept a
  ``DistributedTensorType`` ``dst`` so the Phase-1 stage-in works.
* **N7** distributed_codegen.cpp must emit one
  ``chip_args.add_scalar(ctx.device_ctx[group_idx])`` per
  ``DistributedTensor`` formal parameter (in IR-parameter order), plus the
  ``ContinuousTensor.make(..., child_memory=True)`` wrapper for each
  ``DistributedTensor`` arg.
* **N8** distributed_codegen must thread ``HostBufferStaging`` onto the
  ``orch.allocate_domain(...)`` block for the inferred CommGroup so the
  runtime knows which physical buffer to bind to each rank's window slot.

Drop ``pytest.mark.skip`` (and inline the ``@pl.program`` decorator at module
top-level) once the above land — the programs below and the golden checks are
the canonical e2e contract for ``pld.tensor.put``.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64


def _build_ring_put_program():
    """Build the overwrite (non-atomic) ring-put program at call time.

    Deferred construction lets this file collect even when the parser rejects
    the embedded body (e.g. the Phase-1 ``pl.store`` into a ``DistributedTensor``
    is not yet accepted by ``tile.store``'s verifier). The skip marker on the
    test class ensures the body never runs until the pending host-side work lands.
    """

    @pl.program
    class RingPut:
        @pl.function(type=pl.FunctionType.InCore)
        def ring_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            src: pld.DistributedTensor[[1, SIZE], pl.FP32],
            dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Phase 1: stage-in — local input → this rank's own src window slice.
            local = pl.load(inp, [0, 0], [1, SIZE])
            _ = pl.store(local, [0, 0], src)

            # Phase 2: push our src into the peer's dst slice (synchronous TPUT,
            # plain overwrite). After this returns, the peer's dst holds our input.
            pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.None_)

            # Phase 3: signal the peer that our write to it has landed, then wait
            # for the rank that targets us ((r - 1) % nranks) to have done the same.
            pld.system.notify(
                target=signal,
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

            # Phase 4: read our own dst slice back locally — it was written by the
            # rank whose peer is us — and surface it as the output.
            recv = pl.load(dst, [0, 0], [1, SIZE])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            src: pld.DistributedTensor[[1, SIZE], pl.FP32],
            dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.ring_step(inp, out, src, dst, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, 1, SIZE], pl.FP32]:
            src_buf = pld.alloc_window_buffer(SIZE * 4)  # 1×SIZE × FP32 (4 bytes)
            dst_buf = pld.alloc_window_buffer(SIZE * 4)
            signal_buf = pld.alloc_window_buffer(4)  # 1×1 × INT32

            for r in pl.range(pld.world_size()):
                src = pld.window(src_buf, [1, SIZE], dtype=pl.FP32)
                dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                # Ring partner: rank r pushes to peer = (r + 1) % nranks.
                self.chip_orch(inputs[r], outputs[r], src, dst, signal, (r + 1) % pld.world_size(), device=r)
            return outputs

    return RingPut


def _build_atomic_add_program():
    """Build the ``atomic=Add`` accumulation program at call time.

    Every rank adds its scalar contribution into rank 0's single dst cell. The
    device-side atomic combine must make these concurrent writes commute; the
    golden is the sum over all ranks.
    """

    @pl.program
    class AtomicAddReduce:
        @pl.function(type=pl.FunctionType.InCore)
        def add_step(
            self,
            inp: pl.Tensor[[16, 16], pl.INT32],
            out: pl.Out[pl.Tensor[[16, 16], pl.INT32]],
            src: pld.DistributedTensor[[16, 16], pl.INT32],
            acc: pld.DistributedTensor[[16, 16], pl.INT32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            root: pl.Scalar[pl.INT32],
            nranks: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[16, 16], pl.INT32]:
            # Phase 1: stage our contribution into our own src cell.
            local = pl.load(inp, [0, 0], [16, 16])
            _ = pl.store(local, [0, 0], src)

            # Phase 2: atomically add our contribution into the root rank's acc
            # cell. All ranks target the same peer (root) concurrently — the
            # device-side atomic_add is what makes this correct.
            pld.tensor.put(acc, peer=root, src=src, atomic=pld.AtomicType.Add)

            # Phase 3: every rank bumps the root's signal; the root waits until
            # all nranks contributions (including its own) have landed.
            pld.system.notify(
                target=signal,
                peer=root,
                offsets=[0, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(
                signal=signal,
                offsets=[0, 0],
                expected=nranks,
                cmp=pld.WaitCmp.Ge,
            )

            # Phase 4: the root reads the accumulated sum from its acc cell.
            recv = pl.load(acc, [0, 0], [16, 16])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[16, 16], pl.INT32],
            out: pl.Out[pl.Tensor[[16, 16], pl.INT32]],
            src: pld.DistributedTensor[[16, 16], pl.INT32],
            acc: pld.DistributedTensor[[16, 16], pl.INT32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            root: pl.Scalar[pl.INT32],
            nranks: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[16, 16], pl.INT32]:
            return self.add_step(inp, out, src, acc, signal, root, nranks)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, 16, 16], pl.INT32],
            outputs: pl.Out[pl.Tensor[[2, 16, 16], pl.INT32]],
        ) -> pl.Tensor[[2, 16, 16], pl.INT32]:
            src_buf = pld.alloc_window_buffer(16 * 16 * 4)  # 16×16 × INT32
            acc_buf = pld.alloc_window_buffer(16 * 16 * 4)
            signal_buf = pld.alloc_window_buffer(4)

            for r in pl.range(pld.world_size()):
                src = pld.window(src_buf, [16, 16], dtype=pl.INT32)
                acc = pld.window(acc_buf, [16, 16], dtype=pl.INT32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                # All ranks accumulate into root rank 0.
                self.chip_orch(inputs[r], outputs[r], src, acc, signal, 0, pld.world_size(), device=r)
            return outputs

    return AtomicAddReduce


@pytest.mark.skip(
    reason=(
        "PTOAS drops the synchronisation between the stage-in tile.store and the "
        "subsequent pld.tensor.put -- the put can issue before the local window "
        "slice has been written, so the peer reads stale data. Re-enable once "
        "PTOAS treats the store -> put pair as an ordered dependency."
    )
)
class TestL3Put:
    """L3 distributed runtime: cross-rank write via pld.tensor.put."""

    def test_ring_shuffle(self, test_config, device_ids):
        """Non-atomic overwrite: rank r pushes its input to peer (r + 1) % nranks."""
        if len(device_ids) < 2:
            pytest.skip(f"ring put needs 2 devices, got {device_ids}")

        program = _build_ring_put_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        # rank 0 holds [0, 1, …, SIZE-1]; rank 1 holds [100, 101, …]. After the
        # ring push, rank r's slice is overwritten by (r - 1) % nranks.
        inputs = torch.stack(
            [
                torch.arange(SIZE, dtype=torch.float32).reshape(1, SIZE),
                torch.arange(100.0, 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE),
            ]
        )
        outputs = torch.zeros((2, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        # outputs[r] = inputs[(r - 1) % nranks] → outputs[0] = inputs[1], etc.
        expected = torch.stack([inputs[1], inputs[0]])
        assert torch.allclose(outputs, expected), (
            f"ring put mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    def test_atomic_add_accumulate(self, test_config, device_ids):
        """Atomic add: all ranks accumulate into root rank 0's single cell."""
        if len(device_ids) < 2:
            pytest.skip(f"atomic-add accumulate needs 2 devices, got {device_ids}")

        program = _build_atomic_add_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        # rank 0 contributes 10, rank 1 contributes 32 (broadcast across the
        # [16, 16] tile). The root (rank 0) reads the atomic element-wise sum
        # 42; non-root ranks read whatever their acc cell holds (unchecked —
        # only the root's output is meaningful).
        inputs = torch.stack(
            [
                torch.full((16, 16), 10, dtype=torch.int32),
                torch.full((16, 16), 32, dtype=torch.int32),
            ]
        )
        outputs = torch.zeros((2, 16, 16), dtype=torch.int32)

        compiled(inputs, outputs)

        expected_root = torch.full((16, 16), 42, dtype=torch.int32)
        assert torch.equal(outputs[0], expected_root), (
            f"atomic_add reduce mismatch: root max diff = {(outputs[0] - expected_root).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
