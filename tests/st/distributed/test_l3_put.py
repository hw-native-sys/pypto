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

The non-atomic full-slice, row-offset, and atomic-add scenarios are enabled as
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
            src = pl.store(local, [0, 0], src)

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
            src_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
            dst_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pl.INT32.get_byte())  # 1×1 × INT32

            for r in pl.range(pld.world_size()):
                src = pld.window(src_buf, [1, SIZE], dtype=pl.FP32)
                dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                # Ring partner: rank r pushes to peer = (r + 1) % nranks.
                self.chip_orch(inputs[r], outputs[r], src, dst, signal, (r + 1) % pld.world_size(), device=r)
            return outputs

    return RingPut


# Pipeline (ping-pong) ring put: a window larger than the chunk so pto-isa TPUT
# slides the transfer through two staging tiles. [PIPE_ROWS, PIPE_COLS] with a
# [chunk_rows, chunk_cols] sub-tile gives a 2x2 chunk grid (>1 chunk) so the
# ping-pong path actually engages.
PIPE_ROWS = 8
PIPE_COLS = 128
PIPE_CHUNK_ROWS = 4
PIPE_CHUNK_COLS = 64


def _build_ring_put_pipeline_program():
    """Ring put that double-buffers the cross-rank write (``pipeline=True``)."""

    @pl.program
    class RingPutPipeline:
        @pl.function(type=pl.FunctionType.InCore)
        def ring_step(
            self,
            inp: pl.Tensor[[PIPE_ROWS, PIPE_COLS], pl.FP32],
            out: pl.Out[pl.Tensor[[PIPE_ROWS, PIPE_COLS], pl.FP32]],
            src: pld.DistributedTensor[[PIPE_ROWS, PIPE_COLS], pl.FP32],
            dst: pld.DistributedTensor[[PIPE_ROWS, PIPE_COLS], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[PIPE_ROWS, PIPE_COLS], pl.FP32]:
            local = pl.load(inp, [0, 0], [PIPE_ROWS, PIPE_COLS])
            src = pl.store(local, [0, 0], src)

            # Double-buffered TPUT: pto-isa ping-pongs the [8, 128] transfer
            # through two [4, 64] staging tiles.
            pld.tensor.put(
                dst,
                peer=peer,
                src=src,
                atomic=pld.AtomicType.None_,
                chunk_rows=PIPE_CHUNK_ROWS,
                chunk_cols=PIPE_CHUNK_COLS,
                pipeline=True,
            )

            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

            recv = pl.load(dst, [0, 0], [PIPE_ROWS, PIPE_COLS])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[PIPE_ROWS, PIPE_COLS], pl.FP32],
            out: pl.Out[pl.Tensor[[PIPE_ROWS, PIPE_COLS], pl.FP32]],
            src: pld.DistributedTensor[[PIPE_ROWS, PIPE_COLS], pl.FP32],
            dst: pld.DistributedTensor[[PIPE_ROWS, PIPE_COLS], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[PIPE_ROWS, PIPE_COLS], pl.FP32]:
            return self.ring_step(inp, out, src, dst, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, PIPE_ROWS, PIPE_COLS], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, PIPE_ROWS, PIPE_COLS], pl.FP32]],
        ) -> pl.Tensor[[2, PIPE_ROWS, PIPE_COLS], pl.FP32]:
            nbytes = PIPE_ROWS * PIPE_COLS * pl.FP32.get_byte()
            src_buf = pld.alloc_window_buffer(nbytes)
            dst_buf = pld.alloc_window_buffer(nbytes)
            signal_buf = pld.alloc_window_buffer(pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                src = pld.window(src_buf, [PIPE_ROWS, PIPE_COLS], dtype=pl.FP32)
                dst = pld.window(dst_buf, [PIPE_ROWS, PIPE_COLS], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], src, dst, signal, (r + 1) % pld.world_size(), device=r)
            return outputs

    return RingPutPipeline


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
        ) -> pl.Tensor[[16, 16], pl.INT32]:
            ctx = pld.get_comm_ctx(acc)
            my_rank = pld.rank(ctx)
            nranks = pld.nranks(ctx)

            # Phase 1: stage our contribution into our own src cell.
            local = pl.load(inp, [0, 0], [16, 16])
            src = pl.store(local, [0, 0], src)

            # Phase 2: atomically add our contribution into the root rank's acc
            # cell. All ranks target the same peer (root) concurrently — the
            # device-side atomic_add is what makes this correct.
            pld.tensor.put(acc, peer=root, src=src, atomic=pld.AtomicType.Add)

            # Phase 3: every rank bumps the root's signal; only the root waits
            # on that root-local cell until all contributions, including its
            # own, have landed. Non-root ranks must not wait on their local
            # signal cells because nobody increments them in this reduce-to-root
            # protocol.
            pld.system.notify(
                target=signal,
                peer=root,
                offsets=[0, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )

            if my_rank == root:
                pld.system.wait(
                    signal=signal,
                    offsets=[0, 0],
                    expected=nranks,
                    cmp=pld.WaitCmp.Ge,
                )

                # Phase 4: the root reads the accumulated sum from its acc cell.
                recv = pl.load(acc, [0, 0], [16, 16])
                return pl.store(recv, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[16, 16], pl.INT32],
            out: pl.Out[pl.Tensor[[16, 16], pl.INT32]],
            src: pld.DistributedTensor[[16, 16], pl.INT32],
            acc: pld.DistributedTensor[[16, 16], pl.INT32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            root: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[16, 16], pl.INT32]:
            return self.add_step(inp, out, src, acc, signal, root)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, 16, 16], pl.INT32],
            outputs: pl.Out[pl.Tensor[[2, 16, 16], pl.INT32]],
        ) -> pl.Tensor[[2, 16, 16], pl.INT32]:
            src_buf = pld.alloc_window_buffer(16 * 16 * pl.INT32.get_byte())
            acc_buf = pld.alloc_window_buffer(16 * 16 * pl.INT32.get_byte())
            signal_buf = pld.alloc_window_buffer(pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                src = pld.window(src_buf, [16, 16], dtype=pl.INT32)
                acc = pld.window(acc_buf, [16, 16], dtype=pl.INT32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                # All ranks accumulate into root rank 0.
                self.chip_orch(inputs[r], outputs[r], src, acc, signal, 0, device=r)
            return outputs

    return AtomicAddReduce


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

    def test_ring_shuffle_pipeline(self, test_config, device_ids):
        """Double-buffered overwrite: same ring shuffle, but the TPUT ping-pongs the
        [8, 128] transfer through two [4, 64] staging tiles (``pipeline=True``)."""
        if len(device_ids) < 2:
            pytest.skip(f"ring put needs 2 devices, got {device_ids}")

        program = _build_ring_put_pipeline_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        # rank 0 holds 0..N-1; rank 1 holds 1000..1000+N-1. After the ring push,
        # rank r's slice is overwritten by (r - 1) % nranks.
        n = PIPE_ROWS * PIPE_COLS
        inputs = torch.stack(
            [
                torch.arange(n, dtype=torch.float32).reshape(PIPE_ROWS, PIPE_COLS),
                torch.arange(1000.0, 1000.0 + n, dtype=torch.float32).reshape(PIPE_ROWS, PIPE_COLS),
            ]
        )
        outputs = torch.zeros((2, PIPE_ROWS, PIPE_COLS), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = torch.stack([inputs[1], inputs[0]])
        assert torch.allclose(outputs, expected), (
            f"pipeline ring put mismatch: max diff = {(outputs - expected).abs().max().item()}"
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


def _build_row_put_program():
    """Build a row-offset put program."""

    @pl.program
    class RowPut:
        @pl.function(type=pl.FunctionType.InCore)
        def row_step(
            self,
            inp: pl.Tensor[[2, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            src: pld.DistributedTensor[[2, SIZE], pl.FP32],
            dst: pld.DistributedTensor[[2, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            local = pl.load(inp, [0, 0], [2, SIZE])
            src = pl.store(local, [0, 0], src)

            pld.tensor.put(
                dst,
                peer=peer,
                src=src,
                dst_offsets=[1, 0],
                src_offsets=[0, 0],
                shape=[1, SIZE],
                atomic=pld.AtomicType.None_,
            )
            pld.system.notify(signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

            recv = pl.load(dst, [1, 0], [1, SIZE])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[2, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            src: pld.DistributedTensor[[2, SIZE], pl.FP32],
            dst: pld.DistributedTensor[[2, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.row_step(inp, out, src, dst, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, 2, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, 1, SIZE], pl.FP32]:
            src_buf = pld.alloc_window_buffer(2 * SIZE * pl.FP32.get_byte())
            dst_buf = pld.alloc_window_buffer(2 * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                src = pld.window(src_buf, [2, SIZE], dtype=pl.FP32)
                dst = pld.window(dst_buf, [2, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], src, dst, signal, (r + 1) % pld.world_size(), device=r)
            return outputs

    return RowPut


class TestL3PutSubregion:
    """L3 distributed runtime: row-offset cross-rank write via pld.tensor.put."""

    def test_row_put(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"row put needs 2 devices, got {device_ids}")

        program = _build_row_put_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        inputs = torch.stack(
            [
                torch.stack(
                    [
                        torch.arange(SIZE, dtype=torch.float32),
                        torch.full((SIZE,), -1.0, dtype=torch.float32),
                    ]
                ),
                torch.stack(
                    [
                        torch.arange(100.0, 100.0 + SIZE, dtype=torch.float32),
                        torch.full((SIZE,), -2.0, dtype=torch.float32),
                    ]
                ),
            ]
        )
        outputs = torch.zeros((2, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = torch.stack([inputs[1, 0].reshape(1, SIZE), inputs[0, 0].reshape(1, SIZE)])
        assert torch.allclose(outputs, expected), (
            f"row put mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


CHUNK_ROWS = 8
CHUNK_COLS = 64


def _build_chunked_ring_put_program():
    """Build a ring-put program whose staging tile is smaller than the transfer.

    The ``[CHUNK_ROWS, CHUNK_COLS]`` window is pushed with ``chunk_rows=2,
    chunk_cols=32`` so ``ConvertTensorToTileOps`` allocates a ``[2, 32]`` VEC
    staging tile while the ``pto.comm.tput`` partition views keep the full
    ``[8, 64]`` extent. pto-isa TPUT must therefore 2-D-slide the full transfer
    through the sub-tile (4 row-chunks × 2 col-chunks). The end-to-end result
    must equal the unchunked ring shuffle — this is the device-level proof that
    the auto-chunk path is correct, not just that the codegen is well-formed.
    """

    @pl.program
    class ChunkedRingPut:
        @pl.function(type=pl.FunctionType.InCore)
        def ring_step(
            self,
            inp: pl.Tensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32],
            out: pl.Out[pl.Tensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32]],
            src: pld.DistributedTensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32],
            dst: pld.DistributedTensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32]:
            # Phase 1: stage-in — local input → this rank's own src window slice.
            local = pl.load(inp, [0, 0], [CHUNK_ROWS, CHUNK_COLS])
            src = pl.store(local, [0, 0], src)

            # Phase 2: push our full src into the peer's dst slice, but force the
            # VEC staging tile down to [2, 32] so TPUT auto-chunks the transfer.
            pld.tensor.put(
                dst,
                peer=peer,
                src=src,
                atomic=pld.AtomicType.None_,
                chunk_rows=2,
                chunk_cols=32,
            )

            # Phase 3: handshake — signal the peer, wait for the rank targeting us.
            pld.system.notify(signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

            # Phase 4: read our own dst slice back — written by the rank whose peer is us.
            recv = pl.load(dst, [0, 0], [CHUNK_ROWS, CHUNK_COLS])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32],
            out: pl.Out[pl.Tensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32]],
            src: pld.DistributedTensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32],
            dst: pld.DistributedTensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[CHUNK_ROWS, CHUNK_COLS], pl.FP32]:
            return self.ring_step(inp, out, src, dst, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, CHUNK_ROWS, CHUNK_COLS], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, CHUNK_ROWS, CHUNK_COLS], pl.FP32]],
        ) -> pl.Tensor[[2, CHUNK_ROWS, CHUNK_COLS], pl.FP32]:
            src_buf = pld.alloc_window_buffer(CHUNK_ROWS * CHUNK_COLS * pl.FP32.get_byte())
            dst_buf = pld.alloc_window_buffer(CHUNK_ROWS * CHUNK_COLS * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                src = pld.window(src_buf, [CHUNK_ROWS, CHUNK_COLS], dtype=pl.FP32)
                dst = pld.window(dst_buf, [CHUNK_ROWS, CHUNK_COLS], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], src, dst, signal, (r + 1) % pld.world_size(), device=r)
            return outputs

    return ChunkedRingPut


class TestL3PutChunked:
    """L3 distributed runtime: cross-rank write whose staging tile is sub-tiled,
    exercising pto-isa TPUT's auto-chunk (2-D sliding) path on device."""

    def test_chunked_ring_shuffle(self, test_config, device_ids):
        """A sub-tile staging buffer must still move the full transfer correctly."""
        if len(device_ids) < 2:
            pytest.skip(f"chunked ring put needs 2 devices, got {device_ids}")

        program = _build_chunked_ring_put_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        # rank 0 holds 0..511; rank 1 holds 1000..1511 (reshaped to [8, 64]).
        n = CHUNK_ROWS * CHUNK_COLS
        inputs = torch.stack(
            [
                torch.arange(n, dtype=torch.float32).reshape(CHUNK_ROWS, CHUNK_COLS),
                torch.arange(1000.0, 1000.0 + n, dtype=torch.float32).reshape(CHUNK_ROWS, CHUNK_COLS),
            ]
        )
        outputs = torch.zeros((2, CHUNK_ROWS, CHUNK_COLS), dtype=torch.float32)

        compiled(inputs, outputs)

        # Same ring contract as the unchunked case: outputs[r] = inputs[(r - 1) % nranks].
        expected = torch.stack([inputs[1], inputs[0]])
        assert torch.allclose(outputs, expected), (
            f"chunked ring put mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
