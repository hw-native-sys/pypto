# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Unit tests for the ``InsertCommFence`` pass, as Before/Expected structural comparisons.

The pass enforces the ptoas data-before-signal contract with two purely-local
rules (verified on ptoas 0.50); the ``notify`` itself needs no marker:

* **After every local publishing write** — a window-bound ``pl.store`` (or ``get``
  into a local destination): a whole-tensor region
  ``pl.system.cacheinvalid(target, shape, [0, ...])`` immediately followed by
  ``pl.system.fence()``.
* **After every wait**: a whole-GM ``pl.system.cacheinvalid()`` (no args).

The **remote** writes ``remote_store`` / ``put`` land at a peer-offset address and
are left untouched by the pass — their codegen emits a correct peer-region
cacheinvalid + fence itself (see the codegen tests). So a program whose only
publishing write is a ``remote_store`` is returned unchanged by this pass.

Each test builds a ``Before`` program, runs the pass, and structurally compares
the result against a hand-written ``Expected``. The pass runs inside
``passes.PassContext([])`` so the autouse verification context is bypassed —
mirroring ``test_stamp_tfree_split.py``.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import ir
from pypto.pypto_core import passes

N = 8


def _apply(program):
    """Run insert_comm_fence with verification disabled."""
    with passes.PassContext([]):
        return passes.insert_comm_fence()(program)


def test_window_store_then_notify():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pl.store(local, [0, 0], win)  # publishing: win is window-bound
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pl.store(local, [0, 0], win)
            pl.system.cacheinvalid(win, [1, N], [0, 0])
            pl.system.fence()
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_remote_store_gets_fence_only():
    # A remote_store lands at a peer-offset address the pass can't invalidate, so
    # the pass inserts only the GM `system.fence` (the peer-region cacheinvalid is
    # emitted by codegen — the peer offset is not yet IR-expressible).
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            dst: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pld.tile.remote_store(local, target=dst, peer=peer, offsets=[0, 0])
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            dst: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pld.tile.remote_store(local, target=dst, peer=peer, offsets=[0, 0])
            pl.system.fence()
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_plain_tensor_store_no_markers():
    # A plain (non-window) store is not a publishing write; nothing is inserted.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            outp: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pl.store(local, [0, 0], outp)  # plain tensor — not published to a peer
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Before)


def test_multiple_window_stores_each_get_cacheinvalid_and_fence():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pl.store(local, [0, 0], win)
            pl.store(local, [0, 0], win)
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pl.store(local, [0, 0], win)
            pl.system.cacheinvalid(win, [1, N], [0, 0])
            pl.system.fence()
            pl.store(local, [0, 0], win)
            pl.system.cacheinvalid(win, [1, N], [0, 0])
            pl.system.fence()
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_two_distinct_windows_each_gets_cacheinvalid():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win_a: pld.DistributedTensor[[1, N], pl.FP32],
            win_b: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pl.store(local, [0, 0], win_a)
            pl.store(local, [0, 0], win_b)
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win_a: pld.DistributedTensor[[1, N], pl.FP32],
            win_b: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pl.store(local, [0, 0], win_a)
            pl.system.cacheinvalid(win_a, [1, N], [0, 0])
            pl.system.fence()
            pl.store(local, [0, 0], win_b)
            pl.system.cacheinvalid(win_b, [1, N], [0, 0])
            pl.system.fence()
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_write_inside_if_branch():
    # The window store lives inside the branch; its cacheinvalid + fence are
    # emitted right after it, in the branch. The outer notify needs no marker.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            cond: pl.Scalar[pl.BOOL],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            if cond:
                pl.store(local, [0, 0], win)
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            cond: pl.Scalar[pl.BOOL],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            if cond:
                pl.store(local, [0, 0], win)
                pl.system.cacheinvalid(win, [1, N], [0, 0])
                pl.system.fence()
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_notify_inside_if_write_before():
    # The write is before the if; its cacheinvalid + fence release the data for the
    # conditional notify, which needs no marker of its own.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            cond: pl.Scalar[pl.BOOL],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pl.store(local, [0, 0], win)
            if cond:
                pld.system.notify(
                    target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            cond: pl.Scalar[pl.BOOL],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pl.store(local, [0, 0], win)
            pl.system.cacheinvalid(win, [1, N], [0, 0])
            pl.system.fence()
            if cond:
                pld.system.notify(
                    target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )

    ir.assert_structural_equal(_apply(Before), Expected)


def test_notify_inside_loop_after_write():
    # The pre-loop write's cacheinvalid + fence releases the data for the loop's
    # notify — even across the loop boundary. The notify gets nothing.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pl.store(local, [0, 0], win)
            for i in pl.range(N):
                pld.system.notify(target=signal, peer=i, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pl.store(local, [0, 0], win)
            pl.system.cacheinvalid(win, [1, N], [0, 0])
            pl.system.fence()
            for i in pl.range(N):
                pld.system.notify(target=signal, peer=i, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_loop_back_edge_notify_then_write():
    # for { notify; store } — the tail store gets its cacheinvalid + fence; that
    # fence (previous / final iteration) covers the notify.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            for i in pl.range(N):
                pld.system.notify(
                    target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )
                pl.store(local, [0, 0], win)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            for i in pl.range(N):
                pld.system.notify(
                    target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )
                pl.store(local, [0, 0], win)
                pl.system.cacheinvalid(win, [1, N], [0, 0])
                pl.system.fence()

    ir.assert_structural_equal(_apply(Before), Expected)


def test_combo_ring_barrier_idiom():
    # for s: { for p: (if p != me: notify); store } — the ring-allreduce barrier.
    # Only the tail store gets a marker; the conditional barrier notify gets none.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            me: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            for _s in pl.range(N - 1):
                for p in pl.range(N):
                    if p != me:
                        pld.system.notify(
                            target=signal, peer=p, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                        )
                pl.store(local, [0, 0], win)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            me: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            for _s in pl.range(N - 1):
                for p in pl.range(N):
                    if p != me:
                        pld.system.notify(
                            target=signal, peer=p, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                        )
                pl.store(local, [0, 0], win)
                pl.system.cacheinvalid(win, [1, N], [0, 0])
                pl.system.fence()

    ir.assert_structural_equal(_apply(Before), Expected)


def test_combo_two_phase_loops():
    # for { notify; store }; for { notify; store } — reduce-scatter then allgather.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            for _s in pl.range(N):
                pld.system.notify(
                    target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )
                pl.store(local, [0, 0], win)
            for _t in pl.range(N):
                pld.system.notify(
                    target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )
                pl.store(local, [0, 0], win)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            for _s in pl.range(N):
                pld.system.notify(
                    target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )
                pl.store(local, [0, 0], win)
                pl.system.cacheinvalid(win, [1, N], [0, 0])
                pl.system.fence()
            for _t in pl.range(N):
                pld.system.notify(
                    target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )
                pl.store(local, [0, 0], win)
                pl.system.cacheinvalid(win, [1, N], [0, 0])
                pl.system.fence()

    ir.assert_structural_equal(_apply(Before), Expected)


def test_wait_then_read_inserts_whole_gm_cacheinvalid():
    # Consume side: a whole-GM cacheinvalid right after the wait, before the read.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
        ):
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
            val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
            pl.write(out, [0, 0], val)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
        ):
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
            pl.system.cacheinvalid()
            val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
            pl.write(out, [0, 0], val)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_notify_wait_read_handshake():
    # notify; wait; read — the notify needs nothing; only the wait gets a whole-GM
    # cacheinvalid before the read.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.Set)
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
            val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
            pl.write(out, [0, 0], val)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.Set)
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
            pl.system.cacheinvalid()
            val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
            pl.write(out, [0, 0], val)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_bare_barrier_notify_no_marker():
    # A pure barrier notify (no data) needs nothing.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Before)


def test_orchestration_function_untouched():
    # The data-before-signal contract is InCore-only. An Orchestration function
    # dispatches tasks via cross-function calls; those are not GM publishing
    # writes, and inserting an InCore system.cacheinvalid there is rejected by
    # orchestration codegen. The pass must leave such functions unchanged.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def worker(self, x: pl.Tensor[[1, N], pl.FP32], out: pl.Out[pl.Tensor[[1, N], pl.FP32]]):
            local = pl.load(x, [0, 0], [1, N])
            pl.store(local, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, x: pl.Tensor[[1, N], pl.FP32], out: pl.Out[pl.Tensor[[1, N], pl.FP32]]):
            self.worker(x, out)

    After = _apply(Before)
    # The Orchestration `orch` must be byte-for-byte unchanged (no markers); the
    # InCore `worker` (a plain store, not window-bound) is also unchanged.
    ir.assert_structural_equal(After, Before)


def test_opaque_cross_function_call_gets_whole_gm_marker():
    # A call to a user function is an opaque publishing write: its body is not
    # analysed here and it has no single addressable region, so the pass emits a
    # conservative whole-GM `cacheinvalid()` + `fence()` after it.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def helper(self, x: pl.Tensor[[1, N], pl.FP32], out: pl.Out[pl.Tensor[[1, N], pl.FP32]]):
            local = pl.load(x, [0, 0], [1, N])
            pl.store(local, [0, 0], out)

        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            outp: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            self.helper(inp, outp)
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def helper(self, x: pl.Tensor[[1, N], pl.FP32], out: pl.Out[pl.Tensor[[1, N], pl.FP32]]):
            local = pl.load(x, [0, 0], [1, N])
            pl.store(local, [0, 0], out)

        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            outp: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            self.helper(inp, outp)
            pl.system.cacheinvalid()
            pl.system.fence()
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_idempotent():
    # Re-running the pass on already-marked IR inserts nothing.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pl.store(local, [0, 0], win)
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    once = _apply(Before)
    twice = _apply(once)
    ir.assert_structural_equal(twice, once)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
