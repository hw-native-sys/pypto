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

The pass enforces the ptoas data-before-signal contract with purely-local rules
(verified on ptoas 0.50); the ``notify`` itself needs no marker:

* **After every local publishing write** — a ``pl.store`` or scalar write into a
  window-bound destination, or a ``get`` into a window-bound local destination: a
  whole-tensor region ``pl.system.cacheinvalid(target, shape, [0, ...])``
  immediately followed by ``pl.system.fence()``.
* **After every remote publishing write** — ``remote_store`` / ``put``: only
  ``pl.system.fence()``.
* **After every opaque publishing write** — a ``Submit``, or a call to an
  unregistered user function: a whole-GM ``pl.system.cacheinvalid()`` followed by
  ``pl.system.fence()``.
* **After every wait**: a whole-GM ``pl.system.cacheinvalid()`` (no args).
* **Phase B (orchestration → InCore consumer)**: after an opaque collective
  dispatch in an orchestration / graph body **or** a HOST ``Opaque``
  ``Role.Orchestrator`` body (plain ``Call`` or ``Submit`` from
  ``pl.manual_scope`` / ``pl.submit``), prepend whole-GM
  ``pl.system.cacheinvalid()`` + ``pl.system.fence()`` at the separate InCore
  consumer's function entry (orchestration IR unchanged). One-hop wrappers
  still resolve after ``MaterializeRuntimeScopes`` wraps them in
  ``RuntimeScopeStmt``.

The **remote** writes land at a peer-offset address that a local-target
cacheinvalid cannot address, so the pass inserts only their release fence — the
peer-region cacheinvalid comes from their own codegen (see the codegen tests).
That asymmetry is what ``test_remote_store_gets_fence_only`` pins down.

Each test builds a ``Before`` program, runs the pass, and structurally compares
the result against a hand-written ``Expected``. The pass runs inside
``passes.PassContext([])`` so the autouse verification context is bypassed —
mirroring ``test_stamp_tfree_split.py``.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
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


def test_scalar_write_to_window_then_notify():
    # ConvertTensorToTileOps deliberately keeps `tensor.write` unconverted when its
    # destination is a DistributedTensor (codegen lowers it to `pto.store_scalar`),
    # so the pass must recognise it as a publishing write like `tile.store`.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            val: pl.Scalar[pl.FP32],
        ):
            pl.write(win, [0, 0], val)  # publishing: win is window-bound
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            val: pl.Scalar[pl.FP32],
        ):
            pl.write(win, [0, 0], val)
            pl.system.cacheinvalid(win, [1, N], [0, 0])
            pl.system.fence()
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_scalar_write_to_plain_tensor_is_not_published():
    # Same op, plain Tensor destination: no peer can remote_load it, so no marker.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            plain: pl.Tensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            val: pl.Scalar[pl.FP32],
        ):
            pl.write(plain, [0, 0], val)
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Before)


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


def test_pure_wait_loop_gets_single_whole_gm_cacheinvalid():
    # A wait-all loop (`for src: if src != me: wait(...)`, the mesh composite's
    # per-barrier wait loop) performs no memory access between the waits, so ONE
    # whole-GM cacheinvalid after the loop is equivalent to one after every wait
    # — and (P-1) whole-cache flushes become 1 per barrier generation.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            signal: pld.DistributedTensor[[1, N], pl.INT32],
            nranks: pl.Scalar[pl.INT32],
            me: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ):
            for src in pl.range(nranks):
                if src != me:
                    pld.system.wait(signal=signal, offsets=[0, src], expected=1, cmp=pld.WaitCmp.Ge)
            val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
            pl.write(out, [0, 0], val)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            signal: pld.DistributedTensor[[1, N], pl.INT32],
            nranks: pl.Scalar[pl.INT32],
            me: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ):
            for src in pl.range(nranks):
                if src != me:
                    pld.system.wait(signal=signal, offsets=[0, src], expected=1, cmp=pld.WaitCmp.Ge)
            pl.system.cacheinvalid()
            val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
            pl.write(out, [0, 0], val)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_consecutive_waits_share_one_whole_gm_cacheinvalid():
    # Two consecutive waits with no intervening memory access share ONE
    # whole-GM cacheinvalid after the run.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            signal: pld.DistributedTensor[[1, 2], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ):
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
            pld.system.wait(signal=signal, offsets=[0, 1], expected=1, cmp=pld.WaitCmp.Ge)
            val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
            pl.write(out, [0, 0], val)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            signal: pld.DistributedTensor[[1, 2], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ):
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
            pld.system.wait(signal=signal, offsets=[0, 1], expected=1, cmp=pld.WaitCmp.Ge)
            pl.system.cacheinvalid()
            val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
            pl.write(out, [0, 0], val)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_bare_pure_wait_loop_batches_to_single_invalidate():
    # A pure wait-loop as the SOLE body of the function (no enclosing SeqStmts)
    # must still batch: the per-wait invalidates inside the loop are suppressed
    # while visiting the bare body, and exactly ONE whole-GM cacheinvalid lands
    # after the loop — not one per wait plus one after.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            signal: pld.DistributedTensor[[1, N], pl.INT32],
            nranks: pl.Scalar[pl.INT32],
            me: pl.Scalar[pl.INT32],
        ):
            for src in pl.range(nranks):
                if src != me:
                    pld.system.wait(signal=signal, offsets=[0, src], expected=1, cmp=pld.WaitCmp.Ge)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            signal: pld.DistributedTensor[[1, N], pl.INT32],
            nranks: pl.Scalar[pl.INT32],
            me: pl.Scalar[pl.INT32],
        ):
            for src in pl.range(nranks):
                if src != me:
                    pld.system.wait(signal=signal, offsets=[0, src], expected=1, cmp=pld.WaitCmp.Ge)
            pl.system.cacheinvalid()

    ir.assert_structural_equal(_apply(Before), Expected)


def test_control_expression_read_prevents_batching():
    # A wait-loop whose control expression reads GM (an `if` condition on
    # `pl.read`, which lowers to a cached `tensor.read`) is NOT pure: deferring
    # the consume-side invalidate past the control read could let it observe
    # stale peer data. Each wait keeps its own whole-GM cacheinvalid.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            signal: pld.DistributedTensor[[1, N], pl.INT32],
            nranks: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ):
            for src in pl.range(nranks):
                if pl.read(signal, [0, src]) != 0:
                    pld.system.wait(signal=signal, offsets=[0, src], expected=1, cmp=pld.WaitCmp.Ge)
            val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
            pl.write(out, [0, 0], val)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            signal: pld.DistributedTensor[[1, N], pl.INT32],
            nranks: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ):
            for src in pl.range(nranks):
                if pl.read(signal, [0, src]) != 0:
                    pld.system.wait(signal=signal, offsets=[0, src], expected=1, cmp=pld.WaitCmp.Ge)
                    pl.system.cacheinvalid()
            val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
            pl.write(out, [0, 0], val)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_wait_free_loop_is_left_untouched():
    # A loop whose body has NO wait anywhere (here: empty) is not a pure
    # wait-loop. Before the tri-state fix the empty SeqStmts body was vacuously
    # "wait-only", so the pass appended a whole-GM cacheinvalid after it — a new
    # ENTIRE_DATA_CACHE flush where the pre-pass code emitted nothing, in a pass
    # whose point is removing flushes. It must be left untouched.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            nranks: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ):
            for _ in pl.range(nranks):
                pass
            val: pl.Scalar[pl.INT32] = pl.read(out, [0, 0])
            pl.write(out, [0, 0], val)

    ir.assert_structural_equal(_apply(Before), Before)


def test_wait_free_loop_as_bare_body_is_left_untouched():
    # Same guarantee when the wait-free loop is the SOLE body of the function
    # (no enclosing SeqStmts): the bare-body path must not add a whole-GM
    # cacheinvalid after it either.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            signal: pld.DistributedTensor[[1, N], pl.INT32],
            nranks: pl.Scalar[pl.INT32],
            me: pl.Scalar[pl.INT32],
        ):
            for _ in pl.range(nranks):
                pass

    ir.assert_structural_equal(_apply(Before), Before)


def test_if_with_empty_branches_is_left_untouched():
    # `if c: pass` has no wait either (the empty then-branch is vacuously
    # wait-only), so the loop must not be treated as a pure wait-loop.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            nranks: pl.Scalar[pl.INT32],
            cond: pl.Scalar[pl.BOOL],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ):
            for _ in pl.range(nranks):
                if cond:
                    pass
            val: pl.Scalar[pl.INT32] = pl.read(out, [0, 0])
            pl.write(out, [0, 0], val)

    ir.assert_structural_equal(_apply(Before), Before)


def test_if_with_wait_and_empty_else_is_still_pure():
    # `if c: wait else: <empty>` must stay a pure wait-loop: the empty else
    # classifies kPureNoWait and must NOT disqualify the loop (no memory access
    # occurs on that path either). ONE whole-GM cacheinvalid after the loop.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            signal: pld.DistributedTensor[[1, N], pl.INT32],
            nranks: pl.Scalar[pl.INT32],
            me: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ):
            for src in pl.range(nranks):
                if src != me:
                    pld.system.wait(signal=signal, offsets=[0, src], expected=1, cmp=pld.WaitCmp.Ge)
                else:
                    pass
            val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
            pl.write(out, [0, 0], val)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            signal: pld.DistributedTensor[[1, N], pl.INT32],
            nranks: pl.Scalar[pl.INT32],
            me: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ):
            for src in pl.range(nranks):
                if src != me:
                    pld.system.wait(signal=signal, offsets=[0, src], expected=1, cmp=pld.WaitCmp.Ge)
                else:
                    pass
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


def test_orch_collective_then_consume_prepends_incore_prologue():
    # An orchestration pipeline that dispatches an opaque collective kernel task
    # and then a separate InCore consumer must prepend whole-GM cacheinvalid +
    # fence to the consumer's entry — orchestration codegen cannot host InCore
    # system ops, so the markers land on the callee (same pattern as the manual
    # workaround formerly in the all_to_all_v ST consume_step).
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def collective_kernel(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            data: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            counts: pld.DistributedTensor[[1, 1], pl.INT32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
        ) -> pld.DistributedTensor[[1, N], pl.FP32]:
            pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            data: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            counts: pld.DistributedTensor[[1, 1], pl.INT32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            data = self.collective_kernel(inp, data, signal, counts, recv)
            return self.consume_step(recv, out)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def collective_kernel(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            data: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            counts: pld.DistributedTensor[[1, 1], pl.INT32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
        ) -> pld.DistributedTensor[[1, N], pl.FP32]:
            pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            pl.system.cacheinvalid()
            pl.system.fence()
            val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            data: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            counts: pld.DistributedTensor[[1, 1], pl.INT32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            data = self.collective_kernel(inp, data, signal, counts, recv)
            return self.consume_step(recv, out)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_host_builtin_tensor_collective_then_consume():
    # After LowerHostTensorCollectives, HOST pipelines dispatch ``builtin.tensor.*``
    # directly from orchestration. Phase B must treat that as an opaque publish
    # and prepend the consume prologue to the later InCore callee.
    #
    # Both sides use ``pl.parse_program`` — ``pl.builtin.*`` is a machine-only
    # printer surface with no typed DSL wrapper, so a ``@pl.program`` Expected
    # would fail pyright (reportAttributeAccessIssue on ``pl.builtin``).
    _builtin_call = """
        data = pl.builtin.tensor.all_to_all_v(
            inp,
            data,
            signal,
            counts,
            recv,
            dtype=pl.FP32,
            attrs={
                "device": 0,
                "dtype": pl.FP32,
                "arg_directions": [
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.input,
                    pl.adir.inout,
                ],
            },
        )
"""
    _program_src = """
import pypto.language as pl
import pypto.language.distributed as pld

N = 8

@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
    ) -> pl.Tensor[[1, 1], pl.INT32]:
{consume_body}
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def host_pipeline(
        self,
        inp: pld.DistributedTensor[[1, N], pl.FP32],
        data: pld.DistributedTensor[[1, N], pl.FP32],
        signal: pld.DistributedTensor[[1, 1], pl.INT32],
        counts: pld.DistributedTensor[[1, 1], pl.INT32],
        recv: pld.DistributedTensor[[1, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
    ) -> pl.Tensor[[1, 1], pl.INT32]:
{builtin_call}
        return self.consume_step(recv, out)
"""
    _consume_body = (
        "        val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])\n"
        "        pl.write(out, [0, 0], val)\n"
    )
    Before = pl.parse_program(
        _program_src.format(
            consume_body=_consume_body,
            builtin_call=_builtin_call,
        )
    )
    Expected = pl.parse_program(
        _program_src.format(
            consume_body=(
                "        pl.system.cacheinvalid()\n"
                "        pl.system.fence()\n"
                "        val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])\n"
                "        pl.write(out, [0, 0], val)\n"
            ),
            builtin_call=_builtin_call,
        )
    )

    ir.assert_structural_equal(_apply(Before), Expected)


def test_host_opaque_orchestrator_collective_then_consume():
    # Real L3 host_orch shape: ``@pl.function(level=HOST, role=Orchestrator)``
    # defaults to FunctionType.Opaque. Phase B must still scan it (same eligibility
    # as IsHostOrch in the HOST collective lowers) and mark the later InCore
    # consumer through the one-hop consume_orch wrapper.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def collective_kernel(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
        ) -> pld.DistributedTensor[[1, N], pl.FP32]:
            pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            return self.consume_step(recv, out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            data = self.collective_kernel(data)
            return self.consume_orch(recv, out)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def collective_kernel(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
        ) -> pld.DistributedTensor[[1, N], pl.FP32]:
            pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            pl.system.cacheinvalid()
            pl.system.fence()
            val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            return self.consume_step(recv, out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            data = self.collective_kernel(data)
            return self.consume_orch(recv, out)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_orch_consume_wrapper_unwraps_after_materialize_runtime_scopes():
    # Default pipeline runs MaterializeRuntimeScopes before InsertCommFence, so
    # consume_orch's body is wrapped in an AUTO RuntimeScopeStmt. Phase B's
    # one-hop unwrap must peel that scope or the prologue is silently skipped.
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    try:

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def collective_kernel(
                self,
                data: pld.DistributedTensor[[1, N], pl.FP32],
            ) -> pld.DistributedTensor[[1, N], pl.FP32]:
                pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
                return data

            @pl.function(type=pl.FunctionType.InCore)
            def consume_step(
                self,
                recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
                out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            ) -> pl.Tensor[[1, 1], pl.INT32]:
                val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
                pl.write(out, [0, 0], val)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def consume_orch(
                self,
                recv: pld.DistributedTensor[[1, 1], pl.INT32],
                out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            ) -> pl.Tensor[[1, 1], pl.INT32]:
                return self.consume_step(recv, out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def host_pipeline(
                self,
                data: pld.DistributedTensor[[1, N], pl.FP32],
                recv: pld.DistributedTensor[[1, 1], pl.INT32],
                out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            ) -> pl.Tensor[[1, 1], pl.INT32]:
                data = self.collective_kernel(data)
                return self.consume_orch(recv, out)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV)
            def collective_kernel(
                self,
                data: pld.DistributedTensor[[1, N], pl.FP32],
            ) -> pld.DistributedTensor[[1, N], pl.FP32]:
                pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
                return data

            @pl.function(type=pl.FunctionType.InCore)
            def consume_step(
                self,
                recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
                out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            ) -> pl.Tensor[[1, 1], pl.INT32]:
                pl.system.cacheinvalid()
                pl.system.fence()
                val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
                pl.write(out, [0, 0], val)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def consume_orch(
                self,
                recv: pld.DistributedTensor[[1, 1], pl.INT32],
                out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            ) -> pl.Tensor[[1, 1], pl.INT32]:
                return self.consume_step(recv, out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def host_pipeline(
                self,
                data: pld.DistributedTensor[[1, N], pl.FP32],
                recv: pld.DistributedTensor[[1, 1], pl.INT32],
                out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            ) -> pl.Tensor[[1, 1], pl.INT32]:
                data = self.collective_kernel(data)
                return self.consume_orch(recv, out)

        def _prepare(program):
            return passes.materialize_runtime_scopes()(passes.derive_call_directions()(program))

        ir.assert_structural_equal(_apply(_prepare(Before)), _prepare(Expected))
    finally:
        backend.reset_for_testing()


def test_orch_submit_collective_then_submit_consume():
    # Under pl.manual_scope, task launches lower to Submit (not Call). Phase B
    # must still mark the InCore consumer when both the opaque collective and
    # the later consume are dispatched via pl.submit — otherwise the prologue
    # is silently skipped (pass-submit-awareness).
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def collective_kernel(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
        ) -> pld.DistributedTensor[[1, N], pl.FP32]:
            pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            with pl.manual_scope():
                data, _pub_tid = pl.submit(self.collective_kernel, data)
                out, _con_tid = pl.submit(self.consume_step, recv, out)
            return out

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def collective_kernel(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
        ) -> pld.DistributedTensor[[1, N], pl.FP32]:
            pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            pl.system.cacheinvalid()
            pl.system.fence()
            val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            with pl.manual_scope():
                data, _pub_tid = pl.submit(self.collective_kernel, data)
                out, _con_tid = pl.submit(self.consume_step, recv, out)
            return out

    ir.assert_structural_equal(_apply(Before), Expected)


def test_orch_consume_wrapper_unwraps_to_incore():
    # HOST fan-out uses a one-hop orchestration wrapper (consume_orch) that
    # delegates to the real InCore consumer — phase B must mark consume_step,
    # not the wrapper.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def collective_kernel(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
        ) -> pld.DistributedTensor[[1, N], pl.FP32]:
            pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            return self.consume_step(recv, out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def host_pipeline(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            data = self.collective_kernel(data)
            return self.consume_orch(recv, out)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def collective_kernel(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
        ) -> pld.DistributedTensor[[1, N], pl.FP32]:
            pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            pl.system.cacheinvalid()
            pl.system.fence()
            val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            return self.consume_step(recv, out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def host_pipeline(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            data = self.collective_kernel(data)
            return self.consume_orch(recv, out)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_orch_stage_between_collective_and_consume():
    # Intermediate non-publish InCore helpers must not get the prologue; only
    # the post-collective consumer does.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def collective_kernel(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
        ) -> pld.DistributedTensor[[1, N], pl.FP32]:
            pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def stage_step(
            self,
            x: pl.Tensor[[1, N], pl.FP32],
            out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
        ) -> pl.Tensor[[1, N], pl.FP32]:
            local = pl.load(x, [0, 0], [1, N])
            pl.store(local, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            staged: pl.Tensor[[1, N], pl.FP32],
            data: pld.DistributedTensor[[1, N], pl.FP32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            stage_out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            staged = self.stage_step(staged, stage_out)
            data = self.collective_kernel(data)
            return self.consume_step(recv, out)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def collective_kernel(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
        ) -> pld.DistributedTensor[[1, N], pl.FP32]:
            pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def stage_step(
            self,
            x: pl.Tensor[[1, N], pl.FP32],
            out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
        ) -> pl.Tensor[[1, N], pl.FP32]:
            local = pl.load(x, [0, 0], [1, N])
            pl.store(local, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            pl.system.cacheinvalid()
            pl.system.fence()
            val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            staged: pl.Tensor[[1, N], pl.FP32],
            data: pld.DistributedTensor[[1, N], pl.FP32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            stage_out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            staged = self.stage_step(staged, stage_out)
            data = self.collective_kernel(data)
            return self.consume_step(recv, out)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_orch_consume_before_collective_unmarked():
    # Sequential scan: a consumer dispatched before any opaque publish must not
    # be marked (within the same orchestration body).
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def collective_kernel(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
        ) -> pld.DistributedTensor[[1, N], pl.FP32]:
            pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            out = self.consume_step(recv, out)
            data = self.collective_kernel(data)
            return out

    ir.assert_structural_equal(_apply(Before), Before)


def test_orch_collective_in_if_then_consume_after():
    # Conservative if/else merge: a collective in either branch keeps
    # seen_publish true for code after the if, so consume_step is marked even
    # when the collective ran only on one path at runtime.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def collective_kernel(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
        ) -> pld.DistributedTensor[[1, N], pl.FP32]:
            pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            run: pl.Scalar[pl.BOOL],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            if run:
                data = self.collective_kernel(data)
            else:
                pass
            return self.consume_step(recv, out)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def collective_kernel(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
        ) -> pld.DistributedTensor[[1, N], pl.FP32]:
            pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            pl.system.cacheinvalid()
            pl.system.fence()
            val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            run: pl.Scalar[pl.BOOL],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            if run:
                data = self.collective_kernel(data)
            else:
                pass
            return self.consume_step(recv, out)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_orch_collective_consume_prologue_idempotent():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def collective_kernel(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
        ) -> pld.DistributedTensor[[1, N], pl.FP32]:
            pl.func_attr({"builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v"})
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            recv_counts: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            val: pl.Scalar[pl.INT32] = pl.read(recv_counts, [0, 0])
            pl.write(out, [0, 0], val)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            data: pld.DistributedTensor[[1, N], pl.FP32],
            recv: pld.DistributedTensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[[1, 1], pl.INT32]:
            data = self.collective_kernel(data)
            return self.consume_step(recv, out)

    once = _apply(Before)
    twice = _apply(once)
    ir.assert_structural_equal(twice, once)


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
