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

* **After every publishing write** (remote_store / put / get, or a local store
  into a window-bound ``DistributedTensor``): a whole-tensor region
  ``pl.system.cacheinvalid(target, shape, [0, ...])`` immediately followed by
  ``pl.system.fence()``. ptoas ties the release fence to this cacheinvalid, so a
  later notify — even in a different loop — is satisfied without its own marker.
* **After every wait**: a whole-GM ``pl.system.cacheinvalid()`` (no args) — the
  consume-side invalidate before the next cacheable read.

Each test builds a ``Before`` program, runs the pass, and structurally compares
the result against a hand-written ``Expected`` (the same body with the markers
inserted). The pass runs inside ``passes.PassContext([])`` so the autouse
verification context is bypassed — mirroring ``test_stamp_tfree_split.py``.
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


def test_remote_store_then_notify():
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
            pl.system.cacheinvalid(dst, [1, N], [0, 0])
            pl.system.fence()
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_local_store_into_window_then_notify():
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


def test_local_store_into_plain_tensor_no_markers():
    # A plain store is not a publishing write, and the notify needs no marker.
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

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            outp: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pl.store(local, [0, 0], outp)
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_multiple_writes_each_get_cacheinvalid_and_fence():
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
            pl.system.cacheinvalid(dst, [1, N], [0, 0])
            pl.system.fence()
            pld.tile.remote_store(local, target=dst, peer=peer, offsets=[0, 0])
            pl.system.cacheinvalid(dst, [1, N], [0, 0])
            pl.system.fence()
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_two_distinct_targets_each_gets_cacheinvalid():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            dst_a: pld.DistributedTensor[[1, N], pl.FP32],
            dst_b: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pld.tile.remote_store(local, target=dst_a, peer=peer, offsets=[0, 0])
            pld.tile.remote_store(local, target=dst_b, peer=peer, offsets=[0, 0])
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            dst_a: pld.DistributedTensor[[1, N], pl.FP32],
            dst_b: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pld.tile.remote_store(local, target=dst_a, peer=peer, offsets=[0, 0])
            pl.system.cacheinvalid(dst_a, [1, N], [0, 0])
            pl.system.fence()
            pld.tile.remote_store(local, target=dst_b, peer=peer, offsets=[0, 0])
            pl.system.cacheinvalid(dst_b, [1, N], [0, 0])
            pl.system.fence()
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_inscope_alias_of_param_target_gets_cacheinvalid():
    # The remote_store target is an in-scope alias (a view) of a window param.
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
            dv = pl.tensor.view(dst, [1, N])
            pld.tile.remote_store(local, target=dv, peer=peer, offsets=[0, 0])
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
            dv = pl.tensor.view(dst, [1, N])
            pld.tile.remote_store(local, target=dv, peer=peer, offsets=[0, 0])
            pl.system.cacheinvalid(dv, [1, N], [0, 0])
            pl.system.fence()
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_write_inside_if_branch_cacheinvalidated_in_branch():
    # The write (and its target alias) live inside the branch; the cacheinvalid +
    # fence are emitted right after the write, in the branch. The outer notify
    # needs no marker.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            dst: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            cond: pl.Scalar[pl.BOOL],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            if cond:
                dv = pl.tensor.view(dst, [1, N])
                pld.tile.remote_store(local, target=dv, peer=peer, offsets=[0, 0])
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
            cond: pl.Scalar[pl.BOOL],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            if cond:
                dv = pl.tensor.view(dst, [1, N])
                pld.tile.remote_store(local, target=dv, peer=peer, offsets=[0, 0])
                pl.system.cacheinvalid(dv, [1, N], [0, 0])
                pl.system.fence()
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_notify_inside_if_write_fenced_before_if():
    # The write is before the if; its cacheinvalid + fence release the data for the
    # conditional notify, which needs no marker of its own.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            dst: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            cond: pl.Scalar[pl.BOOL],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pld.tile.remote_store(local, target=dst, peer=peer, offsets=[0, 0])
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
            dst: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            cond: pl.Scalar[pl.BOOL],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pld.tile.remote_store(local, target=dst, peer=peer, offsets=[0, 0])
            pl.system.cacheinvalid(dst, [1, N], [0, 0])
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
            dst: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pld.tile.remote_store(local, target=dst, peer=peer, offsets=[0, 0])
            for i in pl.range(N):
                pld.system.notify(target=signal, peer=i, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

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
            pl.system.cacheinvalid(dst, [1, N], [0, 0])
            pl.system.fence()
            for i in pl.range(N):
                pld.system.notify(target=signal, peer=i, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_loop_back_edge_notify_then_write():
    # for { notify; store } — the tail store gets its cacheinvalid + fence; that
    # fence (from the previous iteration / final iteration) covers the notify.
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
            for i in pl.range(N):
                pld.system.notify(
                    target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )
                pld.tile.remote_store(local, target=dst, peer=peer, offsets=[0, 0])

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
            for i in pl.range(N):
                pld.system.notify(
                    target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )
                pld.tile.remote_store(local, target=dst, peer=peer, offsets=[0, 0])
                pl.system.cacheinvalid(dst, [1, N], [0, 0])
                pl.system.fence()

    ir.assert_structural_equal(_apply(Before), Expected)


def test_nested_loops_write_in_inner_body():
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
            for _i in pl.range(N):
                for _j in pl.range(N):
                    pld.tile.remote_store(local, target=dst, peer=peer, offsets=[0, 0])
                    pld.system.notify(
                        target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                    )

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
            for _i in pl.range(N):
                for _j in pl.range(N):
                    pld.tile.remote_store(local, target=dst, peer=peer, offsets=[0, 0])
                    pl.system.cacheinvalid(dst, [1, N], [0, 0])
                    pl.system.fence()
                    pld.system.notify(
                        target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                    )

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
                pld.tile.remote_store(local, target=win, peer=me, offsets=[0, 0])

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
                pld.tile.remote_store(local, target=win, peer=me, offsets=[0, 0])
                pl.system.cacheinvalid(win, [1, N], [0, 0])
                pl.system.fence()

    ir.assert_structural_equal(_apply(Before), Expected)


def test_combo_two_phase_loops():
    # for { notify; store }; for { notify; store } — reduce-scatter then allgather.
    # Each store gets its cacheinvalid + fence; the notifies get nothing.
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
                pld.tile.remote_store(local, target=win, peer=peer, offsets=[0, 0])
            for _t in pl.range(N):
                pld.system.notify(
                    target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )
                pld.tile.remote_store(local, target=win, peer=peer, offsets=[0, 0])

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
                pld.tile.remote_store(local, target=win, peer=peer, offsets=[0, 0])
                pl.system.cacheinvalid(win, [1, N], [0, 0])
                pl.system.fence()
            for _t in pl.range(N):
                pld.system.notify(
                    target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )
                pld.tile.remote_store(local, target=win, peer=peer, offsets=[0, 0])
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


def test_idempotent():
    # Re-running the pass on already-marked IR inserts nothing.
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

    once = _apply(Before)
    twice = _apply(once)
    ir.assert_structural_equal(twice, once)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
