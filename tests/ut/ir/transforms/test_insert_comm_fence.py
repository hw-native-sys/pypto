# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Unit tests for the ``InsertCommFence`` pass.

The pass emits the ptoas data-before-signal markers, all through the single
``system.cacheinvalid`` op (two forms distinguished by arity):

* **Region** ``pl.system.cacheinvalid(tensor, shapes, offsets)`` + ``fence`` —
  emitted **immediately after every publishing write** (remote_store / put / get,
  or a local store into a window-bound ``DistributedTensor``). ptoas requires the
  fence to directly follow the release marker, so both land at the write site.
* **Whole-GM** ``pl.system.cacheinvalid()`` (no args) + ``fence`` — emitted
  **before a bare barrier notify** (a notify with no pending fenced publishing
  write — a pure signal, or one whose write is in a prior loop). A notify that
  *does* have a pending fenced write needs nothing (the write already fenced).
* **Whole-GM** ``pl.system.cacheinvalid()`` — emitted **after every wait** (the
  consume-side invalidate before the next cacheable read).

Loops are entered with a cleared ``pending`` (ptoas checks the marker lexically,
so a loop-head notify cannot rely on a fence from before the loop or the previous
iteration's tail write), so every loop-body notify gets its own marker.

Assertions run on ``python_print`` output line order.

The pass is run inside ``passes.PassContext([])`` so the autouse verification
context (which would enforce the pass's ``SplitIncoreOrch`` requirement on a
freshly parsed program) is bypassed — mirroring ``test_stamp_tfree_split.py``.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.ir.printer import python_print
from pypto.pypto_core import passes

N = 8


def _run(program) -> list[str]:
    """Run insert_comm_fence (no verification) and return printed body lines."""
    with passes.PassContext([]):
        after = passes.insert_comm_fence()(program)
    return [line.strip() for line in python_print(after).splitlines()]


def _first_index(lines: list[str], needle: str) -> int:
    return next(i for i, line in enumerate(lines) if needle in line)


def _count(lines: list[str], needle: str) -> int:
    return sum(1 for line in lines if needle in line)


def _count_region(lines: list[str]) -> int:
    """Region cacheinvalid: ``pl.system.cacheinvalid(<args>)`` (has arguments)."""
    return sum(
        1 for line in lines if "pl.system.cacheinvalid(" in line and "pl.system.cacheinvalid()" not in line
    )


def _count_all(lines: list[str]) -> int:
    """Whole-GM cacheinvalid: the no-argument ``pl.system.cacheinvalid()`` form."""
    return _count(lines, "pl.system.cacheinvalid()")


def test_remote_store_then_notify_inserts_cacheinvalid_and_fence():
    @pl.program
    class Prog:
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

    lines = _run(Prog)
    assert _count(lines, "pl.system.cacheinvalid(") == 1
    assert _count(lines, "pl.system.fence()") == 1
    store_i = _first_index(lines, "remote_store")
    cinv_i = _first_index(lines, "pl.system.cacheinvalid(")
    fence_i = _first_index(lines, "pl.system.fence()")
    notify_i = _first_index(lines, "system.notify")
    # cacheinvalid immediately after the write; fence before the notify.
    assert store_i < cinv_i < fence_i < notify_i
    # Whole-tensor region: full dst shape [1, N] at zero offsets.
    assert "cacheinvalid(dst, [1, 8], [0, 0])" in lines[cinv_i]


def test_local_store_into_window_then_notify():
    @pl.program
    class Prog:
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

    lines = _run(Prog)
    assert _count(lines, "pl.system.cacheinvalid(") == 1
    assert _count(lines, "pl.system.fence()") == 1
    store_i = _first_index(lines, "tile.store")
    cinv_i = _first_index(lines, "pl.system.cacheinvalid(")
    fence_i = _first_index(lines, "pl.system.fence()")
    notify_i = _first_index(lines, "system.notify")
    assert store_i < cinv_i < fence_i < notify_i
    assert "cacheinvalid(win," in lines[cinv_i]


def test_local_store_into_plain_tensor_no_region_but_barrier_marker():
    @pl.program
    class Prog:
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

    lines = _run(Prog)
    # No publishing write -> no region cacheinvalid; the notify is a bare barrier
    # notify, so it still gets a whole-GM cacheinvalid + fence.
    assert _count_region(lines) == 0
    assert _count_all(lines) == 1
    assert _count(lines, "pl.system.fence()") == 1
    store_i = _first_index(lines, "tile.store")
    all_i = _first_index(lines, "pl.system.cacheinvalid()")
    fence_i = _first_index(lines, "pl.system.fence()")
    notify_i = _first_index(lines, "system.notify")
    assert store_i < all_i < fence_i < notify_i


def test_multiple_writes_one_cacheinvalid_each_single_fence():
    @pl.program
    class Prog:
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

    lines = _run(Prog)
    # Per-address: each write gets its own region cacheinvalid + fence; the notify
    # is then already fenced by the last write, so it adds nothing.
    assert _count_region(lines) == 2
    assert _count(lines, "pl.system.fence()") == 2
    assert _count_all(lines) == 0
    notify_i = _first_index(lines, "system.notify")
    last_store_i = max(i for i, line in enumerate(lines) if "remote_store" in line)
    last_cinv_i = max(i for i, line in enumerate(lines) if "pl.system.cacheinvalid(" in line)
    last_fence_i = max(i for i, line in enumerate(lines) if "pl.system.fence()" in line)
    assert last_store_i < last_cinv_i < last_fence_i < notify_i


def test_two_distinct_targets_each_gets_cacheinvalid():
    @pl.program
    class Prog:
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

    lines = _run(Prog)
    # Each distinct target gets its own region cacheinvalid + fence.
    assert _count_region(lines) == 2
    assert _count(lines, "pl.system.fence()") == 2
    assert _count_all(lines) == 0
    assert any("cacheinvalid(dst_a," in line for line in lines)
    assert any("cacheinvalid(dst_b," in line for line in lines)


def test_inscope_alias_of_param_target_gets_cacheinvalid():
    # The remote_store target is an in-scope alias of a window param (a view).
    # cacheinvalid is emitted right after the write, on the alias.
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            dst: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            dv = pl.tensor.view(dst, [1, N])  # alias of the window param (no write)
            pld.tile.remote_store(local, target=dv, peer=peer, offsets=[0, 0])
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    lines = _run(Prog)
    assert _count(lines, "pl.system.cacheinvalid(") == 1
    assert _count(lines, "pl.system.fence()") == 1
    store_i = _first_index(lines, "remote_store")
    cinv_i = _first_index(lines, "pl.system.cacheinvalid(")
    assert store_i < cinv_i
    assert "cacheinvalid(dv," in lines[cinv_i]


def test_notify_inside_if_write_fenced_before_if():
    # The write is before the if; its cacheinvalid + fence follow it (before the
    # if). The conditional notify is already fenced by that write, so it adds
    # nothing of its own.
    @pl.program
    class Prog:
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

    lines = _run(Prog)
    assert _count_region(lines) == 1
    assert _count(lines, "pl.system.fence()") == 1
    assert _count_all(lines) == 0
    cinv_i = _first_index(lines, "pl.system.cacheinvalid(")
    fence_i = _first_index(lines, "pl.system.fence()")
    if_i = _first_index(lines, "if ")
    notify_i = _first_index(lines, "system.notify")
    # cacheinvalid + fence after the write, before the if; notify inside the branch.
    assert cinv_i < fence_i < if_i < notify_i


def test_notify_in_both_if_branches_single_write_fence():
    @pl.program
    class Prog:
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
            else:
                pld.system.notify(
                    target=signal, peer=peer, offsets=[0, 0], value=2, op=pld.NotifyOp.AtomicAdd
                )

    lines = _run(Prog)
    # One write -> one region cacheinvalid + fence (before the if). Both branches'
    # notifies are already fenced by that write, so neither adds a marker.
    assert _count_region(lines) == 1
    assert _count(lines, "pl.system.fence()") == 1
    assert _count_all(lines) == 0
    cinv_i = _first_index(lines, "pl.system.cacheinvalid(")
    fence_i = _first_index(lines, "pl.system.fence()")
    if_i = _first_index(lines, "if ")
    assert cinv_i < fence_i < if_i


def test_write_inside_if_branch_cacheinvalidated_in_branch():
    # The write (and its target alias) live inside the branch; cacheinvalid is
    # emitted right after the write, in the branch — never out of scope.
    @pl.program
    class Prog:
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
                dv = pl.tensor.view(dst, [1, N])  # branch-local alias
                pld.tile.remote_store(local, target=dv, peer=peer, offsets=[0, 0])
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    lines = _run(Prog)
    # region cacheinvalid + fence at the (conditional) write, inside the branch.
    assert _count_region(lines) == 1
    # The write is only conditional, so after the if it is not proven: the outer
    # notify is a bare barrier notify -> whole-GM cacheinvalid + fence.
    assert _count_all(lines) == 1
    assert _count(lines, "pl.system.fence()") == 2  # write fence + bare-notify fence
    region_i = _first_index(lines, "cacheinvalid(dv,")
    all_i = _first_index(lines, "pl.system.cacheinvalid()")
    store_i = _first_index(lines, "remote_store")
    if_i = _first_index(lines, "if ")
    notify_i = _first_index(lines, "system.notify")
    assert if_i < store_i < region_i < all_i < notify_i


def test_already_marked_write_is_not_duplicated():
    # A write already followed by its region cacheinvalid + fence (the canonical
    # marker) is left untouched, and the following notify is already fenced.
    @pl.program
    class Prog:
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

    lines = _run(Prog)
    # Nothing added: the write is already region-invalidated + fenced, and the
    # notify is already fenced (pending), so no whole-GM marker either.
    assert _count_region(lines) == 1
    assert _count(lines, "pl.system.fence()") == 1
    assert _count_all(lines) == 0


def test_bare_barrier_notify_gets_whole_gm_and_fence():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    lines = _run(Prog)
    # No publishing write at all: the notify still needs the barrier release
    # marker, so a whole-GM cacheinvalid + fence precede it.
    assert _count_region(lines) == 0
    assert _count_all(lines) == 1
    assert _count(lines, "pl.system.fence()") == 1
    all_i = _first_index(lines, "pl.system.cacheinvalid()")
    fence_i = _first_index(lines, "pl.system.fence()")
    notify_i = _first_index(lines, "system.notify")
    assert all_i < fence_i < notify_i


def test_notify_inside_loop_after_write_gets_own_loop_marker():
    @pl.program
    class Prog:
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

    lines = _run(Prog)
    # The pre-loop write gets region cacheinvalid + fence. The loop is entered with
    # a cleared pending (the pre-loop fence does not lexically precede a loop-body
    # notify), so the loop-head notify gets its own whole-GM cacheinvalid + fence.
    assert _count_region(lines) == 1
    assert _count_all(lines) == 1
    assert _count(lines, "pl.system.fence()") == 2
    region_i = _first_index(lines, "cacheinvalid(dst,")
    for_i = _first_index(lines, "for ")
    all_i = _first_index(lines, "pl.system.cacheinvalid()")
    notify_i = _first_index(lines, "system.notify")
    assert region_i < for_i < all_i < notify_i


def test_loop_back_edge_notify_then_write():
    @pl.program
    class Prog:
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

    lines = _run(Prog)
    # The head notify is a bare barrier notify (loop entered with cleared pending):
    # whole-GM cacheinvalid + fence precede it. The tail write gets its own region
    # cacheinvalid + fence.
    assert _count_region(lines) == 1
    assert _count_all(lines) == 1
    assert _count(lines, "pl.system.fence()") == 2
    for_i = _first_index(lines, "for ")
    all_i = _first_index(lines, "pl.system.cacheinvalid()")
    fence_i = _first_index(lines, "pl.system.fence()")
    notify_i = _first_index(lines, "system.notify")
    store_i = _first_index(lines, "remote_store")
    cinv_i = _first_index(lines, "cacheinvalid(dst,")
    assert for_i < all_i < fence_i < notify_i < store_i < cinv_i


def test_nested_loops_write_in_inner_body():
    # A publishing write nested two loops deep still gets its cacheinvalid right
    # after it, in the inner body (also exercises the loop-body visit path).
    @pl.program
    class Prog:
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

    lines = _run(Prog)
    assert _count(lines, "pl.system.cacheinvalid(") == 1
    store_i = _first_index(lines, "remote_store")
    cinv_i = _first_index(lines, "pl.system.cacheinvalid(")
    notify_i = _first_index(lines, "system.notify")
    assert store_i < cinv_i < notify_i
    assert "cacheinvalid(dst," in lines[cinv_i]


def _assert_each_write_then_cacheinvalid(lines: list[str]) -> None:
    """Every publishing write is immediately followed by its cacheinvalid."""
    for i, line in enumerate(lines):
        if "remote_store" in line:
            assert i + 1 < len(lines) and "pl.system.cacheinvalid(" in lines[i + 1], (
                f"write at line {i} not immediately followed by a cacheinvalid: {lines[i : i + 2]}"
            )


# --- combined paradigms ---------------------------------------------------


def test_combo_ring_barrier_idiom():
    # for s: { for p: (if p != me: notify); store }  — the ring-allreduce barrier.
    # The conditional notify has no proven write on iteration 0, so it is a bare
    # barrier notify: whole-GM cacheinvalid + fence inside the branch. The tail
    # store gets its own region cacheinvalid.
    @pl.program
    class Prog:
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

    lines = _run(Prog)
    _assert_each_write_then_cacheinvalid(lines)
    assert _count_region(lines) == 1
    assert _count_all(lines) == 1
    assert _count(lines, "pl.system.fence()") == 2  # bare-notify fence + store fence
    outer_for_i = _first_index(lines, "for ")
    inner_for_i = next(i for i, ln in enumerate(lines) if "for " in ln and i > outer_for_i)
    if_i = _first_index(lines, "if ")
    all_i = _first_index(lines, "pl.system.cacheinvalid()")
    fence_i = _first_index(lines, "pl.system.fence()")
    notify_i = _first_index(lines, "system.notify")
    # whole-GM cacheinvalid + fence sit inside the branch, before the notify.
    assert outer_for_i < inner_for_i < if_i < all_i < fence_i < notify_i


def test_combo_preloop_write_plus_back_edge():
    # store; for { notify; store } — the pre-loop store gets region cacheinvalid +
    # fence. The loop is entered with cleared pending, so the loop-head notify gets
    # its own whole-GM cacheinvalid + fence, and the in-loop store gets its own
    # region cacheinvalid + fence.
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pld.tile.remote_store(local, target=win, peer=peer, offsets=[0, 0])
            for _s in pl.range(N):
                pld.system.notify(
                    target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )
                pld.tile.remote_store(local, target=win, peer=peer, offsets=[0, 0])

    lines = _run(Prog)
    _assert_each_write_then_cacheinvalid(lines)
    assert _count_region(lines) == 2  # pre-loop store + in-loop store
    assert _count_all(lines) == 1  # bare loop-head notify
    assert _count(lines, "pl.system.fence()") == 3
    for_i = _first_index(lines, "for ")
    all_i = _first_index(lines, "pl.system.cacheinvalid()")
    notify_i = _first_index(lines, "system.notify")
    # The whole-GM marker sits inside the loop, at its head, before the notify.
    assert for_i < all_i < notify_i


def test_combo_if_notify_in_back_edge_loop():
    # for { if c: notify; store } — conditional notify inside a back-edge loop:
    # no proven write precedes it, so a whole-GM cacheinvalid + fence go inside
    # the branch, before the notify.
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            c: pl.Scalar[pl.BOOL],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            for _s in pl.range(N):
                if c:
                    pld.system.notify(
                        target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                    )
                pld.tile.remote_store(local, target=win, peer=peer, offsets=[0, 0])

    lines = _run(Prog)
    _assert_each_write_then_cacheinvalid(lines)
    assert _count_region(lines) == 1
    assert _count_all(lines) == 1
    assert _count(lines, "pl.system.fence()") == 2  # bare-notify fence + store fence
    if_i = _first_index(lines, "if ")
    all_i = _first_index(lines, "pl.system.cacheinvalid()")
    fence_i = _first_index(lines, "pl.system.fence()")
    notify_i = _first_index(lines, "system.notify")
    # whole-GM cacheinvalid + fence are inside the branch, before the notify.
    assert if_i < all_i < fence_i < notify_i


def test_combo_nested_if_to_notify():
    # store; if c1: if c2: notify — the fence recurses through both ifs.
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            inp: pl.Tensor[[1, N], pl.FP32],
            win: pld.DistributedTensor[[1, N], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            c1: pl.Scalar[pl.BOOL],
            c2: pl.Scalar[pl.BOOL],
        ):
            local = pl.load(inp, [0, 0], [1, N])
            pld.tile.remote_store(local, target=win, peer=peer, offsets=[0, 0])
            if c1:
                if c2:
                    pld.system.notify(
                        target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                    )

    lines = _run(Prog)
    _assert_each_write_then_cacheinvalid(lines)
    assert _count_region(lines) == 1
    assert _count_all(lines) == 0
    assert _count(lines, "pl.system.fence()") == 1
    cinv_i = _first_index(lines, "pl.system.cacheinvalid(")
    fence_i = _first_index(lines, "pl.system.fence()")
    first_if_i = _first_index(lines, "if ")
    notify_i = _first_index(lines, "system.notify")
    # cacheinvalid + fence after the write (before the ifs); the pending notify
    # inside the nested ifs is already fenced and adds nothing.
    assert cinv_i < fence_i < first_if_i < notify_i


def test_combo_two_phase_loops():
    # for { notify; store }; for { notify; store } — reduce-scatter then allgather.
    # Each loop opens with a notify that has no proven write on iteration 0, so each
    # loop head gets a whole-GM cacheinvalid + fence; each store gets its region
    # cacheinvalid. Two of each.
    @pl.program
    class Prog:
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

    lines = _run(Prog)
    _assert_each_write_then_cacheinvalid(lines)
    assert _count_region(lines) == 2  # one store per loop
    assert _count_all(lines) == 2  # one bare loop-head notify per loop
    assert _count(lines, "pl.system.fence()") == 4  # bare-notify fence + store fence, per loop
    for_idxs = [i for i, ln in enumerate(lines) if "for " in ln]
    assert len(for_idxs) == 2
    # Each loop opens with a whole-GM cacheinvalid then a fence at its head.
    for fi in for_idxs:
        assert "pl.system.cacheinvalid()" in lines[fi + 1]
        assert "pl.system.fence()" in lines[fi + 2]


def test_idempotent():
    @pl.program
    class Prog:
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

    with passes.PassContext([]):
        once = passes.insert_comm_fence()(Prog)
        twice = passes.insert_comm_fence()(once)
    lines = [line.strip() for line in python_print(twice).splitlines()]
    # Re-running inserts nothing new.
    assert _count(lines, "pl.system.cacheinvalid(") == 1
    assert _count(lines, "pl.system.fence()") == 1


# --- consume side (wait) --------------------------------------------------


def test_wait_then_read_inserts_whole_gm_cacheinvalid_after_wait():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
        ):
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
            val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
            pl.write(out, [0, 0], val)

    lines = _run(Prog)
    # Consume side: a whole-GM cacheinvalid right after the wait, before the read.
    assert _count_all(lines) == 1
    assert _count_region(lines) == 0
    wait_i = _first_index(lines, "system.wait")
    all_i = _first_index(lines, "pl.system.cacheinvalid()")
    read_i = _first_index(lines, "tensor.read")
    assert wait_i < all_i < read_i


def test_wait_already_followed_by_cacheinvalid_not_duplicated():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
        ):
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
            pl.system.cacheinvalid()  # user already invalidated
            val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
            pl.write(out, [0, 0], val)

    lines = _run(Prog)
    # Idempotent: the existing whole-GM cacheinvalid after the wait is not doubled.
    assert _count_all(lines) == 1


def test_notify_wait_read_handshake_both_sides():
    # notify; wait; read — the barrier handshake: whole-GM + fence before the bare
    # notify (publish side) and whole-GM after the wait (consume side).
    @pl.program
    class Prog:
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

    lines = _run(Prog)
    assert _count_region(lines) == 0
    assert _count_all(lines) == 2  # one before the notify, one after the wait
    assert _count(lines, "pl.system.fence()") == 1
    pub_all_i = _first_index(lines, "pl.system.cacheinvalid()")
    fence_i = _first_index(lines, "pl.system.fence()")
    notify_i = _first_index(lines, "system.notify")
    wait_i = _first_index(lines, "system.wait")
    read_i = _first_index(lines, "tensor.read")
    con_all_i = max(i for i, ln in enumerate(lines) if "pl.system.cacheinvalid()" in ln)
    assert pub_all_i < fence_i < notify_i < wait_i < con_all_i < read_i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
