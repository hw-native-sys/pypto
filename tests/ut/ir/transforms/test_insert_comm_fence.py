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

Two independent placements:

* ``pl.system.cacheinvalid`` is per-address — one is emitted **immediately after
  every publishing write** (remote_store / put / get, or a local store into a
  window-bound ``DistributedTensor``), covering that write's whole target tensor.
  Its target is in scope at the write, so it is never dropped.
* ``pl.system.fence()`` is per-notify — a single GM barrier before a notify that
  has an unflushed publishing write (multiple writes share one fence).

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


def test_local_store_into_plain_tensor_no_ops():
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
    assert _count(lines, "pl.system.fence()") == 0
    assert _count(lines, "pl.system.cacheinvalid(") == 0


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
    # Per-address: one cacheinvalid after EACH write; the notify shares one fence.
    assert _count(lines, "pl.system.cacheinvalid(") == 2
    assert _count(lines, "pl.system.fence()") == 1
    fence_i = _first_index(lines, "pl.system.fence()")
    notify_i = _first_index(lines, "system.notify")
    last_store_i = max(i for i, line in enumerate(lines) if "remote_store" in line)
    last_cinv_i = max(i for i, line in enumerate(lines) if "pl.system.cacheinvalid(" in line)
    assert last_store_i < last_cinv_i < fence_i < notify_i


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
    assert _count(lines, "pl.system.cacheinvalid(") == 2
    assert _count(lines, "pl.system.fence()") == 1
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


def test_notify_inside_if_fences_inside_branch_write_cacheinvalidated_before():
    # The write is before the if; its cacheinvalid follows it (before the if).
    # The notify is conditional, so its fence goes inside the branch.
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
    assert _count(lines, "pl.system.cacheinvalid(") == 1
    assert _count(lines, "pl.system.fence()") == 1
    cinv_i = _first_index(lines, "pl.system.cacheinvalid(")
    if_i = _first_index(lines, "if ")
    fence_i = _first_index(lines, "pl.system.fence()")
    notify_i = _first_index(lines, "system.notify")
    # cacheinvalid before the if (after the write); fence inside the branch.
    assert cinv_i < if_i < fence_i < notify_i


def test_notify_in_both_if_branches_two_fences_one_cacheinvalid():
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
    # One write -> one cacheinvalid (before the if); each branch's notify fences.
    assert _count(lines, "pl.system.cacheinvalid(") == 1
    assert _count(lines, "pl.system.fence()") == 2
    cinv_i = _first_index(lines, "pl.system.cacheinvalid(")
    if_i = _first_index(lines, "if ")
    assert cinv_i < if_i


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
    # cacheinvalid(dv) is placed at the write, inside the branch (in scope).
    assert _count(lines, "pl.system.cacheinvalid(") == 1
    cinv_i = _first_index(lines, "pl.system.cacheinvalid(")
    store_i = _first_index(lines, "remote_store")
    if_i = _first_index(lines, "if ")
    notify_i = _first_index(lines, "system.notify")
    assert if_i < store_i < cinv_i < notify_i
    assert "cacheinvalid(dv," in lines[cinv_i]
    # Fence releases the (conditional) pending write before the outer notify.
    assert _count(lines, "pl.system.fence()") == 1


def test_existing_user_fence_is_not_duplicated():
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
            pl.system.fence()
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    lines = _run(Prog)
    # A user fence clears the pending write, so no duplicate fence is added; the
    # per-write cacheinvalid is still emitted after the write.
    assert _count(lines, "pl.system.fence()") == 1
    assert _count(lines, "pl.system.cacheinvalid(") == 1


def test_no_publishing_write_no_ops():
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
    assert _count(lines, "pl.system.fence()") == 0
    assert _count(lines, "pl.system.cacheinvalid(") == 0


def test_notify_inside_loop_after_write_hoists_fence_before_loop():
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
    assert _count(lines, "pl.system.cacheinvalid(") == 1
    assert _count(lines, "pl.system.fence()") == 1
    cinv_i = _first_index(lines, "pl.system.cacheinvalid(")
    fence_i = _first_index(lines, "pl.system.fence()")
    for_i = _first_index(lines, "for ")
    notify_i = _first_index(lines, "system.notify")
    # cacheinvalid after the write, fence hoisted before the loop.
    assert cinv_i < fence_i < for_i < notify_i


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
    # The tail write's cacheinvalid follows it (in scope); the head fence orders
    # that write before the next iteration's notify (ring back-edge).
    assert _count(lines, "pl.system.cacheinvalid(") == 1
    assert _count(lines, "pl.system.fence()") == 1
    for_i = _first_index(lines, "for ")
    fence_i = _first_index(lines, "pl.system.fence()")
    notify_i = _first_index(lines, "system.notify")
    store_i = _first_index(lines, "remote_store")
    cinv_i = _first_index(lines, "pl.system.cacheinvalid(")
    assert for_i < fence_i < notify_i < store_i < cinv_i
    assert "cacheinvalid(dst," in lines[cinv_i]


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
    # The tail store is released by the next iteration's head notify (back-edge):
    # a fence sits at the s-loop head (before the p-loop that opens with notify).
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
    assert _count(lines, "pl.system.cacheinvalid(") == 1
    assert _count(lines, "pl.system.fence()") == 1
    outer_for_i = _first_index(lines, "for ")
    fence_i = _first_index(lines, "pl.system.fence()")
    inner_for_i = next(i for i, ln in enumerate(lines) if "for " in ln and i > outer_for_i)
    notify_i = _first_index(lines, "system.notify")
    # fence is at the s-loop head, before the p-loop / notify.
    assert outer_for_i < fence_i < inner_for_i < notify_i


def test_combo_preloop_write_plus_back_edge():
    # store; for { notify; store } — the loop-head fence (needed for the back-edge)
    # already covers the pre-loop store on iteration 0, so NO redundant fence is
    # hoisted before the loop: exactly one fence, at the loop head.
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
    assert _count(lines, "pl.system.cacheinvalid(") == 2
    assert _count(lines, "pl.system.fence()") == 1
    for_i = _first_index(lines, "for ")
    fence_i = _first_index(lines, "pl.system.fence()")
    notify_i = _first_index(lines, "system.notify")
    # The single fence is inside the loop, at its head (not hoisted before it).
    assert for_i < fence_i < notify_i


def test_combo_if_notify_in_back_edge_loop():
    # for { if c: notify; store } — conditional notify inside a back-edge loop:
    # the fence goes inside the branch, before the notify.
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
    assert _count(lines, "pl.system.cacheinvalid(") == 1
    assert _count(lines, "pl.system.fence()") == 1
    if_i = _first_index(lines, "if ")
    fence_i = _first_index(lines, "pl.system.fence()")
    notify_i = _first_index(lines, "system.notify")
    # fence is inside the branch: after the `if`, before the notify.
    assert if_i < fence_i < notify_i


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
    assert _count(lines, "pl.system.cacheinvalid(") == 1
    assert _count(lines, "pl.system.fence()") == 1
    cinv_i = _first_index(lines, "pl.system.cacheinvalid(")
    first_if_i = _first_index(lines, "if ")
    fence_i = _first_index(lines, "pl.system.fence()")
    notify_i = _first_index(lines, "system.notify")
    # cacheinvalid before the ifs (after the write); fence at the innermost notify.
    assert cinv_i < first_if_i < fence_i < notify_i


def test_combo_two_phase_loops():
    # for { notify; store }; for { notify; store } — reduce-scatter then allgather.
    # Each loop head-fences; phase-2's iteration-0 head fence also covers phase-1's
    # tail store, so no separate fence is hoisted between the loops. Two fences.
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
    assert _count(lines, "pl.system.cacheinvalid(") == 2
    # Exactly one head fence per loop, none hoisted between them (a between-loops
    # fence would make it 3). Each `for` is immediately followed by its fence.
    assert _count(lines, "pl.system.fence()") == 2
    for_idxs = [i for i, ln in enumerate(lines) if "for " in ln]
    assert len(for_idxs) == 2
    for fi in for_idxs:
        assert "pl.system.fence()" in lines[fi + 1]


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
