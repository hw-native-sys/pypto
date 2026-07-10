# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression: if/else-yield tile phi under memory_planner=PTOAS (issue #1956).

Under PTOAS the opportunistic MemoryReuse pass (whose YieldFixupMutator aliases a
branch's tile yields onto the phi return_var's canonical handle) is skipped. PTO
codegen must therefore itself point each branch's tile producer at the single
head-declared phi handle; otherwise each branch writes its own handle and the
post-if read of the phi handle reads a buffer no branch ever wrote — a silent
miscompile. This test asserts both branch producers write the one phi handle and
that handle is declared exactly once.
"""

# pyright: reportUndefinedVariable=false

import re

import pypto.language as pl
import pytest
from pypto import ir as _ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


@pl.program
class IfPhiProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        a: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])
        if flag == 0:
            r: pl.Tile[[64, 64], pl.FP32] = pl.add(a, 1.0)
        else:
            r: pl.Tile[[64, 64], pl.FP32] = pl.mul(a, 2.0)
        return pl.store(r, [0, 0], out)


@pl.program
class IfPhiPassThroughProgram:
    """One branch yields a fresh producer; the other yields the outer tile `a`
    unchanged (a pass-through). The pass-through branch has no producer writing
    the phi buffer, so codegen must copy `a` into the phi handle (tmov)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        a: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])
        if flag == 0:
            r: pl.Tile[[64, 64], pl.FP32] = pl.add(a, 1.0)
        else:
            r: pl.Tile[[64, 64], pl.FP32] = a
        return pl.store(r, [0, 0], out)


def _ptoas_pto(program) -> str:
    reset_for_testing()
    set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    return codegen.PTOCodegen().generate(_ir.Program([func], "kernel", optimized.span), emit_tile_addr=False)


def _first_group(pattern: str, text: str) -> str:
    m = re.search(pattern, text)
    assert m is not None, f"pattern {pattern!r} not found in:\n{text}"
    return m.group(1)


def test_if_phi_branches_write_one_head_declared_handle():
    mlir = _ptoas_pto(IfPhiProgram)
    # The tile the store reads is the phi buffer.
    store = next(ln for ln in mlir.splitlines() if "pto.tstore" in ln)
    phi = _first_group(r"pto\.tstore ins\((%[A-Za-z0-9_]+)", store)

    # It is declared exactly once (at the function head, dominating both branches).
    decls = [ln for ln in mlir.splitlines() if f"{phi} = pto.alloc_tile" in ln]
    assert len(decls) == 1, f"phi handle {phi} must be declared exactly once:\n{mlir}"

    # Both branch producers (tadds / tmuls) write that same phi handle directly —
    # a branch-local producer is re-pointed, so no copy is needed.
    producer_outs = [
        _first_group(r"outs\((%[A-Za-z0-9_]+)", ln)
        for ln in mlir.splitlines()
        if ("pto.tadds" in ln or "pto.tmuls" in ln) and "outs(" in ln
    ]
    assert len(producer_outs) == 2, f"expected two branch producers, got {producer_outs}:\n{mlir}"
    for out in producer_outs:
        assert out == phi, f"branch producer writes {out}, not the phi handle {phi}:\n{mlir}"
    assert "pto.tmov" not in mlir, f"branch-local phi must be copy-free (no tmov):\n{mlir}"


def test_if_phi_pass_through_branch_copies_into_phi_handle():
    mlir = _ptoas_pto(IfPhiPassThroughProgram)
    store = next(ln for ln in mlir.splitlines() if "pto.tstore" in ln)
    phi = _first_group(r"pto\.tstore ins\((%[A-Za-z0-9_]+)", store)

    # The producing branch writes the phi handle directly.
    prod = next(ln for ln in mlir.splitlines() if "pto.tadds" in ln)
    assert _first_group(r"outs\((%[A-Za-z0-9_]+)", prod) == phi, f"producer must write phi {phi}:\n{mlir}"

    # The pass-through branch (yields the outer tile unchanged) copies it into the
    # phi handle via tmov, so the post-if store never reads an uninitialised buffer.
    movs = [ln for ln in mlir.splitlines() if "pto.tmov" in ln and f"outs({phi}" in ln]
    assert len(movs) == 1, f"pass-through branch must tmov into the phi handle {phi}:\n{mlir}"


@pl.program
class LoopCarriedIfPhiProgram:
    """A `pl.range` accumulator whose per-iteration update is an if/else. The loop
    init, the iter_arg, the if-phi and the loop result all denote one buffer, so
    under PTOAS they must all collapse onto a single `tile_buf` handle."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        acc: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])
        for i in pl.range(4):  # noqa: B007 - loop index drives the branch
            if i == 0:
                acc = pl.add(acc, 1.0)
            else:
                acc = pl.mul(acc, 2.0)
        return pl.store(acc, [0, 0], out)


def test_loop_carried_if_phi_shares_the_accumulator_handle():
    """The if-phi must reuse the accumulator's handle, not declare a second buffer.

    Under PTOAS no `addr` is baked, so two addr-less `pto.alloc_tile` are two
    independent buffers to ptoas PlanMemory. Head-declaring a fresh handle for the
    phi therefore stranded the accumulator: the branch producers wrote the phi
    buffer while the loop-carried read and the post-loop store used the accumulator
    handle, which no branch ever wrote — a silent miscompile (issue: the loop-carried
    counterpart of the `scf.if` phi fix #1956/#1985).
    """
    mlir = _ptoas_pto(LoopCarriedIfPhiProgram)

    store = next(ln for ln in mlir.splitlines() if "pto.tstore" in ln)
    acc = _first_group(r"pto\.tstore ins\((%[A-Za-z0-9_]+)", store)

    # Exactly one tile_buf is declared for the accumulator, and nothing else.
    decls = [ln for ln in mlir.splitlines() if "= pto.alloc_tile" in ln]
    assert len(decls) == 1, f"expected a single alloc_tile, got {len(decls)}:\n{mlir}"
    assert f"{acc} = pto.alloc_tile" in decls[0], f"the sole alloc_tile must be {acc}:\n{mlir}"

    # Both branch producers write that handle, and both read it back (the carry).
    producers = [ln for ln in mlir.splitlines() if "pto.tadds" in ln or "pto.tmuls" in ln]
    assert len(producers) == 2, f"expected two branch producers, got {producers}:\n{mlir}"
    for prod in producers:
        assert _first_group(r"outs\((%[A-Za-z0-9_]+)", prod) == acc, f"producer must write {acc}:\n{prod}"
        assert _first_group(r"ins\((%[A-Za-z0-9_]+)", prod) == acc, f"producer must read {acc}:\n{prod}"

    assert "pto.tmov" not in mlir, f"a shared handle needs no copy:\n{mlir}"


def test_loop_carried_if_phi_keeps_baked_addr_aliasing_under_pypto():
    """Default planner: each var keeps its own alloc_tile, aliased by a shared addr."""
    reset_for_testing()
    set_backend_type(BackendType.Ascend910B)
    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(LoopCarriedIfPhiProgram)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    mlir = codegen.PTOCodegen().generate(_ir.Program([func], "kernel", optimized.span))

    decls = [ln for ln in mlir.splitlines() if "= pto.alloc_tile" in ln]
    assert len(decls) > 1, f"PyPTO planner declares one alloc_tile per var:\n{mlir}"
    addrs = {_first_group(r"addr = (%[A-Za-z0-9_]+)", ln) for ln in decls}
    assert len(addrs) == 1, f"the aliased tiles must share one baked addr, got {addrs}:\n{mlir}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
