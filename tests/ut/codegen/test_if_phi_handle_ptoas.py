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


def _sole_line(mlir: str, needle: str) -> str:
    lines = [ln for ln in mlir.splitlines() if needle in ln]
    assert len(lines) == 1, f"expected exactly one {needle!r} line, got {lines}:\n{mlir}"
    return lines[0]


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


# ── [N, 1] col-vector carry: view yields must be copied, and the carry written ──

COLVEC_ROWS, COLVEC_STEPS = 16, 4


@pl.program
class ColVectorCarryProgram:
    """A `[N, 1]` col-vector loop carry updated through an if/else — the shape of an
    online-softmax `mi` / `li` carry.

    pypto runs `[N, 1]` elementwise on the `[1, N]` row-major view, so each branch
    yields a *reshape view* of its producer, not the producer itself.
    """

    @pl.function
    def kernel(
        self,
        x: pl.Tensor[[COLVEC_ROWS, 1], pl.FP32],
        y: pl.Tensor[[COLVEC_ROWS, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[COLVEC_ROWS, 1], pl.FP32]],
    ) -> pl.Tensor[[COLVEC_ROWS, 1], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="colvec_carry"):
            li = x[:, :]
            for i in pl.range(COLVEC_STEPS):  # noqa: B007 - index drives the branch
                b = y[:, :]
                if i == 0:
                    li = pl.mul(li, b)
                else:
                    li = pl.add(pl.mul(li, b), b)
            out[:, :] = li
        return out


def _incore_ptoas_pto(program) -> str:
    """Emit PTO for a `pl.at`-outlined kernel under the PTOAS planner."""
    reset_for_testing()
    set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    incore = [f for f in optimized.functions.values() if f.func_type != pl.FunctionType.Orchestration]
    assert len(incore) == 1, f"expected one in-core function, got {[f.name for f in incore]}"
    single = _ir.Program([incore[0]], incore[0].name, optimized.span)
    return codegen.PTOCodegen().generate(single, emit_tile_addr=False)


def _outs_of(line: str) -> str:
    return _first_group(r"outs\((%[A-Za-z0-9_]+)", line)


def _ins_of(line: str) -> str:
    return _first_group(r"ins\((%[A-Za-z0-9_]+)", line)


def test_colvec_carry_branch_views_are_copied_into_the_phi_handle():
    """A branch that yields a zero-copy view must be copied, not re-pointed.

    `tile.reshape` is a Call, so the branch-local-producer test used to accept it
    and re-point it at the phi handle — but a view's codegen emits nothing when the
    target already carries the result type, so the phi buffer stayed unwritten and
    the loop became a silent no-op. Views must fall through to the tmov fallback.
    """
    mlir = _incore_ptoas_pto(ColVectorCarryProgram)

    store = _sole_line(mlir, "pto.tstore")
    carry = _ins_of(store)

    # The final tmov of the iteration writes the carry from the phi handle.
    into_carry = [ln for ln in mlir.splitlines() if "pto.tmov" in ln and _outs_of(ln) == carry]
    assert len(into_carry) == 1, f"the loop carry must be written back exactly once:\n{mlir}"
    phi = _ins_of(into_carry[0])
    assert phi != carry, f"the write-back must copy the phi handle into the carry:\n{into_carry[0]}"

    # Both branches copy their (view) result into that phi handle.
    into_phi = [ln for ln in mlir.splitlines() if "pto.tmov" in ln and _outs_of(ln) == phi]
    assert len(into_phi) == 2, f"each branch must tmov its view yield into {phi}:\n{mlir}"
    for ln in into_phi:
        src = _ins_of(ln)
        assert f"{src} = pto.treshape" in mlir, f"a branch yield should be a reshape view:\n{ln}"


def test_colvec_carry_tmov_annotates_each_operand_with_its_own_type():
    """A `pto.treshape` view carries static valid dims; an `alloc_tile` handle does
    not. Annotating both tmov operands with one type makes the def/use types
    disagree and ptoas rejects the module."""
    mlir = _incore_ptoas_pto(ColVectorCarryProgram)

    store = _sole_line(mlir, "pto.tstore")
    carry = _ins_of(store)
    into_carry = next(ln for ln in mlir.splitlines() if "pto.tmov" in ln and _outs_of(ln) == carry)
    phi = _ins_of(into_carry)

    into_phi = [ln for ln in mlir.splitlines() if "pto.tmov" in ln and _outs_of(ln) == phi]
    assert into_phi, f"no tmov into the phi handle {phi}:\n{mlir}"
    for ln in into_phi:
        src, dst = ln.split("ins(", 1)[1].split(") outs(", 1)
        assert "v_row=?" not in src, f"the view operand must keep its static valid dims:\n{ln}"
        assert "v_row=?" in dst, f"the alloc_tile handle stays dynamic-valid:\n{ln}"


def test_colvec_carry_is_written_back_under_pypto_planner():
    """Default planner: MemoryReuse's YieldFixupMutator inserts the same write-back.

    Aliasing is by baked address, not by SSA name — each var keeps its own
    `alloc_tile` — so the write-back's destination is a *different* handle sitting
    at the carry's address.
    """
    reset_for_testing()
    set_backend_type(BackendType.Ascend910B)
    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(ColVectorCarryProgram)
    incore = [f for f in optimized.functions.values() if f.func_type != pl.FunctionType.Orchestration]
    mlir = codegen.PTOCodegen().generate(_ir.Program([incore[0]], incore[0].name, optimized.span))

    assert "pto.treshape" not in mlir, f"PYPTO expresses views as re-declared allocs:\n{mlir}"

    def addr_of(handle: str) -> str:
        return _first_group(r"addr = (%[A-Za-z0-9_]+)", _sole_line(mlir, f"{handle} = pto.alloc_tile"))

    carry_addr = addr_of(_ins_of(_sole_line(mlir, "pto.tstore")))
    written = [ln for ln in mlir.splitlines() if "pto.tmov" in ln and addr_of(_outs_of(ln)) == carry_addr]
    assert written, f"the carry's buffer must be written before the store:\n{mlir}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
