# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pl.MemRef(slots=N)`` lowered to a ptoas multi-buffer region.

A declared multi-slot allocation says "one allocation, N uniform slots, this use
takes slot k" — which is exactly ptoas's ``pto.alloc_multi_tile`` /
``pto.multi_tile_get``. Describing it that way is what lets ptoas plan the slots
as one region and derive per-slot (dynamic event id) synchronization from the
slot expression, instead of seeing N unrelated buffers.

The default PyPTO path keeps ordinary addressed allocations. Software pipelines
use fixed-address parent subviews for co-live loop slots at ``--pto-level=level3``;
single-live-slot schedules can retain multi-buffer regions.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors.
# pyright: reportUndefinedVariable=false

import math
import re

import pypto.language as pl
import pytest
from pypto import DataType, backend, ir
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes

ROTATING = pl.MemRef(slots=2)
CONST_SLOTS = pl.MemRef(slots=2)
SINGLE = pl.MemRef()
TOO_MANY = pl.MemRef(slots=17)
MIXED_SHAPES = pl.MemRef(slots=2)
MIXED_BINDING = pl.MemRef(slots=2)
MIXED_VALID = pl.MemRef(slots=2)
RUNTIME_VALID = pl.MemRef(slots=2)
CO_LIVE = pl.MemRef(slots=2)
PREFETCH = pl.MemRef(slots=2)
SIBLING_LOOPS = pl.MemRef(slots=2)
PREFETCHED = pl.MemRef(slots=3)
REVERSE_SLOTS = pl.MemRef(slots=2)


@pl.program
class PrefetchedSlots:
    """Two preloads followed by load-next / compute-current and two drain steps."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[64, 1024], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 1024], pl.FP32]],
    ) -> pl.Tensor[[64, 1024], pl.FP32]:
        _first: pl.Tile[[1, 1024], pl.FP32, PREFETCHED[0], pl.Mem.Vec] = pl.load(a, [0, 0], [1, 1024])
        _second: pl.Tile[[1, 1024], pl.FP32, PREFETCHED[1], pl.Mem.Vec] = pl.load(a, [1, 0], [1, 1024])
        for i in pl.range(62):
            _future: pl.Tile[[1, 1024], pl.FP32, PREFETCHED[(i + 2) % 3], pl.Mem.Vec] = pl.load(
                a, [i + 2, 0], [1, 1024]
            )
            current: pl.Tile[[1, 1024], pl.FP32, PREFETCHED[i % 3], pl.Mem.Vec] = pl.tile.create(
                [1, 1024], pl.FP32, target_memory=pl.Mem.Vec
            )
            y: pl.Tile[[1, 1024], pl.FP32, PREFETCHED[i % 3], pl.Mem.Vec] = pl.add(current, 1.0)
            output = pl.store(y, [i, 0], output)
        penultimate: pl.Tile[[1, 1024], pl.FP32, PREFETCHED[2], pl.Mem.Vec] = pl.tile.create(
            [1, 1024], pl.FP32, target_memory=pl.Mem.Vec
        )
        output = pl.store(pl.add(penultimate, 1.0), [62, 0], output)
        last: pl.Tile[[1, 1024], pl.FP32, PREFETCHED[0], pl.Mem.Vec] = pl.tile.create(
            [1, 1024], pl.FP32, target_memory=pl.Mem.Vec
        )
        output = pl.store(pl.add(last, 1.0), [63, 0], output)
        return output


@pl.program
class ReverseConstantSlots:
    """The first tile names slot 1; its address is not the allocation base."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        first: pl.Tile[[64, 64], pl.FP32, REVERSE_SLOTS[1], pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
        second: pl.Tile[[64, 64], pl.FP32, REVERSE_SLOTS[0], pl.Mem.Vec] = pl.exp(first)
        return pl.store(second, [0, 0], output)


@pl.program
class RotatingSlot:
    """The ping-pong: one allocation, the slot alternating per iteration."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[256, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        seed: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Vec)
        for i, (acc_i,) in pl.range(4, init_values=(seed,)):
            t: pl.Tile[[64, 64], pl.FP32, ROTATING[i % 2], pl.Mem.Vec] = pl.load(
                a, [i * 64, 0], [64, 64], target_memory=pl.MemorySpace.Vec
            )
            acc_next: pl.Tile[[64, 64], pl.FP32] = pl.add(acc_i, t)
            r = pl.yield_(acc_next)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(r, [0, 0], output)
        return out


@pl.program
class ConstantSlots:
    """Two constant slots of one allocation — no rotation, still one region."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        t0: pl.Tile[[64, 64], pl.FP32, CONST_SLOTS[0], pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
        t1: pl.Tile[[64, 64], pl.FP32, CONST_SLOTS[1], pl.Mem.Vec] = pl.exp(t0)
        return pl.store(t1, [0, 0], output)


@pl.program
class TooManySlots:
    """17 slots — one past ptoas's `multi_tile_buf` bound.

    Small tiles on purpose: 17 slots must still fit the Vec budget, so that the
    PyPTO planner accepts the program and the only thing under test is ptoas's
    slot-count bound.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[32, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        t0: pl.Tile[[32, 32], pl.FP32, TOO_MANY[0], pl.Mem.Vec] = pl.load(a, [0, 0], [32, 32])
        t1: pl.Tile[[32, 32], pl.FP32, TOO_MANY[1], pl.Mem.Vec] = pl.exp(t0)
        return pl.store(t1, [0, 0], output)


@pl.program
class MixedSlotShapes:
    """Slots holding differently shaped tiles — ptoas slots are uniform."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        big_out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        small_out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> tuple[pl.Tensor[[64, 64], pl.FP32], pl.Tensor[[32, 32], pl.FP32]]:
        big: pl.Tile[[64, 64], pl.FP32, MIXED_SHAPES[0], pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
        r_big: pl.Tensor[[64, 64], pl.FP32] = pl.store(big, [0, 0], big_out)
        small: pl.Tile[[32, 32], pl.FP32, MIXED_SHAPES[1], pl.Mem.Vec] = pl.load(a, [0, 0], [32, 32])
        r_small: pl.Tensor[[32, 32], pl.FP32] = pl.store(small, [0, 0], small_out)
        return r_big, r_small


@pl.program
class MixedSlotValidShapes:
    """Slots of one physical shape but different valid extents.

    The tile_buf type string renders `v_row=?, v_col=?` by design, so these two
    slots print identically — yet the region states one valid extent for both, and
    the second slot would silently take the first's.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        t0: pl.Tile[[64, 64], pl.FP32, MIXED_VALID[0], pl.Mem.Vec] = pl.load(
            a, [0, 0], [64, 64], valid_shape=[64, 64], target_memory=pl.MemorySpace.Vec
        )
        t1: pl.Tile[[64, 64], pl.FP32, MIXED_VALID[1], pl.Mem.Vec] = pl.load(
            a, [0, 0], [64, 64], valid_shape=[32, 64], target_memory=pl.MemorySpace.Vec
        )
        t2: pl.Tile[[64, 64], pl.FP32] = pl.add(t0, t1)
        return pl.store(t2, [0, 0], output)


@pl.program
class CoLiveSlotsInLoop:
    """Two slots of one allocation live at the same time inside a loop.

    ptoas 0.54 left the second load unguarded against the next iteration's write
    (hw-native-sys/PTOAS#1118, fixed in 0.56). The pinned ptoas mis-syncs the
    prefetch form of this shape (hw-native-sys/PTOAS#1519: fixed in 0.63, broken again
    since 0.64), and codegen cannot tell the two apart, so the region is refused rather
    than miscompiled — the ping-pong the region form accelerates takes one slot per
    iteration.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[256, 64], pl.FP32],
        b: pl.Tensor[[256, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
    ) -> pl.Tensor[[256, 64], pl.FP32]:
        for i in pl.range(4):
            lo: pl.Tile[[64, 64], pl.FP32, CO_LIVE[i % 2], pl.Mem.Vec] = pl.load(
                a, [i * 64, 0], [64, 64], target_memory=pl.MemorySpace.Vec
            )
            hi: pl.Tile[[64, 64], pl.FP32, CO_LIVE[(i + 1) % 2], pl.Mem.Vec] = pl.load(
                b, [i * 64, 0], [64, 64], target_memory=pl.MemorySpace.Vec
            )
            s: pl.Tile[[64, 64], pl.FP32] = pl.add(lo, hi)
            output = pl.store(s, [i * 64, 0], output)
        return output


@pl.program
class PrefetchSlotInLoop:
    """The prefetch form: each iteration fills slot ``(i + 1) % 2`` and reads ``i % 2``.

    Block 0 is preloaded into slot 0; ``cur`` binds the slot the previous iteration
    filled without writing it. This is the form hw-native-sys/PTOAS#1519 is about —
    ptoas 0.64 and 0.65 hand slot 0 a spare token, so a prefetch overwrites a slot
    before its read: wrong data for an even trip count, a device hang for an odd one.

    ``pre`` and ``nxt`` are never read as values — they only fill storage — so the
    case depends on those loads reaching codegen. It is left out of
    ``test_pypto_planner_accepts_them_all``: under the PyPTO planner it compiles, but
    no dependency links ``cur`` to the loads, so ptoas inserts no sync for it.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[320, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
    ) -> pl.Tensor[[256, 64], pl.FP32]:
        pre: pl.Tile[[64, 64], pl.FP32, PREFETCH[0], pl.Mem.Vec] = pl.load(  # noqa: F841
            a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Vec
        )
        for i in pl.range(4):
            nxt: pl.Tile[[64, 64], pl.FP32, PREFETCH[(i + 1) % 2], pl.Mem.Vec] = pl.load(  # noqa: F841
                a, [(i + 1) * 64, 0], [64, 64], target_memory=pl.MemorySpace.Vec
            )
            cur: pl.Tile[[64, 64], pl.FP32, PREFETCH[i % 2], pl.Mem.Vec] = pl.tile.create(
                [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            s: pl.Tile[[64, 64], pl.FP32] = pl.add(cur, cur)
            output = pl.store(s, [i * 64, 0], output)
        return output


@pl.program
class SequentialSlotsInSiblingLoops:
    """One slot per loop body, in two sibling loops — never live together.

    The blocker is per loop body, not per function: two slots that cannot be live
    at the same time still describe a region.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[256, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
    ) -> pl.Tensor[[256, 64], pl.FP32]:
        for i in pl.range(2):
            lo: pl.Tile[[64, 64], pl.FP32, SIBLING_LOOPS[i % 2], pl.Mem.Vec] = pl.load(
                a, [i * 64, 0], [64, 64], target_memory=pl.MemorySpace.Vec
            )
            s0: pl.Tile[[64, 64], pl.FP32] = pl.exp(lo)
            output = pl.store(s0, [i * 64, 0], output)
        for j in pl.range(2):
            hi: pl.Tile[[64, 64], pl.FP32, SIBLING_LOOPS[j % 2], pl.Mem.Vec] = pl.load(
                a, [128 + j * 64, 0], [64, 64], target_memory=pl.MemorySpace.Vec
            )
            s1: pl.Tile[[64, 64], pl.FP32] = pl.exp(hi)
            output = pl.store(s1, [128 + j * 64, 0], output)
        return output


@pl.program
class RuntimeValidShapeSlots:
    """Slots whose valid extent is only known at runtime.

    The region is declared in the function head, where a runtime extent's SSA value
    is not yet in scope — so there is no extent to state for the slots at all.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        rows: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        t0: pl.Tile[[64, 64], pl.FP32, RUNTIME_VALID[0], pl.Mem.Vec] = pl.load(
            a, [0, 0], [64, 64], valid_shape=[rows, 64], target_memory=pl.MemorySpace.Vec
        )
        t1: pl.Tile[[64, 64], pl.FP32, RUNTIME_VALID[1], pl.Mem.Vec] = pl.load(
            a, [0, 0], [64, 64], valid_shape=[rows, 64], target_memory=pl.MemorySpace.Vec
        )
        t2: pl.Tile[[64, 64], pl.FP32] = pl.add(t0, t1)
        return pl.store(t2, [0, 0], output)


@pl.program
class UnsubscriptedBinding:
    """One tile takes a slot, another binds the allocation whole."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        t0: pl.Tile[[64, 64], pl.FP32, MIXED_BINDING[0], pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
        t1: pl.Tile[[64, 64], pl.FP32, MIXED_BINDING, pl.Mem.Vec] = pl.exp(t0)
        return pl.store(t1, [0, 0], output)


@pl.program
class SingleSlotDeclaration:
    """An unsubscripted declaration is one buffer, not a region."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        t0: pl.Tile[[64, 64], pl.FP32, SINGLE, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
        return pl.store(t0, [0, 0], output)


def _mark_lowered_slot_producers(program: ir.Program) -> ir.Program:
    """Give hand-written scheduling fixtures the provenance emitted by real lowering."""
    producers = {ir.get_op("tile.load").name, ir.get_op("tile.create").name}

    class MarkProducers(ir.IRMutator):
        def visit_call(self, op: ir.Call) -> ir.Expr:
            call = super().visit_call(op)
            if not isinstance(call, ir.Call) or call.op.name not in producers:
                return call
            return ir.Call(
                call.op,
                call.args,
                call.kwargs,
                {**call.attrs, "software_pipeline_slots": True},
                call.type,
                call.span,
            )

    return MarkProducers().visit_program(program)


def _codegen(
    program: ir.Program,
    planner: passes.MemoryPlanner,
    *,
    enable_software_pipeline: bool = False,
    simulate_lowered_slots: bool = False,
) -> str:
    """Compile ``program``'s InCore kernel under ``planner`` and emit its PTO IR."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext(
        [], memory_planner=planner, enable_buffer_ir=False, enable_software_pipeline=enable_software_pipeline
    ):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program)
    if simulate_lowered_slots:
        optimized = _mark_lowered_slot_producers(optimized)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    single = ir.Program([func], "kernel", optimized.span)
    # The planner also decides the address mode: PTOAS plans (no addr, level2),
    # PyPTO bakes (addr, level3) — same pairing compile() applies.
    emit_tile_addr = planner == passes.MemoryPlanner.PYPTO
    return codegen.PTOCodegen().generate(single, emit_tile_addr=emit_tile_addr)


def _lines(mlir: str, needle: str) -> list[str]:
    return [line.strip() for line in mlir.splitlines() if needle in line]


def _assert_logical_slot_index(mlir: str) -> None:
    """Each selected slot is the corrected ``iv % 2``, without byte scaling."""
    definitions = {
        name: expression.split(" loc(", 1)[0]
        for name, expression in re.findall(r"^\s*(%\w+) = (.+)$", mlir, re.MULTILINE)
    }
    loop = re.search(r"scf\.for (%\w+) =", mlir)
    assert loop is not None, mlir
    gets = _lines(mlir, "pto.multi_tile_get")
    assert gets, mlir
    for get in gets:
        slot = get.split("[")[1].split("]")[0]
        # Two shapes are correct, and which one appears is a property of the
        # loop, not of this pass. FloorMod follows the divisor's sign while
        # `remsi` follows the dividend's, so a negative dividend needs the
        # rem/cmp/add/select correction. When the induction variable cannot be
        # negative — every loop here starts at a non-negative literal — the
        # remainder is already in [0, divisor) and codegen emits the bare
        # `remsi`, because that correction would be dead code. Asserting only
        # the corrected form would pin an optimisation *away*.
        selected = re.fullmatch(r"arith\.select (%\w+), (%\w+), (%\w+) : index", definitions.get(slot, ""))
        if selected is not None:
            condition, corrected, remainder = selected.groups()
            assert definitions[corrected].startswith("arith.addi "), mlir
            comparison = re.fullmatch(
                rf"arith\.cmpi slt, {re.escape(remainder)}, (%\w+) : index", definitions[condition]
            )
            assert comparison is not None, f"negative remainders must select the corrected value:\n{mlir}"
            assert definitions[comparison.group(1)] == "arith.constant 0 : index", mlir
        else:
            remainder = slot
        division = re.fullmatch(r"arith\.remsi (%\w+), (%\w+) : index", definitions.get(remainder, ""))
        assert division is not None, f"slot {slot} must depend on a remainder:\n{mlir}"
        dividend, divisor = division.groups()
        assert dividend == loop.group(1), f"the slot must depend on the loop index:\n{mlir}"
        assert definitions[divisor] == "arith.constant 2 : index", mlir
        if selected is not None:
            assert definitions[corrected] == f"arith.addi {remainder}, {divisor} : index", mlir
        # The chain is rem, optionally over cmp/add/select. A scaled byte offset
        # cannot reach multi_tile_get through any of these checked operands.


_INDEX_OPS = {
    "addi": lambda a, b: a + b,
    "subi": lambda a, b: a - b,
    "muli": lambda a, b: a * b,
    "divsi": lambda a, b: int(a / b),
    "divui": lambda a, b: a // b,
    "floordivsi": lambda a, b: a // b,
    "remsi": lambda a, b: int(math.fmod(a, b)),
    "remui": lambda a, b: a % b,
}


def _definition(mlir: str, ssa: str) -> str:
    line = next((ln.strip() for ln in mlir.splitlines() if ln.strip().startswith(f"{ssa} =")), None)
    assert line is not None, f"{ssa} has no definition:\n{mlir}"
    return line


# Comparison predicates reachable from the corrected-remainder sequence below.
_INDEX_CMP = {
    "slt": lambda a, b: a < b,
    "sle": lambda a, b: a <= b,
    "sgt": lambda a, b: a > b,
    "sge": lambda a, b: a >= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}


def _operands(rest: str) -> list[str]:
    """The operand SSA names of an arith op, with the trailing type annotation dropped."""
    return [x.strip() for x in rest.split(":")[0].split(",")]


def _eval_index(mlir: str, ssa: str, env: dict[str, int]) -> int:
    """Evaluate an index SSA value by walking its arith definitions back to constants.

    ``env`` binds SSA names that have no definition line (the loop's induction
    variable). An op outside the tables below fails loudly, because the slot chain
    must stay plain index arithmetic.

    `cmpi`/`select` are in scope because a signed floor-mod does not lower to a
    bare `remsi`: #3030 emits the corrected sequence rem/cmp/add/select, so that
    `-1 % 2` is 1 rather than C's -1. Evaluating it here keeps this test checking
    the *slot sequence* rather than the spelling of the remainder.
    """
    if ssa in env:
        return env[ssa]
    op, rest = _definition(mlir, ssa).split("=", 1)[1].strip().split(None, 1)
    if op == "arith.constant":
        return int(rest.split(":")[0].strip())
    name = op.removeprefix("arith.")
    if name == "select":
        condition, when_true, when_false = _operands(rest)
        chosen = when_true if _eval_index(mlir, condition, env) else when_false
        return _eval_index(mlir, chosen, env)
    if name == "cmpi":
        predicate, lhs, rhs = _operands(rest)
        assert predicate in _INDEX_CMP, f"unexpected predicate in the slot chain: {predicate} ({ssa})"
        return int(_INDEX_CMP[predicate](_eval_index(mlir, lhs, env), _eval_index(mlir, rhs, env)))
    assert name in _INDEX_OPS, f"unexpected op in the slot chain: {op} ({ssa})"
    lhs, rhs = _operands(rest)
    return _INDEX_OPS[name](_eval_index(mlir, lhs, env), _eval_index(mlir, rhs, env))


def _slot_chain_ops(mlir: str, ssa: str, env: dict[str, int]) -> list[str]:
    """The arith ops (excluding constants) on the path from ``ssa`` back to ``env``.

    A `cmpi`'s first operand is its predicate keyword, not an SSA name, so it is
    skipped; every other operand is followed.
    """
    if ssa in env:
        return []
    op, rest = _definition(mlir, ssa).split("=", 1)[1].strip().split(None, 1)
    if op == "arith.constant":
        return []
    name = op.removeprefix("arith.")
    operands = _operands(rest)
    if name == "cmpi":
        operands = operands[1:]
    ops = [name]
    for operand in operands:
        ops += _slot_chain_ops(mlir, operand, env)
    return ops


def _pipelined_kernel(start: int, stop: int, step: int, stage: int) -> ir.Program:
    """InCore kernel with a loop-invariant load, so any bounds compile."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            seed: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Vec)
            for i, (acc_i,) in pl.pipeline(start, stop, step, stage=stage, init_values=(seed,)):
                t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Vec
                )
                acc_next: pl.Tile[[64, 64], pl.FP32] = pl.add(acc_i, t)
                r = pl.yield_(acc_next)
            out: pl.Tensor[[64, 64], pl.FP32] = pl.store(r, [0, 0], output)
            return out

    return Program


def _slot_parents(mlir: str) -> dict[str, str]:
    """Follow selected slot handles and their reshape aliases to the allocation."""
    parents = {}
    for line in mlir.splitlines():
        selected = re.search(r"(%\w+) = pto\.(?:subview|multi_tile_get) (%\w+)\[", line)
        if selected:
            parents[selected[1]] = selected[2]
        reshaped = re.search(r"(%\w+) = pto\.treshape (%\w+) :", line)
        if reshaped and reshaped[2] in parents:
            parents[reshaped[1]] = parents[reshaped[2]]
    return parents


def _tile_memrefs(program: ir.Program) -> dict[str, ir.MemRef]:
    """Map every TileType assignment's var name to its MemRef."""
    found: dict[str, ir.MemRef] = {}

    def walk(stmt):
        if stmt is None:
            return
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.var.type, ir.TileType):
            memref = stmt.var.type.memref
            if memref is not None:
                found[stmt.var.name_hint] = memref
        for attr in ("body", "then_body", "else_body", "stmts"):
            sub = getattr(stmt, attr, None)
            if sub is None:
                continue
            for child in sub if isinstance(sub, (list, tuple)) else [sub]:
                walk(child)

    for func in program.functions.values():
        walk(func.body)
    return found


class TestPtoasPlannerEmitsMultiBuffer:
    """Under the ptoas planner a slotted declaration becomes one region."""

    def test_rotating_slot_becomes_one_region_read_by_slot(self):
        """N slots are one `alloc_multi_tile`; the use selects its slot."""
        mlir = _codegen(RotatingSlot, passes.MemoryPlanner.PTOAS)

        allocs = _lines(mlir, "pto.alloc_multi_tile")
        assert len(allocs) == 1, f"expected exactly one multi-buffer region:\n{mlir}"
        assert "count=2" in allocs[0], f"the region must carry the declared slot count:\n{allocs[0]}"
        # ptoas PlanMemory owns the placement — an addr operand is rejected at level2.
        assert "addr" not in allocs[0], f"the region must not carry an address:\n{allocs[0]}"

        gets = _lines(mlir, "pto.multi_tile_get")
        assert gets, f"expected the slot to be read from the region:\n{mlir}"
        region = allocs[0].split("=")[0].strip()
        assert all(region in get for get in gets), f"every slot must come from {region}:\n{mlir}"

    def test_rotating_slot_index_reaches_ptoas_as_an_index(self):
        """The slot operand is the index itself, not the byte offset.

        ptoas matches the slot's affine form (`iv % N`) to decide whether two
        accesses can touch the same slot, and that is what earns the per-slot
        event ids. Handing it a byte offset (`i % 2 * 16384`) would defeat it.
        """
        mlir = _codegen(RotatingSlot, passes.MemoryPlanner.PTOAS)
        _assert_logical_slot_index(mlir)

    def test_constant_slots_share_one_region(self):
        """Constant slots are the same story — one region, two selections."""
        mlir = _codegen(ConstantSlots, passes.MemoryPlanner.PTOAS)
        assert len(_lines(mlir, "pto.alloc_multi_tile")) == 1, mlir
        gets = _lines(mlir, "pto.multi_tile_get")
        assert len(gets) == 2, f"expected one get per slot:\n{mlir}"
        slots = {get.split("[")[1].split("]")[0] for get in gets}
        assert len(slots) == 2, f"the two slots must be distinct:\n{mlir}"

    def test_sibling_loops_each_taking_one_slot_still_form_a_region(self):
        """The co-live blocker is per loop body, not per function.

        Two slots that can never be live together — one per sibling loop — still
        describe a region; only slots co-live in one iteration are refused.
        """
        mlir = _codegen(SequentialSlotsInSiblingLoops, passes.MemoryPlanner.PTOAS)
        assert len(_lines(mlir, "pto.alloc_multi_tile")) == 1, mlir
        assert len(_lines(mlir, "pto.multi_tile_get")) == 2, mlir

    def test_slot_tiles_take_no_alloc_tile(self):
        """A slot is taken from the region, never allocated beside it."""
        mlir = _codegen(ConstantSlots, passes.MemoryPlanner.PTOAS)
        get_handles = {get.split("=")[0].strip() for get in _lines(mlir, "pto.multi_tile_get")}
        allocated = {ln.split("=")[0].strip() for ln in _lines(mlir, "pto.alloc_tile")}
        assert not (get_handles & allocated), f"a slot handle must have exactly one definition:\n{mlir}"


class TestFallbacks:
    """Everything the ptoas multi-buffer form does not cover keeps alloc_tile."""

    def test_pypto_planner_keeps_baked_addresses(self):
        """Without the opt-in flag, level3 retains its original allocation path."""
        mlir = _codegen(RotatingSlot, passes.MemoryPlanner.PYPTO)
        assert not _lines(mlir, "pto.alloc_multi_tile"), f"level3 must not use a region:\n{mlir}"
        assert not _lines(mlir, "pto.multi_tile_get"), f"level3 must not use a region:\n{mlir}"
        assert _lines(mlir, "pto.alloc_tile"), f"expected the ordinary alloc path:\n{mlir}"

    def test_single_slot_declaration_is_still_rejected(self):
        """One slot has no ptoas counterpart, so its isolation cannot be kept.

        ptoas requires a count of at least 2, so a single-slot declaration cannot
        become a region — and a plain `alloc_tile` would leave ptoas free to pack
        the buffer the author separated. It stays rejected, now pointing at
        `slots=N` as the supported spelling.
        """
        with pytest.raises(ValueError, match="single-slot declared allocation"):
            _codegen(SingleSlotDeclaration, passes.MemoryPlanner.PTOAS)

    def test_single_slot_declaration_still_works_under_pypto(self):
        """Nothing about the PyPTO planner's handling of declarations changed."""
        mlir = _codegen(SingleSlotDeclaration, passes.MemoryPlanner.PYPTO)
        assert not _lines(mlir, "pto.alloc_multi_tile"), f"a single slot is not a region:\n{mlir}"
        assert _lines(mlir, "pto.alloc_tile"), f"expected the ordinary alloc path:\n{mlir}"


class TestFixedAddressMultiBuffer:
    @pytest.mark.parametrize("stage", [2, 3, 4])
    def test_compute_chain_reuses_one_input_region_through_the_store(self, stage):
        """Last-use compute chains share an input region without extra UB storage."""

        @pl.program
        class BinaryChain:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[65, 1024], pl.FP32],
                b: pl.Tensor[[65, 1024], pl.FP32],
                output: pl.Out[pl.Tensor[[65, 1024], pl.FP32]],
            ) -> pl.Tensor[[65, 1024], pl.FP32]:
                for i in pl.pipeline(65, stage=stage):
                    lhs = pl.load(a, [i, 0], [1, 1024])
                    rhs = pl.load(b, [i, 0], [1, 1024])
                    intermediate = pl.add(lhs, rhs)
                    result = pl.add(intermediate, 1.0)
                    output = pl.store(result, [i, 0], output)
                return output

        mlir = _codegen(BinaryChain, passes.MemoryPlanner.PYPTO, enable_software_pipeline=True)
        assert not _lines(mlir, "pto.alloc_multi_tile"), "Co-live slots require parent subviews"
        regions = _lines(mlir, "pto.alloc_tile addr =")
        assert len(regions) == 2, mlir
        assert all(f"rows={stage}" in line and "cols=1024" in line for line in regions), mlir
        # Two input streams retain exactly stage-1 preloads plus one steady load.
        assert len(_lines(mlir, "pto.tload")) == 2 * stage, mlir
        assert len(_lines(mlir, "pto.tstore")) == stage, mlir
        handle_regions = _slot_parents(mlir)
        input_regions = {
            handle_regions[line.split("outs(")[1].split(" :")[0].strip()]
            for line in _lines(mlir, "pto.tload")
        }
        stored_handles = [line.split("ins(")[1].split(" :")[0].strip() for line in _lines(mlir, "pto.tstore")]
        output_regions = {handle_regions[handle] for handle in stored_handles}
        assert len(input_regions) == 2 and len(output_regions) == 1, mlir
        assert output_regions <= input_regions, mlir
        additions = _lines(mlir, "pto.tadds ")
        assert len(additions) == len(stored_handles), mlir
        for addition, stored in zip(additions, stored_handles, strict=True):
            intermediate = addition.split("ins(")[1].split(",")[0].strip()
            result = addition.split("outs(")[1].split(" :")[0].strip()
            assert intermediate == result, "Last-use compute must retain the same slot handle"
            assert result == stored, "The store must consume the exact current output-slot handle"

        constants = {
            name: int(value) for name, value in re.findall(r"(%\w+) = arith.constant (\d+) : i64", mlir)
        }
        region_ranges = []
        for region in regions:
            address = constants[region.split("addr = ")[1].split()[0]]
            region_ranges.append((address, address + stage * 4096))
        region_ranges.sort()
        assert all(first[1] <= second[0] for first, second in zip(region_ranges, region_ranges[1:]))
        assert len(_lines(mlir, "pto.alloc_tile")) == 2, "The chain needs no independent temporary"
        assert max(end for _, end in region_ranges) == 2 * stage * 4096

    def test_two_inputs_have_distinct_fixed_regions_and_inplace_compute(self):
        """Exercise compiler slots at a nonzero base and keep ordinary modulo signed."""

        @pl.program
        class TwoInputs:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 1024], pl.FP32],
                b: pl.Tensor[[64, 1024], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 1024], pl.FP32]],
            ) -> pl.Tensor[[64, 1024], pl.FP32]:
                for i in pl.pipeline(64, stage=3):
                    row = (i + 1) % 64
                    lhs = pl.load(a, [i, 0], [1, 1024])
                    rhs = pl.load(b, [row, 0], [1, 1024])
                    result = pl.add(lhs, rhs)
                    output = pl.store(result, [i, 0], output)
                return output

        mlir = _codegen(TwoInputs, passes.MemoryPlanner.PYPTO, enable_software_pipeline=True)
        assert not _lines(mlir, "pto.alloc_multi_tile"), "Co-live slots require parent subviews"
        regions = _lines(mlir, "pto.alloc_tile addr =")
        assert len(regions) == 2, mlir
        assert all("rows=3" in line and "cols=1024" in line for line in regions), mlir
        addresses = {line.split("addr = ")[1].split()[0] for line in regions}
        assert addresses == {"%c0_i64", "%c12288_i64"}, mlir
        assert len(_lines(mlir, "pto.tload")) == 6, mlir
        assert len(_lines(mlir, "pto.tadd ")) == 3, mlir
        assert _lines(mlir, "arith.remui"), mlir
        assert _lines(mlir, "arith.remsi"), mlir
        handle_regions = _slot_parents(mlir)
        for line in _lines(mlir, "pto.tadd "):
            first_input = line.split("ins(")[1].split(",")[0].strip()
            output = line.split("outs(")[1].split(" :")[0].strip()
            assert handle_regions[first_input] == handle_regions[output], line
            assert first_input == output, "In-place slot compute must preserve PTOAS slot SSA identity"

    def test_static_prefetch_uses_parent_subviews_and_unsigned_slot_indices(self):
        mlir = _codegen(
            PrefetchedSlots,
            passes.MemoryPlanner.PYPTO,
            enable_software_pipeline=True,
            simulate_lowered_slots=True,
        )
        assert not _lines(mlir, "pto.alloc_multi_tile"), "PTOAS #1519 also affects static prefetch"
        assert not _lines(mlir, "pto.multi_tile_get")
        parents = set(_slot_parents(mlir).values())
        regions = [line for line in _lines(mlir, "pto.alloc_tile addr =") if line.split(" = ")[0] in parents]
        assert len(regions) == 1, mlir
        assert "addr = %c0_i64" in regions[0], mlir
        assert "rows=3, cols=1024" in regions[0], mlir
        assert "valid_row =" not in regions[0], mlir
        assert len(_lines(mlir, "scf.for")) == 1, mlir
        assert len(_lines(mlir, "pto.tload")) == 3, mlir
        assert len(_lines(mlir, "pto.tstore")) == 3, mlir
        assert len(_lines(mlir, "arith.remui")) >= 2, mlir
        assert not _lines(mlir, "arith.remsi"), mlir
        gets = _lines(mlir, "pto.subview")
        handles = {line.split("=")[0].strip() for line in gets}
        assert len(gets) >= 6, mlir
        assert not (handles & {line.split("=")[0].strip() for line in _lines(mlir, "pto.alloc_tile")})
        steady_add = _lines(mlir, "pto.tadds ")[0]
        first_input = steady_add.split("ins(")[1].split(",")[0].strip()
        output = steady_add.split("outs(")[1].split(" :")[0].strip()
        assert first_input == output, "Duplicate same-slot gets prevent PTOAS from deriving overlap"

    def test_first_selected_slot_is_not_used_as_the_region_base(self):
        mlir = _codegen(
            ReverseConstantSlots,
            passes.MemoryPlanner.PYPTO,
            enable_software_pipeline=True,
            simulate_lowered_slots=True,
        )
        region = _lines(mlir, "pto.alloc_multi_tile")
        assert len(region) == 1, mlir
        assert "addr = %c0_i64" in region[0], mlir
        assert "count=2" in region[0], mlir
        gets = _lines(mlir, "pto.multi_tile_get")
        assert len(gets) == 2, "A different output slot must retain its own handle"
        assert "[%c1_index]" in gets[0], mlir

    @pytest.mark.parametrize("program", [MixedSlotShapes, MixedSlotValidShapes, RuntimeValidShapeSlots])
    def test_unsupported_region_keeps_the_existing_addressed_path(self, program):
        # Isolate codegen's region rejection; this does not claim the scheduler
        # accepts these deliberately unsupported hand-written shapes.
        mlir = _codegen(
            program,
            passes.MemoryPlanner.PYPTO,
            enable_software_pipeline=True,
            simulate_lowered_slots=True,
        )
        assert not _lines(mlir, "pto.alloc_multi_tile"), mlir
        assert _lines(mlir, "pto.alloc_tile"), mlir

    @pytest.mark.parametrize("software", [False, True], ids=["default", "software"])
    def test_fallback_slot_allocation_keeps_mutable_valid_shape(self, software, tmp_path):
        """A rejected region retains the mutable descriptor of an ordinary allocation."""

        @pl.program
        class NarrowedSlot:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                rows: pl.Scalar[pl.INDEX],
                cols: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile: pl.Tile[[32, 32], pl.FP32, TOO_MANY[0], pl.Mem.Vec] = pl.load(a, [0, 0], [32, 32])
                narrowed = pl.tile.set_validshape(tile, rows, cols)
                return pl.store(narrowed, [0, 0], out)

        # Seventeen slots force the ordinary addressed fallback even with the
        # optimization enabled; that handle must remain valid for set_validshape.
        mlir = _codegen(
            NarrowedSlot,
            passes.MemoryPlanner.PYPTO,
            enable_software_pipeline=software,
            simulate_lowered_slots=software,
        )
        assert not _lines(mlir, "pto.alloc_multi_tile"), mlir
        allocations = _lines(mlir, "pto.alloc_tile")
        updates = _lines(mlir, "pto.set_validshape")
        assert len(allocations) == len(updates) == 1, mlir
        assert all("v_row=?" in line and "v_col=?" in line for line in allocations + updates), mlir

        if find_ptoas_binary() is None:
            pytest.skip("PTOAS is not available")
        source = tmp_path / "mutable_slot.pto"
        output = tmp_path / "mutable_slot.cpp"
        source.write_text(mlir)
        _run_ptoas(str(source), str(output), ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync"])
        assert output.is_file()

    def test_default_prefetch_codegen_still_uses_individual_allocations(self):
        mlir = _codegen(PrefetchedSlots, passes.MemoryPlanner.PYPTO)
        assert not _lines(mlir, "pto.alloc_multi_tile"), mlir
        assert _lines(mlir, "arith.remsi"), mlir

    def test_slot_proofs_do_not_leak_into_the_next_buffer_function(self):
        """A shared IR expression has different bounds in each function."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        with passes.PassContext([], enable_buffer_ir=False, enable_software_pipeline=True):
            optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(PrefetchedSlots)
        optimized = _mark_lowered_slot_producers(optimized)
        shared_mod = next(
            ref.slot_index_
            for ref in _tile_memrefs(optimized).values()
            if isinstance(ref.slot_index_, ir.FloorMod) and isinstance(ref.slot_index_.left, ir.Var)
        )
        assert isinstance(shared_mod.left, ir.Var)
        span = ir.Span.unknown()
        value = ir.Var("value", ir.ScalarType(DataType.INDEX), span)
        scalar = ir.Function(
            "buffer_scalar",
            [shared_mod.left],
            [],
            ir.AssignStmt(value, shared_mod, span),
            span,
            type=ir.FunctionType.InCore,
            ir_stage=ir.FunctionIRStage.Buffer,
        )
        buffer_program = ir.Program([scalar], "BufferScalar", span)
        tile = next(func for func in optimized.functions.values() if func.name == "kernel")
        generator = codegen.PTOCodegen()
        assert "arith.remui" in generator.generate(ir.Program([tile], "TilePipeline", span))
        reused = generator.generate(buffer_program)
        fresh = codegen.PTOCodegen().generate(buffer_program)
        assert reused == fresh
        assert "arith.remui" not in reused
        assert "arith.select" in reused

    @pytest.mark.parametrize("narrowed", ["numerator", "divisor"])
    def test_non_index_slot_arithmetic_keeps_the_ordinary_addressed_path(self, narrowed, tmp_path):
        """An INDEX result does not prove that its operand arithmetic cannot narrow."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        with passes.PassContext([], enable_buffer_ir=False, enable_software_pipeline=True):
            optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(PrefetchedSlots)

        class NarrowSlotArithmetic(ir.IRMutator):
            def visit_floor_mod(self, op: ir.FloorMod) -> ir.Expr:
                mod = super().visit_floor_mod(op)
                if not isinstance(mod, ir.FloorMod):
                    return mod
                if narrowed == "numerator" and isinstance(mod.left, ir.Add):
                    left = ir.Add(mod.left.left, mod.left.right, DataType.INT32, mod.left.span)
                    return ir.FloorMod(left, mod.right, DataType.INDEX, mod.span)
                if narrowed == "divisor" and isinstance(mod.right, ir.ConstInt):
                    right = ir.ConstInt(mod.right.value, DataType.INT32, mod.right.span)
                    return ir.FloorMod(mod.left, right, DataType.INDEX, mod.span)
                return mod

        # Simulate a lowered schedule, but retain its mixed-width arithmetic.
        # It must decline the region rather than emit an i32 SSA as INDEX remui.
        transformed = _mark_lowered_slot_producers(NarrowSlotArithmetic().visit_program(optimized))
        func = next(func for func in transformed.functions.values() if func.name == "kernel")
        single = ir.Program([func], "kernel", transformed.span)
        mlir = codegen.PTOCodegen().generate(single)
        assert not _lines(mlir, "pto.subview"), mlir
        assert not _lines(mlir, "pto.alloc_multi_tile"), mlir
        assert _lines(mlir, "pto.alloc_tile"), mlir
        assert _lines(mlir, "arith.remsi"), mlir
        if find_ptoas_binary() is None:
            pytest.skip("PTOAS is not available")
        source = tmp_path / "mixed_slot.pto"
        source.write_text(mlir)
        _run_ptoas(
            str(source),
            str(source.with_suffix(".cpp")),
            ["--pto-level=level3", "--pto-arch=a2", "--enable-insert-sync"],
        )


class TestUnsupportedShapesAreLoud:
    """A shape ptoas cannot describe is an error, never a quiet per-slot alloc.

    Falling back would leave ptoas free to plan the slots on top of each other —
    exactly the separation the declaration exists to state.
    """

    @pytest.mark.parametrize(
        ("program", "reason"),
        [
            (TooManySlots, "17"),
            (MixedSlotShapes, "differently shaped tiles"),
            (MixedSlotValidShapes, "different valid shapes"),
            (RuntimeValidShapeSlots, "runtime valid shape"),
            (CoLiveSlotsInLoop, "two of its slots are live at once inside a loop"),
            (PrefetchSlotInLoop, "two of its slots are live at once inside a loop"),
            (UnsubscriptedBinding, "without selecting a slot"),
        ],
        ids=[
            "slot-count-out-of-range",
            "non-uniform-slot-type",
            "non-uniform-valid-shape",
            "runtime-valid-shape",
            "co-live-slots-in-loop",
            "prefetch-slot-in-loop",
            "unsubscripted-binding",
        ],
    )
    def test_blocked_shape_names_itself(self, program, reason):
        with pytest.raises(ValueError, match=reason):
            _codegen(program, passes.MemoryPlanner.PTOAS)

    @pytest.mark.parametrize(
        "program",
        [
            TooManySlots,
            MixedSlotShapes,
            MixedSlotValidShapes,
            RuntimeValidShapeSlots,
            CoLiveSlotsInLoop,
            UnsubscriptedBinding,
        ],
        ids=[
            "slot-count-out-of-range",
            "non-uniform-slot-type",
            "non-uniform-valid-shape",
            "runtime-valid-shape",
            "co-live-slots-in-loop",
            "unsubscripted-binding",
        ],
    )
    def test_pypto_planner_accepts_them_all(self, program):
        """None of these are errors in themselves — only ptoas cannot describe them."""
        mlir = _codegen(program, passes.MemoryPlanner.PYPTO)
        assert _lines(mlir, "pto.alloc_tile"), f"expected the ordinary alloc path:\n{mlir}"


class TestSlotGeometrySurvivesInitMemRef:
    """The resolved MemRef still knows which slot of what it is."""

    def test_resolved_memref_keeps_slot_count_and_index(self):
        """InitMemRef resolves the index into an offset without erasing it."""
        after = passes.init_mem_ref()(ConstantSlots)
        slots = [mr for mr in _tile_memrefs(after).values() if mr.slot_count_ > 1]
        assert slots, "the declaration's slots must survive InitMemRef"
        assert all(mr.slot_index_ is not None for mr in slots), (
            "a resolved slot must still name the slot it selects"
        )
        # ...and the offset is still resolved, as before. A constant slot index
        # folds to a constant offset, which is what keeps the ordinary
        # static-address path unchanged for it.
        offsets = [mr.byte_offset_ for mr in slots]
        assert all(isinstance(off, ir.ConstInt) for off in offsets), (
            f"a constant slot index must fold to a constant offset, got {offsets}"
        )
        assert {off.value for off in offsets if isinstance(off, ir.ConstInt)} == {0, 64 * 64 * 4}

    def test_resolved_slot_round_trips_through_the_printer(self):
        """A post-InitMemRef dump reparses as the same slot, not a bare offset.

        Structural equality compares the slot fields, so dropping them on the way
        out would make the round trip report a MemRef mismatch — and codegen would
        read the reparsed program as N unrelated allocations.
        """
        after = passes.init_mem_ref()(ConstantSlots)
        dumped = after.as_python()
        assert "slots=2" in dumped, f"the printed MemRef must carry its slot count:\n{dumped}"
        ir.assert_structural_equal(pl.parse_program(dumped), after)


@pl.program
class PipelinedLoad:
    """``pl.pipeline`` with no hand-written declaration.

    ``LowerPipelineToSlots`` synthesizes the slotted allocation, so this is the
    compiler-generated counterpart of ``RotatingSlot`` above — the two must reach
    codegen in the same shape.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[256, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        seed: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Vec)
        for i, (acc_i,) in pl.pipeline(4, stage=2, init_values=(seed,)):
            t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                a, [i * 64, 0], [64, 64], target_memory=pl.MemorySpace.Vec
            )
            acc_next: pl.Tile[[64, 64], pl.FP32] = pl.add(acc_i, t)
            r = pl.yield_(acc_next)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(r, [0, 0], output)
        return out


@pl.program
class SteppedPipelinedLoad:
    """``pl.pipeline`` whose step is not 1.

    The induction variable is not the rotation counter here: over ``[0, 256)``
    with ``step = 64`` it only ever takes ``0/64/128/192``, so indexing slots by
    ``i % 2`` directly would select slot 0 on every iteration
    (``0, 0, 0, 0``) and lose the rotation. The rotation counter
    is ``(i - start) / step``, so the slot is ``((i - start) / step) % F`` — still
    one remainder over one SSA value, which is the property ptoas' slot accounting
    reads. Before this shape was admitted, every stepped pipeline fell through to
    ``LowerPipelineLoops`` and paid ``F`` body copies instead.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[256, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        seed: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Vec)
        for i, (acc_i,) in pl.pipeline(0, 256, 64, stage=2, init_values=(seed,)):
            t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                a, [i, 0], [64, 64], target_memory=pl.MemorySpace.Vec
            )
            acc_next: pl.Tile[[64, 64], pl.FP32] = pl.add(acc_i, t)
            r = pl.yield_(acc_next)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(r, [0, 0], output)
        return out


class TestPipelineLowersToARegion:
    """``pl.pipeline`` reaches codegen as one region, not as F copies of the body."""

    def test_emits_one_region_sized_to_the_stage_count(self):
        mlir = _codegen(PipelinedLoad, passes.MemoryPlanner.PTOAS)
        regions = _lines(mlir, "pto.alloc_multi_tile")
        assert len(regions) == 1, f"expected exactly one region:\n{mlir}"
        assert "count=2" in regions[0], regions[0]

    def test_takes_one_slot_per_iteration(self):
        """One `multi_tile_get` per iteration is the shape ptoas guards correctly."""
        mlir = _codegen(PipelinedLoad, passes.MemoryPlanner.PTOAS)
        assert len(_lines(mlir, "pto.multi_tile_get")) == 1, mlir

    def test_slot_operand_is_the_affine_index_not_a_byte_offset(self):
        """ptoas matches `i % 2` to earn the rotation its dynamic event ids."""
        mlir = _codegen(PipelinedLoad, passes.MemoryPlanner.PTOAS)
        _assert_logical_slot_index(mlir)

    def test_the_body_is_not_replicated(self):
        """One load in the loop, and the loop still strides by its original step."""
        mlir = _codegen(PipelinedLoad, passes.MemoryPlanner.PTOAS)
        # Two tloads total: the seed before the loop, and the one in the body.
        assert len(_lines(mlir, "pto.tload")) == 2, mlir
        loops = _lines(mlir, "scf.for")
        assert len(loops) == 1, mlir
        assert "step %c1_index" in loops[0], f"a slotted loop keeps its step:\n{loops[0]}"

    def test_pypto_planner_still_replicates(self):
        """The default planner emits no region and keeps the F-way replication."""
        mlir = _codegen(PipelinedLoad, passes.MemoryPlanner.PYPTO)
        assert _lines(mlir, "pto.alloc_multi_tile") == []
        assert len(_lines(mlir, "pto.tload")) == 3, f"seed + two replicated loads:\n{mlir}"
        assert "step %c2_index" in _lines(mlir, "scf.for")[0]

    def test_a_stepped_pipeline_also_reaches_one_region(self):
        """A non-unit step is a loop shape, not a reason to replicate the body."""
        mlir = _codegen(SteppedPipelinedLoad, passes.MemoryPlanner.PTOAS)
        regions = _lines(mlir, "pto.alloc_multi_tile")
        assert len(regions) == 1, f"expected exactly one region:\n{mlir}"
        assert "count=2" in regions[0], regions[0]
        assert len(_lines(mlir, "pto.multi_tile_get")) == 1, mlir

    def test_a_stepped_slot_is_the_counter_modulo_the_stage_count(self):
        """The slot is a remainder over the *counter*, never over the raw iv.

        ``i`` takes ``0/64/128/192`` here, so ``i % 2`` is ``0`` every time: the
        same slot each iteration, and no ping-pong. The counter ``(i - start) / step``
        is ``0, 1, 2, 3``, so the slot rotates ``0, 1, 0, 1``.
        """
        mlir = _codegen(SteppedPipelinedLoad, passes.MemoryPlanner.PTOAS)
        get = _lines(mlir, "pto.multi_tile_get")[0]
        slot_ssa = get.split("[")[1].split("]")[0]
        definition = next(
            (ln.strip() for ln in mlir.splitlines() if ln.strip().startswith(f"{slot_ssa} =")), None
        )
        assert definition is not None, f"slot operand {slot_ssa} has no definition:\n{mlir}"
        assert "remui" in definition or "remsi" in definition, (
            f"the slot operand must be the index expression, got: {definition}"
        )
        # The remainder is taken over a division (the counter), not over the iv.
        counter_ssa = definition.split("remui")[-1].split("remsi")[-1].split(",")[0].strip()
        counter_def = next(
            (ln.strip() for ln in mlir.splitlines() if ln.strip().startswith(f"{counter_ssa} =")), None
        )
        assert counter_def is not None, f"the counter {counter_ssa} has no definition:\n{mlir}"
        assert "divui" in counter_def or "divsi" in counter_def, (
            f"the rotation counter must be a division, got: {counter_def}"
        )
        assert str(64 * 64 * 4) not in definition, (
            f"the slot operand must not be scaled to a byte offset: {definition}"
        )

    def test_a_stepped_loop_keeps_its_own_step(self):
        """One body, and the loop still strides by the user's step."""
        mlir = _codegen(SteppedPipelinedLoad, passes.MemoryPlanner.PTOAS)
        loops = _lines(mlir, "scf.for")
        assert len(loops) == 1, mlir
        assert "step %c64_index" in loops[0], f"a slotted loop keeps its step:\n{loops[0]}"
        assert len(_lines(mlir, "pto.tload")) == 2, f"seed + one body load:\n{mlir}"

    def test_without_the_region_a_stepped_pipeline_is_replicated(self):
        """The contrast that motivates the shape: F body copies and a doubled step."""
        mlir = _codegen(SteppedPipelinedLoad, passes.MemoryPlanner.PYPTO)
        assert _lines(mlir, "pto.alloc_multi_tile") == []
        assert len(_lines(mlir, "pto.tload")) == 3, f"seed + two replicated loads:\n{mlir}"
        assert "step %c128_index" in _lines(mlir, "scf.for")[0]

    @pytest.mark.parametrize(
        "start, stop, step, stage, expected",
        [
            pytest.param(0, 4, 1, 2, [0, 1, 0, 1], id="unit-aligned"),
            pytest.param(1, 5, 1, 2, [0, 1, 0, 1], id="unit-unaligned"),
            pytest.param(1, 8, 1, 3, [0, 1, 2, 0, 1, 2, 0], id="unit-unaligned-3slots-7trips"),
            pytest.param(0, 256, 64, 2, [0, 1, 0, 1], id="stepped-zero-start"),
            pytest.param(64, 320, 64, 2, [0, 1, 0, 1], id="stepped-nonzero-start"),
            pytest.param(64, 384, 64, 3, [0, 1, 2, 0, 1], id="stepped-3slots-5trips"),
        ],
    )
    def test_emitted_slot_rotates_through_every_slot_in_order(self, start, stop, step, stage, expected):
        """Evaluate the slot operand ptoas receives at every induction value.

        A wrong divisor, modulus or start offset changes this sequence even when
        the chain still ends in a remainder. The op-shape check keeps the chain to
        one remainder over at most one subtraction and one division, which is what
        ptoas' slot matcher peels.
        """
        mlir = _codegen(_pipelined_kernel(start, stop, step, stage), passes.MemoryPlanner.PTOAS)
        regions = _lines(mlir, "pto.alloc_multi_tile")
        assert len(regions) == 1 and f"count={stage}" in regions[0], mlir
        get = _lines(mlir, "pto.multi_tile_get")[0]
        slot_ssa = get.split("[")[1].split("]")[0]
        loop = _lines(mlir, "scf.for")[0]
        iv_match = re.match(r"scf\.for (%\S+) =", loop)
        assert iv_match is not None, f"unexpected scf.for syntax: {loop}"
        iv_ssa = iv_match.group(1)
        seq = [_eval_index(mlir, slot_ssa, {iv_ssa: v}) for v in range(start, stop, step)]
        assert seq == expected, f"slot sequence {seq} != {expected}:\n{mlir}"
        ops = _slot_chain_ops(mlir, slot_ssa, {iv_ssa: start})
        # The slot must come from a modulo rather than from a scaled byte offset.
        # A signed floor-mod does not lower to a bare `remsi`: #3030 emits the
        # corrected rem/cmp/add/select sequence so that `-1 % 2` is 1, so the
        # chain's root is the `select`. Require a remainder at the root or
        # immediately under that correction, and no multiplication anywhere --
        # a byte-scaled offset would have to multiply.
        assert ops[0] in ("remsi", "remui", "select"), f"the slot must be a remainder, got {ops}:\n{mlir}"
        if ops[0] == "select":
            # Operand sub-chains interleave (the counter contributes its own
            # subi/floordivsi), so check membership rather than position.
            assert {"cmpi", "addi"} <= set(ops[1:]), f"unexpected correction shape {ops}:\n{mlir}"
        assert "remsi" in ops or "remui" in ops, f"the slot must come from a modulo, got {ops}:\n{mlir}"
        assert "muli" not in ops, f"a slot must not be byte-scaled, got {ops}:\n{mlir}"
        assert sum(op in ("addi", "subi") for op in ops) <= 1, f"too many add/sub steps for ptoas: {ops}"
        assert sum(op.startswith("div") or op == "floordivsi" for op in ops) == (0 if step == 1 else 1), ops


@pytest.mark.parametrize("program", [RotatingSlot, PipelinedLoad], ids=["declared", "pipeline"])
def test_corrected_slot_preserves_native_dynamic_events(tmp_path, program):
    """The native consumer retains dynamic addresses and per-slot sync."""
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    mlir = _codegen(program, passes.MemoryPlanner.PTOAS)
    _assert_logical_slot_index(mlir)
    source = tmp_path / "rotating_slot.pto"
    output = tmp_path / "rotating_slot.cpp"
    source.write_text(mlir)
    _run_ptoas(str(source), str(output), ["--pto-level=level2", "--pto-arch=a2", "--enable-insert-sync"])
    cpp = output.read_text()
    loop = re.search(r"for \(int64_t (\w+) =", cpp)
    assert loop is not None, cpp
    loop_iv = loop.group(1)
    loop_body = cpp[loop.end() :]
    definitions = dict(re.findall(r"^\s*(?:const )?(?:u?int\d+_t|event_t) (\w+) = (.+);$", cpp, re.MULTILINE))

    def dependencies(value: str) -> set[str]:
        found: set[str] = set()
        pending = [value]
        while pending:
            current = pending.pop()
            if current in found:
                continue
            found.add(current)
            pending.extend(re.findall(r"\b\w+\b", definitions.get(current, "")))
        return found

    remainders = {name for name, expr in definitions.items() if re.search(rf"\b{loop_iv}\s*%", expr)}
    assert len(remainders) == 1, f"the native loop must retain its remainder:\n{cpp}"
    remainder = remainders.pop()
    # Both loops here start at a non-negative literal, so the remainder cannot
    # be negative and codegen emits no floor correction — the slot *is* the
    # remainder. When a correction is present it must survive into the native
    # source rather than being folded away, so follow it when it is there.
    # (A loop that can reach a negative index still emits one; this assertion
    # does not cover that path, and `_assert_logical_slot_index` above pins the
    # MLIR-level shape for both.)
    corrected = {
        name
        for name, expr in definitions.items()
        if "<" in expr and "?" in expr and remainder in dependencies(name)
    }
    assert len(corrected) <= 1, f"at most one floor correction is expected:\n{cpp}"
    slot = corrected.pop() if corrected else remainder
    addresses = re.findall(r"TASSIGN\(\w+, (\w+)\);", loop_body)
    assert addresses and all(slot in dependencies(address) for address in addresses), (
        f"the selected tile address must use the corrected slot:\n{cpp}"
    )
    for operation in ("wait_flag", "set_flag"):
        events = re.findall(rf"{operation}\(PIPE_V, PIPE_MTE2, (\w+)\);", loop_body)
        assert any(slot in dependencies(event) for event in events), (
            f"{operation} must retain a per-slot dynamic event id:\n{cpp}"
        )


@pytest.mark.parametrize("dbc", [False, True], ids=["single-C", "dbC-enabled"])
@pytest.mark.parametrize("M,N,K", [(48, 416, 96), (512, 32, 96), (48, 416, 160)])
@pytest.mark.parametrize("backend_type", [BackendType.Ascend910, BackendType.Ascend950])
def test_int8_auto_tile_k_matches_native_operand_boxing(M, N, K, dbc, backend_type, tmp_path):
    """INT8 extracts and peeled K tails must obey the 32-element box.

    Use fully valid N boxes here: padding-only output partitions are a separate
    bug, so native coverage must isolate K legality from valid-output bounds.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[M, K], pl.INT8],
            b: pl.Tensor[[K, N], pl.INT8],
            out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
        ) -> pl.Tensor[[M, N], pl.INT32]:
            at = pl.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
            bt = pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
            acc = pl.tile.matmul(at, bt)
            return pl.store(acc, [0, 0], out)

    backend.reset_for_testing()
    backend.set_backend_type(backend_type)
    with passes.PassContext(
        [], memory_planner=passes.MemoryPlanner.PYPTO, enable_pypto_l0c_double_buffer=dbc
    ):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before)
    func = next(func for func in optimized.functions.values() if func.name == "kernel")
    mlir = codegen.PTOCodegen().generate(ir.Program([func], func.name, optimized.span), emit_tile_addr=True)
    left_k = re.findall(r"!pto.tile_buf<loc=left,[^>]*cols=(\d+)", mlir)
    right_k = re.findall(r"!pto.tile_buf<loc=right,[^>]*rows=(\d+)", mlir)
    assert left_k and right_k, mlir
    assert all(int(k) % 32 == 0 for k in left_k + right_k), mlir
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source = tmp_path / "int8_k_boxing.pto"
    output = tmp_path / "int8_k_boxing.cpp"
    source.write_text(mlir)
    arch = "a2" if backend_type == BackendType.Ascend910 else "a5"
    _run_ptoas(str(source), str(output), [f"--pto-arch={arch}", "--pto-level=level3", "--enable-insert-sync"])
    assert output.is_file()


@pytest.mark.parametrize(
    "planner,dbc",
    [
        (passes.MemoryPlanner.PYPTO, False),
        (passes.MemoryPlanner.PYPTO, True),
        (passes.MemoryPlanner.DSA_RP, False),
        (passes.MemoryPlanner.PTOAS, False),
    ],
    ids=["pypto", "pypto-dbc", "dsa-rp", "ptoas"],
)
@pytest.mark.parametrize("dtype,out_dtype", [(pl.BF16, pl.FP32), (pl.FP16, pl.FP32), (pl.INT8, pl.INT32)])
@pytest.mark.parametrize("M,N,K", [(48, 400, 128), (48, 17, 128), (17, 400, 512), (7, 7, 128)])
def test_auto_tile_padded_output_has_no_empty_native_partitions(
    M, N, K, dtype, out_dtype, planner, dbc, tmp_path
):
    """Whole padding boxes must not become zero-valid output tiles in either grid route."""
    physical_m, physical_n = 64, 416

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[M, K], dtype],
            b: pl.Tensor[[K, N], dtype],
            out: pl.Out[pl.Tensor[[M, N], out_dtype]],
        ) -> pl.Tensor[[M, N], out_dtype]:
            at = pl.load(a, [0, 0], [physical_m, K], valid_shape=[M, K], target_memory=pl.Mem.Mat)
            bt = pl.load(b, [0, 0], [K, physical_n], valid_shape=[K, N], target_memory=pl.Mem.Mat)
            product = pl.tile.matmul(at, bt)
            return pl.store(product, [0, 0], out)

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=planner, enable_pypto_l0c_double_buffer=dbc):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before)
    func = next(func for func in optimized.functions.values() if func.name == "kernel")
    fixed_addresses = planner != passes.MemoryPlanner.PTOAS
    mlir = codegen.PTOCodegen().generate(
        ir.Program([func], func.name, optimized.span), emit_tile_addr=fixed_addresses
    )
    assert "pto.tmatmul" in mlir
    assert not re.search(r"v_(?:row|col)=0(?:[,>])", mlir), mlir
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source = tmp_path / "padded_output.pto"
    output = tmp_path / "padded_output.cpp"
    source.write_text(mlir)
    level = "level3" if fixed_addresses else "level2"
    _run_ptoas(str(source), str(output), ["--pto-arch=a2", f"--pto-level={level}", "--enable-insert-sync"])
    assert output.is_file()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
