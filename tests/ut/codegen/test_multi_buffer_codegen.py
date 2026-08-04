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

Only under the **PTOAS** planner. Under the PyPTO planner ptoas runs at
``--pto-level=level3``, where the explicit-address fan-out is not constant-folded
and slot narrowing degrades to conservative aliasing (hw-native-sys/PTOAS#1106) —
so the baked-address ``pto.alloc_tile`` path stays.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors.
# pyright: reportUndefinedVariable=false

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
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
SIBLING_LOOPS = pl.MemRef(slots=2)


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

    ptoas 0.54 derives the per-slot WAR guard only for the first `multi_tile_get`
    of an iteration: the second load is emitted with no `wait_flag`, so the next
    iteration overwrites that slot while this one still reads it. Measured wrong on
    device, so the region is refused rather than miscompiled — the ping-pong the
    region form accelerates takes one slot per iteration.
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


def _codegen(program: ir.Program, planner: passes.MemoryPlanner) -> str:
    """Compile ``program``'s InCore kernel under ``planner`` and emit its PTO IR."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=planner):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    single = ir.Program([func], "kernel", optimized.span)
    # The planner also decides the address mode: PTOAS plans (no addr, level2),
    # PyPTO bakes (addr, level3) — same pairing compile() applies.
    emit_tile_addr = planner == passes.MemoryPlanner.PYPTO
    return codegen.PTOCodegen().generate(single, emit_tile_addr=emit_tile_addr)


def _lines(mlir: str, needle: str) -> list[str]:
    return [line.strip() for line in mlir.splitlines() if needle in line]


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
        get = _lines(mlir, "pto.multi_tile_get")[0]
        slot_ssa = get.split("[")[1].split("]")[0]

        # The slot SSA is defined by a remainder over the loop induction variable,
        # with no scaling by the slot size in between.
        definition = next((ln for ln in mlir.splitlines() if ln.strip().startswith(f"{slot_ssa} =")), None)
        assert definition is not None, f"slot operand {slot_ssa} has no definition:\n{mlir}"
        assert "remui" in definition or "remsi" in definition, (
            f"the slot operand must be the index expression, got: {definition}"
        )
        slot_bytes = 64 * 64 * 4
        assert str(slot_bytes) not in definition, (
            f"the slot operand must not be scaled to a byte offset: {definition}"
        )

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
        """level3 gets no region: its address fan-out loses slot narrowing.

        See hw-native-sys/PTOAS#1106 — under an explicit base address ptoas emits
        an unfolded `arith.addi` per slot and falls back to conservative aliasing,
        which is worse than the plain baked-address path.
        """
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
            (UnsubscriptedBinding, "without selecting a slot"),
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
