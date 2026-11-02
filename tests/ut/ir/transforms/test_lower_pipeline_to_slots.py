# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Test both LowerPipelineToSlots paths and their conservative whole-loop fallback.

The existing PTOAS path rotates one body through declared slots. The opt-in
PyPTO path builds preload / steady / drain using ordinary loops and the same
slot MemRefs, with prefetch distance equal to stage count minus one.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.arith import Analyzer
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.language.parser.text_parser import parse
from pypto.pypto_core import ir as _ir_core

_PASS_NAME = "LowerPipelineToSlots"


def _run_to_slots(
    program: ir.Program, planner: passes.MemoryPlanner, *, software: bool = False
) -> ir.Program:
    """Run the Default strategy up to and including LowerPipelineToSlots.

    The pass needs tile-level IR (memory spaces inferred, structure normalized),
    so it cannot run standalone on a freshly parsed program. The PassManager is
    built inside the context because its construction reads the planner.
    """
    with passes.PassContext(
        [], memory_planner=planner, enable_buffer_ir=False, enable_software_pipeline=software
    ):
        manager = PassManager(OptimizationStrategy.Default)
        names = manager.pass_names
        stop = names.index(_PASS_NAME)
        for pass_obj in manager.passes[: stop + 1]:
            pipeline = passes.PassPipeline()
            pipeline.add_pass(pass_obj)
            program = pipeline.run(program)
    return program


def _walk_stmts(program: ir.Program):
    """Yield every statement in every function body, depth-first."""

    def walk(stmt):
        if stmt is None:
            return
        yield stmt
        for attr in ("body", "then_body", "else_body", "stmts"):
            sub = getattr(stmt, attr, None)
            if sub is None:
                continue
            for child in sub if isinstance(sub, (list, tuple)) else [sub]:
                yield from walk(child)

    for func in program.functions.values():
        yield from walk(func.body)


def _source_name(name: str) -> str:
    """Strip the suffix ConvertToSSA appends, so tests can name the DSL variable."""
    return name.split("__ssa")[0]


def _slotted_memrefs(program: ir.Program) -> dict[str, ir.MemRef]:
    """Every tile assignment bound to a multi-slot allocation, by source var name."""
    found: dict[str, ir.MemRef] = {}
    for stmt in _walk_stmts(program):
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.var.type, ir.TileType):
            memref = stmt.var.type.memref
            if memref is not None and memref.slot_count_ > 1:
                found[_source_name(stmt.var.name_hint)] = memref
    return found


def _pipeline_loops(program: ir.Program) -> list[ir.ForStmt]:
    """Every ForStmt still carrying the Pipeline kind."""
    return [
        stmt
        for stmt in _walk_stmts(program)
        if isinstance(stmt, ir.ForStmt) and stmt.kind == ir.ForKind.Pipeline
    ]


def _load_count(program: ir.Program) -> int:
    """How many tile.load calls the program contains (replication detector)."""

    class CountLoads(ir.IRVisitor):
        count = 0

        def visit_call(self, op):
            if op.op.name == ir.get_op("tile.load").name:
                self.count += 1
            super().visit_call(op)

    visitor = CountLoads()
    visitor.visit_program(program)
    return visitor.count


_S = ir.Span.unknown()


def _index(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, DataType.INDEX, _S)


def _pipelined(start: int, stop: int, step: int, stage: int) -> ir.Program:
    """One loop-invariant load per iteration, so any bounds parse.

    The load address does not depend on ``i`` on purpose: the slot index is the
    only use of the induction variable, so these cases isolate the rotation.
    """

    @pl.program
    class Program:
        @pl.function
        def main(
            self,
            a: pl.Tensor[[256, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            for i, (acc,) in pl.pipeline(start, stop, step, stage=stage, init_values=(out,)):
                t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                nxt: pl.Tensor[[64, 64], pl.FP32] = pl.store(e, [0, 0], acc)
                y = pl.yield_(nxt)
            return y

    return Program


def _slot_loop_var(program: ir.Program) -> ir.Var:
    loops = [stmt for stmt in _walk_stmts(program) if isinstance(stmt, ir.ForStmt)]
    assert len(loops) == 1, f"expected exactly one loop, got {len(loops)}"
    return loops[0].loop_var


def _expected_slot_index(loop_var: ir.Var, start: int, step: int, factor: int) -> ir.Expr:
    """The exact expression the pass must emit for this loop shape."""
    idx = DataType.INDEX
    if step == 1 and start % factor == 0:
        counter: ir.Expr = loop_var
    elif step == 1:
        counter = ir.Sub(loop_var, _index(start), idx, _S)
    else:
        counter = ir.FloorDiv(ir.Sub(loop_var, _index(start), idx, _S), _index(step), idx, _S)
    return ir.FloorMod(counter, _index(factor), idx, _S)


def _slot_sequence(index: ir.Expr, loop_var: ir.Var, start: int, stop: int, step: int) -> list[int]:
    """The slot each iteration selects, by folding ``index`` at every iv value."""
    seq = []
    for value in range(start, stop, step):
        analyzer = Analyzer()
        analyzer.bind(loop_var, _index(value))
        folded = analyzer.simplify(index)
        assert isinstance(folded, ir.ConstInt), f"slot index did not fold at iv={value}: {folded}"
        seq.append(folded.value)
    return seq


@pl.program
class SingleLoad:
    """The canonical shape: one i-dependent load per pipeline iteration."""

    @pl.function
    def main(
        self,
        a: pl.Tensor[[256, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
    ) -> pl.Tensor[[256, 64], pl.FP32]:
        for i, (acc,) in pl.pipeline(0, 4, 1, stage=2, init_values=(out,)):
            t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 64, 0], [64, 64])
            e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
            nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(e, [i * 64, 0], acc)
            y = pl.yield_(nxt)
        return y


class TestGating:
    """The planner gate decides whether the pass does anything at all."""

    def test_pypto_planner_leaves_the_loop_for_replication(self):
        """Under the default planner no region is emitted, so nothing is bound."""
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PYPTO)
        assert _slotted_memrefs(after) == {}
        assert len(_pipeline_loops(after)) == 1, "the loop must stay Pipeline for LowerPipelineLoops"

    def test_ptoas_planner_slots_the_load_and_demotes_the_loop(self):
        """Under PTOAS the load takes a slot and the loop stops being a Pipeline."""
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PTOAS)
        assert set(_slotted_memrefs(after)) == {"t"}
        assert _pipeline_loops(after) == [], "a slotted loop carries its ping-pong in the slots"

    def test_body_is_not_replicated(self):
        """One body, not F copies — the whole point of the transform."""
        before_loads = _load_count(SingleLoad)
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PTOAS)
        assert _load_count(after) == before_loads


class TestBinding:
    """The slot geometry written onto the tile's TileType."""

    def test_slot_count_matches_the_stage_count(self):
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PTOAS)
        assert _slotted_memrefs(after)["t"].slot_count_ == 2

    def test_slot_index_is_the_induction_variable_modulo_the_stage_count(self):
        """ptoas matches the index's affine form, so it must be literally ``i % 2``."""
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PTOAS)
        index = _slotted_memrefs(after)["t"].slot_index_
        assert index is not None
        assert isinstance(index, ir.FloorMod)
        dividend = index.left
        assert isinstance(dividend, ir.Var), f"the dividend must be the loop var, got {dividend}"
        loop_vars = [stmt.loop_var.name_hint for stmt in _walk_stmts(after) if isinstance(stmt, ir.ForStmt)]
        assert loop_vars == [dividend.name_hint], (
            f"the dividend must be the loop's own induction variable, got {dividend.name_hint} "
            f"against loop vars {loop_vars}"
        )
        divisor = index.right
        assert isinstance(divisor, ir.ConstInt), f"the modulus must be a literal, got {divisor}"
        assert divisor.value == 2

    def test_declaration_is_pinned_so_init_memref_treats_it_as_the_authors(self):
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PTOAS)
        assert _slotted_memrefs(after)["t"].is_pinned_

    def test_compute_tiles_do_not_take_slots(self):
        """Only load buffers need per-stage privacy; slotting everything overflows."""
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PTOAS)
        assert "e" not in _slotted_memrefs(after)

    def test_load_addressed_through_a_loop_carried_iter_arg_takes_a_slot(self):
        """``off`` is a loop-carried IterArg, so this load reads different data every
        iteration without ever naming the induction variable. Treating "does not read
        the loop var" as proof of loop-invariance stranded it: skipped here, and
        skipped by ``LowerPipelineLoops`` too once the sibling candidate demoted the
        loop to Sequential. Every unbound top-level load is a candidate now."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i, (acc, off) in pl.pipeline(0, 4, 1, stage=2, init_values=(out, 0)):
                    carried_load: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [off, 0], [64, 64])
                    good: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 64, 0], [64, 64])
                    s: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.add(carried_load, good)
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(s, [i * 64, 0], acc)
                    y, y_off = pl.yield_(nxt, off + 64)
                return y

        after = _run_to_slots(Before, passes.MemoryPlanner.PTOAS)
        assert set(_slotted_memrefs(after)) == {"carried_load", "good"}

    def test_survives_print_parse_roundtrip(self):
        """A slotted dump must reparse as the same slot."""
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PTOAS)
        reparsed = pl.parse_program(after.as_python())
        ir.assert_structural_equal(reparsed, after)


# (start, stop, step, stage, expected slot per iteration)
_ADMITTED_SHAPES = [
    pytest.param(0, 4, 1, 2, [0, 1, 0, 1], id="unit-aligned-zero-start"),
    pytest.param(3, 10, 1, 3, [0, 1, 2, 0, 1, 2, 0], id="unit-aligned-nonzero-start-3slots-7trips"),
    pytest.param(0, 256, 64, 2, [0, 1, 0, 1], id="stepped-zero-start"),
    pytest.param(64, 320, 64, 2, [0, 1, 0, 1], id="stepped-nonzero-start"),
    pytest.param(64, 384, 64, 3, [0, 1, 2, 0, 1], id="stepped-3slots-5trips"),
    pytest.param(5, 26, 4, 3, [0, 1, 2, 0, 1, 2], id="stepped-unaligned-start-3slots"),
    pytest.param(1, 5, 1, 2, [0, 1, 0, 1], id="unit-unaligned-start"),
    pytest.param(1, 8, 1, 3, [0, 1, 2, 0, 1, 2, 0], id="unit-unaligned-3slots-7trips"),
    pytest.param(1, 3, 1, 4, [0, 1], id="unit-unaligned-more-slots-than-trips"),
]


class TestSlotIndexFormula:
    """The slot expression's exact operands, and the slot every iteration selects.

    Checking only for a ``FloorMod`` over a ``FloorDiv`` would accept a wrong
    divisor or modulus. These cases compare the whole expression structurally and
    then fold it at every induction value, so a wrong operand changes either the
    tree or the sequence.
    """

    @pytest.mark.parametrize("start, stop, step, stage, expected", _ADMITTED_SHAPES)
    def test_slot_index_is_the_rotation_counter_modulo_the_stage_count(
        self, start, stop, step, stage, expected
    ):
        after = _run_to_slots(_pipelined(start, stop, step, stage), passes.MemoryPlanner.PTOAS)
        memref = _slotted_memrefs(after)["t"]
        assert memref.slot_count_ == stage
        slot_index = memref.slot_index_
        assert slot_index is not None, "a slotted MemRef must carry its slot index"
        loop_var = _slot_loop_var(after)
        ir.assert_structural_equal(slot_index, _expected_slot_index(loop_var, start, step, stage))
        assert _slot_sequence(slot_index, loop_var, start, stop, step) == expected

    def test_a_zero_trip_unaligned_loop_does_not_raise(self):
        """``pl.pipeline(5, 5, 1, stage=2)`` never runs; the pass must stay silent.

        It may slot the loop or leave it for replication. Either is correct for a
        loop with no iterations; raising is not.
        """
        _run_to_slots(_pipelined(5, 5, 1, 2), passes.MemoryPlanner.PTOAS)


class TestFallback:
    """One case per gate that the DSL can express. Every one must decline silently,
    never raise. The memory-space and runtime-valid-shape gates have no case: a
    ``tile.load`` result always lands in Vec/Mat/Acc after ``InferTileMemorySpace``,
    and the tile shapes reaching this pass are static."""

    def _assert_declined(self, program: ir.Program):
        after = _run_to_slots(program, passes.MemoryPlanner.PTOAS)
        assert _slotted_memrefs(after) == {}
        assert len(_pipeline_loops(after)) >= 1, "a declined loop stays Pipeline for replication"

    def test_slots_that_overflow_the_memory_space(self):
        """The declared slots are pinned, so ptoas may not reuse any of them. Two
        128 KB slots exceed the Vec budget, and ptoas answers a region it cannot
        place with a hard ``overflow`` error rather than degrading — so the pass
        must decline and let the replication path's capacity gate shrink the depth
        instead."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 256], pl.FP32]],
            ) -> pl.Tensor[[512, 256], pl.FP32]:
                for i, (acc,) in pl.pipeline(0, 4, 1, stage=2, init_values=(out,)):
                    t: pl.Tile[[128, 256], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 128, 0], [128, 256])
                    e: pl.Tile[[128, 256], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[512, 256], pl.FP32] = pl.store(e, [i * 128, 0], acc)
                    y = pl.yield_(nxt)
                return y

        self._assert_declined(Before)

    def test_load_reaching_a_phi_through_a_bare_alias(self):
        """``InitMemRef`` shares one MemRef across a bare ``v = t`` tile copy, so
        yielding the *alias* carries the candidate's slot into the phi just as
        yielding ``t`` would. Recording only the names that literally appear in the
        ``YieldStmt`` misses it, and the region reaches codegen anyway."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                flag: pl.Scalar[pl.INT32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                seed: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                for i, (carried,) in pl.pipeline(0, 4, 1, stage=2, init_values=(seed,)):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 64, 0], [64, 64])
                    if flag > 0:
                        v: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = t
                        yv = pl.yield_(v)
                    else:
                        v2: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(carried)
                        yv = pl.yield_(v2)
                    y = pl.yield_(yv)
                nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(y, [0, 0], out)
                return nxt

        self._assert_declined(Before)

    def test_author_region_counts_against_the_capacity_budget(self):
        """A declared allocation is pinned too, so ptoas cannot reuse it either.
        Budgeting only the regions this pass synthesizes lets a 128 KB region in
        alongside the author's 128 KB one, and the pair overflows the space."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 256], pl.FP32],
                b: pl.Tensor[[512, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 256], pl.FP32]],
            ) -> pl.Tensor[[512, 256], pl.FP32]:
                for i, (acc,) in pl.pipeline(0, 4, 1, stage=2, init_values=(out,)):
                    mine: pl.Tile[[64, 256], pl.FP32, pl.MemRef("mine", slots=2)[i % 2], pl.Mem.Vec] = (
                        pl.load(a, [i * 64, 0], [64, 256])
                    )
                    t: pl.Tile[[64, 256], pl.FP32, pl.Mem.Vec] = pl.load(b, [i * 64, 0], [64, 256])
                    s: pl.Tile[[64, 256], pl.FP32, pl.Mem.Vec] = pl.add(mine, t)
                    nxt: pl.Tensor[[512, 256], pl.FP32] = pl.store(s, [i * 64, 0], acc)
                    y = pl.yield_(nxt)
                return y

        after = _run_to_slots(Before, passes.MemoryPlanner.PTOAS)
        assert set(_slotted_memrefs(after)) == {"mine"}, "only the author's region may survive"
        assert len(_pipeline_loops(after)) >= 1, "the loop must stay Pipeline for replication"

    def test_load_carried_into_a_nested_loops_iter_arg(self):
        """A nested loop's ``init_values`` reach the phi through ``IterArg.initValue_``
        rather than a ``YieldStmt``, but bind the same way. Slotting the initializer
        used to leave the inner init pointing at the pre-substitution Var, which fails
        SSA verification; the loop must be declined instead."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(0, 4, 1, stage=2, init_values=(out,)):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 64, 0], [64, 64])
                    for _k, (carried,) in pl.range(2, init_values=(t,)):
                        stepped: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(carried)
                        yk = pl.yield_(stepped)
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(yk, [i * 64, 0], acc)
                    y = pl.yield_(nxt)
                return y

        self._assert_declined(Before)

    def test_stepped_loop_is_admitted_with_a_counter_based_rotation(self):
        """A non-unit step counts iterations, so the slot is the counter mod F.

        The induction variable is not the rotation counter here: with ``step ==
        64`` over ``[0, 256)`` the loop runs four times but ``i`` only ever takes
        ``0/64/128/192``. Indexing slots by ``i % 2`` directly would select slot 0
        on every iteration and lose the rotation, so the slot is ``(i - start) / step % F`` — a remainder over
        a single SSA value, which is the property ptoas' slot accounting keys on
        (it returns whatever ``pto.multi_tile_get`` carries and requires only that
        two slots share the *identity* of that value, not its shape).
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(0, 256, 64, stage=2, init_values=(out,)):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i, 0], [64, 64])
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(e, [i, 0], acc)
                    y = pl.yield_(nxt)
                return y

        after = _run_to_slots(Before, passes.MemoryPlanner.PTOAS)
        assert set(_slotted_memrefs(after)) == {"t"}
        assert _pipeline_loops(after) == [], "a slotted loop carries its ping-pong in the slots"
        assert _load_count(after) == _load_count(Before), "one body, not F copies"

        # The slot must be the *counter*, not the raw induction variable: with
        # step 64 over [0, 256) `i` only takes 0/64/128/192, so `i % 2` would select
        # slot 0 every time and lose the rotation. It is `(i - start) / step % F` — still
        # a remainder over one SSA value, which is the property ptoas keys on.
        memref = _slotted_memrefs(after)["t"]
        slot = memref.slot_index_
        assert slot is not None, "a slotted MemRef must carry its slot index"
        assert not isinstance(slot, ir.Var), f"the slot must be the counter, not the raw iv: {slot}"
        assert isinstance(slot, ir.FloorMod), f"the slot must be a remainder, got {type(slot).__name__}"
        counter = slot.left
        assert isinstance(counter, ir.FloorDiv), (
            f"the rotation counter must be a division, got {type(counter).__name__}"
        )

    def test_a_negative_start_is_declined(self):
        """A negative start cannot index a slot, in either form.

        ``iv % factor`` is a *signed* remainder, so `start = -2, step = 1` yields
        `(-1) % 2 == -1` — a slot before the region. That shape satisfies the old
        `start % factor == 0` test, so it reached codegen; this gate closes that
        pre-existing hole as well as guarding the counter form, where
        `iv - start` is not representable for a sufficiently negative start.

        The load address is loop-invariant on purpose: a negative start would
        otherwise produce negative slice offsets and the shape would not parse.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(-2, 6, 1, stage=2, init_values=(out,)):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[64, 64], pl.FP32] = pl.store(e, [0, 0], acc)
                    y = pl.yield_(nxt)
                return y

        self._assert_declined(Before)

    def test_a_start_whose_counter_would_overflow_is_declined(self):
        """The counter subtracts `start`, so a start near the type's floor is out.

        `pl.pipeline(INT64_MIN, 1, 1 << 61, stage=3)` runs five iterations and
        reaches `iv = 0`, where `0 - INT64_MIN` is not representable. The wrapped
        quotient would then make the remainder select slot `-1` of a three-slot
        region — an access before the allocation. Replication handles these bounds
        correctly, so declining is the right answer.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(-(2**63), 1, 1 << 61, stage=3, init_values=(out,)):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[64, 64], pl.FP32] = pl.store(e, [0, 0], acc)
                    y = pl.yield_(nxt)
                return y

        self._assert_declined(Before)

    def test_a_stop_whose_final_update_would_overflow_is_declined(self):
        """A stepped loop's last `iv + step` update must stay representable.

        `pl.pipeline(INT64_MAX - 4, INT64_MAX, 3, stage=3)` runs two iterations, at
        `INT64_MAX - 4` and `INT64_MAX - 1`, but the retained `scf.for` then computes
        `INT64_MAX - 1 + 3`, which overflows the signed index and can keep the loop
        running. Replication emits the tail copies with no loop, so declining is right.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(2**63 - 5, 2**63 - 1, 3, stage=3, init_values=(out,)):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[64, 64], pl.FP32] = pl.store(e, [0, 0], acc)
                    y = pl.yield_(nxt)
                return y

        self._assert_declined(Before)

    def test_a_stop_with_room_for_the_final_update_is_admitted(self):
        """The bound is tight: `stop == INT64_MAX - (step - 1)` still fits.

        `pl.pipeline(INT64_MAX - 7, INT64_MAX - 2, 3, stage=2)` iterates at
        `INT64_MAX - 7` and `INT64_MAX - 4`; the update after the last one is
        `INT64_MAX - 1`, which is representable, so the loop is slotted.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(2**63 - 8, 2**63 - 3, 3, stage=2, init_values=(out,)):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[64, 64], pl.FP32] = pl.store(e, [0, 0], acc)
                    y = pl.yield_(nxt)
                return y

        after = _run_to_slots(Before, passes.MemoryPlanner.PTOAS)
        assert set(_slotted_memrefs(after)) == {"t"}

    def test_a_stepped_loop_with_a_runtime_stop_is_declined(self):
        """A runtime `stop` cannot be shown to leave room for `iv + step`.

        Unlike `step == 1`, a stepped loop can overshoot `stop`, so it is only
        admitted with a constant `stop` the pass can bound.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(0, n, 2, stage=2, init_values=(out,)):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[64, 64], pl.FP32] = pl.store(e, [0, 0], acc)
                    y = pl.yield_(nxt)
                return y

        self._assert_declined(Before)

    def test_load_carried_out_as_a_phi(self):
        """A yielded tile makes the return var share its MemRef — codegen calls that
        'one of its slots is carried out of an if or a loop as a phi'."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                seed: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                for i, (carried,) in pl.pipeline(0, 4, 1, stage=2, init_values=(seed,)):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 64, 0], [64, 64])
                    y = pl.yield_(t)
                nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(y, [0, 0], out)
                return nxt

        self._assert_declined(Before)

    def test_load_consumed_by_a_view_op(self):
        """A reshape's result IS its source's buffer, so it would land on the same
        allocation with a different tile_buf type — a codegen blocker."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(0, 4, 1, stage=2, init_values=(out,)):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 64, 0], [64, 64])
                    r: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t, [32, 128])
                    e: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.exp(r)
                    f: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(e, [64, 64])
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(f, [i * 64, 0], acc)
                    y = pl.yield_(nxt)
                return y

        self._assert_declined(Before)

    def test_stage_count_above_the_ptoas_maximum(self):
        """ptoas describes 2..16 slots; 32 has no region form."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[2048, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[2048, 64], pl.FP32]],
            ) -> pl.Tensor[[2048, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(0, 32, 1, stage=32, init_values=(out,)):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 64, 0], [64, 64])
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[2048, 64], pl.FP32] = pl.store(e, [i * 64, 0], acc)
                    y = pl.yield_(nxt)
                return y

        self._assert_declined(Before)

    def test_loop_nested_under_a_declined_pipeline_loop(self):
        """The outer step-64 loop is replicated, and its F clones would each select one
        slot of the same allocation inside one body — a shape codegen rejects."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for _o, (outer,) in pl.pipeline(0, 256, 64, stage=2, init_values=(out,)):
                    for i, (inner,) in pl.pipeline(0, 4, 1, stage=2, init_values=(outer,)):
                        t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 64, 0], [64, 64])
                        e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                        nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(e, [i * 64, 0], inner)
                        y_in = pl.yield_(nxt)
                    y = pl.yield_(y_in)
                return y

        self._assert_declined(Before)

    def test_one_blocked_load_declines_the_whole_loop(self):
        """Dropping only the blocked load would still demote the loop to Sequential,
        so that load would reach neither these slots nor LowerPipelineLoops' copies."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(0, 4, 1, stage=2, init_values=(out,)):
                    ok: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 64, 0], [64, 64])
                    # Consumed by a view op, so it can never become a slot.
                    blocked: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(b, [i * 64, 0], [64, 64])
                    r: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(blocked, [32, 128])
                    v: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.exp(r)
                    f: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(v, [64, 64])
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(ok)
                    s: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.add(e, f)
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(s, [i * 64, 0], acc)
                    y = pl.yield_(nxt)
                return y

        self._assert_declined(Before)

    def test_author_declared_allocation_is_left_alone(self):
        """A binding the author wrote stays the author's."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(0, 4, 1, stage=2, init_values=(out,)):
                    t: pl.Tile[[64, 64], pl.FP32, pl.MemRef("mine", slots=2)[i % 2], pl.Mem.Vec] = pl.load(
                        a, [i * 64, 0], [64, 64]
                    )
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(e, [i * 64, 0], acc)
                    y = pl.yield_(nxt)
                return y

        after = _run_to_slots(Before, passes.MemoryPlanner.PTOAS)
        bound = _slotted_memrefs(after)
        assert set(bound) == {"t"}
        assert bound["t"].base_.name_hint == "mine", "the pass must not re-base the author's allocation"


class TestChaining:
    """The two passes are complementary, not alternatives."""

    def test_declined_loop_is_still_replicated_by_lower_pipeline_loops(self):
        """A loop this pass declines must still get its F body copies.

        Declined here on the pinned-slot budget rather than on the loop shape: the
        slots are pinned, so two 128 KB slots exceed the Vec budget and ptoas would
        answer the region with a hard ``overflow`` error instead of degrading. The
        loop shape itself is no longer a reason to decline — a non-unit step takes
        the counter-based rotation.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 256], pl.FP32]],
            ) -> pl.Tensor[[512, 256], pl.FP32]:
                for i, (acc,) in pl.pipeline(0, 4, 1, stage=2, init_values=(out,)):
                    t: pl.Tile[[128, 256], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 128, 0], [128, 256])
                    e: pl.Tile[[128, 256], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[512, 256], pl.FP32] = pl.store(e, [i * 128, 0], acc)
                    y = pl.yield_(nxt)
                return y

        before_loads = _load_count(Before)
        after_slots = _run_to_slots(Before, passes.MemoryPlanner.PTOAS)
        assert _slotted_memrefs(after_slots) == {}, "the budget gate must decline this"
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
            replicated = passes.lower_pipeline_loops()(after_slots)
        assert _load_count(replicated) == 2 * before_loads


def _plan_memory(program, *, allocate=False):
    with passes.PassContext(
        [], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=False, enable_software_pipeline=True
    ):
        after = passes.init_mem_ref()(program)
        after = passes.materialize_semantic_aliases()(after)
        after = passes.memory_reuse()(after)
        return passes.allocate_memory_addr()(after) if allocate else after


@pytest.mark.usefixtures("ascend_backend")
class TestSoftwarePipeline:
    """The opt-in PyPTO path splits prefetch from consumption without new IR kinds."""

    @staticmethod
    def _program(stage: int, trips: int, start: int = 0, step: int = 1) -> ir.Program:
        stop = start + trips * step

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[1024, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
            ) -> pl.Tensor[[1024, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(start, stop, step, stage=stage, init_values=(out,)):
                    t: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i, 0], [1, 64])
                    e: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.add(t, 1.0)
                    nxt: pl.Tensor[[1024, 64], pl.FP32] = pl.store(e, [i, 0], acc)
                    y = pl.yield_(nxt)
                return y

        return Before

    @pytest.mark.parametrize("stage", [2, 3, 4])
    @pytest.mark.parametrize("trips", [4, 5, 64, 65])
    def test_preload_steady_and_drain(self, stage, trips):
        after = _run_to_slots(self._program(stage, trips), passes.MemoryPlanner.PYPTO, software=True)
        assert not _pipeline_loops(after)
        loops = [stmt for stmt in _walk_stmts(after) if isinstance(stmt, ir.ForStmt)]
        assert len(loops) == 1
        steady = loops[0]
        assert steady.kind == ir.ForKind.Sequential
        assert isinstance(steady.start, ir.ConstInt)
        assert isinstance(steady.stop, ir.ConstInt)
        assert isinstance(steady.step, ir.ConstInt)
        assert steady.start.value == 0
        assert steady.stop.value == trips - (stage - 1)
        assert steady.step.value == 1
        assert _load_count(after) == stage, "P prologue loads plus one steady load"
        calls = [
            stmt.value
            for stmt in _walk_stmts(after)
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
        ]
        create_name = _ir_core.get_op("tile.create").name
        assert sum(call.op.name == create_name for call in calls) == stage
        memrefs = list(_slotted_memrefs(after).values())
        assert memrefs and all(memref.slot_count_ == stage for memref in memrefs)
        ir.assert_structural_equal(pl.parse_program(after.as_python()), after)

    @pytest.mark.parametrize("stage", [2, 3, 4])
    def test_exact_prefetch_count_has_no_empty_steady_loop(self, stage):
        after = _run_to_slots(self._program(stage, stage - 1), passes.MemoryPlanner.PYPTO, software=True)
        assert not _pipeline_loops(after)
        assert not [stmt for stmt in _walk_stmts(after) if isinstance(stmt, ir.ForStmt)]
        assert _load_count(after) == stage - 1
        assert _slotted_memrefs(after)

    @pytest.mark.parametrize("stage,trips", [(2, 0), (3, 0), (3, 1), (4, 1), (4, 2)])
    def test_short_loop_keeps_existing_lowering(self, stage, trips):
        before = self._program(stage, trips)
        legacy = _run_to_slots(before, passes.MemoryPlanner.PYPTO)
        enabled = _run_to_slots(before, passes.MemoryPlanner.PYPTO, software=True)
        ir.assert_structural_equal(enabled, legacy)
        assert not _slotted_memrefs(enabled)

    def test_normalizes_source_start_and_step(self):
        after = _run_to_slots(self._program(3, 5, start=7, step=3), passes.MemoryPlanner.PYPTO, software=True)
        loops = [stmt for stmt in _walk_stmts(after) if isinstance(stmt, ir.ForStmt)]
        assert len(loops) == 1
        steady = loops[0]
        assert isinstance(steady.start, ir.ConstInt)
        assert isinstance(steady.step, ir.ConstInt)
        assert steady.start.value == 0 and steady.step.value == 1
        refs = [
            stmt.var.type.memref
            for stmt in _walk_stmts(after)
            if isinstance(stmt, ir.AssignStmt)
            and isinstance(stmt.var.type, ir.TileType)
            and stmt.var.type.memref is not None
        ]
        dynamic = [ref.slot_index_ for ref in refs if isinstance(ref.slot_index_, ir.FloorMod)]
        assert dynamic
        assert all(isinstance(index.left, (ir.Var, ir.Add)) for index in dynamic)
        assert "7" in after.as_python()
        ir.assert_structural_equal(pl.parse_program(after.as_python()), after)

    def test_result_reuses_last_used_input_slot(self):
        after = _run_to_slots(self._program(3, 8), passes.MemoryPlanner.PYPTO, software=True)
        after = _plan_memory(after)
        refs = _slotted_memrefs(after)
        assert set(refs) == {"t", "e"}
        assert refs["t"].base_.same_as(refs["e"].base_), (
            "terminal in-place reuse must not add an output region"
        )

    @pytest.mark.parametrize(
        "ascend_backend,split,stage",
        [
            (BackendType.Ascend910, True, 2),
            (BackendType.Ascend910, True, 3),
            (BackendType.Ascend910, False, 3),
            (BackendType.Ascend950, True, 3),
        ],
        indirect=["ascend_backend"],
    )
    def test_input_slot_forwarding_respects_load_tpop_hazard(self, ascend_backend, split, stage):
        """Slot forwarding must preserve the split-AIV load/TPOP target constraint."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
            def main(
                self,
                a: pl.Tensor[[64, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                pl.func_attr({"split": pl.SplitMode.UP_DOWN})
                pipe_chunk: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=1)
                for i, (acc,) in pl.pipeline(8, stage=stage, init_values=(out,)):
                    loaded: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        a, [i * 8, 0], [8, 128], target_memory=pl.Mem.Vec
                    )
                    # This view still reads pipe-owned storage, not a pool allocation.
                    pipe_view: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(pipe_chunk, [8, 128])
                    result: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(loaded, pipe_view)
                    stored = pl.tile.store(result, [i * 8, 0], acc)
                    y = pl.yield_(stored)
                return y

        if not split:
            Before = parse(Before.as_python().replace("SplitMode.UP_DOWN", "SplitMode.NONE"))
        # Start from typed SSA AIV IR: cross-core pipe setup is orthogonal to
        # the scheduling and memory-reuse passes under test.
        with passes.PassContext(
            [],
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_buffer_ir=False,
            enable_software_pipeline=True,
        ):
            lowered = passes.lower_pipeline_to_slots()(Before)
        assert not _pipeline_loops(lowered), "Exercise actual scheduled slots, not fallback replication"
        assert _load_count(lowered) == stage
        after = _plan_memory(lowered)
        outputs = [
            stmt
            for stmt in _walk_stmts(after)
            if isinstance(stmt, ir.AssignStmt)
            and isinstance(stmt.value, ir.Call)
            and stmt.value.op.name == ir.get_op("tile.add").name
        ]
        assert outputs
        hazardous = ascend_backend == BackendType.Ascend910 and split
        for stmt in outputs:
            assert isinstance(stmt.value, ir.Call)
            source_type = stmt.value.args[0].type
            destination_type = stmt.var.type
            assert isinstance(source_type, ir.TileType) and isinstance(destination_type, ir.TileType)
            source = source_type.memref
            destination = destination_type.memref
            assert source is not None and destination is not None
            assert source.slot_count_ == stage
            assert source.base_.same_as(destination.base_) is not hazardous, (
                "Only Ascend910 split-AIV must separate the output from its load slot"
            )

    def test_multiple_inputs_share_no_regions(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[8, 64], pl.FP32],
                b: pl.Tensor[[8, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(8, stage=3, init_values=(out,)):
                    x: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i, 0], [1, 64])
                    y: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.load(b, [i, 0], [1, 64])
                    z: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.add(x, y)
                    nxt = pl.store(z, [i, 0], acc)
                    result = pl.yield_(nxt)
                return result

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        assert not _pipeline_loops(after)
        refs = _slotted_memrefs(after)
        assert not refs["x"].base_.same_as(refs["y"].base_)
        assert _load_count(after) == 6

    def test_compute_chain_reuses_last_used_input_slots(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[8, 64], pl.FP32],
                b: pl.Tensor[[8, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(8, stage=3, init_values=(out,)):
                    x: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i, 0], [1, 64])
                    z: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.load(b, [i, 0], [1, 64])
                    first: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.add(x, z)
                    final: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.add(first, 1.0)
                    nxt = pl.store(final, [i, 0], acc)
                    result = pl.yield_(nxt)
                return result

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        assert not _pipeline_loops(after)
        after = _plan_memory(after)
        refs = _slotted_memrefs(after)
        assert set(refs) == {"x", "z", "first", "final"}
        assert refs["first"].base_.same_as(refs["x"].base_)
        assert refs["final"].slot_count_ == 3
        assert refs["final"].base_.same_as(refs["x"].base_)
        assert not refs["final"].base_.same_as(refs["z"].base_)
        assert _load_count(after) == 6, "output slots must not introduce any preloads"
        ir.assert_structural_equal(pl.parse_program(after.as_python()), after)

    @pytest.mark.parametrize("trips", [2, 3, 64, 65])
    def test_score_reuses_cast_chain_and_last_use_scale_with_private_scratch(self, trips):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                acc: pl.Tensor[[trips * 128, 64], pl.INT32],
                scale: pl.Tensor[[trips * 128, 1], pl.FP32],
                coef: pl.Tensor[[1, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[trips * 128, 1], pl.FP32]],
            ) -> pl.Tensor[[trips * 128, 1], pl.FP32]:
                resident = pl.load(coef, [0, 0], [1, 64], target_memory=pl.Mem.Vec)
                scratch = pl.create_tile([128, 64], dtype=pl.FP32)
                for i in pl.pipeline(trips, stage=3):
                    a = pl.load(acc, [i * 128, 0], [128, 64], target_memory=pl.Mem.Vec)
                    s = pl.load(scale, [i * 128, 0], [128, 1], target_memory=pl.Mem.Vec)
                    fp = pl.cast(a, target_type=pl.FP32, mode="none")
                    positive = pl.maximum(fp, 0.0)
                    weighted = pl.col_expand_mul(positive, resident)
                    reduced = pl.row_sum(weighted, scratch)
                    value = pl.mul(reduced, s)
                    out = pl.store(value, [i * 128, 0], out)
                return out

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        assert not _pipeline_loops(after)
        after = _plan_memory(after)
        refs = _slotted_memrefs(after)
        for name in ("fp", "positive", "weighted"):
            assert refs[name].base_.same_as(refs["a"].base_)
        assert "resident" not in refs and "scratch" not in refs and "reduced" not in refs
        roots = {ref.base_.name_hint for ref in refs.values()}
        assert len(roots) == 2, "The score uses only accumulator and scale rings"
        assert _load_count(after) == (3 if trips == 2 else 4) + (2 if trips == 2 else 3)
        ir.assert_structural_equal(pl.parse_program(after.as_python()), after)

    def test_workspace_read_as_data_keeps_original_schedule(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self, src: pl.Tensor[[8 * 32, 64], pl.FP32], out: pl.Out[pl.Tensor[[8 * 32, 1], pl.FP32]]
            ) -> pl.Tensor[[8 * 32, 1], pl.FP32]:
                scratch = pl.create_tile([32, 64], dtype=pl.FP32)
                for i in pl.pipeline(8, stage=3):
                    x = pl.load(src, [i * 32, 0], [32, 64])
                    reduced = pl.row_sum(x, scratch)
                    reread = pl.row_sum(scratch, x)
                    out = pl.store(pl.add(reduced, reread), [i * 32, 0], out)
                return out

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        assert _pipeline_loops(after)
        assert not _slotted_memrefs(after)

    def test_repeated_operand_in_one_compute_is_one_consumer(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self, a: pl.Tensor[[8, 64], pl.FP32], out: pl.Out[pl.Tensor[[8, 64], pl.FP32]]
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(8, stage=3, init_values=(out,)):
                    x: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i, 0], [1, 64])
                    y: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.add(x, x)
                    nxt = pl.store(y, [i, 0], acc)
                    result = pl.yield_(nxt)
                return result

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        assert not _pipeline_loops(after)
        after = _plan_memory(after)
        refs = _slotted_memrefs(after)
        assert refs["x"].base_.same_as(refs["y"].base_)

    def test_shared_prefetched_input_falls_back(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self, a: pl.Tensor[[8, 64], pl.FP32], out: pl.Out[pl.Tensor[[8, 64], pl.FP32]]
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(8, stage=3, init_values=(out,)):
                    x: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i, 0], [1, 64])
                    y: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.add(x, 1.0)
                    z: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.add(x, y)
                    nxt = pl.store(z, [i, 0], acc)
                    result = pl.yield_(nxt)
                return result

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        assert len(_pipeline_loops(after)) == 1
        assert not _slotted_memrefs(after), "multiple consumers of a rotating input can exhaust PTOAS events"

    def test_before_after_exact_prefetch(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def main(
                self, a: pl.Tensor[[1, 64], pl.FP32], out: pl.Out[pl.Tensor[[1, 64], pl.FP32]]
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(1, stage=2, init_values=(out,)):
                    t: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(a, [i, 0], [1, 64])
                    e: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(t, 1.0)
                    nxt = pl.tile.store(e, [i, 0], acc)
                    y = pl.yield_(nxt)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def main(
                self, a: pl.Tensor[[1, 64], pl.FP32], out: pl.Out[pl.Tensor[[1, 64], pl.FP32]]
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                _preload: pl.Tile[[1, 64], pl.FP32, pl.MemRef("pipe_t_0", slots=2)[0], pl.Mem.Vec] = (
                    pl.tile.load(a, [0, 0], [1, 64], attrs={"software_pipeline_slots": True})
                )
                current: pl.Tile[[1, 64], pl.FP32, pl.MemRef("pipe_t_0", slots=2)[0], pl.Mem.Vec] = (
                    pl.tile.create(
                        [1, 64], pl.FP32, target_memory=pl.Mem.Vec, attrs={"software_pipeline_slots": True}
                    )
                )
                value: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(current, 1.0)
                nxt = pl.tile.store(value, [0, 0], out)
                y = nxt
                return y

        with passes.PassContext(
            [],
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_buffer_ir=False,
            enable_software_pipeline=True,
        ):
            after = passes.lower_pipeline_to_slots()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_forbidden_alias_operand_is_checked_by_value(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self, a: pl.Tensor[[8, 64], pl.FP32], out: pl.Out[pl.Tensor[[8, 64], pl.FP32]]
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(8, stage=3, init_values=(out,)):
                    x: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i, 0], [1, 64])
                    y: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.addsc(x, 1.0, x)
                    nxt = pl.store(y, [i, 0], acc)
                    result = pl.yield_(nxt)
                return result

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        assert not _pipeline_loops(after)
        after = _plan_memory(after)
        refs = _slotted_memrefs(after)
        assert set(refs) == {"x"}  # The forbidden result keeps ordinary storage.

    def test_scalar_recurrence_falls_back(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self, a: pl.Tensor[[8, 64], pl.FP32], out: pl.Out[pl.Tensor[[8, 64], pl.FP32]]
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                for i, (acc, off) in pl.pipeline(8, stage=3, init_values=(out, 0)):
                    x: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [off, 0], [1, 64])
                    nxt = pl.store(x, [i, 0], acc)
                    result, next_off = pl.yield_(nxt, off + 1)
                return result

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        assert len(_pipeline_loops(after)) == 1
        assert not _slotted_memrefs(after)

    def test_same_input_output_tensor_falls_back(self):
        @pl.program
        class Before:
            @pl.function
            def main(self, a: pl.Tensor[[8, 64], pl.FP32]) -> pl.Tensor[[8, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(8, stage=3, init_values=(a,)):
                    x: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i, 0], [1, 64])
                    y: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.add(x, 1.0)
                    nxt = pl.store(y, [i, 0], acc)
                    result = pl.yield_(nxt)
                return result

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        assert len(_pipeline_loops(after)) == 1
        assert not _slotted_memrefs(after)

    @pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910B, BackendType.Ascend950], indirect=True)
    def test_dynamic_bound_has_guarded_preload_and_one_rotating_body(self, ascend_backend):
        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[8, 64], pl.FP32],
                n: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(n, stage=3, init_values=(out,)):
                    x: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i, 0], [1, 64])
                    nxt = pl.store(x, [i, 0], acc)
                    result = pl.yield_(nxt)
                return result

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        if ascend_backend == BackendType.Ascend950:
            assert len(_pipeline_loops(after)) == 1
            assert not _slotted_memrefs(after)
            return
        assert not _pipeline_loops(after)
        assert _slotted_memrefs(after)
        loops = [stmt for stmt in _walk_stmts(after) if isinstance(stmt, ir.ForStmt)]
        assert len(loops) == 2, "One scheduled body and one overflow fallback"
        assert all(loop.kind == ir.ForKind.Sequential for loop in loops)
        guards = [stmt for stmt in _walk_stmts(after) if isinstance(stmt, ir.IfStmt)]
        assert len(guards) == 5, "Nonempty/overflow guards, two preloads and a future load"
        legacy = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=False)
        assert len(_pipeline_loops(legacy)) == 1
        assert not _slotted_memrefs(legacy)

    def test_declined_parent_keeps_nested_pipeline_replicable(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self, a: pl.Tensor[[8, 64], pl.FP32], out: pl.Out[pl.Tensor[[8, 64], pl.FP32]]
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                for outer, (acc,) in pl.pipeline(2, stage=2, init_values=(out,)):
                    for inner, (inner_acc,) in pl.pipeline(4, stage=3, init_values=(acc,)):
                        x: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [outer * 4 + inner, 0], [1, 64])
                        y: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.add(x, 1.0)
                        nxt = pl.store(y, [outer * 4 + inner, 0], inner_acc)
                        inner_result = pl.yield_(nxt)
                    result = pl.yield_(inner_result)
                return result

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        assert len(_pipeline_loops(after)) == 2
        assert not _slotted_memrefs(after)
        with passes.PassContext(
            [],
            memory_planner=passes.MemoryPlanner.PYPTO,
            enable_buffer_ir=False,
            enable_software_pipeline=True,
        ):
            replicated = passes.lower_pipeline_loops()(after)
        assert not _slotted_memrefs(replicated)

    def test_unsupported_stage_falls_back(self):
        before = self._program(5, 8)
        legacy = _run_to_slots(before, passes.MemoryPlanner.PYPTO)
        enabled = _run_to_slots(before, passes.MemoryPlanner.PYPTO, software=True)
        ir.assert_structural_equal(enabled, legacy)
        assert not _slotted_memrefs(enabled)

    def test_stored_intermediate_with_compute_consumer_falls_back(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[8, 64], pl.FP32],
                out1: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
                out2: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[8, 64], pl.FP32], pl.Tensor[[8, 64], pl.FP32]]:
                for i, (acc1, acc2) in pl.pipeline(8, stage=3, init_values=(out1, out2)):
                    x: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i, 0], [1, 64])
                    first: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.add(x, 1.0)
                    nxt1 = pl.store(first, [i, 0], acc1)
                    final: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.add(first, 2.0)
                    nxt2 = pl.store(final, [i, 0], acc2)
                    result1, result2 = pl.yield_(nxt1, nxt2)
                return result1, result2

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        assert len(_pipeline_loops(after)) == 1
        assert not _slotted_memrefs(after), "a stored intermediate must not become a rotating compute input"

    def test_multiple_stores_share_one_output_region(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[8, 64], pl.FP32],
                out1: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
                out2: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[8, 64], pl.FP32], pl.Tensor[[8, 64], pl.FP32]]:
                for i, (acc1, acc2) in pl.pipeline(8, stage=3, init_values=(out1, out2)):
                    x: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i, 0], [1, 64])
                    y: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.add(x, 1.0)
                    nxt1 = pl.store(y, [i, 0], acc1)
                    nxt2 = pl.store(y, [i, 0], acc2)
                    result1, result2 = pl.yield_(nxt1, nxt2)
                return result1, result2

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        assert not _pipeline_loops(after)
        after = _plan_memory(after)
        refs = _slotted_memrefs(after)
        assert set(refs) == {"x", "y"}
        assert refs["x"].base_.same_as(refs["y"].base_)
        store_name = _ir_core.get_op("tile.store").name
        stores = [
            stmt.value
            for stmt in _walk_stmts(after)
            if isinstance(stmt, ir.AssignStmt)
            and isinstance(stmt.value, ir.Call)
            and stmt.value.op.name == store_name
        ]
        assert len(stores) == 6, "one steady and two drain iterations each contain two stores"
        for store in stores:
            source_type = store.args[0].type
            assert isinstance(source_type, ir.TileType)
            assert source_type.memref is not None
            assert source_type.memref.base_.same_as(refs["y"].base_)
        ir.assert_structural_equal(pl.parse_program(after.as_python()), after)

    def test_non_inplace_capacity_is_checked_after_memory_planning(self):
        """A 144 KiB input ring plus an ordinary 48 KiB result exceeds UB."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self, a: pl.Tensor[[768, 64], pl.FP32], out: pl.Out[pl.Tensor[[768, 64], pl.FP32]]
            ) -> pl.Tensor[[768, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(4, stage=3, init_values=(out,)):
                    x: pl.Tile[[192, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 192, 0], [192, 64])
                    y: pl.Tile[[192, 64], pl.FP32, pl.Mem.Vec] = pl.tile.addsc(x, 1.0, x)
                    nxt = pl.store(y, [i * 192, 0], acc)
                    result = pl.yield_(nxt)
                return result

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        with pytest.raises(ValueError, match="Vec buffer usage.*exceeds"):
            _plan_memory(after, allocate=True)

    def test_inplace_output_does_not_duplicate_region_budget(self):
        """The same 48 KiB geometry fits when load and output share one 144 KiB ring."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self, a: pl.Tensor[[768, 64], pl.FP32], out: pl.Out[pl.Tensor[[768, 64], pl.FP32]]
            ) -> pl.Tensor[[768, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(4, stage=3, init_values=(out,)):
                    x: pl.Tile[[192, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 192, 0], [192, 64])
                    y: pl.Tile[[192, 64], pl.FP32, pl.Mem.Vec] = pl.add(x, 1.0)
                    nxt = pl.store(y, [i * 192, 0], acc)
                    result = pl.yield_(nxt)
                return result

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        assert not _pipeline_loops(after)
        after = _plan_memory(after)
        refs = _slotted_memrefs(after)
        assert set(refs) == {"x", "y"}
        assert refs["x"].base_.same_as(refs["y"].base_)

    def test_memory_capacity_reports_error(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self, a: pl.Tensor[[512, 256], pl.FP32], out: pl.Out[pl.Tensor[[512, 256], pl.FP32]]
            ) -> pl.Tensor[[512, 256], pl.FP32]:
                for i, (acc,) in pl.pipeline(4, stage=3, init_values=(out,)):
                    x: pl.Tile[[128, 256], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 128, 0], [128, 256])
                    y: pl.Tile[[128, 256], pl.FP32, pl.Mem.Vec] = pl.add(x, 1.0)
                    nxt = pl.store(y, [i * 128, 0], acc)
                    result = pl.yield_(nxt)
                return result

        after = _run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True)
        with pytest.raises(ValueError, match="Vec buffer usage.*exceeds"):
            _plan_memory(after, allocate=True)


def _allocation_bases(program: ir.Program, names: set[str]) -> dict[str, ir.Var]:
    """Read allocation identities rather than relying on generated base names."""
    refs: dict[str, ir.Var] = {}

    class Storage(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            name = _source_name(op.var.name_hint)
            if name in names:
                assert isinstance(op.var.type, ir.TileType) and op.var.type.memref is not None
                assert name not in refs, "Each source allocation must have exactly one definition"
                refs[name] = op.var.type.memref.base_
            super().visit_assign_stmt(op)

    Storage().visit_program(program)
    assert set(refs) == names
    return refs


@pytest.mark.parametrize("software", [False, True])
@pytest.mark.parametrize("reduction", ["row_max", "row_sum"])
def test_legacy_scratch_isolation_is_scoped_to_lowered_pipelines(software, reduction):
    @pl.program
    class ScratchLifetime:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(self, x: pl.Tensor[[64, 16], pl.FP32], y: pl.Tensor[[64, 16], pl.FP32]):
            dead: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [16, 16])
            _prior_store = pl.tile.store(dead, [0, 0], y)
            scratch: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.create(
                [16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            for i in pl.range(4, attrs={"software_pipeline_slots": 3}):
                value: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [i * 16, 0], [16, 16])
                reduced: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(value, scratch)
                _stored = pl.tile.store(reduced, [i * 16, 0], y)

    refs = {}

    class Allocations(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if op.var.name_hint in {"dead", "scratch"}:
                assert isinstance(op.var.type, ir.TileType) and op.var.type.memref is not None
                refs[op.var.name_hint] = op.var.type.memref.base_
            super().visit_assign_stmt(op)

    source = ir.python_print(ScratchLifetime).replace("tile.row_max", f"tile.{reduction}")
    if not software:
        source = source.replace(', attrs={"software_pipeline_slots": 3}', "")
    Allocations().visit_program(_plan_memory(parse(source)))
    assert set(refs) == {"dead", "scratch"}
    # Legacy reduction scratch is recognized only in transformed loops.
    # A previous MTE3 reader must not share the loop's hidden scratch writes.
    # Ordinary loops retain their existing allocation policy.
    assert refs["dead"].same_as(refs["scratch"]) is not software


def test_static_pipeline_marks_steady_loop_and_isolates_scratch():
    """The static builder must retain the same workspace isolation as dynamic loops."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(self, x: pl.Tensor[[128, 16], pl.FP32], y: pl.Tensor[[128, 16], pl.FP32]):
            dead: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [16, 16])
            _prior_store = pl.tile.store(dead, [0, 0], y)
            scratch: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.create(
                [16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            for i in pl.pipeline(8, stage=3):
                value: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [i * 16, 0], [16, 16])
                reduced: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(value, scratch)
                _stored = pl.tile.store(reduced, [i * 16, 0], y)

    after = _plan_memory(_run_to_slots(Before, passes.MemoryPlanner.PYPTO, software=True))
    loops = [stmt for stmt in _walk_stmts(after) if isinstance(stmt, ir.ForStmt)]
    refs = _allocation_bases(after, {"dead", "scratch"})
    assert len(loops) == 1 and loops[0].kind == ir.ForKind.Sequential
    assert loops[0].attrs["software_pipeline_slots"] == 3
    assert set(refs) == {"dead", "scratch"}
    assert not refs["dead"].same_as(refs["scratch"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
