# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for LowerPipelineToSlots — ``pl.pipeline`` via MemRef slots, not body copies.

The pass keeps ONE loop body and rebinds its top-level loads onto
``pl.MemRef(name, slots=F)[iv % F]``, then demotes the loop to Sequential. It is
self-gated on ``memory_planner=PTOAS`` and falls back to leaving a loop untouched
(for ``LowerPipelineLoops`` to replicate) whenever it cannot guarantee codegen
will accept the region.

Groups:
  * ``TestGating``   — the planner gate, and that a declined loop is left intact
  * ``TestBinding``  — the slot geometry actually written onto the tile
  * ``TestFallback`` — one case per expressible eligibility gate, each asserting NO
    slot binding
  * ``TestChaining`` — LowerPipelineLoops still replicates what this pass declined
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.arith import Analyzer
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

_PASS_NAME = "LowerPipelineToSlots"


def _run_to_slots(program: ir.Program, planner: passes.MemoryPlanner) -> ir.Program:
    """Run the Default strategy up to and including LowerPipelineToSlots.

    The pass needs tile-level IR (memory spaces inferred, structure normalized),
    so it cannot run standalone on a freshly parsed program. The PassManager is
    built inside the context because its construction reads the planner.
    """
    with passes.PassContext([], memory_planner=planner):
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
    """The planner and scheduling contracts decide whether the pass can run."""

    def test_explicit_accumulator_buffering_keeps_the_replication_path(self):
        """Rotating the loads must not discard the requested Acc scheduling."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 64], pl.BF16],
                b: pl.Tensor[[64, 256], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
            ) -> pl.Tensor[[16, 256], pl.FP32]:
                lhs = pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat)
                for i, (output,) in pl.pipeline(
                    4, stage=2, init_values=(out,), attrs={"pipeline_double_buffer_c": True}
                ):
                    rhs = pl.load(b, [0, i * 64], [64, 64], target_memory=pl.Mem.Mat)
                    product = pl.matmul(lhs, rhs)
                    drained = pl.store(product, [0, i * 64], output)
                    final = pl.yield_(drained)
                return final

        after = _run_to_slots(Before, passes.MemoryPlanner.PTOAS)
        assert len(_pipeline_loops(after)) == 1
        assert _slotted_memrefs(after) == {}
        assert _pipeline_loops(after)[0].attrs["pipeline_double_buffer_c"] is True

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
