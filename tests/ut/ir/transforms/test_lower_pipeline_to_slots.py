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
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
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
    with passes.PassContext([], memory_planner=planner, enable_software_pipeline=software):
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
    text = program.as_python()
    return text.count("pl.load(") + text.count("tile.load(")


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

    def test_non_unit_step(self):
        """``((iv - start) / step) % F`` is not an affine form ptoas matches."""

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

    def test_start_not_a_multiple_of_the_stage_count(self):
        """``iv % F`` only walks slot 0 first when ``start`` is a multiple of ``F``."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[320, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[320, 64], pl.FP32]],
            ) -> pl.Tensor[[320, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(1, 5, 1, stage=2, init_values=(out,)):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [i * 64, 0], [64, 64])
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[320, 64], pl.FP32] = pl.store(e, [i * 64, 0], acc)
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
        """A step-64 loop takes no slot, so it must still get its F body copies."""

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

        before_loads = _load_count(Before)
        after_slots = _run_to_slots(Before, passes.MemoryPlanner.PTOAS)
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
            replicated = passes.lower_pipeline_loops()(after_slots)
        assert _load_count(replicated) == 2 * before_loads


def _plan_memory(program, *, allocate=False):
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_software_pipeline=True):
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

        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_software_pipeline=True):
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
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_software_pipeline=True):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
