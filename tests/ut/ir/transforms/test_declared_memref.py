# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for author-declared allocations (``pl.Tile[..., pl.MemRef("name"), ...]``).

The one-argument ``pl.MemRef`` form is manual memory-reuse control: tiles bound
to the same declared allocation share it, and ``MemoryReuse`` never packs
anything else in. The point is to stop the packer from coalescing tiles whose
lifetimes merely happen not to overlap — coalescing them creates a false
dependency that serializes work that could otherwise overlap.

The feature spans three passes, so the tests are grouped by what they pin down:
  * ``TestBinding``      — InitMemRef derives size / space and shares the base
  * ``TestReuseControl`` — MemoryReuse leaves declared allocations alone
  * ``TestPipeline``     — the declaration survives the real pass pipeline
  * ``TestRejects``      — declarations the compiler must refuse
"""

import re

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.language.parser.diagnostics import ParserTypeError


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


def _base_names(program: ir.Program) -> dict[str, str]:
    """Map every TileType assignment's var name to its allocation's base name."""
    return {name: memref.base_.name_hint for name, memref in _tile_memrefs(program).items()}


def _const_offset(memref: ir.MemRef) -> int:
    """The MemRef's byte offset as an int, asserting it really is constant."""
    offset = memref.byte_offset_
    assert isinstance(offset, ir.ConstInt), f"expected a constant offset, got {type(offset).__name__}"
    return offset.value


def _tile_byte_ranges(program: ir.Program) -> list[tuple[str, str, int, int]]:
    """(tile name, base name, start, end) for every addressed on-chip tile MemRef.

    Read off the printed program because after AllocateMemoryAddr the address IS
    the MemRef's byte offset, and the printed form names the base it belongs to.
    """
    pattern = re.compile(r"(\w+): pl\.Tile\[.*?pl\.MemRef\((\w+), pl\.const\((\d+), pl\.INT64\), (\d+)\)")
    ranges = []
    for line in program.as_python().splitlines():
        m = pattern.search(line)
        if m:
            offset, size = int(m.group(3)), int(m.group(4))
            ranges.append((m.group(1), m.group(2), offset, offset + size))
    return ranges


def _alloc_lines(program: ir.Program) -> list[str]:
    """The on-chip allocation lines of the printed program."""
    return [line.strip() for line in program.as_python().splitlines() if ".alloc(pl.Mem." in line]


def _run_memory_pipeline(program: ir.Program) -> ir.Program:
    """init_mem_ref -> materialize_semantic_aliases -> memory_reuse, as in the real pipeline."""
    return passes.memory_reuse()(passes.materialize_semantic_aliases()(passes.init_mem_ref()(program)))


def _run_full_pipeline(program: ir.Program, last_pass: str) -> ir.Program:
    """Run the Default strategy up to and including ``last_pass``."""
    manager = PassManager(OptimizationStrategy.Default)
    names = manager.pass_names
    stop = names.index(last_pass)
    for pass_obj in manager.passes[: stop + 1]:
        pipeline = passes.PassPipeline()
        pipeline.add_pass(pass_obj)
        program = pipeline.run(program)
    return program


def _run_dsa_rp_pipeline(program: ir.Program) -> ir.Program:
    """Run the allocation passes with DSA-RP owning placement."""
    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.DSA_RP,
    ):
        return passes.allocate_memory_addr()(
            passes.materialize_semantic_aliases()(passes.init_mem_ref()(program))
        )


class TestBinding:
    """InitMemRef honors the binding and derives what the author did not write."""

    def test_same_name_shares_one_allocation(self):
        """Two tiles naming one allocation end up on one base Ptr, so one alloc."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("scratch"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, pl.MemRef("scratch"), pl.Mem.Vec] = pl.exp(t0)
                return pl.store(t1, [0, 0], out)

        after = passes.init_mem_ref()(Before)
        bases = _base_names(after)
        assert bases["t0"] == "scratch"
        assert bases["t1"] == "scratch"
        assert len(_alloc_lines(after)) == 1

    def test_distinct_names_stay_distinct(self):
        """Different names are different allocations, even with disjoint lifetimes."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("ping"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, pl.MemRef("pong"), pl.Mem.Vec] = pl.exp(t0)
                return pl.store(t1, [0, 0], out)

        after = passes.init_mem_ref()(Before)
        bases = _base_names(after)
        assert bases["t0"] == "ping"
        assert bases["t1"] == "pong"
        assert len(_alloc_lines(after)) == 2

    def test_declared_name_does_not_capture_a_same_named_variable(self):
        """A declared name is its own namespace, not a Python variable name.

        The base-Ptr interner falls back to a scope lookup so `pl.MemRef(base, ...)`
        can name an alloc-defined Ptr. A declaration has no such Ptr — it is
        resolved before InitMemRef makes one — so that fallback could only
        misfire: a name matching an in-scope variable would take that variable, of
        arbitrary type, as the allocation base, and the alloc would then declare a
        Tensor-typed var as a base Ptr.
        """
        source = """
import pypto.language as pl


@pl.program
class Collide:
    @pl.function
    def main(self, a: pl.Tensor[[64, 64], pl.FP32],
             out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("a"), pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
        return pl.store(t0, [0, 0], out)
"""
        after = passes.init_mem_ref()(pl.parse_program(source))
        base = _tile_memrefs(after)["t0"].base_
        assert isinstance(base.type, ir.PtrType), f"declared base must be a Ptr, got {base.type}"
        allocs = _alloc_lines(after)
        assert len(allocs) == 1 and "pinned=True" in allocs[0], allocs
        assert "pl.Ptr = pl.tile.alloc" in allocs[0], f"alloc must bind a Ptr, got {allocs[0]}"

    def test_declared_allocation_is_referenced_by_variable(self):
        """The preferred form: declare once, reference by variable.

        A misspelled reference is a Python ``NameError`` rather than a silently
        distinct allocation, which is what the inline string form cannot give. An
        unnamed declaration takes the name of the variable it is bound to, so the
        name is written once.
        """
        ping = pl.MemRef()
        pong = pl.MemRef()

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, ping, pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, pong, pl.Mem.Vec] = pl.exp(t0)
                t2: pl.Tile[[64, 64], pl.FP32, ping, pl.Mem.Vec] = pl.exp(t1)
                return pl.store(t2, [0, 0], out)

        after = passes.init_mem_ref()(Before)
        bases = _base_names(after)
        assert bases["t0"] == bases["t2"] == "ping"
        assert bases["t1"] == "pong"
        assert len(_alloc_lines(after)) == 2

    def test_declared_allocation_keeps_its_declared_name(self):
        """An explicit name overrides the variable the declaration is bound to."""
        slot = pl.MemRef("l0c_ping")

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, slot, pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                return pl.store(t0, [0, 0], out)

        assert _base_names(passes.init_mem_ref()(Before))["t0"] == "l0c_ping"

    def test_declaration_is_an_explicit_flag_not_a_zero_size(self):
        """A zero-sized ordinary MemRef is a compiler allocation, not a binding.

        The declaration is carried by ``MemRef.is_pinned_``. Inferring it from
        ``size_ == 0`` instead would make the classification depend on a value
        the size field is merely unlikely to hold, rather than on what the IR
        actually says.
        """
        source = """
import pypto.language as pl


@pl.program
class Zero:
    @pl.function
    def main(self, a: pl.Tensor[[64, 64], pl.FP32],
             out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("mybuf", 0, 0), pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
        return pl.store(t0, [0, 0], out)
"""
        parsed = pl.parse_program(source)
        memref = _tile_memrefs(parsed)["t0"]
        assert memref.size_ == 0 and not memref.is_pinned_
        assert "pinned=True" not in passes.init_mem_ref()(parsed).as_python()

    def test_unresolved_declaration_prints_one_arg_and_round_trips(self):
        """The printed form of a declaration is the form the author wrote.

        A declaration carries no size or address to print — InitMemRef derives
        both — so printing the three-argument form would have to invent them, and
        would lose the distinction on reparse.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("scratch"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                return pl.store(t0, [0, 0], out)

        dumped = Before.as_python()
        assert 'pl.MemRef("scratch")' in dumped, dumped
        ir.assert_structural_equal(Before, pl.parse_program(dumped))
        # Resolution consumes the binding: the pinned alloc carries it from here on.
        assert not any(mr.is_pinned_ for mr in _tile_memrefs(passes.init_mem_ref()(Before)).values())

    def test_binds_a_transpose_output(self):
        """`tile.transpose` owns its output allocation, so it may be bound.

        It inherits the input's memory *space*, but `pto.ttrans` is registered
        `not_inplace_safe()` — the permute lands in a fresh buffer. Treating
        space inheritance as allocation inheritance would refuse a binding the
        hardware has no problem with.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 32], pl.FP32],
                out: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                t0: pl.Tile[[64, 32], pl.FP32, pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 32], target_memory=pl.Mem.Vec
                )
                tr: pl.Tile[[32, 64], pl.FP32, pl.MemRef("trans"), pl.Mem.Vec] = pl.tile.transpose(t0, 0, 1)
                return pl.store(tr, [0, 0], out)

        after = passes.init_mem_ref()(Before)
        assert _base_names(after)["tr"] == "trans"

    def test_size_is_the_largest_bound_tile(self):
        """The author writes no byte count: it is sized to hold any member."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out_big: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                out_small: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> tuple[pl.Tensor[[64, 64], pl.FP32], pl.Tensor[[32, 32], pl.FP32]]:
                big: pl.Tile[[64, 64], pl.FP32, pl.MemRef("scratch"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                r_big: pl.Tensor[[64, 64], pl.FP32] = pl.store(big, [0, 0], out_big)
                # `big` is dead by now, so `small` may legally take over the allocation.
                small: pl.Tile[[32, 32], pl.FP32, pl.MemRef("scratch"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [32, 32], target_memory=pl.Mem.Vec
                )
                r_small: pl.Tensor[[32, 32], pl.FP32] = pl.store(small, [0, 0], out_small)
                return r_big, r_small

        after = passes.init_mem_ref()(Before)
        memrefs = _tile_memrefs(after)
        # 64*64*4 == 16384 dominates 32*32*4 == 4096; both members see that size.
        assert memrefs["big"].size_ == 16384
        assert memrefs["small"].size_ == 16384

    def test_allocation_is_marked_pinned(self):
        """The alloc carries `pinned=True` so later passes can tell it apart."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("scratch"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                return pl.store(t0, [0, 0], out)

        after = passes.init_mem_ref()(Before)
        allocs = _alloc_lines(after)
        assert len(allocs) == 1
        assert "pinned=True" in allocs[0]

    def test_pinned_allocation_round_trips(self):
        """The printed `pinned=True` form re-parses to a structurally equal program."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("scratch"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                return pl.store(t0, [0, 0], out)

        after = passes.init_mem_ref()(Before)
        reparsed = pl.parse_program(after.as_python())
        ir.assert_structural_equal(after, reparsed)


class TestSlots:
    """A declaration may hold N equally-sized slots, selected by subscript.

    The point is a ping-pong the packer cannot collapse: the slots are one
    allocation, so they are contiguous and uniformly sized, and the index may be
    a runtime value so a rotation needs no unrolling.
    """

    def test_slots_are_one_allocation_at_distinct_offsets(self):
        """N slots become one allocation of N x slot, addressed by offset."""
        l0c = pl.MemRef(slots=2)

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, l0c[0], pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, l0c[1], pl.Mem.Vec] = pl.exp(t0)
                return pl.store(t1, [0, 0], out)

        after = passes.init_mem_ref()(Before)
        memrefs = _tile_memrefs(after)
        slot_size = 64 * 64 * 4
        # One allocation, reserving both slots; the slots differ only by offset.
        assert len(_alloc_lines(after)) == 1
        assert f"{2 * slot_size}" in _alloc_lines(after)[0]
        assert memrefs["t0"].base_ is memrefs["t1"].base_
        assert _const_offset(memrefs["t0"]) == 0
        assert _const_offset(memrefs["t1"]) == slot_size
        # Each MemRef spans its OWN slot, so [offset, offset + size_) stays inside
        # the allocation and the two slots do not overlap for MayAlias.
        assert memrefs["t0"].size_ == memrefs["t1"].size_ == slot_size

    def test_slot_size_is_the_largest_tile_on_any_slot(self):
        """Slots are uniform, so one slot must hold the biggest bound tile.

        A per-slot size would make `index * slot_size` an inconsistent stride.
        """
        buf = pl.MemRef(slots=2)

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                big_out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                small_out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> tuple[pl.Tensor[[64, 64], pl.FP32], pl.Tensor[[32, 32], pl.FP32]]:
                big: pl.Tile[[64, 64], pl.FP32, buf[0], pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                r_big: pl.Tensor[[64, 64], pl.FP32] = pl.store(big, [0, 0], big_out)
                small: pl.Tile[[32, 32], pl.FP32, buf[1], pl.Mem.Vec] = pl.load(
                    a, [0, 0], [32, 32], target_memory=pl.Mem.Vec
                )
                r_small: pl.Tensor[[32, 32], pl.FP32] = pl.store(small, [0, 0], small_out)
                return r_big, r_small

        after = passes.init_mem_ref()(Before)
        memrefs = _tile_memrefs(after)
        slot_size = 64 * 64 * 4
        # Slots are uniform: the allocation is 2 x the LARGEST bound tile...
        assert f"{2 * slot_size}" in _alloc_lines(after)[0]
        assert memrefs["big"].size_ == memrefs["small"].size_ == slot_size
        # ...so the small tile still starts one *full* slot in, not one small tile in.
        assert _const_offset(memrefs["small"]) == slot_size

    def test_runtime_slot_index_reaches_the_address(self, ascend_backend):
        """`l0c[i % 2]` rotates at runtime — the whole point of slots.

        The index survives as an expression through SSA renaming and pass
        rebuilds, and InitMemRef scales it into the byte offset, so the address
        is computed per iteration rather than baked to slot 0.
        """
        l0c = pl.MemRef(slots=2)

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                # `main` is not an InCore function, so InferTileMemorySpace never places
                # this parameter; spell the DDR carry buffer the test is about.
                seed: pl.Tile[[64, 64], pl.FP32, pl.Mem.DDR],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                for i, (acc_i,) in pl.range(4, init_values=(seed,)):
                    t: pl.Tile[[64, 64], pl.FP32, l0c[i % 2], pl.Mem.Vec] = pl.load(
                        a, [i * 64, 0], [64, 64], target_memory=pl.MemorySpace.Vec
                    )
                    acc_next: pl.Tile[[64, 64], pl.FP32] = pl.add(acc_i, t)
                    r = pl.yield_(acc_next)
                out: pl.Tensor[[64, 64], pl.FP32] = pl.store(r, [0, 0], output)
                return out

        # Through AllocateMemoryAddr: a dynamic address must survive, not collapse
        # to a constant (which would silently address slot 0 every iteration).
        after = _run_full_pipeline(Before, "AllocateMemoryAddr")
        rotated = [
            mr for name, mr in _tile_memrefs(after).items() if not isinstance(mr.byte_offset_, ir.ConstInt)
        ]
        assert rotated, "the runtime slot address folded to a constant"

        dsa_after = _run_dsa_rp_pipeline(Before)
        dsa_rotated = [
            mr
            for name, mr in _tile_memrefs(dsa_after).items()
            if not isinstance(mr.byte_offset_, ir.ConstInt)
        ]
        assert dsa_rotated, "DSA-RP folded the runtime slot address to a constant"

    def test_unsubscripted_declaration_is_unchanged(self):
        """A declaration with one slot behaves exactly as before slots existed."""
        scratch = pl.MemRef()

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, scratch, pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                return pl.store(t0, [0, 0], out)

        memrefs = _tile_memrefs(passes.init_mem_ref()(Before))
        assert memrefs["t0"].size_ == 64 * 64 * 4
        assert _const_offset(memrefs["t0"]) == 0

    def test_different_slots_may_be_live_together(self, ascend_backend):
        """Two slots co-live is the ping-pong, not a conflict.

        The co-liveness check keys on the slot, not the allocation; keying on the
        allocation would reject exactly the pattern slots exist for.
        """
        l0c = pl.MemRef(slots=2)

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, l0c[0], pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, l0c[1], pl.Mem.Vec] = pl.exp(t0)
                t2: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.add(t0, t1)
                return pl.store(t2, [0, 0], out)

        bases = set(_base_names(_run_memory_pipeline(Before)).values())
        assert "l0c" in bases

    def test_slots_reserve_the_whole_allocation(self, ascend_backend):
        """A multi-slot declaration must reserve every slot, not just the largest one.

        Each slot MemRef is sized to its own slot — that is what keeps `MayAlias`
        from calling the two halves of a ping-pong aliased. The *buffer* underneath
        them is still `slots x slot_size`, and the address allocator has to reserve
        that much: sizing the buffer from the largest member reserves one slot and
        lets the next allocation land exactly on top of slot 1.

        The declaration is named `aaa` so it sorts first — addresses are handed out
        in name order, so this puts the other buffers after it, where a short
        reservation collides.
        """
        aaa = pl.MemRef(slots=2)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                o1: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                o2: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64, 64], pl.FP32], pl.Tensor[[64, 64], pl.FP32]]:
                m0: pl.Tile[[64, 64], pl.FP32, aaa[0], pl.Mem.Vec] = pl.tile.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                m1: pl.Tile[[64, 64], pl.FP32, aaa[1], pl.Mem.Vec] = pl.tile.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                x: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                y: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                s1: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(m0, x)
                s2: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(m1, y)
                r1: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(s1, [0, 0], o1)
                r2: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(s2, [0, 0], o2)
                return r1, r2

        # Driven directly: AllocateMemoryAddr only runs on InCore functions, and the
        # Default strategy skips it entirely under the PTOAS planner.
        placements = {
            "PYPTO": passes.allocate_memory_addr()(_run_memory_pipeline(Before)),
            "DSA_RP": _run_dsa_rp_pipeline(Before),
        }
        for planner, after in placements.items():
            ranges = _tile_byte_ranges(after)
            assert len(ranges) >= 4, f"{planner}: expected addressed tiles, got {ranges}"
            # Slot 1 must sit inside its own allocation's reservation, so nothing
            # on a different base may overlap it.
            for name_a, base_a, start_a, end_a in ranges:
                for name_b, base_b, start_b, end_b in ranges:
                    if base_a >= base_b:
                        continue
                    assert not (start_a < end_b and start_b < end_a), (
                        f"{planner}: {name_a} [{start_a}, {end_a}) on '{base_a}' overlaps "
                        f"{name_b} [{start_b}, {end_b}) on '{base_b}'"
                    )

    def test_slots_round_trip(self):
        """The printed form carries both `slots=` and the subscript.

        Without `slots=` the reparsed subscript would be out of range; without the
        subscript every slot would collapse onto slot 0.
        """
        l0c = pl.MemRef(slots=2)

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, l0c[0], pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, l0c[1], pl.Mem.Vec] = pl.exp(t0)
                return pl.store(t1, [0, 0], out)

        dumped = Before.as_python()
        assert 'pl.MemRef("l0c", slots=2)[1]' in dumped, dumped
        ir.assert_structural_equal(Before, pl.parse_program(dumped))

    def test_slots_survive_serialization_embedded_in_a_tile_type(self):
        """`ir.serialize` / `ir.deserialize` must keep a slot binding a declaration.

        A MemRef inside a TileType goes through the type serializer, not the node
        one, so the standalone-MemRef tests do not cover it. Two things have to
        survive, and each fails differently:

        * the slot fields — dropping them takes `is_pinned_` too, so the
          declaration reads back as an ordinary compiler allocation that
          InitMemRef no longer treats as declared;
        * the **shared base identity** — allocation identity is base_ *pointer*
          identity, but the wire format names the base, so a reader that mints a
          fresh Var per MemRef splits one declaration into one allocation per
          slot. Both slots are needed to see it; a single-slot program round trips
          fine either way.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("buf", slots=2)[0], pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, pl.MemRef("buf", slots=2)[1], pl.Mem.Vec] = pl.exp(t0)
                return pl.store(t1, [0, 0], out)

        after = ir.deserialize(ir.serialize(Before))
        assert isinstance(after, ir.Program)
        ir.assert_structural_equal(after, Before)

        memrefs = _tile_memrefs(after)
        for name, slot in (("t0", 0), ("t1", 1)):
            memref = memrefs[name]
            assert memref.is_pinned_, f"'{name}' came back as a compiler allocation"
            assert memref.slot_count_ == 2
            assert isinstance(memref.slot_index_, ir.ConstInt) and memref.slot_index_.value == slot
        # The two slots must still name ONE allocation...
        assert memrefs["t0"].base_ is memrefs["t1"].base_, "the round trip split the shared base"
        # ...which is what makes InitMemRef emit a single allocation for the set.
        resolved = passes.init_mem_ref()(after)
        assert len(_alloc_lines(resolved)) == 1, _alloc_lines(resolved)

    def test_runtime_slot_index_round_trips_under_variable_renaming(self):
        """A slot index must print with the *disambiguated* name of the var it names.

        Two sibling loops both written `i` are distinct Vars, so the printer renames
        one of them. The slot index lives inside a type annotation and so prints
        through its own printer — if that printer does not inherit the rename map it
        emits the bare `i`, which on reparse rebinds the address to the other loop's
        variable and silently changes which slot is read.
        """
        l0c = pl.MemRef(slots=2)

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                for i in pl.range(2):
                    x: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                        a, [i * 64, 0], [64, 64], target_memory=pl.MemorySpace.Vec
                    )
                    pl.store(x, [0, 0], out)
                for i in pl.range(2):
                    t: pl.Tile[[64, 64], pl.FP32, l0c[i % 2], pl.Mem.Vec] = pl.load(
                        a, [i * 64, 0], [64, 64], target_memory=pl.MemorySpace.Vec
                    )
                    pl.store(t, [0, 0], out)
                return out

        dumped = Before.as_python()
        # Whatever the second loop variable is renamed to, the slot index must use
        # that same name — not the first loop's.
        second_loop = dumped.split("pl.range(2):")[2]
        loop_var = dumped.split("pl.range(2):")[1].rsplit("for ", 1)[1].split(" in ")[0]
        assert f'l0c", slots=2)[{loop_var} % 2]' in second_loop, dumped
        ir.assert_structural_equal(Before, pl.parse_program(dumped))


class TestSlotInvariants:
    """Things outside the passes that must keep a declaration a declaration."""

    def test_substitution_preserves_the_declaration_and_follows_the_slot_index(self):
        """The generic mutator must carry the slot fields and rewrite the index.

        `substitute_expr` rebuilds any MemRef whose base or offset changed. Rebuilding
        it from base/offset/size alone silently demotes a declaration to a compiler
        allocation mid-pipeline; and a slot index naming a substituted Var has to
        follow that substitution or it dangles on the old Var.
        """
        span = ir.Span("f.py", 1, 0, 1, 1)
        i = ir.Var("i", ir.ScalarType(ir.INDEX), span)
        j = ir.Var("j", ir.ScalarType(ir.INDEX), span)
        base = ir.Var("l0c", ir.PtrType(), span)
        other_base = ir.Var("other", ir.PtrType(), span)
        declared = ir.MemRef(base, 0, 0, span, True, 2, i)

        # Substituting the *base* must not drop the declaration fields.
        rebased = ir.substitute_expr(declared, [(base, other_base)])
        assert isinstance(rebased, ir.MemRef)
        assert rebased.is_pinned_, "substitution demoted a declaration to a compiler allocation"
        assert rebased.slot_count_ == 2
        assert rebased.slot_index_ is i

        # Substituting the *index* must rewrite it.
        reindexed = ir.substitute_expr(declared, [(i, j)])
        assert isinstance(reindexed, ir.MemRef)
        assert reindexed.slot_index_ is j, "slot index still refers to the old Var"
        assert reindexed.is_pinned_ and reindexed.slot_count_ == 2

    def test_declaration_ctor_keeps_its_positional_arguments(self):
        """`slots` is a keyword; it must not displace the existing positional `span`."""
        span = ir.Span("f.py", 1, 0, 1, 1)
        assert ir.MemRef(span).is_pinned_
        assert ir.MemRef("buf", span).base_.name_hint == "buf"
        assert ir.MemRef(slots=2).slot_count_ == 2
        assert ir.MemRef("buf", slots=2).slot_count_ == 2

    def test_rejects_a_slot_count_that_overflows_the_total_size(self):
        """`slots x slot_size` is author-controlled, so wrapping must not size a buffer.

        Wrapping would turn an absurd request into a *small* allocation and then hand
        out addresses inside it.
        """
        huge = pl.MemRef(slots=2**60)

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t: pl.Tile[[64, 64], pl.FP32, huge[0], pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                return pl.store(t, [0, 0], out)

        with pytest.raises(ValueError, match="overflows a 64-bit size"):
            passes.init_mem_ref()(Before)

    def test_unbound_slots_still_count_against_the_space_limit(self, ascend_backend):
        """A slot nothing is bound to still occupies memory.

        Each slot MemRef spans only its own slot, so 12 slots with one tile bound
        would be counted as one slot if the footprint were reconstructed from the
        addressed tiles — and a buffer well over the space's capacity would compile
        clean.
        """
        big = pl.MemRef(slots=12)  # 12 x 16 KB = 192 KB > the Vec space

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t: pl.Tile[[64, 64], pl.FP32, big[0], pl.Mem.Vec] = pl.tile.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                return pl.tile.store(t, [0, 0], out)

        with pytest.raises(ValueError, match="exceeds platform limit"):
            passes.allocate_memory_addr()(_run_memory_pipeline(Before))

    def test_a_runtime_slot_index_does_not_hide_an_oversized_allocation(self, ascend_backend):
        """The capacity check must not depend on any tile's address being constant.

        With a runtime index there is no constant offset to add up, so a footprint
        reconstructed from addressed tiles sees nothing at all — and the same 192 KB
        buffer that is rejected above would sail through.
        """
        big = pl.MemRef(slots=12)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[768, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                for i in pl.range(12):
                    t: pl.Tile[[64, 64], pl.FP32, big[i % 12], pl.Mem.Vec] = pl.tile.load(
                        a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    out = pl.tile.store(t, [0, 0], out)
                return out

        with pytest.raises(ValueError, match="exceeds platform limit"):
            passes.allocate_memory_addr()(_run_memory_pipeline(Before))

    def test_binding_only_a_high_slot_does_not_overstate_the_footprint(self, ascend_backend):
        """Nothing forces the author to use slot 0, and skipping it costs nothing.

        `slots=8` reserves 128 KB, which fits. Deriving the buffer's base from the
        lowest bound tile would place it at slot 7's address and count 240 KB, so a
        legal program would be rejected.
        """
        hi = pl.MemRef(slots=8)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t: pl.Tile[[64, 64], pl.FP32, hi[7], pl.Mem.Vec] = pl.tile.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                return pl.tile.store(t, [0, 0], out)

        after = passes.allocate_memory_addr()(_run_memory_pipeline(Before))
        assert f"{8 * 64 * 64 * 4}" in _alloc_lines(after)[0], _alloc_lines(after)


class TestSlotRejects:
    """Slot subscripts the compiler must refuse."""

    def test_rejects_out_of_range_constant_slot(self):
        """A constant index outside the declared count is a compile-time error."""
        l0c = pl.MemRef(slots=2)

        with pytest.raises(ParserTypeError, match="out of range"):

            @pl.program
            class Before:
                @pl.function
                def main(
                    self,
                    a: pl.Tensor[[64, 64], pl.FP32],
                    out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                ) -> pl.Tensor[[64, 64], pl.FP32]:
                    t: pl.Tile[[64, 64], pl.FP32, l0c[5], pl.Mem.Vec] = pl.load(
                        a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    return pl.store(t, [0, 0], out)

    def test_rejects_disagreeing_slot_counts_with_disjoint_uses(self):
        """One name, one slot count — even when no later use re-checks it.

        The "already recorded" sentinel must be a count no declaration can carry.
        With 1 as the sentinel, an unsubscripted declaration recorded first never
        consumed it, so a later `slots=2` on the same name silently overwrote it.

        Overlapping uses hide that: the collector visits a Var at every occurrence,
        so a 1-slot tile still live across the 2-slot declaration gets re-checked
        and trips the mismatch anyway. Here each tile is fully consumed before the
        next declaration appears, so the overwrite is the only thing standing
        between the author and an allocation sized from a declaration they
        contradicted.
        """
        with pytest.raises(ValueError, match="disagree on how many slots"):

            @pl.program
            class Before:
                @pl.function
                def main(
                    self,
                    a: pl.Tensor[[64, 64], pl.FP32],
                    o1: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                    o2: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                ) -> tuple[pl.Tensor[[64, 64], pl.FP32], pl.Tensor[[64, 64], pl.FP32]]:
                    t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("buf"), pl.Mem.Vec] = pl.load(
                        a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    r0: pl.Tensor[[64, 64], pl.FP32] = pl.store(t0, [0, 0], o1)
                    t1: pl.Tile[[64, 64], pl.FP32, pl.MemRef("buf", slots=2)[1], pl.Mem.Vec] = pl.load(
                        a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    r1: pl.Tensor[[64, 64], pl.FP32] = pl.store(t1, [0, 0], o2)
                    return r0, r1

            passes.init_mem_ref()(Before)

    def test_rejects_zero_slots_at_construction(self):
        """`slots=0` is refused where it is written, not as a 0-byte allocation later."""
        with pytest.raises(ValueError, match="at least one slot"):
            pl.MemRef(slots=0)
        with pytest.raises(ValueError, match="at least one slot"):
            pl.MemRef("named", slots=0)

    def test_rejects_subscripting_a_single_slot_declaration(self):
        """Subscripting something with one slot is a mistake worth naming."""
        scratch = pl.MemRef()

        with pytest.raises(ParserTypeError, match="single slot"):

            @pl.program
            class Before:
                @pl.function
                def main(
                    self,
                    a: pl.Tensor[[64, 64], pl.FP32],
                    out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                ) -> pl.Tensor[[64, 64], pl.FP32]:
                    t: pl.Tile[[64, 64], pl.FP32, scratch[0], pl.Mem.Vec] = pl.load(
                        a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    return pl.store(t, [0, 0], out)

    def test_rejects_two_tiles_co_live_on_one_slot(self, ascend_backend):
        """Same slot, overlapping lifetimes — still data corruption."""
        l0c = pl.MemRef(slots=2)

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, l0c[0], pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, l0c[0], pl.Mem.Vec] = pl.exp(t0)
                t2: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.add(t0, t1)
                return pl.store(t2, [0, 0], out)

        with pytest.raises(ValueError, match="same slot"):
            _run_memory_pipeline(Before)


class TestReuseControl:
    """MemoryReuse packs unbound tiles as before, and leaves declared ones alone."""

    def test_unbound_tiles_are_still_packed(self, ascend_backend):
        """Baseline: without bindings the packer coalesces the whole chain."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t0)
                t2: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t1)
                return pl.store(t2, [0, 0], out)

        bases = set(_base_names(_run_memory_pipeline(Before)).values())
        assert len(bases) == 1, f"expected the packer to coalesce the chain, got {bases}"

    def test_bound_tiles_are_not_packed(self, ascend_backend):
        """The same chain, declared: three allocations survive — no false deps."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("in_buf"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, pl.MemRef("mid_buf"), pl.Mem.Vec] = pl.exp(t0)
                t2: pl.Tile[[64, 64], pl.FP32, pl.MemRef("out_buf"), pl.Mem.Vec] = pl.exp(t1)
                return pl.store(t2, [0, 0], out)

        bases = _base_names(_run_memory_pipeline(Before))
        assert bases["t0"] == "in_buf"
        assert bases["t1"] == "mid_buf"
        assert bases["t2"] == "out_buf"

    def test_explicit_sharing_is_preserved(self, ascend_backend):
        """Author-chosen sharing survives: t0 and t2 on one allocation, t1 on another.

        Also pins the touching-lifetimes rule: t0's last read is the statement
        producing t1, and t2 is defined after that, so the overlap check accepts
        the pair rather than treating the shared allocation as a conflict.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("ping"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, pl.MemRef("pong"), pl.Mem.Vec] = pl.exp(t0)
                t2: pl.Tile[[64, 64], pl.FP32, pl.MemRef("ping"), pl.Mem.Vec] = pl.exp(t1)
                return pl.store(t2, [0, 0], out)

        bases = _base_names(_run_memory_pipeline(Before))
        assert bases["t0"] == bases["t2"] == "ping"
        assert bases["t1"] == "pong"

    def test_dsa_rp_preserves_legal_explicit_sharing(self, ascend_backend):
        """DSA-RP accepts non-co-live values explicitly sharing one allocation."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("ping"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, pl.MemRef("pong"), pl.Mem.Vec] = pl.exp(t0)
                t2: pl.Tile[[64, 64], pl.FP32, pl.MemRef("ping"), pl.Mem.Vec] = pl.exp(t1)
                return pl.store(t2, [0, 0], out)

        memrefs = _tile_memrefs(_run_dsa_rp_pipeline(Before))
        assert memrefs["t0"].base_.name_hint == memrefs["t2"].base_.name_hint == "ping"
        assert memrefs["t1"].base_.name_hint == "pong"
        assert isinstance(memrefs["t0"].byte_offset_, ir.ConstInt)
        assert isinstance(memrefs["t1"].byte_offset_, ir.ConstInt)
        assert isinstance(memrefs["t2"].byte_offset_, ir.ConstInt)
        assert memrefs["t0"].byte_offset_.value == memrefs["t2"].byte_offset_.value
        assert memrefs["t0"].byte_offset_.value != memrefs["t1"].byte_offset_.value

    def test_unbound_tiles_never_join_a_declared_alloc(self, ascend_backend):
        """An unbound tile packs with other unbound tiles, never into a declared one."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mine: pl.Tile[[64, 64], pl.FP32, pl.MemRef("mine"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                free0: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(mine)
                free1: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(free0)
                return pl.store(free1, [0, 0], out)

        bases = _base_names(_run_memory_pipeline(Before))
        assert bases["mine"] == "mine"
        assert bases["free0"] != "mine"
        assert bases["free1"] != "mine"

    def test_dsa_rp_keeps_declared_ranges_isolated(self, ascend_backend):
        """DSA-RP preserves declarations although it skips MemoryReuse.

        Every value in this chain has a lifetime-compatible handoff to the next
        and would ordinarily fit at one address. The two declared allocations
        must remain disjoint from each other and from the unbound allocation.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("in_buf"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, pl.MemRef("mid_buf"), pl.Mem.Vec] = pl.exp(t0)
                t2: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t1)
                return pl.store(t2, [0, 0], out)

        after = _run_dsa_rp_pipeline(Before)
        memrefs = _tile_memrefs(after)

        def allocation_range(base_name: str) -> tuple[int, int]:
            matches = [memref for memref in memrefs.values() if memref.base_.name_hint == base_name]
            assert matches, f"missing allocation {base_name}: {memrefs}"
            memref = matches[0]
            assert isinstance(memref.byte_offset_, ir.ConstInt)
            begin = memref.byte_offset_.value
            return begin, begin + memref.size_

        in_range = allocation_range("in_buf")
        mid_range = allocation_range("mid_buf")
        unbound = next(
            memref for memref in memrefs.values() if memref.base_.name_hint not in {"in_buf", "mid_buf"}
        )
        assert isinstance(unbound.byte_offset_, ir.ConstInt)
        unbound_range = (
            unbound.byte_offset_.value,
            unbound.byte_offset_.value + unbound.size_,
        )

        for first, second in (
            (in_range, mid_range),
            (in_range, unbound_range),
            (mid_range, unbound_range),
        ):
            assert first[1] <= second[0] or second[1] <= first[0], (
                f"declared allocation ranges must be disjoint: {first} vs {second}"
            )


class TestPipeline:
    """The binding must survive every pass between the parser and InitMemRef."""

    def test_binding_survives_the_default_pipeline(self, ascend_backend):
        """ConvertToSSA and friends must carry the MemRef through, not drop it."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("ping"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, pl.MemRef("pong"), pl.Mem.Vec] = pl.exp(t0)
                return pl.store(t1, [0, 0], out)

        after = _run_full_pipeline(Before, "AllocateMemoryAddr")
        bases = set(_base_names(after).values())
        assert bases == {"ping", "pong"}, f"binding lost in the pipeline, got {bases}"

    def test_binding_survives_nd_flattening(self, ascend_backend):
        """FlattenTileNdTo2D rebuilds the TileType; it must carry the binding over.

        The flattened tile is the same storage, so dropping the MemRef there would
        silently un-bind every ND user-bound tile — no diagnostic, feature gone.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[4, 16, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 16, 64], pl.FP32]],
            ) -> pl.Tensor[[4, 16, 64], pl.FP32]:
                t0: pl.Tile[[4, 16, 64], pl.FP32, pl.MemRef("nd_buf"), pl.Mem.Vec] = pl.load(
                    a, [0, 0, 0], [4, 16, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[4, 16, 64], pl.FP32, pl.MemRef("nd_buf"), pl.Mem.Vec] = pl.exp(t0)
                return pl.store(t1, [0, 0, 0], out)

        after = _run_full_pipeline(Before, "InitMemRef")
        bases = set(_base_names(after).values())
        assert bases == {"nd_buf"}, f"ND binding lost during flattening, got {bases}"

    def test_binding_survives_an_spmd_cube_kernel(self, ascend_backend):
        """A real on-core kernel: pl.spmd over the Mat/Left/Right/Acc chain.

        Every tile here is already 2D, so the ND-flatten path never runs. The
        binding instead has to ride two rebuilds that only fire once a function is
        actually lowered on-core: the ≤2D re-deduction in FlattenTileNdTo2D (whose
        args get substituted to partition views, so nothing passes through
        untouched) and the LHS-Var type sync in InferTileMemorySpace. Both rebuild
        from the RHS Call, whose deduced type never carries a MemRef — reading it
        from there silently un-binds every tile in the only kernel shape that
        matters in practice, with no diagnostic.

        Two Acc slots is the point: one accumulator forces a TSTORE between
        consecutive TMATMULs (issue #2131).
        """
        m, k, n = 16, 128, 128

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                q: pl.Tensor[[m, k], pl.BF16],
                b: pl.Tensor[[k, 2 * n], pl.BF16],
                out: pl.Out[pl.Tensor[[m, 2 * n], pl.FP32]],
            ) -> pl.Tensor[[m, 2 * n], pl.FP32]:
                for _ in pl.spmd(1, name_hint="cube"):
                    q_l1: pl.Tile[[m, k], pl.BF16, pl.Mem.Mat] = pl.load(
                        q, [0, 0], [m, k], target_memory=pl.MemorySpace.Mat
                    )
                    q_l0: pl.Tile[[m, k], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        q_l1, 0, 0, [m, k], target_memory=pl.MemorySpace.Left
                    )
                    b_l1: pl.Tile[[k, 2 * n], pl.BF16, pl.Mem.Mat] = pl.load(
                        b, [0, 0], [k, 2 * n], target_memory=pl.MemorySpace.Mat
                    )
                    b0: pl.Tile[[k, n], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        b_l1, 0, 0, [k, n], target_memory=pl.MemorySpace.Right
                    )
                    acc0: pl.Tile[[m, n], pl.FP32, pl.MemRef("l0c_ping"), pl.Mem.Acc] = pl.tile.matmul(
                        q_l0, b0
                    )
                    r0: pl.Tensor[[m, 2 * n], pl.FP32] = pl.store(acc0, [0, 0], out)
                    b1: pl.Tile[[k, n], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        b_l1, 0, n, [k, n], target_memory=pl.MemorySpace.Right
                    )
                    acc1: pl.Tile[[m, n], pl.FP32, pl.MemRef("l0c_pong"), pl.Mem.Acc] = pl.tile.matmul(
                        q_l0, b1
                    )
                    r1 = pl.store(acc1, [0, n], r0)
                return r1

        after = _run_full_pipeline(Before, "AllocateMemoryAddr")
        bases = set(_base_names(after).values())
        assert {"l0c_ping", "l0c_pong"} <= bases, f"binding lost in an spmd kernel, got {bases}"
        pinned = [line for line in _alloc_lines(after) if "pinned=True" in line]
        assert len(pinned) == 2, f"expected two pinned Acc allocations, got {pinned}"

    def test_binding_survives_nd_tile_create_flattening(self, ascend_backend):
        """The rank>2 `tile.create` / `tile.full` rewrite must carry the binding too.

        That path re-deduces the 2D call through the OpRegistry, same as the
        tile.load and generic-op paths — and the deduced type carries no MemRef.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[4, 16, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 16, 64], pl.FP32]],
            ) -> pl.Tensor[[4, 16, 64], pl.FP32]:
                made: pl.Tile[[4, 16, 64], pl.FP32, pl.MemRef("made_buf"), pl.Mem.Vec] = pl.tile.create(
                    [4, 16, 64], pl.FP32, target_memory=pl.Mem.Vec
                )
                loaded: pl.Tile[[4, 16, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                    a, [0, 0, 0], [4, 16, 64], target_memory=pl.Mem.Vec
                )
                summed: pl.Tile[[4, 16, 64], pl.FP32, pl.Mem.Vec] = pl.add(made, loaded)
                return pl.store(summed, [0, 0, 0], out)

        after = _run_full_pipeline(Before, "InitMemRef")
        bases = set(_base_names(after).values())
        assert "made_buf" in bases, f"tile.full binding lost during flattening, got {bases}"

    def test_reparsed_dump_is_not_treated_as_declared(self, ascend_backend):
        """A post-allocation dump also carries MemRefs — those are the compiler's.

        The binding is recognised by the parser's size-0 "derive me" marker, not by
        "a MemRef exists". Re-running the passes over a printed program must not
        promote its compiler allocations into pinned, un-reusable ones.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t0)
                return pl.store(t1, [0, 0], out)

        dumped = passes.init_mem_ref()(Before).as_python()
        assert "pinned=True" not in dumped
        reparsed = pl.parse_program(dumped)
        # Re-running InitMemRef over the dump must still produce no pinned allocs.
        assert "pinned=True" not in passes.init_mem_ref()(reparsed).as_python()

    def test_declared_allocs_stay_on_distinct_bases(self, ascend_backend):
        """Two declarations survive to AllocateMemoryAddr as two separate allocations.

        Distinct base Ptrs are what "separate allocations" means here — the
        allocator assigns each base its own address range from there.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("ping"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, pl.MemRef("pong"), pl.Mem.Vec] = pl.exp(t0)
                return pl.store(t1, [0, 0], out)

        after = _run_full_pipeline(Before, "AllocateMemoryAddr")
        memrefs = _tile_memrefs(after)
        ping, pong = memrefs["t0__ssa_v0"], memrefs["t1__ssa_v0"]
        assert ping.base_.name_hint != pong.base_.name_hint
        assert ping.size_ == pong.size_ == 16384
        # Two allocations reach the allocator, neither folded into the other.
        assert len(_alloc_lines(after)) == 2


class TestRejects:
    """Declarations the compiler must refuse, each with a message that says why."""

    def test_rejects_reaching_one_declaration_through_two_names(self):
        """`b = a` is ambiguous once the variable supplies the name.

        An unnamed declaration is named after its variable, so two names for one
        object would silently become two allocations. Rejecting says which name
        it already goes by.
        """
        source = """
import pypto.language as pl

ping = pl.MemRef()
alias = ping


@pl.program
class Aliased:
    @pl.function
    def main(self, a: pl.Tensor[[64, 64], pl.FP32],
             out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        t0: pl.Tile[[64, 64], pl.FP32, ping, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
        t1: pl.Tile[[64, 64], pl.FP32, alias, pl.Mem.Vec] = pl.exp(t0)
        return pl.store(t1, [0, 0], out)
"""
        with pytest.raises(ParserTypeError, match="also referenced as"):
            pl.parse_program(source)

    def test_rejects_two_declarations_claiming_one_name(self):
        """The reverse ambiguity: two objects, one name, would become one alloc."""
        source = """
import pypto.language as pl

first = pl.MemRef("shared")
second = pl.MemRef("shared")


@pl.program
class Collide:
    @pl.function
    def main(self, a: pl.Tensor[[64, 64], pl.FP32],
             out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        t0: pl.Tile[[64, 64], pl.FP32, first, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
        t1: pl.Tile[[64, 64], pl.FP32, second, pl.Mem.Vec] = pl.exp(t0)
        return pl.store(t1, [0, 0], out)
"""
        with pytest.raises(ParserTypeError, match="both resolve to the name"):
            pl.parse_program(source)

    def test_rejects_binding_a_pipelined_tile(self, ascend_backend):
        """One declared allocation cannot back two in-flight pipeline stages.

        `stage=2` clones the body so iteration i and i+1 overlap; both clones name
        the same allocation, so the tile is co-live with itself. Declaring one and
        asking the compiler to multi-buffer it are mutually exclusive requests —
        explicit slots *replace* pipelining at that level, they do not stack on
        top of it. Rejecting says so; silently honoring one of the two would
        either corrupt data or quietly drop the binding.
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
                    t: pl.Tile[[64, 64], pl.FP32, pl.MemRef("staged"), pl.Mem.Vec] = pl.load(
                        a, [i, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(e, [i, 0], acc)
                    y = pl.yield_(nxt)
                return y

        with pytest.raises(ValueError, match="live at the same time"):
            _run_full_pipeline(Before, "MemoryReuse")

    def test_rejects_overlapping_lifetimes(self, ascend_backend):
        """Two co-live tiles on one allocation would corrupt data, not reuse it."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("ping"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                t1: pl.Tile[[64, 64], pl.FP32, pl.MemRef("pong"), pl.Mem.Vec] = pl.exp(t0)
                # Overwrites `ping` while t0 is still needed by the add below.
                t2: pl.Tile[[64, 64], pl.FP32, pl.MemRef("ping"), pl.Mem.Vec] = pl.exp(t1)
                t3: pl.Tile[[64, 64], pl.FP32, pl.MemRef("pong2"), pl.Mem.Vec] = pl.add(t0, t2)
                return pl.store(t3, [0, 0], out)

        with pytest.raises(ValueError, match="live at the same time"):
            _run_memory_pipeline(Before)
        with pytest.raises(ValueError, match="live at the same time"):
            _run_dsa_rp_pipeline(Before)

    def test_rejects_mixed_memory_space(self):
        """One allocation lives in one memory space; bound tiles must agree."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP16],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP16]],
            ) -> pl.Tensor[[64, 64], pl.FP16]:
                vec: pl.Tile[[64, 64], pl.FP16, pl.MemRef("shared"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                mat: pl.Tile[[64, 64], pl.FP16, pl.MemRef("shared"), pl.Mem.Mat] = pl.tile.move(
                    vec, target_memory=pl.Mem.Mat
                )
                back: pl.Tile[[64, 64], pl.FP16, pl.Mem.Vec] = pl.tile.move(mat, target_memory=pl.Mem.Vec)
                return pl.store(back, [0, 0], out)

        with pytest.raises(ValueError, match="same memory space"):
            passes.init_mem_ref()(Before)

    def test_rejects_binding_a_view_output(self):
        """A view already IS its source's allocation; it cannot be given its own."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[32, 128], pl.FP32]],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                view: pl.Tile[[32, 128], pl.FP32, pl.MemRef("elsewhere"), pl.Mem.Vec] = pl.reshape(
                    t0, [32, 128]
                )
                return pl.store(view, [0, 0], out)

        with pytest.raises(ValueError, match="lands in its source tile's allocation"):
            passes.init_mem_ref()(Before)

    def test_requires_explicit_memory_space(self):
        """TileType pairs a MemRef with a memory space; the annotation must say which."""
        source = """
import pypto.language as pl


@pl.program
class Bad:
    @pl.function
    def main(self, a: pl.Tensor[[64, 64], pl.FP32],
             out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef("scratch")] = pl.load(a, [0, 0], [64, 64])
        return pl.store(t0, [0, 0], out)
"""
        with pytest.raises(ParserTypeError, match="explicit memory space"):
            pl.parse_program(source)

    def test_rejects_non_literal_declared_name(self):
        """The name identifies the allocation at parse time, so it must be literal."""
        source = """
import pypto.language as pl

NAME = "scratch"


@pl.program
class Bad:
    @pl.function
    def main(self, a: pl.Tensor[[64, 64], pl.FP32],
             out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        t0: pl.Tile[[64, 64], pl.FP32, pl.MemRef(NAME), pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
        return pl.store(t0, [0, 0], out)
"""
        with pytest.raises(ParserTypeError, match="string literal"):
            pl.parse_program(source)


class TestLoopCarryRoundtrip:
    """A loop-carry init parameter placed in a compiler buffer must stay printable.

    ``InitMemRef`` gives the carry's init parameter the carry buffer's MemRef, and
    ``MaterializeSemanticAliases`` then folds the carried value onto that same
    buffer. Both steps put a parameter or a binding in a state the DSL surface
    syntax has exactly one way to spell, so both used to break the print -> parse
    -> structural-compare roundtrip the pass pipeline runs under.
    """

    @staticmethod
    def _carry_program():
        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                # `main` is not an InCore function, so InferTileMemorySpace never places
                # this parameter; spell the DDR carry buffer the test is about.
                seed: pl.Tile[[64, 64], pl.FP32, pl.Mem.DDR],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                for i, (acc_i,) in pl.range(4, init_values=(seed,)):
                    t: pl.Tile[[64, 64], pl.FP32] = pl.load(
                        a, [i * 64, 0], [64, 64], target_memory=pl.MemorySpace.Vec
                    )
                    acc_next: pl.Tile[[64, 64], pl.FP32] = pl.add(acc_i, t)
                    r = pl.yield_(acc_next)
                out: pl.Tensor[[64, 64], pl.FP32] = pl.store(r, [0, 0], output)
                return out

        return Before

    def test_signature_memref_base_prints_as_a_string(self, ascend_backend):
        """A parameter's MemRef base must never print as a bare forward reference.

        Python evaluates the signature before the body binds any name, so a base
        Ptr that the body allocates has to be spelled as a string there. Emitting
        the bare identifier made the printed program reparse with ``NameError``.
        """
        after = _run_full_pipeline(self._carry_program(), "InitMemRef")
        printed = after.as_python()
        param_line = next(line for line in printed.splitlines() if line.lstrip().startswith("seed"))
        alloc_line = next(line for line in printed.splitlines() if ".alloc(pl.Mem.DDR" in line)

        # The base really is allocated in the body, so the bare name would be a
        # forward reference in the signature...
        base = alloc_line.split(":")[0].strip()
        # ...and the parameter annotation therefore spells it as a string.
        assert f'pl.MemRef("{base}"' in param_line
        # The body, where the base *is* bound, keeps the bare-name form.
        assert f"pl.MemRef({base}," in printed

        # The quoted name must still resolve to the very Var the body allocates:
        # structural equality ignores MemRefs, so only an identity check catches
        # a reparse that hands the parameter a second, unrelated allocation.
        reparsed = pl.parse_program(printed)  # must not raise
        main = reparsed.get_function("main")
        assert main is not None
        assert isinstance(main.body, ir.SeqStmts)
        alloc = next(s for s in main.body.stmts if isinstance(s, ir.AssignStmt) and s.var.name_hint == base)
        seed_type = main.params[1].type
        assert isinstance(seed_type, ir.TileType)
        assert seed_type.memref is not None
        assert seed_type.memref.base_ is alloc.var

    def test_fixed_output_op_is_not_retargeted_across_memory_spaces(self, ascend_backend):
        """A Vec-only producer must keep its Vec buffer even under a DDR carry.

        The carry's init is a Tile parameter, so it lives in DDR, and the carry
        alias would drag the ``tile.add`` that feeds it onto that DDR buffer.
        But ``tile.add`` is registered ``set_output_memory(Vec)`` — it cannot
        write there. Retargeting it anyway would mint IR claiming a Vec-only op
        produces DDR, and would leave the bound Call's type disagreeing with its
        Var, a pair the DSL cannot spell (the printer emits only the Var's
        annotation, so reparsing retypes the Call and the roundtrip compares
        unequal). Declining leaves the carry yielding a different buffer than
        its init, which ``YieldFixupMutator`` reconciles with a real move.
        """
        after = passes.materialize_semantic_aliases()(_run_full_pipeline(self._carry_program(), "InitMemRef"))
        main = after.get_function("main")
        assert main is not None
        assert isinstance(main.body, ir.SeqStmts)
        loop = next(s for s in main.body.stmts if isinstance(s, ir.ForStmt))
        assert isinstance(loop.body, ir.SeqStmts)
        add_stmt = next(
            s
            for s in loop.body.stmts
            if isinstance(s, ir.AssignStmt)
            and isinstance(s.value, ir.Call)
            and s.value.op.name == ir.get_op("tile.add").name
        )
        var_type, value_type = add_stmt.var.type, add_stmt.value.type
        assert isinstance(var_type, ir.TileType)
        assert isinstance(value_type, ir.TileType)

        # The op's registered output space wins over the carry's DDR buffer...
        assert var_type.memory_space == ir.MemorySpace.Vec, "a Vec-only op was dragged into DDR"
        # ...and the binding stays internally consistent, so it round-trips.
        assert value_type.memory_space == var_type.memory_space


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
