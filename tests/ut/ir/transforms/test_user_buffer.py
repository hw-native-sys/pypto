# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for user-owned buffers (``pl.Tile[..., pl.Buffer("name")]``).

A user buffer is manual memory-reuse control: tiles bound to the same buffer
share one allocation, and ``MemoryReuse`` never packs anything else into it.
The point is to stop the packer from coalescing tiles whose lifetimes merely
happen not to overlap — coalescing them creates a false dependency that
serializes work that could otherwise overlap.

The feature spans three passes, so the tests are grouped by what they pin down:
  * ``TestBinding``      — InitMemRef derives size / space and shares the base
  * ``TestReuseControl`` — MemoryReuse leaves user buffers alone
  * ``TestPipeline``     — the binding survives the real pass pipeline
  * ``TestRejects``      — bindings the compiler must refuse
"""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


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
    with passes.PassContext([], passes.VerificationLevel.NONE):
        for pass_obj in manager.passes[: stop + 1]:
            pipeline = passes.PassPipeline()
            pipeline.add_pass(pass_obj)
            program = pipeline.run(program)
    return program


class TestBinding:
    """InitMemRef honors the binding and derives what the author did not write."""

    def test_same_buffer_shares_one_allocation(self):
        """Two tiles naming one buffer end up on one base Ptr, so one allocation."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.Buffer("scratch"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64]
                )
                t1: pl.Tile[[64, 64], pl.FP32, pl.Buffer("scratch"), pl.Mem.Vec] = pl.exp(t0)
                return pl.store(t1, [0, 0], out)

        after = passes.init_mem_ref()(Before)
        bases = _base_names(after)
        assert bases["t0"] == "scratch"
        assert bases["t1"] == "scratch"
        assert len(_alloc_lines(after)) == 1

    def test_distinct_buffers_stay_distinct(self):
        """Different names are different buffers, even though lifetimes are disjoint."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.Buffer("ping"), pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                t1: pl.Tile[[64, 64], pl.FP32, pl.Buffer("pong"), pl.Mem.Vec] = pl.exp(t0)
                return pl.store(t1, [0, 0], out)

        after = passes.init_mem_ref()(Before)
        bases = _base_names(after)
        assert bases["t0"] == "ping"
        assert bases["t1"] == "pong"
        assert len(_alloc_lines(after)) == 2

    def test_buffer_name_does_not_capture_a_same_named_variable(self):
        """Buffer names are their own namespace, not Python variable names.

        The base-Ptr interner falls back to a scope lookup so `pl.MemRef(base, ...)`
        can name an alloc-defined Ptr. A buffer has no such Ptr — it is resolved
        before InitMemRef makes one — so that fallback could only misfire: a buffer
        named after an in-scope variable would take that variable, of arbitrary
        type, as its allocation base, and the alloc would then declare a
        Tensor-typed var as a base Ptr.
        """
        source = """
import pypto.language as pl


@pl.program
class Collide:
    @pl.function
    def main(self, a: pl.Tensor[[64, 64], pl.FP32],
             out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        t0: pl.Tile[[64, 64], pl.FP32, pl.Buffer("a"), pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
        return pl.store(t0, [0, 0], out)
"""
        after = passes.init_mem_ref()(pl.parse_program(source))
        base = _tile_memrefs(after)["t0"].base_
        assert isinstance(base.type, ir.PtrType), f"buffer base must be a Ptr, got {base.type}"
        allocs = _alloc_lines(after)
        assert len(allocs) == 1 and "pinned=True" in allocs[0], allocs
        assert "pl.Ptr = pl.tile.alloc" in allocs[0], f"alloc must bind a Ptr, got {allocs[0]}"

    def test_declared_buffer_object_is_referenced_by_variable(self):
        """The preferred form: declare once, reference by variable.

        A misspelled reference is a Python ``NameError`` rather than a silently
        distinct buffer, which is what the inline string form cannot give. An
        unnamed ``pl.Buffer()`` takes the name of the variable it is bound to,
        so the buffer is named once instead of twice.
        """
        ping = pl.Buffer()
        pong = pl.Buffer()

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, ping, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                t1: pl.Tile[[64, 64], pl.FP32, pong, pl.Mem.Vec] = pl.exp(t0)
                t2: pl.Tile[[64, 64], pl.FP32, ping, pl.Mem.Vec] = pl.exp(t1)
                return pl.store(t2, [0, 0], out)

        after = passes.init_mem_ref()(Before)
        bases = _base_names(after)
        assert bases["t0"] == bases["t2"] == "ping"
        assert bases["t1"] == "pong"
        assert len(_alloc_lines(after)) == 2

    def test_declared_buffer_object_honors_an_explicit_name(self):
        """An explicit name overrides the variable name it is bound to."""
        slot = pl.Buffer("l0c_ping")

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, slot, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                return pl.store(t0, [0, 0], out)

        assert _base_names(passes.init_mem_ref()(Before))["t0"] == "l0c_ping"

    def test_binding_is_an_explicit_flag_not_a_zero_size(self):
        """A zero-sized ordinary MemRef is a compiler allocation, not a binding.

        The binding is carried by ``MemRef.is_user_buffer_``. Inferring it from
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
        assert memref.size_ == 0 and not memref.is_user_buffer_
        assert "pinned=True" not in passes.init_mem_ref()(parsed).as_python()

    def test_unresolved_binding_prints_as_buffer_and_round_trips(self):
        """The printed form of a binding is the form the author wrote.

        A binding carries no size or address to print — InitMemRef derives both —
        so printing it as ``pl.MemRef(...)`` would have to invent them and would
        lose the distinction on reparse.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.Buffer("scratch"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64]
                )
                return pl.store(t0, [0, 0], out)

        dumped = Before.as_python()
        assert 'pl.Buffer("scratch")' in dumped, dumped
        ir.assert_structural_equal(Before, pl.parse_program(dumped))
        # Resolution consumes the binding: the pinned alloc carries it from here on.
        assert "pl.Buffer" not in passes.init_mem_ref()(Before).as_python()

    def test_binds_a_transpose_output(self):
        """`tile.transpose` owns its output buffer, so it may be bound.

        It inherits the input's memory *space*, but `pto.ttrans` is registered
        `not_inplace_safe()` — the permute lands in a fresh buffer. Treating
        space inheritance as buffer inheritance would refuse a binding the
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
                t0: pl.Tile[[64, 32], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 32])
                tr: pl.Tile[[32, 64], pl.FP32, pl.Buffer("trans"), pl.Mem.Vec] = pl.tile.transpose(t0, 0, 1)
                return pl.store(tr, [0, 0], out)

        after = passes.init_mem_ref()(Before)
        assert _base_names(after)["tr"] == "trans"

    def test_size_is_the_largest_bound_tile(self):
        """The author writes no byte count: the buffer is sized to hold any member."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out_big: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                out_small: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> tuple[pl.Tensor[[64, 64], pl.FP32], pl.Tensor[[32, 32], pl.FP32]]:
                big: pl.Tile[[64, 64], pl.FP32, pl.Buffer("scratch"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64]
                )
                r_big: pl.Tensor[[64, 64], pl.FP32] = pl.store(big, [0, 0], out_big)
                # `big` is dead by now, so `small` may legally take over the buffer.
                small: pl.Tile[[32, 32], pl.FP32, pl.Buffer("scratch"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [32, 32]
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
                t0: pl.Tile[[64, 64], pl.FP32, pl.Buffer("scratch"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64]
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
                t0: pl.Tile[[64, 64], pl.FP32, pl.Buffer("scratch"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64]
                )
                return pl.store(t0, [0, 0], out)

        after = passes.init_mem_ref()(Before)
        reparsed = pl.parse_program(after.as_python())
        ir.assert_structural_equal(after, reparsed)


class TestReuseControl:
    """MemoryReuse packs unbound tiles as before, and leaves user buffers alone."""

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
                t0: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                t1: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t0)
                t2: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t1)
                return pl.store(t2, [0, 0], out)

        bases = set(_base_names(_run_memory_pipeline(Before)).values())
        assert len(bases) == 1, f"expected the packer to coalesce the chain, got {bases}"

    def test_bound_tiles_are_not_packed(self, ascend_backend):
        """The same chain, bound: three buffers survive — no false dependencies."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.Buffer("in_buf"), pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                t1: pl.Tile[[64, 64], pl.FP32, pl.Buffer("mid_buf"), pl.Mem.Vec] = pl.exp(t0)
                t2: pl.Tile[[64, 64], pl.FP32, pl.Buffer("out_buf"), pl.Mem.Vec] = pl.exp(t1)
                return pl.store(t2, [0, 0], out)

        bases = _base_names(_run_memory_pipeline(Before))
        assert bases["t0"] == "in_buf"
        assert bases["t1"] == "mid_buf"
        assert bases["t2"] == "out_buf"

    def test_explicit_sharing_is_preserved(self, ascend_backend):
        """Author-chosen sharing survives: t0 and t2 on one buffer, t1 on another.

        Also pins the touching-lifetimes rule: t0's last read is the statement
        producing t1, and t2 is defined after that, so the overlap check accepts
        the pair rather than treating the shared buffer as a conflict.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.Buffer("ping"), pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                t1: pl.Tile[[64, 64], pl.FP32, pl.Buffer("pong"), pl.Mem.Vec] = pl.exp(t0)
                t2: pl.Tile[[64, 64], pl.FP32, pl.Buffer("ping"), pl.Mem.Vec] = pl.exp(t1)
                return pl.store(t2, [0, 0], out)

        bases = _base_names(_run_memory_pipeline(Before))
        assert bases["t0"] == bases["t2"] == "ping"
        assert bases["t1"] == "pong"

    def test_unbound_tiles_never_join_a_user_buffer(self, ascend_backend):
        """An unbound tile is packed with other unbound tiles, never into a user buffer."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mine: pl.Tile[[64, 64], pl.FP32, pl.Buffer("mine"), pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                free0: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(mine)
                free1: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(free0)
                return pl.store(free1, [0, 0], out)

        bases = _base_names(_run_memory_pipeline(Before))
        assert bases["mine"] == "mine"
        assert bases["free0"] != "mine"
        assert bases["free1"] != "mine"


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
                t0: pl.Tile[[64, 64], pl.FP32, pl.Buffer("ping"), pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                t1: pl.Tile[[64, 64], pl.FP32, pl.Buffer("pong"), pl.Mem.Vec] = pl.exp(t0)
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
                t0: pl.Tile[[4, 16, 64], pl.FP32, pl.Buffer("nd_buf"), pl.Mem.Vec] = pl.load(
                    a, [0, 0, 0], [4, 16, 64]
                )
                t1: pl.Tile[[4, 16, 64], pl.FP32, pl.Buffer("nd_buf"), pl.Mem.Vec] = pl.exp(t0)
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
                    acc0: pl.Tile[[m, n], pl.FP32, pl.Buffer("l0c_ping"), pl.Mem.Acc] = pl.tile.matmul(
                        q_l0, b0
                    )
                    r0: pl.Tensor[[m, 2 * n], pl.FP32] = pl.store(acc0, [0, 0], out)
                    b1: pl.Tile[[k, n], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        b_l1, 0, n, [k, n], target_memory=pl.MemorySpace.Right
                    )
                    acc1: pl.Tile[[m, n], pl.FP32, pl.Buffer("l0c_pong"), pl.Mem.Acc] = pl.tile.matmul(
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
                made: pl.Tile[[4, 16, 64], pl.FP32, pl.Buffer("made_buf"), pl.Mem.Vec] = pl.tile.create(
                    [4, 16, 64], pl.FP32
                )
                loaded: pl.Tile[[4, 16, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0, 0], [4, 16, 64])
                summed: pl.Tile[[4, 16, 64], pl.FP32, pl.Mem.Vec] = pl.add(made, loaded)
                return pl.store(summed, [0, 0, 0], out)

        after = _run_full_pipeline(Before, "InitMemRef")
        bases = set(_base_names(after).values())
        assert "made_buf" in bases, f"tile.full binding lost during flattening, got {bases}"

    def test_reparsed_dump_is_not_treated_as_user_buffers(self, ascend_backend):
        """A post-allocation dump also carries MemRefs — those are not user buffers.

        The binding is recognised by the parser's size-0 "derive me" marker, not by
        "a MemRef exists". Re-running the passes over a printed program must not
        promote its compiler allocations into pinned, un-reusable buffers.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                t1: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t0)
                return pl.store(t1, [0, 0], out)

        dumped = passes.init_mem_ref()(Before).as_python()
        assert "pinned=True" not in dumped
        reparsed = pl.parse_program(dumped)
        # Re-running InitMemRef over the dump must still produce no pinned allocs.
        assert "pinned=True" not in passes.init_mem_ref()(reparsed).as_python()

    def test_pinned_buffers_stay_on_distinct_bases(self, ascend_backend):
        """Two user buffers survive to AllocateMemoryAddr as two separate allocations.

        Distinct base Ptrs are what "separate buffers" means at this level — the
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
                t0: pl.Tile[[64, 64], pl.FP32, pl.Buffer("ping"), pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                t1: pl.Tile[[64, 64], pl.FP32, pl.Buffer("pong"), pl.Mem.Vec] = pl.exp(t0)
                return pl.store(t1, [0, 0], out)

        after = _run_full_pipeline(Before, "AllocateMemoryAddr")
        memrefs = _tile_memrefs(after)
        ping, pong = memrefs["t0__ssa_v0"], memrefs["t1__ssa_v0"]
        assert ping.base_.name_hint != pong.base_.name_hint
        assert ping.size_ == pong.size_ == 16384
        # Two allocations reach the allocator, neither folded into the other.
        assert len(_alloc_lines(after)) == 2


class TestRejects:
    """Bindings the compiler must refuse, each with a message that says why."""

    def test_rejects_binding_a_pipelined_tile(self, ascend_backend):
        """One named buffer cannot back two in-flight stages of a pl.pipeline.

        `stage=2` clones the body so iteration i and i+1 overlap; both clones name
        the same buffer, so the tile is co-live with itself. Naming a buffer and
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
                    t: pl.Tile[[64, 64], pl.FP32, pl.Buffer("staged"), pl.Mem.Vec] = pl.load(
                        a, [i, 0], [64, 64]
                    )
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(e, [i, 0], acc)
                    y = pl.yield_(nxt)
                return y

        with pytest.raises(ValueError, match="live at the same time"):
            _run_full_pipeline(Before, "MemoryReuse")

    def test_rejects_overlapping_lifetimes(self, ascend_backend):
        """Two co-live tiles on one buffer would corrupt data, not reuse it."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.Buffer("ping"), pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                t1: pl.Tile[[64, 64], pl.FP32, pl.Buffer("pong"), pl.Mem.Vec] = pl.exp(t0)
                # Overwrites `ping` while t0 is still needed by the add below.
                t2: pl.Tile[[64, 64], pl.FP32, pl.Buffer("ping"), pl.Mem.Vec] = pl.exp(t1)
                t3: pl.Tile[[64, 64], pl.FP32, pl.Buffer("pong2"), pl.Mem.Vec] = pl.add(t0, t2)
                return pl.store(t3, [0, 0], out)

        with pytest.raises(ValueError, match="live at the same time"):
            _run_memory_pipeline(Before)

    def test_rejects_mixed_memory_space(self):
        """One buffer lives in one memory space; bound tiles must agree on it."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP16],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP16]],
            ) -> pl.Tensor[[64, 64], pl.FP16]:
                vec: pl.Tile[[64, 64], pl.FP16, pl.Buffer("shared"), pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64]
                )
                mat: pl.Tile[[64, 64], pl.FP16, pl.Buffer("shared"), pl.Mem.Mat] = pl.tile.move(
                    vec, target_memory=pl.Mem.Mat
                )
                back: pl.Tile[[64, 64], pl.FP16, pl.Mem.Vec] = pl.tile.move(mat, target_memory=pl.Mem.Vec)
                return pl.store(back, [0, 0], out)

        with pytest.raises(ValueError, match="same memory space"):
            passes.init_mem_ref()(Before)

    def test_rejects_binding_a_view_output(self):
        """A view already IS its source's buffer; it cannot be bound elsewhere."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[32, 128], pl.FP32]],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                t0: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
                view: pl.Tile[[32, 128], pl.FP32, pl.Buffer("elsewhere"), pl.Mem.Vec] = pl.reshape(
                    t0, [32, 128]
                )
                return pl.store(view, [0, 0], out)

        with pytest.raises(ValueError, match="reuses its source tile's buffer"):
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
        t0: pl.Tile[[64, 64], pl.FP32, pl.Buffer("scratch")] = pl.load(a, [0, 0], [64, 64])
        return pl.store(t0, [0, 0], out)
"""
        with pytest.raises(Exception, match="explicit memory space"):
            pl.parse_program(source)

    def test_rejects_non_literal_buffer_name(self):
        """The name identifies the buffer at parse time, so it must be a literal."""
        source = """
import pypto.language as pl

NAME = "scratch"


@pl.program
class Bad:
    @pl.function
    def main(self, a: pl.Tensor[[64, 64], pl.FP32],
             out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        t0: pl.Tile[[64, 64], pl.FP32, pl.Buffer(NAME), pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
        return pl.store(t0, [0, 0], out)
"""
        with pytest.raises(Exception, match="string literal"):
            pl.parse_program(source)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
