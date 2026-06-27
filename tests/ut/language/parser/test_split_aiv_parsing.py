# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for parsing the ``for aiv_id in pl.split_aiv(2, mode=...):`` loop form.

The loop form opens exactly ONE bare ``InCoreScopeStmt`` marking an explicit
AIV-split body. The scope carries the requested ``SplitMode`` on ``split`` plus a
``("split_aiv", True)`` attr, and the loop variable is bound to
``pl.tile.get_subblock_idx()`` (the AIV lane / sub-core index) as the first body
statement. Unlike ``pl.spmd``, it does NOT wrap the InCore body in a Spmd scope.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics.exceptions import ParserSyntaxError


class TestSplitAivForLoopParsing:
    """Parsing of ``for aiv_id in pl.split_aiv(2, mode=...):``."""

    @staticmethod
    def _unique_descendant(node, cls):
        """Return the single descendant of ``node`` that is an instance of ``cls``."""
        found = []

        def walk(n):
            if isinstance(n, cls):
                found.append(n)
            if isinstance(n, ir.SeqStmts):
                for s in n.stmts:
                    walk(s)
            elif hasattr(n, "body") and n.body is not None:
                walk(n.body)

        walk(node)
        assert len(found) == 1, f"expected exactly one {cls.__name__}, got {len(found)}"
        return found[0]

    @staticmethod
    def _count_descendants(node, cls):
        count = 0

        def walk(n):
            nonlocal count
            if isinstance(n, cls):
                count += 1
            if isinstance(n, ir.SeqStmts):
                for s in n.stmts:
                    walk(s)
            elif hasattr(n, "body") and n.body is not None:
                walk(n.body)

        walk(node)
        return count

    def test_split_aiv_up_down_builds_bare_incore_scope(self):
        """Loop form emits a single InCoreScopeStmt (no Spmd wrapper) whose split
        mode is UP_DOWN, which carries the ("split_aiv", True) attr, and whose
        first body statement binds the loop var to pl.tile.get_subblock_idx().
        """

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                b: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    offset = aiv_id * 128
                    tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [offset, 0], [128, 128])
                    out = pl.store(pl.add(tile_a, tile_b), [offset, 0], out)
                return out

        main_func = list(TestProgram.functions.values())[0]

        # Exactly ONE InCore scope and NO Spmd wrapper (bare InCore, unlike pl.spmd).
        assert self._count_descendants(main_func.body, ir.SpmdScopeStmt) == 0
        incore = self._unique_descendant(main_func.body, ir.InCoreScopeStmt)

        # Split mode threaded onto ScopeStmt::split_.
        assert incore.split == ir.SplitMode.UP_DOWN
        # Explicit-split marker attr.
        assert "split_aiv" in incore.attrs
        assert incore.attrs["split_aiv"] is True

        # First body statement binds the loop var to tile.get_subblock_idx().
        body = incore.body
        first_stmt = body.stmts[0] if isinstance(body, ir.SeqStmts) else body
        assert isinstance(first_stmt, ir.AssignStmt)
        call = first_stmt.value
        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.get_subblock_idx"
        assert first_stmt.var.name_hint == "aiv_id"

    def test_split_aiv_left_right_mode(self):
        """LEFT_RIGHT mode threads through to ScopeStmt::split_."""

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                    offset = aiv_id * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(TestProgram.functions.values())[0]
        incore = self._unique_descendant(main_func.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.LEFT_RIGHT
        assert incore.attrs["split_aiv"] is True

    def test_split_aiv_rejects_non_two_count(self):
        """n must be hardware-fixed at 2."""
        with pytest.raises(ParserSyntaxError, match="requires n == 2"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for aiv_id in pl.split_aiv(4, mode=pl.SplitMode.UP_DOWN):
                    _ = aiv_id
                return a

    def test_split_aiv_requires_mode(self):
        """mode= is required — no silent default."""
        with pytest.raises(ParserSyntaxError, match="requires mode="):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for aiv_id in pl.split_aiv(2):  # type: ignore[call-arg]
                    _ = aiv_id
                return a

    def test_split_aiv_rejects_loop_kwarg(self):
        """A loop-carried kwarg (init_values=) makes no sense for the split body."""
        with pytest.raises(ParserSyntaxError, match=r"does not accept 'init_values='"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN, init_values=(0,)):  # type: ignore[call-arg]
                    _ = aiv_id
                return a

    def test_split_aiv_rejects_co_present_optimizations(self):
        """pl.split_aiv() IS the split declaration; optimizations=[pl.split(...)] conflicts."""
        with pytest.raises(ParserSyntaxError, match=r"does not accept 'optimizations='"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for aiv_id in pl.split_aiv(  # type: ignore[call-arg]
                    2,
                    mode=pl.SplitMode.UP_DOWN,
                    optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
                ):
                    _ = aiv_id
                return a

    def test_split_aiv_rejects_tuple_target(self):
        """A tuple target on for-split_aiv is rejected (single loop var only)."""
        with pytest.raises(ParserSyntaxError, match="single loop variable"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i, j in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):  # type: ignore[misc]
                    _ = i + j
                return a

    @staticmethod
    def _find_call(node, op_name):
        """Return the single ``ir.Call`` to ``op_name`` reachable from ``node``.

        Descends through ``SeqStmts``, scope/loop bodies, and ``AssignStmt``
        right-hand sides so an op bound by ``x = pl.<op>(...)`` is discovered.
        """
        found = []

        def walk(n):
            if isinstance(n, ir.Call) and n.op.name == op_name:
                found.append(n)
            if isinstance(n, ir.SeqStmts):
                for s in n.stmts:
                    walk(s)
                return
            if isinstance(n, ir.AssignStmt):
                walk(n.value)
            body = getattr(n, "body", None)
            if body is not None:
                walk(body)

        walk(node)
        assert len(found) == 1, f"expected exactly one Call to {op_name}, got {len(found)}"
        return found[0]

    def test_aiv_shard_inherits_up_down_split(self):
        """pl.aiv_shard(tile) inside an UP_DOWN split_aiv scope carries split == 1."""

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
            ) -> pl.Tensor[[256, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    offset = aiv_id * 128
                    qk: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    x = pl.aiv_shard(qk)
                    out = pl.store(x, [offset, 0], out)
                return out

        main_func = list(TestProgram.functions.values())[0]
        incore = self._unique_descendant(main_func.body, ir.InCoreScopeStmt)
        shard = self._find_call(incore.body, "tile.aiv_shard")
        assert shard.kwargs["split"] == 1

    def test_aiv_shard_inherits_left_right_split(self):
        """pl.aiv_shard(tile) inside a LEFT_RIGHT split_aiv scope carries split == 2."""

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
            ) -> pl.Tensor[[256, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                    offset = aiv_id * 128
                    qk: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    x = pl.aiv_shard(qk)
                    out = pl.store(x, [offset, 0], out)
                return out

        main_func = list(TestProgram.functions.values())[0]
        incore = self._unique_descendant(main_func.body, ir.InCoreScopeStmt)
        shard = self._find_call(incore.body, "tile.aiv_shard")
        assert shard.kwargs["split"] == 2

    def test_aiv_shard_rejects_explicit_mode(self):
        """Passing mode= is rejected — the mode is inherited from pl.split_aiv."""
        with pytest.raises(ParserSyntaxError, match="does not take a mode"):

            @pl.function(type=pl.FunctionType.Orchestration)
            def bad(
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
            ) -> pl.Tensor[[256, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    offset = aiv_id * 128
                    qk: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    x = pl.aiv_shard(qk, mode=pl.SplitMode.UP_DOWN)  # type: ignore[call-arg]
                    out = pl.store(x, [offset, 0], out)
                return out

    def test_aiv_shard_outside_split_aiv_scope_rejected(self):
        """pl.aiv_shard outside any split_aiv scope has no mode to inherit -> error."""
        with pytest.raises(ParserSyntaxError, match="pl.split_aiv"):

            @pl.function(type=pl.FunctionType.Orchestration)
            def bad(
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
            ) -> pl.Tensor[[256, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    qk: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                    x = pl.aiv_shard(qk)
                    out = pl.store(x, [0, 0], out)
                return out


class TestSplitAivFlattenInsideCoreGroup:
    """``for aiv_id in pl.split_aiv(...)`` nested inside a ``pl.at(CORE_GROUP)``.

    A split_aiv loop already inside an InCore (CORE_GROUP) scope must NOT open a
    nested InCore sub-scope (which OutlineIncoreScopes would outline as a
    separate tile-I/O sub-function, breaking ConvertTensorToTileOps /
    InferTileMemorySpace). Instead, the loop's split mode + ``("split_aiv",
    True)`` attr are stamped onto the ENCLOSING CORE_GROUP InCore scope, and the
    body is emitted inline — one fused-mixed InCore function (cube + vector).
    """

    @staticmethod
    def _count_descendants(node, cls):
        count = 0

        def walk(n):
            nonlocal count
            if isinstance(n, cls):
                count += 1
            if isinstance(n, ir.SeqStmts):
                for s in n.stmts:
                    walk(s)
            elif hasattr(n, "body") and n.body is not None:
                walk(n.body)

        walk(node)
        return count

    @staticmethod
    def _unique_descendant(node, cls):
        found = []

        def walk(n):
            if isinstance(n, cls):
                found.append(n)
            if isinstance(n, ir.SeqStmts):
                for s in n.stmts:
                    walk(s)
            elif hasattr(n, "body") and n.body is not None:
                walk(n.body)

        walk(node)
        assert len(found) == 1, f"expected exactly one {cls.__name__}, got {len(found)}"
        return found[0]

    @staticmethod
    def _find_subblock_idx_bind(node):
        """Return the AssignStmt binding a var to ``tile.get_subblock_idx()``."""
        found = []

        def walk(n):
            if (
                isinstance(n, ir.AssignStmt)
                and isinstance(n.value, ir.Call)
                and isinstance(n.value.op, ir.Op)
                and n.value.op.name == "tile.get_subblock_idx"
            ):
                found.append(n)
            if isinstance(n, ir.SeqStmts):
                for s in n.stmts:
                    walk(s)
            else:
                body = getattr(n, "body", None)
                if body is not None:
                    walk(body)

        walk(node)
        assert len(found) == 1, f"expected exactly one subblock_idx bind, got {len(found)}"
        return found[0]

    def test_flattens_onto_enclosing_core_group_scope(self):
        """Nested split_aiv stamps split + split_aiv onto the enclosing CORE_GROUP
        InCore scope (no extra nested InCore), and binds aiv_id inline.
        """

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk"):
                    a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                    b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                    a_left = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
                    b_right = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
                    qk = pl.matmul(a_left, b_right)
                    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        qk_h = pl.aiv_shard(qk)
                        sc = pl.mul(qk_h, 2.0)
                        offset = aiv_id * 32
                        out = pl.store(sc, [offset, 0], out)
                return out

        main_func = list(TestProgram.functions.values())[0]

        # Exactly ONE InCore scope — the enclosing CORE_GROUP one. The split_aiv
        # loop did NOT open a nested InCore sub-scope.
        assert self._count_descendants(main_func.body, ir.InCoreScopeStmt) == 1
        assert self._count_descendants(main_func.body, ir.SpmdScopeStmt) == 0
        incore = self._unique_descendant(main_func.body, ir.InCoreScopeStmt)

        # Split mode + explicit-split marker stamped onto the enclosing scope.
        assert incore.split == ir.SplitMode.UP_DOWN
        assert incore.attrs["split_aiv"] is True

        # aiv_id is bound inline to tile.get_subblock_idx() within the same scope.
        bind = self._find_subblock_idx_bind(incore.body)
        assert bind.var.name_hint == "aiv_id"

    def test_left_right_mode_stamps_onto_enclosing_scope(self):
        """LEFT_RIGHT mode threads onto the enclosing CORE_GROUP InCore scope."""

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk"):
                    a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                    b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                    a_left = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
                    b_right = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
                    qk = pl.matmul(a_left, b_right)
                    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                        qk_h = pl.aiv_shard(qk)
                        sc = pl.mul(qk_h, 2.0)
                        offset = aiv_id * 32
                        out = pl.store(sc, [0, offset], out)
                return out

        main_func = list(TestProgram.functions.values())[0]
        assert self._count_descendants(main_func.body, ir.InCoreScopeStmt) == 1
        incore = self._unique_descendant(main_func.body, ir.InCoreScopeStmt)
        assert incore.split == ir.SplitMode.LEFT_RIGHT
        assert incore.attrs["split_aiv"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
