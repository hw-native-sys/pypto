# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for parsing the ``for aiv_id in pl.split_aiv(2, mode=...):`` loop form.

The loop form now builds a first-class ``SplitAivScopeStmt`` region node carrying
the requested ``SplitMode`` on ``split`` (and a fixed ``count == 2``). The region
body begins with ``aiv_id = pl.tile.get_subblock_idx()`` (the AIV lane index). A
bare top-level region is wrapped in a synthesized ``InCoreScopeStmt`` so it stays
eligible for ``OutlineIncoreScopes``; the region may also nest inside an existing
InCore scope (directly or through an intervening ``pl.range`` / ``if``).
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics.exceptions import InvalidOperationError, ParserSyntaxError


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


def _find_call(node, op_name):
    """Return the single ``ir.Call`` to ``op_name`` reachable from ``node``."""
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


def _first_body_stmt(node):
    """Return the first statement of a scope/region ``body`` (handles SeqStmts)."""
    body = node.body
    return body.stmts[0] if isinstance(body, ir.SeqStmts) else body


class TestSplitAivBuildsNode:
    """The loop form builds a first-class ``SplitAivScopeStmt`` region node."""

    def test_builds_node_not_incore_attr(self):
        """Loop form builds a SplitAivScopeStmt (mode/count on the node), NOT an
        InCore scope carrying the legacy ``("split_aiv", True)`` attr.
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

        # Exactly ONE SplitAivScopeStmt node; mode and count ride on the node.
        region = _unique_descendant(main_func.body, ir.SplitAivScopeStmt)
        assert region.split == ir.SplitMode.UP_DOWN
        assert region.count == 2
        assert _count_descendants(main_func.body, ir.SpmdScopeStmt) == 0

        # No InCore scope carries the legacy ("split_aiv", True) attr.
        incore = _unique_descendant(main_func.body, ir.InCoreScopeStmt)
        assert "split_aiv" not in incore.attrs

        # First region-body statement binds aiv_id to tile.get_subblock_idx().
        first_stmt = _first_body_stmt(region)
        assert isinstance(first_stmt, ir.AssignStmt)
        assert isinstance(first_stmt.value, ir.Call)
        assert first_stmt.value.op.name == "tile.get_subblock_idx"
        assert first_stmt.var.name_hint == "aiv_id"

    def test_top_level_wrapped_in_incore(self):
        """A bare top-level split_aiv region is wrapped in a synthesized InCore
        scope (InCoreScopeStmt{ body: SplitAivScopeStmt{...} }).
        """

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

        incore = _unique_descendant(main_func.body, ir.InCoreScopeStmt)
        region = _unique_descendant(incore.body, ir.SplitAivScopeStmt)
        assert region.split == ir.SplitMode.LEFT_RIGHT
        # The synthesized InCore wrapper itself carries no split / split_aiv attr.
        assert incore.split is None
        assert "split_aiv" not in incore.attrs

    def test_builds_node_none_mode(self):
        """``mode=pl.SplitMode.NONE`` builds a task-parallel region: split is
        NONE on the node, count stays 2, and aiv_id is still bound."""

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    offset = aiv_id * 256
                    t: pl.Tile[[256, 128], pl.FP32] = pl.load(a, [offset, 0], [256, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(TestProgram.functions.values())[0]
        region = _unique_descendant(main_func.body, ir.SplitAivScopeStmt)
        assert region.split == ir.SplitMode.NONE
        assert region.count == 2

        # aiv_id still binds tile.get_subblock_idx() even with no halving.
        first_stmt = _first_body_stmt(region)
        assert isinstance(first_stmt, ir.AssignStmt)
        assert isinstance(first_stmt.value, ir.Call)
        assert first_stmt.value.op.name == "tile.get_subblock_idx"
        assert first_stmt.var.name_hint == "aiv_id"

    def test_none_mode_print_roundtrips(self):
        """A NONE region prints as ``pl.split_aiv(2, mode=pl.SplitMode.NONE)``."""

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    offset = aiv_id * 256
                    t: pl.Tile[[256, 128], pl.FP32] = pl.load(a, [offset, 0], [256, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        txt = ir.python_print(TestProgram)
        assert "pl.split_aiv(2, mode=pl.SplitMode.NONE)" in txt

    def test_nested_in_loop_builds_node(self):
        """A split_aiv nested inside a pl.range loop (within a CORE_GROUP scope)
        builds a SplitAivScopeStmt region inside the ForStmt — the region is not
        rejected and does not open a fresh InCore.
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
                    for i in pl.range(2):
                        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                            qk_h = pl.aiv_shard(qk)
                            sc = pl.mul(qk_h, 2.0)
                            offset = aiv_id * 32
                            out = pl.store(sc, [offset, 0], out)
                return out

        main_func = list(TestProgram.functions.values())[0]

        # Exactly ONE InCore scope (the CORE_GROUP one) — the region did NOT open
        # a nested InCore; instead a SplitAivScopeStmt lives inside the ForStmt.
        assert _count_descendants(main_func.body, ir.InCoreScopeStmt) == 1
        for_stmt = _unique_descendant(main_func.body, ir.ForStmt)
        region = _unique_descendant(for_stmt.body, ir.SplitAivScopeStmt)
        assert region.split == ir.SplitMode.UP_DOWN

    def test_nested_in_auto_chunk_core_group_nests_in_place(self):
        """A split_aiv inside an auto-chunk CORE_GROUP scope
        (``pl.at(level=CORE_GROUP, optimizations=[pl.auto_chunk])`` -> AutoInCore)
        nests in place: the region lives directly inside the AutoInCoreScopeStmt
        body, with NO synthesized nested InCoreScopeStmt wrapper.
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
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])
                        tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [0, 0], [64, 64])
                        out = pl.store(pl.add(tile_a, tile_b), [0, 0], out)
                return out

        main_func = list(TestProgram.functions.values())[0]

        # The AutoInCore scope is the existing CORE_GROUP scope; the region nested
        # in place — NO redundant InCore wrapper was synthesized inside it.
        assert _count_descendants(main_func.body, ir.AutoInCoreScopeStmt) == 1
        assert _count_descendants(main_func.body, ir.InCoreScopeStmt) == 0
        auto_incore = _unique_descendant(main_func.body, ir.AutoInCoreScopeStmt)
        region = _unique_descendant(auto_incore.body, ir.SplitAivScopeStmt)
        assert region.split == ir.SplitMode.UP_DOWN

    def test_rejects_n_not_2(self):
        """n must be hardware-fixed at 2."""
        with pytest.raises(ParserSyntaxError, match="requires n == 2"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for aiv_id in pl.split_aiv(4, mode=pl.SplitMode.UP_DOWN):
                    _ = aiv_id
                return a

    def test_requires_mode(self):
        """mode= is required — no silent default."""
        with pytest.raises(ParserSyntaxError, match="requires mode="):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for aiv_id in pl.split_aiv(2):  # type: ignore[call-arg]
                    _ = aiv_id
                return a

    def test_split_aiv_rejects_nested(self):
        """A pl.split_aiv loop cannot be nested inside another split_aiv body."""
        with pytest.raises(ParserSyntaxError, match="nested.*pl.split_aiv"):

            @pl.function(type=pl.FunctionType.Orchestration)
            def bad(
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    for aiv2 in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        offset = (aiv_id + aiv2) * 128
                        t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                        out = pl.store(t, [offset, 0], out)
                return out

    def test_split_aiv_rejects_break(self):
        """break inside a split_aiv body is rejected — the body is a scope, not a loop."""
        with pytest.raises(InvalidOperationError, match="'break' not supported inside"):

            @pl.function(type=pl.FunctionType.Orchestration)
            def bad(
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    offset = aiv_id * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                    break
                return out

    def test_split_aiv_rejects_continue(self):
        """continue inside a split_aiv body is rejected — the body is a scope, not a loop."""
        with pytest.raises(InvalidOperationError, match="'continue' not supported inside"):

            @pl.function(type=pl.FunctionType.Orchestration)
            def bad(
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    offset = aiv_id * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                    continue
                return out

    def test_split_aiv_bare_form_merges_dump_tag(self):
        """Forward-sticky pl.dump_tag tensors are merged onto the bare split_aiv InCore scope."""

        @pl.program
        class TestProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                pl.dump_tag(a)
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    offset = aiv_id * 128
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                    out = pl.store(t, [offset, 0], out)
                return out

        main_func = list(TestProgram.functions.values())[0]
        incore = _unique_descendant(main_func.body, ir.InCoreScopeStmt)
        assert "dump_vars" in incore.attrs
        assert {v.name_hint for v in incore.attrs["dump_vars"]} == {"a"}


class TestSplitAivValidation:
    """Orthogonal validation diagnostics (unchanged by the node migration)."""

    def test_api_rejects_non_two_count_directly(self):
        """pl.split_aiv(n != 2) called directly (outside the parser) raises."""
        with pytest.raises(ValueError, match="must be the integer 2"):
            pl.split_aiv(4, mode=pl.SplitMode.UP_DOWN)

    def test_rejects_loop_kwarg(self):
        """A loop-carried kwarg (init_values=) makes no sense for the split body."""
        with pytest.raises(ParserSyntaxError, match=r"does not accept 'init_values='"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN, init_values=(0,)):  # type: ignore[call-arg]
                    _ = aiv_id
                return a

    def test_rejects_co_present_optimizations(self):
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

    def test_rejects_tuple_target(self):
        """A tuple target on for-split_aiv is rejected (single loop var only)."""
        with pytest.raises(ParserSyntaxError, match="single loop variable"):

            @pl.function
            def bad(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i, j in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):  # type: ignore[misc]
                    _ = i + j
                return a


class TestAivShardInheritsSplit:
    """``pl.aiv_shard`` / ``pl.aic_gather`` inherit the region's split mode."""

    def test_aiv_shard_inherits_up_down_split(self):
        """pl.aiv_shard(tile) inside an UP_DOWN region carries split == 1."""

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
        region = _unique_descendant(main_func.body, ir.SplitAivScopeStmt)
        shard = _find_call(region.body, "tile.aiv_shard")
        assert shard.kwargs["split"] == 1

    def test_aiv_shard_inherits_left_right_split(self):
        """pl.aiv_shard(tile) inside a LEFT_RIGHT region carries split == 2."""

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
        region = _unique_descendant(main_func.body, ir.SplitAivScopeStmt)
        shard = _find_call(region.body, "tile.aiv_shard")
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
        """pl.aiv_shard outside any split_aiv region has no mode to inherit -> error."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
