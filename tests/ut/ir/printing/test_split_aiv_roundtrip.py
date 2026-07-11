# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Print -> parse round-trip tests for the first-class ``SplitAivScopeStmt`` node.

The printer emits ``for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.X):`` and the
parser routes it back to a ``SplitAivScopeStmt`` region. These tests assert the
fixpoint holds for the bare (InCore-wrapped) form and for regions nested inside
``pl.range`` / ``pl.pipeline`` / ``if``; that the redundant per-op ``split=``
kwarg is suppressed inside a region yet reparses; and that a DCE-stripped
``aiv_id`` binding still round-trips via the synthesized-name fallback.
"""

import pypto.language as pl
import pytest
from pypto import ir


def _main_body(prog):
    return list(prog.functions.values())[0].body


def _count_descendants(node, cls):
    count = 0

    def walk(n):
        nonlocal count
        if isinstance(n, cls):
            count += 1
        if isinstance(n, ir.SeqStmts):
            for s in n.stmts:
                walk(s)
        elif isinstance(n, ir.IfStmt):
            walk(n.then_body)
            if n.else_body is not None:
                walk(n.else_body)
        elif hasattr(n, "body") and n.body is not None:
            walk(n.body)

    walk(node)
    return count


def _bare_top_level_program():
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

    return TestProgram


def _shard_program():
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

    return TestProgram


def test_print_basic():
    """Printer emits the ``for aiv_id in pl.split_aiv(2, mode=...)`` loop form."""
    text = ir.python_print(_bare_top_level_program())
    assert "for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):" in text, text


def test_roundtrip_top_level():
    """Bare top-level form round-trips; the region is wrapped in an InCore scope."""
    prog = _bare_top_level_program()
    # The parsed program wraps the region in a synthesized InCore scope.
    assert _count_descendants(_main_body(prog), ir.InCoreScopeStmt) == 1
    assert _count_descendants(_main_body(prog), ir.SplitAivScopeStmt) == 1

    text = ir.python_print(prog)
    reparsed = pl.parse_program(text)
    ir.assert_structural_equal(prog, reparsed)


def test_roundtrip_nested_in_range():
    """A region nested inside pl.range (within CORE_GROUP) round-trips."""

    @pl.program
    class Prog:
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

    assert _count_descendants(_main_body(Prog), ir.SplitAivScopeStmt) == 1
    text = ir.python_print(Prog)
    reparsed = pl.parse_program(text)
    ir.assert_structural_equal(Prog, reparsed)


def test_roundtrip_nested_in_pipeline():
    """A region nested inside pl.pipeline (within CORE_GROUP) round-trips."""

    @pl.program
    class Prog:
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
                for i in pl.pipeline(2, stage=2):
                    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                        qk_h = pl.aiv_shard(qk)
                        sc = pl.mul(qk_h, 2.0)
                        offset = aiv_id * 32
                        out = pl.store(sc, [0, offset], out)
            return out

    assert _count_descendants(_main_body(Prog), ir.SplitAivScopeStmt) == 1
    text = ir.python_print(Prog)
    assert "pl.pipeline(2, stage=2)" in text, text
    reparsed = pl.parse_program(text)
    ir.assert_structural_equal(Prog, reparsed)


def test_roundtrip_nested_in_if():
    """A region nested inside an ``if`` (within CORE_GROUP) round-trips."""

    @pl.program
    class Prog:
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
                    if i < 1:
                        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                            qk_h = pl.aiv_shard(qk)
                            sc = pl.mul(qk_h, 2.0)
                            offset = aiv_id * 32
                            out = pl.store(sc, [offset, 0], out)
            return out

    assert _count_descendants(_main_body(Prog), ir.SplitAivScopeStmt) == 1
    text = ir.python_print(Prog)
    reparsed = pl.parse_program(text)
    ir.assert_structural_equal(Prog, reparsed)


def _find_call(node, op_name):
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


def test_body_split_kwarg_suppressed_yet_reparses():
    """Inside a region the per-op ``split=`` kwarg on aiv_shard is suppressed in
    the printed text, yet the reparsed program re-stamps the identical
    ``split=int(mode)`` attr (UP_DOWN -> 1).
    """
    prog = _shard_program()
    # The original parse stamps split=1 on the aiv_shard.
    assert _find_call(_main_body(prog), "tile.aiv_shard").kwargs["split"] == 1

    text = ir.python_print(prog)
    # The redundant per-op split= kwarg is not printed inside the region.
    assert "pl.tile.aiv_shard(qk)" in text, text
    assert "split=" not in text, text

    reparsed = pl.parse_program(text)
    # The reparsed aiv_shard re-inherits the same split int from the region mode.
    assert _find_call(_main_body(reparsed), "tile.aiv_shard").kwargs["split"] == 1
    ir.assert_structural_equal(prog, reparsed)


def _tensor_shard_program():
    """A region whose input is a high-level Tensor (a ``pl.matmul`` result on GM
    param tensors) fed to ``pl.aiv_shard`` — the ``@pl.jit`` / ``pl.spmd``
    author-facing form that emits ``tensor.aiv_shard`` (converted to the tile op
    at ConvertTensorToTileOps)."""

    @pl.program
    class TestProgram:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.FP32],
            b: pl.Tensor[[128, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):  # noqa: B007
                s = pl.matmul(a, b)
                h = pl.aiv_shard(s)  # noqa: F841
            return out

    return TestProgram


def test_tensor_form_split_kwarg_suppressed_yet_reparses():
    """Inside a region the per-op ``split=`` kwarg on the high-level
    ``tensor.aiv_shard`` is suppressed in the printed text (same as the tile
    form), yet the reparsed program re-stamps the identical ``split=int(mode)``
    attr (UP_DOWN -> 1) and round-trips structurally.
    """
    prog = _tensor_shard_program()
    # The original parse emits the tensor form with split=1 stamped from the mode.
    assert _find_call(_main_body(prog), "tensor.aiv_shard").kwargs["split"] == 1

    text = ir.python_print(prog)
    # The redundant per-op split= kwarg is not printed inside the region.
    assert "pl.tensor.aiv_shard(s)" in text, text
    assert "split=" not in text, text

    reparsed = pl.parse_program(text)
    # The reparsed tensor.aiv_shard re-inherits the same split int from the mode.
    assert _find_call(_main_body(reparsed), "tensor.aiv_shard").kwargs["split"] == 1
    ir.assert_structural_equal(prog, reparsed)


class _DropAivIdBinding(ir.IRMutator):
    """Strip the leading ``aiv_id = tile.get_subblock_idx()`` binding from every
    SplitAivScopeStmt region (simulating DCE of the unused index binding)."""

    def visit_split_aiv_scope_stmt(self, op):
        rewritten = super().visit_split_aiv_scope_stmt(op)
        assert isinstance(rewritten, ir.SplitAivScopeStmt)
        op = rewritten
        body = op.body
        if isinstance(body, ir.SeqStmts) and len(body.stmts) > 1:
            first = body.stmts[0]
            if (
                isinstance(first, ir.AssignStmt)
                and isinstance(first.value, ir.Call)
                and isinstance(first.value.op, ir.Op)
                and first.value.op.name == "tile.get_subblock_idx"
            ):
                new_body = ir.SeqStmts(list(body.stmts[1:]), body.span)
                return ir.SplitAivScopeStmt(op.split, op.count, op.name_hint, body=new_body, span=op.span)
        return op

    def run(self, program):
        return self.visit_program(program)


def _no_aiv_use_program():
    """A region whose body never references ``aiv_id`` — so the binding is
    DCE-eligible without leaving dangling uses."""

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
                tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
                out = pl.store(pl.add(tile_a, tile_b), [0, 0], out)
            return out

    return TestProgram


def test_roundtrip_after_aiv_id_dce():
    """When the ``aiv_id`` binding is stripped (DCE), the printer synthesizes the
    name and prints the full body; reparsing re-inserts the binding, recovering
    the original (binding-present) program.
    """
    prog = _no_aiv_use_program()
    dced = _DropAivIdBinding().run(prog)

    text = ir.python_print(dced)
    # Fallback still emits the loop header with the synthesized aiv_id name and
    # does not print a get_subblock_idx binding (it was stripped).
    assert "for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):" in text, text
    assert "get_subblock_idx" not in text, text

    reparsed = pl.parse_program(text)
    # The parser re-inserts the aiv_id = get_subblock_idx() binding, so the
    # reparsed program matches the original binding-present form.
    ir.assert_structural_equal(prog, reparsed)


def test_roundtrip_incore_function_body():
    """A region directly in an ``InCore`` function body round-trips as-is.

    This is the shape post-outline IR prints as: OutlineIncoreScopes consumes the
    ``InCoreScopeStmt`` and leaves the region directly in an ``InCore`` function
    body. The parser must emit the region in place here — synthesizing the
    bare-top-level ``InCoreScopeStmt`` wrapper would add a scope the original does
    not have (there is nothing left for OutlineIncoreScopes to outline), and every
    pass over such IR would then fail the roundtrip verifier.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def core(
            self,
            a: pl.Tensor[[256, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                offset = aiv_id * 128
                tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [offset, 0], [128, 128])
                out = pl.store(pl.add(tile_a, tile_a), [offset, 0], out)
            return out

    body = Prog.get_function("core").body
    # The region sits directly in the InCore body — no synthesized wrapper scope.
    assert _count_descendants(body, ir.InCoreScopeStmt) == 0
    assert _count_descendants(body, ir.SplitAivScopeStmt) == 1

    text = ir.python_print(Prog)
    reparsed = pl.parse_program(text)
    ir.assert_structural_equal(Prog, reparsed)


def test_roundtrip_split_none_attr_survives():
    """An explicit ``attrs={"split": pl.SplitMode.NONE}`` survives print -> parse.

    The printer emits the attr, so the parser must store it. Dropping NONE made
    print->parse lossy for exactly the functions a task-parallel ``pl.split_aiv``
    region produces (OutlineIncoreScopes derives a representative function-level
    ``split`` mode of NONE from a single NONE region).

    Storing it is behaviour-preserving: ``Function::GetSplitMode()`` maps
    ``split == 0`` to nullopt and is the sole reader of the attr, so a stored NONE
    and an absent ``split`` are indistinguishable to every consumer.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.NONE})
        def core(
            self,
            a: pl.Tensor[[128, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
            out = pl.store(pl.add(tile_a, tile_a), [0, 0], out)
            return out

    assert dict(Prog.get_function("core").attrs)["split"] == ir.SplitMode.NONE.value

    text = ir.python_print(Prog)
    assert 'attrs={"split": pl.SplitMode.NONE}' in text, text
    reparsed = pl.parse_program(text)
    ir.assert_structural_equal(Prog, reparsed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
