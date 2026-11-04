# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Hierarchical storage versions, safe reuse and conservative fallback."""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.language.parser.text_parser import parse

from .test_lower_pipeline_to_slots import _plan_memory, _walk_stmts

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
    def kernel(self, x: pl.Tensor[[128, 16], pl.FP32], y: pl.Tensor[[128, 16], pl.FP32]):
        for group in pl.pipeline(4, stage=3):
            for page in pl.pipeline(2, stage=2):
                value: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [group * 32 + page * 16, 0], [16, 16]
                )
                result: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(value, 1.0)
                _stored: pl.Tensor[[128, 16], pl.FP32] = pl.tile.store(result, [group * 32 + page * 16, 0], y)


@pl.program
class Expected:
    @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
    def kernel(self, x: pl.Tensor[[128, 16], pl.FP32], y: pl.Tensor[[128, 16], pl.FP32]):
        if pl.const(0, pl.INDEX) < pl.const(4, pl.INDEX):
            _prime: pl.Tile[[16, 16], pl.FP32, pl.MemRef("pipe_value_0", slots=6)[0], pl.Mem.Vec] = (
                pl.tile.load(
                    x,
                    [
                        pl.const(0, pl.INDEX) * pl.const(32, pl.INDEX)
                        + pl.const(0, pl.INDEX) * pl.const(16, pl.INDEX),
                        0,
                    ],
                    [16, 16],
                    attrs={"software_pipeline_slots": True},
                )
            )
        for group in pl.range(4, attrs={"software_pipeline_slots": 3}):
            slot0: pl.Scalar[pl.INDEX] = group % 3
            slot1: pl.Scalar[pl.INDEX] = slot0 + 3
            next_slot0: pl.Scalar[pl.INDEX] = (group + 1) % 3
            _load1: pl.Tile[[16, 16], pl.FP32, pl.MemRef("pipe_value_0", slots=6)[slot1], pl.Mem.Vec] = (
                pl.tile.load(
                    x,
                    [group * 32 + pl.const(1, pl.INDEX) * pl.const(16, pl.INDEX), 0],
                    [16, 16],
                    attrs={"software_pipeline_slots": True},
                )
            )
            read0: pl.Tile[[16, 16], pl.FP32, pl.MemRef("pipe_value_0", slots=6)[slot0], pl.Mem.Vec] = (
                pl.tile.create(
                    [16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec, attrs={"software_pipeline_slots": True}
                )
            )
            result0: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(read0, 1.0)
            _out0: pl.Tensor[[128, 16], pl.FP32] = pl.tile.store(
                result0, [group * 32 + pl.const(0, pl.INDEX) * pl.const(16, pl.INDEX), 0], y
            )
            if group + 1 < 4:
                _next: pl.Tile[
                    [16, 16], pl.FP32, pl.MemRef("pipe_value_0", slots=6)[next_slot0], pl.Mem.Vec
                ] = pl.tile.load(
                    x,
                    [(group + 1) * 32 + pl.const(0, pl.INDEX) * pl.const(16, pl.INDEX), 0],
                    [16, 16],
                    attrs={"software_pipeline_slots": True},
                )
            read1: pl.Tile[[16, 16], pl.FP32, pl.MemRef("pipe_value_0", slots=6)[slot1], pl.Mem.Vec] = (
                pl.tile.create(
                    [16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec, attrs={"software_pipeline_slots": True}
                )
            )
            result1: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(read1, 1.0)
            _out1: pl.Tensor[[128, 16], pl.FP32] = pl.tile.store(
                result1, [group * 32 + pl.const(1, pl.INDEX) * pl.const(16, pl.INDEX), 0], y
            )


def _run(program, enabled=True):
    with passes.PassContext([], enable_buffer_ir=False, enable_software_pipeline=enabled):
        after = passes.lower_pipeline_to_slots()(program)
    ir.assert_structural_equal(parse(ir.python_print(after)), after)
    return after


def test_flag_off_preserves_nested_program():
    ir.assert_structural_equal(_run(Before, enabled=False), Before)


def test_nested_schedule_defers_output_storage_to_memory_planning():
    after = _run(Before)
    ir.assert_structural_equal(after, Expected)


@pytest.mark.parametrize("start, step", [(2, 1), (0, 2), (3, 2)])
def test_outer_source_coordinates_are_separate_from_slot_ordinals(start, step):
    """Source addresses retain start/step; parent banks rotate by logical iteration."""
    before = parse(
        ir.python_print(Before)
        .replace("128, 16", "512, 16")
        .replace("pl.pipeline(4, stage=3)", f"pl.pipeline({start}, {start + 4 * step}, {step}, stage=3)")
    )

    # Only addresses change in the expected schedule. In particular, the
    # parent's ring still advances by one even when source indices skip rows.
    current = "group" if step == 1 else f"group * {step}"
    future = "group + 1" if step == 1 else f"(group + 1) * {step}"
    if start:
        current = f"{start} + {current}"
        future = f"{start} + ({future})"
    expected = parse(
        ir.python_print(Expected)
        .replace("128, 16", "512, 16")
        .replace(
            "pl.const(0, pl.INDEX) * pl.const(32, pl.INDEX)",
            f"pl.const({start}, pl.INDEX) * pl.const(32, pl.INDEX)",
        )
        .replace("(group + 1) * 32", f"({future}) * 32")
        .replace("group * 32", f"({current}) * 32")
    )
    ir.assert_structural_equal(_run(before), expected)


@pytest.mark.parametrize("children", [1, 2, 3, 5])
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_storage_depth_multiplies_stages_not_child_length(children, depth):
    before = parse(
        ir.python_print(Before)
        .replace("128, 16", "512, 16")
        .replace("pl.pipeline(2, stage=2)", f"pl.pipeline({children}, stage={depth})")
        .replace("group * 32", f"group * {children * 16}")
    )
    after = _run(before)
    roles = {ir.get_op(name).name: [] for name in ["tile.load", "tile.adds"]}
    loops = []

    class Regions(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if isinstance(op.value, ir.Call) and op.value.op.name in roles:
                assert isinstance(op.var.type, ir.TileType)
                roles[op.value.op.name].append(op.var.type.memref)
            super().visit_assign_stmt(op)

        def visit_for_stmt(self, op):
            loops.append(op)
            super().visit_for_stmt(op)

    Regions().visit_program(after)
    inputs, outputs = roles.values()
    assert inputs and outputs
    assert len(loops) == 1 and loops[0].kind == ir.ForKind.Sequential
    assert all(ref is not None and ref.slot_count_ == 3 * depth for ref in inputs)
    assert all(ref.base_.same_as(inputs[0].base_) for ref in inputs)
    assert all(ref is None for ref in outputs)


def test_nested_capacity_counts_a_shared_region_once():
    @pl.program
    class TooLarge:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(self, x: pl.Tensor[[2048, 256], pl.FP32], y: pl.Tensor[[2048, 256], pl.FP32]):
            for g in pl.pipeline(4, stage=3):
                for p in pl.pipeline(2, stage=2):
                    value: pl.Tile[[256, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        x, [(g * 2 + p) * 256, 0], [256, 256]
                    )
                    result: pl.Tile[[256, 256], pl.FP32, pl.Mem.Vec] = pl.tile.adds(value, 1.0)
                    _stored = pl.tile.store(result, [(g * 2 + p) * 256, 0], y)

    after = _run(TooLarge)
    with pytest.raises(ValueError, match="Vec buffer usage.*exceeds"):
        _plan_memory(after, allocate=True)


def test_pass_through_view_retains_one_stream_without_copy():
    @pl.program
    class PassThrough:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(self, x: pl.Tensor[[8, 64], pl.FP32], y: pl.Tensor[[8, 64], pl.FP32]):
            for g in pl.pipeline(4, stage=3):
                for p in pl.pipeline(2, stage=2):
                    value: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [g * 2 + p, 0], [1, 64])
                    view: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(value, [1, 64])
                    _stored = pl.tile.store(view, [g * 2 + p, 0], y)

    after = _run(PassThrough)
    refs = []
    view_sources = []

    class Regions(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if isinstance(op.var.type, ir.TileType):
                if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.reshape").name:
                    # InitMemRef binds mandatory views from their source later.
                    assert op.var.type.memref is None
                    assert isinstance(op.value.args[0].type, ir.TileType)
                    view_sources.append(op.value.args[0].type.memref)
                else:
                    refs.append(op.var.type.memref)
            super().visit_assign_stmt(op)

    Regions().visit_program(after)
    assert refs and all(ref is not None and ref.slot_count_ == 6 for ref in refs)
    assert all(ref.base_.same_as(refs[0].base_) for ref in refs)
    assert view_sources and all(ref.base_.same_as(refs[0].base_) for ref in view_sources)


def test_parent_and_child_own_hierarchical_versions():
    @pl.program
    class ParentAndChild:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(self, x: pl.Tensor[[12, 64], pl.FP32], y: pl.Tensor[[12, 64], pl.FP32]):
            for g in pl.pipeline(4, stage=3):
                parent_value: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [g * 3, 0], [1, 64])
                parent_result: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(parent_value, 1.0)
                _parent_stored = pl.tile.store(parent_result, [g * 3, 0], y)
                for p in pl.pipeline(2, stage=2):
                    child_value: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        x, [g * 3 + p + 1, 0], [1, 64]
                    )
                    child_result: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(child_value, 1.0)
                    _child_stored = pl.tile.store(child_result, [g * 3 + p + 1, 0], y)

    after = _plan_memory(_run(ParentAndChild))
    refs = {}

    class Regions(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if isinstance(op.var.type, ir.TileType):
                refs.setdefault(op.var.name_hint, []).append(op.var.type.memref)
            super().visit_assign_stmt(op)

    Regions().visit_program(after)
    assert set(refs) == {"parent_value", "parent_result", "child_value", "child_result"}
    for name, family in refs.items():
        expected = 3 if name.startswith("parent_") else 6
        assert all(ref is not None and ref.slot_count_ == expected for ref in family)
        assert all(ref.base_.same_as(family[0].base_) for ref in family)
    assert refs["parent_value"][0].base_.same_as(refs["parent_result"][0].base_)
    assert refs["child_value"][0].base_.same_as(refs["child_result"][0].base_)
    assert not refs["parent_value"][0].base_.same_as(refs["child_value"][0].base_)


@pytest.mark.parametrize("trips", [0, 1])
def test_short_outer_loop_keeps_the_child_plan(trips):
    before = parse(
        ir.python_print(Before).replace("pl.pipeline(4, stage=3)", f"pl.pipeline({trips}, stage=3)")
    )
    expected = parse(
        ir.python_print(Expected)
        .replace("pl.range(4, attrs=", f"pl.range({trips}, attrs=")
        .replace("pl.const(4, pl.INDEX)", f"pl.const({trips}, pl.INDEX)")
        .replace("group + 1 < 4", f"group + 1 < {trips}")
    )
    if not trips:
        # A zero-trip parent has no prologue. Dropping it before cloning also
        # avoids constructing unreachable source indices near INT64_MAX.
        class RemoveEmptyPrologue(ir.IRMutator):
            def visit_if_stmt(self, op):
                if (
                    isinstance(op.condition, ir.Lt)
                    and isinstance(op.condition.left, ir.ConstInt)
                    and isinstance(op.condition.right, ir.ConstInt)
                    and op.condition.left.value == op.condition.right.value == 0
                ):
                    return ir.SeqStmts([], op.span)
                return super().visit_if_stmt(op)

        assert isinstance(expected, ir.Program)
        expected = RemoveEmptyPrologue().visit_program(expected)
    ir.assert_structural_equal(_run(before), expected)


def test_one_parent_near_index_limit_omits_unreachable_prologue():
    """A false preload guard cannot protect host-side construction of an overflowing index."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(self, x: pl.Tensor[[16, 16], pl.FP32], y: pl.Tensor[[16, 16], pl.FP32]):
            for group in pl.pipeline(9223372036854775806, 9223372036854775807, 2, stage=3):
                for page in pl.pipeline(1, stage=4):
                    value: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [16, 16])
                    result: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(value, 1.0)
                    _stored = pl.tile.store(result, [0, 0], y)

    # The address deliberately ignores the source IV. After normalization this
    # is exactly one ordinary parent, with the same twelve declared slots.
    normalized = parse(
        ir.python_print(Before).replace(
            "pl.pipeline(9223372036854775806, 9223372036854775807, 2, stage=3)",
            "pl.pipeline(1, stage=3)",
        )
    )
    after = _run(Before)
    ir.assert_structural_equal(after, _run(normalized))
    loops = [stmt for stmt in _walk_stmts(after) if isinstance(stmt, ir.ForStmt)]
    assert len(loops) == 1 and loops[0].kind == ir.ForKind.Sequential
    assert isinstance(loops[0].stop, ir.ConstInt) and loops[0].stop.value == 1
    refs = [
        stmt.var.type.memref
        for stmt in _walk_stmts(after)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.var.type, ir.TileType)
        and stmt.var.type.memref is not None
    ]
    assert refs and all(ref.slot_count_ == 12 for ref in refs)


@pytest.mark.parametrize("trips", [65, 1024])
def test_oversized_child_retains_unpinned_fallback(trips):
    before = parse(
        ir.python_print(Before).replace("pl.pipeline(2, stage=2)", f"pl.pipeline({trips}, stage=2)")
    )
    ir.assert_structural_equal(_run(before), before)


@pytest.mark.parametrize("change", ["dynamic", "alias", "overflow"])
def test_unproved_nests_preserve_original_ir(change):
    text = ir.python_print(Before)
    if change == "dynamic":
        text = text.replace("pl.pipeline(2, stage=2)", "pl.pipeline(group + 1, stage=2)")
    elif change == "alias":
        text = text.replace("pl.tile.load(x,", "pl.tile.load(y,")
    else:
        text = text.replace("pl.pipeline(4, stage=3)", f"pl.pipeline({(1 << 63) - 1}, stage=3)")
    before = parse(text)
    ir.assert_structural_equal(_run(before), before)


@pytest.mark.parametrize("bound_kind", ["static", "dynamic"])
def test_large_outer_bound_with_safe_lookahead_remains_optimized(bound_kind):
    """Large outer bounds need room for bounded prefetch, not the child stride."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            y: pl.Tensor[[16, 16], pl.FP32],
            groups: pl.Scalar[pl.INDEX],
        ):
            for group in pl.pipeline(4611686018427387904, stage=3):
                for page in pl.pipeline(2, stage=2):
                    value: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [16, 16])
                    result: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(value, 1.0)
                    _stored = pl.tile.store(result, [0, 0], y)

    before = Before
    if bound_kind == "dynamic":
        before = parse(
            ir.python_print(Before).replace(
                f"pl.pipeline({1 << 62}, stage=3)",
                f"pl.pipeline(pl.min(groups, {1 << 62}), stage=3)",
            )
        )

    # The source addresses are loop-invariant. A child stride of two must not
    # double the 2**62 bound: declared stages only prefetch up to three parents.
    after = _run(before)
    loops = [stmt for stmt in _walk_stmts(after) if isinstance(stmt, ir.ForStmt)]
    assert len(loops) == 1 and loops[0].kind == ir.ForKind.Sequential
    refs = [
        stmt.var.type.memref
        for stmt in _walk_stmts(after)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.var.type, ir.TileType)
        and stmt.var.type.memref is not None
    ]
    assert refs and all(ref.slot_count_ == 6 for ref in refs)


@pytest.mark.parametrize(
    "original, replacement",
    [
        ("pl.pipeline(4, stage=3)", "pl.pipeline(-1, 3, stage=3)"),
        ("pl.pipeline(4, stage=3)", "pl.pipeline(4, 0, -1, stage=3)"),
        ("pl.pipeline(2, stage=2)", "pl.pipeline(2, 0, -1, stage=2)"),
        ("pl.pipeline(2, stage=2)", "pl.pipeline(2, stage=1)"),
        ("pl.pipeline(2, stage=2)", "pl.pipeline(2, stage=5)"),
    ],
)
def test_unsupported_nested_axes_preserve_original_ir(original, replacement):
    """A failed child or parent proof must leave the complete nest unmodified."""
    before = parse(ir.python_print(Before).replace(original, replacement))
    ir.assert_structural_equal(_run(before), before)


def test_total_specialization_budget_preserves_original_ir():
    """A 64-leaf tree still exceeds the budget once its eight parent phases count."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(self, x: pl.Tensor[[256, 64], pl.FP32], y: pl.Tensor[[256, 64], pl.FP32]):
            for g in pl.pipeline(4, stage=3):
                for p in pl.pipeline(8, stage=2):
                    for q in pl.pipeline(8, stage=2):
                        row: pl.Scalar[pl.INDEX] = g * 64 + p * 8 + q
                        value: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [row, 0], [1, 64])
                        result: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(value, 1.0)
                        _stored = pl.tile.store(result, [row, 0], y)

    ir.assert_structural_equal(_run(Before), Before)


def test_excessive_nested_depth_preserves_original_ir():
    """Five unit-trip children exceed the depth limit without exceeding phase count."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(self, x: pl.Tensor[[4, 64], pl.FP32], y: pl.Tensor[[4, 64], pl.FP32]):
            for g in pl.pipeline(4, stage=3):
                for p in pl.pipeline(1, stage=2):
                    for q in pl.pipeline(1, stage=2):
                        for r in pl.pipeline(1, stage=2):
                            for s in pl.pipeline(1, stage=2):
                                for t in pl.pipeline(1, stage=2):
                                    value: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                                        x, [g, 0], [1, 64]
                                    )
                                    result: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(value, 1.0)
                                    _stored = pl.tile.store(result, [g, 0], y)

    ir.assert_structural_equal(_run(Before), Before)


@pytest.mark.parametrize("stop, stage", [(4, 3), (5, 2)])
def test_ancestor_dependent_child_bounds_preserve_the_original_nest(stop, stage):
    """Specializing an ancestor must not disguise a nonuniform stream period.

    The innermost trip count becomes constant after cloning each page. Its
    period still differs between pages, invalidating a single-family schedule.
    """

    @pl.program
    class Nonuniform:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(self, x: pl.Tensor[[256, 64], pl.FP32], y: pl.Tensor[[256, 64], pl.FP32]):
            for group in pl.pipeline(4, stage=3):
                for page in pl.pipeline(1, stop, stage=stage):
                    for item in pl.pipeline(page, stage=stage):
                        value: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                            x, [group * 32 + page * 4 + item, 0], [1, 64]
                        )
                        result: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(value, 1.0)
                        _stored = pl.tile.store(result, [group * 32 + page * 4 + item, 0], y)

    ir.assert_structural_equal(_run(Nonuniform), Nonuniform)


@pytest.mark.parametrize("axis", ["parent", "child"])
@pytest.mark.parametrize("dtype", ["INT32", "INT64"])
def test_non_index_axes_preserve_the_original_nest(axis, dtype):
    """Slot coordinates require INDEX throughout, including their loop binders."""
    trips, stage = (4, 3) if axis == "parent" else (2, 2)
    before = parse(
        ir.python_print(Before).replace(
            f"pl.pipeline({trips}, stage={stage})",
            f"pl.pipeline(pl.const(0, pl.{dtype}), pl.const({trips}, pl.{dtype}), "
            f"pl.const(1, pl.{dtype}), stage={stage})",
        )
    )
    ir.assert_structural_equal(_run(before), before)


def test_live_child_result_prevents_speculative_storage_planning():
    @pl.program
    class Carried:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(self, x: pl.Tensor[[128, 16], pl.FP32], y: pl.Tensor[[128, 16], pl.FP32]):
            for group in pl.pipeline(4, stage=3):
                for page, (offset,) in pl.pipeline(2, stage=2, init_values=(pl.const(0, pl.INDEX),)):
                    value: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        x, [group * 32 + offset, 0], [16, 16]
                    )
                    _stored: pl.Tensor[[128, 16], pl.FP32] = pl.tile.store(value, [group * 32 + offset, 0], y)
                    next_offset: pl.Scalar[pl.INDEX] = offset + 16
                    offset = pl.yield_(next_offset)
                escaped: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [offset, 0], [16, 16])
                _final: pl.Tensor[[128, 16], pl.FP32] = pl.tile.store(escaped, [group * 32, 0], y)

    ir.assert_structural_equal(_run(Carried), Carried)


@pytest.mark.parametrize("call_location", ["initializer", "yield"])
@pytest.mark.parametrize("guarded", [False, True])
def test_dead_child_state_calls_preserve_original_ir(call_location, guarded):
    """Dead carry cleanup must retain writes embedded in original state expressions."""
    store = "pl.tile.store(seed, [0, 0], y)"
    initial = store if call_location == "initializer" else "y"
    yielded = "dead" if call_location == "initializer" else store
    guard = "            if group % 2 == 0:\n" if guarded else ""
    indent = "                " if guarded else "            "
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
    def kernel(self, x: pl.Tensor[[128, 16], pl.FP32],
               y: pl.Tensor[[128, 16], pl.FP32], z: pl.Tensor[[128, 16], pl.FP32]):
        seed: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [0, 0], [16, 16])
        for group in pl.pipeline(4, stage=3):
{guard}{indent}for page, (dead,) in pl.pipeline(2, stage=2, init_values=({initial},)):
{indent}    value: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(
{indent}        x, [group * 32 + page * 16, 0], [16, 16])
{indent}    result: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(value, 1.0)
{indent}    _stored = pl.tile.store(result, [group * 32 + page * 16, 0], z)
{indent}    dead = pl.yield_({yielded})
""")

    # The return is unused and the carry has no ordinary body use. Removing
    # either state slot before checking it would silently drop the write to y.
    child = next(stmt for stmt in _walk_stmts(before) if isinstance(stmt, ir.ForStmt) and stmt.iter_args)
    state = child.iter_args[0].initValue
    if call_location == "yield":
        state = next(stmt.value[0] for stmt in _walk_stmts(before) if isinstance(stmt, ir.YieldStmt))
    assert isinstance(state, ir.Call) and state.op.name == ir.get_op("tile.store").name
    ir.assert_structural_equal(_run(before), before)


@pytest.mark.parametrize("yield_form, children", [("expression", 8), ("variable", 32)])
def test_live_scalar_child_carry_bounds_expression_growth(yield_form, children):
    """Inline recurrence expressions must not expand exponentially during cloning."""
    recurrence = "                a = pl.yield_(a + a)"
    if yield_form == "variable":
        recurrence = (
            "                next_a: pl.Scalar[pl.FP32] = a + a\n                a = pl.yield_(next_a)"
        )
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
    def kernel(self, x: pl.Tensor[[128, 64], pl.FP32], y: pl.Tensor[[128, 64], pl.FP32],
               alpha: pl.Scalar[pl.FP32]):
        for group in pl.pipeline(4, stage=3):
            for page, (a,) in pl.pipeline({children}, stage=2, init_values=(alpha,)):
                value: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [group * 32 + page, 0], [1, 64])
                result: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(value, a)
                _stored = pl.tile.store(result, [group * 32 + page, 0], y)
{recurrence}
""")

    # Using a in compute keeps the recurrence live. Keep the negative case
    # small so a reintroduced exponential traversal fails CI promptly rather
    # than hanging it; the 32-step SSA form must still be specialized.
    after = _run(before)
    if yield_form == "expression":
        ir.assert_structural_equal(after, before)
    else:
        loops = [stmt for stmt in _walk_stmts(after) if isinstance(stmt, ir.ForStmt)]
        assert len(loops) == 1 and loops[0].kind == ir.ForKind.Sequential
        assert loops[0].attrs.get("software_pipeline_slots") == 3


@pytest.mark.parametrize("metadata_scope", ["child-return", "child-iter-arg"])
def test_child_result_used_only_in_view_metadata_preserves_original_ir(metadata_scope):
    """Child state referenced only by TileView metadata must survive conservative fallback."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(self, x: pl.Tensor[[128, 16], pl.FP32], y: pl.Tensor[[128, 16], pl.FP32]):
            for group in pl.pipeline(4, stage=3):
                for page, (_metadata_extent,) in pl.pipeline(
                    2, stage=2, init_values=(pl.const(16, pl.INDEX),)
                ):
                    value: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        x, [group * 32 + page * 16, 0], [16, 16]
                    )
                    result: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(value, 1.0)
                    child_view: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(result, [16, 16])
                    _stored = pl.tile.store(child_view, [group * 32 + page * 16, 0], y)
                    _metadata_extent = pl.yield_(pl.const(16, pl.INDEX))
                parent_value: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [group * 32, 0], [16, 16]
                )
                parent_result: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(parent_value, 2.0)
                view: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(parent_result, [16, 16])
                _parent_stored = pl.tile.store(view, [group * 32, 0], y)

    statements = list(_walk_stmts(Before))
    child = next(stmt for stmt in statements if isinstance(stmt, ir.ForStmt) and stmt.return_vars)
    view_name = "view" if metadata_scope == "child-return" else "child_view"
    view = next(
        stmt for stmt in statements if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint == view_name
    )
    assert len(child.return_vars) == len(child.iter_args) == 1 and isinstance(view.var.type, ir.TileType)
    assert isinstance(view.value, ir.Call)
    extent = child.return_vars[0] if metadata_scope == "child-return" else child.iter_args[0]
    old_type = view.var.type
    metadata = old_type.get_effective_tile_view()
    view_type = ir.TileType(
        old_type.shape,
        old_type.dtype,
        old_type.memref,
        ir.TileView(
            valid_shape=[extent, 16],
            stride=metadata.stride,
            start_offset=metadata.start_offset,
            blayout=metadata.blayout,
            slayout=metadata.slayout,
            fractal=metadata.fractal,
            pad=metadata.pad,
            compact=metadata.compact,
        ),
        old_type.memory_space,
    )
    replacement = ir.Var(view.var.name_hint, view_type, view.var.span)

    # Inject both declared/result types after DSL inference. Call operands keep
    # their static [16, 16] shape: extent has no ordinary expression use in
    # either case. Removing the return or iter-arg binder before inspecting
    # types leaves that metadata-only reference dangling.
    class AttachExtentMetadata(ir.IRMutator):
        def visit_var(self, op):
            return replacement if op.same_as(view.var) else op

        def visit_call(self, op):
            if op.same_as(view.value):
                return ir.Call(op.op, op.args, op.kwargs, op.attrs, view_type, op.span)
            return super().visit_call(op)

    before = AttachExtentMetadata().visit_program(Before)
    ir.assert_structural_equal(_run(before), before)


def test_unknown_call_is_preserved_before_nested_cleanup():
    @pl.program
    class UnknownEffects:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def effect(self, y: pl.Tensor[[128, 16], pl.FP32]) -> pl.Tensor[[128, 16], pl.FP32]:
            value: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(y, [0, 0], [16, 16])
            changed: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(value, 7.0)
            output: pl.Tensor[[128, 16], pl.FP32] = pl.tile.store(changed, [0, 0], y)
            return output

        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(self, x: pl.Tensor[[128, 16], pl.FP32], y: pl.Tensor[[128, 16], pl.FP32]):
            for g in pl.pipeline(4, stage=3):
                for p in pl.pipeline(2, stage=2):
                    _ignored: pl.Tensor[[128, 16], pl.FP32] = self.effect(y)
                    value: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        x, [g * 32 + p * 16, 0], [16, 16]
                    )
                    _stored: pl.Tensor[[128, 16], pl.FP32] = pl.tile.store(value, [g * 32 + p * 16, 0], y)

    ir.assert_structural_equal(_run(UnknownEffects), UnknownEffects)


def test_non_inplace_operator_keeps_distinct_storage():
    before = parse(ir.python_print(Before).replace("pl.tile.adds(value, 1.0)", "pl.tile.recip(value)"))
    expected_text = ir.python_print(Expected)
    for phase in (0, 1):
        expected_text = expected_text.replace(
            f"pl.tile.adds(read{phase}, 1.0)", f"pl.tile.recip(read{phase})"
        )
        # Exact printing preserves a call's result type through the dump surface
        # when its input has a slot annotation that the result does not inherit.
        expected_text = expected_text.replace(
            f'pl._dump.call("tile.adds", [read{phase}, 1.0],',
            f'pl._dump.call("tile.recip", [read{phase}],',
        )
        expected_text = expected_text.replace(
            f'result{phase}: pl.Tile[[16, 16], pl.FP32, pl.MemRef("pipe_value_0", slots=6)',
            f'result{phase}: pl.Tile[[16, 16], pl.FP32, pl.MemRef("pipe_result_1", slots=6)',
        )
    ir.assert_structural_equal(_run(before), parse(expected_text))


def test_later_alias_read_prevents_reuse_until_last_use():
    @pl.program
    class LiveAlias:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(
            self,
            x: pl.Tensor[[8, 64], pl.FP32],
            y: pl.Tensor[[8, 64], pl.FP32],
            z: pl.Tensor[[8, 64], pl.FP32],
        ):
            for g in pl.pipeline(4, stage=3):
                for p in pl.pipeline(2, stage=2):
                    value: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [g * 2 + p, 0], [1, 64])
                    view: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(value, [1, 64])
                    early: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(value, 1.0)
                    late: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(view, 2.0)
                    _early = pl.tile.store(early, [g * 2 + p, 0], y)
                    _late = pl.tile.store(late, [g * 2 + p, 0], z)

    after = _plan_memory(_run(LiveAlias))
    refs = {}

    class Regions(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if op.var.name_hint in {"value", "early", "late"}:
                assert isinstance(op.var.type, ir.TileType)
                refs.setdefault(op.var.name_hint, []).append(op.var.type.memref)
            super().visit_assign_stmt(op)

    Regions().visit_program(after)
    assert set(refs) == {"value", "early", "late"}
    assert all(ref.slot_count_ == 6 for name in ("value", "late") for ref in refs[name])
    assert all(ref.slot_count_ == 1 for ref in refs["early"])
    assert all(ref.base_.same_as(refs["value"][0].base_) for ref in refs["late"])
    assert all(not ref.base_.same_as(refs["value"][0].base_) for ref in refs["early"])


def _result_slots(program, outer_iteration):
    """Evaluate declared consumer coordinates, not the pipeline implementation."""
    definitions = []
    results = []
    outer = []

    class Coordinates(ir.IRVisitor):
        def visit_for_stmt(self, op):
            outer.append(op.loop_var)
            super().visit_for_stmt(op)

        def visit_assign_stmt(self, op):
            definitions.append((op.var, op.value))
            if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.adds").name:
                assert isinstance(op.value.args[0].type, ir.TileType)
                results.append(op.value.args[0].type.memref)
            super().visit_assign_stmt(op)

    Coordinates().visit_program(program)
    assert len(outer) == 1

    def evaluate(expr):
        if isinstance(expr, ir.ConstInt):
            return expr.value
        if isinstance(expr, ir.Var):
            if expr.same_as(outer[0]):
                return outer_iteration
            return evaluate(next(value for var, value in definitions if var.same_as(expr)))
        if isinstance(expr, ir.Add):
            return evaluate(expr.left) + evaluate(expr.right)
        if isinstance(expr, ir.Mul):
            return evaluate(expr.left) * evaluate(expr.right)
        if isinstance(expr, ir.FloorMod):
            return evaluate(expr.left) % evaluate(expr.right)
        raise AssertionError(f"Unexpected slot expression: {expr}")

    return [(ref.slot_count_, evaluate(ref.slot_index_)) for ref in results]


@pytest.mark.parametrize("group", [0, 1, 2, 3])
def test_child_trip_count_does_not_define_storage_coordinates(group):
    before = parse(
        ir.python_print(Before)
        .replace("128, 16", "512, 16")
        .replace("pl.pipeline(2, stage=2)", "pl.pipeline(5, stage=2)")
        .replace("group * 32", "group * 80")
    )
    # Three parent contexts, each holding child residues 0 and 1.
    base = group % 3
    expected = [(6, base + page) for page in [0, 3, 0, 3, 0]]
    assert _result_slots(_run(before), group) == expected


def test_three_levels_use_logical_iteration_residues():
    @pl.program
    class ThreeLevels:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], y: pl.Tensor[[64, 64], pl.FP32]):
            for g in pl.pipeline(4, stage=3):
                for p in pl.pipeline(2, 12, 2, stage=2):
                    for q in pl.pipeline(3, 9, 3, stage=3):
                        row: pl.Scalar[pl.INDEX] = g * 10 + (p // 2 - 1) * 2 + q // 3 - 1
                        value: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(x, [row, 0], [1, 64])
                        result: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(value, 1.0)
                        _stored = pl.tile.store(result, [row, 0], y)

    after = _run(ThreeLevels)
    residues = [0, 1, 3, 4, 0, 1, 3, 4, 0, 1]
    assert _result_slots(after, 0) == [(18, 3 * residue) for residue in residues]
    assert _result_slots(after, 1) == [(18, 3 * residue + 1) for residue in residues]
    assert _result_slots(after, 3) == [(18, 3 * residue) for residue in residues]


def test_short_child_cannot_prefetch_over_a_live_parent_context():
    before = parse(
        ir.python_print(Before)
        .replace("pl.pipeline(4, stage=3)", "pl.pipeline(4, stage=2)")
        .replace("pl.pipeline(2, stage=2)", "pl.pipeline(1, stage=4)")
    )
    after = _run(before)
    assert _result_slots(after, 0) == [(8, 0)]
    assert _result_slots(after, 1) == [(8, 1)]
    assert _result_slots(after, 2) == [(8, 0)]
    # The third parent maps onto the current parent's context. Only one future
    # parent can be prefetched; the child still owns all four declared slots.
    loop = []
    loads = []

    class Schedule(ir.IRVisitor):
        def visit_for_stmt(self, op):
            loop.append(op)
            super().visit_for_stmt(op)

        def visit_assign_stmt(self, op):
            if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.load").name:
                loads.append(op)
            super().visit_assign_stmt(op)

    Schedule().visit_program(after)
    assert len(loop) == 1
    assert len(loads) == 2  # One prologue load and one guarded future load.


def test_guarded_short_leaf_does_not_overwrite_an_earlier_parent():
    @pl.program
    class Guarded:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def kernel(self, x: pl.Tensor[[20, 64], pl.FP32], y: pl.Tensor[[20, 64], pl.FP32]):
            for g in pl.pipeline(4, stage=3):
                for p in pl.pipeline(5, stage=2):
                    if (g + p) % 2 == 0:
                        for q in pl.pipeline(1, stage=4):
                            value: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                                x, [g * 5 + p + q, 0], [1, 64]
                            )
                            result: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(value, 1.0)
                            _stored = pl.tile.store(result, [g * 5 + p + q, 0], y)

    after = _run(Guarded)
    operations = []

    class Order(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if isinstance(op.value, ir.Call):
                if op.value.op.name == ir.get_op("tile.load").name:
                    operations.append("load")
                elif op.value.op.name == ir.get_op("tile.adds").name:
                    operations.append("compute")
            super().visit_assign_stmt(op)

    Order().visit_program(after)
    assert operations == [
        "load",
        "load",
        "compute",
        "compute",
        "load",
        "compute",
        "load",
        "compute",
        "load",
        "compute",
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
