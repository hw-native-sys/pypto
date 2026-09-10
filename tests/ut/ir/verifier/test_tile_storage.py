# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Canonical storage boundaries and allocated effective-address validation."""

import pytest
from pypto import DataType, Error, ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

SPAN = ir.Span.unknown()
PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]


def _int(value):
    return ir.ConstInt(value, DataType.INDEX, SPAN)


def _tile(
    name, *, base=None, offset: int | ir.Expr = 0, size=128, space=ir.MemorySpace.Vec, slots=1, slot=None
):
    base = base if base is not None else ir.Var(name + "_base", ir.PtrType(), SPAN)
    memref = ir.MemRef(base, offset, size, SPAN, slots=slots, slot=slot)
    return ir.Var(name, ir.TileType([4, 8], DataType.FP32, memref, None, space), SPAN)


def _memref(value):
    tile = value.type
    assert isinstance(tile, ir.TileType) and tile.memref is not None
    return tile.memref


def _program(body, params=(), func_type=ir.FunctionType.InCore):
    return ir.Program([ir.Function("kernel", list(params), [], body, SPAN, type=func_type)], "Storage", SPAN)


def _verify(program, physical=False):
    properties = passes.IRPropertySet()
    properties.insert(
        passes.IRProperty.TileStorageAllocated if physical else passes.IRProperty.TileStorageLegalized
    )
    return passes.PropertyVerifierRegistry.verify(properties, program)


def _loop(initials, *, while_loop=False, yielded=None, results=None):
    arguments = [ir.IterArg(f"carry_{i}", value.type, value, SPAN) for i, value in enumerate(initials)]
    results = (
        results
        if results is not None
        else [ir.Var(f"result_{i}", v.type, SPAN) for i, v in enumerate(initials)]
    )
    body = ir.YieldStmt(list(arguments) if yielded is None else yielded, SPAN)
    if while_loop:
        return ir.WhileStmt(ir.ConstBool(False, SPAN), arguments, body, results, SPAN)
    induction = ir.Var("i", ir.ScalarType(DataType.INDEX), SPAN)
    return ir.ForStmt(induction, _int(0), _int(4), _int(1), arguments, body, results, SPAN)


def _assert_error(program, message, physical=False):
    diagnostics = _verify(program, physical)
    assert any(message in diagnostic.message for diagnostic in diagnostics), diagnostics


@pytest.mark.parametrize("while_loop", [False, True])
@pytest.mark.parametrize("physical", [False, True])
def test_distinct_carry_windows_are_canonical(while_loop, physical):
    initial = [_tile("a", offset=0), _tile("b", offset=128)]
    assert _verify(_program(_loop(initial, while_loop=while_loop), initial), physical) == []


@pytest.mark.parametrize("while_loop", [False, True])
@pytest.mark.parametrize("mismatch", ["yield", "result"])
def test_every_carry_boundary_must_name_one_window(while_loop, mismatch):
    initial, other = _tile("initial"), _tile("other", offset=128)
    loop = _loop(
        [initial],
        while_loop=while_loop,
        yielded=[other] if mismatch == "yield" else None,
        results=[other] if mismatch == "result" else None,
    )
    _assert_error(_program(loop, [initial, other]), "canonical storage")


@pytest.mark.parametrize("while_loop", [False, True])
def test_two_carries_cannot_overwrite_the_same_initial_window(while_loop):
    initial = _tile("initial")
    _assert_error(_program(_loop([initial, initial], while_loop=while_loop), [initial]), "windows overlap")


def test_same_allocation_disjoint_slots_are_valid():
    first = _tile("first")
    second = _tile("second", base=_memref(first).base_, offset=128)
    assert _verify(_program(_loop([first, second]), [first, second])) == []


def test_unknown_offsets_in_one_allocation_fail_closed():
    first = _tile("first")
    offset = ir.Var("offset", ir.ScalarType(DataType.INDEX), SPAN)
    second = _tile("second", base=_memref(first).base_, offset=offset)
    _assert_error(_program(_loop([first, second]), [first, second, offset]), "unprovable symbolic overlap")


def test_placed_ranges_must_be_disjoint_even_with_distinct_bases():
    first, second = _tile("first"), _tile("second", offset=64)
    program = _program(_loop([first, second]), [first, second])
    assert _verify(program) == []
    _assert_error(program, "windows overlap", physical=True)


def test_distinct_memory_spaces_can_share_an_effective_address():
    first, second = _tile("first"), _tile("second", space=ir.MemorySpace.Mat)
    assert _verify(_program(_loop([first, second]), [first, second]), physical=True) == []


@pytest.mark.parametrize("same_arm", [False, True])
def test_branch_yields_must_match_declared_storage(same_arm):
    target, other = _tile("target"), _tile("other")
    branch = ir.IfStmt(
        ir.ConstBool(True, SPAN),
        ir.YieldStmt([target], SPAN),
        ir.YieldStmt([target if same_arm else other], SPAN),
        [target],
        SPAN,
    )
    program = _program(branch, [target, other])
    if same_arm:
        assert _verify(program) == []
    else:
        _assert_error(program, "Every tile branch arm")


@pytest.mark.parametrize("offset", [0, 64, 128])
def test_allocated_move_checks_effective_overlap(offset):
    source, destination = _tile("source"), _tile("destination", offset=offset)
    call = ir.Call(
        ir.get_op("tile.move"), [source], {"target_memory": ir.MemorySpace.Vec}, destination.type, SPAN
    )
    program = _program(ir.AssignStmt(destination, call, SPAN), [source])
    if offset == 64:
        _assert_error(program, "windows overlap", physical=True)
    else:
        assert _verify(program, physical=True) == []


def test_tiles_without_storage_and_nested_tile_tuples_are_rejected():
    tile = ir.Var("missing", ir.TileType([4, 8], DataType.FP32), SPAN)
    _assert_error(_program(ir.ReturnStmt(SPAN), [tile]), "must have a MemRef")
    nested = ir.Var("tuple", ir.TupleType([ir.TupleType([_tile("a").type])]), SPAN)
    _assert_error(_program(ir.ReturnStmt(SPAN), [nested]), "must be flattened")


def test_scalar_and_gm_control_flow_does_not_require_tile_storage():
    scalar = ir.Var("scalar", ir.ScalarType(DataType.INDEX), SPAN)
    gm = ir.Var("gm", ir.TensorType([4, 8], DataType.FP32), SPAN)
    assert _verify(_program(_loop([scalar, gm]), [scalar, gm]), physical=True) == []


def test_orchestration_is_outside_storage_property_scope():
    tile = ir.Var("tile", ir.TileType([4, 8], DataType.FP32), SPAN)
    assert _verify(_program(ir.ReturnStmt(SPAN), [tile], ir.FunctionType.Orchestration)) == []


def test_named_pass_validates_even_when_automatic_verification_is_disabled():
    first = _tile("first")
    program = _program(_loop([first, first]), [first])
    with passes.PassContext([], passes.VerificationLevel.NONE), pytest.raises(Error, match="windows overlap"):
        passes.verify_tile_storage()(program)


@pytest.mark.parametrize("planner", PLANNERS)
def test_named_storage_check_precedes_address_placement_for_every_planner(planner):
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        names = PassManager(OptimizationStrategy.Default).get_pass_names()
    assert names.count("VerifyTileStorage") == 1
    assert names.index("MaterializeSemanticAliases") < names.index("VerifyTileStorage")
    if "MemoryReuse" in names:
        assert names.index("MemoryReuse") < names.index("VerifyTileStorage")
    if "AllocateMemoryAddr" in names:
        assert names.index("VerifyTileStorage") < names.index("AllocateMemoryAddr")


def test_pipeline_rejects_changing_buffer_mode_after_construction():
    with passes.PassContext([], enable_buffer_ir=True):
        manager = PassManager(OptimizationStrategy.Default)
    with pytest.raises(RuntimeError, match="enable_buffer_ir changed"):
        manager.run_passes(_program(ir.ReturnStmt(SPAN)))


def test_declared_slot_metadata_is_preserved_at_carry_boundaries():
    initial = _tile("initial", slots=2, slot=_int(0))
    other = _tile("other", base=_memref(initial).base_, slots=2, slot=_int(1))
    _assert_error(_program(_loop([initial], yielded=[other]), [initial, other]), "canonical storage")
    sibling = _tile("sibling", base=_memref(initial).base_, offset=128, slots=2, slot=_int(1))
    assert _verify(_program(_loop([initial, sibling]), [initial, sibling])) == []


def test_only_the_trailing_yield_establishes_a_region_boundary():
    initial = _tile("initial")
    body = ir.SeqStmts([ir.YieldStmt([initial], SPAN), ir.EvalStmt(_int(0), SPAN)], SPAN)
    branch = ir.IfStmt(ir.ConstBool(True, SPAN), body, body, [initial], SPAN)
    _assert_error(_program(branch, [initial]), "Every tile branch arm")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
