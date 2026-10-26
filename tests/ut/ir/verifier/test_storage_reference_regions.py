# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Storage references select existing handles without claiming a new allocation."""

import pytest
from pypto import DataType, ir, passes

SPAN = ir.Span.unknown()


def _int(value):
    return ir.ConstInt(value, DataType.INDEX, SPAN)


def _program(body, params):
    function = ir.Function("kernel", params, [], body, SPAN, type=ir.FunctionType.InCore)
    return ir.Program([function], "StorageReferences", SPAN)


def _verify(program, property_):
    properties = passes.IRPropertySet()
    properties.insert(property_)
    return passes.PropertyVerifierRegistry.verify(properties, program)


def _tile(name, *, concrete=False, offset=0):
    memref = None
    if concrete:
        base = ir.Var(name + "_base", ir.PtrType(), SPAN)
        memref = ir.MemRef(base, offset, 512, SPAN)
    return ir.Var(name, ir.TileType([16, 16], DataType.FP16, memref, None, ir.MemorySpace.Mat), SPAN)


def _selector(first, second, result):
    return ir.IfStmt(
        ir.ConstBool(True, SPAN),
        ir.YieldStmt([first], SPAN),
        ir.YieldStmt([second], SPAN),
        [result],
        SPAN,
    )


@pytest.mark.parametrize(
    "property_",
    [passes.IRProperty.TileStorageLegalized, passes.IRProperty.TileStorageAllocated],
)
def test_mat_selector_accepts_distinct_concrete_origins_without_a_result_memref(property_):
    first, second = _tile("first", concrete=True), _tile("second", concrete=True, offset=512)
    selected = _tile("selected")
    body = ir.SeqStmts([_selector(first, second, selected), ir.ReturnStmt(SPAN)], SPAN)
    assert _verify(_program(body, [first, second]), property_) == []


@pytest.mark.parametrize(
    "property_",
    [passes.IRProperty.TileStorageLegalized, passes.IRProperty.TileStorageAllocated],
)
@pytest.mark.parametrize("both_borrowed", [False, True])
def test_selector_accepts_caller_provided_intermediate_tile_values(property_, both_borrowed):
    first = _tile("borrowed_first")
    second = _tile("second", concrete=not both_borrowed, offset=512)
    selected = _tile("selected")
    body = ir.SeqStmts([_selector(first, second, selected), ir.ReturnStmt(SPAN)], SPAN)
    # Caller-provided intermediate Tile values are known storage sources. A
    # selector retains that proof for the borrowed endpoint and its result;
    # final PTO emission still requires these parameters to be inlined or bound.
    assert _verify(_program(body, [first, second]), property_) == []


def test_reference_normalization_preserves_borrowed_parameter_identity():
    first, second, selected = _tile("first"), _tile("second"), _tile("selected")
    program = _program(
        ir.SeqStmts([_selector(first, second, selected), ir.ReturnStmt(SPAN)], SPAN), [first, second]
    )
    normalized = passes.materialize_semantic_aliases()(program)
    function = next(iter(normalized.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    branch = function.body.stmts[0]
    assert isinstance(branch, ir.IfStmt)
    assert isinstance(branch.then_body, ir.YieldStmt)
    assert branch.else_body is not None and isinstance(branch.else_body, ir.YieldStmt)
    assert branch.then_body.value[0].same_as(function.params[0])
    assert branch.else_body.value[0].same_as(function.params[1])
    assert _verify(normalized, passes.IRProperty.TileStorageLegalized) == []


@pytest.mark.parametrize("while_loop", [False, True])
def test_reference_loop_keeps_initializer_and_backedge_handle_semantics(while_loop):
    first, second = _tile("first", concrete=True), _tile("second", concrete=True, offset=512)
    selected, result = _tile("selected"), _tile("result")
    current = ir.IterArg("current", selected.type, selected, SPAN)
    body = ir.YieldStmt([second], SPAN)
    if while_loop:
        loop = ir.WhileStmt(ir.ConstBool(False, SPAN), [current], body, [result], SPAN)
    else:
        induction = ir.Var("i", ir.ScalarType(DataType.INDEX), SPAN)
        loop = ir.ForStmt(induction, _int(0), _int(0), _int(1), [current], body, [result], SPAN)
    program = _program(
        ir.SeqStmts([_selector(first, second, selected), loop, ir.ReturnStmt(SPAN)], SPAN),
        [first, second],
    )
    # A zero-trip exit retains selected; the unreachable backedge does not
    # justify copying second into that initializer's allocation.
    assert _verify(program, passes.IRProperty.TileStorageLegalized) == []


@pytest.mark.parametrize("invalid_source", [False, True])
def test_missing_memref_is_accepted_only_for_a_closed_reference_graph(invalid_source):
    first = _tile("first", concrete=True)
    missing = _tile("missing")
    selected = _tile("selected")
    body = _selector(first, missing, selected) if invalid_source else ir.EvalStmt(missing, SPAN)
    diagnostics = _verify(_program(body, [first]), passes.IRProperty.TileStorageLegalized)
    assert any("must have a MemRef" in diagnostic.message for diagnostic in diagnostics)


def test_concrete_if_result_still_requires_identical_incoming_windows():
    first, second = _tile("first", concrete=True), _tile("second", concrete=True, offset=512)
    result = _tile("result", concrete=True, offset=1024)
    diagnostics = _verify(
        _program(_selector(first, second, result), [first, second]),
        passes.IRProperty.TileStorageLegalized,
    )
    assert any("yield its declared storage" in diagnostic.message for diagnostic in diagnostics)


@pytest.mark.parametrize("difference", ["shape", "dtype", "memory", "layout", "padding"])
def test_tile_reference_region_rejects_incompatible_physical_descriptors(difference):
    first = _tile("first", concrete=True)
    second = _tile("second", concrete=True, offset=512)
    first_type, second_type = first.type, second.type
    assert isinstance(first_type, ir.TileType) and isinstance(second_type, ir.TileType)
    shape = [32, 16] if difference == "shape" else [16, 16]
    dtype = DataType.FP32 if difference == "dtype" else DataType.FP16
    memory = ir.MemorySpace.Vec if difference == "memory" else ir.MemorySpace.Mat
    view = None
    if difference in ("layout", "padding"):
        original = first_type.get_effective_tile_view()
        layout = original.blayout
        if difference == "layout":
            layout = ir.TileLayout.row_major if layout == ir.TileLayout.col_major else ir.TileLayout.col_major
        view = ir.TileView(
            valid_shape=original.valid_shape,
            stride=original.stride,
            start_offset=original.start_offset,
            blayout=layout,
            slayout=original.slayout,
            fractal=original.fractal,
            pad=ir.PadValue.max if difference == "padding" else original.pad,
            compact=original.compact,
        )
    second_type = ir.TileType(shape, dtype, second_type.memref, view, memory)
    second = ir.Var("second", second_type, SPAN)
    if difference == "layout":
        # Mat's implicit block layout is already column-major. The negative
        # fixture must actually differ from that resolved default.
        assert second_type.get_effective_tile_view().blayout != first_type.get_effective_tile_view().blayout
    selected = _tile("selected")
    diagnostics = _verify(
        _program(_selector(first, second, selected), [first, second]),
        passes.IRProperty.TileStorageLegalized,
    )
    assert diagnostics, "A selected handle cannot relayout or reinterpret its incoming storage"


def test_allocated_same_space_reference_copy_requires_proven_disjointness():
    vector = ir.MemorySpace.Vec
    first = _tile("first", concrete=True)
    second = _tile("second", concrete=True, offset=512)
    first_type, second_type = first.type, second.type
    assert isinstance(first_type, ir.TileType) and isinstance(second_type, ir.TileType)
    first = ir.Var("first", ir.TileType([16, 16], DataType.FP16, first_type.memref, None, vector), SPAN)
    second = ir.Var("second", ir.TileType([16, 16], DataType.FP16, second_type.memref, None, vector), SPAN)
    selected = ir.Var("selected", ir.TileType([16, 16], DataType.FP16, memory_space=vector), SPAN)
    # A distinct symbolic base is insufficient after allocation: this output
    # has address zero and may overwrite the selected first source's bytes.
    destination = _tile("destination", concrete=True)
    destination_type = destination.type
    assert isinstance(destination_type, ir.TileType)
    destination = ir.Var(
        "destination", ir.TileType([16, 16], DataType.FP16, destination_type.memref, None, vector), SPAN
    )
    transfer = ir.AssignStmt(
        destination, ir.Call(ir.get_op("tile.move"), [selected], destination.type, SPAN), SPAN
    )
    body = ir.SeqStmts([_selector(first, second, selected), transfer], SPAN)
    diagnostics = _verify(_program(body, [first, second]), passes.IRProperty.TileStorageAllocated)
    assert any("proven disjoint storage" in diagnostic.message for diagnostic in diagnostics)


@pytest.mark.parametrize("kind", ["if", "loop"])
@pytest.mark.parametrize("extra", [False, True])
def test_tile_reference_region_requires_exact_yield_arity(kind, extra):
    first, second = _tile("first", concrete=True), _tile("second", concrete=True, offset=512)
    selected = _tile("selected")
    values: list[ir.Expr] = [first, second] if extra else []
    if kind == "if":
        body = ir.IfStmt(
            ir.ConstBool(True, SPAN),
            ir.YieldStmt(values, SPAN),
            ir.YieldStmt([second], SPAN),
            [selected],
            SPAN,
        )
    else:
        carry = ir.IterArg("carry", selected.type, first, SPAN)
        body = ir.WhileStmt(ir.ConstBool(False, SPAN), [carry], ir.YieldStmt(values, SPAN), [selected], SPAN)
    diagnostics = _verify(_program(body, [first, second]), passes.IRProperty.TileStorageLegalized)
    assert diagnostics, "Each reference region must yield exactly one value for each result slot"


@pytest.mark.parametrize("endpoint", ["initializer", "backedge", "result"])
def test_tile_reference_loop_rejects_a_mismatched_memory_space(endpoint):
    first, second = _tile("first", concrete=True), _tile("second", concrete=True, offset=512)
    selected = _tile("selected")
    invalid_base = ir.Var("invalid_base", ir.PtrType(), SPAN)
    invalid_memory = ir.MemRef(invalid_base, 1024, 512, SPAN)
    invalid_type = ir.TileType([16, 16], DataType.FP16, invalid_memory, None, ir.MemorySpace.Vec)
    invalid = ir.Var("invalid", invalid_type, SPAN)
    initial = invalid if endpoint == "initializer" else first
    yielded = invalid if endpoint == "backedge" else second
    result = invalid if endpoint == "result" else selected
    carry = ir.IterArg("carry", selected.type, initial, SPAN)
    loop = ir.WhileStmt(ir.ConstBool(False, SPAN), [carry], ir.YieldStmt([yielded], SPAN), [result], SPAN)
    diagnostics = _verify(_program(loop, [first, second, invalid]), passes.IRProperty.TileStorageLegalized)
    assert diagnostics, "A loop reference cannot change its selected memory space at a boundary"


def _buffer(name, *, rows=16, space=ir.MemorySpace.Vec):
    return ir.Var(name, ir.BufferType([rows, 16], DataType.FP16, space), SPAN)


def _call(name, arguments, type_=None):
    type_ = type_ if type_ is not None else ir.VoidType()
    return ir.Call(ir.Op(name), arguments, {}, {}, type_, SPAN)


@pytest.mark.parametrize("kind", ["if", "for", "while"])
def test_buffer_regions_accept_matching_handle_signatures(kind):
    first, second, selected = _buffer("first"), _buffer("second"), _buffer("selected")
    if kind == "if":
        region = _selector(first, second, selected)
    else:
        current = ir.IterArg("current", first.type, first, SPAN)
        body = ir.YieldStmt([second], SPAN)
        if kind == "while":
            region = ir.WhileStmt(ir.ConstBool(False, SPAN), [current], body, [selected], SPAN)
        else:
            induction = ir.Var("i", ir.ScalarType(DataType.INDEX), SPAN)
            region = ir.ForStmt(induction, _int(0), _int(2), _int(1), [current], body, [selected], SPAN)
    assert _verify(_program(region, [first, second]), passes.IRProperty.BufferIR) == []


@pytest.mark.parametrize("failure", ["descriptor", "arity", "missing_yield"])
def test_buffer_reference_region_rejects_invalid_endpoint_contracts(failure):
    first, selected = _buffer("first"), _buffer("selected")
    second = _buffer("second", rows=8 if failure == "descriptor" else 16)
    then = ir.YieldStmt([] if failure == "arity" else [first], SPAN)
    otherwise = ir.ReturnStmt(SPAN) if failure == "missing_yield" else ir.YieldStmt([second], SPAN)
    region = ir.IfStmt(ir.ConstBool(True, SPAN), then, otherwise, [selected], SPAN)
    diagnostics = _verify(_program(region, [first, second]), passes.IRProperty.BufferIR)
    expected = {
        "descriptor": "declared descriptor",
        "arity": "Yield values",
        "missing_yield": "final yield",
    }
    assert any(expected[failure] in diagnostic.message for diagnostic in diagnostics)


def test_selected_mat_handle_can_feed_a_cube_operand_copy():
    first = _buffer("first", space=ir.MemorySpace.Mat)
    second = _buffer("second", space=ir.MemorySpace.Mat)
    selected = _buffer("selected", space=ir.MemorySpace.Mat)
    left = _buffer("left", space=ir.MemorySpace.Left)
    copy = ir.EvalStmt(_call("buffer.copy", [selected, left]), SPAN)
    body = ir.SeqStmts([_selector(first, second, selected), copy], SPAN)
    assert _verify(_program(body, [first, second, left]), passes.IRProperty.BufferIR) == []


def test_selected_same_space_window_requires_a_disjointness_proof():
    first, second, selected, output = (_buffer(name) for name in ("first", "second", "selected", "output"))
    copy = ir.EvalStmt(_call("buffer.copy", [selected, output]), SPAN)
    body = ir.SeqStmts([_selector(first, second, selected), copy], SPAN)
    diagnostics = _verify(_program(body, [first, second, output]), passes.IRProperty.BufferIR)
    assert any("runtime-selected storage handles" in diagnostic.message for diagnostic in diagnostics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
