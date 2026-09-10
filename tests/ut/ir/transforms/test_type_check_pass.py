# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for type checking via run_verifier()."""

from typing import Any

import pypto
import pytest
from pypto import DataType, ir, passes

_SPAN = ir.Span.unknown()


def _idx(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, DataType.INDEX, _SPAN)


def _sym(name: str) -> ir.Var:
    return ir.Var(name, ir.ScalarType(DataType.INDEX), _SPAN)


def _tensor_type(valid_extent: ir.Expr, *, pad: ir.PadValue = ir.PadValue.null) -> ir.TensorType:
    view = ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=[valid_extent], pad=pad)
    return ir.TensorType([_idx(16)], DataType.FP32, None, view)


def _typecheck_diagnostics(stmt: ir.Stmt, *, params: list[ir.Var] | None = None) -> list[passes.Diagnostic]:
    body = ir.SeqStmts([stmt, ir.ReturnStmt([], _SPAN)], _SPAN)
    func = ir.Function("main", params or [], [], body, _SPAN)
    program = ir.Program([func], "test", _SPAN)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.TypeChecked)
    return passes.PropertyVerifierRegistry.verify(properties, program)


def _if_with_types(then_type: ir.Type, else_type: ir.Type, return_type: ir.Type) -> ir.IfStmt:
    condition = ir.ConstInt(1, DataType.BOOL, _SPAN)
    then_body = ir.YieldStmt([ir.Var("then_value", then_type, _SPAN)], _SPAN)
    else_body = ir.YieldStmt([ir.Var("else_value", else_type, _SPAN)], _SPAN)
    return_var = ir.Var("result", return_type, _SPAN)
    return ir.IfStmt(condition, then_body, else_body, [return_var], _SPAN)


def test_type_check_for_type_mismatch():
    """Test type checking detects type mismatch in ForStmt."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    iter_arg = ir.IterArg("sum", ir.ScalarType(DataType.INT64), a, span)  # INT64
    yield_value = ir.ConstFloat(1.0, DataType.FP32, span)  # FP32 - mismatch!
    body = ir.YieldStmt([yield_value], span)
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INT64, span),
        ir.ConstInt(10, DataType.INT64, span),
        ir.ConstInt(1, DataType.INT64, span),
        [iter_arg],
        body,
        [result_var],
        span,
    )

    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_type_mismatch", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    # Run type checking via run_verifier - should raise on type mismatch errors
    verify_pass = passes.run_verifier()
    with pytest.raises(pypto.Error, match="Dtype mismatch in ForStmt"):
        verify_pass(program)


def test_type_check_if_type_mismatch():
    """Test type checking detects type mismatch in IfStmt."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
    then_body = ir.YieldStmt([ir.ConstInt(1, DataType.INT64, span)], span)  # INT64
    else_body = ir.YieldStmt([ir.ConstFloat(2.0, DataType.FP32, span)], span)  # FP32 - mismatch!
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    if_stmt = ir.IfStmt(condition, then_body, else_body, [result_var], span)

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_if_type_mismatch", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    # Run type checking via run_verifier - should raise on type mismatch error
    verify_pass = passes.run_verifier()
    with pytest.raises(pypto.Error, match="Dtype mismatch in IfStmt"):
        verify_pass(program)


def test_type_check_tensor_shape_mismatch():
    """Test type checking detects shape mismatch in TensorType."""
    span = ir.Span.unknown()

    shape1 = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(20, DataType.INT64, span)]
    shape2 = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(30, DataType.INT64, span)]  # Different!

    tensor_type1 = ir.TensorType(shape1, DataType.FP32)
    tensor_type2 = ir.TensorType(shape2, DataType.FP32)

    a = ir.Var("a", tensor_type1, span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [tensor_type1]

    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    iter_arg = ir.IterArg("tensor", tensor_type1, a, span)
    temp = ir.Var("temp", tensor_type2, span)
    assign_temp = ir.AssignStmt(temp, iter_arg, span)  # defined from iter_arg (type mismatch intended)
    body = ir.SeqStmts([assign_temp, ir.YieldStmt([temp], span)], span)
    result_var = ir.Var("result", tensor_type1, span)

    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INT64, span),
        ir.ConstInt(10, DataType.INT64, span),
        ir.ConstInt(1, DataType.INT64, span),
        [iter_arg],
        body,
        [result_var],
        span,
    )

    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_shape_mismatch", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    # Run type checking via run_verifier - should raise on shape mismatch error
    verify_pass = passes.run_verifier()
    with pytest.raises(pypto.Error, match="Shape dimension mismatch in ForStmt"):
        verify_pass(program)


def test_type_check_dimension_count_mismatch():
    """Test type checking detects dimension count mismatch."""
    span = ir.Span.unknown()

    shape1 = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(20, DataType.INT64, span)]
    shape2 = [ir.ConstInt(10, DataType.INT64, span)]  # Only 1 dimension!

    tensor_type1 = ir.TensorType(shape1, DataType.FP32)
    tensor_type2 = ir.TensorType(shape2, DataType.FP32)

    a = ir.Var("a", tensor_type1, span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [tensor_type1]

    condition = ir.Gt(
        ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span
    )
    then_body = ir.YieldStmt([ir.Var("t1", tensor_type1, span)], span)
    else_body = ir.YieldStmt([ir.Var("t2", tensor_type2, span)], span)  # Different dimensions!
    result_var = ir.Var("result", tensor_type1, span)

    if_stmt = ir.IfStmt(condition, then_body, else_body, [result_var], span)

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_dim_mismatch", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    # Run type checking via run_verifier - should raise on dimension mismatch error
    verify_pass = passes.run_verifier()
    with pytest.raises(pypto.Error, match="mismatch in IfStmt"):
        verify_pass(program)


def test_type_check_tile_shape_mismatch():
    """Test type checking detects shape mismatch in TileType."""
    span = ir.Span.unknown()

    shape1 = [ir.ConstInt(16, DataType.INT64, span), ir.ConstInt(16, DataType.INT64, span)]
    shape2 = [ir.ConstInt(16, DataType.INT64, span), ir.ConstInt(32, DataType.INT64, span)]  # Different!

    tile_type1 = ir.TileType(shape1, DataType.FP16)
    tile_type2 = ir.TileType(shape2, DataType.FP16)

    a = ir.Var("a", tile_type1, span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [tile_type1]

    condition = ir.Gt(
        ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span
    )
    then_body = ir.YieldStmt([ir.Var("tile1", tile_type1, span)], span)
    else_body = ir.YieldStmt([ir.Var("tile2", tile_type2, span)], span)  # Different shape!
    result_var = ir.Var("result", tile_type1, span)

    if_stmt = ir.IfStmt(condition, then_body, else_body, [result_var], span)

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function(
        "test_tile_shape_mismatch", params, return_types, func_body, span, ir.FunctionType.InCore
    )
    program = ir.Program([func], "test_program", span)

    # Run type checking via run_verifier - should raise on shape mismatch error
    verify_pass = passes.run_verifier()
    with pytest.raises(pypto.Error, match="mismatch in IfStmt"):
        verify_pass(program)


def test_type_check_valid_types():
    """Test valid types pass type checking."""
    span = ir.Span.unknown()

    shape = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(20, DataType.INT64, span)]
    tensor_type = ir.TensorType(shape, DataType.FP32)

    a = ir.Var("a", tensor_type, span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [tensor_type]

    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    iter_arg = ir.IterArg("tensor", tensor_type, a, span)
    body = ir.YieldStmt([iter_arg], span)
    result_var = ir.Var("result", tensor_type, span)

    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INT64, span),
        ir.ConstInt(10, DataType.INT64, span),
        ir.ConstInt(1, DataType.INT64, span),
        [iter_arg],
        body,
        [result_var],
        span,
    )

    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_valid", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    # Run type checking via run_verifier - should pass without errors
    verify_pass = passes.run_verifier()
    result_program = verify_pass(program)
    assert result_program is not None


def test_type_check_if_condition_must_be_bool():
    """Test type checking rejects non-BOOL IfStmt condition."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = []

    # Condition is INT64 ConstInt — must fail BOOL check
    condition = ir.ConstInt(1, DataType.INT64, span)
    then_body = ir.SeqStmts([], span)
    if_stmt = ir.IfStmt(condition, then_body, None, [], span)

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([], span)], span)
    func = ir.Function("test_if_non_bool", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    verify_pass = passes.run_verifier()
    with pytest.raises(pypto.Error, match="IfStmt condition dtype must be BOOL"):
        verify_pass(program)


def test_type_check_while_condition_must_be_bool():
    """Test type checking rejects non-BOOL WhileStmt condition."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = []

    condition = ir.ConstInt(1, DataType.INDEX, span)
    body = ir.SeqStmts([], span)
    while_stmt = ir.WhileStmt(condition, [], body, [], span)

    func_body = ir.SeqStmts([while_stmt, ir.ReturnStmt([], span)], span)
    func = ir.Function("test_while_non_bool", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    verify_pass = passes.run_verifier()
    with pytest.raises(pypto.Error, match="WhileStmt condition dtype must be BOOL"):
        verify_pass(program)


def test_type_check_bool_condition_passes():
    """Test that BOOL-typed conditions pass verification."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = []

    # BOOL-typed comparison condition
    condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
    then_body = ir.SeqStmts([], span)
    if_stmt = ir.IfStmt(condition, then_body, None, [], span)

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([], span)], span)
    func = ir.Function("test_bool_cond", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    verify_pass = passes.run_verifier()
    result_program = verify_pass(program)
    assert result_program is not None


def test_type_check_rejects_branch_valid_shape_disagreement():
    if_stmt = _if_with_types(_tensor_type(_idx(8)), _tensor_type(_idx(9)), _tensor_type(_idx(8)))

    diagnostics = _typecheck_diagnostics(if_stmt)

    assert len(diagnostics) == 1
    assert "Valid shape dimension mismatch in IfStmt" in diagnostics[0].message
    assert "then yield value[0]" in diagnostics[0].message
    assert "else yield value[0]" in diagnostics[0].message


def test_type_check_rejects_symbolically_unknown_join_without_guard():
    then_extent = _sym("then_extent")
    else_extent = _sym("else_extent")
    if_stmt = _if_with_types(_tensor_type(then_extent), _tensor_type(else_extent), _tensor_type(then_extent))

    diagnostics = _typecheck_diagnostics(if_stmt, params=[then_extent, else_extent])

    assert len(diagnostics) == 1
    assert "equality is not provable" in diagnostics[0].message
    assert "no runtime guard" in diagnostics[0].message


def test_type_check_rejects_if_return_var_metadata_disagreement():
    branch_type = _tensor_type(_idx(8))
    if_stmt = _if_with_types(branch_type, branch_type, _tensor_type(_idx(9)))

    diagnostics = _typecheck_diagnostics(if_stmt)

    assert len(diagnostics) == 1
    assert "then yield value[0]" in diagnostics[0].message
    assert "return_var[0]" in diagnostics[0].message


def test_type_check_rejects_tensor_padding_disagreement():
    then_type = _tensor_type(_idx(8), pad=ir.PadValue.zero)
    else_type = _tensor_type(_idx(8), pad=ir.PadValue.max)
    if_stmt = _if_with_types(then_type, else_type, then_type)

    diagnostics = _typecheck_diagnostics(if_stmt)

    assert len(diagnostics) == 1
    assert diagnostics[0].error_code == passes.TypeCheckErrorType.TENSOR_PADDING_MISMATCH.value
    assert "Tensor padding mismatch in IfStmt" in diagnostics[0].message


def test_type_check_rejects_distributed_window_buffer_identity_disagreement():
    base_a = ir.Var("buffer_a", ir.PtrType(), _SPAN)
    base_b = ir.Var("buffer_b", ir.PtrType(), _SPAN)
    window_a = ir.WindowBuffer(base_a, _idx(64), span=_SPAN)
    window_b = ir.WindowBuffer(base_b, _idx(64), span=_SPAN)
    type_a = ir.DistributedTensorType([_idx(16)], DataType.FP32, window_a)
    type_b = ir.DistributedTensorType([_idx(16)], DataType.FP32, window_b)
    if_stmt = _if_with_types(type_a, type_b, type_a)

    diagnostics = _typecheck_diagnostics(if_stmt)

    assert len(diagnostics) == 1
    assert diagnostics[0].error_code == passes.TypeCheckErrorType.DISTRIBUTED_WINDOW_IDENTITY_MISMATCH.value
    assert "Distributed window-buffer identity mismatch in IfStmt" in diagnostics[0].message


def test_type_check_rejects_effective_tile_view_metadata_disagreement():
    view_a = ir.TileView(valid_shape=[_idx(16), _idx(16)], stride=[_idx(16), _idx(1)])
    view_b = ir.TileView(valid_shape=[_idx(16), _idx(16)], stride=[_idx(1), _idx(16)])
    type_a = ir.TileType([_idx(16), _idx(16)], DataType.FP32, None, view_a, ir.MemorySpace.Vec)
    type_b = ir.TileType([_idx(16), _idx(16)], DataType.FP32, None, view_b, ir.MemorySpace.Vec)
    if_stmt = _if_with_types(type_a, type_b, type_a)

    diagnostics = _typecheck_diagnostics(if_stmt)

    assert len(diagnostics) == 1
    assert diagnostics[0].error_code == passes.TypeCheckErrorType.TILE_VIEW_MISMATCH.value
    assert "Tile view metadata mismatch in IfStmt" in diagnostics[0].message


@pytest.mark.parametrize("loop_kind", ["for", "while"])
@pytest.mark.parametrize(
    ("init_valid", "declared_valid", "yield_valid", "return_valid", "expected_carriers"),
    [
        (8, 9, 9, 9, ("initValue", "declared iter_arg[0]")),
        (8, 8, 9, 9, ("declared iter_arg[0]", "yield value[0]")),
        (8, 8, 8, 9, ("yield value[0]", "return_var[0]")),
    ],
)
def test_type_check_rejects_loop_carrier_metadata_disagreement(
    loop_kind: str,
    init_valid: int,
    declared_valid: int,
    yield_valid: int,
    return_valid: int,
    expected_carriers: tuple[str, str],
):
    init_value = ir.Var("init", _tensor_type(_idx(init_valid)), _SPAN)
    iter_arg = ir.IterArg("carry", _tensor_type(_idx(declared_valid)), init_value, _SPAN)
    yield_value = ir.Var("yield_value", _tensor_type(_idx(yield_valid)), _SPAN)
    body = ir.YieldStmt([yield_value], _SPAN)
    return_var = ir.Var("result", _tensor_type(_idx(return_valid)), _SPAN)
    if loop_kind == "for":
        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), _SPAN)
        stmt = ir.ForStmt(
            loop_var,
            _idx(0),
            _idx(1),
            _idx(1),
            [iter_arg],
            body,
            [return_var],
            _SPAN,
        )
    else:
        stmt = ir.WhileStmt(ir.ConstInt(1, DataType.BOOL, _SPAN), [iter_arg], body, [return_var], _SPAN)

    diagnostics = _typecheck_diagnostics(stmt, params=[init_value])

    assert len(diagnostics) == 1
    assert expected_carriers[0] in diagnostics[0].message
    assert expected_carriers[1] in diagnostics[0].message


def _buffer_boundary_type(container: str, **changes: Any) -> ir.Type:
    fields: dict[str, Any] = {"shape": [8, 16], "dtype": DataType.FP32, "memory_space": ir.Mem.Vec}
    slot_count = changes.pop("slot_count", 2)
    fields.update(changes)
    buffer = ir.BufferType(**fields)
    descriptor: ir.Type = buffer
    if container != "buffer":
        descriptor = ir.MultiBufferType(buffer, slot_count)
    if container == "tuple":
        descriptor = ir.TupleType([ir.ScalarType(DataType.INT32), ir.TupleType([descriptor])])
    return descriptor


def _buffer_boundary(kind: str, types: list[ir.Type]) -> ir.Stmt:
    if kind == "if":
        return _if_with_types(*types)
    initial = ir.Var("initial", types[0], _SPAN)
    carry = ir.IterArg("carry", types[1], initial, _SPAN)
    body = ir.YieldStmt([ir.Var("yielded", types[2], _SPAN)], _SPAN)
    result = ir.Var("result", types[3], _SPAN)
    if kind == "while":
        return ir.WhileStmt(ir.ConstBool(True, _SPAN), [carry], body, [result], _SPAN)
    return ir.ForStmt(_sym("i"), _idx(0), _idx(2), _idx(1), [carry], body, [result], _SPAN)


@pytest.mark.parametrize("kind", ["if", "for", "while"])
@pytest.mark.parametrize("container", ["buffer", "multi", "tuple"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("shape", [8, 32]),
        ("dtype", DataType.FP16),
        ("memory_space", ir.Mem.Mat),
        ("valid_shape", [-1, 16]),
        ("blayout", ir.TileLayout.col_major),
        ("slayout", ir.TileLayout.row_major),
        ("fractal", 1024),
        ("pad", ir.PadValue.zero),
        ("compact", ir.CompactMode.normal),
    ],
)
def test_buffer_descriptor_fields_must_match_at_control_flow(kind, container, field, value):
    """Each physical field remains part of the contract, including through nested tuples."""
    types = [_buffer_boundary_type(container) for _ in range(3 if kind == "if" else 4)]
    types[1 if kind == "if" else 2] = _buffer_boundary_type(container, **{field: value})
    diagnostics = _typecheck_diagnostics(_buffer_boundary(kind, types))
    assert diagnostics
    assert all(
        d.error_code == passes.TypeCheckErrorType.BUFFER_DESCRIPTOR_MISMATCH.value for d in diagnostics
    )
    assert all(f"Buffer descriptor mismatch in {kind.title()}Stmt" in d.message for d in diagnostics)
    assert all("BufferType" in d.message for d in diagnostics)


@pytest.mark.parametrize("kind", ["if", "for", "while"])
@pytest.mark.parametrize("container", ["multi", "tuple"])
def test_multi_buffer_slot_counts_must_match_at_control_flow(kind, container):
    """Equal element descriptors do not make different slot counts compatible."""
    types = [_buffer_boundary_type(container) for _ in range(3 if kind == "if" else 4)]
    types[1 if kind == "if" else 2] = _buffer_boundary_type(container, slot_count=3)
    diagnostics = _typecheck_diagnostics(_buffer_boundary(kind, types))
    assert diagnostics
    assert all(
        d.error_code == passes.TypeCheckErrorType.BUFFER_DESCRIPTOR_MISMATCH.value for d in diagnostics
    )
    assert all("MultiBufferType" in d.message for d in diagnostics)


@pytest.mark.parametrize(
    "kind,position",
    [(kind, i) for kind, count in (("if", 3), ("for", 4), ("while", 4)) for i in range(count)],
)
@pytest.mark.parametrize("container", ["buffer", "multi"])
def test_buffer_contract_checks_every_control_flow_binding(kind, position, container):
    """Check initial values, declared carries, yields and result declarations individually."""
    types = [_buffer_boundary_type(container) for _ in range(3 if kind == "if" else 4)]
    types[position] = _buffer_boundary_type(container, dtype=DataType.FP16)
    diagnostics = _typecheck_diagnostics(_buffer_boundary(kind, types))
    assert diagnostics
    assert all(
        d.error_code == passes.TypeCheckErrorType.BUFFER_DESCRIPTOR_MISMATCH.value for d in diagnostics
    )


@pytest.mark.parametrize("kind", ["if", "for", "while"])
@pytest.mark.parametrize("container", ["buffer", "multi", "tuple"])
@pytest.mark.parametrize("valid_shape", [[8, 16], [-1, 16]])
def test_equal_buffer_descriptors_from_distinct_objects_are_compatible(kind, container, valid_shape):
    """Compare descriptor values rather than requiring the same type object."""
    types = [
        _buffer_boundary_type(container, valid_shape=valid_shape) for _ in range(3 if kind == "if" else 4)
    ]
    assert _typecheck_diagnostics(_buffer_boundary(kind, types)) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
