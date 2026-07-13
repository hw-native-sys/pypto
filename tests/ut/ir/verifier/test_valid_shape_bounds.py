# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the standing ``valid_shape <= shape`` invariant .

The invariant is enforced by the ``TypeChecked`` property verifier (the
``TypeCheck`` rule): for any TileType/TensorType carrying an explicit,
non-empty ``valid_shape``,

- ``rank(valid_shape) == rank(shape)`` (or ``valid_shape`` is empty), and
- ``0 <= valid_shape[i] <= shape[i]`` for every dim where **both** the valid
  extent and the physical extent are compile-time constants.

Symbolic (dynamic) valid extents are a supported feature and are skipped, not
rejected. The invariant covers both TileView and TensorView, and is placed in
the always-on ``TypeChecked`` structural property so a malformed ``valid_shape``
is caught before/after every pass rather than only on demand.
"""

import pytest
from pypto import DataType, ir
from pypto.ir.op import tile as tile_ops
from pypto.pypto_core import passes as _passes


def _span():
    return ir.Span.unknown()


def _const(value: int):
    return ir.ConstInt(value, DataType.INDEX, _span())


def _program_with_param_type(t) -> ir.Program:
    """Build a 1-function program whose single parameter has the given type."""
    span = _span()
    var = ir.Var("x", t, span)
    func = ir.Function("f", [var], [], ir.EvalStmt(_const(0), span), span)
    return ir.Program([func], "p", span)


def _verify(program: ir.Program) -> list:
    """Run the TypeChecked verifier and return its diagnostics."""
    props = _passes.IRPropertySet()
    props.insert(_passes.IRProperty.TypeChecked)
    return _passes.PropertyVerifierRegistry.verify(props, program)


def _tile(shape: list[int], valid: list) -> ir.TileType:
    """TileType with an explicit valid_shape (ints stay ConstInt; Var passes through)."""
    valid_exprs = [_const(v) if isinstance(v, int) else v for v in valid]
    view = ir.TileView(valid_shape=valid_exprs)
    return ir.TileType([_const(s) for s in shape], DataType.FP32, None, view, ir.MemorySpace.Vec)


def _tensor(shape: list[int], valid: list, pad=ir.PadValue.null) -> ir.TensorType:
    """TensorType with an explicit valid_shape (layout required alongside valid_shape)."""
    valid_exprs = [_const(v) if isinstance(v, int) else v for v in valid]
    view = ir.TensorView(valid_shape=valid_exprs, layout=ir.TensorLayout.ND, pad=pad)
    return ir.TensorType([_const(s) for s in shape], DataType.FP32, None, view)


def _distributed_tensor(shape: list[int], valid: list) -> ir.DistributedTensorType:
    """DistributedTensorType carrying the same TensorView validity contract."""
    valid_exprs = [_const(v) if isinstance(v, int) else v for v in valid]
    view = ir.TensorView(valid_shape=valid_exprs, layout=ir.TensorLayout.ND)
    return ir.DistributedTensorType([_const(s) for s in shape], DataType.FP32, None, view)


def _program_with_stmt(stmt: ir.Stmt, params: list[ir.Var]) -> ir.Program:
    span = _span()
    body = ir.SeqStmts([stmt], span)
    func = ir.Function("f", params, [], body, span)
    return ir.Program([func], "p", span)


# ============================================================================
# TileType — the [999, 999] / [128, 128] scenario and its neighbours
# ============================================================================


def test_tile_valid_shape_exceeds_shape_rejected():
    """valid_shape [999, 999] on shape [128, 128] is rejected (B3)."""
    diags = _verify(_program_with_param_type(_tile([128, 128], [999, 999])))
    assert len(diags) >= 1
    assert any("out of bounds" in d.message for d in diags), [d.message for d in diags]
    assert any(d.rule_name == "TypeCheck" for d in diags)


def test_tile_narrower_valid_shape_accepted():
    """A narrower static valid_shape [64, 128] on shape [128, 128] is accepted."""
    assert _verify(_program_with_param_type(_tile([128, 128], [64, 128]))) == []


def test_tile_zero_valid_shape_accepted():
    """valid_shape [0, 0] (the fresh-accumulator sentinel) is accepted: 0 <= 0 <= shape."""
    assert _verify(_program_with_param_type(_tile([128, 128], [0, 0]))) == []


def test_tile_dynamic_valid_shape_accepted():
    """A dynamic (symbolic Var) valid extent is skipped, not rejected."""
    m = ir.Var("m", ir.ScalarType(DataType.INDEX), _span())
    # Even a symbolic value that could exceed the shape at runtime is accepted —
    # the bound is only checked for compile-time-constant dims.
    assert _verify(_program_with_param_type(_tile([128, 128], [m, 128]))) == []


def test_tile_rank_mismatch_rejected():
    """rank(valid_shape) != rank(shape) is rejected."""
    diags = _verify(_program_with_param_type(_tile([128, 128], [64])))
    assert len(diags) >= 1
    assert any("valid_shape rank" in d.message and "does not match shape rank" in d.message for d in diags)


# ============================================================================
# TensorType — same invariant, verifying the check covers both view kinds
# ============================================================================


def test_tensor_valid_shape_exceeds_shape_rejected():
    """valid_shape [999, 999] on a [128, 128] TensorType is rejected."""
    diags = _verify(_program_with_param_type(_tensor([128, 128], [999, 999])))
    assert len(diags) >= 1
    assert any("out of bounds" in d.message and "TensorType" in d.message for d in diags)


def test_tensor_narrower_valid_shape_accepted():
    """A narrower static valid_shape on a TensorType is accepted."""
    assert _verify(_program_with_param_type(_tensor([128, 128], [64, 128]))) == []


def test_tensor_dynamic_valid_shape_accepted():
    """A dynamic valid extent on a TensorType is skipped, not rejected."""
    n = ir.Var("n", ir.ScalarType(DataType.INDEX), _span())
    assert _verify(_program_with_param_type(_tensor([128, 128], [n, n]))) == []


def test_distributed_tensor_dynamic_valid_shape_accepted():
    """Ordinary validation preserves symbolic validity on distributed tensors."""
    n = ir.Var("n", ir.ScalarType(DataType.INDEX), _span())
    assert _verify(_program_with_param_type(_distributed_tensor([128, 128], [n, 128]))) == []


# ============================================================================
# Control-flow joins — effective valid_shape is part of the shaped type
# ============================================================================


def test_if_join_compares_distributed_return_var_valid_shape():
    """If yields must also agree with the declared distributed return variable."""
    span = _span()
    m = ir.Var("m", ir.ScalarType(DataType.INDEX), span)
    n = ir.Var("n", ir.ScalarType(DataType.INDEX), span)
    branch_type = _distributed_tensor([128, 128], [m, 128])
    return_type = _distributed_tensor([128, 128], [n, 128])
    then_value = ir.Var("then_value", branch_type, span)
    else_value = ir.Var("else_value", branch_type, span)
    return_var = ir.Var("result", return_type, span)
    condition = ir.Gt(m, _const(0), DataType.BOOL, span)
    stmt = ir.IfStmt(
        condition,
        ir.YieldStmt([then_value], span),
        ir.YieldStmt([else_value], span),
        [return_var],
        span,
    )

    diags = _verify(_program_with_stmt(stmt, [m, n, then_value, else_value]))
    assert any("Effective valid_shape mismatch in IfStmt" in d.message for d in diags)
    assert any(
        "return_var[0]" in d.message and "symbolic equality cannot be proven" in d.message for d in diags
    )


def test_if_join_rejects_tensor_pad_mismatch_for_same_partial_valid_shape():
    """Equal partial extents cannot hide different values in the invalid region."""
    span = _span()
    then_type = _tensor([128, 128], [64, 128], pad=ir.PadValue.zero)
    else_type = _tensor([128, 128], [64, 128], pad=ir.PadValue.min)
    then_value = ir.Var("then_value", then_type, span)
    else_value = ir.Var("else_value", else_type, span)
    return_var = ir.Var("result", then_type, span)
    condition = ir.Gt(_const(1), _const(0), DataType.BOOL, span)
    stmt = ir.IfStmt(
        condition,
        ir.YieldStmt([then_value], span),
        ir.YieldStmt([else_value], span),
        [return_var],
        span,
    )

    diags = _verify(_program_with_stmt(stmt, [then_value, else_value]))
    assert any(
        "TensorView pad mismatch in IfStmt" in d.message
        and "then yield value[0]" in d.message
        and "else yield value[0]" in d.message
        for d in diags
    )


def test_for_join_rejects_unprovable_symbolic_valid_shape_change():
    """Loop-carried validity must be invariant, not merely shape/dtype compatible."""
    span = _span()
    m = ir.Var("m", ir.ScalarType(DataType.INDEX), span)
    n = ir.Var("n", ir.ScalarType(DataType.INDEX), span)
    init = ir.Var("init", _tensor([128, 128], [m, 128]), span)
    yielded = ir.Var("yielded", _tensor([128, 128], [n, 128]), span)
    iter_arg = ir.IterArg("carry", init.type, init, span)
    return_var = ir.Var("result", init.type, span)
    loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
    stmt = ir.ForStmt(
        loop_var,
        _const(0),
        _const(4),
        _const(1),
        [iter_arg],
        ir.YieldStmt([yielded], span),
        [return_var],
        span,
    )

    diags = _verify(_program_with_stmt(stmt, [m, n, init, yielded]))
    assert any(
        "Effective valid_shape mismatch in ForStmt" in d.message
        and "symbolic equality cannot be proven" in d.message
        for d in diags
    )


def test_for_join_rejects_iter_arg_declared_full_when_carriers_are_partial():
    """The body-visible IterArg type cannot widen partial loop carriers to full."""
    span = _span()
    partial_type = _tensor([128, 128], [64, 128])
    full_type = ir.TensorType([_const(128), _const(128)], DataType.FP32)
    init = ir.Var("init", partial_type, span)
    yielded = ir.Var("yielded", partial_type, span)
    iter_arg = ir.IterArg("carry", full_type, init, span)
    return_var = ir.Var("result", partial_type, span)
    loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
    stmt = ir.ForStmt(
        loop_var,
        _const(0),
        _const(4),
        _const(1),
        [iter_arg],
        ir.YieldStmt([yielded], span),
        [return_var],
        span,
    )

    diags = _verify(_program_with_stmt(stmt, [init, yielded]))
    assert any(
        "Effective valid_shape mismatch in ForStmt" in d.message and "iter_arg[0] declared type" in d.message
        for d in diags
    )


def test_while_join_rejects_unprovable_symbolic_valid_shape_change():
    """While loop-carried Tile validity obeys the same strict join invariant."""
    span = _span()
    m = ir.Var("m", ir.ScalarType(DataType.INDEX), span)
    n = ir.Var("n", ir.ScalarType(DataType.INDEX), span)
    init = ir.Var("init", _tile([128, 128], [m, 128]), span)
    yielded = ir.Var("yielded", _tile([128, 128], [n, 128]), span)
    iter_arg = ir.IterArg("carry", init.type, init, span)
    return_var = ir.Var("result", init.type, span)
    condition = ir.Gt(m, _const(0), DataType.BOOL, span)
    stmt = ir.WhileStmt(condition, [iter_arg], ir.YieldStmt([yielded], span), [return_var], span)

    diags = _verify(_program_with_stmt(stmt, [m, n, init, yielded]))
    assert any(
        "Effective valid_shape mismatch in WhileStmt" in d.message
        and "symbolic equality cannot be proven" in d.message
        for d in diags
    )


def test_while_join_rejects_iter_arg_declared_partial_when_carriers_are_full():
    """The body-visible IterArg type cannot narrow fully-valid loop carriers."""
    span = _span()
    partial_type = _tile([128, 128], [64, 128])
    full_type = ir.TileType([_const(128), _const(128)], DataType.FP32, None, None, ir.MemorySpace.Vec)
    init = ir.Var("init", full_type, span)
    yielded = ir.Var("yielded", full_type, span)
    iter_arg = ir.IterArg("carry", partial_type, init, span)
    return_var = ir.Var("result", full_type, span)
    condition = ir.Gt(_const(1), _const(0), DataType.BOOL, span)
    stmt = ir.WhileStmt(condition, [iter_arg], ir.YieldStmt([yielded], span), [return_var], span)

    diags = _verify(_program_with_stmt(stmt, [init, yielded]))
    assert any(
        "Effective valid_shape mismatch in WhileStmt" in d.message
        and "iter_arg[0] declared type" in d.message
        for d in diags
    )


def test_if_join_accepts_same_symbolic_valid_extent_expression():
    """Strict joins still accept the exact shared runtime extent expression."""
    span = _span()
    m = ir.Var("m", ir.ScalarType(DataType.INDEX), span)
    value_type = _tensor([128, 128], [m, 128])
    then_value = ir.Var("then_value", value_type, span)
    else_value = ir.Var("else_value", value_type, span)
    return_var = ir.Var("result", value_type, span)
    condition = ir.Gt(m, _const(0), DataType.BOOL, span)
    stmt = ir.IfStmt(
        condition,
        ir.YieldStmt([then_value], span),
        ir.YieldStmt([else_value], span),
        [return_var],
        span,
    )

    assert _verify(_program_with_stmt(stmt, [m, then_value, else_value])) == []


def test_if_join_rejects_partial_valid_shape_mismatch_inside_tuple():
    """Tuple carriers cannot hide a partial-valid shaped element mismatch."""
    span = _span()
    then_type = ir.TupleType([_tensor([128, 128], [64, 128])])
    else_type = ir.TupleType([_tensor([128, 128], [32, 128])])
    then_value = ir.Var("then_value", then_type, span)
    else_value = ir.Var("else_value", else_type, span)
    return_var = ir.Var("result", then_type, span)
    condition = ir.Gt(_const(1), _const(0), DataType.BOOL, span)
    stmt = ir.IfStmt(
        condition,
        ir.YieldStmt([then_value], span),
        ir.YieldStmt([else_value], span),
        [return_var],
        span,
    )

    diags = _verify(_program_with_stmt(stmt, [then_value, else_value]))
    assert any(
        "Effective valid_shape mismatch in IfStmt" in d.message
        and "then yield value[0] element[0]" in d.message
        and "else yield value[0] element[0]" in d.message
        for d in diags
    )


def test_if_join_rejects_tuple_arity_mismatch():
    """Tuple carriers must agree on their number of element types."""
    span = _span()
    tensor_type = _tensor([128, 128], [64, 128])
    then_type = ir.TupleType([tensor_type])
    else_type = ir.TupleType([tensor_type, ir.ScalarType(DataType.INDEX)])
    then_value = ir.Var("then_value", then_type, span)
    else_value = ir.Var("else_value", else_type, span)
    return_var = ir.Var("result", then_type, span)
    condition = ir.Gt(_const(1), _const(0), DataType.BOOL, span)
    stmt = ir.IfStmt(
        condition,
        ir.YieldStmt([then_value], span),
        ir.YieldStmt([else_value], span),
        [return_var],
        span,
    )

    diags = _verify(_program_with_stmt(stmt, [then_value, else_value]))
    assert any(
        "Tuple arity mismatch in IfStmt" in d.message
        and "then yield value[0]" in d.message
        and "else yield value[0]" in d.message
        for d in diags
    )


def test_if_join_accepts_distributed_tensors_from_same_window_buffer():
    """Distinct DTT objects may join when they reference the same allocation."""
    span = _span()
    base = ir.Var("buf", ir.PtrType(), span)
    window = ir.WindowBuffer(base, _const(4096), span=span)
    then_type = ir.DistributedTensorType([_const(128)], DataType.FP32, window)
    else_type = ir.DistributedTensorType([_const(128)], DataType.FP32, window)
    then_value = ir.Var("then_value", then_type, span)
    else_value = ir.Var("else_value", else_type, span)
    return_var = ir.Var("result", then_type, span)
    condition = ir.Gt(_const(1), _const(0), DataType.BOOL, span)
    stmt = ir.IfStmt(
        condition,
        ir.YieldStmt([then_value], span),
        ir.YieldStmt([else_value], span),
        [return_var],
        span,
    )

    assert _verify(_program_with_stmt(stmt, [then_value, else_value])) == []


def test_if_join_rejects_distributed_tensor_window_buffer_presence_mismatch():
    """A bound DTT cannot join an annotation-only DTT with no allocation."""
    span = _span()
    base = ir.Var("buf", ir.PtrType(), span)
    window = ir.WindowBuffer(base, _const(4096), span=span)
    then_type = ir.DistributedTensorType([_const(128)], DataType.FP32, window)
    else_type = ir.DistributedTensorType([_const(128)], DataType.FP32)
    then_value = ir.Var("then_value", then_type, span)
    else_value = ir.Var("else_value", else_type, span)
    return_var = ir.Var("result", then_type, span)
    condition = ir.Gt(_const(1), _const(0), DataType.BOOL, span)
    stmt = ir.IfStmt(
        condition,
        ir.YieldStmt([then_value], span),
        ir.YieldStmt([else_value], span),
        [return_var],
        span,
    )

    diags = _verify(_program_with_stmt(stmt, [then_value, else_value]))
    assert any(
        "DistributedTensorType window_buffer presence mismatch in IfStmt" in d.message
        and "then yield value[0]" in d.message
        and "else yield value[0]" in d.message
        for d in diags
    )


def test_if_join_rejects_distributed_tensors_from_different_window_buffers():
    """Same-shape DTTs backed by different allocations cannot share a join."""
    span = _span()
    base_a = ir.Var("buf_a", ir.PtrType(), span)
    base_b = ir.Var("buf_b", ir.PtrType(), span)
    window_a = ir.WindowBuffer(base_a, _const(4096), span=span)
    window_b = ir.WindowBuffer(base_b, _const(4096), span=span)
    then_type = ir.DistributedTensorType([_const(128)], DataType.FP32, window_a)
    else_type = ir.DistributedTensorType([_const(128)], DataType.FP32, window_b)
    then_value = ir.Var("then_value", then_type, span)
    else_value = ir.Var("else_value", else_type, span)
    return_var = ir.Var("result", then_type, span)
    condition = ir.Gt(_const(1), _const(0), DataType.BOOL, span)
    stmt = ir.IfStmt(
        condition,
        ir.YieldStmt([then_value], span),
        ir.YieldStmt([else_value], span),
        [return_var],
        span,
    )

    diags = _verify(_program_with_stmt(stmt, [then_value, else_value]))
    assert any(
        "DistributedTensorType window_buffer identity mismatch in IfStmt" in d.message
        and "then yield value[0]" in d.message
        and "else yield value[0]" in d.message
        for d in diags
    )


# ============================================================================
# Bare types (no explicit valid_shape) — fully valid, nothing to check
# ============================================================================


def test_bare_tile_accepted():
    """A TileType without a valid_shape is fully valid and passes."""
    t = ir.TileType([_const(128), _const(128)], DataType.FP32, None, None, ir.MemorySpace.Vec)
    assert _verify(_program_with_param_type(t)) == []


def test_bare_tensor_accepted():
    """A bare TensorType (no view) is fully valid and passes."""
    t = ir.TensorType([_const(128), _const(128)], DataType.FP32)
    assert _verify(_program_with_param_type(t)) == []


# ============================================================================
# Op-deduction path — malformed validity never escapes into a body Var
# ============================================================================


def test_load_invalid_valid_shape_rejected_during_op_deduction():
    """An out-of-bounds ``valid_shapes=`` on ``tile.load`` fails immediately."""
    span = _span()
    x = ir.Var("x", ir.TensorType([_const(128), _const(128)], DataType.FP32), span)
    with pytest.raises(ValueError, match=r"valid_shape\[0\].*exceeds physical shape"):
        tile_ops.load(x, [0, 0], [128, 128], valid_shapes=[999, 999], span=span)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
