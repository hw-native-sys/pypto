# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for PyTorch code emission from PyPTO IR."""

import pytest
import torch
from pypto import DataType, ir
from pypto.debug import torch_codegen
from pypto.debug.torch_codegen import TorchCodegen

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_span = ir.Span.unknown


def _scalar(name: str, dtype: DataType = DataType.INT64) -> ir.Var:
    return ir.Var(name, ir.ScalarType(dtype), _span())


def _tensor_var(name: str, shape: list[int], dtype: DataType = DataType.FP32) -> ir.Var:
    return ir.Var(name, ir.TensorType(shape, dtype), _span())


def _tile_var(name: str, shape: list[int], dtype: DataType = DataType.FP32) -> ir.Var:
    return ir.Var(name, ir.TileType(shape, dtype), _span())


def _int(val: int) -> ir.ConstInt:
    return ir.ConstInt(val, DataType.INT64, _span())


def _float(val: float) -> ir.ConstFloat:
    return ir.ConstFloat(val, DataType.FP32, _span())


def _make_tuple(*exprs: ir.Expr) -> ir.MakeTuple:
    return ir.MakeTuple(list(exprs), _span())


def _op_call(op_name: str, args: list[ir.Expr], kwargs: dict | None = None) -> ir.Call:
    if kwargs:
        return ir.create_op_call(op_name, args, kwargs, _span())
    return ir.create_op_call(op_name, args, _span())


def _simple_function(
    name: str, params: list[ir.Var], body: ir.Stmt, return_types: list[ir.Type] | None = None
) -> ir.Function:
    return ir.Function(name, params, return_types or [], body, _span())


def _program(funcs: list[ir.Function]) -> ir.Program:
    return ir.Program(funcs, "test_program", _span())


# ---------------------------------------------------------------------------
# Test: basic tensor ops
# ---------------------------------------------------------------------------


def test_tensor_add():
    """tensor.add should emit torch.add(a, b)."""
    a = _tensor_var("a", [64, 128])
    b = _tensor_var("b", [64, 128])
    c = _tensor_var("c", [64, 128])
    call = _op_call("tensor.add", [a, b])
    assign = ir.AssignStmt(c, call, _span())
    func = _simple_function("main", [a, b], assign)
    code = torch_codegen(func)
    assert "torch.add(a, b)" in code


def test_tensor_scalar_add():
    """tensor.adds should emit (a + scalar)."""
    a = _tensor_var("a", [64])
    c = _tensor_var("c", [64])
    scalar = _float(1.0)
    call = _op_call("tensor.adds", [a, scalar])
    assign = ir.AssignStmt(c, call, _span())
    func = _simple_function("main", [a], assign)
    code = torch_codegen(func)
    assert "(a + 1.0)" in code


def test_tensor_unary_ops():
    """Unary tensor ops should emit correct torch functions."""
    a = _tensor_var("a", [64])
    for op_name, expected in [
        ("tensor.exp", "torch.exp(a)"),
        ("tensor.neg", "torch.neg(a)"),
        ("tensor.sqrt", "torch.sqrt(a)"),
        ("tensor.rsqrt", "torch.rsqrt(a)"),
        ("tensor.recip", "torch.reciprocal(a)"),
    ]:
        out = _tensor_var("out", [64])
        call = _op_call(op_name, [a])
        assign = ir.AssignStmt(out, call, _span())
        func = _simple_function("f", [a], assign)
        code = torch_codegen(func)
        assert expected in code, f"{op_name}: expected '{expected}' in output"


def test_tensor_matmul_with_transpose():
    """tensor.matmul with a_trans/b_trans should emit .mT."""
    a = _tensor_var("a", [64, 128])
    b = _tensor_var("b", [128, 64])
    out = _tensor_var("out", [64, 64])
    call = _op_call("tensor.matmul", [a, b], {"a_trans": True, "b_trans": False, "c_matrix_nz": False})
    assign = ir.AssignStmt(out, call, _span())
    func = _simple_function("f", [a, b], assign)
    code = torch_codegen(func)
    assert "a.mT" in code
    assert "torch.matmul" in code


def test_tensor_cast():
    """tensor.cast should emit .to(dtype)."""
    a = _tensor_var("a", [64])
    out = _tensor_var("out", [64])
    call = _op_call("tensor.cast", [a], {"target_type": DataType.FP16, "mode": 0})
    assign = ir.AssignStmt(out, call, _span())
    func = _simple_function("f", [a], assign)
    code = torch_codegen(func)
    assert ".to(torch.float16)" in code


def test_tensor_row_reduction():
    """tensor.row_sum/max/min should emit appropriate reductions."""
    a = _tensor_var("a", [64, 128])
    out = _tensor_var("out", [64, 1])
    call = _op_call("tensor.row_sum", [a])
    assign = ir.AssignStmt(out, call, _span())
    func = _simple_function("f", [a], assign)
    code = torch_codegen(func)
    assert ".sum(dim=-1, keepdim=True)" in code


# ---------------------------------------------------------------------------
# Test: tile ops
# ---------------------------------------------------------------------------


def test_tile_load_store():
    """tile.load and tile.store should emit _tile_load/_tile_store helpers."""
    tensor = _tensor_var("t", [256, 256])
    offsets = _make_tuple(_int(0), _int(0))
    shapes = _make_tuple(_int(64), _int(64))
    valid_shapes = _make_tuple(_int(64), _int(64))
    tile = _tile_var("tile", [64, 64])
    output = _tensor_var("out", [256, 256])
    off2 = _make_tuple(_int(64), _int(0))

    load_call = _op_call(
        "tile.load",
        [tensor, offsets, shapes, valid_shapes],
        {"target_memory": ir.MemorySpace.Vec, "transpose": False},
    )
    store_call = _op_call("tile.store", [tile, off2, output])

    body = ir.SeqStmts(
        [
            ir.AssignStmt(tile, load_call, _span()),
            ir.EvalStmt(store_call, _span()),
        ],
        _span(),
    )
    func = _simple_function("f", [tensor, output], body)
    code = torch_codegen(func)
    assert "_tile_load" in code
    assert "_tile_store" in code


def test_tile_load_transpose():
    """tile.load with transpose=True should append .mT."""
    tensor = _tensor_var("t", [64, 64])
    offsets = _make_tuple(_int(0), _int(0))
    shapes = _make_tuple(_int(32), _int(32))
    valid_shapes = _make_tuple(_int(32), _int(32))
    tile = _tile_var("tile", [32, 32])

    call = _op_call(
        "tile.load",
        [tensor, offsets, shapes, valid_shapes],
        {"target_memory": ir.MemorySpace.Mat, "transpose": True},
    )
    assign = ir.AssignStmt(tile, call, _span())
    func = _simple_function("f", [tensor], assign)
    code = torch_codegen(func)
    assert ".mT" in code


def test_tile_compute_ops():
    """Tile compute ops should emit torch equivalents."""
    a = _tile_var("a", [64, 64])
    b = _tile_var("b", [64, 64])
    out = _tile_var("out", [64, 64])

    call = _op_call("tile.add", [a, b])
    assign = ir.AssignStmt(out, call, _span())
    func = _simple_function("f", [a, b], assign)
    code = torch_codegen(func)
    assert "torch.add(a, b)" in code


def test_tile_matmul_acc():
    """tile.matmul_acc should emit (acc + torch.matmul(lhs, rhs))."""
    acc = _tile_var("acc", [64, 64])
    a = _tile_var("a", [64, 128])
    b = _tile_var("b", [128, 64])
    out = _tile_var("out", [64, 64])

    call = _op_call("tile.matmul_acc", [acc, a, b])
    assign = ir.AssignStmt(out, call, _span())
    func = _simple_function("f", [acc, a, b], assign)
    code = torch_codegen(func)
    assert "acc + torch.matmul(a, b).float()" in code


def test_tile_cmp():
    """tile.cmp should emit correct comparison operator."""
    a = _tile_var("a", [64])
    b = _tile_var("b", [64])
    out = _tile_var("mask", [64])

    # cmp_type=2 is LT
    call = _op_call("tile.cmp", [a, b], {"cmp_type": 2})
    assign = ir.AssignStmt(out, call, _span())
    func = _simple_function("f", [a, b], assign)
    code = torch_codegen(func)
    assert "(a < b)" in code


def test_tile_reduction_with_axis():
    """tile.sum with axis kwarg should emit .sum(dim=axis)."""
    a = _tile_var("a", [64, 128])
    out = _tile_var("out", [64, 1])

    call = _op_call("tile.sum", [a], {"axis": -1, "keepdim": True})
    assign = ir.AssignStmt(out, call, _span())
    func = _simple_function("f", [a], assign)
    code = torch_codegen(func)
    assert ".sum(dim=-1, keepdim=True)" in code


def test_tile_get_block_idx():
    """tile.get_block_idx should emit 0."""
    out = _scalar("idx", DataType.UINT64)
    call = _op_call("tile.get_block_idx", [])
    assign = ir.AssignStmt(out, call, _span())
    func = _simple_function("f", [], assign)
    code = torch_codegen(func)
    assert "idx = 0" in code


# ---------------------------------------------------------------------------
# Test: SSA for loop conversion
# ---------------------------------------------------------------------------


def test_for_loop_with_iter_args():
    """ForStmt with iter_args should be converted to imperative mutable pattern."""
    i = _scalar("i")
    init_val = _float(0.0)
    acc_type = ir.ScalarType(DataType.FP32)
    acc = ir.IterArg("acc", acc_type, init_val, _span())
    x = _scalar("x", DataType.FP32)

    # Body: new_acc = acc + x; yield new_acc
    new_acc = _scalar("new_acc", DataType.FP32)
    add_expr = ir.Add(acc, x, DataType.FP32, _span())
    body = ir.SeqStmts(
        [
            ir.AssignStmt(new_acc, add_expr, _span()),
            ir.YieldStmt([new_acc], _span()),
        ],
        _span(),
    )

    result = _scalar("result", DataType.FP32)
    for_stmt = ir.ForStmt(i, _int(0), _int(10), _int(1), [acc], body, [result], _span())

    func = _simple_function("f", [x], for_stmt)
    code = torch_codegen(func)

    # Should have: acc = 0.0 (init), for i in range(0, 10, 1):, acc = ... (yield)
    assert "acc = 0.0" in code
    assert "for i in range(0, 10, 1):" in code
    # The yield should assign back to acc
    assert "acc = " in code


# ---------------------------------------------------------------------------
# Test: while loop
# ---------------------------------------------------------------------------


def test_while_loop_with_iter_args():
    """WhileStmt with iter_args should convert to imperative mutable pattern."""
    init_val = _int(0)
    acc = ir.IterArg("counter", ir.ScalarType(DataType.INT64), init_val, _span())

    # Condition: counter < 10
    cond = ir.Lt(acc, _int(10), DataType.BOOL, _span())

    # Body: counter = counter + 1; yield counter
    one = _int(1)
    new_counter = _scalar("new_counter")
    add_expr = ir.Add(acc, one, DataType.INT64, _span())
    body = ir.SeqStmts(
        [
            ir.AssignStmt(new_counter, add_expr, _span()),
            ir.YieldStmt([new_counter], _span()),
        ],
        _span(),
    )

    result = _scalar("result")
    while_stmt = ir.WhileStmt(cond, [acc], body, [result], _span())

    func = _simple_function("f", [], while_stmt)
    code = torch_codegen(func)

    assert "counter = 0" in code
    assert "while" in code


# ---------------------------------------------------------------------------
# Test: if/else with return_vars
# ---------------------------------------------------------------------------


def test_if_else_with_return_vars():
    """IfStmt with return_vars should use yield to assign in each branch."""
    cond_var = _scalar("cond", DataType.BOOL)
    a = _scalar("a", DataType.FP32)
    b = _scalar("b", DataType.FP32)

    result = _scalar("result", DataType.FP32)
    then_body = ir.YieldStmt([a], _span())
    else_body = ir.YieldStmt([b], _span())

    if_stmt = ir.IfStmt(cond_var, then_body, else_body, [result], _span())
    func = _simple_function("f", [cond_var, a, b], if_stmt)
    code = torch_codegen(func)

    assert "if cond:" in code
    assert "else:" in code
    # Both branches should assign to result
    assert "result = a" in code
    assert "result = b" in code


# ---------------------------------------------------------------------------
# Test: system ops (no-ops)
# ---------------------------------------------------------------------------


def test_system_ops_are_noops():
    """System ops should emit no-op comments."""
    sync_call = _op_call("system.sync_src", [], {"set_pipe": 4, "wait_pipe": 5, "event_id": 0})
    body = ir.EvalStmt(sync_call, _span())
    func = _simple_function("f", [], body)
    code = torch_codegen(func)
    assert "# sync_src" in code


# ---------------------------------------------------------------------------
# Test: cross-core pipe ops
# ---------------------------------------------------------------------------


def test_pipe_ops():
    """tile.tpush/tpop should emit pipe simulation."""
    tile = _tile_var("tile", [64, 64])
    push_call = _op_call("tile.tpush_to_aiv", [tile], {"split": 0})
    body = ir.EvalStmt(push_call, _span())
    func = _simple_function("f", [tile], body)
    code = torch_codegen(func)
    assert "_pipes['to_aiv'].append" in code


# ---------------------------------------------------------------------------
# Test: preamble and program-level
# ---------------------------------------------------------------------------


def test_preamble_included():
    """Generated code should include the preamble with imports and helpers."""
    a = _tensor_var("a", [64])
    body = ir.ReturnStmt([a], _span())
    func = _simple_function("main", [a], body)
    code = torch_codegen(func)
    assert "import torch" in code
    assert "def _tile_load" in code
    assert "def _tile_store" in code


def test_program_with_multiple_functions():
    """torch_codegen on a Program should emit all functions."""
    a = _tensor_var("a", [64])
    b = _tensor_var("b", [64])
    f1 = _simple_function("func_a", [a], ir.ReturnStmt([a], _span()), [ir.TensorType([64], DataType.FP32)])
    f2 = _simple_function("func_b", [b], ir.ReturnStmt([b], _span()), [ir.TensorType([64], DataType.FP32)])
    prog = _program([f1, f2])
    code = torch_codegen(prog)
    assert "def func_a" in code
    assert "def func_b" in code


# ---------------------------------------------------------------------------
# Test: scope transparency
# ---------------------------------------------------------------------------


def test_scope_is_transparent():
    """ScopeStmt should not add any extra output, just emit the body."""
    a = _tensor_var("a", [64])
    b = _tensor_var("b", [64])
    call = _op_call("tensor.neg", [a])
    assign = ir.AssignStmt(b, call, _span())
    scope = ir.ScopeStmt(ir.ScopeKind.InCore, assign, _span())
    func = _simple_function("f", [a], scope)
    code = torch_codegen(func)
    assert "torch.neg(a)" in code
    # Should not contain scope markers
    assert "InCore" not in code


# ---------------------------------------------------------------------------
# Test: binary/unary IR expressions (not op calls)
# ---------------------------------------------------------------------------


def test_binary_ir_expressions():
    """IR binary expressions (Add, Sub, etc.) should emit Python operators."""
    a = _scalar("a", DataType.FP32)
    b = _scalar("b", DataType.FP32)
    c = _scalar("c", DataType.FP32)

    add = ir.Add(a, b, DataType.FP32, _span())
    assign = ir.AssignStmt(c, add, _span())
    func = _simple_function("f", [a, b], assign)
    code = torch_codegen(func)
    assert "(a + b)" in code


def test_break_continue():
    """BreakStmt and ContinueStmt should emit break/continue."""
    i = _scalar("i")
    body = ir.SeqStmts(
        [
            ir.BreakStmt(_span()),
            ir.ContinueStmt(_span()),
        ],
        _span(),
    )
    for_stmt = ir.ForStmt(i, _int(0), _int(10), _int(1), [], body, [], _span())
    func = _simple_function("f", [], for_stmt)
    code = torch_codegen(func)
    assert "break" in code
    assert "continue" in code


# ---------------------------------------------------------------------------
# Test: numerical round-trip (exec generated code)
# ---------------------------------------------------------------------------


def test_numerical_roundtrip_tensor_add():
    """Generated code from tensor.add should be executable and produce correct results."""
    a = _tensor_var("a", [4, 4])
    b = _tensor_var("b", [4, 4])
    c = _tensor_var("c", [4, 4])
    call = _op_call("tensor.add", [a, b])
    assign = ir.AssignStmt(c, call, _span())
    ret = ir.ReturnStmt([c], _span())
    body = ir.SeqStmts([assign, ret], _span())
    func = _simple_function("main", [a, b], body, [ir.TensorType([4, 4], DataType.FP32)])
    code = torch_codegen(func)

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    t_a = torch.ones(4, 4)
    t_b = torch.ones(4, 4) * 2
    result = ns["main"](t_a, t_b)
    assert torch.allclose(result, torch.ones(4, 4) * 3)


def test_numerical_roundtrip_for_loop():
    """Generated for loop code should be executable and accumulate correctly."""
    x = _scalar("x", DataType.FP32)
    i = _scalar("i")
    init_val = _float(0.0)
    acc = ir.IterArg("acc", ir.ScalarType(DataType.FP32), init_val, _span())

    new_acc = _scalar("new_acc", DataType.FP32)
    add_expr = ir.Add(acc, x, DataType.FP32, _span())
    body = ir.SeqStmts(
        [
            ir.AssignStmt(new_acc, add_expr, _span()),
            ir.YieldStmt([new_acc], _span()),
        ],
        _span(),
    )

    result = _scalar("result", DataType.FP32)
    for_stmt = ir.ForStmt(i, _int(0), _int(5), _int(1), [acc], body, [result], _span())
    ret = ir.ReturnStmt([result], _span())
    full_body = ir.SeqStmts([for_stmt, ret], _span())
    func = _simple_function("accumulate", [x], full_body, [ir.ScalarType(DataType.FP32)])
    code = torch_codegen(func)

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    # acc starts at 0.0, adds x=3.0 five times -> 15.0
    result_val = ns["accumulate"](3.0)
    assert result_val == pytest.approx(15.0)


def test_type_error_on_invalid_input():
    """torch_codegen should raise TypeError for non-Program/Function input."""
    with pytest.raises(TypeError, match="torch_codegen expects"):
        torch_codegen(ir.ConstInt(42, DataType.INT64, _span()))  # type: ignore[arg-type]


def test_unsupported_op_raises():
    """torch_codegen should raise ValueError for unregistered ops."""
    a = _tensor_var("a", [64])
    out = _tensor_var("out", [64])
    # Construct a Call with a plain Op (not GlobalVar, not in _OP_MAP)
    fake_op = ir.Op("fake.nonexistent_op")
    call = ir.Call(fake_op, [a], _span())
    assign = ir.AssignStmt(out, call, _span())
    func = _simple_function("f", [a], assign)
    with pytest.raises(ValueError, match="Unsupported op 'fake.nonexistent_op'"):
        torch_codegen(func)


# ---------------------------------------------------------------------------
# Test: write ops return container (not None)
# ---------------------------------------------------------------------------


def test_tile_write_returns_tile():
    """tile.write in AssignStmt context should return the tile, not None."""
    tile = _tile_var("tile", [64, 64])
    idx = _make_tuple(_int(0), _int(0))
    val = _float(1.0)
    result = _tile_var("result", [64, 64])

    call = _op_call("tile.write", [tile, idx, val])
    assign = ir.AssignStmt(result, call, _span())
    ret = ir.ReturnStmt([result], _span())
    body = ir.SeqStmts([assign, ret], _span())
    func = _simple_function("f", [tile], body, [ir.TileType([64, 64], DataType.FP32)])
    code = torch_codegen(func)

    assert "_write_and_return" in code
    # Execute and verify the result is the tile, not None
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    t = torch.zeros(64, 64)
    result_val = ns["f"](t)
    assert isinstance(result_val, torch.Tensor)
    assert result_val.shape == (64, 64)
    assert result_val[0, 0] == 1.0


def test_tensor_write_returns_tensor():
    """tensor.write in AssignStmt context should return the tensor, not None."""
    tensor = _tensor_var("t", [4, 4])
    idx = _make_tuple(_int(1), _int(2))
    val = _float(42.0)
    result = _tensor_var("result", [4, 4])

    call = _op_call("tensor.write", [tensor, idx, val])
    assign = ir.AssignStmt(result, call, _span())
    ret = ir.ReturnStmt([result], _span())
    body = ir.SeqStmts([assign, ret], _span())
    func = _simple_function("f", [tensor], body, [ir.TensorType([4, 4], DataType.FP32)])
    code = torch_codegen(func)

    assert "_write_and_return" in code
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    t = torch.zeros(4, 4)
    result_val = ns["f"](t)
    assert result_val is not None
    assert result_val[1, 2] == 42.0


# ---------------------------------------------------------------------------
# Test: assemble applies source write
# ---------------------------------------------------------------------------


def test_tile_assemble_writes_source():
    """tile.assemble should write source into target at offset."""
    target = _tile_var("target", [8, 8])
    source = _tile_var("source", [4, 4])
    offset = _make_tuple(_int(2), _int(2))
    result = _tile_var("result", [8, 8])

    call = _op_call("tile.assemble", [target, source, offset])
    assign = ir.AssignStmt(result, call, _span())
    ret = ir.ReturnStmt([result], _span())
    body = ir.SeqStmts([assign, ret], _span())
    func = _simple_function("f", [target, source], body, [ir.TileType([8, 8], DataType.FP32)])
    code = torch_codegen(func)

    assert "_assemble" in code
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    tgt = torch.zeros(8, 8)
    src = torch.ones(4, 4)
    result_val = ns["f"](tgt, src)
    # Source should be written at offset [2:6, 2:6]
    assert result_val[2, 2] == 1.0
    assert result_val[5, 5] == 1.0
    # Outside the write region should be zero
    assert result_val[0, 0] == 0.0
    assert result_val[7, 7] == 0.0


def test_tensor_assemble_writes_source():
    """tensor.assemble should write source into target at offset."""
    target = _tensor_var("target", [8, 8])
    source = _tensor_var("source", [4, 4])
    offset = _make_tuple(_int(0), _int(0))
    result = _tensor_var("result", [8, 8])

    call = _op_call("tensor.assemble", [target, source, offset])
    assign = ir.AssignStmt(result, call, _span())
    ret = ir.ReturnStmt([result], _span())
    body = ir.SeqStmts([assign, ret], _span())
    func = _simple_function("f", [target, source], body, [ir.TensorType([8, 8], DataType.FP32)])
    code = torch_codegen(func)

    assert "_assemble" in code
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    tgt = torch.zeros(8, 8)
    src = torch.ones(4, 4) * 5
    result_val = ns["f"](tgt, src)
    assert result_val[0, 0] == 5.0
    assert result_val[3, 3] == 5.0
    assert result_val[4, 4] == 0.0


# ---------------------------------------------------------------------------
# Test: valid_shapes masking in tile.load
# ---------------------------------------------------------------------------


def test_tile_load_valid_shapes_masks_invalid():
    """tile.load should zero out data beyond valid_shapes."""
    tensor = _tensor_var("t", [8, 8])
    offsets = _make_tuple(_int(0), _int(0))
    shapes = _make_tuple(_int(8), _int(8))
    valid_shapes = _make_tuple(_int(4), _int(4))
    tile = _tile_var("tile", [8, 8])

    call = _op_call(
        "tile.load",
        [tensor, offsets, shapes, valid_shapes],
        {"target_memory": ir.MemorySpace.Vec, "transpose": False},
    )
    assign = ir.AssignStmt(tile, call, _span())
    ret = ir.ReturnStmt([tile], _span())
    body = ir.SeqStmts([assign, ret], _span())
    func = _simple_function("f", [tensor], body, [ir.TileType([8, 8], DataType.FP32)])
    code = torch_codegen(func)

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    t = torch.ones(8, 8)
    result = ns["f"](t)
    # Valid region [0:4, 0:4] should have ones
    assert result[0, 0] == 1.0
    assert result[3, 3] == 1.0
    # Invalid region should be masked to zero
    assert result[4, 4] == 0.0
    assert result[7, 7] == 0.0


def test_tile_load_passes_valid_shapes():
    """tile.load codegen should pass valid_shapes as 4th arg to _tile_load."""
    tensor = _tensor_var("t", [64, 64])
    offsets = _make_tuple(_int(0), _int(0))
    shapes = _make_tuple(_int(32), _int(32))
    valid_shapes = _make_tuple(_int(16), _int(16))
    tile = _tile_var("tile", [32, 32])

    call = _op_call(
        "tile.load",
        [tensor, offsets, shapes, valid_shapes],
        {"target_memory": ir.MemorySpace.Vec, "transpose": False},
    )
    assign = ir.AssignStmt(tile, call, _span())
    func = _simple_function("f", [tensor], assign)
    code = torch_codegen(func)
    # Should pass all 4 args including valid_shapes
    assert "_tile_load(t, (0, 0), (32, 32), (16, 16))" in code


# ---------------------------------------------------------------------------
# Test: variable name sanitization
# ---------------------------------------------------------------------------


def test_variable_name_sanitization():
    """Variable names with invalid Python chars should be sanitized."""
    cg = TorchCodegen()
    # Names with double underscores (from BuildName) should be collapsed
    assert cg._unique_name("x__y") == "x_y"

    # Names starting with digits
    assert cg._unique_name("0abc") == "v_0abc"

    # Python keywords
    assert cg._unique_name("for") == "for_v"

    # Names with special chars
    assert cg._unique_name("a.b-c") == "a_b_c"


def test_variable_name_uniquing():
    """Repeated name hints should produce unique suffixed names."""
    cg = TorchCodegen()
    assert cg._unique_name("a") == "a"
    assert cg._unique_name("a") == "a_1"
    assert cg._unique_name("a") == "a_2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
