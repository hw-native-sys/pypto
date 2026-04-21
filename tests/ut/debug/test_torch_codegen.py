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
    # When a_trans=True, K is lhs_shape[0] and M is lhs_shape[1].
    # a = [K=128, M=64], b = [K=128, N=64] -> output [M=64, N=64]
    a = _tensor_var("a", [128, 64])
    b = _tensor_var("b", [128, 64])
    out = _tensor_var("out", [64, 64])
    call = _op_call("tensor.matmul", [a, b], {"a_trans": True, "b_trans": False, "c_matrix_nz": False})
    assign = ir.AssignStmt(out, call, _span())
    func = _simple_function("f", [a, b], assign)
    code = torch_codegen(func)
    assert "a.mT" in code
    assert "torch.matmul" in code


def test_tensor_matmul_respects_out_dtype():
    """tensor.matmul with out_dtype should cast result to the requested dtype."""
    a = _tensor_var("a", [64, 128], DataType.BF16)
    b = _tensor_var("b", [128, 64], DataType.BF16)
    out = _tensor_var("out", [64, 64], DataType.FP32)
    call = _op_call("tensor.matmul", [a, b], {"a_trans": False, "b_trans": False, "out_dtype": DataType.FP32})
    assign = ir.AssignStmt(out, call, _span())
    func = _simple_function("f", [a, b], assign)
    code = torch_codegen(func)
    assert "torch.matmul(a, b).to(torch.float32)" in code


def test_tensor_cast():
    """tensor.cast should emit .to(dtype)."""
    a = _tensor_var("a", [64])
    out = _tensor_var("out", [64])
    call = _op_call("tensor.cast", [a], {"target_type": DataType.FP16, "mode": 0})
    assign = ir.AssignStmt(out, call, _span())
    func = _simple_function("f", [a], assign)
    code = torch_codegen(func)
    assert ".to(torch.float16)" in code


def test_nested_call_arg_preserves_call_codegen():
    """Nested call arg (cast(slice(...))) should emit slice expression, not tuple."""
    hidden_states = _tensor_var("hidden_states", [16, 512], DataType.BF16)
    out = _tensor_var("out", [16, 512], DataType.FP32)
    shapes = _make_tuple(_int(16), _int(512))
    offsets = _make_tuple(_int(0), _int(0))
    slice_call = _op_call("tensor.slice", [hidden_states, shapes, offsets])
    cast_call = _op_call("tensor.cast", [slice_call], {"target_type": DataType.FP32, "mode": 2})
    assign = ir.AssignStmt(out, cast_call, _span())
    func = _simple_function("f", [hidden_states], assign)
    code = torch_codegen(func)

    assert "_tensor_slice(hidden_states, (0, 0), (16, 512)).to(torch.float32)" in code
    assert "(0, 0).to(torch.float32)" not in code


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
    """tile.tpush/tpop should emit pipe simulation with split support."""
    tile = _tile_var("tile", [64, 64])
    push_call = _op_call("tile.tpush_to_aiv", [tile], {"split": 0})
    body = ir.EvalStmt(push_call, _span())
    func = _simple_function("f", [tile], body)
    code = torch_codegen(func)
    assert "_tpush_to_aiv(_pipes['to_aiv'], tile, 0)" in code


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
    scope = ir.InCoreScopeStmt(body=assign, span=_span())
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


def test_tensor_slice_out_of_bounds_is_padded():
    """tensor.slice should pad to requested shape when slicing out of bounds."""
    src = _tensor_var("src", [96, 64], DataType.FP32)
    result = _tensor_var("result", [64, 64], DataType.FP32)
    shapes = _make_tuple(_int(64), _int(64))
    offsets = _make_tuple(_int(64), _int(0))
    call = _op_call("tensor.slice", [src, shapes, offsets])
    assign = ir.AssignStmt(result, call, _span())
    ret = ir.ReturnStmt([result], _span())
    body = ir.SeqStmts([assign, ret], _span())
    func = _simple_function("f", [src], body, [ir.TensorType([64, 64], DataType.FP32)])
    code = torch_codegen(func)

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    x = torch.ones(96, 64, dtype=torch.float32)
    out = ns["f"](x)
    assert out.shape == (64, 64)
    assert torch.allclose(out[:32, :], torch.ones(32, 64))
    assert torch.allclose(out[32:, :], torch.zeros(32, 64))


def test_tensor_fillpad_min_uses_valid_shape():
    """tensor.fillpad should apply pad_value outside valid_shape metadata."""
    src = _tensor_var("src", [8, 64], DataType.FP32)
    result = _tensor_var("result", [8, 64], DataType.FP32)
    shapes = _make_tuple(_int(8), _int(64))
    offsets = _make_tuple(_int(0), _int(0))
    valid_shapes = _make_tuple(_int(8), _int(32))
    sliced = _op_call("tensor.slice", [src, shapes, offsets, valid_shapes])
    padded = _op_call("tensor.fillpad", [sliced], {"pad_value": ir.PadValue.min})
    assign = ir.AssignStmt(result, padded, _span())
    ret = ir.ReturnStmt([result], _span())
    body = ir.SeqStmts([assign, ret], _span())
    func = _simple_function("f", [src], body, [ir.TensorType([8, 64], DataType.FP32)])
    code = torch_codegen(func)

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    x = torch.rand(8, 64, dtype=torch.float32)
    out = ns["f"](x)
    assert out.shape == (8, 64)
    assert torch.allclose(out[:, :32], x[:, :32])
    assert torch.all(out[:, 32:] == torch.finfo(torch.float32).min)


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


# ---------------------------------------------------------------------------
# Test: split-aware tpush/tpop
# ---------------------------------------------------------------------------


def test_tpush_no_split():
    """tpush with split=0 should push whole tensor (backward compatible)."""
    tile = _tile_var("tile", [64, 64])
    push_call = _op_call("tile.tpush_to_aiv", [tile], {"split": 0})
    body = ir.EvalStmt(push_call, _span())
    func = _simple_function("f", [tile], body)
    code = torch_codegen(func)
    assert "_tpush_to_aiv(_pipes['to_aiv'], tile, 0)" in code


def test_tpush_updown_split():
    """tpush with split=1 (UpDown) should use the AIC->AIV split helper."""
    tile = _tile_var("tile", [64, 64])
    push_call = _op_call("tile.tpush_to_aiv", [tile], {"split": 1})
    body = ir.EvalStmt(push_call, _span())
    func = _simple_function("f", [tile], body)
    code = torch_codegen(func)
    assert "_tpush_to_aiv(_pipes['to_aiv'], tile, 1)" in code


def test_tpush_leftright_split():
    """tpush with split=2 (LeftRight) should use the V2C helper."""
    tile = _tile_var("tile", [64, 64])
    push_call = _op_call("tile.tpush_to_aic", [tile], {"split": 2})
    body = ir.EvalStmt(push_call, _span())
    func = _simple_function("f", [tile], body)
    code = torch_codegen(func)
    assert "_tpush_to_aic(_pipes['to_aic'], tile, 2)" in code


def test_tpop_no_split():
    """tpop_from_aic with split=0 should pop whole tensor (backward compatible)."""
    pop_call = _op_call("tile.tpop_from_aic", [], {"split": 0})
    out = _tile_var("out", [64, 64])
    assign = ir.AssignStmt(out, pop_call, _span())
    func = _simple_function("f", [], assign)
    code = torch_codegen(func)
    # tpop_from_aic reads from the AIC→AIV pipe ('to_aiv')
    assert "_tpop_from_aic(_pipes['to_aiv'], 0)" in code


def test_tpop_updown_split():
    """tpop_from_aiv with split=1 should still use the full-tile V2C helper."""
    pop_call = _op_call("tile.tpop_from_aiv", [], {"split": 1})
    out = _tile_var("out", [64, 64])
    assign = ir.AssignStmt(out, pop_call, _span())
    func = _simple_function("f", [], assign)
    code = torch_codegen(func)
    # tpop_from_aiv reads from the AIV→AIC pipe ('to_aic')
    assert "_tpop_from_aiv(_pipes['to_aic'], 1)" in code


def test_tpop_leftright_split():
    """tpop_from_aic with split=2 should use the split-aware C2V helper."""
    pop_call = _op_call("tile.tpop_from_aic", [], {"split": 2})
    out = _tile_var("out", [64, 64])
    assign = ir.AssignStmt(out, pop_call, _span())
    func = _simple_function("f", [], assign)
    code = torch_codegen(func)
    # tpop_from_aic reads from the AIC→AIV pipe ('to_aiv')
    assert "_tpop_from_aic(_pipes['to_aiv'], 2)" in code


# ---------------------------------------------------------------------------
# Test: numerical roundtrip for tpush/tpop with split
# ---------------------------------------------------------------------------


def test_numerical_roundtrip_tpush_tpop_no_split():
    """End-to-end: tpush then tpop with no split should preserve data."""
    tile = _tile_var("tile", [4, 4])
    out = _tile_var("out", [4, 4])

    push_call = _op_call("tile.tpush_to_aiv", [tile], {"split": 0})
    pop_call = _op_call("tile.tpop_from_aic", [], {"split": 0})

    body = ir.SeqStmts(
        [
            ir.EvalStmt(push_call, _span()),
            ir.AssignStmt(out, pop_call, _span()),
            ir.ReturnStmt([out], _span()),
        ],
        _span(),
    )
    func = _simple_function("f", [tile], body, [ir.TileType([4, 4], DataType.FP32)])
    code = torch_codegen(func)

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    t = torch.randn(4, 4)
    result = ns["f"](t)
    assert torch.allclose(result, t)


def test_numerical_roundtrip_tpush_tpop_updown_split():
    """Legacy non-Group roundtrip: tpush(split=UpDown) + tpop reassembles full tile."""
    tile = _tile_var("tile", [4, 4])
    out = _tile_var("out", [4, 4])

    push_call = _op_call("tile.tpush_to_aiv", [tile], {"split": 1})
    pop_call = _op_call("tile.tpop_from_aic", [], {"split": 1})

    body = ir.SeqStmts(
        [
            ir.EvalStmt(push_call, _span()),
            ir.AssignStmt(out, pop_call, _span()),
            ir.ReturnStmt([out], _span()),
        ],
        _span(),
    )
    func = _simple_function("f", [tile], body, [ir.TileType([4, 4], DataType.FP32)])
    code = torch_codegen(func)

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    t = torch.randn(4, 4)
    result = ns["f"](t)
    assert torch.allclose(result, t)


def test_numerical_roundtrip_tpush_tpop_leftright_split():
    """Legacy non-Group roundtrip: V2C tpush+tpop with split kwarg returns full tile.

    Outside the scheduler we do not model two AIV subblocks pushing
    independent halves; the legacy single-subblock path pushes the full
    tile and pops the full tile, so split is informational only.
    """
    tile = _tile_var("tile", [4, 4])
    out = _tile_var("out", [4, 4])

    push_call = _op_call("tile.tpush_to_aic", [tile], {"split": 2})
    pop_call = _op_call("tile.tpop_from_aiv", [], {"split": 2})

    body = ir.SeqStmts(
        [
            ir.EvalStmt(push_call, _span()),
            ir.AssignStmt(out, pop_call, _span()),
            ir.ReturnStmt([out], _span()),
        ],
        _span(),
    )
    func = _simple_function("f", [tile], body, [ir.TileType([4, 4], DataType.FP32)])
    code = torch_codegen(func)

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    t = torch.randn(4, 4)
    result = ns["f"](t)
    assert torch.allclose(result, t)


# ---------------------------------------------------------------------------
# Test: cross-core program simulation
# ---------------------------------------------------------------------------


def test_program_with_aic_aiv_functions():
    """Program with AIC+AIV functions should emit both function definitions."""
    span = _span()

    # AIC function: takes tile, pushes to AIV
    tile = _tile_var("tile", [4, 4])
    push_call = _op_call("tile.tpush_to_aiv", [tile], {"split": 0})
    aic_body = ir.EvalStmt(push_call, span)
    aic_func = ir.Function("aic_compute", [tile], [], aic_body, span, type=ir.FunctionType.AIC)

    # AIV function: pops from AIV, adds 1, returns
    pop_op = ir.get_op("tile.tpop_from_aic")
    pop_call = ir.Call(pop_op, [], {"split": 0}, ir.TileType([4, 4], DataType.FP32), span)
    out = _tile_var("out", [4, 4])
    add_call = _op_call("tile.adds", [pop_call, _float(1.0)])
    aiv_body = ir.SeqStmts(
        [
            ir.AssignStmt(out, add_call, span),
            ir.ReturnStmt([out], span),
        ],
        span,
    )
    aiv_func = ir.Function(
        "aiv_compute", [], [ir.TileType([4, 4], DataType.FP32)], aiv_body, span, type=ir.FunctionType.AIV
    )

    prog = _program([aic_func, aiv_func])
    code = torch_codegen(prog)

    assert "def aic_compute" in code
    assert "def aiv_compute" in code
    assert "_tpush_to_aiv" in code
    assert "_tpop_from_aic" in code


def test_program_with_group_function():
    """Program with Group function should emit coordinated AIC+AIV calls."""
    span = _span()

    # AIC function
    tile = _tile_var("tile", [4, 4])
    push_call = _op_call("tile.tpush_to_aiv", [tile], {"split": 0})
    aic_body = ir.EvalStmt(push_call, span)
    aic_func = ir.Function("aic_kernel", [tile], [], aic_body, span, type=ir.FunctionType.AIC)

    # AIV function
    pop_op = ir.get_op("tile.tpop_from_aic")
    pop_call = ir.Call(pop_op, [], {"split": 0}, ir.TileType([4, 4], DataType.FP32), span)
    out = _tile_var("out", [4, 4])
    add_call = _op_call("tile.adds", [pop_call, _float(1.0)])
    aiv_body = ir.SeqStmts(
        [
            ir.AssignStmt(out, add_call, span),
            ir.ReturnStmt([out], span),
        ],
        span,
    )
    aiv_func = ir.Function(
        "aiv_kernel", [], [ir.TileType([4, 4], DataType.FP32)], aiv_body, span, type=ir.FunctionType.AIV
    )

    # Group function that calls both
    aic_gv = ir.GlobalVar("aic_kernel")
    aiv_gv = ir.GlobalVar("aiv_kernel")
    group_body = ir.SeqStmts(
        [
            ir.EvalStmt(ir.Call(aic_gv, [tile], span), span),
            ir.EvalStmt(ir.Call(aiv_gv, [], span), span),
        ],
        span,
    )
    group_func = ir.Function(
        "my_group", [tile], [ir.TileType([4, 4], DataType.FP32)], group_body, span, type=ir.FunctionType.Group
    )

    prog = _program([aic_func, aiv_func, group_func])
    code = torch_codegen(prog)

    assert "def aic_kernel" in code
    assert "def aiv_kernel" in code
    assert "def my_group" in code
    assert "# Group:" in code


def test_program_with_multiple_group_functions():
    """Program with multiple Group functions should emit all with isolated variable scopes."""
    span = _span()

    # === Group 1: matmul pipeline ===
    # AIC function for group 1
    tile1 = _tile_var("tile", [4, 4])  # same name as tile2, but different scope
    push_call1 = _op_call("tile.tpush_to_aiv", [tile1], {"split": 0})
    aic_body1 = ir.EvalStmt(push_call1, span)
    aic_func1 = ir.Function("aic_matmul", [tile1], [], aic_body1, span, type=ir.FunctionType.AIC)

    # AIV function for group 1
    pop_call1 = ir.Call(
        ir.get_op("tile.tpop_from_aic"), [], {"split": 0}, ir.TileType([4, 4], DataType.FP32), span
    )
    out1 = _tile_var("out", [4, 4])
    add_call1 = _op_call("tile.adds", [pop_call1, _float(1.0)])
    aiv_body1 = ir.SeqStmts(
        [ir.AssignStmt(out1, add_call1, span), ir.ReturnStmt([out1], span)],
        span,
    )
    aiv_func1 = ir.Function(
        "aiv_gelu", [], [ir.TileType([4, 4], DataType.FP32)], aiv_body1, span, type=ir.FunctionType.AIV
    )

    # Group 1 function
    group_body1 = ir.SeqStmts(
        [
            ir.EvalStmt(ir.Call(ir.GlobalVar("aic_matmul"), [tile1], span), span),
            ir.AssignStmt(out1, ir.Call(ir.GlobalVar("aiv_gelu"), [], span), span),
            ir.ReturnStmt([out1], span),
        ],
        span,
    )
    group_func1 = ir.Function(
        "matmul_pipeline",
        [tile1],
        [ir.TileType([4, 4], DataType.FP32)],
        group_body1,
        span,
        type=ir.FunctionType.Group,
    )

    # === Group 2: activation pipeline (AIV pushes to AIC, AIC consumes) ===
    # AIV function for group 2
    tile2 = _tile_var("tile", [4, 4])  # same name as tile1
    push_call2 = _op_call("tile.tpush_to_aic", [tile2], {"split": 0})
    aiv_body2 = ir.EvalStmt(push_call2, span)
    aiv_func2 = ir.Function("aiv_activation", [tile2], [], aiv_body2, span, type=ir.FunctionType.AIV)

    # AIC function for group 2
    pop_call2 = ir.Call(
        ir.get_op("tile.tpop_from_aiv"), [], {"split": 0}, ir.TileType([4, 4], DataType.FP32), span
    )
    out2 = _tile_var("out", [4, 4])
    mul_call2 = _op_call("tile.muls", [pop_call2, _float(2.0)])
    aic_body2 = ir.SeqStmts(
        [ir.AssignStmt(out2, mul_call2, span), ir.ReturnStmt([out2], span)],
        span,
    )
    aic_func2 = ir.Function(
        "aic_norm", [], [ir.TileType([4, 4], DataType.FP32)], aic_body2, span, type=ir.FunctionType.AIC
    )

    # Group 2 function
    group_body2 = ir.SeqStmts(
        [
            ir.EvalStmt(ir.Call(ir.GlobalVar("aiv_activation"), [tile2], span), span),
            ir.AssignStmt(out2, ir.Call(ir.GlobalVar("aic_norm"), [], span), span),
            ir.ReturnStmt([out2], span),
        ],
        span,
    )
    group_func2 = ir.Function(
        "activation_pipeline",
        [tile2],
        [ir.TileType([4, 4], DataType.FP32)],
        group_body2,
        span,
        type=ir.FunctionType.Group,
    )

    # Build program with both groups
    prog = _program([aic_func1, aiv_func1, aiv_func2, aic_func2, group_func1, group_func2])
    code = torch_codegen(prog)

    # Verify all functions are generated
    assert "def aic_matmul" in code
    assert "def aiv_gelu" in code
    assert "def aiv_activation" in code
    assert "def aic_norm" in code
    assert "def matmul_pipeline" in code
    assert "def activation_pipeline" in code

    # Verify both Group comments are present
    assert code.count("# Group:") == 2

    # Execute and verify numerical correctness for both groups
    ns: dict = {}
    exec(code, ns)  # noqa: S102

    t = torch.randn(4, 4)

    # Group 1: adds 1.0
    result1 = ns["matmul_pipeline"](t)
    assert torch.allclose(result1, t + 1.0, atol=1e-6)

    # Group 2: multiplies by 2.0
    result2 = ns["activation_pipeline"](t)
    assert torch.allclose(result2, t * 2.0, atol=1e-6)


def test_program_emits_entry_point():
    """Program codegen should emit a run() entry point for Opaque functions."""
    span = _span()

    a = _tensor_var("a", [4, 4])
    ret = ir.ReturnStmt([a], span)
    func = ir.Function(
        "main", [a], [ir.TensorType([4, 4], DataType.FP32)], ret, span, type=ir.FunctionType.Opaque
    )
    prog = _program([func])
    code = torch_codegen(prog)

    assert "def run(a):" in code
    assert "return main(a)" in code


def test_program_entry_point_sanitizes_parameter_names():
    """Program entry point should sanitize/unique parameter names like function codegen."""
    span = _span()

    class_kw = _tensor_var("class", [4, 4])
    duplicate = _tensor_var("class", [4, 4])
    ret = ir.ReturnStmt([class_kw], span)
    func = ir.Function(
        "main",
        [class_kw, duplicate],
        [ir.TensorType([4, 4], DataType.FP32)],
        ret,
        span,
        type=ir.FunctionType.Opaque,
    )
    prog = _program([func])
    code = torch_codegen(prog)

    assert "def main(class_v, class_v_1):" in code
    assert "def run(class_v, class_v_1):" in code
    compile(code, "<generated>", "exec")


def test_program_entry_point_prefers_orchestration():
    """Program entry point should prefer Orchestration function over Opaque."""
    span = _span()

    a = _tensor_var("a", [4, 4])

    opaque_ret = ir.ReturnStmt([a], span)
    opaque_func = ir.Function(
        "helper", [a], [ir.TensorType([4, 4], DataType.FP32)], opaque_ret, span, type=ir.FunctionType.Opaque
    )

    orch_ret = ir.ReturnStmt([a], span)
    orch_func = ir.Function(
        "orch_main",
        [a],
        [ir.TensorType([4, 4], DataType.FP32)],
        orch_ret,
        span,
        type=ir.FunctionType.Orchestration,
    )

    prog = _program([opaque_func, orch_func])
    code = torch_codegen(prog)

    assert "return orch_main(a)" in code


# ---------------------------------------------------------------------------
# Test: qwen3-style cross-core precision verification
# ---------------------------------------------------------------------------


def test_numerical_cross_core_matmul_residual():
    """End-to-end cross-core: AIC does matmul, pushes to AIV, AIV adds residual.

    Simulates the qwen3 pattern: output = matmul(a, b), then result = output + residual.
    """
    span = _span()

    # AIC function: matmul then tpush
    a = _tile_var("a", [4, 4])
    b = _tile_var("b", [4, 4])
    matmul_call = _op_call("tile.matmul", [a, b])
    result = _tile_var("result", [4, 4])
    push_call = _op_call("tile.tpush_to_aiv", [result], {"split": 0})
    aic_body = ir.SeqStmts(
        [
            ir.AssignStmt(result, matmul_call, span),
            ir.EvalStmt(push_call, span),
        ],
        span,
    )
    aic_func = ir.Function("aic_matmul", [a, b], [], aic_body, span, type=ir.FunctionType.AIC)

    # AIV function: tpop then add residual
    residual = _tile_var("residual", [4, 4])
    pop_op = ir.get_op("tile.tpop_from_aic")
    pop_call = ir.Call(pop_op, [], {"split": 0}, ir.TileType([4, 4], DataType.FP32), span)
    out = _tile_var("out", [4, 4])
    add_call = _op_call("tile.add", [pop_call, residual])
    aiv_body = ir.SeqStmts(
        [
            ir.AssignStmt(out, add_call, span),
            ir.ReturnStmt([out], span),
        ],
        span,
    )
    aiv_func = ir.Function(
        "aiv_add_residual",
        [residual],
        [ir.TileType([4, 4], DataType.FP32)],
        aiv_body,
        span,
        type=ir.FunctionType.AIV,
    )

    # Group function: calls AIC then AIV
    aic_call = ir.Call(ir.GlobalVar("aic_matmul"), [a, b], span)
    aiv_call = ir.Call(ir.GlobalVar("aiv_add_residual"), [residual], span)
    group_body = ir.SeqStmts(
        [
            ir.EvalStmt(aic_call, span),
            ir.AssignStmt(out, aiv_call, span),
            ir.ReturnStmt([out], span),
        ],
        span,
    )
    group_func = ir.Function(
        "matmul_add_group",
        [a, b, residual],
        [ir.TileType([4, 4], DataType.FP32)],
        group_body,
        span,
        type=ir.FunctionType.Group,
    )

    prog = _program([aic_func, aiv_func, group_func])
    code = torch_codegen(prog)

    # Execute and verify
    ns: dict = {}
    exec(code, ns)  # noqa: S102

    t_a = torch.randn(4, 4)
    t_b = torch.randn(4, 4)
    t_residual = torch.randn(4, 4)

    result_val = ns["matmul_add_group"](t_a, t_b, t_residual)
    expected = torch.matmul(t_a, t_b).float() + t_residual
    assert torch.allclose(result_val, expected, atol=1e-5)


def _build_cross_core_matmul_residual_program(split: int) -> ir.Program:
    """Build a cross-core Program: AIC matmul → tpush → AIV tpop + add residual.

    For split == 0: AIV pops the full tile, adds residual, returns it; the
    Group binds the AIV return to ``out`` and returns it.  For split > 0:
    the kernel is restructured as a bidirectional roundtrip — AIC pushes the
    matmul output split, each AIV subblock pops its half, adds the matching
    half of residual (sliced via ``tile.get_subblock_idx``), pushes the
    half-result back to AIC; AIC pops with the same split and reassembles
    the full tile, storing into the ``out`` output tensor.  The Group has
    no return value in this mode (``out`` is an output param).
    """
    span = _span()

    if split == 0:
        a = _tile_var("a", [4, 4])
        b = _tile_var("b", [4, 4])
        matmul_call = _op_call("tile.matmul", [a, b])
        result = _tile_var("result", [4, 4])
        push_call = _op_call("tile.tpush_to_aiv", [result], {"split": 0})
        aic_body = ir.SeqStmts(
            [ir.AssignStmt(result, matmul_call, span), ir.EvalStmt(push_call, span)],
            span,
        )
        aic_func = ir.Function("aic_matmul", [a, b], [], aic_body, span, type=ir.FunctionType.AIC)

        residual = _tile_var("residual", [4, 4])
        pop_op = ir.get_op("tile.tpop_from_aic")
        pop_call = ir.Call(pop_op, [], {"split": 0}, ir.TileType([4, 4], DataType.FP32), span)
        out = _tile_var("out", [4, 4])
        add_call = _op_call("tile.add", [pop_call, residual])
        aiv_body = ir.SeqStmts(
            [ir.AssignStmt(out, add_call, span), ir.ReturnStmt([out], span)],
            span,
        )
        aiv_func = ir.Function(
            "aiv_add_residual",
            [residual],
            [ir.TileType([4, 4], DataType.FP32)],
            aiv_body,
            span,
            type=ir.FunctionType.AIV,
        )

        aic_call = ir.Call(ir.GlobalVar("aic_matmul"), [a, b], span)
        aiv_call = ir.Call(ir.GlobalVar("aiv_add_residual"), [residual], span)
        group_body = ir.SeqStmts(
            [
                ir.EvalStmt(aic_call, span),
                ir.AssignStmt(out, aiv_call, span),
                ir.ReturnStmt([out], span),
            ],
            span,
        )
        group_func = ir.Function(
            "matmul_add_group",
            [a, b, residual],
            [ir.TileType([4, 4], DataType.FP32)],
            group_body,
            span,
            type=ir.FunctionType.Group,
        )
        return _program([aic_func, aiv_func, group_func])

    # split > 0: bidirectional roundtrip with reassembly at AIC.  Output is a
    # full-shape tensor written via tile.store at offset (0, 0).
    full_shape = [4, 4]

    # AIC: result = matmul(a, b); tpush_to_aiv(result, split); reassembled =
    # tpop_from_aiv(split); store(reassembled, (0,0), out_tensor)
    a = _tile_var("a", full_shape)
    b = _tile_var("b", full_shape)
    out_tensor = _tensor_var("out_tensor", full_shape)
    matmul_call = _op_call("tile.matmul", [a, b])
    result = _tile_var("result", full_shape)
    push_call = _op_call("tile.tpush_to_aiv", [result], {"split": split})
    pop_back_op = ir.get_op("tile.tpop_from_aiv")
    pop_back_call = ir.Call(
        pop_back_op,
        [],
        {"split": split},
        ir.TileType(full_shape, DataType.FP32),
        span,
    )
    reassembled = _tile_var("reassembled", full_shape)
    offsets_zero = ir.MakeTuple([_int(0), _int(0)], span)
    store_call = _op_call("tile.store", [reassembled, offsets_zero, out_tensor])
    aic_body = ir.SeqStmts(
        [
            ir.AssignStmt(result, matmul_call, span),
            ir.EvalStmt(push_call, span),
            ir.AssignStmt(reassembled, pop_back_call, span),
            ir.EvalStmt(store_call, span),
        ],
        span,
    )
    aic_func = ir.Function(
        "aic_matmul",
        [a, b, out_tensor],
        [],
        aic_body,
        span,
        type=ir.FunctionType.AIC,
    )

    # AIV body (runs once per subblock): pop half from AIC, add the matching
    # half of residual sliced by subblock_idx, push back to AIC.
    if split == 1:  # UpDown
        half_shape = [full_shape[0] // 2, full_shape[1]]
        # offset along dim 0 = subblock_idx * (H/2)
        offset_dim0 = ir.Mul(
            ir.create_op_call("tile.get_subblock_idx", [], span),
            _int(half_shape[0]),
            DataType.INT64,
            span,
        )
        slice_offsets = ir.MakeTuple([offset_dim0, _int(0)], span)
    else:  # LeftRight
        half_shape = [full_shape[0], full_shape[1] // 2]
        offset_dim1 = ir.Mul(
            ir.create_op_call("tile.get_subblock_idx", [], span),
            _int(half_shape[1]),
            DataType.INT64,
            span,
        )
        slice_offsets = ir.MakeTuple([_int(0), offset_dim1], span)

    residual_full = _tile_var("residual", full_shape)
    pop_op = ir.get_op("tile.tpop_from_aic")
    pop_call = ir.Call(
        pop_op,
        [],
        {"split": split},
        ir.TileType(half_shape, DataType.FP32),
        span,
    )
    popped_half = _tile_var("popped_half", half_shape)
    half_shape_tuple = ir.MakeTuple([_int(d) for d in half_shape], span)
    residual_half_call = _op_call("tile.slice", [residual_full, half_shape_tuple, slice_offsets])
    residual_half = _tile_var("residual_half", half_shape)
    add_call = _op_call("tile.add", [popped_half, residual_half])
    add_half = _tile_var("add_half", half_shape)
    push_back_call = _op_call("tile.tpush_to_aic", [add_half], {"split": split})
    aiv_body = ir.SeqStmts(
        [
            ir.AssignStmt(popped_half, pop_call, span),
            ir.AssignStmt(residual_half, residual_half_call, span),
            ir.AssignStmt(add_half, add_call, span),
            ir.EvalStmt(push_back_call, span),
        ],
        span,
    )
    aiv_func = ir.Function(
        "aiv_add_residual",
        [residual_full],
        [],
        aiv_body,
        span,
        type=ir.FunctionType.AIV,
    )

    aic_call = ir.Call(ir.GlobalVar("aic_matmul"), [a, b, out_tensor], span)
    aiv_call = ir.Call(ir.GlobalVar("aiv_add_residual"), [residual_full], span)
    group_body = ir.SeqStmts(
        [
            ir.EvalStmt(aic_call, span),
            ir.EvalStmt(aiv_call, span),
        ],
        span,
    )
    group_func = ir.Function(
        "matmul_add_group",
        [a, b, residual_full, out_tensor],
        [],
        group_body,
        span,
        type=ir.FunctionType.Group,
    )
    return _program([aic_func, aiv_func, group_func])


def test_numerical_cross_core_matmul_residual_updown_split():
    """Cross-core matmul+residual with UpDown split=1, full-tile correctness.

    The Group runs as a bidirectional scheduler: AIC pushes split=1, both AIV
    subblocks pop their halves and add the matching half of residual, push
    back, AIC reassembles into the full output.
    """
    prog = _build_cross_core_matmul_residual_program(split=1)
    code = torch_codegen(prog)

    ns: dict = {}
    exec(code, ns)  # noqa: S102

    t_a = torch.randn(4, 4)
    t_b = torch.randn(4, 4)
    t_residual = torch.randn(4, 4)
    t_out = torch.zeros(4, 4)

    ns["matmul_add_group"](t_a, t_b, t_residual, t_out)
    expected = torch.matmul(t_a, t_b).float() + t_residual
    assert torch.allclose(t_out, expected, atol=1e-5)


def test_numerical_cross_core_matmul_residual_leftright_split():
    """Cross-core matmul+residual with LeftRight split=2, full-tile correctness."""
    prog = _build_cross_core_matmul_residual_program(split=2)
    code = torch_codegen(prog)

    ns: dict = {}
    exec(code, ns)  # noqa: S102

    t_a = torch.randn(4, 4)
    t_b = torch.randn(4, 4)
    t_residual = torch.randn(4, 4)
    t_out = torch.zeros(4, 4)

    ns["matmul_add_group"](t_a, t_b, t_residual, t_out)
    expected = torch.matmul(t_a, t_b).float() + t_residual
    assert torch.allclose(t_out, expected, atol=1e-5)


def test_nested_tpop_in_expression():
    """tpop used directly inside tile.add (nested Call) should work via dispatch workaround."""
    span = _span()

    tile = _tile_var("tile", [4, 4])
    residual = _tile_var("residual", [4, 4])
    out = _tile_var("out", [4, 4])

    # Build: out = tile.add(tpop_from_aic(), residual) — tpop nested inside add
    pop_op = ir.get_op("tile.tpop_from_aic")
    pop_call = ir.Call(pop_op, [], {"split": 0}, ir.TileType([4, 4], DataType.FP32), span)
    add_call = _op_call("tile.add", [pop_call, residual])

    # Push first, then pop inside add
    push_call = _op_call("tile.tpush_to_aiv", [tile], {"split": 0})
    body = ir.SeqStmts(
        [
            ir.EvalStmt(push_call, span),
            ir.AssignStmt(out, add_call, span),
            ir.ReturnStmt([out], span),
        ],
        span,
    )
    func = _simple_function("f", [tile, residual], body, [ir.TileType([4, 4], DataType.FP32)])
    code = torch_codegen(func)

    # Verify nested tpop is generated correctly
    assert "_tpop_from_aic(" in code

    # Execute and verify numerical correctness
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    t = torch.randn(4, 4)
    r = torch.randn(4, 4)
    result_val = ns["f"](t, r)
    assert torch.allclose(result_val, t + r, atol=1e-6)


def test_bidirectional_pipe_communication():
    """AIC → tpush_to_aiv → AIV tpop_from_aic → tpush_to_aic → AIC tpop_from_aiv: bidirectional pipes."""
    span = _span()

    tile = _tile_var("tile", [4, 4])
    out = _tile_var("out", [4, 4])

    # Step 1: push to to_aiv pipe (AIC→AIV direction)
    push_to_aiv = _op_call("tile.tpush_to_aiv", [tile], {"split": 0})
    # Step 2: pop from to_aiv pipe (AIV reads data from AIC)
    pop_from_aic = ir.Call(
        ir.get_op("tile.tpop_from_aic"), [], {"split": 0}, ir.TileType([4, 4], DataType.FP32), span
    )
    mid = _tile_var("mid", [4, 4])
    # Step 3: push to to_aic pipe (AIV→AIC direction)
    push_to_aic = _op_call("tile.tpush_to_aic", [mid], {"split": 0})
    # Step 4: pop from to_aic pipe (AIC reads data from AIV)
    pop_from_aiv = ir.Call(
        ir.get_op("tile.tpop_from_aiv"), [], {"split": 0}, ir.TileType([4, 4], DataType.FP32), span
    )

    body = ir.SeqStmts(
        [
            ir.EvalStmt(push_to_aiv, span),
            ir.AssignStmt(mid, pop_from_aic, span),
            ir.EvalStmt(push_to_aic, span),
            ir.AssignStmt(out, pop_from_aiv, span),
            ir.ReturnStmt([out], span),
        ],
        span,
    )
    func = _simple_function("f", [tile], body, [ir.TileType([4, 4], DataType.FP32)])
    code = torch_codegen(func)

    assert "_pipes['to_aiv']" in code
    assert "_pipes['to_aic']" in code

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    t = torch.randn(4, 4)
    result_val = ns["f"](t)
    assert torch.allclose(result_val, t, atol=1e-6)


def test_pipe_empty_after_balanced_pushpop():
    """After pushing N times and popping N times, the pipe should be empty."""
    span = _span()

    t1 = _tile_var("t1", [4, 4])
    t2 = _tile_var("t2", [4, 4])
    o1 = _tile_var("o1", [4, 4])
    o2 = _tile_var("o2", [4, 4])

    push1 = _op_call("tile.tpush_to_aiv", [t1], {"split": 0})
    push2 = _op_call("tile.tpush_to_aiv", [t2], {"split": 0})
    pop1 = ir.Call(
        ir.get_op("tile.tpop_from_aic"), [], {"split": 0}, ir.TileType([4, 4], DataType.FP32), span
    )
    pop2 = ir.Call(
        ir.get_op("tile.tpop_from_aic"), [], {"split": 0}, ir.TileType([4, 4], DataType.FP32), span
    )

    body = ir.SeqStmts(
        [
            ir.EvalStmt(push1, span),
            ir.EvalStmt(push2, span),
            ir.AssignStmt(o1, pop1, span),
            ir.AssignStmt(o2, pop2, span),
            ir.ReturnStmt([o1, o2], span),
        ],
        span,
    )
    func = _simple_function(
        "f", [t1, t2], body, [ir.TileType([4, 4], DataType.FP32), ir.TileType([4, 4], DataType.FP32)]
    )
    code = torch_codegen(func)

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    r1, r2 = ns["f"](a, b)

    # FIFO order: first pushed = first popped
    assert torch.allclose(r1, a, atol=1e-6)
    assert torch.allclose(r2, b, atol=1e-6)

    # Pipe should be empty after balanced push/pop
    assert len(ns["_pipes"]["to_aiv"]) == 0


def test_tpop_from_aiv_split_keeps_full_tile_with_shape_checks():
    """AIC-side tpop_from_aiv must keep full-tile shape under split mode."""
    span = _span()

    tile = _tile_var("tile", [4, 4])
    out = _tile_var("out", [4, 4])

    push_call = _op_call("tile.tpush_to_aic", [tile], {"split": 1})
    pop_call = ir.Call(
        ir.get_op("tile.tpop_from_aiv"), [], {"split": 1}, ir.TileType([4, 4], DataType.FP32), span
    )

    body = ir.SeqStmts(
        [
            ir.EvalStmt(push_call, span),
            ir.AssignStmt(out, pop_call, span),
            ir.ReturnStmt([out], span),
        ],
        span,
    )
    func = _simple_function("f", [tile], body, [ir.TileType([4, 4], DataType.FP32)])
    code = torch_codegen(func, check_shapes=True)

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    t = torch.randn(4, 4)
    result = ns["f"](t)
    assert torch.allclose(result, t, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: edge cases and error handling
# ---------------------------------------------------------------------------


def test_invalid_split_mode_fallback():
    """Invalid split_mode (not 0, 1, 2) should fallback to no-split behavior."""
    span = _span()

    # Test split=3 (invalid)
    tile = _tile_var("tile", [4, 4])
    push_call = _op_call("tile.tpush_to_aiv", [tile], {"split": 3})
    body = ir.EvalStmt(push_call, span)
    func = _simple_function("f", [tile], body)
    code = torch_codegen(func)

    # Should fallback to no-split: push 1 chunk
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    t = torch.randn(4, 4)
    ns["f"](t)
    # Pipe should have 1 element (not 2 like split=1 or split=2)
    assert len(ns["_pipes"]["to_aiv"]) == 1


def test_pop_from_empty_pipe_raises():
    """Popping from an empty pipe should raise IndexError."""
    span = _span()

    # Function that only pops without pushing
    pop_call = _op_call("tile.tpop_from_aic", [], {"split": 0})
    out = _tile_var("out", [4, 4])
    body = ir.AssignStmt(out, pop_call, span)
    func = _simple_function("f", [], body, [ir.TileType([4, 4], DataType.FP32)])
    code = torch_codegen(func)

    ns: dict = {}
    exec(code, ns)  # noqa: S102

    # Should raise IndexError when popping from empty pipe
    with pytest.raises(IndexError, match="pop from an empty"):
        ns["f"]()


def test_unbalanced_push_pop_pipe_state():
    """Push 2 times, pop 1 time: pipe should have 1 element remaining."""
    span = _span()

    t1 = _tile_var("t1", [4, 4])
    t2 = _tile_var("t2", [4, 4])
    out = _tile_var("out", [4, 4])

    push1 = _op_call("tile.tpush_to_aiv", [t1], {"split": 0})
    push2 = _op_call("tile.tpush_to_aiv", [t2], {"split": 0})
    pop = ir.Call(ir.get_op("tile.tpop_from_aic"), [], {"split": 0}, ir.TileType([4, 4], DataType.FP32), span)

    body = ir.SeqStmts(
        [
            ir.EvalStmt(push1, span),
            ir.EvalStmt(push2, span),
            ir.AssignStmt(out, pop, span),
            ir.ReturnStmt([out], span),
        ],
        span,
    )
    func = _simple_function("f", [t1, t2], body, [ir.TileType([4, 4], DataType.FP32)])
    code = torch_codegen(func)

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    result = ns["f"](a, b)

    # First pushed should be popped (FIFO)
    assert torch.allclose(result, a, atol=1e-6)

    # Pipe should have 1 element remaining (second push)
    assert len(ns["_pipes"]["to_aiv"]) == 1
    # The remaining element should be the second pushed
    assert torch.allclose(ns["_pipes"]["to_aiv"][0], b, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
