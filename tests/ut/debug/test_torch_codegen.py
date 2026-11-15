# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for PyTorch code emission from PyPTO IR."""

import importlib
from typing import Any

import pypto.language as pl
import pytest
import torch
from pypto import DataType, ir
from pypto.debug import torch_codegen
from pypto.debug.torch_codegen import TorchCodegen

torch_codegen_module = importlib.import_module("pypto.debug.torch_codegen")

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


def test_tensor_view_codegen_and_execution():
    """tensor.view should preserve logical shape and layout semantics."""
    src = _tensor_var("src", [2, 4])
    reshaped = _tensor_var("reshaped", [4, 2])
    transposed = ir.Var(
        "transposed",
        ir.op.tensor.view(src, layout=ir.TensorLayout.DN).type,
        _span(),
    )
    shape_call = ir.op.tensor.view(src, [4, 2])
    layout_call = ir.op.tensor.view(src, layout=ir.TensorLayout.DN)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(reshaped, shape_call, _span()),
            ir.AssignStmt(transposed, layout_call, _span()),
            ir.ReturnStmt([reshaped, transposed], _span()),
        ],
        _span(),
    )
    func = _simple_function("view_main", [src], body, [reshaped.type, transposed.type])

    code = torch_codegen(func)
    assert "_tensor_view(src, (4, 2), False)" in code
    assert "src.mT" in code

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    actual_reshape, actual_transpose = ns["view_main"](value)
    assert torch.equal(actual_reshape, value.reshape(4, 2))
    assert torch.equal(actual_transpose, value.mT)


def test_tensor_view_shape_and_layout_uses_target_stride():
    """A combined shape/layout view should use the target canonical stride."""
    src = _tensor_var("src", [2, 4])
    result = ir.op.tensor.view(src, [4, 2], layout=ir.TensorLayout.DN)
    out = ir.Var("out", result.type, _span())
    body = ir.SeqStmts(
        [ir.AssignStmt(out, result, _span()), ir.ReturnStmt([out], _span())],
        _span(),
    )
    func = _simple_function("combined_view", [src], body, [out.type])

    code = torch_codegen(func)
    assert "_tensor_view(src, (4, 2), True)" in code

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    actual = ns["combined_view"](value)
    expected = torch.as_strided(value, (4, 2), (1, 4))
    assert torch.equal(actual, expected)
    assert actual.stride() == (1, 4)


def test_tensor_view_explicit_valid_shape_codegen_and_execution():
    """A three-argument view should reshape and preserve target validity metadata."""
    src_view = ir.TensorView(
        [_int(2048), _int(128), _int(1)],
        ir.TensorLayout.ND,
        valid_shape=[_int(1), _int(16), _int(128)],
    )
    src = ir.Var("src", ir.TensorType([2, 16, 128], DataType.FP32, None, src_view), _span())
    result = ir.op.tensor.view(src, [32, 128], [16, 128])
    out = ir.Var("out", result.type, _span())
    body = ir.SeqStmts(
        [ir.AssignStmt(out, result, _span()), ir.ReturnStmt([out], _span())],
        _span(),
    )
    func = _simple_function("partial_view", [src], body, [out.type])

    code = torch_codegen(func)
    assert "_tensor_view(src, (32, 128), False, (16, 128))" in code

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    value = torch.arange(4096, dtype=torch.float32).reshape(2, 16, 128)
    actual = ns["partial_view"](value)
    assert actual.shape == (32, 128)
    assert actual.stride() == (128, 1)
    assert actual._pypto_valid_shape == (16, 128)
    assert actual._pypto_full_shape == (32, 128)


def test_tensor_view_same_layout_is_identity():
    """A layout-only view targeting the current layout should be an identity."""
    src = _tensor_var("src", [2, 4])
    result = ir.op.tensor.view(src, layout=ir.TensorLayout.ND)
    out = ir.Var("out", result.type, _span())
    func = _simple_function("same_layout", [src], ir.AssignStmt(out, result, _span()))

    code = torch_codegen(func)
    assert "out = src" in code


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


@pytest.mark.parametrize("mode", ["round_even", "round_away"])
@pytest.mark.parametrize(
    "op_name", ["tensor.matmul", "tensor.matmul_acc", "tile.matmul", "tile.matmul_acc", "tile.matmul_bias"]
)
def test_hf32_matmul_reference_rounds_inputs_per_operation(mode, op_name):
    """HF32 must round both inputs, while bias/accumulator and later FP32 ops stay exact."""
    tensor_op = op_name.startswith("tensor.")
    var = _tensor_var if tensor_op else _tile_var
    a, b = var("a", [2, 2]), var("b", [2, 2])
    seed = var("seed", [1, 2] if op_name.endswith("bias") else [2, 2])
    init = _scalar("init", DataType.BOOL)
    params = [a, b]
    args: list[ir.Expr] = [a, b]
    if op_name.endswith("acc"):
        args = [seed, a, b, init]
        params += [seed, init]
    elif op_name.endswith("bias"):
        args += [seed]
        params += [seed]
    # Exercise transposed, noncontiguous Tensor inputs as well as Tile handlers.
    kwargs = {"a_trans": True, "b_trans": True} if tensor_op else {}
    call = _op_call(op_name, args, {**kwargs, "tf32_mode": mode})
    result, plain = var("result", [2, 2]), var("plain", [2, 2])
    body = ir.SeqStmts(
        [
            ir.AssignStmt(result, call, _span()),
            ir.AssignStmt(plain, _op_call(op_name.split(".")[0] + ".matmul", [a, b], kwargs), _span()),
            ir.ReturnStmt([result, plain], _span()),
        ],
        _span(),
    )
    namespace = {}
    exec(torch_codegen(_simple_function("hf32", params, body, [result.type, plain.type])), namespace)  # noqa: S102
    lhs = torch.tensor([[1 + 2**-11, -1], [-1 - 3 * 2**-11, 1]], dtype=torch.float32)
    rhs = torch.tensor([[1024, 1 + 2**-11], [1024, 1 + 3 * 2**-11]], dtype=torch.float32)
    # Hand-rounded ties include positive/negative values and even/odd retained LSBs.
    rounded_one = 1 if mode == "round_even" else 1 + 2**-10
    rounded_lhs = torch.tensor([[rounded_one, -1], [-1 - 2**-9, 1]])
    rounded_rhs = torch.tensor([[1024, rounded_one], [1024, 1 + 2**-9]])
    product = rounded_lhs @ rounded_rhs
    plain_expected = lhs @ rhs
    seed_data = torch.full((1, 2) if op_name.endswith("bias") else (2, 2), 3 + 2**-11)
    for initialize in (False, True):
        inputs: list[Any] = [lhs.T, rhs.T] if tensor_op else [lhs, rhs]
        expected = product
        if op_name.endswith("acc"):
            inputs += [seed_data, initialize]
            expected = product if initialize else seed_data + product
        elif op_name.endswith("bias"):
            inputs += [seed_data]
            expected = product + seed_data
        actual, plain_actual = namespace["hf32"](*inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(plain_actual, plain_expected, rtol=0, atol=0)
        assert actual.dtype == torch.float32


@pytest.mark.parametrize("mode", ["round_even", "round_away"])
def test_hf32_rounding_boundaries(mode):
    """Rounding must preserve special values and carry across fraction/exponent boundaries."""
    namespace = {}
    exec(torch_codegen_module._PREAMBLE, namespace)  # noqa: S102
    values = torch.tensor(
        [
            0.0,
            -0.0,
            float("inf"),
            -float("inf"),
            float("nan"),
            2 - 2**-11,
            1 + 2**-12,
            1 + 3 * 2**-12,
            -1 - 2**-11,
            -1 - 3 * 2**-11,
        ]
    )
    negative_tie = -1 if mode == "round_even" else -1 - 2**-10
    expected = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 2, 1, 1 + 2**-10, negative_tie, -1 - 2**-9]
    )
    actual = namespace["_hf32_round"](values, mode)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(actual[:2]), torch.signbit(values[:2]))


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


def test_tile_row_sum_row_major_pads_to_declared_shape_and_executes():
    """tile.row_sum(out_layout=row_major) pads the generated expr to [rows, pad_cols] (issue #3319).

    DeduceTileRowReductionType declares [rows, pad_cols] for out_layout=row_major;
    without padding, the generated plain `.sum(dim=-1, keepdim=True)` would only
    produce [rows, 1] and fail the TileType shape check.
    """
    a = _tile_var("a", [4, 48])
    tmp = _tile_var("tmp", [4, 48])
    out = _tile_var("out", [4, 8])
    call = _op_call("tile.row_sum", [a, tmp], {"out_layout": ir.TileLayout.row_major})
    body = ir.SeqStmts(
        [ir.AssignStmt(out, call, _span()), ir.ReturnStmt([out], _span())],
        _span(),
    )
    func = _simple_function("f", [a, tmp], body, [out.type])
    code = torch_codegen(func)
    assert "torch.nn.functional.pad(" in code

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    value = torch.randn(4, 48)
    scratch = torch.zeros(4, 48)
    result = ns["f"](value, scratch)
    assert tuple(result.shape) == (4, 8)
    torch.testing.assert_close(result[:, 0:1], value.sum(dim=-1, keepdim=True))


def test_tile_row_max_row_major_survives_result_named_max():
    """A result literally named ``max`` must not shadow the builtin used to pad it.

    `_unique_name()` is free to pick any valid identifier, including one that
    shadows a builtin -- `max = pl.tile.row_max(...)` is legal PyPTO source.
    A bare `max(...)` call in the generated padding expression would then
    resolve to the just-assigned local (a Tensor), not the builtin, and
    raise instead of padding.
    """
    a = _tile_var("a", [4, 48])
    tmp = _tile_var("tmp", [4, 48])
    out = _tile_var("max", [4, 8])
    call = _op_call("tile.row_max", [a, tmp], {"out_layout": ir.TileLayout.row_major})
    body = ir.SeqStmts(
        [ir.AssignStmt(out, call, _span()), ir.ReturnStmt([out], _span())],
        _span(),
    )
    func = _simple_function("f", [a, tmp], body, [out.type])
    code = torch_codegen(func)

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    value = torch.randn(4, 48)
    scratch = torch.zeros(4, 48)
    result = ns["f"](value, scratch)
    assert tuple(result.shape) == (4, 8)
    torch.testing.assert_close(result[:, 0:1], value.amax(dim=-1, keepdim=True))


def test_tile_row_sum_row_major_survives_result_named_torch():
    """A result literally named ``torch`` must not shadow the torch module.

    `_unique_name()` only avoids Python keywords, not the `torch` import --
    `torch = pl.tile.row_sum(...)` is legal PyPTO source. A bare
    `torch.nn.functional.pad(...)` call in the generated padding expression
    would then resolve `torch` to the just-assigned local (a Tensor) rather
    than the module, in the very statement assigning it, and raise
    UnboundLocalError instead of padding.
    """
    a = _tile_var("a", [4, 48])
    tmp = _tile_var("tmp", [4, 48])
    out = _tile_var("torch", [4, 8])
    call = _op_call("tile.row_sum", [a, tmp], {"out_layout": ir.TileLayout.row_major})
    body = ir.SeqStmts(
        [ir.AssignStmt(out, call, _span()), ir.ReturnStmt([out], _span())],
        _span(),
    )
    func = _simple_function("f", [a, tmp], body, [out.type])
    code = torch_codegen(func)

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    value = torch.randn(4, 48)
    scratch = torch.zeros(4, 48)
    result = ns["f"](value, scratch)
    assert tuple(result.shape) == (4, 8)
    torch.testing.assert_close(result[:, 0:1], value.sum(dim=-1, keepdim=True))


def test_tile_row_max_row_major_pads_by_ir_dtype_not_torch_carrier():
    """Pad width comes from the IR dtype, not the (possibly wider) Torch carrier.

    UINT16's IR element width is 2 bytes, so out_layout=row_major pads to 16
    columns (32 / 2). Torch has no uint16, so _DTYPE_MAP upcasts it to
    torch.int32 -- a 4-byte carrier. Deriving the pad width from that carrier
    at runtime (``element_size()``) would wrongly compute 8 columns instead
    of the 16 DeduceTileRowReductionType actually declared (issue #3319).
    """
    a = _tile_var("a", [4, 48], DataType.UINT16)
    tmp = _tile_var("tmp", [4, 48], DataType.UINT16)
    call = _op_call("tile.row_max", [a, tmp], {"out_layout": ir.TileLayout.row_major})
    out = ir.Var("out", call.type, _span())
    body = ir.SeqStmts(
        [ir.AssignStmt(out, call, _span()), ir.ReturnStmt([out], _span())],
        _span(),
    )
    func = _simple_function("f", [a, tmp], body, [out.type])
    code = torch_codegen(func)

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    value = torch.randint(0, 100, (4, 48), dtype=torch.int32)
    scratch = torch.zeros(4, 48, dtype=torch.int32)
    result = ns["f"](value, scratch)
    assert tuple(result.shape) == (4, 16)
    torch.testing.assert_close(result[:, 0:1], value.amax(dim=-1, keepdim=True))


# ---------------------------------------------------------------------------
# Test: tile ops
# ---------------------------------------------------------------------------


def test_tile_load_store():
    """tile.load and tile.store should emit _tile_load/_tile_store helpers."""
    tensor = _tensor_var("t", [256, 256])
    offsets = _make_tuple(_int(0), _int(0))
    shapes = _make_tuple(_int(64), _int(64))
    valid_shape = _make_tuple(_int(64), _int(64))
    tile = _tile_var("tile", [64, 64])
    output = _tensor_var("out", [256, 256])
    off2 = _make_tuple(_int(64), _int(0))

    load_call = _op_call(
        "tile.load",
        [tensor, offsets, shapes, valid_shape],
        {"target_memory": ir.MemorySpace.Vec},
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


def test_tile_transpose_view():
    """tile.transpose_view should emit .mT (matrix transpose of the trailing dims)."""
    src = _tile_var("src", [64, 32])
    view = _tile_var("view", [32, 64])

    call = _op_call("tile.transpose_view", [src])
    assign = ir.AssignStmt(view, call, _span())
    func = _simple_function("f", [src], assign)
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


@pytest.mark.parametrize(
    ("op_name", "expected_expr"),
    [
        ("addc", "(a + b + carry)"),
        ("subc", "(a - b + carry)"),
        ("addsc", "(a + 2.0 + carry)"),
        ("subsc", "(a - 2.0 + carry)"),
    ],
)
def test_tile_carry_codegen_and_execution(op_name, expected_expr):
    """Carry-in debug references use ``+ carry`` for add and subtract forms."""
    a = _tile_var("a", [2, 2])
    carry = _tile_var("carry", [2, 2])
    out = _tile_var("out", [2, 2])
    if op_name in {"addc", "subc"}:
        b = _tile_var("b", [2, 2])
        args: list[ir.Expr] = [a, b, carry]
        params: list[ir.Var] = [a, b, carry]
    else:
        args = [a, _float(2.0), carry]
        params = [a, carry]

    call = _op_call(f"tile.{op_name}", args)
    body = ir.SeqStmts(
        [ir.AssignStmt(out, call, _span()), ir.ReturnStmt([out], _span())],
        _span(),
    )
    func = _simple_function("carry_ref", params, body, [out.type])
    code = torch_codegen(func)
    assert expected_expr in code

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    a_value = torch.tensor([[5.0, -3.0], [8.0, 1.0]])
    carry_value = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    if op_name in {"addc", "subc"}:
        b_value = torch.tensor([[2.0, 4.0], [-1.0, 3.0]])
        actual = ns["carry_ref"](a_value, b_value, carry_value)
        expected = a_value + b_value + carry_value if op_name == "addc" else a_value - b_value + carry_value
    else:
        actual = ns["carry_ref"](a_value, carry_value)
        expected = a_value + 2.0 + carry_value if op_name == "addsc" else a_value - 2.0 + carry_value
    assert torch.equal(actual, expected)


# (op leaf, expected torch snippet) for the integer bitwise / shift family. The
# reference map registers these for the `tensor` and `tile` prefixes from one shared
# loop, so both namespaces are asserted below (issue #2216).
_BITWISE_REFERENCE = [
    ("and", "torch.bitwise_and(a, b)"),
    ("or", "torch.bitwise_or(a, b)"),
    ("xor", "torch.bitwise_xor(a, b)"),
    ("shl", "(a << b)"),
    ("shr", "(a >> b)"),
]


@pytest.mark.parametrize(("op_leaf", "expected"), _BITWISE_REFERENCE)
def test_tensor_bitwise_ops(op_leaf, expected):
    """Tensor bitwise/shift ops have a torch reference implementation."""
    a = _tensor_var("a", [64, 64], DataType.INT16)
    b = _tensor_var("b", [64, 64], DataType.INT16)
    out = _tensor_var("out", [64, 64], DataType.INT16)

    call = _op_call(f"tensor.{op_leaf}", [a, b])
    func = _simple_function("f", [a, b], ir.AssignStmt(out, call, _span()))
    assert expected in torch_codegen(func)


@pytest.mark.parametrize(("op_leaf", "expected"), _BITWISE_REFERENCE)
def test_tile_bitwise_ops_still_covered(op_leaf, expected):
    """Moving the tile entries into the shared prefix loop must not drop them."""
    a = _tile_var("a", [64, 64], DataType.INT16)
    b = _tile_var("b", [64, 64], DataType.INT16)
    out = _tile_var("out", [64, 64], DataType.INT16)

    # tile.xor takes a trailing scratch operand that the reference ignores.
    if op_leaf == "xor":
        tmp = _tile_var("tmp", [64, 64], DataType.INT16)
        call = _op_call("tile.xor", [a, b, tmp])
        func = _simple_function("f", [a, b, tmp], ir.AssignStmt(out, call, _span()))
    else:
        call = _op_call(f"tile.{op_leaf}", [a, b])
        func = _simple_function("f", [a, b], ir.AssignStmt(out, call, _span()))
    assert expected in torch_codegen(func)


@pytest.mark.parametrize(
    ("op_leaf", "expected"),
    [
        ("ands", "torch.bitwise_and(a, 3)"),
        ("ors", "torch.bitwise_or(a, 3)"),
        ("xors", "torch.bitwise_xor(a, 3)"),
        ("shls", "(a << 3)"),
        ("shrs", "(a >> 3)"),
    ],
)
def test_tensor_bitwise_scalar_ops(op_leaf, expected):
    """The tensor-scalar bitwise/shift forms have a torch reference too."""
    a = _tensor_var("a", [64], DataType.INT16)
    out = _tensor_var("out", [64], DataType.INT16)

    call = _op_call(f"tensor.{op_leaf}", [a, ir.ConstInt(3, DataType.INT16, _span())])
    func = _simple_function("f", [a], ir.AssignStmt(out, call, _span()))
    assert expected in torch_codegen(func)


def test_bitwise_not_reference():
    """tensor.not / tile.not map to torch.bitwise_not."""
    for prefix, var_factory in (("tensor", _tensor_var), ("tile", _tile_var)):
        a = var_factory("a", [64], DataType.INT16)
        out = var_factory("out", [64], DataType.INT16)
        call = _op_call(f"{prefix}.not", [a])
        func = _simple_function("f", [a], ir.AssignStmt(out, call, _span()))
        assert "torch.bitwise_not(a)" in torch_codegen(func)


def test_tile_sels_and_prelu():
    """Selection and PReLU debug codegen must ignore scratch operands."""
    mask = _tile_var("mask", [16, 32], DataType.UINT8)
    src = _tile_var("src", [16, 16])
    slope = _tile_var("slope", [16, 16])
    sels_tmp = _tile_var("sels_tmp", [1, 32], DataType.UINT8)
    prelu_tmp = _tile_var("prelu_tmp", [17, 32], DataType.UINT8)
    sels_out = _tile_var("sels_out", [16, 16])
    prelu_out = _tile_var("prelu_out", [16, 16])

    sels_call = _op_call("tile.sels", [mask, src, sels_tmp, _float(-1.0)])
    prelu_call = _op_call("tile.prelu", [src, slope, prelu_tmp])
    body = ir.SeqStmts(
        [
            ir.AssignStmt(sels_out, sels_call, _span()),
            ir.AssignStmt(prelu_out, prelu_call, _span()),
        ],
        _span(),
    )
    func = _simple_function("f", [mask, src, slope, sels_tmp, prelu_tmp], body)

    code = torch_codegen(func)

    assert "torch.where(mask, src, -1.0)" in code
    assert "torch.where(src > 0, src, src * slope)" in code


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


def test_tile_matmul_acc_runtime_init_cond_overwrites():
    """A runtime init_cond must overwrite the accumulator, not accumulate into it."""

    @pl.program
    class SplitKFromNonZeroAcc:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            seed: pl.Tensor[[16, 16], pl.FP32],
            lhs: pl.Tensor[[16, 32], pl.FP32],
            rhs: pl.Tensor[[32, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            # A *non-zero* accumulator is what separates overwrite from
            # accumulate; the tile.create zero seed the compiler emits today
            # makes the two coincide. Seeding it with a matmul both keeps the
            # accumulator in Acc and gives it a value the k0 == 0 step must drop.
            seed_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
                seed, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
            )
            head: pl.Tile[[16, 16], pl.FP32] = pl.load(
                lhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
            )
            acc_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(seed_tile, head)
            for k0 in pl.range(0, 32, 16):
                a: pl.Tile[[16, 16], pl.FP32] = pl.load(
                    lhs, [0, k0], [16, 16], target_memory=pl.MemorySpace.Mat
                )
                b: pl.Tile[[16, 16], pl.FP32] = pl.load(
                    rhs, [k0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
                )
                acc_tile = pl.matmul_acc(acc_tile, a, b, init_cond=(k0 == 0))
            return pl.store(acc_tile, [0, 0], output)

    code = torch_codegen(SplitKFromNonZeroAcc)
    assert "_acc_init(acc_tile, torch.matmul(a, b).float(), (k0 == 0))" in code

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    seed = torch.full((16, 16), 3.0)
    lhs = torch.ones(16, 32)
    rhs = torch.ones(32, 16)
    out = torch.zeros(16, 16)
    ns["kernel"](seed, lhs, rhs, out)

    # k0 == 0 overwrites the seeded accumulator; k0 == 16 accumulates into it.
    overwrite = lhs[:, :16] @ rhs[:16, :] + lhs[:, 16:] @ rhs[16:, :]
    assert torch.allclose(out, overwrite)
    # Dropping the predicate would fold the seed product in instead.
    assert not torch.allclose(out, seed @ lhs[:, :16] + overwrite)


def test_tile_matmul_acc_literal_init_cond_folds():
    """A literal init_cond picks one arm outright, mirroring the pto backend."""

    @pl.program
    class MatmulAccInitTrue:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tensor[[16, 16], pl.FP32],
            rhs: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            a: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            b: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            acc: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(a, b)
            out_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.matmul_acc(acc, a, b, init_cond=True)
            return pl.store(out_tile, [0, 0], output)

    @pl.program
    class MatmulAccInitFalse:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tensor[[16, 16], pl.FP32],
            rhs: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            a: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            b: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            acc: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(a, b)
            out_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.matmul_acc(acc, a, b, init_cond=False)
            return pl.store(out_tile, [0, 0], output)

    # True selects the overwrite arm outright -- the accumulator drops out.
    assert "out_tile = torch.matmul(a, b).float()" in torch_codegen(MatmulAccInitTrue)
    # False selects the accumulating arm, identical to carrying no predicate.
    assert "out_tile = (acc + torch.matmul(a, b).float())" in torch_codegen(MatmulAccInitFalse)


def test_tile_gemv_acc_runtime_init_cond_overwrites():
    """GEMV carries the same predicate as matmul, so the reference must honour it.

    ``tile.gemv_acc`` shares the ``_tile_matmul_acc`` handler -- GEMV is a matmul
    whose M is 1 on the same cube MAD. This pins that the shared handler reads
    the predicate for GEMV too, rather than modelling an overwrite as an
    accumulate.
    """

    @pl.program
    class GemvSplitKFromNonZeroAcc:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            seed: pl.Tensor[[1, 32], pl.FP32],
            seed_rhs: pl.Tensor[[32, 16], pl.FP32],
            lhs: pl.Tensor[[1, 64], pl.FP32],
            rhs: pl.Tensor[[64, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
        ) -> pl.Tensor[[1, 16], pl.FP32]:
            # Seed with a plain gemv so the accumulator is both Acc-resident and
            # non-zero -- a zero seed would make overwrite and accumulate agree.
            seed_tile = pl.load(seed, [0, 0], [1, 32], target_memory=pl.MemorySpace.Mat)
            seed_rhs_tile = pl.load(seed_rhs, [0, 0], [32, 16], target_memory=pl.MemorySpace.Mat)
            acc_tile = pl.tile.gemv(seed_tile, seed_rhs_tile)
            for k0 in pl.range(0, 64, 32):
                a = pl.load(lhs, [0, k0], [1, 32], target_memory=pl.MemorySpace.Mat)
                b = pl.load(rhs, [k0, 0], [32, 16], target_memory=pl.MemorySpace.Mat)
                acc_tile = pl.tile.gemv_acc(acc_tile, a, b, init_cond=(k0 == 0))
            return pl.store(acc_tile, [0, 0], output)

    code = torch_codegen(GemvSplitKFromNonZeroAcc)
    assert "_acc_init(acc_tile, torch.matmul(a, b).float(), (k0 == 0))" in code

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    seed = torch.full((1, 32), 3.0)
    seed_rhs = torch.full((32, 16), 2.0)
    lhs = torch.ones(1, 64)
    rhs = torch.ones(64, 16)
    out = torch.zeros(1, 16)
    ns["kernel"](seed, seed_rhs, lhs, rhs, out)

    # k0 == 0 overwrites the seeded accumulator; k0 == 32 accumulates into it.
    overwrite = lhs[:, :32] @ rhs[:32, :] + lhs[:, 32:] @ rhs[32:, :]
    assert torch.allclose(out, overwrite)
    # Dropping the predicate would fold the seed product in instead.
    assert not torch.allclose(out, seed @ seed_rhs + overwrite)


def test_tensor_matmul_acc_init_cond_with_transpose():
    """tensor.matmul_acc must honour init_cond alongside a_trans/b_trans."""

    @pl.program
    class TensorSplitK:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            acc: pl.Tensor[[16, 16], pl.FP32],
            # a_trans=True, so lhs is [K=32, M=16] and the product is [16, 16].
            lhs: pl.Tensor[[32, 16], pl.FP32],
            rhs: pl.Tensor[[32, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            result: pl.Tensor[[16, 16], pl.FP32] = acc
            for k0 in pl.range(0, 2):
                result = pl.matmul_acc(result, lhs, rhs, a_trans=True, init_cond=(k0 == 0))
            return pl.assemble(output, result, [0, 0])

    code = torch_codegen(TensorSplitK)
    assert "_acc_init(result, torch.matmul(lhs.mT, rhs), (k0 == 0))" in code


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


@pytest.mark.parametrize(
    "op_name",
    [
        "system.sync_src",
        "system.sync_set",
        "system.sync_wait",
        "system.set_flag",
        "system.wait_flag",
        "system.set_ffts",
    ],
)
def test_system_ops_are_noops(op_name):
    """System ops should emit no-op comments."""
    params = []
    if op_name == "system.sync_src":
        call = _op_call(op_name, [], {"set_pipe": 4, "wait_pipe": 5, "event_id": 0})
    elif op_name in ("system.sync_set", "system.sync_wait"):
        call = _op_call(op_name, [], {"pipe": ir.PipeType.MTE2, "event_id": 0})
    elif op_name in ("system.set_flag", "system.wait_flag"):
        call = _op_call(
            op_name,
            [],
            {"src_pipe": ir.PipeType.MTE2, "dst_pipe": ir.PipeType.V, "event_id": 0},
        )
    else:
        workspace = _tensor_var("workspace", [256], DataType.INT64)
        params = [workspace]
        call = _op_call(op_name, [workspace])
    body = ir.EvalStmt(call, _span())
    func = _simple_function("f", params, body)
    code = torch_codegen(func)
    assert f"# {op_name.split('.')[-1]}" in code


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
    assert "_cross_core_rt.push_to_aiv(tile, 0)" in code


def test_get_subblock_idx():
    """tile.get_subblock_idx should emit thread-local lane helper."""
    idx = _scalar("idx", DataType.INDEX)
    call = _op_call("tile.get_subblock_idx", [])
    assign = ir.AssignStmt(idx, call, _span())
    ret = ir.ReturnStmt([idx], _span())
    body = ir.SeqStmts([assign, ret], _span())
    func = _simple_function("f", [], body, [ir.ScalarType(DataType.INDEX)])
    code = torch_codegen(func)
    assert "_get_subblock_idx()" in code
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    assert ns["f"]() == 0


def test_group_cross_core_split_runs_with_runtime_scheduler():
    """Group calls with split cross-core ops should route through _run_group_call."""

    @pl.program
    class SplitCrossCoreProgram:
        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
        def cross_aic(
            self,
            inp: pl.Tensor[[4, 2], pl.FP32],
            out: pl.Out[pl.Tensor[[4, 2], pl.FP32]],
        ):
            recv: pl.Tile[[4, 2], pl.FP32, pl.MemorySpace.Mat] = pl.tpop_from_aiv(
                shape=[4, 2], dtype=pl.FP32, split=1
            )
            out = pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
        def cross_aiv(
            self,
            inp: pl.Tensor[[4, 2], pl.FP32],
            out: pl.Out[pl.Tensor[[4, 2], pl.FP32]],
        ) -> pl.Tensor[[4, 2], pl.FP32]:
            lane: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
            tile: pl.Tile[[2, 2], pl.FP32, pl.MemorySpace.Vec] = pl.load(inp, [lane * 2, 0], [2, 2])
            pl.tpush_to_aic(tile, split=1)
            return out

        @pl.function(type=pl.FunctionType.Group, attrs={"split": pl.SplitMode.UP_DOWN})
        def cross(
            self,
            inp: pl.Tensor[[4, 2], pl.FP32],
            out: pl.Out[pl.Tensor[[4, 2], pl.FP32]],
        ) -> pl.Tensor[[4, 2], pl.FP32]:
            self.cross_aic(inp, out)
            result: pl.Tensor[[4, 2], pl.FP32] = self.cross_aiv(inp, out)
            return result

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            inp: pl.Tensor[[4, 2], pl.FP32],
            out: pl.Out[pl.Tensor[[4, 2], pl.FP32]],
        ) -> pl.Tensor[[4, 2], pl.FP32]:
            return self.cross(inp, out)

    code = torch_codegen(SplitCrossCoreProgram)
    assert "_run_group_call('cross'" in code
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    src = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    dst = torch.zeros_like(src)
    out = ns["main"](src, dst)
    assert torch.allclose(out, src)


def test_group_cross_core_left_right_c2v_runs_with_runtime_scheduler():
    """LEFT_RIGHT split C->V path should route through _run_group_call and preserve data."""

    @pl.program
    class SplitLeftRightC2VProgram:
        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def cross_aic(
            self,
            inp: pl.Tensor[[2, 4], pl.FP32],
            out: pl.Out[pl.Tensor[[2, 4], pl.FP32]],
        ):
            tile: pl.Tile[[2, 4], pl.FP32, pl.MemorySpace.Mat] = pl.load(
                inp, [0, 0], [2, 4], target_memory=pl.MemorySpace.Mat
            )
            pl.tpush_to_aiv(tile, split=2)

        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def cross_aiv(
            self,
            inp: pl.Tensor[[2, 4], pl.FP32],
            out: pl.Out[pl.Tensor[[2, 4], pl.FP32]],
        ) -> pl.Tensor[[2, 4], pl.FP32]:
            lane: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
            recv: pl.Tile[[2, 2], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(
                shape=[2, 2], dtype=pl.FP32, split=2
            )
            out = pl.store(recv, [0, lane * 2], out)
            return out

        @pl.function(type=pl.FunctionType.Group, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def cross(
            self,
            inp: pl.Tensor[[2, 4], pl.FP32],
            out: pl.Out[pl.Tensor[[2, 4], pl.FP32]],
        ) -> pl.Tensor[[2, 4], pl.FP32]:
            self.cross_aic(inp, out)
            result: pl.Tensor[[2, 4], pl.FP32] = self.cross_aiv(inp, out)
            return result

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            inp: pl.Tensor[[2, 4], pl.FP32],
            out: pl.Out[pl.Tensor[[2, 4], pl.FP32]],
        ) -> pl.Tensor[[2, 4], pl.FP32]:
            return self.cross(inp, out)

    code = torch_codegen(SplitLeftRightC2VProgram)
    assert "_run_group_call('cross'" in code
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    src = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    dst = torch.zeros_like(src)
    out = ns["main"](src, dst)
    assert torch.allclose(out, src)


@pytest.mark.parametrize("pipe_id", [0, 7])
def test_group_cross_core_no_split_bidirect_runs_with_runtime_scheduler(pipe_id):
    """No-split bidirectional path should run concurrently without deadlock."""

    @pl.program
    class NoSplitBidirectProgram:
        @pl.function(type=pl.FunctionType.AIC)
        def cross_aic(
            self,
            inp: pl.Tensor[[4, 2], pl.FP32],
            out: pl.Out[pl.Tensor[[4, 2], pl.FP32]],
        ):
            recv: pl.Tile[[4, 2], pl.FP32, pl.MemorySpace.Mat] = pl.tpop_from_aiv(
                shape=[4, 2], dtype=pl.FP32, split=0, id=pipe_id
            )
            twice: pl.Tile[[4, 2], pl.FP32, pl.MemorySpace.Mat] = pl.add(recv, recv)
            pl.tpush_to_aiv(twice, split=0, id=pipe_id)

        @pl.function(type=pl.FunctionType.AIV)
        def cross_aiv(
            self,
            inp: pl.Tensor[[4, 2], pl.FP32],
            out: pl.Out[pl.Tensor[[4, 2], pl.FP32]],
        ) -> pl.Tensor[[4, 2], pl.FP32]:
            tile: pl.Tile[[4, 2], pl.FP32, pl.MemorySpace.Vec] = pl.load(inp, [0, 0], [4, 2])
            pl.tpush_to_aic(tile, split=0, id=pipe_id)
            recv: pl.Tile[[4, 2], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(
                shape=[4, 2], dtype=pl.FP32, split=0, id=pipe_id
            )
            out = pl.store(recv, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Group)
        def cross(
            self,
            inp: pl.Tensor[[4, 2], pl.FP32],
            out: pl.Out[pl.Tensor[[4, 2], pl.FP32]],
        ) -> pl.Tensor[[4, 2], pl.FP32]:
            self.cross_aic(inp, out)
            result: pl.Tensor[[4, 2], pl.FP32] = self.cross_aiv(inp, out)
            return result

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            inp: pl.Tensor[[4, 2], pl.FP32],
            out: pl.Out[pl.Tensor[[4, 2], pl.FP32]],
        ) -> pl.Tensor[[4, 2], pl.FP32]:
            return self.cross(inp, out)

    code = torch_codegen(NoSplitBidirectProgram)
    assert "_run_group_call('cross'" in code
    if pipe_id:
        assert code.count(f"pipe_id={pipe_id})") == 4
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    src = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    dst = torch.zeros_like(src)
    out = ns["main"](src, dst)
    assert torch.allclose(out, src * 2)


@pytest.mark.parametrize("pipe_id", [0, 7])
@pytest.mark.parametrize("explicit_fp32", [False, True])
@pytest.mark.parametrize("check_shapes", [False, True])
def test_ordinary_groups_reset_c2v_epilogue(pipe_id, explicit_fp32, check_shapes):
    """A default pipe after BF16/ReLU preserves FP32 values across ordinary groups."""
    plain_options = ", c2v_dtype=pl.FP32" if explicit_fp32 else ""
    source = f"""@pl.program
class SequentialGroups:
    @pl.function(type=pl.FunctionType.AIC)
    def producer(self, a: pl.Tensor[[64, 128], pl.FP32],
                 b: pl.Tensor[[128, 128], pl.FP32],
                 out: pl.Out[pl.Tensor[[64, 128], pl.BF16]],
                 raw: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        peer = pl.import_peer_buffer(name="rounded", peer_func="consumer")
        pl.aic_initialize_pipe(c2v_consumer_buf=peer, dir_mask=1,
            slot_size=16384, slot_num=2, id={pipe_id}, c2v_dtype=pl.BF16, pre_relu=True)
        a_mat = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        left = pl.move(a_mat, target_memory=pl.Mem.Left)
        right = pl.move(b_mat, target_memory=pl.Mem.Right)
        acc = pl.matmul(left, right)
        pl.tpush_to_aiv(acc, split=0, id={pipe_id})
        pl.store(acc, [0, 0], raw)

    @pl.function(type=pl.FunctionType.AIV)
    def consumer(self, a: pl.Tensor[[64, 128], pl.FP32],
                 b: pl.Tensor[[128, 128], pl.FP32],
                 out: pl.Out[pl.Tensor[[64, 128], pl.BF16]],
                 raw: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        buf = pl.reserve_buffer(name="rounded", size=32768)
        pl.aiv_initialize_pipe(c2v_consumer_buf=buf, dir_mask=1,
            slot_size=16384, slot_num=2, id={pipe_id}, c2v_dtype=pl.BF16, pre_relu=True)
        value = pl.tpop_from_aic(shape=[64, 128], dtype=pl.BF16, split=0, id={pipe_id})
        pl.store(value, [0, 0], out)
        pl.tfree_to_aic(value, id={pipe_id})

    @pl.function(type=pl.FunctionType.Group)
    def rounded(self, a: pl.Tensor[[64, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 128], pl.BF16]],
                raw: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        self.producer(a, b, out, raw)
        self.consumer(a, b, out, raw)

    @pl.function(type=pl.FunctionType.AIC)
    def plain_producer(self, a: pl.Tensor[[64, 128], pl.FP32],
                       b: pl.Tensor[[128, 128], pl.FP32],
                       out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        peer = pl.import_peer_buffer(name="plain", peer_func="plain_consumer")
        pl.aic_initialize_pipe(c2v_consumer_buf=peer, dir_mask=1,
            slot_size=32768, slot_num=2, id={pipe_id}{plain_options})
        a_mat = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        left = pl.move(a_mat, target_memory=pl.Mem.Left)
        right = pl.move(b_mat, target_memory=pl.Mem.Right)
        acc = pl.matmul(left, right)
        pl.tpush_to_aiv(acc, split=0, id={pipe_id})

    @pl.function(type=pl.FunctionType.AIV)
    def plain_consumer(self, a: pl.Tensor[[64, 128], pl.FP32],
                       b: pl.Tensor[[128, 128], pl.FP32],
                       out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        buf = pl.reserve_buffer(name="plain", size=65536)
        pl.aiv_initialize_pipe(c2v_consumer_buf=buf, dir_mask=1,
            slot_size=32768, slot_num=2, id={pipe_id}{plain_options})
        value = pl.tpop_from_aic(shape=[64, 128], dtype=pl.FP32, split=0, id={pipe_id})
        pl.store(value, [0, 0], out)
        pl.tfree_to_aic(value, id={pipe_id})

    @pl.function(type=pl.FunctionType.Group)
    def plain(self, a: pl.Tensor[[64, 128], pl.FP32],
              b: pl.Tensor[[128, 128], pl.FP32],
              out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        self.plain_producer(a, b, out)
        self.plain_consumer(a, b, out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[64, 128], pl.FP32],
             b: pl.Tensor[[128, 128], pl.FP32],
             rounded_out: pl.Out[pl.Tensor[[64, 128], pl.BF16]],
             plain_out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
             raw: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        self.rounded(a, b, rounded_out, raw)
        self.plain(a, b, plain_out)
"""
    namespace: dict[str, Any] = {}
    exec(torch_codegen(pl.parse(source), check_shapes=check_shapes), namespace)  # noqa: S102
    assert namespace["_GROUP_META"] == {}
    a = torch.zeros(64, 128)
    a[:, :2] = 1
    b = torch.zeros(128, 128)
    b[0] = torch.tensor([1.0, 1.0078125, -1.0, -1.0078125] * 32)
    b[1] = torch.tensor([2**-8, 2**-11, -(2**-8), -(2**-11)] * 32)
    expected = a @ b
    rounded = torch.empty_like(expected, dtype=torch.bfloat16)
    plain = torch.empty_like(expected)
    raw = torch.empty_like(expected)
    for _ in range(3):
        namespace["main"](a, b, rounded, plain, raw)
        torch.testing.assert_close(rounded, expected.clamp_min(0).to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(plain, expected, rtol=0, atol=0)
        torch.testing.assert_close(raw, expected, rtol=0, atol=0)
        assert namespace["_cross_core_rt"].snapshot(pipe_id)["to_aiv"] == 0


@pytest.mark.parametrize("pipe_id", [0, 7])
@pytest.mark.parametrize("dir_mask", [1, 3])
def test_default_c2v_initializer_keeps_other_pipe_epilogues(pipe_id, dir_mask):
    """Reset one payload contract without dropping queued data or another pipe's epilogue."""
    namespace: dict[str, Any] = {}
    exec(torch_codegen_module._PREAMBLE, namespace)  # noqa: S102
    runtime = namespace["_cross_core_rt"]
    other_id = 7 if pipe_id == 0 else 0
    runtime.initialize_c2v_pipe(torch.bfloat16, True, pipe_id)
    runtime.initialize_c2v_pipe(torch.bfloat16, True, other_id)
    acc = torch.tensor([[-1.00390625, 1.00390625]], dtype=torch.float32)
    original = acc.clone()
    runtime.push_to_aiv(acc, 0, pipe_id=other_id)
    code = torch_codegen_module._handle_aic_initialize_pipe([], {"dir_mask": dir_mask, "id": pipe_id})
    exec(code, namespace)  # noqa: S102
    runtime.push_to_aiv(acc, 0, pipe_id=pipe_id)
    torch.testing.assert_close(runtime.pop_from_aic(0, pipe_id), original, rtol=0, atol=0)
    torch.testing.assert_close(
        runtime.pop_from_aic(0, other_id), original.clamp_min(0).to(torch.bfloat16), rtol=0, atol=0
    )
    torch.testing.assert_close(acc, original, rtol=0, atol=0)


def test_v2c_only_initializer_preserves_c2v_epilogue():
    """A V2C-only initializer does not replace the C2V contract on the same ID."""
    namespace: dict[str, Any] = {}
    exec(torch_codegen_module._PREAMBLE, namespace)  # noqa: S102
    runtime = namespace["_cross_core_rt"]
    runtime.initialize_c2v_pipe(torch.bfloat16, True, 7)
    code = torch_codegen_module._handle_aic_initialize_pipe([], {"dir_mask": 2, "id": 7})
    exec(code, namespace)  # noqa: S102
    acc = torch.tensor([[-1.00390625, 1.00390625]], dtype=torch.float32)
    runtime.push_to_aiv(acc, 0, pipe_id=7)
    torch.testing.assert_close(
        runtime.pop_from_aic(0, 7), acc.clamp_min(0).to(torch.bfloat16), rtol=0, atol=0
    )


def test_split_mode_to_int_invalid_value_raises():
    """Invalid split value should fail fast with a clear error."""
    with pytest.raises(ValueError, match="Invalid split mode"):
        torch_codegen_module._split_mode_to_int("bad_split")


def test_cross_core_split_merge_preserves_region_attrs():
    """Split/merge path should preserve valid/full region metadata."""
    ns: dict[str, Any] = {}
    exec(torch_codegen_module._PREAMBLE, ns)  # noqa: S102

    rt = ns["_cross_core_rt"]
    set_lane = ns["_set_subblock_idx"]

    tile = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    setattr(tile, "_pypto_valid_shape", (2, 3))
    setattr(tile, "_pypto_full_shape", (2, 4))

    rt.push_to_aiv(tile, 2)
    set_lane(0)
    lane0 = rt.pop_from_aic(2)
    set_lane(1)
    lane1 = rt.pop_from_aic(2)

    assert getattr(lane0, "_pypto_valid_shape", None) == (2, 2)
    assert getattr(lane1, "_pypto_valid_shape", None) == (2, 1)
    assert getattr(lane0, "_pypto_full_shape", None) == (2, 2)
    assert getattr(lane1, "_pypto_full_shape", None) == (2, 2)

    set_lane(0)
    rt.push_to_aic(lane0, 2)
    set_lane(1)
    rt.push_to_aic(lane1, 2)
    set_lane(0)
    merged = rt.pop_from_aiv(2)

    assert tuple(merged.shape) == (2, 4)
    assert getattr(merged, "_pypto_valid_shape", None) == (2, 3)
    assert getattr(merged, "_pypto_full_shape", None) == (2, 4)


def test_cross_core_rebalanced_split_slices_at_the_box_but_starts_at_the_stride():
    """`lane_stride` moves lane 1's START; the slice WIDTH stays the physical box.

    A ragged boundary rebalanced onto its valid region (16-row box, 13 valid,
    stride 7) gives both lanes the compiler's 8-row box — lane 0 rows 0-7 and
    lane 1 rows 7-14 — with per-lane valid extents 7 and 6. Cutting at the
    stride instead would hand the lanes 7- and 9-row payloads that no longer
    match the tile types the IR carries.
    """
    ns: dict[str, Any] = {}
    exec(torch_codegen_module._PREAMBLE, ns)  # noqa: S102

    rt = ns["_cross_core_rt"]
    set_lane = ns["_set_subblock_idx"]

    tile = torch.arange(16 * 2, dtype=torch.float32).reshape(16, 2)
    setattr(tile, "_pypto_valid_shape", (13, 2))
    setattr(tile, "_pypto_full_shape", (16, 2))

    rt.push_to_aiv(tile, 3, 7)
    set_lane(0)
    lane0 = rt.pop_from_aic(3)
    set_lane(1)
    lane1 = rt.pop_from_aic(3)

    assert tuple(lane0.shape) == (8, 2)
    assert tuple(lane1.shape) == (8, 2)
    # Lane 1 begins at the stride, not at the box half.
    assert torch.equal(lane0, tile[0:8])
    assert torch.equal(lane1, tile[7:15])
    # clamp(13 - lane * 7, 0, 7) -> 7 and 6.
    assert getattr(lane0, "_pypto_valid_shape", None) == (7, 2)
    assert getattr(lane1, "_pypto_valid_shape", None) == (6, 2)


def test_cross_core_odd_split_rejects_lanes_that_are_not_one_apart():
    """The _ODD codes exist to say "lane 1 is one cell shorter" — enforce it."""
    ns: dict[str, Any] = {}
    exec(torch_codegen_module._PREAMBLE, ns)  # noqa: S102

    rt = ns["_cross_core_rt"]
    tile = torch.zeros(16, 2, dtype=torch.float32)
    setattr(tile, "_pypto_valid_shape", (13, 2))

    # Box partition (stride 8) over 13 valid rows gives 8 and 5.
    with pytest.raises(ValueError, match="lanes one cell apart"):
        rt.push_to_aiv(tile, 3)


def test_cross_core_vector_to_cube_rejects_the_odd_codes():
    """pto-isa has no odd Vector -> Cube transport, so the runtime refuses it."""
    ns: dict[str, Any] = {}
    exec(torch_codegen_module._PREAMBLE, ns)  # noqa: S102

    rt = ns["_cross_core_rt"]
    tile = torch.zeros(8, 2, dtype=torch.float32)

    with pytest.raises(ValueError, match="Unsupported odd split mode for push_to_aic"):
        rt.push_to_aic(tile, 3)
    with pytest.raises(ValueError, match="Unsupported odd split mode for pop_from_aiv"):
        rt.pop_from_aiv(4)


def test_cross_core_no_split_dual_dispatch_runtime_pipe_pairing():
    """No-split dual-dispatch runtime should broadcast AIC->AIV and pair AIV->AIC traffic."""
    ns: dict[str, Any] = {}
    exec(torch_codegen_module._PREAMBLE, ns)  # noqa: S102

    rt = ns["_cross_core_rt"]
    set_lane = ns["_set_subblock_idx"]

    rt.reset(no_split_dual_aiv_dispatch=True)
    assert rt.snapshot()["no_split_dual_aiv_dispatch"] is True

    to_aiv_tile = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    rt.push_to_aiv(to_aiv_tile, 0)
    set_lane(0)
    lane0_recv = rt.pop_from_aic(0)
    set_lane(1)
    lane1_recv = rt.pop_from_aic(0)
    assert torch.equal(lane0_recv, to_aiv_tile)
    assert torch.equal(lane1_recv, to_aiv_tile)
    assert rt.snapshot()["to_aiv_dual_nosplit"] == {0: 0, 1: 0}

    lane0_tile = torch.full((2, 2), 1.0, dtype=torch.float32)
    lane1_tile = torch.full((2, 2), 2.0, dtype=torch.float32)
    set_lane(0)
    rt.push_to_aic(lane0_tile, 0)
    set_lane(1)
    rt.push_to_aic(lane1_tile, 0)
    set_lane(0)
    paired = rt.pop_from_aiv(0)
    assert torch.equal(paired, lane0_tile)
    assert rt.snapshot()["to_aic_dual_nosplit"] == {0: 0, 1: 0}


@pytest.mark.parametrize("split,dual", [(0, False), (0, True), (1, False), (2, False)])
def test_cross_core_pipe_ids_isolate_queues_in_both_directions(split, dual):
    """Reverse pipe order must not exchange payloads, including per-lane queues."""
    ns: dict[str, Any] = {}
    exec(torch_codegen_module._PREAMBLE, ns)  # noqa: S102
    rt = ns["_cross_core_rt"]
    set_lane = ns["_set_subblock_idx"]
    rt.reset(no_split_dual_aiv_dispatch=dual)
    payloads = {0: torch.full((4, 4), 1.0), 7: torch.full((4, 4), -3.0)}
    for pipe_id, payload in payloads.items():
        rt.push_to_aiv(payload, split, pipe_id=pipe_id)
    for pipe_id in (7, 0):
        for lane in range(2 if split or dual else 1):
            set_lane(lane)
            actual = rt.pop_from_aic(split, pipe_id=pipe_id)
            expected = payloads[pipe_id]
            if split:
                expected = rt._split_tile(expected, split)[lane]
            torch.testing.assert_close(actual, expected)

    for pipe_id, payload in payloads.items():
        parts = rt._split_tile(payload, split) if split else (payload, payload)
        for lane in range(2 if split or dual else 1):
            set_lane(lane)
            rt.push_to_aic(parts[lane], split, pipe_id=pipe_id)
    for pipe_id in (7, 0):
        actual = rt.pop_from_aiv(split, pipe_id=pipe_id)
        torch.testing.assert_close(actual, payloads[pipe_id])
    rt.reset()
    assert rt.snapshot(7)["to_aiv"] == 0
    assert rt.snapshot(7)["to_aic"] == 0


def test_mixed_kernel_timeout_cancels_waiters_and_runtime_recovers():
    """Timeout should cancel blocked waiters and clear cancel event for next runs."""
    ns: dict[str, Any] = {}
    exec(torch_codegen_module._PREAMBLE, ns)  # noqa: S102

    ns["_PIPE_WAIT_TIMEOUT_SEC"] = 60.0
    ns["_MIXED_KERNEL_TIMEOUT_SEC"] = 0.2

    def blocked_aic(_x):
        return ns["_cross_core_rt"].pop_from_aiv(0)

    def blocked_aiv(_x):
        return ns["_cross_core_rt"].pop_from_aic(0)

    ns["blocked_aic"] = blocked_aic
    ns["blocked_aiv"] = blocked_aiv

    with pytest.raises(RuntimeError, match="Mixed-kernel execution timeout"):
        ns["_run_mixed_kernels"](
            "blocked_group",
            {"aic": "blocked_aic", "aiv": "blocked_aiv", "split": 0},
            torch.tensor(0.0),
        )

    assert ns["_get_runtime_cancel_event"]() is None

    def ok_aic(_x):
        return None

    def ok_aiv(_x):
        return torch.tensor([1.0], dtype=torch.float32)

    ns["ok_aic"] = ok_aic
    ns["ok_aiv"] = ok_aiv
    out = ns["_run_mixed_kernels"](
        "ok_group",
        {"aic": "ok_aic", "aiv": "ok_aiv", "split": 0},
        torch.tensor(0.0),
    )
    assert torch.equal(out, torch.tensor([1.0], dtype=torch.float32))


def test_mixed_kernel_no_split_dual_aiv_dispatch_bidirect_pipe_runs():
    """No-split dual-dispatch bidirectional pipe should complete without deadlock."""
    ns: dict[str, Any] = {}
    exec(torch_codegen_module._PREAMBLE, ns)  # noqa: S102

    def aic(_x):
        recv = ns["_cross_core_rt"].pop_from_aiv(0)
        ns["_cross_core_rt"].push_to_aiv(recv + 10.0, 0)

    def aiv(_x):
        lane = ns["_get_subblock_idx"]()
        payload = torch.tensor([float(lane + 1)], dtype=torch.float32)
        ns["_cross_core_rt"].push_to_aic(payload, 0)
        recv = ns["_cross_core_rt"].pop_from_aic(0)
        if lane == 0:
            return recv
        return None

    ns["aic"] = aic
    ns["aiv"] = aiv
    out = ns["_run_mixed_kernels"](
        "dual_bidirect_group",
        {"aic": "aic", "aiv": "aiv", "split": 0, "dual_aiv_dispatch": True},
        torch.tensor(0.0),
    )
    assert torch.equal(out, torch.tensor([11.0], dtype=torch.float32))


def test_mixed_kernel_no_split_dual_aiv_dispatch_runs_two_lanes():
    """No-split + dual_aiv_dispatch should dispatch AIV on lane0 and lane1."""
    ns: dict[str, Any] = {}
    exec(torch_codegen_module._PREAMBLE, ns)  # noqa: S102

    lanes_seen: list[int] = []
    lane_lock = ns["threading"].Lock()

    def aic(_x):
        return None

    def aiv(_x):
        lane = ns["_get_subblock_idx"]()
        with lane_lock:
            lanes_seen.append(lane)
        if lane == 0:
            return torch.tensor([1.0], dtype=torch.float32)
        return None

    ns["aic"] = aic
    ns["aiv"] = aiv
    out = ns["_run_mixed_kernels"](
        "dual_dispatch_group",
        {"aic": "aic", "aiv": "aiv", "split": 0, "dual_aiv_dispatch": True},
        torch.tensor(0.0),
    )
    assert torch.equal(out, torch.tensor([1.0], dtype=torch.float32))
    assert sorted(lanes_seen) == [0, 1]


def test_mixed_kernel_no_split_without_dual_dispatch_runs_single_lane():
    """No-split without dual_aiv_dispatch should only run AIV lane0."""
    ns: dict[str, Any] = {}
    exec(torch_codegen_module._PREAMBLE, ns)  # noqa: S102

    lanes_seen: list[int] = []
    lane_lock = ns["threading"].Lock()

    def aic(_x):
        return None

    def aiv(_x):
        lane = ns["_get_subblock_idx"]()
        with lane_lock:
            lanes_seen.append(lane)
        return torch.tensor([1.0], dtype=torch.float32)

    ns["aic"] = aic
    ns["aiv"] = aiv
    out = ns["_run_mixed_kernels"](
        "single_dispatch_group",
        {"aic": "aic", "aiv": "aiv", "split": 0},
        torch.tensor(0.0),
    )
    assert torch.equal(out, torch.tensor([1.0], dtype=torch.float32))
    assert lanes_seen == [0]


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


@pytest.mark.parametrize("kind", ["if", "for", "while"])
@pytest.mark.parametrize("body_kind", ["empty", "yield", "nested_yield"])
def test_regions_with_no_emitted_statements_execute(kind, body_kind):
    if body_kind == "empty":
        body = ir.SeqStmts([], _span())
    elif body_kind == "yield":
        body = ir.YieldStmt([], _span())
    else:
        body = ir.SeqStmts([ir.SeqStmts([ir.YieldStmt([], _span())], _span())], _span())
    flag = _scalar("flag", DataType.BOOL)
    if kind == "if":
        region = ir.IfStmt(flag, body, body, [], _span())
    elif kind == "for":
        region = ir.ForStmt(_scalar("i"), _int(0), _int(3), _int(1), [], body, [], _span())
    else:
        region = ir.WhileStmt(ir.ConstBool(False, _span()), [], body, [], _span())
    function = _simple_function(
        "kernel",
        [flag],
        ir.SeqStmts([region, ir.ReturnStmt([_int(7)], _span())], _span()),
        [ir.ScalarType(DataType.INT64)],
    )
    namespace = {}
    exec(torch_codegen(function), namespace)  # noqa: S102
    assert namespace["kernel"](False) == 7
    assert namespace["kernel"](True) == 7


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


@pytest.mark.parametrize("dtype", [DataType.FP4E2M1X2, DataType.FP4E1M2X2])
def test_fp4_debug_reference_rejects_instead_of_emitting_float32(dtype):
    src = _tensor_var("src", [32, 64], dtype)
    func = _simple_function("fp4_identity", [src], ir.ReturnStmt([src], _span()), [src.type])
    with pytest.raises(ValueError, match="does not implement packed FP4 encoding or decoding"):
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
    """tensor.slice should pad to requested shape when slicing out of bounds.

    The window (rows 64..128 of a 96-row source) deliberately overhangs the
    source, so the slice must say so with clamp=True; its valid region is then
    the 32 rows that actually exist and the tail is padded.
    """
    src = _tensor_var("src", [96, 64], DataType.FP32)
    result = _tensor_var("result", [64, 64], DataType.FP32)
    shapes = _make_tuple(_int(64), _int(64))
    offsets = _make_tuple(_int(64), _int(0))
    call = _op_call("tensor.slice", [src, shapes, offsets], {"clamp": True})
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
    valid_shape = _make_tuple(_int(8), _int(32))
    sliced = _op_call("tensor.slice", [src, shapes, offsets, valid_shape])
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
# Test: valid_shape masking in tile.load
# ---------------------------------------------------------------------------


def test_tile_load_valid_shape_masks_invalid():
    """tile.load should zero out data beyond valid_shape."""
    tensor = _tensor_var("t", [8, 8])
    offsets = _make_tuple(_int(0), _int(0))
    shapes = _make_tuple(_int(8), _int(8))
    valid_shape = _make_tuple(_int(4), _int(4))
    tile = _tile_var("tile", [8, 8])

    call = _op_call(
        "tile.load",
        [tensor, offsets, shapes, valid_shape],
        {"target_memory": ir.MemorySpace.Vec},
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


def test_tile_load_passes_valid_shape():
    """tile.load codegen should pass valid_shape as 4th arg to _tile_load."""
    tensor = _tensor_var("t", [64, 64])
    offsets = _make_tuple(_int(0), _int(0))
    shapes = _make_tuple(_int(32), _int(32))
    valid_shape = _make_tuple(_int(16), _int(16))
    tile = _tile_var("tile", [32, 32])

    call = _op_call(
        "tile.load",
        [tensor, offsets, shapes, valid_shape],
        {"target_memory": ir.MemorySpace.Vec},
    )
    assign = ir.AssignStmt(tile, call, _span())
    func = _simple_function("f", [tensor], assign)
    code = torch_codegen(func)
    # Should pass all 4 args including valid_shape
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


def test_var_semantic_key_separates_unknown_span_vars():
    """Two distinct vars must not share a key, even with no source location.

    The regression this guards: keying on name/type/span merged two unknown-span
    vars that agreed on name and type, so ``_name_of`` handed them one generated
    name and the emitted Torch code could merge their values.
    """
    type_ = ir.ScalarType(DataType.INT64)
    a = ir.Var("x", type_, ir.Span.unknown())
    b = ir.Var("x", type_, ir.Span.unknown())

    assert TorchCodegen._var_semantic_key(a) != TorchCodegen._var_semantic_key(b)


def test_var_semantic_key_carries_no_span_coordinates():
    """The key must not embed span coordinates at all.

    An unknown span used to contribute the sentinel ("", -1, -1, -1, -1), which
    reads as a real source location. Identity now comes from ``unique_id``, so
    no span tuple — real or sentinel — belongs in the key.
    """
    var = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span("f.py", 3, 7, 3, 9))

    key = TorchCodegen._var_semantic_key(var)

    assert ("f.py", 3, 7, 3, 9) not in key
    assert ("", -1, -1, -1, -1) not in key
    assert var.unique_id in key


def test_var_semantic_key_is_stable_across_wrappers():
    """Two Python wrappers of one underlying Var must agree.

    This is the property the key exists for: ``id(var)`` is not stable across
    visits, so a per-wrapper identity would split one variable into two names.
    """
    type_ = ir.ScalarType(DataType.INT64)
    a = ir.Var("x", type_, ir.Span.unknown())
    add = ir.Add(a, a, DataType.INT64, ir.Span.unknown())
    left, right = add.left, add.right
    assert isinstance(left, ir.Var) and isinstance(right, ir.Var)

    assert TorchCodegen._var_semantic_key(left) == TorchCodegen._var_semantic_key(a)
    assert TorchCodegen._var_semantic_key(right) == TorchCodegen._var_semantic_key(a)


def test_var_semantic_key_distinguishes_spans():
    """Same name and type, different vars -> different keys."""
    type_ = ir.ScalarType(DataType.INT64)
    a = ir.Var("x", type_, ir.Span("f.py", 1, 1, 1, 2))
    b = ir.Var("x", type_, ir.Span("f.py", 9, 1, 9, 2))

    assert TorchCodegen._var_semantic_key(a) != TorchCodegen._var_semantic_key(b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
