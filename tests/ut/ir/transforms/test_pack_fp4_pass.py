# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for PackFp4: packing rewrite + unsupported-op rejects."""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import DataType, ir, passes
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.language.parser.diagnostics.exceptions import InvalidOperationError, ParserSyntaxError

_FULL_OPS = frozenset({ir.get_op("tensor.full").name, ir.get_op("tile.full").name})
_REINTERPRET = ir.get_op("tensor.reinterpret_view").name
_VIEW = ir.get_op("tensor.view").name
_REJECT = (ValueError, InvalidOperationError, ParserSyntaxError, RuntimeError)


def _run_prefix(program):
    for factory in (
        passes.inline_functions,
        passes.unroll_loops,
        passes.ctrl_flow_transform,
        passes.convert_to_ssa,
        passes.simplify,
        passes.normalize_stmt_structure,
        passes.flatten_call_expr,
        passes.pack_fp4,
    ):
        program = factory()(program)
    return program


def _const_ints(exprs: list) -> list[int]:
    values: list[int] = []
    for dim in exprs:
        assert isinstance(dim, ir.ConstInt)
        values.append(dim.value)
    return values


def _const_shape(type_):
    return _const_ints(list(type_.shape))


def _idx(span: ir.Span, value: int) -> ir.ConstInt:
    return ir.ConstInt(value, DataType.INDEX, span)


def test_pack_fp4_packs_shapes_strides_and_full():
    """Rank-2/3 shapes, ND strides, and tensor.full last dim all pack /2."""
    reset_for_testing()
    set_backend_type(BackendType.Ascend950)

    @pl.program
    class Program:
        @pl.function
        def main(
            self,
            rows: pl.Tensor[[2, 512], pl.FP4],
            batch: pl.Tensor[[2, 16, 64], pl.FP4],
        ) -> pl.Tensor[[16, 64], pl.FP4]:
            return pl.full([16, 64], dtype=pl.FP4, value=0)

    packed = _run_prefix(Program)
    func = next(iter(packed.functions.values()))
    rows, batch = (p.type for p in func.params)
    assert isinstance(rows, ir.TensorType) and isinstance(batch, ir.TensorType)
    assert rows.dtype == DataType.FP4E2M1X2 and _const_shape(rows) == [2, 256]
    assert batch.dtype == DataType.FP4E2M1X2 and _const_shape(batch) == [2, 16, 32]

    span = ir.Span.unknown()
    view = ir.TensorView()
    view.layout = ir.TensorLayout.ND
    view.valid_shape = [_idx(span, 2), _idx(span, 16), _idx(span, 64)]
    view.stride = [_idx(span, 1024), _idx(span, 64), _idx(span, 1)]
    tt = ir.TensorType([_idx(span, 2), _idx(span, 16), _idx(span, 64)], DataType.FP4, None, view)
    param = ir.Var("x", tt, span)
    stride_prog = ir.Program(
        [ir.Function("main", [param], [tt], ir.ReturnStmt([param], span), span, ir.FunctionType.Opaque)],
        "stride",
        span,
    )
    src = next(iter(passes.pack_fp4()(stride_prog).functions.values())).params[0].type
    assert isinstance(src, ir.TensorType)
    assert _const_shape(src) == [2, 16, 32]
    assert src.tensor_view is not None
    assert _const_ints(list(src.tensor_view.stride)) == [512, 32, 1]

    shapes: list[list[int]] = []

    class _Full(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            if op.op.name in _FULL_OPS:
                shape_arg = op.args[0]
                assert isinstance(shape_arg, ir.MakeTuple)
                shapes.append(_const_ints(list(shape_arg.elements)))
            super().visit_call(op)

    _Full().visit_program(packed)
    assert shapes == [[16, 32]]
    reset_for_testing()


def test_pack_fp4_view_and_reinterpret():
    """view packs shape+valid_shape; FP4→UINT8 reinterpret keeps dest-element geometry."""
    span = ir.Span.unknown()
    src_view = ir.TensorView()
    src_view.layout = ir.TensorLayout.ND
    src_view.valid_shape = [_idx(span, 16), _idx(span, 64)]
    src_ty = ir.TensorType([_idx(span, 16), _idx(span, 64)], DataType.FP4, None, src_view)
    src = ir.Var("x", src_ty, span)

    dst_view = ir.TensorView()
    dst_view.layout = ir.TensorLayout.ND
    dst_view.valid_shape = [_idx(span, 8), _idx(span, 48)]
    dst_ty = ir.TensorType([_idx(span, 16), _idx(span, 64)], DataType.FP4, None, dst_view)
    view_call = ir.Call(
        ir.get_op("tensor.view"),
        [
            src,
            ir.MakeTuple([_idx(span, 16), _idx(span, 64)], span),
            ir.MakeTuple([_idx(span, 8), _idx(span, 48)], span),
        ],
        {},
        dst_ty,
        span,
    )
    y = ir.Var("y", dst_ty, span)
    view_prog = ir.Program(
        [
            ir.Function(
                "main",
                [src],
                [dst_ty],
                ir.SeqStmts([ir.AssignStmt(y, view_call, span), ir.ReturnStmt([y], span)], span),
                span,
                ir.FunctionType.Opaque,
            )
        ],
        "view",
        span,
    )
    with passes.PassContext([]):
        packed_view = passes.pack_fp4()(view_prog)

    got_shapes: list[list[int]] = []
    got_valids: list[list[int]] = []

    class _V(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == _VIEW:
                shape_arg, valid_arg = op.args[1], op.args[2]
                assert isinstance(shape_arg, ir.MakeTuple) and isinstance(valid_arg, ir.MakeTuple)
                got_shapes.append(_const_ints(list(shape_arg.elements)))
                got_valids.append(_const_ints(list(valid_arg.elements)))
            super().visit_call(op)

    _V().visit_program(packed_view)
    assert got_shapes == [[16, 32]] and got_valids == [[8, 24]]

    u8 = ir.TensorType([_idx(span, 16), _idx(span, 32)], DataType.UINT8)
    rein = ir.Call(
        ir.get_op("tensor.reinterpret_view"),
        [src, ir.MakeTuple([_idx(span, 16), _idx(span, 32)], span)],
        {"dtype": DataType.UINT8},
        u8,
        span,
    )
    z = ir.Var("z", u8, span)
    rein_prog = ir.Program(
        [
            ir.Function(
                "main",
                [src],
                [u8],
                ir.SeqStmts([ir.AssignStmt(z, rein, span), ir.ReturnStmt([z], span)], span),
                span,
                ir.FunctionType.Opaque,
            )
        ],
        "rein",
        span,
    )
    with passes.PassContext([]):
        packed_rein = passes.pack_fp4()(rein_prog)

    rein_shapes: list[list[int]] = []

    class _R(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == _REINTERPRET:
                shape_arg = op.args[1]
                assert isinstance(shape_arg, ir.MakeTuple)
                rein_shapes.append(_const_ints(list(shape_arg.elements)))
            super().visit_call(op)

    _R().visit_program(packed_rein)
    assert rein_shapes == [[16, 32]]


def test_pack_fp4_rejects_unsupported():
    """Static-only / layout / reshape-transpose-read / distributed / whitelist fallback."""

    def case_dynamic():
        k = pl.dynamic("K")

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[16, k], pl.FP4]) -> pl.Tensor[[16, k], pl.FP4]:
                return x

        _run_prefix(P)

    def case_odd_dim():
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[16, 63], pl.FP4]) -> pl.Tensor[[16, 63], pl.FP4]:
                return x

    def case_cube():
        reset_for_testing()
        set_backend_type(BackendType.Ascend950)
        try:

            @pl.program
            class P:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self, x: pl.Tensor[[16, 64], pl.FP4], out: pl.Out[pl.Tensor[[16, 64], pl.FP4]]
                ) -> pl.Tensor[[16, 64], pl.FP4]:
                    return pl.store(pl.load(x, [0, 0], [16, 64], target_memory=pl.Mem.Mat), [0, 0], out)

            _run_prefix(P)
        finally:
            reset_for_testing()

    def case_dyn_offset():
        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                x: pl.Tensor[[16, 64], pl.FP4],
                out: pl.Out[pl.Tensor[[16, 32], pl.FP4]],
                offset: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 32], pl.FP4]:
                return pl.store(pl.load(x, [0, offset], [16, 32]), [0, 0], out)

        _run_prefix(P)

    def case_read():
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[16, 64], pl.FP4]) -> pl.Scalar[pl.FP4]:
                return pl.read(x, [0, 0])

        _run_prefix(P)

    def case_reshape():
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[8, 16], pl.FP4]) -> pl.Tensor[[4, 32], pl.FP4]:
                return pl.reshape(x, [4, 32])

    def case_transpose():
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[2, 4, 16], pl.FP4]) -> pl.Tensor[[4, 2, 16], pl.FP4]:
                return pl.transpose(x, axis1=0, axis2=1)

    def case_remote():
        reset_for_testing()
        set_backend_type(BackendType.Ascend950)
        try:

            @pl.program
            class P:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    data: pld.DistributedTensor[[2, 512], pl.FP4],
                    out: pl.Tensor[[2, 512], pl.FP4],
                    peer: pl.Scalar[pl.INT32],
                ):
                    pl.store(
                        pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[2, 512]), [0, 0], out
                    )

            _run_prefix(P)
        finally:
            reset_for_testing()

    def case_window():
        span = ir.Span.unknown()
        buf = ir.Var("buf", ir.PtrType.get(), span)
        shape = ir.MakeTuple([_idx(span, 16), _idx(span, 64)], span)
        ty = ir.DistributedTensorType([_idx(span, 16), _idx(span, 64)], DataType.FP4)
        call = ir.Call(ir.get_op("pld.tensor.window"), [buf, shape], {"dtype": DataType.FP4}, ty, span)
        win = ir.Var("win", ty, span)
        prog = ir.Program(
            [
                ir.Function(
                    "main",
                    [buf],
                    [ty],
                    ir.SeqStmts([ir.AssignStmt(win, call, span), ir.ReturnStmt([win], span)], span),
                    span,
                    ir.FunctionType.Opaque,
                )
            ],
            "win",
            span,
        )
        with passes.PassContext([]):
            passes.pack_fp4()(prog)

    def case_uncovered():
        span = ir.Span.unknown()
        shape = ir.MakeTuple([_idx(span, 16), _idx(span, 64)], span)
        ty = ir.TensorType([_idx(span, 16), _idx(span, 64)], DataType.FP4)
        call = ir.Call(ir.get_op("tensor.create_l1"), [shape], {"dtype": DataType.FP4}, ty, span)
        acc = ir.Var("acc", ty, span)
        prog = ir.Program(
            [
                ir.Function(
                    "main",
                    [],
                    [ty],
                    ir.SeqStmts([ir.AssignStmt(acc, call, span), ir.ReturnStmt([acc], span)], span),
                    span,
                    ir.FunctionType.Opaque,
                )
            ],
            "l1",
            span,
        )
        with passes.PassContext([]):
            passes.pack_fp4()(prog)

    def case_handwritten_dn():
        """Hand-written FP4E2M1X2 must still reject DN (no PackType rewrite path)."""
        span = ir.Span.unknown()
        view = ir.TensorView()
        view.layout = ir.TensorLayout.DN
        ty = ir.TensorType([_idx(span, 16), _idx(span, 32)], DataType.FP4E2M1X2, None, view)
        param = ir.Var("x", ty, span)
        prog = ir.Program(
            [ir.Function("main", [param], [ty], ir.ReturnStmt([param], span), span, ir.FunctionType.Opaque)],
            "dn",
            span,
        )
        with passes.PassContext([]):
            passes.pack_fp4()(prog)

    def case_scalar_coord():
        """Unwhitelisted FP4 op with a scalar INDEX operand must loud-fail."""
        span = ir.Span.unknown()
        ty = ir.TensorType([_idx(span, 16), _idx(span, 32)], DataType.FP4E2M1X2)
        offset = _idx(span, 2)
        call = ir.Call(ir.get_op("tensor.create_l1"), [offset], {"dtype": DataType.FP4E2M1X2}, ty, span)
        acc = ir.Var("acc", ty, span)
        prog = ir.Program(
            [
                ir.Function(
                    "main",
                    [],
                    [ty],
                    ir.SeqStmts([ir.AssignStmt(acc, call, span), ir.ReturnStmt([acc], span)], span),
                    span,
                    ir.FunctionType.Opaque,
                )
            ],
            "l1_scalar",
            span,
        )
        with passes.PassContext([]):
            passes.pack_fp4()(prog)

    for match, fn in [
        ("static ConstInt|static-only|FP4E2M1X2|UINT8", case_dynamic),
        ("even", case_odd_dim),
        ("cube|Mat", case_cube),
        ("static ConstInt|static-only|even|packed pair", case_dyn_offset),
        ("read/write|not supported|FP4", case_read),
        ("reshape|not supported|FP4", case_reshape),
        ("transpose|not supported|FP4|packed FP4", case_transpose),
        ("DistributedTensor|distributed|multi-device|not supported|FP4", case_remote),
        ("DistributedTensor|distributed|not supported|FP4", case_window),
        ("not yet rewritten|create_l1", case_uncovered),
        ("DN layout|unsupported", case_handwritten_dn),
        ("not yet rewritten|create_l1", case_scalar_coord),
    ]:
        with pytest.raises(_REJECT, match=match):
            fn()
