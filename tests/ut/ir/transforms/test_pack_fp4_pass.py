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
        span = ir.Span.unknown()
        ty = ir.TensorType([_idx(span, 16), _idx(span, 63)], DataType.FP4)
        param = ir.Var("x", ty, span)
        prog = ir.Program(
            [ir.Function("main", [param], [ty], ir.ReturnStmt([param], span), span, ir.FunctionType.Opaque)],
            "odd",
            span,
        )
        with passes.PassContext([]):
            passes.pack_fp4()(prog)

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
        span = ir.Span.unknown()
        src_ty = ir.TensorType([_idx(span, 8), _idx(span, 16)], DataType.FP4)
        dst_ty = ir.TensorType([_idx(span, 4), _idx(span, 32)], DataType.FP4)
        src = ir.Var("x", src_ty, span)
        shape = ir.MakeTuple([_idx(span, 4), _idx(span, 32)], span)
        call = ir.Call(ir.get_op("tensor.reshape"), [src, shape], {}, dst_ty, span)
        out = ir.Var("y", dst_ty, span)
        prog = ir.Program(
            [
                ir.Function(
                    "main",
                    [src],
                    [dst_ty],
                    ir.SeqStmts([ir.AssignStmt(out, call, span), ir.ReturnStmt([out], span)], span),
                    span,
                    ir.FunctionType.Opaque,
                )
            ],
            "reshape",
            span,
        )
        with passes.PassContext([]):
            passes.pack_fp4()(prog)

    def case_transpose():
        span = ir.Span.unknown()
        src_ty = ir.TensorType([_idx(span, 2), _idx(span, 4), _idx(span, 16)], DataType.FP4)
        dst_ty = ir.TensorType([_idx(span, 4), _idx(span, 2), _idx(span, 16)], DataType.FP4)
        src = ir.Var("x", src_ty, span)
        call = ir.Call(
            ir.get_op("tensor.transpose"),
            [src, ir.ConstInt(0, DataType.INDEX, span), ir.ConstInt(1, DataType.INDEX, span)],
            {},
            dst_ty,
            span,
        )
        out = ir.Var("y", dst_ty, span)
        prog = ir.Program(
            [
                ir.Function(
                    "main",
                    [src],
                    [dst_ty],
                    ir.SeqStmts([ir.AssignStmt(out, call, span), ir.ReturnStmt([out], span)], span),
                    span,
                    ir.FunctionType.Opaque,
                )
            ],
            "transpose",
            span,
        )
        with passes.PassContext([]):
            passes.pack_fp4()(prog)

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


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("dtype", [pl.FP4, pl.FP4E2M1X2])
@pytest.mark.parametrize("rows", [16, pl.dynamic("rows")])
def test_pack_fp4_preserves_tensor_dim(axis, dtype, rows):
    """Logical extents survive packing without breaking NoNestedCalls."""

    @pl.program
    class Program:
        @pl.function
        def main(self, x: pl.Tensor[[rows, 64], dtype]) -> pl.Scalar[pl.INDEX]:
            return pl.tensor.dim(x, axis)

    packed = _run_prefix(Program)
    dim_name = ir.get_op("tensor.dim").name

    class DimFinder(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.dims: list[ir.Call] = []
            self.constants: list[int] = []

        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == dim_name:
                self.dims.append(op)
            super().visit_call(op)

        def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
            if isinstance(op.value, ir.ConstInt):
                self.constants.append(op.value.value)
            super().visit_assign_stmt(op)

    found = DimFinder()
    found.visit_program(packed)
    if dtype == pl.FP4 and axis != 0:
        assert not found.dims
        assert 64 in found.constants
    else:
        assert len(found.dims) == 1
        expected_last = 32 if dtype == pl.FP4 else 64
        tensor_type = found.dims[0].args[0].type
        assert isinstance(tensor_type, ir.TensorType)
        last_extent = tensor_type.shape[-1]
        assert isinstance(last_extent, ir.ConstInt)
        assert last_extent.value == expected_last


def test_pack_fp4_allows_globalvar_call_with_index_scalar():
    """Cross-function Call with FP4 tensor + INDEX scale must not hit coord reject."""
    reset_for_testing()
    set_backend_type(BackendType.Ascend950)

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 64], pl.FP4],
            scale: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP4]],
        ) -> pl.Tensor[[16, 64], pl.FP4]:
            _ = scale
            return pl.store(pl.load(x, [0, 0], [16, 64]), [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            x: pl.Tensor[[16, 64], pl.FP4],
            scale: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP4]],
        ) -> pl.Tensor[[16, 64], pl.FP4]:
            return self.kernel(x, scale, out)

    packed = _run_prefix(Program)
    assert packed is not None
    reset_for_testing()


@pytest.mark.parametrize("dtype", [DataType.INDEX, DataType.INT64, DataType.UINT64])
def test_pack_fp4_tensor_dim_preserves_recorded_integer_dtype(dtype):
    """A recorded dimension read keeps its saved scalar type when folded."""
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([16, 64], DataType.FP4)
    scalar_type = ir.ScalarType(dtype)
    tensor = ir.Var("x", tensor_type, span)
    extent = ir.Var("extent", scalar_type, span)
    dimension = ir.Call(ir.get_op("tensor.dim"), [tensor, _idx(span, -1)], {}, scalar_type, span)
    program = ir.Program(
        [
            ir.Function(
                "main",
                [tensor],
                [scalar_type],
                ir.SeqStmts([ir.AssignStmt(extent, dimension, span), ir.ReturnStmt([extent], span)], span),
                span,
                ir.FunctionType.Opaque,
            )
        ],
        "recorded_dim",
        span,
    )
    packed = passes.pack_fp4()(program)
    function = next(iter(packed.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    assignment = function.body.stmts[0]
    assert isinstance(assignment, ir.AssignStmt)
    assert isinstance(assignment.value, ir.ConstInt)
    assert assignment.value.value == 64
    ir.assert_structural_equal(assignment.value.type, scalar_type)
    ir.assert_structural_equal(assignment.var.type, scalar_type)


def test_pack_fp4_submit_matches_call_for_index_scalar():
    """pl.submit with FP4 + INDEX must follow the same GlobalVar skip as Call."""
    reset_for_testing()
    set_backend_type(BackendType.Ascend950)

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 64], pl.FP4],
            scale: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP4]],
        ) -> pl.Tensor[[16, 64], pl.FP4]:
            _ = scale
            return pl.store(pl.load(x, [0, 0], [16, 64]), [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            x: pl.Tensor[[16, 64], pl.FP4],
            scale: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP4]],
        ) -> pl.Tensor[[16, 64], pl.FP4]:
            result, _tid = pl.submit(self.kernel, x, scale, out)
            return result

    packed = _run_prefix(Program)
    assert packed is not None
    reset_for_testing()


@pytest.mark.parametrize("carry", ["direct", "if", "for", "while"])
@pytest.mark.parametrize(
    ("valid_rows", "result_stride"),
    [(2, [128, 1]), (1, [])],
    ids=["strided", "partial-valid-dense"],
)
def test_pack_fp4_tile_cast_rebuilds_contiguous_stride(carry, valid_rows, result_stride):
    """The canonical cast descriptor follows bindings, phis, carries, and returns."""
    span = ir.Span.unknown()
    src_view = ir.TileView(valid_shape=[valid_rows, 64], stride=[128, 1], start_offset=None)
    src_ty = ir.TileType([2, 64], DataType.BF16, None, src_view, ir.MemorySpace.Vec)
    dst_view = ir.TileView(valid_shape=[valid_rows, 64], stride=result_stride, start_offset=None)
    dst_ty = ir.TileType([2, 64], DataType.FP4, None, dst_view, ir.MemorySpace.Vec)
    src = ir.Var("src", src_ty, span)
    call = ir.Call(
        ir.get_op("tile.cast"),
        [src],
        {"target_type": DataType.FP4, "mode": 2},
        dst_ty,
        span,
    )
    out = ir.Var("out", dst_ty, span)
    alias = ir.Var("alias", dst_ty, span)
    # Sharing a type descriptor does not make this unrelated parameter a cast
    # result: its original non-contiguous packed pitch must stay intact.
    unrelated = ir.Var("unrelated", dst_ty, span)
    statements: list[ir.Stmt] = [
        ir.AssignStmt(out, call, span),
        ir.AssignStmt(alias, out, span),
    ]
    returned = alias
    if carry == "if":
        returned = ir.Var("if_result", dst_ty, span)
        statements.append(
            ir.IfStmt(
                ir.ConstBool(True, span),
                ir.YieldStmt([alias], span),
                ir.YieldStmt([alias], span),
                [returned],
                span,
            )
        )
    elif carry != "direct":
        argument = ir.IterArg("carried", dst_ty, alias, span)
        returned = ir.Var("loop_result", dst_ty, span)
        body = ir.YieldStmt([argument], span)
        if carry == "for":
            loop = ir.ForStmt(
                ir.Var("i", ir.ScalarType(DataType.INDEX), span),
                _idx(span, 0),
                _idx(span, 1),
                _idx(span, 1),
                [argument],
                body,
                [returned],
                span,
            )
        else:
            loop = ir.WhileStmt(ir.ConstBool(False, span), [argument], body, [returned], span)
        statements.append(loop)
    statements.append(ir.ReturnStmt([returned], span))
    prog = ir.Program(
        [
            ir.Function(
                "main",
                [src, unrelated],
                [dst_ty],
                ir.SeqStmts(statements, span),
                span,
                ir.FunctionType.Opaque,
            )
        ],
        "cast_stride",
        span,
    )
    # Keep the default verifier active: the original pass fixed only the RHS,
    # so AssignTypeSymmetry rejected this otherwise symmetric input.
    packed = passes.pack_fp4()(prog)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.TypeChecked)
    assert not passes.PropertyVerifierRegistry.verify(properties, packed)
    ir.verify_structural_bindings(packed)
    function = next(iter(packed.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    cast_assignment, alias_assignment = function.body.stmts[:2]
    assert isinstance(cast_assignment, ir.AssignStmt)
    assert isinstance(alias_assignment, ir.AssignStmt)
    cast_type = cast_assignment.value.type
    assert isinstance(cast_type, ir.TileType)
    assert cast_type.dtype == DataType.FP4E2M1X2
    assert _const_shape(cast_type) == [2, 32]
    assert cast_type.tile_view is not None
    assert _const_ints(list(cast_type.tile_view.stride)) == [32, 1]
    # Full valid extents may use the canonical omitted encoding.
    valid_shape = list(cast_type.tile_view.valid_shape) or list(cast_type.shape)
    assert _const_ints(valid_shape) == [valid_rows, 32]
    ir.assert_structural_equal(cast_assignment.var.type, cast_type)
    ir.assert_structural_equal(alias_assignment.var.type, cast_type)
    ir.assert_structural_equal(function.return_types[0], cast_type)
    assert alias_assignment.value is cast_assignment.var
    packed_unrelated = function.params[1].type
    assert isinstance(packed_unrelated, ir.TileType) and packed_unrelated.tile_view is not None
    assert _const_ints(list(packed_unrelated.tile_view.stride)) == ([64, 1] if result_stride else [])

    final_return = function.body.stmts[-1]
    assert isinstance(final_return, ir.ReturnStmt)
    if carry == "direct":
        assert final_return.value[0] is alias_assignment.var
    elif carry == "if":
        packed_branch = function.body.stmts[2]
        assert isinstance(packed_branch, ir.IfStmt)
        ir.assert_structural_equal(packed_branch.return_vars[0].type, cast_type)
        assert isinstance(packed_branch.then_body, ir.YieldStmt)
        assert isinstance(packed_branch.else_body, ir.YieldStmt)
        assert packed_branch.then_body.value[0] is alias_assignment.var
        assert packed_branch.else_body.value[0] is alias_assignment.var
        assert final_return.value[0] is packed_branch.return_vars[0]
    else:
        packed_loop = function.body.stmts[2]
        assert isinstance(packed_loop, (ir.ForStmt, ir.WhileStmt))
        packed_argument = packed_loop.iter_args[0]
        assert packed_argument.initValue is alias_assignment.var
        ir.assert_structural_equal(packed_argument.type, cast_type)
        ir.assert_structural_equal(packed_loop.return_vars[0].type, cast_type)
        assert isinstance(packed_loop.body, ir.YieldStmt)
        assert packed_loop.body.value[0] is packed_argument
        assert final_return.value[0] is packed_loop.return_vars[0]


@pytest.mark.parametrize("alias_op", ["tile.set_validshape", "tile.assemble"])
@pytest.mark.parametrize("carry", ["if", "for"])
def test_pack_fp4_refreshes_inherited_tile_views(alias_op, carry):
    """Aliases inherit a rewritten cast's pitch without losing valid extent or storage."""
    span = ir.Span.unknown()
    target_storage = ir.MemRef("target_storage", span)
    reference_storage = ir.MemRef("reference_storage", span)

    def tile_type(dtype, valid_rows, storage=None):
        view = ir.TileView(valid_shape=[valid_rows, 64], stride=[128, 1], start_offset=None)
        return ir.TileType([2, 64], dtype, storage, view, ir.MemorySpace.Vec)

    partial_type = tile_type(DataType.FP4, 1)
    full_type = tile_type(DataType.FP4, 2)
    target_type = tile_type(DataType.FP4, 1, target_storage)
    reference_type = tile_type(DataType.FP4, 2, reference_storage)
    alias_type = tile_type(DataType.FP4, 2, target_storage)
    source = ir.Var("source", tile_type(DataType.BF16, 1), span)
    full_source = ir.Var("full_source", tile_type(DataType.BF16, 2), span)
    target = ir.Var("target", target_type, span)
    reference = ir.Var("reference", reference_type, span)
    alias = ir.Var("alias", alias_type, span)
    params = [source, full_source]
    args: list[ir.Expr]
    if alias_op == "tile.set_validshape":
        args = [target, _idx(span, 2), _idx(span, 64)]
    else:
        patch = ir.Var("patch", ir.TileType([1, 64], DataType.FP4, None, None, ir.MemorySpace.Vec), span)
        params.append(patch)
        args = [target, patch, ir.MakeTuple([_idx(span, 1), _idx(span, 0)], span)]
    statements: list[ir.Stmt] = [
        ir.AssignStmt(
            target,
            ir.Call(
                ir.get_op("tile.cast"),
                [source],
                {"target_type": DataType.FP4, "mode": 2},
                partial_type,
                span,
            ),
            span,
        ),
        ir.AssignStmt(
            reference,
            ir.Call(
                ir.get_op("tile.cast"),
                [full_source],
                {"target_type": DataType.FP4, "mode": 2},
                full_type,
                span,
            ),
            span,
        ),
        ir.AssignStmt(alias, ir.Call(ir.get_op(alias_op), args, {}, alias_type, span), span),
    ]
    result = ir.Var("result", reference_type, span)
    control: ir.Stmt
    if carry == "if":
        control = ir.IfStmt(
            ir.ConstBool(True, span),
            ir.YieldStmt([reference], span),
            ir.YieldStmt([alias], span),
            [result],
            span,
        )
    else:
        argument = ir.IterArg("carried", reference_type, reference, span)
        control = ir.ForStmt(
            ir.Var("i", ir.ScalarType(DataType.INDEX), span),
            _idx(span, 0),
            _idx(span, 1),
            _idx(span, 1),
            [argument],
            ir.YieldStmt([alias], span),
            [result],
            span,
        )
    statements.extend([control, ir.ReturnStmt([result], span)])
    program = ir.Program(
        [
            ir.Function(
                "main",
                params,
                [reference_type],
                ir.SeqStmts(statements, span),
                span,
                ir.FunctionType.Opaque,
            )
        ],
        "inherited_tile_view",
        span,
    )
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.TypeChecked)
    assert not passes.PropertyVerifierRegistry.verify(properties, program)
    packed = passes.pack_fp4()(program)
    assert not passes.PropertyVerifierRegistry.verify(properties, packed)
    ir.verify_structural_bindings(packed)
    function = next(iter(packed.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    target_assignment, reference_assignment, alias_assignment = function.body.stmts[:3]
    assert isinstance(target_assignment, ir.AssignStmt)
    assert isinstance(reference_assignment, ir.AssignStmt)
    assert isinstance(alias_assignment, ir.AssignStmt)
    alias_call = alias_assignment.value
    assert isinstance(alias_call, ir.Call)
    assert alias_call.args[0] is target_assignment.var
    for value, storage in (
        (target_assignment.var, target_storage),
        (reference_assignment.var, reference_storage),
        (alias_assignment.var, target_storage),
        (alias_call, target_storage),
    ):
        assert isinstance(value.type, ir.TileType)
        assert value.type.memref is storage
        assert value.type.memory_space == ir.MemorySpace.Vec
    for type_ in (alias_call.type, alias_assignment.var.type, function.return_types[0]):
        assert isinstance(type_, ir.TileType) and type_.tile_view is not None
        assert type_.dtype == DataType.FP4E2M1X2 and _const_shape(type_) == [2, 32]
        assert type_.memory_space == ir.MemorySpace.Vec
        assert _const_ints(list(type_.tile_view.stride)) == [32, 1]
        valid_shape = list(type_.tile_view.valid_shape) or list(type_.shape)
        assert _const_ints(valid_shape) == [2, 32]
    packed_control = function.body.stmts[3]
    assert isinstance(packed_control, (ir.IfStmt, ir.ForStmt))
    assert isinstance(packed_control.return_vars[0].type, ir.TileType)
    assert packed_control.return_vars[0].type.memref is reference_storage
    final_return = function.body.stmts[-1]
    assert isinstance(final_return, ir.ReturnStmt)
    assert final_return.value[0] is packed_control.return_vars[0]


@pytest.mark.parametrize("loop_kind", ["for", "while"])
@pytest.mark.parametrize("tuple_carry", [False, True], ids=["tile", "nested-tuple"])
def test_pack_fp4_preserves_declared_iterarg_storage_identity(loop_kind, tuple_carry):
    """Carrying a declaration-only allocation must not remint a bound IterArg."""
    span = ir.Span.unknown()
    storage = ir.MemRef("carried_storage", span)
    seed_type: ir.Type = ir.TileType([2, 64], DataType.FP4, None, None, ir.MemorySpace.Vec)
    carry_type: ir.Type = ir.TileType([2, 64], DataType.FP4, storage, None, ir.MemorySpace.Vec)
    if tuple_carry:
        scalar_type = ir.ScalarType(DataType.INDEX)
        seed_type = ir.TupleType([ir.TupleType([seed_type]), scalar_type])
        carry_type = ir.TupleType([ir.TupleType([carry_type]), scalar_type])
    seed = ir.Var("seed", seed_type, span)
    argument = ir.IterArg("carried", carry_type, seed, span)
    alias = ir.Var("alias", carry_type, span)
    result = ir.Var("result", carry_type, span)
    body = ir.SeqStmts([ir.AssignStmt(alias, argument, span), ir.YieldStmt([argument], span)], span)
    if loop_kind == "for":
        loop = ir.ForStmt(
            ir.Var("i", ir.ScalarType(DataType.INDEX), span),
            _idx(span, 0),
            _idx(span, 1),
            _idx(span, 1),
            [argument],
            body,
            [result],
            span,
        )
    else:
        loop = ir.WhileStmt(ir.ConstBool(False, span), [argument], body, [result], span)
    program = ir.Program(
        [
            ir.Function(
                "main",
                [seed],
                [carry_type],
                ir.SeqStmts([loop, ir.ReturnStmt([result], span)], span),
                span,
                ir.FunctionType.Opaque,
            )
        ],
        "carried_storage",
        span,
    )
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.TypeChecked)
    assert not passes.PropertyVerifierRegistry.verify(properties, program)
    # Text loop headers infer carry storage from the initializer and cannot
    # represent this declaration-only MemRef. Keep property verification;
    # explicit identity assertions below exercise the raw-IR contract.
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        packed = passes.pack_fp4()(program)
    assert not passes.PropertyVerifierRegistry.verify(properties, packed)
    ir.verify_structural_bindings(packed)
    restored = ir.deserialize(ir.serialize(packed))
    ir.assert_structural_equal(packed, restored)
    assert ir.structural_hash(packed) == ir.structural_hash(restored)
    function = next(iter(packed.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    packed_loop = function.body.stmts[0]
    assert isinstance(packed_loop, (ir.ForStmt, ir.WhileStmt))
    binder = packed_loop.iter_args[0]
    assert isinstance(packed_loop.body, ir.SeqStmts)
    binding, yielded = packed_loop.body.stmts
    assert isinstance(binding, ir.AssignStmt) and binding.value is binder
    assert isinstance(yielded, ir.YieldStmt) and yielded.value[0] is binder
    carried_type = binder.type
    if tuple_carry:
        assert isinstance(carried_type, ir.TupleType)
        nested = carried_type.types[0]
        assert isinstance(nested, ir.TupleType)
        carried_type = nested.types[0]
    assert isinstance(carried_type, ir.TileType)
    assert carried_type.memref is storage


@pytest.mark.parametrize("mode", ["direct", "projection", "if", "for", "while", "multiple-returns"])
def test_pack_fp4_propagates_nested_tuple_cast_descriptors(mode):
    """Tuple aliases/projections carry the cast descriptor and each declaration's storage."""
    span = ir.Span.unknown()
    view = ir.TileView(valid_shape=[2, 64], stride=[128, 1], start_offset=None)
    source_type = ir.TileType([2, 64], DataType.BF16, None, view, ir.MemorySpace.Vec)
    cast_type = ir.TileType([2, 64], DataType.FP4, None, view, ir.MemorySpace.Vec)
    packed_storage = ir.MemRef("packed_storage", span)
    sibling_storage = ir.MemRef("sibling_storage", span)
    packed_type = ir.TileType([2, 64], DataType.FP4, packed_storage, view, ir.MemorySpace.Vec)
    sibling_view = ir.TileView(valid_shape=[2, 64], stride=[64, 1], start_offset=None)
    sibling_type = ir.TileType([2, 64], DataType.BF16, sibling_storage, sibling_view, ir.MemorySpace.Vec)
    tuple_type = ir.TupleType([ir.TupleType([packed_type]), sibling_type, ir.ScalarType(DataType.INDEX)])
    source = ir.Var("source", source_type, span)
    sibling = ir.Var(
        "sibling", ir.TileType([2, 64], DataType.BF16, None, sibling_view, ir.MemorySpace.Vec), span
    )
    implicit_sibling = ir.Var(
        "implicit_sibling", ir.TileType([2, 64], DataType.BF16, None, None, ir.MemorySpace.Vec), span
    )
    converted = ir.Var("converted", cast_type, span)
    pair = ir.Var("pair", tuple_type, span)
    alias = ir.Var("alias", tuple_type, span)
    cast = ir.Call(
        ir.get_op("tile.cast"), [source], {"target_type": DataType.FP4, "mode": 2}, cast_type, span
    )
    statements: list[ir.Stmt] = [
        ir.AssignStmt(converted, cast, span),
        ir.AssignStmt(
            pair, ir.MakeTuple([ir.MakeTuple([converted], span), sibling, _idx(span, 7)], span), span
        ),
        ir.AssignStmt(alias, pair, span),
    ]
    returned = alias
    return_type: ir.Type = tuple_type
    expected_storage = packed_storage
    expected_sibling_storage = sibling_storage
    if mode == "projection":
        returned = ir.Var("selected", packed_type, span)
        selected = ir.TupleGetItemExpr(ir.TupleGetItemExpr(alias, 0, span), 0, span)
        statements.append(ir.AssignStmt(returned, selected, span))
        return_type = packed_type
    elif mode == "if":
        # Phi storage belongs to the result, including its unrelated BF16 leaf.
        expected_storage = ir.MemRef("phi_packed_storage", span)
        expected_sibling_storage = ir.MemRef("phi_sibling_storage", span)
        phi_type = ir.TupleType(
            [
                ir.TupleType(
                    [ir.TileType([2, 64], DataType.FP4, expected_storage, view, ir.MemorySpace.Vec)]
                ),
                ir.TileType([2, 64], DataType.BF16, expected_sibling_storage, None, ir.MemorySpace.Vec),
                ir.ScalarType(DataType.INDEX),
            ]
        )
        returned = ir.Var("phi", phi_type, span)
        # These unrelated BF16 siblings use compatible explicit/omitted dense
        # views. FP4-specific compatibility checks must not make them stricter.
        alternative = ir.MakeTuple(
            [ir.TupleGetItemExpr(alias, 0, span), implicit_sibling, _idx(span, 7)], span
        )
        statements.append(
            ir.IfStmt(
                ir.ConstBool(True, span),
                ir.YieldStmt([alias], span),
                ir.YieldStmt([alternative], span),
                [returned],
                span,
            )
        )
        return_type = phi_type
    elif mode in ("for", "while"):
        argument = ir.IterArg("carried", tuple_type, alias, span)
        returned = ir.Var("result", tuple_type, span)
        body = ir.YieldStmt([argument], span)
        if mode == "for":
            loop = ir.ForStmt(
                ir.Var("i", ir.ScalarType(DataType.INDEX), span),
                _idx(span, 0),
                _idx(span, 1),
                _idx(span, 1),
                [argument],
                body,
                [returned],
                span,
            )
        else:
            loop = ir.WhileStmt(ir.ConstBool(False, span), [argument], body, [returned], span)
        statements.append(loop)
    if mode == "multiple-returns":
        statements.append(
            ir.IfStmt(
                ir.ConstBool(True, span),
                ir.ReturnStmt([alias], span),
                ir.ReturnStmt([alias], span),
                [],
                span,
            )
        )
    else:
        statements.append(ir.ReturnStmt([returned], span))
    program = ir.Program(
        [
            ir.Function(
                "main",
                [source, sibling, implicit_sibling],
                [return_type],
                ir.SeqStmts(statements, span),
                span,
                ir.FunctionType.Opaque,
            )
        ],
        "tuple_cast",
        span,
    )
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.TypeChecked)
    assert not passes.PropertyVerifierRegistry.verify(properties, program)
    packed = passes.pack_fp4()(program)
    assert not passes.PropertyVerifierRegistry.verify(properties, packed)
    ir.verify_structural_bindings(packed)
    function = next(iter(packed.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    pair_binding, alias_binding = function.body.stmts[1:3]
    assert isinstance(pair_binding, ir.AssignStmt) and isinstance(alias_binding, ir.AssignStmt)
    assert alias_binding.value is pair_binding.var

    def packed_leaf(type_: ir.Type) -> ir.TileType:
        if isinstance(type_, ir.TupleType):
            nested = type_.types[0]
            assert isinstance(nested, ir.TupleType)
            type_ = nested.types[0]
        assert isinstance(type_, ir.TileType)
        return type_

    for binding in (pair_binding, alias_binding):
        assert isinstance(binding.var.type, ir.TupleType)
        leaf = packed_leaf(binding.var.type)
        assert leaf.dtype == DataType.FP4E2M1X2
        assert _const_shape(leaf) == [2, 32]
        assert leaf.tile_view is not None
        assert _const_ints(list(leaf.tile_view.stride)) == [32, 1]
        assert leaf.memref is packed_storage
        assert binding.var.type.types[1] is sibling_type
    output_type = function.return_types[0]
    output_leaf = packed_leaf(output_type)
    assert output_leaf.tile_view is not None
    assert _const_ints(list(output_leaf.tile_view.stride)) == [32, 1]
    assert output_leaf.memref is expected_storage
    if mode != "projection":
        assert isinstance(output_type, ir.TupleType)
        output_sibling = output_type.types[1]
        assert isinstance(output_sibling, ir.TileType)
        assert output_sibling.memref is expected_sibling_storage
    if mode in ("for", "while"):
        packed_loop = function.body.stmts[3]
        assert isinstance(packed_loop, (ir.ForStmt, ir.WhileStmt))
        assert isinstance(packed_loop.body, ir.YieldStmt)
        assert packed_loop.body.value[0] is packed_loop.iter_args[0]
    elif mode == "projection":
        projection = function.body.stmts[3]
        assert isinstance(projection, ir.AssignStmt)
        ir.assert_structural_equal(projection.value.type, output_leaf)


@pytest.mark.parametrize("loop_kind", ["for", "while"])
@pytest.mark.parametrize("stride", [[128, 1], []], ids=["strided-seed", "implicit-dense-seed"])
def test_pack_fp4_rejects_incompatible_loop_backedge(loop_kind, stride):
    """Cast canonicalization cannot silently change a carried seed's descriptor."""
    span = ir.Span.unknown()
    view = ir.TileView(valid_shape=[1, 64], stride=stride, start_offset=None)
    fp4_type = ir.TileType([2, 64], DataType.FP4, None, view, ir.MemorySpace.Vec)
    bf16_type = ir.TileType([2, 64], DataType.BF16, None, view, ir.MemorySpace.Vec)
    seed = ir.Var("seed", fp4_type, span)
    source = ir.Var("source", bf16_type, span)
    argument = ir.IterArg("carried", fp4_type, seed, span)
    converted = ir.Var("converted", fp4_type, span)
    result = ir.Var("result", fp4_type, span)
    cast = ir.Call(ir.get_op("tile.cast"), [source], {"target_type": DataType.FP4, "mode": 2}, fp4_type, span)
    body = ir.SeqStmts([ir.AssignStmt(converted, cast, span), ir.YieldStmt([converted], span)], span)
    if loop_kind == "for":
        loop = ir.ForStmt(
            ir.Var("i", ir.ScalarType(DataType.INDEX), span),
            _idx(span, 0),
            _idx(span, 1),
            _idx(span, 1),
            [argument],
            body,
            [result],
            span,
        )
    else:
        loop = ir.WhileStmt(ir.ConstBool(True, span), [argument], body, [result], span)
    program = ir.Program(
        [
            ir.Function(
                "main",
                [seed, source],
                [fp4_type],
                ir.SeqStmts([loop, ir.ReturnStmt([result], span)], span),
                span,
                ir.FunctionType.Opaque,
            )
        ],
        "loop_backedge",
        span,
    )
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.TypeChecked)
    assert not passes.PropertyVerifierRegistry.verify(properties, program)
    # The cast becomes dense, but the seed keeps its existing view. A partial
    # valid shape also makes an empty-vs-explicit dense stride observable to
    # TypeChecked, even when both encodings address the same elements.
    with pytest.raises(_REJECT, match=r"PackFp4: packed .*loop carried descriptors must agree"):
        passes.pack_fp4()(program)


@pytest.mark.parametrize("mode", ["if", "multiple-returns", "for", "while"])
@pytest.mark.parametrize("valid_rows", [1, 2], ids=["explicit-partial-valid", "omitted-full-valid"])
@pytest.mark.parametrize("tuple_result", [False, True], ids=["tile", "tuple"])
def test_pack_fp4_mixed_cast_and_seed_descriptors(mode, valid_rows, tuple_result):
    """Explicit partial views stay strict; an omitted seed view can be refined."""
    span = ir.Span.unknown()
    seed_view = ir.TileView(valid_shape=[valid_rows, 64], stride=[], start_offset=None)
    seed_type = ir.TileType([2, 64], DataType.FP4, None, seed_view, ir.MemorySpace.Vec)
    bf16_type = ir.TileType([2, 64], DataType.BF16, None, seed_view, ir.MemorySpace.Vec)
    # Full-valid seeds omit their view. A normally deduced cast has explicit
    # dense strides, which may refine that provisional view across a branch.
    cast_view = ir.TileView(
        valid_shape=[valid_rows, 64], stride=[64, 1] if valid_rows == 2 else [], start_offset=None
    )
    cast_type = ir.TileType([2, 64], DataType.FP4, None, cast_view, ir.MemorySpace.Vec)
    seed = ir.Var("seed", seed_type, span)
    source = ir.Var("source", bf16_type, span)
    converted = ir.Var("converted", cast_type, span)
    seed_value: ir.Expr = seed
    converted_value: ir.Expr = converted
    result_type: ir.Type = seed_type
    if tuple_result:
        seed_value = ir.MakeTuple([seed, _idx(span, 7)], span)
        converted_value = ir.MakeTuple([converted, _idx(span, 7)], span)
        result_type = ir.TupleType([seed_type, ir.ScalarType(DataType.INDEX)])
    result = ir.Var("result", result_type, span)
    cast = ir.Call(
        ir.get_op("tile.cast"), [source], {"target_type": DataType.FP4, "mode": 2}, cast_type, span
    )
    statements: list[ir.Stmt] = [ir.AssignStmt(converted, cast, span)]
    if mode == "if":
        statements.extend(
            [
                ir.IfStmt(
                    ir.ConstBool(True, span),
                    ir.YieldStmt([converted_value], span),
                    ir.YieldStmt([seed_value], span),
                    [result],
                    span,
                ),
                ir.ReturnStmt([result], span),
            ]
        )
    elif mode == "multiple-returns":
        statements.append(
            ir.IfStmt(
                ir.ConstBool(True, span),
                ir.ReturnStmt([converted_value], span),
                ir.ReturnStmt([seed_value], span),
                [],
                span,
            )
        )
    else:
        argument = ir.IterArg("carried", result_type, seed_value, span)
        body = ir.YieldStmt([converted_value], span)
        if mode == "for":
            loop = ir.ForStmt(
                ir.Var("i", ir.ScalarType(DataType.INDEX), span),
                _idx(span, 0),
                _idx(span, 1),
                _idx(span, 1),
                [argument],
                body,
                [result],
                span,
            )
        else:
            loop = ir.WhileStmt(ir.ConstBool(False, span), [argument], body, [result], span)
        statements.extend([loop, ir.ReturnStmt([result], span)])
    program = ir.Program(
        [
            ir.Function(
                "main",
                [seed, source],
                [result_type],
                ir.SeqStmts(statements, span),
                span,
                ir.FunctionType.Opaque,
            )
        ],
        "mixed_descriptors",
        span,
    )
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.TypeChecked)
    assert not passes.PropertyVerifierRegistry.verify(properties, program)
    if valid_rows == 1:
        with pytest.raises(_REJECT, match=r"PackFp4: packed .*descriptors must agree"):
            passes.pack_fp4()(program)
    else:
        packed = passes.pack_fp4()(program)
        assert not passes.PropertyVerifierRegistry.verify(properties, packed)
        ir.verify_structural_bindings(packed)
        function = next(iter(packed.functions.values()))
        packed_seed = function.params[0]
        assert isinstance(packed_seed.type, ir.TileType)
        assert packed_seed.type.tile_view is None
        assert _const_shape(packed_seed.type) == [2, 32]
        assert isinstance(function.body, ir.SeqStmts)
        branch = function.body.stmts[1]
        if mode == "if":
            assert isinstance(branch, ir.IfStmt)
            assert isinstance(branch.else_body, ir.YieldStmt)
            seed_reference = branch.else_body.value[0]
        elif mode == "multiple-returns":
            assert isinstance(branch, ir.IfStmt)
            assert isinstance(branch.else_body, ir.ReturnStmt)
            seed_reference = branch.else_body.value[0]
        else:
            assert isinstance(branch, (ir.ForStmt, ir.WhileStmt))
            seed_reference = branch.iter_args[0].initValue
        if tuple_result:
            assert isinstance(seed_reference, ir.MakeTuple)
            seed_reference = seed_reference.elements[0]
        assert seed_reference is packed_seed


@pytest.mark.parametrize("dtype", [DataType.INT32, DataType.FP4, DataType.FP4E2M1X2])
def test_pack_fp4_reuses_shared_leaf_types_in_nested_tuples(dtype):
    """Cache immutable type work without expanding repeated logical FP4 leaves."""
    span = ir.Span.unknown()
    leaf = ir.TensorType([2, 64], dtype)
    nested: ir.Type = leaf
    # Each level shares the same leaf. Rechecking every descendant at every
    # level previously made even this ordinary INT32 case quadratic; logical
    # FP4 also minted a separate packed copy of the shared leaf on every visit.
    depth = 64  # Keep the default text-roundtrip parser below its nesting limit.
    for _ in range(depth):
        nested = ir.TupleType([nested, leaf])
    param = ir.Var("x", nested, span)
    program = ir.Program(
        [ir.Function("main", [param], [nested], ir.ReturnStmt([param], span), span, ir.FunctionType.Opaque)],
        "nested_types",
        span,
    )
    packed = passes.pack_fp4()(program)
    function = next(iter(packed.functions.values()))
    result_type = function.params[0].type
    assert isinstance(result_type, ir.TupleType)
    packed_leaf = result_type.types[1]
    assert isinstance(packed_leaf, ir.TensorType)
    expected_dtype = DataType.FP4E2M1X2 if dtype == DataType.FP4 else dtype
    assert packed_leaf.dtype == expected_dtype
    assert _const_shape(packed_leaf) == [2, 32 if dtype == DataType.FP4 else 64]
    for _ in range(depth):
        assert isinstance(result_type, ir.TupleType)
        assert result_type.types[1] is packed_leaf
        result_type = result_type.types[0]
    assert result_type is packed_leaf
    ir.verify_structural_bindings(packed)


def test_pack_fp4_rejects_nonzero_tile_start_offset():
    span = ir.Span.unknown()
    view = ir.TileView(valid_shape=[16, 64], stride=[64, 1], start_offset=2)
    ty = ir.TileType([16, 64], DataType.FP4, None, view, ir.MemorySpace.Vec)
    param = ir.Var("t", ty, span)
    prog = ir.Program(
        [ir.Function("main", [param], [ty], ir.ReturnStmt([param], span), span, ir.FunctionType.Opaque)],
        "offset",
        span,
    )
    with pytest.raises(_REJECT, match=r"start_offset"):
        with passes.PassContext([]):
            passes.pack_fp4()(prog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
