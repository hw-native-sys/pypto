# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ConvertTensorToTileOps pass."""

from collections.abc import Callable

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir import IRBuilder
from pypto.ir.op import tensor as tensor_ops
from pypto.ir.op import tile as tile_ops
from pypto.pypto_core.ir import MemorySpace, PadValue

# ---------------------------------------------------------------------------
# Programmatic IRBuilder factories for parametrized tests.
#
# Most tests in this module follow the same skeleton: an InCore function with
# one or more tensor inputs, a single-op (or chain) body, and an orchestrator
# `main` that calls the InCore function. After the pass, the InCore body
# becomes `tile.load -> tile.<op> -> tile.store` and gains an `Out` parameter,
# and the orchestrator allocates the matching tensor via ``tensor.create``.
#
# The factories below mirror that skeleton so families of similar tests can
# share one parametrized body.
# ---------------------------------------------------------------------------

InSpec = tuple[str, list[int], DataType]  # (param name, shape, dtype)
ExtraSpec = tuple[str, ir.Type]  # (param name, ir type) — non-tensor (e.g. Scalar) extra params
TensorBody = Callable[..., ir.Expr]  # (ib, in_vars[, extras]) -> final tensor var
TileBody = Callable[..., ir.Expr]  # (ib, tile_vars[, extras]) -> final tile var


def _zeros(rank: int) -> list[int]:
    return [0] * rank


def _build_orch(
    ib: IRBuilder,
    incore_gvar: ir.GlobalVar,
    in_specs: list[InSpec],
    out_shape: list[int],
    out_dtype: DataType,
    *,
    pass_out_arg: bool,
    extra_specs: list[ExtraSpec] | None = None,
) -> ir.Function:
    """Build the `main` orchestrator that forwards inputs (and optionally an Out tensor)."""
    in_types = [ir.TensorType(s, d) for _, s, d in in_specs]
    out_type = ir.TensorType(out_shape, out_dtype)
    extra_specs = extra_specs or []
    with ib.function("main") as f:
        params = [f.param(name, t) for (name, _, _), t in zip(in_specs, in_types, strict=False)]
        extras = [f.param(name, t) for name, t in extra_specs]
        f.return_type(out_type)
        if pass_out_arg:
            out_var = ib.let("out_0", tensor_ops.create(out_shape, out_dtype))
            call_args = [*params, *extras, out_var]
        else:
            call_args = [*params, *extras]
        y = ib.let("y", ir.Call(incore_gvar, call_args, ir.Span.unknown()))
        ib.return_stmt(y)
    return f.get_result()


def _make_before(
    *,
    in_specs: list[InSpec],
    out_shape: list[int],
    out_dtype: DataType,
    body: TensorBody,
    incore_name: str = "main_incore_0",
    extra_specs: list[ExtraSpec] | None = None,
) -> ir.Program:
    """Build the *Before* program with InCore body driven by ``body(ib, in_vars[, extras])``."""
    ib = IRBuilder()
    out_type = ir.TensorType(out_shape, out_dtype)
    in_types = [ir.TensorType(s, d) for _, s, d in in_specs]
    extra_specs = extra_specs or []
    with ib.program("main") as prog:
        incore_gvar = prog.declare_function(incore_name)
        prog.declare_function("main")
        with ib.function(incore_name, type=ir.FunctionType.InCore) as f:
            params = [f.param(name, t) for (name, _, _), t in zip(in_specs, in_types, strict=False)]
            extras = [f.param(name, t) for name, t in extra_specs]
            f.return_type(out_type)
            ib.return_stmt(body(ib, list(params), extras) if extras else body(ib, list(params)))
        prog.add_function(f.get_result())
        prog.add_function(
            _build_orch(
                ib,
                incore_gvar,
                in_specs,
                out_shape,
                out_dtype,
                pass_out_arg=False,
                extra_specs=extra_specs,
            )
        )
    return prog.get_result()


def _make_expected(
    *,
    in_specs: list[InSpec],
    out_shape: list[int],
    out_dtype: DataType,
    body: TileBody,
    incore_name: str = "main_incore_0",
    load_shapes: list[list[int] | None] | None = None,
    load_names: list[str | None] | None = None,
    preload: bool = True,
    extra_specs: list[ExtraSpec] | None = None,
) -> ir.Program:
    """Build the *Expected* program: pre-load every input, run ``body``, then store.

    ``load_shapes`` lets a test override the default (full-spec) load shape per
    input, which is useful when the pass lowers the first ``tensor.slice`` of a
    chain to ``tile.load`` with the slice's shape rather than the parameter's.
    ``load_names`` overrides the loaded tile variable's stem (defaults to the
    in-spec name) for the same reason.
    """
    ib = IRBuilder()
    out_type = ir.TensorType(out_shape, out_dtype)
    in_types = [ir.TensorType(s, d) for _, s, d in in_specs]
    extra_specs = extra_specs or []
    with ib.program("main") as prog:
        incore_gvar = prog.declare_function(incore_name)
        prog.declare_function("main")
        with ib.function(incore_name, type=ir.FunctionType.InCore) as f:
            in_params = [f.param(name, t) for (name, _, _), t in zip(in_specs, in_types, strict=False)]
            extras = [f.param(name, t) for name, t in extra_specs]
            out_param = f.param("out_0", out_type, direction=ir.ParamDirection.Out)
            f.return_type(out_type)
            if preload:
                tile_vars: list[ir.Expr] = []
                for i, ((name, shape, _), p) in enumerate(zip(in_specs, in_params, strict=False)):
                    override = None if load_shapes is None else load_shapes[i]
                    load_shape = list(shape if override is None else override)
                    name_override = None if load_names is None else load_names[i]
                    stem = name if name_override is None else name_override
                    tile_vars.append(
                        ib.let(f"{stem}_tile", tile_ops.load(p, _zeros(len(load_shape)), load_shape))
                    )
            else:
                tile_vars = list(in_params)
            final_tile = body(ib, tile_vars, extras) if extras else body(ib, tile_vars)
            store = ib.let("out_0_store", tile_ops.store(final_tile, _zeros(len(out_shape)), out_param))
            ib.return_stmt(store)
        prog.add_function(f.get_result())
        prog.add_function(
            _build_orch(
                ib,
                incore_gvar,
                in_specs,
                out_shape,
                out_dtype,
                pass_out_arg=True,
                extra_specs=extra_specs,
            )
        )
    return prog.get_result()


def _make_pair(
    *,
    in_specs: list[InSpec],
    out_shape: list[int],
    out_dtype: DataType,
    tensor_op: Callable[[list[ir.Expr]], ir.Call],
    tile_op: Callable[[list[ir.Expr]], ir.Call],
) -> tuple[ir.Program, ir.Program]:
    """Build (Before, Expected) for the canonical ``tensor.OP -> load + tile.OP + store`` pattern."""
    before = _make_before(
        in_specs=in_specs,
        out_shape=out_shape,
        out_dtype=out_dtype,
        body=lambda ib, ins: ib.let("y", tensor_op(ins)),
    )
    expected = _make_expected(
        in_specs=in_specs,
        out_shape=out_shape,
        out_dtype=out_dtype,
        body=lambda ib, tiles: ib.let("y_tile", tile_op(tiles)),
    )
    return before, expected


def _assert_convert_equal(before: ir.Program, expected: ir.Program) -> None:
    """Run ConvertTensorToTileOps on ``before`` and assert the result matches ``expected``."""
    after = passes.convert_tensor_to_tile_ops()(before)
    ir.assert_structural_equal(after, expected)


# ---------------------------------------------------------------------------
# Op family parametrization tables.
# ---------------------------------------------------------------------------

# Unary ops on a 1D 64-element FP32 tensor: tensor.OP(x) -> tile.OP(x_tile).
_UNARY_1D_OPS = [
    ("exp", tensor_ops.exp, tile_ops.exp),
    ("neg", tensor_ops.neg, tile_ops.neg),
    ("recip", tensor_ops.recip, tile_ops.recip),
    ("sqrt", tensor_ops.sqrt, tile_ops.sqrt),
]

# 2D row/col-expand-style binary ops with a vector side input.
_ROW_EXPAND_OPS = [
    ("row_expand_mul", tensor_ops.row_expand_mul, tile_ops.row_expand_mul),
    ("row_expand_div", tensor_ops.row_expand_div, tile_ops.row_expand_div),
    ("row_expand_add", tensor_ops.row_expand_add, tile_ops.row_expand_add),
    ("row_expand_sub", tensor_ops.row_expand_sub, tile_ops.row_expand_sub),
    ("row_expand", tensor_ops.row_expand, tile_ops.row_expand),
]
_COL_EXPAND_OPS = [
    ("col_expand_mul", tensor_ops.col_expand_mul, tile_ops.col_expand_mul),
    ("col_expand_div", tensor_ops.col_expand_div, tile_ops.col_expand_div),
    ("col_expand_sub", tensor_ops.col_expand_sub, tile_ops.col_expand_sub),
    ("col_expand", tensor_ops.col_expand, tile_ops.col_expand),
]


class TestConvertTensorToTileOps:
    """Test ConvertTensorToTileOps pass."""

    @pytest.mark.parametrize(
        ("rhs_kind", "tensor_factory", "tile_factory"),
        [
            ("same_input", lambda ins: tensor_ops.add(ins[0], ins[0]), lambda ts: tile_ops.add(ts[0], ts[0])),
            ("two_inputs", lambda ins: tensor_ops.add(ins[0], ins[1]), lambda ts: tile_ops.add(ts[0], ts[1])),
            ("scalar", lambda ins: tensor_ops.adds(ins[0], 1.0), lambda ts: tile_ops.adds(ts[0], 1.0)),
        ],
    )
    def test_elementwise_add_1d(self, rhs_kind, tensor_factory, tile_factory):
        """tensor.add(x, x|y|scalar) -> tile.load(s) + tile.add/adds + tile.store on 1D FP32."""
        in_specs: list[InSpec] = [("x", [64], DataType.FP32)]
        if rhs_kind == "two_inputs":
            in_specs.append(("y", [64], DataType.FP32))
        before, expected = _make_pair(
            in_specs=in_specs,
            out_shape=[64],
            out_dtype=DataType.FP32,
            tensor_op=tensor_factory,
            tile_op=tile_factory,
        )
        _assert_convert_equal(before, expected)

    @pytest.mark.parametrize(("op_name", "tensor_op", "tile_op"), _UNARY_1D_OPS)
    def test_unary_op_1d(self, op_name, tensor_op, tile_op):
        """tensor.<unary>(x) -> tile.<unary>(x_tile) on 1D FP32."""
        before, expected = _make_pair(
            in_specs=[("x", [64], DataType.FP32)],
            out_shape=[64],
            out_dtype=DataType.FP32,
            tensor_op=lambda ins, op=tensor_op: op(ins[0]),
            tile_op=lambda ts, op=tile_op: op(ts[0]),
        )
        _assert_convert_equal(before, expected)

    def test_chained_ops(self):
        """Sequential tensor ops -> correct substitution chain."""
        in_specs: list[InSpec] = [("x", [64], DataType.FP32)]

        def before_body(ib, ins):
            y = ib.let("y", tensor_ops.add(ins[0], ins[0]))
            return ib.let("z", tensor_ops.mul(y, y))

        def expected_body(ib, tiles):
            y = ib.let("y_tile", tile_ops.add(tiles[0], tiles[0]))
            return ib.let("z_tile", tile_ops.mul(y, y))

        before = _make_before(in_specs=in_specs, out_shape=[64], out_dtype=DataType.FP32, body=before_body)
        expected = _make_expected(
            in_specs=in_specs, out_shape=[64], out_dtype=DataType.FP32, body=expected_body
        )
        _assert_convert_equal(before, expected)

    def test_orchestration_unchanged(self):
        """Non-InCore functions pass through unchanged."""

        @pl.program
        class Before:
            @pl.function
            def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Before)

    def test_2d_tensor(self):
        """2D tensor -> correct offsets and shapes for load/store."""
        before, expected = _make_pair(
            in_specs=[("x", [32, 64], DataType.FP16)],
            out_shape=[32, 64],
            out_dtype=DataType.FP16,
            tensor_op=lambda ins: tensor_ops.add(ins[0], ins[0]),
            tile_op=lambda ts: tile_ops.add(ts[0], ts[0]),
        )
        _assert_convert_equal(before, expected)

    def test_rsqrt_high_precision_conversion(self):
        """tensor.rsqrt(high_precision=True) allocates a tmp tile and lowers to 2-arg tile.rsqrt."""
        in_specs: list[InSpec] = [("x", [64], DataType.FP32)]

        def expected_body(ib, tiles):
            tmp = ib.let("rsqrt_tmp", tile_ops.create([64], DataType.FP32))
            return ib.let("y_tile", tile_ops.rsqrt(tiles[0], tmp))

        before = _make_before(
            in_specs=in_specs,
            out_shape=[64],
            out_dtype=DataType.FP32,
            body=lambda ib, ins: ib.let("y", tensor_ops.rsqrt(ins[0], high_precision=True)),
        )
        expected = _make_expected(
            in_specs=in_specs, out_shape=[64], out_dtype=DataType.FP32, body=expected_body
        )
        _assert_convert_equal(before, expected)

    @pytest.mark.parametrize(("op_name", "tensor_op", "tile_op"), _ROW_EXPAND_OPS)
    def test_row_expand_family(self, op_name, tensor_op, tile_op):
        """tensor.<row_expand*>(x, rv) -> load(x) + load(rv) + tile.<row_expand*> + store."""
        before, expected = _make_pair(
            in_specs=[("x", [32, 64], DataType.FP16), ("rv", [32, 1], DataType.FP16)],
            out_shape=[32, 64],
            out_dtype=DataType.FP16,
            tensor_op=lambda ins, op=tensor_op: op(ins[0], ins[1]),
            tile_op=lambda ts, op=tile_op: op(ts[0], ts[1]),
        )
        _assert_convert_equal(before, expected)

    @pytest.mark.parametrize(("op_name", "tensor_op", "tile_op"), _COL_EXPAND_OPS)
    def test_col_expand_family(self, op_name, tensor_op, tile_op):
        """tensor.<col_expand*>(x, cv) -> load(x) + load(cv) + tile.<col_expand*> + store."""
        before, expected = _make_pair(
            in_specs=[("x", [32, 64], DataType.FP16), ("cv", [1, 64], DataType.FP16)],
            out_shape=[32, 64],
            out_dtype=DataType.FP16,
            tensor_op=lambda ins, op=tensor_op: op(ins[0], ins[1]),
            tile_op=lambda ts, op=tile_op: op(ts[0], ts[1]),
        )
        _assert_convert_equal(before, expected)

    @pytest.mark.parametrize(
        ("name", "rhs_shape", "dtype", "b_trans"),
        [
            ("no_trans", [128, 64], DataType.FP16, False),
            ("b_trans", [64, 128], DataType.BF16, True),
        ],
    )
    def test_matmul(self, name, rhs_shape, dtype, b_trans):
        """tensor.matmul[, b_trans] -> tile.load(Mat) for both operands + tile.matmul + store.

        ``lhs`` always loads with one shape arg (no valid_shape, no transpose); ``rhs``
        always loads with valid_shape and a ``transpose`` flag mirroring ``b_trans``.
        """
        lhs_shape = [16, 128]
        out_shape = [16, 64]
        in_specs: list[InSpec] = [("lhs", lhs_shape, dtype), ("rhs", rhs_shape, dtype)]

        def before_body(ib, ins):
            return ib.let("y", tensor_ops.matmul(ins[0], ins[1], b_trans=b_trans))

        def expected_body(ib, params):
            lhs_p, rhs_p = params
            lhs_mat = ib.let(
                "lhs_mat", tile_ops.load(lhs_p, [0, 0], lhs_shape, target_memory=MemorySpace.Mat)
            )
            rhs_mat = ib.let(
                "rhs_mat",
                tile_ops.load(
                    rhs_p, [0, 0], rhs_shape, rhs_shape, target_memory=MemorySpace.Mat, transpose=b_trans
                ),
            )
            return ib.let("y_tile", tile_ops.matmul(lhs_mat, rhs_mat))

        before = _make_before(in_specs=in_specs, out_shape=out_shape, out_dtype=dtype, body=before_body)
        expected = _make_expected(
            in_specs=in_specs, out_shape=out_shape, out_dtype=dtype, body=expected_body, preload=False
        )
        _assert_convert_equal(before, expected)

    def test_matmul_acc_conversion(self):
        """tensor.matmul + tensor.matmul_acc -> tile.matmul + tile.matmul_acc.

        Verifies that tensor.matmul_acc is converted to tile.matmul_acc,
        with lhs/rhs loaded to Mat space and acc passed through from matmul result.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[16, 128], pl.FP32],
                rhs: pl.Tensor[[128, 64], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                acc: pl.Tensor[[16, 64], pl.FP32] = pl.matmul(lhs, rhs)
                result: pl.Tensor[[16, 64], pl.FP32] = pl.matmul_acc(acc, lhs, rhs)
                return result

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 128], pl.FP32],
                rhs: pl.Tensor[[128, 64], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                result: pl.Tensor[[16, 64], pl.FP32] = self.main_incore_0(lhs, rhs)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir_str = str(After)
        assert "tile.matmul_acc" in ir_str
        assert "tile.matmul" in ir_str

    def test_assemble_tile_tile_then_cast_conversion(self):
        """tensor.create + tensor.assemble(tile,tile) + tensor.cast must not crash.

        Regression test: tensor.create → tile.create, so the subsequent
        tensor.assemble sees both args as tiles → tile.assemble (stays TileType).
        The following tensor.cast then sees a tile input and converts to tile.cast
        without error.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[1, 32], pl.FP32],
                b: pl.Tensor[[1, 32], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.BF16]:
                t: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor([1, 64], dtype=pl.FP32)
                t_1: pl.Tensor[[1, 64], pl.FP32] = pl.assemble(t, a, [0, 0])
                t_2: pl.Tensor[[1, 64], pl.FP32] = pl.assemble(t_1, b, [0, 32])
                out: pl.Tensor[[1, 64], pl.BF16] = pl.cast(t_2, target_type=pl.BF16)
                return out

            @pl.function
            def main(
                self,
                a: pl.Tensor[[1, 32], pl.FP32],
                b: pl.Tensor[[1, 32], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.BF16]:
                out: pl.Tensor[[1, 64], pl.BF16] = self.main_incore_0(a, b)
                return out

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir_str = str(After)
        assert "tile.assemble" in ir_str
        assert "tile.cast" in ir_str

    def test_returned_assemble_loop_naive_conversion(self):
        """ConvertTensorToTileOps rewrites assemble loop to store loop with Out param.

        The tensor.assemble inside the loop is converted to tile.store, and the
        Out param becomes the iter-arg init value (dead tensor.create is removed).
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[1, 32], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
                buf: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor([1, 64], dtype=pl.FP32)
                for i, (acc,) in pl.range(2, init_values=(buf,)):
                    off: pl.Scalar[pl.INDEX] = i * 32
                    chunk: pl.Tensor[[1, 32], pl.FP32] = pl.slice(x, [1, 32], [0, 0])
                    acc_next: pl.Tensor[[1, 64], pl.FP32] = pl.assemble(acc, chunk, [0, off])
                    result = pl.yield_(acc_next)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[1, 32], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
                y: pl.Tensor[[1, 64], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[1, 32], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                for i, (acc,) in pl.range(2, init_values=(ret0__out,)):
                    off: pl.Scalar[pl.INDEX] = i * 32
                    chunk__tile: pl.Tile[[1, 32], pl.FP32] = pl.load(
                        x, [0, 0], [1, 32], [1, 32], target_memory=pl.MemorySpace.Vec, transpose=False
                    )
                    acc_next__tile: pl.Tensor[[1, 64], pl.FP32] = pl.store(chunk__tile, [0, off], acc)
                    result: pl.Tensor[[1, 64], pl.FP32] = pl.yield_(acc_next__tile)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[1, 32], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
                ret0__out: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor(
                    [1, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                y: pl.Tensor[[1, 64], pl.FP32] = self.main_incore_0(x, ret0__out)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_returned_assemble_loop_keeps_live_init_assignment(self):
        """Keep the init assignment when the rewritten loop body still references it."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[1, 64], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
                buf: pl.Tensor[[1, 64], pl.FP32] = x
                for i, (acc,) in pl.range(2, init_values=(buf,)):
                    off: pl.Scalar[pl.INDEX] = i * 32
                    chunk: pl.Tensor[[1, 32], pl.FP32] = pl.slice(buf, [1, 32], [0, off])
                    acc_next: pl.Tensor[[1, 64], pl.FP32] = pl.assemble(acc, chunk, [0, off])
                    result = pl.yield_(acc_next)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[1, 64], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
                y: pl.Tensor[[1, 64], pl.FP32] = self.main_incore_0(x)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        after_src = After.as_python()

        assert "buf: pl.Tensor[[1, 64], pl.FP32] = x" in after_src
        assert "init_values=(buf,)" in after_src
        assert "pl.tile.store(" in after_src

    def test_returned_assemble_loop_treats_chunk_expr_as_iter_arg_use(self):
        """Chunk expressions must count as loop-carried uses during rewrite checks."""

        span = ir.Span.unknown()
        idx_type = ir.ScalarType(DataType.INDEX)
        small_tensor_type = ir.TensorType([1, 32], DataType.FP32)
        large_tensor_type = ir.TensorType([1, 64], DataType.FP32)

        x = ir.Var("x", small_tensor_type, span)
        buf_init = ir.Var("buf_init", large_tensor_type, span)
        loop_var = ir.Var("i", idx_type, span)
        iter_arg = ir.IterArg("acc", large_tensor_type, buf_init, span)
        yielded_var = ir.Var("acc_next", large_tensor_type, span)
        return_var = ir.Var("result", large_tensor_type, span)
        inner_loop_var = ir.Var("j", idx_type, span)

        inner_loop = ir.ForStmt(
            inner_loop_var,
            ir.ConstInt(0, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [],
            ir.SeqStmts([], span),
            [],
            span,
            chunk_size=iter_arg,
        )
        assemble_stmt = ir.AssignStmt(yielded_var, ir.op.tensor.assemble(iter_arg, x, [0, 0]), span)
        outer_loop = ir.ForStmt(
            loop_var,
            ir.ConstInt(0, DataType.INDEX, span),
            ir.ConstInt(2, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [iter_arg],
            ir.SeqStmts([inner_loop, assemble_stmt, ir.YieldStmt([yielded_var], span)], span),
            [return_var],
            span,
        )
        func = ir.Function(
            "main_incore_0",
            [x, buf_init],
            [large_tensor_type],
            ir.SeqStmts([outer_loop, ir.ReturnStmt([return_var], span)], span),
            span,
            ir.FunctionType.InCore,
        )
        with pytest.raises(ValueError, match="tensor\\.assemble"):
            passes.convert_tensor_to_tile_ops()(ir.Program([func], "ChunkUseProg", span))

    def test_no_spurious_loads_for_explicit_tile_ops(self):
        """Regression test for #334: no redundant Vec loads when params are consumed by tile ops only.

        When an InCore function explicitly loads tensors to Mat space and uses
        tile.move/tile.matmul/tile.store (none of which are converted tensor ops),
        the pass must NOT insert extra Vec-space tile.load ops for the tensor parameters.
        The output IR must be structurally identical to the input IR.
        """

        @pl.program
        class QKMatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def qk_matmul(
                self,
                qi_0: pl.Tensor[[16, 128], pl.BF16],
                kj_t_0: pl.Tensor[[128, 128], pl.BF16],
                sij_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                qi_l1_0: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    qi_0, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                kj_l1_0: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    kj_t_0, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                qi_l0a_0: pl.Tile[[16, 128], pl.BF16] = pl.move(qi_l1_0, target_memory=pl.MemorySpace.Left)
                kj_l0b_0: pl.Tile[[128, 128], pl.BF16] = pl.move(kj_l1_0, target_memory=pl.MemorySpace.Right)
                sij_l0c_0: pl.Tile[[16, 128], pl.FP32] = pl.matmul(qi_l0a_0, kj_l0b_0)
                out_sij_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(sij_l0c_0, [0, 0], output_tensor=sij_0)
                return out_sij_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                qi_0: pl.Tensor[[16, 128], pl.BF16],
                kj_t_0: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_sij_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                out_sij_1: pl.Tensor[[16, 128], pl.FP32] = self.qk_matmul(qi_0, kj_t_0, out_sij_0)
                return out_sij_1

        After = passes.convert_tensor_to_tile_ops()(QKMatmulProgram)
        ir.assert_structural_equal(After, QKMatmulProgram)

    def test_expand_clone_dim0_conversion(self):
        """tensor.expand_clone (dim0 broadcast) -> looped tile.store into target."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                src: pl.Tensor[[1, 4, 8], pl.FP16],
                target: pl.Tensor[[2, 4, 8], pl.FP16],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                y: pl.Tensor[[2, 4, 8], pl.FP16] = pl.expand_clone(src, target)
                return y

            @pl.function
            def main(
                self,
                src: pl.Tensor[[1, 4, 8], pl.FP16],
                target: pl.Tensor[[2, 4, 8], pl.FP16],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                y: pl.Tensor[[2, 4, 8], pl.FP16] = self.main_incore_0(src, target)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                src: pl.Tensor[[1, 4, 8], pl.FP16],
                target: pl.Out[pl.Tensor[[2, 4, 8], pl.FP16]],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                expand_clone_input: pl.Tile[[1, 4, 8], pl.FP16] = pl.load(src, [0, 0, 0], [1, 4, 8])
                for i, (expand_clone_acc,) in pl.range(2, init_values=(target,)):
                    expand_clone_d0_store: pl.Tensor[[2, 4, 8], pl.FP16] = pl.store(
                        expand_clone_input, [i, 0, 0], expand_clone_acc
                    )
                    expand_clone_d0_result = pl.yield_(expand_clone_d0_store)
                y_tile: pl.Tensor[[2, 4, 8], pl.FP16] = expand_clone_d0_result
                return y_tile

            @pl.function
            def main(
                self,
                src: pl.Tensor[[1, 4, 8], pl.FP16],
                target: pl.Tensor[[2, 4, 8], pl.FP16],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                y: pl.Tensor[[2, 4, 8], pl.FP16] = self.main_incore_0(src, target)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_expand_clone_dim1_conversion(self):
        """tensor.expand_clone (dim1 broadcast) -> per-row load + col_expand + store."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                src: pl.Tensor[[2, 1, 8], pl.FP16],
                target: pl.Tensor[[2, 4, 8], pl.FP16],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                y: pl.Tensor[[2, 4, 8], pl.FP16] = pl.expand_clone(src, target)
                return y

            @pl.function
            def main(
                self,
                src: pl.Tensor[[2, 1, 8], pl.FP16],
                target: pl.Tensor[[2, 4, 8], pl.FP16],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                y: pl.Tensor[[2, 4, 8], pl.FP16] = self.main_incore_0(src, target)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                src: pl.Tensor[[2, 1, 8], pl.FP16],
                target: pl.Out[pl.Tensor[[2, 4, 8], pl.FP16]],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                for i, (expand_clone_acc,) in pl.range(2, init_values=(target,)):
                    expand_clone_d1_input: pl.Tile[[1, 1, 8], pl.FP16] = pl.load(src, [i, 0, 0], [1, 1, 8])
                    expand_clone_d1_target: pl.Tile[[1, 4, 8], pl.FP16] = pl.tile.create(
                        [1, 4, 8], dtype=pl.FP16
                    )
                    expand_clone_d1_col: pl.Tile[[1, 4, 8], pl.FP16] = pl.tile.col_expand(
                        expand_clone_d1_target, expand_clone_d1_input
                    )
                    expand_clone_d1_store: pl.Tensor[[2, 4, 8], pl.FP16] = pl.store(
                        expand_clone_d1_col, [i, 0, 0], expand_clone_acc
                    )
                    expand_clone_d1_result = pl.yield_(expand_clone_d1_store)
                y_tile: pl.Tensor[[2, 4, 8], pl.FP16] = expand_clone_d1_result
                return y_tile

            @pl.function
            def main(
                self,
                src: pl.Tensor[[2, 1, 8], pl.FP16],
                target: pl.Tensor[[2, 4, 8], pl.FP16],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                y: pl.Tensor[[2, 4, 8], pl.FP16] = self.main_incore_0(src, target)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_expand_clone_dim2_conversion(self):
        """tensor.expand_clone (dim2 broadcast) -> row_expand + store."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                src: pl.Tensor[[2, 4, 1], pl.FP16],
                target: pl.Tensor[[2, 4, 8], pl.FP16],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                y: pl.Tensor[[2, 4, 8], pl.FP16] = pl.expand_clone(src, target)
                return y

            @pl.function
            def main(
                self,
                src: pl.Tensor[[2, 4, 1], pl.FP16],
                target: pl.Tensor[[2, 4, 8], pl.FP16],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                y: pl.Tensor[[2, 4, 8], pl.FP16] = self.main_incore_0(src, target)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                src: pl.Tensor[[2, 4, 1], pl.FP16],
                target: pl.Out[pl.Tensor[[2, 4, 8], pl.FP16]],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                expand_clone_input: pl.Tile[[2, 4, 1], pl.FP16] = pl.load(src, [0, 0, 0], [2, 4, 1])
                expand_clone_d2_target: pl.Tile[[2, 4, 8], pl.FP16] = pl.tile.create([2, 4, 8], dtype=pl.FP16)
                expand_clone_d2_row: pl.Tile[[2, 4, 8], pl.FP16] = pl.tile.row_expand(
                    expand_clone_d2_target, expand_clone_input
                )
                y_tile: pl.Tensor[[2, 4, 8], pl.FP16] = pl.store(expand_clone_d2_row, [0, 0, 0], target)
                return y_tile

            @pl.function
            def main(
                self,
                src: pl.Tensor[[2, 4, 1], pl.FP16],
                target: pl.Tensor[[2, 4, 8], pl.FP16],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                y: pl.Tensor[[2, 4, 8], pl.FP16] = self.main_incore_0(src, target)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)


class TestNestedControlFlow:
    """Test ConvertTensorToTileOps with nested control flow."""

    def test_incore_with_if_branch(self):
        """Tensor ops inside IfStmt in InCore -> tile ops in both branches."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                if n == 0:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    z = pl.yield_(y)
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                    z = pl.yield_(y)
                return z

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                z = self.main_incore_0(n, x)
                return z

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                n: pl.Scalar[pl.INT64],
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                if n == 0:
                    y_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(x_tile, x_tile)
                    z = pl.yield_(y_tile)
                else:
                    y_tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                    z = pl.yield_(y_tile)
                out_0_store: pl.Tensor[[64], pl.FP32] = pl.store(z, [0], out_0)
                return out_0_store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(n, x, out_0)
                return z

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_call_inside_for_loop(self):
        """Call to InCore function inside ForStmt -> tensor.create inside loop."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, acc: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(acc, acc)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(3, init_values=(x,)):
                    y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc)
                    result = pl.yield_(y)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc_tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(acc_tile, acc_tile)
                out_0_store: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0_store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(3, init_values=(x,)):
                    out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, out_0)
                    result = pl.yield_(y)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_nested_both_sides(self):
        """Both InCore (IfStmt) and orchestration (ForStmt) have nested control flow."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, acc: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                if n == 0:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(acc, acc)
                    z = pl.yield_(y)
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(acc, acc)
                    z = pl.yield_(y)
                return z

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(3, init_values=(x,)):
                    z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, n)
                    result = pl.yield_(z)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc_tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                if n == 0:
                    y_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(acc_tile, acc_tile)
                    z = pl.yield_(y_tile)
                else:
                    y_tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(acc_tile, acc_tile)
                    z = pl.yield_(y_tile)
                out_0_store: pl.Tensor[[64], pl.FP32] = pl.store(z, [0], out_0)
                return out_0_store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(3, init_values=(x,)):
                    out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, n, out_0)
                    result = pl.yield_(z)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_missing_conversion_raises_error(self):
        """TensorOp with no registered converter raises an error when encountered in InCore body."""
        span = ir.Span.unknown()
        tensor_type = ir.TensorType([4], DataType.FP32)

        x_param = ir.Var("x", tensor_type, span)
        call = ir.create_op_call("test.tensor_op_no_conv", [x_param], {}, span)
        y_var = ir.Var("y", tensor_type, span)
        body = ir.SeqStmts(
            [
                ir.AssignStmt(y_var, call, span),
                ir.ReturnStmt([y_var], span),
            ],
            span,
        )
        func = ir.Function("incore", [x_param], [tensor_type], body, span, ir.FunctionType.InCore)
        prog = ir.Program([func], "test_program", span)

        with pytest.raises(Exception, match="has no registered tile conversion"):
            passes.convert_tensor_to_tile_ops()(prog)

    def test_iter_arg_init_from_tensor_param_gets_preloaded(self):
        """Tensor parameter used only as ForStmt iter_arg initValue must be pre-loaded.

        Regression test: when a TensorType function parameter is used exclusively as
        the init value of a ForStmt iter_arg (not as a direct argument to any converted
        op), Phase 1 must still insert a tile.load for it. Otherwise the iter_arg stays
        TensorType and later tile ops (e.g. tile.add) fail type-checking.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                for i, (running_sum,) in pl.range(3, init_values=(acc,)):
                    new_sum: pl.Tensor[[64], pl.FP32] = pl.add(running_sum, x)
                    result = pl.yield_(new_sum)
                return result

            @pl.function
            def main(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, x)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc_tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                for i, (running_sum,) in pl.range(3, init_values=(acc_tile,)):
                    new_sum_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(running_sum, x_tile)
                    result = pl.yield_(new_sum_tile)
                out_0_store: pl.Tensor[[64], pl.FP32] = pl.store(result, [0], out_0)
                return out_0_store

            @pl.function
            def main(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, x, out_0)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_iter_arg_return_naive_conversion(self):
        """Naive ConvertTensorToTileOps: iter-arg returns get Out params + tensor.create.

        The iter-arg → InOut optimization is handled by OptimizeOrchTensors (Pattern 1).
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                if n == 0:
                    ra: pl.Tensor[[64], pl.FP32] = a
                    rb: pl.Tensor[[64], pl.FP32] = b
                    phi_a, phi_b = pl.yield_(ra, rb)
                else:
                    ra: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                    rb: pl.Tensor[[64], pl.FP32] = pl.mul(a, b)
                    phi_a, phi_b = pl.yield_(ra, rb)
                return phi_a, phi_b

            @pl.function
            def main(
                self,
                a0: pl.Tensor[[64], pl.FP32],
                b0: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                for i, (a, b) in pl.range(3, init_values=(a0, b0)):
                    result: tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]] = self.main_incore_0(
                        a, b, n
                    )
                    new_a: pl.Tensor[[64], pl.FP32] = result[0]
                    new_b: pl.Tensor[[64], pl.FP32] = result[1]
                    out_a, out_b = pl.yield_(new_a, new_b)
                return out_a, out_b

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
                ret1__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                a__tile: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
                b__tile: pl.Tile[[64], pl.FP32] = pl.load(b, [0], [64])
                if n == 0:
                    ra: pl.Tile[[64], pl.FP32] = a__tile
                    rb: pl.Tile[[64], pl.FP32] = b__tile
                    phi_a, phi_b = pl.yield_(ra, rb)
                else:
                    ra__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(a__tile, b__tile)
                    rb__tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(a__tile, b__tile)
                    phi_a, phi_b = pl.yield_(ra__tile, rb__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(phi_a, [0], ret0__out)
                ret1__store: pl.Tensor[[64], pl.FP32] = pl.store(phi_b, [0], ret1__out)
                return ret0__store, ret1__store

            @pl.function
            def main(
                self,
                a0: pl.Tensor[[64], pl.FP32],
                b0: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                for i, (a, b) in pl.range(3, init_values=(a0, b0)):
                    ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    ret1__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    result: tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]] = self.main_incore_0(
                        a, b, n, ret0__out, ret1__out
                    )
                    new_a: pl.Tensor[[64], pl.FP32] = result[0]
                    new_b: pl.Tensor[[64], pl.FP32] = result[1]
                    out_a, out_b = pl.yield_(new_a, new_b)
                return out_a, out_b

        AfterConvert = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(AfterConvert, Expected)


class TestGmLocalTensorConversion:
    """Test gm_tensor vs local_tensor differentiated conversion."""

    @pytest.mark.parametrize("with_valid_shape", [False, True], ids=["plain", "valid_shape"])
    def test_local_tensor_slice_to_tile_slice(self, with_valid_shape):
        """local_tensor.slice (tensor.create result) -> tile.slice (optionally with valid_shape)."""
        in_specs: list[InSpec] = [("x", [8, 32], DataType.FP32)]
        extra_specs: list[ExtraSpec] = (
            [("valid_n", ir.ScalarType(DataType.INDEX))] if with_valid_shape else []
        )

        def _valid_shape(extras):
            # Return ``valid_shape`` list when this variant uses it, else ``None``
            # so we can pass it as a real keyword arg (avoids ``**kwargs`` which
            # confuses pyright's overload matching of ``slice``'s ``span`` param).
            return [8, extras[0]] if with_valid_shape else None

        def before_body(ib, ins, extras=()):
            t = ib.let("t", tensor_ops.create([16, 64], DataType.FP32))
            s = ib.let("s", tensor_ops.slice(t, [8, 32], [0, 0], valid_shape=_valid_shape(extras)))
            return ib.let("y", tensor_ops.add(s, ins[0]))

        def expected_body(ib, tiles, extras=()):
            t_tile = ib.let("t_tile", tile_ops.create([16, 64], DataType.FP32))
            s_tile = ib.let(
                "s_tile", tile_ops.slice(t_tile, [8, 32], [0, 0], valid_shape=_valid_shape(extras))
            )
            return ib.let("y_tile", tile_ops.add(s_tile, tiles[0]))

        before = _make_before(
            in_specs=in_specs,
            out_shape=[8, 32],
            out_dtype=DataType.FP32,
            body=before_body,
            extra_specs=extra_specs,
        )
        expected = _make_expected(
            in_specs=in_specs,
            out_shape=[8, 32],
            out_dtype=DataType.FP32,
            body=expected_body,
            extra_specs=extra_specs,
        )
        _assert_convert_equal(before, expected)

    def test_local_tensor_slice_with_pad_value_forwards_to_tile_slice(self):
        """tensor.slice(..., pad_value=X) on a local tensor lowers to tile.slice(..., pad_value=X)."""
        in_specs: list[InSpec] = [("x", [8, 32], DataType.FP32)]

        def before_body(ib, ins):
            t = ib.let("t", tensor_ops.create([16, 64], DataType.FP32))
            s = ib.let(
                "s",
                tensor_ops.slice(t, [8, 32], [0, 0], valid_shape=[8, 8], pad_value=PadValue.min),
            )
            return ib.let("y", tensor_ops.add(s, ins[0]))

        def expected_body(ib, tiles):
            t_tile = ib.let("t_tile", tile_ops.create([16, 64], DataType.FP32))
            s_tile = ib.let(
                "s_tile",
                tile_ops.slice(t_tile, [8, 32], [0, 0], valid_shape=[8, 8], pad_value=PadValue.min),
            )
            return ib.let("y_tile", tile_ops.add(s_tile, tiles[0]))

        before = _make_before(in_specs=in_specs, out_shape=[8, 32], out_dtype=DataType.FP32, body=before_body)
        expected = _make_expected(
            in_specs=in_specs, out_shape=[8, 32], out_dtype=DataType.FP32, body=expected_body
        )
        _assert_convert_equal(before, expected)

    def test_tensor_fillpad_converts_to_tile_fillpad(self):
        """tensor.fillpad should lower to tile.fillpad after loading the tensor."""
        before, expected = _make_pair(
            in_specs=[("x", [8, 32], DataType.FP32)],
            out_shape=[8, 32],
            out_dtype=DataType.FP32,
            tensor_op=lambda ins: tensor_ops.fillpad(ins[0], pad_value=PadValue.min),
            tile_op=lambda ts: tile_ops.fillpad(ts[0], pad_value=PadValue.min),
        )
        _assert_convert_equal(before, expected)

    def test_tensor_set_validshape_converts_to_tile_set_validshape(self):
        """tensor.set_validshape should lower to tile.set_validshape via RegisterSimple."""
        before, expected = _make_pair(
            in_specs=[("x", [32, 32], DataType.FP32)],
            out_shape=[32, 32],
            out_dtype=DataType.FP32,
            tensor_op=lambda ins: tensor_ops.set_validshape(ins[0], 16, 24),
            tile_op=lambda ts: tile_ops.set_validshape(ts[0], 16, 24),
        )
        _assert_convert_equal(before, expected)

    @pytest.mark.parametrize(
        ("name", "in_shape", "slice_chain", "tail_add"),
        [
            ("single_with_add", [16, 64], [([8, 32], [0, 0])], True),
            ("double", [32, 64], [([16, 32], [0, 0]), ([4, 8], [0, 0])], False),
            ("triple", [64, 128], [([32, 64], [0, 0]), ([8, 16], [0, 0]), ([2, 4], [0, 0])], False),
            ("double_with_add", [32, 64], [([16, 32], [0, 0]), ([4, 8], [0, 0])], True),
        ],
    )
    def test_consecutive_slice(self, name, in_shape, slice_chain, tail_add):
        """N consecutive tensor.slice: first becomes tile.load, rest become tile.slice.

        With ``tail_add=True`` the final slice is fed into ``tensor.add`` to verify
        the chain composes with downstream tile ops.
        """
        first_shape, _ = slice_chain[0]
        out_shape = slice_chain[-1][0]

        def before_body(ib, ins):
            cur = ins[0]
            for i, (shape, off) in enumerate(slice_chain):
                cur = ib.let(f"s{i + 1}", tensor_ops.slice(cur, shape, off))
            if tail_add:
                cur = ib.let("y", tensor_ops.add(cur, cur))
            return cur

        def expected_body(ib, tiles):
            cur = tiles[0]
            for i, (shape, off) in enumerate(slice_chain[1:], start=2):
                cur = ib.let(f"s{i}_tile", tile_ops.slice(cur, shape, off))
            if tail_add:
                cur = ib.let("y_tile", tile_ops.add(cur, cur))
            return cur

        in_specs: list[InSpec] = [("x", in_shape, DataType.FP32)]
        before = _make_before(
            in_specs=in_specs,
            out_shape=out_shape,
            out_dtype=DataType.FP32,
            body=before_body,
        )
        expected = _make_expected(
            in_specs=in_specs,
            out_shape=out_shape,
            out_dtype=DataType.FP32,
            body=expected_body,
            load_shapes=[first_shape],
            load_names=["s1"],
        )
        _assert_convert_equal(before, expected)

    def test_gm_tensor_read_stays_tensor_read(self):
        """gm_tensor.read (function param) stays as tensor.read, no Phase 1 load."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, config: pl.Tensor[[4], pl.FP32], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                scale: pl.Scalar[pl.FP32] = pl.tensor.read(config, [0])
                y: pl.Tensor[[64], pl.FP32] = pl.mul(x, scale)
                return y

            @pl.function
            def main(
                self, config: pl.Tensor[[4], pl.FP32], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(config, x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                config: pl.Tensor[[4], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                scale_tile: pl.Scalar[pl.FP32] = pl.tensor.read(config, [0])
                y_tile: pl.Tile[[64], pl.FP32] = pl.tile.muls(x_tile, scale_tile)
                out_0_store: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0_store

            @pl.function
            def main(
                self, config: pl.Tensor[[4], pl.FP32], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(config, x, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_local_tensor_read_to_tile_read(self):
        """local_tensor.read (tile from tensor.create) -> tile.read."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Scalar[pl.FP32]:
                t: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                v: pl.Scalar[pl.FP32] = pl.tensor.read(t, [0])
                return v

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Scalar[pl.FP32]:
                v: pl.Scalar[pl.FP32] = self.main_incore_0(x)
                return v

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Scalar[pl.FP32]:
                t_tile: pl.Tile[[64], pl.FP32] = pl.tile.create([64], dtype=pl.FP32)
                v_tile: pl.Scalar[pl.FP32] = pl.tile.read(t_tile, [0])
                return v_tile

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Scalar[pl.FP32]:
                v: pl.Scalar[pl.FP32] = self.main_incore_0(x)
                return v

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_gm_tensor_write_stays_tensor_write(self):
        """tensor.write to a gm_tensor (function parameter) stays as tensor.write.

        The tensor.write op is NOT converted (it operates on GM tensors, not tiles).
        However, UpgradeWrittenTensorParamDirections upgrades the param direction
        from In to Out since the param is written via tensor.write.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                dst: pl.Tensor[[4], pl.FP32],
                val: pl.Scalar[pl.FP32],
            ) -> pl.Scalar[pl.FP32]:
                pl.tensor.write(dst, [0], val)
                return val

            @pl.function
            def main(
                self,
                dst: pl.Tensor[[4], pl.FP32],
                val: pl.Scalar[pl.FP32],
            ) -> pl.Scalar[pl.FP32]:
                result: pl.Scalar[pl.FP32] = self.main_incore_0(dst, val)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                dst: pl.Out[pl.Tensor[[4], pl.FP32]],
                val: pl.Scalar[pl.FP32],
            ) -> pl.Scalar[pl.FP32]:
                pl.tensor.write(dst, [0], val)
                return val

            @pl.function
            def main(
                self,
                dst: pl.Tensor[[4], pl.FP32],
                val: pl.Scalar[pl.FP32],
            ) -> pl.Scalar[pl.FP32]:
                result: pl.Scalar[pl.FP32] = self.main_incore_0(dst, val)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_local_tensor_write_to_tile_write(self):
        """tensor.write to a local_tensor (result of tensor.add) converts to tile.write."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]
            ) -> pl.Scalar[pl.FP32]:
                t: pl.Tensor[[4], pl.FP32] = pl.add(a, b)
                val: pl.Scalar[pl.FP32] = pl.tensor.read(a, [0])
                pl.tensor.write(t, [0], val)
                v: pl.Scalar[pl.FP32] = pl.tensor.read(t, [0])
                return v

            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]) -> pl.Scalar[pl.FP32]:
                v: pl.Scalar[pl.FP32] = self.main_incore_0(a, b)
                return v

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]
            ) -> pl.Scalar[pl.FP32]:
                a_tile: pl.Tile[[4], pl.FP32] = pl.load(a, [0], [4])
                b_tile: pl.Tile[[4], pl.FP32] = pl.load(b, [0], [4])
                t_tile: pl.Tile[[4], pl.FP32] = pl.tile.add(a_tile, b_tile)
                val: pl.Scalar[pl.FP32] = pl.tile.read(a_tile, [0])
                pl.tile.write(t_tile, [0], val)
                v: pl.Scalar[pl.FP32] = pl.tile.read(t_tile, [0])
                return v

            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]) -> pl.Scalar[pl.FP32]:
                v: pl.Scalar[pl.FP32] = self.main_incore_0(a, b)
                return v

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)


class TestSliceMatmulConversion:
    """Test tensor.slice + tensor.matmul conversion patterns.

    When a tensor.slice result feeds into tensor.matmul, the slice should produce
    tile.load(Mat, transpose=...) instead of tile.load(Vec), and the matmul should
    skip its own load for that operand (using the tile directly for move + matmul).
    """

    @pytest.mark.parametrize(
        ("name", "lhs_shape", "rhs_shape", "out_shape", "slice_side", "trans_kw"),
        [
            ("no_trans", [16, 128], [128, 64], [16, 64], "rhs", None),
            ("btrans", [1, 128], [120, 128], [1, 120], "rhs", "b_trans"),
            ("atrans", [128, 16], [128, 64], [16, 64], "lhs", "a_trans"),
        ],
    )
    def test_slice_then_matmul(self, name, lhs_shape, rhs_shape, out_shape, slice_side, trans_kw):
        """slice + matmul -> tile.load(Mat, transpose=...) for sliced operand + load + matmul.

        ``slice_side`` selects which operand of matmul is sliced; ``trans_kw`` selects
        which transpose flag (a_trans/b_trans) is set on the matmul (or ``None`` for none).
        """
        in_specs: list[InSpec] = [("a", lhs_shape, DataType.BF16), ("b", rhs_shape, DataType.BF16)]
        slice_shape = lhs_shape if slice_side == "lhs" else rhs_shape
        slice_trans = trans_kw == "a_trans" if slice_side == "lhs" else trans_kw == "b_trans"

        def before_body(ib, ins):
            a_in, b_in = ins
            if slice_side == "lhs":
                sliced = ib.let("a_slice", tensor_ops.slice(a_in, slice_shape, [0, 0]))
                lhs, rhs = sliced, b_in
            else:
                sliced = ib.let("b_slice", tensor_ops.slice(b_in, slice_shape, [0, 0]))
                lhs, rhs = a_in, sliced
            return ib.let(
                "result",
                tensor_ops.matmul(lhs, rhs, a_trans=(trans_kw == "a_trans"), b_trans=(trans_kw == "b_trans")),
            )

        def expected_body(ib, params):
            a_p, b_p = params
            if slice_side == "lhs":
                sliced_tile = ib.let(
                    "a_slice_tile",
                    tile_ops.load(
                        a_p,
                        [0, 0],
                        lhs_shape,
                        lhs_shape,
                        target_memory=MemorySpace.Mat,
                        transpose=slice_trans,
                    ),
                )
                other_tile = ib.let(
                    "rhs_mat",
                    tile_ops.load(
                        b_p, [0, 0], rhs_shape, rhs_shape, target_memory=MemorySpace.Mat, transpose=False
                    ),
                )
                return ib.let("result_tile", tile_ops.matmul(sliced_tile, other_tile))
            sliced_tile = ib.let(
                "b_slice_tile",
                tile_ops.load(
                    b_p, [0, 0], rhs_shape, rhs_shape, target_memory=MemorySpace.Mat, transpose=slice_trans
                ),
            )
            other_tile = ib.let(
                "lhs_mat",
                tile_ops.load(
                    a_p, [0, 0], lhs_shape, lhs_shape, target_memory=MemorySpace.Mat, transpose=False
                ),
            )
            return ib.let("result_tile", tile_ops.matmul(other_tile, sliced_tile))

        before = _make_before(
            in_specs=in_specs, out_shape=out_shape, out_dtype=DataType.BF16, body=before_body
        )
        expected = _make_expected(
            in_specs=in_specs,
            out_shape=out_shape,
            out_dtype=DataType.BF16,
            body=expected_body,
            preload=False,
        )
        _assert_convert_equal(before, expected)


class TestScatterUpdateConversion:
    """Tests for tensor.scatter_update → tile.scatter_update conversion."""

    def test_scatter_update_local_tile_converts(self):
        """tensor.scatter_update on a local tile buffer converts to tile.scatter_update."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                index: pl.Tensor[[2, 4], pl.INT32],
                src: pl.Tensor[[8, 64], pl.FP16],
            ) -> pl.Tensor[[16, 64], pl.FP16]:
                buf: pl.Tensor[[16, 64], pl.FP16] = pl.create_tensor([16, 64], dtype=pl.FP16)
                result: pl.Tensor[[16, 64], pl.FP16] = pl.scatter_update(buf, -2, index, src)
                return result

            @pl.function
            def main(
                self,
                index: pl.Tensor[[2, 4], pl.INT32],
                src: pl.Tensor[[8, 64], pl.FP16],
            ) -> pl.Tensor[[16, 64], pl.FP16]:
                result: pl.Tensor[[16, 64], pl.FP16] = self.main_incore_0(index, src)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        after_str = str(After)
        assert "tile.scatter_update" in after_str
        assert "tensor.scatter_update" not in after_str

    def test_scatter_update_global_tensor_stays(self):
        """tensor.scatter_update on a global tensor also converts to tile.scatter_update
        because the pass first loads all function-parameter tensors into tiles."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                kv_cache: pl.Tensor[[16, 64], pl.FP16],
                index: pl.Tensor[[2, 4], pl.INT32],
                src: pl.Tensor[[8, 64], pl.FP16],
            ) -> pl.Tensor[[16, 64], pl.FP16]:
                result: pl.Tensor[[16, 64], pl.FP16] = pl.scatter_update(kv_cache, -2, index, src)
                return result

            @pl.function
            def main(
                self,
                kv_cache: pl.Tensor[[16, 64], pl.FP16],
                index: pl.Tensor[[2, 4], pl.INT32],
                src: pl.Tensor[[8, 64], pl.FP16],
            ) -> pl.Tensor[[16, 64], pl.FP16]:
                result: pl.Tensor[[16, 64], pl.FP16] = self.main_incore_0(kv_cache, index, src)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        after_str = str(After)
        # The pass loads all tensor parameters to tiles, so scatter_update becomes tile.scatter_update
        assert "tile.scatter_update" in after_str


class TestTensorFullConversion:
    def test_tensor_full_conversion(self):
        """tensor.full -> tile.full conversion."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tensor[[64], pl.FP32] = pl.full([64], dtype=pl.FP32, value=0.0)
                y: pl.Tensor[[64], pl.FP32] = pl.add(t, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir_str = str(After)
        assert "tile.full" in ir_str
        assert "tensor.full" not in ir_str


class TestAssembleParentStride:
    """Tests for physical stride propagation when assemble is in orchestration."""

    def test_out_param_naive_no_parent_stride(self):
        """Naive ConvertTensorToTileOps: Out param uses tile shape strides, not parent strides.

        The parent-stride optimization is handled by OptimizeOrchTensors (Pattern 2).
        After naive conversion, the Out param has shape [32, 32] with default strides.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                mb: pl.Scalar[pl.INDEX],
                nb: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile: pl.Tensor[[32, 32], pl.FP32] = pl.slice(a, [32, 32], [mb, nb])
                return tile

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for mb, (c_iter,) in pl.range(0, 128, 32, init_values=(c,)):
                    for nb, (c_iter2,) in pl.range(0, 128, 32, init_values=(c_iter,)):
                        result: pl.Tensor[[32, 32], pl.FP32] = self.main_incore_0(a, mb, nb)
                        c_next: pl.Tensor[[128, 128], pl.FP32] = pl.assemble(c_iter2, result, [mb, nb])
                        c_rv = pl.yield_(c_next)
                    c_rv2 = pl.yield_(c_rv)
                return c_rv2

        After = passes.convert_tensor_to_tile_ops()(Before)
        after_src = After.as_python()

        # Naive conversion: Out param has NO TensorView (strides derived from tile shape, not parent)
        # The parent-stride optimization is handled by OptimizeOrchTensors (Pattern 2).
        assert "ret0__out" in after_src, "Out param should be created"
        assert "tensor.create" in after_src, "Orchestration should have tensor.create for Out param"


class TestConvertSortOps:
    """Test conversion of tensor sort ops to tile sort ops."""

    def test_sort32_conversion(self):
        """tensor.sort32 -> tile.load (src, idx) + tile.sort32 + tile.store."""
        before, expected = _make_pair(
            in_specs=[("src", [8, 32], DataType.FP32), ("idx", [8, 32], DataType.UINT32)],
            out_shape=[8, 64],
            out_dtype=DataType.FP32,
            tensor_op=lambda ins: tensor_ops.sort32(ins[0], ins[1]),
            tile_op=lambda ts: tile_ops.sort32(ts[0], ts[1]),
        )
        _assert_convert_equal(before, expected)

    def test_mrgsort_format1_conversion(self):
        """tensor.mrgsort(block_len=...) -> tile.load + tile.mrgsort_format1 + tile.store."""
        before, expected = _make_pair(
            in_specs=[("src", [1, 128], DataType.FP32)],
            out_shape=[1, 128],
            out_dtype=DataType.FP32,
            tensor_op=lambda ins: tensor_ops.mrgsort(ins[0], block_len=64),
            tile_op=lambda ts: tile_ops.mrgsort(ts[0], block_len=64),
        )
        _assert_convert_equal(before, expected)

    def test_mrgsort_format2_conversion(self):
        """tensor.mrgsort(s0..s3) -> tile.loads + tile.create(tmp/executed) + tile.mrgsort_format2 + store."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                s0: pl.Tensor[[1, 128], pl.FP32],
                s1: pl.Tensor[[1, 128], pl.FP32],
                s2: pl.Tensor[[1, 128], pl.FP32],
                s3: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 512], pl.FP32]:
                out: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.mrgsort(s0, s1, s2, s3)
                return out

            @pl.function
            def main(
                self,
                s0: pl.Tensor[[1, 128], pl.FP32],
                s1: pl.Tensor[[1, 128], pl.FP32],
                s2: pl.Tensor[[1, 128], pl.FP32],
                s3: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 512], pl.FP32]:
                out: pl.Tensor[[1, 512], pl.FP32] = self.main_incore_0(s0, s1, s2, s3)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                s0: pl.Tensor[[1, 128], pl.FP32],
                s1: pl.Tensor[[1, 128], pl.FP32],
                s2: pl.Tensor[[1, 128], pl.FP32],
                s3: pl.Tensor[[1, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[1, 512], pl.FP32]],
            ) -> pl.Tensor[[1, 512], pl.FP32]:
                s0_tile: pl.Tile[[1, 128], pl.FP32] = pl.load(s0, [0, 0], [1, 128])
                s1_tile: pl.Tile[[1, 128], pl.FP32] = pl.load(s1, [0, 0], [1, 128])
                s2_tile: pl.Tile[[1, 128], pl.FP32] = pl.load(s2, [0, 0], [1, 128])
                s3_tile: pl.Tile[[1, 128], pl.FP32] = pl.load(s3, [0, 0], [1, 128])
                mrgsort2_tmp: pl.Tile[[1, 512], pl.FP32] = pl.tile.create([1, 512], dtype=pl.FP32)
                mrgsort2_executed: pl.Tile[[1, 4], pl.INT16] = pl.tile.create([1, 4], dtype=pl.INT16)
                out_tile: pl.Tile[[1, 512], pl.FP32] = pl.tile.mrgsort(
                    s0_tile, s1_tile, s2_tile, s3_tile, mrgsort2_tmp, mrgsort2_executed
                )
                out_store: pl.Tensor[[1, 512], pl.FP32] = pl.store(out_tile, [0, 0], out_0)
                return out_store

            @pl.function
            def main(
                self,
                s0: pl.Tensor[[1, 128], pl.FP32],
                s1: pl.Tensor[[1, 128], pl.FP32],
                s2: pl.Tensor[[1, 128], pl.FP32],
                s3: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 512], pl.FP32]:
                out_0: pl.Tensor[[1, 512], pl.FP32] = pl.create_tensor([1, 512], dtype=pl.FP32)
                out: pl.Tensor[[1, 512], pl.FP32] = self.main_incore_0(s0, s1, s2, s3, out_0)
                return out

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)


class TestConvertGatherOp:
    """Test conversion of tensor.gather (MVP: 2D + dim=-1)."""

    def test_gather_conversion(self):
        """tensor.gather -> per-row loop of tile.load + tile.gather + tile.store."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                inp: pl.Tensor[[4, 16], pl.FP32],
                idx: pl.Tensor[[4, 3], pl.INT32],
            ) -> pl.Tensor[[4, 3], pl.FP32]:
                out: pl.Tensor[[4, 3], pl.FP32] = pl.tensor.gather(inp, dim=-1, index=idx)
                return out

            @pl.function
            def main(
                self,
                inp: pl.Tensor[[4, 16], pl.FP32],
                idx: pl.Tensor[[4, 3], pl.INT32],
            ) -> pl.Tensor[[4, 3], pl.FP32]:
                out: pl.Tensor[[4, 3], pl.FP32] = self.main_incore_0(inp, idx)
                return out

        After = passes.convert_tensor_to_tile_ops()(Before)
        after_src = After.as_python()

        # Sanity: tensor.gather is fully lowered; tile.gather appears inside a per-row loop.
        assert "tensor.gather" not in after_src
        assert "tile.gather" in after_src
        assert "pl.range(4" in after_src or "range(4" in after_src
        # The index-form tile.gather needs a [1, 3] INT32 scratch tile.
        assert "dtype=pl.INT32" in after_src
        # Per-row slices: [1, 16] from inp, [1, 3] from idx.
        assert "[1, 16]" in after_src
        assert "[1, 3]" in after_src
        # Phase 3 adds an Out tensor param for the result.
        assert "pl.Out[pl.Tensor[[4, 3]" in after_src

    def test_gather_mask_conversion(self):
        """tensor.gather(mask_pattern=...) -> tile.load + tile.gather_mask + tile.store."""
        before, expected = _make_pair(
            in_specs=[("src", [8, 64], DataType.FP32)],
            out_shape=[8, 32],
            out_dtype=DataType.FP32,
            tensor_op=lambda ins: tensor_ops.gather(ins[0], mask_pattern=1),
            tile_op=lambda ts: tile_ops.gather(ts[0], mask_pattern=1),
        )
        _assert_convert_equal(before, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
