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
import pypto.language.distributed as pld
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


class _FirstCallFinder(ir.IRVisitor):
    """Record the first ``Call`` whose callee ``Op`` has the given name.

    Matches both function-typed calls (``Call.op`` is a ``GlobalVar`` whose
    ``name`` equals the function name) and op-typed calls (``Call.op`` is a
    built-in ``Op``, e.g. ``"tensor.create"``).
    """

    def __init__(self, op_name: str) -> None:
        super().__init__()
        self.op_name = op_name
        self.found: ir.Call | None = None

    def visit_call(self, op: ir.Call) -> None:
        if self.found is None and op.op.name == self.op_name:
            self.found = op
        super().visit_call(op)


def _find_first_call_to(func: ir.Function, op_name: str) -> ir.Call | None:
    """Return the first Call in ``func.body`` whose callee ``Op`` has name
    ``op_name``. Used by wrapper-propagation tests to inspect per-call arg
    counts after the pass rewrites them."""
    finder = _FirstCallFinder(op_name)
    finder.visit_stmt(func.body)
    return finder.found


# ---------------------------------------------------------------------------
# Op family parametrization tables.
# ---------------------------------------------------------------------------

# Unary ops on a 1D 64-element FP32 tensor: tensor.OP(x) -> tile.OP(x_tile).
_UNARY_1D_OPS = [
    ("exp", tensor_ops.exp, tile_ops.exp),
    ("neg", tensor_ops.neg, tile_ops.neg),
    ("recip", tensor_ops.recip, tile_ops.recip),
    ("sqrt", tensor_ops.sqrt, tile_ops.sqrt),
    ("abs", tensor_ops.abs, tile_ops.abs),
    ("sin", tensor_ops.sin, tile_ops.sin),
    ("cos", tensor_ops.cos, tile_ops.cos),
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

    @pytest.mark.parametrize(
        ("rhs_kind", "tensor_factory", "tile_factory"),
        [
            (
                "two_inputs_max",
                lambda ins: tensor_ops.maximum(ins[0], ins[1]),
                lambda ts: tile_ops.maximum(ts[0], ts[1]),
            ),
            (
                "scalar_max",
                lambda ins: tensor_ops.maximum(ins[0], 1.0),
                lambda ts: tile_ops.maximums(ts[0], 1.0),
            ),
            (
                "two_inputs_min",
                lambda ins: tensor_ops.minimum(ins[0], ins[1]),
                lambda ts: tile_ops.minimum(ts[0], ts[1]),
            ),
            (
                "scalar_min",
                lambda ins: tensor_ops.minimum(ins[0], 1.0),
                lambda ts: tile_ops.minimums(ts[0], 1.0),
            ),
        ],
    )
    def test_maximum_minimum_dispatch(self, rhs_kind, tensor_factory, tile_factory):
        """tensor.maximum/minimum dispatches by rhs type to tile.{maximum,minimum}{,s}."""
        in_specs: list[InSpec] = [("x", [64], DataType.FP32)]
        if rhs_kind in ("two_inputs_max", "two_inputs_min"):
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

    def test_transpose_emits_tile_create_plus_transpose(self):
        """tensor.transpose lowers to tile.create + tile.transpose(input, axis1, axis2, tmp)."""
        in_specs: list[InSpec] = [("x", [32, 64], DataType.FP16)]

        def expected_body(ib, tiles):
            tmp = ib.let("transpose_tmp", tile_ops.create([32, 64], DataType.FP16))
            return ib.let("y_tile", tile_ops.transpose(tiles[0], 0, 1, tmp=tmp))

        before = _make_before(
            in_specs=in_specs,
            out_shape=[64, 32],
            out_dtype=DataType.FP16,
            body=lambda ib, ins: ib.let("y", tensor_ops.transpose(ins[0], 0, 1)),
        )
        expected = _make_expected(
            in_specs=in_specs, out_shape=[64, 32], out_dtype=DataType.FP16, body=expected_body
        )
        _assert_convert_equal(before, expected)

    def test_put_emits_tile_create_plus_tile_put(self):
        """pld.tensor.put lowers to tile.create(stage) + pld.tile.put(dst, peer, src, stage)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                dst: pld.DistributedTensor[[16, 64], pl.FP16],
                src: pld.DistributedTensor[[16, 64], pl.FP16],
                peer: pl.Scalar[pl.INT32],
            ):
                pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.None_)

        After = passes.convert_tensor_to_tile_ops()(Before)
        kernel = After.get_function("kernel")
        assert kernel is not None, "kernel function missing after conversion"

        # pld.tensor.put has been replaced.
        assert _find_first_call_to(kernel, "pld.tensor.put") is None, (
            "pld.tensor.put must be lowered to pld.tile.put by ConvertTensorToTileOps"
        )

        # tile.create + pld.tile.put are now present.
        assert _find_first_call_to(kernel, "tile.create") is not None
        put_call = _find_first_call_to(kernel, "pld.tile.put")
        assert put_call is not None, "expected pld.tile.put after conversion"

        # pld.tile.put threads the staging tile as the 4th positional arg.
        assert len(put_call.args) == 4
        stage_arg = put_call.args[3]
        assert isinstance(stage_arg, ir.Var)
        assert stage_arg.name_hint == "tput_stage"

        # Stage tile carries the right shape ([rows, cols] flattening of [16, 64] = [16, 64]),
        # FP16 dtype and the VEC memory space — InitMemRef and AllocateMemoryAddr need these
        # to assign a real UB address.
        stage_type = stage_arg.type
        assert isinstance(stage_type, ir.TileType)
        shape_vals: list[int] = []
        for d in stage_type.shape:
            assert isinstance(d, ir.ConstInt)
            shape_vals.append(d.value)
        assert shape_vals == [16, 64]
        assert stage_type.dtype == pl.FP16
        assert stage_type.memory_space == MemorySpace.Vec

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[16, 128], pl.FP32],
                rhs: pl.Tensor[[128, 64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat = pl.load(
                    lhs, [0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Mat, transpose=False
                )
                rhs_mat = pl.load(
                    rhs, [0, 0], [128, 64], [128, 64], target_memory=pl.Mem.Mat, transpose=False
                )
                acc__tile = pl.tile.matmul(lhs_mat, rhs_mat)
                lhs_mat_1 = pl.load(
                    lhs, [0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Mat, transpose=False
                )
                rhs_mat_1 = pl.load(
                    rhs, [0, 0], [128, 64], [128, 64], target_memory=pl.Mem.Mat, transpose=False
                )
                result__tile = pl.tile.matmul_acc(acc__tile, lhs_mat_1, rhs_mat_1)
                ret0__store = pl.store(result__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 128], pl.FP32],
                rhs: pl.Tensor[[128, 64], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                ret0__out = pl.create_tensor([16, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                result = self.main_incore_0(lhs, rhs, ret0__out)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_matmul_nd_dispatches_to_batch_matmul(self):
        """tensor.matmul with any operand of rank > 2 must lower to tile.batch_matmul.

        Verifies the rank-based dispatch in RegisterMatmulOps: a 2D lhs and a 3D
        rhs (the typical MoE expert weight pattern) produces tile.batch_matmul,
        not tile.matmul. b_trans=True is propagated to the rhs load via the
        existing InputSpaceReq mechanism.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[16, 128], pl.BF16],
                rhs: pl.Tensor[[1, 64, 128], pl.BF16],
            ) -> pl.Tensor[[1, 16, 64], pl.FP32]:
                y: pl.Tensor[[1, 16, 64], pl.FP32] = pl.matmul(lhs, rhs, b_trans=True, out_dtype=pl.FP32)
                return y

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 128], pl.BF16],
                rhs: pl.Tensor[[1, 64, 128], pl.BF16],
            ) -> pl.Tensor[[1, 16, 64], pl.FP32]:
                result: pl.Tensor[[1, 16, 64], pl.FP32] = self.main_incore_0(lhs, rhs)
                return result

        # The rank dispatch picks tile.batch_matmul for the whole chain (no plain
        # tile.matmul). The b_trans flag is pushed into the rhs load as transpose=True
        # (a TileView with col-major slayout), not emitted as a separate tile.transpose.
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[16, 128], pl.BF16],
                rhs: pl.Tensor[[1, 64, 128], pl.BF16],
                ret0__out: pl.Out[pl.Tensor[[1, 16, 64], pl.FP32]],
            ) -> pl.Tensor[[1, 16, 64], pl.FP32]:
                lhs_mat = pl.load(
                    lhs, [0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Mat, transpose=False
                )
                rhs_mat: pl.Tile[
                    [1, 128, 64],
                    pl.BF16,
                    pl.Mem.Mat,
                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major),
                ] = pl.load(
                    rhs, [0, 0, 0], [1, 64, 128], [1, 64, 128], target_memory=pl.Mem.Mat, transpose=True
                )
                y__tile = pl.tile.batch_matmul(lhs_mat, rhs_mat)
                ret0__store = pl.store(y__tile, [0, 0, 0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 128], pl.BF16],
                rhs: pl.Tensor[[1, 64, 128], pl.BF16],
            ) -> pl.Tensor[[1, 16, 64], pl.FP32]:
                ret0__out = pl.create_tensor([1, 16, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                result = self.main_incore_0(lhs, rhs, ret0__out)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_matmul_acc_nd_dispatches_to_batch_matmul_acc(self):
        """tensor.matmul_acc with ND operands lowers to tile.batch_matmul_acc.

        Mirrors the matmul ND dispatch: ND acc + ND lhs/rhs combinations should
        select the batched accumulating tile op so the downstream FlattenTileNdTo2D
        can unroll consistently.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[16, 128], pl.BF16],
                rhs0: pl.Tensor[[1, 64, 128], pl.BF16],
                rhs1: pl.Tensor[[1, 64, 128], pl.BF16],
            ) -> pl.Tensor[[1, 16, 64], pl.FP32]:
                acc: pl.Tensor[[1, 16, 64], pl.FP32] = pl.matmul(lhs, rhs0, b_trans=True, out_dtype=pl.FP32)
                result: pl.Tensor[[1, 16, 64], pl.FP32] = pl.matmul_acc(acc, lhs, rhs1, b_trans=True)
                return result

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 128], pl.BF16],
                rhs0: pl.Tensor[[1, 64, 128], pl.BF16],
                rhs1: pl.Tensor[[1, 64, 128], pl.BF16],
            ) -> pl.Tensor[[1, 16, 64], pl.FP32]:
                result: pl.Tensor[[1, 16, 64], pl.FP32] = self.main_incore_0(lhs, rhs0, rhs1)
                return result

        # ND acc + ND lhs/rhs select the batched accumulating tile op: the matmul
        # becomes tile.batch_matmul and the matmul_acc becomes tile.batch_matmul_acc.
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[16, 128], pl.BF16],
                rhs0: pl.Tensor[[1, 64, 128], pl.BF16],
                rhs1: pl.Tensor[[1, 64, 128], pl.BF16],
                ret0__out: pl.Out[pl.Tensor[[1, 16, 64], pl.FP32]],
            ) -> pl.Tensor[[1, 16, 64], pl.FP32]:
                lhs_mat = pl.load(
                    lhs, [0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Mat, transpose=False
                )
                rhs0_mat: pl.Tile[
                    [1, 128, 64],
                    pl.BF16,
                    pl.Mem.Mat,
                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major),
                ] = pl.load(
                    rhs0, [0, 0, 0], [1, 64, 128], [1, 64, 128], target_memory=pl.Mem.Mat, transpose=True
                )
                acc__tile = pl.tile.batch_matmul(lhs_mat, rhs0_mat)
                lhs_mat_1 = pl.load(
                    lhs, [0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Mat, transpose=False
                )
                rhs1_mat: pl.Tile[
                    [1, 128, 64],
                    pl.BF16,
                    pl.Mem.Mat,
                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major),
                ] = pl.load(
                    rhs1, [0, 0, 0], [1, 64, 128], [1, 64, 128], target_memory=pl.Mem.Mat, transpose=True
                )
                result__tile = pl.tile.batch_matmul_acc(acc__tile, lhs_mat_1, rhs1_mat)
                ret0__store = pl.store(result__tile, [0, 0, 0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 128], pl.BF16],
                rhs0: pl.Tensor[[1, 64, 128], pl.BF16],
                rhs1: pl.Tensor[[1, 64, 128], pl.BF16],
            ) -> pl.Tensor[[1, 16, 64], pl.FP32]:
                ret0__out = pl.create_tensor([1, 16, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                result = self.main_incore_0(lhs, rhs0, rhs1, ret0__out)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

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

        # tensor.create -> tile.create, so the assembles see tile args and stay tiles
        # (tile.assemble), and the trailing tensor.cast over a tile becomes tile.cast.
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[1, 32], pl.FP32],
                b: pl.Tensor[[1, 32], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[1, 64], pl.BF16]],
            ) -> pl.Tensor[[1, 64], pl.BF16]:
                t__tile = pl.tile.create([1, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                assemble_src = pl.load(a, [0, 0], [1, 32], [1, 32], target_memory=pl.Mem.Vec, transpose=False)
                t_1__tile = pl.tile.assemble(t__tile, assemble_src, [0, 0])
                assemble_src_1 = pl.load(
                    b, [0, 0], [1, 32], [1, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                t_2__tile = pl.tile.assemble(t_1__tile, assemble_src_1, [0, 32])
                out__tile = pl.tile.cast(t_2__tile, target_type=pl.BF16, mode="round")
                ret0__store = pl.store(out__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self,
                a: pl.Tensor[[1, 32], pl.FP32],
                b: pl.Tensor[[1, 32], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.BF16]:
                ret0__out = pl.create_tensor([1, 64], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                out = self.main_incore_0(a, b, ret0__out)
                return out

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

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
                    chunk__tile = pl.load(
                        x, [0, 0], [1, 32], [1, 32], target_memory=pl.MemorySpace.Vec, transpose=False
                    )
                    acc_next__tile = pl.store(chunk__tile, [0, off], acc)
                    result = pl.yield_(acc_next__tile)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[1, 32], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
                ret0__out = pl.create_tensor([1, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                y = self.main_incore_0(x, ret0__out)
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

        # The init assignment ``buf = x`` is kept because the rewritten loop body
        # still slices from ``buf``; the param ``x`` becomes InOut (loaded + stored).
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.InOut[pl.Tensor[[1, 64], pl.FP32]]) -> pl.Tensor[[1, 64], pl.FP32]:
                buf: pl.Tensor[[1, 64], pl.FP32] = x
                for i, (acc,) in pl.range(2, init_values=(buf,)):
                    off: pl.Scalar[pl.INDEX] = i * 32
                    chunk__tile = pl.load(
                        buf, [0, off], [1, 32], [1, 32], target_memory=pl.Mem.Vec, transpose=False
                    )
                    acc_next__tile = pl.store(chunk__tile, [0, off], acc)
                    result = pl.yield_(acc_next__tile)
                return result

            @pl.function
            def main(self, x: pl.InOut[pl.Tensor[[1, 64], pl.FP32]]) -> pl.Tensor[[1, 64], pl.FP32]:
                y = self.main_incore_0(x)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

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
                expand_clone_input = pl.load(src, [0, 0, 0], [1, 4, 8])
                for i, (expand_clone_acc,) in pl.range(2, init_values=(target,)):
                    expand_clone_d0_store = pl.store(expand_clone_input, [i, 0, 0], expand_clone_acc)
                    expand_clone_d0_result = pl.yield_(expand_clone_d0_store)
                y_tile: pl.Tensor[[2, 4, 8], pl.FP16] = expand_clone_d0_result
                return y_tile

            @pl.function
            def main(
                self,
                src: pl.Tensor[[1, 4, 8], pl.FP16],
                target: pl.Out[pl.Tensor[[2, 4, 8], pl.FP16]],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                y = self.main_incore_0(src, target)
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
                    expand_clone_d1_input = pl.load(src, [i, 0, 0], [1, 1, 8])
                    expand_clone_d1_target = pl.tile.create([1, 4, 8], dtype=pl.FP16)
                    expand_clone_d1_col = pl.tile.col_expand(expand_clone_d1_target, expand_clone_d1_input)
                    expand_clone_d1_store = pl.store(expand_clone_d1_col, [i, 0, 0], expand_clone_acc)
                    expand_clone_d1_result = pl.yield_(expand_clone_d1_store)
                y_tile: pl.Tensor[[2, 4, 8], pl.FP16] = expand_clone_d1_result
                return y_tile

            @pl.function
            def main(
                self,
                src: pl.Tensor[[2, 1, 8], pl.FP16],
                target: pl.Out[pl.Tensor[[2, 4, 8], pl.FP16]],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                y = self.main_incore_0(src, target)
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
                expand_clone_input = pl.load(src, [0, 0, 0], [2, 4, 1])
                expand_clone_d2_target = pl.tile.create([2, 4, 8], dtype=pl.FP16)
                expand_clone_d2_row = pl.tile.row_expand(expand_clone_d2_target, expand_clone_input)
                y_tile = pl.store(expand_clone_d2_row, [0, 0, 0], target)
                return y_tile

            @pl.function
            def main(
                self,
                src: pl.Tensor[[2, 4, 1], pl.FP16],
                target: pl.Out[pl.Tensor[[2, 4, 8], pl.FP16]],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                y = self.main_incore_0(src, target)
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
                x_tile = pl.load(x, [0], [64])
                if n == 0:
                    y_tile = pl.tile.add(x_tile, x_tile)
                    z = pl.yield_(y_tile)
                else:
                    y_tile = pl.tile.mul(x_tile, x_tile)
                    z = pl.yield_(y_tile)
                out_0_store = pl.store(z, [0], out_0)
                return out_0_store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                out_0 = pl.create_tensor([64], dtype=pl.FP32)
                z = self.main_incore_0(n, x, out_0)
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
                acc_tile = pl.load(acc, [0], [64])
                y_tile = pl.tile.add(acc_tile, acc_tile)
                out_0_store = pl.store(y_tile, [0], out_0)
                return out_0_store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(3, init_values=(x,)):
                    out_0 = pl.create_tensor([64], dtype=pl.FP32)
                    y = self.main_incore_0(acc, out_0)
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
                acc_tile = pl.load(acc, [0], [64])
                if n == 0:
                    y_tile = pl.tile.add(acc_tile, acc_tile)
                    z = pl.yield_(y_tile)
                else:
                    y_tile = pl.tile.mul(acc_tile, acc_tile)
                    z = pl.yield_(y_tile)
                out_0_store = pl.store(z, [0], out_0)
                return out_0_store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(3, init_values=(x,)):
                    out_0 = pl.create_tensor([64], dtype=pl.FP32)
                    z = self.main_incore_0(acc, n, out_0)
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
                acc_tile = pl.load(acc, [0], [64])
                x_tile = pl.load(x, [0], [64])
                for i, (running_sum,) in pl.range(3, init_values=(acc_tile,)):
                    new_sum_tile = pl.tile.add(running_sum, x_tile)
                    result = pl.yield_(new_sum_tile)
                out_0_store = pl.store(result, [0], out_0)
                return out_0_store

            @pl.function
            def main(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                out_0 = pl.create_tensor([64], dtype=pl.FP32)
                result = self.main_incore_0(acc, x, out_0)
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
                a__tile = pl.load(a, [0], [64])
                b__tile = pl.load(b, [0], [64])
                if n == 0:
                    ra: pl.Tile[[64], pl.FP32] = a__tile
                    rb: pl.Tile[[64], pl.FP32] = b__tile
                    phi_a, phi_b = pl.yield_(ra, rb)
                else:
                    ra__tile = pl.tile.add(a__tile, b__tile)
                    rb__tile = pl.tile.mul(a__tile, b__tile)
                    phi_a, phi_b = pl.yield_(ra__tile, rb__tile)
                ret0__store = pl.store(phi_a, [0], ret0__out)
                ret1__store = pl.store(phi_b, [0], ret1__out)
                return ret0__store, ret1__store

            @pl.function
            def main(
                self,
                a0: pl.Tensor[[64], pl.FP32],
                b0: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                for i, (a, b) in pl.range(3, init_values=(a0, b0)):
                    ret0__out = pl.create_tensor([64], dtype=pl.FP32)
                    ret1__out = pl.create_tensor([64], dtype=pl.FP32)
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
                x_tile = pl.load(x, [0], [64])
                scale_tile = pl.tensor.read(config, [0])
                y_tile = pl.tile.muls(x_tile, scale_tile)
                out_0_store = pl.store(y_tile, [0], out_0)
                return out_0_store

            @pl.function
            def main(
                self, config: pl.Tensor[[4], pl.FP32], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                out_0 = pl.create_tensor([64], dtype=pl.FP32)
                y = self.main_incore_0(config, x, out_0)
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
                t_tile = pl.tile.create([64], dtype=pl.FP32)
                v_tile = pl.tile.read(t_tile, [0])
                return v_tile

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Scalar[pl.FP32]:
                v = self.main_incore_0(x)
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
                dst: pl.Out[pl.Tensor[[4], pl.FP32]],
                val: pl.Scalar[pl.FP32],
            ) -> pl.Scalar[pl.FP32]:
                result = self.main_incore_0(dst, val)
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
                a_tile = pl.load(a, [0], [4])
                b_tile = pl.load(b, [0], [4])
                t_tile = pl.tile.add(a_tile, b_tile)
                val = pl.tile.read(a_tile, [0])
                pl.tile.write(t_tile, [0], val)
                v = pl.tile.read(t_tile, [0])
                return v

            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]) -> pl.Scalar[pl.FP32]:
                v = self.main_incore_0(a, b)
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

    def test_slice_alias_then_matmul_routes_load_to_mat(self):
        """tensor.slice → SSA alias → tensor.matmul emits tile.load(Mat).

        Reproduction of the qwen3 decode MLP-down pattern: the parser elides
        `y = x` aliases (e.g. from commented-out `pl.fillpad` wrappers), leaving
        a chain `slice → alias → matmul`. ConsumerSpaceCollector must propagate
        matmul's Mat demand backward through the alias so the slice lowers to
        a Mat-targeted load instead of the default Vec (which otherwise routes
        the whole scope through AIV and breaks the AIC/AIV split).
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                a_slice: pl.Tensor[[16, 128], pl.BF16] = pl.slice(a, [16, 128], [0, 0])
                b_slice: pl.Tensor[[128, 128], pl.BF16] = pl.slice(b, [128, 128], [0, 0])
                a_alias: pl.Tensor[[16, 128], pl.BF16] = a_slice
                b_alias: pl.Tensor[[128, 128], pl.BF16] = b_slice
                c: pl.Tensor[[16, 128], pl.BF16] = pl.matmul(a_alias, b_alias)
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.assemble(out_0, c, [0, 0])
                return out_0

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.create_tensor([16, 128], dtype=pl.BF16)
                return self.main_incore_0(a, b, out_0)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                a_slice__tile = pl.tile.load(
                    a, [0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Mat, transpose=False
                )
                b_slice__tile = pl.tile.load(
                    b, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Mat, transpose=False
                )
                a_alias: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = a_slice__tile
                b_alias: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = b_slice__tile
                c__tile = pl.tile.matmul(a_alias, b_alias)
                out_0__tile = pl.tile.store(c__tile, [0, 0], out_0)
                return out_0__tile

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                out_0 = pl.create_tensor([16, 128], dtype=pl.BF16)
                return self.main_incore_0(a, b, out_0)

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_slice_chain_of_aliases_then_matmul(self):
        """Demand propagates through a chain of SSA aliases, not just one hop.

        Ensures the single reverse-order sweep over ``propagation_edges_`` handles
        transitive closure: slice → alias1 → alias2 → matmul must still reach
        the slice-produced var and push Mat onto the emitted tile.load.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                a_slice: pl.Tensor[[16, 128], pl.BF16] = pl.slice(a, [16, 128], [0, 0])
                a_alias1: pl.Tensor[[16, 128], pl.BF16] = a_slice
                a_alias2: pl.Tensor[[16, 128], pl.BF16] = a_alias1
                c: pl.Tensor[[16, 64], pl.BF16] = pl.matmul(a_alias2, b)
                out_0: pl.Tensor[[16, 64], pl.BF16] = pl.assemble(out_0, c, [0, 0])
                return out_0

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 64], pl.BF16],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                out_0: pl.Tensor[[16, 64], pl.BF16] = pl.create_tensor([16, 64], dtype=pl.BF16)
                return self.main_incore_0(a, b, out_0)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                a_slice__tile = pl.tile.load(
                    a, [0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Mat, transpose=False
                )
                a_alias1: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = a_slice__tile
                a_alias2: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = a_alias1
                b__tile = pl.tile.load(
                    b, [0, 0], [128, 64], [128, 64], target_memory=pl.Mem.Mat, transpose=False
                )
                c__tile = pl.tile.matmul(a_alias2, b__tile)
                out_0__tile = pl.tile.store(c__tile, [0, 0], out_0)
                return out_0__tile

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 64], pl.BF16],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                out_0 = pl.create_tensor([16, 64], dtype=pl.BF16)
                return self.main_incore_0(a, b, out_0)

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)


class TestScatterUpdateConversion:
    """tensor.scatter_update lowers to a whole-row tile.scatter (no tile.scatter_update)."""

    def test_scatter_update_lowers_to_tile_scatter(self):
        """scatter_update expands to flat-index tile.scatter + preserve blend, no scatter_update."""

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

        with passes.PassContext([], passes.VerificationLevel.NONE):
            After = passes.convert_tensor_to_tile_ops()(Before)
        text = ir.python_print(After)
        # Assert exact op presence: scatter_update must lower to the index-form tile.scatter,
        # never the mask-form tile.scatter_mask (substring "tile.scatter" would match both).
        assert "pl.tile.scatter(" in text
        assert "pl.tile.scatter_mask(" not in text
        assert text.count("pl.tile.scatter(") >= 1
        assert "scatter_update" not in text

    def test_scatter_update_fp16_rejects_oversized_flat_index(self):
        """2-byte dst with m*d > 32767 overflows the i16 flat index — must raise, not miscompile."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                index: pl.Tensor[[2, 4], pl.INT32],
                src: pl.Tensor[[8, 256], pl.FP16],
            ) -> pl.Tensor[[128, 256], pl.FP16]:  # m*d = 32768 > 32767
                buf: pl.Tensor[[128, 256], pl.FP16] = pl.create_tensor([128, 256], dtype=pl.FP16)
                result: pl.Tensor[[128, 256], pl.FP16] = pl.scatter_update(buf, -2, index, src)
                return result

            @pl.function
            def main(
                self,
                index: pl.Tensor[[2, 4], pl.INT32],
                src: pl.Tensor[[8, 256], pl.FP16],
            ) -> pl.Tensor[[128, 256], pl.FP16]:
                result: pl.Tensor[[128, 256], pl.FP16] = self.main_incore_0(index, src)
                return result

        with pytest.raises(Exception, match="i16 flat-index limit"):
            with passes.PassContext([], passes.VerificationLevel.NONE):
                passes.convert_tensor_to_tile_ops()(Before)

    def test_scatter_update_rejects_4d(self):
        """4D input type-checks but is not yet lowered — must raise a clear user error, not crash."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                index: pl.Tensor[[2, 4], pl.INT32],
                src: pl.Tensor[[2, 4, 1, 64], pl.FP32],
            ) -> pl.Tensor[[4, 4, 1, 64], pl.FP32]:
                buf: pl.Tensor[[4, 4, 1, 64], pl.FP32] = pl.create_tensor([4, 4, 1, 64], dtype=pl.FP32)
                result: pl.Tensor[[4, 4, 1, 64], pl.FP32] = pl.scatter_update(buf, -2, index, src)
                return result

            @pl.function
            def main(
                self,
                index: pl.Tensor[[2, 4], pl.INT32],
                src: pl.Tensor[[2, 4, 1, 64], pl.FP32],
            ) -> pl.Tensor[[4, 4, 1, 64], pl.FP32]:
                result: pl.Tensor[[4, 4, 1, 64], pl.FP32] = self.main_incore_0(index, src)
                return result

        with pytest.raises(Exception, match="only 2D input/src is currently supported"):
            with passes.PassContext([], passes.VerificationLevel.NONE):
                passes.convert_tensor_to_tile_ops()(Before)


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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x__tile = pl.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
                t__tile = pl.tile.full([64], dtype=pl.FP32, value=0.0)
                y__tile = pl.tile.add(t__tile, x__tile)
                ret0__store = pl.store(y__tile, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                ret0__out = pl.create_tensor([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                y = self.main_incore_0(x, ret0__out)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)


class TestTensorCiConversion:
    def test_tensor_ci_conversion(self):
        """tensor.ci -> tile.ci conversion preserves dtype + descending kwargs."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[1, 32], pl.INT32]) -> pl.Tensor[[1, 32], pl.INT32]:
                idx: pl.Tensor[[1, 32], pl.INT32] = pl.tensor.ci(0, [1, 32], dtype=pl.INT32, descending=True)
                y: pl.Tensor[[1, 32], pl.INT32] = pl.add(idx, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[1, 32], pl.INT32]) -> pl.Tensor[[1, 32], pl.INT32]:
                y: pl.Tensor[[1, 32], pl.INT32] = self.main_incore_0(x)
                return y

        # tensor.ci -> tile.ci, preserving dtype and the descending=True kwarg.
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[1, 32], pl.INT32],
                ret0__out: pl.Out[pl.Tensor[[1, 32], pl.INT32]],
            ) -> pl.Tensor[[1, 32], pl.INT32]:
                x__tile = pl.load(x, [0, 0], [1, 32], [1, 32], target_memory=pl.Mem.Vec, transpose=False)
                idx__tile = pl.tile.ci(pl.const(0, pl.INT32), [1, 32], dtype=pl.INT32, descending=True)
                y__tile = pl.tile.add(idx__tile, x__tile)
                ret0__store = pl.store(y__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, x: pl.Tensor[[1, 32], pl.INT32]) -> pl.Tensor[[1, 32], pl.INT32]:
                ret0__out = pl.create_tensor([1, 32], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                y = self.main_incore_0(x, ret0__out)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)


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

        # Naive conversion: the Out param has shape [32, 32] with default (tile-shape)
        # strides — no TensorView from parent strides. The parent-stride optimization
        # is handled later by OptimizeOrchTensors (Pattern 2). The orchestration
        # assemble stays a tensor.assemble.
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                mb: pl.Scalar[pl.INDEX],
                nb: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile__tile = pl.load(
                    a, [mb, nb], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                ret0__store = pl.store(tile__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for mb, (c_iter,) in pl.range(0, 128, 32, init_values=(c,)):
                    for nb, (c_iter2,) in pl.range(0, 128, 32, init_values=(c_iter,)):
                        ret0__out = pl.create_tensor([32, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        result = self.main_incore_0(a, mb, nb, ret0__out)
                        c_next = pl.assemble(c_iter2, result, [mb, nb])
                        c_rv = pl.yield_(c_next)
                    c_rv2 = pl.yield_(c_rv)
                return c_rv2

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)


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
        """tensor.mrgsort(s0..s3) -> tile.loads + tile.create(tmp) + tile.mrgsort_format2 + store."""

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
                s0_tile = pl.load(s0, [0, 0], [1, 128])
                s1_tile = pl.load(s1, [0, 0], [1, 128])
                s2_tile = pl.load(s2, [0, 0], [1, 128])
                s3_tile = pl.load(s3, [0, 0], [1, 128])
                mrgsort2_tmp = pl.tile.create([1, 512], dtype=pl.FP32)
                out_tile = pl.tile.mrgsort(s0_tile, s1_tile, s2_tile, s3_tile, mrgsort2_tmp)
                out_store = pl.store(out_tile, [0, 0], out_0)
                return out_store

            @pl.function
            def main(
                self,
                s0: pl.Tensor[[1, 128], pl.FP32],
                s1: pl.Tensor[[1, 128], pl.FP32],
                s2: pl.Tensor[[1, 128], pl.FP32],
                s3: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 512], pl.FP32]:
                out_0 = pl.create_tensor([1, 512], dtype=pl.FP32)
                out = self.main_incore_0(s0, s1, s2, s3, out_0)
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

        # tensor.gather is fully lowered into a per-row loop: each iteration loads a
        # [1, 16] input row and a [1, 3] index row, runs the index-form tile.gather
        # (which needs a [1, 3] INT32 scratch tile), and assembles the row into the
        # accumulator. Phase 3 adds the Out tensor param for the result.
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                inp: pl.Tensor[[4, 16], pl.FP32],
                idx: pl.Tensor[[4, 3], pl.INT32],
                ret0__out: pl.Out[pl.Tensor[[4, 3], pl.FP32]],
            ) -> pl.Tensor[[4, 3], pl.FP32]:
                gather_acc_init = pl.tile.create([4, 3], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv, (gather_ia,) in pl.range(4, init_values=(gather_acc_init,)):
                    gather_inp_row = pl.load(
                        inp, [gather_lv, 0], [1, 16], [1, 16], target_memory=pl.Mem.Vec, transpose=False
                    )
                    gather_idx_row = pl.load(
                        idx, [gather_lv, 0], [1, 3], [1, 3], target_memory=pl.Mem.Vec, transpose=False
                    )
                    gather_row_tmp = pl.tile.create([1, 3], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                    gather_asmbl = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                    gather_rv = pl.yield_(gather_asmbl)
                out__tile: pl.Tile[[4, 3], pl.FP32, pl.Mem.Vec] = gather_rv
                ret0__store = pl.store(out__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self,
                inp: pl.Tensor[[4, 16], pl.FP32],
                idx: pl.Tensor[[4, 3], pl.INT32],
            ) -> pl.Tensor[[4, 3], pl.FP32]:
                ret0__out = pl.create_tensor([4, 3], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                out = self.main_incore_0(inp, idx, ret0__out)
                return out

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_gather_mask_conversion(self):
        """tensor.gather(mask_pattern=...) -> tile.load + tile.gather_mask + tile.store."""
        before, expected = _make_pair(
            in_specs=[("src", [8, 64], DataType.FP32)],
            out_shape=[8, 32],
            out_dtype=DataType.FP32,
            tensor_op=lambda ins: tensor_ops.gather(ins[0], mask_pattern=1),
            tile_op=lambda ts: tile_ops.gather_mask(ts[0], mask_pattern=1),
        )
        _assert_convert_equal(before, expected)


class TestConvertScatterOp:
    """Test conversion of tensor.scatter (rank-2 dim=-1 MVP) and tensor.scatter_mask."""

    def test_scatter_conversion(self):
        """tensor.scatter -> tile.load(input/index/src) + flat-index build + tile.scatter."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                inp: pl.Tensor[[16, 8], pl.FP32],
                idx: pl.Tensor[[4, 8], pl.INT32],
                src: pl.Tensor[[4, 8], pl.FP32],
            ) -> pl.Tensor[[16, 8], pl.FP32]:
                out: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.scatter(inp, dim=-1, index=idx, src=src)
                return out

            @pl.function
            def main(
                self,
                inp: pl.Tensor[[16, 8], pl.FP32],
                idx: pl.Tensor[[4, 8], pl.INT32],
                src: pl.Tensor[[4, 8], pl.FP32],
            ) -> pl.Tensor[[16, 8], pl.FP32]:
                out: pl.Tensor[[16, 8], pl.FP32] = self.main_incore_0(inp, idx, src)
                return out

        # Runs under the autouse roundtrip instrument. The preserve blend emits
        # tile.cmps whose packed-mask result has valid_shape [N, 1] on a wider
        # physical tile; with #1498 fixed (parser now infers the implicit blayout
        # from the physical tile shape, not valid_shape) the print->parse roundtrip
        # holds, so this conversion no longer needs to be skipped.
        After = passes.convert_tensor_to_tile_ops()(Before)
        after_src = After.as_python()

        # tensor.scatter is fully lowered; the index-form tile call is present.
        # Exact-call match avoids the "tile.scatter" substring also matching
        # "tile.scatter_mask" (index form must not lower to the mask op).
        assert "tensor.scatter" not in after_src
        assert "pl.tile.scatter(" in after_src
        assert "pl.tile.scatter_mask(" not in after_src
        # Column index -> flat index: row_base via muls + row-broadcast add.
        assert "tile.muls" in after_src
        assert "tile.row_expand_add" in after_src
        # Preserve blend (pto.tscatter does not keep unwritten dst elements):
        # values + mask scatters into zeroed bases, then out = sel(mask != 0, values, input).
        # The select avoids a multiply-based blend (pto.tmul rejects bf16/i8).
        assert after_src.count("pl.tile.scatter(") == 2
        assert "tile.full" in after_src
        assert "tile.cmps" in after_src and "tile.sel" in after_src
        # Three Vec tile.load calls (one per tensor input).
        assert after_src.count("tile.load") >= 3
        # Phase 3 stores the resulting tile through an Out tensor param.
        assert "pl.Out[pl.Tensor[[16, 8]" in after_src

    def test_scatter_mask_conversion(self):
        """tensor.scatter_mask -> tile.load(input) + tile.load(dst) + tile.scatter_mask + tile.store."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                inp: pl.Tensor[[4, 8], pl.FP32],
                dst: pl.Tensor[[4, 16], pl.FP32],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                out: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.scatter(inp, mask_pattern=1, dst=dst)
                return out

            @pl.function
            def main(
                self,
                inp: pl.Tensor[[4, 8], pl.FP32],
                dst: pl.Tensor[[4, 16], pl.FP32],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                out: pl.Tensor[[4, 16], pl.FP32] = self.main_incore_0(inp, dst)
                return out

        After = passes.convert_tensor_to_tile_ops()(Before)
        after_src = After.as_python()

        assert "tensor.scatter_mask" not in after_src
        assert "pl.tile.scatter_mask(" in after_src
        assert "mask_pattern=1" in after_src
        assert "pl.Out[pl.Tensor[[4, 16]" in after_src


class TestWrapperForwardPropagation:
    """Phase 2a: propagate Phase-1 added Out params through Spmd/Group wrappers.

    When TransformIncoreFunction appends Out tensor params to an InCore
    signature, each Spmd/Group wrapper that forwards into that InCore must
    mirror the appended params on its own signature and forward them through
    the inner call — otherwise orchestration codegen's
    BuildWrapperReorderedParams invariant (every inner-call Var arg maps to a
    wrapper param) breaks and downstream codegen references an undeclared
    identifier.
    """

    def test_spmd_wrapper_forwards_added_output(self):
        """Spmd wrapper gains Out param mirroring the InCore's appended Out."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Spmd, attrs={"core_num": 4})
            def wrapper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.kernel(x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.wrapper(x)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)

        kernel = After.get_function("kernel")
        wrapper = After.get_function("wrapper")
        main = After.get_function("main")
        assert kernel is not None and wrapper is not None and main is not None

        # InCore gained one Out param (Phase 1).
        assert kernel.func_type == ir.FunctionType.InCore
        assert len(kernel.params) == 2
        assert kernel.param_directions[-1] == ir.ParamDirection.Out

        # Wrapper mirrors the signature change (Phase 2a) — still Spmd, same
        # attrs, one extra param matching the InCore's Out param type.
        assert wrapper.func_type == ir.FunctionType.Spmd
        assert wrapper.attrs.get("core_num") == 4
        assert len(wrapper.params) == 2
        assert wrapper.param_directions[0] == ir.ParamDirection.In
        assert wrapper.param_directions[-1] == ir.ParamDirection.Out
        assert ir.structural_equal(wrapper.params[-1].type, kernel.params[-1].type)

        # Wrapper's inner call now forwards the new Out arg (1 in + 1 out).
        inner_call = _find_first_call_to(wrapper, "kernel")
        assert inner_call is not None
        assert len(inner_call.args) == 2

        # Orchestration allocates the Out tensor and passes it to the wrapper
        # (Phase 2b, now covering both transformed InCore and transformed
        # wrappers via the merged callee map).
        assert main.func_type == ir.FunctionType.Orchestration
        assert _find_first_call_to(main, "tensor.create") is not None
        orch_call = _find_first_call_to(main, "wrapper")
        assert orch_call is not None
        assert len(orch_call.args) == 2

    def test_group_wrapper_forwards_added_output(self):
        """Group wrapper gains Out param mirroring the InCore's appended Out."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Group)
            def wrapper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.kernel(x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.wrapper(x)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)

        kernel = After.get_function("kernel")
        wrapper = After.get_function("wrapper")
        main = After.get_function("main")
        assert kernel is not None and wrapper is not None and main is not None

        assert wrapper.func_type == ir.FunctionType.Group
        assert len(wrapper.params) == 2
        assert wrapper.param_directions[-1] == ir.ParamDirection.Out
        assert ir.structural_equal(wrapper.params[-1].type, kernel.params[-1].type)

        inner_call = _find_first_call_to(wrapper, "kernel")
        assert inner_call is not None
        assert len(inner_call.args) == 2

        orch_call = _find_first_call_to(main, "wrapper")
        assert orch_call is not None
        assert len(orch_call.args) == 2

    def test_wrapper_without_transformed_incore_unchanged(self):
        """Wrapper that does NOT forward to a transformed InCore is pass-through.

        The callee InCore is already pure-tile (no tensor ops to lower), so
        Phase 1 appends zero Out params, ForwardedCallFinder finds no
        matching call, and Phase 2a leaves the wrapper's signature and its
        inner call arg list unchanged.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.add(x_tile, x_tile)
                out_: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out)
                return out_

            @pl.function(type=pl.FunctionType.Spmd, attrs={"core_num": 4})
            def wrapper(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out_: pl.Tensor[[64], pl.FP32] = self.kernel(x, out)
                return out_

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out_: pl.Tensor[[64], pl.FP32] = self.wrapper(x, out)
                return out_

        After = passes.convert_tensor_to_tile_ops()(Before)

        before_wrapper = Before.get_function("wrapper")
        after_wrapper = After.get_function("wrapper")
        assert before_wrapper is not None and after_wrapper is not None
        # Wrapper's signature and call-forwarding are untouched by Phase 2a.
        assert len(after_wrapper.params) == len(before_wrapper.params)
        assert after_wrapper.param_directions == before_wrapper.param_directions
        inner_call = _find_first_call_to(after_wrapper, "kernel")
        before_inner_call = _find_first_call_to(before_wrapper, "kernel")
        assert inner_call is not None and before_inner_call is not None
        assert len(inner_call.args) == len(before_inner_call.args)

    def test_wrapper_return_types_preserved_after_propagation(self):
        """Wrapper's ``return_types_`` is preserved (not overwritten with the inner
        callee's). The forward propagator mirrors Out *params* on the wrapper and
        rewrites the inner call to push extra Out args — it must not redeclare the
        wrapper's return signature, since non-transparent wrappers may construct
        tuple returns whose shape differs from the inner callee's.

        Common-case guard: the wrapper here just forwards ``return self.kernel(x)``,
        so its return arity (1) must stay 1 after propagation.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Spmd, attrs={"core_num": 4})
            def wrapper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.kernel(x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.wrapper(x)
                return y

        before_wrapper = Before.get_function("wrapper")
        assert before_wrapper is not None
        before_return_types = list(before_wrapper.return_types)

        After = passes.convert_tensor_to_tile_ops()(Before)
        after_wrapper = After.get_function("wrapper")
        assert after_wrapper is not None
        after_return_types = list(after_wrapper.return_types)

        # Arity is preserved: a single-return wrapper stays a single-return wrapper.
        assert len(after_return_types) == len(before_return_types) == 1
        # The declared return type is structurally what the wrapper declared,
        # not the inner callee's post-transform store-call type.
        assert ir.structural_equal(after_return_types[0], before_return_types[0])


class TestSpmdBlockIdentityConversion:
    """tensor.get_block_idx / get_subblock_idx / get_block_num lower to tile.* form."""

    def test_get_block_idx_conversion(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                idx: pl.Scalar[pl.INDEX] = pl.tensor.get_block_idx()
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, idx)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x__tile = pl.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
                idx = pl.tile.get_block_idx()
                y__tile = pl.tile.adds(x__tile, idx)
                ret0__store = pl.store(y__tile, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                ret0__out = pl.create_tensor([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                y = self.main_incore_0(x, ret0__out)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_get_subblock_idx_conversion(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                idx: pl.Scalar[pl.INDEX] = pl.tensor.get_subblock_idx()
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, idx)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x__tile = pl.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
                idx = pl.tile.get_subblock_idx()
                y__tile = pl.tile.adds(x__tile, idx)
                ret0__store = pl.store(y__tile, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                ret0__out = pl.create_tensor([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                y = self.main_incore_0(x, ret0__out)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_get_block_num_conversion(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                n: pl.Scalar[pl.INDEX] = pl.tensor.get_block_num()
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, n)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x__tile = pl.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
                n = pl.tile.get_block_num()
                y__tile = pl.tile.adds(x__tile, n)
                ret0__store = pl.store(y__tile, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                ret0__out = pl.create_tensor([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                y = self.main_incore_0(x, ret0__out)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_top_level_pl_aliases(self):
        """``pl.get_block_idx`` / ``get_subblock_idx`` / ``get_block_num`` emit tensor-scope
        ops and lower 1:1 to ``tile.*`` via ConvertTensorToTileOps."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                i: pl.Scalar[pl.INDEX] = pl.get_block_idx()
                s: pl.Scalar[pl.INDEX] = pl.get_subblock_idx()
                n: pl.Scalar[pl.INDEX] = pl.get_block_num()
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, i + s + n)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x__tile = pl.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
                i = pl.tile.get_block_idx()
                s = pl.tile.get_subblock_idx()
                n = pl.tile.get_block_num()
                y__tile = pl.tile.adds(x__tile, i + s + n)
                ret0__store = pl.store(y__tile, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                ret0__out = pl.create_tensor([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                y = self.main_incore_0(x, ret0__out)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
