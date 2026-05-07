# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for FlattenTileNdTo2D pass."""

from collections.abc import Callable
from typing import cast

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir import IRBuilder
from pypto.ir.op import tensor as tensor_ops
from pypto.ir.op import tile as tile_ops

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

# (param_name, original_shape) — dtype is shared across the program.
InSpec = tuple[str, list[int]]

# (ib, in_tiles) -> final compute tile. Body may emit intermediate ``ib.let``
# bindings; the helper wraps the final value with ``ib.let("y_tile", ...)``
# unless it is already a Var.
TileBody = Callable[[IRBuilder, list[ir.Expr]], ir.Expr]


def _load2d(
    tensor: ir.Expr,
    offsets: list,
    shapes: list,
    flat_shape: list,
    dtype: DataType,
) -> ir.Call:
    """Build tile.load that keeps tensor-rank offsets/shapes but yields a 2D TileType.

    After flattening, ``FlattenTileNdTo2D`` keeps the original tensor-rank
    offsets/shapes in ``tile.load`` but overrides the result ``TileType`` to be
    2D (with a fresh ``tile_view``/``memory_space``). This helper builds that
    expected IR shape for tests.
    """
    nd_call = tile_ops.load(tensor, offsets, shapes, span=ir.Span.unknown())
    ref_tensor = ir.Var("_ref", ir.TensorType(flat_shape, dtype), ir.Span.unknown())
    ref_call = tile_ops.load(ref_tensor, [0] * len(flat_shape), flat_shape, span=ir.Span.unknown())
    flat_type = cast(ir.TileType, ref_call.type)
    return ir.Call(nd_call.op, list(nd_call.args), nd_call.kwargs, flat_type, nd_call.span)


def _wrap_main(
    ib: IRBuilder,
    prog,
    incore_gvar: ir.GlobalVar,
    in_specs: list[InSpec],
    out_shape: list[int],
    dtype: DataType,
) -> None:
    """Append the standard ``main`` orchestration function used by every test."""
    out_type = ir.TensorType(out_shape, dtype)
    with ib.function("main") as f:
        in_vars = [f.param(name, ir.TensorType(sh, dtype)) for name, sh in in_specs]
        f.return_type(out_type)
        out_v = ib.let("out_0", tensor_ops.create(out_shape, dtype))
        y = ib.let("y", ir.Call(incore_gvar, [*in_vars, out_v], ir.Span.unknown()))
        ib.return_stmt(y)
    prog.add_function(f.get_result())


def _emit_compute(ib: IRBuilder, in_tiles: list[ir.Expr], body: TileBody) -> ir.Expr:
    """Run ``body`` and ensure its result is bound (as ``y_tile`` if not already a Var)."""
    result = body(ib, in_tiles)
    if isinstance(result, ir.Var):
        return result
    return ib.let("y_tile", result)


def _build_before_nd(
    in_specs: list[InSpec],
    out_shape: list[int],
    dtype: DataType,
    body: TileBody,
    *,
    func_name: str = "main_incore_0",
    func_type: ir.FunctionType = ir.FunctionType.InCore,
) -> ir.Program:
    """Build a Before program: ``tile.load(orig) -> body -> tile.store(orig)``.

    Args:
        in_specs: Tensor input parameters (name + original shape).
        out_shape: Original shape of the ``out_0`` tensor parameter.
        dtype: Element dtype shared by tensors and tiles.
        body: Callable returning the final tile expression to store.
        func_name: InCore-variant function name.
        func_type: Function type (``InCore`` / ``AIC`` / ``AIV``).
    """
    span = ir.Span.unknown()
    out_zeros = [0] * len(out_shape)
    out_type = ir.TensorType(out_shape, dtype)

    ib = IRBuilder()
    with ib.program("main") as prog:
        gvar = prog.declare_function(func_name)
        prog.declare_function("main")

        with ib.function(func_name, type=func_type) as f:
            in_vars = [f.param(name, ir.TensorType(sh, dtype)) for name, sh in in_specs]
            out_p = f.param("out_0", out_type, direction=ir.ParamDirection.Out)
            f.return_type(out_type)
            in_tiles: list[ir.Expr] = [
                ib.let(f"{name}_tile", tile_ops.load(v, [0] * len(sh), sh, span=span))
                for (name, sh), v in zip(in_specs, in_vars, strict=True)
            ]
            result = _emit_compute(ib, in_tiles, body)
            out_r = ib.let("out_0", tile_ops.store(result, out_zeros, out_p))
            ib.return_stmt(out_r)
        prog.add_function(f.get_result())

        _wrap_main(ib, prog, gvar, in_specs, out_shape, dtype)
    return prog.get_result()


def _build_expected_2d(
    in_specs: list[InSpec],
    out_shape: list[int],
    flat_in_shapes: list[list[int]],
    dtype: DataType,
    body: TileBody,
    *,
    func_name: str = "main_incore_0",
    func_type: ir.FunctionType = ir.FunctionType.InCore,
) -> ir.Program:
    """Build an Expected program after flattening: ``_load2d(...) -> body -> tile.store(orig, shapes=)``.

    For inputs whose original rank is ``<= 2``, a regular ``tile.load`` is
    emitted instead of ``_load2d``. The ``tile.store`` always carries the
    original ``out_shape`` as ``shapes=`` when ``out_shape`` is >2D.
    """
    span = ir.Span.unknown()
    out_zeros = [0] * len(out_shape)
    out_type = ir.TensorType(out_shape, dtype)

    ib = IRBuilder()
    with ib.program("main") as prog:
        gvar = prog.declare_function(func_name)
        prog.declare_function("main")

        with ib.function(func_name, type=func_type) as f:
            in_vars = [f.param(name, ir.TensorType(sh, dtype)) for name, sh in in_specs]
            out_p = f.param("out_0", out_type, direction=ir.ParamDirection.Out)
            f.return_type(out_type)
            in_tiles: list[ir.Expr] = []
            for (name, sh), v, flat in zip(in_specs, in_vars, flat_in_shapes, strict=True):
                if len(sh) > 2:
                    in_tiles.append(ib.let(f"{name}_tile", _load2d(v, [0] * len(sh), sh, flat, dtype)))
                else:
                    in_tiles.append(ib.let(f"{name}_tile", tile_ops.load(v, [0] * len(sh), sh, span=span)))
            result = _emit_compute(ib, in_tiles, body)
            store_shapes = out_shape if len(out_shape) > 2 else None
            out_r = ib.let("out_0", tile_ops.store(result, out_zeros, out_p, store_shapes))
            ib.return_stmt(out_r)
        prog.add_function(f.get_result())

        _wrap_main(ib, prog, gvar, in_specs, out_shape, dtype)
    return prog.get_result()


def _build_expected_single_op(
    orig_shape: list,
    flat_shape: list,
    dtype: DataType,
    compute_op: Callable[[ir.Expr], ir.Call],
    *,
    func_name: str = "main_incore_0",
    func_type: ir.FunctionType = ir.FunctionType.InCore,
) -> ir.Program:
    """Single-input convenience wrapper around :func:`_build_expected_2d`."""
    return _build_expected_2d(
        [("x", orig_shape)],
        orig_shape,
        [flat_shape],
        dtype,
        lambda _ib, ts: compute_op(ts[0]),
        func_name=func_name,
        func_type=func_type,
    )


# ----------------------------------------------------------------------------
# Element-wise / scalar single-input ops on ND tiles -> 2D
# ----------------------------------------------------------------------------


class TestFlattenTileNdTo2DSingleInput:
    """Single-input element-wise / unary / scalar ops on >2D tiles get flattened."""

    @pytest.mark.parametrize(
        "orig_shape, flat_shape, dtype, op_factory, func_type, func_name",
        [
            # Element-wise binary op (same operand twice)
            (
                [2, 3, 4],
                [6, 4],
                DataType.FP32,
                lambda t: tile_ops.add(t, t),
                ir.FunctionType.InCore,
                "main_incore_0",
            ),
            (
                [2, 3, 4, 5],
                [24, 5],
                DataType.FP32,
                lambda t: tile_ops.mul(t, t),
                ir.FunctionType.InCore,
                "main_incore_0",
            ),
            (
                [2, 2, 2, 2, 4],
                [16, 4],
                DataType.FP32,
                lambda t: tile_ops.add(t, t),
                ir.FunctionType.InCore,
                "main_incore_0",
            ),
            # Unary ops
            ([2, 3, 4], [6, 4], DataType.FP32, tile_ops.exp, ir.FunctionType.InCore, "main_incore_0"),
            ([4, 2, 8], [8, 8], DataType.FP32, tile_ops.neg, ir.FunctionType.InCore, "main_incore_0"),
            # Tile-scalar ops
            (
                [2, 3, 4],
                [6, 4],
                DataType.FP32,
                lambda t: tile_ops.muls(t, 2.0),
                ir.FunctionType.InCore,
                "main_incore_0",
            ),
            (
                [2, 4, 8],
                [8, 8],
                DataType.FP32,
                lambda t: tile_ops.adds(t, 1.0),
                ir.FunctionType.InCore,
                "main_incore_0",
            ),
            # AIC / AIV variants behave the same as InCore
            (
                [2, 3, 4],
                [6, 4],
                DataType.FP32,
                lambda t: tile_ops.add(t, t),
                ir.FunctionType.AIC,
                "aic_func",
            ),
            ([4, 2, 8], [8, 8], DataType.FP32, tile_ops.exp, ir.FunctionType.AIV, "aiv_func"),
            # Different element dtype
            (
                [2, 4, 8],
                [8, 8],
                DataType.FP16,
                lambda t: tile_ops.add(t, t),
                ir.FunctionType.InCore,
                "main_incore_0",
            ),
        ],
        ids=[
            "add_3d_fp32",
            "mul_4d_fp32",
            "add_5d_fp32",
            "exp_3d_fp32",
            "neg_3d_fp32",
            "muls_3d_fp32",
            "adds_3d_fp32",
            "add_3d_aic",
            "exp_3d_aiv",
            "add_3d_fp16",
        ],
    )
    def test_single_input_op(self, orig_shape, flat_shape, dtype, op_factory, func_type, func_name):
        Before = _build_before_nd(
            [("x", orig_shape)],
            orig_shape,
            dtype,
            lambda _ib, ts: op_factory(ts[0]),
            func_name=func_name,
            func_type=func_type,
        )
        Expected = _build_expected_single_op(
            orig_shape, flat_shape, dtype, op_factory, func_name=func_name, func_type=func_type
        )
        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


# ----------------------------------------------------------------------------
# Two-input element-wise ops on ND tiles -> 2D
# ----------------------------------------------------------------------------


class TestFlattenTileNdTo2DTwoInput:
    """Two-input element-wise ops on >2D tiles get flattened."""

    @pytest.mark.parametrize(
        "orig_shape, flat_shape, op_factory",
        [
            ([2, 3, 4], [6, 4], lambda a, b: tile_ops.add(a, b)),
            ([3, 4, 5], [12, 5], lambda a, b: tile_ops.sub(a, b)),
        ],
        ids=["add_3d", "sub_3d"],
    )
    def test_two_input_op(self, orig_shape, flat_shape, op_factory):
        in_specs: list[InSpec] = [("x", orig_shape), ("y", orig_shape)]
        body: TileBody = lambda _ib, ts: op_factory(ts[0], ts[1])  # noqa: E731
        Before = _build_before_nd(in_specs, orig_shape, DataType.FP32, body)
        Expected = _build_expected_2d(in_specs, orig_shape, [flat_shape, flat_shape], DataType.FP32, body)
        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


# ----------------------------------------------------------------------------
# Reduce ops along the last axis on ND tiles -> 2D
# ----------------------------------------------------------------------------


class TestFlattenTileNdTo2DReduceOps:
    """Reduce ops along the last axis are remapped to axis=1 after flatten."""

    @pytest.mark.parametrize(
        "orig_shape, flat_shape, out_shape, reduce_op",
        [
            ([2, 3, 4], [6, 4], [2, 3, 1], tile_ops.sum),
            ([2, 4, 8], [8, 8], [2, 4, 1], tile_ops.max),
        ],
        ids=["sum_3d", "max_3d"],
    )
    def test_reduce_last_axis(self, orig_shape, flat_shape, out_shape, reduce_op):
        before_axis = len(orig_shape) - 1
        Before = _build_before_nd(
            [("x", orig_shape)],
            out_shape,
            DataType.FP32,
            lambda _ib, ts: reduce_op(ts[0], axis=before_axis, keepdim=True),
        )
        Expected = _build_expected_2d(
            [("x", orig_shape)],
            out_shape,
            [flat_shape],
            DataType.FP32,
            lambda _ib, ts: reduce_op(ts[0], axis=1, keepdim=True),
        )
        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


# ----------------------------------------------------------------------------
# Programs that should be left unchanged by the pass
# ----------------------------------------------------------------------------


class TestFlattenTileNdTo2DUnchanged:
    """Programs the pass must not modify."""

    @pytest.mark.parametrize(
        "shape",
        [[32, 64], [64]],
        ids=["2d_tile", "1d_tile"],
    )
    def test_low_rank_tile_unchanged(self, shape):
        """≤2D tiles in InCore functions are left as-is."""
        Before = _build_before_nd(
            [("x", shape)], shape, DataType.FP32, lambda _ib, ts: tile_ops.add(ts[0], ts[0])
        )
        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Before)

    def test_non_incore_function_unchanged(self):
        """Non-InCore (regular) functions with 2D tiles are not modified."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                x_tile: pl.Tile[[32, 64], pl.FP32] = pl.load(x, [0, 0], [32, 64])
                y_tile: pl.Tile[[32, 64], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[32, 64], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[32, 64], pl.FP32]) -> pl.Tensor[[32, 64], pl.FP32]:
                out_0: pl.Tensor[[32, 64], pl.FP32] = pl.create_tensor([32, 64], dtype=pl.FP32)
                y: pl.Tensor[[32, 64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Before)

    def test_group_function_unchanged(self):
        """Group function is not an InCore variant -> unchanged."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Group)
            def group_func(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                return x

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.group_func(x)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Before)


# ----------------------------------------------------------------------------
# Pass-level errors (CHECK macros surface as ValueError)
# ----------------------------------------------------------------------------


class TestFlattenTileNdTo2DErrors:
    """Pass-level errors surface as ``ValueError`` from C++ ``CHECK`` macros."""

    @pytest.mark.parametrize(
        "orig_shape, out_shape, reduce_op, axis",
        [
            ([2, 3, 4], [1, 3, 4], tile_ops.sum, 0),
            ([2, 3, 4], [2, 1, 4], tile_ops.min, 1),
        ],
        ids=["sum_axis_0", "min_axis_1"],
    )
    def test_reduce_non_last_axis_error(self, orig_shape, out_shape, reduce_op, axis):
        """tile reduce ops must reduce the last axis on >2D tiles."""
        Before = _build_before_nd(
            [("x", orig_shape)],
            out_shape,
            DataType.FP32,
            lambda _ib, ts: reduce_op(ts[0], axis=axis, keepdim=True),
        )
        with pytest.raises(ValueError, match="must reduce along the last axis"):
            passes.flatten_tile_nd_to_2d()(Before)

    def test_dynamic_shape_error(self):
        """Dynamic (non-ConstInt) tile shape on >2D tile -> CHECK error."""
        span = ir.Span.unknown()
        n_var = ir.Var("n", ir.ScalarType(DataType.INT32), span)
        dim2 = ir.ConstInt(3, DataType.INT32, span)
        dim3 = ir.ConstInt(4, DataType.INT32, span)
        dyn_tile_type = ir.TileType([n_var, dim2, dim3], DataType.FP32)
        x_tile = ir.Var("x_tile", dyn_tile_type, span)
        add_call = ir.Call(ir.Op("tile.add"), [x_tile, x_tile], dyn_tile_type, span)
        y_tile = ir.Var("y_tile", dyn_tile_type, span)
        body = ir.AssignStmt(y_tile, add_call, span)
        func = ir.Function("incore_func", [x_tile], [dyn_tile_type], body, span, type=ir.FunctionType.InCore)
        program = ir.Program([func], "test_dyn", span)

        with pytest.raises(ValueError, match="must be static"):
            passes.flatten_tile_nd_to_2d()(program)


# ----------------------------------------------------------------------------
# Chained / multi-step bodies that exercise more than one tile op
# ----------------------------------------------------------------------------


class TestFlattenTileNdTo2DChainedOps:
    """Chained sequences of tile ops on >2D tiles get flattened in lock-step."""

    def test_chained_load_exp_add_muls_store(self):
        """``load -> exp -> add -> muls -> store`` chain on a 3D tile."""

        def body(ib: IRBuilder, ts: list[ir.Expr]) -> ir.Expr:
            (x_tile,) = ts
            a = ib.let("a_tile", tile_ops.exp(x_tile))
            b = ib.let("b_tile", tile_ops.add(a, x_tile))
            return ib.let("c_tile", tile_ops.muls(b, 0.5))

        in_specs: list[InSpec] = [("x", [2, 3, 4])]
        Before = _build_before_nd(in_specs, [2, 3, 4], DataType.FP32, body)
        Expected = _build_expected_2d(in_specs, [2, 3, 4], [[6, 4]], DataType.FP32, body)
        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_sum_then_add_3d(self):
        """``load -> sum(keepdim=True, last axis) -> add -> store`` on a 3D tile."""

        def make_body(axis: int) -> TileBody:
            def body(ib: IRBuilder, ts: list[ir.Expr]) -> ir.Expr:
                s = ib.let("s_tile", tile_ops.sum(ts[0], axis=axis, keepdim=True))
                return ib.let("r_tile", tile_ops.add(s, s))

            return body

        in_specs: list[InSpec] = [("x", [2, 3, 4])]
        Before = _build_before_nd(in_specs, [2, 3, 1], DataType.FP32, make_body(2))
        Expected = _build_expected_2d(in_specs, [2, 3, 1], [[6, 4]], DataType.FP32, make_body(1))
        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


# ----------------------------------------------------------------------------
# tile.create / tile.full inside the chain
# ----------------------------------------------------------------------------


class TestFlattenTileNdTo2DConstantOps:
    """``tile.create`` / ``tile.full`` shapes get flattened alongside the tile."""

    @pytest.mark.parametrize(
        "constant_factory",
        [
            lambda shape: tile_ops.create(shape, DataType.FP32),
            lambda shape: tile_ops.full(shape, DataType.FP32, 0.0),
        ],
        ids=["create", "full"],
    )
    def test_constant_op_shape_flattened(self, constant_factory):
        """``tile.<create|full>([2,3,4]) -> tile.add(load, c) -> store`` is flattened to ``[6, 4]``."""

        def make_body(shape: list[int]) -> TileBody:
            def body(ib: IRBuilder, ts: list[ir.Expr]) -> ir.Expr:
                tmp = ib.let("tmp", constant_factory(shape))
                return ib.let("y_tile", tile_ops.add(ts[0], tmp))

            return body

        in_specs: list[InSpec] = [("x", [2, 3, 4])]
        Before = _build_before_nd(in_specs, [2, 3, 4], DataType.FP32, make_body([2, 3, 4]))
        Expected = _build_expected_2d(in_specs, [2, 3, 4], [[6, 4]], DataType.FP32, make_body([6, 4]))
        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_create_full_add_chain(self):
        """``tile.create + tile.full + tile.add`` chain (no input tile.load) on 3D tiles."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                a_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.create([2, 3, 4], dtype=pl.FP32)
                b_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.full([2, 3, 4], dtype=pl.FP32, value=1.0)
                c_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(a_tile, b_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(c_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                a_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.create([6, 4], dtype=pl.FP32)
                b_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.full([6, 4], dtype=pl.FP32, value=1.0)
                c_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(a_tile, b_tile)
                out_store: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(
                    c_tile, [0, 0, 0], out_0, shapes=[2, 3, 4]
                )
                return out_store

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


# ----------------------------------------------------------------------------
# Multi-store / mixed-rank / multi-function programs
# ----------------------------------------------------------------------------


class TestFlattenTileNdTo2DMultiOutput:
    """Programs with multiple stores, mixed ranks, or multiple InCore functions."""

    def test_mixed_2d_and_3d_tiles(self):
        """3D path is flattened, 2D path is left unchanged within the same function."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                y: pl.Tensor[[32, 64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
                out_1: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                a_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.exp(x_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(a_tile, [0, 0, 0], out_0)
                y_tile: pl.Tile[[32, 64], pl.FP32] = pl.load(y, [0, 0], [32, 64])
                b_tile: pl.Tile[[32, 64], pl.FP32] = pl.tile.add(y_tile, y_tile)
                out_1: pl.Tensor[[32, 64], pl.FP32] = pl.store(b_tile, [0, 0], out_1)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                y: pl.Tensor[[32, 64], pl.FP32],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                out_1: pl.Tensor[[32, 64], pl.FP32] = pl.create_tensor([32, 64], dtype=pl.FP32)
                r: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, y, out_0, out_1)
                return r

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                y = f.param("y", ir.TensorType([32, 64], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                out_1 = f.param(
                    "out_1", ir.TensorType([32, 64], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                a_tile = ib.let("a_tile", tile_ops.exp(x_tile))
                out_0_r = ib.let("out_0", tile_ops.store(a_tile, [0, 0, 0], out_0, [2, 3, 4]))
                y_tile = ib.let("y_tile", tile_ops.load(y, [0, 0], [32, 64]))
                b_tile = ib.let("b_tile", tile_ops.add(y_tile, y_tile))
                ib.let("out_1", tile_ops.store(b_tile, [0, 0], out_1))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                y = f.param("y", ir.TensorType([32, 64], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                out_1 = ib.let("out_1", tensor_ops.create([32, 64], DataType.FP32))
                r = ib.let("r", ir.Call(incore_gvar, [x, y, out_0, out_1], ir.Span.unknown()))
                ib.return_stmt(r)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_two_stores_same_shape(self):
        """Two separate load-compute-store chains on the same 3D shape."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
                out_1: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                a_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(a_tile, [0, 0, 0], out_0)
                b_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                out_1: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(b_tile, [0, 0, 0], out_1)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                out_1: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                r: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0, out_1)
                return r

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                out_1 = f.param(
                    "out_1", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                a_tile = ib.let("a_tile", tile_ops.add(x_tile, x_tile))
                out_0_r = ib.let("out_0", tile_ops.store(a_tile, [0, 0, 0], out_0, [2, 3, 4]))
                b_tile = ib.let("b_tile", tile_ops.mul(x_tile, x_tile))
                ib.let("out_1", tile_ops.store(b_tile, [0, 0, 0], out_1, [2, 3, 4]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                out_1 = ib.let("out_1", tensor_ops.create([2, 3, 4], DataType.FP32))
                r = ib.let("r", ir.Call(incore_gvar, [x, out_0, out_1], ir.Span.unknown()))
                ib.return_stmt(r)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multiple_incore_functions(self):
        """Two sibling InCore functions are independently transformed."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def incore_a(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.InCore)
            def incore_b(
                self,
                x: pl.Tensor[[3, 4, 5], pl.FP32],
                out_0: pl.Out[pl.Tensor[[3, 4, 5], pl.FP32]],
            ) -> pl.Tensor[[3, 4, 5], pl.FP32]:
                x_tile: pl.Tile[[3, 4, 5], pl.FP32] = pl.load(x, [0, 0, 0], [3, 4, 5])
                y_tile: pl.Tile[[3, 4, 5], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                out_0: pl.Tensor[[3, 4, 5], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                y: pl.Tensor[[3, 4, 5], pl.FP32],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_a: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                out_b: pl.Tensor[[3, 4, 5], pl.FP32] = pl.create_tensor([3, 4, 5], dtype=pl.FP32)
                ra: pl.Tensor[[2, 3, 4], pl.FP32] = self.incore_a(x, out_a)
                _rb: pl.Tensor[[3, 4, 5], pl.FP32] = self.incore_b(y, out_b)
                return ra

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_a_gvar = prog.declare_function("incore_a")
            incore_b_gvar = prog.declare_function("incore_b")
            prog.declare_function("main")

            for fname, in_shape, flat_shape, op in [
                ("incore_a", [2, 3, 4], [6, 4], tile_ops.add),
                ("incore_b", [3, 4, 5], [12, 5], tile_ops.mul),
            ]:
                with ib.function(fname, type=ir.FunctionType.InCore) as f:
                    x = f.param("x", ir.TensorType(in_shape, DataType.FP32))
                    out_0 = f.param(
                        "out_0", ir.TensorType(in_shape, DataType.FP32), direction=ir.ParamDirection.Out
                    )
                    f.return_type(ir.TensorType(in_shape, DataType.FP32))
                    x_tile = ib.let(
                        "x_tile",
                        _load2d(x, [0] * len(in_shape), in_shape, flat_shape, DataType.FP32),
                    )
                    y_tile = ib.let("y_tile", op(x_tile, x_tile))
                    out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0] * len(in_shape), out_0, in_shape))
                    ib.return_stmt(out_0_r)
                prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                y = f.param("y", ir.TensorType([3, 4, 5], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out_a = ib.let("out_a", tensor_ops.create([2, 3, 4], DataType.FP32))
                out_b = ib.let("out_b", tensor_ops.create([3, 4, 5], DataType.FP32))
                ra = ib.let("ra", ir.Call(incore_a_gvar, [x, out_a], ir.Span.unknown()))
                _rb = ib.let("_rb", ir.Call(incore_b_gvar, [y, out_b], ir.Span.unknown()))
                ib.return_stmt(ra)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


# ----------------------------------------------------------------------------
# Pass property declarations and TileOps2D verifier
# ----------------------------------------------------------------------------


class TestFlattenTileNdTo2DPassProperties:
    """Pass declarations and the ``TileOps2D`` property verifier."""

    def test_pass_properties(self):
        """Verify the pass declares correct required/produced properties."""
        p = passes.flatten_tile_nd_to_2d()
        required = p.get_required_properties()
        assert required.contains(passes.IRProperty.SSAForm)
        assert required.contains(passes.IRProperty.IncoreTileOps)

        produced = p.get_produced_properties()
        assert produced.contains(passes.IRProperty.SSAForm)
        assert produced.contains(passes.IRProperty.TileOps2D)

    def test_pass_name(self):
        """Verify the pass name."""
        p = passes.flatten_tile_nd_to_2d()
        assert p.get_name() == "FlattenTileNdTo2D"

    def test_verifier_passes_after_flatten(self):
        """``TileOps2D`` verifier passes on a correctly flattened program."""
        Before = _build_before_nd(
            [("x", [2, 3, 4])], [2, 3, 4], DataType.FP32, lambda _ib, ts: tile_ops.add(ts[0], ts[0])
        )
        After = passes.flatten_tile_nd_to_2d()(Before)
        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.TileOps2D)
        passes.verify_properties(props, After, "test_verifier")

    def test_verifier_fails_on_unflatten_program(self):
        """``TileOps2D`` verifier fails on a program with >2D tile ops."""
        Unflatten = _build_before_nd(
            [("x", [2, 3, 4])], [2, 3, 4], DataType.FP32, lambda _ib, ts: tile_ops.add(ts[0], ts[0])
        )
        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.TileOps2D)
        with pytest.raises(Exception, match="TileOps2D"):
            passes.verify_properties(props, Unflatten, "test_verifier_fails")


# ----------------------------------------------------------------------------
# Control-flow regression coverage (#648: return_vars matched by identity)
# ----------------------------------------------------------------------------


class TestFlattenTileNdTo2DControlFlow:
    """Tests for ``ForStmt`` / ``IfStmt`` / ``WhileStmt`` with 3D tile carriers."""

    @pytest.mark.parametrize(
        "loop_kind",
        ["for", "while"],
    )
    def test_loop_with_tile_iter_arg(self, loop_kind):
        """``ForStmt`` / ``WhileStmt`` with 3D tile iter_arg -> verifier reports ``TileOps2D``."""

        if loop_kind == "for":

            @pl.program
            class Before:
                @pl.function(type=pl.FunctionType.InCore)
                def main_incore_0(
                    self,
                    x: pl.Tensor[[2, 3, 4], pl.FP32],
                    out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
                ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                    t = pl.load(x, [0, 0, 0], [2, 3, 4])
                    for i in pl.range(4):
                        t = pl.tile.add(t, t)
                    out_0 = pl.store(t, [0, 0, 0], out_0)
                    return out_0

                @pl.function
                def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                    out_0 = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                    y = self.main_incore_0(x, out_0)
                    return y

        else:

            @pl.program
            class Before:
                @pl.function(type=pl.FunctionType.InCore)
                def main_incore_0(
                    self,
                    x: pl.Tensor[[2, 3, 4], pl.FP32],
                    out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
                ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                    t = pl.load(x, [0, 0, 0], [2, 3, 4])
                    cond = True
                    while cond:
                        t = pl.tile.add(t, t)
                        cond = False
                    out_0 = pl.store(t, [0, 0, 0], out_0)
                    return out_0

                @pl.function
                def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                    out_0 = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                    y = self.main_incore_0(x, out_0)
                    return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.flatten_tile_nd_to_2d()(Before)
        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.TileOps2D)
        passes.verify_properties(props, After, f"test_{loop_kind}_stmt_tile_iter_arg")

    def test_for_stmt_tile_iter_arg_structural(self):
        """``ForStmt`` with 3D tile iter_arg -> structural equality with explicit 2D Expected."""

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_p = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                t0 = ib.let("t", tile_ops.load(x, [0, 0, 0], [2, 3, 4]))
                i = ib.var("i", ir.ScalarType(DataType.INT64))
                with ib.for_loop(i, 0, 4, 1) as loop:
                    acc = loop.iter_arg("acc", t0)
                    loop.return_var("acc_out")
                    r = ib.let("r", tile_ops.add(acc, acc))
                    ib.emit(ir.YieldStmt([r], ir.Span.unknown()))
                acc_out = loop.output()
                out_r = ib.let("out_0", tile_ops.store(acc_out, [0, 0, 0], out_p))
                ib.return_stmt(out_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Before = prog.get_result()

        After = passes.flatten_tile_nd_to_2d()(Before)

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_p = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                t0 = ib.let("t", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                i = ib.var("i", ir.ScalarType(DataType.INT64))
                with ib.for_loop(i, 0, 4, 1) as loop:
                    acc = loop.iter_arg("acc", t0)
                    loop.return_var("acc_out")
                    r = ib.let("r", tile_ops.add(acc, acc))
                    ib.emit(ir.YieldStmt([r], ir.Span.unknown()))
                acc_out = loop.output()
                out_r = ib.let("out_0", tile_ops.store(acc_out, [0, 0, 0], out_p, [2, 3, 4]))
                ib.return_stmt(out_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

        ir.assert_structural_equal(After, Expected)

    def test_if_stmt_tile_return_var(self):
        """``IfStmt`` with 3D tile return_vars -> flattened to 2D via yield-type matching."""

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                cond_param = f.param("cond", ir.ScalarType(DataType.BOOL))
                out_p = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))

                t0 = ib.let("t", tile_ops.load(x, [0, 0, 0], [2, 3, 4]))

                with ib.if_stmt(cond_param) as if_blk:
                    if_blk.return_var("rv", ir.TileType([2, 3, 4], DataType.FP32))
                    a = ib.let("a", tile_ops.add(t0, t0))
                    ib.emit(ir.YieldStmt([a], ir.Span.unknown()))
                    if_blk.else_()
                    b = ib.let("b", tile_ops.mul(t0, t0))
                    ib.emit(ir.YieldStmt([b], ir.Span.unknown()))
                rv = if_blk.output()

                out_r = ib.let("out_0", tile_ops.store(rv, [0, 0, 0], out_p))
                ib.return_stmt(out_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                cond = f.param("cond", ir.ScalarType(DataType.BOOL))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, cond, out], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Before = prog.get_result()

        After = passes.flatten_tile_nd_to_2d()(Before)

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                cond_param = f.param("cond", ir.ScalarType(DataType.BOOL))
                out_p = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))

                load_call = _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32)
                t0 = ib.let("t", load_call)

                with ib.if_stmt(cond_param) as if_blk:
                    # return_var type must match yield type (which carries tile_view from op_registry)
                    if_blk.return_var("rv", load_call.type)
                    a = ib.let("a", tile_ops.add(t0, t0))
                    ib.emit(ir.YieldStmt([a], ir.Span.unknown()))
                    if_blk.else_()
                    b = ib.let("b", tile_ops.mul(t0, t0))
                    ib.emit(ir.YieldStmt([b], ir.Span.unknown()))
                rv = if_blk.output()

                out_r = ib.let("out_0", tile_ops.store(rv, [0, 0, 0], out_p, [2, 3, 4]))
                ib.return_stmt(out_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                cond = f.param("cond", ir.ScalarType(DataType.BOOL))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, cond, out], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

        ir.assert_structural_equal(After, Expected)


# ----------------------------------------------------------------------------
# tile.batch_matmul lowering
# ----------------------------------------------------------------------------


class TestFlattenTileNdTo2DBatchMatmul:
    """Tests for ``tile.batch_matmul`` lowering inside ``FlattenTileNdTo2D``."""

    @staticmethod
    def _flattened_incore(before: ir.Program) -> ir.Function:
        """Run ``FlattenTileNdTo2D`` and return ``main_incore_0``."""
        after = passes.flatten_tile_nd_to_2d()(before)
        after_func = after.get_function("main_incore_0")
        assert after_func is not None
        return after_func

    @staticmethod
    def _top_level_calls(func: ir.Function) -> list[ir.Call]:
        """Return top-level ``AssignStmt`` call values from a function body."""
        body = cast(ir.SeqStmts, func.body)
        return [
            stmt.value
            for stmt in body.stmts
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
        ]

    @staticmethod
    def _tuple_const_values(expr: ir.Expr) -> list[int]:
        """Extract integer values from a ``MakeTuple`` of ``ConstInt`` expressions."""
        tup = cast(ir.MakeTuple, expr)
        return [cast(ir.ConstInt, elem).value for elem in tup.elements]

    def test_batch_matmul_broadcasts_and_unrolls(self):
        """Broadcasted ``[2,1,M,K] x [1,3,K,N]`` expands to 6 per-batch 2D ``tile.matmul``."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[2, 1, 16, 128], pl.FP16],
                rhs: pl.Tensor[[1, 3, 128, 64], pl.FP16],
                out_0: pl.Out[pl.Tensor[[2, 3, 16, 64], pl.FP16]],
            ) -> pl.Tensor[[2, 3, 16, 64], pl.FP16]:
                lhs_tile: pl.Tile[[2, 1, 16, 128], pl.FP16] = pl.load(
                    lhs, [0, 0, 0, 0], [2, 1, 16, 128], target_memory=pl.MemorySpace.Mat
                )
                rhs_tile: pl.Tile[[1, 3, 128, 64], pl.FP16] = pl.load(
                    rhs, [0, 0, 0, 0], [1, 3, 128, 64], target_memory=pl.MemorySpace.Mat
                )
                out_tile: pl.Tile[[2, 3, 16, 64], pl.FP32] = pl.tile.batch_matmul(lhs_tile, rhs_tile)
                out_0 = pl.store(out_tile, [0, 0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[2, 1, 16, 128], pl.FP16],
                rhs: pl.Tensor[[1, 3, 128, 64], pl.FP16],
            ) -> pl.Tensor[[2, 3, 16, 64], pl.FP16]:
                out_0 = pl.create_tensor([2, 3, 16, 64], dtype=pl.FP16)
                y = self.main_incore_0(lhs, rhs, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[2, 1, 16, 128], pl.FP16],
                rhs: pl.Tensor[[1, 3, 128, 64], pl.FP16],
                out_0: pl.Out[pl.Tensor[[2, 3, 16, 64], pl.FP16]],
            ) -> pl.Tensor[[2, 3, 16, 64], pl.FP16]:
                lhs_load_0: pl.Tile[[16, 128], pl.FP16] = pl.load(
                    lhs, [0, 0, 0, 0], [1, 1, 16, 128], target_memory=pl.MemorySpace.Mat
                )
                rhs_load_0: pl.Tile[[128, 64], pl.FP16] = pl.load(
                    rhs, [0, 0, 0, 0], [1, 1, 128, 64], target_memory=pl.MemorySpace.Mat
                )
                matmul_0: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(lhs_load_0, rhs_load_0)
                out_0_0: pl.Tensor[[2, 3, 16, 64], pl.FP16] = pl.store(
                    matmul_0, [0, 0, 0, 0], out_0, shapes=[1, 1, 16, 64]
                )

                lhs_load_1: pl.Tile[[16, 128], pl.FP16] = pl.load(
                    lhs, [0, 0, 0, 0], [1, 1, 16, 128], target_memory=pl.MemorySpace.Mat
                )
                rhs_load_1: pl.Tile[[128, 64], pl.FP16] = pl.load(
                    rhs, [0, 1, 0, 0], [1, 1, 128, 64], target_memory=pl.MemorySpace.Mat
                )
                matmul_1: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(lhs_load_1, rhs_load_1)
                out_0_1: pl.Tensor[[2, 3, 16, 64], pl.FP16] = pl.store(
                    matmul_1, [0, 1, 0, 0], out_0_0, shapes=[1, 1, 16, 64]
                )

                lhs_load_2: pl.Tile[[16, 128], pl.FP16] = pl.load(
                    lhs, [0, 0, 0, 0], [1, 1, 16, 128], target_memory=pl.MemorySpace.Mat
                )
                rhs_load_2: pl.Tile[[128, 64], pl.FP16] = pl.load(
                    rhs, [0, 2, 0, 0], [1, 1, 128, 64], target_memory=pl.MemorySpace.Mat
                )
                matmul_2: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(lhs_load_2, rhs_load_2)
                out_0_2: pl.Tensor[[2, 3, 16, 64], pl.FP16] = pl.store(
                    matmul_2, [0, 2, 0, 0], out_0_1, shapes=[1, 1, 16, 64]
                )

                lhs_load_3: pl.Tile[[16, 128], pl.FP16] = pl.load(
                    lhs, [1, 0, 0, 0], [1, 1, 16, 128], target_memory=pl.MemorySpace.Mat
                )
                rhs_load_3: pl.Tile[[128, 64], pl.FP16] = pl.load(
                    rhs, [0, 0, 0, 0], [1, 1, 128, 64], target_memory=pl.MemorySpace.Mat
                )
                matmul_3: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(lhs_load_3, rhs_load_3)
                out_0_3: pl.Tensor[[2, 3, 16, 64], pl.FP16] = pl.store(
                    matmul_3, [1, 0, 0, 0], out_0_2, shapes=[1, 1, 16, 64]
                )

                lhs_load_4: pl.Tile[[16, 128], pl.FP16] = pl.load(
                    lhs, [1, 0, 0, 0], [1, 1, 16, 128], target_memory=pl.MemorySpace.Mat
                )
                rhs_load_4: pl.Tile[[128, 64], pl.FP16] = pl.load(
                    rhs, [0, 1, 0, 0], [1, 1, 128, 64], target_memory=pl.MemorySpace.Mat
                )
                matmul_4: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(lhs_load_4, rhs_load_4)
                out_0_4: pl.Tensor[[2, 3, 16, 64], pl.FP16] = pl.store(
                    matmul_4, [1, 1, 0, 0], out_0_3, shapes=[1, 1, 16, 64]
                )

                lhs_load_5: pl.Tile[[16, 128], pl.FP16] = pl.load(
                    lhs, [1, 0, 0, 0], [1, 1, 16, 128], target_memory=pl.MemorySpace.Mat
                )
                rhs_load_5: pl.Tile[[128, 64], pl.FP16] = pl.load(
                    rhs, [0, 2, 0, 0], [1, 1, 128, 64], target_memory=pl.MemorySpace.Mat
                )
                matmul_5: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(lhs_load_5, rhs_load_5)
                out_0_5: pl.Tensor[[2, 3, 16, 64], pl.FP16] = pl.store(
                    matmul_5, [1, 2, 0, 0], out_0_4, shapes=[1, 1, 16, 64]
                )
                return out_0_5

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[2, 1, 16, 128], pl.FP16],
                rhs: pl.Tensor[[1, 3, 128, 64], pl.FP16],
            ) -> pl.Tensor[[2, 3, 16, 64], pl.FP16]:
                out_0 = pl.create_tensor([2, 3, 16, 64], dtype=pl.FP16)
                y = self.main_incore_0(lhs, rhs, out_0)
                return y

        after_func = self._flattened_incore(Before)
        expected_func = Expected.get_function("main_incore_0")
        assert expected_func is not None
        ir.assert_structural_equal(after_func, expected_func)

    def test_batch_matmul_with_both_operands_load_transpose_unrolls_per_batch(self):
        """Both operands use ``load(transpose=True)``: per-batch transpose load, no extra transpose op."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[2, 128, 16], pl.FP16],
                rhs: pl.Tensor[[2, 64, 128], pl.FP16],
                out_0: pl.Out[pl.Tensor[[2, 16, 64], pl.FP16]],
            ) -> pl.Tensor[[2, 16, 64], pl.FP16]:
                lhs_tile: pl.Tile[[2, 16, 128], pl.FP16] = pl.load(
                    lhs, [0, 0, 0], [2, 128, 16], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                rhs_tile: pl.Tile[[2, 128, 64], pl.FP16] = pl.load(
                    rhs, [0, 0, 0], [2, 64, 128], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                out_tile: pl.Tile[[2, 16, 64], pl.FP32] = pl.tile.batch_matmul(lhs_tile, rhs_tile)
                out_0 = pl.store(out_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[2, 128, 16], pl.FP16],
                rhs: pl.Tensor[[2, 64, 128], pl.FP16],
            ) -> pl.Tensor[[2, 16, 64], pl.FP16]:
                out_0 = pl.create_tensor([2, 16, 64], dtype=pl.FP16)
                y = self.main_incore_0(lhs, rhs, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[2, 128, 16], pl.FP16],
                rhs: pl.Tensor[[2, 64, 128], pl.FP16],
                out_0: pl.Out[pl.Tensor[[2, 16, 64], pl.FP16]],
            ) -> pl.Tensor[[2, 16, 64], pl.FP16]:
                lhs_load_0: pl.Tile[
                    [16, 128],
                    pl.FP16,
                    pl.MemorySpace.Mat,
                    pl.TileView(
                        valid_shape=[16, 128],
                        blayout=pl.TileLayout.row_major,
                        slayout=pl.TileLayout.col_major,
                    ),
                ] = pl.load(lhs, [0, 0, 0], [1, 128, 16], target_memory=pl.MemorySpace.Mat, transpose=True)
                rhs_load_0: pl.Tile[
                    [128, 64],
                    pl.FP16,
                    pl.MemorySpace.Mat,
                    pl.TileView(
                        valid_shape=[128, 64],
                        blayout=pl.TileLayout.row_major,
                        slayout=pl.TileLayout.col_major,
                    ),
                ] = pl.load(rhs, [0, 0, 0], [1, 64, 128], target_memory=pl.MemorySpace.Mat, transpose=True)
                matmul_0: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(lhs_load_0, rhs_load_0)
                out_0_0: pl.Tensor[[2, 16, 64], pl.FP16] = pl.store(
                    matmul_0, [0, 0, 0], out_0, shapes=[1, 16, 64]
                )

                lhs_load_1: pl.Tile[
                    [16, 128],
                    pl.FP16,
                    pl.MemorySpace.Mat,
                    pl.TileView(
                        valid_shape=[16, 128],
                        blayout=pl.TileLayout.row_major,
                        slayout=pl.TileLayout.col_major,
                    ),
                ] = pl.load(lhs, [1, 0, 0], [1, 128, 16], target_memory=pl.MemorySpace.Mat, transpose=True)
                rhs_load_1: pl.Tile[
                    [128, 64],
                    pl.FP16,
                    pl.MemorySpace.Mat,
                    pl.TileView(
                        valid_shape=[128, 64],
                        blayout=pl.TileLayout.row_major,
                        slayout=pl.TileLayout.col_major,
                    ),
                ] = pl.load(rhs, [1, 0, 0], [1, 64, 128], target_memory=pl.MemorySpace.Mat, transpose=True)
                matmul_1: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(lhs_load_1, rhs_load_1)
                out_0_1: pl.Tensor[[2, 16, 64], pl.FP16] = pl.store(
                    matmul_1, [1, 0, 0], out_0_0, shapes=[1, 16, 64]
                )
                return out_0_1

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[2, 128, 16], pl.FP16],
                rhs: pl.Tensor[[2, 64, 128], pl.FP16],
            ) -> pl.Tensor[[2, 16, 64], pl.FP16]:
                out_0 = pl.create_tensor([2, 16, 64], dtype=pl.FP16)
                y = self.main_incore_0(lhs, rhs, out_0)
                return y

        after_func = self._flattened_incore(Before)
        expected_func = Expected.get_function("main_incore_0")
        assert expected_func is not None
        ir.assert_structural_equal(after_func, expected_func)

    @pytest.mark.parametrize(
        "case",
        [
            # 3D no transpose, 2 batches
            {
                "lhs_shape": [2, 16, 128],
                "rhs_shape": [2, 128, 64],
                "out_shape": [2, 16, 64],
                "lhs_transpose": False,
                "rhs_transpose": False,
                "expected_op_seq": ["tile.load", "tile.load", "tile.matmul", "tile.store"] * 2,
                "expected_lhs_offsets": [[0, 0, 0], [1, 0, 0]],
                "expected_rhs_offsets": [[0, 0, 0], [1, 0, 0]],
                "expected_lhs_shapes": [[1, 16, 128], [1, 16, 128]],
                "expected_rhs_shapes": [[1, 128, 64], [1, 128, 64]],
                "expected_store_offsets": [[0, 0, 0], [1, 0, 0]],
                "expected_store_shapes": [[1, 16, 64], [1, 16, 64]],
                "expected_lhs_t_seq": [False, False],
                "expected_rhs_t_seq": [False, False],
            },
            # 3D, single batch
            {
                "lhs_shape": [1, 16, 128],
                "rhs_shape": [1, 128, 64],
                "out_shape": [1, 16, 64],
                "lhs_transpose": False,
                "rhs_transpose": False,
                "expected_op_seq": ["tile.load", "tile.load", "tile.matmul", "tile.store"],
                "expected_lhs_offsets": [[0, 0, 0]],
                "expected_rhs_offsets": [[0, 0, 0]],
                "expected_lhs_shapes": [[1, 16, 128]],
                "expected_rhs_shapes": [[1, 128, 64]],
                "expected_store_offsets": [[0, 0, 0]],
                "expected_store_shapes": [[1, 16, 64]],
                "expected_lhs_t_seq": [False],
                "expected_rhs_t_seq": [False],
            },
            # lhs uses load(transpose=True), rhs does not
            {
                "lhs_shape": [2, 128, 16],
                "rhs_shape": [2, 128, 64],
                "out_shape": [2, 16, 64],
                "lhs_transpose": True,
                "rhs_transpose": False,
                "expected_op_seq": ["tile.load", "tile.load", "tile.matmul", "tile.store"] * 2,
                "expected_lhs_offsets": [[0, 0, 0], [1, 0, 0]],
                "expected_rhs_offsets": [[0, 0, 0], [1, 0, 0]],
                "expected_lhs_shapes": [[1, 128, 16], [1, 128, 16]],
                "expected_rhs_shapes": [[1, 128, 64], [1, 128, 64]],
                "expected_store_offsets": [[0, 0, 0], [1, 0, 0]],
                "expected_store_shapes": [[1, 16, 64], [1, 16, 64]],
                "expected_lhs_t_seq": [True, True],
                "expected_rhs_t_seq": [False, False],
            },
        ],
        ids=["3d_no_transpose", "single_batch", "lhs_load_transpose"],
    )
    def test_batch_matmul_unrolls_kwargs(self, case):
        """Per-batch ``tile.load``/``tile.store`` kwargs match the broadcast/transpose plan."""
        lhs_shape = case["lhs_shape"]
        rhs_shape = case["rhs_shape"]
        out_shape = case["out_shape"]
        lhs_transpose = case["lhs_transpose"]
        rhs_transpose = case["rhs_transpose"]

        # The DSL hard-codes shapes/types so we synthesize Before via IRBuilder
        # to keep this test parametrizable across batch / transpose variants.
        span = ir.Span.unknown()
        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                lhs = f.param("lhs", ir.TensorType(lhs_shape, DataType.FP16))
                rhs = f.param("rhs", ir.TensorType(rhs_shape, DataType.FP16))
                out_p = f.param(
                    "out_0", ir.TensorType(out_shape, DataType.FP16), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType(out_shape, DataType.FP16))

                # Inferred logical lhs tile shape: same as rhs[K]/[N]; if transposed
                # in load, last two dims swap.
                def load_tile_shape(shape: list[int], transpose: bool) -> list[int]:
                    if transpose:
                        return [*shape[:-2], shape[-1], shape[-2]]
                    return shape

                lhs_tile_shape = load_tile_shape(lhs_shape, lhs_transpose)
                rhs_tile_shape = load_tile_shape(rhs_shape, rhs_transpose)

                lhs_load = tile_ops.load(
                    lhs,
                    [0] * len(lhs_shape),
                    lhs_shape,
                    target_memory=ir.MemorySpace.Mat,
                    transpose=lhs_transpose,
                    span=span,
                )
                lhs_call = ir.Call(
                    lhs_load.op,
                    list(lhs_load.args),
                    lhs_load.kwargs,
                    ir.TileType(lhs_tile_shape, DataType.FP16),
                    lhs_load.span,
                )
                lhs_tile = ib.let("lhs_tile", lhs_call)

                rhs_load = tile_ops.load(
                    rhs,
                    [0] * len(rhs_shape),
                    rhs_shape,
                    target_memory=ir.MemorySpace.Mat,
                    transpose=rhs_transpose,
                    span=span,
                )
                rhs_call = ir.Call(
                    rhs_load.op,
                    list(rhs_load.args),
                    rhs_load.kwargs,
                    ir.TileType(rhs_tile_shape, DataType.FP16),
                    rhs_load.span,
                )
                rhs_tile = ib.let("rhs_tile", rhs_call)

                bmm_op = ir.Op("tile.batch_matmul")
                out_tile = ib.let(
                    "out_tile",
                    ir.Call(bmm_op, [lhs_tile, rhs_tile], ir.TileType(out_shape, DataType.FP32), span),
                )
                out_r = ib.let("out_0", tile_ops.store(out_tile, [0] * len(out_shape), out_p))
                ib.return_stmt(out_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                lhs = f.param("lhs", ir.TensorType(lhs_shape, DataType.FP16))
                rhs = f.param("rhs", ir.TensorType(rhs_shape, DataType.FP16))
                f.return_type(ir.TensorType(out_shape, DataType.FP16))
                out_v = ib.let("out_0", tensor_ops.create(out_shape, DataType.FP16))
                y = ib.let("y", ir.Call(incore_gvar, [lhs, rhs, out_v], span))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Before = prog.get_result()

        func = self._flattened_incore(Before)
        calls = self._top_level_calls(func)
        assert [call.op.name for call in calls] == case["expected_op_seq"]

        load_calls = [call for call in calls if call.op.name == "tile.load"]
        # Loads alternate lhs, rhs, lhs, rhs, ...
        actual_lhs_offsets = [self._tuple_const_values(call.args[1]) for call in load_calls[0::2]]
        actual_rhs_offsets = [self._tuple_const_values(call.args[1]) for call in load_calls[1::2]]
        actual_lhs_shapes = [self._tuple_const_values(call.args[2]) for call in load_calls[0::2]]
        actual_rhs_shapes = [self._tuple_const_values(call.args[2]) for call in load_calls[1::2]]
        actual_lhs_t = [call.kwargs.get("transpose", False) for call in load_calls[0::2]]
        actual_rhs_t = [call.kwargs.get("transpose", False) for call in load_calls[1::2]]
        assert actual_lhs_offsets == case["expected_lhs_offsets"]
        assert actual_rhs_offsets == case["expected_rhs_offsets"]
        assert actual_lhs_shapes == case["expected_lhs_shapes"]
        assert actual_rhs_shapes == case["expected_rhs_shapes"]
        assert actual_lhs_t == case["expected_lhs_t_seq"]
        assert actual_rhs_t == case["expected_rhs_t_seq"]

        store_calls = [call for call in calls if call.op.name == "tile.store"]
        assert [self._tuple_const_values(call.args[1]) for call in store_calls] == case[
            "expected_store_offsets"
        ]
        assert [self._tuple_const_values(call.args[3]) for call in store_calls] == case[
            "expected_store_shapes"
        ]


# ----------------------------------------------------------------------------
# tile.batch_matmul_acc lowering
# ----------------------------------------------------------------------------


class TestFlattenTileNdTo2DBatchMatmulAcc:
    """Tests for ``tile.batch_matmul_acc`` lowering inside ``FlattenTileNdTo2D``.

    The single-batch fast path is covered end-to-end in
    ``TestNdTensorMatmulConversion`` (convert + flatten); the test below
    targets the general ``batch_count > 1`` path, which is structurally
    different (per-batch ``tile.slice`` + ``tile.matmul_acc`` +
    ``tile.assemble``, plus the Vec→Acc round-trip on the loop-carried
    accumulator).
    """

    def test_batch_two_acc_unrolls_with_slice_assemble_and_memory_round_trip(self):
        """batch=2 ``tile.batch_matmul_acc`` unrolls into 2 tile.matmul_acc + slice/assemble.

        The accumulator is produced by an upstream batch=2 ``tensor.matmul``
        (which itself takes the general ``LowerBatchMatmul`` path), so the
        post-flatten acc lives in Vec memory. ``LowerBatchMatmulAcc`` must
        therefore (a) move the acc to Acc before each per-batch
        ``tile.matmul_acc``, (b) slice/assemble in Acc, and (c) move the final
        acc back to Vec to preserve the iter-arg / consumer memory contract.
        """
        ib = IRBuilder()
        with ib.program("main") as prog:
            prog.declare_function("main_incore_0")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                h0 = f.param("h0", ir.TensorType([2, 16, 256], DataType.BF16))
                w0 = f.param("w0", ir.TensorType([2, 64, 256], DataType.BF16))
                h1 = f.param("h1", ir.TensorType([2, 16, 256], DataType.BF16))
                w1 = f.param("w1", ir.TensorType([2, 64, 256], DataType.BF16))
                out_p = f.param(
                    "out_0",
                    ir.TensorType([2, 16, 64], DataType.FP32),
                    direction=ir.ParamDirection.Out,
                )
                f.return_type(ir.TensorType([2, 16, 64], DataType.FP32))

                acc_init = ib.let(
                    "acc_init",
                    tensor_ops.matmul(h0, w0, b_trans=True, out_dtype=DataType.FP32),
                )
                acc_final = ib.let(
                    "acc_final",
                    tensor_ops.matmul_acc(acc_init, h1, w1, b_trans=True),
                )
                out_r = ib.let("out_0", tensor_ops.assemble(out_p, acc_final, [0, 0, 0]))
                ib.return_stmt(out_r)
            prog.add_function(f.get_result())
        before = prog.get_result()

        after = passes.flatten_tile_nd_to_2d()(passes.convert_tensor_to_tile_ops()(before))
        fn = after.get_function("main_incore_0")
        assert fn is not None
        body = cast(ir.SeqStmts, fn.body)
        calls = [
            stmt.value
            for stmt in body.stmts
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
        ]
        names = [c.op.name for c in calls]

        # Both batch ops are fully unrolled.
        assert "tile.batch_matmul" not in names
        assert "tile.batch_matmul_acc" not in names

        # Two batches × {matmul, matmul_acc} = 2 + 2.
        assert names.count("tile.matmul") == 2
        assert names.count("tile.matmul_acc") == 2

        # The general path uses tile.slice + tile.assemble around the per-batch
        # matmul_acc to read/write each [M, N] band of the [batch*M, N] acc.
        # It also emits a tile.move(target_memory=Acc) before the matmul_acc
        # block (Vec→Acc) and a tile.move(target_memory=Vec) after it
        # (Acc→Vec) to keep the loop-carried acc in its original Vec space.
        assert "tile.slice" in names
        assert "tile.assemble" in names
        moves = [c for c in calls if c.op.name == "tile.move"]
        move_targets = [c.kwargs.get("target_memory") for c in moves]
        assert pl.MemorySpace.Acc in move_targets, (
            f"expected a tile.move to Acc on the acc operand, got targets={move_targets}"
        )
        assert pl.MemorySpace.Vec in move_targets, (
            "expected a tile.move back to Vec to preserve original acc memory space, "
            f"got targets={move_targets}"
        )


# ----------------------------------------------------------------------------
# tensor.matmul / tensor.matmul_acc → tile.batch_matmul[_acc] dispatch
# ----------------------------------------------------------------------------


class TestNdTensorMatmulConversion:
    """End-to-end test: tensor.matmul[_acc] with ND inputs lowers via batch ops."""

    def test_nd_tensor_matmul_dispatch(self):
        """tensor.matmul with 2D × 3D operand emits tile.batch_matmul (then unrolls)."""
        ib = IRBuilder()
        with ib.program("main") as prog:
            prog.declare_function("main_incore_0")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                h = f.param("h", ir.TensorType([16, 256], DataType.BF16))
                w = f.param("w", ir.TensorType([1, 64, 256], DataType.BF16))
                out_p = f.param(
                    "out_0", ir.TensorType([16, 64], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([16, 64], DataType.FP32))

                y_acc = ib.let(
                    "y_acc",
                    tensor_ops.matmul(h, w, b_trans=True, out_dtype=DataType.FP32),
                )
                # Squeeze batch=1 result via assemble into 2D out_0.
                # Use tensor.assemble with [0, 0] offset; flatten lowers to per-batch store.
                out_r = ib.let("out_0", tensor_ops.assemble(out_p, y_acc, [0, 0, 0]))
                ib.return_stmt(out_r)
            prog.add_function(f.get_result())
        Before = prog.get_result()

        # Run conversion + flatten passes.
        after = passes.convert_tensor_to_tile_ops()(Before)
        names = []
        fn = after.get_function("main_incore_0")
        assert fn is not None
        body = cast(ir.SeqStmts, fn.body)
        for stmt in body.stmts:
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
                names.append(stmt.value.op.name)
        # ND tensor.matmul should have become tile.batch_matmul (not tile.matmul).
        assert "tile.batch_matmul" in names
        assert "tile.matmul" not in names

    def test_nd_tensor_matmul_acc_dispatch_and_flatten(self):
        """tensor.matmul_acc with 2D × 3D operand emits tile.batch_matmul_acc, then flattens.

        The acc is produced by an earlier ND tensor.matmul (which the conversion
        pass remaps to a tile.batch_matmul result) so the acc operand is already
        a TileType when matmul_acc is converted.

        End-to-end: convert + flatten leaves no batch ops and emits exactly one
        tile.matmul + one tile.matmul_acc (batch=1 fast path).
        """
        ib = IRBuilder()
        with ib.program("main") as prog:
            prog.declare_function("main_incore_0")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                h0 = f.param("h0", ir.TensorType([16, 256], DataType.BF16))
                w0 = f.param("w0", ir.TensorType([1, 64, 256], DataType.BF16))
                h1 = f.param("h1", ir.TensorType([16, 256], DataType.BF16))
                w1 = f.param("w1", ir.TensorType([1, 64, 256], DataType.BF16))
                out_p = f.param(
                    "out_0",
                    ir.TensorType([1, 16, 64], DataType.FP32),
                    direction=ir.ParamDirection.Out,
                )
                f.return_type(ir.TensorType([1, 16, 64], DataType.FP32))

                y_acc = ib.let(
                    "y_acc",
                    tensor_ops.matmul(h0, w0, b_trans=True, out_dtype=DataType.FP32),
                )
                y_acc_2 = ib.let(
                    "y_acc_2",
                    tensor_ops.matmul_acc(y_acc, h1, w1, b_trans=True),
                )
                out_r = ib.let("out_0", tensor_ops.assemble(out_p, y_acc_2, [0, 0, 0]))
                ib.return_stmt(out_r)
            prog.add_function(f.get_result())
        Before = prog.get_result()

        after_convert = passes.convert_tensor_to_tile_ops()(Before)

        def collect_names(prog: ir.Program) -> list[str]:
            fn = prog.get_function("main_incore_0")
            assert fn is not None
            body = cast(ir.SeqStmts, fn.body)
            return [
                stmt.value.op.name
                for stmt in body.stmts
                if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
            ]

        # After conversion: ND ops dispatch to the batch variants.
        names_convert = collect_names(after_convert)
        assert "tile.batch_matmul" in names_convert
        assert "tile.batch_matmul_acc" in names_convert
        assert "tile.matmul" not in names_convert
        assert "tile.matmul_acc" not in names_convert

        # After flatten: batch ops disappear; one per-batch tile.matmul (from
        # batch_matmul) and one per-batch tile.matmul_acc (from batch_matmul_acc)
        # remain (batch=1 fast path).
        after_flatten = passes.flatten_tile_nd_to_2d()(after_convert)
        names_flatten = collect_names(after_flatten)
        assert "tile.batch_matmul" not in names_flatten
        assert "tile.batch_matmul_acc" not in names_flatten
        assert names_flatten.count("tile.matmul") == 1
        assert names_flatten.count("tile.matmul_acc") == 1


# ----------------------------------------------------------------------------
# Regression coverage for #1278 — TileType memory_space presence mismatch on
# print/parse roundtrip after auto-flatten of a Mat tile.load.
#
# Why CI didn't catch the original issue:
#   The bug requires a rank>2 ``tile.load`` with ``target_memory=Mat`` whose
#   result is NOT exclusively consumed by ``tile.batch_matmul[_acc]``. When the
#   var is in ``batch_matmul_only_vars`` (every existing test's pattern), the
#   ``FlattenTileNdTo2D`` pass skips Form A construction and lets Strategy 1
#   reconstruct per-batch loads instead. Layered on top of that,
#   ``OpRegistry::Create`` already backfills ``memory_space`` from the
#   ``target_memory`` kwarg via ``set_output_memory_from_kwarg``
#   (issue #553's fix), so even when Form A fires it reads a coherent
#   ``result_tile->memory_space_``. The dormant scenario surfaces only when a
#   future pass / IRBuilder bypasses ``OpRegistry::Create`` for tile.load
#   construction.
#
# The tests below close the structural coverage gap. Both go through the
# public ``OpRegistry::Create`` path and so cannot probe the deducer in
# isolation — they assert the end-to-end invariant (target_memory in,
# coherent canonical TileType out) and exercise the previously-uncovered
# Form A construction in ``FlattenTileNdTo2D`` under the autouse
# ``RoundtripInstrument``.
# ----------------------------------------------------------------------------


class TestFlattenTileNdTo2DMatLoadRoundtrip:
    """Layered regression coverage for #1278."""

    @pytest.mark.parametrize(
        "target_memory",
        [pl.Mem.Mat, pl.Mem.Vec],
    )
    def test_tile_load_emits_coherent_memory_space(self, target_memory):
        """``tile.load`` result type's ``memory_space`` must match ``target_memory``.

        End-to-end op-creation invariant. The call goes through
        ``tile_ops.load`` -> ``ir.create_op_call`` -> ``OpRegistry::Create``,
        so it exercises the full public construction path. Two layers protect
        this invariant: ``DeduceTileLoadType`` (passes ``target_memory_opt``
        into the ``TileType`` constructor) and ``OpRegistry::Create``'s
        ``set_output_memory_from_kwarg`` backfill. Either alone is sufficient
        for the assertion to hold, so this test fires only if BOTH layers
        regress simultaneously — the deducer self-consistency invariant
        cannot be probed in isolation through this Python entry point.
        """
        x_var = ir.Var("x", ir.TensorType([16, 128], DataType.FP16), ir.Span.unknown())
        call = tile_ops.load(x_var, [0, 0], [16, 128], target_memory=target_memory)
        result = cast(ir.TileType, call.type)
        assert result.memory_space == target_memory
        # Canonical encoding: the implicit Mat-style / Vec-style tile_view
        # collapses to None. Any future change that lets the explicit Mat
        # tile_view linger here would re-introduce the asymmetry between
        # what the printer emits (annotation only) and what the re-parser
        # rebuilds (explicit tile_view).
        assert result.tile_view is None

    def test_rank3_mat_load_consumed_by_move_roundtrips(self):
        """Rank-3 Mat ``tile.load`` -> ``tile.move`` exercises the Form A path.

        The autouse ``pass_verification_context`` fixture (see
        ``tests/ut/conftest.py``) wraps every pass execution with
        ``RoundtripInstrument``, which prints the post-pass IR, re-parses it,
        and asserts structural equality. ``tile.move`` (rather than
        ``tile.batch_matmul``) keeps ``x_mat`` out of
        ``batch_matmul_only_vars`` so the rank>2 Form A construction at
        ``flatten_tile_nd_to_2d_pass.cpp:1523-1526`` is the active branch — the
        scenario the issue reports.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[1, 16, 128], pl.FP16],
                out_0: pl.Out[pl.Tensor[[1, 16, 128], pl.FP16]],
            ) -> pl.Tensor[[1, 16, 128], pl.FP16]:
                x_mat = pl.tile.load(x, [0, 0, 0], [1, 16, 128], target_memory=pl.Mem.Mat)
                x_left = pl.tile.move(x_mat, target_memory=pl.Mem.Left)
                out_0 = pl.tile.store(x_left, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[1, 16, 128], pl.FP16]) -> pl.Tensor[[1, 16, 128], pl.FP16]:
                out_0 = pl.create_tensor([1, 16, 128], dtype=pl.FP16)
                y = self.main_incore_0(x, out_0)
                return y

        # The autouse fixture supplies RoundtripInstrument; this call would
        # raise ``[RoundtripInstrument] Structural equality failed after pass
        # 'FlattenTileNdTo2D'`` if the post-pass IR did not round-trip.
        After = passes.flatten_tile_nd_to_2d()(Before)

        after_func = After.get_function("main_incore_0")
        assert after_func is not None
        body = cast(ir.SeqStmts, after_func.body)
        flat_load = next(
            stmt
            for stmt in body.stmts
            if isinstance(stmt, ir.AssignStmt)
            and isinstance(stmt.value, ir.Call)
            and stmt.value.op.name == "tile.load"
        )
        flat_var_type = cast(ir.TileType, flat_load.var.type)
        flat_call_type = cast(ir.TileType, flat_load.value.type)

        # Form A's flat_tile_type — both Var and Call must share the canonical
        # 2D encoding for Mat (issue #1278 specifically reported this
        # asymmetry on print/parse roundtrip).
        assert flat_var_type.shape == [16, 128]
        assert flat_var_type.memory_space == ir.MemorySpace.Mat
        assert flat_var_type.tile_view is None
        assert flat_call_type.memory_space == flat_var_type.memory_space
        assert flat_call_type.tile_view == flat_var_type.tile_view


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
