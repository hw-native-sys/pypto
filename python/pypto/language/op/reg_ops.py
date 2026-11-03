# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""P0 register operations, valid within an A5 InCore/AIV reg_scope."""

from typing import Any

from pypto.ir.utils import _get_span_or_capture
from pypto.language.typing import Reg, Scalar, Tile
from pypto.pypto_core import DataType, ir


def _expr(value: object, dtype: DataType | None = None) -> ir.Expr:
    if isinstance(value, (Reg, Scalar, Tile)):
        return value.unwrap()
    if dtype is not None and isinstance(value, (ir.ConstInt, ir.ConstFloat)):
        value = value.value
    if isinstance(value, ir.Expr):
        return value
    if isinstance(value, bool):
        raise TypeError("Register integer arguments cannot be bool")
    if isinstance(value, (int, float)):
        actual_dtype = dtype or (DataType.INDEX if isinstance(value, int) else DataType.FP32)
        if actual_dtype.is_float():
            return ir.ConstFloat(float(value), actual_dtype, ir.Span.unknown())
        if not isinstance(value, int):
            raise TypeError("Integer register generators require integer constants")
        return ir.ConstInt(value, actual_dtype, ir.Span.unknown())
    raise TypeError(f"Expected register/scalar/tile expression, got {type(value).__name__}")


def _call(name: str, args: list[ir.Expr], **kwargs: Any) -> ir.Call:
    return ir.create_op_call(f"reg.{name}", args, kwargs, _get_span_or_capture())


def _lanes(value: object) -> int:
    if isinstance(value, ir.ConstInt):
        value = value.value
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"lanes must be a positive static integer, got {value}")
    return value


def load(tile: Tile | ir.Expr, offset: int | Scalar | ir.Expr = 0, *, lanes: int) -> Reg:
    """Read lanes consecutive initialized elements from a Vec Tile."""
    return Reg(_call("load", [_expr(tile), _expr(offset)], lanes=_lanes(lanes)))


def store(
    tile: Tile | ir.Expr,
    offset: int | Scalar | ir.Expr = 0,
    value: Reg | ir.Expr | None = None,
    mask: Reg | ir.Expr | None = None,
) -> Tile:
    """Write active lanes and return a Tile alias of the destination."""
    if value is None:
        raise TypeError("reg.store requires value")
    args = [_expr(tile), _expr(offset), _expr(value)]
    if mask is not None:
        args.append(_expr(mask))
    return Tile(expr=_call("store", args))


def broadcast(scalar: Scalar | ir.Expr | int | float, *, lanes: int, dtype: DataType) -> Reg:
    """Broadcast a scalar without implicitly casting an existing Scalar SSA."""
    return Reg(_call("broadcast", [_expr(scalar, dtype)], lanes=_lanes(lanes), dtype=dtype))


def index_gen(
    base: Scalar | ir.Expr | int | float, *, lanes: int, dtype: DataType, order: str = "ascending"
) -> Reg:
    """Generate ascending lane indices beginning at base."""
    return Reg(_call("index_gen", [_expr(base, dtype)], lanes=_lanes(lanes), dtype=dtype, order=order))


def full_mask(*, lanes: int) -> Reg:
    """Create an all-active boolean register."""
    return Reg(_call("full_mask", [], lanes=_lanes(lanes)))


def active_mask(active_lanes: Scalar | ir.Expr | int, *, lanes: int) -> Reg:
    """Activate a statically proven prefix of the logical lanes."""
    return Reg(_call("active_mask", [_expr(active_lanes)], lanes=_lanes(lanes)))


def _binary(name: str, lhs: Reg | ir.Expr, rhs: Reg | ir.Expr, mask: Reg | ir.Expr | None, pmode: str) -> Reg:
    args = [_expr(lhs), _expr(rhs)]
    if mask is not None:
        args.append(_expr(mask))
    return Reg(_call(name, args, pmode=pmode))


def add(
    lhs: Reg | ir.Expr, rhs: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero"
) -> Reg:
    """Apply add to identical Vec registers; masked inactive lanes become zero."""
    return _binary("add", lhs, rhs, mask, pmode)


def sub(
    lhs: Reg | ir.Expr, rhs: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero"
) -> Reg:
    """Apply sub to identical Vec registers; masked inactive lanes become zero."""
    return _binary("sub", lhs, rhs, mask, pmode)


def mul(
    lhs: Reg | ir.Expr, rhs: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero"
) -> Reg:
    """Apply mul to identical Vec registers; masked inactive lanes become zero."""
    return _binary("mul", lhs, rhs, mask, pmode)


def div(
    lhs: Reg | ir.Expr, rhs: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero"
) -> Reg:
    """Apply div to identical Vec registers; masked inactive lanes become zero."""
    return _binary("div", lhs, rhs, mask, pmode)


def minimum(
    lhs: Reg | ir.Expr, rhs: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero"
) -> Reg:
    """Apply minimum to identical Vec registers; masked inactive lanes become zero."""
    return _binary("minimum", lhs, rhs, mask, pmode)


def maximum(
    lhs: Reg | ir.Expr, rhs: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero"
) -> Reg:
    """Apply maximum to identical Vec registers; masked inactive lanes become zero."""
    return _binary("maximum", lhs, rhs, mask, pmode)


def convert(
    value: Reg | ir.Expr, *, dtype: DataType, rounding: str | None = None, saturate: bool | None = None
) -> Reg:
    """Convert an explicitly supported dtype pair, preserving logical lanes."""
    expr = _expr(value)
    if not isinstance(expr.type, ir.RegType) or expr.type.kind != ir.RegKind.VecReg:
        raise TypeError("reg.convert requires VecReg")
    src = expr.type.dtype
    if src == dtype and rounding is None and saturate is None:
        return Reg(expr)
    kwargs: dict[str, Any] = {"dtype": dtype}
    if rounding is None and src == DataType.FP32:
        if dtype == DataType.FP16:
            rounding = "R"
        elif dtype == DataType.INT32:
            rounding = "Z"
    if rounding is not None:
        kwargs["rounding"] = rounding
    if saturate is not None:
        kwargs["saturate"] = saturate
    return Reg(_call("convert", [expr], **kwargs))


def compare(lhs: Reg | ir.Expr, rhs: Reg | ir.Expr, seed: Reg | ir.Expr, *, predicate: str) -> Reg:
    """Compare same-type Vec registers under a same-lane seed mask."""
    return Reg(_call("compare", [_expr(lhs), _expr(rhs), _expr(seed)], predicate=predicate))


def select(mask: Reg | ir.Expr, true_value: Reg | ir.Expr, false_value: Reg | ir.Expr) -> Reg:
    """Select each lane from true_value or false_value."""
    return Reg(_call("select", [_expr(mask), _expr(true_value), _expr(false_value)]))


def mask_and(lhs: Reg | ir.Expr, rhs: Reg | ir.Expr) -> Reg:
    """Compute boolean mask and."""
    return Reg(_call("mask_and", [_expr(lhs), _expr(rhs)]))


def mask_or(lhs: Reg | ir.Expr, rhs: Reg | ir.Expr) -> Reg:
    """Compute boolean mask or."""
    return Reg(_call("mask_or", [_expr(lhs), _expr(rhs)]))


def mask_xor(lhs: Reg | ir.Expr, rhs: Reg | ir.Expr) -> Reg:
    """Compute boolean mask xor."""
    return Reg(_call("mask_xor", [_expr(lhs), _expr(rhs)]))


def mask_not(value: Reg | ir.Expr) -> Reg:
    """Invert a boolean mask."""
    return Reg(_call("mask_not", [_expr(value)]))


def _unary(name: str, value: Reg | ir.Expr, mask: Reg | ir.Expr | None, pmode: str) -> Reg:
    args = [_expr(value)]
    if mask is not None:
        args.append(_expr(mask))
    return Reg(_call(name, args, pmode=pmode))


def abs(value: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero") -> Reg:
    """Apply abs lane-wise; inactive masked lanes become zero."""
    return _unary("abs", value, mask, pmode)


def neg(value: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero") -> Reg:
    """Apply neg lane-wise; inactive masked lanes become zero."""
    return _unary("neg", value, mask, pmode)


def relu(value: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero") -> Reg:
    """Apply relu lane-wise; inactive masked lanes become zero."""
    return _unary("relu", value, mask, pmode)


def exp(value: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero") -> Reg:
    """Apply exp lane-wise; inactive masked lanes become zero."""
    return _unary("exp", value, mask, pmode)


def log(value: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero") -> Reg:
    """Apply log lane-wise; inactive masked lanes become zero."""
    return _unary("log", value, mask, pmode)


def sqrt(value: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero") -> Reg:
    """Apply sqrt lane-wise; inactive masked lanes become zero."""
    return _unary("sqrt", value, mask, pmode)


def bitwise_not(value: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero") -> Reg:
    """Apply bitwise_not lane-wise; inactive masked lanes become zero."""
    return _unary("bitwise_not", value, mask, pmode)


def bitwise_and(
    lhs: Reg | ir.Expr, rhs: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero"
) -> Reg:
    """Apply bitwise_and to same-type integer registers with optional predication."""
    return _binary("bitwise_and", lhs, rhs, mask, pmode)


def bitwise_or(
    lhs: Reg | ir.Expr, rhs: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero"
) -> Reg:
    """Apply bitwise_or to same-type integer registers with optional predication."""
    return _binary("bitwise_or", lhs, rhs, mask, pmode)


def bitwise_xor(
    lhs: Reg | ir.Expr, rhs: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero"
) -> Reg:
    """Apply bitwise_xor to same-type integer registers with optional predication."""
    return _binary("bitwise_xor", lhs, rhs, mask, pmode)


def shift_left(
    lhs: Reg | ir.Expr, rhs: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero"
) -> Reg:
    """Apply shift_left to same-type integer registers with optional predication."""
    return _binary("shift_left", lhs, rhs, mask, pmode)


def shift_right(
    lhs: Reg | ir.Expr, rhs: Reg | ir.Expr, mask: Reg | ir.Expr | None = None, *, pmode: str = "zero"
) -> Reg:
    """Apply shift_right to same-type integer registers with optional predication."""
    return _binary("shift_right", lhs, rhs, mask, pmode)


def _scalar(
    name: str, value: Reg | ir.Expr, scalar: Scalar | ir.Expr | int | float, mask: Reg | ir.Expr, pmode: str
) -> Reg:
    expr = _expr(value)
    if not isinstance(expr.type, ir.RegType):
        raise TypeError("Vector-scalar operations require VecReg")
    dtype = DataType.INT16 if name in ("shift_lefts", "shift_rights") else expr.type.dtype
    return Reg(_call(name, [expr, _expr(scalar, dtype), _expr(mask)], pmode=pmode))


def adds(
    value: Reg | ir.Expr, scalar: Scalar | ir.Expr | int | float, mask: Reg | ir.Expr, *, pmode: str = "zero"
) -> Reg:
    """Apply adds with a governing mask and an exactly typed scalar.

    Literal scalars are normalized; existing SSA scalars are never cast.
    """
    return _scalar("adds", value, scalar, mask, pmode)


def muls(
    value: Reg | ir.Expr, scalar: Scalar | ir.Expr | int | float, mask: Reg | ir.Expr, *, pmode: str = "zero"
) -> Reg:
    """Apply muls with a governing mask and an exactly typed scalar.

    Literal scalars are normalized; existing SSA scalars are never cast.
    """
    return _scalar("muls", value, scalar, mask, pmode)


def maximums(
    value: Reg | ir.Expr, scalar: Scalar | ir.Expr | int | float, mask: Reg | ir.Expr, *, pmode: str = "zero"
) -> Reg:
    """Apply maximums with a governing mask and an exactly typed scalar.

    Literal scalars are normalized; existing SSA scalars are never cast.
    """
    return _scalar("maximums", value, scalar, mask, pmode)


def minimums(
    value: Reg | ir.Expr, scalar: Scalar | ir.Expr | int | float, mask: Reg | ir.Expr, *, pmode: str = "zero"
) -> Reg:
    """Apply minimums with a governing mask and an exactly typed scalar.

    Literal scalars are normalized; existing SSA scalars are never cast.
    """
    return _scalar("minimums", value, scalar, mask, pmode)


def shift_lefts(
    value: Reg | ir.Expr, scalar: Scalar | ir.Expr | int | float, mask: Reg | ir.Expr, *, pmode: str = "zero"
) -> Reg:
    """Apply shift_lefts with a governing mask and an exactly typed scalar.

    Literal scalars are normalized; existing SSA scalars are never cast.
    Scalar shifts require INT16 amounts, emitted as signless i16.
    """
    return _scalar("shift_lefts", value, scalar, mask, pmode)


def shift_rights(
    value: Reg | ir.Expr, scalar: Scalar | ir.Expr | int | float, mask: Reg | ir.Expr, *, pmode: str = "zero"
) -> Reg:
    """Apply shift_rights with a governing mask and an exactly typed scalar.

    Literal scalars are normalized; existing SSA scalars are never cast.
    Scalar shifts require INT16 amounts, emitted as signless i16.
    """
    return _scalar("shift_rights", value, scalar, mask, pmode)


def compares(
    value: Reg | ir.Expr, scalar: Scalar | ir.Expr | int | float, seed: Reg | ir.Expr, *, predicate: str
) -> Reg:
    """Compare each vector lane with a matching scalar under the seed mask."""
    expr = _expr(value)
    if not isinstance(expr.type, ir.RegType):
        raise TypeError("reg.compares requires VecReg")
    return Reg(_call("compares", [expr, _expr(scalar, expr.type.dtype), _expr(seed)], predicate=predicate))


def gather_lanes(value: Reg | ir.Expr, indices: Reg | ir.Expr) -> Reg:
    """Permute lanes as result[i] = value[indices[i]] using PTOAS vselr.

    Index storage width and lane count must match the source. Runtime indices
    must be within the source lane range. Only physical vselr lane counts apply.
    """
    return Reg(_call("gather_lanes", [_expr(value), _expr(indices)]))


def reduce_sum(value: Reg | ir.Expr, mask: Reg | ir.Expr, *, reassoc: bool = False) -> Reg:
    """Reduce active lanes to a one-lane VecReg of the same dtype.

    Floating sums require reassoc=True to allow the hardware reduction tree.
    Inactive input lanes contribute zero. No implicit widening is performed.
    """
    return Reg(_call("reduce_sum", [_expr(value), _expr(mask)], reassoc=reassoc))


def reduce_max(value: Reg | ir.Expr, mask: Reg | ir.Expr) -> Reg:
    """Reduce to one lane; inactive inputs contribute the dtype minimum or -inf."""
    return Reg(_call("reduce_max", [_expr(value), _expr(mask)]))


def reduce_min(value: Reg | ir.Expr, mask: Reg | ir.Expr) -> Reg:
    """Reduce to one lane; inactive inputs contribute the dtype maximum or +inf."""
    return Reg(_call("reduce_min", [_expr(value), _expr(mask)]))
