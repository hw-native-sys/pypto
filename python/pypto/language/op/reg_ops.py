# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Logical register operations, valid within an A5 InCore/AIV reg_scope."""

from typing import Any

from pypto.ir.utils import _get_span_or_capture, _normalize_reg_block_stride
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


def _lanes(value: object, name: str = "lanes") -> int:
    if isinstance(value, ir.ConstInt):
        value = value.value
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive static integer, got {value}")
    return value


def mem_bar(kind: str) -> ir.Call:
    """Order UB memory accesses within the enclosing register scope.

    ``kind`` is an exact PTO barrier type, such as ``VST_VLD``. This is a
    statement-only DSL operation; its IR carrier has no usable result.
    It does not provide DMA, cross-scope, GM or cross-core synchronization.
    """
    if type(kind) is not str:
        raise TypeError("reg.mem_bar kind must be a compile-time string")
    return _call("mem_bar", [], kind=kind)


def load(
    tile: Tile | ir.Expr,
    offset: int | Scalar | ir.Expr = 0,
    *,
    lanes: int,
    native: bool = False,
    dtype: DataType | None = None,
    dist: str | None = None,
) -> Reg:
    """Read logical Vec lanes; native attrs are the canonical IR spelling."""
    if native:
        if dtype is None or dist is None:
            raise TypeError("Native load requires dtype and dist")
        return load_native(tile, offset, dtype=dtype, lanes=lanes, mode=dist)
    if dtype is not None or dist is not None:
        raise TypeError("dtype/dist require native=True")
    return Reg(_call("load", [_expr(tile), _expr(offset)], lanes=_lanes(lanes)))


def load_native(
    tile: Tile | ir.Expr,
    offset: int | Scalar | ir.Expr = 0,
    *,
    dtype: DataType,
    lanes: int,
    mode: str = "NORM",
) -> Reg:
    """Load a full physical 256-byte register using an explicit A5 distribution.

    Unlike logical load, ``UNPK4`` reads 64 bytes, ``BRC_B32`` one 32-bit
    value, and ``E2B_B16`` eight 16-bit values. ``dtype`` may reinterpret the
    storage bits without changing the byte width. The containing register
    scope is lowered to physical PTO micro-ops, not VMI's logical vectors.
    """
    return Reg(_call("load", [_expr(tile), _expr(offset)], native=True, dtype=dtype, lanes=_lanes(lanes), dist=mode))


def store(
    tile: Tile | ir.Expr,
    offset: int | Scalar | ir.Expr = 0,
    value: Reg | ir.Expr | None = None,
    mask: Reg | ir.Expr | None = None,
    *,
    block_stride: int | None = None,
    native: bool = False,
    dist: str | None = None,
) -> Tile:
    """Write active lanes, optionally striding between 32-byte blocks.

    offset is in elements; block_stride is a positive static i16 integer.
    Gaps and inactive lanes retain their previous destination contents.
    """
    if value is None:
        raise TypeError("reg.store requires value")
    if native:
        if mask is None or dist is None or block_stride is not None:
            raise TypeError("Native store requires mask/dist and does not take block_stride")
        return store_native(tile, offset, value, mask, mode=dist)
    if dist is not None:
        raise TypeError("dist requires native=True")
    args = [_expr(tile), _expr(offset), _expr(value)]
    if mask is not None:
        args.append(_expr(mask))
    kwargs: dict[str, Any] = {}
    stride = _normalize_reg_block_stride(block_stride)
    if stride is not None:
        kwargs["block_stride"] = stride
    return Tile(expr=_call("store", args, **kwargs))


def store_native(
    tile: Tile | ir.Expr,
    offset: int | Scalar | ir.Expr,
    value: Reg | ir.Expr,
    mask: Reg | ir.Expr,
    *,
    mode: str = "NORM",
) -> Tile:
    """Store a physical register using NORM, ONEPT_B32 or PK_B32.

    The latter packs the even BF16 lanes under a 32-bit predicate; ONEPT_B32
    stores only lane zero. Offsets are destination element offsets.
    """
    return Tile(
        expr=_call("store", [_expr(tile), _expr(offset), _expr(value), _expr(mask)], native=True, dist=mode)
    )


def broadcast(scalar: Reg | Scalar | ir.Expr | int | float, *, lanes: int, dtype: DataType) -> Reg:
    """Broadcast a scalar or one-lane VecReg without implicitly casting its dtype."""
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
    """Activate a prefix using a static or runtime INDEX count.

    The caller must keep runtime counts in [0, lanes]. Constants are checked
    eagerly; zero creates an all-false mask. The count is not modified.
    """
    return Reg(_call("active_mask", [_expr(active_lanes)], lanes=_lanes(lanes)))


def interpret_cast(value: Reg | ir.Expr, *, dtype: DataType) -> Reg:
    """Reinterpret the same bits as dtype, without numeric conversion.

    The result has input_lanes * input_bits / output_bits lanes; this must
    be an integer. Both dtypes must be supported vector register dtypes.
    """
    return Reg(_call("interpret_cast", [_expr(value)], dtype=dtype))


def create_group_mask(
    active_elems_per_group: Scalar | ir.Expr | int, *, num_groups: int, group_size: int
) -> Reg:
    """Activate a prefix in each of num_groups contiguous groups.

    Lane i is active when i % group_size < active_elems_per_group. The result
    has num_groups * group_size lanes. Group geometry must be positive static
    integers. The INDEX active count may be dynamic; callers must keep it in
    [0, group_size]. Constant counts are checked at construction. PTOAS 0.70
    requires dynamic counts to use power-of-two groups and 32-bit consumers;
    byte/halfword consumers require a constant active count.
    """
    return Reg(
        _call(
            "create_group_mask",
            [_expr(active_elems_per_group)],
            num_groups=_lanes(num_groups),
            group_size=_lanes(group_size),
        )
    )


def chist(acc: Reg | ir.Expr, source: Reg | ir.Expr, mask: Reg | ir.Expr) -> Reg:
    """Accumulate a cumulative histogram of active UINT8 source lanes.

    acc is a UINT16 vector with 128 or 256 bins. Result bin b is acc[b] plus
    the number of active source values <= b, modulo 65536. A 128-bin result
    covers bins 0 through 127. The mask matches source lanes, not bin count;
    an empty mask preserves acc. This op does not mutate acc.
    """
    return Reg(_call("chist", [_expr(acc), _expr(source), _expr(mask)]))


def dhist(acc: Reg | ir.Expr, source: Reg | ir.Expr, mask: Reg | ir.Expr) -> Reg:
    """Accumulate a distribution histogram of active UINT8 source lanes.

    acc is a UINT16 vector with 128 or 256 bins. Result bin b is acc[b] plus
    the number of active source values == b, modulo 65536. A 128-bin result
    covers bins 0 through 127. The mask matches source lanes, not bin count;
    an empty mask preserves acc. This op does not mutate acc.
    """
    return Reg(_call("dhist", [_expr(acc), _expr(source), _expr(mask)]))


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
    value: Reg | ir.Expr,
    mask: Reg | ir.Expr | None = None,
    *,
    dtype: DataType,
    rounding: str | None = None,
    saturate: bool | None = None,
    native: bool = False,
    part: str | None = None,
    lanes: int | None = None,
) -> Reg:
    """Convert a dtype pair, preserving logical lanes by default."""
    if native:
        if mask is None or part is None or lanes is None:
            raise TypeError("Native convert requires mask, part and lanes")
        return convert_native(
            value, mask, dtype=dtype, lanes=lanes, part=part, rounding=rounding, saturate=saturate
        )
    if mask is not None or part is not None or lanes is not None:
        raise TypeError("mask/part/lanes require native=True")
    expr = _expr(value)
    if not isinstance(expr.type, ir.RegType) or expr.type.kind != ir.RegKind.VecReg:
        raise TypeError("reg.convert requires VecReg")
    src = expr.type.dtype
    if src == dtype and rounding is None and saturate is None:
        return Reg(expr)
    kwargs: dict[str, Any] = {"dtype": dtype}
    if rounding is None and src == DataType.FP32:
        if dtype in (DataType.FP16, DataType.BF16, DataType.FP8E4M3FN):
            rounding = "R"
        elif dtype == DataType.INT32:
            rounding = "Z"
    if rounding is not None:
        kwargs["rounding"] = rounding
    if saturate is not None:
        kwargs["saturate"] = saturate
    return Reg(_call("convert", [expr], **kwargs))


def convert_native(
    value: Reg | ir.Expr,
    mask: Reg | ir.Expr,
    *,
    dtype: DataType,
    lanes: int,
    part: str,
    rounding: str | None = None,
    saturate: bool | None = None,
) -> Reg:
    """Convert full physical registers while preserving PART_P0/EVEN/ODD.

    In particular FP32 -> BF16 returns a full 128-lane physical register
    with populated even or odd lanes, unlike the compact logical conversion.
    Use ``dintlv`` or ``store_native(mode="PK_B32")`` to pack even lanes.
    """
    kwargs: dict[str, Any] = {"dtype": dtype, "lanes": _lanes(lanes), "part": part, "native": True}
    if rounding is not None:
        kwargs["rounding"] = rounding
    if saturate is not None:
        kwargs["saturate"] = saturate
    return Reg(_call("convert", [_expr(value), _expr(mask)], **kwargs))


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
    if not isinstance(expr.type, ir.RegType) or expr.type.kind != ir.RegKind.VecReg:
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
    if not isinstance(expr.type, ir.RegType) or expr.type.kind != ir.RegKind.VecReg:
        raise TypeError("reg.compares requires VecReg")
    return Reg(_call("compares", [expr, _expr(scalar, expr.type.dtype), _expr(seed)], predicate=predicate))


def gather_lanes(value: Reg | ir.Expr, indices: Reg | ir.Expr) -> Reg:
    """Permute lanes as result[i] = value[indices[i]] using PTOAS vselr.

    Index storage width and lane count must match the source. Runtime indices
    must be within the source lane range. Only physical vselr lane counts apply.
    """
    return Reg(_call("gather_lanes", [_expr(value), _expr(indices)]))


def _reduction(
    name: str, value: Reg | ir.Expr, mask: Reg | ir.Expr, group_size: int | None, **kwargs: Any
) -> Reg:
    if group_size is not None:
        kwargs["group_size"] = _lanes(group_size, "group_size")
    return Reg(_call(name, [_expr(value), _expr(mask)], **kwargs))


def reduce_sum(
    value: Reg | ir.Expr,
    mask: Reg | ir.Expr,
    *,
    reassoc: bool = False,
    group_size: int | None = None,
    native: bool = False,
) -> Reg:
    """Reduce active lanes to one lane, or to one lane per group.

    Floating sums require reassoc=True to allow the hardware reduction tree.
    Inactive input lanes contribute zero. No implicit widening is performed.
    With group_size, the source splits into equal contiguous groups and the
    result holds one lane per group in group order; see group_broadcast.
    """
    if native:
        if group_size is not None:
            raise TypeError("Native reduction does not take group_size")
        return reduce_sum_native(value, mask, reassoc=reassoc)
    return _reduction("reduce_sum", value, mask, group_size, reassoc=reassoc)


def reduce_max(
    value: Reg | ir.Expr, mask: Reg | ir.Expr, *, group_size: int | None = None, native: bool = False
) -> Reg:
    """Reduce to one lane, or one lane per group; inactive inputs give the dtype minimum or -inf."""
    if native:
        if group_size is not None:
            raise TypeError("Native reduction does not take group_size")
        return reduce_max_native(value, mask)
    return _reduction("reduce_max", value, mask, group_size)


def reduce_min(value: Reg | ir.Expr, mask: Reg | ir.Expr, *, group_size: int | None = None) -> Reg:
    """Reduce to one lane, or one lane per group; inactive inputs give the dtype maximum or +inf."""
    return _reduction("reduce_min", value, mask, group_size)


def reduce_max_native(value: Reg | ir.Expr, mask: Reg | ir.Expr) -> Reg:
    """Native FP32 maximum reduction: result in lane zero of the full register."""
    return Reg(_call("reduce_max", [_expr(value), _expr(mask)], native=True))


def reduce_sum_native(value: Reg | ir.Expr, mask: Reg | ir.Expr, *, reassoc: bool) -> Reg:
    """Native FP32 sum reduction: result in lane zero of the full register."""
    return Reg(_call("reduce_sum", [_expr(value), _expr(mask)], native=True, reassoc=reassoc))


def group_broadcast(value: Reg | ir.Expr, *, lanes: int) -> Reg:
    """Expand a grouped reduction result back across its groups.

    The source holds one lane per group, as returned by a reduction with
    group_size; lanes is the width to expand back to and must be a multiple of
    that group count. Result lane i takes the source lane of group
    i // (lanes // groups).
    """
    return Reg(_call("group_broadcast", [_expr(value)], lanes=_lanes(lanes)))


def _pair(name: str, args: list[ir.Expr], **kwargs: Any) -> tuple[Reg, Reg]:
    # Both projections must reference one Call so the parser emits the operation
    # once and binds its two independent SSA results.
    call = _call(name, args, **kwargs)
    return Reg(ir.TupleGetItemExpr(call, 0, call.span)), Reg(ir.TupleGetItemExpr(call, 1, call.span))


def _scalar_for(value: Scalar | ir.Expr | int | float, vector: Reg | ir.Expr) -> ir.Expr:
    expr = _expr(vector)
    if not isinstance(expr.type, ir.RegType) or expr.type.kind != ir.RegKind.VecReg:
        raise TypeError("Expected a vector register for scalar dtype inference")
    return _expr(value, expr.type.dtype)


def expdif(x: Reg | ir.Expr, max_value: Reg | ir.Expr, mask: Reg | ir.Expr, *, pmode: str = "zero") -> Reg:
    """Compute exp(x - max_value) on active lanes; always return FP32.

    Inputs must have matching FP16/FP32 types and lane counts. Inactive
    result lanes are zero. This emits the fused VMI operation.
    """
    return Reg(_call("expdif", [_expr(x), _expr(max_value), _expr(mask)], pmode=pmode))


def axpy(
    x: Reg | ir.Expr,
    acc: Reg | ir.Expr,
    alpha: Scalar | ir.Expr | int | float,
    mask: Reg | ir.Expr,
    *,
    pmode: str = "zero",
) -> Reg:
    """Compute fused alpha * x + acc for matching FP16/FP32 vectors.

    Alpha is a scalar matching the vector dtype; Python literals are normalized
    to it. Inactive result lanes are zero.
    """
    return Reg(_call("axpy", [_expr(x), _expr(acc), _scalar_for(alpha, x), _expr(mask)], pmode=pmode))


def mula(
    acc: Reg | ir.Expr,
    lhs: Reg | ir.Expr,
    rhs: Reg | ir.Expr,
    mask: Reg | ir.Expr | None = None,
    *,
    pmode: str = "zero",
) -> Reg:
    """Compute fused acc + lhs * rhs with the accumulator first.

    All vectors must have identical FP16/BF16/FP32 or 16/32-bit integer types.
    The result is a new SSA value. Inactive lanes are zero when masked.
    """
    args = [_expr(acc), _expr(lhs), _expr(rhs)]
    if mask is not None:
        args.append(_expr(mask))
    return Reg(_call("mula", args, pmode=pmode))


def lrelu(
    x: Reg | ir.Expr,
    slope: Scalar | ir.Expr | int | float,
    mask: Reg | ir.Expr,
    *,
    pmode: str = "zero",
) -> Reg:
    """Apply leaky ReLU with one matching FP16/FP32 scalar slope.

    Active lanes compute x if x > 0, otherwise slope * x. Inactive lanes are zero.
    """
    return Reg(_call("lrelu", [_expr(x), _scalar_for(slope, x), _expr(mask)], pmode=pmode))


def prelu(x: Reg | ir.Expr, alpha: Reg | ir.Expr, mask: Reg | ir.Expr, *, pmode: str = "zero") -> Reg:
    """Apply PReLU with a per-lane FP16/FP32 alpha vector matching x.

    Active lanes compute max(x, 0) + alpha * min(x, 0); inactive lanes are zero.
    """
    return Reg(_call("prelu", [_expr(x), _expr(alpha), _expr(mask)], pmode=pmode))


def mull(
    lhs: Reg | ir.Expr,
    rhs: Reg | ir.Expr,
    mask: Reg | ir.Expr,
    *,
    pmode: str = "zero",
) -> tuple[Reg, Reg]:
    """Multiply INT32/UINT32 vectors and return (low, high) 32-bit halves.

    Requires 64, 128, or 256 lanes. Both results retain input signedness;
    inactive lanes are zero."""
    return _pair("mull", [_expr(lhs), _expr(rhs), _expr(mask)], pmode=pmode)


def intlv(
    lhs: Reg | ir.Expr,
    rhs: Reg | ir.Expr,
    mask: Reg | ir.Expr | None = None,
    *,
    pmode: str = "zero",
) -> tuple[Reg, Reg]:
    """Interleave two matching even-length vectors; return (low, high).

    Each result has the input lane count and dtype. With a mask, inactive
    lanes are zero. Omitting the mask rearranges all lanes without clearing
    either result."""
    args = [_expr(lhs), _expr(rhs)]
    if mask is not None:
        args.append(_expr(mask))
    return _pair("intlv", args, pmode=pmode)


def dintlv(
    lhs: Reg | ir.Expr,
    rhs: Reg | ir.Expr,
    mask: Reg | ir.Expr | None = None,
    *,
    pmode: str = "zero",
) -> tuple[Reg, Reg]:
    """Deinterleave two matching even-length vectors; return (even, odd).

    Each result has the input lane count and dtype. With a mask, inactive
    lanes are zero. Omitting the mask rearranges all lanes without clearing
    either result."""
    args = [_expr(lhs), _expr(rhs)]
    if mask is not None:
        args.append(_expr(mask))
    return _pair("dintlv", args, pmode=pmode)


def addc(
    lhs: Reg | ir.Expr,
    rhs: Reg | ir.Expr,
    mask: Reg | ir.Expr,
) -> tuple[Reg, Reg]:
    """Add INT32/UINT32 vectors; return (result, carry_mask).

    Carry masks have the same lane count as the INT32/UINT32 vectors.
    For subtraction, a true carry bit means no borrow.
    """
    return _pair("addc", [_expr(lhs), _expr(rhs), _expr(mask)])


def subc(
    lhs: Reg | ir.Expr,
    rhs: Reg | ir.Expr,
    mask: Reg | ir.Expr,
) -> tuple[Reg, Reg]:
    """Subtract INT32/UINT32 vectors; return (result, not_borrow_mask).

    Carry masks have the same lane count as the INT32/UINT32 vectors.
    For subtraction, a true carry bit means no borrow.
    """
    return _pair("subc", [_expr(lhs), _expr(rhs), _expr(mask)])


def addcs(
    lhs: Reg | ir.Expr,
    rhs: Reg | ir.Expr,
    carry_in: Reg | ir.Expr,
    mask: Reg | ir.Expr,
) -> tuple[Reg, Reg]:
    """Add lhs + rhs + carry_in; return (result, carry_mask).

    Carry masks have the same lane count as the INT32/UINT32 vectors.
    For subtraction, a true carry bit means no borrow.
    """
    return _pair("addcs", [_expr(lhs), _expr(rhs), _expr(carry_in), _expr(mask)])


def subcs(
    lhs: Reg | ir.Expr,
    rhs: Reg | ir.Expr,
    carry_in: Reg | ir.Expr,
    mask: Reg | ir.Expr,
) -> tuple[Reg, Reg]:
    """Subtract lhs - rhs - (1 - carry_in); return (result, not_borrow_mask).

    Carry masks have the same lane count as the INT32/UINT32 vectors.
    For subtraction, a true carry bit means no borrow.
    """
    return _pair("subcs", [_expr(lhs), _expr(rhs), _expr(carry_in), _expr(mask)])


def gather(
    tile: Tile | ir.Expr,
    offsets: Reg | ir.Expr,
    mask: Reg | ir.Expr,
    base_offset: Scalar | ir.Expr | int = 0,
    *,
    pmode: str = "zero",
) -> Reg:
    """Gather active elements from a contiguous Vec Tile; inactive lanes are zero.

    Offsets are UINT16/UINT32 element indices relative to the Tile view origin
    plus the scalar INDEX base_offset (also measured in elements).
    Use UINT16 offsets for 8/16-bit sources and UINT32 for 32-bit sources.
    INT8/UINT8 sources are zero-extended
    to INT16/UINT16 respectively. The caller must ensure active indices are
    in bounds and initialized; the mask suppresses inactive memory accesses.
    """
    args = [_expr(tile), _expr(offsets), _expr(mask)]
    if isinstance(base_offset, bool):
        raise TypeError("base_offset must be an INDEX scalar or integer, not bool")
    if not isinstance(base_offset, int) or base_offset != 0:
        args.append(_expr(base_offset))
    return Reg(_call("gather", args, pmode=pmode))


def scatter(
    tile: Tile | ir.Expr,
    offsets: Reg | ir.Expr,
    value: Reg | ir.Expr,
    mask: Reg | ir.Expr,
    base_offset: Scalar | ir.Expr | int = 0,
) -> Tile:
    """Scatter active lanes and return an alias of the destination Vec Tile.

    Offsets are UINT16/UINT32 element indices relative to the Tile view origin
    plus the scalar INDEX base_offset (also measured in elements).
    Use UINT16 offsets for 8/16-bit values and UINT32 for 32-bit values.
    Values must match the Tile dtype. Value, offsets and mask lane counts must
    agree. Active indices must be in bounds and pairwise distinct. Unwritten
    elements retain their previous contents; valid_shape is not enlarged.
    """
    args = [_expr(tile), _expr(offsets), _expr(value), _expr(mask)]
    if isinstance(base_offset, bool):
        raise TypeError("base_offset must be an INDEX scalar or integer, not bool")
    if not isinstance(base_offset, int) or base_offset != 0:
        args.append(_expr(base_offset))
    return Tile(expr=_call("scatter", args))


def squeeze(value: Reg | ir.Expr, mask: Reg | ir.Expr) -> Reg:
    """Pack selected UINT16/UINT32 lanes in order; the remaining lanes are undefined.

    value must fill one 256-byte register (128 UINT16 or 64 UINT32 lanes).
    Use squeeze_store_step(curr_ureg, dst, value, mask) to append the result to UB.
    """
    return Reg(_call("squeeze", [_expr(value), _expr(mask)]))


def squeeze_store_init(dst: Tile | ir.Expr, offset: int | Scalar | ir.Expr = 0) -> ir.Expr:
    """Initialize the UReg state for dense squeeze-store output.

    Args:
        dst: Static UINT16/UINT32 Vec Tile, one-dimensional or a two-dimensional
            view that is contiguous in flattened row-major order.
        offset: Initial element offset from the Tile view origin, evaluated
            once. The existing view start_offset is included in the address.

    The Tile remains the UB memory destination; this operation returns the
    opaque UReg state and does not initialize the Tile's storage. Offset is a
    linear element index, not a pointer or a Tile coordinate. Carry the
    returned value through a loop with ``pl.range(..., init_values=(ureg,))``
    and finish it with :func:`squeeze_store_flush`, including after zero loop
    iterations. Only one chain may be active in the current AR-based lowering;
    after flush another chain may open in the same register scope.

    Returns:
        Opaque ``RegKind.UReg`` value representing the initialized append state.
    """
    return _call("squeeze_store_init", [_expr(dst), _expr(offset)])


def squeeze_store_step(
    curr_ureg: ir.Expr, dst: Tile | ir.Expr, value: Reg | ir.Expr, mask: Reg | ir.Expr
) -> ir.Expr:
    """Append mask-selected lanes from one VecReg and return updated state.

    Args:
        curr_ureg: Current opaque UReg state returned by init or the previous step.
        dst: The same destination Tile passed to init.
        value: One full 256-byte UINT16/UINT32 VecReg.
        mask: Matching BOOL MaskReg; true lanes are appended in input order.

    The position continues from init and earlier steps; there is no per-step
    offset. The caller guarantees enough output capacity for all selected lanes.
    """
    return _call("squeeze_store_step", [curr_ureg, _expr(dst), _expr(value), _expr(mask)])


def squeeze_store_flush(final_ureg: ir.Expr, dst: Tile | ir.Expr) -> Tile:
    """Finalize the append sequence and return an alias of its destination Tile.

    Args:
        final_ureg: Final opaque UReg state produced by init/step and loop-carried SSA.
        dst: The same destination Tile passed to init and every step.

    Exactly one flush is required even if no step ran. An empty chain leaves
    the destination unchanged; flush does not initialize the unwritten tail.
    """
    return Tile(expr=_call("squeeze_store_flush", [final_ureg, _expr(dst)]))
