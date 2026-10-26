# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile element-wise arithmetic, comparisons, and scalar dispatch."""

from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import (
    Call,
    Expr,
    ScalarType,
    Span,
)

from ..utils import (
    _get_span_or_capture,
    _normalize_const_to_dtype,
    _normalize_scalar_operand,
    _normalize_signless_same_width_scalar_operand,
    resolve_cast_mode,
    resolve_saturation_deviation,
)


def _create_tile_binary_call(
    tile_op_name: str,
    scalar_op_name: str,
    lhs: Expr,
    rhs: int | float | Expr,
    span: Span,
) -> Call:
    """Create a tile binary call with scalar auto-dispatch."""
    rhs_expr = _normalize_scalar_operand(lhs, rhs, span)
    if isinstance(rhs_expr.type, ScalarType):
        return _ir_core.create_op_call(scalar_op_name, [lhs, rhs_expr], {}, span)
    return _ir_core.create_op_call(tile_op_name, [lhs, rhs_expr], {}, span)


# ============================================================================
# Element-wise Operations
# ============================================================================


def mul(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise multiplication of tile and tile or scalar.

    Supports broadcasting for two tiles. Scalar rhs canonicalizes to tile.muls.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile or scalar
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise multiplication
    """
    actual_span = _get_span_or_capture(span)
    return _create_tile_binary_call("tile.mul", "tile.muls", lhs, rhs, actual_span)


def add(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise addition of tile and tile or scalar.

    Supports broadcasting for two tiles. Scalar rhs canonicalizes to tile.adds.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile or scalar
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise addition
    """
    actual_span = _get_span_or_capture(span)
    return _create_tile_binary_call("tile.add", "tile.adds", lhs, rhs, actual_span)


def div(
    lhs: Expr,
    rhs: int | float | Expr,
    span: Span | None = None,
    *,
    high_precision: bool = False,
) -> Call:
    """Element-wise division of tile and tile or scalar.

    Tile-tile division requires identical physical and valid shapes. Scalar rhs
    canonicalizes to tile.divs, which does not expose the tdiv precision mode.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile or scalar
        span: Optional source span for debugging (auto-captured if not provided)
        high_precision: Whether to select PTOAS's high-precision division mode.
            Only available when ``rhs`` has TileType.

    Returns:
        Call expression for element-wise division
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_scalar_operand(lhs, rhs, actual_span)
    if isinstance(rhs_expr.type, ScalarType):
        if high_precision:
            # TypeError, matching the unified pl.* guards: a kwarg this operand
            # combination cannot honour is a wrong-arguments error, not a bad value.
            raise TypeError("tile.div(high_precision=True) requires a Tile rhs")
        return _ir_core.create_op_call("tile.divs", [lhs, rhs_expr], {}, actual_span)
    kwargs: dict[str, Any] = {"high_precision": True} if high_precision else {}
    return _ir_core.create_op_call("tile.div", [lhs, rhs_expr], kwargs, actual_span)


def sub(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise subtraction of tile and tile or scalar.

    Supports broadcasting for two tiles. Scalar rhs canonicalizes to tile.subs.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile or scalar
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise subtraction
    """
    actual_span = _get_span_or_capture(span)
    return _create_tile_binary_call("tile.sub", "tile.subs", lhs, rhs, actual_span)


def rem(
    lhs: Expr,
    rhs: Expr,
    tmp: Expr,
    span: Span | None = None,
    *,
    high_precision: bool = False,
) -> Call:
    """Element-wise remainder (modulo) of two tiles.

    Computes lhs % rhs element-wise. Maps to the TREM hardware intrinsic.
    On A2/A3, every INT32 input element must be in ``[-2**24, 2**24]``.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        tmp: Same-dtype 2D scratch tile. On A2/A3 its physical and valid
            capacity must provably provide at least two rows and cover every
            ``lhs`` column, and it must not overlap either source.
        span: Optional source span for debugging (auto-captured if not provided)
        high_precision: Whether to select PTOAS's high-precision TREM mode.
            This mode is defined only for FP32 and is ignored by A2/A3 hardware.

    Returns:
        Call expression for element-wise remainder
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"high_precision": True} if high_precision else {}
    return _ir_core.create_op_call("tile.rem", [lhs, rhs, tmp], kwargs, actual_span)


def rems(lhs: Expr, rhs: int | float | Expr, tmp: Expr, span: Span | None = None) -> Call:
    """Element-wise remainder (modulo) of tile and scalar.

    Computes lhs % rhs element-wise. Maps to the TREMS hardware intrinsic.
    On A2/A3, source valid extents must be provably positive and every INT32
    source element and scalar must be in ``[-2**24, 2**24]``.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        tmp: Same-dtype 2D scratch tile. On A2/A3 its physical and valid
            capacity must provably provide at least one row and cover every
            ``lhs`` column, and it must not overlap ``lhs``.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise remainder with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.rems", [lhs, rhs_expr, tmp], {}, actual_span)


def part_add(src0: Expr, src1: Expr, span: Span | None = None) -> Call:
    """Partial element-wise add of two tiles.

    Adds over the destination valid region; where only one source is valid the
    result copies that source. Maps to the TPARTADD hardware intrinsic.

    Args:
        src0: First source tile (TileType)
        src1: Second source tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for partial element-wise add
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.part_add", [src0, src1], {}, actual_span)


def part_mul(src0: Expr, src1: Expr, span: Span | None = None) -> Call:
    """Partial element-wise multiply of two tiles.

    Multiplies over the destination valid region; where only one source is valid
    the result copies that source. Maps to the TPARTMUL hardware intrinsic.

    Args:
        src0: First source tile (TileType)
        src1: Second source tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for partial element-wise multiply
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.part_mul", [src0, src1], {}, actual_span)


def part_max(src0: Expr, src1: Expr, span: Span | None = None) -> Call:
    """Partial element-wise max of two tiles.

    Takes the max over the destination valid region; where only one source is
    valid the result copies that source. Maps to the TPARTMAX hardware intrinsic.

    Args:
        src0: First source tile (TileType)
        src1: Second source tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for partial element-wise max
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.part_max", [src0, src1], {}, actual_span)


def part_min(src0: Expr, src1: Expr, span: Span | None = None) -> Call:
    """Partial element-wise min of two tiles.

    Takes the min over the destination valid region; where only one source is
    valid the result copies that source. Maps to the TPARTMIN hardware intrinsic.

    Args:
        src0: First source tile (TileType)
        src1: Second source tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for partial element-wise min
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.part_min", [src0, src1], {}, actual_span)


def fmod(
    lhs: Expr,
    rhs: Expr,
    span: Span | None = None,
    *,
    high_precision: bool = False,
) -> Call:
    """Element-wise truncating remainder of two tiles.

    Computes the truncating remainder of lhs / rhs element-wise (matching
    ``torch.fmod``; the result follows the dividend sign). Maps to TFMOD.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)
        high_precision: Whether to select PTOAS's high-precision TFMOD mode.
            This mode is defined only for FP32.

    Returns:
        Call expression for element-wise truncating remainder
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"high_precision": True} if high_precision else {}
    return _ir_core.create_op_call("tile.fmod", [lhs, rhs], kwargs, actual_span)


def fmods(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise truncating remainder of tile and scalar.

    Computes the truncating remainder of lhs / rhs element-wise (matching
    ``torch.fmod``; the result follows the dividend sign). Maps to TFMODS.
    A2/A3 requires every source valid extent to be provably positive.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise truncating remainder with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.fmods", [lhs, rhs_expr], {}, actual_span)


def shl(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise left shift of two tiles.

    Computes lhs << rhs element-wise. Maps to the TSHL hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise left shift
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.shl", [lhs, rhs], {}, actual_span)


def shls(lhs: Expr, rhs: int | Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise left shift of tile and scalar.

    Computes lhs << rhs element-wise. Maps to the TSHLS hardware intrinsic.

    Note:
        The scalar shift amount must be zero or positive; negative values are
        not supported by the hardware and will be rejected by codegen.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar shift amount; must be >= 0. A constant literal is re-stamped
            to the lhs element dtype (the IR permits any integer width -- codegen
            casts the shift count to i32); a typed Expr is used as-is
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise left shift with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.shls", [lhs, rhs_expr], {}, actual_span)


def shr(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise right shift of two tiles.

    Computes lhs >> rhs element-wise. Maps to the TSHR hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise right shift
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.shr", [lhs, rhs], {}, actual_span)


def shrs(lhs: Expr, rhs: int | Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise right shift of tile and scalar.

    Computes lhs >> rhs element-wise. Maps to the TSHRS hardware intrinsic.

    Note:
        The scalar shift amount must be zero or positive; negative values are
        not supported by the hardware and will be rejected by codegen.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar shift amount; must be >= 0. A constant literal is re-stamped
            to the lhs element dtype (the IR permits any integer width -- codegen
            casts the shift count to i32); a typed Expr is used as-is
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise right shift with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.shrs", [lhs, rhs_expr], {}, actual_span)


def and_(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise AND of two tiles.

    Computes lhs & rhs element-wise. Maps to the TAND hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise AND
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.and", [lhs, rhs], {}, actual_span)


def ands(lhs: Expr, rhs: int | Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise AND of tile and scalar.

    Computes lhs & rhs element-wise. Maps to the TANDS hardware intrinsic.

    Args:
        lhs: Tile (TileType)
        rhs: Same-width signless integer scalar (int/Expr)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise AND with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_signless_same_width_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.ands", [lhs, rhs_expr], {}, actual_span)


def or_(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise OR of two tiles.

    Computes lhs | rhs element-wise. Maps to the TOR hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise OR
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.or", [lhs, rhs], {}, actual_span)


def ors(lhs: Expr, rhs: int | Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise OR of tile and scalar.

    Computes lhs | rhs element-wise. Maps to the TORS hardware intrinsic.

    Args:
        lhs: Tile (TileType)
        rhs: Same-width signless integer scalar (int/Expr)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise OR with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_signless_same_width_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.ors", [lhs, rhs_expr], {}, actual_span)


def xor(lhs: Expr, rhs: Expr, tmp: Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise XOR of two tiles.

    Computes lhs ^ rhs element-wise. Maps to the TXOR hardware intrinsic.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        tmp: Temporary tile (TileType) required by the hardware
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise XOR
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.xor", [lhs, rhs, tmp], {}, actual_span)


def xors(lhs: Expr, rhs: int | Expr, tmp: Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise XOR of tile and scalar.

    Computes lhs ^ rhs element-wise. Maps to the TXORS hardware intrinsic.

    Args:
        lhs: Tile (TileType)
        rhs: Same-width signless integer scalar (int/Expr)
        tmp: Temporary tile (TileType) required by the hardware
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise XOR with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_signless_same_width_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.xors", [lhs, rhs_expr, tmp], {}, actual_span)


def prelu(tile: Expr, slope: Expr, tmp: Expr, span: Span | None = None) -> Call:
    """Element-wise parametric ReLU of a tile.

    Computes prelu(tile, slope) element-wise. Maps to the TPRELU hardware intrinsic.

    Args:
        tile: Input tile (TileType)
        slope: Slope tile (TileType) used for negative values
        tmp: Temporary tile (TileType) required by the hardware
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise parametric ReLU
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.prelu", [tile, slope, tmp], {}, actual_span)


def addc(lhs: Expr, rhs: Expr, rhs2: Expr, span: Span | None = None) -> Call:
    """Element-wise carry addition of three tiles.

    Computes ``src0 + src1 + carry`` element-wise. Maps to TADDC.

    Args:
        lhs: First source tile (TileType)
        rhs: Second source tile (TileType)
        rhs2: Per-element carry-in tile (TileType), normally containing 0 or 1
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise ternary addition
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.addc", [lhs, rhs, rhs2], {}, actual_span)


def subc(lhs: Expr, rhs: Expr, rhs2: Expr, span: Span | None = None) -> Call:
    """Element-wise carry subtraction of three tiles.

    Computes ``src0 - src1 + carry`` element-wise. Maps to TSUBC.

    Args:
        lhs: Minuend tile (TileType)
        rhs: Subtrahend tile (TileType)
        rhs2: Per-element carry-in tile (TileType), normally containing 0 or 1
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise ternary subtraction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.subc", [lhs, rhs, rhs2], {}, actual_span)


def addsc(lhs: Expr, rhs: int | float | Expr, rhs2: Expr, span: Span | None = None) -> Call:
    """Element-wise scalar carry addition.

    Computes ``src0 + scalar + carry`` element-wise. Maps to TADDSC.

    Args:
        lhs: Source tile (TileType)
        rhs: Scalar addend with the same dtype as ``lhs``
        rhs2: Per-element carry-in tile (TileType), normally containing 0 or 1
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise tile-scalar-tile addition
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.addsc", [lhs, rhs_expr, rhs2], {}, actual_span)


def subsc(lhs: Expr, rhs: int | float | Expr, rhs2: Expr, span: Span | None = None) -> Call:
    """Element-wise scalar carry subtraction.

    Computes ``src0 - scalar + carry`` element-wise. Maps to TSUBSC.

    Args:
        lhs: Minuend tile (TileType)
        rhs: Scalar subtrahend with the same dtype as ``lhs``
        rhs2: Per-element carry-in tile (TileType), normally containing 0 or 1
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise tile-scalar-tile subtraction
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.subsc", [lhs, rhs_expr, rhs2], {}, actual_span)


def lrelu(tile: Expr, slope: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise leaky ReLU of a tile with scalar slope.

    Computes max(x, slope * x) element-wise. Maps to the TLRELU hardware intrinsic.

    Args:
        tile: Input tile (TileType)
        slope: Scalar slope for negative values (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise leaky ReLU
    """
    actual_span = _get_span_or_capture(span)
    # The slope is a float coefficient fixed by the op, not a tile element value.
    slope_expr = _normalize_const_to_dtype(slope, DataType.FP32, actual_span)
    return _ir_core.create_op_call("tile.lrelu", [tile, slope_expr], {}, actual_span)


def sel(mask: Expr, lhs: Expr, rhs: Expr, tmp: Expr, span: Span | None = None) -> Call:
    """Per-element selection between two tiles using a predicate mask tile.

    For each element (i, j): dst[i,j] = lhs[i,j] if mask[i,j] is true, else rhs[i,j].
    Maps to the TSEL hardware intrinsic. The mask encoding is target-defined.

    Args:
        mask: Predicate mask tile (TileType); encoding is target-defined
        lhs: Source tile 0, selected where mask is true (TileType)
        rhs: Source tile 1, selected where mask is false (TileType)
        tmp: Scratch tile required by TSEL (TileType UINT32 [1, 16] on A2/A3)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for per-element tile selection
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.sel", [mask, lhs, rhs, tmp], {}, actual_span)


def sels(
    mask: Expr,
    src: Expr,
    tmp: Expr,
    scalar: int | float | Expr,
    span: Span | None = None,
) -> Call:
    """Per-element selection between a source tile and a scalar.

    For each element (i, j): dst[i,j] = src[i,j] if mask[i,j] is true,
    else scalar. Maps to the TSELS hardware intrinsic.

    Args:
        mask: Predicate mask tile (TileType); encoding is target-defined
        src: Source tile, selected where mask is true (TileType)
        tmp: Scratch tile required by TSELS (TileType)
        scalar: Scalar value, selected where mask is false. For an unsigned
            integer src, constants use the same-width signed PTOAS scalar type
            while preserving their bit pattern.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for per-element tile/scalar selection
    """
    actual_span = _get_span_or_capture(span)
    scalar_expr = _normalize_signless_same_width_scalar_operand(
        src,
        scalar,
        actual_span,
        retype_constants=True,
    )
    return _ir_core.create_op_call("tile.sels", [mask, src, tmp, scalar_expr], {}, actual_span)


def select(
    cond: Expr,
    on_true: int | float | Expr,
    on_false: int | float | Expr,
    span: Span | None = None,
) -> Call:
    """Per-element value selection between two operands.

    For each element (i, j): ``dst[i,j] = on_true[i,j] if cond[i,j] else on_false[i,j]``.

    Unlike :func:`sel` / :func:`sels` this is a composite op: it carries no
    ``tmp`` operand, and ``LowerCompositeOps`` expands it into the
    ``tile.create`` / ``tile.sel`` / ``tile.sels`` chain with the architecture's
    scratch geometry filled in.

    Both branches are evaluated -- this selects values, it does not branch.

    Two tile branches must share a physical shape, valid extents, and dtype: the lowered TSEL reads both
    sources element-wise, does not broadcast, and has one element type for both
    sources and the result. Widen with :func:`expands` /
    :func:`row_expand` / :func:`col_expand`, or :func:`cast`, first.

    Args:
        cond: The packed predicate mask returned by :func:`cmp` / :func:`cmps`
            for this result's geometry. A 0/1 value tile is not accepted -- a
            mask's valid extent counts carrier bytes where a value's counts
            elements, so the two cannot be told apart reliably; convert one with
            ``cmps(value, 0, cmp_type=1)``
        on_true: Value selected where cond is true; a tile or a constant scalar
        on_false: Value selected where cond is false; same forms as on_true
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for per-element selection

    Raises:
        ValueError: If both on_true and on_false are scalars -- there is no
            shape to select into -- if two tile branches disagree on shape,
            valid extents, or dtype, or if a scalar branch is not a compile-time
            constant (it is materialized with ``tile.full``, which needs one)
    """
    actual_span = _get_span_or_capture(span)

    def as_tile(value: int | float | Expr) -> Expr | None:
        """The operand as a tile Expr, or None when it is a scalar branch."""
        if isinstance(value, _ir_core.Expr) and isinstance(value.type, _ir_core.TileType):
            return value
        return None

    true_tile = as_tile(on_true)
    false_tile = as_tile(on_false)
    shaped = true_tile if true_tile is not None else false_tile
    if shaped is None:
        raise ValueError(
            "tile.select requires at least one of on_true/on_false to be a Tile, but both are "
            f"scalars ({on_true!r}, {on_false!r}); there is no shape to select into"
        )
    # Re-stamp a scalar branch to the selected tile's element type, matching how
    # tile.sels normalizes its own scalar operand.
    true_arg: Expr = (
        true_tile
        if true_tile is not None
        else _normalize_signless_same_width_scalar_operand(
            shaped, on_true, actual_span, retype_constants=True
        )
    )
    false_arg: Expr = (
        false_tile
        if false_tile is not None
        else _normalize_signless_same_width_scalar_operand(
            shaped, on_false, actual_span, retype_constants=True
        )
    )
    return _ir_core.create_op_call("tile.select", [cond, true_arg, false_arg], {}, actual_span)


def muls(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise multiplication of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise multiplication with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.muls", [lhs, rhs_expr], {}, actual_span)


def adds(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise addition of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise addition with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.adds", [lhs, rhs_expr], {}, actual_span)


def divs(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise division of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise division with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.divs", [lhs, rhs_expr], {}, actual_span)


def subs(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise subtraction of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise subtraction with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.subs", [lhs, rhs_expr], {}, actual_span)


def cmp(lhs: Expr, rhs: Expr, cmp_type: int = 0, span: Span | None = None) -> Call:
    """Element-wise comparison of two tiles (returns a packed predicate mask tile).

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        cmp_type: Comparison type (int):
                  EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5
                  Default: 0 (EQ)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for a packed predicate mask tile.
        Use tile.sel with an explicit tmp tile to materialize values.

    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"cmp_type": cmp_type}
    return _ir_core.create_op_call("tile.cmp", [lhs, rhs], kwargs, actual_span)


def cmps(
    lhs: Expr,
    rhs: int | float | Expr,
    cmp_type: int = 0,
    span: Span | None = None,
) -> Call:
    """Element-wise comparison of tile and scalar (returns a packed predicate mask tile).

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        cmp_type: Comparison type (int):
                  EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5
                  Default: 0 (EQ)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for a packed predicate mask tile.
        Use tile.sel with an explicit tmp tile to materialize values.
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_scalar_operand(lhs, rhs, actual_span)
    kwargs: dict[str, Any] = {"cmp_type": cmp_type}
    return _ir_core.create_op_call("tile.cmps", [lhs, rhs_expr], kwargs, actual_span)


# ============================================================================
# Unary Operations
# ============================================================================


def neg(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise negation of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise negation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.neg", [tile], {}, actual_span)


def exp(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise exponential function of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise exponential
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.exp", [tile], {}, actual_span)


def sin(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise sine of a tile (radians).

    FP32-only: non-FP32 inputs are rejected. Cast explicitly via
    ``pl.cast(tile, pl.FP32)`` before applying.

    Args:
        tile: Input tile (TileType, FP32)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise sine
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.sin", [tile], {}, actual_span)


def cos(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise cosine of a tile (radians).

    FP32-only: non-FP32 inputs are rejected. Cast explicitly via
    ``pl.cast(tile, pl.FP32)`` before applying.

    Args:
        tile: Input tile (TileType, FP32)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise cosine
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.cos", [tile], {}, actual_span)


def recip(tile: Expr, span: Span | None = None, *, high_precision: bool = False) -> Call:
    """Element-wise reciprocal (1/x) of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)
        high_precision: Whether to select PTOAS's high-precision reciprocal mode (FP16/FP32 only)

    Returns:
        Call expression for element-wise reciprocal
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"high_precision": True} if high_precision else {}
    return _ir_core.create_op_call("tile.recip", [tile], kwargs, actual_span)


def sqrt(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise square root of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise square root
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.sqrt", [tile], {}, actual_span)


def rsqrt(tile: Expr, tmp: Expr | None = None, span: Span | None = None) -> Call:
    """Element-wise reciprocal square root (1/sqrt(x)) of a tile.

    Args:
        tile: Input tile (TileType)
        tmp: Optional scratch tile (TileType, same shape/dtype as ``tile``).
            Passing it selects the high-precision PTO lowering.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise reciprocal square root
    """
    actual_span = _get_span_or_capture(span)
    args: list[Expr] = [tile] if tmp is None else [tile, tmp]
    return _ir_core.create_op_call("tile.rsqrt", args, {}, actual_span)


def cast(
    tile: Expr,
    target_type: int | DataType,
    mode: str | int = "round",
    span: Span | None = None,
    *,
    tmp: Expr | None = None,
    saturation_mode: str | int | None = None,
) -> Call:
    """Cast tile to target data type (element-wise).

    Args:
        tile: Input tile (TileType)
        target_type: Target data type (DataType)
        mode: Rounding mode — string name ("none", "rint", "round", "floor",
              "ceil", "trunc", "odd") or int (0–6)
        span: Optional source span for debugging (auto-captured if not provided)
        tmp: Optional A2/A3 PTOAS scratch tile for a non-saturating narrowing
             tcvt. Normally compiler-generated, and only for a cast that opted
             out of saturation — the saturating form is native and reads none.
        saturation_mode: Destination saturation — "on" (1) clamps out-of-range
             results to the destination range, "off" (0) selects the target's
             non-saturating conversion. ``None`` takes the destination's own
             default — ``DEFAULT_SATURATION_MODE`` ("on") for an integer
             destination, the target's own behavior for a float one. Only a
             deviation from that default is recorded on the call.

    Returns:
        Call expression for element-wise cast to target dtype

    Example:
        >>> tile_bf16 = ...  # TileType with BF16 dtype
        >>> tile_fp32 = tile.cast(tile_bf16, DataType.FP32)
    """
    mode_val = resolve_cast_mode(mode)

    actual_span = _get_span_or_capture(span)
    deviation = resolve_saturation_deviation(saturation_mode, target_type)
    kwargs: dict[str, Any] = {"target_type": target_type, "mode": mode_val}
    if deviation is not None:
        kwargs["saturation_mode"] = deviation
    args: list[Expr] = [tile] if tmp is None else [tile, tmp]
    return _ir_core.create_op_call("tile.cast", args, kwargs, actual_span)


def log(tile: Expr, span: Span | None = None, *, high_precision: bool = False) -> Call:
    """Element-wise natural logarithm of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)
        high_precision: Whether to select PTOAS's high-precision logarithm mode

    Returns:
        Call expression for element-wise natural logarithm
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"high_precision": True} if high_precision else {}
    return _ir_core.create_op_call("tile.log", [tile], kwargs, actual_span)


def abs(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise absolute value of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise absolute value
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.abs", [tile], {}, actual_span)


def relu(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise ReLU activation function (max(0, x)) of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise ReLU activation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.relu", [tile], {}, actual_span)


def not_(tile: Expr, span: Span | None = None) -> Call:
    """Element-wise bitwise NOT of a tile.

    Computes ~tile element-wise. Maps to the TNOT hardware intrinsic.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise bitwise NOT
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.not", [tile], {}, actual_span)


def maximum(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Element-wise maximum of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise maximum
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.maximum", [lhs, rhs], {}, actual_span)


def minimum(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Element-wise minimum of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise minimum
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.minimum", [lhs, rhs], {}, actual_span)


def maximums(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise maximum of tile and scalar.

    Computes max(lhs, rhs) element-wise. Maps to the TMAXS hardware intrinsic.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise maximum with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.maximums", [lhs, rhs_expr], {}, actual_span)


def minimums(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise minimum of tile and scalar.

    Computes min(lhs, rhs) element-wise. Maps to the TMINS hardware intrinsic.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise minimum with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = _normalize_scalar_operand(lhs, rhs, actual_span)
    return _ir_core.create_op_call("tile.minimums", [lhs, rhs_expr], {}, actual_span)
