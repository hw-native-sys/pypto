# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile element-wise arithmetic, comparisons, and scalar dispatch."""

from pypto.ir.op import tile_ops as _ir_ops
from pypto.ir.utils import (
    _get_span_or_capture,
)
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import (
    Expr,
)

from ..typing import Scalar, Tile


def _unwrap_rhs(rhs: int | float | Expr | Tile | Scalar) -> int | float | Expr:
    """Unwrap rhs operands into the IR-layer representation."""
    return rhs.unwrap() if isinstance(rhs, (Tile, Scalar)) else rhs


def _scalar_operand_to_expr(value: int | Scalar | Expr) -> Expr:
    """Coerce a scalar-binary operand to an ``Expr``.

    Used by ``min`` / ``max``: ``Scalar`` is unwrapped to its inner ``Expr``, raw
    ``Expr`` is forwarded as-is, and a bare ``int`` is materialized as
    ``ConstInt(.., INDEX)`` with the span pinned by the parser if any (so error
    messages and dumps point at the user's source line, not this wrapper).
    ``INDEX`` matches the dtype the parser uses for plain int literals, so
    round-tripped programs don't sprout spurious casts on otherwise-equivalent
    constants.
    """
    if isinstance(value, Scalar):
        return value.unwrap()
    if isinstance(value, Expr):
        return value
    return _ir_core.ConstInt(value, DataType.INDEX, _get_span_or_capture())


def add(lhs: Tile, rhs: Tile | int | float | Scalar | Expr) -> Tile:
    """Element-wise addition of tile and tile or scalar.

    Supports broadcasting when both operands are tiles. A scalar ``rhs``
    canonicalizes to ``tile.adds``.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile or scalar

    Returns:
        Tile wrapping the add operation
    """
    call_expr = _ir_ops.add(lhs.unwrap(), _unwrap_rhs(rhs))
    return Tile(expr=call_expr)


def sub(lhs: Tile, rhs: Tile | int | float | Scalar | Expr) -> Tile:
    """Element-wise subtraction of tile and tile or scalar.

    Supports broadcasting when both operands are tiles. A scalar ``rhs``
    canonicalizes to ``tile.subs``.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile or scalar

    Returns:
        Tile wrapping the sub operation
    """
    call_expr = _ir_ops.sub(lhs.unwrap(), _unwrap_rhs(rhs))
    return Tile(expr=call_expr)


def mul(lhs: Tile, rhs: Tile | int | float | Scalar | Expr) -> Tile:
    """Element-wise multiplication of tile and tile or scalar.

    Supports broadcasting when both operands are tiles. A scalar ``rhs``
    canonicalizes to ``tile.muls``.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile or scalar

    Returns:
        Tile wrapping the mul operation
    """
    call_expr = _ir_ops.mul(lhs.unwrap(), _unwrap_rhs(rhs))
    return Tile(expr=call_expr)


def div(
    lhs: Tile,
    rhs: Tile | int | float | Scalar | Expr,
    high_precision: bool = False,
) -> Tile:
    """Element-wise division of tile and tile or scalar.

    Tile-tile division requires identical physical and valid shapes. A scalar
    ``rhs`` canonicalizes to ``tile.divs``, which does not expose the ``tdiv``
    precision mode — hence ``high_precision`` applies only to the tile-tile form.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile or scalar
        high_precision: Whether to select PTOAS's high-precision division mode.
            Honoured only on A5: A2/A3 PTO-ISA ignores it and returns the default
            result, which compilation warns about.
            Only available when ``rhs`` is a Tile.

    Returns:
        Tile wrapping the div operation
    """
    call_expr = _ir_ops.div(lhs.unwrap(), _unwrap_rhs(rhs), high_precision=high_precision)
    return Tile(expr=call_expr)


def adds(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    """Element-wise addition of tile and scalar.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the adds operation
    """
    call_expr = _ir_ops.adds(lhs.unwrap(), _unwrap_rhs(rhs))
    return Tile(expr=call_expr)


def subs(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    """Element-wise subtraction of tile and scalar.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the subs operation
    """
    call_expr = _ir_ops.subs(lhs.unwrap(), _unwrap_rhs(rhs))
    return Tile(expr=call_expr)


def muls(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    """Element-wise multiplication of tile and scalar.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the muls operation
    """
    call_expr = _ir_ops.muls(lhs.unwrap(), _unwrap_rhs(rhs))
    return Tile(expr=call_expr)


def divs(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    """Element-wise division of tile and scalar.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the divs operation
    """
    call_expr = _ir_ops.divs(lhs.unwrap(), _unwrap_rhs(rhs))
    return Tile(expr=call_expr)


def neg(tile: Tile) -> Tile:
    """Element-wise negation.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the neg operation
    """
    call_expr = _ir_ops.neg(tile.unwrap())
    return Tile(expr=call_expr)


def exp(tile: Tile) -> Tile:
    """Element-wise exponential.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the exp operation
    """
    call_expr = _ir_ops.exp(tile.unwrap())
    return Tile(expr=call_expr)


def sin(tile: Tile) -> Tile:
    """Element-wise sine of a tile (radians). FP32 only.

    Non-FP32 inputs are rejected rather than promoted — cast explicitly with
    ``pl.cast(tile, pl.FP32)`` first.

    Args:
        tile: Input tile (FP32)

    Returns:
        Tile wrapping the sin operation
    """
    call_expr = _ir_ops.sin(tile.unwrap())
    return Tile(expr=call_expr)


def cos(tile: Tile) -> Tile:
    """Element-wise cosine of a tile (radians). FP32 only.

    Non-FP32 inputs are rejected rather than promoted — cast explicitly with
    ``pl.cast(tile, pl.FP32)`` first.

    Args:
        tile: Input tile (FP32)

    Returns:
        Tile wrapping the cos operation
    """
    call_expr = _ir_ops.cos(tile.unwrap())
    return Tile(expr=call_expr)


def sqrt(tile: Tile) -> Tile:
    """Element-wise square root.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the sqrt operation
    """
    call_expr = _ir_ops.sqrt(tile.unwrap())
    return Tile(expr=call_expr)


def rsqrt(tile: Tile, tmp: Tile | None = None) -> Tile:
    """Element-wise reciprocal square root.

    Args:
        tile: Input tile
        tmp: Optional scratch tile (same shape/dtype as ``tile``) that activates
            the high-precision PTO lowering.

    Returns:
        Tile wrapping the rsqrt operation
    """
    tmp_expr = tmp.unwrap() if tmp is not None else None
    call_expr = _ir_ops.rsqrt(tile.unwrap(), tmp_expr)
    return Tile(expr=call_expr)


def recip(tile: Tile, high_precision: bool = False) -> Tile:
    """Element-wise reciprocal.

    Args:
        tile: Input tile
        high_precision: Whether to select PTOAS's high-precision reciprocal mode
            (FP16/FP32 only). Honoured only on A5: A2/A3 PTO-ISA ignores it and
            returns the default result, which compilation warns about.

    Returns:
        Tile wrapping the recip operation
    """
    call_expr = _ir_ops.recip(tile.unwrap(), high_precision=high_precision)
    return Tile(expr=call_expr)


def log(tile: Tile, high_precision: bool = False) -> Tile:
    """Element-wise natural logarithm.

    Args:
        tile: Input tile
        high_precision: Whether to select PTOAS's high-precision logarithm mode.
            Honoured only on A5: A2/A3 PTO-ISA ignores it and returns the default
            result, which compilation warns about.

    Returns:
        Tile wrapping the log operation
    """
    call_expr = _ir_ops.log(tile.unwrap(), high_precision=high_precision)
    return Tile(expr=call_expr)


def abs(tile: Tile) -> Tile:
    """Element-wise absolute value.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the abs operation
    """
    call_expr = _ir_ops.abs(tile.unwrap())
    return Tile(expr=call_expr)


def relu(tile: Tile) -> Tile:
    """Element-wise ReLU activation (max(0, x)).

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the relu operation
    """
    call_expr = _ir_ops.relu(tile.unwrap())
    return Tile(expr=call_expr)


def cast(
    tile: Tile,
    target_type: int | DataType,
    mode: str | int = "round",
    *,
    tmp: Tile | None = None,
    saturation_mode: str | int | None = None,
) -> Tile:
    """Cast tile to target data type (element-wise).

    Args:
        tile: Input tile (TileType)
        target_type: Target data type (DataType)
        mode: Rounding mode — string name ("none", "rint", "round", "floor",
              "ceil", "trunc", "odd") or int (0–6)
        tmp: Optional A2/A3 PTOAS scratch tile. Normally compiler-generated,
             and only for a cast that opted out of saturation — the saturating
             form is native and reads none.
        saturation_mode: Destination saturation — ``"on"`` (1) clamps a
             rounded value that falls outside the destination range to that
             range; ``"off"`` (0) keeps the target's non-saturating
             conversion, including its overflow and non-finite behavior.
             **Defaults to** ``"on"`` **for an integer destination**:
             nothing standard fixes what an overflowing conversion to an integer
             produces, clamping is the safer of the two to get by accident, and
             it is what the hardware converts natively. A float destination keeps
             the target's own IEEE behavior (an out-of-range narrowing yields an
             infinity) unless you ask otherwise. When the cast lowers to a chain
             of native conversions, the mode applies to the final hop. On A2/A3 a
             saturating narrowing cast needs no ``tmp``, so the compiler
             generates none; a caller-supplied ``tmp`` is still honored.

    Returns:
        Tile wrapping the cast operation

    Example:
        >>> tile_fp32 = pl.tile.cast(tile_bf16, pl.FP32)
    """
    tmp_expr = None if tmp is None else tmp.unwrap()
    call_expr = _ir_ops.cast(tile.unwrap(), target_type, mode, tmp=tmp_expr, saturation_mode=saturation_mode)
    return Tile(expr=call_expr)


def maximum(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise maximum of two tiles.

    Supports broadcasting between the two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the maximum operation
    """
    call_expr = _ir_ops.maximum(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def minimum(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise minimum of two tiles.

    Supports broadcasting between the two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the minimum operation
    """
    call_expr = _ir_ops.minimum(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def cmp(lhs: Tile, rhs: Tile, cmp_type: int = 0) -> Tile:
    """Element-wise comparison of two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile
        cmp_type: Comparison type (EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5)

    Returns:
        Tile wrapping a packed predicate mask. Use tile.sel with an explicit tmp tile to materialize values.
    """
    call_expr = _ir_ops.cmp(lhs.unwrap(), rhs.unwrap(), cmp_type)
    return Tile(expr=call_expr)


def cmps(lhs: Tile, rhs: int | float | Expr | Scalar, cmp_type: int = 0) -> Tile:
    """Element-wise comparison of tile and scalar.

    Args:
        lhs: Tile
        rhs: Scalar value
        cmp_type: Comparison type (EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5)

    Returns:
        Tile wrapping a packed predicate mask. Use tile.sel with an explicit tmp tile to materialize values.
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.cmps(lhs.unwrap(), rhs_expr, cmp_type)
    return Tile(expr=call_expr)


def max(lhs: Scalar | int | Expr, rhs: Scalar | int | Expr) -> Scalar:
    """Scalar max of two values.

    Tile reductions are direction-specific — use [`row_max`][pypto.language.tile.row_max] (collapses the
    last axis) or [`col_max`][pypto.language.tile.col_max] (collapses axis 0).

    Args:
        lhs: First scalar operand
        rhs: Second scalar operand

    Returns:
        Scalar wrapping the max operation
    """
    return Scalar(expr=_ir_core.max_(_scalar_operand_to_expr(lhs), _scalar_operand_to_expr(rhs)))


def min(lhs: Scalar | int | Expr, rhs: Scalar | int | Expr) -> Scalar:
    """Scalar min of two values.

    Tile reductions are direction-specific — use [`row_min`][pypto.language.tile.row_min] (collapses the
    last axis) or [`col_min`][pypto.language.tile.col_min] (collapses axis 0).

    Args:
        lhs: First scalar operand
        rhs: Second scalar operand

    Returns:
        Scalar wrapping the min operation
    """
    return Scalar(expr=_ir_core.min_(_scalar_operand_to_expr(lhs), _scalar_operand_to_expr(rhs)))


def rem(lhs: Tile, rhs: Tile, tmp: Tile, high_precision: bool = False) -> Tile:
    """Element-wise remainder (modulo) of two tiles.

    Computes lhs % rhs element-wise. Maps to the TREM hardware intrinsic.
    On A2/A3, every INT32 input element must be in ``[-2**24, 2**24]``.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile
        tmp: Same-dtype scratch tile whose physical and valid capacity
            provably provides two rows and covers the dividend columns on
            A2/A3. It must not overlap either source there.
        high_precision: Whether to select PTOAS's high-precision TREM mode.
            Honoured only on A5: A2/A3 PTO-ISA ignores it and returns the default
            result, which compilation warns about.
            This mode is defined only for FP32 and is ignored by A2/A3 hardware.

    Returns:
        Tile wrapping the rem operation
    """
    call_expr = _ir_ops.rem(lhs.unwrap(), rhs.unwrap(), tmp.unwrap(), high_precision=high_precision)
    return Tile(expr=call_expr)


def rems(lhs: Tile, rhs: int | float | Expr | Scalar, tmp: Tile) -> Tile:
    """Element-wise remainder (modulo) of tile and scalar.

    Computes lhs % rhs element-wise. Maps to the TREMS hardware intrinsic.
    On A2/A3, source valid extents must be provably positive and every INT32
    source element and scalar must be in ``[-2**24, 2**24]``.

    Args:
        lhs: Tile
        rhs: Scalar value
        tmp: Same-dtype scratch tile whose physical and valid capacity
            provably provides one row and covers the dividend columns on
            A2/A3. It must not overlap the dividend there.

    Returns:
        Tile wrapping the rems operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.rems(lhs.unwrap(), rhs_expr, tmp.unwrap())
    return Tile(expr=call_expr)


def part_add(src0: Tile, src1: Tile) -> Tile:
    """Partial element-wise add of two tiles.

    Adds over the destination valid region; where only one source is valid the
    result copies that source. Maps to the TPARTADD hardware intrinsic.

    Args:
        src0: First source tile
        src1: Second source tile

    Returns:
        Tile wrapping the part_add operation
    """
    call_expr = _ir_ops.part_add(src0.unwrap(), src1.unwrap())
    return Tile(expr=call_expr)


def part_mul(src0: Tile, src1: Tile) -> Tile:
    """Partial element-wise multiply of two tiles.

    Multiplies over the destination valid region; where only one source is valid
    the result copies that source. Maps to the TPARTMUL hardware intrinsic.

    Args:
        src0: First source tile
        src1: Second source tile

    Returns:
        Tile wrapping the part_mul operation
    """
    call_expr = _ir_ops.part_mul(src0.unwrap(), src1.unwrap())
    return Tile(expr=call_expr)


def part_max(src0: Tile, src1: Tile) -> Tile:
    """Partial element-wise max of two tiles.

    Takes the max over the destination valid region; where only one source is
    valid the result copies that source. Maps to the TPARTMAX hardware intrinsic.

    Args:
        src0: First source tile
        src1: Second source tile

    Returns:
        Tile wrapping the part_max operation
    """
    call_expr = _ir_ops.part_max(src0.unwrap(), src1.unwrap())
    return Tile(expr=call_expr)


def part_min(src0: Tile, src1: Tile) -> Tile:
    """Partial element-wise min of two tiles.

    Takes the min over the destination valid region; where only one source is
    valid the result copies that source. Maps to the TPARTMIN hardware intrinsic.

    Args:
        src0: First source tile
        src1: Second source tile

    Returns:
        Tile wrapping the part_min operation
    """
    call_expr = _ir_ops.part_min(src0.unwrap(), src1.unwrap())
    return Tile(expr=call_expr)


def fmod(lhs: Tile, rhs: Tile, high_precision: bool = False) -> Tile:
    """Element-wise truncating remainder of two tiles.

    Computes the truncating remainder of lhs / rhs element-wise, matching
    ``torch.fmod`` with the result taking the dividend sign. Maps to TFMOD.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile
        high_precision: Whether to select PTOAS's high-precision TFMOD mode.
            Honoured only on A5: A2/A3 PTO-ISA ignores it and returns the default
            result, which compilation warns about.
            This mode is defined only for FP32.

    Returns:
        Tile wrapping the fmod operation
    """
    call_expr = _ir_ops.fmod(lhs.unwrap(), rhs.unwrap(), high_precision=high_precision)
    return Tile(expr=call_expr)


def fmods(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    """Element-wise truncating remainder of tile and scalar.

    Computes the truncating remainder of lhs / rhs element-wise, matching
    ``torch.fmod`` with the result taking the dividend sign. Maps to TFMODS.
    A2/A3 requires every source valid extent to be provably positive.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the fmods operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.fmods(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def and_(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise bitwise AND of two tiles.

    Computes lhs & rhs element-wise. Maps to the TAND hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the and operation
    """
    call_expr = _ir_ops.and_(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def ands(lhs: Tile, rhs: int | Expr | Scalar) -> Tile:
    """Element-wise bitwise AND of tile and scalar.

    Computes lhs & rhs element-wise. Maps to the TANDS hardware intrinsic.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the ands operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.ands(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def or_(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise bitwise OR of two tiles.

    Computes lhs | rhs element-wise. Maps to the TOR hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the or operation
    """
    call_expr = _ir_ops.or_(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def ors(lhs: Tile, rhs: int | Expr | Scalar) -> Tile:
    """Element-wise bitwise OR of tile and scalar.

    Computes lhs | rhs element-wise. Maps to the TORS hardware intrinsic.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the ors operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.ors(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def xor(lhs: Tile, rhs: Tile, tmp: Tile) -> Tile:
    """Element-wise bitwise XOR of two tiles.

    Computes lhs ^ rhs element-wise. Maps to the TXOR hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile
        tmp: Temporary tile required by the hardware

    Returns:
        Tile wrapping the xor operation
    """
    call_expr = _ir_ops.xor(lhs.unwrap(), rhs.unwrap(), tmp.unwrap())
    return Tile(expr=call_expr)


def xors(lhs: Tile, rhs: int | Expr | Scalar, tmp: Tile) -> Tile:
    """Element-wise bitwise XOR of tile and scalar.

    Computes lhs ^ rhs element-wise. Maps to the TXORS hardware intrinsic.

    Args:
        lhs: Tile
        rhs: Scalar value
        tmp: Temporary tile required by the hardware

    Returns:
        Tile wrapping the xors operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.xors(lhs.unwrap(), rhs_expr, tmp.unwrap())
    return Tile(expr=call_expr)


def shl(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise bitwise left shift of two tiles.

    Computes lhs << rhs element-wise. Maps to the TSHL hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the shl operation
    """
    call_expr = _ir_ops.shl(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def shls(lhs: Tile, rhs: int | Expr | Scalar) -> Tile:
    """Element-wise bitwise left shift of tile and scalar.

    Computes lhs << rhs element-wise. Maps to the TSHLS hardware intrinsic.

    Note:
        The scalar shift amount must be zero or positive; negative values are
        not supported by the hardware and will be rejected by codegen.

    Args:
        lhs: Tile
        rhs: Scalar shift amount; must be >= 0

    Returns:
        Tile wrapping the shls operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.shls(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def shr(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise bitwise right shift of two tiles.

    Computes lhs >> rhs element-wise. Maps to the TSHR hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the shr operation
    """
    call_expr = _ir_ops.shr(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def shrs(lhs: Tile, rhs: int | Expr | Scalar) -> Tile:
    """Element-wise bitwise right shift of tile and scalar.

    Computes lhs >> rhs element-wise. Maps to the TSHRS hardware intrinsic.

    Note:
        The scalar shift amount must be zero or positive; negative values are
        not supported by the hardware and will be rejected by codegen.

    Args:
        lhs: Tile
        rhs: Scalar shift amount; must be >= 0

    Returns:
        Tile wrapping the shrs operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.shrs(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def maximums(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    """Element-wise maximum of tile and scalar.

    Computes max(lhs, rhs) element-wise. Maps to the TMAXS hardware intrinsic.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the maximums operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.maximums(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def minimums(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    """Element-wise minimum of tile and scalar.

    Computes min(lhs, rhs) element-wise. Maps to the TMINS hardware intrinsic.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the minimums operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.minimums(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def prelu(tile: Tile, slope: Tile, tmp: Tile) -> Tile:
    """Element-wise parametric ReLU of a tile.

    Computes prelu(tile, slope) element-wise. Maps to the TPRELU hardware intrinsic.

    Args:
        tile: Input tile
        slope: Slope tile used for negative values
        tmp: Temporary tile required by the hardware

    Returns:
        Tile wrapping the prelu operation
    """
    call_expr = _ir_ops.prelu(tile.unwrap(), slope.unwrap(), tmp.unwrap())
    return Tile(expr=call_expr)


def not_(tile: Tile) -> Tile:
    """Element-wise bitwise NOT of a tile.

    Computes ~tile element-wise. Maps to the TNOT hardware intrinsic.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the not operation
    """
    call_expr = _ir_ops.not_(tile.unwrap())
    return Tile(expr=call_expr)


def addc(lhs: Tile, rhs: Tile, rhs2: Tile) -> Tile:
    """Element-wise carry addition of three tiles.

    Computes ``src0 + src1 + carry`` element-wise. Maps to TADDC.

    Args:
        lhs: First source tile
        rhs: Second source tile
        rhs2: Per-element carry-in tile, normally containing 0 or 1

    Returns:
        Tile wrapping the addc operation
    """
    call_expr = _ir_ops.addc(lhs.unwrap(), rhs.unwrap(), rhs2.unwrap())
    return Tile(expr=call_expr)


def subc(lhs: Tile, rhs: Tile, rhs2: Tile) -> Tile:
    """Element-wise carry subtraction of three tiles.

    Computes ``src0 - src1 + carry`` element-wise. Maps to TSUBC.

    Args:
        lhs: Minuend tile
        rhs: Subtrahend tile
        rhs2: Per-element carry-in tile, normally containing 0 or 1

    Returns:
        Tile wrapping the subc operation
    """
    call_expr = _ir_ops.subc(lhs.unwrap(), rhs.unwrap(), rhs2.unwrap())
    return Tile(expr=call_expr)


def addsc(lhs: Tile, rhs: int | float | Expr | Scalar, rhs2: Tile) -> Tile:
    """Element-wise scalar carry addition.

    Computes ``src0 + scalar + carry`` element-wise. Maps to TADDSC.

    Args:
        lhs: Source tile
        rhs: Scalar addend with the same dtype as ``lhs``
        rhs2: Per-element carry-in tile, normally containing 0 or 1

    Returns:
        Tile wrapping the addsc operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.addsc(lhs.unwrap(), rhs_expr, rhs2.unwrap())
    return Tile(expr=call_expr)


def subsc(lhs: Tile, rhs: int | float | Expr | Scalar, rhs2: Tile) -> Tile:
    """Element-wise scalar carry subtraction.

    Computes ``src0 - scalar + carry`` element-wise. Maps to TSUBSC.

    Args:
        lhs: Minuend tile
        rhs: Scalar subtrahend with the same dtype as ``lhs``
        rhs2: Per-element carry-in tile, normally containing 0 or 1

    Returns:
        Tile wrapping the subsc operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.subsc(lhs.unwrap(), rhs_expr, rhs2.unwrap())
    return Tile(expr=call_expr)


def lrelu(tile: Tile, slope: int | float | Expr | Scalar) -> Tile:
    """Element-wise leaky ReLU with scalar slope.

    Computes max(tile, slope * tile) element-wise. Maps to the TLRELU hardware intrinsic.

    Args:
        tile: Input tile
        slope: Scalar slope for negative values

    Returns:
        Tile wrapping the lrelu operation
    """
    slope_expr = slope.unwrap() if isinstance(slope, Scalar) else slope
    call_expr = _ir_ops.lrelu(tile.unwrap(), slope_expr)
    return Tile(expr=call_expr)


def sel(mask: Tile, lhs: Tile, rhs: Tile, tmp: Tile) -> Tile:
    """Per-element selection between two tiles using a predicate mask tile.

    For each element (i, j): dst[i,j] = lhs[i,j] if mask[i,j] is true, else rhs[i,j].
    Maps to the TSEL hardware intrinsic. The mask encoding is target-defined.

    Args:
        mask: Predicate mask tile; encoding is target-defined
        lhs: Source tile 0, selected where mask is true
        rhs: Source tile 1, selected where mask is false
        tmp: Scratch tile required by TSEL (UINT32 [1, 16] on A2/A3; unread ABI placeholder on A5)

    Returns:
        Tile wrapping the sel operation
    """
    call_expr = _ir_ops.sel(mask.unwrap(), lhs.unwrap(), rhs.unwrap(), tmp.unwrap())
    return Tile(expr=call_expr)


def sels(mask: Tile, src: Tile, tmp: Tile, scalar: int | float | Expr | Scalar) -> Tile:
    """Per-element selection between a source tile and a scalar.

    For each element (i, j): dst[i,j] = src[i,j] if mask[i,j] is true,
    else scalar. Maps to the TSELS hardware intrinsic.

    Args:
        mask: Predicate mask tile; encoding is target-defined
        src: Source tile, selected where mask is true
        tmp: Scratch tile required by TSELS
        scalar: Scalar value, selected where mask is false

    Returns:
        Tile wrapping the sels operation
    """
    scalar_expr = scalar.unwrap() if isinstance(scalar, Scalar) else scalar
    call_expr = _ir_ops.sels(mask.unwrap(), src.unwrap(), tmp.unwrap(), scalar_expr)
    return Tile(expr=call_expr)


def select(
    cond: Tile,
    on_true: Tile | int | float | Expr | Scalar,
    on_false: Tile | int | float | Expr | Scalar,
) -> Tile:
    """Per-element value selection: ``out[i] = cond[i] ? on_true[i] : on_false[i]``.

    The scratch-free counterpart of :func:`sel` / :func:`sels`: the mask geometry
    and the architecture's scratch tile are derived during lowering, so neither
    appears in the call.

    Both branches are evaluated. This selects values -- it is not a branch, and
    it applies no memory-access or tail masking of its own.

    Two Tile branches must share a shape, valid extents, and dtype; the lowered TSEL reads both sources
    element-wise, does not broadcast, and has one element type for both sources
    and the result. A scalar branch must be a compile-time
    constant and requires a statically shaped result, because the lowering
    materializes it with ``tile.full``.

    Example — clamp, composing two selects::

        hi_clamped = pl.tile.select(pl.tile.cmps(v, upper, cmp_type=3), v, upper)   # v <= upper
        clamped = pl.tile.select(pl.tile.cmps(v, lower, cmp_type=5), hi_clamped, lower)  # v >= lower

    Args:
        cond: The packed predicate mask returned by :func:`cmp` / :func:`cmps`
            for this result's geometry. For a 0/1 value tile, compare it first:
            ``pl.tile.cmps(value, 0, cmp_type=1)``
        on_true: Value selected where cond is true; a Tile or a constant scalar
        on_false: Value selected where cond is false; same forms as on_true

    Returns:
        Tile wrapping the select operation

    Raises:
        ValueError: If cond is not this result's packed predicate mask, if both
            on_true and on_false are scalars, if two Tile branches disagree on
            shape, valid extents, or dtype, or if a scalar branch is not a
            compile-time constant
    """

    def _unwrap(value: Tile | int | float | Expr | Scalar) -> int | float | Expr:
        return value.unwrap() if isinstance(value, (Tile, Scalar)) else value

    call_expr = _ir_ops.select(cond.unwrap(), _unwrap(on_true), _unwrap(on_false))
    return Tile(expr=call_expr)
