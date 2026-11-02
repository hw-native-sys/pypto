# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile matrix products and MX quantization."""

from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import (
    AccPhase,
    Call,
    Expr,
    Span,
)

from ..utils import (
    _get_span_or_capture,
)

# ============================================================================
# MX Quantization Operations
# ============================================================================


def tquant_mx(
    src: Expr,
    *,
    group_axis: int,
    dtype: DataType = DataType.FP8E4M3FN,
    quant_scale_alg: str = "ocp",
    span: Span | None = None,
) -> Call:
    """MX block-32 dynamic quantization returning quantized data and scale."""
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"dtype": dtype, "group_axis": group_axis}
    if quant_scale_alg != "ocp":
        kwargs["quant_scale_alg"] = quant_scale_alg
    return _ir_core.create_op_call("tile.tquant_mx", [src], kwargs, actual_span)


def tquant_mx_raw(
    src: Expr,
    max_scratch: Expr,
    scaling_scratch: Expr,
    *,
    dtype: DataType = DataType.FP8E4M3FN,
    group_axis: int = 1,
    quant_scale_alg: str = "ocp",
    interleave: bool = False,
    span: Span | None = None,
) -> Call:
    """Build the compiler-internal value-returning raw MX quantization Call.

    Returns ``TupleType{INT8 dst, UINT8 exp}``. ``max_scratch`` / ``scaling_scratch``
    are write-only per-group workspaces (same role as ``gather_compare``'s tmp).
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"dtype": dtype, "group_axis": group_axis}
    if quant_scale_alg != "ocp":
        kwargs["quant_scale_alg"] = quant_scale_alg
    if interleave:
        kwargs["interleave"] = True
    return _ir_core.create_op_call(
        "tile.tquant_mx_raw",
        [src, max_scratch, scaling_scratch],
        kwargs,
        actual_span,
    )


def tmov_x2zz(
    src: Expr,
    tmp: Expr,
    *,
    group_axis: int = 1,
    dst_rows: int | None = None,
    dst_cols: int | None = None,
    span: Span | None = None,
) -> Call:
    """Exponent X-to-ZZ layout conversion returning a UINT8 ZZ tile.

    ``tmp`` is a write-only workspace. Axis1 requires capacity
    ``64 + ceil(rows/16) * cols`` bytes and ``dst_rows``/``dst_cols`` (ZZ ``[M,G]``)
    because TQUANT emits a legacy-flat exp ``[1, M*G]``. Axis0 requires a minimal
    32-byte-aligned Vec pad. A5-only.
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"group_axis": group_axis}
    if dst_rows is not None:
        kwargs["dst_rows"] = int(dst_rows)
    if dst_cols is not None:
        kwargs["dst_cols"] = int(dst_cols)
    return _ir_core.create_op_call(
        "tile.tmov_x2zz",
        [src, tmp],
        kwargs,
        actual_span,
    )


# ============================================================================
# Matrix Operations
# ============================================================================


def matmul(
    lhs: Expr,
    rhs: Expr,
    span: Span | None = None,
    *,
    tf32_mode: str | None = None,
    acc_phase: AccPhase = AccPhase.Unspecified,
) -> Call:
    """Matrix multiplication of two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)
        tf32_mode: Optional HF32 input rounding: "round_even" or "round_away".
            Requires 2D FP32 operands, Ascend950, and PTOAS 0.70+. None preserves FP32.
        acc_phase: Producer-side unit-flag phase. ``Final`` must be consumed
            by a store with ``STPhase.Final`` in the same straight-line region.

    Returns:
        Call expression for matrix multiplication
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, int | str] = {} if tf32_mode is None else {"tf32_mode": tf32_mode}
    # Keep the historical default IR unchanged; explicit phases are opt-in.
    phase = int(acc_phase)
    if phase != int(AccPhase.Unspecified):
        kwargs["acc_phase"] = phase
    return _ir_core.create_op_call("tile.matmul", [lhs, rhs], kwargs, actual_span)


def matmul_acc(
    acc: Expr,
    lhs: Expr,
    rhs: Expr,
    span: Span | None = None,
    *,
    init_cond: Expr | None = None,
    tf32_mode: str | None = None,
    acc_phase: AccPhase = AccPhase.Unspecified,
) -> Call:
    """Matrix multiplication with accumulation.

    Performs matrix multiplication and accumulates the result: acc = acc + lhs @ rhs.
    This is commonly used in loop-based matrix multiplication where results are
    accumulated over the K dimension.

    With ``init_cond``, the accumulator's initial value is conditional: on the
    steps where the predicate holds, ``acc`` is overwritten with ``lhs @ rhs``
    instead of accumulated into. This is the split-K ``k == 0`` idiom, and it
    keeps the accumulator single-def where a hand-written if/else would put a
    phi on an in-place Acc buffer.

    Args:
        acc: Accumulator tile (TileType) to accumulate into
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)
        init_cond: Optional BOOL scalar predicate selecting overwrite over accumulate
        tf32_mode: Optional HF32 input rounding: "round_even" or "round_away".
            Requires 2D FP32 operands, Ascend950, and PTOAS 0.70+. None preserves FP32.
        acc_phase: Producer-side unit-flag phase. ``Final`` must be consumed
            by a store with ``STPhase.Final`` in the same straight-line region.

    Returns:
        Call expression for matrix multiplication with accumulation
    """
    actual_span = _get_span_or_capture(span)
    args = [acc, lhs, rhs] if init_cond is None else [acc, lhs, rhs, init_cond]
    kwargs: dict[str, int | str] = {} if tf32_mode is None else {"tf32_mode": tf32_mode}
    # Keep the predicate as an operand, independently of the optional phase.
    phase = int(acc_phase)
    if phase != int(AccPhase.Unspecified):
        kwargs["acc_phase"] = phase
    return _ir_core.create_op_call("tile.matmul_acc", args, kwargs, actual_span)


def matmul_bias(
    lhs: Expr, rhs: Expr, bias: Expr, span: Span | None = None, *, tf32_mode: str | None = None
) -> Call:
    """Matrix multiplication with bias add: C = lhs @ rhs + bias.

    Args:
        lhs: Left-hand side tile (TileType [M, K])
        rhs: Right-hand side tile (TileType [K, N])
        bias: Bias tile (TileType [1, N]) with the accumulator dtype (FP32 for
            floating-point matrix operands, INT32 for integer matrix operands)
        span: Optional source span for debugging (auto-captured if not provided)
        tf32_mode: Optional HF32 input rounding: "round_even" or "round_away".
            Requires 2D FP32 operands, Ascend950, and PTOAS 0.70+. None preserves FP32.

    Returns:
        Call expression for matrix multiplication with bias
    """
    actual_span = _get_span_or_capture(span)
    kwargs = {} if tf32_mode is None else {"tf32_mode": tf32_mode}
    return _ir_core.create_op_call("tile.matmul_bias", [lhs, rhs, bias], kwargs, actual_span)


def matmul_mx(
    lhs: Expr,
    lhs_scale: Expr,
    rhs: Expr,
    rhs_scale: Expr,
    span: Span | None = None,
) -> Call:
    """MX block-scale matrix multiplication: C = matmul_mx(A, A_scale, B, B_scale)."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.matmul_mx", [lhs, lhs_scale, rhs, rhs_scale], {}, actual_span)


def matmul_mx_acc(
    acc: Expr,
    lhs: Expr,
    lhs_scale: Expr,
    rhs: Expr,
    rhs_scale: Expr,
    span: Span | None = None,
) -> Call:
    """MX block-scale matmul with accumulation: acc += matmul_mx(...)."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "tile.matmul_mx_acc", [acc, lhs, lhs_scale, rhs, rhs_scale], {}, actual_span
    )


def matmul_mx_bias(
    lhs: Expr,
    lhs_scale: Expr,
    rhs: Expr,
    rhs_scale: Expr,
    bias: Expr,
    span: Span | None = None,
) -> Call:
    """MX block-scale matmul with bias: C = matmul_mx(...) + bias."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "tile.matmul_mx_bias", [lhs, lhs_scale, rhs, rhs_scale, bias], {}, actual_span
    )


def tget_scale_addr(dst_scale: Expr, src: Expr, span: Span | None = None) -> Call:
    """Build the compiler-internal MX scale-address binding operation."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.tget_scale_addr", [dst_scale, src], {}, actual_span)


def batch_matmul(
    lhs: Expr,
    rhs: Expr,
    span: Span | None = None,
) -> Call:
    """Batch matrix multiplication of two tiles with broadcasting.

    For inputs with shape [...batch_dims, M, K] and [...batch_dims, K, N],
    the output has shape [...broadcast_batch_dims, M, N].

    Args:
        lhs: Left-hand side tile (TileType, at least 2D)
        rhs: Right-hand side tile (TileType, at least 2D)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for batch matrix multiplication
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.batch_matmul", [lhs, rhs], {}, actual_span)


def batch_matmul_acc(
    acc: Expr,
    lhs: Expr,
    rhs: Expr,
    span: Span | None = None,
    *,
    init_cond: Expr | None = None,
) -> Call:
    """Batch matrix multiplication with accumulation.

    Performs ``acc = acc + lhs @ rhs`` with batch-dim broadcasting between lhs and
    rhs. The broadcast batch shape must equal acc's batch shape (acc is the in-place
    accumulation target and is not broadcast).

    ``init_cond`` behaves exactly as it does on
    [`matmul_acc`][pypto.ir.op.tile_ops.matmul_acc]: where the predicate holds,
    ``acc`` is overwritten with ``lhs @ rhs`` instead of accumulated into.
    ``FlattenTileNdTo2D`` forwards it to every 2D ``tile.matmul_acc`` it unrolls
    this op into — each of those is the sole writer of its own row band of the
    accumulator, so the predicate applies band by band.

    Args:
        acc: Accumulator tile (TileType, at least 2D)
        lhs: Left-hand side tile (TileType, at least 2D)
        rhs: Right-hand side tile (TileType, at least 2D)
        span: Optional source span for debugging (auto-captured if not provided)
        init_cond: Optional BOOL scalar predicate selecting overwrite over accumulate

    Returns:
        Call expression for batch matrix multiplication with accumulation
    """
    actual_span = _get_span_or_capture(span)
    args = [acc, lhs, rhs] if init_cond is None else [acc, lhs, rhs, init_cond]
    return _ir_core.create_op_call("tile.batch_matmul_acc", args, {}, actual_span)


def gemv(
    lhs: Expr,
    rhs: Expr,
    span: Span | None = None,
    *,
    acc_phase: AccPhase = AccPhase.Unspecified,
) -> Call:
    """General Matrix-Vector multiplication: C[1,N] = A[1,K] @ B[K,N].

    ``lhs`` must have exactly one physical and logical row. The rhs logical K
    must cover the lhs logical K. Inputs must use the same INT8, FP16, BF16, or FP32
    dtype; the output is INT32 for INT8 inputs and FP32 otherwise.

    Args:
        lhs: Row vector tile (TileType [1, K])
        rhs: Right-hand side tile (TileType [K, N])
        acc_phase: Producer-side unit-flag phase. Use ``AccPhase.Partial`` for
            intermediate chunks and ``AccPhase.Final`` for the last chunk.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for GEMV
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.gemv", [lhs, rhs], {"acc_phase": int(acc_phase)}, actual_span)


def gemv_acc(
    acc: Expr,
    lhs: Expr,
    rhs: Expr,
    span: Span | None = None,
    *,
    acc_phase: AccPhase = AccPhase.Unspecified,
    init_cond: Expr | None = None,
) -> Call:
    """GEMV with accumulation: C[1,N] += A[1,K] @ B[K,N].

    ``acc`` must use the GEMV output dtype. The logical K extents and lhs/rhs
    dtype requirements are identical to :func:`gemv`.

    With ``init_cond``, the accumulator's initial value is conditional: on the
    steps where the predicate holds, ``acc`` is overwritten with ``lhs @ rhs``
    instead of accumulated into. GEMV runs on the same cube MAD as
    :func:`matmul_acc`, so it carries the same predicate; see that function for
    the split-K ``k == 0`` idiom this removes the peel from.

    Args:
        acc: Accumulator tile (TileType [1, N])
        lhs: Row vector tile (TileType [1, K])
        rhs: Right-hand side tile (TileType [K, N])
        acc_phase: Producer-side unit-flag phase. Use ``AccPhase.Partial`` for
            intermediate chunks and ``AccPhase.Final`` for the last chunk.
        span: Optional source span for debugging (auto-captured if not provided)
        init_cond: Optional BOOL scalar predicate selecting overwrite over accumulate

    Returns:
        Call expression for GEMV with accumulation
    """
    actual_span = _get_span_or_capture(span)
    args = [acc, lhs, rhs] if init_cond is None else [acc, lhs, rhs, init_cond]
    return _ir_core.create_op_call(
        "tile.gemv_acc",
        args,
        {"acc_phase": int(acc_phase)},
        actual_span,
    )


def gemv_bias(
    lhs: Expr,
    rhs: Expr,
    bias: Expr,
    span: Span | None = None,
    *,
    acc_phase: AccPhase = AccPhase.Unspecified,
) -> Call:
    """GEMV with bias add: C[1,N] = A[1,K] @ B[K,N] + bias[1,N].

    ``bias`` must use the GEMV output dtype and its valid shape must cover the
    logical output shape ``[1, N]``. The logical K extents and lhs/rhs dtype
    requirements are identical to :func:`gemv`.

    Args:
        lhs: Row vector tile (TileType [1, K])
        rhs: Right-hand side tile (TileType [K, N])
        bias: Bias tile (TileType [1, N]) with the accumulator dtype (FP32 for
            floating-point matrix operands, INT32 for integer matrix operands)
        acc_phase: Producer-side unit-flag phase. Use ``AccPhase.Partial`` for
            intermediate chunks and ``AccPhase.Final`` for the last chunk.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for GEMV with bias
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "tile.gemv_bias",
        [lhs, rhs, bias],
        {"acc_phase": int(acc_phase)},
        actual_span,
    )
