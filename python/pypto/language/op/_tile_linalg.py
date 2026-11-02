# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile matrix products and MX quantization."""

from pypto.ir.op import tile_ops as _ir_ops
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import (
    AccPhase,
)

from ..typing import BoolLike, Tile, predicate_to_expr

# ============================================================================
# MX Quantization Operations
# ============================================================================


def quant_mx(
    src: Tile,
    *,
    group_axis: int,
    dtype: DataType = DataType.FP8E4M3FN,
    quant_scale_alg: str = "ocp",
) -> tuple[Tile, Tile]:
    """MXFP8/MXFP4 block-32 dynamic quantization with PTOAS ``grpAxis`` selection.

    ``group_axis`` is required and must be ``0`` or ``1`` (same meaning as
    PTOAS ``#pto<mx_group_axis axis*>``). Axis 1 is the A-side ``[M, K]`` path;
    axis 0 is the B-side ``[N, K]`` path (LowerCompositeOps transposes to
    ``[K, N]`` before quantization). GM ``TensorLayout.MX_*`` is only for
    tensor / ``tensor.view`` annotations, not this tile op.

    ``dtype=FP4E2M1X2`` requires FP16/BF16 input and packs pairs into bytes;
    the output's last dimension is halved. FP4 axis 0 requires ``N%64==0`` for the 32-byte carrier stride.

    Args:
        src: Source tile (FP16/FP32/BF16, 2D).
            ``group_axis=1``: ``[M, K]`` with ``M%16==0``, ``K%64==0``.
            ``group_axis=0``: ``[N, K]`` with ``N%32==0``, ``K%64==0``.
        group_axis: PTOAS grouping axis — ``1`` (A-side) or ``0`` (B-side).
        dtype: ``FP8E4M3FN`` (default) or ``FP4E2M1X2``.
        quant_scale_alg: ``"ocp"`` (default) or ``"nv"``. FP16/BF16 NV requires axis 0.

    Returns:
        ``group_axis=1``: ``(quant[M,K], scale[M,K/32])`` row/row ZZ scale.
        ``group_axis=0``: ``(quant[K,N], scale[K/32,N])`` — data transposed to
        Cube RHS layout; scale col/col NN.

    Note:
        On Ascend950, ``quant_mx`` and ``matmul_mx`` may share one InCore mixed
        task. The compiler carries both generated results over V2C; the
        FP8E8M0 scale keeps its logical MX scale layout.
    """
    if group_axis not in (0, 1):
        raise ValueError(f"pl.quant_mx group_axis must be 0 or 1, but got {group_axis!r}")
    call_expr = _ir_ops.tquant_mx(
        src.unwrap(), dtype=dtype, group_axis=group_axis, quant_scale_alg=quant_scale_alg
    )
    span = call_expr.span
    return (
        Tile(expr=_ir_core.TupleGetItemExpr(call_expr, 0, span)),
        Tile(expr=_ir_core.TupleGetItemExpr(call_expr, 1, span)),
    )


def quant_mx_raw(
    src: Tile,
    max_workspace: Tile,
    scaling_workspace: Tile,
    *,
    group_axis: int,
    dtype: DataType = DataType.FP8E4M3FN,
    quant_scale_alg: str = "ocp",
    interleave: bool = False,
) -> tuple[Tile, Tile]:
    """Quantize in source orientation and return unpacked UINT8 exponent codes.

    This tile primitive exposes PTOAS TQUANT without MX scale-layout conversion.
    Axis1 accepts any positive row count and a valid column count divisible by
    32 (64 for native FP4); physical columns must equal valid columns. Axis0 groups 32 valid rows.
    FP4 output halves physical and valid columns; native FP4 additionally
    requires physical columns divisible by 64 (32 carrier bytes). Partial valid rows are allowed.

    Both workspaces use the source dtype. Axis0 workspaces have valid shape
    ``[valid_rows/32, valid_cols]`` and the same physical column stride as src.
    Axis1 may use flat valid shape ``[1, valid_rows*valid_cols/32]``; FP16/BF16
    OCP scaling capacity must round this group count up to 128 elements.
    Workspaces are write-only. FP4E2M1X2 returns native carriers; MXFP8 returns
    INT8 carrier bytes. Exponents are flat on axis1 and grouped on axis0.
    ``interleave=True`` interleaves adjacent axis0 exponent groups along columns;
    FP16/BF16 physical and valid source rows must then be divisible by 64.
    """
    call_expr = _ir_ops.tquant_mx_raw(
        src.unwrap(),
        max_workspace.unwrap(),
        scaling_workspace.unwrap(),
        dtype=dtype,
        group_axis=group_axis,
        quant_scale_alg=quant_scale_alg,
        interleave=interleave,
    )
    span = call_expr.span
    return (
        Tile(expr=_ir_core.TupleGetItemExpr(call_expr, 0, span)),
        Tile(expr=_ir_core.TupleGetItemExpr(call_expr, 1, span)),
    )


def tmov_x2zz(
    src: Tile,
    tmp: Tile,
    *,
    group_axis: int = 1,
    dst_rows: int | None = None,
    dst_cols: int | None = None,
) -> Tile:
    """Exponent X-to-ZZ layout conversion (A5-only).

    ``tmp`` is a write-only workspace. Axis1 requires capacity
    ``64 + ceil(rows/16) * cols`` bytes (typically 32-byte-aligned by the
    caller) and ``dst_rows``/``dst_cols`` for the ZZ ``[M, G]`` shape. Axis0
    (TMovDnTo2Zz) still requires a Vec tmp operand; use a minimal 32-byte-aligned
    pad.
    """
    if group_axis not in (0, 1):
        raise ValueError(f"pl.tmov_x2zz group_axis must be 0 or 1, but got {group_axis!r}")
    call_expr = _ir_ops.tmov_x2zz(
        src.unwrap(),
        tmp.unwrap(),
        group_axis=group_axis,
        dst_rows=dst_rows,
        dst_cols=dst_cols,
    )
    return Tile(expr=call_expr)


# ============================================================================
# Matrix Operations
# ============================================================================


def matmul(
    lhs: Tile,
    rhs: Tile,
    *,
    tf32_mode: str | None = None,
    acc_phase: AccPhase = AccPhase.Unspecified,
) -> Tile:
    """Matrix multiplication of two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile
        tf32_mode: Optional HF32 input rounding: "round_even" or "round_away".
            Requires 2D FP32 operands, Ascend950, and PTOAS 0.70+. None preserves FP32.
        acc_phase: Producer-side unit-flag phase. Use ``pl.AccPhase.Partial``
            for intermediate K chunks and ``pl.AccPhase.Final`` for the last
            chunk, paired with ``pl.store(..., st_phase=pl.STPhase.Final)``.

    Returns:
        Tile wrapping the matmul operation
    """
    call_expr = _ir_ops.matmul(lhs.unwrap(), rhs.unwrap(), tf32_mode=tf32_mode, acc_phase=acc_phase)
    return Tile(expr=call_expr)


def batch_matmul(lhs: Tile, rhs: Tile) -> Tile:
    """Batch matrix multiplication of two tiles.

    Broadcasts the batch dims: for inputs shaped ``[...batch_dims, M, K]`` and
    ``[...batch_dims, K, N]``, the output is ``[...broadcast_batch_dims, M, N]``.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the batch_matmul operation
    """
    call_expr = _ir_ops.batch_matmul(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def matmul_acc(
    acc: Tile,
    lhs: Tile,
    rhs: Tile,
    init_cond: BoolLike | None = None,
    *,
    tf32_mode: str | None = None,
    acc_phase: AccPhase = AccPhase.Unspecified,
) -> Tile:
    """Matrix multiplication with accumulation: acc += lhs @ rhs.

    ``init_cond`` makes the accumulator's initial value conditional: on the steps
    where it holds, ``acc`` is overwritten with ``lhs @ rhs`` rather than
    accumulated into. This is the split-K idiom, and it removes the need to zero
    the accumulator or to peel the first K step::

        for n0 in pl.range(N // N_TILE):
            for k0 in pl.pipeline(0, K, K_TILE, stage=2):
                acc_n = pl.tile.slice(acc, [M, N_TILE], [0, n0 * N_TILE])
                pl.tile.matmul_acc(acc_n, a, b, init_cond=(k0 == 0))

    A literal ``True`` / ``False`` selects one form at compile time; a runtime
    predicate lowers to a branch over the two, with no phi on the accumulator.

    Note:
        When ``acc`` is a window of a larger ``Mem.Acc`` (L0C) tile, slice it
        along **columns**, as above -- the window must span every row of its
        parent, or be at most 16 columns wide inside a single 16-column block.
        L0C stores 16x16 blocks column-major, so a *row* window of a parent with
        more than one block column is strided, and the MAD writes its result
        compactly from a bare pointer with no destination stride. Slicing
        ``[ROW_TILE, N]`` at ``[t0, 0]`` out of an ``[M, N]`` accumulator is
        therefore rejected by ``CanonicalizeTileSlice`` rather than silently
        miscomputed. Column windows address the same L0C memory in the order the
        hardware writes it, so nothing is lost by preferring them.

    Args:
        acc: Accumulator tile
        lhs: Left-hand side tile
        rhs: Right-hand side tile
        init_cond: Optional predicate selecting overwrite over accumulate
        tf32_mode: Optional HF32 input rounding: "round_even" or "round_away".
            Requires 2D FP32 operands, Ascend950, and PTOAS 0.70+. None preserves FP32.
        acc_phase: Producer-side unit-flag phase. Use ``pl.AccPhase.Partial``
            for intermediate K chunks and ``pl.AccPhase.Final`` for the last
            chunk, paired with ``pl.store(..., st_phase=pl.STPhase.Final)``.

    Returns:
        Tile wrapping the matmul_acc operation
    """
    call_expr = _ir_ops.matmul_acc(
        acc.unwrap(),
        lhs.unwrap(),
        rhs.unwrap(),
        init_cond=predicate_to_expr(init_cond),
        tf32_mode=tf32_mode,
        acc_phase=acc_phase,
    )
    return Tile(expr=call_expr)


def batch_matmul_acc(acc: Tile, lhs: Tile, rhs: Tile, init_cond: BoolLike | None = None) -> Tile:
    """Batch matrix multiplication with accumulation: acc += lhs @ rhs.

    Performs the in-place ``acc += lhs @ rhs`` with batch-dim broadcasting between
    ``lhs`` and ``rhs``. The broadcast batch shape must equal the batch shape of
    ``acc`` (acc is the in-place accumulation target and is not broadcast).

    ``init_cond`` behaves exactly as on
    [`matmul_acc`][pypto.language.op.tile_ops.matmul_acc]: where it holds, ``acc``
    is overwritten with ``lhs @ rhs`` rather than accumulated into.
    ``FlattenTileNdTo2D`` forwards the predicate to every 2D ``tile.matmul_acc``
    it unrolls this op into.

    Args:
        acc: Accumulator tile (at least 2D)
        lhs: Left-hand side tile (at least 2D)
        rhs: Right-hand side tile (at least 2D)
        init_cond: Optional predicate selecting overwrite over accumulate

    Returns:
        Tile wrapping the batch_matmul_acc operation
    """
    call_expr = _ir_ops.batch_matmul_acc(
        acc.unwrap(), lhs.unwrap(), rhs.unwrap(), init_cond=predicate_to_expr(init_cond)
    )
    return Tile(expr=call_expr)


def matmul_bias(lhs: Tile, rhs: Tile, bias: Tile, *, tf32_mode: str | None = None) -> Tile:
    """Matrix multiplication with bias add: C = lhs @ rhs + bias.

    Args:
        lhs: Left-hand side tile [M, K]
        rhs: Right-hand side tile [K, N]
        bias: Bias tile [1, N] with the accumulator dtype (FP32 for
            floating-point matrix operands, INT32 for integer matrix operands)
        tf32_mode: Optional HF32 input rounding: "round_even" or "round_away".
            Requires 2D FP32 operands, Ascend950, and PTOAS 0.70+. None preserves FP32.

    Returns:
        Tile wrapping the matmul_bias operation
    """
    call_expr = _ir_ops.matmul_bias(lhs.unwrap(), rhs.unwrap(), bias.unwrap(), tf32_mode=tf32_mode)
    return Tile(expr=call_expr)


def matmul_mx(lhs: Tile, lhs_scale: Tile, rhs: Tile, rhs_scale: Tile) -> Tile:
    """MX block-scale matrix multiplication.

    Scales use logical ``[M, K/32]`` / ``[K/32, N]`` shapes. Both data tiles
    passed to this operation must be FP8E4M3FN. For the supported FP4 x FP8
    input form, explicitly cast the FP4 lhs to FP8E4M3FN before calling this
    operation; native FP4 x FP4 is not supported.

    Args:
        lhs: Left-hand side data tile (FP8E4M3FN)
        lhs_scale: Left-hand side scale tile (FP8E8M0)
        rhs: Right-hand side data tile (FP8E4M3FN)
        rhs_scale: Right-hand side scale tile (FP8E8M0)

    Returns:
        Tile wrapping the matmul_mx operation
    """
    call_expr = _ir_ops.matmul_mx(lhs.unwrap(), lhs_scale.unwrap(), rhs.unwrap(), rhs_scale.unwrap())
    return Tile(expr=call_expr)


def matmul_mx_acc(acc: Tile, lhs: Tile, lhs_scale: Tile, rhs: Tile, rhs_scale: Tile) -> Tile:
    """MX block-scale matmul with accumulation.

    Data operands follow [`matmul_mx`][pypto.language.tile.matmul_mx]: an FP4 lhs must first be cast to
    FP8E4M3FN, and the operation itself receives two FP8E4M3FN tiles.

    Args:
        acc: Accumulator tile
        lhs: Left-hand side data tile (FP8E4M3FN)
        lhs_scale: Left-hand side scale tile (FP8E8M0)
        rhs: Right-hand side data tile (FP8E4M3FN)
        rhs_scale: Right-hand side scale tile (FP8E8M0)

    Returns:
        Tile wrapping the matmul_mx_acc operation
    """
    call_expr = _ir_ops.matmul_mx_acc(
        acc.unwrap(), lhs.unwrap(), lhs_scale.unwrap(), rhs.unwrap(), rhs_scale.unwrap()
    )
    return Tile(expr=call_expr)


def matmul_mx_bias(lhs: Tile, lhs_scale: Tile, rhs: Tile, rhs_scale: Tile, bias: Tile) -> Tile:
    """MX block-scale matmul with bias.

    Data operands follow [`matmul_mx`][pypto.language.tile.matmul_mx]: an FP4 lhs must first be cast to
    FP8E4M3FN, and the operation itself receives two FP8E4M3FN tiles.

    Args:
        lhs: Left-hand side data tile (FP8E4M3FN)
        lhs_scale: Left-hand side scale tile (FP8E8M0)
        rhs: Right-hand side data tile (FP8E4M3FN)
        rhs_scale: Right-hand side scale tile (FP8E8M0)
        bias: Bias tile

    Returns:
        Tile wrapping the matmul_mx_bias operation
    """
    call_expr = _ir_ops.matmul_mx_bias(
        lhs.unwrap(), lhs_scale.unwrap(), rhs.unwrap(), rhs_scale.unwrap(), bias.unwrap()
    )
    return Tile(expr=call_expr)


def gemv(lhs: Tile, rhs: Tile, acc_phase: AccPhase = AccPhase.Unspecified) -> Tile:
    """General Matrix-Vector multiplication: C[1,N] = A[1,K] @ B[K,N].

    ``lhs`` must have exactly one physical and logical row. The rhs logical K
    must cover the lhs logical K. Inputs must use the same INT8, FP16, BF16, or FP32
    dtype; the output is INT32 for INT8 inputs and FP32 otherwise.

    Args:
        lhs: Row vector tile [1, K]
        rhs: Right-hand side tile [K, N]
        acc_phase: Producer-side unit-flag phase. Use ``pl.AccPhase.Partial``
            for intermediate chunks and ``pl.AccPhase.Final`` for the last chunk.

    Returns:
        Tile wrapping the gemv operation
    """
    call_expr = _ir_ops.gemv(lhs.unwrap(), rhs.unwrap(), acc_phase=acc_phase)
    return Tile(expr=call_expr)


def gemv_acc(
    acc: Tile,
    lhs: Tile,
    rhs: Tile,
    acc_phase: AccPhase = AccPhase.Unspecified,
    *,
    init_cond: BoolLike | None = None,
) -> Tile:
    """GEMV with accumulation: C[1,N] += A[1,K] @ B[K,N].

    ``acc`` must use the GEMV output dtype. The logical K extents and lhs/rhs
    dtype requirements are identical to [`gemv`][pypto.language.tile.gemv].

    ``init_cond`` makes the accumulator's initial value conditional, exactly as in
    [`matmul_acc`][pypto.language.tile.matmul_acc] — GEMV is a matmul whose M is
    1, run on the same cube MAD, so it carries the same predicate bit. On the
    steps where it holds, ``acc`` is overwritten with ``lhs @ rhs`` rather than
    accumulated into, which removes the peeled first step from split-K::

        for k0 in pl.pipeline(0, K, K_TILE):
            a = pl.load(vec, [0, k0], [1, K_TILE], target_memory=pl.MemorySpace.Mat)
            b = pl.load(mat, [k0, 0], [K_TILE, N], target_memory=pl.MemorySpace.Mat)
            acc = pl.tile.gemv_acc(acc, a, b, init_cond=(k0 == 0))

    A literal ``True`` / ``False`` selects one form at compile time; a runtime
    predicate lowers to a branch over the two, with no phi on the accumulator.

    ``init_cond`` is keyword-only because ``acc_phase`` already owns the fourth
    positional slot.

    Args:
        acc: Accumulator tile [1, N]
        lhs: Row vector tile [1, K]
        rhs: Right-hand side tile [K, N]
        acc_phase: Producer-side unit-flag phase. Use ``pl.AccPhase.Partial``
            for intermediate chunks and ``pl.AccPhase.Final`` for the last chunk.
        init_cond: Optional predicate selecting overwrite over accumulate

    Returns:
        Tile wrapping the gemv_acc operation
    """
    call_expr = _ir_ops.gemv_acc(
        acc.unwrap(),
        lhs.unwrap(),
        rhs.unwrap(),
        acc_phase=acc_phase,
        init_cond=predicate_to_expr(init_cond),
    )
    return Tile(expr=call_expr)


def gemv_bias(
    lhs: Tile,
    rhs: Tile,
    bias: Tile,
    acc_phase: AccPhase = AccPhase.Unspecified,
) -> Tile:
    """GEMV with bias add: C[1,N] = A[1,K] @ B[K,N] + bias[1,N].

    ``bias`` must use the GEMV output dtype and its valid shape must cover the
    logical output shape ``[1, N]``. The logical K extents and lhs/rhs dtype
    requirements are identical to [`gemv`][pypto.language.tile.gemv].

    Args:
        lhs: Row vector tile [1, K]
        rhs: Right-hand side tile [K, N]
        bias: Bias tile [1, N] with the accumulator dtype (FP32 for
            floating-point matrix operands, INT32 for integer matrix operands)
        acc_phase: Producer-side unit-flag phase. Use ``pl.AccPhase.Partial``
            for intermediate chunks and ``pl.AccPhase.Final`` for the last chunk.

    Returns:
        Tile wrapping the gemv_bias operation
    """
    call_expr = _ir_ops.gemv_bias(lhs.unwrap(), rhs.unwrap(), bias.unwrap(), acc_phase=acc_phase)
    return Tile(expr=call_expr)
