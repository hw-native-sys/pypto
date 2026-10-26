# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile indexed access, scatter, and sorting."""

from collections.abc import Sequence
from typing import Any, Literal, overload

from pypto.ir.op import tile_ops as _ir_ops
from pypto.ir.utils import (
    _normalize_expr,
)
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import (
    Expr,
    MemorySpace,
)

from ..typing import Scalar, Tensor, Tile
from ._tile_common import _TensorT


def scatter_update(input: Tile, *args: Any, **kwargs: Any) -> Tile:
    """Update tile rows at positions specified by 2D index tile with values from src.

    Supports two rank variants:

    - 2D: ``input [rows, d]``, ``src [b*s, d]``, ``index [b, s]``
    - 4D: ``input [blockNum, blockSize, 1, d]``, ``src [b, s, 1, d]``, ``index [b, s]``

    Accepts the same flexible call shapes as the IR builder
    ``pypto.ir.op.tile.scatter_update``:

    - ``scatter_update(input, dim, index, src)``
    - ``scatter_update(input, index, src, dim=-2)``
    - ``scatter_update(input, dim, index=..., src=...)``

    Tile / Scalar wrappers are unwrapped before forwarding so the IR builder
    receives raw ``Expr`` operands.
    """

    def _unwrap(v: Any) -> Any:
        return v.unwrap() if isinstance(v, (Tile, Scalar)) else v

    fwd_args = tuple(_unwrap(a) for a in args)
    fwd_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
    return Tile(expr=_ir_ops.scatter_update(input.unwrap(), *fwd_args, **fwd_kwargs))


def sort32(src: Tile, idx: Tile, *, tmp: Tile | None = None) -> Tile:
    """Sort fixed 32-element blocks with explicit index tile.

    Sorts 32-element blocks in src, permuting idx alongside.
    Returns an 8-byte value-index-pair tile. Its last dimension is 2x the input
    width for FP32 and 4x the input width for FP16.

    For FP16 src: initialize idx with [0, 1, 2, ..., 31] per block.
    For FP32 src: initialize idx with [0, 2, 4, ..., 62] per block.

    Args:
        src: Input value tile (FP16 or FP32)
        idx: Input index tile with sequential offsets
        tmp: Optional A2/A3 PTOAS scratch tile. Normally compiler-generated.

    Returns:
        Tile wrapping the dtype-dependent expanded sort32 output
    """
    call_expr = _ir_ops.sort32(src.unwrap(), idx.unwrap(), tmp=None if tmp is None else tmp.unwrap())
    return Tile(expr=call_expr)


def gather(src: Tile, indices: Tile, tmp: Tile) -> Tile:
    """Gather elements from src tile by per-element indices (index form).

    Computes ``dst[i, j] = src[indices[i, j]]``. Maps to PTOAS ``pto.tgather``
    index form. For the hardware mask-pattern variant, use [`gather_mask`][pypto.language.tile.gather_mask].

    Args:
        src: Source tile (FP16, FP32, INT16, or INT32)
        indices: Index tile (INT32 with any src, or INT16 with a 16-bit src — FP16/INT16);
            selects which elements of ``src`` to gather
        tmp: Temporary workspace tile (any Vec dtype; required as an operand but
            not constrained by the A5 index form — A2/A3 narrows this at PTOAS)

    Returns:
        Tile with gathered elements (same dtype as ``src``)
    """
    call_expr = _ir_ops.gather(src.unwrap(), indices.unwrap(), tmp.unwrap())
    return Tile(expr=call_expr)


def gatherb(
    src: Tile,
    offset: Tile,
    *,
    output_dtype: int | DataType | None = None,
) -> Tile:
    """Gather 32-byte blocks from ``src`` by UINT32 byte offsets.

    Each offset selects one 32-byte source block. One offset column expands to
    ``32 / sizeof(output_dtype)`` output elements. ``output_dtype`` defaults to
    ``src.dtype`` and may select another supported byte interpretation.
    A sliced source must have a byte address that PyPTO can prove is 32-byte
    aligned; dynamic column offsets are rejected conservatively.

    Args:
        src: Source tile to gather blocks from.
        offset: UINT32 tile of **byte** offsets into ``src`` -- not element indices.
        output_dtype: Byte interpretation of the result. Defaults to ``src.dtype``.

    Returns:
        Tile wrapping the gatherb operation.
    """
    return Tile(expr=_ir_ops.gatherb(src.unwrap(), offset.unwrap(), output_dtype=output_dtype))


def gather_mask(
    src: Tile,
    mask_pattern: int,
    *,
    output_dtype: int | DataType | None = None,
) -> Tile:
    """Gather elements from src tile by a fixed hardware mask pattern (mask form).

    Selects elements according to a stride/mask pattern baked into the hardware.
    For the per-element indices variant, use [`gather`][pypto.language.tile.gather].

    Args:
        src: Source tile (FP16, FP32, INT16, or INT32)
        mask_pattern: Mask pattern selector (1-7), see [`MaskPattern`][pypto.language.tile.MaskPattern].
            1=P0101, 2=P1010, 3=P0001, 4=P0010, 5=P0100, 6=P1000, 7=P1111
        output_dtype: Optional output dtype. When provided, the result tile has
            this dtype instead of ``src``'s dtype (bit reinterpretation, no
            conversion). Hardware requires ``sizeof(dst_dtype) == sizeof(src_dtype)``.
            Example: ``output_dtype=pl.UINT32`` to extract sort32 index bits from
            FP32 memory.

    Returns:
        Tile with mask-selected elements

    Examples:
        # Same dtype
        out = gather_mask(src, mask_pattern=pl.tile.MaskPattern.P0101)

        # Cross-type output (FP32 bits → UINT32)
        out = gather_mask(src, pl.tile.MaskPattern.P1010, output_dtype=pl.UINT32)
    """
    call_expr = _ir_ops.gather_mask(src.unwrap(), mask_pattern, output_dtype=output_dtype)
    return Tile(expr=call_expr)


def gather_compare(
    src: Tile,
    kvalue: int | Scalar | Expr,
    tmp: Tile,
    *,
    cmp_mode: str | int = "eq",
    offset: int = 0,
    out_cols: int,
    count_dtype: int | DataType | None = None,
) -> tuple[Tile, Tile]:
    """Compare-form gather (tile-level): produce (dst, cdst) — gathered indices
    and per-row match counts.

    Maps to PTOAS ``pto.tgather`` compare-form. Hardware DPS allocation of
    dst/cdst is handled downstream — only the three inputs (src, kvalue, tmp)
    appear at this surface.

    DSL form (inside ``@pl.function``)::

        dst, cdst = pl.tile.gather_compare(src, kvalue, tmp,
                                            cmp_mode="eq", offset=0,
                                            out_cols=K)

    The ``a, b = call(...)`` Python tuple unpack is desugared by the parser
    into ``_tuple = call; a = _tuple[0]; b = _tuple[1]``. The parser
    consumes the underlying tuple-typed ``ir.Call`` returned by
    ``pypto.ir.op.tile_ops.gather_compare``; the ``(Tile, Tile)`` split
    below only runs in interactive Python contexts.

    Args:
        src: Source tile (FP16/FP32/INT16/INT32, 2D).
        kvalue: Scalar threshold (dtype must match ``src``; applied to every row).
        tmp: Workspace tile (UINT8) sized for the codegen kernel.
        cmp_mode: ``"eq"`` / ``"ne"`` / ``"lt"`` / ``"le"`` / ``"gt"`` /
            ``"ge"`` or int ``0..5``. Defaults to ``"eq"``.
        offset: Starting index offset (default 0).
        out_cols: Output column count per row for ``dst`` (positive int, required).
        count_dtype: Per-row count dtype, INT32 or UINT32; defaults to INT32.

    Returns:
        ``(dst, cdst)`` where ``dst`` is a Tile ``[rows, out_cols]`` of INT32
        gathered indices and ``cdst`` is a Tile ``[1, rows]`` of ``count_dtype``
        per-row match counts.
    """
    kv_expr = kvalue.unwrap() if isinstance(kvalue, Scalar) else _normalize_expr(kvalue)
    call_expr = _ir_ops.gather_compare(
        src.unwrap(),
        kv_expr,
        tmp.unwrap(),
        cmp_mode=cmp_mode,
        offset=offset,
        out_cols=out_cols,
        count_dtype=count_dtype,
    )
    span = call_expr.span
    return (
        Tile(expr=_ir_core.TupleGetItemExpr(call_expr, 0, span)),
        Tile(expr=_ir_core.TupleGetItemExpr(call_expr, 1, span)),
    )


def scatter(dst: Tile, src: Tile, indexes: Tile) -> Tile:
    """Scatter elements of ``src`` into ``dst`` at per-element flattened indices.

    Computes ``dst.flat[indexes[i, j]] = src[i, j]``, i.e. ``indexes`` carries the
    *flattened* destination offset for each ``src`` element and therefore has the
    **same [rows, cols] shape as** ``src``. Maps to PTOAS ``pto.tscatter`` index
    form. The op is DPS — ``dst`` is the first (in/out) argument, rewritten in
    place, and the returned Tile aliases the same buffer. For the hardware
    mask-pattern variant, use [`scatter_mask`][pypto.language.tile.scatter_mask].

    Args:
        dst: Destination tile (same dtype as ``src``; rewritten in-place).
            Flat-addressed, so its column count is independent of ``src``.
        src: Source tile (FP16/FP32/BF16/INT8/INT16/INT32, 2D)
        indexes: Per-element flattened destination index tile (INT16 or INT32;
            same shape as ``src``). The element width must match ``dst``: 4-byte
            dst → INT32, 2-byte dst → INT16, 1-byte dst → INT16.

    Returns:
        Tile aliasing the post-scatter ``dst`` tile.
    """
    call_expr = _ir_ops.scatter(dst.unwrap(), src.unwrap(), indexes.unwrap())
    return Tile(expr=call_expr)


def scatter_mask(dst: Tile, src: Tile, mask_pattern: int) -> Tile:
    """Scatter ``src`` rows into mask-marked columns of ``dst`` (mask form).

    For each row, the elements of ``src`` are written into the columns of
    ``dst`` selected by ``mask_pattern`` (the inverse of [`gather_mask`][pypto.language.tile.gather_mask]).

    Unlike [`gather_mask`][pypto.language.tile.gather_mask] (a real ``pto.tgather`` ISA op on A2/A3 and A5),
    mask-pattern scatter is not a distinct pto-isa instruction — PyPTO emits it
    as a ``pto.tscatter`` mask-form construct for A2/A3 / CPU-sim style lowering
    paths.

    Args:
        dst: Destination tile (rewritten on positions selected by ``mask_pattern``)
        src: Source tile (compact rows; same dtype as ``dst``)
        mask_pattern: Mask pattern selector (1-7), see [`MaskPattern`][pypto.language.tile.MaskPattern].
            1=P0101, 2=P1010, 3=P0001, 4=P0010, 5=P0100, 6=P1000, 7=P1111

    Returns:
        Tile aliasing the post-scatter ``dst`` tile.

    Examples:
        out = scatter_mask(dst, src, mask_pattern=pl.tile.MaskPattern.P0101)
    """
    call_expr = _ir_ops.scatter_mask(dst.unwrap(), src.unwrap(), mask_pattern)
    return Tile(expr=call_expr)


def mscatter(src: Tile, idx: Tile, output_tensor: _TensorT) -> _TensorT:
    """Scatter-store tile elements into a tensor at per-element indices.

    Semantics: ``output_tensor[idx[i, j]] = src[i, j]``

    Maps to the PTOAS ``pto.mscatter`` instruction.

    Args:
        src: Source tile (FP16, FP32, INT16, or INT32)
        idx: Index tile (INT32, same rank as src)
        output_tensor: Output tensor to scatter into (same dtype as src)

    Returns:
        Tensor wrapping the mscatter operation

    Example:
        >>> result = pl.tile.mscatter(src_tile, idx_tile, out_tensor)
    """
    call_expr = _ir_ops.mscatter(src.unwrap(), idx.unwrap(), output_tensor.unwrap())
    return output_tensor.__class__(expr=call_expr)


@overload
def mgather(
    mem: Tensor,
    idx: Tile,
    coalesce: str | int = ...,
    *,
    gather_oob: str | int = ...,
    target_memory: Literal[MemorySpace.Vec] = ...,
    scratch: None = ...,
    valid_shape: None = ...,
) -> Tile: ...


@overload
def mgather(
    mem: Tensor,
    idx: Tensor,
    coalesce: Literal["row", 0] = ...,
    *,
    gather_oob: str | int = ...,
    target_memory: Literal[MemorySpace.Mat],
    scratch: None = ...,
    valid_shape: Sequence[int] | None = ...,
) -> Tile: ...


@overload
def mgather(
    mem: Tensor,
    idx: Tensor,
    coalesce: Literal["elem", 1],
    *,
    gather_oob: str | int = ...,
    target_memory: Literal[MemorySpace.Mat],
    scratch: Tensor,
    valid_shape: Sequence[int] | None = ...,
) -> Tile: ...


def mgather(
    mem: Tensor,
    idx: Tile | Tensor,
    coalesce: str | int = "row",
    *,
    gather_oob: str | int = "undefined",
    target_memory: MemorySpace = MemorySpace.Vec,
    scratch: Tensor | None = None,
    valid_shape: Sequence[int] | None = None,
) -> Tile:
    """Gather-load rows or elements from a GM tensor into a fresh Vec or Mat tile.

    Vec output uses a 2D INT32 index tile. Mat output uses a GM INT32 index tensor
    and produces canonical NZ layout; its element mode additionally requires a
    same-dtype GM scratch tensor.

    Args:
        mem: Source tensor in GM.
        idx: Two-dimensional INT32 index tile for Vec output, or GM tensor for
            Mat output.
        coalesce: ``"row"``/``0`` for row gather or ``"elem"``/``1`` for flat
            element gather. Integer values support printed-IR round trips.
        gather_oob: Out-of-bounds handling: ``"undefined"``, ``"clamp"``,
            ``"wrap"``, ``"zero"``, or the corresponding integer ``0..3``.
        target_memory: ``MemorySpace.Vec`` (default) or ``MemorySpace.Mat``.
            This selects the operator *variant*, not merely a placement: the two
            take a different ``idx`` type and produce a different output shape
            and view, so it cannot be left for the compiler to infer.
        scratch: Same-dtype GM workspace required by Mat element gather and
            forbidden by the other forms.
        valid_shape: Optional two-dimensional written region for Mat output.
            Vec output derives its valid region from the index tile.
    """
    return Tile(
        expr=_ir_ops.mgather(
            mem.unwrap(),
            idx.unwrap(),
            coalesce=coalesce,
            gather_oob=gather_oob,
            target_memory=target_memory,
            scratch=None if scratch is None else scratch.unwrap(),
            valid_shape=valid_shape,
        )
    )


@overload
def mrgsort(src0: Tile, *, block_len: int | Scalar) -> Tile: ...


@overload
def mrgsort(
    src0: Tile,
    src1: Tile,
    *,
    tmp: Tile,
    exhausted: bool = ...,
) -> Tile: ...


@overload
def mrgsort(
    src0: Tile,
    src1: Tile,
    src2: Tile,
    *,
    tmp: Tile,
    exhausted: bool = ...,
) -> Tile: ...


@overload
def mrgsort(
    src0: Tile,
    src1: Tile,
    src2: Tile,
    src3: Tile,
    tmp: Tile,
    exhausted: bool = ...,
) -> Tile: ...


@overload
def mrgsort(
    src0: Tile,
    src1: Tile | None = ...,
    src2: Tile | None = ...,
    src3: Tile | None = ...,
    tmp: Tile | None = ...,
    exhausted: bool = ...,
    *,
    block_len: int | Scalar | None = ...,
) -> Tile: ...


def mrgsort(
    src0: Tile,
    src1: Tile | None = None,
    src2: Tile | None = None,
    src3: Tile | None = None,
    tmp: Tile | None = None,
    exhausted: bool = False,
    *,
    block_len: int | Scalar | None = None,
) -> Tile:
    """Merge sort — format1 (single-list) or format2 (2-4 way merge).

    Format1: sorts a tile containing multiple pre-sorted runs of length block_len.
    Format2: merges 2, 3, or 4 pre-sorted input tiles into one sorted output.

    Format1 usage (keyword block_len):
        out = mrgsort(src, block_len=64)

    Format2 2-way usage (keyword tmp):
        out = mrgsort(src0, src1, tmp=tmp_tile)
        out = mrgsort(src0, src1, tmp=tmp_tile, exhausted=True)

    Format2 3-way usage:
        out = mrgsort(src0, src1, src2, tmp=tmp_tile)

    Format2 4-way usage (5 positional args):
        out = mrgsort(src0, src1, src2, src3, tmp)
        out = mrgsort(src0, src1, src2, src3, tmp, exhausted=True)

    Args:
        src0: For format1: input tile with pre-sorted runs (FP16 or FP32).
              For format2: first sorted input tile.
        src1: (format2) Second sorted input tile.
        src2: (format2, optional) Third sorted input tile (3-way or 4-way).
        src3: (format2, optional) Fourth sorted input tile (4-way only).
        tmp: (format2) Temporary workspace tile (same shape as output).
              Pass as keyword arg for 2-way and 3-way.
        exhausted: (format2) If True, marks inputs as exhausted (default: False).
        block_len: (format1, keyword-only) Run length, must be multiple of 64.

    Returns:
        Tile with merged sorted elements
    """
    if block_len is not None:
        # format1: single-list merge sort
        if any(arg is not None for arg in (src1, src2, src3, tmp)):
            raise ValueError(
                "mrgsort() format1 (block_len=...) and format2 (src1, ..., tmp) "
                "are mutually exclusive; do not pass format2 arguments with block_len"
            )
        block_len_expr = block_len.unwrap() if isinstance(block_len, Scalar) else block_len
        call_expr = _ir_ops.mrgsort(src0.unwrap(), block_len=block_len_expr)
        return Tile(expr=call_expr)
    # format2: 2-4 way merge
    if src1 is None:
        raise ValueError(
            "mrgsort() requires either block_len=<int> for format1, "
            "or at least (src0, src1, tmp=<tile>) for format2"
        )
    if tmp is None:
        raise ValueError(
            "mrgsort() format2 requires tmp; use mrgsort(src0, src1[, src2[, src3]], tmp=<tile>)"
        )
    call_expr = _ir_ops.mrgsort(
        src0.unwrap(),
        src1.unwrap(),
        src2.unwrap() if src2 is not None else None,
        src3.unwrap() if src3 is not None else None,
        tmp=tmp.unwrap(),
        exhausted=exhausted,
    )
    return Tile(expr=call_expr)
