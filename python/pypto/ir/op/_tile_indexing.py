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
from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import (
    Call,
    ConstInt,
    Expr,
    MemorySpace,
    Span,
)

from ..utils import (
    _get_span_or_capture,
    _to_make_tuple,
)


def scatter_update(
    input: Expr,
    *args: Expr | int,
    dim: int | Expr | None = None,
    index: Expr | int | None = None,
    src: Expr | int | None = None,
    span: Span | None = None,
) -> Call:
    """Update tile rows at positions specified by 2D index tile with values from src.

    Supports two variants based on input/src rank:
    - 2D: input [rows, d], src [b*s, d], index [b, s]
    - 4D: input [blockNum, blockSize, 1, d], src [b, s, 1, d], index [b, s]

    Accepts both call forms:
    - scatter_update(input, dim, index, src)
    - scatter_update(input, index, src, dim=-2)

    Args:
        input: Destination tile (TileType, 2D or 4D)
        dim: Dimension to scatter along (currently only -2 is supported)
        index: 2D index tile [b, s] of integer dtype
        src: Source tile (same rank as input)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression returning a TileType with the same shape/dtype as input
    """
    if len(args) == 3 and dim is None and index is None and src is None:
        dim, index, src = args
    elif len(args) == 2 and dim is not None and index is None and src is None:
        index, src = args
    elif len(args) == 1 and dim is None and index is not None and src is not None:
        # (input, dim, index=..., src=...) — dim passed positionally
        dim = args[0]
    elif len(args) != 0:
        raise TypeError(
            "scatter_update expects (input, dim, index, src), "
            "(input, index, src, dim=...), or (input, dim, index=..., src=...)"
        )

    if dim is None or index is None or src is None:
        raise TypeError("scatter_update requires input, dim, index, and src")

    actual_span = _get_span_or_capture(span)
    if isinstance(dim, ConstInt):
        dim_val = int(dim.value)
    elif isinstance(dim, int):
        dim_val = dim
    else:
        raise TypeError(f"dim must be int or ConstInt, got {type(dim)}")

    if not isinstance(index, Expr):
        raise TypeError(f"index must be Expr, got {type(index)}")
    if not isinstance(src, Expr):
        raise TypeError(f"src must be Expr, got {type(src)}")
    op_args: list[Expr] = [input, index, src]
    kwargs: dict[str, Any] = {"dim": dim_val}
    return _ir_core.create_op_call("tile.scatter_update", op_args, kwargs, actual_span)


def mscatter(
    src: Expr,
    idx: Expr,
    output_tensor: Expr,
    span: Span | None = None,
) -> Call:
    """Scatter-store elements from src tile to output_tensor at per-element indices.

    Semantics: ``output_tensor[idx[i, j]] = src[i, j]``

    Maps to the PTOAS ``pto.mscatter`` instruction.

    Args:
        src: Source tile (FP16, FP32, INT16, or INT32)
        idx: Index tile (INT32, same rank as src)
        output_tensor: Output tensor (TensorType, same dtype as src)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns the output tensor
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.mscatter", [src, idx, output_tensor], {}, actual_span)


_MGATHER_COALESCE = {"row": 0, "elem": 1}


_MGATHER_GATHER_OOB = {"undefined": 0, "clamp": 1, "wrap": 2, "zero": 3}


def _resolve_mgather_coalesce(coalesce: object) -> int:
    if isinstance(coalesce, str):
        try:
            return _MGATHER_COALESCE[coalesce]
        except KeyError as e:
            raise ValueError(f"mgather coalesce must be 'row', 'elem', 0, or 1, got {coalesce!r}") from e
    if isinstance(coalesce, int) and not isinstance(coalesce, bool) and coalesce in (0, 1):
        return coalesce
    raise ValueError(f"mgather coalesce must be 'row', 'elem', 0, or 1, got {coalesce!r}")


def _resolve_mgather_gather_oob(gather_oob: object) -> int:
    if isinstance(gather_oob, str):
        try:
            return _MGATHER_GATHER_OOB[gather_oob]
        except KeyError as e:
            raise ValueError(
                "mgather gather_oob must be 'undefined', 'clamp', 'wrap', 'zero', or int 0-3, "
                f"got {gather_oob!r}"
            ) from e
    if isinstance(gather_oob, int) and not isinstance(gather_oob, bool) and gather_oob in range(4):
        return gather_oob
    raise ValueError(
        f"mgather gather_oob must be 'undefined', 'clamp', 'wrap', 'zero', or int 0-3, got {gather_oob!r}"
    )


def mgather(
    mem: Expr,
    idx: Expr,
    coalesce: str | int = "row",
    span: Span | None = None,
    *,
    gather_oob: str | int = "undefined",
    target_memory: MemorySpace = MemorySpace.Vec,
    scratch: Expr | None = None,
    valid_shape: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
) -> Call:
    """Gather-load indexed rows or elements from a GM tensor into Vec or Mat.

    Vec output uses a 2D INT32 index tile. Mat output uses a GM INT32 index
    tensor and produces canonical NZ layout; its element mode additionally
    requires a same-dtype GM scratch tensor.
    ``gather_oob`` selects undefined, clamp, wrap, or zero handling.
    """
    if target_memory not in (MemorySpace.Vec, MemorySpace.Mat):
        raise ValueError(
            f"mgather target_memory must be MemorySpace.Vec or MemorySpace.Mat, got {target_memory}"
        )
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"coalesce": _resolve_mgather_coalesce(coalesce)}
    if target_memory != MemorySpace.Vec:
        kwargs["target_memory"] = target_memory
    resolved_gather_oob = _resolve_mgather_gather_oob(gather_oob)
    if resolved_gather_oob != 0:
        kwargs["gather_oob"] = resolved_gather_oob
    args = [mem, idx]
    if scratch is not None:
        args.append(scratch)
    if valid_shape is not None:
        args.append(_to_make_tuple(valid_shape, actual_span))
    return _ir_core.create_op_call(
        "tile.mgather",
        args,
        kwargs,
        actual_span,
    )


# ============================================================================
# Sorting Operations
# ============================================================================


def sort32(src: Expr, idx: Expr, span: Span | None = None, *, tmp: Expr | None = None) -> Call:
    """Sort fixed 32-element blocks with explicit index tile.

    Sorts 32-element blocks in src and permutes idx accordingly.
    Output tile stores 8-byte value-index pairs. Its last dimension is 2x the
    input width for FP32 and 4x the input width for FP16.

    Args:
        src: Input value tile (TileType, FP16 or FP32, Vec memory)
        idx: Input index tile (TileType, Vec memory) with sequential offsets
        span: Optional source span for debugging
        tmp: Optional A2/A3 PTOAS scratch tile. Normally compiler-generated.

    Returns:
        Call expression returning the dtype-dependent expanded sort output
    """
    actual_span = _get_span_or_capture(span)
    args = [src, idx]
    if tmp is not None:
        args.append(tmp)
    return _ir_core.create_op_call("tile.sort32", args, {}, actual_span)


# ============================================================================
# Gather Operations
# ============================================================================


def gather(
    src: Expr,
    indices: Expr,
    tmp: Expr,
    span: Span | None = None,
) -> Call:
    """Gather elements from src by per-element indices (index form).

    Computes ``dst[i, j] = src[indices[i, j]]``. Maps to PTOAS ``pto.tgather`` index form.

    Args:
        src: Source tile (FP16, FP32, INT16, or INT32)
        indices: Index tile (INT32)
        tmp: Temporary workspace tile (INT32)
        span: Optional source span

    Returns:
        Call expression returning gathered tile
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.gather", [src, indices, tmp], {}, actual_span)


def gatherb(
    src: Expr,
    offset: Expr,
    span: Span | None = None,
    *,
    output_dtype: int | DataType | None = None,
) -> Call:
    """Gather 32-byte source blocks by UINT32 byte offsets (``pto.tgatherb``).

    Each offset selects the first byte of one 32-byte block. The output shape
    is ``[offset_rows, offset_cols * (32 / sizeof(output_dtype))]``; the offset
    valid shape is expanded by the same factor.
    """
    actual_span = _get_span_or_capture(span)
    kwargs = {} if output_dtype is None else {"output_dtype": output_dtype}
    return _ir_core.create_op_call("tile.gatherb", [src, offset], kwargs, actual_span)


def gather_mask(
    src: Expr,
    mask_pattern: int,
    *,
    output_dtype: int | DataType | None = None,
    span: Span | None = None,
) -> Call:
    """Gather elements from src using a fixed hardware mask pattern (mask form).

    Maps to PTOAS ``pto.tgather`` mask form.

    Args:
        src: Source tile (FP16, FP32, INT16, or INT32)
        mask_pattern: Mask pattern selector (1-7).
            1=P0101, 2=P1010, 3=P0001, 4=P0010, 5=P0100, 6=P1000, 7=P1111
        output_dtype: Optional output dtype (keyword-only). When provided, the result
            tile has this dtype instead of src's dtype. The hardware only requires
            sizeof(dst_dtype) == sizeof(src_dtype). Useful for extracting UINT32 index
            bits from FP32 sort32 output (bit reinterpretation).
        span: Optional source span

    Returns:
        Call expression returning gathered tile
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"mask_pattern": mask_pattern}
    if output_dtype is not None:
        kwargs["output_dtype"] = output_dtype
    return _ir_core.create_op_call("tile.gather_mask", [src], kwargs, actual_span)


# Compare modes accepted by tile.gather_compare / tensor.gather_compare.
# Maps to PTOAS pto.tgather cmpMode: eq/ne/lt/le/gt/ge.
_GATHER_COMPARE_CMP_MODES = ("eq", "ne", "lt", "le", "gt", "ge")


def resolve_gather_compare_cmp_mode(cmp_mode: object) -> int:
    """Normalize a gather_compare cmp_mode to its integer enum value.

    Accepts either a string in ``{"eq", "ne", "lt", "le", "gt", "ge"}`` or
    an int in ``[0, 5]``. Raises ``ValueError`` on anything else.
    """
    if isinstance(cmp_mode, str):
        try:
            return _GATHER_COMPARE_CMP_MODES.index(cmp_mode)
        except ValueError as e:
            raise ValueError(
                f"Invalid cmp_mode {cmp_mode!r}: expected one of {list(_GATHER_COMPARE_CMP_MODES)} or int 0-5"
            ) from e
    if isinstance(cmp_mode, int) and not isinstance(cmp_mode, bool):
        if 0 <= cmp_mode < len(_GATHER_COMPARE_CMP_MODES):
            return cmp_mode
        raise ValueError(
            f"Invalid cmp_mode {cmp_mode}: int must be in [0, {len(_GATHER_COMPARE_CMP_MODES) - 1}]"
        )
    raise ValueError(f"Invalid cmp_mode {cmp_mode!r}: expected str or int 0-5")


def gather_compare(
    src: Expr,
    kvalue: Expr,
    tmp: Expr,
    *,
    cmp_mode: str | int,
    offset: int = 0,
    out_cols: int,
    count_dtype: int | DataType | None = None,
    span: Span | None = None,
) -> Call:
    """Compare-form gather (tile-level): scan ``src`` per-row against ``kvalue``.

    Maps to PTOAS ``pto.tgather`` compare-form. Returns a single
    :class:`Call` whose result type is a ``TupleType{dst, cdst}``::

        dst  : TileType, [rows, out_cols], INT32  — gathered indices
        cdst : TileType, [1, aligned_rows], count_dtype — per-row match count

    ``aligned_rows`` rounds the source's physical row count up to a 32-byte
    count row (eight INT32/UINT32 elements). The count tile's valid shape is
    ``[1, src_valid_rows]``; padding adds capacity, not additional counts.

    The DSL form ``d, c = pl.tile.gather_compare(src, kvalue, tmp, ...)`` is
    desugared by the parser into ``_tuple = call; d = _tuple[0]; c = _tuple[1]``.

    Args:
        src: Source tile (FP16/FP32/INT16/INT32, 2D).
        kvalue: Scalar threshold (ScalarType; dtype must match ``src``; applied to every row).
        tmp: Workspace tile (UINT8) sized for the codegen kernel.
        cmp_mode: Compare mode — one of ``"eq"`` / ``"ne"`` / ``"lt"`` /
            ``"le"`` / ``"gt"`` / ``"ge"`` or int ``0..5``.
        offset: Starting index offset (default 0).
        out_cols: Output column count per row for ``dst`` (positive int).
        count_dtype: Per-row count dtype, INT32 or UINT32; defaults to INT32.
        span: Optional source span (auto-captured if omitted).
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {
        "cmp_mode": resolve_gather_compare_cmp_mode(cmp_mode),
        "offset": offset,
        "out_cols": out_cols,
    }
    if count_dtype is not None:
        kwargs["count_dtype"] = count_dtype
    return _ir_core.create_op_call("tile.gather_compare", [src, kvalue, tmp], kwargs, actual_span)


# ============================================================================
# Scatter Operations
# ============================================================================


def scatter(
    dst: Expr,
    src: Expr,
    indexes: Expr,
    span: Span | None = None,
) -> Call:
    """Scatter elements of ``src`` into ``dst`` at per-element flattened indices.

    Computes ``dst.flat[indexes[i, j]] = src[i, j]``, i.e. ``indexes`` carries the
    *flattened* destination offset for each ``src`` element and therefore has the
    **same [rows, cols] shape as** ``src``. Maps to PTOAS ``pto.tscatter`` index
    form. The op is DPS — ``dst`` is the first (in/out) argument, rewritten in
    place, and the call's return value aliases ``dst``.

    Args:
        dst: Destination tile (same dtype as src; rewritten in-place via DPS).
            Flat-addressed, so its column count is independent of ``src``.
        src: Source tile (FP16/FP32/BF16/INT8/INT16/INT32, 2D)
        indexes: Per-element flattened destination index tile (INT16 or INT32;
            same shape as ``src``)
        span: Optional source span

    Returns:
        Call expression aliasing the post-scatter ``dst`` tile.
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.scatter", [dst, src, indexes], {}, actual_span)


def scatter_mask(
    dst: Expr,
    src: Expr,
    mask_pattern: int,
    span: Span | None = None,
) -> Call:
    """Scatter ``src`` rows into mask-marked columns of ``dst`` (mask form).

    DPS — ``dst`` is the first (in/out) argument, rewritten in place on
    mask-selected positions, and the call result aliases ``dst``.

    Unlike :func:`gather_mask` (a real ``pto.tgather`` ISA op on A2/A3 and A5),
    mask-pattern scatter is not a distinct pto-isa instruction — PyPTO emits it
    as a ``pto.tscatter`` mask-form construct for A2/A3 / CPU-sim style lowering
    paths.

    Args:
        dst: Destination tile (rewritten on positions selected by ``mask_pattern``)
        src: Source tile (compact rows, same dtype as ``dst``)
        mask_pattern: Mask pattern selector (1-7).
            1=P0101, 2=P1010, 3=P0001, 4=P0010, 5=P0100, 6=P1000, 7=P1111
        span: Optional source span

    Returns:
        Call expression aliasing the post-scatter ``dst`` tile.
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "tile.scatter_mask", [dst, src], {"mask_pattern": mask_pattern}, actual_span
    )


# ============================================================================
# Merge Sort Operations
# ============================================================================


def mrgsort(
    src0: Expr,
    src1: Expr | None = None,
    src2: Expr | None = None,
    src3: Expr | None = None,
    tmp: Expr | None = None,
    exhausted: bool = False,
    *,
    block_len: int | Expr | None = None,
    span: Span | None = None,
) -> Call:
    """Merge sort — format1 (single-list) or format2 (2-4 way merge).

    Format1 (block_len form): sorts a tile containing multiple pre-sorted runs.
    Format2 (2-4 way form): merges 2, 3, or 4 pre-sorted input tiles.

    Args:
        src0: For format1: input tile with pre-sorted runs (FP16 or FP32).
              For format2: first sorted input tile.
        src1: (format2) Second sorted input tile.
        src2: (format2, optional) Third sorted input tile (3-way or 4-way).
        src3: (format2, optional) Fourth sorted input tile (4-way only).
        tmp: (format2) Temporary workspace tile, must be passed as keyword arg for 2/3-way.
        exhausted: (format2) If True, marks inputs as exhausted (default: False).
        block_len: (format1, keyword-only) Run length, must be multiple of 64.
        span: Optional source span for debugging.

    Returns:
        Call expression returning merged sorted tile.
    """
    actual_span = _get_span_or_capture(span)
    if block_len is not None:
        # format1: single-list merge sort (pto.tmrgsort format1)
        if any(arg is not None for arg in (src1, src2, src3, tmp)):
            raise ValueError(
                "mrgsort() format1 (block_len=...) and format2 (src1, ..., tmp) "
                "are mutually exclusive; do not pass format2 arguments with block_len"
            )
        # PTO ISA requires block_len as i32. The parser may emit ConstInt with INDEX dtype,
        # so always extract the integer value and create a fresh INT32 constant.
        if isinstance(block_len, _ir_core.ConstInt):
            block_len_expr = _ir_core.ConstInt(block_len.value, DataType.INT32, actual_span)
        elif isinstance(block_len, Expr):
            block_len_expr = block_len
        else:
            block_len_expr = _ir_core.ConstInt(block_len, DataType.INT32, actual_span)
        return _ir_core.create_op_call("tile.mrgsort_format1", [src0, block_len_expr], {}, actual_span)
    # format2: 2-4 way merge (pto.tmrgsort format2)
    if src1 is None:
        raise ValueError(
            "mrgsort() requires either block_len=<int> for format1, "
            "or at least (src0, src1, tmp=<tile>) for format2"
        )
    if src2 is None and src3 is not None:
        raise ValueError("mrgsort() format2 requires src2 when src3 is provided")
    if tmp is None:
        raise ValueError(
            "mrgsort() format2 requires tmp to be provided as a keyword argument; "
            "use mrgsort(src0, src1[, src2[, src3]], tmp=<tile>)"
        )
    kwargs: dict[str, Any] = {"exhausted": exhausted}
    if src2 is None:
        # 2-way merge
        args = [src0, src1, tmp]
    elif src3 is None:
        # 3-way merge
        args = [src0, src1, src2, tmp]
    else:
        # 4-way merge
        args = [src0, src1, src2, src3, tmp]
    return _ir_core.create_op_call("tile.mrgsort_format2", args, kwargs, actual_span)


def mrgsort_format1(src0: Expr, block_len: int | Expr, span: Span | None = None) -> Call:
    """Single-list merge sort (format1). Used by the parser for roundtrip fidelity.

    Prefer ``mrgsort(src, block_len=...)`` in user code.
    """
    return mrgsort(src0, block_len=block_len, span=span)


def mrgsort_format2(*args: Expr, exhausted: bool = False, span: Span | None = None) -> Call:
    """2-4 way merge sort (format2). Used by the parser for roundtrip fidelity.

    Positional args: ``(src0, src1[, src2[, src3]], tmp)``
    The last positional arg is always ``tmp``.

    Prefer ``mrgsort(src0, src1[, src2[, src3]], tmp=<tile>)`` in user code.
    """
    if len(args) < 3 or len(args) > 5:
        raise ValueError(
            f"mrgsort_format2() requires 3-5 positional arguments "
            f"(src0, src1[, src2[, src3]], tmp), got {len(args)}"
        )
    srcs = args[:-1]
    tmp = args[-1]
    src0 = srcs[0]
    src1 = srcs[1]
    src2 = srcs[2] if len(srcs) > 2 else None
    src3 = srcs[3] if len(srcs) > 3 else None
    return mrgsort(src0, src1, src2, src3, tmp=tmp, exhausted=exhausted, span=span)
