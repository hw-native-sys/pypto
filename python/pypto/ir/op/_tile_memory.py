# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile allocation, memory transfers, initialization, and views."""

from collections.abc import Sequence
from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import (
    Call,
    ConstInt,
    Expr,
    MemorySpace,
    PadValue,
    Span,
    STPhase,
    TensorLayout,
    TileLayout,
)

from ..utils import (
    _get_span_or_capture,
    _normalize_const_to_dtype,
    _normalize_expr,
    _to_int32_scalar,
    _to_make_tuple,
    resolve_fixpipe_epilogue,
)
from ._pad_value import normalize_pad_value


def _fixpipe_epilogue_kwargs(
    op_name: str, pre_quant: float | Expr | None, pre_relu: bool | Expr
) -> dict[str, Any]:
    """Build the FIXPIPE ``pre_quant`` / ``pre_relu`` op kwargs, or none at all.

    Resolution lives in :func:`resolve_fixpipe_epilogue` so the DSL entry points
    can apply the same rules before dispatching. This adds only the op-kwarg
    shape: an epilogue-free call must produce no kwargs, so ordinary tile IR
    prints and compares exactly as it did before the feature existed, and a
    ``float`` is always stored (the op declares ``set_attr<double>``, which an
    ``int`` would fail to cast).
    """
    scale, relu = resolve_fixpipe_epilogue(op_name, pre_quant, pre_relu)
    kwargs: dict[str, Any] = {}
    if scale is not None:
        kwargs["pre_quant"] = scale
    if relu:
        kwargs["pre_relu"] = True
    return kwargs


def _validate_offsets_shapes(offsets_tuple: _ir_core.MakeTuple, shapes_tuple: _ir_core.MakeTuple) -> None:
    """Validate that offsets and shapes have matching, non-zero dimensions.

    Args:
        offsets_tuple: MakeTuple of offset expressions
        shapes_tuple: MakeTuple of shape expressions

    Raises:
        ValueError: If dimensions don't match or are empty
    """
    if len(offsets_tuple.elements) != len(shapes_tuple.elements):
        raise ValueError(
            f"offsets and shapes must have same number of dimensions, "
            f"got {len(offsets_tuple.elements)} offsets and {len(shapes_tuple.elements)} shapes"
        )
    if len(offsets_tuple.elements) == 0:
        raise ValueError("offsets and shapes must have at least one dimension")


# ============================================================================
# Memory Operations
# ============================================================================


def alloc(
    memory_space: int | Expr,
    addr: int | Expr,
    size: int | Expr,
    alloc_id: int | Expr,
    span: Span | None = None,
) -> Call:
    """Allocate memory for a MemRef object.

    Internal op emitted by InitMemRef / AllocateMemoryAddr passes.

    Args:
        memory_space: Memory space enum value
        addr: Starting address
        size: Size in bytes
        alloc_id: MemRef identifier
        span: Optional source span

    Returns:
        Call node representing the tile.alloc operation
    """
    actual_span = _get_span_or_capture(span)
    args = [
        _normalize_expr(memory_space, actual_span),
        _normalize_expr(addr, actual_span),
        _normalize_expr(size, actual_span),
        _normalize_expr(alloc_id, actual_span),
    ]
    return _ir_core.create_op_call("tile.alloc", args, {}, actual_span)


def create(
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    dtype: DataType,
    target_memory: MemorySpace | None = None,
    transpose: bool | None = None,
    span: Span | None = None,
    *,
    flat_layout: bool | None = None,
    compact: bool | None = None,
) -> Call:
    """Create a tile from a shape.

    Args:
        shape: Shape of the tile, or a MakeTuple
        dtype: Data type of the tile
        target_memory: Target memory space (MemorySpace.Vec, .Mat, .Left, .Right).
            ``None`` (the default) leaves the space unset so InferTileMemorySpace
            places the tile from consumer demand; the kwarg is then omitted from
            the op entirely.
        transpose: When True, allocate the transposed Mat (ZN) fractal layout
            (blayout=row_major, slayout=col_major) — the layout a matmul ``b_trans``
            B-operand carries, and the only Mat layout a DN-source ``gather_row``
            (DN2NZ tload) can fill. Default ``None`` keeps the canonical layout and
            is omitted from the op kwargs, so ordinary ``tile.create`` output is
            unchanged; only forwarded to the op when explicitly set.
        span: Optional source span for debugging (auto-captured if not provided)
        flat_layout: Keyword-only. When True, allocate a flat (non-fractal,
            slayout=none_box) L1/cbuf tile — a contiguous byte-staging buffer
            rather than the boxed NZ layout Mat tiles normally carry. Requires
            ``target_memory=Mat`` and is mutually exclusive with ``transpose``.
            Default ``None`` keeps the canonical layout. Kept keyword-only so
            it does not shift ``span``'s positional slot for existing callers.
        compact: Keyword-only. Compiler-internal. When True, declare that this
            L0C buffer holds a valid-region-packed product, i.e. that its
            N-fractal pitch is ``ceil(validRow/16)*16`` rather than the physical
            row count -- the layout ``mad`` writes when the matmul's left operand
            is row-narrowed. Requires ``target_memory=Acc``. Kernels do not set
            this: ``AutoTileMatmulL0`` declares it on the accumulator seed it
            synthesizes, and every reader of that accumulator inherits it.

    Returns:
        Call expression that returns a TileType with the created tile
    """
    actual_span = _get_span_or_capture(span)
    shape_tuple = _to_make_tuple(shape, actual_span)
    kwargs: dict[str, Any] = {"dtype": dtype}
    if target_memory is not None:
        kwargs["target_memory"] = target_memory
    if transpose is not None:
        kwargs["transpose"] = transpose
    if flat_layout is not None:
        kwargs["flat_layout"] = flat_layout
    if compact is not None:
        kwargs["compact"] = compact
    return _ir_core.create_op_call("tile.create", [shape_tuple], kwargs, actual_span)


create_tile = create


def load(
    tensor: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple,
    valid_shape: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    target_memory: MemorySpace | None = None,
    clamp: bool = False,
    span: Span | None = None,
    cache: int | None = None,
) -> Call:
    """Copy data from tensor to specified memory level.

    Only the valid extent is read, so the destination tile may be larger than the
    region that exists in the source. The tile's valid region is the source's
    valid region, shifted by ``offsets`` and cut to the tile — a load can never
    report as real data bytes the source does not have.

    Args:
        tensor: Source tensor (TensorType)
        offsets: Offsets in each dimension (sequence of scalars), or a MakeTuple.
            Always in the source tensor's coordinate system.
        shapes: Shape of the region to load in each dimension (sequence of scalars),
            or a MakeTuple. Always in the source tensor's coordinate system. Every
            element must be an integer scalar, as for ``offsets``.
        valid_shape: Valid shape of the tile in each dimension (sequence of scalars), or a
            MakeTuple. When provided, sets TileView.valid_shape in the output TileType.
            When omitted, shapes is used as valid_shape. Useful for dynamic shapes where
            the actual valid data region differs from the allocated tile size.
            Uses the same coordinate convention as shapes. This is a *request*: it
            narrows the tile, but cannot widen it past what the source has. Every
            element must be an integer scalar — one extent per dimension, never a
            nested tuple.
        target_memory: Target memory space (MemorySpace.Vec or MemorySpace.Mat).
            ``None`` (the default) leaves an ordinary load unset so
            InferTileMemorySpace places the tile from consumer demand. An
            MX-layout load instead defaults to MemorySpace.Mat, the only memory
            that the raw MX load instruction can target.
        clamp: Sanction a read that runs off the end of the source. By default a
            load asserts that ``offsets + valid_shape`` stays inside the source
            and is rejected when that provably fails; with ``clamp=True`` the
            request is cut back to the source edge instead.
        span: Optional source span for debugging (auto-captured if not provided)
        cache: Semantic ``CachePolicy`` integer: DEFAULT=0, BYPASS=1, or an
            explicit A5 policy. These are not ISA bit encodings. ``None`` (the
            default) means the caller stated no policy and omits the kwarg, so
            ordinary loads are unchanged and a scope-level declaration may still
            stamp one later. An explicit 0 is NOT the same as ``None``: it is
            recorded, and it is what makes ``cache=CachePolicy.DEFAULT`` opt a
            single read back into the cache inside a bypassing scope.

    Returns:
        Call expression that returns a TileType with the copied data

    Example:
        >>> # 2D load
        >>> tile = load(tensor, offsets=[0, 0], shapes=[32, 32])
    """
    tensor_view = getattr(tensor.type, "tensor_view", None)
    source_layout = getattr(tensor_view, "layout", None)
    is_mx = source_layout in (TensorLayout.MX_A_ZZ, TensorLayout.MX_B_NN)

    if is_mx and target_memory is None:
        target_memory = MemorySpace.Mat
    elif is_mx and target_memory != MemorySpace.Mat:
        raise ValueError(
            "tile.load of an MX-layout tensor only supports target_memory=MemorySpace.Mat "
            f"(omitting target_memory selects Mat automatically); got {target_memory}"
        )

    # Validate target_memory: only Vec and Mat are allowed for load. ``None``
    # leaves the space unset so InferTileMemorySpace places the tile.
    if target_memory is not None and target_memory not in (MemorySpace.Vec, MemorySpace.Mat):
        raise ValueError(
            f"target_memory for tile.load must be MemorySpace.Vec or MemorySpace.Mat, got {target_memory}"
        )

    actual_span = _get_span_or_capture(span)

    offsets_tuple = _to_make_tuple(offsets, actual_span)
    shapes_tuple = _to_make_tuple(shapes, actual_span)
    _validate_offsets_shapes(offsets_tuple, shapes_tuple)

    kwargs: dict[str, Any] = {}
    if target_memory is not None:
        kwargs["target_memory"] = target_memory
    if clamp:
        kwargs["clamp"] = True
    # `is not None`, not truthiness: an explicit `cache=0` (DEFAULT) is a real
    # per-access override that must out-rank a scope declaration, so it has to
    # survive into the IR. Only an unstated policy omits the kwarg.
    if cache is not None:
        kwargs["cache"] = cache

    valid_shape_tuple = shapes_tuple
    if valid_shape is not None:
        valid_shape_tuple = _to_make_tuple(valid_shape, actual_span)
        if len(valid_shape_tuple.elements) != len(shapes_tuple.elements):
            raise ValueError(
                "valid_shape and shapes must have same number of dimensions, "
                f"got {len(valid_shape_tuple.elements)} valid_shape dimensions "
                f"and {len(shapes_tuple.elements)} shapes"
            )
    return _ir_core.create_op_call(
        "tile.load",
        [tensor, offsets_tuple, shapes_tuple, valid_shape_tuple],
        kwargs,
        actual_span,
    )


def store(
    tile: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    output_tensor: Expr,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    span: Span | None = None,
    *,
    atomic: int = 0,
    st_phase: STPhase = STPhase.Unspecified,
    pre_quant: float | Expr | None = None,
    pre_relu: bool | Expr = False,
    cache: int | None = None,
) -> Call:
    """Copy data from unified buffer (tile) to tensor.

    Args:
        tile: Source tile (TileType)
        offsets: Offsets in each dimension (sequence of scalars), or a MakeTuple
        output_tensor: Output tensor (TensorType)
        shapes: ND partition shape (sequence of ints), or None for 2D tiles. Normally
            injected automatically by FlattenTileNdTo2D for ND tensors. Every element
            must be an integer scalar, as for ``offsets``.
        span: Optional source span for debugging (auto-captured if not provided)
        atomic: ``AtomicType`` underlying int — 0 (``kNone``, plain overwrite) or
            1 (``kAdd``, atomic-add into global memory). The kwarg is omitted
            entirely when 0 so non-atomic stores are unchanged.
        st_phase: Unit-flag-aware store phase. ``STPhase.Unspecified`` (default)
            preserves ordinary stores; ``STPhase.Final`` checks and clears the
            flag published by a final phased accumulator producer. The kwarg is
            omitted entirely for the default so existing stores remain unchanged.
        pre_quant: FIXPIPE pre-quantization scale for an ``Acc`` source. The
            accumulator is multiplied by it in FP32 on the way out, before the
            destination clamp, which is what lets an INT32 accumulator reach an
            FP16 or INT8 tensor without a vector round-trip. ``None`` (default)
            emits no scale at all — distinct from ``1.0``, which still selects
            the quantizing instruction form.
        pre_relu: Apply ReLU inside the same writeback, *after* ``pre_quant`` and
            the clamp — it reproduces ``maximum(tile * pre_quant, 0)``.

        cache: Optional semantic ``CachePolicy`` encoding.

    Returns:
        Call expression that returns the output tensor
    """
    actual_span = _get_span_or_capture(span)
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    if shapes is not None:
        args: list[Expr] = [tile, offsets_tuple, output_tensor, _to_make_tuple(shapes, actual_span)]
    else:
        args = [tile, offsets_tuple, output_tensor]

    kwargs: dict[str, Any] = {"atomic": atomic} if atomic else {}
    if cache is not None:
        kwargs["cache"] = cache
    if st_phase != STPhase.Unspecified:
        kwargs["st_phase"] = int(st_phase)
    kwargs.update(_fixpipe_epilogue_kwargs("tile.store", pre_quant, pre_relu))
    return _ir_core.create_op_call("tile.store", args, kwargs, actual_span)


def assemble(
    target: Expr,
    source: Expr,
    offset: Sequence[int | Expr] | _ir_core.MakeTuple,
    span: Span | None = None,
    *,
    pre_quant: float | Expr | None = None,
    pre_relu: bool | Expr = False,
) -> Call:
    """Write source tile data into target tile at specified offset.

    Args:
        target: Target tile (TileType)
        source: Source tile to write (TileType)
        offset: Offset dimensions for where to write, or a MakeTuple
        span: Optional source span for debugging (auto-captured if not provided)
        pre_quant: FIXPIPE pre-quantization scale for an ``Acc``-to-``Mat``
            writeback. The accumulator is multiplied by it in FP32 on the way
            out, before the destination clamp, which is what lets an INT32
            accumulator land in an FP16 Mat tile the next matmul can read.
            ``None`` (default) emits no scale at all — distinct from ``1.0``,
            which still selects the quantizing instruction form.
        pre_relu: Apply ReLU inside the same writeback, *after* ``pre_quant`` and
            the clamp — it reproduces ``maximum(tile * pre_quant, 0)``.

    Returns:
        Call expression that returns a TileType with the same shape/dtype as target
    """
    actual_span = _get_span_or_capture(span)
    offset_tuple = _to_make_tuple(offset, actual_span)

    kwargs = _fixpipe_epilogue_kwargs("tile.assemble", pre_quant, pre_relu)
    return _ir_core.create_op_call("tile.assemble", [target, source, offset_tuple], kwargs, actual_span)


def gather_row(  # noqa: PLR0913
    dst: Expr,
    src: Expr,
    dst_offset: Sequence[int | Expr] | _ir_core.MakeTuple,
    src_offset: Sequence[int | Expr] | _ir_core.MakeTuple,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple,
    transpose: bool = False,
    span: Span | None = None,
    *,
    valid_shape: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
) -> Call:
    """Load one GM row directly into a sub-region of an on-chip (Mat/Vec) tile.

    Per-row primitive of the paged-gather lowering: emits ``pto.subview`` (of
    ``dst``) + ``pto.partition_view`` (of ``src``) + ``pto.tload`` writing
    GM -> the subview directly (no ``pto.tmov``). DPS — writes into ``dst`` in
    place.

    Args:
        dst: Destination accumulator tile (Mat or Vec).
        src: Source tensor in GM (TensorType).
        dst_offset: ``[row, col]`` offset within ``dst``, or a MakeTuple.
        src_offset: ``[row, col]`` offset within the GM ``src``, or a MakeTuple.
        shapes: GM row window shape ``[r, c]``, or a MakeTuple. Must be
            compile-time constant.
        valid_shape: Runtime transfer extent within ``shapes``, or a MakeTuple.
            May hold runtime ``Scalar[INDEX]`` values. Defaults to ``shapes``.
        transpose: Place the GM row ``[r, c]`` as an L1 column ``[c, r]``.
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression returning ``dst``'s TileType (written in place).
    """
    actual_span = _get_span_or_capture(span)
    dst_off = _to_make_tuple(dst_offset, actual_span)
    src_off = _to_make_tuple(src_offset, actual_span)
    shapes_tuple = _to_make_tuple(shapes, actual_span)
    args: list[Any] = [dst, src, dst_off, src_off, shapes_tuple]
    if valid_shape is not None:
        args.append(_to_make_tuple(valid_shape, actual_span))
    return _ir_core.create_op_call("tile.gather_row", args, {"transpose": transpose}, actual_span)


def concat(
    src0: Expr,
    src1: Expr,
    span: Span | None = None,
) -> Call:
    """Concatenate two tiles along the column dimension.

    Args:
        src0: First source tile (TileType)
        src1: Second source tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise concatenation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.concat", [src0, src1], {}, actual_span)


def transpose_view(
    tile: Expr,
    span: Span | None = None,
) -> Call:
    """Zero-copy fractal-layout reinterpretation (NZ<->ZN) of a tile.

    Swaps the trailing two dims together with the block/scatter layouts, aliasing
    the source buffer byte-for-byte (an NZ ``[..., N, K]`` tile and a ZN
    ``[..., K, N]`` tile over the same L1 bytes are mutual transposes). Emits no
    data movement: it lets one GM->L1 load feed both a ``b_trans=True`` and a
    ``b_trans=False`` matmul on a shared operand.

    Args:
        tile: Input tile (TileType, >=2D; typically Mat-resident)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression returning the transposed-layout view tile
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.transpose_view", [tile], {}, actual_span)


def move(
    tile: Expr,
    target_memory: MemorySpace,
    blayout: TileLayout | None = None,
    slayout: TileLayout | None = None,
    span: Span | None = None,
) -> Call:
    """Move tile between memory levels.

    Args:
        tile: Input tile (TileType)
        target_memory: Target memory space (MemorySpace.Vec, .Mat, .Left, .Right,
            .LeftScale, .RightScale)
        blayout: Optional block layout for the destination tile
        slayout: Optional scatter layout for the destination tile
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType in the target memory space
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {
        "target_memory": target_memory,
    }
    if blayout is not None:
        kwargs["blayout"] = blayout
    if slayout is not None:
        kwargs["slayout"] = slayout

    return _ir_core.create_op_call("tile.move", [tile], kwargs, actual_span)


def full(
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    dtype: DataType,
    value: int | float | Expr,
    span: Span | None = None,
) -> Call:
    """Create a tile from a shape and fill with value in UB.

    Args:
        shape: Shape of the tile, or a MakeTuple
        dtype: Data type of the tile
        value: Numeric fill value or parsed scalar constant (runtime values are not supported)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType with the created tile
    """
    actual_span = _get_span_or_capture(span)
    shape_tuple = _to_make_tuple(shape, actual_span)
    value_expr = _normalize_const_to_dtype(value, dtype, actual_span)
    kwargs: dict[str, Any] = {"dtype": dtype}
    return _ir_core.create_op_call("tile.full", [shape_tuple, value_expr], kwargs, actual_span)


def ci(
    start: int | Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    dtype: DataType = DataType.INT32,
    descending: bool = False,
    span: Span | None = None,
    *,
    tmp: Expr | None = None,
) -> Call:
    """Generate a contiguous integer sequence into a tile (pto.tci).

    For a column index ``k`` in the first row of the destination tile:
    - Ascending: ``dst[0, k] = start + k``
    - Descending: ``dst[0, k] = start - k``

    Note:
        ``pto.tci`` uses the destination's valid-column count as the sequence
        length and does NOT populate additional rows. Leading dimensions must
        be 1 — prefer shapes of the form ``[1, N]``.

    Args:
        start: Starting integer (plain int or a scalar Expr). Its dtype must match ``dtype``.
        shape: Destination tile shape (static, leading dims must be 1, innermost dim != 1).
        dtype: Destination dtype. Must be one of {INT16, INT32}.
        descending: If True, generate a descending sequence.
        span: Optional source span for debugging (auto-captured if not provided).
        tmp: Optional A2/A3 PTOAS scratch tile. Normally compiler-generated.

    Returns:
        Call expression that returns a TileType with the generated sequence.
    """
    actual_span = _get_span_or_capture(span)
    if isinstance(start, Expr):
        if isinstance(start, ConstInt) and start.dtype != dtype:
            start_expr = ConstInt(start.value, dtype, actual_span)
        else:
            start_expr = start
    else:
        start_expr = ConstInt(start, dtype, actual_span)
    shape_tuple = _to_make_tuple(shape, actual_span)
    kwargs: dict[str, Any] = {"dtype": dtype, "descending": descending}
    args = [start_expr, shape_tuple]
    if tmp is not None:
        args.append(tmp)
    return _ir_core.create_op_call("tile.ci", args, kwargs, actual_span)


arange = ci


def tri(
    diagonal: int | Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    valid_shape: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    dtype: DataType = DataType.INT32,
    upper: bool = False,
    span: Span | None = None,
) -> Call:
    """Generate a lower- or upper-triangular mask tile (``pto.ttri``).

    Args:
        diagonal: INT32 diagonal offset, matching ``torch.tril``/``torch.triu``.
        shape: Static two-dimensional physical destination shape.
        valid_shape: Optional written region, bounded by ``shape``.
        dtype: One of INT16, INT32, UINT16, UINT32, FP16, or FP32.
        upper: Generate the upper triangle when true; lower otherwise.
        span: Optional source span.
    """
    actual_span = _get_span_or_capture(span)
    if isinstance(diagonal, Expr):
        if isinstance(diagonal, ConstInt) and diagonal.dtype != DataType.INT32:
            diagonal_expr: Expr = ConstInt(diagonal.value, DataType.INT32, actual_span)
        else:
            diagonal_expr = diagonal
    else:
        diagonal_expr = ConstInt(diagonal, DataType.INT32, actual_span)
    args: list[Expr] = [diagonal_expr, _to_make_tuple(shape, actual_span)]
    if valid_shape is not None:
        args.append(_to_make_tuple(valid_shape, actual_span))
    return _ir_core.create_op_call(
        "tile.tri",
        args,
        {"dtype": dtype, "upper": upper},
        actual_span,
    )


def random(  # noqa: PLR0913
    key0: int | Expr,
    key1: int | Expr,
    counter0: int | Expr,
    counter1: int | Expr,
    counter2: int | Expr,
    counter3: int | Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    valid_shape: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    dtype: DataType = DataType.UINT32,
    rounds: int = 10,
    span: Span | None = None,
) -> Call:
    """Generate counter-based pseudo-random values into a tile (pto.trandom).

    Implements a counter-based (Philox/ChaCha-style) RNG: each destination
    element is derived deterministically from the 64-bit key ``(key0, key1)`` and
    the 128-bit counter ``(counter0..counter3)`` plus the element position, so the
    same seeds always produce the same tile.

    Args:
        key0, key1: The two INT32 key words.
        counter0, counter1, counter2, counter3: The four INT32 counter words.
        shape: Destination tile shape (static, tuple of ConstInt).
        valid_shape: Optional written region (tuple of ConstInt, each ``<= shape``).
            ``pto.trandom`` only fills the dst valid rows/cols; defaults to the full shape.
        dtype: Destination dtype. One of {INT32, UINT32}. Defaults to UINT32.
        rounds: Cipher round count, 7 or 10. Defaults to 10.
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression that returns a TileType filled with random values.
    """
    actual_span = _get_span_or_capture(span)
    seeds = [_to_int32_scalar(v, actual_span) for v in (key0, key1, counter0, counter1, counter2, counter3)]
    op_args: list[Expr] = [*seeds, _to_make_tuple(shape, actual_span)]
    if valid_shape is not None:
        op_args.append(_to_make_tuple(valid_shape, actual_span))
    kwargs: dict[str, Any] = {"dtype": dtype, "rounds": rounds}
    return _ir_core.create_op_call("tile.random", op_args, kwargs, actual_span)


def fillpad(tile: Expr, pad_value: PadValue | int | float = PadValue.zero, span: Span | None = None) -> Call:
    """Fill remaining tile elements with specified padding value.

    Args:
        tile: Input tile (TileType)
        pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
            the literal sugars ``0``, ``math.inf``, ``-math.inf``. Default is
            ``PadValue.zero``. Other values raise — the hardware only supports
            the three padding modes.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns the filled and padded tile
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "tile.fillpad", [tile], {"pad_value": normalize_pad_value(pad_value)}, actual_span
    )


def fillpad_inplace(
    tile: Expr, pad_value: PadValue | int | float = PadValue.zero, span: Span | None = None
) -> Call:
    """Fill padding elements of input tile in place with specified pad value.

    Unlike fillpad which returns a new tile, this operation mutates the input
    tile in place. The valid data region is unchanged; only out-of-bounds
    (padding) elements are written.

    Args:
        tile: Input tile (TileType)
        pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
            the literal sugars ``0``, ``math.inf``, ``-math.inf``. Default is
            ``PadValue.zero``. Other values raise — the hardware only supports
            the three padding modes.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression (result typically discarded since op is in-place)
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "tile.fillpad_inplace", [tile], {"pad_value": normalize_pad_value(pad_value)}, actual_span
    )


def fillpad_expand(
    tile: Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    pad_value: PadValue | int | float = PadValue.zero,
    span: Span | None = None,
) -> Call:
    """Copy a smaller source tile into a larger destination tile, padding the rest.

    Unlike :func:`fillpad` (which requires ``dst.shape == src.shape``), this op
    allows the destination to be larger than the source in either dimension. The
    source's valid region is copied into the top-left of the destination and all
    other destination elements are filled with ``pad_value``.

    Args:
        tile: Source tile (TileType)
        shape: Destination shape; each dimension must be >= the source dimension
        pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
            the literal sugars ``0``, ``math.inf``, ``-math.inf``. Default is
            ``PadValue.zero``. Other values raise — the hardware only supports
            the three padding modes.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns the expanded and padded tile
    """
    actual_span = _get_span_or_capture(span)
    shape_tuple = _to_make_tuple(shape, actual_span)
    return _ir_core.create_op_call(
        "tile.fillpad_expand",
        [tile, shape_tuple],
        {"pad_value": normalize_pad_value(pad_value)},
        actual_span,
    )


def read(tile: Expr, indices: Expr | list[int | Expr] | _ir_core.MakeTuple, span: Span | None = None) -> Call:
    """Read a scalar value from a tile at given indices.

    Args:
        tile: Input tile expression
        indices: A single index expression (for 1-D flat access), a list of index
            expressions (one per tile dimension), or a MakeTuple
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression reading a scalar from the tile
    """
    actual_span = _get_span_or_capture(span)

    # Allow a bare Expr as a flat 1-D index for backwards compatibility
    if isinstance(indices, Expr) and not isinstance(indices, _ir_core.MakeTuple):
        indices = [indices]

    indices_tuple = _to_make_tuple(indices, actual_span)

    args = [tile, indices_tuple]
    return _ir_core.create_op_call("tile.read", args, {}, actual_span)


def write(
    tile: Expr,
    indices: Expr | list[int | Expr] | _ir_core.MakeTuple,
    value: Expr,
    span: Span | None = None,
) -> Call:
    """Write a scalar value into a tile at given indices.

    Args:
        tile: Destination tile expression (TileType)
        indices: A single index expression (for 1-D flat access), a list of index
            expressions (one per tile dimension), or a MakeTuple
        value: Scalar value to write (ScalarType, must match tile dtype)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression returning the tile (for chaining)
    """
    actual_span = _get_span_or_capture(span)

    # Allow a bare Expr as a flat 1-D index for backwards compatibility
    if isinstance(indices, Expr) and not isinstance(indices, _ir_core.MakeTuple):
        indices = [indices]

    indices_tuple = _to_make_tuple(indices, actual_span)

    args = [tile, indices_tuple, value]
    return _ir_core.create_op_call("tile.write", args, {}, actual_span)


# ============================================================================
# Transform Operations
# ============================================================================


def slice(
    tile: Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    offset: Sequence[int | Expr] | _ir_core.MakeTuple,
    valid_shape: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    drop_dims: Sequence[int | Expr] | None = None,
    pad_value: PadValue | int | float | None = None,
    span: Span | None = None,
) -> Call:
    """Create a slice of a tile with static shape and optional valid shape.

    The result is never valid where the source tile is not: its valid region is
    the source's valid region, shifted by ``offset`` and cut to the window.

    Args:
        tile: Input tile expression
        shape: Static shape dimensions, or a MakeTuple. Always full-rank — a
            scalar-indexed axis contributes a unit dim here and is listed in
            ``drop_dims`` to be erased from the result type.
        offset: Offset dimensions for the slice, or a MakeTuple
        valid_shape: Valid shape dimensions, or a MakeTuple. When omitted, shape
            is reused as the valid shape. This is a *request*: it narrows the
            result, but cannot widen it past what the source has under the window.
        drop_dims: Optional axes to erase from the result type (numpy-style rank
            reduction). Each listed axis must be a static unit dim of ``shape``,
            and must still be fully valid after the intersection above.
            Because tiles are physically 2D, the result is clamped back to 2D
            (unit axes prepended) if reduction would take it below 2D.
            ``None`` / ``[]`` is fully backward compatible (drops nothing).
        pad_value: Optional padding mode for out-of-valid-shape elements.
            Accepts ``PadValue.zero`` / ``PadValue.max`` / ``PadValue.min``, or
            the literal sugars ``0``, ``math.inf``, ``-math.inf`` (normalized
            via :func:`normalize_pad_value`). ``PadValue.null`` is passed
            through unchanged and means "no padding". When omitted (``None``),
            the source's padding mode carries through.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating a tile slice

    Note:
        Unlike :func:`pypto.ir.op.tensor.slice`, there is no ``clamp`` option: an
        on-chip window has nothing that could clamp it, so ``offset + shape`` must
        stay inside the source tile.
    """
    actual_span = _get_span_or_capture(span)

    shape_tuple = _to_make_tuple(shape, actual_span)
    offset_tuple = _to_make_tuple(offset, actual_span)
    args = [tile, shape_tuple, offset_tuple]

    valid_shape_tuple = None
    if valid_shape is not None:
        valid_shape_tuple = _to_make_tuple(valid_shape, actual_span)
        # An empty tuple is the explicit "no valid_shape" form (used when only
        # drop_dims is supplied) — skip the rank check for it.
        if len(valid_shape_tuple.elements) not in (0, len(shape_tuple.elements)):
            raise ValueError(
                f"valid_shape and shape must have same number of dimensions, "
                "got "
                f"{len(valid_shape_tuple.elements)} valid_shape dims and "
                f"{len(shape_tuple.elements)} shape dims"
            )

    if drop_dims:
        # drop_dims is the 5th positional operand, so valid_shape (the 4th) must
        # be present; an empty MakeTuple stands in for "no valid_shape".
        args.append(valid_shape_tuple if valid_shape_tuple is not None else _to_make_tuple([], actual_span))
        # `drop_dims` may be ints (direct API) or ConstInt exprs (text parser);
        # _to_make_tuple normalizes either form. Non-ConstInt exprs are rejected
        # by the deducer with a clear message.
        args.append(_to_make_tuple(list(drop_dims), actual_span))
    elif valid_shape_tuple is not None:
        args.append(valid_shape_tuple)

    kwargs: dict[str, Any] = {}
    if pad_value is not None:
        # PadValue.null is a legal "no padding" signal for slice (unlike
        # fillpad, which requires a real padding mode). Pass it through;
        # normalize the rest via the shared helper so numeric sugar and
        # validation match tile.fillpad exactly.
        kwargs["pad_value"] = pad_value if pad_value is PadValue.null else normalize_pad_value(pad_value)

    return _ir_core.create_op_call("tile.slice", args, kwargs, actual_span)


def extract(
    src: Expr,
    index_row: int | Expr,
    index_col: int | Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    target_memory: MemorySpace,
    span: Span | None = None,
) -> Call:
    """Extract a sub-tile from src at (index_row, index_col).

    Maps to ISA TEXTRACT Variant 1 (Standard Extract). The result tile has the
    given static ``shape`` and lives in ``target_memory``.

    Args:
        src: Source tile expression (TileType, 2D)
        index_row: Starting row offset (int or Expr)
        index_col: Starting col offset (int or Expr)
        shape: Static destination shape (2-element sequence or MakeTuple of ConstInt)
        target_memory: Destination memory space (Left/Right for Mat sources, Mat for Acc sources)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression with TileType[shape, src.dtype] in target_memory
    """
    actual_span = _get_span_or_capture(span)
    shape_tuple = _to_make_tuple(shape, actual_span)
    row_expr = _normalize_expr(index_row, actual_span, int_dtype=DataType.INDEX)
    col_expr = _normalize_expr(index_col, actual_span, int_dtype=DataType.INDEX)
    return _ir_core.create_op_call(
        "tile.extract",
        [src, row_expr, col_expr, shape_tuple],
        {"target_memory": target_memory},
        actual_span,
    )


def reshape(
    tile: Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    span: Span | None = None,
) -> Call:
    """Reshape tile to new shape.

    Args:
        tile: Input tile expression
        shape: New shape dimensions, or a MakeTuple
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tile reshape
    """
    actual_span = _get_span_or_capture(span)

    shape_tuple = _to_make_tuple(shape, actual_span)

    args = [tile, shape_tuple]
    return _ir_core.create_op_call("tile.reshape", args, {}, actual_span)


def reinterpret_view(
    data: Expr,
    dtype: DataType,
    *,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    span: Span | None = None,
) -> Call:
    """Reinterpret a tile over the same bytes with a different dtype.

    Either packed FP4 format <-> UINT8/INT8 is a same-shape byte alias for dense
    address-backed 2D row-major Vec tiles with static UB offsets. Reshape
    the byte view separately. A5 codegen requires PYPTO / DSA_RP planning.

    Args:
        data: Input tile expression
        dtype: Target element dtype
        shape: Optional target shape. When omitted, the physically contiguous
            dimension is scaled according to the source/target dtype byte ratio
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for a byte-preserving tile reinterpret view
    """
    actual_span = _get_span_or_capture(span)
    args = [data]
    if shape is not None:
        args.append(_to_make_tuple(shape, actual_span))
    return _ir_core.create_op_call("tile.reinterpret_view", args, {"dtype": dtype}, actual_span)


def _normalize_axis_const(axis: object, span: Span, name: str) -> ConstInt:
    """Normalize an axis argument (int or ConstInt) into a ConstInt(INDEX) expression."""
    if isinstance(axis, ConstInt):
        return axis
    if isinstance(axis, int):
        return ConstInt(axis, DataType.INDEX, span)
    raise TypeError(f"{name} must be int or ConstInt, got {type(axis)}")


def transpose(
    tile: Expr,
    axis1: int | ConstInt,
    axis2: int | ConstInt,
    tmp: Expr | None = None,
    span: Span | None = None,
) -> Call:
    """Transpose tile by swapping two axes.

    The ``pto.ttrans`` scratch buffer is a codegen detail, not a semantic operand:
    ``FlattenTileNdTo2D`` materializes it (the codegen-ready 4-arg form) for both 2D and
    per-page >2D transposes. High-level callers omit ``tmp`` and get the 3-arg form; the
    optional ``tmp`` exists only so the lowered 4-arg form round-trips through the
    printer/parser. It is never auto-created here.

    Args:
        tile: Input tile expression (must be TileType).
        axis1: First axis to swap (supports negative indexing).
        axis2: Second axis to swap (supports negative indexing).
        tmp: Optional pre-allocated scratch tile — compiler-generated lowered IR only.
        span: Optional source span (auto-captured if not provided).

    Returns:
        Call expression for tile transpose (operands: input, axis1, axis2[, tmp]).
    """
    actual_span = _get_span_or_capture(span)
    axis1_expr = _normalize_axis_const(axis1, actual_span, "axis1")
    axis2_expr = _normalize_axis_const(axis2, actual_span, "axis2")

    args: list[Expr] = [tile, axis1_expr, axis2_expr]
    if tmp is not None:
        args.append(tmp)
    return _ir_core.create_op_call("tile.transpose", args, {}, actual_span)


def set_validshape(
    tile: Expr,
    valid_rows: int | Expr,
    valid_cols: int | Expr,
    span: Span | None = None,
) -> Call:
    """Update valid-shape metadata of a tile without data movement.

    .. note::
        The operand must not be a view (a ``pl.tile.slice`` or reshape result): a
        view carries its valid extent in its type, so there is nothing to update.
        Narrow at the slice with ``valid_shape=`` instead.

    Args:
        tile: Input tile expression (must be 2D TileType)
        valid_rows: Number of valid rows (int or Scalar INDEX expression)
        valid_cols: Number of valid columns (int or Scalar INDEX expression)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tile.set_validshape
    """
    actual_span = _get_span_or_capture(span)
    vr_expr = (
        valid_rows if isinstance(valid_rows, Expr) else ConstInt(valid_rows, DataType.INDEX, actual_span)
    )
    vc_expr = (
        valid_cols if isinstance(valid_cols, Expr) else ConstInt(valid_cols, DataType.INDEX, actual_span)
    )
    return _ir_core.create_op_call("tile.set_validshape", [tile, vr_expr, vc_expr], {}, actual_span)
