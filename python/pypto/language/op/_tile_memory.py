# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile allocation, memory transfers, initialization, and views."""

import warnings
from collections.abc import Sequence

from pypto.ir.op import tile_ops as _ir_ops
from pypto.ir.utils import (
    caller_warning_stacklevel,
    has_partial_valid_region,
)
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import (
    AtomicType,
    CachePolicy,
    Expr,
    MemorySpace,
    PadValue,
    STPhase,
    TileLayout,
)

from ..typing import IntLike, Ptr, Scalar, Tensor, Tile
from ._tile_common import _TensorT
from ._utils import _normalize_intlike


def alloc(
    memory_space: MemorySpace,
    size: int,
    *,
    pinned: bool = False,
) -> Ptr:
    """Stub for the internal ``tile.alloc`` IR operation.

    This function is never called in user-written DSL code.  It is emitted
    by the C++ python-printer after the InitMemRef / AllocateMemoryAddr
    passes and must be importable so that the printed source is valid Python
    that the text-parser can ``exec()``.

    The result is a base ``Ptr`` (allocation identity token): the printer
    annotates the assignment target as ``pl.Ptr``, matching the IR design
    where ``tile.alloc`` / ``tensor.alloc`` Calls carry ``PtrType``.

    Args:
        memory_space: Target memory space (e.g. ``pl.Mem.Vec``)
        size: Allocation size in bytes
        pinned: True when the author declared this allocation via a
            one-argument ``pl.MemRef(...)``. PyPTO memory planners then keep
            its membership isolated from other allocations.

    Returns:
        Opaque ``Ptr`` sentinel (unused at runtime — the parser intercepts the
        call in the AST and never invokes this stub)
    """
    return Ptr()


def create(
    shape: Sequence[IntLike],
    dtype: DataType,
    target_memory: MemorySpace | None = None,
    transpose: bool | None = None,
    *,
    flat_layout: bool | None = None,
    compact: bool | None = None,
) -> Tile:
    """Create a tile from a shape.

    Args:
        shape: Shape of the tile
        dtype: Data type of the tile
        target_memory: Target memory space (MemorySpace.Vec, .Mat, .Left, .Right).
            ``None`` (the default) leaves the space unset for the compiler to place.
        transpose: When True, allocate the transposed Mat (ZN) fractal layout for a
            matmul ``b_trans`` B-operand (the layout a DN-source ``gather_row`` fills).
            Default ``None`` keeps the canonical layout and is omitted from the op.
        flat_layout: Keyword-only. When True, allocate a flat (non-fractal,
            slayout=none_box) L1/cbuf tile — a contiguous staging buffer rather
            than the boxed NZ layout Mat tiles normally carry. Requires
            ``target_memory=Mat`` and is mutually exclusive with ``transpose``.
            Default ``None`` keeps the canonical layout.
        compact: Keyword-only. Compiler-internal. Declares that this L0C buffer
            holds a valid-region-packed product -- N-fractal pitch
            ``ceil(validRow/16)*16`` rather than the physical row count, which is
            what ``mad`` writes when the matmul's left operand is row-narrowed.
            Requires ``target_memory=Acc``. Kernels do not set this;
            ``AutoTileMatmulL0`` declares it on the accumulator seed it
            synthesizes for a split K.

    Returns:
        Tile wrapping the create operation
    """
    # create C++ binding accepts Sequence[int]; Expr elements from Scalar
    # unwrapping are valid at DSL parse time (parser reads the AST).
    call_expr = _ir_ops.create(
        _normalize_intlike(shape),
        dtype,
        target_memory,
        transpose,
        flat_layout=flat_layout,
        compact=compact,
    )
    return Tile(expr=call_expr)


create_tile = create


def read(tile: Tile, indices: IntLike | Sequence[IntLike]) -> Scalar:
    """Read a scalar value from a tile at given indices.

    Args:
        tile: Input tile
        indices: A single index expression (for 1-D flat access) or a list of
            index expressions (one per tile dimension)

    Returns:
        Scalar wrapping the read operation
    """
    # Allow a bare IntLike as a flat 1-D index for backwards compatibility
    indices_seq: Sequence[IntLike] = [indices] if not isinstance(indices, Sequence) else indices
    call_expr = _ir_ops.read(tile.unwrap(), _normalize_intlike(indices_seq))
    return Scalar(expr=call_expr)


def write(tile: Tile, indices: IntLike | Sequence[IntLike], value: Scalar | Expr) -> Expr:
    """Write a scalar value into a tile at given indices.

    Args:
        tile: Destination tile
        indices: A single index expression (for 1-D flat access) or a list of
            index expressions (one per tile dimension)
        value: Scalar value to write (DSL Scalar or raw Expr)

    Returns:
        The underlying ``tile.write`` call expression. Direct callers
        typically ignore it; the DSL parser surfaces it as an ``EvalStmt``.
    """
    # Allow a bare IntLike as a flat 1-D index for backwards compatibility
    indices_seq: Sequence[IntLike] = [indices] if not isinstance(indices, Sequence) else indices
    value_expr = value.unwrap() if isinstance(value, Scalar) else value
    return _ir_ops.write(tile.unwrap(), _normalize_intlike(indices_seq), value_expr)


def load(
    tensor: Tensor,
    offsets: Sequence[IntLike],
    shapes: Sequence[IntLike],
    valid_shape: Sequence[IntLike] | None = None,
    target_memory: MemorySpace | None = None,
    clamp: bool = False,
    cache: CachePolicy | None = None,
) -> Tile:
    """Copy data from tensor to unified buffer (tile).

    Only the valid extent is read, so the tile may be larger than the region that
    exists in the source. The tile's valid region is the source's valid region,
    shifted by ``offsets`` and cut to the tile — a load never reports source bytes
    that do not exist as real data.

    Args:
        tensor: Source tensor
        offsets: Offsets in each dimension. Always in the source tensor's
            coordinate system. Each element must be an integer scalar — a float
            (``n / 2`` rather than ``n // 2``) or a nested sequence is rejected.
        shapes: Shape of the region to load in each dimension. Always in the
            source tensor's coordinate system. Each element must be an integer
            scalar, on the same terms as ``offsets``.
        valid_shape: Valid shape of the tile in each dimension. When provided, sets
            TileView.valid_shape in the output TileType. When omitted, shapes is used
            as valid_shape. Uses the same coordinate convention as shapes. Narrows
            the tile; cannot widen it past what the source has. Each element must
            be an integer scalar — one extent per dimension, not a nested
            ``[start, extent]`` pair.
        target_memory: Target memory space (MemorySpace.Vec or MemorySpace.Mat).
            ``None`` (the default) leaves an ordinary load unset for the compiler
            to place. MX-layout tensors default to MemorySpace.Mat so they can be
            passed directly to ``matmul_mx`` and placed from its operand position.
        clamp: Sanction a read that runs off the end of the source. By default a
            load asserts ``offsets + valid_shape`` stays inside the source and is
            rejected when that provably fails; ``clamp=True`` cuts the request back
            to the source edge instead.
        cache: GM cache-access policy for *this* read. ``None`` (the default)
            states no policy, leaving any scope-level declaration to apply.
            ``CachePolicy.BYPASS`` declares a streaming read — it asserts the
            bytes have no reuse worth caching and that nothing writes them while
            the kernel runs; coherency is the author's contract (see
            [`pl.set_cache_policy`][pypto.language.tensor.set_cache_policy] for
            the full contract). An explicit value here always wins over a
            scope-level ``pl.set_cache_policy`` declaration for the same tensor,
            in both directions: ``cache=CachePolicy.DEFAULT`` opts this one read
            back into the cache inside a bypassing scope. Requires PTOAS >= v0.64,
            where a BYPASS read compiles to a load issued against the page's
            uncached alias on A2/A3. New A5 code uses the six modes in
            CachePolicy; legacy CachePolicy.BYPASS reads retain their A5
            NotAllocKeep lowering for compatibility. ISA encodings are selected
            by backend code generation. DEFAULT resets an inherited
            scope policy to an ordinary load on either architecture.

    Returns:
        Tile wrapping the load operation

    Example:
        >>> # 2D load
        >>> tile = load(tensor, offsets=[0, 0], shapes=[32, 32])
        >>> # streaming read, no cache reuse expected
        >>> tile = load(tensor, [0, 0], [32, 32], cache=pl.CachePolicy.BYPASS)
    """
    if cache is not None and type(cache) not in (CachePolicy, int):
        raise TypeError("load cache must be CachePolicy")
    if valid_shape is None:
        valid_shape = shapes
    call_expr = _ir_ops.load(
        tensor.unwrap(),
        _normalize_intlike(offsets),
        _normalize_intlike(shapes),
        _normalize_intlike(valid_shape),
        target_memory,
        clamp=clamp,
        cache=None if cache is None else int(cache),
    )
    return Tile(expr=call_expr)


def store(
    tile: Tile,
    offsets: Sequence[IntLike],
    output_tensor: _TensorT,
    shapes: Sequence[IntLike] | None = None,
    *,
    atomic: AtomicType = AtomicType.None_,
    st_phase: STPhase = STPhase.Unspecified,
    pre_quant: float | None = None,
    pre_relu: bool = False,
    cache: CachePolicy | None = None,
) -> _TensorT:
    """Copy data from tile back to tensor.

    Args:
        tile: Source tile
        offsets: Offsets in each dimension. Each element must be an integer
            scalar — a float or a nested sequence is rejected.
        output_tensor: Output tensor
        shapes: Optional ND partition shape. Injected by FlattenTileNdTo2D for ND tensors.
            Each element must be an integer scalar, on the same terms as ``offsets``.
        atomic: Combine mode for the global-memory write. ``AtomicType.None_``
            (default) overwrites; ``AtomicType.Add`` atomically adds the tile
            into existing GM contents — used for split-K accumulation, where
            several cores accumulate partial products into one output.

            NOTE: atomic-add accumulation order across cores is not fixed, so
            floating-point results are non-deterministic. The destination must
            be zero-initialised before the kernel runs. Supported tile dtypes:
            fp32 / bf16 / fp16 / int32 / int16 / int8. bf16 atomic-add is
            available on the Ascend910 (A2/A3) profile; it is not supported on
            A5, where an fp32 accumulator + cast is required instead.
        st_phase: Consumer-side unit-flag phase. A producer that finishes with
            ``acc_phase=pl.AccPhase.Final`` must be consumed by a store with
            ``st_phase=pl.STPhase.Final`` so the unit flag is cleared.
        pre_quant: Scale applied by the cube's FIXPIPE while it drains an ``Acc``
            tile to global memory — the write becomes ``out = tile * pre_quant``
            in one instruction, with no vector work. This is what makes a
            *scale-bearing* conversion reachable from a cube accumulator: an
            INT32 accumulator into an FP16 tensor (dequantization) or into an
            INT8 one (requantization), neither of which the unscaled writeback
            can do. Legal source/destination dtype pairs are backend-specific
            and rejected here rather than in the assembler. ``None`` (default)
            emits no scale; ``1.0`` is a real identity scale and still selects
            the quantizing form.
        pre_relu: Apply ReLU in the same writeback. It runs on the *accumulator*,
            **before** ``pre_quant`` — pto-isa spells it ``ReluPreMode`` — with
            the destination clamp last, so the pair reproduces
            ``maximum(tile, 0) * pre_quant`` and not
            ``maximum(tile * pre_quant, 0)``. The two agree for every positive
            scale and differ at every element for a negative one.
        cache: CachePolicy.NORMAL_* or NOT_ALLOC_CLEAN for Vec/Acc-to-GM
            stores, including ReLU and quantized writeback. NOT_ALLOC_KEEP
            and NOT_ALLOC_DROP are load-only. On A2/A3 only DEFAULT is supported.

    Returns:
        Tensor wrapping the store operation

    Example:
        >>> # 2D store
        >>> result = store(tile, [0, 0], tensor)
        >>> # 3D store
        >>> result = store(tile, [0, 0, 0], tensor)
        >>> # atomic-add store (split-K)
        >>> result = store(partial, [0, 0], out, atomic=pl.AtomicType.Add)
        >>> # clear the unit flag after a final phased accumulation
        >>> result = store(acc, [0, 0], out, st_phase=pl.STPhase.Final)
        >>> # INT32 accumulator dequantized and ReLU'd straight into FP16 GM
        >>> result = store(acc_i32, [0, 0], out_f16, pre_quant=1.0 / 1024, pre_relu=True)
    """
    if cache is not None and type(cache) is not CachePolicy:
        raise TypeError("store cache must be CachePolicy")
    normalized_offsets = _normalize_intlike(offsets)
    normalized_shapes = _normalize_intlike(shapes) if shapes is not None else None
    call_expr = _ir_ops.store(
        tile.unwrap(),
        normalized_offsets,
        output_tensor.unwrap(),
        normalized_shapes,
        atomic=int(atomic),
        st_phase=st_phase,
        pre_quant=pre_quant,
        pre_relu=pre_relu,
        cache=None if cache is None else int(cache),
    )
    return output_tensor.__class__(expr=call_expr)


def assemble(
    target: Tile,
    source: Tile,
    offset: Sequence[IntLike],
    *,
    pre_quant: float | None = None,
    pre_relu: bool = False,
) -> Tile:
    """Write source tile data into target tile at specified offset.

    Args:
        target: Target tile to update
        source: Source tile to write
        offset: Offset dimensions for where to write
        pre_quant: Scale applied by the cube's FIXPIPE while it drains an ``Acc``
            source into a ``Mat`` target — the write becomes
            ``target[offset] = source * pre_quant`` in one instruction, with no
            vector work. This is what makes a *scale-bearing* conversion
            reachable from a cube accumulator: an INT32 accumulator into an FP16
            Mat tile (dequantization), which the next matmul can then read
            on-chip. ``None`` (default) emits no scale; ``1.0`` is a real
            identity scale and still selects the quantizing form.

            **Currently rejected on every backend** (PTOAS#1570). ptoas assembles the scale
            onto ``pto.tinsert`` but emits a pto-isa call that binds it to the
            row index instead, silently dropping it, so PyPTO refuses the whole
            destination rather than emit a kernel that computes the wrong thing.
            Use the Acc -> GM form — ``tile.store(acc, ..., pre_quant=s)``, which
            is device-verified — or scale in the vector unit. ``pre_relu`` alone
            is unaffected here.
        pre_relu: Apply ReLU in the same writeback. It runs on the *accumulator*,
            **before** ``pre_quant`` — pto-isa spells it ``ReluPreMode`` — with
            the destination clamp last, so the pair reproduces
            ``maximum(source, 0) * pre_quant`` and not
            ``maximum(source * pre_quant, 0)``. The two agree for every positive
            scale and differ at every element for a negative one.

    Returns:
        Tile wrapping the assemble operation

    Example:
        >>> # INT32 accumulator dequantized and ReLU'd into an FP16 Mat scratch
        >>> s = create([128, 128], pl.FP16, target_memory=pl.Mem.Mat)
        >>> s = assemble(s, acc_i32, [0, 0], pre_quant=1.0 / 1024, pre_relu=True)
    """
    call_expr = _ir_ops.assemble(
        target.unwrap(),
        source.unwrap(),
        _normalize_intlike(offset),
        pre_quant=pre_quant,
        pre_relu=pre_relu,
    )
    return Tile(expr=call_expr)


def gather_row(  # noqa: PLR0913
    dst: Tile,
    src: Tensor,
    dst_offset: Sequence[IntLike],
    src_offset: Sequence[IntLike],
    shapes: Sequence[IntLike],
    transpose: bool = False,
    *,
    valid_shape: Sequence[IntLike] | None = None,
) -> Tile:
    """Load one GM row directly into a sub-region of an on-chip tile (DPS).

    Per-row primitive of the paged-gather lowering: DMAs one GM row window
    straight into ``dst`` at ``dst_offset`` (``pto.subview`` of ``dst`` +
    ``pto.tload``, ``GM -> on-chip``, no ``pto.tmov``). The caller computes the
    physical ``src_offset`` (block-table lookup + bias) and the ``dst_offset``
    slot itself, so arbitrary gather logic stays in the kernel. Writes ``dst``
    in place, so a loop-carried accumulator is filled row by row and feeds
    ``pl.matmul`` directly — the tile-level counterpart of
    ``pypto.language.op.tensor_ops.gather_row``.

    Args:
        dst: Destination on-chip accumulator tile (Mat/L1 or Vec/UB).
        src: Source pool in GM (a ``Tensor``).
        dst_offset: ``[row, col]`` slot within ``dst`` to write.
        src_offset: ``[row, col]`` physical offset within the GM ``src``.
        shapes: GM row window shape ``[r, c]`` (typically ``[1, size]``).
            Must be compile-time constant.
        valid_shape: How much of that window to actually transfer, defaulting to
            all of it. May hold runtime ``Scalar`` values, so a dynamic row count
            leaves the tile's allocation and layout untouched. Not supported
            together with ``transpose=True``.
        transpose: Place the GM row ``[r, c]`` as an on-chip column ``[c, r]`` —
            fills a matmul ``b_trans`` B-operand without a GM round-trip
            (Mat/L1 only).

    Returns:
        Tile aliasing ``dst`` (written in place).
    """
    call_expr = _ir_ops.gather_row(
        dst.unwrap(),
        src.unwrap(),
        _normalize_intlike(dst_offset),
        _normalize_intlike(src_offset),
        _normalize_intlike(shapes),
        transpose,
        valid_shape=_normalize_intlike(valid_shape) if valid_shape is not None else None,
    )
    return Tile(expr=call_expr)


def extract(
    src: Tile,
    index_row: IntLike,
    index_col: IntLike,
    shape: Sequence[IntLike],
    *,
    target_memory: MemorySpace,
) -> Tile:
    """Extract a sub-tile from ``src`` at ``(index_row, index_col)`` — ISA TEXTRACT.

    Maps to ISA TEXTRACT Variant 1 (Standard Extract). The result tile has the
    given static ``shape`` and lives in ``target_memory``.

    Args:
        src: Source tile (typically in Mat or Acc memory)
        index_row: Starting row offset
        index_col: Starting col offset
        shape: Static 2D shape of the extracted sub-tile
        target_memory: Destination memory space —
            ``Left`` / ``Right`` for Mat sources, ``Mat`` for Acc sources

    Returns:
        Tile of the requested shape in ``target_memory``
    """
    [row, col] = _normalize_intlike([index_row, index_col])
    call_expr = _ir_ops.extract(
        src.unwrap(),
        row,
        col,
        shape=_normalize_intlike(shape),
        target_memory=target_memory,
    )
    return Tile(expr=call_expr)


def concat(src0: Tile, src1: Tile) -> Tile:
    """Concatenate two tiles along the column dimension.

    Args:
        src0: First source tile
        src1: Second source tile

    Returns:
        Tile with concatenated columns
    """
    call_expr = _ir_ops.concat(src0.unwrap(), src1.unwrap())
    return Tile(expr=call_expr)


def move(
    tile: Tile,
    target_memory: MemorySpace,
    blayout: TileLayout | None = None,
    slayout: TileLayout | None = None,
) -> Tile:
    """Move tile between memory levels.

    Args:
        tile: Input tile
        target_memory: Target memory space (MemorySpace.Vec, .Mat, .Left, .Right,
            .LeftScale, .RightScale)
        blayout: Optional block layout for the destination tile
        slayout: Optional scatter layout for the destination tile

    Returns:
        Tile wrapping the move operation
    """
    call_expr = _ir_ops.move(
        tile.unwrap(),
        target_memory,
        blayout=blayout,
        slayout=slayout,
    )
    return Tile(expr=call_expr)


def full(shape: list[int], dtype: DataType, value: int | float) -> Tile:
    """Create a tile from a shape and fill with value in Vec.

    Args:
        shape: Shape of the tile
        dtype: Data type of the tile
        value: filling scalar

    Returns:
        Tile wrapping the full operation
    """
    call_expr = _ir_ops.full(shape, dtype, value)
    return Tile(expr=call_expr)


def ci(
    start: int | Scalar,
    shape: Sequence[int],
    dtype: DataType = DataType.INT32,
    descending: bool = False,
    *,
    tmp: Tile | None = None,
) -> Tile:
    """Generate a contiguous integer sequence into a tile.

    Equivalent to ``numpy.arange``-style index generation. Maps to ``pto.tci``.
    For a column index ``k`` in the first row of the destination, ascending gives
    ``dst[0, k] = start + k`` and descending gives ``dst[0, k] = start - k``.

    Note:
        ``pto.tci`` uses the destination's valid-column count as the sequence
        length and does NOT populate additional rows. Leading dimensions must
        be 1 — prefer shapes of the form ``[1, N]``.

    Args:
        start: Starting integer (plain int or a Scalar). Must match ``dtype``.
        shape: Shape of the destination tile (static, innermost dim != 1).
        dtype: Destination dtype. One of {INT16, INT32}. Defaults to INT32.
        descending: If True, generate a descending sequence.
        tmp: Optional A2/A3 PTOAS scratch tile. Normally compiler-generated.

    Returns:
        Tile wrapping the ci operation.
    """
    start_expr = start.unwrap() if isinstance(start, Scalar) else start
    call_expr = _ir_ops.ci(
        start_expr,
        list(shape),
        dtype=dtype,
        descending=descending,
        tmp=None if tmp is None else tmp.unwrap(),
    )
    return Tile(expr=call_expr)


arange = ci


def tri(
    diagonal: int | Scalar,
    shape: Sequence[int],
    valid_shape: Sequence[int] | None = None,
    dtype: DataType = DataType.INT32,
    upper: bool = False,
) -> Tile:
    """Generate a lower- or upper-triangular mask tile.

    ``upper=False`` writes one where ``j <= i + diagonal``; ``upper=True``
    writes one where ``j >= i + diagonal``. Only the optional valid region is
    written.

    Args:
        diagonal: Offset of the boundary from the main diagonal, in columns.
            0 includes the diagonal; positive shifts it right, negative left.
            May be a runtime ``Scalar``.
        shape: Shape of the destination tile (static).
        valid_shape: Optional written region (each dim ``<= shape``). Elements
            outside it are not written, so their value is whatever the freshly
            allocated tile holds. Defaults to the full shape.
        dtype: Destination dtype. Defaults to ``INT32``.
        upper: Select the upper triangle instead of the lower.

    Returns:
        A tile holding 1 inside the selected triangle and 0 outside it.
    """
    diagonal_expr = diagonal.unwrap() if isinstance(diagonal, Scalar) else diagonal
    call_expr = _ir_ops.tri(
        diagonal_expr,
        list(shape),
        valid_shape=list(valid_shape) if valid_shape is not None else None,
        dtype=dtype,
        upper=upper,
    )
    return Tile(expr=call_expr)


def random(
    key0: int | Scalar,
    key1: int | Scalar,
    counter0: int | Scalar,
    counter1: int | Scalar,
    counter2: int | Scalar,
    counter3: int | Scalar,
    shape: Sequence[int],
    valid_shape: Sequence[int] | None = None,
    dtype: DataType = DataType.UINT32,
    rounds: int = 10,
) -> Tile:
    """Generate counter-based pseudo-random values into a tile.

    Implements a counter-based (Philox/ChaCha-style) RNG. Each element is derived
    deterministically from the 64-bit key ``(key0, key1)`` and 128-bit counter
    ``(counter0..counter3)`` plus the element position, so the same seeds always
    reproduce the same tile. Maps to ``pto.trandom``.

    Args:
        key0: Low INT32 key word (plain int or Scalar).
        key1: High INT32 key word (plain int or Scalar).
        counter0: First INT32 counter word.
        counter1: Second INT32 counter word.
        counter2: Third INT32 counter word.
        counter3: Fourth INT32 counter word.
        shape: Shape of the destination tile (static).
        valid_shape: Optional written region (each dim ``<= shape``); ``pto.trandom``
            only fills the valid rows/cols. Defaults to the full shape.
        dtype: Destination dtype. One of {INT32, UINT32}. Defaults to UINT32.
        rounds: Cipher round count, 7 or 10. Defaults to 10.

    Returns:
        Tile wrapping the random operation.
    """

    def _seed_expr(seed: int | Scalar) -> int | Expr:
        return seed.unwrap() if isinstance(seed, Scalar) else seed

    vshape = list(valid_shape) if valid_shape is not None else None
    call_expr = _ir_ops.random(
        _seed_expr(key0),
        _seed_expr(key1),
        _seed_expr(counter0),
        _seed_expr(counter1),
        _seed_expr(counter2),
        _seed_expr(counter3),
        list(shape),
        valid_shape=vshape,
        dtype=dtype,
        rounds=rounds,
    )
    return Tile(expr=call_expr)


def fillpad(tile: Tile, pad_value: PadValue | int | float = PadValue.zero) -> Tile:
    """Fill remaining tile elements with specified padding value.

    Args:
        tile: Input tile
        pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
            the literal sugars ``0``, ``math.inf``, ``-math.inf``. Default is
            ``PadValue.zero``. Other values raise — the hardware only supports
            the three padding modes.

    Returns:
        Tile wrapping the fillpad operation
    """
    call_expr = _ir_ops.fillpad(tile.unwrap(), pad_value=pad_value)
    return Tile(expr=call_expr)


def fillpad_inplace(tile: Tile, pad_value: PadValue | int | float = PadValue.zero) -> Tile:
    """Fill padding elements of input tile in place.

    Unlike fillpad which allocates a new output tile, this operation reuses
    the input tile's UB buffer. The result shares the same memory address,
    making it equivalent to TFILLPAD_INPLACE on the hardware.

    Args:
        tile: Input tile
        pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
            the literal sugars ``0``, ``math.inf``, ``-math.inf``. Default is
            ``PadValue.zero``. Other values raise — the hardware only supports
            the three padding modes.

    Returns:
        Tile with padding filled (shares memory with the input tile).
    """
    call_expr = _ir_ops.fillpad_inplace(tile.unwrap(), pad_value=pad_value)
    return Tile(expr=call_expr)


def fillpad_expand(
    tile: Tile, shape: Sequence[IntLike], pad_value: PadValue | int | float = PadValue.zero
) -> Tile:
    """Copy a smaller source tile into a larger destination tile, padding the rest.

    Unlike [`fillpad`][pypto.language.tile.fillpad] (which keeps the same physical shape and only fills the
    valid-region expansion), this op produces a *larger* output tile: the source's
    valid region is copied to the top-left and every other element is filled with
    ``pad_value``. Equivalent to TFILLPAD_EXPAND on the hardware.

    Args:
        tile: Source tile
        shape: Destination shape; each dimension must be >= the source dimension
        pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
            the literal sugars ``0``, ``math.inf``, ``-math.inf``. Default is
            ``PadValue.zero``. Other values raise — the hardware only supports
            the three padding modes.

    Returns:
        Tile wrapping the fillpad_expand operation (a new, larger tile).
    """
    call_expr = _ir_ops.fillpad_expand(tile.unwrap(), _normalize_intlike(shape), pad_value=pad_value)
    return Tile(expr=call_expr)


def slice(
    tile: Tile,
    shape: Sequence[IntLike],
    offset: Sequence[IntLike],
    valid_shape: Sequence[IntLike] | None = None,
    drop_dims: Sequence[int | Expr] | None = None,
    pad_value: PadValue | int | float | None = None,
) -> Tile:
    """Create a slice of a tile with static shape and optional valid shape.

    The slice is never valid where the source tile is not: the source's valid
    region, shifted by ``offset`` and cut to the window, bounds the result.

    Args:
        tile: Input tile
        shape: Static shape dimensions. Always full-rank — a scalar-indexed axis
            contributes a unit dim here and is listed in ``drop_dims``.
        offset: Offset dimensions for the slice
        valid_shape: Valid shape dimensions. When omitted, the source's validity
            under the window is used. Narrows the result; cannot widen it.
        drop_dims: Optional axes to erase from the result type (numpy-style rank
            reduction). Each listed axis must be a static unit dim of ``shape``
            and must still be fully valid after the intersection above.
            Because tiles are physically 2D, the result is clamped back to 2D
            if reduction would take it below 2D. ``None`` / ``[]`` drops nothing.
        pad_value: Optional padding mode for out-of-valid-shape elements.
            ``None`` means the source's padding mode carries through.
            Accepts ``PadValue.zero`` / ``PadValue.max`` / ``PadValue.min``, or
            the literal sugars ``0``, ``math.inf``, ``-math.inf`` (same
            spelling as [`tile.fillpad`][pypto.language.tile.fillpad]). Only meaningful when the
            *effective* valid region is smaller than ``shape`` — which an explicit
            ``valid_shape`` or a partially-valid source tile can each bring about.

    Returns:
        Tile wrapping the slice operation

    Note:
        Unlike [`tensor.slice`][pypto.language.tensor.slice], there is no ``clamp``
        option: an on-chip window has nothing that could clamp it, so
        ``offset + shape`` must stay inside the source tile.
    """
    # pad_value paints whatever falls outside the *effective* valid region, and an
    # explicit valid_shape is only one way to narrow it — a partially-valid source
    # tile narrows it on its own. Warn only when neither can apply.
    if (
        pad_value is not None
        and pad_value is not PadValue.null
        and valid_shape is None
        and not has_partial_valid_region(tile.unwrap())
    ):
        warnings.warn(
            f"tile.slice received pad_value={pad_value!r} but no valid_shape and a "
            f"fully-valid source. "
            f"pad_value has no effect unless the valid region is smaller than shape. "
            f"If you intend to narrow the valid region later via "
            f"tile.set_validshape, you can ignore this warning; otherwise "
            f"pass valid_shape=... to tile.slice.",
            # Not a literal 2: pl.slice forwards here, and a fixed level would
            # name the dispatcher and collapse every call site's warning.
            stacklevel=caller_warning_stacklevel(),
        )

    tile_expr = tile.unwrap()
    normalized_valid_shape = None if valid_shape is None else _normalize_intlike(valid_shape)
    call_expr = _ir_ops.slice(
        tile_expr,
        _normalize_intlike(shape),
        _normalize_intlike(offset),
        normalized_valid_shape,
        drop_dims,
        pad_value=pad_value,
    )
    return Tile(expr=call_expr)


def reshape(tile: Tile, shape: Sequence[IntLike]) -> Tile:
    """Reshape tile to new shape.

    The valid region is carried through, never widened: the result holds real
    data in exactly the cells the input did. See ``pl.reshape`` for the cases
    that always map.

    Args:
        tile: Input tile
        shape: New shape dimensions. A tile is physically 2D, so a higher-rank
            result is an intermediate that ``FlattenTileNdTo2D`` later collapses.

    Returns:
        Tile wrapping the reshape operation

    Raises:
        ValueError: If the element count changes, or if the input holds real
            data in only part of its buffer and no origin-anchored region of
            ``shape`` describes those same cells.
    """
    tile_expr = tile.unwrap()
    call_expr = _ir_ops.reshape(tile_expr, _normalize_intlike(shape))
    return Tile(expr=call_expr)


def reinterpret_view(
    data: Tile,
    dtype: DataType,
    *,
    shape: Sequence[IntLike] | None = None,
) -> Tile:
    """Reinterpret a tile over the same bytes with a different dtype.

    Either packed FP4 format <-> UINT8/INT8 preserves the carrier shape and requires a
    dense address-backed 2D row-major Vec tile with a static UB offset.
    Reshape the byte view separately before aliasing back. On A5, PYPTO
    and DSA_RP support these aliases; PTOAS v0.67 planning is unsupported.

    Args:
        data: Input tile.
        dtype: Target element dtype, which must differ from the source dtype.
        shape: Optional byte-equivalent target shape. When omitted, the
            physically contiguous dimension is scaled according to the
            source/target dtype byte ratio.

    Returns:
        Tile wrapping the zero-copy reinterpret-view operation.
    """
    normalized_shape = None if shape is None else _normalize_intlike(shape)
    call_expr = _ir_ops.reinterpret_view(data.unwrap(), dtype, shape=normalized_shape)
    return Tile(expr=call_expr)


def transpose(tile: Tile, axis1: int, axis2: int, tmp_tile: Tile | None = None) -> Tile:
    """Transpose tile by swapping two axes.

    The ``pto.ttrans`` scratch buffer is a codegen detail allocated later by
    ``FlattenTileNdTo2D``, so user code never supplies ``tmp_tile``. The optional
    parameter exists only so the lowered 4-arg form round-trips through the parser.

    Args:
        tile: Input tile.
        axis1: First axis to swap (supports negative indexing).
        axis2: Second axis to swap (supports negative indexing).
        tmp_tile: Optional scratch tile — compiler-generated lowered IR only.

    Returns:
        Tile wrapping the transpose operation.
    """
    tile_expr = tile.unwrap()
    tmp_expr = tmp_tile.unwrap() if tmp_tile is not None else None
    call_expr = _ir_ops.transpose(tile_expr, axis1, axis2, tmp=tmp_expr)
    return Tile(expr=call_expr)


def transpose_view(tile: Tile) -> Tile:
    """Zero-copy fractal-layout reinterpretation (NZ<->ZN) of a tile.

    Swaps the trailing two dims together with the block/scatter layouts, aliasing
    the source buffer byte-for-byte: an NZ ``[..., N, K]`` tile and a ZN
    ``[..., K, N]`` tile over the same L1 bytes are mutual transposes. Emits no
    data movement, so one GM->L1 load can feed both a ``b_trans=True`` and a
    ``b_trans=False`` matmul on a shared operand.

    Args:
        tile: Input tile (TileType, >=2D; typically Mat-resident).

    Returns:
        Tile wrapping the transposed-layout view.
    """
    tile_expr = tile.unwrap()
    call_expr = _ir_ops.transpose_view(tile_expr)
    return Tile(expr=call_expr)


def set_validshape(tile: Tile, valid_rows: IntLike, valid_cols: IntLike) -> Tile:
    """Update valid-shape metadata of a tile without data movement.

    .. note::
        The operand must not be a view (a ``pl.tile.slice`` or reshape result): a
        view carries its valid extent in its type, so there is nothing to update.
        Narrow at the slice with ``valid_shape=`` instead.

    Args:
        tile: Input tile (must be 2D)
        valid_rows: Number of valid rows (int or Scalar[INDEX])
        valid_cols: Number of valid columns (int or Scalar[INDEX])

    Returns:
        Tile with updated valid_shape metadata
    """
    tile_expr = tile.unwrap()
    vr = valid_rows.unwrap() if isinstance(valid_rows, Scalar) else valid_rows
    vc = valid_cols.unwrap() if isinstance(valid_cols, Scalar) else valid_cols
    call_expr = _ir_ops.set_validshape(tile_expr, vr, vc)
    return Tile(expr=call_expr)
