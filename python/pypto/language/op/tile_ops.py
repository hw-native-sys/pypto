# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile operations for PyPTO Language DSL.

This module provides type-safe wrappers around pypto.ir.op.tile operations
that accept and return Tile types instead of raw Expr/Call objects.

Accessed as ``pl.tile.*``
"""

import warnings
from collections.abc import Sequence
from typing import Any, overload

__all__ = [
    "MemRefType",
    "alloc",
    "create_tile",
    "create",
    "read",
    "write",
    "load",
    "store",
    "assemble",
    "extract",
    "scatter_update",
    "concat",
    "move",
    "full",
    "ci",
    "arange",
    "fillpad",
    "fillpad_inplace",
    "get_block_idx",
    "get_subblock_idx",
    "get_block_num",
    "add",
    "sub",
    "mul",
    "div",
    "adds",
    "subs",
    "muls",
    "divs",
    "neg",
    "exp",
    "sqrt",
    "rsqrt",
    "recip",
    "log",
    "abs",
    "relu",
    "cast",
    "matmul",
    "batch_matmul",
    "matmul_acc",
    "batch_matmul_acc",
    "matmul_bias",
    "gemv",
    "gemv_acc",
    "gemv_bias",
    "row_max",
    "row_sum",
    "row_min",
    "col_sum",
    "col_max",
    "col_min",
    "maximum",
    "row_expand",
    "row_expand_sub",
    "row_expand_div",
    "row_expand_mul",
    "row_expand_add",
    "col_expand",
    "col_expand_mul",
    "col_expand_div",
    "col_expand_sub",
    "expands",
    "minimum",
    "cmp",
    "cmps",
    "sum",
    "max",
    "min",
    "slice",
    "reshape",
    "transpose",
    "set_validshape",
    "rem",
    "rems",
    "and_",
    "ands",
    "or_",
    "ors",
    "xor",
    "xors",
    "shl",
    "shls",
    "shr",
    "shrs",
    "maxs",
    "mins",
    "prelu",
    "not_",
    "addc",
    "subc",
    "addsc",
    "subsc",
    "lrelu",
    "sel",
    "sels",
    "tpush_to_aiv",
    "tpush_to_aic",
    "tpop_from_aic",
    "tpop_from_aiv",
    "sort32",
    "gather",
    "mscatter",
    "MaskPattern",
    "mrgsort",
]

from pypto.ir.op import tile_ops as _ir_ops
from pypto.ir.utils import _get_span_or_capture
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Expr, MemorySpace, PadValue, TileLayout

from ..typing import IntLike, Scalar, Tensor, Tile
from .system_ops import (  # noqa: F401
    tpop_from_aic,
    tpop_from_aiv,
    tpush_to_aic,
    tpush_to_aiv,
)


class MemRefType:
    """Opaque sentinel type for alloc results in printed IR.

    tile.alloc / tensor.alloc are internal IR operations created by the
    InitMemRef and AllocateMemoryAddr passes.  This class exists solely so
    that the Python code emitted by the C++ printer is valid Python.
    """


class MaskPattern:
    """Hardware mask pattern selectors for tile.gather_mask.

    Bit patterns are read right-to-left; lower bits correspond to lower indices.
    """

    P0101 = 1  # stride-2: select positions 0, 2, 4, ...
    P1010 = 2  # stride-2: select positions 1, 3, 5, ...
    P0001 = 3  # stride-4: select positions 0, 4, 8, ...
    P0010 = 4  # stride-4: select positions 1, 5, 9, ...
    P0100 = 5  # stride-4: select positions 2, 6, 10, ...
    P1000 = 6  # stride-4: select positions 3, 7, 11, ...
    P1111 = 7  # select all positions


def alloc(
    memory_space: MemorySpace,
    size: int,
) -> MemRefType:
    """Stub for the internal ``tile.alloc`` IR operation.

    This function is never called in user-written DSL code.  It is emitted
    by the C++ python-printer after the InitMemRef / AllocateMemoryAddr
    passes and must be importable so that the printed source is valid Python
    that the text-parser can ``exec()``.

    Args:
        memory_space: Target memory space (e.g. ``pl.Mem.Vec``)
        size: Allocation size in bytes

    Returns:
        Opaque MemRefType sentinel (unused at runtime)
    """
    return MemRefType()


def _unwrap_rhs(rhs: int | float | Expr | Tile | Scalar) -> int | float | Expr:
    """Unwrap rhs operands into the IR-layer representation."""
    return rhs.unwrap() if isinstance(rhs, (Tile, Scalar)) else rhs


def _normalize_intlike(seq: Sequence[IntLike]) -> list[int | Expr]:
    """Unwrap Scalar elements to Expr so the sequence matches C++ binding types."""
    return [elem.unwrap() if isinstance(elem, Scalar) else elem for elem in seq]


def _scalar_operand_to_expr(value: int | Scalar | Expr) -> Expr:
    """Coerce a scalar-binary operand to an ``Expr``.

    Used by the scalar-pair branches of ``tile.min`` / ``tile.max``: ``Scalar``
    is unwrapped to its inner ``Expr``, raw ``Expr`` is forwarded as-is, and a
    bare ``int`` is materialized as ``ConstInt(.., INDEX)`` with the span
    pinned by the parser if any (so error messages and dumps point at the
    user's source line, not this wrapper). ``INDEX`` matches the dtype the
    parser uses for plain int literals, so round-tripped programs don't
    sprout spurious casts on otherwise-equivalent constants.
    """
    if isinstance(value, Scalar):
        return value.unwrap()
    if isinstance(value, Expr):
        return value
    return _ir_core.ConstInt(value, DataType.INDEX, _get_span_or_capture())


def _axis_to_int(axis: int | Scalar | Expr) -> int:
    """Coerce a compile-time axis argument to a Python ``int``.

    The parser passes integer literals through as raw ``ConstInt`` (to
    preserve dtype) — accept that shape and unwrap. Direct callers can
    still pass a bare ``int``.
    """
    if isinstance(axis, bool):  # bool is an int subclass; reject explicitly
        raise TypeError(f"axis must be int, got bool ({axis!r})")
    if isinstance(axis, int):
        return axis
    if isinstance(axis, _ir_core.ConstInt):
        return int(axis.value)
    if isinstance(axis, Scalar):
        inner = axis.unwrap()
        if isinstance(inner, _ir_core.ConstInt):
            return int(inner.value)
    raise TypeError(
        f"axis must be a compile-time int (or ConstInt/Scalar wrapping one), got {type(axis).__name__}"
    )


def create(
    shape: Sequence[IntLike],
    dtype: DataType,
    target_memory: MemorySpace = MemorySpace.Vec,
) -> Tile:
    """Create a tile from a shape.

    Args:
        shape: Shape of the tile
        dtype: Data type of the tile
        target_memory: Target memory space (MemorySpace.Vec, .Mat, .Left, .Right)

    Returns:
        Tile wrapping the create operation
    """
    # create C++ binding accepts Sequence[int]; Expr elements from Scalar
    # unwrapping are valid at DSL parse time (parser reads the AST).
    call_expr = _ir_ops.create(
        _normalize_intlike(shape),
        dtype,
        target_memory,
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
    valid_shapes: Sequence[IntLike] | None = None,
    target_memory: MemorySpace = MemorySpace.Vec,
    transpose: bool = False,
) -> Tile:
    """Copy data from tensor to unified buffer (tile).

    Args:
        tensor: Source tensor
        offsets: Offsets in each dimension. Always in the source tensor's
            coordinate system.
        shapes: Shape of the region to load in each dimension. Always in the
            source tensor's coordinate system, even when transpose=True. The
            output TileType shape will be transposed automatically.
        target_memory: Target memory space (MemorySpace.Vec default, or MemorySpace.Mat)
        valid_shapes: Valid shape of the tile in each dimension. When provided, sets
            TileView.valid_shape in the output TileType. When omitted, shapes is used
            as valid_shape. Uses the same coordinate convention as shapes.
        transpose: Whether to transpose the tile during load (default: False).
            Only supported when target_memory is MemorySpace.Mat (L1).

    Returns:
        Tile wrapping the load operation

    Example:
        >>> # 2D load
        >>> tile = load(tensor, offsets=[0, 0], shapes=[32, 32])
        >>> # 2D load with transpose to L1 (tensor is [N, K], output tile is [K, N])
        >>> tile = load(tensor, offsets=[0, 0], shapes=[N, K],
        ...             target_memory=pl.MemorySpace.Mat, transpose=True)
    """
    if valid_shapes is None:
        valid_shapes = shapes
    call_expr = _ir_ops.load(
        tensor.unwrap(),
        _normalize_intlike(offsets),
        _normalize_intlike(shapes),
        _normalize_intlike(valid_shapes),
        target_memory,
        transpose,
    )
    return Tile(expr=call_expr)


def store(
    tile: Tile,
    offsets: Sequence[IntLike],
    output_tensor: Tensor,
    shapes: Sequence[IntLike] | None = None,
) -> Tensor:
    """Copy data from tile back to tensor.

    Args:
        tile: Source tile
        offsets: Offsets in each dimension
        output_tensor: Output tensor
        shapes: Optional ND partition shape. Injected by FlattenTileNdTo2D for ND tensors.

    Returns:
        Tensor wrapping the store operation

    Example:
        >>> # 2D store
        >>> result = store(tile, [0, 0], tensor)
        >>> # 3D store
        >>> result = store(tile, [0, 0, 0], tensor)
    """
    normalized_offsets = _normalize_intlike(offsets)
    normalized_shapes = _normalize_intlike(shapes) if shapes is not None else None
    call_expr = _ir_ops.store(tile.unwrap(), normalized_offsets, output_tensor.unwrap(), normalized_shapes)
    return Tensor(expr=call_expr)


def assemble(target: Tile, source: Tile, offset: Sequence[IntLike]) -> Tile:
    """Write source tile data into target tile at specified offset.

    Args:
        target: Target tile to update
        source: Source tile to write
        offset: Offset dimensions for where to write

    Returns:
        Tile wrapping the assemble operation
    """
    call_expr = _ir_ops.assemble(target.unwrap(), source.unwrap(), _normalize_intlike(offset))
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


def scatter_update(input: Tile, *args: Any, **kwargs: Any) -> Tile:
    """Update tile rows at positions specified by 2D index tile with values from src.

    Accepts the same flexible call shapes as the IR builder
    ``pypto.ir.op.tile.scatter_update``:

    - ``scatter_update(input, dim, index, src, scratch)``
    - ``scatter_update(input, index, src, scratch, dim=-2)``
    - ``scatter_update(input, dim, index=..., src=..., scratch=...)``

    Tile / Scalar wrappers are unwrapped before forwarding so the IR builder
    receives raw ``Expr`` operands.
    """

    def _unwrap(v: Any) -> Any:
        return v.unwrap() if isinstance(v, (Tile, Scalar)) else v

    fwd_args = tuple(_unwrap(a) for a in args)
    fwd_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
    return Tile(expr=_ir_ops.scatter_update(input.unwrap(), *fwd_args, **fwd_kwargs))


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
        target_memory: Target memory space (MemorySpace.Vec, .Mat, .Left, .Right)
        blayout: Optional block layout for the destination tile
        slayout: Optional scatter layout for the destination tile

    Returns:
        Tile wrapping the move operation
    """
    call_expr = _ir_ops.move(tile.unwrap(), target_memory, blayout=blayout, slayout=slayout)
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
) -> Tile:
    """Generate a contiguous integer sequence into a tile.

    Equivalent to ``numpy.arange``-style index generation. Maps to ``pto.tci``.

    Args:
        start: Starting integer (plain int or a Scalar). Must match ``dtype``.
        shape: Shape of the destination tile (static, innermost dim != 1).
        dtype: Destination dtype. One of {INT16, INT32}. Defaults to INT32.
        descending: If True, generate a descending sequence.

    Returns:
        Tile wrapping the ci operation.
    """
    start_expr = start.unwrap() if isinstance(start, Scalar) else start
    call_expr = _ir_ops.ci(start_expr, list(shape), dtype=dtype, descending=descending)
    return Tile(expr=call_expr)


arange = ci


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


def get_block_idx() -> Scalar:
    """Get the current block index.

    This operation returns the index of the current compute tile. It is typically
    used in tile-level programming to identify which block of data is being processed.

    Returns:
        Scalar wrapping the get_block_idx operation (INDEX type)

    Example:
        >>> block_idx = pl.tile.get_block_idx()
        >>> if block_idx < 10:
        >>>     # Process first 10 blocks differently
        >>>     ...
    """
    call_expr = _ir_ops.get_block_idx()
    return Scalar(expr=call_expr)


def get_subblock_idx() -> Scalar:
    """Get the current sub-block (vector core) index.

    Returns the index of the current vector core within a split execution.
    Core 0 returns 0, core 1 returns 1.

    Returns:
        Scalar wrapping the get_subblock_idx operation (INDEX type)
    """
    call_expr = _ir_ops.get_subblock_idx()
    return Scalar(expr=call_expr)


def get_block_num() -> Scalar:
    """Get the total number of blocks in the current SPMD task.

    This operation returns the total count of blocks dispatched for the current
    task. Used with get_block_idx() for SPMD work partitioning.

    Returns:
        Scalar wrapping the get_block_num operation (INDEX type)

    Example:
        >>> block_idx = pl.tile.get_block_idx()
        >>> block_num = pl.tile.get_block_num()
    """
    call_expr = _ir_ops.get_block_num()
    return Scalar(expr=call_expr)


def add(lhs: Tile, rhs: Tile | int | float | Scalar | Expr) -> Tile:
    """Element-wise addition of tile and tile or scalar.

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

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile or scalar

    Returns:
        Tile wrapping the mul operation
    """
    call_expr = _ir_ops.mul(lhs.unwrap(), _unwrap_rhs(rhs))
    return Tile(expr=call_expr)


def div(lhs: Tile, rhs: Tile | int | float | Scalar | Expr) -> Tile:
    """Element-wise division of tile and tile or scalar.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile or scalar

    Returns:
        Tile wrapping the div operation
    """
    call_expr = _ir_ops.div(lhs.unwrap(), _unwrap_rhs(rhs))
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


def recip(tile: Tile) -> Tile:
    """Element-wise reciprocal.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the recip operation
    """
    call_expr = _ir_ops.recip(tile.unwrap())
    return Tile(expr=call_expr)


def log(tile: Tile) -> Tile:
    """Element-wise natural logarithm.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the log operation
    """
    call_expr = _ir_ops.log(tile.unwrap())
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
) -> Tile:
    """Cast tile to target data type (element-wise).

    Args:
        tile: Input tile (TileType)
        target_type: Target data type (DataType)
        mode: Rounding mode — string name ("none", "rint", "round", "floor",
              "ceil", "trunc", "odd") or int (0–6)

    Returns:
        Tile wrapping the cast operation
    """
    call_expr = _ir_ops.cast(tile.unwrap(), target_type, mode)
    return Tile(expr=call_expr)


def matmul(lhs: Tile, rhs: Tile) -> Tile:
    """Matrix multiplication of two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the matmul operation
    """
    call_expr = _ir_ops.matmul(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def batch_matmul(lhs: Tile, rhs: Tile) -> Tile:
    """Batch matrix multiplication of two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the batch_matmul operation
    """
    call_expr = _ir_ops.batch_matmul(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def matmul_acc(acc: Tile, lhs: Tile, rhs: Tile) -> Tile:
    """Matrix multiplication with accumulation: acc += lhs @ rhs.

    Args:
        acc: Accumulator tile
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the matmul_acc operation
    """
    call_expr = _ir_ops.matmul_acc(acc.unwrap(), lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def batch_matmul_acc(acc: Tile, lhs: Tile, rhs: Tile) -> Tile:
    """Batch matrix multiplication with accumulation: acc += lhs @ rhs.

    Performs the in-place ``acc += lhs @ rhs`` with batch-dim broadcasting between
    ``lhs`` and ``rhs``. The broadcast batch shape must equal the batch shape of
    ``acc`` (acc is the in-place accumulation target and is not broadcast).

    Args:
        acc: Accumulator tile (at least 2D)
        lhs: Left-hand side tile (at least 2D)
        rhs: Right-hand side tile (at least 2D)

    Returns:
        Tile wrapping the batch_matmul_acc operation
    """
    call_expr = _ir_ops.batch_matmul_acc(acc.unwrap(), lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def matmul_bias(lhs: Tile, rhs: Tile, bias: Tile) -> Tile:
    """Matrix multiplication with bias add: C = lhs @ rhs + bias.

    Args:
        lhs: Left-hand side tile [M, K]
        rhs: Right-hand side tile [K, N]
        bias: Bias tile [1, N]

    Returns:
        Tile wrapping the matmul_bias operation
    """
    call_expr = _ir_ops.matmul_bias(lhs.unwrap(), rhs.unwrap(), bias.unwrap())
    return Tile(expr=call_expr)


def gemv(lhs: Tile, rhs: Tile) -> Tile:
    """General Matrix-Vector multiplication: C[1,N] = A[1,K] @ B[K,N].

    Args:
        lhs: Row vector tile [1, K]
        rhs: Right-hand side tile [K, N]

    Returns:
        Tile wrapping the gemv operation
    """
    call_expr = _ir_ops.gemv(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def gemv_acc(acc: Tile, lhs: Tile, rhs: Tile) -> Tile:
    """GEMV with accumulation: C[1,N] += A[1,K] @ B[K,N].

    Args:
        acc: Accumulator tile [1, N]
        lhs: Row vector tile [1, K]
        rhs: Right-hand side tile [K, N]

    Returns:
        Tile wrapping the gemv_acc operation
    """
    call_expr = _ir_ops.gemv_acc(acc.unwrap(), lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def gemv_bias(lhs: Tile, rhs: Tile, bias: Tile) -> Tile:
    """GEMV with bias add: C[1,N] = A[1,K] @ B[K,N] + bias[1,N].

    Args:
        lhs: Row vector tile [1, K]
        rhs: Right-hand side tile [K, N]
        bias: Bias tile [1, N]

    Returns:
        Tile wrapping the gemv_bias operation
    """
    call_expr = _ir_ops.gemv_bias(lhs.unwrap(), rhs.unwrap(), bias.unwrap())
    return Tile(expr=call_expr)


def row_max(tile: Tile, tmp_tile: Tile) -> Tile:
    """Row-wise max reduction.

    Args:
        tile: Input tile
        tmp_tile: Temporary tile

    Returns:
        Tile wrapping the row_max operation
    """
    call_expr = _ir_ops.row_max(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def row_sum(tile: Tile, tmp_tile: Tile) -> Tile:
    """Row-wise sum reduction.

    Args:
        tile: Input tile
        tmp_tile: Temporary tile

    Returns:
        Tile wrapping the row_sum operation
    """
    call_expr = _ir_ops.row_sum(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def row_min(tile: Tile, tmp_tile: Tile) -> Tile:
    """Row-wise min reduction.

    Args:
        tile: Input tile
        tmp_tile: Temporary tile

    Returns:
        Tile wrapping the row_min operation
    """
    call_expr = _ir_ops.row_min(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def col_sum(tile: Tile, tmp_tile: Tile | None = None) -> Tile:
    """Column-wise sum reduction.

    Passing ``tmp_tile`` activates the binary-tree reduction path; omitting it
    uses the sequential path.

    Args:
        tile: Input tile
        tmp_tile: Optional scratch tile (same shape/dtype as input) that selects
            the binary-tree reduction path.

    Returns:
        Tile wrapping the col_sum operation
    """
    tmp_expr = None if tmp_tile is None else tmp_tile.unwrap()
    call_expr = _ir_ops.col_sum(tile.unwrap(), tmp_expr)
    return Tile(expr=call_expr)


def col_max(tile: Tile) -> Tile:
    """Column-wise max reduction.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the col_max operation
    """
    call_expr = _ir_ops.col_max(tile.unwrap())
    return Tile(expr=call_expr)


def col_min(tile: Tile) -> Tile:
    """Column-wise min reduction.

    Args:
        tile: Input tile

    Returns:
        Tile wrapping the col_min operation
    """
    call_expr = _ir_ops.col_min(tile.unwrap())
    return Tile(expr=call_expr)


def maximum(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise maximum of two tiles.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the maximum operation
    """
    call_expr = _ir_ops.maximum(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def row_expand(target: Tile, row_vec: Tile) -> Tile:
    """Expand row vector to target shape.

    Args:
        target: Target tile defining output shape [M, N]
        row_vec: Row vector to expand [M, 1]

    Returns:
        Tile wrapping the row_expand operation
    """
    call_expr = _ir_ops.row_expand(target.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_sub(tile: Tile, row_vec: Tile) -> Tile:
    """Row-wise broadcast subtraction.

    Args:
        tile: Input tile [M, N]
        row_vec: Row vector [M, 1]

    Returns:
        Tile wrapping the row_expand_sub operation
    """
    call_expr = _ir_ops.row_expand_sub(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_div(tile: Tile, row_vec: Tile) -> Tile:
    """Row-wise broadcast division.

    Args:
        tile: Input tile [M, N]
        row_vec: Row vector [M, 1]

    Returns:
        Tile wrapping the row_expand_div operation
    """
    call_expr = _ir_ops.row_expand_div(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_mul(tile: Tile, row_vec: Tile) -> Tile:
    """Row-wise broadcast multiplication.

    Args:
        tile: Input tile [M, N]
        row_vec: Row vector [M, 1]

    Returns:
        Tile wrapping the row_expand_mul operation
    """
    call_expr = _ir_ops.row_expand_mul(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_add(tile: Tile, row_vec: Tile) -> Tile:
    """Row-wise broadcast addition.

    Args:
        tile: Input tile [M, N]
        row_vec: Row vector [M, 1]

    Returns:
        Tile wrapping the row_expand_add operation
    """
    call_expr = _ir_ops.row_expand_add(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand(target: Tile, col_vec: Tile) -> Tile:
    """Expand column vector to target shape.

    Args:
        target: Target tile defining output shape [M, N]
        col_vec: Column vector to expand [1, N]

    Returns:
        Tile wrapping the col_expand operation
    """
    call_expr = _ir_ops.col_expand(target.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand_mul(tile: Tile, col_vec: Tile) -> Tile:
    """Expand column vector and multiply with tile.

    Args:
        tile: Input tile [M, N]
        col_vec: Column vector [1, N]

    Returns:
        Tile wrapping the col_expand_mul operation
    """
    call_expr = _ir_ops.col_expand_mul(tile.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand_div(tile: Tile, col_vec: Tile) -> Tile:
    """Expand column vector and divide tile by it.

    Args:
        tile: Input tile [M, N]
        col_vec: Column vector [1, N]

    Returns:
        Tile wrapping the col_expand_div operation
    """
    call_expr = _ir_ops.col_expand_div(tile.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand_sub(tile: Tile, col_vec: Tile) -> Tile:
    """Expand column vector and subtract from tile.

    Args:
        tile: Input tile [M, N]
        col_vec: Column vector [1, N]

    Returns:
        Tile wrapping the col_expand_sub operation
    """
    call_expr = _ir_ops.col_expand_sub(tile.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def expands(target: Tile, scalar: int | float | Expr | Scalar) -> Tile:
    """Expand scalar to target tile shape.

    Args:
        target: Target tile defining output shape
        scalar: Scalar value to expand

    Returns:
        Tile wrapping the expands operation
    """
    scalar_expr = scalar.unwrap() if isinstance(scalar, Scalar) else scalar
    call_expr = _ir_ops.expands(target.unwrap(), scalar_expr)
    return Tile(expr=call_expr)


def minimum(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise minimum of two tiles.

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


def sum(tile: Tile, axis: int, keepdim: bool = False) -> Tile:
    """Sum reduction along specified axis.

    Args:
        tile: Input tile
        axis: Reduction axis (0 for rows, 1 for columns, -1 for last)
        keepdim: Whether to keep the reduced dimension as 1

    Returns:
        Tile wrapping the sum operation
    """
    call_expr = _ir_ops.sum(tile.unwrap(), _axis_to_int(axis), keepdim)
    return Tile(expr=call_expr)


@overload
def max(tile: Tile, axis: int, keepdim: bool = False) -> Tile: ...


@overload
def max(tile: Scalar, axis: Scalar | int, keepdim: bool = False) -> Scalar: ...


def max(tile: Tile | Scalar, axis: int | Scalar | Expr = 0, keepdim: bool = False) -> Tile | Scalar:
    """Max reduction along specified axis, or scalar max of two values.

    Args:
        tile: Input tile or first scalar operand
        axis: Reduction axis (for tiles) or second scalar operand
        keepdim: Whether to keep the reduced dimension as 1 (tiles only)

    Returns:
        Tile or Scalar wrapping the max operation
    """
    if isinstance(tile, Scalar):
        return Scalar(expr=_ir_core.max_(tile.unwrap(), _scalar_operand_to_expr(axis)))
    call_expr = _ir_ops.max(tile.unwrap(), _axis_to_int(axis), keepdim)
    return Tile(expr=call_expr)


@overload
def min(tile: Tile, axis: int, keepdim: bool = False) -> Tile: ...


@overload
def min(tile: Scalar, axis: Scalar | int, keepdim: bool = False) -> Scalar: ...


@overload
def min(tile: int, axis: Scalar | int, keepdim: bool = False) -> Scalar: ...


def min(
    tile: Tile | Scalar | int | Expr,
    axis: int | Scalar | Expr = 0,
    keepdim: bool = False,
) -> Tile | Scalar:
    """Min reduction along specified axis, or scalar min of two values.

    Args:
        tile: Input tile or first scalar operand
        axis: Reduction axis (for tiles) or second scalar operand
        keepdim: Whether to keep the reduced dimension as 1 (tiles only)

    Returns:
        Tile or Scalar wrapping the min operation
    """
    if isinstance(tile, (Scalar, int, Expr)):
        lhs = _scalar_operand_to_expr(tile)
        rhs = _scalar_operand_to_expr(axis)
        return Scalar(expr=_ir_core.min_(lhs, rhs))
    call_expr = _ir_ops.min(tile.unwrap(), _axis_to_int(axis), keepdim)
    return Tile(expr=call_expr)


def slice(
    tile: Tile,
    shape: Sequence[IntLike],
    offset: Sequence[IntLike],
    valid_shape: Sequence[IntLike] | None = None,
    pad_value: PadValue | int | float | None = None,
) -> Tile:
    """Create a slice of a tile with static shape and optional valid shape.

    Args:
        tile: Input tile
        shape: Static shape dimensions (at most 2 for TileType)
        offset: Offset dimensions for the slice
        valid_shape: Valid shape dimensions. When omitted, shape is reused as the
            logical valid shape.
        pad_value: Optional padding mode for out-of-valid-shape elements.
            ``None`` or ``PadValue.null`` means no padding (the default).
            Accepts ``PadValue.zero`` / ``PadValue.max`` / ``PadValue.min``, or
            the literal sugars ``0``, ``math.inf``, ``-math.inf`` (same
            spelling as :func:`tile.fillpad`). Only meaningful when
            ``valid_shape`` is smaller than ``shape``.

    Returns:
        Tile wrapping the slice operation
    """
    if pad_value is not None and pad_value is not PadValue.null and valid_shape is None:
        warnings.warn(
            f"tile.slice received pad_value={pad_value!r} but no valid_shape. "
            f"pad_value has no effect unless valid_shape is smaller than shape. "
            f"If you intend to narrow the valid region later via "
            f"tile.set_validshape, you can ignore this warning; otherwise "
            f"pass valid_shape=... to tile.slice.",
            stacklevel=2,
        )

    tile_expr = tile.unwrap()
    normalized_valid_shape = None if valid_shape is None else _normalize_intlike(valid_shape)
    call_expr = _ir_ops.slice(
        tile_expr,
        _normalize_intlike(shape),
        _normalize_intlike(offset),
        normalized_valid_shape,
        pad_value=pad_value,
    )
    return Tile(expr=call_expr)


def reshape(tile: Tile, shape: Sequence[IntLike]) -> Tile:
    """Reshape tile to new shape.

    Args:
        tile: Input tile
        shape: New shape dimensions (at most 2 for TileType)

    Returns:
        Tile wrapping the reshape operation
    """
    tile_expr = tile.unwrap()
    call_expr = _ir_ops.reshape(tile_expr, _normalize_intlike(shape))
    return Tile(expr=call_expr)


def transpose(tile: Tile, axis1: int, axis2: int) -> Tile:
    """Transpose tile by swapping two axes.

    Args:
        tile: Input tile
        axis1: First axis to swap (supports negative indexing)
        axis2: Second axis to swap (supports negative indexing)

    Returns:
        Tile wrapping the transpose operation
    """
    tile_expr = tile.unwrap()
    call_expr = _ir_ops.transpose(tile_expr, axis1, axis2)
    return Tile(expr=call_expr)


def set_validshape(tile: Tile, valid_rows: IntLike, valid_cols: IntLike) -> Tile:
    """Update valid-shape metadata of a tile without data movement.

    .. note::
        Internal API — this op is intended for compiler-generated code only
        and should not be exposed to end users in future releases.

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


def rem(lhs: Tile, rhs: Tile) -> Tile:
    """Element-wise remainder (modulo) of two tiles.

    Computes lhs % rhs element-wise. Maps to the TREM hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile

    Returns:
        Tile wrapping the rem operation
    """
    call_expr = _ir_ops.rem(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def rems(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    """Element-wise remainder (modulo) of tile and scalar.

    Computes lhs % rhs element-wise. Maps to the TREMS hardware intrinsic.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the rems operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.rems(lhs.unwrap(), rhs_expr)
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


def maxs(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    """Element-wise maximum of tile and scalar.

    Computes max(lhs, rhs) element-wise. Maps to the TMAXS hardware intrinsic.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the maxs operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.maxs(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def mins(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    """Element-wise minimum of tile and scalar.

    Computes min(lhs, rhs) element-wise. Maps to the TMINS hardware intrinsic.

    Args:
        lhs: Tile
        rhs: Scalar value

    Returns:
        Tile wrapping the mins operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.mins(lhs.unwrap(), rhs_expr)
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
    """Element-wise addition of three tiles.

    Computes lhs + rhs + rhs2 element-wise. Maps to the TADDC hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile
        rhs2: Third tile

    Returns:
        Tile wrapping the addc operation
    """
    call_expr = _ir_ops.addc(lhs.unwrap(), rhs.unwrap(), rhs2.unwrap())
    return Tile(expr=call_expr)


def subc(lhs: Tile, rhs: Tile, rhs2: Tile) -> Tile:
    """Element-wise subtraction of three tiles.

    Computes lhs - rhs - rhs2 element-wise. Maps to the TSUBC hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Right-hand side tile
        rhs2: Third tile

    Returns:
        Tile wrapping the subc operation
    """
    call_expr = _ir_ops.subc(lhs.unwrap(), rhs.unwrap(), rhs2.unwrap())
    return Tile(expr=call_expr)


def addsc(lhs: Tile, rhs: int | float | Expr | Scalar, rhs2: Tile) -> Tile:
    """Element-wise addition of tile, scalar, and tile.

    Computes lhs + rhs + rhs2 element-wise. Maps to the TADDSC hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Scalar value
        rhs2: Third tile

    Returns:
        Tile wrapping the addsc operation
    """
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.addsc(lhs.unwrap(), rhs_expr, rhs2.unwrap())
    return Tile(expr=call_expr)


def subsc(lhs: Tile, rhs: int | float | Expr | Scalar, rhs2: Tile) -> Tile:
    """Element-wise subtraction of tile, scalar, and tile.

    Computes lhs - rhs - rhs2 element-wise. Maps to the TSUBSC hardware intrinsic.

    Args:
        lhs: Left-hand side tile
        rhs: Scalar value
        rhs2: Third tile

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
        tmp: Scratch tile required by TSEL (UINT8 [1, 32] on A2/A3)

    Returns:
        Tile wrapping the sel operation
    """
    call_expr = _ir_ops.sel(mask.unwrap(), lhs.unwrap(), rhs.unwrap(), tmp.unwrap())
    return Tile(expr=call_expr)


def sels(lhs: Tile, rhs: Tile, select_mode: int | float | Expr | Scalar) -> Tile:
    """Select between two tiles based on a scalar mode.

    Maps to the TSELS hardware intrinsic. The interpretation of select_mode values
    is target-dependent and enforced by codegen.

    Args:
        lhs: Source tile 0
        rhs: Source tile 1
        select_mode: Scalar select mode

    Returns:
        Tile wrapping the sels operation
    """
    select_mode_expr = select_mode.unwrap() if isinstance(select_mode, Scalar) else select_mode
    call_expr = _ir_ops.sels(lhs.unwrap(), rhs.unwrap(), select_mode_expr)
    return Tile(expr=call_expr)


def sort32(src: Tile, idx: Tile) -> Tile:
    """Sort fixed 32-element blocks with explicit index tile.

    Sorts 32-element blocks in src, permuting idx alongside.
    Returns sorted value-index pairs tile with doubled last dimension.

    For FP16 src: initialize idx with [0, 1, 2, ..., 31] per block.
    For FP32 src: initialize idx with [0, 2, 4, ..., 62] per block.

    Args:
        src: Input value tile (FP16 or FP32)
        idx: Input index tile with sequential offsets

    Returns:
        Tile wrapping the sort32 operation (last dim doubled)
    """
    call_expr = _ir_ops.sort32(src.unwrap(), idx.unwrap())
    return Tile(expr=call_expr)


@overload
def gather(src: Tile, indices: Tile, tmp: Tile) -> Tile: ...


@overload
def gather(src: Tile, *, mask_pattern: int, output_dtype: int | DataType | None = None) -> Tile: ...


def gather(
    src: Tile,
    indices: Tile | None = None,
    tmp: Tile | None = None,
    *,
    mask_pattern: int | None = None,
    output_dtype: int | DataType | None = None,
) -> Tile:
    """Gather elements from src tile, using either indices or a fixed mask pattern.

    Index form: dst[i, j] = src[indices[i, j]]. Requires indices and tmp workspace.
    Mask form: selects elements by a hardware mask pattern. No indices or tmp needed.

    Args:
        src: Source tile (FP16, FP32, INT16, or INT32)
        indices: Index tile (INT32). Required for index form.
        tmp: Temporary workspace tile (INT32). Required for index form.
        mask_pattern: Mask pattern selector (1-7), keyword-only. Use for mask form.
            1=P0101, 2=P1010, 3=P0001, 4=P0010, 5=P0100, 6=P1000, 7=P1111
        output_dtype: Optional output dtype for mask form only. When provided, the result
            tile has this dtype instead of src's dtype (bit reinterpretation, no conversion).
            Hardware requires sizeof(dst_dtype) == sizeof(src_dtype). Example: use
            output_dtype=pl.UINT32 to extract sort32 index bits from FP32 memory.

    Returns:
        Tile with gathered elements

    Examples:
        # Index form
        out = gather(src, indices, tmp)

        # Mask form (same dtype)
        out = gather(src, mask_pattern=1)

        # Mask form with cross-type output (FP32 bits → UINT32)
        out = gather(src, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.UINT32)
    """
    if mask_pattern is not None:
        if indices is not None or tmp is not None:
            raise ValueError(
                "gather() mask form (mask_pattern=...) and index form (indices, tmp) "
                "are mutually exclusive; do not pass indices or tmp with mask_pattern"
            )
        call_expr = _ir_ops.gather(src.unwrap(), mask_pattern=mask_pattern, output_dtype=output_dtype)
        return Tile(expr=call_expr)
    if output_dtype is not None:
        raise ValueError("output_dtype is only valid for the mask form of gather(); use mask_pattern=<int>")
    if indices is None or tmp is None:
        raise ValueError(
            "gather() requires either (indices, tmp) for index form, or mask_pattern=<int> for mask form"
        )
    call_expr = _ir_ops.gather(src.unwrap(), indices.unwrap(), tmp.unwrap())
    return Tile(expr=call_expr)


def mscatter(src: Tile, idx: Tile, output_tensor: Tensor) -> Tensor:
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
    return Tensor(expr=call_expr)


@overload
def mrgsort(src0: Tile, *, block_len: int | Scalar) -> Tile: ...


@overload
def mrgsort(
    src0: Tile,
    src1: Tile,
    *,
    tmp: Tile,
    executed: Tile,
    exhausted: bool = ...,
) -> Tile: ...


@overload
def mrgsort(
    src0: Tile,
    src1: Tile,
    src2: Tile,
    *,
    tmp: Tile,
    executed: Tile,
    exhausted: bool = ...,
) -> Tile: ...


@overload
def mrgsort(
    src0: Tile,
    src1: Tile,
    src2: Tile,
    src3: Tile,
    tmp: Tile,
    executed: Tile,
    exhausted: bool = ...,
) -> Tile: ...


def mrgsort(
    src0: Tile,
    src1: Tile | None = None,
    src2: Tile | None = None,
    src3: Tile | None = None,
    tmp: Tile | None = None,
    executed: Tile | None = None,
    exhausted: bool = False,
    *,
    block_len: int | Scalar | None = None,
) -> Tile:
    """Merge sort — format1 (single-list) or format2 (2-4 way merge).

    Format1: sorts a tile containing multiple pre-sorted runs of length block_len.
    Format2: merges 2, 3, or 4 pre-sorted input tiles into one sorted output.

    Format1 usage (keyword block_len):
        out = mrgsort(src, block_len=64)

    Format2 2-way usage (keyword tmp and executed):
        out = mrgsort(src0, src1, tmp=tmp_tile, executed=exec_tile)
        out = mrgsort(src0, src1, tmp=tmp_tile, executed=exec_tile, exhausted=True)

    Format2 3-way usage:
        out = mrgsort(src0, src1, src2, tmp=tmp_tile, executed=exec_tile)

    Format2 4-way usage (6 positional args):
        out = mrgsort(src0, src1, src2, src3, tmp, executed)
        out = mrgsort(src0, src1, src2, src3, tmp, executed, exhausted=True)

    Args:
        src0: For format1: input tile with pre-sorted runs (FP16 or FP32).
              For format2: first sorted input tile.
        src1: (format2) Second sorted input tile.
        src2: (format2, optional) Third sorted input tile (3-way or 4-way).
        src3: (format2, optional) Fourth sorted input tile (4-way only).
        tmp: (format2) Temporary workspace tile (same shape as output).
              Pass as keyword arg for 2-way and 3-way.
        executed: (format2) Exhaustion status tile written by hardware (shape [1, 4] INT16).
                  Pass as keyword arg for 2-way and 3-way.
        exhausted: (format2) If True, marks inputs as exhausted (default: False).
        block_len: (format1, keyword-only) Run length, must be multiple of 64.

    Returns:
        Tile with merged sorted elements
    """
    if block_len is not None:
        # format1: single-list merge sort
        if any(arg is not None for arg in (src1, src2, src3, tmp, executed)):
            raise ValueError(
                "mrgsort() format1 (block_len=...) and format2 (src1, ..., tmp, executed) "
                "are mutually exclusive; do not pass format2 arguments with block_len"
            )
        block_len_expr = block_len.unwrap() if isinstance(block_len, Scalar) else block_len
        call_expr = _ir_ops.mrgsort(src0.unwrap(), block_len=block_len_expr)
        return Tile(expr=call_expr)
    # format2: 2-4 way merge
    if src1 is None:
        raise ValueError(
            "mrgsort() requires either block_len=<int> for format1, "
            "or at least (src0, src1, tmp=<tile>, executed=<tile>) for format2"
        )
    if tmp is None or executed is None:
        raise ValueError(
            "mrgsort() format2 requires tmp and executed; "
            "use mrgsort(src0, src1[, src2[, src3]], tmp=<tile>, executed=<tile>)"
        )
    call_expr = _ir_ops.mrgsort(
        src0.unwrap(),
        src1.unwrap(),
        src2.unwrap() if src2 is not None else None,
        src3.unwrap() if src3 is not None else None,
        tmp=tmp.unwrap(),
        executed=executed.unwrap(),
        exhausted=exhausted,
    )
    return Tile(expr=call_expr)
