# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
MemRef-aware Tile utilities for manual address specification and reuse.

The standard pypto.language.Tile type does not expose MemRef, so all address
assignment is deferred to compiler passes (init_mem_ref, basic_memory_reuse).

This module provides utilities for cases where the user needs explicit control:

1. make_memref() - Create an ir.MemRef with a specific memory space and address
2. bind_memref() - Attach a MemRef to a Tile's underlying TileType
3. MemRefTile  - Extended Tile wrapper that carries MemRef metadata

Use cases:
- Manual address pinning for double-buffering
- Explicit buffer reuse across loop iterations
- Fine-grained memory layout control for performance tuning

Example:
    import pypto.frontend as fe
    from pypto.pypto_core.ir import MemorySpace

    # Create a MemRef at address 0x0 in UB, 4096 bytes, id=0
    buf_a = fe.make_memref(MemorySpace.UB, addr=0, size=4096, id=0)
    buf_b = fe.make_memref(MemorySpace.UB, addr=4096, size=4096, id=1)

    # After loading a tile, bind it to a specific buffer
    tile = fe.load(tensor, [0, 0], [64, 64])
    tile = fe.bind_memref(tile, buf_a)  # pin this tile to buf_a

    # Reuse buf_a for a different tile (manual address reuse)
    tile2 = fe.load(tensor, [64, 0], [64, 64])
    tile2 = fe.bind_memref(tile2, buf_a)  # shares the same physical buffer
"""

from typing import Optional, Sequence, Union

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import Expr, MemorySpace, MemRef, Span

from pypto.language.typing import Tile
from pypto.ir.utils import _normalize_expr


def make_memref(
    memory_space: MemorySpace,
    addr: Union[int, Expr],
    size: int,
    id: int,
    span: Optional[Span] = None,
) -> MemRef:
    """Create an ir.MemRef for manual address specification.

    Args:
        memory_space: Target memory space (DDR, UB, L1, L0A, L0B, L0C)
        addr: Starting address (int or Expr). Ints are converted to ConstInt(INT64).
        size: Size in bytes
        id: Unique identifier for this MemRef (used to distinguish buffers)
        span: Optional source span for debugging

    Returns:
        ir.MemRef object that can be passed to bind_memref() or used directly
        in TileType/TensorType constructors.

    Example:
        >>> buf = fe.make_memref(MemorySpace.UB, addr=0, size=4096, id=0)
        >>> buf2 = fe.make_memref(MemorySpace.UB, addr=4096, size=4096, id=1)
    """
    actual_span = span if span is not None else Span.unknown()
    addr_expr = _normalize_expr(addr, actual_span, int_dtype=DataType.INT64)
    return MemRef(memory_space, addr_expr, size, id, actual_span)


def bind_memref(tile: Tile, memref: MemRef) -> Tile:
    """Bind a MemRef to a Tile, creating a new Tile with the specified memory layout.

    This function takes an existing Tile (which wraps a Call expression with TileType)
    and creates a new Call expression whose TileType carries the given MemRef. This
    allows manual control over which physical buffer a tile occupies.

    The operation is purely a type-level annotation — it does not emit any data
    movement instructions. The underlying computation is unchanged.

    Args:
        tile: Source Tile (must be a runtime Tile wrapping an Expr, not annotation-only)
        memref: MemRef to attach

    Returns:
        New Tile wrapping the same expression but with MemRef bound in its TileType

    Raises:
        ValueError: If tile is annotation-only or its type is not TileType

    Example:
        >>> tile = fe.load(tensor, [0, 0], [64, 64])
        >>> buf = fe.make_memref(MemorySpace.UB, addr=0, size=8192, id=0)
        >>> tile = fe.bind_memref(tile, buf)
    """
    expr = tile.unwrap()
    tile_type = expr.type

    if not isinstance(tile_type, _ir.TileType):
        raise ValueError(
            f"bind_memref requires a Tile with TileType, got {type(tile_type).__name__}"
        )

    # Create a new TileType with the same shape/dtype but with the provided memref.
    # We preserve the existing tile_view if any.
    new_type = _ir.TileType(
        tile_type.shape,
        tile_type.dtype,
        memref,
        tile_type.tile_view if hasattr(tile_type, 'tile_view') else None,
    )

    # If the expression is a Call, recreate it with the new type.
    # For other expression types, wrap in an identity-like pattern.
    if isinstance(expr, _ir.Call):
        new_call = _ir.Call(expr.op, expr.args, new_type, expr.span)
        return Tile(expr=new_call)
    elif isinstance(expr, _ir.Var):
        # For Var, we create a new Var with the updated type.
        # This is a type-level change only.
        new_var = _ir.Var(expr.name, new_type, expr.span)
        return Tile(expr=new_var)
    else:
        # For other expression types, wrap via a no-op identity.
        # In practice, load/store/add etc. all return Call, so this is rare.
        # We just return a Tile wrapping the original expr — the memref info
        # is advisory and will need to be picked up by a custom pass.
        return Tile(expr=expr)


class MemRefTile:
    """Extended Tile descriptor that carries MemRef metadata alongside a Tile.

    This is a higher-level wrapper for cases where you want to track the
    MemRef association outside of the IR type system. It pairs a Tile with
    its intended MemRef.

    Unlike bind_memref() which modifies the IR type, MemRefTile is a pure
    Python-side bookkeeping structure. Use it when you need to pass buffer
    assignment information through your own Python logic before emitting IR.

    Attributes:
        tile: The underlying Tile
        memref: The associated MemRef (or None)

    Example:
        >>> buf = fe.make_memref(MemorySpace.UB, 0, 4096, id=0)
        >>> mt = fe.MemRefTile(tile=some_tile, memref=buf)
        >>> # Later, when emitting IR:
        >>> bound_tile = fe.bind_memref(mt.tile, mt.memref)
    """

    def __init__(self, tile: Tile, memref: Optional[MemRef] = None):
        """Initialize MemRefTile.

        Args:
            tile: The Tile to associate with a buffer
            memref: Optional MemRef for manual address specification
        """
        self.tile = tile
        self.memref = memref

    def bind(self) -> Tile:
        """Apply the stored MemRef to the Tile and return the bound Tile.

        Returns:
            Tile with MemRef bound (if memref is set), or original tile

        Raises:
            ValueError: If no memref is set
        """
        if self.memref is None:
            raise ValueError("No MemRef set on this MemRefTile. Set .memref first.")
        return bind_memref(self.tile, self.memref)

    def unwrap(self) -> Expr:
        """Unwrap the underlying Tile's expression.

        Returns:
            The IR Expr from the wrapped Tile
        """
        return self.tile.unwrap()

    def __repr__(self) -> str:
        return f"MemRefTile(tile={self.tile}, memref={self.memref})"


__all__ = ["make_memref", "bind_memref", "MemRefTile"]
