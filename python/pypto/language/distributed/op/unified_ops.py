# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unified ``pld.<op>`` dispatch — short-form entry points for the distributed DSL.

Most short op names in ``pld`` map to exactly one category, so the short form is
a plain re-export from the canonical 3-segment surface — preserving signatures
and docstrings for IDE help with zero call-chain indirection.

``remote_store`` is the exception: it exists at both IR levels
(``pld.tile.remote_store`` / ``pld.tensor.remote_store``), so the short form
type-dispatches on its source operand the way
:mod:`pypto.language.op.unified_ops` does for ``pl.add`` and friends. The two
forms have the same argument surface and the same semantics — only the level of
``src`` differs — so the dispatch cannot silently change what a call means.
"""

from collections.abc import Sequence

from pypto.language.typing import IntLike, Tile
from pypto.language.typing.tensor import Tensor
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import AtomicType, Call, Expr

from ..typing.distributed_tensor import DistributedTensor
from . import tensor_ops as _tensor
from . import tile_ops as _tile
from .system_ops import get_comm_ctx, nranks, rank, world_size
from .tensor_ops import alloc_window_buffer, window
from .tile_ops import remote_load

__all__ = [
    "alloc_window_buffer",
    "get_comm_ctx",
    "nranks",
    "rank",
    "remote_load",
    "remote_store",
    "window",
    "world_size",
]


def remote_store(
    src: Tile | Tensor | Expr,
    target: DistributedTensor,
    peer: IntLike,
    offsets: Sequence[IntLike],
    *,
    atomic: AtomicType = AtomicType.None_,
) -> Call:
    """Push a local value into a region of ``peer`` rank's slice of ``target``.

    Dispatches on the IR level of ``src``:

    * a :class:`pl.Tile` (``@pl.jit.incore`` / ``@pl.program``) routes to
      :func:`pld.tile.remote_store`;
    * a tensor-level value (``@pl.jit``) routes to
      :func:`pld.tensor.remote_store`, which ``ConvertTensorToTileOps`` lowers
      1:1 to the tile form.

    Either way the value reaches the peer as a single ``pto.tstore``, with no
    global-memory round-trip. See the two canonical entry points for the full
    argument documentation.

    Args:
        src: Local 2-D :class:`pl.Tile` or :class:`pl.Tensor` value (dtype must
            match ``target.dtype``).
        target: Window-bound :class:`pld.DistributedTensor` destination (rank >= 2).
        peer: Peer rank index.
        offsets: Offsets into the remote slice, one per ``target`` dimension.
        atomic: :class:`pld.AtomicType` selecting plain-store (the default) vs
            atomic-add combine semantics on the peer's region (keyword-only).

    Returns:
        A side-effect-only :class:`ir.Call` (no SSA result for downstream use).
    """
    if isinstance(src, Tile):
        return _tile.remote_store(src, target, peer, offsets, atomic=atomic)
    if isinstance(src, Tensor):
        return _tensor.remote_store(src, target, peer, offsets, atomic=atomic)
    # Raw ir.Expr (printer round-trip, hand-built IR): dispatch on the IR type.
    # An unrecognised operand falls through to the tensor form, whose deducer
    # produces the diagnostic naming both entry points.
    if isinstance(src, Expr) and isinstance(src.type, _ir.TileType):
        return _tile.remote_store(src, target, peer, offsets, atomic=atomic)
    return _tensor.remote_store(src, target, peer, offsets, atomic=atomic)
