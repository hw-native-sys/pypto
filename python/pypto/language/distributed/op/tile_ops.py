# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pld.tile.*`` — cross-rank tile DSL wrappers.

Each wrapper accepts DSL types, unwraps to ``ir.Expr``, and delegates to the
matching IR builder in :mod:`pypto.ir.op.distributed.tile_ops`.
"""

from collections.abc import Sequence

from pypto.ir.op.distributed import tile_ops as _ir_tile
from pypto.language.typing import IntLike, Tile
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import Expr

from ..typing.distributed_tensor import DistributedTensor
from ._utils import _normalize_intlike, _unwrap


def remote_load(
    target: DistributedTensor,
    *,
    peer: IntLike,
    offsets: Sequence[IntLike],
    shape: Sequence[IntLike],
) -> Tile:
    """Load a region of ``peer`` rank's slice of a DistributedTensor into a local tile.

    Mirrors :func:`pl.tile.load` at the user-visible surface, but the source
    is a *remote* slice of a window-bound :class:`pld.DistributedTensor`.
    Address translation happens at codegen time via ``CommRemotePtr``.

    Args:
        target: A window-bound :class:`pld.DistributedTensor` (any rank, any
            dtype). The C++ verifier refuses plain :class:`pl.Tensor` here
            (precise ObjectKind match on :class:`ir.DistributedTensorType`).
        peer: Peer rank index (kwarg-only). Accepts an ``int`` literal, a DSL
            ``Scalar``, or a raw ``ir.Expr`` (e.g. ``pld.rank(ctx) + 1``).
        offsets: Offsets into the remote slice, one per ``target`` dimension.
        shape: Per-dimension shape of the tile to load. Determines the output
            :class:`pl.Tile` shape.

    Returns:
        A local :class:`pl.Tile` of the requested shape, dtype equal to
        ``target.dtype``.
    """
    target_expr = _unwrap(target)
    if not isinstance(target_expr, Expr) or not isinstance(target_expr.type, _ir.DistributedTensorType):
        got = (
            _ir.python_print_type(target_expr.type)
            if isinstance(target_expr, Expr)
            else type(target_expr).__name__
        )
        raise TypeError(f"pld.tile.remote_load expects a DistributedTensor target (window-bound); got {got}")

    call = _ir_tile.remote_load(
        target_expr, _unwrap(peer), _normalize_intlike(offsets), _normalize_intlike(shape)
    )
    return Tile(expr=call)


__all__ = ["remote_load"]
