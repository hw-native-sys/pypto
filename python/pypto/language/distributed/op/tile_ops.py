# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pld.tile.*`` — cross-rank tile DSL sentinels.

These functions are parser sentinels — calling them at Python runtime always
raises. They exist so that source code like::

    @pl.function(level=pl.Level.INCORE, role=pl.Role.Worker)
    def remote_read(self, data: pld.DistributedTensor[[N], pl.FP32], peer: pl.int32):
        local = pld.tile.remote_load(data, peer=peer, offsets=[0], shape=[N])
        ...

is syntactically valid Python that the AST parser intercepts and lifts into
an ``ir.OpExpr(pld.tile.remote_load)`` IR node.
"""

from collections.abc import Sequence

from pypto.language.typing import IntLike
from pypto.language.typing.tensor import Tensor
from pypto.language.typing.tile import Tile


def remote_load(
    target: Tensor,  # noqa: ARG001
    *,
    peer: IntLike,  # noqa: ARG001
    offsets: Sequence[IntLike],  # noqa: ARG001
    shape: Sequence[IntLike],  # noqa: ARG001
) -> Tile:
    """Load a region of ``peer`` rank's slice of a DistributedTensor into a local tile.

    Mirrors :func:`pl.tile.load` at the user-visible surface, but the source
    is a *remote* slice of a window-bound :class:`pld.DistributedTensor`.
    Address translation happens at codegen time via ``CommRemotePtr``.

    Args:
        target: A window-bound :class:`pld.DistributedTensor` (any rank, any
            dtype). At parse time the IR type is
            :class:`ir.DistributedTensorType`; the parser refuses plain
            :class:`pl.Tensor` here.
        peer: Peer rank index (kwarg-only). Accepts an ``int`` literal, a DSL
            ``Scalar``, or a raw ``ir.Expr`` (e.g. ``comm_ctx.rank + 1``).
        offsets: Offsets into the remote slice, one per ``target`` dimension.
        shape: Per-dimension shape of the tile to load. Determines the output
            :class:`pl.Tile` shape.

    Returns:
        A local :class:`pl.Tile` of the requested shape, dtype equal to
        ``target.dtype``.

    Raises:
        RuntimeError: Always — this function is a parser sentinel. The parser
            intercepts the call before Python ever invokes the body.
    """
    raise RuntimeError(
        "pld.tile.remote_load must be called inside a @pl.function (level=Level.INCORE, role=Role.Worker)"
    )


__all__ = ["remote_load"]
