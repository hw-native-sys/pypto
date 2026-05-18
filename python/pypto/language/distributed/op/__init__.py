# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Distributed-DSL op sentinels (``pld.<op>``, ``pld.tile.<op>``, ``pld.comm_ctx.<op>``).

Parser sentinels that lift to ``ir.OpExpr(pld.<op>)`` nodes. Files are
grouped by op category (mirroring ``pypto.language.op``):

* ``memory_ops`` — :func:`alloc_window_buffer`, :func:`window` (CommGroup
  window-buffer allocation and view materialisation).
* ``system_ops`` — :func:`world_size`, :func:`get_comm_ctx`. N6 will add
  ``pld.system.notify`` / ``pld.system.wait`` here as well.
* ``tile_ops`` — cross-rank tile ops, exposed as the ``tile`` sub-namespace
  (``pld.tile.remote_load`` ...).
* ``comm_ctx_ops`` — CommContext scalar accessors, exposed as the
  ``comm_ctx`` sub-namespace (``pld.comm_ctx.rank`` / ``pld.comm_ctx.nranks``).
"""

from . import comm_ctx_ops as comm_ctx
from . import tile_ops as tile
from .memory_ops import alloc_window_buffer, window
from .system_ops import get_comm_ctx, world_size

__all__ = [
    "alloc_window_buffer",
    "comm_ctx",
    "get_comm_ctx",
    "tile",
    "window",
    "world_size",
]
