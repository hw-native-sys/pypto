# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Distributed-DSL op sentinels — ``pld.<category>.<op>`` plus unified short form.

Mirrors :mod:`pypto.language.op` for the distributed namespace: the
3-segment surface (``pld.system.<op>`` / ``pld.tensor.<op>`` /
``pld.tile.<op>``) is exposed as real Python sub-modules, while the
2-segment short form (``pld.<op>``) is re-exported from
:mod:`.unified_ops` via name-based dispatch.

Sub-modules (one per op category):

* :mod:`.system_ops` — host queries and CommContext accessors
  (``world_size``, ``get_comm_ctx``, ``rank``, ``nranks``).
* :mod:`.tensor_ops` — CommGroup window-buffer allocation, view
  materialisation, and cross-rank tensor bulk communication
  (``alloc_window_buffer``, ``window``, ``get``, ``put``).
* :mod:`.tile_ops` — cross-rank tile ops (``remote_load``, ...).
"""

from . import system_ops as system
from . import tensor_ops as tensor
from . import tile_ops as tile
from .unified_ops import (
    alloc_window_buffer,
    get_comm_ctx,
    nranks,
    rank,
    remote_load,
    window,
    world_size,
)

__all__ = [
    "alloc_window_buffer",
    "get_comm_ctx",
    "nranks",
    "rank",
    "remote_load",
    "system",
    "tensor",
    "tile",
    "window",
    "world_size",
]
