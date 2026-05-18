# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unified ``pld.<op>`` dispatch — short-form re-exports for the distributed DSL.

Unlike :mod:`pypto.language.op.unified_ops` (which does runtime type-dispatch
between ``pl.tensor`` / ``pl.tile``), every short op name in ``pld`` maps to
exactly one category, so the short form is a plain re-export from the
canonical 3-segment surface — preserving signatures and docstrings for IDE
help with zero call-chain indirection.
"""

from .system_ops import get_comm_ctx, nranks, rank, world_size
from .tensor_ops import alloc_window_buffer, window
from .tile_ops import remote_load

__all__ = [
    "alloc_window_buffer",
    "get_comm_ctx",
    "nranks",
    "rank",
    "remote_load",
    "window",
    "world_size",
]
