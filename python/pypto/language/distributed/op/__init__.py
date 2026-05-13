# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Distributed-DSL op sentinels (``pld.<op>``).

Parser sentinels that lift to ``ir.OpExpr(pld.<op>)`` nodes. Files are
grouped by op category (mirroring ``pypto.language.op``):

* ``memory_ops`` — :func:`alloc_window_buffer`, :func:`window` (CommGroup
  window-buffer allocation and view materialisation).
* ``system_ops`` — :func:`world_size` today; N6 will add
  ``pld.system.notify`` / ``pld.system.wait`` here as well.
"""

from .memory_ops import alloc_window_buffer, window
from .system_ops import world_size

__all__ = ["alloc_window_buffer", "window", "world_size"]
