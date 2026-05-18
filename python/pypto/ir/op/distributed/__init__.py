# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR-layer builders for distributed ops (``pld.*``).

These are the namespace siblings of :mod:`pypto.ir.op.tensor_ops` /
:mod:`pypto.ir.op.tile_ops` / :mod:`pypto.ir.op.system_ops`, exposing the
``pld.<op>`` registered C++ ops as Python builder functions.

Layering (mirrors the ``pl.<ns>.<op>`` stack):

* This module (`pypto.ir.op.distributed`) — raw IR builders that take
  ``ir.Expr`` arguments, call :func:`ir.create_op_call`, and return ``ir.Call``.
* The DSL layer (`pypto.language.distributed.op`) — thin wrappers that accept
  DSL types (``DistributedTensor``, ``Tile``, ``Scalar``, …) and delegate
  here after unwrapping.
* The parser dispatches `pld.<op>` calls through `_dispatch_op` (the same
  helper used for `pl.tile.*` and friends), which routes via the DSL layer.
"""

from . import memory_ops, system_ops
from .memory_ops import alloc_window_buffer, window
from .system_ops import comm_ctx_nranks, comm_ctx_rank, get_comm_ctx, world_size

__all__ = [
    "alloc_window_buffer",
    "comm_ctx_nranks",
    "comm_ctx_rank",
    "get_comm_ctx",
    "memory_ops",
    "system_ops",
    "window",
    "world_size",
]
