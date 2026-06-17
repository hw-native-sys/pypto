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
``pld.<category>.<op>`` registered C++ ops as Python builder functions.

Layering (mirrors the ``pl.<ns>.<op>`` stack):

* This module (`pypto.ir.op.distributed`) — raw IR builders that take
  ``ir.Expr`` arguments, call :func:`ir.create_op_call`, and return ``ir.Call``.
* The DSL layer (`pypto.language.distributed.op`) — thin wrappers that accept
  DSL types (``DistributedTensor``, ``Tile``, ``Scalar``, …) and delegate
  here after unwrapping. ``pld.<op>`` unified dispatch re-exports the short
  form, parallel to ``pl.<op>``.
* The parser dispatches `pld.<category>.<op>` calls through a generic
  3-segment helper and `pld.<op>` through the unified-dispatch path.
"""

from . import system_ops, tensor_ops, tile_ops
from .system_ops import get_comm_ctx, nranks, rank, world_size
from .tensor_ops import alloc_window_buffer, allgather, allreduce, barrier, broadcast, get, put, reduce_scatter, window
from .tile_ops import remote_load

__all__ = [
    "allgather",
    "alloc_window_buffer",
    "allreduce",
    "barrier",
    "broadcast",
    "get",
    "get_comm_ctx",
    "nranks",
    "put",
    "rank",
    "reduce_scatter",
    "remote_load",
    "system_ops",
    "tensor_ops",
    "tile_ops",
    "window",
    "world_size",
]
