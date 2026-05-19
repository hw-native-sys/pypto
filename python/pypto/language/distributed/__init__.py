# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO distributed DSL — namespace ``pypto.language.distributed`` (alias ``pld``).

Provides cross-rank concepts that complement the single-device DSL in
``pypto.language``. Communication-domain metadata (``ir.CommGroup`` /
``ir.WindowBuffer``) is **inferred** by the ``CollectCommGroups`` pass from
``pld.tensor.alloc_window_buffer`` calls in the host orchestrator and the
``device=`` kwarg on dispatch sites; users do not declare ``CommGroup``
manually.

Package layout mirrors :mod:`pypto.language` (3-segment ``pld.<category>.<op>``
plus 2-segment unified-dispatch short form, just like ``pl``):

* :mod:`pypto.language.distributed.op` — parser-sentinel ops. Sub-namespaces:
  ``pld.system.*`` (:func:`world_size`, :func:`get_comm_ctx`, :func:`rank`,
  :func:`nranks`), ``pld.tensor.*`` (:func:`alloc_window_buffer`,
  :func:`window`), and ``pld.tile.*`` (:func:`remote_load`). Per-file split
  mirrors the C++ side (``src/ir/op/distributed/``).
* :mod:`pypto.language.distributed.typing` — DSL type wrappers
  (:class:`DistributedTensor`, :class:`CommCtx`).
* :class:`NotifyOp` / :class:`WaitCmp` — typed enum payloads of
  ``pld.system.notify`` / ``pld.system.wait``, re-exported here so users can
  write ``pld.NotifyOp.AtomicAdd`` / ``pld.WaitCmp.Ge`` without reaching into
  ``pypto.pypto_core.ir``.
"""

from pypto.pypto_core.ir import NotifyOp, WaitCmp

from .op import (
    alloc_window_buffer,
    get_comm_ctx,
    nranks,
    rank,
    remote_load,
    system,
    tensor,
    tile,
    window,
    world_size,
)
from .typing import CommCtx, DistributedTensor

__all__ = [
    "CommCtx",
    "DistributedTensor",
    "NotifyOp",
    "WaitCmp",
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
