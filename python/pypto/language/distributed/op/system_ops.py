# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Distributed system-level op DSL wrappers (``pld.<op>``).

System-level ops cover cross-rank synchronization and runtime queries.
Exposes :func:`world_size` and :func:`get_comm_ctx`; N6 will add
``notify`` / ``wait`` (``pld.system.notify`` / ``pld.system.wait``)
here as well. The ``pld.comm_ctx.rank`` / ``pld.comm_ctx.nranks``
accessors live in :mod:`.comm_ctx_ops`, exposed as the ``comm_ctx``
sub-namespace.

* ``world_size`` — host-only scalar returning the number of devices in the
  current distributed execution. Returns a :class:`Scalar` wrapping an
  :class:`ir.Expr` of type ``ScalarType(INT64)``. Codegen later lowers each
  call site to ``len(contexts)``.
* ``get_comm_ctx`` — lift a :class:`pld.DistributedTensor` to its
  :class:`pld.CommCtx` handle. The op verifier (C++) refuses any
  argument that is not :class:`ir.DistributedTensorType`.

Typical use sites for ``world_size``:

* loop bounds: ``for r in pl.range(pld.world_size()): ...``
* allocation sizes (in bytes): ``pld.alloc_window_buffer(pld.world_size() * 4)``
* per-rank tensor shapes: ``pld.window(buf, [pld.world_size()], dtype=pl.INT32)``
"""

from pypto.ir.op.distributed import system_ops as _ir_system
from pypto.language.typing import Scalar

from ..typing.comm_ctx import CommCtx
from ..typing.distributed_tensor import DistributedTensor
from .memory_ops import _unwrap


def world_size() -> Scalar:
    """Return the distributed world size as an ``INT64`` :class:`Scalar`.

    Parser context (host-only, not inside a nested device-side scope) is
    validated by the parser before this wrapper is invoked. The DSL-side
    return wrapping lets call sites compose naturally with Python operators
    (``pld.world_size() * 4``, ``pl.range(pld.world_size())``), which the
    parser's ``invoke_dsl`` unwraps back to the underlying Call.
    """
    return Scalar(expr=_ir_system.world_size())


def get_comm_ctx(dist_tensor: DistributedTensor) -> CommCtx:
    """Return the :class:`pld.CommCtx` handle of a window-bound DistributedTensor.

    The op verifier (C++) enforces that ``dist_tensor`` carries
    :class:`ir.DistributedTensorType` — passing a plain :class:`pl.Tensor`
    is rejected by precise-ObjectKind match.

    Args:
        dist_tensor: A :class:`pld.DistributedTensor` (annotation form or
            value wrapper). Raw :class:`ir.Expr` is also accepted for
            parser-side invocations that have already unwrapped.

    Returns:
        A :class:`pld.CommCtx` wrapping an :class:`ir.Call` of type
        :class:`ir.CommCtxType`. Pass to :func:`pld.comm_ctx.rank` /
        :func:`pld.comm_ctx.nranks` to read scalar fields.
    """
    return CommCtx(expr=_ir_system.get_comm_ctx(_unwrap(dist_tensor)))


__all__ = ["get_comm_ctx", "world_size"]
