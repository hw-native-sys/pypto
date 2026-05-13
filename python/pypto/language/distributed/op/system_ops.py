# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Distributed system-level op sentinels (``pld.<op>``).

System-level ops cover cross-rank synchronization and runtime queries.
Currently exposes :func:`world_size`; N6 will add ``notify`` / ``wait``
(``pld.system.notify`` / ``pld.system.wait``) here as well.

* ``world_size`` — host-only scalar returning the number of devices in
  the current distributed execution. The parser lifts
  ``pld.world_size()`` to ``ir.OpExpr('pld.world_size')`` of type
  :class:`ir.ScalarType` (INT64). Codegen later lowers each call site
  to ``len(contexts)``.

Typical use sites for ``world_size``:

* loop bounds: ``for r in pl.range(pld.world_size()): ...``
* allocation sizes (in bytes): ``pld.alloc_window_buffer(pld.world_size() * 4)``
* per-rank tensor shapes: ``pld.window(buf, [pld.world_size()], dtype=pl.INT32)``
"""

from pypto.pypto_core.ir import Expr


def world_size() -> Expr:
    """Return the distributed world size as a scalar INT64 expression.

    Must be called inside a function annotated with
    ``@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)``. The
    parser intercepts the call before Python runtime ever sees it; calling
    ``pld.world_size()`` outside a host-level orchestrator body raises.

    Returns:
        An :class:`ir.Expr` of type ``ScalarType(INT64)`` whose value is the
        number of devices in the current distributed execution.

    Raises:
        RuntimeError: Always — this function is a parser sentinel.
    """
    raise RuntimeError(
        "pld.world_size() must be called inside a @pl.function (level=Level.HOST, role=Role.Orchestrator)"
    )


__all__ = ["world_size"]
