# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``make_tensor_arg`` used by generated distributed orchestration code.

The generated ``orchestration/host_orch.py`` builds simpler ``TaskArgs`` by
calling ``make_tensor_arg(tensors["<name>"])`` for every tensor parameter.
This pypto-owned wrapper widens that conversion to also accept worker-resident
:class:`~pypto.runtime.DeviceTensor` handles (and already-built simpler
``Tensor`` values), so distributed programs can be invoked with
pre-uploaded device buffers — mirroring the L2 path in
:func:`pypto.runtime.runner.execute_compiled`.

Host ``torch.Tensor`` arguments are delegated to simpler's ``make_tensor_arg``,
which since the Buffer refactor takes ``(worker, tensor)`` — it memoizes the
tensor as a ``FORK_SHM`` handle *on* the worker. The generated code has no worker
in scope, so the caller binds one for the duration of the entry call through
:func:`bind_worker`; only the device-resident branches are otherwise added here.
"""

import contextvars
from functools import cache
from typing import Any

_CURRENT_WORKER: contextvars.ContextVar[Any] = contextvars.ContextVar("pypto_l3_worker", default=None)


def bind_worker(worker: Any) -> Any:
    """Make *worker* the target of host-tensor arg conversion; returns a reset token.

    A ContextVar rather than an attribute so nested or concurrent submissions each
    see their own worker, and so an exception on the entry path cannot leave a stale
    worker bound for the next request.
    """
    return _CURRENT_WORKER.set(worker)


def unbind_worker(token: Any) -> None:
    _CURRENT_WORKER.reset(token)


@cache
def _modules() -> tuple[Any, Any]:
    """Import and cache the two runtime modules on first ``make_tensor_arg`` call.

    The imports stay inside this function so importing pypto never requires
    simpler (only available in the runtime environment). ``functools.cache``
    runs the body once — instead of on every call — which matters because the
    generated ``host_orch`` calls ``make_tensor_arg`` once per tensor per rank
    (~90 tensors × world_size), where per-call ``from ... import`` was pure
    overhead on the host dispatch loop.

    Only the *module objects* are cached; the individual symbols (``Tensor``,
    ``DeviceTensor``, ``device_tensor_to_tensor``, ``make_tensor_arg``) are
    resolved via attribute access on every call. Caching the module rather than
    the bound symbols keeps ``make_tensor_arg`` responsive to test monkeypatches
    of ``task_interface.make_tensor_arg`` (see ``tests/ut/runtime``), while still
    paying the import cost only once.

    Returns:
        ``(task_interface, device_tensor)`` module objects.
    """
    from . import device_tensor, task_interface  # noqa: PLC0415

    return task_interface, device_tensor


def make_tensor_arg(arg: Any) -> Any:
    """Convert an orchestration tensor argument into a simpler ``Tensor``.

    Args:
        arg: One of:
            - ``torch.Tensor``: a CPU-contiguous host tensor (delegated to
              simpler's ``make_tensor_arg``, which performs the H2D copy).
            - :class:`~pypto.runtime.DeviceTensor`: a worker-resident buffer;
              wrapped as ``Tensor(child_memory=True)`` so the runtime
              skips H2D/D2H (memory is caller-managed).
            - simpler ``Tensor``: returned as-is (already device-side).

    Returns:
        A simpler ``Tensor`` ready to add to ``TaskArgs``.
    """
    task_interface, device_tensor = _modules()

    if isinstance(arg, task_interface.Tensor):
        return arg
    holder = _CURRENT_WORKER.get()
    if isinstance(arg, device_tensor.DeviceTensor):
        convert = getattr(holder, "device_tensor_arg", None)
        if convert is None:
            raise RuntimeError(
                "make_tensor_arg received a DeviceTensor but the bound submitter cannot name device "
                "memory. A Simpler wire Tensor is a view over the owning chip's Buffer, so the "
                "conversion needs the runtime that allocated it (DistributedWorker)."
            )
        return convert(arg)
    # A runner binds itself (it owns the device-buffer registry); the plain
    # non-persistent path binds a Simpler Worker directly.
    worker = getattr(holder, "_w", holder)
    if worker is None:
        raise RuntimeError(
            "make_tensor_arg received a host torch.Tensor with no worker bound. Simpler's "
            "make_tensor_arg memoizes the tensor as a FORK_SHM handle on a specific worker, so "
            "one has to be in scope: build task args inside a `with ChipWorker(...):` block at "
            "L2, or let DistributedWorker's submission path bind itself at L3. Building them "
            "before the worker exists — as execute_compiled's one-shot path still does — cannot "
            "work under the Buffer ABI."
        )
    return task_interface.make_tensor_arg(worker, arg)
