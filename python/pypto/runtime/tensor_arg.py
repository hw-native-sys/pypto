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
calling ``make_tensor_arg(orch._worker, tensors["<name>"])`` for every tensor
parameter. Current simpler deliberately separates this address-free wire
conversion from the direct-chip ``make_chip_tensor_arg`` helper.
This pypto-owned wrapper widens that conversion to also accept worker-resident
:class:`~pypto.runtime.DeviceTensor` handles (and already-built simpler
``Tensor`` values), so distributed programs can be invoked with
pre-uploaded device buffers — mirroring the L2 path in
:func:`pypto.runtime.runner.execute_compiled`.

Host ``torch.Tensor`` arguments are delegated to
``simpler_setup.torch_interop.make_tensor_arg(worker, tensor)``. Never route
this path through :mod:`pypto.runtime.task_interface`: its compatibility
``make_tensor_arg`` alias is chip-only and produces a ``ChipTensor``.
"""

import weakref
from functools import cache
from typing import Any

_PYPTO_OWNER_REF_ATTR = "_pypto_tensor_owner_ref"


def bind_tensor_arg_owner(worker: Any, owner: Any) -> None:
    """Bind a raw simpler Worker to its weak PyPTO owner.

    Generated orchestration receives the raw simpler Worker (``orch._worker``),
    while DeviceTensor liveness is tracked by the public PyPTO Worker.  This
    weak backlink lets wire conversion consult the authoritative PyPTO Buffer
    registry without introducing a reference cycle or changing generated code.
    """
    setattr(worker, _PYPTO_OWNER_REF_ATTR, weakref.ref(owner))
    owner._tensor_arg_worker = worker


def _require_device_tensor_owner(worker: Any, arg: Any) -> None:
    """Require *arg* to be a live DeviceTensor owned by *worker*'s PyPTO wrapper."""
    worker_dict = getattr(worker, "__dict__", None)
    owner_ref = worker_dict.get(_PYPTO_OWNER_REF_ATTR) if isinstance(worker_dict, dict) else None
    if not isinstance(owner_ref, weakref.ReferenceType):
        raise TypeError(
            "DeviceTensor dispatch requires the owning PyPTO Worker; a one-shot or raw simpler "
            "Worker cannot prove that the retained Buffer is live. Dispatch through the same "
            "PyPTO Worker that allocated the tensor."
        )
    owner = owner_ref()
    if owner is None:
        raise ValueError("DeviceTensor's owning PyPTO Worker no longer exists.")
    if getattr(owner, "_tensor_arg_worker", None) is not worker:
        raise ValueError(
            "DeviceTensor dispatch received a stale simpler Worker backend; use the owning PyPTO "
            "Worker's current initialized backend."
        )
    owner._require_owned_resident_tensor(arg, "Tensor argument")


@cache
def _modules() -> tuple[Any, Any, Any]:
    """Import and cache runtime modules on first ``make_tensor_arg`` call.

    The imports stay inside this function so importing pypto never requires
    simpler (only available in the runtime environment). ``functools.cache``
    runs the body once — instead of on every call — which matters because the
    generated ``host_orch`` calls ``make_tensor_arg`` once per tensor per rank
    (~90 tensors × world_size), where per-call ``from ... import`` was pure
    overhead on the host dispatch loop.

    Only the *module objects* are cached; individual symbols are resolved via
    attribute access on every call. This keeps the helper responsive to test
    monkeypatches while still paying the import cost only once.

    Returns:
        ``(task_interface, device_tensor, torch_interop)`` module objects.
    """
    from simpler_setup import torch_interop  # pyright: ignore[reportMissingImports]  # noqa: PLC0415

    from . import device_tensor, task_interface  # noqa: PLC0415

    return task_interface, device_tensor, torch_interop


def make_tensor_arg(worker: Any, arg: Any) -> Any:
    """Convert an orchestration tensor argument into a simpler ``Tensor``.

    Args:
        worker: The raw simpler Worker performing dispatch. DeviceTensor
            conversion requires it to be bound to the same live PyPTO Worker
            that allocated the Buffer; host tensors use it directly for
            simpler's worker-aware conversion.
        arg: One of:
            - ``torch.Tensor``: a CPU-contiguous host tensor (delegated to
              simpler's worker-aware wire helper).
            - :class:`~pypto.runtime.DeviceTensor`: a worker-resident buffer;
              after owner/liveness validation, its retained ``Buffer``
              constructs an address-free wire ``Tensor`` (memory is
              caller-managed).
            - simpler ``Tensor``: returned as-is (already device-side).

    Returns:
        A simpler ``Tensor`` ready to add to ``TaskArgs``.
    """
    task_interface, device_tensor, torch_interop = _modules()

    if isinstance(arg, task_interface.Tensor):
        return arg
    if isinstance(arg, device_tensor.DeviceTensor):
        if arg.buffer is None:
            raise TypeError(
                "A raw-pointer DeviceTensor cannot cross the public Worker.run wire ABI; "
                "allocate it with Worker.alloc_tensor() so it retains its simpler Buffer."
            )
        _require_device_tensor_owner(worker, arg)
        try:
            dtype = task_interface.torch_dtype_to_datatype(arg.dtype)
        except KeyError as e:
            raise ValueError(f"Unsupported DeviceTensor dtype: {arg.dtype}") from e
        return arg.buffer.tensor(shapes=arg.shape, dtype=dtype)
    return torch_interop.make_tensor_arg(worker, arg)
