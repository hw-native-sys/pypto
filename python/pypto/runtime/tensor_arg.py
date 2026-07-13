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

Host ``torch.Tensor`` arguments are delegated unchanged to simpler's
``make_tensor_arg``; only the device-resident branches are added here.
"""

from functools import cache
from typing import Any


@cache
def _bindings() -> tuple[Any, Any, Any, Any]:
    """Resolve and cache the simpler symbols on first ``make_tensor_arg`` call.

    The imports stay inside this function so importing pypto never requires
    simpler (only available in the runtime environment). ``functools.cache``
    runs the body once — instead of on every call — which matters because the
    generated ``host_orch`` calls ``make_tensor_arg`` once per tensor per rank
    (~90 tensors × world_size), where per-call ``from ... import`` was pure
    overhead on the host dispatch loop.

    Returns:
        ``(Tensor, DeviceTensor, device_tensor_to_tensor, make_tensor_arg)``.
    """
    from .device_tensor import DeviceTensor  # noqa: PLC0415
    from .task_interface import (  # noqa: PLC0415
        Tensor,  # pyright: ignore[reportAttributeAccessIssue]
        device_tensor_to_tensor,
    )
    from .task_interface import (  # noqa: PLC0415
        make_tensor_arg as _impl,  # pyright: ignore[reportAttributeAccessIssue]
    )

    return Tensor, DeviceTensor, device_tensor_to_tensor, _impl


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
    tensor_cls, device_tensor_cls, to_tensor, impl = _bindings()

    if isinstance(arg, tensor_cls):
        return arg
    if isinstance(arg, device_tensor_cls):
        return to_tensor(arg)
    return impl(arg)
