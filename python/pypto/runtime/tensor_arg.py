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
:class:`~pypto.runtime.DeviceTensor` handles (and already-built
``ContinuousTensor`` values), so distributed programs can be invoked with
pre-uploaded device buffers — mirroring the L2 path in
:func:`pypto.runtime.runner.execute_compiled`.

Host ``torch.Tensor`` arguments are delegated unchanged to simpler's
``make_tensor_arg``; only the device-resident branches are added here.
"""

from typing import Any


def make_tensor_arg(arg: Any) -> Any:
    """Convert an orchestration tensor argument into a simpler ``ContinuousTensor``.

    Args:
        arg: One of:
            - ``torch.Tensor``: a CPU-contiguous host tensor (delegated to
              simpler's ``make_tensor_arg``, which performs the H2D copy).
            - :class:`~pypto.runtime.DeviceTensor`: a worker-resident buffer;
              wrapped as ``ContinuousTensor(child_memory=True)`` so the runtime
              skips H2D/D2H (memory is caller-managed).
            - ``ContinuousTensor``: returned as-is (already device-side).

    Returns:
        A simpler ``ContinuousTensor`` ready to add to ``TaskArgs``.
    """
    # Imports are lazy: simpler is only available in the runtime environment,
    # and pypto must remain importable without it.
    from .device_tensor import DeviceTensor  # noqa: PLC0415
    from .task_interface import (  # noqa: PLC0415
        ContinuousTensor,  # pyright: ignore[reportAttributeAccessIssue]
        device_tensor_to_continuous,
    )
    from .task_interface import (  # noqa: PLC0415
        make_tensor_arg as _impl,  # pyright: ignore[reportAttributeAccessIssue]
    )

    if isinstance(arg, ContinuousTensor):
        return arg
    if isinstance(arg, DeviceTensor):
        return device_tensor_to_continuous(arg)
    return _impl(arg)
