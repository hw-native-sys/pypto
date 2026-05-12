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
``pld.alloc_window_buffer`` calls in the host orchestrator and the
``device=`` kwarg on dispatch sites; users do not declare ``CommGroup``
manually.

This module is currently a namespace placeholder; concrete entry points
(``pld.DistributedTensor``, ``pld.alloc_window_buffer``, ``pld.world_size``,
``pld.tile.*``, ``pld.system.*``) are added in subsequent milestones (N1.2+).
"""

from .alloc import alloc_window_buffer, window
from .distributed_tensor import DistributedTensor

__all__ = ["DistributedTensor", "alloc_window_buffer", "window"]
