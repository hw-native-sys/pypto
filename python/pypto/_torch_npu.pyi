# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Optional native queue adapter; built only with PYPTO_BUILD_TORCH_NPU."""

from typing import Any

import torch

simpler_revision: str

class LaunchTicket:
    """Native call snapshot retained by the process manager through completion."""

    def enqueue(self) -> None: ...
    def done(self) -> bool: ...
    def wait(self) -> None: ...
    def quiesce(self) -> None: ...

def framework_alive() -> bool: ...
def borrow_context(device_id: int) -> int: ...
def bind_context(context: int) -> None: ...
def retain_until_exit(owner: Any) -> None: ...
def check_call(stream_id: int, device_id: int) -> int: ...
def prepare(
    worker: Any,
    callable_id: int,
    tensors: list[torch.Tensor],
    dtypes: list[int],
    scalar_bits: list[int],
    stream_id: int,
    device_id: int,
) -> LaunchTicket: ...
