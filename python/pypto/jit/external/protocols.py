# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Provider boundary. Registration, binding, and implementations follow in task 34.2."""

from typing import Protocol

from .contracts import KernelRequest, PreparedKernel


class ExternalKernelProvider(Protocol):
    def prepare(self, kernel: object, request: KernelRequest) -> PreparedKernel:
        """Compile an opaque provider-owned reference without mutating JIT state."""
        ...
