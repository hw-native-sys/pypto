# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Provider boundaries for direct preparation and immutable input capture."""

from dataclasses import dataclass
from typing import Protocol

from .contracts import KernelRequest, PreparationIdentity, PreparedKernel


@dataclass(frozen=True)
class CapturedInputs:
    """Provider-encoded snapshot of source and referenced inputs.

    The payload owns its bytes; no AST, live tensor, temporary directory or
    JIT context crosses this boundary. Its SHA-256 must equal kernel_digest.
    complete remains the provider's promise that no compiler input was omitted,
    including dependencies outside the payload. An incomplete capture is valid
    for preparation, but cannot authorize preparation-cache reuse.
    """

    payload: bytes
    identity: PreparationIdentity


class CapturingKernelProvider(Protocol):
    def capture(self, kernel: object, request: KernelRequest) -> CapturedInputs:
        """Snapshot provider-owned inputs without compiling the kernel."""
        ...

    def prepare(self, captured: CapturedInputs, request: KernelRequest) -> PreparedKernel:
        """Compile only the captured inputs and normalized request."""
        ...


class ExternalKernelProvider(Protocol):
    def prepare(self, kernel: object, request: KernelRequest) -> PreparedKernel:
        """Compile an opaque provider-owned reference without mutating JIT state."""
        ...
