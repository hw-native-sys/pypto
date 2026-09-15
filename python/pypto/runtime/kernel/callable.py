# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Complete callable identities and process-bound registrations, never persisted handles."""

import hashlib
import importlib
from dataclasses import dataclass, field
from typing import Any

from pypto._kernel_abi import KernelABI


def callable_identity(callable_: Any, abi: KernelABI) -> bytes:
    """Hash all serialized callable bytes/signatures plus the PyPTO ABI descriptor."""
    sdk = importlib.import_module("simpler.callable_identity")
    descriptor = sdk.build_chip_callable_descriptor(
        target=callable_, platform=abi.platform, runtime=abi.runtime
    )
    return hashlib.sha256(abi.binary_tag() + descriptor).digest()


@dataclass(frozen=True)
class KernelRegistration:
    """Retain the image and Worker owner alongside its opaque context-local handle."""

    identity: bytes
    pid: int
    generation: int
    handle: Any
    owner: Any = field(repr=False, compare=False)
    callable: Any = field(repr=False, compare=False)
    artifact: Any = field(repr=False, compare=False)

    def require_live(self) -> None:
        """Reject stale or inherited registrations before a later launch adapter uses them."""
        self.owner.require_registration(self)
