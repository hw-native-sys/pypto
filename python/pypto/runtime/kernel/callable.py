# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Complete callable identities and process-bound registrations, never persisted handles."""

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

from pypto.ir.compiled_artifact import CompiledArtifact


def artifact_identity(artifact: CompiledArtifact) -> Hashable:
    """Reuse the shared request key; private output directories are mutable."""
    runtime = artifact._artifact_runtime
    return artifact if runtime is None else runtime.handle.key


@dataclass(frozen=True)
class KernelRegistration:
    """Retain the image and Worker owner alongside its opaque context-local handle."""

    identity: Hashable
    pid: int
    generation: int
    handle: Any
    owner: Any = field(repr=False, compare=False)
    callable: Any = field(repr=False, compare=False)
    artifact: Any = field(repr=False, compare=False)

    def require_live(self) -> None:
        """Reject stale or inherited registrations before a later launch adapter uses them."""
        self.owner.require_registration(self)
