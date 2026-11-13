# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Orchestrator APIs inspected during Host-domain artifact loading."""

from collections.abc import Callable, Sequence
from typing import Any

class Orchestrator:
    def allocate_domain(
        self,
        *,
        name: str,
        members: Sequence[Any] | None = None,
        window_size: int,
        buffers: Sequence[Any] = (),
        retain_after_run: bool = False,
        workers: Sequence[int] | None = None,
        workspace_engines: Sequence[str] = (),
        memory_source: Any = ...,
        initializer: Callable[[Any], None] | None = None,
    ) -> Any: ...
    def get_or_allocate_domain(
        self,
        *,
        name: str,
        members: Sequence[Any],
        window_size: int,
        buffers: Sequence[Any] = (),
        retain_after_run: bool = False,
        workspace_engines: Sequence[str] = (),
        memory_source: Any = ...,
        initializer: Callable[[Any], None] | None = None,
    ) -> Any: ...
