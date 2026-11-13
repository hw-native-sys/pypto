# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type declarations for parser-only IR dump syntax.

These markers describe ASTs consumed by @pl.program; they are not executable
Python functions. Result descriptors and call arguments can be any DSL/IR type.
"""

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

from pypto.pypto_core import ir

from .typing import Ptr

WindowBuffer: ir.WindowBufferType
Unknown: ir.UnknownType

def float64(bits: str) -> float: ...
def window_ref(key: str) -> ir.WindowBuffer: ...
def window_buffer(
    key: str,
    *,
    base: str,
    size: Any,
    load_from_host: bool = False,
    store_to_host: bool = False,
    memory_source: str = "device",
    init: str = "",
) -> ir.WindowBuffer: ...
def comm_domain(
    *,
    name: str,
    devices: Sequence[int],
    slots: Sequence[ir.WindowBuffer],
    engines: Sequence[str] = (),
    attrs: Mapping[str, Any] | None = None,
) -> AbstractContextManager[None]: ...
def alloc_window_buffer(
    size: Any, *, name: str, memory_source: str = "device", init: str | None = None
) -> Ptr: ...
def call(
    op: str | Callable[..., Any],
    args: Sequence[Any],
    result_type: Any,
    *,
    kwargs: Mapping[str, Any] | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any: ...
def submit(submission: Any, result_type: Any) -> Any: ...
def yield_(*args: Any, types: Sequence[Any]) -> Any: ...
