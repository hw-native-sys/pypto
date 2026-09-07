# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Per-call namespace snapshots shared by JIT key construction and specialization."""

from collections import ChainMap
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")
_UNBOUND = object()


@dataclass
class _Namespaces:
    modules: dict[int, dict[str, Any]] = field(default_factory=dict)
    functions: dict[Any, Mapping[str, Any]] = field(default_factory=dict)
    values: dict[tuple[Any, Any], Any] = field(default_factory=dict)


_ACTIVE_NAMESPACES: ContextVar[_Namespaces | None] = ContextVar("jit_source_namespaces", default=None)


@contextmanager
def capture_namespaces() -> Iterator[None]:
    """Keep namespace reads consistent within one synchronous compilation request.

    Nested helpers reuse the outer snapshot. Context-local storage keeps
    concurrent callers isolated; the snapshot is discarded even on failure.
    Only bindings are copied, not arbitrary mutable configuration objects.
    """
    if _ACTIVE_NAMESPACES.get() is not None:
        yield
        return
    token = _ACTIVE_NAMESPACES.set(_Namespaces())
    try:
        yield
    finally:
        _ACTIVE_NAMESPACES.reset(token)


def cache_in_snapshot(func: Callable[[T], R]) -> Callable[[T], R]:
    """Memoize a unary helper only within the current compilation request."""

    @wraps(func)
    def wrapped(obj: T) -> R:
        snapshot = _ACTIVE_NAMESPACES.get()
        if snapshot is None:
            with capture_namespaces():
                return wrapped(obj)
        key = (func, obj)
        if key not in snapshot.values:
            snapshot.values[key] = func(obj)
        return cast(R, snapshot.values[key])

    return wrapped


def function_namespace(func: Any) -> Mapping[str, Any]:
    """Return captured module globals with closure bindings taking precedence."""
    snapshot = _ACTIVE_NAMESPACES.get()
    if snapshot is not None and func in snapshot.functions:
        return snapshot.functions[func]

    module_globals = getattr(func, "__globals__", {})
    namespace: Mapping[str, Any]
    if snapshot is None:
        namespace = dict(module_globals)
    else:
        module_id = id(module_globals)
        if module_id not in snapshot.modules:
            snapshot.modules[module_id] = dict(module_globals)
        namespace = snapshot.modules[module_id]

    freevars = getattr(getattr(func, "__code__", None), "co_freevars", ())
    closure = getattr(func, "__closure__", None) or ()
    bindings: dict[str, Any] = {}
    for name, cell in zip(freevars, closure, strict=True):
        try:
            bindings[name] = cell.cell_contents
        except ValueError:
            # An empty lexical cell still shadows the module global.
            bindings[name] = _UNBOUND
    if bindings:
        namespace = ChainMap(bindings, namespace)
    if snapshot is not None:
        snapshot.functions[func] = namespace
    return namespace
