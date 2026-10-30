# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Per-call namespace snapshots shared by JIT key construction and specialization."""

import inspect
import sys
from collections import ChainMap
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from types import CodeType, FunctionType
from typing import Any, TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")
_UNBOUND = object()


@dataclass
class _Namespaces:
    modules: dict[int, dict[str, Any]] = field(default_factory=dict[int, dict[str, Any]])
    functions: dict[Any, Mapping[str, Any]] = field(default_factory=dict[Any, Mapping[str, Any]])
    values: dict[tuple[Any, Any], Any] = field(default_factory=dict[tuple[Any, Any], Any])


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


_EMPTY = inspect.Signature.empty
_SIGNATURE_PROTOCOLS = ("__wrapped__", "__signature__", "__text_signature__", "__annotate__")
_SIGNATURES: dict[FunctionType, tuple[CodeType, dict[str, Any], inspect.Signature]] = {}


@cache_in_snapshot
def function_signature(func: Any) -> inspect.Signature:
    """Reuse plain positional signatures while observing current metadata bindings.

    Keep the request snapshot for all readers. More complex signatures retain
    inspect's original path. On a miss, validate the returned signature against
    the captured bindings before publishing: before/after binding equality alone
    would allow an intervening A-to-B-to-A mutation to poison future requests.
    """
    if type(func) is not FunctionType or sys.version_info >= (3, 14):
        return inspect.signature(func)
    attrs = func.__dict__
    if any(name in attrs for name in _SIGNATURE_PROTOCOLS):
        return inspect.signature(func)
    code = func.__code__
    defaults = func.__defaults__
    kwdefaults = func.__kwdefaults__
    annotations = func.__annotations__
    if (
        defaults is not None
        or kwdefaults is not None
        or code.co_kwonlyargcount
        or code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS)
        or type(annotations) is not dict
    ):
        return inspect.signature(func)
    # A shallow copy gives this guard stable keys during concurrent dict edits.
    annotations = annotations.copy()
    # exact str keys avoid arbitrary hash/equality callbacks, including on hits.
    if any(type(key) is not str for key in annotations):
        return inspect.signature(func)
    entry = _SIGNATURES.get(func)
    if entry is not None:
        old_code, saved, signature = entry
        if (
            code is old_code
            and len(annotations) == len(saved)
            and all(key in annotations and annotations[key] is value for key, value in saved.items())
        ):
            return signature
    # Copy before inspect. Never associate its result with this copy blindly.
    saved = annotations
    signature = inspect.signature(func)
    if type(signature) is not inspect.Signature:
        return signature
    parameters = tuple(signature.parameters.values())
    if len(parameters) != code.co_argcount:
        return signature
    for i, parameter in enumerate(parameters):
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if i < code.co_posonlyargcount
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        if (
            type(parameter) is not inspect.Parameter
            or parameter.name != code.co_varnames[i]
            or parameter.kind != kind
            or parameter.default is not _EMPTY
            or parameter.annotation is not saved.get(parameter.name, _EMPTY)
        ):
            return signature
    if signature.return_annotation is not saved.get("return", _EMPTY):
        return signature
    # Publication validates actual contents, not just before/after tokens.
    # Eviction only affects performance; replace one complete immutable record.
    if len(_SIGNATURES) >= 512:
        _SIGNATURES.clear()
    _SIGNATURES[func] = (code, saved, signature)
    return signature


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
