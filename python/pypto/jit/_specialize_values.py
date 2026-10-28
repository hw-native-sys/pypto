# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Canonical value rendering shared by specialized source and cache keys.

Keep constant folding and cache fingerprints on these same renderers so a
changed captured value cannot change the source without changing its key.
"""

import ast
import enum
import functools
import types
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import TensorLayout

from ._source import _UNBOUND

# ---------------------------------------------------------------------------
# Free-name folding
# ---------------------------------------------------------------------------
#
# The generated ``@pl.program`` source is parsed in a namespace holding only
# ``pl`` and ``pld``, so any name the body inherits from its own module or an
# enclosing function is undefined by the time the parser reads it. Such a name
# has to be replaced, at its use site, by source text that evaluates back to the
# same value -- there is nowhere else to put it, because a multi-function
# ``@pl.jit`` program merges deps that may come from different modules and whose
# namespaces can therefore disagree about a name.
#
# Everything below answers one question: can this value be written as generated
# source? Values that cannot be are left alone, so the name survives and the
# parser reports it exactly as before.


@functools.lru_cache(maxsize=1)
def _pl_symbol_index() -> tuple[dict[int, str], dict[Any, str]]:
    """Index public ``pl`` and ``pld`` values by their canonical source paths.

    Returns ``(enum_classes, values)``: ``id(EnumClass) -> "pl.Mem"`` for
    exported enum types, and ``value -> "pl.INT8"`` for exported singletons
    (in practice the ``DataType`` constants). Visit ``pl`` first to preserve
    existing spellings for shared exports such as ``AtomicType``. Within each
    namespace, sorted names make alias selection deterministic (``pl.Mem``
    precedes ``pl.MemorySpace``), matching the IR printer.

    Callables, classes and modules are excluded: a body naming one is not naming
    a *value* the DSL can consume, and folding it would only turn a clear
    "undefined name" into a confusing type error further in.
    """
    import pypto.language as pl  # noqa: PLC0415 — avoids a circular import at module load
    from pypto.language import distributed as pld  # noqa: PLC0415 — same circular import

    enum_classes: dict[int, str] = {}
    values: dict[Any, str] = {}
    for prefix, namespace in (("pl", pl), ("pld", pld)):
        for attr in sorted(dir(namespace)):
            if attr.startswith("_"):
                continue
            value = getattr(namespace, attr, None)
            path = f"{prefix}.{attr}"
            if isinstance(value, type):
                if issubclass(value, enum.Enum):
                    enum_classes.setdefault(id(value), path)
                continue
            # Enum members go through their class (above); literals render as
            # literals; callables and modules are not values.
            if isinstance(value, (enum.Enum, types.ModuleType, bool, int, float, str)) or callable(value):
                continue
            try:
                values.setdefault(value, path)
            except TypeError:
                continue  # unhashable — not reachable by value lookup
    return enum_classes, values


def _pl_symbol_path(value: Any) -> str | None:
    """The ``pl.`` or ``pld.`` path that evaluates to ``value``, or None."""
    enum_classes, values = _pl_symbol_index()
    if isinstance(value, enum.Enum):
        cls_path = enum_classes.get(id(type(value)))
        return f"{cls_path}.{value.name}" if cls_path else None
    try:
        return values.get(value)
    except TypeError:
        return None  # unhashable, or a type whose __eq__ rejects the comparison


def render_free_value(value: Any) -> ast.expr | None:
    """Render ``value`` as generated-source AST, or None when it cannot be written.

    Covers what a DSL body actually passes to an operator: literals, the ``pl``
    and ``pld`` enum and ``DataType`` constants, and (possibly nested) lists/tuples
    of those -- a shape held in a module constant, for instance. A sequence renders only
    when every element does, so a half-rendered list can never reach the parser.
    """
    if isinstance(value, type):
        return None
    if isinstance(value, enum.Enum):
        path = _pl_symbol_path(value)
        return ast.parse(path, mode="eval").body if path else None
    if value is None or isinstance(value, (bool, int, float, str)):
        # Non-finite floats included: ``ast.unparse`` writes them as evaluable
        # expressions (``1e309``, ``-1e309``, ``(1e309-1e309)``), not as the bare
        # names ``inf`` / ``nan``, so a fill or padding value bound to one folds
        # like any other float. Declining them here would have regressed support
        # that predates this function.
        return ast.Constant(value=value)
    if isinstance(value, (list, tuple)):
        elements = [render_free_value(element) for element in cast("Sequence[Any]", value)]
        if any(element is None for element in elements):
            return None
        rendered = cast("list[ast.expr]", elements)
        if isinstance(value, list):
            return ast.List(elts=rendered, ctx=ast.Load())
        return ast.Tuple(elts=rendered, ctx=ast.Load())
    path = _pl_symbol_path(value)
    return ast.parse(path, mode="eval").body if path else None


def _bind_free_name(name: str, py_globals: Mapping[str, Any]) -> ast.expr | None:
    """Render what ``name`` is bound to in ``py_globals``, or None if it does not fold.

    None covers all three ways a name can fail to fold: it is not bound here, it
    is an unbound closure cell, or its value has no source form.
    """
    if name not in py_globals:
        return None
    value = py_globals[name]
    if value is _UNBOUND:
        return None
    return render_free_value(value)


def constant_source(value: Any) -> str | None:
    """The generated-source text ``value`` folds to, or None when it does not fold.

    The value-keyed sibling of :func:`free_name_source`, for a constant the call
    site supplies directly rather than binds to a name — a ``pl.constexpr``
    argument. Sharing :func:`render_free_value` is the point: a constexpr
    parameter folds by exactly the rule a module-level constant follows, so the
    two can never admit different value types.
    """
    rendered = render_free_value(value)
    return None if rendered is None else ast.unparse(rendered)


def constant_expr(value: Any, node: ast.expr) -> ast.expr | None:
    """:func:`constant_source` as AST, positioned at ``node`` so spans survive."""
    rendered = render_free_value(value)
    if rendered is None:
        return None
    return ast.fix_missing_locations(ast.copy_location(rendered, node))


def free_name_source(name: str, py_globals: Mapping[str, Any]) -> str | None:
    """The generated-source text ``name`` folds to, or None when it does not fold.

    The JIT cache key reads this so that a constant can never reach the generated
    source without also reaching the hash: rebinding ``MODE`` from ``"trunc"`` to
    ``"round"`` must invalidate the artifact exactly as rebinding an ``int``
    extent does. Hashing the rendered *text* rather than the value keeps the two
    definitions from drifting — the key depends on precisely what gets emitted.
    """
    rendered = _bind_free_name(name, py_globals)
    return None if rendered is None else ast.unparse(rendered)


def fold_free_name(name: str, py_globals: Mapping[str, Any], node: ast.expr) -> ast.expr | None:
    """:func:`_bind_free_name`, positioned at ``node`` so spans survive."""
    rendered = _bind_free_name(name, py_globals)
    if rendered is None:
        return None
    return ast.fix_missing_locations(ast.copy_location(rendered, node))


# ---------------------------------------------------------------------------
# DataType → pl.XXX string mapping
# ---------------------------------------------------------------------------

_DTYPE_TO_PL: dict[DataType, str] = {
    DataType.FP4E2M1X2: "pl.FP4E2M1X2",
    DataType.FP4E1M2X2: "pl.FP4E1M2X2",
    DataType.FP8E4M3FN: "pl.FP8E4M3FN",
    DataType.FP8E5M2: "pl.FP8E5M2",
    DataType.FP8E8M0: "pl.FP8E8M0",
    DataType.FP16: "pl.FP16",
    DataType.FP32: "pl.FP32",
    DataType.BF16: "pl.BF16",
    DataType.HF4: "pl.HF4",
    DataType.HF8: "pl.HF8",
    DataType.INT4: "pl.INT4",
    DataType.INT8: "pl.INT8",
    DataType.INT16: "pl.INT16",
    DataType.INT32: "pl.INT32",
    DataType.INT64: "pl.INT64",
    DataType.UINT4: "pl.UINT4",
    DataType.UINT8: "pl.UINT8",
    DataType.UINT16: "pl.UINT16",
    DataType.UINT32: "pl.UINT32",
    DataType.UINT64: "pl.UINT64",
    DataType.BOOL: "pl.BOOL",
    DataType.INDEX: "pl.INDEX",
}


# Set of bare dtype name strings (e.g. "FP32", "INT8") for annotation parsing.
# Derived from _DTYPE_TO_PL to avoid duplication.
DTYPE_NAMES: frozenset[str] = frozenset(v.split(".")[1] for v in _DTYPE_TO_PL.values())


def dtype_str(dt: DataType) -> str:
    if dt not in _DTYPE_TO_PL:
        raise ValueError(f"Unsupported DataType: {dt}")
    return _DTYPE_TO_PL[dt]


# TensorLayout → pl.XXX string mapping, for regenerating an annotation's
# third slot. ND is absent on purpose: it is the default the slot means when
# omitted, so emitting it would only add noise.
_LAYOUT_TO_PL: dict[TensorLayout, str] = {
    TensorLayout.DN: "pl.DN",
    TensorLayout.NZ: "pl.NZ",
    TensorLayout.MX_A_ZZ: "pl.MX_A_ZZ",
    TensorLayout.MX_B_NN: "pl.MX_B_NN",
}


def layout_str(layout: TensorLayout) -> str | None:
    """Render a layout to its ``pl.<NAME>`` annotation form.

    Returns:
        The ``pl.<NAME>`` source form, or None when the layout needs no slot
        (ND — what an omitted slot already means).

    Raises:
        ValueError: If the layout has no DSL constant to render
    """
    if layout == TensorLayout.ND:
        return None
    if layout not in _LAYOUT_TO_PL:
        raise ValueError(f"Unsupported TensorLayout: {layout}")
    return _LAYOUT_TO_PL[layout]


def array_dtype_str(dt: DataType) -> str:
    """Render an ``Array`` element dtype to its ``pl.<NAME>`` source form.

    Prefers the canonical :data:`_DTYPE_TO_PL` mapping so dtypes whose enum
    ``str`` diverges from the DSL constant render correctly (e.g. ``BF16`` →
    ``pl.BF16``, not the enum's ``str`` form ``bfloat16`` → ``pl.BFLOAT16``,
    which does not exist on ``pl``). Falls back to the upper-cased enum string
    for dtypes outside that map — notably ``TASK_ID``, the TaskId-array element
    type, which is intentionally absent from ``_DTYPE_TO_PL`` (it is not a
    tensor/tile element dtype but does have a ``pl.TASK_ID`` constant).
    """
    return _DTYPE_TO_PL.get(dt, f"pl.{str(dt).upper()}")
