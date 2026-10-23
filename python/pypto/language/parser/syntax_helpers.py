# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AST spelling predicates and constants shared by the focused parsers."""

import ast
from collections.abc import Callable, Iterator
from typing import TypeGuard

from pypto.pypto_core import DataType, ir

# Canonical pld.<category>.<op> middle segments. Kept in sync with the
# submodules exposed by pypto.language.distributed.op (system_ops / tensor_ops
# / tile_ops); also surfaced as the hint in _parse_pld_category_op.
PLD_CATEGORIES: frozenset[str] = frozenset({"system", "tensor", "tile"})


# Enum values that op wrappers and printed ``attrs={...}`` dicts take verbatim.
# ``parse_expression`` cannot represent these: it either rejects the standalone
# attribute form outright or (for MemorySpace) lowers it to a ConstInt that the
# IR builders then reject, so every path that reads such a value must recognize
# it before falling back to expression parsing.
ENUM_VALUE_TYPES: tuple[type, ...] = (
    DataType,
    ir.MemorySpace,
    ir.TensorLayout,
    ir.TileLayout,
    ir.PadValue,
    ir.ArgDirection,
)


def is_empty_body(body: list[ast.stmt]) -> bool:
    """True if a function body carries no statements beyond a signature marker.

    Accepts a bare ``...`` (the documented spelling), a bare ``pass`` (what the
    IR printer emits for an empty body — required so external-kernel functions
    survive print -> reparse round-trips), and an optional leading docstring.

    Used to validate external-kernel declarations, whose implementation lives in
    a hand-written C++ source rather than a parsed DSL body.
    """
    non_doc = [
        stmt
        for stmt in body
        if not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
    ]
    if not non_doc:
        return True
    if len(non_doc) != 1:
        return False
    only = non_doc[0]
    if isinstance(only, ast.Pass):
        return True
    return isinstance(only, ast.Expr) and isinstance(only.value, ast.Constant) and only.value.value is ...


def is_pld_call(node: object, attr_name: str) -> TypeGuard[ast.Call]:
    """Match ``pld.<attr_name>(...)`` or ``pld.<category>.<attr_name>(...)``.

    Anchored on the literal ``pld.`` prefix; aliased imports don't match. For
    the 3-segment form the middle segment must be one of the canonical
    categories (``system`` / ``tensor`` / ``tile``) so typos like
    ``pld.typo.alloc_window_buffer(...)`` don't bypass dispatch.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != attr_name:
        return False
    parent = func.value
    # 2-segment: pld.<attr_name>
    if isinstance(parent, ast.Name) and parent.id == "pld":
        return True
    # 3-segment: pld.<category>.<attr_name>
    return (
        isinstance(parent, ast.Attribute)
        and parent.attr in PLD_CATEGORIES
        and isinstance(parent.value, ast.Name)
        and parent.value.id == "pld"
    )


def is_docstring_stmt(node: ast.stmt) -> bool:
    """True when ``node`` is a bare string expression, i.e. a docstring."""
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def is_pl_call(node: object, attr_name: str) -> TypeGuard[ast.Call]:
    """Return True when ``node`` is the AST for a ``pl.<attr_name>(...)`` call.

    Recognises only the dotted ``pl.<attr>`` form. Aliasing under another
    name (``from pypto.language import submit``) is intentionally not matched —
    the parser anchors construct detection on the ``pl.`` prefix.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == attr_name
        and isinstance(func.value, ast.Name)
        and func.value.id == "pl"
    )


def is_cache_policy_marker(node: object) -> TypeGuard[ast.Call]:
    """Return True for either spelling of the ``set_cache_policy`` marker call.

    ``pl.set_cache_policy(...)`` is the canonical form the printer emits, but the
    function lives in ``tensor_ops`` and so is equally reachable as
    ``pl.tensor.set_cache_policy(...)``. Both are recognised here — otherwise the
    submodule spelling would slip past the marker dispatch and fail as an
    ordinary evaluation statement with an error that names neither the marker nor
    its rules. Recognising it normalises it to ``pl.set_cache_policy`` on print.
    """
    if is_pl_call(node, "set_cache_policy"):
        return True
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "set_cache_policy"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "tensor"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "pl"
    )


def is_self_method_call(node: object) -> TypeGuard[ast.Call]:
    """Return True when ``node`` is the AST for a ``self.<method>(...)`` call.

    This is how the DSL spells a cross-function kernel call, so it is the marker
    that distinguishes a *dispatch* body from work written inline.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self"


def spmd_body_calls(body: "list[ast.stmt]", predicate: "Callable[[ast.Call], bool]") -> "Iterator[ast.Call]":
    """Yield each call in ``body`` satisfying ``predicate``.

    ``ast.walk`` recurses the whole body subtree, so a call nested inside a
    ``pl.range`` loop, an argument expression, or a nested ``with`` still counts.
    Shared by the SPMD body classifiers so they cannot drift apart.
    """
    for body_stmt in body:
        for node in ast.walk(body_stmt):
            if isinstance(node, ast.Call) and predicate(node):
                yield node


def spmd_body_has_call(body: "list[ast.stmt]", predicate: "Callable[[ast.Call], bool]") -> bool:
    """Return True when any call in ``body`` satisfies ``predicate``."""
    return next(spmd_body_calls(body, predicate), None) is not None


def is_const_int(value: object) -> bool:
    """Check if a value is a compile-time constant integer.

    Handles plain int, ir.ConstInt, and ir.Neg(ir.ConstInt) (negative literals).
    """
    if isinstance(value, (int, ir.ConstInt)):
        return True
    return isinstance(value, ir.Neg) and isinstance(value.operand, ir.ConstInt)


def is_dep_var_type(type_: ir.Type | None) -> bool:
    """Return True if ``type_`` is acceptable as a ``pl.submit(...)`` ``deps=`` entry.

    Two shapes are accepted:

    * ``ScalarType(TASK_ID)`` — a single TaskId (typically the ``tid`` bound
      by a prior ``_, tid = pl.submit(...)``, or the ``system.task_invalid``
      sentinel a bare ``None`` lowers to).
    * ``ArrayType(..., TASK_ID)`` — a per-slot TaskId array (typically from
      ``pl.array.create(N, pl.TASK_ID)`` threaded through a loop as an
      iter_arg). Codegen expands it to one entry per slot, each guarded by
      ``is_valid()`` (early-phase slots may still hold the invalid sentinel).
    """
    if isinstance(type_, ir.ScalarType) and type_.dtype == DataType.TASK_ID:
        return True
    if isinstance(type_, ir.ArrayType) and type_.dtype == DataType.TASK_ID:
        return True
    return False


def is_per_element_task_id_read(expr: ir.Expr) -> bool:
    """Return True if ``expr`` is ``array.get_element(arr, idx)`` reading a
    TASK_ID from an ``Array[_, TASK_ID]``.

    Accepted as a ``deps=`` entry under Form 1 (per-element subscript). The
    parser desugars these into a fresh ``Array[N, TASK_ID]`` populated by
    ``array.update_element`` calls so the existing whole-array dep codegen
    path can emit the per-slot ``is_valid()``-guarded ``set_dependencies``.
    """
    if not isinstance(expr, ir.Call):
        return False
    if expr.op.name != ir.get_op("array.get_element").name:
        return False
    if not isinstance(expr.type, ir.ScalarType) or expr.type.dtype != DataType.TASK_ID:
        return False
    if not expr.args:
        return False
    arr_type = expr.args[0].type
    return isinstance(arr_type, ir.ArrayType) and arr_type.dtype == DataType.TASK_ID


def is_dsl_call(node: ast.AST, func_name: str) -> bool:
    """Check if `node` is a call to a named DSL function (`pl.func_name(...)`
    or bare `func_name(...)`). Accepts either an `ast.Expr` statement (the
    body-stmt form, e.g. a bare `pl.cond(...)` line) or an expression node
    directly (e.g. the RHS of an `ast.Assign`).
    """
    if isinstance(node, ast.Expr):
        node = node.value
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == func_name
    if isinstance(func, ast.Name):
        return func.id == func_name
    return False
