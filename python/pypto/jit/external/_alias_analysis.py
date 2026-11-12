# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Conservative syntactic alias checks; runtime overlap remains a bridge obligation."""

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from pypto.pypto_core.ir import ParamDirection

if TYPE_CHECKING:
    from ._call_normalizer import CallBinding


@dataclass(frozen=True, eq=False)
class _View:
    """A view's storage root and optional 1-D interval in its parent's coordinates.

    Parent identity is a snapshot, not a variable name. Only sibling slices with
    the same parent and constant nonnegative coordinates prove disjointness;
    reshapes, indexing and unknown slices start a new coordinate system.
    """

    root: str
    parent: "str | _View"
    interval: tuple[int, int] | None = None


_AliasValue: TypeAlias = str | _View | None
_AliasRoots: TypeAlias = _AliasValue | tuple["_AliasRoots", ...] | list["_AliasRoots"]


def _root(value: _AliasValue) -> str | None:
    return value.root if isinstance(value, _View) else value


def _slice_interval(call: ast.Call) -> tuple[int, int] | None:
    operands = {kw.arg: kw.value for kw in call.keywords}
    shape = call.args[1] if len(call.args) > 1 else operands.get("shape")
    offsets = call.args[2] if len(call.args) > 2 else operands.get("offset")
    clamp = call.args[6] if len(call.args) > 6 else operands.get("clamp")
    # Clamping can move the requested interval back into another view's region.
    if clamp is not None and not (isinstance(clamp, ast.Constant) and clamp.value is False):
        return None
    # Rank-one slices cannot hide overlap through another dimension or a reshape.
    # Dynamic/negative coordinates and rank reduction remain conservative.
    drop = call.args[4] if len(call.args) > 4 else operands.get("drop_dims")
    if drop is not None and not (
        isinstance(drop, ast.Constant)
        and drop.value is None
        or isinstance(drop, (ast.List, ast.Tuple))
        and not drop.elts
    ):
        return None
    if not all(isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) == 1 for node in (shape, offsets)):
        return None
    assert isinstance(shape, (ast.List, ast.Tuple)) and isinstance(offsets, (ast.List, ast.Tuple))
    size, start = shape.elts[0], offsets.elts[0]
    if not (
        isinstance(size, ast.Constant)
        and type(size.value) is int
        and size.value > 0
        and isinstance(start, ast.Constant)
        and type(start.value) is int
        and start.value >= 0
    ):
        return None
    return start.value, start.value + size.value


def _disjoint(left: _AliasValue, right: _AliasValue) -> bool:
    return (
        isinstance(left, _View)
        and isinstance(right, _View)
        and left.parent == right.parent
        and left.interval is not None
        and right.interval is not None
        and (left.interval[1] <= right.interval[0] or right.interval[1] <= left.interval[0])
    )


class AliasTracker:
    """Track straight-line aliases, invalidating rebindings across control flow.

    Undo journals touch only modified names. Work is linear in AST size times
    lexical nesting depth, without copying the complete local environment.
    """

    def __init__(self, parameter_names: list[str], fresh: Callable[[], str]) -> None:
        self.aliases: dict[str, _AliasRoots] = {name: name for name in parameter_names}
        self.journals: list[dict[str, _AliasRoots]] = []
        self.fresh = fresh

    def bind(self, name: str, root: _AliasRoots) -> None:
        for journal in self.journals:
            journal.setdefault(name, self.aliases.get(name))
        self.aliases[name] = root

    def root_of(self, expr: ast.expr) -> str | None:
        return _root(self.alias_of(expr))

    def alias_of(self, expr: ast.expr) -> _AliasValue:
        roots = self._roots_of(expr)
        return roots if isinstance(roots, (str, _View)) else None

    def _roots_of(self, expr: ast.expr) -> _AliasRoots:
        if isinstance(expr, ast.Name):
            return self.aliases.get(expr.id)
        if isinstance(expr, ast.Tuple):
            return tuple(self._roots_of(element) for element in expr.elts)
        if isinstance(expr, ast.Subscript):
            roots = self._roots_of(expr.value)
            if isinstance(roots, tuple):
                try:
                    index = ast.literal_eval(expr.slice)
                except (ValueError, TypeError, SyntaxError):
                    return None
                return roots[index] if isinstance(index, int) and -len(roots) <= index < len(roots) else None
            # Tensor indexing retains storage. Aggregate indexing must select
            # an element instead; a tuple itself is never one tensor root.
            return (
                _View(roots.root if isinstance(roots, _View) else roots, roots)
                if isinstance(roots, (str, _View))
                else None
            )
        if isinstance(expr, ast.Call):
            name = ast.unparse(expr.func)
            # Views and update results keep their target's storage. Store's
            # destination is its third operand; the first operand is a Tile.
            operands = {
                "slice": (0, {"tensor", "input"}),
                "reshape": (0, {"tensor", "input"}),
                "assemble": (0, {"target"}),
                "write": (0, {"tensor", "tile"}),
                "store": (2, {"output_tensor"}),
            }
            parts = name.split(".")
            if parts[:-1] in (["pl"], ["pl", "tensor"], ["pl", "tile"]) and parts[-1] in operands:
                index, keywords = operands[parts[-1]]
                operand = (
                    expr.args[index]
                    if len(expr.args) > index
                    else next((kw.value for kw in expr.keywords if kw.arg in keywords), None)
                )
                source = self.alias_of(operand) if operand is not None else None
                if source is None:
                    return None
                root = _root(source)
                assert root is not None
                if parts[-1] == "slice":
                    return _View(root, source, _slice_interval(expr))
                if parts[-1] == "reshape":
                    return _View(root, source)
                return source
        return None

    def _assignment_roots(self, value: ast.expr | None) -> _AliasRoots:
        if isinstance(value, ast.Tuple):
            return tuple(self._assignment_roots(element) for element in value.elts)
        if isinstance(value, ast.List):
            return [self._assignment_roots(element) for element in value.elts]
        if isinstance(value, ast.Call) and ast.unparse(value.func) in {
            "pl.create_tensor",
            "pl.tensor.create_tensor",
        }:
            return self.fresh()
        return self._roots_of(value) if value is not None else None

    def _bind_target(self, target: ast.expr, roots: _AliasRoots) -> None:
        if isinstance(target, ast.Name):
            # Tuple snapshots are immutable. List roots are used only for
            # immediate unpacking: a named list could be mutated through an alias.
            self.bind(target.id, None if isinstance(roots, list) else roots)
        elif isinstance(target, (ast.Tuple, ast.List)):
            elements = roots if isinstance(roots, (tuple, list)) else []
            for index, element in enumerate(target.elts):
                self._bind_target(element, elements[index] if index < len(elements) else None)

    def record_assignment(self, statement: ast.stmt, roots: Sequence[_AliasRoots] | None = None) -> None:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            return
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        # Evaluate every RHS root before rebinding any LHS, including swaps
        # and chained assignments. Aggregate values are not tensor aliases.
        value_roots: _AliasRoots = self._assignment_roots(statement.value) if roots is None else list(roots)
        for target in targets:
            target_roots = value_roots
            if roots is not None and isinstance(target, ast.Name):
                target_roots = roots[0] if len(roots) == 1 else tuple(roots)
            self._bind_target(target, target_roots)

    def check_aliases(self, call: ast.Call, binding: "CallBinding") -> None:
        if binding.alias_permissions is None:
            return
        groups: dict[str, list[int]] = {}
        aliases = [self.alias_of(argument) for argument in call.args]
        for index, argument in enumerate(call.args):
            root = _root(aliases[index])
            if root is None and isinstance(argument, ast.Name):
                root = f"unknown:{argument.id}"  # Same expression still proves overlap.
            if root is not None:
                groups.setdefault(root, []).append(index)
        for group in groups.values():
            for first in group:
                if binding.plan.parameter_directions[first] == ParamDirection.In:
                    continue
                for second in group:
                    pair = (min(first, second), max(first, second))
                    if (
                        first != second
                        and pair not in binding.alias_permissions
                        and not _disjoint(aliases[first], aliases[second])
                    ):
                        raise ValueError(
                            f"External call {ast.unparse(call.func)!r} has potentially overlapping "
                            f"arguments {pair[0]} and {pair[1]} without an AliasPermission"
                        )
