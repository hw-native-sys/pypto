# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Infer full physical GM-buffer spans from PTOAS-generated C++."""

import ast
import re
from dataclasses import asdict, dataclass

_CAST_RE = re.compile(r"\(\s*(?:u?int(?:8|16|32|64)_t|size_t|long|unsigned\s+long)\s*\)")
_CONST_RE = re.compile(r"\bconst\s+(?:u?int(?:32|64)_t|size_t)\s+([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*;")
_ASSIGN_RE = re.compile(r"\b(?:u?int(?:32|64)_t|size_t)\s+([A-Za-z_]\w*)\s*=\s*([^;]+);")
_FOR_RE = re.compile(
    r"for\s*\(\s*(?:u?int(?:32|64)_t|size_t)\s+([A-Za-z_]\w*)\s*=\s*([^;]+);"
    r"\s*\1\s*<\s*([^;]+);\s*\1\s*\+=\s*([^)]+)\)"
)
_VIEW_RE = re.compile(
    r"GlobalTensor<\s*(?P<ctype>[\w:]+)\s*,\s*pto::Shape<(?P<shape>[^>]*)>\s*,\s*"
    r"pto::Stride<(?P<stride>[^>]*)>[^>]*>\s+[A-Za-z_]\w*\s*=\s*"
    r"GlobalTensor<[^(]*\(\s*(?P<base>.*?)\s*,",
    re.S,
)


@dataclass(frozen=True)
class PointerExtent:
    """Maximum physical element touched through one ABI pointer."""

    required_elements: int
    max_base_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    view_count: int


@dataclass(frozen=True)
class ExtentAnalysis:
    """Resolved pointer extents and expressions that could not be bounded."""

    pointers: dict[str, PointerExtent]
    unresolved: tuple[str, ...]

    def manifest(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "pointers": {name: asdict(record) for name, record in sorted(self.pointers.items())},
            "unresolved": list(self.unresolved),
        }


def _clean_expression(expression: str) -> str:
    return _CAST_RE.sub("", expression).strip()


Bound = tuple[int, int]


def _bounds(expression: str, values: dict[str, Bound]) -> Bound:
    tree = ast.parse(_clean_expression(expression), mode="eval")

    def visit(node: ast.AST) -> Bound:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value, node.value
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise KeyError(node.id)
            return values[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            lower, upper = visit(node.operand)
            return (lower, upper) if isinstance(node.op, ast.UAdd) else (-upper, -lower)
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left[0] + right[0], left[1] + right[1]
            if isinstance(node.op, ast.Sub):
                return left[0] - right[1], left[1] - right[0]
            if isinstance(node.op, ast.Mult):
                products = (
                    left[0] * right[0],
                    left[0] * right[1],
                    left[1] * right[0],
                    left[1] * right[1],
                )
                return min(products), max(products)
            if isinstance(node.op, (ast.FloorDiv, ast.Div)):
                if right[0] <= 0:
                    raise ValueError(f"non-positive divisor in {expression!r}")
                quotients = (
                    left[0] // right[0],
                    left[0] // right[1],
                    left[1] // right[0],
                    left[1] // right[1],
                )
                return min(quotients), max(quotients)
            if isinstance(node.op, (ast.LShift, ast.RShift)):
                if right[0] < 0:
                    raise ValueError(f"negative shift in {expression!r}")
                operation = int.__lshift__ if isinstance(node.op, ast.LShift) else int.__rshift__
                shifted = (
                    operation(left[0], right[0]),
                    operation(left[0], right[1]),
                    operation(left[1], right[0]),
                    operation(left[1], right[1]),
                )
                return min(shifted), max(shifted)
            raise ValueError(f"unsupported operator in {expression!r}")
        raise ValueError(f"unsupported expression in {expression!r}")

    return visit(tree)


def _assignments(cpp_text: str) -> list[tuple[str, str]]:
    assignments = []
    for match in _ASSIGN_RE.finditer(cpp_text):
        line_start = cpp_text.rfind("\n", 0, match.start()) + 1
        if "for" in cpp_text[line_start : match.start()]:
            continue
        assignments.append((match.group(1), match.group(2)))
    return assignments


def _bound_environment(
    cpp_text: str,
    scalar_values: dict[str, int],
    spmd_block_index: str | None,
) -> dict[str, Bound]:
    values = {name: (value, value) for name, value in scalar_values.items()}
    values.update({name: (int(value), int(value)) for name, value in _CONST_RE.findall(cpp_text)})
    loops = _FOR_RE.findall(cpp_text)
    assignments = _assignments(cpp_text)

    for _ in range(len(loops) + len(assignments) + 2):
        changed = False
        for variable, start_expression, stop_expression, step_expression in loops:
            try:
                start = _bounds(start_expression, values)
                stop = _bounds(stop_expression, values)
                step = _bounds(step_expression, values)
            except (KeyError, SyntaxError, ValueError):
                continue
            if step[0] <= 0:
                raise ValueError(f"non-positive or variable-sign loop step: {step_expression!r}")
            if spmd_block_index and re.search(rf"\b{re.escape(spmd_block_index)}\b", start_expression):
                bound = min(start[0], 0), max(stop[1] - 1, start[1])
            elif stop[1] <= start[0]:
                bound = start
            else:
                bound = start[0], max(start[1], stop[1] - 1)
            if values.get(variable) != bound:
                values[variable] = bound
                changed = True

        for variable, expression in assignments:
            try:
                value = _bounds(expression, values)
            except (KeyError, SyntaxError, ValueError):
                continue
            if values.get(variable) != value:
                values[variable] = value
                changed = True
        if not changed:
            break
    return values


def _parse_shape(raw: str) -> tuple[int, ...]:
    dimensions = tuple(int(value.strip()) for value in raw.split(","))
    if not dimensions or any(value <= 0 for value in dimensions):
        raise ValueError(f"invalid GlobalTensor shape: {raw!r}")
    return dimensions


def _parse_stride(raw: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = tuple(int(value.strip()) for value in raw.split(","))
    if len(stride) != len(shape):
        raise ValueError(f"GlobalTensor shape/stride rank mismatch: {shape} vs {stride}")
    invalid = [(dimension, step) for dimension, step in zip(shape, stride) if dimension > 1 and step <= 0]
    if invalid:
        raise ValueError(f"non-positive GlobalTensor stride for non-singleton dimensions: {invalid}")
    return stride


def analyze_pointer_extents(
    cpp_text: str,
    pointer_names: set[str],
    scalar_values: dict[str, int],
    *,
    spmd_block_index: str | None = None,
) -> ExtentAnalysis:
    """Bound every generated GlobalTensor view over the full iteration space.

    Args:
        cpp_text: PTOAS-generated C++ containing concrete GlobalTensor views.
        pointer_names: External ABI pointer parameter names.
        scalar_values: Exact integer values for external scalar parameters.
        spmd_block_index: ABI parameter representing the SPMD block index.

    Returns:
        Per-pointer physical spans and any unresolved base expressions.
    """
    values = _bound_environment(cpp_text, scalar_values, spmd_block_index)
    records: dict[str, PointerExtent] = {}
    unresolved: list[str] = []

    for match in _VIEW_RE.finditer(cpp_text):
        shape = _parse_shape(match.group("shape"))
        stride = _parse_stride(match.group("stride"), shape)
        base = _clean_expression(match.group("base"))
        pointer_match = re.match(r"([A-Za-z_]\w*)\s*(?:\+\s*(.*))?$", base, re.S)
        if pointer_match is None or pointer_match.group(1) not in pointer_names:
            referenced = sorted(
                pointer for pointer in pointer_names if re.search(rf"\b{re.escape(pointer)}\b", base)
            )
            if referenced:
                unresolved.append(f"{'/'.join(referenced)}: unsupported base expression {base!r}")
            continue
        pointer = pointer_match.group(1)
        offset_expression = pointer_match.group(2) or "0"
        try:
            minimum_offset, base_offset = _bounds(offset_expression, values)
        except (KeyError, SyntaxError, ValueError) as error:
            unresolved.append(f"{pointer}: {offset_expression} ({error})")
            continue
        if minimum_offset < 0:
            raise ValueError(f"{pointer} may have a negative base offset: {minimum_offset}")
        span = 1 + sum((dimension - 1) * step for dimension, step in zip(shape, stride))
        required = base_offset + span
        previous = records.get(pointer)
        view_count = 1 if previous is None else previous.view_count + 1
        if previous is None or required > previous.required_elements:
            records[pointer] = PointerExtent(required, base_offset, shape, stride, view_count)
        else:
            records[pointer] = PointerExtent(
                previous.required_elements,
                previous.max_base_offset,
                previous.shape,
                previous.stride,
                view_count,
            )

    return ExtentAnalysis(records, tuple(unresolved))
