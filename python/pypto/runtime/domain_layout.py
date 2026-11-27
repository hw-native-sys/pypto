# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Immutable physical domain plans resolved without initializing the runtime."""

import ast
import hashlib
import operator
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class DomainBufferLayout:
    name: str
    logical_nbytes: int
    nbytes: int
    byte_offset: int


@dataclass(frozen=True)
class DomainLayout:
    """Ordered members and named slots; the internal order slot is runtime-owned."""

    name: str
    workers: tuple[int, ...]
    window_size: int
    buffers: tuple[DomainBufferLayout, ...]
    memory_source: str = "device"
    workspace_engines: tuple[str, ...] = ()


def _evaluate(expression: str, bindings: dict[str, int]) -> Any:
    binary: dict[type[ast.operator], Callable[[int, int], int]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: lambda left, right: left // right,
        ast.Mod: operator.mod,
    }

    def visit(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and node.id in bindings and type(bindings[node.id]) is int:
            return bindings[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = visit(node.left), visit(node.right)
            if type(left) is not int or type(right) is not int:
                raise ValueError("domain size arithmetic requires integer operands")
            value = binary[type(node.op)](left, right)
            if abs(value) >= 1 << 63:
                raise ValueError("domain layout arithmetic exceeds int64")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, (ast.Tuple, ast.List)):
            values: list[Any] = []
            for item in node.elts:
                values.extend(visit(item.value)) if isinstance(item, ast.Starred) else values.append(
                    visit(item)
                )
            return tuple(values)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [visit(arg) for arg in node.args]
            if node.func.id == "range":
                result = range(*args)
                if len(result) > 64:
                    raise ValueError("domain layouts support at most 64 ranks")
                return tuple(result)
            if node.func.id in ("tuple", "list") and len(args) == 1:
                return tuple(args[0])
            if node.func.id in ("min", "max") and args:
                return (min if node.func.id == "min" else max)(args)
        raise ValueError(f"domain layout expression requires explicit scalar bindings: {expression!r}")

    if len(expression) > 16384:
        raise ValueError("domain layout expression is too large")
    return visit(ast.parse(expression, mode="eval").body)


def read_layout_specs(path: Path) -> tuple[dict[str, Any], ...]:
    """Read literal compiler metadata without importing generated Python code."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    specs: list[dict[str, Any]] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "_pypto_domain_layout_specs"
            for target in statement.targets
        ):
            specs.extend(ast.literal_eval(statement.value))
    unique = {spec["name"]: spec for spec in specs}
    if len(unique) != len(specs):
        raise ValueError("duplicate generated domain layout names")
    return tuple(specs)


def resolve_layouts(specs: Any, world_size: int, scalars: dict[str, int]) -> tuple[DomainLayout, ...]:
    if type(world_size) is not int or not 0 < world_size <= 64:
        raise ValueError(f"world_size must be in [1, 64], got {world_size!r}")
    bindings = {**scalars, "world_size": world_size}
    layouts: list[DomainLayout] = []
    for spec in specs:
        workers = tuple(_evaluate(spec["workers"], bindings))
        if (
            not workers
            or len(set(workers)) != len(workers)
            or any(type(rank) is not int or not 0 <= rank < world_size for rank in workers)
        ):
            raise ValueError(f"invalid domain members: {workers}")
        buffers: list[DomainBufferLayout] = []
        offset = 0
        for name, logical_expr, size_expr in spec["buffers"]:
            logical, size = _evaluate(logical_expr, bindings), _evaluate(size_expr, bindings)
            if type(logical) is not int or type(size) is not int or not 0 < logical <= size or size % 64:
                raise ValueError(f"invalid domain buffer size for {name!r}: {logical}, {size}")
            buffers.append(DomainBufferLayout(name, logical, size, offset))
            offset += size
        window = _evaluate(spec["window_size"], bindings)
        if type(window) is not int or window != offset + 64:
            raise ValueError(f"domain window must include exactly one reserved order slot: {spec['name']}")
        if len({buffer.name for buffer in buffers}) != len(buffers):
            raise ValueError("domain buffer names must be distinct")
        layouts.append(
            DomainLayout(
                spec["name"],
                workers,
                window - 64,
                tuple(buffers),
                spec["memory_source"],
                tuple(spec["engines"]),
            )
        )
    return tuple(layouts)


def artifact_files(root: Path) -> dict[str, str]:
    """Fingerprint generated source/config dependencies, excluding mutable diagnostics."""
    paths: list[Path] = []
    for directory in (root / "orchestration", root / "next_levels"):
        if directory.exists():
            paths.extend(path for path in directory.rglob("*") if path.suffix in (".py", ".cpp", ".pto"))
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def read_comm_abi(path: Path) -> tuple[int, ...] | None:
    values: list[tuple[int, ...]] = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "_pypto_comm_abi" for target in node.targets
        ):
            values.append(tuple(ast.literal_eval(node.value)))
    if not values:
        return None
    if any(value != values[0] for value in values):
        raise ValueError("generated entries disagree on CommContext ABI")
    return values[0]


def validate_comm_abi(expected: Any, actual: Any) -> None:
    if (
        not isinstance(expected, (tuple, list))
        or len(cast(Any, expected)) != 4
        or any(type(value) is not int or value < 0 for value in cast(Any, expected))
    ):
        raise ValueError("artifact CommContext ABI must contain four nonnegative integers")
    if tuple(cast(Any, expected)) != tuple(actual):
        raise RuntimeError(
            f"CommContext ABI differs: artifact={tuple(cast(Any, expected))}, "
            f"runtime={tuple(actual)}; rebuild together"
        )


def validate_artifact_files(root: Path, manifest: Any) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("artifact file manifest must be a mapping")
    for name, digest in cast(dict[Any, Any], manifest).items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("artifact manifest requires relative paths and SHA256 digests")
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"artifact dependency is missing or outside its root: {name!r}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"artifact dependency changed; recompile before execution: {name}")
