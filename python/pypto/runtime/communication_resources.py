# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit communication resource contracts and static domain provisioning.

The producer promises completeness; fingerprints bind that promise to code.
No source identity or fresh-IR marker is itself a completeness proof.
"""

import ast
import json
import operator
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pypto._identity import ContentRoot, fingerprint_content

RESOURCE_CONTRACT_FILE = "communication_resources.json"
# Optional cache policy, not an estimate of hardware capacity or an optimum:
# the absolute ceilings bound eager duplication on large hosts, while the 1/64
# share keeps at least 63/64 of observed free memory outside this optional cache.
# Device windows receive a larger ceiling than Host descriptors because their
# payloads are larger. Measurements are snapshots, not reservations; allocation
# errors still propagate. One necessary baseline set is charged separately.
# The 2 MiB/context allowance covers opaque allocation/alignment overhead when
# a free-memory delta undercounts it; it is not a proven SDK allocation bound.
HOST_FRAME_CACHE_LIMIT_BYTES = 64 * 1024 * 1024
# Shared Tensor storage retains file descriptors. Leave most currently free
# descriptors for Worker startup, input mappings and other process activity.
HOST_DESCRIPTOR_CACHE_DIVISOR = 4
DEVICE_DOMAIN_CACHE_LIMIT_BYTES = 128 * 1024 * 1024
FREE_MEMORY_CACHE_DIVISOR = 64
DOMAIN_CONTEXT_RESERVE_BYTES = 2 * 1024 * 1024
_MAX_STATIC_ITEMS = 4096
_BINARY_OPERATORS: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


@dataclass(frozen=True)
class DomainDeclaration:
    name: str
    workers: tuple[int, ...]
    window_size: int
    buffers: tuple[tuple[str, str, int, int], ...]

    def allocate(self, provider: Callable[..., Any]) -> None:
        from simpler.task_interface import CommBufferSpec  # noqa: PLC0415

        with provider(
            name=self.name,
            workers=list(self.workers),
            window_size=self.window_size,
            buffers=[
                CommBufferSpec(name=name, dtype=dtype, count=count, nbytes=nbytes)
                for name, dtype, count, nbytes in self.buffers
            ],
        ):
            pass


@dataclass(frozen=True)
class DomainPlan:
    declarations: tuple[DomainDeclaration, ...]

    def warm(self, provider: Callable[..., Any]) -> None:
        """Only allocate declared resources; never execute a Host graph to prewarm."""
        for declaration in self.declarations:
            declaration.allocate(provider)


def _static_value(node: ast.AST, world_size: int) -> Any:
    if isinstance(node, ast.Constant) and type(node.value) in (str, int):
        return node.value
    if isinstance(node, ast.Name) and node.id == "world_size":
        return world_size
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left, right = _static_value(node.left, world_size), _static_value(node.right, world_size)
        if type(left) is not int or type(right) is not int:
            raise ValueError("Domain arithmetic must use integers")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, (ast.List, ast.Tuple)):
        items: list[Any] = []
        for item in node.elts:
            if isinstance(item, ast.Starred):
                items.extend(_static_value(item.value, world_size))
            else:
                items.append(_static_value(item, world_size))
            if len(items) > _MAX_STATIC_ITEMS:
                raise ValueError("Static domain declaration exceeds the item bound")
        return items
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        args = [_static_value(arg, world_size) for arg in node.args]
        if node.func.id == "range" and all(type(arg) is int for arg in args):
            values = range(*args)
            if len(values) <= _MAX_STATIC_ITEMS:
                return list(values)
        if node.func.id == "list" and len(args) == 1 and isinstance(args[0], list):
            return cast("list[Any]", args[0])
    raise ValueError("Domain declaration is not a supported static expression")


def _buffer_declarations(node: ast.AST, world_size: int) -> list[tuple[str, str, int, int]]:
    if not isinstance(node, ast.List):
        raise ValueError("Domain buffers must be a static list")
    buffers: list[tuple[str, str, int, int]] = []
    for buffer in node.elts:
        if not isinstance(buffer, ast.Call) or not isinstance(buffer.func, ast.Name):
            raise ValueError("Unknown domain buffer factory")
        if buffer.func.id != "CommBufferSpec" or buffer.args:
            raise ValueError("Unknown domain buffer factory")
        parts = {item.arg: _static_value(item.value, world_size) for item in buffer.keywords}
        if set(parts) != {"name", "dtype", "count", "nbytes"}:
            raise ValueError("Unsupported domain buffer fields")
        if type(parts["name"]) is not str or type(parts["dtype"]) is not str:
            raise ValueError("Invalid domain buffer name or dtype")
        if any(type(parts[key]) is not int or parts[key] <= 0 for key in ("count", "nbytes")):
            raise ValueError("Invalid domain buffer size")
        buffers.append((parts["name"], parts["dtype"], parts["count"], parts["nbytes"]))
    return buffers


def _declaration(node: ast.With, world_size: int) -> tuple[str, DomainDeclaration]:
    if len(node.items) != 1 or not isinstance(node.items[0].optional_vars, ast.Name):
        raise ValueError("Domain declaration must bind one named domain")
    call = node.items[0].context_expr
    if not isinstance(call, ast.Call) or ast.unparse(call.func) != "_domain_provider or orch.allocate_domain":
        raise ValueError("Unknown communication domain factory")
    fields = {item.arg: item.value for item in call.keywords}
    required = {"name", "workers", "window_size", "buffers"}
    if call.args or not required <= fields.keys() or fields.keys() - required - {"memory_source"}:
        raise ValueError("Unsupported domain controls")
    if "memory_source" in fields and _static_value(fields["memory_source"], world_size) != "device":
        raise ValueError("Domain pooling requires Device windows")
    name = _static_value(fields["name"], world_size)
    members = _static_value(fields["workers"], world_size)
    window_size = _static_value(fields["window_size"], world_size)
    if type(name) is not str or not isinstance(members, list) or not members:
        raise ValueError("Invalid domain name or members")
    workers = cast("list[int]", members)
    if any(type(rank) is not int or not 0 <= rank < world_size for rank in workers):
        raise ValueError("Invalid domain rank")
    if len(set(workers)) != len(workers) or type(window_size) is not int or window_size <= 0:
        raise ValueError("Invalid domain size or duplicate rank")
    buffers = _buffer_declarations(fields["buffers"], world_size)
    if not buffers or len({buffer[0] for buffer in buffers}) != len(buffers):
        raise ValueError("Empty or duplicate domain buffer names")
    if sum(buffer[3] for buffer in buffers) > window_size:
        raise ValueError("Domain buffers exceed the window")
    return node.items[0].optional_vars.id, DomainDeclaration(
        name, tuple(workers), window_size, tuple(buffers)
    )


def _domain_plan(fn: ast.FunctionDef, world_size: int) -> DomainPlan:
    domains: dict[str, DomainDeclaration] = {}
    ranks: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, (ast.AsyncFunctionDef, ast.While, ast.Try, ast.Lambda)):
            raise ValueError("Unsupported Host control flow")
        if isinstance(node, ast.With):
            local, declaration = _declaration(node, world_size)
            if local in domains or any(d.name == declaration.name for d in domains.values()):
                raise ValueError("Duplicate domain declaration")
            domains[local] = declaration
        if isinstance(node, ast.For):
            if not isinstance(node.target, ast.Name) or not isinstance(node.iter, ast.Call):
                raise ValueError("Unsupported rank loop")
            if not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "range":
                raise ValueError("Unsupported rank loop")
            ranks.add(node.target.id)
    if not domains or not ranks:
        raise ValueError("No static rank/domain declarations")
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_scalar":
            continue
        if len(node.args) != 1 or node.keywords:
            raise ValueError("Unknown scalar argument")
        arg = node.args[0]
        rank = isinstance(arg, ast.Name) and arg.id in ranks
        context = (
            isinstance(arg, ast.Attribute)
            and arg.attr == "device_ctx"
            and isinstance(arg.value, ast.Subscript)
            and isinstance(arg.value.value, ast.Name)
            and arg.value.value.id in domains
            and isinstance(arg.value.slice, ast.Name)
            and arg.value.slice.id in ranks
        )
        if not (rank or context):
            raise ValueError("Unknown scalar resource provenance")
    return DomainPlan(tuple(domains.values()))


def _artifact_files(root: Path) -> list[Path]:
    files = [
        p
        for folder in (root / "orchestration", root / "next_levels")
        for p in folder.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix not in (".pyc", ".pyo")
        and p.name != RESOURCE_CONTRACT_FILE
    ]
    if (root / "distributed_meta.json").is_file():
        files.append(root / "distributed_meta.json")
    if not any(p.suffix == ".so" and "next_levels" in p.relative_to(root).parts for p in files):
        raise ValueError("A resource contract requires final native callable binaries")
    return sorted(files)


def _fingerprint(root: Path, dependencies: tuple[Path, ...]) -> str:
    result = fingerprint_content(tuple(ContentRoot(p) for p in (*_artifact_files(root), *dependencies)))
    if result.digest is None:
        raise ValueError(f"Cannot bind resource contract: {result.failure}")
    return result.digest


def write_resource_contract(
    output_dir: Path, *, complete_resources: bool, external_dependencies: tuple[Path, ...], world_size: int
) -> None:
    """Record a trusted producer's explicit complete-resource promise.

    Dependencies must cover all external application/helper code not already
    covered by orchestration/, next_levels/, or distributed_meta.json. Unknown
    dependency closure must not be declared complete. Platform runtime ABI and
    provenance remain independently validated. No opaque behavior is inferred.
    """
    if complete_resources is not True:
        raise ValueError("The complete-resource producer promise must be explicit")
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    root = output_dir.resolve(strict=True)
    tree = ast.parse((root / "orchestration" / "host_orch.py").read_bytes())
    candidates: list[ast.FunctionDef] = []
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef):
            try:
                _domain_plan(fn, world_size)
            except (ValueError, TypeError, ArithmeticError):
                continue
            candidates.append(fn)
    if len(candidates) != 1:
        raise ValueError("Expected one Host entry with supported static domain declarations")
    dependencies = tuple(p.resolve(strict=True) for p in external_dependencies)
    contract = {
        "schema": 1,
        "promise": "declared-device-domains-and-task-args",
        "entry": candidates[0].name,
        "world_size": world_size,
        "dependencies": [str(p) for p in dependencies],
        "digest": _fingerprint(root, dependencies),
    }
    target = root / "orchestration" / RESOURCE_CONTRACT_FILE
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=root, prefix=".communication-resources-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(contract, sort_keys=True) + "\n")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_domain_plan(entry: Any, path: Path, world_size: int) -> DomainPlan | None:
    """Unknown, changed, or unsupported contracts take the conservative path."""
    try:
        contract = json.loads((path.parent / RESOURCE_CONTRACT_FILE).read_text())
        if (
            contract["schema"] != 1
            or contract["promise"] != "declared-device-domains-and-task-args"
            or contract["entry"] != entry.__name__
            or contract["world_size"] != world_size
        ):
            return None
        dependencies = tuple(Path(p) for p in contract["dependencies"])
        if any(not p.is_absolute() for p in dependencies):
            return None
        if _fingerprint(path.parent.parent.resolve(), dependencies) != contract["digest"]:
            return None
        tree = ast.parse(path.read_bytes())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry.__name__)
        return _domain_plan(fn, world_size)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        StopIteration,
        SyntaxError,
        ArithmeticError,
    ):
        return None


def entry_has_control_effects(entry: Any, path: Path) -> bool:
    """Only recognized graph-building calls qualify for the control-free path."""
    pure_calls = {
        "range",
        "list",
        "tuple",
        "len",
        "int",
        "float",
        "min",
        "max",
        "zip",
        "enumerate",
        "TaskArgs",
        "CallConfig",
        "_submit_chip",
        "_submit_sub",
        "add_chip_tensor_arg",
        "make_tensor_arg",
        "check_window_capacity",
    }
    graph_methods = {"add_tensor", "add_scalar", "append", "tensor", "scope_begin", "scope_end"}
    try:
        tree = ast.parse(path.read_bytes())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry.__name__)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in pure_calls:
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr in graph_methods:
                continue
            return True
        return False
    except (OSError, AttributeError, StopIteration, SyntaxError):
        return True
