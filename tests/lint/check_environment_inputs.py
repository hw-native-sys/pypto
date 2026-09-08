# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Require an identity classification for Python/C++ environment reads.

Runs without importing PyPTO or its native extension. Python imports, aliases,
module string constants, getenv, and environ access are recognized. Unresolved
names and bulk reads require a reason attached to the exact file and function.
C++ getenv/secure_getenv arguments must be string literals. This inventories
PyPTO consumers; implicit tool inputs in the registry need separate toolchain
dependency discovery and are not certified by this source scan.
"""

import ast
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

from _cpp_text import strip_cpp_comments

_ROOT = Path(__file__).resolve().parents[2]
_CATEGORIES = {"semantic", "tool_resolution", "fresh_request", "nonsemantic"}


class EnvironmentRead(NamedTuple):
    variable: str | None
    line: int
    function: str


def _qualified(node: ast.AST, aliases: dict[str, set[str]]) -> set[str]:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, set())
    if isinstance(node, ast.Attribute):
        return {f"{name}.{node.attr}" for name in _qualified(node.value, aliases)}
    return set()


def _string(node: ast.AST | None, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Constant) and isinstance(node.value, bytes):
        try:
            return node.value.decode("ascii")
        except UnicodeDecodeError:
            return None
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


_SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _scope_nodes(scope: ast.AST) -> list[ast.AST]:
    """Collect one lexical scope, including control flow but not nested bodies."""
    nodes = []
    pending = list(reversed(list(ast.iter_child_nodes(scope))))
    while pending:
        node = pending.pop()
        nodes.append(node)
        if not isinstance(node, _SCOPES):
            pending.extend(reversed(list(ast.iter_child_nodes(node))))
    return nodes


def _bound_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return {node.id}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.arg):
        return {node.arg}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {alias.asname or alias.name.split(".")[0] for alias in node.names}
    if isinstance(node, ast.ExceptHandler) and node.name is not None:
        return {node.name}
    return set()


def _bindings(scope: ast.AST, inherited: dict[str, set[str]]) -> tuple[dict[str, set[str]], dict[str, str]]:
    nodes = _scope_nodes(scope)
    stores: dict[str, int] = {}
    for node in nodes:
        for name in _bound_names(node):
            stores[name] = stores.get(name, 0) + 1
    aliases = {name: values.copy() for name, values in inherited.items() if name not in stores}
    # Keep every possible imported/environment alias in this scope. A later
    # assignment must not erase an earlier read, or a conditional read path.
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                aliases.setdefault(name, set()).add(alias.name if alias.asname else name)
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                aliases.setdefault(alias.asname or alias.name, set()).add(f"os.{alias.name}")
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            qualified = {name for name in _qualified(node.value, aliases) if name.startswith("os.")}
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and qualified:
                    aliases.setdefault(target.id, set()).update(qualified)
    return aliases, _scope_constants(scope, stores)


def _scope_constants(scope: ast.AST, stores: dict[str, int]) -> dict[str, str]:
    """Accept module constants only when no other write can change the name."""
    constants: dict[str, str] = {}
    body = scope.body if isinstance(scope, (ast.Module, ast.ClassDef)) else []
    for node in body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and stores.get(target.id) == 1:
                    value = _string(node.value, constants)
                    if value is not None:
                        constants[target.id] = value
    for node in ast.walk(scope):
        if isinstance(node, ast.Global):
            for name in node.names:
                constants.pop(name, None)
    return constants


def _function_constants(node: ast.AST, constants: dict[str, str]) -> dict[str, str]:
    """Do not mistake a shadowed module constant for a static environment name."""
    shadowed = {
        item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
    }
    shadowed.update(item.arg for item in ast.walk(node) if isinstance(item, ast.arg))
    return {key: value for key, value in constants.items() if key not in shadowed}


def python_reads(source: str) -> list[EnvironmentRead]:
    """Find static and unresolved environment reads in Python source."""
    tree = ast.parse(source)
    aliases, constants = _bindings(tree, {})
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    scope_aliases: dict[ast.AST, dict[str, set[str]]] = {tree: aliases}

    def aliases_for(scope: ast.AST) -> dict[str, set[str]]:
        if scope not in scope_aliases:
            parent = parents[scope]
            while not isinstance(parent, _SCOPES):
                parent = parents[parent]
            # Method bodies use enclosing function/module globals, not class
            # attributes; a class body itself still sees its own assignments.
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                while isinstance(parent, ast.ClassDef):
                    parent = parents[parent]
                    while not isinstance(parent, _SCOPES):
                        parent = parents[parent]
            scope_aliases[scope] = _bindings(scope, aliases_for(parent))[0]
        return scope_aliases[scope]

    reads = []
    for node in ast.walk(tree):
        scope = parents.get(node, tree)
        while not isinstance(scope, _SCOPES):
            scope = parents[scope]
        aliases = aliases_for(scope)
        argument: ast.AST | None = None
        matched = False
        if isinstance(node, ast.Call):
            names = _qualified(node.func, aliases)
            if names & {
                "os.getenv",
                "os.getenvb",
                "os.environ.get",
                "os.environ.pop",
                "os.environ.setdefault",
                "os.environb.get",
                "os.environb.pop",
                "os.environb.setdefault",
            }:
                argument = (
                    node.args[0]
                    if node.args
                    else next((keyword.value for keyword in node.keywords if keyword.arg == "key"), None)
                )
                matched = True
            elif any(name.startswith(("os.environ.", "os.environb.")) for name in names):
                matched = True
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            if _qualified(node.value, aliases) & {"os.environ", "os.environb"}:
                argument = node.slice
                matched = True
        elif _qualified(node, aliases) & {"os.environ", "os.environb"}:
            parent = parents.get(node)
            # Accesses handled above. Storing the mapping in an alias is also
            # recognized; passing/iterating/copying it is a dynamic bulk read.
            matched = not isinstance(parent, (ast.Attribute, ast.Subscript, ast.Assign, ast.AnnAssign))
        if matched and isinstance(node, ast.expr):
            current = parents.get(node)
            function = "<module>"
            effective_constants = constants
            while current is not None:
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    if function == "<module>" and isinstance(
                        current, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        function = current.name
                    effective_constants = _function_constants(current, effective_constants)
                current = parents.get(current)
            reads.append(EnvironmentRead(_string(argument, effective_constants), node.lineno, function))
    return reads


def cpp_reads(source: str) -> list[EnvironmentRead]:
    """Find getenv calls outside C++ comments and string literals."""
    source = strip_cpp_comments(source)
    string = r'"(?:\\.|[^"\\])*"'
    pattern = re.compile(rf"{string}|\b(?:getenv|secure_getenv)\s*\(")
    argument_pattern = re.compile(rf"\s*((?:{string}\s*)+)\)")
    reads = []
    for match in pattern.finditer(source):
        if match.group().startswith('"'):
            continue
        argument = argument_pattern.match(source, match.end())
        variable = None
        if argument is not None:
            variable = "".join(ast.literal_eval(token) for token in re.findall(string, argument.group(1)))
        reads.append(EnvironmentRead(variable, source.count("\n", 0, match.start()) + 1, "<cpp>"))
    return reads


def check_registry(registry: dict[str, Any]) -> None:
    """Reject unreviewable classifications and broad dynamic-read exceptions."""
    if registry.get("schema") != 1 or not isinstance(registry.get("variables"), dict):
        raise ValueError("Environment registry must have schema=1 and a variables mapping")
    for name, rule in registry["variables"].items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise ValueError(f"Invalid environment variable name: {name}")
        if rule.get("category") not in _CATEGORIES or not rule.get("reason", "").strip():
            raise ValueError(f"Environment classification needs a known category and reason: {name}")
    for rule in registry.get("dynamic_reads", []):
        if not all(
            isinstance(rule.get(key), str) and rule[key].strip() for key in ("path", "function", "reason")
        ):
            raise ValueError("Dynamic-read exceptions require path, function, and reason")
        if rule["path"].startswith("/") or "*" in rule["path"] or "*" in rule["function"]:
            raise ValueError("Dynamic-read exceptions must name an exact relative path and function")


def check_tree(root: Path, registry: dict[str, Any]) -> list[str]:
    """Report every unclassified read in PyPTO Python/C++ production sources."""
    check_registry(registry)
    exceptions = {(item["path"], item["function"]) for item in registry.get("dynamic_reads", [])}
    used = set()
    errors = []
    files = sorted((root / "python" / "pypto").rglob("*.py"))
    for directory in ("src", "include", "python/bindings"):
        files.extend(sorted(path for path in (root / directory).rglob("*") if path.suffix in {".cpp", ".h"}))
    for path in files:
        relative = path.relative_to(root).as_posix()
        reads = python_reads(path.read_text()) if path.suffix == ".py" else cpp_reads(path.read_text())
        for read in reads:
            if read.variable is None and (relative, read.function) in exceptions:
                used.add((relative, read.function))
            elif read.variable not in registry["variables"]:
                name = read.variable or f"dynamic read in {read.function}"
                errors.append(f"{relative}:{read.line}: unclassified environment input: {name}")
    errors.extend(
        f"Unused dynamic-read exception: {path}:{function}" for path, function in sorted(exceptions - used)
    )
    return errors


def main() -> int:
    registry = json.loads((_ROOT / "python/pypto/_environment.json").read_text())
    errors = check_tree(_ROOT, registry)
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
