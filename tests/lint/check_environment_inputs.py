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


def _qualified(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return f"{_qualified(node.value, aliases)}.{node.attr}"
    return ""


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


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"os.{alias.name}"
    return aliases


def _bindings(tree: ast.Module) -> tuple[dict[str, str], dict[str, str]]:
    aliases = _import_aliases(tree)
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    value = _string(node.value, constants)
                    if value is not None:
                        constants[target.id] = value
                    else:
                        constants.pop(target.id, None)
                    qualified = _qualified(node.value, aliases) if node.value is not None else ""
                    if qualified.startswith("os."):
                        aliases[target.id] = qualified
    # Local aliases need recognition too; otherwise assigning env=os.environ
    # inside a function would hide every later env.get() from the inventory.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            qualified = _qualified(node.value, aliases) if node.value is not None else ""
            if qualified.startswith("os."):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        aliases[target.id] = qualified
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            for name in node.names:
                constants.pop(name, None)
    return aliases, constants


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
    aliases, constants = _bindings(tree)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    reads = []
    for node in ast.walk(tree):
        argument: ast.AST | None = None
        matched = False
        if isinstance(node, ast.Call):
            name = _qualified(node.func, aliases)
            if name in {
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
            elif name.startswith(("os.environ.", "os.environb.")):
                matched = True
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            if _qualified(node.value, aliases) in {"os.environ", "os.environb"}:
                argument = node.slice
                matched = True
        elif _qualified(node, aliases) in {"os.environ", "os.environb"}:
            parent = parents.get(node)
            # Accesses handled above. Storing the mapping in an alias is also
            # recognized; passing/iterating/copying it is a dynamic bulk read.
            matched = not isinstance(parent, (ast.Attribute, ast.Subscript, ast.Assign, ast.AnnAssign))
        if matched and isinstance(node, ast.expr):
            current = parents.get(node)
            function = "<module>"
            effective_constants = constants
            while current is not None:
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if function == "<module>":
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
