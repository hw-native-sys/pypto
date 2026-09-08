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

import argparse
import ast
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

from _cpp_text import strip_cpp_comments

_ROOT = Path(__file__).resolve().parents[2]
_CATEGORIES = {"semantic", "tool_resolution", "fresh_request", "nonsemantic"}

# Scan root -> the file suffixes audited under it. python/pypto holds the Python
# sources; src, include and python/bindings hold the C++ ones. python/bindings is a
# sibling of python/pypto, so the two never overlap.
SCAN_ROOTS: dict[str, frozenset[str]] = {
    "python/pypto": frozenset({".py"}),
    "src": frozenset({".cpp", ".h"}),
    "include": frozenset({".cpp", ".h"}),
    "python/bindings": frozenset({".cpp", ".h"}),
}


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


def all_sources(root: Path) -> list[Path]:
    """Every production source the registry governs.

    The scan roots carry different suffixes -- ``python/pypto`` holds the Python sources,
    the other three the C++ ones -- so :data:`SCAN_ROOTS` maps each root to its own set
    rather than applying one filter to all four. Enumeration is by ``rglob`` rather than
    ``git ls-files`` so that a source not yet added to the index is still audited.
    """
    files: list[Path] = []
    for directory, suffixes in SCAN_ROOTS.items():
        files.extend(sorted(p for p in (root / directory).rglob("*") if p.suffix in suffixes))
    return files


def selected_sources(root: Path, selected: Iterable[Path]) -> list[Path]:
    """The subset of *selected* that :func:`all_sources` would have visited.

    pre-commit hands this script the files in the commit; anything outside the scan roots,
    or carrying a suffix that root does not scan, must be dropped rather than audited on
    different terms from a whole-tree run.
    """
    keep = []
    for raw in selected:
        path = raw if raw.is_absolute() else root / raw
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        for directory, suffixes in SCAN_ROOTS.items():
            if relative.startswith(f"{directory}/") and path.suffix in suffixes:
                keep.append(path)
                break
    return sorted(set(keep))


def _reads(path: Path) -> list[EnvironmentRead]:
    """Every environment read in *path*, parsed per its language."""
    text = path.read_text()
    return python_reads(text) if path.suffix == ".py" else cpp_reads(text)


def _exception_set(registry: dict[str, Any]) -> set[tuple[str, str]]:
    return {(item["path"], item["function"]) for item in registry.get("dynamic_reads", [])}


def _unused_report(exceptions: set[tuple[str, str]], used: set[tuple[str, str]]) -> list[str]:
    return [
        f"Unused dynamic-read exception: {path}:{function}" for path, function in sorted(exceptions - used)
    ]


def unused_exceptions(root: Path, registry: dict[str, Any]) -> list[str]:
    """Report dynamic-read exceptions that no source uses.

    ``check_registry`` rejects a wildcard or absolute exception path, so an exception names
    one exact file and only *that* file can mark it used. Reading just those files answers
    the question exactly as a whole-tree sweep would, without re-reading the ~526 sources
    the file-scoped audit has already read.

    ``selected_sources`` applies the same root and suffix filter :func:`all_sources` does,
    so an exception naming a path outside the audited scope stays unused here, as before.
    """
    check_registry(registry)
    exceptions = _exception_set(registry)
    used = set()
    for path in selected_sources(root, [Path(relative) for relative, _ in exceptions]):
        relative = path.relative_to(root).as_posix()
        for read in _reads(path):
            if read.variable is None and (relative, read.function) in exceptions:
                used.add((relative, read.function))
    return _unused_report(exceptions, used)


def check_tree(root: Path, registry: dict[str, Any], files: list[Path] | None = None) -> list[str]:
    """Report every unclassified read in PyPTO Python/C++ production sources.

    Args:
        root: Repository root.
        registry: Parsed ``_environment.json``.
        files: Sources to audit. ``None`` means the whole tree, which additionally reports
            unused dynamic-read exceptions -- free there, since the sweep already visits
            every file that could mark one used. A file-scoped run cannot answer that
            question; :func:`unused_exceptions` answers it on its own.
    """
    check_registry(registry)
    exceptions = _exception_set(registry)
    whole_tree = files is None
    used = set()
    errors = []
    for path in all_sources(root) if whole_tree else files or []:
        relative = path.relative_to(root).as_posix()
        for read in _reads(path):
            if read.variable is None and (relative, read.function) in exceptions:
                used.add((relative, read.function))
            elif read.variable not in registry["variables"]:
                name = read.variable or f"dynamic read in {read.function}"
                errors.append(f"{relative}:{read.line}: unclassified environment input: {name}")
    if whole_tree:
        errors.extend(_unused_report(exceptions, used))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that environment reads are registered.")
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Sources to audit (default: the whole tree, which also reports unused exceptions)",
    )
    parser.add_argument(
        "--unused-exceptions-only",
        action="store_true",
        help="Only report unused dynamic-read exceptions, reading just the files they name",
    )
    args = parser.parse_args()
    if args.unused_exceptions_only and args.files:
        parser.error("--unused-exceptions-only takes no file arguments")

    registry = json.loads((_ROOT / "python/pypto/_environment.json").read_text())
    if args.unused_exceptions_only:
        errors = unused_exceptions(_ROOT, registry)
    else:
        files = selected_sources(_ROOT, args.files) if args.files else None
        errors = check_tree(_ROOT, registry, files)
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
