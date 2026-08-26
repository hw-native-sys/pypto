# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Script to check that reserved ``Function.attrs`` keys are declared once per layer and agree.

Each key is a contract between a writer pass and a reader that may sit on the other side of the
C++/Python boundary -- ``split_aiv`` is stamped by ``ScopeOutliner`` and read by ``MemoryReuse`` to
gate an Ascend910B in-place hazard guard; ``dual_aiv_dispatch`` is stamped in ``src/ir/transforms``
and read by both codegens and by the torch debug emitter. A key spelled as a bare literal at each of
those sites has no single point of rename: a writer renamed without its readers leaves every reader
silently reading an absent attr, which reads as "feature off" rather than as an error.

The keys are therefore declared once in C++ (``include/pypto/ir/function.h``, which also carries the
authoritative per-key lifecycle documentation) and once in Python (``python/pypto/_function_attrs``).
This check enforces three things:

1. *Parity* -- every key the Python module declares pairs with a C++ declaration holding the same
   string, so a key renamed on one side cannot drift from the other. The C++ header is the
   authoritative list and may declare keys Python has no use for (``spmd_unwrapped`` today); the
   reverse is an error, since a Python-only key names an attr no pass writes.
2. *Name agreement* -- ``kAttrDualAivDispatch`` in C++ pairs with ``DUAL_AIV_DISPATCH_ATTR`` in
   Python. The identifiers are mechanically derived from one another, so a mismatched pair is
   reported even when both sides happen to hold the same string.
3. *No redeclaration* -- no other source may spell one of these key strings as a bare literal at a
   site that provably reads or writes ``attrs``. That is the shape the declarations exist to replace.

Scope and deliberate limits:

* Only the keys declared in ``function.h`` are policed. ``Call`` / ``Submit`` attrs live in
  ``expr.h`` and ForStmt / pass-internal attrs in ``transforms/utils/attrs.h``; those are separate
  key spaces with their own owners, and folding them in here would make one failure message span
  three unrelated headers.
* Comments and docstrings are exempt -- prose naming a key is documentation, not a use. C++ comments
  are stripped before scanning; Python is parsed with ``ast`` so only real string literals are seen.
* Tests are exempt. A test that hand-builds ``attrs={"dual_aiv_dispatch": True}`` is asserting on the
  wire format itself, and pinning it to the constant would make the test pass vacuously after a
  rename -- exactly the failure it exists to catch.
* The C++ scan flags any bare occurrence; the Python scan flags only *attrs-anchored* shapes. Two
  keys are overloaded in Python and a blanket scan would be wrong about both: ``split_aiv`` also
  names the ``pl.split_aiv`` DSL construct (an ``__all__`` entry, a ``_VALID_ITERATORS`` member, a
  ``_loop_kind_stack`` sentinel), and ``auto_scope`` / ``external_source`` are also user-facing
  ``@pl.function(...)`` keyword parameters. Those are separate contracts that happen to share a
  spelling. The anchors below are the shapes that only an attr key takes:

  1. ``<expr naming attrs>.get("key", ...)`` -- ``dict(func.attrs).get(...)``,
     ``(func_attrs or {}).get(...)``, ``getattr(f, "attrs", {}).get(...)``.
  2. ``<expr naming attrs>["key"]`` -- subscript load or store.
  3. A dict-literal key in an assignment whose target names attrs --
     ``func_attrs = {**(func_attrs or {}), "auto_scope": False}``.
  4. A module-level collection whose name ends in ``_ATTRS`` -- ``_DECORATOR_ONLY_FUNC_ATTRS``.

  A literal reaching an attrs dict through a helper is not detected; that would need interprocedural
  analysis. Such sites are still worth converting by hand.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

from _cpp_text import strip_cpp_comments

# Declaration sites, relative to the repo root.
CPP_DECL = "include/pypto/ir/function.h"
PY_DECL = "python/pypto/_function_attrs.py"

# Directories scanned for bare literals, relative to the repo root.
SCAN_ROOTS = ("include", "src", "python")

# Paths exempt from the bare-literal scan: the two declaration sites, which necessarily spell each
# key once. ``.pyi`` stubs are outside the scan by construction -- only ``.py`` is parsed.
EXEMPT = frozenset({CPP_DECL, PY_DECL})

# ``inline constexpr const char* kAttrFoo = "foo";`` in function.h. The ``*`` binds loosely so the
# ``const char *kAttrFoo`` spelling matches too -- clang-format normalises it, but a declaration this
# pattern missed would be skipped in silence rather than reported, which is the one failure this
# lint must not have. ``CPP_DECL_SANITY_RE`` below closes the rest of that hole.
CPP_DECL_RE = re.compile(
    r'^\s*inline\s+constexpr\s+const\s+char\s*\*\s*(kAttr\w+)\s*=\s*"([^"]+)"\s*;', re.MULTILINE
)

# Anything that declares a ``kAttr...`` constant, however spelled. Every match must also be matched
# by ``CPP_DECL_RE``; one that is not means the strict pattern has drifted from the header and a key
# is going unpoliced, so it is reported rather than ignored.
CPP_DECL_SANITY_RE = re.compile(r"^.*\b(kAttr\w+)\s*=\s*\"", re.MULTILINE)

# ``FOO_ATTR = "foo"`` in _function_attrs.py.
PY_DECL_RE = re.compile(r'^([A-Z][A-Z0-9_]*_ATTR)\s*=\s*"([^"]+)"\s*$', re.MULTILINE)

# A name that provably holds a Function attrs mapping. Matched against every identifier reachable
# from a ``.get`` receiver / subscript value, so ``func.attrs``, ``func_attrs``, ``outlined_attrs``
# and ``getattr(f, "attrs", {})`` all qualify while ``group_meta`` (the torch emitter's own metadata
# dict, which mirrors the key names but is not an attrs mapping) does not.
ATTRS_NAME_RE = re.compile(r"(^|_)attrs$")

# Module-level collections that hold attr keys, matched on the bound name.
ATTRS_COLLECTION_RE = re.compile(r"_ATTRS$")


def cpp_ident_to_py(ident: str) -> str:
    """``kAttrDualAivDispatch`` -> ``DUAL_AIV_DISPATCH_ATTR``."""
    body = ident[len("kAttr") :]
    return re.sub(r"(?<!^)(?=[A-Z])", "_", body).upper() + "_ATTR"


def parse_declarations(root: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Return (cpp ident->key, python ident->key, errors)."""
    errors: list[str] = []
    cpp_text = strip_cpp_comments((root / CPP_DECL).read_text(encoding="utf-8"))
    cpp = {m.group(1): m.group(2) for m in CPP_DECL_RE.finditer(cpp_text)}
    py_text = (root / PY_DECL).read_text(encoding="utf-8")
    py = {m.group(1): m.group(2) for m in PY_DECL_RE.finditer(py_text)}
    if not cpp:
        errors.append(f'{CPP_DECL}: no `inline constexpr const char* kAttr... = "...";` declarations found')
    # A key the strict pattern missed would be skipped in silence -- ``if not cpp`` cannot fire while
    # the other keys still match. Report the mismatch instead of parsing a partial list.
    for m in CPP_DECL_SANITY_RE.finditer(cpp_text):
        if m.group(1) not in cpp:
            errors.append(
                f"{CPP_DECL}: `{m.group(1)}` is declared in a form CPP_DECL_RE does not match, so it "
                f'would go unpoliced -- write it as `inline constexpr const char* {m.group(1)} = "...";` '
                f"or widen the pattern"
            )
    if not py:
        errors.append(f'{PY_DECL}: no `..._ATTR = "..."` declarations found')
    return cpp, py, errors


def check_parity(cpp: dict[str, str], py: dict[str, str]) -> list[str]:
    """Report Python keys with no C++ pair, and identifier pairs that do not correspond.

    One-directional on purpose: ``function.h`` is the authoritative list, and a key no Python layer
    reads (``spmd_unwrapped``) needs no Python constant. A Python key with no C++ pair is an error --
    it names an attr no pass writes.
    """
    errors: list[str] = []
    expected_py = {cpp_ident_to_py(ident): (ident, key) for ident, key in cpp.items()}
    for ident, key in sorted(py.items()):
        pair = expected_py.get(ident)
        if pair is None:
            errors.append(
                f'{CPP_DECL}: missing a `kAttr...` declaration of "{key}" to pair with `{ident}` in {PY_DECL}'
            )
        elif pair[1] != key:
            errors.append(f'{PY_DECL}: `{ident}` is "{key}" but `{pair[0]}` in {CPP_DECL} is "{pair[1]}"')
    return errors


def scan_cpp(path: Path, rel: str, keys: dict[str, str]) -> list[str]:
    """Report bare key literals in a C++ source, ignoring comments."""
    errors: list[str] = []
    text = strip_cpp_comments(path.read_text(encoding="utf-8", errors="replace"))
    for lineno, line in enumerate(text.splitlines(), start=1):
        for key, ident in keys.items():
            if f'"{key}"' in line:
                errors.append(f'{rel}:{lineno}: bare attr key "{key}" -- use `{ident}` from {CPP_DECL}')
    return errors


def _mentions_attrs(node: ast.AST) -> bool:
    """True when any identifier or string reachable from ``node`` names an attrs mapping."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and ATTRS_NAME_RE.search(sub.id):
            return True
        if isinstance(sub, ast.Attribute) and ATTRS_NAME_RE.search(sub.attr):
            return True
        # getattr(func, "attrs", {}) -- the mapping is named by a literal.
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and ATTRS_NAME_RE.search(sub.value):
            return True
    return False


class _PyAttrKeyScanner(ast.NodeVisitor):
    """Collect bare attr-key literals at sites that provably read or write an attrs mapping."""

    def __init__(self, keys: dict[str, str | None], rel: str) -> None:
        self.keys = keys
        self.rel = rel
        self.errors: list[str] = []

    def _report(self, key: str, lineno: int) -> None:
        ident = self.keys[key]
        hint = f"use `{ident}`" if ident else f"declare it in {PY_DECL} and use that constant"
        self.errors.append(f'{self.rel}:{lineno}: bare attr key "{key}" -- {hint}')

    def _check(self, node: ast.AST | None) -> None:
        if isinstance(node, ast.Constant):
            key = node.value
            if isinstance(key, str) and key in self.keys:
                self._report(key, node.lineno)

    def _check_dict_keys(self, node: ast.AST) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Dict):
                for key in sub.keys:
                    self._check(key)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast visitor protocol
        # Anchor 1: <expr naming attrs>.get("key", ...)
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "get" and node.args:
            if _mentions_attrs(func.value):
                self._check(node.args[0])
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802 - ast visitor protocol
        # Anchor 2: <expr naming attrs>["key"]
        if _mentions_attrs(node.value):
            self._check(node.slice)
        self.generic_visit(node)

    def _visit_assign(self, targets: list[ast.expr], value: ast.AST | None) -> None:
        if value is None:
            return
        # Anchor 3: a dict-literal key in an assignment whose target names attrs.
        if any(_mentions_attrs(t) for t in targets):
            self._check_dict_keys(value)
        # Anchor 4: a module-level collection of attr keys, matched on the bound name.
        if any(isinstance(t, ast.Name) and ATTRS_COLLECTION_RE.search(t.id) for t in targets):
            for sub in ast.walk(value):
                if isinstance(sub, (ast.Set, ast.List, ast.Tuple)):
                    for elt in sub.elts:
                        self._check(elt)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802 - ast visitor protocol
        self._visit_assign(list(node.targets), node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802 - ast visitor protocol
        self._visit_assign([node.target], node.value)
        self.generic_visit(node)


def scan_py(path: Path, rel: str, keys: dict[str, str | None]) -> list[str]:
    """Report bare key literals at attrs-anchored sites, ignoring comments and docstrings."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError as e:
        return [f"{rel}: could not parse ({e})"]
    scanner = _PyAttrKeyScanner(keys, rel)
    scanner.visit(tree)
    return scanner.errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Repository root (default: inferred)")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[2]

    cpp, py, errors = parse_declarations(root)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    errors = check_parity(cpp, py)

    # key -> the identifier a use site should reach it through.
    cpp_keys = {key: ident for ident, key in cpp.items()}
    # Driven by the C++ list so a key with no Python constant yet is still reported at an
    # attrs-anchored Python site; the message then asks for the declaration rather than an import.
    py_by_key = {key: ident for ident, key in py.items()}
    py_keys: dict[str, str | None] = {key: py_by_key.get(key) for key in cpp_keys}

    for scan_root in SCAN_ROOTS:
        base = root / scan_root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if rel in EXEMPT:
                continue
            if path.suffix in (".h", ".hpp", ".cpp", ".cc"):
                errors.extend(scan_cpp(path, rel, cpp_keys))
            elif path.suffix == ".py":
                errors.extend(scan_py(path, rel, py_keys))

    if errors:
        print(
            "Reserved Function attr keys must reach every use site through a declared constant.\n",
            file=sys.stderr,
        )
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(
            f"\n{len(errors)} problem(s). Declare the key in {CPP_DECL} and {PY_DECL}"
            " (with its lifecycle comment) and import it at the use site.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {len(cpp)} reserved Function attr key(s) declared once per layer and in agreement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
