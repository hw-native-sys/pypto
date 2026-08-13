# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Script to check that operator-identity tests route operator names through the registry getter.

``.claude/rules/operator-identity-checks.md`` requires an operator-name literal to reach a comparison
via ``get_op(...)``, which raises on an unregistered name, rather than as a bare string. A bare
``call.op.name == "tile.reshaep"`` silently evaluates to ``False``; the same literal written as
``call.op.name == ir.get_op("tile.reshaep").name`` raises ``ValueError`` at the comparison site.

In test code the silent form is worse than in product code. These comparisons sit inside list
comprehensions and ``next(...)`` filters, so a stale name yields either an empty list that a
``len(...) == 0`` assertion happily accepts, or a bare ``StopIteration`` that reads as a
test-infrastructure fault rather than the rename it actually is.

The check parses each file rather than scanning text, so prose in comments and docstrings is never
flagged. Four shapes are reported, each anchored on something that provably holds operator names:

1. *Direct comparison* -- ``call.op.name == "tile.load"``, ``!=``, and ``in`` / ``not in`` against a
   set, tuple, or list literal.
2. *Sequence comparison* -- ``[c.op.name for c in calls] == ["tile.load", "tile.slice"]``.
3. *Bound-collection use* -- a name bound from an ``.op.name`` comprehension, then tested with
   ``"tile.load" in names``, ``not in``, or ``names.count("tile.load")``. The ``not in`` form is the
   worst of the family: a renamed operator makes the assertion pass vacuously.
4. *Module-level operator-name collection* -- ``_REQUIRED_OPS = {"pld.system.rank", ...}``, the shape
   the rule shows as the canonical anti-pattern, including when wrapped in a ``frozenset(...)``,
   ``set(...)``, ``list(...)`` or ``tuple(...)`` builder.

Scope and deliberate limits:

* Only ``<expr>.op.name`` is treated as an operator-identity site. That is the exact form the rule
  tabulates, and anchoring on it keeps the registry's own tests -- ``ir.get_op("tensor.add").name ==
  "tensor.add"`` in ``tests/ut/ir/operators/test_op_registry.py`` -- from being flagged, since there
  the literal already reaches the comparison through the getter.
* Only *dotted* literals are reported. A callee name carries no dot (``"k1"``, ``"consumer"``,
  ``"main_incore_1"``), and the rule explicitly excludes function names from conversion.
* Shape 4 is the one heuristic here, so it is deliberately narrow: every element must be a string,
  there must be at least two, and every first segment must name a real operator namespace. A
  collection of unrelated dotted strings therefore does not trip it. Widen ``OP_NAMESPACES`` only
  alongside a real namespace.
* Shape 3 tracks its bindings per lexical scope. A ``names`` built from ``.op.name`` in one function
  says nothing about a ``names`` built from something else in another, and rebinding shadows.
* A literal passed into a helper that does the comparison -- ``_collect_call_args(func,
  "tensor.muls")`` -- is *not* detected; that would need interprocedural analysis. Such sites are
  still worth converting by hand.
* Literals that are *constructed* rather than compared are exempt, per the rule: arguments to
  ``ir.Op(...)``, ``get_op(...)``, ``create_op_call(...)``, and the name-keyed registry lookups.
  ``@pytest.mark.parametrize("op_name", [...])`` lists that feed ``get_op(op_name)`` are likewise
  left alone -- ``get_op`` already raises on a typo there.
* The literal is not resolved against the live registry: pre-commit runs this in an isolated
  interpreter with no built ``pypto_core`` extension. Whether each name is registered is exactly
  what ``get_op`` verifies at runtime once the call site is converted.
"""

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

# Sites that legitimately compare a bare literal. Scoped to the exact class or function -- not the
# whole file -- so an unrelated bare comparison added to one of these files later is still reported.
# Each entry is "<repo-relative path>::<dotted qualname prefix>".
ALLOWLIST: frozenset[str] = frozenset()

# Directories scanned, relative to the repo root.
SCAN_ROOTS = ("tests", "python")

# An operator name is a dotted lowercase path: "tile.load", "pld.tensor.window",
# "builtin.tensor.allreduce". A callee name is a plain Python identifier and never matches.
OP_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")

# Leading segments that name a real operator namespace. Used only to keep the module-level
# collection heuristic from firing on unrelated dotted strings.
OP_NAMESPACES = frozenset({"tile", "tensor", "pld", "system", "array", "dist", "builtin"})

# Calls whose string arguments name an operator to *build* or *look up* rather than to compare.
EXEMPT_CALLS = frozenset({"Op", "get_op", "create_op_call", "is_op_registered", "get_op_memory_spec"})

# Builders that wrap a collection literal without changing which literals it holds.
COLLECTION_BUILDERS = frozenset({"frozenset", "set", "list", "tuple"})

LITERAL_OPS = (ast.Eq, ast.NotEq)
MEMBERSHIP_OPS = (ast.In, ast.NotIn)


def get_git_tracked_files(root_dir: Path) -> list[Path]:
    """Get list of git-tracked Python files under the scanned roots."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", *SCAN_ROOTS],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to get git tracked files: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: git command not found", file=sys.stderr)
        sys.exit(1)

    files = []
    for line in result.stdout.splitlines():
        path = root_dir / line
        if line.endswith(".py") and path.is_file():
            files.append(path)
    return files


def _is_op_name_access(node: ast.expr) -> bool:
    """Whether *node* is an ``<expr>.op.name`` attribute access."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "name"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "op"
    )


def _collection_elements(node: ast.expr) -> list[ast.expr] | None:
    """Elements of a collection literal, seeing through one builder call.

    ``frozenset({...})`` is the idiomatic way to spell an immutable module-level set -- and the form
    this checker's own remediation message recommends -- so the literal inside it must be reachable.
    """
    if isinstance(node, ast.Call):
        func = node.func
        builder = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if builder in COLLECTION_BUILDERS and len(node.args) == 1 and not node.keywords:
            node = node.args[0]
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        return list(node.elts)
    return None


def _op_name_literals(node: ast.expr) -> list[ast.Constant]:
    """Return the dotted operator-name literal nodes *node* contributes, as a comparison operand.

    Nodes rather than strings: the caller exempts constructor arguments by AST identity, so a literal
    must stay distinguishable from an identically spelled one elsewhere on the same line.
    """
    elements = _collection_elements(node)
    if elements is None:
        elements = [node]
    return [
        e
        for e in elements
        if isinstance(e, ast.Constant) and isinstance(e.value, str) and OP_NAME_RE.match(e.value)
    ]


def _is_op_name_sequence(node: ast.expr) -> bool:
    """Whether *node* is a comprehension whose elements are ``.op.name`` values."""
    return isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)) and _is_op_name_access(node.elt)


def _compare_violations(node: ast.Compare, op_name_vars: set[str]) -> list[ast.Constant]:
    """Return the bare operator-name literals compared against operator names in *node*.

    *op_name_vars* holds names bound from an ``.op.name`` comprehension, so a membership test
    against one of them is an operator-identity check even though no ``.op.name`` is in sight.
    """
    # A Compare chains operands: `a < b < c` is left=a, comparators=[b, c], ops=[Lt, Lt]. Each op
    # joins operands[i] and operands[i + 1], so a chain is checked pairwise rather than assuming
    # the single-operator shape.
    operands = [node.left, *node.comparators]
    found: list[ast.Constant] = []

    def holds_op_names(e: ast.expr) -> bool:
        return (
            _is_op_name_access(e)
            or _is_op_name_sequence(e)
            or (isinstance(e, ast.Name) and e.id in op_name_vars)
        )

    for i, op in enumerate(node.ops):
        lhs, rhs = operands[i], operands[i + 1]
        if isinstance(op, LITERAL_OPS):
            # Either side may hold the operator names; `"tile.load" == c.op.name` is the same
            # defect, and so is comparing a name sequence against a literal list.
            if holds_op_names(lhs):
                found.extend(_op_name_literals(rhs))
            if holds_op_names(rhs):
                found.extend(_op_name_literals(lhs))
        elif isinstance(op, MEMBERSHIP_OPS):
            # `"tile.load" in names` and `call.op.name in {...}` are both identity tests, but the
            # literal sits on opposite sides, so each direction is checked separately.
            if _is_op_name_access(lhs):
                found.extend(_op_name_literals(rhs))
            if holds_op_names(rhs):
                found.extend(_op_name_literals(lhs))
    return found


def _iter_own_scope(node: ast.AST):
    """Yield the nodes belonging to *node*'s own lexical scope, skipping nested scopes."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield child
        yield from _iter_own_scope(child)


def _scope_op_name_vars(scope: ast.AST, inherited: set[str]) -> set[str]:
    """Names bound to an ``.op.name`` sequence in *scope*, over the enclosing scope's bindings.

    Bindings must not leak sideways: ``names`` holding operator names in one function says nothing
    about a ``names`` built from file suffixes in another, and treating them alike would report a
    violation on unrelated code. Rebinding a name to anything else therefore shadows it here.
    """
    bound = set(inherited)
    for node in _iter_own_scope(scope):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if _is_op_name_sequence(value):
                bound.add(target.id)
            else:
                bound.discard(target.id)
    return bound


def _count_call_violations(node: ast.Call, op_name_vars: set[str]) -> list[ast.Constant]:
    """Literals passed to ``<op-name collection>.count(...)``."""
    fn = node.func
    if not (isinstance(fn, ast.Attribute) and fn.attr == "count" and node.args):
        return []
    if isinstance(fn.value, ast.Name) and fn.value.id in op_name_vars:
        return _op_name_literals(node.args[0])
    return []


def _module_collection_violations(tree: ast.Module) -> list[ast.Constant]:
    """Module-level collections built entirely from operator-name literals.

    Narrow on purpose -- every element must be a string naming a real operator namespace, and there
    must be at least two, so an unrelated set of dotted strings does not trip the check. A builder
    call such as ``frozenset({...})`` is unwrapped first, since the literals it guards are exactly
    as stale-able as those in a bare set.
    """
    out: list[ast.Constant] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
        else:
            continue
        if value is None:
            continue
        elements = _collection_elements(value)
        if elements is None or len(elements) < 2:
            continue
        literals: list[tuple[ast.Constant, str]] = []
        for element in elements:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                literals.append((element, element.value))
        if len(literals) != len(elements):
            continue
        if all(OP_NAME_RE.match(text) and text.split(".")[0] in OP_NAMESPACES for _, text in literals):
            out.extend(literal for literal, _ in literals)
    return out


def _exempt_literals(tree: ast.Module) -> set[int]:
    """Ids of string constants that *construct* or *look up* an operator rather than compare one."""
    exempt: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
        if name in EXEMPT_CALLS:
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    exempt.add(id(arg))
    return exempt


def find_violations(path: Path) -> list[tuple[int, str, str]]:
    """Return (line_number, operator_name, enclosing qualname) for each bare literal in *path*."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as e:
        print(f"Error: Failed to parse {path}: {e}", file=sys.stderr)
        sys.exit(1)

    # (literal node, enclosing qualname) so exemption below can match by AST identity.
    found: list[tuple[ast.Constant, str]] = []

    def walk(node: ast.AST, scope: tuple[str, ...], op_name_vars: set[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, (*scope, child.name), _scope_op_name_vars(child, op_name_vars))
                continue
            if isinstance(child, ast.Compare):
                found.extend(
                    (literal, ".".join(scope)) for literal in _compare_violations(child, op_name_vars)
                )
            elif isinstance(child, ast.Call):
                found.extend(
                    (literal, ".".join(scope)) for literal in _count_call_violations(child, op_name_vars)
                )
            walk(child, scope, op_name_vars)

    walk(tree, (), _scope_op_name_vars(tree, set()))
    found.extend((literal, "<module>") for literal in _module_collection_violations(tree))

    # Construction and name-keyed lookups are exempt per the rule. Matching the exact AST node keeps
    # an exempt lookup from suppressing a real comparison of the same name on the same line, as in
    # `assert call.op.name == "tile.load"; ir.get_op("tile.load")`.
    exempt = _exempt_literals(tree)
    violations: list[tuple[int, str, str]] = []
    for literal, qualname in found:
        text = literal.value
        if id(literal) not in exempt and isinstance(text, str):
            violations.append((literal.lineno, text, qualname))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that operator-name literals reach comparisons through the registry getter."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (defaults to the repo containing this script)",
    )
    args = parser.parse_args()
    root_dir = args.root.resolve()

    total = 0
    for path in get_git_tracked_files(root_dir):
        rel = path.relative_to(root_dir).as_posix()
        if rel == f"tests/lint/{Path(__file__).name}":
            continue
        for lineno, op_name, qualname in find_violations(path):
            site = f"{rel}::{qualname}"
            if any(site == a or site.startswith(f"{a}.") for a in ALLOWLIST):
                continue
            print(f'{rel}:{lineno}: bare operator name "{op_name}" in {qualname or "<module>"}')
            total += 1

    if total:
        print(
            f"\nFound {total} bare operator-name literal comparison(s).\n"
            "Route the literal through the registry getter, which raises on an unregistered name:\n"
            '    call.op.name == "tile.load"       ->  call.op.name == ir.get_op("tile.load").name\n'
            "For a name used more than once in a file, or inside a comprehension predicate, hoist a\n"
            "module-level constant so the literal is validated once at import:\n"
            '    _TILE_LOAD = ir.get_op("tile.load").name\n'
            "    loads = [c for c in calls if c.op.name == _TILE_LOAD]\n"
            "Build membership sets the same way, and keep the runtime check a name test:\n"
            '    _LOAD_LIKE = frozenset({ir.get_op("tile.load").name, ir.get_op("tile.read").name})\n'
            "Callee names carry no dot and are never reported. If a site genuinely must compare a\n"
            f"bare literal, add its `<path>::<qualname>` to ALLOWLIST in {Path(__file__).name}.\n"
            "See .claude/rules/operator-identity-checks.md.",
            file=sys.stderr,
        )
        return 1

    print("All operator-name literals reach their comparison through the registry getter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
