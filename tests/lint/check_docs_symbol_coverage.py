# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fail when a public ``pl.*`` symbol is documented nowhere in the user manual.

The user manual promises that every user-visible capability is findable. This
check enforces the mechanical half of that promise: each name in
``pypto.language.__all__`` must appear in some code span under ``docs/en/user/``.

``__all__`` is read **statically** with ``ast`` rather than by importing
``pypto.language`` — the import would need the compiled extension, which
pre-commit environments do not have.

Symbols whose chapter is not written yet live in ``DEFERRED`` with the batch
that will cover them. That list is expected to shrink to empty; it is not a
place to park a symbol nobody wants to document.

Usage:
    python tests/lint/check_docs_symbol_coverage.py           # gate
    python tests/lint/check_docs_symbol_coverage.py --report  # list, always exit 0
"""

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LANGUAGE_INIT = REPO_ROOT / "python" / "pypto" / "language" / "__init__.py"
USER_DOCS = REPO_ROOT / "docs" / "en" / "user"

# Names that are namespaces or plumbing rather than user-facing operators.
# Documenting `pl.tile` as a symbol is meaningless — the chapter documents the
# namespace, and the operators inside it are checked individually.
NOT_SYMBOLS = {
    "adir",
    "array",
    "optimizations",
    "parser",
    "prefetch",
    "system",
    "tensor",
    "tile",
}

# Symbols whose home chapter is not written yet. Each entry names the batch that
# retires it. Remove entries as those chapters land — a stale entry silently
# weakens the gate.
DEFERRED = {
    # B3 tutorials / B5 performance — the prefetch surface has no guide yet.
    "AsyncEvent": "B6 execution",
    "AsyncSession": "B6 execution",
    "PrefetchAsyncContext": "B6 execution",
    # B6 tools — text parsing / serialization round-trip.
    "loads_program": "B6 tools",
    "parse_program": "B6 tools",
    # B6 execution — IR-level types surfaced mainly in printed IR.
    "MemRefType": "B6 execution",
    "PtrType": "B6 execution",
    "TileView": "B6 execution",
    "CompactMode": "B6 execution",
    "TileLayout": "B6 execution",
    "PipeType": "B6 execution",
}

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def read_all_names(init_path: Path) -> list[str]:
    """Extract ``__all__`` from a module without importing it.

    Args:
        init_path: Path to the module's ``__init__.py``.

    Returns:
        The string entries of the module's ``__all__``.

    Raises:
        SystemExit: If the module has no literal ``__all__`` list.
    """
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            continue
        if not isinstance(node.value, ast.List):
            break
        return [e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    sys.exit(f"error: no literal __all__ found in {init_path}")


def documented_identifiers(docs_root: Path) -> set[str]:
    """Collect every identifier appearing in a code span under ``docs_root``.

    Both fenced blocks and inline spans count. Dotted paths contribute each
    component, so ``pl.tile.load`` documents ``load``.

    Args:
        docs_root: Directory to scan recursively for ``.md`` files.

    Returns:
        The set of identifiers found.
    """
    found: set[str] = set()
    for md in sorted(docs_root.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        for chunk in _CODE_FENCE.findall(text) + _INLINE_CODE.findall(text):
            found.update(_IDENTIFIER.findall(chunk))
    return found


def main() -> int:
    """Report or gate on undocumented public language symbols.

    Returns:
        0 when every non-deferred symbol is documented (or ``--report`` was
        passed), 1 otherwise.
    """
    report_only = "--report" in sys.argv[1:]

    if not USER_DOCS.is_dir():
        sys.exit(f"error: user manual directory not found: {USER_DOCS}")

    exported = [n for n in read_all_names(LANGUAGE_INIT) if n not in NOT_SYMBOLS]
    documented = documented_identifiers(USER_DOCS)

    missing = [n for n in exported if n not in documented]
    undocumented = sorted(n for n in missing if n not in DEFERRED)
    deferred_hits = sorted(n for n in missing if n in DEFERRED)

    covered = len(exported) - len(missing)
    print(f"pl.__all__: {len(exported)} documentable symbols, {covered} documented, {len(missing)} not")

    if deferred_hits:
        print(f"\ndeferred to a later batch ({len(deferred_hits)}):")
        for name in deferred_hits:
            print(f"  {name:<28} -> {DEFERRED[name]}")

    # A DEFERRED entry that is now documented is stale: it weakens the gate.
    stale = sorted(n for n in DEFERRED if n in documented)
    if stale:
        print("\nerror: these symbols are documented but still listed in DEFERRED — remove them:")
        for name in stale:
            print(f"  {name}")

    if undocumented:
        print(f"\nerror: {len(undocumented)} public symbol(s) appear in no code span under docs/en/user/:")
        for name in undocumented:
            print(f"  pl.{name}")
        print("\nDocument each one, or add it to DEFERRED with the batch that will cover it.")

    if report_only:
        return 0
    return 1 if (undocumented or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
