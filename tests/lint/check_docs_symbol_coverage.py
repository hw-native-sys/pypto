# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fail when a public ``pl.*``/``pld.*`` symbol is documented nowhere it should be.

The user manual promises that every user-visible capability is findable. This
check enforces the mechanical half of that promise: each name in
``pypto.language.__all__`` and each name in the ``pld`` family's four
``__all__`` lists must appear in some code span under ``docs/en/user/``.

The catalog is the complete classified operator index, so each
``pld.tensor.*`` operator must also have its own row in the catalog's
``## Distributed`` table. That is a stricter, table-specific check than the
general "documented somewhere" test above.

``__all__`` is read **statically** with ``ast`` rather than by importing the
modules — the import would need the compiled extension, which pre-commit
environments do not have.

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
_DISTRIBUTED = REPO_ROOT / "python" / "pypto" / "language" / "distributed"
DISTRIBUTED_INIT = _DISTRIBUTED / "__init__.py"
TENSOR_OPS = _DISTRIBUTED / "op" / "tensor_ops.py"
TILE_OPS = _DISTRIBUTED / "op" / "tile_ops.py"
SYSTEM_OPS = _DISTRIBUTED / "op" / "system_ops.py"
USER_DOCS = REPO_ROOT / "docs" / "en" / "user"
CATALOG = USER_DOCS / "ops" / "01-catalog.md"

# Each entry is (display prefix, the __all__-bearing files that make up the
# namespace). `pld`'s public surface is spread across its package __init__ and
# the three per-category op modules (pld.tensor / pld.tile / pld.system are
# aliases for these at import time — see op/__init__.py).
NAMESPACES: list[tuple[str, list[Path]]] = [
    ("pl", [LANGUAGE_INIT]),
    ("pld", [DISTRIBUTED_INIT, TENSOR_OPS, TILE_OPS, SYSTEM_OPS]),
]

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

# The async transfer family is intentionally deferred while its runtime support
# is still being completed. Documenting these as user-facing now risks being
# rewritten under us. Move each one out of DEFERRED (and into the chapter and
# catalog) once its support is stable and documented.
#
# put_async/get_async: V1 ops, lowered, but their V2 successors are in flight.
#
# The session/event surface below landed with the V2 IR and DSL ops but is not
# yet usable end to end: PUT has no backend lowering, GET's is in review, and
# in-kernel wait_event is rejected by URMA codegen. Documenting an op a user
# cannot compile would be worse than deferring it. These leave DEFERRED with the
# V2 backend batch, as each direction gains a lowering.
_ASYNC_V1 = "pld async ops -- not documented while runtime support is being completed"
_ASYNC_V2 = "pld V2 session/event surface -- deferred until the V2 backend batch lands its lowering"
DEFERRED: dict[str, str] = {
    "pld.put_async": _ASYNC_V1,
    "pld.get_async": _ASYNC_V1,
    "pld.tensor.put_async": _ASYNC_V1,
    "pld.tensor.get_async": _ASYNC_V1,
    # V2 session/event surface (merged in #3540).
    "pld.make_session": _ASYNC_V2,
    "pld.system.make_session": _ASYNC_V2,
    "pld.wait_event": _ASYNC_V2,
    "pld.system.wait_event": _ASYNC_V2,
    "pld.defer_wait_event": _ASYNC_V2,
    "pld.system.defer_wait_event": _ASYNC_V2,
    "pld.get_async_event": _ASYNC_V2,
    "pld.tensor.get_async_event": _ASYNC_V2,
    "pld.tile.put_async": _ASYNC_V2,
    "pld.tile.get_async": _ASYNC_V2,
}

# pld.tensor names that are permanent language abstractions, not catalog
# operators: they create/view the window a real operator then acts on.
CATALOG_EXEMPT = {"alloc_window_buffer", "window"}

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_QUALIFIED_PLD = re.compile(r"pld\.(?:(?:tensor|tile|system)\.)?[A-Za-z_][A-Za-z0-9_]*")
_PLD_TENSOR = re.compile(r"pld\.tensor\.([A-Za-z_][A-Za-z0-9_]*)")
_DISTRIBUTED_SECTION = re.compile(r"\n## Distributed\n(.*?)(?=\n## |\Z)", re.DOTALL)
_TABLE_SEPARATOR_CELL = re.compile(r":?-{3,}:?")


class CatalogError(Exception):
    """Raised when the catalog is unreadable, missing its section, or malformed."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def is_deferred(qualified_name: str) -> bool:
    """Return whether a fully qualified public symbol is intentionally deferred."""
    return qualified_name in DEFERRED


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


def collect_all_names(paths: list[Path]) -> list[str]:
    """Read and merge ``__all__`` from several files, order-preserving and deduplicated.

    Args:
        paths: Modules to read, in priority order for display purposes.

    Returns:
        Every distinct name across all of them, first-seen order.
    """
    seen: set[str] = set()
    names: list[str] = []
    for path in paths:
        for name in read_all_names(path):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


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


def documented_qualified_pld(docs_root: Path) -> set[str]:
    """Collect qualified ``pld`` API names, including short-form aliases.

    Bare identifiers collapse categories: ``pld.tensor.get`` would otherwise
    satisfy the check for ``pld.tile.get``.

    Args:
        docs_root: Directory to scan recursively for ``.md`` files.

    Returns:
        The set of qualified ``pld.*`` names found, including category-qualified names.
    """
    found: set[str] = set()
    for md in sorted(docs_root.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        for chunk in _CODE_FENCE.findall(text) + _INLINE_CODE.findall(text):
            found.update(_QUALIFIED_PLD.findall(chunk))
    return found


def _table_cells(line: str) -> list[str]:
    """Split a Markdown table row into its cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def catalog_distributed_names(catalog_path: Path) -> set[str]:
    """Collect every ``pld.tensor.<name>`` in the API column of the Distributed table.

    Args:
        catalog_path: Path to the operator catalog.

    Returns:
        The set of ``pld.tensor.*`` names with an API-column entry.

    Raises:
        CatalogError: If the catalog is unreadable, has no ``## Distributed``
            section, or one row names more than one ``pld.tensor`` operator.
    """
    try:
        text = catalog_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError([f"cannot read catalog {catalog_path}: {exc}"]) from exc

    match = _DISTRIBUTED_SECTION.search(text)
    if match is None:
        raise CatalogError([f"no '## Distributed' section found in {catalog_path}"])

    section = _CODE_FENCE.sub("", match.group(1))
    lines = section.splitlines()
    names: set[str] = set()
    row_errors: list[str] = []
    i = 0
    while i < len(lines):
        header = lines[i]
        if not header.lstrip().startswith("|") or i + 1 >= len(lines):
            i += 1
            continue
        header_cells = _table_cells(header)
        separator_cells = _table_cells(lines[i + 1])
        if (
            len(header_cells) > 1
            and header_cells[0] == "Operation"
            and header_cells[1] == "API"
            and len(separator_cells) == len(header_cells)
            and all(_TABLE_SEPARATOR_CELL.fullmatch(cell) for cell in separator_cells)
        ):
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = _table_cells(lines[i])
                if len(cells) > 1:
                    row_names = _PLD_TENSOR.findall(cells[1])
                    if len(row_names) > 1:
                        row_errors.append(
                            f"row {cells[0]} names multiple pld.tensor operators: {', '.join(row_names)}"
                        )
                    names.update(row_names)
                i += 1
            continue
        i += 1
    if row_errors:
        raise CatalogError(row_errors)
    return names


def check_catalog(required_catalog_names: list[str]) -> tuple[list[str], list[str]]:
    """Check the catalog's Distributed table for missing or malformed rows.

    Args:
        required_catalog_names: ``pld.tensor`` names that must have a row.

    Returns:
        A pair ``(missing_catalog, catalog_errors)``; both are empty on success.
    """
    try:
        catalog_names = catalog_distributed_names(CATALOG)
    except CatalogError as exc:
        for error in exc.errors:
            print(f"error: {error}")
        return [], exc.errors

    missing_catalog = sorted(n for n in required_catalog_names if n not in catalog_names)
    print(
        f"\npld.tensor catalog rows: {len(required_catalog_names) - len(missing_catalog)}/"
        f"{len(required_catalog_names)} present"
    )
    if missing_catalog:
        print(
            f"error: {len(missing_catalog)} pld.tensor name(s) have no row in "
            f"{CATALOG}'s ## Distributed table:"
        )
        for name in missing_catalog:
            print(f"  pld.tensor.{name}")
        print("\nAdd a catalog row, or add it to DEFERRED/CATALOG_EXEMPT with a reason.")
    return missing_catalog, []


def main() -> int:
    """Report or gate on undocumented public language symbols.

    Returns:
        0 when every non-deferred symbol is documented and the catalog is
        complete (or ``--report`` was passed), 1 otherwise.
    """
    report_only = "--report" in sys.argv[1:]

    if not USER_DOCS.is_dir():
        sys.exit(f"error: user manual directory not found: {USER_DOCS}")

    documented = documented_identifiers(USER_DOCS)

    any_undocumented = False
    for label, paths in NAMESPACES:
        exported = [n for n in collect_all_names(paths) if n not in NOT_SYMBOLS]
        missing = [n for n in exported if n not in documented]
        undocumented = sorted(n for n in missing if not is_deferred(f"{label}.{n}"))
        deferred_hits = sorted(n for n in missing if is_deferred(f"{label}.{n}"))

        covered = len(exported) - len(missing)
        print(
            f"{label}.__all__: {len(exported)} documentable symbols, {covered} documented, {len(missing)} not"
        )

        if deferred_hits:
            print(f"\ndeferred to a later batch ({len(deferred_hits)}):")
            for name in deferred_hits:
                qualified_name = f"{label}.{name}"
                print(f"  {qualified_name:<28} -> {DEFERRED[qualified_name]}")

        if undocumented:
            any_undocumented = True
            print(
                f"\nerror: {len(undocumented)} public symbol(s) appear in no code span under docs/en/user/:"
            )
            for name in undocumented:
                print(f"  {label}.{name}")
            print("\nDocument each one, or add it to DEFERRED with the batch that will cover it.")

    # Bare identifiers collapse categories: pld.tensor.get must not satisfy the
    # check for pld.tile.get. Require the category-qualified spelling.
    #
    # The package level needs the same treatment for the opposite reason: a bare
    # lookup accepts `foo` extracted from `pld.tensor.foo`, so a short alias in
    # pld.__all__ would count as covered while `pld.foo` is never written. The
    # short form is the spelling users type, so it has to appear on its own.
    qualified_documented = documented_qualified_pld(USER_DOCS)
    for prefix, module in (
        ("pld", DISTRIBUTED_INIT),
        ("pld.tensor", TENSOR_OPS),
        ("pld.tile", TILE_OPS),
        ("pld.system", SYSTEM_OPS),
    ):
        exported = [n for n in read_all_names(module) if n not in NOT_SYMBOLS]
        missing = [n for n in exported if f"{prefix}.{n}" not in qualified_documented]
        undocumented = sorted(n for n in missing if not is_deferred(f"{prefix}.{n}"))

        covered = len(exported) - len(missing)
        print(
            f"{prefix}.__all__: {len(exported)} documentable symbols, "
            f"{covered} documented, {len(missing)} not"
        )

        if undocumented:
            any_undocumented = True
            print(
                f"\nerror: {len(undocumented)} {prefix} symbol(s) appear in no "
                f"category-qualified code span under docs/en/user/:"
            )
            for name in undocumented:
                print(f"  {prefix}.{name}")
            print("\nDocument each one, or add it to DEFERRED with the batch that will cover it.")

    # A DEFERRED entry that is now documented is stale: it weakens the gate.
    stale = sorted(n for n in DEFERRED if n in qualified_documented)
    if stale:
        print("\nerror: these symbols are documented but still listed in DEFERRED — remove them:")
        for name in stale:
            print(f"  {name}")

    # Catalog completeness: every pld.tensor.* operator must have its own row.
    tensor_exported = [n for n in read_all_names(TENSOR_OPS) if n not in NOT_SYMBOLS]
    required_catalog_names = [
        n for n in tensor_exported if n not in CATALOG_EXEMPT and not is_deferred(f"pld.tensor.{n}")
    ]
    missing_catalog, catalog_errors = check_catalog(required_catalog_names)

    if report_only:
        return 0
    return 1 if (any_undocumented or stale or missing_catalog or catalog_errors) else 0


if __name__ == "__main__":
    sys.exit(main())
