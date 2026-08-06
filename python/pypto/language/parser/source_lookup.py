# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Class source lookup for the DSL decorators.

``inspect.getsourcelines`` has no fast path for a *class* before CPython 3.13
(which added ``__firstlineno__``): ``inspect.findsource`` ``ast.parse``-es the
**entire** containing file and walks the tree looking for a matching
``__qualname__``. A module defining N classes therefore pays N full-file parses,
so the cost of decorating them grows as O(N x file_size) — noticeable on
``@pl.program``-heavy modules such as the test suite.

This module resolves the same line number from a per-file index built with a
*single* parse, then defers to ``inspect.getblock`` for the block extraction so
the result stays identical to ``inspect.getsourcelines``. Anything the index
cannot resolve returns ``None`` so callers fall back to ``inspect``, which
remains the source of truth.

On CPython 3.13+ the index is bypassed entirely. There every class carries its
own ``__firstlineno__``, so ``inspect`` already resolves the line without
parsing — and, unlike a qualname-keyed index, it tells apart two classes that
share a ``__qualname__``. Deferring keeps that disambiguation exact instead of
approximating it.
"""

import ast
import inspect
import linecache

__all__ = ["get_class_source_lines"]


class _ClassLineIndexer(ast.NodeVisitor):
    """Index ``qualname -> 1-based first source line`` for every class in a module AST.

    Mirrors CPython's ``inspect._ClassFinder`` bookkeeping so lookups agree with
    ``inspect.findsource``: the ``<locals>`` marker for classes nested in
    functions, the decorator line taking precedence over the ``class`` line, and
    "first match in source order wins" when a qualname is defined more than once.
    """

    def __init__(self) -> None:
        self._stack: list[str] = []
        self.index: dict[str, int] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        self._stack.append("<locals>")
        self.generic_visit(node)
        self._stack.pop()
        self._stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._stack.append(node.name)
        first_line = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        # setdefault: generic_visit walks in source order, and CPython stops at
        # the first qualname match, so an earlier definition must win.
        self.index.setdefault(".".join(self._stack), first_line)
        self.generic_visit(node)
        self._stack.pop()


# Per-file class-line index, keyed by source filename. Each value keeps the exact
# ``lines`` list the index was built from, so staleness is an identity check.
_CLASS_LINE_INDEX_CACHE: dict[str, tuple[list[str], dict[str, int]]] = {}


def _class_line_index(source_file: str, lines: list[str]) -> dict[str, int]:
    """Return the ``qualname -> first line`` index for ``source_file``, parsing once.

    ``linecache`` hands back the *same* list object for a given cache entry and
    builds a fresh one whenever the file is re-read (or a synthetic entry is
    replaced), so holding a reference and comparing identity is both O(1) and
    exact — no size/mtime heuristic can go stale behind it.

    Args:
        source_file: Filename the lines came from (linecache key)
        lines: Full source lines of the file, as returned by linecache

    Returns:
        Mapping from class qualname to its 1-based first source line

    Raises:
        SyntaxError: If the file does not parse
    """
    cached = _CLASS_LINE_INDEX_CACHE.get(source_file)
    if cached is not None and cached[0] is lines:
        return cached[1]

    indexer = _ClassLineIndexer()
    indexer.visit(ast.parse("".join(lines)))
    _CLASS_LINE_INDEX_CACHE[source_file] = (lines, indexer.index)
    return indexer.index


def _records_own_first_line(cls: type) -> bool:
    """True when the runtime stamped ``cls`` with its own definition line.

    CPython 3.13+ writes ``__firstlineno__`` into every class body's namespace.
    Read it through ``vars`` exactly as ``inspect.findsource`` does: ``getattr``
    would inherit a base class's line and silently mislocate a subclass.
    """
    try:
        return "__firstlineno__" in vars(cls)
    except TypeError:
        return False


def _indexed_class_source_lines(cls: type) -> tuple[list[str], int] | None:
    """Resolve ``cls``'s source from the per-file index, or None to defer.

    Args:
        cls: Class to get source lines for

    Returns:
        Tuple of (source_lines, starting_line_1based) matching
        ``inspect.getsourcelines``, or None when the caller must fall back to
        ``inspect`` — either because the class is not in the index, or because
        the runtime already resolves it correctly and without parsing
    """
    # A class carrying its own line number needs no index, and a qualname-keyed
    # index would be *wrong* for it: two classes sharing a __qualname__ resolve
    # to distinct lines that only __firstlineno__ can tell apart. Let inspect do
    # it; the index exists solely for runtimes that would otherwise parse the
    # whole file once per class.
    if _records_own_first_line(cls):
        return None

    qualname = getattr(cls, "__qualname__", None)
    if not qualname:
        return None

    try:
        # Mirror inspect.findsource: getsourcefile() (not getfile) selects the
        # .py that linecache is keyed on, and checkcache() drops a stale entry.
        source_file = inspect.getsourcefile(cls)
        if not source_file:
            return None
        linecache.checkcache(source_file)
        module = inspect.getmodule(cls, source_file)
        lines = linecache.getlines(source_file, module.__dict__ if module else None)
        if not lines:
            return None

        starting_line = _class_line_index(source_file, lines).get(qualname)
        if starting_line is None:
            return None
        return inspect.getblock(lines[starting_line - 1 :]), starting_line
    except (OSError, TypeError, ValueError, SyntaxError, AttributeError):
        # Any surprise (unparseable file, no getblock, exotic loader) falls back
        # to inspect, which stays the source of truth for this lookup.
        return None


def get_class_source_lines(cls: type) -> tuple[list[str], int]:
    """Drop-in ``inspect.getsourcelines`` for a class, without the per-class parse.

    Args:
        cls: Class to get source lines for

    Returns:
        Tuple of (source_lines, starting_line_1based), identical to what
        ``inspect.getsourcelines(cls)`` returns

    Raises:
        OSError: If the source is unavailable, as ``inspect`` raises
        TypeError: If ``cls`` is a built-in or otherwise has no source
    """
    indexed = _indexed_class_source_lines(cls)
    if indexed is not None:
        return indexed
    return inspect.getsourcelines(cls)
