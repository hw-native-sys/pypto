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

**Duplicate qualnames.** A qualname is not unique: two branches of one function
may each define ``class Prog``, and both carry the qualname
``make.<locals>.Prog``. ``inspect.findsource`` resolves that by source order —
it returns the *first* definition for every one of them, so a decorator built on
it silently parses the wrong body. This module instead locates the definition
that actually produced ``cls`` from the line numbers of the code objects its
body defines in this same file, and raises `DuplicateClassDefinitionError` when
nothing in the class object can distinguish the candidates. Guessing is never a
valid answer here: the caller would compile a body the user never wrote.

On CPython 3.13+ the index is bypassed entirely. There every class carries its
own ``__firstlineno__``, so ``inspect`` already resolves the line without
parsing — and it tells duplicate qualnames apart, which a qualname-keyed
index cannot.
"""

import ast
import dataclasses
import inspect
import linecache
import os
import types
from collections.abc import Iterator
from typing import Any

__all__ = ["DuplicateClassDefinitionError", "get_class_source_lines"]


class DuplicateClassDefinitionError(RuntimeError):
    """Raised when a class's source definition cannot be identified unambiguously.

    Carries the candidate definition lines so callers can render an actionable
    diagnostic instead of compiling an arbitrary one of them.
    """

    def __init__(self, cls: type, source_file: str, first_lines: list[int]) -> None:
        self.qualname: str = getattr(cls, "__qualname__", None) or getattr(cls, "__name__", "<unknown>")
        self.source_file = source_file
        self.first_lines = first_lines
        locations = ", ".join(str(line) for line in first_lines)
        super().__init__(
            f"'{self.qualname}' is defined {len(first_lines)} times in {source_file} "
            f"(lines {locations}), and nothing on the class object identifies which "
            "definition built it"
        )


@dataclasses.dataclass(frozen=True)
class _ClassSite:
    """One ``class X: ...`` definition, as located in a file's AST.

    Attributes:
        first_line: 1-based line ``inspect.getsourcelines`` would start the block
            at — the first decorator's line when decorated, else the ``class`` line
        header_line: 1-based line of the ``class`` keyword itself
        end_line: 1-based last line of the class body
    """

    first_line: int
    header_line: int
    end_line: int

    def contains(self, line: int) -> bool:
        """True when ``line`` falls inside this class's body."""
        return self.header_line <= line <= self.end_line


class _ClassLineIndexer(ast.NodeVisitor):
    """Index ``qualname -> [class site]`` for every class in a module AST.

    Mirrors CPython's ``inspect._ClassFinder`` bookkeeping so lookups agree with
    ``inspect.findsource``: the ``<locals>`` marker for classes nested in
    functions, and the decorator line taking precedence over the ``class`` line.
    Unlike ``_ClassFinder`` it keeps *every* definition of a repeated qualname,
    in source order, so the caller can pick the right one rather than assume the
    first.
    """

    def __init__(self) -> None:
        self._stack: list[str] = []
        self.index: dict[str, list[_ClassSite]] = {}

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
        site = _ClassSite(first_line, node.lineno, node.end_lineno or node.lineno)
        # generic_visit walks in source order, so appending keeps the sites sorted.
        self.index.setdefault(".".join(self._stack), []).append(site)
        self.generic_visit(node)
        self._stack.pop()


# Per-file class-line index, keyed by source filename. Each value keeps the exact
# ``lines`` list the index was built from, so staleness is an identity check.
_CLASS_LINE_INDEX_CACHE: dict[str, tuple[list[str], dict[str, list[_ClassSite]]]] = {}


def _class_line_index(source_file: str, lines: list[str]) -> dict[str, list[_ClassSite]]:
    """Return the ``qualname -> class sites`` index for ``source_file``, parsing once.

    ``linecache`` hands back the *same* list object for a given cache entry and
    builds a fresh one whenever the file is re-read (or a synthetic entry is
    replaced), so holding a reference and comparing identity is both O(1) and
    exact — no size/mtime heuristic can go stale behind it.

    Args:
        source_file: Filename the lines came from (linecache key)
        lines: Full source lines of the file, as returned by linecache

    Returns:
        Mapping from class qualname to every definition of it, in source order

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


def _code_objects(value: Any) -> Iterator[types.CodeType]:
    """Yield the code objects a class-body entry directly holds.

    Covers plain functions plus the descriptors a class body commonly wraps them
    in — ``staticmethod`` / ``classmethod`` and ``property``. Anything else
    (constants, nested classes, arbitrary objects) yields nothing and simply
    contributes no evidence.
    """
    if isinstance(value, (staticmethod, classmethod)):
        holders: tuple[Any, ...] = (value.__func__,)
    elif isinstance(value, property):
        holders = (value.fget, value.fset, value.fdel)
    else:
        holders = (value,)

    for holder in holders:
        code = getattr(holder, "__code__", None)
        if isinstance(code, types.CodeType):
            yield code


def _identifies_source_file(candidate: str, source_file: str) -> bool:
    """True when ``candidate`` names the same file as ``source_file``.

    Compares the strings first, which is the normal case — a module's methods
    record the very filename ``inspect.getsourcefile`` reports for its classes —
    and falls back to an inode comparison so a symlinked or differently-spelled
    path still counts. A name no file backs, such as an ``exec`` pseudo-filename,
    identifies nothing.
    """
    if candidate == source_file:
        return True
    try:
        return os.path.samefile(candidate, source_file)
    except (OSError, ValueError):
        return False


def _member_definition_lines(cls: type, source_file: str) -> set[int]:
    """Return the ``source_file`` lines of the code objects defined in ``cls``'s body.

    Each line is where CPython recorded the member's definition — the first
    decorator line for a decorated ``def``, the ``def`` line otherwise. Both fall
    strictly inside the owning ``class`` block, which is what makes them usable
    as evidence of *which* same-named class body ran.

    A line number only means something in the file it was recorded against, so a
    member whose code object comes from *another* file is dropped rather than
    measured against this one. A class body may hold such a member as an ordinary
    attribute (``helper = some_imported_function``), and its unrelated line can
    otherwise land inside a sibling candidate's block and manufacture an
    ambiguity that does not exist. Members assigned from elsewhere in this same
    file need no special case: they land outside every candidate and are simply
    not evidence.
    """
    try:
        namespace = vars(cls)
    except TypeError:
        return set()

    lines: set[int] = set()
    # Filenames repeat across a class body; resolving each one once keeps the
    # inode fallback off the per-member path.
    is_this_file: dict[str, bool] = {}
    for value in namespace.values():
        for code in _code_objects(value):
            filename = code.co_filename
            if filename not in is_this_file:
                is_this_file[filename] = _identifies_source_file(filename, source_file)
            if is_this_file[filename]:
                lines.add(code.co_firstlineno)
    return lines


def _resolve_class_site(cls: type, source_file: str, sites: list[_ClassSite]) -> _ClassSite:
    """Pick the definition among ``sites`` that produced ``cls``.

    Args:
        cls: Class being located
        source_file: File the sites were indexed from; also scopes which member
            code objects count as line evidence
        sites: Every definition of ``cls``'s qualname, in source order

    Returns:
        The single matching site

    Raises:
        DuplicateClassDefinitionError: If the qualname is defined more than once
            and the class body's code objects do not single out one definition
    """
    if len(sites) == 1:
        return sites[0]

    member_lines = _member_definition_lines(cls, source_file)
    matched = [site for site in sites if any(site.contains(line) for line in member_lines)]
    if len(matched) == 1:
        return matched[0]
    raise DuplicateClassDefinitionError(cls, source_file, [site.first_line for site in sites])


def _indexed_class_source_lines(cls: type) -> tuple[list[str], int] | None:
    """Resolve ``cls``'s source from the per-file index, or None to defer.

    Args:
        cls: Class to get source lines for

    Returns:
        Tuple of (source_lines, starting_line_1based) matching
        ``inspect.getsourcelines``, or None when the caller must fall back to
        ``inspect`` — either because the class is not in the index, or because
        the runtime already resolves it correctly and without parsing

    Raises:
        DuplicateClassDefinitionError: If the file defines ``cls``'s qualname
            more than once and the right definition cannot be identified
    """
    # A class carrying its own line number needs no index, and it already
    # resolves duplicate qualnames exactly. Let inspect do it; the index exists
    # solely for runtimes that would otherwise parse the whole file once per
    # class and then mis-resolve those duplicates.
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

        sites = _class_line_index(source_file, lines).get(qualname)
        if not sites:
            return None
    except (OSError, TypeError, ValueError, SyntaxError, AttributeError):
        # Any surprise (unparseable file, no getblock, exotic loader) falls back
        # to inspect, which stays the source of truth for this lookup.
        return None

    # Outside the try: an unresolvable duplicate is a real diagnostic, not a
    # reason to fall back to inspect — which would answer with the first
    # definition and hide the ambiguity again.
    starting_line = _resolve_class_site(cls, source_file, sites).first_line
    try:
        return inspect.getblock(lines[starting_line - 1 :]), starting_line
    except (OSError, TypeError, ValueError, SyntaxError, AttributeError):
        return None


def get_class_source_lines(cls: type) -> tuple[list[str], int]:
    """Drop-in ``inspect.getsourcelines`` for a class, without the per-class parse.

    Args:
        cls: Class to get source lines for

    Returns:
        Tuple of (source_lines, starting_line_1based), identical to what
        ``inspect.getsourcelines(cls)`` returns — except that a repeated
        qualname resolves to the definition that actually built ``cls`` rather
        than to the first one in the file

    Raises:
        DuplicateClassDefinitionError: If the file defines ``cls``'s qualname
            more than once and the right definition cannot be identified
        OSError: If the source is unavailable, as ``inspect`` raises
        TypeError: If ``cls`` is a built-in or otherwise has no source
    """
    indexed = _indexed_class_source_lines(cls)
    if indexed is not None:
        return indexed
    return inspect.getsourcelines(cls)
