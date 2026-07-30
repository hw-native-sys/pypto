# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build safely highlighted, foldable diffs between IR pass snapshots."""

import ast
import difflib
import html
import io
import keyword
import textwrap
import token
import tokenize
from dataclasses import dataclass
from typing import Literal

from .model import DiffHunk, DiffRow, DiffSection, IRTraceError, PassTrace, Snapshot, split_source_lines

_TOKEN_CLASSES = {
    token.STRING: "tok-string",
    token.NUMBER: "tok-number",
    token.COMMENT: "tok-comment",
    token.OP: "tok-operator",
}

_RowKind = Literal["equal", "insert", "delete", "replace"]
_AlignedRow = tuple[_RowKind, int | None, int | None]


@dataclass(frozen=True)
class _SourceRegion:
    function_key: str | None
    function_name: str | None
    start: int
    end: int


@dataclass(frozen=True)
class _RegionPair:
    before: _SourceRegion | None
    after: _SourceRegion | None
    function_key: str | None
    function_name: str | None


@dataclass(frozen=True)
class _SectionAlignment:
    pair: _RegionPair
    rows: tuple[_AlignedRow, ...]
    inserted: int
    deleted: int


def _escape_html(text: str, *, quote: bool = False) -> str:
    """Escape HTML-sensitive text and JavaScript line separators."""
    return html.escape(text, quote=quote).replace("\u2028", "&#x2028;").replace("\u2029", "&#x2029;")


def _python_token_spans(
    text: str,
    lines: tuple[str, ...],
) -> list[list[tuple[int, int, str]]] | None:
    """Collect non-overlapping syntax spans, or return None for invalid source."""
    spans: list[list[tuple[int, int, str]]] = [[] for _ in lines]
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for item in tokens:
            css_class = _TOKEN_CLASSES.get(item.type)
            if item.type == token.NAME and keyword.iskeyword(item.string):
                css_class = "tok-keyword"
            if css_class is None:
                continue

            start_line, start_column = item.start
            end_line, end_column = item.end
            if start_line < 1 or end_line < start_line or end_line > len(lines):
                raise ValueError("token position is outside source lines")
            for line_number in range(start_line, end_line + 1):
                line_index = line_number - 1
                span_start = start_column if line_number == start_line else 0
                span_end = end_column if line_number == end_line else len(lines[line_index])
                if not 0 <= span_start <= span_end <= len(lines[line_index]):
                    raise ValueError("token column is outside source line")
                spans[line_index].append((span_start, span_end, css_class))

        for line_spans in spans:
            current_column = 0
            for span_start, span_end, _css_class in line_spans:
                if span_start < current_column:
                    raise ValueError("token spans overlap")
                current_column = span_end
    except (tokenize.TokenError, SyntaxError, ValueError):
        return None
    return spans


def _render_highlighted_line(
    line: str,
    token_spans: list[tuple[int, int, str]],
    changed_ranges: tuple[tuple[int, int], ...],
    change_class: str | None,
    *,
    escape_quotes: bool,
) -> str:
    """Render one escaped line with syntax and change classes."""
    boundaries = {0, len(line)}
    for span_start, span_end, _css_class in token_spans:
        boundaries.update((span_start, span_end))
    for range_start, range_end in changed_ranges:
        boundaries.update((range_start, range_end))

    fragments: list[str] = []
    ordered = sorted(boundaries)
    for start, end in zip(ordered, ordered[1:], strict=False):
        classes: list[str] = []
        token_class = next(
            (
                css_class
                for span_start, span_end, css_class in token_spans
                if span_start <= start and end <= span_end
            ),
            None,
        )
        if token_class is not None:
            classes.append(token_class)
        if change_class is not None and any(
            range_start <= start and end <= range_end for range_start, range_end in changed_ranges
        ):
            classes.append(change_class)

        fragment = _escape_html(line[start:end], quote=escape_quotes)
        if classes:
            class_names = " ".join(classes)
            fragment = f'<span class="{class_names}">{fragment}</span>'
        fragments.append(fragment)
    return "".join(fragments)


def _highlight_python(
    text: str,
    changed_ranges: dict[int, tuple[tuple[int, int], ...]] | None = None,
    change_class: str | None = None,
) -> tuple[str, ...]:
    """Highlight Python syntax and optional changed character ranges."""
    lines = split_source_lines(text)
    token_spans = _python_token_spans(text, lines)
    tokenization_failed = token_spans is None
    if token_spans is None:
        token_spans = [[] for _ in lines]
    changes = changed_ranges or {}
    return tuple(
        _render_highlighted_line(
            line,
            spans,
            changes.get(index, ()),
            change_class,
            escape_quotes=tokenization_failed,
        )
        for index, (line, spans) in enumerate(zip(lines, token_spans, strict=True))
    )


def highlight_python(text: str) -> tuple[str, ...]:
    """Return one safely escaped HTML fragment for each Python source line.

    Args:
        text: Python source text to highlight.

    Returns:
        Escaped per-line HTML fragments. Invalid source falls back to escaped
        plain text so rendering remains safe.
    """
    return _highlight_python(text)


def _intraline_ranges(
    before_line: str,
    after_line: str,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Return deleted and inserted character ranges for aligned lines."""
    before_ranges: list[tuple[int, int]] = []
    after_ranges: list[tuple[int, int]] = []
    matcher = difflib.SequenceMatcher(a=before_line, b=after_line, autojunk=False)
    for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if tag in ("delete", "replace") and before_start != before_end:
            before_ranges.append((before_start, before_end))
        if tag in ("insert", "replace") and after_start != after_end:
            after_ranges.append((after_start, after_end))
    return tuple(before_ranges), tuple(after_ranges)


def _qualified_name(node: ast.expr) -> str | None:
    """Return a dotted name for a Python expression when one is available."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _qualified_name(node.value)
        return f"{owner}.{node.attr}" if owner is not None else None
    return None


def _line_operation_key(line: str) -> str | None:
    """Return the called operation for one complete assignment/expression line."""
    try:
        module = ast.parse(textwrap.dedent(line))
    except (IndentationError, SyntaxError):
        return None
    if len(module.body) != 1:
        return None

    statement = module.body[0]
    if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
        return None
    value = statement.value
    if not isinstance(value, ast.Call):
        return None
    return _qualified_name(value.func)


def _align_operation_rows(
    before_lines: tuple[str, ...],
    after_lines: tuple[str, ...],
    before_start: int,
    before_end: int,
    after_start: int,
    after_end: int,
) -> tuple[_AlignedRow, ...]:
    """Align a replacement range around ordered matching operations."""
    before_ops = [
        (index, key)
        for index in range(before_start, before_end)
        if (key := _line_operation_key(before_lines[index])) is not None
    ]
    after_ops = [
        (index, key)
        for index in range(after_start, after_end)
        if (key := _line_operation_key(after_lines[index])) is not None
    ]
    matcher = difflib.SequenceMatcher(
        a=[key for _index, key in before_ops],
        b=[key for _index, key in after_ops],
        autojunk=False,
    )
    pairs = tuple(
        (before_ops[match.a + offset][0], after_ops[match.b + offset][0])
        for match in matcher.get_matching_blocks()
        for offset in range(match.size)
    )

    rows: list[_AlignedRow] = []
    if not pairs:
        paired = min(before_end - before_start, after_end - after_start)
        rows.extend(("replace", before_start + offset, after_start + offset) for offset in range(paired))
        rows.extend(("delete", index, None) for index in range(before_start + paired, before_end))
        rows.extend(("insert", None, index) for index in range(after_start + paired, after_end))
        return tuple(rows)

    before_cursor, after_cursor = before_start, after_start
    for before_index, after_index in pairs:
        rows.extend(("delete", index, None) for index in range(before_cursor, before_index))
        rows.extend(("insert", None, index) for index in range(after_cursor, after_index))
        rows.append(("replace", before_index, after_index))
        before_cursor, after_cursor = before_index + 1, after_index + 1
    rows.extend(("delete", index, None) for index in range(before_cursor, before_end))
    rows.extend(("insert", None, index) for index in range(after_cursor, after_end))
    return tuple(rows)


def _align_replace_rows(
    before_lines: tuple[str, ...],
    after_lines: tuple[str, ...],
    before_start: int,
    before_end: int,
    after_start: int,
    after_end: int,
) -> tuple[_AlignedRow, ...]:
    """Align a replacement block by normalized text, then operation name."""
    matcher = difflib.SequenceMatcher(
        a=[line.lstrip() for line in before_lines[before_start:before_end]],
        b=[line.lstrip() for line in after_lines[after_start:after_end]],
        autojunk=False,
    )
    rows: list[_AlignedRow] = []
    for (
        tag,
        local_before_start,
        local_before_end,
        local_after_start,
        local_after_end,
    ) in matcher.get_opcodes():
        block_before_start = before_start + local_before_start
        block_before_end = before_start + local_before_end
        block_after_start = after_start + local_after_start
        block_after_end = after_start + local_after_end
        if tag == "equal":
            for before_index, after_index in zip(
                range(block_before_start, block_before_end),
                range(block_after_start, block_after_end),
                strict=True,
            ):
                kind: _RowKind = (
                    "equal" if before_lines[before_index] == after_lines[after_index] else "replace"
                )
                rows.append((kind, before_index, after_index))
        elif tag == "insert":
            rows.extend(("insert", None, index) for index in range(block_after_start, block_after_end))
        elif tag == "delete":
            rows.extend(("delete", index, None) for index in range(block_before_start, block_before_end))
        elif tag == "replace":
            rows.extend(
                _align_operation_rows(
                    before_lines,
                    after_lines,
                    block_before_start,
                    block_before_end,
                    block_after_start,
                    block_after_end,
                )
            )
    return tuple(rows)


def _align_rows(
    before: Snapshot,
    after: Snapshot,
    before_range: tuple[int, int] | None = None,
    after_range: tuple[int, int] | None = None,
) -> tuple[
    tuple[_AlignedRow, ...],
    int,
    int,
    dict[int, tuple[tuple[int, int], ...]],
    dict[int, tuple[tuple[int, int], ...]],
]:
    """Align source ranges and return global indexes plus intraline changes."""
    inserted = 0
    deleted = 0
    before_start_offset, before_end_offset = before_range or (0, len(before.lines))
    after_start_offset, after_end_offset = after_range or (0, len(after.lines))
    matcher = difflib.SequenceMatcher(
        a=before.lines[before_start_offset:before_end_offset],
        b=after.lines[after_start_offset:after_end_offset],
        autojunk=False,
    )
    opcodes = matcher.get_opcodes()
    aligned_rows: list[_AlignedRow] = []
    before_changes: dict[int, tuple[tuple[int, int], ...]] = {}
    after_changes: dict[int, tuple[tuple[int, int], ...]] = {}

    for tag, before_start, before_end, after_start, after_end in opcodes:
        before_start += before_start_offset
        before_end += before_start_offset
        after_start += after_start_offset
        after_end += after_start_offset
        before_count = before_end - before_start
        after_count = after_end - after_start
        if tag == "equal":
            aligned_rows.extend(
                ("equal", before_start + offset, after_start + offset) for offset in range(before_count)
            )
        elif tag == "insert":
            inserted += after_count
            aligned_rows.extend(("insert", None, index) for index in range(after_start, after_end))
        elif tag == "delete":
            deleted += before_count
            aligned_rows.extend(("delete", index, None) for index in range(before_start, before_end))
        elif tag == "replace":
            inserted += after_count
            deleted += before_count
            aligned_rows.extend(
                _align_replace_rows(
                    before.lines,
                    after.lines,
                    before_start,
                    before_end,
                    after_start,
                    after_end,
                )
            )

    for kind, before_index, after_index in aligned_rows:
        if kind != "replace" or before_index is None or after_index is None:
            continue
        before_ranges, after_ranges = _intraline_ranges(before.lines[before_index], after.lines[after_index])
        before_changes[before_index] = before_ranges
        after_changes[after_index] = after_ranges

    return tuple(aligned_rows), inserted, deleted, before_changes, after_changes


def _materialize_rows(
    aligned_rows: tuple[_AlignedRow, ...],
    before_html: tuple[str, ...],
    after_html: tuple[str, ...],
) -> tuple[DiffRow, ...]:
    """Convert aligned global indexes into rendered diff rows."""
    return tuple(
        DiffRow(
            kind=kind,
            before_number=before_index + 1 if before_index is not None else None,
            before_html=before_html[before_index] if before_index is not None else "",
            after_number=after_index + 1 if after_index is not None else None,
            after_html=after_html[after_index] if after_index is not None else "",
        )
        for kind, before_index, after_index in aligned_rows
    )


def _diff_rows(
    before: Snapshot,
    after: Snapshot,
    before_range: tuple[int, int] | None = None,
    after_range: tuple[int, int] | None = None,
) -> tuple[tuple[DiffRow, ...], int, int]:
    """Align two snapshot ranges into display rows and count changed lines."""
    aligned_rows, inserted, deleted, before_changes, after_changes = _align_rows(
        before, before_range=before_range, after=after, after_range=after_range
    )
    before_html = _highlight_python(before.text, before_changes, "diff-delete")
    after_html = _highlight_python(after.text, after_changes, "diff-insert")
    rows = _materialize_rows(aligned_rows, before_html, after_html)
    return rows, inserted, deleted


def _function_bounds(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int] | None:
    """Return zero-based source bounds including decorators for one function."""
    end = node.end_lineno
    if end is None:
        return None
    first_line = min((decorator.lineno for decorator in node.decorator_list), default=node.lineno)
    return first_line - 1, end


def _extract_source_regions(snapshot: Snapshot) -> tuple[_SourceRegion, ...] | None:
    """Partition a snapshot into direct function regions and anonymous gaps."""
    try:
        module = ast.parse(snapshot.text)
    except SyntaxError:
        return None

    functions: list[_SourceRegion] = []

    def add_function(node: ast.FunctionDef | ast.AsyncFunctionDef, owner: str | None = None) -> bool:
        bounds = _function_bounds(node)
        if bounds is None:
            return False
        start, end = bounds
        if not 0 <= start < end <= len(snapshot.lines):
            return False
        key = f"{owner}.{node.name}" if owner is not None else node.name
        functions.append(_SourceRegion(key, node.name, start, end))
        return True

    for statement in module.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not add_function(statement):
                return None
        elif isinstance(statement, ast.ClassDef):
            for child in statement.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not add_function(
                    child, statement.name
                ):
                    return None

    functions.sort(key=lambda region: region.start)
    keys = [region.function_key for region in functions]
    if len(keys) != len(set(keys)):
        return None

    regions: list[_SourceRegion] = []
    cursor = 0
    for function in functions:
        if function.start < cursor:
            return None
        regions.append(_SourceRegion(None, None, cursor, function.start))
        regions.append(function)
        cursor = function.end
    regions.append(_SourceRegion(None, None, cursor, len(snapshot.lines)))
    return tuple(regions)


def _fallback_section(before: Snapshot, after: Snapshot, context: int) -> DiffSection:
    """Build the existing whole-file diff when section extraction is unavailable."""
    rows, inserted, deleted = _diff_rows(before, after)
    return DiffSection(
        function_key=None,
        function_name=None,
        inserted=inserted,
        deleted=deleted,
        hunks=_fold_rows(rows, context),
    )


def _pair_source_regions(
    before_regions: tuple[_SourceRegion, ...],
    after_regions: tuple[_SourceRegion, ...],
) -> tuple[_RegionPair, ...]:
    """Pair exact function anchors and preserve all one-sided source regions."""
    before_gaps = before_regions[::2]
    after_gaps = after_regions[::2]
    before_functions = before_regions[1::2]
    after_functions = after_regions[1::2]
    before_keys = {region.function_key for region in before_functions}
    after_entries = {
        region.function_key: (region, after_gaps[index + 1]) for index, region in enumerate(after_functions)
    }
    additions_before: dict[str | None, list[tuple[int, _SourceRegion]]] = {}
    next_common_key: str | None = None
    # Keep Before order, placing After-only functions before their next common key.
    for after_index in range(len(after_functions) - 1, -1, -1):
        after_function = after_functions[after_index]
        if after_function.function_key in before_keys:
            next_common_key = after_function.function_key
        else:
            additions_before.setdefault(next_common_key, []).append((after_index, after_function))

    pairs = [_RegionPair(before_gaps[0], after_gaps[0], None, None)]

    def append_additions(next_key: str | None) -> None:
        for after_index, after_function in reversed(additions_before.get(next_key, [])):
            pairs.append(
                _RegionPair(
                    None,
                    after_function,
                    after_function.function_key,
                    after_function.function_name,
                )
            )
            pairs.append(_RegionPair(None, after_gaps[after_index + 1], None, None))

    for before_index, before_function in enumerate(before_functions):
        append_additions(before_function.function_key)
        after_function, after_gap = after_entries.get(before_function.function_key, (None, None))
        pairs.append(
            _RegionPair(
                before_function,
                after_function,
                before_function.function_key,
                before_function.function_name,
            )
        )
        pairs.append(
            _RegionPair(
                before_gaps[before_index + 1],
                after_gap,
                None,
                None,
            )
        )
    append_additions(None)
    return tuple(pairs)


def _build_sections(before: Snapshot, after: Snapshot, context: int) -> tuple[DiffSection, ...]:
    """Build function-aware source sections with one highlight pass per snapshot."""
    before_regions = _extract_source_regions(before)
    after_regions = _extract_source_regions(after)
    if before_regions is None or after_regions is None:
        return (_fallback_section(before, after, context),)

    alignments: list[_SectionAlignment] = []
    all_before_changes: dict[int, tuple[tuple[int, int], ...]] = {}
    all_after_changes: dict[int, tuple[tuple[int, int], ...]] = {}
    for pair in _pair_source_regions(before_regions, after_regions):
        before_range = (pair.before.start, pair.before.end) if pair.before is not None else (0, 0)
        after_range = (pair.after.start, pair.after.end) if pair.after is not None else (0, 0)
        rows, inserted, deleted, before_changes, after_changes = _align_rows(
            before,
            after,
            before_range,
            after_range,
        )
        all_before_changes.update(before_changes)
        all_after_changes.update(after_changes)
        alignments.append(
            _SectionAlignment(
                pair=pair,
                rows=rows,
                inserted=inserted,
                deleted=deleted,
            )
        )

    before_html = _highlight_python(before.text, all_before_changes, "diff-delete")
    after_html = _highlight_python(after.text, all_after_changes, "diff-insert")
    return tuple(
        DiffSection(
            function_key=alignment.pair.function_key,
            function_name=alignment.pair.function_name,
            inserted=alignment.inserted,
            deleted=alignment.deleted,
            hunks=_fold_rows(_materialize_rows(alignment.rows, before_html, after_html), context),
        )
        for alignment in alignments
    )


def _fold_rows(rows: tuple[DiffRow, ...], context: int) -> tuple[DiffHunk, ...]:
    """Fold unchanged row runs while preserving the requested visible context."""
    hunks: list[DiffHunk] = []

    def add_hunk(hunk_rows: tuple[DiffRow, ...], collapsed: bool) -> None:
        if not hunk_rows:
            return
        if not collapsed and hunks and not hunks[-1].collapsed:
            previous = hunks.pop()
            hunks.append(DiffHunk(rows=previous.rows + hunk_rows, collapsed=False))
        else:
            hunks.append(DiffHunk(rows=hunk_rows, collapsed=collapsed))

    row_index = 0
    while row_index < len(rows):
        if rows[row_index].kind != "equal":
            add_hunk((rows[row_index],), collapsed=False)
            row_index += 1
            continue

        equal_end = row_index
        while equal_end < len(rows) and rows[equal_end].kind == "equal":
            equal_end += 1
        equal_rows = rows[row_index:equal_end]
        has_change_before = row_index > 0
        has_change_after = equal_end < len(rows)

        if not has_change_before and not has_change_after:
            add_hunk(equal_rows, collapsed=False)
        elif has_change_before and has_change_after and len(equal_rows) > 2 * context:
            leading = equal_rows[:context]
            collapsed = equal_rows[context : len(equal_rows) - context]
            trailing = equal_rows[len(equal_rows) - context :]
            if leading:
                add_hunk(leading, collapsed=False)
            add_hunk(collapsed, collapsed=True)
            if trailing:
                add_hunk(trailing, collapsed=False)
        elif has_change_before and not has_change_after and len(equal_rows) > context:
            leading = equal_rows[:context]
            collapsed = equal_rows[context:]
            if leading:
                add_hunk(leading, collapsed=False)
            add_hunk(collapsed, collapsed=True)
        elif not has_change_before and has_change_after and len(equal_rows) > context:
            collapsed = equal_rows[: len(equal_rows) - context]
            trailing = equal_rows[len(equal_rows) - context :]
            add_hunk(collapsed, collapsed=True)
            if trailing:
                add_hunk(trailing, collapsed=False)
        else:
            add_hunk(equal_rows, collapsed=False)
        row_index = equal_end
    return tuple(hunks)


def build_trace(snapshots: tuple[Snapshot, ...], context: int) -> tuple[PassTrace, ...]:
    """Build display traces for every consecutive pair of pass snapshots.

    Args:
        snapshots: Frontend and pass snapshots ordered by pass index.
        context: Number of unchanged lines retained next to a changed region.

    Returns:
        One trace per pass snapshot.

    Raises:
        IRTraceError: If ``context`` is negative.
    """
    if context < 0:
        raise IRTraceError(f"context must be non-negative, got {context}")

    traces: list[PassTrace] = []
    for before, after in zip(snapshots, snapshots[1:], strict=False):
        sections = _build_sections(before, after, context)
        inserted = sum(section.inserted for section in sections)
        deleted = sum(section.deleted for section in sections)
        traces.append(
            PassTrace(
                index=after.index,
                name=after.pass_name or "",
                before=before,
                after=after,
                inserted=inserted,
                deleted=deleted,
                sections=sections,
            )
        )
    return tuple(traces)
