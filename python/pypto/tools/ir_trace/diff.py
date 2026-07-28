# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build safely highlighted, foldable diffs between IR pass snapshots."""

import difflib
import html
import io
import keyword
import token
import tokenize

from .model import DiffHunk, DiffRow, IRTraceError, PassTrace, Snapshot, split_source_lines

_TOKEN_CLASSES = {
    token.STRING: "tok-string",
    token.NUMBER: "tok-number",
    token.COMMENT: "tok-comment",
    token.OP: "tok-operator",
}


def _escape_html(text: str, *, quote: bool = False) -> str:
    """Escape HTML-sensitive text and JavaScript line separators."""
    return html.escape(text, quote=quote).replace("\u2028", "&#x2028;").replace("\u2029", "&#x2029;")


def highlight_python(text: str) -> tuple[str, ...]:
    """Return one safely escaped HTML fragment for each Python source line.

    Args:
        text: Python source text to highlight.

    Returns:
        Escaped per-line HTML fragments. Invalid source falls back to escaped
        plain text so rendering remains safe.
    """
    lines = split_source_lines(text)
    escaped_lines = tuple(_escape_html(line, quote=True) for line in lines)
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
    except (tokenize.TokenError, SyntaxError, ValueError):
        return escaped_lines

    highlighted: list[str] = []
    for line, line_spans in zip(lines, spans, strict=True):
        current_column = 0
        fragments: list[str] = []
        for span_start, span_end, css_class in line_spans:
            if span_start < current_column:
                return escaped_lines
            fragments.append(_escape_html(line[current_column:span_start]))
            fragments.append(f'<span class="{css_class}">{_escape_html(line[span_start:span_end])}</span>')
            current_column = span_end
        fragments.append(_escape_html(line[current_column:]))
        highlighted.append("".join(fragments))
    return tuple(highlighted)


def _diff_rows(before: Snapshot, after: Snapshot) -> tuple[tuple[DiffRow, ...], int, int]:
    """Align two snapshots into display rows and count inserted/deleted lines."""
    before_html = highlight_python(before.text)
    after_html = highlight_python(after.text)
    rows: list[DiffRow] = []
    inserted = 0
    deleted = 0
    matcher = difflib.SequenceMatcher(a=before.lines, b=after.lines, autojunk=False)

    for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        before_count = before_end - before_start
        after_count = after_end - after_start
        if tag == "insert":
            inserted += after_count
        elif tag == "delete":
            deleted += before_count
        elif tag == "replace":
            inserted += after_count
            deleted += before_count

        row_count = max(before_count, after_count)
        for offset in range(row_count):
            before_index = before_start + offset if offset < before_count else None
            after_index = after_start + offset if offset < after_count else None
            rows.append(
                DiffRow(
                    kind=tag,
                    before_number=before_index + 1 if before_index is not None else None,
                    before_html=before_html[before_index] if before_index is not None else "",
                    after_number=after_index + 1 if after_index is not None else None,
                    after_html=after_html[after_index] if after_index is not None else "",
                )
            )
    return tuple(rows), inserted, deleted


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
        rows, inserted, deleted = _diff_rows(before, after)
        traces.append(
            PassTrace(
                index=after.index,
                name=after.pass_name or "",
                before=before,
                after=after,
                inserted=inserted,
                deleted=deleted,
                hunks=_fold_rows(rows, context),
            )
        )
    return tuple(traces)
