# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Source comment ownership across statements, blocks and no-IR directives."""

import ast
import warnings
from typing import cast

from .span_tracker import SpanTracker


class SourceCommentTracker:
    """Source comment ownership across statements, blocks and no-IR directives."""

    def __init__(
        self,
        *,
        span_tracker: SpanTracker,
        pending_comments: dict[int, list[tuple[int, str]]] | None,
    ) -> None:
        self.span_tracker = span_tracker
        # Pending comments keyed by 1-based line number, drained by parse_statement.
        # Each entry is (col_offset, text) so the parser can distinguish
        # tail-of-block comments (inside body indent) from outer-scope comments.
        self.pending_comments: dict[int, list[tuple[int, str]]] = pending_comments or {}

    def then_branch_tail_upper(self, stmt: ast.If) -> int:
        """Inclusive upper-bound line for the then-branch's tail-drop range.

        Points to the line just before the ``else:`` keyword when an else
        clause is present (so the then-branch tail does not extend into
        else-body leading comments), and falls back to the if-stmt's end line
        when there is no else clause.
        """
        if not stmt.orelse:
            return stmt.end_lineno or stmt.lineno
        else_line = self.find_else_keyword_line(stmt)
        if else_line is not None:
            return else_line - 1
        return stmt.orelse[0].lineno - 1

    def find_else_keyword_line(self, stmt: ast.If) -> int | None:
        """Return the source line of the ``else:`` keyword for an if-stmt.

        Python's AST has no node for the ``else:`` keyword itself; its line can
        only be recovered by scanning the source text between the last then-body
        stmt and the first orelse stmt. Used to compute the exact upper bound of
        the then-branch for :meth:`SourceCommentTracker.discard_tail_block_comments` so that
        comments physically inside the else branch (e.g. leading comments for
        the first else stmt) are not mistaken for then-branch tails.

        Returns ``None`` if not found (e.g. ``elif`` chains lack an ``else:``
        keyword line).
        """
        if not stmt.orelse:
            return None
        src = self.span_tracker.source_lines
        start = (stmt.body[-1].end_lineno or stmt.body[-1].lineno) + 1
        end = stmt.orelse[0].lineno  # exclusive
        for line_no in range(start, end):
            idx = line_no - 1
            if not (0 <= idx < len(src)):
                continue
            stripped = src[idx].lstrip()
            # Match `else:` (optionally followed by whitespace/comment) but not
            # `elif ...` (which is a separate else-branch chain lowered into
            # ast.If.orelse[0] and has no `else:` keyword).
            if stripped.startswith("else") and stripped[4:5] in (":", " ", "\t"):
                return line_no
        return None

    @staticmethod
    def header_ast_end_line(stmt: ast.stmt) -> int:
        """Last AST line of a compound stmt's header, before the body's colon.

        For a wrapped multi-line header (e.g. ``for i in pl.range(\n  10,\n):``)
        this returns the line of the closing ``)``/``:``; for a simple
        single-line header it returns ``stmt.lineno``.

        Used to classify comments between ``stmt.lineno`` and ``body[0].lineno``:
        comments on a line ``<= header_ast_end`` belong to the header; comments
        on later lines are body-leading for ``body[0]``.
        """
        ends: list[int] = [stmt.lineno]
        if isinstance(stmt, ast.For):
            ends.append(stmt.target.end_lineno or stmt.target.lineno)
            ends.append(stmt.iter.end_lineno or stmt.iter.lineno)
        elif isinstance(stmt, (ast.While, ast.If)):
            ends.append(stmt.test.end_lineno or stmt.test.lineno)
        elif isinstance(stmt, ast.With):
            for item in stmt.items:
                ends.append(item.context_expr.end_lineno or item.context_expr.lineno)
                if item.optional_vars is not None:
                    ends.append(item.optional_vars.end_lineno or item.optional_vars.lineno)
        return max(ends)

    def drain_pending_comments(self, stmt: ast.stmt) -> list[str]:
        """Collect pending comments whose line numbers fall at or before ``stmt``.

        For simple stmts, drain through ``stmt.end_lineno`` so trailing comments
        on the last logical line (e.g. ``y = 1  # note``) attach to the stmt.

        For compound stmts (``for``/``while``/``if``/``with``) with a non-empty
        body, the header can span multiple physical lines. We drain up to the
        line just before the first body stmt and split by column: comments at
        a column *less than* the body's indent are header-level and attach to
        the compound stmt; comments at the body's indent are left pending for
        the first body stmt to pick up.

        Returns comments in source order and removes header/simple entries from
        the pending map.
        """
        if not self.pending_comments:
            return []
        body = getattr(stmt, "body", None)
        leading: list[str] = []
        if isinstance(body, list) and body:
            stmts = cast("list[ast.stmt]", body)
            header_end = stmts[0].lineno - 1
            header_ast_end = self.header_ast_end_line(stmt)
            body_col = stmts[0].col_offset
            for line in sorted(k for k in self.pending_comments if k <= header_end):
                if stmt.lineno <= line <= header_ast_end:
                    # Header-level comment: either an inline trailer on the
                    # header's first line (e.g. `for i in range(16):  # tiles`)
                    # or a continuation-line comment inside a wrapped multi-line
                    # header (e.g. `for i in pl.range(\n  10,  # comment\n):`).
                    # Attach to the compound stmt regardless of column.
                    leading.extend(text for _col, text in self.pending_comments.pop(line))
                    continue
                if line > header_ast_end:
                    # Past the header but before body[0] — this is a body-leading
                    # comment for body[0]. Leave pending for body[0] to pick up.
                    continue
                # line < stmt.lineno: pre-stmt leftover. Split by column —
                # shallower-than-body attaches to this compound stmt; same-or-
                # deeper stays pending (belongs to body).
                kept: list[tuple[int, str]] = []
                for col, text in self.pending_comments[line]:
                    if col < body_col:
                        leading.append(text)
                    else:
                        kept.append((col, text))
                if kept:
                    self.pending_comments[line] = kept
                else:
                    del self.pending_comments[line]
        else:
            end_line = stmt.end_lineno or stmt.lineno
            for line in sorted(k for k in self.pending_comments if k <= end_line):
                leading.extend(text for _col, text in self.pending_comments.pop(line))
        return leading

    def requeue_comments_after(self, stmt: ast.stmt, comments: list[str]) -> None:
        """Re-enqueue ``comments`` onto the line after ``stmt`` ends.

        Used when a stmt does not produce IR (docstring reroute, bare ``pass``)
        but may have collected leading comments that must survive to the next
        stmt. No-op when ``comments`` is empty. Synthetic comments inherit the
        stmt's column offset so they are treated as body-level for tail-drop
        purposes.
        """
        if not comments:
            return
        next_line = (stmt.end_lineno or stmt.lineno) + 1
        col = stmt.col_offset
        entries = [(col, text) for text in comments]
        existing = self.pending_comments.setdefault(next_line, [])
        self.pending_comments[next_line] = [*entries, *existing]

    def discard_tail_block_comments(self, body: list[ast.stmt], upper_line: int | None) -> None:
        """Drop pending tail comments inside a block's physical line range.

        Called after a block's body finishes parsing. Sweeps pending comments on
        lines in ``[body[0].lineno, upper_line]`` whose column is
        ``>= body[0].col_offset`` and drops them with a warning.

        ``upper_line`` must be the inclusive upper bound of the block's physical
        extent (typically the enclosing compound stmt's ``end_lineno``, or the
        line before the next sibling clause such as ``else:``). Passing ``None``
        means "no upper bound" — sweep all remaining pending comments from
        ``body_start`` onward; used for the function body (outermost scope).

        Bounding is necessary because ``_pending_comments`` is populated
        up-front from the entire source, so comments in yet-to-parse sibling
        blocks at the same indent would otherwise be swept in error.
        """
        if not body or not self.pending_comments:
            return
        block_col = body[0].col_offset
        block_start = body[0].lineno
        tail: list[tuple[int, str]] = []
        if upper_line is None:
            candidates = [k for k in self.pending_comments if k >= block_start]
        else:
            candidates = [k for k in self.pending_comments if block_start <= k <= upper_line]
        for line in sorted(candidates):
            remaining: list[tuple[int, str]] = []
            for col, text in self.pending_comments[line]:
                if col >= block_col:
                    tail.append((line, text))
                else:
                    remaining.append((col, text))
            if remaining:
                self.pending_comments[line] = remaining
            else:
                del self.pending_comments[line]
        self.emit_tail_warning(tail)

    def discard_stale_pending_before(self, stmt: ast.stmt) -> None:
        """Drop pending comments on earlier lines at deeper indent than ``stmt``.

        When parsing advances to ``stmt``, any pending comment on a line
        ``< stmt.lineno`` whose column is strictly greater than
        ``stmt.col_offset`` physically lives inside a previous sibling block
        whose body has already closed (dedent). Those comments cannot attach to
        ``stmt`` (wrong indent level) and would otherwise be misattributed by
        the simple-stmt drain, so sweep them here as tail-of-block.

        This is the mechanism that catches tail comments in the "gap" between
        one block's body and the next outer-scope stmt — complementing the
        bounded :meth:`SourceCommentTracker.discard_tail_block_comments` that runs at block exit.
        """
        if not self.pending_comments:
            return
        stale: list[tuple[int, str]] = []
        for line in sorted(k for k in self.pending_comments if k < stmt.lineno):
            remaining: list[tuple[int, str]] = []
            for col, text in self.pending_comments[line]:
                if col > stmt.col_offset:
                    stale.append((line, text))
                else:
                    remaining.append((col, text))
            if remaining:
                self.pending_comments[line] = remaining
            else:
                del self.pending_comments[line]
        self.emit_tail_warning(stale)

    @staticmethod
    def emit_tail_warning(tail: list[tuple[int, str]]) -> None:
        """Emit a UserWarning summarizing dropped tail-of-block comments."""
        if not tail:
            return
        preview = ", ".join(f"line {line}: {text!r}" for line, text in tail[:3])
        if len(tail) > 3:
            preview += f", … (+{len(tail) - 3} more)"
        warnings.warn(
            f"Dropped {len(tail)} tail-of-block comment(s): {preview}. "
            "Tail-of-block comments are not preserved in IR; move them above a "
            "statement or into the outer scope to retain them.",
            UserWarning,
            stacklevel=3,
        )
