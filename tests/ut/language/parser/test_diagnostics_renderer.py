# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the parser error renderer."""

import pytest
from pypto.language.parser.diagnostics import ParserSyntaxError
from pypto.language.parser.diagnostics.renderer import ErrorRenderer


@pytest.fixture
def renderer() -> ErrorRenderer:
    return ErrorRenderer(use_color=False)


def _make_error(message: str, line: str, column: int, hint: str | None = None) -> ParserSyntaxError:
    err = ParserSyntaxError(message, hint=hint)
    err.span = {"filename": "test.py", "begin_line": 1, "begin_column": column, "line": 1, "column": column}
    err.source_lines = [line]
    return err


class TestCaretTokenWidth:
    """Caret should span the full dotted identifier, not stop at the first '.'."""

    def test_dotted_callee(self, renderer: ErrorRenderer):
        assert renderer._calculate_token_length("pl.piepline(1, 2, 3)", 0) == len("pl.piepline")

    def test_multi_segment_dotted_identifier(self, renderer: ErrorRenderer):
        assert renderer._calculate_token_length("foo.bar.baz(x)", 0) == len("foo.bar.baz")

    def test_dotted_identifier_mid_line(self, renderer: ErrorRenderer):
        line = "x + pl.range(n)"
        assert renderer._calculate_token_length(line, 4) == len("pl.range")

    def test_plain_identifier_unaffected(self, renderer: ErrorRenderer):
        assert renderer._calculate_token_length("plain_name(x)", 0) == len("plain_name")

    def test_trailing_dot_does_not_extend(self, renderer: ErrorRenderer):
        # 'a.' has a dot with no identifier after — should not be absorbed.
        assert renderer._calculate_token_length("a.", 0) == 1

    def test_leading_dot_does_not_match(self, renderer: ErrorRenderer):
        # Starting at a bare '.' (no identifier before) falls through to the
        # minimum-1 behavior.
        assert renderer._calculate_token_length(".foo", 0) == 1


class TestInlineMessage:
    """Inline caret annotation must not split messages inside dotted identifiers."""

    def test_message_with_dotted_identifiers_is_not_truncated(self, renderer: ErrorRenderer):
        """The reported bug: full message is 87 chars (>= 50) so no inline
        annotation is emitted. The renderer must NOT fall back to the bogus
        'For loop must use pl' fragment produced by splitting on the first '.'.
        """
        message = "For loop must use pl.range(), pl.parallel(), pl.unroll(), pl.pipeline(), or pl.while_()"
        line = " " * 10 + "pl.piepline(x):"
        err = _make_error(message, line, column=10)
        rendered = renderer.render(err)

        assert "For loop must use pl\n" not in rendered
        assert "For loop must use pl\033" not in rendered  # any ANSI wrap
        assert "^" * len("pl.piepline") in rendered

    def test_short_sentence_still_renders_inline(self, renderer: ErrorRenderer):
        """For short messages, the first real sentence is still used."""
        message = "Variable x not defined. Expected an assignment first."
        line = "x + 1"
        err = _make_error(message, line, column=0)
        rendered = renderer.render(err)

        caret_row = next(row for row in rendered.split("\n") if "^" in row)
        assert "Variable x not defined" in caret_row
        assert "Expected an assignment first" not in caret_row

    def test_dotted_reference_within_message_preserved(self, renderer: ErrorRenderer):
        """A message whose only 'period' sits inside a dotted identifier
        should be emitted in full (when short enough), not truncated."""
        message = "Missing argument for module.attr call"
        line = "module.attr()"
        err = _make_error(message, line, column=0)
        rendered = renderer.render(err)

        assert "Missing argument for module.attr call" in rendered


class TestRealFileSnippet:
    """The snippet is read from the real file named by the span (issue #1612).

    When a span points at an on-disk file — e.g. a @pl.jit kernel span remapped
    to the user's source — the renderer shows that file's lines, not the
    (possibly generated) ``source_lines`` carried on the exception.
    """

    def test_snippet_read_from_real_file(self, renderer: ErrorRenderer, tmp_path):
        real = tmp_path / "kernel.py"
        real.write_text("line one\nt = pl.mul(t, t)  # the real source\nline three\n")

        err = ParserSyntaxError("type mismatch")
        err.span = {
            "filename": str(real),
            "begin_line": 2,
            "begin_column": 0,
            "line": 2,
            "column": 0,
        }
        # Deliberately wrong/generated source lines: linecache must take priority.
        err.source_lines = ["t_v1 = pl.mul(t_v1, t_v1)"]

        rendered = renderer.render(err)

        assert str(real) in rendered  # header points at the real file
        assert "the real source" in rendered  # snippet is the real file's line
        assert "t_v1" not in rendered  # not the generated source_lines

    def test_falls_back_to_source_lines_for_synthetic_filename(self, renderer: ErrorRenderer):
        """A synthetic filename (no on-disk file) falls back to source_lines."""
        err = ParserSyntaxError("type mismatch")
        err.span = {
            "filename": "<jit:kernel>",
            "begin_line": 1,
            "begin_column": 0,
            "line": 1,
            "column": 0,
        }
        err.source_lines = ["t_v1 = pl.mul(t_v1, t_v1)"]

        rendered = renderer.render(err)

        assert "t_v1 = pl.mul(t_v1, t_v1)" in rendered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
