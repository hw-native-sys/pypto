# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser round-trip tests for Stmt.leading_comments attached from DSL `#` and docstrings."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.comment_extractor import extract_line_comments


def _body_stmts(prog: ir.Program) -> list[ir.Stmt]:
    """Flatten the main function body for easy per-stmt inspection."""
    func = list(prog.functions.values())[0]
    body = func.body

    def _flatten(s: ir.Stmt) -> list[ir.Stmt]:
        if isinstance(s, ir.SeqStmts):
            out: list[ir.Stmt] = []
            for inner in s.stmts:
                out.extend(_flatten(inner))
            return out
        return [s]

    return _flatten(body)


class TestCommentExtractor:
    def test_basic_line_comment(self):
        src = "x = 1  # note\n"
        assert extract_line_comments(src) == {1: [(7, "note")]}

    def test_full_line_comment(self):
        src = "# first\n"
        assert extract_line_comments(src) == {1: [(0, "first")]}

    def test_no_space_after_hash(self):
        src = "#compact\n"
        assert extract_line_comments(src) == {1: [(0, "compact")]}

    def test_empty_source(self):
        assert extract_line_comments("") == {}

    def test_captures_column_offset(self):
        src = "for i in range(1):\n    x = 1\n    # indented tail\n"
        result = extract_line_comments(src)
        # The comment is at column 4 (body indent).
        assert result == {3: [(4, "indented tail")]}

    def test_multi_hash_preserves_extra_hashes(self):
        # Strip exactly one leading '#' so "## heading" re-emits as "# heading"
        # rather than collapsing to "heading".
        assert extract_line_comments("## heading\n") == {1: [(0, "# heading")]}
        assert extract_line_comments("###section\n") == {1: [(0, "##section")]}


class TestAttachInParsedProgram:
    def test_leading_comment(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                # annotate
                y = x
                return y

        stmts = _body_stmts(P)
        assert stmts[0].leading_comments == ["annotate"]

    def test_trailing_comment_promoted(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                y = x  # trailing
                return y

        stmts = _body_stmts(P)
        assert stmts[0].leading_comments == ["trailing"]

    def test_docstring_becomes_comment_on_next_stmt(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                """hello world"""
                y = x
                return y

        stmts = _body_stmts(P)
        assert stmts[0].leading_comments == ["hello world"]

    def test_mixed_docstring_and_hash_comment(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                """doc"""
                # hash
                y = x
                return y

        stmts = _body_stmts(P)
        assert stmts[0].leading_comments == ["doc", "hash"]

    def test_compound_for_loop(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                y = x
                # main loop
                for i in pl.range(16):  # tiles
                    # body
                    y = y + 1.0
                return y

        stmts = _body_stmts(P)
        # stmts: [AssignStmt y=x, ForStmt, ReturnStmt]
        for_stmt = next(s for s in stmts if isinstance(s, ir.ForStmt))
        assert for_stmt.leading_comments == ["main loop", "tiles"]
        # Body-inner comment lands on the first body stmt.
        body_stmts: list[ir.Stmt] = []

        def _flatten(s: ir.Stmt) -> None:
            if isinstance(s, ir.SeqStmts):
                for inner in s.stmts:
                    _flatten(inner)
            else:
                body_stmts.append(s)

        _flatten(for_stmt.body)
        assert body_stmts[0].leading_comments == ["body"]

    def test_if_else_branches(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                # check positivity
                if x > 0.0:  # positive
                    # print x
                    y = x
                # fallback
                else:
                    y = -x
                return y

        stmts = _body_stmts(P)
        if_stmt = next(s for s in stmts if isinstance(s, ir.IfStmt))
        assert if_stmt.leading_comments == ["check positivity", "positive"]

        # then body: first stmt gets "print x"
        def _first_of(block: ir.Stmt) -> ir.Stmt:
            if isinstance(block, ir.SeqStmts):
                return block.stmts[0]
            return block

        assert _first_of(if_stmt.then_body).leading_comments == ["print x"]
        # else body first stmt gets "fallback" (the comment above `else:` attaches to first else stmt)
        assert if_stmt.else_body is not None
        assert _first_of(if_stmt.else_body).leading_comments == ["fallback"]


class TestTailOfBlockWarning:
    """Tail-of-block comments are dropped with a UserWarning (documented v1 behavior)."""

    def test_tail_in_for_body_warns_and_drops(self):
        with pytest.warns(UserWarning, match="tail-of-block comment"):

            @pl.program
            class P:
                @pl.function
                def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                    for i in pl.range(4):
                        x = x + 1.0
                        # tail
                    return x

        # The tail comment must NOT attach to the outer return stmt.
        stmts = _body_stmts(P)
        ret = next(s for s in stmts if isinstance(s, ir.ReturnStmt))
        assert ret.leading_comments == []

    def test_tail_in_function_body_warns(self):
        with pytest.warns(UserWarning, match="tail-of-block comment"):

            @pl.program
            class P:  # noqa: F841
                @pl.function
                def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                    y = x
                    return y
                    # trailing after return

    def test_tail_in_if_then_branch_warns(self):
        with pytest.warns(UserWarning, match="tail-of-block comment"):

            @pl.program
            class P:  # noqa: F841
                @pl.function
                def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                    if x > 0.0:
                        y = x
                        # tail in then
                    else:
                        y = -x
                    return y


class TestSiblingBlockAttribution:
    """Leading comments inside a later sibling block must not be swept by the
    previous block's tail-drop (regression for codex P1)."""

    def test_sibling_for_loops_preserve_inner_leading(self):
        with pytest.warns(UserWarning, match="tail-of-block comment"):

            @pl.program
            class P:
                @pl.function
                def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                    for i in pl.range(4):
                        x = x + 1.0
                        # tail of first for
                    for j in pl.range(4):
                        # leading for y
                        y = x * 2.0
                    return y

        stmts = _body_stmts(P)
        # The second for-loop's first body stmt should carry the leading comment.
        for_stmts = [s for s in stmts if isinstance(s, ir.ForStmt)]
        assert len(for_stmts) == 2
        second_body = for_stmts[1].body
        first = second_body.stmts[0] if isinstance(second_body, ir.SeqStmts) else second_body
        assert "leading for y" in first.leading_comments
        # The "tail of first for" must NOT leak into the second loop's body.
        assert "tail of first for" not in first.leading_comments

    def test_sibling_if_blocks_preserve_inner_leading(self):
        with pytest.warns(UserWarning, match="tail-of-block comment"):

            @pl.program
            class P:
                @pl.function
                def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                    if x > 0.0:
                        y = x
                        # tail in first if
                    if x < 0.0:
                        # leading in second if
                        y = -x
                    return y

        stmts = _body_stmts(P)
        if_stmts = [s for s in stmts if isinstance(s, ir.IfStmt)]
        assert len(if_stmts) == 2
        second_then = if_stmts[1].then_body
        first = second_then.stmts[0] if isinstance(second_then, ir.SeqStmts) else second_then
        assert "leading in second if" in first.leading_comments
        assert "tail in first if" not in first.leading_comments


class TestWrappedHeaderComments:
    """Comments inside a wrapped multi-line header attach to the compound stmt,
    not to the first body stmt (regression for codex P2)."""

    def test_wrapped_for_header_comment_attaches_to_for(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                for i in pl.range(
                    4,
                    # wrap comment
                ):
                    x = x + 1.0
                return x

        stmts = _body_stmts(P)
        for_stmt = next(s for s in stmts if isinstance(s, ir.ForStmt))
        assert "wrap comment" in for_stmt.leading_comments
        # The wrap comment must NOT attach to the inner body stmt.
        body = for_stmt.body
        first = body.stmts[0] if isinstance(body, ir.SeqStmts) else body
        assert "wrap comment" not in first.leading_comments


class TestRoundTripIdempotency:
    def test_leading_and_trailing_roundtrip(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                # outer
                y = x  # trailing
                return y

        src1 = ir.python_print(P)
        P2 = pl.parse_program(src1)
        src2 = ir.python_print(P2)
        assert src1 == src2, f"not idempotent:\n--- first ---\n{src1}\n--- second ---\n{src2}"
        ir.assert_structural_equal(P, P2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
