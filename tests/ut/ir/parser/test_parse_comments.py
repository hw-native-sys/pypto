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
        assert extract_line_comments(src) == {1: ["note"]}

    def test_full_line_comment(self):
        src = "# first\n"
        assert extract_line_comments(src) == {1: ["first"]}

    def test_no_space_after_hash(self):
        src = "#compact\n"
        assert extract_line_comments(src) == {1: ["compact"]}

    def test_empty_source(self):
        assert extract_line_comments("") == {}


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
