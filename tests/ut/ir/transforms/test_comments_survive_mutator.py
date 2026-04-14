# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify IRMutator base-class rebuild paths preserve Stmt.leading_comments."""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _collect_leading(stmt: ir.Stmt) -> list[list[str]]:
    """Return leading_comments for every non-SeqStmts stmt, in DFS order."""
    out: list[list[str]] = []

    def _walk(s: ir.Stmt) -> None:
        if isinstance(s, ir.SeqStmts):
            for inner in s.stmts:
                _walk(inner)
            return
        out.append(list(s.leading_comments))
        if isinstance(s, ir.IfStmt):
            _walk(s.then_body)
            if s.else_body is not None:
                _walk(s.else_body)
        elif isinstance(s, (ir.ForStmt, ir.WhileStmt)):
            _walk(s.body)
        elif isinstance(s, ir.ScopeStmt):
            _walk(s.body)

    _walk(stmt)
    return out


class TestCommentsSurviveMutator:
    def test_base_mutator_rebuild_preserves_comments(self):
        """Comments on compound stmts rebuilt through IRMutator's base
        VisitStmt_ path (via MakeLikeStmt) must survive transformation.

        Documented v1 limitation: IRMutator *subclasses* that construct stmts
        directly (bypassing MakeLikeStmt) may drop comments. The ``convert_to_ssa``
        pass goes through the base path for ``ForStmt`` but not for ``AssignStmt``;
        this test only asserts the preservation that the base path guarantees.
        """

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                # loop annotation
                for i in pl.range(4):  # trailing
                    x = x + 1.0
                return x

        before = _collect_leading(list(P.functions.values())[0].body)
        assert ["loop annotation", "trailing"] in before

        P2 = passes.convert_to_ssa()(P)

        after = _collect_leading(list(P2.functions.values())[0].body)
        assert ["loop annotation", "trailing"] in after


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
