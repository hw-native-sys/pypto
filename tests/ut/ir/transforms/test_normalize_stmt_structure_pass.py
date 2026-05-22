# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for NormalizeStmtStructure pass.

This pass normalizes IR structure by:
1. Unwrapping single-child SeqStmts (no redundant nesting)
2. Preventing nested SeqStmts (SeqStmts as child of SeqStmts)

The DSL parser builds function bodies through ``IRBuilder``, whose
``EndFunction``/loop/if scopes already emit ``stmts[0]`` for a single
statement and a flat ``SeqStmts`` for multiple. So a normal ``@pl.program``
can express both pass-input shapes (a bare single-statement body and a flat
multi-statement ``SeqStmts``) directly, and the transform tests below use the
Before/Expected pattern. The pass is a no-op on these already-normalized
shapes, so each compares ``After`` with ``Before``.

The final test exercises a structurally-invalid mid-body ``YieldStmt``, which
the DSL parser cannot emit (yields only appear at trailing positions of
iter-arg scopes); it is built via raw ``ir.*`` constructors on purpose.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes


def test_normalize_simple_function():
    """A function body that is a single bare statement is left untouched.

    The DSL parser emits a bare ``ReturnStmt`` (not a single-child
    ``SeqStmts``) for a return-only body, so the pass has nothing to unwrap.
    """

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return pl.tensor.adds(x, 1.0)

    After = passes.normalize_stmt_structure()(Before)
    ir.assert_structural_equal(After, Before)


def test_normalize_seqstmts_with_bare_assigns():
    """A flat SeqStmts of bare statements is left untouched.

    A multi-statement function body parses to a flat ``SeqStmts`` (no nesting,
    no single-child wrapping), so the pass leaves it unchanged.
    """

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            a = pl.tensor.adds(x, 1.0)
            b = pl.tensor.muls(a, 2.0)
            return b

    After = passes.normalize_stmt_structure()(Before)
    ir.assert_structural_equal(After, Before)


def test_idempotence():
    """Applying normalize twice gives the same result."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return pl.tensor.adds(x, 1.0)

    After = passes.normalize_stmt_structure()(Before)
    ir.assert_structural_equal(After, Before)

    # Apply pass again and verify idempotence.
    After2 = passes.normalize_stmt_structure()(After)
    ir.assert_structural_equal(After2, Before)


def test_no_redundant_blocks_rejects_mid_body_yield():
    """NoRedundantBlocks rejects a YieldStmt at a non-trailing position of a
    SeqStmts. Function body has no iter_args context, so SSAVerify's
    CheckNoMidBodyYield does not apply — only the structural verifier catches it.

    This input is built via raw ``ir.*`` constructors because the DSL parser
    cannot emit a non-trailing ``YieldStmt`` in a plain function body.
    """
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    dummy_var = ir.Var("dummy", ir.ScalarType(DataType.INT64), span)
    assign = ir.AssignStmt(dummy_var, a, span)
    mid_yield = ir.YieldStmt([a], span)
    ret = ir.ReturnStmt([a], span)
    func_body = ir.SeqStmts([assign, mid_yield, ret], span)
    func = ir.Function("main", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    verify_pass = passes.run_verifier()
    with pytest.raises(Exception, match="YieldStmt before the terminating position"):
        verify_pass(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
