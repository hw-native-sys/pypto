# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration coverage for state shared across parser components."""

import linecache

import pypto.language as pl
from pypto import ir


def test_inline_slice_and_op_share_closure_and_restore_source():
    """Inline indexing and ops see the callee closure and preserve both source locations."""

    def make_crop():
        rows = 8
        increment = 2.0

        @pl.inline
        def crop(x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[8, 16], pl.FP32]:
            sliced = x[:rows, :]
            adjusted = pl.add(sliced, increment)
            return adjusted

        return crop

    crop = make_crop()
    caller_rows = 4

    @pl.program
    class Actual:
        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[4, 16], pl.FP32]:
            cropped = crop(x)
            caller_slice = cropped[:caller_rows, :]
            return caller_slice

    @pl.program
    class Expected:
        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[4, 16], pl.FP32]:
            sliced = x[:8, :]
            adjusted = pl.add(sliced, 2.0)
            cropped = adjusted
            caller_slice = cropped[:4, :]
            return caller_slice

    ir.assert_structural_equal(Actual, Expected)
    func = next(iter(Actual.functions.values()))
    assert isinstance(func.body, ir.SeqStmts)
    assignments = {stmt.var.name_hint: stmt for stmt in func.body.stmts if isinstance(stmt, ir.AssignStmt)}
    for name, source in (
        ("sliced", "sliced = x[:rows, :]"),
        ("adjusted", "adjusted = pl.add(sliced, increment)"),
        ("caller_slice", "caller_slice = cropped[:caller_rows, :]"),
    ):
        stmt = assignments[name]
        for span in (stmt.span, stmt.value.span):
            assert linecache.getline(span.filename, span.begin_line).strip() == source
