# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for FlattenSingleStmt pass.

This pass recursively flattens single-statement SeqStmts and OpStmts blocks.

Tests use IR Builder to create before/expected programs (SeqStmts and OpStmts
are not directly exposed in the Python DSL). Each test compares pass output
with expected IR via assert_structural_equal.
"""

from pypto import ir
from pypto.pypto_core import DataType, passes


def test_flatten_seqstmts_with_single_opstmts():
    """Test flattening SeqStmts([OpStmts([AssignStmt])]) to AssignStmt."""
    span = ir.Span.unknown()

    # Build Before IR: SeqStmts([OpStmts([assign])])
    x_before = ir.Var("x", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span)
    assign_before = ir.AssignStmt(
        ir.Var("result", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span),
        ir.Call(
            ir.get_op("tensor.add"),
            [x_before, ir.ConstFloat(1.0, DataType.FP32, span)],
            ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32),
            span,
        ),
        span,
    )
    Before = ir.Program(
        [
            ir.Function(
                "main",
                [x_before],
                [ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32)],
                ir.SeqStmts([ir.OpStmts([assign_before], span)], span),
                span,
            )
        ],
        "test",
        span,
    )

    # Build Expected IR: just assign (flattened)
    x_expected = ir.Var("x", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span)
    assign_expected = ir.AssignStmt(
        ir.Var("result", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span),
        ir.Call(
            ir.get_op("tensor.add"),
            [x_expected, ir.ConstFloat(1.0, DataType.FP32, span)],
            ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32),
            span,
        ),
        span,
    )
    Expected = ir.Program(
        [
            ir.Function(
                "main",
                [x_expected],
                [ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32)],
                assign_expected,
                span,
            )
        ],
        "test",
        span,
    )

    # Apply pass and compare
    After = passes.flatten_single_stmt()(Before)
    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


def test_flatten_opstmts_with_single_assign():
    """Test flattening OpStmts([AssignStmt]) to AssignStmt."""
    span = ir.Span.unknown()

    # Build Before IR: OpStmts([assign])
    x_before = ir.Var("x", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span)
    assign_before = ir.AssignStmt(
        ir.Var("result", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span),
        ir.Call(
            ir.get_op("tensor.add"),
            [x_before, ir.ConstFloat(1.0, DataType.FP32, span)],
            ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32),
            span,
        ),
        span,
    )
    Before = ir.Program(
        [
            ir.Function(
                "main",
                [x_before],
                [ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32)],
                ir.OpStmts([assign_before], span),
                span,
            )
        ],
        "test",
        span,
    )

    # Build Expected IR: just assign (flattened)
    x_expected = ir.Var("x", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span)
    assign_expected = ir.AssignStmt(
        ir.Var("result", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span),
        ir.Call(
            ir.get_op("tensor.add"),
            [x_expected, ir.ConstFloat(1.0, DataType.FP32, span)],
            ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32),
            span,
        ),
        span,
    )
    Expected = ir.Program(
        [
            ir.Function(
                "main",
                [x_expected],
                [ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32)],
                assign_expected,
                span,
            )
        ],
        "test",
        span,
    )

    # Apply pass and compare
    After = passes.flatten_single_stmt()(Before)
    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


def test_no_flatten_multi_stmt_opstmts():
    """Test that OpStmts with multiple statements is not flattened."""
    span = ir.Span.unknown()

    # Build Before IR: OpStmts([assign1, assign2])
    x_before = ir.Var("x", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span)
    a_before = ir.Var("a", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span)
    b_before = ir.Var("b", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span)
    assign1_before = ir.AssignStmt(
        a_before,
        ir.Call(
            ir.get_op("tensor.add"),
            [x_before, ir.ConstFloat(1.0, DataType.FP32, span)],
            ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32),
            span,
        ),
        span,
    )
    assign2_before = ir.AssignStmt(
        b_before,
        ir.Call(
            ir.get_op("tensor.mul"),
            [a_before, ir.ConstFloat(2.0, DataType.FP32, span)],
            ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32),
            span,
        ),
        span,
    )
    Before = ir.Program(
        [
            ir.Function(
                "main",
                [x_before],
                [ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32)],
                ir.OpStmts([assign1_before, assign2_before], span),
                span,
            )
        ],
        "test",
        span,
    )

    # Build Expected IR: OpStmts([assign1, assign2]) - unchanged
    x_expected = ir.Var("x", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span)
    a_expected = ir.Var("a", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span)
    b_expected = ir.Var("b", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span)
    assign1_expected = ir.AssignStmt(
        a_expected,
        ir.Call(
            ir.get_op("tensor.add"),
            [x_expected, ir.ConstFloat(1.0, DataType.FP32, span)],
            ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32),
            span,
        ),
        span,
    )
    assign2_expected = ir.AssignStmt(
        b_expected,
        ir.Call(
            ir.get_op("tensor.mul"),
            [a_expected, ir.ConstFloat(2.0, DataType.FP32, span)],
            ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32),
            span,
        ),
        span,
    )
    Expected = ir.Program(
        [
            ir.Function(
                "main",
                [x_expected],
                [ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32)],
                ir.OpStmts([assign1_expected, assign2_expected], span),
                span,
            )
        ],
        "test",
        span,
    )

    # Apply pass and compare
    After = passes.flatten_single_stmt()(Before)
    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


def test_recursive_flattening():
    """Test recursive flattening of deeply nested single statements."""
    span = ir.Span.unknown()

    # Build Before IR: SeqStmts([SeqStmts([OpStmts([assign])])])
    x_before = ir.Var("x", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span)
    assign_before = ir.AssignStmt(
        ir.Var("result", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span),
        ir.Call(
            ir.get_op("tensor.add"),
            [x_before, ir.ConstFloat(1.0, DataType.FP32, span)],
            ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32),
            span,
        ),
        span,
    )
    Before = ir.Program(
        [
            ir.Function(
                "main",
                [x_before],
                [ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32)],
                ir.SeqStmts(
                    [ir.SeqStmts([ir.OpStmts([assign_before], span)], span)],
                    span,
                ),
                span,
            )
        ],
        "test",
        span,
    )

    # Build Expected IR: just assign (recursively flattened)
    x_expected = ir.Var("x", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span)
    assign_expected = ir.AssignStmt(
        ir.Var("result", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span),
        ir.Call(
            ir.get_op("tensor.add"),
            [x_expected, ir.ConstFloat(1.0, DataType.FP32, span)],
            ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32),
            span,
        ),
        span,
    )
    Expected = ir.Program(
        [
            ir.Function(
                "main",
                [x_expected],
                [ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32)],
                assign_expected,
                span,
            )
        ],
        "test",
        span,
    )

    # Apply pass and compare
    After = passes.flatten_single_stmt()(Before)
    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


def test_idempotence():
    """Test that applying flatten twice gives the same result."""
    span = ir.Span.unknown()

    # Build Before IR: SeqStmts([OpStmts([assign])])
    x_before = ir.Var("x", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span)
    assign_before = ir.AssignStmt(
        ir.Var("result", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span),
        ir.Call(
            ir.get_op("tensor.add"),
            [x_before, ir.ConstFloat(1.0, DataType.FP32, span)],
            ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32),
            span,
        ),
        span,
    )
    Before = ir.Program(
        [
            ir.Function(
                "main",
                [x_before],
                [ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32)],
                ir.SeqStmts([ir.OpStmts([assign_before], span)], span),
                span,
            )
        ],
        "test",
        span,
    )

    # Build Expected IR: just assign (flattened)
    x_expected = ir.Var("x", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span)
    assign_expected = ir.AssignStmt(
        ir.Var("result", ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32), span),
        ir.Call(
            ir.get_op("tensor.add"),
            [x_expected, ir.ConstFloat(1.0, DataType.FP32, span)],
            ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32),
            span,
        ),
        span,
    )
    Expected = ir.Program(
        [
            ir.Function(
                "main",
                [x_expected],
                [ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32)],
                assign_expected,
                span,
            )
        ],
        "test",
        span,
    )

    # Apply pass once and compare
    After = passes.flatten_single_stmt()(Before)
    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    # Apply pass again and verify idempotence
    After2 = passes.flatten_single_stmt()(After)
    ir.assert_structural_equal(After2, Expected, enable_auto_mapping=True)
