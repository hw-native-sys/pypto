# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for SeqStmts class."""

import pytest
from pypto import DataType, ir


class TestSeqStmts:
    """Test SeqStmts class."""

    def test_seq_stmts_creation(self):
        """Test creating a SeqStmts instance."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, ir.ConstInt(0, dtype, span), span)
        seq_stmts = ir.SeqStmts([assign1, assign2], span)

        assert seq_stmts is not None
        assert seq_stmts.span.filename == "test.py"
        assert len(seq_stmts.stmts) == 2

    def test_seq_stmts_has_attributes(self):
        """Test that SeqStmts has stmts attribute."""
        span = ir.Span("test.py", 10, 5, 10, 15)
        dtype = DataType.INT64
        a = ir.Var("a", ir.ScalarType(dtype), span)
        b = ir.Var("b", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(a, b, span)
        assign2 = ir.AssignStmt(b, a, span)
        seq_stmts = ir.SeqStmts([assign1, assign2], span)

        assert seq_stmts.stmts is not None
        assert len(seq_stmts.stmts) == 2
        assert isinstance(seq_stmts.stmts[0], ir.AssignStmt)
        assert isinstance(seq_stmts.stmts[1], ir.AssignStmt)

    def test_seq_stmts_is_stmt(self):
        """Test that SeqStmts is an instance of Stmt."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        seq_stmts = ir.SeqStmts([assign], span)

        assert isinstance(seq_stmts, ir.Stmt)
        assert isinstance(seq_stmts, ir.IRNode)

    def test_seq_stmts_immutability(self):
        """Test that SeqStmts attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        seq_stmts = ir.SeqStmts([assign], span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            seq_stmts.stmts = []  # type: ignore

    def test_seq_stmts_with_empty_list(self):
        """Test SeqStmts with empty statement list."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        seq_stmts = ir.SeqStmts([], span)

        assert len(seq_stmts.stmts) == 0

    def test_seq_stmts_with_single_stmt(self):
        """Test SeqStmts with single statement."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)
        seq_stmts = ir.SeqStmts([assign], span)

        assert len(seq_stmts.stmts) == 1
        assert isinstance(seq_stmts.stmts[0], ir.AssignStmt)

    def test_seq_stmts_with_multiple_stmts(self):
        """Test SeqStmts with multiple statements."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, z, span)
        assign3 = ir.AssignStmt(z, ir.ConstInt(0, dtype, span), span)
        seq_stmts = ir.SeqStmts([assign1, assign2, assign3], span)

        assert len(seq_stmts.stmts) == 3
        assert isinstance(seq_stmts.stmts[0], ir.AssignStmt)
        assert isinstance(seq_stmts.stmts[1], ir.AssignStmt)
        assert isinstance(seq_stmts.stmts[2], ir.AssignStmt)

    def test_seq_stmts_string_representation(self):
        """Test SeqStmts string representation."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, ir.ConstInt(0, dtype, span), span)

        # Single statement
        seq_stmts1 = ir.SeqStmts([assign1], span)
        str_repr1 = str(seq_stmts1)
        assert isinstance(str_repr1, str)

        # Multiple statements
        seq_stmts2 = ir.SeqStmts([assign1, assign2], span)
        str_repr2 = str(seq_stmts2)
        assert isinstance(str_repr2, str)


class TestSeqStmtsHash:
    """Tests for SeqStmts hash function."""

    def test_seq_stmts_same_structure_hash(self):
        """Test SeqStmts nodes with same structure hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        assign1_1 = ir.AssignStmt(x1, y1, span)
        assign1_2 = ir.AssignStmt(y1, ir.ConstInt(0, dtype, span), span)
        seq_stmts1 = ir.SeqStmts([assign1_1, assign1_2], span)

        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        assign2_1 = ir.AssignStmt(x2, y2, span)
        assign2_2 = ir.AssignStmt(y2, ir.ConstInt(0, dtype, span), span)
        seq_stmts2 = ir.SeqStmts([assign2_1, assign2_2], span)

        hash1 = ir.structural_hash(seq_stmts1, enable_auto_mapping=True)
        hash2 = ir.structural_hash(seq_stmts2, enable_auto_mapping=True)
        assert hash1 == hash2

    def test_seq_stmts_different_statements_hash(self):
        """Test SeqStmts nodes with different statements hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, ir.ConstInt(0, dtype, span), span)
        assign3 = ir.AssignStmt(x, ir.ConstInt(1, dtype, span), span)

        seq_stmts1 = ir.SeqStmts([assign1, assign2], span)
        seq_stmts2 = ir.SeqStmts([assign1, assign3], span)

        hash1 = ir.structural_hash(seq_stmts1)
        hash2 = ir.structural_hash(seq_stmts2)
        assert hash1 != hash2

    def test_seq_stmts_different_length_hash(self):
        """Test SeqStmts nodes with different lengths hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, ir.ConstInt(0, dtype, span), span)

        seq_stmts1 = ir.SeqStmts([assign1], span)
        seq_stmts2 = ir.SeqStmts([assign1, assign2], span)

        hash1 = ir.structural_hash(seq_stmts1)
        hash2 = ir.structural_hash(seq_stmts2)
        assert hash1 != hash2

    def test_seq_stmts_empty_hash(self):
        """Test empty SeqStmts hash."""
        span = ir.Span.unknown()
        seq_stmts1 = ir.SeqStmts([], span)
        seq_stmts2 = ir.SeqStmts([], span)

        hash1 = ir.structural_hash(seq_stmts1)
        hash2 = ir.structural_hash(seq_stmts2)
        assert hash1 == hash2


class TestSeqStmtsStructuralEqual:
    """Tests for SeqStmts structural equality function."""

    def test_seq_stmts_structural_equal(self):
        """Test SeqStmts nodes with same structure are equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        assign1_1 = ir.AssignStmt(x1, y1, span)
        assign1_2 = ir.AssignStmt(y1, ir.ConstInt(0, dtype, span), span)
        seq_stmts1 = ir.SeqStmts([assign1_1, assign1_2], span)

        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        assign2_1 = ir.AssignStmt(x2, y2, span)
        assign2_2 = ir.AssignStmt(y2, ir.ConstInt(0, dtype, span), span)
        seq_stmts2 = ir.SeqStmts([assign2_1, assign2_2], span)

        ir.assert_structural_equal(seq_stmts1, seq_stmts2, enable_auto_mapping=True)

    def test_seq_stmts_structural_equal_different_statements(self):
        """Test SeqStmts nodes with different statements are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, ir.ConstInt(0, dtype, span), span)
        assign3 = ir.AssignStmt(x, ir.ConstInt(1, dtype, span), span)

        seq_stmts1 = ir.SeqStmts([assign1, assign2], span)
        seq_stmts2 = ir.SeqStmts([assign1, assign3], span)

        assert not ir.structural_equal(seq_stmts1, seq_stmts2)

    def test_seq_stmts_structural_equal_different_length(self):
        """Test SeqStmts nodes with different lengths are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, ir.ConstInt(0, dtype, span), span)

        seq_stmts1 = ir.SeqStmts([assign1], span)
        seq_stmts2 = ir.SeqStmts([assign1, assign2], span)

        assert not ir.structural_equal(seq_stmts1, seq_stmts2)

    def test_seq_stmts_structural_equal_empty(self):
        """Test empty SeqStmts are equal."""
        span = ir.Span.unknown()
        seq_stmts1 = ir.SeqStmts([], span)
        seq_stmts2 = ir.SeqStmts([], span)

        ir.assert_structural_equal(seq_stmts1, seq_stmts2)

    def test_seq_stmts_structural_equal_multiple_statements(self):
        """Test SeqStmts with multiple statements."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        z1 = ir.Var("z", ir.ScalarType(dtype), span)
        assign1_1 = ir.AssignStmt(x1, y1, span)
        assign1_2 = ir.AssignStmt(y1, z1, span)
        assign1_3 = ir.AssignStmt(z1, ir.ConstInt(0, dtype, span), span)
        seq_stmts1 = ir.SeqStmts([assign1_1, assign1_2, assign1_3], span)

        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        z2 = ir.Var("z", ir.ScalarType(dtype), span)
        assign2_1 = ir.AssignStmt(x2, y2, span)
        assign2_2 = ir.AssignStmt(y2, z2, span)
        assign2_3 = ir.AssignStmt(z2, ir.ConstInt(0, dtype, span), span)
        seq_stmts2 = ir.SeqStmts([assign2_1, assign2_2, assign2_3], span)

        ir.assert_structural_equal(seq_stmts1, seq_stmts2, enable_auto_mapping=True)
