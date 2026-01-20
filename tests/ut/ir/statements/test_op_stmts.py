# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for OpStmts class."""

import pytest
from pypto import DataType, ir


class TestOpStmts:
    """Test OpStmts class."""

    def test_op_stmts_creation(self):
        """Test creating an OpStmts instance."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        op_stmts = ir.OpStmts([assign], span)

        assert op_stmts is not None
        assert op_stmts.span.filename == "test.py"
        assert len(op_stmts.stmts) == 1
        assert isinstance(op_stmts.stmts[0], ir.AssignStmt)

    def test_op_stmts_has_attributes(self):
        """Test that OpStmts has stmts attribute."""
        span = ir.Span("test.py", 10, 5, 10, 15)
        dtype = DataType.INT64
        a = ir.Var("a", ir.ScalarType(dtype), span)
        b = ir.Var("b", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(a, b, span)
        assign2 = ir.AssignStmt(b, a, span)
        op_stmts = ir.OpStmts([assign1, assign2], span)

        assert op_stmts.stmts is not None
        assert len(op_stmts.stmts) == 2
        assert isinstance(op_stmts.stmts[0], ir.AssignStmt)
        assert isinstance(op_stmts.stmts[1], ir.AssignStmt)

    def test_op_stmts_is_stmt(self):
        """Test that OpStmts is an instance of Stmt."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        op_stmts = ir.OpStmts([assign], span)

        assert isinstance(op_stmts, ir.Stmt)
        assert isinstance(op_stmts, ir.IRNode)

    def test_op_stmts_immutability(self):
        """Test that OpStmts attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        op_stmts = ir.OpStmts([assign], span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            op_stmts.stmts = []  # type: ignore

    def test_op_stmts_with_empty_list(self):
        """Test OpStmts with empty statement list."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        op_stmts = ir.OpStmts([], span)

        assert len(op_stmts.stmts) == 0

    def test_op_stmts_with_multiple_statements(self):
        """Test OpStmts with multiple statements."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, z, span)
        assign3 = ir.AssignStmt(z, x, span)
        op_stmts = ir.OpStmts([assign1, assign2, assign3], span)

        assert len(op_stmts.stmts) == 3
        assert isinstance(op_stmts.stmts[0], ir.AssignStmt)
        assert isinstance(op_stmts.stmts[1], ir.AssignStmt)
        assert isinstance(op_stmts.stmts[2], ir.AssignStmt)

    def test_op_stmts_only_accepts_assign_stmts(self):
        """Test OpStmts only accepts AssignStmt types."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)

        # Test with AssignStmt (should work)
        assign = ir.AssignStmt(x, y, span)
        op_stmts = ir.OpStmts([assign], span)
        assert isinstance(op_stmts.stmts[0], ir.AssignStmt)

        # Test with multiple AssignStmts
        assign2 = ir.AssignStmt(y, ir.ConstInt(0, dtype, span), span)
        op_stmts2 = ir.OpStmts([assign, assign2], span)
        assert len(op_stmts2.stmts) == 2
        assert isinstance(op_stmts2.stmts[0], ir.AssignStmt)
        assert isinstance(op_stmts2.stmts[1], ir.AssignStmt)


class TestOpStmtsPrinting:
    """Test printing of OpStmts statements."""

    def test_op_stmts_printing_single(self):
        """Test printing of OpStmts with single statement."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        op_stmts = ir.OpStmts([assign], span)
        assert str(op_stmts) == "x: pi.Int64 = y"

    def test_op_stmts_printing_multiple(self):
        """Test printing of OpStmts with multiple statements."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, z, span)
        assign3 = ir.AssignStmt(z, x, span)
        op_stmts = ir.OpStmts([assign1, assign2, assign3], span)
        assert str(op_stmts) == "x: pi.Int64 = y\ny: pi.Int64 = z\nz: pi.Int64 = x"

    def test_op_stmts_printing_empty(self):
        """Test printing of OpStmts with empty statement list."""
        span = ir.Span.unknown()
        op_stmts = ir.OpStmts([], span)
        assert str(op_stmts) == ""

    def test_op_stmts_printing_multiple_assigns(self):
        """Test printing of OpStmts with multiple assignment statements."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, z, span)
        assign3 = ir.AssignStmt(z, ir.ConstInt(0, dtype, span), span)
        op_stmts = ir.OpStmts([assign1, assign2, assign3], span)
        assert str(op_stmts) == "x: pi.Int64 = y\ny: pi.Int64 = z\nz: pi.Int64 = 0"


class TestOpStmtsHash:
    """Tests for OpStmts hash function."""

    def test_op_stmts_same_structure_hash(self):
        """Test OpStmts nodes with same structure hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x1, y1, span)
        op_stmts1 = ir.OpStmts([assign1], span)

        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        assign2 = ir.AssignStmt(x2, y2, span)
        op_stmts2 = ir.OpStmts([assign2], span)

        hash1 = ir.structural_hash(op_stmts1)
        hash2 = ir.structural_hash(op_stmts2)
        # Different variable pointers result in different hashes without auto_mapping
        assert hash1 != hash2

    def test_op_stmts_different_statements_hash(self):
        """Test OpStmts nodes with different statements hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(x, z, span)

        op_stmts1 = ir.OpStmts([assign1], span)
        op_stmts2 = ir.OpStmts([assign2], span)

        hash1 = ir.structural_hash(op_stmts1)
        hash2 = ir.structural_hash(op_stmts2)
        assert hash1 != hash2

    def test_op_stmts_different_length_hash(self):
        """Test OpStmts nodes with different lengths hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, x, span)

        op_stmts1 = ir.OpStmts([assign1], span)
        op_stmts2 = ir.OpStmts([assign1, assign2], span)

        hash1 = ir.structural_hash(op_stmts1)
        hash2 = ir.structural_hash(op_stmts2)
        assert hash1 != hash2

    def test_op_stmts_empty_hash(self):
        """Test OpStmts nodes with empty list hash."""
        span = ir.Span.unknown()
        op_stmts1 = ir.OpStmts([], span)
        op_stmts2 = ir.OpStmts([], span)

        hash1 = ir.structural_hash(op_stmts1)
        hash2 = ir.structural_hash(op_stmts2)
        assert hash1 == hash2


class TestOpStmtsEquality:
    """Tests for OpStmts structural equality function."""

    def test_op_stmts_structural_equal(self):
        """Test structural equality of OpStmts nodes."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x1, y1, span)
        op_stmts1 = ir.OpStmts([assign1], span)

        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        assign2 = ir.AssignStmt(x2, y2, span)
        op_stmts2 = ir.OpStmts([assign2], span)

        # Without auto_mapping, different variable objects are not equal
        assert not ir.structural_equal(op_stmts1, op_stmts2)

        # With auto_mapping, same structure should be equal
        assert ir.structural_equal(op_stmts1, op_stmts2, enable_auto_mapping=True)

    def test_op_stmts_structural_equal_different_statements(self):
        """Test structural equality of OpStmts with different statements."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(x, z, span)

        op_stmts1 = ir.OpStmts([assign1], span)
        op_stmts2 = ir.OpStmts([assign2], span)

        assert not ir.structural_equal(op_stmts1, op_stmts2)
        assert ir.structural_equal(op_stmts1, op_stmts2, enable_auto_mapping=True)

    def test_op_stmts_structural_equal_different_length(self):
        """Test structural equality of OpStmts with different lengths."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, x, span)

        op_stmts1 = ir.OpStmts([assign1], span)
        op_stmts2 = ir.OpStmts([assign1, assign2], span)

        assert not ir.structural_equal(op_stmts1, op_stmts2)
        assert not ir.structural_equal(op_stmts1, op_stmts2, enable_auto_mapping=True)

    def test_op_stmts_structural_equal_empty(self):
        """Test structural equality of OpStmts with empty lists."""
        span = ir.Span.unknown()
        op_stmts1 = ir.OpStmts([], span)
        op_stmts2 = ir.OpStmts([], span)

        assert ir.structural_equal(op_stmts1, op_stmts2)
        assert ir.structural_equal(op_stmts1, op_stmts2, enable_auto_mapping=True)

    def test_op_stmts_structural_equal_multiple_statements(self):
        """Test structural equality of OpStmts with multiple statements."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        z1 = ir.Var("z", ir.ScalarType(dtype), span)
        assign1_1 = ir.AssignStmt(x1, y1, span)
        assign2_1 = ir.AssignStmt(y1, z1, span)
        op_stmts1 = ir.OpStmts([assign1_1, assign2_1], span)

        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        z2 = ir.Var("z", ir.ScalarType(dtype), span)
        assign1_2 = ir.AssignStmt(x2, y2, span)
        assign2_2 = ir.AssignStmt(y2, z2, span)
        op_stmts2 = ir.OpStmts([assign1_2, assign2_2], span)

        # Without auto_mapping, different variable objects are not equal
        assert not ir.structural_equal(op_stmts1, op_stmts2)

        # With auto_mapping, same structure should be equal
        assert ir.structural_equal(op_stmts1, op_stmts2, enable_auto_mapping=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
