# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for Stmt base class."""

import pytest
from pypto import ir


class TestStmt:
    """Test Stmt base class."""

    def test_stmt_creation(self):
        """Test creating a Stmt instance."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        stmt = ir.Stmt(span)
        assert stmt is not None
        assert stmt.span.filename == "test.py"

    def test_stmt_has_span(self):
        """Test that Stmt has span attribute."""
        span = ir.Span("test.py", 10, 5, 10, 15)
        stmt = ir.Stmt(span)
        assert stmt.span.begin_line == 10
        assert stmt.span.begin_column == 5

    def test_stmt_is_irnode(self):
        """Test that Stmt is an instance of IRNode."""
        span = ir.Span.unknown()
        stmt = ir.Stmt(span)
        assert isinstance(stmt, ir.IRNode)

    def test_stmt_immutability(self):
        """Test that Stmt attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        stmt = ir.Stmt(span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            stmt.span = ir.Span("other.py", 2, 2, 2, 5)  # type: ignore

    def test_stmt_with_unknown_span(self):
        """Test creating Stmt with unknown span."""
        span = ir.Span.unknown()
        stmt = ir.Stmt(span)
        assert stmt.span.is_valid() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
