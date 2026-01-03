# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Type stubs for PyPTO IR (Intermediate Representation) module."""

from typing import Final, List

class Span:
    """Source location information tracking file, line, and column positions."""

    filename: Final[str]
    """Source filename."""

    begin_line: Final[int]
    """Beginning line (1-indexed)."""

    begin_column: Final[int]
    """Beginning column (1-indexed)."""

    end_line: Final[int]
    """Ending line (1-indexed)."""

    end_column: Final[int]
    """Ending column (1-indexed)."""

    def __init__(
        self,
        filename: str,
        begin_line: int,
        begin_column: int,
        end_line: int = -1,
        end_column: int = -1,
    ) -> None:
        """Create a source span.

        Args:
            filename: Source filename
            begin_line: Beginning line (1-indexed)
            begin_column: Beginning column (1-indexed)
            end_line: Ending line (1-indexed, -1 means unknown)
            end_column: Ending column (1-indexed, -1 means unknown)
        """
        ...

    def to_string(self) -> str:
        """Convert span to string representation.

        Returns:
            String in format "filename:begin_line:begin_column"
        """
        ...

    def is_valid(self) -> bool:
        """Check if the span has valid coordinates.

        Returns:
            True if all line/column numbers are positive
        """
        ...

    @staticmethod
    def unknown() -> Span:
        """Create an unknown/invalid span for cases where source location is unavailable.

        Returns:
            Span with empty filename and invalid coordinates
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Op:
    """Represents callable operations in the IR."""

    name: Final[str]
    """Operation name."""

    def __init__(self, name: str) -> None:
        """Create an operation with the given name.

        Args:
            name: Operation name
        """
        ...

class IRNode:
    """Base class for all IR nodes."""

    span: Final[Span]
    """Source location of this IR node."""

class Expr(IRNode):
    """Base class for all expressions."""

    def __str__(self) -> str:
        """String representation of the expression.

        Returns:
            Expression as a string with minimal parentheses
        """
        ...

    def __repr__(self) -> str:
        """Detailed representation of the expression.

        Returns:
            Expression with type information
        """
        ...

class Var(Expr):
    """Variable reference expression."""

    name: Final[str]
    """Variable name."""

    def __init__(self, name: str, span: Span) -> None:
        """Create a variable reference expression.

        Args:
            name: Variable name
            span: Source location
        """
        ...

class ConstInt(Expr):
    """Constant integer expression."""

    value: Final[int]
    """Constant integer value."""

    def __init__(self, value: int, span: Span) -> None:
        """Create a constant integer expression.

        Args:
            value: Integer value
            span: Source location
        """
        ...

class Call(Expr):
    """Function call expression."""

    op: Final[Op]
    """Operation/function."""

    args: Final[List[Expr]]
    """Arguments."""

    def __init__(self, op: Op, args: List[Expr], span: Span) -> None:
        """Create a function call expression.

        Args:
            op: Operation/function to call
            args: List of argument expressions
            span: Source location
        """
        ...

class BinaryExpr(Expr):
    """Base class for binary operations."""

    left: Final[Expr]
    """Left operand."""

    right: Final[Expr]
    """Right operand."""

class UnaryExpr(Expr):
    """Base class for unary operations."""

    operand: Final[Expr]
    """Operand."""

class Add(BinaryExpr):
    """Addition expression (left + right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create an addition expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class Sub(BinaryExpr):
    """Subtraction expression (left - right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a subtraction expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class Mul(BinaryExpr):
    """Multiplication expression (left * right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a multiplication expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class FloorDiv(BinaryExpr):
    """Floor division expression (left // right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a floor division expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class FloorMod(BinaryExpr):
    """Floor modulo expression (left % right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a floor modulo expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class FloatDiv(BinaryExpr):
    """Float division expression (left / right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a float division expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class Min(BinaryExpr):
    """Minimum expression (min(left, right))."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a minimum expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class Max(BinaryExpr):
    """Maximum expression (max(left, right))."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a maximum expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class Pow(BinaryExpr):
    """Power expression (left ** right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a power expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class Eq(BinaryExpr):
    """Equality expression (left == right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create an equality expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class Ne(BinaryExpr):
    """Inequality expression (left != right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create an inequality expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class Lt(BinaryExpr):
    """Less than expression (left < right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a less than expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class Le(BinaryExpr):
    """Less than or equal to expression (left <= right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a less than or equal to expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class Gt(BinaryExpr):
    """Greater than expression (left > right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a greater than expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class Ge(BinaryExpr):
    """Greater than or equal to expression (left >= right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a greater than or equal to expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class And(BinaryExpr):
    """Logical and expression (left and right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a logical and expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class Or(BinaryExpr):
    """Logical or expression (left or right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a logical or expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class Xor(BinaryExpr):
    """Logical xor expression (left xor right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a logical xor expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class BitAnd(BinaryExpr):
    """Bitwise and expression (left & right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a bitwise and expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class BitOr(BinaryExpr):
    """Bitwise or expression (left | right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a bitwise or expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class BitXor(BinaryExpr):
    """Bitwise xor expression (left ^ right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a bitwise xor expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class BitShiftLeft(BinaryExpr):
    """Bitwise left shift expression (left << right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a bitwise left shift expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class BitShiftRight(BinaryExpr):
    """Bitwise right shift expression (left >> right)."""

    def __init__(self, left: Expr, right: Expr, span: Span) -> None:
        """Create a bitwise right shift expression.

        Args:
            left: Left operand
            right: Right operand
            span: Source location
        """
        ...

class Abs(UnaryExpr):
    """Absolute value expression (abs(operand))."""

    def __init__(self, operand: Expr, span: Span) -> None:
        """Create an absolute value expression.

        Args:
            operand: Operand expression
            span: Source location
        """
        ...

class Neg(UnaryExpr):
    """Negation expression (-operand)."""

    def __init__(self, operand: Expr, span: Span) -> None:
        """Create a negation expression.

        Args:
            operand: Operand expression
            span: Source location
        """
        ...

class Not(UnaryExpr):
    """Logical not expression (not operand)."""

    def __init__(self, operand: Expr, span: Span) -> None:
        """Create a logical not expression.

        Args:
            operand: Operand expression
            span: Source location
        """
        ...

class BitNot(UnaryExpr):
    """Bitwise not expression (~operand)."""

    def __init__(self, operand: Expr, span: Span) -> None:
        """Create a bitwise not expression.

        Args:
            operand: Operand expression
            span: Source location
        """
        ...
