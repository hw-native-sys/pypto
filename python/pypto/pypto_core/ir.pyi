# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Type stubs for PyPTO IR (Intermediate Representation) module."""

from typing import Final, Mapping, Sequence, Union, overload

from pypto import DataType

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

    def to_string(self) -> str:
        """Convert span to string representation.

        Returns:
            String in format "filename:begin_line:begin_column"
        """

    def is_valid(self) -> bool:
        """Check if the span has valid coordinates.

        Returns:
            True if all line/column numbers are positive
        """

    @staticmethod
    def unknown() -> Span:
        """Create an unknown/invalid span for cases where source location is unavailable.

        Returns:
            Span with empty filename and invalid coordinates
        """

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

    def get_attr(self, key: str) -> str | int | bool:
        """Get an attribute value (automatically determines type).

        Args:
            key: Attribute key

        Returns:
            The attribute value (str, int, or bool)

        Raises:
            RuntimeError: If attribute doesn't exist or has unsupported type
        """

    def has_attr(self, key: str) -> bool:
        """Check if an attribute exists.

        Args:
            key: Attribute key

        Returns:
            True if the attribute exists
        """

    def get_attr_keys(self) -> list[str]:
        """Get all attribute keys.

        Returns:
            List of all attribute keys
        """

class GlobalVar(Op):
    """Global variable reference for functions in a program.

    Can be used in Call expressions to invoke functions within the same program.
    The name of the GlobalVar should match the name of the function it references.
    """

    def __init__(self, name: str) -> None:
        """Create a global variable reference with the given name.

        Args:
            name: GlobalVar name (should match the function name)
        """

class IRNode:
    """Base class for all IR nodes."""

    span: Final[Span]
    """Source location of this IR node."""

    def same_as(self, other: IRNode) -> bool:
        """Check if this IR node is the same as another IR node."""

class Expr(IRNode):
    """Base class for all expressions."""

    type: Final[Type]
    """Type of the expression result."""

    # Binary operators (only work with ScalarType)
    def __add__(self, other: ScalarExprType) -> Expr:
        """Addition operator (self + other). Only works with ScalarType variables."""

    def __sub__(self, other: ScalarExprType) -> Expr:
        """Subtraction operator (self - other). Only works with ScalarType variables."""

    def __mul__(self, other: ScalarExprType) -> Expr:
        """Multiplication operator (self * other). Only works with ScalarType variables."""

    def __truediv__(self, other: ScalarExprType) -> Expr:
        """Division operator (self / other). Only works with ScalarType variables."""

    def __floordiv__(self, other: ScalarExprType) -> Expr:
        """Floor division operator (self // other). Only works with ScalarType variables."""

    def __mod__(self, other: ScalarExprType) -> Expr:
        """Modulo operator (self % other). Only works with ScalarType variables."""

    def __pow__(self, other: ScalarExprType) -> Expr:
        """Power operator (self ** other). Only works with ScalarType variables."""

    # Comparison operators (only work with ScalarType)
    def __eq__(self, other: ScalarExprType) -> Expr:  # type: ignore[override]
        """Equality operator (self == other). Only works with ScalarType variables."""

    def __ne__(self, other: ScalarExprType) -> Expr:  # type: ignore[override]
        """Inequality operator (self != other). Only works with ScalarType variables."""

    def __lt__(self, other: ScalarExprType) -> Expr:
        """Less than operator (self < other). Only works with ScalarType variables."""

    def __le__(self, other: ScalarExprType) -> Expr:
        """Less than or equal operator (self <= other). Only works with ScalarType variables."""

    def __gt__(self, other: ScalarExprType) -> Expr:
        """Greater than operator (self > other). Only works with ScalarType variables."""

    def __ge__(self, other: ScalarExprType) -> Expr:
        """Greater than or equal operator (self >= other). Only works with ScalarType variables."""

    # Bitwise operators (only work with ScalarType)
    def __and__(self, other: ScalarExprType) -> Expr:
        """Bitwise and operator (self & other). Only works with ScalarType variables."""

    def __or__(self, other: ScalarExprType) -> Expr:
        """Bitwise or operator (self | other). Only works with ScalarType variables."""

    def __xor__(self, other: ScalarExprType) -> Expr:
        """Bitwise xor operator (self ^ other). Only works with ScalarType variables."""

    def __lshift__(self, other: ScalarExprType) -> Expr:
        """Bitwise left shift operator (self << other). Only works with ScalarType variables."""

    def __rshift__(self, other: ScalarExprType) -> Expr:
        """Bitwise right shift operator (self >> other). Only works with ScalarType variables."""

    # Unary operators (only work with ScalarType)
    def __neg__(self) -> Expr:
        """Negation operator (-self). Only works with ScalarType variables."""

    def __invert__(self) -> Expr:
        """Bitwise not operator (~self). Only works with ScalarType variables."""

    # Reverse operators (only work with ScalarType)
    def __radd__(self, other: ScalarExprType) -> Expr:
        """Reverse addition operator (other + self). Only works with ScalarType variables."""

    def __rsub__(self, other: ScalarExprType) -> Expr:
        """Reverse subtraction operator (other - self). Only works with ScalarType variables."""

    def __rmul__(self, other: ScalarExprType) -> Expr:
        """Reverse multiplication operator (other * self). Only works with ScalarType variables."""

    def __rtruediv__(self, other: ScalarExprType) -> Expr:
        """Reverse division operator (other / self). Only works with ScalarType variables."""

    def __rfloordiv__(self, other: ScalarExprType) -> Expr:
        """Reverse floor division operator (other // self). Only works with ScalarType variables."""

    def __rmod__(self, other: ScalarExprType) -> Expr:
        """Reverse modulo operator (other % self). Only works with ScalarType variables."""

    def __rpow__(self, other: ScalarExprType) -> Expr:
        """Reverse power operator (other ** self). Only works with ScalarType variables."""

    def __rand__(self, other: ScalarExprType) -> Expr:
        """Reverse bitwise and operator (other & self). Only works with ScalarType variables."""

    def __ror__(self, other: ScalarExprType) -> Expr:
        """Reverse bitwise or operator (other | self). Only works with ScalarType variables."""

    def __rxor__(self, other: ScalarExprType) -> Expr:
        """Reverse bitwise xor operator (other ^ self). Only works with ScalarType variables."""

    def __rlshift__(self, other: ScalarExprType) -> Expr:
        """Reverse bitwise left shift operator (other << self). Only works with ScalarType variables."""

    def __rrshift__(self, other: ScalarExprType) -> Expr:
        """Reverse bitwise right shift operator (other >> self). Only works with ScalarType variables."""

# ========== Type System ==========

class Type:
    """Base class for type representations."""

class UnknownType(Type):
    """Unknown or unspecified type representation.

    Used as the default type for expressions when type information is not available.
    """

    def __init__(self) -> None:
        """Create an unknown type."""

    @staticmethod
    def get() -> UnknownType:
        """Get the singleton UnknownType instance.

        Returns:
            The singleton UnknownType instance
        """

class ScalarType(Type):
    """Scalar type representation."""

    dtype: Final[DataType]
    """Data type."""

    def __init__(self, dtype: DataType) -> None:
        """Create a scalar type.

        Args:
            dtype: Data type
        """

class TensorType(Type):
    """Tensor type representation."""

    dtype: Final[DataType]
    """Element data type."""

    shape: Final[Sequence[Expr]]
    """Shape dimensions."""

    @overload
    def __init__(self, shape: Sequence[Expr], dtype: DataType) -> None:
        """Create a tensor type.

        Args:
            shape: Shape dimensions as Expr nodes
            dtype: Element data type
        """

    @overload
    def __init__(self, shape: Sequence[int], dtype: DataType) -> None:
        """Create a tensor type.

        Args:
            shape: Shape dimensions as integers (automatically converted to ConstInt)
            dtype: Element data type
        """

class TileType(Type):
    """Tile type representation (2D tensor with at most 2 dimensions)."""

    dtype: Final[DataType]
    """Element data type."""

    shape: Final[Sequence[Expr]]
    """Shape dimensions (at most 2 dimensions)."""

    @overload
    def __init__(self, shape: Sequence[Expr], dtype: DataType) -> None:
        """Create a tile type (validates shape has at most 2 dimensions).

        Args:
            shape: Shape dimensions as Expr nodes
            dtype: Element data type

        Raises:
            Exception: If shape has more than 2 dimensions
        """

    @overload
    def __init__(self, shape: Sequence[int], dtype: DataType) -> None:
        """Create a tile type (validates shape has at most 2 dimensions).

        Args:
            shape: Shape dimensions as integers (automatically converted to ConstInt)
            dtype: Element data type

        Raises:
            Exception: If shape has more than 2 dimensions
        """

class TupleType(Type):
    """Tuple type representation (contains multiple types)."""

    types: Final[Sequence[Type]]
    """Types in the tuple."""

    def __init__(self, types: Sequence[Type]) -> None:
        """Create a tuple type from a list of types.

        Args:
            types: List of types in the tuple
        """

DYNAMIC_DIM: Final[int]
"""Constant representing a dynamic dimension (value: -1).

Used to indicate dimensions with runtime-determined sizes.
"""

ScalarExprType = Union[Expr, int, float]

class Var(Expr):
    """Variable reference expression."""

    name: Final[str]
    """Variable name."""

    def __init__(self, name: str, type: Type, span: Span) -> None:
        """Create a variable reference expression.

        Args:
            name: Variable name
            type: Type of the variable (ScalarType or TensorType)
            span: Source location
        """

    def __str__(self) -> str:
        """String representation of the variable."""

    def __repr__(self) -> str:
        """Detailed representation of the variable."""

class IterArg(Var):
    """Iteration argument variable."""

    initValue: Final[Expr]
    """Initial value expression (can be any Expr)."""

    def __init__(self, name: str, type: Type, initValue: Expr, span: Span) -> None:
        """Create an iteration argument with initial value and current value.

        Args:
            name: Variable name
            type: Type of the variable (ScalarType or TensorType)
            initValue: Initial value expression (can be any Expr)
            span: Source location
        """

    def __str__(self) -> str:
        """String representation of the iteration argument."""

    def __repr__(self) -> str:
        """Detailed representation of the iteration argument."""

class ConstInt(Expr):
    """Constant integer expression."""

    value: Final[int]
    """Constant integer value."""

    def __init__(self, value: int, dtype: DataType, span: Span) -> None:
        """Create a constant integer expression.

        Args:
            value: Integer value
            dtype: Data type
            span: Source location
        """

    @property
    def dtype(self) -> DataType:
        """Data type of the expression."""

class ConstFloat(Expr):
    """Constant floating-point expression."""

    value: Final[float]
    """Constant floating-point value."""

    def __init__(self, value: float, dtype: DataType, span: Span) -> None:
        """Create a constant floating-point expression.

        Args:
            value: Floating-point value
            dtype: Data type
            span: Source location
        """

    @property
    def dtype(self) -> DataType:
        """Data type of the expression."""

class ConstBool(Expr):
    """Constant boolean expression."""

    value: Final[bool]
    """Constant boolean value."""

    def __init__(self, value: bool, span: Span) -> None:
        """Create a constant boolean expression.

        Args:
            value: Boolean value
            span: Source location

        Note:
            dtype is always DataType.BOOL - no need to specify.
        """

    @property
    def dtype(self) -> DataType:
        """Data type of the expression (always DataType.BOOL)."""

class Call(Expr):
    """Function call expression."""

    op: Final[Op]
    """Operation/function."""

    args: Final[Sequence[Expr]]
    """Positional arguments."""

    kwargs: Final[Mapping[str, Union[int, bool, str, float, DataType]]]
    """Keyword arguments (metadata)."""

    @overload
    def __init__(self, op: Op, args: Sequence[Expr], span: Span) -> None:
        """Create a function call expression.

        Args:
            op: Operation/function to call
            args: List of argument expressions
            kwargs: Keyword arguments (metadata)
            span: Source location
        """
        ...

    @overload
    def __init__(
        self,
        op: Op,
        args: Sequence[Expr],
        kwargs: Mapping[str, Union[int, bool, str, float, DataType]],
        type: Type,
        span: Span,
    ) -> None:
        """Create a function call expression with explicit type.

        Args:
            op: Operation/function to call
            args: List of argument expressions
            kwargs: Keyword arguments (metadata)
            type: Explicit result type
            span: Source location
        """
        ...

    def __str__(self) -> str:
        """String representation of the call expression."""

    def __repr__(self) -> str:
        """Detailed representation of the call expression."""

class TupleGetItemExpr(Expr):
    """Tuple element access expression."""

    tuple: Final[Expr]
    """Tuple expression (must have TupleType)."""

    index: Final[int]
    """Index of the element to access (0-based)."""

    def __init__(self, tuple: Expr, index: int, span: Span) -> None:
        """Create a tuple element access expression.

        Args:
            tuple: Tuple expression (must have TupleType type)
            index: Index of the element (0-based, must be within bounds)
            span: Source location

        Raises:
            Exception: If tuple does not have TupleType
            Exception: If index is out of bounds
        """

    def __str__(self) -> str:
        """String representation of the tuple access expression."""

    def __repr__(self) -> str:
        """Detailed representation of the tuple access expression."""

class BinaryExpr(Expr):
    """Base class for binary operations."""

    dtype: Final[DataType]
    """Data type of the expression."""

    left: Final[Expr]
    """Left operand."""

    right: Final[Expr]
    """Right operand."""

class UnaryExpr(Expr):
    """Base class for unary operations."""

    dtype: Final[DataType]
    """Data type of the expression."""

    operand: Final[Expr]
    """Operand."""

class Add(BinaryExpr):
    """Addition expression (left + right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create an addition expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class Sub(BinaryExpr):
    """Subtraction expression (left - right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a subtraction expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class Mul(BinaryExpr):
    """Multiplication expression (left * right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a multiplication expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class FloorDiv(BinaryExpr):
    """Floor division expression (left // right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a floor division expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class FloorMod(BinaryExpr):
    """Floor modulo expression (left % right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a floor modulo expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class FloatDiv(BinaryExpr):
    """Float division expression (left / right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a float division expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class Min(BinaryExpr):
    """Minimum expression (min(left, right))."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a minimum expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class Max(BinaryExpr):
    """Maximum expression (max(left, right))."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a maximum expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class Pow(BinaryExpr):
    """Power expression (left ** right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a power expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class Eq(BinaryExpr):
    """Equality expression (left == right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create an equality expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class Ne(BinaryExpr):
    """Inequality expression (left != right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create an inequality expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class Lt(BinaryExpr):
    """Less than expression (left < right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a less than expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class Le(BinaryExpr):
    """Less than or equal to expression (left <= right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a less than or equal to expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class Gt(BinaryExpr):
    """Greater than expression (left > right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a greater than expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class Ge(BinaryExpr):
    """Greater than or equal to expression (left >= right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a greater than or equal to expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class And(BinaryExpr):
    """Logical and expression (left and right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a logical and expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class Or(BinaryExpr):
    """Logical or expression (left or right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a logical or expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class Xor(BinaryExpr):
    """Logical xor expression (left xor right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a logical xor expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class BitAnd(BinaryExpr):
    """Bitwise and expression (left & right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a bitwise and expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class BitOr(BinaryExpr):
    """Bitwise or expression (left | right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a bitwise or expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class BitXor(BinaryExpr):
    """Bitwise xor expression (left ^ right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a bitwise xor expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class BitShiftLeft(BinaryExpr):
    """Bitwise left shift expression (left << right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a bitwise left shift expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class BitShiftRight(BinaryExpr):
    """Bitwise right shift expression (left >> right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a bitwise right shift expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """

class Abs(UnaryExpr):
    """Absolute value expression (abs(operand))."""

    def __init__(self, operand: Expr, dtype: DataType, span: Span) -> None:
        """Create an absolute value expression.

        Args:
            operand: Operand expression
            dtype: Data type
            span: Source location
        """

class Neg(UnaryExpr):
    """Negation expression (-operand)."""

    def __init__(self, operand: Expr, dtype: DataType, span: Span) -> None:
        """Create a negation expression.

        Args:
            operand: Operand expression
            dtype: Data type
            span: Source location
        """

class Not(UnaryExpr):
    """Logical not expression (not operand)."""

    def __init__(self, operand: Expr, dtype: DataType, span: Span) -> None:
        """Create a logical not expression.

        Args:
            operand: Operand expression
            dtype: Data type
            span: Source location
        """

class BitNot(UnaryExpr):
    """Bitwise not expression (~operand)."""

    def __init__(self, operand: Expr, dtype: DataType, span: Span) -> None:
        """Create a bitwise not expression.

        Args:
            operand: Operand expression
            dtype: Data type
            span: Source location
        """

class Cast(UnaryExpr):
    """Cast expression (cast operand to dtype)."""

    def __init__(self, operand: Expr, dtype: DataType, span: Span) -> None:
        """Create a cast expression.

        Args:
            operand: Operand expression
            dtype: Target data type
            span: Source location
        """

class Stmt(IRNode):
    """Base class for all statements."""

    def __init__(self, span: Span) -> None:
        """Create a statement.

        Args:
            span: Source location
        """

class AssignStmt(Stmt):
    """Assignment statement: var = value."""

    var: Final[Var]
    """Variable."""

    value: Final[Expr]
    """Expression."""

    def __init__(self, var: Var, value: Expr, span: Span) -> None:
        """Create an assignment statement.

        Args:
            var: Variable
            value: Expression
            span: Source location
        """

class IfStmt(Stmt):
    """Conditional statement: if condition then then_body else else_body."""

    condition: Final[Expr]
    """Condition expression."""

    then_body: Final[Stmt]
    """Then branch statement."""

    else_body: Final[Stmt | None]
    """Else branch statement (can be None)."""

    return_vars: Final[list[Var]]
    """Return variables (can be empty)."""

    def __init__(
        self,
        condition: Expr,
        then_body: Stmt,
        else_body: Stmt | None,
        return_vars: list[Var],
        span: Span,
    ) -> None:
        """Create a conditional statement with then and else branches.

        Args:
            condition: Condition expression
            then_body: Then branch statement
            else_body: Else branch statement (can be None)
            return_vars: Return variables (can be empty)
            span: Source location
        """
        ...

class YieldStmt(Stmt):
    """Yield statement: yield value."""

    value: Final[list[Expr]]
    """List of variables to yield (can be empty)."""

    @overload
    def __init__(self, value: list[Expr], span: Span) -> None:
        """Create a yield statement with a list of variables.

        Args:
            value: List of variables to yield
            span: Source location
        """
        ...

    @overload
    def __init__(self, span: Span) -> None:
        """Create a yield statement without values.

        Args:
            span: Source location
        """
        ...

class ReturnStmt(Stmt):
    """Return statement: return value."""

    value: Final[list[Expr]]
    """List of expressions to return (can be empty)."""

    @overload
    def __init__(self, value: list[Expr], span: Span) -> None:
        """Create a return statement with a list of expressions.

        Args:
            value: List of expressions to return
            span: Source location
        """
        ...

    @overload
    def __init__(self, span: Span) -> None:
        """Create a return statement without values.

        Args:
            span: Source location
        """
        ...

class ForStmt(Stmt):
    """For loop statement: for loop_var in range(start, stop, step): body."""

    loop_var: Final[Var]
    """Loop variable."""

    start: Final[Expr]
    """Start value expression."""

    stop: Final[Expr]
    """Stop value expression."""

    step: Final[Expr]
    """Step value expression."""

    iter_args: Final[list[IterArg]]
    """Iteration arguments (can be empty)."""

    body: Final[Stmt]
    """Loop body statement."""

    return_vars: Final[list[Var]]
    """Return variables (can be empty)."""

    def __init__(
        self,
        loop_var: Var,
        start: Expr,
        stop: Expr,
        step: Expr,
        iter_args: list[IterArg],
        body: Stmt,
        return_vars: list[Var],
        span: Span,
    ) -> None:
        """Create a for loop statement.

        Args:
            loop_var: Loop variable
            start: Start value expression
            stop: Stop value expression
            step: Step value expression
            body: Loop body statements
            return_vars: Return variables (can be empty)
            span: Source location
        """

class SeqStmts(Stmt):
    """Sequence of statements: a sequence of statements."""

    stmts: Final[list[Stmt]]
    """List of statements."""

    def __init__(self, stmts: list[Stmt], span: Span) -> None:
        """Create a sequence of statements.

        Args:
            stmts: List of statements
            span: Source location
        """

    def __str__(self) -> str:
        """String representation of the sequence of statements.

        Returns:
            Sequence of statements as a string
        """

    def __repr__(self) -> str:
        """Detailed representation of the sequence of statements.

        Returns:
            Sequence of statements with type information
        """

class OpStmts(Stmt):
    """Operation statements: a sequence of assignment statements."""

    stmts: Final[list[AssignStmt]]
    """List of assignment statements."""

    def __init__(self, stmts: list[AssignStmt], span: Span) -> None:
        """Create an operation statements.

        Args:
            stmts: List of assignment statements
            span: Source location
        """

    def __str__(self) -> str:
        """String representation of the operation statements.

        Returns:
            Operation statements as a string
        """

    def __repr__(self) -> str:
        """Detailed representation of the operation statements.

        Returns:
            Operation statements with type information
        """

class Function(IRNode):
    """Function definition with name, parameters, return types, and body."""

    name: Final[str]
    """Function name."""

    params: Final[list[Var]]
    """Parameter variables."""

    return_types: Final[list[Type]]
    """Return types."""

    body: Final[Stmt]
    """Function body statement (use SeqStmts for multiple statements)."""

    def __init__(
        self,
        name: str,
        params: list[Var],
        return_types: list[Type],
        body: Stmt,
        span: Span,
    ) -> None:
        """Create a function definition.

        Args:
            name: Function name
            params: Parameter variables
            return_types: Return types
            body: Function body statement (use SeqStmts for multiple statements)
            span: Source location
        """

    def __str__(self) -> str:
        """String representation of the function.

        Returns:
            Function as a string
        """

    def __repr__(self) -> str:
        """Detailed representation of the function.

        Returns:
            Function with type information
        """

class Program(IRNode):
    """Program definition with functions mapped by GlobalVar references.

    Functions are automatically sorted by name for deterministic ordering.
    The GlobalVar name must match the function name and be unique within the program.
    """

    name: Final[str]
    """Program name."""

    functions: Final[dict[GlobalVar, Function]]
    """Map of GlobalVar references to their corresponding functions, sorted by GlobalVar name."""

    def __init__(
        self,
        functions: list[Function],
        name: str,
        span: Span,
    ) -> None:
        """Create a program from a list of functions.

        GlobalVar references are created automatically from function names.

        Args:
            functions: List of functions
            name: Program name (optional)
            span: Source location
        """

    def get_function(self, name: str) -> Function | None:
        """Get a function by name.

        Args:
            name: Function name to look up

        Returns:
            Function if found, None otherwise
        """

    def get_global_var(self, name: str) -> GlobalVar | None:
        """Get a GlobalVar by name.

        Args:
            name: GlobalVar name to look up

        Returns:
            GlobalVar if found, None otherwise
        """

    def __str__(self) -> str:
        """String representation of the program.

        Returns:
            Program as a string
        """

    def __repr__(self) -> str:
        """Detailed representation of the program.

        Returns:
            Program with type information
        """

@overload
def structural_hash(node: IRNode, enable_auto_mapping: bool = False) -> int: ...
@overload
def structural_hash(node: Type, enable_auto_mapping: bool = False) -> int: ...
def structural_hash(node: IRNode | Type, enable_auto_mapping: bool = False) -> int:
    """Compute structural hash of an IR node or type.

    Ignores source location (Span). Two objects with identical structure hash to the same value.
    If enable_auto_mapping=True, variable names are ignored (e.g., x+1 and y+1 hash the same).
    If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same name).

    Args:
        node: IR node or type to compute hash for
        enable_auto_mapping: Whether to ignore variable identity and auto-map variables

    Returns:
        Hash value of the object structure
    """

@overload
def structural_equal(lhs: IRNode, rhs: IRNode, enable_auto_mapping: bool = False) -> bool: ...
@overload
def structural_equal(lhs: Type, rhs: Type, enable_auto_mapping: bool = False) -> bool: ...
def structural_equal(lhs: IRNode | Type, rhs: IRNode | Type, enable_auto_mapping: bool = False) -> bool:
    """Check if two IR nodes or types are structurally equal.

    Ignores source location (Span). Returns True if objects have identical structure.
    If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1).
    If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same name).

    Args:
        lhs: Left-hand side IR node or type
        rhs: Right-hand side IR node or type
        enable_auto_mapping: Whether to automatically map variables

    Returns:
        True if objects are structurally equal, False otherwise
    """

@overload
def assert_structural_equal(lhs: IRNode, rhs: IRNode, enable_auto_mapping: bool = False) -> None: ...
@overload
def assert_structural_equal(lhs: Type, rhs: Type, enable_auto_mapping: bool = False) -> None: ...
def assert_structural_equal(
    lhs: IRNode | Type, rhs: IRNode | Type, enable_auto_mapping: bool = False
) -> None:
    """Assert two IR nodes or types are structurally equal.

    Like structural_equal but raises ValueError with detailed error message showing
    the first mismatch location and Python-printed IR context. Useful for debugging.

    Ignores source location (Span).
    If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1).
    If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same name).

    Args:
        lhs: Left-hand side IR node or type
        rhs: Right-hand side IR node or type
        enable_auto_mapping: Whether to automatically map variables

    Raises:
        ValueError: If objects are not structurally equal, with detailed diagnostic message
    """

def serialize(node: IRNode) -> bytes:
    """Serialize an IR node to MessagePack bytes.

    The serialized data preserves:
    - All node structure and field values
    - Pointer sharing (if a node is referenced multiple times, it's serialized once)
    - Source location (Span) information
    - Type information

    Args:
        node: IR node to serialize

    Returns:
        MessagePack-encoded bytes representing the IR node

    Example:
        >>> x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        >>> data = ir.serialize(x)
        >>> restored = ir.deserialize(data)
        >>> ir.structural_equal(x, restored, enable_auto_mapping=True)
        True
    """

def deserialize(data: bytes) -> IRNode:
    """Deserialize an IR node from MessagePack bytes.

    Reconstructs the IR node from serialized data, preserving:
    - All node structure and field values
    - Pointer sharing (shared references are restored correctly)
    - Source location (Span) information
    - Type information

    Args:
        data: MessagePack-encoded bytes

    Returns:
        The deserialized IR node

    Raises:
        RuntimeError: If the data is corrupt or invalid

    Example:
        >>> x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        >>> data = ir.serialize(x)
        >>> restored = ir.deserialize(data)
        >>> ir.structural_equal(x, restored, enable_auto_mapping=True)
        True
    """

def serialize_to_file(node: IRNode, path: str) -> None:
    """Serialize an IR node to a file.

    Convenience function that serializes the node and writes it to a file.

    Args:
        node: IR node to serialize
        path: Path to the output file

    Raises:
        RuntimeError: If the file cannot be written

    Example:
        >>> x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        >>> ir.serialize_to_file(x, "node.msgpack")
        >>> restored = ir.deserialize_from_file("node.msgpack")
        >>> ir.structural_equal(x, restored, enable_auto_mapping=True)
        True
    """

def deserialize_from_file(path: str) -> IRNode:
    """Deserialize an IR node from a file.

    Convenience function that reads a file and deserializes the IR node.

    Args:
        path: Path to the input file

    Returns:
        The deserialized IR node

    Raises:
        RuntimeError: If the file cannot be read or the data is invalid

    Example:
        >>> x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        >>> ir.serialize_to_file(x, "node.msgpack")
        >>> restored = ir.deserialize_from_file("node.msgpack")
        >>> ir.structural_equal(x, restored, enable_auto_mapping=True)
        True
    """

# ========== Operator Registry ==========

@overload
def create_op_call(op_name: str, args: Sequence[Expr], span: Span) -> Call:
    """Create a Call expression (backward compatibility).

    Args:
        op_name: Name of the registered operator
        args: List of argument expressions
        span: Source location

    Returns:
        Call expression with automatically deduced result type

    Raises:
        Exception: If operator is not registered or type deduction fails
    """

@overload
def create_op_call(
    op_name: str,
    args: Sequence[Expr],
    kwargs: Mapping[str, int | bool | str | float | DataType],
    span: Span,
) -> Call:
    """Create a Call expression with args and kwargs.

    Args:
        op_name: Name of the registered operator
        args: Positional Expr arguments
        kwargs: Keyword arguments (metadata)
        span: Source location

    Returns:
        Call expression with automatically deduced result type

    Raises:
        Exception: If operator is not registered or type deduction fails
    """

def is_op_registered(op_name: str) -> bool:
    """Check if an operator is registered.

    Args:
        op_name: Name of the operator to check

    Returns:
        True if the operator is registered, False otherwise
    """

def get_op(op_name: str) -> Op:
    """Get an operator instance by name.

    Args:
        op_name: Name of the operator

    Returns:
        The operator instance

    Raises:
        Exception: If operator is not registered
    """

# ========== IR Builder ==========

class IRBuilder:
    """IR Builder for incremental IR construction with context management.

    The IRBuilder provides a stateful API for building IR incrementally using
    Begin/End patterns. It maintains a context stack to track nested scopes
    and validates proper construction.
    """

    def __init__(self) -> None:
        """Create an IR builder."""

    # Function building
    def BeginFunction(self, name: str, span: Span) -> None:
        """Begin building a function.

        Args:
            name: Function name
            span: Source location for function definition
        """

    def FuncArg(self, name: str, type: Type, span: Span) -> Var:
        """Add a function parameter.

        Args:
            name: Parameter name
            type: Parameter type
            span: Source location for parameter

        Returns:
            Variable representing the parameter
        """

    def ReturnType(self, type: Type) -> None:
        """Add a return type to the current function.

        Args:
            type: Return type
        """

    def EndFunction(self, end_span: Span) -> Function:
        """End building a function.

        Args:
            end_span: Source location for end of function

        Returns:
            The built function
        """

    # For loop building
    def BeginForLoop(self, loop_var: Var, start: Expr, stop: Expr, step: Expr, span: Span) -> None:
        """Begin building a for loop.

        Args:
            loop_var: Loop variable
            start: Start value expression
            stop: Stop value expression
            step: Step value expression
            span: Source location for loop definition
        """

    def AddIterArg(self, iter_arg: IterArg) -> None:
        """Add an iteration argument to the current for loop.

        Args:
            iter_arg: Iteration argument with initial value
        """

    def AddReturnVar(self, var: Var) -> None:
        """Add a return variable to the current for loop.

        Args:
            var: Return variable
        """

    def EndForLoop(self, end_span: Span) -> ForStmt:
        """End building a for loop.

        Args:
            end_span: Source location for end of loop

        Returns:
            The built for statement
        """

    # If statement building
    def BeginIf(self, condition: Expr, span: Span) -> None:
        """Begin building an if statement.

        Args:
            condition: Condition expression
            span: Source location for if statement
        """

    def BeginElse(self, span: Span) -> None:
        """Begin the else branch of the current if statement.

        Args:
            span: Source location for else keyword
        """

    def AddIfReturnVar(self, var: Var) -> None:
        """Add a return variable to the current if statement.

        Args:
            var: Return variable
        """

    def EndIf(self, end_span: Span) -> IfStmt:
        """End building an if statement.

        Args:
            end_span: Source location for end of if

        Returns:
            The built if statement
        """

    # Statement recording
    def Emit(self, stmt: Stmt) -> None:
        """Emit a statement in the current context.

        Args:
            stmt: Statement to emit
        """

    def Assign(self, var: Var, value: Expr, span: Span) -> AssignStmt:
        """Create an assignment statement and emit it.

        Args:
            var: Variable to assign to
            value: Expression value
            span: Source location for assignment

        Returns:
            The created assignment statement
        """

    def Var(self, name: str, type: Type, span: Span) -> Var:
        """Create a variable (does not emit).

        Args:
            name: Variable name
            type: Variable type
            span: Source location

        Returns:
            The created variable
        """

    def Return(self, values: list[Expr], span: Span) -> ReturnStmt:
        """Create a return statement and emit it.

        Args:
            values: List of expressions to return (can be empty)
            span: Source location for return statement

        Returns:
            The created return statement
        """

    # Context state queries
    def InFunction(self) -> bool:
        """Check if currently inside a function.

        Returns:
            True if inside a function context
        """

    def InLoop(self) -> bool:
        """Check if currently inside a for loop.

        Returns:
            True if inside a for loop context
        """

    def InIf(self) -> bool:
        """Check if currently inside an if statement.

        Returns:
            True if inside an if statement context
        """

# ========== Python Printer ==========
def python_print(node: IRNode, prefix: str = "pi") -> str:
    """Print an IR node as a Python string.

    Args:
        node: IR node to print
        prefix: Module prefix (default 'pi' for 'import pypto.ir as pi')

    Returns:
        String representation of the IR node
    """

def add(lhs: Expr, rhs: Expr, span: Span) -> Expr:
    """Addition operator (lhs + rhs)."""

def sub(lhs: Expr, rhs: Expr, span: Span) -> Expr:
    """Subtraction operator (lhs - rhs)."""

def mul(lhs: Expr, rhs: Expr, span: Span) -> Expr:
    """Multiplication operator (lhs * rhs)."""

def floor_div(lhs: Expr, rhs: Expr, span: Span) -> Expr:
    """Floor division operator (lhs // rhs)."""

def floor_mod(lhs: Expr, rhs: Expr, span: Span) -> Expr:
    """Floor modulo operator (lhs % rhs)."""

def pow(lhs: Expr, rhs: Expr, span: Span) -> Expr:
    """Power operator (lhs ** rhs)."""

def cast(operand: Expr, dtype: DataType, span: Span) -> Expr:
    """Cast operator (cast operand to dtype)."""

def bit_and(lhs: Expr, rhs: Expr, span: Span) -> Expr:
    """Bitwise and operator (lhs & rhs)."""

def bit_or(lhs: Expr, rhs: Expr, span: Span) -> Expr:
    """Bitwise or operator (lhs | rhs)."""

def bit_xor(lhs: Expr, rhs: Expr, span: Span) -> Expr:
    """Bitwise xor operator (lhs ^ rhs)."""

def bit_shift_left(lhs: Expr, rhs: Expr, span: Span) -> Expr:
    """Bitwise left shift operator (lhs << rhs)."""

def bit_shift_right(lhs: Expr, rhs: Expr, span: Span) -> Expr:
    """Bitwise right shift operator (lhs >> rhs)."""

def bit_not(operand: Expr, span: Span) -> Expr:
    """Bitwise not operator (~operand)."""
