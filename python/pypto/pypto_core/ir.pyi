# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Type stubs for PyPTO IR (Intermediate Representation) module."""

import enum
from collections.abc import Mapping, Sequence
from typing import Final, overload

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

    pipe: Final[PipeType | None]
    """Pipeline type associated with this operation."""

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

class ShapedType(Type):
    """Base class for shaped types (tensors and tiles)."""

    dtype: Final[DataType]
    """Element data type."""

    shape: Final[Sequence[Expr]]
    """Shape dimensions."""

    memref: Final[MemRef | None]
    """Optional memory reference."""

    def shares_memref_with(self, other: ShapedType) -> bool:
        """Check if this ShapedType shares the same MemRef object with another ShapedType.

        Args:
            other: Another ShapedType to compare with

        Returns:
            True if both have MemRef and they point to the same object, False otherwise
        """
        ...

class TensorLayout(enum.Enum):
    """Tensor layout type enumeration."""

    ND = ...
    """ND layout."""

    DN = ...
    """DN layout."""

    NZ = ...
    """NZ layout."""

class TensorView:
    """Tensor view representation with stride and layout."""

    stride: Sequence[Expr]
    """Stride for each dimension."""

    layout: TensorLayout
    """Tensor layout type."""

    @overload
    def __init__(self) -> None:
        """Create an empty tensor view with default ND layout."""

    @overload
    def __init__(self, stride: Sequence[Expr], layout: TensorLayout) -> None:
        """Create a tensor view with stride and layout.

        Args:
            stride: Stride for each dimension
            layout: Tensor layout type (ND, DN, or NZ)
        """

class TensorType(ShapedType):
    """Tensor type representation."""

    tensor_view: Final[TensorView | None]
    """Optional tensor view information."""

    @overload
    def __init__(self, shape: Sequence[Expr], dtype: DataType) -> None:
        """Create a tensor type without memory reference.

        Args:
            shape: Shape dimensions as Expr nodes
            dtype: Element data type
        """

    @overload
    def __init__(self, shape: Sequence[Expr], dtype: DataType, memref: MemRef | None) -> None:
        """Create a tensor type with memory reference.

        Args:
            shape: Shape dimensions as Expr nodes
            dtype: Element data type
            memref: Optional memory reference
        """

    @overload
    def __init__(
        self,
        shape: Sequence[Expr],
        dtype: DataType,
        memref: MemRef | None,
        tensor_view: TensorView | None,
    ) -> None:
        """Create a tensor type with memory reference and tensor view.

        Args:
            shape: Shape dimensions as Expr nodes
            dtype: Element data type
            memref: Optional memory reference
            tensor_view: Optional tensor view information
        """

    @overload
    def __init__(self, shape: Sequence[int], dtype: DataType) -> None:
        """Create a tensor type without memory reference.

        Args:
            shape: Shape dimensions as integers (automatically converted to ConstInt)
            dtype: Element data type
        """

    @overload
    def __init__(self, shape: Sequence[int], dtype: DataType, memref: MemRef | None) -> None:
        """Create a tensor type with memory reference.

        Args:
            shape: Shape dimensions as integers (automatically converted to ConstInt)
            dtype: Element data type
            memref: Optional memory reference
        """

    @overload
    def __init__(
        self,
        shape: Sequence[int],
        dtype: DataType,
        memref: MemRef | None,
        tensor_view: TensorView | None,
    ) -> None:
        """Create a tensor type with memory reference and tensor view.

        Args:
            shape: Shape dimensions as integers (automatically converted to ConstInt)
            dtype: Element data type
            memref: Optional memory reference
            tensor_view: Optional tensor view information
        """

class TileView:
    """Tile view representation with valid shape, stride, and start offset."""

    valid_shape: Sequence[Expr]
    """Valid shape dimensions."""

    stride: Sequence[Expr]
    """Stride for each dimension."""

    start_offset: Expr
    """Starting offset."""

    @overload
    def __init__(self) -> None:
        """Create an empty tile view."""

    @overload
    def __init__(self, valid_shape: Sequence[Expr], stride: Sequence[Expr], start_offset: Expr) -> None:
        """Create a tile view with valid_shape, stride, and start_offset.

        Args:
            valid_shape: Valid shape dimensions
            stride: Stride for each dimension
            start_offset: Starting offset
        """

class TileType(ShapedType):
    """Tile type representation (multi-dimensional tensor)."""

    tile_view: Final[TileView | None]
    """Optional tile view information."""

    @overload
    def __init__(self, shape: Sequence[Expr], dtype: DataType) -> None:
        """Create a tile type without memory reference.

        Args:
            shape: Shape dimensions as Expr nodes
            dtype: Element data type
        """

    @overload
    def __init__(self, shape: Sequence[Expr], dtype: DataType, memref: MemRef | None) -> None:
        """Create a tile type with memory reference.

        Args:
            shape: Shape dimensions as Expr nodes
            dtype: Element data type
            memref: Optional memory reference
        """

    @overload
    def __init__(
        self, shape: Sequence[Expr], dtype: DataType, memref: MemRef | None, tile_view: TileView | None
    ) -> None:
        """Create a tile type with memory reference and tile view.

        Args:
            shape: Shape dimensions as Expr nodes (supports multi-dimensional tensors)
            dtype: Element data type
            memref: Optional memory reference
            tile_view: Optional tile view information

        Note:
            Code generation currently only supports up to 2D tiles.
        """

    @overload
    def __init__(self, shape: Sequence[int], dtype: DataType) -> None:
        """Create a tile type without memory reference (validates shape has at most 2 dimensions).

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

class PipeType(enum.IntEnum):
    """Pipeline type enumeration for hardware execution units."""

    MTE1 = ...
    MTE2 = ...
    MTE3 = ...
    M = ...
    V = ...
    S = ...
    FIX = ...
    ALL = ...

class CoreType(enum.IntEnum):
    """Core type enumeration."""

    VECTOR = ...
    CUBE = ...

class FunctionType(enum.Enum):
    """Function type classification.

    Categorizes functions by their execution context and purpose:
    - Opaque: Unspecified (default)
    - Orchestration: Runs on host/AICPU for control flow and dependency analysis
    - InCore: Sub-graph on specific AICore
    """

    Opaque = ...
    """Unspecified function type (default)."""

    Orchestration = ...
    """Host/AICPU control and coordination."""

    InCore = ...
    """AICore sub-graph execution."""

class ForKind(enum.Enum):
    """For loop kind classification.

    Distinguishes sequential vs parallel for loops:
    - Sequential: Standard sequential for loop (default)
    - Parallel: Parallel for loop
    """

    Sequential = ...
    """Standard sequential for loop (default)."""

    Parallel = ...
    """Parallel for loop."""

class MemorySpace(enum.Enum):
    """Memory space enumeration."""

    DDR = ...
    """DDR memory (off-chip)."""

    UB = ...
    """Unified Buffer (on-chip)."""

    L1 = ...
    """L1 cache."""

    L0A = ...
    """L0A buffer."""

    L0B = ...
    """L0B buffer."""

    L0C = ...
    """L0C buffer."""

class MemRef(Var):
    """Memory reference variable for shaped types (inherits from Var)."""

    memory_space_: MemorySpace
    """Memory space (DDR, UB, L1, etc.)."""

    addr_: Expr
    """Starting address expression."""

    size_: int
    """Size in bytes (64-bit unsigned)."""

    id_: int
    """Unique identifier for this MemRef instance."""

    def __init__(self, memory_space: MemorySpace, addr: Expr, size: int, id: int, span: Span = ...) -> None:
        """Create a memory reference with memory_space, addr, size, id, and span.

        Args:
            memory_space: Memory space (DDR, UB, L1, etc.)
            addr: Starting address expression
            size: Size in bytes
            id: Unique identifier for this MemRef instance
            span: Source location (defaults to Span.unknown())
        """

DYNAMIC_DIM: Final[int]
"""Constant representing a dynamic dimension (value: -1).

Used to indicate dimensions with runtime-determined sizes.
"""

ScalarExprType = Expr | int | float

class Var(Expr):
    """Variable reference expression."""

    name: Final[str]
    """Variable name."""

    def __init__(self, name: str, type: Type, span: Span) -> None:
        """Create a variable reference.

        Args:
            name: Variable name
            type: Type of the variable (ScalarType, TensorType, or TileType)
                  Memory reference information is stored in ShapedType for Tensor/Tile types
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
        """Create an iteration argument.

        Args:
            name: Variable name
            type: Type of the variable (ScalarType, TensorType, or TileType)
                  Memory reference information is stored in ShapedType for Tensor/Tile types
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

    kwargs: Final[Mapping[str, int | bool | str | float | DataType | MemorySpace]]
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
        type: Type,
        span: Span,
    ) -> None:
        """Create a function call expression with explicit type.

        Args:
            op: Operation/function to call
            args: List of argument expressions
            type: Explicit result type
            span: Source location
        """
        ...

    @overload
    def __init__(
        self,
        op: Op,
        args: Sequence[Expr],
        kwargs: Mapping[str, int | bool | str | float | DataType | MemorySpace],
        span: Span,
    ) -> None:
        """Create a function call expression with kwargs.

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
        kwargs: Mapping[str, int | bool | str | float | DataType | MemorySpace],
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

class MakeTuple(Expr):
    """Tuple construction expression."""

    elements: Final[Sequence[Expr]]
    """Elements of the tuple."""

    def __init__(self, elements: Sequence[Expr], span: Span) -> None:
        """Create a tuple construction expression.

        Args:
            elements: Expressions to be tuple elements
            span: Source location

        The result type is automatically set to TupleType containing
        the types of all input expressions.
        """

    def __str__(self) -> str:
        """String representation of the tuple construction expression."""

    def __repr__(self) -> str:
        """Detailed representation of the tuple construction expression."""

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

    kind: Final[ForKind]
    """Loop kind (Sequential or Parallel)."""

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
        kind: ForKind = ForKind.Sequential,
    ) -> None:
        """Create a for loop statement.

        Args:
            loop_var: Loop variable
            start: Start value expression
            stop: Stop value expression
            step: Step value expression
            iter_args: Iteration arguments (can be empty)
            body: Loop body statements
            return_vars: Return variables (can be empty)
            span: Source location
            kind: Loop kind (default: Sequential)
        """

class WhileStmt(Stmt):
    """While loop statement: while condition: body."""

    condition: Final[Expr]
    """Condition expression."""

    iter_args: Final[list[IterArg]]
    """Iteration arguments (can be empty)."""

    body: Final[Stmt]
    """Loop body statement."""

    return_vars: Final[list[Var]]
    """Return variables (can be empty)."""

    def __init__(
        self,
        condition: Expr,
        iter_args: list[IterArg],
        body: Stmt,
        return_vars: list[Var],
        span: Span,
    ) -> None:
        """Create a while loop statement.

        Args:
            condition: Condition expression
            iter_args: Iteration arguments (can be empty)
            body: Loop body statement
            return_vars: Return variables (can be empty)
            span: Source location
        """

class ScopeKind(enum.Enum):
    """Scope kind classification."""

    InCore = 0
    """InCore scope for AICore sub-graphs."""

class ScopeStmt(Stmt):
    """Scope statement: marks a region with specific execution context."""

    scope_kind: Final[ScopeKind]
    """The kind of scope."""

    body: Final[Stmt]
    """The nested statements."""

    def __init__(self, scope_kind: ScopeKind, body: Stmt, span: Span) -> None:
        """Create a scope statement.

        Args:
            scope_kind: The kind of scope (e.g., ScopeKind.InCore)
            body: The nested statements
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
    """Operation statements: a sequence of assignment and/or evaluation statements."""

    stmts: Final[list[AssignStmt | EvalStmt]]
    """List of assignment and/or evaluation statements."""

    def __init__(self, stmts: list[AssignStmt | EvalStmt], span: Span) -> None:
        """Create an operation statements.

        Args:
            stmts: List of assignment and/or evaluation statements
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

class EvalStmt(Stmt):
    """Evaluation statement: expr."""

    expr: Final[Expr]
    """Expression."""

    def __init__(self, expr: Expr, span: Span) -> None:
        """Create an evaluation statement.

        Args:
            expr: Expression to execute
            span: Source location
        """

class BreakStmt(Stmt):
    """Break statement: break."""

    def __init__(self, span: Span) -> None:
        """Create a break statement.

        Args:
            span: Source location
        """

class ContinueStmt(Stmt):
    """Continue statement: continue."""

    def __init__(self, span: Span) -> None:
        """Create a continue statement.

        Args:
            span: Source location
        """

class Function(IRNode):
    """Function definition with name, parameters, return types, and body."""

    name: Final[str]
    """Function name."""

    func_type: Final[FunctionType]
    """Function type (orchestration, incore, or opaque)."""

    params: Final[list[Var]]
    """Parameter variables."""

    return_types: Final[list[Type]]
    """Return types."""

    body: Final[Stmt]
    """Function body statement (use SeqStmts for multiple statements)."""

    def __init__(
        self,
        name: str,
        params: Sequence[Var],
        return_types: Sequence[Type],
        body: Stmt,
        span: Span,
        type: FunctionType = FunctionType.Opaque,
    ) -> None:
        """Create a function definition.

        Args:
            name: Function name
            params: Parameter variables
            return_types: Return types
            body: Function body statement (use SeqStmts for multiple statements)
            span: Source location
            type: Function type (default: Opaque)
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

@overload
def memref_init(func: Function) -> Function: ...
@overload
def memref_init(program: Program) -> Program: ...
def memref_init(func_or_program: Function | Program) -> Function | Program:
    """Initialize MemRef for all Tile/Tensor variables.

    Creates default MemRef objects for variables with TileType or TensorType
    that don't already have a MemRef attached.

    Default memory space allocation strategy:
    - TileType → MemorySpace.UB (Unified Buffer)
    - TensorType → MemorySpace.DDR (DDR memory)

    Args:
        func_or_program: Function or Program to transform

    Returns:
        Transformed Function or Program with MemRef initialized

    Example:
        >>> func = ... # Create function with Tile/Tensor variables
        >>> func_with_memref = ir.memref_init(func)
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
    kwargs: Mapping[str, int | bool | str | float | DataType | MemorySpace],
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

# ========== Op Conversion Registry ==========

def register_op_conversion(from_op: str, to_op: str) -> None:
    """Register a simple tensor-to-block op name mapping.

    Args:
        from_op: Source op name (e.g., 'tensor.add')
        to_op: Target op name (e.g., 'block.add')
    """

def register_op_conversion_custom(from_op: str, func: object) -> None:
    """Register a custom conversion function for a tensor op.

    Args:
        from_op: Source op name
        func: Callable(args, kwargs, span) -> Expr | tuple[list[Stmt], Expr]
    """

def has_op_conversion(op_name: str) -> bool:
    """Check if a conversion rule exists for an operator.

    Args:
        op_name: The operator name to check
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
    def begin_function(self, name: str, span: Span, type: FunctionType = FunctionType.Opaque) -> None:
        """Begin building a function.

        Args:
            name: Function name
            span: Source location for function definition
            type: Function type (default: Opaque)
        """

    def func_arg(self, name: str, type: Type, span: Span) -> Var:
        """Add a function parameter.

        Args:
            name: Parameter name
            type: Parameter type
            span: Source location for parameter

        Returns:
            Variable representing the parameter
        """

    def return_type(self, type: Type) -> None:
        """Add a return type to the current function.

        Args:
            type: Return type
        """

    def end_function(self, end_span: Span) -> Function:
        """End building a function.

        Args:
            end_span: Source location for end of function

        Returns:
            The built function
        """

    # For loop building
    def begin_for_loop(
        self,
        loop_var: Var,
        start: Expr,
        stop: Expr,
        step: Expr,
        span: Span,
        kind: ForKind = ForKind.Sequential,
    ) -> None:
        """Begin building a for loop.

        Args:
            loop_var: Loop variable
            start: Start value expression
            stop: Stop value expression
            step: Step value expression
            span: Source location for loop definition
            kind: Loop kind (default: Sequential)
        """

    def add_iter_arg(self, iter_arg: IterArg) -> None:
        """Add an iteration argument to the current for loop.

        Args:
            iter_arg: Iteration argument with initial value
        """

    def add_return_var(self, var: Var) -> None:
        """Add a return variable to the current for loop.

        Args:
            var: Return variable
        """

    def end_for_loop(self, end_span: Span) -> ForStmt:
        """End building a for loop.

        Args:
            end_span: Source location for end of loop

        Returns:
            The built for statement
        """

    # While loop building
    def begin_while_loop(self, condition: Expr, span: Span) -> None:
        """Begin building a while loop.

        Creates a new while loop context. Must be closed with end_while_loop().

        Args:
            condition: Condition expression
            span: Source location for loop definition
        """

    def add_while_iter_arg(self, iter_arg: IterArg) -> None:
        """Add an iteration argument to the current while loop.

        Iteration arguments are loop-carried values (SSA-style).

        Args:
            iter_arg: Iteration argument with initial value
        """

    def add_while_return_var(self, var: Var) -> None:
        """Add a return variable to the current while loop.

        Return variables capture the final values of iteration arguments.

        Args:
            var: Return variable
        """

    def set_while_loop_condition(self, condition: Expr) -> None:
        """Set the condition for the current while loop.

        Used to update the loop condition after setting up iter_args. This allows
        the condition to reference iter_arg variables that are defined in the loop.

        Args:
            condition: New condition expression
        """

    def end_while_loop(self, end_span: Span) -> WhileStmt:
        """End building a while loop.

        Finalizes the loop and returns it.

        Args:
            end_span: Source location for end of loop

        Returns:
            The built while statement
        """

    # If statement building
    def begin_if(self, condition: Expr, span: Span) -> None:
        """Begin building an if statement.

        Args:
            condition: Condition expression
            span: Source location for if statement
        """

    def begin_else(self, span: Span) -> None:
        """Begin the else branch of the current if statement.

        Args:
            span: Source location for else keyword
        """

    def add_if_return_var(self, var: Var) -> None:
        """Add a return variable to the current if statement.

        Args:
            var: Return variable
        """

    def end_if(self, end_span: Span) -> IfStmt:
        """End building an if statement.

        Args:
            end_span: Source location for end of if

        Returns:
            The built if statement
        """

    # Scope building
    def begin_scope(self, scope_kind: ScopeKind, span: Span) -> None:
        """Begin building a scope statement.

        Args:
            scope_kind: The kind of scope (e.g., ScopeKind.InCore)
            span: Source location for scope statement
        """

    def end_scope(self, end_span: Span) -> ScopeStmt:
        """End building a scope statement.

        Args:
            end_span: Source location for end of scope

        Returns:
            The built scope statement
        """

    # Program building
    def begin_program(self, name: str, span: Span) -> None:
        """Begin building a program.

        Args:
            name: Program name
            span: Source location for program definition
        """

    def declare_function(self, func_name: str) -> GlobalVar:
        """Declare a function and get its GlobalVar for cross-function calls.

        Args:
            func_name: Function name to declare

        Returns:
            GlobalVar that can be used in Call expressions
        """

    def get_global_var(self, func_name: str) -> GlobalVar:
        """Get GlobalVar for a declared function.

        Args:
            func_name: Function name

        Returns:
            GlobalVar for the function
        """

    def add_function(self, func: Function) -> None:
        """Add a completed function to the current program.

        Args:
            func: Function to add
        """

    def end_program(self, end_span: Span) -> Program:
        """End building a program.

        Args:
            end_span: Source location for end of program

        Returns:
            The built program
        """

    def get_function_return_types(self, gvar: GlobalVar) -> list[Type]:
        """Get return types for a function by its GlobalVar.

        Returns the return types for a function if it has been added to the program.
        Returns empty list if not inside a program or function not yet added.

        Args:
            gvar: GlobalVar for the function

        Returns:
            Vector of return types
        """

    # Statement recording
    def emit(self, stmt: Stmt) -> None:
        """Emit a statement in the current context.

        Args:
            stmt: Statement to emit
        """

    def assign(self, var: Var, value: Expr, span: Span) -> AssignStmt:
        """Create an assignment statement and emit it.

        Args:
            var: Variable to assign to
            value: Expression value
            span: Source location for assignment

        Returns:
            The created assignment statement
        """

    def var(self, name: str, type: Type, span: Span) -> Var:
        """Create a variable (does not emit).

        Args:
            name: Variable name
            type: Variable type
            span: Source location

        Returns:
            The created variable
        """

    @overload
    def return_(self, values: list[Expr], span: Span) -> ReturnStmt:
        """Create a return statement and emit it.

        Args:
            values: List of expressions to return
            span: Source location for return statement

        Returns:
            The created return statement
        """

    @overload
    def return_(self, span: Span) -> ReturnStmt:
        """Create an empty return statement and emit it.

        Args:
            span: Source location for return statement

        Returns:
            The created return statement
        """

    # Context state queries
    def in_function(self) -> bool:
        """Check if currently inside a function.

        Returns:
            True if inside a function context
        """

    def in_loop(self) -> bool:
        """Check if currently inside a for loop.

        Returns:
            True if inside a for loop context
        """

    def in_if(self) -> bool:
        """Check if currently inside an if statement.

        Returns:
            True if inside an if statement context
        """

    def in_program(self) -> bool:
        """Check if currently inside a program.

        Returns:
            True if inside a program context
        """

class ProgramBuilder:
    """Helper for building programs within a program context.

    This class is used as a context manager helper for IRBuilder.program().
    It provides methods for declaring functions, managing GlobalVars, and
    constructing the final Program.
    """

    def declare_function(self, name: str) -> GlobalVar:
        """Declare a function and get its GlobalVar for cross-function calls.

        This should be called before building the function to enable other
        functions to reference it via Call expressions.

        Args:
            name: Function name to declare

        Returns:
            GlobalVar that can be used in Call expressions
        """

    def get_global_var(self, name: str) -> GlobalVar:
        """Get GlobalVar for a declared function.

        Args:
            name: Function name

        Returns:
            GlobalVar for the function

        Raises:
            RuntimeError: If function not declared
        """

    def add_function(self, func: Function) -> None:
        """Add a function to the program.

        The function name must match a previously declared function name.

        Args:
            func: Function to add
        """

    def get_result(self) -> Program:
        """Get the built Program.

        Returns:
            The completed program IR node

        Raises:
            AssertionError: If called before program is complete
        """

# ========== Python Printer ==========
def python_print(node: IRNode, prefix: str = "pl") -> str:
    """Print an IR node as a Python string.

    Args:
        node: IR node to print
        prefix: Module prefix (default 'pl' for 'import pypto.language as pl')

    Returns:
        String representation of the IR node
    """

def python_print_type(type: Type, prefix: str = "pl") -> str:
    """Print a Type object as a Python string.

    Args:
        type: Type object to print
        prefix: Module prefix (default 'pl' for 'import pypto.language as pl')

    Returns:
        String representation of the Type
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

class ParentStmtAnalysis:
    """Utility class for analyzing parent-child relationships in statement trees.

    This class builds a mapping from each statement to its parent statement within
    a function's body. It is useful for passes that need to traverse upward in the
    IR tree or understand the context of a statement.

    Example usage:
        analysis = ir.ParentStmtAnalysis()
        analysis.build_map(function)
        parent = analysis.get_parent(some_stmt)
        if parent:
            # Use parent statement

    Note: The analysis becomes invalid after IR transformations. Call build_map again
    if the IR tree is modified.
    """

    def __init__(self) -> None:
        """Create a ParentStmtAnalysis instance."""

    def build_map(self, func: Function) -> None:
        """Build the parent mapping from a function's body.

        Traverses the function's statement tree and records parent-child relationships.
        This method clears any existing mapping before building the new one.

        Args:
            func: The function to analyze

        Parent relationships established:
        - For SeqStmts/OpStmts: Each child statement's parent is the SeqStmts/OpStmts
        - For IfStmt: then_body and else_body (if present) have IfStmt as parent
        - For ForStmt: body has ForStmt as parent
        - Root statement (function.body) has no parent
        """

    def get_parent(self, stmt: Stmt) -> Stmt | None:
        """Get the parent statement of a given statement.

        Args:
            stmt: The statement to query

        Returns:
            Parent statement, or None if:
            - stmt is the root statement (function body)
            - stmt is not found in the analyzed tree
            - stmt is None
        """

    def has_parent(self, stmt: Stmt) -> bool:
        """Check if a statement has a recorded parent.

        Args:
            stmt: The statement to check

        Returns:
            True if stmt has a parent in the map, False otherwise
        """

    def clear(self) -> None:
        """Clear the parent mapping.

        Removes all recorded parent-child relationships. Useful for reusing
        the same ParentStmtAnalysis instance with different functions.
        """
