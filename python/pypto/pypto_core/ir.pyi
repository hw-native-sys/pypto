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
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, overload

from pypto import DataType

if TYPE_CHECKING:
    from pypto.language.typing.scalar import Scalar

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

    def as_python(self, prefix: str = "pl", concise: bool = False, format: bool = True) -> str:
        """Convert to Python-style string representation.

        Args:
            prefix: Module prefix (default 'pl' for 'import pypto.language as pl')
            concise: If true, omit intermediate type annotations (default false)
            format: If true, apply registered format callback (default true)

        Returns:
            Python-style string representation
        """

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

    def __eq__(self, other: object) -> bool:
        """Structural equality comparison (delegates to structural_equal)."""

    def __hash__(self) -> int:
        """Structural hash (consistent with __eq__)."""

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

    @property
    def memory_space(self) -> MemorySpace | None:
        """Canonical memory space for this shaped type."""
        ...

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

class TileLayout(enum.Enum):
    """Tile layout enumeration (shared by blayout and slayout)."""

    none_box = ...
    """No layout constraint."""

    row_major = ...
    """Row-major layout."""

    col_major = ...
    """Column-major layout."""

class PadValue(enum.Enum):
    """Tile pad mode enumeration."""

    null = ...
    """No padding."""

    zero = ...
    """Zero padding."""

    max = ...
    """Max value padding."""

    min = ...
    """Min value padding."""

class TensorView:
    """Tensor view representation with stride, layout, valid shape, and pad mode."""

    stride: Sequence[Expr]
    """Stride for each dimension."""

    layout: TensorLayout
    """Tensor layout type."""

    valid_shape: Sequence[Expr]
    """Valid shape for each dimension (empty means use full shape)."""

    pad: PadValue
    """Pad mode for out-of-valid-shape accesses (default PadValue.null)."""

    @overload
    def __init__(self) -> None:
        """Create an empty tensor view with default ND layout."""

    @overload
    def __init__(
        self,
        stride: Sequence[Expr | int | Scalar],
        layout: TensorLayout,
        valid_shape: Sequence[Expr | int | Scalar] = ...,
        pad: PadValue = ...,
    ) -> None:
        """Create a tensor view with stride, layout, optional valid shape, and optional pad.

        Args:
            stride: Stride for each dimension (Expr, int, or Scalar/DynVar)
            layout: Tensor layout type (ND, DN, or NZ)
            valid_shape: Valid shape for each dimension (optional, defaults to empty)
            pad: Pad mode for out-of-valid-shape accesses (defaults to PadValue.null)
        """

class _TensorViewSemanticsModule:
    """Canonical-form helpers for TensorType.tensor_view_ (RFC #1300)."""

    @staticmethod
    def build_logical_strides_from_layout(shape: Sequence[Expr], layout: TensorLayout) -> list[Expr]:
        """Build packed canonical strides for (shape, layout).

        Raises:
            ValueError: if layout is NZ, or DN with rank < 2.
        """

    @staticmethod
    def derive_layout_from_strides(shape: Sequence[Expr], stride: Sequence[Expr]) -> TensorLayout | None:
        """Statically derive layout from (shape, stride).

        Returns ``None`` for symbolic / non-canonical cases.
        """

    @staticmethod
    def check_canonical_view(
        shape: Sequence[Expr],
        stride: Sequence[Expr],
        layout: TensorLayout,
        relaxed_symbolic: bool = True,
    ) -> tuple[bool, str]:
        """Verify (shape, stride, layout) is canonical. Returns (ok, reason)."""

    @staticmethod
    def is_canonical_view(
        shape: Sequence[Expr],
        stride: Sequence[Expr],
        layout: TensorLayout,
        relaxed_symbolic: bool = True,
    ) -> bool:
        """Convenience wrapper around check_canonical_view returning only the ok flag."""

    @staticmethod
    def canonicalize_view(shape: Sequence[Expr], layout: TensorLayout) -> TensorView:
        """Build a packed canonical TensorView for (shape, layout)."""

    @staticmethod
    def compute_shape_product(shape: Sequence[Expr]) -> int:
        """Static product of shape dimensions; -1 if any dim is dynamic."""

tensor_view_semantics: _TensorViewSemanticsModule
"""Submodule with canonical-form helpers for TensorType.tensor_view_ (RFC #1300)."""

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

class DistributedTensorType(TensorType):
    """Tensor backed by a per-rank slice of a CommGroup HCCL window buffer.

    Subclass of :class:`TensorType` distinguished only by ``ObjectKind`` so
    that verifiers for cross-rank ops can reject plain :class:`TensorType`
    arguments. Note ``As<TensorType>`` does NOT match
    ``DistributedTensorType`` (precise ObjectKind match) — use
    ``As<DistributedTensorType>`` to dispatch on the distributed variant.

    Mirrors :class:`TensorType`'s constructor surface — memref / tensor_view
    are equally valid on the distributed flavour. Additionally carries an
    optional ``window_buffer`` back-reference (populated by ``pld.window``)
    so that two same-shape / same-dtype slices of different
    :class:`WindowBuffer` allocations stay structurally distinct.
    """

    window_buffer: Final[WindowBuffer | None]
    """Source :class:`WindowBuffer` for slices materialised via ``pld.window``;
    ``None`` for user-declared parameter annotations such as
    ``pld.DistributedTensor[[N], pl.FP32]``."""

    @overload
    def __init__(self, shape: Sequence[Expr], dtype: DataType) -> None:
        """Create a distributed tensor type."""

    @overload
    def __init__(self, shape: Sequence[Expr], dtype: DataType, memref: MemRef | None) -> None:
        """Create a distributed tensor type with optional memref."""

    @overload
    def __init__(
        self,
        shape: Sequence[Expr],
        dtype: DataType,
        memref: MemRef | None,
        tensor_view: TensorView | None,
    ) -> None:
        """Create a distributed tensor type with optional memref and tensor_view."""

    @overload
    def __init__(self, shape: Sequence[int], dtype: DataType) -> None:
        """Create a distributed tensor type with constant shape."""

    @overload
    def __init__(self, shape: Sequence[int], dtype: DataType, memref: MemRef | None) -> None:
        """Create a distributed tensor type with constant shape and optional memref."""

    @overload
    def __init__(
        self,
        shape: Sequence[int],
        dtype: DataType,
        memref: MemRef | None,
        tensor_view: TensorView | None,
    ) -> None:
        """Create a distributed tensor type with constant shape, optional memref and tensor_view."""

    @overload
    def __init__(self, shape: Sequence[Expr], dtype: DataType, window_buffer: WindowBuffer) -> None:
        """Create a distributed tensor type bound to a specific WindowBuffer (produced by ``pld.window``)."""

class TileView:
    """Tile view: read-only representation of valid shape, stride, start offset,
    layouts, fractal, and pad. Construct with all values; fields cannot be mutated
    after construction (so hash/equality stay stable for use as set/dict keys)."""

    valid_shape: Final[Sequence[Expr]]
    """Valid shape dimensions."""

    stride: Final[Sequence[Expr]]
    """Stride for each dimension."""

    start_offset: Final[Expr | None]
    """Starting offset (None for an empty tile view)."""

    blayout: Final[TileLayout]
    """Block layout."""

    slayout: Final[TileLayout]
    """Scatter layout."""

    fractal: Final[int]
    """Fractal size."""

    pad: Final[PadValue]
    """Pad mode."""

    def __init__(
        self,
        valid_shape: Sequence[Expr | int | Scalar] | None = None,
        stride: Sequence[Expr | int | Scalar] | None = None,
        start_offset: Expr | int | Scalar | None = None,
        blayout: TileLayout = ...,
        slayout: TileLayout = ...,
        fractal: int = ...,
        pad: PadValue = ...,
    ) -> None:
        """Create a tile view; all fields default to empty/None/row_major/none_box/512/null.

        Args:
            valid_shape: Valid shape dimensions (Expr/int/Scalar, ints auto-converted to ConstInt)
            stride: Stride for each dimension (Expr/int/Scalar, ints auto-converted to ConstInt)
            start_offset: Starting offset (Expr/int/Scalar, int auto-converted to ConstInt; None allowed)
            blayout: Block layout (default: row_major)
            slayout: Scatter layout (default: none_box)
            fractal: Fractal size (default: 512)
            pad: Pad mode (default: null)
        """

    def __eq__(self, other: object) -> bool:
        """Structural equality comparison."""

    def __ne__(self, other: object) -> bool:
        """Structural inequality comparison."""

    def __hash__(self) -> int:
        """Hash consistent with __eq__ (ConstInt fields by value, others by pointer identity)."""

class TileType(ShapedType):
    """Tile type representation (multi-dimensional tensor)."""

    tile_view: Final[TileView | None]
    """Optional tile view information."""

    memory_space: Final[MemorySpace | None]
    """Memory space (None = not yet inferred)."""

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
        self,
        shape: Sequence[Expr],
        dtype: DataType,
        memref: MemRef | None,
        tile_view: TileView | None,
        memory_space: MemorySpace | None = None,
    ) -> None:
        """Create a tile type with memory reference, tile view, and memory space.

        Args:
            shape: Shape dimensions as Expr nodes (supports multi-dimensional tensors)
            dtype: Element data type
            memref: Optional memory reference
            tile_view: Optional tile view information
            memory_space: Optional memory space
        """

    @overload
    def __init__(self, shape: Sequence[int], dtype: DataType) -> None:
        """Create a tile type without memory reference.

        Args:
            shape: Shape dimensions as integers (automatically converted to ConstInt)
            dtype: Element data type
        """

    @overload
    def __init__(self, shape: Sequence[int], dtype: DataType, memref: MemRef | None) -> None:
        """Create a tile type with memory reference.

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
        tile_view: TileView | None,
        memory_space: MemorySpace | None = None,
    ) -> None:
        """Create a tile type with memory reference, tile view, and memory space.

        Args:
            shape: Shape dimensions as integers (automatically converted to ConstInt)
            dtype: Element data type
            memref: Optional memory reference
            tile_view: Optional tile view information
            memory_space: Optional memory space
        """

    def get_effective_tile_view(self) -> TileView:
        """Return the effective TileView for this tile.

        If ``tile_view`` is set explicitly, returns that. Otherwise returns the
        implicit view derived from ``(shape, memory_space)``. Under canonicalization
        an implicit view is stored as ``None``, so callers that need to inspect
        layout fields should use this method rather than reading ``tile_view``
        directly.
        """

class ArrayType(ShapedType):
    """On-core array type (lives on scalar register file / C stack).

    Homogeneous, 1-D, fixed extent, integer dtype. Cannot cross function
    boundaries — see the ArrayNotEscapedVerifier for the rule.
    """

    @overload
    def __init__(self, dtype: DataType, extent: Expr) -> None:
        """Create an ArrayType from a dtype and a ConstInt extent."""

    @overload
    def __init__(self, dtype: DataType, extent: int) -> None:
        """Create an ArrayType from a dtype and an int extent."""

    @property
    def extent(self) -> Expr:
        """Number of elements (always a ConstInt)."""

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
    - InCore: Sub-graph on specific AICore (unspecialized)
    - AIC: Cube core kernel (specialized InCore)
    - AIV: Vector core kernel (specialized InCore)
    - Group: Co-scheduled group of AIC + AIV kernels
    - Spmd: SPMD data-parallel dispatch
    - Inline: Whole-body substitution at every call site by the
      InlineFunctions pass (eliminated before any other pass runs)
    """

    Opaque = ...
    """Unspecified function type (default)."""

    Orchestration = ...
    """Host/AICPU control and coordination."""

    InCore = ...
    """AICore sub-graph execution (unspecialized)."""

    AIC = ...
    """Cube core kernel (specialized InCore)."""

    AIV = ...
    """Vector core kernel (specialized InCore)."""

    Group = ...
    """Co-scheduled group of AIC + AIV kernels."""

    Spmd = ...
    """SPMD data-parallel dispatch."""

    Inline = ...
    """Whole-body substitution at every call site."""

class Level(enum.Enum):
    """Hierarchy level in the Linqu machine model.

    Levels map bottom-up from individual cores (Level 0) to the global
    coordinator. Alias values resolve to the same level as their primary name.
    """

    AIV = ...
    """Single AIV (Vector) core."""

    AIC = ...
    """Single AIC (Cube) core."""

    CORE_GROUP = ...
    """Core-group (e.g. 1 AIC + 2 AIV)."""

    CHIP_DIE = ...
    """Chip die (optional in single-die models)."""

    CHIP = ...
    """Chip (UMA)."""

    HOST = ...
    """Host (single OS instance)."""

    CLUSTER_0 = ...
    """Cluster-level-0 (pod)."""

    CLUSTER_1 = ...
    """Cluster-level-1 (supernode)."""

    CLUSTER_2 = ...
    """Cluster-level-2 (cross-rack)."""

    GLOBAL = ...
    """Global coordinator."""

    # Readability aliases
    L2CACHE = ...
    """Alias for CHIP_DIE."""

    PROCESSOR = ...
    """Alias for CHIP."""

    UMA = ...
    """Alias for CHIP."""

    NODE = ...
    """Alias for HOST."""

    POD = ...
    """Alias for CLUSTER_0."""

    CLOS1 = ...
    """Alias for CLUSTER_1."""

    CLOS2 = ...
    """Alias for CLUSTER_2."""

class Role(enum.Enum):
    """Function role at L3-L7 hierarchy levels.

    Distinguishes orchestrators (which build task DAGs and submit work)
    from sub-workers (which execute concrete compute or data tasks
    dispatched by an orchestrator at the same level).
    """

    Orchestrator = ...
    """Builds DAG, submits tasks, never computes directly."""

    SubWorker = ...
    """Executes compute/data tasks dispatched by the orchestrator at the same level."""

class ParamDirection(enum.Enum):
    """Parameter direction classification.

    Models kernel-style parameter directions:
    - In: Read-only input parameter (default)
    - Out: Write-only output parameter
    - InOut: Read-write parameter
    """

    In = ...
    """Read-only input (default)."""

    Out = ...
    """Write-only output."""

    InOut = ...
    """Read-write input/output."""

class ArgDirection(enum.Enum):
    """Call-site argument direction (mirrors runtime TensorArgType one-to-one).

    Distinct from `ParamDirection` (callee function-signature contract).
    `ArgDirection` describes the call-site task-submission behavior:
    which runtime API to invoke, what dependency edges to establish,
    and whether the buffer is allocated by the runtime.
    """

    Input = ...
    """Read-only input → add_input / TensorArgType::INPUT."""

    Output = ...
    """Runtime-allocated output buffer → add_output(create_info) / TensorArgType::OUTPUT."""

    InOut = ...
    """Read-then-write → add_inout / TensorArgType::INOUT."""

    OutputExisting = ...
    """Write-only into an existing tensor → add_output(tensor) / TensorArgType::OUTPUT_EXISTING."""

    NoDep = ...
    """No-dependency existing tensor → add_no_dep / TensorArgType::NO_DEP."""

    Scalar = ...
    """Scalar (non-tensor) argument → add_scalar."""

class ForKind(enum.Enum):
    """For loop kind classification.

    Distinguishes sequential, parallel, unroll, and pipeline for loops:
    - Sequential: Standard sequential for loop (default)
    - Parallel: Parallel for loop
    - Unroll: Compile-time unrolled for loop
    - Pipeline: Software-pipelined loop (transient marker; stripped by CanonicalizeIOOrder)
    """

    Sequential = ...
    """Standard sequential for loop (default)."""

    Parallel = ...
    """Parallel for loop."""

    Unroll = ...
    """Compile-time unrolled for loop."""

    Pipeline = ...
    """Software-pipelined loop — lowered by ``LowerPipelineLoops``. The kind
    persists as a marker through ``CanonicalizeIOOrder`` (which demotes it to
    ``Sequential`` on exit) and must not survive past that pass."""

class ChunkPolicy(enum.Enum):
    """Chunk policy for loop chunking.

    Controls how iterations are distributed across chunks.
    """

    LeadingFull = ...
    """Full chunks first, smaller remainder at end (splits into main + remainder kernels)."""

    Guarded = ...
    """Single loop over ceil(N/C) chunks with per-iteration if-guard (default)."""

class ChunkConfig:
    """Chunk configuration for parallel loop splitting."""

    size: Final[Expr]
    """Chunk size expression."""

    policy: Final[ChunkPolicy]
    """Chunk distribution policy."""

    def __init__(self, size: Expr, policy: ChunkPolicy = ChunkPolicy.Guarded) -> None:
        """Create a chunk configuration.

        Args:
            size: Chunk size expression
            policy: Chunk distribution policy (default: Guarded)
        """

class LoopOrigin(enum.Enum):
    """Loop origin classification.

    Tracks how a loop was generated:
    - Original: Regular loop (default)
    - ChunkOuter: Outer loop from chunk splitting
    - ChunkInner: Inner loop from chunk splitting
    - ChunkRemainder: Remainder loop from chunk splitting
    """

    Original = ...
    """Regular loop (default)."""

    ChunkOuter = ...
    """Outer loop from chunk splitting."""

    ChunkInner = ...
    """Inner loop from chunk splitting."""

    ChunkRemainder = ...
    """Remainder loop from chunk splitting."""

class MemorySpace(enum.Enum):
    """Memory space enumeration."""

    DDR = ...
    """DDR memory (off-chip)."""

    Vec = ...
    """Vector/unified buffer (on-chip)."""

    Mat = ...
    """Matrix/L1 buffer."""

    Left = ...
    """Left matrix operand buffer."""

    Right = ...
    """Right matrix operand buffer."""

    Acc = ...
    """Accumulator buffer."""

    Bias = ...
    """Bias buffer."""

    ScalarLocal = ...
    """On-core scalar register file / C stack (ArrayType)."""

Mem = MemorySpace
"""Short alias for MemorySpace (e.g., Mem.Vec instead of MemorySpace.Vec)."""

class PtrType(Type):
    """Pointer type for allocation identity tokens (returned by tile.alloc/tensor.alloc)."""

    def __init__(self) -> None: ...
    @staticmethod
    def get() -> PtrType:
        """Get the singleton PtrType instance."""
        ...

class WindowBufferType(Type):
    """Singleton marker type for ``pld.alloc_window_buffer`` outputs.

    Mirrors :class:`MemRefType`: the type carries no per-instance fields. All
    allocation metadata (name, size, dtype, host-staging flags) lives on the
    companion :class:`WindowBuffer` Var subclass.
    """

    def __init__(self) -> None: ...
    @staticmethod
    def get() -> WindowBufferType:
        """Get the shared singleton WindowBufferType instance."""
        ...

class MemRef(Var):
    """Memory reference variable for shaped types (inherits from Var)."""

    base_: Var
    """Base Ptr variable (allocation identity token)."""

    byte_offset_: Expr
    """Byte offset from base (0 for root alloc, computed for views)."""

    size_: int
    """Size in bytes (64-bit unsigned)."""

    @overload
    def __init__(self, base: Var, byte_offset: int, size: int, span: Span = ...) -> None: ...
    @overload
    def __init__(self, base: Var, byte_offset: Expr, size: int, span: Span = ...) -> None: ...
    @overload
    def __init__(self, base: str, byte_offset: int, size: int, span: Span = ...) -> None: ...
    @overload
    def __init__(self, addr: int, size: int, id: int, span: Span = ...) -> None: ...
    @overload
    def __init__(
        self, memory_space: MemorySpace, addr: Expr | int, size: int, id: int, span: Span = ...
    ) -> None: ...
    def __init__(self, *args, **kwargs) -> None:
        """Create a memory reference.

        New API: MemRef(base, byte_offset, size)
        Legacy API: MemRef(memory_space, addr, size, id) or MemRef(addr, size, id)
        """

    @staticmethod
    def same_allocation(a: MemRef, b: MemRef) -> bool:
        """Check if two MemRefs share the same allocation (same base_ Ptr)."""
        ...

    @staticmethod
    def may_alias(a: MemRef, b: MemRef) -> bool:
        """Check if two MemRefs may alias (same base + overlapping byte ranges)."""
        ...

DYNAMIC_DIM: Final[int]
"""Constant representing a dynamic dimension (value: -1).

Used to indicate dimensions with runtime-determined sizes.
"""

ScalarExprType = Expr | int | float

class Var(Expr):
    """Variable reference expression."""

    name_hint: Final[str]
    """Variable name hint (cosmetic label, not an identifier)."""

    def __init__(self, name_hint: str, type: Type, span: Span) -> None:
        """Create a variable reference.

        Args:
            name_hint: Variable name hint (cosmetic label, not an identifier)
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

    def __init__(self, name_hint: str, type: Type, initValue: Expr, span: Span) -> None:
        """Create an iteration argument.

        Args:
            name_hint: Variable name hint (cosmetic label, not an identifier)
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

    @property
    def arg_directions(self) -> Sequence[ArgDirection]:
        """Resolved per-argument call-site directions.

        Stored under ``attrs['arg_directions']``. Returns an empty list when
        not yet derived (legacy / pre-DeriveCallDirections state). When non-empty
        the length must match ``args`` and is the source of truth for runtime
        task-submission semantics.
        """

    @property
    def attrs(self) -> Mapping[str, Any]:
        """Compiler-internal node metadata.

        Reserved keys:

        - ``"arg_directions"`` -> ``list[ArgDirection]`` (use the
          :attr:`arg_directions` shortcut for typed access).
        """

    kwargs: Final[Mapping[str, int | bool | str | float | DataType | MemorySpace | PadValue]]
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
        kwargs: Mapping[str, int | bool | str | float | DataType | MemorySpace | PadValue],
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
        kwargs: Mapping[str, int | bool | str | float | DataType | MemorySpace | PadValue],
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

    @overload
    def __init__(
        self,
        op: Op,
        args: Sequence[Expr],
        kwargs: Mapping[str, int | bool | str | float | DataType | MemorySpace | PadValue],
        attrs: Mapping[str, object] | Sequence[tuple[str, object]] | None,
        type: Type,
        span: Span,
    ) -> None:
        """Create a function call expression with kwargs, explicit attrs and type.

        Args:
            op: Operation/function to call
            args: List of argument expressions
            kwargs: Keyword arguments (metadata)
            attrs: Compiler-internal node metadata. Reserved keys:
                ``"arg_directions"`` -> ``Sequence[ArgDirection]``.
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

    leading_comments: Final[list[str]]
    """Source-level comments printed above this statement.

    IgnoreField metadata — never participates in structural_equal or hashing.
    Read-only from Python; use :func:`attach_leading_comments` to modify.
    """

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
    """Loop kind (Sequential, Parallel, or Unroll)."""

    chunk_config: Final[ChunkConfig | None]
    """Chunk configuration (None = no chunking)."""

    chunk_size: Final[Expr | None]
    """Chunk size expression (None if no chunking). Convenience for chunk_config.size."""

    chunk_policy: Final[ChunkPolicy]
    """Chunk distribution policy. Convenience for chunk_config.policy."""

    attrs: Final[dict[str, object]]
    """Loop-level attributes (key-value metadata)."""

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
        chunk_size: Expr | None = None,
        chunk_policy: ChunkPolicy = ChunkPolicy.Guarded,
        attrs: dict[str, object] | list[tuple[str, object]] | None = None,
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
            chunk_size: Optional chunk size for loop chunking
            chunk_policy: Chunk distribution policy (default: Guarded)
            attrs: Loop-level attributes (default: empty)
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

    AutoInCore = 1
    """AutoInCore scope for automatic chunking."""

    Cluster = 2
    """Cluster scope for co-scheduled AIC + AIV groups."""

    Hierarchy = 3
    """Distributed hierarchy scope (uses level/role on ScopeStmt)."""

    Spmd = 4
    """SPMD dispatch scope (core_num/sync_start on ScopeStmt)."""

    Runtime = 5
    """Runtime orchestration scope (PTO2_SCOPE wrapper, manual on/off)."""

class SplitMode(enum.Enum):
    """Split mode for cross-core data transfer."""

    NONE = 0
    """No split."""

    UP_DOWN = 1
    """Split vertically (height halved)."""

    LEFT_RIGHT = 2
    """Split horizontally (width halved)."""

class ScopeStmt(Stmt):
    """Scope statement: marks a region with specific execution context (abstract base).

    Abstract — instantiate one of the concrete subclasses below.
    """

    scope_kind: Final[ScopeKind]
    """The kind of scope (discriminator)."""

    name_hint: Final[str]
    """User-provided scope name hint (empty string = auto-generate)."""

    body: Final[Stmt]
    """The nested statements."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """ScopeStmt is abstract — construct an InCoreScopeStmt, AutoInCoreScopeStmt,
        ClusterScopeStmt, HierarchyScopeStmt, or SpmdScopeStmt instead."""

class InCoreScopeStmt(ScopeStmt):
    """InCore scope: AICore sub-graph region."""

    split: Final[SplitMode | None]
    """Split mode for cross-core transfer (None or SplitMode.None for no split)."""

    def __init__(
        self,
        split: SplitMode | None = None,
        name_hint: str = "",
        *,
        body: Stmt,
        span: Span,
    ) -> None:
        """Create an InCore scope statement."""

class AutoInCoreScopeStmt(ScopeStmt):
    """AutoInCore scope: InCore region with automatic chunking."""

    split: Final[SplitMode | None]
    """Split mode for cross-core transfer (None or SplitMode.None for no split)."""

    def __init__(
        self,
        split: SplitMode | None = None,
        name_hint: str = "",
        *,
        body: Stmt,
        span: Span,
    ) -> None:
        """Create an AutoInCore scope statement."""

class ClusterScopeStmt(ScopeStmt):
    """Cluster scope: co-scheduled AIC + AIV group."""

    def __init__(self, name_hint: str = "", *, body: Stmt, span: Span) -> None:
        """Create a Cluster scope statement."""

class HierarchyScopeStmt(ScopeStmt):
    """Hierarchy scope: distributed-hierarchy region."""

    level: Final[Level]
    """Hierarchy level (required)."""

    role: Final[Role | None]
    """Function role (Orchestrator or Worker; None for unspecified)."""

    def __init__(
        self,
        level: Level,
        role: Role | None = None,
        name_hint: str = "",
        *,
        body: Stmt,
        span: Span,
    ) -> None:
        """Create a Hierarchy scope statement."""

class SpmdScopeStmt(ScopeStmt):
    """SPMD dispatch scope."""

    core_num: Final[Expr]
    """SPMD block count expression."""

    sync_start: Final[bool]
    """Require sync-start for SPMD dispatch."""

    def __init__(
        self,
        core_num: int | Expr,
        sync_start: bool = False,
        name_hint: str = "",
        *,
        body: Stmt,
        span: Span,
    ) -> None:
        """Create an SPMD scope statement.

        Accepts either a Python ``int`` (auto-wrapped as ``ConstInt``) or any
        ``Expr`` of integer type.
        """

class RuntimeScopeStmt(ScopeStmt):
    """Runtime orchestration scope: a PTO2_SCOPE wrapper at codegen.

    The ``manual`` flag picks between two emission modes:
      - ``manual=False`` → ``PTO2_SCOPE() { ... }`` (auto-dep via TensorMap)
      - ``manual=True``  → ``PTO2_SCOPE(PTO2ScopeMode::MANUAL) { ... }``
        (no auto-dep; compiler emits explicit ``add_dep`` from SSA data flow
        plus user-supplied ``deps=[...]`` on each kernel call)
    """

    manual: Final[bool]
    """True = MANUAL scope; False = AUTO scope (default)."""

    def __init__(
        self,
        manual: bool = False,
        name_hint: str = "",
        *,
        body: Stmt,
        span: Span,
    ) -> None:
        """Create a Runtime scope statement."""

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

    def __getitem__(self, index: int) -> Stmt:
        """Get statement by index, supports negative indexing.

        Args:
            index: Statement index (negative indices count from end)

        Returns:
            Statement at the given index

        Raises:
            IndexError: If index is out of range
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

class InlineLanguage(enum.Enum):
    """Source language carried by an InlineStmt body."""

    Python = 0
    """Python source."""

class InlineStmt(Stmt):
    """Inline statement: opaque source body in a target language.

    Used to embed verbatim source text (e.g. a SubWorker's Python body) directly
    in the IR. Treated as a leaf by passes — no children to traverse.
    """

    body: Final[str]
    """Verbatim source text."""

    language: Final[InlineLanguage]
    """Language of the body string."""

    def __init__(self, body: str, language: InlineLanguage, span: Span) -> None:
        """Create an inline statement.

        Args:
            body: Verbatim source text
            language: Language of the body
            span: Source location
        """

class Function(IRNode):
    """Function definition with name, parameters, return types, and body."""

    name: Final[str]
    """Function name."""

    func_type: Final[FunctionType]
    """Function type (Opaque, Orchestration, InCore, AIC, AIV, or Group)."""

    level: Final[Level | None]
    """Hierarchy level (None = unspecified)."""

    role: Final[Role | None]
    """Function role (None = unspecified)."""

    attrs: Final[dict[str, Any]]
    """Function-level attributes (key-value metadata)."""

    split: Final[SplitMode | None]
    """Split mode for cross-core transfer (convenience accessor into attrs)."""

    params: Final[list[Var]]
    """Parameter variables."""

    param_directions: Final[list[ParamDirection]]
    """Parameter directions corresponding to each parameter."""

    return_types: Final[list[Type]]
    """Return types."""

    body: Final[Stmt]
    """Function body statement (use SeqStmts for multiple statements)."""

    def __init__(
        self,
        name: str,
        params: Sequence[Var | tuple[Var, ParamDirection]],
        return_types: Sequence[Type],
        body: Stmt,
        span: Span,
        type: FunctionType = FunctionType.Opaque,
        level: Level | None = None,
        role: Role | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """Create a function definition.

        Args:
            name: Function name
            params: Parameter variables, either Var (defaults to In) or (Var, ParamDirection) tuples
            return_types: Return types
            body: Function body statement (use SeqStmts for multiple statements)
            span: Source location
            type: Function type (default: Opaque)
            level: Hierarchy level (default: None — unspecified)
            role: Function role (default: None — unspecified)
            attrs: Function-level attributes dict (default: None)
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

    comm_groups: Final[list[CommGroup]]
    """CommGroups declared on the program. Participates in structural
    equality / hashing through reflection."""

    @overload
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

    @overload
    def __init__(
        self,
        functions: list[Function],
        comm_groups: list[CommGroup],
        name: str,
        span: Span,
    ) -> None:
        """Create a program from a list of functions and CommGroup metadata."""

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

    def __getitem__(self, name: str) -> Function | None:
        """Get function by name, returns None if not found.

        Enables copy-paste navigation of structural equality error paths:
            program['main'].body[1].var

        Args:
            name: Function name to look up

        Returns:
            Function if found, None otherwise
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

class WindowBuffer(Var):
    """Per-rank CommGroup HCCL window-buffer allocation, modelled as a specialised Var.

    Mirrors :class:`MemRef`: its SSA-edge type is the singleton
    :class:`WindowBufferType`; allocation metadata (the underlying Ptr Var,
    per-rank size in bytes, host-staging flags) lives on the Var subclass
    itself. The buffer's runtime-unique identifier flows through the
    inherited :attr:`Var.name_hint` (taken from ``base.name_hint``) — there
    is no separate ``name`` field, because two structurally-identical
    allocations should compare equal regardless of the chosen variable name.

    ``WindowBuffer`` instances are constructed by the comm-collection pass;
    at parse time the :func:`alloc_window_buffer` op's LHS is a plain
    :class:`Var` of type :class:`PtrType`.
    """

    base: Final[Var]
    """The underlying ``Ptr`` Var produced by ``pld.alloc_window_buffer``.
    Same role as :attr:`MemRef.base`."""

    size: Final[Expr]
    """Per-rank allocation size in **bytes** (matches
    :attr:`tile.alloc`'s size argument and the runtime's
    ``ChipBufferSpec.nbytes`` field)."""

    load_from_host: Final[bool]
    """``True`` if this slot participates in pre-fork H2D staging."""

    store_to_host: Final[bool]
    """``True`` if this slot participates in post-task D2H staging."""

    def __init__(
        self,
        base: Var,
        size: Expr,
        load_from_host: bool = False,
        store_to_host: bool = False,
        span: Span = ...,
    ) -> None:
        """Create a WindowBuffer wrapping the given Ptr ``base`` Var."""

class CommGroup(IRNode):
    """A communication group inferred for a ``@pl.program``.

    The ``CollectCommGroups`` pass (added in N4) populates
    ``program.comm_groups`` from ``pld.alloc_window_buffer`` ops and their
    dispatch coverage. Participates in structural equality / hashing via
    reflection.
    """

    devices: Final[list[int]]
    """Ascending-sorted physical device-id list. **Empty list means "all
    devices"** (every entry of ``DistributedConfig.device_ids``, resolved by
    the driver at submit-time)."""

    slots: Final[list[WindowBuffer]]
    """Allocation slots shared by every rank in the group (alloc-order)."""

    def __init__(
        self,
        devices: list[int],
        slots: list[WindowBuffer],
        span: Span = ...,
    ) -> None:
        """Create a CommGroup. Pass an empty ``devices`` list for "all devices"."""

@overload
def structural_hash(node: IRNode, enable_auto_mapping: bool = False) -> int: ...
@overload
def structural_hash(node: Type, enable_auto_mapping: bool = False) -> int: ...
def structural_hash(node: IRNode | Type, enable_auto_mapping: bool = False) -> int:
    """Compute deterministic structural hash of an IR node or type.

    Hashes based on node structure; variable identity is part of the hash unless
    auto-mapping is enabled. The hash is deterministic within a single process run.
    For IR nodes: ignores source location (Span).
    If enable_auto_mapping=True, variable names are ignored (e.g., x+1 and y+1 hash the same).
    If enable_auto_mapping=False (default), different variable objects produce different hashes.
    For types: enable_auto_mapping only affects variables embedded in the type (e.g., shape expressions).

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

def attach_leading_comments(stmt: Stmt, comments: list[str]) -> Stmt:
    """Attach leading comments to an existing statement.

    ``Stmt.leading_comments`` is read-only from Python; this helper is the
    sanctioned mutation channel for IgnoreField metadata (e.g., used by the
    DSL parser to attach extracted source comments). The input statement is
    mutated in place and returned.

    Args:
        stmt: Statement to annotate
        comments: Comment lines (without leading ``#``)

    Returns:
        The same statement, with ``leading_comments`` replaced by ``comments``
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
    - TileType → MemorySpace.Vec (Vector/unified buffer)
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
    kwargs: Mapping[str, int | bool | str | float | DataType | MemorySpace | PadValue],
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

def is_incore_type(func_type: FunctionType) -> bool:
    """Check if a FunctionType is an InCore variant (InCore, AIC, or AIV).

    Args:
        func_type: The function type to check

    Returns:
        True if the type is InCore, AIC, or AIV
    """

def level_to_linqu_level(level: Level) -> int:
    """Map Level enum value to Linqu hierarchy level number (0-7).

    Multiple Level values may map to the same Linqu level
    (e.g. AIV, AIC, CORE_GROUP all map to 0).

    Args:
        level: The hierarchy level

    Returns:
        Integer Linqu level (0-7)
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

def get_op_memory_spec(op_name: str) -> dict[str, Any] | None:
    """Get memory space specification for a registered operator.

    Args:
        op_name: Name of the operator

    Returns:
        Dict with 'input_constraints' (list of lists of MemorySpace) and
        'output_memory' keys, or None if the operator has no memory spec or
        is not registered. 'output_memory' is one of:

        * A ``MemorySpace`` — fixed or kwarg-resolved (e.g. `tile.matmul` →
          ``MemorySpace.Acc``).
        * ``'inherit_from_input'`` — the output takes its memory space from
          the first tile-typed input (e.g. `tile.slice`, `tile.reshape`).
        * ``'deferred'`` — a retargetable producer whose ``target_memory``
          kwarg was absent; `InferTileMemorySpace` resolves it later from
          consumer demand (e.g. `tile.load`, `tile.create`).
        * ``None`` — no resolver registered for this op.
    """

# ========== Op Conversion Registry ==========

def register_op_conversion(from_op: str, to_op: str) -> None:
    """Register a simple tensor-to-tile op name mapping.

    Args:
        from_op: Source op name (e.g., 'tensor.add')
        to_op: Target op name (e.g., 'tile.add')
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
    def begin_function(
        self,
        name: str,
        span: Span,
        type: FunctionType = FunctionType.Opaque,
        level: Level | None = None,
        role: Role | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """Begin building a function.

        Args:
            name: Function name
            span: Source location for function definition
            type: Function type (default: Opaque)
            level: Hierarchy level (default: None)
            role: Function role (default: None)
            attrs: Function-level attributes dict (default: None)
        """

    def func_arg(
        self,
        name: str,
        type: Type,
        span: Span,
        direction: ParamDirection = ParamDirection.In,
    ) -> Var:
        """Add a function parameter.

        Args:
            name: Parameter name
            type: Parameter type
            span: Source location for parameter
            direction: Parameter direction (default: In)

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
        chunk_size: Expr | None = None,
        chunk_policy: ChunkPolicy = ChunkPolicy.Guarded,
        attrs: dict[str, object] | list[tuple[str, object]] | None = None,
    ) -> None:
        """Begin building a for loop.

        Args:
            loop_var: Loop variable
            start: Start value expression
            stop: Stop value expression
            step: Step value expression
            span: Source location for loop definition
            kind: Loop kind (default: Sequential)
            chunk_size: Optional chunk size for loop chunking
            chunk_policy: Chunk distribution policy (default: Guarded)
            attrs: Loop-level attributes (default: empty)
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
    def begin_scope(
        self,
        scope_kind: ScopeKind,
        span: Span,
        level: Level | None = None,
        role: Role | None = None,
        split: SplitMode | None = None,
        name_hint: str = "",
        core_num: Expr | None = None,
        sync_start: bool | None = None,
        manual: bool | None = None,
    ) -> None:
        """Begin building a scope statement.

        Args:
            scope_kind: The kind of scope (e.g., ScopeKind.InCore)
            span: Source location for scope statement
            level: Hierarchy level (default: None)
            role: Hierarchy scope role (default: None)
            split: Split mode for cross-core transfer (default: None)
            name_hint: User-provided scope name hint (default: empty, auto-generated)
            core_num: SPMD block count expression (default: None)
            sync_start: Require sync-start for SPMD dispatch (default: None)
            manual: Required for ``ScopeKind.Runtime`` — selects manual vs.
                auto dependency tracking. Must be None for other scope kinds.
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

    def push_pending_leading_comments(self, comments: list[str]) -> None:
        """Push leading comments onto the pending stack.

        The DSL parser calls this before dispatching to a ``parse_*`` helper.
        Pair every push with exactly one ``pop_pending_leading_comments``.

        Args:
            comments: Comment lines (without leading ``#``)
        """

    def pop_pending_leading_comments(self) -> list[str]:
        """Pop the top pending entry, returning whatever stayed unconsumed.

        Returns ``[]`` when the matching emit already consumed the queue.
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
def python_print(node: IRNode, prefix: str = "pl", concise: bool = False, format: bool = True) -> str:
    """Print an IR node as a Python string.

    Args:
        node: IR node to print
        prefix: Module prefix (default 'pl' for 'import pypto.language as pl')
        concise: If true, omit intermediate type annotations (default false)
        format: If true, apply registered format callback (default true)

    Returns:
        String representation of the IR node
    """

def python_print_type(type: Type, prefix: str = "pl", format: bool = True) -> str:
    """Print a Type object as a Python string.

    Args:
        type: Type object to print
        prefix: Module prefix (default 'pl' for 'import pypto.language as pl')
        format: If true, apply registered format callback (default true)

    Returns:
        String representation of the Type
    """

def register_format_callback(callback: Callable[[str], str] | None) -> None:
    """Register a Python callable to post-process printed IR output.

    The callback receives a code string and returns the formatted code.
    Pass None to unregister and revert to raw output.
    """

def add(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Addition operator (lhs + rhs)."""

def sub(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Subtraction operator (lhs - rhs)."""

def mul(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Multiplication operator (lhs * rhs)."""

def truediv(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """True division operator (lhs / rhs)."""

def floordiv(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Floor division operator (lhs // rhs)."""

def mod(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Modulo operator (lhs % rhs)."""

def pow(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Power operator (lhs ** rhs)."""

def eq(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Equality operator (lhs == rhs)."""

def ne(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Inequality operator (lhs != rhs)."""

def lt(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Less than operator (lhs < rhs)."""

def le(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Less than or equal operator (lhs <= rhs)."""

def gt(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Greater than operator (lhs > rhs)."""

def ge(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Greater than or equal operator (lhs >= rhs)."""

def neg(operand: Expr, span: Span = ...) -> Expr:
    """Negation operator (-operand)."""

def cast(operand: Expr, dtype: DataType, span: Span = ...) -> Expr:
    """Cast operator (cast operand to dtype)."""

def bit_and(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Bitwise and operator (lhs & rhs)."""

def bit_or(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Bitwise or operator (lhs | rhs)."""

def bit_xor(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Bitwise xor operator (lhs ^ rhs)."""

def bit_shift_left(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Bitwise left shift operator (lhs << rhs)."""

def bit_shift_right(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Bitwise right shift operator (lhs >> rhs)."""

def bit_not(operand: Expr, span: Span = ...) -> Expr:
    """Bitwise not operator (~operand)."""

def not_(operand: Expr, span: Span = ...) -> Expr:
    """Logical not operator (not operand)."""

def and_(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Logical and operator (lhs and rhs)."""

def or_(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Logical or operator (lhs or rhs)."""

def min_(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Minimum operator (min(lhs, rhs))."""

def max_(lhs: Expr, rhs: Expr, span: Span = ...) -> Expr:
    """Maximum operator (max(lhs, rhs))."""

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
        - For SeqStmts: Each child statement's parent is the SeqStmts
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

def flatten_to_stmts(stmt: Stmt) -> list[Stmt]:
    """Unwrap a statement into a flat list.

    Returns children of SeqStmts, or a single-element list.
    """

def collect_def_vars(stmt: Stmt) -> list[Var]:
    """Collect all AssignStmt LHS variables (definition sites) from a statement tree."""

def find_yield_stmt(body: Stmt) -> YieldStmt | None:
    """Find the first YieldStmt inside a statement body (searches through SeqStmts)."""

def get_last_yield_stmt(body: Stmt) -> YieldStmt | None:
    """Find the trailing YieldStmt in a statement body (checks only the last element)."""

def substitute_expr(expr: Expr, var_map: list[tuple[Var, Var]]) -> Expr:
    """Substitute variables in an expression using a list of (original_var, replacement_var) pairs."""

def substitute_stmt(body: Stmt, var_map: list[tuple[Var, Var]]) -> Stmt:
    """Substitute variable references in a statement subtree using (original_var, replacement_var) pairs."""

def deep_clone(
    body: Stmt,
    var_map: list[tuple[Var, Expr]] = ...,
) -> tuple[Stmt, list[tuple[Var, Var]]]:
    """Deep-clone a statement subtree, creating fresh Var objects at definition sites.

    All Var, IterArg, and MemRef objects at definition sites inside the statement
    tree are freshly created to avoid shared identity with the original.

    Args:
        body: The statement subtree to clone
        var_map: Optional seeded substitutions. Each (original_var, replacement_expr)
            pair replaces references to original_var with replacement_expr inside
            the clone (including expressions embedded in type annotations).

    Returns:
        Tuple of (cloned_body, def_var_map) where def_var_map is a list of
        (original_var, cloned_var) pairs for the definition sites freshly
        cloned by the traversal. Seeded entries from var_map are NOT included.
    """

def deduce_call_return_type(
    callee_params: Sequence[Var],
    args: Sequence[Expr],
    return_types: Sequence[Type],
) -> list[Type]:
    """Deduce return types for a cross-function call.

    Substitutes dynamic shape variables in callee return types
    with concrete values from actual argument types.

    Args:
        callee_params: Callee function parameter variables
        args: Actual call argument expressions
        return_types: Callee's declared return types

    Returns:
        Substituted return types (unchanged if no dynamic vars found)
    """

class IRVisitor:
    """Read-only IR visitor. Subclass and override visit_* methods.

    Default implementations recursively traverse all children.
    Call super().visit_*() from your override to keep the default recursion.
    """

    def __init__(self) -> None: ...
    def visit_program(self, program: Program) -> None:
        """Visit all functions in a program."""
    def visit_function(self, func: Function) -> None:
        """Visit a function's parameters and body."""
    def visit_expr(self, expr: Expr) -> None:
        """Dispatch to type-specific expression handler."""
    def visit_stmt(self, stmt: Stmt) -> None:
        """Dispatch to type-specific statement handler."""
    def visit_var_like(self, op: Var) -> None:
        """Visit Var/IterArg shared logic (type shape expressions)."""
    def visit_binary_expr(self, op: BinaryExpr) -> None:
        """Visit any binary expression (default: visit left and right)."""
    def visit_unary_expr(self, op: UnaryExpr) -> None:
        """Visit any unary expression (default: visit operand)."""
    def visit_var(self, op: Var) -> None: ...
    def visit_iter_arg(self, op: IterArg) -> None: ...
    def visit_mem_ref(self, op: MemRef) -> None: ...
    def visit_const_int(self, op: ConstInt) -> None: ...
    def visit_const_float(self, op: ConstFloat) -> None: ...
    def visit_const_bool(self, op: ConstBool) -> None: ...
    def visit_call(self, op: Call) -> None: ...
    def visit_make_tuple(self, op: MakeTuple) -> None: ...
    def visit_tuple_get_item_expr(self, op: TupleGetItemExpr) -> None: ...
    # Individual binary expression handlers (default: delegates to visit_binary_expr)
    def visit_add(self, op: Add) -> None: ...
    def visit_sub(self, op: Sub) -> None: ...
    def visit_mul(self, op: Mul) -> None: ...
    def visit_floor_div(self, op: FloorDiv) -> None: ...
    def visit_floor_mod(self, op: FloorMod) -> None: ...
    def visit_float_div(self, op: FloatDiv) -> None: ...
    def visit_min(self, op: Min) -> None: ...
    def visit_max(self, op: Max) -> None: ...
    def visit_pow(self, op: Pow) -> None: ...
    def visit_eq(self, op: Eq) -> None: ...
    def visit_ne(self, op: Ne) -> None: ...
    def visit_lt(self, op: Lt) -> None: ...
    def visit_le(self, op: Le) -> None: ...
    def visit_gt(self, op: Gt) -> None: ...
    def visit_ge(self, op: Ge) -> None: ...
    def visit_and(self, op: And) -> None: ...
    def visit_or(self, op: Or) -> None: ...
    def visit_xor(self, op: Xor) -> None: ...
    def visit_bit_and(self, op: BitAnd) -> None: ...
    def visit_bit_or(self, op: BitOr) -> None: ...
    def visit_bit_xor(self, op: BitXor) -> None: ...
    def visit_bit_shift_left(self, op: BitShiftLeft) -> None: ...
    def visit_bit_shift_right(self, op: BitShiftRight) -> None: ...
    # Individual unary expression handlers (default: delegates to visit_unary_expr)
    def visit_abs(self, op: Abs) -> None: ...
    def visit_neg(self, op: Neg) -> None: ...
    def visit_not(self, op: Not) -> None: ...
    def visit_bit_not(self, op: BitNot) -> None: ...
    def visit_cast(self, op: Cast) -> None: ...
    # Statement handlers
    def visit_assign_stmt(self, op: AssignStmt) -> None: ...
    def visit_if_stmt(self, op: IfStmt) -> None: ...
    def visit_for_stmt(self, op: ForStmt) -> None: ...
    def visit_while_stmt(self, op: WhileStmt) -> None: ...
    def visit_in_core_scope_stmt(self, op: InCoreScopeStmt) -> None: ...
    def visit_auto_in_core_scope_stmt(self, op: AutoInCoreScopeStmt) -> None: ...
    def visit_cluster_scope_stmt(self, op: ClusterScopeStmt) -> None: ...
    def visit_hierarchy_scope_stmt(self, op: HierarchyScopeStmt) -> None: ...
    def visit_spmd_scope_stmt(self, op: SpmdScopeStmt) -> None: ...
    def visit_runtime_scope_stmt(self, op: RuntimeScopeStmt) -> None: ...
    def visit_seq_stmts(self, op: SeqStmts) -> None: ...
    def visit_yield_stmt(self, op: YieldStmt) -> None: ...
    def visit_return_stmt(self, op: ReturnStmt) -> None: ...
    def visit_eval_stmt(self, op: EvalStmt) -> None: ...
    def visit_break_stmt(self, op: BreakStmt) -> None: ...
    def visit_continue_stmt(self, op: ContinueStmt) -> None: ...
    def visit_inline_stmt(self, op: InlineStmt) -> None: ...

class IRMutator:
    """IR mutator with copy-on-write semantics.

    Subclass and override visit_* methods to transform IR.
    Default implementations recurse into children and reconstruct
    nodes only when children change (copy-on-write).
    """

    def __init__(self) -> None: ...
    def visit_program(self, program: Program) -> Program:
        """Mutate all functions in a program."""
    def visit_function(self, func: Function) -> Function:
        """Mutate a function's body."""
    def visit_expr(self, expr: Expr) -> Expr:
        """Dispatch to type-specific expression mutator."""
    def visit_stmt(self, stmt: Stmt) -> Stmt:
        """Dispatch to type-specific statement mutator."""
    def visit_binary_expr(self, op: BinaryExpr) -> Expr:
        """Mutate any binary expression (default: visit children, reconstruct if changed)."""
    def visit_unary_expr(self, op: UnaryExpr) -> Expr:
        """Mutate any unary expression (default: visit operand, reconstruct if changed)."""
    def visit_var(self, op: Var) -> Expr: ...
    def visit_iter_arg(self, op: IterArg) -> Expr: ...
    def visit_mem_ref(self, op: MemRef) -> Expr: ...
    def visit_const_int(self, op: ConstInt) -> Expr: ...
    def visit_const_float(self, op: ConstFloat) -> Expr: ...
    def visit_const_bool(self, op: ConstBool) -> Expr: ...
    def visit_call(self, op: Call) -> Expr: ...
    def visit_make_tuple(self, op: MakeTuple) -> Expr: ...
    def visit_tuple_get_item_expr(self, op: TupleGetItemExpr) -> Expr: ...
    # Individual binary expression handlers (default: delegates to visit_binary_expr)
    def visit_add(self, op: Add) -> Expr: ...
    def visit_sub(self, op: Sub) -> Expr: ...
    def visit_mul(self, op: Mul) -> Expr: ...
    def visit_floor_div(self, op: FloorDiv) -> Expr: ...
    def visit_floor_mod(self, op: FloorMod) -> Expr: ...
    def visit_float_div(self, op: FloatDiv) -> Expr: ...
    def visit_min(self, op: Min) -> Expr: ...
    def visit_max(self, op: Max) -> Expr: ...
    def visit_pow(self, op: Pow) -> Expr: ...
    def visit_eq(self, op: Eq) -> Expr: ...
    def visit_ne(self, op: Ne) -> Expr: ...
    def visit_lt(self, op: Lt) -> Expr: ...
    def visit_le(self, op: Le) -> Expr: ...
    def visit_gt(self, op: Gt) -> Expr: ...
    def visit_ge(self, op: Ge) -> Expr: ...
    def visit_and(self, op: And) -> Expr: ...
    def visit_or(self, op: Or) -> Expr: ...
    def visit_xor(self, op: Xor) -> Expr: ...
    def visit_bit_and(self, op: BitAnd) -> Expr: ...
    def visit_bit_or(self, op: BitOr) -> Expr: ...
    def visit_bit_xor(self, op: BitXor) -> Expr: ...
    def visit_bit_shift_left(self, op: BitShiftLeft) -> Expr: ...
    def visit_bit_shift_right(self, op: BitShiftRight) -> Expr: ...
    # Individual unary expression handlers (default: delegates to visit_unary_expr)
    def visit_abs(self, op: Abs) -> Expr: ...
    def visit_neg(self, op: Neg) -> Expr: ...
    def visit_not(self, op: Not) -> Expr: ...
    def visit_bit_not(self, op: BitNot) -> Expr: ...
    def visit_cast(self, op: Cast) -> Expr: ...
    # Statement handlers
    def visit_assign_stmt(self, op: AssignStmt) -> Stmt: ...
    def visit_if_stmt(self, op: IfStmt) -> Stmt: ...
    def visit_for_stmt(self, op: ForStmt) -> Stmt: ...
    def visit_while_stmt(self, op: WhileStmt) -> Stmt: ...
    def visit_in_core_scope_stmt(self, op: InCoreScopeStmt) -> Stmt: ...
    def visit_auto_in_core_scope_stmt(self, op: AutoInCoreScopeStmt) -> Stmt: ...
    def visit_cluster_scope_stmt(self, op: ClusterScopeStmt) -> Stmt: ...
    def visit_hierarchy_scope_stmt(self, op: HierarchyScopeStmt) -> Stmt: ...
    def visit_spmd_scope_stmt(self, op: SpmdScopeStmt) -> Stmt: ...
    def visit_runtime_scope_stmt(self, op: RuntimeScopeStmt) -> Stmt: ...
    def visit_seq_stmts(self, op: SeqStmts) -> Stmt: ...
    def visit_yield_stmt(self, op: YieldStmt) -> Stmt: ...
    def visit_return_stmt(self, op: ReturnStmt) -> Stmt: ...
    def visit_eval_stmt(self, op: EvalStmt) -> Stmt: ...
    def visit_break_stmt(self, op: BreakStmt) -> Stmt: ...
    def visit_continue_stmt(self, op: ContinueStmt) -> Stmt: ...
    def visit_inline_stmt(self, op: InlineStmt) -> Stmt: ...
