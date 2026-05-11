# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type annotation resolution for IR parsing."""

import ast
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from pypto.language.typing.dynamic import DynVar
from pypto.pypto_core import DataType, ir

from .diagnostics import ParserTypeError
from .expr_evaluator import ExprEvaluator


def _const_int_value(value: object) -> int | None:
    """Extract integer value from a compile-time constant, or None."""
    if isinstance(value, int):
        return value
    if isinstance(value, ir.ConstInt):
        return value.value
    if isinstance(value, ir.Neg) and isinstance(value.operand, ir.ConstInt):
        return -value.operand.value
    return None


def _infer_implicit_tile_layout_from_shape(shape: "Sequence[ir.Expr | int]") -> "ir.TileLayout":
    """Infer the block layout represented by omitted TileView syntax."""
    if len(shape) != 2:
        return ir.TileLayout.row_major
    rows_value = _const_int_value(shape[0])
    cols_value = _const_int_value(shape[1])
    if rows_value is None or cols_value is None:
        return ir.TileLayout.row_major
    return ir.TileLayout.col_major if cols_value == 1 and rows_value > 1 else ir.TileLayout.row_major


def _implicit_tile_view_defaults(
    shape: "Sequence[ir.Expr | int]",
    memory_space: "ir.MemorySpace | None" = None,
) -> "tuple[ir.TileLayout, ir.TileLayout, int]":
    """Return (blayout, slayout, fractal) implicit defaults for a given shape + memory space."""
    default_blayout = _infer_implicit_tile_layout_from_shape(shape)
    default_slayout = ir.TileLayout.none_box
    default_fractal = ir.TileView().fractal
    if memory_space in (ir.MemorySpace.Mat, ir.MemorySpace.Left):
        default_blayout = ir.TileLayout.col_major
        default_slayout = ir.TileLayout.row_major
    elif memory_space == ir.MemorySpace.Right:
        default_slayout = ir.TileLayout.col_major
    elif memory_space == ir.MemorySpace.Acc:
        default_blayout = ir.TileLayout.col_major
        default_slayout = ir.TileLayout.row_major
        default_fractal = 1024
    return default_blayout, default_slayout, default_fractal


if TYPE_CHECKING:
    from .span_tracker import SpanTracker


def _try_get_static_dim(dim: ir.Expr) -> int | None:
    """Return static int value from shape dim, or None if dynamic."""
    if isinstance(dim, ir.ConstInt):
        return dim.value
    return None


def _is_index_expr_type(t: "ir.Type | None") -> bool:
    """Whether ``t`` is admissible for a TileView/TensorView field expression.

    Mirrors the C++ contract: TileView fields are integer/index scalars
    (constants, dyn vars, and arithmetic over them). Excludes tile, tensor,
    tuple, float, and bool types.
    """
    if not isinstance(t, ir.ScalarType):
        return False
    return t.dtype == DataType.INDEX or t.dtype.is_int()


def _is_pl_yield_call(node: ast.expr) -> bool:
    """Detect ``pl.yield_(...)`` / bare ``yield_(...)`` call nodes.

    ``parse_yield_call`` emits an ``ir.YieldStmt`` to the builder as a side
    effect — incompatible with the pure-expression contract type-annotation
    parsing assumes.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "yield_":
        return True
    return isinstance(func, ast.Name) and func.id == "yield_"


_TYPE_KIND_NAMES: dict[type, str] = {
    ir.TensorType: "Tensor",
    ir.TileType: "Tile",
    ir.ScalarType: "Scalar",
    ir.ArrayType: "Array",
}


def _dtypes_compatible(ann: DataType, inf: DataType) -> bool:
    """Check if annotation dtype is compatible with inferred dtype.

    INDEX is an internal type for loop variables / addressing. Users naturally
    annotate these as INT32/INT64, so INDEX is treated as compatible with any
    integer type.
    """
    if ann == inf:
        return True
    # INDEX is compatible with integer dtypes in both directions
    if ann == DataType.INDEX and inf.is_int():
        return True
    if inf == DataType.INDEX and ann.is_int():
        return True
    return False


class TypeResolver:
    """Resolves Python type annotations to IR types."""

    _DTYPE_MAP: dict[str, DataType] = {
        "FP4": DataType.FP4,
        "FP8E4M3FN": DataType.FP8E4M3FN,
        "FP8E5M2": DataType.FP8E5M2,
        "FP16": DataType.FP16,
        "FP32": DataType.FP32,
        "BF16": DataType.BF16,
        "HF4": DataType.HF4,
        "HF8": DataType.HF8,
        "INT4": DataType.INT4,
        "INT8": DataType.INT8,
        "INT16": DataType.INT16,
        "INT32": DataType.INT32,
        "INT64": DataType.INT64,
        "UINT4": DataType.UINT4,
        "UINT8": DataType.UINT8,
        "UINT16": DataType.UINT16,
        "UINT32": DataType.UINT32,
        "UINT64": DataType.UINT64,
        "BOOL": DataType.BOOL,
        "INDEX": DataType.INDEX,
    }

    _DIRECTION_MAP: dict[str, "ir.ParamDirection"] = {
        "InOut": ir.ParamDirection.InOut,
        "Out": ir.ParamDirection.Out,
    }

    _LAYOUT_MAP: dict[str, "ir.TensorLayout"] = {
        "ND": ir.TensorLayout.ND,
        "DN": ir.TensorLayout.DN,
        "NZ": ir.TensorLayout.NZ,
    }

    _MEMORY_SPACE_MAP: dict[str, "ir.MemorySpace"] = {
        "DDR": ir.MemorySpace.DDR,
        "Vec": ir.MemorySpace.Vec,
        "Mat": ir.MemorySpace.Mat,
        "Left": ir.MemorySpace.Left,
        "Right": ir.MemorySpace.Right,
        "Acc": ir.MemorySpace.Acc,
        "Bias": ir.MemorySpace.Bias,
    }

    def __init__(
        self,
        expr_evaluator: ExprEvaluator,
        scope_lookup: Callable[[str], Any | None] | None = None,
        span_tracker: "SpanTracker | None" = None,
        dyn_var_cache: dict[str, ir.Var] | None = None,
        parse_expression: Callable[[ast.expr], "ir.Expr"] | None = None,
    ):
        """Initialize type resolver.

        Args:
            expr_evaluator: Evaluator for resolving expressions from closure variables
            scope_lookup: Callback to look up variables in the parser scope
                (for Scalar IR vars used in inline annotations)
            span_tracker: Optional span tracker for accurate source locations
            dyn_var_cache: Optional shared cache mapping dynamic var names to ir.Var
                objects. When provided, multiple TypeResolvers share the same cache,
                ensuring the same DynVar produces the same ir.Var across functions
                in a program.
            parse_expression: Optional callback to the enclosing parser's full
                expression parser. When set, TileView/TensorView fields accept any
                index expression the DSL itself can parse (matching what the C++
                ``ir.TileView`` constructor accepts), not just integer constants
                and bare names.
        """
        self.expr_evaluator = expr_evaluator
        self.scope_lookup = scope_lookup
        self.span_tracker = span_tracker
        self._dyn_var_cache: dict[str, ir.Var] = dyn_var_cache if dyn_var_cache is not None else {}
        self._parse_expression = parse_expression

    def resolve_param_type(self, type_node: ast.expr) -> "tuple[ir.Type, ir.ParamDirection]":
        """Resolve AST type annotation to (ir.Type, ParamDirection) for function parameters.

        Detects InOut[...] and Out[...] wrappers and extracts the direction.
        Default direction is In.

        Args:
            type_node: AST expression representing the type annotation

        Returns:
            Tuple of (resolved IR type, parameter direction)

        Raises:
            ParserTypeError: If type annotation cannot be resolved or has invalid direction
        """
        direction = ir.ParamDirection.In

        # Check for InOut[...] or Out[...] wrapper
        if isinstance(type_node, ast.Subscript):
            wrapper_name = self._get_direction_wrapper(type_node.value)
            if wrapper_name is not None:
                direction = self._DIRECTION_MAP[wrapper_name]
                type_node = type_node.slice

        resolved = self.resolve_type(type_node)
        if isinstance(resolved, list):
            raise ParserTypeError(
                "Parameter type cannot be a tuple",
                span=self._get_span(type_node),
                hint="Tuple types are only supported as return types",
            )

        # Validate: Scalar + InOut is not allowed
        if direction == ir.ParamDirection.InOut and isinstance(resolved, ir.ScalarType):
            raise ParserTypeError(
                "Scalar parameters cannot have InOut direction",
                span=self._get_span(type_node),
                hint="Only Tensor and Tile parameters support InOut direction",
            )

        return resolved, direction

    def _get_direction_wrapper(self, node: ast.expr) -> str | None:
        """Check if an AST node is an InOut or Out wrapper reference.

        Args:
            node: AST expression to check

        Returns:
            "InOut" or "Out" if it's a direction wrapper, None otherwise
        """
        if isinstance(node, ast.Attribute) and node.attr in ("InOut", "Out"):
            return node.attr
        if isinstance(node, ast.Name) and node.id in ("InOut", "Out"):
            return node.id
        return None

    def _get_type_name(self, node: ast.expr) -> str | None:
        """Extract the type name from an AST node referencing Tensor, Tile, Scalar,
        Tuple, or DistributedTensor.

        Handles both ``pl.Tensor`` (ast.Attribute) and bare ``Tensor`` (ast.Name);
        ``pld.DistributedTensor`` is recognized either as ``pld.DistributedTensor``
        or bare ``DistributedTensor``.

        Args:
            node: AST expression to check

        Returns:
            Type name string if recognized, None otherwise
        """
        valid = ("Tensor", "Tile", "Scalar", "Array", "Tuple", "DistributedTensor")
        if isinstance(node, ast.Attribute) and node.attr in valid:
            return node.attr
        if isinstance(node, ast.Name) and node.id in valid:
            return node.id
        return None

    def resolve_type(self, type_node: ast.expr) -> "ir.Type | list[ir.Type]":
        """Resolve AST type annotation to ir.Type or list of types.

        Args:
            type_node: AST expression representing the type annotation

        Returns:
            Corresponding IR type, or list of IR types for tuple[T1, T2, ...] annotations

        Raises:
            ValueError: If type annotation cannot be resolved
        """
        # Handle subscript notation: pl.Tensor[...], pl.Tile[...], pl.Scalar[...], tuple[...]
        if isinstance(type_node, ast.Subscript):
            # Check for tuple[T1, T2, ...] return type annotation
            value = type_node.value
            if isinstance(value, ast.Name) and value.id == "tuple":
                return self._resolve_tuple_type(type_node)
            return self._resolve_subscript_type(type_node)

        # Handle pl.Tensor((64, 128), pl.FP16) call notation (legacy)
        if isinstance(type_node, ast.Call):
            return self._resolve_call_type(type_node)

        # Handle attribute access like pl.Tensor
        if isinstance(type_node, ast.Attribute):
            raise ParserTypeError(
                f"Incomplete type annotation: {ast.unparse(type_node)}",
                span=self._get_span(type_node),
                hint="Use pl.Tensor[[shape], dtype], pl.Tile[[shape], dtype], or pl.Scalar[dtype]",
            )

        raise ParserTypeError(
            f"Unsupported type annotation: {ast.unparse(type_node)}",
            span=self._get_span(type_node),
            hint="Use pl.Tensor[[shape], dtype], pl.Tile[[shape], dtype], or pl.Scalar[dtype]",
        )

    def _resolve_subscript_type(self, subscript_node: ast.Subscript) -> ir.Type:  # noqa: PLR0912
        """Resolve subscript type annotation.

        Supports:
        - pl.Tensor[[64, 128], pl.FP16]
        - pl.Tensor[[64, 128], pl.FP16, pl.NZ]
        - pl.Tensor[[64, 128], pl.FP16, pl.MemRef(...)]
        - pl.Tensor[[64, 128], pl.FP16, pl.NZ, pl.MemRef(...)]
        - pl.Tile[[64, 64], pl.FP32]
        - pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec]
        - pl.Tile[[64, 64], pl.FP32, pl.MemRef(...), pl.Mem.Vec]
        - pl.Tile[[64, 64], pl.FP32, pl.MemRef(...), pl.Mem.Vec, pl.TileView(...)]

        Args:
            subscript_node: AST Subscript node

        Returns:
            IR type

        Raises:
            ParserTypeError: If subscript cannot be resolved to a type
        """
        value = subscript_node.value
        type_name = self._get_type_name(value)

        if type_name is None:
            raise ParserTypeError(
                f"Unknown type in subscript: {ast.unparse(value)}",
                span=self._get_span(value),
                hint="Use pl.Tensor for tensor types, pl.Tile for tile types, or pl.Scalar for scalar types",
            )

        slice_value = subscript_node.slice

        if type_name == "Scalar":
            dtype = self.resolve_dtype(slice_value)
            return ir.ScalarType(dtype)

        if type_name == "Array":
            # pl.Array[N, dtype] — N is an int literal, dtype is a DataType ref.
            if not isinstance(slice_value, ast.Tuple) or len(slice_value.elts) != 2:
                raise ParserTypeError(
                    f"Array subscript requires [extent, dtype], got: {ast.unparse(slice_value)}",
                    span=self._get_span(slice_value),
                    hint="Use pl.Array[N, pl.INT32]",
                )
            extent_node, dtype_node = slice_value.elts
            success, extent_value = self.expr_evaluator.try_eval_expr(extent_node)
            if not success or not isinstance(extent_value, int) or isinstance(extent_value, bool):
                raise ParserTypeError(
                    f"Array extent must be an int literal or compile-time constant, "
                    f"got: {ast.unparse(extent_node)}",
                    span=self._get_span(extent_node),
                    hint="Use a positive integer literal for the extent",
                )
            dtype = self.resolve_dtype(dtype_node)
            return ir.ArrayType(dtype, extent_value)

        if type_name == "Tuple":
            return self._resolve_tuple_subscript_type(subscript_node)

        # Tensor: [shape, dtype], [shape, dtype, layout_or_memref], [shape, dtype, layout, memref].
        # DistributedTensor: same forms as Tensor — only the IR ObjectKind differs.
        # Tile: [shape, dtype] plus any ordering of TileView/MemRef/MemorySpace,
        # with the constraint that MemRef requires explicit MemorySpace.
        is_distributed = type_name == "DistributedTensor"
        is_tensor_like = type_name == "Tensor" or is_distributed
        tensor_ctor = ir.DistributedTensorType if is_distributed else ir.TensorType
        valid_counts = (2, 3, 4) if is_tensor_like else (2, 3, 4, 5)
        if not isinstance(slice_value, ast.Tuple) or len(slice_value.elts) not in valid_counts:
            if is_tensor_like:
                message = (
                    f"{type_name} subscript requires [shape, dtype], [shape, dtype, layout_or_memref], "
                    f"or [shape, dtype, layout, memref], got: {ast.unparse(slice_value)}"
                )
                hint = (
                    f"Use {type_name}[[shape], dtype], {type_name}[[shape], dtype, layout], "
                    f"or {type_name}[[shape], dtype, pl.MemRef(...)] format"
                )
            else:
                message = (
                    f"{type_name} subscript requires [shape, dtype], "
                    f"[shape, dtype, tileview_or_memory_space], "
                    f"[shape, dtype, memref, memory_space], "
                    f"or [shape, dtype, memref, memory_space, tileview], "
                    f"got: {ast.unparse(slice_value)}"
                )
                hint = (
                    f"Use pl.{type_name}[[shape], dtype], "
                    f"pl.{type_name}[[shape], dtype, pl.Mem.Vec], "
                    f"pl.{type_name}[[shape], dtype, pl.MemRef(...), pl.Mem.Vec], "
                    f"or pl.{type_name}[[shape], dtype, pl.MemRef(...), pl.Mem.Vec, pl.TileView(...)]"
                )
            raise ParserTypeError(message, span=self._get_span(slice_value), hint=hint)

        shape_node = slice_value.elts[0]
        dtype_node = slice_value.elts[1]

        shape = self._to_ir_shape(self._parse_shape(shape_node))
        dtype = self.resolve_dtype(dtype_node)

        n_elts = len(slice_value.elts)

        # 2 args: [shape, dtype]
        if n_elts == 2:
            if type_name == "Tile":
                return ir.TileType(shape, dtype)
            return tensor_ctor(shape, dtype, None, None)

        if type_name == "Tile":
            return self._resolve_tile_annotation_args(shape, dtype, list(slice_value.elts[2:]))

        # 3 args: [shape, dtype, layout_or_memref_or_tensorview] for Tensor / DistributedTensor
        if n_elts == 3:
            third = slice_value.elts[2]
            if self._is_memref_node(third):
                memref = self.resolve_memref(third)
                return tensor_ctor(shape, dtype, memref, None)
            if self._is_tensorview_node(third):
                tensor_view = self._resolve_tensorview(third)
                return tensor_ctor(shape, dtype, None, tensor_view)
            layout = self.resolve_layout(third)
            tensor_view = ir.TensorView([], layout)
            return tensor_ctor(shape, dtype, None, tensor_view)

        # Tensor / DistributedTensor 4 args: [shape, dtype, layout_or_tensorview, memref]
        third = slice_value.elts[2]
        if self._is_tensorview_node(third):
            tensor_view = self._resolve_tensorview(third)
        else:
            layout = self.resolve_layout(third)
            tensor_view = ir.TensorView([], layout)
        memref_node = slice_value.elts[3]
        if not self._is_memref_node(memref_node):
            raise ParserTypeError(
                f"{type_name} 4th argument must be pl.MemRef(...)",
                span=self._get_span(memref_node),
                hint=f"Use {type_name}[[shape], dtype, layout, pl.MemRef(...)]",
            )
        memref = self.resolve_memref(memref_node)
        return tensor_ctor(shape, dtype, memref, tensor_view)

    def _resolve_tile_annotation_args(
        self, shape: "list[int] | list[ir.Expr]", dtype: DataType, extra_nodes: list[ast.expr]
    ) -> "ir.TileType":
        """Resolve Tile trailing annotation args.

        Accepted arguments are any ordering of:
        - `pl.TileView(...)`
        - `pl.Mem.<space>` / `pl.MemorySpace.<space>`
        - `pl.MemRef(...)`
        - a previously defined `pl.MemRefType` variable name

        Constraint:
        - If `pl.MemRef(...)` is present, an explicit memory-space argument is required.
        """
        memref_node: ast.expr | None = None
        tile_view_node: ast.expr | None = None
        memory_space_node: ast.expr | None = None

        for node in extra_nodes:
            if self._is_memref_node(node) or self._resolve_memref_var_ref(node) is not None:
                if memref_node is not None:
                    raise ParserTypeError(
                        "Tile annotation can contain at most one memref argument",
                        span=self._get_span(node),
                        hint="Remove the duplicate pl.MemRef(...) or MemRefType variable argument",
                    )
                memref_node = node
                continue

            if self._is_tileview_node(node):
                if tile_view_node is not None:
                    raise ParserTypeError(
                        "Tile annotation can contain at most one pl.TileView(...)",
                        span=self._get_span(node),
                        hint="Remove the duplicate pl.TileView(...) argument",
                    )
                tile_view_node = node
                continue

            if self._is_memory_space_node(node):
                if memory_space_node is not None:
                    raise ParserTypeError(
                        "Tile annotation can contain at most one memory-space argument",
                        span=self._get_span(node),
                        hint="Remove the duplicate pl.Mem.<space> argument",
                    )
                memory_space_node = node
                continue

            if self._is_layout_node(node):
                raise ParserTypeError(
                    f"Tile does not accept layouts like {ast.unparse(node)}",
                    span=self._get_span(node),
                    hint="Use pl.TileView(...) for tile views, or use pl.Tensor[...] for layout annotations",
                )

            raise ParserTypeError(
                f"Unsupported Tile annotation argument: {ast.unparse(node)}",
                span=self._get_span(node),
                hint="Use pl.TileView(...), pl.Mem.<space>, pl.MemRef(...), and/or a MemRefType variable",
            )

        if memref_node is not None and memory_space_node is None:
            raise ParserTypeError(
                "Tile annotation with a memref argument must also specify explicit memory space",
                span=self._get_span(memref_node),
                hint="Use pl.Tile[[shape], dtype, pl.MemRef(base, offset, size), pl.Mem.Vec] or "
                "pl.Tile[[shape], dtype, memref_var, pl.Mem.Vec]",
            )

        if memref_node is None:
            memref = None
        elif self._is_memref_node(memref_node):
            memref = self.resolve_memref(memref_node)
        else:
            memref = self._resolve_memref_var_ref(memref_node)
        # Resolve memory_space first so it can be passed to _resolve_tileview for
        # memory-space-aware implicit blayout/slayout/fractal defaults.
        #
        # Known limitation: when the annotation omits memory_space but the inferred
        # RHS type provides it (merged later in ast_parser.py), the TileView defaults
        # here use the annotation-local memory_space (which may be None).  In practice
        # this does not affect roundtrip because the printer always emits the memory
        # space annotation when one is present.  User code that writes
        # ``pl.Tile[..., pl.TileView(fractal=1024)]`` without ``pl.Mem.Acc`` would get
        # shape-based blayout/slayout defaults instead of Acc defaults for unspecified
        # fields — a known edge case that requires a larger refactor to fix properly.
        memory_space = (
            self._resolve_memory_space(memory_space_node) if memory_space_node is not None else None
        )
        tile_view = (
            self._resolve_tileview(tile_view_node, shape, memory_space)
            if tile_view_node is not None
            else None
        )
        return ir.TileType(shape, dtype, memref, tile_view, memory_space)

    def _resolve_memref_var_ref(self, node: ast.expr) -> "ir.MemRef | None":
        """Resolve a previously bound MemRef variable used in a Tile annotation."""
        if not isinstance(node, ast.Name) or self.scope_lookup is None:
            return None
        var = self.scope_lookup(node.id)
        if isinstance(var, ir.MemRef):
            return var
        return None

    def _resolve_tuple_type(self, subscript_node: ast.Subscript) -> list[ir.Type]:
        """Resolve tuple[T1, T2, ...] return type annotation.

        Args:
            subscript_node: AST Subscript node with tuple base

        Returns:
            List of IR types
        """
        slice_value = subscript_node.slice
        elts = slice_value.elts if isinstance(slice_value, ast.Tuple) else [slice_value]

        types = []
        for elt in elts:
            resolved = self.resolve_type(elt)
            if isinstance(resolved, list):
                raise ParserTypeError(
                    "Nested tuple types are not supported",
                    span=self._get_span(elt),
                    hint="Use a flat tuple like tuple[pl.Tensor[...], pl.Tensor[...]]",
                )
            types.append(resolved)
        return types

    def _resolve_call_type(self, call_node: ast.Call) -> ir.Type:
        """Resolve a function call type annotation.

        Args:
            call_node: AST Call node

        Returns:
            IR type

        Raises:
            ValueError: If call cannot be resolved to a type
        """
        func = call_node.func
        type_name = self._get_type_name(func)

        resolvers = {
            "Tensor": self._resolve_tensor_type,
            "Tile": self._resolve_tile_type,
            "Scalar": self._resolve_scalar_type,
            "Tuple": self._resolve_tuple_call_type,
        }
        resolver = resolvers.get(type_name) if type_name is not None else None
        if resolver is not None:
            return resolver(call_node)

        raise ParserTypeError(
            f"Unknown type constructor: {ast.unparse(func)}",
            span=self._get_span(call_node),
            hint="Use pl.Tensor[[shape], dtype], pl.Tile[[shape], dtype], or pl.Scalar[dtype]",
        )

    def _resolve_tensor_type(self, call_node: ast.Call) -> ir.TensorType:
        """Resolve pl.Tensor((shape), dtype) annotation (legacy)."""
        result = self._resolve_shaped_type(call_node, "Tensor", ir.TensorType)
        assert isinstance(result, ir.TensorType)
        return result

    def _resolve_tile_type(self, call_node: ast.Call) -> ir.TileType:
        """Resolve pl.Tile((shape), dtype) annotation (legacy)."""
        result = self._resolve_shaped_type(call_node, "Tile", ir.TileType)
        assert isinstance(result, ir.TileType)
        return result

    def _resolve_shaped_type(
        self,
        call_node: ast.Call,
        type_name: str,
        type_ctor: type[ir.TensorType] | type[ir.TileType],
    ) -> ir.TensorType | ir.TileType:
        """Resolve a shaped type (Tensor or Tile) from a legacy call annotation.

        Args:
            call_node: AST Call node for the type constructor
            type_name: "Tensor" or "Tile" for error messages
            type_ctor: IR type constructor (ir.TensorType or ir.TileType)

        Returns:
            Constructed IR type

        Raises:
            ParserTypeError: If type annotation is malformed
        """
        if len(call_node.args) < 2:
            raise ParserTypeError(
                f"{type_name} type requires shape and dtype arguments, got {len(call_node.args)}",
                span=self._get_span(call_node),
                hint=f"Use pl.{type_name}[[shape], dtype] format",
            )

        shape = self._to_ir_shape(self._parse_shape(call_node.args[0]))
        dtype = self.resolve_dtype(call_node.args[1])
        return type_ctor(shape, dtype)

    def _resolve_scalar_type(self, call_node: ast.Call) -> ir.ScalarType:
        """Resolve pl.Scalar(dtype) annotation (legacy).

        Args:
            call_node: AST Call node for Scalar constructor

        Returns:
            ScalarType

        Raises:
            ParserTypeError: If scalar type annotation is malformed
        """
        if len(call_node.args) < 1:
            raise ParserTypeError(
                f"Scalar type requires dtype argument, got {len(call_node.args)}",
                span=self._get_span(call_node),
                hint="Use pl.Scalar[dtype] format, e.g., pl.Scalar[pl.FP32]",
            )

        # Parse dtype (first argument)
        dtype_node = call_node.args[0]
        dtype = self.resolve_dtype(dtype_node)

        # Create ScalarType
        return ir.ScalarType(dtype)

    def _resolve_tuple_subscript_type(self, subscript_node: ast.Subscript) -> ir.TupleType:
        """Resolve pl.Tuple[T1, T2, ...] or pl.Tuple[()] annotation to ir.TupleType."""
        slice_value = subscript_node.slice
        # Handle empty tuple: pl.Tuple[()]
        if isinstance(slice_value, ast.Constant) and slice_value.value == ():
            return ir.TupleType([])
        if isinstance(slice_value, ast.Tuple) and len(slice_value.elts) == 0:
            return ir.TupleType([])
        elts = slice_value.elts if isinstance(slice_value, ast.Tuple) else [slice_value]
        return ir.TupleType(self._resolve_tuple_element_types(elts))

    def _resolve_tuple_call_type(self, call_node: ast.Call) -> ir.TupleType:
        """Resolve pl.Tuple([type1, type2, ...]) annotation to ir.TupleType (legacy)."""
        if len(call_node.args) != 1 or not isinstance(call_node.args[0], ast.List):
            raise ParserTypeError(
                f"Tuple type requires a list of types, got: {ast.unparse(call_node)}",
                span=self._get_span(call_node),
                hint="Use pl.Tuple[pl.Tensor[...], pl.Tile[...], ...] format",
            )
        return ir.TupleType(self._resolve_tuple_element_types(call_node.args[0].elts))

    def _resolve_tuple_element_types(self, elts: Sequence[ast.expr]) -> list[ir.Type]:
        """Resolve a sequence of AST type nodes into IR types, rejecting nested tuples."""
        types: list[ir.Type] = []
        for elt in elts:
            resolved = self.resolve_type(elt)
            if isinstance(resolved, (list, ir.TupleType)):
                raise ParserTypeError(
                    "Nested tuple types are not supported",
                    span=self._get_span(elt),
                    hint="Use a flat pl.Tuple[pl.Tensor[...], pl.Tile[...], ...]",
                )
            types.append(resolved)
        return types

    def _parse_shape(self, shape_node: ast.expr) -> list[int | ir.Expr]:
        """Parse shape from AST node.

        Supports integer literals, variable names that resolve to int values
        from the enclosing scope, pl.dynamic() variables, Scalar IR
        variables from the parser scope, and arbitrary expressions that
        evaluate to lists/tuples via ExprEvaluator.

        Args:
            shape_node: AST node representing shape (tuple or list)

        Returns:
            List of shape dimensions (int for static, ir.Expr for dynamic)

        Raises:
            ParserTypeError: If shape cannot be parsed
        """
        if isinstance(shape_node, (ast.Tuple, ast.List)):
            return self._parse_shape_elements(shape_node.elts)

        # Handle variable name or arbitrary expression that resolves to a list/tuple
        if isinstance(shape_node, ast.Name):
            # Try eval first — handles both simple names and expressions
            success, value = self.expr_evaluator.try_eval_expr(shape_node)
            if success:
                return self._validate_shape_value(value, shape_node.id, self._get_span(shape_node))
            raise ParserTypeError(
                f"Unknown shape variable: {shape_node.id}",
                span=self._get_span(shape_node),
                hint="Use a list like [64, 128] or a variable holding a list",
            )

        # Try evaluating arbitrary expressions (e.g., get_shape(), dims[0:2])
        success, value = self.expr_evaluator.try_eval_expr(shape_node)
        if success:
            return self._validate_shape_value(value, ast.unparse(shape_node), self._get_span(shape_node))

        raise ParserTypeError(
            f"Shape must be a list, tuple, or variable: {ast.unparse(shape_node)}",
            span=self._get_span(shape_node),
            hint="Use a list like [64, 128] or a variable holding a list",
        )

    def _validate_shape_value(self, value: Any, source_name: str, span: ir.Span) -> list[int | ir.Expr]:
        """Validate a Python value as a shape (list/tuple of int/DynVar).

        Args:
            value: Python value to validate
            source_name: Description of value source for error messages
            span: Source span for error messages

        Returns:
            List of shape dimensions
        """
        if not isinstance(value, (list, tuple)):
            raise ParserTypeError(
                f"Shape '{source_name}' must be a list or tuple, got {type(value).__name__}",
                span=span,
                hint="Use a list like [64, 128] or a variable holding a list",
            )

        dims: list[int | ir.Expr] = []
        for i, elem in enumerate(value):
            if isinstance(elem, int):
                dims.append(elem)
            elif isinstance(elem, DynVar):
                if elem._ir_var is None:
                    name = elem.name
                    if name not in self._dyn_var_cache:
                        self._dyn_var_cache[name] = ir.Var(name, ir.ScalarType(DataType.INDEX), span)
                    elem._ir_var = self._dyn_var_cache[name]
                elif elem.name not in self._dyn_var_cache:
                    self._dyn_var_cache[elem.name] = elem._ir_var
                dims.append(elem._ir_var)
            else:
                raise ParserTypeError(
                    f"Shape '{source_name}' element {i} must be int or pl.dynamic(), "
                    f"got {type(elem).__name__}",
                    span=span,
                )
        return dims

    def _validate_dim_value(self, value: Any, source_name: str, span: ir.Span) -> int | ir.Expr:
        """Validate a Python value as a single shape dimension.

        Args:
            value: Python value to validate
            source_name: Description of value source for error messages
            span: Source span for error messages

        Returns:
            int for static dimension, ir.Expr for dynamic
        """
        if isinstance(value, int):
            return value
        if isinstance(value, DynVar):
            if value._ir_var is None:
                name = value.name
                if name not in self._dyn_var_cache:
                    self._dyn_var_cache[name] = ir.Var(name, ir.ScalarType(DataType.INDEX), span)
                value._ir_var = self._dyn_var_cache[name]
            elif value.name not in self._dyn_var_cache:
                self._dyn_var_cache[value.name] = value._ir_var
            return value._ir_var
        raise ParserTypeError(
            f"Shape variable '{source_name}' must be int or pl.dynamic(), got {type(value).__name__}",
            span=span,
        )

    def _parse_shape_elements(self, elts: list[ast.expr]) -> list[int | ir.Expr]:
        """Parse individual shape dimension elements.

        Args:
            elts: List of AST expression nodes for each dimension

        Returns:
            List of shape dimensions
        """
        dims: list[int | ir.Expr] = []
        for elt in elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                dims.append(elt.value)
            elif isinstance(elt, ast.Name):
                dims.append(self._resolve_shape_dim(elt))
            else:
                # Try evaluating arbitrary expressions (e.g., x * 2, len(shape))
                success, value = self.expr_evaluator.try_eval_expr(elt)
                if success:
                    dims.append(self._validate_dim_value(value, ast.unparse(elt), self._get_span(elt)))
                else:
                    raise ParserTypeError(
                        f"Shape dimension must be int literal, variable, or evaluable expression: "
                        f"{ast.unparse(elt)}",
                        span=self._get_span(elt),
                        hint="Use integer literals, variables, or expressions for shape dimensions",
                    )
        return dims

    def _get_span(self, node: ast.AST) -> ir.Span:
        """Get span for an AST node, falling back to unknown."""
        if self.span_tracker is not None:
            return self.span_tracker.get_span(node)
        return ir.Span.unknown()

    def _resolve_shape_dim(self, name_node: ast.Name) -> int | ir.Expr:
        """Resolve a variable name used as a shape dimension.

        Resolution order:
        1. ExprEvaluator (compile-time int or pl.dynamic DynVar from closure)
        2. Parser scope variables (Scalar IR vars from function body)

        Args:
            name_node: AST Name node for the variable

        Returns:
            int for compile-time constants, ir.Expr for dynamic dimensions
        """
        name = name_node.id
        span = self._get_span(name_node)

        # Fast path: direct dict lookup avoids compile+eval overhead for simple names
        if name in self.expr_evaluator.closure_vars:
            return self._validate_dim_value(self.expr_evaluator.closure_vars[name], name, span)

        # 2. Check parser scope (Scalar IR vars in function body)
        if self.scope_lookup:
            var = self.scope_lookup(name)
            if var is not None:
                return var

        raise ParserTypeError(
            f"Unknown shape variable: {name}",
            span=span,
            hint="Use an integer, pl.dynamic() variable, or a Scalar variable defined earlier",
        )

    def _to_ir_shape(self, shape: list[int | ir.Expr]) -> list[int] | list[ir.Expr]:
        """Convert shape to format accepted by IR constructors.

        TensorType/TileType accept either list[int] or list[Expr], not mixed.
        When the shape contains any Expr elements, all int elements are
        converted to ConstInt.

        Args:
            shape: Mixed list of int and ir.Expr dimensions

        Returns:
            Pure int list or pure Expr list
        """
        if all(isinstance(d, int) for d in shape):
            return cast(list[int], shape)

        # Convert all to Expr
        return [ir.ConstInt(d, DataType.INDEX, ir.Span.unknown()) if isinstance(d, int) else d for d in shape]

    def resolve_dtype(self, dtype_node: ast.expr) -> DataType:
        """Resolve dtype annotation.

        Args:
            dtype_node: AST node representing dtype

        Returns:
            DataType enum value

        Raises:
            ValueError: If dtype cannot be resolved
        """
        span = self._get_span(dtype_node)

        # Handle pl.FP16, pl.FP32, etc.
        if isinstance(dtype_node, ast.Attribute):
            dtype_name = dtype_node.attr
            if dtype_name in self._DTYPE_MAP:
                return self._DTYPE_MAP[dtype_name]

            # Distinguish DataType.UNKNOWN from pl.UNKNOWN for error message quality
            if isinstance(dtype_node.value, ast.Name) and dtype_node.value.id == "DataType":
                raise ParserTypeError(
                    f"Unknown DataType: {dtype_name}",
                    span=span,
                    hint="Use a valid dtype like pl.FP32, pl.INT32, etc. Available: "
                    f"{', '.join(self._DTYPE_MAP.keys())}",
                )

            raise ParserTypeError(
                f"Unknown dtype: {dtype_name}",
                span=span,
                hint="Use a valid dtype like pl.FP32, pl.INT32, etc. Available: "
                f"{', '.join(self._DTYPE_MAP.keys())}",
            )

        # Handle simple name like FP16 (if imported directly) or variable from closure
        if isinstance(dtype_node, ast.Name):
            dtype_name = dtype_node.id
            if dtype_name in self._DTYPE_MAP:
                return self._DTYPE_MAP[dtype_name]

            # Try evaluating via ExprEvaluator for DataType values from closure
            success, value = self.expr_evaluator.try_eval_expr(dtype_node)
            if success:
                if isinstance(value, DataType):
                    return value
                raise ParserTypeError(
                    f"Dtype variable '{dtype_name}' must be a DataType, got {type(value).__name__}",
                    span=span,
                    hint="Use a valid dtype like pl.FP32, pl.INT32, etc.",
                )

            raise ParserTypeError(
                f"Unknown dtype: {dtype_name}",
                span=span,
                hint="Use a valid dtype like pl.FP32, pl.INT32, etc. Available: "
                f"{', '.join(self._DTYPE_MAP.keys())}",
            )

        raise ParserTypeError(
            f"Cannot resolve dtype: {ast.unparse(dtype_node)}",
            span=span,
            hint="Use pl.FP32, pl.INT32, or other supported dtype constants",
        )

    def resolve_layout(self, layout_node: ast.expr) -> "ir.TensorLayout":
        """Resolve layout annotation to ir.TensorLayout.

        Args:
            layout_node: AST node representing layout (e.g., pl.NZ, NZ, or a variable)

        Returns:
            TensorLayout enum value

        Raises:
            ParserTypeError: If layout cannot be resolved
        """
        span = self._get_span(layout_node)

        if isinstance(layout_node, ast.Attribute):
            layout_name = layout_node.attr
            if layout_name in self._LAYOUT_MAP:
                return self._LAYOUT_MAP[layout_name]
            raise ParserTypeError(
                f"Unknown layout: {layout_name}",
                span=span,
                hint=f"Use a valid layout: {', '.join(self._LAYOUT_MAP.keys())}",
            )

        if isinstance(layout_node, ast.Name):
            layout_name = layout_node.id
            if layout_name in self._LAYOUT_MAP:
                return self._LAYOUT_MAP[layout_name]

            success, value = self.expr_evaluator.try_eval_expr(layout_node)
            if success:
                if isinstance(value, ir.TensorLayout):
                    return value
                raise ParserTypeError(
                    f"Layout variable '{layout_name}' must be a TensorLayout, got {type(value).__name__}",
                    span=span,
                    hint=f"Use a valid layout: {', '.join(self._LAYOUT_MAP.keys())}",
                )

            raise ParserTypeError(
                f"Unknown layout: {layout_name}",
                span=span,
                hint=f"Use a valid layout: {', '.join(self._LAYOUT_MAP.keys())}",
            )

        raise ParserTypeError(
            f"Cannot resolve layout: {ast.unparse(layout_node)}",
            span=span,
            hint="Use pl.ND, pl.DN, or pl.NZ",
        )

    def validate_annotation_consistency(
        self,
        annotation_type: ir.Type,
        inferred_type: ir.Type,
        var_name: str,
        span: ir.Span | None,
    ) -> None:
        """Validate that a user-written type annotation is consistent with the inferred type.

        Checks type kind, dtype, rank, and static dimension values. Skips validation
        when the inferred type is UnknownType (not enough info to validate).

        Args:
            annotation_type: Type resolved from the user's annotation
            inferred_type: Type inferred from the RHS expression
            var_name: Variable name for error messages
            span: Source span for error location

        Raises:
            ParserTypeError: If the annotation contradicts the inferred type
        """
        if isinstance(inferred_type, ir.UnknownType):
            return

        ann_kind = type(annotation_type)
        inf_kind = type(inferred_type)

        if ann_kind is not inf_kind:
            ann_name = _TYPE_KIND_NAMES.get(ann_kind, ann_kind.__name__)
            inf_name = _TYPE_KIND_NAMES.get(inf_kind, inf_kind.__name__)
            raise ParserTypeError(
                f"Type annotation for '{var_name}' is {ann_name} but expression has type {inf_name}",
                span=span,
                hint=f"Change annotation to: {ir.python_print_type(inferred_type)}",
            )

        # Check dtype for types that expose it (ScalarType and ShapedType)
        self._check_dtype_consistency(annotation_type, inferred_type, var_name, span)

        # ShapedType (TensorType / TileType): also check rank and dims
        if not (isinstance(annotation_type, ir.ShapedType) and isinstance(inferred_type, ir.ShapedType)):
            return

        ann_shape = annotation_type.shape
        inf_shape = inferred_type.shape
        if len(ann_shape) != len(inf_shape):
            raise ParserTypeError(
                f"Type annotation for '{var_name}' has rank {len(ann_shape)} "
                f"but expression has rank {len(inf_shape)}",
                span=span,
                hint=f"Change annotation to: {ir.python_print_type(inferred_type)}",
            )

        for i, (ann_dim, inf_dim) in enumerate(zip(ann_shape, inf_shape)):
            ann_val = _try_get_static_dim(ann_dim)
            inf_val = _try_get_static_dim(inf_dim)
            if ann_val is not None and inf_val is not None and ann_val != inf_val:
                raise ParserTypeError(
                    f"Type annotation for '{var_name}' has shape dimension {i} = {ann_val} "
                    f"but expression has shape dimension {i} = {inf_val}",
                    span=span,
                    hint=f"Change annotation to: {ir.python_print_type(inferred_type)}",
                )

    def _check_dtype_consistency(
        self,
        annotation_type: ir.Type,
        inferred_type: ir.Type,
        var_name: str,
        span: ir.Span | None,
    ) -> None:
        """Raise ParserTypeError if annotation and inferred dtypes conflict.

        Works for both ScalarType and ShapedType (which both expose .dtype).
        """
        ann_dtype: DataType | None = getattr(annotation_type, "dtype", None)
        inf_dtype: DataType | None = getattr(inferred_type, "dtype", None)
        if ann_dtype is not None and inf_dtype is not None and not _dtypes_compatible(ann_dtype, inf_dtype):
            raise ParserTypeError(
                f"Type annotation for '{var_name}' has dtype {ann_dtype} "
                f"but expression has dtype {inf_dtype}",
                span=span,
                hint=f"Change annotation to: {ir.python_print_type(inferred_type)}",
            )

    def _resolve_tile_third_arg(
        self, shape: "list[int] | list[ir.Expr]", dtype: DataType, third: ast.expr
    ) -> "ir.TileType":
        """Resolve legacy helper for a 3-arg Tile annotation."""
        return self._resolve_tile_annotation_args(shape, dtype, [third])

    def _resolve_tile_four_args(
        self,
        shape: "list[int] | list[ir.Expr]",
        dtype: DataType,
        third: ast.expr,
        fourth: ast.expr,
    ) -> "ir.TileType":
        """Resolve legacy helper for a 4-arg Tile annotation."""
        return self._resolve_tile_annotation_args(shape, dtype, [third, fourth])

    def _is_tensorview_node(self, node: ast.expr) -> bool:
        """Check if an AST node is a pl.TensorView(...) call."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (isinstance(func, ast.Attribute) and func.attr == "TensorView") or (
            isinstance(func, ast.Name) and func.id == "TensorView"
        )

    def _resolve_tensorview(self, node: ast.expr) -> "ir.TensorView":
        """Resolve a pl.TensorView(...) AST call to ir.TensorView.

        Args:
            node: AST Call node for pl.TensorView(...)

        Returns:
            ir.TensorView instance

        Raises:
            ParserTypeError: If the TensorView call is malformed
        """
        if not isinstance(node, ast.Call):
            raise ParserTypeError(
                f"Expected pl.TensorView(...) call, got: {ast.unparse(node)}",
                span=self._get_span(node),
                hint="Use pl.TensorView(valid_shape=[...], stride=[...], layout=pl.TensorLayout.NZ)",
            )
        if node.args:
            raise ParserTypeError(
                f"pl.TensorView() does not accept positional arguments, got: {ast.unparse(node)}",
                span=self._get_span(node),
                hint="Use keyword arguments: pl.TensorView(stride=[...], layout=pl.TensorLayout.NZ)",
            )
        tv = ir.TensorView()
        for kw in node.keywords:
            if kw.arg == "valid_shape":
                tv.valid_shape = self._parse_tileview_expr_list(kw.value)
            elif kw.arg == "stride":
                tv.stride = self._parse_tileview_expr_list(kw.value)
            elif kw.arg == "layout":
                tv.layout = self.resolve_layout(kw.value)
            else:
                raise ParserTypeError(
                    f"Unknown TensorView keyword argument: {kw.arg!r}",
                    span=self._get_span(kw),
                    hint="Supported: valid_shape, stride, layout",
                )
        return tv

    def _is_tileview_node(self, node: ast.expr) -> bool:
        """Check if an AST node is a pl.TileView(...) call."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (isinstance(func, ast.Attribute) and func.attr == "TileView") or (
            isinstance(func, ast.Name) and func.id == "TileView"
        )

    def _resolve_tileview(  # noqa: PLR0912
        self,
        node: ast.expr,
        tile_shape: "Sequence[int | ir.Expr] | None" = None,
        memory_space: "ir.MemorySpace | None" = None,
    ) -> "ir.TileView":
        """Resolve a pl.TileView(...) AST call to ir.TileView.

        Args:
            node: AST Call node for pl.TileView(...)
            tile_shape: Optional tile shape to use as default valid_shape when not explicit.
            memory_space: Optional memory space used to fill implicit blayout/slayout/fractal
                defaults for fields not explicitly specified in the annotation.  This mirrors
                the printer's memory-space-aware omission logic so that a round-tripped
                annotation like ``pl.TileView(fractal=512)`` inside ``pl.Mem.Acc`` recovers
                the correct col_major/row_major layouts rather than the Python constructor
                defaults.

        Returns:
            ir.TileView instance

        Raises:
            ParserTypeError: If the TileView call is malformed
        """
        if not isinstance(node, ast.Call):
            raise ParserTypeError(
                f"Expected pl.TileView(...) call, got: {ast.unparse(node)}",
                span=self._get_span(node),
                hint="Use pl.TileView(valid_shape=[...], stride=[...], ...)",
            )
        if node.args:
            raise ParserTypeError(
                f"pl.TileView() does not accept positional arguments, got: {ast.unparse(node)}",
                span=self._get_span(node),
                hint="Use keyword arguments: pl.TileView(valid_shape=[...], stride=[...], ...)",
            )
        # TileView is immutable from Python — accumulate fields here, then
        # construct once at the end. Track which fields were explicitly given
        # so the implicit-defaults pass below knows what to fill in.
        valid_shape: list[ir.Expr] | None = None
        stride: list[ir.Expr] = []
        start_offset: ir.Expr | None = None
        blayout: ir.TileLayout | None = None
        slayout: ir.TileLayout | None = None
        fractal: int | None = None
        pad: ir.PadValue = ir.PadValue.null
        for kw in node.keywords:
            if kw.arg == "valid_shape":
                valid_shape = self._parse_tileview_expr_list(kw.value)
            elif kw.arg == "stride":
                stride = self._parse_tileview_expr_list(kw.value)
            elif kw.arg == "start_offset":
                start_offset = self._parse_tileview_expr(kw.value)
            elif kw.arg == "blayout":
                blayout = self._resolve_tilelayout(kw.value)
            elif kw.arg == "slayout":
                slayout = self._resolve_tilelayout(kw.value)
            elif kw.arg == "fractal":
                val = self._try_resolve_int(kw.value)
                if val is None:
                    raise ParserTypeError(
                        f"TileView fractal must be an integer, got: {ast.unparse(kw.value)}",
                        span=self._get_span(kw.value),
                    )
                fractal = val
            elif kw.arg == "pad":
                pad = self._resolve_padvalue(kw.value)
            else:
                raise ParserTypeError(
                    f"Unknown TileView keyword argument: {kw.arg!r}",
                    span=self._get_span(kw),
                    hint="Supported: valid_shape, stride, start_offset, blayout, slayout, fractal, pad",
                )
        # If valid_shape was not explicitly given, inherit from tile_shape so roundtrip is stable
        if valid_shape is None and tile_shape is not None:
            valid_shape = self._tile_shape_to_expr_list(tile_shape)
        # Apply implicit defaults for any field NOT explicitly set.  The printer omits
        # fields matching the memory-space-aware implicit defaults; the parser must
        # recover them from the same rules — regardless of whether memory_space is
        # present, since shape-derived defaults (e.g. col_major for [N,1]) also apply.
        impl_blayout, impl_slayout, impl_fractal = _implicit_tile_view_defaults(
            valid_shape if valid_shape else (tile_shape or []), memory_space
        )
        return ir.TileView(
            valid_shape=valid_shape if valid_shape is not None else [],
            stride=stride,
            start_offset=start_offset,
            blayout=blayout if blayout is not None else impl_blayout,
            slayout=slayout if slayout is not None else impl_slayout,
            fractal=fractal if fractal is not None else impl_fractal,
            pad=pad,
        )

    def _tile_shape_to_expr_list(self, shape: "Sequence[int | ir.Expr]") -> "list[ir.Expr]":
        """Convert a tile shape (list of int or Expr) to a list of Expr for TileView.valid_shape."""
        result = []
        for dim in shape:
            if isinstance(dim, int):
                result.append(ir.ConstInt(dim, DataType.INDEX, ir.Span.unknown()))
            else:
                result.append(dim)
        return result

    def _parse_tileview_expr_list(self, node: ast.expr) -> list["ir.Expr"]:
        """Parse a list literal of integer expressions for TileView fields."""
        if not isinstance(node, ast.List):
            raise ParserTypeError(
                f"Expected a list, got: {ast.unparse(node)}",
                span=self._get_span(node),
                hint="Use a list like [64, 32]",
            )
        return [self._parse_tileview_expr(elt) for elt in node.elts]

    def _parse_tileview_expr(self, node: ast.expr) -> "ir.Expr":
        """Parse a single expression for a TileView field.

        TileView fields admit arbitrary index expressions, matching what the
        C++ ``ir.TileView`` constructor accepts. Integer constants and bare
        names use fast paths here; richer expressions (calls, arithmetic,
        attribute access, etc.) fall through to the same expression parser
        that the DSL uses everywhere else.
        """
        val = self._try_resolve_int(node)
        if val is not None:
            return ir.ConstInt(val, DataType.INDEX, self._get_span(node))
        if isinstance(node, ast.Name):
            name = node.id
            # 1. Check parser scope first (IR variables from function body)
            if self.scope_lookup:
                var = self.scope_lookup(name)
                if var is not None:
                    return var
            # 2. Check closure variables (DynVar or int from Python scope)
            if name in self.expr_evaluator.closure_vars:
                value = self.expr_evaluator.closure_vars[name]
                if isinstance(value, DynVar):
                    if value._ir_var is None:
                        if value.name not in self._dyn_var_cache:
                            self._dyn_var_cache[value.name] = ir.Var(
                                value.name, ir.ScalarType(DataType.INDEX), self._get_span(node)
                            )
                        value._ir_var = self._dyn_var_cache[value.name]
                    elif value.name not in self._dyn_var_cache:
                        self._dyn_var_cache[value.name] = value._ir_var
                    return value._ir_var
                if isinstance(value, int):
                    return ir.ConstInt(value, DataType.INDEX, self._get_span(node))
                raise ParserTypeError(
                    f"TileView dimension {name!r} is bound to {type(value).__name__}, expected DynVar or int",
                    span=self._get_span(node),
                    hint="Use pl.dynamic() or an integer for TileView dimension variables",
                )
            # Auto-create a dynamic variable for unknown names to support roundtrip with dynamic shapes.
            # When re-parsing printed IR, dynamic vars like M, N are defined as pl.dynamic() at module
            # scope but may not be captured in closure_vars from the decorator frame.
            if name not in self._dyn_var_cache:
                self._dyn_var_cache[name] = ir.Var(name, ir.ScalarType(DataType.INDEX), self._get_span(node))
            return self._dyn_var_cache[name]
        if self._parse_expression is not None:
            # Pre-flight: pl.yield_() emits an ir.YieldStmt to the builder as a side
            # effect during parsing. Type-annotation parsing must stay pure, so reject
            # the call before delegation rather than after — otherwise the spurious
            # YieldStmt is already in the builder when we throw.
            if _is_pl_yield_call(node):
                raise ParserTypeError(
                    "TileView field cannot contain pl.yield_() — it would emit a "
                    "YieldStmt as a side effect of type annotation parsing.",
                    span=self._get_span(node),
                    hint="Use a pure index expression (constants, vars, arithmetic, pl.max/pl.min, ...).",
                )
            result = self._parse_expression(node)
            # Backstop: the delegated parser may still return non-Expr (e.g. None)
            # or non-index expressions (e.g. pl.tile.create(...) returns a Tile).
            # The C++ TileView contract is index expressions only.
            if not isinstance(result, ir.Expr) or not _is_index_expr_type(result.type):
                got = type(result).__name__ if not isinstance(result, ir.Expr) else type(result.type).__name__
                raise ParserTypeError(
                    f"TileView field must be an index expression, got {got}: {ast.unparse(node)}",
                    span=self._get_span(node),
                    hint=(
                        "Use integer literals, dynamic variables, or arithmetic over them "
                        "(pl.max, pl.min, +, -, *, ...)"
                    ),
                )
            return result
        raise ParserTypeError(
            f"TileView expression must be an integer constant or bare variable, got: {ast.unparse(node)}",
            span=self._get_span(node),
            hint="Standalone TypeResolver only handles integer literals and bare names; "
            "richer expressions require the resolver to be wired into an ASTParser.",
        )

    def _resolve_tilelayout(self, node: ast.expr) -> "ir.TileLayout":
        """Resolve pl.TileLayout.xxx to ir.TileLayout."""
        _TILELAYOUT_MAP = {
            "none_box": ir.TileLayout.none_box,
            "row_major": ir.TileLayout.row_major,
            "col_major": ir.TileLayout.col_major,
        }
        if isinstance(node, ast.Attribute):
            if node.attr in _TILELAYOUT_MAP:
                return _TILELAYOUT_MAP[node.attr]
        raise ParserTypeError(
            f"Unknown TileLayout value: {ast.unparse(node)}",
            span=self._get_span(node),
            hint="Use pl.TileLayout.none_box, pl.TileLayout.row_major, or pl.TileLayout.col_major",
        )

    def _resolve_padvalue(self, node: ast.expr) -> "ir.PadValue":
        """Resolve pl.PadValue.xxx to ir.PadValue."""
        _PADVALUE_MAP = {
            "null": ir.PadValue.null,
            "zero": ir.PadValue.zero,
            "max": ir.PadValue.max,
            "min": ir.PadValue.min,
        }
        if isinstance(node, ast.Attribute):
            if node.attr in _PADVALUE_MAP:
                return _PADVALUE_MAP[node.attr]
        raise ParserTypeError(
            f"Unknown PadValue value: {ast.unparse(node)}",
            span=self._get_span(node),
            hint="Use pl.PadValue.null, pl.PadValue.zero, pl.PadValue.max, or pl.PadValue.min",
        )

    def _is_memref_node(self, node: ast.expr) -> bool:
        """Check if an AST node is a pl.MemRef(...) call."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (isinstance(func, ast.Attribute) and func.attr == "MemRef") or (
            isinstance(func, ast.Name) and func.id == "MemRef"
        )

    def _is_memory_space_node(self, node: ast.expr) -> bool:
        """Check if an AST node is a pl.Mem.<space> or pl.MemorySpace.<space> reference."""
        if not isinstance(node, ast.Attribute):
            return False
        value = node.value
        is_memory_space_base = (
            isinstance(value, ast.Attribute) and value.attr in ("MemorySpace", "Mem")
        ) or (isinstance(value, ast.Name) and value.id in ("MemorySpace", "Mem"))
        return is_memory_space_base and node.attr in self._MEMORY_SPACE_MAP

    def _is_layout_node(self, node: ast.expr) -> bool:
        """Check if an AST node resolves to a TensorLayout."""
        try:
            self.resolve_layout(node)
        except ParserTypeError:
            return False
        return True

    def resolve_memref(self, node: ast.expr) -> "ir.MemRef":
        """Resolve a pl.MemRef(base, byte_offset, size) AST call to ir.MemRef.

        Supports:
        - New format: pl.MemRef(base_name, byte_offset, size) — bare name or string ref
        - Legacy format: pl.MemRef(addr, size, id) — integer addr

        Args:
            node: AST Call node for pl.MemRef(...)

        Returns:
            ir.MemRef instance

        Raises:
            ParserTypeError: If the MemRef call is malformed
        """
        if not isinstance(node, ast.Call):
            raise ParserTypeError(
                f"Expected pl.MemRef(...) call, got: {ast.unparse(node)}",
                span=self._get_span(node),
                hint="Use pl.MemRef(base_name, byte_offset, size)",
            )

        span = self._get_span(node)

        if len(node.args) == 3:
            first_arg = node.args[0]

            # New format: pl.MemRef(base_name, byte_offset, size)
            # base_name is a bare Name or a string literal
            base_name = self._try_resolve_memref_base(first_arg)
            if base_name is not None:
                byte_offset = self._resolve_memref_byte_offset(node.args[1])
                size = self._resolve_int_literal(node.args[2], "size", non_negative=True)
                base_var = self._intern_base_ptr(base_name, span)
                return ir.MemRef(base_var, byte_offset, size, span)

            # Legacy fallback: pl.MemRef(addr_int, size, id)
            addr_val = self._resolve_int_literal(node.args[0], "addr")
            size = self._resolve_int_literal(node.args[1], "size", non_negative=True)
            memref_id = self._resolve_int_literal(node.args[2], "id", non_negative=True)
            return ir.MemRef(addr_val, size, memref_id, span)

        if len(node.args) == 4:
            # Legacy: pl.MemRef(memory_space, addr, size, id)
            memory_space = self._resolve_memory_space(node.args[0])
            addr_expr = self._resolve_memref_addr(node.args[1])
            size = self._resolve_int_literal(node.args[2], "size", non_negative=True)
            memref_id = self._resolve_int_literal(node.args[3], "id", non_negative=True)
            return ir.MemRef(memory_space, addr_expr, size, memref_id, span)

        raise ParserTypeError(
            f"pl.MemRef requires 3 arguments (base, byte_offset, size), got {len(node.args)}",
            span=span,
            hint="Use pl.MemRef(base_name, byte_offset, size)",
        )

    def _intern_base_ptr(self, name: str, span: "ir.Span") -> "ir.Var":
        """Get or create a shared Var for a base Ptr name.

        Ensures that two MemRef annotations referencing the same base name
        share the same Var instance, so MemRef.SameAllocation() works after
        parse round-trips. Checks scope_lookup first (for alloc-defined vars),
        then falls back to a per-resolver cache.
        """
        if self.scope_lookup is not None:
            existing = self.scope_lookup(name)
            if existing is not None:
                return existing
        if not hasattr(self, "_base_ptr_cache"):
            self._base_ptr_cache: dict[str, ir.Var] = {}
        if name not in self._base_ptr_cache:
            self._base_ptr_cache[name] = ir.Var(name, ir.PtrType(), span)
        return self._base_ptr_cache[name]

    def _try_resolve_memref_base(self, node: ast.expr) -> str | None:
        """Try to resolve the first arg of pl.MemRef as a base name.

        Returns the name string if the node is a bare Name or string literal,
        None if it's an integer (legacy addr format).
        """
        # String literal: pl.MemRef("mem_ddr_0", ...)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        # Bare name: pl.MemRef(mem_vec_0, ...)
        if isinstance(node, ast.Name):
            return node.id
        # Integer → legacy format, not a base name
        return None

    def _resolve_memref_byte_offset(self, node: ast.expr) -> "ir.Expr":
        """Resolve a MemRef byte_offset to an IR expression (int, variable, or arithmetic)."""
        value = self._try_resolve_int(node)
        if value is not None:
            return ir.ConstInt(value, DataType.INDEX, self._get_span(node))

        # Try resolving as a variable reference (for symbolic offsets)
        if isinstance(node, ast.Name) and self.scope_lookup is not None:
            var = self.scope_lookup(node.id)
            if var is not None:
                return var

        # Handle arithmetic expressions (e.g., var * 4, var + 128)
        if isinstance(node, ast.BinOp):
            lhs = self._resolve_memref_byte_offset(node.left)
            rhs = self._resolve_memref_byte_offset(node.right)
            op_map: dict[type, str] = {
                ast.Add: "__add__",
                ast.Sub: "__sub__",
                ast.Mult: "__mul__",
            }
            method = op_map.get(type(node.op))
            if method is not None:
                return getattr(lhs, method)(rhs)

        raise ParserTypeError(
            f"MemRef byte_offset must be an integer or variable, got: {ast.unparse(node)}",
            span=self._get_span(node),
            hint="Use an integer value for the byte offset, e.g., 0 or 1024",
        )

    def _resolve_memory_space(self, node: ast.expr) -> "ir.MemorySpace":
        """Resolve a memory space AST node (e.g., pl.Mem.DDR or pl.MemorySpace.DDR)."""
        span = self._get_span(node)

        if isinstance(node, ast.Attribute):
            name = node.attr
            if name in self._MEMORY_SPACE_MAP:
                return self._MEMORY_SPACE_MAP[name]
            raise ParserTypeError(
                f"Unknown memory space: {name}",
                span=span,
                hint=f"Use one of: {', '.join(self._MEMORY_SPACE_MAP.keys())}",
            )

        if isinstance(node, ast.Name):
            name = node.id
            if name in self._MEMORY_SPACE_MAP:
                return self._MEMORY_SPACE_MAP[name]

        raise ParserTypeError(
            f"Cannot resolve memory space: {ast.unparse(node)}",
            span=span,
            hint="Use pl.Mem.DDR (or pl.MemorySpace.DDR), pl.Mem.Vec, etc.",
        )

    def _resolve_memref_addr(self, node: ast.expr) -> "ir.Expr":
        """Resolve a MemRef address to an IR expression."""
        value = self._try_resolve_int(node)
        if value is not None:
            return ir.ConstInt(value, DataType.INT64, self._get_span(node))

        raise ParserTypeError(
            f"MemRef address must be an integer, got: {ast.unparse(node)}",
            span=self._get_span(node),
            hint="Use an integer value for the address, e.g., 0 or 1024",
        )

    def _resolve_int_literal(self, node: ast.expr, name: str, *, non_negative: bool = False) -> int:
        """Resolve an AST node to an integer literal."""
        value = self._try_resolve_int(node)
        if value is not None:
            if non_negative and value < 0:
                raise ParserTypeError(
                    f"MemRef {name} must be >= 0, got: {value}",
                    span=self._get_span(node),
                    hint=f"Use a non-negative integer value for {name}",
                )
            return value

        raise ParserTypeError(
            f"MemRef {name} must be an integer, got: {ast.unparse(node)}",
            span=self._get_span(node),
            hint=f"Use an integer value for {name}",
        )

    def _try_resolve_int(self, node: ast.expr) -> int | None:
        """Try to resolve an AST node to a Python int.

        Handles integer literals, unary negation of integer literals,
        and expressions evaluable via ExprEvaluator.

        Args:
            node: AST expression node

        Returns:
            Integer value, or None if the node cannot be resolved to an int
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value

        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
        ):
            return -node.operand.value

        success, value = self.expr_evaluator.try_eval_expr(node)
        if success and isinstance(value, int):
            return value

        return None


__all__ = ["TypeResolver"]
