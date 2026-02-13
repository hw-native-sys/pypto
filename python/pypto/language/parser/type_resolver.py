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
from typing import Any, Callable

from pypto.language.typing.dynamic import DynVar
from pypto.pypto_core import DataType, ir

from .diagnostics import ParserTypeError


class TypeResolver:
    """Resolves Python type annotations to IR types."""

    def __init__(
        self,
        closure_vars: dict[str, Any] | None = None,
        scope_lookup: Callable[[str], Any | None] | None = None,
    ):
        """Initialize type resolver.

        Args:
            closure_vars: Variables from the enclosing scope (for compile-time
                dynamic shapes and pl.dynamic() variables)
            scope_lookup: Callback to look up variables in the parser scope
                (for Scalar IR vars used in inline annotations)
        """
        self.closure_vars = closure_vars or {}
        self.scope_lookup = scope_lookup
        # Map of dtype names to DataType enum values
        self.dtype_map = {
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
        }

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
                hint="Use pl.Tensor[[shape], dtype], pl.Tile[[shape], dtype], or pl.Scalar[dtype]",
            )

        raise ParserTypeError(
            f"Unsupported type annotation: {ast.unparse(type_node)}",
            hint="Use pl.Tensor[[shape], dtype], pl.Tile[[shape], dtype], or pl.Scalar[dtype]",
        )

    def _resolve_subscript_type(self, subscript_node: ast.Subscript) -> ir.Type:
        """Resolve subscript type annotation like pl.Tensor[[64, 128], pl.FP16] or pl.Tile[[64, 64], pl.FP32].

        Args:
            subscript_node: AST Subscript node

        Returns:
            IR type

        Raises:
            ValueError: If subscript cannot be resolved to a type
        """
        # Get the base (should be pl.Tensor/Tensor or pl.Tile/Tile)
        value = subscript_node.value

        # Check if it's Tensor, Tile, or Scalar
        type_name = None
        if isinstance(value, ast.Attribute):
            if value.attr in ("Tensor", "Tile", "Scalar"):
                type_name = value.attr
        elif isinstance(value, ast.Name):
            if value.id in ("Tensor", "Tile", "Scalar"):
                type_name = value.id

        if type_name is None:
            raise ParserTypeError(
                f"Unknown type in subscript: {ast.unparse(value)}",
                hint="Use pl.Tensor for tensor types, pl.Tile for tile types, or pl.Scalar for scalar types",
            )

        # Parse the subscript
        slice_value = subscript_node.slice

        # Scalar only needs dtype, not shape
        if type_name == "Scalar":
            # Scalar subscript format: pl.Scalar[dtype]
            # slice_value should be a single dtype, not a tuple
            dtype = self.resolve_dtype(slice_value)
            return ir.ScalarType(dtype)

        # Tensor/Tile need [shape, dtype] tuple
        if not isinstance(slice_value, ast.Tuple) or len(slice_value.elts) != 2:
            raise ParserTypeError(
                f"{type_name} subscript requires [shape, dtype], got: {ast.unparse(slice_value)}",
                hint=f"Use pl.{type_name}[[shape], dtype] format, e.g., pl.{type_name}[[64, 128], pl.FP32]",
            )

        shape_node = slice_value.elts[0]
        dtype_node = slice_value.elts[1]

        # Parse shape and normalize for IR constructors
        shape = self._to_ir_shape(self._parse_shape(shape_node))

        # Parse dtype
        dtype = self.resolve_dtype(dtype_node)

        # Create appropriate type
        if type_name == "Tile":
            return ir.TileType(shape, dtype)
        else:
            return ir.TensorType(shape, dtype)

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
        # Get the function being called
        func = call_node.func

        # Handle pl.Tensor(...) or Tensor(...)
        if isinstance(func, ast.Attribute) and func.attr == "Tensor":
            return self._resolve_tensor_type(call_node)

        if isinstance(func, ast.Name) and func.id == "Tensor":
            return self._resolve_tensor_type(call_node)

        # Handle pl.Tile(...) or Tile(...)
        if isinstance(func, ast.Attribute) and func.attr == "Tile":
            return self._resolve_tile_type(call_node)

        if isinstance(func, ast.Name) and func.id == "Tile":
            return self._resolve_tile_type(call_node)

        # Handle pl.Scalar(...) or Scalar(...)
        if isinstance(func, ast.Attribute) and func.attr == "Scalar":
            return self._resolve_scalar_type(call_node)

        if isinstance(func, ast.Name) and func.id == "Scalar":
            return self._resolve_scalar_type(call_node)

        raise ParserTypeError(
            f"Unknown type constructor: {ast.unparse(func)}",
            hint="Use pl.Tensor[[shape], dtype], pl.Tile[[shape], dtype], or pl.Scalar[dtype]",
        )

    def _resolve_tensor_type(self, call_node: ast.Call) -> ir.TensorType:
        """Resolve pl.Tensor((shape), dtype) annotation (legacy).

        Args:
            call_node: AST Call node for Tensor constructor

        Returns:
            TensorType

        Raises:
            ValueError: If tensor type annotation is malformed
        """
        if len(call_node.args) < 2:
            raise ParserTypeError(
                f"Tensor type requires shape and dtype arguments, got {len(call_node.args)}",
                hint="Use pl.Tensor[[shape], dtype] format, e.g., pl.Tensor[[64, 128], pl.FP32]",
            )

        # Parse shape (first argument) and normalize
        shape_node = call_node.args[0]
        shape = self._to_ir_shape(self._parse_shape(shape_node))

        # Parse dtype (second argument)
        dtype_node = call_node.args[1]
        dtype = self.resolve_dtype(dtype_node)

        # Create TensorType
        return ir.TensorType(shape, dtype)

    def _resolve_tile_type(self, call_node: ast.Call) -> ir.TileType:
        """Resolve pl.Tile((shape), dtype) annotation (legacy).

        Args:
            call_node: AST Call node for Tile constructor

        Returns:
            TileType

        Raises:
            ValueError: If tile type annotation is malformed
        """
        if len(call_node.args) < 2:
            raise ParserTypeError(
                f"Tile type requires shape and dtype arguments, got {len(call_node.args)}",
                hint="Use pl.Tile[[shape], dtype] format, e.g., pl.Tile[[64, 64], pl.FP32]",
            )

        # Parse shape (first argument) and normalize
        shape_node = call_node.args[0]
        shape = self._to_ir_shape(self._parse_shape(shape_node))

        # Parse dtype (second argument)
        dtype_node = call_node.args[1]
        dtype = self.resolve_dtype(dtype_node)

        # Create TileType
        return ir.TileType(shape, dtype)

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
                hint="Use pl.Scalar[dtype] format, e.g., pl.Scalar[pl.FP32]",
            )

        # Parse dtype (first argument)
        dtype_node = call_node.args[0]
        dtype = self.resolve_dtype(dtype_node)

        # Create ScalarType
        return ir.ScalarType(dtype)

    def _parse_shape(self, shape_node: ast.expr) -> list[int | ir.Expr]:
        """Parse shape from AST node.

        Supports integer literals, variable names that resolve to int values
        from the enclosing scope, pl.dynamic() variables, and Scalar IR
        variables from the parser scope.

        Args:
            shape_node: AST node representing shape (tuple or list)

        Returns:
            List of shape dimensions (int for static, ir.Expr for dynamic)

        Raises:
            ParserTypeError: If shape cannot be parsed
        """
        if isinstance(shape_node, (ast.Tuple, ast.List)):
            dims: list[int | ir.Expr] = []
            for elt in shape_node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    dims.append(elt.value)
                elif isinstance(elt, ast.Name):
                    dims.append(self._resolve_shape_dim(elt.id))
                else:
                    raise ParserTypeError(
                        f"Shape dimension must be int literal or variable: {ast.unparse(elt)}",
                        hint="Use integer literals or variables for shape dimensions",
                    )
            return dims

        raise ParserTypeError(
            f"Shape must be tuple or list: {ast.unparse(shape_node)}",
            hint="Use a list or tuple for shape, e.g., [64, 128]",
        )

    def _resolve_shape_dim(self, name: str) -> int | ir.Expr:
        """Resolve a variable name used as a shape dimension.

        Resolution order:
        1. Closure variables (compile-time int or pl.dynamic DynVar)
        2. Parser scope variables (Scalar IR vars from function body)

        Args:
            name: Variable name

        Returns:
            int for compile-time constants, ir.Expr for dynamic dimensions
        """
        # 1. Check closure variables (compile-time dynamic)
        if name in self.closure_vars:
            value = self.closure_vars[name]
            if isinstance(value, int):
                return value
            if isinstance(value, DynVar):
                return ir.Var(
                    value.name,
                    ir.ScalarType(DataType.INT64),
                    ir.Span.unknown(),
                )
            raise ParserTypeError(
                f"Shape variable '{name}' must be int or pl.dynamic(), got {type(value).__name__}",
            )

        # 2. Check parser scope (Scalar IR vars in function body)
        if self.scope_lookup:
            var = self.scope_lookup(name)
            if var is not None:
                return var

        raise ParserTypeError(
            f"Unknown shape variable: {name}",
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
            return shape  # type: ignore[return-value]

        # Convert all to Expr
        return [ir.ConstInt(d, DataType.INT64, ir.Span.unknown()) if isinstance(d, int) else d for d in shape]

    def resolve_dtype(self, dtype_node: ast.expr) -> DataType:
        """Resolve dtype annotation.

        Args:
            dtype_node: AST node representing dtype

        Returns:
            DataType enum value

        Raises:
            ValueError: If dtype cannot be resolved
        """
        # Handle pl.FP16, pl.FP32, etc.
        if isinstance(dtype_node, ast.Attribute):
            dtype_name = dtype_node.attr
            if dtype_name in self.dtype_map:
                return self.dtype_map[dtype_name]

            # Check if it's DataType.FP16
            if isinstance(dtype_node.value, ast.Name) and dtype_node.value.id == "DataType":
                if dtype_name in self.dtype_map:
                    return self.dtype_map[dtype_name]
                raise ParserTypeError(
                    f"Unknown DataType: {dtype_name}",
                    hint="Use a valid dtype like pl.FP32, pl.INT32, etc. Available: "
                    f"{', '.join(self.dtype_map.keys())}",
                )

            raise ParserTypeError(
                f"Unknown dtype: {dtype_name}",
                hint="Use a valid dtype like pl.FP32, pl.INT32, etc. Available: "
                f"{', '.join(self.dtype_map.keys())}",
            )

        # Handle simple name like FP16 (if imported directly)
        if isinstance(dtype_node, ast.Name):
            dtype_name = dtype_node.id
            if dtype_name in self.dtype_map:
                return self.dtype_map[dtype_name]
            raise ParserTypeError(
                f"Unknown dtype: {dtype_name}",
                hint="Use a valid dtype like pl.FP32, pl.INT32, etc. Available: "
                f"{', '.join(self.dtype_map.keys())}",
            )

        raise ParserTypeError(
            f"Cannot resolve dtype: {ast.unparse(dtype_node)}",
            hint="Use pl.FP32, pl.INT32, or other supported dtype constants",
        )


__all__ = ["TypeResolver"]
