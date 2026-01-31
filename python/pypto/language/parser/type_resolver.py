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

from pypto.pypto_core import DataType, ir

from .diagnostics import ParserTypeError


class TypeResolver:
    """Resolves Python type annotations to IR types."""

    def __init__(self):
        """Initialize type resolver."""
        # Map of dtype names to DataType enum values
        self.dtype_map = {
            "FP4": DataType.FP4,
            "FP8": DataType.FP8,
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

    def resolve_type(self, type_node: ast.expr) -> ir.Type:
        """Resolve AST type annotation to ir.Type.

        Args:
            type_node: AST expression representing the type annotation

        Returns:
            Corresponding IR type

        Raises:
            ValueError: If type annotation cannot be resolved
        """
        # Handle pl.Tensor[[64, 128], pl.FP16] or pl.Tile[[64, 64], pl.FP32] subscript notation
        if isinstance(type_node, ast.Subscript):
            return self._resolve_subscript_type(type_node)

        # Handle pl.Tensor((64, 128), pl.FP16) call notation (legacy)
        if isinstance(type_node, ast.Call):
            return self._resolve_call_type(type_node)

        # Handle attribute access like pl.Tensor
        if isinstance(type_node, ast.Attribute):
            raise ParserTypeError(
                f"Incomplete type annotation: {ast.unparse(type_node)}",
                hint="Use pl.Tensor[[shape], dtype] or pl.Tile[[shape], dtype]",
            )

        raise ParserTypeError(
            f"Unsupported type annotation: {ast.unparse(type_node)}",
            hint="Use pl.Tensor[[shape], dtype] or pl.Tile[[shape], dtype]",
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

        # Check if it's Tensor or Tile
        type_name = None
        if isinstance(value, ast.Attribute):
            if value.attr in ("Tensor", "Tile"):
                type_name = value.attr
        elif isinstance(value, ast.Name):
            if value.id in ("Tensor", "Tile"):
                type_name = value.id

        if type_name is None:
            raise ParserTypeError(
                f"Unknown type in subscript: {ast.unparse(value)}",
                hint="Use pl.Tensor for tensor types or pl.Tile for tile types",
            )

        # Parse the subscript: should be a tuple (shape, dtype)
        slice_value = subscript_node.slice
        if not isinstance(slice_value, ast.Tuple) or len(slice_value.elts) != 2:
            raise ParserTypeError(
                f"{type_name} subscript requires [shape, dtype], got: {ast.unparse(slice_value)}",
                hint=f"Use pl.{type_name}[[shape], dtype] format, e.g., pl.{type_name}[[64, 128], pl.FP32]",
            )

        shape_node = slice_value.elts[0]
        dtype_node = slice_value.elts[1]

        # Parse shape
        shape = self._parse_shape(shape_node)

        # Parse dtype
        dtype = self.resolve_dtype(dtype_node)

        # Create appropriate type
        if type_name == "Tile":
            return ir.TileType(shape, dtype)
        else:
            return ir.TensorType(shape, dtype)

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

        raise ParserTypeError(
            f"Unknown type constructor: {ast.unparse(func)}",
            hint="Use pl.Tensor[[shape], dtype] for tensor types or pl.Tile[[shape], dtype] for tile types",
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

        # Parse shape (first argument)
        shape_node = call_node.args[0]
        shape = self._parse_shape(shape_node)

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

        # Parse shape (first argument)
        shape_node = call_node.args[0]
        shape = self._parse_shape(shape_node)

        # Parse dtype (second argument)
        dtype_node = call_node.args[1]
        dtype = self.resolve_dtype(dtype_node)

        # Create TileType
        return ir.TileType(shape, dtype)

    def _parse_shape(self, shape_node: ast.expr) -> list[int]:
        """Parse shape from AST node.

        Args:
            shape_node: AST node representing shape (tuple or list)

        Returns:
            List of shape dimensions

        Raises:
            ValueError: If shape cannot be parsed
        """
        # Handle tuple like (64, 128)
        if isinstance(shape_node, ast.Tuple):
            dims = []
            for elt in shape_node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    dims.append(elt.value)
                else:
                    raise ParserTypeError(
                        f"Shape dimension must be constant: {ast.unparse(elt)}",
                        hint="Use integer literals for shape dimensions, e.g., [64, 128]",
                    )
            return dims

        # Handle list like [64, 128]
        if isinstance(shape_node, ast.List):
            dims = []
            for elt in shape_node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    dims.append(elt.value)
                else:
                    raise ParserTypeError(
                        f"Shape dimension must be constant: {ast.unparse(elt)}",
                        hint="Use integer literals for shape dimensions, e.g., [64, 128]",
                    )
            return dims

        raise ParserTypeError(
            f"Shape must be tuple or list: {ast.unparse(shape_node)}",
            hint="Use a list or tuple for shape, e.g., [64, 128]",
        )

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
