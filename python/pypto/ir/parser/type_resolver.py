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
        # Handle pl.Tensor[[64, 128], pl.FP16] subscript notation
        if isinstance(type_node, ast.Subscript):
            return self._resolve_subscript_type(type_node)

        # Handle pl.Tensor((64, 128), pl.FP16) call notation (legacy)
        if isinstance(type_node, ast.Call):
            return self._resolve_call_type(type_node)

        # Handle attribute access like pl.Tensor
        if isinstance(type_node, ast.Attribute):
            raise ValueError(f"Incomplete type annotation: {ast.unparse(type_node)}")

        raise ValueError(f"Unsupported type annotation: {ast.unparse(type_node)}")

    def _resolve_subscript_type(self, subscript_node: ast.Subscript) -> ir.Type:
        """Resolve subscript type annotation like pl.Tensor[[64, 128], pl.FP16].

        Args:
            subscript_node: AST Subscript node

        Returns:
            IR type

        Raises:
            ValueError: If subscript cannot be resolved to a type
        """
        # Get the base (should be pl.Tensor or Tensor)
        value = subscript_node.value

        # Check if it's Tensor
        is_tensor = False
        if isinstance(value, ast.Attribute) and value.attr == "Tensor":
            is_tensor = True
        elif isinstance(value, ast.Name) and value.id == "Tensor":
            is_tensor = True

        if not is_tensor:
            raise ValueError(f"Unknown type in subscript: {ast.unparse(value)}")

        # Parse the subscript: should be a tuple (shape, dtype)
        slice_value = subscript_node.slice
        if not isinstance(slice_value, ast.Tuple) or len(slice_value.elts) != 2:
            raise ValueError(f"Tensor subscript requires [shape, dtype], got: {ast.unparse(slice_value)}")

        shape_node = slice_value.elts[0]
        dtype_node = slice_value.elts[1]

        # Parse shape
        shape = self._parse_shape(shape_node)

        # Parse dtype
        dtype = self.resolve_dtype(dtype_node)

        # Create TensorType
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

        raise ValueError(f"Unknown type constructor: {ast.unparse(func)}")

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
            raise ValueError(f"Tensor type requires shape and dtype arguments, got {len(call_node.args)}")

        # Parse shape (first argument)
        shape_node = call_node.args[0]
        shape = self._parse_shape(shape_node)

        # Parse dtype (second argument)
        dtype_node = call_node.args[1]
        dtype = self.resolve_dtype(dtype_node)

        # Create TensorType
        return ir.TensorType(shape, dtype)

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
                    raise ValueError(f"Shape dimension must be constant: {ast.unparse(elt)}")
            return dims

        # Handle list like [64, 128]
        if isinstance(shape_node, ast.List):
            dims = []
            for elt in shape_node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    dims.append(elt.value)
                else:
                    raise ValueError(f"Shape dimension must be constant: {ast.unparse(elt)}")
            return dims

        raise ValueError(f"Shape must be tuple or list: {ast.unparse(shape_node)}")

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
            raise ValueError(f"Unknown dtype: {dtype_name}")

        # Handle DataType.FP16, etc.
        if isinstance(dtype_node, ast.Attribute):
            if isinstance(dtype_node.value, ast.Name) and dtype_node.value.id == "DataType":
                dtype_name = dtype_node.attr
                if dtype_name in self.dtype_map:
                    return self.dtype_map[dtype_name]
                raise ValueError(f"Unknown DataType: {dtype_name}")

        # Handle simple name like FP16 (if imported directly)
        if isinstance(dtype_node, ast.Name):
            dtype_name = dtype_node.id
            if dtype_name in self.dtype_map:
                return self.dtype_map[dtype_name]
            raise ValueError(f"Unknown dtype: {dtype_name}")

        raise ValueError(f"Cannot resolve dtype: {ast.unparse(dtype_node)}")


__all__ = ["TypeResolver"]
