# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scalar types carried alongside the source-ordered tensor metadata walk."""

import ast
from collections.abc import Callable, Mapping
from typing import Any

from pypto.pypto_core import DataType, ir


class ScalarTypes:
    """Infer types with the parser's IR operators, without specializing values.

    Bound locals are represented by typed Vars, even when assigned a literal.
    Only literal operands stay constants, preserving the IR's weak-literal
    promotion rules. Undo journals let the enclosing walker join changed names
    without copying the complete environment at every branch.
    """

    def __init__(
        self,
        seed: Mapping[str, DataType] | None = None,
        fold: Callable[[ast.expr], ast.expr | None] = lambda node: node,
        static_value: Callable[[ast.expr], Any] = lambda node: None,
        tensor_dtype: Callable[[ast.expr], DataType | None] = lambda node: None,
    ) -> None:
        self.types = dict(seed or {})
        self.undo: list[tuple[str, DataType | None]] = []
        self.fold = fold
        self.static_value = static_value
        self.tensor_dtype = tensor_dtype
        self.span = ir.Span.unknown()

    def bind(self, name: str, dtype: DataType | None) -> None:
        self.undo.append((name, self.types.get(name)))
        if dtype is None:
            self.types.pop(name, None)
        else:
            self.types[name] = dtype

    def rewind(self, mark: int) -> dict[str, DataType | None]:
        result = {name: self.types.get(name) for name, _ in self.undo[mark:]}
        for name, previous in reversed(self.undo[mark:]):
            if previous is None:
                self.types.pop(name, None)
            else:
                self.types[name] = previous
        del self.undo[mark:]
        return result

    def join(self, left: Mapping[str, DataType | None], right: Mapping[str, DataType | None]) -> None:
        for name in left.keys() | right.keys():
            before = self.types.get(name)
            lhs, rhs = left.get(name, before), right.get(name, before)
            self.bind(name, lhs if lhs is not None and rhs is not None and lhs == rhs else None)

    def dtype(self, node: ast.expr) -> DataType | None:
        expr = self.expression(node)
        if expr is None or not isinstance(expr.type, ir.ScalarType):
            return None
        dtype = expr.type.dtype
        return DataType.FP32 if dtype == DataType.DEFAULT_CONST_FLOAT else dtype

    def _var(self, dtype: DataType) -> ir.Var:
        return ir.Var("scalar_metadata", ir.ScalarType(dtype), self.span)

    def expression(self, node: ast.expr) -> ir.Expr | None:
        folded = self.fold(node)
        if folded is not None:
            node = folded
        if isinstance(node, ast.Name) and node.id in self.types:
            return self._var(self.types[node.id])
        if isinstance(node, (ast.Constant, ast.Name, ast.Attribute)):
            value = node.value if isinstance(node, ast.Constant) else self.static_value(node)
            if isinstance(value, bool):
                return ir.ConstBool(value, self.span)
            if isinstance(value, int):
                return ir.ConstInt(value, DataType.INDEX, self.span)
            if isinstance(value, float):
                return ir.ConstFloat(value, DataType.DEFAULT_CONST_FLOAT, self.span)
            return None
        if isinstance(node, ast.BinOp):
            lhs, rhs = self.expression(node.left), self.expression(node.right)
            operators: dict[type[ast.operator], Callable[[ir.Expr, ir.Expr, ir.Span], ir.Expr]] = {
                ast.Add: ir.add,
                ast.Sub: ir.sub,
                ast.Mult: ir.mul,
                ast.Div: ir.truediv,
                ast.FloorDiv: ir.floordiv,
                ast.Mod: ir.mod,
                ast.Pow: ir.pow,
                ast.BitAnd: ir.bit_and,
                ast.BitOr: ir.bit_or,
                ast.BitXor: ir.bit_xor,
                ast.LShift: ir.bit_shift_left,
                ast.RShift: ir.bit_shift_right,
            }
            operation = operators.get(type(node.op))
            if lhs is not None and rhs is not None and operation is not None:
                return operation(lhs, rhs, self.span)
        elif isinstance(node, ast.UnaryOp):
            return self._unary(node)
        elif isinstance(node, (ast.Compare, ast.BoolOp)):
            return self._var(DataType.BOOL)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if node.value.attr == "shape":
                return self._var(DataType.INDEX)
        elif isinstance(node, ast.Call):
            return self._call(node)
        return None

    def _unary(self, node: ast.UnaryOp) -> ir.Expr | None:
        operand = self.expression(node.operand)
        if operand is None:
            return None
        if isinstance(node.op, ast.Not):
            return self._var(DataType.BOOL)
        if isinstance(node.op, ast.USub):
            return ir.neg(operand, self.span)
        if isinstance(node.op, ast.Invert):
            return ir.bit_not(operand, self.span)
        return operand

    def _call(self, node: ast.Call) -> ir.Expr | None:
        name = ast.unparse(node.func)
        keywords = {kw.arg: kw.value for kw in node.keywords}
        if name in {"pl.tensor.dim", "pl.dim"}:
            return self._var(DataType.INDEX)
        if name == "pl.const" and len(node.args) == 2 and not node.keywords:
            dtype = self.static_value(node.args[1])
            value_node = node.args[0]
            negate = isinstance(value_node, ast.UnaryOp) and isinstance(value_node.op, ast.USub)
            if negate and isinstance(value_node, ast.UnaryOp):
                value_node = value_node.operand
            if (
                isinstance(dtype, DataType)
                and isinstance(value_node, ast.Constant)
                and isinstance(value_node.value, (int, float))
            ):
                value = -value_node.value if negate else value_node.value
                # Match parse_typed_constant, including typed literal promotion
                # when the result participates in a larger scalar expression.
                if isinstance(value, float):
                    return ir.ConstFloat(value, dtype, self.span)
                return ir.ConstInt(value, dtype, self.span)
        if name in {"pl.cast", "pl.scalar.cast"}:
            dtype_node = node.args[1] if len(node.args) > 1 else keywords.get("dtype")
            dtype = self.static_value(dtype_node) if dtype_node is not None else None
            return self._var(dtype) if isinstance(dtype, DataType) else None
        if name in {"pl.read", "pl.tensor.read", "pl.tile.read"}:
            operand_name = {"pl.read": "src", "pl.tensor.read": "tensor", "pl.tile.read": "tile"}[name]
            operand = node.args[0] if node.args else keywords.get(operand_name)
            dtype = self.tensor_dtype(operand) if operand is not None else None
            return self._var(dtype) if dtype is not None else None
        if name in {"pl.min", "pl.max"} and len(node.args) == 2:
            lhs, rhs = (self.expression(arg) for arg in node.args)
            if lhs is not None and rhs is not None:
                return (ir.min_ if name == "pl.min" else ir.max_)(lhs, rhs, self.span)
        return None

    def update(self, statement: ast.stmt) -> None:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            dtype = self.dtype(statement.value) if statement.value is not None else None
            if isinstance(statement, ast.AnnAssign):
                annotation = statement.annotation
                if isinstance(annotation, ast.Subscript):
                    declared = self.static_value(annotation.slice)
                    if isinstance(declared, DataType):
                        dtype = declared
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                        self.bind(name.id, dtype if isinstance(target, ast.Name) else None)
        elif isinstance(statement, ast.For):
            for name in ast.walk(statement.target):
                if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                    self.bind(name.id, DataType.INDEX)
        elif isinstance(statement, ast.Delete):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    self.bind(target.id, None)
