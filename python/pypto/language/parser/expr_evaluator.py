# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Centralized expression evaluator for resolving Python expressions against closure variables."""

import ast
from typing import TYPE_CHECKING, Any

from pypto.pypto_core import DataType, ir

from ..typing.dynamic import DynVar
from .diagnostics import ParserTypeError

if TYPE_CHECKING:
    from .span_tracker import SpanTracker

# Safe subset of builtins allowed during expression evaluation
_SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "max": max,
    "min": min,
    "abs": abs,
    "sum": sum,
    "round": round,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "range": range,
    "isinstance": isinstance,
    "type": type,
    "True": True,
    "False": False,
    "None": None,
}


class ExprEvaluator:
    """Evaluates Python AST expressions against closure variables.

    Uses Python's eval() with a restricted builtins whitelist for safety.
    Can return raw Python values or convert them to IR expressions.
    """

    def __init__(
        self,
        closure_vars: dict[str, Any],
        span_tracker: "SpanTracker | None" = None,
    ):
        """Initialize expression evaluator.

        Args:
            closure_vars: Variables from the enclosing scope
            span_tracker: Optional span tracker for source locations in errors
        """
        self.closure_vars = closure_vars
        self.span_tracker = span_tracker

    def eval_expr(self, node: ast.expr) -> Any:
        """Evaluate an AST expression node against closure variables.

        Args:
            node: AST expression node to evaluate

        Returns:
            The Python value resulting from evaluation

        Raises:
            ParserTypeError: If expression cannot be evaluated
        """
        span = self._get_span(node)
        expr_str = ast.unparse(node)

        try:
            code = compile(ast.Expression(body=node), "<pypto-eval>", "eval")
            # Security note: closure_vars come from the user's own enclosing Python scope.
            # The DSL parser is not a sandbox — users already have full control of the
            # Python process. The builtins whitelist prevents accidental access to dangerous
            # builtins (open, __import__, exec) but does not prevent calling methods on
            # objects the user placed in scope, which is by design.
            return eval(code, {"__builtins__": _SAFE_BUILTINS}, dict(self.closure_vars))  # noqa: S307
        except NameError as e:
            raise ParserTypeError(
                f"Cannot resolve expression '{expr_str}': {e}",
                span=span,
                hint="Make sure the variable is defined in the enclosing scope",
            ) from e
        except Exception as e:
            raise ParserTypeError(
                f"Failed to evaluate expression '{expr_str}': {e}",
                span=span,
            ) from e

    def try_eval_expr(self, node: ast.expr) -> tuple[bool, Any]:
        """Try to evaluate an AST expression, returning success status.

        Non-throwing variant of eval_expr for cases where evaluation failure
        should fall through to other resolution strategies.

        Args:
            node: AST expression node to evaluate

        Returns:
            Tuple of (success, value). On failure, value is None.
        """
        try:
            return (True, self.eval_expr(node))
        except ParserTypeError:
            return (False, None)

    def python_value_to_ir(self, value: Any, span: ir.Span) -> ir.Expr:
        """Convert a Python value to an IR expression.

        Args:
            value: Python value (bool, int, float, ir.Expr, DynVar, list, or tuple)
            span: Source span for the expression

        Returns:
            IR expression representing the value
        """
        # bool before int because isinstance(True, int) is True
        if isinstance(value, bool):
            return ir.ConstBool(value, span)
        if isinstance(value, int):
            return ir.ConstInt(value, DataType.DEFAULT_CONST_INT, span)
        if isinstance(value, float):
            return ir.ConstFloat(value, DataType.DEFAULT_CONST_FLOAT, span)
        if isinstance(value, ir.Expr):
            return value
        if isinstance(value, DynVar):
            return ir.Var(value.name, ir.ScalarType(DataType.INDEX), span)
        if isinstance(value, (list, tuple)):
            return ir.MakeTuple([self.python_value_to_ir(elt, span) for elt in value], span)
        raise ParserTypeError(
            f"Unsupported closure variable type: {type(value).__name__}",
            span=span,
            hint="Closure variables must be int, float, bool, list, tuple, or IR expressions",
        )

    def try_eval_as_ir(self, node: ast.expr) -> ir.Expr | None:
        """Try to evaluate an AST node and convert to an IR expression.

        Combines try_eval_expr + python_value_to_ir. Returns None when the
        name cannot be resolved from closure variables.

        Args:
            node: AST expression node to evaluate

        Returns:
            IR expression, or None if the name cannot be resolved.

        Raises:
            ParserTypeError: If the value is resolved but has an unsupported type.
        """
        success, value = self.try_eval_expr(node)
        if not success:
            return None
        return self.python_value_to_ir(value, self._get_span(node))

    def _get_span(self, node: ast.AST) -> "ir.Span":
        """Get span for an AST node, falling back to unknown."""
        if self.span_tracker is not None:
            return self.span_tracker.get_span(node)

        return ir.Span.unknown()


__all__ = ["ExprEvaluator"]
