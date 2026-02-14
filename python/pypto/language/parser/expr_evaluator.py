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

from pypto.pypto_core import ir

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
    Returns raw Python values (int, list, DynVar, DataType, etc.) without
    IR conversion — callers are responsible for converting to IR types.
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

    def _get_span(self, node: ast.AST) -> "ir.Span":
        """Get span for an AST node, falling back to unknown."""
        if self.span_tracker is not None:
            return self.span_tracker.get_span(node)

        return ir.Span.unknown()


__all__ = ["ExprEvaluator"]
