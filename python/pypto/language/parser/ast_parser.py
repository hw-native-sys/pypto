# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AST parsing for converting Python DSL to IR builder calls."""

import ast
from typing import Any, Optional

from pypto.ir import IRBuilder
from pypto.ir import op as ir_op
from pypto.pypto_core import DataType, ir

from .diagnostics import (
    InvalidOperationError,
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)
from .scope_manager import ScopeManager
from .span_tracker import SpanTracker
from .type_resolver import TypeResolver

# TODO(syfeng): Enhance type checking and fix all type issues.
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportReturnType=false
# pyright: reportOptionalOperand=false, reportOperatorIssue=false


class ASTParser:
    """Parses Python AST and builds IR using IRBuilder."""

    def __init__(
        self,
        source_file: str,
        source_lines: list[str],
        line_offset: int = 0,
        col_offset: int = 0,
        global_vars: Optional[dict[str, ir.GlobalVar]] = None,
        gvar_to_func: Optional[dict[ir.GlobalVar, ir.Function]] = None,
        strict_ssa: bool = False,
    ):
        """Initialize AST parser.

        Args:
            source_file: Path to source file
            source_lines: Lines of source code (dedented for parsing)
            line_offset: Line number offset to add to AST line numbers (for dedented code)
            col_offset: Column offset to add to AST column numbers (for dedented code)
            global_vars: Optional map of function names to GlobalVars for cross-function calls
            gvar_to_func: Optional map of GlobalVars to parsed Functions for type inference
            strict_ssa: If True, enforce SSA (single assignment). If False (default), allow reassignment.
        """
        self.span_tracker = SpanTracker(source_file, source_lines, line_offset, col_offset)
        self.scope_manager = ScopeManager(strict_ssa=strict_ssa)
        self.type_resolver = TypeResolver()
        self.builder = IRBuilder()
        self.global_vars = global_vars or {}  # Track GlobalVars for cross-function calls
        self.gvar_to_func = gvar_to_func or {}  # Track parsed functions for type inference

        # Track context for handling yields and returns
        self.in_for_loop = False
        self.in_while_loop = False
        self.in_if_stmt = False
        self.current_if_builder = None
        self.current_loop_builder = None

    def parse_function(
        self,
        func_def: ast.FunctionDef,
        func_type: ir.FunctionType = ir.FunctionType.Opaque,
    ) -> ir.Function:
        """Parse function definition and build IR.

        Args:
            func_def: AST FunctionDef node
            func_type: Function type (default: Opaque)

        Returns:
            IR Function object
        """
        func_name = func_def.name
        func_span = self.span_tracker.get_span(func_def)

        # Enter function scope
        self.scope_manager.enter_scope("function")

        # Begin building function
        with self.builder.function(func_name, func_span, type=func_type) as f:
            # Parse parameters (skip 'self' if it's the first parameter without annotation)
            for arg in func_def.args.args:
                param_name = arg.arg

                # Skip 'self' parameter if it has no annotation (shouldn't happen if decorator stripped it)
                if param_name == "self" and arg.annotation is None:
                    continue

                if arg.annotation is None:
                    raise ParserTypeError(
                        f"Parameter '{param_name}' missing type annotation",
                        span=self.span_tracker.get_span(arg),
                        hint="Add a type annotation like: x: pl.Tensor[[64], pl.FP32]",
                    )

                param_type = self.type_resolver.resolve_type(arg.annotation)
                param_span = self.span_tracker.get_span(arg)

                if isinstance(param_type, list):
                    raise ParserSyntaxError(
                        f"Parameter '{param_name}' cannot have tuple type annotation",
                        span=param_span,
                        hint="Tuple types are only supported as return types",
                    )

                # Add parameter to function
                param_var = f.param(param_name, param_type, param_span)

                # Register in scope
                self.scope_manager.define_var(param_name, param_var, allow_redef=True)

            # Parse return type
            if func_def.returns:
                return_type = self.type_resolver.resolve_type(func_def.returns)
                if isinstance(return_type, list):
                    # tuple[T1, T2, ...] -> multiple return types
                    for rt in return_type:
                        f.return_type(rt)
                else:
                    f.return_type(return_type)

            # Parse function body (skip docstrings)
            for i, stmt in enumerate(func_def.body):
                # Skip docstrings (string constants as first statement or after decorators)
                if i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    if isinstance(stmt.value.value, str):
                        continue  # Skip docstring
                self.parse_statement(stmt)

        # Exit function scope
        self.scope_manager.exit_scope()

        return f.get_result()

    def parse_statement(self, stmt: ast.stmt) -> None:
        """Parse a statement node.

        Args:
            stmt: AST statement node
        """
        if isinstance(stmt, ast.AnnAssign):
            self.parse_annotated_assignment(stmt)
        elif isinstance(stmt, ast.Assign):
            self.parse_assignment(stmt)
        elif isinstance(stmt, ast.For):
            self.parse_for_loop(stmt)
        elif isinstance(stmt, ast.While):
            self.parse_while_loop(stmt)
        elif isinstance(stmt, ast.If):
            self.parse_if_statement(stmt)
        elif isinstance(stmt, ast.Return):
            self.parse_return(stmt)
        elif isinstance(stmt, ast.Expr):
            self.parse_evaluation_statement(stmt)
        else:
            raise UnsupportedFeatureError(
                f"Unsupported statement type: {type(stmt).__name__}",
                span=self.span_tracker.get_span(stmt),
                hint="Only assignments, for loops, while loops, if statements, "
                "and returns are supported in DSL functions",
            )

    def parse_annotated_assignment(self, stmt: ast.AnnAssign) -> None:
        """Parse annotated assignment: var: type = value.

        Args:
            stmt: AnnAssign AST node
        """
        if not isinstance(stmt.target, ast.Name):
            raise ParserSyntaxError(
                "Only simple variable assignments supported",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use a simple variable name for assignment targets",
            )

        var_name = stmt.target.id
        span = self.span_tracker.get_span(stmt)

        # Check if this is a yield assignment: var: type = pl.yield_(...)
        if isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Attribute) and func.attr == "yield_":
                # Handle yield assignment
                yield_exprs = []
                for arg in stmt.value.args:
                    expr = self.parse_expression(arg)
                    yield_exprs.append(expr)

                # Emit yield statement
                yield_span = self.span_tracker.get_span(stmt.value)
                self.builder.emit(ir.YieldStmt(yield_exprs, yield_span))

                # Track variable name for if statement output registration
                if hasattr(self, "_current_yield_vars") and self._current_yield_vars is not None:
                    self._current_yield_vars.append(var_name)

                # Don't register in scope yet - will be done when if statement completes
                return

        # Parse value expression
        if stmt.value is None:
            raise UnsupportedFeatureError(
                "Yield assignment with no value is not supported",
                span=self.span_tracker.get_span(stmt),
                hint="Provide a value for the assignment",
            )
        value_expr = self.parse_expression(stmt.value)

        # Create variable with let
        var = self.builder.let(var_name, value_expr, span=span)

        # Register in scope
        self.scope_manager.define_var(var_name, var, span=span)

    def parse_assignment(self, stmt: ast.Assign) -> None:
        """Parse regular assignment: var = value or tuple unpacking.

        Args:
            stmt: Assign AST node
        """
        # Handle tuple unpacking for yields
        if len(stmt.targets) == 1:
            target = stmt.targets[0]

            # Handle tuple unpacking: (a, b, c) = pl.yield_(...) or self.func(...)
            if isinstance(target, ast.Tuple):
                # Check if value is a pl.yield_() call
                if isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "yield_":
                        # This is handled in yield parsing
                        self.parse_yield_assignment(target, stmt.value)
                        return

                # General tuple unpacking for function calls returning TupleType
                span = self.span_tracker.get_span(stmt)
                value_expr = self.parse_expression(stmt.value)

                # Bind the tuple result to a temporary variable
                tuple_var = self.builder.let("_tuple_tmp", value_expr, span=span)

                # Extract each element using TupleGetItemExpr
                for i, elt in enumerate(target.elts):
                    if not isinstance(elt, ast.Name):
                        raise ParserSyntaxError(
                            f"Tuple unpacking target must be a variable name, got {ast.unparse(elt)}",
                            span=self.span_tracker.get_span(elt),
                            hint="Use simple variable names in tuple unpacking: a, b, c = func()",
                        )
                    item_expr = ir.TupleGetItemExpr(tuple_var, i, span)
                    var = self.builder.let(elt.id, item_expr, span=span)
                    self.scope_manager.define_var(elt.id, var, span=span)
                return

            # Handle simple assignment
            if isinstance(target, ast.Name):
                var_name = target.id
                span = self.span_tracker.get_span(stmt)

                # Check if this is a yield assignment: var = pl.yield_(...)
                if isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "yield_":
                        # Handle yield assignment
                        yield_exprs = []
                        for arg in stmt.value.args:
                            expr = self.parse_expression(arg)
                            if not isinstance(expr, ir.Expr):
                                raise ParserSyntaxError(
                                    f"Yield argument must be an IR expression, got {type(expr)}",
                                    span=self.span_tracker.get_span(arg),
                                    hint="Ensure yield arguments are valid expressions",
                                )
                            yield_exprs.append(expr)

                        # Emit yield statement
                        yield_span = self.span_tracker.get_span(stmt.value)
                        self.builder.emit(ir.YieldStmt(yield_exprs, yield_span))

                        # Track variable name for loop/if output registration
                        if hasattr(self, "_current_yield_vars") and self._current_yield_vars is not None:
                            self._current_yield_vars.append(var_name)

                        # Don't register in scope yet - will be done when loop/if completes
                        return

                value_expr = self.parse_expression(stmt.value)
                var = self.builder.let(var_name, value_expr, span=span)
                self.scope_manager.define_var(var_name, var, span=span)
                return

        raise ParserSyntaxError(
            f"Unsupported assignment: {ast.unparse(stmt)}",
            span=self.span_tracker.get_span(stmt),
            hint="Use simple variable assignments or tuple unpacking with pl.yield_()",
        )

    def parse_yield_assignment(self, target: ast.Tuple, value: ast.Call) -> None:
        """Parse yield assignment: (a, b) = pl.yield_(x, y).

        Args:
            target: Tuple of target variable names
            value: Call to pl.yield_()
        """
        # Parse yield expressions
        yield_exprs = []
        for arg in value.args:
            expr = self.parse_expression(arg)
            # Ensure it's an IR Expr
            if not isinstance(expr, ir.Expr):
                raise ParserSyntaxError(
                    f"Yield argument must be an IR expression, got {type(expr)}",
                    span=self.span_tracker.get_span(arg),
                    hint="Ensure yield arguments are valid expressions",
                )
            yield_exprs.append(expr)

        # Emit yield statement
        span = self.span_tracker.get_span(value)
        self.builder.emit(ir.YieldStmt(yield_exprs, span))

        # Track yielded variable names for if/for statement processing
        if hasattr(self, "_current_yield_vars") and self._current_yield_vars is not None:
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self._current_yield_vars.append(elt.id)

        # For tuple yields at the for/while loop level, register the variables
        # (they'll be available as loop.get_result().return_vars)
        if (self.in_for_loop or self.in_while_loop) and not self.in_if_stmt:
            # Register yielded variable names in scope
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Name):
                    var_name = elt.id
                    # Will be resolved from loop outputs
                    self.scope_manager.define_var(var_name, f"loop_yield_{i}")

    def _validate_for_loop_iterator(self, stmt: ast.For) -> tuple[ast.Call, str]:
        """Validate that for loop uses pl.range(), pl.parallel(), or pl.while_() and return the call node.

        Returns:
            Tuple of (call_node, iterator_type) where iterator_type is "range", "parallel", or "while_"
        """
        if not isinstance(stmt.iter, ast.Call):
            raise ParserSyntaxError(
                "For loop must use pl.range(), pl.parallel(), or pl.while_()",
                span=self.span_tracker.get_span(stmt.iter),
                hint="Use pl.range(), pl.parallel(), or pl.while_() as the iterator",
            )

        iter_call = stmt.iter
        func = iter_call.func
        if isinstance(func, ast.Attribute):
            if func.attr == "range":
                return iter_call, "range"
            if func.attr == "parallel":
                return iter_call, "parallel"
            if func.attr == "while_":
                return iter_call, "while_"
        raise ParserSyntaxError(
            "For loop must use pl.range(), pl.parallel(), or pl.while_()",
            span=self.span_tracker.get_span(stmt.iter),
            hint="Use pl.range(), pl.parallel(), or pl.while_() as the iterator",
        )

    def _parse_for_loop_target(self, stmt: ast.For) -> tuple[str, Optional[ast.AST], bool]:
        """Parse for loop target, returning (loop_var_name, iter_args_node, is_simple_for)."""
        if isinstance(stmt.target, ast.Name):
            return stmt.target.id, None, True

        if isinstance(stmt.target, ast.Tuple) and len(stmt.target.elts) == 2:
            loop_var_node = stmt.target.elts[0]
            iter_args_node = stmt.target.elts[1]

            if not isinstance(loop_var_node, ast.Name):
                raise ParserSyntaxError(
                    "Loop variable must be a simple name",
                    span=self.span_tracker.get_span(loop_var_node),
                    hint="Use a simple variable name for the loop counter",
                )
            return loop_var_node.id, iter_args_node, False

        raise ParserSyntaxError(
            "For loop target must be a simple name or: (loop_var, (iter_args...))",
            span=self.span_tracker.get_span(stmt.target),
            hint="Use: for i in pl.range(n) or for i, (var1,) in pl.range(n, init_values=(...,))",
        )

    def _setup_iter_args(self, loop: Any, iter_args_node: ast.AST, init_values: list) -> None:
        """Set up iter_args and return_vars for Pattern A loops."""
        if not isinstance(iter_args_node, ast.Tuple):
            raise ParserSyntaxError(
                "Iter args must be a tuple",
                span=self.span_tracker.get_span(iter_args_node),
                hint="Wrap iteration variables in parentheses: (var1, var2)",
            )

        if len(iter_args_node.elts) != len(init_values):
            raise ParserSyntaxError(
                f"Mismatch: {len(iter_args_node.elts)} iter_args but {len(init_values)} init_values",
                span=self.span_tracker.get_span(iter_args_node),
                hint=f"Provide exactly {len(init_values)} iteration variable(s) to match init_values",
            )

        for i, iter_arg_node in enumerate(iter_args_node.elts):
            if not isinstance(iter_arg_node, ast.Name):
                raise ParserSyntaxError(
                    "Iter arg must be a simple name",
                    span=self.span_tracker.get_span(iter_arg_node),
                    hint="Use simple variable names for iteration variables",
                )
            iter_arg_var = loop.iter_arg(iter_arg_node.id, init_values[i])
            self.scope_manager.define_var(iter_arg_node.id, iter_arg_var, allow_redef=True)

        for iter_arg_node in iter_args_node.elts:
            loop.return_var(f"{iter_arg_node.id}_out")

    def parse_for_loop(self, stmt: ast.For) -> None:
        """Parse for loop with pl.range(), pl.parallel(), or while-as-for with pl.while_().

        Supports patterns for range/parallel:
          Pattern A (explicit): for i, (vars,) in pl.range(..., init_values=(...,))
          Pattern B (simple):   for i in pl.range(n)

        Supports pattern for while-as-for:
          for (vars,) in pl.while_(init_values=(...,)):
              pl.cond(condition)

        Both patterns also work with pl.parallel() for parallel loops.
        Pattern B produces a ForStmt without iter_args/return_vars/yield.
        The C++ ConvertToSSA pass handles converting to SSA form.
        """
        iter_call, iterator_type = self._validate_for_loop_iterator(stmt)

        # Handle pl.while_() case
        if iterator_type == "while_":
            self._parse_while_as_for(stmt, iter_call)
            return

        # Handle pl.range() or pl.parallel()
        is_parallel = iterator_type == "parallel"
        loop_var_name, iter_args_node, is_simple_for = self._parse_for_loop_target(stmt)
        range_args = self._parse_range_call(iter_call)

        if is_simple_for and range_args["init_values"]:
            raise ParserSyntaxError(
                "For loop target must be a tuple when init_values is provided",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use: for i, (var1,) in pl.range(n, init_values=(val1,)) to include iter_args",
            )

        kind = ir.ForKind.Parallel if is_parallel else ir.ForKind.Sequential
        loop_var = self.builder.var(loop_var_name, ir.ScalarType(DataType.INT64))
        span = self.span_tracker.get_span(stmt)
        loop_output_vars: list[str] = []

        with self.builder.for_loop(
            loop_var,
            range_args["start"],
            range_args["stop"],
            range_args["step"],
            span,
            kind,
        ) as loop:
            self.current_loop_builder = loop
            self.in_for_loop = True
            self.scope_manager.enter_scope("for")
            self.scope_manager.define_var(loop_var_name, loop_var, allow_redef=True)

            if not is_simple_for:
                assert iter_args_node is not None  # Guaranteed by _parse_for_loop_target
                self._setup_iter_args(loop, iter_args_node, range_args["init_values"])

            prev_yield_tracker = getattr(self, "_current_yield_vars", None)
            self._current_yield_vars = []

            for body_stmt in stmt.body:
                self.parse_statement(body_stmt)

            loop_output_vars = self._current_yield_vars[:]
            self._current_yield_vars = prev_yield_tracker

            should_leak = is_simple_for and not loop_output_vars
            self.scope_manager.exit_scope(leak_vars=should_leak)
            self.in_for_loop = False
            self.current_loop_builder = None

        if not is_simple_for:
            loop_result = loop.get_result()
            if hasattr(loop_result, "return_vars") and loop_result.return_vars and loop_output_vars:
                for i, var_name in enumerate(loop_output_vars):
                    if i < len(loop_result.return_vars):
                        self.scope_manager.define_var(var_name, loop_result.return_vars[i])

    def _parse_range_call(self, call: ast.Call) -> dict[str, Any]:
        """Parse pl.range() call arguments.

        Args:
            call: AST Call node for pl.range()

        Returns:
            Dictionary with start, stop, step, init_values
        """
        # Parse positional arguments
        if len(call.args) < 1:
            raise ParserSyntaxError(
                "pl.range() requires at least 1 argument (stop)",
                span=self.span_tracker.get_span(call),
                hint="Provide at least the stop value: pl.range(10) or pl.range(0, 10)",
            )

        # Default values
        start = 0
        step = 1

        if len(call.args) == 1:
            # range(stop)
            stop = self.parse_expression(call.args[0])
        elif len(call.args) == 2:
            # range(start, stop)
            start = self.parse_expression(call.args[0])
            stop = self.parse_expression(call.args[1])
        elif len(call.args) >= 3:
            # range(start, stop, step)
            start = self.parse_expression(call.args[0])
            stop = self.parse_expression(call.args[1])
            step = self.parse_expression(call.args[2])

        # Parse keyword arguments
        init_values = []
        for keyword in call.keywords:
            if keyword.arg == "init_values":
                # Parse list of init values
                if isinstance(keyword.value, (ast.List, ast.Tuple)):
                    for elt in keyword.value.elts:
                        init_values.append(self.parse_expression(elt))
                else:
                    raise ParserSyntaxError(
                        "init_values must be a list or tuple",
                        span=self.span_tracker.get_span(keyword.value),
                        hint="Use a tuple for init_values: init_values=(var1, var2)",
                    )

        return {"start": start, "stop": stop, "step": step, "init_values": init_values}

    def _is_cond_call(self, stmt: ast.stmt) -> bool:
        """Check if statement is a pl.cond() call (without parsing).

        Args:
            stmt: AST statement node

        Returns:
            True if statement is pl.cond() call, False otherwise
        """
        if not isinstance(stmt, ast.Expr):
            return False

        call = stmt.value
        if not isinstance(call, ast.Call):
            return False

        # Check if this is pl.cond() or cond()
        if isinstance(call.func, ast.Attribute):
            # pl.cond() form
            return call.func.attr == "cond"
        elif isinstance(call.func, ast.Name):
            # cond() form (if imported directly)
            return call.func.id == "cond"

        return False

    def _extract_cond_call(self, stmt: ast.stmt) -> Optional[ir.Expr]:
        """Extract condition from pl.cond() call statement.

        Args:
            stmt: AST statement node

        Returns:
            Parsed condition expression if statement is pl.cond(), None otherwise
        """
        if not self._is_cond_call(stmt):
            return None

        call = stmt.value  # type: ignore[union-attr]

        # Parse the condition argument
        if len(call.args) != 1:  # type: ignore[attr-defined]
            raise ParserSyntaxError(
                "pl.cond() requires exactly 1 argument",
                span=self.span_tracker.get_span(call),
                hint="Use: pl.cond(condition)",
            )

        return self.parse_expression(call.args[0])  # type: ignore[attr-defined]

    def _validate_while_call_args(self, while_call: ast.Call) -> None:
        """Validate that pl.while_() has no positional arguments."""
        if len(while_call.args) > 0:
            raise ParserSyntaxError(
                "pl.while_() takes no positional arguments",
                span=self.span_tracker.get_span(while_call),
                hint="Use: pl.while_(init_values=(...,)) with pl.cond(condition) as first statement in body",
            )

    def _parse_while_init_values(self, while_call: ast.Call) -> list[ir.Expr]:
        """Parse init_values from pl.while_() keyword arguments."""
        init_values = []
        for keyword in while_call.keywords:
            if keyword.arg == "init_values":
                if isinstance(keyword.value, (ast.List, ast.Tuple)):
                    for elt in keyword.value.elts:
                        init_values.append(self.parse_expression(elt))
                else:
                    raise ParserSyntaxError(
                        "init_values must be a tuple or list",
                        span=self.span_tracker.get_span(keyword.value),
                        hint="Use a tuple for init_values (lists also accepted): init_values=(var1, var2)",
                    )

        if not init_values:
            raise ParserSyntaxError(
                "pl.while_() requires init_values",
                span=self.span_tracker.get_span(while_call),
                hint="Provide init_values: pl.while_(init_values=(val1, val2))",
            )

        return init_values

    def _validate_while_body(self, stmt: ast.For) -> None:
        """Validate pl.while_() body structure."""
        if not stmt.body:
            raise ParserSyntaxError(
                "pl.while_() body cannot be empty",
                span=self.span_tracker.get_span(stmt),
                hint="Add pl.cond(condition) as first statement",
            )

        if not self._is_cond_call(stmt.body[0]):
            raise ParserSyntaxError(
                "First statement in pl.while_() body must be pl.cond(condition)",
                span=self.span_tracker.get_span(stmt.body[0]),
                hint="Add pl.cond(condition) as first statement",
            )

    def _validate_while_target(self, stmt: ast.For, init_values: list[ir.Expr]) -> ast.Tuple:
        """Validate and return pl.while_() target tuple."""
        if not isinstance(stmt.target, ast.Tuple):
            raise ParserSyntaxError(
                "While loop target must be a tuple for pl.while_()",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use: for (var1, var2) in pl.while_(init_values=(...,))",
            )

        iter_args_node = stmt.target

        if len(iter_args_node.elts) != len(init_values):
            raise ParserSyntaxError(
                f"Mismatch: {len(iter_args_node.elts)} iter_args but {len(init_values)} init_values",
                span=self.span_tracker.get_span(iter_args_node),
                hint=f"Provide exactly {len(init_values)} iteration variable(s) to match init_values",
            )

        return iter_args_node

    def _setup_while_iter_args(
        self, loop: Any, iter_args_node: ast.Tuple, init_values: list[ir.Expr]
    ) -> None:
        """Set up iter_args for pl.while_() loop."""
        for i, iter_arg_node in enumerate(iter_args_node.elts):
            if not isinstance(iter_arg_node, ast.Name):
                raise ParserSyntaxError(
                    "Iter arg must be a simple name",
                    span=self.span_tracker.get_span(iter_arg_node),
                    hint="Use simple variable names for iteration variables",
                )
            iter_arg_var = loop.iter_arg(iter_arg_node.id, init_values[i])
            self.scope_manager.define_var(iter_arg_node.id, iter_arg_var, allow_redef=True)

    def _parse_while_body_statements(self, stmt: ast.For) -> list[str]:
        """Parse body statements for pl.while_() loop, return yielded vars."""
        prev_yield_tracker = getattr(self, "_current_yield_vars", None)
        self._current_yield_vars = []

        # Parse body (skip first statement which is pl.cond())
        for i, body_stmt in enumerate(stmt.body):
            if i == 0:
                continue  # Skip the pl.cond() statement

            # Check if pl.cond() appears anywhere else in body
            if self._is_cond_call(body_stmt):
                raise ParserSyntaxError(
                    "pl.cond() can only be the first statement in a pl.while_() loop body",
                    span=self.span_tracker.get_span(body_stmt),
                    hint="Remove this pl.cond() - condition is already specified at the start",
                )

            self.parse_statement(body_stmt)

        loop_output_vars = self._current_yield_vars[:]
        self._current_yield_vars = prev_yield_tracker
        return loop_output_vars

    def _register_while_outputs(self, loop: Any, loop_output_vars: list[str]) -> None:
        """Register output variables from pl.while_() loop."""
        loop_result = loop.get_result()
        if hasattr(loop_result, "return_vars") and loop_result.return_vars and loop_output_vars:
            for i, var_name in enumerate(loop_output_vars):
                if i < len(loop_result.return_vars):
                    self.scope_manager.define_var(var_name, loop_result.return_vars[i])

    def _parse_while_as_for(self, stmt: ast.For, while_call: ast.Call) -> None:
        """Parse while loop using for...in pl.while_() pattern.

        Pattern: for (var1, var2) in pl.while_(init_values=(val1, val2)):
                     pl.cond(condition)
                     ...

        Args:
            stmt: For AST node
            while_call: Call to pl.while_()
        """
        # Validate and parse arguments
        self._validate_while_call_args(while_call)
        init_values = self._parse_while_init_values(while_call)
        self._validate_while_body(stmt)
        iter_args_node = self._validate_while_target(stmt, init_values)

        span = self.span_tracker.get_span(stmt)
        placeholder_condition = ir.ConstBool(True, span)

        with self.builder.while_loop(placeholder_condition, span) as loop:
            self.current_loop_builder = loop
            self.in_while_loop = True
            self.scope_manager.enter_scope("while")

            # Set up iter_args
            self._setup_while_iter_args(loop, iter_args_node, init_values)

            # Parse and set the condition (now that iter_args are in scope)
            condition = self._extract_cond_call(stmt.body[0])
            if condition is None:
                raise ParserSyntaxError(
                    "First statement in pl.while_() body must be pl.cond(condition)",
                    span=self.span_tracker.get_span(stmt.body[0]),
                    hint="Add pl.cond(condition) as first statement",
                )
            loop.set_condition(condition)

            # Add return_vars
            for iter_arg_node in iter_args_node.elts:
                loop.return_var(f"{iter_arg_node.id}_out")

            # Parse body statements
            loop_output_vars = self._parse_while_body_statements(stmt)

            self.scope_manager.exit_scope(leak_vars=False)
            self.in_while_loop = False
            self.current_loop_builder = None

        # Register output variables
        self._register_while_outputs(loop, loop_output_vars)

    def parse_while_loop(self, stmt: ast.While) -> None:
        """Parse natural while loop syntax.

        Natural while syntax: while condition: body

        This creates a WhileStmt without iter_args (non-SSA form).
        The C++ ConvertToSSA pass will convert it to SSA form if needed.

        Args:
            stmt: While AST node
        """
        # Parse natural while syntax: while condition:
        condition = self.parse_expression(stmt.test)
        span = self.span_tracker.get_span(stmt)

        with self.builder.while_loop(condition, span) as loop:
            self.current_loop_builder = loop
            self.in_while_loop = True
            self.scope_manager.enter_scope("while")

            # Parse body statements
            for body_stmt in stmt.body:
                self.parse_statement(body_stmt)

            # Variables leak to outer scope (ConvertToSSA will handle)
            self.scope_manager.exit_scope(leak_vars=True)
            self.in_while_loop = False
            self.current_loop_builder = None

    def parse_if_statement(self, stmt: ast.If) -> None:
        """Parse if statement with phi nodes.

        When pl.yield_() is used, phi nodes are created via return_vars.
        When no yields are used (plain syntax), variables leak to outer scope
        and the C++ ConvertToSSA pass handles creating phi nodes.

        Args:
            stmt: If AST node
        """
        # Parse condition
        condition = self.parse_expression(stmt.test)
        span = self.span_tracker.get_span(stmt)

        # Track yield output variable names from both branches
        then_yield_vars = []

        # Begin if statement
        with self.builder.if_stmt(condition, span) as if_builder:
            self.current_if_builder = if_builder
            self.in_if_stmt = True

            # Parse then branch to collect yield variable names first
            # We need to know what variables will be yielded to declare return_vars
            prev_yield_tracker = getattr(self, "_current_yield_vars", None)
            self._current_yield_vars = []

            # Scan then branch for yields (without executing)
            then_yield_vars = self._scan_for_yields(stmt.body)

            # Declare return vars based on yields
            for var_name in then_yield_vars:
                # Get type from annotation if available
                # For now, use a generic tensor type - ideally we'd infer from yield expr
                if_builder.return_var(var_name, ir.TensorType([1], DataType.INT32))

            # Determine if we should leak variables (no explicit yields)
            should_leak = not bool(then_yield_vars)

            # Now parse then branch
            self.scope_manager.enter_scope("if")
            for then_stmt in stmt.body:
                self.parse_statement(then_stmt)
            self.scope_manager.exit_scope(leak_vars=should_leak)

            # Parse else branch if present
            if stmt.orelse:
                if_builder.else_()
                self.scope_manager.enter_scope("else")
                for else_stmt in stmt.orelse:
                    self.parse_statement(else_stmt)
                self.scope_manager.exit_scope(leak_vars=should_leak)

            # Restore previous yield tracker
            self._current_yield_vars = prev_yield_tracker

        # After if statement completes, register the output variables in the outer scope
        if then_yield_vars:
            # Get the output variables from the if statement
            if_result = if_builder.get_result()
            if hasattr(if_result, "return_vars") and if_result.return_vars:
                # Register each output variable with its name
                for i, var_name in enumerate(then_yield_vars):
                    if i < len(if_result.return_vars):
                        output_var = if_result.return_vars[i]
                        self.scope_manager.define_var(var_name, output_var)

        self.in_if_stmt = False
        self.current_if_builder = None

    def parse_return(self, stmt: ast.Return) -> None:
        """Parse return statement.

        Args:
            stmt: Return AST node
        """
        span = self.span_tracker.get_span(stmt)

        if stmt.value is None:
            self.builder.return_stmt(None, span)
            return

        # Handle tuple return
        if isinstance(stmt.value, ast.Tuple):
            return_exprs = []
            for elt in stmt.value.elts:
                return_exprs.append(self.parse_expression(elt))
            self.builder.return_stmt(return_exprs, span)
        else:
            # Single return value
            return_expr = self.parse_expression(stmt.value)
            self.builder.return_stmt([return_expr], span)

    def parse_evaluation_statement(self, stmt: ast.Expr) -> None:
        """Parse evaluation statement (EvalStmt).

        Evaluation statements represent operations executed for their side effects,
        with the return value discarded (e.g., synchronization barriers).

        Args:
            stmt: Expr AST node
        """
        expr = self.parse_expression(stmt.value)
        span = self.span_tracker.get_span(stmt)

        # Validate that we got an IR expression (not a list literal, etc.)
        if not isinstance(expr, ir.Expr):
            raise ParserSyntaxError(
                f"Evaluation statement must be an IR expression, got {type(expr).__name__}",
                span=span,
                hint="Only function calls and operations can be used as standalone statements",
            )

        # Emit EvalStmt using builder method
        self.builder.eval_stmt(expr, span)

    def parse_expression(self, expr: ast.expr) -> ir.Expr:
        """Parse expression and return IR Expr.

        Args:
            expr: AST expression node

        Returns:
            IR expression or Python value for list literals
        """
        if isinstance(expr, ast.Name):
            return self.parse_name(expr)
        elif isinstance(expr, ast.Constant):
            return self.parse_constant(expr)
        elif isinstance(expr, ast.BinOp):
            return self.parse_binop(expr)
        elif isinstance(expr, ast.Compare):
            return self.parse_compare(expr)
        elif isinstance(expr, ast.Call):
            return self.parse_call(expr)
        elif isinstance(expr, ast.Attribute):
            return self.parse_attribute(expr)
        elif isinstance(expr, ast.UnaryOp):
            return self.parse_unaryop(expr)
        elif isinstance(expr, ast.List):
            return self.parse_list(expr)
        elif isinstance(expr, ast.Tuple):
            return self.parse_tuple_literal(expr)
        elif isinstance(expr, ast.Subscript):
            return self.parse_subscript(expr)
        else:
            raise UnsupportedFeatureError(
                f"Unsupported expression type: {type(expr).__name__}",
                span=self.span_tracker.get_span(expr),
                hint="Use supported expressions like variables, constants, operations, or function calls",
            )

    def parse_name(self, name: ast.Name) -> ir.Var:
        """Parse variable name reference.

        Args:
            name: Name AST node

        Returns:
            IR Var
        """
        var_name = name.id
        var = self.scope_manager.lookup_var(var_name)

        if var is None:
            raise UndefinedVariableError(
                f"Undefined variable '{var_name}'",
                span=self.span_tracker.get_span(name),
                hint="Check if the variable is defined before using it",
            )

        # Return the IR Var
        return var

    def parse_constant(self, const: ast.Constant) -> ir.Expr:
        """Parse constant value.

        Args:
            const: Constant AST node

        Returns:
            IR constant expression
        """
        span = self.span_tracker.get_span(const)
        value = const.value

        if isinstance(value, int):
            return ir.ConstInt(value, DataType.INT64, span)
        elif isinstance(value, float):
            return ir.ConstFloat(value, DataType.FP32, span)
        elif isinstance(value, bool):
            return ir.ConstBool(value, span)
        else:
            raise ParserTypeError(
                f"Unsupported constant type: {type(value)}",
                span=self.span_tracker.get_span(const),
                hint="Use int, float, or bool constants",
            )

    def parse_binop(self, binop: ast.BinOp) -> ir.Expr:
        """Parse binary operation.

        Args:
            binop: BinOp AST node

        Returns:
            IR binary expression
        """
        span = self.span_tracker.get_span(binop)
        left = self.parse_expression(binop.left)
        right = self.parse_expression(binop.right)

        # Map operator to IR function
        op_map = {
            ast.Add: lambda lhs, rhs, span: ir.add(lhs, rhs, span),
            ast.Sub: lambda lhs, rhs, span: ir.sub(lhs, rhs, span),
            ast.Mult: lambda lhs, rhs, span: ir.mul(lhs, rhs, span),
            ast.Div: lambda lhs, rhs, span: ir.truediv(lhs, rhs, span),
            ast.FloorDiv: lambda lhs, rhs, span: ir.floordiv(lhs, rhs, span),
            ast.Mod: lambda lhs, rhs, span: ir.mod(lhs, rhs, span),
        }

        op_type = type(binop.op)
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported binary operator: {op_type.__name__}",
                span=self.span_tracker.get_span(binop),
                hint="Use supported operators: +, -, *, /, //, %",
            )

        return op_map[op_type](left, right, span)

    def parse_compare(self, compare: ast.Compare) -> ir.Expr:
        """Parse comparison operation.

        Args:
            compare: Compare AST node

        Returns:
            IR comparison expression
        """
        if len(compare.ops) != 1 or len(compare.comparators) != 1:
            raise ParserSyntaxError(
                "Only simple comparisons supported",
                span=self.span_tracker.get_span(compare),
                hint="Use single comparison operators like: a < b, not chained comparisons",
            )

        span = self.span_tracker.get_span(compare)
        left = self.parse_expression(compare.left)
        right = self.parse_expression(compare.comparators[0])

        # Map comparison to IR function
        op_map = {
            ast.Eq: lambda lhs, rhs, span: ir.eq(lhs, rhs, span),
            ast.NotEq: lambda lhs, rhs, span: ir.ne(lhs, rhs, span),
            ast.Lt: lambda lhs, rhs, span: ir.lt(lhs, rhs, span),
            ast.LtE: lambda lhs, rhs, span: ir.le(lhs, rhs, span),
            ast.Gt: lambda lhs, rhs, span: ir.gt(lhs, rhs, span),
            ast.GtE: lambda lhs, rhs, span: ir.ge(lhs, rhs, span),
        }

        op_type = type(compare.ops[0])
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported comparison: {op_type.__name__}",
                span=self.span_tracker.get_span(compare),
                hint="Use supported comparisons: ==, !=, <, <=, >, >=",
            )

        return op_map[op_type](left, right, span)

    def parse_unaryop(self, unary: ast.UnaryOp) -> ir.Expr:
        """Parse unary operation.

        Args:
            unary: UnaryOp AST node

        Returns:
            IR unary expression
        """
        span = self.span_tracker.get_span(unary)
        operand = self.parse_expression(unary.operand)

        op_map = {
            ast.USub: lambda o, s: ir.neg(o, s),
            ast.Not: lambda o, s: ir.bit_not(o, s),
        }

        op_type = type(unary.op)
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported unary operator: {op_type.__name__}",
                span=self.span_tracker.get_span(unary),
                hint="Use supported unary operators: -, not",
            )

        return op_map[op_type](operand, span)

    def parse_call(self, call: ast.Call) -> ir.Expr:
        """Parse function call.

        Args:
            call: Call AST node

        Returns:
            IR expression from call
        """
        func = call.func

        # Handle pl.yield_() specially
        if isinstance(func, ast.Attribute) and func.attr == "yield_":
            return self.parse_yield_call(call)

        # Handle cross-function calls via self.method_name() in @pl.program classes
        if isinstance(func, ast.Attribute):
            # Check for self.method_name pattern
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                method_name = func.attr
                if method_name in self.global_vars:
                    # This is a cross-function call, use GlobalVar
                    gvar = self.global_vars[method_name]
                    args = [self.parse_expression(arg) for arg in call.args]
                    span = self.span_tracker.get_span(call)

                    # Determine the return type for the call
                    # If we have the function parsed, use its return type
                    # Otherwise, type will be inferred later by the Program
                    return_type = None
                    if gvar in self.gvar_to_func:
                        func_obj = self.gvar_to_func[gvar]
                        if func_obj.return_types:
                            if len(func_obj.return_types) == 1:
                                return_type = func_obj.return_types[0]
                            else:
                                return_type = ir.TupleType(func_obj.return_types)

                    # Create Call with the determined return type (or None if not yet known)
                    # Call constructor: Call(op, args, type, span) or Call(op, args, span)
                    if return_type is not None:
                        call_expr = ir.Call(gvar, args, return_type, span)
                    else:
                        call_expr = ir.Call(gvar, args, span)

                    return call_expr
                else:
                    raise UndefinedVariableError(
                        f"Function '{method_name}' not defined in program",
                        span=self.span_tracker.get_span(call),
                        hint=f"Available functions: {list(self.global_vars.keys())}",
                    )

            # Handle pl.tensor.*, pl.block.*, and pl.* operation calls
            return self.parse_op_call(call)

        raise UnsupportedFeatureError(
            f"Unsupported function call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint="Use pl.* operations, pl.yield_(), or self.method() for cross-function calls",
        )

    def parse_yield_call(self, call: ast.Call) -> ir.Expr:
        """Parse pl.yield_() call.

        Args:
            call: Call to pl.yield_() or pl.yield_()

        Returns:
            IR expression (first yielded value for single yield)
        """
        span = self.span_tracker.get_span(call)
        yield_exprs = []

        for arg in call.args:
            expr = self.parse_expression(arg)
            yield_exprs.append(expr)

        # Emit yield statement
        self.builder.emit(ir.YieldStmt(yield_exprs, span))

        # Track yielded variables for if statement processing
        # This is for single assignment like: var = pl.yield_(expr)
        # We'll return a placeholder that gets resolved when if statement completes

        # Return first expression as the "value" of the yield
        # This handles: var = pl.yield_(expr)
        if len(yield_exprs) == 1:
            return yield_exprs[0]

        # For multiple yields, this should be handled as tuple assignment
        raise ParserSyntaxError(
            "Multiple yields should use tuple unpacking assignment",
            span=self.span_tracker.get_span(call),
            hint="Use tuple unpacking: (a, b) = pl.yield_(x, y)",
        )

    def parse_op_call(self, call: ast.Call) -> ir.Expr:
        """Parse operation call like pl.tensor.create_tensor() or pl.add().

        Args:
            call: Call AST node

        Returns:
            IR expression from operation
        """
        func = call.func

        # Navigate through attribute chain to find operation
        # e.g., pl.tensor.create_tensor -> ["pl", "tensor", "create_tensor"]
        # e.g., pl.add -> ["pl", "add"]
        attrs = []
        node = func
        while isinstance(node, ast.Attribute):
            attrs.insert(0, node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            attrs.insert(0, node.id)

        # pl.tensor.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] == "pl" and attrs[1] == "tensor":
            op_name = attrs[2]
            return self._parse_tensor_op(op_name, call)

        # pl.block.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] == "pl" and attrs[1] == "block":
            op_name = attrs[2]
            return self._parse_block_op(op_name, call)

        # pl.{operation} (2-segment, unified dispatch or promoted ops)
        if len(attrs) >= 2 and attrs[0] == "pl" and attrs[1] not in ("tensor", "block"):
            op_name = attrs[1]
            return self._parse_unified_op(op_name, call)

        raise UnsupportedFeatureError(
            f"Unsupported operation call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint="Use pl.*, pl.tensor.*, or pl.block.* operations",
        )

    def _parse_op_kwargs(self, call: ast.Call) -> dict[str, Any]:
        """Parse keyword arguments for an operation call.

        Shared helper for tensor, block, and unified op parsing.

        Args:
            call: Call AST node

        Returns:
            Dictionary of keyword argument names to values
        """
        kwargs = {}
        for keyword in call.keywords:
            key = keyword.arg
            value = keyword.value

            # Handle dtype specially
            if key == "dtype":
                kwargs[key] = self.type_resolver.resolve_dtype(value)
            elif isinstance(value, ast.Constant):
                kwargs[key] = value.value
            elif isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
                # Handle negative numbers like -1
                if isinstance(value.operand, ast.Constant):
                    kwargs[key] = -value.operand.value
                else:
                    kwargs[key] = self.parse_expression(value)
            elif isinstance(value, ast.Name):
                if value.id in ["True", "False"]:
                    kwargs[key] = value.id == "True"
                else:
                    kwargs[key] = self.parse_expression(value)
            elif isinstance(value, ast.Attribute):
                # Handle DataType.FP16 etc
                kwargs[key] = self.type_resolver.resolve_dtype(value)
            elif isinstance(value, ast.List):
                kwargs[key] = self.parse_list(value)
            else:
                kwargs[key] = self.parse_expression(value)
        return kwargs

    def _parse_tensor_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse tensor operation.

        Args:
            op_name: Name of tensor operation
            call: Call AST node

        Returns:
            IR expression from tensor operation
        """
        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)

        # Map language-level operation name to IR-level name if needed
        ir_op_name = self._TENSOR_OP_NAME_MAP.get(op_name, op_name)

        # Call the appropriate tensor operation
        if hasattr(ir_op.tensor, ir_op_name):
            op_func = getattr(ir_op.tensor, ir_op_name)
            call_span = self.span_tracker.get_span(call)
            return op_func(*args, **kwargs, span=call_span)

        raise InvalidOperationError(
            f"Unknown tensor operation: {op_name}",
            span=self.span_tracker.get_span(call),
            hint=f"Check if '{op_name}' is a valid tensor operation",
        )

    def _parse_block_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse block operation.

        Args:
            op_name: Name of block operation
            call: Call AST node

        Returns:
            IR expression from block operation
        """
        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)

        # Call the appropriate block operation
        if hasattr(ir_op.block, op_name):
            op_func = getattr(ir_op.block, op_name)
            call_span = self.span_tracker.get_span(call)
            return op_func(*args, **kwargs, span=call_span)

        raise InvalidOperationError(
            f"Unknown block operation: {op_name}",
            span=self.span_tracker.get_span(call),
            hint=f"Check if '{op_name}' is a valid block operation",
        )

    # Maps unified op names to the scalar variant for block ops.
    # Only binary arithmetic ops have scalar auto-dispatch.
    _BLOCK_SCALAR_OPS: dict[str, str] = {
        "add": "adds",
        "sub": "subs",
        "mul": "muls",
        "div": "divs",
    }

    # Maps language-level tensor operation names to IR-level names.
    _TENSOR_OP_NAME_MAP: dict[str, str] = {
        "create_tensor": "create",
    }

    # Ops that exist only in one module (no dispatch needed).
    _TENSOR_ONLY_OPS = {
        "create_tensor",
        "dim",
        "assemble",
        "add_scalar",
        "sub_scalar",
        "mul_scalar",
        "div_scalar",
    }
    _BLOCK_ONLY_OPS = {
        "load",
        "store",
        "l0c_store",
        "move",
        "neg",
        "sqrt",
        "rsqrt",
        "recip",
        "log",
        "abs",
        "relu",
        "matmul_acc",
        "minimum",
        "cmp",
        "cmps",
        "adds",
        "subs",
        "muls",
        "divs",
        "sum",
        "max",
        "min",
        "row_min",
        "row_expand_add",
        "row_expand_sub",
        "row_expand_mul",
        "row_expand_div",
        "col_expand",
        "col_expand_mul",
        "col_expand_div",
        "col_expand_sub",
        "expands",
        "create_tile",
    }

    def _parse_unified_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse unified operation call (pl.{op_name}).

        Dispatches to tensor or block IR op based on the first argument's type.

        Args:
            op_name: Name of the operation
            call: Call AST node

        Returns:
            IR expression from the dispatched operation
        """
        # Short-circuit for ops that only exist in one module
        if op_name in self._TENSOR_ONLY_OPS:
            return self._parse_tensor_op(op_name, call)
        if op_name in self._BLOCK_ONLY_OPS:
            return self._parse_block_op(op_name, call)

        call_span = self.span_tracker.get_span(call)

        if not call.args:
            raise InvalidOperationError(
                f"Unified operation '{op_name}' requires at least one argument for type dispatch",
                span=call_span,
                hint="Provide a Tensor or Tile as the first argument",
            )

        # Parse only the first arg to determine dispatch target
        first_arg = self.parse_expression(call.args[0])
        first_type = first_arg.type

        if isinstance(first_type, ir.TensorType):
            return self._parse_tensor_op(op_name, call)

        if isinstance(first_type, ir.TileType):
            # For binary arithmetic ops, check if rhs is scalar → use scalar variant
            scalar_op = self._BLOCK_SCALAR_OPS.get(op_name)
            if scalar_op and len(call.args) >= 2:
                rhs_arg = self.parse_expression(call.args[1])
                if isinstance(rhs_arg.type, ir.ScalarType):
                    return self._parse_block_op(scalar_op, call)

            return self._parse_block_op(op_name, call)

        raise InvalidOperationError(
            f"Cannot dispatch '{op_name}': first argument has type {first_type.TypeName()}, "
            f"expected TensorType or TileType",
            span=call_span,
            hint="Use pl.tensor.* or pl.block.* for explicit dispatch",
        )

    def parse_attribute(self, attr: ast.Attribute) -> ir.Expr:
        """Parse attribute access.

        Args:
            attr: Attribute AST node

        Returns:
            IR expression
        """
        # This might be accessing a DataType enum or similar
        # For now, this is primarily used in calls, not standalone
        raise UnsupportedFeatureError(
            f"Standalone attribute access not supported: {ast.unparse(attr)}",
            span=self.span_tracker.get_span(attr),
            hint="Attribute access is only supported within function calls",
        )

    def parse_list(self, list_node: ast.List) -> list[Any]:
        """Parse list literal.

        Args:
            list_node: List AST node

        Returns:
            Python list of parsed elements (not IR Expr)
        """
        # For list literals like [64, 128], return a Python list
        # These are used as arguments to operations
        result = []
        for elt in list_node.elts:
            if isinstance(elt, ast.Constant):
                result.append(elt.value)
            else:
                # Try to parse as expression
                parsed = self.parse_expression(elt)
                result.append(parsed)
        return result

    def parse_tuple_literal(self, tuple_node: ast.Tuple) -> ir.MakeTuple:
        """Parse tuple literal like (x, y, z).

        Args:
            tuple_node: Tuple AST node

        Returns:
            MakeTuple IR expression

        Example Python syntax:
            result = (x, y)         # Creates MakeTuple([x, y])
            singleton = (x,)        # Creates MakeTuple([x])
            empty = ()              # Creates MakeTuple([])
        """
        span = self.span_tracker.get_span(tuple_node)

        # Parse all elements
        elements = []
        for elt in tuple_node.elts:
            elements.append(self.parse_expression(elt))

        return ir.MakeTuple(elements, span)

    def parse_subscript(self, subscript: ast.Subscript) -> ir.Expr:
        """Parse subscript expression like tuple[0].

        Args:
            subscript: Subscript AST node

        Returns:
            IR expression (TupleGetItemExpr for tuple access)

        Example Python syntax:
            first = my_tuple[0]      # Creates TupleGetItemExpr(my_tuple, 0)
            nested = my_tuple[1][2]  # Creates nested TupleGetItemExpr
        """
        span = self.span_tracker.get_span(subscript)
        value_expr = self.parse_expression(subscript.value)

        # Parse index from slice
        if isinstance(subscript.slice, ast.Constant):
            index = subscript.slice.value
            if not isinstance(index, int):
                raise ParserSyntaxError(
                    "Tuple index must be an integer",
                    span=span,
                    hint="Use integer index like tuple[0]",
                )
        else:
            raise UnsupportedFeatureError(
                "Only constant integer indices supported for tuple access",
                span=span,
                hint="Use a constant integer index like tuple[0]",
            )

        # Check if value is tuple type (runtime check)
        value_type = value_expr.type
        if not isinstance(value_type, ir.TupleType):
            raise ParserTypeError(
                f"Subscript requires tuple type, got {value_type.TypeName()}",
                span=span,
                hint="Only tuple types support subscript access in this context",
            )

        # Create TupleGetItemExpr
        return ir.TupleGetItemExpr(value_expr, index, span)

    def _scan_for_yields(self, stmts: list[ast.stmt]) -> list[str]:
        """Scan statements for yield assignments to determine output variable names.

        Args:
            stmts: List of statements to scan

        Returns:
            List of variable names that are yielded
        """
        yield_vars = []

        for stmt in stmts:
            # Check for annotated assignment with yield_: var: type = pl.yield_(...)
            if isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name) and isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "yield_":
                        yield_vars.append(stmt.target.id)

            # Check for regular assignment with yield_: var = pl.yield_(...)
            elif isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    # Single variable assignment
                    if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if isinstance(func, ast.Attribute) and func.attr == "yield_":
                            yield_vars.append(target.id)
                    # Tuple unpacking: (a, b) = pl.yield_(...)
                    elif isinstance(target, ast.Tuple) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if isinstance(func, ast.Attribute) and func.attr == "yield_":
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    yield_vars.append(elt.id)

            # Recursively scan nested if statements
            elif isinstance(stmt, ast.If):
                yield_vars.extend(self._scan_for_yields(stmt.body))
                if stmt.orelse:
                    # Only take yields from else if they match then branch
                    # For simplicity, just take from then branch
                    pass

        return yield_vars


__all__ = ["ASTParser"]
