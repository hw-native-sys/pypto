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
from typing import Any

from pypto.ir import IRBuilder
from pypto.ir import op as ir_op
from pypto.pypto_core import DataType, ir

from .scope_manager import ScopeManager
from .span_tracker import SpanTracker
from .type_resolver import TypeResolver

# TODO(syfeng): Enhance type checking and fix all type issues.
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportReturnType=false
# pyright: reportOptionalOperand=false, reportOperatorIssue=false


class ASTParser:
    """Parses Python AST and builds IR using IRBuilder."""

    def __init__(self, source_file: str, source_lines: list[str]):
        """Initialize AST parser.

        Args:
            source_file: Path to source file
            source_lines: Lines of source code
        """
        self.span_tracker = SpanTracker(source_file, source_lines)
        self.scope_manager = ScopeManager()
        self.type_resolver = TypeResolver()
        self.builder = IRBuilder()

        # Track context for handling yields and returns
        self.in_for_loop = False
        self.in_if_stmt = False
        self.current_if_builder = None
        self.current_loop_builder = None

    def parse_function(self, func_def: ast.FunctionDef) -> ir.Function:
        """Parse function definition and build IR.

        Args:
            func_def: AST FunctionDef node

        Returns:
            IR Function object
        """
        func_name = func_def.name
        func_span = self.span_tracker.get_span(func_def)

        # Enter function scope
        self.scope_manager.enter_scope("function")

        # Begin building function
        with self.builder.function(func_name, func_span) as f:
            # Parse parameters
            for arg in func_def.args.args:
                param_name = arg.arg
                if arg.annotation is None:
                    raise ValueError(f"Parameter '{param_name}' missing type annotation")

                param_type = self.type_resolver.resolve_type(arg.annotation)
                param_span = self.span_tracker.get_span(arg)

                # Add parameter to function
                param_var = f.param(param_name, param_type, param_span)

                # Register in scope
                self.scope_manager.define_var(param_name, param_var, allow_redef=True)

            # Parse return type
            if func_def.returns:
                return_type = self.type_resolver.resolve_type(func_def.returns)
                f.return_type(return_type)

            # Parse function body
            for stmt in func_def.body:
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
        elif isinstance(stmt, ast.If):
            self.parse_if_statement(stmt)
        elif isinstance(stmt, ast.Return):
            self.parse_return(stmt)
        elif isinstance(stmt, ast.Expr):
            # Expression statement (e.g., standalone function call)
            self.parse_expression(stmt.value)
        else:
            raise ValueError(f"Unsupported statement type: {type(stmt).__name__}")

    def parse_annotated_assignment(self, stmt: ast.AnnAssign) -> None:
        """Parse annotated assignment: var: type = value.

        Args:
            stmt: AnnAssign AST node
        """
        if not isinstance(stmt.target, ast.Name):
            raise ValueError("Only simple variable assignments supported")

        var_name = stmt.target.id
        span = self.span_tracker.get_span(stmt)

        # Check if this is a yield assignment: var: type = pl.yeild(...)
        if isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Attribute) and func.attr == "yeild":
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
            raise NotImplementedError("Yield assignment with no value is not supported")
        value_expr = self.parse_expression(stmt.value)

        # Create variable with let
        var = self.builder.let(var_name, value_expr, span=span)

        # Register in scope
        self.scope_manager.define_var(var_name, var)

    def parse_assignment(self, stmt: ast.Assign) -> None:
        """Parse regular assignment: var = value or tuple unpacking.

        Args:
            stmt: Assign AST node
        """
        # Handle tuple unpacking for yields
        if len(stmt.targets) == 1:
            target = stmt.targets[0]

            # Handle tuple unpacking: (a, b, c) = pl.yeild(...)
            if isinstance(target, ast.Tuple):
                # Check if value is a pl.yeild() call
                if isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "yeild":
                        # This is handled in yield parsing
                        self.parse_yield_assignment(target, stmt.value)
                        return

                raise ValueError("Tuple unpacking only supported for pl.yeild()")

            # Handle simple assignment
            if isinstance(target, ast.Name):
                var_name = target.id
                span = self.span_tracker.get_span(stmt)

                # Check if this is a yield assignment: var = pl.yeild(...)
                if isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "yeild":
                        # Handle yield assignment
                        yield_exprs = []
                        for arg in stmt.value.args:
                            expr = self.parse_expression(arg)
                            if not isinstance(expr, ir.Expr):
                                raise ValueError(f"Yield argument must be an IR expression, got {type(expr)}")
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
                self.scope_manager.define_var(var_name, var)
                return

        raise ValueError(f"Unsupported assignment: {ast.unparse(stmt)}")

    def parse_yield_assignment(self, target: ast.Tuple, value: ast.Call) -> None:
        """Parse yield assignment: (a, b) = pl.yeild(x, y).

        Args:
            target: Tuple of target variable names
            value: Call to pl.yeild()
        """
        # Parse yield expressions
        yield_exprs = []
        for arg in value.args:
            expr = self.parse_expression(arg)
            # Ensure it's an IR Expr
            if not isinstance(expr, ir.Expr):
                raise ValueError(f"Yield argument must be an IR expression, got {type(expr)}")
            yield_exprs.append(expr)

        # Emit yield statement
        span = self.span_tracker.get_span(value)
        self.builder.emit(ir.YieldStmt(yield_exprs, span))

        # Track yielded variable names for if/for statement processing
        if hasattr(self, "_current_yield_vars") and self._current_yield_vars is not None:
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self._current_yield_vars.append(elt.id)

        # For tuple yields at the for loop level, register the variables
        # (they'll be available as loop.get_result().return_vars)
        if self.in_for_loop and not self.in_if_stmt:
            # Register yielded variable names in scope
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Name):
                    var_name = elt.id
                    # Will be resolved from loop outputs
                    self.scope_manager.define_var(var_name, f"loop_yield_{i}")

    def parse_for_loop(self, stmt: ast.For) -> None:
        """Parse for loop with pl.range().

        Args:
            stmt: For AST node
        """
        # Check if iterator is pl.range()
        if not isinstance(stmt.iter, ast.Call):
            raise ValueError("For loop must use pl.range()")

        iter_call = stmt.iter
        func = iter_call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "range"):
            raise ValueError("For loop must use pl.range()")

        # Parse target: should be tuple like (i, (var1, var2, ...))
        if not isinstance(stmt.target, ast.Tuple) or len(stmt.target.elts) != 2:
            raise ValueError("For loop target must be: (loop_var, (iter_args...))")

        loop_var_node = stmt.target.elts[0]
        iter_args_node = stmt.target.elts[1]

        if not isinstance(loop_var_node, ast.Name):
            raise ValueError("Loop variable must be a simple name")

        loop_var_name = loop_var_node.id

        # Parse pl.range() arguments
        range_args = self._parse_range_call(iter_call)

        # Create loop variable
        loop_var = self.builder.var(loop_var_name, ir.ScalarType(DataType.INT64))

        # Begin for loop
        span = self.span_tracker.get_span(stmt)

        # Track loop output variable names
        loop_output_vars = []

        with self.builder.for_loop(
            loop_var, range_args["start"], range_args["stop"], range_args["step"], span
        ) as loop:
            self.current_loop_builder = loop
            self.in_for_loop = True
            self.scope_manager.enter_scope("for")

            # Register loop variable
            self.scope_manager.define_var(loop_var_name, loop_var, allow_redef=True)

            # Parse iter_args
            if not isinstance(iter_args_node, ast.Tuple):
                raise ValueError("Iter args must be a tuple")

            init_values = range_args["init_values"]
            if len(iter_args_node.elts) != len(init_values):
                raise ValueError(
                    f"Mismatch: {len(iter_args_node.elts)} iter_args but {len(init_values)} init_values"
                )

            # Add iter_args to loop
            for i, iter_arg_node in enumerate(iter_args_node.elts):
                if not isinstance(iter_arg_node, ast.Name):
                    raise ValueError("Iter arg must be a simple name")

                iter_arg_name = iter_arg_node.id
                init_value = init_values[i]

                # Add iter_arg
                iter_arg_var = loop.iter_arg(iter_arg_name, init_value)
                self.scope_manager.define_var(iter_arg_name, iter_arg_var, allow_redef=True)

            # Add return variables (same count as iter_args)
            for iter_arg_node in iter_args_node.elts:
                iter_arg_name = iter_arg_node.id
                return_var_name = f"{iter_arg_name}_out"
                loop.return_var(return_var_name)

            # Track yield outputs from loop
            prev_yield_tracker = getattr(self, "_current_yield_vars", None)
            self._current_yield_vars = []

            # Parse loop body
            for body_stmt in stmt.body:
                self.parse_statement(body_stmt)

            # Get the yielded variable names from the last yield
            loop_output_vars = self._current_yield_vars[:]

            # Restore yield tracker
            self._current_yield_vars = prev_yield_tracker

            # Exit for scope
            self.scope_manager.exit_scope()
            self.in_for_loop = False
            self.current_loop_builder = None

        # After for loop completes, register the output variables in the outer scope
        loop_result = loop.get_result()
        if hasattr(loop_result, "return_vars") and loop_result.return_vars and loop_output_vars:
            # Register each output variable with its name
            for i, var_name in enumerate(loop_output_vars):
                if i < len(loop_result.return_vars):
                    output_var = loop_result.return_vars[i]
                    self.scope_manager.define_var(var_name, output_var)

    def _parse_range_call(self, call: ast.Call) -> dict[str, Any]:
        """Parse pl.range() call arguments.

        Args:
            call: AST Call node for pl.range()

        Returns:
            Dictionary with start, stop, step, init_values
        """
        # Parse positional arguments
        if len(call.args) < 1:
            raise ValueError("pl.range() requires at least 1 argument (stop)")

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
                if isinstance(keyword.value, ast.List):
                    for elt in keyword.value.elts:
                        init_values.append(self.parse_expression(elt))
                else:
                    raise ValueError("init_values must be a list")

        return {"start": start, "stop": stop, "step": step, "init_values": init_values}

    def parse_if_statement(self, stmt: ast.If) -> None:
        """Parse if statement with phi nodes.

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

            # Now parse then branch
            self.scope_manager.enter_scope("if")
            for then_stmt in stmt.body:
                self.parse_statement(then_stmt)
            self.scope_manager.exit_scope()

            # Parse else branch if present
            if stmt.orelse:
                if_builder.else_()
                self.scope_manager.enter_scope("else")
                for else_stmt in stmt.orelse:
                    self.parse_statement(else_stmt)
                self.scope_manager.exit_scope()

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
        else:
            raise ValueError(f"Unsupported expression type: {type(expr).__name__}")

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
            raise ValueError(f"Undefined variable: {var_name}")

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
            raise ValueError(f"Unsupported constant type: {type(value)}")

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
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

        return op_map[op_type](left, right, span)

    def parse_compare(self, compare: ast.Compare) -> ir.Expr:
        """Parse comparison operation.

        Args:
            compare: Compare AST node

        Returns:
            IR comparison expression
        """
        if len(compare.ops) != 1 or len(compare.comparators) != 1:
            raise ValueError("Only simple comparisons supported")

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
            raise ValueError(f"Unsupported comparison: {op_type.__name__}")

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
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

        return op_map[op_type](operand, span)

    def parse_call(self, call: ast.Call) -> ir.Expr:
        """Parse function call.

        Args:
            call: Call AST node

        Returns:
            IR expression from call
        """
        func = call.func

        # Handle pl.yeild() specially
        if isinstance(func, ast.Attribute) and func.attr == "yeild":
            return self.parse_yeild_call(call)

        # Handle pl.op.tensor.* calls
        if isinstance(func, ast.Attribute):
            return self.parse_op_call(call)

        raise ValueError(f"Unsupported function call: {ast.unparse(call)}")

    def parse_yeild_call(self, call: ast.Call) -> ir.Expr:
        """Parse pl.yeild() call.

        Args:
            call: Call to pl.yeild() or pl.yeild()

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
        # This is for single assignment like: var = pl.yeild(expr)
        # We'll return a placeholder that gets resolved when if statement completes

        # Return first expression as the "value" of the yield
        # This handles: var = pl.yeild(expr)
        if len(yield_exprs) == 1:
            return yield_exprs[0]

        # For multiple yields, this should be handled as tuple assignment
        raise ValueError("Multiple yields should use tuple unpacking assignment")

    def parse_op_call(self, call: ast.Call) -> ir.Expr:
        """Parse operation call like pl.op.tensor.create().

        Args:
            call: Call AST node

        Returns:
            IR expression from operation
        """
        func = call.func

        # Navigate through attribute chain to find operation
        # e.g., pl.op.tensor.create -> ["pl", "op", "tensor", "create"]
        attrs = []
        node = func
        while isinstance(node, ast.Attribute):
            attrs.insert(0, node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            attrs.insert(0, node.id)

        # We expect: pl.op.tensor.{operation} or pl.op.tensor.{operation}
        if len(attrs) >= 4 and attrs[1] == "op" and attrs[2] == "tensor":
            op_name = attrs[3]
            return self._parse_tensor_op(op_name, call)

        raise ValueError(f"Unsupported operation call: {ast.unparse(call)}")

    def _parse_tensor_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse tensor operation.

        Args:
            op_name: Name of tensor operation
            call: Call AST node

        Returns:
            IR expression from tensor operation
        """
        # Parse arguments
        args = [self.parse_expression(arg) for arg in call.args]

        # Parse keyword arguments
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
                    # Complex unary expression
                    kwargs[key] = self.parse_expression(value)
            elif isinstance(value, ast.Name):
                if value.id in ["True", "False"]:
                    kwargs[key] = value.id == "True"
                else:
                    # It's a variable reference
                    kwargs[key] = self.parse_expression(value)
            elif isinstance(value, ast.Attribute):
                # Handle DataType.FP16 etc
                kwargs[key] = self.type_resolver.resolve_dtype(value)
            elif isinstance(value, ast.List):
                # Handle list literals
                kwargs[key] = self.parse_list(value)
            else:
                # Try to parse as expression
                kwargs[key] = self.parse_expression(value)

        # Call the appropriate tensor operation
        if hasattr(ir_op.tensor, op_name):
            op_func = getattr(ir_op.tensor, op_name)
            return op_func(*args, **kwargs)

        raise ValueError(f"Unknown tensor operation: {op_name}")

    def parse_attribute(self, attr: ast.Attribute) -> ir.Expr:
        """Parse attribute access.

        Args:
            attr: Attribute AST node

        Returns:
            IR expression
        """
        # This might be accessing a DataType enum or similar
        # For now, this is primarily used in calls, not standalone
        raise ValueError(f"Standalone attribute access not supported: {ast.unparse(attr)}")

    def parse_list(self, list_node: ast.List):
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

    def _scan_for_yields(self, stmts: list[ast.stmt]) -> list[str]:
        """Scan statements for yield assignments to determine output variable names.

        Args:
            stmts: List of statements to scan

        Returns:
            List of variable names that are yielded
        """
        yield_vars = []

        for stmt in stmts:
            # Check for annotated assignment with yeild: var: type = pl.yeild(...)
            if isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name) and isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "yeild":
                        yield_vars.append(stmt.target.id)

            # Check for regular assignment with yeild: var = pl.yeild(...)
            elif isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    # Single variable assignment
                    if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if isinstance(func, ast.Attribute) and func.attr == "yeild":
                            yield_vars.append(target.id)
                    # Tuple unpacking: (a, b) = pl.yeild(...)
                    elif isinstance(target, ast.Tuple) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if isinstance(func, ast.Attribute) and func.attr == "yeild":
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
