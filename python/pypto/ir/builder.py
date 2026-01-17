# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
IR Builder for incremental IR construction with context management.

Provides a Pythonic API for building IR using context managers with
automatic span tracking via the inspect module.
"""

import inspect
from contextlib import contextmanager
from typing import Optional

from pypto.pypto_core import ir


class IRBuilder:
    """IR Builder with context management and automatic span tracking.

    The IRBuilder provides a convenient API for building IR incrementally
    using context managers. Spans are automatically captured from the call
    site using Python's inspect module, or can be explicitly provided.

    Example:
        >>> ib = IRBuilder()
        >>> with ib.function("my_func") as f:
        ...     x = f.param("x", ir.ScalarType(ir.DataType.INT64))
        ...     y = f.param("y", ir.ScalarType(ir.DataType.INT64))
        ...     f.return_type(ir.ScalarType(ir.DataType.INT64))
        ...     result = ib.var("result", ir.ScalarType(ir.DataType.INT64))
        ...     ib.assign(result, ir.Add(x, y, ir.DataType.INT64, ir.Span.unknown()))
        >>> func = f.get_result()
    """

    def __init__(self):
        """Initialize the IR builder."""
        # Import here to avoid circular dependency
        from pypto.pypto_core.ir import (  # noqa: PLC0415
            IRBuilder as CppIRBuilder,  # type: ignore[attr-defined]
        )

        self._builder = CppIRBuilder()
        self._begin_spans = {}  # Track begin spans for multi-line contexts

    # ========== Context Managers for Multi-line Constructs ==========

    @contextmanager
    def function(self, name: str, span: Optional[ir.Span] = None):
        """Context manager for building functions.

        Args:
            name: Function name
            span: Optional explicit span. If None, automatically captured from call site.

        Yields:
            FunctionBuilder: Helper object for building the function

        Example:
            >>> with ib.function("add") as f:
            ...     x = f.param("x", ir.ScalarType(ir.DataType.INT64))
            ...     f.return_type(ir.ScalarType(ir.DataType.INT64))
        """
        begin_span = span if span is not None else self._capture_call_span()
        ctx_id = id(begin_span)
        self._begin_spans[ctx_id] = begin_span

        self._builder.BeginFunction(name, begin_span)
        builder_obj = FunctionBuilder(self)
        try:
            yield builder_obj
        finally:
            end_span = self._capture_call_span() if span is None else span
            combined_span = self._combine_spans(self._begin_spans[ctx_id], end_span)
            result = self._builder.EndFunction(combined_span)
            builder_obj._result = result
            del self._begin_spans[ctx_id]

    @contextmanager
    def for_loop(self, loop_var, start, stop, step, span: Optional[ir.Span] = None):
        """Context manager for building for loops.

        Args:
            loop_var: Loop variable
            start: Start value expression
            stop: Stop value expression
            step: Step value expression
            span: Optional explicit span. If None, automatically captured.

        Yields:
            ForLoopBuilder: Helper object for building the loop

        Example:
            >>> i = ib.var("i", ir.ScalarType(ir.DataType.INT64))
            >>> with ib.for_loop(i, start, stop, step) as loop:
            ...     sum_iter = loop.iter_arg("sum", type, init_val)
        """
        begin_span = span if span is not None else self._capture_call_span()
        ctx_id = id(begin_span) + 1  # Different id
        self._begin_spans[ctx_id] = begin_span

        self._builder.BeginForLoop(loop_var, start, stop, step, begin_span)
        builder_obj = ForLoopBuilder(self)
        try:
            yield builder_obj
        finally:
            end_span = self._capture_call_span() if span is None else span
            combined_span = self._combine_spans(self._begin_spans[ctx_id], end_span)
            result = self._builder.EndForLoop(combined_span)
            builder_obj._result = result
            del self._begin_spans[ctx_id]

    @contextmanager
    def if_stmt(self, condition, span: Optional[ir.Span] = None):
        """Context manager for building if statements.

        Args:
            condition: Condition expression
            span: Optional explicit span. If None, automatically captured.

        Yields:
            IfStmtBuilder: Helper object for building the if statement

        Example:
            >>> with ib.if_stmt(condition) as if_builder:
            ...     # then branch
            ...     ib.assign(x, value)
            ...     if_builder.else_()
            ...     # else branch
            ...     ib.assign(x, other_value)
        """
        begin_span = span if span is not None else self._capture_call_span()
        ctx_id = id(begin_span) + 2
        self._begin_spans[ctx_id] = begin_span

        self._builder.BeginIf(condition, begin_span)
        builder_obj = IfStmtBuilder(self)
        try:
            yield builder_obj
        finally:
            end_span = self._capture_call_span() if span is None else span
            combined_span = self._combine_spans(self._begin_spans[ctx_id], end_span)
            result = self._builder.EndIf(combined_span)
            builder_obj._result = result
            del self._begin_spans[ctx_id]

    # ========== Single-line Methods with Optional Explicit Span ==========

    def var(self, name: str, type, span: Optional[ir.Span] = None):
        """Create a variable with span from call site or explicit span.

        Args:
            name: Variable name
            type: Variable type
            span: Optional explicit span. If None, captured from call site.

        Returns:
            Var: The created variable
        """
        actual_span = span if span is not None else self._capture_call_span()
        return self._builder.Var(name, type, actual_span)

    def assign(self, var, value, span: Optional[ir.Span] = None):
        """Create assignment statement and emit it.

        Args:
            var: Variable to assign to
            value: Expression value
            span: Optional explicit span. If None, captured from call site.

        Returns:
            AssignStmt: The created assignment statement
        """
        actual_span = span if span is not None else self._capture_call_span()
        return self._builder.Assign(var, value, actual_span)

    def emit(self, stmt):
        """Add a statement to the current context.

        Args:
            stmt: Statement to emit
        """
        self._builder.Emit(stmt)

    def return_stmt(self, values=None, span: Optional[ir.Span] = None):
        """Create return statement and emit it.

        Args:
            values: Expression value(s) to return. Can be:
                   - None for empty return
                   - Single expression
                   - List of expressions
            span: Optional explicit span. If None, captured from call site.

        Returns:
            ReturnStmt: The created return statement
        """
        actual_span = span if span is not None else self._capture_call_span()

        # Normalize values to list
        if values is None:
            value_list = []
        elif isinstance(values, list):
            value_list = values
        else:
            value_list = [values]

        return self._builder.Return(value_list, actual_span)

    # ========== Context State Queries ==========

    def in_function(self) -> bool:
        """Check if currently inside a function."""
        return self._builder.InFunction()

    def in_loop(self) -> bool:
        """Check if currently inside a for loop."""
        return self._builder.InLoop()

    def in_if(self) -> bool:
        """Check if currently inside an if statement."""
        return self._builder.InIf()

    # ========== Private Span Tracking Helpers ==========

    def _capture_call_span(self) -> ir.Span:
        """Capture span from immediate caller using inspect.

        Returns:
            Span: Source location of the caller
        """
        # Go back 2 frames:
        # frame 0 = _capture_call_span
        # frame 1 = our wrapper method (var, assign, etc.)
        # frame 2 = user's code (what we want)
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            frame = frame.f_back.f_back
        if frame is not None:
            info = inspect.getframeinfo(frame)
            return ir.Span(info.filename, info.lineno, 0)
        return ir.Span.unknown()

    def _combine_spans(self, begin: ir.Span, end: ir.Span) -> ir.Span:
        """Combine begin and end spans into a multi-line span.

        Args:
            begin: Begin span (from context enter)
            end: End span (from context exit)

        Returns:
            Span: Combined span covering the range
        """
        return ir.Span(begin.filename, begin.begin_line, begin.begin_column, end.begin_line, end.begin_column)


class FunctionBuilder:
    """Helper for building functions within a function context."""

    def __init__(self, builder: IRBuilder):
        """Initialize function builder.

        Args:
            builder: Parent IR builder
        """
        self._builder = builder
        self._result = None

    def param(self, name: str, type, span: Optional[ir.Span] = None):
        """Add function parameter.

        Args:
            name: Parameter name
            type: Parameter type
            span: Optional explicit span. If None, captured from call site.

        Returns:
            Var: The parameter variable
        """
        actual_span = span if span is not None else self._builder._capture_call_span()
        return self._builder._builder.FuncArg(name, type, actual_span)

    def return_type(self, type):
        """Add return type to the function.

        Args:
            type: Return type
        """
        self._builder._builder.ReturnType(type)

    def get_result(self):  # type: ignore[return]
        """Get the built Function.

        Returns:
            Function: The completed function IR node (or None if not yet finalized)
        """
        return self._result


class ForLoopBuilder:
    """Helper for building for loops within a loop context."""

    def __init__(self, builder: IRBuilder):
        """Initialize for loop builder.

        Args:
            builder: Parent IR builder
        """
        self._builder = builder
        self._result = None

    def iter_arg(self, name: str, type, init_value, span: Optional[ir.Span] = None):
        """Add iteration argument (loop-carried value).

        Args:
            name: Iteration argument name
            type: Variable type
            init_value: Initial value expression
            span: Optional explicit span. If None, captured from call site.

        Returns:
            IterArg: The iteration argument variable
        """
        actual_span = span if span is not None else self._builder._capture_call_span()
        iter_arg = ir.IterArg(name, type, init_value, actual_span)
        self._builder._builder.AddIterArg(iter_arg)
        return iter_arg

    def return_var(self, name: str, type, span: Optional[ir.Span] = None):
        """Add return variable to capture final iteration value.

        Args:
            name: Return variable name
            type: Variable type
            span: Optional explicit span. If None, captured from call site.

        Returns:
            Var: The return variable
        """
        actual_span = span if span is not None else self._builder._capture_call_span()
        var = ir.Var(name, type, actual_span)
        self._builder._builder.AddReturnVar(var)
        return var

    def get_result(self):
        """Get the built ForStmt.

        Returns:
            ForStmt: The completed for loop IR node
        """
        return self._result


class IfStmtBuilder:
    """Helper for building if statements within an if context."""

    def __init__(self, builder: IRBuilder):
        """Initialize if statement builder.

        Args:
            builder: Parent IR builder
        """
        self._builder = builder
        self._result = None

    def else_(self, span: Optional[ir.Span] = None):
        """Begin else branch of the if statement.

        Args:
            span: Optional explicit span. If None, captured from call site.
        """
        actual_span = span if span is not None else self._builder._capture_call_span()
        self._builder._builder.BeginElse(actual_span)

    def return_var(self, name: str, type, span: Optional[ir.Span] = None):
        """Add return variable for SSA phi node.

        Args:
            name: Return variable name
            type: Variable type
            span: Optional explicit span. If None, captured from call site.

        Returns:
            Var: The return variable
        """
        actual_span = span if span is not None else self._builder._capture_call_span()
        var = ir.Var(name, type, actual_span)
        self._builder._builder.AddIfReturnVar(var)
        return var

    def get_result(self):
        """Get the built IfStmt.

        Returns:
            IfStmt: The completed if statement IR node
        """
        return self._result


__all__ = ["IRBuilder", "FunctionBuilder", "ForLoopBuilder", "IfStmtBuilder"]
