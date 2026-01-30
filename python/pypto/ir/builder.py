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
from typing import Iterator, List, Optional, Sequence, Union

from pypto.pypto_core import DataType, ir
from pypto.pypto_core.ir import IRBuilder as CppIRBuilder

from .utils import _normalize_expr


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

    def __init__(self) -> None:
        """Initialize the IR builder."""
        # Import here to avoid circular dependency

        self._builder = CppIRBuilder()
        self._begin_spans: dict[int, ir.Span] = {}  # Track begin spans for multi-line contexts

    # ========== Context Managers for Multi-line Constructs ==========

    @contextmanager
    def function(self, name: str, span: Optional[ir.Span] = None) -> Iterator["FunctionBuilder"]:
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

        self._builder.begin_function(name, begin_span)
        builder_obj = FunctionBuilder(self)
        try:
            yield builder_obj
        finally:
            end_span = self._capture_call_span() if span is None else span
            combined_span = self._combine_spans(self._begin_spans[ctx_id], end_span)
            result = self._builder.end_function(combined_span)
            builder_obj._result = result
            del self._begin_spans[ctx_id]

    @contextmanager
    def for_loop(
        self,
        loop_var: ir.Var,
        start: Union[int, ir.Expr],
        stop: Union[int, ir.Expr],
        step: Union[int, ir.Expr],
        span: Optional[ir.Span] = None,
    ) -> Iterator["ForLoopBuilder"]:
        """Context manager for building for loops.

        Args:
            loop_var: Loop variable
            start: Start value (int or Expr)
            stop: Stop value (int or Expr)
            step: Step value (int or Expr)
            span: Optional explicit span. If None, automatically captured.

        Yields:
            ForLoopBuilder: Helper object for building the loop

        Example:
            >>> i = ib.var("i", ir.ScalarType(ir.DataType.INT64))
            >>> with ib.for_loop(i, 0, 10, 1) as loop:
            ...     sum_iter = loop.iter_arg("sum", init_val)
        """
        begin_span = span if span is not None else self._capture_call_span()
        ctx_id = id(begin_span) + 1  # Different id
        self._begin_spans[ctx_id] = begin_span

        # Normalize all expression parameters
        start_expr = _normalize_expr(start, begin_span)
        stop_expr = _normalize_expr(stop, begin_span)
        step_expr = _normalize_expr(step, begin_span)

        self._builder.begin_for_loop(loop_var, start_expr, stop_expr, step_expr, begin_span)
        builder_obj = ForLoopBuilder(self)
        try:
            yield builder_obj
        finally:
            end_span = self._capture_call_span() if span is None else span
            combined_span = self._combine_spans(self._begin_spans[ctx_id], end_span)
            result = self._builder.end_for_loop(combined_span)
            builder_obj._result = result
            del self._begin_spans[ctx_id]

    @contextmanager
    def if_stmt(
        self, condition: Union[int, ir.Expr], span: Optional[ir.Span] = None
    ) -> Iterator["IfStmtBuilder"]:
        """Context manager for building if statements.

        Args:
            condition: Condition expression (int or Expr)
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

        condition_expr = _normalize_expr(condition, begin_span)
        self._builder.begin_if(condition_expr, begin_span)
        builder_obj = IfStmtBuilder(self)
        try:
            yield builder_obj
        finally:
            end_span = self._capture_call_span() if span is None else span
            combined_span = self._combine_spans(self._begin_spans[ctx_id], end_span)
            result = self._builder.end_if(combined_span)
            builder_obj._result = result
            del self._begin_spans[ctx_id]

    @contextmanager
    def program(self, name: str, span: Optional[ir.Span] = None) -> Iterator["ProgramBuilder"]:
        """Context manager for building programs.

        Args:
            name: Program name
            span: Optional explicit span. If None, automatically captured.

        Yields:
            ProgramBuilder: Helper object for building the program

        Example:
            >>> with ib.program("my_program") as p:
            ...     # Declare functions up front
            ...     helper_gvar = p.declare_function("helper")
            ...
            ...     # Build function1
            ...     with ib.function("func1") as f:
            ...         # function body
            ...     func1 = f.get_result()
            ...     p.add_function(func1)
            ...
            ...     # Build function2
            ...     with ib.function("func2") as f:
            ...         # function body
            ...     func2 = f.get_result()
            ...     p.add_function(func2)
            >>> program = p.get_result()
        """
        begin_span = span if span is not None else self._capture_call_span()
        ctx_id = id(begin_span) + 3  # Different id
        self._begin_spans[ctx_id] = begin_span

        self._builder.begin_program(name, begin_span)
        builder_obj = ProgramBuilder(self)
        try:
            yield builder_obj
        finally:
            end_span = self._capture_call_span() if span is None else span
            combined_span = self._combine_spans(self._begin_spans[ctx_id], end_span)
            result = self._builder.end_program(combined_span)
            builder_obj._result = result
            del self._begin_spans[ctx_id]

    # ========== Single-line Methods with Optional Explicit Span ==========

    def var(self, name: str, type: ir.Type, span: Optional[ir.Span] = None) -> ir.Var:
        """Create a variable with span from call site or explicit span.

        Args:
            name: Variable name
            type: Variable type
            span: Optional explicit span. If None, captured from call site.

        Returns:
            Var: The created variable
        """
        actual_span = span if span is not None else self._capture_call_span()
        return self._builder.var(name, type, actual_span)

    def assign(
        self,
        var: ir.Var,
        value: Union[int, float, ir.Expr],
        span: Optional[ir.Span] = None,
    ) -> ir.AssignStmt:
        """Create assignment statement and emit it.

        Args:
            var: Variable to assign to (must be an ir.Var)
            value: Expression value (int, float, or Expr)
            span: Optional explicit span. If None, captured from call site.

        Returns:
            AssignStmt: The created assignment statement
        """
        actual_span = span if span is not None else self._capture_call_span()
        value_expr = _normalize_expr(value, actual_span)
        return self._builder.assign(var, value_expr, actual_span)

    def let(
        self,
        name: str,
        value: Union[int, float, ir.Expr],
        type: Optional[ir.Type] = None,
        span: Optional[ir.Span] = None,
    ) -> ir.Var:
        """Create a variable and assign a value to it in one statement.

        This is a convenience method that combines var() and assign() for the
        common pattern of creating a variable and immediately assigning to it.

        The type is automatically inferred from the value expression. If an explicit
        type is provided, it is used to validate that the inferred type matches.

        For Call expressions with GlobalVar ops, the return type is automatically
        inferred from the function signature if available.

        Args:
            name: Variable name
            value: Expression value (int, float, or Expr)
            type: Optional type for validation. If provided, must match the inferred type.
            span: Optional explicit span. If None, captured from call site.

        Returns:
            Var: The created variable

        Raises:
            ValueError: If explicit type is provided and doesn't match inferred type

        Example:
            >>> # Type is inferred from the expression:
            >>> x = ib.let("x", 42)
            >>> # Or with explicit type validation:
            >>> x = ib.let("x", 42, type=ir.ScalarType(ir.DataType.INT64))
            >>> # For function calls, type is auto-inferred from function signature:
            >>> result = ib.let("result", ir.Call(func_gvar, [x], span))
        """
        actual_span = span if span is not None else self._capture_call_span()
        value_expr = _normalize_expr(value, actual_span)

        # Auto-infer return type for Call expressions with GlobalVar
        if isinstance(value_expr, ir.Call) and isinstance(value_expr.op, ir.GlobalVar):
            # Check if the Call has UnknownType and we're inside a program
            if isinstance(value_expr.type, ir.UnknownType) and self._builder.in_program():
                # Try to get return types from the function signature
                return_types = self._builder.get_function_return_types(value_expr.op)
                if len(return_types) == 1:
                    # Recreate the Call with the correct return type
                    value_expr = ir.Call(value_expr.op, value_expr.args, return_types[0], actual_span)
                elif len(return_types) > 1:
                    raise ValueError(
                        f"Function '{value_expr.op.name}' returns {len(return_types)} values, "
                        f"but let() can only assign single return values. "
                        f"Use explicit tuple unpacking or multiple let() statements."
                    )

        # Infer type from the value expression
        inferred_type = value_expr.type

        # If explicit type is provided, validate it matches the inferred type
        if type is not None and type != inferred_type:
            raise ValueError(
                f"Type mismatch in let statement for variable '{name}':\n"
                f"  Inferred type: {inferred_type}\n"
                f"  Provided type: {type}"
            )
        final_type = inferred_type

        var = self._builder.var(name, final_type, actual_span)
        self._builder.assign(var, value_expr, actual_span)
        return var

    def emit(self, stmt: ir.Stmt) -> None:
        """Add a statement to the current context.

        Args:
            stmt: Statement to emit
        """
        self._builder.emit(stmt)

    def return_stmt(
        self,
        values: Optional[Union[int, float, ir.Expr, Sequence[Union[int, float, ir.Expr]]]] = None,
        span: Optional[ir.Span] = None,
    ) -> ir.ReturnStmt:
        """Create return statement and emit it.

        Args:
            values: Expression value(s) to return. Can be:
                   - None for empty return
                   - Single expression (int, float, or Expr)
                   - List of expressions (int, float, or Expr)
            span: Optional explicit span. If None, captured from call site.

        Returns:
            ReturnStmt: The created return statement
        """
        actual_span = span if span is not None else self._capture_call_span()

        # Normalize values to list and convert each element
        if values is None:
            value_list = []
        elif isinstance(values, Sequence):
            value_list = [_normalize_expr(v, actual_span) for v in values]
        else:
            value_list = [_normalize_expr(values, actual_span)]

        return self._builder.return_(value_list, actual_span)

    def eval_stmt(
        self,
        expr: Union[int, float, ir.Expr],
        span: Optional[ir.Span] = None,
    ) -> ir.EvalStmt:
        """Create evaluation statement and emit it.

        Evaluation statements execute expressions for their side effects,
        discarding any return value. Useful for operations like barriers,
        synchronization primitives, or other side-effect-only operations.

        Args:
            expr: Expression to evaluate (int, float, or Expr)
            span: Optional explicit span. If None, captured from call site.

        Returns:
            EvalStmt: The created evaluation statement
        """
        actual_span = span if span is not None else self._capture_call_span()
        expr_normalized = _normalize_expr(expr, actual_span)
        stmt = ir.EvalStmt(expr_normalized, actual_span)
        self._builder.emit(stmt)
        return stmt

    # ========== Context State Queries ==========

    def in_function(self) -> bool:
        """Check if currently inside a function."""
        return self._builder.in_function()

    def in_loop(self) -> bool:
        """Check if currently inside a for loop."""
        return self._builder.in_loop()

    def in_if(self) -> bool:
        """Check if currently inside an if statement."""
        return self._builder.in_if()

    # ========== Type and MemRef Creation Helpers ==========

    def memref(
        self,
        memory_space: ir.MemorySpace,
        addr: Union[int, ir.Expr],
        size: int,
        id: int,
        span: Optional[ir.Span] = None,
    ) -> ir.MemRef:
        """Create a MemRef with normalized address expression.

        Args:
            memory_space: Memory space (DDR, UB, L1, L0A, L0B, L0C)
            addr: Address expression (int or Expr)
            size: Size in bytes
            id: Unique identifier for this MemRef
            span: Optional explicit span. If None, captured from call site.

        Returns:
            MemRef: The created memory reference

        Example:
            >>> addr = ir.ConstInt(0x1000, DataType.INT64, ir.Span.unknown())
            >>> memref = ib.memref(ir.MemorySpace.DDR, addr, 1024, 0)
        """
        actual_span = span if span is not None else self._capture_call_span()
        addr_expr = _normalize_expr(addr, actual_span)
        return ir.MemRef(memory_space, addr_expr, size, id, actual_span)

    def tile_view(
        self,
        valid_shape: Sequence[Union[int, ir.Expr]],
        stride: Sequence[Union[int, ir.Expr]],
        start_offset: Union[int, ir.Expr],
        span: Optional[ir.Span] = None,
    ) -> ir.TileView:
        """Create a TileView with normalized expressions.

        Args:
            valid_shape: Valid shape dimensions (list of int or Expr)
            stride: Stride for each dimension (list of int or Expr)
            start_offset: Starting offset (int or Expr)
            span: Optional explicit span. If None, captured from call site.

        Returns:
            TileView: The created tile view

        Example:
            >>> valid_shape = [16, 16]
            >>> stride = [1, 16]
            >>> start_offset = 0
            >>> tv = ib.tile_view(valid_shape, stride, start_offset)
        """
        actual_span = span if span is not None else self._capture_call_span()
        valid_shape_exprs = [_normalize_expr(dim, actual_span) for dim in valid_shape]
        stride_exprs = [_normalize_expr(s, actual_span) for s in stride]
        start_offset_expr = _normalize_expr(start_offset, actual_span)
        return ir.TileView(valid_shape_exprs, stride_exprs, start_offset_expr)

    def tensor_type(
        self,
        shape: Sequence[Union[int, ir.Expr]],
        dtype: DataType,
        memref: Optional[ir.MemRef] = None,
        span: Optional[ir.Span] = None,
    ) -> ir.TensorType:
        """Create a TensorType with normalized shape and optional memref.

        Args:
            shape: Shape dimensions (list of int or Expr)
            dtype: Element data type
            memref: Optional memory reference
            span: Optional explicit span. If None, captured from call site.

        Returns:
            TensorType: The created tensor type

        Example:
            >>> # Simple tensor type
            >>> tensor_t = ib.tensor_type([64, 128], DataType.FP32)
            >>> # Tensor type with memref
            >>> memref = ib.memref(ir.MemorySpace.DDR, 0x1000, 1024)
            >>> tensor_t = ib.tensor_type([64, 128], DataType.FP32, memref=memref)
        """
        actual_span = span if span is not None else self._capture_call_span()
        shape_exprs = [_normalize_expr(dim, actual_span) for dim in shape]
        return ir.TensorType(shape_exprs, dtype, memref)

    def tile_type(
        self,
        shape: Sequence[Union[int, ir.Expr]],
        dtype: DataType,
        memref: Optional[ir.MemRef] = None,
        tile_view: Optional[ir.TileView] = None,
        span: Optional[ir.Span] = None,
    ) -> ir.TileType:
        """Create a TileType with normalized shape, optional memref and tile_view.

        Args:
            shape: Shape dimensions (list of int or Expr, at most 2 dimensions)
            dtype: Element data type
            memref: Optional memory reference
            tile_view: Optional tile view information
            span: Optional explicit span. If None, captured from call site.

        Returns:
            TileType: The created tile type

        Raises:
            ValueError: If shape has more than 2 dimensions

        Example:
            >>> # Simple tile type
            >>> tile_t = ib.tile_type([16, 16], DataType.FP16)
            >>> # Tile type with memref and tile_view
            >>> memref = ib.memref(ir.MemorySpace.L0A, 0, 512)
            >>> tv = ib.tile_view([16, 16], [1, 16], 0)
            >>> tile_t = ib.tile_type([16, 16], DataType.FP16, memref=memref, tile_view=tv)
        """
        actual_span = span if span is not None else self._capture_call_span()
        shape_exprs = [_normalize_expr(dim, actual_span) for dim in shape]
        return ir.TileType(shape_exprs, dtype, memref, tile_view)

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
            return ir.Span(info.filename, info.lineno, -1)
        return ir.Span.unknown()

    def _combine_spans(self, begin: ir.Span, end: ir.Span) -> ir.Span:
        """Combine begin and end spans into a multi-line span.

        Args:
            begin: Begin span (from context enter)
            end: End span (from context exit)

        Returns:
            Span: Combined span covering the range
        """
        return ir.Span(
            begin.filename,
            begin.begin_line,
            begin.begin_column,
            end.begin_line,
            end.begin_column,
        )


class FunctionBuilder:
    """Helper for building functions within a function context."""

    def __init__(self, builder: IRBuilder) -> None:
        """Initialize function builder.

        Args:
            builder: Parent IR builder
        """
        self._builder = builder
        self._result: Optional[ir.Function] = None

    def param(self, name: str, type: ir.Type, span: Optional[ir.Span] = None) -> ir.Var:
        """Add function parameter.

        Args:
            name: Parameter name
            type: Parameter type
            span: Optional explicit span. If None, captured from call site.

        Returns:
            Var: The parameter variable
        """
        actual_span = span if span is not None else self._builder._capture_call_span()
        return self._builder._builder.func_arg(name, type, actual_span)

    def return_type(self, type: ir.Type) -> None:
        """Add return type to the function.

        Args:
            type: Return type
        """
        self._builder._builder.return_type(type)

    def get_result(self) -> ir.Function:
        """Get the built Function.

        Returns:
            Function: The completed function IR node (or None if not yet finalized)
        """
        assert self._result is not None
        return self._result


class ForLoopBuilder:
    """Helper for building for loops within a loop context."""

    def __init__(self, builder: IRBuilder) -> None:
        """Initialize for loop builder.

        Args:
            builder: Parent IR builder
        """
        self._builder = builder
        self._result: Optional[ir.ForStmt] = None
        self._iter_args: List[ir.IterArg] = []  # Track iter_args for type inference
        self._return_var_count = 0  # Track number of return_vars added

    def iter_arg(
        self,
        name: str,
        init_value: Union[int, float, ir.Expr],
        type: Optional[ir.Type] = None,
        span: Optional[ir.Span] = None,
    ) -> ir.IterArg:
        """Add iteration argument (loop-carried value).

        The type is automatically inferred from the init_value expression. If an explicit
        type is provided, it is used to validate that the inferred type matches.

        Args:
            name: Iteration argument name
            init_value: Initial value (int, float, or Expr)
            type: Optional type for validation. If provided, must match the inferred type.
            span: Optional explicit span. If None, captured from call site.

        Returns:
            IterArg: The iteration argument variable

        Raises:
            ValueError: If explicit type is provided and doesn't match inferred type

        Example:
            >>> # Type is inferred from the initial value:
            >>> sum_iter = loop.iter_arg("sum", 0)
            >>> # Or with explicit type validation:
            >>> sum_iter = loop.iter_arg("sum", 0, type=ir.ScalarType(ir.DataType.INT64))
        """
        actual_span = span if span is not None else self._builder._capture_call_span()
        init_expr = _normalize_expr(init_value, actual_span)

        # Infer type from the init_value expression
        inferred_type = init_expr.type

        # If explicit type is provided, validate it matches the inferred type
        if type is not None and type != inferred_type:
            raise ValueError(
                f"Type mismatch in iter_arg for '{name}':\n"
                f"  Inferred type: {inferred_type}\n"
                f"  Provided type: {type}"
            )
        final_type = inferred_type

        iter_arg = ir.IterArg(name, final_type, init_expr, actual_span)
        self._builder._builder.add_iter_arg(iter_arg)
        self._iter_args.append(iter_arg)  # Track for return_var type inference
        return iter_arg

    def return_var(self, name: str, type: Optional[ir.Type] = None, span: Optional[ir.Span] = None) -> ir.Var:
        """Add return variable to capture final iteration value.

        The type can be automatically inferred from the corresponding iter_arg (by index).
        If explicit type is provided, it is used to validate against the inferred type.

        Args:
            name: Return variable name
            type: Optional type. If None, inferred from corresponding iter_arg by index.
            span: Optional explicit span. If None, captured from call site.

        Returns:
            Var: The return variable

        Raises:
            ValueError: If type cannot be inferred or provided type doesn't match

        Example:
            >>> # Type is inferred from corresponding iter_arg:
            >>> sum_final = loop.return_var("sum_final")
            >>> # Or with explicit type validation:
            >>> sum_final = loop.return_var("sum_final", type=ir.ScalarType(ir.DataType.INT64))
        """
        actual_span = span if span is not None else self._builder._capture_call_span()

        # Try to infer type from corresponding iter_arg by index
        inferred_type = None
        if self._return_var_count < len(self._iter_args):
            inferred_type = self._iter_args[self._return_var_count].type

        # Determine final type
        if type is None:
            if inferred_type is None:
                raise ValueError(
                    f"Cannot infer type for return_var '{name}': "
                    f"no corresponding iter_arg found. Please provide explicit type."
                )
            final_type = inferred_type
        else:
            # Validate provided type if we have inferred type
            if inferred_type is not None and type != inferred_type:
                raise ValueError(
                    f"Type mismatch in return_var '{name}':\n"
                    f"  Inferred type (from iter_arg): {inferred_type}\n"
                    f"  Provided type: {type}"
                )
            final_type = type

        var = ir.Var(name, final_type, actual_span)
        self._builder._builder.add_return_var(var)
        self._return_var_count += 1
        return var

    def output(self, index: int = 0) -> ir.Var:
        """Get a single output return variable from the for loop.

        This is a convenience method to access the return variables after the for
        loop is built. Use the index parameter to select which return variable.

        Args:
            index: Index of the return variable to get (default: 0)

        Returns:
            Var: The return variable at the specified index

        Raises:
            AssertionError: If called before for loop is complete
            IndexError: If index is out of range

        Example:
            >>> with ib.for_loop(i, 0, 10, 1) as loop:
            ...     sum_iter = loop.iter_arg("sum", 0)
            ...     loop.return_var("sum_final")
            ...     # ... loop body ...
            >>> result = loop.output()  # Get the first return variable
            >>> # Or for multiple return vars:
            >>> result1 = loop.output(0)
            >>> result2 = loop.output(1)
        """
        assert self._result is not None, "For loop not yet complete"
        if index >= len(self._result.return_vars):
            raise IndexError(
                f"Return variable index {index} out of range "
                f"(for loop has {len(self._result.return_vars)} return vars)"
            )
        return self._result.return_vars[index]

    def outputs(self) -> List[ir.Var]:
        """Get all output return variables from the for loop.

        This is a convenience method to access all return variables at once after
        the for loop is built.

        Returns:
            List[Var]: List of all return variables

        Raises:
            AssertionError: If called before for loop is complete

        Example:
            >>> with ib.for_loop(i, 0, 10, 1) as loop:
            ...     sum_iter = loop.iter_arg("sum", 0)
            ...     prod_iter = loop.iter_arg("prod", 1)
            ...     loop.return_var("sum_final")
            ...     loop.return_var("prod_final")
            ...     # ... loop body ...
            >>> sum_result, prod_result = loop.outputs()  # Get all return variables
        """
        assert self._result is not None, "For loop not yet complete"
        return list(self._result.return_vars)

    def get_result(self) -> ir.ForStmt:
        """Get the built ForStmt.

        Returns:
            ForStmt: The completed for loop IR node
        """
        assert self._result is not None
        return self._result


class IfStmtBuilder:
    """Helper for building if statements within an if context."""

    def __init__(self, builder: IRBuilder) -> None:
        """Initialize if statement builder.

        Args:
            builder: Parent IR builder
        """
        self._builder = builder
        self._result: Optional[ir.IfStmt] = None

    def else_(self, span: Optional[ir.Span] = None) -> None:
        """Begin else branch of the if statement.

        Args:
            span: Optional explicit span. If None, captured from call site.
        """
        actual_span = span if span is not None else self._builder._capture_call_span()
        self._builder._builder.begin_else(actual_span)

    def return_var(self, name: str, type: ir.Type, span: Optional[ir.Span] = None) -> None:
        """Add return variable for SSA phi node.

        Note: Type must be provided explicitly. Type inference is not supported for
        if statement return_vars because they are declared before yield statements
        are executed. Type inference could be implemented in C++ EndIf logic.

        Args:
            name: Return variable name
            type: Variable type (required)
            span: Optional explicit span. If None, captured from call site.

        Example:
            >>> # Type must be provided explicitly:
            >>> if_builder.return_var("result", ir.ScalarType(ir.DataType.INT64))
        """
        actual_span = span if span is not None else self._builder._capture_call_span()
        var = ir.Var(name, type, actual_span)
        self._builder._builder.add_if_return_var(var)

    def output(self, index: int = 0) -> ir.Var:
        """Get a single output return variable from the if statement.

        This is a convenience method to access the return variables after the if
        statement is built. Use the index parameter to select which return variable.

        Args:
            index: Index of the return variable to get (default: 0)

        Returns:
            Var: The return variable at the specified index

        Raises:
            AssertionError: If called before if statement is complete
            IndexError: If index is out of range

        Example:
            >>> with ib.if_stmt(condition) as if_builder:
            ...     if_builder.return_var("result", ir.ScalarType(DataType.INT64))
            ...     # ... if/else branches ...
            >>> result = if_builder.output()  # Get the first return variable
            >>> # Or for multiple return vars:
            >>> result1 = if_builder.output(0)
            >>> result2 = if_builder.output(1)
        """
        assert self._result is not None, "If statement not yet complete"
        if index >= len(self._result.return_vars):
            raise IndexError(
                f"Return variable index {index} out of range "
                f"(if statement has {len(self._result.return_vars)} return vars)"
            )
        return self._result.return_vars[index]

    def outputs(self) -> List[ir.Var]:
        """Get all output return variables from the if statement.

        This is a convenience method to access all return variables at once after
        the if statement is built.

        Returns:
            List[Var]: List of all return variables

        Raises:
            AssertionError: If called before if statement is complete

        Example:
            >>> with ib.if_stmt(condition) as if_builder:
            ...     if_builder.return_var("x", ir.ScalarType(DataType.INT64))
            ...     if_builder.return_var("y", ir.ScalarType(DataType.INT64))
            ...     # ... if/else branches ...
            >>> x, y = if_builder.outputs()  # Get all return variables
        """
        assert self._result is not None, "If statement not yet complete"
        return list(self._result.return_vars)

    def get_result(self) -> ir.IfStmt:
        """Get the built IfStmt.

        Returns:
            IfStmt: The completed if statement IR node
        """
        assert self._result is not None
        return self._result


class ProgramBuilder:
    """Helper for building programs within a program context."""

    def __init__(self, builder: IRBuilder) -> None:
        """Initialize program builder.

        Args:
            builder: Parent IR builder
        """
        self._builder = builder
        self._result: Optional[ir.Program] = None

    def declare_function(self, name: str) -> ir.GlobalVar:
        """Declare a function and get its GlobalVar for cross-function calls.

        This should be called before building the function to enable other
        functions to reference it via Call expressions.

        Args:
            name: Function name to declare

        Returns:
            GlobalVar that can be used in Call expressions

        Example:
            >>> with ib.program("my_program") as p:
            ...     # Declare functions up front
            ...     helper_gvar = p.declare_function("helper")
            ...     main_gvar = p.declare_function("main")
            ...
            ...     # Build helper function
            ...     with ib.function("helper") as f:
            ...         # ... function body
            ...     p.add_function(f.get_result())
            ...
            ...     # Build main function that calls helper
            ...     with ib.function("main") as f:
            ...         x = f.param("x", ir.ScalarType(DataType.INT64))
            ...         # Call helper function using its GlobalVar
            ...         result = ib.let("result", ir.Call(helper_gvar, [x], span))
            ...         ib.return_stmt(result)
            ...     p.add_function(f.get_result())
        """
        return self._builder._builder.declare_function(name)

    def get_global_var(self, name: str) -> ir.GlobalVar:
        """Get GlobalVar for a declared function.

        Args:
            name: Function name

        Returns:
            GlobalVar for the function

        Raises:
            RuntimeError: If function not declared
        """
        return self._builder._builder.get_global_var(name)

    def add_function(self, func: ir.Function) -> None:
        """Add a function to the program.

        The function name must match a previously declared function name.

        Args:
            func: Function to add
        """
        self._builder._builder.add_function(func)

    def get_result(self) -> ir.Program:
        """Get the built Program.

        Returns:
            Program: The completed program IR node

        Raises:
            AssertionError: If called before program is complete
        """
        assert self._result is not None, "Program not yet complete"
        return self._result


__all__ = ["IRBuilder", "FunctionBuilder", "ForLoopBuilder", "IfStmtBuilder", "ProgramBuilder"]
