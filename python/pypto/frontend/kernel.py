# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Kernel decorator for PyPTO Frontend.

The @kernel decorator provides a simplified API that wraps a single function
into a Program, combining the behavior of @pl.program and @pl.function.

This is the common case for kernel development: you write one function and
want a compilable Program out of it.

Usage:
    import pypto.frontend as fe

    @fe.kernel
    def vector_add(
        x: fe.Tensor[[1024], fe.FP32],
        y: fe.Tensor[[1024], fe.FP32],
    ) -> fe.Tensor[[1024], fe.FP32]:
        tile_x = fe.load(x, [0], [1024])
        tile_y = fe.load(y, [0], [1024])
        result = fe.add(tile_x, tile_y)
        out = fe.create([1024], dtype=fe.FP32)
        return fe.store(result, [0], [1024], out)

    # vector_add is now an ir.Program containing one function named "vector_add"
"""

import ast
import sys
import inspect
import textwrap
from typing import Callable, Optional

from pypto.pypto_core import ir
from pypto.language.parser.ast_parser import ASTParser
from pypto.language.parser.decorator import (
    _calculate_col_offset,
    _parse_ast_tree,
    _find_ast_node,
    _attach_source_lines_to_error,
    _extract_function_type_from_decorator,
)
from pypto.language.parser.diagnostics import ParserError, ParserSyntaxError


def _call_meta_and_capture_env(meta_fn):
    """Run meta_fn() and capture its local namespace (for types etc.). Returns (return_value, env dict)."""
    env = {}

    if meta_fn is None:
        return None, env
    def trace(frame, event, arg):
        if event == "return":
            env.clear()
            env.update(frame.f_locals)
        return trace

    old_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        result = meta_fn()
    finally:
        sys.settrace(old_trace)
    return result, env


def kernel(
    func: Optional[Callable] = None,
    meta_data=None,
    *,
    name: Optional[str] = None,
    type: ir.FunctionType = ir.FunctionType.Opaque,
    strict_ssa: bool = False,
) -> ir.Program:
    """Decorator that parses a single DSL function and wraps it in a Program.

    This is a convenience decorator that combines @pl.function and @pl.program
    for the common single-kernel use case. The decorated function is parsed into
    an ir.Function and then wrapped in an ir.Program.

    Args:
        func: Python function to parse
        name: Optional program name (defaults to function name)
        type: Function type (Opaque, Orchestration, or InCore)
        strict_ssa: If True, enforce SSA (single assignment per variable).

    Returns:
        ir.Program containing the single parsed function

    Example:
        >>> @fe.kernel
        ... def my_kernel(x: fe.Tensor[[64], fe.FP32]) -> fe.Tensor[[64], fe.FP32]:
        ...     tile = fe.load(x, [0], [64])
        ...     result = fe.add(tile, tile)
        ...     return fe.store(result, [0], [64], x)
        >>> assert isinstance(my_kernel, ir.Program)
        >>> assert my_kernel.get_function("my_kernel") is not None

        >>> @fe.kernel(name="my_program", type=fe.FunctionType.InCore)
        ... def compute(x: fe.Tensor[[64], fe.FP32]) -> fe.Tensor[[64], fe.FP32]:
        ...     return x
    """

    def _decorator(f: Callable) -> ir.Program:
        program_name = name if name is not None else f.__name__

        # Get source code and file information
        source_file = inspect.getfile(f)
        source_lines_raw, starting_line = inspect.getsourcelines(f)
        source_code = "".join(source_lines_raw)

        # Calculate indentation offset before dedenting
        col_offset = _calculate_col_offset(source_lines_raw)

        # Remove leading indentation so ast.parse() can parse it
        source_code = textwrap.dedent(source_code)
        source_lines = source_code.split("\n")

        # Calculate line offset
        line_offset = starting_line - 1

        try:
            tree = _parse_ast_tree(source_code, "function")
            func_def = _find_ast_node(tree, ast.FunctionDef, f.__name__, "function")

            # Create parser and parse the function
            parser = ASTParser(
                source_file,
                source_lines,
                line_offset,
                col_offset,
                strict_ssa=strict_ssa,
            )

            try:
                ir_func = parser.parse_function(func_def, func_type=type)
            except ParserError:
                raise
            except Exception as e:
                raise ParserSyntaxError(
                    f"Failed to parse kernel function '{f.__name__}': {e}",
                    hint="Check your function definition for errors",
                ) from e

            # Wrap the function in a Program
            program_span = ir.Span(source_file, starting_line, col_offset)
            prog = ir.Program([ir_func], program_name, program_span)
            return prog

        except ParserError as e:
            _attach_source_lines_to_error(e, source_file, source_lines_raw)
            raise

    # Support both @fe.kernel and @fe.kernel(name=..., type=...)
    if func is None:
        return _decorator  # type: ignore[return-value]
    else:
        return _decorator(func)


__all__ = ["kernel"]
