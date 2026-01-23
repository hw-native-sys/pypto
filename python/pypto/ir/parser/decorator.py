# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Decorator for parsing DSL functions to IR."""

import ast
import inspect
import textwrap
from typing import Callable

from pypto.pypto_core import ir

from .ast_parser import ASTParser
from .diagnostics import ParserError, ParserSyntaxError


def function(func: Callable) -> ir.Function:
    """Decorator that parses a DSL function and returns IR Function.

    This decorator analyzes the decorated function's AST, parses the DSL
    constructs (type annotations, pl.range, pl.yield_, etc.), and builds
    an IR Function object.

    Args:
        func: Python function decorated with @pl.function

    Returns:
        IR Function object

    Example:
        >>> @pl.function
        ... def my_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
        ...     result = pl.op.tensor.create([64, 128], dtype=pl.FP32)
        ...     return result
    """

    # Get source code and file information
    source_file = inspect.getfile(func)

    # Get source lines and starting line number

    source_lines_raw, starting_line = inspect.getsourcelines(func)
    source_code = "".join(source_lines_raw)

    # Calculate indentation offset before dedenting
    # This is needed because ast.parse() requires code starting at column 0,
    # but we need to report errors at the correct column in the original file
    col_offset = 0
    for line in source_lines_raw:
        if line.strip():  # Skip empty lines
            col_offset = len(line) - len(line.lstrip())
            break

    # Remove leading indentation so ast.parse() can parse it
    source_code = textwrap.dedent(source_code)

    # Use dedented source lines so column offsets align with AST
    source_lines = source_code.split("\n")

    # Calculate line offset (AST line numbers are 1-based, but we want to map to original file)
    line_offset = starting_line - 1

    try:
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            # Convert Python syntax error to ParserSyntaxError
            raise ParserSyntaxError(
                f"Failed to parse function source: {e.msg}",
                hint="Check for Python syntax errors in your function",
            )

        # Find the function definition in the AST
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
                func_def = node
                break

        if func_def is None:
            raise ParserSyntaxError(
                f"Could not find function definition for {func.__name__}",
                hint="Ensure the function is properly defined",
            )

        # Create parser and parse the function
        parser = ASTParser(source_file, source_lines, line_offset, col_offset)

        try:
            ir_func = parser.parse_function(func_def)
        except ParserError:
            # Re-raise ParserError as-is, it already has source lines
            raise
        except Exception as e:
            # Wrap unexpected exceptions as ParserError
            raise ParserSyntaxError(
                f"Failed to parse function '{func.__name__}': {e}",
                hint="Check your function definition for errors",
            ) from e

        return ir_func

    except ParserError as e:
        # Attach source lines if not already present
        # Use the full file content for proper line number display
        if e.source_lines is None:
            try:
                with open(source_file, encoding="utf-8") as f:
                    e.source_lines = f.read().split("\n")
            except Exception:
                # Fallback to the function source lines if we can't read the file
                e.source_lines = source_lines_raw

        # Clean up the parser to release C++ objects before raising
        # This prevents memory leaks when exceptions are caught
        if "parser" in locals():
            del parser

        # Always raise the exception - let the excepthook handle uncaught cases
        raise


__all__ = ["function"]
