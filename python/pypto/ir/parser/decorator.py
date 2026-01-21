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
from typing import Callable

from pypto.pypto_core import ir

from .ast_parser import ASTParser


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
    source_lines = inspect.getsource(func).split("\n")
    source_file = inspect.getfile(func)

    # Parse source to AST
    source_code = inspect.getsource(func)

    # Remove any leading indentation to make it parseable
    import textwrap  # noqa: PLC0415

    source_code = textwrap.dedent(source_code)

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        raise ValueError(f"Failed to parse function source: {e}")

    # Find the function definition in the AST
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
            func_def = node
            break

    if func_def is None:
        raise ValueError(f"Could not find function definition for {func.__name__}")

    # Create parser and parse the function
    parser = ASTParser(source_file, source_lines)

    try:
        ir_func = parser.parse_function(func_def)
    except Exception as e:
        raise ValueError(f"Failed to parse function '{func.__name__}': {e}") from e

    return ir_func


__all__ = ["function"]
