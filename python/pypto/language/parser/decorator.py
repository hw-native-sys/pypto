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
import linecache
import sys
import textwrap
from collections.abc import Callable
from typing import TypeVar

from pypto.pypto_core import ir

from .ast_parser import ASTParser
from .diagnostics import ParserError, ParserSyntaxError


def _calculate_col_offset(source_lines: list[str]) -> int:
    """Calculate the column offset (indentation) of the first non-empty line.

    This is needed because ast.parse() requires code starting at column 0,
    but we need to report errors at the correct column in the original file.

    Args:
        source_lines: List of source code lines

    Returns:
        Column offset (number of leading spaces/tabs in first non-empty line)
    """
    for line in source_lines:
        if line.strip():  # Skip empty lines
            return len(line) - len(line.lstrip())
    return 0


def _parse_ast_tree(source_code: str, entity_type: str) -> ast.AST:
    """Parse source code into an AST tree with proper error handling.

    Args:
        source_code: Python source code to parse
        entity_type: Type of entity being parsed ("function" or "class") for error messages

    Returns:
        Parsed AST tree

    Raises:
        ParserSyntaxError: If the source code has syntax errors
    """
    try:
        return ast.parse(source_code)
    except SyntaxError as e:
        raise ParserSyntaxError(
            f"Failed to parse {entity_type} source: {e.msg}",
            hint=f"Check for Python syntax errors in your {entity_type}",
        )


TypeASTNode = TypeVar("TypeASTNode", bound=ast.FunctionDef | ast.ClassDef)


def _find_ast_node(tree: ast.AST, node_type: type[TypeASTNode], name: str, entity_type: str) -> TypeASTNode:
    """Find a specific AST node by type and name.

    Args:
        tree: AST tree to search
        node_type: Type of AST node to find (ast.FunctionDef or ast.ClassDef)
        name: Name of the node to find
        entity_type: Type of entity for error messages ("function" or "class")

    Returns:
        Found AST node

    Raises:
        ParserSyntaxError: If the node cannot be found
    """
    for node in ast.walk(tree):
        if isinstance(node, node_type) and node.name == name:
            return node

    raise ParserSyntaxError(
        f"Could not find {entity_type} definition for {name}",
        hint=f"Ensure the {entity_type} is properly defined",
    )


def _attach_source_lines_to_error(error: ParserError, source_file: str, source_lines_raw: list[str]) -> None:
    """Attach source lines to a ParserError if not already present.

    Args:
        error: ParserError to attach source lines to
        source_file: Path to the source file
        source_lines_raw: Raw source lines as fallback
    """
    if error.source_lines is None:
        try:
            with open(source_file, encoding="utf-8") as f:
                error.source_lines = f.read().split("\n")
        except Exception:
            # Fallback to the raw source lines if we can't read the file
            error.source_lines = source_lines_raw


def _has_pl_function_decorator(node: ast.FunctionDef) -> bool:
    """Check if a function node has @pl.function decorator.

    Args:
        node: AST FunctionDef node to check

    Returns:
        True if the node has @pl.function decorator
    """
    for decorator in node.decorator_list:
        # Check various decorator patterns
        # ast.Attribute: pl.function
        if isinstance(decorator, ast.Attribute):
            if decorator.attr == "function":
                return True
        # ast.Name: function (if imported directly)
        elif isinstance(decorator, ast.Name):
            if decorator.id == "function":
                return True
        # ast.Call: @pl.function() with parentheses
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "function":
                return True
            elif isinstance(decorator.func, ast.Name) and decorator.func.id == "function":
                return True
    return False


def _extract_function_type_from_decorator(node: ast.FunctionDef) -> ir.FunctionType:
    """Extract function type from @pl.function(type=...) decorator.

    Searches through the function's decorators to find @pl.function(type=...)
    and extracts the FunctionType value. If no type parameter is found,
    returns FunctionType.Opaque as the default.

    Args:
        node: AST FunctionDef node to extract function type from

    Returns:
        FunctionType extracted from decorator, or FunctionType.Opaque if not specified
    """
    for decorator in node.decorator_list:
        # Look for @pl.function(...) or @function(...) with call syntax
        if isinstance(decorator, ast.Call):
            # Check if it's a pl.function or function call
            is_function_decorator = False
            if isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "function":
                is_function_decorator = True
            elif isinstance(decorator.func, ast.Name) and decorator.func.id == "function":
                is_function_decorator = True

            if is_function_decorator:
                # Look for type= keyword argument
                for keyword in decorator.keywords:
                    if keyword.arg == "type":
                        # The value should be an attribute like pl.FunctionType.Orchestration
                        # or FunctionType.Orchestration
                        value = keyword.value
                        if isinstance(value, ast.Attribute):
                            # Extract the enum name (e.g., "Orchestration", "InCore", "Opaque")
                            type_name = value.attr
                            if type_name == "Orchestration":
                                return ir.FunctionType.Orchestration
                            elif type_name == "InCore":
                                return ir.FunctionType.InCore
                            elif type_name == "Opaque":
                                return ir.FunctionType.Opaque

    # Default to Opaque if no type specified
    return ir.FunctionType.Opaque


def _is_class_method(func: Callable) -> bool:
    """Check if a function is a method inside a class (not a standalone function).

    This performs strict validation to determine if a function with 'self' as the first
    parameter is actually defined inside a class, rather than a standalone function that
    just happens to have 'self' as a parameter name.

    Args:
        func: Function to check

    Returns:
        True if the function is a method inside a class
    """
    # Check if first parameter is 'self'
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        if not (params and params[0] == "self"):
            return False
    except (ValueError, TypeError):
        return False

    # Check if __qualname__ indicates this is a method (contains a dot)
    qualname = func.__qualname__
    if "." not in qualname:
        # No dot in qualname means it's a standalone function, not a method
        return False

    # Verify it has indentation (defined inside a class, not at module level)
    try:
        source_lines_raw, _ = inspect.getsourcelines(func)
        col_offset = _calculate_col_offset(source_lines_raw)
        if col_offset > 0:
            # This is an indented method inside a class
            return True
    except (OSError, TypeError):
        # If we can't get source lines, assume it's a method based on qualname
        # (This can happen with dynamically generated code)
        return True

    return False


def _get_source_file(entity: Callable | type) -> str:
    """Get source filename for an entity, with fallback to code object attributes.

    Args:
        entity: Function or class to get source file for

    Returns:
        Source filename string
    """
    try:
        return inspect.getfile(entity)
    except (OSError, TypeError):
        pass

    # Fallback: extract from code object
    if callable(entity) and hasattr(entity, "__code__"):
        return entity.__code__.co_filename

    # For classes, find a method with a code object
    if isinstance(entity, type):
        for attr in entity.__dict__.values():
            if callable(attr) and hasattr(attr, "__code__"):
                return attr.__code__.co_filename

    return "<unknown>"


def _find_entity_in_source(all_lines: list[str], name: str, entity_type: str) -> tuple[list[str], int] | None:
    """Find an entity definition in source lines using AST parsing.

    Args:
        all_lines: All source lines from the file
        name: Name of the entity to find
        entity_type: "function" or "class"

    Returns:
        Tuple of (source_lines, starting_line_1based) or None if not found
    """
    source_text = "".join(all_lines)
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None

    node_type = ast.FunctionDef if entity_type == "function" else ast.ClassDef
    for node in ast.walk(tree):
        if isinstance(node, node_type) and node.name == name:
            # Start from the first decorator line if present
            start_line = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            end_line = node.end_lineno or node.lineno
            # Lines are 1-based in AST
            source_lines = all_lines[start_line - 1 : end_line]
            return source_lines, start_line

    return None


def _get_source_info(entity: Callable | type, entity_type: str) -> tuple[str, list[str], int]:
    """Get source file, source lines, and starting line for an entity.

    Tries multiple strategies:
    1. Standard inspect.getsourcelines()
    2. linecache fallback (handles IPython, pre-populated cache)
    3. sys.orig_argv for `python -c` invocations
    4. Clear error with actionable hint

    Args:
        entity: Function or class to get source for
        entity_type: "function" or "class"

    Returns:
        Tuple of (source_file, source_lines_raw, starting_line)

    Raises:
        ParserSyntaxError: If source cannot be retrieved by any strategy
    """
    name = entity.__name__ if hasattr(entity, "__name__") else str(entity)

    # Strategy 1: Standard inspect
    try:
        source_file = inspect.getfile(entity)
        source_lines_raw, starting_line = inspect.getsourcelines(entity)
        return source_file, source_lines_raw, starting_line
    except (OSError, TypeError):
        pass

    # Get source file via fallback for strategies 2-3
    source_file = _get_source_file(entity)

    # Strategy 2: linecache fallback
    all_lines = linecache.getlines(source_file)
    if all_lines:
        result = _find_entity_in_source(all_lines, name, entity_type)
        if result is not None:
            return source_file, result[0], result[1]

    # Strategy 3: sys.orig_argv for `python -c`
    if source_file == "<string>" and hasattr(sys, "orig_argv"):
        orig_argv = sys.orig_argv
        try:
            c_index = orig_argv.index("-c")
            if c_index + 1 < len(orig_argv):
                code_str = orig_argv[c_index + 1]
                code_lines = code_str.splitlines(keepends=True)
                # Populate linecache so future lookups also work
                linecache.cache["<string>"] = (
                    len(code_str),
                    None,
                    code_lines,
                    "<string>",
                )
                result = _find_entity_in_source(code_lines, name, entity_type)
                if result is not None:
                    return source_file, result[0], result[1]
        except ValueError:
            pass

    # Strategy 4: Clear error
    raise ParserSyntaxError(
        f"Cannot retrieve source code for {entity_type} '{name}'",
        hint="Save your code to a .py file, or use pl.parse() / pl.parse_program() to parse from a string",
    )


def function(
    func: Callable | None = None,
    *,
    type: ir.FunctionType = ir.FunctionType.Opaque,
    strict_ssa: bool = False,
) -> ir.Function:
    """Decorator that parses a DSL function and returns IR Function.

    This decorator analyzes the decorated function's AST, parses the DSL
    constructs (type annotations, pl.range, pl.yield_, etc.), and builds
    an IR Function object.

    Args:
        func: Python function decorated with @pl.function
        type: Function type (Opaque, Orchestration, or InCore)
        strict_ssa: If True, enforce SSA (single assignment per variable).
                   If False (default), allow variable reassignment (non-SSA mode).

    Returns:
        IR Function object (or decorator if used with parameters)

    Example:
        >>> @pl.function
        ... def my_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
        ...     result = pl.create_tensor([64, 128], dtype=pl.FP32)
        ...     return result
        >>> @pl.function(type=pl.FunctionType.Orchestration)
        ... def orchestrator():
        ...     pass
    """

    # Capture the caller's scope for variable resolution in type annotations
    caller_frame = sys._getframe(1)
    closure_vars = {**caller_frame.f_globals, **caller_frame.f_locals}

    def _decorator(f: Callable) -> ir.Function:
        # Check if this is a method inside a class decorated with @pl.program
        # If so, return the original function - it will be parsed by @pl.program decorator
        if _is_class_method(f):
            # Don't parse now - let @pl.program handle it with proper global_vars context
            return f  # type: ignore[return-value]

        # Get source code and file information
        source_file, source_lines_raw, starting_line = _get_source_info(f, "function")
        source_code = "".join(source_lines_raw)

        # Calculate indentation offset before dedenting
        col_offset = _calculate_col_offset(source_lines_raw)

        # Remove leading indentation so ast.parse() can parse it
        source_code = textwrap.dedent(source_code)

        # Use dedented source lines so column offsets align with AST
        source_lines = source_code.split("\n")

        # Calculate line offset (AST line numbers are 1-based, but we want to map to original file)
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
                closure_vars=closure_vars,
            )

            try:
                ir_func = parser.parse_function(func_def, func_type=type)
            except ParserError:
                # Re-raise ParserError as-is, it already has source lines
                raise
            except Exception as e:
                # Wrap unexpected exceptions as ParserError
                raise ParserSyntaxError(
                    f"Failed to parse function '{f.__name__}': {e}",
                    hint="Check your function definition for errors",
                ) from e

            return ir_func

        except ParserError as e:
            # Attach source lines if not already present
            _attach_source_lines_to_error(e, source_file, source_lines_raw)
            # Always raise the exception - let the excepthook handle uncaught cases
            raise

    # Support both @pl.function and @pl.function(type=...)
    if func is None:
        # Called with parameters: @pl.function(type=...)
        return _decorator  # type: ignore[return-value]
    else:
        # Called without parameters: @pl.function
        return _decorator(func)


def program(cls: type | None = None, *, strict_ssa: bool = False) -> ir.Program:
    """Decorator that parses a class with @pl.function methods into a Program.

    The class should contain one or more methods decorated with @pl.function.
    Each method is parsed as a separate function and added to the program.
    Methods must have 'self' as the first parameter (standard Python syntax),
    which is automatically stripped from the IR.

    Args:
        cls: Class with @pl.function decorated methods
        strict_ssa: If True, enforce SSA (single assignment per variable).
                   If False (default), allow variable reassignment (non-SSA mode).

    Returns:
        IR Program object (or decorator if used with parameters)

    Example:
        >>> @pl.program
        ... class MyProgram:
        ...     @pl.function
        ...     def add(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        ...         result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
        ...         return result
        ...
        ...     @pl.function
        ...     def mul(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        ...         result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
        ...         return result
        >>> # MyProgram is now an ir.Program object
    """

    # Capture the caller's scope for variable resolution in type annotations
    caller_frame = sys._getframe(1)
    closure_vars = {**caller_frame.f_globals, **caller_frame.f_locals}

    def _decorator(c: type) -> ir.Program:
        # Get source code and file information
        source_file, source_lines_raw, starting_line = _get_source_info(c, "class")
        source_code = "".join(source_lines_raw)

        # Calculate indentation offset before dedenting
        col_offset = _calculate_col_offset(source_lines_raw)

        # Remove leading indentation so ast.parse() can parse it
        source_code = textwrap.dedent(source_code)

        # Use dedented source lines so column offsets align with AST
        source_lines = source_code.split("\n")

        # Calculate line offset (AST line numbers are 1-based, but we want to map to original file)
        line_offset = starting_line - 1

        try:
            tree = _parse_ast_tree(source_code, "class")
            class_def = _find_ast_node(tree, ast.ClassDef, c.__name__, "class")

            # Pass 1: Collect all @pl.function methods and create GlobalVars
            global_vars = {}
            func_defs = []

            for node in class_def.body:
                if isinstance(node, ast.FunctionDef):
                    if _has_pl_function_decorator(node):
                        # Create GlobalVar for this function
                        gvar = ir.GlobalVar(node.name)
                        global_vars[node.name] = gvar
                        func_defs.append(node)

            if not func_defs:
                raise ParserSyntaxError(
                    f"Class '{c.__name__}' contains no @pl.function decorated methods",
                    hint="Add at least one method decorated with @pl.function",
                )

            # Pass 2: Parse each function body with GlobalVar map for cross-function calls
            # Build a map from GlobalVar to parsed functions as we go, so later functions
            # can use return type information from earlier functions
            functions = []
            gvar_to_func = {}

            for func_def in func_defs:
                # Extract function type from decorator
                func_type = _extract_function_type_from_decorator(func_def)

                # Strip 'self' parameter if present (must be done before parsing)
                func_def_to_parse = func_def
                if func_def.args.args and func_def.args.args[0].arg == "self":
                    # Create a new arguments object with self removed
                    new_args = ast.arguments(
                        posonlyargs=func_def.args.posonlyargs,
                        args=func_def.args.args[1:],  # Skip 'self'
                        vararg=func_def.args.vararg,
                        kwonlyargs=func_def.args.kwonlyargs,
                        kw_defaults=func_def.args.kw_defaults,
                        kwarg=func_def.args.kwarg,
                        defaults=func_def.args.defaults,
                    )

                    # Create a new function def node with self removed
                    func_def_to_parse = ast.FunctionDef(
                        name=func_def.name,
                        args=new_args,
                        body=func_def.body,
                        decorator_list=func_def.decorator_list,
                        returns=func_def.returns,
                        type_comment=func_def.type_comment,
                        lineno=func_def.lineno,
                        col_offset=func_def.col_offset,
                    )
                    # Copy end line numbers if they exist
                    if hasattr(func_def, "end_lineno"):
                        func_def_to_parse.end_lineno = func_def.end_lineno
                    if hasattr(func_def, "end_col_offset"):
                        func_def_to_parse.end_col_offset = func_def.end_col_offset

                # Create parser with global_vars and gvar_to_func map for cross-function call resolution
                parser = ASTParser(
                    source_file,
                    source_lines,
                    line_offset,
                    col_offset,
                    global_vars=global_vars,
                    gvar_to_func=gvar_to_func,
                    strict_ssa=strict_ssa,
                    closure_vars=closure_vars,
                )

                try:
                    ir_func = parser.parse_function(func_def_to_parse, func_type=func_type)
                except ParserError:
                    raise
                except SyntaxError as e:
                    raise ParserSyntaxError(
                        f"Failed to parse function '{func_def_to_parse.name}': {e.msg}",
                        hint="Check for Python syntax errors in your function definition",
                    ) from e

                functions.append(ir_func)
                # Update gvar_to_func map so subsequent functions can use this function's return type
                gvar = global_vars[ir_func.name]
                gvar_to_func[gvar] = ir_func

            # Create Program with class name and span
            program_span = ir.Span(source_file, starting_line, col_offset)
            prog = ir.Program(functions, c.__name__, program_span)

            return prog

        except ParserError as e:
            # Attach source lines if not already present
            _attach_source_lines_to_error(e, source_file, source_lines_raw)
            raise

    # Support both @pl.program and @pl.program(strict_ssa=...)
    if cls is None:
        # Called with parameters: @pl.program(strict_ssa=...)
        return _decorator  # type: ignore[return-value]
    else:
        # Called without parameters: @pl.program
        return _decorator(cls)


__all__ = ["function", "program"]
