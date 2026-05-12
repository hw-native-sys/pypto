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
import dataclasses
import inspect
import linecache
import sys
import textwrap
from collections.abc import Callable
from typing import Any, TypeAlias, TypeVar, cast, overload

from pypto.compile_profiling import CompileProfiler, get_active_profiler
from pypto.pypto_core import ir

from .ast_parser import ASTParser
from .comment_extractor import extract_line_comments
from .diagnostics import ParserError, ParserSyntaxError, concise_error_message
from .enum_utils import FUNCTION_TYPE_MAP, LEVEL_MAP, ROLE_MAP, SPLIT_MODE_MAP, extract_enum_value


def _capture_subworker_source(func_def: ast.FunctionDef) -> str:
    """Return the SubWorker function *body* as plain Python source text.

    Body statements are unparsed (no ``def`` header, no decorators) so the IR
    ``Function`` keeps the signature and the attached ``InlineStmt`` carries
    only the body.

    Raises:
        ParserSyntaxError: if the SubWorker declares a ``self`` parameter —
            SubWorkers must be self-contained inside ``@pl.program``.
    """
    if func_def.args.args and func_def.args.args[0].arg == "self":
        raise ParserSyntaxError(
            f"SubWorker function '{func_def.name}' must not declare 'self'; "
            "declare it as a self-contained function inside @pl.program.",
            hint="Remove the 'self' parameter from the SubWorker definition.",
        )
    return "\n".join(ast.unparse(stmt) for stmt in func_def.body)


@dataclasses.dataclass
class InlineFunction:
    """Stores AST and metadata for a function to be inlined at call sites."""

    name: str
    func_def: ast.FunctionDef
    param_names: list[str]
    source_file: str
    source_lines: list[str]
    line_offset: int
    col_offset: int
    closure_vars: dict[str, Any]


def _strip_self_parameter(func_def: ast.FunctionDef) -> ast.FunctionDef:
    """Return a copy of func_def with the leading 'self' parameter removed.

    If the first parameter is not 'self', returns the original node unchanged.

    Args:
        func_def: AST FunctionDef node

    Returns:
        FunctionDef with 'self' stripped, or the original if no 'self' found
    """
    if not func_def.args.args or func_def.args.args[0].arg != "self":
        return func_def

    new_args = ast.arguments(
        posonlyargs=func_def.args.posonlyargs,
        args=func_def.args.args[1:],
        vararg=func_def.args.vararg,
        kwonlyargs=func_def.args.kwonlyargs,
        kw_defaults=func_def.args.kw_defaults,
        kwarg=func_def.args.kwarg,
        defaults=func_def.args.defaults,
    )
    new_func_def = ast.FunctionDef(
        name=func_def.name,
        args=new_args,
        body=func_def.body,
        decorator_list=func_def.decorator_list,
        returns=func_def.returns,
        type_comment=func_def.type_comment,
        lineno=func_def.lineno,
        col_offset=func_def.col_offset,
    )
    if hasattr(func_def, "end_lineno"):
        new_func_def.end_lineno = func_def.end_lineno
    if hasattr(func_def, "end_col_offset"):
        new_func_def.end_col_offset = func_def.end_col_offset
    return new_func_def


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
FunctionDecorator: TypeAlias = Callable[[Callable[..., Any]], ir.Function]
ProgramDecorator: TypeAlias = Callable[[type], ir.Program]


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
        # Use the span's filename if it differs (e.g., error in an inline function)
        target_file = source_file
        if error.span and isinstance(error.span, dict):
            span_file = error.span.get("filename")
            if span_file and span_file != source_file:
                target_file = span_file
        try:
            with open(target_file, encoding="utf-8") as f:
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


def _find_function_decorator_call(node: ast.FunctionDef) -> ast.Call | None:
    """Find the @pl.function(...) Call decorator on a FunctionDef, if present.

    Args:
        node: AST FunctionDef node to search

    Returns:
        The ast.Call node for the @pl.function(...) decorator, or None
    """
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if (isinstance(func, ast.Attribute) and func.attr == "function") or (
            isinstance(func, ast.Name) and func.id == "function"
        ):
            return decorator
    return None


def _extract_function_type_from_decorator(node: ast.FunctionDef) -> ir.FunctionType:
    """Extract function type from @pl.function(type=...) decorator.

    Args:
        node: AST FunctionDef node to extract function type from

    Returns:
        FunctionType extracted from decorator, or FunctionType.Opaque if not specified
    """
    decorator = _find_function_decorator_call(node)
    if decorator is None:
        return ir.FunctionType.Opaque

    for keyword in decorator.keywords:
        if keyword.arg is None:
            raise ParserSyntaxError(
                "Unsupported `@pl.function(**kwargs)` in `@pl.program`",
                hint="Use a literal type=pl.FunctionType.<name>.",
            )
        if keyword.arg != "type":
            continue

        value = keyword.value
        if not isinstance(value, ast.Attribute):
            raise ParserSyntaxError(
                "Unsupported `@pl.function`(type=...) value",
                hint="Use pl.FunctionType.<name>.",
            )
        is_function_type_attr = (isinstance(value.value, ast.Name) and value.value.id == "FunctionType") or (
            isinstance(value.value, ast.Attribute)
            and isinstance(value.value.value, ast.Name)
            and value.value.value.id == "pl"
            and value.value.attr == "FunctionType"
        )
        if not is_function_type_attr or value.attr not in FUNCTION_TYPE_MAP:
            raise ParserSyntaxError(
                "Unsupported `@pl.function`(type=...) value",
                hint="Use pl.FunctionType.<name>.",
            )
        return FUNCTION_TYPE_MAP[value.attr]

    return ir.FunctionType.Opaque


def _extract_function_level_role_from_decorator(
    node: ast.FunctionDef,
) -> tuple[ir.Level | None, ir.Role | None]:
    """Extract level and role from @pl.function(level=..., role=...) decorator.

    Args:
        node: AST FunctionDef node to extract level/role from

    Returns:
        Tuple of (level, role), either or both may be None
    """
    decorator = _find_function_decorator_call(node)
    if decorator is None:
        return None, None

    level = None
    role = None
    for keyword in decorator.keywords:
        if keyword.arg == "level":
            if isinstance(keyword.value, ast.Constant) and keyword.value.value is None:
                level = None
            else:
                level = extract_enum_value(keyword.value, LEVEL_MAP, "Level", "pl.Level")
        elif keyword.arg == "role":
            if isinstance(keyword.value, ast.Constant) and keyword.value.value is None:
                role = None
            else:
                role = extract_enum_value(keyword.value, ROLE_MAP, "Role", "pl.Role")
    return level, role


def _normalize_attrs(attrs: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize function attrs: convert SplitMode enums to int values for C++ storage.

    SplitMode.NONE entries are dropped (equivalent to no split).
    Returns None if the result is empty.
    """
    if not attrs:
        return None
    result: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, ir.SplitMode):
            if value != ir.SplitMode.NONE:
                result[key] = value.value
        else:
            result[key] = value
    return result or None


def _extract_function_attrs_from_decorator(node: ast.FunctionDef) -> dict[str, Any]:
    """Extract function attrs from @pl.function(attrs={...}) decorator.

    Supports attrs={"split": pl.SplitMode.UP_DOWN, ...} syntax.
    Returns a normalized dict with enum values converted to ints.
    """
    decorator = _find_function_decorator_call(node)
    if decorator is None:
        return {}

    for keyword in decorator.keywords:
        if keyword.arg != "attrs":
            continue
        if isinstance(keyword.value, ast.Constant) and keyword.value.value is None:
            return {}
        if not isinstance(keyword.value, ast.Dict):
            raise ParserSyntaxError(
                "Unsupported `@pl.function(attrs=...)` value",
                hint='Use a dict literal, e.g. attrs={"split": pl.SplitMode.UP_DOWN}.',
            )
        attrs: dict[str, Any] = {}
        for k, v in zip(keyword.value.keys, keyword.value.values):
            if k is None:
                raise ParserSyntaxError(
                    "Unsupported `**` unpacking in `@pl.function(attrs={...})`",
                    hint="Use only string literal keys in the attrs dict.",
                )
            if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
                raise ParserSyntaxError(
                    f"Attrs dict key must be a string literal, got {ast.dump(k)}",
                    hint='Use string literal keys, e.g. attrs={"split": ...}.',
                )
            attr_key = k.value
            if attr_key == "split":
                split_mode = extract_enum_value(v, SPLIT_MODE_MAP, "SplitMode", "pl.SplitMode")
                if split_mode != ir.SplitMode.NONE:
                    attrs["split"] = split_mode.value
            elif isinstance(v, ast.Constant):
                attrs[attr_key] = v.value
        return attrs
    return {}


def _prescan_reserve_buffers(
    func_def: ast.FunctionDef, buffer_name_meta: dict[tuple[str, str], dict[str, Any]]
) -> None:
    """Pre-scan a function body for pl.reserve_buffer calls and register their metadata.

    This enables import_peer_buffer to resolve .base from a peer function's reserve_buffer
    regardless of function definition order within a @pl.program class.
    """
    for node in ast.walk(func_def):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "reserve_buffer":
            continue
        meta: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is not None and isinstance(kw.value, ast.Constant):
                meta[kw.arg] = kw.value.value
        buf_name = meta.get("name")
        if buf_name is not None:
            buffer_name_meta[(func_def.name, buf_name)] = meta


def _is_class_method(func: Callable) -> bool:
    """Check if a function is a method inside a class (not a standalone function).

    Recognizes both classic methods (first parameter ``self``) and self-contained
    functions defined inside a class (no ``self`` parameter, e.g. SubWorker
    functions declared inside ``@pl.program``). The check uses ``__qualname__``
    plus indentation so it does not misclassify standalone functions whose first
    parameter happens to be named ``self``.

    Args:
        func: Function to check

    Returns:
        True if the function is defined inside a class
    """
    qualname = getattr(func, "__qualname__", "")
    if "." not in qualname:
        return False

    # Nested-function qualnames look like "outer.<locals>.inner" — those are not
    # class methods.
    parent = qualname.rsplit(".", 1)[0]
    if parent.endswith("<locals>"):
        return False

    # Verify it has indentation (defined inside a class, not at module level)
    try:
        source_lines_raw, _ = inspect.getsourcelines(func)
        col_offset = _calculate_col_offset(source_lines_raw)
        if col_offset > 0:
            return True
    except (OSError, TypeError):
        # If we can't get source lines, trust qualname.
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


def _find_entity_in_source(
    all_lines: list[str], name: str, entity_type: str, start_line_hint: int | None = None
) -> tuple[list[str], int] | None:
    """Find an entity definition in source lines using AST parsing.

    Args:
        all_lines: All source lines from the file
        name: Name of the entity to find
        entity_type: "function" or "class"
        start_line_hint: Optional line number to disambiguate entities with the same name

    Returns:
        Tuple of (source_lines, starting_line_1based) or None if not found
    """
    source_text = "".join(all_lines)
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None

    node_type = ast.FunctionDef if entity_type == "function" else ast.ClassDef
    candidates = [node for node in ast.walk(tree) if isinstance(node, node_type) and node.name == name]

    if not candidates:
        return None

    if len(candidates) == 1:
        node = candidates[0]
    elif start_line_hint is not None:
        # Disambiguate using the code object's line number
        node = min(candidates, key=lambda n: abs(n.lineno - start_line_hint))
    else:
        node = candidates[0]

    # Start from the first decorator line if present
    start_line = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    end_line = node.end_lineno or node.lineno
    # Lines are 1-based in AST
    source_lines = all_lines[start_line - 1 : end_line]
    return source_lines, start_line


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

    # Get a line number hint from the code object to disambiguate same-name entities
    start_line_hint: int | None = None
    if callable(entity) and hasattr(entity, "__code__"):
        start_line_hint = entity.__code__.co_firstlineno

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
        result = _find_entity_in_source(all_lines, name, entity_type, start_line_hint)
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
                # Temporarily populate linecache for the lookup, preserving any existing entry
                prev_entry = linecache.cache.get("<string>")
                linecache.cache["<string>"] = (
                    len(code_str),
                    None,
                    code_lines,
                    "<string>",
                )
                try:
                    result = _find_entity_in_source(code_lines, name, entity_type, start_line_hint)
                    if result is not None:
                        return source_file, result[0], result[1]
                finally:
                    if prev_entry is not None:
                        linecache.cache["<string>"] = prev_entry
                    else:
                        linecache.cache.pop("<string>", None)
        except ValueError:
            pass

    # Strategy 4: Clear error
    raise ParserSyntaxError(
        f"Cannot retrieve source code for {entity_type} '{name}'",
        hint="Save your code to a .py file, or use pl.parse() / pl.parse_program() to parse from a string",
    )


@overload
def function(
    func: Callable[..., Any],
    *,
    type: ir.FunctionType = ir.FunctionType.Opaque,
    level: ir.Level | None = None,
    role: ir.Role | None = None,
    attrs: dict[str, Any] | None = None,
    strict_ssa: bool = False,
) -> ir.Function: ...


@overload
def function(
    func: None = None,
    *,
    type: ir.FunctionType = ir.FunctionType.Opaque,
    level: ir.Level | None = None,
    role: ir.Role | None = None,
    attrs: dict[str, Any] | None = None,
    strict_ssa: bool = False,
) -> FunctionDecorator: ...


def function(
    func: Callable[..., Any] | None = None,
    *,
    type: ir.FunctionType = ir.FunctionType.Opaque,
    level: ir.Level | None = None,
    role: ir.Role | None = None,
    attrs: dict[str, Any] | None = None,
    strict_ssa: bool = False,
) -> ir.Function | FunctionDecorator:
    """Decorator that parses a DSL function and returns IR Function.

    This decorator analyzes the decorated function's AST, parses the DSL
    constructs (type annotations, pl.range, pl.yield_, etc.), and builds
    an IR Function object.

    Args:
        func: Python function decorated with @pl.function
        type: Function type (Opaque, Orchestration, or InCore)
        level: Hierarchy level (e.g. pl.Level.HOST)
        role: Function role (e.g. pl.Role.SubWorker)
        attrs: Function-level attributes dict (e.g. {"split": pl.SplitMode.UP_DOWN})
        strict_ssa: If True, enforce SSA (single assignment per variable).
                   If False (default), allow variable reassignment (non-SSA mode).

    Returns:
        IR Function object (or decorator if used with parameters)

    Example:
        >>> @pl.function
        ... def my_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
        ...     result = pl.create_tensor([64, 128], dtype=pl.FP32)
        ...     return result
        >>> @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
        ... def sub_worker(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        ...     return x
    """

    # Capture the caller's scope for variable resolution in type annotations
    caller_frame = sys._getframe(1)
    closure_vars = {**caller_frame.f_globals, **caller_frame.f_locals}

    def _decorator(f: Callable[..., Any]) -> ir.Function | Callable[..., Any]:
        # Check if this is a method inside a class decorated with @pl.program
        # If so, return the original function - it will be parsed by @pl.program decorator
        if _is_class_method(f):
            # Don't parse now - let @pl.program handle it with proper global_vars context
            return f

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
                pending_comments=extract_line_comments(source_code),
            )

            # Normalize attrs: convert enum values to ints for storage
            func_attrs = _normalize_attrs(attrs) if attrs else None

            try:
                ir_func = parser.parse_function(
                    func_def,
                    func_type=type,
                    func_level=level,
                    func_role=role,
                    func_attrs=func_attrs,
                )
            except ParserError:
                # Re-raise ParserError as-is, it already has source lines
                raise
            except Exception as e:
                # Wrap unexpected exceptions as ParserError
                raise ParserSyntaxError(
                    f"Failed to parse function '{f.__name__}': {concise_error_message(e)}",
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
        return cast(FunctionDecorator, _decorator)
    else:
        # Called without parameters: @pl.function
        return cast(ir.Function, _decorator(func))


def inline(func: Callable) -> InlineFunction:
    """Decorator that captures a function for inlining at call sites.

    Unlike @pl.function which parses to an ir.Function immediately,
    @pl.inline defers parsing until the function is called within a
    @pl.program. The body is expanded in-place at each call site.

    Args:
        func: Python function to capture for inlining

    Returns:
        InlineFunction object with captured AST and metadata

    Example:
        >>> @pl.inline
        ... def normalize(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        ...     result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
        ...     return result
    """
    caller_frame = sys._getframe(1)
    closure_vars = {**caller_frame.f_globals, **caller_frame.f_locals}

    source_file, source_lines_raw, starting_line = _get_source_info(func, "function")
    source_code = textwrap.dedent("".join(source_lines_raw))
    col_offset = _calculate_col_offset(source_lines_raw)
    source_lines = source_code.split("\n")
    line_offset = starting_line - 1

    tree = _parse_ast_tree(source_code, "function")
    func_def = _find_ast_node(tree, ast.FunctionDef, func.__name__, "function")

    if _is_class_method(func):
        func_def = _strip_self_parameter(func_def)

    param_names = [arg.arg for arg in func_def.args.args]

    return InlineFunction(
        name=func.__name__,
        func_def=func_def,
        param_names=param_names,
        source_file=source_file,
        source_lines=source_lines,
        line_offset=line_offset,
        col_offset=col_offset,
        closure_vars=closure_vars,
    )


@overload
def program(cls: type) -> ir.Program: ...


@overload
def program(cls: None = None, *, strict_ssa: bool = False) -> ProgramDecorator: ...


def program(cls: type | None = None, *, strict_ssa: bool = False) -> ir.Program | ProgramDecorator:
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
        prof = get_active_profiler()
        if prof is not None:
            return _parse_program_with_profiling(c, prof, strict_ssa, closure_vars)
        return _parse_program_body(c, strict_ssa, closure_vars)

    def _parse_program_with_profiling(
        c: type,
        prof: CompileProfiler,
        strict_ssa: bool,
        closure_vars: dict[str, Any],
    ) -> ir.Program:
        with prof.stage("parse"):
            return _parse_program_body(c, strict_ssa, closure_vars)

    def _parse_program_body(  # noqa: PLR0912
        c: type,
        strict_ssa: bool,
        closure_vars: dict[str, Any],
    ) -> ir.Program:
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
            pending_comments = extract_line_comments(source_code)

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
            external_functions: dict[str, ir.Function] = {}

            # Pre-scan: collect reserve_buffer metadata from all functions so that
            # import_peer_buffer can resolve .base across functions regardless of order.
            buffer_name_meta: dict[tuple[str, str], dict[str, Any]] = {}
            for func_def in func_defs:
                _prescan_reserve_buffers(func_def, buffer_name_meta)

            # Shared dyn_var_cache so all functions in this program share the same
            # ir.Var objects for dynamic dimension variables (issue #618).
            dyn_var_cache: dict[str, ir.Var] = {}

            # Shared set of pld.alloc_window_buffer names so name-uniqueness
            # checks span every function in the program.
            alloc_window_buffer_names: set[str] = set()

            # Compute per-method line-range boundaries. Each method owns comments from
            # its first-line up to the line just before the next method (or end of
            # class for the last one). This captures tail-of-block comments inside
            # the method body that AST end_lineno excludes, without letting earlier-
            # or later-method comments leak across.
            method_boundaries: dict[int, tuple[int, int]] = {}
            sorted_defs = sorted(func_defs, key=lambda d: d.lineno)
            for idx, fd in enumerate(sorted_defs):
                start = fd.lineno
                if idx + 1 < len(sorted_defs):
                    end = sorted_defs[idx + 1].lineno - 1
                else:
                    end = max((fd.end_lineno or fd.lineno), max(pending_comments, default=fd.lineno))
                method_boundaries[id(fd)] = (start, end)

            for func_def in func_defs:
                # Extract function type, level/role, and attrs from decorator
                func_type = _extract_function_type_from_decorator(func_def)
                func_level, func_role = _extract_function_level_role_from_decorator(func_def)
                func_attrs = _extract_function_attrs_from_decorator(func_def)

                # HOST SubWorkers carry their pure-Python body inline in the IR
                # via an InlineStmt — no DSL parsing, no implicit `self` stripping
                # (`_capture_subworker_source` rejects `self` with a clear error).
                is_sub_worker = (
                    func_level is not None
                    and ir.level_to_linqu_level(func_level) >= 3
                    and func_role == ir.Role.SubWorker
                )

                if is_sub_worker:
                    inline_body = _capture_subworker_source(func_def)
                    func_def_to_parse = func_def
                else:
                    inline_body = None
                    func_def_to_parse = _strip_self_parameter(func_def)

                method_start, method_end = method_boundaries[id(func_def)]
                method_comments = {
                    k: list(v) for k, v in pending_comments.items() if method_start <= k <= method_end
                }
                parser = ASTParser(
                    source_file,
                    source_lines,
                    line_offset,
                    col_offset,
                    global_vars=global_vars,
                    gvar_to_func=gvar_to_func,
                    strict_ssa=strict_ssa,
                    closure_vars=closure_vars,
                    buffer_name_meta=buffer_name_meta,
                    dyn_var_cache=dyn_var_cache,
                    pending_comments=method_comments,
                    alloc_window_buffer_names=alloc_window_buffer_names,
                )

                try:
                    ir_func = parser.parse_function(
                        func_def_to_parse,
                        func_type=func_type,
                        func_level=func_level,
                        func_role=func_role,
                        func_attrs=func_attrs or None,
                        inline_body=inline_body,
                    )
                except SyntaxError as e:
                    raise ParserSyntaxError(
                        f"Failed to parse function '{func_def_to_parse.name}': {e.msg}",
                        span=parser.span_tracker.get_span(func_def_to_parse),
                        hint="Check for Python syntax errors in your function definition",
                    ) from e
                except ParserError:
                    raise
                except Exception as e:
                    raise ParserSyntaxError(
                        f"Failed to parse function '{func_def_to_parse.name}': {concise_error_message(e)}",
                        span=parser.span_tracker.get_span(func_def_to_parse),
                        hint="Check your function definition for errors",
                    ) from e

                functions.append(ir_func)
                # Update gvar_to_func map so subsequent functions can use this function's return type
                gvar = global_vars[ir_func.name]
                gvar_to_func[gvar] = ir_func

                # Merge external functions discovered by the parser.
                # The parser already validates against global_vars within each method,
                # so here we only check for cross-method conflicts (different objects
                # with the same name used in different methods).
                for ext_name, ext_func in parser.external_funcs.items():
                    if ext_name in external_functions and external_functions[ext_name] is not ext_func:
                        raise ParserSyntaxError(
                            f"Conflicting external functions with name '{ext_name}'",
                            hint="External functions must have unique names; rename one of the functions",
                        )
                    external_functions[ext_name] = ext_func

            # Combine internal and external functions
            all_functions = functions + list(external_functions.values())

            # Create Program with class name and span
            program_span = ir.Span(source_file, starting_line, col_offset)
            prog = ir.Program(all_functions, c.__name__, program_span)

            return prog

        except ParserError as e:
            # Attach source lines if not already present
            _attach_source_lines_to_error(e, source_file, source_lines_raw)
            raise

    # Support both @pl.program and @pl.program(strict_ssa=...)
    if cls is None:
        # Called with parameters: @pl.program(strict_ssa=...)
        return cast(ProgramDecorator, _decorator)
    else:
        # Called without parameters: @pl.program
        return _decorator(cls)


__all__ = ["function", "inline", "program", "InlineFunction"]
