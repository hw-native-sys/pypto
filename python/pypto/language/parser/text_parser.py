# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parse DSL functions from text or files without requiring decorator syntax."""

import linecache
import sys
import types

from pypto.pypto_core import ir

from .diagnostics.exceptions import ParserError


def parse(code: str, filename: str = "<string>") -> ir.Function | ir.Program:
    """Parse a DSL function or program from a string.

    This function takes Python source code containing a @pl.function decorated
    function or @pl.program decorated class and parses it into an IR Function
    or Program object. The code is executed dynamically, automatically importing
    pypto.language as pl if not already present.

    Args:
        code: Python source code containing @pl.function or @pl.program
        filename: Optional filename for error reporting (default: "<string>")

    Returns:
        Parsed ir.Function or ir.Program object (auto-detected)

    Raises:
        ValueError: If the code contains nothing to parse or multiple items
        ParserError: If parsing fails (syntax errors, type errors, etc.)

    Warning:
        This function uses `exec()` to execute the provided code string.
        It should only be used with trusted input, as executing untrusted
        code can lead to arbitrary code execution vulnerabilities.

    Examples:
        >>> import pypto.language as pl

        >>> # Parse a function
        >>> func_code = '''
        ... @pl.function
        ... def add(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        ...     result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
        ...     return result
        ... '''
        >>> func = pl.parse(func_code)
        >>> print(func.name)
        add

        >>> # Parse a program
        >>> prog_code = '''
        ... @pl.program
        ... class MyProgram:
        ...     @pl.function
        ...     def add(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        ...         return pl.add(x, 1.0)
        ... '''
        >>> prog = pl.parse(prog_code)
        >>> print(prog.name)
        MyProgram
    """
    # Import pypto.language here to avoid circular imports
    import pypto.language as pl  # noqa: PLC0415

    # Make the source code available to inspect.getsourcelines() via linecache
    # Store ORIGINAL code (not modified) for accurate line numbers
    code_lines = code.splitlines(keepends=True)
    linecache.cache[filename] = (
        len(code),
        None,  # mtime
        code_lines,
        filename,
    )

    # Compile the code with the specified filename for proper error reporting
    try:
        compiled_code = compile(code, filename, "exec")
    except SyntaxError as e:
        raise SyntaxError(f"Failed to compile code from {filename}: {e}") from e

    # Create a temporary module for execution
    # This ensures inspect.getfile() works correctly for @pl.program
    module_name = f"__pypto_parse_{id(code)}__"
    temp_module = types.ModuleType(module_name)
    temp_module.__file__ = filename
    temp_module.__setattr__("pl", pl)

    # Add module to sys.modules so inspect can find it
    sys.modules[module_name] = temp_module

    # Execute the code in the module's namespace
    try:
        exec(compiled_code, temp_module.__dict__)
    except ParserError as e:
        # Re-raise ParserError as-is, it already has source lines
        raise e
    except Exception as e:
        # Re-raise with context about where the error occurred
        raise RuntimeError(f"Error executing code from {filename}: {e}") from e
    finally:
        # Clean up linecache entry
        if filename in linecache.cache:
            del linecache.cache[filename]
        # Clean up temporary module
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Get namespace from executed module
    namespace = temp_module.__dict__

    # Scan namespace for ir.Function and ir.Program instances
    functions = []
    programs = []
    for name, value in namespace.items():
        if isinstance(value, ir.Function):
            functions.append(value)
        elif isinstance(value, ir.Program):
            programs.append((name, value))

    # Determine what we found and return appropriate type
    total_items = len(functions) + len(programs)

    if total_items == 0:
        raise ValueError(
            f"No @pl.function or @pl.program found in {filename}. "
            "Make sure your code contains a function decorated with @pl.function "
            "or a class decorated with @pl.program."
        )
    elif total_items > 1:
        item_names = [f.name for f in functions] + [name for name, _ in programs]
        raise ValueError(
            f"Multiple functions/programs found in {filename}: {item_names}. "
            f"pl.parse() can only parse code containing a single function or program. "
            f"Consider using separate calls or parsing from separate files."
        )

    # Return the single item we found
    if functions:
        return functions[0]
    else:
        return programs[0][1]


def loads(filepath: str) -> ir.Function | ir.Program:
    """Load a DSL function or program from a file.

    This function reads a Python file containing a @pl.function decorated
    function or @pl.program decorated class and parses it into an IR Function
    or Program object (auto-detected).

    Args:
        filepath: Path to Python file containing @pl.function or @pl.program

    Returns:
        Parsed ir.Function or ir.Program object (auto-detected)

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file contains nothing to parse or multiple items
        ParserError: If parsing fails (syntax errors, type errors, etc.)

    Warning:
        This function reads a file and executes its contents. It should only
        be used with trusted files, as executing code from untrusted sources
        can lead to arbitrary code execution vulnerabilities.

    Examples:
        >>> import pypto.language as pl

        >>> # Load a function
        >>> func = pl.loads('my_kernel.py')
        >>> print(func.name)

        >>> # Load a program
        >>> prog = pl.loads('my_program.py')
        >>> print(prog.name)
    """
    # Read file content
    with open(filepath, encoding="utf-8") as f:
        code = f.read()

    # Parse using parse() with the filepath for proper error reporting
    return parse(code, filename=filepath)


def parse_program(code: str, filename: str = "<string>") -> ir.Program:
    """Parse a DSL program from a string.

    .. deprecated::
        Use :func:`parse` instead, which auto-detects functions and programs.

    This is now an alias for :func:`parse` that validates the result is a Program.

    Args:
        code: Python source code containing a @pl.program decorated class
        filename: Optional filename for error reporting (default: "<string>")

    Returns:
        Parsed ir.Program object

    Raises:
        ValueError: If the code contains a function instead of a program
        ParserError: If parsing fails (syntax errors, type errors, etc.)
    """
    result = parse(code, filename)
    if not isinstance(result, ir.Program):
        raise ValueError(
            f"Expected @pl.program but found @pl.function in {filename}. "
            f"Use pl.parse() for auto-detection or ensure your code contains @pl.program."
        )
    return result


def loads_program(filepath: str) -> ir.Program:
    """Load a DSL program from a file.

    .. deprecated::
        Use :func:`loads` instead, which auto-detects functions and programs.

    This is now an alias for :func:`loads` that validates the result is a Program.

    Args:
        filepath: Path to Python file containing @pl.program decorated class

    Returns:
        Parsed ir.Program object

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file contains a function instead of a program
        ParserError: If parsing fails (syntax errors, type errors, etc.)
    """
    result = loads(filepath)
    if not isinstance(result, ir.Program):
        raise ValueError(
            f"Expected @pl.program but found @pl.function in {filepath}. "
            f"Use pl.loads() for auto-detection or ensure your file contains @pl.program."
        )
    return result


__all__ = ["parse", "loads", "parse_program", "loads_program"]
