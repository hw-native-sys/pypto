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
import re
import sys
import types

from pypto.pypto_core import ir

from .diagnostics.exceptions import ParserError


def parse(code: str, filename: str = "<string>") -> ir.Function:
    """Parse a DSL function from a string.

    This function takes Python source code containing a @pl.function decorated
    function and parses it into an IR Function object. The code is executed
    dynamically, automatically importing pypto.language as pl if not already present.

    Args:
        code: Python source code containing a @pl.function decorated function
        filename: Optional filename for error reporting (default: "<string>")

    Returns:
        Parsed ir.Function object

    Raises:
        ValueError: If the code contains no functions or multiple functions
        ParserError: If parsing fails (syntax errors, type errors, etc.)

    Warning:
        This function uses `exec()` to execute the provided code string.
        It should only be used with trusted input, as executing untrusted
        code can lead to arbitrary code execution vulnerabilities.

    Example:
        >>> import pypto.language as pl
        >>> code = '''
        ... @pl.function
        ... def add(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        ...     result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)
        ...     return result
        ... '''
        >>> func = pl.parse(code)
        >>> print(func.name)
        add
    """
    # Import pypto.language here to avoid circular imports
    import pypto.language as pl  # noqa: PLC0415

    # Check if import statement is already present
    # Look for "import pypto.language as pl" or "from pypto import language as pl"
    has_import = bool(
        re.search(r"^\s*import\s+pypto\.language\s+as\s+pl\s*$", code, re.MULTILINE)
        or re.search(r"^\s*from\s+pypto\s+import\s+language\s+as\s+pl\s*$", code, re.MULTILINE)
    )

    # Prepend import if not present
    if not has_import:
        code = "import pypto.language as pl\n" + code

    # Make the source code available to inspect.getsourcelines() via linecache
    # This allows the decorator to work with dynamically executed code
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
        # Re-raise with context
        raise SyntaxError(f"Failed to compile code from {filename}: {e}") from e

    # Create execution namespace with pl available
    namespace = {
        "pl": pl,
        "__name__": "__main__",
        "__file__": filename,
    }

    # Execute the code
    try:
        exec(compiled_code, namespace)
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

    # Scan namespace for ir.Function instances
    functions = []
    for name, value in namespace.items():
        if isinstance(value, ir.Function):
            functions.append(value)

    # Validate we found exactly one function
    if len(functions) == 0:
        raise ValueError(
            f"No @pl.function decorated functions found in {filename}. "
            "Make sure your code contains a function decorated with @pl.function."
        )
    elif len(functions) > 1:
        func_names = [f.name for f in functions]
        raise ValueError(
            f"Multiple functions found in {filename}: {func_names}. "
            f"pl.parse() can only parse code containing a single function. "
            f"Consider using separate calls or parsing from separate files."
        )

    return functions[0]


def load(filepath: str) -> ir.Function:
    """Load a DSL function from a file.

    This function reads a Python file containing a @pl.function decorated
    function and parses it into an IR Function object.

    Args:
        filepath: Path to Python file containing @pl.function decorated function

    Returns:
        Parsed ir.Function object

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file contains no functions or multiple functions
        ParserError: If parsing fails (syntax errors, type errors, etc.)

    Warning:
        This function reads a file and executes its contents. It should only
        be used with trusted files, as executing code from untrusted sources
        can lead to arbitrary code execution vulnerabilities.

    Example:
        >>> import pypto.language as pl
        >>> # Assuming 'my_kernel.py' contains a @pl.function decorated function
        >>> func = pl.load('my_kernel.py')
        >>> print(func.name)
    """
    # Read file content
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    # Parse using parse() with the filepath for proper error reporting
    return parse(code, filename=filepath)


def parse_program(code: str, filename: str = "<string>") -> ir.Program:
    """Parse a DSL program from a string.

    This function takes Python source code containing a @pl.program decorated
    class and parses it into an IR Program object. The code is executed
    dynamically, automatically importing pypto.language as pl if not already present.

    Args:
        code: Python source code containing a @pl.program decorated class
        filename: Optional filename for error reporting (default: "<string>")

    Returns:
        Parsed ir.Program object

    Raises:
        ValueError: If the code contains no programs or multiple programs
        ParserError: If parsing fails (syntax errors, type errors, etc.)

    Warning:
        This function uses `exec()` to execute the provided code string.
        It should only be used with trusted input, as executing untrusted
        code can lead to arbitrary code execution vulnerabilities.

    Example:
        >>> import pypto.language as pl
        >>> code = '''
        ... @pl.program
        ... class MyProgram:
        ...     @pl.function
        ...     def add(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        ...         result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)
        ...         return result
        ... '''
        >>> program = pl.parse_program(code)
        >>> print(program.name)
        MyProgram
    """
    # Import pypto.language here to avoid circular imports
    import pypto.language as pl  # noqa: PLC0415

    # Make the source code available to inspect.getsourcelines() via linecache
    # IMPORTANT: We store the ORIGINAL code in linecache, not modified with import
    # This ensures line numbers match when inspect.getsourcelines() is called
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
    # This ensures inspect.getfile() finds the correct __file__ attribute
    # We use a unique module name to avoid conflicts with existing modules
    module_name = f"__pypto_parse_{id(code)}__"
    temp_module = types.ModuleType(module_name)
    temp_module.__file__ = filename
    temp_module.__setattr__("pl", pl)

    # Add module to sys.modules so inspect can find it
    sys.modules[module_name] = temp_module

    # Execute the code in the module's namespace
    # Note: Keep linecache entry until after execution completes so @pl.program can use it
    try:
        exec(compiled_code, temp_module.__dict__)
    except ParserError as e:
        # Re-raise ParserError as-is, it already has source lines
        raise e
    except Exception as e:
        # Re-raise with context about where the error occurred
        raise RuntimeError(f"Error executing code from {filename}: {e}") from e
    finally:
        # Clean up linecache entry after program is fully parsed
        if filename in linecache.cache:
            del linecache.cache[filename]
        # Clean up temporary module
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Get namespace from executed module
    namespace = temp_module.__dict__

    # Scan namespace for ir.Program instances
    programs = []
    for name, value in namespace.items():
        if isinstance(value, ir.Program):
            programs.append((name, value))

    # Validate we found exactly one program
    if len(programs) == 0:
        raise ValueError(
            f"No @pl.program decorated classes found in {filename}. "
            "Make sure your code contains a class decorated with @pl.program."
        )
    elif len(programs) > 1:
        prog_names = [name for name, _ in programs]
        raise ValueError(
            f"Multiple programs found in {filename}: {prog_names}. "
            f"pl.parse_program() can only parse code containing a single program. "
            f"Consider using separate calls or parsing from separate files."
        )

    return programs[0][1]


def load_program(filepath: str) -> ir.Program:
    """Load a DSL program from a file.

    This function reads a Python file containing a @pl.program decorated
    class and parses it into an IR Program object.

    Args:
        filepath: Path to Python file containing @pl.program decorated class

    Returns:
        Parsed ir.Program object

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file contains no programs or multiple programs
        ParserError: If parsing fails (syntax errors, type errors, etc.)

    Warning:
        This function reads a file and executes its contents. It should only
        be used with trusted files, as executing code from untrusted sources
        can lead to arbitrary code execution vulnerabilities.

    Example:
        >>> import pypto.language as pl
        >>> # Assuming 'my_program.py' contains a @pl.program decorated class
        >>> program = pl.load_program('my_program.py')
        >>> print(program.name)
    """
    # Read file content
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    # Parse using parse_program() with the filepath for proper error reporting
    return parse_program(code, filename=filepath)


__all__ = ["parse", "load", "parse_program", "load_program"]
