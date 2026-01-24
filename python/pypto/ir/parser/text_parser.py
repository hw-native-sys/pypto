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

from pypto.pypto_core import ir


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

    Example:
        >>> import pypto.language as pl
        >>> # Assuming 'my_kernel.py' contains a @pl.function decorated function
        >>> func = pl.load('my_kernel.py')
        >>> print(func.name)
    """
    # Read file content
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {filepath}") from e
    except Exception as e:
        raise IOError(f"Error reading file {filepath}: {e}") from e

    # Parse using parse() with the filepath for proper error reporting
    return parse(code, filename=filepath)


__all__ = ["parse", "load"]
