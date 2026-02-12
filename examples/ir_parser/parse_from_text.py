# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Example demonstrating pl.parse() and pl.loads() for parsing DSL functions from text/files.

This example shows how to:
1. Parse DSL functions from inline code strings using pl.parse()
2. Load DSL functions from external files using pl.loads()
3. Handle errors during parsing
4. Use cases for dynamic code generation and loading kernels
"""

import os
import tempfile
from typing import cast

import pypto
import pypto.language as pl


def example_parse_from_string():
    """Example 1: Parse a DSL function from an inline string."""
    print("=" * 70)
    print("Example 1: Parse from inline string")
    print("=" * 70)

    # Define DSL code as a string
    code = """
@pl.function
def vector_add(
    x: pl.Tensor[[128], pl.FP32],
    y: pl.Tensor[[128], pl.FP32],
) -> pl.Tensor[[128], pl.FP32]:
    result: pl.Tensor[[128], pl.FP32] = pl.add(x, y)
    return result
"""

    # Parse the function
    func = cast(pypto.ir.Function, pl.parse(code))

    print(f"Parsed function: {func.name}")
    print(f"Number of parameters: {len(func.params)}")
    print(f"Number of return types: {len(func.return_types)}")
    print("\nFunction IR:")
    print(pypto.ir.python_print(func))
    print()


def example_parse_without_import():
    """Example 2: Parse DSL code without import statement (auto-injected)."""
    print("=" * 70)
    print("Example 2: Parse without import (auto-injected)")
    print("=" * 70)

    # Notice: no "import pypto.language as pl" in the code
    code = """
@pl.function
def vector_mul(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    return result
"""

    func = pl.parse(code)
    print(f"Parsed function: {func.name}")
    print("✓ Import statement was automatically injected")
    print()


def example_load_from_file():
    """Example 3: Load a DSL function from a file."""
    print("=" * 70)
    print("Example 3: Load from file")
    print("=" * 70)

    # Create a temporary file with DSL code
    code = """
import pypto.language as pl

@pl.function
def matrix_transpose(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[128, 64], pl.FP16]:
    # Example: simplified transpose operation
    result: pl.Tensor[[128, 64], pl.FP16] = pl.view(x, [128, 64], [1, 0])
    return result
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        # Load the function from file
        func = cast(pypto.ir.Function, pl.loads(temp_path))
        print(f"Loaded function from: {temp_path}")
        print(f"Function name: {func.name}")
        print(f"Parameters: {len(func.params)}")
        print()
    finally:
        # Clean up

        os.unlink(temp_path)


def example_complex_function():
    """Example 4: Parse a complex function with control flow."""
    print("=" * 70)
    print("Example 4: Parse complex function with control flow")
    print("=" * 70)

    code = """
@pl.function
def accumulate(
    x: pl.Tensor[[10], pl.FP32],
    iterations: pl.Tensor[[1], pl.INT32],
) -> pl.Tensor[[10], pl.FP32]:
    # Initialize accumulator
    init_sum: pl.Tensor[[10], pl.FP32] = pl.create_tensor([10], dtype=pl.FP32)

    # Accumulate over iterations
    for i, (running_sum,) in pl.range(5, init_values=(init_sum,)):
        new_sum: pl.Tensor[[10], pl.FP32] = pl.add(running_sum, x)
        result = pl.yield_(new_sum)

    return result
"""

    func = pl.parse(code)
    print(f"Parsed function: {func.name}")
    print("✓ Function includes for loop control flow")
    print("\nFunction IR:")
    print(pypto.ir.python_print(func))
    print()


def example_error_handling():
    """Example 5: Handle parsing errors."""
    print("=" * 70)
    print("Example 5: Error handling")
    print("=" * 70)

    # Example 5a: No function defined
    print("5a. Code with no @pl.function:")
    code_no_func = """
x = 42
y = x + 1
"""
    try:
        pl.parse(code_no_func)
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
        print()

    # Example 5b: Multiple functions
    print("5b. Code with multiple functions:")
    code_multi_func = """
@pl.function
def func1(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x

@pl.function
def func2(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
"""
    try:
        pl.parse(code_multi_func)
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
        print()

    # Example 5c: Syntax error
    print("5c. Code with syntax error:")
    code_syntax_error = """
@pl.function
def bad_syntax(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x +
"""
    try:
        pl.parse(code_syntax_error)
    except SyntaxError as e:
        print(f"✓ Caught expected error: {type(e).__name__}")
        print()


def example_use_case_dynamic_generation():
    """Example 6: Use case - Dynamic code generation."""
    print("=" * 70)
    print("Example 6: Use case - Dynamic kernel generation")
    print("=" * 70)

    # Simulate generating kernels from a template
    def generate_elementwise_kernel(operation: str, op_func: str) -> str:
        """Generate an elementwise operation kernel from a template."""
        return f"""
@pl.function
def elementwise_{operation}(
    x: pl.Tensor[[1024], pl.FP32],
    y: pl.Tensor[[1024], pl.FP32],
) -> pl.Tensor[[1024], pl.FP32]:
    result: pl.Tensor[[1024], pl.FP32] = pl.{op_func}(x, y)
    return result
"""

    # Generate and parse different kernels
    operations = [
        ("add", "add"),
        ("subtract", "sub"),
        ("multiply", "mul"),
    ]

    for op_name, op_func in operations:
        code = generate_elementwise_kernel(op_name, op_func)
        func = pl.parse(code)
        print(f"✓ Generated and parsed: {func.name}")

    print("\n💡 This pattern is useful for:")
    print("   - Generating specialized kernels from configuration")
    print("   - Loading kernel implementations from a database")
    print("   - Building domain-specific code generators")
    print()


def example_serialization():
    """Example 7: Serialize parsed functions."""
    print("=" * 70)
    print("Example 7: Serialize parsed functions")
    print("=" * 70)

    code = """
@pl.function
def simple_add(x: pl.Tensor[[32], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
    result: pl.Tensor[[32], pl.FP32] = pl.add(x, 1.0)
    return result
"""

    # Parse function
    func = pl.parse(code)
    print(f"Parsed function: {func.name}")

    # Serialize to msgpack
    data = pypto.ir.serialize(func)
    print(f"✓ Serialized to {len(data)} bytes")

    # Deserialize
    restored = cast(pypto.ir.Function, pypto.ir.deserialize(data))
    print(f"✓ Deserialized function: {restored.name}")
    print("\n💡 Parsed functions can be serialized just like decorated functions")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "pl.parse() and pl.loads() Examples" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    example_parse_from_string()
    example_parse_without_import()
    example_load_from_file()
    example_complex_function()
    example_error_handling()
    example_use_case_dynamic_generation()
    example_serialization()

    print("=" * 70)
    print("All examples completed successfully! ✓")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
