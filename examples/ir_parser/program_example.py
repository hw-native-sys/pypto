# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Example demonstrating @pl.program decorator with cross-function calls.

Key points:
- Methods in @pl.program class must have 'self' as first parameter (valid Python syntax)
- Cross-function calls use self.method_name() syntax
- The parser automatically strips 'self' from IR - it won't appear in generated IR functions
- Cross-function calls are resolved to GlobalVar references automatically
"""

import pypto
import pypto.language as pl

# Define a program where functions call each other
# NOTE: For now, test with pl.parse_program to avoid decorator nesting issues
program_code = """
@pl.program
class MathOps:
    @pl.function
    def square(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        result: pl.Tensor[[1], pl.INT32] = pl.op.mul(x, x)
        return result

    @pl.function
    def sum_of_squares(
        self,
        a: pl.Tensor[[1], pl.INT32],
        b: pl.Tensor[[1], pl.INT32],
    ) -> pl.Tensor[[1], pl.INT32]:
        # Call the square method using self.square()
        a_squared: pl.Tensor[[1], pl.INT32] = self.square(a)
        b_squared: pl.Tensor[[1], pl.INT32] = self.square(b)
        result: pl.Tensor[[1], pl.INT32] = pl.op.add(a_squared, b_squared)
        return result

    @pl.function
    def pythagorean(
        self,
        a: pl.Tensor[[1], pl.INT32],
        b: pl.Tensor[[1], pl.INT32],
    ) -> pl.Tensor[[1], pl.INT32]:
        # Call another function in the program using self
        result: pl.Tensor[[1], pl.INT32] = self.sum_of_squares(a, b)
        return result
"""

# Parse the program from the string
MathOps = pl.parse_program(program_code)


def main():
    """Demonstrate program usage and introspection."""
    # MathOps is now an ir.Program object
    print("=" * 70)
    print("Program Information")
    print("=" * 70)
    print(f"Program name: {MathOps.name}")
    print(f"Number of functions: {len(MathOps.functions)}")
    print(f"Function names: {[f.name for f in MathOps.functions.values()]}")

    # Verify cross-function calls
    print("\n" + "=" * 70)
    print("Function Details")
    print("=" * 70)
    sum_func = MathOps.get_function("sum_of_squares")
    assert sum_func is not None
    print(f"Function 'sum_of_squares' has {len(sum_func.params)} parameters (self was stripped)")
    print(f"Parameters: {[p.name for p in sum_func.params]}")
    print("It calls 'square' internally via GlobalVar references")

    # Print the program back as Python code
    print("\n" + "=" * 70)
    print("Program as Python Code")
    print("=" * 70)
    code = pypto.ir.python_print(MathOps)
    print(code)

    print("\n" + "=" * 70)
    print("Round-Trip Test")
    print("=" * 70)
    # Parse the printed code back
    reparsed = pl.parse_program(code)
    print(f"Reparsed program name: {reparsed.name}")
    print(f"Reparsed function count: {len(reparsed.functions)}")
    print("Round-trip successful!")


if __name__ == "__main__":
    main()
