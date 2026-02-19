# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Example demonstrating manual Program construction with IRBuilder.

This example shows how to:
- Build programs with multiple functions using IRBuilder
- Use declare_function() to enable cross-function calls
- Create GlobalVar references for calling functions within a program
- Print the resulting program as Python code
"""

from typing import cast

import pypto
from pypto import DataType, ir
from pypto.ir import IRBuilder


def build_math_library():
    """Build a program with functions that call each other.

    Returns:
        IR Program with square, cube, and sum_of_squares functions
    """
    ib = IRBuilder()

    with ib.program("MathLib") as prog:
        # Declare functions up front to enable cross-function calls
        square_gvar = prog.declare_function("square")
        prog.declare_function("cube")
        prog.declare_function("sum_of_squares")

        # Build function 1: square(x) = x * x
        print("Building square function...")
        with ib.function("square") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            result = ib.let("result", x * x)
            ib.return_stmt(result)

        square_func = f.get_result()
        prog.add_function(square_func)

        # Build function 2: cube(x) = x * square(x)
        print("Building cube function (calls square)...")
        with ib.function("cube") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            # Call square function using its GlobalVar
            x_squared = ib.let("x_squared", ir.Call(square_gvar, [x], ir.Span.unknown()))
            result = ib.let("result", x * x_squared)
            ib.return_stmt(result)

        cube_func = f.get_result()
        prog.add_function(cube_func)

        # Build function 3: sum_of_squares(a, b) = square(a) + square(b)
        print("Building sum_of_squares function (calls square twice)...")
        with ib.function("sum_of_squares") as f:
            a = f.param("a", ir.ScalarType(DataType.INT64))
            b = f.param("b", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            # Call square twice using GlobalVar
            a_sq = ib.let("a_sq", ir.Call(square_gvar, [a], ir.Span.unknown()))
            b_sq = ib.let("b_sq", ir.Call(square_gvar, [b], ir.Span.unknown()))
            result = ib.let("result", a_sq + b_sq)
            ib.return_stmt(result)

        sum_func = f.get_result()
        prog.add_function(sum_func)

    program = prog.get_result()
    return program


def main():
    """Build and demonstrate the program."""
    print("=" * 70)
    print("Building MathLib Program")
    print("=" * 70)

    math_lib = build_math_library()

    print("\n" + "=" * 70)
    print("Program Information")
    print("=" * 70)
    print(f"Program name: {math_lib.name}")
    print(f"Number of functions: {len(math_lib.functions)}")
    print(f"Functions: {[f.name for f in math_lib.functions.values()]}")

    # Verify cross-function calls
    print("\n" + "=" * 70)
    print("Cross-Function Call Verification")
    print("=" * 70)
    sum_func = cast(ir.Function, math_lib.get_function("sum_of_squares"))
    print(f"sum_of_squares has {len(sum_func.params)} parameters: {[p.name for p in sum_func.params]}")
    print("It calls the 'square' function internally via GlobalVar references")

    cube_func = cast(ir.Function, math_lib.get_function("cube"))
    print(f"cube has {len(cube_func.params)} parameters: {[p.name for p in cube_func.params]}")
    print("It also calls 'square' via GlobalVar")

    # Print as Python code
    print("\n" + "=" * 70)
    print("Generated Python Code (@pl.program format)")
    print("=" * 70)
    code = pypto.ir.python_print(math_lib)
    print(code)

    print("\n" + "=" * 70)
    print("Note: The printed code includes:")
    print("  - @pl.program decorator on the class")
    print("  - 'self' parameter added to all methods")
    print("  - Cross-function calls printed as self.square(...)")
    print("  - Valid Python that can be parsed back")
    print("=" * 70)


if __name__ == "__main__":
    main()
