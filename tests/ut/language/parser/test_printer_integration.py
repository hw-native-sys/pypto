# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration tests for parser and printer round-trip."""

import pypto
import pypto.language as pl
import pytest
from pypto import DataType, ir


class TestPrinterIntegration:
    """Tests for printer integration with new subscript syntax."""

    def test_tensor_type_printed_with_subscript(self):
        """Test that TensorType is printed with subscript notation."""
        tensor_type = ir.TensorType([64, 128], DataType.FP16)

        result = pypto.ir.python_print(tensor_type)

        # Should use subscript notation
        assert "pl.Tensor[[64, 128], pl.FP16]" in result
        # Should NOT use call notation
        assert "pl.Tensor((" not in result

    def test_tile_type_printed_with_subscript(self):
        """Test that TileType is printed with subscript notation."""
        tile_type = ir.TileType([16, 16], DataType.FP32)

        result = pypto.ir.python_print(tile_type)

        # Should use subscript notation
        assert "pl.Tile[[16, 16], pl.FP32]" in result
        # Should NOT use call notation
        assert "pl.Tile((" not in result

    def test_function_printed_with_subscript_types(self):
        """Test that function parameters use subscript notation."""

        @pl.function
        def test_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.cast(x, target_type=pl.FP32)
            return result

        # Print the function
        printed = pypto.ir.python_print(test_func)

        # Check subscript notation is used
        assert "pl.Tensor[[64, 128], pl.FP16]" in printed
        assert "pl.Tensor[[64, 128], pl.FP32]" in printed
        # Check old notation is NOT used
        assert "pl.Tensor((" not in printed

    def test_parsed_function_printer_round_trip(self):
        """Test that parsed functions can be printed correctly."""

        @pl.function
        def round_trip(
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            sum_val: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
            result: pl.Tensor[[64], pl.FP32] = pl.mul(sum_val, 2.0)
            return result

        # Print and check syntax
        printed = pypto.ir.python_print(round_trip)

        assert "def round_trip" in printed
        assert "pl.Tensor[[64], pl.FP32]" in printed
        # Printer uses simplified tensor operation notation
        assert "tensor.add" in printed or "pl.add" in printed

    def test_yield_type_annotation_in_if_statement(self):
        """Test that type annotations on yield assignments are printed (issue #185)."""

        @pl.function
        def func_with_if_yield(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)

            for i, (acc,) in pl.range(5, init_values=(init,)):
                if i == 0:
                    out_c: pl.Tensor[[64, 128], pl.FP32] = pl.mul(acc, 2.0)
                    val: pl.Tensor[[64, 128], pl.FP32] = pl.yield_(out_c)
                else:
                    val: pl.Tensor[[64, 128], pl.FP32] = pl.yield_(acc)

                result = pl.yield_(val)

            return result

        # Print and verify type annotation is present
        printed = pypto.ir.python_print(func_with_if_yield)

        # Should have type annotation on single-variable yield (not just "val = pl.yield_(out_c)")
        assert "val: pl.Tensor[[64, 128], pl.FP32] = pl.yield_" in printed

    def test_tuple_yield_no_type_annotation(self):
        """Test that tuple yields don't print type annotations (not valid Python syntax)."""

        @pl.function
        def func_with_tuple_yield(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
            init1: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
            init2: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)

            for i, (a1, a2) in pl.range(5, init_values=(init1, init2)):
                if i == 0:
                    new1: pl.Tensor[[64], pl.FP32] = pl.mul(a1, 2.0)
                    new2: pl.Tensor[[64], pl.FP32] = pl.mul(a2, 3.0)
                    val1, val2 = pl.yield_(new1, new2)
                else:
                    val1, val2 = pl.yield_(a1, a2)

                out1, out2 = pl.yield_(val1, val2)

            return out1

        # Print and verify tuple yields don't have type annotations
        printed = pypto.ir.python_print(func_with_tuple_yield)

        # Tuple unpacking should NOT have type annotations
        assert "val1, val2 = pl.yield_" in printed
        # Ensure no type annotations are added to tuple-unpacked variables
        assert "val1: pl.Tensor" not in printed
        assert "val2: pl.Tensor" not in printed


class TestWhileLoopRoundTrip:
    """Round-trip tests for while loop parsing and printing."""

    def test_while_loop_natural_syntax(self):
        """Test that natural while loop can be parsed and printed."""

        @pl.function
        def while_natural(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            x: pl.Scalar[pl.INT64] = 0
            while x < n:
                x = x + 1
            return x

        # Print the function
        printed = pypto.ir.python_print(while_natural)

        # Check that natural syntax is present
        assert "while" in printed
        assert "x < n" in printed or "x<n" in printed

        # Verify structural properties
        assert isinstance(while_natural, ir.Function)
        assert while_natural.name == "while_natural"

    def test_while_loop_with_multiple_variables(self):
        """Test while loop with multiple variable updates."""

        @pl.function
        def while_multi(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            x: pl.Scalar[pl.INT64] = 0
            y: pl.Scalar[pl.INT64] = 1
            while x < n:
                x = x + 1
                y = y * 2
            return y

        # Print the function
        printed = pypto.ir.python_print(while_multi)

        # Check for while loop
        assert "while" in printed
        # Check for both variables
        assert "x" in printed and "y" in printed

    def test_nested_while_loops_round_trip(self):
        """Test nested while loops round-trip."""

        @pl.function
        def nested_while(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            x: pl.Scalar[pl.INT64] = 0
            while x < n:
                y: pl.Scalar[pl.INT64] = 0
                while y < 3:
                    y = y + 1
                x = x + 1
            return x

        # Print the function
        printed = pypto.ir.python_print(nested_while)

        # Should have multiple while loops
        assert printed.count("while") >= 2

    def test_while_in_for_round_trip(self):
        """Test while loop inside for loop round-trip."""

        @pl.function
        def while_in_for(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            init_sum: pl.Scalar[pl.INT64] = 0

            for i, (sum_val,) in pl.range(5, init_values=(init_sum,)):
                x: pl.Scalar[pl.INT64] = 0
                while x < i:
                    x = x + 1
                new_sum: pl.Scalar[pl.INT64] = sum_val + x
                sum_out = pl.yield_(new_sum)

            return sum_out

        # Print the function
        printed = pypto.ir.python_print(while_in_for)

        # Should have both for and while
        assert "pl.range" in printed
        assert "while" in printed

    def test_for_in_while_round_trip(self):
        """Test for loop inside while loop round-trip."""

        @pl.function
        def for_in_while(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            x: pl.Scalar[pl.INT64] = 0
            while x < n:
                init_acc: pl.Scalar[pl.INT64] = x
                for i, (acc,) in pl.range(3, init_values=(init_acc,)):
                    new_acc: pl.Scalar[pl.INT64] = acc + 1
                    acc_out = pl.yield_(new_acc)
                x = acc_out
            return x

        # Print the function
        printed = pypto.ir.python_print(for_in_while)

        # Should have both while and for
        assert "while" in printed
        assert "pl.range" in printed

    def test_while_structural_equality_after_print(self):
        """Test that while loop structure is preserved after printing."""

        @pl.function
        def original(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            x: pl.Scalar[pl.INT64] = 0
            while x < n:
                x = x + 1
            return x

        # Verify key structural elements
        assert isinstance(original, ir.Function)
        # Find the while statement
        body = original.body
        while_stmt = None
        if isinstance(body, ir.SeqStmts):
            for stmt in body.stmts:
                if isinstance(stmt, ir.WhileStmt):
                    while_stmt = stmt
                    break
        elif isinstance(body, ir.WhileStmt):
            while_stmt = body

        assert while_stmt is not None
        # Natural syntax has no iter_args initially (ConvertToSSA adds them)
        # Condition should be a comparison
        assert isinstance(while_stmt.condition, ir.Lt)

    def test_while_with_tensor_operations_round_trip(self):
        """Test while loop with tensor operations."""

        @pl.function
        def while_tensors(n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            i: pl.Scalar[pl.INT64] = 0
            acc: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
            while i < n:
                i = i + 1
                acc = pl.add(acc, x)
            return acc

        # Print the function
        printed = pypto.ir.python_print(while_tensors)

        # Should have while loop and tensor operations
        assert "while" in printed
        assert "pl.add" in printed or "tensor.add" in printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
