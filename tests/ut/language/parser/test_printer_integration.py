# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration tests for parser and printer round-trip."""

import struct

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.ir import op
from pypto.language.parser.diagnostics import ParserSyntaxError, ParserTypeError
from pypto.language.parser.text_parser import parse


class TestPrinterIntegration:
    """Tests for printer integration with new subscript syntax."""

    def test_tensor_type_printed_with_subscript(self):
        """Test that TensorType is printed with subscript notation."""
        tensor_type = ir.TensorType([64, 128], DataType.FP16)

        result = ir.python_print_type(tensor_type)

        # Should use subscript notation
        assert "pl.Tensor[[64, 128], pl.FP16]" in result
        # Should NOT use call notation
        assert "pl.Tensor((" not in result

    def test_tile_type_printed_with_subscript(self):
        """Test that TileType is printed with subscript notation."""
        tile_type = ir.TileType([16, 16], DataType.FP32)

        result = ir.python_print_type(tile_type)

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
        printed = test_func.as_python()

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
        printed = round_trip.as_python()

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
        printed = func_with_if_yield.as_python()

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
        printed = func_with_tuple_yield.as_python()

        # Tuple unpacking should NOT have type annotations
        assert "val1, val2 = pl.yield_" in printed
        # Ensure no type annotations are added to tuple-unpacked variables
        assert "val1: pl.Tensor" not in printed
        assert "val2: pl.Tensor" not in printed


class TestCastModeRoundTrip:
    """Tests for cast mode printing as string name and parsing both string/int modes."""

    def test_printer_outputs_mode_as_string_name(self):
        """Test that printer outputs mode='round' instead of mode=2."""

        @pl.function
        def cast_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.cast(x, target_type=pl.FP32, mode="round")
            return result

        printed = cast_func.as_python()

        # Mode should be printed as string name, not integer (quote style may vary)
        assert "mode='round'" in printed or 'mode="round"' in printed
        assert "mode=2" not in printed

    def test_printer_outputs_all_mode_names(self):
        """Test that all cast modes are printed as string names."""
        mode_names = ["none", "rint", "round", "floor", "ceil", "trunc", "odd"]

        def _make_cast_func(mode_name: str):
            @pl.function
            def cast_func(x: pl.Tensor[[64], pl.FP16]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.cast(x, target_type=pl.FP32, mode=mode_name)
                return result

            return cast_func

        for name in mode_names:
            cast_func = _make_cast_func(name)
            printed = cast_func.as_python()
            assert f"mode='{name}'" in printed or f'mode="{name}"' in printed, (
                f"Expected mode='{name}' in printed output, got: {printed}"
            )

    def test_parser_accepts_int_mode(self):
        """Test that parser accepts mode=2 (int) via IR API."""
        span = ir.Span.unknown()
        dim64 = ir.ConstInt(64, DataType.INT32, span)
        tensor_type = ir.TensorType([dim64], DataType.FP16)
        tensor_var = ir.Var("x", tensor_type, span)

        # Call with int mode
        call = op.tensor.cast(tensor_var, DataType.FP32, mode=2, span=span)
        assert isinstance(call, ir.Call)
        assert call.op.name == ir.get_op("tensor.cast").name

    def test_cast_mode_round_trip(self):
        """Test parse → print → re-parse round-trip with mode='round'."""

        @pl.function
        def original(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.cast(x, target_type=pl.FP32, mode="round")
            return result

        # Print to string
        printed = original.as_python()

        # Re-parse the printed output
        reparsed = pl.parse(printed)

        # Verify structural equality
        ir.assert_structural_equal(original, reparsed)

    def test_cast_default_mode_round_trip(self):
        """Test that cast with default mode (no explicit mode) round-trips correctly."""

        @pl.function
        def original(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.cast(x, target_type=pl.FP32)
            return result

        printed = original.as_python()

        # Default mode is "round", so it should still print as 'round' (quote style may vary)
        assert "mode='round'" in printed or 'mode="round"' in printed

        reparsed = pl.parse(printed)
        ir.assert_structural_equal(original, reparsed)


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
        printed = while_natural.as_python()

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
        printed = while_multi.as_python()

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
        printed = nested_while.as_python()

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
        printed = while_in_for.as_python()

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
        printed = for_in_while.as_python()

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
        assert len(while_stmt.iter_args) == 0
        # Condition should be a comparison
        assert isinstance(while_stmt.condition, ir.Lt)

        # The structure must survive a print -> reparse round trip unchanged
        reparsed = pl.parse(original.as_python())
        assert isinstance(reparsed, ir.Function)
        ir.assert_structural_equal(reparsed, original)

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
        printed = while_tensors.as_python()

        # Should have while loop and tensor operations
        assert "while" in printed
        assert "pl.add" in printed or "tensor.add" in printed

    def test_tensor_create_round_trip(self):
        """Test that pl.tensor.create round-trips through printer and parser."""

        @pl.function
        def func() -> pl.Tensor[[16, 1], pl.FP32]:
            y: pl.Tensor[[16, 1], pl.FP32] = pl.create_tensor([16, 1], dtype=pl.FP32)
            return y

        printed = func.as_python()
        assert "pl.tensor.create(" in printed

        reparsed = parse("import pypto.language as pl\n\n" + printed)
        ir.assert_structural_equal(func, reparsed)

    def test_tile_create_round_trip(self):
        """Test that pl.tile.create round-trips through printer and parser."""

        @pl.function
        def func(t: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
            _tile: pl.Tile[[64, 16], pl.FP32] = pl.create_tile(
                [64, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            return t

        printed = func.as_python()
        assert "pl.tile.create(" in printed
        assert "pl.tile.create_tile(" not in printed


def _roundtrip_program(params=(), values=()):
    """Check a complete signature and body, including expression node kinds."""
    span = ir.Span.unknown()
    program = ir.Program(
        [ir.Function("f", list(params), [v.type for v in values], ir.ReturnStmt(list(values), span), span)],
        "Roundtrip",
        span,
    )
    printed = ir.python_print(program, format=False)
    reparsed = pl.parse_program(printed)
    ir.assert_structural_equal(program, reparsed)
    assert ir.structural_hash(program) == ir.structural_hash(reparsed)
    return reparsed


class TestTextFormRoundtrip:
    """Regression coverage for supported values and internal signature markers."""

    @pytest.mark.parametrize("op_type,dtype", [(ir.Xor, DataType.BOOL), (ir.BitXor, DataType.INT32)])
    def test_xor_node_kind(self, op_type, dtype):
        span = ir.Span.unknown()
        x, y = [ir.Var(name, ir.ScalarType(dtype), span) for name in ("x", "y")]
        expr = op_type(x, y, dtype, span)
        _roundtrip_program([x, y], [expr])

    @pytest.mark.parametrize("dtype", [DataType.FP16, DataType.FP32, DataType.BF16])
    @pytest.mark.parametrize(
        "bits",
        [
            "7ff0000000000000",
            "fff0000000000000",
            "7ff8000000000000",
            "7ff8000000000012",
            "fff8000000000012",
            "7ff0000000000001",
        ],
    )
    def test_nonfinite_float(self, bits, dtype):
        value = struct.unpack(">d", bytes.fromhex(bits))[0]
        expr = ir.ConstFloat(value, dtype, ir.Span.unknown())
        reparsed = _roundtrip_program(values=[expr])
        func = reparsed.get_function("f")
        assert func is not None and isinstance(func.body, ir.ReturnStmt)
        result = func.body.value[0]
        assert isinstance(result, ir.ConstFloat)
        assert struct.pack(">d", result.value).hex() == bits

    @pytest.mark.parametrize(
        "left,right,equal",
        [
            ("7ff8000000000012", "7ff8000000000012", True),
            ("7ff8000000000012", "7ff8000000000013", False),
            ("7ff8000000000012", "fff8000000000012", False),
            ("7ff0000000000000", "fff0000000000000", False),
            ("0000000000000000", "8000000000000000", True),
        ],
    )
    def test_float_structural_identity(self, left, right, equal):
        span = ir.Span.unknown()
        lhs, rhs = [
            ir.ConstFloat(struct.unpack(">d", bytes.fromhex(bits))[0], DataType.FP32, span)
            for bits in (left, right)
        ]
        assert ir.structural_equal(lhs, rhs) == equal
        if equal:
            ir.assert_structural_equal(lhs, rhs)
            assert ir.structural_hash(lhs) == ir.structural_hash(rhs)
        else:
            with pytest.raises(ValueError, match="Structural equality assertion failed"):
                ir.assert_structural_equal(lhs, rhs)
        other_dtype = ir.ConstFloat(lhs.value, DataType.FP16, span)
        assert not ir.structural_equal(lhs, other_dtype)

    @pytest.mark.parametrize(
        "expression,error,match",
        [
            ("pl._dump.logical_xor(x, x)", ParserTypeError, "BOOL scalar operands"),
            ("pl._dump.logical_xor(True)", ParserSyntaxError, "two positional operands"),
            ("pl._dump.logical_xor(True, False, extra=True)", ParserSyntaxError, "two positional operands"),
            ('pl._dump.float64("nan")', ParserSyntaxError, "16 hexadecimal digits"),
            ('pl._dump.float64("7ff8000000000000", extra=1)', ParserSyntaxError, "one hexadecimal"),
            (
                'pl.const(pl._dump.float64("7ff8000000000000"), pl.INT32)',
                ParserTypeError,
                "floating-point dtype",
            ),
        ],
    )
    def test_malformed_dump_expression(self, expression, error, match):
        code = f"""
@pl.program
class Invalid:
    @pl.function
    def f(self, x: pl.Scalar[pl.INT32]):
        return {expression}
"""
        with pytest.raises(error, match=match):
            pl.parse_program(code)

    def test_logical_xor_nesting(self):
        span = ir.Span.unknown()
        x, y = [ir.Var(name, ir.ScalarType(DataType.BOOL), span) for name in ("x", "y")]
        expr = ir.Not(ir.Xor(ir.And(x, y, DataType.BOOL, span), x, DataType.BOOL, span), DataType.BOOL, span)
        _roundtrip_program([x, y], [expr])

    @pytest.mark.parametrize("namespace", ["tensor", "tile"])
    @pytest.mark.parametrize("value", ["math.inf", "-math.inf", "math.nan"])
    def test_nonfinite_fill_from_dsl(self, namespace, value):
        program = pl.parse_program(f"""
import math
FILL = {value}
@pl.program
class Fill:
    @pl.function
    def f(self):
        x = pl.{namespace}.full([16, 16], dtype=pl.FP32, value=FILL)
        return
""")
        ir.assert_structural_equal(program, pl.parse_program(ir.python_print(program)))

    @pytest.mark.parametrize("nested", [False, True])
    def test_tuple_signature(self, nested):
        scalar = ir.ScalarType(DataType.INT32)
        typ = ir.TupleType([ir.TupleType([scalar]), ir.TupleType([])]) if nested else ir.TupleType([scalar])
        x = ir.Var("x", typ, ir.Span.unknown())
        _roundtrip_program([x], [x])

    def test_parser_produced_nested_tuple(self):
        program = pl.parse_program("""
@pl.program
class Nested:
    @pl.function
    def f(self, x: pl.Scalar[pl.INT32]) -> pl.Tuple[pl.Tuple[pl.Scalar[pl.INT32]], pl.Scalar[pl.INT32]]:
        inner = (x,)
        outer = (inner, x)
        value = outer[0][0]
        return outer
""")
        ir.assert_structural_equal(program, pl.parse_program(ir.python_print(program)))

    def test_nested_return_list_still_rejected(self):
        with pytest.raises(ParserTypeError, match="Nested tuple types"):
            pl.parse_program("""
@pl.program
class Invalid:
    @pl.function
    def f(self, x: pl.Tuple[tuple[pl.Scalar[pl.INT32]]]):
        return
""")

    def test_void_is_not_a_tuple_value(self):
        with pytest.raises(ValueError, match="cannot use VoidType"):
            ir.TupleType([ir.VoidType()])

    @pytest.mark.parametrize(
        "type_factory",
        [
            ir.PtrType,
            ir.WindowBufferType,
            ir.UnknownType,
            ir.CommCtxType.get,
            ir.PrefetchAsyncContextType.get,
            ir.AsyncEventType.get,
            ir.AsyncSessionType.get,
        ],
    )
    def test_internal_signature(self, type_factory):
        x = ir.Var("x", type_factory(), ir.Span.unknown())
        _roundtrip_program([x], [x])

    @pytest.mark.parametrize(
        "op_type",
        [
            ir.Add,
            ir.Sub,
            ir.Mul,
            ir.FloorDiv,
            ir.FloorMod,
            ir.Pow,
            ir.BitAnd,
            ir.BitOr,
            ir.BitXor,
            ir.BitShiftLeft,
            ir.BitShiftRight,
        ],
    )
    @pytest.mark.parametrize("dynamic", [False, True])
    def test_composite_shape(self, op_type, dynamic):
        span = ir.Span.unknown()
        n = ir.Var("n", ir.ScalarType(DataType.INDEX), span)
        lhs = n if dynamic else ir.ConstInt(10, DataType.INDEX, span)
        dim = op_type(lhs, ir.ConstInt(3, DataType.INDEX, span), DataType.INDEX, span)
        x = ir.Var("x", ir.TensorType([dim], DataType.FP32), span)
        _roundtrip_program([n, x] if dynamic else [x], [x])

    @pytest.mark.parametrize(
        "op_type", [ir.Pow, ir.BitAnd, ir.BitOr, ir.BitXor, ir.BitShiftLeft, ir.BitShiftRight]
    )
    @pytest.mark.parametrize("type_factory", [ir.TensorType, ir.TileType])
    def test_nested_typed_composite_shape(self, op_type, type_factory):
        span = ir.Span.unknown()
        lhs, rhs = [ir.ConstInt(n, DataType.INT32, span) for n in (10, 3)]
        inner = op_type(lhs, rhs, DataType.INT32, span)
        dim = ir.Add(inner, rhs, DataType.INT32, span)
        x = ir.Var("x", type_factory([dim], DataType.FP32), span)
        _roundtrip_program([x], [x])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
