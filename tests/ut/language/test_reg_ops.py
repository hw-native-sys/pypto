# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Register regions construct canonical IR and enforce ownership."""

import inspect

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir.printer import python_print
from pypto.language.parser import ParserError


def test_reg_region_roundtrip():
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        src: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec], dst: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec]
    ):
        with pl.reg_scope():
            x = pl.reg.load(src, offset=0, lanes=64)
            m = pl.reg.full_mask(lanes=64)
            y = pl.reg.add(x, x, mask=m)
            out = pl.reg.store(dst, offset=0, value=y)
        _z = pl.tile.add(out, out)

    printed = python_print(kernel, format=False)
    assert "with pl.reg_scope():" in printed
    assert "pl.reg.store(dst, 0, y)" in printed
    restored = pl.parse(printed)
    ir.assert_structural_equal(kernel, restored, enable_auto_mapping=True)


def test_reg_escape_rejected():
    with pytest.raises((ParserError, ValueError), match="escape|outside|RegScope"):

        @pl.function(type=pl.FunctionType.InCore)
        def kernel(src: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec]):
            with pl.reg_scope():
                x = pl.reg.load(src, lanes=64)
            _y = pl.reg.add(x, x)


def test_bare_store_rejected():
    with pytest.raises((ParserError, ValueError), match="assign|Assign|bare"):

        @pl.function(type=pl.FunctionType.InCore)
        def kernel(src: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec]):
            with pl.reg_scope():
                x = pl.reg.load(src, lanes=64)
                pl.reg.store(src, value=x)


def test_nested_scope_rejected():
    with pytest.raises((ParserError, ValueError), match="nest"):

        @pl.function(type=pl.FunctionType.InCore)
        def kernel():
            with pl.reg_scope():
                with pl.reg_scope():
                    _x = pl.reg.full_mask(lanes=64)


def test_all_p0_operations_roundtrip_with_normalized_attributes():
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        src: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec],
        base: pl.Scalar[pl.FP32],
    ):
        with pl.reg_scope():
            x = pl.reg.load(src, lanes=64)
            b = pl.reg.broadcast(base, lanes=64, dtype=pl.FP32)
            _indices = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
            full = pl.reg.full_mask(lanes=64)
            active = pl.reg.active_mask(32, lanes=64)
            both = pl.reg.mask_and(full, active)
            either = pl.reg.mask_or(full, active)
            different = pl.reg.mask_xor(both, either)
            inverse = pl.reg.mask_not(different)
            a = pl.reg.add(x, b, mask=active)
            s = pl.reg.sub(a, b)
            m = pl.reg.mul(s, b)
            d = pl.reg.div(m, b)
            low = pl.reg.minimum(d, x)
            high = pl.reg.maximum(low, x)
            condition = pl.reg.compare(high, b, inverse, predicate="lt")
            selected = pl.reg.select(condition, high, b)
            half = pl.reg.convert(selected, dtype=pl.FP16, saturate=False)
            widened = pl.reg.convert(half, dtype=pl.FP32)
            out = pl.reg.store(src, value=widened, mask=active)
        _consumer = pl.tile.add(out, out)

    printed = python_print(kernel, format=False)
    assert 'rounding="R"' in printed or "rounding='R'" in printed
    restored = pl.parse(printed)
    ir.assert_structural_equal(kernel, restored, enable_auto_mapping=True)


def test_captures_argument_rejected():
    with pytest.raises((ParserError, ValueError), match="no arguments|implicit"):
        pl.parse("""
@pl.function(type=pl.FunctionType.InCore)
def kernel(src: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec]):
    with pl.reg_scope(captures=[src]):
        x = pl.reg.load(src, lanes=64)
""")


def test_same_name_store_binding_can_escape_as_tile():
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(src: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec]):
        with pl.reg_scope():
            value = pl.reg.load(src, lanes=64)
            src = pl.reg.store(src, value=value)
        _consumer = pl.tile.add(src, src)

    restored = pl.parse(python_print(kernel, format=False))
    ir.assert_structural_equal(kernel, restored, enable_auto_mapping=True)


def test_builder_reg_scope_span_points_to_with_callsite():
    builder = ir.IRBuilder()
    with builder.function("kernel", type=ir.FunctionType.InCore) as function:
        with_line = inspect.stack()[0].lineno + 1
        with builder.reg_scope():
            pass
    scope = function.get_result().body
    assert scope.span.filename.endswith("test_reg_ops.py")
    assert scope.span.begin_line == with_line


@pytest.mark.parametrize("op_name", ["broadcast", "index_gen"])
@pytest.mark.parametrize("dtype,value", [(pl.INT32, 2), (pl.FP32, 1.5)])
def test_raw_generator_literal_roundtrip(op_name, dtype, value):
    # Builder-created literals must keep their type through the DSL printer.
    span = ir.Span.unknown()
    literal = ir.ConstInt(value, dtype, span) if isinstance(value, int) else ir.ConstFloat(value, dtype, span)
    builder = ir.IRBuilder()
    with builder.function("kernel", type=ir.FunctionType.InCore) as function:
        with builder.reg_scope():
            kwargs = {"lanes": 64, "dtype": dtype}
            if op_name == "index_gen":
                kwargs["order"] = "ascending"
            builder.emit_reg("value", f"reg.{op_name}", [literal], kwargs)
    original = function.get_result()
    restored = pl.parse(python_print(original, format=False))
    ir.assert_structural_equal(original, restored, enable_auto_mapping=True)


@pytest.mark.parametrize(
    ("op_name", "kwargs"),
    [("reg.add", {}), ("reg.index_gen", {"lanes": 64, "dtype": pl.INT32})],
)
def test_raw_reg_defaults_roundtrip(op_name, kwargs):
    """Raw EmitReg calls use the same defaults as their public DSL wrappers."""
    span = ir.Span.unknown()
    builder = ir.IRBuilder()
    with builder.function("kernel", type=ir.FunctionType.InCore) as function:
        with builder.reg_scope():
            scalar = ir.ConstFloat(1.0, pl.FP32, span)
            value = builder.emit_reg(
                "value", "reg.broadcast", [scalar], {"lanes": 64, "dtype": pl.FP32}, span
            )
            args = [value, value] if op_name == "reg.add" else [ir.ConstInt(0, pl.INT32, span)]
            builder.emit_reg("result", op_name, args, kwargs, span)
    original = function.get_result()
    restored = pl.parse(python_print(original, format=False))
    ir.assert_structural_equal(original, restored, enable_auto_mapping=True)


def test_raw_same_dtype_convert_roundtrip():
    """A no-op raw conversion matches the DSL's operand-preserving behavior."""
    span = ir.Span.unknown()
    builder = ir.IRBuilder()
    with builder.function("kernel", type=ir.FunctionType.InCore) as function:
        with builder.reg_scope():
            scalar = ir.ConstFloat(1.0, pl.FP32, span)
            value = builder.emit_reg(
                "value", "reg.broadcast", [scalar], {"lanes": 64, "dtype": pl.FP32}, span
            )
            builder.emit_reg("result", "reg.convert", [value], {"dtype": pl.FP32}, span)
    original = function.get_result()
    restored = pl.parse(python_print(original, format=False))
    ir.assert_structural_equal(original, restored, enable_auto_mapping=True)


@pytest.mark.parametrize(
    ("op_name", "kwargs"),
    [("reg.add", {}), ("reg.index_gen", {"lanes": 64, "dtype": pl.INT32})],
)
def test_raw_create_op_defaults_roundtrip(op_name, kwargs):
    """Public raw op construction applies the same register defaults as DSL calls."""
    span = ir.Span.unknown()
    scalar = ir.ConstFloat(1.0, pl.FP32, span)
    value = ir.create_op_call("reg.broadcast", [scalar], {"lanes": 64, "dtype": pl.FP32}, span)
    value_var = ir.Var("value", value.type, span)
    args = [value_var, value_var] if op_name == "reg.add" else [ir.ConstInt(0, pl.INT32, span)]
    raw = ir.create_op_call(op_name, args, kwargs, span)
    result_var = ir.Var("result", raw.type, span)
    body = ir.RegScopeStmt(
        "",
        ir.SeqStmts(
            [ir.AssignStmt(value_var, value, span), ir.AssignStmt(result_var, raw, span)],
            span,
        ),
        span,
    )
    function = ir.Function("kernel", [], [], body, span, type=ir.FunctionType.InCore)
    printed = python_print(function, format=False)
    restored = pl.parse(printed)
    ir.assert_structural_equal(function, restored, enable_auto_mapping=True)


def test_raw_create_op_rejects_same_dtype_convert():
    """Public raw construction rejects the DSL's operand-preserving no-op form."""
    span = ir.Span.unknown()
    scalar = ir.ConstFloat(1.0, pl.FP32, span)
    value = ir.create_op_call("reg.broadcast", [scalar], {"lanes": 64, "dtype": pl.FP32}, span)
    with pytest.raises(ValueError, match="same-dtype|no-op|canonical"):
        ir.create_op_call("reg.convert", [value], {"dtype": pl.FP32}, span)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
