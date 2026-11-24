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
from pypto import DataType, ir
from pypto.ir.printer import python_print
from pypto.language.parser import ParserError


def test_dynamic_active_mask_roundtrip():
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(src: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec], count: pl.Scalar[pl.INDEX]):
        with pl.reg_scope():
            mask = pl.reg.active_mask(count, lanes=64)
            value = pl.reg.load(src, lanes=64)
            _out = pl.reg.store(src, value=value, mask=mask)

    printed = python_print(kernel, format=False)
    restored = pl.parse(printed)
    ir.assert_structural_equal(kernel, restored, enable_auto_mapping=True)


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


def test_bf16_rounding_and_reduction_broadcast_roundtrip():
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(src: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec]):
        with pl.reg_scope():
            full = pl.reg.full_mask(lanes=64)
            value = pl.reg.load(src, lanes=64)
            bf = pl.reg.convert(value, dtype=pl.BF16, saturate=False)
            widened = pl.reg.convert(bf, dtype=pl.FP32)
            total = pl.reg.reduce_sum(widened, full, reassoc=True)
            broadcast = pl.reg.broadcast(total, lanes=64, dtype=pl.FP32)
            out = pl.reg.store(src, value=broadcast)
        _consumer = pl.tile.add(out, out)

    printed = python_print(kernel, format=False)
    restored = pl.parse(printed)
    ir.assert_structural_equal(kernel, restored, enable_auto_mapping=True)


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


def test_extended_vf_frontend_roundtrip():
    @pl.function(type=pl.FunctionType.AIV)
    def extended(
        src: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec],
        dst: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec],
        amount: pl.Scalar[pl.FP32],
    ):
        with pl.reg_scope():
            x = pl.reg.load(src, lanes=64)
            mask = pl.reg.active_mask(32, lanes=64)
            y = pl.reg.abs(x, mask)
            y = pl.reg.neg(y)
            y = pl.reg.relu(y)
            y = pl.reg.exp(y, mask)
            y = pl.reg.log(y)
            y = pl.reg.sqrt(y)
            y = pl.reg.adds(y, amount, mask)
            y = pl.reg.muls(y, 2.0, mask)
            y = pl.reg.maximums(y, 0.0, mask)
            y = pl.reg.minimums(y, 4.0, mask)
            idx = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
            idx = pl.reg.bitwise_and(idx, idx, mask)
            idx = pl.reg.bitwise_or(idx, idx)
            idx = pl.reg.bitwise_xor(idx, idx)
            idx = pl.reg.bitwise_not(idx)
            idx = pl.reg.shift_left(idx, idx, mask)
            idx = pl.reg.shift_right(idx, idx)
            idx = pl.reg.shift_lefts(idx, 1, mask)
            idx = pl.reg.shift_rights(idx, 1, mask)
            y = pl.reg.gather_lanes(y, idx)
            condition = pl.reg.compares(y, 0.0, mask, predicate="gt")
            selected = pl.reg.select(condition, y, x)
            total = pl.reg.reduce_sum(selected, mask, reassoc=True)
            _max = pl.reg.reduce_max(selected, mask)
            _min = pl.reg.reduce_min(selected, mask)
            out = pl.reg.store(dst, value=total)
        _consumer = pl.tile.add(out, out)

    printed = python_print(extended, format=False)
    restored = pl.parse(printed)
    ir.assert_structural_equal(extended, restored, enable_auto_mapping=True)


def test_index_gen_bf16_rejected_by_frontend():
    # Parse-time type deduction must reject the unsupported VCI dtype.
    with pytest.raises((ValueError, ParserError), match="index_gen.*BF16|BF16.*VCI"):

        @pl.function(type=pl.FunctionType.AIV)
        def kernel():
            with pl.reg_scope():
                _indices = pl.reg.index_gen(0.0, lanes=64, dtype=pl.BF16)


def test_group_scaling_pipeline_roundtrip():
    # The MX shape a V-side quantization kernel needs: a per-32-element amax
    # broadcast back over its own group so each lane can scale itself. The
    # butterfly this replaces cost ten register ops per 64 lanes.
    @pl.function(type=pl.FunctionType.AIV)
    def quantize(
        src: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec],
        dst: pl.InOut[pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec]],
    ):
        with pl.reg_scope():
            value = pl.reg.load(src, lanes=64)
            mask = pl.reg.full_mask(lanes=64)
            magnitude = pl.reg.abs(value, mask)
            amax = pl.reg.reduce_max(magnitude, mask, group_size=32)
            scale = pl.reg.group_broadcast(amax, lanes=64)
            scaled = pl.reg.div(value, scale, mask)
            out = pl.reg.store(dst, value=scaled)
        _consumer = pl.tile.add(out, out)

    printed = python_print(quantize, format=False)
    restored = pl.parse(printed)
    ir.assert_structural_equal(quantize, restored, enable_auto_mapping=True)


@pytest.mark.parametrize("group_size", [8, 16, 32, 64])
def test_grouped_reduction_lane_counts(group_size):
    # 64 lanes over these widths request 8, 4, 2 and 1 groups; all are
    # executable A5 group counts.
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        src: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec],
        dst: pl.InOut[pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec]],
    ):
        with pl.reg_scope():
            x = pl.reg.load(src, lanes=64)
            mask = pl.reg.full_mask(lanes=64)
            reduced = pl.reg.reduce_min(x, mask, group_size=group_size)
            spread = pl.reg.group_broadcast(reduced, lanes=64)
            _out = pl.reg.store(dst, value=spread)

    assert kernel is not None


@pytest.mark.parametrize(
    ("group_size", "message"),
    [
        # 64 / 4 = 16 groups: beyond the 1/2/4/8 the hardware can lay out.
        (4, "1, 2, 4 or 8 groups"),
        # 24 does not divide 64 lanes.
        (24, "divide"),
    ],
)
def test_grouped_reduction_rejects_unexecutable_geometry(group_size, message):
    with pytest.raises((ValueError, ParserError), match=message):

        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            src: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec],
            dst: pl.InOut[pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec]],
        ):
            with pl.reg_scope():
                x = pl.reg.load(src, lanes=64)
                mask = pl.reg.full_mask(lanes=64)
                reduced = pl.reg.reduce_max(x, mask, group_size=group_size)
                _out = pl.reg.store(dst, value=reduced)


def test_group_broadcast_requires_divisible_result():
    with pytest.raises((ValueError, ParserError), match="divide"):

        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            src: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec],
            dst: pl.InOut[pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec]],
        ):
            with pl.reg_scope():
                x = pl.reg.load(src, lanes=64)
                mask = pl.reg.full_mask(lanes=64)
                # Two groups cannot expand into an odd lane count.
                reduced = pl.reg.reduce_max(x, mask, group_size=32)
                spread = pl.reg.group_broadcast(reduced, lanes=63)
                _out = pl.reg.store(dst, value=spread)


@pytest.mark.parametrize("stride", [True, 1.5, "16"])
def test_store_rejects_non_integer_block_stride(stride):
    dst = ir.Var(
        "dst", ir.TileType([2048], DataType.BF16, memory_space=ir.MemorySpace.Vec), ir.Span.unknown()
    )
    value = ir.Var("value", ir.RegType(128, DataType.BF16, ir.RegKind.VecReg), ir.Span.unknown())
    with pytest.raises(TypeError, match="block_stride"):
        pl.reg.store(dst, value=value, block_stride=stride)


def test_block_stride_store_round_trip():
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        src: pl.Tile[[128], pl.BF16, pl.MemorySpace.Vec],
        dst: pl.InOut[pl.Tile[[2048], pl.BF16, pl.MemorySpace.Vec]],
    ):
        with pl.reg_scope():
            x = pl.reg.load(src, lanes=128)
            _out = pl.reg.store(dst, 16, x, block_stride=16)

    printed = python_print(kernel, format=False)
    assert "block_stride=16" in printed
    ir.assert_structural_equal(kernel, pl.parse(printed), enable_auto_mapping=True)


def test_manual_reg_scope_mem_bar_roundtrip():
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(src: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec]):
        with pl.reg_scope(auto_sync=False):
            x = pl.reg.load(src, lanes=64)
            pl.reg.mem_bar("VLD_VST")
            _out = pl.reg.store(src, value=x)

    printed = python_print(kernel, format=False)
    assert "auto_sync=False" in printed
    assert "pl.reg.mem_bar(" in printed
    restored = pl.parse(printed)
    ir.assert_structural_equal(kernel, restored, enable_auto_mapping=True)


@pytest.mark.parametrize("option", ["1", "'false'", "None", "pl.const(0, pl.INT32)"])
def test_reg_auto_sync_rejects_non_bool(option):
    source = f"""
@pl.function(type=pl.FunctionType.InCore)
def kernel():
    with pl.reg_scope(auto_sync={option}):
        pl.reg.mem_bar("VV_ALL")
"""
    with pytest.raises((ParserError, ValueError, TypeError), match="auto_sync|bool"):
        pl.parse(source)


def test_reg_mem_bar_value_rejected():
    with pytest.raises((ParserError, ValueError, TypeError), match="mem_bar|statement|Assign"):

        @pl.function(type=pl.FunctionType.InCore)
        def kernel():
            with pl.reg_scope():
                _value = pl.reg.mem_bar("VV_ALL")


@pytest.mark.parametrize("arguments", ["False", "unknown=False", "auto_sync=True, auto_sync=False"])
def test_reg_auto_sync_rejects_invalid_arguments(arguments):
    source = f"""
@pl.function(type=pl.FunctionType.InCore)
def kernel():
    with pl.reg_scope({arguments}):
        pl.reg.mem_bar("VV_ALL")
"""
    with pytest.raises((ParserError, ValueError, TypeError, SyntaxError), match="auto_sync|keyword"):
        pl.parse(source)


def test_reg_auto_sync_compiletime_bool_and_default_normalization():
    policy = False

    @pl.function(type=pl.FunctionType.InCore)
    def manual():
        with pl.reg_scope(auto_sync=policy):
            pl.reg.mem_bar(kind="VV_ALL")

    assert "auto_sync=False" in python_print(manual, format=False)

    @pl.function(type=pl.FunctionType.InCore)
    def automatic():
        with pl.reg_scope(auto_sync=True):
            pl.reg.mem_bar("VV_ALL")

    assert "with pl.reg_scope():" in python_print(automatic, format=False)


def test_reg_auto_sync_runtime_policy_rejected():
    with pytest.raises((ParserError, ValueError, TypeError), match="auto_sync|compile-time"):
        pl.parse("""
@pl.function(type=pl.FunctionType.InCore)
def kernel(policy: pl.Scalar[pl.BOOL]):
    with pl.reg_scope(auto_sync=policy):
        pl.reg.mem_bar("VV_ALL")
""")


def test_squeeze_store_explicit_state_roundtrip():
    kernel = pl.parse("""
@pl.function(type=pl.FunctionType.AIV)
def kernel(
    src: pl.Tile[[256], pl.UINT32, pl.MemorySpace.Vec],
    dst: pl.Tile[[320], pl.UINT32, pl.MemorySpace.Vec],
):
    with pl.reg_scope():
        init_ureg = pl.reg.squeeze_store_init(dst)
        for i, (curr_ureg,) in pl.range(0, 4, init_values=(init_ureg,)):
            value = pl.reg.load(src, offset=i * 64, lanes=64)
            keep = pl.reg.active_mask(9, lanes=64)
            next_ureg = pl.reg.squeeze_store_step(curr_ureg, dst, value, keep)
            curr_ureg = pl.yield_(next_ureg)
        written = pl.reg.squeeze_store_flush(curr_ureg, dst)
        observed = pl.reg.load(dst, lanes=64)
""")
    printed = python_print(kernel, format=False)
    assert printed.count("pl.reg.squeeze_store_init(") == 1
    assert printed.count("pl.reg.squeeze_store_step(") == 1
    assert printed.count("pl.reg.squeeze_store_flush(") == 1
    assert "pl.reg.squeeze_store_init(dst" in printed
    # Loop arguments and loop results may share a source spelling; the printer
    # disambiguates their SSA names. Check operands without fixing that spelling.
    assert ", dst, value, keep)" in printed
    restored = pl.parse(printed)
    ir.assert_structural_equal(kernel, restored, enable_auto_mapping=True)


def test_squeeze_store_operations_are_public_without_stream_handle():
    assert callable(pl.reg.squeeze_store_init)
    assert callable(pl.reg.squeeze_store_step)
    assert callable(pl.reg.squeeze_store_flush)
    assert not hasattr(pl.reg, "store_stream")
    assert not hasattr(pl.reg, "StoreStream")
    assert list(inspect.signature(pl.reg.squeeze_store_init).parameters) == ["dst", "offset"]
    assert list(inspect.signature(pl.reg.squeeze_store_step).parameters) == [
        "curr_ureg",
        "dst",
        "value",
        "mask",
    ]
    assert list(inspect.signature(pl.reg.squeeze_store_flush).parameters) == ["final_ureg", "dst"]


@pytest.mark.parametrize(
    "shape,view",
    [
        ("[128]", ""),
        ("[128]", ", pl.TileView(stride=[1])"),
        ("[2, 64]", ""),
        ("[2, 64]", ", pl.TileView(stride=[64, 1])"),
        ("[1, 128]", ", pl.TileView(stride=[1024, 1])"),
        ("[128, 1]", ""),
    ],
)
def test_squeeze_store_accepts_flattenable_destination(shape, view):
    # Explicit dense strides and implicit layout describe the same linear
    # address sequence. A singleton axis contributes no physical gap.
    function = pl.parse(f"""
@pl.function(type=pl.FunctionType.AIV)
def kernel(src: pl.Tile[[64], pl.UINT32, pl.MemorySpace.Vec],
           dst: pl.Tile[{shape}, pl.UINT32, pl.MemorySpace.Vec{view}]):
    with pl.reg_scope():
        ureg = pl.reg.squeeze_store_init(dst, offset=7)
        value = pl.reg.load(src, lanes=64)
        keep = pl.reg.active_mask(3, lanes=64)
        ureg = pl.reg.squeeze_store_step(ureg, dst, value, keep)
        written = pl.reg.squeeze_store_flush(ureg, dst)
""")
    ir.assert_structural_equal(
        function, pl.parse(python_print(function, format=False)), enable_auto_mapping=True
    )


@pytest.mark.parametrize(
    "shape,view",
    [
        ("[128]", "pl.TileView(stride=[2])"),
        ("[2, 64]", "pl.TileView(stride=[96, 1])"),
        ("[2, 64]", "pl.TileView(stride=[1, 2], blayout=pl.TileLayout.col_major)"),
        ("[2, 64]", "pl.TileView(pad=pl.PadValue.zero)"),
        ("[2, 64]", "pl.TileView(slayout=pl.TileLayout.row_major)"),
    ],
)
def test_squeeze_store_rejects_non_linear_destination(shape, view):
    with pytest.raises((ParserError, ValueError), match="contiguous"):
        pl.parse(f"""
@pl.function(type=pl.FunctionType.AIV)
def kernel(dst: pl.Tile[{shape}, pl.UINT32, pl.MemorySpace.Vec, {view}]):
    with pl.reg_scope():
        ureg = pl.reg.squeeze_store_init(dst)
        written = pl.reg.squeeze_store_flush(ureg, dst)
""")


@pytest.mark.parametrize(
    "body,message",
    [
        ("other = pl.reg.squeeze_store_init(dst)", "cannot nest"),
        ("observed = pl.reg.load(dst, lanes=64)", "Flush"),
        ("pass", "must be flushed"),
        ("return", "ReturnStmt is not supported"),
        ("if True:\n            written = pl.reg.squeeze_store_flush(ureg, dst)", "conditionals"),
        (
            "first = pl.reg.squeeze_store_step(ureg, dst, value, keep)\n"
            "        second = pl.reg.squeeze_store_step(ureg, dst, value, keep)\n"
            "        written = pl.reg.squeeze_store_flush(second, dst)",
            "consumed exactly once",
        ),
    ],
)
def test_squeeze_store_rejects_invalid_lifecycle(body, message):
    with pytest.raises((ParserError, ValueError), match=message):
        pl.parse(f"""
@pl.function(type=pl.FunctionType.AIV)
def kernel(src: pl.Tile[[64], pl.UINT32, pl.MemorySpace.Vec],
           dst: pl.Tile[[128], pl.UINT32, pl.MemorySpace.Vec]):
    with pl.reg_scope():
        value = pl.reg.load(src, lanes=64)
        keep = pl.reg.active_mask(3, lanes=64)
        ureg = pl.reg.squeeze_store_init(dst)
        {body}
""")


def test_squeeze_store_allows_more_than_four_sequential_chains():
    # The implementation restricts overlapping live chains, not the total
    # number of init/flush pairs in a register scope.
    pairs = "\n".join(
        f"        state{i} = pl.reg.squeeze_store_init(dst, offset={i})\n"
        f"        result{i} = pl.reg.squeeze_store_flush(state{i}, dst)"
        for i in range(5)
    )
    function = pl.parse(f"""
@pl.function(type=pl.FunctionType.AIV)
def kernel(dst: pl.Tile[[128], pl.UINT32, pl.MemorySpace.Vec]):
    with pl.reg_scope():
{pairs}
""")
    assert python_print(function, format=False).count("pl.reg.squeeze_store_init(") == 5


def test_squeeze_store_rejects_add_before_opening_stream():
    with pytest.raises(
        (ParserError, ValueError), match=r"reg\.add is not supported in a RegScope with squeeze-store"
    ):
        pl.parse("""
@pl.function(type=pl.FunctionType.AIV)
def kernel(src: pl.Tile[[64], pl.UINT32, pl.MemorySpace.Vec],
           dst: pl.Tile[[128], pl.UINT32, pl.MemorySpace.Vec]):
    with pl.reg_scope():
        value = pl.reg.load(src, lanes=64)
        added = pl.reg.add(value, value)
        keep = pl.reg.full_mask(lanes=64)
        initial = pl.reg.squeeze_store_init(dst)
        updated = pl.reg.squeeze_store_step(initial, dst, added, keep)
        written = pl.reg.squeeze_store_flush(updated, dst)
""")


@pytest.mark.parametrize("outer_loop", ["parallel", "unroll"])
def test_squeeze_store_rejects_opening_inside_nonsequential_ancestor(outer_loop):
    with pytest.raises((ParserError, ValueError), match="Squeeze-store writes require sequential loops"):
        pl.parse(f"""
@pl.function(type=pl.FunctionType.AIV)
def kernel(src: pl.Tile[[64], pl.UINT32, pl.MemorySpace.Vec],
           dst: pl.Tile[[128], pl.UINT32, pl.MemorySpace.Vec]):
    with pl.reg_scope():
        for i in pl.{outer_loop}(2):
            for j in pl.range(2):
                value = pl.reg.load(src, lanes=64)
                keep = pl.reg.full_mask(lanes=64)
                initial = pl.reg.squeeze_store_init(dst)
                updated = pl.reg.squeeze_store_step(initial, dst, value, keep)
                written = pl.reg.squeeze_store_flush(updated, dst)
""")


@pytest.mark.parametrize("mask_expression", ["pl.reg.full_mask(lanes=64)", "pl.reg.active_mask(3, lanes=64)"])
def test_squeeze_store_rejects_equivalent_stored_producers(mask_expression):
    # Distinct mask variables do not make two identical SQZ producers safe:
    # PTOAS can merge them and lose the second required SQZN push.
    with pytest.raises((ParserError, ValueError), match="common stored SQZ"):
        pl.parse(f"""
@pl.function(type=pl.FunctionType.AIV)
def kernel(src: pl.Tile[[64], pl.UINT32, pl.MemorySpace.Vec],
           dst: pl.Tile[[128], pl.UINT32, pl.MemorySpace.Vec]):
    with pl.reg_scope():
        value = pl.reg.load(src, lanes=64)
        first_mask = {mask_expression}
        second_mask = {mask_expression}
        initial = pl.reg.squeeze_store_init(dst)
        first = pl.reg.squeeze_store_step(initial, dst, value, first_mask)
        second = pl.reg.squeeze_store_step(first, dst, value, second_mask)
        written = pl.reg.squeeze_store_flush(second, dst)
""")


@pytest.mark.parametrize("source", ["outside", "carry", "empty_nested_loop"])
def test_squeeze_store_rejects_loop_invariant_producers(source):
    carried = source != "outside"
    loop_args = "current, value" if carried else "current,"
    initial_values = "initial, value" if carried else "initial,"
    yield_names = "final, _last" if carried else "final"
    yield_values = "updated, value" if carried else "updated"
    inner_loop = ""
    if source == "empty_nested_loop":
        inner_loop = """
            for j, (_cached,) in pl.range(0, 0, init_values=(value,)):
                loaded = pl.reg.load(src, lanes=64)
                nested_value = pl.yield_(loaded)"""
        yield_values = "updated, nested_value"
    with pytest.raises((ParserError, ValueError), match="loop-invariant"):
        pl.parse(f"""
@pl.function(type=pl.FunctionType.AIV)
def kernel(src: pl.Tile[[64], pl.UINT32, pl.MemorySpace.Vec],
           dst: pl.Tile[[128], pl.UINT32, pl.MemorySpace.Vec]):
    with pl.reg_scope():
        value = pl.reg.load(src, lanes=64)
        keep = pl.reg.active_mask(3, lanes=64)
        initial = pl.reg.squeeze_store_init(dst)
        for i, ({loop_args}) in pl.range(0, 4, init_values=({initial_values})):
            updated = pl.reg.squeeze_store_step(current, dst, value, keep){inner_loop}
            {yield_names} = pl.yield_({yield_values})
        written = pl.reg.squeeze_store_flush(final, dst)
""")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
