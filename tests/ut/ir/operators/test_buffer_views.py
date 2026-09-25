# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static views alias explicit Buffer storage through typed SSA results."""

import pytest
from pypto import DataType, ir
from pypto.pypto_core import ir as _ir

_SPAN = ir.Span.unknown()


def _descriptor(shape=(128, 32), dtype=DataType.UINT8, **kwargs):
    return ir.BufferType(list(shape), dtype, ir.MemorySpace.Vec, **kwargs)


def _var(type_=None):
    return ir.Var("source", type_ or _descriptor(), _SPAN)


def _offsets(row=0, col=0, dtype=DataType.INDEX):
    return ir.MakeTuple([ir.ConstInt(row, dtype, _SPAN), ir.ConstInt(col, dtype, _SPAN)], _SPAN)


def _view(name, args, type_, **kwargs):
    return _ir._create_internal_op_call(name, args, kwargs, type_, _SPAN)


@pytest.mark.parametrize("name", ["buffer.subview", "buffer.reshape"])
def test_view_contract_is_an_alias_with_no_data_access(name):
    effect = ir.get_op_buffer_arg_effect(name, 0)
    assert effect.data == ir.BufferAccess.None_
    assert effect.metadata == ir.BufferAccess.Read
    result = ir.get_op_buffer_result_spec(name)
    assert result.behavior == ir.BufferResultBehavior.Alias
    assert result.alias_arg == 0
    assert ir.get_op_ir_stage(name) == ir.OpIRStage.Buffer
    assert ir.get_op_output_arity(name) == 1
    if name == "buffer.subview":
        assert ir.get_op_buffer_arg_effect(name, 1).non_memory


@pytest.mark.parametrize(
    "dtype", [DataType.FP16, DataType.BF16, DataType.FP32, DataType.INT16, DataType.INT32, DataType.UINT8]
)
def test_static_window_can_be_reinterpreted_without_equal_element_counts(dtype):
    window = _view("buffer.subview", [_var(), _offsets(64)], _descriptor((64, 32)))
    shape = {
        DataType.FP16: (32, 32),
        DataType.BF16: (32, 32),
        DataType.FP32: (16, 32),
        DataType.INT16: (32, 32),
        DataType.INT32: (16, 32),
        DataType.UINT8: (32, 64),
    }[dtype]
    descriptor = _descriptor(shape, dtype)
    call = _view("buffer.reshape", [ir.Var("window", window.type, _SPAN)], descriptor)
    ir.assert_structural_equal(call.type, descriptor)
    assert call.kwargs == {}


@pytest.mark.parametrize("row,col", [(-1, 0), (65, 0), (0, 1)])
def test_subview_rejects_out_of_bounds_or_noncontiguous_windows(row, col):
    with pytest.raises(ValueError, match="static INDEX offsets|exceeds source capacity"):
        _view("buffer.subview", [_var(), _offsets(row, col)], _descriptor((64, 32)))


def test_subview_offsets_are_a_rank_2_tuple_of_integer_scalars():
    dynamic = ir.Var("row", ir.ScalarType(DataType.INDEX), _SPAN)
    runtime = ir.MakeTuple([dynamic, ir.ConstInt(0, DataType.INDEX, _SPAN)], _SPAN)
    for accepted in (runtime, _offsets(dtype=DataType.INT64)):
        ir.assert_structural_equal(
            _view("buffer.subview", [_var(), accepted], _descriptor((64, 32))).type, _descriptor((64, 32))
        )
    fractional = ir.MakeTuple(
        [ir.ConstFloat(0.0, DataType.FP32, _SPAN), ir.ConstInt(0, DataType.INDEX, _SPAN)], _SPAN
    )
    for invalid, message in ((ir.MakeTuple([], _SPAN), "rank-2 MakeTuple"), (fractional, "integer or INDEX")):
        with pytest.raises(ValueError, match=message):
            _view("buffer.subview", [_var(), invalid], _descriptor((64, 32)))


@pytest.mark.parametrize(
    "descriptor,message",
    [
        (_descriptor((32, 64)), "exceeds the source"),
        (_descriptor((16, 32), DataType.FP32), "keep the element type"),
    ],
)
def test_subview_result_keeps_the_element_type_and_fits_the_source(descriptor, message):
    with pytest.raises(ValueError, match=message):
        _view("buffer.subview", [_var(), _offsets()], descriptor)


def test_subview_result_may_have_a_narrower_static_valid_window():
    narrowed = _descriptor((64, 32), valid_shape=[32, 32])
    ir.assert_structural_equal(_view("buffer.subview", [_var(), _offsets()], narrowed).type, narrowed)


@pytest.mark.parametrize(
    "descriptor,message",
    [
        (_descriptor((8, 32), DataType.FP32), "equal physical byte sizes"),
        (_descriptor((8, 32), DataType.INT64), "dense rank-2"),
        (_descriptor((16, 32), DataType.FP32, valid_shape=[-1, 32]), "static valid extents"),
        (_descriptor((16, 32), DataType.FP32, blayout=ir.TileLayout.col_major), "dense rank-2"),
    ],
)
def test_reshape_rejects_unresolved_or_incompatible_descriptors(descriptor, message):
    with pytest.raises(ValueError, match=message):
        _view("buffer.reshape", [_var(_descriptor((64, 32)))], descriptor)


def test_reshape_rejects_unaligned_rows_even_when_total_byte_size_matches():
    source = _var(_descriptor((8, 32)))
    with pytest.raises(ValueError, match="physical rows aligned to 32 bytes"):
        _view("buffer.reshape", [source], _descriptor((16, 4), DataType.FP32))


@pytest.mark.parametrize("name", ["buffer.subview", "buffer.reshape"])
def test_view_arity_and_kwargs_are_not_dropped(name):
    args: list[ir.Expr] = [_var()]
    descriptor = _descriptor()
    if name == "buffer.subview":
        args.append(_offsets())
    # buffer.subview also takes an optional valid tuple, so a fourth operand is the surplus one.
    surplus = [*args, _offsets(), _offsets()] if name == "buffer.subview" else [*args, _offsets()]
    for invalid in (args[:-1], surplus):
        with pytest.raises(ValueError, match="requires"):
            _view(name, invalid, descriptor)
    with pytest.raises(ValueError, match="Unknown kwarg"):
        _view(name, args, descriptor, byte_offset=32)
    with pytest.raises(ValueError, match="internal-only"):
        ir.create_op_call(name, args, _SPAN)


def _fp32(shape, **kwargs):
    return _descriptor(shape, DataType.FP32, **kwargs)


def test_subview_is_a_strided_window_that_keeps_the_source_pitch():
    # A [16, 32] column window of a [64, 64] tile at a static offset.
    call = _view("buffer.subview", [_var(_fp32((64, 64))), _offsets(16, 32)], _fp32((16, 32)))
    ir.assert_structural_equal(call.type, _fp32((16, 32)))
    with pytest.raises(ValueError, match="exceeds source capacity on dimension 1"):
        _view("buffer.subview", [_var(_fp32((64, 64))), _offsets(16, 40)], _fp32((16, 32)))


def test_subview_dynamic_valid_result_takes_every_extent_from_the_valid_tuple():
    rows = ir.Var("rows", ir.ScalarType(DataType.INDEX), _SPAN)
    dynamic = _fp32((16, 64), valid_shape=[-1, 64])
    with pytest.raises(ValueError, match="requires a valid-extents tuple"):
        _view("buffer.subview", [_var(_fp32((64, 64))), _offsets()], dynamic)
    valid = ir.MakeTuple([rows, ir.ConstInt(64, DataType.INDEX, _SPAN)], _SPAN)
    call = _view("buffer.subview", [_var(_fp32((64, 64))), _offsets(), valid], dynamic)
    assert call.args[2].same_as(valid)
    mismatched = ir.MakeTuple([rows, ir.ConstInt(32, DataType.INDEX, _SPAN)], _SPAN)
    with pytest.raises(ValueError, match="must equal the static descriptor dimension"):
        _view("buffer.subview", [_var(_fp32((64, 64))), _offsets(), mismatched], dynamic)


def test_subview_rejects_non_vec_or_non_dense_sources():
    mat = ir.BufferType(
        [64, 64],
        DataType.FP32,
        ir.MemorySpace.Mat,
        [64, 64],
        ir.TileLayout.col_major,
        ir.TileLayout.row_major,
    )
    with pytest.raises(ValueError, match="dense rank-2 row-major Vec"):
        _view("buffer.subview", [_var(mat), _offsets()], _fp32((16, 64)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
