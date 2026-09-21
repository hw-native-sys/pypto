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


def test_subview_requires_exact_static_index_tuple():
    dynamic = ir.Var("row", ir.ScalarType(DataType.INDEX), _SPAN)
    offsets = ir.MakeTuple([dynamic, ir.ConstInt(0, DataType.INDEX, _SPAN)], _SPAN)
    for invalid in (offsets, _offsets(dtype=DataType.INT64), ir.MakeTuple([], _SPAN)):
        with pytest.raises(ValueError, match="static INDEX offsets|rank-2 MakeTuple"):
            _view("buffer.subview", [_var(), invalid], _descriptor((64, 32)))


@pytest.mark.parametrize(
    "descriptor",
    [
        _descriptor((32, 64)),
        _descriptor((16, 32), DataType.FP32),
        _descriptor((64, 32), valid_shape=[32, 32]),
    ],
)
def test_subview_result_remains_a_full_valid_byte_carrier(descriptor):
    with pytest.raises(ValueError, match="full-valid UINT8"):
        _view("buffer.subview", [_var(), _offsets()], descriptor)


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
    for invalid in (args[:-1], [*args, _offsets()]):
        with pytest.raises(ValueError, match="requires"):
            _view(name, invalid, descriptor)
    with pytest.raises(ValueError, match="Unknown kwarg"):
        _view(name, args, descriptor, byte_offset=32)
    with pytest.raises(ValueError, match="internal-only"):
        ir.create_op_call(name, args, _SPAN)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
