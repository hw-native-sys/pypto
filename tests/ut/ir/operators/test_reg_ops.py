# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""P0 register schemas validate independently of lowering and hardware."""

import pytest
from pypto import DataType, ir

SPAN = ir.Span.unknown()


def integer(value):
    return ir.ConstInt(value, DataType.INDEX, SPAN)


def typed_integer(value, dtype):
    return ir.ConstInt(value, dtype, SPAN)


def vec(lanes=64, dtype=DataType.FP32):
    return ir.Var("v", ir.RegType(lanes, dtype, ir.RegKind.VecReg), SPAN)


def mask(lanes=64):
    return ir.Var("m", ir.RegType(lanes, DataType.BOOL, ir.RegKind.MaskReg), SPAN)


def tile(size=128, memory=ir.MemorySpace.Vec, view=None):
    return ir.Var("t", ir.TileType([size], DataType.FP32, tile_view=view, memory_space=memory), SPAN)


def call(name, args=(), **kwargs):
    return ir.create_op_call(f"reg.{name}", list(args), kwargs, SPAN)


def reg_type(expr: ir.Expr) -> ir.RegType:
    assert isinstance(expr.type, ir.RegType)
    return expr.type


def test_load_store_type_alias_and_effect():
    src = tile()
    loaded = call("load", [src, integer(8)], lanes=64)
    assert reg_type(loaded).lanes == 64
    stored = call("store", [src, integer(8), loaded])
    ir.assert_structural_equal(stored.type, src.type)
    assert ir.get_reg_op_spec("reg.store")["output_reuses_input_arg"] == 0
    assert ir.get_op_arg_effect("reg.store", 0) == ir.ArgEffect.Write
    effect = ir.get_reg_memory_effect(stored)
    assert effect["target"] is src
    assert effect["begin"] == 8 and effect["end"] == 72
    assert effect["access"] == ir.ArgEffect.Write
    assert effect["mask"] is None


@pytest.mark.parametrize("name", ["add", "sub", "mul", "div", "minimum", "maximum"])
def test_binary_schema(name):
    result = call(name, [vec(), vec(), mask()], pmode="zero")
    assert reg_type(result).lanes == 64
    with pytest.raises(ValueError, match="lane|shape"):
        call(name, [vec(), vec(32)])
    with pytest.raises(ValueError, match="dtype"):
        call(name, [vec(), vec(dtype=DataType.INT32)])
    with pytest.raises(ValueError, match="VecReg"):
        call(name, [vec(), tile()])
    with pytest.raises(ValueError, match="pmode"):
        call(name, [vec(), vec()], pmode="merge")


def test_mask_compare_select_convert_and_generators():
    full = call("full_mask", lanes=64)
    active = call("active_mask", [integer(32)], lanes=64)
    for name in ("mask_and", "mask_or", "mask_xor"):
        assert reg_type(call(name, [full, active])).kind == ir.RegKind.MaskReg
    assert reg_type(call("mask_not", [active])).kind == ir.RegKind.MaskReg
    compared = call("compare", [vec(), vec(), full], predicate="lt")
    assert reg_type(compared).kind == ir.RegKind.MaskReg
    assert reg_type(call("select", [compared, vec(), vec()])).dtype == DataType.FP32
    assert (
        reg_type(call("convert", [vec()], dtype=DataType.FP16, rounding="R", saturate=False)).dtype
        == DataType.FP16
    )
    assert (
        reg_type(call("broadcast", [typed_integer(2, DataType.INT32)], lanes=64, dtype=DataType.INT32)).lanes
        == 64
    )
    assert (
        reg_type(
            call(
                "index_gen",
                [typed_integer(0, DataType.INT32)],
                lanes=64,
                dtype=DataType.INT32,
                order="ascending",
            )
        ).lanes
        == 64
    )
    with pytest.raises(ValueError, match="dtype"):
        call("broadcast", [integer(2)], lanes=64, dtype=DataType.INT32)
    with pytest.raises(ValueError, match="dtype"):
        call(
            "index_gen",
            [ir.ConstFloat(0.5, DataType.FP32, SPAN)],
            lanes=64,
            dtype=DataType.INT32,
            order="ascending",
        )
    with pytest.raises(ValueError, match="active_lanes"):
        call("active_mask", [integer(65)], lanes=64)
    with pytest.raises(ValueError, match="predicate"):
        call("compare", [vec(), vec(), full], predicate="bad")


@pytest.mark.parametrize("offset,lanes", [(-1, 64), (100, 64), (0, 0)])
def test_access_bounds(offset, lanes):
    with pytest.raises(ValueError):
        call("load", [tile(), integer(offset)], lanes=lanes)


@pytest.mark.parametrize("dynamic_valid", [False, True])
def test_dynamic_memory_operands_pass_through(dynamic_valid):
    dynamic = ir.Var("dynamic", ir.ScalarType(DataType.INDEX), SPAN)
    view = ir.TileView([dynamic if dynamic_valid else 32], [1], integer(0))
    # Physical storage remains contiguous. Dynamic validity or access offsets
    # are user-managed; only fully static intervals are checked for bounds.
    target = tile(view=view)
    offset = integer(0) if dynamic_valid else dynamic
    call("load", [target, offset], lanes=64)
    call("store", [target, offset, vec()])


def test_reject_nonvec_and_noncontiguous_and_uninitialized():
    with pytest.raises(ValueError, match="Vec"):
        call("load", [tile(memory=ir.MemorySpace.Mat), integer(0)], lanes=64)
    view = ir.TileView([128], [2], integer(0))
    with pytest.raises(ValueError, match="contiguous"):
        call("load", [tile(view=view), integer(0)], lanes=64)
    view = ir.TileView([32], [1], integer(0))
    with pytest.raises(ValueError, match="initializ|valid"):
        call("load", [tile(view=view), integer(0)], lanes=64)
    # A mask never relaxes destination capacity.
    with pytest.raises(ValueError, match="capacity|bounds"):
        call("store", [tile(32), integer(0), vec(), mask()])


@pytest.mark.parametrize(
    "dtype",
    [
        DataType.FP16,
        DataType.BF16,
        DataType.FP32,
        DataType.INT8,
        DataType.UINT8,
        DataType.INT16,
        DataType.UINT16,
        DataType.INT32,
        DataType.UINT32,
        DataType.INT64,
        DataType.UINT64,
        DataType.INT4,
        DataType.FP8E4M3FN,
        DataType.INDEX,
    ],
)
@pytest.mark.parametrize("name", ["add", "sub", "mul", "div", "minimum", "maximum", "compare", "select"])
def test_dtype_matrix(name, dtype):
    v = vec(dtype=dtype)
    rejected = (
        dtype
        not in {
            DataType.FP16,
            DataType.BF16,
            DataType.FP32,
            DataType.INT8,
            DataType.UINT8,
            DataType.INT16,
            DataType.UINT16,
            DataType.INT32,
            DataType.UINT32,
        }
        or (name == "mul" and dtype in (DataType.INT8, DataType.UINT8))
        or (name == "div" and dtype not in (DataType.FP16, DataType.FP32))
        or (
            name == "compare" and dtype not in (DataType.FP16, DataType.FP32, DataType.INT32, DataType.UINT32)
        )
        or (name == "select" and dtype == DataType.BF16)
    )

    def construct():
        if name == "compare":
            return call(name, [v, v, mask()], predicate="eq")
        if name == "select":
            return call(name, [mask(), v, v])
        return call(name, [v, v])

    if rejected:
        with pytest.raises(ValueError, match="dtype|FP16|BF16"):
            construct()
    else:
        assert reg_type(construct()).lanes == 64


@pytest.mark.parametrize(
    "name,args",
    [
        ("mask_and", [mask(), vec()]),
        ("mask_or", [mask(), mask(32)]),
        ("mask_xor", [vec(), vec()]),
        ("mask_not", [vec()]),
        ("compare", [vec(), vec(), mask(32)]),
        ("select", [mask(32), vec(), vec()]),
        ("store", [tile(), integer(0), vec(), mask(32)]),
    ],
)
def test_mask_kind_lane_rejections(name, args):
    with pytest.raises(ValueError):
        call(name, args, **({"predicate": "eq"} if name == "compare" else {}))


@pytest.mark.parametrize(
    "src,dst,attrs",
    [
        (DataType.FP16, DataType.FP32, {}),
        (DataType.FP32, DataType.FP16, {"rounding": "R", "saturate": False}),
        (DataType.FP32, DataType.INT32, {"rounding": "Z", "saturate": True}),
        (DataType.INT32, DataType.FP32, {}),
        (DataType.INT8, DataType.FP16, {}),
        (DataType.INT8, DataType.INT16, {}),
        (DataType.INT16, DataType.INT32, {}),
        (DataType.UINT8, DataType.UINT16, {}),
        (DataType.UINT16, DataType.UINT32, {}),
        (DataType.INT32, DataType.INT8, {"saturate": False}),
    ],
)
def test_convert_whitelist(src, dst, attrs):
    assert reg_type(call("convert", [vec(dtype=src)], dtype=dst, **attrs)).dtype == dst


@pytest.mark.parametrize(
    "src,dst,attrs",
    [
        (DataType.BF16, DataType.FP32, {}),
        (DataType.UINT32, DataType.FP32, {}),
        (DataType.INT32, DataType.UINT32, {}),
        (DataType.FP32, DataType.FP16, {"rounding": "R"}),
        (DataType.FP32, DataType.FP16, {"rounding": "F", "saturate": False}),
        (DataType.FP16, DataType.FP32, {"rounding": "R"}),
        (DataType.INT32, DataType.INT8, {"saturate": True}),
        (DataType.INT32, DataType.FP32, {"saturate": False}),
    ],
)
def test_convert_rejects_unapproved_pairs_and_attributes(src, dst, attrs):
    with pytest.raises(ValueError):
        call("convert", [vec(dtype=src)], dtype=dst, **attrs)


def test_effect_snapshot_view_root_mask_and_unknown_interval():
    base = ir.Var("base", ir.PtrType(), SPAN)
    memref = ir.MemRef(base, 64, 1024, SPAN)
    view = ir.TileView([128], [1], integer(16))
    target = ir.Var("dst", ir.TileType([128], DataType.FP32, memref, view, ir.MemorySpace.Vec), SPAN)
    m = mask()
    effect = ir.get_reg_memory_effect(call("store", [target, integer(8), vec(), m]))
    assert effect["root"] is base
    assert (effect["begin"], effect["end"]) == (40, 104)
    assert effect["mask"] is m
    dynamic = ir.Var("offset", ir.ScalarType(DataType.INDEX), SPAN)
    effect = ir.get_reg_memory_effect(call("load", [target, dynamic], lanes=64))
    assert effect["access"] == ir.ArgEffect.Read
    assert effect["begin"] is None and effect["end"] is None


def test_schema_fixture_declares_scope_and_future_emitter():
    for name in ("load", "store", "add", "full_mask", "mask_not", "convert"):
        spec = ir.get_reg_op_spec(f"reg.{name}")
        assert spec["requires_reg_scope"] is True
        assert spec["emitter"].startswith("pto.vmi.")


@pytest.mark.parametrize(
    "name,args",
    [("store", [tile(), integer(0), vec()]), ("mask_not", [mask()]), ("select", [mask(), vec(), vec()])],
)
def test_ops_without_attributes_reject_unknown_kwargs(name, args):
    with pytest.raises(ValueError, match="pmode|keyword|attribute"):
        call(name, args, pmode="zero")


def test_index_gen_rejects_bf16_during_type_deduction():
    # This must fail at construction, without a backend or PTOCodegen instance.
    with pytest.raises(ValueError, match="index_gen.*BF16|BF16.*VCI"):
        call("index_gen", [ir.ConstFloat(0.0, DataType.BF16, SPAN)], lanes=64, dtype=DataType.BF16)


_INTEGER_DTYPES = {
    DataType.INT8,
    DataType.UINT8,
    DataType.INT16,
    DataType.UINT16,
    DataType.INT32,
    DataType.UINT32,
}
_FLOAT_DTYPES = {DataType.FP16, DataType.FP32}
_ORDINARY_DTYPES = _INTEGER_DTYPES | _FLOAT_DTYPES
_WIDE_DTYPES = _ORDINARY_DTYPES - {DataType.INT8, DataType.UINT8}
_COMPUTE_DTYPES = {
    "abs": _FLOAT_DTYPES | {DataType.INT8, DataType.INT16, DataType.INT32},
    "neg": _ORDINARY_DTYPES,
    "relu": _FLOAT_DTYPES | {DataType.INT32},
    "exp": _FLOAT_DTYPES,
    "log": _FLOAT_DTYPES,
    "sqrt": _FLOAT_DTYPES,
    "bitwise_not": _INTEGER_DTYPES,
    "bitwise_and": _INTEGER_DTYPES,
    "bitwise_or": _INTEGER_DTYPES,
    "bitwise_xor": _INTEGER_DTYPES,
    "shift_left": _INTEGER_DTYPES,
    "shift_right": _INTEGER_DTYPES,
    "adds": _ORDINARY_DTYPES,
    "muls": _WIDE_DTYPES,
    "maximums": _ORDINARY_DTYPES,
    "minimums": _ORDINARY_DTYPES,
    "shift_lefts": _INTEGER_DTYPES,
    "shift_rights": _INTEGER_DTYPES,
    "compares": _FLOAT_DTYPES | {DataType.INT32, DataType.UINT32},
    "gather_lanes": _ORDINARY_DTYPES,
    "reduce_sum": _WIDE_DTYPES,
    "reduce_max": _WIDE_DTYPES,
    "reduce_min": _WIDE_DTYPES,
}
_DTYPE_CANDIDATES = [
    DataType.FP16,
    DataType.BF16,
    DataType.FP32,
    DataType.INT8,
    DataType.UINT8,
    DataType.INT16,
    DataType.UINT16,
    DataType.INT32,
    DataType.UINT32,
    DataType.INT64,
    DataType.UINT64,
    DataType.INT4,
    DataType.FP8E4M3FN,
    DataType.INDEX,
]


def _construct_compute(name, dtype):
    value = vec(dtype=dtype)
    m = mask()
    if name in ("abs", "neg", "relu", "exp", "log", "sqrt", "bitwise_not"):
        return call(name, [value, m])
    if name in ("bitwise_and", "bitwise_or", "bitwise_xor", "shift_left", "shift_right"):
        return call(name, [value, value, m])
    if name.startswith("reduce_"):
        return call(name, [value, m], **({"reassoc": True} if name == "reduce_sum" else {}))
    if name == "gather_lanes":
        index_dtype = {8: DataType.UINT8, 16: DataType.UINT16, 32: DataType.UINT32}.get(
            dtype.get_bit(), DataType.UINT32
        )
        return call(name, [value, vec(dtype=index_dtype)])
    scalar_dtype = DataType.INT16 if name in ("shift_lefts", "shift_rights") else dtype
    scalar = ir.Var("scalar", ir.ScalarType(scalar_dtype), SPAN)
    return call(name, [value, scalar, m], **({"predicate": "eq"} if name == "compares" else {}))


@pytest.mark.parametrize("dtype", _DTYPE_CANDIDATES)
@pytest.mark.parametrize("name", _COMPUTE_DTYPES)
def test_compute_dtype_matrix_before_codegen(name, dtype):
    # Exercise the registry directly: unsupported inputs must never need codegen.
    if dtype not in _COMPUTE_DTYPES[name]:
        with pytest.raises(ValueError, match="dtype|type|FP16|BF16"):
            _construct_compute(name, dtype)
    else:
        result = _construct_compute(name, dtype)
        expected_lanes = 1 if name.startswith("reduce_") else 64
        assert reg_type(result).lanes == expected_lanes
        assert reg_type(result).dtype == (DataType.BOOL if name == "compares" else dtype)


@pytest.mark.parametrize("name", ["load", "store", "broadcast", "index_gen"])
@pytest.mark.parametrize("dtype", _DTYPE_CANDIDATES)
def test_memory_generator_dtype_matrix_before_codegen(name, dtype):
    def construct():
        if name in ("load", "store"):
            target = ir.Var("tile", ir.TileType([128], dtype, memory_space=ir.MemorySpace.Vec), SPAN)
            args = [target, integer(0)]
            if name == "store":
                args.append(vec(dtype=dtype))
            return call(name, args, **({"lanes": 64} if name == "load" else {}))
        scalar = ir.Var("scalar", ir.ScalarType(dtype), SPAN)
        return call(name, [scalar], lanes=64, dtype=dtype)

    supported = _ORDINARY_DTYPES | ({DataType.BF16} if name != "index_gen" else set())
    if dtype not in supported:
        with pytest.raises(ValueError, match="dtype|type|BF16"):
            construct()
    else:
        result_type = construct().type
        assert isinstance(result_type, (ir.RegType, ir.TileType))
        assert result_type.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
