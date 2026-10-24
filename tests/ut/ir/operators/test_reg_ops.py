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
    ],
)
@pytest.mark.parametrize("name", ["add", "sub", "mul", "div", "minimum", "maximum", "compare", "select"])
def test_dtype_matrix(name, dtype):
    v = vec(dtype=dtype)
    rejected = (
        (name == "mul" and dtype in (DataType.INT8, DataType.UINT8))
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
