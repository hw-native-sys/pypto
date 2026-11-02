# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type and effect contracts for fused, paired and indexed register operations."""

import pypto.language as pl
import pytest
from pypto import DataType, ir

SPAN = ir.Span.unknown()


def vec(dtype=DataType.FP32, lanes=64):
    return ir.Var("v", ir.RegType(lanes, dtype, ir.RegKind.VecReg), SPAN)


def mask(lanes=64):
    return ir.Var("m", ir.RegType(lanes, DataType.BOOL, ir.RegKind.MaskReg), SPAN)


def tile(dtype=DataType.FP32, size=128, memory=ir.MemorySpace.Vec):
    return ir.Var("t", ir.TileType([size], dtype, memory_space=memory), SPAN)


def call(name, args, **kwargs):
    return ir.create_op_call(f"reg.{name}", args, kwargs, SPAN)


@pytest.mark.parametrize("name", ["expdif", "axpy", "mula", "lrelu", "prelu"])
@pytest.mark.parametrize("dtype", [DataType.FP16, DataType.FP32])
def test_fused_types_and_literal_normalization(name, dtype):
    x, m = vec(dtype), mask()
    if name == "axpy":
        result = pl.reg.axpy(x, x, 0.5, m)
    elif name == "lrelu":
        result = pl.reg.lrelu(x, 0.25, m)
    elif name == "mula":
        result = pl.reg.mula(x, x, x, m)
    else:
        result = getattr(pl.reg, name)(x, x, m)
    expr = result.unwrap()
    assert isinstance(expr, ir.Call) and isinstance(expr.type, ir.RegType)
    assert expr.type.dtype == (DataType.FP32 if name == "expdif" else dtype)
    assert expr.type.lanes == 64
    assert ir.get_reg_op_spec(f"reg.{name}")["emitter"] == f"pto.vmi.v{name}"
    if name in ("axpy", "lrelu"):
        assert isinstance(expr.args[-2].type, ir.ScalarType)
        assert expr.args[-2].type.dtype == dtype


@pytest.mark.parametrize("name", ["expdif", "prelu", "mula"])
def test_fused_rejects_mismatches(name):
    args = [vec(), vec(DataType.FP16)]
    if name == "mula":
        args.append(vec())
    with pytest.raises(ValueError, match="dtype"):
        call(name, [*args, mask()])
    args = [vec(), vec()] + ([vec()] if name == "mula" else [])
    with pytest.raises(ValueError, match="Mask lane"):
        call(name, [*args, mask(32)])
    with pytest.raises(ValueError, match="pmode"):
        call(name, [*args, mask()], pmode="merge")


def test_fused_scalar_ssa_is_not_implicitly_cast():
    scalar = ir.Var("alpha", ir.ScalarType(DataType.FP32), SPAN)
    with pytest.raises(ValueError, match="Scalar dtype"):
        pl.reg.axpy(vec(DataType.FP16), vec(DataType.FP16), scalar, mask())
    with pytest.raises(ValueError, match="FP16/FP32"):
        pl.reg.prelu(vec(DataType.INT32), vec(DataType.INT32), mask())
    with pytest.raises(ValueError, match="16/32-bit"):
        pl.reg.mula(vec(DataType.INT8), vec(DataType.INT8), vec(DataType.INT8))
    result = pl.reg.mula(vec(DataType.UINT32), vec(DataType.UINT32), vec(DataType.UINT32)).unwrap()
    assert isinstance(result.type, ir.RegType)
    assert result.type.dtype == DataType.UINT32


@pytest.mark.parametrize("name", ["mull", "addc", "subc", "addcs", "subcs", "intlv", "dintlv"])
@pytest.mark.parametrize("dtype", [DataType.INT32, DataType.UINT32])
def test_paired_types_and_shared_call(name, dtype):
    args = [vec(dtype), vec(dtype), mask()]
    if name in ("addcs", "subcs"):
        args.append(mask())
    first, second = getattr(pl.reg, name)(*args)
    lhs, rhs = first.unwrap(), second.unwrap()
    assert isinstance(lhs, ir.TupleGetItemExpr) and isinstance(rhs, ir.TupleGetItemExpr)
    assert isinstance(lhs.type, ir.RegType) and isinstance(rhs.type, ir.RegType)
    assert lhs.tuple is rhs.tuple  # A single operation, not duplicated per result.
    assert lhs.index == 0 and rhs.index == 1
    assert lhs.type.dtype == dtype
    carry = name in ("addc", "subc", "addcs", "subcs")
    assert rhs.type.kind == (ir.RegKind.MaskReg if carry else ir.RegKind.VecReg)
    assert rhs.type.dtype == (DataType.BOOL if carry else dtype)
    with pytest.raises(ValueError, match="Mask lane"):
        call(name, [*args[:-1], mask(32)])


@pytest.mark.parametrize("name", ["mull", "addc", "subc", "addcs", "subcs"])
def test_integer_pairs_reject_float(name):
    args = [vec(), vec(), mask()] + ([mask()] if name.endswith("cs") else [])
    with pytest.raises(ValueError, match="INT32/UINT32"):
        call(name, args)


def test_paired_lane_restrictions():
    with pytest.raises(ValueError, match="64, 128, or 256"):
        pl.reg.mull(vec(DataType.INT32, 8), vec(DataType.INT32, 8), mask(8))
    for name in ("intlv", "dintlv"):
        with pytest.raises(ValueError, match="even"):
            call(name, [vec(lanes=3), vec(lanes=3), mask(3)])
    with pytest.raises(ValueError, match="Mask lane"):
        pl.reg.subcs(vec(DataType.INT32), vec(DataType.INT32), mask(32), mask())


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (DataType.FP32, DataType.FP32),
        (DataType.FP16, DataType.FP16),
        (DataType.INT8, DataType.INT16),
        (DataType.UINT8, DataType.UINT16),
    ],
)
def test_gather_types_and_unknown_effect(dtype, expected):
    indices = vec(DataType.UINT32 if dtype == DataType.FP32 else DataType.UINT16)
    src, m = tile(dtype, size=1), mask()
    # Repeated indices permit more lanes than elements in the source Tile.
    result = pl.reg.gather(src, indices, m).unwrap()
    assert isinstance(result, ir.Call) and isinstance(result.type, ir.RegType)
    assert result.type.dtype == expected
    effect = ir.get_reg_memory_effect(result)
    assert effect["access"] == ir.ArgEffect.Read
    assert effect["begin"] is None and effect["end"] is None
    assert effect["mask"] is m


def test_scatter_alias_and_effect():
    dst, m = tile(), mask()
    result = pl.reg.scatter(dst, vec(DataType.UINT32), vec(), m).unwrap()
    assert isinstance(result, ir.Call)
    ir.assert_structural_equal(result.type, dst.type)
    assert ir.get_reg_op_spec("reg.scatter")["output_reuses_input_arg"] == 0
    assert ir.get_op_arg_effect("reg.scatter", 0) == ir.ArgEffect.Write
    effect = ir.get_reg_memory_effect(result)
    assert effect["access"] == ir.ArgEffect.Write
    assert effect["begin"] is None and effect["end"] is None
    assert effect["target"] is dst and effect["mask"] is m


def test_indexed_memory_rejects_invalid_contracts():
    with pytest.raises(ValueError, match="UINT16/UINT32"):
        pl.reg.gather(tile(), vec(DataType.INT32), mask())
    with pytest.raises(ValueError, match="UINT16 offsets"):
        pl.reg.gather(tile(), vec(DataType.UINT16), mask())
    with pytest.raises(ValueError, match="Vec UB"):
        pl.reg.gather(tile(memory=ir.MemorySpace.Mat), vec(DataType.UINT32), mask())
    with pytest.raises(ValueError, match="Scatter dtype"):
        pl.reg.scatter(tile(), vec(DataType.UINT32), vec(DataType.INT32), mask())
    with pytest.raises(ValueError, match="lane"):
        pl.reg.scatter(tile(), vec(DataType.UINT32), vec(lanes=32), mask())
    with pytest.raises(ValueError, match="expects"):
        call("scatter", [tile()])
    with pytest.raises(ValueError, match="expects"):
        ir.get_reg_memory_effect(ir.Call(ir.get_op("reg.gather"), [tile()], SPAN))


def test_indexed_effect_preserves_root_and_mask_without_claiming_a_range():
    root = ir.Var("base", ir.PtrType(), SPAN)
    storage = ir.MemRef(root, 64, 1024, SPAN)
    view = ir.TileView([128], [1], ir.ConstInt(16, DataType.INDEX, SPAN))
    target = ir.Var("view", ir.TileType([128], DataType.FP32, storage, view, ir.MemorySpace.Vec), SPAN)
    indices, m = vec(DataType.UINT32), mask()
    read = call("gather", [target, indices, m])
    write = call("scatter", [target, indices, vec(), m])
    for expr in (read, write):
        effect = ir.get_reg_memory_effect(expr)
        assert effect["root"] is root and effect["mask"] is m
        assert effect["begin"] is None and effect["end"] is None


def test_paired_schema_declares_arity():
    for name in ("mull", "addc", "subc", "addcs", "subcs", "intlv", "dintlv"):
        assert ir.get_op_output_arity(f"reg.{name}") == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
