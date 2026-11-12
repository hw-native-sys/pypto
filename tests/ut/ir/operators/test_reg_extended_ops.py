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


@pytest.mark.parametrize(
    "src,dst,lanes,expected",
    [
        (DataType.FP32, DataType.UINT32, 64, 64),
        (DataType.UINT32, DataType.UINT8, 64, 256),
        (DataType.UINT8, DataType.FP32, 256, 64),
        (DataType.BF16, DataType.INT16, 128, 128),
        (DataType.INT32, DataType.UINT32, 64, 64),
    ],
)
def test_interpret_cast_bit_capacity(src, dst, lanes, expected):
    result = pl.reg.interpret_cast(vec(src, lanes), dtype=dst).unwrap()
    assert isinstance(result.type, ir.RegType)
    assert result.type.dtype == dst and result.type.lanes == expected


def test_interpret_cast_rejects_invalid_contracts():
    with pytest.raises(ValueError, match="bit count is too large"):
        pl.reg.interpret_cast(vec(DataType.UINT32, 2**62), dtype=DataType.UINT8)
    with pytest.raises(ValueError, match="whole destination lanes"):
        pl.reg.interpret_cast(vec(DataType.UINT8, 3), dtype=DataType.UINT32)
    for dtype in (DataType.BOOL, DataType.INDEX, DataType.INT64):
        with pytest.raises(ValueError, match="destination dtype"):
            pl.reg.interpret_cast(vec(), dtype=dtype)
    with pytest.raises(ValueError, match="VecReg"):
        pl.reg.interpret_cast(mask(), dtype=DataType.UINT32)


@pytest.mark.parametrize("name", ["chist", "dhist"])
@pytest.mark.parametrize("bins", [128, 256])
@pytest.mark.parametrize("lanes", [1, 8, 64, 128, 256, 512])
def test_histogram_independent_source_and_bin_counts(name, bins, lanes):
    result = getattr(pl.reg, name)(
        vec(DataType.UINT16, bins), vec(DataType.UINT8, lanes), mask(lanes)
    ).unwrap()
    assert isinstance(result.type, ir.RegType)
    assert result.type.dtype == DataType.UINT16 and result.type.lanes == bins


@pytest.mark.parametrize("name", ["chist", "dhist"])
def test_histogram_invalid_contracts(name):
    for acc in (vec(DataType.INT16, 128), vec(DataType.UINT16, 64)):
        with pytest.raises(ValueError, match="accumulator"):
            call(name, [acc, vec(DataType.UINT8), mask()])
    with pytest.raises(ValueError, match="source must be UINT8"):
        call(name, [vec(DataType.UINT16, 256), vec(DataType.INT8), mask()])
    with pytest.raises(ValueError, match="Mask lane"):
        call(name, [vec(DataType.UINT16, 256), vec(DataType.UINT8), mask(256)])
    with pytest.raises(ValueError, match="expects"):
        call(name, [vec(DataType.UINT16, 256)])


@pytest.mark.parametrize("active", [0, 5, 16])
def test_group_mask_geometry(active):
    result = pl.reg.create_group_mask(active, num_groups=8, group_size=16).unwrap()
    assert isinstance(result.type, ir.RegType)
    assert result.type.lanes == 128 and result.type.kind == ir.RegKind.MaskReg


def test_group_mask_dynamic_count_and_invalid_geometry():
    active = ir.Var("n", ir.ScalarType(DataType.INDEX), SPAN)
    with pytest.raises(ValueError, match="power-of-two"):
        pl.reg.create_group_mask(active, num_groups=4, group_size=15)
    result = pl.reg.create_group_mask(active, num_groups=4, group_size=16).unwrap()
    assert isinstance(result, ir.Call)
    assert result.args[0] is active
    for count in (-1, 17):
        with pytest.raises(ValueError, match="active_elems_per_group"):
            pl.reg.create_group_mask(count, num_groups=4, group_size=16)
    for groups, size in ((0, 16), (4, -1), (2147483647, 16)):
        with pytest.raises(ValueError, match="positive|representable"):
            call(
                "create_group_mask",
                [ir.ConstInt(1, DataType.INDEX, SPAN)],
                num_groups=groups,
                group_size=size,
            )
    with pytest.raises(ValueError, match="dtype"):
        call("create_group_mask", [ir.ConstInt(1, DataType.INT32, SPAN)], num_groups=4, group_size=16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize("write", [False, True])
def test_indexed_scalar_base_offset(write):
    src, m = tile(size=256), mask()
    base = ir.Var("base", ir.ScalarType(DataType.INDEX), SPAN)
    indices = vec(DataType.UINT32)
    result = (
        pl.reg.scatter(src, indices, vec(), m, base_offset=base)
        if write
        else pl.reg.gather(src, indices, m, base_offset=base)
    ).unwrap()
    assert isinstance(result, ir.Call)
    assert result.args[-1] is base
    effect = ir.get_reg_memory_effect(result)
    # The scalar base is not the predicate. Indexed accesses still describe
    # unknown intervals and must never establish contiguous initialization.
    assert effect["mask"] is m
    assert effect["begin"] is None and effect["end"] is None
    for bad_base in [-1, 256, ir.Var("wrong", ir.ScalarType(DataType.INT32), SPAN)]:
        with pytest.raises(ValueError, match="bounds|INDEX"):
            if write:
                pl.reg.scatter(src, indices, vec(), m, base_offset=bad_base)
            else:
                pl.reg.gather(src, indices, m, base_offset=bad_base)


@pytest.mark.parametrize("base", [True, False])
def test_indexed_base_rejects_boolean(base):
    with pytest.raises(TypeError, match="not bool"):
        pl.reg.gather(tile(), vec(DataType.UINT32), mask(), base_offset=base)
    with pytest.raises(TypeError, match="not bool"):
        pl.reg.scatter(tile(), vec(DataType.UINT32), vec(), mask(), base_offset=base)


@pytest.mark.parametrize("name", ["intlv", "dintlv"])
@pytest.mark.parametrize("dtype", [DataType.FP32, DataType.BF16])
def test_unmasked_rearrangement_types(name, dtype):
    first, second = getattr(pl.reg, name)(vec(dtype, 128), vec(dtype, 128))
    assert first.unwrap().tuple is second.unwrap().tuple
    assert len(first.unwrap().tuple.args) == 2
    assert first.unwrap().type.dtype == dtype
    assert first.unwrap().type.shape == 128
    with pytest.raises(ValueError, match="lane shape"):
        getattr(pl.reg, name)(vec(dtype, 128), vec(dtype, 64))
