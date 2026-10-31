# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ordinary VF families: typed frontend contracts and real PTOAS compilation."""

import subprocess

import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.builder import IRBuilder
from pypto.ir.op import tile

SPAN = ir.Span.unknown()
UNARY = ["abs", "neg", "relu", "exp", "log", "sqrt", "bitwise_not"]
BINARY = ["bitwise_and", "bitwise_or", "bitwise_xor", "shift_left", "shift_right"]
SCALAR = ["adds", "muls", "maximums", "minimums", "shift_lefts", "shift_rights"]
REDUCE = ["reduce_sum", "reduce_max", "reduce_min"]
OPS = UNARY + BINARY + SCALAR + REDUCE + ["compares", "gather_lanes"]
OPCODES = dict(
    zip(
        OPS,
        [
            "vabs",
            "vneg",
            "vrelu",
            "vexp",
            "vln",
            "vsqrt",
            "vnot",
            "vand",
            "vor",
            "vxor",
            "vshl",
            "vshr",
            "vadds",
            "vmuls",
            "vmaxs",
            "vmins",
            "vshls",
            "vshrs",
            "vcadd",
            "vcmax",
            "vcmin",
            "vcmps",
            "vselr",
        ],
        strict=True,
    )
)


@pytest.fixture(autouse=True)
def a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def vector(dtype=DataType.FP32, lanes=64):
    return ir.Var("x", ir.RegType(lanes, dtype, ir.RegKind.VecReg), SPAN)


def mask(lanes=64):
    return ir.Var("m", ir.RegType(lanes, DataType.BOOL, ir.RegKind.MaskReg), SPAN)


def invoke(op, x, m, masked=True):
    fn = getattr(pl.reg, op)
    if op in UNARY:
        return fn(x, m if masked else None).unwrap()
    if op in BINARY:
        return fn(x, x, m if masked else None).unwrap()
    if op in SCALAR:
        return fn(x, 1, m).unwrap()
    if op in REDUCE:
        return fn(x, m, **({"reassoc": True} if op == "reduce_sum" else {})).unwrap()
    if op == "compares":
        return fn(x, 1, m, predicate="ge").unwrap()
    idx_dtype = {1: DataType.UINT8, 2: DataType.UINT16, 4: DataType.UINT32}[x.type.dtype.get_byte()]
    idx = pl.reg.index_gen(0, lanes=64, dtype=idx_dtype)
    return fn(x, idx).unwrap()


def kernel(op, dtype, masked):
    ib = IRBuilder()
    with ib.function("extended", type=ir.FunctionType.AIV, span=SPAN) as f:
        ty = ir.TileType(
            [1, 128],
            dtype,
            tile_view=ir.TileView([1, 128], [128, 1], ir.ConstInt(0, DataType.INDEX, SPAN)),
            memory_space=ir.MemorySpace.Vec,
        )
        src = ib.let("src", tile.full([1, 128], dtype=dtype, value=1), type=ty)
        dst = ib.let("dst", tile.full([1, 128], dtype=dtype, value=0), type=ty)
        with ib.reg_scope(SPAN):
            x = ib.emit_reg("x", "reg.load", [src, ir.ConstInt(0, DataType.INDEX, SPAN)], {"lanes": 64}, SPAN)
            m = ib.emit_reg(
                "m", "reg.active_mask", [ir.ConstInt(32, DataType.INDEX, SPAN)], {"lanes": 64}, SPAN
            )
            if op == "gather_lanes":
                idx = ib.emit_reg(
                    "idx",
                    "reg.index_gen",
                    [ir.ConstInt(0, DataType.UINT32, SPAN)],
                    {"lanes": 64, "dtype": DataType.UINT32},
                    SPAN,
                )
                call = pl.reg.gather_lanes(x, idx).unwrap()
            else:
                call = invoke(op, x, m, masked)
            y = ib.let("y", call)
            if op == "compares":
                y = ib.emit_reg("selected", "reg.select", [y, x, x], span=SPAN)
            ib.emit_reg_store("out", dst, y, span=SPAN)
    with passes.PassContext([]):
        return passes.init_mem_ref()(ir.Program([f.get_result()], "Extended", SPAN))


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("masked", [False, True])
def test_basic_family_compiles(op, masked, tmp_path):
    dtype = (
        DataType.INT32
        if op in BINARY or op in ("bitwise_not", "shift_lefts", "shift_rights")
        else DataType.FP32
    )
    mlir = codegen.PTOCodegen().generate(
        kernel(op, dtype, masked), emit_tile_addr=False, emit_source_loc=False
    )
    assert f"pto.vmi.{OPCODES[op]} " in mlir
    if op in REDUCE:
        assert "-> !pto.vmi.vreg<1xf32>" in mlir
    compiler = find_ptoas_binary()
    assert compiler, "PTOAS required for extended VF validation"
    source = tmp_path / "extended.pto"
    source.write_text(mlir)
    result = subprocess.run(
        [compiler, str(source), "--pto-arch=a5", "--emit-vpto", "-o", str(tmp_path / "out.vpto")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"{mlir}\n{result.stderr}"


@pytest.mark.parametrize(
    "op", ["adds", "muls", "maximums", "minimums", "compares", "shift_lefts", "shift_rights"]
)
@pytest.mark.parametrize("dtype", [DataType.INT32, DataType.UINT32])
@pytest.mark.ptoas_compile
def test_integer_scalar_compiles(op, dtype, tmp_path):
    mlir = codegen.PTOCodegen().generate(kernel(op, dtype, True), emit_tile_addr=False, emit_source_loc=False)
    source = tmp_path / "integer.pto"
    source.write_text(mlir)
    compiler = find_ptoas_binary()
    assert compiler is not None
    result = subprocess.run(
        [compiler, str(source), "--pto-arch=a5", "--emit-vpto", "-o", str(tmp_path / "out.vpto")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"{mlir}\n{result.stderr}"


@pytest.mark.parametrize(
    "op,dtype",
    [
        ("abs", DataType.UINT32),
        ("neg", DataType.BF16),
        ("relu", DataType.INT16),
        ("exp", DataType.INT32),
        ("log", DataType.INT32),
        ("sqrt", DataType.INT32),
        ("bitwise_not", DataType.FP32),
        ("shift_left", DataType.FP32),
        ("muls", DataType.INT8),
        ("reduce_sum", DataType.INT8),
        ("reduce_max", DataType.BF16),
        ("compares", DataType.INT16),
    ],
)
def test_unsupported_dtype_rejected(op, dtype):
    with pytest.raises((ValueError, TypeError), match="dtype|type|support"):
        invoke(op, vector(dtype), mask())


def test_float_sum_requires_explicit_reassociation():
    with pytest.raises(ValueError, match="reassoc"):
        pl.reg.reduce_sum(vector(), mask())


@pytest.mark.parametrize("op", UNARY + BINARY + SCALAR + REDUCE + ["compares"])
def test_mask_lane_mismatch_rejected(op):
    dtype = DataType.INT32 if "bitwise" in op or "shift" in op else DataType.FP32
    with pytest.raises(ValueError, match="[Mm]ask|lane"):
        invoke(op, vector(dtype), mask(32))


def test_scalar_ssa_is_not_implicitly_cast():
    with pytest.raises(ValueError, match="[Ss]calar|dtype"):
        pl.reg.adds(vector(), ir.Var("s", ir.ScalarType(DataType.INT32), SPAN), mask())


def test_scalar_shift_requires_i16_ssa():
    with pytest.raises(ValueError, match="[Ss]calar|i16|dtype"):
        pl.reg.shift_lefts(vector(DataType.INT32), ir.Var("s", ir.ScalarType(DataType.INT32), SPAN), mask())


@pytest.mark.parametrize("lanes,dtype", [(32, DataType.UINT32), (64, DataType.UINT16)])
def test_gather_rejects_unsupported_lanes_or_index_width(lanes, dtype):
    with pytest.raises(ValueError, match="lane|width"):
        pl.reg.gather_lanes(vector(lanes=lanes), vector(dtype, lanes))
