# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile the default FP4 cast paths without loading a device runtime."""

import re
import sys
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir
from pypto.backend import BackendType
from pypto.pypto_core import passes
from pypto.runtime import RunConfig

M, K, N = 64, 64, 64
_VEC2VEC_TMOV = re.compile(r"pto\.tmov ins\([^)]*loc=vec[^)]*\) outs\([^)]*loc=vec")


@pytest.fixture(autouse=True)
def _compile_without_runtime(monkeypatch):
    backend.reset_for_testing()
    monkeypatch.setitem(sys.modules, "simpler", None)
    monkeypatch.setitem(sys.modules, "simpler_setup", None)
    yield
    backend.reset_for_testing()


@pl.jit
def fp4_rhs_cast_push(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N // 2], pl.FP4E2M1X2],
    b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """KN V2C path with carrier extents on input and logical FP8 on output."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fp4_rhs_cast_push_tile"):
        at = pl.load(a, [0, 0], [M, K])
        ats = pl.load(a_scale, [0, 0], [M, K // 32])
        bt8 = pl.cast(pl.load(b, [0, 0], [K, N // 2]), pl.FP8E4M3FN)
        bts = pl.load(b_scale, [0, 0], [K // 32, N])
        acc = pl.matmul_mx(at, ats, bt8, bts)
        c = pl.store(acc, [0, 0], c)
    return c


@pl.jit
def fp4_cast_store_only(
    b: pl.Tensor[[K, N // 2], pl.FP4E2M1X2],
    c: pl.Out[pl.Tensor[[K, N], pl.FP8E4M3FN]],
):
    """FP4->FP8 cast whose consumer is a plain store (NOT a matmul rhs)."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fp4_cast_store_only_tile"):
        bt8 = pl.cast(pl.load(b, [0, 0], [K, N // 2]), pl.FP8E4M3FN)
        c = pl.store(bt8, [0, 0], c)
    return c


@pl.jit
def fp4_cast_dynamic_rows_store(
    b: pl.Tensor[[32, 32], pl.FP4E2M1X2],
    c: pl.Out[pl.Tensor[[32, 64], pl.FP8E4M3FN]],
    valid_rows: pl.Scalar[pl.INDEX],
):
    """Keep runtime rows on the ordinary ND conversion and its store."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fp4_cast_dynamic_rows_store_tile"):
        source = pl.load(b, [0, 0], [32, 32], valid_shape=[valid_rows, 32])
        c = pl.store(pl.cast(source, pl.FP8E4M3FN), [0, 0], c)
    return c


@pl.jit
def fp4_lhs_dynamic_rows_cast_push(
    a: pl.Tensor[[32, 32], pl.FP4E2M1X2],
    a_scale: pl.Tensor[[32, 2], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[64, 64], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[2, 64], pl.FP8E8M0, pl.MX_B_NN],
    c: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
    valid_rows: pl.Scalar[pl.INDEX],
):
    """The sole-use left cast writes NZ while preserving runtime M."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fp4_lhs_dynamic_rows_cast_push_tile"):
        lhs = pl.cast(pl.load(a, [0, 0], [32, 32], valid_shape=[valid_rows, 32]), pl.FP8E4M3FN)
        lhs_scale = pl.load(a_scale, [0, 0], [32, 2], valid_shape=[valid_rows, 2])
        rhs = pl.load(b, [0, 0], [64, 64])
        rhs_scale = pl.load(b_scale, [0, 0], [2, 64])
        c = pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], c)
    return c


@pl.jit
def fp4_nk_rhs_cast_push(
    a: pl.Tensor[[32, 64], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[32, 2], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[128, 32], pl.FP4E2M1X2],
    b_scale: pl.Tensor[[128, 2], pl.FP8E8M0, pl.MX_A_ZZ],
    c: pl.Out[pl.Tensor[[32, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fp4_nk_rhs_cast_push_tile"):
        lhs = pl.load(a, [0, 0], [32, 64])
        lhs_scale = pl.load(a_scale, [0, 0], [32, 2])
        rhs = pl.tile.transpose_view(
            pl.move(pl.cast(pl.load(b, [0, 0], [128, 32]), pl.FP8E4M3FN), pl.Mem.Mat)
        )
        rhs_scale = pl.tile.transpose_view(pl.load(b_scale, [0, 0], [128, 2]))
        c = pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], c)
    return c


@pl.jit
def fp4_rhs_cast_with_store(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N // 2], pl.FP4E2M1X2],
    b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    d: pl.Out[pl.Tensor[[K, N], pl.FP8E4M3FN]],
):
    """A real ND store keeps the cast live alongside its V2C consumer."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fp4_rhs_cast_with_store_tile"):
        lhs = pl.load(a, [0, 0], [M, K])
        lhs_scale = pl.load(a_scale, [0, 0], [M, K // 32])
        rhs = pl.cast(pl.load(b, [0, 0], [K, N // 2]), pl.FP8E4M3FN)
        rhs_scale = pl.load(b_scale, [0, 0], [K // 32, N])
        d = pl.store(rhs, [0, 0], d)
        c = pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], c)
    return c, d


@pl.jit
def fp4_full_256_rhs_cast_push(
    a: pl.Tensor[[48, 256], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[48, 8], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[256, 128], pl.FP4E2M1X2],
    b_scale: pl.Tensor[[8, 256], pl.FP8E8M0, pl.MX_B_NN],
    c: pl.Out[pl.Tensor[[48, 256], pl.FP32]],
):
    # The authored B cast is 256x256. Eligibility is checked after UP_DOWN
    # splitting, when each AIV owns the proven full 128x256 output.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="fp4_full_256_rhs_cast_push_tile",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=4)],
    ):
        lhs = pl.load(a, [0, 0], [48, 256])
        lhs_scale = pl.load(a_scale, [0, 0], [48, 8])
        rhs = pl.cast(pl.load(b, [0, 0], [256, 128]), pl.FP8E4M3FN)
        rhs_scale = pl.load(b_scale, [0, 0], [8, 256])
        c = pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], c)
    return c


def _compile(kernel, output: Path, *, planner=passes.MemoryPlanner.PYPTO, nz_row_plus_one=False) -> str:
    context = passes.PassContext.current()
    assert context is not None, "The UT fixture supplies pass verification instruments"
    with passes.PassContext(
        context.get_instruments(),
        verification_level=context.get_verification_level(),
        memory_planner=planner,
        enable_fp4_nz_row_plus_one=nz_row_plus_one,
    ):
        kernel.compile(
            config=RunConfig(
                backend_type=BackendType.Ascend950,
                save_kernels_dir=str(output),
            )
        )
    files = sorted(output.rglob("*.pto"))
    assert files, f"No generated PTO files in {output}"
    return "\n".join(path.read_text() for path in files)


def _direct_cast(pto: str) -> str:
    casts = [line for line in pto.splitlines() if "pto.tcvt" in line and "dtype=!pto.f4E2M1x2" in line]
    assert len(casts) == 1, pto
    assert "dtype=f8E4M3FN" in casts[0]
    assert "pto.tgather" not in pto and "dtype=ui16" not in pto
    return casts[0]


def _assert_identity_valid_views(pto: str, *, dynamic: bool) -> list[str]:
    """TCVT refines valid columns without shrinking geometry or adding storage."""
    cast = _direct_cast(pto)
    views = [line for line in pto.splitlines() if " = pto.subview " in line and "tcvt_" in line]
    assert len(views) == 2, pto
    rows = []
    for view in views:
        match = re.search(
            r"(%\w+) = pto\.subview (%\w+)\[(%\w+), (%\w+)\] sizes \[(\d+), (\d+)\]"
            r" valid \[(%\w+), (%\w+)\] : (!pto\.tile_buf<[^>]+>) -> (!pto\.tile_buf<[^>]+>)",
            view,
        )
        assert match, view
        alias, base, row_offset, col_offset, physical_rows, physical_cols, row, col, original, refined = (
            match.groups()
        )
        assert row_offset == col_offset
        assert re.search(rf"{re.escape(row_offset)} = arith.constant 0 : index", pto)
        assert f"rows={physical_rows}, cols={physical_cols}," in original
        assert f"rows={physical_rows}, cols={physical_cols}," in refined
        # Compare every field apart from the two valid extents; layout, dtype,
        # physical size, pad and compact must remain the base handle's fields.
        assert re.sub(r"v_row=[^,]+, v_col=[^,]+", "", original) == re.sub(
            r"v_row=[^,]+, v_col=[^,]+", "", refined
        )
        assert "v_col=?" not in refined
        valid_col = re.search(r"v_col=(\d+)", refined)
        assert valid_col
        assert re.search(rf"{re.escape(col)} = arith.constant {valid_col[1]} : index", pto)
        assert ("v_row=?" in refined) == dynamic
        assert re.search(rf"{re.escape(alias)}(?: \{{[^}}]+\}})? : {re.escape(refined)}", cast)
        assert re.search(rf"{re.escape(base)} = pto\.(?:alloc_tile|subview)", pto)
        rows.append(row)
    assert rows[0] == rows[1], views
    if dynamic:
        assert not re.search(rf"{re.escape(rows[0])} = arith.constant", pto), views
    assert not re.search(r"pto\.treshape[^\n]*dtype=(?:!pto.f4E2M1x2|f8E4M3FN)", pto)
    return views


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("fused", [False, True])
def test_runtime_rows_keep_static_columns_on_identity_cast_views(tmp_path, planner, fused):
    kernel = fp4_lhs_dynamic_rows_cast_push if fused else fp4_cast_dynamic_rows_store
    pto = _compile(kernel, tmp_path, planner=planner)
    views = _assert_identity_valid_views(pto, dynamic=True)
    assert "dtype=!pto.f4E2M1x2, rows=32, cols=32, v_row=?, v_col=32," in views[0]
    assert "dtype=f8E4M3FN, rows=32, cols=64, v_row=?, v_col=64," in views[1]
    expected_layout = "col_major, slayout=row_major" if fused else "row_major, slayout=none_box"
    assert f"blayout={expected_layout}" in views[1]
    if fused:
        assert "pto.tpush_to_aic" in pto
        assert not _VEC2VEC_TMOV.search(pto)
    else:
        # A load plus its widened result is sufficient; subviews add no alloc.
        assert pto.count(" = pto.alloc_tile ") == 2, pto
        assert "pto.tstore" in pto


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_kn_cast_preserves_row_major_gather_and_adapts_before_push(tmp_path, planner):
    pto = _compile(fp4_rhs_cast_push, tmp_path, planner=planner)
    cast = _direct_cast(pto)
    assert "blayout=col_major, slayout=row_major" in cast.split("outs(")[1]
    assert not _VEC2VEC_TMOV.search(pto), "The sole-use native conversion already writes NZ"
    assert "pto.tpush_to_aic" in pto


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_cast_to_store_keeps_row_major_gather(tmp_path, planner):
    pto = _compile(fp4_cast_store_only, tmp_path, planner=planner)
    cast = _direct_cast(pto)
    assert "blayout=row_major, slayout=none_box" in cast.split("outs(")[1]
    assert "pto.tstore" in pto


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_nk_cast_fuses_transpose_into_non_square_boundary_pop(tmp_path, planner):
    pto = _compile(fp4_nk_rhs_cast_push, tmp_path, planner=planner)
    cast = _direct_cast(pto)
    assert "blayout=col_major, slayout=row_major" in cast.split("outs(")[1]
    assert not _VEC2VEC_TMOV.search(pto)
    assert re.search(r"dtype=f8E4M3FN, rows=64, cols=128,", pto)
    assert not re.search(r"pto\.treshape[^\n]*dtype=f8E4M3FN", pto)
    assert "pto.tpop_from_aiv" in pto


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_multi_consumer_cast_keeps_native_nd_and_separate_nz_adapter(tmp_path, planner):
    pto = _compile(fp4_rhs_cast_with_store, tmp_path, planner=planner)
    cast = _direct_cast(pto)
    assert "blayout=row_major, slayout=none_box" in cast.split("outs(")[1]
    assert _VEC2VEC_TMOV.search(pto), "The ND store and NZ transport need distinct results"
    assert "pto.tpush_to_aic" in pto and "pto.tstore" in pto


def test_ptoas_planner_uses_native_cast_chain(tmp_path):
    for planner in (passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.PYPTO):
        pto = _compile(fp4_rhs_cast_push, tmp_path / planner.name, planner=planner)
        _direct_cast(pto)
        assert not _VEC2VEC_TMOV.search(pto)


def _padded_cast_program(dtype, fused, *, dynamic=False, empty=False):
    """Different physical padding must survive refinement of equal valid extents."""
    span = ir.Span.unknown()
    row = (
        ir.Var("valid_rows", ir.ScalarType(DataType.INDEX), span)
        if dynamic
        else ir.ConstInt(0 if empty else 23, DataType.INDEX, span)
    )
    columns, valid_columns = 160, 129
    source_type = ir.TileType(
        [32, columns],
        dtype,
        ir.MemRef("source_base", 0, 5120),
        ir.TileView(valid_shape=[row, valid_columns]),
        ir.MemorySpace.Vec,
    )
    result_type = ir.TileType(
        [48, 384],
        DataType.FP8E4M3FN,
        ir.MemRef("result_base", 5120, 18432),
        ir.TileView(
            valid_shape=[row, 258],
            blayout=ir.TileLayout.col_major if fused else ir.TileLayout.row_major,
            slayout=ir.TileLayout.row_major if fused else ir.TileLayout.none_box,
        ),
        ir.MemorySpace.Vec,
    )
    source = ir.Var("source", source_type, span)
    result = ir.Var("result", result_type, span)
    gm = ir.Var("gm", ir.TensorType([32, columns], dtype), span)
    indices = ir.MakeTuple([ir.ConstInt(0, DataType.INDEX, span)] * 2, span)
    shape = ir.MakeTuple([ir.ConstInt(n, DataType.INDEX, span) for n in (32, columns)], span)
    valid = ir.MakeTuple([row, ir.ConstInt(valid_columns, DataType.INDEX, span)], span)
    load = ir.Call(ir.get_op("tile.load"), [gm, indices, shape, valid], {}, source_type, span)
    cast = ir.Call(
        ir.get_op("tile.cast_to_nz" if fused else "tile.cast"),
        [source],
        {"target_type": DataType.FP8E4M3FN, "mode": 2},
        result_type,
        span,
    )
    body = ir.SeqStmts(
        [ir.AssignStmt(source, load, span), ir.AssignStmt(result, cast, span), ir.ReturnStmt([], span)], span
    )
    params = [(gm, ir.ParamDirection.In)]
    if dynamic:
        assert isinstance(row, ir.Var)
        params.append((row, ir.ParamDirection.In))
    function = ir.Function("padded_cast", params, [], body, span, ir.FunctionType.AIV)
    return ir.Program([function], "padded_cast", span)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_padded_cast_views_preserve_physical_geometry_and_allocations(planner, fused, dynamic):
    backend.set_backend_type(BackendType.Ascend950)
    program = _padded_cast_program(DataType.FP4E2M1X2, fused, dynamic=dynamic)
    with passes.PassContext([], memory_planner=planner):
        pto = codegen.PTOCodegen().generate(program, emit_tile_addr=planner == passes.MemoryPlanner.PYPTO)
    views = _assert_identity_valid_views(pto, dynamic=dynamic)
    # The packed source keeps carrier geometry and valid columns. The
    # destination's extra physical rows/columns remain padding, not valid data.
    assert "dtype=!pto.f4E2M1x2, rows=32, cols=160," in views[0]
    assert "dtype=f8E4M3FN, rows=48, cols=384," in views[1]
    assert "v_col=129," in views[0] and "v_col=258," in views[1]
    assert pto.count(" = pto.alloc_tile ") == 2, pto
    assert "pto.tmov" not in pto and "pto.textract" not in pto
    if planner == passes.MemoryPlanner.PYPTO:
        allocations = [line for line in pto.splitlines() if " = pto.alloc_tile " in line]
        assert "addr = %c0_i64" in allocations[0]
        assert "addr = %c5120_i64" in allocations[1]


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("fused", [False, True])
def test_static_zero_valid_rows_preserve_allocations_without_conversion(planner, fused):
    backend.set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], memory_planner=planner):
        pto = codegen.PTOCodegen().generate(
            _padded_cast_program(DataType.FP4E2M1X2, fused, empty=True),
            emit_tile_addr=planner == passes.MemoryPlanner.PYPTO,
        )
    allocations = [line for line in pto.splitlines() if " = pto.alloc_tile " in line]
    assert len(allocations) == 2, pto
    assert all("valid_row = %c0_index" in line and "v_row=?, v_col=?" in line for line in allocations)
    assert "dtype=!pto.f4E2M1x2, rows=32, cols=160," in allocations[0]
    assert "dtype=f8E4M3FN, rows=48, cols=384," in allocations[1]
    layout = "col_major, slayout=row_major" if fused else "row_major, slayout=none_box"
    assert f"blayout={layout}" in allocations[1]
    assert "pto.tcvt" not in pto and "pto.subview" not in pto
    assert "pto.tmov" not in pto and "pto.textract" not in pto
    if planner == passes.MemoryPlanner.PYPTO:
        assert "addr = %c0_i64" in allocations[0]
        assert "addr = %c5120_i64" in allocations[1]


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_explicit_packed_fp4_fp8_addresses_require_disjoint_windows(fused, overlap):
    backend.set_backend_type(BackendType.Ascend950)
    span = ir.Span.unknown()
    source_type = ir.TileType(
        [32, 256], DataType.FP4E2M1X2, ir.MemRef("source_base", 0, 8192), None, ir.MemorySpace.Vec
    )
    source = ir.Var("source", source_type, span)
    op_name = "tile.cast_to_nz" if fused else "tile.cast"
    kwargs = {"target_type": DataType.FP8E4M3FN, "mode": 2}
    inferred = ir.create_op_call(op_name, [source], kwargs, span).type
    assert isinstance(inferred, ir.TileType)
    result_type = ir.TileType(
        [32, 512],
        DataType.FP8E4M3FN,
        ir.MemRef("result_base", 4096 if overlap else 8192, 16384),
        inferred.tile_view,
        ir.MemorySpace.Vec,
    )
    result = ir.Var("result", result_type, span)
    gm = ir.Var("gm", ir.TensorType([32, 256], DataType.FP4E2M1X2), span)
    indices = ir.MakeTuple([ir.ConstInt(0, DataType.INDEX, span)] * 2, span)
    shape = ir.MakeTuple([ir.ConstInt(n, DataType.INDEX, span) for n in (32, 256)], span)
    load = ir.Call(ir.get_op("tile.load"), [gm, indices, shape, shape], {}, source_type, span)
    cast = ir.Call(ir.get_op(op_name), [source], kwargs, result_type, span)
    body = ir.SeqStmts(
        [ir.AssignStmt(source, load, span), ir.AssignStmt(result, cast, span), ir.ReturnStmt([], span)], span
    )
    function = ir.Function(
        "explicit_cast", [(gm, ir.ParamDirection.In)], [], body, span, ir.FunctionType.InCore
    )
    program = ir.Program([function], "explicit_cast", span)
    if overlap:
        with pytest.raises(ValueError, match="requires non-overlapping"):
            codegen.PTOCodegen().generate(program, emit_tile_addr=True)
    else:
        assert "pto.tcvt" in codegen.PTOCodegen().generate(program, emit_tile_addr=True)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("enabled", [False, True])
def test_row_plus_one_actual_dsl_split_keeps_padded_tpush_and_compact_fifo(tmp_path, planner, enabled):
    pto = _compile(fp4_full_256_rhs_cast_push, tmp_path, planner=planner, nz_row_plus_one=enabled)
    cast = _direct_cast(pto)
    assert "rows=128, cols=256" in cast
    dst = cast.split("outs(", 1)[1]
    assert (", compact=2" in dst) is enabled
    pushes = [
        line
        for line in pto.splitlines()
        if "pto.tpush_to_aic" in line and "dtype=f8E4M3FN" in line and "rows=128, cols=256" in line
    ]
    assert len(pushes) == 1, pto
    assert (", compact=2" in pushes[0]) is enabled
    assert re.search(r"slot_size\s*=\s*65536\b", pto)
    mat_lines = [line for line in pto.splitlines() if "loc=mat" in line]
    assert mat_lines and all(", compact=2" not in line for line in mat_lines)
    assert "rows=129" not in pto


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
