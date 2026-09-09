# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical Buffer branch/carry acceptance with several runtime paths per artifact."""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto import ir, passes

_ROWS, _COLS, _BAND, _CASES = 16, 32, 32, 4
_PLANNERS = (passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS)


@pl.jit.incore
def _branch_kernel(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    saved: pl.InOut[pl.Tensor],
    flag: pl.Scalar[pl.INT64],
    base: pl.Scalar[pl.INT64],
):
    lhs = pl.load(a, [0, 0], [_ROWS, _COLS])
    rhs = pl.load(b, [0, 0], [_ROWS, _COLS])
    if flag > 0:
        value, column = pl.yield_(pl.add(lhs, rhs), 1)
    else:
        value, column = pl.yield_(pl.mul(lhs, rhs), 3)
    stored = pl.store(value, [base, column], out)
    preserved = pl.store(lhs, [base, 0], saved)
    return stored, preserved


@pl.jit
def _branch_entry(
    a: pl.Tensor, b: pl.Tensor, config: pl.Tensor, out: pl.InOut[pl.Tensor], saved: pl.InOut[pl.Tensor]
):
    for case_index in pl.range(_CASES):
        flag = pl.tensor.read(config, [case_index, 1])
        base = case_index * _BAND
        out, saved = _branch_kernel(a, b, out, saved, flag, base)
    return out, saved


@pl.jit.incore
def _for_kernel(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    saved: pl.InOut[pl.Tensor],
    count: pl.Scalar[pl.INT64],
    flag: pl.Scalar[pl.INT64],
    base: pl.Scalar[pl.INT64],
):
    lhs = pl.load(a, [0, 0], [_ROWS, _COLS])
    rhs = pl.load(b, [0, 0], [_ROWS, _COLS])
    for _i, (left, row, right, column, gm) in pl.range(0, count, init_values=(lhs, 0, rhs, 0, out)):
        if flag > 0:
            selected = pl.yield_(left)
        else:
            selected = pl.yield_(right)
        stored = pl.store(selected, [base + row, column], gm)
        final_left, final_row, final_right, final_column, final_gm = pl.yield_(
            right, row + 1, left, column + 2, stored
        )
    weighted = pl.mul(final_right, rhs)
    result = pl.add(final_left, weighted)
    stored = pl.store(result, [base + final_row, final_column], final_gm)
    preserved = pl.store(lhs, [base, 0], saved)
    return stored, preserved


@pl.jit
def _for_entry(
    a: pl.Tensor, b: pl.Tensor, config: pl.Tensor, out: pl.InOut[pl.Tensor], saved: pl.InOut[pl.Tensor]
):
    for case_index in pl.range(_CASES):
        count = pl.tensor.read(config, [case_index, 0])
        flag = pl.tensor.read(config, [case_index, 1])
        base = case_index * _BAND
        out, saved = _for_kernel(a, b, out, saved, count, flag, base)
    return out, saved


@pl.jit.incore
def _while_kernel(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    saved: pl.InOut[pl.Tensor],
    count: pl.Scalar[pl.INT64],
    flag: pl.Scalar[pl.INT64],
    base: pl.Scalar[pl.INT64],
):
    lhs = pl.load(a, [0, 0], [_ROWS, _COLS])
    rhs = pl.load(b, [0, 0], [_ROWS, _COLS])
    for left, row, right, column, gm in pl.while_(init_values=(lhs, 0, rhs, 0, out)):
        pl.cond(row < count)
        if flag > 0:
            selected = pl.yield_(left)
        else:
            selected = pl.yield_(right)
        stored = pl.store(selected, [base + row, column], gm)
        final_left, final_row, final_right, final_column, final_gm = pl.yield_(
            right, row + 1, left, column + 2, stored
        )
    weighted = pl.mul(final_right, rhs)
    result = pl.add(final_left, weighted)
    stored = pl.store(result, [base + final_row, final_column], final_gm)
    preserved = pl.store(lhs, [base, 0], saved)
    return stored, preserved


@pl.jit
def _while_entry(
    a: pl.Tensor, b: pl.Tensor, config: pl.Tensor, out: pl.InOut[pl.Tensor], saved: pl.InOut[pl.Tensor]
):
    for case_index in pl.range(_CASES):
        count = pl.tensor.read(config, [case_index, 0])
        flag = pl.tensor.read(config, [case_index, 1])
        base = case_index * _BAND
        out, saved = _while_kernel(a, b, out, saved, count, flag, base)
    return out, saved


@pl.jit.incore
def _nested_kernel(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    saved: pl.InOut[pl.Tensor],
    outer_count: pl.Scalar[pl.INT64],
    inner_count: pl.Scalar[pl.INT64],
    base: pl.Scalar[pl.INT64],
):
    lhs = pl.load(a, [0, 0], [_ROWS, _COLS])
    rhs = pl.load(b, [0, 0], [_ROWS, _COLS])
    for _i, (outer,) in pl.range(outer_count, init_values=(lhs,)):
        for _j, (inner,) in pl.range(inner_count, init_values=(outer,)):
            total = pl.add(inner, rhs)
            inner_result = pl.yield_(total)
        outer_result = pl.yield_(inner_result)
    stored = pl.store(outer_result, [base, 0], out)
    preserved = pl.store(lhs, [base, 0], saved)
    return stored, preserved


@pl.jit
def _nested_entry(
    a: pl.Tensor, b: pl.Tensor, config: pl.Tensor, out: pl.InOut[pl.Tensor], saved: pl.InOut[pl.Tensor]
):
    for case_index in pl.range(_CASES):
        outer_count = pl.tensor.read(config, [case_index, 0])
        inner_count = pl.tensor.read(config, [case_index, 1])
        base = case_index * _BAND
        out, saved = _nested_kernel(a, b, out, saved, outer_count, inner_count, base)
    return out, saved


@pl.jit.incore
def _fanout_kernel(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    saved: pl.InOut[pl.Tensor],
    count: pl.Scalar[pl.INT64],
    base: pl.Scalar[pl.INT64],
):
    lhs = pl.load(a, [0, 0], [_ROWS, _COLS])
    rhs = pl.load(b, [0, 0], [_ROWS, _COLS])
    for _i, (left, row, right, column) in pl.range(0, count, init_values=(lhs, 0, rhs, 0)):
        final_left, final_row, final_right, final_column = pl.yield_(right, row + 1, right, column + 2)
    weighted = pl.mul(final_right, rhs)
    result = pl.add(final_left, weighted)
    stored = pl.store(result, [base + final_row, final_column], out)
    preserved = pl.store(lhs, [base, 0], saved)
    return stored, preserved


@pl.jit
def _fanout_entry(
    a: pl.Tensor, b: pl.Tensor, config: pl.Tensor, out: pl.InOut[pl.Tensor], saved: pl.InOut[pl.Tensor]
):
    for case_index in pl.range(_CASES):
        count = pl.tensor.read(config, [case_index, 0])
        base = case_index * _BAND
        out, saved = _fanout_kernel(a, b, out, saved, count, base)
    return out, saved


def _golden(kind: str, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out, saved = tensors["out"].clone(), tensors["saved"].clone()
    lhs, rhs = tensors["a"], tensors["b"]
    for case_index, (count, flag) in enumerate(tensors["config"].tolist()):
        base = case_index * _BAND
        row, column = 0, 0
        if kind == "branch":
            result, column = (lhs + rhs, 1) if flag > 0 else (lhs * rhs, 3)
        elif kind == "nested":
            result = lhs.clone()
            for _ in range(count):
                for _ in range(flag):
                    result = result + rhs
        else:
            left, right = lhs, rhs
            for _ in range(count):
                if kind != "fanout":
                    selected = left if flag > 0 else right
                    out[base + row : base + row + _ROWS, column : column + _COLS] = selected
                left, right = (right, right) if kind == "fanout" else (right, left)
                row, column = row + 1, column + 2
            # Asymmetric weights expose incorrect swap order; left + right
            # alone would hide odd/even carry mistakes.
            result = left + right * rhs
        out[base + row : base + row + _ROWS, column : column + _COLS] = result
        saved[base : base + _ROWS, :_COLS] = lhs
    return {"out": out, "saved": saved}


_PROGRAMS = {
    "branch": _branch_entry,
    "for": _for_entry,
    "while": _while_entry,
    "nested": _nested_entry,
    "fanout": _fanout_entry,
}
_GRID = torch.arange(_ROWS * _COLS, dtype=torch.float32).reshape(_ROWS, _COLS)
_A = (_GRID.remainder(29) - 14) / 8
_B = (_GRID.remainder(17) + 1) / 16
_LOOP_CONFIG = torch.tensor([[0, 0], [1, 0], [2, 1], [3, 1]], dtype=torch.int64)
_NESTED_CONFIG = torch.tensor([[0, 3], [1, 2], [2, 0], [2, 3]], dtype=torch.int64)


@st.cases(
    *[
        st.case(
            program,
            _A,
            _B,
            _NESTED_CONFIG if kind == "nested" else _LOOP_CONFIG,
            torch.full((_CASES * _BAND, 64), -777.0),
            torch.full((_CASES * _BAND, 64), -555.0),
            name=f"buffer_{kind}_{planner.name.lower()}",
            golden=lambda tensors, kind=kind: _golden(kind, tensors),
            memory_planner=planner,
            enable_buffer_ir=True,
            rtol=1e-6,
            atol=1e-6,
        )
        for kind, program in _PROGRAMS.items()
        for planner in _PLANNERS
    ]
)
def test_public_buffer_control_flow(case_run, request):
    case_run.assert_passed()
    assert case_run.case.get_enable_buffer_ir()
    if request.config.getoption("--precompile-workers") is not None:
        assert case_run.work_dir is not None, "Buffer case bypassed the precompile pipeline"
        program = ir.deserialize((case_run.work_dir / "buffer_ir.msgpack").read_bytes())
        assert isinstance(program, ir.Program)
        kernels = [
            function for function in program.functions.values() if ir.is_incore_type(function.func_type)
        ]
        assert len(kernels) == 1 and kernels[0].ir_stage == ir.FunctionIRStage.Buffer
        sources = list((case_run.work_dir / "ptoas").glob("*.pto"))
        assert len(sources) == 1, "Expected the exact source of the one executed device kernel"
        text = sources[0].read_text()
        kind = case_run.case.get_name().split("_")[1]
        assert text.count("pto.tload ins(") == 2
        assert text.count("pto.tstore ins(") == (3 if kind in {"for", "while"} else 2)
        assert (" addr = " in text) == (case_run.case.memory_planner != passes.MemoryPlanner.PTOAS)
        headers = [line for line in text.splitlines() if "scf.for " in line or "scf.while " in line]
        assert len(headers) == {"branch": 0, "for": 1, "while": 1, "nested": 2, "fanout": 1}[kind]
        assert all("tile_buf" not in header for header in headers)
        assert ("scf.while " in text) == (kind == "while")
        if kind in {"branch", "for", "while"}:
            assert "scf.if " in text
        if kind in {"for", "while", "fanout"}:
            # Final scalar inference types store-offset carries as INDEX.
            assert "-> (index, index)" in headers[0]
            assert "pto.tmov ins(" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
