# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public JIT -> Buffer -> native runtime numerical acceptance for every planner.

The harness checks the actual final device representation in both compile paths.
With --precompile-workers, also inspect the same cached program and native source
whose artifact executes. A passing legacy kernel cannot satisfy these checks.
"""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto import ir, passes

_ROWS, _COLS = 16, 32


@pl.jit.incore
def _buffer_arithmetic(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    lhs = pl.load(a, [0, 0], [_ROWS, _COLS])
    rhs = pl.load(b, [0, 0], [_ROWS, _COLS])
    total = pl.add(lhs, rhs)
    product = pl.mul(total, rhs)
    return pl.store(product, [0, 0], out)


@pl.jit
def _buffer_entry(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    out = _buffer_arithmetic(a, b, out)
    return out


_GRID = torch.arange(_ROWS * _COLS, dtype=torch.float32).reshape(_ROWS, _COLS)
_A = (_GRID.remainder(29) - 14) / 8
_B = (_GRID.remainder(17) + 1) / 16
_PLANNERS = (passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS)


@st.cases(
    *[
        st.case(
            _buffer_entry,
            _A,
            _B,
            torch.full((_ROWS, _COLS), -777.0),
            name=f"buffer_arithmetic_{planner.name.lower()}",
            golden=lambda tensors: (tensors["a"] + tensors["b"]) * tensors["b"],
            memory_planner=planner,
            enable_buffer_ir=True,
            rtol=1e-6,
            atol=1e-6,
        )
        for planner in _PLANNERS
    ]
)
def test_public_buffer_arithmetic(case_run, request):
    case_run.assert_passed()
    assert case_run.case.get_enable_buffer_ir()
    if request.config.getoption("--precompile-workers") is not None:
        assert case_run.work_dir is not None, "Buffer case bypassed the precompile pipeline"
        program = ir.deserialize((case_run.work_dir / "buffer_ir.msgpack").read_bytes())
        assert isinstance(program, ir.Program)
        kernels = [f for f in program.functions.values() if ir.is_incore_type(f.func_type)]
        assert kernels and all(f.ir_stage == ir.FunctionIRStage.Buffer for f in kernels)
        native = list((case_run.work_dir / "ptoas").glob("*.pto"))
        assert native, "The executed artifact has no generated PTO source"
        text = "\n".join(path.read_text() for path in native)
        assert text.count("pto.tload ins(") == 2
        assert text.count("pto.tstore ins(") == 1
        assert text.count("pto.tadd ins(") == 1
        assert text.count("pto.tmul ins(") == 1
        assert (" addr = " in text) == (case_run.case.memory_planner != passes.MemoryPlanner.PTOAS)


def _executed_native(case_run, request) -> str | None:
    """Native source of the executed artifact, after checking its Buffer stage."""
    case_run.assert_passed()
    assert case_run.case.get_enable_buffer_ir()
    if request.config.getoption("--precompile-workers") is None:
        return None
    assert case_run.work_dir is not None, "Buffer case bypassed the precompile pipeline"
    program = ir.deserialize((case_run.work_dir / "buffer_ir.msgpack").read_bytes())
    assert isinstance(program, ir.Program)
    kernels = [f for f in program.functions.values() if ir.is_incore_type(f.func_type)]
    assert kernels and all(f.ir_stage == ir.FunctionIRStage.Buffer for f in kernels)
    native = list((case_run.work_dir / "ptoas").glob("*.pto"))
    assert native, "The executed artifact has no generated PTO source"
    text = "\n".join(path.read_text() for path in native)
    assert (" addr = " in text) == (case_run.case.memory_planner != passes.MemoryPlanner.PTOAS)
    return text


# Small dyadic operands keep every product and FP32/INT32 partial sum exact.
_M = _K = _N = 64


@pl.jit.incore
def _cube_twice(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    a_mat = pl.load(a, [0, 0], [_M, _K], target_memory=pl.MemorySpace.Mat)
    b_mat = pl.load(b, [0, 0], [_K, _N], target_memory=pl.MemorySpace.Mat)
    a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
    b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
    product = pl.matmul(a_left, b_right)
    total = pl.matmul_acc(product, a_left, b_right)
    return pl.store(total, [0, 0], out)


@pl.jit
def _cube_twice_entry(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    out = _cube_twice(a, b, out)
    return out


@pl.jit.incore
def _cube_k_loop(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    acc: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.tile.create(
        [16, 16], pl.FP32, target_memory=pl.MemorySpace.Acc
    )
    for k0 in pl.range(0, 64, 16):
        a_mat: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, k0], [16, 16], target_memory=pl.MemorySpace.Mat)
        b_mat: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [k0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        acc = pl.matmul_acc(acc, a_mat, b_mat, init_cond=(k0 == 0))
    return pl.store(acc, [0, 0], out)


@pl.jit
def _cube_k_loop_entry(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    out = _cube_k_loop(a, b, out)
    return out


@pl.jit
def _auto_tiled_entry(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tiled"):
        out[:, :] = pl.matmul(a[:, :], b[:, :], out_dtype=pl.FP32)
    return out


@pl.jit
def _transposed_entry(q: pl.Tensor, k: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk"):
        out[:, :] = pl.matmul(q[:, :], k[:, :], b_trans=True, out_dtype=pl.FP32)
    return out


def _operand(rows: int, cols: int, dtype: torch.dtype, modulus: int = 7, scale: int = 4) -> torch.Tensor:
    grid = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    values = grid.remainder(modulus) - modulus // 2
    return (values if dtype == torch.int8 else values / scale).to(dtype)


def _product(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    wide = torch.int64 if dtype == torch.int32 else torch.float64
    return (a.to(wide) @ b.to(wide)).to(dtype)


_CUBE_TYPES = [(torch.float16, torch.float32), (torch.bfloat16, torch.float32), (torch.int8, torch.int32)]


@st.cases(
    *[
        st.case(
            _cube_twice_entry,
            _operand(_M, _K, operand),
            _operand(_K, _N, operand, modulus=5),
            torch.full((_M, _N), -777, dtype=accumulator),
            name=f"buffer_cube_twice_{str(operand).split('.')[1]}_{planner.name.lower()}",
            golden=lambda tensors, accumulator=accumulator: 2
            * _product(tensors["a"], tensors["b"], accumulator),
            memory_planner=planner,
            enable_buffer_ir=True,
            rtol=0,
            atol=0,
        )
        for operand, accumulator in _CUBE_TYPES
        for planner in _PLANNERS
    ]
)
def test_public_buffer_matmul_and_accumulate(case_run, request):
    text = _executed_native(case_run, request)
    if text is not None:
        assert text.count("pto.tmatmul ins(") == 1 and text.count("pto.tmatmul.acc ins(") == 1
        assert text.count("pto.tmov ins(") == 2 and "loc=acc" in text


@st.cases(
    *[
        st.case(
            _cube_k_loop_entry,
            _operand(16, 64, torch.float32),
            _operand(64, 16, torch.float32, modulus=5),
            torch.full((16, 16), -777.0),
            name=f"buffer_cube_init_cond_{planner.name.lower()}",
            golden=lambda tensors: _product(tensors["a"], tensors["b"], torch.float32),
            memory_planner=planner,
            enable_buffer_ir=True,
            rtol=0,
            atol=0,
        )
        for planner in _PLANNERS
    ]
)
def test_public_buffer_runtime_init_cond_overwrites_then_accumulates(case_run, request):
    text = _executed_native(case_run, request)
    if text is not None:
        assert text.count("scf.if") >= 1
        assert "pto.tmatmul ins(" in text and "pto.tmatmul.acc ins(" in text


@st.cases(
    *[
        st.case(
            _auto_tiled_entry,
            _operand(128, 512, torch.bfloat16, modulus=5),
            _operand(512, 128, torch.bfloat16, modulus=3),
            torch.full((128, 128), -777.0),
            name=f"buffer_cube_auto_tiled_{planner.name.lower()}",
            golden=lambda tensors: _product(tensors["a"], tensors["b"], torch.float32),
            memory_planner=planner,
            enable_buffer_ir=True,
            rtol=0,
            atol=0,
        )
        for planner in _PLANNERS
    ]
    + [
        st.case(
            _transposed_entry,
            _operand(16, 128, torch.bfloat16),
            _operand(128, 128, torch.bfloat16, modulus=5),
            torch.full((16, 128), -777.0),
            name=f"buffer_cube_b_trans_{planner.name.lower()}",
            golden=lambda tensors: _product(tensors["q"], tensors["k"].T, torch.float32),
            memory_planner=planner,
            enable_buffer_ir=True,
            rtol=0,
            atol=0,
        )
        for planner in _PLANNERS
    ]
)
def test_public_buffer_tensor_matmul_extracts_and_relabels_operands(case_run, request):
    text = _executed_native(case_run, request)
    if text is not None:
        assert "pto.tmatmul" in text
        assert ("pto.textract ins(" in text) or ("blayout=row_major, slayout=col_major" in text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
