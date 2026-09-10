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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
