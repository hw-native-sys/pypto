# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical proof that automatic Buffer views preserve one allocation's bytes."""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto import ir, passes

_PLANNERS = (passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS)


@pl.jit.incore
def _view_kernel(source: pl.Tensor, out: pl.InOut[pl.Tensor]):
    loaded = pl.load(source, [0, 0], [16, 32])
    first = pl.reshape(loaded, [8, 64])
    out = pl.store(first, [0, 0], out)
    out = pl.store(loaded, [16, 0], out)
    second = pl.reshape(first, [4, 128])
    out = pl.store(second, [40, 0], out)
    return out


@pl.jit
def _view_entry(source: pl.Tensor, out: pl.InOut[pl.Tensor]):
    out = _view_kernel(source, out)
    return out


def _golden(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    source = tensors["source"]
    out = tensors["out"].clone()
    # Store each independent interpretation plus the original, with untouched
    # sentinel rows and columns. No inverse chain can hide a bad intermediate.
    out[:8, :64] = source.reshape(8, 64)
    out[16:32, :32] = source
    out[40:44, :128] = source.reshape(4, 128)
    return {"out": out}


_SOURCE = (torch.arange(512, dtype=torch.float32).reshape(16, 32) * 13 - 1777) / 16


@st.cases(
    *[
        st.case(
            _view_entry,
            _SOURCE.to(dtype),
            torch.full((48, 128), -777, dtype=dtype),
            name=f"buffer_views_{planner.name.lower()}_{str(dtype).removeprefix('torch.')}",
            golden=_golden,
            memory_planner=planner,
            enable_buffer_ir=True,
            rtol=0,
            atol=0,
        )
        for planner in _PLANNERS
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int32)
    ]
)
def test_public_buffer_views(case_run, request):
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
        properties = passes.IRPropertySet()
        properties.insert(passes.IRProperty.BufferIR)
        assert passes.PropertyVerifierRegistry.verify(properties, program) == []
        sources = list((case_run.work_dir / "ptoas").glob("*.pto"))
        assert len(sources) == 1, "Expected the source of the executed device kernel"
        text = sources[0].read_text()
        assert text.count("pto.alloc_tile") == 1
        assert text.count("pto.treshape") == 3
        assert text.count("pto.tload ins(") == 1 and text.count("pto.tstore ins(") == 3
        assert "pto.tmov" not in text and "pto.textract" not in text
        assert (" addr = " in text) == (case_run.case.memory_planner != passes.MemoryPlanner.PTOAS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
