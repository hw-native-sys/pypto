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
@pytest.mark.platforms("a2a3", "a2a3sim", reason="Default Buffer rollout is limited to a2/a3.")
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


@pl.jit.incore
def _static_windows_kernel(source: pl.Tensor, out: pl.InOut[pl.Tensor]):
    whole = pl.load(source, [0, 0], [64, 64])
    outer = pl.tile.slice(whole, [32, 48], [16, 16])
    window = pl.tile.slice(outer, [16, 32], [0, 16])
    doubled = pl.add(window, window)
    target = pl.load(source, [0, 0], [32, 32])
    placed = pl.tile.assemble(target, doubled, [16, 0])
    out = pl.store(placed, [0, 0], out)
    out = pl.store(whole, [64, 0], out)
    return out


@pl.jit
def _static_windows_entry(source: pl.Tensor, out: pl.InOut[pl.Tensor]):
    return _static_windows_kernel(source, out)


@pl.jit.incore
def _overlapping_assemble_kernel(source: pl.Tensor, out: pl.Out[pl.Tensor]):
    whole = pl.load(source, [0, 0], [32, 64])
    window = pl.tile.slice(whole, [32, 32], [0, 0])
    placed = pl.tile.assemble(whole, window, [0, 16])
    return pl.store(placed, [0, 0], out)


@pl.jit
def _overlapping_assemble_entry(source: pl.Tensor, out: pl.Out[pl.Tensor]):
    return _overlapping_assemble_kernel(source, out)


@pl.jit.incore
def _disjoint_assemble_kernel(source: pl.Tensor, out: pl.Out[pl.Tensor]):
    whole = pl.load(source, [0, 0], [128, 256])
    outer = pl.tile.slice(whole, [96, 256], [0, 0])
    window = pl.tile.slice(outer, [64, 256], [0, 0])
    placed = pl.tile.assemble(whole, window, [64, 0])
    return pl.store(placed, [0, 0], out)


@pl.jit
def _disjoint_assemble_entry(source: pl.Tensor, out: pl.Out[pl.Tensor]):
    return _disjoint_assemble_kernel(source, out)


@pl.jit.incore
def _chained_disjoint_assemble_kernel(source: pl.Tensor, out: pl.Out[pl.Tensor]):
    whole = pl.load(source, [0, 0], [128, 256])
    window = pl.tile.slice(whole, [32, 256], [0, 0])
    first = pl.tile.assemble(whole, window, [32, 0])
    second = pl.tile.assemble(first, window, [64, 0])
    updated = pl.tile.slice(second, [32, 256], [64, 0])
    third = pl.tile.assemble(second, updated, [96, 0])
    return pl.store(third, [0, 0], out)


@pl.jit
def _chained_disjoint_assemble_entry(source: pl.Tensor, out: pl.Out[pl.Tensor]):
    return _chained_disjoint_assemble_kernel(source, out)


@pl.jit.incore
def _reinterpret_window_kernel(source: pl.Tensor, out: pl.Out[pl.Tensor]):
    whole = pl.load(source, [0, 0], [32, 64])
    window = pl.tile.slice(whole, [16, 64], [16, 0])
    bits = pl.tile.reinterpret_view(window, pl.INT32)
    return pl.store(bits, [0, 0], out)


@pl.jit
def _reinterpret_window_entry(source: pl.Tensor, out: pl.Out[pl.Tensor]):
    return _reinterpret_window_kernel(source, out)


def _static_windows_golden(tensors):
    source = tensors["source"]
    out = tensors["out"].clone()
    out[:32, :32] = source[:32, :32]
    out[16:32, :32] = source[16:32, 32:] * 2
    out[64:] = source
    return {"out": out}


def _overlapping_assemble_golden(tensors):
    source = tensors["source"]
    out = source.clone()
    out[:, 16:48] = source[:, :32]
    return {"out": out}


def _disjoint_assemble_golden(tensors):
    source = tensors["source"]
    out = source.clone()
    out[64:] = source[:64]
    return {"out": out}


def _chained_disjoint_assemble_golden(tensors):
    return {"out": tensors["source"][:32].repeat(4, 1)}


def _reinterpret_window_golden(tensors):
    return {"out": tensors["source"][16:].view(torch.int32)}


@st.cases(
    *[
        st.case(
            _static_windows_entry,
            torch.arange(4096, dtype=torch.float32).reshape(64, 64).to(dtype),
            torch.full((128, 64), -777, dtype=dtype),
            name=f"static_windows_{planner.name.lower()}_{str(dtype).removeprefix('torch.')}",
            golden=_static_windows_golden,
            memory_planner=planner,
            rtol=0,
            atol=0,
        )
        for planner in _PLANNERS
        for dtype in (torch.float16, torch.float32, torch.int32)
    ],
    *[
        st.case(
            _overlapping_assemble_entry,
            torch.arange(2048, dtype=torch.float32).reshape(32, 64),
            torch.empty((32, 64)),
            name=f"overlapping_assemble_{planner.name.lower()}",
            golden=_overlapping_assemble_golden,
            memory_planner=planner,
            rtol=0,
            atol=0,
        )
        for planner in _PLANNERS
    ],
    *[
        st.case(
            _disjoint_assemble_entry,
            torch.arange(128 * 256, dtype=torch.float32).reshape(128, 256),
            torch.empty((128, 256)),
            name=f"disjoint_assemble_{planner.name.lower()}",
            golden=_disjoint_assemble_golden,
            memory_planner=planner,
            rtol=0,
            atol=0,
        )
        for planner in _PLANNERS
    ],
    *[
        st.case(
            _chained_disjoint_assemble_entry,
            torch.arange(128 * 256, dtype=torch.float32).reshape(128, 256),
            torch.empty((128, 256)),
            name=f"chained_disjoint_assemble_{planner.name.lower()}",
            golden=_chained_disjoint_assemble_golden,
            memory_planner=planner,
            rtol=0,
            atol=0,
        )
        for planner in _PLANNERS
    ],
    *[
        st.case(
            _reinterpret_window_entry,
            torch.arange(2048, dtype=torch.float32).reshape(32, 64),
            torch.empty((16, 64), dtype=torch.int32),
            name=f"reinterpret_window_{planner.name.lower()}",
            golden=_reinterpret_window_golden,
            memory_planner=planner,
            rtol=0,
            atol=0,
        )
        for planner in _PLANNERS
    ],
)
@pytest.mark.platforms("a2a3", "a2a3sim", reason="Static Buffer views default to a2/a3.")
def test_default_static_buffer_windows(case_run, request):
    case_run.assert_passed()
    if request.config.getoption("--precompile-workers") is not None:
        assert case_run.work_dir is not None
        program = ir.deserialize((case_run.work_dir / "buffer_ir.msgpack").read_bytes())
        assert isinstance(program, ir.Program)
        kernels = [
            function for function in program.functions.values() if ir.is_incore_type(function.func_type)
        ]
        assert len(kernels) == 1 and kernels[0].ir_stage == ir.FunctionIRStage.Buffer
        sources = list((case_run.work_dir / "ptoas").glob("*.pto"))
        assert len(sources) == 1
        assert "pto.subview" in sources[0].read_text()
        if "disjoint_assemble" in request.node.name:
            assert sources[0].read_text().count("pto.alloc_tile") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
