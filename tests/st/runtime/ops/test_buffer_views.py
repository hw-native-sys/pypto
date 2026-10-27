# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical proof that automatic Buffer views preserve one allocation's bytes."""

from functools import partial

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto import ir, passes
from pypto.runtime import ChipWorker

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


@pl.jit.incore
def _dynamic_nested_kernel(
    source: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    row: pl.Scalar[pl.INDEX],
    col: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
    cols: pl.Scalar[pl.INDEX],
    base: pl.Scalar[pl.INDEX],
):
    whole = pl.load(source, [0, 0], [64, 64])
    outer = pl.tile.slice(whole, [32, 48], [row, 0])
    window = pl.tile.slice(outer, [16, 32], [8, col * 16])
    value = pl.add(window, window)
    out = pl.store(whole, [base, 0], out)
    return pl.store(value, [base, 0], out)


@pl.jit
def _dynamic_nested_entry(source: pl.Tensor, out: pl.InOut[pl.Tensor]):
    # All invocations share one kernel artifact; empty -> partial -> full ->
    # empty catches metadata left over from a preceding nonempty invocation.
    for index in pl.range(7):
        row = index * 5
        col = index % 2
        if index == 1:
            rows = 1
            cols = 1
        elif index == 2:
            rows = 7
            cols = 13
        elif index == 3:
            rows = 16
            cols = 32
        elif index == 4:
            rows = 16
            cols = 0
        elif index == 5:
            rows = 0
            cols = 0
        else:
            rows = 0
            cols = 32
        base = index * 64
        out = _dynamic_nested_kernel(source, out, row, col, rows, cols, base)
    return out


@pl.jit.incore
def _dynamic_assemble_kernel(
    source: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    row: pl.Scalar[pl.INDEX],
    col: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
    cols: pl.Scalar[pl.INDEX],
    base: pl.Scalar[pl.INDEX],
):
    whole = pl.load(source, [0, 0], [64, 64])
    outer = pl.tile.slice(whole, [32, 64], [row, 0])
    window = pl.tile.slice(outer, [16, 64], [8, 0])
    value = pl.tile.assemble(whole, window, [col * 16, 0])
    return pl.store(value, [base, 0], out)


@pl.jit
def _dynamic_assemble_entry(source: pl.Tensor, out: pl.InOut[pl.Tensor]):
    # All invocations share one kernel artifact; empty -> partial -> full ->
    # empty catches metadata left over from a preceding nonempty invocation.
    for index in pl.range(7):
        row = index * 5
        col = index % 2
        if index == 1:
            rows = 1
            cols = 1
        elif index == 2:
            rows = 7
            cols = 13
        elif index == 3:
            rows = 16
            cols = 32
        elif index == 4:
            rows = 16
            cols = 0
        elif index == 5:
            rows = 0
            cols = 0
        else:
            rows = 0
            cols = 32
        base = index * 64
        out = _dynamic_assemble_kernel(source, out, row, col, rows, cols, base)
    return out


@pl.jit.incore
def _dynamic_valid_load_kernel(
    source: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    row: pl.Scalar[pl.INDEX],
    col: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
    cols: pl.Scalar[pl.INDEX],
    base: pl.Scalar[pl.INDEX],
):
    loaded = pl.load(source, [0, 0], [16, 32], valid_shape=[rows, cols])
    value = pl.add(loaded, loaded)
    return pl.store(value, [base, 0], out)


@pl.jit
def _dynamic_valid_load_entry(source: pl.Tensor, out: pl.InOut[pl.Tensor]):
    # All invocations share one kernel artifact; empty -> partial -> full ->
    # empty catches metadata left over from a preceding nonempty invocation.
    for index in pl.range(7):
        row = index * 5
        col = index % 2
        if index == 1:
            rows = 1
            cols = 1
        elif index == 2:
            rows = 7
            cols = 13
        elif index == 3:
            rows = 16
            cols = 32
        elif index == 4:
            rows = 16
            cols = 0
        elif index == 5:
            rows = 0
            cols = 0
        else:
            rows = 0
            cols = 32
        base = index * 64
        out = _dynamic_valid_load_kernel(source, out, row, col, rows, cols, base)
    return out


@pl.jit.incore
def _dynamic_valid_setter_kernel(
    source: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    row: pl.Scalar[pl.INDEX],
    col: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
    cols: pl.Scalar[pl.INDEX],
    base: pl.Scalar[pl.INDEX],
):
    whole = pl.load(source, [0, 0], [16, 32])
    loaded = pl.tile.set_validshape(whole, rows, cols)
    value = pl.add(loaded, loaded)
    return pl.store(value, [base, 0], out)


@pl.jit
def _dynamic_valid_setter_entry(source: pl.Tensor, out: pl.InOut[pl.Tensor]):
    # All invocations share one kernel artifact; empty -> partial -> full ->
    # empty catches metadata left over from a preceding nonempty invocation.
    for index in pl.range(7):
        row = index * 5
        col = index % 2
        if index == 1:
            rows = 1
            cols = 1
        elif index == 2:
            rows = 7
            cols = 13
        elif index == 3:
            rows = 16
            cols = 32
        elif index == 4:
            rows = 16
            cols = 0
        elif index == 5:
            rows = 0
            cols = 0
        else:
            rows = 0
            cols = 32
        base = index * 64
        out = _dynamic_valid_setter_kernel(source, out, row, col, rows, cols, base)
    return out


@pl.jit.incore
def _dynamic_valid_mixed_kernel(
    source: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    row: pl.Scalar[pl.INDEX],
    col: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
    cols: pl.Scalar[pl.INDEX],
    base: pl.Scalar[pl.INDEX],
):
    whole = pl.load(source, [0, 0], [64, 64])
    window = pl.tile.slice(whole, [16, 32], [row, col * 16], valid_shape=[16, cols])
    loaded = pl.load(source, [row, col * 16], [16, 32], valid_shape=[16, cols])
    value = pl.add(window, loaded)
    return pl.store(value, [base, 0], out)


@pl.jit
def _dynamic_valid_mixed_entry(source: pl.Tensor, out: pl.InOut[pl.Tensor]):
    # All invocations share one kernel artifact; empty -> partial -> full ->
    # empty catches metadata left over from a preceding nonempty invocation.
    for index in pl.range(7):
        row = index * 5
        col = index % 2
        if index == 1:
            rows = 1
            cols = 1
        elif index == 2:
            rows = 7
            cols = 13
        elif index == 3:
            rows = 16
            cols = 32
        elif index == 4:
            rows = 16
            cols = 0
        elif index == 5:
            rows = 0
            cols = 0
        else:
            rows = 0
            cols = 32
        base = index * 64
        out = _dynamic_valid_mixed_kernel(source, out, row, col, rows, cols, base)
    return out


def _dynamic_golden(tensors: dict[str, torch.Tensor], *, kind: str) -> dict[str, torch.Tensor]:
    source = tensors["source"]
    expected = tensors["out"].clone()
    for index, (rows, cols) in enumerate([(0, 32), (1, 1), (7, 13), (16, 32), (16, 0), (0, 0), (0, 32)]):
        row, col, base = index * 5, (index % 2) * 16, index * 64
        if kind == "nested":
            expected[base : base + 64] = source
            expected[base : base + 16, :32] = 2 * source[row + 8 : row + 24, col : col + 32]
        elif kind == "assemble":
            expected[base : base + 64] = source
            expected[base + col : base + col + 16] = source[row + 8 : row + 24].clone()
        elif kind == "valid_mixed":
            expected[base : base + 16, :cols] = 2 * source[row : row + 16, col : col + cols]
        else:
            expected[base : base + rows, :cols] = 2 * source[:rows, :cols]
    return {"out": expected}


@st.cases(
    *[
        st.case(
            entry,
            (torch.arange(4096).reshape(64, 64).remainder(37) - 18).to(dtype),
            torch.full((448, 64), -777, dtype=dtype),
            name=f"dynamic_{kind}_{planner.name.lower()}_{str(dtype).removeprefix('torch.')}",
            golden=partial(_dynamic_golden, kind=kind),
            memory_planner=planner,
            rtol=0,
            atol=0,
        )
        for kind, entry in (
            ("nested", _dynamic_nested_entry),
            ("assemble", _dynamic_assemble_entry),
            ("valid_load", _dynamic_valid_load_entry),
            ("valid_setter", _dynamic_valid_setter_entry),
            ("valid_mixed", _dynamic_valid_mixed_entry),
        )
        for planner in _PLANNERS
        for dtype in (torch.float16, torch.float32, torch.int32)
    ]
)
@pytest.mark.platforms("a2a3", "a2a3sim", reason="Dynamic Vec Buffer rollout is limited to a2/a3.")
def test_default_dynamic_buffer_windows(case_run, request):
    case_run.assert_passed()
    if request.config.getoption("--precompile-workers") is not None:
        assert case_run.work_dir is not None
        program = ir.deserialize((case_run.work_dir / "buffer_ir.msgpack").read_bytes())
        assert isinstance(program, ir.Program)
        kernels = [f for f in program.functions.values() if ir.is_incore_type(f.func_type)]
        assert len(kernels) == 1 and kernels[0].ir_stage == ir.FunctionIRStage.Buffer
        sources = list((case_run.work_dir / "ptoas").glob("*.pto"))
        assert len(sources) == 1
        text = sources[0].read_text()
        if "valid" in case_run.case.name:
            assert "pto.set_validshape" in text and "scf.if" in text
        if (
            "nested" in case_run.case.name
            or "assemble" in case_run.case.name
            or "mixed" in case_run.case.name
        ):
            assert "pto.subview" in text


@pl.jit.incore
def _dynamic_state_kernel(
    source: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    row: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
    cols: pl.Scalar[pl.INDEX],
    count: pl.Scalar[pl.INDEX],
    base: pl.Scalar[pl.INDEX],
):
    whole = pl.load(source, [0, 0], [64, 64])
    window = pl.tile.slice(whole, [16, 32], [row, 16])
    narrowed = pl.tile.set_validshape(window, rows, cols)
    out = pl.store(pl.add(narrowed, narrowed), [base, 0], out)
    out = pl.store(pl.add(window, window), [base + 16, 0], out)
    loaded = pl.load(source, [0, 0], [16, 32])
    partial = pl.tile.set_validshape(loaded, rows, cols)
    out = pl.store(pl.add(partial, partial), [base + 32, 0], out)
    out = pl.store(pl.add(loaded, loaded), [base + 48, 0], out)

    # A dynamic-valid Tile carry preserves its valid expressions. The same
    # allocation may have different metadata after the old-alias reads above.
    if count > 1:
        selected = pl.yield_(pl.add(partial, partial))
    else:
        selected = pl.yield_(partial)
    for _tile_index, (for_value,) in pl.range(count, init_values=(selected,)):
        for_result = pl.yield_(pl.add(for_value, for_value))
    for while_index, while_value in pl.while_(init_values=(0, for_result)):
        pl.cond(while_index < count)
        while_result_index, while_result = pl.yield_(while_index + 1, pl.add(while_value, while_value))
    out = pl.store(while_result, [base + 64, 0], out)

    # Changing valid extents travel as scalar SSA, including simultaneous
    # swaps. Zero iterations must retain the initial scalar state.
    for for_index, (for_width, for_other) in pl.range(count, init_values=(cols, 32)):
        for_view = pl.tile.set_validshape(window, 16, for_width)
        out = pl.store(for_view, [base + 96 + for_index * 16, 0], out)
        final_for_width, final_for_other = pl.yield_(for_other, for_width)
    final_for_view = pl.tile.set_validshape(window, 16, final_for_width)
    out = pl.store(final_for_view, [base + 144, 0], out)
    for state_index, state_width, state_other in pl.while_(init_values=(0, cols, 32)):
        pl.cond(state_index < count)
        state_view = pl.tile.set_validshape(window, 16, state_width)
        out = pl.store(state_view, [base + 160 + state_index * 16, 0], out)
        final_state_index, final_state_width, final_state_other = pl.yield_(
            state_index + 1, state_other, state_width
        )
    final_state_view = pl.tile.set_validshape(window, 16, final_state_width)
    out = pl.store(final_state_view, [base + 208, 0], out)
    return pl.store(whole, [base + 256, 0], out)


@pl.jit
def _dynamic_state_entry(source: pl.Tensor, config: pl.Tensor, out: pl.InOut[pl.Tensor]):
    for index in pl.range(7):
        rows = pl.tensor.read(config, [index, 0])
        cols = pl.tensor.read(config, [index, 1])
        count = pl.tensor.read(config, [index, 2])
        row = index * 5
        base = index * 320
        out = _dynamic_state_kernel(source, out, row, rows, cols, count, base)
    return out


_STATE_CONFIG = torch.tensor(
    [(0, 0, 0), (1, 1, 1), (7, 13, 2), (16, 32, 3), (16, 0, 2), (0, 32, 1), (0, 0, 0)],
    dtype=torch.int64,
)


def _dynamic_state_golden(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    source = tensors["source"]
    expected = tensors["out"].clone()
    for index, (rows, cols, count) in enumerate(tensors["config"].tolist()):
        base = index * 320
        window = source[index * 5 : index * 5 + 16, 16:48]
        expected[base : base + rows, :cols] = 2 * window[:rows, :cols]
        expected[base + 16 : base + 32, :32] = 2 * window
        expected[base + 32 : base + 32 + rows, :cols] = 2 * source[:rows, :cols]
        expected[base + 48 : base + 64, :32] = 2 * source[:16, :32]
        factor = (2 if count > 1 else 1) * 4**count
        expected[base + 64 : base + 64 + rows, :cols] = factor * source[:rows, :cols]
        for offset in (96, 160):
            for iteration in range(count):
                width = cols if iteration % 2 == 0 else 32
                start = base + offset + iteration * 16
                expected[start : start + 16, :width] = window[:, :width]
        width = cols if count % 2 == 0 else 32
        for offset in (144, 208):
            expected[base + offset : base + offset + 16, :width] = window[:, :width]
        expected[base + 256 : base + 320] = source
    return {"out": expected}


@st.cases(
    *[
        st.case(
            _dynamic_state_entry,
            (torch.arange(4096).reshape(64, 64).remainder(17) - 8).to(dtype),
            _STATE_CONFIG,
            torch.full((2240, 64), -777, dtype=dtype),
            name=f"dynamic_state_{planner.name.lower()}_{str(dtype).removeprefix('torch.')}",
            golden=_dynamic_state_golden,
            memory_planner=planner,
            rtol=0,
            atol=0,
        )
        for planner in _PLANNERS
        for dtype in (torch.float16, torch.float32, torch.int32)
    ]
)
@pytest.mark.platforms("a2a3", "a2a3sim", reason="Dynamic Buffer state rollout is limited to a2/a3.")
def test_default_dynamic_buffer_state(case_run, request):
    case_run.assert_passed()
    if request.config.getoption("--precompile-workers") is not None:
        assert case_run.work_dir is not None
        program = ir.deserialize((case_run.work_dir / "buffer_ir.msgpack").read_bytes())
        assert isinstance(program, ir.Program)
        kernels = [f for f in program.functions.values() if ir.is_incore_type(f.func_type)]
        assert len(kernels) == 1 and kernels[0].ir_stage == ir.FunctionIRStage.Buffer
        properties = passes.IRPropertySet()
        properties.insert(passes.IRProperty.BufferIR)
        assert passes.PropertyVerifierRegistry.verify(properties, program) == []
        sources = list((case_run.work_dir / "ptoas").glob("*.pto"))
        assert len(sources) == 1
        text = sources[0].read_text()
        assert "pto.subview" in text and "pto.set_validshape" in text
        assert "scf.if" in text and "scf.for" in text and "scf.while" in text


_NATIVE_PAGES = pl.dynamic("native_pages")
_NATIVE_PAGE_BYTES = pl.dynamic("native_page_bytes")


@pl.jit.incore
def _strided_key_view_kernel(
    native_cache: pl.InOut[pl.Tensor[[_NATIVE_PAGES, _NATIVE_PAGE_BYTES], pl.INT8]],
    query: pl.Tensor[[16, 128], pl.INT8],
    first_page: pl.Tensor[[1], pl.INT32],
    output: pl.InOut[pl.Tensor[[18, 96], pl.INT32]],
):
    pages = pl.tensor.dim(native_cache, 0)
    pitch = pl.tensor.dim(native_cache, 1)
    keys = pl.tensor.view(native_cache, [pages, 32, 128], strides=[pitch, 128, 1])
    page = pl.tensor.read(first_page, [0])
    pair = pl.load(keys, [page, 0, 0], [2, 32, 128], target_memory=pl.Mem.Mat)
    flat = pl.tile.reshape(pair, [64, 128])
    right = pl.move(pl.tile.transpose_view(flat), target_memory=pl.Mem.Right)
    query_mat = pl.load(query, [0, 0], [16, 128], target_memory=pl.Mem.Mat)
    left = pl.move(query_mat, target_memory=pl.Mem.Left)
    product = pl.matmul(left, right, out_dtype=pl.INT32)
    output = pl.store(product, [1, 16], output)
    return native_cache, output


@pl.jit
def _strided_key_view_entry(
    native_cache: pl.InOut[pl.Tensor[[_NATIVE_PAGES, _NATIVE_PAGE_BYTES], pl.INT8]],
    query: pl.Tensor[[16, 128], pl.INT8],
    first_page: pl.Tensor[[1], pl.INT32],
    output: pl.InOut[pl.Tensor[[18, 96], pl.INT32]],
):
    return _strided_key_view_kernel(native_cache, query, first_page, output)


@pytest.mark.platforms("a2a3", "a2a3sim")
@pytest.mark.parametrize("page_bytes", [4160, 4224, 8192])
def test_strided_gm_view_multi_nd_mat_load_and_replay(test_config, page_bytes):
    """Skip page tails while loading adjacent pairs, including odd/last pairs."""
    native_cache = torch.full((9, page_bytes), -101, dtype=torch.int8)
    # Every page/row/column influences the exact dot product. Padding differs
    # from all keys, so treating the page pitch as packed corrupts the result.
    keys = (
        11 * torch.arange(9)[:, None, None]
        + 5 * torch.arange(32)[None, :, None]
        + 3 * torch.arange(128)[None, None, :]
    ) % 31 - 15
    native_cache[:, :4096] = keys.reshape(9, 4096).to(torch.int8)
    original_cache = native_cache.clone()
    query = ((7 * torch.arange(16)[:, None] + 11 * torch.arange(128)[None, :]) % 17 - 8).to(torch.int8)
    first_page = torch.zeros(1, dtype=torch.int32)
    output = torch.full((18, 96), -2000000001, dtype=torch.int32)
    compiled = _strided_key_view_entry.compile(native_cache, query, first_page, output, config=test_config)
    sources = list((compiled.output_dir / "ptoas").glob("*.pto"))
    assert len(sources) == 1
    # Check the actual JIT pipeline as well as the values: one Key TLOAD keeps
    # its 3D GM window and writes the flattened Mat. The query has its own load.
    key_loads = [
        line
        for line in sources[0].read_text().splitlines()
        if "pto.tload ins(" in line and "partition_tensor_view<2x32x128xi8>" in line
    ]
    assert len(key_loads) == 1 and "rows=64, cols=128" in key_loads[0]
    if test_config.codegen_only:
        return

    # Keep one registered program and the same argument objects while the
    # device-read page number changes A -> B -> C -> A. InOut on native_cache
    # requests a readback even for CPU-backed arguments, making this input
    # immutability check observe device writes rather than an untouched host copy.
    with ChipWorker(config=test_config, runtime=compiled.runtime_name) as worker:
        replay = worker.register(compiled)
        for page in (0, 1, 7, 0):
            first_page.fill_(page)
            output.fill_(-2000000001)
            expected = output.clone()
            pair = keys[page : page + 2].reshape(64, 128).to(torch.int32)
            expected[1:17, 16:80] = query.to(torch.int32) @ pair.T
            replay(native_cache, query, first_page, output, config=test_config)
            assert torch.equal(output, expected), (
                f"Wrong pair or damaged guards: pitch={page_bytes}, page={page}"
            )
            assert torch.equal(native_cache, original_cache), (
                f"Native keys or page tails changed: pitch={page_bytes}, page={page}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
