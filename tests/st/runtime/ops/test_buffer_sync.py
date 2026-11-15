# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 manual Buffer-ID RAW/WAR/WAW and loop-reuse runtime acceptance.

Run with --platform=a5 (device). The harness saves inputs,
golden data, PTO/C++, version and execution logs and enforces its execution
watchdog. Each test parameter performs an independent launch. Named MemRef
slots keep physical ping/pong allocations aligned with the dynamic IDs.
Automatic event insertion is disabled only by this test fixture so the numeric
result depends on the manual protocol. Default-pipeline compilation is covered
by the unit integration test.

The entry uses @pl.jit with an explicit @pl.jit.aiv kernel. The fixture also
forwards the case execution mode explicitly through the shared compile harness.
"""

import json
from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto import ir
from pypto.backend import pto_backend
from pypto.runtime import ExecutionMode

ITERATIONS = 8
ROWS = 16
COLS = 32


def make_buffer_sync_program(dynamic: bool):
    """Create an explicit AIV kernel with separate input/output allocations."""
    slot_count = 2 if dynamic else 1

    @pl.jit.aiv
    def kernel(
        x: pl.Tensor[[ITERATIONS * ROWS, COLS], pl.FP32],
        out: pl.Out[pl.Tensor[[ITERATIONS * ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ITERATIONS * ROWS, COLS], pl.FP32]:
        for i in pl.range(ITERATIONS):
            slot: pl.Scalar[pl.INDEX] = i % slot_count
            if dynamic:
                pl.system.get_buf(slot, pipe=pl.PipeType.MTE2)
            else:
                pl.system.get_buf(0, pipe=pl.PipeType.MTE2)
            tile: pl.Tile[[ROWS, COLS], pl.FP32, pl.MemRef("input_slots", slots=2)[slot], pl.Mem.Vec] = (
                pl.tile.load(x, [i * ROWS, 0], [ROWS, COLS])
            )
            if dynamic:
                pl.system.rls_buf(slot, pipe=pl.PipeType.MTE2)
                pl.system.get_buf(slot, pipe=pl.PipeType.V)
                pl.system.get_buf(slot + 2, pipe=pl.PipeType.V)
            else:
                pl.system.rls_buf(0, pipe=pl.PipeType.MTE2)
                pl.system.get_buf(0, pipe=pl.PipeType.V)
                pl.system.get_buf(2, pipe=pl.PipeType.V)
            result: pl.Tile[[ROWS, COLS], pl.FP32, pl.MemRef("output_slots", slots=2)[slot], pl.Mem.Vec] = (
                pl.tile.add(tile, 1.0)
            )
            if dynamic:
                pl.system.rls_buf(slot, pipe=pl.PipeType.V)
                pl.system.rls_buf(slot + 2, pipe=pl.PipeType.V)
                pl.system.get_buf(slot + 2, pipe=pl.PipeType.MTE3)
            else:
                pl.system.rls_buf(0, pipe=pl.PipeType.V)
                pl.system.rls_buf(2, pipe=pl.PipeType.V)
                pl.system.get_buf(2, pipe=pl.PipeType.MTE3)
            pl.tile.store(result, [i * ROWS, 0], out)
            if dynamic:
                pl.system.rls_buf(slot + 2, pipe=pl.PipeType.MTE3)
            else:
                pl.system.rls_buf(2, pipe=pl.PipeType.MTE3)
        return out

    @pl.jit
    def orchestrator(
        x: pl.Tensor[[ITERATIONS * ROWS, COLS], pl.FP32],
        out: pl.Out[pl.Tensor[[ITERATIONS * ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ITERATIONS * ROWS, COLS], pl.FP32]:
        out = kernel(x, out)
        return out

    return orchestrator.specialize()


class BufferSyncCase(PTOTestCase):
    """Data changes every iteration; compare every output element independently."""

    __test__ = False

    def __init__(self, dynamic: bool, launch: int, **kwargs):
        super().__init__(**kwargs)
        self.dynamic = dynamic
        self.launch = launch

    def get_name(self) -> str:
        return f"manual_buffer_sync_{'dynamic' if self.dynamic else 'static'}_{self.launch}"

    def define_tensors(self) -> list[TensorSpec]:
        generator = torch.Generator().manual_seed(41 + self.launch)
        data = torch.randn(ITERATIONS * ROWS, COLS, generator=generator)
        return [
            TensorSpec("x", list(data.shape), DataType.FP32, init_value=data),
            TensorSpec("out", list(data.shape), DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return make_buffer_sync_program(self.dynamic)

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["x"] + 1.0


@pytest.fixture
def manual_sync_only(monkeypatch, test_config):
    """Exercise manual ordering without adding a public auto-sync policy API."""
    assert test_config.execution_mode is ExecutionMode.ONBOARD
    original = pto_backend._get_ptoas_flags
    original_compile = ir.compile

    def compile_onboard(*args, **kwargs):
        # The shared harness currently drops RunConfig.execution_mode.
        # Pin this hardware-only acceptance case to its requested mode.
        kwargs["execution_mode"] = test_config.execution_mode
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(ir, "compile", compile_onboard)
    seen: list[list[str]] = []

    def flags(*args, **kwargs):
        result = [flag for flag in original(*args, **kwargs) if flag != "--enable-insert-sync"]
        seen.append(result)
        return result

    monkeypatch.setenv("PYPTO_CACHE", "0")
    monkeypatch.setattr(pto_backend, "_get_ptoas_flags", flags)
    return seen


@pytest.mark.inline_case(reason="Manual sync and onboard-mode compiler patches require the test fixtures")
@pytest.mark.platforms("a5")
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "pingpong"])
@pytest.mark.parametrize("launch", [0, 1, 2])
def test_manual_buffer_sync(test_runner, test_config, dynamic, launch, manual_sync_only):
    case = BufferSyncCase(dynamic, launch, config=test_config)
    result = test_runner.run(case)
    assert result.passed, f"Manual Buffer-ID failed: {result.error}"
    assert manual_sync_only, "Manual acceptance must compile with the test's native compiler flags"
    if test_config.save_kernels_dir:
        case_dir = Path(test_config.save_kernels_dir) / case.get_name()
        metadata = json.loads((case_dir / "compiled_meta.json").read_text())
        assert metadata["platform"] == "a5"
        sources = list((case_dir / "kernels").rglob("*.cpp"))
        assert sources
        for source in sources:
            text = source.read_text()
            _, _, start, end = pto_backend._find_ptoas_function(source.stem, text)
            text = text[start:end]
            assert "get_buf(PIPE_MTE2," in text
            assert "get_buf(PIPE_V," in text
            assert "get_buf(PIPE_MTE3," in text
            assert "set_flag(" not in text and "wait_flag(" not in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
