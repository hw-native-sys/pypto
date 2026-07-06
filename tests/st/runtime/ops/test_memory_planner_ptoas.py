# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end runtime tests for ``compile(memory_planner=MemoryPlanner.PTOAS)``.

Under ``PTOAS`` the pipeline skips PyPTO's opportunistic ``MemoryReuse`` and
``AllocateMemoryAddr`` and lets the ptoas ``PlanMemory`` pass own lifetime reuse
and address assignment at ``--pto-level=level2``. ``MaterializeSemanticAliases``
still runs, so semantics-required aliasing (loop-carried accumulators, in-place
ops) is preserved as a shared ``tile_buf`` handle.

Each kernel is run under **both** planners against the same golden — a PTOAS
result that matches the PYPTO result proves the must-alias handoff is correct.
The loop-carried accumulator is the regression case: without
``MaterializeSemanticAliases`` the addr-less allocs would be planned into
distinct ptoas buffers and the accumulation would be silently lost.
"""

from typing import Any

import pypto.language as pl
import pytest
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.pypto_core.passes import MemoryPlanner


def _planner_tag(mp: MemoryPlanner | None) -> str:
    return "ptoas" if mp == MemoryPlanner.PTOAS else "pypto"


# ---------------------------------------------------------------------------
# Kernel programs
# ---------------------------------------------------------------------------


@pl.program
class ElementwiseAddProgram:
    """c = a + b on a single 64x64 tile (no aliasing — basic PTOAS path)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        ta: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])
        tb: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [0, 0], [64, 64])
        tc: pl.Tile[[64, 64], pl.FP32] = pl.add(ta, tb)
        c = pl.store(tc, [0, 0], c)
        return c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        c = self.kernel(a, b, c)
        return c


@pl.program
class LoopAccumProgram:
    """Loop-carried tile accumulator: acc must stay one buffer across iterations.

    Loads 4 chunks of 64x64 (all 2.0) and accumulates into a single carried
    tile via yield. Expected: c[:] = 4 * 2.0 = 8.0. This is the must-alias
    regression case for memory_planner=PTOAS.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_accum(
        self,
        a: pl.Tensor[[256, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_init: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])
        for i, (acc,) in pl.range(1, 4, init_values=(tile_init,)):
            offset_i = i * 64
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [offset_i, 0], [64, 64])
            new_acc: pl.Tile[[64, 64], pl.FP32] = pl.add(acc, tile_a)
            result = pl.yield_(new_acc)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], c)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[256, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        c = self.kernel_accum(a, c)
        return c


# ---------------------------------------------------------------------------
# Test cases (parametrized by memory planner)
# ---------------------------------------------------------------------------


class ElementwiseAddCase(PTOTestCase):
    """c = a + b, run under the given memory planner."""

    def __init__(self, memory_planner: MemoryPlanner | None = None, *, platform=None, config=None):
        super().__init__(config, platform=platform, memory_planner=memory_planner)
        self._mp = memory_planner

    def get_name(self) -> str:
        return f"memplan_elementwise_add_{_planner_tag(self._mp)}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [64, 64], DataType.FP32, init_value=3.0),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ElementwiseAddProgram

    def compute_expected(self, tensors, params=None) -> None:
        tensors["c"][:] = tensors["a"] + tensors["b"]


class LoopAccumCase(PTOTestCase):
    """Loop-carried accumulator, run under the given memory planner."""

    def __init__(self, memory_planner: MemoryPlanner | None = None, *, platform=None, config=None):
        super().__init__(config, platform=platform, memory_planner=memory_planner)
        self._mp = memory_planner

    def get_name(self) -> str:
        return f"memplan_loop_accum_{_planner_tag(self._mp)}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [256, 64], DataType.FP32, init_value=2.0),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return LoopAccumProgram

    def compute_expected(self, tensors, params=None) -> None:
        tensors["c"][:] = 4 * 2.0


# ---------------------------------------------------------------------------
# pytest wrappers
# ---------------------------------------------------------------------------


_PLANNERS = [MemoryPlanner.PYPTO, MemoryPlanner.PTOAS]


class TestMemoryPlannerPtoas:
    """PTOAS memory planner produces correct on-device results (matches PYPTO)."""

    @pytest.mark.parametrize("planner", _PLANNERS, ids=_planner_tag)
    def test_elementwise_add(self, test_runner, planner):
        result = test_runner.run(ElementwiseAddCase(planner))
        assert result.passed, f"elementwise add ({_planner_tag(planner)}) failed: {result.error}"

    @pytest.mark.parametrize("planner", _PLANNERS, ids=_planner_tag)
    def test_loop_carried_accumulator(self, test_runner, planner):
        # PTOAS is the regression case: the loop-carried accumulator must stay in
        # one buffer even though MemoryReuse/AllocateMemoryAddr are skipped.
        result = test_runner.run(LoopAccumCase(planner))
        assert result.passed, f"loop accumulator ({_planner_tag(planner)}) failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
