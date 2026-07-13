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

import numpy as np
import pypto.language as pl
import pytest
import torch
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


_COLVEC_ROWS = 16
_COLVEC_STEPS = 4


@pl.program
class ColVecIfPhiCarryProgram:
    """Online-softmax-shaped ``[N, 1]`` col-vector loop-carried if-phi.

    Two carries recur through an ``if``/``else`` inside a ``pl.range`` loop:

    - ``s`` (the ``li`` accumulator) is yielded straight from ``pl.mul`` /
      ``pl.add`` — its yield source is the ``[N, 1]`` reshape-back.
    - ``m`` (the ``mi`` running max) is ``m = m_new`` where ``m_new`` is ALSO
      consumed as a ``[1, N]`` intermediate (the ``exp(m - m_new)`` rescale),
      so ``m``'s yield is an SSA bare alias of the reshape-back rather than the
      reshape node itself.

    Under ``memory_planner=PTOAS`` the ``m`` bare-alias must resolve to the
    ``[N, 1]`` view SSA so its branch write-back ``pto.tmov`` gets matching
    src/dst shapes; binding it to the shared ``[1, N]`` op-result handle emits a
    ``[1, N] -> [N, 1]`` tmov that ptoas rejects. This is the regression case
    for that codegen fix; ``s`` is the already-correct control.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[_COLVEC_ROWS, 1], pl.FP32],
        y: pl.Tensor[[_COLVEC_ROWS, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[_COLVEC_ROWS, 1], pl.FP32]],
        acc: pl.Out[pl.Tensor[[_COLVEC_ROWS, 1], pl.FP32]],
    ) -> pl.Tensor[[_COLVEC_ROWS, 1], pl.FP32]:
        m: pl.Tile[[_COLVEC_ROWS, 1], pl.FP32] = pl.load(x, [0, 0], [_COLVEC_ROWS, 1])
        s: pl.Tile[[_COLVEC_ROWS, 1], pl.FP32] = pl.load(x, [0, 0], [_COLVEC_ROWS, 1])
        for i in pl.range(_COLVEC_STEPS):
            c: pl.Tile[[_COLVEC_ROWS, 1], pl.FP32] = pl.load(y, [0, 0], [_COLVEC_ROWS, 1])
            if i == 0:
                m_new = pl.maximum(m, c)
                alpha = pl.exp(pl.sub(m, m_new))
                s = pl.mul(s, alpha)
                m = m_new
            else:
                m_new = pl.maximum(m, c)
                alpha = pl.exp(pl.sub(m, m_new))
                beta = pl.exp(pl.sub(c, m_new))
                s = pl.add(pl.mul(s, alpha), beta)
                m = m_new
        out = pl.store(m, [0, 0], out)
        acc = pl.store(s, [0, 0], acc)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[_COLVEC_ROWS, 1], pl.FP32],
        y: pl.Tensor[[_COLVEC_ROWS, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[_COLVEC_ROWS, 1], pl.FP32]],
        acc: pl.Out[pl.Tensor[[_COLVEC_ROWS, 1], pl.FP32]],
    ) -> pl.Tensor[[_COLVEC_ROWS, 1], pl.FP32]:
        out = self.kernel(x, y, out, acc)
        return out


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


def _colvec_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Distinct per-row non-zero inputs so a dropped carry / wrong row cannot
    accidentally match the golden (a zero-input run would pass on a no-op)."""
    x = torch.arange(_COLVEC_ROWS, dtype=torch.float32).reshape(_COLVEC_ROWS, 1) * 0.1 - 0.5
    y = torch.arange(_COLVEC_ROWS, dtype=torch.float32).reshape(_COLVEC_ROWS, 1) * 0.05 + 0.25
    return x, y


class ColVecIfPhiCarryCase(PTOTestCase):
    """``[N, 1]`` col-vector loop-carried if-phi, run under the given planner."""

    def __init__(self, memory_planner: MemoryPlanner | None = None, *, platform=None, config=None):
        super().__init__(config, platform=platform, memory_planner=memory_planner)
        self._mp = memory_planner
        self._x, self._y = _colvec_inputs()

    def get_name(self) -> str:
        return f"memplan_colvec_ifphi_carry_{_planner_tag(self._mp)}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [_COLVEC_ROWS, 1], DataType.FP32, init_value=self._x),
            TensorSpec("y", [_COLVEC_ROWS, 1], DataType.FP32, init_value=self._y),
            TensorSpec("out", [_COLVEC_ROWS, 1], DataType.FP32, is_output=True),
            TensorSpec("acc", [_COLVEC_ROWS, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColVecIfPhiCarryProgram

    def compute_expected(self, tensors, params=None) -> None:
        x = np.asarray(tensors["x"], dtype=np.float64)
        y = np.asarray(tensors["y"], dtype=np.float64)
        m = x.copy()
        s = x.copy()
        for i in range(_COLVEC_STEPS):
            c = y
            m_new = np.maximum(m, c)
            alpha = np.exp(m - m_new)
            if i == 0:
                s = s * alpha
            else:
                beta = np.exp(c - m_new)
                s = s * alpha + beta
            m = m_new
        tensors["out"][:] = torch.from_numpy(m.astype(np.float32))
        tensors["acc"][:] = torch.from_numpy(s.astype(np.float32))


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

    @pytest.mark.parametrize("planner", _PLANNERS, ids=_planner_tag)
    def test_colvec_ifphi_carry(self, test_runner, planner):
        # PTOAS is the regression case: an ``[N, 1]`` col-vector loop-carried
        # if-phi whose ``m = m_new`` yield is an SSA bare alias of the reshaped
        # branch value. The branch write-back tmov must move the ``[N, 1]`` view,
        # not the shared ``[1, N]`` op-result buffer.
        result = test_runner.run(ColVecIfPhiCarryCase(planner))
        assert result.passed, f"colvec if-phi carry ({_planner_tag(planner)}) failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
