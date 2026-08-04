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

The multi-buffer cases cover the other direction — a declared allocation PTOAS is
told to keep *apart*. ``pl.MemRef(slots=N)`` becomes one ``pto.alloc_multi_tile``
region plus a ``pto.multi_tile_get`` per use, so ptoas plans the slots and derives
per-slot synchronization from the slot index. Under PYPTO the same source keeps
the baked-address ``alloc_tile`` path (see ``test_memref_slots.py``), so running
both planners against one golden is what shows the region form is equivalent.

The double-buffer case is the shape the region form exists for — one slot live per
iteration — and its golden checks the WAR edge ptoas derives from the slot index.
Two slots live at once inside a loop is **rejected** under PTOAS: ptoas 0.54 guards
only the first ``multi_tile_get`` of an iteration, so the second is read while the
next iteration overwrites it. That was measured wrong on device before codegen
started refusing it, and ``MultiBufferCoLiveProgram`` is the case that pins the
refusal.
"""

from typing import Any

import numpy as np
import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto import backend as _backend
from pypto import ir
from pypto.backend import BackendType
from pypto.backend.pto_backend import PartialCodegenError
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


# Vec multi-buffer: STEPS row tiles of [TR, TC] fp32 — 16 KB per slot.
_MB_TR, _MB_TC = 64, 64
_MB_STEPS = 4
_MB_ROWS = _MB_TR * _MB_STEPS

# Acc multi-buffer: [M, K] @ [K, N] in NT column tiles. One fp32 slot is
# M * _MB_TN * 4 = 16 KB, so the pair fits L0C with room to spare.
_MB_M, _MB_K, _MB_N = 64, 64, 256
_MB_TN = 64
_MB_NT = _MB_N // _MB_TN

MULTI_BUF_COLIVE = pl.MemRef(slots=2)
MULTI_BUF_ACC = pl.MemRef(slots=2)
MULTI_BUF_CONST = pl.MemRef(slots=2)


@pl.program
class MultiBufferCoLiveProgram:
    """Two UB slots co-live per iteration — rejected under PTOAS, correct under PYPTO.

    ptoas derives the per-slot WAR guard only for the first ``multi_tile_get`` of a
    region in an iteration; with two co-live slots the second load is unguarded and
    the next iteration overwrites the slot while this one is still reading it.
    Measured wrong on device (ptoas 0.54): ``out`` came back as ``a[block i+1] +
    a[block i]`` instead of ``a + b``. Codegen therefore refuses the shape, and this
    kernel is the ST that pins that refusal — under PYPTO the same source is fine.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[_MB_ROWS, _MB_TC], pl.FP32],
        b: pl.Tensor[[_MB_ROWS, _MB_TC], pl.FP32],
        out: pl.Out[pl.Tensor[[_MB_ROWS, _MB_TC], pl.FP32]],
    ) -> pl.Tensor[[_MB_ROWS, _MB_TC], pl.FP32]:
        for i in pl.range(_MB_STEPS):
            lo: pl.Tile[[_MB_TR, _MB_TC], pl.FP32, MULTI_BUF_COLIVE[i % 2], pl.Mem.Vec] = pl.load(
                a, [i * _MB_TR, 0], [_MB_TR, _MB_TC], target_memory=pl.Mem.Vec
            )
            hi: pl.Tile[[_MB_TR, _MB_TC], pl.FP32, MULTI_BUF_COLIVE[(i + 1) % 2], pl.Mem.Vec] = pl.load(
                b, [i * _MB_TR, 0], [_MB_TR, _MB_TC], target_memory=pl.Mem.Vec
            )
            # Both slots are still live here — collapse them and `lo` is gone.
            s: pl.Tile[[_MB_TR, _MB_TC], pl.FP32, pl.Mem.Vec] = pl.add(lo, hi)
            out = pl.store(s, [i * _MB_TR, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[_MB_ROWS, _MB_TC], pl.FP32],
        b: pl.Tensor[[_MB_ROWS, _MB_TC], pl.FP32],
        out: pl.Out[pl.Tensor[[_MB_ROWS, _MB_TC], pl.FP32]],
    ) -> pl.Tensor[[_MB_ROWS, _MB_TC], pl.FP32]:
        out = self.kernel(a, b, out)
        return out


MULTI_BUF_DB = pl.MemRef(slots=2)


@pl.program
class MultiBufferDoubleBufferProgram:
    """The ping-pong the region form exists for: ONE slot live per iteration.

    Iteration i loads into slot ``i % 2`` and consumes it; iteration i+1 loads into
    the other slot while i's add is still running. The correctness gate is the WAR
    edge — if i+1's load lands in a slot before i's add has read it, the sum is
    wrong — so the golden checks the per-slot synchronization, not just addressing.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[_MB_ROWS, _MB_TC], pl.FP32],
        out: pl.Out[pl.Tensor[[_MB_TR, _MB_TC], pl.FP32]],
    ) -> pl.Tensor[[_MB_TR, _MB_TC], pl.FP32]:
        seed: pl.Tile[[_MB_TR, _MB_TC], pl.FP32, pl.Mem.Vec] = pl.load(
            a, [0, 0], [_MB_TR, _MB_TC], target_memory=pl.Mem.Vec
        )
        for i, (acc,) in pl.range(1, _MB_STEPS, init_values=(seed,)):
            t: pl.Tile[[_MB_TR, _MB_TC], pl.FP32, MULTI_BUF_DB[i % 2], pl.Mem.Vec] = pl.load(
                a, [i * _MB_TR, 0], [_MB_TR, _MB_TC], target_memory=pl.Mem.Vec
            )
            nxt: pl.Tile[[_MB_TR, _MB_TC], pl.FP32, pl.Mem.Vec] = pl.add(acc, t)
            r = pl.yield_(nxt)
        out = pl.store(r, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[_MB_ROWS, _MB_TC], pl.FP32],
        out: pl.Out[pl.Tensor[[_MB_TR, _MB_TC], pl.FP32]],
    ) -> pl.Tensor[[_MB_TR, _MB_TC], pl.FP32]:
        out = self.kernel(a, out)
        return out


@pl.program
class MultiBufferConstSlotProgram:
    """The same allocation selected by two *constant* slots, both co-live.

    A constant index folds to a `ConstInt`, so this is the case where the region's
    operand is an `arith.constant` rather than a `remsi` — one region still backs
    both, and the two halves must not share storage.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[_MB_TR, _MB_TC], pl.FP32],
        b: pl.Tensor[[_MB_TR, _MB_TC], pl.FP32],
        out: pl.Out[pl.Tensor[[_MB_TR, _MB_TC], pl.FP32]],
    ) -> pl.Tensor[[_MB_TR, _MB_TC], pl.FP32]:
        lo: pl.Tile[[_MB_TR, _MB_TC], pl.FP32, MULTI_BUF_CONST[0], pl.Mem.Vec] = pl.load(
            a, [0, 0], [_MB_TR, _MB_TC], target_memory=pl.Mem.Vec
        )
        hi: pl.Tile[[_MB_TR, _MB_TC], pl.FP32, MULTI_BUF_CONST[1], pl.Mem.Vec] = pl.load(
            b, [0, 0], [_MB_TR, _MB_TC], target_memory=pl.Mem.Vec
        )
        s: pl.Tile[[_MB_TR, _MB_TC], pl.FP32, pl.Mem.Vec] = pl.sub(lo, hi)
        out = pl.store(s, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[_MB_TR, _MB_TC], pl.FP32],
        b: pl.Tensor[[_MB_TR, _MB_TC], pl.FP32],
        out: pl.Out[pl.Tensor[[_MB_TR, _MB_TC], pl.FP32]],
    ) -> pl.Tensor[[_MB_TR, _MB_TC], pl.FP32]:
        out = self.kernel(a, b, out)
        return out


@pl.program
class MultiBufferAccProgram:
    """The L0C double buffer — the Acc memory space of a region, one slot per iteration.

    Acc is the third space `IsMultiBufferMemorySpace` admits, and the only one whose
    slots are written by MAD and drained by FIXPIPE rather than MTE2/V. The golden
    checks that WAR edge: if iteration i+1's matmul lands in a slot before iteration
    i's store has drained it, the column tile is wrong.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[_MB_M, _MB_K], pl.FP32],
        b: pl.Tensor[[_MB_K, _MB_N], pl.FP32],
        out: pl.Out[pl.Tensor[[_MB_M, _MB_N], pl.FP32]],
    ) -> pl.Tensor[[_MB_M, _MB_N], pl.FP32]:
        la: pl.Tile[[_MB_M, _MB_K], pl.FP32, pl.Mem.Mat] = pl.load(
            a, [0, 0], [_MB_M, _MB_K], target_memory=pl.Mem.Mat
        )
        for i in pl.range(_MB_NT):
            lb: pl.Tile[[_MB_K, _MB_TN], pl.FP32, pl.Mem.Mat] = pl.load(
                b, [0, i * _MB_TN], [_MB_K, _MB_TN], target_memory=pl.Mem.Mat
            )
            # One L0C slot live per iteration: iteration i+1's MAD into the other
            # slot overlaps iteration i's FIXPIPE drain out of this one.
            cur: pl.Tile[[_MB_M, _MB_TN], pl.FP32, MULTI_BUF_ACC[i % 2], pl.Mem.Acc] = pl.tile.matmul(la, lb)
            out = pl.store(cur, [0, i * _MB_TN], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[_MB_M, _MB_K], pl.FP32],
        b: pl.Tensor[[_MB_K, _MB_N], pl.FP32],
        out: pl.Out[pl.Tensor[[_MB_M, _MB_N], pl.FP32]],
    ) -> pl.Tensor[[_MB_M, _MB_N], pl.FP32]:
        out = self.kernel(a, b, out)
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


def _ramp(rows: int, cols: int, scale: float, bias: float) -> torch.Tensor:
    """Per-element distinct values, so a wrong slot *or* a wrong row is visible.

    A constant fill would let a kernel that read the wrong tile still match."""
    n = rows * cols
    return (torch.arange(n, dtype=torch.float32).reshape(rows, cols) * scale + bias) / n


class MultiBufferCoLiveCase(PTOTestCase):
    """Two co-live UB slots — runnable under PYPTO, refused by PTOAS codegen."""

    def __init__(self, memory_planner: MemoryPlanner | None = None, *, platform=None, config=None):
        super().__init__(config, platform=platform, memory_planner=memory_planner)
        self._mp = memory_planner

    def get_name(self) -> str:
        return f"memplan_multi_buffer_colive_{_planner_tag(self._mp)}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [_MB_ROWS, _MB_TC], DataType.FP32, init_value=_ramp(_MB_ROWS, _MB_TC, 1.0, 0.0)),
            TensorSpec("b", [_MB_ROWS, _MB_TC], DataType.FP32, init_value=_ramp(_MB_ROWS, _MB_TC, -3.0, 7.0)),
            TensorSpec("out", [_MB_ROWS, _MB_TC], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MultiBufferCoLiveProgram

    def compute_expected(self, tensors, params=None) -> None:
        tensors["out"][:] = tensors["a"] + tensors["b"]


class MultiBufferDoubleBufferCase(PTOTestCase):
    """One slot live per iteration, run under the given memory planner."""

    def __init__(self, memory_planner: MemoryPlanner | None = None, *, platform=None, config=None):
        super().__init__(config, platform=platform, memory_planner=memory_planner)
        self._mp = memory_planner

    def get_name(self) -> str:
        return f"memplan_multi_buffer_db_{_planner_tag(self._mp)}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [_MB_ROWS, _MB_TC], DataType.FP32, init_value=_ramp(_MB_ROWS, _MB_TC, 1.0, 0.0)),
            TensorSpec("out", [_MB_TR, _MB_TC], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MultiBufferDoubleBufferProgram

    def compute_expected(self, tensors, params=None) -> None:
        a = tensors["a"]
        acc = a[0:_MB_TR, :].clone()
        for i in range(1, _MB_STEPS):
            acc += a[i * _MB_TR : (i + 1) * _MB_TR, :]
        tensors["out"][:] = acc


class MultiBufferConstSlotCase(PTOTestCase):
    """Two constant slots of one region, run under the given memory planner."""

    def __init__(self, memory_planner: MemoryPlanner | None = None, *, platform=None, config=None):
        super().__init__(config, platform=platform, memory_planner=memory_planner)
        self._mp = memory_planner

    def get_name(self) -> str:
        return f"memplan_multi_buffer_const_{_planner_tag(self._mp)}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [_MB_TR, _MB_TC], DataType.FP32, init_value=_ramp(_MB_TR, _MB_TC, 1.0, 0.0)),
            TensorSpec("b", [_MB_TR, _MB_TC], DataType.FP32, init_value=_ramp(_MB_TR, _MB_TC, -3.0, 7.0)),
            TensorSpec("out", [_MB_TR, _MB_TC], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MultiBufferConstSlotProgram

    def compute_expected(self, tensors, params=None) -> None:
        # Subtraction, not addition: `a - b` and `b - a` differ, so a swapped slot
        # is caught as well as a collapsed one.
        tensors["out"][:] = tensors["a"] - tensors["b"]


class MultiBufferAccCase(PTOTestCase):
    """Rotating L0C slots, run under the given memory planner."""

    def __init__(self, memory_planner: MemoryPlanner | None = None, *, platform=None, config=None):
        super().__init__(config, platform=platform, memory_planner=memory_planner)
        self._mp = memory_planner

    def get_name(self) -> str:
        return f"memplan_multi_buffer_acc_{_planner_tag(self._mp)}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [_MB_M, _MB_K], DataType.FP32, init_value=_ramp(_MB_M, _MB_K, 1.0, 0.0)),
            TensorSpec("b", [_MB_K, _MB_N], DataType.FP32, init_value=_ramp(_MB_K, _MB_N, 2.0, -1.0)),
            TensorSpec("out", [_MB_M, _MB_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MultiBufferAccProgram

    def compute_expected(self, tensors, params=None) -> None:
        tensors["out"][:] = tensors["a"] @ tensors["b"]


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

    def test_multi_buffer_colive_slots_run_under_pypto(self, test_runner):
        # Two co-live slots are a legal program — the baked-address path runs it
        # correctly. This is the control for the PTOAS refusal below: what is being
        # rejected there is the lowering, not the source.
        result = test_runner.run(MultiBufferCoLiveCase(MemoryPlanner.PYPTO))
        assert result.passed, f"multi-buffer co-live (pypto) failed: {result.error}"

    def test_multi_buffer_colive_slots_rejected_under_ptoas(self):
        # ptoas 0.54 emits the per-slot WAR pair only for the first multi_tile_get
        # of an iteration, so the second slot is read while the next iteration
        # overwrites it — measured wrong on device. Refuse rather than miscompile.
        #
        # Asserted here through the real `ir.compile` entry point, which surfaces a
        # kernel's ValueError as PartialCodegenError; the exact reason string is
        # pinned in tests/ut/codegen/test_multi_buffer_codegen.py.
        _backend.reset_for_testing()
        _backend.set_backend_type(BackendType.Ascend910B)
        with pytest.raises(PartialCodegenError):
            ir.compile(MultiBufferCoLiveProgram, memory_planner=MemoryPlanner.PTOAS)

    @pytest.mark.parametrize("planner", _PLANNERS, ids=_planner_tag)
    def test_multi_buffer_double_buffer(self, test_runner, planner):
        # The shape the region form exists for: one slot live per iteration, so the
        # WAR edge between iteration i's add and i+1's load into the other slot is
        # what the golden checks.
        result = test_runner.run(MultiBufferDoubleBufferCase(planner))
        assert result.passed, f"multi-buffer double buffer ({_planner_tag(planner)}) failed: {result.error}"

    @pytest.mark.parametrize("planner", _PLANNERS, ids=_planner_tag)
    def test_multi_buffer_constant_slots(self, test_runner, planner):
        # Constant slot indices share one region; the two halves must stay apart.
        result = test_runner.run(MultiBufferConstSlotCase(planner))
        assert result.passed, f"multi-buffer const slots ({_planner_tag(planner)}) failed: {result.error}"

    @pytest.mark.parametrize("planner", _PLANNERS, ids=_planner_tag)
    def test_multi_buffer_acc_rotating_slots(self, test_runner, planner):
        # The Acc (L0C) space of a region: MAD writes the slot, FIXPIPE drains it.
        result = test_runner.run(MultiBufferAccCase(planner))
        assert result.passed, f"multi-buffer acc ({_planner_tag(planner)}) failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
