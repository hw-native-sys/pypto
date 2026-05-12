# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board ST for ``pl.tile.comm_notify`` + ``pl.tile.comm_wait``.

Single-rank loopback that mirrors the two real usage patterns from simpler's
``examples/workers/l3/ep_dispatch_combine`` (commit c65dda3):

1. ``count_exchange`` — atomic-add a count into a remote rank's slot.
   Models ``dispatch.cpp``'s ``TNOTIFY(pub_counts_local + idx, v,
   NotifyOp::AtomicAdd)`` where a producer rank publishes its per-expert
   element count into the consumer rank's slot.

2. ``done_barrier`` — atomic-add 1 into a remote rank's done slot, then
   spin-wait on the local slot until it reaches the expected value.
   Models the cross-rank "I'm done" barrier in both ``dispatch.cpp`` and
   ``combine.cpp``:

       for peer != my_rank:
           TNOTIFY(remote_done_sig[my_rank], 1, NotifyOp::AtomicAdd)
       for src != my_rank:
           TWAIT(local_done_sig[src], 1, WaitCmp::GE)

Single-rank loopback uses the same slot for both notify and wait, so the
"remote" and "local" views collapse — this exercises the codegen and runtime
paths for ``pto.comm.tnotify`` / ``pto.comm.twait`` without binding a real
HCCL window.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# --- Programs ---


@pl.program
class CountExchangeProgram:
    """Pattern 1: atomic-add a count into a remote rank's slot.

    Slot pre-initialized to 3; kernel adds 5; expected final value 8.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        pl.tile.comm_notify(signal, 5, op="atomic_add")
        return signal

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        return self.kernel(signal)


@pl.program
class DoneBarrierProgram:
    """Pattern 2: notify(atomic_add, 1) → wait(ge, 1) — the done barrier.

    Slot pre-initialized to 0; kernel adds 1 then spin-waits on ge 1.
    Expected final value 1.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        pl.tile.comm_notify(signal, 1, op="atomic_add")
        pl.tile.comm_wait(signal, 1, cmp="ge")
        return signal

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        return self.kernel(signal)


# --- Test cases ---


class _CommSignalBase(PTOTestCase):
    __test__ = False

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


class CountExchangeTestCase(_CommSignalBase):
    def get_name(self) -> str:
        return "comm_signal_count_exchange"

    def define_tensors(self) -> list[TensorSpec]:
        # Pre-initialize the slot to 3; atomic_add 5 → 8.
        return [TensorSpec("signal", [1], DataType.INT32, init_value=3, is_output=True)]

    def get_program(self) -> Any:
        return CountExchangeProgram

    def compute_expected(self, tensors, params=None):
        tensors["signal"][:] = torch.tensor([8], dtype=torch.int32)


class DoneBarrierTestCase(_CommSignalBase):
    def get_name(self) -> str:
        return "comm_signal_done_barrier"

    def define_tensors(self) -> list[TensorSpec]:
        # Pre-initialize the slot to 0; atomic_add 1 → 1; wait ge 1 returns immediately.
        return [TensorSpec("signal", [1], DataType.INT32, init_value=0, is_output=True)]

    def get_program(self) -> Any:
        return DoneBarrierProgram

    def compute_expected(self, tensors, params=None):
        tensors["signal"][:] = torch.tensor([1], dtype=torch.int32)


# --- Tests ---


class TestCommNotifyWait:
    """On-board verification of pl.tile.comm_notify + pl.tile.comm_wait (single-rank loopback).

    Mirrors the two real usage patterns from simpler's ep_dispatch_combine
    kernels: count exchange (atomic-add) and the done barrier (atomic-add +
    wait ge).
    """

    def test_count_exchange(self, test_runner):
        result = test_runner.run(CountExchangeTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_done_barrier(self, test_runner):
        result = test_runner.run(DoneBarrierTestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
