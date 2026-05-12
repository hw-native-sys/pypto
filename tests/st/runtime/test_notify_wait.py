# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board ST for ``pl.tile.comm_notify`` + ``pl.tile.comm_wait``.

Single-rank loopback exercises three codegen paths:

1. ``count_exchange`` — notify-only: atomic-add 5 into a slot pre-set to 3,
   expect 8.
2. ``wait_only`` — wait-only: pre-set the slot to 7, ``comm_wait ge 1``
   must return immediately without touching the slot.
3. ``done_barrier`` — notify + wait combined: atomic-add 1, then wait ge 1.
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
        signal = self.kernel(signal)
        return signal


@pl.program
class WaitOnlyProgram:
    """Wait-only: slot pre-set to 7; ``comm_wait ge 1`` returns immediately.

    Isolates the ``pto.comm.twait`` codegen path with no notify in the kernel.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        pl.tile.comm_wait(signal, 1, cmp="ge")
        return signal

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        signal = self.kernel(signal)
        return signal


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
        signal = self.kernel(signal)
        return signal


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


class WaitOnlyTestCase(_CommSignalBase):
    def get_name(self) -> str:
        return "comm_signal_wait_only"

    def define_tensors(self) -> list[TensorSpec]:
        # Pre-initialize the slot to 7; wait ge 1 returns immediately, slot unchanged.
        return [TensorSpec("signal", [1], DataType.INT32, init_value=7, is_output=True)]

    def get_program(self) -> Any:
        return WaitOnlyProgram

    def compute_expected(self, tensors, params=None):
        tensors["signal"][:] = torch.tensor([7], dtype=torch.int32)


# --- Tests ---


class TestCommNotifyWait:
    """On-board verification of pl.tile.comm_notify + pl.tile.comm_wait (single-rank loopback)."""

    def test_count_exchange(self, test_runner):
        result = test_runner.run(CountExchangeTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_wait_only(self, test_runner):
        result = test_runner.run(WaitOnlyTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_done_barrier(self, test_runner):
        result = test_runner.run(DoneBarrierTestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
