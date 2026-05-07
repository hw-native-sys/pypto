# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board ST for ``pl.tile.notify`` + ``pl.tile.wait``.

Single-rank loopback: the signal tensor passed as a kernel argument is a
1-element INT32 GM tensor.  Within one kernel we ``notify`` it (write or
atomic-add) and immediately ``wait`` on the same slot.  The kernel returns
the signal as an output, which the harness reads back and validates.

This validates the codegen + runtime path for ``pto.comm.tnotify`` and
``pto.comm.twait`` without needing real cross-rank HCCL window binding.

Cases:
  notify_set_wait_eq      — set 7;  wait eq 7
  notify_add_wait_ge      — init 3 + atomic_add 4 = 7;  wait ge 5
  notify_set_wait_gt      — set 10; wait gt 5
  notify_set_wait_lt      — set 1;  wait lt 5
  notify_set_wait_le      — set 2;  wait le 5
  notify_set_wait_ne      — set 9;  wait ne 5
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
class NotifySetWaitEqProgram:
    """notify(set, 7) → wait(eq, 7)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        pl.tile.notify(signal, 7, op="set")
        pl.tile.wait(signal, 7, cmp="eq")
        return signal

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        return self.kernel(signal)


@pl.program
class NotifyAddWaitGeProgram:
    """notify(atomic_add, 4) on a slot pre-initialized to 3 → 7; wait(ge, 5)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        pl.tile.notify(signal, 4, op="atomic_add")
        pl.tile.wait(signal, 5, cmp="ge")
        return signal

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        return self.kernel(signal)


@pl.program
class NotifySetWaitGtProgram:
    """notify(set, 10) → wait(gt, 5)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        pl.tile.notify(signal, 10, op="set")
        pl.tile.wait(signal, 5, cmp="gt")
        return signal

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        return self.kernel(signal)


@pl.program
class NotifySetWaitLtProgram:
    """notify(set, 1) → wait(lt, 5)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        pl.tile.notify(signal, 1, op="set")
        pl.tile.wait(signal, 5, cmp="lt")
        return signal

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        return self.kernel(signal)


@pl.program
class NotifySetWaitLeProgram:
    """notify(set, 2) → wait(le, 5)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        pl.tile.notify(signal, 2, op="set")
        pl.tile.wait(signal, 5, cmp="le")
        return signal

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        return self.kernel(signal)


@pl.program
class NotifySetWaitNeProgram:
    """notify(set, 9) → wait(ne, 5)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        pl.tile.notify(signal, 9, op="set")
        pl.tile.wait(signal, 5, cmp="ne")
        return signal

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        signal: pl.Out[pl.Tensor[[1], pl.INT32]],
    ) -> pl.Tensor[[1], pl.INT32]:
        return self.kernel(signal)


# --- Test cases ---


class _NotifyWaitBase(PTOTestCase):
    __test__ = False

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


class NotifySetWaitEqTestCase(_NotifyWaitBase):
    def get_name(self) -> str:
        return "notify_set_wait_eq"

    def define_tensors(self) -> list[TensorSpec]:
        return [TensorSpec("signal", [1], DataType.INT32, init_value=0, is_output=True)]

    def get_program(self) -> Any:
        return NotifySetWaitEqProgram

    def compute_expected(self, tensors, params=None):
        tensors["signal"][:] = torch.tensor([7], dtype=torch.int32)


class NotifyAddWaitGeTestCase(_NotifyWaitBase):
    def get_name(self) -> str:
        return "notify_add_wait_ge"

    def define_tensors(self) -> list[TensorSpec]:
        # Pre-initialize the slot to 3; atomic_add 4 → 7.
        return [TensorSpec("signal", [1], DataType.INT32, init_value=3, is_output=True)]

    def get_program(self) -> Any:
        return NotifyAddWaitGeProgram

    def compute_expected(self, tensors, params=None):
        tensors["signal"][:] = torch.tensor([7], dtype=torch.int32)


class NotifySetWaitGtTestCase(_NotifyWaitBase):
    def get_name(self) -> str:
        return "notify_set_wait_gt"

    def define_tensors(self) -> list[TensorSpec]:
        return [TensorSpec("signal", [1], DataType.INT32, init_value=0, is_output=True)]

    def get_program(self) -> Any:
        return NotifySetWaitGtProgram

    def compute_expected(self, tensors, params=None):
        tensors["signal"][:] = torch.tensor([10], dtype=torch.int32)


class NotifySetWaitLtTestCase(_NotifyWaitBase):
    def get_name(self) -> str:
        return "notify_set_wait_lt"

    def define_tensors(self) -> list[TensorSpec]:
        return [TensorSpec("signal", [1], DataType.INT32, init_value=0, is_output=True)]

    def get_program(self) -> Any:
        return NotifySetWaitLtProgram

    def compute_expected(self, tensors, params=None):
        tensors["signal"][:] = torch.tensor([1], dtype=torch.int32)


class NotifySetWaitLeTestCase(_NotifyWaitBase):
    def get_name(self) -> str:
        return "notify_set_wait_le"

    def define_tensors(self) -> list[TensorSpec]:
        return [TensorSpec("signal", [1], DataType.INT32, init_value=0, is_output=True)]

    def get_program(self) -> Any:
        return NotifySetWaitLeProgram

    def compute_expected(self, tensors, params=None):
        tensors["signal"][:] = torch.tensor([2], dtype=torch.int32)


class NotifySetWaitNeTestCase(_NotifyWaitBase):
    def get_name(self) -> str:
        return "notify_set_wait_ne"

    def define_tensors(self) -> list[TensorSpec]:
        return [TensorSpec("signal", [1], DataType.INT32, init_value=0, is_output=True)]

    def get_program(self) -> Any:
        return NotifySetWaitNeProgram

    def compute_expected(self, tensors, params=None):
        tensors["signal"][:] = torch.tensor([9], dtype=torch.int32)


# --- Tests ---


class TestNotifyWait:
    """On-board verification of pl.tile.notify + pl.tile.wait (single-rank loopback)."""

    def test_notify_set_wait_eq(self, test_runner):
        result = test_runner.run(NotifySetWaitEqTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_notify_add_wait_ge(self, test_runner):
        result = test_runner.run(NotifyAddWaitGeTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_notify_set_wait_gt(self, test_runner):
        result = test_runner.run(NotifySetWaitGtTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_notify_set_wait_lt(self, test_runner):
        result = test_runner.run(NotifySetWaitLtTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_notify_set_wait_le(self, test_runner):
        result = test_runner.run(NotifySetWaitLeTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_notify_set_wait_ne(self, test_runner):
        result = test_runner.run(NotifySetWaitNeTestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
