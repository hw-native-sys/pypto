# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify direct L2 dispatch builds worker-owned wire ``TaskArgs``.

The high-level simpler ``Worker.run`` contract accepts ``TaskArgs`` containing
address-free ``Tensor`` values. It must never receive the direct-chip
``ChipStorageTaskArgs`` / ``ChipTensor`` pair.
"""

import ctypes
from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto.runtime import DeviceTensor


class FakeBuffer:
    def __init__(self, base: int) -> None:
        self.base = base
        self.tensor_calls: list[tuple[tuple[int, ...], object]] = []
        self.wire_tensor = MagicMock(name=f"Tensor(0x{base:x})")

    def tensor(self, *, shapes, dtype):
        self.tensor_calls.append((tuple(shapes), dtype))
        return self.wire_tensor


@pytest.fixture
def task_args_fixture():
    task_args = MagicMock(name="TaskArgs")
    owner = MagicMock(name="simpler_worker")
    with patch("pypto.runtime.task_interface.TaskArgs", return_value=task_args):
        yield task_args, owner


def test_host_tensor_uses_worker_aware_tensor_helper(task_args_fixture):
    task_args, owner = task_args_fixture
    host = torch.zeros(4, 4, dtype=torch.float32)
    wire_tensor = MagicMock(name="wire_tensor")

    with patch("simpler_setup.torch_interop.make_tensor_arg", return_value=wire_tensor) as make:
        from pypto.runtime.runner import _coerced_to_orch_args  # noqa: PLC0415

        result = _coerced_to_orch_args([host], owner)

    assert result is task_args
    make.assert_called_once_with(owner, host)
    task_args.add_tensor.assert_called_once_with(wire_tensor)


def test_device_tensor_uses_retained_buffer_tensor(task_args_fixture):
    task_args, owner = task_args_fixture
    buffer = FakeBuffer(0xABCD)
    device_tensor = DeviceTensor.from_buffer(buffer, (8, 16), torch.float16)

    from pypto.runtime.runner import _coerced_to_orch_args  # noqa: PLC0415

    _coerced_to_orch_args([device_tensor], owner)

    assert len(buffer.tensor_calls) == 1
    shapes, _dtype = buffer.tensor_calls[0]
    assert shapes == (8, 16)
    task_args.add_tensor.assert_called_once_with(buffer.wire_tensor)


def test_raw_pointer_device_tensor_is_rejected(task_args_fixture):
    _task_args, owner = task_args_fixture
    raw = DeviceTensor(0xABCD, (8,), torch.float16)

    from pypto.runtime.runner import _coerced_to_orch_args  # noqa: PLC0415

    with pytest.raises(ValueError, match="raw-pointer DeviceTensor"):
        _coerced_to_orch_args([raw], owner)


def test_tensors_keep_order_and_scalars_use_separate_pool(task_args_fixture):
    task_args, owner = task_args_fixture
    first = torch.zeros(2, dtype=torch.float32)
    second = torch.ones(2, dtype=torch.float32)
    scalar = ctypes.c_int32(7)
    wire_first = MagicMock(name="wire_first")
    wire_second = MagicMock(name="wire_second")

    with patch("simpler_setup.torch_interop.make_tensor_arg", side_effect=[wire_first, wire_second]):
        from pypto.runtime.runner import _coerced_to_orch_args  # noqa: PLC0415

        _coerced_to_orch_args([first, scalar, second], owner)

    assert task_args.add_tensor.call_args_list[0].args == (wire_first,)
    assert task_args.add_tensor.call_args_list[1].args == (wire_second,)
    assert task_args.add_scalar.call_count == 1
