# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Focused tests for simpler wire ``MappedArg`` to torch conversion."""

import ctypes
from types import SimpleNamespace

import torch
from pypto.runtime.distributed_runner import _tensor_from_continuous


class _Float32:
    def __str__(self) -> str:
        return "DataType.FLOAT32"


def test_mapped_arg_buffer_is_zero_copy():
    backing = bytearray(6 * torch.tensor([], dtype=torch.float32).element_size())
    source = torch.arange(6, dtype=torch.float32)
    torch.frombuffer(backing, dtype=torch.float32).copy_(source)
    arg = SimpleNamespace(buffer=memoryview(backing), shapes=(2, 3), strides=(3, 1), dtype=_Float32())

    tensor = _tensor_from_continuous(arg)

    torch.testing.assert_close(tensor, source.reshape(2, 3))
    tensor[1, 2] = 99
    assert torch.frombuffer(backing, dtype=torch.float32)[5].item() == 99


def test_mapped_arg_strides_are_preserved():
    backing = bytearray(6 * torch.tensor([], dtype=torch.float32).element_size())
    torch.frombuffer(backing, dtype=torch.float32).copy_(torch.arange(6, dtype=torch.float32))
    arg = SimpleNamespace(buffer=memoryview(backing), shapes=(2, 2), strides=(3, 1), dtype=_Float32())

    tensor = _tensor_from_continuous(arg)

    torch.testing.assert_close(tensor, torch.tensor([[0.0, 1.0], [3.0, 4.0]]))


def test_legacy_data_pointer_remains_supported():
    backing = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    arg = SimpleNamespace(data=ctypes.addressof(backing), shapes=(2, 2), dtype=_Float32())

    tensor = _tensor_from_continuous(arg)

    torch.testing.assert_close(tensor, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_unrecognized_arg_shape_fails_explicitly():
    arg = SimpleNamespace(shapes=(1,), dtype=_Float32())

    try:
        _tensor_from_continuous(arg)
    except TypeError as exc:
        assert "MappedArg" in str(exc)
    else:
        raise AssertionError("missing .buffer/.data should be rejected")
