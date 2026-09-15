# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate call frames against actual torch_npu storage and stream contexts."""

import ctypes

import pytest
import torch
from pypto.ir.param_info import ParamInfo
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
from pypto.torch.interop import CallSignature


@pytest.fixture
def npu_context(test_config):
    """Use the selected real NPU; simulator/codegen jobs cannot test framework storage."""
    if test_config.codegen_only or test_config.platform.endswith("sim"):
        pytest.skip("torch_npu metadata checks require a real NPU")
    npu = pytest.importorskip("torch_npu")
    if not npu.npu.is_available():
        pytest.skip("torch_npu reports no available NPU")
    with npu.npu.device(test_config.device_id):
        yield npu


def test_real_npu_offset_alias_and_stream_snapshots(npu_context):
    """Two calls borrow the correct views and current streams without a PyPTO launch."""
    npu = npu_context
    base = torch.empty((4, 3), dtype=torch.float32, device="npu")
    x = base[1:3]
    out = torch.empty_like(x)
    signature = CallSignature(
        [
            ParamInfo("x", ParamDirection.In, [2, 3], DataType.FP32),
            ParamInfo("step", ParamDirection.In, None, DataType.INT32),
            ParamInfo("out", ParamDirection.InOut, [2, 3], DataType.FP32),
        ],
        return_aliases=(2,),
    )
    first_stream, second_stream = npu.npu.Stream(), npu.npu.Stream()
    step = ctypes.c_int32(1)
    with npu.npu.stream(first_stream):
        first = signature.describe_call((x, step, out))
    step.value = 2
    with npu.npu.stream(second_stream):
        second = signature.describe_call((x, step, out))
    assert first.stream == first_stream and second.stream == second_stream
    assert first.scalars[0].value == 1 and second.scalars[0].value == 2
    assert first.tensors[0].metadata.data_ptr == base.data_ptr() + 3 * base.element_size()
    assert first.tensors[0].metadata.storage_offset == 3
    assert first.tensors[0].metadata.format == int(npu.get_npu_format(x))
    assert first.alias_result() is out and second.alias_result() is out


@pytest.mark.parametrize("column_start", [0, 1, 3])
def test_real_npu_empty_slice_storage_offset(npu_context, column_start):
    """Legal empty NPU slices may point at or beyond their storage's end."""
    base = torch.empty((2, 3), device="npu")
    view = base[2:, column_start:]
    signature = CallSignature(
        [ParamInfo("x", ParamDirection.In, [-1, -1], DataType.FP32)], return_aliases=(0,)
    )
    frame = signature.describe_call((view,))
    metadata = frame.tensors[0].metadata
    assert metadata.shape == (0, 3 - column_start)
    assert metadata.storage_offset == 6 + column_start
    assert metadata.nbytes == 0
    assert metadata.storage_ptr == base.untyped_storage().data_ptr()
    assert frame.alias_result() is view


def test_real_npu_rejects_transposed_input(npu_context):
    """A noncontiguous NPU view is rejected rather than copied or normalized."""
    value = torch.empty((3, 2), device="npu").t()
    signature = CallSignature([ParamInfo("x", ParamDirection.In, [2, 3], DataType.FP32)])
    with pytest.raises(ValueError, match="contiguous strided tensor"):
        signature.describe_call((value,))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
