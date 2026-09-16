# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Framework metadata tests using real CPU storage and a stub NPU context."""

import ctypes
import gc
import weakref
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pypto.ir.param_info import ParamInfo
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
from pypto.torch import interop
from torch._subclasses.fake_tensor import FakeTensorMode


class _NPUTensor(torch.Tensor):
    """Use real torch shape/stride/storage operations with a test device label."""

    device_index: int = 0

    @property
    def device(self):
        """Report the stub device without allocating accelerator storage."""
        return SimpleNamespace(type="npu", index=getattr(self, "device_index", 0))


class _Stream:
    """Stream owner whose raw handle must not be requested by metadata code."""

    def __init__(self, index=0):
        """Keep only device metadata; no native stream exists in this fixture."""
        self.device = SimpleNamespace(type="npu", index=index)

    @property
    def npu_stream(self):
        """Fail if metadata adaptation asks for a queue-sensitive raw handle."""
        pytest.fail("metadata adaptation read a raw stream handle")


def _tensor(shape=(2, 3), dtype=torch.float32, device=0):
    """Create real CPU storage while emulating the tensor's NPU device label."""
    tensor = torch.empty(shape, dtype=dtype).as_subclass(_NPUTensor)
    tensor.device_index = device
    return tensor


def _param(name="x", shape=(2, 3), dtype=DataType.FP32, direction=ParamDirection.In):
    """Construct the same parameter metadata used by compiled programs."""
    return ParamInfo(name, direction, list(shape) if shape is not None else None, dtype)


@pytest.fixture
def npu(monkeypatch):
    """Stub only accelerator format and current-context queries."""
    module = SimpleNamespace(
        get_npu_format=Mock(return_value=2),
        npu=SimpleNamespace(current_device=Mock(return_value=0), current_stream=Mock(return_value=_Stream())),
    )
    module.load = Mock(return_value=module)
    monkeypatch.setattr(interop, "_load_torch_npu", module.load)
    return module


def test_each_call_snapshots_values_and_current_stream(npu):
    """Changed scalar/address/stream values never mutate an earlier call frame."""
    signature = interop.CallSignature(
        [_param(), _param("step", None, DataType.INT32), _param("out", direction=ParamDirection.Out)],
        return_aliases=(2,),
    )
    x, first_out, second_out = _tensor(), _tensor(), _tensor()
    first_stream, second_stream = _Stream(), _Stream()
    npu.npu.current_stream.side_effect = [first_stream, second_stream]
    step = ctypes.c_int32(7)
    first = signature.describe_call((x, step, first_out))
    step.value = 9
    second = signature.describe_call((x, step, second_out))
    assert first.scalars[0].value == 7
    assert second.scalars[0].value == 9
    assert [t.metadata.param_index for t in first.tensors] == [0, 2]
    assert first.scalars[0].param_index == 1
    assert first.stream is first_stream and second.stream is second_stream
    assert first.tensors[1].metadata.data_ptr != second.tensors[1].metadata.data_ptr
    assert first.alias_result() is first_out and second.alias_result() is second_out
    assert npu.npu.current_device.call_count == npu.npu.current_stream.call_count == 2
    with pytest.raises(FrozenInstanceError):
        setattr(first, "device_index", 1)


def test_frame_keeps_storage_and_return_aliases_alive(npu):
    """Deleting caller references preserves the exact tensor and its borrowed storage."""
    value = _tensor()
    owner = weakref.ref(value)
    frame = interop.CallSignature([_param()], return_aliases=(0, 0)).describe_call((value,))
    del value
    gc.collect()
    assert owner() is not None
    assert frame.alias_result()[0] is owner()
    assert frame.alias_result()[1] is owner()
    assert frame.tensors[0].storage.data_ptr() == frame.tensors[0].metadata.storage_ptr
    npu.get_npu_format.reset_mock()
    del frame
    gc.collect()
    assert owner() is None


def test_offset_view_uses_logical_pointer_and_immutable_metadata(npu):
    """A contiguous slice keeps its offset and snapshots shape before later mutation."""
    base = _tensor((4, 3))
    view = base[1:3]
    frame = interop.CallSignature([_param()]).describe_call((view,))
    metadata = frame.tensors[0].metadata
    assert metadata.storage_offset == 3
    assert metadata.data_ptr == base.data_ptr() + 3 * base.element_size()
    assert metadata.storage_ptr == base.data_ptr()
    assert metadata.strides == (3, 1)
    view.resize_(6)
    assert metadata.shape == (2, 3)
    assert frame.alias_result() is None


@pytest.mark.parametrize("shape", [(1, 3), (5, 3), (0, 3)])
def test_dynamic_carrier_dimensions(npu, shape):
    """Dynamic axes accept current extents, including an empty view."""
    frame = interop.CallSignature([_param(shape=(-1, 3))]).describe_call((_tensor(shape),))
    assert frame.tensors[0].metadata.shape == shape


@pytest.mark.parametrize("column_start", [0, 1, 3])
def test_empty_slice_can_have_offset_at_or_beyond_storage(npu, column_start):
    """Empty slices access no elements even when their offset exceeds capacity."""
    base = _tensor()
    view = base[2:, column_start:]
    assert view.is_contiguous() and view.numel() == 0
    frame = interop.CallSignature([_param(shape=(-1, -1))], return_aliases=(0,)).describe_call((view,))
    metadata = frame.tensors[0].metadata
    assert metadata.shape == (0, 3 - column_start)
    assert metadata.storage_offset == 6 + column_start
    assert metadata.storage_nbytes == 6 * base.element_size()
    assert metadata.nbytes == 0
    assert metadata.data_ptr == view.data_ptr()
    assert frame.tensors[0].storage.data_ptr() == base.untyped_storage().data_ptr()
    assert frame.alias_result() is view


def test_signature_copies_mutable_parameter_metadata(npu):
    """Mutating the caller's ParamInfo later does not change the prepared signature."""
    param = _param()
    signature = interop.CallSignature([param])
    param.shape[0] = 10
    param.dtype = DataType.INT32
    assert signature.describe_call((_tensor(),)).tensors[0].metadata.shape == (2, 3)


@pytest.mark.parametrize("direction", [ParamDirection.Out, ParamDirection.InOut])
def test_missing_outputs_fail_before_framework_queries(npu, direction):
    """All outputs are mandatory even when an output is also a return alias."""
    signature = interop.CallSignature([_param(), _param("out", direction=direction)], return_aliases=(1,))
    with pytest.raises(TypeError, match="including all Out/InOut"):
        signature.describe_call((_tensor(),))
    npu.load.assert_not_called()


@pytest.mark.parametrize("alias", [-1, 2, True, "0", 1])
def test_invalid_return_aliases(npu, alias):
    """Aliases must index tensor parameters, not scalars or absent slots."""
    with pytest.raises(ValueError, match="Return alias"):
        interop.CallSignature([_param(), _param("step", None, DataType.INT32)], return_aliases=(alias,))
    npu.load.assert_not_called()


def test_metadata_adapter_never_allocates_outputs_or_copies(npu, monkeypatch):
    """Validation preserves caller objects without tensor allocation or normalization."""
    x, out = _tensor(), _tensor()
    signature = interop.CallSignature([_param(), _param("out", direction=ParamDirection.Out)])
    forbidden = Mock(side_effect=AssertionError("unexpected allocation or copy"))
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "zeros", forbidden)
    monkeypatch.setattr(_NPUTensor, "contiguous", forbidden)
    monkeypatch.setattr(_NPUTensor, "to", forbidden)
    frame = signature.describe_call((x, out))
    assert frame.tensors[1].tensor is out
    forbidden.assert_not_called()


@pytest.mark.parametrize("kind", ["host", "meta", "fake", "foreign", "dtype", "shape", "stride", "grad"])
def test_invalid_tensor_inputs_fail_before_npu_loading(npu, kind):
    """Reject invalid tensor metadata before current-device or stream side effects."""
    tensor = _tensor()
    if kind == "host":
        tensor = torch.empty(2, 3)
    elif kind == "meta":
        tensor = torch.empty(2, 3, device="meta")
    elif kind == "fake":
        with FakeTensorMode():
            tensor = torch.empty(2, 3)
    elif kind == "foreign":
        tensor = SimpleNamespace(shape=(2, 3), dtype=torch.float32, data_ptr=lambda: 1)
    elif kind == "dtype":
        tensor = _tensor(dtype=torch.int32)
    elif kind == "shape":
        tensor = _tensor((2, 4))
    elif kind == "stride":
        tensor = _tensor((3, 2)).t()
    elif kind == "grad":
        tensor.requires_grad_(True)
    error = TypeError if kind in ("host", "meta", "fake", "foreign", "dtype") else ValueError
    with pytest.raises(error, match="Parameter 'x'"):
        interop.CallSignature([_param()]).describe_call((tensor,))
    npu.load.assert_not_called()


def test_mixed_devices_are_rejected_before_framework_queries(npu):
    """Tensor device mismatches do not select or change the current device."""
    with pytest.raises(ValueError, match="requires one NPU device"):
        interop.CallSignature([_param(), _param("y")]).describe_call((_tensor(), _tensor(device=1)))
    npu.load.assert_not_called()


def test_current_device_and_stream_must_match(npu):
    """The adapter reads the caller's context without switching devices."""
    signature = interop.CallSignature([_param()])
    npu.npu.current_device.return_value = 1
    with pytest.raises(ValueError, match="differ from current device"):
        signature.describe_call((_tensor(),))
    npu.npu.current_stream.assert_not_called()
    npu.npu.current_device.return_value = 0
    npu.npu.current_stream.return_value = _Stream(1)
    with pytest.raises(ValueError, match="current stream belongs"):
        signature.describe_call((_tensor(),))


@pytest.mark.parametrize("tensor_format", [1, 3, 29, -1])
def test_unsupported_storage_formats_fail_before_current_context(npu, tensor_format):
    """Hidden physical layouts are not silently interpreted as dense ND storage."""
    npu.get_npu_format.return_value = tensor_format
    with pytest.raises(ValueError, match="requires base format"):
        interop.CallSignature([_param()]).describe_call((_tensor(),))
    npu.npu.current_device.assert_not_called()


@pytest.mark.parametrize("tensor_format", [0, 2])
def test_supported_base_formats_are_preserved(npu, tensor_format):
    """Record the actual supported format without casting the tensor."""
    npu.get_npu_format.return_value = tensor_format
    frame = interop.CallSignature([_param()]).describe_call((_tensor(),))
    assert frame.tensors[0].metadata.format == tensor_format


@pytest.mark.parametrize("writable", [False, True])
def test_partial_overlap_requires_read_only_arguments(npu, writable):
    """Read-only views may overlap; partial writes need a richer alias contract."""
    base = _tensor((10,))
    signature = interop.CallSignature(
        [
            _param("x", (6,)),
            _param("y", (6,), direction=ParamDirection.InOut if writable else ParamDirection.In),
        ]
    )
    if writable:
        with pytest.raises(ValueError, match="writable alias"):
            signature.describe_call((base[:6], base[4:]))
        npu.npu.current_device.assert_not_called()
    else:
        assert len(signature.describe_call((base[:6], base[4:])).tensors) == 2


def test_exact_alias_and_disjoint_views_are_accepted(npu):
    """Sharing a storage owner does not by itself imply unsupported overlap."""
    base = _tensor((12,))
    signature = interop.CallSignature(
        [_param("x", (6,)), _param("out", (6,), direction=ParamDirection.Out), _param("y", (6,))],
        return_aliases=(1,),
    )
    first, second = base[:6], base[6:]
    frame = signature.describe_call((first, first, second))
    assert frame.alias_result() is first
    assert {t.metadata.storage_ptr for t in frame.tensors} == {base.data_ptr()}


@pytest.mark.parametrize(
    ("dtype", "value", "expected"),
    [
        (DataType.INT8, -128, -128),
        (DataType.UINT8, 255, 255),
        (DataType.INT32, 12, 12),
        (DataType.INT64, -(1 << 63), -(1 << 63)),
        (DataType.UINT64, (1 << 64) - 1, (1 << 64) - 1),
        (DataType.INDEX, 4, 4),
        (DataType.FP32, 1.25, 1.25),
        (DataType.BF16, 2.5, 2.5),
        (DataType.FP16, 0.5, 0.5),
        (DataType.BOOL, True, True),
    ],
)
def test_scalar_only_frames_use_current_context(npu, dtype, value, expected):
    """Scalar-only calls still obtain an NPU context and retain runtime values."""
    frame = interop.CallSignature([_param("value", None, dtype)]).describe_call((value,))
    assert frame.tensors == ()
    assert frame.scalars[0].value == expected
    assert type(frame.scalars[0].value) is type(expected)
    npu.npu.current_stream.assert_called_once_with(0)


@pytest.mark.parametrize(
    ("dtype", "value", "error"),
    [
        (DataType.INT8, 128, ValueError),
        (DataType.UINT8, -1, ValueError),
        (DataType.INT32, 1.5, TypeError),
        (DataType.INT32, True, TypeError),
        (DataType.BOOL, 1, TypeError),
        (DataType.FP32, "1", TypeError),
        (DataType.INT32, ctypes.c_int64(1), TypeError),
    ],
)
def test_invalid_scalars_fail_before_framework_queries(npu, dtype, value, error):
    """Reject implicit integer truncation, overflow and incorrect scalar types."""
    with pytest.raises(error, match="Parameter 'value'"):
        interop.CallSignature([_param("value", None, dtype)]).describe_call((value,))
    npu.load.assert_not_called()


def test_optional_torch_npu_dependency_error(monkeypatch):
    """Missing framework support gives a targeted error only when it is needed."""
    original = interop.importlib.import_module

    def import_module(name):
        """Hide torch_npu while preserving every unrelated module import."""
        if name == "torch_npu":
            raise ModuleNotFoundError("torch_npu missing")
        return original(name)

    monkeypatch.setattr(interop.importlib, "import_module", import_module)
    with pytest.raises(RuntimeError, match="requires a compatible torch_npu"):
        interop.CallSignature([_param("n", None, DataType.INT32)]).describe_call((1,))


def test_import_does_not_require_npu_or_simpler(run_without_optional_runtime):
    """A fresh process can import the package with optional runtime imports forbidden."""
    source = """
import pypto
import pypto.torch
import pypto.torch.interop
assert pypto.torch.__all__ == []
"""
    result = run_without_optional_runtime(source)
    assert result.returncode == 0, result.stderr


def test_inference_context_allows_gradient_owned_inputs(npu):
    """Inference may borrow model parameters without promising backward support."""
    tensor = _tensor().requires_grad_(True)
    with torch.no_grad():
        frame = interop.CallSignature([_param()]).describe_call((tensor,))
    assert frame.tensors[0].tensor is tensor


@pytest.mark.parametrize("device", [-1, None, True])
def test_scalar_only_call_requires_valid_device(npu, device):
    """Scalar-only calls do not bypass validation of the current NPU context."""
    npu.npu.current_device.return_value = device
    with pytest.raises(ValueError, match="valid current NPU device"):
        interop.CallSignature([_param("n", None, DataType.INT32)]).describe_call((1,))
    npu.npu.current_stream.assert_not_called()


@pytest.mark.parametrize("invalid", ["offset", "pointer"])
def test_invalid_storage_descriptions_fail_before_current_context(npu, monkeypatch, invalid):
    """Reject a logical view outside its owner or a pointer that ignores the offset."""
    tensor = _tensor()
    if invalid == "offset":
        monkeypatch.setattr(_NPUTensor, "storage_offset", lambda self: 1000)
    else:
        monkeypatch.setattr(_NPUTensor, "data_ptr", lambda self: 0)
    with pytest.raises(ValueError, match="storage bounds|logical data pointer"):
        interop.CallSignature([_param()]).describe_call((tensor,))
    npu.npu.current_device.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
