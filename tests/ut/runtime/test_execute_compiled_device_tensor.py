# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify ``execute_compiled`` translates :class:`DeviceTensor` arguments to
``ContinuousTensor.make(..., child_memory=True)`` while ``torch.Tensor``
arguments still take the ordinary ``make_tensor_arg`` path.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto.runtime import DeviceTensor

# ``device_runner`` and ``task_interface`` eagerly import the optional
# ``simpler`` runtime package; skip the module when simpler is unavailable
# (the same pattern test_worker_reuse.py uses for execute_on_device tests).
try:
    import simpler  # noqa: F401  # pyright: ignore[reportMissingImports]
except ImportError:
    _has_simpler = False
else:
    _has_simpler = True

pytestmark = pytest.mark.skipif(not _has_simpler, reason="execute_compiled requires the simpler package")


@pytest.fixture
def patched_runtime(tmp_path):
    """Patch every import inside ``execute_compiled`` so it runs without a device.

    Captures the per-arg list passed to ``orch_args.add_tensor`` so individual
    tests can assert on the resulting ContinuousTensor descriptors.
    """
    captured: dict = {
        "tensors": [],
        "scalars": [],
        "make_calls": [],  # ContinuousTensor.make kwargs
        "make_tensor_arg_calls": [],
    }

    chip_args = MagicMock(name="ChipStorageTaskArgs_instance")

    def _record_tensor(t):
        captured["tensors"].append(t)

    def _record_scalar(s):
        captured["scalars"].append(s)

    chip_args.add_tensor.side_effect = _record_tensor
    chip_args.add_scalar.side_effect = _record_scalar

    def _make(*, data, shapes, dtype, child_memory=False):
        captured["make_calls"].append(
            {
                "data": data,
                "shapes": tuple(shapes),
                "dtype": dtype,
                "child_memory": child_memory,
            }
        )
        return MagicMock(name=f"ContinuousTensor(0x{data:x})")

    def _make_tensor_arg(t):
        captured["make_tensor_arg_calls"].append(t)
        return MagicMock(name="ContinuousTensor(host)")

    def _torch_dtype_to_datatype(dt):
        # Sentinel — not asserted on directly; child_memory and shape carry the signal.
        return f"<dtype:{dt}>"

    with (
        patch("pypto.runtime.runner._patch_orchestration_headers"),
        patch(
            "pypto.runtime.device_runner.compile_and_assemble",
            return_value=(MagicMock(name="chip_callable"), "host_build_graph", {}),
        ),
        patch("pypto.runtime.device_runner.execute_on_device"),
        patch("pypto.runtime.device_runner.ChipStorageTaskArgs", return_value=chip_args),
        patch("pypto.runtime.device_runner.make_tensor_arg", side_effect=_make_tensor_arg),
        patch("pypto.runtime.device_runner.scalar_to_uint64", side_effect=lambda s: int(s.value)),
        patch("pypto.runtime.task_interface.ContinuousTensor.make", side_effect=_make),
        patch("pypto.runtime.task_interface.torch_dtype_to_datatype", side_effect=_torch_dtype_to_datatype),
    ):
        yield captured, tmp_path


class TestExecuteCompiledDeviceTensor:
    def test_device_tensor_produces_child_memory_true(self, patched_runtime):
        captured, tmp = patched_runtime
        dt = DeviceTensor(0xABCD, (8, 16), torch.float16)

        from pypto.runtime.runner import execute_compiled  # noqa: PLC0415

        execute_compiled(tmp, [dt], platform="a2a3sim", device_id=0)

        assert len(captured["make_calls"]) == 1
        call = captured["make_calls"][0]
        assert call["data"] == 0xABCD
        assert call["shapes"] == (8, 16)
        assert call["child_memory"] is True
        assert call["dtype"] == "<dtype:torch.float16>"
        # Host path was not used.
        assert captured["make_tensor_arg_calls"] == []

    def test_torch_tensor_uses_make_tensor_arg_path(self, patched_runtime):
        captured, tmp = patched_runtime
        host = torch.zeros(4, 4, dtype=torch.float32)

        from pypto.runtime.runner import execute_compiled  # noqa: PLC0415

        execute_compiled(tmp, [host], platform="a2a3sim", device_id=0)

        # Host tensor goes through make_tensor_arg, NOT through ContinuousTensor.make
        # (so child_memory cannot be True for it).
        assert captured["make_tensor_arg_calls"] == [host]
        assert captured["make_calls"] == []

    def test_mixed_args_preserve_order(self, patched_runtime):
        captured, tmp = patched_runtime
        host = torch.zeros(4, 4, dtype=torch.float32)
        dt = DeviceTensor(0x9000, (4, 4), torch.float32)

        from pypto.runtime.runner import execute_compiled  # noqa: PLC0415

        execute_compiled(tmp, [host, dt, host], platform="a2a3sim", device_id=0)

        # add_tensor was called three times in order: host, device, host.
        assert len(captured["tensors"]) == 3
        # Only the device-tensor slot uses ContinuousTensor.make with child_memory=True.
        assert len(captured["make_calls"]) == 1
        assert captured["make_calls"][0]["child_memory"] is True
        assert len(captured["make_tensor_arg_calls"]) == 2

    def test_unsupported_arg_raises(self, patched_runtime):
        _, tmp = patched_runtime
        from pypto.runtime.runner import execute_compiled  # noqa: PLC0415

        with pytest.raises(TypeError, match="DeviceTensor"):
            execute_compiled(tmp, ["not a tensor"], platform="a2a3sim", device_id=0)  # type: ignore[list-item]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
