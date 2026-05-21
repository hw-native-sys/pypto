# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify the pypto-owned ``make_tensor_arg`` used by generated distributed
orchestration code.

It must:
- wrap a worker-resident :class:`DeviceTensor` as
  ``ContinuousTensor.make(..., child_memory=True)``;
- pass an already-built ``ContinuousTensor`` through unchanged;
- delegate a host ``torch.Tensor`` to simpler's ``make_tensor_arg``.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto.runtime import DeviceTensor

# ``task_interface`` eagerly imports the optional ``simpler`` runtime package;
# skip the module when simpler is unavailable (same pattern as
# test_execute_compiled_device_tensor.py).
try:
    import simpler  # noqa: F401  # pyright: ignore[reportMissingImports]
except ImportError:
    _has_simpler = False
else:
    _has_simpler = True

pytestmark = pytest.mark.skipif(not _has_simpler, reason="make_tensor_arg requires the simpler package")


def test_device_tensor_produces_child_memory_true():
    captured: dict = {"make_calls": []}

    def _make(*, data, shapes, dtype, child_memory=False):
        captured["make_calls"].append(
            {"data": data, "shapes": tuple(shapes), "dtype": dtype, "child_memory": child_memory}
        )
        return MagicMock(name=f"ContinuousTensor(0x{data:x})")

    dt = DeviceTensor(0xABCD, (8, 16), torch.float16)

    with (
        patch("pypto.runtime.task_interface.ContinuousTensor.make", side_effect=_make),
        patch(
            "pypto.runtime.task_interface.torch_dtype_to_datatype",
            side_effect=lambda d: f"<dtype:{d}>",
        ),
    ):
        from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

        make_tensor_arg(dt)

    assert len(captured["make_calls"]) == 1
    call = captured["make_calls"][0]
    assert call["data"] == 0xABCD
    assert call["shapes"] == (8, 16)
    assert call["child_memory"] is True
    assert call["dtype"] == "<dtype:torch.float16>"


def test_continuous_tensor_passes_through():
    from pypto.runtime.task_interface import ContinuousTensor, torch_dtype_to_datatype  # noqa: PLC0415
    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    ct = ContinuousTensor.make(0x1000, (4,), torch_dtype_to_datatype(torch.float32), child_memory=True)
    assert make_tensor_arg(ct) is ct


def test_host_tensor_delegates_to_simpler():
    host = torch.zeros(4, 4, dtype=torch.float32)
    sentinel = MagicMock(name="ContinuousTensor(host)")

    with patch("pypto.runtime.task_interface.make_tensor_arg", return_value=sentinel) as impl:
        from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

        result = make_tensor_arg(host)

    impl.assert_called_once_with(host)
    assert result is sentinel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
