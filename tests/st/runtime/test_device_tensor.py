# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end tests for the worker-resident DeviceTensor flow.

Validates that ``Worker.alloc_tensor`` produces a buffer the runtime can
consume via ``CompiledProgram(...)`` with ``ContinuousTensor.child_memory=True``
— i.e. no H2D upload of the DeviceTensor on entry, no D2H copy-back on exit.

Both tests run on hardware/simulator and depend on the ``simpler`` runtime
package; the ``check_hardware_availability`` fixture in this directory's
``conftest.py`` skips them on hosts without a device when only an
onboard platform is requested.
"""

from datetime import datetime
from pathlib import Path

import pytest
import torch
from examples.kernels.elementwise import TileAddProgram
from pypto import ir
from pypto.runtime import RunConfig, Worker

_BUILD_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "build_output" / "test_device_tensor"


@pytest.fixture(scope="module")
def output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = _BUILD_OUTPUT_DIR / timestamp
    root.mkdir(parents=True, exist_ok=True)
    return root


def _worker_config(test_config: RunConfig) -> RunConfig:
    """Materialize a RunConfig that the active Worker uses for binding match.

    ``Worker.current()`` matches on (level, platform, device_id, runtime).
    The Worker we open here uses ``test_config.platform`` and
    ``test_config.device_id``; the compiled program is compiled with the
    same platform so ``CompiledProgram.__call__`` can reuse this Worker
    rather than spinning up a one-shot one.
    """
    return RunConfig(platform=test_config.platform, device_id=test_config.device_id)


class TestDeviceTensorEndToEnd:
    """End-to-end DeviceTensor execution on hardware/simulator."""

    def test_device_tensor_input_skips_h2d_per_call(self, output_root, test_config):
        """``compiled(host_a, weight_dev, host_out)`` produces ``a + b``.

        ``b`` is uploaded once to a worker-resident DeviceTensor; subsequent
        calls reuse the same device buffer.  This verifies that:

        1. ``Worker.alloc_tensor(..., init=host_b)`` actually populates the
           device buffer (otherwise the kernel would compute ``a + 0``).
        2. The runtime treats DeviceTensor as ``child_memory=True`` and does
           not overwrite the device buffer with stale host bytes on entry.
        3. The handle survives across multiple kernel invocations bound to
           the same Worker.
        """
        compiled = ir.compile(
            TileAddProgram,
            output_dir=str(output_root / "add_devtensor_input"),
            platform=test_config.platform,
        )

        host_a1 = torch.full((128, 128), 2.0, dtype=torch.float32)
        host_a2 = torch.full((128, 128), 7.0, dtype=torch.float32)
        host_b = torch.full((128, 128), 3.0, dtype=torch.float32)

        out1 = torch.zeros((128, 128), dtype=torch.float32)
        out2 = torch.zeros((128, 128), dtype=torch.float32)

        with Worker(config=_worker_config(test_config)) as w:
            weight = w.alloc_tensor((128, 128), torch.float32, init=host_b)
            try:
                compiled(host_a1, weight, out1, config=test_config)
                compiled(host_a2, weight, out2, config=test_config)
            finally:
                w.free_tensor(weight)

        expected1 = torch.full((128, 128), 5.0, dtype=torch.float32)
        expected2 = torch.full((128, 128), 10.0, dtype=torch.float32)
        assert torch.allclose(out1, expected1, rtol=1e-5, atol=1e-5), (
            f"first call: max diff = {(out1 - expected1).abs().max().item()}"
        )
        assert torch.allclose(out2, expected2, rtol=1e-5, atol=1e-5), (
            f"second call (reuse): max diff = {(out2 - expected2).abs().max().item()}"
        )

    def test_alloc_tensor_then_copy_from_roundtrip(self, test_config):
        """``alloc_tensor(init=...)`` → ``copy_from`` recovers the original bytes.

        Exercises the Worker primitives in isolation: this does NOT involve
        a CompiledProgram — it just verifies that the H2D upload performed
        by ``alloc_tensor`` lands the exact host bytes on device, and that
        ``copy_from`` reads them back correctly.  A failure here would
        manifest as garbage data in the DeviceTensor consumed by kernels.
        """
        host_in = torch.arange(256, dtype=torch.float32).view(16, 16)
        host_out = torch.zeros_like(host_in)

        with Worker(config=_worker_config(test_config)) as w:
            t = w.alloc_tensor((16, 16), torch.float32, init=host_in)
            try:
                w.copy_from(host_out.data_ptr(), t.data_ptr, t.nbytes)
            finally:
                w.free_tensor(t)

        assert torch.equal(host_out, host_in), (
            f"copy_from did not recover input bytes; max diff = {(host_out - host_in).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
