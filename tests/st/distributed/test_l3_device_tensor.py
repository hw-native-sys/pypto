# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 device-resident input via DeviceTensor + pypto ``make_tensor_arg``.

Validates the G1 conversion path on hardware: a worker-resident buffer wrapped
as a :class:`~pypto.runtime.DeviceTensor` is marshalled by the pypto-owned
``make_tensor_arg`` (``pypto.runtime.tensor_arg``) into a
``ContinuousTensor(child_memory=True)``, so the runtime consumes it **without an
H2D upload** — the same conversion ``DistributedCompiledProgram.__call__`` relies
on when a caller passes a ``DeviceTensor``.

Why the manual L3 driving (cf. ``test_l3_manual.py``) instead of the
``compiled(host_a, weight_dev, host_out)`` form used by
``tests/st/runtime/test_device_tensor.py``:

  * ``pypto.runtime.Worker`` only supports ``level=2``; the L2 reuse trick
    (open a Worker, ``alloc_tensor``, let ``CompiledProgram`` reuse it) has no
    L3 analogue yet.
  * ``execute_distributed`` builds its own one-shot ``Worker(level=3)`` per
    call, so a DeviceTensor allocated by a separate Worker would not be valid in
    the executing worker's address space.

Driving the level-3 Worker directly lets us allocate the resident buffer on the
**same** chip worker that runs the kernel (``orch.malloc`` / ``orch.copy_to``),
then exercise the real ``make_tensor_arg`` DeviceTensor branch end-to-end.

Computation: ``f = a + b``, with ``b`` uploaded once as a resident DeviceTensor.
"""

import sys

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.runtime import DeviceTensor
from pypto.runtime.device_runner import compile_and_assemble
from pypto.runtime.tensor_arg import make_tensor_arg


@pl.program
class L2OnlyAddProgram:
    """L2 only: ``tile_add`` + ``chip_orch`` (``f = a + b``).

    ``ir.compile()`` returns a :class:`CompiledProgram` for this shape and the
    chip artefacts land directly under ``output_dir/`` (no HOST-level
    outlining), so ``compile_and_assemble`` can be pointed at the root.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_f = pl.add(tile_a, tile_b)
        return pl.store(tile_f, [0, 0], f)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_f = self.tile_add(a, b, f)
        return out_f


class TestL3DeviceTensor:
    """Resident DeviceTensor input flows through pypto make_tensor_arg on device."""

    def test_resident_device_tensor_skips_h2d(self, test_config, device_ids, tmp_path):
        """``f = a + b`` with ``b`` uploaded once as a resident DeviceTensor.

        Asserts ``f == a + b`` (not ``a + 0``), proving the resident buffer —
        marshalled via ``make_tensor_arg(DeviceTensor)`` →
        ``ContinuousTensor(child_memory=True)`` — was the actual chip input and
        the runtime did not clobber it with an H2D upload.
        """
        if not device_ids:
            pytest.skip("L3 DeviceTensor test needs at least one device")

        from simpler.task_interface import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            CallConfig,
            TaskArgs,
            TensorArgType,
        )
        from simpler.worker import Worker  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        # 1) Compile L2 only; output_dir is the chip work dir.
        out_dir = tmp_path / "l2_build"
        ir.compile(L2OnlyAddProgram, output_dir=str(out_dir), platform=test_config.platform)

        # 2) Assemble the ChipCallable (same call execute_distributed makes).
        chip_callable, runtime_name, _ = compile_and_assemble(out_dir, platform=test_config.platform)

        # 3) Host tensors. share_memory_() before init() so the chip worker
        #    child process inherits the backing storage. ``b`` stays host-side
        #    here — it is uploaded to the device inside orch_fn below.
        a = torch.full((128, 128), 2.0, dtype=torch.float32).share_memory_()
        b = torch.full((128, 128), 3.0, dtype=torch.float32).share_memory_()
        f = torch.zeros((128, 128), dtype=torch.float32).share_memory_()
        expected = torch.full((128, 128), 5.0, dtype=torch.float32)

        # 4) Level-3 Worker; register the chip callable before init().
        w = Worker(
            level=3,
            device_ids=device_ids[:1],
            num_sub_workers=0,
            platform=test_config.platform,
            runtime=runtime_name,
            chip_bootstrap_configs=None,
        )
        chip_cid = w.register(chip_callable)
        w.init()

        call_config = CallConfig()
        call_config.block_dim = 3
        call_config.aicpu_thread_num = 4

        b_nbytes = b.numel() * b.element_size()

        def orch_fn(orch, _unused_args, _unused_cfg) -> None:
            del _unused_args, _unused_cfg  # required by simpler's orch_fn signature

            # Upload ``b`` once to the chip worker (worker_id=0) and wrap the
            # device pointer as a caller-managed DeviceTensor.
            dev_ptr = orch.malloc(worker_id=0, size=b_nbytes)
            orch.copy_to(worker_id=0, dst=dev_ptr, src=b.data_ptr(), size=b_nbytes)
            b_dev = DeviceTensor(dev_ptr, (128, 128), torch.float32)

            chip_ta = TaskArgs()
            chip_ta.add_tensor(make_tensor_arg(a), TensorArgType.INPUT)       # host
            chip_ta.add_tensor(make_tensor_arg(b_dev), TensorArgType.INPUT)   # resident → child_memory=True
            chip_ta.add_tensor(make_tensor_arg(f), TensorArgType.OUTPUT_EXISTING)
            orch.submit_next_level(chip_cid, chip_ta, call_config)

        try:
            w.run(orch_fn)
        finally:
            w.close()

        torch.testing.assert_close(f, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
