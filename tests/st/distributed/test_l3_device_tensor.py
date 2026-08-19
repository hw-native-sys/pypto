# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end L3 reuse via ``DistributedCompiledProgram.prepare()``.

The L3 analogue of ``tests/st/runtime/framework_and_models/test_device_tensor.py``: prepare a
reusable handle once, upload a static weight to a worker-resident DeviceTensor
via ``rt.alloc_tensor`` (once), then dispatch multiple times reusing both the
handle (no re-setup) and the resident weight. Three dispatches use distinct
mutable IO buffers: the first two occupy the bounded asynchronous metadata
frames, and the third waits for the oldest frame before reuse.

Per-call IO uses shared-memory host tensors allocated **before** ``prepare()``
and reused in place — the forked chip worker reads/writes them through the
inherited shared mapping, so the output is read straight back from ``host_out``
(no ``copy_from``). Only the weight is device-resident; its retained simpler
``Buffer`` supplies the address-free wire Tensor for every dispatch while the
test exercises the public asynchronous dispatch contract.

Computation: ``f = a + b``, with ``b`` resident across both dispatches.
"""

import sys

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig


@pl.program
class L3AddProgram:
    """L3: HOST orch → CHIP worker (``f = a + b``)."""

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

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_f: pl.Tensor[[128, 128], pl.FP32] = self.chip_orch(a, b, f)
        return out_f


class TestL3DeviceTensorReuse:
    """Prepare once, retain one weight, and submit through two bounded frames."""

    def test_resident_weight_reused_across_dispatches(self, test_config, device_ids):
        """``f = a + b`` over three bounded submissions; ``b`` is uploaded once.

        Verifies that:
        1. ``rt.alloc_tensor(init=host_b)`` populates a resident buffer
           (otherwise the kernel would compute ``a + 0``).
        2. Two submissions can own distinct mutable IO frames concurrently.
        3. A third submission drains the oldest frame before bounded reuse.
        4. The resident weight's retained Buffer survives every dispatch
           without re-upload.
        5. The handle reuses its Worker — setup is not repeated per call.
        """
        if not device_ids:
            pytest.skip("L3 DeviceTensor test needs at least one device")

        compiled = ir.compile(
            L3AddProgram,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:1],
                num_sub_workers=0,
                aicpu_thread_num=4,
            ),
        )

        # Shared-memory host buffers MUST be allocated before prepare() so the
        # forked chip worker inherits their mappings: host_b is the resident
        # weight's upload source. Every overlapping dispatch owns a distinct
        # mutable input/output pair until its handle publishes completion.
        host_inputs = [
            torch.full((128, 128), value, dtype=torch.float32).share_memory_() for value in (2.0, 7.0, 11.0)
        ]
        host_outputs = [torch.zeros((128, 128), dtype=torch.float32).share_memory_() for _ in host_inputs]
        host_b = torch.full((128, 128), 3.0, dtype=torch.float32).share_memory_()

        with compiled.prepare() as rt:
            weight = rt.alloc_tensor((128, 128), torch.float32, init=host_b)  # uploaded once
            try:
                first = rt.submit(compiled, host_inputs[0], weight, host_outputs[0])
                second = rt.submit(compiled, host_inputs[1], weight, host_outputs[1])
                # Both metadata frames are now owned. This call waits for the
                # oldest handle before it constructs and publishes dispatch 3.
                third = rt.submit(compiled, host_inputs[2], weight, host_outputs[2])

                for handle, host_out, expect_val in zip(
                    (first, second, third),
                    host_outputs,
                    (5.0, 10.0, 14.0),
                    strict=True,
                ):
                    handle.result()
                    expected = torch.full((128, 128), expect_val, dtype=torch.float32)
                    torch.testing.assert_close(host_out, expected, rtol=1e-5, atol=1e-5)
            finally:
                rt.free_tensor(weight)


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
