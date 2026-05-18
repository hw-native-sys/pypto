# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Hand-written L3 over a PyPTO L2-only program.

This ST covers the pattern where PyPTO is used only as an L2 (CHIP) compiler
and the user describes the entire L3 layer directly against the simpler
runtime API — no HOST-level DSL, no generated ``host_orch.py`` /
``sub_workers/*.py``.

The test exercises the public boundary that PyPTO's own
``DistributedCompiledProgram`` relies on:

  * :func:`compile_and_assemble` to obtain a ``ChipCallable`` from an L2 build
  * ``simpler.worker.Worker(level=3, ...)`` with manual ``register``/``init``/``run``
  * ``TaskArgs`` + ``make_tensor_arg`` + ``TensorArgType`` for argument marshalling
  * A Python SubWorker that **closes over host-side state** (``done_flag``) — a
    capability the DSL form (``@pl.function(level=HOST, role=SubWorker)``)
    cannot express because its body is captured as ``InlineStmt`` and only
    sees module globals.

Computation: ``f = a + b``; a closure-captured SubWorker validates the result
and flips a host-side flag the test asserts on.
"""

import sys

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.runtime.device_runner import compile_and_assemble
from pypto.runtime.distributed_runner import _tensor_from_continuous


@pl.program
class L2OnlyAddProgram:
    """L2 only: ``tile_add`` + ``chip_orch``. No HOST-level functions.

    ``ir.compile()`` returns a :class:`CompiledProgram` (not a
    :class:`DistributedCompiledProgram`) for this shape, and the chip
    artefacts land directly under ``output_dir/`` rather than under
    ``output_dir/next_levels/<name>/``.
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
        return self.tile_add(a, b, f)


class TestL3Manual:
    """Drive an L2 PyPTO build from a hand-written L3 using simpler directly."""

    def test_manual_l3(self, test_config, device_ids, tmp_path):
        # Conftest's session fixture inserts ``simpler`` into ``sys.path``;
        # importing inside the test guarantees the path is in place. Skipped
        # automatically under --codegen-only since simpler isn't required there.
        if not device_ids:
            pytest.skip("manual L3 test needs at least one device")

        from simpler.task_interface import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            TaskArgs,
            TensorArgType,
        )
        from simpler.worker import Worker  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        from simpler_setup.torch_interop import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            make_tensor_arg,
        )

        # 1) Compile L2 only. The result is a CompiledProgram; ``output_dir``
        # itself is the chip's work dir (contains kernel_config.py, kernels/,
        # orchestration/).
        out_dir = tmp_path / "l2_build"
        ir.compile(
            L2OnlyAddProgram,
            output_dir=str(out_dir),
            platform=test_config.platform,
        )

        # 2) Assemble the ChipCallable ourselves — the same call
        # ``execute_distributed`` makes per next_levels/<name>/, but pointed at
        # the L2 root because no HOST-level outlining happened.
        chip_callable, runtime_name, _ = compile_and_assemble(out_dir, platform=test_config.platform)

        # 3) Host-side tensors. ``share_memory_()`` must happen before
        # ``Worker.init()`` so the chip/sub-worker child processes inherit the
        # same backing storage via copy-on-write.
        a = torch.full((128, 128), 2.0, dtype=torch.float32).share_memory_()
        b = torch.full((128, 128), 3.0, dtype=torch.float32).share_memory_()
        f = torch.zeros((128, 128), dtype=torch.float32).share_memory_()
        # done_flag demonstrates the closure-capture capability: a DSL
        # SubWorker body (InlineStmt) cannot reach a tensor allocated at
        # runtime in this scope, but a hand-written Python SubWorker can.
        done_flag = torch.zeros((1,), dtype=torch.int32).share_memory_()

        expected = torch.full((128, 128), 5.0, dtype=torch.float32)

        def verify(args) -> None:
            f_view = _tensor_from_continuous(args.tensor(0))
            if not torch.allclose(f_view, expected, rtol=1e-5, atol=1e-5):
                raise AssertionError(
                    f"manual SubWorker verify failed: max diff = {(f_view - expected).abs().max().item()}"
                )
            done_flag.fill_(1)

        # 4) Build the Worker. Both the chip callable and the Python SubWorker
        # are registered before ``init()``; the level-3 fork copies the
        # registry to its children.
        w = Worker(
            level=3,
            device_ids=device_ids[:1],
            num_sub_workers=1,
            platform=test_config.platform,
            runtime=runtime_name,
            chip_bootstrap_configs=None,
        )
        chip_cid = w.register(chip_callable)
        verify_cid = w.register(verify)
        w.init()

        # 5) Hand-written L3 orchestrator. ``submit_next_level`` queues chip
        # work; ``submit_sub`` queues the Python SubWorker. Both calls are
        # non-blocking; the implicit scope around ``orch_fn`` waits at return.
        def orch_fn(orch, _unused_args, _unused_cfg) -> None:
            del _unused_args, _unused_cfg  # required by simpler's orch_fn signature
            chip_ta = TaskArgs()
            chip_ta.add_tensor(make_tensor_arg(a), TensorArgType.INPUT)
            chip_ta.add_tensor(make_tensor_arg(b), TensorArgType.INPUT)
            chip_ta.add_tensor(make_tensor_arg(f), TensorArgType.OUTPUT_EXISTING)
            orch.submit_next_level(chip_cid, chip_ta)

            verify_ta = TaskArgs()
            verify_ta.add_tensor(make_tensor_arg(f), TensorArgType.INPUT)
            orch.submit_sub(verify_cid, verify_ta)

        try:
            w.run(orch_fn)
        finally:
            w.close()

        # 6) Check device write and SubWorker side-effect.
        torch.testing.assert_close(f, expected, rtol=1e-5, atol=1e-5)
        assert done_flag.item() == 1, (
            "verify SubWorker did not run — closure-captured done_flag was not flipped"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
