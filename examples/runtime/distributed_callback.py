# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime-bound SubWorker callbacks in an L3 distributed program.

A HOST-level ``SubWorker`` is a pure-Python callback that runs in the forked
orchestrator process. When its logic cannot be written at compile time — a
sampling closure that needs live model state, a host-side metric collector, a
result inspector — declare its body as ``...`` to mark it an **abstract,
runtime-bound callback point**:

    @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
    def inspect_result(f: pl.Tensor[[128, 128], pl.FP32]):
        ...   # implementation supplied at runtime, NOT here

The implementation is then bound at ``prepare(callbacks={...})``. Forgetting to
bind it raises ``ValueError`` at prepare time (not silently at dispatch), and
codegen emits a guard stub that raises if the callback is ever dispatched
unbound.

This script shows:

  1. Declaring an abstract ``...``-body SubWorker.
  2. Binding a real closure to it via ``compiled.prepare(callbacks={...})``.
  3. The fail-fast error when a required callback is omitted.

The pipeline is HOST orchestrator → CHIP add (``f = a + b``) → ``inspect_result``
callback, mirroring tests/st/distributed/test_l3_distributed.py.

Run:  python examples/runtime/distributed_callback.py
      (needs the simpler L3 runtime; defaults to the a2a3 simulator platform)
"""

import pypto.language as pl
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

PLATFORM = "a2a3sim"  # swap for "a2a3" to target a real device


@pl.program
class InspectProgram:
    """L3: HOST orch → CHIP worker (a + b) → abstract ``inspect_result`` callback."""

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

    @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
    def inspect_result(f: pl.Tensor[[128, 128], pl.FP32]):
        # Abstract body: this SubWorker is a runtime-bound callback. The real
        # implementation is supplied via prepare(callbacks={"inspect_result": ...}).
        ...

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_f: pl.Tensor[[128, 128], pl.FP32] = self.chip_orch(a, b, f)
        self.inspect_result(out_f)
        return out_f


def show_abstract_flag() -> None:
    """The ``...`` body marks the SubWorker as runtime-bound at the IR level."""
    fn = InspectProgram.get_function("inspect_result")
    assert fn is not None
    assert fn.requires_runtime_binding is True
    print("inspect_result.requires_runtime_binding =", fn.requires_runtime_binding)


def run_with_callback() -> None:
    """Bind a real closure to the abstract callback and dispatch once."""
    from pypto.runtime.distributed_runner import _tensor_from_continuous  # noqa: PLC0415

    compiled = ir.compile(
        InspectProgram,
        platform=PLATFORM,
        distributed_config=DistributedConfig(
            device_ids=[0], num_sub_workers=1, block_dim=3, aicpu_thread_num=4
        ),
    )

    # Shared-memory IO, allocated BEFORE prepare() so the forked workers inherit
    # the host mappings. ``observed`` is the callback's side channel back to us.
    host_a = torch.full((128, 128), 2.0, dtype=torch.float32).share_memory_()
    host_b = torch.full((128, 128), 3.0, dtype=torch.float32).share_memory_()
    host_f = torch.zeros((128, 128), dtype=torch.float32).share_memory_()
    observed = torch.zeros(1, dtype=torch.float32).share_memory_()

    def inspect_result(args) -> None:
        # Real implementation: read the computed tensor and stash a summary.
        f = _tensor_from_continuous(args.tensor(0))
        observed[0] = float(f.reshape(-1)[0].item())

    with compiled.prepare(callbacks={"inspect_result": inspect_result}) as rt:
        rt(host_a, host_b, host_f)

    expected = torch.full((128, 128), 5.0, dtype=torch.float32)
    assert torch.allclose(host_f, expected, rtol=1e-5, atol=1e-5)
    assert abs(observed[0].item() - 5.0) < 1e-5, observed[0].item()
    print(f"callback ran — observed f[0] = {observed[0].item():.1f} (expected 5.0)")


def show_missing_binding_error() -> None:
    """Omitting a required callback fails fast at prepare(), not at dispatch."""
    compiled = ir.compile(
        InspectProgram,
        platform=PLATFORM,
        distributed_config=DistributedConfig(
            device_ids=[0], num_sub_workers=1, block_dim=3, aicpu_thread_num=4
        ),
    )
    try:
        compiled.prepare()  # no callbacks → inspect_result is unbound
    except ValueError as exc:
        print("missing binding rejected at prepare():", exc)
    else:
        raise AssertionError("expected prepare() to reject the missing callback")


if __name__ == "__main__":
    show_abstract_flag()
    run_with_callback()
    show_missing_binding_error()
    print("OK")
