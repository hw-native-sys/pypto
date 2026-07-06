# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: a leading-dim-stacked ``[B, N, M]`` weight uploaded once and
made resident per card via :meth:`DistributedWorker.alloc_stacked_tensor`.

A HOST orchestrator slices a ``[B, N, M]`` weight along its leading dimension and
dispatches a per-rank ``f = a + b`` child, where ``a`` / ``f`` are per-call host
IO and ``b`` is the stacked resident weight. ``b`` is uploaded **once** (shard
``i`` to card ``worker_ids[i]``) and reused across multiple dispatches, exactly
like the single-card resident weight in ``test_l3_device_tensor.py`` but spread
across cards.

Residency is proven the same way as the single-card test: after uploading the
stack, the host source buffer is zeroed, so a kernel re-reading host memory would
compute ``a + 0``. Getting ``a + b`` on the second dispatch means the per-card
buffers stayed resident.

Two placements are covered:

* :meth:`test_identity` — canonical ``for r in range(world_size): child(x[r],
  device=r)``; shard ``i`` resides on card ``i`` (default ``worker_ids``).
* :meth:`test_permuted` — a non-identity placement: the orchestrator dispatches
  ``x[0]`` to card 1 and ``x[1]`` to card 0 (literal ConstInt ``device=``), so the
  stack must be uploaded with ``worker_ids=[1, 0]`` to keep each shard on the card
  its consumer runs on.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.runtime import StackedDeviceTensor

N_RANKS = 2
DIM = 128


def _build_identity_program():
    """Canonical rank-sliced dispatch: ``for r: child(x[r], device=r)``."""

    @pl.program
    class StackedIdentity:
        @pl.function(type=pl.FunctionType.InCore)
        def tile_add(
            self,
            a: pl.Tensor[[DIM, DIM], pl.FP32],
            b: pl.Tensor[[DIM, DIM], pl.FP32],
            f: pl.Out[pl.Tensor[[DIM, DIM], pl.FP32]],
        ) -> pl.Tensor[[DIM, DIM], pl.FP32]:
            tile_a = pl.load(a, [0, 0], [DIM, DIM])
            tile_b = pl.load(b, [0, 0], [DIM, DIM])
            tile_f = pl.add(tile_a, tile_b)
            return pl.store(tile_f, [0, 0], f)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            a: pl.Tensor[[DIM, DIM], pl.FP32],
            b: pl.Tensor[[DIM, DIM], pl.FP32],
            f: pl.Out[pl.Tensor[[DIM, DIM], pl.FP32]],
        ) -> pl.Tensor[[DIM, DIM], pl.FP32]:
            return self.tile_add(a, b, f)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            a: pl.Tensor[[N_RANKS, DIM, DIM], pl.FP32],
            b: pl.Tensor[[N_RANKS, DIM, DIM], pl.FP32],
            f: pl.Out[pl.Tensor[[N_RANKS, DIM, DIM], pl.FP32]],
        ):
            for r in pl.range(pld.world_size()):
                self.chip_orch(a[r], b[r], f[r], device=r)

    return StackedIdentity


def _build_permuted_program():
    """Non-identity placement: shard 0 -> card 1, shard 1 -> card 0 (literal device=)."""

    @pl.program
    class StackedPermuted:
        @pl.function(type=pl.FunctionType.InCore)
        def tile_add(
            self,
            a: pl.Tensor[[DIM, DIM], pl.FP32],
            b: pl.Tensor[[DIM, DIM], pl.FP32],
            f: pl.Out[pl.Tensor[[DIM, DIM], pl.FP32]],
        ) -> pl.Tensor[[DIM, DIM], pl.FP32]:
            tile_a = pl.load(a, [0, 0], [DIM, DIM])
            tile_b = pl.load(b, [0, 0], [DIM, DIM])
            tile_f = pl.add(tile_a, tile_b)
            return pl.store(tile_f, [0, 0], f)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            a: pl.Tensor[[DIM, DIM], pl.FP32],
            b: pl.Tensor[[DIM, DIM], pl.FP32],
            f: pl.Out[pl.Tensor[[DIM, DIM], pl.FP32]],
        ) -> pl.Tensor[[DIM, DIM], pl.FP32]:
            return self.tile_add(a, b, f)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            a: pl.Tensor[[N_RANKS, DIM, DIM], pl.FP32],
            b: pl.Tensor[[N_RANKS, DIM, DIM], pl.FP32],
            f: pl.Out[pl.Tensor[[N_RANKS, DIM, DIM], pl.FP32]],
        ):
            # x[0] consumed on card 1, x[1] on card 0 -> worker_ids=[1, 0].
            self.chip_orch(a[0], b[0], f[0], device=1)
            self.chip_orch(a[1], b[1], f[1], device=0)

    return StackedPermuted


def _run_stacked(test_config, device_ids, program, worker_ids):
    """Upload a [N_RANKS, DIM, DIM] weight once via alloc_stacked_tensor, dispatch
    twice with different per-call inputs, and assert ``f == a + b`` each time."""
    compiled = ir.compile(
        program,
        platform=test_config.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids[:N_RANKS],
            num_sub_workers=0,
        ),
    )

    # Shared-memory IO + weight source, allocated BEFORE prepare() so the forked
    # chip workers inherit the mappings.
    host_a = torch.zeros((N_RANKS, DIM, DIM), dtype=torch.float32).share_memory_()
    host_f = torch.zeros((N_RANKS, DIM, DIM), dtype=torch.float32).share_memory_()
    # Distinct per-shard weight so a wrong shard->card placement is caught.
    host_b = torch.empty((N_RANKS, DIM, DIM), dtype=torch.float32).share_memory_()
    for i in range(N_RANKS):
        host_b[i].fill_(10.0 + i)  # shard i holds 10 + i
    # D2H read-back destination — like every host buffer the forked worker writes,
    # it must be shared memory allocated BEFORE prepare() (see copy_stacked_from).
    host_readback = torch.zeros((N_RANKS, DIM, DIM), dtype=torch.float32).share_memory_()

    with compiled.prepare() as rt:
        weight = rt.alloc_stacked_tensor(host_b, worker_ids=worker_ids)
        assert isinstance(weight, StackedDeviceTensor)  # fail fast if the API contract changes
        # Prove residency: scribble over the upload source. A kernel that re-read
        # host memory would now compute a + 0 instead of a + b.
        host_b.zero_()
        try:
            for host_a_val in (2.0, 7.0):
                host_a.fill_(host_a_val)
                host_f.zero_()
                rt(host_a, weight, host_f)

                expected = torch.empty((N_RANKS, DIM, DIM), dtype=torch.float32)
                for i in range(N_RANKS):
                    expected[i].fill_(host_a_val + 10.0 + i)  # a + (10 + i)
                torch.testing.assert_close(host_f, expected, rtol=1e-5, atol=1e-5)

            # Read the resident stack back to the host (D2H). The kernel only
            # reads ``b``, so each shard still holds ``10 + i`` even though the
            # host source was zeroed after upload — a clean copy_stacked_from proof.
            host_readback.zero_()
            rt.copy_stacked_from(weight, host_readback)
            for i in range(N_RANKS):
                torch.testing.assert_close(
                    host_readback[i], torch.full((DIM, DIM), 10.0 + i), rtol=1e-5, atol=1e-5
                )
        finally:
            rt.free_stacked_tensor(weight)


class TestL3StackedDeviceTensor:
    """L3 distributed runtime: per-card resident stacked weight, reused across dispatches."""

    def test_identity(self, test_config, device_ids):
        """Default ``worker_ids=range(B)`` with a ``device=r`` orchestrator."""
        if len(device_ids) < N_RANKS:
            pytest.skip(f"stacked-device-tensor identity needs {N_RANKS} devices, got {device_ids}")
        _run_stacked(test_config, device_ids, _build_identity_program(), worker_ids=None)

    def test_permuted(self, test_config, device_ids):
        """Non-identity ``worker_ids=[1, 0]`` matching a permuted ``device=`` dispatch."""
        if len(device_ids) < N_RANKS:
            pytest.skip(f"stacked-device-tensor permuted needs {N_RANKS} devices, got {device_ids}")
        _run_stacked(test_config, device_ids, _build_permuted_program(), worker_ids=[1, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
