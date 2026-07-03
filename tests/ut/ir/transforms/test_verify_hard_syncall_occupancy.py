# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile-time occupancy check for hard-form ``pl.system.syncall`` (issue #1935).

A hard (FFTS) ``syncall`` waits for every physical core of its ``core_type`` to
reach the barrier, so the enclosing ``pl.spmd`` must fill all of them exactly; a
partial launch deadlocks on device (507018). The ``HardSyncallOccupancy``
verifier (produced by ``ExpandMixedKernel``, in ``GetVerifiedProperties()``)
rejects such launches at compile time.

Tests drive the full Default pipeline on Ascend910B (48 VECTOR / 24 CUBE cores).
"""

import pypto.language as pl
import pytest
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

from pypto import backend

TR = TC = 128


def _run(program_cls) -> None:
    """Compile a program through the Default pipeline on Ascend910B."""
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910B)
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    pm.run_passes(program_cls)


def _aiv_program(n: int):
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def add(
            self,
            a: pl.Tensor[[n * TR, TC], pl.FP32],
            b: pl.Tensor[[n * TR, TC], pl.FP32],
            out: pl.Out[pl.Tensor[[n * TR, TC], pl.FP32]],
        ) -> pl.Tensor[[n * TR, TC], pl.FP32]:
            i = pl.tile.get_block_idx()
            o = i * TR
            ta = pl.load(a, [o, 0], [TR, TC])
            tb = pl.load(b, [o, 0], [TR, TC])
            pl.system.syncall(core_type="aiv_only")  # HARD barrier
            out = pl.store(pl.add(ta, tb), [o, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            a: pl.Tensor[[n * TR, TC], pl.FP32],
            b: pl.Tensor[[n * TR, TC], pl.FP32],
            out: pl.Out[pl.Tensor[[n * TR, TC], pl.FP32]],
        ) -> pl.Tensor[[n * TR, TC], pl.FP32]:
            with pl.spmd(n):
                out = self.add(a, b, out)
            return out

    return Prog


def _soft_program(n: int):
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def add(
            self,
            a: pl.Tensor[[n * TR, TC], pl.FP32],
            b: pl.Tensor[[n * TR, TC], pl.FP32],
            ws: pl.Tensor[[n * 8], pl.INT32],
            out: pl.Out[pl.Tensor[[n * TR, TC], pl.FP32]],
        ) -> pl.Tensor[[n * TR, TC], pl.FP32]:
            i = pl.tile.get_block_idx()
            o = i * TR
            ta = pl.load(a, [o, 0], [TR, TC])
            tb = pl.load(b, [o, 0], [TR, TC])
            pl.system.syncall(mode="soft", core_type="aiv_only", gm_workspace=ws, used_cores=n)
            out = pl.store(pl.add(ta, tb), [o, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            a: pl.Tensor[[n * TR, TC], pl.FP32],
            b: pl.Tensor[[n * TR, TC], pl.FP32],
            ws: pl.Tensor[[n * 8], pl.INT32],
            out: pl.Out[pl.Tensor[[n * TR, TC], pl.FP32]],
        ) -> pl.Tensor[[n * TR, TC], pl.FP32]:
            with pl.spmd(n):
                out = self.add(a, b, ws, out)
            return out

    return Prog


def _mixed_program(n: int):
    M = K = NN = 64

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def mixed(
            self,
            a: pl.Tensor[[M, K], pl.FP16],
            b: pl.Tensor[[K, NN], pl.FP16],
            bias: pl.Tensor[[M, NN], pl.FP32],
            out: pl.Out[pl.Tensor[[M, NN], pl.FP32]],
        ) -> pl.Tensor[[M, NN], pl.FP32]:
            ta = pl.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
            tb = pl.load(b, [0, 0], [K, NN], target_memory=pl.Mem.Mat)
            tal = pl.move(ta, target_memory=pl.Mem.Left)
            tbl = pl.move(tb, target_memory=pl.Mem.Right)
            tc = pl.matmul(tal, tbl)
            tcv = pl.move(tc, target_memory=pl.Mem.Vec)
            tbias = pl.load(bias, [0, 0], [M, NN])
            tsum = pl.add(tcv, tbias)
            pl.system.syncall(core_type="mix")  # HARD mix barrier
            out = pl.store(tsum, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            a: pl.Tensor[[M, K], pl.FP16],
            b: pl.Tensor[[K, NN], pl.FP16],
            bias: pl.Tensor[[M, NN], pl.FP32],
            out: pl.Out[pl.Tensor[[M, NN], pl.FP32]],
        ) -> pl.Tensor[[M, NN], pl.FP32]:
            with pl.spmd(n):
                out = self.mixed(a, b, bias, out)
            return out

    return Prog


def _bare_kernel_program():
    """A hard-syncall InCore kernel with no pl.spmd launch (mirrors the codegen UT)."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_syncall(
            self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Tensor[[16, 16], pl.FP32]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            tile = pl.load(x, [0, 0], [16, 16])
            pl.system.syncall(core_type="aiv_only")
            updated = pl.store(tile, [0, 0], out)
            return updated

    return Prog


class TestHardSyncallOccupancy:
    """Compile-time occupancy check for the hard (FFTS) syncall (issue #1935)."""

    def test_partial_aiv_occupancy_rejected(self):
        """pl.spmd(24) < 48 AIV cores + hard aiv_only barrier is rejected at compile time."""
        with pytest.raises(Exception, match="fill all 48 AIV cores"):
            _run(_aiv_program(24))

    def test_full_aiv_occupancy_accepted(self):
        """pl.spmd(48) == 48 AIV cores compiles cleanly (occupancy is exactly full)."""
        _run(_aiv_program(48))

    def test_over_aiv_occupancy_rejected(self):
        """pl.spmd(96) > 48 AIV cores is rejected (hard barrier needs exactly-full occupancy)."""
        with pytest.raises(Exception, match="fill all 48 AIV cores"):
            _run(_aiv_program(96))

    def test_soft_form_not_checked(self):
        """The soft (GM-polling) form works at partial occupancy and is not rejected."""
        _run(_soft_program(4))

    def test_bare_kernel_without_spmd_not_checked(self):
        """A hard-syncall kernel with no pl.spmd launch is not an occupancy target."""
        _run(_bare_kernel_program())

    def test_mixed_full_occupancy_accepted(self):
        """Mixed kernel + hard mix barrier at pl.spmd(24) == 24 core-groups compiles cleanly."""
        _run(_mixed_program(24))

    def test_mixed_partial_occupancy_rejected(self):
        """Mixed kernel + hard mix barrier at pl.spmd(12) < 24 core-groups is rejected."""
        with pytest.raises(Exception, match="core-groups"):
            _run(_mixed_program(12))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
