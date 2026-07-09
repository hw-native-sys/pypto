# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile-time occupancy + sync_start check for hard-form ``pl.system.syncall`` (issue #1935).

A hard (FFTS) ``syncall`` waits for every physical core of its ``core_type`` to
reach the barrier, so the enclosing ``pl.spmd`` must satisfy two conditions:

1. **Full occupancy** — fill all those cores exactly (one block per core); a
   partial launch leaves unlaunched cores that never reach the barrier.
2. **sync_start=True** — all blocks co-resident at once; without it the runtime
   may dispatch blocks in waves and the barrier deadlocks even at full occupancy.

Either gap deadlocks on device (507018). The ``HardSyncallOccupancy`` verifier
(produced by ``ExpandMixedKernel``, in ``GetVerifiedProperties()``) rejects both
at compile time.

Tests drive the full Default pipeline on Ascend910B (48 VECTOR / 24 CUBE cores).
"""

import pypto.language as pl
import pytest
from pypto import backend
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

TR = TC = 128


def _run(program_cls) -> None:
    """Compile a program through the Default pipeline on Ascend910B."""
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910B)
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    pm.run_passes(program_cls)


# A hard barrier now requires BOTH full occupancy and sync_start=True, so the
# primary builders launch with the literal ``sync_start=True`` (occupancy-only
# rejection tests still fail on occupancy first, before the sync_start check).
# The ``_no_sync`` variants below launch without it to exercise the sync_start
# rejection. Each variant lives in its OWN builder function: the DSL parser needs
# a ``sync_start=`` boolean *literal*, and ``inspect.getsource`` (used by
# ``@pl.program``) locates a class by qualname — two same-named classes in one
# function would both resolve to the first one's source.
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
            with pl.spmd(n, sync_start=True):
                out = self.add(a, b, out)
            return out

    return Prog


def _aiv_program_no_sync(n: int):
    """Full-occupancy AIV launch WITHOUT sync_start — exercises the sync_start check."""

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
            with pl.spmd(n):  # no sync_start (DSL default False)
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
            with pl.spmd(n, sync_start=True):
                out = self.mixed(a, b, bias, out)
            return out

    return Prog


def _mixed_program_no_sync(n: int):
    """Full-occupancy mixed-kernel launch WITHOUT sync_start — exercises the sync_start check."""
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
            with pl.spmd(n):  # no sync_start (DSL default False)
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


def _aiv_default_mix_program(n: int):
    """Pure-AIV kernel with the DEFAULT (mix) hard barrier — unsatisfiable in an AIV launch."""

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
            pl.system.syncall()  # default core_type="mix" — no AIC participants in an AIV launch
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


def _spmd_submit_program(n: int):
    """pl.spmd_submit(..., core_num=n) — the block count rides on the Submit node."""

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
            pl.system.syncall(core_type="aiv_only")
            out = pl.store(pl.add(ta, tb), [o, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def orchestrator(
            self,
            a: pl.Tensor[[n * TR, TC], pl.FP32],
            b: pl.Tensor[[n * TR, TC], pl.FP32],
            out: pl.Out[pl.Tensor[[n * TR, TC], pl.FP32]],
        ) -> pl.Tensor[[n * TR, TC], pl.FP32]:
            with pl.manual_scope():
                out, _tid = pl.spmd_submit(self.add, a, b, out, core_num=n, sync_start=True)
            return out

    return Prog


def _spmd_submit_program_no_sync(n: int):
    """Full-occupancy pl.spmd_submit WITHOUT sync_start — exercises the Submit sync_start check."""

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
            pl.system.syncall(core_type="aiv_only")
            out = pl.store(pl.add(ta, tb), [o, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def orchestrator(
            self,
            a: pl.Tensor[[n * TR, TC], pl.FP32],
            b: pl.Tensor[[n * TR, TC], pl.FP32],
            out: pl.Out[pl.Tensor[[n * TR, TC], pl.FP32]],
        ) -> pl.Tensor[[n * TR, TC], pl.FP32]:
            with pl.manual_scope():
                out, _tid = pl.spmd_submit(self.add, a, b, out, core_num=n)  # no sync_start
            return out

    return Prog


def _cluster_spmd_program(n: int):
    """pl.cluster() wrapping pl.spmd(n) — unwrapped to a Group carrying a core_num attr."""

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
            pl.system.syncall(core_type="aiv_only")
            out = pl.store(pl.add(ta, tb), [o, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            a: pl.Tensor[[n * TR, TC], pl.FP32],
            b: pl.Tensor[[n * TR, TC], pl.FP32],
            out: pl.Out[pl.Tensor[[n * TR, TC], pl.FP32]],
        ) -> pl.Tensor[[n * TR, TC], pl.FP32]:
            with pl.cluster():
                with pl.spmd(n, sync_start=True):
                    out = self.add(a, b, out)
            return out

    return Prog


def _cluster_spmd_program_no_sync(n: int):
    """Full-occupancy pl.cluster()-nested pl.spmd, no sync_start — exercises the Group check."""

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
            pl.system.syncall(core_type="aiv_only")
            out = pl.store(pl.add(ta, tb), [o, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            a: pl.Tensor[[n * TR, TC], pl.FP32],
            b: pl.Tensor[[n * TR, TC], pl.FP32],
            out: pl.Out[pl.Tensor[[n * TR, TC], pl.FP32]],
        ) -> pl.Tensor[[n * TR, TC], pl.FP32]:
            with pl.cluster():
                with pl.spmd(n):  # no sync_start (DSL default False)
                    out = self.add(a, b, out)
            return out

    return Prog


class TestHardSyncallOccupancy:
    """Compile-time occupancy + sync_start check for the hard (FFTS) syncall (issue #1935)."""

    def test_partial_aiv_occupancy_rejected(self):
        """pl.spmd(24) < 48 AIV cores + hard aiv_only barrier is rejected at compile time."""
        with pytest.raises(Exception, match="fill all 48 AIV cores"):
            _run(_aiv_program(24))

    def test_full_aiv_occupancy_accepted(self):
        """pl.spmd(48, sync_start=True) at full AIV occupancy compiles cleanly."""
        _run(_aiv_program(48))

    def test_over_aiv_occupancy_rejected(self):
        """pl.spmd(96) > 48 AIV cores is rejected (hard barrier needs exactly-full occupancy)."""
        with pytest.raises(Exception, match="fill all 48 AIV cores"):
            _run(_aiv_program(96))

    def test_full_aiv_occupancy_without_sync_start_rejected(self):
        """pl.spmd(48) at full occupancy but without sync_start is rejected (blocks not co-resident)."""
        with pytest.raises(Exception, match="sync_start=True"):
            _run(_aiv_program_no_sync(48))

    def test_soft_form_not_checked(self):
        """The soft (GM-polling) form works at partial occupancy and is not rejected."""
        _run(_soft_program(4))

    def test_bare_kernel_without_spmd_not_checked(self):
        """A hard-syncall kernel with no pl.spmd launch is not an occupancy target."""
        _run(_bare_kernel_program())

    def test_mixed_full_occupancy_accepted(self):
        """Mixed kernel + hard mix barrier at pl.spmd(24, sync_start=True) compiles cleanly."""
        _run(_mixed_program(24))

    def test_mixed_partial_occupancy_rejected(self):
        """Mixed kernel + hard mix barrier at pl.spmd(12) < 24 core-groups is rejected."""
        with pytest.raises(Exception, match="core-groups"):
            _run(_mixed_program(12))

    def test_mixed_full_occupancy_without_sync_start_rejected(self):
        """Mixed kernel at full 24 core-groups but without sync_start is rejected."""
        with pytest.raises(Exception, match="sync_start=True"):
            _run(_mixed_program_no_sync(24))

    def test_standalone_default_mix_barrier_rejected(self):
        """A pure-AIV kernel with the default (mix) hard barrier can never complete (no AIC)."""
        with pytest.raises(Exception, match="can never complete"):
            _run(_aiv_default_mix_program(48))

    def test_spmd_submit_partial_occupancy_rejected(self):
        """pl.spmd_submit(core_num=24) carries the block count on the Submit — still checked."""
        with pytest.raises(Exception, match="fill all 48 AIV cores"):
            _run(_spmd_submit_program(24))

    def test_spmd_submit_full_occupancy_accepted(self):
        """pl.spmd_submit(core_num=48, sync_start=True) at full AIV occupancy compiles cleanly."""
        _run(_spmd_submit_program(48))

    def test_spmd_submit_full_occupancy_without_sync_start_rejected(self):
        """pl.spmd_submit(core_num=48) at full occupancy but without sync_start is rejected."""
        with pytest.raises(Exception, match="sync_start=True"):
            _run(_spmd_submit_program_no_sync(48))

    def test_cluster_spmd_partial_occupancy_rejected(self):
        """pl.cluster()-nested pl.spmd(24) (a Group with core_num) is checked and rejected."""
        with pytest.raises(Exception, match="fill all 48 AIV cores"):
            _run(_cluster_spmd_program(24))

    def test_cluster_spmd_full_occupancy_accepted(self):
        """pl.cluster()-nested pl.spmd(48, sync_start=True) at full AIV occupancy compiles cleanly."""
        _run(_cluster_spmd_program(48))

    def test_cluster_spmd_full_occupancy_without_sync_start_rejected(self):
        """pl.cluster()-nested pl.spmd(48) at full occupancy but without sync_start is rejected."""
        with pytest.raises(Exception, match="sync_start=True"):
            _run(_cluster_spmd_program_no_sync(48))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
