# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime test for the cross-core all-participant barrier ``pl.system.syncall``.

``pl.system.syncall(core_type="aiv_only")`` lowers to the hard/FFTS form of
``pto::SYNCALL``, which issues ``ffts_cross_core_sync(SYNC_AIV_ONLY_ALL)`` and
waits for *every* AIV core in the FFTS group to arrive. This imposes two hard
requirements on the launch: (1) **full occupancy** — it must fill **all**
physical AIV cores, or the unlaunched cores never reach the barrier; and (2)
**sync_start=True** — all blocks must be co-resident at once, or the scheduler
may dispatch them in waves and the FFTS wait deadlocks even at full occupancy.
Either gap leaves the FFTS wait unable to complete and the AICore times out
(507018). The soft (GM-polling) form exists for partial occupancy.

The kernel therefore runs a full-occupancy SPMD elementwise add: ``CORE_NUM``
blocks, one per physical AIV core, launched with ``sync_start=True``, with the
barrier between the input loads and the compute. The barrier is a no-op for the
numeric result (``out = a + b``) but exercises the full DSL -> pass pipeline ->
codegen -> runtime path on device.

``CORE_NUM`` is derived from the Ascend910B backend's SoC model (the same core
description the compiler targets), so the launch always fills the AIV grid the
codegen assumes. This test targets the a2a3 (910B) profile; the 950/a5 AIV
count differs, so it is parametrized over the a2a3 platforms only.
"""

import sys
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto import backend
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.pypto_core import ir as _ir


def _core_count(bk: backend.Backend, core_type: "_ir.CoreType") -> int:
    """Total physical core count of ``core_type`` described by a backend's SoC model."""
    total = 0
    for die, die_n in bk.soc.die_counts.items():
        for cluster, cluster_n in die.cluster_counts.items():
            for core, core_n in cluster.core_counts.items():
                if core.core_type == core_type:
                    total += core_n * cluster_n * die_n
    return total


def _aiv_core_count(bk: backend.Backend) -> int:
    """Total VECTOR (AIV) core count described by a backend's SoC model."""
    return _core_count(bk, _ir.CoreType.VECTOR)


def _aic_core_count(bk: backend.Backend) -> int:
    """Total CUBE (AIC) core count described by a backend's SoC model."""
    return _core_count(bk, _ir.CoreType.CUBE)


# Hard-form SYNCALL requires full AIV-core occupancy (see module docstring).
# Derive the count from the 910B SoC model rather than hardcoding it.
CORE_NUM = _aiv_core_count(backend.Backend910B.instance())  # 48 on Ascend910B
TILE_ROWS = 128
TILE_COLS = 128
TOTAL_ROWS = CORE_NUM * TILE_ROWS  # 48 * 128 = 6144 on 910B

# Hard-form SYNCALL needs the full AIV grid; the 950/a5 AIV count differs, so
# restrict this test to the a2a3 (910B) profile it was validated against.
A2A3_PLATFORMS = ("a2a3", "a2a3sim")

# Soft-form SYNCALL polls a shared GM workspace, so it works at *partial*
# occupancy. Use a small block count to exercise that (a hard barrier would
# deadlock here). The GM workspace needs used_cores * 8 zero-initialized int32
# slots, shared across all blocks (passed as a kernel parameter).
SOFT_CORE_NUM = 4
SOFT_TOTAL_ROWS = SOFT_CORE_NUM * TILE_ROWS  # 512
SOFT_WS_SLOTS = SOFT_CORE_NUM * 8

# Mixed (AIC + AIV) soft-form SYNCALL. On 910B one cube block pairs with
# AIV_RATIO vector subblocks (1 AIC + 2 AIV), so a launch of B cube blocks has
# B * (1 + AIV_RATIO) soft participants. Derive the ratio from the SoC model.
_BK910B = backend.Backend910B.instance()
AIV_RATIO = _aiv_core_count(_BK910B) // _aic_core_count(_BK910B)  # 48 // 24 = 2
MIX_CUBE_BLOCKS = 2  # partial occupancy (a hard mix barrier would deadlock here)
MIX_USED_CORES = MIX_CUBE_BLOCKS * (1 + AIV_RATIO)  # 2 * 3 = 6 total AIC+AIV participants
MIX_WS_SLOTS = MIX_USED_CORES * 8  # 48
MIX_M = 32
MIX_K = 64
MIX_N = 32
MIX_ROW_TILE = MIX_M // MIX_CUBE_BLOCKS  # 16 rows per cube block

# AIC-only soft-form SYNCALL: only cube cores participate, so a launch of B cube
# blocks has exactly B soft participants.
AIC_CUBE_BLOCKS = 2
AIC_USED_CORES = AIC_CUBE_BLOCKS  # cube cores only
AIC_WS_SLOTS = AIC_USED_CORES * 8  # 16
AIC_M = 32
AIC_K = 64
AIC_N = 32
AIC_ROW_TILE = AIC_M // AIC_CUBE_BLOCKS  # 16


@pl.program
class SPMDSyncAllProgram:
    """SPMD add with an AIV-only cross-core barrier between load and compute."""

    @pl.function(type=pl.FunctionType.InCore)
    def spmd_syncall_add(
        self,
        a: pl.Tensor[[TOTAL_ROWS, TILE_COLS], pl.FP32],
        b: pl.Tensor[[TOTAL_ROWS, TILE_COLS], pl.FP32],
        out: pl.Out[pl.Tensor[[TOTAL_ROWS, TILE_COLS], pl.FP32]],
    ) -> pl.Tensor[[TOTAL_ROWS, TILE_COLS], pl.FP32]:
        block_idx = pl.tile.get_block_idx()
        offset = block_idx * TILE_ROWS
        tile_a = pl.load(a, [offset, 0], [TILE_ROWS, TILE_COLS])
        tile_b = pl.load(b, [offset, 0], [TILE_ROWS, TILE_COLS])
        # All AIV cores arrive here before any core proceeds to compute.
        pl.system.syncall(core_type="aiv_only")
        tile_c = pl.add(tile_a, tile_b)
        out = pl.store(tile_c, [offset, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[TOTAL_ROWS, TILE_COLS], pl.FP32],
        b: pl.Tensor[[TOTAL_ROWS, TILE_COLS], pl.FP32],
        out: pl.Out[pl.Tensor[[TOTAL_ROWS, TILE_COLS], pl.FP32]],
    ) -> pl.Tensor[[TOTAL_ROWS, TILE_COLS], pl.FP32]:
        # sync_start=True: the hard/FFTS barrier needs all CORE_NUM blocks
        # co-resident at once, not merely full occupancy (see module docstring).
        with pl.spmd(CORE_NUM, sync_start=True):
            out = self.spmd_syncall_add(a, b, out)
        return out


class SPMDSyncAllTestCase(PTOTestCase):
    """SPMD add + aiv_only syncall: 4 blocks, each processes [128, 128] of [512, 128]."""

    __test__ = False

    def get_name(self) -> str:
        return "spmd_syncall_aiv_512x128"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [TOTAL_ROWS, TILE_COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SPMDSyncAllProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["a"] + tensors["b"]


@pl.program
class SPMDSyncAllSoftProgram:
    """SPMD add with an AIV-only *soft* (GM-polling) barrier at partial occupancy."""

    @pl.function(type=pl.FunctionType.InCore)
    def spmd_syncall_soft_add(
        self,
        a: pl.Tensor[[SOFT_TOTAL_ROWS, TILE_COLS], pl.FP32],
        b: pl.Tensor[[SOFT_TOTAL_ROWS, TILE_COLS], pl.FP32],
        sync_ws: pl.Tensor[[SOFT_WS_SLOTS], pl.INT32],
        out: pl.Out[pl.Tensor[[SOFT_TOTAL_ROWS, TILE_COLS], pl.FP32]],
    ) -> pl.Tensor[[SOFT_TOTAL_ROWS, TILE_COLS], pl.FP32]:
        block_idx = pl.tile.get_block_idx()
        offset = block_idx * TILE_ROWS
        tile_a = pl.load(a, [offset, 0], [TILE_ROWS, TILE_COLS])
        tile_b = pl.load(b, [offset, 0], [TILE_ROWS, TILE_COLS])
        # GM-polling barrier across the SOFT_CORE_NUM participating AIV cores.
        pl.system.syncall(mode="soft", core_type="aiv_only", gm_workspace=sync_ws, used_cores=SOFT_CORE_NUM)
        tile_c = pl.add(tile_a, tile_b)
        out = pl.store(tile_c, [offset, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[SOFT_TOTAL_ROWS, TILE_COLS], pl.FP32],
        b: pl.Tensor[[SOFT_TOTAL_ROWS, TILE_COLS], pl.FP32],
        sync_ws: pl.Tensor[[SOFT_WS_SLOTS], pl.INT32],
        out: pl.Out[pl.Tensor[[SOFT_TOTAL_ROWS, TILE_COLS], pl.FP32]],
    ) -> pl.Tensor[[SOFT_TOTAL_ROWS, TILE_COLS], pl.FP32]:
        with pl.spmd(SOFT_CORE_NUM):
            out = self.spmd_syncall_soft_add(a, b, sync_ws, out)
        return out


class SPMDSyncAllSoftTestCase(PTOTestCase):
    """SPMD add + aiv_only soft syncall at partial occupancy (4 blocks)."""

    __test__ = False

    def get_name(self) -> str:
        return "spmd_syncall_soft_aiv_512x128"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [SOFT_TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [SOFT_TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            # Shared GM workspace: zero-initialized int32 counter slots.
            TensorSpec("sync_ws", [SOFT_WS_SLOTS], DataType.INT32, init_value=torch.zeros),
            TensorSpec("out", [SOFT_TOTAL_ROWS, TILE_COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SPMDSyncAllSoftProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["a"] + tensors["b"]


@pl.program
class SPMDSyncAllMixSoftProgram:
    """Mixed (AIC + AIV) *soft* syncall at partial occupancy.

    A fused cube+vector kernel: a vector op produces the matmul operand (V->C)
    and a vector op consumes the result (C->V), so the scope is genuinely mixed
    and ExpandMixedKernel splits it across both lanes. The mix soft barrier is
    duplicated onto both lanes (SHARED) and rendezvouses all B*(1+ratio)
    participants. The barrier is numerically a no-op: ``out = (a + 1) @ b + 1``.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[MIX_M, MIX_K], pl.FP32],
        b: pl.Tensor[[MIX_K, MIX_N], pl.FP32],
        sync_ws: pl.Tensor[[MIX_WS_SLOTS], pl.INT32],
        out: pl.Out[pl.Tensor[[MIX_M, MIX_N], pl.FP32]],
    ) -> pl.Tensor[[MIX_M, MIX_N], pl.FP32]:
        for ob in pl.spmd(MIX_CUBE_BLOCKS, name_hint="mix_syncall"):
            m0 = ob * MIX_ROW_TILE
            a_slice = pl.slice(a, [MIX_ROW_TILE, MIX_K], [m0, 0])
            a_add = pl.add(a_slice, 1.0)  # vector produces the matmul operand (V->C)
            # Cross-core barrier across every participating AIC block and AIV subblock.
            pl.system.syncall(mode="soft", core_type="mix", gm_workspace=sync_ws, used_cores=MIX_USED_CORES)
            c_tile = pl.matmul(a_add, b)  # cube
            c_vec = pl.add(c_tile, 1.0)  # vector consumes the matmul result (C->V)
            out = pl.assemble(out, c_vec, [m0, 0])
        return out


class SPMDSyncAllMixSoftTestCase(PTOTestCase):
    """Mixed soft syncall at partial occupancy (2 cube blocks -> 6 participants)."""

    __test__ = False

    def get_name(self) -> str:
        return "spmd_syncall_soft_mix_32x32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [MIX_M, MIX_K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [MIX_K, MIX_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("sync_ws", [MIX_WS_SLOTS], DataType.INT32, init_value=torch.zeros),
            TensorSpec("out", [MIX_M, MIX_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SPMDSyncAllMixSoftProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        b = tensors["b"]
        for m0 in range(0, MIX_M, MIX_ROW_TILE):
            tensors["out"][m0 : m0 + MIX_ROW_TILE] = torch.matmul(a[m0 : m0 + MIX_ROW_TILE] + 1.0, b) + 1.0


@pl.program
class SPMDSyncAllAicSoftProgram:
    """AIC-only *soft* syncall in a cube-only kernel at partial occupancy.

    A pure matmul per row-tile; only cube cores participate in the barrier.
    The barrier is numerically a no-op: ``out = a @ b``.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[AIC_M, AIC_K], pl.FP32],
        b: pl.Tensor[[AIC_K, AIC_N], pl.FP32],
        sync_ws: pl.Tensor[[AIC_WS_SLOTS], pl.INT32],
        out: pl.Out[pl.Tensor[[AIC_M, AIC_N], pl.FP32]],
    ) -> pl.Tensor[[AIC_M, AIC_N], pl.FP32]:
        for ob in pl.spmd(AIC_CUBE_BLOCKS, name_hint="aic_syncall"):
            m0 = ob * AIC_ROW_TILE
            a_slice = pl.slice(a, [AIC_ROW_TILE, AIC_K], [m0, 0])
            # Cube-only cross-core barrier across the AIC_CUBE_BLOCKS cube cores.
            pl.system.syncall(
                mode="soft", core_type="aic_only", gm_workspace=sync_ws, used_cores=AIC_USED_CORES
            )
            c_tile = pl.matmul(a_slice, b)  # cube
            out = pl.assemble(out, c_tile, [m0, 0])
        return out


class SPMDSyncAllAicSoftTestCase(PTOTestCase):
    """AIC-only soft syncall at partial occupancy (2 cube blocks)."""

    __test__ = False

    def get_name(self) -> str:
        return "spmd_syncall_soft_aic_32x32"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [AIC_M, AIC_K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [AIC_K, AIC_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("sync_ws", [AIC_WS_SLOTS], DataType.INT32, init_value=torch.zeros),
            TensorSpec("out", [AIC_M, AIC_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SPMDSyncAllAicSoftProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        b = tensors["b"]
        for m0 in range(0, AIC_M, AIC_ROW_TILE):
            tensors["out"][m0 : m0 + AIC_ROW_TILE] = torch.matmul(a[m0 : m0 + AIC_ROW_TILE], b)


class TestSyncAll:
    """Cross-core all-participant barrier (pl.system.syncall) system test."""

    @pytest.mark.parametrize("platform", A2A3_PLATFORMS)
    def test_spmd_syncall_aiv_only(self, test_runner, platform):
        """SPMD add with an aiv_only syncall barrier: compile, run, and verify out = a + b."""
        result = test_runner.run(SPMDSyncAllTestCase(platform=platform))
        assert result.passed, f"SPMD aiv_only syncall failed: {result.error}"

    @pytest.mark.parametrize("platform", A2A3_PLATFORMS)
    def test_spmd_syncall_soft_aiv_only(self, test_runner, platform):
        """SPMD add with an aiv_only *soft* barrier at partial occupancy: verify out = a + b."""
        result = test_runner.run(SPMDSyncAllSoftTestCase(platform=platform))
        assert result.passed, f"SPMD aiv_only soft syncall failed: {result.error}"

    @pytest.mark.parametrize("platform", A2A3_PLATFORMS)
    def test_spmd_syncall_soft_mix(self, test_runner, platform):
        """Mixed (AIC+AIV) *soft* barrier at partial occupancy: verify out = (a + 1) @ b + 1."""
        result = test_runner.run(SPMDSyncAllMixSoftTestCase(platform=platform))
        assert result.passed, f"SPMD mix soft syncall failed: {result.error}"

    @pytest.mark.parametrize("platform", A2A3_PLATFORMS)
    def test_spmd_syncall_soft_aic_only(self, test_runner, platform):
        """AIC-only *soft* barrier at partial occupancy: verify out = a @ b."""
        result = test_runner.run(SPMDSyncAllAicSoftTestCase(platform=platform))
        assert result.passed, f"SPMD aic_only soft syncall failed: {result.error}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
