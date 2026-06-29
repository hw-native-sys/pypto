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
waits for *every* AIV core in the FFTS group to arrive. This is a hard
requirement: the hard form only terminates when the launch fills **all**
physical AIV cores. A partial-occupancy launch (fewer blocks than cores) leaves
some cores never reaching the barrier, so the FFTS wait never completes and the
AICore times out (507018).

The kernel therefore runs a full-occupancy SPMD elementwise add: ``CORE_NUM``
blocks, one per physical AIV core, with the barrier between the input loads and
the compute. The barrier is a no-op for the numeric result (``out = a + b``) but
exercises the full DSL -> pass pipeline -> codegen -> runtime path on device.

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


def _aiv_core_count(bk: backend.Backend) -> int:
    """Total VECTOR (AIV) core count described by a backend's SoC model."""
    total = 0
    for die, die_n in bk.soc.die_counts.items():
        for cluster, cluster_n in die.cluster_counts.items():
            for core, core_n in cluster.core_counts.items():
                if core.core_type == _ir.CoreType.VECTOR:
                    total += core_n * cluster_n * die_n
    return total


# Hard-form SYNCALL requires full AIV-core occupancy (see module docstring).
# Derive the count from the 910B SoC model rather than hardcoding it.
CORE_NUM = _aiv_core_count(backend.Backend910B.instance())  # 48 on Ascend910B
TILE_ROWS = 128
TILE_COLS = 128
TOTAL_ROWS = CORE_NUM * TILE_ROWS  # 48 * 128 = 6144 on 910B

# Hard-form SYNCALL needs the full AIV grid; the 950/a5 AIV count differs, so
# restrict this test to the a2a3 (910B) profile it was validated against.
A2A3_PLATFORMS = ("a2a3", "a2a3sim")


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
        with pl.spmd(CORE_NUM):
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


class TestSyncAll:
    """Cross-core all-participant barrier (pl.system.syncall) system test."""

    @pytest.mark.parametrize("platform", A2A3_PLATFORMS)
    def test_spmd_syncall_aiv_only(self, test_runner, platform):
        """SPMD add with an aiv_only syncall barrier: compile, run, and verify out = a + b."""
        result = test_runner.run(SPMDSyncAllTestCase(platform=platform))
        assert result.passed, f"SPMD aiv_only syncall failed: {result.error}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
