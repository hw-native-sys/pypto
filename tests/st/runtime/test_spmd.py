# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
SPMD (Single Program Multiple Data) System Tests.

Aligns with simpler examples under examples/a5/tensormap_and_ringbuffer/spmd_*:

  SpmdBasicTest        : Single-block SPMD launch (core_num=1).
                         Verifies the basic SPMD orchestration codegen path.
  SpmdMultiblockTest   : Multi-launch SPMD (multiple spmd_launch calls with
                         increasing core_num=4,16,24,48 and mixed sync_start).
                         Each block processes a disjoint row slice via
                         pl.tile.get_block_idx() dynamic offsets.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE_H = 16
TILE_W = 16
MAX_BLOCKS = 48
TOTAL_H_MULTI = MAX_BLOCKS * TILE_H  # 768

# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------


@pl.program
class SpmdBasicProgram:
    """Single-block SPMD: basic launch with core_num=1.

    Corresponds to the simplest SPMD case where a single block runs
    a tile add-scalar operation.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[TILE_H, TILE_W], pl.FP32],
        output: pl.InOut[pl.Tensor[[TILE_H, TILE_W], pl.FP32]],
    ):
        tile_a: pl.Tile[[TILE_H, TILE_W], pl.FP32] = pl.load(a, [0, 0], [TILE_H, TILE_W])
        tile_result: pl.Tile[[TILE_H, TILE_W], pl.FP32] = pl.adds(tile_a, 1.0)
        pl.store(tile_result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[TILE_H, TILE_W], pl.FP32],
        output: pl.InOut[pl.Tensor[[TILE_H, TILE_W], pl.FP32]],
    ):
        pl.spmd_launch(self.kernel, a, output, core_num=1)


@pl.program
class SpmdMultiblockProgram:
    """Multi-launch SPMD: multiple spmd_launch calls with increasing core_num.

    Corresponds to spmd_multiblock_mix / spmd_sync_start orchestration pattern
    where multiple tasks are submitted with increasing block counts and mixed
    sync_start.  Each block uses get_block_idx() to compute its row offset.

    Launches:
      T0: core_num=4                — basic multi-block
      T1: core_num=16, sync_start   — saturates one sched thread
      T2: core_num=24               — forces cross-thread dispatch
      T3: core_num=48, sync_start   — two full rounds of all clusters
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[TOTAL_H_MULTI, TILE_W], pl.FP32],
        output: pl.InOut[pl.Tensor[[TOTAL_H_MULTI, TILE_W], pl.FP32]],
    ):
        block_idx = pl.tile.get_block_idx()
        row_start = block_idx * TILE_H
        tile_a: pl.Tile[[TILE_H, TILE_W], pl.FP32] = pl.load(a, [row_start, 0], [TILE_H, TILE_W])
        tile_result: pl.Tile[[TILE_H, TILE_W], pl.FP32] = pl.adds(tile_a, 1.0)
        pl.store(tile_result, [row_start, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[TOTAL_H_MULTI, TILE_W], pl.FP32],
        output: pl.InOut[pl.Tensor[[TOTAL_H_MULTI, TILE_W], pl.FP32]],
    ):
        pl.spmd_launch(self.kernel, a, output, core_num=4)
        pl.spmd_launch(self.kernel, a, output, core_num=16, sync_start=True)
        pl.spmd_launch(self.kernel, a, output, core_num=24)
        pl.spmd_launch(self.kernel, a, output, core_num=48, sync_start=True)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class SpmdBasicTest(PTOTestCase):
    """SPMD basic: single-block launch with core_num=1."""

    __test__ = False

    def get_name(self) -> str:
        return "spmd_basic"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [TILE_H, TILE_W], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [TILE_H, TILE_W], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SpmdBasicProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["output"][:] = tensors["a"] + 1.0


class SpmdMultiblockTest(PTOTestCase):
    """SPMD multiblock: multiple launches with core_num=4,16,24,48."""

    __test__ = False

    def get_name(self) -> str:
        return "spmd_multiblock"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [TOTAL_H_MULTI, TILE_W], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [TOTAL_H_MULTI, TILE_W], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SpmdMultiblockProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["output"][:] = tensors["a"] + 1.0


# ---- A5 (Ascend 950) variants ----


class SpmdBasicA5Test(SpmdBasicTest):
    """SPMD basic on A5 (Ascend 950) backend."""

    __test__ = False

    def get_name(self) -> str:
        return "spmd_basic_a5"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950


class SpmdMultiblockA5Test(SpmdMultiblockTest):
    """SPMD multiblock on A5 (Ascend 950) backend."""

    __test__ = False

    def get_name(self) -> str:
        return "spmd_multiblock_a5"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950


# =============================================================================
# pytest test functions
# =============================================================================


class TestSpmd:
    """SPMD dispatch system tests (Ascend 910B / A2A3)."""

    def test_spmd_basic(self, test_runner):
        """Single-block SPMD launch with core_num=1."""
        result = test_runner.run(SpmdBasicTest())
        assert result.passed, f"SPMD basic failed: {result.error}"

    def test_spmd_multiblock(self, test_runner):
        """Multi-launch SPMD with core_num=4,16,24,48 and mixed sync_start."""
        result = test_runner.run(SpmdMultiblockTest())
        assert result.passed, f"SPMD multiblock failed: {result.error}"

    # ---- A5 (Ascend 950) tests ----

    @pytest.mark.a5
    def test_spmd_basic_a5(self, test_runner):
        """Single-block SPMD launch on A5 (Ascend 950)."""
        result = test_runner.run(SpmdBasicA5Test())
        assert result.passed, f"SPMD basic (A5) failed: {result.error}"

    @pytest.mark.a5
    def test_spmd_multiblock_a5(self, test_runner):
        """Multi-launch SPMD on A5 (Ascend 950)."""
        result = test_runner.run(SpmdMultiblockA5Test())
        assert result.passed, f"SPMD multiblock (A5) failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
