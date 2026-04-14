# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for SPMD (Single Program Multiple Data) execution.

This module tests multi-block data-parallel dispatch using pl.spmd(core_num=N)
and pl.tile.get_block_idx(). Each block processes a different slice of the input
tensors and writes its result to the corresponding output region.

Tests cover:
  - Single SPMD submission smoke tests (add, mul)
  - Three sequential SPMD submissions (add, mul, sub pipeline)
  - Escalating core_num dispatch (4 -> 8 -> 12 -> 16 -> 24 blocks)
  - MixedKernel SPMD: matmul + bias add (cube + vector → AIC + AIV split)
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# --- Programs ---

CORE_NUM = 4
TILE_ROWS = 128
TILE_COLS = 128
TOTAL_ROWS = CORE_NUM * TILE_ROWS  # 512


@pl.program
class SPMDAddProgram:
    """SPMD elementwise add: 4 blocks each process a [128, 128] slice of a [512, 128] tensor."""

    @pl.function(type=pl.FunctionType.InCore)
    def spmd_add(
        self,
        a: pl.Tensor[[512, 128], pl.FP32],
        b: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        block_idx = pl.tile.get_block_idx()
        offset = block_idx * 128
        tile_a = pl.load(a, [offset, 0], [128, 128])
        tile_b = pl.load(b, [offset, 0], [128, 128])
        tile_c = pl.add(tile_a, tile_b)
        out = pl.store(tile_c, [offset, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[512, 128], pl.FP32],
        b: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.spmd(core_num=4):
            out = self.spmd_add(a, b, out)
        return out


@pl.program
class SPMDMulProgram:
    """SPMD elementwise mul: 4 blocks each process a [128, 128] slice of a [512, 128] tensor."""

    @pl.function(type=pl.FunctionType.InCore)
    def spmd_mul(
        self,
        a: pl.Tensor[[512, 128], pl.FP32],
        b: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        block_idx = pl.tile.get_block_idx()
        offset = block_idx * 128
        tile_a = pl.load(a, [offset, 0], [128, 128])
        tile_b = pl.load(b, [offset, 0], [128, 128])
        tile_c = pl.mul(tile_a, tile_b)
        out = pl.store(tile_c, [offset, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[512, 128], pl.FP32],
        b: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.spmd(core_num=4):
            out = self.spmd_mul(a, b, out)
        return out


@pl.program
class SPMDThreeSubmitProgram:
    """Three sequential SPMD submissions: add, mul, sub.

    Submission 1: t1  = a + b    (4 blocks)
    Submission 2: t2  = t1 * a   (4 blocks)
    Submission 3: out = t2 - b   (4 blocks)
    Result: out = (a + b) * a - b
    """

    @pl.function(type=pl.FunctionType.InCore)
    def spmd_add(
        self,
        a: pl.Tensor[[512, 128], pl.FP32],
        b: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        block_idx = pl.tile.get_block_idx()
        offset = block_idx * 128
        tile_a = pl.load(a, [offset, 0], [128, 128])
        tile_b = pl.load(b, [offset, 0], [128, 128])
        tile_c = pl.add(tile_a, tile_b)
        out = pl.store(tile_c, [offset, 0], out)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def spmd_mul(
        self,
        a: pl.Tensor[[512, 128], pl.FP32],
        b: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        block_idx = pl.tile.get_block_idx()
        offset = block_idx * 128
        tile_a = pl.load(a, [offset, 0], [128, 128])
        tile_b = pl.load(b, [offset, 0], [128, 128])
        tile_c = pl.mul(tile_a, tile_b)
        out = pl.store(tile_c, [offset, 0], out)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def spmd_sub(
        self,
        a: pl.Tensor[[512, 128], pl.FP32],
        b: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        block_idx = pl.tile.get_block_idx()
        offset = block_idx * 128
        tile_a = pl.load(a, [offset, 0], [128, 128])
        tile_b = pl.load(b, [offset, 0], [128, 128])
        tile_c = pl.sub(tile_a, tile_b)
        out = pl.store(tile_c, [offset, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[512, 128], pl.FP32],
        b: pl.Tensor[[512, 128], pl.FP32],
        t1: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        t2: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.spmd(core_num=4):
            t1 = self.spmd_add(a, b, t1)
        with pl.spmd(core_num=4):
            t2 = self.spmd_mul(t1, a, t2)
        with pl.spmd(core_num=4):
            out = self.spmd_sub(t2, b, out)
        return out


@pl.program
class SPMDEscalating5Program:
    """5 sequential SPMD submissions with escalating core_num.

    Mirrors the hardware SPMD multiblock pattern: each submission increases
    the block count to exercise progressively wider dispatch.

    T0: core_num=1,  base=0,  rows [0, 128)    — basic multi-block
    T1: core_num=2,  base=1,  rows [128, 384)  — saturate one sched thread
    T2: core_num=3,  base=3,  rows [384, 768)  — cross-thread dispatch
    T3: core_num=4,  base=6,  rows [768, 1280) — occupy all AIV cores
    T4: core_num=6,  base=10, rows [1280, 2048) — multi-round full dispatch
    Total: 16 blocks, output [2048, 128]
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add(
        self,
        a: pl.Tensor[[2048, 128], pl.FP32],
        b: pl.Tensor[[2048, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[2048, 128], pl.FP32]],
        base_offset: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[2048, 128], pl.FP32]:
        block_idx = pl.tile.get_block_idx()
        offset = (block_idx + base_offset) * 128
        tile_a = pl.load(a, [offset, 0], [128, 128])
        tile_b = pl.load(b, [offset, 0], [128, 128])
        tile_c = pl.add(tile_a, tile_b)
        out = pl.store(tile_c, [offset, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[2048, 128], pl.FP32],
        b: pl.Tensor[[2048, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[2048, 128], pl.FP32]],
    ) -> pl.Tensor[[2048, 128], pl.FP32]:
        with pl.spmd(core_num=1):
            out = self.kernel_add(a, b, out, 0)
        with pl.spmd(core_num=2):
            out = self.kernel_add(a, b, out, 1)
        with pl.spmd(core_num=3):
            out = self.kernel_add(a, b, out, 3)
        with pl.spmd(core_num=4):
            out = self.kernel_add(a, b, out, 6)
        with pl.spmd(core_num=6):
            out = self.kernel_add(a, b, out, 10)
        return out


# MixedKernel SPMD: InCore with split=UP_DOWN containing matmul + add
# 4 blocks, each processes a [64, 64] row slice: out[i] = a[i] @ b + bias[i]
MIX_M_PER_BLOCK = 64
MIX_K = 64
MIX_N = 64
MIX_BLOCKS = 4
MIX_TOTAL_M = MIX_BLOCKS * MIX_M_PER_BLOCK  # 256


@pl.program
class SPMDMixedKernelProgram:
    """SPMD MixedKernel: 4 blocks each compute a [64, 64] matmul + bias add.

    Uses InCore with split=UP_DOWN to enable AIC/AIV mixed kernel split
    under SPMD dispatch. Each block computes:
        out[block_slice] = a[block_slice] @ b + bias[block_slice]
    """

    @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
    def matmul_bias(
        self,
        a: pl.Tensor[[256, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        bias: pl.Tensor[[256, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
    ) -> pl.Tensor[[256, 64], pl.FP32]:
        block_idx = pl.tile.get_block_idx()
        row_offset = block_idx * 64

        # Cube: matmul → AIC
        tile_a_l1 = pl.load(a, [row_offset, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_mm = pl.matmul(tile_a_l0a, tile_b_l0b)

        # Vector: add bias → AIV
        tile_bias = pl.load(bias, [row_offset, 0], [64, 64])
        tile_out = pl.add(tile_mm, tile_bias)
        out = pl.store(tile_out, [row_offset, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[256, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        bias: pl.Tensor[[256, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
    ) -> pl.Tensor[[256, 64], pl.FP32]:
        with pl.spmd(core_num=4):
            out = self.matmul_bias(a, b, bias, out)
        return out


# --- Test Cases ---


class SPMDAddTestCase(PTOTestCase):
    """SPMD add: 4 blocks, each processes [128, 128] of a [512, 128] tensor."""

    def get_name(self) -> str:
        return "spmd_add_512x128"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [TOTAL_ROWS, TILE_COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SPMDAddProgram

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = tensors["a"] + tensors["b"]


class SPMDMulTestCase(PTOTestCase):
    """SPMD mul: 4 blocks, each processes [128, 128] of a [512, 128] tensor."""

    def get_name(self) -> str:
        return "spmd_mul_512x128"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [TOTAL_ROWS, TILE_COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SPMDMulProgram

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = tensors["a"] * tensors["b"]


# Escalating-5: T0(4) + T1(8) + T2(12) + T3(16) + T4(24) = 64 blocks, 8192 rows
ESC5_TOTAL_ROWS = 2048


class _BaseSPMDTestCase(PTOTestCase):
    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


class SPMDThreeSubmitTestCase(_BaseSPMDTestCase):
    """SPMD three submissions: out = (a + b) * a - b."""

    def get_name(self) -> str:
        return "spmd_three_submit_512x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("t1", [TOTAL_ROWS, TILE_COLS], DataType.FP32),
            TensorSpec("t2", [TOTAL_ROWS, TILE_COLS], DataType.FP32),
            TensorSpec("out", [TOTAL_ROWS, TILE_COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SPMDThreeSubmitProgram

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = (tensors["a"] + tensors["b"]) * tensors["a"] - tensors["b"]


class SPMDEscalating5TestCase(_BaseSPMDTestCase):
    """5 escalating SPMD submissions: core_num 1 -> 2 -> 3 -> 4 -> 6, all writing a + b."""

    def get_name(self) -> str:
        return "spmd_escalating5_2048x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [ESC5_TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [ESC5_TOTAL_ROWS, TILE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [ESC5_TOTAL_ROWS, TILE_COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SPMDEscalating5Program

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = tensors["a"] + tensors["b"]


class SPMDMixedKernelTestCase(_BaseSPMDTestCase):
    """SPMD MixedKernel: 4 blocks, each computes 64x64 matmul + bias add."""

    def get_name(self) -> str:
        return "spmd_mixed_matmul_bias_256x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [MIX_TOTAL_M, MIX_K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [MIX_K, MIX_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("bias", [MIX_TOTAL_M, MIX_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [MIX_TOTAL_M, MIX_N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SPMDMixedKernelProgram

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = torch.matmul(tensors["a"], tensors["b"]) + tensors["bias"]


# --- Tests ---


class TestSPMDOperations:
    """Test suite for SPMD multi-block dispatch."""

    @staticmethod
    def _run_case(test_runner, test_case):
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize(
        ("test_case_cls", "description"),
        [
            pytest.param(SPMDAddTestCase, "add", id="add"),
            pytest.param(SPMDMulTestCase, "mul", id="mul"),
        ],
    )
    def test_spmd_single_submit(self, test_runner, test_case_cls, description):
        """Single-submit SPMD smoke tests for basic vector kernels."""
        self._run_case(test_runner, test_case_cls())

    def test_spmd_three_submit(self, test_runner):
        """Three-submit chain covers sequential SPMD dependency handling."""
        self._run_case(test_runner, SPMDThreeSubmitTestCase())

    def test_spmd_escalating_5(self, test_runner):
        """Wide escalating dispatch covers the smaller 3-submit escalating case."""
        self._run_case(test_runner, SPMDEscalating5TestCase())

    @pytest.mark.xfail(reason="SPMD+MixedKernel precision issue under investigation")
    def test_spmd_mixed_kernel(self, test_runner):
        """SPMD MixedKernel: matmul + bias (cube + vector → AIC + AIV split)."""
        self._run_case(test_runner, SPMDMixedKernelTestCase())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
