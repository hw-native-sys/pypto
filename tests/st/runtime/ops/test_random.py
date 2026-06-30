# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Test the random operation (TRANDOM, A5-only).

``pto.trandom`` is a counter-based (Philox/ChaCha-style) RNG that fills a tile
with pseudo-random 32-bit values derived from a 64-bit key + 128-bit counter and
the element position. It is implemented on **A5 (Ascend950) only** — there is no
A2/A3 device path — so these cases are gated with
``@pytest.mark.platforms("a5", "a5sim")``.

Verification strategy — **determinism + shape** (not a bit-exact numeric golden):
the device output of a counter RNG is a complex function of the seeds and the
hardware's internal lane/round schedule, so the golden is intentionally not
reproduced here. Instead each kernel computes ``random(seed) - random(seed)``
with identical seeds; because the generator is deterministic the two draws are
bit-identical and the difference is exactly zero. This proves the op:

- assembles on A5 (the ``pto.trandom`` codegen format is accepted),
- executes without crashing, and
- is deterministic (same key/counter -> same output).

Coverage spans both output dtypes (UINT32, INT32), both round counts (7, 10),
several shapes, a wide-column case (> 4*lanes) that drives the multi-iteration
column loop, a narrow ``valid_shape`` case (trandom fills only the valid rows/cols;
the rest is zeroed with ``fillpad`` so the golden stays deterministic), and a
tensor-level case. The seeds are non-zero and vary per case.

Known limitation: a hypothetical device defect that made ``trandom`` emit an
all-zero tile would still satisfy ``0 - 0 == 0`` and pass. Catching that would
require reproducing the exact vectorized Philox schedule as a numeric golden,
which is deliberately out of scope for this determinism-focused suite.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# =============================================================================
# Programs — one explicit @pl.program per scenario. Each kernel draws the same
# seed twice and subtracts, so the (deterministic) draws cancel to zero.
# =============================================================================


@pl.program
class RandomDetermUInt32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, output: pl.Out[pl.Tensor[[4, 256], pl.UINT32]]) -> pl.Tensor[[4, 256], pl.UINT32]:
        a: pl.Tile[[4, 256], pl.UINT32] = pl.tile.random(0x1234, 0x5678, 1, 2, 3, 4, [4, 256])
        b: pl.Tile[[4, 256], pl.UINT32] = pl.tile.random(0x1234, 0x5678, 1, 2, 3, 4, [4, 256])
        diff: pl.Tile[[4, 256], pl.UINT32] = pl.tile.sub(a, b)
        return pl.store(diff, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(self, output: pl.Out[pl.Tensor[[4, 256], pl.UINT32]]) -> pl.Tensor[[4, 256], pl.UINT32]:
        return self.kernel(output)


@pl.program
class RandomDetermInt32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, output: pl.Out[pl.Tensor[[8, 128], pl.INT32]]) -> pl.Tensor[[8, 128], pl.INT32]:
        a: pl.Tile[[8, 128], pl.INT32] = pl.tile.random(0x0ABC, 0x0DEF, 7, 0, 0, 0, [8, 128], dtype=pl.INT32)
        b: pl.Tile[[8, 128], pl.INT32] = pl.tile.random(0x0ABC, 0x0DEF, 7, 0, 0, 0, [8, 128], dtype=pl.INT32)
        diff: pl.Tile[[8, 128], pl.INT32] = pl.tile.sub(a, b)
        return pl.store(diff, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(self, output: pl.Out[pl.Tensor[[8, 128], pl.INT32]]) -> pl.Tensor[[8, 128], pl.INT32]:
        return self.kernel(output)


@pl.program
class RandomDetermRounds7:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, output: pl.Out[pl.Tensor[[16, 64], pl.UINT32]]) -> pl.Tensor[[16, 64], pl.UINT32]:
        a: pl.Tile[[16, 64], pl.UINT32] = pl.tile.random(0x1111, 0x2222, 5, 6, 7, 8, [16, 64], rounds=7)
        b: pl.Tile[[16, 64], pl.UINT32] = pl.tile.random(0x1111, 0x2222, 5, 6, 7, 8, [16, 64], rounds=7)
        diff: pl.Tile[[16, 64], pl.UINT32] = pl.tile.sub(a, b)
        return pl.store(diff, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(self, output: pl.Out[pl.Tensor[[16, 64], pl.UINT32]]) -> pl.Tensor[[16, 64], pl.UINT32]:
        return self.kernel(output)


@pl.program
class RandomDetermWideCols:
    """Wide columns (512 > 4*lanes=256) exercise the multi-iteration column loop
    and its per-loop counter advance."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, output: pl.Out[pl.Tensor[[4, 512], pl.UINT32]]) -> pl.Tensor[[4, 512], pl.UINT32]:
        a: pl.Tile[[4, 512], pl.UINT32] = pl.tile.random(0x9, 0xA, 3, 0, 0, 0, [4, 512])
        b: pl.Tile[[4, 512], pl.UINT32] = pl.tile.random(0x9, 0xA, 3, 0, 0, 0, [4, 512])
        diff: pl.Tile[[4, 512], pl.UINT32] = pl.tile.sub(a, b)
        return pl.store(diff, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(self, output: pl.Out[pl.Tensor[[4, 512], pl.UINT32]]) -> pl.Tensor[[4, 512], pl.UINT32]:
        return self.kernel(output)


@pl.program
class RandomDetermValidShape:
    """Narrow ``valid_shape`` ([10, 80] inside a [16, 128] tile): trandom only fills
    the valid rows/cols. The determinism subtract zeroes the valid region; fillpad
    then zeroes the remaining (invalid) region so the whole output is deterministic.
    Exercises validRow < 16 (tail) and a non-256-aligned valid column count."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, output: pl.Out[pl.Tensor[[16, 128], pl.UINT32]]) -> pl.Tensor[[16, 128], pl.UINT32]:
        a: pl.Tile[[16, 128], pl.UINT32] = pl.tile.random(
            0x55, 0x66, 1, 0, 0, 0, [16, 128], valid_shape=[10, 80]
        )
        b: pl.Tile[[16, 128], pl.UINT32] = pl.tile.random(
            0x55, 0x66, 1, 0, 0, 0, [16, 128], valid_shape=[10, 80]
        )
        diff: pl.Tile[[16, 128], pl.UINT32] = pl.tile.sub(a, b)
        filled: pl.Tile[[16, 128], pl.UINT32] = pl.tile.fillpad(diff, pad_value=pl.PadValue.zero)
        return pl.store(filled, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self, output: pl.Out[pl.Tensor[[16, 128], pl.UINT32]]
    ) -> pl.Tensor[[16, 128], pl.UINT32]:
        return self.kernel(output)


@pl.program
class RandomDetermTensorUInt32:
    """Tensor-level entry point: pl.random on a whole Tensor, lowered by
    ConvertTensorToTileOps to tile.random."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, output: pl.Out[pl.Tensor[[4, 256], pl.UINT32]]) -> pl.Tensor[[4, 256], pl.UINT32]:
        a: pl.Tensor[[4, 256], pl.UINT32] = pl.random(0x1234, 0x5678, 1, 2, 3, 4, [4, 256])
        b: pl.Tensor[[4, 256], pl.UINT32] = pl.random(0x1234, 0x5678, 1, 2, 3, 4, [4, 256])
        diff: pl.Tensor[[4, 256], pl.UINT32] = pl.sub(a, b)
        return pl.assemble(output, diff, [0, 0])

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(self, output: pl.Out[pl.Tensor[[4, 256], pl.UINT32]]) -> pl.Tensor[[4, 256], pl.UINT32]:
        return self.kernel(output)


# =============================================================================
# Test cases — golden is all-zeros (identical seeds cancel under subtraction).
# =============================================================================

_TORCH_DTYPE = {
    DataType.UINT32: torch.int32,  # torch has no uint32; int32 bit-pattern compare is fine for x - x == 0
    DataType.INT32: torch.int32,
}


class _RandomDetermCase(PTOTestCase):
    """Base case: device output must equal an all-zero tile."""

    __test__ = False

    program: Any = None
    case_name: str = ""
    out_shape: list[int] = []
    dtype: DataType = DataType.UINT32

    def get_name(self) -> str:
        return self.case_name

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def define_tensors(self) -> list[TensorSpec]:
        return [TensorSpec("output", self.out_shape, self.dtype, is_output=True)]

    def get_program(self) -> Any:
        return self.program

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.zeros(self.out_shape, dtype=_TORCH_DTYPE[self.dtype])


class RandomDetermUInt32Case(_RandomDetermCase):
    program = RandomDetermUInt32
    case_name = "random_determ_uint32"
    out_shape = [4, 256]
    dtype = DataType.UINT32


class RandomDetermInt32Case(_RandomDetermCase):
    program = RandomDetermInt32
    case_name = "random_determ_int32"
    out_shape = [8, 128]
    dtype = DataType.INT32


class RandomDetermRounds7Case(_RandomDetermCase):
    program = RandomDetermRounds7
    case_name = "random_determ_rounds7"
    out_shape = [16, 64]
    dtype = DataType.UINT32


class RandomDetermWideColsCase(_RandomDetermCase):
    program = RandomDetermWideCols
    case_name = "random_determ_wide_cols"
    out_shape = [4, 512]
    dtype = DataType.UINT32


class RandomDetermValidShapeCase(_RandomDetermCase):
    program = RandomDetermValidShape
    case_name = "random_determ_valid_shape"
    out_shape = [16, 128]
    dtype = DataType.UINT32


class RandomDetermTensorUInt32Case(_RandomDetermCase):
    program = RandomDetermTensorUInt32
    case_name = "random_determ_tensor_uint32"
    out_shape = [4, 256]
    dtype = DataType.UINT32


# =============================================================================
# Tests (A5-only)
# =============================================================================


@pytest.mark.platforms("a5", "a5sim")
class TestRandom:
    """trandom: counter-based RNG, determinism verified by self-subtraction."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_determ_uint32(self, test_runner, platform):
        result = test_runner.run(RandomDetermUInt32Case(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_determ_int32(self, test_runner, platform):
        result = test_runner.run(RandomDetermInt32Case(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_determ_rounds7(self, test_runner, platform):
        result = test_runner.run(RandomDetermRounds7Case(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_determ_wide_cols(self, test_runner, platform):
        result = test_runner.run(RandomDetermWideColsCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_determ_valid_shape(self, test_runner, platform):
        result = test_runner.run(RandomDetermValidShapeCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_determ_tensor_uint32(self, test_runner, platform):
        result = test_runner.run(RandomDetermTensorUInt32Case(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
