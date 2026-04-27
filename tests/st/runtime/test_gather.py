# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end tests for ``pl.tensor.gather`` (issue #676) — torch-style semantics.

Covers the generalized contract beyond the original MVP (rank-2 + dim=-1):

1. Rank-2 + dim=-1 (baseline / regression).
2. Rank-2 + dim=-1 with ``index.shape[0] < input.shape[0]`` (smaller index leading).
3. Rank-3 + dim=-1 (collapses leading dims via ``tile.reshape``).
4. Rank-3 + dim=1 (middle axis — flat-index gather).
5. Rank-3 + dim=-3 (negative-dim normalization on the first axis).

All cases are validated against a torch ``gather`` reference.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# --- Shared init helpers ---


def _rand_indices(low: int, high: int, shape: tuple[int, ...]) -> torch.Tensor:
    """Random INT32 indices uniformly in [low, high)."""
    return torch.randint(low, high, shape, dtype=torch.int32)


# --- Programs ---


@pl.program
class GatherRank2LastDimProgram:
    """Baseline rank-2 + dim=-1: ``out[b, k] = input[b, index[b, k]]``."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[4, 16], pl.FP32],
        idx: pl.Tensor[[4, 8], pl.INT32],
        output: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.gather(inp, dim=-1, index=idx)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class GatherRank2SmallerLeadingProgram:
    """Rank-2 + dim=-1 with ``index.shape[0] (=2) < input.shape[0] (=4)``."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[4, 16], pl.FP32],
        idx: pl.Tensor[[2, 8], pl.INT32],
        output: pl.Out[pl.Tensor[[2, 8], pl.FP32]],
    ) -> pl.Tensor[[2, 8], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.gather(inp, dim=-1, index=idx)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class GatherRank3LastDimProgram:
    """Rank-3 + dim=-1. Lowering collapses leading dims via ``tile.reshape``.

    idx last-dim is 8 (8×4=32 bytes) to satisfy the hardware tile column
    alignment requirement (Cols * sizeof(dtype) % 32 == 0).
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[2, 3, 16], pl.FP32],
        idx: pl.Tensor[[2, 3, 8], pl.INT32],
        output: pl.Out[pl.Tensor[[2, 3, 8], pl.FP32]],
    ) -> pl.Tensor[[2, 3, 8], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.gather(inp, dim=-1, index=idx)
            output = pl.assemble(output, out, [0, 0, 0])
        return output


@pl.program
class GatherRank3MiddleDimProgram:
    """Rank-3 + dim=1 (middle axis) — flat-index gather.

    Last dim is 8 (8×4=32 bytes) to satisfy the hardware tile column
    alignment requirement.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[2, 8, 8], pl.FP32],
        idx: pl.Tensor[[2, 3, 8], pl.INT32],
        output: pl.Out[pl.Tensor[[2, 3, 8], pl.FP32]],
    ) -> pl.Tensor[[2, 3, 8], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.gather(inp, dim=1, index=idx)
            output = pl.assemble(output, out, [0, 0, 0])
        return output


@pl.program
class GatherRank3NegFirstDimProgram:
    """Rank-3 + dim=-3 (== dim=0): negative-dim normalization on the first axis.

    Last dim is 8 (8×4=32 bytes) to satisfy the hardware tile column
    alignment requirement.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[8, 2, 8], pl.FP32],
        idx: pl.Tensor[[3, 2, 8], pl.INT32],
        output: pl.Out[pl.Tensor[[3, 2, 8], pl.FP32]],
    ) -> pl.Tensor[[3, 2, 8], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.gather(inp, dim=-3, index=idx)
            output = pl.assemble(output, out, [0, 0, 0])
        return output


# --- Test cases ---


class _GatherBaseTestCase(PTOTestCase):
    __test__ = False

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


class GatherRank2LastDimTestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_rank2_last_dim"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [4, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec(
                "idx",
                [4, 8],
                DataType.INT32,
                init_value=lambda: _rand_indices(0, 16, (4, 8)),
            ),
            TensorSpec("output", [4, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherRank2LastDimProgram

    def compute_expected(self, tensors, params=None):
        # torch.gather semantics: out[b, k] = inp[b, idx[b, k]]
        inp = tensors["inp"]
        idx = tensors["idx"].to(torch.int64)
        tensors["output"][:] = torch.gather(inp, dim=-1, index=idx)


class GatherRank2SmallerLeadingTestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_rank2_smaller_leading"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [4, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec(
                "idx",
                [2, 8],
                DataType.INT32,
                init_value=lambda: _rand_indices(0, 16, (2, 8)),
            ),
            TensorSpec("output", [2, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherRank2SmallerLeadingProgram

    def compute_expected(self, tensors, params=None):
        # torch's index broadcast along the non-gather axis must match the
        # PyPTO contract: rows of the input beyond index.shape[0] are unused.
        inp = tensors["inp"][: tensors["idx"].shape[0]]
        idx = tensors["idx"].to(torch.int64)
        tensors["output"][:] = torch.gather(inp, dim=-1, index=idx)


class GatherRank3LastDimTestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_rank3_last_dim"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [2, 3, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec(
                "idx",
                [2, 3, 8],
                DataType.INT32,
                init_value=lambda: _rand_indices(0, 16, (2, 3, 8)),
            ),
            TensorSpec("output", [2, 3, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherRank3LastDimProgram

    def compute_expected(self, tensors, params=None):
        inp = tensors["inp"]
        idx = tensors["idx"].to(torch.int64)
        tensors["output"][:] = torch.gather(inp, dim=-1, index=idx)


class GatherRank3MiddleDimTestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_rank3_middle_dim"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [2, 8, 8], DataType.FP32, init_value=torch.randn),
            TensorSpec(
                "idx",
                [2, 3, 8],
                DataType.INT32,
                init_value=lambda: _rand_indices(0, 8, (2, 3, 8)),
            ),
            TensorSpec("output", [2, 3, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherRank3MiddleDimProgram

    def compute_expected(self, tensors, params=None):
        inp = tensors["inp"]
        idx = tensors["idx"].to(torch.int64)
        tensors["output"][:] = torch.gather(inp, dim=1, index=idx)


class GatherRank3NegFirstDimTestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_rank3_neg_first_dim"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [8, 2, 8], DataType.FP32, init_value=torch.randn),
            TensorSpec(
                "idx",
                [3, 2, 8],
                DataType.INT32,
                init_value=lambda: _rand_indices(0, 8, (3, 2, 8)),
            ),
            TensorSpec("output", [3, 2, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherRank3NegFirstDimProgram

    def compute_expected(self, tensors, params=None):
        inp = tensors["inp"]
        idx = tensors["idx"].to(torch.int64)
        # dim=-3 normalizes to dim=0 on rank-3
        tensors["output"][:] = torch.gather(inp, dim=0, index=idx)


# --- Tests ---


class TestGather:
    """Verify ``pl.tensor.gather`` against a torch reference for the
    generalized rank/dim contract introduced by issue #676."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_rank2_last_dim(self, test_runner, platform):
        result = test_runner.run(GatherRank2LastDimTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_rank2_smaller_leading(self, test_runner, platform):
        result = test_runner.run(GatherRank2SmallerLeadingTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_rank3_last_dim(self, test_runner, platform):
        result = test_runner.run(GatherRank3LastDimTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_rank3_middle_dim(self, test_runner, platform):
        result = test_runner.run(GatherRank3MiddleDimTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_rank3_neg_first_dim(self, test_runner, platform):
        result = test_runner.run(GatherRank3NegFirstDimTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
