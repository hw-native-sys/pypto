# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile.mscatter (pto.mscatter) — scatter-store to global memory.

Semantics: output_tensor[idx[i, j]] = src[i, j]

Test matrix:
  - Dtypes: FP32, FP16, INT32
  - Shapes: 8x32, 16x64
  - Index patterns: sequential, reversed, random permutation, strided, sparse

NOTE: pto.mscatter is only implemented on A5 (Ascend950).  A3 support is
pending — do NOT add A2A3 test variants until the PTOAS A3 lowering lands.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType

# =============================================================================
# Programs
# =============================================================================


@pl.program
class MscatterFP32_8x32Program:
    """Scatter-store FP32 [8, 32] tile into a 1D tensor."""

    @pl.function(type=pl.FunctionType.InCore)
    def mscatter_kernel(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.FP32],
        idx_tensor: pl.Tensor[[8, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[256], pl.FP32]],
    ) -> pl.Tensor[[256], pl.FP32]:
        src_tile: pl.Tile[[8, 32], pl.FP32] = pl.load(src_tensor, offsets=[0, 0], shapes=[8, 32])
        idx_tile: pl.Tile[[8, 32], pl.INT32] = pl.load(idx_tensor, offsets=[0, 0], shapes=[8, 32])
        out: pl.Tensor[[256], pl.FP32] = pl.mscatter(src_tile, idx_tile, output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.FP32],
        idx_tensor: pl.Tensor[[8, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[256], pl.FP32]],
    ) -> pl.Tensor[[256], pl.FP32]:
        output = self.mscatter_kernel(src_tensor, idx_tensor, output)
        return output


@pl.program
class MscatterFP32_8x32_LargeOutputProgram:
    """Scatter-store FP32 [8, 32] tile into a larger 1D tensor (512 elements).

    Tests sparse scatter: 256 elements written into 512-element output.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def mscatter_kernel(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.FP32],
        idx_tensor: pl.Tensor[[8, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[512], pl.FP32]],
    ) -> pl.Tensor[[512], pl.FP32]:
        src_tile: pl.Tile[[8, 32], pl.FP32] = pl.load(src_tensor, offsets=[0, 0], shapes=[8, 32])
        idx_tile: pl.Tile[[8, 32], pl.INT32] = pl.load(idx_tensor, offsets=[0, 0], shapes=[8, 32])
        out: pl.Tensor[[512], pl.FP32] = pl.mscatter(src_tile, idx_tile, output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.FP32],
        idx_tensor: pl.Tensor[[8, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[512], pl.FP32]],
    ) -> pl.Tensor[[512], pl.FP32]:
        output = self.mscatter_kernel(src_tensor, idx_tensor, output)
        return output


@pl.program
class MscatterFP16_8x32Program:
    """Scatter-store FP16 [8, 32] tile into a 1D tensor."""

    @pl.function(type=pl.FunctionType.InCore)
    def mscatter_kernel(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.FP16],
        idx_tensor: pl.Tensor[[8, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[256], pl.FP16]],
    ) -> pl.Tensor[[256], pl.FP16]:
        src_tile: pl.Tile[[8, 32], pl.FP16] = pl.load(src_tensor, offsets=[0, 0], shapes=[8, 32])
        idx_tile: pl.Tile[[8, 32], pl.INT32] = pl.load(idx_tensor, offsets=[0, 0], shapes=[8, 32])
        out: pl.Tensor[[256], pl.FP16] = pl.mscatter(src_tile, idx_tile, output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.FP16],
        idx_tensor: pl.Tensor[[8, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[256], pl.FP16]],
    ) -> pl.Tensor[[256], pl.FP16]:
        output = self.mscatter_kernel(src_tensor, idx_tensor, output)
        return output


@pl.program
class MscatterINT32_8x32Program:
    """Scatter-store INT32 [8, 32] tile into a 1D tensor."""

    @pl.function(type=pl.FunctionType.InCore)
    def mscatter_kernel(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.INT32],
        idx_tensor: pl.Tensor[[8, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[256], pl.INT32]],
    ) -> pl.Tensor[[256], pl.INT32]:
        src_tile: pl.Tile[[8, 32], pl.INT32] = pl.load(src_tensor, offsets=[0, 0], shapes=[8, 32])
        idx_tile: pl.Tile[[8, 32], pl.INT32] = pl.load(idx_tensor, offsets=[0, 0], shapes=[8, 32])
        out: pl.Tensor[[256], pl.INT32] = pl.mscatter(src_tile, idx_tile, output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src_tensor: pl.Tensor[[8, 32], pl.INT32],
        idx_tensor: pl.Tensor[[8, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[256], pl.INT32]],
    ) -> pl.Tensor[[256], pl.INT32]:
        output = self.mscatter_kernel(src_tensor, idx_tensor, output)
        return output


@pl.program
class MscatterFP32_16x64Program:
    """Scatter-store FP32 [16, 64] tile into a 1D tensor (larger shape)."""

    @pl.function(type=pl.FunctionType.InCore)
    def mscatter_kernel(
        self,
        src_tensor: pl.Tensor[[16, 64], pl.FP32],
        idx_tensor: pl.Tensor[[16, 64], pl.INT32],
        output: pl.Out[pl.Tensor[[1024], pl.FP32]],
    ) -> pl.Tensor[[1024], pl.FP32]:
        src_tile: pl.Tile[[16, 64], pl.FP32] = pl.load(src_tensor, offsets=[0, 0], shapes=[16, 64])
        idx_tile: pl.Tile[[16, 64], pl.INT32] = pl.load(idx_tensor, offsets=[0, 0], shapes=[16, 64])
        out: pl.Tensor[[1024], pl.FP32] = pl.mscatter(src_tile, idx_tile, output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src_tensor: pl.Tensor[[16, 64], pl.FP32],
        idx_tensor: pl.Tensor[[16, 64], pl.INT32],
        output: pl.Out[pl.Tensor[[1024], pl.FP32]],
    ) -> pl.Tensor[[1024], pl.FP32]:
        output = self.mscatter_kernel(src_tensor, idx_tensor, output)
        return output


@pl.program
class MscatterFP16_16x64Program:
    """Scatter-store FP16 [16, 64] tile into a 1D tensor (larger shape)."""

    @pl.function(type=pl.FunctionType.InCore)
    def mscatter_kernel(
        self,
        src_tensor: pl.Tensor[[16, 64], pl.FP16],
        idx_tensor: pl.Tensor[[16, 64], pl.INT32],
        output: pl.Out[pl.Tensor[[1024], pl.FP16]],
    ) -> pl.Tensor[[1024], pl.FP16]:
        src_tile: pl.Tile[[16, 64], pl.FP16] = pl.load(src_tensor, offsets=[0, 0], shapes=[16, 64])
        idx_tile: pl.Tile[[16, 64], pl.INT32] = pl.load(idx_tensor, offsets=[0, 0], shapes=[16, 64])
        out: pl.Tensor[[1024], pl.FP16] = pl.mscatter(src_tile, idx_tile, output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src_tensor: pl.Tensor[[16, 64], pl.FP16],
        idx_tensor: pl.Tensor[[16, 64], pl.INT32],
        output: pl.Out[pl.Tensor[[1024], pl.FP16]],
    ) -> pl.Tensor[[1024], pl.FP16]:
        output = self.mscatter_kernel(src_tensor, idx_tensor, output)
        return output


# =============================================================================
# Named init_value functions for golden_writer.
# Each function must be self-contained (golden_writer extracts source code
# into an isolated environment with no access to module-level references).
# =============================================================================


def _init_sequential_8x32() -> torch.Tensor:
    return torch.arange(8 * 32, dtype=torch.int32).reshape(8, 32)


def _init_reversed_8x32() -> torch.Tensor:
    return torch.arange(8 * 32 - 1, -1, -1, dtype=torch.int32).reshape(8, 32)


def _init_random_perm_8x32() -> torch.Tensor:
    return torch.randperm(8 * 32, dtype=torch.int32).reshape(8, 32)


def _init_strided_8x32() -> torch.Tensor:
    return (torch.arange(8 * 32, dtype=torch.int32) * 2).reshape(8, 32)


def _init_sequential_16x64() -> torch.Tensor:
    return torch.arange(16 * 64, dtype=torch.int32).reshape(16, 64)


def _init_random_perm_16x64() -> torch.Tensor:
    return torch.randperm(16 * 64, dtype=torch.int32).reshape(16, 64)


def _init_randint_8x32() -> torch.Tensor:
    return torch.randint(-1000, 1000, (8, 32), dtype=torch.int32)


# =============================================================================
# Test Cases — FP32 8x32
# =============================================================================


class MscatterFP32SeqTestCase(PTOTestCase):
    """FP32 8x32, sequential indices [0..255]."""

    __test__ = False

    def get_name(self) -> str:
        return "mscatter_fp32_8x32_seq"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [8, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx_tensor", [8, 32], DataType.INT32, init_value=_init_sequential_8x32),
            TensorSpec("output", [256], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MscatterFP32_8x32Program

    def compute_expected(self, tensors, params=None):
        out = torch.zeros(256, dtype=torch.float32)
        out[tensors["idx_tensor"].flatten().long()] = tensors["src_tensor"].flatten()
        tensors["output"][:] = out


class MscatterFP32RevTestCase(PTOTestCase):
    """FP32 8x32, reversed indices [255..0]."""

    __test__ = False

    def get_name(self) -> str:
        return "mscatter_fp32_8x32_rev"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [8, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx_tensor", [8, 32], DataType.INT32, init_value=_init_reversed_8x32),
            TensorSpec("output", [256], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MscatterFP32_8x32Program

    def compute_expected(self, tensors, params=None):
        out = torch.zeros(256, dtype=torch.float32)
        out[tensors["idx_tensor"].flatten().long()] = tensors["src_tensor"].flatten()
        tensors["output"][:] = out


class MscatterFP32RandPermTestCase(PTOTestCase):
    """FP32 8x32, random permutation of [0..255]."""

    __test__ = False

    def get_name(self) -> str:
        return "mscatter_fp32_8x32_rand_perm"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [8, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx_tensor", [8, 32], DataType.INT32, init_value=_init_random_perm_8x32),
            TensorSpec("output", [256], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MscatterFP32_8x32Program

    def compute_expected(self, tensors, params=None):
        out = torch.zeros(256, dtype=torch.float32)
        out[tensors["idx_tensor"].flatten().long()] = tensors["src_tensor"].flatten()
        tensors["output"][:] = out


class MscatterFP32StridedTestCase(PTOTestCase):
    """FP32 8x32, strided indices [0, 2, 4, ...] into 512-element output.

    Tests sparse scatter pattern: only even positions are written.
    """

    __test__ = False

    def get_name(self) -> str:
        return "mscatter_fp32_8x32_strided"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [8, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx_tensor", [8, 32], DataType.INT32, init_value=_init_strided_8x32),
            TensorSpec("output", [512], DataType.FP32, is_output=True, init_value=torch.zeros),
        ]

    def get_program(self) -> Any:
        return MscatterFP32_8x32_LargeOutputProgram

    def compute_expected(self, tensors, params=None):
        out = torch.zeros(512, dtype=torch.float32)
        out[tensors["idx_tensor"].flatten().long()] = tensors["src_tensor"].flatten()
        tensors["output"][:] = out


# =============================================================================
# Test Cases — FP16 8x32
# =============================================================================


class MscatterFP16SeqTestCase(PTOTestCase):
    """FP16 8x32, sequential indices."""

    __test__ = False

    def get_name(self) -> str:
        return "mscatter_fp16_8x32_seq"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [8, 32], DataType.FP16, init_value=torch.randn),
            TensorSpec("idx_tensor", [8, 32], DataType.INT32, init_value=_init_sequential_8x32),
            TensorSpec("output", [256], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return MscatterFP16_8x32Program

    def compute_expected(self, tensors, params=None):
        out = torch.zeros(256, dtype=torch.float16)
        out[tensors["idx_tensor"].flatten().long()] = tensors["src_tensor"].flatten().to(torch.float16)
        tensors["output"][:] = out


class MscatterFP16RandPermTestCase(PTOTestCase):
    """FP16 8x32, random permutation indices."""

    __test__ = False

    def get_name(self) -> str:
        return "mscatter_fp16_8x32_rand_perm"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [8, 32], DataType.FP16, init_value=torch.randn),
            TensorSpec("idx_tensor", [8, 32], DataType.INT32, init_value=_init_random_perm_8x32),
            TensorSpec("output", [256], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return MscatterFP16_8x32Program

    def compute_expected(self, tensors, params=None):
        out = torch.zeros(256, dtype=torch.float16)
        out[tensors["idx_tensor"].flatten().long()] = tensors["src_tensor"].flatten().to(torch.float16)
        tensors["output"][:] = out


# =============================================================================
# Test Cases — INT32 8x32
# =============================================================================


class MscatterINT32SeqTestCase(PTOTestCase):
    """INT32 8x32, sequential indices."""

    __test__ = False

    def get_name(self) -> str:
        return "mscatter_int32_8x32_seq"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [8, 32], DataType.INT32, init_value=_init_randint_8x32),
            TensorSpec("idx_tensor", [8, 32], DataType.INT32, init_value=_init_sequential_8x32),
            TensorSpec("output", [256], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MscatterINT32_8x32Program

    def compute_expected(self, tensors, params=None):
        out = torch.zeros(256, dtype=torch.int32)
        out[tensors["idx_tensor"].flatten().long()] = tensors["src_tensor"].flatten()
        tensors["output"][:] = out


class MscatterINT32RevTestCase(PTOTestCase):
    """INT32 8x32, reversed indices."""

    __test__ = False

    def get_name(self) -> str:
        return "mscatter_int32_8x32_rev"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [8, 32], DataType.INT32, init_value=_init_randint_8x32),
            TensorSpec("idx_tensor", [8, 32], DataType.INT32, init_value=_init_reversed_8x32),
            TensorSpec("output", [256], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MscatterINT32_8x32Program

    def compute_expected(self, tensors, params=None):
        out = torch.zeros(256, dtype=torch.int32)
        out[tensors["idx_tensor"].flatten().long()] = tensors["src_tensor"].flatten()
        tensors["output"][:] = out


# =============================================================================
# Test Cases — Larger shape 16x64
# =============================================================================


class MscatterFP32_16x64SeqTestCase(PTOTestCase):
    """FP32 16x64, sequential indices — tests larger tile shape."""

    __test__ = False

    def get_name(self) -> str:
        return "mscatter_fp32_16x64_seq"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [16, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx_tensor", [16, 64], DataType.INT32, init_value=_init_sequential_16x64),
            TensorSpec("output", [1024], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MscatterFP32_16x64Program

    def compute_expected(self, tensors, params=None):
        out = torch.zeros(1024, dtype=torch.float32)
        out[tensors["idx_tensor"].flatten().long()] = tensors["src_tensor"].flatten()
        tensors["output"][:] = out


class MscatterFP32_16x64RandPermTestCase(PTOTestCase):
    """FP32 16x64, random permutation — tests larger tile with non-trivial scatter."""

    __test__ = False

    def get_name(self) -> str:
        return "mscatter_fp32_16x64_rand_perm"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [16, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx_tensor", [16, 64], DataType.INT32, init_value=_init_random_perm_16x64),
            TensorSpec("output", [1024], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MscatterFP32_16x64Program

    def compute_expected(self, tensors, params=None):
        out = torch.zeros(1024, dtype=torch.float32)
        out[tensors["idx_tensor"].flatten().long()] = tensors["src_tensor"].flatten()
        tensors["output"][:] = out


class MscatterFP16_16x64RandPermTestCase(PTOTestCase):
    """FP16 16x64, random permutation — tests larger tile with FP16."""

    __test__ = False

    def get_name(self) -> str:
        return "mscatter_fp16_16x64_rand_perm"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src_tensor", [16, 64], DataType.FP16, init_value=torch.randn),
            TensorSpec("idx_tensor", [16, 64], DataType.INT32, init_value=_init_random_perm_16x64),
            TensorSpec("output", [1024], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return MscatterFP16_16x64Program

    def compute_expected(self, tensors, params=None):
        out = torch.zeros(1024, dtype=torch.float16)
        out[tensors["idx_tensor"].flatten().long()] = tensors["src_tensor"].flatten().to(torch.float16)
        tensors["output"][:] = out


# =============================================================================
# pytest test functions
# =============================================================================


@pytest.mark.a5
class TestMscatterFP32_8x32:
    """FP32 [8, 32] mscatter with various index patterns."""

    def test_sequential(self, test_runner):
        """Sequential indices [0..255] — identity scatter."""
        result = test_runner.run(MscatterFP32SeqTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_reversed(self, test_runner):
        """Reversed indices [255..0] — reverse element order in output."""
        result = test_runner.run(MscatterFP32RevTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_random_permutation(self, test_runner):
        """Random permutation of [0..255] — each position written exactly once."""
        result = test_runner.run(MscatterFP32RandPermTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_strided(self, test_runner):
        """Strided indices [0, 2, 4, ...] into 512-element output — sparse scatter."""
        result = test_runner.run(MscatterFP32StridedTestCase())
        assert result.passed, f"Test failed: {result.error}"


@pytest.mark.a5
class TestMscatterFP16_8x32:
    """FP16 [8, 32] mscatter tests."""

    def test_sequential(self, test_runner):
        """FP16 sequential indices."""
        result = test_runner.run(MscatterFP16SeqTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_random_permutation(self, test_runner):
        """FP16 random permutation."""
        result = test_runner.run(MscatterFP16RandPermTestCase())
        assert result.passed, f"Test failed: {result.error}"


@pytest.mark.a5
class TestMscatterINT32_8x32:
    """INT32 [8, 32] mscatter tests."""

    def test_sequential(self, test_runner):
        """INT32 sequential indices."""
        result = test_runner.run(MscatterINT32SeqTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_reversed(self, test_runner):
        """INT32 reversed indices."""
        result = test_runner.run(MscatterINT32RevTestCase())
        assert result.passed, f"Test failed: {result.error}"


@pytest.mark.a5
class TestMscatterLargeShape:
    """Larger tile shape [16, 64] mscatter tests."""

    def test_fp32_sequential(self, test_runner):
        """FP32 16x64 sequential indices."""
        result = test_runner.run(MscatterFP32_16x64SeqTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_fp32_random_permutation(self, test_runner):
        """FP32 16x64 random permutation."""
        result = test_runner.run(MscatterFP32_16x64RandPermTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_fp16_random_permutation(self, test_runner):
        """FP16 16x64 random permutation."""
        result = test_runner.run(MscatterFP16_16x64RandPermTestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
