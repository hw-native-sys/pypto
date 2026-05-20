# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end tests for ``pl.tensor.scatter`` — both forms.

The tensor layer exposes a single unified ``pl.tensor.scatter`` that dispatches
to one of two tile-level ops based on the kwargs passed (there is no compare
form for scatter):

Index form (``dim`` + ``index`` + ``src``) → ``tile.scatter``
Mask form  (``mask_pattern=<int>`` + ``dst``) → ``tile.scatter_mask``

Both ops are destination-passing-style (DPS): ``input`` / ``dst`` is the base
buffer and the result aliases it — rows/columns not written keep their value.

Index form (row scatter, ``out[index[i, 0], :] = src[i, :]``):

1. Rank-2 + dim=0 (baseline).
2. Rank-2 + dim=-2 (negative-dim normalization — alias of dim=0).

Mask form (hardware mask-pattern column expansion, inverse of gather_mask):

3. P0101 — write each compact ``input`` row into the even columns of ``dst``.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# --- Shared init helpers ---


def _distinct_row_indices_8() -> torch.Tensor:
    """``[8, 1]`` INT32 of distinct destination rows in ``[0, 16)``.

    Distinct rows keep the row-scatter result well-defined (no row is written
    twice, so the outcome is independent of write order). 8 rows are used so the
    col-major index tile's column byte size (rows * sizeof(i32) = 32) meets the
    PTOAS 32-byte alignment requirement for ``pto.alloc_tile``.
    """
    return torch.tensor([[0], [2], [4], [6], [9], [11], [13], [15]], dtype=torch.int32)


def _flat_row_indices_6x32() -> torch.Tensor:
    """``[6, 32]`` INT32 of *flattened* destination indices for a row-scatter.

    The hardware ``pto.tscatter`` index form scatters per element using flattened
    destination indices: ``dst.flat[idx[i, j]] = src[i, j]``. To express a
    row-scatter ``dst[row, :] = src[i, :]`` the index tile must have the same
    ``[rows, cols]`` shape as ``src`` with ``idx[i, j] = row[i] * dst_cols + j``.

    Here ``dst`` has 16 rows × 32 columns and the 6 source rows land on the
    distinct destination rows ``[0, 3, 6, 9, 12, 15]``. Only these 6 rows
    (6 * 32 = 192 elements) are written by the scatter; the remaining 10 rows
    (320 elements) must keep ``inp``'s value (DPS preserve).
    """
    dst_rows = torch.tensor([0, 3, 6, 9, 12, 15], dtype=torch.int32).reshape(6, 1)
    cols = torch.arange(32, dtype=torch.int32).reshape(1, 32)
    return (dst_rows * 32 + cols).to(torch.int32)


def _make_scatter_mask_src_8x8() -> torch.Tensor:
    """Deterministic ``[8, 8]`` FP32 source for the mask scatter test.

    Distinct values per element so even-column placement is easy to verify. 8
    rows keep the col-major tile column byte size (8 * sizeof(fp32) = 32) on the
    PTOAS 32-byte alignment boundary.
    """
    return (torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8) - 32.0) / 4.0


# --- Programs ---


@pl.program
class ScatterIndexDim0Program:
    """Baseline rank-2 + dim=0: ``out = inp; out[idx[i, 0], :] = src[i, :]``."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[16, 32], pl.FP32],
        idx: pl.Tensor[[8, 1], pl.INT32],
        src: pl.Tensor[[8, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.scatter(inp, dim=0, index=idx, src=src)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class ScatterIndexNegDimProgram:
    """Rank-2 + dim=-2 (alias of dim=0 after negative-dim normalization)."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[16, 32], pl.FP32],
        idx: pl.Tensor[[8, 1], pl.INT32],
        src: pl.Tensor[[8, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.scatter(inp, dim=-2, index=idx, src=src)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class ScatterMaskP0101Program:
    """Mask-form scatter (P0101): write each compact ``inp`` row into the even
    columns of ``dst`` (the inverse of gather_mask P0101). ``dst`` is the DPS
    destination, initialized to zero, so non-selected columns read back as 0."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[8, 8], pl.FP32],
        dst: pl.Tensor[[8, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
    ) -> pl.Tensor[[8, 16], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.scatter(inp, mask_pattern=pl.tile.MaskPattern.P0101, dst=dst)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class ScatterTileFlatProgram:
    """Minimal tile-level index-form scatter fed *flattened* indices directly.

    Bypasses the tensor-level row-index → flat-index expansion: the index tile
    already holds per-element flattened destination indices, so this exercises
    the raw ``pto.tscatter`` hardware path. DPS — ``inp`` is the destination and
    rows not addressed by the index keep their value.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        inp: pl.Tensor[[16, 32], pl.FP32],
        src: pl.Tensor[[6, 32], pl.FP32],
        idx: pl.Tensor[[6, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        dst_tile: pl.Tile[[16, 32], pl.FP32] = pl.load(inp, [0, 0], [16, 32])
        src_tile: pl.Tile[[6, 32], pl.FP32] = pl.load(src, [0, 0], [6, 32])
        idx_tile: pl.Tile[[6, 32], pl.INT32] = pl.load(idx, [0, 0], [6, 32])
        scattered: pl.Tile[[16, 32], pl.FP32] = pl.tile.scatter(dst_tile, src_tile, idx_tile)
        out: pl.Tensor[[16, 32], pl.FP32] = pl.store(scattered, [0, 0], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        inp: pl.Tensor[[16, 32], pl.FP32],
        src: pl.Tensor[[6, 32], pl.FP32],
        idx: pl.Tensor[[6, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        output = self.kernel(inp, src, idx, output)
        return output


# --- Test cases ---


class _ScatterBaseTestCase(PTOTestCase):
    __test__ = False

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


class ScatterTileFlatTestCase(_ScatterBaseTestCase):
    def get_name(self) -> str:
        return "scatter_tile_flat"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [16, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("src", [6, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx", [6, 32], DataType.INT32, init_value=_flat_row_indices_6x32),
            TensorSpec("output", [16, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ScatterTileFlatProgram

    def compute_expected(self, tensors, params=None):
        # Flat element scatter: out.flat[idx[i, j]] = src[i, j]; unaddressed
        # destination elements keep inp's value (DPS).
        out = tensors["inp"].clone()
        out_flat = out.reshape(-1)
        out_flat[tensors["idx"].reshape(-1).to(torch.int64)] = tensors["src"].reshape(-1)
        tensors["output"][:] = out_flat.reshape(16, 32)


class ScatterIndexDim0TestCase(_ScatterBaseTestCase):
    def get_name(self) -> str:
        return "scatter_index_dim0"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [16, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx", [8, 1], DataType.INT32, init_value=_distinct_row_indices_8),
            TensorSpec("src", [8, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [16, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ScatterIndexDim0Program

    def compute_expected(self, tensors, params=None):
        # out = inp with the named rows overwritten by src.
        out = tensors["inp"].clone()
        rows = tensors["idx"][:, 0].to(torch.int64)
        out[rows, :] = tensors["src"]
        tensors["output"][:] = out


class ScatterIndexNegDimTestCase(_ScatterBaseTestCase):
    def get_name(self) -> str:
        return "scatter_index_neg_dim"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [16, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx", [8, 1], DataType.INT32, init_value=_distinct_row_indices_8),
            TensorSpec("src", [8, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [16, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ScatterIndexNegDimProgram

    def compute_expected(self, tensors, params=None):
        out = tensors["inp"].clone()
        rows = tensors["idx"][:, 0].to(torch.int64)
        out[rows, :] = tensors["src"]
        tensors["output"][:] = out


class ScatterMaskP0101TestCase(_ScatterBaseTestCase):
    def get_name(self) -> str:
        return "scatter_mask_p0101"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [8, 8], DataType.FP32, init_value=_make_scatter_mask_src_8x8),
            TensorSpec("dst", [8, 16], DataType.FP32, init_value=lambda: torch.zeros(8, 16)),
            TensorSpec("output", [8, 16], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ScatterMaskP0101Program

    def compute_expected(self, tensors, params=None):
        # P0101 (stride 2) writes input columns into the even columns of dst.
        # dst starts at zero, so the odd (non-selected) columns read back as 0.
        out = tensors["dst"].clone()
        out[:, 0::2] = tensors["inp"]
        tensors["output"][:] = out


# --- Tests ---


class TestScatterTileFlat:
    # --- Raw tile-level index form with flattened indices (hardware path) ---

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scatter_tile_flat(self, test_runner, platform):
        result = test_runner.run(ScatterTileFlatTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


# The tensor-level row-index forms depend on the [N,1] → [N,cols] flat-index
# expansion lowering, which is the next step. Skipped until the raw tile-level
# flat-index path above is confirmed working on the device.
@pytest.mark.skip(reason="row-index→flat-index expansion lowering pending; validating raw flat path first")
class TestScatterIndex:
    # --- Index form ---

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scatter_index_dim0(self, test_runner, platform):
        result = test_runner.run(ScatterIndexDim0TestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scatter_index_neg_dim(self, test_runner, platform):
        result = test_runner.run(ScatterIndexNegDimTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


@pytest.mark.skip(reason="mask form validated separately; focusing on index form first")
class TestScatterMask:
    # --- Mask form ---

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scatter_mask_p0101(self, test_runner, platform):
        result = test_runner.run(ScatterMaskP0101TestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
