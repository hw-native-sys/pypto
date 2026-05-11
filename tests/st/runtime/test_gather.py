# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end tests for ``pl.tensor.gather`` — all three forms.

The tensor layer exposes a single unified ``pl.tensor.gather`` that dispatches
to one of three tile-level ops based on the kwargs passed:

Index form  (``dim`` + ``index``)                       → ``tile.gather``
Mask form   (``mask_pattern=<int>``)                    → ``tile.gather_mask``
Compare form (``kvalue`` + ``cmp_mode`` + ``out_cols``) → ``tile.gather_compare``

Index form (torch-style semantics, validated against ``torch.gather``):

1. Rank-2 + dim=-1 (baseline / regression).
2. Rank-2 + dim=-1 with ``index.shape[0] < input.shape[0]`` (smaller index leading).
3. Rank-3 + dim=-1 (collapses leading dims via ``tile.reshape``).
4. Rank-3 + dim=1 (middle axis — flat-index gather).
5. Rank-3 + dim=-3 (negative-dim normalization on the first axis).

Mask form (hardware mask-pattern column selection):

6. P0101 — select even columns of each row.
7. P1010 + ``output_dtype=UINT32`` — select odd columns and bit-reinterpret.

Compare form (per-row threshold compare; returns ``(dst, cdst)``):

8. ``cmp_mode='eq'`` with INT32 src/kvalue.
9. ``cmp_mode='gt'`` with FP16 src/kvalue.
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


def _make_gather_mask_src_8x16() -> torch.Tensor:
    """Deterministic ``[8, 16]`` FP32 source for mask gather tests.

    Distinct values per element so that even/odd column selection produces
    a unique, easily verified output.
    """
    return (torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16) - 64.0) / 8.0


def _make_gather_compare_src_eq() -> torch.Tensor:
    """``[16, 64]`` INT32 with exactly 8 ``eq`` matches per row.

    For row ``r``::

        src[r, j] = 100             if j % 8 == 0   # 8 matches
                    101             otherwise

    With ``kvalue = 100`` and ``cmp_mode='eq'``, matches occur at
    ``j = 0, 8, 16, 24, 32, 40, 48, 56`` — exactly ``out_cols=8`` per row.
    """
    out = torch.zeros(16, 64, dtype=torch.int32)
    out[:, :] = 101
    out[:, 0::8] = 100
    return out


def _make_gather_compare_kvalue_eq() -> torch.Tensor:
    """``[1]`` INT32 scalar carrier: ``kvalue = 100``."""
    return torch.tensor([100], dtype=torch.int32)


def _make_gather_compare_src_gt() -> torch.Tensor:
    """``[16, 64]`` FP16 with exactly 8 ``gt`` matches per row.

    All zeros except ``src[r, 56:64] = 100``. With ``kvalue = 50`` and
    ``cmp_mode='gt'``, matches occur at ``j = 56..63`` — exactly ``out_cols=8``.
    """
    out = torch.zeros(16, 64, dtype=torch.float16)
    out[:, 56:64] = 100
    return out


def _make_gather_compare_kvalue_gt() -> torch.Tensor:
    """``[1]`` FP16 scalar carrier: ``kvalue = 50``."""
    return torch.tensor([50], dtype=torch.float16)


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


@pl.program
class GatherMaskP0101Program:
    """Mask-form gather (P0101): select even-position columns of each row.

    ``src [8, 16] FP32`` → ``output [8, 8] FP32`` (cols shrink by 2).
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[8, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
    ) -> pl.Tensor[[8, 8], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.gather(inp, mask_pattern=pl.tile.MaskPattern.P0101)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class GatherMaskOutputDtypeProgram:
    """Mask-form gather (P1010) with ``output_dtype=UINT32``.

    Selects odd-position columns of each row, then bit-reinterprets the FP32
    payload as UINT32 (same bit width) — the same pattern sort32 uses to
    extract packed index bits stored in float slots.

    ``src [8, 16] FP32`` → ``output [8, 8] UINT32``.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[8, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[8, 8], pl.UINT32]],
    ) -> pl.Tensor[[8, 8], pl.UINT32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.gather(inp, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.UINT32)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class GatherCompareEqProgram:
    """Compare-form gather (``cmp_mode='eq'``): scalar equality match.

    For each row ``r``, scan ``src[r, :]`` against the scalar ``kvalue``.
    Indices where ``src[r, j] == kvalue`` are written to ``dst`` (up to
    ``out_cols``), and the count of matches is written to ``cdst``.

    ``src       [16, 64] INT32``
    ``kv_buf    [1]      INT32`` — carries the scalar kvalue
    →
    ``dst    [16, 8]  INT32``  — gathered indices
    ``cdst   [1, 16]  INT32``  — per-row match count
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        src: pl.Tensor[[16, 64], pl.INT32],
        kv_buf: pl.Tensor[[1], pl.INT32],
        out_dst: pl.Out[pl.Tensor[[16, 8], pl.INT32]],
        out_cdst: pl.Out[pl.Tensor[[1, 16], pl.INT32]],
    ) -> tuple[pl.Tensor[[16, 8], pl.INT32], pl.Tensor[[1, 16], pl.INT32]]:
        with pl.at(level=pl.Level.CORE_GROUP):
            kvalue: pl.Scalar[pl.INT32] = pl.tensor.read(kv_buf, [0])
            dst, cdst = pl.tensor.gather(src, kvalue=kvalue, cmp_mode="eq", out_cols=8)
            out_dst = pl.assemble(out_dst, dst, [0, 0])
            out_cdst = pl.assemble(out_cdst, cdst, [0, 0])
        return out_dst, out_cdst


@pl.program
class GatherCompareGtFP16Program:
    """Compare-form gather (``cmp_mode='gt'``) with FP16 src/kvalue.

    ``src       [16, 64] FP16``
    ``kv_buf    [1]      FP16`` — carries the scalar kvalue
    →
    ``dst    [16, 8]  INT32``  — gathered indices
    ``cdst   [1, 16]  INT32``  — per-row match count
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        src: pl.Tensor[[16, 64], pl.FP16],
        kv_buf: pl.Tensor[[1], pl.FP16],
        out_dst: pl.Out[pl.Tensor[[16, 8], pl.INT32]],
        out_cdst: pl.Out[pl.Tensor[[1, 16], pl.INT32]],
    ) -> tuple[pl.Tensor[[16, 8], pl.INT32], pl.Tensor[[1, 16], pl.INT32]]:
        with pl.at(level=pl.Level.CORE_GROUP):
            kvalue: pl.Scalar[pl.FP16] = pl.tensor.read(kv_buf, [0])
            dst, cdst = pl.tensor.gather(src, kvalue=kvalue, cmp_mode="gt", out_cols=8, count_dtype=pl.INT32)
            out_dst = pl.assemble(out_dst, dst, [0, 0])
            out_cdst = pl.assemble(out_cdst, cdst, [0, 0])
        return out_dst, out_cdst


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


class GatherMaskP0101TestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_mask_p0101"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [8, 16], DataType.FP32, init_value=_make_gather_mask_src_8x16),
            TensorSpec("output", [8, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherMaskP0101Program

    def compute_expected(self, tensors, params=None):
        # P0101 selects positions 0, 2, 4, ..., 14 of each row.
        tensors["output"][:] = tensors["inp"][:, 0::2]


class GatherMaskOutputDtypeTestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_mask_output_dtype_uint32"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [8, 16], DataType.FP32, init_value=_make_gather_mask_src_8x16),
            TensorSpec("output", [8, 8], DataType.UINT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherMaskOutputDtypeProgram

    def compute_expected(self, tensors, params=None):
        # P1010 selects positions 1, 3, 5, ..., 15 of each row.
        # output_dtype=UINT32 → FP32 bits are reinterpreted (no value conversion).
        odd_cols_fp32 = tensors["inp"][:, 1::2].contiguous()
        tensors["output"][:] = odd_cols_fp32.view(torch.int32)


class GatherCompareEqTestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_compare_eq"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src", [16, 64], DataType.INT32, init_value=_make_gather_compare_src_eq),
            TensorSpec(
                "kv_buf",
                [1],
                DataType.INT32,
                init_value=_make_gather_compare_kvalue_eq,
            ),
            TensorSpec("out_dst", [16, 8], DataType.INT32, is_output=True),
            TensorSpec("out_cdst", [1, 16], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherCompareEqProgram

    def compute_expected(self, tensors, params=None):
        # Data is constructed so each row has exactly 8 matches at
        # j = 0, 8, 16, 24, 32, 40, 48, 56 — all dst positions are well-defined.
        match_positions = torch.tensor([0, 8, 16, 24, 32, 40, 48, 56], dtype=torch.int32)
        tensors["out_dst"][:] = match_positions.unsqueeze(0).expand(16, -1).contiguous()
        tensors["out_cdst"][:] = 8


class GatherCompareGtFP16TestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_compare_gt_fp16"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src", [16, 64], DataType.FP16, init_value=_make_gather_compare_src_gt),
            TensorSpec(
                "kv_buf",
                [1],
                DataType.FP16,
                init_value=_make_gather_compare_kvalue_gt,
            ),
            TensorSpec("out_dst", [16, 8], DataType.INT32, is_output=True),
            TensorSpec("out_cdst", [1, 16], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherCompareGtFP16Program

    def compute_expected(self, tensors, params=None):
        # Data is constructed so each row has exactly 8 src entries > kvalue
        # at j = 56..63 — all dst positions are well-defined.
        match_positions = torch.arange(56, 64, dtype=torch.int32)
        tensors["out_dst"][:] = match_positions.unsqueeze(0).expand(16, -1).contiguous()
        tensors["out_cdst"][:] = 8


# --- Tests ---


class TestGatherIndex:
    # --- Index form ---

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


class TestGatherMask:
    # --- Mask form ---

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_mask_p0101(self, test_runner, platform):
        result = test_runner.run(GatherMaskP0101TestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_mask_output_dtype_uint32(self, test_runner, platform):
        result = test_runner.run(GatherMaskOutputDtypeTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


@pytest.mark.skip(reason="PTOAS handling for gather compare form is broken; pending upstream fix")
class TestGatherCompare:
    # --- Compare form ---

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_compare_eq(self, test_runner, platform):
        result = test_runner.run(GatherCompareEqTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_compare_gt_fp16(self, test_runner, platform):
        result = test_runner.run(GatherCompareGtFP16TestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
