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
6. Rank-2 + dim=-1 on a local ``TileType`` input — regression for #1622: the
   per-row ``tile.slice`` must be view-only so the source tile is not mutated.

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
class GatherRank2LastDimTopLevelProgram:
    """Same as ``GatherRank2LastDimProgram`` but uses the promoted top-level
    ``pl.gather`` alias instead of ``pl.tensor.gather``."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[4, 16], pl.FP32],
        idx: pl.Tensor[[4, 8], pl.INT32],
        output: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.gather(inp, dim=-1, index=idx)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class GatherA5INT16IndexProgram:
    """A5 (Ascend950) index-form gather with INT16 indices.

    ``inp [4, 16] FP16``, ``idx [4, 16] INT16`` → ``output [4, 16] FP16``.
    INT16 indices are native to A5 but rejected by A2/A3 (the PTOAS verifier only
    permits i32 indices there), so this case is pinned to A5. It exercises the
    ``tile.gather`` type deduction — INT16 indices (valid only with a 16-bit src)
    plus the A5 index form's unconstrained ``tmp`` — end-to-end on hardware.

    FP16 src + INT16 indices selects the A5 ``TGather_b16`` path (2-byte src,
    2-byte indices), the hardware-supported INT16-index combination — mirroring
    the scatter INT16 tests. INT16 indices with a 32-bit src (FP32/INT32) are
    rejected outright by the ``tile.gather`` deducer: the only reachable form
    would be ``TGather_b32``, which reinterprets the indices as u32 and is not
    INT16-safe in this pto-isa revision.

    idx last-dim is 16 (16×2=32 bytes) to satisfy the hardware tile column
    alignment requirement (Cols * sizeof(dtype) % 32 == 0) for INT16.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[4, 16], pl.FP16],
        idx: pl.Tensor[[4, 16], pl.INT16],
        output: pl.Out[pl.Tensor[[4, 16], pl.FP16]],
    ) -> pl.Tensor[[4, 16], pl.FP16]:
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
class GatherRank2ExpandProgram:
    """Rank-2 expand gather: ``index`` cols (64) > ``input`` cols (32).

    Mirrors the DeepSeek rope cos/sin interleave pattern (column expand via
    ``j >> 1`` indices). On A5 this exercises the full-tile flat-index path.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[4, 32], pl.FP32],
        idx: pl.Tensor[[4, 64], pl.INT32],
        output: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
    ) -> pl.Tensor[[4, 64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.gather(inp, dim=-1, index=idx)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class GatherRank2SwapProgram:
    """Rank-2 swap gather: adjacent-pair swap via ``j ^ 1`` indices.

    Mirrors the DeepSeek rope swap pattern on the last dim.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[4, 64], pl.FP32],
        idx: pl.Tensor[[4, 64], pl.INT32],
        output: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
    ) -> pl.Tensor[[4, 64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.gather(inp, dim=-1, index=idx)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class GatherRank2Swap16RowProgram:
    """Rank-2 swap gather with 16 source rows (j^1 indices).

    16 rows spans two A5 FP32 vector boxes, so this exercises the A5 full-tile
    flat-index gather beyond the one-box (<=8-row) case that the 4-row variant
    covers. Regression for the A5 ``pto.tgather`` >8-row index-tile corruption.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[16, 64], pl.FP32],
        idx: pl.Tensor[[16, 64], pl.INT32],
        output: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
    ) -> pl.Tensor[[16, 64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.gather(inp, dim=-1, index=idx)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class GatherRank2Swap16RowStridedProgram:
    """Rank-2 swap gather over a STRIDED column slice of a wider tile.

    Mirrors DeepSeek sparse_attn inverse-RoPE exactly: the source is a [16, 128]
    tile whose columns [64:128] (ROPE_DIM=64) are gathered, so the gather source
    is a non-contiguous column slice with row stride 128, not 64. With a [16, 64]
    swap index this is the shape that corrupted on A5. The contiguous [16, 64]
    variant (GatherRank2Swap16RowProgram) does NOT reproduce it.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[16, 128], pl.FP32],
        idx: pl.Tensor[[16, 64], pl.INT32],
        output: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
    ) -> pl.Tensor[[16, 64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            local = pl.create_tensor([16, 128], dtype=pl.FP32)
            local = pl.assemble(local, inp, [0, 0])
            rope = local[:, 64:128]
            out = pl.tensor.gather(rope, dim=-1, index=idx)
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
class GatherTileInputSourceUnchangedProgram:
    """Regression for #1622: ``pl.tensor.gather`` on a local ``TileType`` input
    must not mutate the source tile.

    Builds a local tensor from ``src`` via ``pl.create_tensor`` + ``pl.assemble``
    (so the upstream conversion lowers it to a ``TileType`` before
    ``ConvertTensorToTileOps`` sees the gather), runs ``pl.tensor.gather`` on
    it, then echoes the local tensor back out as ``src_echo``. With the pre-fix
    lowering, the per-row ``tile.slice`` was emitted with an explicit
    ``valid_shape`` equal to the slice shape, which triggered a materializing
    ``pto.tmov`` whose destination aliased the source allocation — corrupting
    row 0 of the source to the last-materialized row. The fixed (view-only)
    lowering leaves ``src_echo`` bit-identical to ``src``.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        src: pl.Tensor[[16, 64], pl.FP32],
        idx: pl.Tensor[[16, 8], pl.INT32],
        src_echo: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        gathered: pl.Out[pl.Tensor[[16, 8], pl.FP32]],
    ) -> tuple[pl.Tensor[[16, 64], pl.FP32], pl.Tensor[[16, 8], pl.FP32]]:
        with pl.at(level=pl.Level.CORE_GROUP):
            local = pl.create_tensor([16, 64], dtype=pl.FP32)
            local = pl.assemble(local, src, [0, 0])
            g = pl.tensor.gather(local, dim=-1, index=idx)
            gathered = pl.assemble(gathered, g, [0, 0])
            src_echo = pl.assemble(src_echo, local, [0, 0])
        return src_echo, gathered


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
class GatherMaskP0101TopLevelProgram:
    """Same as ``GatherMaskP0101Program`` but uses the promoted top-level
    ``pl.gather`` alias (mask form) instead of ``pl.tensor.gather``."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[8, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
    ) -> pl.Tensor[[8, 8], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.gather(inp, mask_pattern=pl.tile.MaskPattern.P0101)
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


class GatherRank2LastDimTopLevelTestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_rank2_last_dim_toplevel"

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
        return GatherRank2LastDimTopLevelProgram

    def compute_expected(self, tensors, params=None):
        # torch.gather semantics: out[b, k] = inp[b, idx[b, k]]
        inp = tensors["inp"]
        idx = tensors["idx"].to(torch.int64)
        tensors["output"][:] = torch.gather(inp, dim=-1, index=idx)


class GatherA5INT16IndexTestCase(_GatherBaseTestCase):
    """A5 index-form gather with INT16 indices (A5-native; A2/A3 rejects it)."""

    def get_name(self) -> str:
        return "gather_a5_int16_index"

    def get_backend_type(self) -> BackendType:
        # INT16 indices are an A5 feature. Pin the backend so the global
        # ``set_backend_type`` matches the parametrized ``a5`` platform — the
        # base class otherwise defaults to Ascend910B.
        return BackendType.Ascend950

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "inp", [4, 16], DataType.FP16, init_value=lambda: torch.randn(4, 16).to(torch.float16)
            ),
            TensorSpec(
                "idx",
                [4, 16],
                DataType.INT16,
                init_value=lambda: torch.randint(0, 16, (4, 16), dtype=torch.int16),
            ),
            TensorSpec("output", [4, 16], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherA5INT16IndexProgram

    def compute_expected(self, tensors, params=None):
        # torch.gather semantics: out[b, k] = inp[b, idx[b, k]] (exact in FP16:
        # pure indexing, no arithmetic, so it matches the device bit-for-bit).
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


def _expand_indices_4x64() -> torch.Tensor:
    """Per-row interleave indices: ``idx[r, j] = j >> 1`` (32 → 64 expand)."""
    j = torch.arange(64, dtype=torch.int32)
    return (j >> 1).unsqueeze(0).expand(4, -1).contiguous()


def _swap_indices_4x64() -> torch.Tensor:
    """Per-row adjacent swap: ``idx[r, j] = j ^ 1``."""
    j = torch.arange(64, dtype=torch.int32)
    return (j ^ 1).unsqueeze(0).expand(4, -1).contiguous()


def _swap_indices_16x64() -> torch.Tensor:
    """Per-row adjacent swap over 16 rows: ``idx[r, j] = j ^ 1``."""
    j = torch.arange(64, dtype=torch.int32)
    return (j ^ 1).unsqueeze(0).expand(16, -1).contiguous()


class GatherRank2ExpandTestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_rank2_expand"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [4, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx", [4, 64], DataType.INT32, init_value=_expand_indices_4x64),
            TensorSpec("output", [4, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherRank2ExpandProgram

    def compute_expected(self, tensors, params=None):
        inp = tensors["inp"]
        idx = tensors["idx"].to(torch.int64)
        tensors["output"][:] = torch.gather(inp, dim=-1, index=idx)


class GatherRank2SwapTestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_rank2_swap"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [4, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx", [4, 64], DataType.INT32, init_value=_swap_indices_4x64),
            TensorSpec("output", [4, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherRank2SwapProgram

    def compute_expected(self, tensors, params=None):
        inp = tensors["inp"]
        idx = tensors["idx"].to(torch.int64)
        tensors["output"][:] = torch.gather(inp, dim=-1, index=idx)


class GatherRank2Swap16RowTestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_rank2_swap_16row"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [16, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx", [16, 64], DataType.INT32, init_value=_swap_indices_16x64),
            TensorSpec("output", [16, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherRank2Swap16RowProgram

    def compute_expected(self, tensors, params=None):
        inp = tensors["inp"]
        idx = tensors["idx"].to(torch.int64)
        tensors["output"][:] = torch.gather(inp, dim=-1, index=idx)


class GatherRank2Swap16RowStridedTestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_rank2_swap_16row_strided"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [16, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("idx", [16, 64], DataType.INT32, init_value=_swap_indices_16x64),
            TensorSpec("output", [16, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherRank2Swap16RowStridedProgram

    def compute_expected(self, tensors, params=None):
        rope = tensors["inp"][:, 64:128]
        idx = tensors["idx"].to(torch.int64)
        tensors["output"][:] = torch.gather(rope, dim=-1, index=idx)


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


class GatherTileInputSourceUnchangedTestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_tile_input_source_unchanged"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src", [16, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec(
                "idx",
                [16, 8],
                DataType.INT32,
                init_value=lambda: _rand_indices(0, 64, (16, 8)),
            ),
            TensorSpec("src_echo", [16, 64], DataType.FP32, is_output=True),
            TensorSpec("gathered", [16, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherTileInputSourceUnchangedProgram

    def compute_expected(self, tensors, params=None):
        src = tensors["src"]
        idx = tensors["idx"].to(torch.int64)
        tensors["gathered"][:] = torch.gather(src, dim=-1, index=idx)
        # src_echo must remain bit-identical to src: gather is read-only on
        # its source. Pre-fix, the local tile's row 0 was overwritten by the
        # last-materialized row, so this assertion would fail.
        tensors["src_echo"][:] = src


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


class GatherMaskP0101TopLevelTestCase(_GatherBaseTestCase):
    def get_name(self) -> str:
        return "gather_mask_p0101_toplevel"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("inp", [8, 16], DataType.FP32, init_value=_make_gather_mask_src_8x16),
            TensorSpec("output", [8, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GatherMaskP0101TopLevelProgram

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
    def test_gather_rank2_last_dim_toplevel(self, test_runner, platform):
        """Top-level pl.gather alias (index form) matches pl.tensor.gather."""
        result = test_runner.run(GatherRank2LastDimTopLevelTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_rank2_smaller_leading(self, test_runner, platform):
        result = test_runner.run(GatherRank2SmallerLeadingTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a5", "a5sim")
    @pytest.mark.parametrize("platform", [pytest.param("a5sim", id="a5sim"), pytest.param("a5", id="a5")])
    def test_gather_rank2_expand(self, test_runner, platform):
        """A5-focused expand gather (K>S1) — rope-style interleave indices."""
        result = test_runner.run(GatherRank2ExpandTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a5", "a5sim")
    @pytest.mark.parametrize("platform", [pytest.param("a5sim", id="a5sim"), pytest.param("a5", id="a5")])
    def test_gather_rank2_swap(self, test_runner, platform):
        """A5-focused swap gather (j^1) — rope-style adjacent swap."""
        result = test_runner.run(GatherRank2SwapTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a5", "a5sim")
    @pytest.mark.parametrize("platform", [pytest.param("a5sim", id="a5sim"), pytest.param("a5", id="a5")])
    def test_gather_rank2_swap_16row(self, test_runner, platform):
        """A5 swap gather over 16 rows (>8) — regression for tgather >8-row corruption."""
        result = test_runner.run(GatherRank2Swap16RowTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a5", "a5sim")
    @pytest.mark.parametrize("platform", [pytest.param("a5sim", id="a5sim"), pytest.param("a5", id="a5")])
    def test_gather_rank2_swap_16row_strided(self, test_runner, platform):
        """A5 swap gather over a strided column slice — mirrors sparse_attn inverse-RoPE."""
        result = test_runner.run(GatherRank2Swap16RowStridedTestCase(platform=platform))
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

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_tile_input_source_unchanged(self, test_runner, platform):
        """Regression for #1622: gather on a local TileType input must not
        mutate the source tile.

        Pre-fix, the per-row tile.slice in the TileType-input lowering used the
        4-arg form (with an explicit valid_shape == shape), which routed
        codegen through a materializing pto.tmov whose destination aliased the
        source allocation. After the gather loop completed, the source tile's
        row 0 held the last-materialized row's content instead of the
        original. This test echoes the local source tile back out after the
        gather and asserts it is bit-identical to the input.
        """
        result = test_runner.run(GatherTileInputSourceUnchangedTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a5", "a5sim")
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_a5_int16_index(self, test_runner, platform):
        """A5 index-form gather with INT16 indices (A5-native; A2/A3 rejects it).

        INT16 indices are accepted by the relaxed ``tile.gather`` type deduction
        and by the A5 (Ascend950) PTOAS verifier, then executed on hardware and
        checked against ``torch.gather``. A2/A3 only permits i32 indices, so this
        case is pinned to A5 via ``@pytest.mark.platforms``.
        """
        result = test_runner.run(GatherA5INT16IndexTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


class TestGatherMask:
    # --- Mask form ---

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_mask_p0101(self, test_runner, platform):
        result = test_runner.run(GatherMaskP0101TestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_mask_p0101_toplevel(self, test_runner, platform):
        """Top-level pl.gather alias (mask form) matches pl.tensor.gather."""
        result = test_runner.run(GatherMaskP0101TopLevelTestCase(platform=platform))
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
