# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end tests for ``pl.tensor.scatter`` (index form, ``dim=-1`` column scatter).

Semantics (torch ``scatter_`` along the last axis)::

    out = base.clone()
    out[b, index[b, k]] = val[b, k]        # for all b, k (k ascending = last-wins)
    # columns no index[b, :] selects keep base's value (DPS preserve)

**Why these inputs (avoiding the degenerate round-trip)**

A previous version used ``base == val == src`` and ``expected == src``, so a
scatter that did *nothing* (identity passthrough of ``base``) still produced the
expected output and passed. To make every test discriminating, here:

- ``base`` holds a **negative sentinel** ``base[b, j] = -(j + 1 + b)`` —
  distinct per (row, col) and disjoint from any written value.
- ``val`` holds **positive, row+col-varying** values.
- ``expected`` is computed from ``base`` + ``index`` + ``val`` (not a copy of any
  input), so a no-op (``out == base``, all negative) mismatches immediately, and
  the DPS-preserved columns are checked to still hold their sentinel.

**dtype / size rules exercised** (one program per element type):

| dst/src | bytes | indexes | cols rule (cols*sizeof % 32 == 0)        |
| ------- | ----- | ------- | ---------------------------------------- |
| FP32    | 4     | INT32   | base S=16, val/idx K=8                    |
| INT32   | 4     | INT32   | base S=16, val/idx K=8                    |
| FP16    | 2     | INT16   | base S=32, val/idx K=16                   |
| BF16    | 2     | INT16   | base S=32, val/idx K=16                   |
| INT16   | 2     | INT16   | base S=32, val/idx K=16                   |

Row counts (B) also satisfy the lowering's internal arange alignment
(rows*sizeof(indexes) % 32 == 0): B=8 for the i32 path, B=16 for the i16 path.
A separate B=1 FP32 case is the regression for issue #1586: a single-row scatter
must not emit a ``[1, 1]`` ``tile.ci`` (which the ``pto.tci`` Cols!=1 ISA check
rejects) — the lone row's base offset is 0, so the column index is the flat index
directly.

BF16 works because the scatter lowering rebuilds the DPS-preserve blend with a
select (``out = sel(mask != 0, scattered, input)``) rather than ``input * mask``
— the latter's ``tile.mul`` lowers to ``pto.tmul``, which A2/A3 rejects for
bf16. (INT8 is left uncovered: the 1-byte path needs a separate validation.)

A separate FP32 case feeds **repeated** indices with distinct values to pin the
ascending-k last-wins ordering (the round-trip version hid this by writing equal
values to repeated targets).

**Mask form (A2/A3).** ``TestScatterMaskForm`` covers ``tensor.scatter_mask``
(``mask_pattern=<int>`` + ``dst``), the column-wise inverse of the mask-form
gather: each compact ``input`` row is written into the mask-selected columns of
the wider ``dst`` (``dst.cols == input.cols * stride``). The form runs on the
A2/A3 backend (``BackendType.Ascend910B``); A5 (``Ascend950``) rejects it, so
those cases are pinned to A2/A3 only. Column selection mirrors gather: P0101
hits even columns (``0::2``), P1010 hits odd columns (``1::2``).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# --- Data builders (negative sentinel base, positive distinct values) ---


def _make_base(b: int, s: int, torch_dtype: torch.dtype) -> torch.Tensor:
    """``[B, S]`` negative sentinel: ``base[i, j] = -(j + 1 + i)`` (disjoint from val)."""
    rows = torch.arange(b, dtype=torch.int32).reshape(b, 1)
    cols = torch.arange(s, dtype=torch.int32).reshape(1, s)
    return (-(cols + 1 + rows)).to(torch_dtype).contiguous()


def _make_index(b: int, s: int, k: int, torch_idx_dtype: torch.dtype) -> torch.Tensor:
    """``[B, K]`` per-row distinct columns: ``index[i, m] = (m + i) % S`` (K <= S).

    Distinct within a row (a per-row cyclic shift), so the scatter is well-defined
    regardless of write order; different rows hit different columns, exercising
    per-row index usage; K < S leaves preserve columns to validate DPS.
    """
    rows = torch.arange(b, dtype=torch.int64).reshape(b, 1)
    sel = torch.arange(k, dtype=torch.int64).reshape(1, k)
    return ((sel + rows) % s).to(torch_idx_dtype).contiguous()


def _make_repeat_index(b: int, s: int, k: int, torch_idx_dtype: torch.dtype) -> torch.Tensor:
    """``[B, K]`` repeated columns: ``index[i, m] = (m // 2 + i) % S``.

    Each of the K/2 target columns is written twice (m even, then m odd). With
    distinct ``val`` the odd-m write must win under ascending-k last-wins.
    """
    rows = torch.arange(b, dtype=torch.int64).reshape(b, 1)
    sel = torch.arange(k, dtype=torch.int64).reshape(1, k)
    return ((sel // 2 + rows) % s).to(torch_idx_dtype).contiguous()


def _make_values(b: int, k: int, torch_dtype: torch.dtype) -> torch.Tensor:
    """``[B, K]`` positive distinct values: ``val[i, m] = i*K + m + 1`` (<= B*K).

    Distinct per element and never collides with the negative sentinel base.
    """
    rows = torch.arange(b, dtype=torch.int32).reshape(b, 1)
    sel = torch.arange(k, dtype=torch.int32).reshape(1, k)
    return (rows * k + sel + 1).to(torch_dtype).contiguous()


def _apply_scatter(base: torch.Tensor, index: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Reference column scatter: ``out[i, index[i, m]] = values[i, m]`` (ascending m)."""
    out = base.clone()
    b, k = index.shape
    for i in range(b):
        for m in range(k):
            out[i, int(index[i, m])] = values[i, m]
    return out


def _scatter_specs(
    b: int,
    s: int,
    k: int,
    dt: DataType,
    torch_dt: torch.dtype,
    idt: DataType,
    torch_idt: torch.dtype,
    *,
    repeat: bool = False,
) -> list[TensorSpec]:
    """Build the (base, idx, val, output) TensorSpecs for a scatter case."""
    index_fn = _make_repeat_index if repeat else _make_index
    return [
        TensorSpec("base", [b, s], dt, init_value=lambda: _make_base(b, s, torch_dt)),
        TensorSpec("idx", [b, k], idt, init_value=lambda: index_fn(b, s, k, torch_idt)),
        TensorSpec("val", [b, k], dt, init_value=lambda: _make_values(b, k, torch_dt)),
        TensorSpec("output", [b, s], dt, is_output=True),
    ]


def _scatter_mask_specs(b: int, c: int, stride: int, dt: DataType, torch_dt: torch.dtype) -> list[TensorSpec]:
    """Build the (inp, dst, output) TensorSpecs for a mask-form scatter case.

    ``inp`` is ``[B, C]`` of distinct positive values; ``dst`` is ``[B, C*stride]``
    initialised to zeros so the expected unselected columns are zero regardless
    of whether pto.tscatter zeros or preserves them (its index form zeros
    unwritten slots; a zero base makes the test robust either way).
    """
    dst_cols = c * stride
    return [
        TensorSpec("inp", [b, c], dt, init_value=lambda: _make_values(b, c, torch_dt)),
        TensorSpec("dst", [b, dst_cols], dt, init_value=lambda: torch.zeros(b, dst_cols, dtype=torch_dt)),
        TensorSpec("output", [b, dst_cols], dt, is_output=True),
    ]


# --- Programs (one per element type; shapes satisfy the 32-byte row alignment) ---


@pl.program
class ScatterFP32Program:
    """FP32 dst/src + INT32 indexes (4-byte path)."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        base: pl.Tensor[[8, 16], pl.FP32],
        idx: pl.Tensor[[8, 8], pl.INT32],
        val: pl.Tensor[[8, 8], pl.FP32],
        output: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
    ) -> pl.Tensor[[8, 16], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.scatter(base, dim=-1, index=idx, src=val)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class ScatterINT32Program:
    """INT32 dst/src + INT32 indexes (4-byte integer path)."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        base: pl.Tensor[[8, 16], pl.INT32],
        idx: pl.Tensor[[8, 8], pl.INT32],
        val: pl.Tensor[[8, 8], pl.INT32],
        output: pl.Out[pl.Tensor[[8, 16], pl.INT32]],
    ) -> pl.Tensor[[8, 16], pl.INT32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.scatter(base, dim=-1, index=idx, src=val)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class ScatterFP16Program:
    """FP16 dst/src + INT16 indexes (2-byte path)."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        base: pl.Tensor[[16, 32], pl.FP16],
        idx: pl.Tensor[[16, 16], pl.INT16],
        val: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP16]],
    ) -> pl.Tensor[[16, 32], pl.FP16]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.scatter(base, dim=-1, index=idx, src=val)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class ScatterBF16Program:
    """BF16 dst/src + INT16 indexes (2-byte path; preserve blend via tile.sel)."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        base: pl.Tensor[[16, 32], pl.BF16],
        idx: pl.Tensor[[16, 16], pl.INT16],
        val: pl.Tensor[[16, 16], pl.BF16],
        output: pl.Out[pl.Tensor[[16, 32], pl.BF16]],
    ) -> pl.Tensor[[16, 32], pl.BF16]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.scatter(base, dim=-1, index=idx, src=val)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class ScatterINT16Program:
    """INT16 dst/src + INT16 indexes (2-byte integer path)."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        base: pl.Tensor[[16, 32], pl.INT16],
        idx: pl.Tensor[[16, 16], pl.INT16],
        val: pl.Tensor[[16, 16], pl.INT16],
        output: pl.Out[pl.Tensor[[16, 32], pl.INT16]],
    ) -> pl.Tensor[[16, 32], pl.INT16]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.scatter(base, dim=-1, index=idx, src=val)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class Scatter1RowProgram:
    """Single-row dst/src/index, FP32 + INT32 (regression for issue #1586).

    Before the fix, the scatter lowering built its per-row base arange with a
    ``tile.ci`` of shape ``[1, rows]``; for ``rows == 1`` that is ``[1, 1]``,
    which trips the ``pto.tci`` "innermost dim (Cols) != 1" ISA check and fails
    compilation in ``convert_tensor_to_tile_ops``. With ``rows == 1`` the row
    base offset is always 0, so the lowering now uses the column index directly.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        base: pl.Tensor[[1, 16], pl.FP32],
        idx: pl.Tensor[[1, 8], pl.INT32],
        val: pl.Tensor[[1, 8], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.scatter(base, dim=-1, index=idx, src=val)
            output = pl.assemble(output, out, [0, 0])
        return output


# --- Mask-form programs (A2/A3 only; A5/Ascend950 rejects pto.tscatter mask form) ---


@pl.program
class ScatterMaskP0101Program:
    """Mask-form scatter (P0101, A2/A3): write ``inp`` into even columns of ``dst``.

    ``inp [8, 8] FP32`` expands into ``dst [8, 16] FP32`` (stride 2): the inverse
    of the P0101 mask gather, so ``dst[:, 0::2] = inp``.
    """

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
class ScatterMaskP1010Program:
    """Mask-form scatter (P1010, A2/A3): write ``inp`` into odd columns of ``dst``.

    ``inp [8, 8] FP32`` expands into ``dst [8, 16] FP32`` (stride 2): the inverse
    of the P1010 mask gather, so ``dst[:, 1::2] = inp``.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        inp: pl.Tensor[[8, 8], pl.FP32],
        dst: pl.Tensor[[8, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
    ) -> pl.Tensor[[8, 16], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.tensor.scatter(inp, mask_pattern=pl.tile.MaskPattern.P1010, dst=dst)
            output = pl.assemble(output, out, [0, 0])
        return output


# --- Test cases ---


class _ScatterBaseTestCase(PTOTestCase):
    __test__ = False

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        # Ground truth derived from the actual index + values (not a copy of any
        # input): a no-op leaves `base` (all negative) and fails immediately, and
        # the unselected columns must keep their sentinel (DPS preserve).
        tensors["output"][:] = _apply_scatter(tensors["base"], tensors["idx"], tensors["val"])


class ScatterFP32TestCase(_ScatterBaseTestCase):
    def get_name(self) -> str:
        return "scatter_fp32"

    def define_tensors(self) -> list[TensorSpec]:
        return _scatter_specs(8, 16, 8, DataType.FP32, torch.float32, DataType.INT32, torch.int32)

    def get_program(self) -> Any:
        return ScatterFP32Program


class ScatterINT32TestCase(_ScatterBaseTestCase):
    def get_name(self) -> str:
        return "scatter_int32"

    def define_tensors(self) -> list[TensorSpec]:
        return _scatter_specs(8, 16, 8, DataType.INT32, torch.int32, DataType.INT32, torch.int32)

    def get_program(self) -> Any:
        return ScatterINT32Program


class ScatterFP16TestCase(_ScatterBaseTestCase):
    def get_name(self) -> str:
        return "scatter_fp16"

    def define_tensors(self) -> list[TensorSpec]:
        return _scatter_specs(16, 32, 16, DataType.FP16, torch.float16, DataType.INT16, torch.int16)

    def get_program(self) -> Any:
        return ScatterFP16Program


class ScatterBF16TestCase(_ScatterBaseTestCase):
    def get_name(self) -> str:
        return "scatter_bf16"

    def define_tensors(self) -> list[TensorSpec]:
        return _scatter_specs(16, 32, 16, DataType.BF16, torch.bfloat16, DataType.INT16, torch.int16)

    def get_program(self) -> Any:
        return ScatterBF16Program


class ScatterINT16TestCase(_ScatterBaseTestCase):
    def get_name(self) -> str:
        return "scatter_int16"

    def define_tensors(self) -> list[TensorSpec]:
        return _scatter_specs(16, 32, 16, DataType.INT16, torch.int16, DataType.INT16, torch.int16)

    def get_program(self) -> Any:
        return ScatterINT16Program


class ScatterRepeatLastWinsTestCase(_ScatterBaseTestCase):
    """FP32 with repeated indices + distinct values: pins ascending-k last-wins."""

    def get_name(self) -> str:
        return "scatter_repeat_last_wins"

    def define_tensors(self) -> list[TensorSpec]:
        return _scatter_specs(
            8, 16, 8, DataType.FP32, torch.float32, DataType.INT32, torch.int32, repeat=True
        )

    def get_program(self) -> Any:
        return ScatterFP32Program


class Scatter1RowTestCase(_ScatterBaseTestCase):
    """Single-row scatter (regression for issue #1586)."""

    def get_name(self) -> str:
        return "scatter_1row"

    def define_tensors(self) -> list[TensorSpec]:
        return _scatter_specs(1, 16, 8, DataType.FP32, torch.float32, DataType.INT32, torch.int32)

    def get_program(self) -> Any:
        return Scatter1RowProgram


class _ScatterMaskBaseTestCase(PTOTestCase):
    """Base for mask-form scatter cases. Pinned to A2/A3 (Ascend910B).

    Subclasses set ``_start`` (0 for P0101, 1 for P1010) and ``_stride`` (2);
    ``compute_expected`` writes ``inp`` into the ``[start::stride]`` columns of a
    zero ``[B, C*stride]`` tensor (the unselected columns stay zero).
    """

    __test__ = False
    _start: int = 0
    _stride: int = 2

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        # Mask-form pto.tscatter is an A2/A3 feature; A5 (Ascend950) rejects it.
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None):
        b, dst_cols = tensors["output"].shape
        out = torch.zeros(b, dst_cols, dtype=tensors["inp"].dtype)
        out[:, self._start :: self._stride] = tensors["inp"]
        tensors["output"][:] = out


class ScatterMaskP0101TestCase(_ScatterMaskBaseTestCase):
    _start = 0  # P0101 selects even columns

    def get_name(self) -> str:
        return "scatter_mask_p0101"

    def define_tensors(self) -> list[TensorSpec]:
        return _scatter_mask_specs(8, 8, 2, DataType.FP32, torch.float32)

    def get_program(self) -> Any:
        return ScatterMaskP0101Program


class ScatterMaskP1010TestCase(_ScatterMaskBaseTestCase):
    _start = 1  # P1010 selects odd columns

    def get_name(self) -> str:
        return "scatter_mask_p1010"

    def define_tensors(self) -> list[TensorSpec]:
        return _scatter_mask_specs(8, 8, 2, DataType.FP32, torch.float32)

    def get_program(self) -> Any:
        return ScatterMaskP1010Program


# --- Tests ---


class TestScatterIndexForm:
    """Index-form column scatter across the dst/src + indexes dtype matrix."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scatter_fp32(self, test_runner, platform):
        result = test_runner.run(ScatterFP32TestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scatter_int32(self, test_runner, platform):
        result = test_runner.run(ScatterINT32TestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scatter_fp16(self, test_runner, platform):
        result = test_runner.run(ScatterFP16TestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scatter_bf16(self, test_runner, platform):
        result = test_runner.run(ScatterBF16TestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scatter_int16(self, test_runner, platform):
        result = test_runner.run(ScatterINT16TestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scatter_repeat_last_wins(self, test_runner, platform):
        result = test_runner.run(ScatterRepeatLastWinsTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scatter_1row(self, test_runner, platform):
        result = test_runner.run(Scatter1RowTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


class TestScatterMaskForm:
    """Mask-form row scatter — A2/A3 only (A5/Ascend950 rejects the mask form).

    Each compact ``inp`` row is written into the mask-selected columns of the
    wider ``dst`` (``dst.cols == inp.cols * stride``); column selection mirrors
    the mask gather (P0101 → even, P1010 → odd).
    """

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scatter_mask_p0101(self, test_runner, platform):
        result = test_runner.run(ScatterMaskP0101TestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scatter_mask_p1010(self, test_runner, platform):
        result = test_runner.run(ScatterMaskP1010TestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
