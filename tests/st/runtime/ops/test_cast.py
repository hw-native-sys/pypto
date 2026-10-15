# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile.cast (pto.tcvt) narrowing conversions.

Regression for #1549: pto.tcvt mis-orders elements when its source tile is
col_major. A [1, N] -> [N, 1] reshape yields a col_major [N, 1] tile; narrowing
that view i32 -> i16 used to reverse the elements (silent wrong output). The
tile.cast row_major layout spec lets ResolveBackendOpLayouts repair col_major
sources by reshaping [N, 1] -> [1, N] row_major around the cast.

  TileCastColMajorNarrowTestCase : narrow a col_major [N, 1] view i32 -> i16
                                   (the regression — must preserve element order).
  TileCastRowMajorNarrowTestCase : narrow a row_major [1, N] tile i32 -> i16
                                   (always-correct control / baseline).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.pypto_core.passes import MemoryPlanner

N = 16


def _arange_row() -> torch.Tensor:
    """Row-major [1, N] i32 with distinct, non-symmetric values [0 .. N-1].

    Distinct ascending values make element reordering observable: a reversed
    cast would produce [N-1 .. 0] instead of [0 .. N-1].
    """
    return torch.arange(0, N, dtype=torch.int32).reshape(1, N)


# ---------------------------------------------------------------------------
# Kernel programs
# ---------------------------------------------------------------------------


@pl.program
class TileCastColMajorNarrowProgram:
    """Narrow a col_major [N, 1] index-style view i32 -> i16.

    Reshaping [1, N] -> [N, 1] produces a col_major tile; the cast runs on that
    col_major source (the #1549 trigger). The result is reshaped back to a
    32-byte-aligned [1, N] row before the store so only the cast source is
    col_major.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[1, N], pl.INT32],
        out: pl.Out[pl.Tensor[[1, N], pl.INT16]],
    ) -> pl.Tensor[[1, N], pl.INT16]:
        tile_a: pl.Tile[[1, N], pl.INT32] = pl.load(a, [0, 0], [1, N])
        col: pl.Tile[[N, 1], pl.INT32] = pl.tile.reshape(tile_a, [N, 1])
        narrowed: pl.Tile[[N, 1], pl.INT16] = pl.tile.cast(col, target_type=pl.INT16)
        row: pl.Tile[[1, N], pl.INT16] = pl.tile.reshape(narrowed, [1, N])
        out = pl.store(row, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[1, N], pl.INT32],
        out: pl.Out[pl.Tensor[[1, N], pl.INT16]],
    ) -> pl.Tensor[[1, N], pl.INT16]:
        out = self.kernel(a, out)
        return out


@pl.program
class TileCastRowMajorNarrowProgram:
    """Narrow a row_major [1, N] tile i32 -> i16 (control, no col_major view)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[1, N], pl.INT32],
        out: pl.Out[pl.Tensor[[1, N], pl.INT16]],
    ) -> pl.Tensor[[1, N], pl.INT16]:
        tile_a: pl.Tile[[1, N], pl.INT32] = pl.load(a, [0, 0], [1, N])
        narrowed: pl.Tile[[1, N], pl.INT16] = pl.tile.cast(tile_a, target_type=pl.INT16)
        out = pl.store(narrowed, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[1, N], pl.INT32],
        out: pl.Out[pl.Tensor[[1, N], pl.INT16]],
    ) -> pl.Tensor[[1, N], pl.INT16]:
        out = self.kernel(a, out)
        return out


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TileCastColMajorNarrowTestCase(PTOTestCase):
    """i32 -> i16 cast on a col_major [N, 1] view must preserve element order."""

    def get_name(self) -> str:
        return "tile_cast_col_major_narrow"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [1, N], DataType.INT32, init_value=_arange_row),
            TensorSpec("out", [1, N], DataType.INT16, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileCastColMajorNarrowProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def compute_expected(self, tensors, params=None) -> None:
        # Reshape round-trip is a no-op on values; the cast only narrows the dtype,
        # so each element keeps its value and position: out[0, k] == a[0, k].
        tensors["out"][:] = tensors["a"].to(torch.int16)


class TileCastRowMajorNarrowTestCase(PTOTestCase):
    """i32 -> i16 cast on a row_major [1, N] tile (always-correct baseline)."""

    def get_name(self) -> str:
        return "tile_cast_row_major_narrow"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [1, N], DataType.INT32, init_value=_arange_row),
            TensorSpec("out", [1, N], DataType.INT16, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileCastRowMajorNarrowProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def compute_expected(self, tensors, params=None) -> None:
        tensors["out"][:] = tensors["a"].to(torch.int16)


# ---------------------------------------------------------------------------
# pytest wrappers
# ---------------------------------------------------------------------------


class TestCast:
    """End-to-end tile.cast narrowing tests, covering col_major and row_major sources."""

    def test_tile_cast_col_major_narrow(self, test_runner):
        """Regression for #1549: i32 -> i16 on a col_major [N, 1] view keeps element order."""
        result = test_runner.run(TileCastColMajorNarrowTestCase())
        assert result.passed, f"col_major narrow cast failed: {result.error}"

    def test_tile_cast_row_major_narrow(self, test_runner):
        """Control: i32 -> i16 on a row_major [1, N] tile is correct."""
        result = test_runner.run(TileCastRowMajorNarrowTestCase())
        assert result.passed, f"row_major narrow cast failed: {result.error}"


# Regression for #2973: multi-row TCVT heads must not overwrite unread tails.
# Exercise Tile storage explicitly; eligible Buffer kernels use a different allocator.
ROWS = 16


def _tail_cast_case(width: int, memory_planner=None):
    @pl.jit
    def cast_store(
        x: pl.Tensor[[ROWS, width], pl.FP32],
        y: pl.Out[pl.Tensor[[ROWS, width], pl.INT8]],
    ):
        for _blk in pl.spmd(1, name_hint="cast_store"):
            y[0:ROWS, 0:width] = pl.cast(x[0:ROWS, 0:width], target_type=pl.INT8, mode="rint")
        return y

    def golden(tensors):
        return tensors["x"].round().clamp(-128, 127).to(torch.int8)

    def compare(actual, expected):
        delta = (actual["y"].to(torch.int32) - expected["y"].to(torch.int32)).abs()
        bad = (delta > 1).nonzero()
        columns = f", columns {int(bad[:, 1].min())}..{int(bad[:, 1].max())}" if len(bad) else ""
        message = f"W={width}: {len(bad)} wrong elements, worst {int(delta.max())}{columns}"
        assert len(bad) == 0, message

    generator = torch.Generator().manual_seed(7)
    x = torch.randn(ROWS, width, generator=generator) * 40.0
    return st.case(
        cast_store,
        x,
        torch.zeros(ROWS, width, dtype=torch.int8),
        name=f"issue_2973_w{width}" + (f"_{memory_planner}" if memory_planner is not None else ""),
        golden=golden,
        compare=compare,
        memory_planner=memory_planner,
        enable_buffer_ir=False,
    )


@pytest.mark.platforms("a2a3", "a2a3sim", reason="A2/A3 narrowing cast alias regression")
@st.cases(_tail_cast_case(512), _tail_cast_case(544))
def test_issue_2973(case_run):
    case_run.assert_passed()


def _tail_cast_single_step_case(source_dtype, target_dtype, width: int, keep_source: bool, planner):
    @pl.jit.incore
    def cast_kernel(
        x: pl.Tensor[[ROWS, width], source_dtype],
        y: pl.Out[pl.Tensor[[ROWS, width], target_dtype]],
        echo: pl.Out[pl.Tensor[[ROWS, width], source_dtype]],
    ):
        source = pl.load(x, [0, 0], [ROWS, width])
        converted = pl.cast(source, target_type=target_dtype, mode="rint")
        pl.store(converted, [0, 0], y)
        if keep_source:
            pl.store(source, [0, 0], echo)
        return y

    @pl.jit
    def cast_program(
        x: pl.Tensor[[ROWS, width], source_dtype],
        y: pl.Out[pl.Tensor[[ROWS, width], target_dtype]],
        echo: pl.Out[pl.Tensor[[ROWS, width], source_dtype]],
    ):
        return cast_kernel(x, y, echo)

    torch_types = {
        pl.FP32: torch.float32,
        pl.FP16: torch.float16,
        pl.BF16: torch.bfloat16,
        pl.INT8: torch.int8,
    }
    source_type, target_type = torch_types[source_dtype], torch_types[target_dtype]
    generator = torch.Generator().manual_seed(7)
    x = (torch.randn(ROWS, width, generator=generator) * 40.0).to(source_type)

    def golden(tensors):
        value = tensors["x"]
        if target_type == torch.int8:
            value = value.round().clamp(-128, 127)
        tensors["y"].copy_(value.to(target_type))
        if keep_source:
            tensors["echo"].copy_(tensors["x"])

    def compare(actual, expected):
        delta = (actual["y"].float() - expected["y"].float()).abs()
        bad = (~torch.isfinite(delta) | (delta != 0)).nonzero()
        columns = f", columns {int(bad[:, 1].min())}..{int(bad[:, 1].max())}" if len(bad) else ""
        message = (
            f"{source_type}->{target_type} W={width} keep={keep_source}: "
            f"{len(bad)} wrong, max={float(delta.max())}{columns}"
        )
        if keep_source:
            assert torch.equal(actual["echo"], expected["echo"]), "source overwritten"
        assert len(bad) == 0, message

    return st.case(
        cast_program,
        x,
        torch.zeros(ROWS, width, dtype=target_type),
        torch.zeros_like(x),
        name=f"issue_2973_{source_dtype}_{target_dtype}_w{width}_keep{keep_source}_{planner}",
        memory_planner=planner,
        enable_buffer_ir=False,
        golden=golden,
        compare=compare,
    )


@pytest.mark.platforms("a2a3", "a2a3sim", reason="A2/A3 narrowing cast alias regression")
@st.cases(
    *[
        _tail_cast_single_step_case(source, target, width, keep, planner)
        for source, target, width in [
            (pl.FP32, pl.FP16, 544),
            (pl.FP16, pl.INT8, 544),
            (pl.FP32, pl.BF16, 528),
        ]
        for keep in (False, True)
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.DSA_RP)
    ]
)
def test_issue_2973_isolate_alias(case_run):
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a2a3sim", reason="A2/A3 narrowing cast alias regression")
@st.cases(
    _tail_cast_case(544, MemoryPlanner.DSA_RP),
    _tail_cast_case(544, MemoryPlanner.PTOAS),
)
def test_issue_2973_planners(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
