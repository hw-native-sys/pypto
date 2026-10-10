# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end value tests for ``pl.reinterpret_view``.

These cases complement the unit tests for inferred shapes and byte counts by
executing the generated PTO kernel and comparing every output bit.  The input
contains distinct positive and negative FP32 values so an accidental numeric
conversion, element reorder, truncation, or duplication is observable.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.pypto_core.passes import MemoryPlanner

ROWS = 8
COLS = 16
_PLANNERS = [MemoryPlanner.PYPTO, MemoryPlanner.DSA_RP, MemoryPlanner.PTOAS]


def _planner_tag(planner: MemoryPlanner) -> str:
    return {
        MemoryPlanner.PYPTO: "pypto",
        MemoryPlanner.DSA_RP: "dsa_rp",
        MemoryPlanner.PTOAS: "ptoas",
    }[planner]


def _make_source() -> torch.Tensor:
    """Return deterministic FP32 values with varied signs and bit patterns."""
    values = torch.arange(ROWS * COLS, dtype=torch.float32).reshape(ROWS, COLS)
    signs = torch.where(torch.arange(COLS) % 2 == 0, 1.0, -1.0)
    return ((values + 0.25) * signs).contiguous()


@pl.program
class SameWidthReinterpretProgram:
    """Reinterpret FP32 as INT32 without changing the logical shape."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        data: pl.Tensor[[ROWS, COLS], pl.FP32],
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        source: pl.Tile[[ROWS, COLS], pl.FP32] = pl.load(data, [0, 0], [ROWS, COLS])
        viewed: pl.Tile[[ROWS, COLS], pl.INT32] = pl.reinterpret_view(source, pl.INT32)
        return pl.store(viewed, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        data: pl.Tensor[[ROWS, COLS], pl.FP32],
        output: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.INT32]:
        return self.kernel(data, output)


@pl.program
class NarrowerElementReinterpretProgram:
    """Reinterpret FP32 as INT16 and infer twice as many columns."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        data: pl.Tensor[[ROWS, COLS], pl.FP32],
        output: pl.Out[pl.Tensor[[ROWS, COLS * 2], pl.INT16]],
    ) -> pl.Tensor[[ROWS, COLS * 2], pl.INT16]:
        source: pl.Tile[[ROWS, COLS], pl.FP32] = pl.load(data, [0, 0], [ROWS, COLS])
        viewed: pl.Tile[[ROWS, COLS * 2], pl.INT16] = pl.reinterpret_view(source, pl.INT16)
        return pl.store(viewed, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        data: pl.Tensor[[ROWS, COLS], pl.FP32],
        output: pl.Out[pl.Tensor[[ROWS, COLS * 2], pl.INT16]],
    ) -> pl.Tensor[[ROWS, COLS * 2], pl.INT16]:
        return self.kernel(data, output)


class SameWidthReinterpretTestCase(PTOTestCase):
    """FP32 and INT32 have one target element per source element."""

    def __init__(self, memory_planner: MemoryPlanner):
        super().__init__(memory_planner=memory_planner)
        self._memory_planner = memory_planner

    def get_name(self) -> str:
        return f"reinterpret_view_fp32_to_int32_values_{_planner_tag(self._memory_planner)}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("data", [ROWS, COLS], DataType.FP32, init_value=_make_source),
            TensorSpec("output", [ROWS, COLS], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SameWidthReinterpretProgram

    def compute_expected(self, tensors, params=None) -> None:
        tensors["output"][:] = tensors["data"].contiguous().view(torch.int32)


class NarrowerElementReinterpretTestCase(PTOTestCase):
    """FP32 to INT16 produces two ordered target elements per source element."""

    def __init__(self, memory_planner: MemoryPlanner):
        super().__init__(memory_planner=memory_planner)
        self._memory_planner = memory_planner

    def get_name(self) -> str:
        return f"reinterpret_view_fp32_to_int16_values_{_planner_tag(self._memory_planner)}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("data", [ROWS, COLS], DataType.FP32, init_value=_make_source),
            TensorSpec("output", [ROWS, COLS * 2], DataType.INT16, is_output=True),
        ]

    def get_program(self) -> Any:
        return NarrowerElementReinterpretProgram

    def compute_expected(self, tensors, params=None) -> None:
        expected = tensors["data"].contiguous().view(torch.int16)
        assert list(expected.shape) == [ROWS, COLS * 2]
        assert expected.numel() == tensors["data"].numel() * 2
        tensors["output"][:] = expected


@pytest.mark.platforms("a2a3", "a2a3sim")
class TestReinterpretView:
    """Bit-exact runtime coverage for same-width and width-changing views."""

    @pytest.mark.parametrize("planner", _PLANNERS, ids=_planner_tag)
    def test_same_width_values_are_bit_exact(self, test_runner, planner):
        result = test_runner.run(SameWidthReinterpretTestCase(planner))
        assert result.passed, f"same-width reinterpret_view ({_planner_tag(planner)}) failed: {result.error}"

    @pytest.mark.parametrize("planner", _PLANNERS, ids=_planner_tag)
    def test_narrower_element_values_and_count_are_bit_exact(self, test_runner, planner):
        result = test_runner.run(NarrowerElementReinterpretTestCase(planner))
        assert result.passed, (
            f"width-changing reinterpret_view ({_planner_tag(planner)}) failed: {result.error}"
        )


@pl.jit.incore
def _packed_fp4_alias_kernel(
    data: pl.Tensor[[16, 64], pl.FP4E2M1X2],
    output: pl.Out[pl.Tensor[[8, 256], pl.BF16]],
    BYTE_DTYPE: pl.constexpr,
    TENSOR_SURFACE: pl.constexpr,
):
    if TENSOR_SURFACE:
        byte_view = pl.tensor.reinterpret_view(data, BYTE_DTYPE)
        reshaped = pl.tensor.reshape(byte_view, [8, 128])
        packed_view = pl.tensor.reinterpret_view(reshaped, pl.FP4E2M1X2)
        cast = pl.tensor.cast(packed_view, pl.BF16)
        result = pl.tensor.assemble(output, cast, [0, 0])
    else:
        source = pl.load(data, [0, 0], [16, 64])
        byte_tile = pl.tile.reinterpret_view(source, BYTE_DTYPE)
        reshaped_tile = pl.tile.reshape(byte_tile, [8, 128])
        tile = pl.tile.reinterpret_view(reshaped_tile, pl.FP4E2M1X2)
        result = pl.store(pl.cast(tile, pl.BF16), [0, 0], output)
    return result


@pl.jit
def _packed_fp4_alias_entry(
    data: pl.Tensor[[16, 64], pl.FP4E2M1X2],
    output: pl.Out[pl.Tensor[[8, 256], pl.BF16]],
    BYTE_DTYPE: pl.constexpr,
    TENSOR_SURFACE: pl.constexpr,
):
    return _packed_fp4_alias_kernel(data, output, BYTE_DTYPE, TENSOR_SURFACE)


def _packed_fp4_source():
    # Every E2M1 code appears in both nibble positions, with a different row pitch.
    codes = torch.arange(1024).reshape(16, 64).to(torch.uint8)
    return codes.view(torch.float4_e2m1fn_x2)


def _packed_fp4_alias_golden(tensors):
    raw = tensors["data"].contiguous().view(torch.uint8).reshape(8, 128)
    codes = torch.empty((8, 256), dtype=torch.long)
    codes[:, 0::2] = (raw & 15).to(torch.long)
    codes[:, 1::2] = (raw >> 4).to(torch.long)
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.bfloat16,
    )
    return {"output": values[codes]}


def _packed_fp4_alias_cases():
    if not hasattr(torch, "float4_e2m1fn_x2"):
        return []
    return [
        st.case(
            _packed_fp4_alias_entry,
            _packed_fp4_source(),
            torch.empty((8, 256), dtype=torch.bfloat16),
            byte_dtype,
            tensor_surface,
            name=(
                f"fp4_byte_reshape_{byte_dtype}_"
                f"{'tensor' if tensor_surface else 'tile'}_{_planner_tag(planner)}"
            ),
            golden=_packed_fp4_alias_golden,
            memory_planner=planner,
            rtol=0.0,
            atol=0.0,
        )
        for planner in _PLANNERS
        for byte_dtype in (pl.UINT8, pl.INT8)
        for tensor_surface in (True, False)
        if planner != MemoryPlanner.PTOAS
    ]


@pytest.mark.platforms("a5")
@st.cases(*_packed_fp4_alias_cases())
def test_packed_fp4_byte_reshape_round_trip(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
