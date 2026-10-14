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

from dataclasses import replace
from typing import Any

import pypto.language as pl
import pytest
import torch
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


def make_fp4_byte_decode_region(row_start: int, column_start: int):
    """Decode two tiles through one UB allocation, retaining GM byte offsets."""

    @pl.jit(auto_scope=False)
    def decode(
        packed: pl.Tensor[[128, 320], pl.UINT8],
        output: pl.InOut[pl.Tensor[[128, 640], pl.BF16]],
    ):
        with pl.spmd(1, sync_start=False):
            worker = pl.tile.get_block_idx()
            for row in pl.range(row_start + worker * 64, row_start + 64, 32):
                payload = pl.tile.load(packed, [row, column_start], [32, 256])
                fp4 = pl.reinterpret_view(payload, pl.FP4E2M1X2)
                decoded = pl.cast(fp4, pl.BF16)
                pl.tile.store(decoded, [row, column_start * 2], output)
        return output

    return decode


def fp4_byte_reference_bits(packed: torch.Tensor) -> torch.Tensor:
    """Independent E2M1 LUT of BF16 bit patterns, including both signed zeros."""
    lut = torch.tensor(
        [
            0x0000,
            0x3F00,
            0x3F80,
            0x3FC0,
            0x4000,
            0x4040,
            0x4080,
            0x40C0,
            0x8000,
            0xBF00,
            0xBF80,
            0xBFC0,
            0xC000,
            0xC040,
            0xC080,
            0xC0C0,
        ],
        dtype=torch.int32,
    ).to(torch.int16)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    return lut[codes].reshape(packed.shape[0], packed.shape[1] * 2)


@pytest.mark.platforms("a5")
@pytest.mark.parametrize("planner", [MemoryPlanner.PYPTO, MemoryPlanner.DSA_RP], ids=_planner_tag)
@pytest.mark.parametrize(("row_start", "column_start"), [(0, 0), (32, 0), (64, 32)])
def test_fp4_byte_decode_offsets_and_reuse_are_bit_exact(test_config, planner, row_start, column_start):
    decode = make_fp4_byte_decode_region(row_start, column_start)
    packed = torch.zeros((128, 320), dtype=torch.uint8)
    output = torch.empty((128, 640), dtype=torch.bfloat16)
    config = replace(test_config, memory_planner=planner)
    compiled = decode.compile(packed, output, config=config)
    if config.codegen_only:
        return

    for repeat in range(3):
        # Each loaded row contains every packed byte; the row shift makes a
        # wrong physical row observable, and repeat changes reused buffers.
        packed[:] = ((torch.arange(320)[None, :] + 17 * torch.arange(128)[:, None] + 37 * repeat) % 256).to(
            torch.uint8
        )
        expected_bits = torch.full((128, 640), 0x7FC1, dtype=torch.int16)
        output.view(torch.int16).copy_(expected_bits)
        expected_bits[row_start : row_start + 64, column_start * 2 : column_start * 2 + 512] = (
            fp4_byte_reference_bits(packed[row_start : row_start + 64, column_start : column_start + 256])
        )
        compiled(packed, output, config=config)
        assert torch.equal(output.view(torch.int16), expected_bits), (
            f"{_planner_tag(planner)}: BF16 bit mismatch at row={row_start}, "
            f"column={column_start}, repeat={repeat}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
