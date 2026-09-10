# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Same-name hardware tests for 32-byte block-offset ``pto.tgatherb``."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

ROWS = 4
OFFSETS_PER_ROW = 8
BLOCK_BYTES = 32
_PL_DTYPE = {
    DataType.INT8: pl.INT8,
    DataType.UINT8: pl.UINT8,
    DataType.INT16: pl.INT16,
    DataType.UINT16: pl.UINT16,
    DataType.INT32: pl.INT32,
    DataType.UINT32: pl.UINT32,
    DataType.BF16: pl.BF16,
    DataType.FP16: pl.FP16,
    DataType.FP32: pl.FP32,
}
_TORCH_DTYPE = {
    DataType.INT8: torch.int8,
    DataType.UINT8: torch.uint8,
    DataType.INT16: torch.int16,
    DataType.UINT16: torch.int16,
    DataType.INT32: torch.int32,
    DataType.UINT32: torch.int32,
    DataType.BF16: torch.bfloat16,
    DataType.FP16: torch.float16,
    DataType.FP32: torch.float32,
}
_ELEMENT_BYTES = {
    DataType.INT8: 1,
    DataType.UINT8: 1,
    DataType.INT16: 2,
    DataType.UINT16: 2,
    DataType.INT32: 4,
    DataType.UINT32: 4,
    DataType.BF16: 2,
    DataType.FP16: 2,
    DataType.FP32: 4,
}


# PTOAS v0.61 narrowed `pto.tgatherb`'s A2/A3 destination width from {1, 2, 4}
# bytes to {2, 4} in the shared op verifier (PTOAS#971), to match a new VPTO
# lowering path whose `llvm.hivm.VGATHERB` intrinsics only come in `.b16` and
# `.b32`. The EmitC path these tests take lowers a 1-byte destination through
# the ISA's dedicated `GatherBInstrB8`, and every case below passed on real
# A2/A3 hardware under v0.60 — so this is an assembler regression, not a pypto
# limitation, and nothing in pypto can make them compile again.
#
# Tracked as hw-native-sys/PTOAS#1495. Drop these marks (do not "fix" the
# tests) once the verifier accepts 1-byte destinations again.
_PTOAS_1495 = (
    "PTOAS >= v0.61 rejects a 1-byte pto.tgatherb destination on A2/A3, which the "
    "ISA implements via GatherBInstrB8 and hardware computes correctly "
    "(hw-native-sys/PTOAS#1495)"
)


def _skip_one_byte_dst(dtype: DataType) -> list[pytest.MarkDecorator]:
    """Collection-time skip marks for a destination PTOAS >= v0.61 refuses.

    A collection-time mark rather than a ``pytest.skip`` in the body, so the
    case is not pre-compiled either — compilation is where it fails.
    """
    return [pytest.mark.skip(reason=_PTOAS_1495)] if _ELEMENT_BYTES[dtype] == 1 else []


def _byte_offsets(pattern: str, rows: int, offsets_per_row: int) -> torch.Tensor:
    source_blocks = rows * offsets_per_row
    indices = torch.arange(source_blocks, dtype=torch.int64)
    if pattern == "reverse":
        return (source_blocks - 1 - indices) * BLOCK_BYTES
    if pattern == "roll3":
        return ((indices + 3) % source_blocks) * BLOCK_BYTES
    raise ValueError(f"unknown pattern {pattern!r}")


class GatherbTestCase(PTOTestCase):
    """Gather one 32-byte source block per in-bounds UINT32 byte offset.

    PTOAS leaves out-of-bounds TGATHERB offsets undefined, so every active
    offset in this suite names a complete source block.
    """

    __test__ = False

    def __init__(
        self,
        *,
        dtype: DataType = DataType.FP32,
        output_dtype: DataType | None = None,
        pattern: str = "reverse",
        shape: tuple[int, int] = (ROWS, OFFSETS_PER_ROW),
        valid_blocks: tuple[int, int] | None = None,
        platform: str | None = None,
    ):
        super().__init__(platform=platform)
        self._dtype = dtype
        self._output_dtype = output_dtype or dtype
        self._pattern = pattern
        self._shape = shape
        self._valid_blocks = valid_blocks or shape

    def get_name(self) -> str:
        rows, blocks = self._shape
        valid_rows, valid_blocks = self._valid_blocks
        return (
            f"gatherb_{self._dtype.value}_to_{self._output_dtype.value}_"
            f"{self._pattern}_{rows}x{blocks}_v{valid_rows}x{valid_blocks}blocks"
        )

    def define_tensors(self) -> list[TensorSpec]:
        torch_dtype = _TORCH_DTYPE[self._dtype]
        rows, offsets_per_row = self._shape
        source_blocks = rows * offsets_per_row
        source_cols = source_blocks * BLOCK_BYTES // _ELEMENT_BYTES[self._dtype]
        output_cols = offsets_per_row * BLOCK_BYTES // _ELEMENT_BYTES[self._output_dtype]
        source = torch.arange(source_cols).reshape(1, source_cols).to(torch_dtype)
        offsets = (
            _byte_offsets(self._pattern, rows, offsets_per_row).to(torch.int32).reshape(rows, offsets_per_row)
        )
        return [
            TensorSpec("src", [1, source_cols], self._dtype, init_value=source),
            TensorSpec("offset", [rows, offsets_per_row], DataType.UINT32, init_value=offsets),
            TensorSpec(
                "out", [rows, output_cols], self._output_dtype, is_output=True, init_value=torch.zeros
            ),
        ]

    def get_program(self) -> Any:
        dtype = _PL_DTYPE[self._dtype]
        output_dtype = _PL_DTYPE[self._output_dtype]
        rows, offsets_per_row = self._shape
        source_blocks = rows * offsets_per_row
        source_cols = source_blocks * BLOCK_BYTES // _ELEMENT_BYTES[self._dtype]
        output_cols = offsets_per_row * BLOCK_BYTES // _ELEMENT_BYTES[self._output_dtype]
        valid_blocks = list(self._valid_blocks)

        @pl.program
        class GatherbProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[1, source_cols], dtype],
                offset: pl.Tensor[[rows, offsets_per_row], pl.UINT32],
                out: pl.InOut[pl.Tensor[[rows, output_cols], output_dtype]],
            ) -> pl.Tensor[[rows, output_cols], output_dtype]:
                src_tile: pl.Tile[[1, source_cols], dtype] = pl.load(src, [0, 0], [1, source_cols])
                offset_tile: pl.Tile[[rows, offsets_per_row], pl.UINT32] = pl.load(
                    offset, [0, 0], [rows, offsets_per_row], valid_shape=valid_blocks
                )
                result: pl.Tile[[rows, output_cols], output_dtype] = pl.tile.gatherb(
                    src_tile, offset_tile, output_dtype=output_dtype
                )
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[1, source_cols], dtype],
                offset: pl.Tensor[[rows, offsets_per_row], pl.UINT32],
                out: pl.InOut[pl.Tensor[[rows, output_cols], output_dtype]],
            ) -> pl.Tensor[[rows, output_cols], output_dtype]:
                return self.kernel(src, offset, out)

        return GatherbProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        valid_rows, valid_blocks = self._valid_blocks
        rows, _ = self._shape
        expected = torch.zeros_like(tensors["out"])
        source_bytes = tensors["src"].view(torch.uint8).reshape(-1)
        expected_bytes = expected.view(torch.uint8).reshape(rows, -1)
        for row in range(valid_rows):
            for block in range(valid_blocks):
                offset = int(tensors["offset"][row, block].item())
                begin = block * BLOCK_BYTES
                expected_bytes[row, begin : begin + BLOCK_BYTES] = source_bytes[offset : offset + BLOCK_BYTES]
        tensors["out"][:] = expected


class TestGatherb:
    """TGATHERB dtype, block permutation, and valid-shape branches."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "dtype",
        [pytest.param(dt, marks=_skip_one_byte_dst(dt), id=str(dt)) for dt in _PL_DTYPE],
    )
    def test_dtypes(self, test_runner, platform, dtype):
        result = test_runner.run(GatherbTestCase(dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    # Both patterns gather into UINT8, so both are refused by PTOAS >= v0.61.
    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "pattern",
        [pytest.param(p, marks=_skip_one_byte_dst(DataType.UINT8), id=p) for p in ("reverse", "roll3")],
    )
    def test_byte_offset_patterns(self, test_runner, platform, pattern):
        result = test_runner.run(GatherbTestCase(dtype=DataType.UINT8, pattern=pattern, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        ("dtype", "output_dtype"),
        [
            pytest.param(src, dst, marks=_skip_one_byte_dst(dst), id=f"{src}-{dst}")
            for src, dst in (
                (DataType.INT16, DataType.INT32),
                (DataType.UINT8, DataType.UINT16),
                (DataType.INT32, DataType.INT8),
                (DataType.FP32, DataType.UINT8),
                (DataType.UINT32, DataType.FP16),
            )
        ],
    )
    def test_cross_dtype_byte_reinterpretation(self, test_runner, platform, dtype, output_dtype):
        result = test_runner.run(GatherbTestCase(dtype=dtype, output_dtype=output_dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "valid_blocks",
        [(3, OFFSETS_PER_ROW), (ROWS, 5), (3, 5), (1, 1)],
        ids=("row-tail", "col-tail", "row-col-tail", "single-block-boundary"),
    )
    def test_valid_shape(self, test_runner, platform, valid_blocks):
        result = test_runner.run(GatherbTestCase(valid_blocks=valid_blocks, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("shape", [(1, 8), (2, 16)], ids=("single-row", "multi-repeat"))
    def test_physical_shape_boundaries(self, test_runner, platform, shape):
        result = test_runner.run(GatherbTestCase(shape=shape, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
