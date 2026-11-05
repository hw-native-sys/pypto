# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 round trip of FP8E5M2 data through the runtime's UINT8 byte carrier.

The pinned runtime has no E5M2 ``DataType``; PyPTO passes ``torch.float8_e5m2``
host tensors as zero-copy UINT8 views while the kernel keeps its FP8E5M2 type.
A load/store copy from a non-zero offset must return every finite E5M2 code
(normals, subnormals, signed zeros, and the 57344 maximum) unchanged.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

if not hasattr(torch, "float8_e5m2"):
    pytest.skip("torch.float8_e5m2 required", allow_module_level=True)

ROWS, COLS = 32, 256
SRC_ROWS = 2 * ROWS
ROW_OFFSET = ROWS


def _finite_e5m2_codes() -> torch.Tensor:
    """All E5M2 byte codes except Inf (x7C) and NaN (x7D-x7F), for both signs."""
    codes = torch.arange(256, dtype=torch.int32)
    magnitude = codes & 0x7F
    return codes[magnitude < 0x7C].to(torch.uint8)


@pl.jit.incore
def e5m2_copy_kernel(
    src: pl.Tensor[[SRC_ROWS, COLS], pl.FP8E5M2],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP8E5M2]],
) -> pl.Tensor[[ROWS, COLS], pl.FP8E5M2]:
    tile = pl.load(src, [ROW_OFFSET, 0], [ROWS, COLS])
    out = pl.store(tile, [0, 0], out)
    return out


@pl.jit
def e5m2_copy(
    src: pl.Tensor[[SRC_ROWS, COLS], pl.FP8E5M2],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP8E5M2]],
) -> pl.Tensor[[ROWS, COLS], pl.FP8E5M2]:
    return e5m2_copy_kernel(src, out)


class Fp8E5M2CopyTestCase(PTOTestCase):
    """Exact FP8E5M2 GM -> Vec -> GM copy through the byte-carrier runtime path."""

    __test__ = False

    def __init__(self):
        super().__init__(RunConfig(rtol=0.0, atol=0.0), platform="a5")

    def get_name(self) -> str:
        return "fp8e5m2_offset_copy"

    def define_tensors(self) -> list[TensorSpec]:
        finite = _finite_e5m2_codes()
        generator = torch.Generator().manual_seed(5)
        picks = torch.randint(0, finite.numel(), (SRC_ROWS, COLS), generator=generator)
        codes = finite[picks]
        # Every finite code, including subnormals and both zeros, lands in the copied window.
        codes[ROW_OFFSET].copy_(torch.cat([finite, finite[: COLS - finite.numel()]]))
        src = codes.view(torch.float8_e5m2)
        return [
            TensorSpec("src", [SRC_ROWS, COLS], DataType.FP8E5M2, init_value=src),
            TensorSpec("out", [ROWS, COLS], DataType.FP8E5M2, is_output=True),
        ]

    def get_program(self) -> Any:
        return e5m2_copy.specialize()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["src"][ROW_OFFSET : ROW_OFFSET + ROWS]


@pytest.mark.platforms("a5")
class TestFp8E5M2Transfer:
    """Runtime transfer coverage for the FP8E5M2 byte carrier."""

    def test_e5m2_offset_copy_is_exact(self, test_runner):
        result = test_runner.run(Fp8E5M2CopyTestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
