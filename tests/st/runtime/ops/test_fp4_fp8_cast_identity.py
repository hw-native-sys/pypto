# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 identity test for the FP4 -> FP8E4M3FN single-pass cast.

LegalizeTileCast lowers the cast to in-kernel bit ops on the packed carrier
(reinterpret U8 -> U16 zero-extend -> nibble arithmetic -> reinterpret FP8),
which must be numerically identical to decoding every E2M1 code and
re-encoding it in E4M3 -- including the +-0 and the subnormal 0.5 codes. The
input sweeps every packed byte value 0..255 so both nibble positions of every
code are covered. The FP8 result is widened through the native FP8->FP32 cast
for the harness comparison (exact: E4M3 embeds in FP32).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

_REQUIRED_TORCH_DTYPES = ("float4_e2m1fn_x2",)
if not all(hasattr(torch, name) for name in _REQUIRED_TORCH_DTYPES):
    pytest.skip("torch FP4 dtype required", allow_module_level=True)

ROWS, LOGICAL_COLS = 32, 512  # carriers: [ROWS, 256] -- every byte value 0..255 x ROWS


@pl.jit.incore
def _fp4_to_fp8_kernel(
    x: pl.Tensor[[ROWS, LOGICAL_COLS], pl.FP4],
    out: pl.Out[pl.Tensor[[ROWS, LOGICAL_COLS], pl.FP32]],
) -> pl.Tensor[[ROWS, LOGICAL_COLS], pl.FP32]:
    t = pl.load(x, [0, 0], [ROWS, LOGICAL_COLS])
    return pl.store(pl.cast(pl.cast(t, pl.FP8E4M3FN), pl.FP32), [0, 0], out)


@pl.jit
def fp4_to_fp8(
    x: pl.Tensor[[ROWS, LOGICAL_COLS], pl.FP4],
    out: pl.Out[pl.Tensor[[ROWS, LOGICAL_COLS], pl.FP32]],
) -> pl.Tensor[[ROWS, LOGICAL_COLS], pl.FP32]:
    return _fp4_to_fp8_kernel(x, out)


class Fp4ToFp8IdentityCase(PTOTestCase):
    """Every packed byte 0..255 through the single-pass cast, exact golden."""

    def get_name(self) -> str:
        return "fp4_to_fp8_bitop_identity"

    def define_tensors(self) -> list[TensorSpec]:
        carrier_cols = LOGICAL_COLS // 2
        byte_values = torch.arange(256, dtype=torch.uint8)
        packed = byte_values.repeat(ROWS).reshape(ROWS, carrier_cols)
        x = packed.contiguous().view(torch.float4_e2m1fn_x2)
        return [
            TensorSpec("x", list(x.shape), DataType.FP4, init_value=x),
            TensorSpec("out", [ROWS, LOGICAL_COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return fp4_to_fp8.specialize()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        packed = tensors["x"].contiguous().view(torch.uint8)
        codes = torch.empty((ROWS, LOGICAL_COLS), dtype=torch.long)
        codes[:, 0::2] = (packed & 0x0F).to(torch.long)
        codes[:, 1::2] = ((packed >> 4) & 0x0F).to(torch.long)
        values = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32,
        )
        # Exact by construction: the E2M1 values are representable in E4M3 and
        # the FP8->FP32 widening is exact, so any bit difference in the
        # intermediate FP8 result surfaces as a float32 inequality here.
        tensors["out"][:] = values[codes]


@pytest.mark.platforms("a5")
class TestFp4ToFp8Identity:
    def test_bitop_cast_identity_all_codes(self, test_runner):
        case = Fp4ToFp8IdentityCase()
        result = test_runner.run(case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
