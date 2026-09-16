# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 runtime: multi-row FP4 GM pitch (copy/view) and FP4→BF16 cast."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime.runner import RunConfig

if not hasattr(torch, "float4_e2m1fn_x2"):
    pytest.skip("torch.float4_e2m1fn_x2 required", allow_module_level=True)

ROWS, LOGICAL_K, PACKED_K = 2, 512, 256
CAST_ROWS, CAST_LOGICAL_K = 4, 64


def _xor_row_fp4() -> torch.Tensor:
    physical = torch.empty((ROWS, PACKED_K), dtype=torch.uint8)
    physical[0] = torch.arange(PACKED_K, dtype=torch.uint8)
    physical[1] = torch.arange(PACKED_K, dtype=torch.uint8).bitwise_xor(0xA5)
    return physical.view(torch.float4_e2m1fn_x2)


def _make_fp4_cast_src(shape: tuple[int, int]) -> torch.Tensor:
    rows, cols = shape
    assert cols % 2 == 0
    generator = torch.Generator().manual_seed(41)
    codes = torch.randint(0, 16, (rows, cols), generator=generator).to(torch.uint8)
    codes[1:] = codes[1:].bitwise_xor(0x05)
    packed = ((codes[:, 1::2] & 0x0F) << 4) | (codes[:, 0::2] & 0x0F)
    return packed.contiguous().view(torch.float4_e2m1fn_x2)


def _decode_fp4_data(data: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    packed = data.contiguous().view(torch.uint8).reshape(rows, cols // 2)
    codes = torch.empty((rows, cols), dtype=torch.long)
    codes[:, 0::2] = (packed & 0x0F).to(torch.long)
    codes[:, 1::2] = ((packed >> 4) & 0x0F).to(torch.long)
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float64,
    )
    return values[codes]


@pl.program
class Fp4TwoRowCopyProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        src: pl.Tensor[[ROWS, LOGICAL_K], pl.FP4],
        out: pl.Out[pl.Tensor[[ROWS, LOGICAL_K], pl.FP4]],
    ) -> pl.Tensor[[ROWS, LOGICAL_K], pl.FP4]:
        return pl.store(pl.load(src, [0, 0], [ROWS, LOGICAL_K]), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src: pl.Tensor[[ROWS, LOGICAL_K], pl.FP4],
        out: pl.Out[pl.Tensor[[ROWS, LOGICAL_K], pl.FP4]],
    ) -> pl.Tensor[[ROWS, LOGICAL_K], pl.FP4]:
        return self.kernel(src, out)


@pl.program
class Fp4TwoRowViewCopyProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        src: pl.Tensor[[ROWS, LOGICAL_K], pl.FP4],
        out: pl.Out[pl.Tensor[[ROWS, LOGICAL_K], pl.FP4]],
    ) -> pl.Tensor[[ROWS, LOGICAL_K], pl.FP4]:
        viewed: pl.Tensor[[ROWS, LOGICAL_K], pl.FP4] = pl.tensor.view(src, [ROWS, LOGICAL_K])
        return pl.store(pl.load(viewed, [0, 0], [ROWS, LOGICAL_K]), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src: pl.Tensor[[ROWS, LOGICAL_K], pl.FP4],
        out: pl.Out[pl.Tensor[[ROWS, LOGICAL_K], pl.FP4]],
    ) -> pl.Tensor[[ROWS, LOGICAL_K], pl.FP4]:
        return self.kernel(src, out)


@pl.program
class Fp4MultiRowCastProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        src: pl.Tensor[[CAST_ROWS, CAST_LOGICAL_K], pl.FP4],
        out: pl.Out[pl.Tensor[[CAST_ROWS, CAST_LOGICAL_K], pl.BF16]],
    ) -> pl.Tensor[[CAST_ROWS, CAST_LOGICAL_K], pl.BF16]:
        return pl.store(pl.cast(pl.load(src, [0, 0], [CAST_ROWS, CAST_LOGICAL_K]), pl.BF16), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src: pl.Tensor[[CAST_ROWS, CAST_LOGICAL_K], pl.FP4],
        out: pl.Out[pl.Tensor[[CAST_ROWS, CAST_LOGICAL_K], pl.BF16]],
    ) -> pl.Tensor[[CAST_ROWS, CAST_LOGICAL_K], pl.BF16]:
        return self.kernel(src, out)


class Fp4TwoRowCopyTest(PTOTestCase):
    __test__ = False

    def __init__(self):
        super().__init__(RunConfig(rtol=0.0, atol=0.0), platform="a5")

    def get_name(self) -> str:
        return "fp4_two_row_copy"

    def define_tensors(self) -> list[TensorSpec]:
        src = _xor_row_fp4()
        return [
            TensorSpec("src", list(src.shape), DataType.FP4, init_value=src),
            TensorSpec("out", [ROWS, PACKED_K], DataType.FP4, is_output=True),
        ]

    def get_program(self) -> Any:
        return Fp4TwoRowCopyProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"].view(torch.uint8).copy_(tensors["src"].view(torch.uint8))


class Fp4TwoRowViewCopyTest(PTOTestCase):
    __test__ = False

    def __init__(self):
        super().__init__(RunConfig(rtol=0.0, atol=0.0), platform="a5")

    def get_name(self) -> str:
        return "fp4_two_row_view_copy"

    def define_tensors(self) -> list[TensorSpec]:
        src = _xor_row_fp4()
        return [
            TensorSpec("src", list(src.shape), DataType.FP4, init_value=src),
            TensorSpec("out", [ROWS, PACKED_K], DataType.FP4, is_output=True),
        ]

    def get_program(self) -> Any:
        return Fp4TwoRowViewCopyProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"].view(torch.uint8).copy_(tensors["src"].view(torch.uint8))


class Fp4MultiRowCastTest(PTOTestCase):
    __test__ = False

    def __init__(self):
        super().__init__(RunConfig(rtol=1e-2, atol=1e-2), platform="a5")

    def get_name(self) -> str:
        return "fp4_multi_row_cast_bf16"

    def define_tensors(self) -> list[TensorSpec]:
        src = _make_fp4_cast_src((CAST_ROWS, CAST_LOGICAL_K))
        return [
            TensorSpec("src", list(src.shape), DataType.FP4, init_value=src),
            TensorSpec("out", [CAST_ROWS, CAST_LOGICAL_K], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        return Fp4MultiRowCastProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        decoded = _decode_fp4_data(tensors["src"], CAST_ROWS, CAST_LOGICAL_K)
        tensors["out"].copy_(decoded.to(torch.bfloat16))


@pytest.mark.platforms("a5")
class TestFp4:
    def test_two_row_fp4_copy(self, test_runner):
        result = test_runner.run(Fp4TwoRowCopyTest())
        assert result.passed, f"Test failed: {result.error}"

    def test_two_row_view_fp4_copy(self, test_runner):
        result = test_runner.run(Fp4TwoRowViewCopyTest())
        assert result.passed, f"Test failed: {result.error}"

    def test_multi_row_fp4_cast_bf16(self, test_runner):
        result = test_runner.run(Fp4MultiRowCastTest())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
