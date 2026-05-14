# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for the unified ``pl.set_validshape`` wrapper on tensors.

``pl.set_validshape`` updates a tensor's ``valid_shape`` metadata without
moving data. To make the metadata observable end-to-end we chain it with
``pl.fillpad``, which writes a sentinel pad value into the now-invalid
region. The expected output therefore has the original data inside the
narrowed valid window and the pad value everywhere else.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

M = 64
N = 64
VALID_ROWS = 48
VALID_COLS = 56


@pl.program
class TensorSetValidShapeZeroProgram:
    """Narrow valid_shape with set_validshape, then fillpad with zero."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        input_tensor: pl.Tensor[[M, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            narrowed = pl.set_validshape(input_tensor, VALID_ROWS, VALID_COLS)
            padded = pl.fillpad(narrowed, pad_value=pl.PadValue.zero)
            output = pl.assemble(output, padded, [0, 0])
        return output


@pl.program
class TensorSetValidShapeMaxProgram:
    """Narrow valid_shape with set_validshape, then fillpad with FP32 max."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        input_tensor: pl.Tensor[[M, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            narrowed = pl.set_validshape(input_tensor, VALID_ROWS, VALID_COLS)
            padded = pl.fillpad(narrowed, pad_value=pl.PadValue.max)
            output = pl.assemble(output, padded, [0, 0])
        return output


@pl.program
class TensorSetValidShapeMinProgram:
    """Narrow valid_shape with set_validshape, then fillpad with FP32 min."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        input_tensor: pl.Tensor[[M, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            narrowed = pl.set_validshape(input_tensor, VALID_ROWS, VALID_COLS)
            padded = pl.fillpad(narrowed, pad_value=pl.PadValue.min)
            output = pl.assemble(output, padded, [0, 0])
        return output


def _expected(input_tensor: torch.Tensor, fill: float) -> torch.Tensor:
    expected = torch.full((M, N), fill, dtype=torch.float32)
    expected[:VALID_ROWS, :VALID_COLS] = input_tensor[:VALID_ROWS, :VALID_COLS]
    return expected


class TensorSetValidShapeZeroTestCase(PTOTestCase):
    """set_validshape + fillpad(zero): invalid region must be 0.0."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "tensor_set_validshape_zero"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [M, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TensorSetValidShapeZeroProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["output"][:] = _expected(tensors["input_tensor"], 0.0)


class TensorSetValidShapeMaxTestCase(PTOTestCase):
    """set_validshape + fillpad(max): invalid region must be +inf."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "tensor_set_validshape_max"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [M, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TensorSetValidShapeMaxProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["output"][:] = _expected(tensors["input_tensor"], float("inf"))


class TensorSetValidShapeMinTestCase(PTOTestCase):
    """set_validshape + fillpad(min): invalid region must be -inf."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "tensor_set_validshape_min"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_tensor", [M, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TensorSetValidShapeMinProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["output"][:] = _expected(tensors["input_tensor"], float("-inf"))


class TestTensorSetValidShape:
    """Test tensor-level set_validshape via the unified pl.set_validshape wrapper."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tensor_set_validshape_zero(self, test_runner, platform):
        result = test_runner.run(TensorSetValidShapeZeroTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tensor_set_validshape_max(self, test_runner, platform):
        result = test_runner.run(TensorSetValidShapeMaxTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_tensor_set_validshape_min(self, test_runner, platform):
        result = test_runner.run(TensorSetValidShapeMinTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
